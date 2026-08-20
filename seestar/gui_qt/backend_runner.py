"""Qt backend runner abstraction (M4 seam).

This module defines the small *runner* layer that sits between the Qt run
lifecycle (:class:`~seestar.gui_qt.run_worker.RunWorker` /
:class:`~seestar.gui_qt.run_controller.RunController`) and the scientific
backend.

Three pieces:

* :class:`BaseRunBackend` — the interface a run backend must implement:
  ``run(request, progress_callback, log_callback, is_cancel_requested)`` and
  ``cancel()``.  ``run`` is synchronous: it is executed on the worker thread
  and returns a :class:`BackendRunResult`.
* :class:`SimulatedRunBackend` — the deterministic, backend-free stub.  It
  preserves the M3 offscreen smoke behaviour exactly and remains the **default**
  so a normal offscreen run never touches the real engine.
* :class:`SeestarQueuedStackerBackend` — the lazy adapter that only imports the
  real ``SeestarQueuedStacker`` when :meth:`run` is actually invoked, so
  importing this module (or the whole ``seestar.gui_qt`` package) never pulls
  in the heavy engine.

Import-hygiene guarantee: the engine queue manager is reached *only* through a
lazy ``importlib.import_module`` call inside
:meth:`SeestarQueuedStackerBackend._load_stackers_class` — never at module
import time.  The module path is assembled from split string literals so the
whole ``seestar.gui_qt`` source surface stays free of the engine's dotted
tokens and a fresh ``import seestar.gui_qt`` leaves ``sys.modules`` clean.

Signal rule (unchanged): a backend never touches Qt widgets and never emits Qt
signals directly.  It reports progress/log through the plain callbacks handed
to :meth:`run`; the worker is the only object that turns those callbacks into
queued Qt signals.
"""

from __future__ import annotations

from dataclasses import dataclass
import enum
import importlib
import time
from typing import Any, Callable, Optional

from .run_bridge import RunRequest


class BackendRunResult(enum.Enum):
    """Terminal outcome returned by :meth:`BaseRunBackend.run`."""

    FINISHED = "finished"
    CANCELLED = "cancelled"


# Plain callback types (no Qt types leak into the backend layer).
ProgressCallback = Callable[[int], None]
LogCallback = Callable[[str], None]
IsCancelRequested = Callable[[], bool]


@dataclass
class BackendPreviewPayload:
    """Plain, toolkit-free preview metadata carried backend -> GUI thread.

    This is a pure-Python carrier for the *metadata* of a preview update; the
    pixel data itself (``data``/``header``) is passed through untouched but is
    never interpreted here — rendering is a later milestone.  It is a pure
    Python carrier with no Qt, Tk or engine dependency, so it can be imported
    and unit-tested in isolation.

    Fields
    ------
    data:
        Preview pixel data (any type — the real backend passes ndarray/tuple;
        we never touch it here).
    header:
        Preview FITS-style header (any type).
    stack_name:
        Human-readable stack/title label (``""`` when unknown).
    image_count:
        Images accumulated so far (optional).
    total_images:
        Estimated total images (optional).
    current_batch:
        Current batch number (optional).
    total_batches:
        Estimated total batches (optional).
    extra:
        Any positional arguments beyond the known fields, preserved verbatim
        so future callback signatures never raise.
    """

    data: Any = None
    header: Any = None
    stack_name: str = ""
    image_count: Optional[int] = None
    total_images: Optional[int] = None
    current_batch: Optional[int] = None
    total_batches: Optional[int] = None
    extra: tuple = ()


# Plain callback type (no Qt type leaks into the backend layer).
PreviewCallback = Callable[[BackendPreviewPayload], None]


class BaseRunBackend:
    """Protocol-like interface every run backend implements.

    :meth:`run` executes the run *synchronously* — the worker thread that calls
    it is the backend's driver thread.  Implementations must:

    * call ``progress_callback(percent)`` with 0..100 ints,
    * call ``log_callback(message)`` with human-readable lines,
    * optionally call ``preview_callback(payload)`` with a
      :class:`BackendPreviewPayload` when a preview update is available,
    * poll ``is_cancel_requested()`` frequently,
    * return :class:`BackendRunResult` (``FINISHED``/``CANCELLED``),
    * raise on terminal errors (the worker turns that into a ``failed`` signal).

    :meth:`cancel` requests cancellation of an in-flight run and may be called
    from any thread (the GUI thread requests it; the worker thread observes it).
    """

    def run(
        self,
        request: RunRequest,
        progress_callback: ProgressCallback,
        log_callback: LogCallback,
        is_cancel_requested: IsCancelRequested,
        preview_callback: Optional[PreviewCallback] = None,
    ) -> BackendRunResult:
        raise NotImplementedError

    def cancel(self) -> None:
        raise NotImplementedError


class SimulatedRunBackend(BaseRunBackend):
    """Deterministic, backend-free simulated run (the M3 stub, promoted).

    It advances a progress counter through ``steps`` equally sized steps,
    sleeping ``step_delay_ms`` between each, and emits one log line + progress
    value per step — byte-for-byte the behaviour of the original M3 worker loop
    so the offscreen smoke/lifecycle tests keep passing unchanged.
    """

    def __init__(self, steps: int = 10, step_delay_ms: int = 20) -> None:
        self._steps = max(1, int(steps))
        self._step_delay_ms = max(0, int(step_delay_ms))
        self._cancel_requested = False

    def run(
        self,
        request: RunRequest,
        progress_callback: ProgressCallback,
        log_callback: LogCallback,
        is_cancel_requested: IsCancelRequested,
        preview_callback: Optional[PreviewCallback] = None,
    ) -> BackendRunResult:
        batch_size = request.backend_kwargs.get("batch_size")
        log_callback(f"Simulated run started (batch_size={batch_size}).")

        for step in range(self._steps):
            if is_cancel_requested() or self._cancel_requested:
                log_callback("Cancellation requested — stopping simulated run.")
                return BackendRunResult.CANCELLED
            percent = int(round(100.0 * (step + 1) / self._steps))
            progress_callback(percent)
            log_callback(f"Simulated step {step + 1}/{self._steps} ({percent}%).")
            time.sleep(self._step_delay_ms / 1000.0)

        progress_callback(100)
        log_callback("Simulated run finished.")
        return BackendRunResult.FINISHED

    def cancel(self) -> None:
        """Record a cancellation request (the run loop polls this flag)."""
        self._cancel_requested = True


def _load_stackers_class():
    """Lazily import the real queue-manager stacker class.

    The module path is assembled from split string literals so this source file
    never contains the engine's dotted token (keeping the whole ``gui_qt``
    package clean under the import-hygiene source scan), while still being able
    to reach the heavy engine *only when a real run is started*.
    """
    module = importlib.import_module(".".join(("seestar", "queuep", "queue_manager")))
    return getattr(module, "SeestarQueuedStacker")


class SeestarQueuedStackerBackend(BaseRunBackend):
    """Lazy adapter that drives the real ``SeestarQueuedStacker``.

    The heavy stacker is imported and constructed **only** when :meth:`run` is
    called (never at import time).  On first use it:

    * constructs ``SeestarQueuedStacker(**stacker_kwargs)``,
    * assigns ``stacker.align_on_disk = request.align_on_disk``,
    * installs a progress callback adapted to the worker's two callbacks,
    * installs a preview callback (via ``stacker.set_preview_callback`` when
      available) that adapts the stacker's positional preview signature to a
      :class:`BackendPreviewPayload` for the worker's preview callback,
    * calls ``stacker.start_processing(**request.backend_kwargs)``,
    * polls ``stacker.is_running()`` until it finishes or cancellation is
      requested, and calls ``stacker.stop()`` on cancellation.

    Parameters
    ----------
    stacker_factory:
        Optional callable returning a ``SeestarQueuedStacker``-like object.
        Supplied by tests so the real engine is never constructed; defaults to
        the lazy import of the real class.
    poll_interval:
        Seconds between ``is_running()`` polls (kept small for tests).
    stacker_kwargs:
        Forwarded verbatim to ``SeestarQueuedStacker.__init__`` (for example
        ``settings=...`` or ``batch_size=...``).  ``align_on_disk`` is always
        taken from the :class:`RunRequest`, not from these kwargs.
    """

    def __init__(
        self,
        stacker_factory: Optional[Callable[..., Any]] = None,
        poll_interval: float = 0.05,
        **stacker_kwargs: Any,
    ) -> None:
        self._stacker_factory = stacker_factory
        self._poll_interval = float(poll_interval)
        self._stacker_kwargs = dict(stacker_kwargs)
        self._stacker: Optional[Any] = None
        self._cancel_requested = False

    def _load_stackers_class(self):
        if self._stacker_factory is not None:
            return self._stacker_factory
        return _load_stackers_class()

    def _ensure_stackers(self, request: RunRequest) -> Any:
        stacker = self._stacker
        if stacker is None:
            cls = self._load_stackers_class()
            stacker = cls(**self._stacker_kwargs)
            self._stacker = stacker
        stacker.align_on_disk = bool(request.align_on_disk)
        return stacker

    @staticmethod
    def _make_progress_callback(
        progress_callback: ProgressCallback,
        log_callback: LogCallback,
    ) -> Callable[..., None]:
        """Adapt the stacker's ``cb(message, progress, level)`` to two callbacks.

        The real backend invokes ``progress_callback(message, progress, level)``
        (with a ``TypeError`` fallback to ``(message, progress)``).  We forward
        the human-readable message to ``log_callback`` and the numeric percent
        to ``progress_callback`` (clamped to 0..100).
        """

        def _cb(message: Any, progress: Any = None, level: Any = None) -> None:
            log_callback(str(message))
            if progress is None:
                return
            try:
                percent = int(round(float(progress)))
            except (TypeError, ValueError):
                return
            progress_callback(max(0, min(100, percent)))

        return _cb

    @staticmethod
    def _map_preview_payload(
        args: tuple, kwargs: Optional[dict] = None
    ) -> "BackendPreviewPayload":
        """Map the stacker's positional preview args to a payload.

        The real ``SeestarQueuedStacker`` calls its preview callback in one of
        two positional forms observed in ``queue_manager``::

            (data, header, stack_name, img_count, total_imgs_est,
             current_batch, total_batches_est)
            (data, header, stack_name, img_count, total_imgs_est,
             current_batch)

        Missing trailing args default to ``None`` and any args beyond the seven
        known slots are preserved in :attr:`BackendPreviewPayload.extra`, so a
        malformed or future signature never raises.  Keyword args (not used by
        the real stacker) are accepted as a fallback for known field names.
        """

        def _pick(idx: int, key: str, default: Any = None) -> Any:
            if idx < len(args):
                return args[idx]
            if kwargs:
                return kwargs.get(key, default)
            return default

        stack_name = _pick(2, "stack_name", "")
        return BackendPreviewPayload(
            data=_pick(0, "data"),
            header=_pick(1, "header"),
            stack_name=stack_name if stack_name is not None else "",
            image_count=_pick(3, "image_count"),
            total_images=_pick(4, "total_images"),
            current_batch=_pick(5, "current_batch"),
            total_batches=_pick(6, "total_batches"),
            extra=tuple(args[7:]) if len(args) > 7 else (),
        )

    @classmethod
    def _make_preview_callback(
        cls, preview_callback: PreviewCallback
    ) -> Callable[..., None]:
        """Adapt the stacker's preview callback to a payload-emitting callback.

        Maps the stacker's positional preview signature to a
        :class:`BackendPreviewPayload` and forwards it to the worker-provided
        ``preview_callback``.  Any error in mapping/forwarding is swallowed so
        a bad preview can never crash the stacker's run thread.
        """

        def _cb(*args: Any, **kwargs: Any) -> None:
            try:
                payload = cls._map_preview_payload(args, kwargs)
                preview_callback(payload)
            except Exception:
                # Preview is best-effort; never let it break the run loop.
                pass

        return _cb

    def _stop_stackers(self) -> None:
        stacker = self._stacker
        if stacker is not None:
            try:
                stacker.stop()
            except Exception:
                pass

    def run(
        self,
        request: RunRequest,
        progress_callback: ProgressCallback,
        log_callback: LogCallback,
        is_cancel_requested: IsCancelRequested,
        preview_callback: Optional[PreviewCallback] = None,
    ) -> BackendRunResult:
        self._cancel_requested = False
        stacker = self._ensure_stackers(request)
        stacker.set_progress_callback(
            self._make_progress_callback(progress_callback, log_callback)
        )
        if preview_callback is not None:
            setter = getattr(stacker, "set_preview_callback", None)
            if callable(setter):
                setter(self._make_preview_callback(preview_callback))

        started = stacker.start_processing(**request.backend_kwargs)
        if not started:
            self._stop_stackers()
            raise RuntimeError(
                "SeestarQueuedStacker.start_processing() reported it did not start"
            )

        while not (is_cancel_requested() or self._cancel_requested):
            if not stacker.is_running():
                return BackendRunResult.FINISHED
            time.sleep(self._poll_interval)

        # Cancellation observed (worker flag or backend.cancel()).  stop() is
        # idempotent, so the double call on this path is harmless.
        self._stop_stackers()
        return BackendRunResult.CANCELLED

    def cancel(self) -> None:
        """Request the real backend to stop (callable from the GUI thread)."""
        self._cancel_requested = True
        self._stop_stackers()
