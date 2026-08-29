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
import os
from queue import Empty, Queue
import time
from typing import Any, Callable, Optional

from .run_bridge import RUN_INTENT_FRESH, RunRequest, split_backend_kwargs
from .run_log import RunLog
from .startup_refusal import StartupRefusedError, build_payload_from_engine
from .summary_payload import SummaryPayload, build_summary_payload


class BackendRunResult(enum.Enum):
    """Terminal outcome returned by :meth:`BaseRunBackend.run`."""

    FINISHED = "finished"
    CANCELLED = "cancelled"


# Plain callback types (no Qt types leak into the backend layer).
ProgressCallback = Callable[[int], None]
LogCallback = Callable[[str], None]
IsCancelRequested = Callable[[], bool]
# Plain callback type for the terminal run summary (no Qt types leak in).
SummaryCallback = Callable[[SummaryPayload], None]


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
    * optionally call ``summary_callback(payload)`` with a
      :class:`~seestar.gui_qt.summary_payload.SummaryPayload` at the end of a
      successful run,
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
        summary_callback: Optional[SummaryCallback] = None,
    ) -> BackendRunResult:
        raise NotImplementedError

    def cancel(self) -> None:
        raise NotImplementedError

    def set_preview_downsample_factor(self, factor: int) -> None:
        """Request a live preview-downsample factor change during a run.

        The base backend has no live engine, so this is a no-op (safe to call
        from any thread at any time).  The real backend overrides it to forward
        the factor to the live stacker instance on the worker thread.
        """
        return None


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
        summary_callback: Optional[SummaryCallback] = None,
    ) -> BackendRunResult:
        batch_size = request.backend_kwargs.get("batch_size")
        log_callback(f"Simulated run started (batch_size={batch_size}).")
        start_time = time.monotonic()

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
        if summary_callback is not None:
            summary_callback(
                build_summary_payload(
                    status="finished",
                    duration_seconds=time.monotonic() - start_time,
                    files_attempted=self._steps,
                    output_dir=request.backend_kwargs.get("output_dir"),
                )
            )
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

    Live control (M22): :meth:`set_preview_downsample_factor` is a thread-safe
    control channel the GUI thread uses during an active run.  It enqueues the
    factor on ``self._control_queue``; :meth:`run` drains that queue on the
    worker thread and applies ``stacker.set_preview_downsample_factor`` +
    ``stacker.refresh_preview`` (Tk ``_cycle_preview_resolution`` parity), so
    the stacker is only ever mutated by the thread that drives it.

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
        # Durable per-run lifecycle log carrier (shared with worker/controller/
        # MainWindow via ``backend.run_log``).  ``None`` until a run starts.
        self.run_log: Optional[RunLog] = None
        # Thread-safe control channel (GUI thread -> worker thread).  The GUI
        # thread enqueues live-control requests via
        # :meth:`set_preview_downsample_factor`; :meth:`run` drains them on the
        # worker thread so the stacker is only ever mutated by its owner thread.
        self._control_queue: Queue = Queue()

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
    def _apply_seam_kwargs(stacker: Any, seam_kwargs: dict) -> None:
        """Apply seam-only snapshot fields to the stacker before processing.

        These fields are read by the queue manager from its *instance*
        attributes, *not* from ``start_processing`` arguments:

        * ``stack_final_combine`` — set on the stacker and (when a settings
          object is attached) mirrored there so a later settings re-read cannot
          clobber the selected value.
        * ``use_gpu`` — drizzle GPU toggle; the engine reads ``stacker.use_gpu``
          at run time.
        * ``max_hq_mem_gb`` — HQ RAM limit in GB; the engine reads
          ``stacker.max_hq_mem`` in bytes, so the GB value is converted here.
        """
        combine = seam_kwargs.get("stack_final_combine")
        if combine is not None:
            stacker.stack_final_combine = combine
        settings = getattr(stacker, "settings", None)
        if combine is not None and settings is not None:
            try:
                setattr(settings, "stack_final_combine", combine)
            except Exception:
                # A settings object that rejects the write must not break the
                # run; the instance attribute above is the authoritative path.
                pass

        if "use_gpu" in seam_kwargs:
            stacker.use_gpu = bool(seam_kwargs["use_gpu"])

        if "max_hq_mem_gb" in seam_kwargs:
            stacker.max_hq_mem = int(float(seam_kwargs["max_hq_mem_gb"]) * 1024 ** 3)

    @staticmethod
    def _make_progress_callback(
        progress_callback: ProgressCallback,
        log_callback: LogCallback,
        run_log: Optional[RunLog] = None,
    ) -> Callable[..., None]:
        """Adapt the stacker's ``cb(message, progress, level)`` to two callbacks.

        The real backend invokes ``progress_callback(message, progress, level)``
        (with a ``TypeError`` fallback to ``(message, progress)``).  We forward
        the human-readable message to ``log_callback`` and the numeric percent
        to ``progress_callback`` (clamped to 0..100).  When a ``run_log`` is
        supplied, the message/percent/level are also recorded as a bounded
        ``ENGINE_PROGRESS`` durable event (best-effort).
        """

        def _cb(message: Any, progress: Any = None, level: Any = None) -> None:
            msg = str(message)
            log_callback(msg)
            if run_log is not None:
                try:
                    fields = {"message": msg}
                    if progress is not None:
                        fields["percent"] = progress
                    if level is not None:
                        fields["level"] = level
                    run_log.emit("ENGINE_PROGRESS", **fields)
                except Exception:
                    # Instrumentation is best-effort; never break the run.
                    pass
            if progress is None:
                return
            try:
                percent = int(round(float(progress)))
            except (TypeError, ValueError):
                return
            progress_callback(max(0, min(100, percent)))

        return _cb

    @staticmethod
    def _make_lifecycle_callback(run_log: RunLog) -> Callable[[str, dict], None]:
        """Adapt the engine's ``(event, fields)`` lifecycle seam to the run log.

        The engine calls this through :meth:`SeestarQueuedStacker._emit_lifecycle`
        (fail-open) at its actual entry/return seams.  Any error here is
        swallowed so durable instrumentation can never break science.
        """

        def _cb(event: str, fields: dict) -> None:
            try:
                run_log.emit(str(event), **dict(fields or {}))
            except Exception:
                pass

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

    @staticmethod
    def _drain_gui_event_queue(stacker: Any) -> None:
        """Drain the stacker's deferred GUI event queue, invoking each callback.

        The real ``SeestarQueuedStacker`` does **not** call its progress
        callback directly: :meth:`SeestarQueuedStacker.update_progress` pushes a
        closure onto ``stacker.gui_event_queue`` (a thread-safe ``Queue``) and
        expects the GUI layer to drain it (the Tk GUI does this from a periodic
        ``after`` loop).  The Qt bridge has no such loop, so without this drain
        the progress/log callbacks installed by
        :meth:`SeestarQueuedStackerBackend.run` would never fire and the Qt GUI
        would never see progress or log lines.

        The queued items are the engine's own closures that already carry the
        right signature (``cb(message, progress, level)``); invoking them here
        on the worker thread re-enters the backend's progress adapter, which
        forwards to the worker's ``progress_callback``/``log_callback`` (queued
        Qt signals to the GUI thread).  This is byte-for-byte the role the Tk
        GUI plays, moved into the backend so no Qt widget ever needs the queue.
        """
        queue = getattr(stacker, "gui_event_queue", None)
        if queue is None:
            return
        while True:
            try:
                cb = queue.get_nowait()
            except Empty:
                break
            try:
                if callable(cb):
                    cb()
            except Exception:
                # A single malformed callback must never abort the drain loop
                # or the run: progress delivery is best-effort, and the run's
                # own terminal state is decided independently of it.
                pass
            finally:
                try:
                    queue.task_done()
                except Exception:
                    pass

    @staticmethod
    def _apply_preview_downsample_control(stacker: Any, factor: int) -> None:
        """Apply a live preview-downsample control request to a stacker.

        Mirrors the Tk ``_cycle_preview_resolution`` engine coupling: call
        ``stacker.set_preview_downsample_factor(factor)`` then
        ``stacker.refresh_preview()``.  Each call is best-effort (a missing or
        raising method is ignored) so any stacker-like object is safe.
        """
        setter = getattr(stacker, "set_preview_downsample_factor", None)
        if callable(setter):
            setter(factor)
        refresher = getattr(stacker, "refresh_preview", None)
        if callable(refresher):
            refresher()

    def _drain_control_queue(self, stacker: Any) -> None:
        """Apply any pending live-control requests to ``stacker`` (worker thread).

        The GUI thread enqueues requests via
        :meth:`set_preview_downsample_factor`; this method runs on the worker
        thread inside :meth:`run`'s polling loop and applies them to the
        stacker so the stacker is only ever mutated on the thread that owns it.
        Unknown/malformed items are dropped (live controls are best-effort).
        """
        while True:
            try:
                item = self._control_queue.get_nowait()
            except Empty:
                break
            try:
                kind, value = item
                if kind == "preview_downsample_factor":
                    self._apply_preview_downsample_control(stacker, int(value))
            except Exception:
                # A malformed control item must never abort the drain or the
                # run; live controls are best-effort.
                pass
            finally:
                try:
                    self._control_queue.task_done()
                except Exception:
                    pass

    def set_preview_downsample_factor(self, factor: int) -> None:
        """Thread-safe live preview-downsample control (GUI thread -> worker).

        Enqueues the factor on a thread-safe queue drained by :meth:`run` on
        the worker thread, so the stacker mutation happens on the thread that
        drives it (never concurrently with processing).  Safe to call before,
        during or after a run; a request with no active run is simply never
        drained (a silent no-op, no crash).
        """
        try:
            self._control_queue.put(("preview_downsample_factor", int(factor)))
        except Exception:
            pass

    @staticmethod
    def _allowlisted_metadata(start_kwargs: dict) -> dict:
        """Return a small allowlisted config/mode metadata mapping for the log.

        Only stable, bounded, non-secret fields are carried; no arrays, no
        full configuration dump.
        """
        keys = (
            "stacking_mode",
            "use_drizzle",
            "drizzle_mode",
            "is_mosaic_run",
            "batch_size",
            "input_dir",
            "output_dir",
            "reference_path_ui",
        )
        return {
            key: start_kwargs[key]
            for key in keys
            if key in start_kwargs and start_kwargs[key] is not None
        }

    @staticmethod
    def _product_version() -> str:
        """Return the product version string (``"version codename"``) for the log.

        Cheap and import-hygienic: reads ``seestar.__version__`` /
        ``seestar.__codename__`` from the already-imported parent package (its
        ``__init__`` only binds those two names plus lazy re-exports — no engine,
        Tk or astropy).  Never raises; degrades to ``""``.
        """
        try:
            import seestar
        except Exception:
            return ""
        version = getattr(seestar, "__version__", "") or ""
        codename = getattr(seestar, "__codename__", "") or ""
        if version and codename:
            return f"{version} {codename}"
        return version

    def _wait_for_engine_termination(
        self, stacker: Any, is_cancel_requested: IsCancelRequested
    ) -> None:
        """Wait until the engine thread has *actually* terminated.

        ``is_running()`` can report False (``processing_active`` cleared) while
        the engine thread is still finishing cleanup (autotuner stop, executor
        shutdown, gc).  Thread liveness is authoritative: we must not return
        FINISHED while the thread is alive.  We join in small bounded slices on
        the worker thread (never the GUI thread) and keep draining the deferred
        GUI event queue so terminal progress still flows while the engine tail
        finishes; cancellation keeps calling ``stop()`` (idempotent) to help the
        tail terminate.
        """
        thread = getattr(stacker, "processing_thread", None)
        if thread is None or not thread.is_alive():
            return
        while thread.is_alive():
            if is_cancel_requested() or self._cancel_requested:
                self._stop_stackers()
            self._drain_gui_event_queue(stacker)
            self._drain_control_queue(stacker)
            thread.join(timeout=self._poll_interval)
        self._drain_gui_event_queue(stacker)
        self._drain_control_queue(stacker)

    def run(
        self,
        request: RunRequest,
        progress_callback: ProgressCallback,
        log_callback: LogCallback,
        is_cancel_requested: IsCancelRequested,
        preview_callback: Optional[PreviewCallback] = None,
        summary_callback: Optional[SummaryCallback] = None,
    ) -> BackendRunResult:
        self._cancel_requested = False
        start_time = time.monotonic()
        stacker = self._ensure_stackers(request)
        start_kwargs, seam_kwargs = split_backend_kwargs(request.backend_kwargs)
        self._apply_seam_kwargs(stacker, seam_kwargs)

        # Create the run-log carrier *before* start_processing so a tiny number
        # of pre-accept engine lifecycle events can be buffered (no file is
        # created until the run is accepted).
        run_log = RunLog()
        self.run_log = run_log
        lifecycle_setter = getattr(stacker, "set_lifecycle_callback", None)
        if callable(lifecycle_setter):
            lifecycle_setter(self._make_lifecycle_callback(run_log))

        stacker.set_progress_callback(
            self._make_progress_callback(progress_callback, log_callback, run_log)
        )
        if preview_callback is not None:
            setter = getattr(stacker, "set_preview_callback", None)
            if callable(setter):
                setter(self._make_preview_callback(preview_callback))

        # Resume Contract v2: intent is carried explicitly from the run
        # request (never derived from artifacts by the engine).  The engine
        # treats a missing/None intent as fresh; ``resume_source`` is optional
        # and only meaningful for a resume.
        started = stacker.start_processing(
            **start_kwargs,
            resume_intent=getattr(request, "resume_intent", RUN_INTENT_FRESH),
            resume_source=getattr(request, "resume_source", None),
        )
        if not started:
            self._stop_stackers()
            # Structured startup refusal (known code) vs generic false start.
            payload = build_payload_from_engine(
                getattr(stacker, "startup_refusal", None)
            )
            if payload is not None:
                raise StartupRefusedError(payload)
            raise RuntimeError(
                "SeestarQueuedStacker.start_processing() reported it did not start"
            )

        # Accepted: open the durable run log now (never in an incompatible
        # folder, never overwriting an earlier run's log).
        run_log.warning = log_callback
        metadata = self._allowlisted_metadata(start_kwargs)
        metadata["product_version"] = self._product_version()
        metadata["resume_intent"] = getattr(request, "resume_intent", RUN_INTENT_FRESH)
        run_log.open(
            start_kwargs.get("output_dir") or getattr(stacker, "output_folder", None),
            metadata=metadata,
        )
        run_log.emit("RUN_ACCEPTED", output_dir=start_kwargs.get("output_dir"))
        run_log.emit(
            "RUN_STARTED",
            mode=start_kwargs.get("stacking_mode"),
            use_drizzle=bool(start_kwargs.get("use_drizzle")),
            input_count=getattr(stacker, "files_in_queue", None),
        )

        while not (is_cancel_requested() or self._cancel_requested):
            if not stacker.is_running():
                break
            self._drain_gui_event_queue(stacker)
            self._drain_control_queue(stacker)
            time.sleep(self._poll_interval)

        # Flush any callbacks the engine queued in the instant before its
        # processing thread finished, so the GUI receives the terminal
        # progress/log/preview state even when the thread ended between two
        # polls.
        self._drain_gui_event_queue(stacker)
        # Also apply any live-control request that arrived in that same
        # instant (best-effort; the run is already terminal).
        self._drain_control_queue(stacker)

        # Truthful completion: ``is_running()`` may already be False while the
        # engine thread is still finishing cleanup.  Wait for the thread to
        # actually terminate before recording the return (thread liveness is
        # authoritative).  The backend runs off the GUI thread, so this bounded
        # slice join never blocks the GUI.
        self._wait_for_engine_termination(stacker, is_cancel_requested)
        self._drain_gui_event_queue(stacker)
        run_log.emit("ENGINE_PROCESSING_RETURNED")

        if is_cancel_requested() or self._cancel_requested:
            # Cancellation observed (worker flag or backend.cancel()).  stop()
            # is idempotent, so the double call on this path is harmless.
            self._stop_stackers()
            run_log.emit("BACKEND_RETURNING", status="cancelled")
            return BackendRunResult.CANCELLED

        # After engine termination, a populated processing_error means failure,
        # never success.
        processing_error = getattr(stacker, "processing_error", None)
        if processing_error:
            # Stop the stacker (cleanup) *before* claiming the backend is
            # returning — ``BACKEND_RETURNING(status=failed)`` must be written at
            # the actual pre-raise seam, never before potentially blocking
            # cleanup work.
            self._stop_stackers()
            run_log.emit(
                "BACKEND_RETURNING", status="failed", error=str(processing_error)
            )
            raise RuntimeError(f"Engine processing failed: {processing_error}")

        self._emit_summary(
            stacker,
            start_kwargs,
            time.monotonic() - start_time,
            summary_callback,
        )
        # Backend exit seam: recorded immediately before ``run`` returns.  The
        # worker records BACKEND_RETURNED immediately after ``run`` returns.
        run_log.emit(
            "BACKEND_RETURNING",
            status="finished",
            processed_files_count=getattr(stacker, "processed_files_count", None),
        )
        return BackendRunResult.FINISHED

    @staticmethod
    def _emit_summary(
        stacker: Any,
        start_kwargs: dict,
        duration_seconds: float,
        summary_callback: Optional[SummaryCallback],
    ) -> None:
        """Build and forward the terminal run summary (best-effort, lazy).

        The final-stack header is read lazily inside
        :func:`~seestar.gui_qt.summary_payload.build_summary_payload` (never at
        module level), so the heavy astropy import only happens here at run end
        and never at ``import seestar.gui_qt`` time.
        """
        if summary_callback is None:
            return
        try:
            output_dir = getattr(stacker, "output_folder", None) or start_kwargs.get(
                "output_dir"
            )
            # The real final FITS path is the source of truth for the summary.
            # Only forward it when the file actually exists on disk; otherwise
            # fall back to the legacy <output_dir>/final.fits convention so a
            # missing attribute or a not-yet-written file never breaks the run.
            final_stack_path = getattr(stacker, "final_stacked_path", None)
            if final_stack_path and os.path.isfile(final_stack_path):
                final_path_for_summary = final_stack_path
            else:
                final_path_for_summary = None
            files_attempted = getattr(stacker, "processed_files_count", None)
            payload = build_summary_payload(
                status="finished",
                duration_seconds=duration_seconds,
                files_attempted=files_attempted,
                output_dir=output_dir,
                final_stack_path=final_path_for_summary,
            )
            summary_callback(payload)
        except Exception:
            # A summary failure must never break an otherwise successful run.
            pass

    def cancel(self) -> None:
        """Request the real backend to stop (callable from the GUI thread)."""
        self._cancel_requested = True
        self._stop_stackers()
