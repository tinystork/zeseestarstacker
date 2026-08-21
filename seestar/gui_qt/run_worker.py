"""Qt lifecycle worker (M3/M4 seam): a run driver with an injectable backend.

This module defines :class:`RunWorker`, the object that executes one run inside
a dedicated ``QThread``.  It is the Qt-side counterpart of the Tk GUI's starter
thread, but — per the M3/M4 seam rules — it never imports the Tk GUI, never
imports the scientific engine, and never touches Qt widgets.

The worker no longer hardcodes a simulated loop.  It delegates to a *run
backend* (a :class:`~seestar.gui_qt.backend_runner.BaseRunBackend`):

* the default is the deterministic
  :class:`~seestar.gui_qt.backend_runner.SimulatedRunBackend` (safe offscreen
  smoke behaviour),
* tests (or a future activation milestone) can inject a real or fake backend
  through :meth:`RunController.start(..., backend=...)`.

The worker communicates with the GUI *exclusively* through queued Qt signals
(:attr:`progress`, :attr:`log`, :attr:`finished`, :attr:`failed`,
:attr:`cancelled`, :attr:`preview`).  It holds no widget references; the GUI
never mutates worker internals directly.  Cancellation is requested through
the thread-safe
:meth:`RunWorker.request_cancel`, which both sets the poll flag and forwards a
``cancel()`` call to the backend so a real backend can stop its engine.
"""

from __future__ import annotations

import enum
import inspect
from typing import Optional

from PySide6.QtCore import QMutex, QObject, QThread, Signal, Slot

from .backend_runner import (
    BackendRunResult,
    BaseRunBackend,
    SimulatedRunBackend,
)
from .run_bridge import RunRequest


class RunStatus(enum.Enum):
    """Lifecycle state of a run as seen from the GUI thread."""

    IDLE = "idle"
    RUNNING = "running"
    CANCELLED = "cancelled"
    FINISHED = "finished"
    FAILED = "failed"


class RunWorker(QObject):
    """A QObject moved onto a QThread that executes one run via a backend.

    Signals (all delivered to the GUI thread via queued connections):

    * ``progress(int)`` — 0..100 percent complete,
    * ``log(str)`` — human-readable log line,
    * ``finished()`` — completed without cancellation,
    * ``failed(str)`` — terminal error (message),
    * ``cancelled()`` — cancellation was requested and honoured,
    * ``preview(object)`` — a
      :class:`~seestar.gui_qt.backend_runner.BackendPreviewPayload` carrying a
      preview metadata update from the backend,
    * ``summary(object)`` — a
      :class:`~seestar.gui_qt.summary_payload.SummaryPayload` carrying the
      terminal run summary from the backend.

    Parameters
    ----------
    backend:
        A :class:`~seestar.gui_qt.backend_runner.BaseRunBackend` to execute the
        run.  Defaults to :class:`SimulatedRunBackend(steps, step_delay_ms)`.
    steps, step_delay_ms:
        Only used to construct the default simulated backend (tuned by tests);
        ignored when an explicit ``backend`` is supplied.
    """

    progress = Signal(int)
    log = Signal(str)
    finished = Signal()
    failed = Signal(str)
    cancelled = Signal()
    preview = Signal(object)
    summary = Signal(object)

    def __init__(
        self,
        backend: Optional[BaseRunBackend] = None,
        steps: int = 10,
        step_delay_ms: int = 20,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        if backend is None:
            backend = SimulatedRunBackend(
                steps=steps, step_delay_ms=step_delay_ms
            )
        self._backend: BaseRunBackend = backend
        self._request: Optional[RunRequest] = None
        self._cancel_mutex = QMutex()
        self._cancel_requested = False

    def set_request(self, request: RunRequest) -> None:
        """Attach the run request the worker will execute (GUI thread only)."""
        self._request = request

    def request_cancel(self) -> None:
        """Thread-safe cancellation request, polled by the run loop.

        Also forwards ``cancel()`` to the backend so a real backend can stop
        its engine immediately (the simulated backend records the flag).
        """
        self._cancel_mutex.lock()
        try:
            self._cancel_requested = True
        finally:
            self._cancel_mutex.unlock()
        backend = self._backend
        if backend is not None:
            try:
                backend.cancel()
            except Exception:
                pass

    def set_preview_downsample_factor(self, factor: int) -> None:
        """Forward a live preview-downsample request to the backend (GUI thread).

        Thread-safe in the same way as :meth:`request_cancel`: the backend is
        responsible for applying the factor to the live stacker on the worker
        thread (or no-op when it has no engine).  A missing backend is a silent
        no-op.
        """
        backend = self._backend
        if backend is not None:
            try:
                backend.set_preview_downsample_factor(factor)
            except Exception:
                pass

    def _is_cancel_requested(self) -> bool:
        self._cancel_mutex.lock()
        try:
            return self._cancel_requested
        finally:
            self._cancel_mutex.unlock()

    def _emit_progress(self, percent: int) -> None:
        self.progress.emit(int(percent))

    def _emit_log(self, message: str) -> None:
        self.log.emit(str(message))

    def _emit_preview(self, payload) -> None:
        self.preview.emit(payload)

    def _emit_summary(self, payload) -> None:
        self.summary.emit(payload)

    def _accepts_summary_callback(self) -> bool:
        """True when the injected backend's ``run`` accepts ``summary_callback``.

        Existing test fakes subclass :class:`BaseRunBackend` with a narrower
        ``run(..., preview_callback=None)`` signature; passing an unexpected
        keyword would raise.  We introspect the bound method so the summary
        channel is additive and old backends keep working unchanged.
        """
        try:
            sig = inspect.signature(self._backend.run)
        except (TypeError, ValueError):
            return False
        return "summary_callback" in sig.parameters

    @Slot()
    def run(self) -> None:
        """Execute the run via the backend.  Runs in the worker thread."""
        thread = self.thread()
        try:
            request = self._request
            if request is None:
                self.failed.emit("RunWorker.run() called without a RunRequest")
                return

            run_kwargs = {
                "progress_callback": self._emit_progress,
                "log_callback": self._emit_log,
                "is_cancel_requested": self._is_cancel_requested,
                "preview_callback": self._emit_preview,
            }
            if self._accepts_summary_callback():
                run_kwargs["summary_callback"] = self._emit_summary
            result = self._backend.run(request, **run_kwargs)

            if result is BackendRunResult.CANCELLED:
                self.cancelled.emit()
            else:
                self.finished.emit()
        except Exception as exc:  # pragma: no cover - defensive guard
            self.failed.emit(str(exc))
        finally:
            # Stop the worker thread's event loop as soon as ``run`` returns.
            # ``QThread.quit`` is thread-safe, so no GUI-thread round trip is
            # needed (and shutdown can wait without deadlocking the GUI).
            if thread is not None:
                thread.quit()
