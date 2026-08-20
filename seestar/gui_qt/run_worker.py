"""Qt lifecycle worker (M3 seam): a deterministic, backend-free simulated run.

This module defines :class:`RunWorker`, the object that performs a *simulated*
stacking run inside a dedicated ``QThread``.  It is the Qt-side counterpart of
the Tk GUI's starter thread, but — per the M3 seam rules — it never imports the
Tk GUI, never imports the scientific engine, and never touches Qt widgets.

The worker communicates with the GUI *exclusively* through queued Qt signals
(:attr:`progress`, :attr:`log`, :attr:`finished`, :attr:`failed`,
:attr:`cancelled`).  It holds no widget references, and the GUI never mutates
worker internals directly: cancellation is requested through the thread-safe
:meth:`RunWorker.request_cancel` flag, which the run loop polls between steps.
"""

from __future__ import annotations

import enum
from typing import Optional

from PySide6.QtCore import QMutex, QObject, QThread, Signal, Slot

from .run_bridge import RunRequest


class RunStatus(enum.Enum):
    """Lifecycle state of a (simulated) run as seen from the GUI thread."""

    IDLE = "idle"
    RUNNING = "running"
    CANCELLED = "cancelled"
    FINISHED = "finished"
    FAILED = "failed"


class RunWorker(QObject):
    """A QObject moved onto a QThread that executes one simulated run.

    The worker is deliberately deterministic: it advances a progress counter
    through ``steps`` equally sized steps, sleeping ``step_delay_ms`` between
    each, and emits a log line + progress value per step.  It accepts a
    canonical :class:`~seestar.gui_qt.run_bridge.RunRequest` so the signal
    payloads can later be replaced by real backend events without changing the
    controller/UI wiring.

    Signals (all delivered to the GUI thread via queued connections):

    * ``progress(int)`` — 0..100 percent complete,
    * ``log(str)`` — human-readable log line,
    * ``finished()`` — completed without cancellation,
    * ``failed(str)`` — terminal error (message),
    * ``cancelled()`` — cancellation was requested and honoured.
    """

    progress = Signal(int)
    log = Signal(str)
    finished = Signal()
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        steps: int = 10,
        step_delay_ms: int = 20,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._steps = max(1, int(steps))
        self._step_delay_ms = max(0, int(step_delay_ms))
        self._request: Optional[RunRequest] = None
        self._cancel_mutex = QMutex()
        self._cancel_requested = False

    def set_request(self, request: RunRequest) -> None:
        """Attach the run request the worker will simulate (GUI thread only)."""
        self._request = request

    def request_cancel(self) -> None:
        """Thread-safe cancellation request, polled by the run loop."""
        self._cancel_mutex.lock()
        try:
            self._cancel_requested = True
        finally:
            self._cancel_mutex.unlock()

    def _is_cancel_requested(self) -> bool:
        self._cancel_mutex.lock()
        try:
            return self._cancel_requested
        finally:
            self._cancel_mutex.unlock()

    @Slot()
    def run(self) -> None:
        """Execute the simulated run.  Runs in the worker thread."""
        thread = self.thread()
        try:
            request = self._request
            if request is None:
                self.failed.emit("RunWorker.run() called without a RunRequest")
                return

            batch_size = request.backend_kwargs.get("batch_size")
            self.log.emit(f"Simulated run started (batch_size={batch_size}).")

            for step in range(self._steps):
                if self._is_cancel_requested():
                    self.log.emit("Cancellation requested — stopping simulated run.")
                    self.cancelled.emit()
                    return
                percent = int(round(100.0 * (step + 1) / self._steps))
                self.progress.emit(percent)
                self.log.emit(f"Simulated step {step + 1}/{self._steps} ({percent}%).")
                QThread.msleep(self._step_delay_ms)

            self.progress.emit(100)
            self.log.emit("Simulated run finished.")
            self.finished.emit()
        except Exception as exc:  # pragma: no cover - defensive guard
            self.failed.emit(str(exc))
        finally:
            # Stop the worker thread's event loop as soon as ``run`` returns.
            # ``QThread.quit`` is thread-safe, so no GUI-thread round trip is
            # needed (and shutdown can wait without deadlocking the GUI).
            if thread is not None:
                thread.quit()
