"""Qt lifecycle controller (M3 seam).

:class:`RunController` is the GUI-thread owner of the run lifecycle.  It turns
a canonical :class:`~seestar.gui_qt.run_bridge.RunRequest` into a simulated run
by moving a :class:`~seestar.gui_qt.run_worker.RunWorker` onto a dedicated
``QThread`` and relaying the worker's queued signals back to the GUI thread.

Threading rule (enforced by design): the controller lives on the GUI thread and
is the *only* object that may own run state and QThread teardown.  Widgets are
updated by the MainWindow slots connected to the controller's signals — never
by the worker, which holds no widget references.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal, Slot

from .run_bridge import RunRequest
from .run_worker import RunStatus, RunWorker


class RunController(QObject):
    """Owns the QThread + RunWorker pair for one run and relays its signals.

    Signals (emitted on the GUI thread):

    * ``started()`` — a run was started (emitted synchronously by ``start``),
    * ``progress_changed(int)`` — 0..100,
    * ``log_message(str)`` — log line,
    * ``finished()`` — run completed normally,
    * ``failed(str)`` — run failed with message,
    * ``cancelled()`` — run was cancelled.
    """

    started = Signal()
    progress_changed = Signal(int)
    log_message = Signal(str)
    finished = Signal()
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._status: RunStatus = RunStatus.IDLE
        self._thread: Optional[QThread] = None
        self._worker: Optional[RunWorker] = None

    # ------------------------------------------------------------------ state
    @property
    def status(self) -> RunStatus:
        return self._status

    @property
    def is_running(self) -> bool:
        return self._status is RunStatus.RUNNING

    @property
    def has_live_thread(self) -> bool:
        """True while a QThread owned by this controller is still running."""
        thread = self._thread
        return thread is not None and thread.isRunning()

    # ---------------------------------------------------------------- start
    def start(
        self,
        request: RunRequest,
        *,
        steps: int = 10,
        step_delay_ms: int = 20,
    ) -> None:
        """Start a simulated run for ``request`` on a fresh QThread.

        ``steps``/``step_delay_ms`` tune the deterministic stub (used by tests
        to keep runs fast); they will disappear when the real backend is wired
        in.  Raises if a run is already active, so a double-start is always a
        programming error rather than a silent no-op.
        """
        if self._status is RunStatus.RUNNING or self.has_live_thread:
            raise RuntimeError("RunController.start() called while a run is active")
        if not isinstance(request, RunRequest):
            raise TypeError("RunController.start() requires a RunRequest")

        thread = QThread(self)
        worker = RunWorker(steps=steps, step_delay_ms=step_delay_ms)
        worker.set_request(request)
        worker.moveToThread(thread)

        # Worker (worker thread) -> controller (GUI thread): auto connections
        # are queued because the two objects live on different threads.
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_worker_progress)
        worker.log.connect(self._on_worker_log)
        worker.finished.connect(self._on_worker_finished)
        worker.failed.connect(self._on_worker_failed)
        worker.cancelled.connect(self._on_worker_cancelled)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._thread = thread
        self._worker = worker
        self._status = RunStatus.RUNNING
        thread.start()
        self.started.emit()

    # ------------------------------------------------------- signal relays
    @Slot(int)
    def _on_worker_progress(self, percent: int) -> None:
        if self._status is not RunStatus.RUNNING:
            return
        self.progress_changed.emit(percent)

    @Slot(str)
    def _on_worker_log(self, message: str) -> None:
        if self._status is not RunStatus.RUNNING:
            return
        self.log_message.emit(message)

    @Slot()
    def _on_worker_finished(self) -> None:
        if self._status is not RunStatus.RUNNING:
            return
        self._status = RunStatus.FINISHED
        self.finished.emit()

    @Slot(str)
    def _on_worker_failed(self, message: str) -> None:
        if self._status is not RunStatus.RUNNING:
            return
        self._status = RunStatus.FAILED
        self.failed.emit(message)

    @Slot()
    def _on_worker_cancelled(self) -> None:
        if self._status is not RunStatus.RUNNING:
            return
        self._status = RunStatus.CANCELLED
        self.cancelled.emit()

    @Slot()
    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        # Defensive: a thread that ended without a terminal worker signal is
        # not a normal path for this stub, but never leave it stuck RUNNING.
        if self._status is RunStatus.RUNNING:
            self._status = RunStatus.FINISHED

    # ---------------------------------------------------------------- stop
    def cancel(self) -> None:
        """Request cancellation of the active run (non-blocking).

        The worker polls the flag and emits ``cancelled`` from its own thread;
        the GUI learns of the outcome through the ``cancelled`` signal.
        """
        worker = self._worker
        if worker is not None:
            worker.request_cancel()

    def shutdown(self) -> None:
        """Idempotent teardown: cancel any active run and reap its QThread.

        Safe to call multiple times and safe when idle.  After this returns,
        no QThread owned by this controller is running and the status is IDLE.
        """
        if self._status is RunStatus.RUNNING or self.has_live_thread:
            self.cancel()
            thread = self._thread
            if thread is not None:
                thread.wait(5000)
        self._thread = None
        self._worker = None
        self._status = RunStatus.IDLE
