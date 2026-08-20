"""Qt lifecycle controller (M3/M4 seam).

:class:`RunController` is the GUI-thread owner of the run lifecycle.  It turns
a canonical :class:`~seestar.gui_qt.run_bridge.RunRequest` into a run by moving
a :class:`~seestar.gui_qt.run_worker.RunWorker` onto a dedicated ``QThread`` and
relaying the worker's queued signals back to the GUI thread.

The *run backend* that actually executes the request is injectable: the default
is the simulated runner (safe for offscreen smoke tests), while tests (and the
next activation milestone) can pass a real or fake
:class:`~seestar.gui_qt.backend_runner.BaseRunBackend` via ``backend=``.

Threading rule (enforced by design): the controller lives on the GUI thread and
is the *only* object that may own run state and QThread teardown.  Widgets are
updated by the MainWindow slots connected to the controller's signals — never
by the worker or backend, which hold no widget references.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal, Slot

from .backend_runner import BaseRunBackend
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
    * ``cancelled()`` — run was cancelled,
    * ``preview_updated(object)`` — a
      :class:`~seestar.gui_qt.backend_runner.BackendPreviewPayload` relayed
      from the backend (dropped after shutdown, see
      :meth:`_on_worker_preview`).
    """

    started = Signal()
    progress_changed = Signal(int)
    log_message = Signal(str)
    finished = Signal()
    failed = Signal(str)
    cancelled = Signal()
    preview_updated = Signal(object)

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
        backend: Optional[BaseRunBackend] = None,
    ) -> None:
        """Start a run for ``request`` on a fresh QThread.

        ``backend`` selects the run backend.  When ``None`` (the default), the
        deterministic :class:`SimulatedRunBackend` is used and
        ``steps``/``step_delay_ms`` tune it (used by tests to keep runs fast);
        those two knobs disappear from relevance once the real backend is wired
        in.  Raises if a run is already active, so a double-start is always a
        programming error rather than a silent no-op.
        """
        if self._status is RunStatus.RUNNING or self.has_live_thread:
            raise RuntimeError("RunController.start() called while a run is active")
        if not isinstance(request, RunRequest):
            raise TypeError("RunController.start() requires a RunRequest")
        if backend is not None and not isinstance(backend, BaseRunBackend):
            raise TypeError("RunController.start() backend must be a BaseRunBackend")

        thread = QThread(self)
        worker = RunWorker(
            backend=backend, steps=steps, step_delay_ms=step_delay_ms
        )
        worker.set_request(request)
        worker.moveToThread(thread)

        # Worker (worker thread) -> controller (GUI thread): auto connections
        # are queued because the two objects live on different threads.
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_worker_progress)
        worker.log.connect(self._on_worker_log)
        worker.preview.connect(self._on_worker_preview)
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

    @Slot(object)
    def _on_worker_preview(self, payload) -> None:
        if self._status is not RunStatus.RUNNING:
            return
        self.preview_updated.emit(payload)

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

        The worker polls the flag (and forwards ``cancel()`` to the backend) and
        emits ``cancelled`` from its own thread; the GUI learns of the outcome
        through the ``cancelled`` signal.
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
