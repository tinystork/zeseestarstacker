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

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

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
      :meth:`_on_worker_preview`),
    * ``summary_updated(object)`` — a
      :class:`~seestar.gui_qt.summary_payload.SummaryPayload` relayed from the
      backend at the end of a successful run (see :meth:`_on_worker_summary`).
    """

    started = Signal()
    progress_changed = Signal(int)
    log_message = Signal(str)
    finished = Signal()
    failed = Signal(str)
    cancelled = Signal()
    refused = Signal(object)
    preview_updated = Signal(object)
    summary_updated = Signal(object)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._status: RunStatus = RunStatus.IDLE
        self._thread: Optional[QThread] = None
        self._worker: Optional[RunWorker] = None
        # Terminal worker outcome stored until the QThread has actually
        # finished (then published from the GUI thread).  Tuple of
        # (kind, payload) where kind is one of finished/failed/cancelled/refused.
        self._pending_outcome: Optional[tuple] = None
        # Durable run-log carrier handed back by the worker (shared by
        # reference); consumed/closed by the terminal GUI handler.
        self._run_log = None
        # True while a one-event-turn deferred publication is pending (used to
        # close the cross-sender queued-signal race without a GUI-thread wait).
        self._publish_deferred = False

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

    @property
    def run_log(self):
        """The shared durable run-log carrier for the current run (or None)."""
        return self._run_log

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

        # Reset per-run terminal state (a controller may be reused across runs).
        self._pending_outcome = None
        self._run_log = None
        self._publish_deferred = False

        # Worker (worker thread) -> controller (GUI thread): auto connections
        # are queued because the two objects live on different threads.
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_worker_progress)
        worker.log.connect(self._on_worker_log)
        worker.preview.connect(self._on_worker_preview)
        worker.summary.connect(self._on_worker_summary)
        worker.lifecycle_log.connect(self._on_worker_lifecycle_log)
        worker.finished.connect(self._on_worker_finished)
        worker.failed.connect(self._on_worker_failed)
        worker.cancelled.connect(self._on_worker_cancelled)
        worker.refused.connect(self._on_worker_refused)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        worker.refused.connect(worker.deleteLater)
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

    @Slot(object)
    def _on_worker_summary(self, payload) -> None:
        if self._status is not RunStatus.RUNNING:
            return
        self.summary_updated.emit(payload)

    @Slot()
    def _on_worker_finished(self) -> None:
        if self._status is not RunStatus.RUNNING:
            return
        self._pending_outcome = ("finished", None)

    @Slot(str)
    def _on_worker_failed(self, message: str) -> None:
        if self._status is not RunStatus.RUNNING:
            return
        self._pending_outcome = ("failed", message)

    @Slot()
    def _on_worker_cancelled(self) -> None:
        if self._status is not RunStatus.RUNNING:
            return
        self._pending_outcome = ("cancelled", None)

    @Slot(object)
    def _on_worker_refused(self, payload) -> None:
        if self._status is not RunStatus.RUNNING:
            return
        self._pending_outcome = ("refused", payload)

    @Slot(object)
    def _on_worker_lifecycle_log(self, run_log) -> None:
        self._run_log = run_log

    @Slot()
    def _on_thread_finished(self) -> None:
        """Reap the finished QThread, then publish the stored worker outcome.

        The public terminal signal (which makes MainWindow idle / re-enables
        Start) must not fire before the owned QThread has actually finished:
        the outcome is stored when the worker signals arrive and published here,
        after the thread is dead, on the GUI thread.
        """
        self._thread = None
        self._worker = None
        if self._status is not RunStatus.RUNNING:
            # ``shutdown`` already tore the run down (status IDLE); never
            # publish a spurious terminal signal after a forced shutdown.
            return
        if self._pending_outcome is None and not self._publish_deferred:
            # Cross-sender queued-signal race: the worker's terminal signal may
            # still be in the GUI event queue.  Defer exactly one GUI event turn
            # (no GUI-thread wait/join) so the queued outcome can land first.
            self._publish_deferred = True
            QTimer.singleShot(0, self._on_deferred_publication)
            return
        self._publish_thread_outcome()

    @Slot()
    def _on_deferred_publication(self) -> None:
        """Publish the thread outcome after one GUI event turn (race-close)."""
        self._publish_deferred = False
        self._publish_thread_outcome()

    def _publish_thread_outcome(self) -> None:
        """Publish the stored terminal outcome (or an explicit missing outcome)."""
        if self._status is not RunStatus.RUNNING:
            return
        outcome = self._pending_outcome
        self._pending_outcome = None
        run_log = self._run_log
        if run_log is not None:
            run_log.emit("QTHREAD_FINISHED")
        if outcome is None:
            # A thread that ended without a terminal worker notification must
            # never become false success: report an explicit failure.
            if run_log is not None:
                run_log.emit("WORKER_OUTCOME_MISSING")
            self._status = RunStatus.FAILED
            self.failed.emit(
                "Worker thread finished without a terminal outcome notification"
            )
            self._finalize_run_log("failed")
            return
        self._publish_outcome(outcome)

    def _publish_outcome(self, outcome: tuple) -> None:
        """Publish the stored terminal outcome from the GUI thread."""
        kind, payload = outcome
        if kind == "failed":
            self._status = RunStatus.FAILED
            self.failed.emit(payload)
        elif kind == "cancelled":
            self._status = RunStatus.CANCELLED
            self.cancelled.emit()
        elif kind == "refused":
            self._status = RunStatus.FAILED
            self.refused.emit(payload)
            # A refused run never opened a run log; nothing to finalize.
            return
        else:
            self._status = RunStatus.FINISHED
            self.finished.emit()
        # ``self.<signal>.emit(...)`` returns only after the connected terminal
        # slots (MainWindow) have actually returned.  Record the truthful
        # QT_COMPLETION_HANDLER_RETURNED and close the durable run log here, on
        # the controller side, so the event is never written before the handler
        # (or a blocking close/tail) has truly finished.
        self._finalize_run_log(kind)

    def _finalize_run_log(self, outcome: str) -> None:
        """Record the truthful handler-return and close the durable run log.

        Called immediately after the public terminal signal emit has returned,
        so ``QT_COMPLETION_HANDLER_RETURNED`` is only written once the MainWindow
        terminal slot has actually finished.  A refused run never opened a run
        log (``_run_log`` is ``None``), so this is a no-op for it.
        """
        run_log = self._run_log
        if run_log is None:
            return
        run_log.emit("QT_COMPLETION_HANDLER_RETURNED", outcome=outcome)
        run_log.close()

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

    def set_preview_downsample_factor(self, factor: int) -> None:
        """Request a live preview-downsample factor change during a run.

        Thread-safe control channel (GUI thread -> worker -> backend -> live
        stacker).  A no-op when no run is active (idle Res clicks stay
        display-only), and the backend applies the factor on the worker thread
        so the stacker is only ever mutated by its owner thread.
        """
        worker = self._worker
        if worker is not None:
            worker.set_preview_downsample_factor(factor)

    def shutdown(self, wait_ms: int = 5000) -> bool:
        """Idempotent teardown: cancel any active run and reap its QThread.

        Safe to call multiple times and safe when idle.  Returns ``True`` when
        no QThread owned by this controller is still running after the teardown
        (fully shut down).  Returns ``False`` when the worker/thread refused to
        finish within ``wait_ms``; in that case the thread and worker
        references are **retained** (never destroyed while still running, so
        the classic ``QThread: Destroyed while thread is still running`` crash
        cannot happen) and cleanup is deferred: the worker's own
        ``finished``/``failed``/``cancelled`` connections still delete it when
        the thread eventually stops, and a later :meth:`shutdown` call retries.
        """
        if self._status is RunStatus.RUNNING or self.has_live_thread:
            self.cancel()
            thread = self._thread
            if thread is not None:
                thread.requestInterruption()
                thread.quit()
                if not thread.wait(wait_ms):
                    # Do NOT drop the references: dropping them would let the
                    # QThread be destroyed while still running.
                    self._status = RunStatus.CANCELLED
                    return False
        self._thread = None
        self._worker = None
        self._pending_outcome = None
        self._run_log = None
        self._publish_deferred = False
        self._status = RunStatus.IDLE
        return True
