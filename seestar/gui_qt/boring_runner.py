"""Qt boring-stack subprocess runner (QProcess-based, GUI-thread friendly).

Launches ``seestar/gui/boring_stack.py`` as a child process *without* blocking
the GUI thread.  ``QProcess`` emits its signals on the object's thread (the GUI
thread for a runner parented to the window), so :class:`MainWindow` connects
directly to them and never has to spin a synchronous ``subprocess`` call.

Import-hygiene: this module imports only QtCore and the pure-stdlib
:mod:`seestar.gui_qt.boring_route`.  The boring script is launched by
*filesystem path* (see :func:`boring_route.resolve_boring_script_path`) and is
never imported, so ``import seestar.gui_qt`` stays free of the Tk GUI and the
scientific engine.

``BoringRunnerBase`` is the injectable interface used by tests (a fake records
the request and drives terminal signals without spawning a real process);
``QProcessBoringRunner`` is the default real runner, used only outside tests.
"""

from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import QObject, QProcess, Signal

from .boring_route import BoringRunRequest, parse_stack_plan_csv
from .summary_payload import SummaryPayload, build_summary_payload


class BoringRunnerBase(QObject):
    """Interface + base for boring subprocess runners (injectable for tests).

    Signal contract (all emitted on the GUI thread):

    * ``started()`` — a launch was requested,
    * ``finished(int)`` — the process exited (``0`` == success),
    * ``failed(str)`` — the process could not launch or exited non-zero,
    * ``cancelled()`` — the process was terminated after a cancel request,
    * ``log_message(str)`` — one line of merged stdout/stderr,
    * ``summary(object)`` — a :class:`SummaryPayload` emitted at the end of a
      successful run.
    """

    started = Signal()
    finished = Signal(int)
    failed = Signal(str)
    cancelled = Signal()
    log_message = Signal(str)
    summary = Signal(object)

    def start(self, request: BoringRunRequest) -> None:
        raise NotImplementedError

    def cancel(self) -> None:
        raise NotImplementedError

    def is_running(self) -> bool:
        raise NotImplementedError


class QProcessBoringRunner(BoringRunnerBase):
    """Real boring runner driving a ``QProcess`` for ``boring_stack.py``."""

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._proc: Optional[QProcess] = None
        self._cancel_requested = False
        self._start_time: Optional[float] = None
        self._request: Optional[BoringRunRequest] = None

    def start(self, request: BoringRunRequest) -> None:
        if self.is_running():
            raise RuntimeError("boring runner is already active")
        self._cancel_requested = False
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        proc.setProgram(request.command[0])
        proc.setArguments(request.command[1:])
        self._proc = proc
        self._start_time = time.monotonic()
        self._request = request
        self.started.emit()
        proc.start()

    def cancel(self) -> None:
        """Request cancellation via SIGTERM (boring_stack.py stops gracefully)."""
        self._cancel_requested = True
        proc = self._proc
        if proc is not None and proc.state() != QProcess.ProcessState.NotRunning:
            proc.terminate()

    def is_running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.state() != QProcess.ProcessState.NotRunning

    # ------------------------------------------------------------------ slots
    def _on_output(self) -> None:
        proc = self._proc
        if proc is None:
            return
        raw = bytes(proc.readAllStandardOutput())
        text = raw.decode("utf-8", errors="replace")
        for line in text.splitlines():
            if line.strip():
                self.log_message.emit(line)

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            proc = self._proc
            detail = proc.errorString() if proc is not None else "unknown error"
            self._proc = None
            self.failed.emit(f"boring_stack.py failed to start: {detail}")

    def _on_finished(self, exit_code: int, exit_status) -> None:
        self._on_output()  # flush any remaining buffered output
        self._proc = None
        if self._cancel_requested:
            self.cancelled.emit()
            return
        if exit_code == 0:
            self._emit_summary()
            self.finished.emit(exit_code)
        else:
            self.failed.emit(f"boring_stack.py exited with code {exit_code}")

    def _files_attempted(self) -> Optional[int]:
        """Return the number of files the boring CSV listed (None on failure)."""
        request = self._request
        if request is None:
            return None
        try:
            parsed = parse_stack_plan_csv(request.csv_path)
            return len(parsed.ordered_files)
        except Exception:
            return None

    def _emit_summary(self) -> None:
        """Build and emit the terminal summary payload for a successful run."""
        request = self._request
        output_dir = request.output_dir if request is not None else None
        duration = (
            time.monotonic() - self._start_time
            if self._start_time is not None
            else None
        )
        payload = build_summary_payload(
            status="finished",
            duration_seconds=duration,
            files_attempted=self._files_attempted(),
            output_dir=output_dir,
        )
        self.summary.emit(payload)
