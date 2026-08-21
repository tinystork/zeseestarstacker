"""ZSSS-QT-RELIABILITY-LIFECYCLE tests: SUCCESS vs EMPTY/NO OUTPUT terminal runs.

Covers the presentation-layer distinction between the four terminal states at
the Qt level:

* SUCCESS — a run finished without cancellation and produced a final output
  (``final_stack_exists`` True, ``images_in_final_stack`` not None/0, and
  ``can_open_output`` True).  This keeps the legacy "Finished." / "Run
  finished." presentation and the summary dialog shows "SUCCESS".
* EMPTY/NO OUTPUT — a run finished without cancellation but produced no final
  output (or emitted no summary payload at all).  This is presented distinctly
  ("No output produced." / "Run finished with no output.") and the summary
  dialog shows "EMPTY/NO OUTPUT" instead of "SUCCESS".
* FAILED / CANCELLED — already handled by the existing ``failed``/``cancelled``
  signals (regression-asserted only).

Also verifies a second run does not reuse the first run's payload state (the
summary payload is consumed and cleared; a second run shows its own payload).

Engine-free (fake backends only).  ``QT_QPA_PLATFORM=offscreen`` is set
defensively before any ``QApplication`` is created.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QLabel

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt.backend_runner import BackendRunResult, BaseRunBackend
from seestar.gui_qt.summary_payload import SummaryPayload, derive_terminal_status


def _pump_until(qapp: QApplication, predicate, timeout_ms: int = 5000) -> bool:
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    qapp.processEvents()
    return bool(predicate())


@pytest.fixture(scope="session")
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


@pytest.fixture()
def window(qapp):
    win = MainWindow()
    yield win
    win.shutdown()


def _summary_label_text(win: MainWindow) -> str:
    assert win._summary_dialog is not None, "no summary dialog shown"
    labels = win._summary_dialog.findChildren(QLabel)
    assert labels, "summary dialog has no QLabel"
    return labels[0].text()


def _success_payload() -> SummaryPayload:
    return SummaryPayload(
        status="finished",
        duration_seconds=30.0,
        files_attempted=5,
        final_stack_file="/out/final.fits",
        final_stack_exists=True,
        images_in_final_stack=5,
        total_exposure_seconds=60.0,
        can_open_output=True,
    )


def _empty_payload() -> SummaryPayload:
    return SummaryPayload(
        status="finished",
        duration_seconds=30.0,
        files_attempted=5,
        final_stack_file="/out/final.fits",
        final_stack_exists=False,
        images_in_final_stack=None,
        total_exposure_seconds=None,
        can_open_output=False,
    )


# --------------------------------------------------------------------------
# Pure derivation helper
# --------------------------------------------------------------------------
def test_derive_terminal_status_pure():
    assert derive_terminal_status(None) == "empty"
    assert (
        derive_terminal_status(SummaryPayload(status="finished", final_stack_exists=False))
        == "empty"
    )
    assert (
        derive_terminal_status(
            SummaryPayload(status="finished", final_stack_exists=True, images_in_final_stack=None)
        )
        == "empty"
    )
    assert (
        derive_terminal_status(
            SummaryPayload(status="finished", final_stack_exists=True, images_in_final_stack=0)
        )
        == "empty"
    )
    assert (
        derive_terminal_status(
            SummaryPayload(
                status="finished",
                final_stack_exists=True,
                images_in_final_stack=3,
                can_open_output=False,
            )
        )
        == "empty"
    )
    assert derive_terminal_status(_success_payload()) == "success"
    assert derive_terminal_status(_empty_payload()) == "empty"


# --------------------------------------------------------------------------
# _on_run_finished presentation
# --------------------------------------------------------------------------
def test_run_finished_success_keeps_legacy_presentation(window):
    win = window
    win.log("before")
    win._last_summary_payload = _success_payload()
    win._on_run_finished()

    assert win.statusBar().currentMessage() == "Finished."
    assert win.log_view.toPlainText() == "before\nRun finished."
    assert "Status: SUCCESS" in _summary_label_text(win)
    # The payload is consumed so it cannot leak into a later run.
    assert win._last_summary_payload is None


def test_run_finished_empty_uses_distinct_presentation(window):
    win = window
    win.log("before")
    win._last_summary_payload = _empty_payload()
    win._on_run_finished()

    assert win.statusBar().currentMessage() == "No output produced."
    assert win.log_view.toPlainText() == "before\nRun finished with no output."
    assert "Status: EMPTY/NO OUTPUT" in _summary_label_text(win)
    assert win._last_summary_payload is None


def test_run_finished_with_no_payload_is_empty_not_success(window):
    """A run that ends with no summary payload at all must not read as success."""
    win = window
    assert win._last_summary_payload is None
    win.log("before")
    win._on_run_finished()

    assert win.statusBar().currentMessage() == "No output produced."
    assert win.log_view.toPlainText() == "before\nRun finished with no output."
    # No payload -> no summary dialog either.
    assert win._summary_dialog is None


# --------------------------------------------------------------------------
# boring route: same distinction
# --------------------------------------------------------------------------
def test_boring_finished_empty_uses_distinct_presentation(window):
    win = window
    win.log("before")
    win._last_summary_payload = _empty_payload()
    win._on_boring_finished(0)

    assert win.statusBar().currentMessage() == "Boring stack finished with no output."
    assert win.log_view.toPlainText() == "before\nBoring stack finished with no output."
    assert "Status: EMPTY/NO OUTPUT" in _summary_label_text(win)


def test_boring_finished_success_keeps_legacy_presentation(window):
    win = window
    win.log("before")
    win._last_summary_payload = _success_payload()
    win._on_boring_finished(0)

    assert win.statusBar().currentMessage() == "Boring stack finished."
    assert win.log_view.toPlainText() == "before\nBoring stack finished."
    assert "Status: SUCCESS" in _summary_label_text(win)


# --------------------------------------------------------------------------
# two successive runs -> no state reuse
# --------------------------------------------------------------------------
class _PayloadBackend(BaseRunBackend):
    """Emits one fixed payload per run, drawn from a queue, and finishes."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.summary_calls = []

    def run(
        self,
        request,
        progress_callback,
        log_callback,
        is_cancel_requested,
        preview_callback=None,
        summary_callback=None,
    ):
        progress_callback(100)
        payload = self.payloads.pop(0)
        if summary_callback is not None:
            self.summary_calls.append(payload)
            summary_callback(payload)
        return BackendRunResult.FINISHED

    def cancel(self):
        pass


def test_two_successive_runs_do_not_reuse_payload_state(qapp):
    backend = _PayloadBackend([_success_payload(), _empty_payload()])
    win = MainWindow(backend_factory=lambda: backend)
    try:
        # First run -> SUCCESS.
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)
        assert win._last_summary_payload is None  # consumed, not lingering
        assert "Status: SUCCESS" in _summary_label_text(win)

        # Buttons are re-enabled and ready for a fresh run.
        assert win.start_button.isEnabled()
        assert not win.stop_button.isEnabled()
        assert win._last_summary_payload is None

        # Second run -> EMPTY (its own payload, not the first run's).
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)
        assert win._last_summary_payload is None
        text = _summary_label_text(win)
        assert "Status: EMPTY/NO OUTPUT" in text
        assert "Status: SUCCESS" not in text
    finally:
        win.shutdown()


def test_two_successive_runs_empty_then_success(qapp):
    backend = _PayloadBackend([_empty_payload(), _success_payload()])
    win = MainWindow(backend_factory=lambda: backend)
    try:
        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)
        assert "Status: EMPTY/NO OUTPUT" in _summary_label_text(win)

        win.start_button.click()
        assert _pump_until(qapp, lambda: win.is_running is False)
        assert win._last_summary_payload is None
        text = _summary_label_text(win)
        assert "Status: SUCCESS" in text
        assert "Status: EMPTY/NO OUTPUT" not in text
    finally:
        win.shutdown()
