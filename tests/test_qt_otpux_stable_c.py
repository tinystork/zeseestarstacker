"""ZSSS-OTPUX-STABLE-C tests: unmistakable terminal-failure presentation.

The Qt shell previously surfaced terminal failures (normal backend failure,
Boring failure, preflight refusal and structured startup refusal) only through
the status bar and log tab.  STABLE-C makes those failures impossible to miss
by presenting them through a genuine, owned, non-blocking ``QMessageBox`` while
keeping the existing status-bar/log text exactly truthful.

Contract under test:

* ``_on_run_failed`` / ``_on_boring_failed`` show exactly one **Critical** box.
* ``_report_preflight_failure`` and ``_on_run_refused`` show exactly one
  **Warning** box (user-correctable), reusing ``_format_refusal`` for the
  refusal title/body.
* The box is non-blocking (``show`` + window modality, never ``exec``, never a
  static ``QMessageBox.critical/warning/information``) so the controller-owned
  terminal cleanup always runs.
* A second distinct failure replaces the first in the *same* owned box — never
  a second stacked box.
* Success / empty / cancelled show no error box and keep the summary dialog.
* Titles localize EN/FR; shutdown closes any outstanding box idempotently.

Engine-free (fake backends only).  ``QT_QPA_PLATFORM=offscreen`` is set
defensively before any ``QApplication`` is created.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from seestar.gui_qt import MainWindow, RunStatus, create_application
from seestar.gui_qt import localization
from seestar.gui_qt.backend_runner import BaseRunBackend
from seestar.gui_qt.startup_refusal import StartupRefusalPayload


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


def _assert_box(win: MainWindow, icon, title: str, body: str) -> QMessageBox:
    """Assert the owned box is visible, correct severity/title/body, plain text."""
    box = win.error_message_box
    assert box is not None, "no error message box present"
    assert isinstance(box, QMessageBox)
    assert box.isVisible(), "error box is not visible"
    assert box.icon() is icon, f"expected icon {icon}, got {box.icon()}"
    assert box.windowTitle() == title, f"title {box.windowTitle()!r} != {title!r}"
    assert box.text() == body, f"body {box.text()!r} != {body!r}"
    assert box.textFormat() == Qt.TextFormat.PlainText, "body must be plain text"
    return box


class FailingBackend(BaseRunBackend):
    """A fake backend that raises, driving the controller ``failed`` signal."""

    def __init__(self) -> None:
        self.run_calls = 0

    def run(
        self,
        request,
        progress_callback,
        log_callback,
        is_cancel_requested,
        preview_callback=None,
        summary_callback=None,
    ):
        self.run_calls += 1
        progress_callback(10)
        raise RuntimeError("backend exploded")

    def cancel(self) -> None:
        pass


# --------------------------------------------------------------------------
# A. normal backend terminal failure
# --------------------------------------------------------------------------
def test_run_failed_shows_one_critical_box(window):
    win = window
    win.log("before")
    win._on_run_failed("boom")

    # Status bar / log stay exactly truthful (additive dialog, unchanged text).
    assert win.statusBar().currentMessage() == "Failed: boom"
    assert win.log_view.toPlainText() == "before\nRun failed: boom"
    # State restored (idle, buttons consistent).
    assert not win.is_running
    assert win.start_button.isEnabled()
    assert not win.stop_button.isEnabled()

    box = _assert_box(win, QMessageBox.Icon.Critical, "Run failed", "boom")
    assert win.error_box_count == 1
    assert box.parent() is win


def test_fake_backend_failure_reaches_terminal_without_blocking(qapp):
    """End-to-end witness: the failure box must not block controller cleanup."""
    backend = FailingBackend()
    win = MainWindow(backend_factory=lambda: backend)
    try:
        win.start_button.click()

        # The controller reaches FAILED and reaps its thread.  If the box were
        # shown with exec()/a static blocking call, this would never complete.
        assert _pump_until(
            qapp,
            lambda: win.is_running is False
            and win.controller.status is RunStatus.FAILED,
        )
        assert not win.controller.has_live_thread
        assert backend.run_calls == 1

        box = _assert_box(win, QMessageBox.Icon.Critical, "Run failed", "backend exploded")
        assert box.isVisible()
        assert win.error_box_count == 1
        assert win.start_button.isEnabled()
        assert not win.stop_button.isEnabled()
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# B. Boring terminal failure
# --------------------------------------------------------------------------
def test_boring_failed_shows_one_critical_box(window):
    win = window
    win._on_boring_failed("kaboom")

    assert win.statusBar().currentMessage() == "Boring stack failed: kaboom"
    assert "Boring stack failed: kaboom" in win.log_view.toPlainText()
    assert not win.is_running
    assert win.start_button.isEnabled()

    _assert_box(win, QMessageBox.Icon.Critical, "Boring stack failed", "kaboom")
    assert win.error_box_count == 1


# --------------------------------------------------------------------------
# C. preflight failure (multiple errors -> one Warning box)
# --------------------------------------------------------------------------
def test_preflight_failure_multiple_errors_one_warning_box(window):
    win = window
    errors = ["Input folder is empty.", "Output folder is empty."]
    win._report_preflight_failure("Cannot start real backend", errors)

    message = "Cannot start real backend: Input folder is empty.; Output folder is empty."
    assert win.statusBar().currentMessage() == message
    assert message in win.log_view.toPlainText()
    assert not win.is_running
    # Preflight never touches the controller: stays idle with no live thread.
    assert win.controller.status is RunStatus.IDLE
    assert not win.controller.has_live_thread

    _assert_box(win, QMessageBox.Icon.Warning, "Cannot start run", message)
    assert win.error_box_count == 1


def test_boring_preflight_failure_one_warning_box(window):
    win = window
    win._report_preflight_failure("Cannot start boring stack", ["Input folder is empty."])

    message = "Cannot start boring stack: Input folder is empty."
    assert win.statusBar().currentMessage() == message
    assert win.controller.status is RunStatus.IDLE
    _assert_box(win, QMessageBox.Icon.Warning, "Cannot start run", message)
    assert win.error_box_count == 1


# --------------------------------------------------------------------------
# D. structured startup refusal
# --------------------------------------------------------------------------
def test_run_refused_shows_one_warning_box_localized(window):
    win = window
    payload = StartupRefusalPayload(
        code="OUTPUT_STATE_INCOMPATIBLE",
        technical_detail="existing artifacts",
        semantic_data={"mode": "drizzle"},
    )
    win._on_run_refused(payload)

    title = localization.translate(
        "startup_refusal_output_state_incompatible_title", "en"
    )
    body = localization.translate(
        "startup_refusal_output_state_incompatible_body", "en"
    )
    assert win.statusBar().currentMessage() == title
    assert body in win.log_view.toPlainText()
    assert not win.is_running

    _assert_box(win, QMessageBox.Icon.Warning, title, body)
    assert win.error_box_count == 1


# --------------------------------------------------------------------------
# E. replacement / dedup ownership
# --------------------------------------------------------------------------
def test_second_failure_replaces_first_box(window):
    win = window
    win._on_run_failed("first")
    box1 = win.error_message_box
    assert box1 is not None and box1.text() == "first"
    assert win.error_box_count == 1

    win._on_run_failed("second")
    box2 = win.error_message_box
    # Same owned box reused (replaced), never a second stacked box.
    assert box2 is box1
    assert box2.text() == "second"
    assert box2.windowTitle() == "Run failed"
    assert box2.isVisible()
    assert win.error_box_count == 2


def test_cross_severity_replacement_updates_single_box(window):
    win = window
    win._on_run_failed("hard failure")
    box = win.error_message_box
    assert box.icon() is QMessageBox.Icon.Critical

    # A later preflight (user-correctable) replaces severity in the same box.
    win._report_preflight_failure("Cannot start real backend", ["Input folder is empty."])
    assert win.error_message_box is box
    assert box.icon() is QMessageBox.Icon.Warning
    assert win.error_box_count == 2


# --------------------------------------------------------------------------
# F. negative cases: success / empty / cancelled show no error box
# --------------------------------------------------------------------------
def test_success_empty_cancelled_show_no_error_box(window):
    win = window
    # Success path (summary dialog) — no error box.
    win._last_summary_payload = None
    win._on_run_finished()
    assert win.error_message_box is None
    assert win.error_box_count == 0

    # Empty/no-output — summary dialog still shown, still no error box.
    win._on_run_finished()
    assert win.error_message_box is None
    assert win.error_box_count == 0

    # Cancelled — no error box.
    win._on_run_cancelled()
    assert win.error_message_box is None
    assert win.error_box_count == 0


def test_success_still_shows_summary_dialog_not_error_box(window):
    from seestar.gui_qt.summary_payload import SummaryPayload

    win = window
    win._last_summary_payload = SummaryPayload(
        status="finished",
        duration_seconds=1.0,
        files_attempted=1,
        final_stack_file="/out/final.fits",
        final_stack_exists=True,
        images_in_final_stack=1,
        total_exposure_seconds=1.0,
        can_open_output=True,
    )
    win._on_run_finished()
    assert win._summary_dialog is not None
    assert win.error_message_box is None
    assert win.error_box_count == 0


# --------------------------------------------------------------------------
# G. FR/EN title parity + localization key completeness
# --------------------------------------------------------------------------
def test_error_box_title_keys_have_en_fr_parity():
    for key in (
        "error_box_run_failed_title",
        "error_box_boring_failed_title",
        "error_box_preflight_title",
    ):
        entry = localization.TRANSLATIONS[key]
        assert set(entry) == {"en", "fr"}, f"key {key!r} must have en+fr"
        assert entry["en"], f"key {key!r} has empty en"
        assert entry["fr"], f"key {key!r} has empty fr"


def test_error_box_titles_translate_fr(window):
    win = window
    win.language_combo.setCurrentText("Français")

    win._on_run_failed("boom")
    assert win.error_message_box.windowTitle() == "Échec de l'exécution"

    win._on_boring_failed("kaboom")
    assert win.error_message_box.windowTitle() == "Échec de l'empilement Boring"

    win._report_preflight_failure("Cannot start real backend", ["Input folder is empty."])
    assert win.error_message_box.windowTitle() == "Impossible de démarrer"


# --------------------------------------------------------------------------
# H. shutdown with a visible box is clean / idempotent
# --------------------------------------------------------------------------
def test_shutdown_with_visible_box_is_clean_and_idempotent(qapp):
    win = MainWindow()
    win._on_run_failed("boom")
    box = win.error_message_box
    assert box is not None and box.isVisible()

    assert win.shutdown() is True
    assert not box.isVisible(), "error box must be closed on shutdown"
    # Idempotent teardown.
    assert win.shutdown() is True
