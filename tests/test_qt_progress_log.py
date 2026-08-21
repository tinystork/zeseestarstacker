"""M6 progress/log ergonomics tests: Copy Log + elapsed/remaining time surface.

Offscreen tests for the two M6 deliverables:

* a ``Copy Log`` button that copies the full plain-text log to the clipboard
  (disabled while the log is empty, armed on first log line, never mutating the
  log or run state), and
* visible elapsed / remaining labels driven only by the existing progress
  lifecycle signals, using a deterministic injected clock so no test sleeps.

Also covers the boring (single-batch CSV) route: its log messages feed the same
shared ``log()`` surface and its time labels stay honest (elapsed visible,
remaining unknown throughout, done/failed/cancelled at the terminal state).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt.boring_runner import BoringRunnerBase
from seestar.gui_qt.progress_time import (
    UNKNOWN,
    estimate_remaining_seconds,
    format_duration,
)


@pytest.fixture(scope="session", autouse=True)
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


class FakeClock:
    """Deterministic monotonic-clock stand-in for the time surface."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeBoringRunner(BoringRunnerBase):
    """Minimal boring runner that records nothing and emits terminal signals."""

    def start(self, request) -> None:  # noqa: D401
        self.started.emit()

    def cancel(self) -> None:
        pass

    def is_running(self) -> bool:
        return False


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------
def test_format_duration():
    assert format_duration(0.0) == "0:00"
    assert format_duration(30) == "0:30"
    assert format_duration(90) == "1:30"
    assert format_duration(3661) == "1:01:01"
    assert format_duration(None) == UNKNOWN
    assert format_duration(-1) == UNKNOWN


def test_estimate_remaining_seconds():
    assert estimate_remaining_seconds(30, 50) == pytest.approx(30.0)
    assert estimate_remaining_seconds(60, 75) == pytest.approx(20.0)
    assert estimate_remaining_seconds(30, 25) == pytest.approx(90.0)
    # No divide-by-zero at 0% and no bogus estimate at >= 100%.
    assert estimate_remaining_seconds(30, 0) is None
    assert estimate_remaining_seconds(30, 100) is None
    # Unknown / invalid inputs.
    assert estimate_remaining_seconds(None, 50) is None
    assert estimate_remaining_seconds(-5, 50) is None
    assert estimate_remaining_seconds(30, None) is None
    assert estimate_remaining_seconds(30, "nope") is None


# --------------------------------------------------------------------------
# Copy Log
# --------------------------------------------------------------------------
def test_copy_log_button_initial_disabled_and_empty_log():
    win = MainWindow()
    try:
        assert not win.copy_log_button.isEnabled()
        assert win.log_view.toPlainText() == ""
    finally:
        win.shutdown()


def test_copy_log_enabled_after_log_line_and_copies_exact_text():
    win = MainWindow()
    try:
        win.log("alpha")
        win.log("beta")
        assert win.copy_log_button.isEnabled()

        win.copy_log_button.click()
        assert QApplication.clipboard().text() == "alpha\nbeta"
    finally:
        win.shutdown()


def test_copy_log_does_not_mutate_log_or_run_state():
    win = MainWindow()
    try:
        win.log("hello world")
        before = win.log_view.toPlainText()
        running_before = win.is_running

        win.copy_log_button.click()

        assert win.log_view.toPlainText() == before
        assert win.is_running is running_before
        assert QApplication.clipboard().text() == "hello world"
    finally:
        win.shutdown()


def test_copy_log_stays_enabled_after_run_finishes():
    win = MainWindow()
    try:
        win.log("line before run")
        win._on_run_started()
        win._on_run_finished()

        assert win.copy_log_button.isEnabled()
        assert win.log_view.toPlainText() == "line before run\nRun finished."
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# Time surface (deterministic injected clock)
# --------------------------------------------------------------------------
def test_initial_time_labels():
    win = MainWindow()
    try:
        assert win.elapsed_label.text() == "Elapsed: 0:00"
        assert win.remaining_label.text() == f"Remaining: {UNKNOWN}"
    finally:
        win.shutdown()


def test_run_start_resets_time_labels():
    clock = FakeClock(5000.0)
    win = MainWindow(clock=clock)
    try:
        # A previous value must not leak into the new run.
        win._on_run_started()
        assert win.elapsed_label.text() == "Elapsed: 0:00"
        assert win.remaining_label.text() == f"Remaining: {UNKNOWN}"
    finally:
        win.shutdown()


def test_progress_updates_elapsed_and_remaining():
    clock = FakeClock(1000.0)
    win = MainWindow(clock=clock)
    try:
        win._on_run_started()
        clock.now = 1030.0  # 30 s elapsed
        win._on_progress(50)
        assert win.elapsed_label.text() == "Elapsed: 0:30"
        assert win.remaining_label.text() == "Remaining: 0:30"

        clock.now = 1060.0  # 60 s elapsed
        win._on_progress(75)  # remaining = 60 * 25 / 75 = 20 s
        assert win.elapsed_label.text() == "Elapsed: 1:00"
        assert win.remaining_label.text() == "Remaining: 0:20"
    finally:
        win.shutdown()


def test_progress_zero_percent_no_divide_by_zero():
    clock = FakeClock(1000.0)
    win = MainWindow(clock=clock)
    try:
        win._on_run_started()
        clock.now = 1010.0
        win._on_progress(0)
        assert win.elapsed_label.text() == "Elapsed: 0:10"
        assert win.remaining_label.text() == f"Remaining: {UNKNOWN}"
    finally:
        win.shutdown()


def test_progress_100_shows_done():
    clock = FakeClock(1000.0)
    win = MainWindow(clock=clock)
    try:
        win._on_run_started()
        clock.now = 1045.0
        win._on_progress(100)
        assert win.elapsed_label.text() == "Elapsed: 0:45"
        assert win.remaining_label.text() == "Remaining: 0:00"
    finally:
        win.shutdown()


def test_finish_sets_remaining_done_and_keeps_copy_enabled():
    clock = FakeClock(1000.0)
    win = MainWindow(clock=clock)
    try:
        win.log("hi")
        win._on_run_started()
        clock.now = 1090.0
        win._on_run_finished()

        assert win.elapsed_label.text() == "Elapsed: 1:30"
        assert win.remaining_label.text() == "Remaining: 0:00"
        assert win.copy_log_button.isEnabled()
    finally:
        win.shutdown()


def test_failure_leaves_honest_remaining():
    clock = FakeClock(1000.0)
    win = MainWindow(clock=clock)
    try:
        win._on_run_started()
        clock.now = 1030.0
        win._on_run_failed("boom")

        assert win.elapsed_label.text() == "Elapsed: 0:30"
        assert win.remaining_label.text() == "Remaining: failed"
    finally:
        win.shutdown()


def test_cancel_leaves_honest_remaining():
    clock = FakeClock(1000.0)
    win = MainWindow(clock=clock)
    try:
        win._on_run_started()
        clock.now = 1020.0
        win._on_run_cancelled()

        assert win.elapsed_label.text() == "Elapsed: 0:20"
        assert win.remaining_label.text() == "Remaining: cancelled"
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# Boring route: shared log surface + honest time labels
# --------------------------------------------------------------------------
def test_boring_time_surface_and_log_feed():
    clock = FakeClock(2000.0)
    fake = FakeBoringRunner()
    win = MainWindow(boring_runner_factory=lambda: fake, clock=clock)
    try:
        runner = win._resolve_boring_runner()
        assert runner is fake

        win._on_boring_started()
        assert win.elapsed_label.text() == "Elapsed: 0:00"
        assert win.remaining_label.text() == f"Remaining: {UNKNOWN}"

        # Boring stdout lines flow through the shared log() surface and arm the
        # same Copy Log button.
        fake.log_message.emit("boring stdout line")
        assert "boring stdout line" in win.log_view.toPlainText()
        assert win.copy_log_button.isEnabled()

        clock.now = 2075.0
        win._on_boring_finished(0)
        assert win.elapsed_label.text() == "Elapsed: 1:15"
        assert win.remaining_label.text() == "Remaining: 0:00"
    finally:
        win.shutdown()


def test_boring_failed_and_cancelled_stay_honest():
    clock = FakeClock(3000.0)
    win = MainWindow(boring_runner_factory=FakeBoringRunner, clock=clock)
    try:
        win._on_boring_started()
        clock.now = 3060.0
        win._on_boring_failed("kaboom")
        assert win.elapsed_label.text() == "Elapsed: 1:00"
        assert win.remaining_label.text() == "Remaining: failed"

        win._on_boring_started()
        clock.now = 3090.0
        win._on_boring_cancelled()
        assert win.elapsed_label.text() == "Elapsed: 0:30"
        assert win.remaining_label.text() == "Remaining: cancelled"
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# Source / import hygiene for the new helper module
# --------------------------------------------------------------------------
def test_progress_time_source_is_tk_engine_free():
    from pathlib import Path

    from seestar import gui_qt

    pkg_dir = Path(gui_qt.__file__).resolve().parent
    text = (pkg_dir / "progress_time.py").read_text(encoding="utf-8")
    forbidden = (
        "seestar.core",
        "seestar.alignment",
        "seestar.enhancement",
        "seestar.queuep",
        "tkinter",
        "seestar.gui.settings",
        "seestar.gui.main_window",
        "seestar.gui.boring_stack",
        "zesolver_adapter",
        "zesolver.api",
        "zealfie",
        "QtWidgets",
        "QtGui",
        "QtCore",
    )
    for token in forbidden:
        assert token not in text, f"progress_time.py references {token}"
