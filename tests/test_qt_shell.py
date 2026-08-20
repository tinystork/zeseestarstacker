"""Offscreen smoke tests for the non-default PySide6 GUI shell.

These tests construct the minimal :class:`seestar.gui_qt.MainWindow` under the
``offscreen`` Qt platform plugin so they run headlessly (no X11/Wayland).  They
verify importability and basic construction only — no real stacking, no engine,
no worker threads.

``QT_QPA_PLATFORM=offscreen`` must be set before any ``QApplication`` is
created; we set it defensively here so the suite also passes when invoked
without the env var.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QTabWidget

import seestar.gui_qt as gui_qt
from seestar.gui_qt import DEFAULT_TITLE, MainWindow, create_application
from seestar.gui_qt.main_window import (
    TAB_LOG,
    TAB_PREVIEW,
    TAB_SETTINGS,
    TAB_STACK,
)


@pytest.fixture(scope="session")
def qapp():
    """Single process-wide QApplication for the whole session."""
    app = create_application([])
    assert app is QApplication.instance()
    return app


@pytest.fixture()
def window(qapp):
    win = MainWindow()
    yield win
    win.shutdown()


# --------------------------------------------------------------------------
# Import / namespace invariants
# --------------------------------------------------------------------------
def test_gui_qt_public_surface():
    assert hasattr(gui_qt, "MainWindow")
    assert hasattr(gui_qt, "create_application")
    assert hasattr(gui_qt, "run_qt_app")
    assert gui_qt.DEFAULT_TITLE == DEFAULT_TITLE


def test_tk_gui_still_imports():
    """Invariant: adding the Qt shell must not break the Tk GUI import."""
    import seestar.gui as tk_gui

    assert tk_gui.SeestarStackerGUI is not None


def test_shell_does_not_import_engine():
    """Structural invariant: gui_qt source files never import the engine/Tk."""
    from pathlib import Path

    pkg_dir = Path(gui_qt.__file__).resolve().parent
    forbidden = (
        "seestar.core",
        "seestar.alignment",
        "seestar.enhancement",
        "seestar.queuep",
        "tkinter",
    )
    for py in pkg_dir.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{py.name} references {token}"


# --------------------------------------------------------------------------
# Construction / layout
# --------------------------------------------------------------------------
def test_main_window_constructs_offscreen(qapp):
    win = MainWindow()
    try:
        assert win.windowTitle() == DEFAULT_TITLE
        assert isinstance(win.centralWidget(), QTabWidget)
        assert win.tabs.count() >= 4
    finally:
        win.shutdown()


def test_tab_labels():
    win = MainWindow()
    try:
        labels = [win.tabs.tabText(i) for i in range(win.tabs.count())]
        assert TAB_STACK in labels
        assert TAB_SETTINGS in labels
        assert TAB_PREVIEW in labels
        assert TAB_LOG in labels
    finally:
        win.shutdown()


def test_controls_and_progress_exist():
    win = MainWindow()
    try:
        assert win.start_button is not None
        assert win.stop_button is not None
        assert win.progress is not None
        assert win.progress.minimum() == 0
        assert win.progress.maximum() == 100
        assert win.statusBar() is not None
        assert win.log_view.isReadOnly()
    finally:
        win.shutdown()


def test_custom_title():
    win = MainWindow(title="Custom Qt Shell")
    try:
        assert win.windowTitle() == "Custom Qt Shell"
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# Start / stop inert controls
# --------------------------------------------------------------------------
def test_start_stop_toggles_state():
    win = MainWindow()
    try:
        assert win.is_running is False
        assert win.start_button.isEnabled()
        assert not win.stop_button.isEnabled()

        win.start_button.click()
        assert win.is_running is True
        assert not win.start_button.isEnabled()
        assert win.stop_button.isEnabled()

        win.stop_button.click()
        assert win.is_running is False
        assert win.start_button.isEnabled()
        assert not win.stop_button.isEnabled()
    finally:
        win.shutdown()


def test_start_stop_signals_fire():
    win = MainWindow()
    events = []
    win.started.connect(lambda: events.append("started"))
    win.stopped.connect(lambda: events.append("stopped"))
    try:
        win.start_button.click()
        win.stop_button.click()
        assert events == ["started", "stopped"]
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# Shutdown hook
# --------------------------------------------------------------------------
def test_clean_shutdown_hook_is_idempotent():
    win = MainWindow()
    assert win.shutdown_called is False
    win.shutdown()
    assert win.shutdown_called is True
    # Calling again must not raise or reset state.
    win.shutdown()
    assert win.shutdown_called is True


def test_close_event_runs_shutdown_hook():
    win = MainWindow()
    win.start_button.click()  # leave it in "running" state
    assert win.is_running is True

    win.close()
    assert win.shutdown_called is True
    assert win.is_running is False
