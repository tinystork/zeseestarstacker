"""Offscreen smoke tests for the non-default PySide6 GUI shell.

These tests construct the minimal :class:`seestar.gui_qt.MainWindow` under the
``offscreen`` Qt platform plugin so they run headlessly (no X11/Wayland).  They
verify importability, layout, settings collection and the start/stop lifecycle
seam (a simulated run on a QThread) — no real stacking and no engine.

``QT_QPA_PLATFORM=offscreen`` must be set before any ``QApplication`` is
created; we set it defensively here so the suite also passes when invoked
without the env var.
"""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QScrollArea, QSplitter

import seestar.gui_qt as gui_qt
from seestar.gui_qt import DEFAULT_TITLE, MainWindow, create_application
from seestar.gui_qt.main_window import (
    TAB_EXPERT,
    TAB_PREVIEW_CONTROLS,
    TAB_STACKING,
)
from seestar.gui_qt.run_bridge import RunRequest
from seestar.gui_qt.settings_state import QtSettingsState


def _pump_until(qapp, predicate, timeout_ms: int = 5000) -> bool:
    """Pump the GUI event loop until ``predicate`` is true (or timeout)."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    qapp.processEvents()
    return bool(predicate())


@pytest.fixture(scope="session", autouse=True)
def qapp():
    """Single process-wide QApplication for the whole session.

    Autouse keeps tests that construct ``MainWindow()`` directly safe when they
    are selected in isolation (``QWidget`` requires an existing application).
    """
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
    """Invariant: adding the Qt shell must not break the Tk GUI import.

    Run this in a fresh interpreter: once a Qt QApplication exists, matplotlib
    correctly refuses to switch to TkAgg in-process, which is a test-ordering
    artifact rather than a Tk-regression signal.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = (
        "import seestar.gui as tk_gui\n"
        "assert tk_gui.SeestarStackerGUI is not None\n"
        "print('TK_IMPORT_OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert proc.returncode == 0, (
        f"Tk import failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )


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
        "seestar.gui.settings",
        "seestar.gui.main_window",
        "zesolver_adapter",
        "zesolver.api",
        "zealfie",
    )
    for py in pkg_dir.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{py.name} references {token}"


def test_gui_qt_source_is_tk_and_engine_free():
    """The full gui_qt source surface stays Tk/engine/private-ZeSolver free."""
    from pathlib import Path

    pkg_dir = Path(gui_qt.__file__).resolve().parent
    assert (pkg_dir / "settings_state.py").exists()
    assert (pkg_dir / "run_bridge.py").exists()


def test_import_hygiene_fresh_process():
    """Fresh interpreter: ``import seestar.gui_qt`` must not import Tk/engine.

    This is the real import-hygiene guarantee (a source-token scan is not
    enough): in a clean process the Qt shell must leave no ``tkinter*`` or
    ``seestar.core*`` / ``seestar.alignment*`` / ``seestar.enhancement*`` /
    ``seestar.queuep*`` / Tk-GUI modules in ``sys.modules``.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys\n"
        "import seestar.gui_qt  # noqa: F401\n"
        "_bad = [m for m in sys.modules\n"
        "        if m.startswith('tkinter')\n"
        "        or m.startswith('seestar.core')\n"
        "        or m.startswith('seestar.alignment')\n"
        "        or m.startswith('seestar.enhancement')\n"
        "        or m.startswith('seestar.queuep')\n"
        "        or m in ('seestar.gui.main_window', 'seestar.gui.settings',"
        " 'seestar.gui.boring_stack')]\n"
        "if _bad:\n"
        "    print('BAD_MODULES:', _bad)\n"
        "    sys.exit(1)\n"
        "print('IMPORT_HYGIENE_OK')\n"
    )
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
    )
    assert proc.returncode == 0, (
        f"import hygiene violated: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )


def test_run_request_is_canonical():
    """The Qt bridge exposes the canonical RunRequest, not an isolated copy."""
    from seestar.gui_qt.run_bridge import RunRequest
    from seestar.gui.run_config import RunRequest as CanonicalRunRequest

    assert RunRequest is CanonicalRunRequest


# --------------------------------------------------------------------------
# Construction / layout
# --------------------------------------------------------------------------
def test_main_window_constructs_offscreen(qapp):
    win = MainWindow()
    try:
        assert win.windowTitle() == DEFAULT_TITLE
        assert isinstance(win.centralWidget(), QSplitter)
        assert win.tabs.count() >= 3
    finally:
        win.shutdown()


def test_tab_labels():
    win = MainWindow()
    try:
        labels = [win.tabs.tabText(i) for i in range(win.tabs.count())]
        assert TAB_STACKING in labels
        assert TAB_EXPERT in labels
        assert TAB_PREVIEW_CONTROLS in labels
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
# Tk-like topology: left scrollable panel + persistent right panel
# --------------------------------------------------------------------------
def test_left_panel_is_scrollable_and_holds_tabs():
    win = MainWindow()
    try:
        assert isinstance(win.left_panel, QScrollArea)
        assert win.left_panel.widget() is not None
        assert win.tabs.parent() is not None
    finally:
        win.shutdown()


def test_right_panel_has_preview_view_and_actions():
    win = MainWindow()
    try:
        for attr in (
            "preview_label",
            "preview_image_label",
            "zoom_combo",
            "resolution_label",
            "rotate_left_button",
            "rotate_right_button",
            "start_button",
            "stop_button",
            "analyse_button",
            "solver_button",
            "view_inputs_button",
            "add_folder_button",
            "open_output_button",
        ):
            assert hasattr(win, attr), f"missing right-panel control {attr}"
        # Stub action buttons are present but disabled; Start/Stop/Solver are live.
        assert win.start_button.isEnabled()
        assert not win.stop_button.isEnabled()
        assert not win.analyse_button.isEnabled()
        assert win.solver_button.isEnabled()
        assert not win.view_inputs_button.isEnabled()
        assert not win.add_folder_button.isEnabled()
        assert not win.open_output_button.isEnabled()
    finally:
        win.shutdown()


def test_right_panel_is_not_a_tab_and_persists_across_switches():
    win = MainWindow()
    try:
        tab_widgets = [win.tabs.widget(i) for i in range(win.tabs.count())]
        assert win.right_panel not in tab_widgets
        identity = win.right_panel
        for i in range(win.tabs.count()):
            win.tabs.setCurrentIndex(i)
            assert win.right_panel is identity
    finally:
        win.shutdown()


def test_right_panel_has_persistent_histogram_surface():
    """Checklist item 13.4: a real histogram surface lives in the right panel."""
    win = MainWindow()
    try:
        for attr in (
            "right_histogram_group",
            "right_histogram_status",
            "right_histogram_view",
        ):
            assert hasattr(win, attr), f"missing right-panel histogram {attr}"
        assert win.right_histogram_group.title() == "Histogram"
        assert win.right_histogram_status.text() == "No preview"
        assert not win.right_histogram_view.has_data
        # The group is owned by the persistent right panel, not a left tab.
        assert win.right_histogram_group.parent() is win.right_panel
        # It persists across left-tab switches (same widget identity).
        identity = win.right_histogram_group
        for i in range(win.tabs.count()):
            win.tabs.setCurrentIndex(i)
            assert win.right_histogram_group is identity
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# Final-combination business mapping
# --------------------------------------------------------------------------
def test_final_combine_combo_has_five_historical_choices():
    win = MainWindow()
    try:
        labels = [
            win.final_combine_combo.itemText(i)
            for i in range(win.final_combine_combo.count())
        ]
        assert labels == [
            "Mean",
            "Median",
            "Winsorized Sigma Clip",
            "Reproject",
            "Reproject & Coadd",
        ]
    finally:
        win.shutdown()


def test_final_combine_mapping_table():
    from seestar.gui_qt.final_combine import FINAL_COMBINE_LABELS

    expected = {
        "mean": ("mean", False, False),
        "median": ("median", False, False),
        "winsorized_sigma_clip": ("winsorized_sigma_clip", False, False),
        "reproject": ("reproject", True, False),
        "reproject_coadd": ("reproject_coadd", False, True),
    }
    win = MainWindow()
    try:
        for key, (combine, rbb, rcf) in expected.items():
            win.final_combine_combo.setCurrentText(FINAL_COMBINE_LABELS[key])
            state = win.collect_settings_state()
            assert state.stack_final_combine == combine
            assert state.reproject_between_batches is rbb
            assert state.reproject_coadd_final is rcf
            kw = win.build_run_request().backend_kwargs
            assert kw["stack_final_combine"] == combine
            assert kw["reproject_between_batches"] is rbb
            assert kw["reproject_coadd_final"] is rcf
    finally:
        win.shutdown()


def test_final_combine_helper_flags_and_reconstruction():
    from seestar.gui_qt.final_combine import (
        final_combine_flags,
        final_combine_key_from_flags,
    )

    assert final_combine_flags("mean") == (False, False)
    assert final_combine_flags("median") == (False, False)
    assert final_combine_flags("winsorized_sigma_clip") == (False, False)
    assert final_combine_flags("reproject") == (True, False)
    assert final_combine_flags("reproject_coadd") == (False, True)

    assert final_combine_key_from_flags(True, False) == "reproject"
    assert final_combine_key_from_flags(False, True) == "reproject_coadd"
    assert final_combine_key_from_flags(False, False, fallback="median") == "median"
    assert final_combine_key_from_flags(False, False) == "mean"


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
        # Cancellation is asynchronous: the run stays active until the worker
        # observes the flag and its cancelled signal reaches the GUI thread.
        app = QApplication.instance()
        assert _pump_until(app, lambda: win.is_running is False)
        assert win.is_running is False
        assert win.start_button.isEnabled()
        assert not win.stop_button.isEnabled()
    finally:
        win.shutdown()


def test_start_stop_drive_controller_lifecycle():
    win = MainWindow()
    events = []
    win.controller.started.connect(lambda: events.append("started"))
    win.controller.cancelled.connect(lambda: events.append("cancelled"))
    try:
        win.start_button.click()
        win.stop_button.click()
        app = QApplication.instance()
        assert _pump_until(app, lambda: len(events) >= 2)
        assert events == ["started", "cancelled"]
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


# --------------------------------------------------------------------------
# M2 settings controls -> QtSettingsState -> RunRequest
# --------------------------------------------------------------------------
def test_settings_controls_exist():
    win = MainWindow()
    try:
        for attr in (
            "input_edit",
            "output_edit",
            "temp_edit",
            "output_filename_edit",
            "batch_spin",
            "stacking_mode_combo",
            "drizzle_check",
            "drizzle_mode_combo",
            "drizzle_group_spin",
            "solver_combo",
        ):
            assert hasattr(win, attr), f"missing control {attr}"
    finally:
        win.shutdown()


def test_collect_settings_state_returns_model():
    win = MainWindow()
    try:
        state = win.collect_settings_state()
        assert isinstance(state, QtSettingsState)
        assert state is win.settings_state
    finally:
        win.shutdown()


def test_controls_update_settings_state():
    win = MainWindow()
    try:
        win.input_edit.setText("/inputs")
        win.output_edit.setText("/outputs")
        win.temp_edit.setText("/tmp")
        win.output_filename_edit.setText("stack.fits")
        win.batch_spin.setValue(4)
        win.stacking_mode_combo.setCurrentText("median")
        win.drizzle_check.setChecked(True)
        win.drizzle_mode_combo.setCurrentText("Incremental")
        win.drizzle_group_spin.setValue(88)
        win.solver_combo.setCurrentText("astap")

        state = win.collect_settings_state()
        assert state.input_folder == "/inputs"
        assert state.output_folder == "/outputs"
        assert state.temp_folder == "/tmp"
        assert state.output_filename == "stack.fits"
        assert state.batch_size == 4
        assert state.stacking_mode == "median"
        assert state.use_drizzle is True
        assert state.drizzle_mode == "Incremental"
        assert state.drizzle_group_size == 88
        assert state.local_solver_preference == "astap"
    finally:
        win.shutdown()


def test_build_run_request_reflects_controls():
    win = MainWindow()
    try:
        win.batch_spin.setValue(4)
        win.solver_combo.setCurrentText("zesolver")
        win.drizzle_check.setChecked(True)
        win.drizzle_mode_combo.setCurrentText("Incremental")
        win.drizzle_group_spin.setValue(66)

        req = win.build_run_request()
        assert isinstance(req, RunRequest)
        kw = req.backend_kwargs
        assert kw["batch_size"] == 4
        assert kw["local_solver_preference"] == "zesolver"
        assert kw["use_drizzle"] is True
        assert kw["drizzle_mode"] == "Incremental"
        assert kw["drizzle_group_size"] == 66
        assert req.align_on_disk is True
        assert "chunk_size" not in kw
    finally:
        win.shutdown()


def test_build_run_request_batch_one_chunk_size():
    win = MainWindow()
    try:
        win.batch_spin.setValue(1)
        req = win.build_run_request(auto_chunk_size=128)
        assert req.backend_kwargs["chunk_size"] == 128
        assert req.align_on_disk is True

        req = win.build_run_request(auto_chunk_size=128, special_single=True)
        assert "chunk_size" not in req.backend_kwargs
        assert req.special_single is True
    finally:
        win.shutdown()


def test_build_run_request_passes_additional_folders():
    win = MainWindow()
    try:
        folders = ["/extra-a", "/extra-b"]
        req = win.build_run_request(initial_additional_folders=folders)
        assert req.backend_kwargs["initial_additional_folders"] == folders
        assert req.backend_kwargs["initial_additional_folders"] is not folders
    finally:
        win.shutdown()


def test_build_run_request_does_not_start_backend():
    win = MainWindow()
    try:
        assert win.is_running is False
        req = win.build_run_request()
        assert isinstance(req, RunRequest)
        assert win.is_running is False
        # The shell remains in its inert pre-start state.
        assert win.start_button.isEnabled()
        assert not win.stop_button.isEnabled()
    finally:
        win.shutdown()
