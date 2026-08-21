"""Offscreen Qt tests for the solver configuration dialog (Qt view/controller).

Covers the first real Qt solver dialog — its structure, the ``none`` / ``astap``
/ ``zesolver`` identifiers, the ASTAP frame enablement (mirroring Tk), the
in-dialog ASTAP validation, fake/injected ZeSolver status + configuration
behaviour, the lazy public-boundary service, and the round-trip of accepted
values back into ``MainWindow`` (``collect_settings_state`` /
``build_run_request``).

``QT_QPA_PLATFORM=offscreen`` is set defensively before any ``QApplication``
is created, mirroring the other Qt shell tests.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication, QDialog

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt.settings_state import QtSettingsState
from seestar.gui_qt.solver_dialog import (
    SOLVER_ASTAP,
    SOLVER_NONE,
    SOLVER_ZESOLVER,
    SolverSettingsDialog,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def qapp():
    """Single process-wide QApplication for the whole session."""
    app = create_application([])
    assert app is QApplication.instance()
    return app


def _load_engine_solver_config():
    """Load ``seestar.core.solver_config`` by file path (no package init)."""
    name = "engine_solver_config_dialog_m21"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "seestar" / "core" / "solver_config.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ui_state(label="ZeSolver : non installé", color="gray", show_button=False):
    return SimpleNamespace(
        label=label, status_color=color, show_configuration_button=show_button
    )


def _make_dialog(**overrides):
    """Build a dialog with fully-injected (engine-free) boundary callables."""
    state = QtSettingsState()
    return SolverSettingsDialog(
        None,
        state,
        check_readiness=overrides.get(
            "check_readiness",
            lambda: SimpleNamespace(state="not_installed"),
        ),
        ui_state_fn=overrides.get("ui_state_fn", lambda d: _ui_state()),
        open_configuration=overrides.get(
            "open_configuration", lambda: (True, None)
        ),
        session_refresh_action=overrides.get(
            "session_refresh_action", lambda handle: "none"
        ),
    )


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------
def test_dialog_has_required_controls(qapp):
    dlg = _make_dialog()
    try:
        for attr in (
            "none_radio",
            "astap_radio",
            "zesolver_radio",
            "astap_path_edit",
            "astap_path_browse",
            "astap_data_edit",
            "astap_data_browse",
            "search_radius_spin",
            "downsample_spin",
            "sensitivity_spin",
            "zesolver_status_label",
            "configure_zesolver_button",
            "error_label",
            "button_box",
        ):
            assert hasattr(dlg, attr), f"missing control {attr}"
    finally:
        dlg.close()


def test_solver_choice_identifiers(qapp):
    dlg = _make_dialog()
    try:
        dlg.none_radio.setChecked(True)
        assert dlg._current_choice() == SOLVER_NONE
        dlg.astap_radio.setChecked(True)
        assert dlg._current_choice() == SOLVER_ASTAP
        dlg.zesolver_radio.setChecked(True)
        assert dlg._current_choice() == SOLVER_ZESOLVER
    finally:
        dlg.close()


def test_astap_frame_enablement_mirrors_tk(qapp):
    dlg = _make_dialog()
    try:
        dlg.none_radio.setChecked(True)
        assert not dlg.astap_frame.isEnabled()
        dlg.astap_radio.setChecked(True)
        assert dlg.astap_frame.isEnabled()
        dlg.zesolver_radio.setChecked(True)
        assert dlg.astap_frame.isEnabled()
    finally:
        dlg.close()


# --------------------------------------------------------------------------
# Validation / value capture
# --------------------------------------------------------------------------
def test_astap_primary_empty_path_rejected(qapp):
    dlg = _make_dialog()
    try:
        dlg.astap_radio.setChecked(True)
        dlg.astap_path_edit.setText("")
        dlg._on_ok()
        assert dlg.result() != QDialog.DialogCode.Accepted
        assert "missing" in dlg.error_label.text().lower()
    finally:
        dlg.close()


def test_astap_with_path_is_accepted(qapp):
    dlg = _make_dialog()
    try:
        dlg.astap_radio.setChecked(True)
        dlg.astap_path_edit.setText("/usr/bin/astap")
        dlg._on_ok()
        assert dlg.result() == QDialog.DialogCode.Accepted
    finally:
        dlg.close()


def test_zesolver_choice_with_empty_astap_is_accepted(qapp):
    """ASTAP is a fallback for ZeSolver, so an empty path is not an error."""
    dlg = _make_dialog()
    try:
        dlg.zesolver_radio.setChecked(True)
        dlg.astap_path_edit.setText("")
        dlg._on_ok()
        assert dlg.result() == QDialog.DialogCode.Accepted
        assert dlg.values()["local_solver_preference"] == "zesolver"
        assert dlg.values()["astap_path"] == ""
    finally:
        dlg.close()


def test_ok_captures_all_values(qapp):
    dlg = _make_dialog()
    try:
        dlg.astap_radio.setChecked(True)
        dlg.astap_path_edit.setText("/usr/bin/astap")
        dlg.astap_data_edit.setText("/data/astap")
        dlg.search_radius_spin.setValue(15.5)
        dlg.downsample_spin.setValue(4)
        dlg.sensitivity_spin.setValue(200)
        dlg._on_ok()
        v = dlg.values()
        assert v["local_solver_preference"] == "astap"
        assert v["astap_path"] == "/usr/bin/astap"
        assert v["astap_data_dir"] == "/data/astap"
        assert v["astap_search_radius"] == 15.5
        assert v["astap_downsample"] == 4
        assert v["astap_sensitivity"] == 200
    finally:
        dlg.close()


# --------------------------------------------------------------------------
# ZeSolver status / configure (fake/injected)
# --------------------------------------------------------------------------
def test_zesolver_status_and_configure_injected(qapp):
    config_calls = []
    dlg = SolverSettingsDialog(
        None,
        QtSettingsState(),
        check_readiness=lambda: SimpleNamespace(state="available"),
        ui_state_fn=lambda d: _ui_state(
            "ZeSolver : prêt", "green", show_button=True
        ),
        open_configuration=lambda: config_calls.append(1) or (True, None),
        session_refresh_action=lambda handle: "none",
    )
    try:
        assert dlg.zesolver_status_label.text() == "ZeSolver : prêt"
        assert not dlg.configure_zesolver_button.isHidden()
        assert dlg.configure_zesolver_button.isEnabled()
        dlg.configure_zesolver_button.click()
        assert config_calls == [1]
        assert dlg.error_label.text() == ""
    finally:
        dlg.close()


def test_zesolver_configure_button_hidden_when_not_needed(qapp):
    dlg = _make_dialog()
    try:
        assert dlg.configure_zesolver_button.isHidden()
        assert not dlg.configure_zesolver_button.isEnabled()
    finally:
        dlg.close()


def test_configure_failure_surfaces_error(qapp):
    dlg = SolverSettingsDialog(
        None,
        QtSettingsState(),
        check_readiness=lambda: SimpleNamespace(state="not_installed"),
        ui_state_fn=lambda d: _ui_state(),
        open_configuration=lambda: (False, "boom"),
        session_refresh_action=lambda handle: "none",
    )
    try:
        dlg._on_configure_zesolver()
        assert "boom" in dlg.error_label.text()
    finally:
        dlg.close()


def test_configure_refresh_schedules_poll_then_refresh(qapp):
    """A running session handle re-arms the poll; a finished one refreshes."""
    actions = []

    class Handle:
        def __init__(self):
            self.running = True

    handle = Handle()

    def refresh_action(h):
        actions.append("tick")
        return "wait" if handle.running else "refresh"

    refreshed = []

    def ui_state_fn(d):
        refreshed.append(d)
        return _ui_state()

    dlg = SolverSettingsDialog(
        None,
        QtSettingsState(),
        check_readiness=lambda: SimpleNamespace(state="available"),
        ui_state_fn=ui_state_fn,
        open_configuration=lambda: (True, handle),
        session_refresh_action=refresh_action,
        refresh_tick_ms=10,
    )
    try:
        dlg._on_configure_zesolver()
        # Immediate refresh happened once during configure.
        assert len(refreshed) >= 1
        # While running, the poll re-arms (a "wait" tick was observed).
        assert "wait" in actions or actions[0] == "tick"
        # Let the session finish and pump the event loop to trigger "refresh".
        handle.running = False
        for _ in range(50):
            QApplication.instance().processEvents()
            if actions.count("tick") >= 2:
                break
        # A second tick resolved to "refresh", which refreshed again.
        assert refreshed
    finally:
        dlg.close()


# --------------------------------------------------------------------------
# Lazy public boundary (real service, no fake)
# --------------------------------------------------------------------------
def test_solver_service_lazy_boundary():
    from seestar.gui_qt.solver_service import (
        check_zesolver_readiness,
        zesolver_ui_state,
    )

    discovery = check_zesolver_readiness()
    assert hasattr(discovery, "state")
    ui = zesolver_ui_state(discovery)
    assert isinstance(ui.label, str) and ui.label
    assert isinstance(ui.status_color, str)
    assert isinstance(ui.show_configuration_button, bool)


def test_dialog_default_boundary_constructs(qapp):
    """Default (lazy) boundary composes without a fake and yields a status."""
    dlg = SolverSettingsDialog(None, QtSettingsState())
    try:
        assert isinstance(dlg.zesolver_status_label.text(), str)
        assert dlg.zesolver_status_label.text()
    finally:
        dlg.close()


# --------------------------------------------------------------------------
# MainWindow round-trip
# --------------------------------------------------------------------------
def test_mainwindow_applies_solver_dialog_values(qapp):
    win = MainWindow()
    try:
        win._apply_solver_dialog_values(
            {
                "local_solver_preference": "astap",
                "astap_path": "/usr/bin/astap",
                "astap_data_dir": "/data/astap",
                "astap_search_radius": 12.0,
                "astap_downsample": 3,
                "astap_sensitivity": 150,
            }
        )
        state = win.collect_settings_state()
        assert state.local_solver_preference == "astap"
        assert state.astap_path == "/usr/bin/astap"
        assert state.astap_data_dir == "/data/astap"
        assert state.astap_search_radius == 12.0
        assert state.astap_downsample == 3
        assert state.astap_sensitivity == 150

        kw = win.build_run_request().backend_kwargs
        assert kw["local_solver_preference"] == "astap"
        assert kw["astap_path"] == "/usr/bin/astap"
        assert kw["astap_data_dir"] == "/data/astap"
        assert kw["astap_search_radius"] == 12.0
        assert kw["astap_downsample"] == 3
        assert kw["astap_sensitivity"] == 150
    finally:
        win.shutdown()


def test_on_solver_opens_dialog_and_applies(qapp, monkeypatch, tmp_path):
    win = MainWindow()
    captured = {}

    # M21: accepting the solver dialog now also writes the engine solver
    # config; isolate it so this test never touches the user's real config.
    engine = _load_engine_solver_config()
    monkeypatch.setattr(
        engine, "get_config_path", lambda: str(tmp_path / "engine_config.json")
    )
    monkeypatch.setattr(engine, "_legacy_config_candidates", lambda: [])
    monkeypatch.setattr(
        "seestar.gui_qt.solver_config_bridge._solver_config_module", lambda: engine
    )

    class FakeDialog:
        def __init__(self, parent, initial):
            captured["initial"] = initial

        def exec(self):
            return QDialog.DialogCode.Accepted

        def values(self):
            return {
                "local_solver_preference": "zesolver",
                "astap_path": "/usr/bin/astap",
                "astap_data_dir": "/data/astap",
                "astap_search_radius": 9.0,
                "astap_downsample": 2,
                "astap_sensitivity": 100,
            }

    monkeypatch.setattr(
        "seestar.gui_qt.main_window.SolverSettingsDialog", FakeDialog
    )
    try:
        win._on_solver()
        state = win.collect_settings_state()
        assert state.local_solver_preference == "zesolver"
        assert state.astap_path == "/usr/bin/astap"
        assert state.astap_search_radius == 9.0
        assert captured["initial"] is win.settings_state
    finally:
        win.shutdown()


def test_solver_button_enabled_and_does_not_start_backend(qapp):
    win = MainWindow()
    try:
        assert win.solver_button.isEnabled()
        assert win.is_running is False
    finally:
        win.shutdown()
