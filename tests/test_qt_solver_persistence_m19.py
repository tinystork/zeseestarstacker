"""M19 seam tests: solver settings persistence through the Qt JSON surface.

The six solver-dialog fields (``local_solver_preference``, ``astap_path``,
``astap_data_dir``, ``astap_search_radius``, ``astap_downsample``,
``astap_sensitivity``) live in :class:`QtSettingsState` and must round-trip
through the M8 ``settings_persistence`` JSON surface — *settings-only*, never
through the engine ``seestar.core.solver_config``.

These tests verify, under the ``offscreen`` platform plugin (and, for the pure
model/JSON layers, with no ``QApplication`` at all):

* solver-field defaults match the Tk ``SettingsManager`` GUI defaults, with the
  documented ``astap_downsample`` delta vs the engine ``solver_config``
  (``1`` in Qt/Tk GUI vs ``2`` in the engine config),
* save → load through ``settings_persistence`` preserves all six fields,
* absent keys fall back to defaults (``QtSettingsState.from_dict`` coercion),
* the dialog prefills from a persisted state and ``accept`` writes back into a
  state object,
* accepted values survive a full ``MainWindow`` save → reload round-trip,
* no write reaches ``seestar/core/solver_config.py`` (engine source + user
  config file are untouched; the Qt JSON uses its own keys, never the engine
  keys),
* M19 introduces no new localization keys (the dialog is a self-contained
  view).

No engine, no Tk, no FITS/PNG writes; ``_preview_source`` is never touched.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QDialog

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt.settings_state import QtSettingsState
from seestar.gui_qt import settings_persistence

ROOT = Path(__file__).resolve().parents[1]

# The six solver fields persisted by M19 and their canonical Qt defaults
# (mirroring the Tk SettingsManager GUI defaults).
_SOLVER_FIELDS = (
    "local_solver_preference",
    "astap_path",
    "astap_data_dir",
    "astap_search_radius",
    "astap_downsample",
    "astap_sensitivity",
)
_QT_SOLVER_DEFAULTS = {
    "local_solver_preference": "none",
    "astap_path": "",
    "astap_data_dir": "",
    "astap_search_radius": 3.0,
    "astap_downsample": 1,
    "astap_sensitivity": 100,
}


@pytest.fixture(scope="session")
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


def _load_settings_manager():
    """Load the Tk ``SettingsManager`` under an alias (pure, side-by-side)."""
    spec = importlib.util.spec_from_file_location(
        "seestar_settings_manager_m19", ROOT / "seestar" / "gui" / "settings.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["seestar_settings_manager_m19"] = mod
    spec.loader.exec_module(mod)
    return mod.SettingsManager


def _load_engine_solver_config():
    """Load ``seestar.core.solver_config`` in isolation (no ``seestar.core``
    package init, so no engine import chain is triggered)."""
    spec = importlib.util.spec_from_file_location(
        "engine_solver_config_m19", ROOT / "seestar" / "core" / "solver_config.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["engine_solver_config_m19"] = mod
    spec.loader.exec_module(mod)
    return mod


def _ui_state(label="ZeSolver : non installé", color="gray", show_button=False):
    return SimpleNamespace(
        label=label, status_color=color, show_configuration_button=show_button
    )


def _make_dialog(initial, **overrides):
    """Build the dialog with fully-injected (engine-free) boundary callables."""
    from seestar.gui_qt.solver_dialog import SolverSettingsDialog

    return SolverSettingsDialog(
        None,
        initial,
        check_readiness=overrides.get(
            "check_readiness", lambda: SimpleNamespace(state="not_installed")
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
# (1) defaults match Tk GUI defaults + documented downsample delta vs engine
# --------------------------------------------------------------------------
def test_solver_field_defaults_match_tk_and_document_downsample_delta():
    defaults = QtSettingsState.defaults()
    for field, expected in _QT_SOLVER_DEFAULTS.items():
        assert defaults[field] == expected, field

    # Tk GUI parity source: SettingsManager.get_default_values().
    sm = _load_settings_manager()(settings_file="unused.json")
    sm_defaults = sm.get_default_values()
    for field, expected in _QT_SOLVER_DEFAULTS.items():
        assert field in sm_defaults, f"SettingsManager missing {field}"
        assert sm_defaults[field] == expected, field

    # Engine config (never imported by the Qt shell) uses different key names
    # and a different downsample default — the documented delta.
    engine = _load_engine_solver_config()
    ecfg = engine.DEFAULT_CONFIG
    assert ecfg["astap_executable_path"] == ""
    assert ecfg["astap_data_directory_path"] == ""
    assert ecfg["astap_default_search_radius"] == 3.0
    assert ecfg["astap_default_sensitivity"] == 100
    # Documented divergence: Qt/Tk GUI downsample = 1, engine = 2.
    assert ecfg["astap_default_downsample"] == 2
    assert defaults["astap_downsample"] == 1
    assert defaults["astap_downsample"] != ecfg["astap_default_downsample"]


# --------------------------------------------------------------------------
# (2) settings_persistence save -> load round-trip preserves all 6 fields
# --------------------------------------------------------------------------
def test_solver_fields_round_trip_through_settings_persistence(tmp_path):
    state = QtSettingsState()
    state.local_solver_preference = "astap"
    state.astap_path = "/usr/bin/astap"
    state.astap_data_dir = "/data/astap"
    state.astap_search_radius = 12.5
    state.astap_downsample = 3
    state.astap_sensitivity = 200

    p = tmp_path / "settings.json"
    assert settings_persistence.save_settings_json(str(p), state.to_dict()) is True

    loaded = settings_persistence.load_settings_json(str(p))
    assert loaded, "settings JSON must not be empty"
    restored = QtSettingsState.from_dict(loaded)
    assert restored.local_solver_preference == "astap"
    assert restored.astap_path == "/usr/bin/astap"
    assert restored.astap_data_dir == "/data/astap"
    assert restored.astap_search_radius == 12.5
    assert restored.astap_downsample == 3
    assert restored.astap_sensitivity == 200


def test_solver_fields_round_trip_through_json_file(tmp_path):
    """The six solver keys appear literally in the persisted JSON surface."""
    state = QtSettingsState()
    state.astap_path = "/opt/astap"
    state.astap_search_radius = 5.0
    p = tmp_path / "settings.json"
    settings_persistence.save_settings_json(str(p), state.to_dict())
    raw = json.loads(Path(p).read_text(encoding="utf-8"))
    for field in _SOLVER_FIELDS:
        assert field in raw, field


# --------------------------------------------------------------------------
# (3) absent keys -> defaults
# --------------------------------------------------------------------------
def test_absent_solver_keys_fall_back_to_defaults():
    # A dict with none of the solver keys yields the documented defaults.
    state = QtSettingsState.from_dict({"batch_size": 4, "input_folder": "/in"})
    for field, expected in _QT_SOLVER_DEFAULTS.items():
        assert getattr(state, field) == expected, field


def test_corrupt_solver_values_fall_back_to_defaults():
    state = QtSettingsState.from_dict(
        {
            "astap_downsample": "not-a-number",
            "astap_search_radius": "not-a-number",
            "astap_sensitivity": "not-a-number",
        }
    )
    assert state.astap_downsample == 1
    assert state.astap_search_radius == 3.0
    assert state.astap_sensitivity == 100


# --------------------------------------------------------------------------
# (4) dialog prefill from persisted state + accept -> state updated
# --------------------------------------------------------------------------
def test_dialog_prefills_from_persisted_state_and_accept_updates_state(qapp):
    from seestar.gui_qt.solver_dialog import SOLVER_ASTAP

    persisted = {
        "local_solver_preference": "astap",
        "astap_path": "/usr/bin/astap",
        "astap_data_dir": "/data/astap",
        "astap_search_radius": 8.0,
        "astap_downsample": 2,
        "astap_sensitivity": 150,
    }
    initial = QtSettingsState.from_dict(persisted)

    dlg = _make_dialog(initial)
    try:
        # Prefill: the dialog reads the persisted state.
        assert dlg.astap_radio.isChecked()
        assert dlg._current_choice() == SOLVER_ASTAP
        assert dlg.astap_path_edit.text() == "/usr/bin/astap"
        assert dlg.astap_data_edit.text() == "/data/astap"
        assert dlg.search_radius_spin.value() == pytest.approx(8.0)
        assert dlg.downsample_spin.value() == 2
        assert dlg.sensitivity_spin.value() == 150

        # Change values and accept.
        dlg.search_radius_spin.setValue(22.5)
        dlg.downsample_spin.setValue(5)
        dlg.sensitivity_spin.setValue(300)
        dlg._on_ok()
        assert dlg.result() == QDialog.DialogCode.Accepted

        target = QtSettingsState()
        dlg.apply_to(target)
        assert target.local_solver_preference == "astap"
        assert target.astap_path == "/usr/bin/astap"
        assert target.astap_data_dir == "/data/astap"
        assert target.astap_search_radius == pytest.approx(22.5)
        assert target.astap_downsample == 5
        assert target.astap_sensitivity == 300
    finally:
        dlg.close()


# --------------------------------------------------------------------------
# (5) accepted values survive a MainWindow save -> reload round-trip (smoke)
# --------------------------------------------------------------------------
def test_dialog_accept_values_survive_save_load_round_trip(qapp, tmp_path, monkeypatch):
    settings_file = str(tmp_path / "qt_settings.json")

    win1 = MainWindow(backend_mode="simulated", settings_path=settings_file)

    class AcceptedDialog:
        def __init__(self, parent, initial):
            self.initial = initial

        def exec(self):
            return QDialog.DialogCode.Accepted

        def values(self):
            return {
                "local_solver_preference": "astap",
                "astap_path": "/usr/bin/astap",
                "astap_data_dir": "/data/astap",
                "astap_search_radius": 11.5,
                "astap_downsample": 4,
                "astap_sensitivity": 210,
            }

    monkeypatch.setattr(
        "seestar.gui_qt.main_window.SolverSettingsDialog", AcceptedDialog
    )
    try:
        # Open the solver dialog, "accept", save the surface.
        win1._on_solver()
        win1._save_persisted_settings()
    finally:
        win1.shutdown()

    # Reload the window from the same settings file -> values restored.
    win2 = MainWindow(backend_mode="simulated", settings_path=settings_file)
    try:
        state = win2.collect_settings_state()
        assert state.local_solver_preference == "astap"
        assert state.astap_path == "/usr/bin/astap"
        assert state.astap_data_dir == "/data/astap"
        assert state.astap_search_radius == pytest.approx(11.5)
        assert state.astap_downsample == 4
        assert state.astap_sensitivity == 210
        # The restored values are the single live model surface, and building a
        # run request still works (settings-only; no backend started).
        assert win2.build_run_request() is not None
    finally:
        win2.shutdown()


def test_persisted_solver_values_restore_into_dialog_prefill(qapp, tmp_path):
    """A reloaded window's state prefills a fresh solver dialog unchanged."""
    settings_file = str(tmp_path / "qt_settings.json")
    win = MainWindow(backend_mode="simulated", settings_path=settings_file)
    try:
        win._apply_solver_dialog_values(
            {
                "local_solver_preference": "zesolver",
                "astap_path": "/usr/bin/astap",
                "astap_data_dir": "/data/astap",
                "astap_search_radius": 6.5,
                "astap_downsample": 3,
                "astap_sensitivity": 160,
            }
        )
        win._save_persisted_settings()
    finally:
        win.shutdown()

    win2 = MainWindow(backend_mode="simulated", settings_path=settings_file)
    try:
        dlg = _make_dialog(win2.collect_settings_state())
        try:
            assert dlg.zesolver_radio.isChecked()
            assert dlg.astap_path_edit.text() == "/usr/bin/astap"
            assert dlg.astap_data_edit.text() == "/data/astap"
            assert dlg.search_radius_spin.value() == pytest.approx(6.5)
            assert dlg.downsample_spin.value() == 3
            assert dlg.sensitivity_spin.value() == 160
        finally:
            dlg.close()
    finally:
        win2.shutdown()


# --------------------------------------------------------------------------
# (6) no writes to seestar/core/solver_config.py
# --------------------------------------------------------------------------
def test_no_writes_to_engine_solver_config(tmp_path):
    engine_src_path = ROOT / "seestar" / "core" / "solver_config.py"
    engine_src_before = engine_src_path.read_bytes()

    engine = _load_engine_solver_config()
    engine_config_path = engine.get_config_path()
    had_config = os.path.exists(engine_config_path)
    config_before = None
    if had_config:
        with open(engine_config_path, "rb") as fh:
            config_before = fh.read()

    # Exercise the Qt persistence surface with every solver field populated.
    state = QtSettingsState()
    state.local_solver_preference = "astap"
    state.astap_path = "/usr/bin/astap"
    state.astap_data_dir = "/data/astap"
    state.astap_search_radius = 9.0
    state.astap_downsample = 2
    state.astap_sensitivity = 175

    p = tmp_path / "qt_settings.json"
    assert settings_persistence.save_settings_json(str(p), state.to_dict()) is True
    raw = json.loads(Path(p).read_text(encoding="utf-8"))

    # The Qt JSON surface uses Qt keys, never the engine solver_config keys.
    for field in _SOLVER_FIELDS:
        assert field in raw, field
    for engine_key in (
        "astap_executable_path",
        "astap_data_directory_path",
        "astap_default_search_radius",
        "astap_default_downsample",
        "astap_default_sensitivity",
    ):
        assert engine_key not in raw, engine_key

    # Engine source and engine user config file are untouched.
    assert engine_src_path.read_bytes() == engine_src_before
    assert os.path.exists(engine_config_path) == had_config
    if had_config:
        with open(engine_config_path, "rb") as fh:
            assert fh.read() == config_before


# --------------------------------------------------------------------------
# (7) no new localization keys (dialog is a self-contained view)
# --------------------------------------------------------------------------
def test_solver_persistence_adds_no_new_localization_keys():
    from seestar.gui_qt import localization

    # Full parity holds for every registered key (no dangling half-localized key).
    for key, entry in localization.TRANSLATIONS.items():
        assert set(entry) == {"en", "fr"}, key
        assert entry["en"] and entry["fr"], key

    # The solver dialog does not route its labels through the Qt localization
    # module, and M19 (persistence wiring) adds no user-facing labels.
    import seestar.gui_qt.solver_dialog as solver_dialog

    assert "localization" not in solver_dialog.__dict__


def test_solver_persistence_does_not_mutate_preview_source(qapp, tmp_path):
    settings_file = str(tmp_path / "qt_settings.json")
    win = MainWindow(backend_mode="simulated", settings_path=settings_file)
    try:
        before = win._preview_source
        win._apply_solver_dialog_values(
            {
                "local_solver_preference": "astap",
                "astap_path": "/usr/bin/astap",
                "astap_data_dir": "/data/astap",
                "astap_search_radius": 7.0,
                "astap_downsample": 2,
                "astap_sensitivity": 120,
            }
        )
        win._save_persisted_settings()
        assert win._preview_source is before
    finally:
        win.shutdown()
