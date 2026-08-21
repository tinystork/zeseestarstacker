"""M21 seam tests: the Qt → engine solver-config bridge (checklist 3.6).

When the ``SolverSettingsDialog`` is *accepted*, the Qt shell must persist the
collected solver fields into the engine solver config
(``seestar.core.solver_config``) — the same user file the Tk GUI writes
(``seestar_config.json``) — using the engine's own ``load_config`` /
``save_config`` merge + legacy-migration semantics, with Tk-identical timing
(write on OK, nothing on cancel/ESC).

These tests verify, under the ``offscreen`` platform plugin (and for the pure
mapping/JSON layers, with no ``QApplication`` at all):

1. the exact Qt-field → engine-key mapping (+ type coercion),
2. accept → the engine config file is written,
3. Tk-parity golden: the Qt bridge and a direct Tk-style ``save_config`` write
   of the same mapped values produce byte-identical files,
4. cancel/ESC → no write,
5. absent config → defaults + overlaid values (created on save),
6. legacy ``zemosaic_config.json`` soft migration is untouched and still seeds,
7. lazy-import hygiene: importing the bridge module pulls nothing engine/Tk
   (fresh process),
8. dialog-level integration: accept through ``MainWindow._on_solver`` lands the
   values in the (isolated) engine config.

No FITS/PNG writes, no science/backend calls; ``_preview_source`` is never
touched.  The engine module is loaded *by file path* (as in
``test_solver_config.py``) so these tests never import the heavy ``seestar``
package tree; the config path is isolated per-test via ``get_config_path``.
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

ROOT = Path(__file__).resolve().parents[1]

_ENGINE_MODULE_NAME = "engine_solver_config_m21"


def _load_engine_solver_config():
    """Load ``seestar.core.solver_config`` by file path (no package init)."""
    if _ENGINE_MODULE_NAME in sys.modules:
        return sys.modules[_ENGINE_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(
        _ENGINE_MODULE_NAME, ROOT / "seestar" / "core" / "solver_config.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_ENGINE_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


def _isolate_engine(tmp_path, monkeypatch):
    """Load the engine module and redirect its config path into ``tmp_path``."""
    engine = _load_engine_solver_config()
    cfg_path = tmp_path / "seestar_config.json"
    monkeypatch.setattr(engine, "get_config_path", lambda: str(cfg_path))
    monkeypatch.setattr(engine, "_legacy_config_candidates", lambda: [])
    return engine, cfg_path


def _ui_state(label="ZeSolver : non installé", color="gray", show_button=False):
    return SimpleNamespace(
        label=label, status_color=color, show_configuration_button=show_button
    )


_FULL_VALUES = {
    "local_solver_preference": "astap",
    "astap_path": "/usr/bin/astap",
    "astap_data_dir": "/data/astap",
    "astap_search_radius": 12.5,
    "astap_downsample": 3,
    "astap_sensitivity": 200,
}


@pytest.fixture(scope="session")
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


# --------------------------------------------------------------------------
# (1) exact field mapping (+ coercion)
# --------------------------------------------------------------------------
def test_map_solver_fields_exact():
    from seestar.gui_qt.solver_config_bridge import map_solver_fields

    mapped = map_solver_fields(_FULL_VALUES)
    assert mapped == {
        "astap_executable_path": "/usr/bin/astap",
        "astap_data_directory_path": "/data/astap",
        "astap_default_search_radius": 12.5,
        "astap_default_downsample": 3,
        "astap_default_sensitivity": 200,
    }
    # The solver preference has no engine key and is never mapped.
    assert "local_solver_preference" not in mapped
    # Only the five known fields are emitted.
    assert set(mapped) == {
        "astap_executable_path",
        "astap_data_directory_path",
        "astap_default_search_radius",
        "astap_default_downsample",
        "astap_default_sensitivity",
    }


def test_map_solver_fields_coerces_types_and_strips_paths():
    from seestar.gui_qt.solver_config_bridge import map_solver_fields

    mapped = map_solver_fields(
        {
            "astap_path": "  /usr/bin/astap  ",
            "astap_data_dir": "  /data/astap  ",
            "astap_search_radius": "7.5",  # string -> float
            "astap_downsample": "4",       # string -> int
            "astap_sensitivity": 175.9,    # float -> int
        }
    )
    assert mapped["astap_executable_path"] == "/usr/bin/astap"
    assert mapped["astap_data_directory_path"] == "/data/astap"
    assert mapped["astap_default_search_radius"] == 7.5
    assert isinstance(mapped["astap_default_search_radius"], float)
    assert mapped["astap_default_downsample"] == 4
    assert isinstance(mapped["astap_default_downsample"], int)
    assert mapped["astap_default_sensitivity"] == 175
    assert isinstance(mapped["astap_default_sensitivity"], int)


def test_map_solver_fields_ignores_missing_and_unknown_keys():
    from seestar.gui_qt.solver_config_bridge import map_solver_fields

    mapped = map_solver_fields({"astap_downsample": 2, "other": "x"})
    assert mapped == {"astap_default_downsample": 2}


# --------------------------------------------------------------------------
# (2) accept -> engine config file written (isolated path)
# --------------------------------------------------------------------------
def test_write_solver_config_writes_engine_file(tmp_path, monkeypatch):
    from seestar.gui_qt.solver_config_bridge import write_solver_config

    engine, cfg_path = _isolate_engine(tmp_path, monkeypatch)

    assert write_solver_config(_FULL_VALUES, module=engine) is True
    assert cfg_path.exists()

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["astap_executable_path"] == "/usr/bin/astap"
    assert data["astap_data_directory_path"] == "/data/astap"
    assert data["astap_default_search_radius"] == 12.5
    assert data["astap_default_downsample"] == 3
    assert data["astap_default_sensitivity"] == 200
    # The solver choice is not an engine-config key (Tk parity).
    assert "local_solver_preference" not in data


# --------------------------------------------------------------------------
# (3) Tk-parity golden: byte-identical to a direct save_config write
# --------------------------------------------------------------------------
def test_tk_parity_golden_byte_identical(tmp_path, monkeypatch):
    from seestar.gui_qt.solver_config_bridge import map_solver_fields, write_solver_config

    engine = _load_engine_solver_config()
    qt_path = tmp_path / "qt_config.json"
    tk_path = tmp_path / "tk_config.json"

    # Qt bridge path (load -> overlay -> save via the engine's own functions).
    monkeypatch.setattr(engine, "get_config_path", lambda: str(qt_path))
    monkeypatch.setattr(engine, "_legacy_config_candidates", lambda: [])
    assert write_solver_config(_FULL_VALUES, module=engine) is True

    # Tk-equivalent path: exactly what Tk's ``_on_ok`` does with the same
    # mapped dict — ``load_config``, update the mapped keys, ``save_config``.
    monkeypatch.setattr(engine, "get_config_path", lambda: str(tk_path))
    tk_config = engine.load_config()
    tk_config.update(map_solver_fields(_FULL_VALUES))
    assert engine.save_config(tk_config) is True

    assert qt_path.read_bytes() == tk_path.read_bytes()


def test_tk_parity_round_trip_matches_load(tmp_path, monkeypatch):
    """After a bridge write, ``load_config`` returns the overlaid values."""
    from seestar.gui_qt.solver_config_bridge import write_solver_config

    engine, cfg_path = _isolate_engine(tmp_path, monkeypatch)
    assert write_solver_config(_FULL_VALUES, module=engine) is True

    loaded = engine.load_config()
    assert loaded["astap_executable_path"] == "/usr/bin/astap"
    assert loaded["astap_default_search_radius"] == 12.5
    assert loaded["astap_default_downsample"] == 3


# --------------------------------------------------------------------------
# (4) cancel -> no write (file absent)
# --------------------------------------------------------------------------
def test_cancel_does_not_write_engine_config(qapp, tmp_path, monkeypatch):
    engine, cfg_path = _isolate_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "seestar.gui_qt.solver_config_bridge._solver_config_module", lambda: engine
    )

    class RejectedDialog:
        def __init__(self, parent, initial):
            pass

        def exec(self):
            return QDialog.DialogCode.Rejected

        def values(self):
            raise AssertionError("values() must not be called on cancel")

    monkeypatch.setattr(
        "seestar.gui_qt.main_window.SolverSettingsDialog", RejectedDialog
    )

    win = MainWindow(backend_mode="simulated", settings_path=str(tmp_path / "qt.json"))
    try:
        win._on_solver()
    finally:
        win.shutdown()

    assert not cfg_path.exists()


# --------------------------------------------------------------------------
# (5) absent config -> defaults + overlaid values
# --------------------------------------------------------------------------
def test_absent_config_created_with_defaults_plus_overlay(tmp_path, monkeypatch):
    from seestar.gui_qt.solver_config_bridge import write_solver_config

    engine, cfg_path = _isolate_engine(tmp_path, monkeypatch)
    assert not cfg_path.exists()

    assert write_solver_config(
        {"astap_path": "/opt/astap", "astap_downsample": 4, "astap_sensitivity": 150},
        module=engine,
    ) is True

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    # Overlaid values.
    assert data["astap_executable_path"] == "/opt/astap"
    assert data["astap_default_downsample"] == 4
    assert data["astap_default_sensitivity"] == 150
    # Unspecified solver keys stay at engine defaults.
    assert data["astap_data_directory_path"] == ""
    assert data["astap_default_search_radius"] == 3.0
    # Every DEFAULT_CONFIG key is present (save_config filters to known keys,
    # and load_config seeded them).
    for key in engine.DEFAULT_CONFIG:
        assert key in data, key


# --------------------------------------------------------------------------
# (6) legacy zemosaic_config.json migration untouched (and still seeds)
# --------------------------------------------------------------------------
def test_legacy_migration_untouched(tmp_path, monkeypatch):
    from seestar.gui_qt.solver_config_bridge import write_solver_config

    engine = _load_engine_solver_config()
    new_path = tmp_path / "seestar_config.json"
    legacy = tmp_path / "zemosaic_config.json"
    legacy_raw = json.dumps(
        {"solver_method": "ansvr", "astap_default_search_radius": 5.0}
    )
    legacy.write_text(legacy_raw, encoding="utf-8")

    monkeypatch.setattr(engine, "get_config_path", lambda: str(new_path))
    monkeypatch.setattr(engine, "_legacy_config_candidates", lambda: [legacy])

    assert write_solver_config({"astap_downsample": 3}, module=engine) is True

    # Legacy file is read-only and left byte-identical.
    assert legacy.read_text(encoding="utf-8") == legacy_raw

    # New user file exists and was seeded from the legacy radius, then overlaid.
    data = json.loads(new_path.read_text(encoding="utf-8"))
    assert data["astap_default_search_radius"] == 5.0  # seeded from legacy
    assert data["astap_default_downsample"] == 3       # overlay
    # Legacy-only keys are dropped by the engine's own filter (unchanged).
    assert "solver_method" not in data


# --------------------------------------------------------------------------
# (7) lazy-import hygiene (fresh process)
# --------------------------------------------------------------------------
def test_bridge_module_import_is_engine_and_tk_free_fresh_process():
    import subprocess

    code = (
        "import sys\n"
        "import seestar.gui_qt.solver_config_bridge  # noqa: F401\n"
        "_bad = [m for m in sys.modules\n"
        "        if m.startswith('tkinter')\n"
        "        or m.startswith('seestar.core')\n"
        "        or m.startswith('seestar.alignment')\n"
        "        or m.startswith('seestar.enhancement')\n"
        "        or m.startswith('seestar.queuep')]\n"
        "if _bad:\n"
        "    print('BAD_MODULES:', _bad)\n"
        "    sys.exit(1)\n"
        "print('BRIDGE_IMPORT_CLEAN')\n"
    )
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    assert proc.returncode == 0, (
        f"bridge import hygiene violated: stdout={proc.stdout!r} "
        f"stderr={proc.stderr!r}"
    )


# --------------------------------------------------------------------------
# (8) dialog-level integration: accept through MainWindow._on_solver
# --------------------------------------------------------------------------
def test_dialog_accept_through_mainwindow_writes_engine_config(
    qapp, tmp_path, monkeypatch
):
    from seestar.gui_qt.solver_dialog import SolverSettingsDialog as RealSolverDialog

    engine, cfg_path = _isolate_engine(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "seestar.gui_qt.solver_config_bridge._solver_config_module", lambda: engine
    )

    # Keep the real dialog UI but inject the ZeSolver boundary + auto-accept so
    # the modal exec() is driven headlessly and no engine boundary is hit.
    monkeypatch.setattr(
        "seestar.gui_qt.solver_dialog._default_check_readiness",
        lambda: SimpleNamespace(state="not_installed"),
    )
    monkeypatch.setattr(
        "seestar.gui_qt.solver_dialog._default_ui_state",
        lambda d: _ui_state(),
    )
    monkeypatch.setattr(
        "seestar.gui_qt.solver_dialog._default_open_configuration",
        lambda: (True, None),
    )
    monkeypatch.setattr(
        "seestar.gui_qt.solver_dialog._default_session_refresh",
        lambda handle: "none",
    )

    class AutoAcceptDialog(RealSolverDialog):
        def exec(self):
            self.astap_radio.setChecked(True)
            self.astap_path_edit.setText("/usr/bin/astap")
            self.astap_data_edit.setText("/data/astap")
            self.search_radius_spin.setValue(9.5)
            self.downsample_spin.setValue(2)
            self.sensitivity_spin.setValue(180)
            self._on_ok()
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        "seestar.gui_qt.main_window.SolverSettingsDialog", AutoAcceptDialog
    )

    win = MainWindow(backend_mode="simulated", settings_path=str(tmp_path / "qt.json"))
    try:
        win._on_solver()
    finally:
        win.shutdown()

    assert cfg_path.exists()
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["astap_executable_path"] == "/usr/bin/astap"
    assert data["astap_data_directory_path"] == "/data/astap"
    assert data["astap_default_search_radius"] == pytest.approx(9.5)
    assert data["astap_default_downsample"] == 2
    assert data["astap_default_sensitivity"] == 180
    assert "local_solver_preference" not in data
