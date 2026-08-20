"""M3-D-2 settings tests for ``drizzle_group_size``.

``drizzle_group_size`` is a RESOURCE/PREVIEW policy knob (size of the group
used for the incremental DISPLAY-ONLY preview cadence), never a science
setting.  These tests cover the settings-side contract only (no Tk display is
required):

* default == 50,
* invalid values (<1, non-numeric) fall back to 50 with a clear message,
* ``save_settings`` serializes the key (the project has no separate ``to_dict``;
  the JSON dict built in ``save_settings`` is the serialization surface),
* old JSON files without the key still load with the default (backward compat).

The GUI-side label/radio strings (Standard / Large dataset / incremental,
values Final/Incremental preserved) are checked statically against the source
below, since a real Tk headless run is not available in this environment.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Load seestar.gui.settings standalone (it only imports stdlib + numpy + tkinter,
# no other seestar submodules).
seestar_pkg = types.ModuleType("seestar")
seestar_pkg.__path__ = [str(ROOT / "seestar")]
sys.modules["seestar"] = seestar_pkg

gui_pkg = types.ModuleType("seestar.gui")
gui_pkg.__path__ = []
sys.modules["seestar.gui"] = gui_pkg

settings_spec = importlib.util.spec_from_file_location(
    "seestar.gui.settings", ROOT / "seestar" / "gui" / "settings.py"
)
settings_mod = importlib.util.module_from_spec(settings_spec)
sys.modules["seestar.gui.settings"] = settings_mod
settings_spec.loader.exec_module(settings_mod)

SettingsManager = settings_mod.SettingsManager


def test_default_group_size():
    sm = SettingsManager(settings_file="unused.json")
    assert sm.drizzle_group_size == 50
    assert sm.get_default_values()["drizzle_group_size"] == 50


def test_validate_group_size_valid_kept():
    sm = SettingsManager(settings_file="unused.json")
    sm.drizzle_group_size = 120
    sm.validate_settings()
    assert sm.drizzle_group_size == 120


def test_validate_group_size_zero_fallback():
    sm = SettingsManager(settings_file="unused.json")
    sm.drizzle_group_size = 0
    msgs = sm.validate_settings()
    assert sm.drizzle_group_size == 50
    assert any("groupe" in m.lower() for m in msgs)


def test_validate_group_size_negative_fallback():
    sm = SettingsManager(settings_file="unused.json")
    sm.drizzle_group_size = -7
    msgs = sm.validate_settings()
    assert sm.drizzle_group_size == 50
    assert any("groupe" in m.lower() for m in msgs)


def test_validate_group_size_non_numeric_fallback():
    sm = SettingsManager(settings_file="unused.json")
    sm.drizzle_group_size = "abc"
    msgs = sm.validate_settings()
    assert sm.drizzle_group_size == 50
    assert any("groupe" in m.lower() for m in msgs)


def test_save_settings_includes_group_size(tmp_path):
    out = tmp_path / "s.json"
    sm = SettingsManager(settings_file=str(out))
    sm.drizzle_group_size = 77
    sm.save_settings()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["drizzle_group_size"] == 77


def test_load_settings_old_json_without_key_defaults(tmp_path):
    out = tmp_path / "old.json"
    # A minimal legacy settings file with no drizzle_group_size key.
    out.write_text(
        json.dumps({"drizzle_mode": "Final", "drizzle_scale": 2}),
        encoding="utf-8",
    )
    sm = SettingsManager(settings_file=str(out))
    sm.load_settings()
    assert sm.drizzle_group_size == 50


# --- static source checks (labels vs values) ---


def _main_window_source() -> str:
    return (ROOT / "seestar" / "gui" / "main_window.py").read_text(encoding="utf-8")


def test_radio_values_preserved_labels_only():
    src = _main_window_source()
    # Radio VALUES must stay Final / Incremental for settings compatibility.
    assert 'value="Final"' in src
    assert 'value="Incremental"' in src
    # Visible labels changed to avoid implying two distinct sciences.
    assert 'text="Final"' not in src
    assert 'text="Incremental"' not in src
    assert 'drizzle_processing_standard' in src
    assert 'drizzle_processing_incremental' in src
    assert 'text="Mode:"' not in src
    assert 'drizzle_processing_label' in src


def test_group_size_var_and_state_wiring():
    src = _main_window_source()
    assert "drizzle_group_size_var" in src
    assert "drizzle_group_size_spinbox" in src
    # Group size enabled only for Large dataset / Incremental policy.
    assert 'self.drizzle_mode_var.get() == "Incremental"' in src
