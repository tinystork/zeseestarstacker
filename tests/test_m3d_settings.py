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

Isolation note: ``seestar.gui.settings`` is loaded standalone behind a fake
``seestar`` / ``seestar.gui`` package stub.  Those stubs are injected *only*
inside the :func:`_load_settings_isolated` context manager and their exact
previous ``sys.modules`` state is restored on exit, so a combined pytest run
that later imports the real ``seestar.gui.run_config`` (Qt shell) is never
poisoned by leftover fake modules.
"""

import importlib.util
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The exact ``sys.modules`` keys this module touches.  Every one of them is
# captured before injection and restored afterwards.
_SETTINGS_MODULE_KEYS = ("seestar", "seestar.gui", "seestar.gui.settings")


@contextmanager
def _load_settings_isolated():
    """Load ``seestar.gui.settings`` standalone and yield ``SettingsManager``.

    The fake ``seestar`` / ``seestar.gui`` packages exist only for the duration
    of this context.  On exit the exact previous ``sys.modules`` entries for the
    touched keys are restored, so no permanent mutation survives at module
    collection time or across tests.
    """
    saved = {name: sys.modules.get(name) for name in _SETTINGS_MODULE_KEYS}
    try:
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

        yield settings_mod.SettingsManager
    finally:
        for name in _SETTINGS_MODULE_KEYS:
            prev = saved.get(name)
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


@pytest.fixture
def settings_manager():
    """Yield ``SettingsManager`` with sys.modules restored after the test."""
    with _load_settings_isolated() as manager:
        yield manager


def test_default_group_size(settings_manager):
    sm = settings_manager(settings_file="unused.json")
    assert sm.drizzle_group_size == 50
    assert sm.get_default_values()["drizzle_group_size"] == 50


def test_fresh_wht_threshold_default_is_zero(settings_manager):
    sm = settings_manager(settings_file="unused.json")
    assert sm.drizzle_wht_threshold == 0.0
    assert sm.get_default_values()["drizzle_wht_threshold"] == 0.0


def test_validate_wht_threshold_zero_and_existing_positive_are_kept(settings_manager):
    sm = settings_manager(settings_file="unused.json")
    sm.drizzle_wht_threshold = 0.0
    sm.validate_settings()
    assert sm.drizzle_wht_threshold == 0.0

    sm.drizzle_wht_threshold = 0.7
    sm.validate_settings()
    assert sm.drizzle_wht_threshold == 0.7


def test_validate_group_size_valid_kept(settings_manager):
    sm = settings_manager(settings_file="unused.json")
    sm.drizzle_group_size = 120
    sm.validate_settings()
    assert sm.drizzle_group_size == 120


def test_validate_group_size_zero_fallback(settings_manager):
    sm = settings_manager(settings_file="unused.json")
    sm.drizzle_group_size = 0
    msgs = sm.validate_settings()
    assert sm.drizzle_group_size == 50
    assert any("groupe" in m.lower() for m in msgs)


def test_validate_group_size_negative_fallback(settings_manager):
    sm = settings_manager(settings_file="unused.json")
    sm.drizzle_group_size = -7
    msgs = sm.validate_settings()
    assert sm.drizzle_group_size == 50
    assert any("groupe" in m.lower() for m in msgs)


def test_validate_group_size_non_numeric_fallback(settings_manager):
    sm = settings_manager(settings_file="unused.json")
    sm.drizzle_group_size = "abc"
    msgs = sm.validate_settings()
    assert sm.drizzle_group_size == 50
    assert any("groupe" in m.lower() for m in msgs)


def test_save_settings_includes_group_size(tmp_path, settings_manager):
    out = tmp_path / "s.json"
    sm = settings_manager(settings_file=str(out))
    sm.drizzle_group_size = 77
    sm.save_settings()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["drizzle_group_size"] == 77


def test_load_settings_old_json_without_key_defaults(tmp_path, settings_manager):
    out = tmp_path / "old.json"
    # A minimal legacy settings file with no drizzle_group_size key.
    out.write_text(
        json.dumps({"drizzle_mode": "Final", "drizzle_scale": 2}),
        encoding="utf-8",
    )
    sm = settings_manager(settings_file=str(out))
    sm.load_settings()
    assert sm.drizzle_group_size == 50


def test_sys_modules_restored_after_loading():
    """The fake ``seestar`` / ``seestar.gui`` stubs must not leak into
    ``sys.modules`` once loading finishes — later Qt tests import the real
    ``seestar.gui.run_config`` and would fail on a leftover stub."""
    before = {
        name: sys.modules.get(name) for name in _SETTINGS_MODULE_KEYS
    }
    with _load_settings_isolated() as manager:
        assert manager is not None
        # While inside the context the fake stubs are present.
        assert "seestar" in sys.modules
        assert "seestar.gui" in sys.modules
    after = {
        name: sys.modules.get(name) for name in _SETTINGS_MODULE_KEYS
    }
    assert after == before


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
