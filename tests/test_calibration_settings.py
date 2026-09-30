"""C3 settings tests: calibration fields persist but never affect the science.

Covers:
* the two persisted Calibration fields (enable intent + last master folder) are
  present, defaulted, and round-trip through ``QtSettingsState``;
* a persisted "enabled" flag survives a disappeared provider (it is user intent,
  never provider availability — the tab is gated by the probe, not this flag);
* the calibration fields are NOT consumed by the run-request builder, so
  checking/unchecking the box cannot change the scientific path at this stage.

These tests are Qt-free in spirit: ``settings_state`` is pure stdlib and
``seestar.gui.run_config`` imports nothing GUI-related (no Tk, no Qt).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ss = importlib.import_module("seestar.gui_qt.settings_state")


def test_calibration_fields_default_and_roundtrip():
    st = ss.QtSettingsState()
    assert st.calibration_enabled is False
    assert st.calibration_master_folder == ""

    st.calibration_enabled = True
    st.calibration_master_folder = "/masters"
    rt = ss.QtSettingsState.from_dict(st.to_dict())
    assert rt.calibration_enabled is True
    assert rt.calibration_master_folder == "/masters"


def test_calibration_fields_coerced_from_corrupt_persisted():
    # int 1 -> bool True (the canonical JSON bool encoding accepted for a bool
    # field); a non-string folder degrades to the empty default. Never raises.
    st = ss.QtSettingsState.from_dict(
        {"calibration_enabled": 1, "calibration_master_folder": 123}
    )
    assert st.calibration_enabled is True
    assert st.calibration_master_folder == ""


def test_enabled_persisted_survives_provider_gone():
    # An "enabled" flag persisted in the settings file must not break ZSSS when
    # the provider later disappears: the state loads cleanly with the flag intact
    # (tab visibility is decided separately by the probe, never by this flag).
    raw = json.dumps(
        {"calibration_enabled": True, "calibration_master_folder": "/masters"}
    )
    st = ss.QtSettingsState.from_dict(json.loads(raw))
    assert st.calibration_enabled is True
    assert st.calibration_master_folder == "/masters"
    # The flag is NOT provider availability; it never turns the tab on by itself.


def test_calibration_fields_do_not_change_run_request():
    rc = importlib.import_module("seestar.gui.run_config")
    base = ss.QtSettingsState()
    enabled = ss.QtSettingsState()
    enabled.calibration_enabled = True
    enabled.calibration_master_folder = "/masters"

    r1 = rc.build_run_request(base)
    r2 = rc.build_run_request(enabled)

    assert dict(r1.backend_kwargs) == dict(r2.backend_kwargs)
    assert r1.align_on_disk == r2.align_on_disk
    assert r1.resume_intent == r2.resume_intent
    assert r1.resume_source == r2.resume_source
