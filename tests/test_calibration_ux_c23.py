"""C23 — calibration UX + P1 content-validity parity witnesses.

(A) enabled-without-folder -> clear message; calibrated path publishes
content-validity evidence from the provider DQ mask.
(C) master folder is NOT restored at startup (honest empty state) and is only
re-read after Browse.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from seestar.queuep.queue_manager import SeestarQueuedStacker


# ---------------------------------------------------------------------------
# (A) enabled-without-folder -> clear message
# ---------------------------------------------------------------------------
def test_enabled_without_folder_produces_clear_message():
    s = object.__new__(SeestarQueuedStacker)
    s._calibration_enabled = True
    s._calibration_master_folder = ""
    s._calibration_integrator = None
    messages = []
    s.update_progress = lambda msg, level=None: messages.append(msg)
    s._open_calibration_session()
    assert messages, "expected a clear message for enabled-without-folder"
    assert any("no master folder" in m for m in messages), messages
    assert s._calibration_integrator is None


def test_disabled_session_opens_silently():
    s = object.__new__(SeestarQueuedStacker)
    s._calibration_enabled = False
    s._calibration_master_folder = ""
    s._calibration_integrator = None
    messages = []
    s.update_progress = lambda msg, level=None: messages.append(msg)
    s._open_calibration_session()
    assert messages == []  # disabled -> no message, no session
    assert s._calibration_integrator is None


# ---------------------------------------------------------------------------
# (A) calibrated path publishes content-validity evidence from the DQ mask
# ---------------------------------------------------------------------------
def test_dq_mask_becomes_content_invalid_evidence():
    from seestar.core.overlap_normalization import content_validity_after_loader

    dq = np.zeros((5, 5), dtype=np.uint16)
    dq[0, 0] = 0x0001  # INPUT_INVALID
    dq[2, 3] = 0x0008  # SATURATED
    invalid = np.asarray(dq) != 0
    cv = content_validity_after_loader(invalid, bayer=False)
    assert cv.dtype == bool
    assert bool(cv[0, 0]) is False  # invalid pixel excluded
    assert bool(cv[2, 3]) is False  # saturated pixel excluded
    assert bool(cv[1, 1]) is True   # clean pixel kept


# ---------------------------------------------------------------------------
# (C) master folder not restored at startup
# ---------------------------------------------------------------------------
def test_master_folder_not_restored_at_startup(tmp_path):
    from seestar.gui_qt import MainWindow, create_application
    from PySide6.QtWidgets import QApplication

    app = create_application([])
    assert app is QApplication.instance()
    settings_path = tmp_path / "s.json"
    settings_path.write_text(json.dumps({
        "calibration_enabled": True,
        "calibration_master_folder": "/tmp/synthetic_masters",
    }))
    win = MainWindow(settings_path=str(settings_path))
    try:
        # Honest empty state: the persisted folder is NOT restored (it would be
        # a value that was never re-read into a live session).
        assert win.settings_state.calibration_master_folder == ""
        # The enable intent itself stays persisted.
        assert win.settings_state.calibration_enabled is True
    finally:
        win.shutdown()


def _provider_available(monkeypatch):
    from seestar.gui_qt import calibration_service
    from seestar.core.calibration_port import ProviderInfo, ProviderState

    def _fake(state):
        return ProviderInfo(state=state, provider_id="zecalibrator",
                            api_version="1.1", api_major="1", product_version="0.1.0")

    monkeypatch.setattr(calibration_service, "check_calibration_availability",
                        lambda: _fake(ProviderState.AVAILABLE))


def test_master_folder_field_empty_with_provider(monkeypatch, tmp_path):
    _provider_available(monkeypatch)
    from seestar.gui_qt import MainWindow, create_application
    from PySide6.QtWidgets import QApplication

    app = create_application([])
    assert app is QApplication.instance()
    settings_path = tmp_path / "s.json"
    settings_path.write_text(json.dumps({
        "calibration_enabled": True,
        "calibration_master_folder": "/tmp/synthetic_masters",
    }))
    win = MainWindow(settings_path=str(settings_path))
    try:
        assert win.calibration_folder_edit.text() == ""
    finally:
        win.shutdown()


def test_master_folder_persisted_when_browsed(monkeypatch, tmp_path):
    _provider_available(monkeypatch)
    from seestar.gui_qt import MainWindow, create_application
    from PySide6.QtWidgets import QApplication

    app = create_application([])
    assert app is QApplication.instance()
    settings_path = tmp_path / "s.json"
    win = MainWindow(settings_path=str(settings_path))
    try:
        win.calibration_folder_edit.setText("/tmp/selected_masters")
        win._sync_state_from_controls()
        assert win.settings_state.calibration_master_folder == "/tmp/selected_masters"
    finally:
        win.shutdown()
