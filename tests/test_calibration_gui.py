"""C3 GUI tests (offscreen): the Calibration tab is conditional + import hygiene.

* The Calibration tab is inserted only when the provider probe is AVAILABLE;
  absent / broken / incompatible -> no tab (ZSSS behaves exactly as before).
* Import hygiene is measured in a separate process: ``import seestar.gui_qt``
  must not import ``zecalibrator``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt import calibration_service
from seestar.core.calibration_port import ProviderInfo, ProviderState


@pytest.fixture(scope="session", autouse=True)
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


def _fake_info(state):
    return ProviderInfo(
        state=state,
        provider_id="zecalibrator",
        api_version="1.1",
        api_major="1",
        product_version="0.0.5",
    )


def _tab_exists(win) -> bool:
    return win._calibration_tab is not None


def test_calibration_tab_absent_when_provider_not_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        calibration_service,
        "check_calibration_availability",
        lambda: _fake_info(ProviderState.NOT_INSTALLED),
    )
    win = MainWindow(settings_path=str(tmp_path / "s.json"))
    try:
        assert _tab_exists(win) is False
    finally:
        win.close()


def test_calibration_tab_absent_when_provider_unhealthy(monkeypatch, tmp_path):
    monkeypatch.setattr(
        calibration_service,
        "check_calibration_availability",
        lambda: _fake_info(ProviderState.UNHEALTHY),
    )
    win = MainWindow(settings_path=str(tmp_path / "s.json"))
    try:
        assert _tab_exists(win) is False
    finally:
        win.close()


def test_calibration_tab_absent_when_provider_incompatible(monkeypatch, tmp_path):
    monkeypatch.setattr(
        calibration_service,
        "check_calibration_availability",
        lambda: _fake_info(ProviderState.INCOMPATIBLE),
    )
    win = MainWindow(settings_path=str(tmp_path / "s.json"))
    try:
        assert _tab_exists(win) is False
    finally:
        win.close()


def test_calibration_tab_present_when_provider_available(monkeypatch, tmp_path):
    monkeypatch.setattr(
        calibration_service,
        "check_calibration_availability",
        lambda: _fake_info(ProviderState.AVAILABLE),
    )
    win = MainWindow(settings_path=str(tmp_path / "s.json"))
    try:
        assert _tab_exists(win) is True
        # Provider status line renders name / product / API versions.
        text = win.calibration_provider_label.text()
        assert "zecalibrator" in text
        assert "0.0.5" in text
        assert "1.1" in text
    finally:
        win.close()


def test_calibration_tab_index_between_stacking_and_expert(monkeypatch, tmp_path):
    monkeypatch.setattr(
        calibration_service,
        "check_calibration_availability",
        lambda: _fake_info(ProviderState.AVAILABLE),
    )
    win = MainWindow(settings_path=str(tmp_path / "s.json"))
    try:
        labels = [win.tabs.tabText(i) for i in range(win.tabs.count())]
        assert labels[0] == "Stacking"
        assert "Calibration" in labels
        assert labels.index("Calibration") == 1  # between Stacking and Expert
        assert labels[2] == "Expert"
    finally:
        win.close()


# ---------------------------------------------------------------------------
# Import hygiene (separate process)
# ---------------------------------------------------------------------------
def test_import_gui_qt_does_not_import_zecalibrator_subprocess():
    code = (
        "import sys\n"
        "import seestar.gui_qt\n"
        "assert 'zecalibrator' not in sys.modules, 'zecalibrator was imported'\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
