"""M12 seam tests: initial-preview auto-load (first FITS of the input folder).

These tests exercise the Qt shell's Tk-parity initial-preview behaviour under
the ``offscreen`` Qt platform plugin:

* a real small FITS fixture in a temp folder loads and sets the preview source
  (image) + enables the view/preview controls,
* a missing input folder clears the preview with a localized message,
* an empty folder (no ``.fit``/``.fits``) clears the preview,
* a 2D Bayer FITS is debayered to a 3-channel result (pure loader unit test),
* the GUI thread stays responsive while a load is in flight (smoke),
* a repeated call with an unchanged folder does not reload (redundant-reload
  guard).

The loader runs on a daemon ``threading.Thread`` and delivers its result to the
GUI thread through a queued Qt signal; these tests pump the event loop to
observe that delivery.
"""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import MainWindow, create_application
from seestar.gui_qt import initial_preview as ip


def _pump_until(qapp: QApplication, predicate, timeout_ms: int = 8000) -> bool:
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    qapp.processEvents()
    return bool(predicate())


@pytest.fixture(scope="session")
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


def _write_fits(path, data, header=None):
    """Write a small synthetic FITS file (astropy is a test dependency)."""
    hdu = fits.PrimaryHDU(data=np.asarray(data))
    if header:
        for key, value in header.items():
            hdu.header[key] = value
    hdu.writeto(str(path), overwrite=True)


def _gradient(shape=(16, 16)):
    arr = np.linspace(0.0, 1.0, int(np.prod(shape))).reshape(shape).astype(np.float32)
    return arr


# --------------------------------------------------------------------------
# (1) real FITS loads and sets the preview source + enables controls
# --------------------------------------------------------------------------
def test_real_fits_loads_and_sets_preview(qapp, tmp_path):
    fits_path = tmp_path / "frame.fits"
    _write_fits(fits_path, _gradient())

    win = MainWindow()
    try:
        win.input_edit.setText(str(tmp_path))
        win._try_show_first_input_image()

        assert _pump_until(qapp, lambda: win.has_preview_image)
        assert win._preview_source is not None
        assert not win._preview_source.isNull()
        # A loaded preview enables the view + preview controls.
        assert win.zoom_combo.isEnabled()
        assert win.rotate_left_button.isEnabled()
        assert win.stretch_combo.isEnabled()
        assert win.wb_r_spin.isEnabled()
        # The detail label reports the loaded filename.
        assert "frame.fits" in win.preview_label.text()
        assert win._last_preview_folder == os.path.abspath(str(tmp_path))
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (2) missing folder clears preview with a localized message
# --------------------------------------------------------------------------
def test_missing_folder_clears_preview(qapp, tmp_path):
    win = MainWindow()
    try:
        missing = str(tmp_path / "does-not-exist")
        win.input_edit.setText(missing)
        win._try_show_first_input_image()

        assert not win.has_preview_image
        assert "Input folder not found or not set" in win.preview_label.text()
        assert not win.zoom_combo.isEnabled()
        assert not win.stretch_combo.isEnabled()
        assert win._last_preview_folder is None
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (3) empty folder (no .fit/.fits) clears preview
# --------------------------------------------------------------------------
def test_empty_folder_clears_preview(qapp, tmp_path):
    # A stray non-FITS file must not be picked up.
    (tmp_path / "notes.txt").write_text("not a fits")
    win = MainWindow()
    try:
        win.input_edit.setText(str(tmp_path))
        win._try_show_first_input_image()

        assert not win.has_preview_image
        assert "No FITS files in input folder" in win.preview_label.text()
        assert win._last_preview_folder is None
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (4) 2D Bayer FITS is debayered (3-channel result) — pure loader unit test
# --------------------------------------------------------------------------
def test_bayer_fits_is_debayered_to_three_channels(tmp_path):
    fits_path = tmp_path / "bayer.fits"
    # RGGB Bayer pattern in the header; 2D grayscale data.
    _write_fits(fits_path, _gradient(shape=(16, 16)), header={"BAYERPAT": "RGGB"})

    data, header = ip.load_initial_preview(str(tmp_path), "bayer.fits", "GRBG")

    assert data is not None
    assert data.ndim == 3
    assert data.shape[-1] == 3


# --------------------------------------------------------------------------
# (5) GUI thread stays responsive while a load is in flight (smoke)
# --------------------------------------------------------------------------
def test_gui_thread_stays_responsive_during_load(qapp, tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_load(folder, filename, bayer_pattern):
        started.set()
        release.wait(timeout=10.0)
        return np.full((16, 16), 0.5, dtype=np.float32), None

    monkeypatch.setattr(ip, "load_initial_preview", slow_load)
    (tmp_path / "slow.fits").write_bytes(b"dummy")

    win = MainWindow()
    try:
        win.input_edit.setText(str(tmp_path))
        win._try_show_first_input_image()
        assert started.wait(timeout=5.0), "worker thread never started"

        # While the worker is blocked, a zero-delay GUI timer must still fire,
        # proving the GUI thread is not frozen by the in-flight load.
        fired = []
        QTimer.singleShot(0, lambda: fired.append(True))
        deadline = time.monotonic() + 2.0
        while not fired and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.001)
        assert fired == [True], "GUI thread blocked while preview load in flight"
        assert not win.has_preview_image

        release.set()
        assert _pump_until(qapp, lambda: win.has_preview_image)
    finally:
        release.set()
        win.shutdown()


# --------------------------------------------------------------------------
# (6) redundant-reload guard: unchanged folder does not reload
# --------------------------------------------------------------------------
def test_unchanged_folder_does_not_reload(qapp, tmp_path, monkeypatch):
    fits_path = tmp_path / "frame.fits"
    _write_fits(fits_path, _gradient())

    calls = []
    real_load = ip.load_initial_preview

    def counting_load(folder, filename, bayer_pattern):
        calls.append(filename)
        return real_load(folder, filename, bayer_pattern)

    monkeypatch.setattr(ip, "load_initial_preview", counting_load)

    win = MainWindow()
    try:
        win.input_edit.setText(str(tmp_path))
        win._try_show_first_input_image()
        assert _pump_until(qapp, lambda: win.has_preview_image)
        assert calls == ["frame.fits"]

        # Same folder, no change: must be a no-op (no second load).
        win._try_show_first_input_image()
        qapp.processEvents()
        qapp.processEvents()
        assert calls == ["frame.fits"]
        assert win.has_preview_image
    finally:
        win.shutdown()
