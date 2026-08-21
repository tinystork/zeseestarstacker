"""M13 seam tests: full preview-controls parity (display-only).

Offscreen tests for the Tk-parity preview-adjustment surface delivered on top
of M10:

* Auto WB computes R/G/B gains from the display image and applies them,
* black / white / gamma controls change the display image within the Tk ranges,
* brightness / contrast / saturation adjust and reset back to the baseline,
* the defaults match Tk exactly (Asinh stretch, WB 1/1/1, bp 0.01 / wp 0.99 /
  gamma 1.0 / B-C-S 1.0),
* numeric values are shown next to the RGB WB sliders (slider + spinbox pair),
* a stale initial-preview result whose folder no longer matches the selected
  input folder is ignored (fast folder-switch hardening).

No real stacking, no engine, no Tk.  ``QT_QPA_PLATFORM=offscreen`` is set
defensively before any ``QApplication`` is created.
"""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QSlider

from seestar.gui_qt import BackendPreviewPayload, MainWindow, create_application
from seestar.gui_qt.preview_adjust import WB_MAX, WB_MIN, WB_STEP


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


def _rgb(width: int, height: int, r: int, g: int, b: int):
    """Build a ``width x height`` RGB uint8 array filled with a constant colour."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = r
    arr[:, :, 1] = g
    arr[:, :, 2] = b
    return arr


def _set_identity_stretch(win) -> None:
    """Neutral WB + linear stretch with 0/1 black/white + unit gamma/B-C-S."""
    win.stretch_bp_spin.setValue(0.0)
    win.stretch_wp_spin.setValue(1.0)
    win.stretch_combo.setCurrentText("linear")


# --------------------------------------------------------------------------
# (1) Auto WB changes the display image and sets WB gains
# --------------------------------------------------------------------------
def test_auto_wb_changes_display_and_sets_gains(qapp):
    win = MainWindow()
    try:
        win._on_preview(
            BackendPreviewPayload(data=_rgb(8, 8, 200, 100, 50), stack_name="awb")
        )
        before = win.preview_image_label.pixmap().toImage().pixelColor(0, 0)
        assert before.red() > before.green() > before.blue()

        win.auto_wb_button.click()

        r, g, b = win._wb
        assert abs(g - 1.0) < 1e-6
        assert abs(r - 0.5) < 0.05
        assert abs(b - 2.0) < 0.05
        # The gains are mirrored into the WB spinboxes.
        assert abs(win.wb_r_spin.value() - r) < 1e-6
        assert abs(win.wb_b_spin.value() - b) < 1e-6

        # The display changed and the colour cast is neutralised (r == g == b).
        after = win.preview_image_label.pixmap().toImage().pixelColor(0, 0)
        assert after != before
        assert abs(after.red() - after.green()) <= 1
        assert abs(after.green() - after.blue()) <= 1
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (2) black / white / gamma change the display image within Tk ranges
# --------------------------------------------------------------------------
def test_black_white_gamma_change_display_within_tk_ranges(qapp):
    win = MainWindow()
    try:
        grad = np.tile(np.linspace(0, 255, 64).astype(np.uint8), (8, 1))
        win._on_preview(BackendPreviewPayload(data=grad, stack_name="bwg"))
        _set_identity_stretch(win)

        # Ranges/steps must match Tk exactly.
        assert (
            win.stretch_bp_spin.minimum(),
            win.stretch_bp_spin.maximum(),
            win.stretch_bp_spin.singleStep(),
        ) == (0.0, 1.0, 0.001)
        assert (
            win.stretch_wp_spin.minimum(),
            win.stretch_wp_spin.maximum(),
            win.stretch_wp_spin.singleStep(),
        ) == (0.0, 1.0, 0.001)
        assert (
            win.stretch_gamma_spin.minimum(),
            win.stretch_gamma_spin.maximum(),
            win.stretch_gamma_spin.singleStep(),
        ) == (0.1, 5.0, 0.01)

        mid = win.preview_image_label.pixmap().toImage().pixelColor(32, 0).red()

        # Raising the black point darkens the mid-tone.
        win.stretch_bp_spin.setValue(0.3)
        assert win.preview_image_label.pixmap().toImage().pixelColor(32, 0).red() < mid

        # Lowering the white point brightens the mid-tone.
        win.stretch_bp_spin.setValue(0.0)
        win.stretch_wp_spin.setValue(0.5)
        assert win.preview_image_label.pixmap().toImage().pixelColor(32, 0).red() > mid

        # Gamma > 1 darkens mid-tones; gamma < 1 brightens them.
        win.stretch_wp_spin.setValue(1.0)
        win.stretch_gamma_spin.setValue(2.0)
        assert win.preview_image_label.pixmap().toImage().pixelColor(32, 0).red() < mid
        win.stretch_gamma_spin.setValue(0.5)
        assert win.preview_image_label.pixmap().toImage().pixelColor(32, 0).red() > mid
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (3) brightness / contrast / saturation adjust + reset returns to baseline
# --------------------------------------------------------------------------
def test_brightness_contrast_saturation_adjust_and_reset(qapp):
    win = MainWindow()
    try:
        win._on_preview(
            BackendPreviewPayload(data=_rgb(6, 6, 180, 90, 45), stack_name="bcs")
        )
        _set_identity_stretch(win)

        base_color = win.preview_image_label.pixmap().toImage().pixelColor(0, 0)

        # Tk ranges/steps.
        assert (
            win.brightness_spin.minimum(),
            win.brightness_spin.maximum(),
            win.brightness_spin.singleStep(),
        ) == (0.1, 3.0, 0.01)
        assert (
            win.contrast_spin.minimum(),
            win.contrast_spin.maximum(),
            win.contrast_spin.singleStep(),
        ) == (0.1, 3.0, 0.01)
        assert (
            win.saturation_spin.minimum(),
            win.saturation_spin.maximum(),
            win.saturation_spin.singleStep(),
        ) == (0.0, 3.0, 0.01)

        # Brightness multiplies -> red brightens.
        win.brightness_spin.setValue(1.5)
        bright = win.preview_image_label.pixmap().toImage().pixelColor(0, 0)
        assert bright.red() > base_color.red()

        # Contrast blends away from the band-0 mean -> green moves.
        win.brightness_spin.setValue(1.0)
        win.contrast_spin.setValue(1.5)
        contr = win.preview_image_label.pixmap().toImage().pixelColor(0, 0)
        assert contr.green() != base_color.green()

        # Saturation 0 -> fully desaturated (equal channels).
        win.contrast_spin.setValue(1.0)
        win.saturation_spin.setValue(0.0)
        sat = win.preview_image_label.pixmap().toImage().pixelColor(0, 0)
        assert abs(sat.red() - sat.green()) <= 1
        assert abs(sat.green() - sat.blue()) <= 1

        # Reset returns to the baseline colour.
        win.bcs_reset_button.click()
        reset = win.preview_image_label.pixmap().toImage().pixelColor(0, 0)
        assert reset == base_color
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (4) defaults match Tk
# --------------------------------------------------------------------------
def test_defaults_match_tk(qapp):
    win = MainWindow()
    try:
        assert win.stretch_combo.currentText() == "asinh"
        assert (
            win.wb_r_spin.value(),
            win.wb_g_spin.value(),
            win.wb_b_spin.value(),
        ) == (1.0, 1.0, 1.0)
        assert win.stretch_bp_spin.value() == pytest.approx(0.01)
        assert win.stretch_wp_spin.value() == pytest.approx(0.99)
        assert win.stretch_gamma_spin.value() == 1.0
        assert win.brightness_spin.value() == 1.0
        assert win.contrast_spin.value() == 1.0
        assert win.saturation_spin.value() == 1.0

        # Stored adjustment state mirrors the defaults.
        assert win._stretch == "asinh"
        assert win._wb == (1.0, 1.0, 1.0)
        assert win._black_point == 0.01
        assert win._white_point == 0.99
        assert win._gamma == 1.0
        assert win._brightness == 1.0
        assert win._contrast == 1.0
        assert win._saturation == 1.0
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (5) numeric values are shown next to the RGB WB sliders
# --------------------------------------------------------------------------
def test_wb_sliders_show_numeric_values(qapp):
    win = MainWindow()
    try:
        win._on_preview(
            BackendPreviewPayload(data=_rgb(16, 16, 128, 128, 128), stack_name="num")
        )
        for slider, spin in (
            (win.wb_r_slider, win.wb_r_spin),
            (win.wb_g_slider, win.wb_g_spin),
            (win.wb_b_slider, win.wb_b_spin),
        ):
            assert isinstance(slider, QSlider)
            assert isinstance(spin, QDoubleSpinBox)
            # The spinbox (numeric value) sits next to the slider in the same row.
            assert spin.parent() is slider.parent()

        # WB slider range/step match Tk (0.1..5.0, step 0.01).
        assert (win.wb_r_spin.minimum(), win.wb_r_spin.maximum(), win.wb_r_spin.singleStep()) == (
            WB_MIN,
            WB_MAX,
            WB_STEP,
        )

        # Moving a slider updates its numeric spinbox (and vice versa).
        target = 3.0
        pos = int(round((target - WB_MIN) / WB_STEP))
        win.wb_r_slider.setValue(pos)
        assert win.wb_r_spin.value() == pytest.approx(target, abs=0.02)

        win.wb_r_spin.setValue(0.5)
        assert win.wb_r_slider.value() == int(round((0.5 - WB_MIN) / WB_STEP))
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (6) staleness guard: a stale initial-preview result must not overwrite the
#     new folder's preview
# --------------------------------------------------------------------------
def test_stale_initial_preview_result_is_ignored(qapp, tmp_path, monkeypatch):
    import seestar.gui_qt.initial_preview as ip

    folder_a = tmp_path / "A"
    folder_b = tmp_path / "B"
    folder_a.mkdir()
    folder_b.mkdir()
    (folder_a / "a.fits").write_bytes(b"x")
    (folder_b / "b.fits").write_bytes(b"x")

    arr_a = _rgb(4, 4, 255, 0, 0)  # red
    arr_b = _rgb(4, 4, 0, 0, 255)  # blue

    release_a = threading.Event()
    a_done = threading.Event()

    def fake_load(folder, filename, bayer_pattern):
        if str(folder) == str(folder_a):
            release_a.wait(timeout=10.0)
            try:
                return arr_a.copy(), None
            finally:
                a_done.set()
        return arr_b.copy(), None

    monkeypatch.setattr(ip, "load_initial_preview", fake_load)

    win = MainWindow()
    try:
        # Start loading folder A (blocks in the fake loader), then switch to B.
        win.input_edit.setText(str(folder_a))
        win._try_show_first_input_image()
        win.input_edit.setText(str(folder_b))
        win._try_show_first_input_image()

        # B's (fast) result arrives first and is displayed.
        assert _pump_until(
            qapp, lambda: win._last_preview_folder == os.path.abspath(str(folder_b))
        )
        color = win.preview_image_label.pixmap().toImage().pixelColor(0, 0)
        assert color.blue() > 200 and color.red() < 20

        # Let A's stale result arrive and be delivered through the event loop.
        release_a.set()
        assert a_done.wait(timeout=5.0), "stale loader never finished"
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.005)

        # The stale A result must NOT overwrite B's preview.
        assert win._last_preview_folder == os.path.abspath(str(folder_b))
        color2 = win.preview_image_label.pixmap().toImage().pixelColor(0, 0)
        assert color2.blue() > 200 and color2.red() < 20
    finally:
        release_a.set()
        win.shutdown()
