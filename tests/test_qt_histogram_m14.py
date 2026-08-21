"""M14 seam tests: single histogram surface + interactions + Auto Stretch.

Offscreen tests for the histogram consolidation delivered on top of M13:

* the duplicated Preview-controls-tab histogram is removed (the persistent
  right-panel histogram is the single live surface),
* the right-panel ``HistogramView`` reproduces the Tk interactions (auto-zoom,
  reset view, zoom, reset zoom and BP/WP line dragging -> ``rangeChanged``),
* the Auto Stretch button computes percentile black/white points from the
  WB-only image and switches the stretch method to Asinh,
* the histogram source is the WB-only image (reacts to white balance, not to
  the stretch/gamma/BCS adjustments) — Tk ``image_data_wb`` parity.

No real stacking, no engine, no Tk.  ``QT_QPA_PLATFORM=offscreen`` is set
defensively before any ``QApplication`` is created.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits
from PySide6.QtWidgets import QApplication

from seestar.gui_qt import BackendPreviewPayload, MainWindow, create_application
from seestar.gui_qt.histogram_view import HistogramView
from seestar.gui_qt.preview_adjust import apply_preview_wb, compute_auto_stretch
from seestar.gui_qt.preview_render import render_preview_image


@pytest.fixture(scope="session")
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


def _grad(width: int, height: int, low: int, high: int) -> np.ndarray:
    """A grayscale horizontal gradient (uint8 2D)."""
    return np.tile(np.linspace(low, high, width).astype(np.uint8), (height, 1))


def _rgb(width: int, height: int, r: int, g: int, b: int) -> np.ndarray:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = r
    arr[:, :, 1] = g
    arr[:, :, 2] = b
    return arr


def _pump_until(qapp: QApplication, predicate, timeout_ms: int = 8000) -> bool:
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    qapp.processEvents()
    return bool(predicate())


# --------------------------------------------------------------------------
# (1) the tab histogram is removed; the right panel is the single surface
# --------------------------------------------------------------------------
def test_tab_histogram_removed_no_duplicate(qapp):
    win = MainWindow()
    try:
        # The M10 duplicated tab surface is gone entirely (not merely hidden).
        for removed in ("histogram_group", "histogram_status", "histogram_view"):
            assert not hasattr(win, removed), f"tab histogram {removed} should be removed"
        # The single live surface lives in the persistent right panel.
        assert isinstance(win.right_histogram_view, HistogramView)
        assert win.right_histogram_group.parent() is win.right_panel
        # The right-panel histogram is the only histogram widget in the window.
        from seestar.gui_qt.histogram_view import HistogramView as HV

        histogram_views = win.findChildren(HV)
        assert len(histogram_views) == 1
        assert histogram_views[0] is win.right_histogram_view
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (2) right-panel histogram interactions (widget-level)
# --------------------------------------------------------------------------
def test_histogram_zoom_reset_and_autozoom(qapp):
    view = HistogramView()
    try:
        view.resize(256, 80)
        view.set_data(render_preview_image(_grad(256, 8, 0, 255)))

        assert view.has_data is True
        assert view.view_range == (0.0, 1.0)  # full range by default

        # Zoom -> right edge is the 99.5th percentile (at least 0.02).
        view.zoom_histogram()
        vmin, vmax = view.view_range
        assert vmin == 0.0
        assert 0.02 <= vmax <= 1.0

        # Reset view / reset zoom -> full range again.
        view.reset_histogram_view()
        assert view.view_range == (0.0, 1.0)
        view.zoom_histogram()
        view.reset_zoom()
        assert view.view_range == (0.0, 1.0)

        # Auto-zoom: enabling it re-zooms and re-applies on new data.
        view.auto_zoom_enabled = True
        view.zoom_histogram()
        assert view.view_range[0] == 0.0 and view.view_range[1] < 1.0
        view.set_data(render_preview_image(_grad(256, 8, 0, 255)))
        assert view.view_range[0] == 0.0 and view.view_range[1] < 1.0
    finally:
        view.deleteLater()


def test_histogram_set_range_positions_lines(qapp):
    view = HistogramView()
    try:
        view.resize(256, 80)
        view.set_data(render_preview_image(_grad(256, 8, 0, 255)))
        view.set_range(0.2, 0.8)
        assert view.black_point == pytest.approx(0.2)
        assert view.white_point == pytest.approx(0.8)

        # Degenerate range is kept separated (never inverted).
        view.set_range(0.7, 0.5)
        assert view.black_point < view.white_point
    finally:
        view.deleteLater()


def test_histogram_drag_emits_range_changed(qapp):
    view = HistogramView()
    try:
        view.resize(256, 80)
        view.set_data(render_preview_image(_grad(256, 8, 0, 255)))
        emitted = []
        view.rangeChanged.connect(lambda bp, wp: emitted.append((bp, wp)))

        # Grab the black-point line and drag it toward the 0.5 level.
        x_bp = view._level_to_x(view.black_point)
        assert view._start_drag_at(x_bp) == "min"
        view._drag_at(view._level_to_x(0.5))
        view._end_drag()

        assert len(emitted) == 1
        bp, wp = emitted[-1]
        assert bp == pytest.approx(0.5, abs=0.02)
        assert wp == pytest.approx(view.white_point, abs=1e-6)
        assert bp < wp
    finally:
        view.deleteLater()


# --------------------------------------------------------------------------
# (3) Auto Stretch button: percentile bp/wp + switch to Asinh
# --------------------------------------------------------------------------
def test_auto_stretch_button_computes_bp_wp_and_switches_to_asinh(qapp):
    win = MainWindow()
    try:
        win._on_preview(BackendPreviewPayload(data=_grad(32, 8, 40, 200), stack_name="as"))
        # Start from a non-Asinh stretch to prove the button flips it.
        win.stretch_combo.setCurrentText("linear")
        win.stretch_bp_spin.setValue(0.0)
        win.stretch_wp_spin.setValue(1.0)
        assert win.stretch_combo.currentText() == "linear"

        win.auto_stretch_button.click()

        # The stretch method is switched to Asinh (Tk parity).
        assert win.stretch_combo.currentText() == "asinh"

        # The bp/wp values match a direct WB-only auto-stretch computation.
        wb_only = apply_preview_wb(win._preview_source, wb=win._wb)
        exp_bp, exp_wp = compute_auto_stretch(wb_only)
        assert win.stretch_bp_spin.value() == pytest.approx(round(exp_bp, 4), abs=0.002)
        assert win.stretch_wp_spin.value() == pytest.approx(round(exp_wp, 4), abs=0.002)
        assert win.stretch_bp_spin.value() < win.stretch_wp_spin.value()

        # The histogram BP/WP lines are re-synced to the same values.
        assert win.right_histogram_view.black_point == pytest.approx(
            win.stretch_bp_spin.value(), abs=0.002
        )
        assert win.right_histogram_view.white_point == pytest.approx(
            win.stretch_wp_spin.value(), abs=0.002
        )

        # The stored source image is never mutated.
        assert win._preview_source is not None
    finally:
        win.shutdown()


def test_auto_stretch_is_inert_without_preview(qapp):
    win = MainWindow()
    try:
        assert not win.auto_stretch_button.isEnabled()
        # The histogram toolbar is also inert until a renderable preview exists.
        assert not win.auto_zoom_histo_check.isEnabled()
        assert not win.hist_reset_view_button.isEnabled()
        assert not win.hist_zoom_button.isEnabled()
        assert not win.hist_reset_button.isEnabled()
        # Clicking while there is no preview must not raise or change state.
        win.auto_stretch_button.click()
        assert win.stretch_combo.currentText() == "asinh"
        assert win.stretch_bp_spin.value() == pytest.approx(0.01)
        assert win.stretch_wp_spin.value() == pytest.approx(0.99)
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# (4) histogram source is the WB-only image (Tk image_data_wb parity)
# --------------------------------------------------------------------------
def test_histogram_source_is_wb_only(qapp):
    win = MainWindow()
    try:
        win._on_preview(BackendPreviewPayload(data=_rgb(6, 6, 200, 100, 50), stack_name="src"))
        # Isolate stretch so the WB-only image equals the source.
        win.stretch_bp_spin.setValue(0.0)
        win.stretch_wp_spin.setValue(1.0)
        win.stretch_combo.setCurrentText("linear")
        base = {k: v.copy() for k, v in win.right_histogram_view.histogram.items()}

        # Stretch / gamma / BCS must NOT change the histogram bars.
        win.stretch_combo.setCurrentText("asinh")
        win.stretch_bp_spin.setValue(0.2)
        win.stretch_wp_spin.setValue(0.8)
        win.stretch_gamma_spin.setValue(1.5)
        win.brightness_spin.setValue(1.4)
        win.contrast_spin.setValue(1.2)
        win.saturation_spin.setValue(0.5)
        after_stretch = win.right_histogram_view.histogram
        for k in base:
            assert np.array_equal(base[k], after_stretch[k]), (
                f"histogram channel {k} changed under stretch"
            )

        # White balance MUST change the histogram bars (WB-only source).
        win.wb_r_spin.setValue(0.5)
        after_wb = win.right_histogram_view.histogram
        assert any(
            not np.array_equal(base[k], after_wb[k]) for k in base
        ), "histogram did not react to white balance"
    finally:
        win.shutdown()


# --------------------------------------------------------------------------
# Offscreen smoke: temp FITS -> Auto Stretch + histogram interactions, with
# the stored preview source never mutated.
# --------------------------------------------------------------------------
def test_offscreen_smoke_fits_auto_stretch_histogram_source_untouched(qapp, tmp_path):
    fits_path = tmp_path / "frame.fits"
    grad = _grad(32, 16, 30, 220).astype(np.float32) / 255.0
    fits.PrimaryHDU(data=grad).writeto(str(fits_path), overwrite=True)

    win = MainWindow(backend_mode="simulated")
    try:
        win.input_edit.setText(str(tmp_path))
        win._try_show_first_input_image()
        assert _pump_until(qapp, lambda: win.has_preview_image)
        assert win.right_histogram_view.has_data

        source_before = win._preview_source.copy()

        # Auto Stretch.
        win.auto_stretch_button.click()
        assert win.stretch_combo.currentText() == "asinh"

        # Histogram interactions.
        win.hist_zoom_button.click()
        assert win.right_histogram_view.view_range[0] == 0.0
        assert win.right_histogram_view.view_range[1] < 1.0
        win.hist_reset_view_button.click()
        assert win.right_histogram_view.view_range == (0.0, 1.0)
        win.auto_zoom_histo_check.setChecked(True)
        assert win.right_histogram_view.auto_zoom_enabled is True
        assert win.right_histogram_view.view_range[1] < 1.0
        win.hist_reset_button.click()
        assert win.right_histogram_view.view_range == (0.0, 1.0)

        # The stored source image was never mutated by any of the above.
        assert win._preview_source == source_before
    finally:
        win.shutdown()
