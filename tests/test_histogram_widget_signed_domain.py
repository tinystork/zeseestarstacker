"""Signed float32 histogram-domain regression tests (Tk widget, headless).

Mission: zsss-signed-float32-display-histogram-20260911

The Tk histogram widget used to clamp the histogram domain floor to 0
(``calculated_min = max(0.0, calculated_min)`` plus ``0``-floored
``zoom_histogram`` / ``reset_histogram_view``), truncating the visible domain
of signed float32 stacks (reference product range approx. -4.733 ... +82.809).

These tests exercise the PURE computation only. No Tk root/window is created:
a bare widget instance is built with ``HistogramWidget.__new__`` and the
relevant attributes are populated manually, so the tests run headless with no
display. Inputs are 2-D because ``_calculate_hist_data`` only handles 2-D/3-D.
"""

from __future__ import annotations

import numpy as np
import pytest

histogram_widget = pytest.importorskip("seestar.gui.histogram_widget")
HistogramWidget = histogram_widget.HistogramWidget

# Reference signed float32 sample (2-D so the histogram path accepts it).
SIGNED = np.float32([[-4.733, -2.0, 0.0, 1.0, 82.809]])


# --------------------------------------------------------------------------- #
# Bare (no-Tk) widget builder
# --------------------------------------------------------------------------- #
def _bare_widget():
    w = HistogramWidget.__new__(HistogramWidget)
    w.data_min_for_current_plot = 0.0
    w.data_max_for_current_plot = 1.0
    w.freeze_x_range = False
    w._current_hist_data_details = None
    w.x_scale_mode = "linear"
    w._stored_xlim = None
    return w


class _FakeCanvas:
    def __init__(self):
        self.draws = 0

    def draw(self):
        self.draws += 1

    def draw_idle(self):
        self.draws += 1


class _FakeAx:
    def __init__(self, xlim=(0.0, 1.0)):
        self.xlim = tuple(xlim)
        self.scale = None

    def set_xlim(self, lo, hi=None):
        if hi is None:
            lo, hi = lo
        self.xlim = (float(lo), float(hi))

    def get_xlim(self):
        return self.xlim

    def set_xscale(self, scale):
        self.scale = scale


# --------------------------------------------------------------------------- #
# _calculate_hist_data — signed domain
# --------------------------------------------------------------------------- #
def test_signed_float32_domain_keeps_negatives():
    """Signed input: domain min < 0 and domain max > 0 (negatives not clipped)."""
    w = _bare_widget()

    res = w._calculate_hist_data(SIGNED)

    assert res is not None
    assert w.data_min_for_current_plot < 0.0, (
        f"negative tail clipped: min={w.data_min_for_current_plot!r}"
    )
    assert w.data_max_for_current_plot > 0.0, (
        f"positive max lost: max={w.data_max_for_current_plot!r}"
    )

    # True finite minimum must be present in the domain floor.
    assert w.data_min_for_current_plot == pytest.approx(float(np.min(SIGNED)), rel=0, abs=1e-5)
    # And the domain must cover the true finite maximum.
    assert w.data_max_for_current_plot >= float(np.max(SIGNED))

    # Bins span the signed domain and include negative edges.
    bins = res["bins"]
    assert bins[0] == pytest.approx(float(w.data_min_for_current_plot), rel=0, abs=1e-9)
    assert bins[-1] == pytest.approx(float(w.data_max_for_current_plot), rel=0, abs=1e-6)
    assert bins[0] < 0.0
    assert bins[-1] > 0.0

    # No pixel is dropped by clipping: total counts == number of finite pixels.
    total_counts = int(sum(int(np.sum(h)) for h in res["hists"]))
    assert total_counts == int(np.isfinite(SIGNED).sum())


def test_all_negative_domain_keeps_true_max():
    """Purely negative input: upper edge must not be shrunk below the true max."""
    data = np.float32([[-8.0, -4.733, -2.0]])
    w = _bare_widget()

    w._calculate_hist_data(data)

    assert w.data_min_for_current_plot == pytest.approx(-8.0, rel=0, abs=1e-5)
    assert w.data_max_for_current_plot > -2.0


# --------------------------------------------------------------------------- #
# _calculate_hist_data — non-negative invariance (byte-stable)
# --------------------------------------------------------------------------- #
def test_non_negative_input_domain_unchanged():
    """0-1 style input keeps the historical domain: min == 0, max == 1.001."""
    data = np.float32([[0.0, 0.25], [0.5, 1.0]])
    w = _bare_widget()

    res = w._calculate_hist_data(data)

    # Reproduce the pre-change domain arithmetic exactly (float32 preserved).
    expected_min = 0.0
    expected_max = np.max(data) * 1.001  # np.float32 result, as before
    assert w.data_min_for_current_plot == expected_min
    assert w.data_max_for_current_plot == expected_max

    # Byte-stable bins vs. the pre-change behaviour.
    domain = (float(w.data_min_for_current_plot), float(w.data_max_for_current_plot))
    fin = data.ravel()[np.isfinite(data.ravel())]
    expected_counts, expected_bins = np.histogram(
        np.clip(fin, domain[0], domain[1]), bins=256, range=domain
    )
    assert np.array_equal(res["bins"], expected_bins)
    assert np.array_equal(res["hists"][0], expected_counts)


def test_positive_adu_input_domain_unchanged():
    """Positive-ADU input keeps min == 0 and the *1.001 upper headroom."""
    data = np.float32([[0.0, 120.0], [2500.0, 3000.0]])
    w = _bare_widget()

    w._calculate_hist_data(data)

    assert w.data_min_for_current_plot == 0.0
    assert w.data_max_for_current_plot == np.max(data) * 1.001


def test_tiny_negative_noise_still_snapped_to_zero():
    """The tiny-negative-noise heuristic is preserved for clean frames."""
    data = np.float32([[-5e-7, 0.0], [0.5, 1.0]])
    w = _bare_widget()

    w._calculate_hist_data(data)

    assert w.data_min_for_current_plot == 0.0


# --------------------------------------------------------------------------- #
# zoom_histogram / reset_histogram_view — domain floor
# --------------------------------------------------------------------------- #
def test_zoom_histogram_respects_negative_floor():
    w = _bare_widget()
    w._current_data = SIGNED
    w.data_min_for_current_plot = float(np.min(SIGNED))
    w.data_max_for_current_plot = float(np.max(SIGNED)) * 1.001
    w.ax = _FakeAx()
    w.canvas = _FakeCanvas()

    w.zoom_histogram(percentile_max=99.5)

    assert w.ax.xlim[0] == pytest.approx(w.data_min_for_current_plot, rel=0, abs=1e-9)
    assert w.ax.xlim[0] < 0.0
    assert w.ax.xlim[1] > 0.0


def test_zoom_histogram_non_negative_floor_zero():
    data = np.float32([[0.0, 0.25, 0.5, 1.0]])
    w = _bare_widget()
    w._current_data = data
    w.data_min_for_current_plot = 0.0
    w.data_max_for_current_plot = 1.001
    w.ax = _FakeAx()
    w.canvas = _FakeCanvas()

    w.zoom_histogram(percentile_max=99.5)

    assert w.ax.xlim[0] == 0.0


def test_reset_histogram_view_signed_uses_data_domain():
    w = _bare_widget()
    w.data_min_for_current_plot = -4.733
    w.data_max_for_current_plot = 82.809 * 1.001
    w.ax = _FakeAx()
    w.canvas = _FakeCanvas()

    w.reset_histogram_view()

    assert w.ax.xlim[0] == pytest.approx(-4.733, rel=0, abs=1e-9)
    assert w.ax.xlim[1] == pytest.approx(82.809 * 1.001, rel=0, abs=1e-9)


def test_reset_histogram_view_non_negative_keeps_0_1():
    w = _bare_widget()
    w.data_min_for_current_plot = 0.0
    w.data_max_for_current_plot = 1.001
    w.ax = _FakeAx()
    w.canvas = _FakeCanvas()

    w.reset_histogram_view()

    assert w.ax.xlim == (0.0, 1.0)
