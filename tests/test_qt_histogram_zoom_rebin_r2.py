"""R2 — narrow-zoom adaptive histogram re-binning (headless).

Mission: zsss-signed-float32-display-histogram-20260911 (phase r2-zoom-rebin).

Covers:

* signed domain re-binning (negative visible sub-range);
* non-negative legacy/display domain re-binning;
* narrow histogram zoom resolves the visible window (no sparse comb);
* reset after zoom restores the full-domain distribution;
* repeated zoom/reset is idempotent (no state drift);
* X tick formatting renders negative levels correctly;
* un-zoomed default view / non-narrow zoom keep the exact global bars;
* bounded + deterministic re-binning; ``compute_histogram_float`` keys unchanged.

The Qt view logic is exercised without a live display via the offscreen
platform (the ``qapp`` fixture pattern used by the existing histogram tests);
the pure re-binning helper is tested directly with numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from PySide6.QtWidgets import QApplication

import seestar.gui_qt.histogram_view as hv
from seestar.gui_qt import create_application
from seestar.gui_qt.histogram_view import (
    HistogramView,
    adaptive_rebin_bars,
    format_axis_level,
)
from seestar.gui_qt.preview_analysis import (
    MAX_SAMPLE_PIXELS,
    compute_histogram_float,
)
from seestar.gui_qt.scientific_histogram import compute_scientific_histogram


@pytest.fixture(scope="session")
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


def _narrow_display_model(seed: int = 4):
    """Synthetic display model that reproduces the witnessed sparse comb.

    The plotted bar domain is wide ``(0, 100)`` (512 global bins, ~0.195/bin)
    while the robust viewport is a NARROW window ``(0.68, 0.83)`` holding a
    dense sampled population: the global grid resolves it into ONE occupied bin
    (the comb).  The model retains the bounded sample so the view can re-bin.
    """
    rng = np.random.default_rng(seed)
    n_bins = 512
    domain = 100.0
    sample = rng.uniform(0.68, 0.83, size=90_000)
    counts, _ = np.histogram(sample, bins=n_bins, range=(0.0, domain))
    counts = counts.astype(np.int64)
    return {
        "bins": n_bins,
        "range": (0.0, domain),
        "full_range": (0.0, domain),
        "bin_range": (0.0, domain),
        "full_hist_range": (0.0, domain),
        "channels": ["L"],
        "counts": {"L": counts},
        "log_counts": {"L": np.log1p(counts.astype(np.float64))},
        "full_counts": {"L": counts},
        "full_log_counts": {"L": np.log1p(counts.astype(np.float64))},
        "stats": {"L": {"min": 0.68, "max": 0.83, "median": 0.75,
                        "mean": 0.75, "std": 0.04}},
        "x_range": (0.68, 0.83),
        "overflow": {"L": 0},
        "overflow_total": 0,
        "samples": {"L": sample},
    }


def _global_bins_in_window(model):
    """Number of OCCUPIED global bins whose centre lies in the x_range window."""
    lo, hi = model["x_range"]
    bin_lo, bin_hi = model["bin_range"]
    heights = model["log_counts"]["L"]
    n = len(heights)
    count = 0
    for i in range(n):
        center = bin_lo + (i + 0.5) / n * (bin_hi - bin_lo)
        if lo <= center <= hi and heights[i] > 0:
            count += 1
    return count


# --------------------------------------------------------------------------- #
# Pure helper — bounded, deterministic, domain-agnostic
# --------------------------------------------------------------------------- #
def test_adaptive_rebin_none_for_wide_window():
    model = compute_histogram_float(np.linspace(0.0, 1.0, 4096, dtype=np.float32))
    # Full-width window is NOT narrow -> no re-binning.
    assert adaptive_rebin_bars(model, 0.0, 1.0) is None


def test_adaptive_rebin_none_without_retained_sample():
    # Synthetic model without a retained sample (legacy-compatible).
    model = {"bins": 512, "range": (0.0, 4.0), "bin_range": (0.0, 4.0),
             "log_counts": {"L": np.zeros(512)}}
    assert adaptive_rebin_bars(model, 0.68, 0.83) is None


def test_adaptive_rebin_non_negative_narrow_domain():
    model = _narrow_display_model()
    lo, hi = model["x_range"]

    result = adaptive_rebin_bars(model, lo, hi)
    assert result is not None
    heights, rlo, rhi, draw_overflow = result
    assert rlo == pytest.approx(lo, rel=0, abs=1e-9)
    assert rhi == pytest.approx(hi, rel=0, abs=1e-9)
    assert draw_overflow is False
    occupied = int((heights["L"] > 0).sum())
    assert occupied > 100, f"narrow re-bin not resolved: {occupied} occupied bins"


def test_adaptive_rebin_signed_narrow_domain():
    rng = np.random.default_rng(7)
    arr = rng.normal(-2.0, 1.0, size=(200, 200)).astype(np.float32)
    model = compute_scientific_histogram(arr)
    assert model["range"][0] < 0.0  # signed carrier

    result = adaptive_rebin_bars(model, -2.5, -1.5)
    assert result is not None
    heights, rlo, rhi, draw_overflow = result
    assert rlo == pytest.approx(-2.5, rel=0, abs=1e-9)
    assert rhi == pytest.approx(-1.5, rel=0, abs=1e-9)
    assert rlo < 0.0 and rhi < 0.0  # fully negative window
    assert draw_overflow is False
    assert int((heights["L"] > 0).sum()) > 50


def test_adaptive_rebin_bounded_and_deterministic():
    model = _narrow_display_model()
    lo, hi = model["x_range"]
    a = adaptive_rebin_bars(model, lo, hi)
    b = adaptive_rebin_bars(model, lo, hi)
    assert a is not None and b is not None
    for ch in a[0]:
        assert np.array_equal(a[0][ch], b[0][ch])
    # Operates on the retained bounded sample, never the full frame.
    assert model["samples"]["L"].size <= MAX_SAMPLE_PIXELS


def test_adaptive_rebin_returns_none_on_degenerate_window():
    model = _narrow_display_model()
    assert adaptive_rebin_bars(model, 0.5, 0.5) is None
    assert adaptive_rebin_bars(model, 200.0, 300.0) is None  # outside the domain


# --------------------------------------------------------------------------- #
# View integration — narrow zoom resolves the window (no sparse comb)
# --------------------------------------------------------------------------- #
def test_narrow_zoom_rebins_visible_window(qapp):
    model = _narrow_display_model()
    view = HistogramView()
    try:
        view.resize(420, 120)
        view.set_model(model)

        # Global bars restricted to the visible window are a sparse comb.
        assert _global_bins_in_window(model) <= 1

        # Zoom (manual) into the narrow robust window -> adaptive re-bin.
        view.auto_zoom_enabled = False
        view.zoom_histogram()
        heights, r_lo, r_hi, draw_overflow = view._bars_for_current_mode()
        assert r_lo == pytest.approx(0.68, rel=0, abs=1e-9)
        assert r_hi == pytest.approx(0.83, rel=0, abs=1e-9)
        assert draw_overflow is False
        assert int((heights["L"] > 0).sum()) > 100
    finally:
        view.deleteLater()


def test_default_and_non_narrow_view_keep_global_bars(qapp):
    model = _narrow_display_model()
    view = HistogramView()
    try:
        view.set_model(model)
        # Default (un-zoomed) view: exact global bars, byte-stable.
        heights, lo, hi, draw_overflow = view._bars_for_current_mode()
        assert heights is model["log_counts"]
        assert (lo, hi) == pytest.approx(model["bin_range"])
        assert draw_overflow is True

        # A NON-narrow manual window: still the exact global bars.
        view._view_mode = "manual"
        view._view_min, view._view_max = 0.0, 90.0  # 460 global bins visible
        heights2, _, _, _ = view._bars_for_current_mode()
        assert heights2 is model["log_counts"]
    finally:
        view.deleteLater()


def test_reset_after_zoom_restores_full_distribution(qapp):
    model = _narrow_display_model()
    view = HistogramView()
    try:
        view.set_model(model)
        view.zoom_histogram()  # manual narrow zoom
        rebinned, _, _, _ = view._bars_for_current_mode()
        assert rebinned is not model["log_counts"]

        view.reset_histogram_view()
        assert view._view_mode == "full"
        heights, lo, hi, draw_overflow = view._bars_for_current_mode()
        assert heights is model["full_log_counts"]
        assert (lo, hi) == pytest.approx(model["full_hist_range"])
        assert draw_overflow is False
    finally:
        view.deleteLater()


def test_repeated_zoom_reset_is_idempotent(qapp):
    model = _narrow_display_model()
    view = HistogramView()
    try:
        view.set_model(model)

        view.zoom_histogram()
        first_range = view.view_range
        first_bars = view._bars_for_current_mode()[0]

        view.reset_histogram_view()
        assert view._view_mode == "full"

        view.zoom_histogram()
        assert view.view_range == first_range
        again = view._bars_for_current_mode()[0]
        for ch in first_bars:
            assert np.array_equal(first_bars[ch], again[ch])

        view.reset_histogram_view()
        view.reset_histogram_view()
        assert view._view_mode == "full"
        assert view.view_range == view._full_range()
    finally:
        view.deleteLater()


# --------------------------------------------------------------------------- #
# X tick / axis-label formatting (negative domains)
# --------------------------------------------------------------------------- #
def test_axis_level_formatting_negative_domain():
    assert format_axis_level(-4.733) == "-4.73"
    assert format_axis_level(-0.5) == "-0.50"
    assert format_axis_level(0.5) == "0.50"
    assert format_axis_level(82.809) == "82.81"


# --------------------------------------------------------------------------- #
# compute_histogram_float: additive sample key, existing keys unchanged
# --------------------------------------------------------------------------- #
def test_display_model_gains_sample_and_counts_unchanged():
    arr = np.array([[0.0, 0.1], [0.25, 0.5]], dtype=np.float32)
    model = compute_histogram_float(arr)

    assert "samples" in model
    assert set(model["samples"]) == set(model["channels"])
    assert model["samples"]["L"].size == 4
    # Existing ratified keys are unchanged (ratified algorithm re-derivation).
    in_domain = np.asarray(arr, dtype=np.float64).ravel()
    counts, _ = np.histogram(in_domain, bins=512, range=(0.0, 1.0))
    assert np.array_equal(model["counts"]["L"], counts.astype(np.int64))
    assert model["bin_range"] == (0.0, 1.0)
