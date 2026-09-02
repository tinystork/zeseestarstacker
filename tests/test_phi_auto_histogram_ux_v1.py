"""PHI-AUTO-HISTOGRAM-UX-V1 — Auto Stretch analysis units + robust plot/bin domain.

Focused deterministic tests reproducing the two production-observed issues and
proving the corrections, without touching science/engine state:

1. **Auto Stretch capped-policy regression** — the old sample excluded every
   value >= 1 and clipped the white point at the display-window top ``1.0``,
   so a frame with a meaningful (dense) bright tail above ``1`` stayed
   saturated.  The corrected estimator operates in the preserved analysis
   units, keeps finite values above ``1`` in the sample and bounds the final
   separation clip by the *robust* analysis high ``D = max(1.0, p99.5(S))``:
   with a meaningful tail ``WP > 1`` is selected (non-saturated highlight
   structure recovered by the fixed-reference analysis-unit stretch); with
   only an isolated extreme outlier ``WP`` is never expanded to it; in-window
   ``[0, 1]`` buffers stay bit-identical to the ratified §5.5 algorithm.
2. **Histogram sparse-extreme-tail binning** — 512 fixed bins spread over the
   full finite analysis range degrade to a few widely spaced vertical spikes
   when a sparse extreme tail exists (e.g. maxima ~282 while the robust
   window is ~0.25-2.36).  The corrected model bins over an explicit robust
   **plot/bin range** ``bin_range`` (full analysis upper when the top is dense
   or the sample small; the robust p99.5 top otherwise), reports the tail
   above it as truthful per-channel ``overflow`` (full range + stats ``max``
   unchanged), keeps manual BP/WP over the full analysis domain and paints the
   bars dense inside the auto/default visible window (paint/model assertion
   included).

Pure-float model assertions need no Qt; the widget/paint assertions run
offscreen.  ``QT_QPA_PLATFORM=offscreen`` is set defensively before any
``QApplication`` is created.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QImage, QPainter

from seestar.gui_qt import create_application
from seestar.gui_qt.histogram_view import HistogramView
from seestar.gui_qt.preview_adjust import render_analysis_display
from seestar.gui_qt.preview_analysis import (
    AUTO_STRETCH_DEFAULTS,
    ANCHOR_SEP,
    HISTOGRAM_BINS,
    compute_auto_stretch_float,
    compute_histogram_float,
)


@pytest.fixture(scope="session")
def qapp():
    app = create_application([])
    from PySide6.QtWidgets import QApplication

    assert app is QApplication.instance()
    return app


# ---------------------------------------------------------------------------
# Fixtures: deterministic analysis-domain buffers
# ---------------------------------------------------------------------------

def _sky_plus_nebula(seed: int = 11, size: int = 400, tail: float = 2.4,
                     frac: float = 0.30):
    """Deterministic mono analysis buffer: dim sky + a dense bright 'nebula'.

    The nebula occupies ``frac`` of the pixels with values in ``(1.0, tail]``
    — a *meaningful* (dense) bright tail above the legacy display-window top,
    exactly the Option-A/WB headroom regime the user evidence showed.
    """
    rng = np.random.default_rng(seed)
    buf = rng.uniform(0.2, 0.7, size=(size, size)).astype(np.float64)
    neb = rng.random((size, size)) < frac
    buf[neb] = rng.uniform(1.0, tail, size=int(neb.sum())).astype(np.float64)
    return buf


def _sparse_extreme_mono(seed: int = 5, size: int = 400):
    """Dense bulk uniform in (0.25, 2.36) + 3 isolated extreme finite px."""
    rng = np.random.default_rng(seed)
    buf = rng.uniform(0.25, 2.36, size=(size, size)).astype(np.float64)
    for k, v in enumerate((40.0, 100.0, 282.0)):
        buf[k // size, k % size] = v
    return buf


def _sparse_extreme_rgb(seed: int = 6, size: int = 140):
    """RGB: dense channels to ~2.3, red channel with a sparse extreme px."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.2, 2.3, size=(size, size, 3)).astype(np.float64)
    base[0, 0, 0] = 100.0
    base[1, 1, 0] = 282.0
    return base


def _legacy_reference_auto_stretch(vals: np.ndarray, sep: float = ANCHOR_SEP):
    """Inline reference implementation of the *old* (capped) §5.5 float spec.

    Sample = finite values in ``(0, 1)`` (headroom above 1 discarded), final
    clip at the display-window top ``1.0``.  Used to prove legacy parity of
    the corrected function on in-window buffers and to demonstrate the
    capped-policy regression on headroom buffers.
    """
    flat = vals.ravel()
    S = flat[np.isfinite(flat) & (flat > 0.0) & (flat < 1.0)]
    if S.size < 20:
        return AUTO_STRETCH_DEFAULTS
    p005 = float(np.percentile(S, 0.5))
    p60 = float(np.percentile(S, 60.0))
    p995 = float(np.percentile(S, 99.5))
    B = S[S <= p60]
    if B.size == 0 or not np.all(np.isfinite(B)):
        bp, wp = p005, p995
    else:
        bg = float(np.median(B))
        mad = float(np.median(np.abs(B - bg)))
        sigma = 1.4826 * mad
        if sigma == 0.0 or not np.isfinite(sigma):
            bp, wp = p005, p995
        else:
            bp = max(p005, bg - 2.8 * sigma)
            wp = max(p995, bg + 8.0 * sigma)
    bp = float(np.clip(bp, 0.0, 1.0 - sep))
    wp = float(np.clip(wp, bp + sep, 1.0))
    return (bp, wp)


def _qimage_channel(img: QImage, channel: int) -> np.ndarray:
    """Read a QImage back into a uint8 (H, W, 3) array (like the PHI suite)."""
    import seestar.gui_qt.preview_adjust as _pa

    arr = _pa._image_to_array(_pa._load_numpy(), img)
    assert arr is not None
    if arr.ndim == 2:
        return arr
    return arr[..., channel]


# ---------------------------------------------------------------------------
# (1) Auto Stretch — capped-policy regression, analysis units, robust top
# ---------------------------------------------------------------------------

def test_auto_stretch_capped_policy_regression_selects_wp_above_one():
    """A dense (meaningful) bright tail above 1 must produce WP > 1.

    The old policy discarded every value >= 1 and clipped WP at 1.0; the
    corrected analysis-unit estimator keeps the headroom in the sample and
    bounds the clip by the robust analysis high, so the bright structure is
    no longer hard-saturated at the display window top.
    """
    buf = _sky_plus_nebula(seed=11)
    assert float(np.max(buf)) > 1.0
    # The old capped reference saturates at the display-window top.
    old_bp, old_wp = _legacy_reference_auto_stretch(buf)
    assert old_wp <= 1.0 + 1e-12

    bp, wp = compute_auto_stretch_float(buf)
    assert bp < wp
    assert bp >= 0.0
    # Meaningful dense tail -> WP above 1, but bounded by the robust top
    # (p99.5 of the sample) — never the raw max.
    assert wp > 1.0 + 0.2, f"expected WP > 1 for a meaningful tail, got {wp}"
    assert wp < float(np.max(buf))
    # Ordered/quantized within the analysis domain (no hidden [0,1] cap).
    assert wp <= max(1.0, float(np.percentile(buf, 99.5))) + 1e-9


def test_auto_stretch_render_recovers_non_saturated_highlights(qapp):
    """Render-level witness: WP > 1 maps mid-tail analysis values below 255.

    With the old capped pair the same analysis value (a preserved headroom
    pixel in the bright structure) renders saturated white; with the
    corrected pair it renders at its linear analysis-unit level.
    """
    buf = _sky_plus_nebula(seed=12, size=200)
    bp, wp = compute_auto_stretch_float(buf)
    assert wp > 1.0

    old_bp, old_wp = _legacy_reference_auto_stretch(buf)
    img_new = render_analysis_display(buf, stretch="linear",
                                      black_point=bp, white_point=wp)
    img_old = render_analysis_display(buf, stretch="linear",
                                      black_point=old_bp, white_point=old_wp)
    assert img_new is not None and img_old is not None
    new_arr = _qimage_channel(img_new, 0)
    old_arr = _qimage_channel(img_old, 0)

    # A preserved headroom analysis value well inside the tail (1 < x < wp).
    band = (buf > 1.2) & (buf < float(wp) - 0.3)
    assert band.any(), "fixture must contain headroom-band pixels"
    idx = np.argwhere(band)[0]
    y, x = int(idx[0]), int(idx[1])
    x_val = float(buf[y, x])
    # Old capped policy: x >= 1 > wp_old <= 1 -> saturated white.
    assert old_arr[y, x] == 255
    # Corrected policy: linear analysis-unit mapping, not saturated.
    expected = int(round((x_val - bp) / (wp - bp) * 255.0))
    assert 0 < new_arr[y, x] < 255
    assert abs(int(new_arr[y, x]) - expected) <= 2, (
        new_arr[y, x], expected
    )


def test_auto_stretch_isolated_extreme_outlier_never_expands_wp():
    """An isolated extreme outlier must not pull the white point to it/max."""
    base = _sky_plus_nebula(seed=13, size=400)
    with_outlier = base.copy()
    with_outlier[0, 0] = 1000.0
    with_outlier[1, 1] = 1000.0
    bp_a, wp_a = compute_auto_stretch_float(base)
    bp_b, wp_b = compute_auto_stretch_float(with_outlier)
    # Outlier-robust: the two extreme pixels do not move the estimate
    # materially (percentiles/MAD are insensitive to isolated outliers).
    assert bp_a == bp_b
    assert wp_a == pytest.approx(wp_b, abs=1e-3)
    # The white point is never expanded to the outlier/max.
    assert wp_a < 1000.0
    assert wp_a <= max(1.0, float(np.percentile(with_outlier, 99.5))) + 1e-9


def test_auto_stretch_legacy_in_window_bit_identical():
    """In-window [0,1] buffers keep the ratified §5.5 output exactly."""
    rng = np.random.default_rng(21)
    cases = []
    cases.append(rng.uniform(0.1, 0.9, size=(200, 200)).astype(np.float64))
    # Sky + stars within the window.
    sky = rng.uniform(0.15, 0.35, size=(200, 200)).astype(np.float64)
    stars = rng.random((200, 200)) < 0.01
    sky[stars] = rng.uniform(0.8, 0.99, size=int(stars.sum()))
    cases.append(sky)
    cases.append(np.clip(np.linspace(0.0, 1.0, 40_000), 1e-6, 1 - 1e-6)
                 .reshape(200, 200))
    for arr in cases:
        assert float(np.max(arr)) <= 1.0
        got = compute_auto_stretch_float(arr)
        ref = _legacy_reference_auto_stretch(arr)
        assert got == pytest.approx(ref, abs=1e-12), (got, ref)
        bp, wp = got
        assert 0.0 <= bp <= 1.0 - ANCHOR_SEP
        assert bp + ANCHOR_SEP <= wp <= 1.0


def test_auto_stretch_defaults_and_exclusions_unchanged():
    """Exact-0/1-only and tiny samples still fall back deterministically."""
    arr01 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    assert compute_auto_stretch_float(arr01) == AUTO_STRETCH_DEFAULTS
    tiny = np.full((4, 4), 0.5, dtype=np.float64)
    assert compute_auto_stretch_float(tiny) == AUTO_STRETCH_DEFAULTS


# ---------------------------------------------------------------------------
# (2) Histogram — robust plot/bin range with sparse extreme tails
# ---------------------------------------------------------------------------

def test_histogram_sparse_extreme_tail_dense_bins_and_truthful_overflow():
    """The production defect: 512 bins over full [0, ~282] -> few spikes.

    With the corrected bin/plot domain the robust visible window holds dense
    occupied bins, the sparse tail above the plot top is reported as truthful
    overflow metadata (full range + stats max unchanged), and the counts +
    overflow still describe the exact same in-domain population.
    """
    buf = _sparse_extreme_mono(seed=5)
    model = compute_histogram_float(buf)
    assert model is not None
    assert model["channels"] == ["L"]

    analysis_upper = float(model["full_range"][1])
    assert analysis_upper == pytest.approx(282.0)  # full-range truth kept
    bin_hi = float(model["bin_range"][1])
    x_lo, x_hi = model["x_range"]

    # The robust window is fully inside the binned domain.
    assert x_hi <= bin_hi + 1e-9
    # The bin high is far below the sparse extreme (that is the fix).
    assert bin_hi < analysis_upper / 10.0
    # The bin high sits at the robust top of the dense bulk (>= ~2.3).
    assert bin_hi >= 2.3 - 1e-9

    # Truthful overflow: exactly the in-domain values above the plot top.
    in_domain = buf[np.isfinite(buf) & (buf >= 0.0)]
    expected_overflow = int(np.count_nonzero(in_domain > bin_hi))
    assert expected_overflow > 0  # the sparse extremes are above the plot top
    assert int(model["overflow"]["L"]) == expected_overflow
    assert int(model["overflow_total"]) == expected_overflow

    # Counts + overflow == the exact same in-domain population.
    assert int(model["counts"]["L"].sum()) + expected_overflow == in_domain.size

    # The per-channel stats still see the full analysis truth (max 282).
    assert model["stats"]["L"]["max"] == pytest.approx(282.0)

    # Dense bin occupancy inside the auto/default visible window: the bins
    # whose centres lie in the robust window are almost all occupied (uniform
    # dense bulk) — before the fix only ~4 bins existed in that window.
    bins = model["counts"]["L"]
    bw = bin_hi / HISTOGRAM_BINS
    lo_idx = max(0, int(np.floor(x_lo / bw)) - 1)
    hi_idx = min(HISTOGRAM_BINS - 1, int(np.ceil(x_hi / bw)) + 1)
    occupied_in_window = int(np.count_nonzero(bins[lo_idx:hi_idx + 1] > 0))
    window_bins = hi_idx - lo_idx + 1
    assert window_bins >= 100, (window_bins, bin_hi)
    assert occupied_in_window >= window_bins * 0.9, (
        occupied_in_window, window_bins
    )

    # Explicit model roles: bin_range <= full_range == range (analysis).
    assert model["range"] == model["full_range"]
    assert model["bin_range"][0] == 0.0
    assert model["bin_range"][1] <= model["full_range"][1]


def test_histogram_rgb_sparse_extreme_overflow_per_channel():
    """Per-channel overflow is truthful on RGB with channel-local extremes."""
    buf = _sparse_extreme_rgb(seed=6)
    model = compute_histogram_float(buf)
    assert model is not None
    assert model["channels"] == ["R", "G", "B"]
    assert float(model["full_range"][1]) == pytest.approx(282.0)
    bin_hi = float(model["bin_range"][1])
    assert bin_hi < 282.0

    for ch, c in zip(model["channels"], range(3)):
        plane = buf[..., c]
        in_domain = plane[np.isfinite(plane) & (plane >= 0.0)]
        expected = int(np.count_nonzero(in_domain > bin_hi))
        assert int(model["overflow"][ch]) == expected, ch
        assert int(model["counts"][ch].sum()) + expected == in_domain.size
    assert int(model["overflow_total"]) == sum(
        int(model["overflow"][ch]) for ch in model["channels"]
    )
    # Red channel carries the extremes -> its overflow is nonzero.
    assert int(model["overflow"]["R"]) >= 2
    # Stats still see the extremes per channel.
    assert model["stats"]["R"]["max"] == pytest.approx(282.0)
    assert model["stats"]["G"]["max"] == pytest.approx(
        float(np.max(buf[..., 1]))
    )


def test_histogram_dense_top_keeps_full_range_zero_overflow():
    """A genuinely dense top (population really reaches its max) keeps the
    full analysis range binning — bit-identical to the legacy/R3 model."""
    rng = np.random.default_rng(7)
    buf = rng.uniform(0.2, 2.5, size=(300, 300)).astype(np.float64)
    model = compute_histogram_float(buf)
    assert model is not None
    assert model["bin_range"] == model["full_range"] == model["range"]
    assert float(model["bin_range"][1]) == pytest.approx(
        max(1.0, float(np.max(buf)))
    )
    assert int(model["overflow_total"]) == 0
    assert int(model["counts"]["L"].sum()) == buf.size
    # Dense occupancy across the whole plot domain.
    occupied = int(np.count_nonzero(model["counts"]["L"] > 0))
    assert occupied >= HISTOGRAM_BINS * 0.9


def test_histogram_small_sample_and_no_headroom_full_range_parity():
    """Small in-domain samples and no-headroom buffers stay bit-identical:
    the sparse-tail cut only applies to large samples (> 512 values)."""
    hdr = np.array([[0.2, 0.7, 1.4], [2.6, 0.05, 3.5]], dtype=np.float64)
    m = compute_histogram_float(hdr)
    assert m["bin_range"] == m["full_range"] == m["range"] == (0.0, 3.5)
    assert int(m["counts"]["L"].sum()) == 6
    assert int(m["overflow_total"]) == 0

    # No headroom at all: legacy [0, 1] binning, all values counted.
    rng = np.random.default_rng(8)
    inwin = rng.uniform(0.0, 1.0, size=(300, 300)).astype(np.float64)
    m2 = compute_histogram_float(inwin)
    assert m2["bin_range"] == m2["full_range"] == m2["range"] == (0.0, 1.0)
    assert int(m2["overflow_total"]) == 0
    assert int(m2["counts"]["L"].sum()) == inwin.size


# ---------------------------------------------------------------------------
# (3) Widget — bar domain, overflow marker, zoom/reset coherence, retention
# ---------------------------------------------------------------------------

def _render_view_to_image(view: HistogramView) -> QImage:
    """Offscreen-render a HistogramView into an ARGB32 QImage."""
    img = QImage(view.size(), QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 255))
    painter = QPainter(img)
    view.render(painter, QPoint(0, 0))
    painter.end()
    return img


def _model_from(buf) -> dict:
    model = compute_histogram_float(buf)
    assert model is not None
    return model


def test_histogram_view_uses_bin_range_and_exposes_overflow(qapp):
    """set_model adopts the model bin range for the plotted bars and exposes
    the overflow total; markers stay in the full analysis domain."""
    buf = _sparse_extreme_mono(seed=5)
    model = _model_from(buf)
    view = HistogramView()
    try:
        view.resize(420, 120)
        view.set_model(model)
        assert view.model is model
        assert view.bin_range == model["bin_range"]
        assert view.overflow_total == int(model["overflow_total"]) > 0
        # The view's analysis range (zoom/reset/marker domain) is unchanged.
        assert view._model_range == model["range"]
        # Marker domain (set by the owner) is the analysis grid upper: a WP
        # above the plotted bin top is a first-class retained marker.
        analysis_upper = float(model["range"][1])
        view.set_analysis_domain(analysis_upper)
        wp = float(model["bin_range"][1]) + 0.5
        assert wp < analysis_upper
        view.set_range(0.1, wp)
        assert view.white_point == pytest.approx(wp)  # not clamped to bin top
        assert view.black_point < view.white_point
    finally:
        view.deleteLater()


def test_histogram_view_dense_bars_inside_visible_window_paint(qapp):
    """Paint/model assertion: bars are dense inside the robust visible window
    (many occupied columns) instead of a few widely spaced spikes."""
    buf = _sparse_extreme_mono(seed=5)
    model = _model_from(buf)
    view = HistogramView()
    try:
        view.resize(420, 120)
        view.set_model(model)
        bin_hi = float(model["bin_range"][1])
        assert bin_hi < float(model["full_range"][1]) / 10.0

        # Zoom to the robust visible window (what the user actually views).
        view.auto_zoom_enabled = True
        view.zoom_histogram()
        rect = view._plot_rect()
        img = _render_view_to_image(view)
        occupied_cols = 0
        # A column counts if any sampled row inside the bar area is lit.
        rows = {int(rect.bottom()) - 2, int(rect.top() + rect.height() * 0.5)}
        for x in range(int(rect.left()), int(rect.right()) + 1):
            lit = any(
                img.pixelColor(x, y).red() + img.pixelColor(x, y).green()
                + img.pixelColor(x, y).blue() > 0
                for y in rows
            )
            if lit:
                occupied_cols += 1
        # The robust window previously held only a handful of bin columns.
        assert occupied_cols >= 120, occupied_cols
    finally:
        view.deleteLater()


def test_histogram_view_overflow_marker_and_reset_coherence(qapp):
    """FRP-H1 marker coherence: FULL mode (reset) genuinely shows the full
    distribution — the tail bars are painted and NO overflow marker is drawn
    (nothing is un-binned); a non-FULL window that exposes the plot-top
    boundary paints the overflow marker there; without overflow no marker is
    painted in any mode."""
    buf_sparse = _sparse_extreme_mono(seed=5)
    model = _model_from(buf_sparse)

    rng = np.random.default_rng(9)
    buf_dense = rng.uniform(0.25, 2.36, size=(300, 300)).astype(np.float64)
    dense = _model_from(buf_dense)
    assert int(dense["overflow_total"]) == 0

    upper = float(model["range"][1])
    bin_hi = float(model["bin_range"][1])
    assert bin_hi < upper  # genuine sparse extreme tail

    view = HistogramView()
    view2 = HistogramView()
    try:
        view.resize(420, 120)
        # FULL mode (reset): window = the full analysis range, the full-domain
        # tail bars ARE painted near the true extreme, and the overflow marker
        # is NOT drawn anywhere in the plot (the tail is genuinely binned).
        view.set_model(model)
        view.reset_histogram_view()
        assert view.view_range == (0.0, pytest.approx(upper))
        assert view.overflow_total > 0
        assert bin_hi <= view.view_range[1]
        img = _render_view_to_image(view)
        rect = view._plot_rect()
        assert not _image_contains_marker(img, rect), (
            "FULL mode must not paint the overflow marker"
        )
        # The full-domain bar at the true analysis maximum (right edge of the
        # domain) is genuinely lit near the plot bottom.
        bins = int(model["bins"])
        tail_center = (bins - 0.5) / bins * upper
        tail_x = int(round(view._level_to_x(tail_center)))
        bottom_y = int(rect.bottom()) - 2
        assert any(
            img.pixelColor(x, bottom_y).red() > 150
            for x in range(tail_x - 4, tail_x + 5)
        ), "FULL-mode tail bar at the true max is not painted"

        # Non-FULL mode: a view window that exposes the plot-top boundary
        # (robust bars still placed in ``bin_range``, no reset — the mode
        # stays ``default``) paints the overflow marker exactly at ``bin_hi``.
        view2.resize(420, 120)
        view2.set_model(model)
        assert view2.view_range == (0.0, 1.0)  # default window (no reset)
        view2.set_view_range(0.0, upper)  # widen the window, NOT the mode
        heights, lo, hi, draw_overflow = view2._bars_for_current_mode()
        assert draw_overflow is True
        assert (lo, hi) == pytest.approx((0.0, bin_hi))
        assert view2.view_range == (0.0, pytest.approx(upper))
        img3 = _render_view_to_image(view2)
        x_boundary = int(round(view2._level_to_x(bin_hi)))
        color = img3.pixelColor(x_boundary, int(view2._plot_rect().top()) + 1)
        assert color == _marker_color(), color

        # Dense model: no overflow -> no marker anywhere.
        view.set_model(dense)
        assert view.overflow_total == 0
        view.reset_histogram_view()
        img2 = _render_view_to_image(view)
        rect = view._plot_rect()
        y = int(rect.top() + (rect.bottom() - rect.top()) * 0.5)
        found = False
        for x in range(int(rect.left()), int(rect.right()) + 1):
            c = img2.pixelColor(x, y)
            if c == _marker_color():
                found = True
                break
        assert not found, "overflow marker must not be painted without overflow"
    finally:
        view.deleteLater()
        view2.deleteLater()


def _image_contains_marker(img: QImage, rect) -> bool:
    """True when an exact overflow-marker-colour pixel is painted in the plot.

    The marker is a full-height vertical line, so a handful of rows across the
    whole plot width is a complete presence probe.
    """
    rows = {
        int(rect.top()),
        int(rect.top()) + 8,
        int(rect.top() + rect.height() * 0.5),
        int(rect.bottom()) - 8,
    }
    for y in rows:
        for x in range(int(rect.left()), int(rect.right()) + 1):
            if img.pixelColor(x, y) == _marker_color():
                return True
    return False


def _marker_color():
    import seestar.gui_qt.histogram_view as hv

    return hv._OVERFLOW_MARKER_COLOR


def test_histogram_view_zoom_stays_inside_binned_domain(qapp):
    """Auto/manual zoom to the robust x-range stays within the binned domain
    (no zooming into the empty sparse-tail region by default)."""
    buf = _sparse_extreme_mono(seed=5)
    model = _model_from(buf)
    view = HistogramView()
    try:
        view.resize(420, 120)
        view.set_model(model)
        view.auto_zoom_enabled = True
        view.zoom_histogram()
        lo, hi = view.view_range
        assert lo >= 0.0
        assert hi <= float(model["bin_range"][1]) + 1e-9
        assert view.view_range == pytest.approx(model["x_range"])
    finally:
        view.deleteLater()


def test_histogram_model_roles_not_mislabeled(qapp):
    """Range/full_range (analysis/control), bin_range (plot) and x_range
    (viewport) are distinct explicit roles; an analysis histogram is never
    relabelled as a display histogram."""
    buf = _sparse_extreme_mono(seed=5)
    model = _model_from(buf)
    assert model["range"] == model["full_range"]  # analysis/control truth
    assert model["bin_range"][0] == 0.0
    assert model["bin_range"][1] < model["full_range"][1]
    assert model["x_range"][0] < model["x_range"][1]
    assert model["x_range"][1] <= model["bin_range"][1] + 1e-9
    assert set(model["overflow"].keys()) == set(model["channels"])


def test_histogram_overflow_status_annotation(qapp):
    """The status-line overflow annotation truthfully reports the tail above
    the plotted top and stays empty for no-overflow models."""
    from seestar.gui_qt.histogram_view import format_histogram_overflow

    sparse = _model_from(_sparse_extreme_mono(seed=5))
    note = format_histogram_overflow(sparse)
    assert note.startswith(" (+")
    assert "px above plot top" in note
    assert str(int(sparse["overflow_total"])) in note

    rng = np.random.default_rng(9)
    dense = _model_from(
        rng.uniform(0.25, 2.36, size=(300, 300)).astype(np.float64)
    )
    assert int(dense["overflow_total"]) == 0
    assert format_histogram_overflow(dense) == ""
    # Legacy-format models (no overflow keys) keep the empty annotation.
    assert format_histogram_overflow({"stats": {}}) == ""
