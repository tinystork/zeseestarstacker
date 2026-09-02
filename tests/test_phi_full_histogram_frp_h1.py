"""FRP-H1 — dual-domain histogram (robust bin range + full domain) and
persistent view-mode semantics (AUTO / FULL / MANUAL / DEFAULT).

Focused deterministic tests proving the final-release polish:

1. **Full-domain histogram** — ``compute_histogram_float`` now also bins the
   *complete* sampled distribution: 512 ``full_counts``/``full_log_counts``
   bins over the full analysis range ``(0.0, upper)`` (the true maximum), from
   the **exact same** in-domain sample as the robust ``counts`` (no second
   image traversal), so per channel ``sum(counts) + overflow == sampled_count
   AND sum(full_counts) == sampled_count``.  With a sparse extreme tail the
   robust bins stay dense over the bulk while the full bins carry the real
   tail distribution up to the true max; with a dense high-end population the
   two histograms are degenerate-identical.
2. **Persistent view modes** — ``HistogramView`` drives its view from an
   explicit ``_view_mode`` (``"default"`` / ``"auto"`` / ``"full"`` /
   ``"manual"``): AUTO re-zooms to each new model's robust range; FULL (set by
   ``reset_histogram_view`` / ``reset_zoom``) is persistent across new models
   (view + bars follow the new full domain); MANUAL freezes a robust zoom that
   survives domain growth and is reconciled/clamped on domain shrink.
3. **Full-mode painting** — Reset/Full genuinely paints the full-domain bars
   over the full domain (not just a widened axis around robust-only bars) and
   draws no overflow marker (nothing is un-binned in FULL mode).

Pure-float model assertions need no Qt; widget/paint assertions run offscreen.
``QT_QPA_PLATFORM=offscreen`` is set defensively before any ``QApplication``
is created.
"""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor, QImage, QPainter

from seestar.gui_qt import create_application
from seestar.gui_qt.histogram_view import HistogramView
from seestar.gui_qt.preview_analysis import (
    HISTOGRAM_BINS,
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

def _mono(seed: int, size: int = 120, bulk_hi: float = 2.0,
          extremes: tuple = ()) -> np.ndarray:
    """Deterministic mono analysis buffer: uniform dense bulk + sparse tail.

    ``extremes`` are a handful of isolated finite values far above the bulk
    (a sparse HDR tail), deterministically placed on the first rows.
    """
    rng = np.random.default_rng(seed)
    buf = rng.uniform(0.05, bulk_hi, size=(size, size)).astype(np.float64)
    for k, v in enumerate(extremes):
        buf[k // size, k % size] = v
    return buf


def _rgb(seed: int, size: int = 120, bulk_hi: float = 2.0,
         red_extremes: tuple = (), green_extremes: tuple = ()) -> np.ndarray:
    """Deterministic RGB analysis buffer with channel-local sparse extremes."""
    rng = np.random.default_rng(seed)
    buf = rng.uniform(0.05, bulk_hi, size=(size, size, 3)).astype(np.float64)
    for k, v in enumerate(red_extremes):
        buf[k // size, k % size, 0] = v
    for k, v in enumerate(green_extremes):
        buf[k // size, k % size, 1] = v
    return buf


def _in_domain(np_: Any, plane: np.ndarray) -> np.ndarray:
    """Mirror of the analysis sample filter (finite, >= analysis floor)."""
    return plane[np.isfinite(plane) & (plane >= 0.0)]


def _bin_index(value: float, upper: float) -> int:
    """Index of the 512-bin over ``(0, upper)`` histogram holding ``value``."""
    if value >= upper:
        return HISTOGRAM_BINS - 1
    return min(HISTOGRAM_BINS - 1, int(value / upper * HISTOGRAM_BINS))


def _render_view_to_image(view: HistogramView) -> QImage:
    """Offscreen-render a HistogramView into an ARGB32 QImage."""
    img = QImage(view.size(), QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 255))
    painter = QPainter(img)
    view.render(painter, QPoint(0, 0))
    painter.end()
    return img


# ---------------------------------------------------------------------------
# (1) Sparse HDR tail: dense robust bins + real full-domain tail distribution
# ---------------------------------------------------------------------------

def test_sparse_tail_full_histogram_reaches_true_max_with_real_tail_bins():
    """A sparse extreme tail must NOT vanish: the robust histogram stays dense
    over the bulk (bin_range high near the bulk top) while the full-domain
    histogram reaches the true max and its tail bins carry the real injected
    population; per-channel count conservation holds on both histograms."""
    buf = _mono(seed=5, size=200, bulk_hi=2.0, extremes=(22.0, 40.0, 100.0, 282.0))
    model = compute_histogram_float(buf)
    assert model is not None
    assert model["channels"] == ["L"]

    upper = float(model["full_range"][1])
    assert upper == pytest.approx(282.0)  # the true analysis maximum
    assert model["range"] == model["full_range"]
    # full_hist_range == the full analysis domain == (0.0, upper).
    assert tuple(model["full_hist_range"]) == pytest.approx((0.0, 282.0))
    assert tuple(model["full_hist_range"]) == pytest.approx(model["range"])

    bin_hi = float(model["bin_range"][1])
    # Robust bin high sits near the dense bulk top, far below the sparse max.
    assert bin_hi >= 1.9 and bin_hi < upper / 10.0

    in_domain = _in_domain(np, buf)
    n = in_domain.size
    assert n > HISTOGRAM_BINS  # large enough for the sparse-tail cut to apply

    # Conservation on BOTH histograms over the exact same population.
    counts = model["counts"]["L"]
    full_counts = model["full_counts"]["L"]
    assert int(counts.sum()) + int(model["overflow"]["L"]) == n
    assert int(model["overflow_total"]) == int(model["overflow"]["L"])
    assert int(full_counts.sum()) == n
    # Overflow values really are re-binned in the full histogram: the robust
    # counts alone hold n - overflow <= n values, the full histogram holds n.
    assert int(counts.sum()) < int(full_counts.sum())

    # The tail bins carry the real injected distribution: every extreme value
    # lands in a full bin at its own level (isolated tail bins, not a dropped
    # overflow blob), and bins beyond the bulk top hold exactly the tail.
    for v in (22.0, 40.0, 100.0, 282.0):
        idx = _bin_index(v, upper)
        assert int(full_counts[idx]) >= 1, (v, idx)
    bulk_top_bin = _bin_index(2.0, upper)  # bulk ends at 2.0
    assert int(full_counts[:bulk_top_bin + 1].sum()) == n - 4  # bulk only
    # The extreme exactly at the analysis maximum is in the last bin (right
    # edge inclusive — never dropped).
    assert int(full_counts[HISTOGRAM_BINS - 1]) >= 1

    # Robust bins are dense inside the auto/default window (not a few spikes).
    bw = bin_hi / HISTOGRAM_BINS
    x_lo, x_hi = model["x_range"]
    lo_idx = max(0, int(np.floor(x_lo / bw)) - 1)
    hi_idx = min(HISTOGRAM_BINS - 1, int(np.ceil(x_hi / bw)) + 1)
    window_bins = hi_idx - lo_idx + 1
    occupied = int(np.count_nonzero(counts[lo_idx:hi_idx + 1] > 0))
    assert window_bins >= 100
    assert occupied >= window_bins * 0.9


def test_dense_high_end_full_histogram_identical_to_robust():
    """A genuinely dense high-end population (the max is part of the dense
    body) keeps ``bin_hi == upper``: the full histogram is degenerate-
    identical to the robust one (same bins over the same domain) — documented
    behaviour, zero divergence."""
    rng = np.random.default_rng(7)
    buf = rng.uniform(0.2, 2.5, size=(200, 200)).astype(np.float64)
    model = compute_histogram_float(buf)
    assert model is not None
    assert model["bin_range"] == model["full_range"] == model["range"]
    assert tuple(model["full_hist_range"]) == pytest.approx(model["range"])
    assert int(model["overflow_total"]) == 0
    for ch in model["channels"]:
        assert np.array_equal(model["full_counts"][ch], model["counts"][ch])
        assert np.array_equal(
            model["full_log_counts"][ch], model["log_counts"][ch]
        )
        assert int(model["full_counts"][ch].sum()) == buf.size


# ---------------------------------------------------------------------------
# (3) RGB channel-local extremes: each channel's full bins reach its own max
# ---------------------------------------------------------------------------

def test_rgb_channel_extremes_full_histogram_per_channel_truth():
    """R max >> G max >> B max: the full-domain histogram of each channel
    reaches that channel's own true max (R bins up to the global upper, G and
    B end at their own tops), per-channel conservation holds, and the stats
    stay truthful on every channel."""
    buf = _rgb(
        seed=6, size=160, bulk_hi=2.0,
        red_extremes=(22.0, 60.0, 150.0, 300.0),
        green_extremes=(22.0, 40.0),
    )
    model = compute_histogram_float(buf)
    assert model is not None
    assert model["channels"] == ["R", "G", "B"]

    upper = float(model["full_range"][1])
    assert upper == pytest.approx(300.0)  # the R channel carries the global max
    assert tuple(model["full_hist_range"]) == pytest.approx((0.0, 300.0))
    bin_hi = float(model["bin_range"][1])
    assert bin_hi < upper / 10.0  # sparse tail present

    for ch, c in zip(model["channels"], range(3)):
        plane = buf[..., c]
        in_domain = _in_domain(np, plane)
        counts = model["counts"][ch]
        full_counts = model["full_counts"][ch]
        # Per-channel conservation on both histograms.
        assert int(counts.sum()) + int(model["overflow"][ch]) == in_domain.size
        assert int(full_counts.sum()) == in_domain.size
        # Stats truth: per-channel max == the true per-channel analysis max.
        assert model["stats"][ch]["max"] == pytest.approx(
            float(np.max(in_domain))
        )
        # Full bins reach this channel's own true max and stop after it.
        ch_max = float(np.max(in_domain))
        last_idx = _bin_index(ch_max, upper)
        assert int(full_counts[last_idx]) >= 1
        assert int(full_counts[last_idx + 1:].sum()) == 0, ch

    # Channel separation in the tail: only R has values beyond G's top (40);
    # bins above the G top hold exactly the three R extremes 60/150/300.
    g_top_bin = _bin_index(40.0, upper)
    assert int(model["full_counts"]["G"][g_top_bin + 1:].sum()) == 0
    assert int(model["full_counts"]["B"][g_top_bin + 1:].sum()) == 0
    assert int(model["full_counts"]["R"][g_top_bin + 1:].sum()) == 3
    # The R extreme at the analysis upper sits in the last (edge-inclusive) bin.
    assert int(model["full_counts"]["R"][HISTOGRAM_BINS - 1]) >= 1
    # Per-channel overflow is exactly the in-domain population above the
    # robust bin high (dense-bulk values above the p99.5 cut included).
    for ch, c in zip(model["channels"], range(3)):
        plane = buf[..., c]
        in_domain = _in_domain(np, plane)
        expected = int(np.count_nonzero(in_domain > bin_hi))
        assert int(model["overflow"][ch]) == expected, ch
    # Sanity: every injected R extreme is part of the overflow tail.
    assert int(model["overflow"]["R"]) >= 4


# ---------------------------------------------------------------------------
# (4) View lifecycle: AUTO / FULL / MANUAL across successive models
# ---------------------------------------------------------------------------

@pytest.fixture()
def _batches():
    """Three deterministic mono batches with distinct full-domain maxima."""
    small = _mono(seed=11, size=96, bulk_hi=2.0, extremes=(8.0, 15.0, 22.0))
    big = _mono(seed=12, size=96, bulk_hi=4.0, extremes=(40.0, 100.0, 315.0))
    low = _mono(seed=13, size=96, bulk_hi=1.1, extremes=())
    models = [compute_histogram_float(b) for b in (small, big, low)]
    assert all(m is not None for m in models)
    m_small, m_big, m_low = models
    assert float(m_small["range"][1]) == pytest.approx(22.0)
    assert float(m_big["range"][1]) == pytest.approx(315.0)
    assert 1.0 < float(m_low["range"][1]) < 1.2
    # Distinct robust viewports (the batches differ materially).
    assert float(m_big["x_range"][1]) > float(m_small["x_range"][1]) + 1.0
    return m_small, m_big, m_low


def test_view_lifecycle_auto_follows_new_robust_range(qapp, _batches):
    """AUTO: enabling auto-zoom zooms to the current model's robust range and
    a new model re-zooms to the *new* model's robust range (unfrozen)."""
    m_small, m_big, _m_low = _batches
    view = HistogramView()
    try:
        view.resize(420, 120)
        view.set_model(m_small)
        view.auto_zoom_enabled = True
        view.zoom_histogram()
        assert view._view_mode == "auto"
        assert view.view_range == pytest.approx(m_small["x_range"])
        assert view._frozen_range is None  # AUTO never freezes

        view.set_model(m_big)
        assert view._view_mode == "auto"
        assert view.view_range == pytest.approx(m_big["x_range"])
        assert view.view_range[1] > float(m_small["x_range"][1]) + 1.0
        assert view._frozen_range is None
    finally:
        view.deleteLater()


def test_view_lifecycle_full_persists_across_new_models(qapp, _batches):
    """FULL: reset_histogram_view shows the full domain of the current model
    and the choice PERSISTS — a new model keeps FULL semantics and the view
    follows the new model's full domain (max 22 -> view 0-22, then max 315 ->
    view 0-315), never a silent fallback to auto/default."""
    m_small, m_big, _m_low = _batches
    view = HistogramView()
    try:
        view.resize(420, 120)
        view.set_model(m_small)
        view.reset_histogram_view()
        assert view._view_mode == "full"
        assert view._frozen_range is None
        assert view.view_range[0] == 0.0
        assert view.view_range[1] == pytest.approx(22.0)

        # New model (full domain 315): FULL persists, view follows.
        view.set_model(m_big)
        assert view._view_mode == "full"
        assert view._frozen_range is None
        assert view.view_range[0] == 0.0
        assert view.view_range[1] == pytest.approx(315.0)

        # And back: a later smaller model pulls the FULL view back down.
        view.set_model(m_small)
        assert view._view_mode == "full"
        assert view.view_range[0] == 0.0
        assert view.view_range[1] == pytest.approx(22.0)
    finally:
        view.deleteLater()


def test_view_lifecycle_manual_frozen_preserved_and_reconciled(qapp, _batches):
    """MANUAL: a manual robust zoom freezes a window that survives domain
    growth verbatim and is reconciled (clamped into the domain) on shrink —
    it never silently returns to AUTO or the [0, 1] default window."""
    m_small, m_big, m_low = _batches
    view = HistogramView()
    try:
        view.resize(420, 120)
        view.set_model(m_big)
        view.zoom_histogram()  # auto is off -> MANUAL + freeze
        assert view._view_mode == "manual"
        assert view.auto_zoom_enabled is False
        frozen = view.view_range
        assert view._frozen_range == frozen
        assert frozen[1] > 1.0

        # Domain grows to 315 -> frozen window still valid, preserved verbatim.
        view.set_model(m_big)
        assert view._view_mode == "manual"
        assert view.view_range == pytest.approx(frozen)
        assert view._frozen_range == view.view_range

        # Domain shrinks far below the frozen top (upper ~1.1): the window is
        # clamped into the new domain, lo kept, never inverted/degenerate.
        view.set_model(m_low)
        upper_low = float(m_low["range"][1])
        assert view._view_mode == "manual"
        assert view._frozen_range is not None
        lo, hi = view.view_range
        assert lo == pytest.approx(frozen[0])
        assert hi == pytest.approx(upper_low)
        assert view._frozen_range == view.view_range
        assert 0.0 <= lo < hi <= upper_low + 1e-9
        assert not (hi == pytest.approx(1.0) and lo == pytest.approx(0.0))
    finally:
        view.deleteLater()


# ---------------------------------------------------------------------------
# (5) Reset genuinely paints the full-domain bars over the full domain
# ---------------------------------------------------------------------------

def test_reset_paints_full_distribution_over_full_domain(qapp):
    """FRP-H1 core: after ``reset_histogram_view`` (FULL mode) the painted
    bars are the full-domain histogram spread over the full domain — the
    sparse tail is visible as real bars near its true levels, not a widened
    empty axis around robust-only bars (and no overflow marker is drawn)."""
    buf = _mono(seed=5, size=160, bulk_hi=2.0, extremes=(40.0, 100.0, 282.0))
    model = compute_histogram_float(buf)
    assert model is not None
    upper = float(model["range"][1])
    assert upper == pytest.approx(282.0)
    bin_hi = float(model["bin_range"][1])
    assert bin_hi < upper / 10.0  # genuine sparse tail

    view = HistogramView()
    try:
        view.resize(420, 120)
        view.set_model(model)
        view.reset_histogram_view()
        assert view._view_mode == "full"
        assert view.view_range[0] == 0.0
        assert view.view_range[1] == pytest.approx(upper)

        # Bar-selection helper: FULL paints full_log_counts over the full
        # domain and never the overflow marker; the other modes keep the
        # robust bars over bin_range with the overflow marker.
        heights, lo, hi, draw_overflow = view._bars_for_current_mode()
        assert draw_overflow is False
        assert (lo, hi) == pytest.approx((0.0, upper))
        assert heights is model["full_log_counts"]  # the full-domain heights
        assert heights["L"] is not None and heights["L"].shape == (HISTOGRAM_BINS,)

        img = _render_view_to_image(view)
        rect = view._plot_rect()
        bottom_y = int(rect.bottom()) - 2  # near the bar baseline

        def _lit_near(level: float) -> bool:
            """Any lit bar pixel in a ±3 column window around ``level``."""
            idx = _bin_index(level, upper)
            center = (idx + 0.5) * upper / HISTOGRAM_BINS
            x = int(round(view._level_to_x(center)))
            return any(
                img.pixelColor(c, bottom_y).red() > 150
                for c in range(x - 3, x + 4)
            )

        # Real tail bars exist at each injected extreme's level — far right of
        # the robust bulk (which ends near x of bin_hi, ~1% of the width).
        assert _lit_near(40.0), "tail bar at level 40 missing in FULL view"
        assert _lit_near(100.0), "tail bar at level 100 missing in FULL view"
        assert _lit_near(282.0), "tail bar at the true max missing in FULL view"
        # The dense bulk bars are still painted at their own (far-left) levels.
        assert _lit_near(1.0), "bulk bars missing in FULL view"
        # Empty regions between the tail bars carry no population -> dark
        # (robust-only bars would end at bin_hi, leaving everything else empty).
        assert not _lit_near(200.0), "empty tail gap painted as a bar"
        assert not _lit_near(150.0), "empty tail gap painted as a bar"
        # Robust-only bars would end at bin_hi (~x=7): a mid axis level far
        # beyond the bulk must be empty — full-domain bars span the domain.
        assert not _lit_near(250.0), "empty tail region painted as a bar"
    finally:
        view.deleteLater()


# ---------------------------------------------------------------------------
# (6) Architecture: full histogram reuses the same sample (no 2nd traversal)
# ---------------------------------------------------------------------------

def test_full_histogram_reuses_same_sample_no_second_traversal(monkeypatch):
    """``compute_histogram_float`` feeds both the robust and the full-domain
    ``np.histogram`` the *exact same* in-domain array instance per channel
    (full_counts derived from the same sample; no second image traversal /
    re-derivation): exactly two histogram calls per channel, on identical
    sample arrays."""
    import numpy as _np

    buf = _rgb(
        seed=9, size=80, bulk_hi=2.0,
        red_extremes=(22.0, 60.0, 150.0, 300.0),
        green_extremes=(22.0, 40.0),
    )
    calls = []
    real_histogram = _np.histogram

    def _spy(a, bins=10, range=None, density=None, weights=None):
        calls.append((id(a), int(a.size)))
        return real_histogram(a, bins=bins, range=range, density=density,
                              weights=weights)

    monkeypatch.setattr(_np, "histogram", _spy)
    model = compute_histogram_float(buf)
    monkeypatch.undo()
    assert model is not None
    assert model["channels"] == ["R", "G", "B"]
    # Two histogram calls per channel (robust + full) and nothing else.
    assert len(calls) == 2 * len(model["channels"])
    for i in range(0, len(calls), 2):
        # Both histograms of a channel share the exact same sample array
        # object: the full histogram reuses the robust in-domain sample.
        assert calls[i][0] == calls[i + 1][0]
        assert calls[i][1] == calls[i + 1][1]
    # Sanity: identical-sample binning -> full conservation (see (1)/(3)).
    for ch in model["channels"]:
        assert int(model["full_counts"][ch].sum()) == buf.shape[0] * buf.shape[1]
