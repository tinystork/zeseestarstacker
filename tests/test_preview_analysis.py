"""ZSSS-OTPUX-PREVIEW-CORE-01 — pure float preview-analysis core tests.

Covers the ratified float-domain contracts in
``docs/output_truthfulness_preview_audit.md`` §5.2/§5.3/§5.5/§5.6, exercised
on the toolkit-free :mod:`seestar.gui_qt.preview_analysis` module (no Qt
widgets, no ``QImage``, no scientific-engine imports):

* 512-bin float histogram over ``[0, 1]`` with deterministic per-channel
  counts / ``log1p`` counts / min/max/median/mean/std on the *same* sample,
  RGB overlay + mono semantics;
* robust plotted X range (outlier-resistant) + explicit full ``[0, 1]`` range;
* Auto Stretch exact background-population algorithm (outlier insensitivity,
  repeatability, valid ``bp < wp``, no min/max renormalization);
* Auto WB true-background-band algorithm (correction direction, idempotence,
  zero-border / saturated-star / bright-emission exclusion, neutral fallbacks);
* stable/adaptive-anchor successive-preview witnesses (small changes preserve
  a fixed pixel while large drift avoids mass clipping);
* deterministic sampling cap;
* import / science-isolation hygiene (no forbidden imports, no eager numpy).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from seestar.gui_qt.preview_analysis import (
    ANCHOR_DRIFT_HYSTERESIS,
    ANCHOR_SEP,
    AUTO_STRETCH_DEFAULTS,
    HISTOGRAM_BINS,
    MAX_SAMPLE_PIXELS,
    NEUTRAL_WB,
    adapt_anchors_for_drift,
    apply_wb_float,
    compute_anchors,
    compute_auto_stretch_float,
    compute_auto_wb_float,
    compute_histogram_float,
    compute_histogram_stats_float,
    compute_robust_x_range,
    extract_raw_linear,
    map_raw_linear,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# §5.2 — payload extraction + stable/adaptive anchor mapping
# ---------------------------------------------------------------------------

def test_extract_raw_linear_tuple_and_legacy_fallback():
    legacy = np.full((3, 4, 3), 0.5, dtype=np.float32)
    raw = np.full((3, 4, 3), 2.0, dtype=np.float32)

    out = extract_raw_linear((legacy, raw))
    assert out is not None
    assert out.shape == (3, 4, 3)
    assert float(out[0, 0, 0]) == 2.0

    # Immutable copy: mutating the returned array never touches the source.
    out[0, 0, 0] = 99.0
    assert float(raw[0, 0, 0]) == 2.0

    # Legacy single-array payload.
    out2 = extract_raw_linear(legacy)
    assert float(out2[0, 0, 0]) == 0.5

    # Tuple whose second element is not a 2D/3D array -> fall back to first.
    out3 = extract_raw_linear((legacy, "not-an-array"))
    assert float(out3[0, 0, 0]) == 0.5
    out4 = extract_raw_linear((legacy, np.array([1.0, 2.0])))  # 1D -> skip
    assert float(out4[0, 0, 0]) == 0.5

    # Missing / non-array data.
    assert extract_raw_linear(None) is None
    assert extract_raw_linear("garbage") is None
    assert extract_raw_linear(12345) is None


def test_extract_raw_linear_empty_and_malformed_sequences():
    # Empty sequences must return None without raising (regression: IndexError).
    assert extract_raw_linear(()) is None
    assert extract_raw_linear([]) is None

    # One-element malformed sequences (second element absent, first invalid).
    assert extract_raw_linear((None,)) is None
    assert extract_raw_linear(("garbage",)) is None
    assert extract_raw_linear((12345,)) is None
    assert extract_raw_linear((np.array([1.0, 2.0, 3.0]),)) is None  # 1D

    # One-element valid array still falls back to the lone array (legacy).
    legacy = np.full((2, 2, 3), 0.25, dtype=np.float32)
    out = extract_raw_linear((legacy,))
    assert out is not None
    assert float(out[0, 0, 0]) == 0.25

    # Two-element malformed: second invalid, first invalid -> None.
    assert extract_raw_linear((None, None)) is None
    assert extract_raw_linear(("garbage", "also-garbage")) is None


def test_anchors_percentile_and_degenerate_fallbacks():
    rng = np.random.default_rng(0)
    raw = rng.uniform(1.0, 100.0, size=(64, 64)).astype(np.float32)
    lo, hi = compute_anchors(raw)
    finite_pos = raw[(np.isfinite(raw)) & (raw > 0.0)]
    assert lo == pytest.approx(np.percentile(finite_pos, 0.5))
    assert hi == pytest.approx(np.percentile(finite_pos, 99.5))
    assert hi > lo + ANCHOR_SEP

    # Constant image -> finite min/max degenerate -> symmetric widening, valid.
    const = np.full((16, 16), 7.0, dtype=np.float32)
    clo, chi = compute_anchors(const)
    assert chi > clo + ANCHOR_SEP
    assert map_raw_linear(const, clo, chi)[0, 0] == pytest.approx(0.5)

    # All non-finite -> neutral anchors.
    nan_arr = np.full((8, 8), np.nan, dtype=np.float32)
    assert compute_anchors(nan_arr) == (0.0, 1.0)
    # Empty -> neutral anchors.
    assert compute_anchors(np.zeros((0, 0), dtype=np.float32)) == (0.0, 1.0)


def test_map_raw_linear_frozen_mapping():
    raw = np.array([[0.0, 5.0, 10.0]], dtype=np.float32)
    mapped = map_raw_linear(raw, 0.0, 10.0)
    assert np.allclose(mapped, [[0.0, 0.5, 1.0]])
    assert raw[0, 0] == 0.0 and raw[0, 2] == 10.0  # input unchanged


def test_unchanged_reference_pixel_stable_when_no_adaptation_warranted():
    """§5.2 bounded stability: no drift -> frozen anchors -> fixed pixel maps
    identically (the anti-pumping invariant, now conditioned on "no adaptation
    is warranted")."""
    rng = np.random.default_rng(11)
    frame1 = rng.uniform(1.0, 10.0, size=(32, 32)).astype(np.float32)
    ref = (5, 7)
    frame1[ref] = 5.0

    lo, hi = compute_anchors(frame1)
    mapped1 = map_raw_linear(frame1, lo, hi)
    m1 = float(mapped1[ref])

    # Frame 2: a *small* evolution (within the hysteresis band) with the
    # reference pixel unchanged -> the effective anchors stay frozen and the
    # reference pixel maps identically.
    frame2 = frame1 * 1.05
    frame2[ref] = 5.0
    nlo, nhi = adapt_anchors_for_drift(lo, hi, frame2)
    assert (nlo, nhi) == (lo, hi)
    m2 = float(map_raw_linear(frame2, nlo, nhi)[ref])
    assert m2 == m1


def test_adapt_anchors_modest_evolution_stays_frozen():
    """A modest +10% photometric evolution stays inside the hysteresis band and
    leaves the frozen anchors bit-identical (no per-frame percentile pumping)."""
    rng = np.random.default_rng(12)
    frame1 = rng.uniform(100.0, 200.0, size=(64, 64, 3)).astype(np.float32)
    lo, hi = compute_anchors(frame1)
    span = hi - lo
    assert span > 0.0

    # +10% multiplicative: robust high tail moves ~0.20 * span, below the 0.25
    # hysteresis band -> frozen.
    frame2 = frame1 * 1.10
    assert (adapt_anchors_for_drift(lo, hi, frame2)) == (lo, hi)


def test_adapt_anchors_2x_3x_drift_no_whiteout():
    """A legitimate 2x / 3x global evolution widens the high anchor so the bulk
    of the image no longer maps to exactly 1.0 (no artificial saturation)."""
    rng = np.random.default_rng(13)
    frame1 = rng.uniform(100.0, 200.0, size=(64, 64, 3)).astype(np.float32)
    lo, hi = compute_anchors(frame1)

    for scale in (2.0, 3.0):
        frame = frame1 * scale
        nlo, nhi = adapt_anchors_for_drift(lo, hi, frame)
        # Bright drift widens only the high side (low anchor stays frozen).
        assert nlo == lo
        assert nhi > hi
        assert nhi > lo

        mapped = map_raw_linear(frame, nlo, nhi)
        in_dom = mapped[np.isfinite(mapped)]
        frac1 = float((in_dom == 1.0).mean())
        # Regression: before the fix the *entire* frame clipped to 1.0.
        assert frac1 < 0.5, f"scale={scale}: majority still clipped (frac1={frac1})"
        assert 0.0 < float(np.median(in_dom)) < 1.0
        # Only the natural ~0.5% robust high tail is allowed to saturate.
        assert frac1 < 0.05, f"scale={scale}: unexpected saturation frac1={frac1}"


def test_adapt_anchors_dark_drift_widens_low_anchor():
    """A dark drift widens only the low anchor (ratchet is symmetric outward)."""
    rng = np.random.default_rng(14)
    frame1 = rng.uniform(100.0, 200.0, size=(64, 64, 3)).astype(np.float32)
    lo, hi = compute_anchors(frame1)

    frame_dark = frame1 * 0.25  # strong dark drift
    nlo, nhi = adapt_anchors_for_drift(lo, hi, frame_dark)
    assert nlo < lo
    assert nhi == hi
    mapped = map_raw_linear(frame_dark, nlo, nhi)
    frac0 = float((mapped == 0.0).mean())
    assert frac0 < 0.05  # not majority-black either
    assert 0.0 < float(np.median(mapped)) < 1.0


def test_adapt_anchors_ratchet_never_shrinks():
    """The ratchet only widens: a transient dimmer frame after a bright drift
    never shrinks the mapping back (temporal anti-pumping)."""
    rng = np.random.default_rng(15)
    frame1 = rng.uniform(100.0, 200.0, size=(64, 64, 3)).astype(np.float32)
    lo, hi = compute_anchors(frame1)

    bright = frame1 * 3.0
    lo2, hi2 = adapt_anchors_for_drift(lo, hi, bright)
    assert hi2 > hi

    # A subsequent frame identical to the original (dimmer than the widened
    # anchors) must not shrink the range back.
    lo3, hi3 = adapt_anchors_for_drift(lo2, hi2, frame1)
    assert lo3 == lo2
    assert hi3 == hi2


def test_adapt_anchors_degenerate_and_invalid_inputs_safe():
    """Degenerate / non-finite / empty inputs leave anchors unchanged (or fall
    back to fresh anchors when the frozen pair is itself non-finite)."""
    rng = np.random.default_rng(16)
    frame1 = rng.uniform(1.0, 10.0, size=(16, 16)).astype(np.float32)
    lo, hi = compute_anchors(frame1)

    # All-NaN new frame -> no drift info -> unchanged.
    assert adapt_anchors_for_drift(lo, hi, np.full((8, 8), np.nan)) == (lo, hi)
    # Empty new frame -> unchanged.
    assert adapt_anchors_for_drift(lo, hi, np.zeros((0, 0))) == (lo, hi)
    # Non-finite frozen anchors -> fresh anchor computation.
    nlo, nhi = adapt_anchors_for_drift(np.nan, np.inf, frame1)
    assert np.isfinite(nlo) and np.isfinite(nhi) and nhi > nlo

    # Constant new frame (finite-positive sample exists but degenerate) is
    # handled without raising and stays non-degenerate.
    clo, chi = adapt_anchors_for_drift(lo, hi, np.full((16, 16), 1000.0))
    assert chi > clo


# ---------------------------------------------------------------------------
# §5.3 — histogram / stats / X range
# ---------------------------------------------------------------------------

def test_histogram_512_bins_domain_and_counts():
    rng = np.random.default_rng(42)
    H, W = 40, 50
    arr = rng.random((H, W, 3)).astype(np.float32)

    res = compute_histogram_float(arr)
    assert res["bins"] == HISTOGRAM_BINS == 512
    assert res["range"] == (0.0, 1.0)
    assert res["channels"] == ["R", "G", "B"]
    assert res["full_range"] == (0.0, 1.0)

    for ch, idx in (("R", 0), ("G", 1), ("B", 2)):
        counts = res["counts"][ch]
        assert counts.shape == (512,)
        assert counts.dtype == np.int64
        assert counts.sum() == H * W  # no capping at this size
        # log1p visualization counts, empty bin == 0 preserved.
        assert np.allclose(res["log_counts"][ch], np.log1p(counts.astype(np.float64)))
        # Stats from the exact same sample as the histogram.
        plane = arr[..., idx]
        finite = plane[np.isfinite(plane)]
        s = res["stats"][ch]
        assert s["min"] == pytest.approx(float(finite.min()))
        assert s["max"] == pytest.approx(float(finite.max()))
        assert s["median"] == pytest.approx(float(np.median(finite)))
        assert s["mean"] == pytest.approx(float(finite.mean()))
        assert s["std"] == pytest.approx(float(finite.std()))


def test_histogram_known_bins_exact_placement():
    arr = np.zeros((4, 4, 3), dtype=np.float32)
    arr[..., 0] = 0.0  # -> bin 0
    arr[..., 1] = 0.5  # -> bin 256
    arr[..., 2] = 1.0  # -> bin 511 (last bin is right-inclusive)
    res = compute_histogram_float(arr)
    r, g, b = res["counts"]["R"], res["counts"]["G"], res["counts"]["B"]
    assert r[0] == 16 and r.sum() == 16
    assert g[256] == 16 and g.sum() == 16
    assert b[511] == 16 and b.sum() == 16


def test_histogram_mono_safe():
    res = compute_histogram_float(np.full((6, 6), 0.25, dtype=np.float32))
    assert res["channels"] == ["L"]
    assert res["counts"]["L"].shape == (512,)
    assert res["counts"]["L"].sum() == 36

    # (H, W, 1) is also mono.
    res2 = compute_histogram_float(np.full((6, 6, 1), 0.25, dtype=np.float32))
    assert res2["channels"] == ["L"]

    # Non-image / empty input is safe.
    assert compute_histogram_float(None) is None
    assert compute_histogram_float(np.zeros((0, 0, 3), dtype=np.float32)) is None


def test_histogram_all_nan_fail_closed_no_fabrication():
    # All-NaN mono / RGB must not fabricate a synthetic zero-valued pixel.
    assert compute_histogram_float(np.full((8, 8), np.nan, dtype=np.float32)) is None
    assert compute_histogram_float(np.full((8, 8, 3), np.nan, dtype=np.float32)) is None
    assert compute_histogram_stats_float(np.full((8, 8), np.nan, dtype=np.float32)) is None
    assert compute_histogram_stats_float(np.full((8, 8, 3), np.nan, dtype=np.float32)) is None


def test_histogram_partially_invalid_channel_fail_closed():
    rng = np.random.default_rng(44)
    arr = rng.random((8, 8, 3)).astype(np.float32)
    arr[..., 2] = np.nan  # B channel entirely unusable
    assert compute_histogram_float(arr) is None


def test_histogram_in_channel_nan_dropped_not_fabricated():
    rng = np.random.default_rng(45)
    arr = rng.random((8, 8)).astype(np.float32)
    arr[0, 0] = np.nan
    res = compute_histogram_float(arr)
    assert res is not None
    assert res["counts"]["L"].sum() == 63  # the single NaN is dropped, not fabricated
    finite = arr[np.isfinite(arr)]
    assert res["stats"]["L"]["mean"] == pytest.approx(float(finite.mean()))


def test_histogram_bins_override_cannot_bypass_contract():
    rng = np.random.default_rng(46)
    arr = rng.random((8, 8)).astype(np.float32)
    # Exactly 512 bins, always; the public API exposes no bin override.
    assert compute_histogram_float(arr)["bins"] == HISTOGRAM_BINS == 512
    with pytest.raises(TypeError):
        compute_histogram_float(arr, 64)
    with pytest.raises(TypeError):
        compute_histogram_float(arr, bins=64)


def test_histogram_out_of_domain_restricted_same_sample():
    # Values outside [0, 1] are excluded from BOTH counts and stats so the
    # histogram never silently drops values the stats still describe.
    arr = np.array([[0.0, 0.5, 1.0], [2.0, -1.0, np.nan]], dtype=np.float32)
    res = compute_histogram_float(arr)
    assert res is not None
    counts = res["counts"]["L"]
    assert counts.sum() == 3  # only 0.0, 0.5, 1.0 are in-domain
    s = res["stats"]["L"]
    assert s["min"] == 0.0
    assert s["max"] == 1.0
    assert s["median"] == 0.5
    assert s["mean"] == pytest.approx(0.5)


def test_histogram_stats_wrapper_same_sample():
    rng = np.random.default_rng(43)
    arr = rng.random((20, 20, 3)).astype(np.float32)
    stats = compute_histogram_stats_float(arr)
    full = compute_histogram_float(arr)
    for ch in ("R", "G", "B"):
        assert stats[ch] == full["stats"][ch]


def test_robust_x_range_outlier_resistant():
    rng = np.random.default_rng(7)
    bulk = rng.uniform(0.1, 0.3, size=(100, 100)).astype(np.float32)
    bulk[0, 0] = 0.0
    bulk[0, 1] = 0.999999
    lo, hi = compute_robust_x_range(bulk)
    assert 0.09 <= lo <= 0.12
    assert 0.28 <= hi <= 0.32
    assert lo < hi


def test_apply_wb_float_non_mutating():
    rng = np.random.default_rng(8)
    arr = rng.random((8, 8, 3)).astype(np.float32)
    before = arr.copy()
    out = apply_wb_float(arr, (1.5, 1.0, 0.5))
    assert np.array_equal(arr, before)  # input untouched
    assert np.allclose(out[..., 0], np.clip(before[..., 0] * 1.5, 0, 1))
    assert np.allclose(out[..., 2], np.clip(before[..., 2] * 0.5, 0, 1))
    # mono is unaffected
    mono = np.full((4, 4), 0.5, dtype=np.float32)
    assert np.array_equal(apply_wb_float(mono, (2.0, 2.0, 2.0)), mono)


# ---------------------------------------------------------------------------
# §5.5 — Auto Stretch
# ---------------------------------------------------------------------------

def test_auto_stretch_repeatable_and_outlier_resistant():
    rng = np.random.default_rng(3)
    bg = rng.normal(0.2, 0.01, size=(200, 200)).astype(np.float32)
    arr = bg.copy()
    for _ in range(20):
        y, x = rng.integers(0, 200, size=2)
        arr[y, x] = 0.95  # bright stars / hot pixels

    a = compute_auto_stretch_float(arr)
    b = compute_auto_stretch_float(arr)
    assert a == b  # deterministic / repeatable
    bp, wp = a
    assert bp < wp
    assert 0.0 <= bp <= 1.0 - ANCHOR_SEP
    assert bp + ANCHOR_SEP <= wp <= 1.0
    # Outliers do not push wp up to the stars.
    assert wp < 0.95


def test_auto_stretch_excludes_exact_01():
    arr = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    assert compute_auto_stretch_float(arr) == AUTO_STRETCH_DEFAULTS


def test_auto_stretch_min_sample_defaults():
    arr = np.full((4, 4), 0.5, dtype=np.float32)  # 16 < 20
    assert compute_auto_stretch_float(arr) == AUTO_STRETCH_DEFAULTS


def test_auto_stretch_constant_image_valid_pair():
    arr = np.full((50, 50), 0.4, dtype=np.float32)
    bp, wp = compute_auto_stretch_float(arr)
    assert bp < wp
    assert wp - bp >= ANCHOR_SEP - 1e-9


def test_auto_stretch_known_linear_gradient():
    # A pure linear ramp: background population is the lower 60%; the result
    # must be a valid, separated pair inside [0, 1] (no min/max renormalization).
    ramp = np.linspace(0.0, 1.0, 1000, dtype=np.float32).reshape(50, 20)
    ramp = np.clip(ramp, 0.0, 1.0)
    bp, wp = compute_auto_stretch_float(ramp)
    assert 0.0 <= bp < wp <= 1.0
    assert wp - bp >= ANCHOR_SEP
    # The ramp's exact 0.0 / 1.0 endpoints are excluded from the sample.
    assert bp >= 0.0 and wp <= 1.0


# ---------------------------------------------------------------------------
# §5.6 — Auto WB
# ---------------------------------------------------------------------------

def _pedestal(size, rng, r=1.0, g=1.0, b=1.0, bg=0.2, noise=0.004):
    base = rng.normal(bg, noise, size=size)
    R = base * r + rng.normal(0, 1e-3, size=size)
    G = base * g + rng.normal(0, 1e-3, size=size)
    B = base * b + rng.normal(0, 1e-3, size=size)
    return np.clip(np.stack([R, G, B], axis=-1), 0.0, 0.97).astype(np.float32)


def test_auto_wb_red_cast_correction_direction():
    rng = np.random.default_rng(1)
    arr = _pedestal((64, 64), rng, r=1.4, g=1.0, b=1.0)
    gr, gg, gb = compute_auto_wb_float(arr)
    assert gg == 1.0
    assert 0.2 < gr < 1.0  # red too strong -> reduce red
    assert abs(gb - 1.0) < 0.05  # blue stays neutral
    assert abs(gr - 1.0 / 1.4) < 0.05  # magnitude ~ centre_g / centre_r


def test_auto_wb_blue_cast_correction_direction():
    rng = np.random.default_rng(6)
    arr = _pedestal((64, 64), rng, r=1.0, g=1.0, b=1.4)
    gr, gg, gb = compute_auto_wb_float(arr)
    assert gg == 1.0
    assert abs(gr - 1.0) < 0.05
    assert 0.2 < gb < 1.0  # blue too strong -> reduce blue


def test_auto_wb_idempotent():
    rng = np.random.default_rng(2)
    arr = _pedestal((48, 48), rng, r=1.2, b=0.9)
    assert compute_auto_wb_float(arr) == compute_auto_wb_float(arr)


def test_auto_wb_zero_borders_excluded():
    rng = np.random.default_rng(3)
    inner = _pedestal((40, 40), rng, r=1.4)
    arr = np.zeros((48, 48, 3), dtype=np.float32)
    arr[4:44, 4:44] = inner
    g1 = compute_auto_wb_float(arr)
    g2 = compute_auto_wb_float(inner)
    assert abs(g1[0] - g2[0]) < 0.05
    assert g1[0] < 1.0  # still detects the red cast despite the zero border


def test_auto_wb_saturated_stars_excluded():
    rng = np.random.default_rng(4)
    arr = _pedestal((64, 64), rng, r=1.4)
    for _ in range(30):
        y, x = rng.integers(0, 64, size=2)
        arr[y, x] = 0.999  # saturated (>= 0.98)
    gr, gg, gb = compute_auto_wb_float(arr)
    assert gg == 1.0
    assert abs(gr - 1.0 / 1.4) < 0.05  # stars excluded


def test_auto_wb_bright_colored_emission_excluded():
    rng = np.random.default_rng(5)
    arr = _pedestal((80, 80), rng, r=1.0, g=1.0, b=1.0)  # neutral grey bg
    arr[60:80, 60:80, 0] = 0.9  # bright red "nebula" (upper luminance band)
    arr[60:80, 60:80, 1] = 0.2
    arr[60:80, 60:80, 2] = 0.1
    gr, gg, gb = compute_auto_wb_float(arr)
    # Background is neutral grey -> gains ~ neutral (nebula excluded from band).
    assert abs(gr - 1.0) < 0.05
    assert abs(gb - 1.0) < 0.05


def test_auto_wb_neutral_fallbacks():
    # < 3 channels.
    assert compute_auto_wb_float(np.zeros((8, 8), dtype=np.float32)) == NEUTRAL_WB
    # All zeros (no strictly-positive pixel).
    assert compute_auto_wb_float(np.zeros((8, 8, 3), dtype=np.float32)) == NEUTRAL_WB
    # Fewer than 64 valid pixels.
    assert compute_auto_wb_float(np.full((4, 4, 3), 0.5, dtype=np.float32)) == NEUTRAL_WB
    # Flat image -> degenerate luminance band.
    assert compute_auto_wb_float(np.full((32, 32, 3), 0.5, dtype=np.float32)) == NEUTRAL_WB


# ---------------------------------------------------------------------------
# Deterministic sampling cap + non-mutation
# ---------------------------------------------------------------------------

def test_sampling_cap_deterministic():
    rng = np.random.default_rng(12)
    arr = rng.random((1200, 1200), dtype=np.float32)  # 1.44M > 1M cap
    h1 = compute_histogram_float(arr)
    h2 = compute_histogram_float(arr)
    assert np.array_equal(h1["counts"]["L"], h2["counts"]["L"])
    total = int(h1["counts"]["L"].sum())
    assert 0 < total <= MAX_SAMPLE_PIXELS


def test_analysis_never_mutates_inputs():
    rng = np.random.default_rng(13)
    arr = rng.random((16, 16, 3)).astype(np.float32)
    before = arr.copy()
    compute_histogram_float(arr)
    compute_histogram_stats_float(arr)
    compute_auto_stretch_float(arr)
    compute_auto_wb_float(arr)
    apply_wb_float(arr, (1.2, 1.0, 0.8))
    compute_robust_x_range(arr)
    compute_anchors(arr)
    map_raw_linear(arr, 0.1, 0.9)
    adapt_anchors_for_drift(0.1, 0.9, arr)
    extract_raw_linear((arr, arr))
    assert np.array_equal(arr, before)


# ---------------------------------------------------------------------------
# Import / science-isolation hygiene
# ---------------------------------------------------------------------------

def test_preview_analysis_source_is_science_and_widget_free():
    pkg_dir = ROOT / "seestar" / "gui_qt"
    text = (pkg_dir / "preview_analysis.py").read_text(encoding="utf-8")
    forbidden = (
        "seestar.core",
        "seestar.alignment",
        "seestar.enhancement",
        "seestar.queuep",
        "tkinter",
        "seestar.gui.settings",
        "seestar.gui.main_window",
        "seestar.gui.boring_stack",
        "zesolver_adapter",
        "zesolver.api",
        "zealfie",
        "PIL",
        "matplotlib",
        "PySide6",
        "QtGui",
        "QtWidgets",
        "QtCore",
    )
    for token in forbidden:
        assert token not in text, f"preview_analysis.py references {token}"
    # numpy stays a lazy import: no top-level ``import numpy`` / ``from numpy``.
    for line in text.splitlines():
        stripped = line.strip()
        assert not stripped.startswith(("import numpy", "from numpy")), (
            f"preview_analysis.py imports numpy at top level: {line!r}"
        )


def test_preview_analysis_import_no_eager_numpy():
    code = (
        "import sys\n"
        "import seestar.gui_qt.preview_analysis  # noqa: F401\n"
        "bad = [m for m in sys.modules\n"
        "       if m == 'numpy'\n"
        "       or m.startswith('seestar.core')\n"
        "       or m.startswith('seestar.alignment')\n"
        "       or m.startswith('seestar.enhancement')\n"
        "       or m.startswith('seestar.queuep')]\n"
        "if bad:\n"
        "    print('BAD_MODULES:', bad)\n"
        "    sys.exit(1)\n"
        "print('IMPORT_HYGIENE_OK')\n"
    )
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    assert proc.returncode == 0, (
        f"preview_analysis import hygiene violated: stdout={proc.stdout!r} "
        f"stderr={proc.stderr!r}"
    )
