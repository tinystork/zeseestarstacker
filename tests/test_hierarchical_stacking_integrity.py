"""Focused provenance-aware stacking integrity tests (ZSSS-HSI-1).

These tests pin the corrected classic batch reduction semantics:

* uniform and quality-weighted arithmetic means are invariant to legitimate
  batch decomposition;
* invalid spatial samples are treated as *missing* (never numeric zero) in
  median / rejection algorithms;
* every reduction exposes the effective per-pixel / per-channel denominator W
  such that ``V * W`` is the numerator of that reduction;
* the RAM path and the tiled/HQ path share the same semantics and compose
  hidden subgroups via effective denominators, not geometric coverage.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
# Lightweight test environments may lack the heavy optional runtime
# dependencies that ``seestar.queuep.queue_manager`` imports at module top
# level (alignment, drizzle, ccdproc, OpenCV).  None of them are exercised by
# these stacking tests, so stub only the modules that are genuinely absent.
import importlib.util as _ilu  # noqa: E402

_missing_optional = {
    _name for _name in ("cv2", "astroalign", "ccdproc", "drizzle")
    if _ilu.find_spec(_name) is None
}

for _name in ("cv2", "astroalign"):
    if _name in _missing_optional:
        sys.modules.setdefault(_name, types.ModuleType(_name))

if "ccdproc" in _missing_optional:
    _ccdproc = types.ModuleType("ccdproc")
    _ccdproc.combine = None
    sys.modules.setdefault("ccdproc", _ccdproc)

if "drizzle" in _missing_optional:
    _drizzle = types.ModuleType("drizzle")
    _drizzle_resample = types.ModuleType("drizzle.resample")

    class _DummyDrizzle:
        pass

    _drizzle_resample.Drizzle = _DummyDrizzle
    _drizzle.resample = _drizzle_resample
    sys.modules.setdefault("drizzle", _drizzle)
    sys.modules.setdefault("drizzle.resample", _drizzle_resample)

if "seestar.gui" not in sys.modules:
    seestar_pkg = types.ModuleType("seestar")
    seestar_pkg.__path__ = [str(ROOT / "seestar")]
    gui_pkg = types.ModuleType("seestar.gui")
    gui_pkg.__path__ = []
    settings_mod = types.ModuleType("seestar.gui.settings")

    class DummySettingsManager:
        pass

    settings_mod.SettingsManager = DummySettingsManager
    gui_pkg.settings = settings_mod
    seestar_pkg.gui = gui_pkg
    sys.modules["seestar"] = seestar_pkg
    sys.modules["seestar.gui"] = gui_pkg
    sys.modules["seestar.gui.settings"] = settings_mod

from seestar.queuep.queue_manager import SeestarQueuedStacker  # noqa: E402
import seestar.queuep.queue_manager as qm  # noqa: E402
from seestar.core.stack_methods import (  # noqa: E402
    _stack_kappa_sigma,
    _stack_mean,
    _stack_median,
    _winsorize_axis0_numpy,
    _stack_winsorized_sigma,
)

HEADER = fits.Header()


def make_stack(
    mode="mean",
    use_quality_weighting=False,
    weight_by_snr=True,
    weight_by_stars=False,
    snr_exponent=1.0,
    stars_exponent=0.5,
    min_weight=0.0,
    kappa_low=3.0,
    kappa_high=3.0,
    max_hq_mem=1_000_000_000,
    batch_size=10,
):
    """Build a lightweight SeestarQueuedStacker without running __init__."""
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o.stacking_mode = mode
    o.normalize_method = "none"
    o.weighting_method = "none"
    o.use_quality_weighting = use_quality_weighting
    o.weight_by_snr = weight_by_snr
    o.weight_by_stars = weight_by_stars
    o.snr_exponent = snr_exponent
    o.stars_exponent = stars_exponent
    o.min_weight = min_weight
    o.apply_batch_feathering = False
    o.reproject_between_batches = False
    o.reproject_coadd_final = False
    o.drizzle_active_session = False
    o.is_mosaic_run = False
    o.stack_kappa_low = kappa_low
    o.stack_kappa_high = kappa_high
    o.winsor_limits = (0.05, 0.05)
    o.stack_reject_algo = "none"
    o.max_hq_mem = max_hq_mem
    o.batch_size = batch_size
    o.settings = None
    o.reference_header_for_wcs = None
    o.reference_wcs_object = None
    o.interbatch_norm_active = False
    o.max_stack_workers = 1
    o._current_batch_paths = []
    # P5-FIX: pin q_ref explicitly (absolute domain, q_ref == 1.0) so the
    # decomposition-invariance harness never relies on the removed raw-domain
    # fallback.
    o._quality_reference_scale = 1.0
    return o


def item(value, mask=1.0, snr=1.0, stars=0.0, shape=(1, 1)):
    """Build one batch item (image, header, scores, wcs, validity mask)."""
    if isinstance(mask, (int, float)):
        m = np.full(shape, mask, dtype=bool)
    else:
        m = np.asarray(mask, dtype=bool)
        shape = m.shape
    img = np.full(shape, value, dtype=np.float32)
    return (img, HEADER, {"snr": snr, "stars": stars}, None, m)


def reduce_decomposition(stack, batches):
    """Reduce a list of sub-batches and combine via sum(V*W)/sum(W)."""
    num = None
    den = None
    for batch in batches:
        V, _hdr, W = stack._stack_batch(batch, 1, 1)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        if W.ndim == 2 and V.ndim == 3:
            W = W[..., None]
        num = V * W if num is None else num + V * W
        den = W if den is None else den + W
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def combine_vw(batches):
    """Combine (V, W) pairs via sum(V*W)/sum(W)."""
    num = None
    den = None
    for V, W in batches:
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        if W.ndim == 2 and V.ndim == 3:
            W = W[..., None]
        num = V * W if num is None else num + V * W
        den = W if den is None else den + W
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


# ---------------------------------------------------------------------------
# 1. Baseline counterexamples
# ---------------------------------------------------------------------------


def test_quality_weighted_mean_is_decomposition_invariant():
    stack = make_stack("mean", use_quality_weighting=True, snr_exponent=1.0)
    items = [
        item(0, snr=1.0),
        item(10, snr=9.0),
        item(100, snr=1.0),
    ]

    whole = reduce_decomposition(stack, [items])
    two_plus_one = reduce_decomposition(stack, [items[:2], items[2:]])
    one_each = reduce_decomposition(stack, [items[:1], items[1:2], items[2:]])

    assert np.isclose(whole[0, 0], 190.0 / 11.0, rtol=1e-6)
    assert np.isclose(two_plus_one[0, 0], 190.0 / 11.0, rtol=1e-6)
    assert np.isclose(one_each[0, 0], 190.0 / 11.0, rtol=1e-6)


def test_kappa_rejected_mass_is_survivor_mass_not_geometric():
    stack = make_stack("kappa-sigma", kappa_low=1.0, kappa_high=1.0)
    items = [item(v) for v in (0, 0, 100, 10)]

    whole = reduce_decomposition(stack, [items])
    three_plus_one = reduce_decomposition(stack, [items[:3], items[3:]])

    # Surviving samples are 0, 0, 10 -> mean 10/3.
    assert np.isclose(whole[0, 0], 10.0 / 3.0, rtol=1e-5)
    assert np.isclose(three_plus_one[0, 0], 10.0 / 3.0, rtol=1e-5)


def test_invalid_sample_is_missing_not_zero():
    # median and kappa-sigma over one valid (10) and one invalid (20) sample
    # must return 10, not 5.
    invalid_item = (np.array([[20.0]], dtype=np.float32), HEADER,
                    {"snr": 1.0, "stars": 0.0}, None,
                    np.array([[False]], dtype=bool))
    valid_item = (np.array([[10.0]], dtype=np.float32), HEADER,
                  {"snr": 1.0, "stars": 0.0}, None,
                  np.array([[True]], dtype=bool))
    batch = [valid_item, invalid_item]

    for mode, klo, khi in [("median", 3.0, 3.0), ("kappa-sigma", 3.0, 3.0)]:
        stack = make_stack(mode, kappa_low=klo, kappa_high=khi)
        V, _hdr, W = stack._stack_batch(batch, 1, 1)
        assert np.isclose(V[0, 0], 10.0, rtol=1e-5), mode
        assert np.isclose(W[0, 0], 1.0, rtol=1e-5), mode


# ---------------------------------------------------------------------------
# 2. Unequal decomposition: 61 observations split 3 + 17 + 41
# ---------------------------------------------------------------------------


def test_uniform_mean_unequal_decomposition_61():
    rng = np.random.default_rng(42)
    values = rng.normal(100.0, 7.0, size=61).astype(np.float32)
    items = [item(float(v)) for v in values]

    stack = make_stack("mean")

    whole = reduce_decomposition(stack, [items])
    split = reduce_decomposition(stack, [items[:3], items[3:20], items[20:]])

    # Reordered (shuffled) decomposition must agree too.
    idx = rng.permutation(61)
    shuffled = [items[int(i)] for i in idx]
    split_shuffled = reduce_decomposition(
        stack, [shuffled[:3], shuffled[3:20], shuffled[20:]]
    )

    # Multi-level composition: [3] + [17] -> [20], then [20] + [41] -> [61].
    lvl_a = stack._stack_batch(items[:3], 1, 1)
    lvl_b = stack._stack_batch(items[3:20], 1, 1)
    lvl_c = stack._stack_batch(items[20:], 1, 1)
    two_level = combine_vw([(lvl_a[0], lvl_a[2]), (lvl_b[0], lvl_b[2])])
    three_level = combine_vw([(two_level, lvl_a[2] + lvl_b[2]), (lvl_c[0], lvl_c[2])])

    expected = np.mean(values)
    assert np.isclose(whole[0, 0], expected, rtol=1e-5)
    assert np.isclose(split[0, 0], expected, rtol=1e-5)
    assert np.isclose(split_shuffled[0, 0], expected, rtol=1e-5)
    assert np.isclose(three_level[0, 0], expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# 3. Nonuniform quality weights + spatial validity masks
# ---------------------------------------------------------------------------


def test_quality_weights_and_spatial_masks_decomposition_invariant():
    rng = np.random.default_rng(7)
    values = rng.normal(50.0, 5.0, size=12).astype(np.float32)
    snrs = rng.uniform(0.5, 3.0, size=12).astype(np.float32)

    shape = (4, 4)
    items = []
    for v, s in zip(values, snrs):
        mask = rng.random(shape) > 0.25  # ~25% invalid pixels
        items.append(item(float(v), mask=mask, snr=float(s)))

    stack = make_stack("mean", use_quality_weighting=True, snr_exponent=1.0)

    whole = reduce_decomposition(stack, [items])
    split = reduce_decomposition(stack, [items[:5], items[5:9], items[9:]])

    # Reference: manually weighted mean per pixel.
    num = np.zeros(shape, dtype=np.float64)
    den = np.zeros(shape, dtype=np.float64)
    for v, s, it in zip(values, snrs, items):
        m = it[4].astype(np.float64)
        num += m * float(v) * float(s)
        den += m * float(s)
    expected = num / den

    assert np.allclose(whole, expected, rtol=1e-4)
    assert np.allclose(split, expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# 4. Per-channel rejection denominators are retained
# ---------------------------------------------------------------------------


def test_per_channel_rejection_denominators_retained():
    stack = make_stack("kappa-sigma", kappa_low=1.0, kappa_high=1.0)
    shape = (1, 1, 3)
    # R channel has an outlier (100), G and B are all equal.
    values = [
        np.array([[[10.0, 10.0, 10.0]]], dtype=np.float32),
        np.array([[[10.0, 10.0, 10.0]]], dtype=np.float32),
        np.array([[[10.0, 10.0, 10.0]]], dtype=np.float32),
        np.array([[[100.0, 10.0, 10.0]]], dtype=np.float32),
    ]
    batch = [
        (v, HEADER, {"snr": 1.0, "stars": 0.0}, None, np.ones((1, 1), dtype=bool))
        for v in values
    ]

    V, _hdr, W = stack._stack_batch(batch, 1, 1)

    assert W.ndim == 3, "per-channel rejection must return a 3-D weight map"
    # R rejects the outlier (3 survivors), G and B keep all 4.
    assert np.allclose(W[0, 0], [3.0, 4.0, 4.0], rtol=1e-5), W[0, 0]
    assert np.allclose(V[0, 0], [10.0, 10.0, 10.0], rtol=1e-5), V[0, 0]


def test_combine_batch_result_accumulates_per_channel_denominator():
    stack = make_stack("kappa-sigma", kappa_low=1.0, kappa_high=1.0)
    stack.memmap_shape = (1, 1, 3)
    stack.memmap_dtype_sum = np.float32
    stack.memmap_dtype_wht = np.float32
    stack.cumulative_sum_memmap = np.zeros((1, 1, 3), dtype=np.float32)
    stack.cumulative_wht_memmap = np.zeros((1, 1, 3), dtype=np.float32)
    stack.stacked_batches_count = 0
    stack.images_in_cumulative_stack = 0
    stack.total_exposure_seconds = 0.0
    stack.failed_stack_count = 0
    stack.current_stack_header = None
    stack.logger = types.SimpleNamespace(warning=lambda *a, **k: None)

    # A batch whose per-channel value/denominator are (V, W) = ([10,10,10], [3,4,4]).
    V = np.full((1, 1, 3), 10.0, dtype=np.float32)
    W = np.array([[[3.0, 4.0, 4.0]]], dtype=np.float32)
    hdr = fits.Header()
    hdr["NIMAGES"] = 4

    stack._combine_batch_result(V, hdr, W)

    assert np.allclose(stack.cumulative_sum_memmap[0, 0], [30.0, 40.0, 40.0], rtol=1e-5)
    assert np.allclose(stack.cumulative_wht_memmap[0, 0], [3.0, 4.0, 4.0], rtol=1e-5)


# ---------------------------------------------------------------------------
# 5. RAM vs tiled/HQ agreement
# ---------------------------------------------------------------------------


def _make_color_batch(n=8, shape=(3, 3, 3)):
    rng = np.random.default_rng(3)
    imgs = [rng.normal(100.0, 5.0, size=shape).astype(np.float32) for _ in range(n)]
    masks = [np.ones(shape[:2], dtype=bool) for _ in range(n)]
    return imgs, masks


def test_ram_vs_tiled_mean_agreement_and_hidden_subgroups():
    imgs, masks = _make_color_batch()
    qw = np.ones(len(imgs), dtype=np.float32)

    # RAM path.
    ram = make_stack("mean")
    items = [
        (im, HEADER, {"snr": 1.0, "stars": 0.0}, None, m)
        for im, m in zip(imgs, masks)
    ]
    V_ram, _hdr, W_ram = ram._stack_batch(items, 1, 1)

    # Tiled path without hidden subgroup split.
    for max_mem in (1_000_000_000, 1):
        tiled = make_stack("mean", max_hq_mem=max_mem)
        V_tile, W_tile = tiled._combine_hq_by_tiles(
            imgs, masks, 3.0, (0.05, 0.05),
            masks_list=masks, quality_weights=qw, use_memmap=False, tile_h=2,
        )
        # Mean composes exactly even when hidden subgroups are forced.
        assert np.allclose(V_tile, V_ram, rtol=1e-4)
        assert np.allclose(W_tile, W_ram, rtol=1e-4)


def test_ram_vs_tiled_rejection_agree_without_hidden_split():
    imgs, masks = _make_color_batch()
    qw = np.ones(len(imgs), dtype=np.float32)

    ram = make_stack("kappa-sigma", kappa_low=3.0, kappa_high=3.0)
    items = [
        (im, HEADER, {"snr": 1.0, "stars": 0.0}, None, m)
        for im, m in zip(imgs, masks)
    ]
    V_ram, _hdr, W_ram = ram._stack_batch(items, 1, 1)

    # Tiled with a single subgroup (no hidden split).
    tiled = make_stack("kappa-sigma", kappa_low=3.0, kappa_high=3.0,
                       max_hq_mem=1_000_000_000)
    V_tile, W_tile = tiled._combine_hq_by_tiles(
        imgs, masks, 3.0, (0.05, 0.05),
        masks_list=masks, quality_weights=qw, use_memmap=False, tile_h=2,
    )
    assert np.allclose(V_tile, V_ram, rtol=1e-4)
    assert np.allclose(W_tile, W_ram, rtol=1e-4)


# ---------------------------------------------------------------------------
# 6. Median / rejection: invalid excluded + boundary-dependent approximation
# ---------------------------------------------------------------------------


def test_median_invalid_excluded():
    imgs = [np.array([[5.0]], dtype=np.float32), np.array([[15.0]], dtype=np.float32)]
    masks = [np.array([[True]], dtype=bool), np.array([[False]], dtype=bool)]
    nan_imgs = [np.where(m, a, np.nan) for a, m in zip(imgs, masks)]
    result, W, _ = _stack_median(nan_imgs, None, return_weights=True)
    assert np.isclose(result[0, 0], 5.0, rtol=1e-5)
    assert np.isclose(W[0, 0], 1.0, rtol=1e-5)


def test_kappa_invalid_excluded_and_survivor_weight():
    imgs = [np.array([[10.0]], dtype=np.float32), np.array([[20.0]], dtype=np.float32)]
    masks = [np.array([[True]], dtype=bool), np.array([[False]], dtype=bool)]
    nan_imgs = [np.where(m, a, np.nan) for a, m in zip(imgs, masks)]
    result, W, _ = _stack_kappa_sigma(nan_imgs, None, sigma_low=3.0, sigma_high=3.0,
                                      return_weights=True)
    assert np.isclose(result[0, 0], 10.0, rtol=1e-5)
    assert np.isclose(W[0, 0], 1.0, rtol=1e-5)


def test_median_is_documented_approximation_not_global_exact():
    # Hierarchical median is a count-weighted mean of local medians, which is
    # NOT the global median.  Characterise (do not assert global equality).
    a = [np.array([[0.0]], dtype=np.float32), np.array([[100.0]], dtype=np.float32)]
    b = [np.array([[10.0]], dtype=np.float32)]
    ma, Wa, _ = _stack_median(a, None, return_weights=True)
    mb, Wb, _ = _stack_median(b, None, return_weights=True)
    hierarchical = (ma * Wa + mb * Wb) / (Wa + Wb)

    global_median = np.median(np.array([0.0, 100.0, 10.0]))

    # Defined bounded-memory combination of local medians:
    assert np.isclose(hierarchical[0, 0], (50.0 * 2 + 10.0 * 1) / 3.0, rtol=1e-5)
    # And it differs from the global median (documented boundary dependence).
    assert not np.isclose(hierarchical[0, 0], global_median, rtol=1e-5)


def test_winsorized_replacement_semantics_define_denominator():
    # apply_rewinsor=True: the rejected (outlier) sample is substituted with a
    # winsorized value and still contributes its weight -> W == 5.
    data = [np.array([[10.0]], dtype=np.float32) for _ in range(4)]
    data.append(np.array([[100.0]], dtype=np.float32))
    result, W, _ = _stack_winsorized_sigma(
        data, None, kappa=3.0, winsor_limits=(0.2, 0.2),
        apply_rewinsor=True, return_weights=True,
    )
    assert np.isclose(result[0, 0], 10.0, rtol=1e-5)
    assert np.isclose(W[0, 0], 5.0, rtol=1e-5)

    # apply_rewinsor=False: the rejected sample is excluded -> W == 4.
    result2, W2, _ = _stack_winsorized_sigma(
        data, None, kappa=3.0, winsor_limits=(0.2, 0.2),
        apply_rewinsor=False, return_weights=True,
    )
    assert np.isclose(result2[0, 0], 10.0, rtol=1e-5)
    assert np.isclose(W2[0, 0], 4.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# 7. HSI-1 corrective regressions (ZSSS-HSI-1-C1)
# ---------------------------------------------------------------------------


class _FinalizerDummy:
    """Lightweight stand-in with the attributes ``_save_final_stack`` needs."""

    pass


def _make_finalizer(tmp_path, sum_, wht):
    obj = _FinalizerDummy()
    obj.update_progress = lambda *a, **k: None
    obj._close_memmaps = lambda: None
    obj.save_final_as_float32 = True
    obj.preserve_linear_output = True
    obj.drizzle_wht_threshold = 0
    obj.images_in_cumulative_stack = 1
    obj.total_exposure_seconds = 1.0
    obj.output_folder = str(tmp_path)
    obj.output_filename = "out.fit"
    obj.current_stack_header = fits.Header()
    obj.drizzle_active_session = False
    obj.is_mosaic_run = False
    obj.drizzle_mode = "Final"
    obj.drizzle_output_wcs = None
    obj.drizzle_fillval = "0.0"
    obj.reproject_between_batches = False
    obj.reproject_coadd_final = False
    obj.batch_size = 0
    obj.cumulative_sum_memmap = np.asarray(sum_, dtype=np.float32)
    obj.cumulative_wht_memmap = np.asarray(wht, dtype=np.float32)
    obj.logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    return obj


def test_save_final_stack_scientific_hwc_wht_per_channel_division(tmp_path):
    # Defect 1 regression: a 3-D (HWC) scientific WHT must divide each channel
    # by its own denominator.  Collapsing W to a 2-D mean would give
    # [30/4, 40/4, 50/4] = [7.5, 10, 12.5] instead of [10, 10, 10].
    sum_ = np.array(
        [[[30.0, 40.0, 50.0], [60.0, 80.0, 100.0]],
         [[90.0, 120.0, 150.0], [120.0, 160.0, 200.0]]],
        dtype=np.float32,
    )
    wht = np.array(
        [[[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]],
         [[9.0, 12.0, 15.0], [12.0, 16.0, 20.0]]],
        dtype=np.float32,
    )
    obj = _make_finalizer(tmp_path, sum_, wht)

    qm.SeestarQueuedStacker._save_final_stack(obj)

    saved = fits.getdata(obj.final_stacked_path)  # saved CHW -> (3, H, W)
    expected = (sum_ / wht).astype(np.float32)  # all exactly 10.0
    assert np.allclose(saved.astype(np.float32), np.moveaxis(expected, -1, 0),
                       rtol=1e-5), saved
    assert np.allclose(np.unique(saved.astype(np.float32)), [10.0], rtol=1e-5)


def test_save_final_stack_legacy_hw_wht_broadcasts(tmp_path):
    # Defect 1 (legacy): a 2-D WHT must remain accepted and broadcast across
    # colour channels without exception.
    sum_ = np.array([[[30.0, 40.0, 50.0]]], dtype=np.float32)
    wht = np.array([[3.0]], dtype=np.float32)
    obj = _make_finalizer(tmp_path, sum_, wht)

    qm.SeestarQueuedStacker._save_final_stack(obj)

    saved = fits.getdata(obj.final_stacked_path)  # (3, 1, 1)
    expected = (sum_ / wht[..., None]).astype(np.float32)  # [10, 13.333, 16.667]
    assert np.allclose(saved.astype(np.float32), np.moveaxis(expected, -1, 0),
                       rtol=1e-4), saved


def test_winsorize_axis0_selects_by_rank_not_position():
    # Defect 2 regression: winsorization must replace the actual low/high
    # *values* (ordered rank), not the first/last memory entries.  Here the
    # smallest valid sample (1.0) is at index 1 and the largest (9.0) is at
    # index 2, with a NaN at index 0 (missing, must stay NaN).
    arr = np.array(
        [[np.nan], [1.0], [9.0], [5.0], [3.0], [7.0], [2.0], [4.0]],
        dtype=np.float32,
    )
    out = _winsorize_axis0_numpy(arr, (0.2, 0.2))
    # n_valid = 7, lowidx = floor(1.4) = 1 -> smallest value 1.0 -> 2.0.
    # highidx = 1 -> upidx = 6 -> largest value 9.0 -> 7.0.
    expected = np.array(
        [[np.nan], [2.0], [7.0], [5.0], [3.0], [7.0], [2.0], [4.0]],
        dtype=np.float32,
    )
    assert np.allclose(out, expected, equal_nan=True), out
    assert np.isnan(out[0, 0])


def test_process_batches_retains_hwc_wht(tmp_path, monkeypatch):
    # Defect 3 regression: the incremental reproject accumulation must keep the
    # per-channel (HWC) WHT dimension and channel-wise V*W / W semantics.
    obj = qm.SeestarQueuedStacker.__new__(qm.SeestarQueuedStacker)
    obj.update_progress = lambda *a, **k: None
    obj.reference_shape = (2, 2)
    obj.stack_final_combine = "mean"
    obj._get_final_match_background = lambda default=False: False
    obj.ref_wcs_header = fits.Header()
    obj.use_gpu = False
    obj.interbatch_norm_active = False
    obj.enable_preview = False
    obj.cumulative_wht_path = str(tmp_path / "cumulative_WHT.npy")
    obj.cumulative_sum_memmap = np.lib.format.open_memmap(
        str(tmp_path / "cumulative_SUM.npy"), mode="w+",
        dtype=np.float32, shape=(2, 2, 3),
    )
    obj.cumulative_wht_memmap = np.lib.format.open_memmap(
        obj.cumulative_wht_path, mode="w+",
        dtype=np.float32, shape=(2, 2, 3),
    )

    # _reproject_worker contract: returns a premultiplied numerator (V*W) and
    # the weight map W, both HWC, with per-channel differing W.
    def fake_worker(fits_path, ref_wcs_header=None, shape_out=None,
                    use_gpu=False, match_background=False):
        num = np.full((2, 2, 3), [30.0, 40.0, 40.0], dtype=np.float32)
        wht = np.full((2, 2, 3), [3.0, 4.0, 4.0], dtype=np.float32)
        return num, wht

    monkeypatch.setattr(
        obj, "_load_and_prepare_simple",
        lambda p: (np.zeros((2, 2, 3), np.float32), None, fits.Header(), None),
    )
    monkeypatch.setattr(qm, "_reproject_worker", fake_worker)

    obj._process_batches(["a.fits"])

    assert obj.cumulative_wht_memmap.shape == (2, 2, 3)
    assert np.allclose(obj.cumulative_wht_memmap[0, 0], [3.0, 4.0, 4.0], rtol=1e-5)
    assert np.allclose(obj.cumulative_sum_memmap[0, 0], [30.0, 40.0, 40.0], rtol=1e-5)
    # V*W / W channel semantics survive per channel (here [10, 10, 10]).
    with np.errstate(divide="ignore", invalid="ignore"):
        result = obj.cumulative_sum_memmap / obj.cumulative_wht_memmap
    assert np.allclose(result[0, 0], [10.0, 10.0, 10.0], rtol=1e-5)


def test_winsorized_sigma_clip_spelling_ram_path():
    # Defect 4 regression (RAM dispatch): Qt sends
    # ``stacking_mode="winsorized-sigma-clip"``; the kernel must select
    # winsorized rejection, not the arithmetic mean (which would give 208).
    stack = make_stack("winsorized-sigma-clip", kappa_low=3.0, kappa_high=3.0)
    stack.stack_reject_algo = "none"
    stack.winsor_limits = (0.2, 0.2)
    items = [item(10.0) for _ in range(4)] + [item(1000.0)]

    V, _hdr, W = stack._stack_batch(items, 1, 1)

    assert np.isclose(V[0, 0], 10.0, rtol=1e-3), V[0, 0]
    assert np.isclose(W[0, 0], 5.0, rtol=1e-3), W[0, 0]


def test_winsorized_sigma_clip_spelling_tiled_path():
    # Defect 4 regression (tiled/HQ dispatch): same spelling must select
    # winsorized rejection inside ``_combine_hq_by_tiles``.
    stack = make_stack("winsorized-sigma-clip", kappa_low=3.0, kappa_high=3.0)
    stack.stack_reject_algo = "none"
    stack.winsor_limits = (0.2, 0.2)
    imgs = [np.full((1, 1), 10.0, np.float32) for _ in range(4)]
    imgs.append(np.full((1, 1), 1000.0, np.float32))
    masks = [np.ones((1, 1), dtype=bool) for _ in range(5)]
    qw = np.ones(5, dtype=np.float32)

    V, W = stack._combine_hq_by_tiles(
        imgs, masks, 3.0, (0.2, 0.2),
        masks_list=masks, quality_weights=qw, use_memmap=False, tile_h=1,
    )

    assert np.isclose(V[0, 0, 0], 10.0, rtol=1e-3), V
    assert np.isclose(W[0, 0], 5.0, rtol=1e-3), W


# ---------------------------------------------------------------------------
# 8. HSI-1 corrective regressions (ZSSS-HSI-1-C2)
# ---------------------------------------------------------------------------


def test_winsorize_numpy_matches_scipy_inclusive_floor():
    # Canonical semantics are scipy.stats.mstats.winsorize(inclusive=(True,
    # True)): the number of samples replaced on each side is floor(n * limit).
    # The NumPy fallback must agree with that on every valid (non-NaN) sample
    # over unsorted data with missing (NaN) entries, while preserving
    # NaN-as-missing (scipy's masked-array path overwrites masked entries, so
    # only the non-NaN positions are compared).
    scipy_winsorize = pytest.importorskip("scipy.stats.mstats").winsorize

    # Unsorted, per-column data with a NaN (missing) sample in two columns.
    arr = np.array(
        [
            [1.0, np.nan, 9.0],
            [9.0, 5.0, 1.0],
            [5.0, 3.0, 7.0],
            [3.0, 7.0, 2.0],
            [7.0, 2.0, 4.0],
            [2.0, 4.0, np.nan],
            [4.0, 1.0, 6.0],
            [8.0, 6.0, 3.0],
            [6.0, 8.0, 8.0],
            [10.0, 9.0, 5.0],
        ],
        dtype=np.float32,
    )
    limits = (0.2, 0.2)
    out = _winsorize_axis0_numpy(arr, limits)

    for c in range(arr.shape[1]):
        col = arr[:, c]
        valid = ~np.isnan(col)
        masked = np.ma.array(col, mask=~valid)
        ref = np.asarray(
            scipy_winsorize(masked, limits=limits, inclusive=(True, True)).filled(np.nan)
        )
        assert np.allclose(out[:, c][valid], ref[valid], rtol=1e-5), c
        assert np.all(np.isnan(out[:, c][~valid])), c


def test_process_batches_mixed_hw_hwc_wht_preserves_hwc(tmp_path, monkeypatch):
    # Defect 3 (HW broadcast): a later HW weight map must NOT downgrade an
    # already-accumulated HWC scientific WHT accumulator to HW, and must not
    # discard previously accumulated denominators.  An HW weight broadcasts
    # across channels; an HWC weight stays channel-specific.
    obj = qm.SeestarQueuedStacker.__new__(qm.SeestarQueuedStacker)
    obj.update_progress = lambda *a, **k: None
    obj.reference_shape = (2, 2)
    obj.stack_final_combine = "mean"
    obj._get_final_match_background = lambda default=False: False
    obj.ref_wcs_header = fits.Header()
    obj.use_gpu = False
    obj.interbatch_norm_active = False
    obj.enable_preview = False
    obj.cumulative_wht_path = str(tmp_path / "cumulative_WHT.npy")
    obj.cumulative_sum_memmap = np.lib.format.open_memmap(
        str(tmp_path / "cumulative_SUM.npy"), mode="w+",
        dtype=np.float32, shape=(2, 2, 3),
    )
    obj.cumulative_wht_memmap = np.lib.format.open_memmap(
        obj.cumulative_wht_path, mode="w+",
        dtype=np.float32, shape=(2, 2, 3),
    )

    # First input: HWC weight with differing channels.  Second: HW weight
    # (channel-invariant) with an HWC numerator.
    responses = [
        (np.full((2, 2, 3), [30.0, 40.0, 40.0], dtype=np.float32),
         np.full((2, 2, 3), [3.0, 4.0, 4.0], dtype=np.float32)),
        (np.full((2, 2, 3), [20.0, 20.0, 20.0], dtype=np.float32),
         np.full((2, 2), 2.0, dtype=np.float32)),
    ]

    def fake_worker(fits_path, ref_wcs_header=None, shape_out=None,
                    use_gpu=False, match_background=False):
        return responses.pop(0)

    monkeypatch.setattr(
        obj, "_load_and_prepare_simple",
        lambda p: (np.zeros((2, 2, 3), np.float32), None, fits.Header(), None),
    )
    monkeypatch.setattr(qm, "_reproject_worker", fake_worker)

    obj._process_batches(["a.fits", "b.fits"])

    assert obj.cumulative_wht_memmap.shape == (2, 2, 3)
    assert np.allclose(obj.cumulative_wht_memmap[0, 0], [5.0, 6.0, 6.0], rtol=1e-5)
    assert np.allclose(obj.cumulative_sum_memmap[0, 0], [50.0, 60.0, 60.0], rtol=1e-5)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = obj.cumulative_sum_memmap / obj.cumulative_wht_memmap
    assert np.allclose(result[0, 0], [10.0, 10.0, 10.0], rtol=1e-5)


def test_apply_rewinsor_no_rejection_preserves_survivors():
    # apply_rewinsor=True must substitute *rejected* samples only.  A run with
    # no rejection must return the original mean and full weight, not a
    # globally winsorized mean.  Asymmetric values plus nonzero winsor limits
    # expose the bug: clamping every survivor to the 20/80% winsor bounds
    # would collapse the mean from 104.5 to 5.5.
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0]
    data = [np.array([[v]], dtype=np.float32) for v in values]

    result, W, pct = _stack_winsorized_sigma(
        data, None, kappa=1e9, winsor_limits=(0.2, 0.2),
        apply_rewinsor=True, return_weights=True,
    )
    assert np.isclose(result[0, 0], 104.5, rtol=1e-5), result[0, 0]
    assert np.isclose(W[0, 0], 10.0, rtol=1e-5)
    assert pct == 0.0

    # Weighted variant: same guarantee under per-frame weights.
    weights = np.array(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0], dtype=np.float32
    )
    expected = float(np.sum(np.asarray(values) * weights) / np.sum(weights))
    result2, W2, _ = _stack_winsorized_sigma(
        data, weights, kappa=1e9, winsor_limits=(0.2, 0.2),
        apply_rewinsor=True, return_weights=True,
    )
    assert np.isclose(result2[0, 0], expected, rtol=1e-5), result2[0, 0]
    assert np.isclose(W2[0, 0], float(np.sum(weights)), rtol=1e-5)


# ---------------------------------------------------------------------------
# 9. HSI-1 corrective regression (ZSSS-HSI-1-C3)
# ---------------------------------------------------------------------------


def test_winsorized_sigma_scipy_remasks_missing_entries(monkeypatch):
    # scipy.stats.mstats.winsorize clears the mask and overwrites masked
    # (missing) entries with the high winsor bound.  Without remasking, that
    # synthetic bound enters mu_w/sigma_w and changes which *genuine* samples
    # survive, so a spatially invalid sample silently alters the result.
    # This pins cross-path (NumPy vs SciPy) equality and NaN-as-missing:
    # removing the missing sample must not change anything.
    scipy_winsorize = pytest.importorskip("scipy.stats.mstats").winsorize

    import seestar.core.stack_methods as sm

    # Adversarial witness: one missing sample plus five genuine samples.
    # Under SciPy the NaN position is overwritten with the high bound (11),
    # inflating mu_w/sigma_w enough to flip the boundary pair 8/11 so that
    # the genuine sample 11 survives and 8 is rejected (result 11.0, pct 80%)
    # instead of the correct 8/9 survivors (result 8.6, pct 60%).
    values = [14.0, 8.0, 0.0, 9.0, np.nan, 11.0]
    data = [np.array([[v]], dtype=np.float32) for v in values]
    data_without_missing = [
        np.array([[v]], dtype=np.float32) for v in values if not np.isnan(v)
    ]

    def run(data):
        return _stack_winsorized_sigma(
            data, None, kappa=1.0, winsor_limits=(0.2, 0.2),
            apply_rewinsor=True, return_weights=True,
        )

    with monkeypatch.context() as m:
        m.setattr(sm, "SCIPY_AVAILABLE", False)
        m.setattr(sm, "_scipy_winsorize", None)
        res_np, W_np, pct_np = run(data)

        m.setattr(sm, "SCIPY_AVAILABLE", True)
        m.setattr(sm, "_scipy_winsorize", scipy_winsorize)
        res_sp, W_sp, pct_sp = run(data)

        # Missing samples do not affect statistics: dropping the NaN frame
        # must reproduce the exact same result, weight and rejection
        # percentage (computed deterministically on the NumPy path).
        m.setattr(sm, "SCIPY_AVAILABLE", False)
        m.setattr(sm, "_scipy_winsorize", None)
        res_ref, W_ref, pct_ref = run(data_without_missing)

    # Cross-path equality of result, weight and rejection percentage.
    assert np.isclose(res_np[0, 0], res_sp[0, 0], rtol=1e-5), (res_np, res_sp)
    assert np.isclose(W_np[0, 0], W_sp[0, 0], rtol=1e-5), (W_np, W_sp)
    assert np.isclose(pct_np, pct_sp, atol=1e-5), (pct_np, pct_sp)

    # The surviving samples are {8, 9}; outliers 0 and 14 are rejected and 11
    # is rejected (its survival would be an artifact of the synthetic bound).
    # With apply_rewinsor the rejected values are substituted by the survivor
    # bounds, giving (8 + 9 + 8 + 9 + 9) / 5 = 8.6 and W == 5.
    assert np.isclose(res_np[0, 0], 8.6, rtol=1e-5), res_np[0, 0]
    assert np.isclose(W_np[0, 0], 5.0, rtol=1e-5), W_np[0, 0]
    assert np.isclose(pct_np, 60.0, atol=1e-5), pct_np

    assert np.isclose(res_ref[0, 0], res_np[0, 0], rtol=1e-5), (res_ref, res_np)
    assert np.isclose(W_ref[0, 0], W_np[0, 0], rtol=1e-5)
    assert np.isclose(pct_ref, pct_np, atol=1e-5)


# ---------------------------------------------------------------------------
# 10. HSI-2A: per-channel classic intermediate persistence
# ---------------------------------------------------------------------------


def _write_classic_batch(output_folder, batch_idx, V, W):
    """Drive the real writer with a minimal stacker and return (sci, wht_paths)."""
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    obj = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    obj.output_folder = str(output_folder)
    obj.update_progress = lambda *a, **k: None
    obj.solve_batches = False
    obj.reproject_coadd_final = False
    obj.reference_header_for_wcs = None
    obj.reference_wcs_object = None
    obj.reference_shape = None
    obj.apply_master_tile_crop = False
    obj.master_tile_crop_percent_decimal = 0.0
    obj.intermediate_classic_batch_files = []
    obj.unsolved_classic_batch_files = set()
    obj._last_classic_batch_solved = True
    return obj._save_and_solve_classic_batch(
        np.asarray(V, dtype=np.float32),
        np.asarray(W, dtype=np.float32),
        fits.Header(),
        batch_idx,
    )


def test_classic_batch_writer_persists_per_channel_wht(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    V = np.array([[[10.0, 10.0, 10.0]]], dtype=np.float32)
    W = np.array([[[3.0, 4.0, 5.0]]], dtype=np.float32)

    sci, wht_paths = _write_classic_batch(out_dir, 0, V, W)

    assert len(wht_paths) == 3
    vals = []
    for ch, p in enumerate(wht_paths):
        with fits.open(p) as hdul:
            hdr = hdul[0].header
            data = hdul[0].data
        assert hdr.get("HSIVER") == 2, f"sidecar {ch} missing HSIVER=2"
        assert hdr.get("WHTSEM") == "EFF_DENOM", f"sidecar {ch} missing WHTSEM"
        assert int(hdr.get("WHTCH")) == ch, f"sidecar {ch} wrong WHTCH"
        assert int(hdr.get("WHTNCH")) == 3, f"sidecar {ch} wrong WHTNCH"
        vals.append(float(np.asarray(data)[0, 0]))
    assert vals == [3.0, 4.0, 5.0], vals

    with fits.open(sci) as hdul:
        assert hdul[0].header.get("HSIVER") == 2
        assert hdul[0].header.get("WHTNCH") == 3


def test_classic_batch_writer_broadcasts_legacy_2d_wht(tmp_path):
    # A channel-invariant (2-D) denominator must be broadcast to every sidecar,
    # not collapsed (it was never per-channel).
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    V = np.array([[[10.0, 20.0, 30.0]]], dtype=np.float32)
    W = np.array([[7.0]], dtype=np.float32)

    sci, wht_paths = _write_classic_batch(out_dir, 0, V, W)

    for ch, p in enumerate(wht_paths):
        with fits.open(p) as hdul:
            assert float(np.asarray(hdul[0].data)[0, 0]) == 7.0
            assert hdul[0].header.get("HSIVER") == 2
            assert int(hdul[0].header.get("WHTCH")) == ch


def test_legacy_collapsed_sidecar_refused_scientifically(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # An unversioned legacy snapshot: three identical 2-D sidecars (the old
    # collapsed-per-channel bug).  Presented as scientific provenance this must
    # be refused, not broadcast across channels.
    wht_paths = []
    for ch in range(3):
        p = out_dir / f"classic_batch_000_wht_{ch}.fits"
        fits.PrimaryHDU(data=np.full((1, 1), 4.0, dtype=np.float32)).writeto(
            p, overwrite=True
        )
        wht_paths.append(str(p))

    with pytest.raises(ValueError, match="legacy collapsed"):
        qm._load_classic_batch_wht(wht_paths, 3, (1, 1))

    # The clearly non-scientific display path may still derive a broadcast view.
    cov = qm._load_classic_batch_wht(wht_paths, 3, (1, 1), display_only=True)
    assert cov.shape == (3, 1, 1)
    assert np.allclose(cov, 4.0)


def test_reproject_classic_batches_zm_uses_matching_channel_weights(
    tmp_path, monkeypatch
):
    import seestar.enhancement.reproject_utils as ru
    from astropy.wcs import WCS
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def tiny_wcs_header():
        hdr = fits.Header()
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["CRVAL1"] = 0.0
        hdr["CRVAL2"] = 0.0
        hdr["CRPIX1"] = 1.0
        hdr["CRPIX2"] = 1.0
        hdr["CDELT1"] = 1e-4
        hdr["CDELT2"] = 1e-4
        hdr["CD1_1"] = 1e-4
        hdr["CD1_2"] = 0.0
        hdr["CD2_1"] = 0.0
        hdr["CD2_2"] = 1e-4
        hdr["CUNIT1"] = "deg"
        hdr["CUNIT2"] = "deg"
        return hdr

    # Two batches with channel-specific effective denominators.
    V0 = np.array([[[10.0, 10.0, 10.0]]], dtype=np.float32)
    W0 = np.array([[[3.0, 4.0, 5.0]]], dtype=np.float32)
    V1 = np.array([[[20.0, 20.0, 20.0]]], dtype=np.float32)
    W1 = np.array([[[1.0, 2.0, 7.0]]], dtype=np.float32)

    sci0, wht0 = _write_classic_batch(out_dir, 0, V0, W0)
    sci1, wht1 = _write_classic_batch(out_dir, 1, V1, W1)

    # Inject a valid WCS so the reader treats the batches as already solved.
    for sci in (sci0, sci1):
        with fits.open(sci, memmap=False) as hdul:
            data = hdul[0].data
        hdr = tiny_wcs_header()
        fits.PrimaryHDU(data=data, header=hdr).writeto(sci, overwrite=True)

    captured = []

    def fake_coadd(
        inputs_ch,
        output_projection=None,
        shape_out=None,
        input_weights=None,
        reproject_function=None,
        combine_function="mean",
        match_background=True,
        **kw,
    ):
        imgs = [np.asarray(im, dtype=np.float64) for im, _w in inputs_ch]
        wts = [np.asarray(w, dtype=np.float64) for w in input_weights]
        num = sum(im * w for im, w in zip(imgs, wts))
        den = sum(wts)
        with np.errstate(divide="ignore", invalid="ignore"):
            sci = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        captured.append((imgs, wts, sci, den))
        return sci.astype(np.float32), den.astype(np.float32)

    monkeypatch.setattr(ru, "reproject_and_coadd", fake_coadd)

    obj = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    obj.update_progress = lambda *a, **k: None
    obj.unsolved_classic_batch_files = set()
    obj.reproject_coadd_final = False
    # batch_size != 0/1 keeps the zm reader on the primary astropy path and
    # avoids the blank-frame local fallback retry for this focused test.
    obj.batch_size = 2
    obj.reference_wcs_object = WCS(tiny_wcs_header(), naxis=2)
    obj.reference_shape = (1, 1)
    obj.reference_header_for_wcs = tiny_wcs_header()
    obj.freeze_reference_wcs = False
    obj.drizzle_scale = 1.0
    obj.drizzle_active_session = False
    obj._get_final_match_background = lambda default=False: False
    obj._ensure_reference_wcs_for_mode0 = lambda bf: None
    obj._crop_to_wht_bbox = lambda img, cov, wcs: (img, cov, wcs)
    obj._crop_to_reference_wcs = lambda img, cov, wcs: (img, cov, wcs)
    obj._save_final_stack = lambda *a, **k: None

    obj._reproject_classic_batches_zm([(sci0, wht0), (sci1, wht1)])

    assert len(captured) == 3
    expected_w = [[3.0, 1.0], [4.0, 2.0], [5.0, 7.0]]
    expected_sci = [12.5, 80.0 / 6.0, 190.0 / 12.0]
    for ch in range(3):
        imgs, wts, sci, den = captured[ch]
        assert [float(w[0, 0]) for w in wts] == expected_w[ch], (ch, wts)
        assert [float(im[0, 0]) for im in imgs] == [10.0, 20.0], (ch, imgs)
        assert np.isclose(float(sci[0, 0]), expected_sci[ch], rtol=1e-5), (
            ch,
            sci,
        )


# ---------------------------------------------------------------------------
# 11. HSI-2A-C1: fail-closed writer input, complete v2 sidecar contract, and
#     crop parity between the science cube and every WHT channel.
# ---------------------------------------------------------------------------


def test_hsi_validate_wht_accepts_hw_and_hwc():
    qm._hsi_validate_wht(np.ones((2, 2), dtype=np.float32), (2, 2), 3)
    qm._hsi_validate_wht(np.ones((2, 2, 3), dtype=np.float32), (2, 2), 3)


def test_hsi_validate_wht_rejects_bad_rank():
    with pytest.raises(ValueError, match="2-D.*3-D"):
        qm._hsi_validate_wht(
            np.ones((2, 2, 3, 1), dtype=np.float32), (2, 2), 3
        )


def test_hsi_validate_wht_rejects_spatial_shape_mismatch():
    with pytest.raises(ValueError, match="spatial shape"):
        qm._hsi_validate_wht(np.ones((3, 2), dtype=np.float32), (2, 2), 3)


def test_hsi_validate_wht_rejects_channel_count_mismatch():
    # A 3-D WHT with too few channels must never fall back to channel 0.
    with pytest.raises(ValueError, match="channels"):
        qm._hsi_validate_wht(np.ones((2, 2, 2), dtype=np.float32), (2, 2), 3)


def test_classic_batch_writer_refuses_malformed_wht(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    V = np.ones((1, 1, 3), dtype=np.float32)

    # Too few channels (previously silently fell back to channel 0).
    with pytest.raises(ValueError, match="channels"):
        _write_classic_batch(out_dir, 0, V, np.ones((1, 1, 2), dtype=np.float32))
    # Spatial shape does not match the science cube.
    with pytest.raises(ValueError, match="spatial shape"):
        _write_classic_batch(out_dir, 0, V, np.ones((2, 2, 3), dtype=np.float32))
    # Unsupported rank (previously returned ones).
    with pytest.raises(ValueError, match="2-D.*3-D"):
        _write_classic_batch(
            out_dir, 0, V, np.ones((1, 1, 3, 1), dtype=np.float32)
        )

    # Nothing was written for any of the refused inputs (fail closed before
    # any science cube or sidecar is persisted).
    batch_dir = out_dir / "classic_batch_outputs"
    assert list(batch_dir.iterdir()) == []


def _write_versioned_sidecars(out_dir, shape, sem="EFF_DENOM", nch=3,
                              wch_values=None, version=2):
    """Write three v2 sidecars; return their paths."""
    paths = []
    for ch in range(3):
        p = out_dir / f"w_{ch}.fits"
        hdr = fits.Header()
        hdr["HSIVER"] = version
        hdr["WHTSEM"] = sem
        hdr["WHTNCH"] = nch
        hdr["WHTCH"] = ch if wch_values is None else wch_values[ch]
        fits.writeto(p, np.ones(shape, dtype=np.float32), hdr, overwrite=True)
        paths.append(str(p))
    return paths


def test_sidecar_corrupt_metadata_fails_closed(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with pytest.raises(ValueError, match="WHTSEM"):
        qm._load_classic_batch_wht(
            _write_versioned_sidecars(out_dir, (2, 2), sem="COLLAPSED"),
            3, (2, 2),
        )
    with pytest.raises(ValueError, match="WHTNCH"):
        qm._load_classic_batch_wht(
            _write_versioned_sidecars(out_dir, (2, 2), nch=4), 3, (2, 2)
        )
    with pytest.raises(ValueError, match="WHTCH"):
        qm._load_classic_batch_wht(
            _write_versioned_sidecars(
                out_dir, (2, 2), wch_values=[0, 0, 2]
            ),
            3, (2, 2),
        )
    # Shape mismatch (sidecars are 3x3, requested 2x2).
    with pytest.raises(ValueError, match="shape"):
        qm._load_classic_batch_wht(
            _write_versioned_sidecars(out_dir, (3, 3)), 3, (2, 2)
        )


def test_sidecar_finite_nonnegative_denominator_policy(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    paths = []
    for ch, val in enumerate([np.nan, np.inf, -5.0]):
        p = out_dir / f"w_{ch}.fits"
        hdr = fits.Header()
        hdr["HSIVER"] = 2
        hdr["WHTSEM"] = "EFF_DENOM"
        hdr["WHTNCH"] = 3
        hdr["WHTCH"] = ch
        fits.writeto(
            p, np.full((1, 1), val, dtype=np.float32), hdr, overwrite=True
        )
        paths.append(str(p))
    cov = qm._load_classic_batch_wht(paths, 3, (1, 1))
    # Non-finite and negative denominators are zero contribution, never NaN.
    assert np.allclose(cov, 0.0)
    assert np.all(np.isfinite(cov))


def test_reproject_classic_batches_zm_crop_aligns_v_and_w(tmp_path, monkeypatch):
    import seestar.enhancement.reproject_utils as ru
    from astropy.wcs import WCS
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def tiny_wcs_header():
        hdr = fits.Header()
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["CRVAL1"] = 0.0
        hdr["CRVAL2"] = 0.0
        hdr["CRPIX1"] = 1.0
        hdr["CRPIX2"] = 1.0
        hdr["CDELT1"] = 1e-4
        hdr["CDELT2"] = 1e-4
        hdr["CD1_1"] = 1e-4
        hdr["CD1_2"] = 0.0
        hdr["CD2_1"] = 0.0
        hdr["CD2_2"] = 1e-4
        hdr["CUNIT1"] = "deg"
        hdr["CUNIT2"] = "deg"
        return hdr

    H = W = 6
    rng = np.random.default_rng(11)
    V0 = rng.normal(10.0, 1.0, size=(H, W, 3)).astype(np.float32)
    W0 = rng.uniform(1.0, 5.0, size=(H, W, 3)).astype(np.float32)
    V1 = rng.normal(20.0, 1.0, size=(H, W, 3)).astype(np.float32)
    W1 = rng.uniform(1.0, 5.0, size=(H, W, 3)).astype(np.float32)

    sci0, wht0 = _write_classic_batch(out_dir, 0, V0, W0)
    sci1, wht1 = _write_classic_batch(out_dir, 1, V1, W1)

    for sci in (sci0, sci1):
        with fits.open(sci, memmap=False) as hdul:
            data = hdul[0].data
        hdr = tiny_wcs_header()
        fits.PrimaryHDU(data=data, header=hdr).writeto(sci, overwrite=True)

    captured = []

    def fake_coadd(
        inputs_ch,
        output_projection=None,
        shape_out=None,
        input_weights=None,
        reproject_function=None,
        combine_function="mean",
        match_background=True,
        **kw,
    ):
        imgs = [np.asarray(im, dtype=np.float64) for im, _w in inputs_ch]
        wts = [np.asarray(w, dtype=np.float64) for w in input_weights]
        captured.append((imgs, wts))
        num = sum(im * w for im, w in zip(imgs, wts))
        den = sum(wts)
        with np.errstate(divide="ignore", invalid="ignore"):
            sci = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return sci.astype(np.float32), den.astype(np.float32)

    monkeypatch.setattr(ru, "reproject_and_coadd", fake_coadd)

    obj = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    obj.update_progress = lambda *a, **k: None
    obj.unsolved_classic_batch_files = set()
    obj.reproject_coadd_final = False
    obj.batch_size = 2
    obj.reference_wcs_object = WCS(tiny_wcs_header(), naxis=2)
    obj.reference_shape = (H, W)
    obj.reference_header_for_wcs = tiny_wcs_header()
    obj.freeze_reference_wcs = False
    obj.drizzle_scale = 1.0
    obj.drizzle_active_session = False
    obj.apply_master_tile_crop = True
    obj.master_tile_crop_percent_decimal = 0.25
    obj._get_final_match_background = lambda default=False: False
    obj._ensure_reference_wcs_for_mode0 = lambda bf: None
    obj._crop_to_wht_bbox = lambda img, cov, wcs: (img, cov, wcs)
    obj._crop_to_reference_wcs = lambda img, cov, wcs: (img, cov, wcs)
    obj._save_final_stack = lambda *a, **k: None

    obj._reproject_classic_batches_zm([(sci0, wht0), (sci1, wht1)])

    # Crop window: dh = dw = int(6 * 0.25) = 1 -> (4, 4) cropped.
    dh = dw = 1
    expected_h = H - 2 * dh
    expected_w = W - 2 * dw
    assert len(captured) == 3
    for ch in range(3):
        imgs, wts = captured[ch]
        assert [im.shape for im in imgs] == [(expected_h, expected_w)] * 2
        assert [w.shape for w in wts] == [(expected_h, expected_w)] * 2
        # V and every W channel are cropped with the exact same window.
        assert np.allclose(imgs[0], V0[dh:-dh, dw:-dw, ch], rtol=1e-5)
        assert np.allclose(imgs[1], V1[dh:-dh, dw:-dw, ch], rtol=1e-5)
        assert np.allclose(wts[0], W0[dh:-dh, dw:-dw, ch], rtol=1e-5)
        assert np.allclose(wts[1], W1[dh:-dh, dw:-dw, ch], rtol=1e-5)

    # Non-destructive persistence: the durable v2 SCI/W pair stays at the full,
    # mutually compatible shape and reloads to the original values.  The crop
    # is applied in memory only, so repeated use never crops the pair again.
    for sci, wht_paths, V_full, W_full in (
        (sci0, wht0, V0, W0),
        (sci1, wht1, V1, W1),
    ):
        with fits.open(sci, memmap=False) as hdul:
            persisted_v = np.moveaxis(hdul[0].data, 0, -1)
        assert persisted_v.shape == (H, W, 3), persisted_v.shape
        assert np.allclose(persisted_v, V_full, rtol=1e-5)
        cov = qm._load_classic_batch_wht(wht_paths, 3, (H, W))
        assert cov.shape == (3, H, W), cov.shape
        assert np.allclose(cov, np.moveaxis(W_full, -1, 0), rtol=1e-5)


# ---------------------------------------------------------------------------
# 12. HSI-2A-C2: finite/nonnegative writer output, torn-pair prevention, and
#     non-destructive late-crop persistence.
# ---------------------------------------------------------------------------


def test_classic_batch_writer_sanitizes_nonfinite_negative_wht(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    V = np.array([[[10.0, 10.0, 10.0]]], dtype=np.float32)
    # +inf must not become float32 max (np.nan_to_num default), a finite
    # negative must not be persisted, and NaN must become zero contribution.
    W = np.array([[[np.inf, -5.0, np.nan]]], dtype=np.float32)

    sci, wht_paths = _write_classic_batch(out_dir, 0, V, W)

    vals = []
    for p in wht_paths:
        with fits.open(p) as hdul:
            d = np.asarray(hdul[0].data, dtype=np.float32)
            assert np.all(np.isfinite(d)), d
            vals.append(float(d[0, 0]))
    assert vals == [0.0, 0.0, 0.0], vals


def test_torn_pair_bad_transformed_wht_leaves_prior_pair_intact(tmp_path, monkeypatch):
    from astropy.wcs import WCS
    from seestar.queuep.queue_manager import SeestarQueuedStacker
    import seestar.queuep.queue_manager as qm

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def tiny_wcs_header():
        hdr = fits.Header()
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["CRVAL1"] = 0.0
        hdr["CRVAL2"] = 0.0
        hdr["CRPIX1"] = 1.0
        hdr["CRPIX2"] = 1.0
        hdr["CDELT1"] = 1e-4
        hdr["CDELT2"] = 1e-4
        hdr["CD1_1"] = 1e-4
        hdr["CD1_2"] = 0.0
        hdr["CD2_1"] = 0.0
        hdr["CD2_2"] = 1e-4
        hdr["CUNIT1"] = "deg"
        hdr["CUNIT2"] = "deg"
        return hdr

    obj = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    obj.output_folder = str(out_dir)
    obj.update_progress = lambda *a, **k: None
    obj.solve_batches = False
    obj.reproject_coadd_final = False
    obj.reference_header_for_wcs = tiny_wcs_header()
    obj.reference_wcs_object = WCS(tiny_wcs_header(), naxis=2)
    obj.reference_shape = (2, 2)
    obj.apply_master_tile_crop = False
    obj.master_tile_crop_percent_decimal = 0.0
    obj.intermediate_classic_batch_files = []
    obj.unsolved_classic_batch_files = set()
    obj._last_classic_batch_solved = True

    V = np.ones((2, 2, 3), dtype=np.float32)
    W = np.full((2, 2, 3), 3.0, dtype=np.float32)

    calls = {"n": 0}

    def fake_reproject(data, input_wcs, ref_wcs, ref_shape):
        # V first, W second: return a transformed V whose spatial shape no
        # longer matches the transformed W (the torn-pair counterexample).
        calls["n"] += 1
        if calls["n"] == 1:
            return np.ones((3, 3, 3), dtype=np.float32)
        return np.ones((1, 1, 3), dtype=np.float32)

    monkeypatch.setattr(qm, "reproject_to_reference_wcs", fake_reproject)

    with pytest.raises(ValueError, match="spatial shape"):
        obj._save_and_solve_classic_batch(V, W, tiny_wcs_header(), 0)

    # The transformed-W validation failed *before* any rewrite, so the
    # previously persisted valid SCI/W pair is still mutually compatible and
    # unchanged.
    out = out_dir / "classic_batch_outputs"
    sci = out / "classic_batch_000.fits"
    with fits.open(sci, memmap=False) as hdul:
        persisted = hdul[0].data
    assert persisted.shape == (3, 2, 2), persisted.shape
    assert np.allclose(persisted, 1.0), persisted
    for ch in range(3):
        p = out / f"classic_batch_000_wht_{ch}.fits"
        with fits.open(p, memmap=False) as hdul:
            d = hdul[0].data
        assert d.shape == (2, 2), d.shape
        assert np.allclose(d, 3.0), d
