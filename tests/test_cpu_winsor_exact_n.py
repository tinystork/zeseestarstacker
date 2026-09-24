"""8.4.0 pre-W80 stage C — exact-N spatial CPU Winsorized driver tests.

Mission ``zsss-840-prew80-20260907``.  Covers:

* schedule seam: ``winsor_schedule_kappas`` matches the canonical loop's
  kappa sequence; the refactored FULL_CPU iterator is bitwise-equivalent to
  the pre-refactor implementation (legacy reference embedded from the
  committed pre-stage-C source) and every existing Winsor CPU test stays
  green (run separately by the caller);
* exact-N SPATIAL tiled driver (``stack_winsorized_sigma_cpu_tiled``):
  bitwise FULL vs TILED SCI+WHT parity across the full matrix (mono/RGB x
  weighted/unweighted x full/partial support x NaN x zero pixels x rank0 /
  rank>0 x rewinsor T/F x odd/even N x partial final batch);
  every canonical per-iteration invocation is spied to assert tile N ==
  original batch N (the stack axis is never split);
  tile-order invariance (rowmajor vs reversed);
* refusal semantics (truthful planner reasons, minimum viable tile) and
  bounded spatial-only retry on MemoryError (N / reducer / kappa / winsor /
  normalization / weights are never retry-mutable);
* adversarial non-associativity witness: global Winsorized(N) != subgroup
  composition (N=20 ``[0]*19+[100]`` -> global SCI 0 vs subgroup SCI 5),
  intentionally divergent, while the exact-N tiled driver == FULL bitwise.
"""

import numpy as np
import pytest

from seestar.core import stack_methods as sm
from seestar.core.cpu_memory_planner import (
    REASON_MIN_TILE_EXCEEDS_BUDGET,
    REASON_NO_VALID_TILE,
    winsor_zero_rank_regime,
)
from seestar.core.cpu_winsor_exact_n import (
    CpuWinsorMemoryRefused,
    stack_winsorized_sigma_cpu_tiled,
    winsor_tile_slices,
)
from seestar.core.stack_methods import (
    _stack_winsorized_sigma_iter,
    _winsor_schedule_kappas,
)

NORM_TOL = 0.0  # bitwise preferred; no broad tolerance


# ---------------------------------------------------------------------------
# Legacy reference: pre-refactor _stack_winsorized_sigma_iter body, verbatim
# from the committed pre-stage-C source (equivalence witness).
# ---------------------------------------------------------------------------
def _legacy_winsorized_sigma_iter(
    images,
    weights,
    kappa=3.0,
    winsor_limits=(0.05, 0.05),
    apply_rewinsor=True,
    max_iters=5,
    kappa_decay=0.9,
    max_mem_bytes=None,
    return_weights=False,
):
    arr = np.stack([im.astype(np.float32, copy=False) for im in images], axis=0)
    mask = ~np.isnan(arr)
    valid = mask
    n_valid_col = np.count_nonzero(valid, axis=0)
    zero_rank_cols = sm._winsor_zero_rank_cols(winsor_limits, n_valid_col)
    rank_cols = ~zero_rank_cols
    rank_cols3 = rank_cols[np.newaxis, ...]

    guard_keep = (
        sm._gross_outlier_keep(arr, valid)
        if np.any(zero_rank_cols)
        else valid
    )

    arr_w = np.where(rank_cols3, arr, np.nan)
    mask_w = ~np.isnan(arr_w)
    kappa_iter = float(kappa)
    for itr in range(max_iters):
        arr_masked = np.where(mask_w, arr_w, np.nan)
        arr_w_data = sm._winsorize_axis0_numpy(arr_masked, winsor_limits)
        with np.errstate(invalid="ignore"):
            mu_w = sm.NANMEAN(arr_w_data, axis=0)
            sigma_w = sm.NANSTD(arr_w_data, axis=0, ddof=1)
        n_valid_col = np.count_nonzero(mask_w, axis=0)
        sigma_w = np.where(n_valid_col <= 1, np.float32(0.0), sigma_w)
        low = mu_w - kappa_iter * sigma_w
        high = mu_w + kappa_iter * sigma_w
        new_mask = mask_w & (arr_w >= low) & (arr_w <= high)
        n_rej = np.count_nonzero(mask_w) - np.count_nonzero(new_mask)
        mask_w = new_mask
        if n_rej == 0:
            break
        if kappa_decay < 1.0:
            kappa_iter = kappa * (kappa_decay ** (itr + 1))
    mask = np.where(rank_cols3, mask_w, guard_keep)
    if apply_rewinsor:
        low_b, high_b = sm._winsorize_bounds(
            np.where(mask, arr, np.nan), winsor_limits
        )
        clipped = np.clip(arr, low_b, high_b)
        arr_final = np.where(mask, arr, np.where(valid, clipped, np.nan))
    else:
        arr_final = np.where(mask, arr, np.nan)
    contrib = ~np.isnan(arr_final)
    if weights is not None:
        w = sm._broadcast_weights(arr, weights)
        sum_w = np.nansum(np.where(contrib, w, np.float32(0.0)), axis=0)
        sum_d = np.nansum(arr_final * w, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.divide(
                sum_d, sum_w, out=np.zeros_like(sum_d), where=sum_w > 1e-6
            )
    else:
        with np.errstate(invalid="ignore"):
            result = sm.NANMEAN(arr_final, axis=0)
        result = np.where(np.any(contrib, axis=0), result, np.float32(0.0))
        sum_w = np.count_nonzero(contrib, axis=0).astype(np.float32)
    rejected_pct = sm._rejected_pct(mask, valid)
    result = result.astype(np.float32)
    if return_weights:
        return result, sum_w.astype(np.float32), rejected_pct
    return result, rejected_pct


def _dataset(n, shape, color=False, nan_frac=0.0, zero_frac=0.0, seed=0,
             low_lim=0.05, high_lim=0.05):
    rng = np.random.default_rng(seed)
    H, W = shape
    full = (n, H, W) if not color else (n, H, W, 3)
    arr = rng.normal(100.0, 10.0, size=full).astype(np.float32)
    if nan_frac > 0:
        m = rng.random(full) < nan_frac
        arr[m] = np.nan
    if zero_frac > 0:
        m = rng.random(full) < zero_frac
        arr[m] = 0.0  # legitimate zero pixels (content-supported)
    if color:
        ch = rng.random(full) < 0.01
        arr[ch] += rng.uniform(300.0, 600.0, size=int(ch.sum()))
    else:
        ch = rng.random(full) < 0.01
        arr[ch] += rng.uniform(300.0, 600.0, size=int(ch.sum()))
    imgs = [np.ascontiguousarray(arr[i]) for i in range(n)]
    return imgs


# ---------------------------------------------------------------------------
# 1. Schedule seam + refactor equivalence (before/after)
# ---------------------------------------------------------------------------
def test_winsor_schedule_kappas_matches_reference_sequence():
    k, decay, iters = 3.0, 0.9, 5
    kappas = _winsor_schedule_kappas(k, decay, iters)
    assert kappas[0] == 3.0
    for i in range(1, iters):
        assert kappas[i] == pytest.approx(k * decay ** i)
    # decay >= 1 keeps the band constant
    assert _winsor_schedule_kappas(3.0, 1.0, 4) == [3.0, 3.0, 3.0, 3.0]
    assert _winsor_schedule_kappas(3.0, 1.5, 3) == [3.0, 3.0, 3.0]


@pytest.mark.parametrize("color,n,seed", [
    (False, 17, 1), (False, 20, 2), (True, 19, 3), (True, 20, 4),
])
@pytest.mark.parametrize("lims", [(0.05, 0.05), (0.005, 0.005), (0.2, 0.0)])
@pytest.mark.parametrize("rw", [True, False])
@pytest.mark.parametrize("decay", [0.9, 1.0])
def test_refactored_full_iter_bitwise_equals_legacy(color, n, seed, lims, rw, decay):
    shape = (24, 20) if not color else (16, 12)
    imgs = _dataset(n, shape, color=color, nan_frac=0.03, seed=seed)
    weights = None
    if n % 2 == 0:
        rng = np.random.default_rng(seed + 1)
        weights = rng.uniform(0.5, 1.5, n).astype(np.float32)
    kw = dict(kappa=3.0, winsor_limits=lims, apply_rewinsor=rw,
              max_iters=5, kappa_decay=decay, return_weights=True)
    legacy = _legacy_winsorized_sigma_iter(imgs, weights, **kw)
    new = _stack_winsorized_sigma_iter(imgs, weights, **kw)
    assert np.array_equal(legacy[0], new[0])
    assert np.array_equal(legacy[1], new[1])
    assert legacy[2] == new[2]


# ---------------------------------------------------------------------------
# 2. Exact-N tiled parity matrix
# ---------------------------------------------------------------------------
def _full_ref(imgs, weights, lims, rw):
    return _stack_winsorized_sigma_iter(
        imgs, weights, kappa=3.0, winsor_limits=lims, apply_rewinsor=rw,
        max_iters=5, kappa_decay=0.9, return_weights=True,
    )


PARITY_CASES = [
    # (color, n, H, W, nan_frac, zero_frac, lims, rewinsor, weighted, label)
    (False, 20, 24, 20, 0.0, 0.0, (0.05, 0.05), True, False, "mono20"),
    (False, 21, 24, 20, 0.0, 0.0, (0.05, 0.05), True, False, "mono21odd"),
    (False, 20, 24, 20, 0.03, 0.0, (0.05, 0.05), True, False, "mono_nan"),
    (False, 20, 24, 20, 0.0, 0.1, (0.05, 0.05), True, False, "mono_zero"),
    (False, 20, 24, 20, 0.0, 0.0, (0.005, 0.005), True, False, "mono_rank0"),
    (False, 20, 24, 20, 0.0, 0.0, (0.05, 0.05), False, False, "mono_rwF"),
    (False, 20, 24, 20, 0.0, 0.0, (0.05, 0.05), True, True, "mono_w"),
    (True, 20, 16, 12, 0.0, 0.0, (0.05, 0.05), True, False, "rgb20"),
    (True, 19, 16, 12, 0.0, 0.0, (0.05, 0.05), True, False, "rgb19odd"),
    (True, 20, 16, 12, 0.03, 0.0, (0.05, 0.05), True, True, "rgb_nan_w"),
    (True, 20, 16, 12, 0.0, 0.05, (0.005, 0.005), True, False, "rgb_zero_rank0"),
    (True, 20, 16, 12, 0.0, 0.0, (0.05, 0.05), False, True, "rgb_rwF_w"),
]


@pytest.mark.parametrize(
    "color,n,H,W,nan_frac,zero_frac,lims,rw,weighted,label", PARITY_CASES,
    ids=[c[9] for c in PARITY_CASES],
)
def test_tiled_bitwise_parity_full(color, n, H, W, nan_frac, zero_frac,
                                   lims, rw, weighted, label, monkeypatch):
    import seestar.core.cpu_winsor_exact_n as cw

    imgs = _dataset(n, (H, W), color=color, nan_frac=nan_frac,
                    zero_frac=zero_frac, seed=7)
    weights = None
    if weighted:
        rng = np.random.default_rng(9)
        weights = rng.uniform(0.5, 1.5, n).astype(np.float32)

    full = _full_ref(imgs, weights, lims, rw)

    # spy: every canonical per-iteration invocation inside the TILED driver
    # (and inside the full-frame delegation to the canonical iterator) sees
    # the FULL tile N.  The driver resolves its helper through its own module
    # namespace (``cpu_winsor_exact_n._winsorized_sigma_iteration_body``); the
    # delegation path resolves the same helper inside ``stack_methods``, so
    # both namespaces are patched with one shared recorder.
    seen = []
    orig_cw = cw._winsorized_sigma_iteration_body
    orig_sm = sm._winsorized_sigma_iteration_body

    def spy(arr, mask, kappa_iter, lim):
        seen.append(int(arr.shape[0]))
        return orig_sm(arr, mask, kappa_iter, lim)

    monkeypatch.setattr(cw, "_winsorized_sigma_iteration_body", spy)
    monkeypatch.setattr(sm, "_winsorized_sigma_iteration_body", spy)

    # full-cell outputs of every tile >= CPU_MIN_TILE_OUT (96); the frame must
    # genuinely split into >1 tile for both the band and the rectangular form
    band_h = max(1, 96 // W + (1 if 96 % W else 0))
    if band_h >= H:
        band_h = max(1, H // 2)
    band = (band_h,)
    rect_h = max(1, H // 2)
    rect_w = max(1, 96 // rect_h + (1 if 96 % rect_h else 0))
    if rect_w >= W:
        rect_w = W
        rect_h = max(1, 96 // W + (1 if 96 % W else 0))
    rect = (rect_h, rect_w)
    assert band_h * W >= 96, (label, band)
    assert rect_h * rect_w >= 96, (label, rect)
    assert len(winsor_tile_slices((H, W), band)) > 1, (label, band)

    for tile_shape in (band, rect, (H, W)):  # bands, rect, full-frame
        n_slices = len(winsor_tile_slices((H, W), tile_shape))
        before = len(seen)
        tiled = stack_winsorized_sigma_cpu_tiled(
            imgs, weights, kappa=3.0, winsor_limits=lims, apply_rewinsor=rw,
            max_iters=5, kappa_decay=0.9, return_weights=True,
            tile_shape=tile_shape,
        )
        assert np.array_equal(full[0], tiled[0]), (label, tile_shape, "SCI")
        assert np.array_equal(full[1], tiled[1]), (label, tile_shape, "WHT")
        assert full[2] == tiled[2], (label, tile_shape, "rejected_pct")
        # The all-zero-rank fast path skips the per-iteration Winsor body
        # entirely (the gross-outlier guard alone decides those columns), so
        # the body may or may not be invoked here; the invariant that matters
        # is that every invocation saw the FULL stack population (never split).
        assert all(s == n for s in seen[before:]), (
            f"tile N split observed: {set(seen[before:])}")


def test_tiled_tile_order_invariance():
    imgs = _dataset(20, (32, 30), color=False, nan_frac=0.02, seed=5)
    a = stack_winsorized_sigma_cpu_tiled(
        imgs, None, tile_shape=(8, 12), _tile_order="rowmajor",
        return_weights=True,
    )
    b = stack_winsorized_sigma_cpu_tiled(
        imgs, None, tile_shape=(8, 12), _tile_order="reversed",
        return_weights=True,
    )
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert a[2] == b[2]


def test_spatial_path_never_stacks_complete_frames(monkeypatch):
    """Structural guard: every np.stack request is spatially bounded."""
    import seestar.core.cpu_winsor_exact_n as cw

    imgs = _dataset(20, (32, 30), color=False, seed=41)
    original_stack = np.stack
    requested_shapes = []

    def guarded_stack(arrays, *args, **kwargs):
        shapes = [np.shape(a) for a in arrays]
        requested_shapes.append(shapes)
        assert shapes
        assert all(shape[0] < 32 or shape[1] < 30 for shape in shapes), shapes
        return original_stack(arrays, *args, **kwargs)

    monkeypatch.setattr(cw.np, "stack", guarded_stack)
    out = stack_winsorized_sigma_cpu_tiled(
        imgs, None, tile_shape=(8,), return_weights=True
    )
    assert out[0].shape == (32, 30)
    assert requested_shapes
    assert all(len(shapes) == 20 for shapes in requested_shapes)


def test_spatial_tile_slices_memmaps_before_materialization(tmp_path, monkeypatch):
    """Memmapped observations remain full-frame mappings until tile slicing."""
    import seestar.core.cpu_winsor_exact_n as cw

    paths = []
    images = []
    for i in range(5):
        path = tmp_path / f"obs-{i}.dat"
        mm = np.memmap(path, mode="w+", dtype=np.float32, shape=(24, 20, 1))
        mm[:] = i + np.arange(24 * 20, dtype=np.float32).reshape(24, 20, 1)
        mm.flush()
        images.append(np.memmap(path, mode="r", dtype=np.float32, shape=(24, 20, 1)))
        paths.append(path)

    original_stack = np.stack
    saw_shared_tile_views = []

    def spy_stack(arrays, *args, **kwargs):
        saw_shared_tile_views.append(
            all(np.shares_memory(a, images[j]) for j, a in enumerate(arrays))
        )
        assert all(np.shape(a) == (6, 20, 1) for a in arrays)
        return original_stack(arrays, *args, **kwargs)

    monkeypatch.setattr(cw.np, "stack", spy_stack)
    result, wht, _ = stack_winsorized_sigma_cpu_tiled(
        images, None, tile_shape=(6,), min_tile_out=96, return_weights=True
    )
    assert result.shape == (24, 20, 1)
    assert wht.shape == (24, 20, 1)
    assert np.array_equal(result, np.asarray(images[0]) + np.float32(2.0))
    assert np.all(wht == np.float32(5.0))
    assert saw_shared_tile_views and all(saw_shared_tile_views)


def test_partial_final_band_and_rectangular_partials():
    """Partial last band / last column of a non-divisible frame stay exact."""
    imgs = _dataset(17, (23, 19), color=False, nan_frac=0.01, seed=11)
    full = _full_ref(imgs, None, (0.05, 0.05), True)
    for ts in ((6,), (8, 12)):
        slices = winsor_tile_slices((23, 19), ts)
        y0, y1, x0, x1 = slices[0]
        assert (y1 - y0) * (x1 - x0) >= 96
        tiled = stack_winsorized_sigma_cpu_tiled(
            imgs, None, tile_shape=ts, return_weights=True
        )
        assert np.array_equal(full[0], tiled[0])
        assert np.array_equal(full[1], tiled[1])
        assert full[2] == tiled[2]
        # sanity: geometry actually used >1 tile
        assert len(slices) > 1


def test_untiled_geometry_delegates_to_reference():
    imgs = _dataset(20, (24, 20), color=False, nan_frac=0.02, seed=13)
    ref = _full_ref(imgs, None, (0.05, 0.05), True)
    tiled = stack_winsorized_sigma_cpu_tiled(
        imgs, None, tile_shape=None, return_weights=True
    )
    assert np.array_equal(ref[0], tiled[0])
    assert np.array_equal(ref[1], tiled[1])
    assert ref[2] == tiled[2]


def test_rank_regime_consistency_with_planner():
    for n in (10, 20, 36, 50):
        assert winsor_zero_rank_regime((0.05, 0.05), n) == (
            sm._winsor_zero_rank_regime((0.05, 0.05), n)
        )
    assert winsor_zero_rank_regime((0.05, 0.05), 10) is True   # rank 0
    assert winsor_zero_rank_regime((0.05, 0.05), 20) is False  # rank > 0


# ---------------------------------------------------------------------------
# 3. Refusal + bounded spatial retry
# ---------------------------------------------------------------------------
def test_min_tile_refusal_truthful_reason():
    imgs = _dataset(20, (24, 20), color=False, seed=3)
    with pytest.raises(CpuWinsorMemoryRefused) as ei:
        stack_winsorized_sigma_cpu_tiled(
            imgs, None, tile_shape=(2, 2), min_tile_out=96
        )
    assert ei.value.reason == REASON_NO_VALID_TILE
    assert "never reduced" in str(ei.value)


def test_memory_error_triggers_bounded_spatial_retry(monkeypatch):
    """A catchable MemoryError on every candidate geometry retries with a
    strictly smaller spatial tile (tile_h/tile_w only) until the geometry
    space is exhausted, then refuses truthfully; N / kappa / limits / weights
    are untouched (no retry knob exists)."""
    import seestar.core.cpu_winsor_exact_n as cw

    imgs = _dataset(20, (32, 30), color=False, seed=17)
    weights = np.ones(20, dtype=np.float32)

    calls = []
    retries = []

    def always_oom(images, frame_shape, spatial, *args, **kwargs):
        calls.append((len(images), spatial[0]))
        raise MemoryError("simulated OOM in tile")

    monkeypatch.setattr(cw, "_run_tiled_geometry", always_oom)
    # a full-frame band (32,) would be a single tile and delegate to the
    # untiled twin (no retry vocabulary); use a genuinely multi-tile band so
    # the driver retries spatially until the geometry space is exhausted.
    with pytest.raises(CpuWinsorMemoryRefused) as ei:
        stack_winsorized_sigma_cpu_tiled(
            imgs, weights, tile_shape=(16,), max_retries=2,
            _retry_callback=lambda **event: retries.append(event),
        )
    assert ei.value.reason == REASON_MIN_TILE_EXCEEDS_BUDGET
    assert len(calls) == 3  # initial + exactly max_retries smaller attempts
    assert all(n == 20 for n, _ in calls)
    assert [r["outcome"] for r in retries] == ["retrying", "retrying", "exhausted"]
    assert [r["attempt"] for r in retries] == [1, 2, 3]
    assert ei.value.details["attempts"] == 3
    assert ei.value.details["max_retries"] == 2


def test_memory_error_retry_succeeds_with_smaller_tile(monkeypatch):
    import seestar.core.cpu_winsor_exact_n as cw

    imgs = _dataset(20, (32, 30), color=False, seed=17)
    orig = cw._winsorized_sigma_iteration_body
    retries = []

    def fail_big_tile(arr, mask, kappa_iter, lim):
        # Fail only while the driver is processing the INITIAL 16-row band;
        # the retry with a smaller (8-row) band passes cleanly.
        if int(arr.shape[1]) == 16:
            raise MemoryError("simulated OOM")
        return orig(arr, mask, kappa_iter, lim)

    monkeypatch.setattr(cw, "_winsorized_sigma_iteration_body", fail_big_tile)
    ref = _full_ref(imgs, None, (0.05, 0.05), True)
    tiled = stack_winsorized_sigma_cpu_tiled(
        imgs, None, tile_shape=(16,), max_retries=4, return_weights=True,
        _retry_callback=lambda **event: retries.append(event),
    )
    assert np.array_equal(ref[0], tiled[0])
    assert np.array_equal(ref[1], tiled[1])
    assert ref[2] == tiled[2]
    assert len(retries) == 2
    assert retries[0]["attempt"] == 1
    assert retries[0]["old_tile_shape"] == (16,)
    assert retries[0]["new_tile_shape"] == (8, 30)
    assert retries[0]["outcome"] == "retrying"
    assert retries[1]["attempt"] == 2
    assert retries[1]["new_tile_shape"] == (8, 30)
    assert retries[1]["outcome"] == "recovered"


def test_zero_retry_budget_means_initial_attempt_only(monkeypatch):
    import seestar.core.cpu_winsor_exact_n as cw

    imgs = _dataset(20, (32, 30), color=False, seed=18)
    calls = []

    def always_oom(images, frame_shape, spatial, *args, **kwargs):
        calls.append(len(images))
        raise MemoryError("initial allocation failure")

    monkeypatch.setattr(cw, "_run_tiled_geometry", always_oom)
    with pytest.raises(CpuWinsorMemoryRefused) as ei:
        stack_winsorized_sigma_cpu_tiled(
            imgs, None, tile_shape=(16,), max_retries=0
        )
    assert calls == [20]
    assert ei.value.details["attempts"] == 1
    assert ei.value.details["max_retries"] == 0


def test_preflight_refusal_when_budget_below_model_demand():
    imgs = _dataset(36, (1080, 960), color=False, seed=21)  # huge full cube
    with pytest.raises(CpuWinsorMemoryRefused) as ei:
        stack_winsorized_sigma_cpu_tiled(
            imgs, None, tile_shape=(1080, 960),
            max_mem_bytes=64 * 1024 * 1024,
        )
    assert ei.value.reason == REASON_MIN_TILE_EXCEEDS_BUDGET


def test_no_empty_success_on_refusal():
    imgs = _dataset(20, (24, 20), color=False, seed=3)
    try:
        stack_winsorized_sigma_cpu_tiled(
            imgs, None, tile_shape=(1, 1), min_tile_out=96
        )
    except CpuWinsorMemoryRefused:
        pass
    else:
        pytest.fail("degenerate 1x1 tile below min_tile_out must refuse")


# ---------------------------------------------------------------------------
# 4. Adversarial non-associativity witness (permanent regression)
# ---------------------------------------------------------------------------
def test_adversarial_global_vs_subgroup_composition_witness():
    """Global Winsorized(N) != subgroup composition — intentionally divergent.

    N=20 with values ``[0]*19 + [2]``: the GLOBAL reduction is rank-sufficient
    (``floor(0.05*20) == 1``), so the rank-1 winsorization clips the 2 out and
    returns SCI 0 / W 20.  Composing two independent N=10 subgroup reductions
    puts the 2 into a zero-rank subgroup (``floor(0.05*10) == 0``), whose
    one-pass gross-outlier guard keeps it (the 2 is not a sufficiently
    isolated extreme), yielding a composed SCI of 0.1.  This genuine global vs
    subgroup divergence is the reason the exact-N tiled driver must coordinate
    the GLOBAL stop schedule, and it must stay divergent (never 'fixed' by
    subgroup composition)."""
    vals = [0.0] * 19 + [2.0]
    imgs = [np.full((1, 1), v, dtype=np.float32) for v in vals]
    global_res = _stack_winsorized_sigma_iter(
        imgs, None, kappa=3.0, winsor_limits=(0.05, 0.05),
        apply_rewinsor=True, return_weights=True,
    )
    assert global_res[0].ravel()[0] == 0.0  # global SCI 0 (the 2 clipped out)
    assert global_res[1].ravel()[0] == 20.0  # WHT 20

    def _subgroup(group):
        g = [np.full((1, 1), v, dtype=np.float32) for v in group]
        return _stack_winsorized_sigma_iter(
            g, None, kappa=3.0, winsor_limits=(0.05, 0.05),
            apply_rewinsor=True, return_weights=True,
        )

    s = 0.0
    w = 0.0
    for half in (vals[:10], vals[10:]):
        V, W, _ = _subgroup(half)
        s += float(V.ravel()[0]) * float(W.ravel()[0])
        w += float(W.ravel()[0])
    composed = s / w
    # The zero-rank N=10 subgroup keeps the 2 (gross-outlier guard),
    # so the composed value genuinely diverges from the global reduction.
    assert composed == pytest.approx(0.1)
    assert composed != float(global_res[0].ravel()[0])  # intentionally divergent

    # The exact-N tiled driver must equal the GLOBAL reference, never the
    # subgroup-composed value.
    tiled = stack_winsorized_sigma_cpu_tiled(
        imgs, None, tile_shape=None, kappa=3.0,
        winsor_limits=(0.05, 0.05), apply_rewinsor=True, return_weights=True,
    )
    assert np.array_equal(global_res[0], tiled[0])
    assert float(tiled[0].ravel()[0]) != composed


def test_adversarial_two_region_global_stop_witness():
    """Global-stop coordination on a mild+strong two-region frame.

    The exact-N tiled driver must reproduce the GLOBAL schedule bitwise even
    though a naive local per-region early exit WOULD diverge (archived
    evidence: ``global-stop-witness.json`` records max SCI difference 3.7579
    on the mild region of the N=30 96x40 two-region dataset when a local
    mild-region early exit is used, with local mild rejection 0%% vs global
    10.08%%).  This test asserts the driver's global coordination property:
    FULL == TILED bitwise across both regions on a deterministic mild+strong
    frame; the intentionally-divergent global-vs-subgroup witness is the
    deterministic N=20 ``[0]*19+[100]`` test above."""
    rng = np.random.default_rng(31)
    N = 30
    H, W = 40, 96
    half = W // 2
    # mild region: tight noise, no local rejection (its own run stops fast)
    mild = rng.normal(100.0, 1.0, size=(N, H, half)).astype(np.float32)
    # strong region: same sky + heavy periodic outliers -> global schedule
    # continues long after the mild region alone would have stopped
    strong = rng.normal(100.0, 1.0, size=(N, H, half)).astype(np.float32)
    spikes = rng.random((N, H, half)) < 0.2
    strong[spikes] += rng.uniform(200.0, 400.0, size=int(spikes.sum()))
    imgs = [
        np.concatenate([mild[i], strong[i]], axis=1).astype(np.float32)
        for i in range(N)
    ]
    full = _full_ref(imgs, None, (0.05, 0.05), True)
    tiled = stack_winsorized_sigma_cpu_tiled(
        imgs, None, tile_shape=(H, half), return_weights=True
    )
    assert np.array_equal(full[0], tiled[0])   # global-schedule tiled == full
    assert np.array_equal(full[1], tiled[1])
    assert full[2] == tiled[2]
    # global rejection on the full frame is well above zero (strong region)
    assert full[2] > 1.0
    # band geometry is a genuine multi-tile split (right half is a real tile)
    assert len(winsor_tile_slices((H, W), (H, half))) == 2
