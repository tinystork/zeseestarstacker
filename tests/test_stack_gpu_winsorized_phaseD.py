"""Phase D (Track P2) tests: order/rank temporary elimination of the CuPy
Winsorized Sigma Clip SLOW path (``_winsorize_axis0_cp`` /
``_winsorize_bounds_cp``).

CPU reference (``seestar/core/stack_methods.py::_stack_winsorized_sigma_iter``
-> ``_winsorize_axis0_numpy`` / ``_winsorize_bounds``) stays the SCIENTIFIC
AUTHORITY and is never modified.

What phase D changes
--------------------
The slow path no longer materializes the two full-size int64 temporaries
``order = argsort(sort_key)`` and ``rank = argsort(order)`` (N,H,W int64,
the largest VRAM consumers of the slow path):

1. rank-based replacement is replaced by VALUE clipping: the CPU replaces
   exactly the samples strictly below the ``lowidx`` order statistic with
   ``low_bound`` and strictly above the ``keep_idx`` order statistic
   (``rank >= upidx``) with ``high_bound``; samples equal to a boundary
   order statistic that the rank rule would "replace" keep an identical bit
   pattern, so the value tests select exactly the samples whose replacement
   changes a value and the writes are bit-identical to the rank scatter
   (ties at the boundary included);
2. ``sorted_vals`` comes from a direct ``cp.sort`` of the NaN-sentinel key
   instead of ``argsort`` + ``take_along_axis``.

The ONLY input class value clipping cannot reproduce is the degenerate
OVERLAP corner (both sides active with ``floor(low * n) + floor(high * n) >
n`` on some column): there the sequential low-then-high rank replacement can
split one boundary tie group between the two bounds, so those inputs fall
back to the exact pre-Phase-D rank implementation
(``_winsorize_axis0_rank_path_cp``).  The Phase C zero-rank fast path is
untouched and still skips both functions entirely.

Tests prove, mechanically and bitwise:

* ``cp.sort`` == ``argsort`` + ``take_along_axis`` (direct-sort proof),
* the new slow path == the PRE-Phase-D slow path BITWISE across the whole
  §31 matrix and beyond (embedded verbatim legacy reference below),
* the full §31 matrix on the OPTIMIZED slow path vs the CPU authority
  (documented parity tolerance rtol=1e-3 / atol=1e-2, |rejected_pct|<=1.0,
  per-case machine-readable metrics recorded in ``PD_METRICS``),
* extreme-limit IndexError (1.0, 0.0) and zero-limit (0.0, 0.0) /
  (0.05, 0.0) witnesses on the optimized path,
* the overlap corner equals the legacy path (fallback) and matches the CPU
  when values are distinct,
* no ``argsort`` device work at all on the optimized slow path (probe
  marks: ``winsor_direct_sort`` present, ``winsor_argsort`` /
  ``winsor_rank_argsort`` / ``winsor_take_along`` absent).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import seestar.core.stack_gpu as sgp
import seestar.core.stack_methods as sm
from seestar.core.stack_gpu import (
    stack_winsorized_sigma_gpu,
)
from seestar.core.stack_methods import _stack_winsorized_sigma_iter

try:
    import cupy as cp  # noqa: F401

    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - non-GPU hosts
    cp = None
    CUPY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not CUPY_AVAILABLE, reason="CuPy not installed (no GPU stack available)"
)

import test_stack_gpu_winsorized_fastpath as tfp  # noqa: E402 (matrix source)

DEFAULT_LIMITS = (0.05, 0.05)
PD_TOL = (1e-3, 1e-2)  # rtol, atol — identical to the B6 / phase C contract

# ---------------------------------------------------------------------------
# VERBATIM pre-Phase-D reference implementations (commit 6a3a26d, probe
# calls stripped — they are pure recorders, never value-affecting).  These
# are the ACCEPTED slow-path helpers; phase D must reproduce them bitwise.
# ---------------------------------------------------------------------------


def _winsorize_axis0_legacy(cp_mod, arr, limits):
    """Exact pre-Phase-D ``_winsorize_axis0_cp`` (rank-based scatter)."""
    low, high = limits
    arr = arr.astype(cp_mod.float32, copy=False)
    result = arr.copy()

    valid = ~cp_mod.isnan(arr)
    n_valid = cp_mod.count_nonzero(valid, axis=0)

    if not bool(cp_mod.any(n_valid > 0)):
        return result

    sort_key = cp_mod.where(valid, arr, cp_mod.float32(cp_mod.inf))
    order = cp_mod.argsort(sort_key, axis=0)
    sorted_vals = cp_mod.take_along_axis(sort_key, order, axis=0)

    rank = cp_mod.argsort(order, axis=0)

    if low > 0:
        lowidx = cp_mod.clip(
            cp_mod.floor(low * n_valid).astype(cp_mod.int64), 0, None
        )
        if bool(cp_mod.any(lowidx >= arr.shape[0])):
            raise IndexError(
                f"index {int(cp_mod.max(lowidx))} is out of bounds for axis 0 "
                f"with size {arr.shape[0]}"
            )
        low_bound = cp_mod.take_along_axis(
            sorted_vals, lowidx[cp_mod.newaxis], axis=0
        )
        low_sel = valid & (rank < lowidx[cp_mod.newaxis])
        if bool(cp_mod.any(low_sel)):
            result[low_sel] = cp_mod.broadcast_to(
                low_bound, result.shape
            )[low_sel]

    if high > 0:
        highidx = cp_mod.clip(
            cp_mod.floor(high * n_valid).astype(cp_mod.int64), 0, None
        )
        upidx = cp_mod.clip(n_valid - highidx, 0, None)
        keep_idx = cp_mod.clip(upidx - 1, 0, None)
        high_bound = cp_mod.take_along_axis(
            sorted_vals, keep_idx[cp_mod.newaxis], axis=0
        )
        high_sel = valid & (rank >= upidx[cp_mod.newaxis])
        if bool(cp_mod.any(high_sel)):
            result[high_sel] = cp_mod.broadcast_to(
                high_bound, result.shape
            )[high_sel]

    return result


def _winsorize_bounds_legacy(cp_mod, arr, limits):
    """Exact pre-Phase-D ``_winsorize_bounds_cp`` (argsort + gather)."""
    low, high = limits
    valid = ~cp_mod.isnan(arr)
    n_valid = cp_mod.count_nonzero(valid, axis=0)
    sort_key = cp_mod.where(valid, arr, cp_mod.float32(cp_mod.inf))
    order = cp_mod.argsort(sort_key, axis=0)
    sorted_vals = cp_mod.take_along_axis(sort_key, order, axis=0)

    max_idx = cp_mod.maximum(n_valid - 1, 0)
    lowidx = cp_mod.clip(
        cp_mod.floor(low * n_valid).astype(cp_mod.int64), 0, max_idx
    )
    highidx = cp_mod.clip(
        n_valid - 1 - cp_mod.floor(high * n_valid).astype(cp_mod.int64),
        0,
        max_idx,
    )

    low_b = cp_mod.take_along_axis(sorted_vals, lowidx[cp_mod.newaxis], axis=0)
    high_b = cp_mod.take_along_axis(
        sorted_vals, highidx[cp_mod.newaxis], axis=0
    )
    return low_b, high_b


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _images(a):
    return [a[i] for i in range(a.shape[0])]


def _cpu(a, weights, **kw):
    return _stack_winsorized_sigma_iter(
        _images(a), weights, return_weights=True, **kw
    )


def _gpu(a, weights, **kw):
    return stack_winsorized_sigma_gpu(
        _images(a), weights, return_weights=True, **kw
    )


def _asnumpy(x):
    if isinstance(x, tuple):
        return tuple(_asnumpy(v) for v in x)
    return cp.asnumpy(x)


PD_METRICS = []  # [(tag, metrics)] machine-readable, consumed by the report


def _record(tag, cpu, gpu):
    c_res, c_w, c_pct = cpu
    g_res, g_w, g_pct = gpu
    tol = PD_TOL[1] + PD_TOL[0] * np.abs(c_res.astype(np.float64))
    diff = np.abs(g_res.astype(np.float64) - c_res.astype(np.float64))
    diff = np.where(np.isnan(diff), 0.0, diff)
    differing = int(np.sum(diff > tol))
    max_abs = float(np.max(diff)) if diff.size else 0.0
    flat_idx = int(np.argmax(diff)) if diff.size else 0
    worst = None
    if diff.size:
        coords = np.unravel_index(flat_idx, diff.shape)
        worst = {
            "coords": [int(v) for v in coords],
            "max_abs": float(diff[coords]),
            "cpu": float(c_res[coords]) if np.isfinite(c_res[coords]) else None,
            "gpu": float(g_res[coords]) if np.isfinite(g_res[coords]) else None,
        }
    metrics = {
        "tag": tag,
        "max_abs": max_abs,
        "max_rel": tfp._finite_maxrel(g_res, c_res),
        "differing": differing,
        "weight_max_abs": tfp._finite_maxabs(g_w, c_w),
        "pct_cpu": float(c_pct),
        "pct_gpu": float(g_pct),
        "worst": worst,
    }
    PD_METRICS.append((tag, metrics))
    return metrics


def assert_cpu_parity(tag, cpu, gpu, strict=True):
    """Documented-tolerance parity + return contract (identical to B6)."""
    c_res, c_w, c_pct = cpu
    g_res, g_w, g_pct = gpu
    assert isinstance(g_res, np.ndarray) and g_res.dtype == np.float32
    assert isinstance(g_w, np.ndarray) and g_w.dtype == np.float32
    assert isinstance(g_pct, float)
    assert not isinstance(g_res, cp.ndarray)
    assert not isinstance(g_w, cp.ndarray)
    assert g_res.shape == c_res.shape
    if strict:
        np.testing.assert_allclose(
            g_res, c_res, rtol=PD_TOL[0], atol=PD_TOL[1], equal_nan=True
        )
        np.testing.assert_allclose(
            g_w, c_w, rtol=PD_TOL[0], atol=PD_TOL[1], equal_nan=True
        )
        assert abs(float(g_pct) - float(c_pct)) <= 1.0, (g_pct, c_pct, tag)
    return _record(tag, cpu, gpu)


def _force_slow(monkeypatch):
    """Route EVERYTHING through the (optimized) slow path."""
    monkeypatch.setattr(sgp, "_winsor_zero_rank_regime", lambda *a: False)


def _force_legacy(monkeypatch):
    """Route every winsor sort helper through the pre-Phase-D code."""
    monkeypatch.setattr(
        sgp, "_winsorize_axis0_cp", _winsorize_axis0_legacy
    )
    monkeypatch.setattr(
        sgp, "_winsorize_bounds_cp", _winsorize_bounds_legacy
    )


# ---------------------------------------------------------------------------
# adversarial input families (beyond the §31 matrix builders)
# ---------------------------------------------------------------------------

LEGACY_LIMITS = [
    (0.05, 0.05),
    (0.0, 0.0),
    (0.05, 0.0),
    (0.0, 0.05),
    (0.03, 0.05),
    (0.2, 0.2),
    (0.25, 0.25),
    (0.3, 0.3),
    (0.1, 0.4),
    (0.4, 0.1),
    (0.5, 0.5),  # boundary: floor sums == n, no overlap, ties straddle both
    (0.0, 1.0),  # degenerate single-side: everything -> min (no overlap)
    (1.0, 0.0),  # only with per-column NaN so no IndexError (see below)
]


def _tie_at_boundary_columns():
    """Columns crafted so tie groups straddle the low AND high order
    statistics: [1,1,1,2,2] with (0.4, 0.4) -> lowidx 2 (tie at 1.0),
    upidx 3 / keep 2 (tie at 1.0)."""
    n = 5
    col = np.array([1.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    a = np.broadcast_to(col[:, None, None], (n, 4, 5)).copy()
    return a


def _overlap_tie_columns():
    """Overlap corner WITH a boundary tie: [1,1,2,10] under (0.75, 0.75) ->
    lowidx 3, upidx 1: the sequential rank replacement splits the two 1.0s
    (one gets low_bound 10, the other high_bound 1) — no value comparison
    can express that, so the fallback must engage and match the legacy
    path."""
    col = np.array([1.0, 1.0, 2.0, 10.0], dtype=np.float32)
    a = np.broadcast_to(col[:, None, None], (4, 3, 4)).copy()
    return a


def _overlap_distinct_columns():
    """Overlap corner with DISTINCT values (deterministic rank semantics,
    comparable to the CPU): [1,2,3,10] under (0.75, 0.75)."""
    col = np.array([1.0, 2.0, 3.0, 10.0], dtype=np.float32)
    a = np.broadcast_to(col[:, None, None], (4, 3, 4)).copy()
    return a


def _signed_zero_stack():
    n = 8
    rng = np.random.default_rng(300)
    a = rng.normal(1000.0, 50.0, size=(n, 6, 7)).astype(np.float32)
    a[0] = -0.0
    a[1] = 0.0
    a[2, :, :] = -0.0
    a[3, 0, 0] = -0.0
    a[4, 0, 0] = 0.0
    return a


def _signed_zero_boundary_stack():
    """A column of pure zeros with one negative zero + several valid bigger
    samples: the (0.5, 0.5) boundary lands inside the zero tie group."""
    n = 6
    a = np.full((n, 5, 6), np.nan, dtype=np.float32)
    z = np.zeros(n, dtype=np.float32)
    z[0] = -0.0
    a[:, 0, 0] = z
    a[:, 1, 0] = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    return a


def _extreme_single_side_full_to_inf():
    """(1.0, 0.0) with NaN in every spatial column: no IndexError
    (n_valid < N everywhere); the low bound degenerates to the +inf sentinel
    -> every valid sample of those columns is replaced by +inf (rank rule)
    == value-clip to +inf."""
    n, h, w = 8, 6, 7
    a = np.arange(n * h * w, dtype=np.float32).reshape(n, h, w) + 100.0
    for r in range(h):
        for c in range(w):
            a[(r * w + c) % n, r, c] = np.nan  # every spatial column keeps
    return a


# ---------------------------------------------------------------------------
# 1. direct-sort proof: cp.sort == argsort + take_along_axis (bitwise)
# ---------------------------------------------------------------------------


def _sort_key_of(a):
    valid = ~np.isnan(a)
    return np.where(valid, a, np.inf).astype(np.float32)


SORT_PROOF_ARRAYS = [
    ("random", lambda: np.random.default_rng(1).normal(0, 1, (20, 9, 11)).astype(np.float32)),
    ("ties", lambda: np.round(np.random.default_rng(2).normal(0, 1, (20, 9, 11)).astype(np.float32) * 2.0)),
    ("equal_all", lambda: np.full((7, 4, 5), 3.5, dtype=np.float32)),
    ("zeros_mixed_sign", lambda: _signed_zero_stack()),
    ("pos_neg_inf", lambda: np.array(
        [[[np.inf], [-np.inf], [1.0], [-1.0], [0.0]]], dtype=np.float32).repeat(6, axis=1)),
    ("single_row", lambda: np.array([[5.0, -2.0, 5.0, 0.0]], dtype=np.float32)),
    ("all_inf", lambda: np.full((5, 2, 3), np.inf, dtype=np.float32)),
    ("nan_sentinel", lambda: _sort_key_of(np.where(
        np.random.default_rng(3).random((12, 4, 6)) < 0.3, np.nan,
        np.random.default_rng(4).normal(0, 1, (12, 4, 6))).astype(np.float32))),
]


@pytest.mark.parametrize(
    "tag", [t[0] for t in SORT_PROOF_ARRAYS], ids=[t[0] for t in SORT_PROOF_ARRAYS]
)
def test_direct_sort_bitwise_equals_argsort_gather(tag):
    a = next(t for t in SORT_PROOF_ARRAYS if t[0] == tag)[1]()
    key = cp.asarray(a)
    direct = cp.sort(key, axis=0)
    order = cp.argsort(key, axis=0)
    gathered = cp.take_along_axis(key, order, axis=0)
    assert np.array_equal(
        cp.asnumpy(direct), cp.asnumpy(gathered)
    ), f"cp.sort != argsort+take_along on {tag}"


# ---------------------------------------------------------------------------
# 2. helper level: value-clip / direct-sort == pre-Phase-D path (bitwise)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lim", LEGACY_LIMITS, ids=["%s_%s" % (a, b) for a, b in LEGACY_LIMITS]
)
def test_axis0_valueclip_bitwise_equals_legacy(lim):
    """Optimized helper vs the verbatim pre-Phase-D helper, bitwise, on the
    phase C §31 matrix builders (all validity/ties/inf/limit regimes)."""
    for tag, builder, _expected_fast in tfp.FP_MATRIX:
        a, w, kw = builder()
        limits = kw.get("winsor_limits", DEFAULT_LIMITS)
        # the helper only depends on the stack + limits actually used
        if limits != lim:
            continue
        arr = cp.asarray(a)
        new = _asnumpy(sgp._winsorize_axis0_cp(cp, arr, lim))
        old = _asnumpy(_winsorize_axis0_legacy(cp, arr, lim))
        assert np.array_equal(new, old, equal_nan=True), (lim, tag)


@pytest.mark.parametrize(
    "lim",
    [(0.05, 0.05), (0.2, 0.2), (0.5, 0.5), (0.0, 0.0), (0.0, 1.0), (1.0, 0.0)],
    ids=["%s_%s" % (a, b) for a, b in [(0.05, 0.05), (0.2, 0.2), (0.5, 0.5), (0.0, 0.0), (0.0, 1.0), (1.0, 0.0)]],
)
def test_axis0_crafted_tie_cases_bitwise_equals_legacy(lim):
    """Crafted tie-at-boundary / degenerate / signed-zero stacks, bitwise."""
    stacks = [
        _tie_at_boundary_columns(),
        _signed_zero_stack(),
        _signed_zero_boundary_stack(),
        _extreme_single_side_full_to_inf(),
    ]
    if lim[0] == 1.0:
        # (1.0, 0.0): every crafted stack must keep a NaN per column
        stacks = [_extreme_single_side_full_to_inf()]
    for a in stacks:
        arr = cp.asarray(a)
        new = _asnumpy(sgp._winsorize_axis0_cp(cp, arr, lim))
        old = _asnumpy(_winsorize_axis0_legacy(cp, arr, lim))
        assert np.array_equal(new, old, equal_nan=True), lim


def test_axis0_crafted_tie_values_match_expected_hand_computation():
    """[1,1,1,2,2] under (0.4, 0.4): lowidx 2 / upidx 3, both bounds are the
    order statistic at index 2 == 1.0.  The two 2.0s (rank >= 3) are replaced
    by high_bound 1.0, the three 1.0s keep their value (boundary-tie
    replacement is a same-bit no-op) -> the column becomes all 1.0s."""
    a = _tie_at_boundary_columns()
    out = cp.asnumpy(sgp._winsorize_axis0_cp(cp, cp.asarray(a), (0.4, 0.4)))
    np.testing.assert_array_equal(out, np.ones_like(a))


def test_axis0_overlap_tie_falls_back_and_equals_legacy():
    """[1,1,2,10] under (0.75, 0.75): overlap corner (lowidx 3 > upidx 1).
    The fallback engages; result equals the legacy rank path bitwise (and is
    sort-tie-order dependent between the two 1.0s, exactly like the CPU)."""
    a = _overlap_tie_columns()
    arr = cp.asarray(a)
    new = _asnumpy(sgp._winsorize_axis0_cp(cp, arr, (0.75, 0.75)))
    old = _asnumpy(_winsorize_axis0_legacy(cp, arr, (0.75, 0.75)))
    assert np.array_equal(new, old, equal_nan=True)
    # the CPU produces the same per-column MULTISET of values regardless of
    # its own sort tie order (one 1.0 -> 10, the other 1.0 -> 1, 2 -> 1,
    # 10 -> 1); compare sorted columns so tie order cannot mask agreement.
    cpu = _cpu_winsor_axis0_np(a, (0.75, 0.75))
    np.testing.assert_array_equal(np.sort(new, axis=0), np.sort(cpu, axis=0))
    # every column must contain exactly one 10 and three 1s
    np.testing.assert_array_equal(np.sort(new[:, 0, 0]), [1.0, 1.0, 1.0, 10.0])


def _cpu_winsor_axis0_np(a, limits):
    """Thin direct call into the CPU authority helper."""
    return sm._winsorize_axis0_numpy(a.copy(), limits)


def test_axis0_overlap_distinct_matches_cpu():
    """[1,2,3,10] under (0.75, 0.75) with distinct values: the overlap
    outcome is deterministic and must match the CPU authority bitwise."""
    a = _overlap_distinct_columns()
    gpu = cp.asnumpy(sgp._winsorize_axis0_cp(cp, cp.asarray(a), (0.75, 0.75)))
    cpu = _cpu_winsor_axis0_np(a, (0.75, 0.75))
    assert np.array_equal(gpu, cpu, equal_nan=True)
    # explicit hand-check: lowidx 3, upidx 1 -> rank<1 -> low_bound 10;
    # rank>=1 -> high_bound 1  =>  [10, 1, 1, 1] per column
    np.testing.assert_array_equal(gpu[:, 0, 0], [10.0, 1.0, 1.0, 1.0])


def test_bounds_directsort_bitwise_equals_legacy():
    for tag, builder, _expected_fast in tfp.FP_MATRIX:
        a, w, kw = builder()
        limits = kw.get("winsor_limits", DEFAULT_LIMITS)
        arr = cp.asarray(a)
        new = _asnumpy(sgp._winsorize_bounds_cp(cp, arr, limits))
        old = _asnumpy(_winsorize_bounds_legacy(cp, arr, limits))
        for nv, ov in zip(new, old):
            assert np.array_equal(nv, ov, equal_nan=True), (tag, limits)


def test_axis0_nan_map_and_dtype_contract():
    a = _tie_at_boundary_columns()
    a[0, 1, 1] = np.nan
    arr = cp.asarray(a)
    out = sgp._winsorize_axis0_cp(cp, arr, (0.05, 0.05))
    assert isinstance(out, cp.ndarray)
    out_np = cp.asnumpy(out)
    np.testing.assert_array_equal(np.isnan(out_np), np.isnan(a))
    assert out_np.dtype == np.float32


# ---------------------------------------------------------------------------
# 3. stack level: optimized slow path == pre-Phase-D slow path (bitwise)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [20, 32, 50])
def test_stack_slowpath_new_bitwise_equals_legacy(n, monkeypatch):
    """The full reduction through the OPTIMIZED slow path must be bitwise
    identical (result, weight map, rejected_pct) to the same reduction
    through the pre-Phase-D helpers, at N_batch 20 / 32 / 50."""
    a = tfp._make_stack(n, (24, 32), seed=400 + n, nan_frac=0.05, spike=0.05)
    w = tfp._weights(n, seed=30)
    _force_legacy(monkeypatch)
    legacy = _gpu(a, w)
    monkeypatch.undo()
    new = _gpu(a, w)
    assert np.array_equal(new[0], legacy[0], equal_nan=True), n
    assert np.array_equal(new[1], legacy[1], equal_nan=True), n
    assert new[2] == legacy[2], n
    # and the optimized slow path stays CPU-exact within the tolerance
    assert_cpu_parity("stack_new_n%d" % n, _cpu(a, w), new)


@pytest.mark.parametrize(
    "tag", [e[0] for e in tfp.FP_MATRIX], ids=[e[0] for e in tfp.FP_MATRIX]
)
def test_full_matrix_optimized_slowpath_vs_cpu_and_legacy(tag, monkeypatch):
    """The FULL §31 matrix, every case forced through the optimized slow
    path: CPU parity within the documented tolerance AND bitwise identity
    with the pre-Phase-D slow path."""
    entry = next(e for e in tfp.FP_MATRIX if e[0] == tag)
    _, builder, _expected_fast = entry
    a, w, kw = builder()
    cpu = _cpu(a, w, **kw)
    _force_legacy(monkeypatch)
    legacy = _gpu(a, w, **kw)
    monkeypatch.undo()
    _force_slow(monkeypatch)
    new_slow = _gpu(a, w, **kw)
    assert np.array_equal(
        new_slow[0], legacy[0], equal_nan=True
    ), "new slow != legacy slow: %s" % tag
    assert np.array_equal(
        new_slow[1], legacy[1], equal_nan=True
    ), "new slow != legacy slow (weights): %s" % tag
    assert new_slow[2] == legacy[2], tag
    assert_cpu_parity("slow_%s" % tag, cpu, new_slow)


# ---------------------------------------------------------------------------
# 4. CPU exception + degenerate-limit witnesses on the optimized path
# ---------------------------------------------------------------------------


def test_extreme_limit_1_0_all_valid_column_raises_indexerror():
    """(1.0, 0.0) with an all-valid column: lowidx == n_valid == N -> the
    explicit out-of-bounds check must raise IndexError (mirroring the CPU),
    NOT silently clip lowidx to max_idx."""
    rng = np.random.default_rng(51)
    a = rng.normal(10.0, 2.0, size=(8, 5, 5)).astype(np.float32)
    imgs = [a[i] for i in range(8)]
    w = tfp._weights(8, seed=13)
    with pytest.raises(IndexError):
        _stack_winsorized_sigma_iter(
            imgs, w, winsor_limits=(1.0, 0.0), return_weights=True
        )
    with pytest.raises(IndexError):
        stack_winsorized_sigma_gpu(
            imgs, w, winsor_limits=(1.0, 0.0), return_weights=True
        )


def test_extreme_limit_1_0_nan_in_every_column_stays_finite():
    """(1.0, 0.0) with a NaN in EVERY column: n_valid < N -> no IndexError;
    the optimized slow path must reproduce the CPU output (valid samples of
    fully... partially-valid columns go through the degenerate +inf bound)."""
    a = _extreme_single_side_full_to_inf()
    w = tfp._weights(a.shape[0], seed=14)
    cpu = _cpu(a, w, winsor_limits=(1.0, 0.0))
    gpu = _gpu(a, w, winsor_limits=(1.0, 0.0))
    assert_cpu_parity("lim_1_0_nan_cols", cpu, gpu)


@pytest.mark.parametrize(
    "limits",
    [(0.0, 0.0), (0.05, 0.0), (0.0, 0.05), (0.0, 1.0)],
    ids=["0_0", "05_0", "0_05", "0_1"],
)
def test_zero_and_single_side_limits_optimized_slowpath(limits, monkeypatch):
    """Zero-limit / single-side cases forced through the optimized slow path:
    CPU parity + bitwise identity with the legacy path."""
    a = tfp._make_stack(20, (20, 24), seed=500, nan_frac=0.04, spike=0.06)
    w = tfp._weights(20, seed=15)
    kw = dict(winsor_limits=limits)
    cpu = _cpu(a, w, **kw)
    _force_legacy(monkeypatch)
    legacy = _gpu(a, w, **kw)
    monkeypatch.undo()
    _force_slow(monkeypatch)
    new_slow = _gpu(a, w, **kw)
    assert np.array_equal(new_slow[0], legacy[0], equal_nan=True), limits
    assert np.array_equal(new_slow[1], legacy[1], equal_nan=True), limits
    assert new_slow[2] == legacy[2], limits
    assert_cpu_parity("zero_lim_%s_%s" % limits, cpu, new_slow)


def test_1_0_single_side_partial_helper_matches_cpu():
    """Helper level: (1.0, 0.0) with partial columns -> low bound +inf,
    every valid sample replaced by +inf, exactly like the CPU."""
    a = _extreme_single_side_full_to_inf()
    gpu = cp.asnumpy(
        sgp._winsorize_axis0_cp(cp, cp.asarray(a), (1.0, 0.0))
    )
    cpu = _cpu_winsor_axis0_np(a, (1.0, 0.0))
    np.testing.assert_array_equal(np.isnan(gpu), np.isnan(cpu))
    np.testing.assert_allclose(gpu, cpu, rtol=0.0, atol=0.0, equal_nan=True)


# ---------------------------------------------------------------------------
# 5. mechanical: no argsort / no int64 order+rank on the optimized path
# ---------------------------------------------------------------------------


def test_slowpath_probe_marks_show_direct_sort_no_argsort(monkeypatch):
    """N_batch=20 (default limits, slow regime): the probe marks of the run
    must show ``winsor_direct_sort`` (cp.sort) and NO ``winsor_argsort`` /
    ``winsor_rank_argsort`` / ``winsor_take_along`` — the int64 order/rank
    machinery is gone — while ``_winsorize_bounds_cp`` also emits
    ``bounds_direct_sort`` with no ``bounds_argsort``."""
    a = tfp._make_stack(20, (16, 20), seed=600, nan_frac=0.04, spike=0.05)
    w = tfp._weights(20, seed=16)
    assert sgp._winsor_zero_rank_regime(DEFAULT_LIMITS, 20) is False
    os.environ["ZSSS_GPU_PROFILE"] = "1"
    try:
        sgp._PROBE = None
        _gpu(a, w)
        marks = [name for (name, _t, ev) in sgp._PROBE.marks if ev is not None]
    finally:
        sgp._PROBE = None
        os.environ.pop("ZSSS_GPU_PROFILE", None)
    assert any(m == "winsor_direct_sort" for m in marks), marks
    assert any(m == "bounds_direct_sort" for m in marks), marks
    for forbidden in (
        "winsor_argsort",
        "winsor_rank_argsort",
        "winsor_take_along",
        "bounds_argsort",
        "bounds_take_along",
    ):
        assert not any(m == forbidden for m in marks), (forbidden, marks)


def test_overlap_input_probe_note_fallback(monkeypatch):
    """An overlap input emits the fallback note and still matches the CPU
    (deterministic distinct-value case)."""
    a = _overlap_distinct_columns()
    cpu = _cpu_winsor_axis0_np(a, (0.75, 0.75))
    os.environ["ZSSS_GPU_PROFILE"] = "1"
    try:
        sgp._PROBE = None
        sgp._ensure_probe()
        out = cp.asnumpy(
            sgp._winsorize_axis0_cp(cp, cp.asarray(a), (0.75, 0.75))
        )
        notes = list(sgp._PROBE.notes)
    finally:
        sgp._PROBE = None
        os.environ.pop("ZSSS_GPU_PROFILE", None)
    assert any("winsorize_overlap_rank_fallback" in x for x in notes), notes
    assert np.array_equal(out, cpu, equal_nan=True)
