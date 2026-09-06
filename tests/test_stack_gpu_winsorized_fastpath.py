"""Phase C (Track P1) tests: exact small-N zero-rank fast path of the CuPy
Winsorized Sigma Clip twin.

CPU reference (``seestar/core/stack_methods.py::_stack_winsorized_sigma_iter``
-> ``_winsorize_axis0_numpy`` / ``_winsorize_bounds``) stays the SCIENTIFIC
AUTHORITY and is never modified.

Regime proof under test
-----------------------
``N_batch`` is the ACTUAL current reduction population (this call's frame
count; the engine flushes the final partial batch as-is, so the twin never
sees a frozen ``B_resolved``).  Every per-pixel valid count satisfies
``n_valid <= N_batch`` in every iteration and in the survivor rewinsor pass.
``floor(limit * n)`` is non-decreasing in ``n``, therefore

    floor(low * N_batch) == 0 AND floor(high * N_batch) == 0

implies the winsor rank is zero on EVERY pixel of EVERY iteration:
winsorization is the identity (nothing is ever replaced), so the
per-iteration order-statistics work of ``_winsorize_axis0_cp`` (argsort +
``take_along_axis`` + inverse-rank argsort + replacement) is skipped and the
location/scale statistics are computed directly on the masked stack; the
``apply_rewinsor`` survivor bounds degenerate to the survivor min/max (the
reference order statistics at indices 0 and ``n_valid - 1``), computed without
any sort.  Default ``winsor_limits=(0.05, 0.05)`` -> N_batch <= 19.

The tests prove the branch mechanically (spy counters on the sort helpers —
not timing), prove the fast path is bit-identical to the slow path in the
regime, prove boundary behaviour (N_batch = 19 fast vs N_batch = 20 slow),
and re-run the full §31 qualification matrix (all finite, NaNs, all invalid,
one/two valid, varying Nvalid, ties, repeated values, strong positive
outliers, negative values, +Inf/-Inf where the CPU accepts them, unequal
weights, zero/nonzero-rank boundaries, asymmetric limits, apply_rewinsor
true/false, early exit, full max_iters, kappa decay) against the CPU with the
documented parity tolerance (rtol=1e-3, atol=1e-2, |rejected_pct| <= 1.0),
recording per-case machine-readable metrics into ``FP_METRICS``.
"""

from __future__ import annotations

import numpy as np
import pytest

import seestar.core.stack_gpu as sgp
import seestar.core.stack_methods as sm
from seestar.core.stack_gpu import (
    _winsor_zero_rank_regime,
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

DEFAULT_LIMITS = (0.05, 0.05)
FP_TOL = (1e-3, 1e-2)  # rtol, atol — identical to the B6 contract


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _weights(n, seed=7):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.4, 1.6, size=n).astype(np.float32)


def _make_stack(n, shape, seed=1, nan_frac=0.03, spike=0.03, loc=1000.0):
    """Deterministic synthetic stack (float32): sky + noise + spikes + NaN."""
    rng = np.random.default_rng(seed)
    out_shape = (n,) + shape
    arr = rng.normal(loc, 20.0, size=out_shape).astype(np.float32)
    if nan_frac > 0:
        arr[rng.random(out_shape) < nan_frac] = np.nan
    arr = arr + np.where(
        rng.random(out_shape) < spike, 400.0, 0.0
    ).astype(np.float32)
    return arr


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


def _finite_maxabs(a, b):
    both = np.isfinite(a) & np.isfinite(b)
    if not np.any(both):
        return 0.0
    return float(
        np.max(np.abs(a[both].astype(np.float64) - b[both].astype(np.float64)))
    )


def _finite_maxrel(a, b):
    both = np.isfinite(a) & np.isfinite(b)
    if not np.any(both):
        return 0.0
    denom = np.maximum(np.abs(b[both].astype(np.float64)), 1e-12)
    return float(
        np.max(
            np.abs(a[both].astype(np.float64) - b[both].astype(np.float64))
            / denom
        )
    )


# ---------------------------------------------------------------------------
# §31 qualification matrix (single source of truth for the tests AND the
# phase C profiling driver, which reproduces the cases for the report).
# ---------------------------------------------------------------------------

_H, _W = 24, 32


def _b_all_finite():
    a = _make_stack(12, (_H, _W), seed=101, nan_frac=0.0)
    return a, None, {}


def _b_nan_scattered():
    a = _make_stack(12, (_H, _W), seed=102, nan_frac=0.08)
    return a, _weights(12, seed=1), {}


def _b_all_invalid_slices():
    n = 10
    a = _make_stack(n, (_H, _W), seed=103, nan_frac=0.0)
    a[0, :, :] = np.nan                 # fully-invalid frame
    a[:, 0, 0] = np.nan                 # fully-invalid column
    a[:, :, 3] = np.nan                 # fully-invalid spatial column
    return a, _weights(n, seed=2), {}


def _b_one_valid_per_column():
    n = 12
    a = _make_stack(n, (_H, _W), seed=104, nan_frac=0.0)
    for j in range(_W):
        keep = j % n
        for i in range(n):
            if i != keep:
                a[i, :, j] = np.nan
    return a, None, {}


def _b_two_valid_per_column():
    n = 12
    a = _make_stack(n, (_H, _W), seed=105, nan_frac=0.0)
    for j in range(_W):
        keep = {(2 * j) % n, (2 * j + 1) % n}
        for i in range(n):
            if i not in keep:
                a[i, :, j] = np.nan
    return a, _weights(n, seed=3), {}


def _b_varying_nvalid_per_pixel():
    """Half the columns keep exactly 3 valid samples, half stay full:
    per-pixel valid counts genuinely vary inside one reduction."""
    n = 12
    a = _make_stack(n, (_H, _W), seed=106, nan_frac=0.0)
    for j in range(_W):
        if j % 2 == 0:
            keep = {j % n, (j + 1) % n, (j + 2) % n}
            for i in range(n):
                if i not in keep:
                    a[i, :, j] = np.nan
    a[5, 5, 5] = np.nan  # scattered NaN too
    return a, _weights(n, seed=4), {}


def _b_ties():
    a = _make_stack(12, (_H, _W), seed=107, nan_frac=0.03)
    a = np.round(a / 8.0) * 8.0
    return a, None, {}


def _b_repeated_values():
    n = 12
    base = _make_stack(4, (_H, _W), seed=108, nan_frac=0.05)
    a = np.concatenate([base] * 3, axis=0)[:n]
    return a, _weights(n, seed=5), {}


def _b_strong_positive_outliers():
    a = _make_stack(12, (_H, _W), seed=109, nan_frac=0.02, spike=0.0)
    a[3] += 5000.0                      # whole-frame outlier (e.g. cloud)
    a[7, 2, 2] = 1e6                    # single hot pixel
    a[8, 4, 4] = 1e7
    return a, None, {}


def _b_negative_values():
    a = _make_stack(12, (_H, _W), seed=110, nan_frac=0.03, loc=-1000.0)
    a[7, 6, 6] = -1e6                   # low outlier
    a[9, 8, 8] = -5e5
    return a, _weights(12, seed=6), {}


def _b_pos_inf_samples():
    a = _make_stack(12, (_H, _W), seed=111, nan_frac=0.03)
    a[1, :, 5] = np.inf
    a[2, 2, 6] = np.inf
    return a, None, {}


def _b_neg_inf_samples():
    a = _make_stack(12, (_H, _W), seed=112, nan_frac=0.03)
    a[3, :, 7] = -np.inf
    a[4, 4, 8] = -np.inf
    return a, _weights(12, seed=7), {}


def _b_pos_neg_inf_mix():
    a = _make_stack(12, (_H, _W), seed=113, nan_frac=0.03)
    a[1, :, 5] = np.inf
    a[3, :, 7] = -np.inf
    a[2, 2, 6] = np.inf
    a[4, 4, 8] = -np.inf
    return a, _weights(12, seed=8), {}


def _b_unequal_weights():
    a = _make_stack(12, (_H, _W), seed=114, nan_frac=0.05)
    w = np.geomspace(1e-3, 1e3, 12).astype(np.float32)
    return a, w, {}


def _b_zero_rank_boundary_n19():
    a = _make_stack(19, (_H, _W), seed=115, nan_frac=0.04, spike=0.05)
    return a, _weights(19, seed=9), {}


def _b_nonzero_rank_boundary_n20():
    a = _make_stack(20, (_H, _W), seed=116, nan_frac=0.04, spike=0.05)
    return a, _weights(20, seed=10), {}


def _b_asymmetric_low_only():
    a = _make_stack(12, (_H, _W), seed=117, nan_frac=0.03, spike=0.06)
    return a, _weights(12, seed=11), dict(winsor_limits=(0.05, 0.0))


def _b_asymmetric_high_only():
    a = _make_stack(12, (_H, _W), seed=118, nan_frac=0.03, spike=0.06)
    return a, _weights(12, seed=12), dict(winsor_limits=(0.0, 0.05))


def _b_asymmetric_uneven():
    a = _make_stack(19, (_H, _W), seed=119, nan_frac=0.03, spike=0.06)
    return a, _weights(19, seed=13), dict(winsor_limits=(0.03, 0.05))


def _b_zero_limits():
    a = _make_stack(20, (_H, _W), seed=120, nan_frac=0.03, spike=0.06)
    return a, _weights(20, seed=14), dict(winsor_limits=(0.0, 0.0))


def _b_extreme_small_n_rank():
    # n=6, (0.3, 0.3): floor(0.3*6) = 1 -> rank > 0 at SMALL N (slow path).
    a = _make_stack(6, (_H, _W), seed=121, nan_frac=0.03, spike=0.06)
    return a, _weights(6, seed=15), dict(winsor_limits=(0.3, 0.3))


def _b_extreme_small_n_rank_zero():
    # n=4, (0.2, 0.2): floor(0.2*4) = 0 -> still the zero-rank fast path.
    a = _make_stack(4, (_H, _W), seed=122, nan_frac=0.02, spike=0.08)
    return a, None, dict(winsor_limits=(0.2, 0.2))


def _b_rewinsor_false():
    a = _make_stack(12, (_H, _W), seed=123, nan_frac=0.03, spike=0.07)
    return a, _weights(12, seed=16), dict(apply_rewinsor=False)


def _b_early_exit():
    # Inlier-only stack: n_rej == 0 at iteration 0 -> immediate early exit.
    a = _make_stack(12, (_H, _W), seed=124, nan_frac=0.02, spike=0.0)
    a += np.linspace(-1.0, 1.0, 12).reshape(12, 1, 1).astype(np.float32)
    return a, None, {}


def _b_full_max_iters_kappa_decay_fast():
    # n=19 (fast regime), drifting + decaying kappa: full 5 iterations run.
    n = 19
    a = _make_stack(n, (16, 16), seed=125, nan_frac=0.02, spike=0.05)
    a = a + np.linspace(0.0, 60.0, n).reshape(n, 1, 1).astype(np.float32)
    return a, _weights(n, seed=17), dict(
        kappa=2.0, max_iters=5, kappa_decay=0.6
    )


def _b_full_max_iters_kappa_decay_slow():
    # n=30 (slow regime), drifting + decaying kappa: full 5 iterations run.
    n = 30
    a = _make_stack(n, (16, 16), seed=126, nan_frac=0.02, spike=0.05)
    a = a + np.linspace(0.0, 60.0, n).reshape(n, 1, 1).astype(np.float32)
    return a, _weights(n, seed=18), dict(
        kappa=2.0, max_iters=5, kappa_decay=0.6
    )


def _b_catastrophic_column_rejection():
    """One column loses ALL its valid samples in iteration 0 (tiny kappa):
    the rewinsor pass runs over an EMPTY survivor column and must reproduce
    the CPU's degenerate +inf sentinel bounds bit-for-bit."""
    n = 12
    a = _make_stack(n, (_H, _W), seed=127, nan_frac=0.0)
    a[0:3, :, 0] = np.array([-1000.0, -999.0, 1000.0])[:, None]
    a[3:, :, 0] = np.nan
    a[5, 3, 3] = 5000.0
    a[6, 4, 4] = -4000.0
    return a, _weights(n, seed=19), dict(kappa=0.5, max_iters=6)


def _b_partial_final_batch():
    """Engine-level final partial batch: B_resolved would be 20 but only 12
    frames are actually flushed to this reduction (the twin never sees
    B_resolved — N_batch is the actual population of the call)."""
    a = _make_stack(12, (_H, _W), seed=128, nan_frac=0.04, spike=0.05)
    return a, _weights(12, seed=20), {}


def _b_rgb():
    rng = np.random.default_rng(129)
    a = rng.normal(1000.0, 20.0, size=(12, 8, 8, 3)).astype(np.float32)
    a[rng.random((12, 8, 8, 3)) < 0.03] = np.nan
    a = a + np.where(
        rng.random((12, 8, 8, 3)) < 0.03, 400.0, 0.0
    ).astype(np.float32)
    return a, None, {}


# (tag, builder, expected_fast_regime or None = do not assert)
FP_MATRIX = [
    ("all_finite", _b_all_finite, True),
    ("nan_scattered", _b_nan_scattered, True),
    ("all_invalid_slices", _b_all_invalid_slices, True),
    ("one_valid_per_column", _b_one_valid_per_column, True),
    ("two_valid_per_column", _b_two_valid_per_column, True),
    ("varying_nvalid_per_pixel", _b_varying_nvalid_per_pixel, True),
    ("ties_duplicates", _b_ties, True),
    ("repeated_values", _b_repeated_values, True),
    ("strong_positive_outliers", _b_strong_positive_outliers, True),
    ("negative_values", _b_negative_values, True),
    ("pos_inf_samples", _b_pos_inf_samples, True),
    ("neg_inf_samples", _b_neg_inf_samples, True),
    ("pos_neg_inf_mix", _b_pos_neg_inf_mix, True),
    ("unequal_weights", _b_unequal_weights, True),
    ("zero_rank_boundary_n19", _b_zero_rank_boundary_n19, True),
    ("nonzero_rank_boundary_n20", _b_nonzero_rank_boundary_n20, False),
    ("asymmetric_low_only", _b_asymmetric_low_only, True),
    ("asymmetric_high_only", _b_asymmetric_high_only, True),
    ("asymmetric_uneven_n19", _b_asymmetric_uneven, True),
    ("zero_limits_n20", _b_zero_limits, True),
    ("extreme_small_n_rank_gt0", _b_extreme_small_n_rank, False),
    ("extreme_small_n_rank_zero", _b_extreme_small_n_rank_zero, True),
    ("rewinsor_false", _b_rewinsor_false, True),
    ("early_exit", _b_early_exit, True),
    ("full_max_iters_decay_fast_n19", _b_full_max_iters_kappa_decay_fast, True),
    ("full_max_iters_decay_slow_n30", _b_full_max_iters_kappa_decay_slow, False),
    ("catastrophic_column_rejection", _b_catastrophic_column_rejection, True),
    ("partial_final_batch_12_of_20", _b_partial_final_batch, True),
    ("rgb", _b_rgb, True),
]

FP_METRICS = []  # [(tag, metrics)] machine-readable, consumed by the report


def _expected_regime(a, kw):
    limits = kw.get("winsor_limits", DEFAULT_LIMITS)
    return _winsor_zero_rank_regime(limits, int(a.shape[0]))


def assert_parity(tag, cpu, gpu, strict=True):
    """Documented-tolerance parity + return contract (identical to B6)."""
    c_res, c_w, c_pct = cpu
    g_res, g_w, g_pct = gpu
    assert isinstance(g_res, np.ndarray) and g_res.dtype == np.float32
    assert isinstance(g_w, np.ndarray) and g_w.dtype == np.float32
    assert isinstance(g_pct, float)
    assert not isinstance(g_res, cp.ndarray)
    assert not isinstance(g_w, cp.ndarray)
    if strict:
        np.testing.assert_allclose(
            g_res, c_res, rtol=FP_TOL[0], atol=FP_TOL[1], equal_nan=True
        )
        np.testing.assert_allclose(
            g_w, c_w, rtol=FP_TOL[0], atol=FP_TOL[1], equal_nan=True
        )
        assert abs(float(g_pct) - float(c_pct)) <= 1.0, (g_pct, c_pct, tag)
        assert g_res.shape == c_res.shape
    return _record(tag, cpu, gpu)


def _record(tag, cpu, gpu):
    """Record machine-readable parity metrics (changed-pixel count vs the
    documented combined tolerance, max abs/rel error, worst specimen)."""
    c_res, c_w, c_pct = cpu
    g_res, g_w, g_pct = gpu
    tol = FP_TOL[1] + FP_TOL[0] * np.abs(c_res.astype(np.float64))
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
        "max_rel": _finite_maxrel(g_res, c_res),
        "differing": differing,
        "weight_max_abs": _finite_maxabs(g_w, c_w),
        "pct_cpu": float(c_pct),
        "pct_gpu": float(g_pct),
        "worst": worst,
    }
    FP_METRICS.append((tag, metrics))
    return metrics


def run_matrix_case(tag):
    """Run one §31 matrix case against CPU + GPU; returns (cpu, gpu, kw, a)."""
    entry = next(e for e in FP_MATRIX if e[0] == tag)
    _, builder, _ = entry
    a, w, kw = builder()
    cpu = _cpu(a, w, **kw)
    gpu = _gpu(a, w, **kw)
    return cpu, gpu, a, w, kw


# ---------------------------------------------------------------------------
# 1. regime predicate boundaries (pure host arithmetic)
# ---------------------------------------------------------------------------


def test_regime_predicate_default_limit_boundary_is_19():
    assert _winsor_zero_rank_regime((0.05, 0.05), 19) is True
    assert _winsor_zero_rank_regime((0.05, 0.05), 20) is False
    assert _winsor_zero_rank_regime((0.05, 0.05), 1) is True
    assert _winsor_zero_rank_regime((0.05, 0.05), 0) is True  # degenerate
    # asymmetric single-sided
    assert _winsor_zero_rank_regime((0.05, 0.0), 19) is True
    assert _winsor_zero_rank_regime((0.0, 0.05), 19) is True
    assert _winsor_zero_rank_regime((0.03, 0.05), 19) is True
    assert _winsor_zero_rank_regime((0.03, 0.05), 20) is False  # high side
    assert _winsor_zero_rank_regime((0.0, 0.0), 10**6) is True
    # rank > 0 at small N -> never fast
    assert _winsor_zero_rank_regime((0.25, 0.25), 12) is False
    assert _winsor_zero_rank_regime((0.3, 0.3), 4) is False
    assert _winsor_zero_rank_regime((0.2, 0.2), 4) is True
    # negative / NaN / infinite limits never qualify (CPU branch mirrors)
    assert _winsor_zero_rank_regime((-0.05, 0.05), 12) is False
    assert _winsor_zero_rank_regime((0.05, -0.05), 12) is False
    assert _winsor_zero_rank_regime((float("nan"), 0.05), 12) is False
    assert _winsor_zero_rank_regime((0.05, float("nan")), 12) is False
    assert _winsor_zero_rank_regime((float("inf"), 0.05), 12) is False
    assert _winsor_zero_rank_regime((0.05, float("inf")), 12) is False


# ---------------------------------------------------------------------------
# 2. mechanical branch proofs (spy counters on the sort helpers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [12, 19])
def test_zero_rank_fastpath_skips_all_winsor_sorts(n, monkeypatch):
    """N_batch=12 and 19 (default limits): NOT one call to the winsorize
    axis-0 sort helper nor to the survivor-bound sort helper — the fast path
    is taken — while the result stays CPU-exact and genuinely exercises
    rejection (rejected_pct > 0, i.e. the iterative loop really ran)."""
    a = _make_stack(n, (_H, _W), seed=200 + n, nan_frac=0.03, spike=0.06)
    w = _weights(n, seed=21)
    calls = {"axis0": 0, "bounds": 0}
    real_axis0 = sgp._winsorize_axis0_cp
    real_bounds = sgp._winsorize_bounds_cp

    def axis0_spy(mod, arr, limits):
        calls["axis0"] += 1
        return real_axis0(mod, arr, limits)

    def bounds_spy(mod, arr, limits):
        calls["bounds"] += 1
        return real_bounds(mod, arr, limits)

    monkeypatch.setattr(sgp, "_winsorize_axis0_cp", axis0_spy)
    monkeypatch.setattr(sgp, "_winsorize_bounds_cp", bounds_spy)
    cpu = _cpu(a, w)
    gpu = _gpu(a, w)
    assert calls == {"axis0": 0, "bounds": 0}, calls
    assert float(cpu[2]) > 0.0, "test design: expected real rejections"
    assert_parity(f"fastpath_n{n}", cpu, gpu)


def test_n19_zero_rank_fastpath_rewinsor_true_bounds_minmax_no_sort(
    monkeypatch,
):
    """N_batch=19 with apply_rewinsor=True: rejected samples ARE rewinsor-
    substituted (rejected_pct > 0) yet the survivor-bound sort helper is
    never called (min/max replacement), and parity with the CPU holds."""
    n = 19
    a = _make_stack(n, (_H, _W), seed=201, nan_frac=0.03, spike=0.08)
    w = _weights(n, seed=22)
    calls = {"axis0": 0, "bounds": 0}
    real_axis0 = sgp._winsorize_axis0_cp
    real_bounds = sgp._winsorize_bounds_cp

    def axis0_spy(mod, arr, limits):
        calls["axis0"] += 1
        return real_axis0(mod, arr, limits)

    def bounds_spy(mod, arr, limits):
        calls["bounds"] += 1
        return real_bounds(mod, arr, limits)

    monkeypatch.setattr(sgp, "_winsorize_axis0_cp", axis0_spy)
    monkeypatch.setattr(sgp, "_winsorize_bounds_cp", bounds_spy)
    cpu = _cpu(a, w)
    gpu = _gpu(a, w)
    assert calls == {"axis0": 0, "bounds": 0}, calls
    assert float(cpu[2]) > 0.0, "test design: expected real rejections"
    assert_parity("fastpath_n19_rewinsor", cpu, gpu)


def test_n20_nonzero_rank_takes_slow_path_and_stays_exact(monkeypatch):
    """N_batch=20 (default limits): the fast path MUST NOT trigger — the
    order-statistics helpers are called — and the result stays CPU-exact."""
    n = 20
    a = _make_stack(n, (_H, _W), seed=202, nan_frac=0.04, spike=0.05)
    w = _weights(n, seed=23)
    calls = {"axis0": 0, "bounds": 0}
    real_axis0 = sgp._winsorize_axis0_cp
    real_bounds = sgp._winsorize_bounds_cp

    def axis0_spy(mod, arr, limits):
        calls["axis0"] += 1
        return real_axis0(mod, arr, limits)

    def bounds_spy(mod, arr, limits):
        calls["bounds"] += 1
        return real_bounds(mod, arr, limits)

    monkeypatch.setattr(sgp, "_winsorize_axis0_cp", axis0_spy)
    monkeypatch.setattr(sgp, "_winsorize_bounds_cp", bounds_spy)
    cpu = _cpu(a, w)
    gpu = _gpu(a, w)
    assert calls["axis0"] >= 1, calls  # per-iteration winsorization ran
    assert calls["bounds"] == 1, calls  # apply_rewinsor survivor-bound sort
    assert _winsor_zero_rank_regime(DEFAULT_LIMITS, n) is False
    assert_parity("slowpath_n20", cpu, gpu)


def test_zero_limits_any_n_fastpath(monkeypatch):
    """(0.0, 0.0): the regime is zero-rank at ANY population (no winsor side
    ever replaces) — N_batch=40 must skip the sort helpers too."""
    n = 40
    a = _make_stack(n, (_H, _W), seed=203, nan_frac=0.03, spike=0.06)
    calls = {"axis0": 0}
    real_axis0 = sgp._winsorize_axis0_cp

    def axis0_spy(mod, arr, limits):
        calls["axis0"] += 1
        return real_axis0(mod, arr, limits)

    monkeypatch.setattr(sgp, "_winsorize_axis0_cp", axis0_spy)
    cpu = _cpu(a, None, winsor_limits=(0.0, 0.0))
    gpu = _gpu(a, None, winsor_limits=(0.0, 0.0))
    assert calls == {"axis0": 0}
    assert_parity("zero_limits_n40", cpu, gpu)


def test_rewinsor_false_fastpath_no_sort_at_all(monkeypatch):
    """apply_rewinsor=False in the zero-rank regime: NO winsor sorting at
    all (neither iterative nor survivor-bound) while rejections still occur."""
    n = 12
    a = _make_stack(n, (_H, _W), seed=204, nan_frac=0.03, spike=0.08)
    w = _weights(n, seed=24)
    calls = {"axis0": 0, "bounds": 0}
    real_axis0 = sgp._winsorize_axis0_cp
    real_bounds = sgp._winsorize_bounds_cp

    def axis0_spy(mod, arr, limits):
        calls["axis0"] += 1
        return real_axis0(mod, arr, limits)

    def bounds_spy(mod, arr, limits):
        calls["bounds"] += 1
        return real_bounds(mod, arr, limits)

    monkeypatch.setattr(sgp, "_winsorize_axis0_cp", axis0_spy)
    monkeypatch.setattr(sgp, "_winsorize_bounds_cp", bounds_spy)
    cpu = _cpu(a, w, apply_rewinsor=False)
    gpu = _gpu(a, w, apply_rewinsor=False)
    assert calls == {"axis0": 0, "bounds": 0}, calls
    assert float(cpu[2]) > 0.0, "test design: expected real rejections"
    assert_parity("rewinsor_false_fastpath", cpu, gpu)


# ---------------------------------------------------------------------------
# 3. bit-identity of the fast path vs the slow path (forced regime)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag",
    [
        "nan_scattered",
        "zero_rank_boundary_n19",
        "ties_duplicates",
        "pos_neg_inf_mix",
        "catastrophic_column_rejection",
        "rewinsor_false",
    ],
)
def test_fastpath_bitwise_identical_to_slowpath(tag, monkeypatch):
    """Forcing the slow path (regime predicate -> False) must produce a
    BITWISE identical result, weight map and rejected_pct to the fast path."""
    entry = next(e for e in FP_MATRIX if e[0] == tag)
    _, builder, expected_fast = entry
    assert expected_fast is True, "bit-identity test needs a fast-regime case"
    a, w, kw = builder()

    monkeypatch.setattr(sgp, "_winsor_zero_rank_regime", lambda *a: False)
    slow = _gpu(a, w, **kw)
    monkeypatch.undo()
    fast = _gpu(a, w, **kw)

    assert np.array_equal(fast[0], slow[0], equal_nan=True), tag
    assert np.array_equal(fast[1], slow[1], equal_nan=True), tag
    assert fast[2] == slow[2], tag
    # and both agree with the CPU reference within the documented tolerance
    cpu = _cpu(a, w, **kw)
    assert_parity(tag + "_fast_vs_cpu", cpu, fast)


# ---------------------------------------------------------------------------
# 4. §31 qualification matrix (CPU authoritative, metrics recorded)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag", [e[0] for e in FP_MATRIX], ids=[e[0] for e in FP_MATRIX]
)
def test_qualification_matrix(tag):
    entry = next(e for e in FP_MATRIX if e[0] == tag)
    _, _builder, expected_fast = entry
    a, w, kw = _builder()
    if expected_fast is not None:
        # mechanical trigger check: the regime decision matches the matrix
        # expectation for THIS reduction population and limits
        assert _expected_regime(a, kw) is expected_fast, tag
    cpu = _cpu(a, w, **kw)
    gpu = _gpu(a, w, **kw)
    assert_parity(tag, cpu, gpu)


def test_full_max_iters_fast_regime_ran_multiple_iterations(monkeypatch):
    """The n=19 fast-regime drift case runs the FULL max_iters on the CPU
    reference (5 winsorize calls; no early exit) — proven by a CPU spy — and
    the GPU fast path still reproduces the CPU exactly."""
    n = 19
    a = _make_stack(n, (16, 16), seed=125, nan_frac=0.02, spike=0.05)
    a = a + np.linspace(0.0, 60.0, n).reshape(n, 1, 1).astype(np.float32)
    w = _weights(n, seed=17)
    kw = dict(kappa=2.0, max_iters=5, kappa_decay=0.6)

    calls = {"cpu": 0, "gpu": 0}
    cpu_orig = sm._winsorize_axis0_numpy
    gpu_orig = sgp._winsorize_axis0_cp

    def cpu_spy(arr, limits):
        calls["cpu"] += 1
        return cpu_orig(arr, limits)

    def gpu_spy(mod, arr, limits):
        calls["gpu"] += 1
        return gpu_orig(mod, arr, limits)

    monkeypatch.setattr(sm, "_winsorize_axis0_numpy", cpu_spy)
    monkeypatch.setattr(sgp, "_winsorize_axis0_cp", gpu_spy)
    cpu = _cpu(a, w, **kw)
    gpu = _gpu(a, w, **kw)
    assert calls["cpu"] == 5, calls  # full max_iters, no early exit
    assert calls["gpu"] == 0, calls  # fast path: no iterative winsor sort
    assert float(cpu[2]) > 50.0, "test design: heavy multi-iteration cascade"
    assert_parity("full_max_iters_fast_regime", cpu, gpu)


def test_partial_final_batch_uses_actual_population_not_frozen_resolved(
    monkeypatch,
):
    """Engine partial-final-batch semantics: B_resolved may be frozen at 20
    while the actual flush to the GPU twin holds 12 frames.  The twin only
    ever sees the 12 actual frames (N_batch = 12): the zero-rank trigger is
    computed on that ACTUAL population (fast path, no sort), and the result
    is CPU-exact for the same 12-frame reduction."""
    a = _make_stack(12, (_H, _W), seed=205, nan_frac=0.04, spike=0.05)
    w = _weights(12, seed=25)
    assert _winsor_zero_rank_regime(DEFAULT_LIMITS, 20) is False  # B_resolved
    assert _winsor_zero_rank_regime(DEFAULT_LIMITS, 12) is True   # N_batch
    calls = {"axis0": 0}
    real_axis0 = sgp._winsorize_axis0_cp

    def axis0_spy(mod, arr, limits):
        calls["axis0"] += 1
        return real_axis0(mod, arr, limits)

    monkeypatch.setattr(sgp, "_winsorize_axis0_cp", axis0_spy)
    cpu = _cpu(a, w)
    gpu = _gpu(a, w)
    assert calls == {"axis0": 0}, calls  # fast path on the ACTUAL 12 frames
    assert_parity("partial_final_batch_12_of_20", cpu, gpu)


def test_nan_padded_batch_remains_exact_on_slow_path():
    """Conservative witness: a 12-frame reduction padded with 8 all-NaN
    frames reaches the twin as a 20-frame call (N_batch=20 -> slow path even
    though only 12 frames carry data).  The result must stay CPU-exact for
    that same padded stack (the CPU winsorizes per-pixel valid counts, so
    this is still the identity regime — the trigger is only conservative)."""
    a = _make_stack(12, (_H, _W), seed=206, nan_frac=0.04, spike=0.05)
    pad = np.full((8,) + a.shape[1:], np.nan, dtype=np.float32)
    padded = np.concatenate([a, pad], axis=0)
    w12 = _weights(12, seed=26)
    w = np.concatenate([w12, np.full(8, 0.5, dtype=np.float32)])
    assert _winsor_zero_rank_regime(DEFAULT_LIMITS, padded.shape[0]) is False
    cpu = _cpu(padded, w)
    gpu = _gpu(padded, w)
    assert_parity("nan_padded_20_exact_slow", cpu, gpu)
