"""M3 (Track B) tests: CuPy Winsorized Sigma Clip scientific twin.

The CPU implementation in ``seestar/core/stack_methods.py``
(``_stack_winsorized_sigma_iter`` -> ``_winsorize_axis0_numpy`` /
``_winsorize_bounds``) is the SCIENTIFIC AUTHORITY and is never modified.
These tests prove the CuPy twin in ``seestar/core/stack_gpu.py``
(``stack_winsorized_sigma_gpu`` + ``_winsorize_axis0_cp`` /
``_winsorize_bounds_cp``) reproduces it:

* ordinary parity CPU vs GPU on RGB and monochrome stacks, weighted +
  unweighted (deterministic seeds),
* adversarial parity: NaNs, fully-invalid slices, one-valid-sample columns,
  two-valid-sample columns, outliers, ties, extreme winsor limits (0.0/1.0),
  ``apply_rewinsor=False``, and multi-iteration convergence (no early exit),
* return contract: NumPy float32 result/weight map, Python-float
  ``rejected_pct`` — never a CuPy array.

Real CuPy required; the whole module skips cleanly when cupy is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from seestar.core.stack_methods import (
    _stack_winsorized_sigma_iter,
    _winsorize_axis0_numpy,
    _winsorize_bounds,
)
from seestar.core.stack_gpu import (
    _winsorize_axis0_cp,
    _winsorize_bounds_cp,
    stack_winsorized_sigma_gpu,
)

try:
    import cupy as cp  # noqa: F401

    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - non-GPU hosts
    cp = None
    CUPY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not CUPY_AVAILABLE, reason="CuPy not installed (no GPU stack available)"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_stack(n, shape, channels=None, seed=0, nan_frac=0.03, spike=0.03):
    """Deterministic synthetic stack: sky + noise + spikes + NaN samples."""
    rng = np.random.default_rng(seed)
    out_shape = (n,) + shape + ((channels,) if channels else ())
    arr = rng.normal(1000.0, 20.0, size=out_shape).astype(np.float32)
    arr[rng.random(out_shape) < nan_frac] = np.nan
    arr = arr + np.where(
        rng.random(out_shape) < spike, 400.0, 0.0
    ).astype(np.float32)
    return arr


def _weights(n, seed=7):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.4, 1.6, size=n).astype(np.float32)


# ---------------------------------------------------------------------------
# B6 metric recorder
# ---------------------------------------------------------------------------
# Every B6 matrix case records machine-readable parity metrics into
# ``B6_METRICS`` (consumed by the qualification report) and asserts the
# documented parity contract.  ``differing`` counts elements whose |cpu-gpu|
# exceeds the documented combined tolerance (atol + rtol*|cpu|).

B6_METRICS = []


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


def _record_b6(tag, cpu, gpu, tolerance=(1e-3, 1e-2), strict=True):
    """Assert documented-tolerance parity and record B6 metrics.

    ``strict=False`` (adversarial boundary cases): parity metrics are still
    recorded, but the ordinary-case allclose contract is NOT asserted — the
    caller asserts the dedicated bounded-divergence classification instead.
    """
    c_res, c_w, c_pct = cpu
    g_res, g_w, g_pct = gpu
    if strict:
        np.testing.assert_allclose(
            g_res, c_res, rtol=tolerance[0], atol=tolerance[1], equal_nan=True
        )
        np.testing.assert_allclose(
            g_w, c_w, rtol=tolerance[0], atol=tolerance[1], equal_nan=True
        )
        assert abs(float(g_pct) - float(c_pct)) <= 1.0, (g_pct, c_pct, tag)
        assert g_res.shape == c_res.shape
    tol = tolerance[1] + tolerance[0] * np.abs(c_res.astype(np.float64))
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
            "cpu": (
                float(c_res[coords]) if np.isfinite(c_res[coords]) else None
            ),
            "gpu": (
                float(g_res[coords]) if np.isfinite(g_res[coords]) else None
            ),
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
    B6_METRICS.append((tag, metrics))
    return metrics


def _assert_parity(cpu, gpu, tag=""):
    c_res, c_w, c_pct = cpu
    g_res, g_w, g_pct = gpu
    # Return contract: NumPy arrays float32 + float rejected_pct.
    assert isinstance(g_res, np.ndarray), type(g_res)
    assert isinstance(g_w, np.ndarray), type(g_w)
    assert g_res.dtype == np.float32, g_res.dtype
    assert g_w.dtype == np.float32, g_w.dtype
    assert isinstance(g_pct, float), type(g_pct)
    assert not isinstance(g_res, cp.ndarray), "GPU array leaked out"
    assert not isinstance(g_w, cp.ndarray), "GPU array leaked out"
    # Numeric parity.
    np.testing.assert_allclose(g_res, c_res, rtol=1e-3, atol=1e-2, equal_nan=True)
    np.testing.assert_allclose(g_w, c_w, rtol=1e-3, atol=1e-2, equal_nan=True)
    assert abs(float(g_pct) - float(c_pct)) <= 1.0, (g_pct, c_pct, tag)
    assert g_res.shape == c_res.shape


# ---------------------------------------------------------------------------
# 1. ordinary parity (RGB + mono, weighted + unweighted)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [None, 3])
@pytest.mark.parametrize("weighted", [True, False])
def test_ordinary_parity(channels, weighted):
    n = 12
    arr = _make_stack(n, (24, 32), channels=channels, seed=1)
    images = list(arr)
    weights = _weights(n) if weighted else None
    cpu = _stack_winsorized_sigma_iter(
        images, weights, return_weights=True
    )
    gpu = stack_winsorized_sigma_gpu(
        images, weights, return_weights=True
    )
    _assert_parity(cpu, gpu, f"channels={channels} weighted={weighted}")


# ---------------------------------------------------------------------------
# 2. adversarial cases (start of B6)
# ---------------------------------------------------------------------------

_N, _H, _W = 10, 6, 6


def _adversarial_stack():
    rng = np.random.default_rng(13)
    a = rng.normal(1000.0, 20.0, size=(_N, _H, _W)).astype(np.float32)
    a[0, :, :] = np.nan                       # fully-invalid frame
    a[:, 0, 0] = np.nan                        # fully-invalid column
    a[1:, 1, 1] = np.nan                        # one-valid-sample column
    a[2:, 2, 2] = np.nan                        # two-valid-sample column
    a[3, 3, 3] = 1e6                           # outlier
    a[3, 4, 4] = -1e6                          # low outlier
    a[:, 5, 5] = 1000.0                         # ties (duplicates)
    a[4, 0, 1] = np.nan                         # scattered NaN
    return [a[i] for i in range(_N)]


@pytest.mark.parametrize("weighted", [True, False])
def test_adversarial_nan_invalid_1v_2v_outlier_ties(weighted):
    images = _adversarial_stack()
    weights = _weights(_N, seed=3) if weighted else None
    cpu = _stack_winsorized_sigma_iter(images, weights, return_weights=True)
    gpu = stack_winsorized_sigma_gpu(images, weights, return_weights=True)
    _assert_parity(cpu, gpu, f"adversarial weighted={weighted}")


@pytest.mark.parametrize("weighted", [True, False])
def test_adversarial_apply_rewinsor_false(weighted):
    images = _adversarial_stack()
    weights = _weights(_N, seed=5) if weighted else None
    cpu = _stack_winsorized_sigma_iter(
        images, weights, apply_rewinsor=False, return_weights=True
    )
    gpu = stack_winsorized_sigma_gpu(
        images, weights, apply_rewinsor=False, return_weights=True
    )
    _assert_parity(cpu, gpu, f"rewinsor=False weighted={weighted}")


@pytest.mark.parametrize("limits", [(0.0, 0.0), (0.0, 1.0), (0.5, 0.5)])
def test_adversarial_extreme_winsor_limits(limits):
    # 0.0/1.0 limits hit the clip/bound edge arithmetic on both sides.
    images = _adversarial_stack()
    weights = _weights(_N, seed=9)
    cpu = _stack_winsorized_sigma_iter(
        images, weights, winsor_limits=limits, return_weights=True
    )
    gpu = stack_winsorized_sigma_gpu(
        images, weights, winsor_limits=limits, return_weights=True
    )
    _assert_parity(cpu, gpu, f"limits={limits}")


def test_adversarial_multi_iteration_convergence_no_early_exit():
    # A drifting stack + decaying kappa forces several iterations: this case
    # does NOT early-exit on iteration 1 (verified: CPU rejects > 1 time).
    rng = np.random.default_rng(21)
    n = 30
    a = _make_stack(n, (10, 10), seed=22, nan_frac=0.02, spike=0.04)
    a = a + np.linspace(0, 40, n).reshape(n, 1, 1).astype(np.float32)
    images = list(a)
    weights = _weights(n, seed=4)
    cpu = _stack_winsorized_sigma_iter(
        images, weights, kappa=2.5, max_iters=8, kappa_decay=0.7,
        return_weights=True,
    )
    gpu = stack_winsorized_sigma_gpu(
        images, weights, kappa=2.5, max_iters=8, kappa_decay=0.7,
        return_weights=True,
    )
    # Sanity: the CPU reference itself did not early-break at iteration 1.
    assert float(cpu[2]) > 1.0, "test design: expected multi-iteration rejection"
    _assert_parity(cpu, gpu, "multi-iteration convergence")


def test_winsorize_axis0_helper_parity():
    """Helper-level parity: winsorization index arithmetic is bit-identical."""
    rng = np.random.default_rng(5)
    arr = rng.normal(100.0, 5.0, size=(12, 20, 30)).astype(np.float32)
    arr[rng.random(arr.shape) < 0.1] = np.nan
    for limits in [(0.05, 0.05), (0.2, 0.1), (0.0, 0.0), (0.0, 1.0)]:
        cpu = _winsorize_axis0_numpy(arr, limits)
        gpu = cp.asnumpy(_winsorize_axis0_cp(cp, cp.asarray(arr), limits))
        np.testing.assert_array_equal(
            np.isnan(cpu), np.isnan(gpu), err_msg=f"NaN map {limits}"
        )
        np.testing.assert_allclose(
            cpu, gpu, rtol=1e-6, atol=1e-6, equal_nan=True
        )
        lb_cpu, hb_cpu = _winsorize_bounds(arr, limits)
        lb_gpu, hb_gpu = (
            cp.asnumpy(x)
            for x in _winsorize_bounds_cp(cp, cp.asarray(arr), limits)
        )
        np.testing.assert_allclose(lb_cpu, lb_gpu, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(hb_cpu, hb_gpu, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 2b. B6 qualification matrix (M5): full adversarial coverage + metrics
# ---------------------------------------------------------------------------
# Every case runs BOTH implementations and records (tag, metrics) into
# ``B6_METRICS`` for the qualification report while asserting the documented
# tolerance contract (rtol=1e-3, atol=1e-2; rejected_pct within 1.0).  The
# CPU ``_stack_winsorized_sigma_iter`` remains the scientific authority.


def _stack_from_array(a, weights=None, **kw):
    imgs = [a[i] for i in range(a.shape[0])]
    cpu = _stack_winsorized_sigma_iter(imgs, weights, return_weights=True, **kw)
    gpu = stack_winsorized_sigma_gpu(imgs, weights, return_weights=True, **kw)
    return cpu, gpu


def _one_valid_columns(n=8, shape=(5, 4)):
    """Every column keeps exactly one valid (non-NaN) sample."""
    rng = np.random.default_rng(31)
    a = rng.normal(100.0, 10.0, size=(n,) + shape).astype(np.float32)
    for j in range(shape[1]):
        keep = j % n
        for i in range(n):
            if i != keep:
                a[i, :, j] = np.nan
    return a


def _two_valid_columns(n=8, shape=(5, 4)):
    """Every column keeps exactly two valid samples."""
    rng = np.random.default_rng(32)
    a = rng.normal(100.0, 10.0, size=(n,) + shape).astype(np.float32)
    for j in range(shape[1]):
        keep = {(j * 2) % n, (j * 2 + 1) % n}
        for i in range(n):
            if i not in keep:
                a[i, :, j] = np.nan
    return a


def _fully_invalid_slices(n=10, shape=(6, 6)):
    rng = np.random.default_rng(33)
    a = rng.normal(100.0, 10.0, size=(n,) + shape).astype(np.float32)
    a[0, :, :] = np.nan          # fully-invalid frame
    a[:, 0, 0] = np.nan          # fully-invalid column
    a[:, :, 2] = np.nan          # fully-invalid spatial column
    return a


def _unequal_weights_stack(n=12, shape=(12, 12), seed=34):
    rng = np.random.default_rng(seed)
    a = rng.normal(1000.0, 30.0, size=(n,) + shape).astype(np.float32)
    a[rng.random(a.shape) < 0.03] = np.nan
    w = np.geomspace(1e-3, 1e3, n).astype(np.float32)  # strongly unequal
    return a, w


def _extreme_low_variance(n=12, shape=(12, 12), seed=35):
    rng = np.random.default_rng(seed)
    a = np.full((n,) + shape, 1000.0, dtype=np.float32)
    a += rng.normal(0.0, 1e-3, size=(n,) + shape).astype(np.float32)
    return a


def _outlier_frame_stack(n=12, shape=(10, 10), seed=47):
    rng = np.random.default_rng(seed)
    a = rng.normal(1000.0, 25.0, size=(n,) + shape).astype(np.float32)
    a[3] += 5000.0                      # whole-frame outlier (e.g. cloud)
    a[7, 2, 2] = 1e6                    # single hot pixel
    a[9, 5, 5] = -1e6
    return a


def _ties_stack(n=12, shape=(8, 8), seed=48):
    rng = np.random.default_rng(seed)
    a = rng.normal(1000.0, 20.0, size=(n,) + shape).astype(np.float32)
    a = np.round(a / 8.0) * 8.0          # heavy ties / duplicates
    a[rng.random(a.shape) < 0.03] = np.nan
    return a


B6_TOL = (1e-3, 1e-2)


@pytest.mark.parametrize(
    "tag, builder",
    [
        ("mono_unweighted", lambda: _make_stack(12, (12, 12), seed=40)),
        ("mono_weighted", lambda: (_make_stack(12, (12, 12), seed=40), _weights(12, seed=1))),
        ("rgb_unweighted", lambda: _make_stack(12, (12, 12), channels=3, seed=41)),
        ("rgb_weighted", lambda: (_make_stack(12, (12, 12), channels=3, seed=41), _weights(12, seed=2))),
        ("dense_nan_40pct", lambda: _make_stack(10, (8, 8), seed=42, nan_frac=0.4)),
        ("partially_invalid_scattered", lambda: _make_stack(10, (8, 8), seed=43, nan_frac=0.12)),
        ("one_valid_per_column", lambda: _one_valid_columns()),
        ("one_valid_per_column_weighted", lambda: (_one_valid_columns(), _weights(8, seed=6))),
        ("two_valid_per_column", lambda: _two_valid_columns()),
        ("two_valid_per_column_weighted", lambda: (_two_valid_columns(), _weights(8, seed=7))),
        ("fully_invalid_slices", lambda: _fully_invalid_slices()),
        ("fully_invalid_slices_weighted", lambda: (_fully_invalid_slices(), _weights(10, seed=8))),
        ("strongly_unequal_weights", lambda: _unequal_weights_stack()),
        ("extreme_low_variance", lambda: (_extreme_low_variance(), _weights(12, seed=9))),
        ("outlier_frame", lambda: _outlier_frame_stack()),
        ("ties_duplicates", lambda: _ties_stack()),
        ("rewinsor_false", lambda: (_make_stack(10, (8, 8), seed=44), _weights(10, seed=10), dict(apply_rewinsor=False))),
        ("kappa_decay_0p5", lambda: (_make_stack(14, (8, 8), seed=45), _weights(14, seed=11), dict(kappa=3.0, max_iters=4, kappa_decay=0.5))),
        ("kappa_decay_1p0_narrow", lambda: (_make_stack(14, (8, 8), seed=46), _weights(14, seed=12), dict(kappa=1.2, max_iters=3, kappa_decay=1.0))),
    ],
)
def test_b6_qualification_matrix(tag, builder):
    """B6: parity + recorded metrics for every matrix case."""
    built = builder()
    if isinstance(built, tuple) and len(built) == 3:
        a, w, kw = built
    elif isinstance(built, tuple):
        a, w = built
        kw = {}
    else:
        a, w, kw = built, None, {}
    imgs = [a[i] for i in range(a.shape[0])]
    cpu = _stack_winsorized_sigma_iter(imgs, w, return_weights=True, **kw)
    gpu = stack_winsorized_sigma_gpu(imgs, w, return_weights=True, **kw)
    _record_b6(tag, cpu, gpu)


def test_b6_extreme_limits_0_1_and_1_0_nan_columns():
    """B6: (0.0, 1.0) and (1.0, 0.0) with a NaN in EVERY column.  The CPU
    computes (the low/high bound index stays in-bounds because
    ``n_valid < N``); the twin must reproduce the same output INCLUDING any
    degenerate inf handling, not diverge."""
    rng = np.random.default_rng(50)
    n, h, w = 8, 6, 5
    a = rng.normal(10.0, 2.0, size=(n, h, w)).astype(np.float32)
    for c in range(w):
        for r in range(h):
            a[r % n, r, c] = np.nan   # every column keeps at least one NaN
    for lim in [(0.0, 1.0), (1.0, 0.0)]:
        cpu, gpu = _stack_from_array(a, None, winsor_limits=lim)
        _record_b6(f"limits_{lim[0]}_{lim[1]}_nan_cols", cpu, gpu)


def test_b6_limits_1_0_all_valid_column_mirrors_indexerror():
    """B6: (1.0, 0.0) with an ALL-VALID column makes the CPU reference raise
    IndexError (``floor(1.0 * n_valid) == n_valid == N`` -> out-of-bounds
    take_along_axis).  The GPU twin must MIRROR that same failure instead of
    silently wrapping the out-of-range index."""
    rng = np.random.default_rng(51)
    a = rng.normal(10.0, 2.0, size=(8, 5, 5)).astype(np.float32)  # no NaN
    imgs = [a[i] for i in range(8)]
    w = _weights(8, seed=13)
    with pytest.raises(IndexError):
        _stack_winsorized_sigma_iter(
            imgs, w, winsor_limits=(1.0, 0.0), return_weights=True
        )
    with pytest.raises(IndexError):
        stack_winsorized_sigma_gpu(
            imgs, w, winsor_limits=(1.0, 0.0), return_weights=True
        )


def test_b6_ulp_boundary_divergence_is_bounded_not_algorithmic():
    """B6: a case that DOES land samples exactly on the mu +/- kappa*sigma
    clip boundary.  CPU (bottleneck nanmean/nanstd) and GPU (cupy reductions)
    differ in the last ULPs, so a handful of boundary pixels can flip
    acceptance and shift the rewinsorized value by ~1-2 ADU.  This must be a
    BOUNDED handful (weight maps identical, few differing elements, small max
    abs), i.e. the expected ULP threshold-boundary phenomenon, never an
    algorithmic divergence."""
    rng = np.random.default_rng(3)   # deterministic ULP-boundary seed
    n, h, width = 20, 160, 192
    a = rng.normal(1000.0, 30.0, size=(n, h, width)).astype(np.float32)
    a[rng.random(a.shape) < 0.01] = np.nan
    a = a + np.where(rng.random(a.shape) < 0.02, 300.0, 0.0).astype(np.float32)
    imgs = [a[i] for i in range(n)]
    weights = np.random.default_rng(103).uniform(0.5, 1.5, n).astype(np.float32)
    cpu = _stack_winsorized_sigma_iter(imgs, weights, return_weights=True)
    gpu = stack_winsorized_sigma_gpu(imgs, weights, return_weights=True)
    c_res, c_w, c_pct = cpu
    g_res, g_w, g_pct = gpu
    # Deterministic ULP-boundary witness: this seed lands >= 1 sample exactly
    # on the mu +/- kappa*sigma clip boundary, so CPU (bottleneck nanmean /
    # nanstd) and GPU (cupy reductions) differ in the last ULPs and a handful
    # of boundary pixels flip acceptance.  Outputs must stay FINITE and the
    # divergence must be a bounded handful of ~1-2 ADU shifts at ~1000 ADU
    # scale (verified: 2 differing pixels, max abs ~1.94 ADU on this host).
    assert not np.isinf(g_res).any() and not np.isinf(c_res).any()
    # Masks / weight maps must be essentially identical (ULP-level only).
    np.testing.assert_allclose(g_w, c_w, rtol=1e-4, atol=1e-3)
    assert abs(float(g_pct) - float(c_pct)) <= 1.0
    # The divergence must be a small, bounded handful of pixels.
    tol = 1e-2 + 1e-3 * np.abs(c_res.astype(np.float64))
    diff = np.abs(g_res.astype(np.float64) - c_res.astype(np.float64))
    diff = np.where(np.isnan(diff), 0.0, diff)
    differing = int(np.sum(diff > tol))
    max_abs = float(np.max(diff)) if diff.size else 0.0
    total = int(np.prod(c_res.shape))
    assert differing >= 1, "witness case must exercise the ULP boundary"
    assert differing <= max(20, total // 200), (differing, total)
    assert max_abs <= 10.0, max_abs  # ~1-2 ADU shifts at ~1000 ADU scale
    _record_b6("ulp_boundary_witness", cpu, gpu, strict=False)


def test_b6_max_iters_edge_full_5_iterations_no_early_exit(monkeypatch):
    """B6: a case that does NOT early-exit — the full ``max_iters`` (5)
    iterations run on both paths (kappa decay keeps rejecting).  Verified by
    counting winsorize-axis0 calls (one per loop iteration)."""
    import seestar.core.stack_methods as sm
    import seestar.core.stack_gpu as sgp

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
    rng = np.random.default_rng(53)
    n = 40
    a = rng.normal(1000.0, 25.0, size=(n, 12, 12)).astype(np.float32)
    a = a + np.linspace(0.0, 60.0, n).reshape(n, 1, 1).astype(np.float32)
    a[rng.random(a.shape) < 0.06] += 300.0
    w = _weights(n, seed=14)
    cpu, gpu = _stack_from_array(a, w, kappa=2.0, max_iters=5, kappa_decay=0.6)
    assert calls["cpu"] == 5, calls   # full 5 iterations, no early break
    assert calls["gpu"] == 5, calls
    _record_b6("max_iters_5_no_early_exit", cpu, gpu)


# ---------------------------------------------------------------------------
# 3. return contract
# ---------------------------------------------------------------------------


def test_return_contract_numpy_never_cupy():
    images = list(_make_stack(8, (10, 10), seed=2))
    weights = _weights(8, seed=1)
    for kw in ({"return_weights": True}, {"return_weights": False}):
        gpu = stack_winsorized_sigma_gpu(images, weights, **kw)
        for candidate in gpu:
            if isinstance(candidate, np.ndarray):
                assert candidate.dtype == np.float32
                assert not isinstance(candidate, cp.ndarray)
            else:
                assert isinstance(candidate, float)


def test_defaults_match_cpu_wrapper_defaults():
    """The GPU twin's defaults must equal the CPU ``_stack_winsorized_sigma``
    wrapper defaults (kappa=3.0, limits=(0.05,0.05), apply_rewinsor=True,
    max_iters=5, kappa_decay=0.9)."""
    import inspect

    cpu_sig = inspect.signature(_stack_winsorized_sigma_iter)
    gpu_sig = inspect.signature(stack_winsorized_sigma_gpu)
    for name in ("kappa", "winsor_limits", "apply_rewinsor", "max_iters",
                 "kappa_decay"):
        assert gpu_sig.parameters[name].default == cpu_sig.parameters[name].default


# ---------------------------------------------------------------------------
# B7: production dispatcher wiring (real _stack_batch -> _gpu_reduce)
# ---------------------------------------------------------------------------

import logging  # noqa: E402

import astropy.io.fits as fits  # noqa: E402

import seestar.queuep.queue_manager as queue_manager_module  # noqa: E402
from seestar.core.gpu import GpuCapabilities  # noqa: E402
from seestar.queuep.queue_manager import SeestarQueuedStacker  # noqa: E402

_WINSOR_HEADER = fits.Header()


def _gpu_winsor_stack(shape=(2, 2), request_gpu=True):
    """Lightweight SeestarQueuedStacker for the REAL ``_stack_batch``
    winsorized non-tiled dispatch (no ``__init__``; same attribute set as the
    HSI ``make_stack`` harness plus GPU intent/capabilities)."""
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o.stacking_mode = "winsorized-sigma-clip"
    o.normalize_method = "none"
    o.weighting_method = "none"
    o.use_quality_weighting = False
    o.weight_by_snr = False
    o.weight_by_stars = False
    o.snr_exponent = 1.0
    o.stars_exponent = 0.5
    o.min_weight = 0.0
    o.apply_batch_feathering = False
    o.reproject_between_batches = False
    o.reproject_coadd_final = False
    o.drizzle_active_session = False
    o.is_mosaic_run = False
    o.stack_kappa_low = 3.0
    o.stack_kappa_high = 3.0
    o.winsor_limits = (0.2, 0.2)
    o.stack_reject_algo = "none"
    o.max_hq_mem = 1_000_000_000
    o.batch_size = 10
    o.settings = None
    o.reference_header_for_wcs = None
    o.reference_wcs_object = None
    o.interbatch_norm_active = False
    o.max_stack_workers = 1
    o._current_batch_paths = []
    o._quality_reference_scale = 1.0
    o.logger = logging.getLogger("zsss.gpu.winsorized.dispatch")
    o.request_gpu = request_gpu
    o._acceleration_policy = None  # re-resolved from caps + intent below
    o._gpu_capabilities = GpuCapabilities(
        gpu_detected=True,
        cuda_runtime_ready=True,
        cupy_ready=True,
        opencv_cuda_ready=False,
        backend_ready=True,
        device_name="Dispatch Test GPU",
        device_vram_mb=2048,
        compute_capability="6.1",
        failure_reason=None,
        state="ready",
    )
    return o


def _winsor_item(value, shape=(2, 2)):
    """One batch item: constant float32 image + full-validity mask."""
    img = np.full(shape, value, dtype=np.float32)
    mask = np.ones(shape, dtype=bool)
    return (img, _WINSOR_HEADER, {"snr": 1.0, "stars": 0.0}, None, mask)


def _winsor_batch(shape=(2, 2)):
    # 4 inlier frames + 1 outlier: winsorized (0.2, 0.2) kappa=3 gives
    # V == 10.0 and W == 5 on both the CPU reference and the GPU twin.
    return [_winsor_item(10.0, shape) for _ in range(4)] + [
        _winsor_item(1000.0, shape)
    ]


def test_stack_batch_winsorized_reaches_gpu_twin(monkeypatch):
    """B7: stacking_mode=winsorized-sigma-clip + request_gpu=True + cupy ready
    + workload fits -> the production ``_stack_batch`` non-tiled path invokes
    ``stack_winsorized_sigma_gpu`` (spy) through ``_gpu_reduce``."""
    calls = []

    real_gpu = queue_manager_module.stack_winsorized_sigma_gpu

    def spy(*args, **kwargs):
        calls.append(args)
        return real_gpu(*args, **kwargs)

    monkeypatch.setattr(queue_manager_module, "stack_winsorized_sigma_gpu", spy)
    stack = _gpu_winsor_stack(request_gpu=True)
    assert stack.effective_backend == "cupy"
    V, _hdr, W = stack._stack_batch(_winsor_batch(), 1, 1)
    assert len(calls) == 1, "GPU twin must be invoked when backend=cupy"
    assert np.isclose(V[0, 0], 10.0, rtol=1e-3), V[0, 0]
    assert np.isclose(W[0, 0], 5.0, rtol=1e-3), W[0, 0]


def test_stack_batch_winsorized_uses_cpu_when_gpu_not_requested(monkeypatch):
    """B7: same workload + request_gpu=False -> CPU reference path executes
    and the GPU twin is NOT invoked."""
    calls = []
    real_gpu = queue_manager_module.stack_winsorized_sigma_gpu

    def spy(*args, **kwargs):
        calls.append(args)
        raise AssertionError("GPU kernel must not run without GPU intent")

    monkeypatch.setattr(queue_manager_module, "stack_winsorized_sigma_gpu", spy)
    stack = _gpu_winsor_stack(request_gpu=False)
    assert stack.effective_backend == "cpu"
    V, _hdr, W = stack._stack_batch(_winsor_batch(), 1, 1)
    assert calls == []
    assert np.isclose(V[0, 0], 10.0, rtol=1e-3), V[0, 0]
    assert np.isclose(W[0, 0], 5.0, rtol=1e-3), W[0, 0]


def test_stack_batch_winsorized_vram_no_fit_falls_back_cpu(
    monkeypatch, caplog
):
    """B7: request_gpu=True but the workload does NOT fit VRAM -> CPU fallback
    + durable fallback diagnostic (vram_reject), GPU twin never invoked."""
    import cupy as _cp

    class _EmptyPool:
        def free_bytes(self):
            return 0

    monkeypatch.setattr(_cp, "get_default_memory_pool", lambda: _EmptyPool())
    monkeypatch.setattr(
        _cp.cuda.runtime, "memGetInfo", lambda: (512 * 1024, 2 * 1024 ** 3)
    )
    calls = []
    real_gpu = queue_manager_module.stack_winsorized_sigma_gpu

    def spy(*args, **kwargs):
        calls.append(args)
        raise AssertionError("GPU must not run when VRAM does not fit")

    monkeypatch.setattr(queue_manager_module, "stack_winsorized_sigma_gpu", spy)
    caplog.set_level(logging.WARNING, logger="zsss.gpu.winsorized.dispatch")
    stack = _gpu_winsor_stack(request_gpu=True)
    # A genuinely non-fitting winsorized workload (6x footprint on 2 GiB).
    V, _hdr, W = stack._stack_batch(_winsor_batch(shape=(1080, 1920)), 1, 1)
    assert calls == []
    assert "vram_reject" in stack._gpu_fallback_logged
    assert "reduction needs" in caplog.text
    assert V.shape == (1080, 1920)
    assert np.isclose(W[0, 0], 5.0, rtol=1e-3), W[0, 0]


def test_stack_batch_winsorized_backend_error_falls_back_cpu(monkeypatch):
    """B7: the GPU twin raises a recoverable failure -> CPU fallback with the
    correct output and a visible diagnostic (warning)."""
    calls = []

    def boom(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("simulated GPU kernel failure")

    monkeypatch.setattr(queue_manager_module, "stack_winsorized_sigma_gpu", boom)
    stack = _gpu_winsor_stack(request_gpu=True)
    assert stack.effective_backend == "cupy"
    V, _hdr, W = stack._stack_batch(_winsor_batch(), 1, 1)
    assert len(calls) == 1
    assert np.isclose(V[0, 0], 10.0, rtol=1e-3), V[0, 0]
    assert np.isclose(W[0, 0], 5.0, rtol=1e-3), W[0, 0]


# ---------------------------------------------------------------------------
# B5: high-VRAM eligibility validation (simulation, real _reduction_xp)
# ---------------------------------------------------------------------------
# The eligibility model must scale naturally across capacities using the SAME
# general memory model (shape x dtype x per-operation footprint factor x 0.6
# headroom) — no hardware-name rules, no fixed N ceiling.  Capacities below
# are VALIDATION POINTS simulated via memGetInfo + pool.free_bytes.

import logging as _logging  # noqa: E402

_B5_CAPACITIES = [2, 8, 16, 24]  # GiB, validation points (not product tiers)


class _B5EligibleStacker:
    """Minimal stand-in exposing what ``_reduction_xp`` touches."""

    def __init__(self, logger_name="zsss.gpu.b5"):
        self._backend = "cupy"
        self.logger = _logging.getLogger(logger_name)
        self._gpu_fallback_logged = set()

    @property
    def effective_backend(self):
        return self._backend

    _reduction_xp = SeestarQueuedStacker._reduction_xp
    _log_gpu_fallback_once = SeestarQueuedStacker._log_gpu_fallback_once


def _b5_fake_images(n, shape=(1080, 1920)):
    """Fake frames carrying only ``shape``: the VRAM guard never reads data."""
    return [type("_Frame", (), {"shape": shape})() for _ in range(n)]


def _b5_admissible_n(capacity_gib, monkeypatch):
    """Largest N admissible at ``capacity_gib`` for a 1080x1920 mono stack
    under the REAL ``_reduction_xp`` with the winsorized footprint factor."""
    import cupy as _cp

    free_bytes = int(capacity_gib * (1024 ** 3))

    class _FullPool:
        def free_bytes(self):
            return 0

    monkeypatch.setattr(_cp, "get_default_memory_pool", lambda: _FullPool())
    monkeypatch.setattr(
        _cp.cuda.runtime, "memGetInfo", lambda: (free_bytes, free_bytes)
    )
    stacker = _B5EligibleStacker()
    lo, hi = 0, 4096
    while lo < hi:  # largest N with _reduction_xp(images) not None
        mid = (lo + hi + 1) // 2
        if stacker._reduction_xp(_b5_fake_images(mid)) is not None:
            lo = mid
        else:
            hi = mid - 1
    return lo


def test_b5_vram_eligibility_scales_with_capacity(monkeypatch):
    """B5: a 1080x1920xN stack ineligible at 2 GiB becomes eligible at larger
    simulated capacities; admissible N grows with capacity under the SAME
    memory model (no hardware-name rules, no fixed N ceiling)."""
    import seestar.queuep.queue_manager as _qm

    # Force the winsorized footprint factor (6x) through the real guard by
    # calling _reduction_xp directly with the factor argument (mirrors the
    # _gpu_reduce footprint_factor forwarding used by the B7 dispatch path).
    import cupy as _cp

    boundaries = {}
    for cap in _B5_CAPACITIES:
        free_bytes = int(cap * (1024 ** 3))

        class _FullPool:
            def free_bytes(self):
                return 0

        monkeypatch.setattr(_cp, "get_default_memory_pool", lambda: _FullPool())
        monkeypatch.setattr(
            _cp.cuda.runtime, "memGetInfo", lambda: (free_bytes, free_bytes)
        )
        stacker = _B5EligibleStacker()
        # Binary search largest admissible N at this capacity.
        lo, hi = 0, 4096
        while lo < hi:
            mid = (lo + hi + 1) // 2
            ok = (
                stacker._reduction_xp(
                    _b5_fake_images(mid),
                    footprint_factor=_qm._GPU_FOOTPRINT_FACTOR_WINSORIZED,
                )
                is not None
            )
            if ok:
                lo = mid
            else:
                hi = mid - 1
        boundaries[cap] = lo
    # Validation points are monotonic and strictly growing.
    caps = list(boundaries)
    for a, b in zip(caps, caps[1:]):
        assert boundaries[b] > boundaries[a], boundaries
    # A stack that is ineligible at 2 GiB becomes eligible at >= 8 GiB.
    n_ineligible_2gb = boundaries[2] + 1
    assert n_ineligible_2gb <= boundaries[8], boundaries
    # Same stack, larger capacity -> eligible under the identical model.
    stacker = _B5EligibleStacker()
    for cap in _B5_CAPACITIES:
        free_bytes = int(cap * (1024 ** 3))

        class _FullPool2:
            def free_bytes(self):
                return 0

        monkeypatch.setattr(_cp, "get_default_memory_pool", lambda: _FullPool2())
        monkeypatch.setattr(
            _cp.cuda.runtime, "memGetInfo", lambda: (free_bytes, free_bytes)
        )
        ok = (
            stacker._reduction_xp(
                _b5_fake_images(n_ineligible_2gb),
                footprint_factor=_qm._GPU_FOOTPRINT_FACTOR_WINSORIZED,
            )
            is not None
        )
        if cap >= 8:
            assert ok, (cap, n_ineligible_2gb)
        else:
            assert not ok, (cap, n_ineligible_2gb)
    # Record for the report: B5_BOUNDARIES[(capacity_gib)] = admissible N.
    B5_BOUNDARIES = boundaries
    assert B5_BOUNDARIES[2] < B5_BOUNDARIES[8] < B5_BOUNDARIES[16] < B5_BOUNDARIES[24]
