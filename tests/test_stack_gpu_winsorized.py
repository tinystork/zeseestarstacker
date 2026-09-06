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
