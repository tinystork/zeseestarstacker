"""Tests for the CuPy reduction kernels and their queue-manager wiring.

Three layers:

A. CPU/GPU parity on synthetic stacks (real CuPy; skipped cleanly when cupy
   is absent so CI without a GPU stays green) — result, weight map and
   rejected-pct must match ``seestar.core.stack_methods`` CPU reference.
B. ``SeestarQueuedStacker._gpu_reduce`` dispatch: GPU invoked only when the
   backend resolves to ``"cupy"``; CPU otherwise; VRAM no-fit and GPU-failure
   both degrade to CPU without propagating.
C. Return contract: every GPU kernel returns NumPy float32 arrays (never CuPy).
"""

import logging

import numpy as np
import pytest

from seestar.core.stack_gpu import (
    stack_kappa_sigma_gpu,
    stack_linear_fit_clip_gpu,
    stack_median_gpu,
)
from seestar.core.stack_methods import (
    _stack_kappa_sigma,
    _stack_linear_fit_clip,
    _stack_median,
)

from seestar.queuep.queue_manager import SeestarQueuedStacker

try:
    import cupy  # noqa: F401

    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on non-GPU hosts
    CUPY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not CUPY_AVAILABLE, reason="CuPy not installed (no GPU stack available)"
)


# ---------------------------------------------------------------------------
# synthetic stacks
# ---------------------------------------------------------------------------


def _make_stack(n=12, shape=(32, 32), channels=None, seed=1234, nan_frac=0.05):
    """Synthetic stack: plausible ADU levels + noise + spikes + ~5% NaN."""
    rng = np.random.default_rng(seed)
    out_shape = (n,) + shape + ((channels,) if channels else ())
    sky = rng.uniform(800.0, 1200.0, size=out_shape)
    noise = rng.normal(0.0, 20.0, size=out_shape)
    arr = sky + noise
    spikes = rng.random(out_shape) < 0.04
    arr = np.where(spikes, arr + 500.0, arr)  # cosmic-ray-like positives
    arr = arr.astype(np.float32)
    nan_pos = rng.random(out_shape) < nan_frac
    arr[nan_pos] = np.nan
    return arr


def _weights(n, seed=7):
    rng = np.random.default_rng(seed)
    return list(rng.uniform(0.4, 1.6, size=n).astype(np.float32))


# ---------------------------------------------------------------------------
# A. parity: CPU reference vs GPU twin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [None, 3])
@pytest.mark.parametrize("weighted", [True, False])
def test_kappa_sigma_parity(channels, weighted):
    n = 12
    arr = _make_stack(n=n, channels=channels)
    images = list(arr)  # production passes a list of frames
    weights = _weights(n) if weighted else None
    cpu = _stack_kappa_sigma(
        images, weights, sigma_low=3.0, sigma_high=3.0, return_weights=True
    )
    gpu = stack_kappa_sigma_gpu(
        images, weights, sigma_low=3.0, sigma_high=3.0, return_weights=True
    )
    _assert_parity(cpu, gpu)


@pytest.mark.parametrize("channels", [None, 3])
@pytest.mark.parametrize("weighted", [True, False])
def test_linear_fit_clip_parity(channels, weighted):
    n = 12
    arr = _make_stack(n=n, channels=channels, seed=99)
    images = list(arr)
    weights = _weights(n, seed=3) if weighted else None
    cpu = _stack_linear_fit_clip(images, weights, sigma=3.0, return_weights=True)
    gpu = stack_linear_fit_clip_gpu(images, weights, sigma=3.0, return_weights=True)
    _assert_parity(cpu, gpu)


@pytest.mark.parametrize("channels", [None, 3])
def test_median_parity(channels):
    n = 12
    arr = _make_stack(n=n, channels=channels, seed=5)
    images = list(arr)
    weights = _weights(n)  # must be IGNORED by both kernels
    cpu = _stack_median(images, weights, return_weights=True)
    gpu = stack_median_gpu(images, weights, return_weights=True)
    _assert_parity(cpu, gpu, expect_rejection=False)


def _assert_parity(cpu, gpu, expect_rejection=True):
    # Return contract: NumPy arrays, float32.
    for candidate in gpu[:2]:
        assert isinstance(candidate, np.ndarray), type(candidate)
        assert candidate.dtype == np.float32, candidate.dtype
    # Numeric parity (GPU reductions may reorder float sums / sorts).
    np.testing.assert_allclose(gpu[0], cpu[0], rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(gpu[1], cpu[1], rtol=1e-3, atol=1e-2)
    # Rejection percentages must agree within 1.0 point.
    assert abs(float(gpu[2]) - float(cpu[2])) <= 1.0, (gpu[2], cpu[2])
    if expect_rejection:
        # Both kernels must actually reject something in this synthetic stack.
        assert cpu[2] > 0.0
    # Shapes preserved.
    assert gpu[0].shape == cpu[0].shape


def test_median_never_returns_cupy_arrays():
    arr = _make_stack(n=8)
    images = list(arr)
    gpu = stack_median_gpu(images, return_weights=True)
    assert isinstance(gpu[0], np.ndarray)
    assert isinstance(gpu[1], np.ndarray)
    assert gpu[1].dtype == np.float32


# ---------------------------------------------------------------------------
# B. _gpu_reduce dispatch (selection, VRAM fallback, failure fallback)
# ---------------------------------------------------------------------------


class _ProbeStacker:
    """Minimal stand-in exposing only what ``_gpu_reduce`` touches."""

    def __init__(self, backend):
        self._backend = backend
        self.logger = logging.getLogger("test_stack_gpu")

    @property
    def effective_backend(self):
        return self._backend

    _gpu_reduce = SeestarQueuedStacker._gpu_reduce
    _reduction_xp = SeestarQueuedStacker._reduction_xp


def test_gpu_reduce_uses_gpu_when_backend_cupy():
    stacker = _ProbeStacker("cupy")
    calls = {"cpu": 0, "gpu": 0}

    def fn_cpu(images, weights=None, **kw):
        calls["cpu"] += 1
        return "cpu-result"

    def fn_gpu(images, weights=None, **kw):
        calls["gpu"] += 1
        return "gpu-result"

    result = stacker._gpu_reduce(fn_cpu, fn_gpu, [np.zeros((2, 2), np.float32)])
    assert result == "gpu-result"
    assert calls == {"cpu": 0, "gpu": 1}


def test_gpu_reduce_uses_cpu_when_backend_cpu():
    stacker = _ProbeStacker("cpu")
    calls = {"cpu": 0, "gpu": 0}

    def fn_cpu(images, weights=None, **kw):
        calls["cpu"] += 1
        return "cpu-result"

    def fn_gpu(images, weights=None, **kw):
        calls["gpu"] += 1
        return "gpu-result"

    result = stacker._gpu_reduce(fn_cpu, fn_gpu, [np.zeros((2, 2), np.float32)])
    assert result == "cpu-result"
    assert calls == {"cpu": 1, "gpu": 0}


def test_gpu_reduce_vram_no_fit_falls_back_to_cpu(monkeypatch):
    stacker = _ProbeStacker("cupy")
    monkeypatch.setattr(stacker, "_reduction_xp", lambda images: None)
    calls = {"cpu": 0, "gpu": 0}

    def fn_cpu(images, weights=None, **kw):
        calls["cpu"] += 1
        return "cpu-result"

    def fn_gpu(images, weights=None, **kw):
        calls["gpu"] += 1
        return "gpu-result"

    result = stacker._gpu_reduce(fn_cpu, fn_gpu, [np.zeros((2, 2), np.float32)])
    assert result == "cpu-result"
    assert calls == {"cpu": 1, "gpu": 0}


def test_gpu_reduce_gpu_failure_falls_back_to_cpu(monkeypatch):
    stacker = _ProbeStacker("cupy")
    cpu_value = ("cpu-result", "cpu-weights", 12.5)

    def fn_cpu(images, weights=None, **kw):
        return cpu_value

    def fn_gpu(images, weights=None, **kw):
        raise RuntimeError("simulated GPU kernel failure")

    # Force the GPU branch (tiny stack fits VRAM) but make fn_gpu raise.
    result = stacker._gpu_reduce(fn_cpu, fn_gpu, [np.zeros((2, 2), np.float32)])
    assert result == cpu_value  # CPU fallback result, no exception propagated


# ---------------------------------------------------------------------------
# C. end-to-end: real GPU kernels through _gpu_reduce equal CPU kernels
# ---------------------------------------------------------------------------


def test_gpu_reduce_real_parity_small_stack():
    stacker = _ProbeStacker("cupy")
    arr = _make_stack(n=10)
    images = list(arr)
    weights = _weights(10)

    cpu = _stack_kappa_sigma(
        images, weights, sigma_low=3.0, sigma_high=3.0, return_weights=True
    )
    gpu = stacker._gpu_reduce(
        _stack_kappa_sigma,
        stack_kappa_sigma_gpu,
        images,
        weights,
        sigma_low=3.0,
        sigma_high=3.0,
        return_weights=True,
    )
    np.testing.assert_allclose(gpu[0], cpu[0], rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(gpu[1], cpu[1], rtol=1e-3, atol=1e-2)
    assert abs(float(gpu[2]) - float(cpu[2])) <= 1.0


# ---------------------------------------------------------------------------
# F7: boring subprocess constructs SeestarQueuedStacker(gpu=args.request_gpu);
# prove the constructor-resolved backend drives _gpu_reduce end to end.
# ---------------------------------------------------------------------------


def test_boring_style_gpu_true_constructor_invokes_gpu_kernel():
    """``SeestarQueuedStacker(gpu=True)`` (as boring_stack.py now does after
    F7) resolves ``effective_backend == \"cupy\"`` on a CuPy host and routes
    an eligible reduction through the GPU kernel with CPU parity."""
    stacker = SeestarQueuedStacker(gpu=True)
    try:
        assert stacker.effective_backend == "cupy"
        images = list(_make_stack(n=8, seed=21))
        weights = _weights(8, seed=2)

        gpu_calls = []

        def spied_gpu(*args, **kwargs):
            gpu_calls.append(1)
            return stack_kappa_sigma_gpu(*args, **kwargs)

        cpu_ref = _stack_kappa_sigma(
            images, weights, sigma_low=3.0, sigma_high=3.0, return_weights=True
        )
        result = stacker._gpu_reduce(
            _stack_kappa_sigma,
            spied_gpu,
            images,
            weights,
            sigma_low=3.0,
            sigma_high=3.0,
            return_weights=True,
        )
        assert gpu_calls == [1], "GPU kernel must be invoked when backend=cupy"
        np.testing.assert_allclose(result[0], cpu_ref[0], rtol=1e-3, atol=1e-2)
        np.testing.assert_allclose(result[1], cpu_ref[1], rtol=1e-3, atol=1e-2)
    finally:
        stacker.quality_executor.shutdown(wait=False, cancel_futures=True)


def test_boring_style_no_gpu_intent_uses_cpu_kernel():
    """``SeestarQueuedStacker()`` (no ``--gpu``; request_gpu=False) resolves
    ``effective_backend == \"cpu\"`` and never invokes the GPU kernel."""
    stacker = SeestarQueuedStacker()  # gpu defaults to False
    try:
        assert stacker.effective_backend == "cpu"
        images = list(_make_stack(n=8, seed=22))
        weights = _weights(8, seed=4)
        gpu_calls = []

        def spied_gpu(*args, **kwargs):
            gpu_calls.append(1)
            raise AssertionError("GPU kernel must not run when backend=cpu")

        result = stacker._gpu_reduce(
            _stack_kappa_sigma,
            spied_gpu,
            images,
            weights,
            sigma_low=3.0,
            sigma_high=3.0,
            return_weights=True,
        )
        assert gpu_calls == []
        cpu_ref = _stack_kappa_sigma(
            images, weights, sigma_low=3.0, sigma_high=3.0, return_weights=True
        )
        np.testing.assert_allclose(result[0], cpu_ref[0], rtol=1e-6, atol=1e-5)
    finally:
        stacker.quality_executor.shutdown(wait=False, cancel_futures=True)
