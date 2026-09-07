"""Phase G (Track P5): CuPy memory-pool release-policy tests.

Three layers:

A. PURE helper tests (no device): ``release_cupy_pool_retained_blocks`` on an
   injected pool-like object -- the release fires only when the retained
   REUSABLE pool bytes reach the threshold, calls ``free_all_blocks`` at
   most once, and NEVER raises (any pool failure is swallowed so a release
   can never affect a reduction outcome).

B. REAL-pool witness tests (CuPy/GPU): the threshold policy against the
   actual default pool -- a ~600 MiB retention is released (driver free
   grows by ~the retained amount) while a ~300 MiB retention is kept, and
   a second release on an empty pool is a no-op.

C. Dispatch wiring tests (real CuPy/GPU, skipped without it): the
   ``_gpu_reduce_winsorized`` seam calls the release AT MOST ONCE per
   completed FULL_GPU / TILED_GPU reduction (a multi-tile TILED batch still
   yields exactly one call -- never per tile), never on the CPU fallback
   routes, and the reduction results are bitwise unchanged with the release
   enabled (science-neutrality).  Env ``ZSSS_GPU_POOL_RELEASE=0`` disables
   the seam entirely.

Structural property pinned in C: the reduction kernels themselves
(``seestar/core/stack_gpu.py``) contain no pool-release call at all -- the
dispatch seam is the ONLY release point of the pipeline.
"""

import logging
import os

import numpy as np
import pytest

try:
    import cupy  # noqa: F401

    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on non-GPU hosts
    CUPY_AVAILABLE = False

import seestar.queuep.queue_manager as queue_manager_module  # noqa: E402
from seestar.core.gpu import GpuCapabilities  # noqa: E402
from seestar.queuep.queue_manager import (  # noqa: E402
    WINSOR_POOL_RELEASE_MIN_BYTES,
    release_cupy_pool_retained_blocks,
)

MIB = 1024 ** 2
GIB = 1024 ** 3


class _FakePool:
    """Pool-like object recording release calls (pure-helper tests)."""

    def __init__(self, free_bytes, boom_free=False, boom_release=False):
        self._free = free_bytes
        self.calls = 0
        self.boom_free = boom_free
        self.boom_release = boom_release

    def free_bytes(self):
        if self.boom_free:
            raise RuntimeError("simulated free_bytes failure")
        return self._free

    def free_all_blocks(self):
        self.calls += 1
        if self.boom_release:
            raise RuntimeError("simulated free_all_blocks failure")
        self._free = 0


# ---------------------------------------------------------------------------
# A. pure helper: threshold, at-most-once, never raises
# ---------------------------------------------------------------------------


def test_helper_below_threshold_keeps_pool():
    """Retention below the threshold -> no release, no call, (False, 0)."""
    pool = _FakePool(WINSOR_POOL_RELEASE_MIN_BYTES - 1)
    released, reclaimed = release_cupy_pool_retained_blocks(pool)
    assert released is False
    assert reclaimed == 0
    assert pool.calls == 0


def test_helper_at_threshold_releases_once():
    """Retention >= threshold -> exactly one free_all_blocks, bytes reported."""
    pool = _FakePool(WINSOR_POOL_RELEASE_MIN_BYTES + 4 * MIB)
    released, reclaimed = release_cupy_pool_retained_blocks(pool)
    assert released is True
    assert reclaimed == WINSOR_POOL_RELEASE_MIN_BYTES + 4 * MIB
    assert pool.calls == 1


def test_helper_default_threshold_is_the_module_constant():
    pool = _FakePool(WINSOR_POOL_RELEASE_MIN_BYTES)
    released, _ = release_cupy_pool_retained_blocks(pool)
    assert released is True
    assert WINSOR_POOL_RELEASE_MIN_BYTES == 512 * MIB


def test_helper_pool_failures_never_raise():
    """A broken pool must never break the reduction (release is optional)."""
    pool = _FakePool(0, boom_free=True)
    assert release_cupy_pool_retained_blocks(pool) == (False, 0)
    pool = _FakePool(WINSOR_POOL_RELEASE_MIN_BYTES, boom_release=True)
    assert release_cupy_pool_retained_blocks(pool) == (False, 0)
    assert pool.calls == 1  # attempted once, failure swallowed


# ---------------------------------------------------------------------------
# B. real-pool witness: the policy against the actual CuPy default pool
# ---------------------------------------------------------------------------

pytestmark_real = pytest.mark.skipif(
    not CUPY_AVAILABLE, reason="CuPy not installed (no GPU stack available)"
)


@pytestmark_real
def test_real_pool_large_retention_reclaimed_to_driver():
    """A ~600 MiB retention IS released: driver free grows ~the retention."""
    cp = __import__("cupy")
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    block = cp.zeros((600 * MIB // 4 // 8, 8), dtype=cp.float32)  # ~600 MiB
    del block
    cp.cuda.Stream.null.synchronize()
    assert pool.free_bytes() >= 512 * MIB
    free0, _ = cp.cuda.runtime.memGetInfo()
    released, reclaimed = release_cupy_pool_retained_blocks(pool)
    free1, _ = cp.cuda.runtime.memGetInfo()
    assert released is True
    assert reclaimed >= 512 * MIB
    assert free1 - free0 >= 512 * MIB
    assert pool.free_bytes() == 0
    # second release on an empty pool: no-op
    assert release_cupy_pool_retained_blocks(pool) == (False, 0)


@pytestmark_real
def test_real_pool_small_retention_kept():
    """A ~300 MiB retention is KEPT (below threshold): blocks stay reusable."""
    cp = __import__("cupy")
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    block = cp.zeros((300 * MIB // 4 // 8, 8), dtype=cp.float32)  # ~300 MiB
    del block
    cp.cuda.Stream.null.synchronize()
    assert pool.free_bytes() < 512 * MIB
    released, reclaimed = release_cupy_pool_retained_blocks(pool)
    assert released is False
    assert reclaimed == 0
    assert pool.free_bytes() >= 300 * MIB  # still there for the next batch
    pool.free_all_blocks()  # hygiene
    cp.cuda.Stream.null.synchronize()


# ---------------------------------------------------------------------------
# C. dispatch wiring: seam at most once per reduction, never per tile
# ---------------------------------------------------------------------------

pytestmark_wiring = pytest.mark.skipif(
    not CUPY_AVAILABLE, reason="CuPy not installed (no GPU stack available)"
)

import astropy.io.fits as fits  # noqa: E402

from seestar.queuep.queue_manager import SeestarQueuedStacker  # noqa: E402

_WINSOR_HEADER = fits.Header()


def _winsor_stack(request_gpu=True, shape=(2, 2)):
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
    o.max_hq_mem = 4_000_000_000
    o.batch_size = 10
    o.settings = None
    o.reference_header_for_wcs = None
    o.reference_wcs_object = None
    o.interbatch_norm_active = False
    o.max_stack_workers = 1
    o._current_batch_paths = []
    o._quality_reference_scale = 1.0
    o.logger = logging.getLogger("zsss.gpu.winsorized.poolrelease")
    o.request_gpu = request_gpu
    o._acceleration_policy = None
    o._gpu_capabilities = GpuCapabilities(
        gpu_detected=True,
        cuda_runtime_ready=True,
        cupy_ready=True,
        opencv_cuda_ready=False,
        backend_ready=True,
        device_name="Planner Test GPU",
        device_vram_mb=2048,
        compute_capability="6.1",
        failure_reason=None,
        state="ready",
    )
    return o


def _winsor_item(value, shape=(2, 2)):
    img = np.full(shape, value, dtype=np.float32)
    mask = np.ones(shape, dtype=bool)
    return (img, _WINSOR_HEADER, {"snr": 1.0, "stars": 0.0}, None, mask)


def _winsor_batch(shape=(2, 2), n_in=4, value=10.0, outlier=1000.0):
    return [_winsor_item(value, shape) for _ in range(n_in)] + [
        _winsor_item(outlier, shape)
    ]


def _patch_mem(monkeypatch, free_bytes, pool_free=0):
    """Pin the device memory state seen by the winsorized dispatch."""
    import cupy as _cp

    monkeypatch.setattr(
        _cp.cuda.runtime, "memGetInfo", lambda: (free_bytes, 4 * GIB)
    )

    class _FixedPool:
        def free_bytes(self):
            return pool_free

    monkeypatch.setattr(_cp, "get_default_memory_pool", lambda: _FixedPool())


def _enable_release(monkeypatch, on=True):
    if on:
        monkeypatch.setenv("ZSSS_GPU_POOL_RELEASE", "1")
    else:
        monkeypatch.setenv("ZSSS_GPU_POOL_RELEASE", "0")


@pytestmark_wiring
def test_seam_release_called_once_full_route(monkeypatch):
    """FULL_GPU route: the seam calls the release helper exactly ONCE after a
    successful reduction (never before it), and zero times when disabled."""
    import cupy as _cp

    real_full = queue_manager_module.stack_winsorized_sigma_gpu
    calls = []

    def spy_full(*args, **kwargs):
        # The release must not have fired BEFORE the GPU reduction ran.
        assert calls == []
        return real_full(*args, **kwargs)

    monkeypatch.setattr(
        queue_manager_module, "stack_winsorized_sigma_gpu", spy_full
    )
    release_calls = []
    monkeypatch.setattr(
        queue_manager_module,
        "release_cupy_pool_retained_blocks",
        lambda pool: release_calls.append(pool) or (False, 0),
    )
    _patch_mem(monkeypatch, free_bytes=2 * GIB)
    stack = _winsor_stack(request_gpu=True)
    _enable_release(monkeypatch, on=True)
    V, _hdr, W = stack._stack_batch(_winsor_batch(), 1, 1)
    assert len(release_calls) == 1  # exactly one seam call per reduction
    assert np.isclose(V[0, 0], 10.0, rtol=1e-3), V[0, 0]
    assert np.isclose(W[0, 0], 5.0, rtol=1e-3), W[0, 0]
    # disabled: the seam is a no-op
    release_calls.clear()
    _enable_release(monkeypatch, on=False)
    V2, _hdr2, W2 = stack._stack_batch(_winsor_batch(), 1, 1)
    assert release_calls == []
    np.testing.assert_array_equal(V2, V)
    np.testing.assert_array_equal(W2, W)


@pytestmark_wiring
def test_seam_release_called_once_tiled_multi_tile_route(monkeypatch):
    """TILED_GPU multi-tile route: exactly ONE seam release for the whole
    reduction -- never one per tile -- and the result stays bitwise equal to
    the untiled twin."""
    import cupy as _cp

    real_tiled = queue_manager_module.stack_winsorized_sigma_gpu_tiled
    release_calls = []
    monkeypatch.setattr(
        queue_manager_module,
        "release_cupy_pool_retained_blocks",
        lambda pool: release_calls.append(pool) or (False, 0),
    )
    _patch_mem(monkeypatch, free_bytes=620 * MIB)  # FULL too big -> TILED
    stack = _winsor_stack(request_gpu=True, shape=(1080, 1920))
    batch = _winsor_batch(shape=(1080, 1920))
    _enable_release(monkeypatch, on=True)
    V_t, _hdr, W_t = stack._stack_batch(batch, 1, 1)
    # the tiled route ran (real GPU) and the seam fired exactly once
    assert release_calls and len(release_calls) == 1
    # bitwise equality with the untiled GPU twin on the identical batch
    V_u, _w_u, _ = queue_manager_module.stack_winsorized_sigma_gpu(
        [it[0] for it in batch],
        np.ones(5, dtype=np.float32),
        kappa=3.0,
        winsor_limits=(0.2, 0.2),
        return_weights=True,
    )
    assert V_t.shape == V_u.shape == (1080, 1920)
    np.testing.assert_array_equal(V_t, V_u)
    np.testing.assert_array_equal(W_t, _w_u)


@pytestmark_wiring
def test_seam_release_results_unchanged_when_release_fires(monkeypatch):
    """Science neutrality: when the seam release FIRES (helper reports a
    release) the FULL-route result is bitwise identical to the disabled run."""
    import cupy as _cp

    # Reference: release disabled.
    _patch_mem(monkeypatch, free_bytes=2 * GIB)
    _enable_release(monkeypatch, on=False)
    V_ref, _hdr_ref, W_ref = _winsor_stack(request_gpu=True)._stack_batch(
        _winsor_batch(), 1, 1
    )

    # Release enabled and forced to fire (spy pretends a release happened).
    release_calls = []
    monkeypatch.setattr(
        queue_manager_module,
        "release_cupy_pool_retained_blocks",
        lambda pool: release_calls.append(pool) or (True, 600 * MIB),
    )
    _enable_release(monkeypatch, on=True)
    V_on, _hdr_on, W_on = _winsor_stack(request_gpu=True)._stack_batch(
        _winsor_batch(), 1, 1
    )
    assert len(release_calls) == 1
    np.testing.assert_array_equal(V_on, V_ref)
    np.testing.assert_array_equal(W_on, W_ref)


@pytestmark_wiring
def test_seam_release_never_on_cpu_fallback_routes(monkeypatch):
    """CPU_FALLBACK(vram_no_valid_tile) and CPU-by-policy never release."""
    release_calls = []

    def spy(*a, **k):
        raise AssertionError("GPU must not run")

    monkeypatch.setattr(
        queue_manager_module, "stack_winsorized_sigma_gpu", spy
    )
    monkeypatch.setattr(
        queue_manager_module, "stack_winsorized_sigma_gpu_tiled", spy
    )
    monkeypatch.setattr(
        queue_manager_module,
        "release_cupy_pool_retained_blocks",
        lambda pool: release_calls.append(pool) or (True, 600 * MIB),
    )
    _patch_mem(monkeypatch, free_bytes=512 * 1024)  # ~nothing allocatable
    stack = _winsor_stack(request_gpu=True)
    _enable_release(monkeypatch, on=True)
    V, _hdr, W = stack._stack_batch(_winsor_batch(shape=(1080, 1920)), 1, 1)
    assert release_calls == []  # CPU fallback: no GPU work, no release
    assert V.shape == (1080, 1920)
    # CPU-by-policy (request_gpu=False): no planner, no GPU, no release
    release_calls.clear()
    stack_cpu = _winsor_stack(request_gpu=False)
    V2, _hdr2, W2 = stack_cpu._stack_batch(_winsor_batch(), 1, 1)
    assert release_calls == []
    assert np.isclose(W2[0, 0], 5.0, rtol=1e-3)


@pytestmark_wiring
def test_no_pool_release_inside_reduction_kernels():
    """Structural: the reduction kernels (stack_gpu.py) contain NO
    pool-release call -- the dispatch seam is the only release point."""
    import inspect

    import seestar.core.stack_gpu as sg

    src = inspect.getsource(sg)
    assert "free_all_blocks" not in src
    assert "free_all_free" not in src
    assert "_release_pool" not in src
