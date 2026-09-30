"""Lot C synthetic bounded GPU probe (MX150, CuPy runtime).

Forces a controlled FIRST-attempt GPU OOM (by monkeypatching the full driver
to raise a REAL cupy OutOfMemoryError, never a system OOM), then proves that
the dispatch seam's bounded retry recovers into a REAL spatial (tiled) GPU
reduction and returns the correct result.

Checks:
1. no_full_oom_fake: the injected OOM is a genuine cupy OOM (classifier says oom).
2. full_oom_recovers_tiled: seam retries FULL OOM -> TILED -> executed=gpu.
3. exhausted_to_cpu: every attempt OOMs -> CPU fallback, reason gpu_oom.
4. non_oom_no_retry: a RuntimeError does NOT trigger a spatial retry.

No M74 data, no heavy run.  Exits nonzero on any failure.
"""
from __future__ import annotations

import os
import sys

REPO = "/home/tristan/.openclaw/workspace/projects/zeseestarstacker"
sys.path.insert(0, REPO)

import logging  # noqa: E402
import numpy as np  # noqa: E402

import seestar.queuep.queue_manager as qm  # noqa: E402
from seestar.queuep.queue_manager import (  # noqa: E402
    GPU_EXEC_REASON_GPU_KERNEL,
    GPU_EXEC_REASON_GPU_OOM,
    WINSOR_GPU_OOM_MAX_ATTEMPTS,
    _winsorized_gpu_oom_kind,
)

failures = []


def check(tag, fn):
    try:
        fn()
        print("PASS %s" % tag)
    except Exception as exc:  # noqa: BLE001
        failures.append((tag, exc))
        print("FAIL %s: %r" % (tag, exc))


def _oom(cp):
    return cp.cuda.memory.OutOfMemoryError(1, 1, 1)


def _classifier():
    import cupy as cp

    return _winsorized_gpu_oom_kind(_oom(cp), cp)


def probe_classifier():
    assert _classifier() == "oom"
    import cupy as cp

    assert _winsorized_gpu_oom_kind(RuntimeError("bug"), cp) == "kernel"
    assert _winsorized_gpu_oom_kind(MemoryError("host"), cp) == "kernel"


def probe_full_oom_recovers_tiled():
    """Drive the real seam: FULL OOM (injected) -> spatial retry -> gpu."""
    import astropy.io.fits as fits
    import cupy as cp

    from seestar.core.gpu import GpuCapabilities
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    real_full = qm.stack_winsorized_sigma_gpu
    real_tiled = qm.stack_winsorized_sigma_gpu_tiled
    orig_mem = cp.cuda.runtime.memGetInfo
    orig_pool = cp.get_default_memory_pool

    tiled_calls = []

    class _FixedPool:
        def free_bytes(self):
            return 0

        def free_all_blocks(self):
            return None

    def boom_full(*a, **k):
        raise _oom(cp)

    def spy_tiled(*a, **k):
        tiled_calls.append(k.get("tile_shape"))
        return real_tiled(*a, **k)

    qm.stack_winsorized_sigma_gpu = boom_full
    qm.stack_winsorized_sigma_gpu_tiled = spy_tiled
    cp.cuda.runtime.memGetInfo = lambda: (2 * 1024 ** 3, 4 * 1024 ** 3)
    cp.get_default_memory_pool = lambda: _FixedPool()
    try:
        o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
        o.update_progress = lambda *a, **k: None
        o.logger = logging.getLogger("zsss.probe")
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
        o.request_gpu = True
        o._acceleration_policy = None
        o._gpu_capabilities = GpuCapabilities(
            gpu_detected=True, cuda_runtime_ready=True, cupy_ready=True,
            opencv_cuda_ready=False, backend_ready=True,
            device_name="MX150", device_vram_mb=2048, compute_capability="6.1",
            failure_reason=None, state="ready",
        )
        hdr = fits.Header()
        batch = [
            (np.full((300, 400), 10.0, np.float32), hdr,
             {"snr": 1.0, "stars": 0.0}, None, np.ones((300, 400), bool))
            for _ in range(4)
        ] + [
            (np.full((300, 400), 1000.0, np.float32), hdr,
             {"snr": 1.0, "stars": 0.0}, None, np.ones((300, 400), bool))
        ]
        V, _hdr, W = o._stack_batch(batch, 1, 1)
        assert tiled_calls, "expected a spatial retry after the injected OOM"
        assert np.isclose(W[0, 0], 5.0, rtol=1e-3)
        ev = o._gpu_execution_events[-1]
        assert ev["executed"] == "gpu" and ev["gpu_memory_mode"] == "tiled"
    finally:
        qm.stack_winsorized_sigma_gpu = real_full
        qm.stack_winsorized_sigma_gpu_tiled = real_tiled
        cp.cuda.runtime.memGetInfo = orig_mem
        cp.get_default_memory_pool = orig_pool


def probe_exhausted_to_cpu():
    import astropy.io.fits as fits
    import cupy as cp

    from seestar.core.gpu import GpuCapabilities
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    real_full = qm.stack_winsorized_sigma_gpu
    real_tiled = qm.stack_winsorized_sigma_gpu_tiled
    orig_mem = cp.cuda.runtime.memGetInfo
    orig_pool = cp.get_default_memory_pool

    class _FixedPool:
        def free_bytes(self):
            return 0

        def free_all_blocks(self):
            return None

    def boom(*a, **k):
        raise _oom(cp)

    qm.stack_winsorized_sigma_gpu = boom
    qm.stack_winsorized_sigma_gpu_tiled = boom
    cp.cuda.runtime.memGetInfo = lambda: (2 * 1024 ** 3, 4 * 1024 ** 3)
    cp.get_default_memory_pool = lambda: _FixedPool()
    try:
        o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
        o.update_progress = lambda *a, **k: None
        o.logger = logging.getLogger("zsss.probe")
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
        o.request_gpu = True
        o._acceleration_policy = None
        o._gpu_capabilities = GpuCapabilities(
            gpu_detected=True, cuda_runtime_ready=True, cupy_ready=True,
            opencv_cuda_ready=False, backend_ready=True,
            device_name="MX150", device_vram_mb=2048, compute_capability="6.1",
            failure_reason=None, state="ready",
        )
        hdr = fits.Header()
        batch = [
            (np.full((300, 400), 10.0, np.float32), hdr,
             {"snr": 1.0, "stars": 0.0}, None, np.ones((300, 400), bool))
            for _ in range(4)
        ] + [
            (np.full((300, 400), 1000.0, np.float32), hdr,
             {"snr": 1.0, "stars": 0.0}, None, np.ones((300, 400), bool))
        ]
        V, _hdr, W = o._stack_batch(batch, 1, 1)
        assert np.isclose(W[0, 0], 5.0, rtol=1e-3)  # CPU produced a result
        ev = o._gpu_execution_events[-1]
        assert ev["executed"] == "cpu"
        assert ev["fallback_reason"] == GPU_EXEC_REASON_GPU_OOM
    finally:
        qm.stack_winsorized_sigma_gpu = real_full
        qm.stack_winsorized_sigma_gpu_tiled = real_tiled
        cp.cuda.runtime.memGetInfo = orig_mem
        cp.get_default_memory_pool = orig_pool


def probe_non_oom_no_retry():
    import astropy.io.fits as fits
    import cupy as cp

    from seestar.core.gpu import GpuCapabilities
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    real_full = qm.stack_winsorized_sigma_gpu
    real_tiled = qm.stack_winsorized_sigma_gpu_tiled
    orig_mem = cp.cuda.runtime.memGetInfo
    orig_pool = cp.get_default_memory_pool
    tiled_calls = []

    class _FixedPool:
        def free_bytes(self):
            return 0

        def free_all_blocks(self):
            return None

    def boom_full(*a, **k):
        raise RuntimeError("not an OOM")

    def spy_tiled(*a, **k):
        tiled_calls.append(a)
        return real_tiled(*a, **k)

    qm.stack_winsorized_sigma_gpu = boom_full
    qm.stack_winsorized_sigma_gpu_tiled = spy_tiled
    cp.cuda.runtime.memGetInfo = lambda: (2 * 1024 ** 3, 4 * 1024 ** 3)
    cp.get_default_memory_pool = lambda: _FixedPool()
    try:
        o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
        o.update_progress = lambda *a, **k: None
        o.logger = logging.getLogger("zsss.probe")
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
        o.request_gpu = True
        o._acceleration_policy = None
        o._gpu_capabilities = GpuCapabilities(
            gpu_detected=True, cuda_runtime_ready=True, cupy_ready=True,
            opencv_cuda_ready=False, backend_ready=True,
            device_name="MX150", device_vram_mb=2048, compute_capability="6.1",
            failure_reason=None, state="ready",
        )
        hdr = fits.Header()
        batch = [
            (np.full((2, 2), 10.0, np.float32), hdr,
             {"snr": 1.0, "stars": 0.0}, None, np.ones((2, 2), bool))
            for _ in range(4)
        ] + [
            (np.full((2, 2), 1000.0, np.float32), hdr,
             {"snr": 1.0, "stars": 0.0}, None, np.ones((2, 2), bool))
        ]
        V, _hdr, W = o._stack_batch(batch, 1, 1)
        assert tiled_calls == []  # NO retry for a non-OOM
        ev = o._gpu_execution_events[-1]
        assert ev["executed"] == "cpu"
        assert ev["fallback_reason"] == GPU_EXEC_REASON_GPU_KERNEL
    finally:
        qm.stack_winsorized_sigma_gpu = real_full
        qm.stack_winsorized_sigma_gpu_tiled = real_tiled
        cp.cuda.runtime.memGetInfo = orig_mem
        cp.get_default_memory_pool = orig_pool


if __name__ == "__main__":
    import cupy as cp

    props = cp.cuda.runtime.getDeviceProperties(0)
    print("device: %s (vram %d MiB)" % (
        props["name"].decode(), props["totalGlobalMem"] // (1024 * 1024)))
    print("WINSOR_GPU_OOM_MAX_ATTEMPTS = %d" % WINSOR_GPU_OOM_MAX_ATTEMPTS)
    check("classifier_oom_vs_kernel", probe_classifier)
    check("full_oom_recovers_tiled", probe_full_oom_recovers_tiled)
    check("exhausted_to_cpu", probe_exhausted_to_cpu)
    check("non_oom_no_retry", probe_non_oom_no_retry)
    if failures:
        print("\n%d FAILURE(S)" % len(failures))
        sys.exit(1)
    print("\nALL PROBES PASSED")
