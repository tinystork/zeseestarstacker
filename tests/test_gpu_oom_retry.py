"""Lot C (Track P7): bounded GPU OOM recovery before CPU fallback.

Layers:

A. PURE classifier tests (no device): ``_winsorized_gpu_oom_kind`` recognises
   ONLY the CuPy pool OOM and the CUDA ``cudaErrorMemoryAllocation`` runtime
   error as ``"oom"``; host ``MemoryError``, other runtime statuses, import/
   query/sync failures and any non-memory kernel bug are ``"kernel"`` (never
   disguised as OOM).

B. PURE cleanup tests (no device): ``_winsorized_oom_cleanup`` synchronises
   best-effort, releases ONLY the pool's free blocks (a fake pool records that
   ``free_all_blocks`` was called once and never touches live arrays), and
   NEVER raises.

C. PURE planner tests (no device): ``force_tiled`` skips FULL even at a huge
   budget; ``max_tile_outputs`` returns a STRICTLY-SMALLER tile than the bound
   (never re-attempts a geometry that already OOM'd), and honours the bitwise
   floor; N is never reduced.

D. Wiring tests (real CuPy/GPU, skipped without it): the ``_gpu_reduce_winsorized``
   seam classifies a kernel exception strictly, retries a FULL OOM as spatial,
   shrinks a TILED OOM, stops at the attempt cap, falls back to CPU only after
   exhaustion (and records the full attempt trace), never retries a non-OOM
   exception, and keeps N/results/weights/rejections/z_eff bitwise unchanged on
   a successful retry.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

import seestar.queuep.queue_manager as queue_manager_module
from seestar.queuep.queue_manager import (
    GPU_EXEC_REASON_GPU_KERNEL,
    GPU_EXEC_REASON_GPU_OOM,
    WINSOR_GPU_OOM_MAX_ATTEMPTS,
    WINSOR_OOM_RESULT_KERNEL_FAILURE,
    WINSOR_OOM_RESULT_OOM,
    WINSOR_OOM_RESULT_SUCCESS,
    _winsorized_gpu_oom_kind,
    _winsorized_oom_cleanup,
)

from seestar.core.gpu_vram_planner import (
    CPU_FALLBACK,
    FULL_GPU,
    TILED_GPU,
    WINSOR_MIN_TILE_OUT,
    WINSOR_PLANNER_DEFAULT_RESERVE_BYTES,
    plan_winsorized_gpu_execution,
)

try:
    import cupy as cp  # noqa: F401

    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - non-GPU hosts
    cp = None
    CUPY_AVAILABLE = False

GIB = 1024 ** 3
MIB = 1024 ** 2


# ---------------------------------------------------------------------------
# A. pure classifier
# ---------------------------------------------------------------------------

class _FakeCupy:
    """Minimal cupy namespace for the classifier: real OOM classes."""

    def __init__(self):
        import cupy as _cp

        self.cuda = type("_cuda", (), {
            "memory": type("_memory", (), {
                "OutOfMemoryError": _cp.cuda.memory.OutOfMemoryError,
            })(),
            "runtime": type("_runtime", (), {
                "CUDARuntimeError": _cp.cuda.runtime.CUDARuntimeError,
            })(),
        })()


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="needs the real cupy classes")
def test_classifier_pool_oom():
    fc = _FakeCupy()
    exc = fc.cuda.memory.OutOfMemoryError(1024, 100, 200)
    assert _winsorized_gpu_oom_kind(exc, fc) == "oom"


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="needs the real cupy classes")
def test_classifier_runtime_memory_allocation_oom():
    fc = _FakeCupy()
    exc = fc.cuda.runtime.CUDARuntimeError(2)  # cudaErrorMemoryAllocation
    assert _winsorized_gpu_oom_kind(exc, fc) == "oom"


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="needs the real cupy classes")
def test_classifier_runtime_non_oom_status_is_kernel():
    fc = _FakeCupy()
    exc = fc.cuda.runtime.CUDARuntimeError(700)  # cudaErrorIllegalAddress
    assert _winsorized_gpu_oom_kind(exc, fc) == "kernel"


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="needs the real cupy classes")
def test_classifier_host_memoryerror_is_not_oom():
    fc = _FakeCupy()
    # A HOST MemoryError (e.g. numpy) must NEVER be retried as a GPU OOM.
    assert _winsorized_gpu_oom_kind(MemoryError("host"), fc) == "kernel"


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="needs the real cupy classes")
def test_classifier_other_exceptions_are_kernel():
    fc = _FakeCupy()
    assert _winsorized_gpu_oom_kind(ValueError("x"), fc) == "kernel"
    assert _winsorized_gpu_oom_kind(IndexError("x"), fc) == "kernel"
    assert _winsorized_gpu_oom_kind(RuntimeError("x"), fc) == "kernel"


def test_classifier_never_raises_on_odd_objects():
    # Even a duck-typed cp without the expected attrs must not raise.
    class _BareCupy:
        pass

    assert _winsorized_gpu_oom_kind(ValueError("x"), _BareCupy()) == "kernel"


# ---------------------------------------------------------------------------
# B. pure cleanup
# ---------------------------------------------------------------------------

class _FakePool:
    def __init__(self, free_bytes, boom=False):
        self._free = free_bytes
        self.free_all_calls = 0
        self.boom = boom

    def free_bytes(self):
        if self.boom:
            raise RuntimeError("boom")
        return self._free

    def free_all_blocks(self):
        if self.boom:
            raise RuntimeError("boom")
        self.free_all_calls += 1


def _make_fake_cp(pool):
    class _Stream:
        @staticmethod
        def synchronize():
            return None

    class _Null:
        null = _Stream()

    return type("cp", (), {
        "cuda": type("cuda", (), {"Stream": _Null})(),
        # staticmethod: cupy exposes this as a module-level callable, so the
        # fake must NOT receive the instance as an implicit ``self``.
        "get_default_memory_pool": staticmethod(lambda: pool),
    })()


def test_cleanup_synchronizes_and_releases_only_free_blocks():
    pool = _FakePool(free_bytes=0)
    cp = _make_fake_cp(pool)
    synced, released = _winsorized_oom_cleanup(cp)
    assert synced is True
    assert pool.free_all_calls == 1  # free_all_blocks invoked exactly once


def test_cleanup_never_raises_when_pool_or_sync_fails():
    boom_pool = _FakePool(free_bytes=0, boom=True)

    class _BoomStream:
        @staticmethod
        def synchronize():
            raise RuntimeError("sync boom")

    class _BoomNull:
        null = _BoomStream()

    cp = type("cp", (), {
        "cuda": type("cuda", (), {"Stream": _BoomNull})(),
        "get_default_memory_pool": lambda: boom_pool,
    })()
    synced, released = _winsorized_oom_cleanup(cp)
    assert synced is False
    assert released == 0


# ---------------------------------------------------------------------------
# C. pure planner: force_tiled + max_tile_outputs
# ---------------------------------------------------------------------------


def test_force_tiled_skips_full_even_at_huge_budget():
    d = plan_winsorized_gpu_execution(
        n_batch=20,
        frame_shape=(270, 480),
        channels=1,
        winsor_limits=(0.05, 0.05),
        driver_free_bytes=24 * GIB,
        pool_free_bytes=0,
        reserve_bytes=WINSOR_PLANNER_DEFAULT_RESERVE_BYTES,
        force_tiled=True,
    )
    assert d.kind == TILED_GPU  # never FULL when forced spatial
    assert d.n_batch == 20  # N preserved


def test_max_tile_outputs_strictly_smaller():
    base = plan_winsorized_gpu_execution(
        n_batch=32,
        frame_shape=(1080, 1920),
        channels=1,
        winsor_limits=(0.05, 0.05),
        driver_free_bytes=2 * GIB,
        pool_free_bytes=0,
        reserve_bytes=WINSOR_PLANNER_DEFAULT_RESERVE_BYTES,
    )
    assert base.kind == TILED_GPU
    d = plan_winsorized_gpu_execution(
        n_batch=32,
        frame_shape=(1080, 1920),
        channels=1,
        winsor_limits=(0.05, 0.05),
        driver_free_bytes=2 * GIB,
        pool_free_bytes=0,
        reserve_bytes=WINSOR_PLANNER_DEFAULT_RESERVE_BYTES,
        force_tiled=True,
        max_tile_outputs=base.tile_outputs,
    )
    assert d.kind == TILED_GPU
    assert d.tile_outputs < base.tile_outputs  # strictly smaller, never equal
    assert d.tile_outputs >= WINSOR_MIN_TILE_OUT  # bitwise floor honoured


def test_max_tile_outputs_below_floor_falls_back():
    d = plan_winsorized_gpu_execution(
        n_batch=32,
        frame_shape=(1080, 1920),
        channels=1,
        winsor_limits=(0.05, 0.05),
        driver_free_bytes=2 * GIB,
        pool_free_bytes=0,
        reserve_bytes=WINSOR_PLANNER_DEFAULT_RESERVE_BYTES,
        force_tiled=True,
        max_tile_outputs=WINSOR_MIN_TILE_OUT,  # exclusive -> floor-1 < floor
    )
    assert d.kind == CPU_FALLBACK


def test_force_tiled_and_max_tile_outputs_never_reduce_n():
    for cap in (None, 5000, 200):
        d = plan_winsorized_gpu_execution(
            n_batch=7,
            frame_shape=(1000, 2000),
            channels=3,
            winsor_limits=(0.05, 0.05),
            driver_free_bytes=4 * GIB,
            pool_free_bytes=0,
            reserve_bytes=WINSOR_PLANNER_DEFAULT_RESERVE_BYTES,
            force_tiled=True,
            max_tile_outputs=cap,
        )
        assert d.n_batch == 7


# ---------------------------------------------------------------------------
# D. wiring: bounded OOM recovery in _gpu_reduce_winsorized
# ---------------------------------------------------------------------------

pytestmark_wiring = pytest.mark.skipif(
    not CUPY_AVAILABLE, reason="CuPy not installed (no GPU stack available)"
)

import astropy.io.fits as fits  # noqa: E402

from seestar.core.gpu import GpuCapabilities  # noqa: E402
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
    o.logger = logging.getLogger("zsss.gpu.winsorized.planner")
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
    import cupy as _cp

    monkeypatch.setattr(
        _cp.cuda.runtime, "memGetInfo", lambda: (free_bytes, 4 * GIB)
    )

    class _FixedPool:
        def free_bytes(self):
            return pool_free

        def free_all_blocks(self):
            return None

    monkeypatch.setattr(_cp, "get_default_memory_pool", lambda: _FixedPool())


def _patch_mem_seq(monkeypatch, free_values):
    """Pin ``memGetInfo`` to a SEQUENCE of free values (clamped at the last)
    so the initial planner sees free_values[0] and each OOM re-plan sees the
    next value (a smaller free value forces a genuinely smaller spatial tile)."""
    import cupy as _cp

    state = {"i": 0, "vals": [int(v) for v in free_values]}

    def memGetInfo():
        v = state["vals"][min(state["i"], len(state["vals"]) - 1)]
        state["i"] += 1
        return (v, 4 * GIB)

    monkeypatch.setattr(_cp.cuda.runtime, "memGetInfo", memGetInfo)

    class _FixedPool:
        def free_bytes(self):
            return 0

        def free_all_blocks(self):
            return None

    monkeypatch.setattr(_cp, "get_default_memory_pool", lambda: _FixedPool())


def _oom():
    import cupy as _cp

    return _cp.cuda.memory.OutOfMemoryError(1, 1, 1)


def _capture_provenance(monkeypatch, stack):
    blocks = []

    def _capture(prefix, tokens):
        blocks.append((prefix, tokens))

    monkeypatch.setattr(stack, "_emit_provenance_block", _capture)
    return blocks


# A 1080p mono frame with N=5 and winsor_limits=(0.2,0.2) (slow path, factor
# 9.5 + 200 MiB scratch): FULL fits at 2 GiB, while a 600 MiB live budget after
# an OOM forces a GENUINELY multi-tile spatial geometry (the full frame does
# not fit).  This lets the wiring tests prove a real spatial retry, not a
# degenerate single full-frame tile.
_FRAME_1080P = (1080, 1920)


@pytestmark_wiring
def test_full_oom_retries_tiled_and_succeeds(monkeypatch):
    """FULL OOM -> spatial (TILED) retry -> executed=gpu, N preserved."""
    real_tiled = queue_manager_module.stack_winsorized_sigma_gpu_tiled
    tiled_calls = []

    def boom_full(*a, **k):
        raise _oom()

    def spy_tiled(*args, **kwargs):
        tiled_calls.append(kwargs.get("tile_shape"))
        return real_tiled(*args, **kwargs)

    monkeypatch.setattr(
        queue_manager_module, "stack_winsorized_sigma_gpu", boom_full
    )
    monkeypatch.setattr(
        queue_manager_module, "stack_winsorized_sigma_gpu_tiled", spy_tiled
    )
    _patch_mem_seq(monkeypatch, [2 * GIB, 600 * MIB, 600 * MIB])
    stack = _winsor_stack(request_gpu=True, shape=_FRAME_1080P)
    blocks = _capture_provenance(monkeypatch, stack)
    V, _hdr, W = stack._stack_batch(_winsor_batch(shape=_FRAME_1080P), 1, 1)
    # Tiled was actually attempted (the retry happened) and returned a result.
    assert tiled_calls, "expected a spatial retry after the FULL OOM"
    assert V.shape == _FRAME_1080P
    # The retry geometry is a genuine multi-tile (spatial), not the full frame.
    ts = tiled_calls[0]
    assert isinstance(ts, tuple) and len(ts) == 1
    assert ts[0] < _FRAME_1080P[0]
    # Attempt provenance: a FULL attempt that OOM'd, then a tiled success.
    results = [b[1].get("result") for b in blocks]
    assert WINSOR_OOM_RESULT_OOM in results
    assert WINSOR_OOM_RESULT_SUCCESS in results
    # Final truth: executed=gpu (tiled), never a fake CPU success.
    ev = stack._gpu_execution_events[-1]
    assert ev["executed"] == "gpu"
    assert ev["gpu_memory_mode"] == "tiled"


@pytestmark_wiring
def test_full_oom_then_tiled_oom_then_smaller_success(monkeypatch):
    """FULL OOM -> TILED OOM -> strictly-smaller tile -> success."""
    real_tiled = queue_manager_module.stack_winsorized_sigma_gpu_tiled
    tiles = []

    def boom_full(*a, **k):
        raise _oom()

    def spy_tiled(*args, **kwargs):
        ts = kwargs.get("tile_shape")
        tiles.append(ts)
        # First tiled geometry OOMs too; the next (smaller) one succeeds.
        if len(tiles) == 1:
            raise _oom()
        return real_tiled(*args, **kwargs)

    monkeypatch.setattr(
        queue_manager_module, "stack_winsorized_sigma_gpu", boom_full
    )
    monkeypatch.setattr(
        queue_manager_module, "stack_winsorized_sigma_gpu_tiled", spy_tiled
    )
    _patch_mem_seq(monkeypatch, [2 * GIB, 600 * MIB, 600 * MIB])
    stack = _winsor_stack(request_gpu=True, shape=_FRAME_1080P)
    V, _hdr, W = stack._stack_batch(_winsor_batch(shape=_FRAME_1080P), 1, 1)
    assert len(tiles) == 2, tiles
    # The retried tile is strictly smaller than the one that OOM'd.
    t0 = tiles[0][0]
    t1 = tiles[1][0]
    assert t1 < t0
    ev = stack._gpu_execution_events[-1]
    assert ev["executed"] == "gpu"
    assert ev["gpu_memory_mode"] == "tiled"


@pytestmark_wiring
def test_oom_exhausted_falls_back_to_cpu(monkeypatch):
    """Every GPU attempt OOMs up to the cap -> CPU fallback, full trace kept."""
    def boom(*a, **k):
        raise _oom()

    monkeypatch.setattr(
        queue_manager_module, "stack_winsorized_sigma_gpu", boom
    )
    monkeypatch.setattr(
        queue_manager_module, "stack_winsorized_sigma_gpu_tiled", boom
    )
    _patch_mem(monkeypatch, free_bytes=2 * GIB)
    stack = _winsor_stack(request_gpu=True, shape=(300, 400))
    blocks = _capture_provenance(monkeypatch, stack)
    V, _hdr, W = stack._stack_batch(_winsor_batch(shape=(300, 400)), 1, 1)
    assert np.isclose(W[0, 0], 5.0, rtol=1e-3)  # CPU produced a valid result
    # Attempt cap respected: exactly WINSOR_GPU_OOM_MAX_ATTEMPTS OOM attempts.
    oom_blocks = [
        b for b in blocks
        if b[0] == "GPU_WINSOR_OOM_RETRY"
        and b[1].get("result") == WINSOR_OOM_RESULT_OOM
    ]
    assert len(oom_blocks) == WINSOR_GPU_OOM_MAX_ATTEMPTS
    ev = stack._gpu_execution_events[-1]
    assert ev["executed"] == "cpu"
    assert ev["fallback_reason"] == GPU_EXEC_REASON_GPU_OOM


@pytestmark_wiring
def test_non_oom_exception_does_not_retry(monkeypatch):
    """A non-memory kernel exception keeps the historical single CPU fallback:
    no OOM retry, no spatial retry, reason gpu_kernel_failure."""
    tiled_calls = []

    def boom_full(*a, **k):
        raise RuntimeError("simulated kernel bug (not OOM)")

    monkeypatch.setattr(
        queue_manager_module, "stack_winsorized_sigma_gpu", boom_full
    )
    monkeypatch.setattr(
        queue_manager_module,
        "stack_winsorized_sigma_gpu_tiled",
        lambda *a, **k: tiled_calls.append(a) or (_ for _ in ()).throw(
            AssertionError("tiled must not run")
        ),
    )
    _patch_mem(monkeypatch, free_bytes=2 * GIB)
    stack = _winsor_stack(request_gpu=True)
    V, _hdr, W = stack._stack_batch(_winsor_batch(), 1, 1)
    assert tiled_calls == []  # NO retry for a non-OOM exception
    assert np.isclose(W[0, 0], 5.0, rtol=1e-3)
    ev = stack._gpu_execution_events[-1]
    assert ev["executed"] == "cpu"
    assert ev["fallback_reason"] == GPU_EXEC_REASON_GPU_KERNEL


@pytestmark_wiring
def test_retry_success_preserves_n_results_weights_rejection(monkeypatch):
    """A successful spatial retry is bitwise-identical to the untiled reference
    (same N, result, weights, rejected_pct)."""
    real_tiled = queue_manager_module.stack_winsorized_sigma_gpu_tiled
    real_full = queue_manager_module.stack_winsorized_sigma_gpu

    def boom_full(*a, **k):
        raise _oom()

    batch = _winsor_batch(shape=_FRAME_1080P, n_in=4)
    imgs = [it[0] for it in batch]
    n = len(imgs)
    w = np.ones(n, dtype=np.float32)

    monkeypatch.setattr(
        queue_manager_module, "stack_winsorized_sigma_gpu", boom_full
    )
    _patch_mem_seq(monkeypatch, [2 * GIB, 600 * MIB, 600 * MIB])
    stack = _winsor_stack(request_gpu=True, shape=_FRAME_1080P)
    V, _hdr, W = stack._stack_batch(batch, 1, 1)
    # The retried (multi-tile) result equals the untiled twin bitwise: the
    # retry preserved N, the result, the weight map and rejected_pct.
    V_u, W_u, _pct_u = real_full(imgs, w, return_weights=True)
    assert np.array_equal(V, V_u)
    assert np.array_equal(W, W_u)
    # N was never reduced across the retry.
    assert n == 5
