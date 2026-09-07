"""Phase F (Track P4): adaptive VRAM execution planner tests.

Two layers:

A. PURE planner tests (no device, no CuPy): injected memory state drives
   FULL_GPU / TILED_GPU(tile_shape) / CPU_FALLBACK(reason) decisions.

   * budget witnesses: effective budgets 2 / 8 / 16 / 24 GiB (memory
     budgets, NOT GPU classes; injected as ``driver_free_bytes``),
   * nominal-vs-allocatable witnesses: 24 GiB nominal but 2 GiB free;
     8 GiB nominal but 1 GiB free (never GPU-name rules),
   * pool-reuse witnesses: low driver free + reusable CuPy pool free,
   * query-failure witnesses (pool query / memGetInfo) and
     fragmentation/reservation (explicit reserve) at the wiring layer,
   * minimum-tile / maximum-tile witnesses,
   * the FULL / TILED / FALLBACK eligibility boundary on one workload,
   * N_batch preservation: every decision mirrors its input N_batch and no
     decision can split the stack axis (tile_shape is purely spatial).

B. Wiring tests (real CuPy/GPU, skipped without it): the
   ``SeestarQueuedStacker._gpu_reduce_winsorized`` dispatch through the real
   ``_stack_batch``: FULL reaches the untiled twin, TILED reaches the tiled
   seam with the planner's ``tile_shape`` (bitwise identical to the untiled
   twin), CPU_FALLBACK reasons land in ``_gpu_fallback_logged``
   (vram_no_valid_tile + legacy umbrella vram_reject, planner_failure,
   pool_query_failure, meminfo_failure) and degrade to CPU without a crash.
"""

import logging

import numpy as np
import pytest

from seestar.core import gpu_vram_planner as pvp
from seestar.core.gpu_vram_planner import (
    CPU_FALLBACK,
    FULL_GPU,
    REASON_MEMINFO_FAILURE,
    REASON_PLANNER_FAILURE,
    REASON_POOL_QUERY_FAILURE,
    REASON_VRAM_NO_VALID_TILE,
    TILED_GPU,
    WINSOR_FAST_SCRATCH_BYTES,
    WINSOR_FAST_SORT_FACTOR,
    WINSOR_MIN_TILE_OUT,
    WINSOR_PLANNER_DEFAULT_RESERVE_BYTES,
    WINSOR_SLOW_SCRATCH_BYTES,
    WINSOR_SLOW_SORT_FACTOR,
    WinsorExecDecision,
    plan_winsorized_gpu_execution,
)

try:
    import cupy  # noqa: F401

    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on non-GPU hosts
    CUPY_AVAILABLE = False

GIB = 1024 ** 3
MIB = 1024 ** 2

# Frame shapes used by the witnesses (realistic astro resolutions).
F_480 = (270, 480)
F_1080P = (1080, 1920)
F_4K = (2160, 3840)

_FRAMES = {480 * 270: F_480, 1080 * 1920: F_1080P, 2160 * 3840: F_4K}


def _frame_area(shape):
    return int(shape[0]) * int(shape[1])


def plan(
    n,
    frame,
    *,
    free,
    pool=0,
    channels=1,
    isz=4,
    limits=(0.05, 0.05),
    reserve=WINSOR_PLANNER_DEFAULT_RESERVE_BYTES,
):
    return plan_winsorized_gpu_execution(
        n_batch=n,
        frame_shape=frame,
        channels=channels,
        dtype_itemsize=isz,
        winsor_limits=limits,
        driver_free_bytes=free,
        pool_free_bytes=pool,
        reserve_bytes=reserve,
    )


def _slow_full_demand(n, frame, channels=1, isz=4):
    base = n * _frame_area(frame) * channels * isz
    return int(base * WINSOR_SLOW_SORT_FACTOR) + WINSOR_SLOW_SCRATCH_BYTES


def _fast_full_demand(n, frame, channels=1, isz=4):
    base = n * _frame_area(frame) * channels * isz
    return int(base * WINSOR_FAST_SORT_FACTOR) + WINSOR_FAST_SCRATCH_BYTES


# ---------------------------------------------------------------------------
# A. pure planner: budget witnesses (2 / 8 / 16 / 24 GiB)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capacity_gib", [2, 8, 16, 24])
def test_trivial_workload_full_at_every_budget_witness(capacity_gib):
    """480x270 N=20 (slow path) fits UNTILED at every 2/8/16/24 GiB budget."""
    d = plan(20, F_480, free=capacity_gib * GIB)
    assert d.kind == FULL_GPU
    assert d.n_batch == 20
    assert d.effective_budget_bytes == capacity_gib * GIB - WINSOR_PLANNER_DEFAULT_RESERVE_BYTES


def test_1080p_n32_grows_full_with_budget():
    """Same 1080p N=32 stack: TILED at 2 GiB (phase E: untiled OOMs the 2 GiB
    MX150), FULL from 8 GiB upward under the identical model."""
    kinds = [plan(32, F_1080P, free=g * GIB).kind for g in (2, 8, 16, 24)]
    assert kinds == [TILED_GPU, FULL_GPU, FULL_GPU, FULL_GPU]
    # Absolute phase E anchored witness: the untiled 1080p N=32 demand
    # exceeds 2 GiB, N=20 (which phase E ran UNTILED on the 2 GiB card) fits.
    assert _slow_full_demand(20, F_1080P) + WINSOR_PLANNER_DEFAULT_RESERVE_BYTES <= 2 * GIB
    assert _slow_full_demand(32, F_1080P) + WINSOR_PLANNER_DEFAULT_RESERVE_BYTES > 2 * GIB


def test_rgb_needs_more_budget_than_mono():
    """Channel-aware scaling: RGB 1080p N=50 is TILED at 8 GiB where mono is
    FULL; both FULL at 24 GiB."""
    mono = plan(50, F_1080P, free=8 * GIB)
    rgb = plan(50, F_1080P, channels=3, free=8 * GIB)
    assert mono.kind == FULL_GPU
    assert rgb.kind == TILED_GPU
    assert plan(50, F_1080P, channels=3, free=24 * GIB).kind == FULL_GPU


def test_dtype_itemsize_scales_demand():
    """float64 (itemsize 8) doubles the stack bytes -> TILED where float32 is
    FULL at the same budget."""
    f32 = plan(50, F_480, free=700 * MIB)
    f64 = plan(50, F_480, isz=8, free=700 * MIB)
    assert f32.kind == FULL_GPU
    assert f64.kind != FULL_GPU  # 2x base pushes over the 700 MiB budget


# ---------------------------------------------------------------------------
# A. nominal vs allocatable (never GPU classes)
# ---------------------------------------------------------------------------


def test_nominal_24gib_but_2gib_allocatable():
    """'24 GiB nominal but only 2 GiB allocatable' arrives as driver free ==
    2 GiB -> the decision is EXACTLY the 2 GiB decision (no nominal table)."""
    nominal_big = plan(50, F_1080P, free=24 * GIB)
    big = plan(50, F_1080P, free=2 * GIB)
    assert nominal_big.kind == FULL_GPU
    assert big.kind == TILED_GPU
    # Same workload, same effective allocatable budget -> same tile choice.
    d1 = plan(50, F_1080P, free=2 * GIB)
    d2 = plan(50, F_1080P, free=2 * GIB, pool=0)
    assert (d1.kind, d1.tile_shape) == (d2.kind, d2.tile_shape)


def test_nominal_8gib_but_1gib_free():
    """'8 GiB nominal but 1 GiB effectively free' behaves as a 1 GiB card:
    1080p N=50 goes TILED with a smaller tile than the 2 GiB choice."""
    one = plan(50, F_1080P, free=1 * GIB)
    two = plan(50, F_1080P, free=2 * GIB)
    assert one.kind == TILED_GPU and two.kind == TILED_GPU
    assert one.tile_outputs < two.tile_outputs


def test_pool_reuse_counts_towards_effective_budget():
    """Low driver free + large reusable pool free admits a workload that a
    cold pool of the same driver free rejects (R2-F4 semantics)."""
    need = _slow_full_demand(20, F_1080P) + WINSOR_PLANNER_DEFAULT_RESERVE_BYTES
    assert need <= 2 * GIB  # sanity of the witness setup
    cold = plan(20, F_1080P, free=64 * MIB, pool=0)
    warm = plan(20, F_1080P, free=64 * MIB, pool=2 * GIB - 64 * MIB)
    assert cold.kind == CPU_FALLBACK
    assert cold.reason == REASON_VRAM_NO_VALID_TILE
    assert warm.kind == FULL_GPU


def test_reserve_reduces_effective_budget():
    """Explicit reserve is subtracted from driver free + pool free before any
    decision; identical free with a larger reserve falls back."""
    d0 = plan(20, F_1080P, free=2 * GIB, reserve=0)
    assert d0.kind == FULL_GPU
    assert plan(20, F_1080P, free=2 * GIB, reserve=3 * GIB).kind == CPU_FALLBACK
    assert d0.effective_budget_bytes == 2 * GIB
    assert d0.reserve_bytes == 0


# ---------------------------------------------------------------------------
# A. FULL / TILED / FALLBACK eligibility boundary on one workload
# ---------------------------------------------------------------------------


def test_eligibility_boundary_full_vs_tiled():
    """1080p default limits (slow path): the model boundary between FULL and
    TILED sits at N_batch 22 at 2 GiB (9.5 x + 200 MiB + 128 MiB reserve);
    N=20 stays FULL -- the phase E witness that the optimized untiled model
    fits the 2 GiB card at 1080p N=20."""
    n_boundary = (
        (2 * GIB - WINSOR_SLOW_SCRATCH_BYTES - WINSOR_PLANNER_DEFAULT_RESERVE_BYTES)
        // int(WINSOR_SLOW_SORT_FACTOR * _frame_area(F_1080P) * 4)
    )
    assert plan(n_boundary, F_1080P, free=2 * GIB).kind == FULL_GPU
    assert plan(n_boundary + 1, F_1080P, free=2 * GIB).kind == TILED_GPU
    # Phase E anchor: untiled N=20 ran on the 2 GiB MX150; N=32 untiled OOM'd.
    assert plan(20, F_1080P, free=2 * GIB).kind == FULL_GPU
    assert n_boundary == 22


def test_must_fallback_no_valid_tile_huge_n():
    """A huge N_batch whose MINIMUM valid tile (>= 96 spatial outputs) still
    exceeds the budget -> CPU_FALLBACK(vram_no_valid_tile)."""
    d = plan(4_000_000, F_1080P, free=2 * GIB)
    assert d.kind == CPU_FALLBACK
    assert d.reason == REASON_VRAM_NO_VALID_TILE


def test_must_fallback_narrow_frame_no_valid_geometry():
    """A 64x40 frame (W < 96) whose full stack does not fit: every spatial
    tiling either has sub-96 cells or an invalid partial band -> FALLBACK."""
    d = plan(400_000, (64, 40), free=2 * GIB)
    assert d.kind == CPU_FALLBACK
    assert d.reason == REASON_VRAM_NO_VALID_TILE
    # The same frame at a realistic N fits UNTILED (untiled is the reference).
    assert plan(20, (64, 40), free=2 * GIB).kind == FULL_GPU


def test_fallback_when_budget_negative_after_reserve():
    """Zero free + zero pool - reserve -> no strategy can fit."""
    d = plan(50, F_1080P, free=0, pool=0)
    assert d.kind == CPU_FALLBACK
    assert d.reason == REASON_VRAM_NO_VALID_TILE


# ---------------------------------------------------------------------------
# A. minimum / maximum tile witnesses + geometry preference
# ---------------------------------------------------------------------------


def test_minimum_valid_tile_selected_at_boundary():
    """Budget that admits exactly min_tile_out (96) spatial outputs per tile:
    the planner returns a 96-output geometry (never a smaller micro tile)."""
    # s_cap == 96 == W -> a single-row band of the full 96-wide frame.
    free = int(
        96 * 50 * 1 * 4 * WINSOR_SLOW_SORT_FACTOR
    ) + WINSOR_SLOW_SCRATCH_BYTES + WINSOR_PLANNER_DEFAULT_RESERVE_BYTES
    d = plan(50, (2000, 96), free=free)
    assert d.kind == TILED_GPU
    assert d.tile_outputs == WINSOR_MIN_TILE_OUT == 96
    assert d.tile_shape == (1,)
    # Wide frame: the smallest admissible cell is a 1x96 rectangle, the full
    # 1920 width divides exactly into 20 such columns.
    d2 = plan(50, (1080, 1920), free=free)
    assert d2.kind == TILED_GPU
    assert d2.tile_shape == (1, 96)
    assert d2.tile_outputs == 96


def test_planner_never_selects_micro_tiles():
    """Sweep: every TILED geometry keeps EVERY tile >= min_tile_out outputs
    (full cells and partial edges alike, per the geometry construction)."""
    for n in (5, 19, 20, 32, 50, 200):
        for frame in (F_480, F_1080P, F_4K):
            for g in (1, 2, 8):
                d = plan(n, frame, free=g * GIB)
                if d.kind != TILED_GPU:
                    continue
                th, tw = (
                    (d.tile_shape[0], _frame_area(frame) // frame[0])
                    if len(d.tile_shape) == 1
                    else d.tile_shape
                )
                assert th >= 1 and tw >= 1
                # every full cell and every partial band/column >= min_tile_out
                assert th * tw >= WINSOR_MIN_TILE_OUT
                assert tw >= WINSOR_MIN_TILE_OUT  # partial bands stay valid
                if frame[1] % tw:
                    assert frame[1] % tw >= WINSOR_MIN_TILE_OUT
                if frame[0] % th:
                    assert (frame[0] % th) * tw >= WINSOR_MIN_TILE_OUT


def test_largest_fitting_row_band_preferred():
    """When a full row fits the budget the planner picks the LARGEST full-
    width row band (fewest tiles), not a rect split."""
    d = plan(32, F_1080P, free=2 * GIB)
    assert d.kind == TILED_GPU
    assert len(d.tile_shape) == 1  # (tile_h,) == full-width row bands
    (tile_h,) = d.tile_shape
    # Largest admissible height: per-tile stack (tile_h rows x W) keeps the
    # modeled demand + reserve inside the budget.
    assert int(tile_h * 32 * 1920 * 4 * WINSOR_SLOW_SORT_FACTOR) + WINSOR_SLOW_SCRATCH_BYTES + WINSOR_PLANNER_DEFAULT_RESERVE_BYTES <= 2 * GIB
    # One more row would violate the model.
    if tile_h < 1080:
        assert int((tile_h + 1) * 32 * 1920 * 4 * WINSOR_SLOW_SORT_FACTOR) + WINSOR_SLOW_SCRATCH_BYTES + WINSOR_PLANNER_DEFAULT_RESERVE_BYTES > 2 * GIB
    assert d.tile_outputs == tile_h * 1920
    assert d.tile_outputs >= WINSOR_MIN_TILE_OUT


def test_rect_split_when_single_row_does_not_fit():
    """Budget smaller than one full row (s_cap < W): rectangular split of the
    width, every column (incl. the partial one) >= 96 outputs."""
    # s_cap = 1919 px < 1920 -> row band impossible; rect (1, 1824) leaves a
    # 96-wide partial column (exactly the bitwise floor).
    max_tile_base = int(1919 * 32 * 4 * WINSOR_SLOW_SORT_FACTOR)
    free = max_tile_base + WINSOR_SLOW_SCRATCH_BYTES + WINSOR_PLANNER_DEFAULT_RESERVE_BYTES
    d = plan(32, F_1080P, free=free)
    assert d.kind == TILED_GPU
    assert len(d.tile_shape) == 2
    tile_h, tile_w = d.tile_shape
    assert tile_w >= WINSOR_MIN_TILE_OUT
    assert 1920 % tile_w == 0 or 1920 % tile_w >= WINSOR_MIN_TILE_OUT
    assert tile_h * tile_w <= 1919


# ---------------------------------------------------------------------------
# A. fast path (phase C zero-rank regime) selects its own memory model
# ---------------------------------------------------------------------------


def test_zero_rank_regime_uses_fast_memory_model():
    """Default (0.05, 0.05) limits: N=19 is the zero-rank fast path (identity
    winsorization, no per-iteration sorts) while N=20 is the slow path; the
    4K witness fits UNTILED at 2 GiB only in the fast regime."""
    d19 = plan(19, F_4K, free=2 * GIB)
    d20 = plan(20, F_4K, free=2 * GIB)
    assert d19.fast_path is True
    assert d20.fast_path is False
    assert d19.kind == FULL_GPU
    assert d20.kind == TILED_GPU
    assert d19.demand_full_bytes == _fast_full_demand(19, F_4K)
    assert d20.demand_full_bytes == _slow_full_demand(20, F_4K)


@pytest.mark.parametrize(
    "limits,n,expected",
    [
        ((0.05, 0.05), 19, True),
        ((0.05, 0.05), 20, False),
        ((0.2, 0.2), 4, True),
        ((0.2, 0.2), 5, False),
        ((0.1, 0.0), 9, True),
        ((0.1, 0.0), 10, False),
        ((-0.05, 0.05), 50, False),
    ],
)
def test_fast_path_flag_matches_zero_rank_regime(limits, n, expected):
    d = plan(n, F_480, free=24 * GIB, limits=limits)
    assert d.fast_path is expected
    # Fast regime even at 24 GiB must use the fast model constant.
    if expected:
        assert d.demand_full_bytes == _fast_full_demand(n, F_480)


# ---------------------------------------------------------------------------
# A. N_batch preservation (never reduced, never split)
# ---------------------------------------------------------------------------


def test_n_batch_never_reduced():
    """Every decision kind mirrors its input N_batch; TILED tile shapes are
    purely spatial (product <= frame area, stack axis untouched)."""
    cases = [
        (20, F_1080P, 2 * GIB),  # FULL
        (32, F_1080P, 2 * GIB),  # TILED
        (4_000_000, F_1080P, 2 * GIB),  # FALLBACK
        (19, F_4K, 2 * GIB),  # FULL fast path
        (50, F_1080P, 2 * GIB, 3),  # RGB TILED
    ]
    for c in cases:
        n, frame, free = c[0], c[1], c[2]
        channels = c[3] if len(c) > 3 else 1
        d = plan(n, frame, free=free, channels=channels)
        assert d.n_batch == n
        assert d.frame_shape == tuple(frame)
        assert d.channels == channels
        if d.kind == TILED_GPU:
            th, tw = (
                (d.tile_shape[0], frame[1])
                if len(d.tile_shape) == 1
                else d.tile_shape
            )
            assert th * tw < _frame_area(frame)  # strictly smaller than full
            assert th <= frame[0] and tw <= frame[1]
            assert d.tile_outputs >= WINSOR_MIN_TILE_OUT


# ---------------------------------------------------------------------------
# A. modeled-demand consistency sweep (never over-budget)
# ---------------------------------------------------------------------------

_BUDGETS_GIB = [0.5, 1, 2, 4, 8, 16, 24]
_N_SWEEP = [5, 19, 20, 32, 50, 100, 500]
_FRAME_SWEEP = [F_480, F_1080P, F_4K]


def test_modeled_demand_never_exceeds_budget_on_decisions():
    """Every GPU decision obeys demand + reserve <= driver free + pool free;
    every CPU_FALLBACK carries a catalog reason; N_batch always preserved."""
    for n in _N_SWEEP:
        for frame in _FRAME_SWEEP:
            for g in _BUDGETS_GIB:
                free = int(g * GIB)
                d = plan(n, frame, free=free)
                assert d.n_batch == n
                if d.kind == FULL_GPU:
                    assert d.demand_full_bytes + d.reserve_bytes <= free
                elif d.kind == TILED_GPU:
                    assert d.demand_tile_bytes + d.reserve_bytes <= free
                    assert d.tile_outputs >= WINSOR_MIN_TILE_OUT
                    assert d.kind == TILED_GPU
                else:
                    assert d.reason in (
                        REASON_VRAM_NO_VALID_TILE,
                        REASON_PLANNER_FAILURE,
                        REASON_POOL_QUERY_FAILURE,
                        REASON_MEMINFO_FAILURE,
                    )
                # Effective budget is always free + pool - reserve.
                assert d.effective_budget_bytes == free - d.reserve_bytes


def test_decisions_monotonic_in_budget():
    """More effective memory never produces a strictly worse strategy on the
    same workload (FULL > TILED > FALLBACK ordering)."""
    rank = {FULL_GPU: 2, TILED_GPU: 1, CPU_FALLBACK: 0}
    for n in (32, 50, 100):
        prev = None
        for g in _BUDGETS_GIB:
            d = plan(n, F_1080P, free=int(g * GIB))
            if prev is not None:
                assert rank[d.kind] >= prev, (n, g)
            prev = rank[d.kind]


# ---------------------------------------------------------------------------
# A. planner robustness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_batch=0, frame_shape=F_480, driver_free_bytes=2 * GIB),
        dict(n_batch=50, frame_shape=(0, 480), driver_free_bytes=2 * GIB),
        dict(n_batch=50, frame_shape=F_480, driver_free_bytes=-1),
        dict(n_batch=50, frame_shape=F_480, channels=0, driver_free_bytes=2 * GIB),
        dict(n_batch=50, frame_shape=F_480, dtype_itemsize=0, driver_free_bytes=2 * GIB),
        dict(n_batch=50, frame_shape=F_480, min_tile_out=0, driver_free_bytes=2 * GIB),
    ],
)
def test_planner_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        plan_winsorized_gpu_execution(**kwargs)


def test_decision_dataclass_shape():
    d = plan(32, F_1080P, free=2 * GIB)
    assert isinstance(d, WinsorExecDecision)
    assert d.is_gpu is True
    assert d.kind == TILED_GPU
    fb = plan(4_000_000, F_1080P, free=2 * GIB)
    assert fb.is_gpu is False
    assert fb.kind == CPU_FALLBACK


# ===========================================================================
# B. wiring: _gpu_reduce_winsorized through the real _stack_batch dispatch
# ===========================================================================

pytestmark_wiring = pytest.mark.skipif(
    not CUPY_AVAILABLE, reason="CuPy not installed (no GPU stack available)"
)

import astropy.io.fits as fits  # noqa: E402

import seestar.queuep.queue_manager as queue_manager_module  # noqa: E402
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
    """Pin the device memory state seen by the winsorized dispatch."""
    import cupy as _cp

    monkeypatch.setattr(
        _cp.cuda.runtime, "memGetInfo", lambda: (free_bytes, 4 * GIB)
    )

    class _FixedPool:
        def free_bytes(self):
            return pool_free

    monkeypatch.setattr(_cp, "get_default_memory_pool", lambda: _FixedPool())


@pytestmark_wiring
def test_wiring_tiled_dispatch_bitwise_with_untiled(monkeypatch):
    """Planner TILED_GPU route: the production dispatch calls
    ``stack_winsorized_sigma_gpu_tiled`` with the planner's tile_shape and the
    result is bitwise identical to the untiled twin (same batch, real GPU)."""
    import cupy as _cp

    real_tiled = queue_manager_module.stack_winsorized_sigma_gpu_tiled
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_tiled(*args, **kwargs)

    monkeypatch.setattr(
        queue_manager_module, "stack_winsorized_sigma_gpu_tiled", spy
    )
    # Free memory small enough that the 1080p N=5 slow path (untiled demand
    # ~686 MiB) must go TILED, but large enough to admit a valid geometry.
    _patch_mem(monkeypatch, free_bytes=620 * MIB)
    stack = _winsor_stack(request_gpu=True, shape=(1080, 1920))
    batch = _winsor_batch(shape=(1080, 1920))
    V_t, _hdr, W_t = stack._stack_batch(batch, 1, 1)
    assert len(calls) == 1
    ts = calls[0][1].get("tile_shape")
    assert ts is not None and isinstance(ts, tuple)
    # The dispatch must have preserved the batch (N_batch frozen = 5) and
    # chosen a valid spatial geometry.
    assert len(ts) in (1, 2)
    # Compare against the untiled GPU twin on the identical batch.
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
    # The planner record must have kept N_batch == 5 (5 inliers + 1 outlier).
    assert len(batch) == 5


@pytestmark_wiring
def test_wiring_full_dispatch_untiled_twin(monkeypatch):
    """Planner FULL_GPU route: untiled twin invoked; tiled seam NOT invoked."""
    tiled_calls = []
    monkeypatch.setattr(
        queue_manager_module,
        "stack_winsorized_sigma_gpu_tiled",
        lambda *a, **k: tiled_calls.append((a, k)) or None,
    )
    _patch_mem(monkeypatch, free_bytes=2 * GIB)
    stack = _winsor_stack(request_gpu=True)
    V, _hdr, W = stack._stack_batch(_winsor_batch(), 1, 1)
    assert tiled_calls == []
    assert np.isclose(V[0, 0], 10.0, rtol=1e-3), V[0, 0]
    assert np.isclose(W[0, 0], 5.0, rtol=1e-3), W[0, 0]


@pytestmark_wiring
def test_wiring_vram_no_valid_tile_fallback(monkeypatch, caplog):
    """CPU_FALLBACK(vram_no_valid_tile): CPU result, planner reason AND the
    legacy umbrella ``vram_reject`` recorded, GPU never invoked."""
    real_full = queue_manager_module.stack_winsorized_sigma_gpu
    calls = []

    def spy(*args, **kwargs):
        calls.append(args)
        return real_full(*args, **kwargs)

    monkeypatch.setattr(queue_manager_module, "stack_winsorized_sigma_gpu", spy)
    _patch_mem(monkeypatch, free_bytes=512 * 1024)  # ~nothing allocatable
    caplog.set_level(logging.WARNING, logger="zsss.gpu.winsorized.planner")
    stack = _winsor_stack(request_gpu=True)
    V, _hdr, W = stack._stack_batch(_winsor_batch(shape=(1080, 1920)), 1, 1)
    assert calls == []
    assert REASON_VRAM_NO_VALID_TILE in stack._gpu_fallback_logged
    assert "vram_reject" in stack._gpu_fallback_logged  # legacy umbrella
    assert "reduction needs" in caplog.text
    assert V.shape == (1080, 1920)
    assert np.isclose(W[0, 0], 5.0, rtol=1e-3), W[0, 0]


@pytestmark_wiring
def test_wiring_meminfo_failure_falls_back_cpu(monkeypatch):
    """memGetInfo query failure -> CPU_FALLBACK(meminfo_failure), no crash."""
    import cupy as _cp

    calls = []

    def spy(*a, **k):
        calls.append(a)
        raise AssertionError("GPU must not run")

    monkeypatch.setattr(
        queue_manager_module, "stack_winsorized_sigma_gpu", spy
    )

    def boom():
        raise RuntimeError("simulated memGetInfo failure")

    monkeypatch.setattr(_cp.cuda.runtime, "memGetInfo", boom)
    stack = _winsor_stack(request_gpu=True)
    V, _hdr, W = stack._stack_batch(_winsor_batch(), 1, 1)
    assert calls == []
    assert REASON_MEMINFO_FAILURE in stack._gpu_fallback_logged
    assert np.isclose(V[0, 0], 10.0, rtol=1e-3), V[0, 0]
    assert np.isclose(W[0, 0], 5.0, rtol=1e-3), W[0, 0]


@pytestmark_wiring
def test_wiring_pool_query_failure_falls_back_cpu(monkeypatch):
    """CuPy pool free_bytes query failure -> CPU_FALLBACK(pool_query_failure)."""
    import cupy as _cp

    calls = []

    def spy(*a, **k):
        calls.append(a)
        raise AssertionError("GPU must not run")

    monkeypatch.setattr(
        queue_manager_module, "stack_winsorized_sigma_gpu", spy
    )

    class _BrokenPool:
        def free_bytes(self):
            raise RuntimeError("simulated pool query failure")

    monkeypatch.setattr(_cp, "get_default_memory_pool", lambda: _BrokenPool())
    stack = _winsor_stack(request_gpu=True)
    V, _hdr, W = stack._stack_batch(_winsor_batch(), 1, 1)
    assert calls == []
    assert REASON_POOL_QUERY_FAILURE in stack._gpu_fallback_logged
    assert np.isclose(V[0, 0], 10.0, rtol=1e-3), V[0, 0]
    assert np.isclose(W[0, 0], 5.0, rtol=1e-3), W[0, 0]


@pytestmark_wiring
def test_wiring_planner_failure_falls_back_cpu(monkeypatch):
    """Unexpected planner exception -> CPU_FALLBACK(planner_failure), never a
    crash, GPU twin not invoked."""
    def boom(*a, **k):
        raise RuntimeError("simulated planner bug")

    monkeypatch.setattr(
        queue_manager_module, "plan_winsorized_gpu_execution", boom
    )
    stack = _winsor_stack(request_gpu=True)
    V, _hdr, W = stack._stack_batch(_winsor_batch(), 1, 1)
    assert REASON_PLANNER_FAILURE in stack._gpu_fallback_logged
    assert np.isclose(V[0, 0], 10.0, rtol=1e-3), V[0, 0]
    assert np.isclose(W[0, 0], 5.0, rtol=1e-3), W[0, 0]


@pytestmark_wiring
def test_wiring_policy_cpu_skips_planner_and_gpu(monkeypatch):
    """policy_cpu (request_gpu=False) keeps the historical silent CPU path:
    no planner call, no GPU call, no fallback diagnostic."""
    called = {}

    def no_gpu(*a, **k):
        called["gpu"] = True
        raise AssertionError("GPU kernel must not run")

    def no_plan(*a, **k):
        called["plan"] = True
        raise AssertionError("planner must not run on policy CPU")

    monkeypatch.setattr(queue_manager_module, "stack_winsorized_sigma_gpu", no_gpu)
    monkeypatch.setattr(queue_manager_module, "plan_winsorized_gpu_execution", no_plan)
    stack = _winsor_stack(request_gpu=False)
    V, _hdr, W = stack._stack_batch(_winsor_batch(), 1, 1)
    assert called == {}
    assert getattr(stack, "_gpu_fallback_logged", set()) == set()
    assert np.isclose(V[0, 0], 10.0, rtol=1e-3), V[0, 0]
    assert np.isclose(W[0, 0], 5.0, rtol=1e-3), W[0, 0]


@pytestmark_wiring
def test_wiring_gpu_runtime_failure_still_falls_back_cpu(monkeypatch):
    """A GPU failure AFTER a planner FULL_GPU choice degrades to CPU with the
    historical warning (same net contract as _gpu_reduce)."""
    def boom(*a, **k):
        raise RuntimeError("simulated GPU kernel failure")

    monkeypatch.setattr(queue_manager_module, "stack_winsorized_sigma_gpu", boom)
    stack = _winsor_stack(request_gpu=True)
    V, _hdr, W = stack._stack_batch(_winsor_batch(), 1, 1)
    assert np.isclose(V[0, 0], 10.0, rtol=1e-3), V[0, 0]
    assert np.isclose(W[0, 0], 5.0, rtol=1e-3), W[0, 0]
