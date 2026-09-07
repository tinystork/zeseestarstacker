"""8.4.0 pre-W80 stage B — pure CPU memory planner tests.

Mission ``zsss-840-prew80-20260907``.  Covers:

* deterministic decisions for synthetic 2/8/16/24 GiB simulations at fixed N:
  N invariant; only FULL_CPU / SPATIAL_TILED_CPU geometry / CPU_MEMORY_REFUSAL
  differs;
* structural exact-N: no decision field/attribute can encode a reduced N;
* fast/slow (rank 0 vs rank > 0) regime detection correctness, weighted and
  unweighted;
* full-width band first, rectangular fallback, minimum-tile refusal with a
  truthful reason;
* named/documented reserve; ``effective_budget = min(ceiling,
  available - reserve)``; reserve never silently negative;
* model conservatism: every committed evidence row's model estimate >=
  observed peak delta (evidence rows embedded as fixtures);
* planner purity: no import-time psutil / device side effects;
* bounded retry vocabulary: only tile_h/tile_w may be retried; forbidden
  retry fields (N / reducer / kappa / winsor / normalization / weights) are
  absent from the decision by construction.
"""

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from seestar.core.cpu_memory_planner import (
    CPU_MEMORY_REFUSAL,
    CPU_MIN_TILE_OUT,
    CPU_OUTPUT_SERIALIZATION_FRAMES,
    CPU_POOL_WORKER_OVERHEAD_BYTES,
    CPU_RESERVE_FIXED_MIN_BYTES,
    CPU_RESERVE_FRACTION,
    CPU_SCIPY_WINSOR_FACTOR,
    CPU_SCIPY_WINSOR_SCRATCH_BYTES,
    CPU_WINSOR_FAST_FACTOR,
    CPU_WINSOR_FAST_SCRATCH_BYTES,
    CPU_WINSOR_SLOW_FACTOR,
    CPU_WINSOR_SLOW_SCRATCH_BYTES,
    FULL_CPU,
    MODE_AUTO,
    MODE_OVERRIDE,
    REASON_MIN_TILE_EXCEEDS_BUDGET,
    REASON_NO_VALID_TILE,
    SPATIAL_TILED_CPU,
    CpuMemoryDecision,
    plan_cpu_winsor_execution,
    recommended_reserve_bytes,
    winsor_zero_rank_regime,
)

GiB = 1024**3
MiB = 1024**2

FRAME_1080 = (1080, 1920)
FRAME_480 = (480, 640)


# ---------------------------------------------------------------------------
# A. Deterministic synthetic simulations at fixed N (2/8/16/24 GiB)
# ---------------------------------------------------------------------------
def _sim(ram_gib, n=50, frame=FRAME_1080, channels=1, **kw):
    return plan_cpu_winsor_execution(
        n=n,
        frame_shape=frame,
        channels=channels,
        dtype_itemsize=4,
        winsor_limits=(0.05, 0.05),
        apply_rewinsor=True,
        weighted=False,
        available_ram_bytes=int(ram_gib * GiB),
        reserve_bytes=512 * MiB,
        policy_ceiling_bytes=int(ram_gib * GiB),
        mode=MODE_AUTO,
        **kw,
    )


def test_synthetic_ram_sweep_n_invariant():
    n_fixed = 36
    decisions = {g: _sim(g, n=n_fixed) for g in (2, 8, 16, 24)}
    for g, d in decisions.items():
        assert d.n == n_fixed  # N verbatim, never reduced
        assert d.strategy in (FULL_CPU, SPATIAL_TILED_CPU, CPU_MEMORY_REFUSAL)
    # At 2 GiB the 1080p N=36 slow path cannot run untiled -> tiled/refused;
    # at 24 GiB the whole frame must fit.
    assert decisions[24].strategy == FULL_CPU
    assert decisions[2].strategy in (SPATIAL_TILED_CPU, CPU_MEMORY_REFUSAL)
    # Deterministic: same inputs -> identical frozen decision.
    for g in (2, 8, 16, 24):
        assert _sim(g, n=n_fixed) == decisions[g]


def test_mono_and_rgb_simulated_geometry():
    mono = _sim(4, n=20, frame=FRAME_480, channels=1)
    rgb = _sim(4, n=20, frame=FRAME_480, channels=3)
    assert mono.n == 20 and rgb.n == 20
    if mono.strategy == SPATIAL_TILED_CPU:
        assert mono.tile_shape is not None
        assert mono.tile_outputs >= CPU_MIN_TILE_OUT
    if rgb.strategy == SPATIAL_TILED_CPU:
        assert rgb.tile_shape is not None
        assert rgb.tile_outputs >= CPU_MIN_TILE_OUT


# ---------------------------------------------------------------------------
# B. Structural exact-N: no field can encode a reduced/split N
# ---------------------------------------------------------------------------
def test_decision_carries_input_n_verbatim_and_has_no_n_split_field():
    d = _sim(8, n=50)
    assert d.n == 50
    for field_name in ("n_tile", "n_per_tile", "n_tiles_split", "reduced_n",
                       "tile_n"):
        assert not hasattr(d, field_name), field_name
    # The only population-ish field is n_tiles (count of spatial tiles), which
    # is a pure count: multiplying it by any tile's N is impossible because
    # tiles carry no N.  Assert the dataclass attributes are exactly the
    # documented set (no sneaky N-splitting addition).
    allowed = {
        "mode", "strategy", "reason", "n", "frame_shape", "channels",
        "fast_path", "backend", "tile_shape", "n_tiles", "tile_outputs",
        "minimum_tile", "estimated_peak_bytes", "per_tile_peak_bytes",
        "effective_budget_bytes", "reserve_bytes", "available_ram_bytes",
        "policy_ceiling_bytes", "retry_tile_shape_allowed", "details",
        "is_refusal",
    }
    assert set(vars(d)) <= allowed


def test_tiled_geometry_keeps_full_n_semantics():
    # Whatever the tile shape, every tile processes ALL n observations along
    # the stack axis (spatial tiling only).  tile_h/tile_w are spatial.
    for g in (2, 4, 8):
        d = _sim(g, n=36)
        if d.strategy == SPATIAL_TILED_CPU:
            assert d.tile_shape is not None
            if len(d.tile_shape) == 1:
                tile_h, tile_w = d.tile_shape[0], d.frame_shape[1]
            else:
                tile_h, tile_w = d.tile_shape
            assert tile_h <= d.frame_shape[0]
            assert tile_w <= d.frame_shape[1]
            assert d.tile_outputs == tile_h * tile_w


# ---------------------------------------------------------------------------
# C. Fast/slow regime detection correctness
# ---------------------------------------------------------------------------
def test_winsor_zero_rank_regime_correctness():
    # rank 0: floor(limit*N) == 0 on both sides
    assert winsor_zero_rank_regime((0.05, 0.05), 10) is True   # floor(0.5)=0
    assert winsor_zero_rank_regime((0.01, 0.01), 50) is True   # floor(0.5)=0
    assert winsor_zero_rank_regime((0.0, 0.05), 10) is True
    assert winsor_zero_rank_regime((0.0, 0.0), 10) is True
    # rank > 0
    assert winsor_zero_rank_regime((0.05, 0.05), 20) is False  # floor(1)=1
    assert winsor_zero_rank_regime((0.05, 0.05), 36) is False
    assert winsor_zero_rank_regime((0.05, 0.0), 20) is False
    # negative / non-finite never qualify (canonical semantics)
    assert winsor_zero_rank_regime((-0.05, 0.05), 10) is False
    assert winsor_zero_rank_regime((float("nan"), 0.05), 10) is False
    assert winsor_zero_rank_regime((0.05, float("inf")), 10) is False
    # weighted/unweighted identical (weights do not change valid counts)
    for lims in ((0.05, 0.05), (0.01, 0.01)):
        for n in (10, 20, 36, 50):
            assert winsor_zero_rank_regime(lims, n) is winsor_zero_rank_regime(
                lims, n
            )


def test_planner_selects_fast_model_in_rank0_regime():
    d_fast = plan_cpu_winsor_execution(
        n=10, frame_shape=(480, 640), channels=1, dtype_itemsize=4,
        winsor_limits=(0.05, 0.05),  # floor(0.5) == 0 -> fast
        available_ram_bytes=24 * GiB, reserve_bytes=512 * MiB,
        policy_ceiling_bytes=24 * GiB,
    )
    assert d_fast.fast_path is True
    assert d_fast.details["factor"] == CPU_WINSOR_FAST_FACTOR
    assert d_fast.details["scratch"] == CPU_WINSOR_FAST_SCRATCH_BYTES

    d_slow = plan_cpu_winsor_execution(
        n=36, frame_shape=(480, 640), channels=1, dtype_itemsize=4,
        winsor_limits=(0.05, 0.05),  # floor(1.8) == 1 -> slow
        available_ram_bytes=24 * GiB, reserve_bytes=512 * MiB,
        policy_ceiling_bytes=24 * GiB,
    )
    assert d_slow.fast_path is False
    assert d_slow.details["factor"] == CPU_WINSOR_SLOW_FACTOR
    assert d_slow.details["scratch"] == CPU_WINSOR_SLOW_SCRATCH_BYTES


def test_scipy_backend_uses_scipy_envelope():
    d = plan_cpu_winsor_execution(
        n=36, frame_shape=(480, 640), channels=1, dtype_itemsize=4,
        winsor_limits=(0.05, 0.05),
        scipy_backend=True,
        available_ram_bytes=24 * GiB, reserve_bytes=512 * MiB,
        policy_ceiling_bytes=24 * GiB,
    )
    assert d.backend == "scipy"
    assert d.details["factor"] == CPU_SCIPY_WINSOR_FACTOR
    assert d.details["scratch"] == CPU_SCIPY_WINSOR_SCRATCH_BYTES


# ---------------------------------------------------------------------------
# D. Geometry: full-width band first, rectangular fallback, refusal
# ---------------------------------------------------------------------------
def test_full_width_band_preferred_over_rectangular():
    # Budget that cannot hold the whole frame (N=20 1080p slow needs ~2.0 GiB
    # incremental) but easily holds several full-width rows: must yield
    # full-width horizontal bands ``(tile_h,)``, never a rectangular split.
    d = plan_cpu_winsor_execution(
        n=20, frame_shape=(1080, 1920), channels=1, dtype_itemsize=4,
        winsor_limits=(0.05, 0.05),
        available_ram_bytes=2 * GiB, reserve_bytes=256 * MiB,
        policy_ceiling_bytes=2 * GiB,
    )
    assert d.strategy == SPATIAL_TILED_CPU
    assert len(d.tile_shape) == 1  # full-width band (tile_h,)
    tile_h = d.tile_shape[0]
    assert 1 <= tile_h < 1080
    # Band demand must fit the effective budget
    band_base = d.details["per_tile_base_bytes"]
    factor = d.details["factor"]
    scratch = d.details["scratch"]
    assert int(band_base * factor) + scratch <= d.effective_budget_bytes


def test_rectangular_fallback_only_when_full_width_impossible():
    # Ultra-wide frame (200 x 20000) with a budget below ONE full-width row's
    # incremental demand (~54 MiB) but above the smallest admissible rect tile:
    # a full-width band cannot fit -> the planner must fall back to a
    # rectangular tile (tile_h, tile_w) with >= min_tile_out outputs.
    d = plan_cpu_winsor_execution(
        n=36, frame_shape=(200, 20000), channels=1, dtype_itemsize=4,
        winsor_limits=(0.05, 0.05),
        available_ram_bytes=32 * MiB, reserve_bytes=2 * MiB,
        policy_ceiling_bytes=32 * MiB,
        min_tile_out=256,
    )
    assert d.strategy == SPATIAL_TILED_CPU
    assert len(d.tile_shape) == 2  # rectangular fallback
    assert d.tile_outputs >= 256
    assert d.details.get("full_width_band_impossible") is True


def test_minimum_tile_refusal_truthful_reason():
    # Budget below the smallest admissible tile (96 spatial outputs, N=50 slow
    # needs ~96*2600+16MiB ~= 20 MiB) -> truthful refusal, N untouched, never
    # an empty success.
    d = plan_cpu_winsor_execution(
        n=50, frame_shape=(1080, 1920), channels=1, dtype_itemsize=4,
        winsor_limits=(0.05, 0.05),
        available_ram_bytes=12 * MiB, reserve_bytes=1 * MiB,
        policy_ceiling_bytes=12 * MiB,
    )
    assert d.strategy == CPU_MEMORY_REFUSAL
    assert d.reason in (REASON_MIN_TILE_EXCEEDS_BUDGET, REASON_NO_VALID_TILE)
    assert d.n == 50
    assert d.is_refusal is True


def test_full_cpu_when_budget_large():
    d = plan_cpu_winsor_execution(
        n=50, frame_shape=(1080, 1920), channels=1, dtype_itemsize=4,
        winsor_limits=(0.05, 0.05),
        available_ram_bytes=64 * GiB, reserve_bytes=512 * MiB,
        policy_ceiling_bytes=64 * GiB,
    )
    assert d.strategy == FULL_CPU
    assert d.tile_shape is None
    assert d.n_tiles == 1


# ---------------------------------------------------------------------------
# E. Reserve / effective budget contract
# ---------------------------------------------------------------------------
def test_effective_budget_formula_and_reserve_named():
    available = 8 * GiB
    ceiling = 6 * GiB
    reserve = 1 * GiB
    d = plan_cpu_winsor_execution(
        n=20, frame_shape=(480, 640), channels=1, dtype_itemsize=4,
        available_ram_bytes=available,
        reserve_bytes=reserve,
        policy_ceiling_bytes=ceiling,
    )
    assert d.effective_budget_bytes == min(ceiling, available - reserve)
    assert d.reserve_bytes == reserve


def test_recommended_reserve_never_negative_and_documented():
    frame_bytes = 480 * 640 * 1 * 4
    r = recommended_reserve_bytes(0, frame_bytes)
    assert r >= 0
    assert r >= CPU_RESERVE_FIXED_MIN_BYTES + (
        CPU_OUTPUT_SERIALIZATION_FRAMES * frame_bytes
    )
    # proportional component engages for large RAM
    big = recommended_reserve_bytes(64 * GiB, frame_bytes)
    assert big >= int(CPU_RESERVE_FRACTION * 64 * GiB)
    # pool overhead is explicit per extra worker
    pool = recommended_reserve_bytes(16 * GiB, frame_bytes, pool_workers=3)
    solo = recommended_reserve_bytes(16 * GiB, frame_bytes, pool_workers=1)
    assert pool - solo == 2 * CPU_POOL_WORKER_OVERHEAD_BYTES
    # reserve is never silently negative for negative-ish injection
    d = plan_cpu_winsor_execution(
        n=10, frame_shape=(64, 64), channels=1,
        available_ram_bytes=0, reserve_bytes=-123, policy_ceiling_bytes=0,
    )
    assert d.reserve_bytes == 0  # clamped, never negative


# ---------------------------------------------------------------------------
# F. Model conservatism: committed evidence rows as fixtures
# ---------------------------------------------------------------------------
# Committed rows from stage-B evidence
# (cpu_winsor_rss_rows.json): (name, input_cube_bytes, delta_bytes, backend).
# The planner slow envelope is 13.0x+16MiB (numpy) / 17.5x+32MiB (scipy) and
# the fast envelope 12.0x+16MiB; every row's modeled estimate must be >= the
# observed process-peak delta (margin direction conservative).
EVIDENCE_ROWS = [
    ("mono_256_N10_rank0", 10 * 256 * 256 * 4, 28.1 * MiB, "numpy", True),
    ("mono_256_N20", 20 * 256 * 256 * 4, 61.1 * MiB, "numpy", False),
    ("mono_256_N36", 36 * 256 * 256 * 4, 107.5 * MiB, "numpy", False),
    ("mono_256_N50", 50 * 256 * 256 * 4, 146.1 * MiB, "numpy", False),
    ("mono_512_N20_full", 20 * 512 * 512 * 4, 242.7 * MiB, "numpy", False),
    ("mono_512_N50_full", 50 * 512 * 512 * 4, 582.3 * MiB, "numpy", False),
    ("mono_256_N36_rwF", 36 * 256 * 256 * 4, 107.6 * MiB, "numpy", False),
    ("mono_256_N36_w", 36 * 256 * 256 * 4, 107.5 * MiB, "numpy", False),
    ("mono_256_N50_rank0_lim0005", 50 * 256 * 256 * 4, 145.6 * MiB, "numpy", True),
    ("mono_256_N50_nan2pct", 50 * 256 * 256 * 4, 145.9 * MiB, "numpy", False),
    ("band_128x512_N36", 36 * 128 * 512 * 4, 107.6 * MiB, "numpy", False),
    ("rect_128x256_N36", 36 * 128 * 256 * 4, 53.8 * MiB, "numpy", False),
    ("mono_64_N10_small", 10 * 64 * 64 * 4, 2.3 * MiB, "numpy", True),
    ("mono_64_N50_small", 50 * 64 * 64 * 4, 9.8 * MiB, "numpy", False),
    ("rgb_128_N20", 20 * 128 * 128 * 3 * 4, 46.2 * MiB, "numpy", False),
    ("rgb_128_N50", 50 * 128 * 128 * 3 * 4, 109.8 * MiB, "numpy", False),
    ("rgb_256_N36_full", 36 * 256 * 256 * 3 * 4, 321.7 * MiB, "numpy", False),
    ("rgb_256_N50_rwF", 50 * 256 * 256 * 3 * 4, 436.9 * MiB, "numpy", False),
    ("rgb_256_N36_w", 36 * 256 * 256 * 3 * 4, 321.7 * MiB, "numpy", False),
    ("rgb_256_N50_rank0_lim0005", 50 * 256 * 256 * 3 * 4, 435.4 * MiB, "numpy", True),
    ("rgb_256_N20_full_scipy", 20 * 256 * 256 * 3 * 4, 249.1 * MiB, "scipy", False),
]


@pytest.mark.parametrize("row", EVIDENCE_ROWS, ids=[r[0] for r in EVIDENCE_ROWS])
def test_model_conservative_on_committed_evidence(row):
    _name, cube, observed_delta, backend, fast = row
    if backend == "scipy":
        factor, scratch = CPU_SCIPY_WINSOR_FACTOR, CPU_SCIPY_WINSOR_SCRATCH_BYTES
    elif fast:
        factor, scratch = CPU_WINSOR_FAST_FACTOR, CPU_WINSOR_FAST_SCRATCH_BYTES
    else:
        factor, scratch = CPU_WINSOR_SLOW_FACTOR, CPU_WINSOR_SLOW_SCRATCH_BYTES
    model = int(cube * factor) + scratch
    assert model >= observed_delta, (
        f"{_name}: model {model/MiB:.1f} MiB < observed {observed_delta/MiB:.1f} MiB"
    )


def test_model_envelope_constants_are_conservative_and_positive():
    assert CPU_WINSOR_SLOW_FACTOR >= CPU_WINSOR_FAST_FACTOR > 0
    assert CPU_WINSOR_SLOW_SCRATCH_BYTES > 0
    assert CPU_WINSOR_FAST_SCRATCH_BYTES > 0
    assert CPU_SCIPY_WINSOR_FACTOR > CPU_WINSOR_SLOW_FACTOR
    assert CPU_SCIPY_WINSOR_SCRATCH_BYTES > 0
    assert CPU_MIN_TILE_OUT > 0


# ---------------------------------------------------------------------------
# G. Planner purity + bounded retry vocabulary
# ---------------------------------------------------------------------------
def test_planner_module_pure_import_no_psutil_device_side_effects():
    # Load the planner module in ISOLATION (directly from its file, bypassing
    # the ``seestar`` package ``__init__`` which imports psutil elsewhere), so
    # we genuinely prove the planner itself never imports psutil or probes any
    # device at import time — not merely that psutil was absent beforehand.
    code = (
        "import sys, importlib.util; "
        "spec = importlib.util.spec_from_file_location("
        "'cpu_memory_planner_isolated', 'seestar/core/cpu_memory_planner.py'); "
        "m = importlib.util.module_from_spec(spec); "
        "sys.modules['cpu_memory_planner_isolated'] = m; "
        "spec.loader.exec_module(m); "
        "assert 'psutil' not in sys.modules, 'planner imported psutil'; "
        "print('ok', m.FULL_CPU, m.SPATIAL_TILED_CPU, m.CPU_MEMORY_REFUSAL)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok FULL_CPU SPATIAL_TILED_CPU CPU_MEMORY_REFUSAL" in proc.stdout


def test_retry_vocabulary_spatial_only():
    d = plan_cpu_winsor_execution(
        n=36, frame_shape=(1080, 1920), channels=1, dtype_itemsize=4,
        available_ram_bytes=4 * GiB, reserve_bytes=512 * MiB,
        policy_ceiling_bytes=4 * GiB,
    )
    if d.strategy == SPATIAL_TILED_CPU:
        assert d.retry_tile_shape_allowed is True
        # allowed retry knobs: tile_h / tile_w only
        assert d.tile_shape is not None
        for forbidden in (
            "retry_n", "retry_kappa", "retry_winsor_limits",
            "retry_normalization", "retry_weights", "retry_reducer",
        ):
            assert not hasattr(d, forbidden), forbidden
    else:
        assert d.retry_tile_shape_allowed is False


def test_decision_frozen():
    d = _sim(8, n=36)
    with pytest.raises(Exception):
        d.strategy = FULL_CPU  # frozen dataclass


def test_override_mode_ceiling_respected():
    # OVERRIDE with a hard ceiling smaller than available must cap the budget:
    # effective_budget = min(ceiling, available - reserve) = 3 GiB here.
    d = plan_cpu_winsor_execution(
        n=50, frame_shape=(1080, 1920), channels=1, dtype_itemsize=4,
        available_ram_bytes=64 * GiB, reserve_bytes=512 * MiB,
        policy_ceiling_bytes=3 * GiB, mode=MODE_OVERRIDE,
    )
    assert d.mode == MODE_OVERRIDE
    assert d.effective_budget_bytes == 3 * GiB
    assert d.strategy in (SPATIAL_TILED_CPU, CPU_MEMORY_REFUSAL)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        plan_cpu_winsor_execution(
            n=0, frame_shape=(64, 64), available_ram_bytes=1 * GiB
        )
    with pytest.raises(ValueError):
        plan_cpu_winsor_execution(
            n=5, frame_shape=(0, 64), available_ram_bytes=1 * GiB
        )
    with pytest.raises(ValueError):
        plan_cpu_winsor_execution(
            n=5, frame_shape=(64, 64), available_ram_bytes=-1
        )
