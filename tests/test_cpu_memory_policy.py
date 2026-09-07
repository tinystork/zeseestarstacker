"""Stage E1 — automatic CPU memory policy engine + production wiring.

Contract (see ``docs/8.4.0_state.md`` stage E1 and
``.a2a-reports/.../stage-e1-contract.md``): normal product execution resolves
the CPU memory policy automatically (AUTO) from actual machine/workload
state — never from the legacy user HQ RAM value, never from GPU
model/VRAM/CUDA name; a single resolved byte budget flows through the CPU
Winsorized chain; the exact-N spatial CPU tiled driver (stage C) and the pure
planner (stage B) are actually used (FULL_CPU / SPATIAL_TILED_CPU /
CPU_MEMORY_REFUSAL); GPU CPU_FALLBACK routes through the SAME automatic CPU
policy; policy + per-decision provenance is emitted bounded + fail-open.

Structure:

* pure policy module tests (deterministic 2/8/16/24 GiB simulations at fixed
  N — N invariant, only FULL/SPATIAL geometry/REFUSAL differs; AUTO ceiling
  vs OVERRIDE; runtime re-evaluation formula; reserve named/never-negative;
  provenance record builders are plain JSON-safe dicts, no image/mask
  arrays);
* production wiring tests through the REAL queue_manager seams (real
  ``_stack_batch`` Winsorized path on the ``test_gpu_pool_release``-style
  bare stacker, real ``_gpu_reduce_winsorized`` fallback, real
  ``_process_completed_batch``):
  - AUTO ignores a legacy ``max_hq_mem``;
  - OVERRIDE (env / internal seam) is provenance-visible and honours the
    requested budget;
  - FULL_CPU uses the untiled wrapper; SPATIAL_TILED_CPU uses the exact-N
    tiled driver with the planner tile_shape; refusal raises (and through
    stage D is a truthful terminal failure — no source move / no counter
    advance);
  - GPU CPU_FALLBACK resolves the automatic CPU policy (not ``max_hq_mem``);
  - sentinel byte-for-byte budget propagation through the WIRED path (spy at
    the worker tuple boundary — the same boundary the stage C sentinel uses),
    no hidden 1/2 GiB fallback;
  - provenance records emitted through the durable sink (MEMORY_POLICY /
    CPU_WINSOR_MEMORY_DECISION / CPU_WINSOR_MEMORY_RETRY /
    CPU_WINSOR_MEMORY_REFUSAL) are present and shaped correctly (fail-open,
    bounded, no arrays).

Harness style mirrors ``tests/test_gpu_pool_release.py`` (real ``_stack_batch``
Winsorized on a bare ``SeestarQueuedStacker``).
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pytest

import seestar.queuep.queue_manager as qm
from seestar.core import cpu_memory_policy as cmp
from seestar.core.cpu_memory_planner import (
    CPU_MEMORY_REFUSAL,
    CPU_MIN_TILE_OUT,
    FULL_CPU,
    MODE_AUTO,
    MODE_OVERRIDE,
    SPATIAL_TILED_CPU,
    CpuMemoryDecision,
    recommended_reserve_bytes,
)
from seestar.core.cpu_winsor_exact_n import CpuWinsorMemoryRefused
from seestar.queuep.queue_manager import (
    BatchReductionError,
    SeestarQueuedStacker,
)

MIB = 1024 ** 2
GIB = 1024 ** 3
SENTINEL = 24 * GIB + 173  # stage C archaeology sentinel (24 GiB + 173)


# ---------------------------------------------------------------------------
# Harness: production-style bare Winsorized stacker (mirrors
# test_gpu_pool_release._winsor_stack) + deterministic RAM overrides.
# ---------------------------------------------------------------------------

def _policy_stack(tmp_path=None, *, request_gpu=False, available=64 * GIB,
                  total=64 * GIB, rss=400 * MIB, name="e1"):
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o.logger = logging.getLogger("zsss.e1.cpu.memory.policy")
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
    o.max_hq_mem = 4_000_000_000  # LEGACY value: must NOT be consulted.
    o.batch_size = 10
    o.settings = None
    o.reference_header_for_wcs = None
    o.reference_wcs_object = None
    o.interbatch_norm_active = False
    o.max_stack_workers = 1
    o._current_batch_paths = []
    o._quality_reference_scale = 1.0
    o.request_gpu = request_gpu
    o._cpu_total_ram_bytes_override = int(total)
    o._cpu_available_ram_bytes_override = int(available)
    o._cpu_process_rss_bytes_override = int(rss)
    o._cpu_mem_policy_preflight = None
    o._last_classic_batch_solved = True
    o.stop_processing = False
    o.processing_error = None
    o.failed_stack_count = 0
    o.stacked_batches_count = 0
    o.images_in_cumulative_stack = 0
    o.align_on_disk = False
    o.aligned_temp_paths = []
    o._indices_cache = {}
    o._send_eta_update = lambda *a, **k: None
    o._update_preview_sum_w = lambda *a, **k: None
    o._solve_cumulative_stack = lambda *a, **k: (None, None)
    o.partial_save_interval = 0
    o.output_filename = "stack"
    if tmp_path is not None:
        out = tmp_path / name
        out.mkdir(exist_ok=True)
        o.output_folder = str(out)
        o.stacked_subdir_name = "stacked"
        o.move_stacked = True
        o.batch_count_path = str(out / "batch_count.txt")
        shape = (4, 5, 3)
        memdir = out / "memmap_accumulators"
        memdir.mkdir(exist_ok=True)
        o.sum_memmap_path = str(memdir / "cumulative_SUM.npy")
        o.wht_memmap_path = str(memdir / "cumulative_WHT.npy")
        o.cumulative_sum_memmap = np.lib.format.open_memmap(
            o.sum_memmap_path, mode="w+", dtype=np.float32, shape=shape
        )
        o.cumulative_wht_memmap = np.lib.format.open_memmap(
            o.wht_memmap_path, mode="w+", dtype=np.float32, shape=shape
        )
        o.cumulative_sum_memmap[:] = 0.0
        o.cumulative_wht_memmap[:] = 0.0
        o.memmap_shape = shape
        o.memmap_dtype_sum = np.float32
        o.memmap_dtype_wht = np.float32
    return o


def _winsor_item(value, shape=(2, 2)):
    img = np.full(shape, value, dtype=np.float32)
    mask = np.ones(shape, dtype=bool)
    hdr = __import__("astropy.io.fits", fromlist=["Header"]).Header()
    return (img, hdr, {"snr": 1.0, "stars": 0.0}, None, mask)


def _winsor_batch(shape=(2, 2), n_in=4, value=10.0, outlier=1000.0):
    return [_winsor_item(value, shape) for _ in range(n_in)] + [
        _winsor_item(outlier, shape)
    ]


def _lines(stacker):
    """Return the provenance lines emitted through update_progress."""
    out = []

    def sink(message, progress=None, level=None):
        if isinstance(message, str):
            out.append(message)

    stacker.update_progress = sink
    return out


# ---------------------------------------------------------------------------
# 1. Pure module: 2/8/16/24 GiB simulations at a fixed scientific N.
# ---------------------------------------------------------------------------

def _ram_sim(available_gib):
    return cmp.resolve_cpu_winsor_decision(
        mode=MODE_AUTO,
        n=36,
        frame_shape=(1080, 1920),
        channels=3,
        dtype_itemsize=4,
        winsor_limits=(0.05, 0.05),
        available_ram_bytes=available_gib * GIB,
    )


def test_ram_sweep_n_invariant_strategy_differs():
    n_fixed = 36
    d2 = _ram_sim(2)
    d8 = _ram_sim(8)
    d16 = _ram_sim(16)
    d24 = _ram_sim(24)

    # N invariant across the sweep (structural exact-N contract).
    for d in (d2, d8, d16, d24):
        assert d.n == n_fixed
    # Only FULL_CPU / SPATIAL_TILED_CPU geometry differs between budgets.
    assert {d.strategy for d in (d2, d8, d16, d24)} <= {
        FULL_CPU,
        SPATIAL_TILED_CPU,
    }
    # 16/24 GiB can host the whole frame (FULL); 2/8 GiB cannot (SPATIAL).
    assert d16.strategy == FULL_CPU
    assert d24.strategy == FULL_CPU
    assert d2.strategy == SPATIAL_TILED_CPU
    assert d8.strategy == SPATIAL_TILED_CPU
    # Geometry strictly differs with the budget (bigger RAM -> bigger tiles).
    assert d8.tile_shape != d2.tile_shape
    assert d8.tile_shape[0] > d2.tile_shape[0]
    # FULL has no tile geometry; effective budget grows monotonically and is
    # never negative; reserve is named and never negative.
    assert d16.tile_shape is None
    assert d2.effective_budget_bytes < d8.effective_budget_bytes < d16.effective_budget_bytes
    for d in (d2, d8, d16, d24):
        assert d.reserve_bytes >= 0
        assert d.effective_budget_bytes >= 0


def test_ram_sweep_refusal_when_effective_budget_cannot_host_min_tile():
    # An extreme-low effective budget (below the minimum viable tile) refuses
    # truthfully — never an empty success, never a reduced N.
    d = cmp.resolve_cpu_winsor_decision(
        mode=MODE_AUTO,
        n=36,
        frame_shape=(1080, 1920),
        channels=3,
        dtype_itemsize=4,
        winsor_limits=(0.05, 0.05),
        available_ram_bytes=100 * MIB,
    )
    assert d.strategy == CPU_MEMORY_REFUSAL
    assert d.is_refusal
    assert d.n == 36
    assert d.reason in (
        "cpu_budget_negative",
        "cpu_min_tile_exceeds_budget",
        "cpu_no_valid_tile",
    )


# ---------------------------------------------------------------------------
# 2. Pure module: AUTO vs OVERRIDE + runtime re-evaluation formula.
# ---------------------------------------------------------------------------

def test_auto_ceiling_from_available_minus_reserve():
    p = cmp.resolve_cpu_policy_preflight(
        available_ram_preflight_bytes=8 * GIB,
        total_ram_bytes=16 * GIB,
        mode=MODE_AUTO,
    )
    assert p.mode == MODE_AUTO
    assert p.total_ram_bytes == 16 * GIB
    assert p.available_ram_preflight_bytes == 8 * GIB
    assert p.reserve_bytes == recommended_reserve_bytes(8 * GIB, 0, 1)
    assert p.policy_ceiling_bytes == max(0, 8 * GIB - p.reserve_bytes)
    assert p.requested_budget_bytes is None
    assert p.is_override is False


def test_override_ceiling_is_requested_budget_and_visible():
    p = cmp.resolve_cpu_policy_preflight(
        available_ram_preflight_bytes=8 * GIB,
        total_ram_bytes=16 * GIB,
        mode=MODE_OVERRIDE,
        requested_budget_bytes=SENTINEL,
    )
    assert p.is_override is True
    assert p.policy_ceiling_bytes == SENTINEL
    assert p.requested_budget_bytes == SENTINEL
    assert p.reserve_bytes >= 0


def test_runtime_reevaluation_effective_is_min_ceiling_available_minus_reserve():
    available_now = 6 * GIB
    ceiling = 4 * GIB  # preflight AUTO ceiling
    # Reserve recomputed at runtime from available RAM and the workload frame
    # bytes (named policy: fixed min + proportional + 2x output frames).
    reserve = recommended_reserve_bytes(available_now, 64 * 64 * 1 * 4, 1)
    d = cmp.resolve_cpu_winsor_decision(
        mode=MODE_AUTO,
        n=8,
        frame_shape=(64, 64),
        channels=1,
        dtype_itemsize=4,
        available_ram_bytes=available_now,
        policy_ceiling_bytes=ceiling,
        winsor_limits=(0.2, 0.2),
    )
    assert d.effective_budget_bytes == min(
        ceiling, available_now - reserve
    )
    assert d.policy_ceiling_bytes == ceiling
    # Reserve is recomputed at runtime from available RAM (named policy) and
    # is never negative.
    assert d.reserve_bytes == reserve
    assert d.reserve_bytes >= 0


def test_runtime_reevaluation_ceiling_binds_when_ram_grew():
    # More RAM now than at preflight: the preflight ceiling still binds.
    ceiling = 2 * GIB
    d = cmp.resolve_cpu_winsor_decision(
        mode=MODE_AUTO,
        n=8,
        frame_shape=(64, 64),
        channels=1,
        dtype_itemsize=4,
        available_ram_bytes=16 * GIB,
        policy_ceiling_bytes=ceiling,
        winsor_limits=(0.2, 0.2),
    )
    assert d.effective_budget_bytes == min(
        ceiling, 16 * GIB - recommended_reserve_bytes(16 * GIB, 0, 1)
    )


# ---------------------------------------------------------------------------
# 3. Pure module: provenance record builders (bounded, JSON-safe, no arrays).
# ---------------------------------------------------------------------------

def _fake_decision(**over):
    base = dict(
        mode=MODE_AUTO,
        strategy=FULL_CPU,
        reason=None,
        n=8,
        frame_shape=(64, 64),
        channels=1,
        fast_path=True,
        backend="numpy",
        tile_shape=None,
        n_tiles=1,
        tile_outputs=4096,
        minimum_tile=(1, 96),
        estimated_peak_bytes=123456789,
        per_tile_peak_bytes=123456789,
        effective_budget_bytes=987654321,
        reserve_bytes=268435456,
        available_ram_bytes=4 * GIB,
        policy_ceiling_bytes=4 * GIB,
        retry_tile_shape_allowed=False,
        details={},
    )
    base.update(over)
    return CpuMemoryDecision(**base)


def test_provenance_builders_are_json_safe_no_arrays():
    p = cmp.resolve_cpu_policy_preflight(
        available_ram_preflight_bytes=8 * GIB,
        mode=MODE_OVERRIDE,
        requested_budget_bytes=SENTINEL,
    )
    recs = [
        cmp.cpu_memory_policy_tokens(p),
        cmp.cpu_winsor_decision_tokens(_fake_decision(), 4 * GIB),
        cmp.cpu_winsor_retry_tokens(
            old_tile_shape=(128,), new_tile_shape=(96,), reason="allocation_failure"
        ),
        cmp.cpu_winsor_refusal_tokens(
            scientific_n=8,
            effective_budget_bytes=100,
            minimum_estimated_bytes=999,
            reason="cpu_min_tile_exceeds_budget",
        ),
    ]
    for rec in recs:
        # Plain JSON-safe scalars only (no image/mask arrays, no nested
        # payloads) — json round-trip must succeed.
        assert json.loads(json.dumps(rec)) == rec
    # Override visibility in the policy record.
    policy = cmp.cpu_memory_policy_tokens(p)
    assert policy["mode"] == MODE_OVERRIDE
    assert policy["requested_budget_bytes"] == SENTINEL
    # Decision record shape: strategy tokens full|spatial_tiled|refused.
    dec = cmp.cpu_winsor_decision_tokens(_fake_decision(), 4 * GIB)
    assert dec["strategy"] == "full"
    assert dec["scientific_n"] == 8
    dec_t = cmp.cpu_winsor_decision_tokens(
        _fake_decision(strategy=SPATIAL_TILED_CPU, tile_shape=(128, 64), n_tiles=30),
        4 * GIB,
    )
    assert dec_t["strategy"] == "spatial_tiled"
    assert dec_t["tile_shape"] == "128,64"
    assert dec_t["tile_count"] == 30


# ---------------------------------------------------------------------------
# 4. Production wiring: preflight capture (MEMORY_POLICY) once per run.
# ---------------------------------------------------------------------------

def test_preflight_capture_emits_memory_policy_and_stores_ceiling(tmp_path, monkeypatch):
    o = _policy_stack(tmp_path, available=8 * GIB, total=16 * GIB)
    lines = _lines(o)
    monkeypatch.setenv(cmp.CPU_MEMORY_OVERRIDE_ENV, str(SENTINEL))
    pre = o._capture_cpu_memory_policy_preflight()
    assert pre is not None
    assert pre.mode == MODE_OVERRIDE
    assert pre.policy_ceiling_bytes == SENTINEL
    assert pre.total_ram_bytes == 16 * GIB
    assert o._cpu_mem_preflight_record() is pre
    policy_lines = [l for l in lines if l.startswith("MEMORY_POLICY ")]
    assert len(policy_lines) == 1
    assert "mode=OVERRIDE" in policy_lines[0]
    assert f"requested_budget_bytes={SENTINEL}" in policy_lines[0]
    assert "policy_ceiling_bytes=" in policy_lines[0]


def test_auto_preflight_ignores_legacy_max_hq_mem(tmp_path):
    o = _policy_stack(tmp_path, available=8 * GIB)
    o.max_hq_mem = 1234  # legacy value must never influence AUTO.
    lines = _lines(o)
    pre = o._capture_cpu_memory_policy_preflight()
    assert pre.mode == MODE_AUTO
    assert "max_hq_mem" not in " ".join(lines)
    assert pre.policy_ceiling_bytes == max(
        0, 8 * GIB - recommended_reserve_bytes(8 * GIB, 0, 1)
    )


def test_override_internal_seam_visible_in_provenance(tmp_path):
    o = _policy_stack(tmp_path, available=8 * GIB)
    o._cpu_memory_override_bytes_attr = 2 * GIB
    lines = _lines(o)
    pre = o._capture_cpu_memory_policy_preflight()
    assert pre.mode == MODE_OVERRIDE
    assert pre.policy_ceiling_bytes == 2 * GIB
    assert any("mode=OVERRIDE" in l and "requested_budget_bytes=2147483648" in l for l in lines)


# ---------------------------------------------------------------------------
# 5. Production wiring: dispatch + sentinel budget propagation.
# ---------------------------------------------------------------------------

def test_full_cpu_wired_budget_reaches_worker_tuple_byte_for_byte(
    tmp_path, monkeypatch
):
    """§7 sentinel: the WIRED path propagates the exact resolved budget
    (decision.effective_budget_bytes) to the worker tuple — never a hidden
    1/2 GiB default, never the legacy max_hq_mem."""
    o = _policy_stack(tmp_path, available=32 * GIB, total=64 * GIB)
    o.max_hq_mem = 42  # legacy poison: if consulted, the test fails loudly.
    o._cpu_memory_override_bytes_attr = SENTINEL  # OVERRIDE ceiling.
    seen = {}

    def spy_worker(args):
        seen["tuple"] = args
        img0 = np.asarray(args[1][0])
        return (
            np.zeros(img0.shape, dtype=np.float32),
            np.ones(img0.shape[:2], dtype=np.float32),
            0.0,
        )

    monkeypatch.setattr(qm, "_stack_worker", spy_worker)
    lines = _lines(o)
    # Real _stack_batch Winsorized path (CPU: request_gpu=False).
    o._stack_batch(_winsor_batch(shape=(2, 2), n_in=4), 1, 1)
    args = seen["tuple"]
    assert len(args) == 9
    assert args[8] == SENTINEL, "wired budget must be the resolved sentinel"
    assert args[8] != 2_000_000_000  # no deep 2 GiB default
    assert args[8] != 1_000_000_000  # no wrapper 1 GiB default
    assert args[8] != 42  # legacy max_hq_mem never consulted
    # Decision provenance was emitted with the same value.
    dec_lines = [l for l in lines if l.startswith("CPU_WINSOR_MEMORY_DECISION ")]
    assert len(dec_lines) == 1
    assert f"effective_budget_bytes={SENTINEL}" in dec_lines[0]
    assert "strategy=full" in dec_lines[0]


def test_spatial_tiled_dispatch_uses_exact_n_driver_with_planner_tile(
    tmp_path, monkeypatch
):
    """SPATIAL_TILED_CPU -> the stage-C exact-N tiled driver with the
    planner's tile_shape and the explicit resolved budget."""
    o = _policy_stack(tmp_path, available=64 * GIB, total=64 * GIB)
    o._cpu_memory_override_bytes_attr = 300 * MIB  # force SPATIAL for the cube
    seen = {}

    def spy_tiled(images, weights=None, **_kw):
        seen["tile_shape"] = _kw.get("tile_shape")
        seen["max_mem_bytes"] = _kw.get("max_mem_bytes")
        img0 = np.asarray(images[0])
        return (
            np.zeros(img0.shape, dtype=np.float32),
            np.ones(img0.shape[:2], dtype=np.float32),
            0.0,
        )

    monkeypatch.setattr(qm, "stack_winsorized_sigma_cpu_tiled", spy_tiled)
    lines = _lines(o)
    # 20 mono 640x640 frames: full-frame modelled demand ~442 MiB > 300 MiB
    # ceiling -> the planner must choose SPATIAL_TILED_CPU.
    o._stack_batch(_winsor_batch(shape=(640, 640), n_in=19), 1, 1)
    assert seen["tile_shape"] is not None
    assert len(seen["tile_shape"]) >= 1
    # The budget forwarded equals the decision's effective budget (planner
    # decision, not the legacy max_hq_mem).
    pre = o._cpu_mem_preflight_record()
    assert pre.mode == MODE_OVERRIDE
    assert seen["max_mem_bytes"] == 300 * MIB
    dec_lines = [l for l in lines if l.startswith("CPU_WINSOR_MEMORY_DECISION ")]
    assert len(dec_lines) == 1
    assert "strategy=spatial_tiled" in dec_lines[0]
    assert f"tile_shape={seen['tile_shape'][0]}" in dec_lines[0]


def test_refusal_raises_truthfully_through_stack_batch(tmp_path):
    """CPU_MEMORY_REFUSAL raises (stage D converts it into a truthful
    terminal failure); never (None, None, None), never an empty success."""
    o = _policy_stack(tmp_path, available=64 * GIB, total=64 * GIB)
    o._cpu_memory_override_bytes_attr = 8 * MIB  # below minimum viable tile
    with pytest.raises(BatchReductionError) as ei:
        o._stack_batch(_winsor_batch(shape=(64, 64), n_in=9), 1, 1)
    assert "batch commit refused" in str(ei.value) or "reducer" in str(ei.value).lower()


def test_refusal_through_stage_d_no_source_move_no_counter_advance(tmp_path):
    """Refusal through the transactional helper (stage D): no source move, no
    committed-counter advance, no count-file, cumulative untouched."""
    o = _policy_stack(tmp_path, available=64 * GIB, total=64 * GIB)
    o._cpu_memory_override_bytes_attr = 8 * MIB
    src_dir = tmp_path / "input"
    src_dir.mkdir(exist_ok=True)
    src_paths = []
    for n_ in ("a.fit", "b.fit"):
        p = src_dir / n_
        p.write_bytes(b"\x00" * 32)
        src_paths.append(str(p))
    items = _winsor_batch(shape=(64, 64), n_in=9)
    o._current_batch_paths = list(src_paths)
    with pytest.raises(BatchReductionError):
        o._process_completed_batch(items, 1, 1, None)
    assert o.stacked_batches_count == 0
    assert not os.path.exists(o.batch_count_path)
    for p in src_paths:
        assert os.path.exists(p)
    assert not (src_dir / "stacked").exists()
    assert not np.any(o.cumulative_sum_memmap)
    assert not np.any(o.cumulative_wht_memmap)
    assert o.images_in_cumulative_stack == 0


# ---------------------------------------------------------------------------
# 6. GPU CPU_FALLBACK routes through the automatic CPU policy.
# ---------------------------------------------------------------------------

def test_gpu_cpu_fallback_uses_auto_policy_not_max_hq_mem(tmp_path, monkeypatch):
    """Mock the GPU backend/planner to return CPU_FALLBACK, then assert the
    CPU closure resolves FULL/SPATIAL/REFUSAL from the automatic CPU policy
    (explicit planner budget), never the legacy max_hq_mem."""
    from types import SimpleNamespace

    o = _policy_stack(tmp_path, available=32 * GIB, total=64 * GIB,
                      request_gpu=True)
    o.max_hq_mem = 42  # legacy poison
    o._cpu_memory_override_bytes_attr = SENTINEL
    seen = {}

    def spy_worker(args):
        seen["tuple"] = args
        img0 = np.asarray(args[1][0])
        return (
            np.zeros(img0.shape, dtype=np.float32),
            np.ones(img0.shape[:2], dtype=np.float32),
            0.0,
        )

    monkeypatch.setattr(qm, "_stack_worker", spy_worker)
    # GPU planner forced to CPU_FALLBACK (real cupy device present on this
    # host): the seam must then run fn_cpu — the automatic CPU policy closure.
    monkeypatch.setattr(
        qm,
        "plan_winsorized_gpu_execution",
        lambda **k: SimpleNamespace(
            kind="CPU_FALLBACK",
            reason="planner_failure",
            demand_full_bytes=1,
            demand_tile_bytes=1,
            effective_budget_bytes=1,
            reserve_bytes=1,
            tile_shape=None,
            n_tiles=0,
            n_batch=5,
        ),
    )
    lines = _lines(o)
    o._stack_batch(_winsor_batch(shape=(2, 2), n_in=4), 1, 1)
    assert seen.get("tuple") is not None, "CPU fallback closure must have run"
    args = seen["tuple"]
    assert args[8] == SENTINEL  # automatic policy budget, byte-for-byte
    assert args[8] != 42
    dec_lines = [l for l in lines if l.startswith("CPU_WINSOR_MEMORY_DECISION ")]
    assert len(dec_lines) == 1


# ---------------------------------------------------------------------------
# 7. Provenance records through the durable sink (fail-open, bounded).
# ---------------------------------------------------------------------------

def test_provenance_emission_fail_open_never_breaks_run(tmp_path, monkeypatch):
    """A raising provenance record BUILDER (the emission path is internally
    fail-open) must never alter a valid CPU run — capture and the reduction
    still complete with results."""
    o = _policy_stack(tmp_path, available=32 * GIB, total=64 * GIB)
    o._cpu_memory_override_bytes_attr = SENTINEL

    seen = {}

    def spy_worker(args):
        seen["tuple"] = args
        img0 = np.asarray(args[1][0])
        return (
            np.zeros(img0.shape, dtype=np.float32),
            np.ones(img0.shape[:2], dtype=np.float32),
            0.0,
        )

    monkeypatch.setattr(qm, "_stack_worker", spy_worker)
    # Fail-open: the provenance BUILDERS/emission raising must NEVER break
    # capture or the CPU run — the policy record is still resolved and stored
    # (the emission sink itself is internally fail-open).
    def boom_line(*a, **k):
        raise RuntimeError("provenance builder boom")

    o._provenance_line = boom_line
    pre = o._capture_cpu_memory_policy_preflight()
    assert pre is not None
    assert pre.mode == MODE_OVERRIDE
    assert pre.policy_ceiling_bytes == SENTINEL
    # CPU run still completes.
    o._stack_batch(_winsor_batch(shape=(2, 2), n_in=4), 1, 1)
    assert seen["tuple"] is not None


def test_decision_provenance_bounded_small_record():
    """Per-decision records stay bounded (scalar tokens only — no arrays)."""
    d = _ram_sim(8)
    rec = cmp.cpu_winsor_decision_tokens(d, 8 * GIB)
    assert len(json.dumps(rec)) < 2048
    assert all(
        not isinstance(v, np.ndarray)
        and not hasattr(v, "shape")
        for v in rec.values()
    )


# ---------------------------------------------------------------------------
# 8. CLOSURE REWORK-1 (Nono false-cpu_budget_negative): the per-reduction
#    reserve must never include the process-pool duplication overhead — a
#    high configured ``max_stack_workers`` must not refuse a valid small batch
#    on a low-RAM multi-core host.
# ---------------------------------------------------------------------------

def test_low_ram_high_workers_tiny_batch_never_refused(tmp_path):
    """Real E1 capture + real ``_run_cpu_winsor_policy`` seam: with
    ``max_stack_workers = 8`` (8-core-style) and only ~1.5 GiB available RAM,
    a tiny valid Winsorized batch is NOT refused ``cpu_budget_negative`` — the
    AUTO ceiling stays positive and the reserve stays <= available RAM (pool
    duplication overhead is a preflight capability, excluded from the
    per-reduction reserve)."""
    o = _policy_stack(tmp_path, available=1500 * MIB, total=8 * GIB)
    o.max_stack_workers = 8  # high configured worker count (prod default ~6)
    pre = o._capture_cpu_memory_policy_preflight()
    assert pre is not None
    assert pre.mode == MODE_AUTO
    assert pre.policy_ceiling_bytes > 0, "AUTO ceiling zeroed by pool overhead"
    assert pre.reserve_bytes <= 1500 * MIB
    assert pre.requested_budget_bytes is None

    seen = {}

    def spy_wrapper(images, weights=None, **kw):
        seen["budget"] = kw.get("max_mem_bytes")
        img0 = np.asarray(images[0])
        return (
            np.zeros(img0.shape, dtype=np.float32),
            np.ones(img0.shape[:2], dtype=np.float32),
            0.0,
        )

    o._stack_winsorized_sigma = spy_wrapper
    imgs = [np.full((2, 2), 10.0, dtype=np.float32) for _ in range(4)] + [
        np.full((2, 2), 1000.0, dtype=np.float32)
    ]
    # Real seam: no exception, FULL_CPU dispatch with a positive explicit
    # budget (never cpu_budget_negative).
    o._run_cpu_winsor_policy(
        imgs, None, kappa=3.0, winsor_limits=(0.2, 0.2), return_weights=True
    )
    assert seen.get("budget") is not None
    assert seen["budget"] > 0


def test_preflight_record_excludes_pool_overhead_as_capability(tmp_path):
    """MEMORY_POLICY records the pool worker count as a CAPABILITY quantity
    (pool_overhead_excluded=true) — never silently double-counted into the
    AUTO ceiling reserve."""
    o = _policy_stack(tmp_path, available=2 * GIB, total=8 * GIB)
    o.max_stack_workers = 8
    lines = _lines(o)
    pre = o._capture_cpu_memory_policy_preflight()
    assert pre is not None
    assert pre.policy_ceiling_bytes == max(
        0, 2 * GIB - recommended_reserve_bytes(2 * GIB, 0, 1)
    )
    policy_lines = [l for l in lines if l.startswith("MEMORY_POLICY ")]
    assert len(policy_lines) == 1
    assert "pool_workers_capability=8" in policy_lines[0]
    assert "pool_overhead_excluded=true" in policy_lines[0]
