#!/usr/bin/env python
"""Phase F profiler for the adaptive VRAM execution planner (Track P4).

Standalone, opt-in (ZSSS_GPU_PROFILE=1), deterministic, NON-scientific:
it only observes; it can never change reduction results.

Validates, on the CURRENT tree (feature/winsorized-gpu-perf-memory,
phase F), on the 2 GiB MX150:

  * the planner decision table at the REAL device memory state (memGetInfo
    + reusable pool free, minus the 128 MiB explicit reserve) for the phase E
    fit witnesses: 480x270 and 1080p at N_batch 20/32/50, 4K N_batch 32/50,
    RGB 1080p N_batch 50, and the fallback witnesses (huge N, narrow
    frame): FULL_GPU / TILED_GPU(tile_shape) / CPU_FALLBACK(reason),
  * EXECUTION of every GPU decision the planner makes at 1080p and 480x270:
    FULL_GPU runs the untiled twin, TILED_GPU runs the seam with the
    planner-chosen tile_shape, and the tiled results stay bitwise identical
    to the phase E validated reference geometries (cross-shape identity),
    with wall time + peak pool demand recorded,
  * phase E consistency: N=20 -> FULL_GPU (untiled fits the optimized
    model on this card), N=32/50 -> TILED_GPU (untiled OutOfMemoryError),
    huge N -> CPU_FALLBACK(vram_no_valid_tile).

Usage:
  .venv/bin/python profiling/prof_winsorized_gpu_phaseF.py
      [--outdir profiling/results_phaseF] [--quick] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("ZSSS_GPU_PROFILE", "1")  # opt-in instrumentation ON

REPO = "/home/tristan/.openclaw/workspace/projects/zeseestarstacker"
sys.path.insert(0, REPO)

import cupy as cp  # noqa: E402

from prof_winsorized_gpu_phaseE import (  # noqa: E402
    bitwise,
    gpu_full,
    gpu_tiled,
    make_images,
    make_weights,
    run_instrumented,
)

from seestar.core.gpu_vram_planner import (  # noqa: E402
    CPU_FALLBACK,
    FULL_GPU,
    TILED_GPU,
    plan_winsorized_gpu_execution,
)

WINSOR_LIMITS = (0.05, 0.05)
MB = 1024 * 1024
RESERVE = 128 * MB
# Phase E validated reference row bands at 1080p for the cross-shape proof.
REF_TILES = {32: 540, 50: 384}


def mib(b):
    return b / (1024.0 * 1024.0)


def live_device_state():
    """(driver_free_bytes, pool_free_bytes) exactly like the dispatch."""
    free, _total = cp.cuda.runtime.memGetInfo()
    pool_free = cp.get_default_memory_pool().free_bytes()
    return int(free), int(pool_free)


def plan_row(tag, n, frame, channels=1):
    free, pool_free = live_device_state()
    d = plan_winsorized_gpu_execution(
        n_batch=n,
        frame_shape=frame,
        channels=channels,
        dtype_itemsize=4,
        winsor_limits=WINSOR_LIMITS,
        driver_free_bytes=free,
        pool_free_bytes=pool_free,
        reserve_bytes=RESERVE,
    )
    row = {
        "tag": tag,
        "n_batch": n,
        "frame": list(frame),
        "channels": channels,
        "kind": d.kind,
        "reason": d.reason,
        "tile_shape": list(d.tile_shape) if d.tile_shape is not None else None,
        "fast_path": d.fast_path,
        "demand_full_mib": round(mib(d.demand_full_bytes), 2),
        "demand_tile_mib": round(mib(d.demand_tile_bytes), 2),
        "effective_budget_mib": round(mib(d.effective_budget_bytes), 2),
        "reserve_mib": round(mib(d.reserve_bytes), 2),
        "driver_free_mib": round(mib(free), 2),
        "pool_free_mib": round(mib(pool_free), 2),
        "n_tiles": d.n_tiles,
        "tile_outputs": d.tile_outputs,
    }
    return d, row


def run_one(fn, snap_mem=True):
    """Cold-pool instrumented run -> (outcome, wall_ms, mem) without raise."""
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    try:
        res, wall, mem = run_instrumented(fn, snap_mem=snap_mem)
        return "OK", res, wall, mem
    except Exception as exc:
        return "%s: %s" % (type(exc).__name__, exc), None, None, None


def execution_evidence():
    rows = []

    # -- 480x270: untiled fits at every phase E N_batch -------------------
    for n in (20, 32, 50):
        d, row = plan_row("480x270_n%d" % n, n, (270, 480))
        imgs = make_images(n, (270, 480), seed=11, nan_frac=0.05)
        w = make_weights(n)
        outcome, res, wall, mem = run_one(lambda: gpu_full(imgs, w))
        row["expected_kind"] = FULL_GPU
        row["outcome"] = outcome
        row["wall_ms"] = None if wall is None else round(wall * 1000.0, 1)
        row["mem"] = mem
        if res is not None:
            row["rejected_pct"] = float(res[2])
        rows.append(row)
        print("480x270 n=%d -> %s (%s) %s"
              % (n, d.kind, outcome, row["wall_ms"]))

    # -- 1080p N=20: phase E fit witness (untiled fits the optimized model) -
    n = 20
    d, row = plan_row("fit1080p_n20", n, (1080, 1920))
    imgs = make_images(n, (1080, 1920), seed=11, nan_frac=0.05)
    w = make_weights(n)
    outcome, res, wall, mem = run_one(lambda: gpu_full(imgs, w))
    row["expected_kind"] = FULL_GPU
    row["outcome"] = outcome
    row["wall_ms"] = None if wall is None else round(wall * 1000.0, 1)
    row["mem"] = mem
    if res is not None:
        row["rejected_pct"] = float(res[2])
    rows.append(row)
    print("1080p n=20 -> %s (%s) %s" % (d.kind, outcome, row["wall_ms"]))

    # -- 1080p N=32 / N=50: untiled OOMs (phase E), planner goes TILED ----
    for n in (32, 50):
        d, row = plan_row("fit1080p_n%d" % n, n, (1080, 1920))
        assert d.kind == TILED_GPU, (n, d.kind, d.reason)
        imgs = make_images(n, (1080, 1920), seed=11, nan_frac=0.05)
        w = make_weights(n)
        # whole-stack attempt (phase E: OutOfMemoryError on this card)
        outcome, _res, _wall, _mem = run_one(
            lambda: gpu_full(imgs, w), snap_mem=False
        )
        row["untiled_outcome"] = outcome
        # planner-chosen geometry executes
        outcome_p, res_p, wall_p, mem_p = run_one(
            lambda: gpu_tiled(imgs, w, tile_shape=d.tile_shape)
        )
        row["planner_tile_outcome"] = outcome_p
        row["planner_tile_wall_ms"] = (
            None if wall_p is None else round(wall_p * 1000.0, 1)
        )
        row["planner_tile_mem"] = mem_p
        # phase E validated reference geometry (cross-shape bitwise proof)
        outcome_r, res_r, wall_r, mem_r = run_one(
            lambda: gpu_tiled(imgs, w, tile_shape=REF_TILES[n])
        )
        row["ref_tile_outcome"] = outcome_r
        row["ref_tile_wall_ms"] = (
            None if wall_r is None else round(wall_r * 1000.0, 1)
        )
        row["ref_tile_mem"] = mem_r
        if res_p is not None and res_r is not None:
            row["cross_shape_bitwise"] = bool(bitwise(res_p, res_r))
            row["planner_pct"] = float(res_p[2])
        else:
            row["cross_shape_bitwise"] = None
        rows.append(row)
        print("1080p n=%d -> %s tile=%s (%s, ref %s) cross=%s"
              % (n, d.kind, d.tile_shape, outcome_p, outcome_r,
                 row.get("cross_shape_bitwise")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir",
        default=os.path.join(REPO, "profiling", "results_phaseF"),
    )
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dev = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    log_path = os.path.join(args.outdir, "progress.log")
    log = open(log_path, "w")
    print("host_gpu = %s" % dev, file=log)
    print("cupy = %s" % cp.__version__, file=log)
    print("mode = %s" % ("quick" if args.quick else "full"), file=log)

    results = {
        "host_gpu": dev,
        "cupy": cp.__version__,
        "note": "Phase F adaptive VRAM execution planner validation (Track P4)",
        "sections": {},
    }

    # -- 1. pure decision table at the real device memory state ----------
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    table = []
    for n, frame, tag, ch in [
        (20, (270, 480), "480x270_n20", 1),
        (32, (270, 480), "480x270_n32", 1),
        (50, (270, 480), "480x270_n50", 1),
        (20, (1080, 1920), "1080p_n20", 1),
        (32, (1080, 1920), "1080p_n32", 1),
        (50, (1080, 1920), "1080p_n50", 1),
        (100, (1080, 1920), "1080p_n100", 1),
        (32, (2160, 3840), "4k_n32", 1),
        (50, (2160, 3840), "4k_n50", 1),
        (50, (1080, 1920), "1080p_rgb_n50", 3),
        (4_000_000, (1080, 1920), "1080p_n4M_must_fallback", 1),
        (400_000, (64, 40), "narrow_64x40_n400k_must_fallback", 1),
    ]:
        d, row = plan_row(tag, n, frame, channels=ch)
        table.append(row)
        print("%-34s -> %-12s %s%s" % (
            tag, d.kind, d.reason or "",
            list(d.tile_shape) if d.tile_shape else ""), file=log)
    results["sections"]["decision_table"] = table

    # -- 2. execution evidence for the GPU decisions ----------------------
    results["sections"]["execution"] = execution_evidence()

    with open(os.path.join(args.outdir, "phaseF_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.close()
    print("written:", os.path.join(args.outdir, "phaseF_results.json"))


if __name__ == "__main__":
    main()
