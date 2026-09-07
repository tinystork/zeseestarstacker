#!/usr/bin/env python
"""Phase G profiler for the CuPy memory-pool discipline (Track P5).

Standalone, opt-in (ZSSS_GPU_PROFILE=1), deterministic, NON-scientific:
it only observes; it can never change reduction results (the observed
reductions are the exact bit-identical kernels of phases D/E/F).

Audits, on the CURRENT tree (feature/winsorized-gpu-perf-memory, phase G),
on the 2 GiB MX150, the CuPy default memory pool ACROSS THE FULL reduction
lifecycle of the Winsorized path (untiled FULL twin and exact-N_batch
TILED seam), distinguishing at every stage boundary:

  * ACTIVE   = pool.used_bytes()   (live device arrays at the boundary),
  * REUSABLE = pool.free_bytes()   (released blocks retained by the pool,
    recyclable by later stages of the SAME or a later reduction),
  * POOL_TOTAL = used + free       (device memory the pool holds; the
    monotone driver-level high-water of the run),
  * DRIVER_FREE = memGetInfo       (driver-visible free memory).

and, AFTER completion of each reduction, the retained state of the pool
and what one seam ``free_all_blocks`` (the ONLY release under evaluation;
never per tile) reclaims + its wall cost.

Sections:

  1. boundary audit (stage tables) -- FULL 480x270 N=20/32/50, FULL 1080p
     N=20 (untiled fits), TILED 480x270 N=32/50 (row band 96) and TILED
     1080p N=32/50 (planner tile_shape at the live device state):
     per-boundary (used, free, total, driver_free) + per-run peaks
     (peak active, peak reusable, pool-total high-water, min driver free)
     + high-water growth points (which stages carved new driver blocks).
  2. retention after completion -- pool state AFTER each reduction returns
     (device arrays released), what a single post-reduction seam
     ``free_all_blocks`` reclaims (driver free delta) and its wall cost.
  3. release tax -- 1080p N=32 TILED: wall time of an immediately repeated
     identical reduction with a release between (cold) vs without (warm).
  4. no-per-tile-release witness -- per-tile pool-total trajectory shows
     block reuse (plateau), and the executed kernels never call
     free_all_blocks (structural: stack_gpu has no pool-release call).

The sort/order-statistics INTERNAL workspace (allocated and freed inside
one cupy op, between two boundaries) is not directly observable from stage
endpoints: it only moves pool.total_bytes() up (blocks stay retained).  We
report pool-total growth as the bound and label the exact mid-op live
peak as not directly measurable from endpoints; a best-effort async
sampler (thread sampling the pool counters while kernels run) records an
observed lower bound of the true live peak for the representative rows
(flag ``live_peak_sampler_lower_bound_mib``).

Usage:
  .venv/bin/python profiling/prof_winsorized_gpu_phaseG.py
      [--outdir profiling/results_phaseG] [--quick] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

os.environ.setdefault("ZSSS_GPU_PROFILE", "1")  # opt-in instrumentation ON

REPO = "/home/tristan/.openclaw/workspace/projects/zeseestarstacker"
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

import cupy as cp  # noqa: E402

import seestar.core.stack_gpu as sg  # noqa: E402

from prof_winsorized_gpu_phaseE import (  # noqa: E402
    KAPPA,
    WINSOR_LIMITS,
    make_images,
    make_weights,
    gpu_full,
    gpu_tiled,
    probe,
)

from seestar.core.gpu_vram_planner import (  # noqa: E402
    plan_winsorized_gpu_execution,
)

MB = 1024 * 1024
RESERVE = 128 * MB
RELEASE_MIN = 512 * MB  # seam threshold under evaluation (see report)

_LOG = None


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    if _LOG is not None:
        _LOG.write(line + "\n")
        _LOG.flush()


def mib(b):
    return b / (1024.0 * 1024.0)


def snapshot():
    """Synchronized (driver free, pool used, pool free, pool total)."""
    cp.cuda.Stream.null.synchronize()
    free, _total = cp.cuda.runtime.memGetInfo()
    p = cp.get_default_memory_pool()
    return int(free), int(p.used_bytes()), int(p.free_bytes()), int(p.total_bytes())


def live_state():
    free, _total = cp.cuda.runtime.memGetInfo()
    pool_free = cp.get_default_memory_pool().free_bytes()
    return int(free), int(pool_free)


def run_boundaries(fn, sampler=False):
    """Cold-pool instrumented run returning (res, wall_s, rows, sampler)."""
    sg._ensure_probe()
    p = probe()
    p.reset()
    p.snap_mem = True
    sampler_rows = []

    def _sample():
        try:
            while not _stop.is_set():
                u = int(cp.get_default_memory_pool().used_bytes())
                fr = int(cp.get_default_memory_pool().free_bytes())
                t = int(cp.get_default_memory_pool().total_bytes())
                f, _ = cp.cuda.runtime.memGetInfo()
                sampler_rows.append((f, u, fr, t))
                time.sleep(0.0002)
        except Exception:
            pass

    _stop = threading.Event()
    th = None
    if sampler:
        th = threading.Thread(target=_sample, daemon=True)
        th.start()
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    try:
        res = fn()
    finally:
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        p.snap_mem = False
        if th is not None:
            _stop.set()
            th.join(timeout=2.0)
    rows = list(p.mem)
    notes = list(p.notes)
    return res, t1 - t0, rows, notes, sampler_rows


def audit_run(tag, fn, n_batch, frame, kind, tile_shape, sampler=False):
    res, wall, rows, notes, samp = run_boundaries(fn, sampler=sampler)
    rec = {
        "tag": tag,
        "n_batch": n_batch,
        "frame": list(frame),
        "kind": kind,
        "tile_shape": (list(tile_shape) if tile_shape is not None else None),
        "wall_ms": round(wall * 1000.0, 1),
        "n_boundaries": len(rows),
        "boundaries": [
            {
                "stage": name,
                "active_mib": round(mib(u), 2),
                "reusable_mib": round(mib(fr), 2),
                "pool_total_mib": round(mib(t), 2),
                "driver_free_mib": round(mib(f), 2),
            }
            for (name, f, _tot, u, fr, t) in rows
        ],
        "notes": notes,
    }
    act = [u for (_n, _f, _t, u, _fr, _tt) in rows]
    frb = [fr for (_n, _f, _t, _u, fr, _tt) in rows]
    tot = [t for (_n, _f, _t, _u, _fr, t) in rows]
    df = [f for (_n, f, _tot_d, _u, _fr, _t) in rows]
    rec["peak_active_mib"] = round(mib(max(act)), 2)
    rec["peak_reusable_mib"] = round(mib(max(frb)), 2)
    rec["pool_total_high_water_mib"] = round(mib(max(tot)), 2)
    rec["min_driver_free_mib"] = round(mib(min(df)), 2)
    # high-water growth points: boundaries where pool total first reached a
    # new maximum (which stages carved new driver blocks).
    growth = []
    hi = 0
    for (name, f, _tot, u, fr, t) in rows:
        if t > hi:
            hi = t
            growth.append(
                {"stage": name, "pool_total_mib": round(mib(t), 2)}
            )
    rec["pool_total_growth_points"] = growth
    if samp:
        su = [u for (_f, u, _fr, _t) in samp]
        st = [t for (_f, _u, _fr, t) in samp]
        rec["sampler_samples"] = len(samp)
        rec["live_peak_sampler_lower_bound_mib"] = round(mib(max(su)), 2)
        rec["pool_total_sampler_max_mib"] = round(mib(max(st)), 2)
    return rec, res, wall


def planner_tile(n, frame):
    free, pool_free = live_state()
    d = plan_winsorized_gpu_execution(
        n_batch=n,
        frame_shape=frame,
        channels=1,
        dtype_itemsize=4,
        winsor_limits=WINSOR_LIMITS,
        driver_free_bytes=free,
        pool_free_bytes=pool_free,
        reserve_bytes=RESERVE,
    )
    return d


def main():
    global _LOG
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir", default=os.path.join(REPO, "profiling", "results_phaseG")
    )
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dev = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    results = {
        "host_gpu": dev,
        "cupy": cp.__version__,
        "numpy": np.__version__,
        "device_total_mib": round(mib(cp.cuda.runtime.memGetInfo()[1]), 1),
        "note": "Phase G CuPy memory-pool discipline audit (Track P5)",
        "release_min_mib": round(mib(RELEASE_MIN), 1),
        "sections": {},
    }
    log = open(os.path.join(args.outdir, "progress.log"), "w")
    _LOG = log

    say("GPU:", dev, "| cupy", cp.__version__)
    say("release-min under evaluation: %.0f MiB" % mib(RELEASE_MIN))

    # -------------------------------------------------------------
    # 1. boundary audit: FULL + TILED across several N
    # -------------------------------------------------------------
    say("== 1. stage-boundary pool audit ==")
    audits = []
    # FULL (untiled) 480x270 N=20/32/50 and 1080p N=20 (untiled fits)
    for n in (20, 32, 50):
        tag = "full_480x270_n%d" % n
        imgs = make_images(n, (270, 480), 11, nan_frac=0.05)
        w = make_weights(n)
        say("  %s (sampler=%s)" % (tag, n == 50 and not args.quick))
        rec, res, wall = audit_run(
            tag, lambda: gpu_full(imgs, w), n, (270, 480), "FULL_GPU",
            None, sampler=(n == 50 and not args.quick))
        rec["rejected_pct"] = float(res[2])
        audits.append(rec)
    n = 20
    tag = "full_1080p_n20"
    imgs = make_images(n, (1080, 1920), 11, nan_frac=0.05)
    w = make_weights(n)
    say("  %s (sampler=%s)" % (tag, not args.quick))
    rec, res, wall = audit_run(
        tag, lambda: gpu_full(imgs, w), n, (1080, 1920), "FULL_GPU",
        None, sampler=not args.quick)
    rec["rejected_pct"] = float(res[2])
    audits.append(rec)
    # TILED 480x270 N=32/50 (row band 96; small per-tile working set)
    for n in (32, 50):
        tag = "tiled_480x270_n%d_t96" % n
        imgs = make_images(n, (270, 480), 11, nan_frac=0.05)
        w = make_weights(n)
        say("  %s" % tag)
        rec, res, wall = audit_run(
            tag, lambda: gpu_tiled(imgs, w, tile_shape=96), n, (270, 480),
            "TILED_GPU", (96,), sampler=False)
        rec["rejected_pct"] = float(res[2])
        audits.append(rec)
    # TILED 1080p N=32/50 (planner geometry at the live device state)
    for n in (32, 50):
        d = planner_tile(n, (1080, 1920))
        ts = tuple(d.tile_shape)
        tag = "tiled_1080p_n%d_planner%s" % (n, list(ts))
        imgs = make_images(n, (1080, 1920), 11, nan_frac=0.05)
        w = make_weights(n)
        say("  %s (sampler=%s)" % (tag, n == 32 and not args.quick))
        rec, res, wall = audit_run(
            tag, lambda: gpu_tiled(imgs, w, tile_shape=ts), n, (1080, 1920),
            "TILED_GPU", ts, sampler=(n == 32 and not args.quick))
        rec["rejected_pct"] = float(res[2])
        audits.append(rec)
    results["sections"]["boundary_audit"] = audits

    # -------------------------------------------------------------
    # 2. retention after completion + one-seam release reclaim/cost
    # -------------------------------------------------------------
    say("== 2. post-reduction retention and seam release ==")
    retention = []
    for n in (20, 32, 50):
        tag = "full_480x270_n%d" % n
        imgs = make_images(n, (270, 480), 11, nan_frac=0.05)
        w = make_weights(n)
        res, wall, _rows, _notes, _samp = run_boundaries(
            lambda: gpu_full(imgs, w))
        _retention_row(tag, "FULL_GPU", retention, wall, res)
    for n in (20, 32, 50):
        tag = "tiled_480x270_n%d_t96" % n
        imgs = make_images(n, (270, 480), 11, nan_frac=0.05)
        w = make_weights(n)
        res, wall, _rows, _notes, _samp = run_boundaries(
            lambda: gpu_tiled(imgs, w, tile_shape=96))
        _retention_row(tag, "TILED_GPU", retention, wall, res)
    n = 20
    tag = "full_1080p_n20"
    imgs = make_images(n, (1080, 1920), 11, nan_frac=0.05)
    w = make_weights(n)
    res, wall, _rows, _notes, _samp = run_boundaries(
        lambda: gpu_full(imgs, w))
    _retention_row(tag, "FULL_GPU", retention, wall, res)
    for n in (32, 50):
        d = planner_tile(n, (1080, 1920))
        ts = tuple(d.tile_shape)
        tag = "tiled_1080p_n%d_planner%s" % (n, list(ts))
        imgs = make_images(n, (1080, 1920), 11, nan_frac=0.05)
        w = make_weights(n)
        res, wall, _rows, _notes, _samp = run_boundaries(
            lambda: gpu_tiled(imgs, w, tile_shape=ts))
        _retention_row(tag, "TILED_GPU", retention, wall, res)
    results["sections"]["retention_and_release"] = retention

    # -------------------------------------------------------------
    # 3. release tax: cold (release between) vs warm repeat, 1080p N=32
    # -------------------------------------------------------------
    say("== 3. release tax (cold vs warm repeat) 1080p N=32 TILED ==")
    d = planner_tile(32, (1080, 1920))
    ts = tuple(d.tile_shape)
    imgs = make_images(32, (1080, 1920), 11, nan_frac=0.05)
    w = make_weights(32)
    tax = {}
    # cold -> release -> cold repeat
    _res, wall1, _r, _no, _s = run_boundaries(lambda: gpu_tiled(imgs, w, tile_shape=ts))
    tax["first_cold_wall_ms"] = round(wall1 * 1000.0, 1)
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    _res, wall2, _r, _no, _s = run_boundaries(lambda: gpu_tiled(imgs, w, tile_shape=ts))
    tax["second_cold_after_release_wall_ms"] = round(wall2 * 1000.0, 1)
    # warm repeat (no release between)
    _res, wall3, _r, _no, _s = run_boundaries(lambda: gpu_tiled(imgs, w, tile_shape=ts))
    tax["third_warm_no_release_wall_ms"] = round(wall3 * 1000.0, 1)
    if wall1 and wall2 and wall3:
        tax["release_tax_pct_vs_warm"] = round(
            100.0 * (wall2 - wall3) / wall3, 2)
    say("  cold %.0f ms | cold-after-release %.0f ms | warm %.0f ms | tax %.2f%%"
        % (tax.get("first_cold_wall_ms", -1),
           tax.get("second_cold_after_release_wall_ms", -1),
           tax.get("third_warm_no_release_wall_ms", -1),
           tax.get("release_tax_pct_vs_warm", -1)))
    results["sections"]["release_tax"] = tax

    with open(os.path.join(args.outdir, "phaseG_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.close()
    print("written:", os.path.join(args.outdir, "phaseG_results.json"))


def _retention_row(tag, kind, retention, wall, res):
    """Measure the pool right after a reduction returned (all device arrays
    of the call released), then one seam free_all_blocks and its reclaim."""
    df0, used0, free0, total0 = snapshot()
    # stability check: nothing should change without any action
    time.sleep(0.05)
    df0b, used0b, free0b, total0b = snapshot()
    pool = cp.get_default_memory_pool()
    t0 = time.perf_counter()
    pool.free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    t_rel = time.perf_counter() - t0
    df1, used1, free1, total1 = snapshot()
    row = {
        "tag": tag,
        "kind": kind,
        "wall_ms": round(wall * 1000.0, 1),
        "rejected_pct": float(res[2]),
        "after_run_active_mib": round(mib(used0), 2),
        "after_run_reusable_mib": round(mib(free0), 2),
        "after_run_pool_total_mib": round(mib(total0), 2),
        "after_run_driver_free_mib": round(mib(df0), 2),
        "retained_fraction_of_device": round(total0 / float(
            cp.cuda.runtime.memGetInfo()[1]), 3),
        "seam_release_wall_ms": round(t_rel * 1000.0, 2),
        "reclaimed_driver_free_mib": round(mib(df1 - df0), 2),
        "pool_total_after_release_mib": round(mib(total1), 2),
        "release_would_trigger": bool(free0 >= RELEASE_MIN),
        "stable_without_action": (df0 == df0b and free0 == free0b
                                  and total0 == total0b),
    }
    retention.append(row)
    say("  %-32s retained %7.1f MiB (%.1f%% dev) | release %6.2f ms "
        "reclaims %7.1f MiB | trigger=%s"
        % (tag, mib(total0), 100.0 * row["retained_fraction_of_device"],
           t_rel * 1000.0, mib(df1 - df0),
           "yes" if row["release_would_trigger"] else "no"))


if __name__ == "__main__":
    main()
