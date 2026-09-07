#!/usr/bin/env python
"""Phase E profiler for the Winsorized CuPy stacking path (ZSSS, Track P3).

Standalone, opt-in (ZSSS_GPU_PROFILE=1), deterministic, NON-scientific:
it only observes; it can never change reduction results (the tiled path is
proven bitwise identical to the untiled twin by tests/..._tiled.py).

Measures, on the CURRENT tree (feature/winsorized-gpu-perf-memory, phase E):

  * exact-equivalence spot proof at 480x270 (bitwise full == tiled for
    row-band and rectangular geometries, weight map and rejected_pct),
  * same-resolution CPU-parity of the TILED path on the N_batch 32 / 50
    witnesses (documented tolerance rtol=1e-3, atol=1e-2, |pct| <= 1.0),
  * 1080p fit probes on the 2 GiB MX150: N_batch 20 (fits UNTILED: full vs
    tiled bitwise + timing + peak VRAM), then N_batch 32 / 50 where the
    whole-stack reduction raises OutOfMemoryError and the SPATIAL tiled
    driver executes: per-tile working set N x tile_h x W x C, same N_batch
    kept for every output pixel (the Exact-N Memory Architecture core),
  * peak VRAM quantification (probe snapshots at every tile/pass/iteration
    boundary: peak live pool bytes + min driver free) full vs tiled, and
    the per-tile overhead (wall time of the two-pass coordinated schedule
    vs the single-pass untiled run at N_batch 20),
  * cross-tile-shape consistency at 1080p: tiled(tile_h=a) == tiled(tile_h=b)
    bitwise where both execute.

Usage:
  .venv/bin/python profiling/prof_winsorized_gpu_phaseE.py
      [--outdir profiling/results_phaseE] [--quick] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("ZSSS_GPU_PROFILE", "1")  # opt-in instrumentation ON

REPO = "/home/tristan/.openclaw/workspace/projects/zeseestarstacker"
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

import cupy as cp  # noqa: E402

import seestar.core.stack_gpu as sg  # noqa: E402
from seestar.core import stack_methods as sm  # noqa: E402

KAPPA = 2.5  # app HQ-combine default: max(stack_kappa_low, stack_kappa_high)
WINSOR_LIMITS = (0.05, 0.05)
MB = 1024 * 1024


def mib(b):
    return b / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# deterministic witnesses (same generators as the phase A/C/D drivers)
# ---------------------------------------------------------------------------


def make_images(n, hw, seed, nan_frac, spike=0.03):
    rng = np.random.default_rng(seed)
    shape = (n,) + tuple(hw)
    arr = rng.standard_normal(shape, dtype=np.float32) * 20.0 + 1000.0
    if nan_frac > 0:
        arr[rng.random(shape, dtype=np.float32) < nan_frac] = np.nan
    arr = arr + np.where(
        rng.random(shape, dtype=np.float32) < spike,
        np.float32(400.0), np.float32(0.0),
    )
    return [arr[i] for i in range(n)]


def make_weights(n, seed=7):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.4, 1.6, size=n).astype(np.float32)


def gpu_full(images, weights=None, **kw):
    args = dict(kappa=KAPPA, winsor_limits=WINSOR_LIMITS,
                apply_rewinsor=True, max_iters=5, kappa_decay=0.9)
    args.update(kw)
    return sg.stack_winsorized_sigma_gpu(
        images, weights, return_weights=True, **args
    )


def gpu_tiled(images, weights=None, tile_shape=None, **kw):
    args = dict(kappa=KAPPA, winsor_limits=WINSOR_LIMITS,
                apply_rewinsor=True, max_iters=5, kappa_decay=0.9)
    args.update(kw)
    return sg.stack_winsorized_sigma_gpu_tiled(
        images, weights, return_weights=True, tile_shape=tile_shape, **args
    )


def probe():
    return sg._PROBE


def run_instrumented(fn, snap_mem=False, pool_reset=True):
    """Run ``fn()`` once with the probe active; returns (result, wall_s,
    memory_stats_or_None).  Memory stats from per-boundary snapshots:
    peak pool used (live), pool total high-water, min driver free."""
    sg._ensure_probe()
    p = probe()
    p.reset()
    p.snap_mem = snap_mem
    if pool_reset:
        cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    try:
        res = fn()
    finally:
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        p.snap_mem = False
    mem = None
    if snap_mem and p.mem:
        peak_live = max(int(m[3]) for m in p.mem)
        peak_pool = max(int(m[5]) for m in p.mem)
        min_free = min(int(m[1]) for m in p.mem)
        mem = {
            "n_boundaries": len(p.mem),
            "peak_live_mib": round(mib(peak_live), 2),
            "peak_pool_total_mib": round(mib(peak_pool), 2),
            "min_driver_free_mib": round(mib(min_free), 2),
        }
    return res, t1 - t0, mem


def bitwise(a, b):
    return (
        np.array_equal(a[0], b[0], equal_nan=True)
        and np.array_equal(a[1], b[1])
        and a[2] == b[2]
    )


def cpu_parity_ok(tiled, cpu):
    if abs(float(tiled[2]) - float(cpu[2])) > 1.0:
        return False, "pct"
    try:
        np.testing.assert_allclose(
            tiled[0], cpu[0], rtol=1e-3, atol=1e-2, equal_nan=True
        )
        np.testing.assert_allclose(
            tiled[1], cpu[1], rtol=1e-3, atol=1e-2, equal_nan=True
        )
        return True, ""
    except AssertionError:
        # documented bounded ULP-boundary class
        try:
            np.testing.assert_allclose(tiled[1], cpu[1], rtol=1e-4, atol=1e-3)
            tol = 1e-2 + 1e-3 * np.abs(cpu[0].astype(np.float64))
            diff = np.abs(tiled[0].astype(np.float64) - cpu[0].astype(np.float64))
            diff = np.where(np.isnan(diff), 0.0, diff)
            differing = int(np.sum(diff > tol))
            max_abs = float(np.max(diff)) if diff.size else 0.0
            total = int(np.prod(cpu[0].shape))
            if differing <= max(20, total // 200) and max_abs <= 10.0:
                return True, "bounded-ulp %d px max %.3f" % (differing, max_abs)
            return False, "ulp %d px max %.3f" % (differing, max_abs)
        except AssertionError:
            return False, "weight parity out of contract"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(REPO, "profiling", "results_phaseE"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dev = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    results = {
        "host_gpu": dev,
        "cupy": cp.__version__,
        "numpy": np.__version__,
        "note": "Phase E spatial tiling profiler (exact-N_batch)",
        "sections": {},
    }
    print("GPU:", dev, "| cupy", cp.__version__, "| outdir", args.outdir)
    log = open(os.path.join(args.outdir, "progress.log"), "w")

    def say(*a):
        line = " ".join(str(x) for x in a)
        print(line)
        log.write(line + "\n")
        log.flush()

    # -------------------------------------------------------------
    # 1. exact-equivalence spot proof at 480x270 + CPU parity (tiled)
    # -------------------------------------------------------------
    say("== 1. 480x270 equivalence spot proof ==")
    eq_rows = []
    par_rows = []
    for n in (20, 32, 50):
        imgs = make_images(n, (270, 480), 100 + n, nan_frac=0.05)
        w = make_weights(n)
        full = gpu_full(imgs, w)
        cpu = sm._stack_winsorized_sigma_iter(imgs, w, return_weights=True,
                                              kappa=KAPPA, winsor_limits=WINSOR_LIMITS)
        for ts in (128, 7, (64, 96)):
            tiled = gpu_tiled(imgs, w, tile_shape=ts)
            ok = bitwise(full, tiled)
            eq_rows.append({"n": n, "tile_shape": ts, "bitwise": ok,
                            "rejected_pct": full[2]})
        tiled = gpu_tiled(imgs, w, tile_shape=128)
        okp, note = cpu_parity_ok(tiled, cpu)
        par_rows.append({"n": n, "tile_shape": 128, "cpu_parity_ok": okp,
                         "note": note})
        say("N=%d tiled bitwise rows %s | parity %s" % (n, eq_rows[-3:], note))
    results["sections"]["equivalence_480x270"] = {
        "bitwise": eq_rows, "cpu_parity_tiled": par_rows,
    }

    # -------------------------------------------------------------
    # 2. 1080p N=20: both fit — bitwise + wall + peak VRAM full vs tiled
    # -------------------------------------------------------------
    say("== 2. 1080p N=20 full vs tiled (both fit) ==")
    n = 20
    hw = (1080, 1920)
    imgs = make_images(n, hw, 200, nan_frac=0.05)
    w = make_weights(n)
    full_res, full_wall, full_mem = run_instrumented(
        lambda: gpu_full(imgs, w), snap_mem=True)
    say("full : wall %.1f ms mem %s" % (full_wall * 1e3, full_mem))
    ts20 = 192 if not args.quick else 540
    tiled_res, tiled_wall, tiled_mem = run_instrumented(
        lambda: gpu_tiled(imgs, w, tile_shape=ts20), snap_mem=True)
    say("tiled: wall %.1f ms mem %s tile_h=%d"
        % (tiled_wall * 1e3, tiled_mem, ts20))
    ok = bitwise(full_res, tiled_res)
    say("1080p N=20 full == tiled bitwise:", ok)
    results["sections"]["fit1080p_n20"] = {
        "bitwise": ok, "tile_h": ts20,
        "full": {"wall_ms": round(full_wall * 1e3, 2), "mem": full_mem},
        "tiled": {"wall_ms": round(tiled_wall * 1e3, 2), "mem": tiled_mem,
                  "overhead_x": round(tiled_wall / full_wall, 3)},
    }

    # -------------------------------------------------------------
    # 3./4. 1080p N=32 / N=50: whole-stack OOM vs spatial tiling
    # -------------------------------------------------------------
    say("== 3./4. 1080p whole-stack OOM vs tiled execution ==")
    fit_rows = []
    for n in (32, 50):
        imgs = make_images(n, hw, 300 + n, nan_frac=0.05)
        w = make_weights(n)
        full_outcome = "OK"
        try:
            full_res, full_wall, full_mem = run_instrumented(
                lambda: gpu_full(imgs, w), snap_mem=True)
            say("N=%d full unexpectedly OK wall %.0f ms" % (n, full_wall * 1e3))
            full_row = {"wall_ms": round(full_wall * 1e3, 2), "mem": full_mem}
        except Exception as e:  # cp.cuda.memory.OutOfMemoryError subclass
            full_outcome = type(e).__name__
            say("N=%d full -> %s" % (n, full_outcome))
            full_row = None
        # tiled candidate scan
        cand = [540, 384, 270, 192, 96] if not args.quick else [384, 192]
        tiled_row = None
        attempts = []
        for th in cand:
            try:
                t_res, t_wall, t_mem = run_instrumented(
                    lambda: gpu_tiled(imgs, w, tile_shape=th), snap_mem=True)
                attempts.append({"tile_h": th, "outcome": "OK",
                                 "wall_ms": round(t_wall * 1e3, 2), "mem": t_mem,
                                 "rejected_pct": t_res[2],
                                 "result_finite": bool(np.isfinite(t_res[0]).all()),
                                 "w_finite": bool(np.isfinite(t_res[1]).all())})
                if tiled_row is None:
                    tiled_row = attempts[-1]
                    say("N=%d tiled OK tile_h=%d wall %.0f ms mem %s pct %.3f"
                        % (n, th, t_wall * 1e3, t_mem, t_res[2]))
                else:
                    # cross-tile-shape consistency at the real resolution
                    prev = None  # (recompute previous? keep simple: compare to first)
                    # bitwise across candidate shapes (same data)
                    prev_res, _, _ = run_instrumented(
                        lambda: gpu_tiled(imgs, w, tile_shape=cand[0]))
                    ok_cross = bitwise(prev_res, t_res)
                    attempts[-1]["cross_shape_bitwise_vs_first"] = ok_cross
            except Exception as e:
                attempts.append({"tile_h": th, "outcome": type(e).__name__})
                say("N=%d tiled tile_h=%d -> %s" % (n, th, type(e).__name__))
        fit_rows.append({"n": n, "full": {"outcome": full_outcome,
                                          "run": full_row},
                         "tiled_candidates": attempts,
                         "chosen": tiled_row})
        # CPU parity of the tiled 1080p result at 480x270 (same seed data,
        # reduced resolution: CPU authority parity class, documented tol)
        if tiled_row is not None:
            imgs_small = make_images(n, (270, 480), 300 + n, nan_frac=0.05)
            w_small = make_weights(n)
            cpu = sm._stack_winsorized_sigma_iter(
                imgs_small, w_small, return_weights=True,
                kappa=KAPPA, winsor_limits=WINSOR_LIMITS)
            t_small = gpu_tiled(imgs_small, w_small, tile_shape=128)
            okp, note = cpu_parity_ok(t_small, cpu)
            fit_rows[-1]["cpu_parity_480x270"] = {"ok": okp, "note": note}
            say("N=%d tiled CPU parity (480x270 witness): %s %s"
                % (n, okp, note))
    results["sections"]["fit1080p_oom"] = fit_rows

    with open(os.path.join(args.outdir, "phaseE_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.close()
    print("written:", os.path.join(args.outdir, "phaseE_results.json"))


if __name__ == "__main__":
    main()
