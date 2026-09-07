#!/usr/bin/env python
"""Continuation: 1080p N=50 whole-stack OOM vs spatial tiling (phase E)."""
import os, sys, time, json
os.environ.setdefault("ZSSS_GPU_PROFILE", "1")
REPO = "/home/tristan/.openclaw/workspace/projects/zeseestarstacker"
sys.path.insert(0, REPO)
import numpy as np
import cupy as cp
import seestar.core.stack_gpu as sg
from seestar.core import stack_methods as sm

KAPPA, LIMITS = 2.5, (0.05, 0.05)
MB = 1024 * 1024
def mib(b): return b / (1024.0 * 1024.0)

def make_images(n, hw, seed, nan_frac, spike=0.03):
    rng = np.random.default_rng(seed)
    shape = (n,) + tuple(hw)
    arr = rng.standard_normal(shape, dtype=np.float32) * 20.0 + 1000.0
    if nan_frac > 0:
        arr[rng.random(shape, dtype=np.float32) < nan_frac] = np.nan
    arr = arr + np.where(rng.random(shape, dtype=np.float32) < spike,
                         np.float32(400.0), np.float32(0.0))
    return [arr[i] for i in range(n)]

def make_weights(n, seed=7):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.4, 1.6, size=n).astype(np.float32)

def gpu_full(images, weights=None, **kw):
    args = dict(kappa=KAPPA, winsor_limits=LIMITS, apply_rewinsor=True,
                max_iters=5, kappa_decay=0.9); args.update(kw)
    return sg.stack_winsorized_sigma_gpu(images, weights, return_weights=True, **args)

def gpu_tiled(images, weights=None, tile_shape=None, **kw):
    args = dict(kappa=KAPPA, winsor_limits=LIMITS, apply_rewinsor=True,
                max_iters=5, kappa_decay=0.9); args.update(kw)
    return sg.stack_winsorized_sigma_gpu_tiled(images, weights, return_weights=True,
                                               tile_shape=tile_shape, **args)

def run_instrumented(fn, snap_mem=False):
    sg._ensure_probe(); p = sg._PROBE; p.reset(); p.snap_mem = snap_mem
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    try:
        res = fn()
    finally:
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter(); p.snap_mem = False
    mem = None
    if snap_mem and p.mem:
        mem = {"n_boundaries": len(p.mem),
               "peak_live_mib": round(mib(max(int(m[3]) for m in p.mem)), 2),
               "peak_pool_total_mib": round(mib(max(int(m[5]) for m in p.mem)), 2),
               "min_driver_free_mib": round(mib(min(int(m[1]) for m in p.mem)), 2)}
    return res, t1 - t0, mem

outdir = os.path.join(REPO, "profiling", "results_phaseE")
log = open(os.path.join(outdir, "progress.log"), "a")
def say(*a):
    line = " ".join(str(x) for x in a); print(line); log.write(line + "\n"); log.flush()

n = 50
imgs = make_images(n, (1080, 1920), 350, nan_frac=0.05)
w = make_weights(n)
full_outcome = "OK"; full_row = None
try:
    fr, fw, fm = run_instrumented(lambda: gpu_full(imgs, w), snap_mem=True)
    say("N=50 full OK wall %.0f ms mem %s" % (fw * 1e3, fm))
    full_row = {"wall_ms": round(fw * 1e3, 2), "mem": fm}
except Exception as e:
    full_outcome = type(e).__name__
    say("N=50 full ->", full_outcome)

attempts, tiled_row, first_res = [], None, None
for th in (384, 270, 192, 96):
    try:
        tr, tw, tm = run_instrumented(lambda: gpu_tiled(imgs, w, tile_shape=th),
                                      snap_mem=True)
        row = {"tile_h": th, "outcome": "OK", "wall_ms": round(tw * 1e3, 2),
               "mem": tm, "rejected_pct": tr[2],
               "result_finite": bool(np.isfinite(tr[0]).all()),
               "w_finite": bool(np.isfinite(tr[1]).all())}
        if first_res is not None:
            row["cross_shape_bitwise_vs_first"] = bool(
                np.array_equal(tr[0], first_res[0], equal_nan=True)
                and np.array_equal(tr[1], first_res[1]) and tr[2] == first_res[2])
        else:
            first_res = tr
        attempts.append(row)
        if tiled_row is None:
            tiled_row = row
        say("N=50 tiled tile_h=%d OK wall %.0f ms %s pct %.4f%s" % (
            th, tw * 1e3, tm, tr[2],
            "" if "cross_shape_bitwise_vs_first" not in row else
            " cross_bitwise=%s" % row["cross_shape_bitwise_vs_first"]))
    except Exception as e:
        attempts.append({"tile_h": th, "outcome": type(e).__name__})
        say("N=50 tiled tile_h=%d -> %s" % (th, type(e).__name__))

# CPU parity witness 480x270 same seed data
par = None
if tiled_row is not None:
    imgs_s = make_images(n, (270, 480), 350, nan_frac=0.05)
    w_s = make_weights(n)
    cpu = sm._stack_winsorized_sigma_iter(imgs_s, w_s, return_weights=True,
                                          kappa=KAPPA, winsor_limits=LIMITS)
    ts = gpu_tiled(imgs_s, w_s, tile_shape=128)
    okp = (abs(float(ts[2]) - float(cpu[2])) <= 1.0
           and np.allclose(ts[0], cpu[0], rtol=1e-3, atol=1e-2, equal_nan=True)
           and np.allclose(ts[1], cpu[1], rtol=1e-3, atol=1e-2, equal_nan=True))
    par = {"ok": bool(okp), "pct_cpu": float(cpu[2]), "pct_tiled": float(ts[2])}
    say("N=50 tiled CPU parity (480x270 witness):", okp)

res = {"n": 50, "full": {"outcome": full_outcome, "run": full_row},
       "tiled_candidates": attempts, "chosen": tiled_row, "cpu_parity_480x270": par}
with open(os.path.join(outdir, "phaseE_n50.json"), "w") as f:
    json.dump(res, f, indent=2)
say("written phaseE_n50.json")
log.close()
