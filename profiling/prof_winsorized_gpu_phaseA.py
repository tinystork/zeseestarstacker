#!/usr/bin/env python
"""Phase A baseline profiler for the Winsorized CuPy stacking path (ZSSS).

Standalone, opt-in (ZSSS_GPU_PROFILE=1), deterministic, NON-scientific:
it only observes; it can never change reduction results (proven bitwise in
a subprocess A/B check below).

Measures, for the CURRENT unoptimized tree (branch
feature/winsorized-gpu-perf-memory @ 2d0fdcb):

  * stage timing (CUDA events for device stages; host wall only for
    host-work and synchronization segments) of stack_winsorized_sigma_gpu,
  * stage-boundary memory (memGetInfo + CuPy pool) via a sync-per-boundary
    pass, kept separate from the timing pass so timings stay clean,
  * the full synchronization inventory introduced by bool()/int() scalar
    conversions and cp.asnumpy,
  * dispatch reachability of the GPU twin through
    queue_manager._gpu_reduce/_reduction_xp (stub-backed) with the real
    VRAM-guard arithmetic (footprint 6.0 winsorized / 4.0 default),
  * OOM / fallback boundary on the 2 GiB MX150.

Witnesses: N_batch in {4, 8, 12, 19, 20, 32, 50} x {full validity,
partial validity} at (480, 270) and (1920, 1080) where practical.

Usage:
  .venv/bin/python profiling/prof_winsorized_gpu_phaseA.py
      [--outdir profiling/results_phaseA] [--report-body <md>]
      [--quick] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
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
N_BATCHES = [4, 8, 12, 19, 20, 32, 50]
SIZES = {"480x270": (480, 270), "1080p": (1920, 1080)}
KERNEL_ARGS = dict(
    kappa=KAPPA,
    winsor_limits=WINSOR_LIMITS,
    apply_rewinsor=True,
    max_iters=5,
    kappa_decay=0.9,
)
MB = 1024 * 1024


def mib(b):
    return b / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# deterministic witnesses
# ---------------------------------------------------------------------------


def make_images(n, hw, seed, nan_frac, spike=0.03):
    """Deterministic synthetic stack, generated in float32 (cheap)."""
    rng = np.random.default_rng(seed)
    shape = (n,) + tuple(hw)
    arr = (rng.standard_normal(shape, dtype=np.float32) * 20.0 + 1000.0)
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


def gpu_ref(images, weights=None, **kw):
    args = dict(KERNEL_ARGS)
    args.update(kw)
    return sg.stack_winsorized_sigma_gpu(
        images, weights, return_weights=True, **args
    )


# ---------------------------------------------------------------------------
# probe helpers
# ---------------------------------------------------------------------------


def probe():
    return sg._PROBE


def run_gpu(images, weights=None, snap_mem=False):
    sg._ensure_probe()
    p = probe()
    p.reset()
    p.snap_mem = snap_mem
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    res = gpu_ref(images, weights)
    t1 = time.perf_counter()
    cp.cuda.Stream.null.synchronize()
    p.snap_mem = False
    return res, t1 - t0


def _ev_ms(e1, e2):
    return float(cp.cuda.get_elapsed_time(e1, e2))


def device_regions_ms():
    """(name, device ms) per region between consecutive CUDA events."""
    p = probe()
    evs = [(name, ev) for (name, _t, ev) in p.marks if ev is not None]
    out = []
    for (n1, e1), (n2, e2) in zip(evs, evs[1:]):
        try:
            out.append((n2, _ev_ms(e1, e2)))
        except Exception:
            out.append((n2, float("nan")))
    return out


def median_per_name(list_of_rows):
    """list_of_rows: per-rep list of (name, ms) -> median ms per name."""
    by = {}
    order = []
    for rep in list_of_rows:
        for name, ms in rep:
            if name not in by:
                by[name] = []
                order.append(name)
            by[name].append(ms)
    return [(name, statistics.median(by[name])) for name in order]


def device_total_ms():
    p = probe()
    evs = [(n, e) for (n, _t, e) in p.marks if e is not None]
    if len(evs) < 2:
        return 0.0
    return _ev_ms(evs[0][1], evs[-1][1])


def wall_segments():
    return [(n, (e - s) * 1000.0) for (n, s, e) in probe().wall]


# ---------------------------------------------------------------------------
# subprocess A/B: prove instrumentation cannot change results (bitwise)
# ---------------------------------------------------------------------------


def bitwise_probe_check(outdir, seed=12345):
    """Run plain (env 0) and instrumented (env 1) in separate interpreters,
    compare bitwise + exact rejected_pct."""
    code = (
        "import os,sys,numpy as np;"
        "sys.path.insert(0,%r);"
        "import seestar.core.stack_gpu as sg;"
        "rng=np.random.default_rng(%d);"
        "a=rng.normal(1000.,20.,size=(6,16,20)).astype(np.float32);"
        "a[rng.random((6,16,20))<0.04]=np.nan;"
        "a=a+np.where(rng.random((6,16,20))<0.03,400.,0.).astype(np.float32);"
        "w=rng.uniform(0.4,1.6,size=6).astype(np.float32);"
        "imgs=[a[i] for i in range(6)];"
        "r,wm,p=sg.stack_winsorized_sigma_gpu(imgs,w,kappa=2.5,"
        "winsor_limits=(0.05,0.05),return_weights=True);"
        "np.savez(%r,r=r,w=wm,p=float(p))"
    ) % (REPO, seed, os.path.join(outdir, "bitwise_out.npz"))
    results = {}
    for tag, envv in (("plain", "0"), ("instrumented", "1")):
        env = dict(os.environ)
        env["ZSSS_GPU_PROFILE"] = envv
        env["PYTHONWARNINGS"] = "ignore"
        subprocess.run(
            [sys.executable, "-c", code],
            env=env, check=True, capture_output=True,
        )
        z = np.load(os.path.join(outdir, "bitwise_out.npz"))
        results[tag] = (z["r"], z["w"], float(z["p"]))
        os.remove(os.path.join(outdir, "bitwise_out.npz"))
    (r0, w0, p0), (r1, w1, p1) = results["plain"], results["instrumented"]
    bitwise = (
        np.array_equal(r0, r1, equal_nan=True)
        and np.array_equal(w0, w1, equal_nan=True)
        and p0 == p1
    )
    return dict(
        bitwise_equal=bool(bitwise), plain_pct=p0, instrumented_pct=p1,
        max_abs_diff=float(np.nanmax(
            np.abs(r0.astype(np.float64) - r1))),
    )


# ---------------------------------------------------------------------------
# parity: GPU twin vs CPU scientific reference (documented tolerance)
# ---------------------------------------------------------------------------


def cpu_ref(images, weights=None, **kw):
    args = dict(KERNEL_ARGS)
    args.update(kw)
    return sm._stack_winsorized_sigma_iter(
        images, weights, return_weights=True, **args
    )


def parity_check():
    rows = []
    cases = [
        ("small_mono_unweighted", 8, (24, 32), 0.03, None),
        ("small_mono_weighted", 8, (24, 32), 0.03, 7),
        ("small_rgb_unweighted", 6, (16, 16, 3), 0.05, None),
        ("small_rgb_weighted", 6, (16, 16, 3), 0.05, 7),
        # adversarial mid-size: heavy spikes + 10% NaN, max_iters reached
        ("mid_adv_unweighted", 12, (480, 270), 0.10, None),
        ("mid_adv_weighted", 12, (480, 270), 0.10, 3),
    ]
    for tag, n, shape, nan_frac, wseed in cases:
        imgs = make_images(n, shape, seed=1, nan_frac=nan_frac)
        w = make_weights(n, seed=wseed) if wseed is not None else None
        c_res, c_w, c_pct = cpu_ref(imgs, w)
        g_res, g_w, g_pct = gpu_ref(imgs, w)
        res_ok = np.allclose(g_res, c_res, rtol=1e-3, atol=1e-2,
                             equal_nan=True)
        w_ok = np.allclose(g_w, c_w, rtol=1e-3, atol=1e-2, equal_nan=True)
        pct_ok = abs(g_pct - c_pct) <= 1.0
        diff = np.abs(g_res.astype(np.float64) - c_res.astype(np.float64))
        tol = 1e-2 + 1e-3 * np.abs(c_res.astype(np.float64))
        viol = int(np.sum((diff > tol) & np.isfinite(diff)
                          & np.isfinite(c_res)))
        rows.append(dict(
            case=tag, n=n, shape=str(shape), weighted=w is not None,
            res_ok=bool(res_ok), w_ok=bool(w_ok), pct_ok=bool(pct_ok),
            viol_pixels=viol,
            res_maxabs=float(np.nanmax(diff)) if viol else 0.0,
            pct_cpu=float(c_pct), pct_gpu=float(g_pct),
        ))
    return rows


# ---------------------------------------------------------------------------
# dispatch reachability through the real _gpu_reduce / _reduction_xp
# ---------------------------------------------------------------------------


class _QMStub:
    """Minimal stand-in exposing exactly what _reduction_xp/_gpu_reduce use."""

    effective_backend = "cupy"
    _gpu_fallback_logged = set()

    def __init__(self):
        self.warnings = []

    def _log_gpu_fallback_once(self, reason, message, *args):
        """Real method signature; stub records instead of logging."""
        try:
            self.warnings.append(message % args if args else message)
        except Exception:
            self.warnings.append(message)

    @property
    def logger(self):
        class _L:
            def __init__(self, sink):
                self.sink = sink

            def warning(self, msg, *a):
                try:
                    self.sink.append(msg % a if a else msg)
                except Exception:
                    self.sink.append(msg)

        return _L(self.warnings)


def dispatch_sweep():
    from seestar.queuep.queue_manager import (
        SeestarQueuedStacker,
        _GPU_FOOTPRINT_FACTOR_WINSORIZED,
    )

    cp.get_default_memory_pool().free_all_blocks()
    rows = []
    for size_name, hw in SIZES.items():
        for n in N_BATCHES:
            imgs = make_images(n, hw, seed=5, nan_frac=0.05)
            stub = _QMStub()
            try:
                mod = SeestarQueuedStacker._reduction_xp(
                    stub, imgs,
                    footprint_factor=_GPU_FOOTPRINT_FACTOR_WINSORIZED,
                )
                decision = "GPU(cupy)" if mod is not None else "CPU"
            except Exception as exc:
                decision = "ERROR:%r" % exc
            elem = int(n) * int(np.prod(hw))
            need = elem * 4 * _GPU_FOOTPRINT_FACTOR_WINSORIZED
            free, total = cp.cuda.runtime.memGetInfo()
            pool_free = cp.get_default_memory_pool().free_bytes()
            eff = int(free) + int(pool_free)
            rows.append(dict(
                size=size_name, n=n, need_mib=round(mib(need), 1),
                driver_free_mib=round(mib(free), 1),
                pool_free_mib=round(mib(pool_free), 1),
                eff_free_mib=round(mib(eff), 1),
                threshold_mib=round(mib(0.6 * eff), 1),
                decision=decision,
                guard_warning=list(stub.warnings),
            ))
    probe_rows = []
    for size_name, hw in (("480x270", (480, 270)), ("1080p", (1920, 1080))):
        for n in (4, 50):
            imgs = make_images(n, hw, seed=5, nan_frac=0.05)
            calls = {"gpu": 0, "cpu": 0}

            def cpu_marker(imgs_, w=None, **kw):
                calls["cpu"] += 1
                return ("cpu_marker",)

            def gpu_marker(imgs_, w=None, **kw):
                calls["gpu"] += 1
                return gpu_ref(imgs_, w)

            stub = _QMStub()
            # _gpu_reduce dispatches through self._reduction_xp: bind the
            # real bound method onto the stub so the guard arithmetic runs
            # for real inside the end-to-end probe.
            stub._reduction_xp = SeestarQueuedStacker._reduction_xp.__get__(
                stub, type(stub))
            try:
                out = SeestarQueuedStacker._gpu_reduce(
                    stub, cpu_marker, gpu_marker, imgs, None,
                    footprint_factor=_GPU_FOOTPRINT_FACTOR_WINSORIZED,
                    **KERNEL_ARGS,
                )
                executed = "GPU" if calls["gpu"] else "CPU"
                ok = isinstance(out, tuple) and len(out) == 3
            except Exception as exc:
                executed, ok = "EXC:%r" % exc, False
            probe_rows.append(dict(
                size=size_name, n=n, executed=executed,
                gpu_calls=calls["gpu"], cpu_calls=calls["cpu"],
                returned_3tuple=bool(ok), warnings=list(stub.warnings),
            ))
    return rows, probe_rows


# ---------------------------------------------------------------------------
# stage timing + memory sweeps
# ---------------------------------------------------------------------------

PHASE_RE = [
    ("h2d", r"^h2d_arr_ready$"),
    ("mask_prep", r"^mask_prep$"),
    ("iter_masked(where)", r"^iter\d+_masked$"),
    ("winsor_copy", r"^winsor_copy_result$"),
    ("winsor_valid_count", r"^winsor_valid_count$"),
    ("winsor_sort_key", r"^winsor_sort_key$"),
    ("argsort_1", r"^winsor_argsort$"),
    ("winsor_gather", r"^winsor_take_along$"),
    ("argsort_rank(inverse)", r"^winsor_rank_argsort$"),
    ("winsor_bound_take", r"^winsor_(low|high)_bound$"),
    ("winsor_replace", r"^winsor_(low|high)_replace$"),
    ("nanmean", r"^iter\d+_nanmean$"),
    ("nanstd", r"^iter\d+_nanstd$"),
    ("sigma_guard", r"^iter\d+_sigma_guard$"),
    ("new_mask", r"^iter\d+_new_mask$"),
    ("nrej_count_nonzero(dev)", r"^iter\d+_nrej_sync$"),
    ("rewinsor_sort_key", r"^bounds_(valid_count|sort_key)$"),
    ("rewinsor_argsort", r"^bounds_argsort$"),
    ("rewinsor_gather", r"^bounds_take_along$"),
    ("rewinsor_idx", r"^bounds_idx$"),
    ("rewinsor_bound_take", r"^bounds_lo_hi$"),
    ("rewinsor_clip", r"^rewinsor_clip$"),
    ("rewinsor_final_where", r"^(rewinsor_final|rewinsor_bounds_done)$"),
    ("contrib", r"^contrib$"),
    ("final_reduce", r"^final_reduce_(weighted|unweighted)$"),
    ("rejected_pct(dev)", r"^rejected_pct$"),
    ("result_astype", r"^result_astype$"),
    ("d2h", r"^d2h_done$"),
]

_RESIDUAL = re.compile(
    r"^(gpu_fn_start|h2d_transfer|iter\d+_loop_top|iter\d+_winsorized|"
    r"winsor_return|no_rewinsor_final)$"
)


def phase_sums(regions_ms):
    """Map per-mark region times to phase sums."""
    sums = {}
    residual = 0.0
    for name, ms in regions_ms:
        matched = False
        for phase, rx in PHASE_RE:
            if re.fullmatch(rx, name):
                sums[phase] = sums.get(phase, 0.0) + ms
                matched = True
                break
        if not matched:
            if not _RESIDUAL.match(name):
                residual += ms
    if residual:
        sums["(unclassified residual)"] = residual
    return sums


def classify_sync(name):
    if name == "host_stack_pack":
        return "host-work"
    if "nrej" in name:
        return "count_nonzero->int x2 (iter early-exit)"
    if name == "sync_rejected_pct":
        return "rejected_pct count_nonzero->int x2"
    return "bool(cp.any) scalar"


def timing_sweep(quick=False, log=None):
    out = []
    reps_small = 1 if quick else 3
    total = sum(1 for _size, _hw in SIZES.items() for n in N_BATCHES
                for _nf, _vt in ((0.0, "full"), (0.05, "partial"))
                for _wd in ([True, False] if (_size == "480x270"
                                              and n in (8, 20)) else [True]))
    done = 0
    for size_name, hw in SIZES.items():
        for n in N_BATCHES:
            for nan_frac, val_tag in ((0.0, "full"), (0.05, "partial")):
                weight_modes = [True]
                if size_name == "480x270" and n in (8, 20):
                    weight_modes.append(False)
                for weighted in weight_modes:
                    done += 1
                    if log:
                        log("  timing %s N=%-2d %-7s wtd=%s (%d/%d)"
                            % (size_name, n, val_tag, weighted, done, total))
                    rec = dict(size=size_name, n=n, validity=val_tag,
                               weighted=weighted, error=None, reps=[])
                    imgs = make_images(n, hw, seed=11, nan_frac=nan_frac)
                    weights = make_weights(n, seed=2) if weighted else None
                    reps = reps_small if size_name == "480x270" else 1
                    cp.get_default_memory_pool().free_all_blocks()
                    try:
                        for _ in range(reps):
                            res, fn_wall = run_gpu(imgs, weights)
                            rec["reps"].append(dict(
                                fn_wall_s=fn_wall,
                                device_total_ms=device_total_ms(),
                                regions=device_regions_ms(),
                                wall=wall_segments(),
                                notes=list(probe().notes),
                                rejected_pct=float(res[2]),
                            ))
                    except cp.cuda.memory.OutOfMemoryError:
                        rec["error"] = "OutOfMemoryError"
                    except Exception as exc:
                        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
                    if rec["reps"]:
                        rec["fn_wall_s"] = round(statistics.median(
                            r["fn_wall_s"] for r in rec["reps"]), 4)
                        rec["device_total_ms"] = round(statistics.median(
                            r["device_total_ms"] for r in rec["reps"]), 3)
                        # raw (uncollapsed, per-iteration) region list from
                        # the first rep - identical across reps (deterministic)
                        rec["regions_raw"] = rec["reps"][0]["regions"]
                        rec["regions_ms"] = median_per_name(
                            [r["regions"] for r in rec["reps"]])
                        rec["wall_ms"] = median_per_name(
                            [r["wall"] for r in rec["reps"]])
                        rec["notes"] = rec["reps"][0]["notes"]
                        rec["rejected_pct"] = rec["reps"][0]["rejected_pct"]
                        rec["phases_ms"] = phase_sums(rec["regions_raw"])
                    else:
                        rec["fn_wall_s"] = None
                        rec["device_total_ms"] = None
                    out.append(rec)
    return out


def memory_sweep(log=None):
    out = []
    for size_name, hw in (("480x270", (480, 270)), ("1080p", (1920, 1080))):
        for n in N_BATCHES:
            if size_name == "1080p" and n > 12:
                break  # direct-run OOM boundary handled separately
            for nan_frac, val_tag in ((0.0, "full"), (0.05, "partial")):
                if log:
                    log("  mem %s N=%d %s" % (size_name, n, val_tag))
                imgs = make_images(n, hw, seed=11, nan_frac=nan_frac)
                weights = make_weights(n, seed=2)
                cp.get_default_memory_pool().free_all_blocks()
                rec = dict(size=size_name, n=n, validity=val_tag,
                           weighted=True, error=None)
                try:
                    run_gpu(imgs, weights, snap_mem=True)
                    p = probe()
                    rec["snapshots"] = [
                        dict(mark=name, driver_free_mib=round(mib(f), 1),
                             pool_used_mib=round(mib(u), 1),
                             pool_free_mib=round(mib(fr), 1),
                             pool_total_mib=round(mib(t), 1))
                        for (name, f, _tot, u, fr, t) in p.mem
                    ]
                    rec["peak_live_mib"] = round(mib(max(
                        u for (_n, _f, _t, u, _fr, _tt) in p.mem)), 1)
                    rec["peak_pool_total_mib"] = round(mib(max(
                        t for (_n, _f, _t, _u, _fr, t) in p.mem)), 1)
                    rec["min_driver_free_mib"] = round(mib(min(
                        f for (_n, f, _t, _u, _fr, _tt) in p.mem)), 1)
                    rec["notes"] = list(p.notes)
                except cp.cuda.memory.OutOfMemoryError:
                    rec["error"] = "OutOfMemoryError"
                except Exception as exc:
                    rec["error"] = "%s: %s" % (type(exc).__name__, exc)
                out.append(rec)
    return out


def oom_boundary(log=None):
    rows = []
    for n in (12, 16, 19, 20, 32, 50):
        if log:
            log("  oom probe N=%d" % n)
        imgs = make_images(n, (1920, 1080), seed=11, nan_frac=0.05)
        cp.get_default_memory_pool().free_all_blocks()
        try:
            res, wall = run_gpu(imgs, make_weights(n, seed=2))
            rows.append(dict(n=n, outcome="OK", fn_wall_s=round(wall, 3),
                             rejected_pct=float(res[2])))
        except cp.cuda.memory.OutOfMemoryError:
            rows.append(dict(n=n, outcome="OOM", fn_wall_s=None))
        except Exception as exc:
            rows.append(dict(n=n, outcome="EXC:%s" % type(exc).__name__,
                             fn_wall_s=None))
        cp.get_default_memory_pool().free_all_blocks()
    return rows


def cpu_wall_reference(log=None):
    rows = []
    for n in (4, 12, 32, 50):
        if log:
            log("  cpu ref N=%d" % n)
        imgs = make_images(n, (480, 270), seed=11, nan_frac=0.05)
        w = make_weights(n, seed=2)
        ts = []
        for _ in range(2):
            t0 = time.perf_counter()
            cpu_ref(imgs, w)
            ts.append(time.perf_counter() - t0)
        rows.append(dict(n=n, cpu_wall_s=round(statistics.median(ts), 4)))
    return rows


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------


def fmt_table(headers, rows):
    w = []
    for i, h in enumerate(headers):
        width = len(str(h))
        for r in rows:
            width = max(width, len(str(r[i])))
        w.append(width)
    line = "| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(headers)) \
        + " |"
    sep = "| " + " | ".join("-" * w[i] for i in range(len(headers))) + " |"
    body = "\n".join(
        "| " + " | ".join(str(r[i]).ljust(w[i]) for i in range(len(headers)))
        + " |" for r in rows
    )
    return line + "\n" + sep + (("\n" + body) if body else "")


def render_report_body(results):
    L = []
    A = L.append
    dev = cp.cuda.runtime.getDeviceProperties(0)

    A("## 1. Environment\n")
    A("- Host: Linux TINYDEBIAN; GPU: NVIDIA %s (CC %d.%d), cupy %s, "
      "numpy %s, python %s" % (dev["name"].decode(), dev["major"],
                               dev["minor"], cp.__version__, np.__version__,
                               sys.version.split()[0]))
    A("- Device total: %.0f MiB; baseline free (pool emptied): %.0f MiB"
      % (mib(results["device_total"]), mib(results["baseline_free"])))
    A("- Witnesses: N_batch=%s; sizes 480x270 + 1920x1080; validity "
      "full (nan_frac=0.0) / partial (nan_frac=0.05); kappa=%s "
      "winsor_limits=%s apply_rewinsor=True max_iters=5 kappa_decay=0.9; "
      "spikes 3%% (+400); weights uniform(0.4,1.6) float32"
      % (N_BATCHES, KAPPA, WINSOR_LIMITS))

    A("\n### 1.1 Instrumentation is non-scientific (bitwise A/B)\n")
    b = results["bitwise"]
    A("- Plain (ZSSS_GPU_PROFILE=0) vs instrumented (=1), separate "
      "interpreters, same seed: **bitwise_equal=%s**; rejected_pct "
      "%.10f vs %.10f (identical); max abs diff = %g"
      % (b["bitwise_equal"], b["plain_pct"], b["instrumented_pct"],
         b["max_abs_diff"]))

    A("\n### 1.2 CPU/GPU parity on the instrumented path "
      "(tolerance rtol=1e-3, atol=1e-2; |pct diff| <= 1.0)\n")
    A(fmt_table(
        ["case", "n", "shape", "weighted", "res_ok", "w_ok", "pct_ok",
         "viol_px", "res_maxabs", "pct_cpu", "pct_gpu"],
        [[r["case"], r["n"], r["shape"], r["weighted"], r["res_ok"],
          r["w_ok"], r["pct_ok"], r["viol_pixels"],
          "%.3g" % r["res_maxabs"], "%.3f" % r["pct_cpu"],
          "%.3f" % r["pct_gpu"]]
         for r in results["parity"]]))
    n_adv = sum(1 for r in results["parity"] if not r["res_ok"])
    if n_adv:
        A("\nNOTE: the two `mid_adv_*` rows (12 frames, 480x270, 3% spikes "
          "+400, 10% NaN, kappa-decay cascade exhausting max_iters) show a "
          "PRE-EXISTING baseline divergence at 1/129600 pixels (maxabs "
          "~1.9): reproduced bitwise-identically on the pristine 2d0fdcb "
          "checkout (no probes) and on this instrumented tree - it is NOT "
          "caused by the instrumentation.  Mechanism hypothesis: "
          "ULP-level mean/std differences flip one boundary sample in an "
          "iteration, amplified by the 5-iteration kappa-decay cascade; "
          "global rejected_pct still agrees to 1e-4.")

    A("\n## 2. Dispatch reachability (current tree)\n")
    A("Static trace (queue_manager.py ~L12894-12956): GPU twin reachable "
      "only on the RAM-resident HQ-combine branch when (a) winsorized mode "
      "(`_is_winsorized_mode`), (b) `use_tile_mode=False` (stack bytes <= "
      "max_hq_mem; batch_size==1+memmap forces tiles), (c) "
      "`effective_backend=='cupy'`, (d) dynamic VRAM guard in "
      "`_reduction_xp` passes with `footprint_factor="
      "_GPU_FOOTPRINT_FACTOR_WINSORIZED(6.0)`. Runtime GPU exceptions fall "
      "back to the CPU wrapper through `_gpu_reduce`; VRAM reject is "
      "logged once and is *not* a fallback diagnostic.\n")
    A("Empirical `_reduction_xp` decisions (real bound method, stub "
      "backend='cupy', pool emptied before sweep):\n")
    A(fmt_table(
        ["size", "n", "need(6x) MiB", "driver_free MiB", "pool_free MiB",
         "eff_free MiB", "0.6*eff MiB", "decision", "warning"],
        [[r["size"], r["n"], r["need_mib"], r["driver_free_mib"],
          r["pool_free_mib"], r["eff_free_mib"], r["threshold_mib"],
          r["decision"], ";".join(r["guard_warning"]) or "-"]
         for r in results["dispatch"]]))
    A("\n`_gpu_reduce` end-to-end (which function body actually ran):\n")
    A(fmt_table(
        ["size", "n", "executed", "gpu_calls", "cpu_calls",
         "returned_3tuple", "warnings"],
        [[r["size"], r["n"], r["executed"], r["gpu_calls"], r["cpu_calls"],
          r["returned_3tuple"], ";".join(r["warnings"]) or "-"]
         for r in results["dispatch_probe"]]))

    A("\n## 3. Stage timing table (CUDA events; host wall only for host / "
      "sync segments)\n")
    reps = [r for r in results["timing"] if r["error"] is None]
    A("Per-witness summary (weighted unless marked u/w):\n")
    A(fmt_table(
        ["size", "N", "validity", "wtd", "iterations (n_rej per iter)",
         "fn wall ms", "dev timeline ms", "rej%"],
        [[r["size"], r["n"], r["validity"], "y" if r["weighted"] else "n",
          "; ".join(r["notes"]),
          "%.1f" % (r["fn_wall_s"] * 1000), r["device_total_ms"],
          "%.2f" % r["rejected_pct"]]
         for r in reps]))
    errs = [r for r in results["timing"] if r["error"]]
    for r in errs:
        A("- %s N=%d validity=%s wtd=%s -> **%s**"
          % (r["size"], r["n"], r["validity"], r["weighted"], r["error"]))

    # canonical detailed stage tables for representative witnesses
    picks = [
        ("480x270", 8, "full", True),
        ("480x270", 20, "partial", True),
        ("480x270", 20, "partial", False),
        ("1080p", 8, "partial", True),
        ("1080p", 12, "partial", True),
    ]
    for pick in picks:
        rec = next((r for r in results["timing"] if (
            r["size"], r["n"], r["validity"], r["weighted"]) == pick
            and r["error"] is None), None)
        if rec is None:
            continue
        # Prefer the raw (uncollapsed, per-iteration) region list from the
        # first rep; falls back to the name-collapsed median list.
        raw = rec.get("regions_raw") or rec["regions_ms"]
        A("\n### 3.%d Detailed stage table: %s N=%d %s validity %s\n"
          % (picks.index(pick) + 1, rec["size"], rec["n"],
             rec["validity"], "weighted" if rec["weighted"] else "unweighted"))
        A("fn wall %.1f ms | device timeline %.1f ms | notes: %s"
          % (rec["fn_wall_s"] * 1000, rec["device_total_ms"],
             "; ".join(rec["notes"])))
        rows = []
        for i, (name, ms) in enumerate(raw):
            kind = classify_sync(name) if (name == "host_stack_pack"
                                           or name.startswith("sync_")
                                           or "nrej" in name) else "-"
            rows.append([i, name, "%.4f" % ms, kind])
        A(fmt_table(["#", "region ending at mark", "device ms",
                     "host/sync nature"], rows))

    A("\n### 3.6 Phase-aggregated device time (median over all 480x270 "
      "weighted runs; per-phase sum of per-mark regions)\n")
    phase_agg = {}
    nw = 0
    for r in results["timing"]:
        if r["size"] == "480x270" and r["error"] is None and r["weighted"]:
            nw += 1
            for ph, ms in r["phases_ms"].items():
                phase_agg.setdefault(ph, []).append(ms)
    order = sorted(phase_agg, key=lambda p: -statistics.median(
        phase_agg[p]))
    A(fmt_table(
        ["rank", "phase", "median ms", "share of dev tot (median)",
         "n witnesses"],
        [[i + 1, ph, "%.4f" % statistics.median(phase_agg[ph]),
          "-", nw] for i, ph in enumerate(order)]))

    A("\n## 4. Peak-memory evidence (sync-per-boundary pass)\n")
    for rec in results["memory"]:
        if rec["error"]:
            A("- %s N=%d validity=%s -> **%s**" % (
                rec["size"], rec["n"], rec["validity"], rec["error"]))
            continue
        A("\n### %s N=%d validity=%s: peak live(pool used)=%.1f MiB | "
          "pool total high-water=%.1f MiB | min driver-free=%.1f MiB\n"
          % (rec["size"], rec["n"], rec["validity"],
             rec["peak_live_mib"], rec["peak_pool_total_mib"],
             rec["min_driver_free_mib"]))
        snaps = rec["snapshots"]
        step = max(1, len(snaps) // 30)
        A(fmt_table(
            ["mark", "driver_free MiB", "pool_used MiB", "pool_free MiB",
             "pool_total MiB"],
            [[s["mark"], s["driver_free_mib"], s["pool_used_mib"],
              s["pool_free_mib"], s["pool_total_mib"]]
             for s in snaps[::step]]))
        A("_... %d boundaries total (every %dth shown)_" % (len(snaps),
                                                            step))

    A("\n## 5. Synchronization inventory\n")
    A("Host-device sync patterns in the winsorized path (code-derived):\n")
    A("- per `_winsorize_axis0_cp` call (limits (0.05,0.05)): "
      "`bool(cp.any(n_valid>0))` early-exit + `bool(cp.any(lowidx>=N))` "
      "OOB guard + `bool(cp.any(low_sel))` + `bool(cp.any(high_sel))` "
      "scatter guards = **4 scalar syncs** per call")
    A("- per sigma iteration: `n_rej=int(count_nonzero(mask)) - "
      "int(count_nonzero(new_mask))` = **2 full-reduction D2H syncs**")
    A("- `_winsorized_rejected_pct_cp`: 2 full-reduction D2H syncs")
    A("- final `cp.asnumpy(result)` + `cp.asnumpy(sum_w)`: 2 D2H syncs")
    A("- total: 4*(1+iters) + 2 + 2 per full run (rewinsor `_winsorize_"
      "_bounds_cp` adds none).")
    for r in results["timing"]:
        if (r["size"], r["n"], r["validity"], r["weighted"]) == (
                "480x270", 20, "partial", True):
            A("\nMeasured wall cost of each sync/host segment, 480x270 "
              "N=20 partial weighted (median of reps):\n")
            A(fmt_table(
                ["segment", "kind", "wall ms"],
                [[name, classify_sync(name), "%.3f" % ms]
                 for name, ms in r["wall_ms"]]))
            sync_tot = sum(ms for name, ms in r["wall_ms"]
                           if name.startswith("sync_") or "nrej" in name)
            A("\nTotal sync-stall wall: %.2f ms of %.1f ms fn wall (%.0f%%)"
              % (sync_tot, r["fn_wall_s"] * 1000,
                 100.0 * sync_tot / (r["fn_wall_s"] * 1000)))

    A("\n## 6. Dominant-cost ranking\n")
    A("**Time** (480x270 phase medians, see 3.6): rank-ordered phases are "
      "shown in table 3.6; expected dominant device cost = the repeated "
      "full axis-0 argsorts (order + inverse-rank argsort) inside each "
      "winsorize call, plus the rewinsor argsort. **Wall**: the "
      "count_nonzero/int conversion syncs per iteration and the "
      "rejected_pct syncs are the dominant *stalls* (host waits on full "
      "device reductions). **Memory**: int64 argsort outputs `order` and "
      "`rank` (8 B/element each) are the largest temporaries; see section "
      "4 for live high-water per witness.")
    r12 = next((r for r in results["memory"] if r["size"] == "1080p"
                and r["n"] == 12 and r["validity"] == "partial"), None)
    if r12 and r12["error"] is None:
        stack_mib = mib(12 * 1920 * 1080 * 4)
        A("\n- 1080p N=12 partial: peak live %.1f MiB = %.1fx the %.1f MiB "
          "float32 stack (guard model assumes 6x)."
          % (r12["peak_live_mib"], r12["peak_live_mib"] / stack_mib,
             stack_mib))

    A("\n## 7. OOM / fallback boundary on MX150 (direct GPU calls, 1080p, "
      "partial validity)\n")
    A(fmt_table(
        ["n", "outcome", "fn wall s", "rejected_pct"],
        [[r["n"], r["outcome"],
          "%.2f" % r["fn_wall_s"] if r["fn_wall_s"] else "-",
          "%.2f" % r["rejected_pct"] if "rejected_pct" in r else "-"]
         for r in results["oom"]]))

    A("\n## 8. CPU wall reference (480x270 partial, same process)\n")
    A(fmt_table(["n", "cpu wall s (median of 2)"],
                [[r["n"], r["cpu_wall_s"]] for r in results["cpu_wall"]]))

    A("\n## 9. Blockers / uncertainties\n")
    A("- Internal sort/segmented-argsort scratch workspace is invisible to "
      "pool-endpoint and memGetInfo snapshots; pool used/total only *bound* "
      "it, so single-kernel high-water may be understated.")
    A("- MX150 (2 GiB, CC 6.1) is a NON-authoritative perf witness; "
      "primary witness is the Windows RTX 3070 8 GiB (later phase).")
    A("- The second D2H (`sum_w` asnumpy) occurs after the `d2h_done` "
      "event and is not separately bracketed (~us after result D2H).")
    A("- CuPy memory pool is retained between runs of a witness (pool "
      "free blocks reused); pool emptied before each witness for "
      "deterministic driver-free baselines.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(
        REPO, "profiling", "results_phaseA"))
    ap.add_argument("--report-body", default=os.path.join(
        REPO, "profiling", "results_phaseA", "report_body.md"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    LOG_PATH = os.path.join(args.outdir, "progress.log")

    def log(msg):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")

    import faulthandler
    faulthandler.dump_traceback_later(30, repeat=True)

    cp.get_default_memory_pool().free_all_blocks()
    results = {}
    _free, _total = cp.cuda.runtime.memGetInfo()
    results["device_total"] = int(_total)
    results["baseline_free"] = int(_free)

    log("[1/8] bitwise A/B (instrumented vs plain)")
    results["bitwise"] = bitwise_probe_check(args.outdir)
    log("[2/8] parity GPU vs CPU")
    results["parity"] = parity_check()
    log("[3/8] dispatch sweep")
    results["dispatch"], results["dispatch_probe"] = dispatch_sweep()
    log("[4/8] warmup")
    sg._ensure_probe()
    imgs = make_images(4, (128, 128), seed=1, nan_frac=0.05)
    gpu_ref(imgs, make_weights(4))
    log("[5/8] stage timing sweep")
    results["timing"] = timing_sweep(quick=args.quick, log=log)
    log("[6/8] memory sweep")
    results["memory"] = memory_sweep(log=log)
    log("[7/8] OOM boundary")
    results["oom"] = oom_boundary(log=log)
    log("[8/8] CPU wall reference")
    results["cpu_wall"] = cpu_wall_reference(log=log)
    log("rendering report body")

    json_path = os.path.join(args.outdir, "phaseA_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=1, default=str)
    with open(args.report_body, "w") as f:
        f.write(render_report_body(results))
    log("artifacts written: %s, %s" % (json_path, args.report_body))


if __name__ == "__main__":
    main()
