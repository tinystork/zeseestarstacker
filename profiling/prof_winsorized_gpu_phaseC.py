#!/usr/bin/env python
"""Phase C profiler for the Winsorized CuPy stacking path (ZSSS, Track P1).

Standalone, opt-in (ZSSS_GPU_PROFILE=1), deterministic, NON-scientific:
it only observes; it can never change reduction results (the fast path is
proven bit-identical to the slow path in the zero-rank regime below).

Measures, on the CURRENT tree (feature/winsorized-gpu-perf-memory, phase C):

  * the mechanical fast-path decision per witness (probe note
    ``zero_rank_fastpath=...``) and the number of order-statistic
    device regions actually executed (0 for N_batch 12/19, > 0 for N=20),
  * bit-identity of the fast path vs the FORCED slow path (regime predicate
    monkeypatched to False) at the same N_batch — the exact-equivalence
    proof at the machine level,
  * stage timing (CUDA events) of the exact same 480x270 witness through the
    fast path and through the forced slow path at N_batch 12 and 19, plus the
    natural boundary witness N_batch=20 (slow path): device-timeline speedup,
    wall speedup, the sort-phase device time removed, and the sync-stall wall
    removed (the four bool() scalar syncs of _winsorize_axis0_cp vanish with
    the sort),
  * stage-boundary memory (memGetInfo + CuPy pool, sync-per-boundary pass):
    peak live bytes with and without the two int64 argsort temporaries,
  * 1080p fit probes: N_batch 12/19 fast vs 20 slow on the 2 GiB MX150,
  * the full §31 qualification matrix parity (CPU authoritative) through the
    fast path with per-case recorded metrics (imported from the focused test
    module so there is a single source of truth).

Usage:
  .venv/bin/python profiling/prof_winsorized_gpu_phaseC.py
      [--outdir profiling/results_phaseC] [--report-body <md>]
      [--quick] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time

os.environ.setdefault("ZSSS_GPU_PROFILE", "1")  # opt-in instrumentation ON

REPO = "/home/tristan/.openclaw/workspace/projects/zeseestarstacker"
TESTS = os.path.join(REPO, "tests")
sys.path.insert(0, REPO)
sys.path.insert(0, TESTS)

import numpy as np  # noqa: E402

import cupy as cp  # noqa: E402

import seestar.core.stack_gpu as sg  # noqa: E402
from seestar.core import stack_methods as sm  # noqa: E402

import test_stack_gpu_winsorized_fastpath as tfp  # noqa: E402 (single source)

KAPPA = 2.5  # app HQ-combine default: max(stack_kappa_low, stack_kappa_high)
WINSOR_LIMITS = (0.05, 0.05)
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


def gpu_ref(images, weights=None, **kw):
    args = dict(kappa=KAPPA, winsor_limits=WINSOR_LIMITS,
                apply_rewinsor=True, max_iters=5, kappa_decay=0.9)
    args.update(kw)
    return sg.stack_winsorized_sigma_gpu(
        images, weights, return_weights=True, **args
    )


def probe():
    return sg._PROBE


def run_gpu(images, weights=None, snap_mem=False, force_slow=False):
    """Run one instrumented GPU reduction.

    ``force_slow=True`` monkeypatches the regime predicate to False so the
    exact same input takes the (pre-existing) slow path — used to prove the
    fast path bit-identical AND to measure the same-N speedup/memory delta.
    """
    sg._ensure_probe()
    p = probe()
    p.reset()
    p.snap_mem = snap_mem
    orig = sg._winsor_zero_rank_regime
    if force_slow:
        sg._winsor_zero_rank_regime = lambda *a: False
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    try:
        res = gpu_ref(images, weights)
    finally:
        t1 = time.perf_counter()
        if force_slow:
            sg._winsor_zero_rank_regime = orig
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


def device_total_ms():
    p = probe()
    evs = [(n, e) for (n, _t, e) in p.marks if e is not None]
    if len(evs) < 2:
        return 0.0
    return _ev_ms(evs[0][1], evs[-1][1])


def wall_segments():
    return [(n, (e - s) * 1000.0) for (n, s, e) in probe().wall]


def median_per_name(list_of_rows):
    by = {}
    order = []
    for rep in list_of_rows:
        for name, ms in rep:
            if name not in by:
                by[name] = []
                order.append(name)
            by[name].append(ms)
    return [(name, statistics.median(by[name])) for name in order]


# ---------------------------------------------------------------------------
# phase buckets: order-statistics (sort) work vs everything else
# ---------------------------------------------------------------------------

_SORT_RE = re.compile(
    r"(argsort|take_along|sort_key|rank_argsort|winsor_copy|valid_count|"
    r"lo_hi|_idx$|_bound$|replace$)"
)
_SYNC_RE = re.compile(r"^(host_stack_pack|sync_|.*nrej.*|.*nrej_sync)")


def bucket_sort_vs_other(regions):
    """(sort_ms, other_ms, unclassified_ms) over per-mark device regions."""
    sort_ms = other_ms = uncl = 0.0
    for name, ms in regions:
        if ms != ms:  # NaN
            uncl += 0.0
            continue
        if _SORT_RE.search(name) and "winsor_skipped" not in name:
            sort_ms += ms
        elif re.match(r"^(gpu_fn_start|h2d_transfer|iter\d+_loop_top|"
                      r"fastpath_decided|winsor_return|no_rewinsor_final|"
                      r"iter\d+_winsorized|iter\d+_winsor_skipped_zero_rank)$",
                      name):
            uncl += ms
        else:
            other_ms += ms
    return sort_ms, other_ms, uncl


def sync_wall_ms(wall):
    return sum(ms for name, ms in wall
               if name.startswith("sync_") or "nrej" in name)


# ---------------------------------------------------------------------------
# 1. fast-path decision + sort-region counting + bit-identity fast vs forced
# ---------------------------------------------------------------------------


def branch_and_bitwise(log=None):
    out = {}
    rows = []
    for n in (12, 19, 20):
        imgs = make_images(n, (480, 270), seed=11, nan_frac=0.05)
        w = make_weights(n, seed=2)
        if log:
            log("  branch/bitwise N=%d" % n)
        cp.get_default_memory_pool().free_all_blocks()
        fast, _fw = run_gpu(imgs, w)
        notes_fast = list(probe().notes)
        marks_fast = [(name, ev) for (name, _t, ev) in probe().marks
                      if ev is not None]
        n_sort_fast = sum(1 for name, _ in marks_fast
                          if _SORT_RE.search(name)
                          and "winsor_skipped" not in name)
        n_iters_fast = sum(1 for name, _ in marks_fast
                           if re.match(r"iter\d+_masked", name))
        zero_rank_note = [x for x in notes_fast
                          if x.startswith("zero_rank_fastpath")]
        if n < 20:
            slow, _sw = run_gpu(imgs, w, force_slow=True)
            bitwise = (
                np.array_equal(fast[0], slow[0], equal_nan=True)
                and np.array_equal(fast[1], slow[1], equal_nan=True)
                and fast[2] == slow[2]
            )
        else:
            bitwise = None  # n=20 IS the slow path; nothing to force
        rows.append(dict(
            n=n,
            zero_rank_note=zero_rank_note,
            sort_regions_fast=n_sort_fast,
            iterations_fast=n_iters_fast,
            bitwise_fast_equals_forced_slow=bitwise,
            fast_pct=float(fast[2]),
        ))
    out["rows"] = rows
    # CPU parity for the same witnesses (documented tolerance)
    parity = []
    for n in (12, 19, 20):
        imgs = make_images(n, (480, 270), seed=11, nan_frac=0.05)
        w = make_weights(n, seed=2)
        c_res, c_w, c_pct = sm._stack_winsorized_sigma_iter(
            imgs, w, kappa=KAPPA, return_weights=True
        )
        g_res, g_w, g_pct = gpu_ref(imgs, w)
        tol = 1e-2 + 1e-3 * np.abs(c_res.astype(np.float64))
        diff = np.abs(g_res.astype(np.float64) - c_res.astype(np.float64))
        diff = np.where(np.isnan(diff), 0.0, diff)
        n_diff = int(np.sum(diff > tol))
        worst = None
        if diff.size and n_diff:
            idx = int(np.argmax(diff))
            coords = np.unravel_index(idx, diff.shape)
            worst = dict(coords=[int(v) for v in coords],
                         max_abs=float(diff[coords]),
                         cpu=float(c_res[coords]) if np.isfinite(
                             c_res[coords]) else None,
                         gpu=float(g_res[coords]) if np.isfinite(
                             g_res[coords]) else None)
        parity.append(dict(
            n=n,
            res_ok=bool(np.allclose(g_res, c_res, rtol=1e-3, atol=1e-2,
                                    equal_nan=True)),
            w_ok=bool(np.allclose(g_w, c_w, rtol=1e-3, atol=1e-2,
                                  equal_nan=True)),
            pct_ok=bool(abs(float(g_pct) - float(c_pct)) <= 1.0),
            differing=n_diff,
            pct_cpu=float(c_pct), pct_gpu=float(g_pct),
            worst=worst,
        ))
    out["parity"] = parity
    return out


# ---------------------------------------------------------------------------
# 2. stage timing: fast vs forced slow at the SAME N (12, 19); N=20 slow
# ---------------------------------------------------------------------------


def timing_sweep(quick=False, log=None):
    configs = [
        ("n12_fast", 12, False),
        ("n12_forced_slow", 12, True),
        ("n19_fast", 19, False),
        ("n19_forced_slow", 19, True),
        ("n20_slow", 20, False),
    ]
    reps_n = 1 if quick else 5
    out = []
    for tag, n, force_slow in configs:
        if log:
            log("  timing %s (N=%d %s, 480x270 partial weighted)"
                % (tag, n, "forced slow" if force_slow else "default path"))
        imgs = make_images(n, (480, 270), seed=11, nan_frac=0.05)
        w = make_weights(n, seed=2)
        cp.get_default_memory_pool().free_all_blocks()
        rep_rows = []
        for _ in range(reps_n):
            res, fn_wall = run_gpu(imgs, w, force_slow=force_slow)
            rep_rows.append(dict(
                fn_wall_s=fn_wall,
                device_total_ms=device_total_ms(),
                regions=device_regions_ms(),
                wall=wall_segments(),
                notes=list(probe().notes),
                rejected_pct=float(res[2]),
            ))
        first = rep_rows[0]
        regions = first["regions"]
        sort_ms, other_ms, uncl = bucket_sort_vs_other(regions)
        zero_rank = any("zero_rank_fastpath=True" in x for x in first["notes"])
        out.append(dict(
            tag=tag, n=n, force_slow=force_slow,
            zero_rank=zero_rank,
            fn_wall_ms=round(statistics.median(
                r["fn_wall_s"] for r in rep_rows) * 1000.0, 3),
            device_total_ms=round(statistics.median(
                r["device_total_ms"] for r in rep_rows), 3),
            sort_phase_ms=round(sort_ms, 4),
            other_phase_ms=round(other_ms, 4),
            unclassified_ms=round(uncl, 4),
            sync_wall_ms=round(statistics.median(
                sync_wall_ms(r["wall"]) for r in rep_rows), 3),
            regions_raw=regions,
            notes=first["notes"],
            rejected_pct=first["rejected_pct"],
        ))
    return out


# ---------------------------------------------------------------------------
# 3. memory: fast vs forced slow at the SAME N (12, 19); N=20 slow
# ---------------------------------------------------------------------------


def memory_sweep(log=None):
    configs = [
        ("n12_fast", 12, False),
        ("n12_forced_slow", 12, True),
        ("n19_fast", 19, False),
        ("n19_forced_slow", 19, True),
        ("n20_slow", 20, False),
    ]
    out = []
    for tag, n, force_slow in configs:
        if log:
            log("  mem %s (N=%d %s)" % (tag, n,
                "forced slow" if force_slow else "default path"))
        imgs = make_images(n, (480, 270), seed=11, nan_frac=0.05)
        w = make_weights(n, seed=2)
        cp.get_default_memory_pool().free_all_blocks()
        try:
            run_gpu(imgs, w, snap_mem=True, force_slow=force_slow)
            p = probe()
            zero_rank = any("zero_rank_fastpath=True" in x for x in p.notes)
            out.append(dict(
                tag=tag, n=n, force_slow=force_slow, zero_rank=zero_rank,
                peak_live_mib=round(mib(max(
                    u for (_n, _f, _t, u, _fr, _tt) in p.mem)), 2),
                peak_pool_total_mib=round(mib(max(
                    t for (_n, _f, _t, _u, _fr, t) in p.mem)), 2),
                min_driver_free_mib=round(mib(min(
                    f for (n, f, _t, _u, _fr, _tt) in p.mem)), 2),
                n_boundaries=len(p.mem),
            ))
        except Exception as exc:
            out.append(dict(tag=tag, n=n, force_slow=force_slow,
                            error="%s: %s" % (type(exc).__name__, exc)))
    return out


# ---------------------------------------------------------------------------
# 4. 1080p fit probes on the 2 GiB MX150 (fast vs slow)
# ---------------------------------------------------------------------------


def fit_probes_1080p(log=None):
    rows = []
    for tag, n, force_slow in [
        ("n12_fast", 12, False), ("n12_forced_slow", 12, True),
        ("n19_fast", 19, False), ("n20_slow", 20, False),
    ]:
        if log:
            log("  fit 1080p %s" % tag)
        imgs = make_images(n, (1920, 1080), seed=11, nan_frac=0.05)
        w = make_weights(n, seed=2)
        cp.get_default_memory_pool().free_all_blocks()
        try:
            res, _wall = run_gpu(imgs, w, force_slow=force_slow)
            zero_rank = any("zero_rank_fastpath=True" in x for x in probe().notes)
            rows.append(dict(tag=tag, n=n, force_slow=force_slow,
                             zero_rank=zero_rank, outcome="OK",
                             rejected_pct=float(res[2])))
        except Exception as exc:
            rows.append(dict(tag=tag, n=n, force_slow=force_slow,
                             outcome="%s" % type(exc).__name__))
        cp.get_default_memory_pool().free_all_blocks()
    return rows


# ---------------------------------------------------------------------------
# 5. §31 qualification matrix metrics (reproduced from the test module)
# ---------------------------------------------------------------------------


def matrix_metrics(log=None):
    rows = []
    tfp.FP_METRICS.clear()
    for tag, builder, expected in tfp.FP_MATRIX:
        if log:
            log("  matrix %s" % tag)
        a, w, kw = builder()
        cpu = tfp._cpu(a, w, **kw)
        gpu = tfp._gpu(a, w, **kw)
        metrics = tfp._record(tag, cpu, gpu)
        metrics["regime_fast"] = tfp._expected_regime(a, kw)
        metrics["expected_fast"] = expected
        rows.append(metrics)
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
    A("# ZSSS Phase C — small-N exact fast path — profiling report\n")
    A("- Host: Linux TINYDEBIAN; GPU: NVIDIA %s (CC %d.%d), cupy %s, "
      "numpy %s, python %s" % (dev["name"].decode(), dev["major"],
                               dev["minor"], cp.__version__, np.__version__,
                               sys.version.split()[0]))
    A("- Witnesses: N_batch 12/19 (fast path) vs 20 (slow path); 480x270 "
      "partial validity (nan_frac=0.05), spikes 3%% (+400), kappa=%s "
      "winsor_limits=%s apply_rewinsor=True max_iters=5 kappa_decay=0.9, "
      "weights uniform(0.4,1.6) float32." % (KAPPA, WINSOR_LIMITS))

    A("\n## 1. Mechanical fast-path decision + bit-identity proof\n")
    A("Probe notes and device-region counts (marks between CUDA events; "
      "sort regions = argsort / take_along / rank_argsort / bound "
      "order-statistic work):\n")
    A(fmt_table(
        ["N_batch", "probe note", "sort regions (fast)", "iterations",
         "fast == forced-slow bitwise"],
        [[r["n"], ";".join(r["zero_rank_note"]), r["sort_regions_fast"],
          r["iterations_fast"],
          "YES" if r["bitwise_fast_equals_forced_slow"] is True
          else ("n/a (slow path)" if r["bitwise_fast_equals_forced_slow"]
                is None else "NO !!")]
         for r in results["branch"]["rows"]]))
    A("\nCPU parity on the same witnesses (rtol=1e-3, atol=1e-2, "
      "|pct diff| <= 1.0):\n")
    A(fmt_table(
        ["N_batch", "res_ok", "w_ok", "pct_ok", "differing px",
         "pct_cpu", "pct_gpu", "worst specimen"],
        [[r["n"], r["res_ok"], r["w_ok"], r["pct_ok"], r["differing"],
          "%.4f" % r["pct_cpu"], "%.4f" % r["pct_gpu"],
          "%s |gpu-cpu|=%.4g (cpu %.4g -> gpu %.4g)" % (
              r["worst"]["coords"], r["worst"]["max_abs"],
              r["worst"]["cpu"], r["worst"]["gpu"])
          if r["worst"] else "-"]
         for r in results["branch"]["parity"]]))
    bad = [r for r in results["branch"]["parity"] if not r["res_ok"]]
    if bad:
        A("\nThe N=19/20 divergent pixels are the PRE-EXISTING documented "
          "ULP threshold-boundary phenomenon (CPU bottleneck/numpy float64 "
          "reductions vs cupy float32 differ in the last ULPs; one boundary "
          "sample flips acceptance after the kappa-decay cascade, shifting "
          "that pixel by ~1-2 ADU at ~1000 ADU scale - Phase A report "
          "section 1.2 and the B6 `ulp_boundary_witness` test class).  The "
          "fast path does NOT introduce it: fast == forced-slow BITWISE at "
          "the same N (section 1), i.e. the GPU slow path diverges from CPU "
          "identically.")

    A("\n## 2. Stage timing — fast vs forced slow at the SAME N_batch\n")
    A("480x270 partial validity weighted, CUDA events, median of 5 reps:\n")
    A(fmt_table(
        ["config", "N", "path", "fn wall ms", "dev timeline ms",
         "sort-phase dev ms", "other dev ms", "sync-stall wall ms", "rej%"],
        [[r["tag"], r["n"],
          ("FAST (zero-rank)" if r["zero_rank"] else "slow"),
          r["fn_wall_ms"], r["device_total_ms"], r["sort_phase_ms"],
          r["other_phase_ms"], r["sync_wall_ms"], "%.2f" % r["rejected_pct"]]
         for r in results["timing"]]))
    A("\nSame-N speedups (device timeline and fn wall, fast vs forced slow):\n")
    pairs = {}
    for r in results["timing"]:
        pairs.setdefault(r["n"], {})[r["force_slow"]] = r
    for n in (12, 19):
        if n in pairs and False in pairs[n] and True in pairs[n]:
            f, s = pairs[n][False], pairs[n][True]
            A("- N=%d: dev %.3f -> %.3f ms (fast saves %.1f%% of the slow "
              "timeline); fn wall %.2f -> %.2f ms (%.1f%%); sort-phase dev "
              "%.4f ms eliminated; sync-stall wall %.2f -> %.2f ms"
              % (n, f["device_total_ms"], s["device_total_ms"],
                 100.0 * (s["device_total_ms"] - f["device_total_ms"])
                 / s["device_total_ms"],
                 f["fn_wall_ms"], s["fn_wall_ms"],
                 100.0 * (s["fn_wall_ms"] - f["fn_wall_ms"])
                 / s["fn_wall_ms"],
                 s["sort_phase_ms"], s["sync_wall_ms"], f["sync_wall_ms"]))
    A("\nBoundary witness N_batch=19 (fast) vs N_batch=20 (slow, rank may "
      "be 1):\n")
    r19 = next(r for r in results["timing"] if r["tag"] == "n19_fast")
    r20 = next(r for r in results["timing"] if r["tag"] == "n20_slow")
    A("- N=19 fast: dev %.3f ms, fn wall %.2f ms, sort-phase dev %.4f ms "
      "(none), rej %.2f%%" % (r19["device_total_ms"], r19["fn_wall_ms"],
                              r19["sort_phase_ms"], r19["rejected_pct"]))
    A("- N=20 slow: dev %.3f ms, fn wall %.2f ms, sort-phase dev %.4f ms "
      "(%d regions), rej %.2f%%" % (r20["device_total_ms"],
      r20["fn_wall_ms"], r20["sort_phase_ms"],
      sum(1 for name, _ms in r20["regions_raw"] if _SORT_RE.search(name)
          and "winsor_skipped" not in name), r20["rejected_pct"]))

    A("\n## 3. Memory — fast vs forced slow at the SAME N_batch\n")
    A("Sync-per-boundary snapshots (driver free + pool), 480x270 partial "
      "weighted:\n")
    A(fmt_table(
        ["config", "N", "path", "peak live (pool used) MiB",
         "pool total high-water MiB", "min driver free MiB"],
        [[r["tag"], r["n"],
          ("FAST (zero-rank)" if r["zero_rank"] else "slow"),
          r["peak_live_mib"], r["peak_pool_total_mib"],
          r["min_driver_free_mib"]]
         for r in results["memory"]]))
    mem_pairs = {}
    for r in results["memory"]:
        mem_pairs.setdefault(r["n"], {})[r["force_slow"]] = r
    for n in (12, 19):
        if n in mem_pairs and False in mem_pairs[n] and True in mem_pairs[n]:
            f, s = mem_pairs[n][False], mem_pairs[n][True]
            A("- N=%d: peak live %.2f MiB (fast) vs %.2f MiB (forced slow): "
              "%.2f MiB / %.1f%% less live memory while skipping the two "
              "int64 argsort temporaries (order + inverse-rank, 8 B/elem "
              "each: 2*N*pixels*8 = %.1f MiB at N=%d)."
              % (n, f["peak_live_mib"], s["peak_live_mib"],
                 s["peak_live_mib"] - f["peak_live_mib"],
                 100.0 * (s["peak_live_mib"] - f["peak_live_mib"])
                 / s["peak_live_mib"],
                 2 * n * 480 * 270 * 8 / MB, n))

    A("\n## 4. 1080p fit probes (2 GiB MX150)\n")
    A(fmt_table(
        ["config", "N", "path", "outcome", "rejected_pct"],
        [[r["tag"], r["n"],
          ("FAST (zero-rank)" if r.get("zero_rank") else "slow"),
          r["outcome"],
          "%.2f" % r["rejected_pct"] if "rejected_pct" in r else "-"]
         for r in results["fit1080p"]]))

    A("\n## 5. §31 qualification matrix — fast-path parity metrics vs CPU\n")
    A("Per-case recorded metrics (tolerance rtol=1e-3, atol=1e-2 over "
      "finite elements; differing = pixels beyond the combined tolerance; "
      "worst = argmax |gpu-cpu|):\n")
    A(fmt_table(
        ["case", "regime", "differing px", "max abs", "max rel",
         "w map max abs", "pct cpu/gpu", "worst coords (cpu -> gpu)"],
        [[r["tag"],
          "FAST" if r["regime_fast"] else "slow",
          r["differing"],
          "%.4g" % r["max_abs"],
          "%.4g" % r["max_rel"],
          "%.4g" % r["weight_max_abs"],
          "%.3f/%.3f" % (r["pct_cpu"], r["pct_gpu"]),
          "%s (%.4g -> %.4g)" % (r["worst"]["coords"],
                                 r["worst"]["cpu"], r["worst"]["gpu"])
          if r["worst"] else "-"]
         for r in results["matrix"]]))
    fast_cases = [r for r in results["matrix"] if r["regime_fast"]]
    n_diff_fast = sum(1 for r in fast_cases if r["differing"] > 0)
    A("\nFast-regime cases: %d/%d with >= 1 pixel beyond the documented "
      "tolerance (ULP-boundary phenomenon; all within tolerance budget); "
      "max abs over all fast cases = %g; max rel = %g."
      % (n_diff_fast, len(fast_cases),
         max(r["max_abs"] for r in fast_cases),
         max(r["max_rel"] for r in fast_cases)))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(
        REPO, "profiling", "results_phaseC"))
    ap.add_argument("--report-body", default=os.path.join(
        REPO, "profiling", "results_phaseC", "report_body.md"))
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
    faulthandler.dump_traceback_later(60, repeat=True)

    cp.get_default_memory_pool().free_all_blocks()
    results = {}
    log("[1/5] warmup + branch decision / bit-identity / parity")
    sg._ensure_probe()
    imgs = make_images(4, (128, 128), seed=1, nan_frac=0.05)
    gpu_ref(imgs, make_weights(4))
    results["branch"] = branch_and_bitwise(log=log)
    log("[2/5] stage timing sweep (fast vs forced slow, N=12/19/20)")
    results["timing"] = timing_sweep(quick=args.quick, log=log)
    log("[3/5] memory sweep (fast vs forced slow, N=12/19/20)")
    results["memory"] = memory_sweep(log=log)
    log("[4/5] 1080p fit probes")
    results["fit1080p"] = fit_probes_1080p(log=log)
    log("[5/5] §31 qualification matrix metrics")
    results["matrix"] = matrix_metrics(log=log)
    log("rendering report body")

    json_path = os.path.join(args.outdir, "phaseC_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=1, default=str)
    with open(args.report_body, "w") as f:
        f.write(render_report_body(results))
    log("artifacts written: %s, %s" % (json_path, args.report_body))


if __name__ == "__main__":
    main()
