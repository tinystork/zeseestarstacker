#!/usr/bin/env python
"""Phase D profiler for the Winsorized CuPy stacking path (ZSSS, Track P2).

Standalone, opt-in (ZSSS_GPU_PROFILE=1), deterministic, NON-scientific:
it only observes; it can never change reduction results (the optimized slow
path is proven bit-identical to the pre-Phase-D slow path below).

Measures, on the CURRENT tree (feature/winsorized-gpu-perf-memory, phase D):

  * the exact-equivalence proof at the machine level: the OPTIMIZED slow
    path (value-clip + direct ``cp.sort``, no int64 ``order``/``rank``)
    vs the PRE-Phase-D slow path (``argsort`` + inverse-rank scatter),
    forced through the retained verbatim legacy helpers, at N_batch
    20 / 32 / 50 — BITWISE result, weight map and rejected_pct;
  * mechanical no-argsort proof: probe marks of an optimized slow-path run
    contain ``winsor_direct_sort`` / ``bounds_direct_sort`` and ZERO
    ``winsor_argsort`` / ``winsor_rank_argsort`` / ``winsor_take_along`` /
    ``bounds_argsort`` marks;
  * stage timing (CUDA events) + stage-boundary memory (memGetInfo + CuPy
    pool, sync-per-boundary pass) of the SAME 480x270 partial-validity
    weighted witness through the optimized slow path and through the forced
    LEGACY slow path at N_batch 20 / 32 / 50: device-timeline delta,
    sort-phase delta (cp.sort vs argsort + gather + rank + scatter), the
    sync-stall delta and the VRAM peak reduction (order/rank removed:
    2 * N * pixels * 8 bytes plus the argsort internal scratch);
  * 1080p fit probes on the 2 GiB MX150: optimized slow path at N 20/32/50
    vs the legacy slow path (previously MemoryError at N >= 12 slow);
  * CPU parity of the optimized slow path on the same witnesses
    (documented tolerance rtol=1e-3, atol=1e-2, |rejected_pct| <= 1.0);
  * a reduced re-run of the PHASE A driver sweep (same stage bucketing code)
    at N_batch 20 / 32 / 50 so the per-stage deltas vs the recorded
    results_phaseA baseline are measured with the identical driver.

Usage:
  .venv/bin/python profiling/prof_winsorized_gpu_phaseD.py
      [--outdir profiling/results_phaseD] [--report-body <md>]
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
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "profiling"))

import numpy as np  # noqa: E402

import cupy as cp  # noqa: E402

import seestar.core.stack_gpu as sg  # noqa: E402
from seestar.core import stack_methods as sm  # noqa: E402

KAPPA = 2.5  # app HQ-combine default: max(stack_kappa_low, stack_kappa_high)
WINSOR_LIMITS = (0.05, 0.05)
SLOW_N = [20, 32, 50]
MB = 1024 * 1024


def mib(b):
    return b / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# deterministic witnesses (same generators as the phase A/C drivers)
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


def gpu_ref(images, weights=None, **kw):
    args = dict(kappa=KAPPA, winsor_limits=WINSOR_LIMITS,
                apply_rewinsor=True, max_iters=5, kappa_decay=0.9)
    args.update(kw)
    return sg.stack_winsorized_sigma_gpu(
        images, weights, return_weights=True, **args
    )


def probe():
    return sg._PROBE


def run_gpu(images, weights=None, snap_mem=False, legacy=False):
    """Run one instrumented GPU reduction.

    ``legacy=True`` monkeypatches the two winsor sort helpers back to the
    pre-Phase-D implementations (the retained verbatim rank path for axis0 +
    an embedded verbatim legacy bounds) so the exact same input runs through
    the OLD slow path — used to prove the optimized path bit-identical AND
    to measure the same-N speedup/memory delta.  The legacy axis0 is
    ``sg._winsorize_axis0_rank_path_cp`` itself (probe events included); the
    legacy bounds copy below emits the old probe marks via ``sg._p_event``.
    """
    sg._ensure_probe()
    p = probe()
    p.reset()
    p.snap_mem = snap_mem
    orig_a = sg._winsorize_axis0_cp
    orig_b = sg._winsorize_bounds_cp
    if legacy:
        sg._winsorize_axis0_cp = sg._winsorize_axis0_rank_path_cp
        sg._winsorize_bounds_cp = _winsorize_bounds_legacy
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    try:
        res = gpu_ref(images, weights)
    finally:
        t1 = time.perf_counter()
        if legacy:
            sg._winsorize_axis0_cp = orig_a
            sg._winsorize_bounds_cp = orig_b
    cp.cuda.Stream.null.synchronize()
    p.snap_mem = False
    return res, t1 - t0


def _winsorize_bounds_legacy(cp_mod, arr, limits):
    """Verbatim pre-Phase-D ``_winsorize_bounds_cp`` (probe events kept,
    routed through the stack_gpu module helpers)."""
    low, high = limits
    valid = ~cp_mod.isnan(arr)
    n_valid = cp_mod.count_nonzero(valid, axis=0)
    sg._p_event("bounds_valid_count")
    sort_key = cp_mod.where(valid, arr, cp_mod.float32(cp_mod.inf))
    sg._p_event("bounds_sort_key")
    order = cp_mod.argsort(sort_key, axis=0)
    sg._p_event("bounds_argsort")
    sorted_vals = cp_mod.take_along_axis(sort_key, order, axis=0)
    sg._p_event("bounds_take_along")

    max_idx = cp_mod.maximum(n_valid - 1, 0)
    lowidx = cp_mod.clip(cp_mod.floor(low * n_valid).astype(cp_mod.int64),
                         0, max_idx)
    highidx = cp_mod.clip(
        n_valid - 1 - cp_mod.floor(high * n_valid).astype(cp_mod.int64),
        0, max_idx)
    sg._p_event("bounds_idx")

    low_b = cp_mod.take_along_axis(sorted_vals, lowidx[cp_mod.newaxis],
                                   axis=0)
    high_b = cp_mod.take_along_axis(sorted_vals, highidx[cp_mod.newaxis],
                                    axis=0)
    sg._p_event("bounds_lo_hi")
    return low_b, high_b


def _ev_ms(e1, e2):
    return float(cp.cuda.get_elapsed_time(e1, e2))


def device_regions_ms():
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


def sync_wall_ms(wall):
    return sum(ms for name, ms in wall
               if name.startswith("sync_") or "nrej" in name)


# sort-phase bucket covering BOTH mark vocabularies (old + new)
_SORT_RE = re.compile(
    r"(argsort|take_along|sort_key|direct_sort|rank_argsort|winsor_copy|"
    r"valid_count|lo_hi|_idx$|_bound$|replace$)"
)


def bucket_sort_vs_other(regions):
    sort_ms = other_ms = uncl = 0.0
    for name, ms in regions:
        if ms != ms:  # NaN
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


def count_marks(rx):
    marks = [name for (name, _t, ev) in probe().marks if ev is not None]
    return sum(1 for name in marks if rx.search(name))


# ---------------------------------------------------------------------------
# 1. bit-identity: optimized slow path == legacy slow path (N 20/32/50)
# ---------------------------------------------------------------------------


def bitwise_identity(log=None):
    out = []
    for n in SLOW_N:
        if log:
            log("  bitwise N=%d" % n)
        imgs = make_images(n, (480, 270), seed=11, nan_frac=0.05)
        w = make_weights(n, seed=2)
        cp.get_default_memory_pool().free_all_blocks()
        new, _w1 = run_gpu(imgs, w)
        leg, _w2 = run_gpu(imgs, w, legacy=True)
        assert np.array_equal(new[0], leg[0], equal_nan=True), n
        assert np.array_equal(new[1], leg[1], equal_nan=True), n
        assert new[2] == leg[2], n
        out.append(dict(n=n, bitwise=True, rejected_pct=float(new[2])))
    return out


# ---------------------------------------------------------------------------
# 2. mechanical no-argsort proof via probe marks
# ---------------------------------------------------------------------------


def marks_proof(log=None):
    if log:
        log("  marks proof N=20")
    imgs = make_images(20, (480, 270), seed=11, nan_frac=0.05)
    w = make_weights(20, seed=2)
    cp.get_default_memory_pool().free_all_blocks()
    run_gpu(imgs, w)
    marks = [name for (name, _t, ev) in probe().marks if ev is not None]
    old_sort_marks = [m for m in marks if re.search(
        r"(^winsor_argsort$|^winsor_rank_argsort$|^winsor_take_along$|"
        r"^bounds_argsort$|^bounds_take_along$)", m)]
    direct_sort_marks = [m for m in marks if re.search(
        r"^(winsor|bounds)_direct_sort$", m)]
    notes = list(probe().notes)
    return dict(
        n=20,
        old_int64_sort_marks=old_sort_marks,
        direct_sort_marks=direct_sort_marks,
        n_old=len(old_sort_marks),
        n_direct=len(direct_sort_marks),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# 3. stage timing: optimized slow path vs legacy slow path (same N)
# ---------------------------------------------------------------------------


def timing_sweep(quick=False, log=None):
    reps_n = 1 if quick else 5
    out = []
    for n in SLOW_N:
        for legacy in (False, True):
            tag = "n%d_%s" % (n, "legacy" if legacy else "new")
            if log:
                log("  timing %s (N=%d %s, 480x270 partial weighted)"
                    % (tag, n, "legacy slow" if legacy else "optimized slow"))
            imgs = make_images(n, (480, 270), seed=11, nan_frac=0.05)
            w = make_weights(n, seed=2)
            cp.get_default_memory_pool().free_all_blocks()
            rep_rows = []
            pcts = []
            for _ in range(reps_n):
                res, fn_wall = run_gpu(imgs, w, legacy=legacy)
                pcts.append(float(res[2]))
                regions = device_regions_ms()
                sort_ms, other_ms, uncl = bucket_sort_vs_other(regions)
                rep_rows.append(dict(
                    fn_wall_s=fn_wall,
                    device_total_ms=device_total_ms(),
                    sort_phase_ms=sort_ms,
                    other_phase_ms=other_ms,
                    unclassified_ms=uncl,
                    sync_wall_ms=sync_wall_ms(wall_segments()),
                    n_iters=sum(1 for name, _ms in regions
                                if re.match(r"iter\d+_masked", name)),
                ))
            first = rep_rows[0]
            out.append(dict(
                tag=tag, n=n, legacy=legacy,
                fn_wall_ms=round(statistics.median(
                    r["fn_wall_s"] for r in rep_rows) * 1000.0, 3),
                device_total_ms=round(statistics.median(
                    r["device_total_ms"] for r in rep_rows), 3),
                sort_phase_ms=round(statistics.median(
                    r["sort_phase_ms"] for r in rep_rows), 4),
                other_phase_ms=round(statistics.median(
                    r["other_phase_ms"] for r in rep_rows), 4),
                unclassified_ms=round(statistics.median(
                    r["unclassified_ms"] for r in rep_rows), 4),
                sync_wall_ms=round(statistics.median(
                    r["sync_wall_ms"] for r in rep_rows), 3),
                n_iters=first["n_iters"],
                rejected_pct=round(statistics.median(pcts), 4),
            ))
    return out


# ---------------------------------------------------------------------------
# 4. memory: optimized vs legacy slow path (same N), pool high-water
# ---------------------------------------------------------------------------


def memory_sweep(log=None):
    out = []
    for n in SLOW_N:
        for legacy in (False, True):
            tag = "n%d_%s" % (n, "legacy" if legacy else "new")
            if log:
                log("  mem %s (N=%d %s)" % (
                    tag, n, "legacy slow" if legacy else "optimized slow"))
            imgs = make_images(n, (480, 270), seed=11, nan_frac=0.05)
            w = make_weights(n, seed=2)
            cp.get_default_memory_pool().free_all_blocks()
            try:
                run_gpu(imgs, w, snap_mem=True, legacy=legacy)
                p = probe()
                out.append(dict(
                    tag=tag, n=n, legacy=legacy,
                    peak_live_mib=round(mib(max(
                        u for (_n, _f, _t, u, _fr, _tt) in p.mem)), 2),
                    peak_pool_total_mib=round(mib(max(
                        t for (_n, _f, _t, _u, _fr, t) in p.mem)), 2),
                    min_driver_free_mib=round(mib(min(
                        f for (n, f, _t, _u, _fr, _tt) in p.mem)), 2),
                    n_boundaries=len(p.mem),
                ))
            except Exception as exc:
                out.append(dict(tag=tag, n=n, legacy=legacy,
                                error="%s: %s" % (type(exc).__name__, exc)))
    return out


# ---------------------------------------------------------------------------
# 5. 1080p fit probes on the 2 GiB MX150 (new slow vs legacy slow)
# ---------------------------------------------------------------------------


def fit_probes_1080p(log=None):
    rows = []
    for tag, n, legacy in [
        ("n20_new_1080p", 20, False), ("n20_legacy_1080p", 20, True),
        ("n32_new_1080p", 32, False), ("n32_legacy_1080p", 32, True),
        ("n50_new_1080p", 50, False), ("n50_legacy_1080p", 50, True),
    ]:
        if log:
            log("  fit 1080p %s" % tag)
        imgs = make_images(n, (1920, 1080), seed=11, nan_frac=0.05)
        w = make_weights(n, seed=2)
        cp.get_default_memory_pool().free_all_blocks()
        try:
            res, _wall = run_gpu(imgs, w, legacy=legacy)
            rows.append(dict(tag=tag, n=n, legacy=legacy, outcome="OK",
                             rejected_pct=float(res[2])))
        except Exception as exc:
            rows.append(dict(tag=tag, n=n, legacy=legacy,
                             outcome="%s" % type(exc).__name__))
        cp.get_default_memory_pool().free_all_blocks()
    return rows


# ---------------------------------------------------------------------------
# 6. CPU parity of the optimized slow path (480x270 partial weighted)
# ---------------------------------------------------------------------------


def parity_sweep(log=None):
    out = []
    for n in SLOW_N:
        if log:
            log("  parity N=%d" % n)
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
        out.append(dict(
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
    return out


# ---------------------------------------------------------------------------
# 7. reduced PHASE A driver re-run (identical stage-bucketing code)
# ---------------------------------------------------------------------------


def phaseA_rerun(quick=False, log=None):
    """Run the phase A driver sweep restricted to N in {20,32,50} at
    480x270 partial/full + 1080p partial (identical instrumentation), so the
    per-stage sums on the optimized tree are produced by the very same code
    that recorded the results_phaseA baseline."""
    import prof_winsorized_gpu_phaseA as pA

    pA.N_BATCHES = list(SLOW_N)
    # 480x270 only: the phase A 1080p timing sweep would re-run the heavy
    # slow-path configs that the phase A baseline itself skipped/errored on
    # (1080p slow-path direct runs OOM on the 2 GiB MX150 - covered by the
    # dedicated fit probes in section 5); the 480x270 stage comparison is
    # the apples-to-apples per-stage delta vs the recorded baseline.
    pA.SIZES = {"480x270": (480, 270)}
    timing = []
    try:
        raw = pA.timing_sweep(quick=quick, log=log)
    except Exception as exc:  # noqa: BLE001 - keep the sweep going
        log("  phaseA timing_sweep failed: %r" % exc)
        raw = []
    for r in raw:
        if r["error"] or not r["reps"]:
            continue
        timing.append(dict(
            size=r["size"], n=r["n"], validity=r["validity"],
            weighted=r["weighted"],
            device_total_ms=r["device_total_ms"],
            fn_wall_ms=round(float(r["fn_wall_s"]) * 1000.0, 2),
            phases=r["phases_ms"],
        ))
    mem = []
    try:
        raw_mem = pA.memory_sweep(log=log)
    except Exception as exc:  # noqa: BLE001
        log("  phaseA memory_sweep failed: %r" % exc)
        raw_mem = []
    for r in raw_mem:
        if r.get("error"):
            mem.append(dict(size=r.get("size"), n=r.get("n"),
                            error=r["error"]))
            continue
        mem.append(dict(
            size=r["size"], n=r["n"], validity=r["validity"],
            weighted=r["weighted"],
            peak_live_mib=r["peak_live_mib"],
            peak_pool_total_mib=r["peak_pool_total_mib"],
            min_driver_free_mib=r["min_driver_free_mib"],
        ))
    return dict(timing=timing, memory=mem)


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
    A("# ZSSS Phase D — order/rank temporary elimination (Track P2) — "
      "profiling report\n")
    A("- Host: Linux TINYDEBIAN; GPU: NVIDIA %s (CC %d.%d), cupy %s, "
      "numpy %s, python %s" % (dev["name"].decode(), dev["major"],
                               dev["minor"], cp.__version__, np.__version__,
                               sys.version.split()[0]))
    A("- Witnesses: N_batch 20/32/50 (slow path) at 480x270 partial "
      "validity (nan_frac=0.05), spikes 3%% (+400), kappa=%s "
      "winsor_limits=%s apply_rewinsor=True max_iters=5 kappa_decay=0.9, "
      "weights uniform(0.4,1.6) float32." % (KAPPA, WINSOR_LIMITS))

    A("\n## 1. Exact-equivalence proof — optimized slow path == legacy slow "
      "path (bitwise)\n")
    A("The optimized slow path (value-clip + direct cp.sort, no int64 "
      "order/rank) vs the pre-Phase-D slow path (argsort + inverse-rank "
      "scatter) on the SAME 480x270 partial weighted witness:\n")
    A(fmt_table(
        ["N_batch", "result bitwise", "weight map bitwise", "pct equal",
         "rejected_pct"],
        [[r["n"], "YES", "YES", "YES", "%.4f" % r["rejected_pct"]]
         for r in results["bitwise"]]))
    A("\nProbe marks of an optimized slow-path run (N=20): %d legacy int64 "
      "sort marks (winsor_argsort / winsor_rank_argsort / winsor_take_along "
      "/ bounds_argsort / bounds_take_along) vs %d direct-sort marks "
      "(winsor_direct_sort / bounds_direct_sort) -> %s."
      % (results["marks"]["n_old"], results["marks"]["n_direct"],
         "NO argsort machinery remains" if results["marks"]["n_old"] == 0
         else "UNEXPECTED argsort machinery present"))

    A("\n## 2. Stage timing — optimized slow vs legacy slow at the SAME "
      "N_batch (480x270 partial weighted, CUDA events, median of 5)\n")
    A(fmt_table(
        ["config", "N", "path", "fn wall ms", "dev timeline ms",
         "sort-phase dev ms", "other dev ms", "sync-stall wall ms",
         "rej%"],
        [[r["tag"], r["n"],
          "legacy slow" if r["legacy"] else "optimized slow",
          r["fn_wall_ms"], r["device_total_ms"], r["sort_phase_ms"],
          r["other_phase_ms"], r["sync_wall_ms"], "%.2f" % r["rejected_pct"]]
         for r in results["timing"]]))
    A("\nSame-N deltas (optimized vs legacy; negative = faster):\n")
    pairs = {}
    for r in results["timing"]:
        pairs.setdefault(r["n"], {})[r["legacy"]] = r
    for n in SLOW_N:
        new, leg = pairs[n][False], pairs[n][True]
        d_dev = new["device_total_ms"] - leg["device_total_ms"]
        d_sort = new["sort_phase_ms"] - leg["sort_phase_ms"]
        d_wall = new["fn_wall_ms"] - leg["fn_wall_ms"]
        d_sync = new["sync_wall_ms"] - leg["sync_wall_ms"]
        A("- N=%d: dev %.2f -> %.2f ms (%+.2f ms, %+.1f%%); sort-phase dev "
          "%.2f -> %.2f ms (%+.2f ms); fn wall %.2f -> %.2f ms (%+.2f ms); "
          "sync-stall wall %.2f -> %.2f ms (%+.2f ms)."
          % (n, leg["device_total_ms"], new["device_total_ms"], d_dev,
             100.0 * d_dev / leg["device_total_ms"],
             leg["sort_phase_ms"], new["sort_phase_ms"], d_sort,
             leg["fn_wall_ms"], new["fn_wall_ms"], d_wall,
             leg["sync_wall_ms"], new["sync_wall_ms"], d_sync))

    A("\n## 3. Memory — optimized vs legacy slow path (sync-per-boundary "
      "snapshots, 480x270 partial weighted)\n")
    A(fmt_table(
        ["config", "N", "path", "peak live (pool used) MiB",
         "pool total high-water MiB", "min driver free MiB"],
        [[r["tag"], r["n"],
          "legacy slow" if r["legacy"] else "optimized slow",
          r["peak_live_mib"], r["peak_pool_total_mib"],
          r["min_driver_free_mib"]]
         for r in results["memory"]]))
    mem_pairs = {}
    for r in results["memory"]:
        mem_pairs.setdefault(r["n"], {})[r["legacy"]] = r
    for n in SLOW_N:
        if n not in mem_pairs or not (False in mem_pairs[n]
                                      and True in mem_pairs[n]):
            continue
        new, leg = mem_pairs[n][False], mem_pairs[n][True]
        if "peak_live_mib" not in leg or "peak_live_mib" not in new:
            continue
        A("- N=%d: peak live %.2f MiB (legacy) -> %.2f MiB (optimized): "
          "%.2f MiB / %.1f%% less live memory; pool high-water %.2f -> "
          "%.2f MiB.  Theoretical int64 order+rank payload at N=%d: "
          "2*N*pixels*8 = %.1f MiB."
          % (n, leg["peak_live_mib"], new["peak_live_mib"],
             leg["peak_live_mib"] - new["peak_live_mib"],
             100.0 * (leg["peak_live_mib"] - new["peak_live_mib"])
             / leg["peak_live_mib"],
             leg["peak_pool_total_mib"], new["peak_pool_total_mib"],
             n, 2 * n * 480 * 270 * 8 / MB))

    A("\n## 4. 1080p fit probes (2 GiB MX150, slow path)\n")
    A(fmt_table(
        ["config", "N", "path", "outcome", "rejected_pct"],
        [[r["tag"], r["n"],
          "legacy slow" if r["legacy"] else "optimized slow",
          r["outcome"],
          "%.2f" % r["rejected_pct"] if "rejected_pct" in r else "-"]
         for r in results["fit1080p"]]))

    A("\n## 5. CPU parity of the optimized slow path (480x270 partial "
      "weighted, tolerance rtol=1e-3 atol=1e-2, |pct diff| <= 1.0)\n")
    A(fmt_table(
        ["N_batch", "res_ok", "w_ok", "pct_ok", "differing px",
         "pct_cpu", "pct_gpu", "worst specimen"],
        [[r["n"], r["res_ok"], r["w_ok"], r["pct_ok"], r["differing"],
          "%.4f" % r["pct_cpu"], "%.4f" % r["pct_gpu"],
          "%s |gpu-cpu|=%.4g (cpu %.4g -> gpu %.4g)" % (
              r["worst"]["coords"], r["worst"]["max_abs"],
              r["worst"]["cpu"], r["worst"]["gpu"])
          if r["worst"] else "-"]
         for r in results["parity"]]))

    A("\n## 6. Reduced Phase A driver re-run on the optimized tree "
      "(N 20/32/50; identical stage bucketing as results_phaseA)\n")
    A("Per-stage device sums (ms) at 480x270 partial weighted:\n")
    rows = [r for r in results["phaseA"]["timing"]
            if r["size"] == "480x270" and r["validity"] == "partial"
            and r["weighted"]]
    stage_names = ["argsort_1", "argsort_rank(inverse)", "winsor_gather",
                   "winsor_sort_key", "winsor_bound_take", "winsor_replace",
                   "rewinsor_argsort", "rewinsor_gather", "nanmean",
                   "nanstd", "new_mask"]
    hdr = ["N"] + stage_names + ["dev total"]
    A(fmt_table(
        hdr,
        [[r["n"]] + ["%.2f" % r["phases"].get(s, 0.0) for s in stage_names]
         + ["%.2f" % r["device_total_ms"]] for r in rows]))
    A("(the pre-optimization baseline for the same configs from "
      "results_phaseA: N=20 dev total 855.8 ms, N=32 1374.3 ms, N=50 "
      "2389.0 ms with argsort_1 + argsort_rank(inverse) + winsor_gather "
      "dominating the timeline; peak live 120.1 / 186.9 / 287.0 MiB.)")
    mem_rows = [r for r in results["phaseA"]["memory"]
                if r.get("size") == "480x270" and r.get("validity") ==
                "partial" and r.get("weighted")]
    if mem_rows:
        A(fmt_table(
            ["N", "peak live MiB", "peak pool MiB", "min free MiB"],
            [[r["n"], r["peak_live_mib"], r["peak_pool_total_mib"],
              r["min_driver_free_mib"]] for r in mem_rows]))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(
        REPO, "profiling", "results_phaseD"))
    ap.add_argument("--report-body", default=os.path.join(
        REPO, "profiling", "results_phaseD", "report_body.md"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--render-only", action="store_true",
                    help="re-render report_body.md from an existing "
                         "phaseD_results.json (no GPU sweeps)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    LOG_PATH = os.path.join(args.outdir, "progress.log")

    def log(msg):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")

    if args.render_only:
        json_path = os.path.join(args.outdir, "phaseD_results.json")
        with open(json_path) as f:
            results = json.load(f)
        with open(args.report_body, "w") as f:
            f.write(render_report_body(results))
        log("rendered %s from %s" % (args.report_body, json_path))
        return

    import faulthandler
    faulthandler.dump_traceback_later(120, repeat=True)

    cp.get_default_memory_pool().free_all_blocks()
    results = {}
    log("[1/7] warmup")
    sg._ensure_probe()
    imgs = make_images(4, (128, 128), seed=1, nan_frac=0.05)
    gpu_ref(imgs, make_weights(4))
    log("[2/7] bit-identity optimized == legacy (N 20/32/50)")
    results["bitwise"] = bitwise_identity(log=log)
    log("[3/7] probe marks proof")
    results["marks"] = marks_proof(log=log)
    log("[4/7] stage timing sweep (optimized vs legacy, N 20/32/50)")
    results["timing"] = timing_sweep(quick=args.quick, log=log)
    log("[5/7] memory sweep (optimized vs legacy, N 20/32/50)")
    results["memory"] = memory_sweep(log=log)
    log("[6/7] 1080p fit probes")
    results["fit1080p"] = fit_probes_1080p(log=log)
    log("[7/7] CPU parity + reduced Phase A driver re-run")
    results["parity"] = parity_sweep(log=log)
    results["phaseA"] = phaseA_rerun(quick=args.quick, log=log)
    log("rendering report body")

    json_path = os.path.join(args.outdir, "phaseD_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=1, default=str)
    with open(args.report_body, "w") as f:
        f.write(render_report_body(results))
    log("artifacts written: %s, %s" % (json_path, args.report_body))


if __name__ == "__main__":
    main()
