"""Isolated memory witness for the P1 Drizzle science diagnostics.

ZSSS-DRIZZLE-CLOSURE-P1 rework-1 (F3).  Runs one synthetic size per process and
reports the peak-RSS delta introduced by ``summarize_run`` on top of the
resident arrays.  Not a CI test (memory numbers are environment dependent); it
is an operator/witness probe.

Usage:
    python profiling/prof_drizzle_diag_memory_p1.py --size 3072 --mode diag
    python profiling/prof_drizzle_diag_memory_p1.py --size 3072 --mode baseline
"""

import argparse
import json
import resource
import sys
import time

import numpy as np


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--mode", choices=("baseline", "diag"), required=True)
    args = ap.parse_args()
    n = args.size

    from seestar.core import drizzle_science_diagnostics as dsd

    rng = np.random.default_rng(0)
    sci = rng.normal(size=(n, n, 3)).astype(np.float32)
    wht = rng.normal(size=(n, n, 3)).astype(np.float32)
    w1 = (np.abs(rng.random((n, n))) * 3.0).astype(np.float32)
    w2 = (w1 * w1 + 0.5).astype(np.float32)
    mask = np.isfinite(w1) & (w1 > 0)

    base = _rss_mb()
    t0 = time.time()
    extra = {"boundary_bytes": None, "sample_caps": {}}
    if args.mode == "diag":
        sec = dsd.summarize_run(
            sci, wht, sup_w1=w1, sup_w2=w2, support_mask=mask,
            crop={"x0": 0, "y0": 0},
        )
        wb = sec["meta"]["boundary_work_buffer"]
        extra["boundary_bytes"] = wb.get("bytes")
        extra["sample_caps"] = {
            "sci": max((s.get("sample_count", 0) for s in sec["sci_stats"]), default=0),
            "cap": dsd.MAX_SAMPLE_COUNT,
        }
    peak = _rss_mb()
    print(json.dumps({
        "size": n, "mode": args.mode,
        "cube_mb": 3 * n * n * 4 / (1024 * 1024),
        "rss_before_mb": round(base, 1), "rss_peak_mb": round(peak, 1),
        "delta_mb": round(peak - base, 1),
        "seconds": round(time.time() - t0, 2),
        **extra,
    }))


if __name__ == "__main__":
    sys.exit(main())
