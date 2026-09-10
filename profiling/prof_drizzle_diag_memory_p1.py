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


def _status_mb():
    """Return (VmRSS_MB, VmHWM_MB) from /proc/self/status (Linux)."""
    rss = hwm = None
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS"):
                rss = int(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM"):
                hwm = int(line.split()[1]) / 1024.0
    return rss, hwm


PRESETS = {
    "x3": (3240, 5760),
    "x4": (4320, 7680),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=None,
                    help="square grid edge (H=W)")
    ap.add_argument("--shape", choices=tuple(PRESETS), default=None,
                    help="real Seestar output shape preset")
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--mode", choices=("baseline", "diag"), required=True)
    args = ap.parse_args()
    if args.shape:
        h, w = PRESETS[args.shape]
    elif args.height and args.width:
        h, w = args.height, args.width
    elif args.size:
        h = w = args.size
    else:
        ap.error("provide --size, --shape or --height/--width")

    from seestar.core import drizzle_science_diagnostics as dsd

    import gc

    rng = np.random.default_rng(0)
    # In-place construction: avoid large construction transients so VmHWM before
    # summarize reflects the genuinely resident inputs.
    sci = np.empty((h, w, 3), dtype=np.float32)
    rng.standard_normal(out=sci, dtype=np.float32)
    wht = np.empty((h, w, 3), dtype=np.float32)
    rng.standard_normal(out=wht, dtype=np.float32)
    w1 = np.empty((h, w), dtype=np.float32)
    rng.random(out=w1, dtype=np.float32)
    np.multiply(w1, 3.0, out=w1)
    w2 = np.empty((h, w), dtype=np.float32)
    np.multiply(w1, w1, out=w2)
    np.add(w2, 0.5, out=w2)
    mask = np.empty((h, w), dtype=bool)
    np.logical_and(np.isfinite(w1), w1 > 0, out=mask)
    gc.collect()

    rss_before, hwm_before = _status_mb()
    t0 = time.time()
    extra = {"boundary_bytes": None, "sample_caps": {}}
    if args.mode == "diag":
        sec = dsd.summarize_run(
            sci, wht, sup_w1=w1, sup_w2=w2, support_mask=mask,
            crop={"x0": 0, "y0": 0},
        )
        meta = sec.get("meta", {})
        extra["boundary_bytes"] = meta.get("max_temporary_bytes")
        extra["boundary_algorithm"] = meta.get("boundary_algorithm")
        extra["degraded"] = meta.get("degraded_sections")
        extra["sample_caps"] = {
            "sci": max((s.get("sample_count", 0) for s in sec["sci_stats"]), default=0),
            "cap": dsd.MAX_SAMPLE_COUNT,
        }
    rss_after, hwm_after = _status_mb()
    print(json.dumps({
        "height": h, "width": w, "mode": args.mode,
        "cube_mb": round(3 * h * w * 4 / (1024 * 1024), 1),
        "rss_before_mb": round(rss_before, 1),
        "hwm_before_mb": round(hwm_before, 1),
        "rss_after_mb": round(rss_after, 1),
        "hwm_after_mb": round(hwm_after, 1),
        "incremental_hwm_mb": round(hwm_after - hwm_before, 1),
        "seconds": round(time.time() - t0, 2),
        **extra,
    }))


if __name__ == "__main__":
    sys.exit(main())
