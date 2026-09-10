"""P2-C bounded conditioning-rule research probe (research; no production change).

Quantifies, on the REAL delivered Phase-1/P2-A artifacts, whether the signed
native-WHT conditioning proxy separates unstable from stable divisions, and how
a candidate threshold trades false positives against coverage.

Rule family evaluated (the form identified in Phase 1 / P2-A):

    quality(p) = |WHT(p)| / ref(p)          evaluated only on physical support
    reject if quality(p) < threshold

with ``ref`` a robust POSITIVE native-WHT reference, either

  * ``global_positive_p90`` — p90 of all positive WHT samples, or
  * ``local_tiled_positive_p90`` — p90 of positive WHT inside a 64x64 tile
    (fallback to the global value when a tile has no positive sample).

Writes ``artifacts/p2c_conditioning_rule_scan.json``.
"""

from __future__ import annotations

import json
import os

import numpy as np

ROOT = "/home/tristan/Téléchargements/out"
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
OUT = os.path.join(ART, "p2c_conditioning_rule_scan.json")

TILE = 64
THRESHOLDS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
RUNS = {"L2": "lanczos2", "L3": "lanczos3", "stop-r4": "lanczos2", "square": "square"}


def _load_wht_img(run):
    ck = os.path.join(ROOT, run, ".m3d_checkpoint")
    with open(os.path.join(ck, "checkpoint.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    ch = meta["channels"][0]
    wht = np.load(os.path.join(ck, ch["out_wht"]["file"]))
    img = np.load(os.path.join(ck, ch["out_img"]["file"]))
    return wht, img, meta


def _positive_p90(a):
    pos = a[a > 0.0]
    if pos.size == 0:
        return None
    return float(np.percentile(pos, 90.0))


def _local_ref(wht, tile=TILE):
    """Tiled positive-p90 reference (bounded chunks), None tiles use global."""
    glob = _positive_p90(wht)
    ref = np.full(wht.shape, np.nan, dtype=np.float64)
    h, w = wht.shape
    for r0 in range(0, h, tile):
        for c0 in range(0, w, tile):
            blk = wht[r0:r0 + tile, c0:c0 + tile]
            val = _positive_p90(blk)
            ref[r0:r0 + tile, c0:c0 + tile] = glob if val is None else val
    return glob, ref


def main():
    os.makedirs(ART, exist_ok=True)
    runs = []
    for run, kernel in RUNS.items():
        path = os.path.join(ROOT, run)
        if not os.path.isdir(path):
            continue
        wht, img, meta = _load_wht_img(run)
        finite = np.isfinite(wht) & np.isfinite(img)
        support = finite & (wht > 1e-9)
        glob, locref = _local_ref(wht)
        ratio_global = np.zeros(wht.shape, dtype=np.float64)
        ratio_local = np.zeros(wht.shape, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_global[support] = np.abs(wht[support]) / glob if glob else np.nan
            ratio_local[support] = np.abs(wht[support]) / locref[support]

        abs_sci = np.abs(img)
        row = {
            "run": run,
            "kernel": kernel,
            "support_pixels": int(support.sum()),
            "global_positive_p90": glob,
            "candidates": [],
        }
        base = {
            "global_positive_p90": ratio_global[support],
            "local_tiled_positive_p90": ratio_local[support],
        }
        for refname, ratio in base.items():
            entry = {"reference": refname, "thresholds": []}
            for t in THRESHOLDS:
                rej = ratio < t
                n_rej = int(rej.sum())
                rejected_sci = abs_sci[support][rej]
                entry["thresholds"].append(
                    {
                        "threshold": t,
                        "rejected_pixels": n_rej,
                        "rejected_fraction": float(n_rej / max(1, rej.size)),
                        "rejected_abs_sci_max": (
                            float(rejected_sci.max()) if n_rej else 0.0
                        ),
                        "rejected_abs_sci_p99": (
                            float(np.percentile(rejected_sci, 99)) if n_rej else 0.0
                        ),
                        "rejected_abs_sci_gt_100": int((rejected_sci > 100.0).sum()),
                        "accepted_pixels": int(rej.size - n_rej),
                        "accepted_abs_sci_max": (
                            float(abs_sci[support][~rej].max()) if (~rej).any() else 0.0
                        ),
                    }
                )
            row["candidates"].append(entry)
        runs.append(row)
    art = {
        "schema_version": "p2c.scan.1",
        "mission_id": "zsss-drizzle-scientific-closure-p2-20260910",
        "phase": "P2-C conditioning-rule research probe",
        "diagnostic_only": True,
        "sample_limits": (
            "Extrema-selected Phase-1/P2-A delivered native buffers (channel 0) "
            "and delivered final SCI; no corrected-physical-frame claim."
        ),
        "tile": TILE,
        "thresholds": list(THRESHOLDS),
        "runs": runs,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(art, fh, indent=1, sort_keys=True)
    print("wrote", OUT)
    for r in runs:
        loc = next(c for c in r["candidates"] if c["reference"] == "local_tiled_positive_p90")
        for th in loc["thresholds"]:
            print(
                f"  {r['run']:>8s} {r['kernel']:<9s} t={th['threshold']:.0e} "
                f"rej={th['rejected_pixels']:>8d} ({th['rejected_fraction']*100:6.3f}%) "
                f"rej|SCI|max={th['rejected_abs_sci_max']:.3g} "
                f"rej|SCI|>100={th['rejected_abs_sci_gt_100']:>7d} "
                f"acc|SCI|max={th['accepted_abs_sci_max']:.3g}"
            )
    return art


if __name__ == "__main__":
    main()
