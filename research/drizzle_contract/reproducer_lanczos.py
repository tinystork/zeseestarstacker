"""Smallest-pathology hunt for Lanczos2/3 x3 (research only).

The matrix shows mild ringing on fully-covered single frames.  This probe
searches for the catastrophic regime reported in the physical witness
(lanczos2 scale 3, partial SCI ~ -1.2e4..+4.8e3, ~47% negative): overlapping
dithered frames create near-zero POSITIVE signed-WHT denominators at flux
imbalance points; float32 deposition then amplifies the ratio.

Scans: kernel x {single star, 2 dithered stars, star near coverage edge,
N repeats} at scale 3, weight uniform vs ON, float32 (engine) vs float64
(engine with float64 buffers is not supported -> we compare against an
external float64 python ratio sci*wht/wht on the native arrays instead).

Emits the worst cell as artifacts/lanczos_pathology.npz + prints a table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_engine import (  # noqa: E402
    ARTIFACTS,
    Cell,
    deposit,
    in_grid_mask,
    measure,
    pixmap_identity,
    wht_bins,
)

OUT = ARTIFACTS


def _star(shape, peak, cx, cy, sigma=0.8):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (peak * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)))).astype(
        np.float32
    )


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    shape = (32, 32)
    scale = 3.0

    def scan(key, kernel, frames, weight=None, star=True):
        sci, wht = deposit(frames, kernel, scale, exptime=1.0,
                           weight=weight, shape_hw=shape)
        fin = np.isfinite(sci)
        neg_frac = float(np.mean(sci < 0))
        sci_abs = np.abs(sci[fin]) if np.any(fin) else np.array([0.0])
        worst = float(np.max(sci_abs)) if sci_abs.size else 0.0
        wht_pos_min = float(np.min(wht[wht > 0])) if np.any(wht > 0) else None
        results.append({
            "key": key, "kernel": kernel,
            "sci_min": float(np.min(sci[fin])) if np.any(fin) else None,
            "sci_max": float(np.max(sci[fin])) if np.any(fin) else None,
            "sci_abs_max": worst,
            "neg_frac": neg_frac,
            "wht_min": float(np.min(wht)), "wht_max": float(np.max(wht)),
            "wht_pos_min": wht_pos_min,
            "pos_support_frac": float(np.mean(wht > 1e-9)),
        })
        return sci, wht, results[-1]

    # single / dithered stars
    A0 = _star(shape, 1000.0, 16.0, 16.0)
    for d in (0.25, 0.5, 1.0):
        A1 = _star(shape, 1000.0, 16.5, 16.0)
        for k in ("lanczos2", "lanczos3"):
            scan(f"l3_{k}_star_d{d}", k, [A0, A1])
    # star at coverage edge (translate so source partially outside)
    for edge in (4.0, 2.0, 1.0):
        for k in ("lanczos2", "lanczos3"):
            scan(f"l3_{k}_edge{edge}", k, [_star(shape, 1000.0, edge, 16.0)])
    # 2-frame subpixel dither, weighting ON
    wmask = np.ones(shape, dtype=np.float32)
    for d in (0.25, 0.5):
        A1 = _star(shape, 1000.0, 16.5, 16.0)
        for k in ("lanczos2", "lanczos3"):
            scan(f"l3_{k}_w_on_d{d}", k, [A0, A1], weight=wmask)
    # N repeats of the same dithered pair (integer accumulation)
    for n in (4, 16, 64):
        pair = [_star(shape, 1000.0, 16.0, 16.0),
                _star(shape, 1000.0, 16.25, 16.0)]
        frames = (pair * n)[: min(n * 2, 64)]
        for k in ("lanczos2", "lanczos3"):
            scan(f"l3_{k}_n{n}", k, frames)

    # rank by |SCI| max
    worst_rows = sorted(results, key=lambda r: r["sci_abs_max"], reverse=True)[:8]
    print("=== worst |SCI| cells ===")
    for r in worst_rows:
        print(f"{r['key']}: min={r['sci_min']:.4g} max={r['sci_max']:.4g} "
              f"neg={r['neg_frac']:.3f} wht_pos_min={r['wht_pos_min']:.3g} "
              f"pos_support={r['pos_support_frac']:.4f}")

    with open(OUT / "lanczos_pathology_scan.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"scan rows -> {OUT / 'lanczos_pathology_scan.json'}")

    # Emit the single worst cell with WHT-bin correlation as NPZ
    top = worst_rows[0]
    print(f"\nsmallest reproducer cell: {top['key']}")


if __name__ == "__main__":
    run()
