"""High-contrast / partial-overlap Lanczos probe (research only).

Chases the reported catastrophic regime (-1.2e4..+4.8e3, ~47% negative) with
bright stars on a partial footprint, subpixel dithers and float32 engine
deposition at scale 3.  Emits per-cell WHT-bin/SCI correlation for any cell
whose |SCI| exceeds 1e3 and saves the worst reproducer NPZ.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_engine import (  # noqa: E402
    ARTIFACTS,
    deposit,
    wht_bins,
)

OUT = ARTIFACTS


def star(shape, peak, cx, cy, sigma=0.9, bg=100.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    img = np.full(shape, bg, dtype=np.float32)
    img += (
        peak
        * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)))
    ).astype(np.float32)
    return img.astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shape = (48, 48)
    results = []
    scale = 3.0

    def cell(key, kernel, frames):
        sci, wht = deposit(frames, kernel, scale, exptime=1.0,
                           shape_hw=shape)
        fin = np.isfinite(sci)
        rec = {
            "key": key, "kernel": kernel,
            "sci_min": float(np.min(sci[fin])) if np.any(fin) else None,
            "sci_max": float(np.max(sci[fin])) if np.any(fin) else None,
            "sci_abs_max": float(np.max(np.abs(sci[fin]))) if np.any(fin) else 0.0,
            "neg_frac": float(np.mean(sci < 0)),
            "wht_min": float(np.min(wht)),
            "wht_max": float(np.max(wht)),
            "wht_pos_min": float(np.min(wht[wht > 0])) if np.any(wht > 0) else None,
        }
        results.append(rec)
        print(f"{key}: sci[{rec['sci_min']:.4g},{rec['sci_max']:.4g}] "
              f"|max|={rec['sci_abs_max']:.4g} neg={rec['neg_frac']:.3f} "
              f"wht_pos_min={rec['wht_pos_min']:.3g}")
        if rec["sci_abs_max"] > 1e3:
            np.savez_compressed(str(OUT / key) + ".npz", sci=sci, wht=wht)
            bins = wht_bins(sci, wht)
            with open(str(OUT / key) + "_bins.json", "w") as fh:
                json.dump(bins, fh, indent=1)
            print("   -> saved NPZ + bins (catastrophic cell)")
        return sci, wht

    # bright star on background, 2/3 subpixel dithers
    for peak in (5e3, 2e4, 1e5):
        frames = [
            star(shape, peak, 24.0, 24.0),
            star(shape, peak, 24.33, 24.0),
        ]
        for k in ("lanczos2", "lanczos3"):
            cell(f"hc_{k}_p{peak:g}_d2", k, frames)
    # 3-frame dither cycle
    peaks = [1e4, 1e5]
    for peak in peaks:
        frames = [
            star(shape, peak, 24.0, 24.0),
            star(shape, peak, 24.25, 24.5),
            star(shape, peak, 23.8, 24.2),
        ]
        for k in ("lanczos2", "lanczos3"):
            cell(f"hc_{k}_p{peak:g}_d3", k, frames)
    # bright star at partial-coverage border (shifted out of grid)
    for shift in (6.0, 3.0):
        s = star(shape, 1e5, 24.0 - shift, 24.0)
        for k in ("lanczos2", "lanczos3"):
            cell(f"hc_{k}_border{shift:g}", k, [s])
    # double star where one sits on the other's negative lobe
    for k in ("lanczos2", "lanczos3"):
        f0 = star(shape, 1e5, 24.0, 24.0, sigma=1.2)
        # second star ~2.2 px away at scale-3-grid = 2.2*3 lobes distance
        f1 = star(shape, 1e5, 24.0 + 2.2, 24.0, sigma=1.2)
        cell(f"hc_{k}_double", k, [f0, f1])

    worst = max(results, key=lambda r: r["sci_abs_max"])
    print("\nworst:", worst["key"], worst["sci_abs_max"], worst["neg_frac"])
    with open(OUT / "lanczos_high_contrast.json", "w") as fh:
        json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
