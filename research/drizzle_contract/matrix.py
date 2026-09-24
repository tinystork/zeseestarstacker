"""Run the deterministic real-engine witness matrix (research only).

Usage: .venv/bin/python research/drizzle_contract/matrix.py
Writes research/drizzle_contract/artifacts/*.
"""

from __future__ import annotations

import csv
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
    input_constant,
    input_edge,
    input_partial,
    input_star,
    measure,
    pixmap_identity,
    save_cell,
    wht_bins,
)

KERNELS = ["square", "lanczos2", "lanczos3", "point", "turbo", "gaussian"]
ROWS = []


def _emit(cell, bins=False):
    m = measure(cell)
    row = save_cell(cell, m)
    if bins:
        row["wht_bins"] = json.dumps(wht_bins(cell.sci, cell.wht))
    ROWS.append(row)
    print(f"[{cell.key}] kernel={cell.kernel} scale={cell.scale} "
          f"sci[min,max]=({m['sci'].get('min')},{m['sci'].get('max')}) "
          f"neg={m['sci'].get('negative_fraction')} "
          f"wht[min,max]=({m['wht'].get('min')},{m['wht'].get('max')})")


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)

    # ---- Inputs (seeded/deterministic, small) ----
    A = input_constant((32, 32), value=100.0)
    B = input_star((48, 48), peak=1000.0, sigma=1.2)
    C = input_edge((32, 32), level=200.0)
    D_img, D_shift_dict = input_partial((32, 32), shift=9.0)
    D_shift = (float(D_shift_dict["shift"]), float(D_shift_dict["shift"]))
    # E: identical small frame repeated N times
    E_frame = input_constant((16, 16), value=50.0, seed=7)

    def star_mask(shape):
        m = np.ones(shape, dtype=np.float32)
        return m

    # ---- 1. Kernel x scale 1/2/3/4 on input A (constant) and B (star) ----
    for kernel in KERNELS:
        for scale in (1, 2, 3, 4):
            for name, img, extra in (("A", A, {}), ("B", B, {"star": True})):
                sci, wht = deposit([img], kernel, scale, exptime=1.0,
                                   shape_hw=img.shape[:2])
                cell = Cell(key=f"{kernel}_s{scale}_{name}", kernel=kernel,
                            scale=scale, input_name=name,
                            sci=sci, wht=wht, extras=extra)
                _emit(cell, bins=(kernel in ("lanczos2", "lanczos3")
                                  and scale == 3))

    # ---- 2. Inputs C (edge) and D (partial) at scale 3 (lanczos2/3 + square)
    # ----    and E repeats (N = 1,2,4,16,64) ----
    for kernel in ("square", "lanczos2", "lanczos3"):
        sci, wht = deposit([C], kernel, 3, shape_hw=C.shape[:2])
        _emit(Cell(key=f"{kernel}_s3_C", kernel=kernel, scale=3,
                   input_name="C", sci=sci, wht=wht))
        D_pix = pixmap_identity(D_img.shape[:2], 3, shift=D_shift)
        out_shape = (int(round(32 * 3)), int(round(32 * 3)))
        mask = in_grid_mask(D_pix, out_shape)
        d = deposit([D_img], kernel, 3, weight=np.ones_like(D_img),
                    shape_hw=D_img.shape[:2])
        sci_d, wht_d = d
        _emit(Cell(key=f"{kernel}_s3_D", kernel=kernel, scale=3,
                   input_name="D", sci=sci_d, wht=wht_d,
                   mask_in=mask))

    for n in (1, 2, 4, 16, 64):
        for kernel in ("square", "lanczos2"):
            frames = [E_frame] * n
            sci, wht = deposit(frames, kernel, 2, exptime=1.0,
                               shape_hw=E_frame.shape[:2])
            _emit(Cell(key=f"E_{kernel}_s2_n{n}", kernel=kernel, scale=2,
                       input_name="E", n_frames=n, sci=sci, wht=wht))

    # ---- 3. Current wiring vs explicit pixel_scale_ratio at scale 3 ----
    for kernel in ("square", "gaussian", "turbo", "lanczos2", "lanczos3"):
        # current ZSSS wiring: no pixel_scale_ratio (upstream default 1.0)
        sci_cur, wht_cur = deposit([B], kernel, 3, shape_hw=B.shape[:2])
        _emit(Cell(key=f"AB_{kernel}_s3_current", kernel=kernel, scale=3,
                   input_name="B", sci=sci_cur, wht=wht_cur,
                   pixel_scale_ratio=1.0, extras={"star": True}))
        # explicit geometrically-derived pixel_scale_ratio = scale
        sci_exp, wht_exp = deposit([B], kernel, 3, pixel_scale_ratio=3.0,
                                   shape_hw=B.shape[:2])
        _emit(Cell(key=f"AB_{kernel}_s3_explicit_psr3", kernel=kernel, scale=3,
                   input_name="B", sci=sci_exp, wht=wht_exp,
                   pixel_scale_ratio=3.0, extras={"star": True}))

    # ---- 4. Weighting OFF/ON, square/lanczos2/lanczos3 at scale 3 ----
    wmask = np.ones(A.shape, dtype=np.float32)
    wmask[4:8, 20:26] = 0.5  # deterministic non-uniform valid mask
    for kernel in ("square", "lanczos2", "lanczos3"):
        sci_off, wht_off = deposit([A], kernel, 3, shape_hw=A.shape[:2])
        _emit(Cell(key=f"W_{kernel}_s3_off", kernel=kernel, scale=3,
                   input_name="A", weighting="off", sci=sci_off, wht=wht_off))
        sci_on, wht_on = deposit([A], kernel, 3, weight=wmask,
                                 shape_hw=A.shape[:2])
        _emit(Cell(key=f"W_{kernel}_s3_on", kernel=kernel, scale=3,
                   input_name="A", weighting="on", sci=sci_on, wht=wht_on))

    # ---- 5. Exposure witnesses 10/20/30 s, square, scale 1 & 3 ----
    for expt in (10.0, 20.0, 30.0):
        for scale in (1, 3):
            sci, wht = deposit([A], "square", scale, exptime=expt,
                               shape_hw=A.shape[:2])
            _emit(Cell(key=f"EXP_square_s{scale}_t{int(expt)}", kernel="square",
                       scale=scale, input_name="A", exptime=expt,
                       sci=sci, wht=wht))

    # ---- 6. Continuous == checkpoint+resume (square x3, lanczos2 x3) ----
    frames = [input_constant((16, 16), value=100.0, seed=i) for i in range(3)]
    for kernel in ("square", "lanczos2"):
        # continuous
        sci_c, wht_c = deposit(frames, kernel, 2, exptime=5.0,
                               shape_hw=frames[0].shape[:2])
        # resume: 1 frame, snapshot, reconstruct with disable_ctx=True
        sci_1, wht_1 = deposit([frames[0]], kernel, 2, exptime=5.0,
                               shape_hw=frames[0].shape[:2])
        total = 5.0
        out_img = np.array(sci_1, dtype=np.float32, copy=True)
        out_wht = np.array(wht_1, dtype=np.float32, copy=True)
        # build output grid for scale 2 (16x16 -> 32x32)
        d2 = None
        # second deposit continues from the snapshot
        d = deposit([frames[1], frames[2]], kernel, 2, exptime=5.0,
                    resume_state=(out_img, out_wht, total), disable_ctx=True,
                    shape_hw=frames[0].shape[:2])
        sci_r, wht_r = d
        same_img = np.array_equal(sci_c, sci_r)
        same_wht = np.array_equal(wht_c, wht_r)
        ROWS.append({
            "key": f"resume_{kernel}_s2",
            "continuous_eq_img": bool(same_img),
            "continuous_eq_wht": bool(same_wht),
        })
        print(f"[resume_{kernel}_s2] sci equal={same_img} "
              f"wht equal={same_wht} max|diff_img|="
              f"{np.max(np.abs(sci_c - sci_r)) if not same_img else 0.0}")

    # ---- write aggregated metrics ----
    with open(ARTIFACTS / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ROWS[0].keys()))
        writer.writeheader()
        for row in ROWS:
            writer.writerow({k: row.get(k) for k in ROWS[0].keys()})
    with open(ARTIFACTS / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(ROWS, fh, indent=1)
    print(f"\nwrote {len(ROWS)} rows -> {ARTIFACTS / 'metrics.csv'}")


if __name__ == "__main__":
    main()
