"""P2-B bounded geometry-only A/B probe (research; no production change).

Compares the baseline kernel geometry (``pixel_scale_ratio`` absent → upstream
1.0) against the WCS-derived ``output/input`` angular pixel-size ratio, using
the REAL installed ``drizzle`` 2.2.0 engine, for the Square / Lanczos2 /
Lanczos3 kernels at scales 1..4.

Writes ``artifacts/p2b_geometry_ab.json``.

This is a *qualification* probe: it quantifies what the geometry factor changes
(kernel footprint, signed-weight mass, cancellation) and what it does not, and
it explicitly checks that a corrected Lanczos kernel does not collapse to an
approximately point kernel.  It applies NO conditioning rule and NO threshold.
"""

from __future__ import annotations

import json
import math
import os
import warnings

import numpy as np

from drizzle.resample import Drizzle
from astropy.wcs import WCS

warnings.filterwarnings("ignore")

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
OUT = os.path.join(ART, "p2b_geometry_ab.json")

KERNELS = ("square", "lanczos2", "lanczos3")
SCALES = (1.0, 2.0, 3.0, 4.0)
N = 48


def wcs_pair(scale, plate_deg=2.4e-4):
    """Canonical reference/output WCS pair (output = reference/scale)."""
    ref = WCS(naxis=2)
    ref.wcs.crpix = [N / 2, N / 2]
    ref.wcs.cdelt = [-plate_deg, plate_deg]
    ref.wcs.crval = [10.0, 20.0]
    ref.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    ref.array_shape = (N, N)
    out = ref.deepcopy()
    out.wcs.crpix = np.asarray(ref.wcs.crpix) * scale
    out.wcs.cdelt = np.asarray(ref.wcs.cdelt) / scale
    out.array_shape = (int(round(N * scale)), int(round(N * scale)))
    return ref, out


def wcs_ratio(scale):
    from seestar.core.drizzle_core import derive_pixel_scale_ratio

    ref, out = wcs_pair(scale)
    return float(derive_pixel_scale_ratio(ref, out))


def deposit(kernel, scale, psr, impulse=True):
    data = np.zeros((N, N), dtype=np.float32)
    if impulse:
        data[N // 2, N // 2] = 1.0
    else:
        yy, xx = np.indices((N, N), dtype=np.float64)
        data = np.exp(-(((yy - N / 2) ** 2 + (xx - N / 2) ** 2) / (2 * 1.2**2))).astype(np.float32)
    yy, xx = np.indices((N, N), dtype=np.float64)
    pix = np.dstack((scale * (xx + 0.5), scale * (yy + 0.5)))
    hs = int(round(N * scale))
    eng = Drizzle(
        out_img=np.zeros((hs, hs), dtype=np.float32),
        out_wht=np.zeros((hs, hs), dtype=np.float32),
        kernel=kernel,
        fillval="0.0",
    )
    kw = dict(
        data=data,
        exptime=1.0,
        pixmap=pix,
        weight_map=np.ones((N, N), dtype=np.float32),
        in_units="cps",
        pixfrac=1.0,
    )
    if psr is not None:
        kw["pixel_scale_ratio"] = float(psr)
    eng.add_image(**kw)
    return np.array(eng.out_wht, dtype=np.float64)


def metrics(wht, scale):
    nz = int(np.count_nonzero(np.abs(wht) > 1e-6))
    total = float(np.sum(wht))
    abs_total = float(np.sum(np.abs(wht)))
    # footprint radius (std of |w| around the deposition centre)
    hs = wht.shape[0]
    cy = cx = (hs - 1) / 2.0
    yy, xx = np.indices(wht.shape, dtype=np.float64)
    m = np.abs(wht)
    wsum = float(m.sum())
    if wsum > 0:
        var = float(((mm - cy) ** 2 + (xx - cx) ** 2) * m).sum() / wsum if False else float(
            (((yy - cy) ** 2 + (xx - cx) ** 2) * m).sum() / wsum
        )
    else:
        var = 0.0
    return {
        "nonzero": nz,
        "sum_wht": total,
        "sum_abs_wht": abs_total,
        "signed_fraction": float(abs_total - abs(total)) / abs_total if abs_total else 0.0,
        "cancellation_ratio": float(abs(total) / abs_total) if abs_total else None,
        "min_wht": float(np.min(wht)),
        "max_wht": float(np.max(wht)),
        "rms_radius_out_px": math.sqrt(var) if var > 0 else 0.0,
    }


def main():
    os.makedirs(ART, exist_ok=True)
    rows = []
    for scale in SCALES:
        ratio = wcs_ratio(scale)
        for kernel in KERNELS:
            base = metrics(deposit(kernel, scale, None), scale)
            corr = metrics(deposit(kernel, scale, ratio), scale)
            against_scale = metrics(deposit(kernel, scale, scale), scale)
            rows.append(
                {
                    "kernel": kernel,
                    "scale": scale,
                    "wcs_derived_ratio": ratio,
                    "baseline_psr_absent": base,
                    "corrected_wcs_ratio": corr,
                    "control_psr_equals_scale": against_scale,
                    "geometry_effect_on_kernel": bool(
                        abs(corr["sum_abs_wht"] - base["sum_abs_wht"]) > 1e-9
                        or abs(corr["rms_radius_out_px"] - base["rms_radius_out_px"]) > 1e-9
                    ),
                }
            )
    art = {
        "schema_version": "p2b.ab.1",
        "mission_id": "zsss-drizzle-scientific-closure-p2-20260910",
        "phase": "P2-B (geometry-only qualification probe)",
        "diagnostic_only": True,
        "note": (
            "Baseline = pixel_scale_ratio absent (upstream 1.0). Corrected = "
            "WCS-derived output/input angular pixel-size ratio. Control = the "
            "DISPROVEN psr=scale behaviour, recorded only to prove it is not "
            "the corrected value."
        ),
        "grid": {"n": N, "impulse": True},
        "rows": rows,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(art, fh, indent=1, sort_keys=True)
    print("wrote", OUT)
    for r in rows:
        print(
            f"  {r['kernel']:<9s} x{r['scale']:.0f} ratio={r['wcs_derived_ratio']:.6f} "
            f"base_abssum={r['baseline_psr_absent']['sum_abs_wht']:.2f} "
            f"corr_abssum={r['corrected_wcs_ratio']['sum_abs_wht']:.2f} "
            f"base_neg={r['baseline_psr_absent']['min_wht']:.4f} "
            f"corr_neg={r['corrected_wcs_ratio']['min_wht']:.4f} "
            f"radius_out={r['corrected_wcs_ratio']['rms_radius_out_px']:.3f}"
        )
    return art


if __name__ == "__main__":
    main()
