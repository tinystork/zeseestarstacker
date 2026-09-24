"""REWORK-1 corrected scenes + measurements (research only).

* Aperture/background photometry for the star (bounded aperture, background
  annulus) — no global-min subtraction.
* FWHM from the radial profile inside the aperture.
* Area-normalised flux accounting for the output pixel area (scale s).
* Independent positive support pair: a SEPARATE square-kernel flux-free
  drizzle deposit (same geometry/pixmap/weight) gives W1/W2 coverage
  semantics like the Standard-M3 drizzle_sup pair.
* Honest excursion criterion: output SCI excursion beyond the INPUT domain
  (data min/max mapped through the same scene), reported separately from
  bright-input magnitude.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geo import deposit, real_pixmap, tf_identity  # noqa: E402


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

def scene_constant(shape=(48, 48), value=100.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full(shape, value, dtype=np.float32)
    if noise > 0:
        img += rng.normal(0.0, noise, shape).astype(np.float32)
    return img.astype(np.float32)


def scene_star(shape=(48, 48), peak=1000.0, cx=24.0, cy=24.0, sigma=1.5,
               bg=0.0, noise=0.0, seed=0):
    """Analytic Gaussian star sampled at input pixel centres (x index)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    img = bg + peak * np.exp(
        -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    )
    if noise > 0:
        img += rng.normal(0.0, noise, shape)
    return img.astype(np.float32)


def scene_edge(shape=(48, 48), level=200.0):
    img = np.zeros(shape, dtype=np.float32)
    img[:, : shape[1] // 2] = level
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Measurements (input-domain honest)
# ---------------------------------------------------------------------------

def stats_2d(a):
    a = np.asarray(a, dtype=np.float64)
    f = a[np.isfinite(a)]
    if f.size == 0:
        return {}
    return {
        "min": float(f.min()), "max": float(f.max()),
        "mean": float(f.mean()), "median": float(np.median(f)),
        "p1": float(np.percentile(f, 1)), "p5": float(np.percentile(f, 5)),
        "p95": float(np.percentile(f, 95)), "p99": float(np.percentile(f, 99)),
        "neg_frac": float(np.mean(a < 0)),
        "pos_frac": float(np.mean(a > 0)),
        "nonfinite": float(np.mean(~np.isfinite(a))),
    }


def aperture_photometry(sci, cx, cy, r_apert=6.0, r_bg_in=9.0, r_bg_out=12.0):
    """Bounded-aperture photometry with local background annulus.

    Returns aperture sum (background-subtracted), background level,
    centroid (first moment on background-subtracted positive signal inside the
    aperture) and FWHM estimate from the radial profile.
    """
    h, w = sci.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ap = rr <= r_apert
    bg_ann = (rr > r_bg_in) & (rr <= r_bg_out)
    s = sci.astype(np.float64)
    if np.count_nonzero(bg_ann) >= 8:
        bg = float(np.median(s[bg_ann]))
    else:
        bg = 0.0
    sb = s - bg
    ap_sel = sb * ap
    total = float(np.sum(ap_sel))
    if total > 0:
        cxs = float(np.sum((xx * ap_sel)) / np.sum(ap_sel))
        cys = float(np.sum((yy * ap_sel)) / np.sum(ap_sel))
    else:
        cxs = cys = None
    # FWHM from radial profile (0.5*peak radius)
    peak = float(np.max(sb * ap)) if np.any(ap) else 0.0
    fwhm = None
    if peak > 0:
        prof = []
        for r in np.arange(0.5, r_apert + 0.5, 0.5):
            ring = (rr > r - 0.5) & (rr <= r + 0.5) & ap
            if np.any(ring):
                prof.append((r, float(np.median(sb[ring]))))
        arr = np.array(prof) if prof else np.zeros((0, 2))
        half = peak / 2.0
        above = arr[arr[:, 1] >= half]
        if above.size:
            fwhm = float(2.0 * above[0, 0])
    return {"aperture_sum": total, "background": bg, "peak": peak,
            "centroid_x": cxs, "centroid_y": cys, "fwhm": fwhm,
            "n_aperture_pix": int(np.count_nonzero(ap))}


def support_pair(data, tf, scale, *, kernel="square", exptime=1.0,
                 weight_map=None, pixfrac=1.0, ref_wcs=None):
    """Independent positive support: square-kernel WHT of the SAME geometry.

    Mirrors the M3 support semantics: coverage (validity) is deposited with
    the square kernel on identical pixmap geometry, so the native WHT of this
    separate accumulator is the independent positive-support domain (never
    signed, never Lanczos).  ``wht_scale=1`` keeps it a pure coverage count.
    """
    w = np.ones(data.shape[:2], dtype=np.float32)
    if weight_map is not None:
        w = np.asarray(weight_map, dtype=np.float32)
    sci_sup, wht_sup, _, _ = deposit(
        np.ones(data.shape[:2], dtype=np.float32), tf, scale,
        kernel="square", exptime=exptime, weight_map=w,
        pixfrac=pixfrac, wht_scale=1.0, ref_wcs=ref_wcs,
    )
    return wht_sup  # coverage counts, positive domain


def excursion(sci, data_min, data_max, wht, eps=1e-9):
    """Honest excursion of SCI on WHT>eps relative to the INPUT domain."""
    pos = (wht > eps) & np.isfinite(sci)
    s = sci[pos]
    if s.size == 0:
        return {"valid_support_pixels": 0}
    lo = float(s.min()) - float(data_min)
    hi = float(s.max()) - float(data_max)
    return {
        "valid_support_pixels": int(s.size),
        "sci_min": float(s.min()),
        "sci_max": float(s.max()),
        "below_input_min": lo,
        "above_input_max": hi,
        "overshoot_factor_max": float(s.max()) / float(data_max)
        if data_max > 0 else None,
        "undershoot_factor_min": float(s.min()) / float(data_min)
        if data_min > 0 else None,
        "neg_frac_on_support": float(np.mean(s < 0)),
    }
