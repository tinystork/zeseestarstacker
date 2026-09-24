"""REWORK-1 corrected geometry core (research only).

Replaces the r0 hand-claimed pixmap with the ACTUAL ZSSS helpers
(``seestar.core.drizzle_core.build_output_grid`` /
``pixmap_from_alignment``) on a deterministic celestial WCS, and proves the
numeric coordinate formula empirically.  psr candidates are the upstream
semantics (output/input pixel size): current ZSSS = omitted (default 1.0),
explicit candidate 1/s, upstream estimate (pixel_scale_ratio=None), plus an
inverted ``s`` cell kept ONLY as a wrongness diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.wcs import WCS

from drizzle.resample import Drizzle
from drizzle.utils import estimate_pixel_scale_ratio

from seestar.core.drizzle_core import build_output_grid, pixmap_from_alignment

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def make_reference_wcs(shape_hw=(48, 48), cdelt=1.0):
    """Deterministic celestial reference WCS (TAN, cdelt arcsec)."""
    h, w = shape_hw
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.cdelt = [-cdelt, cdelt]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.pixel_shape = (w, h)
    return wcs


def zsss_output(shape_hw, scale, ref_wcs=None):
    """Real ZSSS output grid for a scale factor.

    Mirrors production: ``build_output_grid`` divides CDELT and multiplies
    CRPIX by ``scale`` but leaves the WCS ``pixel_shape`` at the INPUT size;
    the production callers (``queue_manager`` grid builders) then fix the
    output WCS shape attributes (``pixel_shape`` / ``_naxis1/2``).  We do the
    same here so ``pixmap_from_alignment`` masks against the real output
    grid.
    """
    ref_wcs = ref_wcs or make_reference_wcs(shape_hw)
    out_wcs, out_shape_hw = build_output_grid(ref_wcs, shape_hw, scale)
    out_h, out_w = out_shape_hw
    out_wcs.pixel_shape = (out_w, out_h)
    out_wcs.array_shape = (out_h, out_w)
    out_wcs._naxis1 = out_w
    out_wcs._naxis2 = out_h
    return ref_wcs, out_wcs, out_shape_hw


def tf_identity():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def tf_shift(dx, dy=0.0):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)


def tf_rotate(theta_deg, cx, cy):
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    t = np.array([[c, -s, 0.0], [s, c, 0.0]], dtype=np.float64)
    # rotation about (cx, cy)
    t[0, 2] = cx - c * cx + s * cy
    t[1, 2] = cy - s * cx - c * cy
    return t


def real_pixmap(shape_hw, tf, scale, ref_wcs=None):
    """Real ZSSS pixmap for an affine tf (output pixel centres)."""
    ref_wcs, out_wcs, out_shape = zsss_output(shape_hw, scale, ref_wcs)
    pix, mask = pixmap_from_alignment(shape_hw, tf, ref_wcs, out_wcs)
    return pix, mask, out_shape, out_wcs, ref_wcs


def prove_formula(scale, shape=(24, 24)):
    """Empirically fit out = A*tf(x,y) + B for the identity transform."""
    pix, mask, out_shape, _, _ = real_pixmap(shape, tf_identity(), scale)
    yy, xx = np.indices(shape, dtype=np.float64)
    # least-squares per axis
    X = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], axis=1)
    for ax, coords in ((0, pix[..., 0]), (1, pix[..., 1])):
        coef, *_ = np.linalg.lstsq(X, coords.ravel(), rcond=None)
        # residuals after linear fit with the s*x hypothesis
        resid = np.max(np.abs(coords.ravel() - (scale * xx.ravel() * (1 - ax) +
                                                scale * yy.ravel() * ax)))
        resid_s1 = np.max(np.abs(coords.ravel() -
                                 (scale * xx.ravel() * (1 - ax) +
                                  scale * yy.ravel() * ax + (scale - 1))))
    out = {}
    pix0 = pix[..., 0]
    pix1 = pix[..., 1]
    A = np.polyfit(xx.ravel(), pix0.ravel(), 1)
    B = np.polyfit(yy.ravel(), pix1.ravel(), 1)
    out = {"scale": scale,
           "fit_x_slope": float(A[0]), "fit_x_offset": float(A[1]),
           "fit_y_slope": float(B[0]), "fit_y_offset": float(B[1]),
           "expected_if_sx_plus_sm1": scale - 1}
    return out


def deposit(data, tf, scale, *, kernel="square", exptime=1.0,
            weight_map=None, pixfrac=1.0, pixel_scale_ratio=None,
            iscale=1.0, wht_scale=None, out_img=None, out_wht=None,
            total_exptime=None, ref_wcs=None, in_units="counts"):
    """Deposit through the REAL engine on the REAL ZSSS pixmap geometry.

    Rebuilds the pixmap per frame from ``tf``; supports continued
    accumulation when ``out_img/out_wht/total_exptime`` are given (mirrors
    ``DrizzleAccumulator.from_native_state`` with ``disable_ctx=True``).
    """
    shape_hw = data.shape[:2]
    pix, mask, out_shape, _, ref = real_pixmap(shape_hw, tf, scale, ref_wcs)
    if out_img is None:
        out_img = np.zeros(out_shape, dtype=np.float32)
        out_wht = np.zeros(out_shape, dtype=np.float32)
        d = Drizzle(out_img=out_img, out_wht=out_wht, kernel=kernel,
                    fillval="0.0")
    else:
        d = Drizzle(out_img=out_img, out_wht=out_wht, kernel=kernel,
                    fillval="0.0", exptime=total_exptime, disable_ctx=True)
    wm = np.asarray(weight_map if weight_map is not None else
                    np.ones(shape_hw, dtype=np.float32), dtype=np.float32)
    wm = wm * mask.astype(np.float32)
    kwargs = dict(data=np.asarray(data, dtype=np.float32), exptime=exptime,
                  pixmap=pix, weight_map=wm, in_units=in_units,
                  pixfrac=pixfrac, wht_scale=(exptime if wht_scale is None
                                              else wht_scale))
    if pixel_scale_ratio is not None:
        kwargs["pixel_scale_ratio"] = pixel_scale_ratio
    if iscale != 1.0:
        kwargs["iscale"] = iscale
    d.add_image(**kwargs)
    return np.array(d.out_img, dtype=np.float32, copy=True), np.array(
        d.out_wht, dtype=np.float32, copy=True
    ), pix, mask


def psr_none_est(scale, shape, tf=None, ref_wcs=None):
    """Upstream estimate_pixel_scale_ratio(input->output) for reference."""
    ref_wcs, out_wcs, _ = zsss_output(shape, scale, ref_wcs)
    return estimate_pixel_scale_ratio(
        ref_wcs, out_wcs,
        refpix_from=np.array(ref_wcs.wcs.crpix),
        refpix_to=np.array(out_wcs.wcs.crpix),
    )
