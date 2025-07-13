"""Utility helpers for pre-scanning FITS headers and computing a fixed output WCS."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import math

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
from astropy import units as u


def standardize_wcs(wcs: WCS) -> WCS:
    """Return a copy of ``wcs`` converted to ICRS RA/DEC coordinates.

    This helps ensure that reprojection remains consistent regardless of
    observation location or time.
    """

    try:
        wcs_cel = wcs.celestial
    except Exception:
        return wcs

    wcs_new = wcs_cel.deepcopy()

    try:
        wcs_new.wcs.radesys = "ICRS"
        wcs_new.wcs.equinox = 2000.0
    except Exception:
        pass

    try:
        ctype1 = wcs_new.wcs.ctype[0].upper() if wcs_new.wcs.ctype else ""
        ctype2 = wcs_new.wcs.ctype[1].upper() if wcs_new.wcs.ctype else ""
        if not ctype1.startswith("RA"):
            ctype1 = "RA---TAN"
        if not ctype2.startswith("DEC"):
            ctype2 = "DEC--TAN"
        wcs_new.wcs.ctype = [ctype1, ctype2]
    except Exception:
        pass

    return wcs_new


HeaderInfo = Tuple[Tuple[int, int], WCS]


def collect_headers(filepaths: Iterable[str]) -> List[HeaderInfo]:
    """Return a list of ``(shape_hw, WCS)`` tuples for each FITS file path."""
    infos: List[HeaderInfo] = []
    for path in filepaths:
        try:
            hdr = fits.getheader(path, memmap=False)
            wcs = WCS(hdr)
            if not wcs.is_celestial:
                continue
            wcs = standardize_wcs(wcs)
            naxis1 = int(hdr.get("NAXIS1"))
            naxis2 = int(hdr.get("NAXIS2"))
            shape_hw = (naxis2, naxis1)
            infos.append((shape_hw, wcs))
        except Exception:
            continue
    return infos


def compute_final_output_grid(header_infos: Iterable[HeaderInfo], scale: float = 1.0) -> Tuple[WCS, Tuple[int, int]]:
    """Compute a fixed output WCS and shape covering all inputs."""
    header_list = list(header_infos)
    if not header_list:
        raise ValueError("No header infos provided")

    ref_wcs = standardize_wcs(header_list[0][1])

    xmin = math.inf
    ymin = math.inf
    xmax = -math.inf
    ymax = -math.inf
    pixel_scales = []

    for shape_hw, wcs in header_list:
        wcs = standardize_wcs(wcs)
        h, w = shape_hw
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=float)
        sky = wcs.pixel_to_world(corners[:, 0], corners[:, 1])
        x, y = ref_wcs.world_to_pixel(sky)
        xmin = min(xmin, float(np.min(x)))
        xmax = max(xmax, float(np.max(x)))
        ymin = min(ymin, float(np.min(y)))
        ymax = max(ymax, float(np.max(y)))
        try:
            pixel_scales.append(np.mean(np.abs(proj_plane_pixel_scales(wcs))))
        except Exception:
            pass

    if not pixel_scales:
        pixel_scales.append(np.mean(np.abs(proj_plane_pixel_scales(ref_wcs))))

    min_scale_deg = float(np.min(pixel_scales))
    output_scale_deg = min_scale_deg / max(scale, 1.0)

    xmin_f = math.floor(xmin) - 1
    ymin_f = math.floor(ymin) - 1
    xmax_f = math.ceil(xmax) + 1
    ymax_f = math.ceil(ymax) + 1

    out_w = int(xmax_f - xmin_f)
    out_h = int(ymax_f - ymin_f)

    out_wcs = WCS(naxis=2)
    out_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    out_wcs.wcs.radesys = "ICRS"
    out_wcs.wcs.equinox = 2000.0
    out_wcs.wcs.crval = list(ref_wcs.wcs.crval)
    out_wcs.wcs.crpix = [-xmin_f + 0.5, -ymin_f + 0.5]
    out_wcs.wcs.cd = np.array([[-output_scale_deg, 0.0], [0.0, output_scale_deg]])
    out_wcs.pixel_shape = (out_w, out_h)
    try:
        out_wcs._naxis1 = out_w
        out_wcs._naxis2 = out_h
    except Exception:
        pass

    return out_wcs, (out_h, out_w)

