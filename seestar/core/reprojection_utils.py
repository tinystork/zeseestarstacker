"""Utility helpers for pre-scanning FITS headers and computing a fixed output WCS."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import math

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


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
            naxis1 = int(hdr.get("NAXIS1"))
            naxis2 = int(hdr.get("NAXIS2"))
            shape_hw = (naxis2, naxis1)
            infos.append((shape_hw, wcs))
        except Exception:
            continue
    return infos


def compute_final_output_grid(
    header_infos: Iterable[HeaderInfo],
    scale: float = 1.0,
    *,
    auto_rotate: bool = False,
) -> Tuple[WCS, Tuple[int, int]]:
    """Compute a fixed output WCS and shape covering all inputs.

    Parameters
    ----------
    header_infos : iterable of ``(shape_hw, WCS)``
        Pre-parsed header information as returned by :func:`collect_headers`.
    scale : float, optional
        Drizzle scale factor to apply to the output grid. Defaults to ``1.0``.
    auto_rotate : bool, optional
        If ``True`` the resulting grid orientation is chosen to minimise the
        bounding box of all inputs. Otherwise the orientation of the first
        header is preserved.
    """
    header_list = list(header_infos)
    if not header_list:
        raise ValueError("No header infos provided")

    if auto_rotate:
        center_ra = np.median([w.wcs.crval[0] for _, w in header_list])
        center_dec = np.median([w.wcs.crval[1] for _, w in header_list])

        ref_wcs = WCS(naxis=2)
        ref_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        ref_wcs.wcs.crval = [center_ra, center_dec]
        ref_wcs.wcs.crpix = [0.0, 0.0]
        ref_wcs.wcs.cd = np.eye(2)
    else:
        ref_wcs = header_list[0][1]

    xmin = math.inf
    ymin = math.inf
    xmax = -math.inf
    ymax = -math.inf
    pixel_scales = []
    all_xy = []

    for shape_hw, wcs in header_list:
        h, w = shape_hw
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=float)
        sky = wcs.pixel_to_world(corners[:, 0], corners[:, 1])
        x, y = ref_wcs.world_to_pixel(sky)
        all_xy.append(np.column_stack([x, y]))
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

    if auto_rotate:
        xy = np.vstack(all_xy)
        eigvals, eigvecs = np.linalg.eig(np.cov(xy, rowvar=False))
        theta = math.atan2(eigvecs[1, 0], eigvecs[0, 0])
        rot = np.array(
            [[math.cos(-theta), -math.sin(-theta)], [math.sin(-theta), math.cos(-theta)]]
        )
        xy_rot = xy @ rot.T
        xmin = float(np.min(xy_rot[:, 0]))
        xmax = float(np.max(xy_rot[:, 0]))
        ymin = float(np.min(xy_rot[:, 1]))
        ymax = float(np.max(xy_rot[:, 1]))
    else:
        theta = 0.0

    xmin_f = math.floor(xmin) - 1
    ymin_f = math.floor(ymin) - 1
    xmax_f = math.ceil(xmax) + 1
    ymax_f = math.ceil(ymax) + 1

    out_w = int(xmax_f - xmin_f)
    out_h = int(ymax_f - ymin_f)

    out_wcs = WCS(naxis=2)
    out_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    out_wcs.wcs.crval = list(ref_wcs.wcs.crval)
    out_wcs.wcs.crpix = [-xmin_f + 1.0, -ymin_f + 1.0]
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    out_wcs.wcs.cd = output_scale_deg * np.array([[-cos_t, sin_t], [sin_t, cos_t]])
    out_wcs.pixel_shape = (out_w, out_h)
    try:
        out_wcs._naxis1 = out_w
        out_wcs._naxis2 = out_h
    except Exception:
        pass

    return out_wcs, (out_h, out_w)

