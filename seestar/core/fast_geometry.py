"""Fast, strict inverse-TAN geometry + footprint coherence gate (GAR-04).

This is deliberately *not* a general WCS engine.  It supports only the exact
2-D celestial TAN subset observed in the ZeSeestarStacker witnesses: RA/DEC
axis order, degree units, and either a complete ``CD`` matrix or ``PC`` +
``CDELT``.  Anything else fails closed (``fast_geometry`` returns ``None``).

Pure ``math`` + ``numpy``; no ``astropy`` import at runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

MIN_GEOMETRY_FRACTION = 0.8
MIN_COHERENT_FRACTION = 0.8
MAX_CENTER_OFFSET_DIAGONALS = 1.5
MIN_FRAMES = 3
CENTRAL_CANDIDATES = 32


@dataclass(frozen=True)
class FastGeometry:
    """Centre and footprint diagonal extracted from one FITS header."""
    center_ra_deg: float
    center_dec_deg: float
    footprint_diag_deg: float
    representation: str


@dataclass
class FootprintGateResult:
    """Outcome of the multi-frame footprint coherence gate."""
    accepted: bool
    reason: Optional[str]
    n_frames: int
    n_geometry: int
    geometry_fraction: float
    center_ra_deg: Optional[float] = None
    center_dec_deg: Optional[float] = None
    representative_diagonal_deg: Optional[float] = None
    coherent_fraction: Optional[float] = None


_DISTORTION_PREFIXES = ("A_", "B_", "AP_", "BP_", "PV1_", "PV2_", "CPDIS", "DET2IM")
_EXTRA_AXIS_KEYS = ("CTYPE3", "CRVAL3", "CRPIX3", "CDELT3", "CD3_3", "PC3_3", "NAXIS3")


def _finite_number(header: Mapping, key: str) -> float:
    value = float(header[key])
    if not math.isfinite(value):
        raise ValueError("non-finite " + key)
    return value


def _linear_matrix_deg(header: Mapping):
    cd_keys = ("CD1_1", "CD1_2", "CD2_1", "CD2_2")
    if any(key in header for key in cd_keys):
        if not all(key in header for key in cd_keys):
            raise ValueError("incomplete CD matrix")
        matrix = (
            _finite_number(header, "CD1_1"),
            _finite_number(header, "CD1_2"),
            _finite_number(header, "CD2_1"),
            _finite_number(header, "CD2_2"),
        )
        representation = "CD"
    else:
        if "CDELT1" not in header or "CDELT2" not in header:
            raise ValueError("missing CD and PC+CDELT linear transform")
        cdelt1 = _finite_number(header, "CDELT1")
        cdelt2 = _finite_number(header, "CDELT2")
        pc = (
            float(header.get("PC1_1", 1.0)),
            float(header.get("PC1_2", 0.0)),
            float(header.get("PC2_1", 0.0)),
            float(header.get("PC2_2", 1.0)),
        )
        if not all(math.isfinite(value) for value in pc):
            raise ValueError("non-finite PC matrix")
        matrix = (cdelt1 * pc[0], cdelt1 * pc[1], cdelt2 * pc[2], cdelt2 * pc[3])
        representation = "PC+CDELT"
    if abs(matrix[0] * matrix[3] - matrix[1] * matrix[2]) < 1e-20:
        raise ValueError("degenerate linear transform")
    return matrix, representation


def _tan_plane_coordinates(x, y, crpix1, crpix2, matrix_deg):
    dx = x + 1.0 - crpix1
    dy = y + 1.0 - crpix2
    xi = math.radians(matrix_deg[0] * dx + matrix_deg[1] * dy)
    eta = math.radians(matrix_deg[2] * dx + matrix_deg[3] * dy)
    return xi, eta


def _tan_pixel_to_sky(x, y, crpix1, crpix2, crval1_deg, crval2_deg, matrix_deg):
    xi, eta = _tan_plane_coordinates(x, y, crpix1, crpix2, matrix_deg)
    ra0 = math.radians(crval1_deg)
    dec0 = math.radians(crval2_deg)
    sin_dec0, cos_dec0 = math.sin(dec0), math.cos(dec0)
    denominator = cos_dec0 - eta * sin_dec0
    ra = ra0 + math.atan2(xi, denominator)
    dec = math.atan2(sin_dec0 + eta * cos_dec0, math.hypot(denominator, xi))
    return float(math.degrees(ra) % 360.0), float(math.degrees(dec))


def _tan_plane_separation_deg(first, second):
    xi1, eta1 = first
    xi2, eta2 = second
    cross_x = eta1 - eta2
    cross_y = xi2 - xi1
    cross_z = xi1 * eta2 - eta1 * xi2
    cross_norm = math.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)
    dot = xi1 * xi2 + eta1 * eta2 + 1.0
    return math.degrees(math.atan2(cross_norm, dot))


def unit_vector(ra_deg, dec_deg):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.array(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)],
        dtype=np.float64,
    )


def angular_separation_deg(a, b):
    ra1, dec1 = map(math.radians, a)
    ra2, dec2 = map(math.radians, b)
    hav = math.sin((dec2 - dec1) / 2.0) ** 2
    hav += math.cos(dec1) * math.cos(dec2) * math.sin((ra2 - ra1) / 2.0) ** 2
    return math.degrees(2.0 * math.asin(math.sqrt(min(1.0, max(0.0, hav)))))


def fast_geometry(header: Mapping):
    try:
        ctype1 = str(header.get("CTYPE1", "")).strip().upper()
        ctype2 = str(header.get("CTYPE2", "")).strip().upper()
        if (ctype1, ctype2) != ("RA---TAN", "DEC--TAN"):
            return None
        cunit1 = str(header.get("CUNIT1", "deg")).strip().lower()
        cunit2 = str(header.get("CUNIT2", "deg")).strip().lower()
        if cunit1 not in ("deg", "degree", "degrees") or cunit2 not in ("deg", "degree", "degrees"):
            return None
        if int(header.get("WCSAXES", 2)) != 2:
            return None
        if any(key in header for key in _EXTRA_AXIS_KEYS):
            return None
        if any(str(key).upper().startswith(_DISTORTION_PREFIXES) for key in header.keys()):
            return None
        if "LONPOLE" in header and abs(float(header["LONPOLE"]) - 180.0) > 1e-10:
            return None
        width = _finite_number(header, "NAXIS1")
        height = _finite_number(header, "NAXIS2")
        if width <= 1 or height <= 1:
            return None
        crpix1 = _finite_number(header, "CRPIX1")
        crpix2 = _finite_number(header, "CRPIX2")
        crval1 = _finite_number(header, "CRVAL1")
        crval2 = _finite_number(header, "CRVAL2")
        if not -90.0 <= crval2 <= 90.0:
            return None
        matrix, representation = _linear_matrix_deg(header)
        center = _tan_pixel_to_sky(
            (width - 1.0) / 2.0, (height - 1.0) / 2.0,
            crpix1, crpix2, crval1, crval2, matrix,
        )
        corners = (
            _tan_plane_coordinates(0.0, 0.0, crpix1, crpix2, matrix),
            _tan_plane_coordinates(width - 1.0, 0.0, crpix1, crpix2, matrix),
            _tan_plane_coordinates(width - 1.0, height - 1.0, crpix1, crpix2, matrix),
            _tan_plane_coordinates(0.0, height - 1.0, crpix1, crpix2, matrix),
        )
        diagonal = max(
            _tan_plane_separation_deg(corners[0], corners[2]),
            _tan_plane_separation_deg(corners[1], corners[3]),
        )
        if not math.isfinite(diagonal) or diagonal <= 0.0:
            return None
        return FastGeometry(center[0], center[1], diagonal, representation)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def evaluate_footprint_gate(geometries, dataset_size=None,
                            min_wcs_fraction=MIN_GEOMETRY_FRACTION,
                            min_coherent_fraction=MIN_COHERENT_FRACTION,
                            max_center_offset_diagonals=MAX_CENTER_OFFSET_DIAGONALS,
                            min_frames=MIN_FRAMES):
    n = len(geometries)
    total = n if dataset_size is None else int(dataset_size)
    if total < n or total < 0:
        raise ValueError("dataset_size must be >= geometry count")
    fraction = n / total if total else 0.0
    common = dict(n_frames=total, n_geometry=n, geometry_fraction=fraction)
    if fraction < min_wcs_fraction:
        return FootprintGateResult(
            False,
            "insufficient fast geometry: %d/%d (%.1f%% < %.0f%%)" % (n, total, fraction * 100, min_wcs_fraction * 100),
            **common,
        )
    if n < min_frames:
        return FootprintGateResult(
            False, "insufficient fast-geometry frames: %d < %d" % (n, min_frames), **common
        )
    vectors = np.stack([unit_vector(g.center_ra_deg, g.center_dec_deg) for g in geometries])
    center = vectors.mean(axis=0)
    norm = float(np.linalg.norm(center))
    if norm < 1e-12:
        return FootprintGateResult(False, "degenerate spherical centre", **common)
    center /= norm
    dots = np.clip(vectors @ center, -1.0, 1.0)
    cross_norm = np.linalg.norm(np.cross(vectors, center), axis=1)
    separations = np.degrees(np.arctan2(cross_norm, dots))
    diagonals = np.array([g.footprint_diag_deg for g in geometries], dtype=np.float64)
    if not np.all(np.isfinite(diagonals)) or np.any(diagonals <= 0.0):
        return FootprintGateResult(False, "invalid frame footprint", **common)
    normalized = separations / diagonals
    within = normalized <= max_center_offset_diagonals
    coherent_fraction = float(within.mean())
    center_ra = float(np.degrees(np.arctan2(center[1], center[0])) % 360.0)
    center_dec = float(np.degrees(np.arcsin(np.clip(center[2], -1.0, 1.0))))
    representative = float(np.median(diagonals))
    if coherent_fraction < min_coherent_fraction:
        return FootprintGateResult(
            False,
            "incoherent/mixed targets: %d/%d within %.2f frame diagonals (%.1f%% < %.0f%%)" % (
                int(within.sum()), n, max_center_offset_diagonals,
                coherent_fraction * 100, min_coherent_fraction * 100,
            ),
            center_ra_deg=center_ra, center_dec_deg=center_dec,
            representative_diagonal_deg=representative,
            coherent_fraction=coherent_fraction, **common,
        )
    return FootprintGateResult(
        True, None,
        center_ra_deg=center_ra, center_dec_deg=center_dec,
        representative_diagonal_deg=representative,
        coherent_fraction=coherent_fraction, **common,
    )

