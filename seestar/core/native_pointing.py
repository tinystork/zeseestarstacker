"""Strict, header-only native telescope pointing extraction and coherence.

Native pointing is deliberately lower-level than solved WCS.  It describes
where the telescope was aimed well enough to rank frames inside one dataset;
it is never used for image registration or sub-pixel geometry.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

PROVENANCE_NATIVE_POINTING = "NATIVE_POINTING"

# Native metadata must represent essentially the whole run.  This is a
# separate policy from GAR-04's 80% WCS gate: solved WCS may be sparse, while a
# native-pointing population is selected only when it is homogeneous.
MIN_NATIVE_POINTING_FRACTION = 0.95
MIN_NATIVE_POINTING_FRAMES = 3
MIN_NATIVE_COHERENT_FRACTION = 0.95

# A conservative target-coherence guardrail, not a Seestar field-size model.
# It admits normal drift and multi-session recentering while rejecting widely
# separated targets (for example M16 mixed with M31).  Outliers outside this
# radius are excluded from the native candidate population.
MAX_NATIVE_CENTRE_OFFSET_DEG = 5.0

_KEY_PAIRS = (
    ("RA", "DEC"),
    ("OBJCTRA", "OBJCTDEC"),
    ("TELRA", "TELDEC"),
    ("OBJRA", "OBJDEC"),
    ("RA2000", "DEC2000"),
)


@dataclass(frozen=True)
class NativePointing:
    """One validated native telescope pointing in decimal degrees."""

    ra_deg: float
    dec_deg: float
    provenance: str = PROVENANCE_NATIVE_POINTING
    ra_key: Optional[str] = None
    dec_key: Optional[str] = None


@dataclass(frozen=True)
class NativePointingParseResult:
    """Explicit parser result; ``pointing`` is absent on invalid metadata."""

    pointing: Optional[NativePointing]
    reason: Optional[str] = None


@dataclass(frozen=True)
class NativePointingGateResult:
    """Dataset-level native-pointing availability and coherence decision."""

    accepted: bool
    reason: Optional[str]
    n_frames: int
    n_pointing: int
    pointing_fraction: float
    center_ra_deg: Optional[float] = None
    center_dec_deg: Optional[float] = None
    coherent_fraction: Optional[float] = None
    median_offset_deg: Optional[float] = None
    p95_offset_deg: Optional[float] = None
    max_offset_deg: Optional[float] = None
    inlier_mask: Optional[tuple] = None


def _decimal_degrees(value, coordinate: str) -> float:
    """Parse only unambiguous decimal degrees.

    Accepted forms are finite real numbers or decimal strings, optionally
    suffixed by ``deg``, ``degree(s)``, or ``d``.  Colon/space sexagesimal and
    hour-angle strings are intentionally rejected because FITS headers do not
    carry a universal unit contract for these non-standard keywords.
    """

    if isinstance(value, bool):
        raise ValueError("boolean coordinate")
    if isinstance(value, numbers.Real):
        result = float(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        for suffix in ("degrees", "degree", "deg", "d"):
            if text.endswith(suffix):
                text = text[: -len(suffix)].strip()
                break
        if not text or any(mark in text for mark in (":", "h", "m", "s", " ")):
            raise ValueError("ambiguous or unsupported coordinate string")
        result = float(text)
    else:
        raise ValueError("unsupported coordinate type")
    if not math.isfinite(result):
        raise ValueError("non-finite coordinate")
    if coordinate == "ra":
        if result < 0.0 or result > 360.0:
            raise ValueError("RA outside [0, 360] degrees")
        return result % 360.0
    if result < -90.0 or result > 90.0:
        raise ValueError("DEC outside [-90, 90] degrees")
    return result


def parse_native_pointing_result(header: Mapping) -> NativePointingParseResult:
    """Extract the first complete, valid documented RA/DEC key pair.

    Keys are paired by family; values from unrelated keyword families are
    never combined.  A present but invalid higher-priority pair does not block
    a later complete valid pair, and the final unavailable reason is explicit.
    """

    invalid = []
    found_any = False
    for ra_key, dec_key in _KEY_PAIRS:
        has_ra = ra_key in header
        has_dec = dec_key in header
        if not has_ra and not has_dec:
            continue
        found_any = True
        if not (has_ra and has_dec):
            invalid.append(f"incomplete {ra_key}/{dec_key} pair")
            continue
        try:
            ra_deg = _decimal_degrees(header[ra_key], "ra")
            dec_deg = _decimal_degrees(header[dec_key], "dec")
        except (TypeError, ValueError, OverflowError) as exc:
            invalid.append(f"invalid {ra_key}/{dec_key}: {exc}")
            continue
        return NativePointingParseResult(
            NativePointing(ra_deg, dec_deg, ra_key=ra_key, dec_key=dec_key)
        )
    if invalid:
        return NativePointingParseResult(None, "; ".join(invalid))
    if found_any:
        return NativePointingParseResult(None, "native pointing unavailable")
    return NativePointingParseResult(None, "no native pointing keywords")


def parse_native_pointing(header: Mapping) -> Optional[NativePointing]:
    """Compatibility-friendly convenience wrapper returning a point or None."""

    return parse_native_pointing_result(header).pointing


def unit_vector(pointing: NativePointing) -> np.ndarray:
    ra = math.radians(pointing.ra_deg)
    dec = math.radians(pointing.dec_deg)
    return np.array(
        (math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)),
        dtype=np.float64,
    )


def robust_spherical_center(pointings: Sequence[NativePointing]) -> tuple[float, float]:
    """Return a robust spherical centre using a normalized geometric median."""

    if not pointings:
        raise ValueError("no pointings")
    vectors = np.stack([unit_vector(pointing) for pointing in pointings])
    current = np.median(vectors, axis=0)
    norm = float(np.linalg.norm(current))
    if norm < 1e-12:
        current = vectors.mean(axis=0)
        norm = float(np.linalg.norm(current))
    if norm < 1e-12:
        raise ValueError("degenerate spherical centre")
    current /= norm
    for _ in range(32):
        distances = np.linalg.norm(vectors - current, axis=1)
        if float(distances.min()) < 1e-12:
            candidate = vectors[int(distances.argmin())]
        else:
            weights = 1.0 / np.maximum(distances, 1e-12)
            candidate = (vectors * weights[:, None]).sum(axis=0) / float(weights.sum())
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm < 1e-12:
            raise ValueError("degenerate spherical centre")
        candidate /= candidate_norm
        if float(np.linalg.norm(candidate - current)) < 1e-12:
            current = candidate
            break
        current = candidate
    ra_deg = float(math.degrees(math.atan2(current[1], current[0])) % 360.0)
    dec_deg = float(math.degrees(math.asin(float(np.clip(current[2], -1.0, 1.0)))))
    return ra_deg, dec_deg


def angular_offsets_deg(pointings: Sequence[NativePointing], center: tuple[float, float]) -> np.ndarray:
    """Vectorized great-circle offsets from ``center`` in degrees."""

    if not pointings:
        return np.empty(0, dtype=np.float64)
    vectors = np.stack([unit_vector(pointing) for pointing in pointings])
    ra = math.radians(center[0])
    dec = math.radians(center[1])
    centre_vector = np.array(
        (math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)),
        dtype=np.float64,
    )
    dots = np.clip(vectors @ centre_vector, -1.0, 1.0)
    crosses = np.linalg.norm(np.cross(vectors, centre_vector), axis=1)
    return np.degrees(np.arctan2(crosses, dots))


def evaluate_native_pointing_gate(
    pointings: Sequence[NativePointing],
    dataset_size: Optional[int] = None,
    min_pointing_fraction: float = MIN_NATIVE_POINTING_FRACTION,
    min_coherent_fraction: float = MIN_NATIVE_COHERENT_FRACTION,
    max_center_offset_deg: float = MAX_NATIVE_CENTRE_OFFSET_DEG,
    min_frames: int = MIN_NATIVE_POINTING_FRAMES,
) -> NativePointingGateResult:
    """Accept a homogeneous native population and identify its inliers."""

    n = len(pointings)
    total = n if dataset_size is None else int(dataset_size)
    if total < n or total < 0:
        raise ValueError("dataset_size must be >= pointing count")
    fraction = n / total if total else 0.0
    common = dict(n_frames=total, n_pointing=n, pointing_fraction=fraction)
    if fraction < min_pointing_fraction:
        return NativePointingGateResult(
            False,
            "insufficient native pointing: %d/%d (%.1f%% < %.0f%%)"
            % (n, total, fraction * 100.0, min_pointing_fraction * 100.0),
            **common,
        )
    if n < min_frames:
        return NativePointingGateResult(
            False, "insufficient native-pointing frames: %d < %d" % (n, min_frames), **common
        )
    try:
        center = robust_spherical_center(pointings)
    except ValueError as exc:
        return NativePointingGateResult(False, str(exc), **common)
    offsets = angular_offsets_deg(pointings, center)
    inliers = offsets <= max_center_offset_deg
    coherent_fraction = float(inliers.mean())
    diagnostics = dict(
        center_ra_deg=center[0],
        center_dec_deg=center[1],
        coherent_fraction=coherent_fraction,
        median_offset_deg=float(np.median(offsets)),
        p95_offset_deg=float(np.percentile(offsets, 95)),
        max_offset_deg=float(offsets.max()),
        inlier_mask=tuple(bool(value) for value in inliers),
    )
    if coherent_fraction < min_coherent_fraction:
        return NativePointingGateResult(
            False,
            "incoherent native pointing: %.1f%% within %.1f deg (< %.0f%%)"
            % (coherent_fraction * 100.0, max_center_offset_deg, min_coherent_fraction * 100.0),
            **common,
            **diagnostics,
        )
    return NativePointingGateResult(True, None, **common, **diagnostics)
