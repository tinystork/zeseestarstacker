"""Geometry-aware automatic reference selection (GAR-04 M2).

Wires the fast-TAN geometry (module seestar.core.fast_geometry) to the historical
reference quality metric to select, at most once per run, a real central frame.
The caller (queue manager) pins the resulting absolute path onto the aligner so
RF2 keeps a single immutable reference.  Anything uncertain fails closed to the
historical auto-selection fallback.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from astropy.io import fits

from .fast_geometry import (
    CENTRAL_CANDIDATES,
    FootprintGateResult,
    angular_separation_deg,
    evaluate_footprint_gate,
    fast_geometry,
)

ORIGIN_RESUME = "RESUME"
ORIGIN_USER = "USER"
ORIGIN_ZEANALYSER = "ZEANALYSER_V1"
ORIGIN_AUTO_GEOMETRY = "AUTO_GEOMETRY"
ORIGIN_AUTO_LEGACY = "AUTO_LEGACY"

_FITS_EXTS = (".fit", ".fits")
_REF_PREFIXES_TO_SKIP = ("stack_", "mosaic_final_", "aligned_", "drizzle_")
_REF_SUBSTRINGS_TO_SKIP = (
    "_reproject", "_sum.", "_wht.", "_preview.", "_temp.",
    "reference_image.fit", "cumulative_sum.npy", "cumulative_wht.npy",
)


@dataclass(frozen=True)
class ResolvedReference:
    """A single reference decision: an absolute path and its provenance."""
    path: Optional[str]
    origin: Optional[str]


@dataclass(frozen=True)
class GeometrySelection:
    """Result of one geometry-aware selection attempt."""
    resolved: ResolvedReference
    gate: Optional[FootprintGateResult] = None
    source_count: int = 0
    geometry_count: int = 0
    candidate_count: int = 0
    selected_metric: Optional[float] = None
    selected_distance_diagonals: Optional[float] = None


def ref_name_filtered(basename: str) -> bool:
    """True when the basename is a generated artifact that must be skipped."""
    lower = basename.lower()
    for prefix in _REF_PREFIXES_TO_SKIP:
        if lower.startswith(prefix):
            return True
    for sub in _REF_SUBSTRINGS_TO_SKIP:
        if sub in lower:
            return True
    return False


def files_from_stack_plan(plan_path: str, input_folder: Optional[str] = None) -> list:
    """Ordered flat file list from a stack_plan.csv (batch_id, file_path)."""
    by_batch = {}
    order = []
    with open(plan_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            bid = row.get("batch_id")
            fp = row.get("file_path")
            if not fp or bid is None:
                continue
            if input_folder and not os.path.isfile(fp):
                fp = os.path.join(input_folder, os.path.basename(fp))
            if bid not in by_batch:
                order.append(bid)
            by_batch.setdefault(bid, []).append(fp)
    flat = []
    seen = set()
    for bid in order:
        for fp in by_batch.get(bid, []):
            real = os.path.realpath(fp)
            if real in seen:
                continue
            seen.add(real)
            flat.append(real)
    return flat


def canonical_session_sources(
    current_folder: str,
    initial_additional_folders: Optional[Iterable[str]] = None,
    stack_plan_path: Optional[str] = None,
    apply_name_filter: bool = False,
) -> list:
    """Deterministic active source list for one initial session.

    A supplied authoritative stack plan wins (order preserved, files outside it
    excluded); otherwise the current folder and all initial additional folders
    are aggregated, de-duplicated by realpath, and sorted by basename then path.
    """
    if stack_plan_path and os.path.isfile(stack_plan_path):
        candidates = files_from_stack_plan(stack_plan_path, os.path.realpath(current_folder))
        ordered = True
    else:
        roots = [current_folder]
        roots.extend(initial_additional_folders or [])
        candidates = []
        for root in roots:
            if not root or not os.path.isdir(root):
                continue
            for name in sorted(os.listdir(root)):
                if name.lower().endswith(_FITS_EXTS):
                    candidates.append(os.path.join(root, name))
        ordered = False

    seen = set()
    result = []
    for candidate in candidates:
        real = os.path.realpath(candidate)
        basename = os.path.basename(real)
        if real in seen or not os.path.isfile(real):
            continue
        if not basename.lower().endswith(_FITS_EXTS):
            continue
        if apply_name_filter and ref_name_filtered(basename):
            continue
        seen.add(real)
        result.append(real)
    if ordered:
        return result
    return sorted(result, key=lambda p: (os.path.basename(p).casefold(), p))


def reference_quality_metric(
    path: str,
    bayer_pattern: str = "GRBG",
    correct_hot_pixels: bool = True,
    hot_pixel_threshold: float = 3.0,
    neighborhood_size: int = 5,
) -> Optional[float]:
    """Reproduce the historical auto-reference quality metric exactly.

    Returns the metric median / (1.4826 * MAD) over the production
    load/normalise/debayer/hot-pixel pipeline, or None when the candidate is
    rejected.  Same metric and preprocessing as the legacy auto-selection in
    seestar.core.alignment.SeestarAligner.
    """
    from .image_processing import debayer_image, load_and_validate_fits
    from .hot_pixels import detect_and_correct_hot_pixels

    try:
        loaded = load_and_validate_fits(path)
        if loaded is None or loaded[0] is None:
            return None
        image, header = loaded
        if header is None:
            header = fits.Header()
        if float(np.std(image)) < 0.0005:
            return None
        prepared = image.astype(np.float32, copy=True)
        if prepared.ndim == 2:
            bayer = header.get("BAYERPAT", bayer_pattern)
            if isinstance(bayer, str) and bayer.upper() in ("GRBG", "RGGB", "GBRG", "BGGR"):
                try:
                    prepared = debayer_image(prepared, bayer.upper())
                except ValueError:
                    pass
        if correct_hot_pixels:
            try:
                prepared = detect_and_correct_hot_pixels(
                    prepared, hot_pixel_threshold, neighborhood_size
                )
            except Exception:
                pass
        median = float(np.median(prepared))
        mad = float(np.median(np.abs(prepared - median)))
        approx_std = mad * 1.4826
        if median <= 1e-9 or approx_std <= 1e-9:
            return None
        metric = median / (approx_std + 1e-9)
        if not np.isfinite(metric) or metric < -1e8:
            return None
        return float(metric)
    except Exception:
        return None


def select_geometry_reference(
    sources: Sequence[str],
    quality_fn: Optional[Callable[[str], Optional[float]]] = None,
    bayer_pattern: str = "GRBG",
    correct_hot_pixels: bool = True,
    hot_pixel_threshold: float = 3.0,
    neighborhood_size: int = 5,
    stop_requested: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> GeometrySelection:
    """Select the best central candidate, or fall back to legacy auto."""
    if quality_fn is None:
        quality_fn = lambda p: reference_quality_metric(
            p, bayer_pattern, correct_hot_pixels, hot_pixel_threshold, neighborhood_size
        )

    geometries = []
    frames = []
    total = len(sources)
    for index, path in enumerate(sources):
        if stop_requested is not None and stop_requested():
            return GeometrySelection(ResolvedReference(None, ORIGIN_AUTO_LEGACY))
        if progress is not None:
            progress("Scanning dataset geometry... %d / %d" % (index + 1, total))
        try:
            header = fits.getheader(path)
        except Exception:
            continue
        geometry = fast_geometry(header)
        if geometry is None:
            continue
        geometries.append(geometry)
        frames.append((path, geometry))

    gate = evaluate_footprint_gate(geometries, dataset_size=total)
    if not gate.accepted:
        return GeometrySelection(
            ResolvedReference(None, ORIGIN_AUTO_LEGACY),
            gate=gate, source_count=total, geometry_count=len(geometries),
        )

    center = (gate.center_ra_deg, gate.center_dec_deg)
    seps = []
    for path, geometry in frames:
        sep = angular_separation_deg((geometry.center_ra_deg, geometry.center_dec_deg), center)
        seps.append((sep, path, geometry))
    seps.sort(key=lambda item: item[0])
    pool = seps[: min(CENTRAL_CANDIDATES, len(seps))]

    best_path = None
    best_metric = -math.inf
    best_sep = None
    for index, (sep, path, _geometry) in enumerate(pool):
        if stop_requested is not None and stop_requested():
            return GeometrySelection(ResolvedReference(None, ORIGIN_AUTO_LEGACY))
        if progress is not None:
            progress("Evaluating central candidates... %d / %d" % (index + 1, len(pool)))
        metric = quality_fn(path)
        if metric is None:
            continue
        if metric > best_metric:
            best_metric = metric
            best_path = path
            best_sep = sep

    if best_path is None:
        return GeometrySelection(
            ResolvedReference(None, ORIGIN_AUTO_LEGACY),
            gate=gate, source_count=total, geometry_count=len(geometries),
            candidate_count=len(pool),
        )

    distance_diagonals = None
    if gate.representative_diagonal_deg:
        distance_diagonals = best_sep / gate.representative_diagonal_deg
    return GeometrySelection(
        ResolvedReference(os.path.realpath(best_path), ORIGIN_AUTO_GEOMETRY),
        gate=gate, source_count=total, geometry_count=len(geometries),
        candidate_count=len(pool), selected_metric=best_metric,
        selected_distance_diagonals=distance_diagonals,
    )


def resolve_reference_precedence(
    resume_path: Optional[str] = None,
    user_path: Optional[str] = None,
    zeanalyser_path: Optional[str] = None,
    geometry_path: Optional[str] = None,
    legacy_path: Optional[str] = None,
) -> ResolvedReference:
    """Resolve the reference precedence without introducing a new protocol."""
    for origin, candidate in (
        (ORIGIN_RESUME, resume_path),
        (ORIGIN_USER, user_path),
        (ORIGIN_ZEANALYSER, zeanalyser_path),
    ):
        if candidate and os.path.isfile(candidate):
            return ResolvedReference(os.path.realpath(candidate), origin)
    if geometry_path:
        return ResolvedReference(os.path.realpath(geometry_path), ORIGIN_AUTO_GEOMETRY)
    if legacy_path:
        return ResolvedReference(os.path.realpath(legacy_path), ORIGIN_AUTO_LEGACY)
    return ResolvedReference(None, None)

