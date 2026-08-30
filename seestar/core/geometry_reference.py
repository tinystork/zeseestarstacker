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
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
from astropy.io import fits

from .fast_geometry import (
    CENTRAL_CANDIDATES,
    angular_separation_deg,
    evaluate_footprint_gate,
    fast_geometry,
)
from .native_pointing import (
    angular_offsets_deg,
    evaluate_native_pointing_gate,
    parse_native_pointing,
)

ORIGIN_RESUME = "RESUME"
ORIGIN_USER = "USER"
ORIGIN_ZEANALYSER = "ZEANALYSER_V1"
ORIGIN_AUTO_GEOMETRY = "AUTO_GEOMETRY"
ORIGIN_AUTO_LEGACY = "AUTO_LEGACY"

GEOMETRY_SOURCE_NATIVE = "native_pointing"
GEOMETRY_SOURCE_WCS = "wcs"
GEOMETRY_SOURCE_UNAVAILABLE = "unavailable"

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
    gate: Optional[Any] = None
    source_count: int = 0
    geometry_count: int = 0
    pointing_count: int = 0
    pointing_fraction: float = 0.0
    candidate_count: int = 0
    selected_metric: Optional[float] = None
    selected_distance_diagonals: Optional[float] = None
    selected_offset_deg: Optional[float] = None
    geometry_source: str = GEOMETRY_SOURCE_UNAVAILABLE
    center_ra_deg: Optional[float] = None
    center_dec_deg: Optional[float] = None
    fallback_reason: Optional[str] = None
    scan_elapsed_s: float = 0.0
    scan_files_per_s: float = 0.0
    progress_event_count: int = 0


@dataclass(frozen=True)
class LegacySelection:
    """One historical first-20 quality decision, without materialization."""

    resolved: ResolvedReference
    candidate_count: int = 0
    selected_metric: Optional[float] = None
    reason: Optional[str] = None


class _CoarseProgress:
    """Bounded progress adapter supporting modern and legacy callbacks."""

    def __init__(self, callback):
        self.callback = callback
        self.count = 0
        self._last_scan_bucket = None

    def emit(self, message: str, percent: Optional[int] = None, level: str = "INFO") -> None:
        if self.callback is None:
            return
        self.count += 1
        try:
            self.callback(message, percent, level=level)
        except TypeError:
            try:
                self.callback(message, percent)
            except TypeError:
                self.callback(message)

    def scan(self, completed: int, total: int) -> None:
        percent = 100 if total <= 0 else int((completed * 100) // total)
        bucket = min(10, percent // 10)
        if self._last_scan_bucket == bucket:
            return
        self._last_scan_bucket = bucket
        shown = bucket * 10
        self.emit(f"Scanning dataset pointing... {shown}%", shown)


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


def legacy_session_sources(
    current_folder: str,
    initial_additional_folders: Optional[Iterable[str]] = None,
) -> list:
    """Return the historical first folder's sorted FITS population.

    This intentionally does not aggregate roots: AUTO_LEGACY retains its
    pre-GAR semantics while becoming an explicit, once-per-run decision.
    """

    roots = [current_folder]
    roots.extend(initial_additional_folders or [])
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        paths = [
            os.path.realpath(os.path.join(root, name))
            for name in sorted(os.listdir(root))
            if name.lower().endswith(_FITS_EXTS)
        ]
        if paths:
            return paths
    return []


def select_legacy_reference(
    sources: Sequence[str],
    quality_fn: Optional[Callable[[str], Optional[float]]] = None,
    bayer_pattern: str = "GRBG",
    correct_hot_pixels: bool = True,
    hot_pixel_threshold: float = 3.0,
    neighborhood_size: int = 5,
    stop_requested: Optional[Callable[[], bool]] = None,
) -> LegacySelection:
    """Resolve historical first-20 quality selection exactly once."""

    if quality_fn is None:
        quality_fn = lambda path: reference_quality_metric(
            path, bayer_pattern, correct_hot_pixels, hot_pixel_threshold, neighborhood_size
        )
    subset = list(sources[:20])
    candidates = [path for path in subset if not ref_name_filtered(os.path.basename(path))]
    if not candidates:
        return LegacySelection(
            ResolvedReference(None, ORIGIN_AUTO_LEGACY),
            reason="no valid legacy candidate among first 20 inputs",
        )
    best_path = None
    best_metric = -math.inf
    for path in candidates:
        if stop_requested is not None and stop_requested():
            return LegacySelection(
                ResolvedReference(None, ORIGIN_AUTO_LEGACY),
                candidate_count=len(candidates),
                reason="reference selection cancelled",
            )
        metric = quality_fn(path)
        if metric is not None and metric > best_metric:
            best_path = path
            best_metric = metric
    if best_path is None:
        return LegacySelection(
            ResolvedReference(None, ORIGIN_AUTO_LEGACY),
            candidate_count=len(candidates),
            reason="no legacy candidate passed the historical quality metric",
        )
    return LegacySelection(
        ResolvedReference(os.path.realpath(best_path), ORIGIN_AUTO_LEGACY),
        candidate_count=len(candidates),
        selected_metric=float(best_metric),
    )


def select_geometry_reference(
    sources: Sequence[str],
    quality_fn: Optional[Callable[[str], Optional[float]]] = None,
    bayer_pattern: str = "GRBG",
    correct_hot_pixels: bool = True,
    hot_pixel_threshold: float = 3.0,
    neighborhood_size: int = 5,
    stop_requested: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[..., None]] = None,
) -> GeometrySelection:
    """Select one coherent geometry source, then the best central frame.

    Native pointing has run-level priority.  If its dataset gate rejects, the
    existing strict WCS/TAN population is evaluated intact.  Populations are
    never mixed image by image.
    """
    if quality_fn is None:
        quality_fn = lambda p: reference_quality_metric(
            p, bayer_pattern, correct_hot_pixels, hot_pixel_threshold, neighborhood_size
        )

    reporter = _CoarseProgress(progress)
    scan_started = time.monotonic()
    geometries = []
    wcs_frames = []
    pointings = []
    pointing_frames = []
    total = len(sources)
    reporter.scan(0, total)
    for index, path in enumerate(sources):
        if stop_requested is not None and stop_requested():
            elapsed = time.monotonic() - scan_started
            return GeometrySelection(
                ResolvedReference(None, ORIGIN_AUTO_LEGACY),
                source_count=total,
                fallback_reason="reference selection cancelled",
                scan_elapsed_s=elapsed,
                scan_files_per_s=(index / elapsed if elapsed > 0.0 else 0.0),
                progress_event_count=reporter.count,
            )
        try:
            header = fits.getheader(path)
        except Exception:
            reporter.scan(index + 1, total)
            continue
        pointing = parse_native_pointing(header)
        if pointing is not None:
            pointings.append(pointing)
            pointing_frames.append((path, pointing))
        geometry = fast_geometry(header)
        if geometry is not None:
            geometries.append(geometry)
            wcs_frames.append((path, geometry))
        reporter.scan(index + 1, total)

    scan_elapsed = time.monotonic() - scan_started
    scan_rate = total / scan_elapsed if scan_elapsed > 0.0 else 0.0
    native_gate = evaluate_native_pointing_gate(pointings, dataset_size=total)
    wcs_gate = None
    gate = native_gate
    source = GEOMETRY_SOURCE_UNAVAILABLE
    center = None
    frames = []
    native_reason = native_gate.reason or "native pointing unavailable"
    if native_gate.accepted:
        source = GEOMETRY_SOURCE_NATIVE
        center = (native_gate.center_ra_deg, native_gate.center_dec_deg)
        inlier_mask = native_gate.inlier_mask or tuple(True for _ in pointing_frames)
        frames = [
            (path, pointing)
            for (path, pointing), is_inlier in zip(pointing_frames, inlier_mask)
            if is_inlier
        ]
    else:
        wcs_gate = evaluate_footprint_gate(geometries, dataset_size=total)
        gate = wcs_gate
        if wcs_gate.accepted:
            source = GEOMETRY_SOURCE_WCS
            center = (wcs_gate.center_ra_deg, wcs_gate.center_dec_deg)
            frames = list(wcs_frames)
        else:
            reason = "native: %s; wcs: %s" % (
                native_reason,
                wcs_gate.reason or "strict WCS geometry unavailable",
            )
            return GeometrySelection(
                ResolvedReference(None, ORIGIN_AUTO_LEGACY),
                gate=wcs_gate,
                source_count=total,
                geometry_count=len(geometries),
                pointing_count=len(pointings),
                pointing_fraction=(len(pointings) / total if total else 0.0),
                geometry_source=GEOMETRY_SOURCE_UNAVAILABLE,
                fallback_reason=reason,
                scan_elapsed_s=scan_elapsed,
                scan_files_per_s=scan_rate,
                progress_event_count=reporter.count,
            )

    if source == GEOMETRY_SOURCE_NATIVE:
        offsets = angular_offsets_deg([item for _path, item in frames], center)
        seps = [(float(sep), path, item) for sep, (path, item) in zip(offsets, frames)]
    else:
        seps = []
        for path, geometry in frames:
            sep = angular_separation_deg(
                (geometry.center_ra_deg, geometry.center_dec_deg), center
            )
            seps.append((sep, path, geometry))
    seps.sort(key=lambda item: item[0])
    pool = seps[: min(CENTRAL_CANDIDATES, len(seps))]

    best_path = None
    best_metric = -math.inf
    best_sep = None
    for index, (sep, path, _geometry) in enumerate(pool):
        if stop_requested is not None and stop_requested():
            return GeometrySelection(ResolvedReference(None, ORIGIN_AUTO_LEGACY))
        if index == 0 or index + 1 == len(pool) or (index + 1) % 8 == 0:
            percent = int(round(100.0 * (index + 1) / len(pool))) if pool else 100
            reporter.emit(
                "Evaluating central candidates... %d / %d" % (index + 1, len(pool)),
                percent,
            )
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
            pointing_count=len(pointings),
            pointing_fraction=(len(pointings) / total if total else 0.0),
            candidate_count=len(pool), geometry_source=source,
            center_ra_deg=center[0], center_dec_deg=center[1],
            fallback_reason="no central candidate passed the historical quality metric",
            scan_elapsed_s=scan_elapsed, scan_files_per_s=scan_rate,
            progress_event_count=reporter.count,
        )

    distance_diagonals = None
    if source == GEOMETRY_SOURCE_WCS and gate.representative_diagonal_deg:
        distance_diagonals = best_sep / gate.representative_diagonal_deg
    return GeometrySelection(
        ResolvedReference(os.path.realpath(best_path), ORIGIN_AUTO_GEOMETRY),
        gate=gate, source_count=total, geometry_count=len(geometries),
        pointing_count=len(pointings),
        pointing_fraction=(len(pointings) / total if total else 0.0),
        candidate_count=len(pool), selected_metric=best_metric,
        selected_distance_diagonals=distance_diagonals,
        selected_offset_deg=best_sep,
        geometry_source=source,
        center_ra_deg=center[0], center_dec_deg=center[1],
        scan_elapsed_s=scan_elapsed, scan_files_per_s=scan_rate,
        progress_event_count=reporter.count,
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
