"""C4 — calibration session preflight + flat policy + freeze payload.

Produces, at Start (before any processing), a bounded session plan: open a
session over the masters folder, resolve each selected light's route, apply the
group-level flat policy (mission §6), compute the library fingerprint, and build
the JSON-safe freeze payload for the run contract (§15 / D6).

**Grouping key (discovered):** ZSSS produces ONE scientific stack per run — all
accepted frames are coadded together regardless of exposure (exposure is only
per-frame metadata; the final ``EXPTIME`` is written only when uniform, cf.
``queue_manager._apply_exposure_metadata``).  Therefore the *flat policy* group
is the whole run.  The summary groups lights by exposure only for display: the
*additive* plan is per-light (ZeCalibrator matches the dark by exposure), while
the flat decision is a single whole-run decision.

Zero science in ZSSS: no matching, no master selection, no equation —
ZeCalibrator decides every route; ZSSS only aggregates and freezes JSON-safe
values (never provider objects).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from seestar.calibration.zecalibrator_adapter import ZeCalibratorProvider
from seestar.core.calibration_port import (
    LightSource,
    ProviderState,
    RouteResolution,
)

# The exact informative message required by mission §6.  Localized upstream by
# the caller; stored here as the canonical English text.
FLAT_UNAVAILABLE_MESSAGE = (
    "Compatible additive calibration found. No common compatible flat is "
    "available for all frames; stacking will continue with dark/bias "
    "calibration only."
)


@dataclass(frozen=True)
class GroupSummary:
    """Display summary of one exposure group (same additive plan)."""

    exposure_label: str  # "60", "180", "unknown"
    light_count: int
    additive_mode: str
    flat_mode: str  # "none" | "apply" (whole-run decision, identical across groups)


@dataclass(frozen=True)
class LightPlanEntry:
    """Per-class freeze entry (for ``calibration_plan_map``).

    ``signature`` is the **acquisition signature** (header-only: exposure, gain,
    binning, CFA phase, detector, dimensions) — never a per-frame content hash.
    Lights sharing a signature share the same auto-routed plan class.
    """

    signature: str  # acquisition signature (header-only, not a content hash)
    plan_id: str
    composition: Mapping[str, Any]  # JSON-safe


@dataclass
class PreflightResult:
    """The bounded preflight outcome (displayable + freezable)."""

    available: bool
    fingerprint: str = ""
    provider_id: str = ""
    product_version: str = ""
    api_version: str = ""
    groups: Tuple[GroupSummary, ...] = ()
    flat_applied: bool = False
    flat_message: str = ""
    plan_entries: Tuple[LightPlanEntry, ...] = ()
    summary: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without ZeCalibrator)
# ---------------------------------------------------------------------------
def light_signature(path: str) -> str:
    """Return a content SHA-256 of a light file (stable, never a filename rule)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_header(path: str):
    """Read a light's FITS header (metadata only — never decodes pixels)."""
    try:
        from astropy.io import fits
    except Exception:  # noqa: BLE001 - astropy optional for header reads
        return None
    try:
        with fits.open(path) as hdul:
            return hdul[0].header
    except Exception:  # noqa: BLE001 - never crash the preflight on a header read
        return None


def _night_from_dateobs(hdr) -> str:
    """Return the observing **night** (coarse calendar date) for a light.

    Astronomical nights cross midnight: an observation at 00:30 belongs to the
    PREVIOUS evening's night, not the wall-clock date.  The day boundary is
    therefore shifted to local noon (DATE-OBS minus 12h) before taking the
    calendar date — so 2026-09-14T23:41 and 2026-09-15T00:41 both map to
    ``2026-09-14``.  A full timestamp would make one class per image and destroy
    the per-class cache; this coarse night keeps a whole night as one class.
    """
    val = hdr.get("DATE-OBS")
    if val is None:
        return ""
    try:
        from datetime import datetime, timedelta

        dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return ""
    return (dt - timedelta(hours=12)).date().isoformat()


def acquisition_signature(path: str) -> str:
    """Return a stable acquisition signature from a light's FITS **header only**.

    Key = (exposure, gain, binning, CFA phase, detector, dimensions, night,
    filter).  ``night`` is the coarse observing night (see :func:`_night_from_dateobs`),
    never a full timestamp.  Header-only (no pixel decode): two lights sharing the
    signature share the same auto-routed plan class.  Deterministic; never a
    content hash and never a filename rule.
    """
    hdr = _read_header(path)
    if hdr is None:
        return "unknown"

    exposure = None
    for key in ("EXPTIME", "EXPOSURE"):
        v = hdr.get(key)
        if v is not None:
            try:
                exposure = float(v)
            except (TypeError, ValueError):
                exposure = None
            if exposure is not None:
                break

    gain = hdr.get("GAIN")
    binx = hdr.get("XBINNING", hdr.get("BINNING"))
    biny = hdr.get("YBINNING", hdr.get("BINNING"))
    cfa = hdr.get("BAYERPAT", hdr.get("BAYERPOL"))
    detector = hdr.get("INSTRUME", hdr.get("DETECTOR"))
    nx = hdr.get("NAXIS1")
    ny = hdr.get("NAXIS2")
    filter_ = hdr.get("FILTER")
    night = _night_from_dateobs(hdr)

    def _s(v) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, (int, float)):
            return str(v)
        return str(v).strip().upper()

    return "|".join([
        _s(exposure), _s(gain), f"{_s(binx)}x{_s(biny)}",
        _s(cfa), _s(detector), f"{_s(nx)}x{_s(ny)}",
        night, _s(filter_),
    ])


def flat_policy(flat_applied_flags: Sequence[bool]) -> Tuple[bool, str]:
    """Decide the whole-run flat mode from each light's flat decision.

    Returns ``(flat_applied, message)``.  ``flat_applied`` is ``True`` only when
    **every** light applies a flat; otherwise flat is OFF for the whole group
    (dark/bias conserved) and the §6 informative message is emitted.
    """
    if not flat_applied_flags:
        return False, ""
    if all(flat_applied_flags):
        return True, ""
    return False, FLAT_UNAVAILABLE_MESSAGE


def _exposure_label(path: str) -> str:
    """Return a display label for a light's EXPTIME/EXPOSURE header card."""
    exposure = _read_exposure(path)
    if exposure is None:
        return "unknown"
    if float(exposure).is_integer():
        return str(int(exposure))
    return str(exposure)


def _read_exposure(path: str) -> Optional[float]:
    """Read EXPTIME (then EXPOSURE) from a light's FITS header (metadata only)."""
    try:
        from astropy.io import fits
    except Exception:  # noqa: BLE001 - astropy is optional for the preflight display
        return None
    try:
        with fits.open(path) as hdul:
            hdr = hdul[0].header
            for key in ("EXPTIME", "EXPOSURE"):
                val = hdr.get(key)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        return None
        return None
    except Exception:  # noqa: BLE001 - never crash the preflight on a header read
        return None


# ---------------------------------------------------------------------------
# Main preflight chain
# ---------------------------------------------------------------------------
def preflight_calibration(
    masters_folder: str,
    lights: Sequence[str],
    *,
    provider=None,
) -> PreflightResult:
    """Run the preflight chain and return a bounded, freezable result.

    ``lights`` are raw FITS light paths.  ZeCalibrator decides every route; ZSSS
    only aggregates (per-exposure display groups, whole-run flat decision) and
    freezes JSON-safe values.  Never raises: an unavailable provider or a failed
    open yields ``available=False``.
    """
    provider = provider or ZeCalibratorProvider()
    info = provider.probe()
    if info.state is not ProviderState.AVAILABLE:
        return PreflightResult(available=False, error=info.message or "provider unavailable")

    session_result = provider.open_session(masters_folder)
    session = getattr(session_result, "session", None)
    if session is None:
        return PreflightResult(available=False, error="no admissible masters (empty session)")

    # Group lights by the canonical route-class key (C26 light_route_key,
    # header-only facts via the provider).  One representative resolved per
    # class.  An unresolvable light (decode failure) is a DISTINCT class (never
    # grouped with a known class).
    lights_by_acq: Dict[str, List[str]] = {}
    for path in lights:
        key = provider.route_key(path) or f"unresolvable:{path}"
        lights_by_acq.setdefault(key, []).append(path)

    # Resolve ONE representative per acquisition class (1 decode per class,
    # never one per frame).  ``resolved`` = [(acq_sig, representative_path, rr)].
    resolved: List[Tuple[str, str, RouteResolution]] = []
    for acq_sig, paths in lights_by_acq.items():
        rep = paths[0]
        rr = session.resolve_light(LightSource(path=rep))
        resolved.append((acq_sig, rep, rr))

    # Flat policy (whole-run, mission §6) from the representatives' flat decision.
    flat_flags = [
        bool(getattr(rr.composition, "flat_applied", False))
        for _sig, _rep, rr in resolved
        if rr.composition is not None
    ]
    flat_applied, flat_message = flat_policy(flat_flags)

    # Group by exposure label for display (additive plan is per-class).
    groups_by_exposure: Dict[str, List[RouteResolution]] = {}
    for _sig, rep, rr in resolved:
        label = _exposure_label(rep)
        groups_by_exposure.setdefault(label, []).append(rr)

    groups: List[GroupSummary] = []
    for label, rrs in groups_by_exposure.items():
        additive = _common_additive(rrs)
        groups.append(
            GroupSummary(
                exposure_label=label,
                light_count=len(rrs),
                additive_mode=additive,
                flat_mode="apply" if flat_applied else "none",
            )
        )
    groups.sort(key=lambda g: _exposure_sort_key(g.exposure_label))

    # One freeze entry per acquisition class (key = acquisition signature).
    plan_entries: List[LightPlanEntry] = []
    for acq_sig, _rep, rr in resolved:
        if rr.plan is None or rr.composition is None:
            continue
        plan_entries.append(
            LightPlanEntry(
                signature=acq_sig,
                plan_id=rr.plan.plan_id,
                composition=rr.composition.to_dict(),
            )
        )

    result = PreflightResult(
        available=True,
        fingerprint=session_result.fingerprint or "",
        provider_id=getattr(info, "provider_id", None) or "",
        product_version=getattr(info, "product_version", None) or "",
        api_version=getattr(info, "api_version", None) or "",
        groups=tuple(groups),
        flat_applied=flat_applied,
        flat_message=flat_message,
        plan_entries=tuple(plan_entries),
    )
    result.summary = render_summary(result)
    return result


def _common_additive(rrs: Sequence[RouteResolution]) -> str:
    """Return the common additive mode of a group (first non-None, else unknown)."""
    for rr in rrs:
        if rr.composition is not None:
            additive = getattr(rr.composition, "additive_state", None)
            if additive:
                return str(additive)
    return "unknown"


def _exposure_sort_key(label: str) -> Tuple[int, float]:
    try:
        return (0, float(label))
    except ValueError:
        return (1, 0.0)


def render_summary(result: PreflightResult) -> str:
    """Render the bounded, displayable/loggable summary text."""
    lines = ["Calibration plan summary"]
    for g in result.groups:
        lines.append(
            f"  {g.exposure_label} s : {g.light_count:>2} lights"
            f" | additive: {g.additive_mode} | flat: {g.flat_mode}"
        )
    lines.append(
        f"  fingerprint: {result.fingerprint}"
        f"   provider: {result.provider_id} {result.product_version}"
        f" API {result.api_version}"
    )
    return "\n".join(lines)


def build_freeze(result: PreflightResult) -> Dict[str, Any]:
    """Build the JSON-safe calibration freeze (the 7 run-contract fields).

    Returns an empty dict when calibration was not enabled/frozen (absence is
    the "calibration disabled" signal — never an invented value).  The
    ``calibration_plan_map`` stores, per light signature, ``plan_id`` + the
    JSON-safe composition (applied roles, level, additive state, flat applied) —
    never a provider object.
    """
    if not result.available or not result.plan_entries:
        return {}
    plan_map: Dict[str, Any] = {}
    for entry in result.plan_entries:
        plan_map[entry.signature] = {
            "plan_id": entry.plan_id,
            "composition": dict(entry.composition),
        }
    return {
        "calibration_enabled": True,
        "calibration_provider": result.provider_id,
        "calibration_api_version": result.api_version,
        "calibration_product_version": result.product_version,
        "calibration_library_fingerprint": result.fingerprint,
        "calibration_contract_versions": {},  # frozen science/matching/provenance schema versions
        "calibration_plan_map": plan_map,
    }


__all__ = [
    "FLAT_UNAVAILABLE_MESSAGE",
    "GroupSummary",
    "LightPlanEntry",
    "PreflightResult",
    "acquisition_signature",
    "build_freeze",
    "flat_policy",
    "light_signature",
    "preflight_calibration",
    "render_summary",
]
