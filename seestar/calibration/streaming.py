"""C5 — streaming calibration integrator (per-frame, bounded memory).

The runtime glue between the frozen C4 session plan and the per-frame streaming
loop.  It is the *only* place under the pipeline that reaches the calibration
adapter, and it holds **no provider object** in any long-lived structure except
the opaque session handle.

Design invariants (§16/§17/§18):
* per-frame, inside the existing batch loop — no session-wide materialisation;
* masters are reused via the session (never reloaded per light);
* a light without a frozen plan (or a failed resolve/calibrate) falls back to
  the historical path — never a hard failure;
* the scientific batch policy of ZSSS is not modified.
"""

from __future__ import annotations

from typing import Optional, Tuple

from seestar.calibration.scan import (
    FITS_SUFFIXES,
    flatten_masters,
    scan_masters_recursive,
)
from seestar.calibration.zecalibrator_adapter import ZeCalibratorProvider
from seestar.core.calibration_port import (
    CalibrationState,
    LightSource,
    ProviderInfo,
    ProviderState,
    RouteKeyResult,
)

# Bounding constants for the runtime provenance (JSON/log safe, never
# arbitrary provider text or tracebacks):
_MAX_REASON_KEYS = 48          # max distinct keys per reason->count aggregate
_MAX_CODE_LEN = 48            # max normalized provider reason-code length
_MAX_CODES_PER_RESOLUTION = 24  # max provider codes captured per resolution
_MAX_MSG_LEN = 200            # max bounded message length (open/probe reasons)


def _bound_message(value) -> str:
    """Collapse whitespace and bound a message (never a path/secret/traceback)."""
    s = " ".join(str(value or "").split())
    return s[:_MAX_MSG_LEN]


def _normalize_code(value) -> str:
    """Tokenize + bound an arbitrary provider reason code (JSON/log safe)."""
    s = str(value or "").strip()
    s = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in s.upper())
    if not s:
        return "unknown"
    return s[:_MAX_CODE_LEN]


def _add_bounded(store: dict, key: str) -> None:
    """Increment a reason->count aggregate, keeping its key count bounded."""
    key = str(key) or "unknown"
    if key in store:
        store[key] += 1
    elif len(store) < _MAX_REASON_KEYS:
        store[key] = 1
    else:
        store["_overflow"] = store.get("_overflow", 0) + 1


def _bounded_codes(rr) -> tuple:
    """Extract bounded, normalized provider reason codes from a RouteResolution.

    ``RouteResolution.reasons`` is already a tuple of code strings (the adapter
    maps each provider rejection to its stable ``code``); this just normalizes
    and bounds them (dedup, order-preserving, capped).
    """
    codes: list = []
    for r in getattr(rr, "reasons", ()) or ():
        token = _normalize_code(r)
        if token and token not in codes:
            codes.append(token)
        if len(codes) >= _MAX_CODES_PER_RESOLUTION:
            break
    return tuple(codes)


def _route_key_with_reason(provider, path) -> RouteKeyResult:
    """Return the canonical route-class key + a stable reason (never a traceback).

    Uses the richer ``route_key_result`` when the provider exposes it (the
    ZeCalibrator adapter does); otherwise falls back to the key-only
    ``route_key`` so duck-typed fake providers in tests keep working.
    """
    fn = getattr(provider, "route_key_result", None)
    if callable(fn):
        return fn(path)
    key = provider.route_key(path)
    return RouteKeyResult(key=key, reason=None if key is not None else "unavailable")


def _route_resolution_outcome(rr) -> tuple:
    """Return ``(plan, composition, outcome_token, codes)`` for a resolution.

    ``outcome_token`` is a stable, bounded ZSSS token on failure (``failed`` /
    ``cancelled`` / ``unavailable`` / ``no_match`` / ``ambiguous`` /
    ``no_plan`` / ``no_session``) and ``None`` on success.  ``codes`` is the
    bounded, normalized tuple of **real provider reason codes** (e.g.
    ``("EXPOSURE_MISMATCH",)``) carried by ``RouteResolution.reasons`` — never
    dropped, never a provider object, never a traceback.
    """
    if rr is None:
        return None, None, "no_session", ()
    state = getattr(rr, "state", None)
    if state is CalibrationState.CANCELLED:
        return None, None, "cancelled", ()
    if state is CalibrationState.FAILED:
        return None, None, "failed", _bounded_codes(rr)
    if state is CalibrationState.UNAVAILABLE:
        return None, None, "unavailable", ()
    if state is not CalibrationState.COMPLETED:
        return None, None, "failed", _bounded_codes(rr)
    plan = getattr(rr, "plan", None)
    composition = getattr(rr, "composition", None)
    if plan is None or composition is None:
        outcome = str(getattr(rr, "outcome", None) or "").upper()
        codes = _bounded_codes(rr)
        if outcome == "NO_MATCH":
            return None, None, "no_match", codes
        if outcome == "AMBIGUOUS":
            return None, None, "ambiguous", codes
        return None, None, "no_plan", codes
    return plan, composition, None, ()


def _calibration_result_reason(result) -> str:
    """Return a stable, bounded reason token for a non-COMPLETED calibration."""
    state = getattr(result, "state", None)
    if state is CalibrationState.CANCELLED:
        return "cancelled"
    if state is CalibrationState.UNAVAILABLE:
        return "unavailable"
    if state is CalibrationState.FAILED:
        return "failed"
    return "failed"


def _session_failure_reason(result) -> str:
    """Return a stable, bounded reason token for a failed/empty ``open_session``.

    Preserves the real cause without leaking a traceback or provider object:
    ``unavailable:<state>`` (probe refusal / import failure), ``failed``
    (operational error), ``cancelled``, or ``no_admissible_masters``.
    """
    state = getattr(result, "state", None)
    if state is CalibrationState.CANCELLED:
        return "cancelled"
    if state is CalibrationState.FAILED:
        return "failed"
    if state is CalibrationState.UNAVAILABLE:
        err = getattr(result, "error", None)
        if err is not None:
            msg = getattr(err, "message", None) or ""
            return f"unavailable:{msg}" if msg else "unavailable"
        return "unavailable"
    if getattr(result, "session", None) is None:
        return "no_admissible_masters"
    return ""


def _format_reason_aggregate(reasons: dict) -> str:
    """Render a bounded reason->count aggregate as a compact, sorted token list."""
    if not reasons:
        return ""
    return ",".join(f"{k}:{v}" for k, v in sorted(reasons.items()))


class CalibrationIntegrator:
    """Holds an open session and calibrates one light at a time.

    ``plan_map`` is the frozen C4 ``calibration_plan_map`` (light content
    signature -> ``{plan_id, composition}``).  When provided, only lights whose
    signature is present are calibrated; everything else falls back to the
    historical path.
    """

    def __init__(self, masters_folder: str, *, provider=None, plan_map=None, sensor_orientation=None) -> None:
        self._provider = provider or ZeCalibratorProvider()
        self._masters_folder = masters_folder
        self._plan_map = dict(plan_map) if plan_map else None
        self._plan_cache: dict = {}  # acquisition signature -> resolved plan object
        self._flat_dir = None  # scratch dir for the flattened recursive scan
        self._sensor_orientation = sensor_orientation  # neutral fallback declaration
        self._session = None
        self._session_result = None
        # Durable, bounded per-frame provenance (requested / applied / skipped /
        # failed + stable reasons + effective roles + DQ presence).  This is the
        # *runtime* counter of what actually got calibrated, kept distinct from
        # the frozen plan map (which only proves a plan was resolved at Start).
        self._requested = 0
        self._applied = 0
        self._skipped = 0
        self._failed = 0
        self._skip_reasons: dict = {}
        self._failure_reasons: dict = {}
        self._effective_roles: set = set()
        self._dq_present = False
        self._plan_reason = None  # why plan_map is empty when session opened
        self._open_reason = None  # real reason when the session could not open
        self._close_reason = None  # bounded reason when session.close() raised
        self._resolve_reasons: dict = {}  # real resolve reason -> count (bounded)

    # ------------------------------------------------------------------ open
    def open(self) -> bool:
        """Open the session (admission only). Returns True when usable.

        The chosen masters folder may be nested (e.g. M74 masters depth 3-4);
        ZeCalibrator's admission is top-level only, so when the root has no
        top-level FITS it is scanned **recursively (bounded)** and the candidates
        are flattened into a scratch dir before admission.  Admission and role
        identification stay ZeCalibrator's (raw non-masters are rejected with
        diagnostics).
        """
        import os
        import tempfile

        root = self._masters_folder
        admit_root = root
        self._flat_dir = None
        # C25: the scan must NOT stop when the root already contains a FITS —
        # descend into the selected subfolders too (mixed case), bounded depth,
        # output exclusions handled by ``scan_masters_recursive``.
        top_level: set = set()
        if os.path.isdir(root):
            try:
                top_level = {
                    os.path.join(root, e)
                    for e in os.listdir(root)
                    if os.path.isfile(os.path.join(root, e))
                    and e.lower().endswith(FITS_SUFFIXES)
                }
            except OSError:
                top_level = set()
        candidates = scan_masters_recursive(root)
        if candidates and set(candidates) != top_level:
            self._flat_dir = tempfile.mkdtemp(prefix="zsss_masters_flat_")
            admit_root = flatten_masters(candidates, self._flat_dir)
        result = self._provider.open_session(
            admit_root, sensor_orientation=self._sensor_orientation
        )
        self._session_result = result
        self._session = getattr(result, "session", None)
        if self._session is None:
            # Real, stable reason for a failed/empty open (never a traceback,
            # never a provider object) so the caller can surface it actionably.
            self._open_reason = _session_failure_reason(result)
        else:
            self._open_reason = None
        return self._session is not None

    @property
    def session(self):
        return self._session

    @property
    def fingerprint(self) -> str:
        return getattr(self._session_result, "fingerprint", "") or ""

    @property
    def open_reason(self) -> str:
        """Real, stable reason the session did not open (``""`` when open)."""
        return self._open_reason or ""

    @property
    def close_reason(self) -> str:
        """Bounded reason when ``session.close()`` raised (``""`` otherwise)."""
        return self._close_reason or ""

    def probe_info(self):
        """Return the provider's probe result (never raises).

        F4: a provider whose ``probe()`` raises (broken install) maps to a
        bounded ``ProviderInfo(UNHEALTHY, message=type)`` — never a traceback,
        never a crash at Start.
        """
        try:
            return self._provider.probe()
        except Exception as exc:  # noqa: BLE001 — fail-open probe (F4)
            return ProviderInfo(
                state=ProviderState.UNHEALTHY,
                message=f"probe raised {type(exc).__name__}",
            )

    @property
    def context_preparations(self) -> int:
        """Master-context preparations so far (§18 audit counter)."""
        if self._session is None:
            return 0
        return getattr(self._session, "context_preparation_count", 0)

    def is_planned(self, file_path: str) -> bool:
        """True when the light's route class has a frozen plan.

        Keyed by the **canonical route-class key** (C26 ``light_route_key``),
        not a homemade signature: the plan is per-class, so a light is planned
        when its route class was resolved at preflight (or when there is no plan
        map at all).
        """
        if self._plan_map is None:
            return True
        key = self._provider.route_key(file_path)
        return key is not None and key in self._plan_map

    # ------------------------------------------------------------ provenance
    def record_requested(self) -> None:
        """Count one frame that entered the calibration path (C-bounded)."""
        self._requested += 1

    def record_applied(self, *, roles=(), dq_present=False) -> None:
        """Count one frame whose calibrated pixels were actually consumed."""
        self._applied += 1
        for role in roles or ():
            if role:
                self._effective_roles.add(str(role))
        if dq_present:
            self._dq_present = True

    def record_skipped(self, reason: str) -> None:
        """Count one frame that fell back to the historical path, with a reason."""
        self._skipped += 1
        _add_bounded(self._skip_reasons, reason)

    def record_failed(self, reason: str) -> None:
        """Count one calibration failure (never silently a skip), with a reason."""
        self._failed += 1
        _add_bounded(self._failure_reasons, reason)

    def _record_resolve(self, outcome: str, codes=()) -> None:
        """Record a resolution failure: stable outcome **and** real provider codes.

        Both land in the same bounded ``_resolve_reasons`` aggregate, kept
        distinguishable by key: bare stable outcome tokens (``no_match`` /
        ``ambiguous`` / ``failed`` / …) versus ``code:<NORMALIZED_CODE>`` for the
        real provider reason codes — never a traceback, never a provider object.
        """
        _add_bounded(self._resolve_reasons, outcome)
        for code in codes or ():
            _add_bounded(self._resolve_reasons, f"code:{_normalize_code(code)}")

    def provenance_snapshot(self) -> dict:
        """Return the JSON-safe, bounded runtime provenance.

        Distinguishes three states explicitly (never conflated):
          * ``session_open``       — a session was admitted (fingerprint non-empty)
          * ``classes_planned``    — plan map resolved at Start
          * ``frames applied/skipped/failed`` — pixels actually calibrated
        """
        return {
            "calibration_enabled_requested": True,
            "calibration_requested": self._requested,
            "calibration_session_open": self._session is not None,
            "calibration_library_fingerprint": self.fingerprint or "",
            "calibration_classes_planned": len(self._plan_cache),
            "calibration_plan_reason": self._plan_reason,
            "calibration_open_reason": self._open_reason,
            "calibration_close_reason": self._close_reason,
            "calibration_frames_applied": self._applied,
            "calibration_frames_skipped": self._skipped,
            "calibration_frames_failed": self._failed,
            "calibration_skip_reasons": dict(self._skip_reasons),
            "calibration_failure_reasons": dict(self._failure_reasons),
            "calibration_resolve_reasons": dict(self._resolve_reasons),
            "calibration_effective_roles": sorted(self._effective_roles),
            "calibration_dq_present": bool(self._dq_present),
        }

    # ---------------------------------------------------------------- per frame
    def resolve(self, file_path: str):
        """Resolve a light's plan (cached by route class — no re-decode).

        The plan is looked up by the light's **canonical route-class key** (C26).
        A class already resolved at preflight returns the cached plan object with
        no ``resolve_light`` decode; a new class (not in the frozen map) falls
        back to a direct resolve.  Returns ``None`` when not MATCHED / not planned
        / unresolvable.  A failed/skipped resolve records its **real** reason
        (stable outcome + provider codes) into the bounded
        ``calibration_skip_reasons`` / ``calibration_resolve_reasons`` aggregates.
        """
        if self._session is None:
            return None
        key_result = _route_key_with_reason(self._provider, file_path)
        acq_sig = key_result.key
        if acq_sig is None:
            reason = f"route_key:{key_result.reason or 'unavailable'}"
            self.record_skipped(reason)
            _add_bounded(self._resolve_reasons, reason)
            return None
        cached = self._plan_cache.get(acq_sig)
        if cached is not None:
            return cached
        if not self.is_planned(file_path):
            self.record_skipped("not_planned")
            return None
        plan, composition, outcome, codes = self._resolve_direct(file_path)
        if plan is None:
            self.record_skipped(f"resolve:{outcome}")
            self._record_resolve(outcome, codes)
            return None
        self._plan_cache[acq_sig] = plan
        if composition is not None:
            for role in getattr(composition, "applied_roles", ()) or ():
                if role:
                    self._effective_roles.add(str(role))
        return plan

    def calibrate(self, file_path: str, plan) -> Optional[Tuple]:
        """Calibrate one light -> ``(physical_float32, mask)`` or ``None``.

        ``mask`` is the provider DQ mask (``mask != 0`` == invalid) — carried and
        returned for C6, never silently dropped.  A cancellation in flight
        propagates (the port already maps it) and is not swallowed here.  A
        failed calibration records its **real** reason into the bounded
        ``calibration_failure_reasons`` aggregate.
        """
        if self._session is None or plan is None:
            self.record_failed("no_session" if self._session is None else "no_plan")
            return None
        result = self._session.calibrate(LightSource(path=file_path), plan)
        state = getattr(result, "state", None)
        if state is not CalibrationState.COMPLETED:
            self.record_failed(_calibration_result_reason(result))
            return None
        data = getattr(result, "data", None)
        mask = getattr(result, "mask", None)
        if data is None:
            self.record_failed("no_data")
            return None
        if mask is not None:
            self._dq_present = True
        return data, mask

    def freeze_snapshot(self, lights=()) -> dict:
        """Build the JSON-safe calibration freeze (7 fields) for the run contract.

        Provider id / api / product version come from ``probe()``; the library
        fingerprint from the open session.  The ``calibration_plan_map`` is keyed
        by the **acquisition signature** (header-only): lights are grouped by
        acquisition class and ONE representative per class is resolved (1 decode
        per class, never one per frame).  The resolved plan objects are cached
        in ``self._plan_cache`` for the streaming loop, and the **effective
        roles** are recorded at cache time so a later cached ``resolve()`` never
        loses them.  Returns ``{}`` when no session is open (== "calibration
        disabled").  Never carries a provider object.
        """
        if self._session is None:
            return {}
        info = self.probe_info()
        # Reset THIS freeze's resolve-reason aggregate: freeze_snapshot is
        # re-run (bootstrap -> plan-binding -> resume), and a stale aggregate
        # would double-count across reconstructions.
        self._resolve_reasons = {}
        plan_map = {}
        representatives: dict = {}
        for path in lights or ():
            key_result = _route_key_with_reason(self._provider, path)
            if key_result.key is not None:
                representatives.setdefault(key_result.key, path)
            else:
                # A light that cannot be route-classed is a real freeze cause.
                _add_bounded(
                    self._resolve_reasons,
                    f"route_key:{key_result.reason or 'unavailable'}",
                )
        for acq_sig, rep in representatives.items():
            plan, composition, outcome, codes = self._resolve_direct(rep)
            if plan is None or composition is None:
                # Preserve the real resolve reason (stable outcome + provider
                # codes) as a bounded reason->count aggregate.
                self._record_resolve(outcome, codes)
                continue
            self._plan_cache[acq_sig] = plan
            # F1: record effective roles NOW (at cache time) so the cached
            # resolve() path (which never re-reads the composition) still
            # reports dark/flat in calibration_effective_roles.
            for role in getattr(composition, "applied_roles", ()) or ():
                if role:
                    self._effective_roles.add(str(role))
            plan_map[acq_sig] = {
                "plan_id": plan.plan_id,
                "composition": composition.to_dict(),
            }
        # Durable reason when the session opened but NO plan could be resolved
        # (never a silent empty plan map): a bounded aggregate of the real
        # causes (route_key failures + resolve outcomes + provider codes).
        if not plan_map:
            self._plan_reason = (
                _format_reason_aggregate(self._resolve_reasons)
                or "no representatives resolved"
            )
        else:
            self._plan_reason = None
        return {
            "calibration_enabled": True,
            "calibration_provider": getattr(info, "provider_id", None) or "",
            "calibration_api_version": getattr(info, "api_version", None) or "",
            "calibration_product_version": getattr(info, "product_version", None) or "",
            "calibration_library_fingerprint": self.fingerprint or "",
            "calibration_contract_versions": {},
            "calibration_plan_map": plan_map,
            "calibration_orientation_declaration": self._sensor_orientation,
        }

    def _resolve_direct(self, file_path: str):
        """Directly resolve a light (decode) -> ``(plan, composition, outcome, codes)``.

        ``outcome`` is a stable, ZSSS-owned token on failure (``failed`` /
        ``cancelled`` / ``unavailable`` / ``no_match`` / ``ambiguous`` /
        ``no_plan`` / ``no_session``); ``None`` on success.  ``codes`` is the
        bounded tuple of real provider reason codes.  Never a provider object
        and never a traceback.
        """
        if self._session is None:
            return None, None, "no_session", ()
        rr = self._session.resolve_light(LightSource(path=file_path))
        return _route_resolution_outcome(rr)

    def close(self) -> None:
        """Close the session idempotently; a raising provider close is captured.

        Robustness: a ``session.close()`` that raises must neither break the
        caller's finally nor lose the durable proof.  The failure is recorded as
        a bounded ``close_failed:<type>`` reason in ``_close_reason`` so the
        final artifact can still be emitted/written.
        """
        if self._session is not None:
            try:
                self._session.close()
            except Exception as exc:  # noqa: BLE001 — bounded close failure
                self._close_reason = f"close_failed:{type(exc).__name__}"
            finally:
                self._session = None
                self._session_result = None
        if getattr(self, "_flat_dir", None):
            import shutil

            shutil.rmtree(self._flat_dir, ignore_errors=True)
            self._flat_dir = None


__all__ = ["CalibrationIntegrator"]
