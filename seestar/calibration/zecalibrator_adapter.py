"""Optional ZeCalibrator API v1 integration adapter (public-import-only contract).

This module is the *only* place in Zsss that talks to ZeCalibrator, and it does
so strictly through the public, stable API surface ``zecalibrator.api.v1``.  It
never imports ZeCalibrator internals or private modules (``zecalibrator.api.v1._*``
is forbidden), never searches a sibling checkout, and never mutates the module
search path.

ZeCalibrator is optional: absence, incompatibility or a broken installation must
never break importing Zsss nor change the historical (non-calibrated) path.  The
public API module is therefore imported *lazily* (only inside ``probe()`` /
``open_session()``); there is no top-level import of ``zecalibrator``.

Capability negotiation (frozen contract): the provider is accepted only when
``get_api_info().api_version`` has major ``"1"`` **and** the capability IDs
``{session_library, auto_route, calibrate_frame, cancel}`` are all present.
Never a product-version comparison, never a minor equality, never ``>=``.
"""

from __future__ import annotations

import importlib
from typing import Any

from seestar.core.calibration_port import (
    CalibrationComposition,
    CalibrationError,
    CalibrationPlan,
    CalibrationResult,
    CalibrationSession,
    CalibrationState,
    CancellationHandle,
    ErrorKind,
    LightSource,
    MasterAdmission,
    ProviderInfo,
    ProviderState,
    RejectionDiagnostic,
    RouteKeyResult,
    RouteResolution,
    SessionResult,
)

# Referenced only as a string; never imported at module level.
_API_MODULE = "zecalibrator.api.v1"

PROVIDER_ID = "zecalibrator"

# The only supported ZeCalibrator public API major (exact match, never >=).
REQUIRED_API_MAJOR = "1"

# Hard requirement: the session-library + auto-route + unitary calibration +
# cooperative-cancellation capabilities.  Absence of any one makes the provider
# unusable (reported with the missing capability names).
REQUIRED_CAPABILITIES = frozenset(
    {"session_library", "auto_route", "calibrate_frame", "cancel"}
)

# Hard requirement on the public *symbol* surface (C26 ``light_route_key``).
# C26 is additive under API 1.1 (no capability/version bump), so it cannot be
# negotiated through ``REQUIRED_CAPABILITIES``; the adapter's ``route_key``
# depends on it, so its absence must be detected here and reported as
# INCOMPATIBLE (never silently swallowed into an empty plan map).
REQUIRED_API_SYMBOLS = frozenset({"light_route_key"})

# The light import contract source (raw 2-D sensor/CFA light, decoded by the
# provider).  Acquisition/geometry facts come from the FITS header; this only
# supplies the raw-domain evidence the strict decoder requires.
_LIGHT_CONTRACT_SOURCE = "zsss_light_contract"


def _import_api():
    """Import the public ZeCalibrator API lazily (first call only)."""
    return importlib.import_module(_API_MODULE)


def _is_zecalibrator_module_absent(exc: BaseException) -> bool:
    """True when ``exc`` means the public ZeCalibrator module itself is absent.

    A :class:`ModuleNotFoundError` naming the ``zecalibrator`` / ``zecalibrator.api``
    / ``zecalibrator.api.v1`` chain means "not installed"; one naming any *other*
    module means the public module was found but one of its internal imports
    failed ("installed but broken").
    """
    name = getattr(exc, "name", None)
    if not isinstance(name, str) or not name:
        return False
    return name == "zecalibrator" or name.startswith("zecalibrator.")


def _parse_major(version: Any) -> str | None:
    """Return the leading major component of an API version string ("1.1" -> "1")."""
    if version is None:
        return None
    try:
        return str(version).split(".")[0]
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _unavailable(exc: BaseException) -> CalibrationError:
    return CalibrationError(
        ErrorKind.UNAVAILABLE, f"{type(exc).__name__}: {exc}"
    )


def _failed(exc: BaseException) -> CalibrationError:
    return CalibrationError(ErrorKind.FAILED, f"{type(exc).__name__}: {exc}")


def probe() -> ProviderInfo:
    """Lazily discover and negotiate the installed ZeCalibrator public API v1.

    Compatibility is decided exclusively on the public ``api_version`` major
    (exact ``"1"``) plus the declared capabilities — never on Git branch or
    product version.  States: ``NOT_INSTALLED`` (public module absent),
    ``UNHEALTHY`` (present but broken import/probe), ``INCOMPATIBLE`` (wrong
    major or missing required capability), ``AVAILABLE`` otherwise.
    """
    try:
        api = _import_api()
    except ModuleNotFoundError as exc:
        if _is_zecalibrator_module_absent(exc):
            return ProviderInfo(
                state=ProviderState.NOT_INSTALLED,
                message=f"{type(exc).__name__}: {exc}",
            )
        return ProviderInfo(
            state=ProviderState.UNHEALTHY,
            message=f"import failed: {type(exc).__name__}: {exc}",
        )
    except Exception as exc:
        return ProviderInfo(
            state=ProviderState.UNHEALTHY,
            message=f"import failed: {type(exc).__name__}: {exc}",
        )

    try:
        info = api.get_api_info()
    except Exception as exc:
        return ProviderInfo(
            state=ProviderState.UNHEALTHY,
            message=f"get_api_info failed: {type(exc).__name__}: {exc}",
        )

    api_version = getattr(info, "api_version", None)
    major = _parse_major(api_version)
    product_version = getattr(info, "product_version", None)
    capabilities = tuple(getattr(info, "capabilities", ()) or ())

    if major != REQUIRED_API_MAJOR:
        return ProviderInfo(
            state=ProviderState.INCOMPATIBLE,
            provider_id=PROVIDER_ID,
            api_version=api_version,
            api_major=major,
            product_version=product_version,
            capabilities=capabilities,
            message=(
                f"incompatible API major {major!r} (expected {REQUIRED_API_MAJOR!r})"
            ),
        )

    missing = sorted(REQUIRED_CAPABILITIES - set(capabilities))
    if missing:
        return ProviderInfo(
            state=ProviderState.INCOMPATIBLE,
            provider_id=PROVIDER_ID,
            api_version=api_version,
            api_major=major,
            product_version=product_version,
            capabilities=capabilities,
            message=f"missing capabilities: {missing}",
        )

    # C26: the canonical route-class key is a hard dependency of ``route_key``.
    # It is additive (not a capability), so negotiate its presence directly via
    # the public ``__all__`` contract; a provider without it cannot build a plan
    # map and must be reported INCOMPATIBLE rather than silently degrading.
    _all = tuple(getattr(api, "__all__", ()) or ())
    missing_symbols = sorted(s for s in REQUIRED_API_SYMBOLS if s not in _all)
    if missing_symbols:
        return ProviderInfo(
            state=ProviderState.INCOMPATIBLE,
            provider_id=PROVIDER_ID,
            api_version=api_version,
            api_major=major,
            product_version=product_version,
            capabilities=capabilities,
            message=(
                f"missing public API symbol(s): {missing_symbols} "
                "(provider predates the C26 canonical route-class key)"
            ),
        )

    return ProviderInfo(
        state=ProviderState.AVAILABLE,
        provider_id=PROVIDER_ID,
        api_version=api_version,
        api_major=major,
        product_version=product_version,
        capabilities=capabilities,
    )


def calibration_tab_should_exist(info) -> bool:
    """Pure decision: should the Calibration tab exist for a provider probe?

    Only an ``AVAILABLE`` provider (imports cleanly, correct API major, all
    required capabilities) yields a Calibration tab. Absent / broken /
    incompatible / ``None`` → no tab, and ZSSS behaves exactly as before.

    This is a pure function (no Qt, no zecalibrator import) so the visibility
    decision is unit-testable with a fake :class:`ProviderInfo`.
    """
    if info is None:
        return False
    return getattr(info, "state", None) is ProviderState.AVAILABLE


class _LiveToken:
    """Provider-compatible cancellation token delegating to a neutral handle.

    RW-4: the mapping is **live** — ``is_cancelled()`` and ``raise_if_cancelled()``
    poll the neutral handle's CURRENT state at each call, so a cancellation that
    happens mid-operation is honoured by the provider's cooperative checkpoints
    (no dedicated thread, no lock, no deadlock). ``cancel`` is a no-op: the
    neutral handle owns the cancellation state.
    """

    __slots__ = ("_api", "_handle")

    def __init__(self, api, handle) -> None:
        self._api = api
        self._handle = handle

    def _handle_is_cancelled(self) -> bool:
        is_cancelled = getattr(self._handle, "is_cancelled", None)
        return bool(callable(is_cancelled) and is_cancelled())

    def is_cancelled(self) -> bool:
        return self._handle_is_cancelled()

    def raise_if_cancelled(self) -> None:
        if self._handle_is_cancelled():
            op_cancelled = getattr(self._api, "OperationCancelled", None)
            if op_cancelled is not None:
                raise op_cancelled()

    def cancel(self) -> None:
        # The neutral handle owns the cancellation state; nothing to propagate.
        return None


def _to_token(api, cancel: CancellationHandle | None):
    """Map a neutral cancellation handle onto a provider cancellation token.

    RW-4: the mapping is LIVE. When a neutral handle is supplied, a
    :class:`_LiveToken` is returned whose ``is_cancelled`` / ``raise_if_cancelled``
    delegate to the handle's CURRENT state at every checkpoint (an in-flight
    cancellation is therefore honoured). When ``cancel`` is None, a fresh provider
    token is returned (never cancelled).
    """
    if cancel is None:
        return api.CancellationToken()
    return _LiveToken(api, cancel)


def _is_cancelled(api, exc: BaseException) -> bool:
    op_cancelled = getattr(api, "OperationCancelled", None)
    return op_cancelled is not None and isinstance(exc, op_cancelled)


def _light_source(api, source: LightSource):
    declaration = api.ImportDeclaration(
        source=_LIGHT_CONTRACT_SOURCE,
        identity=source.logical_id or source.path,
        version="1",
        domain="raw",
        units="ADU",
    )
    return api.FitsFrameSource(path=source.path, declaration=declaration)


def _map_composition(plan) -> CalibrationComposition | None:
    """Map a provider plan's composition to a neutral :class:`CalibrationComposition`.

    RW-5: the effective composition (applied roles, level, additive state, flat
    applied, audit) is needed at preflight for the run freeze — not only after a
    calibration. Absent/None composition maps to None.
    """
    provider_composition = getattr(plan, "composition", None)
    if provider_composition is None:
        return None
    to_dict = getattr(provider_composition, "to_dict", None)
    if not callable(to_dict):
        return None
    return CalibrationComposition.from_dict(to_dict())


def _map_flat_facts(plan) -> tuple:
    """Map the flat form + bound master identities (role + content SHA-256) from
    a provider plan (C27).  Returns ``(flat_form, bound_masters)``.

    Provider-agnostic: only the neutral ``flat_form`` string and the
    ``(role, content_sha256)`` pairs cross the boundary — never a provider object.
    """
    flat_form = None
    bound_masters = ()
    masters = getattr(plan, "masters", None)
    if not masters:
        return flat_form, bound_masters
    flat_binding = masters.get("flat")
    if flat_binding is not None:
        role_descriptor = getattr(flat_binding, "role_descriptor", None)
        if role_descriptor is not None:
            flat_form = getattr(role_descriptor, "flat_form", None)
    bound_masters = tuple(
        (role, getattr(binding, "content_sha256", None))
        for role, binding in masters.items()
    )
    return flat_form, bound_masters


def _map_session_result(api, res, sensor_orientation=None) -> SessionResult:
    if res.operation_status == "CANCELLED":
        return SessionResult(
            state=CalibrationState.CANCELLED,
            warnings=tuple(getattr(res, "warnings", ()) or ()),
        )
    admissions = tuple(
        MasterAdmission(
            role=a.role,
            path=a.path,
            content_sha256=a.content_sha256,
            size_bytes=a.size_bytes,
            needs_attention=tuple(getattr(a, "needs_attention", ()) or ()),
        )
        for a in (res.admissions or ())
    )
    rejected = tuple(
        RejectionDiagnostic(path=r.path, reason_code=r.reason_code, detail=r.detail)
        for r in (res.rejected or ())
    )
    session = _ZeCalibratorSession(api, res.handle) if res.handle is not None else None
    return SessionResult(
        state=CalibrationState.COMPLETED,
        session=session,
        fingerprint=res.fingerprint or "",
        admissions=admissions,
        rejected=rejected,
        counts_by_role=dict(res.counts_by_role or {}),
        warnings=tuple(res.warnings or ()),
        sensor_orientation=sensor_orientation,
    )


class _ZeCalibratorSession:
    """Concrete :class:`CalibrationSession` over a ZeCalibrator ``SessionLibrary``.

    Mono-thread by contract: one instance per worker, never shared.  The provider
    plan objects are held opaquely and unwrapped only here (plan provenance C1).
    """

    def __init__(self, api, handle) -> None:
        self._api = api
        self._handle = handle

    @property
    def fingerprint(self) -> str:
        return self._handle.fingerprint

    @property
    def context_preparation_count(self) -> int:
        """Number of master-context preparations so far (S-a / §18 audit).

        Delegates to the provider handle's live counter: one preparation per
        distinct calibrated plan; reusing one plan across lights never re-prepares.
        """
        return getattr(self._handle, "context_preparation_count", 0)

    def resolve_light(self, source: LightSource, *, cancel=None) -> RouteResolution:
        token = _to_token(self._api, cancel)
        try:
            rr = self._handle.resolve_light(_light_source(self._api, source), cancel=token)
        except Exception as exc:
            if _is_cancelled(self._api, exc):
                return RouteResolution(
                    state=CalibrationState.CANCELLED,
                    error=CalibrationError(ErrorKind.CANCELLED, "cancelled"),
                )
            return RouteResolution(state=CalibrationState.FAILED, error=_failed(exc))

        if rr.operation_status == "CANCELLED":
            return RouteResolution(state=CalibrationState.CANCELLED)
        if rr.operation_status == "FAILED":
            return RouteResolution(
                state=CalibrationState.FAILED,
                error=CalibrationError(ErrorKind.FAILED, rr.details or "resolve failed"),
            )
        plan = (
            CalibrationPlan(plan_id=rr.plan.plan_id, provider_plan=rr.plan)
            if rr.plan is not None
            else None
        )
        composition = _map_composition(rr.plan) if rr.plan is not None else None
        if composition is not None:
            flat_form, bound_masters = _map_flat_facts(rr.plan)
            composition = CalibrationComposition(
                applied_roles=composition.applied_roles,
                skipped_roles=composition.skipped_roles,
                level=composition.level,
                additive_state=composition.additive_state,
                flat_applied=composition.flat_applied,
                no_candidate_roles=composition.no_candidate_roles,
                rejected_masters=composition.rejected_masters,
                flat_form=flat_form,
                bound_masters=bound_masters,
            )
        return RouteResolution(
            state=CalibrationState.COMPLETED,
            outcome=rr.outcome,
            plan=plan,
            composition=composition,
            reasons=tuple(getattr(r, "code", str(r)) for r in (rr.reasons or ())),
            warnings=tuple(getattr(r, "code", str(r)) for r in (rr.unverified or ())),
        )

    def calibrate(self, source: LightSource, plan: CalibrationPlan, *, cancel=None) -> CalibrationResult:
        if not isinstance(plan, CalibrationPlan):
            return CalibrationResult(
                state=CalibrationState.FAILED,
                error=CalibrationError(
                    ErrorKind.FAILED,
                    "plan must be a CalibrationPlan issued by resolve_light",
                ),
            )
        token = _to_token(self._api, cancel)
        try:
            cr = self._handle.calibrate(
                _light_source(self._api, source), plan._provider_plan, cancel=token
            )
        except Exception as exc:
            if _is_cancelled(self._api, exc):
                return CalibrationResult(
                    state=CalibrationState.CANCELLED,
                    error=CalibrationError(ErrorKind.CANCELLED, "cancelled"),
                )
            return CalibrationResult(state=CalibrationState.FAILED, error=_failed(exc))

        if cr.status == "CANCELLED":
            return CalibrationResult(
                state=CalibrationState.CANCELLED,
                warnings=tuple(getattr(cr, "warnings", ()) or ()),
            )
        if cr.status == "FAILED":
            return CalibrationResult(
                state=CalibrationState.FAILED,
                warnings=tuple(getattr(cr, "warnings", ()) or ()),
                error=CalibrationError(
                    ErrorKind.FAILED, getattr(cr, "reason_code", None) or "calibration failed"
                ),
            )
        provenance = {}
        prov = getattr(cr, "provenance", None)
        to_dict = getattr(prov, "to_dict", None)
        if callable(to_dict):
            provenance = dict(to_dict())
        return CalibrationResult(
            state=CalibrationState.COMPLETED,
            data=getattr(cr, "data", None),
            mask=getattr(cr, "mask", None),
            provenance=provenance,
            warnings=tuple(getattr(cr, "warnings", ()) or ()),
        )

    def close(self) -> None:
        self._handle.close()


class ZeCalibratorProvider:
    """Concrete :class:`CalibrationProvider` over ``zecalibrator.api.v1``.

    ``probe`` never raises; ``open_session`` never raises for expected
    operational failures (absence/incompatibility/errors map to neutral states).
    """

    name = PROVIDER_ID

    def probe(self) -> ProviderInfo:
        return probe()

    def open_session(self, root: str, *, cancel=None, sensor_orientation=None) -> SessionResult:
        try:
            api = _import_api()
        except Exception as exc:
            return SessionResult(state=CalibrationState.UNAVAILABLE, error=_unavailable(exc))

        # Refuse to build a session unless the provider negotiates as AVAILABLE.
        info = probe()
        if not info.available:
            return SessionResult(
                state=CalibrationState.UNAVAILABLE,
                error=CalibrationError(ErrorKind.UNAVAILABLE, info.message or "provider unavailable"),
            )

        token = _to_token(api, cancel)
        declaration = None
        if sensor_orientation is not None:
            declaration = api.SessionDeclaration(orientation=sensor_orientation)
        try:
            res = api.open_session_library(root, cancel=token, declaration=declaration)
        except Exception as exc:
            if _is_cancelled(api, exc):
                return SessionResult(
                    state=CalibrationState.CANCELLED,
                    error=CalibrationError(ErrorKind.CANCELLED, "cancelled"),
                )
            return SessionResult(state=CalibrationState.FAILED, error=_failed(exc))
        return _map_session_result(api, res, sensor_orientation=sensor_orientation)

    def route_key(self, path: str) -> str | None:
        """Return the canonical route-class key for a light (C26 helper).

        Header-only facts: decodes the light via the public ``inspect_frame`` and
        feeds the resulting ``LightConstraints`` into the provider's canonical
        ``light_route_key``.  ``None`` when the light cannot be decoded (absent
        / conflicting facts).  ZSSS never recomputes its own key.
        """
        return self.route_key_result(path).key

    def route_key_result(self, path: str) -> RouteKeyResult:
        """Return the canonical route-class key **plus a stable reason** when absent.

        Same contract as :meth:`route_key`, but the ``None`` key carries a
        bounded, ZSSS-owned reason token so consumers can aggregate ``reason ->
        count`` (never a provider object, never a traceback, never a per-frame
        log entry).  The C26 symbol absence is reported distinctly so the probe
        INCOMPATIBLE surface stays actionable.
        """
        try:
            api = _import_api()
        except Exception:
            return RouteKeyResult(key=None, reason="api_import_failed")
        try:
            src = _light_source(api, LightSource(path=path))
            res = api.inspect_frame(src)
        except Exception:
            return RouteKeyResult(key=None, reason="inspect_failed")
        if getattr(res, "operation_status", None) != "COMPLETED" or getattr(res, "inspection", None) is None:
            return RouteKeyResult(key=None, reason="inspection_incomplete")
        try:
            lc = api.light_constraints_from_sensor_metadata(res.inspection.metadata)
            key = api.light_route_key(lc)
        except AttributeError:
            return RouteKeyResult(key=None, reason="light_route_key_missing")
        except Exception:
            return RouteKeyResult(key=None, reason="constraints_failed")
        return RouteKeyResult(key=key)


__all__ = [
    "PROVIDER_ID",
    "REQUIRED_API_MAJOR",
    "REQUIRED_CAPABILITIES",
    "ZeCalibratorProvider",
    "calibration_tab_should_exist",
    "probe",
]
