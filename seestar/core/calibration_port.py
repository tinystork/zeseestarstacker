"""Internal CalibrationPort boundary for master calibration in ZeSeestarStacker.

This module defines the *internal*, transport-neutral contract between the Zsss
pipeline and any concrete calibration provider (ZeCalibrator today).  It is
deliberately decoupled from a specific provider so that:

* the existing pipeline keeps running unchanged when no provider is installed,
  and
* the optional ZeCalibrator integration
  (:mod:`seestar.calibration.zecalibrator_adapter`) can be layered on top
  without touching the scientific processing semantics, and
* the port itself can be tested against a fake provider (no ZeCalibrator).

Consumed contract (ZeCalibrator public API v1, ``zecalibrator.api.v1``)
------------------------------------------------------------------------

The adapter is written against the *public, stable* surface only:
``get_api_info``, ``open_session_library``, ``SessionLibrary`` (``fingerprint``,
``resolve_light``, ``calibrate``, ``close``), ``SessionLibraryResult``,
``RouteResolution``, ``MasterAdmission``, ``RejectionDiagnostic``,
``PlanSourceMismatchError``, ``CancellationToken``.  Capability negotiation uses
``get_api_info().api_version`` major == "1" **and** the presence of the
capability IDs ``{session_library, auto_route, calibrate_frame, cancel}`` —
never a product-version comparison, never a minor equality, never ``>=``.

Design rules (ZeSoftware integration guidelines)
------------------------------------------------

* **Optionality.**  Calibration is an optional capability.  The absence,
  incompatibility or a broken installation of the provider must never break
  importing Zsss nor change the historical (non-calibrated) path.  Discovery is
  always lazy (never at import time).
* **Public API only.**  Only ``zecalibrator.api.v1`` is ever touched; private
  modules are never imported and ``sys.path`` is never mutated.
* **No type leak.**  No provider (ZeCalibrator) type ever crosses this boundary;
  everything here is a Zsss-neutral value object.

This module imports **only the standard library**.  In particular it must never
import ``zecalibrator``, NumPy, Astropy or Qt at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol


# State/enum semantics (RW-6 S2):
#   CalibrationState — outcome of ONE operation (open / resolve / calibrate).
#   ProviderState    — discovery/negotiation result (is the provider usable at all?).
#   ErrorKind        — category of a CalibrationError (why an operation is not COMPLETED).


class CalibrationState(str, Enum):
    """Outcome state of a calibration operation (internal, not the provider's).

    These are the *Zsss* states; the adapter is responsible for mapping the
    provider's operation status onto them.
    """

    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"


class ProviderState(str, Enum):
    """Lazy-discovery state for the optional calibration provider."""

    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed"
    INCOMPATIBLE = "incompatible"
    UNHEALTHY = "unhealthy"


class ErrorKind(str, Enum):
    """Neutral error taxonomy.  ``UNAVAILABLE`` is deliberately distinct from
    ``FAILED``: the former means "the provider cannot be used at all" (absent,
    incompatible, unhealthy), the latter means "a usable provider produced an
    error for this specific operation".
    """

    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CalibrationError(Exception):
    """Transport-neutral calibration error (no provider type leaks).

    ``kind`` is one of :class:`ErrorKind`; ``message`` is a human-readable,
    provider-agnostic description.
    """

    def __init__(self, kind: ErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind if isinstance(kind, ErrorKind) else ErrorKind(kind)
        self.message = str(message)


@dataclass(frozen=True)
class ProviderInfo:
    """Result of a lazy calibration-provider discovery / capability probe.

    ``available`` is ``True`` only when the provider imports cleanly, exposes
    the required API major, and declares every required capability.
    """

    state: ProviderState
    provider_id: str | None = None
    api_version: str | None = None
    api_major: str | None = None
    product_version: str | None = None
    capabilities: tuple[str, ...] = ()
    message: str | None = None  # reason when not AVAILABLE (never a traceback)

    @property
    def available(self) -> bool:
        return self.state is ProviderState.AVAILABLE


@dataclass(frozen=True)
class LightSource:
    """A single raw FITS light the provider decodes and calibrates.

    Only the filesystem path is carried: the provider is responsible for
    decoding the light (never Zsss, which would double-decode or invent
    metadata).  ``logical_id`` is an optional Zsss-side identity for provenance
    (it never reaches the provider as a scientific fact).
    """

    path: str
    logical_id: str | None = None


class CalibrationPlan:
    """Opaque, transport-neutral plan handle.

    ``plan_id`` is the stable identifier Zsss freezes per light (library freeze
    / resume).  The provider-specific plan object is held opaquely and may only
    be unwrapped by the adapter that produced it (plan provenance, contract C1).
    """

    def __init__(self, plan_id: str, provider_plan: Any) -> None:
        self._plan_id = str(plan_id)
        self._provider_plan = provider_plan

    @property
    def plan_id(self) -> str:
        return self._plan_id


@dataclass(frozen=True)
class MasterAdmission:
    """One master admitted into the session library (neutral).

    ``role`` is ``bias`` / ``dark`` / ``flat`` / ``flat_dark``; content identity
    is the whole-file SHA-256 + byte size of the master FITS.  ``needs_attention``
    carries informational evidence gaps (e.g. flat quality evidence) — never an
    admission filter, resolved as UNVERIFIED by the matcher.
    """

    role: str
    path: str
    content_sha256: str
    size_bytes: int
    needs_attention: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "needs_attention", tuple(self.needs_attention))


@dataclass(frozen=True)
class RejectionDiagnostic:
    """One master refused at admission, with a structured reason."""

    path: str
    reason_code: str
    detail: str = ""


@dataclass(frozen=True)
class CalibrationComposition:
    """Neutral, frozen, JSON-safe effective composition of a resolved route.

    Filled by the adapter at resolve time (preflight) from the provider's plan
    composition, so the freeze can record what a plan will actually apply
    (applied roles, availability-relative level, additive state, flat applied,
    and the skip/no-candidate/rejected audit) **before** any calibration runs.
    Provider-agnostic: no ZSSS vocabulary (no ``group``/``stack``).
    """

    applied_roles: tuple[str, ...] = ()
    skipped_roles: tuple = ()
    level: str | None = None
    additive_state: str | None = None
    flat_applied: bool | None = None
    no_candidate_roles: tuple[str, ...] = ()
    rejected_masters: tuple = ()
    flat_form: str | None = None  # C27: effective flat form (normalize_only/...)
    bound_masters: tuple = ()  # C27: ((role, content_sha256), ...) identities

    def __post_init__(self) -> None:
        object.__setattr__(self, "applied_roles", tuple(self.applied_roles))
        object.__setattr__(self, "skipped_roles", tuple(self.skipped_roles))
        object.__setattr__(self, "no_candidate_roles", tuple(self.no_candidate_roles))
        object.__setattr__(self, "rejected_masters", tuple(self.rejected_masters))
        object.__setattr__(self, "bound_masters", tuple(tuple(b) for b in self.bound_masters))

    def to_dict(self) -> dict:
        return {
            "applied_roles": list(self.applied_roles),
            "skipped_roles": [dict(s) for s in self.skipped_roles],
            "level": self.level,
            "additive_state": self.additive_state,
            "flat_applied": self.flat_applied,
            "no_candidate_roles": list(self.no_candidate_roles),
            "rejected_masters": [dict(r) for r in self.rejected_masters],
            "flat_form": self.flat_form,
            "bound_masters": [list(b) for b in self.bound_masters],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "CalibrationComposition":
        return cls(
            applied_roles=tuple(d.get("applied_roles", ()) or ()),
            skipped_roles=tuple(d.get("skipped_roles", ()) or ()),
            level=d.get("level"),
            additive_state=d.get("additive_state"),
            flat_applied=d.get("flat_applied"),
            no_candidate_roles=tuple(d.get("no_candidate_roles", ()) or ()),
            rejected_masters=tuple(d.get("rejected_masters", ()) or ()),
            flat_form=d.get("flat_form"),
            bound_masters=tuple(tuple(b) for b in (d.get("bound_masters", ()) or ())),
        )


@dataclass
class SessionResult:
    """Transport-neutral ``open_session`` outcome envelope.

    ``state`` is ``UNAVAILABLE`` when the provider cannot be used, ``CANCELLED``
    when the operation was cancelled, ``FAILED`` on an operational error, and
    ``COMPLETED`` on success (``session`` may still be ``None`` when zero masters
    were admissible — that is an informative result, not an error).

    ``context_preparations`` exposes the §18 audit counter (see also
    :attr:`CalibrationSession.context_preparation_count`): one preparation per
    distinct calibrated plan, so a consumer can verify the masters are not
    reloaded per light.
    """

    state: CalibrationState = CalibrationState.UNAVAILABLE
    session: "CalibrationSession | None" = None
    fingerprint: str = ""
    admissions: tuple[MasterAdmission, ...] = ()
    rejected: tuple[RejectionDiagnostic, ...] = ()
    counts_by_role: Mapping[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error: CalibrationError | None = None
    sensor_orientation: str | None = None  # neutral fallback-only declaration ("identity" | None)

    @property
    def context_preparations(self) -> int:
        """Number of master-context preparations so far (S-a / §18 audit)."""
        if self.session is None:
            return 0
        return getattr(self.session, "context_preparation_count", 0)


@dataclass
class RouteResolution:
    """Transport-neutral auto-route decision for one light."""

    state: CalibrationState = CalibrationState.UNAVAILABLE
    outcome: str | None = None  # MATCHED / NO_MATCH / AMBIGUOUS (when COMPLETED)
    plan: CalibrationPlan | None = None
    composition: CalibrationComposition | None = None
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error: CalibrationError | None = None


@dataclass
class CalibrationResult:
    """Transport-neutral calibration result for one light (in-memory).

    ``data`` is a float32 physical-ADU array; ``mask`` is a uint16 DQ mask with
    ``mask != 0`` meaning "invalid sample" (DQ is primary, never short-circuited
    by a luminance mask).  ``provenance`` is a provider-agnostic mapping.
    """

    state: CalibrationState = CalibrationState.UNAVAILABLE
    data: Any = None
    mask: Any = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error: CalibrationError | None = None


class CancellationHandle(Protocol):
    """Minimal cooperative-cancellation handle (duck-typed)."""

    def is_cancelled(self) -> bool: ...


class CalibrationSession(Protocol):
    """An open calibration session over an admitted master library.

    Contract: **mono-thread** — one session per worker, never shared, never
    re-entrant.  ``resolve_light`` auto-routes a light (the scientific route is
    decided by the provider); ``calibrate`` calibrates in-memory and reuses the
    prepared masters across lights of the same session; ``close`` is idempotent.
    """

    fingerprint: str
    context_preparation_count: int  # master-context preparations so far (S-a/§18)

    def resolve_light(
        self, source: LightSource, *, cancel: CancellationHandle | None = None
    ) -> RouteResolution: ...

    def calibrate(
        self,
        source: LightSource,
        plan: CalibrationPlan,
        *,
        cancel: CancellationHandle | None = None,
    ) -> CalibrationResult: ...

    def close(self) -> None: ...


class CalibrationProvider(Protocol):
    """Minimal protocol implemented by every concrete calibration adapter.

    ``probe`` never raises (returns a :class:`ProviderInfo`); ``open_session``
    returns a :class:`SessionResult` and never raises for expected operational
    failures (absence/incompatibility/errors are mapped to neutral states).
    """

    name: str

    def probe(self) -> ProviderInfo: ...

    def open_session(
        self,
        root: str,
        *,
        cancel: CancellationHandle | None = None,
        sensor_orientation: str | None = None,
    ) -> SessionResult: ...


__all__ = [
    "CalibrationComposition",
    "CalibrationError",
    "CalibrationPlan",
    "CalibrationProvider",
    "CalibrationResult",
    "CalibrationSession",
    "CalibrationState",
    "CancellationHandle",
    "ErrorKind",
    "LightSource",
    "MasterAdmission",
    "ProviderInfo",
    "ProviderState",
    "RejectionDiagnostic",
    "RouteResolution",
    "SessionResult",
]
