"""Internal SolverPort boundary for plate solving in ZeSeestarStacker (Zsss).

This module defines the *internal*, transport-neutral contract between the Zsss
pipeline and any concrete plate solver.  It is deliberately decoupled from a
specific solver so that:

* the existing local solvers (ASTAP / ANSVR / Astrometry.net) keep working
  through :mod:`seestar.alignment.astrometry_solver`, and
* the optional ZeSolver integration (see :mod:`seestar.alignment.zesolver_adapter`)
  can be layered on top without touching the scientific processing semantics.

Consumed contract (ZeSolver public API v1, ``zesolver.api.v1``)
---------------------------------------------------------------

This boundary is written against the *public, stable* ZeSolver API surface
(the ``zesolver.api.v1`` module): ``API_VERSION``, ``API_MAJOR``, ``get_api_info``,
``probe``, ``create_solver_runtime``, ``SolverRuntime``, ``SolverSession``,
``SolveRequest``, ``SolveHints``, ``SolveOptions``, ``SolveResult``,
``CanonicalWcsHeader``, ``BackendPolicy``, ``GpuPolicy``, ``NetworkPolicy``,
``WritePolicy``, ``SolveStatus``, ``FailureCode``, ``CancellationToken``,
``ProgressEvent``, ``ZeSolverApiError``, ``SolverClosedError``,
``InvalidRequestError``.

Relevant contract points consumed here:

* ``get_api_info()`` exposes ``supported_capabilities`` (the *static, declared*
  capability IDs) and ``product_version``.
* ``probe(check_catalogs=False, check_gpu=False)`` exposes *negotiated* per
  capability availability (``AVAILABLE`` / ``UNAVAILABLE`` / ``NOT_CHECKED``).
* Capability IDs declared by v1: ``near_solve``, ``blind_solve``, ``wcs_write``,
  ``gpu``, ``cancel``.  There is no literal ``"solve"`` ID — "can solve at all"
  is expressed as "at least one solve backend (``near_solve``/``blind_solve``)".
* API 1.x is **local-only**: ``network_policy`` must remain ``DISABLED``.

Design rules (ZeSoftware integration guidelines)
------------------------------------------------

* **Optionality.**  ZeSolver is an optional dependency.  Its absence,
  incompatibility or an unhealthy installation must never break importing Zsss
  nor change the behaviour of the existing solvers.  Discovery is always lazy
  (never at import time).
* **Public API only.**  Only ``zesolver.api.v1`` is ever touched; internal /
  private ZeSolver modules are never imported and ``sys.path`` is never mutated.
* **No network.**  Solving never performs a network query.

This module imports **only the standard library**.  In particular it must never
import ``zesolver`` (or any of its submodules) at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class SolveStatus(str, Enum):
    """Outcome status of a single solve attempt (internal, not ZeSolver's).

    These are the *Zsss* statuses; the adapter is responsible for mapping the
    ZeSolver public ``SolveStatus`` onto them.
    """

    SOLVED = "solved"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"


@dataclass
class SolverOutcome:
    """Transport-neutral result returned by every :class:`SolverAdapter`.

    ``wcs``, when not ``None``, is an Astropy :class:`~astropy.wcs.WCS` — this is
    the object the Zsss pipeline expects in return from a solve.
    ``header``, when not ``None``, is an Astropy :class:`~astropy.io.fits.Header`
    carrying the canonical WCS cards to write back downstream (used by the
    ZeSolver adapter, which receives canonical cards instead of an in-place
    header update).
    ``should_write_header_back`` tells the caller whether it must write the WCS
    solution back into the FITS file itself.
    """

    status: SolveStatus = SolveStatus.UNAVAILABLE
    wcs: Any = None
    header: Any = None
    should_write_header_back: bool = False
    failure_code: str | None = None
    message: str | None = None
    backend_used: str | None = None


class DiscoveryState(str, Enum):
    """Lazy-discovery state for the optional ZeSolver integration."""

    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed"
    INCOMPATIBLE = "incompatible"
    UNHEALTHY = "unhealthy"


@dataclass
class SolverDiscovery:
    """Result of a lazy ZeSolver discovery / health check."""

    state: DiscoveryState
    api_version: str | None = None
    product_version: str | None = None
    message: str | None = None


# ---------------------------------------------------------------------------
# ZeSolver capability negotiation (public v1 contract).
#
# ZeSolver exposes two capability signals through its public v1 API:
#   * ``get_api_info().supported_capabilities`` — the *static, declared* set of
#     capability IDs the installed API supports (cheap, no I/O); and
#   * ``probe(...).capabilities`` — the *negotiated* per-capability availability
#     (AVAILABLE / UNAVAILABLE / NOT_CHECKED).
#
# The adapter challenges only these public IDs.  There is no literal ``"solve"``
# ID — "can solve at all" is expressed as "at least one solve backend declared".
# ---------------------------------------------------------------------------

# Hard requirement for the v1 adapter path: ZeSolver must be able to return a
# canonical WCS header (``wcs_write``), otherwise the adapter cannot consume a
# solved result and the backend is reported unavailable/incompatible.
REQUIRED_CAPABILITIES: tuple[str, ...] = ("wcs_write",)

# At least one of these solve-backend IDs must be declared.  Their *runtime*
# availability (catalog presence) is intentionally NOT a discovery blocker: the
# cheap probe reports them NOT_CHECKED and real availability is negotiated
# lazily at solve time.
SOLVE_BACKEND_CAPABILITIES: tuple[str, ...] = ("near_solve", "blind_solve")

# Optional capabilities: their absence never blocks the adapter.  ``cancel``
# enables cooperative cancellation (best-effort); ``gpu`` enables GPU-accelerated
# solving (policy-dependent).
OPTIONAL_CAPABILITIES: tuple[str, ...] = ("cancel", "gpu")

# The only supported ZeSolver public API major version (API v1).
SUPPORTED_API_MAJOR: int = 1


class SolverAdapter(Protocol):
    """Minimal protocol implemented by every concrete solver adapter.

    The Zsss pipeline calls ``solve(**kwargs)`` and always receives a
    :class:`SolverOutcome` — adapters never raise for expected operational
    failures.
    """

    name: str

    def solve(self, **kwargs) -> SolverOutcome: ...
