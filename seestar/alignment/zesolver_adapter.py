"""Optional ZeSolver API v1 integration adapter (public-import-only contract).

This module is the *only* place in Zsss that talks to ZeSolver, and it does so
strictly through the public, stable API surface ``zesolver.api.v1``.  It never
imports ZeSolver internals or private modules, never searches a sibling
ZeSolver checkout, and never mutates the module search path.

ZeSolver is optional: absence, incompatibility or an unhealthy installation must
never break importing Zsss or running the existing solvers.  The public API
module is therefore imported *lazily* (only inside ``discover_zesolver()`` /
``_import_api()``); there is no top-level import of the ``zesolver`` package.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
from pathlib import Path
from typing import Any

from .solver_port import (
    DiscoveryState,
    REQUIRED_CAPABILITIES,
    SOLVE_BACKEND_CAPABILITIES,
    SUPPORTED_API_MAJOR,
    SolverDiscovery,
    SolverOutcome,
    SolveStatus,
)

logger = logging.getLogger("seestar.alignment.zesolver_adapter")

# Referenced only as a string; never imported at module level.
_ZESOLVER_API_MODULE = "zesolver.api.v1"

# Zsss settings keys consumed by the adapter (see _build_request).
#   use_radec_hints         -> bool  : feed RA/Dec hints (from the FITS header)
#   scale_est_arcsec_per_pix-> float : pixel scale hint (arcsec/pixel)
#   astap_timeout_sec       -> number: solve timeout (single source)
#   zesolver_backend_policy -> str   : auto / near_only / blind_only (per-solve)
_DEFAULT_SOLVE_TIMEOUT_S = 120.0


def _basename(path: Any) -> str:
    try:
        return os.path.basename(os.path.normpath(str(path)))
    except Exception:  # pragma: no cover - defensive
        return str(path)


def _parse_major(version: Any) -> int | None:
    """Parse the leading integer of an API version string ("1.0" -> 1)."""
    if version is None:
        return None
    try:
        return int(str(version).split(".")[0])
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _is_zesolver_module_absent(exc: BaseException) -> bool:
    """True when ``exc`` means the public ZeSolver module itself is absent.

    A :class:`ModuleNotFoundError` naming the public ``zesolver`` /
    ``zesolver.api`` / ``zesolver.api.v1`` chain means "not installed".  One
    naming any *other* module means the public module was found but one of its
    internal imports failed (i.e. "installed but broken").
    """
    name = getattr(exc, "name", None)
    if not isinstance(name, str) or not name:
        return False
    return name == "zesolver" or name.startswith("zesolver.api")


def discover_zesolver() -> SolverDiscovery:
    """Lazily discover and health-check the installed ZeSolver public API v1.

    Compatibility is decided exclusively on the public ``API_MAJOR`` (parsed
    from ``API_VERSION`` when absent) plus the declared/negotiated capabilities
    — never on Git branch or product version.  The health check is deliberately
    cheap: ``probe`` is called without a catalog scan, without GPU/CuPy import
    and without network access.

    States: ``NOT_INSTALLED`` (public module absent), ``UNHEALTHY`` (present but
    broken import/probe), ``INCOMPATIBLE`` (wrong major, missing required
    capability, or no solve backend), ``AVAILABLE`` otherwise.
    """
    try:
        v1 = importlib.import_module(_ZESOLVER_API_MODULE)
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via fakes
        if _is_zesolver_module_absent(exc):
            return SolverDiscovery(
                state=DiscoveryState.NOT_INSTALLED,
                message=f"{type(exc).__name__}: {exc}",
            )
        # Public module found but one of its internal imports failed.
        return SolverDiscovery(
            state=DiscoveryState.UNHEALTHY,
            message=f"import failed: {type(exc).__name__}: {exc}",
        )
    except Exception as exc:  # pragma: no cover - exercised via fakes
        return SolverDiscovery(
            state=DiscoveryState.UNHEALTHY,
            message=f"import failed: {type(exc).__name__}: {exc}",
        )

    api_version = getattr(v1, "API_VERSION", None)
    api_major = getattr(v1, "API_MAJOR", None)
    if api_major is None:
        api_major = _parse_major(api_version)

    product_version = None
    declared_capabilities: tuple[str, ...] = ()
    try:
        info_fn = getattr(v1, "get_api_info", None)
        if info_fn is not None:
            info = info_fn()
            product_version = getattr(info, "product_version", None)
            declared = getattr(info, "supported_capabilities", None)
            if isinstance(declared, (tuple, list)):
                declared_capabilities = tuple(str(c) for c in declared)
    except Exception:  # pragma: no cover - defensive
        product_version = None

    if api_major != SUPPORTED_API_MAJOR:
        return SolverDiscovery(
            state=DiscoveryState.INCOMPATIBLE,
            api_version=str(api_version),
            product_version=product_version,
            message=(
                f"unsupported ZeSolver API major version {api_major!r} "
                f"(expected {SUPPORTED_API_MAJOR})"
            ),
        )

    declared_set = set(declared_capabilities)
    missing_required = [c for c in REQUIRED_CAPABILITIES if c not in declared_set]
    if missing_required:
        return SolverDiscovery(
            state=DiscoveryState.INCOMPATIBLE,
            api_version=str(api_version),
            product_version=product_version,
            message=(
                "ZeSolver is missing required capability(ies) "
                f"{', '.join(missing_required)} (declared: "
                f"{', '.join(declared_capabilities) or 'none'})"
            ),
        )

    # "solve" challenge: at least one solve backend must be declared.
    if not (declared_set & set(SOLVE_BACKEND_CAPABILITIES)):
        return SolverDiscovery(
            state=DiscoveryState.INCOMPATIBLE,
            api_version=str(api_version),
            product_version=product_version,
            message=(
                "ZeSolver declares no solve backend (expected at least one of "
                f"{', '.join(SOLVE_BACKEND_CAPABILITIES)})"
            ),
        )

    # Negotiated availability: required capabilities must not be reported
    # UNAVAILABLE (NOT_CHECKED is tolerated — the cheap probe does not scan).
    probe_fn = getattr(v1, "probe", None)
    if probe_fn is not None:
        try:
            probe_result = probe_fn(check_catalogs=False, check_gpu=False)
        except Exception as exc:  # pragma: no cover - exercised via fakes
            return SolverDiscovery(
                state=DiscoveryState.UNHEALTHY,
                api_version=str(api_version),
                product_version=product_version,
                message=f"probe failed: {type(exc).__name__}: {exc}",
            )
        negotiated: dict[str, Any] = {}
        for cap_state in getattr(probe_result, "capabilities", ()) or ():
            cap_id = getattr(cap_state, "id", None)
            availability = getattr(cap_state, "availability", None)
            if cap_id is not None:
                negotiated[str(cap_id)] = getattr(availability, "value", availability)
        for cap in REQUIRED_CAPABILITIES:
            if negotiated.get(cap) == "unavailable":
                return SolverDiscovery(
                    state=DiscoveryState.UNHEALTHY,
                    api_version=str(api_version),
                    product_version=product_version,
                    message=f"required capability {cap!r} reported unavailable",
                )

    return SolverDiscovery(
        state=DiscoveryState.AVAILABLE,
        api_version=str(api_version),
        product_version=product_version,
    )


class ZeSolverAdapter:
    """Plate solving through the public ``zesolver.api.v1`` API.

    Lifecycle: one :class:`SolverRuntime` per adapter instance (lazily created on
    first solve, guarded by a lock) and one :class:`SolverSession` per worker
    thread (``threading.local``).  :meth:`close` tears the runtime and every
    tracked session down; it is idempotent and the adapter stays usable (the next
    :meth:`solve` transparently recreates a fresh runtime/session).

    Discovery is performed lazily at the first solve; a non-``AVAILABLE``
    discovery yields an ``UNAVAILABLE`` outcome (``failure_code`` = discovery
    state) instead of raising.
    """

    name = "zesolver"

    def __init__(
        self,
        *,
        resources_path: str | Path | None = None,
        gpu_policy: str | None = None,
        network_policy: str | None = None,
    ) -> None:
        self._resources_path = resources_path
        self._gpu_policy = gpu_policy
        self._network_policy = network_policy
        self._runtime = None
        self._runtime_lock = threading.Lock()
        self._sessions = threading.local()
        self._session_registry: set = set()
        self._session_registry_lock = threading.Lock()
        self._active_tokens: set = set()
        self._active_tokens_lock = threading.Lock()
        self._discovery: SolverDiscovery | None = None
        self._discovery_lock = threading.Lock()

    # -- public ------------------------------------------------------------

    def solve(
        self,
        *,
        image_fits_path: str,
        fits_header,
        settings,
        progress_callback=None,
        log=None,
        **kwargs,
    ) -> SolverOutcome:
        """Solve one file and return a :class:`SolverOutcome`.

        Expected operational failures (unavailable backend, failed solve, bad
        WCS conversion) are returned as ``UNAVAILABLE``/``FAILED`` outcomes;
        unexpected public API errors are logged and returned as a ``FAILED``
        outcome — this method never raises to its caller.
        """
        filename = _basename(image_fits_path)
        try:
            discovery = self._ensure_discovery()
            if discovery.state is not DiscoveryState.AVAILABLE:
                return SolverOutcome(
                    status=SolveStatus.UNAVAILABLE,
                    failure_code=discovery.state.value,
                    message=discovery.message,
                    backend_used=self.name,
                )

            v1 = self._import_api()
            request = self._build_request(
                v1, image_fits_path, fits_header, settings or {}
            )
            cancellation = v1.CancellationToken()
            progress = self._make_progress_forwarder(progress_callback)
            self._register_active_token(cancellation)
            try:
                result = self._session().solve(
                    request, cancellation=cancellation, progress=progress
                )
            finally:
                self._unregister_active_token(cancellation)
            return self._convert_result(v1, result, fits_header, filename)
        except Exception as exc:  # noqa: BLE001 - fail this file, never the caller
            if log is not None:
                try:
                    log(f"ZeSolver solve failed for {filename}: {exc}")
                except Exception:  # pragma: no cover - logging must never raise
                    pass
            log_message = f"{type(exc).__name__}: {exc}"
            logger.error("ZeSolver solve failed for '%s': %s", filename, log_message)
            return SolverOutcome(
                status=SolveStatus.FAILED,
                failure_code="unexpected_error",
                message=log_message,
                backend_used=self.name,
            )

    def close(self) -> None:
        """Close the runtime and every tracked thread-local session.

        Idempotent and safe to call from any thread: closing twice (or closing
        an adapter whose runtime was never created) is a no-op.  After close the
        adapter remains usable.
        """
        try:
            self._sessions.session = None
        except Exception:  # pragma: no cover - defensive
            pass

        self.cancel_active_solve()
        with self._active_tokens_lock:
            self._active_tokens.clear()

        sessions_to_close = []
        with self._session_registry_lock:
            sessions_to_close = list(self._session_registry)
            self._session_registry.clear()

        for session in sessions_to_close:
            try:
                close_session = getattr(session, "close", None)
                if callable(close_session):
                    close_session()
            except Exception:  # pragma: no cover - close must never raise
                logger.debug("ZeSolver session close failed", exc_info=True)

        runtime = None
        with self._runtime_lock:
            runtime = self._runtime
            self._runtime = None
        if runtime is not None:
            try:
                close_runtime = getattr(runtime, "close", None)
                if callable(close_runtime):
                    close_runtime()
            except Exception:  # pragma: no cover - close must never raise
                logger.debug("ZeSolver runtime close failed", exc_info=True)

    def cancel_active_solve(self) -> None:
        """Cooperatively cancel any in-flight ZeSolver solve(s) (best-effort).

        Thread-safe and idempotent: it cancels every active token tracked by
        this adapter (one per worker thread with an in-flight solve) and never
        raises.  Cancellation is cooperative — it only has an effect when the
        backend declares the optional ``cancel`` capability.
        """
        with self._active_tokens_lock:
            tokens = list(self._active_tokens)
        for token in tokens:
            try:
                cancel = getattr(token, "cancel", None)
                if callable(cancel):
                    cancel()
            except Exception:  # pragma: no cover - cancel must never raise
                logger.debug("ZeSolver token cancel failed", exc_info=True)

    # -- internal ----------------------------------------------------------

    def _import_api(self):
        return importlib.import_module(_ZESOLVER_API_MODULE)

    def _ensure_discovery(self) -> SolverDiscovery:
        if self._discovery is None:
            with self._discovery_lock:
                if self._discovery is None:
                    self._discovery = discover_zesolver()
        return self._discovery

    def _register_active_token(self, token) -> None:
        with self._active_tokens_lock:
            self._active_tokens.add(token)

    def _unregister_active_token(self, token) -> None:
        with self._active_tokens_lock:
            self._active_tokens.discard(token)

    def _ensure_runtime(self):
        if self._runtime is not None:
            return self._runtime
        with self._runtime_lock:
            if self._runtime is None:
                v1 = self._import_api()
                create = v1.create_solver_runtime
                kwargs: dict[str, Any] = {}
                if self._resources_path is not None:
                    kwargs["resources_path"] = Path(self._resources_path)
                if self._gpu_policy is not None:
                    kwargs["gpu_policy"] = self._resolve_gpu_policy(
                        self._gpu_policy, v1
                    )
                if self._network_policy is not None:
                    kwargs["network_policy"] = self._resolve_network_policy(
                        self._network_policy, v1
                    )
                self._runtime = create(**kwargs)
        return self._runtime

    def _session(self):
        session = getattr(self._sessions, "session", None)
        if session is None:
            session = self._ensure_runtime().create_session()
            self._sessions.session = session
            with self._session_registry_lock:
                self._session_registry.add(session)
        return session

    def _make_progress_forwarder(self, progress_callback):
        """Forward ZeSolver ``ProgressEvent`` objects to the Zsss progress
        callback (signature ``(message, progress)``), best-effort."""
        if progress_callback is None:
            return None

        def forward(event) -> None:
            phase = getattr(event, "phase", None)
            phase_value = getattr(phase, "value", None)
            if phase_value is None:
                phase_value = str(phase)
            message = getattr(event, "message", None)
            text = f"ZeSolver: {phase_value}"
            if message:
                text += f" - {message}"
            try:
                progress_callback(text, None)
            except Exception:  # pragma: no cover - progress must never raise
                pass

        return forward

    @staticmethod
    def _select_least_destructive_write_policy(v1):
        """Select the least destructive ``WritePolicy`` member exposed by v1.

        ZeSolver must never modify the input file: the adapter only consumes the
        returned canonical WCS header and lets the Zsss caller write it back.
        We therefore prefer a no-write policy when the installed API exposes one;
        ``OVERWRITE_INPUT`` is used only as a last-resort fallback when no such
        member exists.
        """
        write_policy = getattr(v1, "WritePolicy", None)
        if write_policy is None:
            return None
        preferred = (
            "WRITE_NONE",
            "NO_WRITE",
            "NEVER_WRITE",
            "WRITE_NEVER",
            "READ_ONLY",
            "NONE",
            "DISABLED",
        )
        for name in preferred:
            member = getattr(write_policy, name, None)
            if member is not None:
                return member
        return getattr(write_policy, "OVERWRITE_INPUT", None)

    def _build_request(self, v1, image_fits_path: str, fits_header, settings: dict):
        hints_kwargs: dict[str, Any] = {}

        # RA/Dec hints: enabled by settings and read from the FITS header
        # (RA/DEC, falling back to CRVAL1/CRVAL2), mirroring the local solver.
        if settings.get("use_radec_hints"):
            ra, dec = self._extract_radec_hints(fits_header)
            if isinstance(ra, (int, float)) and isinstance(dec, (int, float)):
                hints_kwargs["ra_deg"] = float(ra)
                hints_kwargs["dec_deg"] = float(dec)

        # Pixel scale hint.
        scale = settings.get("scale_est_arcsec_per_pix")
        if scale is not None:
            try:
                hints_kwargs["pixel_scale_arcsec"] = float(scale)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass

        hints = v1.SolveHints(**hints_kwargs)

        options_kwargs: dict[str, Any] = {
            # API v1 is local-only: network always disabled.
            "network_policy": v1.NetworkPolicy.DISABLED,
            # ZeSolver must never modify the input file: the adapter only
            # consumes the returned canonical WCS header and lets the Zsss
            # caller write it back.  We therefore pick the least destructive
            # write policy exposed by the installed API.
            "write_policy": self._select_least_destructive_write_policy(v1),
        }

        backend_policy = settings.get("zesolver_backend_policy")
        if backend_policy:
            try:
                options_kwargs["backend_policy"] = self._resolve_backend_policy(
                    backend_policy, v1
                )
            except Exception:  # pragma: no cover - defensive
                pass

        timeout_s = self._resolve_timeout(settings)
        if timeout_s is not None:
            options_kwargs["timeout_s"] = float(timeout_s)

        options = v1.SolveOptions(**options_kwargs)
        return v1.SolveRequest(
            input_path=Path(image_fits_path), hints=hints, options=options
        )

    def _convert_result(self, v1, result, fits_header, filename) -> SolverOutcome:
        status = getattr(result, "status", None)
        status_value = getattr(status, "value", status)

        if status_value == "solved":
            if getattr(result, "wcs_header", None) is None:
                return SolverOutcome(
                    status=SolveStatus.FAILED,
                    failure_code="missing_wcs_header",
                    message="solve succeeded but no canonical WCS header returned",
                    backend_used=self.name,
                )
            try:
                header = self._canonical_to_header(result.wcs_header)
                wcs = self._header_to_wcs(header)
            except Exception as exc:
                return SolverOutcome(
                    status=SolveStatus.FAILED,
                    failure_code="wcs_conversion_failed",
                    message=f"{type(exc).__name__}: {exc}",
                    backend_used=self.name,
                )
            merged_header = self._merge_header(fits_header, header)
            return SolverOutcome(
                status=SolveStatus.SOLVED,
                wcs=wcs,
                header=merged_header,
                # The solution originates from ZeSolver and a canonical header
                # is available -> the caller must write it back.
                should_write_header_back=True,
                backend_used=self.name,
            )

        if status_value == "cancelled":
            return SolverOutcome(
                status=SolveStatus.CANCELLED,
                failure_code="cancelled",
                message=getattr(result, "message", None),
                backend_used=self.name,
            )

        failure_code = getattr(
            getattr(result, "failure_code", None),
            "value",
            getattr(result, "failure_code", None),
        )

        # SKIPPED_EXISTING_WCS: the file already carries a WCS -> not a failure.
        if status_value == "skipped" or (
            failure_code is not None
            and "SKIPPED_EXISTING_WCS" in str(failure_code).upper()
        ):
            return self._skipped_outcome(result, fits_header)

        # FAILED / any unrecognised status -> FAILED.
        if failure_code is None:
            failure_code = f"solve_status_{status_value}" if status_value else "solve_failed"
        return SolverOutcome(
            status=SolveStatus.FAILED,
            failure_code=failure_code,
            message=getattr(result, "message", None),
            backend_used=self.name,
        )

    def _skipped_outcome(self, result, fits_header) -> SolverOutcome:
        """Build a ``SKIPPED`` outcome for ``SKIPPED_EXISTING_WCS``.

        If the result carries an existing canonical WCS header, convert it so
        the caller can reuse it; otherwise report the skip with a clear message.
        """
        existing = getattr(result, "wcs_header", None)
        if existing is not None:
            try:
                header = self._canonical_to_header(existing)
                wcs = self._header_to_wcs(header)
            except Exception as exc:
                return SolverOutcome(
                    status=SolveStatus.SKIPPED,
                    failure_code="skipped_existing_wcs",
                    message=f"file already has WCS (conversion failed: {exc})",
                    backend_used=self.name,
                )
            return SolverOutcome(
                status=SolveStatus.SKIPPED,
                wcs=wcs,
                header=self._merge_header(fits_header, header),
                should_write_header_back=False,
                message=getattr(result, "message", None),
                backend_used=self.name,
            )
        return SolverOutcome(
            status=SolveStatus.SKIPPED,
            failure_code="skipped_existing_wcs",
            message=getattr(result, "message", None) or "file already has WCS",
            backend_used=self.name,
        )

    @staticmethod
    def _canonical_to_header(canonical):
        """Convert a public ``CanonicalWcsHeader`` into an Astropy ``Header``."""
        from astropy.io import fits

        cards = getattr(canonical, "cards", None)
        if cards is None:
            raise ValueError("CanonicalWcsHeader has no 'cards' attribute")
        cards_str = "\n".join(str(c) for c in cards)
        return fits.Header.fromstring(cards_str, sep="\n")

    @staticmethod
    def _header_to_wcs(header):
        from astropy.wcs import WCS

        return WCS(header)

    @staticmethod
    def _merge_header(fits_header, canonical_header):
        """Merge the canonical WCS cards into the original header (best effort).

        Always returns an Astropy ``Header`` (the canonical cards, plus any
        original non-WCS keys when the original header is usable).
        """
        if fits_header is None:
            return canonical_header
        try:
            from astropy.io import fits

            merged = canonical_header.copy()
            if isinstance(fits_header, fits.Header):
                for key in fits_header:
                    if key not in merged:
                        merged[key] = fits_header[key]
            elif hasattr(fits_header, "items"):
                for key, value in fits_header.items():
                    if key not in merged:
                        merged[key] = value
            return merged
        except Exception:  # pragma: no cover - defensive
            return canonical_header

    @staticmethod
    def _extract_radec_hints(fits_header):
        if fits_header is None:
            return None, None
        try:
            ra = fits_header.get("RA", fits_header.get("CRVAL1"))
            dec = fits_header.get("DEC", fits_header.get("CRVAL2"))
        except Exception:  # pragma: no cover - defensive
            return None, None
        return ra, dec

    @staticmethod
    def _resolve_timeout(settings: dict) -> float:
        """Resolve the solve timeout (seconds).

        Only ``astap_timeout_sec`` is consulted (ZeSolver reuses the ASTAP
        timeout as the single source); default :data:`_DEFAULT_SOLVE_TIMEOUT_S`.
        """
        value = settings.get("astap_timeout_sec")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass
        return _DEFAULT_SOLVE_TIMEOUT_S

    @staticmethod
    def _resolve_gpu_policy(value, v1):
        text = str(value).strip().lower()
        mapping = {
            "auto": v1.GpuPolicy.AUTO,
            "disabled": v1.GpuPolicy.DISABLED,
            "required": v1.GpuPolicy.REQUIRED,
        }
        return mapping[text]

    @staticmethod
    def _resolve_backend_policy(value, v1):
        text = str(value).strip().lower()
        mapping = {
            "auto": v1.BackendPolicy.AUTO,
            "near_only": v1.BackendPolicy.NEAR_ONLY,
            "blind_only": v1.BackendPolicy.BLIND_ONLY,
        }
        return mapping[text]

    @staticmethod
    def _resolve_network_policy(value, v1):
        text = str(value).strip().lower()
        # API v1 is strictly local-only: NetworkPolicy exposes only DISABLED.
        if text != "disabled":
            raise ValueError(
                f"network_policy must be 'disabled' (API v1 is local-only), got {value!r}"
            )
        return v1.NetworkPolicy.DISABLED
