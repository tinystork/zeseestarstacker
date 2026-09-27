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

from seestar.calibration.preflight import acquisition_signature
from seestar.calibration.zecalibrator_adapter import ZeCalibratorProvider
from seestar.core.calibration_port import CalibrationState, LightSource


class CalibrationIntegrator:
    """Holds an open session and calibrates one light at a time.

    ``plan_map`` is the frozen C4 ``calibration_plan_map`` (light content
    signature -> ``{plan_id, composition}``).  When provided, only lights whose
    signature is present are calibrated; everything else falls back to the
    historical path.
    """

    def __init__(self, masters_folder: str, *, provider=None, plan_map=None) -> None:
        self._provider = provider or ZeCalibratorProvider()
        self._masters_folder = masters_folder
        self._plan_map = dict(plan_map) if plan_map else None
        self._plan_cache: dict = {}  # acquisition signature -> resolved plan object
        self._session = None
        self._session_result = None

    # ------------------------------------------------------------------ open
    def open(self) -> bool:
        """Open the session (admission only). Returns True when usable."""
        result = self._provider.open_session(self._masters_folder)
        self._session_result = result
        self._session = getattr(result, "session", None)
        return self._session is not None

    @property
    def session(self):
        return self._session

    @property
    def fingerprint(self) -> str:
        return getattr(self._session_result, "fingerprint", "") or ""

    @property
    def context_preparations(self) -> int:
        """Master-context preparations so far (§18 audit counter)."""
        if self._session is None:
            return 0
        return getattr(self._session, "context_preparation_count", 0)

    def is_planned(self, file_path: str) -> bool:
        """True when the light's acquisition class has a frozen plan.

        Keyed by the **acquisition signature** (header-only), not a content hash:
        the plan is per-class, so a light is planned when its acquisition class
        was resolved at preflight (or when there is no plan map at all).
        """
        if self._plan_map is None:
            return True
        return acquisition_signature(file_path) in self._plan_map

    # ---------------------------------------------------------------- per frame
    def resolve(self, file_path: str):
        """Resolve a light's plan (cached by acquisition class — no re-decode).

        The plan is looked up by the light's **acquisition signature** (header-
        only read).  A class already resolved at preflight returns the cached
        plan object with no ``resolve_light`` decode; a new class (not in the
        frozen map) falls back to a direct resolve.  Returns ``None`` when not
        MATCHED / not planned.
        """
        if self._session is None:
            return None
        acq_sig = acquisition_signature(file_path)
        cached = self._plan_cache.get(acq_sig)
        if cached is not None:
            return cached
        if not self.is_planned(file_path):
            return None
        plan, _composition = self._resolve_direct(file_path)
        if plan is not None:
            self._plan_cache[acq_sig] = plan
        return plan

    def calibrate(self, file_path: str, plan) -> Optional[Tuple]:
        """Calibrate one light -> ``(physical_float32, mask)`` or ``None``.

        ``mask`` is the provider DQ mask (``mask != 0`` == invalid) — carried and
        returned for C6, never silently dropped.  A cancellation in flight
        propagates (the port already maps it) and is not swallowed here.
        """
        if self._session is None or plan is None:
            return None
        result = self._session.calibrate(LightSource(path=file_path), plan)
        if getattr(result, "state", None) is not CalibrationState.COMPLETED:
            return None
        return getattr(result, "data", None), getattr(result, "mask", None)

    def freeze_snapshot(self, lights=()) -> dict:
        """Build the JSON-safe calibration freeze (7 fields) for the run contract.

        Provider id / api / product version come from ``probe()``; the library
        fingerprint from the open session.  The ``calibration_plan_map`` is keyed
        by the **acquisition signature** (header-only): lights are grouped by
        acquisition class and ONE representative per class is resolved (1 decode
        per class, never one per frame).  The resolved plan objects are cached
        in ``self._plan_cache`` for the streaming loop.  Returns ``{}`` when no
        session is open (== "calibration disabled").  Never carries a provider
        object.
        """
        if self._session is None:
            return {}
        info = self._provider.probe()
        plan_map = {}
        representatives: dict = {}
        for path in lights or ():
            representatives.setdefault(acquisition_signature(path), path)
        for acq_sig, rep in representatives.items():
            plan, composition = self._resolve_direct(rep)
            if plan is None or composition is None:
                continue
            self._plan_cache[acq_sig] = plan
            plan_map[acq_sig] = {
                "plan_id": plan.plan_id,
                "composition": composition.to_dict(),
            }
        return {
            "calibration_enabled": True,
            "calibration_provider": getattr(info, "provider_id", None) or "",
            "calibration_api_version": getattr(info, "api_version", None) or "",
            "calibration_product_version": getattr(info, "product_version", None) or "",
            "calibration_library_fingerprint": self.fingerprint or "",
            "calibration_contract_versions": {},
            "calibration_plan_map": plan_map,
        }

    def _resolve_direct(self, file_path: str):
        """Directly resolve a light (decode) -> ``(plan, composition)`` or ``(None, None)``."""
        if self._session is None:
            return None, None
        rr = self._session.resolve_light(LightSource(path=file_path))
        if getattr(rr, "state", None) is not CalibrationState.COMPLETED:
            return None, None
        return getattr(rr, "plan", None), getattr(rr, "composition", None)

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
            self._session_result = None


__all__ = ["CalibrationIntegrator"]
