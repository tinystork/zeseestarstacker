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

from seestar.calibration.preflight import light_signature
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
        """True when the light has a frozen plan (or when there is no plan map)."""
        if self._plan_map is None:
            return True
        return light_signature(file_path) in self._plan_map

    # ---------------------------------------------------------------- per frame
    def resolve(self, file_path: str):
        """Resolve a light's plan (None when not MATCHED or not planned)."""
        if self._session is None:
            return None
        if not self.is_planned(file_path):
            return None
        rr = self._session.resolve_light(LightSource(path=file_path))
        if getattr(rr, "state", None) is not CalibrationState.COMPLETED:
            return None
        return getattr(rr, "plan", None)

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

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
            self._session_result = None


__all__ = ["CalibrationIntegrator"]
