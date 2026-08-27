"""Structured startup-refusal carrier (pure stdlib, no Qt/Tk/engine).

ZSSS-LIFECYCLE-01 boundary A.  The scientific engine sets a structured refusal
on itself when it refuses to start; the Qt backend adapter reads it (by
duck-typing, never importing the engine) and raises a distinct
:class:`StartupRefusedError` carrying this payload so the Qt shell can map the
stable code through its existing localization architecture — without parsing
progress/log strings.

A generic false start (engine returns ``False`` with no structured refusal)
remains distinguishable: it keeps the plain
``"start_processing() reported it did not start"`` runtime path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# Stable code for "the output folder already holds processing/resume state that
# this run cannot resume": any session that requested resume and was refused
# (non-plain mode without supported resume, missing/corrupt/legacy manifest,
# scientific fingerprint or dtype mismatch, incompatible reference shape,
# invalid quality reference scale, etc.).
CODE_OUTPUT_STATE_INCOMPATIBLE = "OUTPUT_STATE_INCOMPATIBLE"


@dataclass(frozen=True)
class StartupRefusalPayload:
    """Toolkit-free, immutable refusal payload carried adapter -> GUI thread.

    Fields
    ------
    code:
        Stable machine code (e.g. ``OUTPUT_STATE_INCOMPATIBLE``).
    technical_detail:
        Free-form technical detail from the engine (never parsed by the GUI).
    semantic_key:
        User-facing semantic key consumed by the localization layer
        (e.g. ``output_state_incompatible``).
    semantic_data:
        Optional bounded key/value data for the presentation layer
        (e.g. the selected mode).  Never used to decide behaviour.
    """

    code: str
    technical_detail: str = ""
    semantic_key: Optional[str] = None
    semantic_data: Dict[str, Any] = field(default_factory=dict)


class StartupRefusedError(RuntimeError):
    """Distinct structured startup-refusal exception (vs. generic false start).

    Raised by the backend adapter when the engine reports a structured refusal.
    The worker maps this to the ``refused`` signal instead of the generic
    ``failed`` signal so the Qt shell can present clear, localized EN/FR
    semantics for the known code.
    """

    def __init__(self, payload: StartupRefusalPayload) -> None:
        self.payload = payload
        super().__init__(payload.technical_detail or payload.code)


def build_payload_from_engine(refusal: Any) -> Optional[StartupRefusalPayload]:
    """Build a payload from an engine refusal object by duck-typing.

    The engine object exposes ``code`` / ``technical_detail`` / ``semantic_key``
    / ``semantic_data`` attributes; this reads them defensively so the adapter
    never imports the engine and never raises on a malformed carrier.  Returns
    ``None`` when the object carries no stable code.
    """
    if refusal is None:
        return None
    code = getattr(refusal, "code", None)
    if not code or not isinstance(code, str):
        return None
    technical_detail = getattr(refusal, "technical_detail", "") or ""
    semantic_key = getattr(refusal, "semantic_key", None) or None
    semantic_data = getattr(refusal, "semantic_data", None) or {}
    if not isinstance(semantic_data, dict):
        semantic_data = {}
    return StartupRefusalPayload(
        code=str(code),
        technical_detail=str(technical_detail),
        semantic_key=str(semantic_key) if semantic_key else None,
        semantic_data=dict(semantic_data),
    )
