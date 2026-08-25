"""Passive registration diagnostics (RF2).

Lightweight, observational, **fail-open** structured logging of per-frame
registration for normal stacking runs.  The only output is a versioned
JSON-Lines file in the output folder; nothing here ever feeds back into
alignment, stacking, or any scientific result.

Design contract (RF2):

* **Passive** — records are written only; they are never read by the stacker.
* **Observational** — ``raw_scale`` and ``residual_px`` are diagnostic-only
  quantities (the raw astroalign similarity scale *before* it is discarded by
  the Euclidean conversion, and the residual of the returned match pairs under
  the *applied* Euclidean matrix).  They are explicitly not science inputs.
* **Fail-open** — any serialization / I/O error is caught and returns ``False``;
  it must never raise and never affect the alignment result.
* **Privacy-safe** — only basenames / provenance identifiers are recorded, never
  full source paths.
* **Append-safe / session-scoped** — records are appended to a single file
  ``registration_diagnostics.jsonl`` in the output folder; each record carries a
  ``session_id`` so records from different runs cannot be conflated.

The schema is versioned (``schema_version``); consumers must key on that field.
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
DIAGNOSTICS_FILENAME = "registration_diagnostics.jsonl"

# Target policy: the initially-selected reference image is held immutable for
# the whole run (RF2 acceptance).  This is a descriptive label, not a setting.
TARGET_POLICY = "immutable_selected_reference"

# Selected/applied model: astroalign similarity fit whose scale is discarded
# (forced to 1.0) so only rotation + translation survive.
MODEL = "euclidean"

# Fields that are observational/diagnostic only and must never be treated as
# science inputs by any consumer.
DIAGNOSTIC_ONLY_FIELDS = ("raw_scale", "residual_px")


def build_record(
    *,
    frame,
    reference_provenance=None,
    target_policy=TARGET_POLICY,
    model=MODEL,
    success=False,
    raw_scale=None,
    applied_rotation_deg=None,
    applied_translation=None,
    match_count=None,
    residual_px=None,
    error=None,
    session_id=None,
    ts=None,
    schema_version=SCHEMA_VERSION,
):
    """Build a versioned, machine-parseable registration diagnostic record.

    ``residual_px``, when present, is a dict ``{"p50": ..., "p95": ...,
    "rms": ...}`` computed from the returned match pairs under the *applied*
    Euclidean matrix (see ``SeestarAligner._align_image``).
    """
    return {
        "schema_version": schema_version,
        "ts": ts if ts is not None else time.time(),
        "session_id": session_id,
        "event": "registration",
        "frame": frame,
        "target_policy": target_policy,
        "reference_provenance": reference_provenance,
        "model": model,
        "success": bool(success),
        "raw_scale": raw_scale,
        "applied_rotation_deg": applied_rotation_deg,
        "applied_translation": applied_translation,
        "match_count": match_count,
        "residual_px": residual_px,
        "error": error,
        "diagnostic_only": list(DIAGNOSTIC_ONLY_FIELDS),
    }


def append_record(path, record):
    """Append one JSON record to ``path``.  Fail-open: never raises.

    Returns ``True`` on success, ``False`` if the write could not be performed
    (the error is logged at debug level and swallowed — diagnostics I/O must
    never affect registration success or the scientific result).
    """
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        return True
    except Exception as exc:  # noqa: BLE001 — fail-open is the contract
        logger.debug("registration diagnostics write failed (non-fatal): %s", exc)
        return False
