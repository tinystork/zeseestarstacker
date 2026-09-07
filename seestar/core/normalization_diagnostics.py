"""Durable per-run normalization diagnostics (8.4.0 pre-W80, stage A).

Passive, observational, **fail-open** structured logging of every
support-aware plain-Classic normalization event (``sky_mean`` /
``linear_fit`` accepted or neutral) of a normal stacking run.  The only
output is a versioned JSON-Lines artifact in the output folder; nothing here
ever feeds back into normalization, stacking, or any scientific result.

Design contract (mirrors :mod:`seestar.core.registration_diagnostics`):

* **Passive** — records are written only; they are never read by the stacker.
* **Per-run isolated artifact** — the artifact filename is suffixed with a
  run/session token (``normalization_diagnostics_<token>.jsonl``), so
  repeated starts on the same stacker (including resume) append to a
  *distinct* file and never overwrite prior evidence.  Within one run every
  event appends to the same artifact.  ``append_record`` never truncates.
* **Allowlisted / scalar-bounded** — only bounded scalar fields are
  serialized; no image / mask / M / pixel arrays and no arbitrary nested
  diagnostic payload.  ``linear_a`` / ``linear_b`` are the only lists and are
  bounded to ``MAX_CHANNELS`` finite numbers.  Records are rejected (never
  partially written) when they violate the allowlist, contain non-finite
  values, or exceed ``MAX_RECORD_BYTES``.
* **Fail-open** — any serialization / validation / I/O error is caught and
  returns ``False``; it must never raise and never affect the normalization
  result or abort a run.
* **Privacy-safe** — only the original FITS basename is recorded (never full
  source paths); a frame whose identity evidence is absent is recorded as an
  explicit unknown (``frame=None``, ``frame_evidence="unknown"``) — never a
  synthetic index label and never a fabricated all-valid name.

Count / fraction semantics (denominators are explicit)
-------------------------------------------------------
* ``canvas_area`` = ``H * W`` of the reference canvas (the shared grid).
* ``geometric_support_pixel_count`` (``n_geometric``) = number of canvas
  pixels inside the eroded geometry-support mask (real source ``M``
  footprint).  Unknown (``None``) when the geometry was never computed
  (missing/degenerate ``M``, or content evidence missing before the mask
  could be derived) — never fabricated as all-valid.
* ``effective_support_pixel_count`` (``n_effective``) = full common count:
  geometry AND source content AND reference content AND finite on both
  sides, *unsampled*.
* ``sampled_estimator_count`` (``n_overlap``) = the deterministic bounded
  sample actually used by the estimator (<= ``n_effective``).
* ``geometric_support_fraction`` = ``n_geometric / canvas_area``.
* ``effective_support_fraction`` = ``n_effective / canvas_area``.
* ``overlap_pixel_count`` = ``n_effective`` (the true full overlap count).
* ``overlap_fraction`` = ``n_effective / n_geometric`` when both are known
  and ``n_geometric > 0`` (share of the geometric support that yields valid
  common overlap); ``None`` otherwise.

Every numeric field is JSON-safe and finite (JSON ``null`` encodes explicit
unknown / not-applicable).  The schema is versioned (``schema_version``);
consumers must key on that field.
"""

import json
import logging
import math
import os
import time

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
ARTIFACT_PREFIX = "normalization_diagnostics"
MAX_CHANNELS = 3  # Classic support: mono (1) or RGB (3) channel vectors.
MAX_RECORD_BYTES = 8192  # hard bound; includes any (short) frame basename

# Event kinds (status of one frame normalization attempt).
STATUS_ACCEPTED = "accepted"
STATUS_NEUTRAL = "neutral"

# How the recorded frame identity was established.
EVIDENCE_HEADER_SRCFILE = "header_srcfile"
EVIDENCE_UNKNOWN = "unknown"

# Top-level allowlist enforced by validate/append.
ALLOWED_KEYS = frozenset(
    {
        "schema_version",
        "ts",
        "session_id",
        "event",
        "frame",
        "frame_evidence",
        "normalization_method",
        "status",
        "reason",
        "neutral_reason",
        "estimator",
        "estimator_reason",
        "canvas_area",
        "geometric_support_pixel_count",
        "geometric_support_fraction",
        "effective_support_pixel_count",
        "effective_support_fraction",
        "overlap_pixel_count",
        "overlap_fraction",
        "sampled_estimator_count",
        "sky_offset",
        "linear_a",
        "linear_b",
    }
)
# Keys whose value must be ``None`` or a finite JSON number (int/float).
NUMERIC_KEYS = frozenset(
    {
        "canvas_area",
        "geometric_support_pixel_count",
        "geometric_support_fraction",
        "effective_support_pixel_count",
        "effective_support_fraction",
        "overlap_pixel_count",
        "overlap_fraction",
        "sampled_estimator_count",
        "sky_offset",
    }
)
# Keys whose value must be ``None`` or a bounded list of finite numbers.
CHANNEL_VECTOR_KEYS = ("linear_a", "linear_b")


def artifact_filename(session_id):
    """Per-run isolated artifact filename for ``session_id``.

    ``session_id`` must be a non-empty string of filesystem-safe characters
    (the queue manager uses ``time.time_ns()``-based run tokens).
    """
    session_id = str(session_id or "")
    if not session_id or any(c in session_id for c in "/\\\x00"):
        raise ValueError(f"invalid session_id for artifact filename: {session_id!r}")
    return f"{ARTIFACT_PREFIX}_{session_id}.jsonl"


def _json_number(value):
    """Return a JSON-safe finite number, or ``None`` for non-finite/None.

    Numpy scalars (``np.float32`` / ``np.float64`` / ``np.int64`` ...) are
    converted to plain Python int/float first, then checked for finiteness.
    Non-finite or non-numeric scalar values yield ``None`` (explicit unknown)
    so a single bad estimator output can never poison a whole record.

    Array-likes / lists / tuples / dicts are NOT scalar numbers: they are
    rejected with ``ValueError`` so arbitrary matrix payloads can never be
    smuggled through a numeric field (the caller turns that into a refused,
    fail-open record).
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (list, tuple, dict)):
        raise ValueError("non-scalar numeric field (list/tuple/dict)")
    # numpy ndarray (and other array-likes): reject unless truly 0-dimensional
    ndim = getattr(value, "ndim", None)
    if ndim is not None and int(ndim) != 0:
        raise ValueError("non-scalar numeric field (array)")
    try:
        if isinstance(value, int):
            return int(value)
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError, OverflowError):
        return None


def build_record(
    *,
    frame,
    normalization_method,
    status,
    reason,
    estimator,
    canvas_area,
    geometric_support_pixel_count=None,
    geometric_support_fraction=None,
    effective_support_pixel_count=None,
    effective_support_fraction=None,
    overlap_pixel_count=None,
    overlap_fraction=None,
    sampled_estimator_count=None,
    sky_offset=None,
    linear_a=None,
    linear_b=None,
    neutral_reason=None,
    frame_evidence=EVIDENCE_HEADER_SRCFILE,
    estimator_reason=None,
    session_id=None,
    ts=None,
    schema_version=SCHEMA_VERSION,
):
    """Build one versioned, allowlisted normalization diagnostic record.

    Only the bounded scalar fields listed in :data:`ALLOWED_KEYS` are
    produced; arbitrary nested payloads are impossible by construction (the
    caller passes explicit keyword arguments, not a free-form dict).  All
    numbers are converted to finite JSON-safe values (``None`` replaces
    non-finite or non-numeric input).  Channel vectors are validated against
    ``MAX_CHANNELS``.  ``frame`` is stored verbatim (never truncated).
    """
    if estimator_reason is None:
        estimator_reason = reason
    if linear_a is not None or linear_b is not None:
        a_list = list(linear_a or [])
        b_list = list(linear_b or [])
        if not (1 <= len(a_list) <= MAX_CHANNELS and len(a_list) == len(b_list)):
            raise ValueError(
                "linear channel vectors must be equal-length bounded lists "
                f"(1..{MAX_CHANNELS})"
            )
        a_clean = [_json_number(v) for v in a_list]
        b_clean = [_json_number(v) for v in b_list]
        if any(v is None for v in a_clean) or any(v is None for v in b_clean):
            raise ValueError("linear channel vectors must contain finite numbers")
        linear_a_out = a_clean
        linear_b_out = b_clean
    else:
        linear_a_out = None
        linear_b_out = None

    return {
        "schema_version": schema_version,
        "ts": ts if ts is not None else time.time(),
        "session_id": session_id,
        "event": "normalization",
        "frame": None if frame is None else str(frame),
        "frame_evidence": frame_evidence,
        "normalization_method": normalization_method,
        "status": status,
        "reason": reason,
        "neutral_reason": neutral_reason,
        "estimator": estimator,
        "estimator_reason": estimator_reason,
        "canvas_area": _json_number(canvas_area),
        "geometric_support_pixel_count": _json_number(
            geometric_support_pixel_count
        ),
        "geometric_support_fraction": _json_number(geometric_support_fraction),
        "effective_support_pixel_count": _json_number(
            effective_support_pixel_count
        ),
        "effective_support_fraction": _json_number(effective_support_fraction),
        "overlap_pixel_count": _json_number(overlap_pixel_count),
        "overlap_fraction": _json_number(overlap_fraction),
        "sampled_estimator_count": _json_number(sampled_estimator_count),
        "sky_offset": _json_number(sky_offset),
        "linear_a": linear_a_out,
        "linear_b": linear_b_out,
    }


def validate_record(record):
    """Validate a record against the allowlist; ``(ok, reason)``.

    Rejects unknown top-level keys, non-scalar values (except the bounded
    ``linear_a``/``linear_b`` channel vectors), non-finite numbers, records
    that are too large, and channel vectors outside ``1..MAX_CHANNELS``.
    ``None`` is an accepted explicit-unknown value for every field.
    """
    if not isinstance(record, dict):
        return False, "record is not a dict"
    unknown = set(record) - ALLOWED_KEYS
    if unknown:
        return False, f"unknown keys: {sorted(unknown)!r}"
    for key in ALLOWED_KEYS:
        if key not in record:
            return False, f"missing key: {key!r}"
    for key in NUMERIC_KEYS:
        val = record[key]
        if val is None:
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return False, f"{key} is not a scalar number"
        if isinstance(val, float) and not math.isfinite(val):
            return False, f"{key} is not finite"
    for key in CHANNEL_VECTOR_KEYS:
        val = record[key]
        if val is None:
            continue
        if not isinstance(val, (list, tuple)):
            return False, f"{key} is not a list"
        if not (1 <= len(val) <= MAX_CHANNELS):
            return False, f"{key} length out of bounds (1..{MAX_CHANNELS})"
        for item in val:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return False, f"{key} contains a non-number"
            if isinstance(item, float) and not math.isfinite(item):
                return False, f"{key} contains a non-finite number"
    if record["linear_a"] is None and record["linear_b"] is not None:
        return False, "linear_b present without linear_a"
    if record["linear_b"] is None and record["linear_a"] is not None:
        return False, "linear_a present without linear_b"
    if (
        isinstance(record["linear_a"], (list, tuple))
        and isinstance(record["linear_b"], (list, tuple))
        and len(record["linear_a"]) != len(record["linear_b"])
    ):
        return False, "linear_a/linear_b length mismatch"
    for key in ("frame", "frame_evidence", "normalization_method", "status",
                "reason", "neutral_reason", "estimator", "estimator_reason",
                "session_id", "event", "schema_version"):
        val = record[key]
        if val is not None and not isinstance(val, str):
            return False, f"{key} is not a string"
    if not isinstance(record["ts"], (int, float)) or isinstance(
        record["ts"], bool
    ):
        return False, "ts is not a number"
    if isinstance(record["ts"], float) and not math.isfinite(record["ts"]):
        return False, "ts is not finite"
    try:
        payload = json.dumps(record, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        return False, f"not JSON-serializable: {exc}"
    if len(payload.encode("utf-8")) > MAX_RECORD_BYTES:
        return False, "record exceeds MAX_RECORD_BYTES"
    return True, "ok"


def append_record(path, record):
    """Append one JSON record to ``path``.  Fail-open: never raises.

    The record is first validated against the allowlist / scalar bounds; a
    violating record is refused (``False``) and never partially written.  Any
    I/O or serialization error is caught, logged at debug level and swallowed
    — diagnostics I/O must never affect normalization success or the
    scientific result.  Appends only (``mode="a"``): an artifact is never
    truncated or reused ambiguously by this module.
    """
    try:
        ok, why = validate_record(record)
        if not ok:
            logger.debug(
                "normalization diagnostics record refused (non-fatal): %s", why
            )
            return False
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        line = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
        return True
    except Exception as exc:  # noqa: BLE001 — fail-open is the contract
        logger.debug(
            "normalization diagnostics write failed (non-fatal): %s", exc
        )
        return False
