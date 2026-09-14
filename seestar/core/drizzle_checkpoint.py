"""Production write-only checkpoint for the native M3 Drizzle accumulator state.

RSM2-D1: this module persists an *exact, self-describing* snapshot of the three
per-channel :class:`~seestar.core.drizzle_core.DrizzleAccumulator` native
buffers (``out_img`` weighted-mean science and ``out_wht`` total signed weight),
plus the runtime-effective scientific configuration, the output WCS/grid and the
session/source ledger, at safe accepted-pose boundaries.

The D1 writer is **write-only**: no ``open``/``resume``/``finalize``/preview/
derived-SCI persistence.  RSM2-D2A adds a strictly **read-only** loader /
validator (:func:`read_drizzle_checkpoint`) that never mutates checkpoint
bytes, source files, the output directory or any live runtime state, and that
reconstructs the three accumulators with
:meth:`DrizzleAccumulator.from_native_state` only after the *entire* checkpoint
validates.  It does **not** activate Resume in queue_manager / GUI / lifecycle.
The native float32 buffers are persisted bit-exactly (the signed Lanczos WHT is
never abs'ed / clipped / thresholded), so the reader can reconstruct the
accumulators and continue deposition bit-identically (proved by
``tests/test_drizzle_resume_continuation.py`` and
``tests/test_drizzle_checkpoint_reader.py``).

RSM2-D2B1 adds the *continuation-writer seam*: the classmethod
:meth:`DrizzleCheckpointWriter.from_validated_result` re-arms the atomic writer
at generation ``N+1`` **only** from an already-validated
:class:`DrizzleCheckpointResult` (which now carries immutable
``source_output_dir`` provenance).  The public ``__init__`` remains
fresh-run-only and continues to refuse any non-empty ``.m3d_checkpoint``
exactly as D1 — there is no ``allow_existing``-style public bypass.  A re-armed
writer commits atomically and monotonically (no rollback / rewrite / reorder /
divergent prefix, no cumulative-counter rollback), claiming generation ``N+1``
artifacts exclusively, and garbage-collects generation ``N`` only *after* the
``N+1`` manifest commits.  The factory performs a **fresh, full** re-read of
the on-disk checkpoint (never trusting the shallow-frozen mutable payloads of
the supplied result) and returns a dedicated :class:`DrizzleContinuation`
re-arm result carrying the fresh writer and the fresh reconstructed
accumulators / session / counters / ledger, so the lifecycle cannot
accidentally continue from stale/tampered result state.  No lifecycle /
queue_manager / GUI activation is performed.  Three final invariants harden the
continuation seam: (1) the exact next continuation baseline is deep-copied
entirely during preflight, so after a successful manifest commit only
non-fallible scalar/reference assignments occur (GC stays best-effort);
(2) cumulative unknown/known exposure arithmetic forbids retroactive
reclassification of already-committed frames and any fabrication of the
known-exposure summary when only unknown frames are added; (3)
``source_output_dir`` is bound to the canonical real path (``realpath``) and
the factory refuses a symlink swap, so a validated result can never be rebound
to another run's checkpoint.

RSM2-D2B2A adds the *source-resolution seam*: :func:`read_drizzle_checkpoint`
accepts an explicit, opt-in ``resolver`` callback / policy.  The default
(``resolver=None``) remains **strict D2A** — every persisted source identity is
re-stat'ed at its original path (exact size + mtime_ns, no rename / move /
missing fallback).  With a resolver, the reader offers each canonical identity
(plus a context dict: ``role`` / ``index`` / ``is_completed`` / ``output_dir``
/ ``input_roots``) to the resolver, which may return an *ordered* candidate
path list; the reader itself re-stats every candidate and only accepts a
regular, non-symlink file whose size + mtime_ns match exactly (never trusting
the callback).  Resolution is injective and order-preserving for *distinct*
canonical identities: two distinct identities resolving to one on-disk path is
refused as ambiguous, while repeated use of the exact same canonical identity
(the alignment reference also being one of the plan observations) resolves
legitimately to the same path.  The shipped production policy
:class:`SafeStackedSourceResolver` is an **immutable**
(frozen) object that only ever returns (a) the original path or (b) the
deterministic ``<original_dir>/<stacked_subdir_name>/<basename>`` counterpart —
the exact ``move_stacked`` destination of ``tools.file_ops.move_to_stacked`` —
with no directory search / glob / basename-only fallback / hashless remap /
arbitrary rename, and a ``_dup_<timestamp>`` collision name is never guessed.
The validated :class:`DrizzleCheckpointResult` exposes ``resolved_reference`` /
``resolved_plan_paths`` / ``resolved_completed_paths`` /
``resolved_remaining_paths`` and carries the immutable ``resolution_policy`` as
provenance; :meth:`DrizzleCheckpointWriter.from_validated_result` re-applies it
on its fresh re-read (never a mutable callback), while the continuation writer
still commits the **original canonical** plan/ledger identities — never a
rewritten stacked path.  Persisted manifest/session identities and source bytes
are never mutated by the reader or re-arm.

Layout
------

::

    <output>/.m3d_checkpoint/
        checkpoint.json                  # the ONLY commit point (manifest)
        gen-00000001-ch0-out_img.npy     # generation-unique native arrays
        gen-00000001-ch0-out_wht.npy
        gen-00000001-ch1-out_img.npy
        gen-00000001-ch1-out_wht.npy
        gen-00000001-ch2-out_img.npy
        gen-00000001-ch2-out_wht.npy
        gen-00000001-support_w1.npy     # optional additive positive support
        gen-00000001-support_w2.npy
    <output>/run_config.cfg              # canonical schema-v2 run config (stable)

This namespace is **dedicated** to Drizzle and never reuses the classic
``memmap_accumulators/resume_manifest.json`` SUM/W artifacts (which remain
plain-classic only and are never overloaded or weakened).

Restart safety (fresh writer refuses; Resume re-arms)
--------------------------------------

Resume is enabled via the re-armed writer (from_validated_result); a freshly
constructed writer never continues an existing namespace and therefore **refuses** (fail closed,
preserving every pre-existing byte) whenever ``<output>/.m3d_checkpoint`` is
non-empty — i.e. it already contains a manifest, an allowlisted generation
artifact, a manifest temp or a writer temp.  An empty existing directory is
allowed.  This refusal runs both at construction and defensively at the first
commit, so a second writer/process can never reuse, overwrite, clean or GC a
prior generation (the "gen-00000001" name collision is structurally impossible:
the namespace is refused before any write, and every generation artifact is
additionally claimed with ``O_CREAT | O_EXCL``, never ``os.replace`` onto a
pre-existing path).

Copy-on-write / commit protocol
-------------------------------

Every generation writes its six native array artifacts and optional two-array
positive support under generation-unique final names claimed **exclusively**
(``O_CREAT | O_EXCL`` + in-place write + fsync), computes a SHA-256 and exact
byte size for each, then writes
a per-attempt **owned** manifest temp
``checkpoint.json.tmp.<pid>.<seq>.<nonce>`` (claimed exclusively with
``open(..., "x")``, fsync) and ``os.replace``-s it to ``checkpoint.json``
**last**.  The temp name is unique per attempt/process, so no writer can ever
delete or overwrite another writer's manifest temp.  ``checkpoint.json`` is the
single commit point: a crash before that replace leaves the prior manifest and
every file it references byte-identical and usable; the attempt's own files are
cleaned best-effort (never a pre-existing path).  ``json.dumps(...,
allow_nan=False)`` forbids NaN/Inf anywhere in the manifest, and both the
checkpoint directory and the output directory are fsync'ed after their renames
so directory entries are durable.  After a successful commit, stale previous
generations may be garbage-collected only from the explicit writer-owned
``gen-*.npy`` allowlist pattern — never a broad directory delete and never the
current generation.
"""

from __future__ import annotations

import copy
import hashlib
import io
import itertools
import json
import os
import re
import secrets
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from seestar import run_contract
from seestar.core.drizzle_core import (
    DrizzleAccumulator,
    LANCZOS_KERNELS,
    PIXEL_SCALE_RATIO_SOURCE,
    VALID_DRIZZLE_KERNELS,
    build_output_grid,
    derive_pixel_scale_ratio,
)

__all__ = [
    "DrizzleCheckpointError",
    "DrizzleCheckpointWriter",
    "DrizzleCheckpointResult",
    "DrizzleContinuation",
    "SafeStackedSourceResolver",
    "read_drizzle_checkpoint",
    "build_drizzle_canonical_config",
    "serialize_wcs_header",
    "serialize_input_reference_geometry",
    "reconstruct_input_reference_wcs",
    "output_grid_contract_identity",
    "INPUT_REFERENCE_GEOMETRY_CONTRACT",
    "INPUT_REFERENCE_GEOMETRY_VERSION",
    "CHECKPOINT_DIRNAME",
    "MANIFEST_FILENAME",
    "SCHEMA_VERSION",
    "MODE_TOKEN",
    "STATE_CLEAN",
    "RUN_CONFIG_FILENAME",
]


class DrizzleCheckpointError(RuntimeError):
    """Raised when a Drizzle checkpoint persist/commit operation fails.

    A checkpoint failure is *mandatory-abort*: the caller must stop processing
    before the source is moved, never warn-and-continue.  The prior committed
    checkpoint (if any) stays byte-identical and usable.
    """


# ---------------------------------------------------------------------------
# Namespace / schema constants
# ---------------------------------------------------------------------------

CHECKPOINT_DIRNAME = ".m3d_checkpoint"
MANIFEST_FILENAME = "checkpoint.json"
MANIFEST_TMP_FILENAME = "checkpoint.json.tmp"
MANIFEST_TMP_PREFIX = "checkpoint.json.tmp."
RUN_CONFIG_FILENAME = "run_config.cfg"

SCHEMA_VERSION = 1
MODE_TOKEN = "drizzle_native_v1"
STATE_CLEAN = "clean"

# Stable, documented runtime-effective contract tokens (D1 pins them; a later
# task may bump them if the underlying science contracts change).  They are
# fingerprint inputs only — never behavioural switches.
_WHT_POLICY_TOKEN = "relative_coverage_v1"
_BACKGROUND_MATCH_CONTRACT = "dpic01_bgmatch_v1"
_BACKGROUND_MATCH_CONTRACT_VERSION = 1
# v2 (GAR-06): the Standard output grid is an exact scaled copy of the frozen
# reference grid (projection/frame/orientation/handedness preserved, effective
# matrix divided by the scale, FITS edge/centre-preserving CRPIX).  v1 was the
# historical north-up / raw-CRPIX-scaled grid; it is deliberately NOT migratable
# in place (accumulated arrays are never silently converted).
_OUTPUT_GRID_CONTRACT = "m3_output_grid_v2"
_OUTPUT_GRID_CONTRACT_VERSION = 2
_LEGACY_OUTPUT_GRID_CONTRACTS = {("m3_output_grid_v1", 1)}
_REGISTRATION_CONTRACT = "m3_tf_registration_v1"
_REGISTRATION_CONTRACT_VERSION = 1

# Versioned frozen input-reference geometry payload persisted in the session
# binding so a resume can restore the exact input-reference WCS (re-solving
# alone is not proof of stability).
INPUT_REFERENCE_GEOMETRY_CONTRACT = "m3_input_reference_geometry_v1"
INPUT_REFERENCE_GEOMETRY_VERSION = 1

# Explicit allowlist for generation-unique array artifacts.  Garbage collection
# and failure cleanup may only ever touch names matching these patterns.  The
# positive-support artifacts are additive to schema v1: a legacy manifest may
# omit them, but a manifest that advertises support must reference both.
_CHANNEL_ARTIFACT_RE = re.compile(
    r"^gen-(\d{8})-ch([0-2])-out_(img|wht)\.npy$"
)
_SUPPORT_ARTIFACT_RE = re.compile(
    r"^gen-(\d{8})-support_(w1|w2)\.npy$"
)
_ARTIFACT_RE = re.compile(
    r"^gen-(\d{8})-(?:ch[0-2]-out_(?:img|wht)|support_(?:w1|w2))\.npy$"
)

# A ``_dup_<timestamp>`` collision name (produced by
# ``tools.file_ops.move_to_stacked`` when the deterministic destination already
# exists).  The source-resolution policy must never *guess* such a target: the
# deterministic ``<stacked>/<basename>`` path is the only acceptable move.
_DUP_COLLISION_RE = re.compile(r"_dup_\d+")

# Legacy same-directory writer-temp prefix/suffix (still recognized as a
# restart-refusal trigger; the array artifacts themselves are now claimed
# exclusively in place via ``O_CREAT | O_EXCL``).
_ARRAY_TMP_PREFIX = ".tmp-"
_ARRAY_TMP_SUFFIX = ".npy"

# Per-process monotonic counter used to make manifest temporary names unique
# across attempts within one process (cross-process uniqueness additionally
# relies on the pid and a random nonce, plus the exclusive ``open(..., "x")``
# claim).  It never needs to be reset: monotonicity alone is enough.
_MANIFEST_TMP_COUNTER = itertools.count()


def _is_manifest_temp(name: str) -> bool:
    """Return True if ``name`` is a manifest temporary file (any supported form).

    Covers the legacy shared ``checkpoint.json.tmp`` name as well as every
    per-attempt owned form ``checkpoint.json.tmp.<pid>.<seq>.<nonce>`` produced
    by :meth:`DrizzleCheckpointWriter._claim_manifest_temp`, so restart refusal
    recognizes every supported writer-temp naming form.
    """
    return name == MANIFEST_TMP_FILENAME or name.startswith(MANIFEST_TMP_PREFIX)


def _drizzle_lib_version() -> str:
    try:
        import drizzle

        return str(getattr(drizzle, "__version__", "") or "")
    except Exception:
        return ""


def _numpy_version() -> str:
    return str(np.__version__)


def _json_scalar(value):
    """Return a strictly JSON-safe scalar (rejecting NaN/Inf and non-scalars)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if not np.isfinite(f):
            raise ValueError("non-finite number")
        return f
    if isinstance(value, (list, tuple)):
        return [_json_scalar(v) for v in value]
    raise ValueError(f"non-JSON value of type {type(value).__name__}")


def _strict_int(value, name):
    """Validate a strict non-bool integer, raising :class:`DrizzleCheckpointError`.

    Only genuine integral scalars (``int`` / :class:`numpy.integer`) are
    accepted.  Floats, strings and other numeric lookalikes are rejected rather
    than silently truncated or coerced.
    """
    if isinstance(value, bool):
        raise DrizzleCheckpointError(f"{name} must be an integer, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, np.integer):
        return int(value)
    raise DrizzleCheckpointError(
        f"{name} must be a strict integer, got {type(value).__name__}"
    )


def _strict_float(value, name, *, allow_none=False):
    """Coerce to a finite float, raising :class:`DrizzleCheckpointError` on failure."""
    if value is None:
        if allow_none:
            return None
        raise DrizzleCheckpointError(f"{name} must be a number, not None")
    if isinstance(value, bool):
        raise DrizzleCheckpointError(f"{name} must be a number, not bool")
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DrizzleCheckpointError(f"{name} must be a number: {exc}") from exc
    if not np.isfinite(f):
        raise DrizzleCheckpointError(f"{name} must be finite, got {value!r}")
    return f


def _validate_identity(entry, where):
    """Validate one source identity and return its canonical JSON-safe form.

    ``path`` must be a non-empty string and ``size`` / ``mtime_ns`` strict
    non-bool integers.  ``name`` is preserved when present (a non-empty string)
    and otherwise derived from ``path``.  Unknown extra keys are dropped.
    """
    if not isinstance(entry, dict):
        raise DrizzleCheckpointError(f"non-identity entry in {where}")
    path = entry.get("path")
    if not isinstance(path, str) or not path:
        raise DrizzleCheckpointError(f"unstattable source in {where}: missing path")
    size = entry.get("size")
    mtime = entry.get("mtime_ns")
    if isinstance(size, bool) or not isinstance(size, int):
        raise DrizzleCheckpointError(
            f"unstattable source in {where}: size must be a strict integer"
        )
    if isinstance(mtime, bool) or not isinstance(mtime, int):
        raise DrizzleCheckpointError(
            f"unstattable source in {where}: mtime_ns must be a strict integer"
        )
    name = entry.get("name")
    if name is not None and (not isinstance(name, str) or not name):
        raise DrizzleCheckpointError(f"unstattable source in {where}: invalid name")
    return {
        "path": path,
        "name": name or os.path.basename(path),
        "size": int(size),
        "mtime_ns": int(mtime),
    }


def _normalize_fillval(value, where="fillval"):
    """Normalize a fillval to a comparable canonical form.

    Scientific/serialization equivalence rule (documented contract): a numeric
    fillval and a string that parses to the *same finite float* are equivalent
    (``0.0`` == ``"0.0"`` == ``"0.00"``).  A string that is not a finite-float
    literal (e.g. ``"INDEF"``) is compared by exact string identity and is
    never coerced to a number.  Bools, non-finite numbers and any other type
    are rejected (they are never a valid serialized fillval).
    """
    if isinstance(value, bool):
        raise DrizzleCheckpointError(f"{where} must not be a bool")
    if isinstance(value, str):
        text = value
        try:
            f = float(text)
        except ValueError:
            return ("str", text)
        if not np.isfinite(f):
            return ("str", text)
        return ("num", f)
    if isinstance(value, (int, float, np.integer, np.floating)):
        f = float(value)
        if not np.isfinite(f):
            raise DrizzleCheckpointError(f"{where} must be finite")
        return ("num", f)
    raise DrizzleCheckpointError(
        f"{where} must be a string or finite number, got "
        f"{type(value).__name__}"
    )


def _check_deposition_matches_canonical(kernel, pixfrac, fillval, scientific,
                                        where):
    """Fail closed when runtime deposition params disagree with canonical config.

    ``kernel`` / ``pixfrac`` / ``fillval`` are the runtime-effective per-channel
    deposition parameters; ``scientific`` is the canonical scientific mapping
    whose ``drizzle_kernel_effective`` / ``drizzle_pixfrac_effective`` /
    ``drizzle_fillval`` fields are the single source of truth.  ``fillval`` is
    compared with scientific/serialization equivalence via
    :func:`_normalize_fillval` (a numeric ``0.0`` equals the canonical string
    ``"0.0"``).
    """
    canon_kernel = scientific.get("drizzle_kernel_effective")
    if not isinstance(canon_kernel, str) or canon_kernel != kernel:
        raise DrizzleCheckpointError(
            f"{where} kernel {kernel!r} != canonical "
            f"drizzle_kernel_effective {canon_kernel!r}"
        )
    canon_pixfrac = _strict_float(
        scientific.get("drizzle_pixfrac_effective"),
        "canonical drizzle_pixfrac_effective",
    )
    # P2-D: a pre-existing checkpoint whose canonical effective pixfrac exceeds
    # the canonical envelope (> 1.0) cannot be continued; it is not
    # scientifically legal to keep depositing into an accumulator created with a
    # different effective parameter, and the stored state is never rewritten.
    if canon_pixfrac > 1.0:
        raise DrizzleCheckpointError(
            "checkpoint_pixfrac_effective_gt_one_incompatible: canonical "
            f"drizzle_pixfrac_effective {canon_pixfrac!r} > 1.0 cannot be "
            "continued (P2-D canonical pixfrac envelope is (0, 1])"
        )
    if canon_pixfrac != float(pixfrac):
        raise DrizzleCheckpointError(
            f"{where} pixfrac {pixfrac!r} != canonical "
            f"drizzle_pixfrac_effective {canon_pixfrac!r}"
        )
    canon_fillval = scientific.get("drizzle_fillval")
    if _normalize_fillval(fillval, f"{where} fillval") != _normalize_fillval(
        canon_fillval, "canonical drizzle_fillval"
    ):
        raise DrizzleCheckpointError(
            f"{where} fillval {fillval!r} != canonical drizzle_fillval "
            f"{canon_fillval!r}"
        )


def _fsync_dir(path):
    """Best-effort fsync of a directory (POSIX); no-op elsewhere.

    Durability of ``os.replace`` / ``os.open``-created directory entries
    requires the *parent directory* to be fsync'ed in addition to the file.
    This helper is strictly best-effort and never raises: an unsupported
    platform or a transient OSError degrades to file-fsync-only durability.
    """
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        try:
            os.fsync(fd)
        except Exception:  # noqa: BLE001 - best-effort durability
            pass
    finally:
        try:
            os.close(fd)
        except Exception:  # noqa: BLE001 - best-effort durability
            pass


def serialize_wcs_header(wcs) -> dict:
    """Serialize an ``astropy.wcs.WCS`` output grid to a JSON-safe dict.

    ``checkpoint.json`` needs the *serialized* output WCS so a future reader can
    reconstruct the exact same grid; the ``output_shape_hw`` is recorded
    separately and re-attached as ``array_shape``.  Fail closed (raise
    :class:`DrizzleCheckpointError`) when the WCS is unavailable or any card is
    non-JSON/non-finite — a checkpoint without a faithful grid is never
    published.
    """
    if wcs is None:
        raise DrizzleCheckpointError("output WCS unavailable")
    try:
        header = wcs.to_header(relax=True)
    except Exception as exc:  # noqa: BLE001 - fail closed, never partial
        raise DrizzleCheckpointError(f"cannot serialize output WCS: {exc}") from exc

    out: dict = {}
    for key in header.keys():
        if key in ("HISTORY", "COMMENT", ""):
            continue
        try:
            out[str(key)] = _json_scalar(header[key])
        except (ValueError, TypeError) as exc:
            raise DrizzleCheckpointError(
                f"non-JSON output WCS card {key!r}: {exc}"
            ) from exc

    # Canonical SIP persistence: a forward-only SIP (AP_ORDER/BP_ORDER == 0
    # with no AP_/BP_ coefficients) is stored without the empty order cards so
    # that the persisted header round-trips exactly through WCS re-parsing
    # (Astropy omits the zero order cards on reserialization).
    sip_inverse_coeffs = [
        key
        for key in out
        if key.startswith(("AP_", "BP_")) and key not in ("AP_ORDER", "BP_ORDER")
    ]
    if out.get("AP_ORDER") == 0 and not sip_inverse_coeffs:
        out.pop("AP_ORDER", None)
        out.pop("BP_ORDER", None)
    return out


def serialize_input_reference_geometry(reference_wcs, reference_shape_hw=None,
                                        reference_identity=None):
    """Serialize the frozen *input-reference* geometry (or ``None``).

    Versioned so a future contract change is detectable.  The persisted WCS
    reproduces the exact input reference used for registration, which the
    output-grid payload alone cannot prove.
    """
    if reference_wcs is None:
        return None
    shape = None
    if reference_shape_hw is not None and len(tuple(reference_shape_hw)) == 2:
        shape = [int(reference_shape_hw[0]), int(reference_shape_hw[1])]
    identity = None
    if isinstance(reference_identity, dict):
        identity = dict(reference_identity)
    return {
        "contract": INPUT_REFERENCE_GEOMETRY_CONTRACT,
        "contract_version": INPUT_REFERENCE_GEOMETRY_VERSION,
        "shape_hw": shape,
        "wcs": serialize_wcs_header(reference_wcs),
        "identity": identity,
    }


def _validate_input_reference_geometry(payload, where="input reference geometry"):
    """Validate a persisted input-reference geometry payload (or ``None``)."""
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise DrizzleCheckpointError(f"{where} must be a mapping")
    if payload.get("contract") != INPUT_REFERENCE_GEOMETRY_CONTRACT:
        raise DrizzleCheckpointError(
            f"{where} contract mismatch: {payload.get('contract')!r}"
        )
    version = _strict_int(payload.get("contract_version"), f"{where} version")
    if version != INPUT_REFERENCE_GEOMETRY_VERSION:
        raise DrizzleCheckpointError(
            f"{where} unsupported contract_version {version}"
        )
    shape = None
    shape_raw = payload.get("shape_hw")
    if shape_raw is not None:
        if not isinstance(shape_raw, list) or len(shape_raw) != 2:
            raise DrizzleCheckpointError(f"{where} shape_hw must be a 2-element list")
        h = _strict_int(shape_raw[0], f"{where} shape_hw[0]")
        w = _strict_int(shape_raw[1], f"{where} shape_hw[1]")
        if h <= 0 or w <= 0:
            raise DrizzleCheckpointError(f"{where} invalid shape_hw {(h, w)}")
        shape = [h, w]
    wcs_dict = payload.get("wcs")
    if not isinstance(wcs_dict, dict) or not wcs_dict:
        raise DrizzleCheckpointError(f"{where} wcs is missing or malformed")
    _wcs_from_cards(wcs_dict, where)
    identity = None
    if payload.get("identity") is not None:
        identity = _validate_identity(payload.get("identity"), f"{where} identity")
    return {
        "contract": INPUT_REFERENCE_GEOMETRY_CONTRACT,
        "contract_version": INPUT_REFERENCE_GEOMETRY_VERSION,
        "shape_hw": shape,
        "wcs": dict(wcs_dict),
        "identity": identity,
    }


def reconstruct_input_reference_wcs(reference_geometry):
    """Rebuild the frozen input-reference WCS from a validated payload.

    Returns ``None`` when no payload is present (callers then keep whatever
    reference preparation produced; a legacy checkpoint without the payload is
    refused earlier by the output-grid contract check).
    """
    if not isinstance(reference_geometry, dict):
        return None
    wcs_dict = reference_geometry.get("wcs")
    if not isinstance(wcs_dict, dict) or not wcs_dict:
        return None
    wcs = _wcs_from_cards(wcs_dict, "input reference geometry")
    shape = reference_geometry.get("shape_hw")
    if isinstance(shape, list) and len(shape) == 2:
        h, w = int(shape[0]), int(shape[1])
        try:
            wcs.array_shape = (h, w)
        except Exception:  # noqa: BLE001
            pass
        try:
            wcs.pixel_shape = (w, h)
        except Exception:  # noqa: BLE001
            pass
    return wcs


def output_grid_contract_identity():
    """Return the stable output-grid contract identity (token + version)."""
    return {
        "contract": _OUTPUT_GRID_CONTRACT,
        "contract_version": _OUTPUT_GRID_CONTRACT_VERSION,
    }


def _resolve_geometry_facts(qm):
    """Return ``(derived, effective, source)`` for the kernel-scale factor.

    Prefers the engine's ONE frozen geometry seam
    (``qm._freeze_drizzle_geometry``) so every consumer of the canonical
    config sees exactly the same value; otherwise derives it deterministically
    from the canonical reference WCS and the canonical output grid.  Returns
    ``(None, None, None)`` when the geometry is not resolvable here; the
    deposition path still fails closed before ever using the 1.0 default.
    """
    freeze = getattr(qm, "_freeze_drizzle_geometry", None)
    if callable(freeze):
        try:
            ratio = freeze()
        except Exception:  # noqa: BLE001 - treat as unresolved here
            ratio = None
        if ratio is not None:
            return (
                getattr(qm, "drizzle_pixel_scale_ratio_derived", ratio),
                ratio,
                getattr(qm, "drizzle_pixel_scale_ratio_source", PIXEL_SCALE_RATIO_SOURCE),
            )
    eff = getattr(qm, "drizzle_pixel_scale_ratio_effective", None)
    if eff is not None:
        return (
            getattr(qm, "drizzle_pixel_scale_ratio_derived", eff),
            eff,
            getattr(qm, "drizzle_pixel_scale_ratio_source", PIXEL_SCALE_RATIO_SOURCE),
        )
    ref_wcs = getattr(qm, "reference_wcs_object", None)
    if ref_wcs is None:
        return (None, None, None)
    try:
        out_wcs = getattr(qm, "drizzle_output_wcs", None)
        if out_wcs is None:
            scale = float(getattr(qm, "drizzle_scale", 1.0) or 1.0)
            out_wcs = build_output_grid(ref_wcs, (1, 1), scale)[0]
        ratio = float(derive_pixel_scale_ratio(ref_wcs, out_wcs))
    except Exception:  # noqa: BLE001 - unresolved here; deposition fails closed
        return (None, None, None)
    try:
        qm.drizzle_pixel_scale_ratio_requested = None
        qm.drizzle_pixel_scale_ratio_derived = ratio
        qm.drizzle_pixel_scale_ratio_effective = ratio
        qm.drizzle_pixel_scale_ratio_source = PIXEL_SCALE_RATIO_SOURCE
    except Exception:  # noqa: BLE001
        pass
    return (ratio, ratio, PIXEL_SCALE_RATIO_SOURCE)


def build_drizzle_canonical_config(qm, product_version: str = "") -> run_contract.RunConfig:
    """Build the canonical schema-v2 :class:`run_contract.RunConfig` for a
    Drizzle run from the runtime-effective engine state.

    ``run_contract.drizzle_fingerprint`` requires every effective Drizzle field
    (fail closed, never a partial payload); this helper supplies all of them
    from the engine instance plus the stable D1 contract tokens.  It performs
    no I/O.  The ``product_version`` is supplied by the caller (the engine's
    ``_canonical_product_version``).
    """
    _psr_derived, _psr_effective, _psr_source = _resolve_geometry_facts(qm)
    scientific = {
        # Shared weighting / hot-pixel / debayer contract (both domains).
        "weighting_method": str(getattr(qm, "weighting_method", "none") or "none"),
        "use_quality_weighting": bool(getattr(qm, "use_quality_weighting", False)),
        "weight_by_snr": bool(getattr(qm, "weight_by_snr", True)),
        "weight_by_stars": bool(getattr(qm, "weight_by_stars", True)),
        "snr_exponent": float(getattr(qm, "snr_exponent", 1.0) or 1.0),
        "stars_exponent": float(getattr(qm, "stars_exponent", 0.5) or 0.5),
        "min_weight": float(getattr(qm, "min_weight", 0.01) or 0.01),
        "correct_hot_pixels": bool(getattr(qm, "correct_hot_pixels", True)),
        "hot_pixel_threshold": float(getattr(qm, "hot_pixel_threshold", 3.0) or 3.0),
        "neighborhood_size": int(getattr(qm, "neighborhood_size", 5) or 5),
        "bayer_pattern": str(getattr(qm, "bayer_pattern", "GRBG") or "GRBG"),
        # Effective drizzle deposition contract.
        "drizzle_scale_effective": float(getattr(qm, "drizzle_scale", 1.0) or 1.0),
        "drizzle_kernel_effective": str(
            getattr(qm, "drizzle_kernel", "square") or "square"
        ),
        "drizzle_pixfrac_effective": float(
            getattr(qm, "drizzle_pixfrac", 1.0) or 1.0
        ),
        "drizzle_wht_threshold_effective": float(
            getattr(
                qm,
                "drizzle_wht_threshold_effective",
                getattr(qm, "drizzle_wht_threshold", 0.0) or 0.0,
            )
            or 0.0
        ),
        "drizzle_wht_policy": _WHT_POLICY_TOKEN,
        "drizzle_fillval": str(getattr(qm, "drizzle_fillval", "0.0") or "0.0"),
        "drizzle_double_norm_fix": bool(
            getattr(qm, "drizzle_double_norm_fix", True)
        ),
        # P2-B geometry: ONE frozen WCS-derived kernel pixel-scale factor.
        # ``requested`` is truthfully None (never a fabricated user request).
        "pixel_scale_ratio_requested": None,
        "pixel_scale_ratio_derived": _psr_derived,
        "pixel_scale_ratio_effective": _psr_effective,
        "pixel_scale_ratio_source": _psr_source,
        "background_match_contract": _BACKGROUND_MATCH_CONTRACT,
        "background_match_contract_version": _BACKGROUND_MATCH_CONTRACT_VERSION,
        "output_grid_contract": _OUTPUT_GRID_CONTRACT,
        "output_grid_contract_version": _OUTPUT_GRID_CONTRACT_VERSION,
        "registration_contract": _REGISTRATION_CONTRACT,
        "registration_contract_version": _REGISTRATION_CONTRACT_VERSION,
    }
    # P2-D1: pixfrac provenance keys are CONDITIONAL so legacy configs that
    # predate them keep an identical full digest: omit the reason when absent
    # (never a synthetic null key) and the requested key when it equals the
    # effective value.
    _pf_req_v = getattr(qm, "drizzle_pixfrac_requested", None)
    _pf_eff_v = scientific.get("drizzle_pixfrac_effective")
    if (
        _pf_req_v is not None
        and _pf_eff_v is not None
        and float(_pf_req_v) != float(_pf_eff_v)
    ):
        scientific["drizzle_pixfrac_requested"] = float(_pf_req_v)
    _pf_rsn_v = getattr(qm, "drizzle_pixfrac_reason", None)
    if _pf_rsn_v:
        scientific["drizzle_pixfrac_reason"] = str(_pf_rsn_v)
    execution = {
        "drizzle_mode": str(getattr(qm, "drizzle_mode", "Final") or "Final"),
        "drizzle_group_size": int(getattr(qm, "drizzle_group_size", 50) or 50),
    }
    provenance = {"drizzle_lib_version": _drizzle_lib_version()}
    # ------------------------------------------------------------------
    # D4: deterministic RUN-START facts for the canonical run_config.cfg.
    # The canonical cfg is built once per run (checkpoint-init time) and must
    # never change digest across a resume, so every token below derives ONLY
    # from session state that is byte-identical at original-run time and at
    # resume-validation time (``drizzle_active_session``, the requested/effective
    # drizzle kernel, ``save_final_as_float32``) — never from mutable
    # finalization-time state.  Duck-typed harness objects without a drizzle
    # session keep their historical byte-identical cfgs (no token added).
    if bool(getattr(qm, "drizzle_active_session", False)) and not bool(
        getattr(qm, "is_mosaic_run", False)
    ):
        # The standard Drizzle path bypasses the Classic reducers entirely
        # (direct accumulation, no rejection): record the execution-aware
        # stacking semantics + the explicit substitution reason (D1.3 mirror).
        scientific["stacking_mode_effective"] = "drizzle_direct_accumulation"
        scientific["stacking_mode_substitution_reason"] = (
            "classic_reducer_not_used_by_drizzle_path"
        )
        # Requested vs effective drizzle kernel (the effective kernel is
        # already canonicalized by ``initialize`` / the read-only preflight
        # before the writer is constructed; an invalid request spelling is
        # therefore visible as requested != effective).
        effective_kernel = str(
            getattr(qm, "drizzle_kernel", "square") or "square"
        )
        scientific["drizzle_kernel_requested"] = str(
            getattr(qm, "_drizzle_kernel_requested", None)
            or effective_kernel
        )
        # D4 float32 canonicalization: a signed Lanczos kernel (lanczos2 /
        # lanczos3) legitimately produces negative ringing that a requested
        # uint16 export would silently clip, so the effective save dtype is a
        # deterministic RUN-START fact: float32 whenever the effective kernel
        # is signed Lanczos, whatever the caller requested.  ``requested``
        # keeps what the caller asked; ``effective`` is what the engine will
        # write; the explicit reason makes the canonicalization visible.
        save_requested = bool(getattr(qm, "save_final_as_float32", False))
        force_reason = (
            None
            if save_requested
            else (
                "signed_lanczos_requires_float32"
                if effective_kernel in LANCZOS_KERNELS
                else None
            )
        )
        execution["save_as_float32_requested"] = save_requested
        execution["save_as_float32_effective"] = (
            save_requested or force_reason is not None
        )
        if force_reason:
            execution["save_as_float32_reason"] = force_reason
    return run_contract.RunConfig.from_sections(
        product_version=product_version,
        scientific=scientific,
        execution=execution,
        provenance=provenance,
    )


class DrizzleCheckpointWriter:
    """Atomic, fail-closed writer for native Drizzle checkpoint generations.

    Constructed once per run (after the effective Drizzle configuration and
    output grid are known); each :meth:`commit` publishes one generation.

    The canonical ``run_config.cfg`` (and its digest / ``scientific_config`` /
    Drizzle fingerprint) is derived from the immutable ``canonical_cfg`` at
    construction, so every generation of one run carries identical scientific
    provenance.
    """

    def __init__(self, output_dir, product_version, canonical_cfg, output_wcs,
                 output_shape_hw):
        self.output_dir = str(output_dir)
        self.product_version = str(product_version or "")
        self.canonical_cfg = canonical_cfg
        self.output_wcs = output_wcs
        self.output_shape_hw = self._validate_output_shape_hw(output_shape_hw)

        self._dir = os.path.join(self.output_dir, CHECKPOINT_DIRNAME)

        # Restart safety: a fresh-run writer must never reuse a prior checkpoint
        # namespace.  Refuse (fail closed, preserve every pre-existing byte)
        # before any write if the dedicated namespace is non-empty.  This is the
        # ONLY construction path that runs the refusal; continuation writers are
        # created exclusively via :meth:`from_validated_result` (no public
        # ``allow_existing``-style bypass exists).
        self._refuse_existing_checkpoint()

        # Fail closed *before* any write: a malformed canonical config (missing
        # effective field) or an unserializable WCS must never yield a writer
        # that later publishes an unusable manifest.
        self._bind_canonical_identity()

        self._next_generation = 1
        self._current_generation = 0
        # Manifest temp owned by the *current* commit attempt (if any).  Set by
        # ``_claim_manifest_temp`` and cleared on replace/cleanup, so cleanup can
        # only ever remove the temp this attempt created — never a foreign one.
        self._manifest_tmp_path = None
        # Fresh run: no loaded checkpoint.  Continuation-mode monotonicity
        # checks stay disabled until :meth:`from_validated_result` rearms this
        # writer from an already-validated :class:`DrizzleCheckpointResult`.
        self._continuation_state = None

    @staticmethod
    def _validate_output_shape_hw(output_shape_hw):
        """Validate ``output_shape_hw`` into a canonical ``(H, W)`` tuple."""
        try:
            shape = tuple(int(v) for v in output_shape_hw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DrizzleCheckpointError(f"invalid output_shape_hw: {exc}") from exc
        if len(shape) != 2:
            raise DrizzleCheckpointError(
                f"output_shape_hw must be (H, W), got {shape!r}"
            )
        return shape

    def _bind_canonical_identity(self):
        """Derive the immutable canonical identity (fingerprint / digest /
        scientific config / serialized WCS) from the bound config and grid.

        Fail closed *before* any write: a malformed canonical config (missing
        effective field) or an unserializable WCS must never yield a writer
        that later publishes an unusable manifest.
        """
        try:
            self.fingerprint = run_contract.drizzle_fingerprint(self.canonical_cfg)
        except run_contract.ConfigError as exc:
            raise DrizzleCheckpointError(
                f"malformed canonical Drizzle config: {exc}"
            ) from exc
        self.run_config_digest = self.canonical_cfg.full_digest()
        self.scientific_config = dict(self.canonical_cfg.scientific)
        self._wcs_dict = serialize_wcs_header(self.output_wcs)

    @classmethod
    def from_validated_result(cls, result):
        """Re-arm a continuation writer from an already-validated checkpoint.

        This is the **only** supported entry into continuation mode.  The
        public :meth:`__init__` remains fresh-run-only (refusing any non-empty
        ``.m3d_checkpoint`` exactly as D1), so there is no public
        ``allow_existing``-style bypass that can be invoked without an
        already-validated :class:`DrizzleCheckpointResult`.

        Only two fields of ``result`` are trusted: the immutable, validated
        ``source_output_dir`` provenance and the frozen ``generation`` (used as
        a stale-result token).  Every other payload of ``result`` (manifest /
        session / counters / config / WCS / accumulators) is deliberately
        **not** trusted — those are shallow-frozen mutable payloads that may
        have been tampered with, and the result may be stale.  The factory
        therefore performs a **fresh, full**
        :func:`read_drizzle_checkpoint` of ``source_output_dir`` (exact library
        versions required) and binds *all* continuation state — config / WCS /
        grid / session / ledger / counters / per-channel total exposure / the
        reconstructed accumulators — from that freshly validated read.  If the
        freshly read generation differs from ``result.generation`` (another
        writer already continued, or the checkpoint changed), re-arm fails
        closed.

        Returns a dedicated :class:`DrizzleContinuation` re-arm result carrying
        the fresh writer **and** the fresh reconstructed accumulators / session
        / counters / ledger / ``next_source_index``, so the lifecycle cannot
        accidentally continue from the stale/tampered ``result`` payloads.
        Re-arm performs **no** writes and **no** garbage collection; the last
        committed manifest stays authoritative.
        """
        if not isinstance(result, DrizzleCheckpointResult):
            raise DrizzleCheckpointError(
                "from_validated_result requires a validated "
                "DrizzleCheckpointResult (got "
                f"{type(result).__name__}); re-arming from an arbitrary dict / "
                "unvalidated path is refused"
            )
        output_dir = result.source_output_dir
        if not isinstance(output_dir, str) or not output_dir:
            raise DrizzleCheckpointError(
                "DrizzleCheckpointResult has missing/invalid source_output_dir "
                "provenance"
            )
        # Canonical real-path re-resolution (D2B1 finding 3): a validated result
        # is bound to the exact real directory it was validated against.  If the
        # provenance now resolves elsewhere (a symlink was retargeted, or the
        # validated directory was swapped for a symlink to another checkpoint),
        # refuse instead of silently binding the other checkpoint.
        canonical_output_dir = os.path.realpath(output_dir)
        if canonical_output_dir != output_dir:
            raise DrizzleCheckpointError(
                "continuation source_output_dir provenance no longer resolves "
                "to its validated real directory (symlink swap/retarget "
                "detected); re-arm refused"
            )
        output_dir = canonical_output_dir
        expected_generation = _strict_int(result.generation, "result.generation")

        # Fresh, authoritative read-only validation of the on-disk checkpoint.
        # This is the ONLY source of truth for continuation state; the supplied
        # (possibly tampered / stale) result payloads are never trusted.
        # Re-apply the immutable source-resolution policy carried by the
        # validated result (never a mutable callback).  ``None`` means the
        # original read was strict D2A and the fresh re-read stays strict.
        fresh = read_drizzle_checkpoint(
            output_dir,
            require_exact_versions=True,
            resolver=result.resolution_policy,
        )
        if fresh.generation != expected_generation:
            raise DrizzleCheckpointError(
                f"stale continuation result: supplied generation "
                f"{expected_generation} != on-disk generation "
                f"{fresh.generation}; re-arm refused (another writer may have "
                "already continued)"
            )

        writer = object.__new__(cls)
        writer.output_dir = output_dir
        writer.product_version = str(fresh.config.product_version)
        writer.canonical_cfg = fresh.config
        writer.output_wcs = fresh.wcs
        writer.output_shape_hw = tuple(fresh.output_shape_hw)
        writer._dir = os.path.join(writer.output_dir, CHECKPOINT_DIRNAME)

        # Bind the canonical identity from the *fresh* config/WCS and re-check
        # it equals the fresh manifest (defense-in-depth; the reader already
        # validated digest / fingerprint / scientific_config / WCS).
        writer._bind_canonical_identity()
        writer._verify_bound_identity(fresh)

        writer._current_generation = int(fresh.generation)
        writer._next_generation = int(fresh.generation) + 1
        writer._manifest_tmp_path = None
        writer._continuation_state = writer._build_continuation_state(fresh)

        return DrizzleContinuation(
            writer=writer,
            accumulators=fresh.accumulators,
            support_accumulators=fresh.support_accumulators,
            session=copy.deepcopy(fresh.session),
            counters=copy.deepcopy(fresh.counters),
            completed_sources=copy.deepcopy(fresh.completed_sources),
            rejected_sources=copy.deepcopy(fresh.rejected_sources),
            generation=int(fresh.generation),
            next_source_index=int(fresh.next_source_index),
        )

    def _verify_bound_identity(self, fresh):
        """Fail closed if the bound identity diverges from the fresh manifest."""
        if self.run_config_digest != fresh.manifest["run_config_digest"]:
            raise DrizzleCheckpointError(
                "continuation run_config_digest diverges from the loaded "
                "manifest"
            )
        if self.fingerprint != fresh.manifest["scientific_fingerprint"]:
            raise DrizzleCheckpointError(
                "continuation scientific fingerprint diverges from the loaded "
                "manifest"
            )
        if self.scientific_config != fresh.manifest["scientific_config"]:
            raise DrizzleCheckpointError(
                "continuation scientific_config diverges from the loaded "
                "manifest"
            )
        if self._wcs_dict != fresh.manifest["wcs"]:
            raise DrizzleCheckpointError(
                "continuation output WCS diverges from the loaded manifest"
            )

    def _build_continuation_state(self, fresh):
        """Build the monotonic continuation baseline from a fresh read."""
        per_channel_total = [
            float(getattr(acc, "_total_exptime", 0.0))
            for acc in fresh.accumulators
        ]
        support = fresh.support_accumulators
        support_wht = None
        if support is not None:
            support_wht = tuple(
                np.array(acc._out_wht, dtype=np.float32, copy=True)
                for acc in support
            )
        return {
            "generation": int(fresh.generation),
            "session": copy.deepcopy(fresh.session),
            "counters": copy.deepcopy(fresh.counters),
            "completed": copy.deepcopy(fresh.completed_sources),
            "rejected": copy.deepcopy(fresh.rejected_sources),
            "channel_total_exptime": per_channel_total,
            "support_wht": support_wht,
        }

    # ------------------------------------------------------------------ state
    @property
    def has_committed(self) -> bool:
        return self._current_generation > 0

    @property
    def current_generation(self) -> int:
        return self._current_generation

    @property
    def next_generation(self) -> int:
        return self._next_generation

    def _artifact_name(self, generation: int, channel: int, kind: str) -> str:
        return f"gen-{int(generation):08d}-ch{int(channel)}-out_{kind}.npy"

    def _support_artifact_name(self, generation: int, kind: str) -> str:
        return f"gen-{int(generation):08d}-support_{kind}.npy"

    # ------------------------------------------------------------ restart
    def _refuse_existing_checkpoint(self):
        """Refuse a non-empty existing checkpoint namespace (fail closed).

        A fresh writer must not collide with a prior run: any prior manifest, allowlisted
        generation artifact, manifest temp or writer temp means a fresh writer
        would collide with / could destroy a prior run's state.  An empty
        existing directory is allowed; every pre-existing byte is preserved.
        """
        if not os.path.exists(self._dir):
            return
        if not os.path.isdir(self._dir):
            raise DrizzleCheckpointError(
                f"Drizzle checkpoint namespace {self._dir!r} exists but is not "
                "a directory"
            )
        try:
            entries = os.listdir(self._dir)
        except OSError as exc:
            raise DrizzleCheckpointError(
                f"cannot inspect Drizzle checkpoint namespace {self._dir!r}: {exc}"
            ) from exc
        if not entries:
            return
        found = []
        for name in entries:
            if name == MANIFEST_FILENAME:
                found.append("checkpoint.json")
            elif _is_manifest_temp(name):
                found.append("checkpoint temp")
            elif _ARTIFACT_RE.match(name):
                found.append("generation artifact")
            elif name.startswith(_ARRAY_TMP_PREFIX) and name.endswith(_ARRAY_TMP_SUFFIX):
                found.append("writer temp")
            else:
                found.append(f"unexpected entry {name!r}")
        raise DrizzleCheckpointError(
            "refusing to reuse a non-empty Drizzle checkpoint namespace "
            f"{self._dir!r} (found {', '.join(sorted(set(found)))}); use "
            "explicit Resume to continue this run, or use an empty output "
            "folder for a Fresh run"
        )

    # -------------------------------------------------------------- validation
    def _snapshot_channels(self, accumulators):
        """Own and validate the three native accumulator buffers.

        Returns a list of per-channel snapshot dicts, each holding owned float32
        copies (never aliased to the live engine buffers) plus the exact
        kernel/pixfrac/fillval/total_exptime.  Fail closed on any inconsistency.
        """
        if accumulators is None:
            accs = []
        else:
            if not isinstance(accumulators, (list, tuple)):
                raise DrizzleCheckpointError("accumulators must be a list")
            accs = list(accumulators)
        if len(accs) != 3:
            raise DrizzleCheckpointError(
                f"expected 3 drizzle accumulators, got {len(accs)}"
            )

        snapshots = []
        ref_total = None
        ref_kernel = None
        ref_pixfrac = None
        ref_fillval = None
        for c, acc in enumerate(accs):
            if acc is None:
                raise DrizzleCheckpointError(f"accumulator channel {c} is None")
            shape = tuple(getattr(acc, "out_shape_hw", None) or ())
            if shape != self.output_shape_hw:
                raise DrizzleCheckpointError(
                    f"channel {c} shape {shape} != output_shape_hw "
                    f"{self.output_shape_hw}"
                )

            kernel = getattr(acc, "kernel", None)
            pixfrac = _strict_float(
                getattr(acc, "pixfrac", 1.0), f"channel {c} pixfrac"
            )
            fillval = getattr(acc, "fillval", None)
            total = _strict_float(
                getattr(acc, "_total_exptime", 0.0), f"channel {c} total_exptime"
            )
            if ref_kernel is None:
                ref_kernel, ref_pixfrac, ref_fillval, ref_total = (
                    kernel, pixfrac, fillval, total,
                )
            if kernel != ref_kernel or pixfrac != ref_pixfrac or fillval != ref_fillval:
                raise DrizzleCheckpointError(
                    f"inconsistent per-channel drizzle config at channel {c}"
                )
            if total < 0.0:
                raise DrizzleCheckpointError(
                    f"negative total_exptime {total!r} at channel {c}"
                )
            if total != ref_total:
                raise DrizzleCheckpointError(
                    f"inconsistent per-channel total_exptime at channel {c}"
                )

            img = self._owned_float32_buffer(
                getattr(acc, "_out_img", None), f"channel {c} out_img"
            )
            wht = self._owned_float32_buffer(
                getattr(acc, "_out_wht", None), f"channel {c} out_wht"
            )
            snapshots.append(
                {
                    "channel": c,
                    "kernel": kernel,
                    "pixfrac": pixfrac,
                    "fillval": fillval,
                    "total_exptime": total,
                    "out_img": img,
                    "out_wht": wht,
                }
            )
        # Writer-side preflight (validation hardening, not a protocol redesign):
        # D1 must never publish accumulator runtime deposition parameters that
        # disagree with the canonical run_config.cfg scientific fields.  Runs
        # before any artifact / checkpoint dir / run_config creation.
        _check_deposition_matches_canonical(
            ref_kernel, ref_pixfrac, ref_fillval, self.scientific_config,
            "accumulator",
        )
        return snapshots

    def _snapshot_support(self, support_accumulators, frame_count):
        """Own and validate optional positive-support accumulator state.

        ``None`` is the explicit legacy/no-support state.  Otherwise exactly
        two square-kernel accumulators are required, representing SUP_W1 and
        SUP_W2 through their native WHT buffers.  Both buffers are snapshotted
        before any generation artifact is written.
        """
        if support_accumulators is None:
            return None
        if not isinstance(support_accumulators, (list, tuple)):
            raise DrizzleCheckpointError(
                "support_accumulators must be a two-element list/tuple or None"
            )
        support = list(support_accumulators)
        if len(support) != 2:
            raise DrizzleCheckpointError(
                f"expected 2 support accumulators, got {len(support)}"
            )
        snapshots = {}
        total_exptime = None
        fillval = None
        for kind, acc in zip(("w1", "w2"), support):
            if acc is None:
                raise DrizzleCheckpointError(
                    f"support accumulator {kind} is None"
                )
            shape = tuple(getattr(acc, "out_shape_hw", None) or ())
            if shape != self.output_shape_hw:
                raise DrizzleCheckpointError(
                    f"support accumulator {kind} shape {shape} != "
                    f"output_shape_hw {self.output_shape_hw}"
                )
            if getattr(acc, "kernel", None) != "square":
                raise DrizzleCheckpointError(
                    f"support accumulator {kind} must use square kernel"
                )
            pixfrac = _strict_float(
                getattr(acc, "pixfrac", None),
                f"support accumulator {kind} pixfrac",
            )
            if pixfrac != 1.0:
                raise DrizzleCheckpointError(
                    f"support accumulator {kind} pixfrac {pixfrac} != 1.0"
                )
            current_fillval = _validate_fillval(
                getattr(acc, "fillval", None),
                f"support accumulator {kind} fillval",
            )
            current_total = _strict_float(
                getattr(acc, "_total_exptime", None),
                f"support accumulator {kind} total_exptime",
            )
            if current_total != float(frame_count):
                raise DrizzleCheckpointError(
                    f"support accumulator {kind} total_exptime {current_total} "
                    f"!= frame_count {frame_count}"
                )
            if total_exptime is None:
                total_exptime = current_total
                fillval = current_fillval
            elif current_total != total_exptime or current_fillval != fillval:
                raise DrizzleCheckpointError(
                    "inconsistent support accumulator configuration"
                )
            arr = self._owned_float32_buffer(
                getattr(acc, "_out_wht", None),
                f"support accumulator {kind} out_wht",
            )
            if tuple(arr.shape) != self.output_shape_hw:
                raise DrizzleCheckpointError(
                    f"support accumulator {kind} array shape {arr.shape} != "
                    f"output_shape_hw {self.output_shape_hw}"
                )
            if np.any(arr < 0.0):
                raise DrizzleCheckpointError(
                    f"support accumulator {kind} contains negative samples"
                )
            snapshots[kind] = arr
        return {
            "schema_version": 1,
            "kernel": "square",
            "pixfrac": 1.0,
            "fillval": fillval,
            "total_exptime": total_exptime,
            "w1": snapshots["w1"],
            "w2": snapshots["w2"],
        }

    @staticmethod
    def _owned_float32_buffer(buf, name):
        """Validate and own a native float32 (H, W) buffer.

        Returns a private float32 copy (never aliased), so later engine
        mutations cannot race the file output.  Fails closed on wrong dtype /
        ndim / shape / non-finite samples.
        """
        arr = np.asarray(buf)
        if arr.dtype != np.float32:
            raise DrizzleCheckpointError(f"{name} must be float32, got {arr.dtype}")
        if arr.ndim != 2:
            raise DrizzleCheckpointError(f"{name} must be 2-D, got ndim={arr.ndim}")
        if not np.all(np.isfinite(arr)):
            raise DrizzleCheckpointError(f"{name} contains non-finite samples")
        return np.array(arr, dtype=np.float32, copy=True)

    @classmethod
    def _validate_counters(cls, counters):
        """Validate and canonicalize the accepted-exposure counters.

        Counters are validated strictly: integral fields must be genuine
        non-bool integers (no float/string truncation or coercion) and exposure
        values must be finite.  Enforces ``frame_count > 0``, non-negative
        counts, finite non-negative total exposure,
        ``exposure_unknown_count <= frame_count`` and
        ``exposure_min <= exposure_max`` when both are present (legitimate
        unknown-exposure runs may omit min/max).

        ``plan_cursor`` (optional) is the number of plan sources whose
        disposition is final (accepted *or* rejected).  When absent it defaults
        to ``frame_count`` (the legacy prefix-only contract).
        """
        if not isinstance(counters, dict):
            raise DrizzleCheckpointError("counters must be a mapping")
        frame_count = _strict_int(counters.get("frame_count", 0), "frame_count")
        if frame_count < 0:
            raise DrizzleCheckpointError(
                "negative frame_count"
            )
        plan_cursor = _strict_int(
            counters.get("plan_cursor", frame_count), "plan_cursor"
        )
        if plan_cursor < frame_count:
            raise DrizzleCheckpointError(
                f"plan_cursor {plan_cursor} < frame_count {frame_count}"
            )
        if frame_count == 0 and plan_cursor == 0:
            raise DrizzleCheckpointError(
                "refusing to publish an empty checkpoint (no accepted frames "
                "and no disposed sources)"
            )
        stacked = _strict_int(
            counters.get("stacked_batches_count", 0), "stacked_batches_count"
        )
        if stacked < 0:
            raise DrizzleCheckpointError("negative stacked_batches_count")
        total_exp = _strict_float(
            counters.get("total_exposure_seconds", 0.0),
            "total_exposure_seconds",
        )
        if total_exp < 0.0:
            raise DrizzleCheckpointError(
                f"negative total_exposure_seconds {total_exp!r}"
            )
        unknown = _strict_int(
            counters.get("exposure_unknown_count", 0), "exposure_unknown_count"
        )
        if unknown < 0:
            raise DrizzleCheckpointError("negative exposure_unknown_count")
        if unknown > frame_count:
            raise DrizzleCheckpointError(
                f"exposure_unknown_count {unknown} > frame_count {frame_count}"
            )

        exp_min = _strict_float(
            counters.get("exposure_min", None), "exposure_min", allow_none=True
        )
        exp_max = _strict_float(
            counters.get("exposure_max", None), "exposure_max", allow_none=True
        )
        if exp_min is not None and exp_max is not None and exp_min > exp_max:
            raise DrizzleCheckpointError(
                f"exposure_min {exp_min} > exposure_max {exp_max}"
            )

        return {
            "frame_count": frame_count,
            "plan_cursor": plan_cursor,
            "stacked_batches_count": stacked,
            "total_exposure_seconds": total_exp,
            "exposure_unknown_count": unknown,
            "exposure_min": exp_min,
            "exposure_max": exp_max,
        }

    @classmethod
    def _validate_session_binding(cls, session_binding):
        """Strictly validate/canonicalize the session binding (roots/ref/plan).

        Every source identity (reference and plan) is canonicalized through
        :func:`_validate_identity` and plan identities must be unique.  Any
        malformed nested value raises :class:`DrizzleCheckpointError` before
        any artifact is written.
        """
        sb = session_binding if session_binding is not None else {}
        if not isinstance(sb, dict):
            raise DrizzleCheckpointError("session_binding must be a mapping")
        roots = sb.get("input_roots")
        if not isinstance(roots, list) or not roots:
            raise DrizzleCheckpointError("missing session input_roots")
        roots_clean = []
        for r in roots:
            if not isinstance(r, str) or not r:
                raise DrizzleCheckpointError(
                    "session input_roots entries must be non-empty strings"
                )
            roots_clean.append(r)

        reference = sb.get("reference")
        ref_clean = _validate_identity(reference, "session reference")

        plan = sb.get("plan")
        if not isinstance(plan, dict):
            raise DrizzleCheckpointError("missing session observation plan")
        sources = plan.get("sources")
        if not isinstance(sources, list):
            raise DrizzleCheckpointError("session observation plan sources must be a list")
        if not sources:
            raise DrizzleCheckpointError("session observation plan is empty")
        sources_clean = []
        seen = set()
        for entry in sources:
            ident = _validate_identity(entry, "session plan source")
            key = (ident["path"], ident["size"], ident["mtime_ns"])
            if key in seen:
                raise DrizzleCheckpointError(
                    f"duplicate source identity in session plan: {ident['name']}"
                )
            seen.add(key)
            sources_clean.append(ident)

        plan_clean = {"sources": sources_clean}
        decomposition = plan.get("decomposition")
        if decomposition is not None:
            if not isinstance(decomposition, list):
                raise DrizzleCheckpointError("session plan decomposition must be a list")
            deco_clean = []
            for b in decomposition:
                bi = _strict_int(b, "session plan decomposition element")
                if bi <= 0:
                    raise DrizzleCheckpointError(
                        "session plan decomposition elements must be positive"
                    )
                deco_clean.append(bi)
            plan_clean["decomposition"] = deco_clean

        return {
            "input_roots": roots_clean,
            "reference": ref_clean,
            "plan": plan_clean,
            "reference_geometry": _validate_input_reference_geometry(
                sb.get("reference_geometry")
            ),
        }

    @classmethod
    def _validate_ledger(cls, completed_sources):
        """Validate the completed-source ledger (strict identities, unique)."""
        if completed_sources is None:
            ledger = []
        else:
            if not isinstance(completed_sources, (list, tuple)):
                raise DrizzleCheckpointError("completed_sources must be a list")
            ledger = list(completed_sources)
        clean = []
        seen = set()
        for entry in ledger:
            ident = _validate_identity(entry, "completed ledger")
            key = (ident["path"], ident["size"], ident["mtime_ns"])
            if key in seen:
                raise DrizzleCheckpointError(
                    f"duplicate source identity in completed ledger: {ident['name']}"
                )
            seen.add(key)
            clean.append(ident)
        return clean

    @classmethod
    def _validate_rejected_ledger(cls, rejected_sources):
        """Validate the rejected-source disposition ledger (strict, unique)."""
        if rejected_sources is None:
            return []
        if not isinstance(rejected_sources, (list, tuple)):
            raise DrizzleCheckpointError("rejected_sources must be a list")
        clean = []
        seen = set()
        for entry in rejected_sources:
            ident = _validate_identity(entry, "rejected disposition ledger")
            key = (ident["path"], ident["size"], ident["mtime_ns"])
            if key in seen:
                raise DrizzleCheckpointError(
                    f"duplicate source identity in rejected ledger: "
                    f"{ident['name']}"
                )
            seen.add(key)
            clean.append(ident)
        return clean

    @staticmethod
    def _validate_manifest_consistency(
        counters_clean, session_clean, ledger_clean, rejected_clean=None
    ):
        """Enforce the self-consistent manifest ledger/plan/counter invariant.

        Under the current Drizzle runtime ``stacked_batches_count`` increments
        once per accepted pose, so it must equal ``frame_count``.  The
        disposition partition invariant (rejection-aware):

        * ``plan_cursor`` is the number of plan sources with a final
          disposition (accepted or rejected);
        * ``plan[0:plan_cursor]`` is exactly the plan-ordered interleaving of
          the completed (accepted science) ledger and the rejected ledger;
        * neither ledger may contain a source that reappears in the remaining
          plan suffix ``plan[plan_cursor:]`` (no double-processing);
        * a source can never be both accepted and rejected.

        The legacy prefix-only contract is the exact special case with an
        empty rejected ledger and ``plan_cursor == frame_count``.
        """
        frame_count = counters_clean["frame_count"]
        plan_cursor = counters_clean["plan_cursor"]
        stacked = counters_clean["stacked_batches_count"]
        if stacked != frame_count:
            raise DrizzleCheckpointError(
                f"stacked_batches_count {stacked} != frame_count {frame_count}"
            )
        plan_sources = session_clean["plan"]["sources"]
        if len(ledger_clean) != frame_count:
            raise DrizzleCheckpointError(
                f"completed_sources length {len(ledger_clean)} != frame_count "
                f"{frame_count}"
            )
        if frame_count > len(plan_sources):
            raise DrizzleCheckpointError(
                f"frame_count {frame_count} exceeds session plan length "
                f"{len(plan_sources)}"
            )
        rejected_clean = rejected_clean or []
        if plan_cursor > len(plan_sources):
            raise DrizzleCheckpointError(
                f"plan_cursor {plan_cursor} exceeds session plan length "
                f"{len(plan_sources)}"
            )
        if plan_cursor != len(ledger_clean) + len(rejected_clean):
            raise DrizzleCheckpointError(
                f"plan_cursor {plan_cursor} != completed "
                f"{len(ledger_clean)} + rejected {len(rejected_clean)}"
            )
        rejected_keys = {
            (e["path"], e["size"], e["mtime_ns"]) for e in rejected_clean
        }
        completed_keys = {
            (e["path"], e["size"], e["mtime_ns"]) for e in ledger_clean
        }
        reference = session_clean["reference"]
        reference_key = (
            reference["path"], reference["size"], reference["mtime_ns"]
        )
        if reference_key in rejected_keys:
            raise DrizzleCheckpointError(
                "the session reference observation cannot be rejected: a "
                "disposed reference would make the alignment reference "
                "unresolvable on Resume"
            )
        if rejected_keys & completed_keys:
            raise DrizzleCheckpointError(
                "a source identity is both accepted and rejected"
            )
        if not rejected_clean and not identity_lists_equal(
            ledger_clean, plan_sources[:frame_count]
        ):
            raise DrizzleCheckpointError(
                "completed_sources is not the exact ordered prefix of the "
                "session plan"
            )
        for ident in plan_sources[plan_cursor:]:
            key = (ident["path"], ident["size"], ident["mtime_ns"])
            if key in rejected_keys:
                raise DrizzleCheckpointError(
                    f"rejected source {ident['name']} reappears in the "
                    "remaining session plan"
                )
            if key in completed_keys:
                raise DrizzleCheckpointError(
                    f"completed source {ident['name']} reappears in the "
                    "remaining session plan"
                )
        # Plan-ordered interleaving walk over the final-disposition prefix.
        accepted_ptr = 0
        rejected_ptr = 0
        for plan_index, ident in enumerate(plan_sources[:plan_cursor]):
            key = (ident["path"], ident["size"], ident["mtime_ns"])
            if (
                accepted_ptr < len(ledger_clean)
                and key
                == (
                    ledger_clean[accepted_ptr]["path"],
                    ledger_clean[accepted_ptr]["size"],
                    ledger_clean[accepted_ptr]["mtime_ns"],
                )
            ):
                accepted_ptr += 1
            elif (
                rejected_ptr < len(rejected_clean)
                and key
                == (
                    rejected_clean[rejected_ptr]["path"],
                    rejected_clean[rejected_ptr]["size"],
                    rejected_clean[rejected_ptr]["mtime_ns"],
                )
            ):
                rejected_ptr += 1
            else:
                raise DrizzleCheckpointError(
                    f"plan source at index {plan_index} has no matching "
                    "accepted/rejected disposition "
                    f"({ident['name']}); completed_sources must be an ordered "
                    "subsequence of the session plan and rejected_sources must "
                    "match the disposed plan positions"
                )
        if accepted_ptr != len(ledger_clean) or rejected_ptr != len(
            rejected_clean
        ):
            raise DrizzleCheckpointError(
                "disposition ledgers do not exhaust the plan prefix "
                f"(completed {accepted_ptr}/{len(ledger_clean)}, rejected "
                f"{rejected_ptr}/{len(rejected_clean)})"
            )

    def _check_monotonic_extension(self, counters_clean, session_clean,
                                   ledger_clean, snapshots, support_snapshot,
                                   rejected_clean=None):
        """Enforce monotonic continuation for a re-armed writer.

        No-op for a fresh-run writer (``_continuation_state is None``).  For a
        continuation writer (created only via
        :meth:`DrizzleCheckpointWriter.from_validated_result`), the next commit
        must *extend* the loaded checkpoint — never roll back, rewrite, reorder
        or diverge from it.  This covers **cumulative truth**, not just
        ``frame_count`` / ``total_exposure_seconds``:

        * the session binding (input roots / reference / ordered plan) must be
          identical to the loaded checkpoint;
        * ``frame_count`` must strictly increase (strictly longer prefix);
        * ``total_exposure_seconds`` must not roll back;
        * ``exposure_unknown_count`` must not decrease;
        * a known loaded ``exposure_min`` must not increase nor disappear, and
          a known loaded ``exposure_max`` must not decrease nor disappear (a
          loaded ``None`` may still become known once later known frames
          arrive);
        * every channel's native ``total_exptime`` must strictly increase (frame
          count grows) and never roll back;
        * positive support must remain either present or legacy-absent for the
          whole run, and present SUP_W1/SUP_W2 must never decrease;
        * the completed ledger must keep the loaded ledger as its exact prefix.

        Rejection-aware extension (RJK): a continuation whose only delta is
        newly-finalized *rejected* dispositions (``frame_count`` unchanged) is
        legal only when **every** scientific byte is proven unchanged — same
        ledger, same exposure counters, same per-channel totals, byte-identical
        support — and ``plan_cursor`` strictly advances with the loaded
        rejected ledger preserved as an exact prefix.  Any accepted frame
        always increases ``frame_count``, so no accepted science can hide in a
        cursor-only commit.

        This runs *before* any write, inside the commit try-block, so a
        divergent continuation is refused with the previous committed
        generation (and every file it references) byte-identical.
        """
        if self._continuation_state is None:
            return
        loaded = self._continuation_state
        loaded_session = loaded["session"]
        # The core session binding (input roots / reference / plan) must be
        # byte-identical.  The additive, versioned input-reference geometry is
        # handled separately below: it must match when both sides carry it, and
        # when the runtime binding omits it (a resume path that re-solves the
        # reference) the loaded geometry is carried forward for the same run so
        # the persisted geometry never drifts.  This preserves the historical
        # continuation semantics for checkpoints written before the payload
        # existed.
        core_new = {
            k: v for k, v in session_clean.items() if k != "reference_geometry"
        }
        core_loaded = {
            k: v for k, v in loaded_session.items() if k != "reference_geometry"
        }
        if core_new != core_loaded:
            raise DrizzleCheckpointError(
                "continuation session binding diverges from the loaded "
                "checkpoint (input_roots/reference/plan must be identical)"
            )
        new_geom = session_clean.get("reference_geometry")
        loaded_geom = loaded_session.get("reference_geometry")
        if new_geom is not None and loaded_geom is not None and new_geom != loaded_geom:
            raise DrizzleCheckpointError(
                "continuation input-reference geometry diverges from the "
                "loaded checkpoint"
            )
        if new_geom is None and loaded_geom is not None:
            session_clean["reference_geometry"] = loaded_geom
        if session_clean.get("reference_geometry") is None:
            raise DrizzleCheckpointError(
                "continuation must not omit the mandatory input-reference "
                "geometry"
            )
        loaded_counters = loaded["counters"]
        loaded_ledger = loaded["completed"]
        loaded_rejected = list(loaded.get("rejected") or [])
        new_frame = counters_clean["frame_count"]
        loaded_frame = loaded_counters["frame_count"]
        new_cursor = counters_clean["plan_cursor"]
        loaded_cursor = loaded_counters.get("plan_cursor", loaded_frame)
        if new_frame < loaded_frame:
            raise DrizzleCheckpointError(
                f"continuation must extend the loaded checkpoint: frame_count "
                f"{new_frame} < loaded frame_count {loaded_frame}"
            )
        rejected_clean = rejected_clean or []
        if new_frame == loaded_frame:
            # Cursor-only extension: a rejection disposition commit.  Science
            # must be byte-proven unchanged; only the plan cursor / rejected
            # ledger may advance.
            if not identity_lists_equal(ledger_clean, loaded_ledger):
                raise DrizzleCheckpointError(
                    "cursor-only continuation must preserve the exact loaded "
                    "completed ledger (no rewrite/reorder/divergence)"
                )
            if new_cursor <= loaded_cursor:
                raise DrizzleCheckpointError(
                    f"cursor-only continuation must advance plan_cursor "
                    f"({new_cursor} <= loaded {loaded_cursor})"
                )
            if not identity_lists_equal(
                rejected_clean[: len(loaded_rejected)], loaded_rejected
            ):
                raise DrizzleCheckpointError(
                    "continuation rejected_sources must preserve the exact "
                    "loaded rejected ledger prefix"
                )
            if len(rejected_clean) <= len(loaded_rejected):
                raise DrizzleCheckpointError(
                    "cursor-only continuation must extend the rejected ledger"
                )
            for field in (
                "total_exposure_seconds",
                "exposure_unknown_count",
                "exposure_min",
                "exposure_max",
            ):
                if counters_clean[field] != loaded_counters[field]:
                    raise DrizzleCheckpointError(
                        f"cursor-only continuation must not change "
                        f"{field} ({counters_clean[field]!r} != "
                        f"{loaded_counters[field]!r})"
                    )
            if counters_clean["stacked_batches_count"] != loaded_counters[
                "stacked_batches_count"
            ]:
                raise DrizzleCheckpointError(
                    "cursor-only continuation must not change "
                    "stacked_batches_count"
                )
            loaded_totals = loaded["channel_total_exptime"]
            new_totals = [float(s["total_exptime"]) for s in snapshots]
            if len(new_totals) != len(loaded_totals) or any(
                new_t != loaded_t for new_t, loaded_t in zip(new_totals, loaded_totals)
            ):
                raise DrizzleCheckpointError(
                    "cursor-only continuation per-channel total_exptime must "
                    "be byte-unchanged"
                )
            loaded_wht = loaded.get("support_wht")
            if (loaded_wht is None) != (support_snapshot is None):
                raise DrizzleCheckpointError(
                    "cursor-only continuation positive-support availability "
                    "changed"
                )
            if loaded_wht is not None:
                for kind, previous in zip(("w1", "w2"), loaded_wht):
                    if not np.array_equal(support_snapshot[kind], previous):
                        raise DrizzleCheckpointError(
                            f"cursor-only continuation support {kind} must be "
                            "byte-unchanged"
                        )
            return
        # Accepted-frame extension: strict monotonic growth on every axis.
        self._check_cumulative_counters(loaded_counters, counters_clean)
        self._check_channel_total_monotonic(loaded, snapshots)
        self._check_support_monotonic(loaded, support_snapshot)

        if not identity_lists_equal(
            ledger_clean[: len(loaded_ledger)], loaded_ledger
        ):
            raise DrizzleCheckpointError(
                "continuation completed_sources must preserve the exact loaded "
                "ledger prefix (no rewrite/reorder/divergent prefix)"
            )
        if not identity_lists_equal(
            rejected_clean[: len(loaded_rejected)], loaded_rejected
        ):
            raise DrizzleCheckpointError(
                "continuation rejected_sources must preserve the exact loaded "
                "rejected ledger prefix"
            )
        if new_cursor < loaded_cursor:
            raise DrizzleCheckpointError(
                f"continuation plan_cursor must not decrease "
                f"({new_cursor} < {loaded_cursor})"
            )

    def _check_cumulative_counters(self, loaded_counters, counters_clean):
        """Enforce cumulative unknown/known exposure arithmetic (D2B1 finding 2).

        With ``delta_frame = new_frame - loaded_frame`` (> 0, already enforced)
        and ``delta_unknown = new_unknown - loaded_unknown``, the cumulative
        unknown count may only grow by counting *new* frames — never by
        retroactively reclassifying already-committed frames:

        * ``0 <= delta_unknown <= delta_frame``;
        * ``known_added = delta_frame - delta_unknown``;
        * if ``known_added == 0`` (every new frame is unknown) then
          ``total_exposure_seconds`` / ``exposure_min`` / ``exposure_max`` must
          remain *exactly* unchanged (including ``None``) — no fabricating or
          rewriting the cumulative known-exposure summary;
        * if ``known_added > 0`` then ``total_exposure_seconds`` must strictly
          increase and ``exposure_min`` / ``exposure_max`` must be known after
          the commit.

        Known loaded ``exposure_min`` / ``exposure_max`` still obey their own
        monotonic rules (:meth:`_check_exposure_minmax_monotonic`), enforced at
        the end regardless of ``known_added``.
        """
        delta_frame = (
            counters_clean["frame_count"] - loaded_counters["frame_count"]
        )
        delta_unknown = (
            counters_clean["exposure_unknown_count"]
            - loaded_counters["exposure_unknown_count"]
        )
        if delta_unknown < 0:
            raise DrizzleCheckpointError(
                "continuation exposure_unknown_count must not decrease "
                f"(delta {delta_unknown})"
            )
        if delta_unknown > delta_frame:
            raise DrizzleCheckpointError(
                "continuation exposure_unknown_count cannot grow by more than "
                f"the new frames: delta_unknown {delta_unknown} > delta_frame "
                f"{delta_frame} (retroactive inflation refused)"
            )
        known_added = delta_frame - delta_unknown
        if known_added == 0:
            if (
                counters_clean["total_exposure_seconds"]
                != loaded_counters["total_exposure_seconds"]
            ):
                raise DrizzleCheckpointError(
                    "continuation total_exposure_seconds must be unchanged when "
                    "no known frames are added (all new frames unknown)"
                )
            if counters_clean["exposure_min"] != loaded_counters["exposure_min"]:
                raise DrizzleCheckpointError(
                    "continuation exposure_min must be unchanged when no known "
                    "frames are added (all new frames unknown)"
                )
            if counters_clean["exposure_max"] != loaded_counters["exposure_max"]:
                raise DrizzleCheckpointError(
                    "continuation exposure_max must be unchanged when no known "
                    "frames are added (all new frames unknown)"
                )
        else:
            if (
                counters_clean["total_exposure_seconds"]
                <= loaded_counters["total_exposure_seconds"]
            ):
                raise DrizzleCheckpointError(
                    "continuation total_exposure_seconds must strictly increase "
                    "when known frames are added "
                    f"({counters_clean['total_exposure_seconds']} <= "
                    f"{loaded_counters['total_exposure_seconds']})"
                )
            if (
                counters_clean["exposure_min"] is None
                or counters_clean["exposure_max"] is None
            ):
                raise DrizzleCheckpointError(
                    "continuation exposure_min/exposure_max must be known when "
                    "known frames are committed"
                )
        self._check_exposure_minmax_monotonic(
            loaded_counters, counters_clean
        )

    @staticmethod
    def _check_exposure_minmax_monotonic(loaded_counters, new_counters):
        """Cumulative min/max exposure monotonicity (known values only).

        A known loaded ``exposure_min`` can only stay or decrease and must not
        disappear; a known loaded ``exposure_max`` can only stay or increase and
        must not disappear.  A loaded ``None`` may become known (or stay None)
        once later known frames arrive — that transition is semantically legal
        and is not refused.
        """
        loaded_min = loaded_counters["exposure_min"]
        loaded_max = loaded_counters["exposure_max"]
        new_min = new_counters["exposure_min"]
        new_max = new_counters["exposure_max"]
        if loaded_min is not None:
            if new_min is None:
                raise DrizzleCheckpointError(
                    "continuation exposure_min must not disappear"
                )
            if new_min > loaded_min:
                raise DrizzleCheckpointError(
                    f"continuation exposure_min must not increase "
                    f"({new_min} > loaded {loaded_min})"
                )
        if loaded_max is not None:
            if new_max is None:
                raise DrizzleCheckpointError(
                    "continuation exposure_max must not disappear"
                )
            if new_max < loaded_max:
                raise DrizzleCheckpointError(
                    f"continuation exposure_max must not decrease "
                    f"({new_max} < loaded {loaded_max})"
                )

    def _check_channel_total_monotonic(self, loaded, snapshots):
        """Every channel's native ``total_exptime`` must strictly increase."""
        loaded_totals = loaded["channel_total_exptime"]
        new_totals = [float(s["total_exptime"]) for s in snapshots]
        if len(new_totals) != len(loaded_totals):
            raise DrizzleCheckpointError(
                "continuation per-channel total_exptime count changed "
                f"({len(new_totals)} != {len(loaded_totals)})"
            )
        for c, (new_t, loaded_t) in enumerate(zip(new_totals, loaded_totals)):
            if new_t <= loaded_t:
                raise DrizzleCheckpointError(
                    f"continuation channel {c} total_exptime must strictly "
                    f"increase (got {new_t} <= loaded {loaded_t})"
                )

    @staticmethod
    def _check_support_monotonic(loaded, support_snapshot):
        """Refuse support loss, fabrication, or cumulative rollback."""
        loaded_wht = loaded.get("support_wht")
        if (loaded_wht is None) != (support_snapshot is None):
            raise DrizzleCheckpointError(
                "continuation positive-support availability changed; legacy "
                "support cannot be fabricated and committed support cannot be "
                "dropped"
            )
        if loaded_wht is None:
            return
        for kind, previous in zip(("w1", "w2"), loaded_wht):
            current = support_snapshot[kind]
            if np.any(current < previous):
                raise DrizzleCheckpointError(
                    f"continuation support {kind} must not decrease"
                )

    def _build_next_continuation_state(self, generation, counters_clean,
                                       session_clean, ledger_clean, snapshots,
                                       support_snapshot, rejected_clean=None):
        """Build (deep-copied) the continuation baseline for the next commit.

        Pure and fallible: the deep copies and the per-channel total-exposure
        list are materialized here, during preflight, so any allocation failure
        (e.g. ``MemoryError``) is raised *before* any artifact write or manifest
        commit.  Returns the exact next ``_continuation_state`` dict, which the
        caller assigns by reference **only after** the manifest commits.
        """
        support_wht = None
        if support_snapshot is not None:
            support_wht = (
                np.array(support_snapshot["w1"], dtype=np.float32, copy=True),
                np.array(support_snapshot["w2"], dtype=np.float32, copy=True),
            )
        return {
            "generation": int(generation),
            "session": copy.deepcopy(session_clean),
            "counters": copy.deepcopy(counters_clean),
            "completed": copy.deepcopy(ledger_clean),
            "rejected": copy.deepcopy(rejected_clean or []),
            "channel_total_exptime": [
                float(s["total_exptime"]) for s in snapshots
            ],
            "support_wht": support_wht,
        }

    def _preflight_json_payload(self, counters_clean, session_clean, ledger_clean,
                                rejected_clean=None):
        """Preflight-serialize the non-artifact manifest payload (fail closed).

        Serializes every persisted field except the array descriptors with
        ``allow_nan=False`` so a non-finite / non-JSON value is refused *before*
        any artifact is created.
        """
        payload = {
            "schema_version": SCHEMA_VERSION,
            "mode": MODE_TOKEN,
            "state": STATE_CLEAN,
            "product_version": self.product_version,
            "producer": "zeseestarstacker",
            "output_shape_hw": list(self.output_shape_hw),
            "wcs": self._wcs_dict,
            "scientific_fingerprint": self.fingerprint,
            "scientific_config": self.scientific_config,
            "run_config_digest": self.run_config_digest,
            "frame_count": counters_clean["frame_count"],
            "stacked_batches_count": counters_clean["stacked_batches_count"],
            "total_exposure_seconds": counters_clean["total_exposure_seconds"],
            "exposure_unknown_count": counters_clean["exposure_unknown_count"],
            "exposure_min": counters_clean["exposure_min"],
            "exposure_max": counters_clean["exposure_max"],
            "session": session_clean,
            "completed_sources": ledger_clean,
        }
        if rejected_clean:
            payload["rejected_sources"] = rejected_clean
        if counters_clean["plan_cursor"] != counters_clean["frame_count"]:
            payload["plan_cursor"] = counters_clean["plan_cursor"]
        try:
            json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise DrizzleCheckpointError(
                f"checkpoint payload is not strict JSON: {exc}"
            ) from exc

    # ------------------------------------------------------------------ writes
    @staticmethod
    def _npy_bytes(arr):
        """Serialize a float32 array to the exact ``.npy`` file bytes."""
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue()

    def _write_array_artifact(self, arr, final_name):
        """Write one native array to a generation-unique final name.

        The final name is *claimed* exclusively with ``O_CREAT | O_EXCL``
        (never overwrites a pre-existing path) and written in place; the file
        is fsync'ed before the (later) manifest commit references it.  Returns
        the exact bytes written so the caller can record a SHA-256 / size over
        the final artifact itself.
        """
        data = self._npy_bytes(arr)
        path = os.path.join(self._dir, final_name)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            raise DrizzleCheckpointError(
                f"generation artifact {final_name!r} already exists; refusing "
                "to overwrite"
            ) from None
        except OSError as exc:
            raise DrizzleCheckpointError(
                f"cannot create generation artifact {final_name!r}: {exc}"
            ) from exc
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        except BaseException:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
        return data

    def _write_cfg(self):
        """Atomically persist the canonical run config (stable across commits)."""
        run_contract.write_cfg(
            self.canonical_cfg, os.path.join(self.output_dir, RUN_CONFIG_FILENAME)
        )

    def _write_manifest(self, manifest):
        """Write a uniquely-owned manifest temp, fsync, ``os.replace`` (commit).

        The manifest is serialized with ``allow_nan=False``, written to a
        per-attempt *owned* temp (see :meth:`_claim_manifest_temp`), fsync'ed,
        then atomically ``os.replace``-d onto ``checkpoint.json`` as the single
        commit point.  Returns ``True`` once committed (replace + directory
        fsync).  Any exception means the replace did not happen and the prior
        committed state is still authoritative; only this attempt's own temp is
        removed, never another writer's temp or artifact.
        """
        manifest_path = os.path.join(self._dir, MANIFEST_FILENAME)
        payload = json.dumps(
            manifest, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
        )
        tmp_path = self._claim_manifest_temp(payload)
        try:
            os.replace(tmp_path, manifest_path)
            self._manifest_tmp_path = None
            _fsync_dir(self._dir)
            return True
        except BaseException:
            # Remove only this attempt's owned temp; after a successful replace
            # the name no longer exists, so the unlink is a safe no-op.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            self._manifest_tmp_path = None
            raise

    def _claim_manifest_temp(self, payload: str) -> str:
        """Create and fsync a uniquely-owned manifest temp, returning its path.

        The temp name is ``checkpoint.json.tmp.<pid>.<seq>.<nonce>`` and is
        claimed **exclusively** via ``open(..., "x")`` (``O_CREAT | O_EXCL``),
        so it can never collide with a temp owned by another writer/attempt.
        The payload is written and fsync'ed before the path is returned; only
        this attempt may later ``os.replace`` it onto the manifest or unlink it.
        Raises :class:`DrizzleCheckpointError` on write failure or (extremely
        unlikely) exhaustion of unique names.
        """
        os.makedirs(self._dir, exist_ok=True)
        for _ in range(128):
            token = (
                f"{os.getpid()}.{next(_MANIFEST_TMP_COUNTER)}."
                f"{secrets.token_hex(4)}"
            )
            candidate = os.path.join(self._dir, MANIFEST_TMP_PREFIX + token)
            try:
                fh = open(candidate, "x", encoding="utf-8")
            except FileExistsError:
                continue
            except OSError as exc:
                raise DrizzleCheckpointError(
                    f"cannot create manifest temp {candidate!r}: {exc}"
                ) from exc
            try:
                with fh:
                    fh.write(payload)
                    fh.write("\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            except BaseException:
                try:
                    os.unlink(candidate)
                except OSError:
                    pass
                raise
            self._manifest_tmp_path = candidate
            return candidate
        raise DrizzleCheckpointError(
            f"could not claim a unique manifest temp in {self._dir!r}"
        )

    def _gc_stale_generations(self, current_generation):
        """Best-effort removal of older generations from the explicit allowlist.

        Only ``gen-*.npy`` names matching :data:`_ARTIFACT_RE` with a generation
        strictly older than ``current_generation`` are removed.  Never touches
        the manifest, the current generation, or any unrelated file.
        """
        try:
            names = os.listdir(self._dir)
        except OSError:
            return
        for name in names:
            m = _ARTIFACT_RE.match(name)
            if not m:
                continue
            gen = int(m.group(1))
            if gen < current_generation:
                try:
                    os.unlink(os.path.join(self._dir, name))
                except OSError:
                    pass

    def _cleanup_attempt(self, created_final_names):
        """Best-effort cleanup of this attempt's own uncommitted artifacts.

        Removes only the manifest temp owned by this attempt (tracked in
        ``_manifest_tmp_path``) and the generation final names actually created
        by this attempt.  Never touches a pre-existing path, a foreign manifest
        temp, or an artifact owned by another writer/attempt.
        """
        if self._manifest_tmp_path is not None:
            try:
                os.unlink(self._manifest_tmp_path)
            except OSError:
                pass
            self._manifest_tmp_path = None
        for name in created_final_names:
            try:
                os.unlink(os.path.join(self._dir, name))
            except OSError:
                pass

    # ------------------------------------------------------------------ commit
    def commit(self, accumulators, *, session_binding, counters,
               completed_sources, support_accumulators=None,
               rejected_sources=None):
        """Persist one generation and atomically commit the manifest.

        Parameters
        ----------
        accumulators :
            The three per-channel :class:`DrizzleAccumulator` instances.
        session_binding :
            ``{"input_roots": [...], "reference": {...}, "plan": {...}}``.
        counters :
            ``{"frame_count": int, "plan_cursor": int|absent,
            "stacked_batches_count": int,
            "total_exposure_seconds": float, "exposure_unknown_count": int,
            "exposure_min": float|None, "exposure_max": float|None}``.
            ``plan_cursor`` defaults to ``frame_count`` (legacy contract).
        completed_sources :
            Ordered ledger of accepted source identities (the accepted half of
            the plan-ordered disposition partition of length ``frame_count``).
        rejected_sources :
            Optional ordered ledger of rejected (disposed, non-admitted) source
            identities.  A rejected source never enters SCI/WHT/SUPPORT and
            never increments ``frame_count`` / ``stacked_batches_count``; the
            plan cursor (``counters["plan_cursor"]``) records its final
            disposition so Resume continues beyond it without replaying it.
        support_accumulators :
            Optional ``(SUP_W1, SUP_W2)`` positive-support accumulators.  When
            present their WHT buffers are committed by the same manifest as
            the native SCI/WHT generation.  ``None`` preserves an explicit
            legacy support-less run.

        Returns
        -------
        int
            The committed generation id.

        Raises
        ------
        DrizzleCheckpointError
            On any validation or persistence failure; the prior committed
            manifest (and every file it references) stays byte-identical.
        """
        # 0. Restart safety (first commit only): never write into a namespace
        #    that already holds a checkpoint from a prior writer/process.
        if not self.has_committed:
            self._refuse_existing_checkpoint()

        generation = int(self._next_generation)
        manifest_committed = False
        created_final_names = []
        next_continuation_state = None

        try:
            # 1. Validate everything *before* any write (fail closed, never a
            #    partial/mixed generation).  Validation lives inside the try so
            #    any malformed caller-provided value is converted to a
            #    DrizzleCheckpointError instead of leaking a raw
            #    AttributeError/TypeError/ValueError.
            counters_clean = self._validate_counters(counters)
            session_clean = self._validate_session_binding(session_binding)
            # F1: the v2 output-grid checkpoint mandates the frozen
            # input-reference geometry.  A fresh (non-continuation) commit
            # refuses to publish without it; a continuation may carry it
            # forward from the loaded checkpoint (handled in
            # ``_check_monotonic_extension``, which also refuses if neither
            # side supplies it).
            if (
                session_clean.get("reference_geometry") is None
                and self._continuation_state is None
            ):
                raise DrizzleCheckpointError(
                    "input-reference geometry is mandatory for the v2 "
                    "output-grid checkpoint; refusing to publish without it"
                )
            ledger_clean = self._validate_ledger(completed_sources)
            rejected_clean = self._validate_rejected_ledger(rejected_sources)
            snapshots = self._snapshot_channels(accumulators)
            support_snapshot = self._snapshot_support(
                support_accumulators, counters_clean["frame_count"]
            )

            # 2. Manifest self-consistency invariants (truthful
            #    ledger/plan/counter).
            self._validate_manifest_consistency(
                counters_clean, session_clean, ledger_clean, rejected_clean
            )

            # 2b. Continuation monotonicity: a re-armed writer must extend the
            #     loaded checkpoint (never roll back / rewrite / reorder /
            #     diverge / roll back cumulative counters or native per-channel
            #     total exposure).  No-op for a fresh-run writer.
            self._check_monotonic_extension(
                counters_clean, session_clean, ledger_clean, snapshots,
                support_snapshot, rejected_clean,
            )

            # 3. Preflight strict-JSON serialization of the non-artifact payload.
            self._preflight_json_payload(
                counters_clean, session_clean, ledger_clean, rejected_clean
            )

            # 3b. Build the exact next continuation baseline (deep copies)
            #     entirely during preflight, BEFORE any artifact write or
            #     manifest commit.  Any fallible allocation (e.g. MemoryError)
            #     therefore fails here — with generation N still byte-identical —
            #     instead of surfacing to the caller after the N+1 manifest is
            #     already authoritative.  No fallible continuation-state work
            #     may happen after `_write_manifest`.
            if self._continuation_state is not None:
                next_continuation_state = self._build_next_continuation_state(
                    generation, counters_clean, session_clean, ledger_clean,
                    snapshots, support_snapshot, rejected_clean,
                )

            os.makedirs(self._dir, exist_ok=True)
            # 4. Write the six native generation artifacts plus the optional
            #    positive-support pair (exclusive claims, never overwrite a
            #    pre-existing path).  Every artifact is durable before the one
            #    manifest commit point can reference it.
            written = []  # (channel, short, name, digest, size)
            for snap in snapshots:
                for kind in ("out_img", "out_wht"):
                    arr = snap[kind]
                    short = "img" if kind == "out_img" else "wht"
                    name = self._artifact_name(generation, snap["channel"], short)
                    file_bytes = self._write_array_artifact(arr, name)
                    created_final_names.append(name)
                    digest = hashlib.sha256(file_bytes).hexdigest()
                    written.append(
                        (
                            snap["channel"],
                            short,
                            name,
                            digest,
                            int(len(file_bytes)),
                        )
                    )
            support_written = {}
            if support_snapshot is not None:
                for kind in ("w1", "w2"):
                    name = self._support_artifact_name(generation, kind)
                    file_bytes = self._write_array_artifact(
                        support_snapshot[kind], name
                    )
                    created_final_names.append(name)
                    support_written[kind] = (
                        name,
                        hashlib.sha256(file_bytes).hexdigest(),
                        int(len(file_bytes)),
                    )
            _fsync_dir(self._dir)

            # 5. Persist the canonical config before the manifest; fsync the
            #    output directory so the run_config.cfg rename is durable.
            self._write_cfg()
            _fsync_dir(self.output_dir)

            # 6. Build the deterministic manifest.
            channels = []
            for snap in snapshots:
                c = snap["channel"]
                img_entry = next(
                    e for e in written if e[0] == c and e[1] == "img"
                )
                wht_entry = next(
                    e for e in written if e[0] == c and e[1] == "wht"
                )
                channels.append(
                    {
                        "channel": c,
                        "kernel": snap["kernel"],
                        "pixfrac": snap["pixfrac"],
                        "fillval": snap["fillval"],
                        "total_exptime": snap["total_exptime"],
                        "out_img": {
                            "file": img_entry[2],
                            "dtype": "float32",
                            "shape": list(self.output_shape_hw),
                            "size": img_entry[4],
                            "sha256": img_entry[3],
                        },
                        "out_wht": {
                            "file": wht_entry[2],
                            "dtype": "float32",
                            "shape": list(self.output_shape_hw),
                            "size": wht_entry[4],
                            "sha256": wht_entry[3],
                        },
                    }
                )

            manifest = {
                "schema_version": SCHEMA_VERSION,
                "mode": MODE_TOKEN,
                "state": STATE_CLEAN,
                "generation": generation,
                "product_version": self.product_version,
                "producer": "zeseestarstacker",
                "drizzle_lib_version": _drizzle_lib_version(),
                "numpy_version": _numpy_version(),
                "output_shape_hw": list(self.output_shape_hw),
                "wcs": self._wcs_dict,
                "scientific_fingerprint": self.fingerprint,
                "scientific_config": self.scientific_config,
                "run_config_digest": self.run_config_digest,
                "frame_count": counters_clean["frame_count"],
                "stacked_batches_count": counters_clean["stacked_batches_count"],
                "total_exposure_seconds": counters_clean["total_exposure_seconds"],
                "exposure_unknown_count": counters_clean["exposure_unknown_count"],
                "exposure_min": counters_clean["exposure_min"],
                "exposure_max": counters_clean["exposure_max"],
                "session": session_clean,
                "completed_sources": ledger_clean,
                "channels": channels,
            }
            if rejected_clean:
                manifest["rejected_sources"] = rejected_clean
            if counters_clean["plan_cursor"] != counters_clean["frame_count"]:
                manifest["plan_cursor"] = counters_clean["plan_cursor"]
            if support_snapshot is not None:
                manifest["support"] = {
                    "schema_version": support_snapshot["schema_version"],
                    "kernel": support_snapshot["kernel"],
                    "pixfrac": support_snapshot["pixfrac"],
                    "fillval": support_snapshot["fillval"],
                    "total_exptime": support_snapshot["total_exptime"],
                    "sup_w1": {
                        "file": support_written["w1"][0],
                        "dtype": "float32",
                        "shape": list(self.output_shape_hw),
                        "size": support_written["w1"][2],
                        "sha256": support_written["w1"][1],
                    },
                    "sup_w2": {
                        "file": support_written["w2"][0],
                        "dtype": "float32",
                        "shape": list(self.output_shape_hw),
                        "size": support_written["w2"][2],
                        "sha256": support_written["w2"][1],
                    },
                }

            # 7. Commit the manifest LAST (single commit point).  Once
            #    `_write_manifest` returns, `manifest_committed` is set and a
            #    later failure can never roll back the newly referenced files.
            manifest_committed = self._write_manifest(manifest)

        except BaseException as exc:
            # Best-effort cleanup of this attempt's OWN uncommitted files only,
            # and only if the manifest was not committed.  Never touches the
            # prior committed generation or an unrelated/pre-existing path.
            if not manifest_committed:
                self._cleanup_attempt(created_final_names)
            if isinstance(exc, DrizzleCheckpointError):
                raise
            raise DrizzleCheckpointError(
                f"drizzle checkpoint persist failed: {exc}"
            ) from exc

        # 8. Advance the generation and best-effort GC older generations.
        #    After a successful manifest commit only non-fallible scalar /
        #    reference assignments may occur: the next continuation baseline was
        #    already deep-copied during preflight, so it is adopted here by a
        #    single reference assignment.  GC remains strictly best-effort.
        self._current_generation = generation
        self._next_generation = generation + 1
        if next_continuation_state is not None:
            self._continuation_state = next_continuation_state
        self._gc_stale_generations(generation)
        return generation


# ---------------------------------------------------------------------------
# RSM2-D2A: read-only loader / validator (no Resume activation yet)
# ---------------------------------------------------------------------------
#
# Documented exact-continuation version policy: bit-identical native
# continuation requires the *same* drizzle and numpy rounding behaviour, so
# :func:`read_drizzle_checkpoint` refuses (fail closed) when the persisted
# ``drizzle_lib_version`` / ``numpy_version`` differ from the runtime library
# versions.  This is an intentional strict policy.  D2B may later relax it to
# an explicit WARN under a separately reviewed decision, but the D2A reader
# never silently continues across a library version boundary.


@dataclass(frozen=True)
class DrizzleCheckpointResult:
    """Validated read-only reconstruction of a native Drizzle checkpoint.

    Produced only after the *entire* checkpoint has validated (fail closed, no
    partial externally visible restore).  ``accumulators`` is a list of three
    :class:`~seestar.core.drizzle_core.DrizzleAccumulator` instances
    reconstructed via :meth:`DrizzleAccumulator.from_native_state`; ``wcs`` is
    the reconstructed :class:`astropy.wcs.WCS` with ``array_shape`` attached;
    ``next_source_index`` is the 0-based index of the first source not yet
    disposed (== ``plan_cursor`` — on a legacy prefix-only checkpoint this
    equals ``frame_count``).  Suitable for later D2B lifecycle wiring (which
    is *not* performed here).

    ``source_output_dir`` is the immutable, validated provenance of the exact
    output directory the checkpoint was read from (normalized to an absolute
    path in :meth:`__post_init__`).  It is the *only* path a continuation
    writer (:meth:`DrizzleCheckpointWriter.from_validated_result`) may bind to,
    so a validated result can never be re-armed against a different directory.
    The dataclass is frozen: no field (including ``source_output_dir``) can be
    reassigned after validation.
    """

    manifest: dict
    session: dict
    counters: dict
    completed_sources: list
    config: object          # run_contract.RunConfig
    wcs: object             # astropy.wcs.WCS (array_shape attached)
    output_shape_hw: tuple
    accumulators: list      # [DrizzleAccumulator x3]
    support_accumulators: object  # (SUP_W1, SUP_W2) | None for legacy
    next_source_index: int
    generation: int
    source_output_dir: str
    resolved_reference: object = None        # verified on-disk path (str) | None
    resolved_plan_paths: tuple = ()          # ordered verified plan paths
    resolved_completed_paths: tuple = ()     # resolved_plan_paths[:next_source_index]
    resolved_remaining_paths: tuple = ()     # resolved_plan_paths[next_source_index:]
    resolution_policy: object = None         # immutable SafeStackedSourceResolver | None
    reference_geometry: object = None        # versioned frozen input-reference geometry | None
    rejected_sources: object = None          # finalized rejected dispositions
    plan_cursor: int = 0                     # number of plan sources with a
                                             # final disposition

    def __post_init__(self):
        """Validate and normalize the source-output provenance (read-only).

        The provenance is bound to the **canonical real path** (``realpath``),
        not merely an absolute path: any symlink component in the supplied path
        (including a symlink root) is resolved once, here, so a later symlink
        retargeting cannot rebind this validated result to a different run's
        checkpoint directory.
        """
        d = self.source_output_dir
        if not isinstance(d, (str, os.PathLike)) or not os.fspath(d):
            raise DrizzleCheckpointError(
                "DrizzleCheckpointResult requires a non-empty source_output_dir"
            )
        d = os.path.realpath(os.fspath(d))
        if not os.path.isdir(d):
            raise DrizzleCheckpointError(
                f"DrizzleCheckpointResult source_output_dir {d!r} is not a "
                "directory"
            )
        object.__setattr__(self, "source_output_dir", d)
        plan = tuple(self.resolved_plan_paths or ())
        object.__setattr__(self, "resolved_plan_paths", plan)
        n = int(self.next_source_index)
        resolved_completed = [
            p for p in plan[:n] if p is not None
        ]
        resolved_remaining = [
            p for p in plan[n:] if p is not None
        ]
        object.__setattr__(self, "resolved_completed_paths", tuple(resolved_completed))
        object.__setattr__(self, "resolved_remaining_paths", tuple(resolved_remaining))
        if self.rejected_sources is None:
            object.__setattr__(self, "rejected_sources", ())
        else:
            object.__setattr__(self, "rejected_sources", tuple(self.rejected_sources))


@dataclass(frozen=True)
class DrizzleContinuation:
    """Unambiguous D2B1 re-arm result: a fresh writer + fresh disk state.

    Produced only by
    :meth:`DrizzleCheckpointWriter.from_validated_result`, which **freshly**
    re-reads and re-validates the on-disk checkpoint (never trusting the
    shallow-frozen mutable payloads of the supplied
    :class:`DrizzleCheckpointResult`).  The lifecycle must continue by mutating
    ``accumulators`` (the freshly reconstructed native buffers) and then
    calling ``writer.commit(...)`` with the freshly loaded ``session`` /
    ``counters`` / ``completed_sources`` extended for the new frames.  It must
    **not** continue from the original (possibly tampered or stale) result
    payloads — those are deliberately not part of this object.

    The object is frozen (field *names* cannot be reassigned); the mutable
    payloads (``session`` / ``counters`` / ``completed_sources``) are fresh
    deep copies, and ``accumulators`` are the live reconstructed buffers that
    the lifecycle is expected to advance.
    """

    writer: DrizzleCheckpointWriter
    accumulators: list          # [DrizzleAccumulator x3] fresh from disk
    support_accumulators: object  # (SUP_W1, SUP_W2) | None for legacy
    session: dict               # fresh loaded session binding (baseline)
    counters: dict              # fresh loaded counters (baseline)
    completed_sources: list     # fresh loaded ledger (baseline)
    rejected_sources: list      # fresh loaded rejected dispositions (baseline)
    generation: int
    next_source_index: int


def _reject_json_constant(value: str):
    """Reject non-standard JSON constants (``NaN``/``Infinity``/``-Infinity``)."""
    raise DrizzleCheckpointError(f"non-finite JSON number {value!r}")


def _require_regular_file(path, what):
    """Require ``path`` to be an existing regular file, not a symlink."""
    if os.path.islink(path):
        raise DrizzleCheckpointError(f"{what} {path!r} is a symlink")
    if not os.path.isfile(path):
        raise DrizzleCheckpointError(
            f"{what} {path!r} is missing or not a regular file"
        )


def _validate_fillval(value, where):
    """Validate a per-channel ``fillval`` (string or finite number)."""
    if isinstance(value, bool):
        raise DrizzleCheckpointError(
            f"{where} must be a string or finite number, not bool"
        )
    if isinstance(value, str):
        if not value:
            raise DrizzleCheckpointError(f"{where} must be a non-empty string")
        return value
    if isinstance(value, (int, float)):
        f = float(value)
        if not np.isfinite(f):
            raise DrizzleCheckpointError(f"{where} must be finite")
        return f
    raise DrizzleCheckpointError(
        f"{where} must be a string or finite number, got "
        f"{type(value).__name__}"
    )


def _restat_identity(ident, where):
    """Re-stat one persisted source identity; path/size/mtime_ns must match.

    A renamed / missing / modified source fails closed (never a silent
    fallback), matching the documented "same poses, same bytes" continuation
    contract.
    """
    path = ident["path"]
    try:
        st = os.stat(path)
    except OSError as exc:
        raise DrizzleCheckpointError(
            f"{where} source {path!r} is missing/unreadable: {exc}"
        ) from exc
    if st.st_size != ident["size"]:
        raise DrizzleCheckpointError(
            f"{where} source {path!r} size changed: checkpoint "
            f"{ident['size']} vs disk {st.st_size}"
        )
    if st.st_mtime_ns != ident["mtime_ns"]:
        raise DrizzleCheckpointError(
            f"{where} source {path!r} mtime changed: checkpoint "
            f"{ident['mtime_ns']} vs disk {st.st_mtime_ns}"
        )


def _coerce_candidates(raw, where):
    """Normalize a resolver return value into an ordered candidate list.

    ``None`` (no candidate) yields an empty list; a single path string yields a
    one-element list; a list/tuple is returned in order.  Anything else is
    refused (fail closed, never a partial resolution).
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return list(raw)
    raise DrizzleCheckpointError(
        f"{where}: resolver must return a path, an ordered list of paths, or "
        f"None (got {type(raw).__name__})"
    )


def _verify_candidate(ident, path):
    """Return True iff ``path`` currently carries the exact identity evidence.

    Only a regular, non-symlink file with ``st_size`` == ``ident['size']`` and
    ``st_mtime_ns`` == ``ident['mtime_ns']`` matches.  A missing / symlink /
    non-regular / mismatched candidate returns ``False`` so the caller skips it
    and tries the next candidate — a renamed / tampered / duplicated source is
    never silently accepted.
    """
    if not isinstance(path, str) or not path:
        return False
    if os.path.islink(path):
        return False
    if not os.path.isfile(path):
        return False
    try:
        st = os.stat(path)
    except OSError:
        return False
    return st.st_size == ident["size"] and st.st_mtime_ns == ident["mtime_ns"]


def _resolve_identity(ident, where, resolver, context):
    """Resolve one canonical identity to a verified on-disk path.

    With ``resolver=None`` this is the strict D2A path (re-stat the original
    path, fail closed on any deviation).  With a resolver, the canonical
    identity + context are offered to the resolver and each returned candidate
    is re-stat'ed by the reader (never trusting the callback); the first exact
    regular-file match wins.  If no candidate matches, fail closed.
    """
    if resolver is None:
        _restat_identity(ident, where)
        return ident["path"]
    raw = resolver(ident, context)
    candidates = _coerce_candidates(raw, where)
    for cand in candidates:
        if not isinstance(cand, str) or not cand:
            raise DrizzleCheckpointError(
                f"{where}: resolver returned invalid candidate {cand!r}"
            )
        if _verify_candidate(ident, cand):
            return cand
    # RJK-R3: a nested non-canonical replay location (stacked/stacked or any
    # deeper nesting) must be named explicitly instead of surfacing as a
    # generic missing/tampered refusal.
    nested = _probe_nested_stacked(ident, resolver)
    if nested is not None:
        raise DrizzleCheckpointError(
            f"{where} source {ident['path']!r} was found at a non-canonical "
            f"nested stacked location {nested!r}: only the canonical original "
            "path or the exact canonical stacked counterpart are legal; "
            "refusing"
        )
    raise DrizzleCheckpointError(
        f"{where} source {ident['path']!r} could not be resolved: no candidate "
        "matches the persisted size/mtime_ns (missing, moved off-policy, "
        "tampered, duplicated, or a symlink)"
    )


def _probe_nested_stacked(ident, resolver):
    """Probe for a nested ``<src>/<stacked>/<stacked>/<basename>`` location.

    Diagnostic only (never a legal resolution): returns the nested path when
    it exists as a regular file, else ``None``.  Used to name the
    non-canonical nested-stacked failure mode explicitly.
    """
    path = ident.get("path") if isinstance(ident, dict) else None
    if not isinstance(path, str) or not path:
        return None
    sub = "stacked"
    if type(resolver) is SafeStackedSourceResolver:
        sub = resolver.stacked_subdir_name
    src_dir = os.path.dirname(path)
    base = os.path.basename(path)
    if not src_dir or not base:
        return None
    probe = os.path.join(src_dir, sub, sub, base)
    try:
        if os.path.isfile(probe):
            return probe
    except OSError:  # noqa: BLE001 - diagnostic only
        return None
    return None


def _identity_key(ident):
    """Return the canonical identity key ``(path, size, mtime_ns)``."""
    return (ident["path"], ident["size"], ident["mtime_ns"])


def identity_names_equivalent(name_a, name_b, normcase=None):
    """Host-aware basename identity comparison (filesystem path semantics).

    On a case-insensitive filesystem (Windows) ``os.path.normcase`` maps both
    names to the same normalized form, so a case-only difference is the same
    basename.  On POSIX ``os.path.normcase`` is the identity, so case-only
    differences remain distinct names.  Never a custom lowercase rule.

    ``normcase`` is injectable (default :func:`os.path.normcase`) so Windows
    semantics can be exercised explicitly (``ntpath.normcase``) on a POSIX
    test host without monkeypatching unrelated filesystem behavior.
    """
    if normcase is None:
        normcase = os.path.normcase
    try:
        return normcase(str(name_a)) == normcase(str(name_b))
    except Exception:  # noqa: BLE001 - malformed name never matches
        return False


def identities_equivalent_host(a, b, normcase=None):
    """Whole-identity equality under host filesystem path semantics.

    ``size`` / ``mtime_ns`` must match exactly (strict evidence); ``path``
    and the basename ``name`` follow host path semantics via ``normcase``
    (case-insensitive on Windows, case-sensitive on POSIX).  Used wherever
    two canonical identities are compared as whole dicts (ledger/plan prefix
    checks), so a case-only display-name difference on Windows can never
    produce a false rejection.
    """
    if a is b:
        return True
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if normcase is None:
        normcase = os.path.normcase
    if a.get("size") != b.get("size") or a.get("mtime_ns") != b.get(
        "mtime_ns"
    ):
        return False
    try:
        if normcase(str(a.get("path", ""))) != normcase(
            str(b.get("path", ""))
        ):
            return False
    except Exception:  # noqa: BLE001 - malformed identity never matches
        return False
    return identity_names_equivalent(
        a.get("name"), b.get("name"), normcase=normcase
    )


def identity_lists_equal(list_a, list_b, normcase=None):
    """Element-wise host-aware identity-list equality (order preserved)."""
    if len(list_a) != len(list_b):
        return False
    return all(
        identities_equivalent_host(x, y, normcase=normcase)
        for x, y in zip(list_a, list_b)
    )


def _resolve_sources(session, counters, resolver, output_dir,
                     rejected_sources=None, completed_sources=None):
    """Resolve the reference + ordered plan sources (strict or opt-in).

    Returns ``(resolved_reference, resolved_plan_paths)``.  Resolution is
    injective and order-preserving for *distinct* canonical identities: two
    distinct identities resolving to the same on-disk file (ambiguous /
    duplicated destination) is refused.  Repeated use of the *exact same*
    canonical identity is legitimate — the alignment reference is allowed to
    also be one of the plan observations — so it resolves to the same path
    without being reported as ambiguous.

    Rejected (disposed) plan sources are *never* resolved: their disposition
    is final, their science is absent by contract, and their physical location
    (``unaligned_by_stacker``, possibly collision-renamed) is irrelevant to
    both the scientific reconstruction and the remaining work.  Their slots in
    ``resolved_plan_paths`` are ``None`` placeholders preserving plan
    position.  A rejected identity appearing in the remaining suffix is
    already refused by the disposition validation.
    """
    reference = session["reference"]
    plan_sources = session["plan"]["sources"]
    rejected_keys = {
        (e["path"], e["size"], e["mtime_ns"]) for e in (rejected_sources or [])
    }
    completed_keys = {
        (e["path"], e["size"], e["mtime_ns"]) for e in (completed_sources or [])
    }
    input_roots = list(session.get("input_roots", []))
    real_output_dir = os.path.realpath(output_dir)

    def _context(role, index, is_completed):
        return {
            "role": role,
            "index": index,
            "is_completed": is_completed,
            "output_dir": real_output_dir,
            "input_roots": input_roots,
        }

    resolved_reference = _resolve_identity(
        reference, "session reference", resolver,
        _context("reference", None, True),
    )

    # Map resolved on-disk path -> (canonical identity, description).  This
    # enforces injectivity of *distinct* canonical identities while allowing
    # the same canonical identity to be legitimately referenced more than once
    # (e.g. the reference also appearing in the plan).
    claimed = {resolved_reference: (reference, "session reference")}
    resolved_plan = []
    for idx, ident in enumerate(plan_sources):
        if (ident["path"], ident["size"], ident["mtime_ns"]) in rejected_keys:
            resolved_plan.append(None)
            continue
        is_completed = (
            (ident["path"], ident["size"], ident["mtime_ns"]) in completed_keys
        )
        path = _resolve_identity(
            ident, "session plan source", resolver,
            _context("plan", idx, is_completed),
        )
        if path in claimed:
            claimed_ident, claimed_desc = claimed[path]
            if _identity_key(claimed_ident) != _identity_key(ident):
                raise DrizzleCheckpointError(
                    f"ambiguous source resolution: {claimed_desc} and session "
                    f"plan source index {idx} both resolve to {path!r}"
                )
        else:
            claimed[path] = (ident, f"session plan source index {idx}")
        resolved_plan.append(path)
    return resolved_reference, resolved_plan


@dataclass(frozen=True)
class SafeStackedSourceResolver:
    """Immutable, deterministic source-resolution policy (original-or-stacked).

    The production policy for the source-resolution seam: a completed /
    reference source moved by ``tools.file_ops.move_to_stacked`` keeps its size
    and mtime and lands at ``<src_dir>/<stacked_subdir>/<basename>``.  This
    resolver therefore returns exactly two candidate paths, in order:

    1. the canonical original path (``ident['path']``);
    2. ``<original_dir>/<stacked_subdir_name>/<basename>``.

    It performs **no** directory listing, glob, basename-only fallback, hashless
    remap, or arbitrary-rename search, and it never guesses a
    ``_dup_<timestamp>`` collision name (a basename carrying that marker is
    refused outright).  The reader re-stats every returned candidate and only
    accepts an exact size + mtime_ns regular-file match, so this policy object
    is pure and performs no verification itself.

    The object is frozen (immutable / hashable), so it can be carried as
    validated provenance in :class:`DrizzleCheckpointResult` and re-applied by
    :meth:`DrizzleCheckpointWriter.from_validated_result` without trusting a
    mutable callback.
    """

    stacked_subdir_name: str = "stacked"

    def __post_init__(self):
        sub = self.stacked_subdir_name
        if not isinstance(sub, str) or not sub:
            raise DrizzleCheckpointError(
                "SafeStackedSourceResolver stacked_subdir_name must be a "
                "non-empty string"
            )
        if (
            sub in (".", "..")
            or sub != os.path.basename(sub)
            or sub.startswith(("/", "\\"))
        ):
            raise DrizzleCheckpointError(
                f"stacked_subdir_name {sub!r} is not a plain subdirectory name"
            )
        object.__setattr__(self, "stacked_subdir_name", sub)

    def __call__(self, ident, context):
        return self.resolve(ident, context)

    def resolve(self, ident, context):
        """Return the deterministic ordered candidate paths for ``ident``."""
        path = ident.get("path") if isinstance(ident, dict) else None
        if not isinstance(path, str) or not path:
            return None
        base = os.path.basename(path)
        if not base or _DUP_COLLISION_RE.search(base):
            # Never guess a _dup_<timestamp> collision target.
            return None
        src_dir = os.path.dirname(path)
        stacked = os.path.join(src_dir, self.stacked_subdir_name, base)
        return [path, stacked]


def read_drizzle_checkpoint(output_dir, *, require_exact_versions=True,
                            resolver=None):
    """Read, validate and reconstruct a native Drizzle checkpoint (read-only).

    Locates ``<output_dir>/.m3d_checkpoint/checkpoint.json`` and
    ``<output_dir>/run_config.cfg``.  Fails closed as
    :class:`DrizzleCheckpointError` on any of: missing / malformed / non-strict
    JSON, unknown schema, wrong mode / state, invalid generation, product /
    config / fingerprint / digest mismatch, invalid output shape / WCS, library
    version mismatch (exact-continuation policy), malformed session / plan /
    ledger / counters, unsafe artifact names / path traversal / symlinks /
    descriptors, missing / extra / mixed-generation channel artifacts, and any
    artifact whose exact size / SHA-256 / dtype / shape / finiteness does not
    match.  Never mutates checkpoint bytes, source files, the output directory
    or live runtime state; arrays are loaded with ``allow_pickle=False`` and
    returned as private float32 copies.

    Every persisted source (the reference, every plan source and every
    completed-ledger source) is re-stat'ed: path / size / mtime_ns must match
    exactly (rejected disposition sources are terminal and never re-stat'ed —
    their physical location is irrelevant to the science and the remaining
    work).  The disposition partition (completed + rejected, plan-ordered, up
    to ``plan_cursor``) is enforced; on a legacy prefix-only checkpoint this
    is exactly the historical ``completed_sources == plan[:frame_count]``
    contract.  The three accumulators are reconstructed only after the entire
    checkpoint validates (no partial externally visible restore).

    When ``resolver`` is ``None`` (the default) source re-stat is **strict**:
    each identity must still exist at its original persisted path with the exact
    size + mtime_ns (a moved / renamed / missing / tampered source fails closed,
    exactly as D2A).  When a resolver is supplied (explicit opt-in), each
    canonical identity is offered to it together with a context dict, and the
    reader itself re-stats every returned candidate path (regular, non-symlink
    file, exact size + mtime_ns) — a callback claim is never trusted.  The
    resolved reference / ordered plan paths are exposed on the returned result
    (``resolved_reference`` / ``resolved_plan_paths`` /
    ``resolved_completed_paths`` / ``resolved_remaining_paths``); persisted
    manifest/session identities stay canonical (original paths), never rewritten.

    Parameters
    ----------
    output_dir :
        Explicit output / run directory containing the checkpoint namespace.
    require_exact_versions :
        When ``True`` (default) a persisted ``drizzle_lib_version`` /
        ``numpy_version`` different from the runtime library is refused
        (documented exact-continuation policy).
    resolver :
        Optional explicit source-resolution callback / policy.  Must be callable
        as ``resolver(ident, context) -> path | [paths] | None``.  Only the
        exact shipped :class:`SafeStackedSourceResolver` type (never a subclass,
        which could smuggle mutable state and/or override resolution) is
        carried forward as ``resolution_policy`` provenance (and thus re-applied
        by the continuation factory); an arbitrary mutable callable — including
        a ``SafeStackedSourceResolver`` subclass — is honoured for the immediate
        read but is **not** carried (re-arm then falls back to strict).

    Returns
    -------
    DrizzleCheckpointResult
    """
    output_dir = os.fspath(output_dir)
    ckpt_dir = os.path.join(output_dir, CHECKPOINT_DIRNAME)
    manifest_path = os.path.join(ckpt_dir, MANIFEST_FILENAME)
    cfg_path = os.path.join(output_dir, RUN_CONFIG_FILENAME)

    if os.path.islink(ckpt_dir):
        raise DrizzleCheckpointError(
            f"checkpoint directory {ckpt_dir!r} is a symlink"
        )
    if not os.path.isdir(ckpt_dir):
        raise DrizzleCheckpointError(
            f"checkpoint directory {ckpt_dir!r} is missing"
        )

    manifest = _read_manifest_strict(manifest_path)
    generation = _validate_top_level(manifest)
    config = _validate_config(manifest, cfg_path)
    output_shape_hw = _validate_output_shape(manifest)
    wcs = _reconstruct_wcs(manifest, output_shape_hw)
    counters = _validate_counters(manifest)
    session = _validate_session(manifest)
    # F1: a v2 output-grid checkpoint must carry the versioned frozen
    # input-reference geometry; refuse a stripped payload before any
    # reconstruction/mutation.
    if session.get("reference_geometry") is None:
        raise DrizzleCheckpointError(
            "missing mandatory input-reference geometry in the v2 output-grid "
            "checkpoint"
        )
    ledger, rejected = _validate_ledger(manifest, session, counters)
    resolved_reference, resolved_plan_paths = _resolve_sources(
        session, counters, resolver, output_dir,
        rejected_sources=rejected, completed_sources=ledger,
    )
    channels, support = _validate_channels(
        manifest, generation, ckpt_dir, output_shape_hw
    )
    _validate_channel_vs_canonical(config, channels)
    _validate_versions(manifest, require_exact_versions)

    # Reconstruct only after the entire checkpoint validated.  P2-B: the frozen
    # WCS-derived kernel-geometry factor is restored from the validated
    # canonical scientific config so a continuation deposits with EXACTLY the
    # same geometry as the run that wrote the checkpoint.
    _psr = config.get(run_contract.Section.SCIENTIFIC, "pixel_scale_ratio_effective")
    accumulators = _reconstruct_accumulators(
        channels, output_shape_hw, pixel_scale_ratio=_psr
    )
    support_accumulators = _reconstruct_support(support, output_shape_hw)

    # Only the exact shipped immutable policy *type* is carried as re-arm
    # provenance.  ``isinstance`` would admit subclasses that can add mutable
    # state and/or override ``resolve``/``__call__``; ``type(...) is ...``
    # admits only the canonical frozen policy (fail closed).  Any other
    # callable — including a ``SafeStackedSourceResolver`` subclass — is
    # honoured for the immediate read but never re-applied later.
    resolution_policy = (
        resolver if type(resolver) is SafeStackedSourceResolver else None
    )

    return DrizzleCheckpointResult(
        manifest=manifest,
        session=session,
        counters=counters,
        completed_sources=ledger,
        rejected_sources=rejected,
        plan_cursor=counters["plan_cursor"],
        config=config,
        wcs=wcs,
        output_shape_hw=output_shape_hw,
        accumulators=accumulators,
        support_accumulators=support_accumulators,
        next_source_index=counters["plan_cursor"],
        generation=generation,
        source_output_dir=output_dir,
        resolved_reference=resolved_reference,
        resolved_plan_paths=resolved_plan_paths,
        resolution_policy=resolution_policy,
        reference_geometry=session.get("reference_geometry"),
    )


def _read_manifest_strict(manifest_path):
    """Read and strictly parse ``checkpoint.json`` (no symlink, no NaN/Inf)."""
    _require_regular_file(manifest_path, "checkpoint manifest")
    try:
        with open(manifest_path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise DrizzleCheckpointError(
            f"cannot read checkpoint manifest {manifest_path!r}: {exc}"
        ) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DrizzleCheckpointError(
            f"checkpoint manifest is not valid UTF-8: {exc}"
        ) from exc
    try:
        data = json.loads(text, parse_constant=_reject_json_constant)
    except DrizzleCheckpointError:
        raise
    except ValueError as exc:
        raise DrizzleCheckpointError(
            f"checkpoint manifest is not strict JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise DrizzleCheckpointError(
            "checkpoint manifest top-level is not a JSON object"
        )
    return data


def _validate_top_level(manifest):
    """Validate schema / mode / state / generation / product / producer."""
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DrizzleCheckpointError(
            f"unknown schema_version {manifest.get('schema_version')!r} "
            f"(expected {SCHEMA_VERSION})"
        )
    if manifest.get("mode") != MODE_TOKEN:
        raise DrizzleCheckpointError(
            f"wrong mode {manifest.get('mode')!r} (expected {MODE_TOKEN!r})"
        )
    if manifest.get("state") != STATE_CLEAN:
        raise DrizzleCheckpointError(
            f"wrong state {manifest.get('state')!r} (expected {STATE_CLEAN!r})"
        )
    generation = _strict_int(manifest.get("generation"), "generation")
    if generation < 1:
        raise DrizzleCheckpointError(f"invalid generation {generation}")
    product_version = manifest.get("product_version")
    if not isinstance(product_version, str):
        raise DrizzleCheckpointError("product_version must be a string")
    producer = manifest.get("producer")
    if producer is not None and producer != "zeseestarstacker":
        raise DrizzleCheckpointError(f"unknown producer {producer!r}")
    return generation


def _validate_output_shape(manifest):
    """Validate ``output_shape_hw`` into a positive ``(H, W)`` tuple."""
    raw = manifest.get("output_shape_hw")
    if not isinstance(raw, list) or len(raw) != 2:
        raise DrizzleCheckpointError(
            "output_shape_hw must be a 2-element list"
        )
    h = _strict_int(raw[0], "output_shape_hw[0]")
    w = _strict_int(raw[1], "output_shape_hw[1]")
    if h <= 0 or w <= 0:
        raise DrizzleCheckpointError(f"invalid output_shape_hw {(h, w)}")
    return (h, w)


def _validate_config(manifest, cfg_path):
    """Read ``run_config.cfg`` and cross-check digest / fingerprint / product /
    embedded scientific_config (fail closed on any mismatch)."""
    if os.path.islink(cfg_path):
        raise DrizzleCheckpointError("run_config.cfg is a symlink")
    try:
        report = run_contract.read_cfg(cfg_path)
    except run_contract.ConfigError as exc:
        raise DrizzleCheckpointError(f"invalid run_config.cfg: {exc}") from exc
    except OSError as exc:
        raise DrizzleCheckpointError(f"cannot read run_config.cfg: {exc}") from exc
    config = report.config

    # GAR-06: the output-grid geometry contract is explicitly versioned.  A
    # checkpoint written under the legacy north-up / raw-CRPIX grid contract is
    # refused here (read-only, before any mutation) with a clear restart
    # requirement; accumulated SCI/WHT arrays are never silently converted.
    persisted_grid = (
        config.scientific.get("output_grid_contract"),
        config.scientific.get("output_grid_contract_version"),
    )
    if persisted_grid != (_OUTPUT_GRID_CONTRACT, _OUTPUT_GRID_CONTRACT_VERSION):
        if persisted_grid in _LEGACY_OUTPUT_GRID_CONTRACTS:
            raise DrizzleCheckpointError(
                "legacy Drizzle output-grid checkpoint "
                f"({persisted_grid[0]} v{persisted_grid[1]}): the north-up / "
                "raw-CRPIX grid contract cannot be migrated in place; start a "
                "fresh run (accumulated arrays are never silently converted)."
            )
        raise DrizzleCheckpointError(
            "unsupported Drizzle output-grid contract "
            f"{persisted_grid[0]!r} v{persisted_grid[1]!r}"
        )

    if config.product_version != manifest.get("product_version"):
        raise DrizzleCheckpointError(
            "run_config.cfg product_version does not match the manifest"
        )

    expected_digest = manifest.get("run_config_digest")
    if (
        not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or any(ch not in "0123456789abcdef" for ch in expected_digest)
    ):
        raise DrizzleCheckpointError(
            "manifest run_config_digest is not a SHA-256 hex string"
        )
    if config.full_digest() != expected_digest:
        raise DrizzleCheckpointError("run_config.cfg digest mismatch")

    expected_fp = manifest.get("scientific_fingerprint")
    if (
        not isinstance(expected_fp, str)
        or len(expected_fp) != 64
        or any(ch not in "0123456789abcdef" for ch in expected_fp)
    ):
        raise DrizzleCheckpointError(
            "manifest scientific_fingerprint is not a SHA-256 hex string"
        )
    if config.drizzle_fingerprint() != expected_fp:
        raise DrizzleCheckpointError("drizzle scientific fingerprint mismatch")

    embedded_sci = manifest.get("scientific_config")
    if not isinstance(embedded_sci, dict):
        raise DrizzleCheckpointError(
            "manifest scientific_config is missing or malformed"
        )
    if config.scientific != embedded_sci:
        raise DrizzleCheckpointError("manifest scientific_config mismatch")
    return config


def _wcs_from_cards(wcs_dict, where="output WCS"):
    """Rebuild a 2-axis :class:`astropy.wcs.WCS` from strict JSON cards.

    Fail closed on any non-JSON/non-finite card or unreadable frame; the
    caller attaches the shape metadata it validates separately.
    """
    header = fits.Header()
    for key, value in wcs_dict.items():
        if not isinstance(key, str) or key in ("", "HISTORY", "COMMENT"):
            raise DrizzleCheckpointError(f"invalid WCS card key {key!r}")
        if isinstance(value, bool):
            header[key] = value
        elif isinstance(value, (int, float)):
            if not np.isfinite(float(value)):
                raise DrizzleCheckpointError(f"non-finite WCS card {key!r}")
            header[key] = value
        elif isinstance(value, str):
            header[key] = value
        else:
            raise DrizzleCheckpointError(
                f"non-JSON WCS card {key!r} value type {type(value).__name__}"
            )
    try:
        wcs = WCS(header)
    except Exception as exc:  # noqa: BLE001 - fail closed, never partial
        raise DrizzleCheckpointError(
            f"cannot reconstruct {where}: {exc}"
        ) from exc
    if wcs.naxis != 2:
        raise DrizzleCheckpointError(f"{where} has naxis {wcs.naxis} != 2")
    # Preserve inverse polynomials explicitly, including order-1 AP/BP
    # accepted in memory but omitted by Astropy 8 on header re-parse.
    # Standard inverse polynomials of order >=2 already survive WCS parsing.
    sip = _sip_from_cards(header, where)
    if sip is not None:
        try:
            wcs.sip = sip
        except Exception as exc:  # noqa: BLE001 - fail closed
            raise DrizzleCheckpointError(
                f"cannot attach SIP distortion to {where}: {exc}"
            ) from exc
    return wcs


def _sip_from_cards(header, where="output WCS"):
    """Rebuild a full :class:`astropy.wcs.Sip` from persisted order cards.

    Returns ``None`` when the header carries no SIP.  Fails closed on an
    inconsistent forward/inverse order or non-finite coefficient.
    """
    from astropy.wcs import Sip

    def _order(name):
        try:
            return int(header.get(name, 0) or 0)
        except (TypeError, ValueError) as exc:
            raise DrizzleCheckpointError(
                f"{where} has a malformed {name} card"
            ) from exc

    a_order = _order("A_ORDER")
    b_order = _order("B_ORDER")
    ap_order = _order("AP_ORDER")
    bp_order = _order("BP_ORDER")
    if max(a_order, b_order, ap_order, bp_order) <= 0:
        return None
    if a_order != b_order:
        raise DrizzleCheckpointError(f"{where} has mismatched SIP A/B orders")
    if ap_order != bp_order:
        raise DrizzleCheckpointError(f"{where} has mismatched SIP AP/BP orders")

    def _matrix(order, prefix):
        matrix = np.zeros((order + 1, order + 1), dtype=float)
        for i in range(order + 1):
            for j in range(order + 1):
                key = f"{prefix}_{i}_{j}"
                if key in header:
                    try:
                        matrix[i, j] = float(header[key])
                    except (TypeError, ValueError) as exc:
                        raise DrizzleCheckpointError(
                            f"{where} has a malformed {key} coefficient"
                        ) from exc
        if not np.all(np.isfinite(matrix)):
            raise DrizzleCheckpointError(f"{where} has non-finite SIP coefficients")
        return matrix

    a = _matrix(a_order, "A")
    b = _matrix(b_order, "B")
    if ap_order > 0:
        ap = _matrix(ap_order, "AP")
        bp = _matrix(bp_order, "BP")
    else:
        ap = np.zeros((1, 1), dtype=float)
        bp = np.zeros((1, 1), dtype=float)
    try:
        crpix = np.array(
            [float(header.get("CRPIX1", 0.0)), float(header.get("CRPIX2", 0.0))],
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise DrizzleCheckpointError(f"{where} has a malformed CRPIX") from exc
    if not np.all(np.isfinite(crpix)):
        raise DrizzleCheckpointError(f"{where} has non-finite CRPIX")
    try:
        return Sip(a, b, ap, bp, crpix)
    except Exception as exc:  # noqa: BLE001 - fail closed
        raise DrizzleCheckpointError(
            f"cannot reconstruct SIP distortion for {where}: {exc}"
        ) from exc


def _reconstruct_wcs(manifest, output_shape_hw):
    """Reconstruct and validate the output WCS; attach ``array_shape``."""
    wcs_dict = manifest.get("wcs")
    if not isinstance(wcs_dict, dict) or not wcs_dict:
        raise DrizzleCheckpointError("manifest wcs is missing or malformed")
    wcs = _wcs_from_cards(wcs_dict, "output WCS")
    wcs.array_shape = tuple(output_shape_hw)

    # Exact output-grid contract: the reconstructed WCS must round-trip back
    # to exactly the persisted card dict.
    try:
        reserialized = serialize_wcs_header(wcs)
    except DrizzleCheckpointError:
        raise
    if reserialized != wcs_dict:
        raise DrizzleCheckpointError("output WCS does not round-trip exactly")
    # Astropy axis convention: ``array_shape`` is ``(H, W)`` (numpy order)
    # while ``pixel_shape`` is ``(W, H)`` (FITS NAXIS order).  ``output_shape_hw``
    # is the ``(H, W)`` grid shape, so the two checks must use the *reversed*
    # comparison for ``pixel_shape``.
    if wcs.array_shape != tuple(output_shape_hw):
        raise DrizzleCheckpointError(
            f"output WCS array_shape {wcs.array_shape} != output_shape_hw "
            f"{output_shape_hw}"
        )
    if wcs.pixel_shape != (output_shape_hw[1], output_shape_hw[0]):
        raise DrizzleCheckpointError(
            f"output WCS pixel_shape {wcs.pixel_shape} != (W, H) "
            f"{(output_shape_hw[1], output_shape_hw[0])}"
        )
    return wcs


def _validate_counters(manifest):
    """Validate the persisted accepted-exposure counters (strict)."""
    frame_count = _strict_int(manifest.get("frame_count"), "frame_count")
    if frame_count < 0:
        raise DrizzleCheckpointError("negative frame_count")
    plan_cursor = _strict_int(
        manifest.get("plan_cursor", frame_count), "plan_cursor"
    )
    if plan_cursor < frame_count:
        raise DrizzleCheckpointError(
            f"plan_cursor {plan_cursor} < frame_count {frame_count}"
        )
    if frame_count == 0 and plan_cursor == 0:
        raise DrizzleCheckpointError("empty checkpoint (frame_count <= 0)")
    stacked = _strict_int(
        manifest.get("stacked_batches_count"), "stacked_batches_count"
    )
    if stacked < 0:
        raise DrizzleCheckpointError("negative stacked_batches_count")
    if stacked != frame_count:
        raise DrizzleCheckpointError(
            f"stacked_batches_count {stacked} != frame_count {frame_count}"
        )
    total = _strict_float(
        manifest.get("total_exposure_seconds"), "total_exposure_seconds"
    )
    if total < 0.0:
        raise DrizzleCheckpointError(
            f"negative total_exposure_seconds {total!r}"
        )
    unknown = _strict_int(
        manifest.get("exposure_unknown_count"), "exposure_unknown_count"
    )
    if unknown < 0:
        raise DrizzleCheckpointError("negative exposure_unknown_count")
    if unknown > frame_count:
        raise DrizzleCheckpointError(
            f"exposure_unknown_count {unknown} > frame_count {frame_count}"
        )
    exp_min = _strict_float(
        manifest.get("exposure_min"), "exposure_min", allow_none=True
    )
    exp_max = _strict_float(
        manifest.get("exposure_max"), "exposure_max", allow_none=True
    )
    if exp_min is not None and exp_max is not None and exp_min > exp_max:
        raise DrizzleCheckpointError(
            f"exposure_min {exp_min} > exposure_max {exp_max}"
        )
    return {
        "frame_count": frame_count,
        "plan_cursor": plan_cursor,
        "stacked_batches_count": stacked,
        "total_exposure_seconds": total,
        "exposure_unknown_count": unknown,
        "exposure_min": exp_min,
        "exposure_max": exp_max,
    }


def _validate_session(manifest):
    """Validate and re-stat the session binding (roots / reference / plan)."""
    session = manifest.get("session")
    if not isinstance(session, dict):
        raise DrizzleCheckpointError("session is missing or malformed")

    roots = session.get("input_roots")
    if not isinstance(roots, list) or not roots:
        raise DrizzleCheckpointError("missing session input_roots")
    roots_clean = []
    for r in roots:
        if not isinstance(r, str) or not r:
            raise DrizzleCheckpointError(
                "session input_roots entries must be non-empty strings"
            )
        roots_clean.append(r)

    reference = _validate_identity(session.get("reference"), "session reference")

    plan = session.get("plan")
    if not isinstance(plan, dict):
        raise DrizzleCheckpointError("missing session observation plan")
    sources = plan.get("sources")
    if not isinstance(sources, list) or not sources:
        raise DrizzleCheckpointError(
            "session observation plan sources must be a non-empty list"
        )
    sources_clean = []
    seen = set()
    for entry in sources:
        ident = _validate_identity(entry, "session plan source")
        key = (ident["path"], ident["size"], ident["mtime_ns"])
        if key in seen:
            raise DrizzleCheckpointError(
                f"duplicate source identity in session plan: {ident['name']}"
            )
        seen.add(key)
        sources_clean.append(ident)

    plan_clean = {"sources": sources_clean}
    decomposition = plan.get("decomposition")
    if decomposition is not None:
        if not isinstance(decomposition, list):
            raise DrizzleCheckpointError(
                "session plan decomposition must be a list"
            )
        deco_clean = []
        for b in decomposition:
            bi = _strict_int(b, "session plan decomposition element")
            if bi <= 0:
                raise DrizzleCheckpointError(
                    "session plan decomposition elements must be positive"
                )
            deco_clean.append(bi)
        plan_clean["decomposition"] = deco_clean

    return {
        "input_roots": roots_clean,
        "reference": reference,
        "plan": plan_clean,
        "reference_geometry": _validate_input_reference_geometry(
            session.get("reference_geometry")
        ),
    }


def _validate_ledger(manifest, session, counters):
    """Validate the completed + rejected disposition ledgers (partition).

    Returns ``(ledger, rejected)``.  Legacy checkpoints (no ``plan_cursor`` /
    ``rejected_sources``) keep the exact historical prefix-only semantics;
    rejection-aware checkpoints must satisfy the plan-ordered disposition
    partition (see the writer's ``_validate_manifest_consistency``): every plan
    source before ``plan_cursor`` carries exactly one final disposition
    (accepted science or rejected), every disposition ledger is plan-ordered
    and disjoint, and no disposed source reappears in the remaining suffix.
    """
    raw = manifest.get("completed_sources")
    if not isinstance(raw, list):
        raise DrizzleCheckpointError("completed_sources must be a list")
    ledger = []
    seen = set()
    for entry in raw:
        ident = _validate_identity(entry, "completed ledger")
        key = (ident["path"], ident["size"], ident["mtime_ns"])
        if key in seen:
            raise DrizzleCheckpointError(
                f"duplicate source identity in completed ledger: {ident['name']}"
            )
        seen.add(key)
        ledger.append(ident)

    raw_rejected = manifest.get("rejected_sources")
    if raw_rejected is None:
        raw_rejected = []
    if not isinstance(raw_rejected, list):
        raise DrizzleCheckpointError("rejected_sources must be a list")
    rejected = []
    seen_rejected = set()
    for entry in raw_rejected:
        ident = _validate_identity(entry, "rejected disposition ledger")
        key = (ident["path"], ident["size"], ident["mtime_ns"])
        if key in seen_rejected:
            raise DrizzleCheckpointError(
                f"duplicate source identity in rejected ledger: {ident['name']}"
            )
        seen_rejected.add(key)
        rejected.append(ident)

    frame_count = counters["frame_count"]
    plan_cursor = counters["plan_cursor"]
    plan_sources = session["plan"]["sources"]
    if len(ledger) != frame_count:
        raise DrizzleCheckpointError(
            f"completed_sources length {len(ledger)} != frame_count {frame_count}"
        )
    if frame_count > len(plan_sources):
        raise DrizzleCheckpointError(
            f"frame_count {frame_count} exceeds session plan length "
            f"{len(plan_sources)}"
        )
    if plan_cursor > len(plan_sources):
        raise DrizzleCheckpointError(
            f"plan_cursor {plan_cursor} exceeds session plan length "
            f"{len(plan_sources)}"
        )
    if plan_cursor != len(ledger) + len(rejected):
        raise DrizzleCheckpointError(
            f"plan_cursor {plan_cursor} != completed {len(ledger)} + rejected "
            f"{len(rejected)}"
        )
    completed_keys = {(e["path"], e["size"], e["mtime_ns"]) for e in ledger}
    rejected_keys = {(e["path"], e["size"], e["mtime_ns"]) for e in rejected}
    reference = session["reference"]
    reference_key = (reference["path"], reference["size"], reference["mtime_ns"])
    if reference_key in rejected_keys:
        raise DrizzleCheckpointError(
            "the session reference observation cannot be rejected: a disposed "
            "reference would make the alignment reference unresolvable on "
            "Resume"
        )
    if completed_keys & rejected_keys:
        raise DrizzleCheckpointError(
            "a source identity is both accepted and rejected"
        )
    if not rejected and not identity_lists_equal(
        ledger, plan_sources[:frame_count]
    ):
        raise DrizzleCheckpointError(
            "completed_sources is not the exact ordered prefix of the session plan"
        )
    for ident in plan_sources[plan_cursor:]:
        key = (ident["path"], ident["size"], ident["mtime_ns"])
        if key in rejected_keys:
            raise DrizzleCheckpointError(
                f"rejected source {ident['name']} reappears in the remaining "
                "session plan"
            )
        if key in completed_keys:
            raise DrizzleCheckpointError(
                f"completed source {ident['name']} reappears in the remaining "
                "session plan"
            )
    accepted_ptr = 0
    rejected_ptr = 0
    for plan_index, ident in enumerate(plan_sources[:plan_cursor]):
        key = (ident["path"], ident["size"], ident["mtime_ns"])
        if (
            accepted_ptr < len(ledger)
            and key
            == (
                ledger[accepted_ptr]["path"],
                ledger[accepted_ptr]["size"],
                ledger[accepted_ptr]["mtime_ns"],
            )
        ):
            accepted_ptr += 1
        elif (
            rejected_ptr < len(rejected)
            and key
            == (
                rejected[rejected_ptr]["path"],
                rejected[rejected_ptr]["size"],
                rejected[rejected_ptr]["mtime_ns"],
            )
        ):
            rejected_ptr += 1
        else:
            raise DrizzleCheckpointError(
                f"plan source at index {plan_index} has no matching "
                f"accepted/rejected disposition ({ident['name']})"
            )
    if accepted_ptr != len(ledger) or rejected_ptr != len(rejected):
        raise DrizzleCheckpointError(
            "disposition ledgers do not exhaust the plan prefix "
            f"(completed {accepted_ptr}/{len(ledger)}, rejected "
            f"{rejected_ptr}/{len(rejected)})"
        )
    return ledger, rejected


def _validate_channels(manifest, generation, ckpt_dir, output_shape_hw):
    """Validate native channels plus optional support (fail closed).

    Also verifies the checkpoint directory contains *exactly* the referenced
    generation artifacts (no missing / extra / mixed-generation artifacts, no
    unexpected entry, no symlink, no path traversal).
    """
    channels = manifest.get("channels")
    if not isinstance(channels, list) or len(channels) != 3:
        raise DrizzleCheckpointError(
            "expected exactly 3 channel entries in the manifest"
        )

    try:
        dir_entries = os.listdir(ckpt_dir)
    except OSError as exc:
        raise DrizzleCheckpointError(
            f"cannot list checkpoint directory {ckpt_dir!r}: {exc}"
        ) from exc

    expected_files = set()
    channels_clean = []
    seen_channels = set()
    ref = None

    for ch in channels:
        if not isinstance(ch, dict):
            raise DrizzleCheckpointError("channel entry is not a JSON object")
        c = _strict_int(ch.get("channel"), "channel index")
        if c not in (0, 1, 2):
            raise DrizzleCheckpointError(f"invalid channel index {c}")
        if c in seen_channels:
            raise DrizzleCheckpointError(f"duplicate channel index {c}")
        seen_channels.add(c)

        kernel = ch.get("kernel")
        if not isinstance(kernel, str) or kernel not in VALID_DRIZZLE_KERNELS:
            raise DrizzleCheckpointError(
                f"channel {c} has unknown kernel {kernel!r}"
            )
        pixfrac = _strict_float(ch.get("pixfrac"), f"channel {c} pixfrac")
        if not (0.0 < pixfrac <= 1.0):
            raise DrizzleCheckpointError(
                f"channel {c} pixfrac {pixfrac} outside (0, 1]"
            )
        fillval = _validate_fillval(ch.get("fillval"), f"channel {c} fillval")
        total = _strict_float(
            ch.get("total_exptime"), f"channel {c} total_exptime"
        )
        if total < 0.0:
            raise DrizzleCheckpointError(
                f"channel {c} negative total_exptime {total!r}"
            )

        current = (kernel, pixfrac, fillval, total)
        if ref is None:
            ref = current
        elif current != ref:
            raise DrizzleCheckpointError(
                f"inconsistent per-channel drizzle config at channel {c}"
            )

        loaded = {}
        for kind in ("out_img", "out_wht"):
            desc = ch.get(kind)
            if not isinstance(desc, dict):
                raise DrizzleCheckpointError(
                    f"channel {c} {kind} descriptor missing"
                )
            loaded[kind] = _validate_artifact(
                desc, generation, c, kind, ckpt_dir, output_shape_hw
            )
            expected_files.add(desc["file"])

        channels_clean.append(
            {
                "channel": c,
                "kernel": kernel,
                "pixfrac": pixfrac,
                "fillval": fillval,
                "total_exptime": total,
                "out_img": loaded["out_img"],
                "out_wht": loaded["out_wht"],
            }
        )

    support_clean, support_files = _validate_support(
        manifest, generation, ckpt_dir, output_shape_hw
    )
    expected_files.update(support_files)

    # Exactly the referenced artifacts; no extra / mixed-generation artifacts
    # and no unexpected (temp / foreign) entry in the namespace.
    actual_gen_files = set()
    for name in dir_entries:
        if name == MANIFEST_FILENAME:
            continue
        if _ARTIFACT_RE.match(name):
            actual_gen_files.add(name)
            continue
        raise DrizzleCheckpointError(
            f"unexpected entry in checkpoint directory: {name!r}"
        )
    if actual_gen_files != expected_files:
        missing = sorted(expected_files - actual_gen_files)
        extra = sorted(actual_gen_files - expected_files)
        raise DrizzleCheckpointError(
            "generation artifact mismatch: "
            f"missing {missing}, extra {extra}"
        )

    return channels_clean, support_clean


def _validate_support(manifest, generation, ckpt_dir, output_shape_hw):
    """Validate optional additive SUP_W1/SUP_W2 manifest state.

    Absence of the top-level field is the sole legacy signal.  Once present,
    the support object and both generation-bound artifacts are mandatory and
    validated with the same size/digest/dtype/shape rules as native channels.
    """
    if "support" not in manifest:
        return None, set()
    support = manifest["support"]
    if not isinstance(support, dict):
        raise DrizzleCheckpointError("support entry is not a JSON object")
    if support.get("schema_version") != 1:
        raise DrizzleCheckpointError(
            f"unsupported support schema_version "
            f"{support.get('schema_version')!r}"
        )
    if support.get("kernel") != "square":
        raise DrizzleCheckpointError("support kernel must be 'square'")
    pixfrac = _strict_float(support.get("pixfrac"), "support pixfrac")
    if pixfrac != 1.0:
        raise DrizzleCheckpointError(
            f"support pixfrac {pixfrac} != 1.0"
        )
    fillval = _validate_fillval(support.get("fillval"), "support fillval")
    total_exptime = _strict_float(
        support.get("total_exptime"), "support total_exptime"
    )
    frame_count = _strict_int(manifest.get("frame_count"), "frame_count")
    if total_exptime != float(frame_count):
        raise DrizzleCheckpointError(
            f"support total_exptime {total_exptime} != frame_count "
            f"{frame_count}"
        )

    loaded = {}
    expected_files = set()
    for field, kind in (("sup_w1", "w1"), ("sup_w2", "w2")):
        desc = support.get(field)
        if not isinstance(desc, dict):
            raise DrizzleCheckpointError(
                f"support {field} descriptor missing"
            )
        loaded[kind] = _validate_support_artifact(
            desc, generation, kind, ckpt_dir, output_shape_hw
        )
        expected_files.add(desc["file"])

    return {
        "kernel": "square",
        "pixfrac": pixfrac,
        "fillval": fillval,
        "total_exptime": total_exptime,
        "w1": loaded["w1"],
        "w2": loaded["w2"],
    }, expected_files


def _validate_support_artifact(desc, generation, kind, ckpt_dir,
                               output_shape_hw):
    """Validate and load one generation-bound positive-support array."""
    what = f"support {kind}"
    file_name = desc.get("file")
    if not isinstance(file_name, str) or not file_name:
        raise DrizzleCheckpointError(f"{what} has invalid file name")
    if (
        file_name != os.path.basename(file_name)
        or file_name.startswith(("/", "\\"))
        or ".." in file_name
    ):
        raise DrizzleCheckpointError(
            f"{what} has unsafe file name {file_name!r}"
        )
    match = _SUPPORT_ARTIFACT_RE.match(file_name)
    if not match:
        raise DrizzleCheckpointError(
            f"{what} has non-allowlisted file name {file_name!r}"
        )
    if int(match.group(1)) != generation:
        raise DrizzleCheckpointError(
            f"{what} file generation {match.group(1)} != manifest generation "
            f"{generation}"
        )
    if match.group(2) != kind:
        raise DrizzleCheckpointError(
            f"{what} file kind {match.group(2)!r} != {kind!r}"
        )
    if desc.get("dtype") != "float32":
        raise DrizzleCheckpointError(
            f"{what} dtype {desc.get('dtype')!r} != 'float32'"
        )
    shape = desc.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise DrizzleCheckpointError(
            f"{what} shape must be a 2-element list"
        )
    parsed_shape = (
        _strict_int(shape[0], f"{what} shape[0]"),
        _strict_int(shape[1], f"{what} shape[1]"),
    )
    if parsed_shape != tuple(output_shape_hw):
        raise DrizzleCheckpointError(
            f"{what} shape {parsed_shape} != output_shape_hw "
            f"{tuple(output_shape_hw)}"
        )
    size = desc.get("size")
    if isinstance(size, bool) or not isinstance(size, int):
        raise DrizzleCheckpointError(f"{what} size must be a strict integer")
    sha = desc.get("sha256")
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or any(ch not in "0123456789abcdef" for ch in sha)
    ):
        raise DrizzleCheckpointError(
            f"{what} sha256 must be a 64-char hex string"
        )

    path = os.path.join(ckpt_dir, file_name)
    if os.path.islink(path):
        raise DrizzleCheckpointError(
            f"{what} artifact {file_name!r} is a symlink"
        )
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise DrizzleCheckpointError(
            f"{what} artifact missing/unreadable: {exc}"
        ) from exc
    if len(raw) != size:
        raise DrizzleCheckpointError(
            f"{what} size mismatch: manifest {size} vs disk {len(raw)}"
        )
    if hashlib.sha256(raw).hexdigest() != sha:
        raise DrizzleCheckpointError(f"{what} SHA-256 mismatch")
    try:
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
    except (ValueError, OSError) as exc:
        raise DrizzleCheckpointError(
            f"{what} cannot load array: {exc}"
        ) from exc
    arr = np.asarray(arr)
    if arr.dtype != np.float32:
        raise DrizzleCheckpointError(
            f"{what} array dtype {arr.dtype} != float32"
        )
    if arr.ndim != 2 or tuple(arr.shape) != tuple(output_shape_hw):
        raise DrizzleCheckpointError(
            f"{what} array shape {arr.shape} != {tuple(output_shape_hw)}"
        )
    if not np.all(np.isfinite(arr)):
        raise DrizzleCheckpointError(
            f"{what} array contains non-finite samples"
        )
    if np.any(arr < 0.0):
        raise DrizzleCheckpointError(
            f"{what} array contains negative samples"
        )
    return np.array(arr, dtype=np.float32, copy=True)


def _validate_channel_vs_canonical(config, channels):
    """Require every channel's deposition params to equal the canonical config.

    The reader already validates the canonical config / fingerprint / digest
    and the channel entries *separately*; this closes the remaining gap where a
    manifest-only edit of a channel's ``kernel`` / ``pixfrac`` / ``fillval``
    (fingerprint still valid) would reconstruct with the wrong deposition
    parameters.  Fails closed before any accumulator is reconstructed.
    """
    scientific = config.scientific
    for ch in channels:
        _check_deposition_matches_canonical(
            ch["kernel"], ch["pixfrac"], ch["fillval"], scientific,
            f"channel {ch['channel']}",
        )


def _validate_artifact(desc, generation, channel, kind, ckpt_dir, output_shape_hw):
    """Validate one artifact descriptor and load its array (private float32)."""
    file_name = desc.get("file")
    if not isinstance(file_name, str) or not file_name:
        raise DrizzleCheckpointError(
            f"channel {channel} {kind} has invalid file name"
        )
    # Unsafe name / path traversal: must be a plain basename on the allowlist.
    if (
        file_name != os.path.basename(file_name)
        or file_name.startswith(("/", "\\"))
        or ".." in file_name
    ):
        raise DrizzleCheckpointError(
            f"channel {channel} {kind} has unsafe file name {file_name!r}"
        )
    m = _CHANNEL_ARTIFACT_RE.match(file_name)
    if not m:
        raise DrizzleCheckpointError(
            f"channel {channel} {kind} has non-allowlisted file name "
            f"{file_name!r}"
        )
    if int(m.group(1)) != generation:
        raise DrizzleCheckpointError(
            f"channel {channel} {kind} file generation {m.group(1)} != "
            f"manifest generation {generation}"
        )
    if int(m.group(2)) != channel:
        raise DrizzleCheckpointError(
            f"channel {channel} {kind} file channel {m.group(2)} != {channel}"
        )
    expected_kind = "img" if kind == "out_img" else "wht"
    if m.group(3) != expected_kind:
        raise DrizzleCheckpointError(
            f"channel {channel} {kind} file kind {m.group(3)} != {expected_kind}"
        )

    if desc.get("dtype") != "float32":
        raise DrizzleCheckpointError(
            f"channel {channel} {kind} dtype {desc.get('dtype')!r} != 'float32'"
        )
    shape = desc.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise DrizzleCheckpointError(
            f"channel {channel} {kind} shape must be a 2-element list"
        )
    sh = (
        _strict_int(shape[0], "shape[0]"),
        _strict_int(shape[1], "shape[1]"),
    )
    if sh != tuple(output_shape_hw):
        raise DrizzleCheckpointError(
            f"channel {channel} {kind} shape {sh} != output_shape_hw "
            f"{tuple(output_shape_hw)}"
        )
    size = desc.get("size")
    if isinstance(size, bool) or not isinstance(size, int):
        raise DrizzleCheckpointError(
            f"channel {channel} {kind} size must be a strict integer"
        )
    sha = desc.get("sha256")
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or any(ch not in "0123456789abcdef" for ch in sha)
    ):
        raise DrizzleCheckpointError(
            f"channel {channel} {kind} sha256 must be a 64-char hex string"
        )

    path = os.path.join(ckpt_dir, file_name)
    if os.path.islink(path):
        raise DrizzleCheckpointError(
            f"channel {channel} {kind} artifact {file_name!r} is a symlink"
        )
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise DrizzleCheckpointError(
            f"channel {channel} {kind} artifact missing/unreadable: {exc}"
        ) from exc
    if len(raw) != size:
        raise DrizzleCheckpointError(
            f"channel {channel} {kind} size mismatch: manifest {size} vs "
            f"disk {len(raw)}"
        )
    if hashlib.sha256(raw).hexdigest() != sha:
        raise DrizzleCheckpointError(
            f"channel {channel} {kind} SHA-256 mismatch"
        )
    try:
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
    except (ValueError, OSError) as exc:
        raise DrizzleCheckpointError(
            f"channel {channel} {kind} cannot load array: {exc}"
        ) from exc
    arr = np.asarray(arr)
    if arr.dtype != np.float32:
        raise DrizzleCheckpointError(
            f"channel {channel} {kind} array dtype {arr.dtype} != float32"
        )
    if arr.ndim != 2 or tuple(arr.shape) != tuple(output_shape_hw):
        raise DrizzleCheckpointError(
            f"channel {channel} {kind} array shape {arr.shape} != "
            f"{tuple(output_shape_hw)}"
        )
    if not np.all(np.isfinite(arr)):
        raise DrizzleCheckpointError(
            f"channel {channel} {kind} array contains non-finite samples"
        )
    return np.array(arr, dtype=np.float32, copy=True)


def _validate_versions(manifest, require_exact_versions):
    """Enforce the documented exact-continuation library version policy."""
    stored_drizzle = manifest.get("drizzle_lib_version")
    stored_numpy = manifest.get("numpy_version")
    if not isinstance(stored_drizzle, str) or not isinstance(stored_numpy, str):
        raise DrizzleCheckpointError(
            "manifest drizzle_lib_version / numpy_version must be strings"
        )
    if not require_exact_versions:
        return
    cur_drizzle = _drizzle_lib_version()
    cur_numpy = _numpy_version()
    if stored_drizzle != cur_drizzle:
        raise DrizzleCheckpointError(
            f"drizzle library version mismatch: checkpoint {stored_drizzle!r} "
            f"vs runtime {cur_drizzle!r} (exact-continuation policy)"
        )
    if stored_numpy != cur_numpy:
        raise DrizzleCheckpointError(
            f"numpy version mismatch: checkpoint {stored_numpy!r} vs runtime "
            f"{cur_numpy!r} (exact-continuation policy)"
        )


def _reconstruct_accumulators(channels, output_shape_hw, pixel_scale_ratio=None):
    """Reconstruct the three accumulators (only after full validation)."""
    accs = []
    for ch in sorted(channels, key=lambda c: c["channel"]):
        try:
            acc = DrizzleAccumulator.from_native_state(
                output_shape_hw,
                ch["out_img"],
                ch["out_wht"],
                kernel=ch["kernel"],
                pixfrac=ch["pixfrac"],
                fillval=ch["fillval"],
                total_exptime=ch["total_exptime"],
                pixel_scale_ratio=pixel_scale_ratio,
            )
        except (TypeError, ValueError) as exc:
            raise DrizzleCheckpointError(
                f"cannot reconstruct accumulator for channel "
                f"{ch['channel']}: {exc}"
            ) from exc
        accs.append(acc)
    return accs


def _reconstruct_support(support, output_shape_hw):
    """Reconstruct optional SUP_W1/SUP_W2 only after full validation."""
    if support is None:
        return None
    accumulators = []
    zeros = np.zeros(output_shape_hw, dtype=np.float32)
    for kind in ("w1", "w2"):
        try:
            acc = DrizzleAccumulator.from_native_state(
                output_shape_hw,
                zeros,
                support[kind],
                kernel=support["kernel"],
                pixfrac=support["pixfrac"],
                fillval=support["fillval"],
                total_exptime=support["total_exptime"],
            )
        except (TypeError, ValueError) as exc:
            raise DrizzleCheckpointError(
                f"cannot reconstruct support accumulator {kind}: {exc}"
            ) from exc
        accumulators.append(acc)
    return tuple(accumulators)
