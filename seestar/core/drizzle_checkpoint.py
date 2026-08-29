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
    <output>/run_config.cfg              # canonical schema-v2 run config (stable)

This namespace is **dedicated** to Drizzle and never reuses the classic
``memmap_accumulators/resume_manifest.json`` SUM/W artifacts (which remain
plain-classic only and are never overloaded or weakened).

Restart safety (write-only, no Resume)
--------------------------------------

Drizzle Resume is disabled in D1, so a checkpoint namespace can never be
continued.  A freshly constructed writer therefore **refuses** (fail closed,
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

Every generation writes its six array artifacts under generation-unique final
names claimed **exclusively** (``O_CREAT | O_EXCL`` + in-place write + fsync),
computes a SHA-256 and exact byte size for each, then writes
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
from seestar.core.drizzle_core import DrizzleAccumulator, VALID_DRIZZLE_KERNELS

__all__ = [
    "DrizzleCheckpointError",
    "DrizzleCheckpointWriter",
    "DrizzleCheckpointResult",
    "DrizzleContinuation",
    "read_drizzle_checkpoint",
    "build_drizzle_canonical_config",
    "serialize_wcs_header",
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
_OUTPUT_GRID_CONTRACT = "m3_output_grid_v1"
_OUTPUT_GRID_CONTRACT_VERSION = 1
_REGISTRATION_CONTRACT = "m3_tf_registration_v1"
_REGISTRATION_CONTRACT_VERSION = 1

# Explicit allowlist for generation-unique array artifacts.  Garbage collection
# and failure cleanup may only ever touch names matching this pattern.
_ARTIFACT_RE = re.compile(r"^gen-(\d{8})-ch([0-2])-out_(img|wht)\.npy$")

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
    return out


def build_drizzle_canonical_config(qm, product_version: str = "") -> run_contract.RunConfig:
    """Build the canonical schema-v2 :class:`run_contract.RunConfig` for a
    Drizzle run from the runtime-effective engine state.

    ``run_contract.drizzle_fingerprint`` requires every effective Drizzle field
    (fail closed, never a partial payload); this helper supplies all of them
    from the engine instance plus the stable D1 contract tokens.  It performs
    no I/O.  The ``product_version`` is supplied by the caller (the engine's
    ``_canonical_product_version``).
    """
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
        "background_match_contract": _BACKGROUND_MATCH_CONTRACT,
        "background_match_contract_version": _BACKGROUND_MATCH_CONTRACT_VERSION,
        "output_grid_contract": _OUTPUT_GRID_CONTRACT,
        "output_grid_contract_version": _OUTPUT_GRID_CONTRACT_VERSION,
        "registration_contract": _REGISTRATION_CONTRACT,
        "registration_contract_version": _REGISTRATION_CONTRACT_VERSION,
    }
    provenance = {"drizzle_lib_version": _drizzle_lib_version()}
    return run_contract.RunConfig.from_sections(
        product_version=product_version,
        scientific=scientific,
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
        fresh = read_drizzle_checkpoint(output_dir, require_exact_versions=True)
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
            session=copy.deepcopy(fresh.session),
            counters=copy.deepcopy(fresh.counters),
            completed_sources=copy.deepcopy(fresh.completed_sources),
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
        return {
            "generation": int(fresh.generation),
            "session": copy.deepcopy(fresh.session),
            "counters": copy.deepcopy(fresh.counters),
            "completed": copy.deepcopy(fresh.completed_sources),
            "channel_total_exptime": per_channel_total,
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

    # ------------------------------------------------------------ restart
    def _refuse_existing_checkpoint(self):
        """Refuse a non-empty existing checkpoint namespace (fail closed).

        Drizzle Resume is disabled in D1: any prior manifest, allowlisted
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
            f"{self._dir!r} (found {', '.join(sorted(set(found)))}); Drizzle "
            "resume is disabled — a fresh run must use an empty or absent "
            "checkpoint directory"
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
        """
        if not isinstance(counters, dict):
            raise DrizzleCheckpointError("counters must be a mapping")
        frame_count = _strict_int(counters.get("frame_count", 0), "frame_count")
        if frame_count <= 0:
            raise DrizzleCheckpointError(
                "refusing to publish an empty checkpoint (frame_count <= 0)"
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

    @staticmethod
    def _validate_manifest_consistency(counters_clean, session_clean, ledger_clean):
        """Enforce the self-consistent manifest ledger/plan/counter invariant.

        Under the current Drizzle runtime ``stacked_batches_count`` increments
        once per accepted pose, so it must equal ``frame_count``; the completed
        ledger must be exactly the ordered plan prefix of length ``frame_count``.
        """
        frame_count = counters_clean["frame_count"]
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
        if ledger_clean != plan_sources[:frame_count]:
            raise DrizzleCheckpointError(
                "completed_sources is not the exact ordered prefix of the "
                "session plan"
            )

    def _check_monotonic_extension(self, counters_clean, session_clean,
                                   ledger_clean, snapshots):
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
        * the completed ledger must keep the loaded ledger as its exact prefix.

        This runs *before* any write, inside the commit try-block, so a
        divergent continuation is refused with the previous committed
        generation (and every file it references) byte-identical.
        """
        if self._continuation_state is None:
            return
        loaded = self._continuation_state
        if session_clean != loaded["session"]:
            raise DrizzleCheckpointError(
                "continuation session binding diverges from the loaded "
                "checkpoint (input_roots/reference/plan must be identical)"
            )
        loaded_counters = loaded["counters"]
        new_frame = counters_clean["frame_count"]
        loaded_frame = loaded_counters["frame_count"]
        if new_frame <= loaded_frame:
            raise DrizzleCheckpointError(
                f"continuation must extend the loaded checkpoint: frame_count "
                f"{new_frame} <= loaded frame_count {loaded_frame}"
            )
        self._check_cumulative_counters(loaded_counters, counters_clean)
        self._check_channel_total_monotonic(loaded, snapshots)

        loaded_ledger = loaded["completed"]
        if ledger_clean[: len(loaded_ledger)] != loaded_ledger:
            raise DrizzleCheckpointError(
                "continuation completed_sources must preserve the exact loaded "
                "ledger prefix (no rewrite/reorder/divergent prefix)"
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

    def _build_next_continuation_state(self, generation, counters_clean,
                                       session_clean, ledger_clean, snapshots):
        """Build (deep-copied) the continuation baseline for the next commit.

        Pure and fallible: the deep copies and the per-channel total-exposure
        list are materialized here, during preflight, so any allocation failure
        (e.g. ``MemoryError``) is raised *before* any artifact write or manifest
        commit.  Returns the exact next ``_continuation_state`` dict, which the
        caller assigns by reference **only after** the manifest commits.
        """
        return {
            "generation": int(generation),
            "session": copy.deepcopy(session_clean),
            "counters": copy.deepcopy(counters_clean),
            "completed": copy.deepcopy(ledger_clean),
            "channel_total_exptime": [
                float(s["total_exptime"]) for s in snapshots
            ],
        }

    def _preflight_json_payload(self, counters_clean, session_clean, ledger_clean):
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
    def commit(self, accumulators, *, session_binding, counters, completed_sources):
        """Persist one generation and atomically commit the manifest.

        Parameters
        ----------
        accumulators :
            The three per-channel :class:`DrizzleAccumulator` instances.
        session_binding :
            ``{"input_roots": [...], "reference": {...}, "plan": {...}}``.
        counters :
            ``{"frame_count": int, "stacked_batches_count": int,
            "total_exposure_seconds": float, "exposure_unknown_count": int,
            "exposure_min": float|None, "exposure_max": float|None}``.
        completed_sources :
            Ordered ledger of accepted source identities (must equal the exact
            ordered plan prefix of length ``frame_count``).

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
            ledger_clean = self._validate_ledger(completed_sources)
            snapshots = self._snapshot_channels(accumulators)

            # 2. Manifest self-consistency invariants (truthful
            #    ledger/plan/counter).
            self._validate_manifest_consistency(
                counters_clean, session_clean, ledger_clean
            )

            # 2b. Continuation monotonicity: a re-armed writer must extend the
            #     loaded checkpoint (never roll back / rewrite / reorder /
            #     diverge / roll back cumulative counters or native per-channel
            #     total exposure).  No-op for a fresh-run writer.
            self._check_monotonic_extension(
                counters_clean, session_clean, ledger_clean, snapshots
            )

            # 3. Preflight strict-JSON serialization of the non-artifact payload.
            self._preflight_json_payload(counters_clean, session_clean, ledger_clean)

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
                    snapshots,
                )

            os.makedirs(self._dir, exist_ok=True)
            # 4. Write the six generation-unique array artifacts (exclusive
            #    claim, never overwrite a pre-existing path).
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
    accumulated (== ``frame_count``, because ``completed_sources`` is the exact
    ordered prefix of the session plan).  Suitable for later D2B lifecycle
    wiring (which is *not* performed here).

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
    next_source_index: int
    generation: int
    source_output_dir: str

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
    session: dict               # fresh loaded session binding (baseline)
    counters: dict              # fresh loaded counters (baseline)
    completed_sources: list     # fresh loaded ledger (baseline)
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


def read_drizzle_checkpoint(output_dir, *, require_exact_versions=True):
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
    exactly.  The completed ledger must remain the exact ordered prefix of the
    session plan.  The three accumulators are reconstructed only after the
    entire checkpoint validates (no partial externally visible restore).

    Parameters
    ----------
    output_dir :
        Explicit output / run directory containing the checkpoint namespace.
    require_exact_versions :
        When ``True`` (default) a persisted ``drizzle_lib_version`` /
        ``numpy_version`` different from the runtime library is refused
        (documented exact-continuation policy).

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
    ledger = _validate_ledger(manifest, session, counters)
    channels = _validate_channels(manifest, generation, ckpt_dir, output_shape_hw)
    _validate_channel_vs_canonical(config, channels)
    _validate_versions(manifest, require_exact_versions)

    # Reconstruct only after the entire checkpoint validated.
    accumulators = _reconstruct_accumulators(channels, output_shape_hw)

    return DrizzleCheckpointResult(
        manifest=manifest,
        session=session,
        counters=counters,
        completed_sources=ledger,
        config=config,
        wcs=wcs,
        output_shape_hw=output_shape_hw,
        accumulators=accumulators,
        next_source_index=counters["frame_count"],
        generation=generation,
        source_output_dir=output_dir,
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


def _reconstruct_wcs(manifest, output_shape_hw):
    """Reconstruct and validate the output WCS; attach ``array_shape``."""
    wcs_dict = manifest.get("wcs")
    if not isinstance(wcs_dict, dict) or not wcs_dict:
        raise DrizzleCheckpointError("manifest wcs is missing or malformed")

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
            f"cannot reconstruct output WCS: {exc}"
        ) from exc
    if wcs.naxis != 2:
        raise DrizzleCheckpointError(
            f"output WCS has naxis {wcs.naxis} != 2"
        )
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
    if frame_count <= 0:
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

    # Re-stat the persisted reference and every plan source (must match).
    _restat_identity(reference, "session reference")
    for ident in sources_clean:
        _restat_identity(ident, "session plan source")

    return {
        "input_roots": roots_clean,
        "reference": reference,
        "plan": plan_clean,
    }


def _validate_ledger(manifest, session, counters):
    """Validate the completed ledger and require the exact ordered plan prefix."""
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

    frame_count = counters["frame_count"]
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
    if ledger != plan_sources[:frame_count]:
        raise DrizzleCheckpointError(
            "completed_sources is not the exact ordered prefix of the session plan"
        )
    # Defense-in-depth: re-stat every completed ledger source too.
    for ident in ledger:
        _restat_identity(ident, "completed ledger source")
    return ledger


def _validate_channels(manifest, generation, ckpt_dir, output_shape_hw):
    """Validate the channel table and load every artifact (fail closed).

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

    return channels_clean


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
    m = _ARTIFACT_RE.match(file_name)
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


def _reconstruct_accumulators(channels, output_shape_hw):
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
            )
        except (TypeError, ValueError) as exc:
            raise DrizzleCheckpointError(
                f"cannot reconstruct accumulator for channel "
                f"{ch['channel']}: {exc}"
            ) from exc
        accs.append(acc)
    return accs
