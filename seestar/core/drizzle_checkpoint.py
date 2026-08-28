"""Production write-only checkpoint for the native M3 Drizzle accumulator state.

RSM2-D1: this module persists an *exact, self-describing* snapshot of the three
per-channel :class:`~seestar.core.drizzle_core.DrizzleAccumulator` native
buffers (``out_img`` weighted-mean science and ``out_wht`` total signed weight),
plus the runtime-effective scientific configuration, the output WCS/grid and the
session/source ledger, at safe accepted-pose boundaries.

It is **write-only**: there is deliberately no reader, no ``open``/``resume``
path and no ``finalize``/preview/derived-SCI persistence in this module.  The
native float32 buffers are persisted bit-exactly (the signed Lanczos WHT is
never abs'ed / clipped / thresholded), so a future reader can reconstruct the
accumulators with :meth:`DrizzleAccumulator.from_native_state` and continue
deposition bit-identically (proved by
``tests/test_drizzle_resume_continuation.py``).

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

import hashlib
import io
import itertools
import json
import os
import re
import secrets

import numpy as np

from seestar import run_contract

__all__ = [
    "DrizzleCheckpointError",
    "DrizzleCheckpointWriter",
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

        try:
            shape = tuple(int(v) for v in output_shape_hw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DrizzleCheckpointError(f"invalid output_shape_hw: {exc}") from exc
        if len(shape) != 2:
            raise DrizzleCheckpointError(
                f"output_shape_hw must be (H, W), got {shape!r}"
            )
        self.output_shape_hw = shape

        self._dir = os.path.join(self.output_dir, CHECKPOINT_DIRNAME)

        # Restart safety: a fresh-run writer must never reuse a prior checkpoint
        # namespace.  Refuse (fail closed, preserve every pre-existing byte)
        # before any write if the dedicated namespace is non-empty.
        self._refuse_existing_checkpoint()

        # Fail closed *before* any write: a malformed canonical config (missing
        # effective field) or an unserializable WCS must never yield a writer
        # that later publishes an unusable manifest.
        try:
            self.fingerprint = run_contract.drizzle_fingerprint(self.canonical_cfg)
        except run_contract.ConfigError as exc:
            raise DrizzleCheckpointError(
                f"malformed canonical Drizzle config: {exc}"
            ) from exc
        self.run_config_digest = self.canonical_cfg.full_digest()
        self.scientific_config = dict(self.canonical_cfg.scientific)
        self._wcs_dict = serialize_wcs_header(output_wcs)

        self._next_generation = 1
        self._current_generation = 0
        # Manifest temp owned by the *current* commit attempt (if any).  Set by
        # ``_claim_manifest_temp`` and cleared on replace/cleanup, so cleanup can
        # only ever remove the temp this attempt created — never a foreign one.
        self._manifest_tmp_path = None

    # ------------------------------------------------------------------ state
    @property
    def has_committed(self) -> bool:
        return self._current_generation > 0

    @property
    def current_generation(self) -> int:
        return self._current_generation

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

            # 3. Preflight strict-JSON serialization of the non-artifact payload.
            self._preflight_json_payload(counters_clean, session_clean, ledger_clean)

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
        self._current_generation = generation
        self._next_generation = generation + 1
        self._gc_stale_generations(generation)
        return generation
