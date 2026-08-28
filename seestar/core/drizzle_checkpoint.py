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

Copy-on-write / commit protocol
-------------------------------

Every generation writes its six array artifacts under generation-unique final
names (same-directory temp file + ``os.replace``), computes a SHA-256 and exact
byte size for each, then writes ``checkpoint.json.tmp`` (fsync) and
``os.replace``-s it to ``checkpoint.json`` **last**.  ``checkpoint.json`` is
the single commit point: a crash before that replace leaves the prior manifest
and every file it references byte-identical and usable; the attempt's temp /
uncommitted files are cleaned best-effort.  After a successful commit, stale
previous generations may be garbage-collected only from the explicit
writer-owned ``gen-*.npy`` allowlist pattern — never a broad directory delete
and never the current generation.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile

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

# Same-directory temp-file prefix used for the array artifacts (mkstemp).
_ARRAY_TMP_PREFIX = ".tmp-"
_ARRAY_TMP_SUFFIX = ".npy"


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
        self.output_shape_hw = tuple(int(v) for v in output_shape_hw)

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

        self._dir = os.path.join(self.output_dir, CHECKPOINT_DIRNAME)
        self._next_generation = 1
        self._current_generation = 0

    # ------------------------------------------------------------------ state
    @property
    def has_committed(self) -> bool:
        return self._current_generation > 0

    @property
    def current_generation(self) -> int:
        return self._current_generation

    def _artifact_name(self, generation: int, channel: int, kind: str) -> str:
        return f"gen-{int(generation):08d}-ch{int(channel)}-out_{kind}.npy"

    # -------------------------------------------------------------- validation
    def _snapshot_channels(self, accumulators):
        """Own and validate the three native accumulator buffers.

        Returns a list of per-channel snapshot dicts, each holding owned float32
        copies (never aliased to the live engine buffers) plus the exact
        kernel/pixfrac/fillval/total_exptime.  Fail closed on any inconsistency.
        """
        accs = list(accumulators or [])
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
            pixfrac = float(getattr(acc, "pixfrac", 1.0))
            fillval = getattr(acc, "fillval", None)
            total = float(getattr(acc, "_total_exptime", 0.0))
            if ref_kernel is None:
                ref_kernel, ref_pixfrac, ref_fillval, ref_total = (
                    kernel, pixfrac, fillval, total,
                )
            if kernel != ref_kernel or pixfrac != ref_pixfrac or fillval != ref_fillval:
                raise DrizzleCheckpointError(
                    f"inconsistent per-channel drizzle config at channel {c}"
                )
            if not np.isfinite(total) or total < 0.0:
                raise DrizzleCheckpointError(
                    f"non-finite/negative total_exptime {total!r} at channel {c}"
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

    @staticmethod
    def _validate_counters(counters):
        frame_count = int(counters.get("frame_count", 0) or 0)
        if frame_count <= 0:
            raise DrizzleCheckpointError(
                "refusing to publish an empty checkpoint (frame_count <= 0)"
            )
        stacked = int(counters.get("stacked_batches_count", 0) or 0)
        if stacked < 0 or frame_count < 0:
            raise DrizzleCheckpointError("negative counter in checkpoint metadata")

        total_exp = float(counters.get("total_exposure_seconds", 0.0) or 0.0)
        if not np.isfinite(total_exp) or total_exp < 0.0:
            raise DrizzleCheckpointError(
                f"non-finite/negative total_exposure_seconds {total_exp!r}"
            )
        unknown = int(counters.get("exposure_unknown_count", 0) or 0)
        if unknown < 0:
            raise DrizzleCheckpointError("negative exposure_unknown_count")

        exp_min = counters.get("exposure_min", None)
        exp_max = counters.get("exposure_max", None)
        for name, val in (("exposure_min", exp_min), ("exposure_max", exp_max)):
            if val is not None:
                f = float(val)
                if not np.isfinite(f):
                    raise DrizzleCheckpointError(f"non-finite {name} {val!r}")

        return {
            "frame_count": frame_count,
            "stacked_batches_count": stacked,
            "total_exposure_seconds": total_exp,
            "exposure_unknown_count": unknown,
            "exposure_min": exp_min,
            "exposure_max": exp_max,
        }

    @staticmethod
    def _validate_session_binding(session_binding):
        sb = session_binding or {}
        roots = sb.get("input_roots")
        if not isinstance(roots, list) or not roots:
            raise DrizzleCheckpointError("missing session input_roots")
        reference = sb.get("reference")
        if not isinstance(reference, dict) or not reference.get("path"):
            raise DrizzleCheckpointError("missing session reference identity")
        plan = sb.get("plan")
        if not isinstance(plan, dict) or not isinstance(plan.get("sources"), list):
            raise DrizzleCheckpointError("missing session observation plan")
        if not plan["sources"]:
            raise DrizzleCheckpointError("session observation plan is empty")
        return {
            "input_roots": [_json_scalar(r) for r in roots],
            "reference": reference,
            "plan": plan,
        }

    @staticmethod
    def _validate_ledger(completed_sources):
        ledger = list(completed_sources or [])
        seen = set()
        for entry in ledger:
            if not isinstance(entry, dict):
                raise DrizzleCheckpointError(
                    "non-identity entry in completed ledger"
                )
            path = entry.get("path")
            size = entry.get("size")
            mtime = entry.get("mtime_ns")
            if (
                not path
                or not isinstance(size, int)
                or not isinstance(mtime, int)
            ):
                raise DrizzleCheckpointError(
                    f"unstattable source in completed ledger: {entry.get('name')}"
                )
            key = (
                path,
                size,
                mtime,
            )
            if key in seen:
                raise DrizzleCheckpointError(
                    f"duplicate source identity in completed ledger: "
                    f"{entry.get('name')}"
                )
            seen.add(key)
        return ledger

    # ------------------------------------------------------------------ writes
    @staticmethod
    def _npy_bytes(arr):
        """Serialize a float32 array to the exact ``.npy`` file bytes."""
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue()

    def _write_array_artifact(self, arr, final_name):
        """Write one native array to a generation-unique final name, atomically.

        Same-directory temp file + ``os.replace``; fsync before replace so the
        bytes are durable before the (later) manifest commit references them.
        Returns the exact bytes written (the file content), so the caller can
        record a SHA-256 / size over the *final artifact* itself.
        """
        data = self._npy_bytes(arr)
        path = os.path.join(self._dir, final_name)
        fd, tmp = tempfile.mkstemp(
            dir=self._dir, prefix=_ARRAY_TMP_PREFIX, suffix=_ARRAY_TMP_SUFFIX
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
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
        """Write ``checkpoint.json.tmp``, fsync, then ``os.replace`` (the commit)."""
        os.makedirs(self._dir, exist_ok=True)
        tmp_path = os.path.join(self._dir, MANIFEST_TMP_FILENAME)
        manifest_path = os.path.join(self._dir, MANIFEST_FILENAME)
        payload = json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False)
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, manifest_path)

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

    def _cleanup_attempt(self, final_names):
        """Best-effort cleanup of this attempt's uncommitted artifacts.

        Removes the manifest temp, any leftover ``mkstemp`` temp files (our
        prefix/suffix) and the generation-unique final names written by this
        attempt.  Never touches unrelated files.
        """
        try:
            os.unlink(os.path.join(self._dir, MANIFEST_TMP_FILENAME))
        except OSError:
            pass
        for name in final_names:
            try:
                os.unlink(os.path.join(self._dir, name))
            except OSError:
                pass
        try:
            for name in os.listdir(self._dir):
                if name.startswith(_ARRAY_TMP_PREFIX) and name.endswith(
                    _ARRAY_TMP_SUFFIX
                ):
                    try:
                        os.unlink(os.path.join(self._dir, name))
                    except OSError:
                        pass
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
            Ordered ledger of accepted source identities.

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
        # 1. Validate everything *before* any write (fail closed, never a
        #    partial/mixed generation).
        counters_clean = self._validate_counters(counters)
        session_clean = self._validate_session_binding(session_binding)
        ledger_clean = self._validate_ledger(completed_sources)
        snapshots = self._snapshot_channels(accumulators)

        generation = int(self._next_generation)
        final_names = [
            self._artifact_name(generation, c, kind)
            for c in range(3)
            for kind in ("img", "wht")
        ]

        try:
            os.makedirs(self._dir, exist_ok=True)
            # 2. Write the six generation-unique array artifacts.
            written = []  # (final_name, sha256, size)
            for snap in snapshots:
                for kind in ("out_img", "out_wht"):
                    arr = snap[kind]
                    short = "img" if kind == "out_img" else "wht"
                    name = self._artifact_name(generation, snap["channel"], short)
                    file_bytes = self._write_array_artifact(arr, name)
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

            # 3. Persist the canonical config before the manifest.
            self._write_cfg()

            # 4. Build the deterministic manifest.
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

            # 5. Commit the manifest LAST.
            self._write_manifest(manifest)

        except BaseException as exc:
            # Best-effort cleanup of this attempt's uncommitted files, without
            # touching the prior committed generation or unrelated files.
            self._cleanup_attempt(final_names)
            if isinstance(exc, DrizzleCheckpointError):
                raise
            raise DrizzleCheckpointError(
                f"drizzle checkpoint persist failed: {exc}"
            ) from exc

        # 6. Advance the generation and best-effort GC older generations.
        self._current_generation = generation
        self._next_generation = generation + 1
        self._gc_stale_generations(generation)
        return generation
