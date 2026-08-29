"""Qt-independent resume locator + per-run config restore helper (RSM2-02C).

This module turns an explicit user "Resume" selection (a previous-stack FITS
path, or an output/run directory) into the owning run directory and, when the
recognized scientific checkpoint exists, a restored per-run configuration.

It is deliberately toolkit- and engine-independent: only the Python standard
library plus :mod:`seestar.run_contract` (itself pure stdlib) are imported, so
it can be imported and unit-tested in complete isolation and never pulls in
``numpy``/``astropy``/Qt/Tk.

Contract (Resume Contract v2 — GUI convenience / first protection layer):

* A FIT or a directory is a *human locator* only — it is never, by itself, a
  checkpoint.
* The owning run directory is resolved deterministically and boundedly (walking
  up from the selected path) by recognized state: ``run_config.cfg``, the
  Classic ``memmap_accumulators/resume_manifest.json`` manifest, and/or the
  standard-Drizzle ``.m3d_checkpoint/checkpoint.json`` manifest.
* The preferred configuration source is the schema-v2 ``run_config.cfg``
  (read via :func:`run_contract.read_cfg`).  A legacy 5.x ``.cfg`` is a
  fallback *restoration object* only, never checkpoint evidence.
* Resume may proceed (status :data:`STATUS_READY`) only when BOTH a recognized
  checkpoint manifest exists AND a restorable config is available.  Classic
  may use v2 or migrated legacy config; Drizzle requires a coherent v2 config.
* Every other case fails closed with a stable status + bounded reason; no file
  is written, no setting is invented, and no hidden resume intent is produced.

The scientific backend remains the independent validation authority; this
module is the convenience/first-protection layer only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

from seestar import run_contract

__all__ = [
    "MEMORY_DIR",
    "RESUME_MANIFEST_FILENAME",
    "DRIZZLE_CHECKPOINT_DIR",
    "DRIZZLE_MANIFEST_FILENAME",
    "RUN_CONFIG_FILENAME",
    "MAX_SEARCH_DEPTH",
    "STATUS_READY",
    "STATUS_NO_RUN_DIR",
    "STATUS_NO_CHECKPOINT",
    "STATUS_CONFIG_UNAVAILABLE",
    "STATUS_CORRUPT_CONFIG",
    "STATUS_UNSAFE_CONFIG",
    "STATUS_AMBIGUOUS_LEGACY",
    "STATUS_AMBIGUOUS_CHECKPOINT",
    "STATUS_CORRUPT_CHECKPOINT",
    "CHECKPOINT_KIND_CLASSIC",
    "CHECKPOINT_KIND_DRIZZLE",
    "STATUS_REASON_KEYS",
    "DiscoveryResult",
    "manifest_path",
    "drizzle_manifest_path",
    "run_config_path",
    "has_manifest",
    "has_run_config",
    "recognized_state",
    "resolve_run_directory",
    "find_legacy_cfg",
    "discover_resume",
    "restore_to_settings",
]

MEMORY_DIR = "memmap_accumulators"
RESUME_MANIFEST_FILENAME = "resume_manifest.json"
DRIZZLE_CHECKPOINT_DIR = ".m3d_checkpoint"
DRIZZLE_MANIFEST_FILENAME = "checkpoint.json"
RUN_CONFIG_FILENAME = "run_config.cfg"

CHECKPOINT_KIND_CLASSIC = "classic"
CHECKPOINT_KIND_DRIZZLE = "drizzle"

# Bounded upward search from a selected FIT / directory to its owning run
# directory.  Small on purpose: a previous stack lives at (or within a couple
# of levels of) its run directory.
MAX_SEARCH_DEPTH = 6

# Stable discovery statuses.  ``STATUS_READY`` is the only status that may
# produce a Resume run intent.
STATUS_READY = "ready"
STATUS_NO_RUN_DIR = "no_run_dir"
STATUS_NO_CHECKPOINT = "no_checkpoint"
STATUS_CONFIG_UNAVAILABLE = "config_unavailable"
STATUS_CORRUPT_CONFIG = "corrupt_config"
STATUS_UNSAFE_CONFIG = "unsafe_config"
STATUS_AMBIGUOUS_LEGACY = "ambiguous_legacy"
STATUS_AMBIGUOUS_CHECKPOINT = "ambiguous_checkpoint"
STATUS_CORRUPT_CHECKPOINT = "corrupt_checkpoint"

# Status -> localization key for the GUI warning body.  Every non-ready status
# maps to one bounded, actionable message (the GUI owns the FR/EN text).
STATUS_REASON_KEYS = {
    STATUS_NO_RUN_DIR: "resume_refuse_no_run",
    STATUS_NO_CHECKPOINT: "resume_refuse_no_checkpoint",
    STATUS_CONFIG_UNAVAILABLE: "resume_refuse_config_unavailable",
    STATUS_CORRUPT_CONFIG: "resume_refuse_corrupt_config",
    STATUS_UNSAFE_CONFIG: "resume_refuse_unsafe_config",
    STATUS_AMBIGUOUS_LEGACY: "resume_refuse_ambiguous_legacy",
    STATUS_AMBIGUOUS_CHECKPOINT: "resume_refuse_ambiguous_checkpoint",
    STATUS_CORRUPT_CHECKPOINT: "resume_refuse_corrupt_checkpoint",
}

# Bound the technical detail surfaced on a failed discovery: it is data from an
# external file (field names / types only — never values, never secrets), so it
# is kept short and never dumped wholesale into the UI.
_MAX_DETAIL_CHARS = 400


@dataclass
class DiscoveryResult:
    """Outcome of locating a run directory and discovering its configuration.

    ``config`` is a validated :class:`run_contract.RunConfig` (``None`` when
    absent/unrecoverable); ``config_source`` is ``"v2"`` (``run_config.cfg``)
    or ``"legacy"`` (migrated 5.x ``.cfg``).  ``reason_key`` is the stable
    localization key for the refusal body (``None`` for :data:`STATUS_READY`);
    ``detail`` is an optional bounded technical note (data only, no secrets).
    """

    status: str
    run_dir: Optional[str] = None
    config: Optional[run_contract.RunConfig] = None
    config_source: Optional[str] = None
    checkpoint_kind: Optional[str] = None
    reason_key: Optional[str] = None
    detail: Optional[str] = None


def manifest_path(run_dir: Any) -> str:
    """Return the checkpoint-manifest path under ``run_dir``."""
    return os.path.join(os.fspath(run_dir), MEMORY_DIR, RESUME_MANIFEST_FILENAME)


def drizzle_manifest_path(run_dir: Any) -> str:
    """Return the standard-Drizzle checkpoint-manifest path."""
    return os.path.join(
        os.fspath(run_dir), DRIZZLE_CHECKPOINT_DIR, DRIZZLE_MANIFEST_FILENAME
    )


def run_config_path(run_dir: Any) -> str:
    """Return the schema-v2 ``run_config.cfg`` path under ``run_dir``."""
    return os.path.join(os.fspath(run_dir), RUN_CONFIG_FILENAME)


def has_manifest(run_dir: Any) -> bool:
    """True when the recognized checkpoint manifest exists under ``run_dir``.

    A coarse presence signal (mirrors the backend's coarse ``_can_resume``
    check); the backend performs the authoritative validation.  A FIT or a CFG
    is never a substitute for this manifest.
    """
    return os.path.isfile(manifest_path(run_dir))


def has_run_config(run_dir: Any) -> bool:
    """True when a schema-v2 ``run_config.cfg`` exists under ``run_dir``."""
    return os.path.isfile(run_config_path(run_dir))


def recognized_state(run_dir: Any) -> bool:
    """True when ``run_dir`` carries recognized per-run state.

    Recognized state is ``run_config.cfg``, the Classic resume manifest and/or
    the standard-Drizzle checkpoint manifest — the facts a locator may use to
    claim a directory is (or is inside) a run directory.
    """
    return (
        has_run_config(run_dir)
        or has_manifest(run_dir)
        or os.path.isfile(drizzle_manifest_path(run_dir))
    )


def _strict_json(path: str) -> Any:
    """Read strict UTF-8 JSON, rejecting non-standard NaN/Infinity tokens."""

    def _reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh, parse_constant=_reject_constant)


def _validate_drizzle_manifest_and_config(
    manifest_file: str, config: run_contract.RunConfig
) -> Optional[str]:
    """Return a bounded refusal detail, or ``None`` when GUI readiness is safe.

    This deliberately validates only the pure-stdlib identity/config seam.  The
    backend remains authoritative for WCS, arrays, versions, ledgers and source
    resolution.  It is nevertheless strict enough that an absent/partial
    manifest or a mismatched effective CFG can never arm a Classic request.
    """
    try:
        manifest = _strict_json(manifest_file)
    except (OSError, UnicodeError, ValueError) as exc:
        return _bounded(f"invalid Drizzle checkpoint manifest: {exc}")
    if not isinstance(manifest, dict):
        return "Drizzle checkpoint manifest is not a JSON object"
    if (
        isinstance(manifest.get("schema_version"), bool)
        or manifest.get("schema_version") != 1
    ):
        return "unsupported Drizzle checkpoint schema"
    if manifest.get("mode") != "drizzle_native_v1":
        return "unrecognized Drizzle checkpoint mode"
    if manifest.get("state") != "clean":
        return "Drizzle checkpoint is not clean"
    generation = manifest.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        return "Drizzle checkpoint generation is missing or invalid"
    if not isinstance(manifest.get("scientific_config"), dict):
        return "Drizzle checkpoint scientific configuration is missing"
    if manifest.get("producer") != "zeseestarstacker":
        return "Drizzle checkpoint producer is missing"
    for field in ("drizzle_lib_version", "numpy_version"):
        if not isinstance(manifest.get(field), str):
            return f"Drizzle checkpoint {field} is missing"
    shape = manifest.get("output_shape_hw")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in shape)
    ):
        return "Drizzle checkpoint output shape is missing or invalid"
    if not isinstance(manifest.get("wcs"), dict) or not manifest["wcs"]:
        return "Drizzle checkpoint WCS is missing"
    integer_fields = (
        "frame_count",
        "stacked_batches_count",
        "exposure_unknown_count",
    )
    if any(
        isinstance(manifest.get(field), bool)
        or not isinstance(manifest.get(field), int)
        or manifest[field] < 0
        for field in integer_fields
    ):
        return "Drizzle checkpoint counters are missing or invalid"
    if not isinstance(manifest.get("total_exposure_seconds"), (int, float)):
        return "Drizzle checkpoint exposure total is missing"
    if not isinstance(manifest.get("session"), dict):
        return "Drizzle checkpoint session is missing"
    if not isinstance(manifest.get("completed_sources"), list):
        return "Drizzle checkpoint completed-source ledger is missing"
    channels = manifest.get("channels")
    if not isinstance(channels, list) or len(channels) != 3:
        return "Drizzle checkpoint channel descriptors are missing"
    checkpoint_dir = os.path.dirname(manifest_file)
    for channel_index, channel in enumerate(channels):
        if not isinstance(channel, dict) or channel.get("channel") != channel_index:
            return "Drizzle checkpoint channel descriptor is invalid"
        for artifact_kind in ("out_img", "out_wht"):
            artifact = channel.get(artifact_kind)
            if not isinstance(artifact, dict):
                return f"Drizzle checkpoint {artifact_kind} descriptor is missing"
            filename = artifact.get("file")
            if (
                not isinstance(filename, str)
                or not filename
                or os.path.basename(filename) != filename
            ):
                return f"Drizzle checkpoint {artifact_kind} filename is invalid"
            artifact_path = os.path.join(checkpoint_dir, filename)
            if not os.path.isfile(artifact_path) or os.path.islink(artifact_path):
                return f"Drizzle checkpoint {artifact_kind} artifact is missing"
    for field in ("scientific_fingerprint", "run_config_digest"):
        value = manifest.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)
        ):
            return f"Drizzle checkpoint {field} is invalid"
    try:
        fingerprint = config.drizzle_fingerprint()
    except run_contract.ConfigError as exc:
        return _bounded(exc)
    if config.product_version != manifest.get("product_version"):
        return "Drizzle checkpoint product version does not match run_config.cfg"
    if config.full_digest() != manifest["run_config_digest"]:
        return "Drizzle checkpoint run_config.cfg digest mismatch"
    if fingerprint != manifest["scientific_fingerprint"]:
        return "Drizzle checkpoint scientific fingerprint mismatch"
    if config.scientific != manifest["scientific_config"]:
        return "Drizzle checkpoint scientific configuration mismatch"

    # Values must be exactly representable by the existing Qt controls.  Do
    # not clamp and accidentally build a scientifically different request.
    sci = config.scientific
    scale = sci.get("drizzle_scale_effective")
    kernel = sci.get("drizzle_kernel_effective")
    pixfrac = sci.get("drizzle_pixfrac_effective")
    threshold = sci.get("drizzle_wht_threshold_effective")
    if scale not in (1.0, 2.0, 3.0, 4.0):
        return "Drizzle effective scale is not representable by this UI"
    if kernel not in {
        "square", "gaussian", "point", "tophat", "turbo", "lanczos2", "lanczos3"
    }:
        return "Drizzle effective kernel is not representable by this UI"
    if (
        not isinstance(pixfrac, (int, float))
        or isinstance(pixfrac, bool)
        or not (0.01 <= pixfrac <= 2.0)
    ):
        return "Drizzle effective pixfrac is not representable by this UI"
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not (0.0 <= threshold <= 1.0)
    ):
        return "Drizzle effective WHT threshold is not representable by this UI"
    return None


def resolve_run_directory(locator: Any, max_depth: int = MAX_SEARCH_DEPTH) -> Optional[str]:
    """Resolve a FIT/directory locator to its owning run directory.

    Deterministic and bounded: starting from the locator's directory (its
    parent when it is a file), walk upward at most ``max_depth`` levels and
    return the first directory with recognized state.  Returns ``None`` when
    nothing is recognized.  The locator itself is never treated as a checkpoint.
    """
    if not locator:
        return None
    path = os.path.abspath(os.fspath(locator))
    current = path if os.path.isdir(path) else os.path.dirname(path)
    depth = max(0, int(max_depth))
    for _ in range(depth + 1):
        if recognized_state(current):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def find_legacy_cfg(run_dir: Any) -> Optional[str]:
    """Return the deterministic legacy 5.x ``.cfg`` candidate, or ``None``.

    Only the run directory is scanned (non-recursive, bounded).  ``run_config.cfg``
    is excluded; the legacy ``_stack_*.cfg`` naming is preferred; ties break
    lexicographically.  This file is a *restoration* candidate only — never
    checkpoint evidence.
    """
    try:
        names = os.listdir(os.fspath(run_dir))
    except OSError:
        return None
    candidates = []
    for name in names:
        if name == RUN_CONFIG_FILENAME:
            continue
        if not name.lower().endswith(".cfg"):
            continue
        p = os.path.join(os.fspath(run_dir), name)
        if not os.path.isfile(p):
            continue
        candidates.append(p)
    if not candidates:
        return None
    stack_named = [p for p in candidates if os.path.basename(p).startswith("_stack_")]
    pool = stack_named or candidates
    return sorted(pool)[0]


def _bounded(text: Any) -> str:
    """Return ``text`` as a bounded string (data only, no secrets)."""
    s = str(text or "")
    if len(s) > _MAX_DETAIL_CHARS:
        return s[:_MAX_DETAIL_CHARS] + "..."
    return s


def discover_resume(locator: Any, max_depth: int = MAX_SEARCH_DEPTH) -> DiscoveryResult:
    """Locate the owning run directory and discover its configuration.

    Read-only: nothing is written and no setting is invented.  The decision is:

    * no run directory recognized -> :data:`STATUS_NO_RUN_DIR`;
    * recognized run dir, but no checkpoint manifest -> :data:`STATUS_NO_CHECKPOINT`
      (a FIT or a CFG alone is never checkpoint evidence);
    * checkpoint manifest present but config absent/unrecoverable ->
      :data:`STATUS_CONFIG_UNAVAILABLE` (e.g. a manifest-v1 hash-only checkpoint
      with no ``run_config.cfg`` and no legacy ``.cfg``);
    * corrupt / unsafe / ambiguous config -> the matching fail-closed status;
    * checkpoint present + restorable config -> :data:`STATUS_READY`.
    """
    run_dir = resolve_run_directory(locator, max_depth=max_depth)
    if run_dir is None:
        return DiscoveryResult(
            status=STATUS_NO_RUN_DIR,
            reason_key=STATUS_REASON_KEYS[STATUS_NO_RUN_DIR],
        )

    classic_checkpoint = has_manifest(run_dir)
    drizzle_checkpoint = os.path.isfile(drizzle_manifest_path(run_dir))
    if classic_checkpoint and drizzle_checkpoint:
        return DiscoveryResult(
            status=STATUS_AMBIGUOUS_CHECKPOINT,
            run_dir=run_dir,
            reason_key=STATUS_REASON_KEYS[STATUS_AMBIGUOUS_CHECKPOINT],
        )
    checkpoint_kind = (
        CHECKPOINT_KIND_DRIZZLE
        if drizzle_checkpoint
        else CHECKPOINT_KIND_CLASSIC if classic_checkpoint else None
    )
    config: Optional[run_contract.RunConfig] = None
    config_source: Optional[str] = None

    cfg_path = run_config_path(run_dir)
    if checkpoint_kind == CHECKPOINT_KIND_DRIZZLE and os.path.islink(cfg_path):
        return DiscoveryResult(
            status=STATUS_CORRUPT_CHECKPOINT,
            run_dir=run_dir,
            checkpoint_kind=checkpoint_kind,
            reason_key=STATUS_REASON_KEYS[STATUS_CORRUPT_CHECKPOINT],
            detail="Drizzle run_config.cfg must not be a symbolic link",
        )
    if os.path.isfile(cfg_path):
        try:
            report = run_contract.read_cfg(cfg_path)
        except run_contract.UnsafeConfigError as exc:
            return DiscoveryResult(
                status=STATUS_UNSAFE_CONFIG,
                run_dir=run_dir,
                reason_key=STATUS_REASON_KEYS[STATUS_UNSAFE_CONFIG],
                detail=_bounded(exc),
            )
        except run_contract.ConfigError as exc:
            return DiscoveryResult(
                status=STATUS_CORRUPT_CONFIG,
                run_dir=run_dir,
                reason_key=STATUS_REASON_KEYS[STATUS_CORRUPT_CONFIG],
                detail=_bounded(exc),
            )
        config = report.config
        config_source = "v2"
    elif checkpoint_kind != CHECKPOINT_KIND_DRIZZLE:
        legacy = find_legacy_cfg(run_dir)
        if legacy is not None:
            try:
                data = run_contract.parse_legacy_cfg(legacy)
                migrated = run_contract.migrate_legacy(data)
            except run_contract.UnsafeLegacyError as exc:
                return DiscoveryResult(
                    status=STATUS_UNSAFE_CONFIG,
                    run_dir=run_dir,
                    reason_key=STATUS_REASON_KEYS[STATUS_UNSAFE_CONFIG],
                    detail=_bounded(exc),
                )
            except run_contract.AmbiguousLegacyError as exc:
                return DiscoveryResult(
                    status=STATUS_AMBIGUOUS_LEGACY,
                    run_dir=run_dir,
                    reason_key=STATUS_REASON_KEYS[STATUS_AMBIGUOUS_LEGACY],
                    detail=_bounded(exc),
                )
            except run_contract.ConfigError as exc:
                return DiscoveryResult(
                    status=STATUS_CORRUPT_CONFIG,
                    run_dir=run_dir,
                    reason_key=STATUS_REASON_KEYS[STATUS_CORRUPT_CONFIG],
                    detail=_bounded(exc),
                )
            config = migrated.config
            config_source = "legacy"

    if checkpoint_kind is None:
        return DiscoveryResult(
            status=STATUS_NO_CHECKPOINT,
            run_dir=run_dir,
            config=config,
            config_source=config_source,
            reason_key=STATUS_REASON_KEYS[STATUS_NO_CHECKPOINT],
        )
    if config is None:
        return DiscoveryResult(
            status=STATUS_CONFIG_UNAVAILABLE,
            run_dir=run_dir,
            reason_key=STATUS_REASON_KEYS[STATUS_CONFIG_UNAVAILABLE],
            checkpoint_kind=checkpoint_kind,
        )

    if checkpoint_kind == CHECKPOINT_KIND_DRIZZLE:
        detail = _validate_drizzle_manifest_and_config(
            drizzle_manifest_path(run_dir), config
        )
        if detail is not None:
            return DiscoveryResult(
                status=STATUS_CORRUPT_CHECKPOINT,
                run_dir=run_dir,
                config=config,
                config_source=config_source,
                checkpoint_kind=checkpoint_kind,
                reason_key=STATUS_REASON_KEYS[STATUS_CORRUPT_CHECKPOINT],
                detail=detail,
            )

    return DiscoveryResult(
        status=STATUS_READY,
        run_dir=run_dir,
        config=config,
        config_source=config_source,
        checkpoint_kind=checkpoint_kind,
    )


def restore_to_settings(
    config: run_contract.RunConfig,
    settings: Any,
    *,
    checkpoint_kind: str = CHECKPOINT_KIND_CLASSIC,
) -> run_contract.ApplyReport:
    """Restore ``config``'s allowlisted, restore-eligible fields onto ``settings``.

    Uses :func:`run_contract.apply_to_settings` (the single field-spec source),
    then normalises the one canonical/Qt representation mismatch (the winsor
    limits list -> the ``stack_winsor_limits`` comma string).  Returns the
    :class:`run_contract.ApplyReport`.
    """
    report = run_contract.apply_to_settings(config, settings)
    if checkpoint_kind == CHECKPOINT_KIND_DRIZZLE:
        scientific = config.scientific
        settings.use_drizzle = True
        settings.drizzle_mode = "Final"
        settings.drizzle_scale = int(scientific["drizzle_scale_effective"])
        settings.drizzle_kernel = scientific["drizzle_kernel_effective"]
        settings.drizzle_pixfrac = float(scientific["drizzle_pixfrac_effective"])
        settings.drizzle_wht_threshold = float(
            scientific["drizzle_wht_threshold_effective"]
        )
        settings.mosaic_mode_active = False
        settings.stack_final_combine = "mean"
        settings.reproject_between_batches = False
        settings.reproject_coadd_final = False
    winsor = getattr(settings, "stack_winsor_limits", None)
    if isinstance(winsor, (list, tuple)) and len(winsor) == 2:
        try:
            settings.stack_winsor_limits = ",".join(str(float(v)) for v in winsor)
        except (TypeError, ValueError):
            pass
    return report
