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
  up from the selected path) by recognized state: ``run_config.cfg`` and/or
  ``memmap_accumulators/resume_manifest.json``.
* The preferred configuration source is the schema-v2 ``run_config.cfg``
  (read via :func:`run_contract.read_cfg`).  A legacy 5.x ``.cfg`` is a
  fallback *restoration object* only, never checkpoint evidence.
* Resume may proceed (status :data:`STATUS_READY`) only when BOTH a recognized
  checkpoint manifest exists AND a restorable config (v2 or migrated legacy)
  is available.
* Every other case fails closed with a stable status + bounded reason; no file
  is written, no setting is invented, and no hidden resume intent is produced.

The scientific backend remains the independent validation authority; this
module is the convenience/first-protection layer only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from seestar import run_contract

__all__ = [
    "MEMORY_DIR",
    "RESUME_MANIFEST_FILENAME",
    "RUN_CONFIG_FILENAME",
    "MAX_SEARCH_DEPTH",
    "STATUS_READY",
    "STATUS_NO_RUN_DIR",
    "STATUS_NO_CHECKPOINT",
    "STATUS_CONFIG_UNAVAILABLE",
    "STATUS_CORRUPT_CONFIG",
    "STATUS_UNSAFE_CONFIG",
    "STATUS_AMBIGUOUS_LEGACY",
    "STATUS_REASON_KEYS",
    "DiscoveryResult",
    "manifest_path",
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
RUN_CONFIG_FILENAME = "run_config.cfg"

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

# Status -> localization key for the GUI warning body.  Every non-ready status
# maps to one bounded, actionable message (the GUI owns the FR/EN text).
STATUS_REASON_KEYS = {
    STATUS_NO_RUN_DIR: "resume_refuse_no_run",
    STATUS_NO_CHECKPOINT: "resume_refuse_no_checkpoint",
    STATUS_CONFIG_UNAVAILABLE: "resume_refuse_config_unavailable",
    STATUS_CORRUPT_CONFIG: "resume_refuse_corrupt_config",
    STATUS_UNSAFE_CONFIG: "resume_refuse_unsafe_config",
    STATUS_AMBIGUOUS_LEGACY: "resume_refuse_ambiguous_legacy",
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
    reason_key: Optional[str] = None
    detail: Optional[str] = None


def manifest_path(run_dir: Any) -> str:
    """Return the checkpoint-manifest path under ``run_dir``."""
    return os.path.join(os.fspath(run_dir), MEMORY_DIR, RESUME_MANIFEST_FILENAME)


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

    Recognized state is ``run_config.cfg`` and/or
    ``memmap_accumulators/resume_manifest.json`` — the two facts a locator may
    use to claim a directory is (or is inside) a run directory.
    """
    return has_run_config(run_dir) or has_manifest(run_dir)


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

    checkpoint = has_manifest(run_dir)
    config: Optional[run_contract.RunConfig] = None
    config_source: Optional[str] = None

    cfg_path = run_config_path(run_dir)
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
    else:
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

    if not checkpoint:
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
        )

    return DiscoveryResult(
        status=STATUS_READY, run_dir=run_dir, config=config, config_source=config_source
    )


def restore_to_settings(config: run_contract.RunConfig, settings: Any) -> run_contract.ApplyReport:
    """Restore ``config``'s allowlisted, restore-eligible fields onto ``settings``.

    Uses :func:`run_contract.apply_to_settings` (the single field-spec source),
    then normalises the one canonical/Qt representation mismatch (the winsor
    limits list -> the ``stack_winsor_limits`` comma string).  Returns the
    :class:`run_contract.ApplyReport`.
    """
    report = run_contract.apply_to_settings(config, settings)
    winsor = getattr(settings, "stack_winsor_limits", None)
    if isinstance(winsor, (list, tuple)) and len(winsor) == 2:
        try:
            settings.stack_winsor_limits = ",".join(str(float(v)) for v in winsor)
        except (TypeError, ValueError):
            pass
    return report
