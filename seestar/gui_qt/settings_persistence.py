"""Pure-stdlib JSON persistence helper for the Qt settings state (M8).

This module is the file-backed persistence layer for
:class:`~seestar.gui_qt.settings_state.QtSettingsState`.  It deliberately knows
nothing about Qt, Tk, the scientific engine, or the settings *model* itself: it
only loads and saves a plain JSON ``dict`` and delegates type coercion /
unknown-key filtering to ``QtSettingsState.from_dict``.

Design constraints honoured here:

* **No Qt, no Tk, no numpy, no engine imports.**  Only ``json`` / ``os`` /
  ``platform``, so it can be unit-tested in complete isolation.
* **Never crash the GUI.**  A missing file is the first-run default and a
  corrupt file must not prevent the window opening, so both load paths return an
  empty dict; a save failure returns ``False`` instead of raising.
* **Deterministic output.**  ``sort_keys`` + ``indent=2`` + ``ensure_ascii=False``
  (UTF-8) so the file is human-readable and byte-stable for tests.
* **Platform-aware user-config default (M25.5-B).**  The default settings live
  in a per-user, platform-specific directory (Windows ``%APPDATA%``, macOS
  ``~/Library/Application Support``, Linux ``$XDG_CONFIG_HOME`` / ``~/.config``)
  so the installed GUI finds the *same* settings no matter which directory it is
  launched from.  A legacy CWD ``seestar_settings.json`` (the historical Tk-era
  convention) is migrated non-destructively into that location on first run.
  Callers (and tests) may still inject any path instead.
"""

from __future__ import annotations

import json
import os
import platform
from typing import Any, Dict, Optional

from seestar.settings_migration import (
    CURRENT_SETTINGS_SCHEMA_VERSION,
    SETTINGS_SCHEMA_VERSION_KEY,
    migrate_settings_data,
)

DEFAULT_SETTINGS_FILENAME = "seestar_settings.json"

# Per-user platform-aware directory name.  Mirrors the engine config convention
# (``seestar/core/solver_config.py``) *without importing it*, so this module
# stays free of engine imports.
USER_CONFIG_DIR_NAME = "ZeSeestarStacker"


def platform_settings_dir() -> str:
    """Return the platform-aware per-user settings directory.

    Windows: ``%APPDATA%\\ZeSeestarStacker`` (fallback ``~/AppData/Roaming``).
    macOS:   ``~/Library/Application Support/ZeSeestarStacker``.
    Linux:   ``$XDG_CONFIG_HOME/ZeSeestarStacker`` (default ``~/.config``).

    Pure ``os`` / ``platform`` — the same semantics as the engine's
    ``_user_config_dir``, replicated here so this module never imports the
    engine.  Never raises.
    """
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Roaming"
        )
        return os.path.join(base, USER_CONFIG_DIR_NAME)
    if platform.system() == "Darwin":
        base = os.path.join(
            os.path.expanduser("~"), "Library", "Application Support"
        )
        return os.path.join(base, USER_CONFIG_DIR_NAME)
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config"
    )
    return os.path.join(base, USER_CONFIG_DIR_NAME)


def default_settings_path() -> str:
    """Return the default settings JSON path (platform-aware user config)."""
    return os.path.join(platform_settings_dir(), DEFAULT_SETTINGS_FILENAME)


def legacy_settings_path() -> str:
    """Return the legacy CWD-based settings path (historical Tk convention)."""
    try:
        cwd = os.getcwd()
    except OSError:
        return ""
    return os.path.join(cwd, DEFAULT_SETTINGS_FILENAME)


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of ``path`` (``os.makedirs(exist_ok=True)``)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_settings_json(path: Optional[str]) -> Dict[str, Any]:
    """Load a settings dict from ``path``; return ``{}`` on missing/corrupt file.

    A missing file, an unreadable file, non-JSON content, or a JSON document
    that is not a mapping all degrade to an empty dict so the caller always gets
    the code defaults and the GUI always opens.  ``None``/empty ``path`` is
    treated as "persistence disabled".
    """
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def save_settings_json(path: Optional[str], data: Dict[str, Any]) -> bool:
    """Write ``data`` to ``path`` as UTF-8 JSON; return success (``bool``).

    Never raises: a write error returns ``False`` so a save failure can never
    take the GUI down.  The parent directory is created on demand
    (``os.makedirs(exist_ok=True)``) so saving into a freshly-resolved platform
    path works on the very first run.  ``sort_keys`` keeps the file
    deterministic for tests; ``ensure_ascii=False`` keeps non-ASCII paths
    human-readable.
    """
    if not path:
        return False
    try:
        _ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        return True
    except OSError:
        return False


def migrate_legacy_settings(new_path: str, legacy_path: str) -> bool:
    """Copy ``legacy_path`` to ``new_path`` non-destructively; return success.

    The legacy file is **never** deleted.  The whole JSON document is copied
    verbatim — recognised keys **and** unknown keys — because the *file layer*
    does not filter (filtering is a ``QtSettingsState.from_dict`` model concern
    that happens only on the later load).

    A missing / corrupt / non-mapping / empty legacy file is treated as "nothing
    to migrate": it returns ``True`` and writes nothing, so the caller falls
    through to defaults and an invalid legacy file is never copied.

    Returns ``False`` (and never raises) only when the target location cannot be
    written, so the caller can fall back to persistence-disabled defaults.
    """
    data = load_settings_json(legacy_path)
    if not data:
        return True
    try:
        _ensure_parent_dir(new_path)
        with open(new_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        return True
    except OSError:
        return False


def resolve_settings_path() -> Optional[str]:
    """Resolve the default settings path, migrating a legacy CWD file.

    Exact priority (documented and tested):

    1. The platform-aware settings file already exists -> return it (new wins;
       the legacy file is never touched).
    2. Else a legacy CWD ``seestar_settings.json`` exists -> migrate it into the
       platform location (creating directories), **preserve** the legacy file,
       and return the new path.
    3. Else -> return the new path (no file yet; the code defaults apply until
       the first save).  The user-config directory is created eagerly so the
       first save works without an extra mkdir round-trip.

    Failure mode: if the user-config location cannot be created or written
    (e.g. it is not writable), ``None`` is returned — never raised — so the
    caller disables persistence and the GUI still opens with the code defaults.
    """
    new_path = default_settings_path()
    legacy = legacy_settings_path()
    if os.path.exists(new_path):
        return new_path
    if os.path.exists(legacy):
        if not migrate_legacy_settings(new_path, legacy):
            return None
        return new_path
    try:
        _ensure_parent_dir(new_path)
    except OSError:
        return None
    return new_path
