"""Versioned product-settings migrations shared by the Qt and Tk shells.

Migrations operate on a copy of the decoded JSON mapping and are deliberately
pure stdlib.  A migration is applied at most once, then the schema marker is
persisted by the caller on the normal settings path.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


SETTINGS_SCHEMA_VERSION_KEY = "settings_schema_version"
CURRENT_SETTINGS_SCHEMA_VERSION = 1


def _schema_version(data: Mapping[str, Any]) -> int:
    """Return a safe non-negative schema version; malformed values are legacy."""
    raw = data.get(SETTINGS_SCHEMA_VERSION_KEY, 0)
    if isinstance(raw, bool):
        return 0
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return 0
    return value if value >= 0 else 0


def migrate_settings_data(data: Any) -> Tuple[Dict[str, Any], bool]:
    """Return ``(migrated_copy, changed)`` for a decoded settings mapping.

    Schema v1 retires persisted pre-COV inverse-WHT feathering.  Before this
    marker existed, ``apply_feathering=true`` represented the deprecated final
    ``blur(WHT)/WHT`` cosmetic gain.  It is forced OFF exactly once.  Settings
    from a future schema are preserved and never downgraded.
    """
    if not isinstance(data, dict):
        return {}, False

    migrated = dict(data)
    version = _schema_version(migrated)
    if version >= CURRENT_SETTINGS_SCHEMA_VERSION:
        return migrated, False

    migrated["apply_feathering"] = False
    migrated[SETTINGS_SCHEMA_VERSION_KEY] = CURRENT_SETTINGS_SCHEMA_VERSION
    return migrated, True
