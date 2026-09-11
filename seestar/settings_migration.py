"""Versioned product-settings migrations shared by the Qt and Tk shells.

Migrations operate on a copy of the decoded JSON mapping and are deliberately
pure stdlib.  A migration is applied at most once, then the schema marker is
persisted by the caller on the normal settings path.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Tuple


SETTINGS_SCHEMA_VERSION_KEY = "settings_schema_version"
CURRENT_SETTINGS_SCHEMA_VERSION = 1


def _migrate_pixfrac_contract(data: Dict[str, Any]) -> bool:
    """Migrate legacy persisted pixfrac values while raw values are visible.

    This migration is deliberately independent of the settings schema marker:
    schema-v1 files may still contain the historical UI range up to 2.0.  The
    active value is canonicalised, while the original numeric request and a
    stable reason remain available to both GUI shells and run provenance.
    """
    changed = False

    def _migrate(mapping: Dict[str, Any]) -> bool:
        if "pixfrac" not in mapping:
            return False
        raw_value = mapping.get("pixfrac")
        if isinstance(raw_value, bool):
            return False
        try:
            raw = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(raw):
            return False
        if raw > 1.0:
            mapping["pixfrac"] = 1.0
            mapping["pixfrac_requested_raw"] = raw
            mapping["pixfrac_reason"] = "pixfrac_gt_one_coerced_to_one"
            return True
        if 0.0 < raw < 0.01:
            mapping["pixfrac"] = 0.01
            mapping["pixfrac_requested_raw"] = raw
            mapping["pixfrac_reason"] = "pixfrac_below_minimum_clamped"
            return True
        return False

    # Standard Drizzle uses top-level settings names.
    if "drizzle_pixfrac" in data:
        standard = {"pixfrac": data.get("drizzle_pixfrac")}
        if _migrate(standard):
            data["drizzle_pixfrac"] = standard["pixfrac"]
            data["drizzle_pixfrac_requested_raw"] = standard[
                "pixfrac_requested_raw"
            ]
            data["drizzle_pixfrac_reason"] = standard["pixfrac_reason"]
            changed = True

    # Mosaic Drizzle keeps the same contract in a nested mapping.
    mosaic = data.get("mosaic_settings")
    if isinstance(mosaic, dict):
        migrated_mosaic = dict(mosaic)
        if _migrate(migrated_mosaic):
            data["mosaic_settings"] = migrated_mosaic
            changed = True

    return changed


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
    pixfrac_changed = _migrate_pixfrac_contract(migrated)
    version = _schema_version(migrated)
    if version >= CURRENT_SETTINGS_SCHEMA_VERSION:
        return migrated, pixfrac_changed

    migrated["apply_feathering"] = False
    migrated[SETTINGS_SCHEMA_VERSION_KEY] = CURRENT_SETTINGS_SCHEMA_VERSION
    return migrated, True
