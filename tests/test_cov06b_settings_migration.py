"""COV-06B versioned retirement of persisted inverse-WHT Feathering."""

import json

from seestar.gui.settings import SettingsManager
from seestar.gui_qt.settings_state import QtSettingsState
from seestar.settings_migration import (
    CURRENT_SETTINGS_SCHEMA_VERSION,
    SETTINGS_SCHEMA_VERSION_KEY,
    migrate_settings_data,
)


def test_fresh_installation_has_safe_cov_defaults():
    state = QtSettingsState()
    assert state.apply_feathering is False
    assert state.apply_batch_feathering is True
    assert state.apply_coverage_render is False


def test_old_apply_feathering_true_migrates_to_safe_state():
    migrated, changed = migrate_settings_data(
        {"apply_feathering": True, "batch_size": 12}
    )
    assert changed is True
    assert migrated["apply_feathering"] is False
    assert migrated["batch_size"] == 12
    assert migrated[SETTINGS_SCHEMA_VERSION_KEY] == CURRENT_SETTINGS_SCHEMA_VERSION


def test_modern_settings_do_not_rerun_or_flip_values():
    modern = {
        SETTINGS_SCHEMA_VERSION_KEY: CURRENT_SETTINGS_SCHEMA_VERSION,
        "apply_feathering": False,
        "apply_coverage_render": True,
        "batch_size": 9,
    }
    migrated, changed = migrate_settings_data(modern)
    assert changed is False
    assert migrated == modern
    assert migrated is not modern


def test_future_schema_is_never_downgraded():
    future = {
        SETTINGS_SCHEMA_VERSION_KEY: CURRENT_SETTINGS_SCHEMA_VERSION + 1,
        "apply_feathering": True,
    }
    migrated, changed = migrate_settings_data(future)
    assert changed is False
    assert migrated == future


def test_coverage_render_state_round_trip():
    state = QtSettingsState(apply_coverage_render=True)
    restored = QtSettingsState.from_dict(state.to_dict())
    assert restored.apply_coverage_render is True


def test_tk_loader_migrates_and_records_schema_once(tmp_path):
    path = tmp_path / "seestar_settings.json"
    path.write_text(
        json.dumps({"apply_feathering": True, "batch_size": 3}),
        encoding="utf-8",
    )

    first = SettingsManager(settings_file=str(path))
    assert first.load_settings() is True
    assert first.apply_feathering is False
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["apply_feathering"] is False
    assert saved[SETTINGS_SCHEMA_VERSION_KEY] == CURRENT_SETTINGS_SCHEMA_VERSION

    second = SettingsManager(settings_file=str(path))
    assert second.load_settings() is True
    assert second.apply_feathering is False
    assert json.loads(path.read_text(encoding="utf-8")) == saved
