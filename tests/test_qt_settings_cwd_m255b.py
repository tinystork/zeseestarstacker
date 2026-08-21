"""M25.5-B tests: platform-aware settings path + non-destructive legacy migration.

The Qt settings default moved from a CWD ``seestar_settings.json`` (the
historical Tk-era convention) to a per-user, platform-aware location, so the
installed GUI finds the *same* settings regardless of the launch directory.
This module exercises ``seestar.gui_qt.settings_persistence`` — pure stdlib, no
Qt — and never touches the real user config: every test monkeypatches
``HOME`` / ``XDG_CONFIG_HOME`` / ``APPDATA`` (and the platform branch) to an
isolated tmp dir.

Covered behaviours:

* new platform file only -> loaded from the platform dir (new wins);
* legacy CWD file only -> migrated to the platform dir, legacy **preserved**,
  content equal (recognised keys + unknown keys);
* new + legacy -> new wins, legacy untouched;
* two different CWDs -> same resolved path and same content;
* unknown/extra JSON keys -> preserved at the file layer (the model layer
  ``QtSettingsState.from_dict`` is the thing that filters; see
  ``test_qt_settings_state.py``);
* invalid legacy JSON -> ``load`` -> ``{}``, never copied during migration;
* unwritable user-config dir -> clean failure (no raise, defaults, save False);
* save creates the parent directory when missing.
"""

from __future__ import annotations

import json
import os

import pytest

import seestar.gui_qt.settings_persistence as persistence


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    return path


def _platform_file(tmp_path):
    """The platform-aware settings file under the monkeypatched XDG dir."""
    return os.path.join(
        str(tmp_path / "xdg"),
        "ZeSeestarStacker",
        persistence.DEFAULT_SETTINGS_FILENAME,
    )


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    """Force the Linux platform branch with isolated HOME/XDG dirs + clean CWD."""
    monkeypatch.setattr(persistence.platform, "system", lambda: "Linux")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("APPDATA", raising=False)
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    return tmp_path


# --------------------------------------------------------------------------
# Platform-aware path
# --------------------------------------------------------------------------
def test_platform_settings_dir_linux_xdg(isolated_env, tmp_path):
    assert persistence.platform_settings_dir() == os.path.join(
        str(tmp_path / "xdg"), "ZeSeestarStacker"
    )


def test_platform_settings_dir_linux_xdg_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence.platform, "system", lambda: "Linux")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    assert persistence.platform_settings_dir() == os.path.join(
        str(home), ".config", "ZeSeestarStacker"
    )


def test_platform_settings_dir_macos(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence.platform, "system", lambda: "Darwin")
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    assert persistence.platform_settings_dir() == os.path.join(
        str(home), "Library", "Application Support", "ZeSeestarStacker"
    )


def test_platform_settings_dir_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence.os, "name", "nt", raising=False)
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("APPDATA", str(appdata))
    assert persistence.platform_settings_dir() == os.path.join(
        str(appdata), "ZeSeestarStacker"
    )


def test_platform_settings_dir_windows_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence.os, "name", "nt", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    assert persistence.platform_settings_dir() == os.path.join(
        str(home), "AppData", "Roaming", "ZeSeestarStacker"
    )


# --------------------------------------------------------------------------
# Resolution priority
# --------------------------------------------------------------------------
def test_new_settings_only_loaded_from_platform_dir(isolated_env, tmp_path):
    new_path = _platform_file(tmp_path)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    data = {"batch_size": 7, "kappa": 2.5}
    _write_json(new_path, data)

    resolved = persistence.resolve_settings_path()
    assert resolved == new_path
    assert persistence.load_settings_json(resolved) == data


def test_legacy_only_migrates_and_preserves(isolated_env, tmp_path):
    legacy = os.path.join(str(tmp_path / "cwd"), persistence.DEFAULT_SETTINGS_FILENAME)
    data = {"batch_size": 3, "kappa": 3.0, "totally_unknown_key": "keep-me"}
    _write_json(legacy, data)

    new_path = _platform_file(tmp_path)
    resolved = persistence.resolve_settings_path()

    assert resolved == new_path
    # Legacy file is preserved (never deleted) and untouched.
    assert os.path.exists(legacy)
    assert persistence.load_settings_json(legacy) == data
    # Migrated content equals the legacy content (recognised + unknown keys).
    assert persistence.load_settings_json(new_path) == data


def test_new_plus_legacy_new_wins(isolated_env, tmp_path):
    legacy = os.path.join(str(tmp_path / "cwd"), persistence.DEFAULT_SETTINGS_FILENAME)
    _write_json(legacy, {"batch_size": 1, "legacy_only": True})

    new_path = _platform_file(tmp_path)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    _write_json(new_path, {"batch_size": 99})

    resolved = persistence.resolve_settings_path()
    assert resolved == new_path
    assert persistence.load_settings_json(new_path) == {"batch_size": 99}
    # Legacy untouched.
    assert persistence.load_settings_json(legacy) == {"batch_size": 1, "legacy_only": True}


def test_different_cwd_same_settings_path(isolated_env, monkeypatch, tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    monkeypatch.chdir(a)
    path_a = persistence.default_settings_path()
    resolved_a = persistence.resolve_settings_path()

    monkeypatch.chdir(b)
    path_b = persistence.default_settings_path()
    resolved_b = persistence.resolve_settings_path()

    assert path_a == path_b
    assert resolved_a == resolved_b
    # No file anywhere -> both resolve to defaults with the same content.
    assert persistence.load_settings_json(path_a) == {}
    assert persistence.load_settings_json(path_b) == {}


def test_no_settings_anywhere_resolves_platform_path(isolated_env, tmp_path):
    new_path = _platform_file(tmp_path)
    assert not os.path.exists(new_path)
    resolved = persistence.resolve_settings_path()
    assert resolved == new_path
    # The parent dir is created eagerly so the first save works.
    assert os.path.isdir(os.path.dirname(new_path))
    assert persistence.load_settings_json(resolved) == {}


# --------------------------------------------------------------------------
# Unknown / invalid / unwritable edge cases
# --------------------------------------------------------------------------
def test_unknown_keys_preserved_at_file_layer(tmp_path):
    p = tmp_path / "s.json"
    data = {"batch_size": 4, "unknown_key": "x", "nested_unknown": {"a": 1}}
    assert persistence.save_settings_json(str(p), data) is True
    # The file layer never filters: unknown keys round-trip verbatim.
    assert persistence.load_settings_json(str(p)) == data


def test_invalid_legacy_json_is_not_copied(isolated_env, tmp_path):
    legacy = os.path.join(str(tmp_path / "cwd"), persistence.DEFAULT_SETTINGS_FILENAME)
    with open(legacy, "w", encoding="utf-8") as fh:
        fh.write("{ not valid json")

    new_path = _platform_file(tmp_path)
    resolved = persistence.resolve_settings_path()

    assert resolved == new_path
    # Invalid legacy is treated as missing for migration: never copied.
    assert not os.path.exists(new_path)
    assert os.path.exists(legacy)  # legacy preserved
    # Historical load behaviour: invalid JSON -> {}.
    assert persistence.load_settings_json(legacy) == {}
    assert persistence.load_settings_json(resolved) == {}


def test_unwritable_user_config_dir_fails_cleanly(isolated_env, monkeypatch, tmp_path):
    # Make the user-config location un-creatable: XDG points under a FILE.
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(blocker / "xdg"))

    new_path = os.path.join(
        str(blocker / "xdg"),
        "ZeSeestarStacker",
        persistence.DEFAULT_SETTINGS_FILENAME,
    )
    # Clean failure: no raise, returns None (persistence disabled -> defaults).
    assert persistence.resolve_settings_path() is None
    assert persistence.load_settings_json(new_path) == {}
    assert persistence.save_settings_json(new_path, {"batch_size": 1}) is False


def test_save_creates_parent_dir(isolated_env, tmp_path):
    new_path = _platform_file(tmp_path)
    assert not os.path.exists(os.path.dirname(new_path))
    assert persistence.save_settings_json(new_path, {"batch_size": 2}) is True
    assert os.path.exists(new_path)
    assert persistence.load_settings_json(new_path) == {"batch_size": 2}
