"""M8 seam tests: ``seestar.gui_qt.settings_persistence`` JSON helper.

The helper is pure stdlib (``json`` / ``os`` only) and knows nothing about Qt,
Tk or the settings model, so these tests run with no ``QApplication`` and no
engine:

* save/load round-trip through a UTF-8 JSON file,
* missing file -> empty dict (code defaults), no crash,
* corrupt JSON / non-mapping JSON -> empty dict, no crash,
* deterministic output (sort_keys) so the file is stable for tests,
* the default path matches the Tk ``seestar_settings.json`` CWD convention.

``QtSettingsState.from_dict`` coercion / unknown-key filtering is covered in
``test_qt_settings_state.py``; only the file layer is exercised here.
"""

import json
import os
from pathlib import Path

import seestar.gui_qt.settings_persistence as persistence


def _write(path, text):
    path.write_text(text, encoding="utf-8")


def test_default_settings_path_matches_tk_convention(monkeypatch, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    assert persistence.default_settings_path() == os.path.join(
        str(workdir), persistence.DEFAULT_SETTINGS_FILENAME
    )
    assert persistence.DEFAULT_SETTINGS_FILENAME == "seestar_settings.json"


def test_load_missing_file_returns_empty_dict(tmp_path):
    missing = tmp_path / "nope.json"
    assert persistence.load_settings_json(str(missing)) == {}


def test_load_none_or_empty_path_returns_empty_dict():
    assert persistence.load_settings_json(None) == {}
    assert persistence.load_settings_json("") == {}


def test_load_corrupt_json_returns_empty_dict(tmp_path):
    p = tmp_path / "bad.json"
    _write(p, "{ not valid json !!!")
    assert persistence.load_settings_json(str(p)) == {}


def test_load_non_mapping_json_returns_empty_dict(tmp_path):
    p = tmp_path / "list.json"
    _write(p, "[1, 2, 3]")
    assert persistence.load_settings_json(str(p)) == {}


def test_save_and_load_round_trip(tmp_path):
    p = tmp_path / "settings.json"
    data = {
        "input_folder": "/inputs",
        "batch_size": 4,
        "mosaic_settings": {"kernel": "gaussian"},
        "window_geometry": "aGVsbG8=",
    }
    assert persistence.save_settings_json(str(p), data) is True

    loaded = persistence.load_settings_json(str(p))
    assert loaded == data


def test_save_is_deterministic(tmp_path):
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    data = {"b": 2, "a": 1, "nested": {"z": 0, "y": 1}}
    persistence.save_settings_json(str(p1), data)
    persistence.save_settings_json(str(p2), data)
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")
    # sort_keys guarantees "a" appears before "b".
    assert p1.read_text(encoding="utf-8").index('"a"') < p1.read_text(
        encoding="utf-8"
    ).index('"b"')


def test_save_utf8_and_trailing_newline(tmp_path):
    p = tmp_path / "settings.json"
    persistence.save_settings_json(str(p), {"input_folder": "/séparé/é"})
    raw = p.read_bytes()
    assert raw.endswith(b"\n")
    # ensure_ascii=False keeps non-ASCII readable.
    assert "séparé".encode("utf-8") in raw
    assert json.loads(raw.decode("utf-8"))["input_folder"] == "/séparé/é"


def test_save_none_path_returns_false():
    assert persistence.save_settings_json(None, {"a": 1}) is False
    assert persistence.save_settings_json("", {"a": 1}) is False


def test_save_to_unwritable_path_returns_false(tmp_path):
    # A directory path cannot be opened as a file for writing.
    assert persistence.save_settings_json(str(tmp_path), {"a": 1}) is False
