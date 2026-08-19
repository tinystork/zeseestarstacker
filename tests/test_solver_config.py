"""Tests for the native ``seestar.core.solver_config`` module.

The module is loaded directly by file path so the test does not pull in the
heavy ``seestar`` package tree (which requires optional deps like OpenCV that
are absent from this test environment).
"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "seestar_solver_config", ROOT / "seestar" / "core" / "solver_config.py"
)
solver_config = importlib.util.module_from_spec(spec)
sys.modules["seestar_solver_config"] = solver_config
spec.loader.exec_module(solver_config)


def test_default_config_has_key_keys():
    assert "solver_method" in solver_config.DEFAULT_CONFIG
    assert "astrometry_local_path" in solver_config.DEFAULT_CONFIG
    assert "astrometry_api_key" in solver_config.DEFAULT_CONFIG
    assert "astap_default_search_radius" in solver_config.DEFAULT_CONFIG


def test_config_round_trip_preserves_keys(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(solver_config, "get_config_path", lambda: str(cfg_path))
    monkeypatch.setattr(solver_config, "_legacy_config_candidates", lambda: [])

    data = solver_config.DEFAULT_CONFIG.copy()
    data["solver_method"] = "ansvr"
    data["astrometry_local_path"] = "/tmp/ansvr"
    data["astrometry_api_key"] = "XYZ123"

    assert solver_config.save_config(data)
    loaded = solver_config.load_config()

    assert loaded["solver_method"] == "ansvr"
    assert loaded["astrometry_local_path"] == "/tmp/ansvr"
    assert loaded["astrometry_api_key"] == "XYZ123"


def test_soft_migration_from_legacy(tmp_path, monkeypatch):
    new_path = tmp_path / "seestar_config.json"
    legacy = tmp_path / "zemosaic_config.json"
    legacy.write_text(
        json.dumps(
            {"solver_method": "ansvr", "astap_default_search_radius": 5.0}
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(solver_config, "get_config_path", lambda: str(new_path))
    monkeypatch.setattr(solver_config, "_legacy_config_candidates", lambda: [legacy])

    loaded = solver_config.load_config()
    assert loaded["solver_method"] == "ansvr"
    assert loaded["astap_default_search_radius"] == 5.0
    # Legacy file is consulted read-only and left in place.
    assert legacy.exists()

    # Saving writes to the new user location, not the legacy file.
    assert solver_config.save_config(loaded)
    assert new_path.exists()
    assert json.loads(legacy.read_text(encoding="utf-8"))["solver_method"] == "ansvr"


def test_getters_return_defaults(monkeypatch):
    monkeypatch.setattr(
        solver_config,
        "get_config_path",
        lambda: str(Path("/nonexistent") / "cfg.json"),
    )
    monkeypatch.setattr(solver_config, "_legacy_config_candidates", lambda: [])

    assert solver_config.get_astap_default_search_radius() == 3.0
    assert solver_config.get_astap_default_downsample() == 2
    assert solver_config.get_astap_default_sensitivity() == 100
    assert solver_config.get_astap_executable_path() == ""
    assert solver_config.get_astap_data_directory_path() == ""
