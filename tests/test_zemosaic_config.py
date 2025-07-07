import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "zemosaic_config", ROOT / "zemosaic" / "zemosaic_config.py"
)
zemosaic_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zemosaic_config)


def test_default_config_has_new_keys():
    assert "solver_method" in zemosaic_config.DEFAULT_CONFIG
    assert "astrometry_local_path" in zemosaic_config.DEFAULT_CONFIG
    assert "astrometry_api_key" in zemosaic_config.DEFAULT_CONFIG


def test_config_round_trip_preserves_new_keys(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(zemosaic_config, "get_config_path", lambda: str(cfg_path))

    data = zemosaic_config.DEFAULT_CONFIG.copy()
    data["solver_method"] = "ansvr"
    data["astrometry_local_path"] = "/tmp/ansvr"
    data["astrometry_api_key"] = "XYZ123"

    assert zemosaic_config.save_config(data)
    loaded = zemosaic_config.load_config()

    assert loaded["solver_method"] == "ansvr"
    assert loaded["astrometry_local_path"] == "/tmp/ansvr"
    assert loaded["astrometry_api_key"] == "XYZ123"


def test_resolve_astap_app(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin", raising=False)
    app_path = "/Applications/ASTAP.app"
    internal = "/Applications/ASTAP.app/Contents/MacOS/astap"
    monkeypatch.setattr(zemosaic_config.os.path, "isdir", lambda p: p == app_path)
    monkeypatch.setattr(zemosaic_config.os.path, "isfile", lambda p: p == internal)

    resolved = zemosaic_config.resolve_astap_executable_path(app_path)
    assert resolved == internal
