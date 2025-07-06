import importlib
import sys
import types

# minimal package stubs for queue_manager dependencies
if "seestar.gui" not in sys.modules:
    seestar_pkg = types.ModuleType("seestar")
    base = str(__file__).split("tests")[0] + "seestar"
    seestar_pkg.__path__ = [base]

    gui_pkg = types.ModuleType("seestar.gui")
    gui_pkg.__path__ = []
    settings_mod = types.ModuleType("seestar.gui.settings")
    settings_mod.SettingsManager = object
    gui_pkg.settings = settings_mod
    seestar_pkg.gui = gui_pkg
    sys.modules["seestar"] = seestar_pkg
    sys.modules["seestar.gui"] = gui_pkg
    sys.modules["seestar.gui.settings"] = settings_mod

    zmod = types.ModuleType("zemosaic")
    zmod.zemosaic_config = types.SimpleNamespace(
        get_astap_default_search_radius=lambda: 0
    )
    sys.modules.setdefault("zemosaic", zmod)

qm = importlib.import_module("seestar.queuep.queue_manager")


def test_quality_pool_size(monkeypatch):
    monkeypatch.setattr(qm.os, "cpu_count", lambda: 8)
    assert qm._suggest_pool_size(0.75) == 6
