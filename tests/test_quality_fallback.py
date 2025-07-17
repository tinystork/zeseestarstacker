import importlib
import sys
import types
import numpy as np

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


def dummy_worker(data):
    return {"snr": 1.0, "stars": 1.0}, None, 1


def test_quality_fallback_large(monkeypatch):
    monkeypatch.setattr(qm, "_quality_metrics_worker", dummy_worker)
    s = qm.SeestarQueuedStacker()
    # create >32 MB array
    big = np.zeros((4096, 4096), dtype=np.float32)
    res = s._calculate_quality_metrics(big)
    assert res["snr"] == 1.0
    s.__class__.stop_processing(s)
