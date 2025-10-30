import importlib
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

if "seestar.gui" not in sys.modules:
    seestar_pkg = types.ModuleType("seestar")
    seestar_pkg.__path__ = [str(ROOT / "seestar")]
    gui_pkg = types.ModuleType("seestar.gui")
    gui_pkg.__path__ = []
    settings_mod = types.ModuleType("seestar.gui.settings")

    class DummySettingsManager:
        pass

    settings_mod.SettingsManager = DummySettingsManager
    hist_mod = types.ModuleType("seestar.gui.histogram_widget")
    hist_mod.HistogramWidget = object
    gui_pkg.settings = settings_mod
    gui_pkg.histogram_widget = hist_mod
    seestar_pkg.gui = gui_pkg
    sys.modules["seestar"] = seestar_pkg
    sys.modules["seestar.gui"] = gui_pkg
    sys.modules["seestar.gui.settings"] = settings_mod
    sys.modules["seestar.gui.histogram_widget"] = hist_mod

qm = importlib.import_module("seestar.queuep.queue_manager")


def test_interbatch_background_preserves_baseline(monkeypatch):
    stacker = qm.SeestarQueuedStacker()
    stacker._interbatch_start_session()
    stacker.update_progress = lambda *a, **k: None

    base_level = 1200.0
    y, x = np.mgrid[:20, :20]
    gradient = (x + y).astype(np.float32)
    original = (base_level + gradient).astype(np.float32)
    weights = np.ones_like(original, dtype=np.float32)

    def fake_estimate_background(image, wht, downsample=4):
        return (base_level + gradient).astype(np.float32)

    monkeypatch.setattr(qm, "estimate_background_2d", fake_estimate_background)

    result, _ = qm.SeestarQueuedStacker._apply_interbatch_normalization(
        stacker, original.copy(), weights.copy(), context="classic", batch_num=1
    )

    original_median = float(np.nanmedian(original))
    result_median = float(np.nanmedian(result))

    assert np.isclose(result_median, original_median, atol=1e-3)
    assert result_median > base_level * 0.5
