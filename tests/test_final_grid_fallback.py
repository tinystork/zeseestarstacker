import importlib
import sys
from pathlib import Path

import numpy as np
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))




# Utility to create simple TAN WCS with given shape

def make_wcs(shape=(4, 4)):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = np.array([-0.01, 0.01])
    w.wcs.crval = [0, 0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (shape[1], shape[0])
    return w


def test_fallback_on_invalid_optimal_grid(monkeypatch):
    if "seestar.gui" not in sys.modules:
        import types

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

    obj = qm.SeestarQueuedStacker()
    obj.update_progress = lambda *a, **k: None
    obj.drizzle_scale = 1.0

    w1 = make_wcs()
    w2 = make_wcs()
    hdr1 = w1.to_header()
    hdr2 = w2.to_header()

    def fake_optimal(inputs_for_optimal, **kwargs):
        bad_wcs = make_wcs((1, 1))
        bad_wcs.pixel_shape = (0, 0)
        return bad_wcs, (0, 0)

    monkeypatch.setitem(sys.modules, "reproject.mosaicking", __import__("reproject.mosaicking"))
    monkeypatch.setattr("reproject.mosaicking.find_optimal_celestial_wcs", fake_optimal)

    out_wcs, out_shape = obj._calculate_final_mosaic_grid([w1, w2], [hdr1, hdr2])
    assert out_shape[0] >= 1 and out_shape[1] >= 1
