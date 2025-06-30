import importlib
import types
import logging
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub GUI modules to avoid Tk dependence during import
if "seestar.gui" not in sys.modules:
    seestar_pkg = types.ModuleType("seestar")
    seestar_pkg.__path__ = [str(ROOT / "seestar")]
    gui_pkg = types.ModuleType("seestar.gui")
    gui_pkg.__path__ = [str(ROOT / "seestar" / "gui")]
    settings_mod = types.ModuleType("seestar.gui.settings")
    settings_mod.SettingsManager = object
    hist_mod = types.ModuleType("seestar.gui.histogram_widget")
    hist_mod.HistogramWidget = object
    gui_pkg.settings = settings_mod
    gui_pkg.histogram_widget = hist_mod
    seestar_pkg.gui = gui_pkg
    sys.modules["seestar"] = seestar_pkg
    sys.modules["seestar.gui"] = gui_pkg
    sys.modules["seestar.gui.settings"] = settings_mod
    sys.modules["seestar.gui.histogram_widget"] = hist_mod

SeestarStackerGUI = importlib.import_module("seestar.gui.main_window").SeestarStackerGUI


class DummyVar:
    def __init__(self, value=None):
        self._val = value

    def get(self):
        return self._val

    def set(self, val):
        self._val = val


def test_dynamic_autostretch_triggers():
    gui = SeestarStackerGUI.__new__(SeestarStackerGUI)
    gui.root = types.SimpleNamespace(after=lambda *a, **k: None)
    gui.logger = logging.getLogger("test")
    gui._final_stretch_set_by_processing_finished = False
    gui.initial_auto_stretch_done = True
    gui.preview_black_point = DummyVar(0.25)
    gui.preview_white_point = DummyVar(0.75)
    gui.drizzle_mode_var = DummyVar("Incremental")

    def fake_refresh(recalculate_histogram=True):
        pass

    gui.refresh_preview = fake_refresh

    def fake_auto_stretch():
        gui.preview_black_point.set(0.1)
        gui.preview_white_point.set(0.6)

    gui.apply_auto_stretch = fake_auto_stretch

    gui.current_preview_data = np.full((100, 100), 0.02, dtype=np.float32)

    SeestarStackerGUI.update_preview_from_stacker(
        gui,
        gui.current_preview_data,
        None,
        "dummy",
        5,
        10,
        1,
        2,
    )

    assert gui.preview_black_point.get() == 0.1
