import importlib
import sys
import types
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

# Stub GUI modules to avoid Tk dependence during import
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

class Dummy:
    pass

def _make_obj(tmp_path, save_as_float32):
    obj = Dummy()
    obj.update_progress = lambda *a, **k: None
    obj._close_memmaps = lambda: None
    obj.save_final_as_float32 = save_as_float32
    obj.drizzle_wht_threshold = 0
    obj.images_in_cumulative_stack = 1
    obj.total_exposure_seconds = 1.0
    obj.output_folder = str(tmp_path)
    obj.output_filename = "out.fit"
    obj.current_stack_header = fits.Header()
    obj.drizzle_active_session = False
    obj.is_mosaic_run = False
    obj.drizzle_mode = "Final"
    obj.drizzle_output_wcs = None
    return obj


def test_save_final_stack_preserve_linear_float32(tmp_path):
    obj = _make_obj(tmp_path, True)
    data = np.array([[0.2, 0.5], [0.3, 0.4]], dtype=np.float32)
    wht = np.ones_like(data, dtype=np.float32)
    qm.SeestarQueuedStacker._save_final_stack(
        obj,
        output_filename_suffix="_mosaic_reproject",
        drizzle_final_sci_data=data,
        drizzle_final_wht_data=wht,
        preserve_linear_output=True,
    )
    saved = fits.getdata(obj.final_stacked_path)
    assert saved.dtype.kind == "f" and saved.dtype.itemsize == 4
    assert np.allclose(saved.astype(np.float32), data)


def test_save_final_stack_preserve_linear_uint16(tmp_path):
    obj = _make_obj(tmp_path, False)
    data = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
    wht = np.ones_like(data, dtype=np.float32)
    qm.SeestarQueuedStacker._save_final_stack(
        obj,
        output_filename_suffix="_mosaic_reproject",
        drizzle_final_sci_data=data,
        drizzle_final_wht_data=wht,
        preserve_linear_output=True,
    )
    saved = fits.getdata(obj.final_stacked_path)
    assert saved.dtype == np.uint16
    expected = (np.clip(data, 0.0, 1.0) * 65535).astype(np.uint16)
    assert np.array_equal(saved, expected)
