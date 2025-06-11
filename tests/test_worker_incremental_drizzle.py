import importlib
import sys
import types
import queue
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

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


def make_wcs(shape=(2, 2)):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = np.array([-0.01, 0.01])
    w.wcs.crval = [0, 0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (shape[1], shape[0])
    return w


def _make_worker(tmp_path):
    obj = qm.SeestarQueuedStacker()
    obj.perform_cleanup = False
    obj.stop_processing = False
    obj.current_folder = str(tmp_path)
    obj.output_folder = str(tmp_path)
    obj.queue = queue.Queue()
    fits.writeto(Path(tmp_path) / "in.fits", np.zeros((2, 2), dtype=np.float32), overwrite=True)
    obj.queue.put(str(Path(tmp_path) / "in.fits"))
    obj.additional_folders = []
    obj.files_in_queue = 1
    obj.batch_size = 1
    obj.drizzle_active_session = True
    obj.drizzle_mode = "Incremental"
    obj.stacked_batches_count = 0
    obj.total_batches_estimated = 1
    obj.mosaic_settings_dict = {}
    obj.update_progress = lambda *a, **k: None
    obj.local_solver_preference = "none"
    obj.astap_search_radius = 1.0
    obj.astap_downsample = 1
    obj.astap_sensitivity = 100
    obj.reference_pixel_scale_arcsec = 1.0
    obj.astap_path = ""
    obj.astap_data_dir = ""
    obj.local_ansvr_path = ""
    obj.api_key = None
    obj.ansvr_timeout_sec = 5
    obj.astap_timeout_sec = 5
    obj.astrometry_net_timeout_sec = 5
    obj.drizzle_fillval = "0.0"
    obj.update_progress = lambda *a, **k: None

    # stub simple reference FITS for _get_reference_image
    ref_path = Path(tmp_path) / "temp_processing" / "reference_image.fit"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(ref_path, np.zeros((2, 2), dtype=np.float32), overwrite=True)

    class DummyAligner:
        def __init__(self):
            self.correct_hot_pixels = True
            self.hot_pixel_threshold = 3.0
            self.neighborhood_size = 5
            self.bayer_pattern = "GRBG"

        def _get_reference_image(self, folder, files, out_folder):
            return np.zeros((2, 2, 3), dtype=np.float32), fits.Header()

    obj.aligner = DummyAligner()

    class DummySolver:
        def solve(self, *a, **k):
            return make_wcs()

    obj.astrometry_solver = DummySolver()
    obj._create_drizzle_output_wcs = lambda ref_wcs, shape, scale: (make_wcs(shape), shape)

    dummy_data = np.zeros((2, 2, 3), dtype=np.float32)
    obj._process_file = lambda *a, **k: (
        dummy_data,
        fits.Header(),
        None,
        None,
        None,
        np.ones((2, 2), dtype=np.float32),
    )
    obj._save_drizzle_input_temp = lambda d, h: str(Path(tmp_path) / "tmp.fits")
    obj._save_final_stack = lambda *a, **k: None

    from drizzle.resample import Drizzle
    obj.incremental_drizzle_objects = [Drizzle(out_shape=(2, 2)) for _ in range(3)]

    calls = {"incremental": 0}

    def fake_incremental(batch, num, total):
        calls["incremental"] += 1
        obj.stop_processing = True

    obj._process_incremental_drizzle_batch = fake_incremental
    obj._process_and_save_drizzle_batch = lambda *a, **k: (_ for _ in ()).throw(AssertionError("final called"))
    obj._process_completed_batch = lambda *a, **k: (_ for _ in ()).throw(AssertionError("classic called"))
    obj.cleanup_temp_reference = lambda: None
    obj._cleanup_drizzle_temp_files = lambda: None
    obj._cleanup_drizzle_batch_outputs = lambda: None
    obj._cleanup_mosaic_panel_stacks_temp = lambda: None

    return obj, calls


def test_worker_calls_incremental_drizzle(tmp_path):
    obj, calls = _make_worker(tmp_path)
    qm.SeestarQueuedStacker._worker(obj)
    assert calls["incremental"] == 1
