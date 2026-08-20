"""Worker wiring test for the M3 single-accumulator drizzle path.

Historically this file asserted that the worker drove the *incremental* drizzle
mode (``_process_incremental_drizzle_batch`` / ``incremental_drizzle_objects``).
That mode has been unified into a single per-channel accumulator
(``drizzle_accumulators`` + ``_add_frame_to_drizzle_accumulators``), so the
worker no longer calls the incremental path.  This test now asserts the worker
feeds each ORIGINAL pose to ``_add_frame_to_drizzle_accumulators`` instead.
"""

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
    obj.drizzle_mode = "Final"  # M3: drizzle_mode is now without effect
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
    tf = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, -0.25]], dtype=np.float64)
    # _process_file returns (data, header, scores, wcs, matrix_m, mask); the
    # worker must feed (data, header, matrix_m, mask, native_wcs=wcs) to the
    # accumulator.
    obj._process_file = lambda *a, **k: (
        dummy_data,
        fits.Header(),
        None,
        None,
        tf,
        np.ones((2, 2), dtype=np.float32),
    )

    calls = {"add_frame": 0}

    def fake_add_frame(original_data, header, tf_val, weight_map, native_wcs=None):
        calls["add_frame"] += 1
        assert native_wcs is None
        obj.stop_processing = True
        return True

    obj._add_frame_to_drizzle_accumulators = fake_add_frame
    obj._move_to_stacked = lambda *a, **k: None
    obj._save_partial_stack = lambda *a, **k: None
    obj._update_batch_count_file = lambda *a, **k: None
    obj._send_eta_update = lambda *a, **k: None
    obj._save_final_stack = lambda *a, **k: None
    # Classic / incremental paths must NOT be exercised anymore.
    obj._process_incremental_drizzle_batch = lambda *a, **k: (_ for _ in ()).throw(AssertionError("incremental called"))
    obj._start_drizzle_process = lambda *a, **k: (_ for _ in ()).throw(AssertionError("incremental start called"))
    obj._process_and_save_drizzle_batch = lambda *a, **k: (_ for _ in ()).throw(AssertionError("final batch called"))
    obj._process_completed_batch = lambda *a, **k: (_ for _ in ()).throw(AssertionError("classic called"))
    obj.cleanup_temp_reference = lambda: None
    obj._cleanup_drizzle_temp_files = lambda: None
    obj._cleanup_drizzle_batch_outputs = lambda: None
    obj._cleanup_mosaic_panel_stacks_temp = lambda: None
    obj._wait_drizzle_processes = lambda: None

    return obj, calls


def test_worker_calls_add_frame_to_drizzle_accumulators(tmp_path):
    obj, calls = _make_worker(tmp_path)
    qm.SeestarQueuedStacker._worker(obj)
    assert calls["add_frame"] == 1
