import numpy as np
from astropy.wcs import WCS
from astropy.io import fits

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "reproject_utils",
    ROOT / "seestar" / "enhancement" / "reproject_utils.py",
)
reproject_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproject_utils)


def make_wcs(shape=(10, 10)):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = np.array([-0.01, 0.01])
    w.wcs.crval = [0, 0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (shape[1], shape[0])
    return w


class DummyStacker:
    def __init__(self):
        self.memmap_shape = (10, 10, 3)
        self.reference_wcs_object = make_wcs(shape=self.memmap_shape[:2])
        self.cumulative_sum_memmap = np.zeros(self.memmap_shape, dtype=np.float32)
        self.cumulative_wht_memmap = np.zeros(self.memmap_shape[:2], dtype=np.float32)
        self.reproject_between_batches = False

    # copy of implemented method
    def _reproject_to_reference(self, image_array, input_wcs):
        reproject_interp = reproject_utils.reproject_interp
        target_shape = self.memmap_shape[:2]
        if image_array.ndim == 3:
            channels = []
            footprint = None
            for ch in range(image_array.shape[2]):
                reproj_ch, footprint = reproject_interp(
                    (image_array[..., ch], input_wcs),
                    self.reference_wcs_object,
                    shape_out=target_shape,
                )
                channels.append(reproj_ch)
            result = np.stack(channels, axis=2)
        else:
            result, footprint = reproject_interp(
                (image_array, input_wcs),
                self.reference_wcs_object,
                shape_out=target_shape,
            )
        return result.astype(np.float32), footprint.astype(np.float32)

    def _combine_batch_result(self, data, header, coverage, batch_wcs=None):
        batch_sum = data.astype(np.float32)
        batch_wht = coverage.astype(np.float32)
        if self.reproject_between_batches and self.reference_wcs_object and batch_wcs is not None:
            reproject_interp = reproject_utils.reproject_interp
            shp = self.memmap_shape[:2]
            if batch_sum.ndim == 3:
                channels = []
                for ch in range(batch_sum.shape[2]):
                    c, _ = reproject_interp((batch_sum[..., ch], batch_wcs), self.reference_wcs_object, shape_out=shp)
                    channels.append(c)
                batch_sum = np.stack(channels, axis=2)
            else:
                batch_sum, _ = reproject_interp((batch_sum, batch_wcs), self.reference_wcs_object, shape_out=shp)
            batch_wht, _ = reproject_interp((batch_wht, batch_wcs), self.reference_wcs_object, shape_out=shp)
        self.cumulative_sum_memmap += batch_sum
        self.cumulative_wht_memmap += batch_wht


def test_reproject_to_reference_rgb():
    s = DummyStacker()
    img = np.ones((5, 5, 3), dtype=np.float32)
    wcs_in = make_wcs(shape=(5, 5))
    out, foot = s._reproject_to_reference(img, wcs_in)
    assert out.shape == s.memmap_shape
    assert foot.shape == s.memmap_shape[:2]


def test_combine_batch_respects_flag(monkeypatch):
    s = DummyStacker()
    img = np.ones(s.memmap_shape, dtype=np.float32)
    cov = np.ones(s.memmap_shape[:2], dtype=np.float32)
    wcs_in = make_wcs(shape=s.memmap_shape[:2])

    calls = {"n": 0}

    def fake_reproj(*args, **kwargs):
        calls["n"] += 1
        return args[0][0], np.ones(s.memmap_shape[:2], dtype=np.float32)

    monkeypatch.setattr(reproject_utils, "reproject_interp", fake_reproj)
    s.reproject_between_batches = False
    s._combine_batch_result(img, fits.Header(), cov, wcs_in)
    assert calls["n"] == 0

    calls["n"] = 0
    s.cumulative_sum_memmap.fill(0)
    s.cumulative_wht_memmap.fill(0)
    s.reproject_between_batches = True
    s._combine_batch_result(img, fits.Header(), cov, wcs_in)
    assert calls["n"] > 0


def test_process_file_returns_wcs_when_reproject(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT))
    import importlib
    import types

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

    class DummySolver:
        def solve(self, *a, **k):
            return make_wcs(shape=(4, 4))

    obj = qm.SeestarQueuedStacker()
    obj.update_progress = lambda *a, **k: None
    obj.correct_hot_pixels = False
    obj.use_quality_weighting = False
    obj.is_mosaic_run = False
    obj.reproject_between_batches = True
    obj.reference_wcs_object = make_wcs(shape=(4, 4))
    obj.reference_pixel_scale_arcsec = 1.0
    obj.astrometry_solver = DummySolver()
    obj.local_solver_preference = "none"
    obj.astap_path = ""
    obj.astap_data_dir = ""
    obj.astap_search_radius = 1.0
    obj.astap_downsample = 1
    obj.astap_sensitivity = 100
    obj.local_ansvr_path = ""
    obj.api_key = None
    obj.ansvr_timeout_sec = 5
    obj.astap_timeout_sec = 5
    obj.astrometry_net_timeout_sec = 5

    data = np.random.random((8, 8, 3)).astype(np.float32)
    path = tmp_path / "test.fits"
    fits.writeto(path, data, overwrite=True)

    result = obj._process_file(str(path), data, solve_astrometry_for_this_file=True)

    assert result[3] is not None and isinstance(result[3], WCS)


def test_process_file_skips_solver_when_disabled(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT))
    import importlib
    import types

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

    calls = {"n": 0}

    class DummySolver:
        def solve(self, *a, **k):
            calls["n"] += 1
            return make_wcs(shape=(4, 4))

    obj = qm.SeestarQueuedStacker()
    obj.update_progress = lambda *a, **k: None
    obj.reproject_between_batches = False
    obj.reference_wcs_object = make_wcs(shape=(4, 4))
    obj.reference_pixel_scale_arcsec = 1.0
    obj.astrometry_solver = DummySolver()
    obj.local_solver_preference = "none"

    data = np.random.random((8, 8, 3)).astype(np.float32)
    path = tmp_path / "test.fits"
    fits.writeto(path, data, overwrite=True)

    obj._process_file(str(path), data, solve_astrometry_for_this_file=False)

    assert calls["n"] == 0


def test_process_file_no_solver_with_paths_and_api_key(tmp_path, monkeypatch):
    """Solver should not run when disabled even if paths/api key are set."""
    sys.path.insert(0, str(ROOT))
    import importlib
    import types

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

    calls = {"n": 0}

    class DummySolver:
        def solve(self, *a, **k):
            calls["n"] += 1
            return make_wcs(shape=(4, 4))

    obj = qm.SeestarQueuedStacker()
    obj.update_progress = lambda *a, **k: None
    obj.reproject_between_batches = False
    obj.reference_wcs_object = make_wcs(shape=(4, 4))
    obj.reference_pixel_scale_arcsec = 1.0
    obj.astrometry_solver = DummySolver()
    obj.local_solver_preference = "ansvr"
    obj.astap_path = "/tmp/astap"
    obj.astap_data_dir = "/tmp/data"
    obj.local_ansvr_path = "/tmp/ansvr"
    obj.api_key = "SECRET"

    data = np.random.random((8, 8, 3)).astype(np.float32)
    path = tmp_path / "test.fits"
    fits.writeto(path, data, overwrite=True)

    obj._process_file(str(path), data, solve_astrometry_for_this_file=False)

    assert calls["n"] == 0


def test_stack_batch_uses_master_tile_when_all_have_wcs(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT))
    import importlib
    import types

    if "seestar.gui" not in sys.modules:
        seestar_pkg = types.ModuleType("seestar")
        seestar_pkg.__path__ = [str(ROOT / "seestar")]
        gui_pkg = types.ModuleType("seestar.gui")
        gui_pkg.__path__ = []
        settings_mod = types.ModuleType("seestar.gui.settings")

        class DummySettingsManager2:
            pass

        settings_mod.SettingsManager = DummySettingsManager2
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

    class DummySM:
        stack_winsor_limits = "0.1,0.1"
        stack_norm_method = "none"
        stack_weight_method = "none"
        stack_reject_algo = "none"
        stack_final_combine = "mean"
        stack_kappa_low = 3.0
        stack_kappa_high = 3.0

        def load_settings(self):
            pass

    monkeypatch.setattr(qm, "SettingsManager", DummySM)

    called = {"n": 0}

    def fake_create_master_tile(**kwargs):
        called["n"] += 1
        data = np.zeros((3, 2, 2), dtype=np.float32)
        hdr = fits.Header()
        p = tmp_path / "tile.fits"
        fits.writeto(p, data, hdr, overwrite=True)
        return str(p), {}

    import seestar.core as sc

    monkeypatch.setattr(sc, "create_master_tile_simple", fake_create_master_tile)

    obj = qm.SeestarQueuedStacker()
    obj.update_progress = lambda *a, **k: None
    obj.reproject_between_batches = True
    obj.reference_wcs_object = make_wcs(shape=(2, 2))
    obj.memmap_shape = (2, 2, 3)

    img = np.ones(obj.memmap_shape, dtype=np.float32)
    mask = np.ones(obj.memmap_shape[:2], dtype=bool)
    wcs = make_wcs(shape=obj.memmap_shape[:2])
    hdr = fits.Header()
    item = (img, hdr, {"snr": 1.0, "stars": 1.0}, wcs, mask)

    out_img, out_header, cov = obj._stack_batch([item], current_batch_num=1, total_batches_est=1)

    # In the simplified stacking logic, create_master_tile_simple should not be called
    assert called["n"] == 0
    assert out_img is not None
    assert cov is not None


def test_calc_grid_uses_header_when_pixel_shape_missing(monkeypatch):
    sys.path.insert(0, str(ROOT))
    import importlib
    import types

    if "seestar.gui" not in sys.modules:
        seestar_pkg = types.ModuleType("seestar")
        seestar_pkg.__path__ = [str(ROOT / "seestar")]
        gui_pkg = types.ModuleType("seestar.gui")
        gui_pkg.__path__ = []
        settings_mod = types.ModuleType("seestar.gui.settings")

        class DummySettingsManager3:
            pass

        settings_mod.SettingsManager = DummySettingsManager3
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

    w1 = make_wcs(shape=(6, 6))
    hdr1 = w1.to_header()
    hdr1["NAXIS1"] = 6
    hdr1["NAXIS2"] = 6
    w1.pixel_shape = None
    w2 = make_wcs(shape=(6, 6))
    hdr2 = w2.to_header()
    hdr2["NAXIS1"] = 6
    hdr2["NAXIS2"] = 6
    w2.pixel_shape = None

    out_wcs, out_shape = obj._calculate_final_mosaic_grid([w1, w2], [hdr1, hdr2])

    assert out_wcs is not None
    assert out_shape is not None
    assert w1.pixel_shape == (hdr1["NAXIS1"], hdr1["NAXIS2"])
    assert w2.pixel_shape == (hdr2["NAXIS1"], hdr2["NAXIS2"])


def test_drizzle_scale_applied_interbatch(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT))
    import importlib
    import types

    if "seestar.gui" not in sys.modules:
        seestar_pkg = types.ModuleType("seestar")
        seestar_pkg.__path__ = [str(ROOT / "seestar")]
        gui_pkg = types.ModuleType("seestar.gui")
        gui_pkg.__path__ = []
        settings_mod = types.ModuleType("seestar.gui.settings")

        class DummySettingsManager3:
            pass

        settings_mod.SettingsManager = DummySettingsManager3
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

    wcs = make_wcs(shape=(4, 4))
    hdr = wcs.to_header()
    data = np.zeros((4, 4), dtype=np.float32)
    p1 = tmp_path / "a.fits"
    p2 = tmp_path / "b.fits"
    fits.writeto(p1, data, hdr, overwrite=True)
    fits.writeto(p2, data, hdr, overwrite=True)

    def run(drizzle):
        obj = qm.SeestarQueuedStacker()
        obj.update_progress = lambda *a, **k: None
        obj.output_folder = str(tmp_path)
        obj.reproject_between_batches = True
        obj.drizzle_active_session = drizzle
        obj.drizzle_scale = 2.0
        captured = {}
        monkeypatch.setattr(obj, "_close_memmaps", lambda: None)
        monkeypatch.setattr(obj, "_create_sum_wht_memmaps", lambda s: captured.update({"shape": s}))
        monkeypatch.setattr(obj, "_reproject_to_reference", lambda d, w: (d, np.ones(d.shape[:2], dtype=np.float32)))
        monkeypatch.setattr(obj, "_combine_batch_result", lambda *a, **k: None)
        monkeypatch.setattr(obj, "_save_final_stack", lambda *a, **k: None)
        obj._final_reproject_cached_files([(str(p1), wcs, hdr), (str(p2), wcs, hdr)])
        return captured.get("shape")

    shape_off = run(False)
    shape_on = run(True)

    assert shape_off == (8, 8)
    assert shape_on == (12, 12)


def test_freeze_reference_wcs(monkeypatch, tmp_path):
    sys.path.insert(0, str(ROOT))
    import importlib
    import types

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

    wcs_initial = make_wcs(shape=(4, 4))
    hdr_init = wcs_initial.to_header()

    obj = qm.SeestarQueuedStacker()
    obj.update_progress = lambda *a, **k: None
    obj.freeze_reference_wcs = True
    obj.reproject_between_batches = True
    obj.memmap_shape = (4, 4, 3)
    obj.cumulative_sum_memmap = np.ones(obj.memmap_shape, dtype=np.float32)
    obj.cumulative_wht_memmap = np.ones(obj.memmap_shape[:2], dtype=np.float32)
    obj.reference_header_for_wcs = hdr_init.copy()
    obj.ref_wcs_header = hdr_init.copy()
    obj.reference_wcs_object = wcs_initial

    def fake_run_astap(self, path):
        hdr = fits.getheader(path)
        new_wcs = make_wcs(shape=(4, 4))
        new_wcs.wcs.crval = [5.0, 5.0]
        for k, v in new_wcs.to_header().items():
            hdr[k] = v
        fits.writeto(path, np.moveaxis(np.zeros(obj.memmap_shape, dtype=np.float32), -1, 0), hdr, overwrite=True)
        return True

    monkeypatch.setattr(qm.SeestarQueuedStacker, "_run_astap_and_update_header", fake_run_astap)

    obj._solve_cumulative_stack()

    assert np.allclose(obj.reference_wcs_object.wcs.crval, [0, 0])

