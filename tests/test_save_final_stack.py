import importlib
import sys
import types
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


class Dummy:
    pass


def make_wcs(shape=(2, 2)):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = np.array([-0.01, 0.01])
    w.wcs.crval = [0, 0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (shape[1], shape[0])
    return w


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
    obj.drizzle_fillval = "0.0"
    obj.reproject_between_batches = False
    obj.cumulative_sum_memmap = None
    obj.cumulative_wht_memmap = None
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


def test_save_final_stack_preserve_linear_int16(tmp_path):
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
    header = fits.getheader(obj.final_stacked_path)
    assert saved.dtype == np.uint16
    assert header['BZERO'] == 32768
    expected = (np.clip(data, 0.0, 1.0) * 65535).astype(np.uint16)
    assert np.array_equal(saved, expected)


def test_save_final_stack_incremental_drizzle_objects(tmp_path):
    obj = _make_obj(tmp_path, True)
    obj.drizzle_active_session = True
    obj.drizzle_mode = "Incremental"
    obj.preserve_linear_output = True

    shape = (2, 2)
    from drizzle.resample import Drizzle

    obj.incremental_drizzle_objects = [Drizzle(out_shape=shape) for _ in range(3)]
    obj.incremental_drizzle_objects[0].out_img[:] = 1.0
    obj.incremental_drizzle_objects[1].out_img[:] = 2.0
    obj.incremental_drizzle_objects[2].out_img[:] = 3.0
    for d in obj.incremental_drizzle_objects:
        d.out_wht[:] = 1.0

    qm.SeestarQueuedStacker._save_final_stack(
        obj,
        output_filename_suffix="_drizzle_incr_true",
        preserve_linear_output=True,
    )

    saved = fits.getdata(obj.final_stacked_path)
    assert saved.dtype.kind == "f"
    assert saved.shape == (3, 2, 2)
    assert np.any(saved != 0)


def test_save_final_stack_incremental_drizzle_batch(tmp_path):
    obj = _make_obj(tmp_path, True)
    obj.drizzle_active_session = True
    obj.drizzle_mode = "Incremental"
    obj.preserve_linear_output = True
    obj.stop_processing = False
    obj.perform_cleanup = False
    obj.preview_callback = None
    obj._update_preview_incremental_drizzle = lambda: None
    obj.reproject_between_batches = False
    obj.reference_wcs_object = None
    obj.drizzle_output_shape_hw = (5, 5)
    obj.drizzle_output_wcs = make_wcs(shape=obj.drizzle_output_shape_hw)
    obj.drizzle_scale = 1.0
    obj.drizzle_pixfrac = 1.0
    obj.drizzle_kernel = "square"
    obj.images_in_cumulative_stack = 0
    obj.failed_stack_count = 0
    obj.current_stack_header = None

    from drizzle.resample import Drizzle

    obj.incremental_drizzle_objects = [Drizzle(out_shape=obj.drizzle_output_shape_hw) for _ in range(3)]

    wcs = make_wcs(shape=obj.drizzle_output_shape_hw)
    data = np.stack([
        np.full(obj.drizzle_output_shape_hw, c + 1, dtype=np.float32) for c in range(3)
    ], axis=0)
    header = wcs.to_header()
    header["EXPTIME"] = 1.0
    path = tmp_path / "tmp.fits"
    fits.writeto(path, data, header, overwrite=True)

    qm.SeestarQueuedStacker._process_incremental_drizzle_batch(
        obj, [str(path)], current_batch_num=1, total_batches_est=1
    )

    for d in obj.incremental_drizzle_objects:
        assert np.sum(d.out_wht) > 0

    qm.SeestarQueuedStacker._save_final_stack(
        obj,
        output_filename_suffix="_drizzle_incr_true_batch",
        preserve_linear_output=True,
    )

    saved = fits.getdata(obj.final_stacked_path)
    assert saved.shape == (3, 5, 5)
    assert saved[0].max() >= 0.9
    assert saved[1].max() >= 1.9
    assert saved[2].max() >= 2.9


def test_save_final_stack_zero_weights_abort(tmp_path):
    obj = _make_obj(tmp_path, True)
    obj.drizzle_active_session = True
    obj.drizzle_mode = "Incremental"
    obj.preserve_linear_output = True

    shape = (2, 2)
    from drizzle.resample import Drizzle

    obj.incremental_drizzle_objects = [Drizzle(out_shape=shape) for _ in range(3)]
    for idx, d in enumerate(obj.incremental_drizzle_objects):
        d.out_img[:] = idx + 1.0
        d.out_wht[:] = 0.0

    qm.SeestarQueuedStacker._save_final_stack(
        obj,
        output_filename_suffix="_drizzle_incr_true_zero",
        preserve_linear_output=True,
    )

    assert obj.final_stacked_path is None or not Path(obj.final_stacked_path).exists()


def test_incremental_drizzle_batch_weight_override(tmp_path):
    def run_batch(weight=None):
        obj = _make_obj(tmp_path, True)
        obj.drizzle_active_session = True
        obj.drizzle_mode = "Incremental"
        obj.preserve_linear_output = True
        obj.stop_processing = False
        obj.perform_cleanup = False
        obj.preview_callback = None
        obj._update_preview_incremental_drizzle = lambda: None
        obj.reproject_between_batches = False
        obj.reference_wcs_object = None
        obj.drizzle_output_shape_hw = (5, 5)
        obj.drizzle_output_wcs = make_wcs(shape=obj.drizzle_output_shape_hw)
        obj.drizzle_scale = 1.0
        obj.drizzle_pixfrac = 1.0
        obj.drizzle_kernel = "square"
        obj.images_in_cumulative_stack = 0
        obj.failed_stack_count = 0
        obj.current_stack_header = None

        from drizzle.resample import Drizzle

        obj.incremental_drizzle_objects = [Drizzle(out_shape=obj.drizzle_output_shape_hw) for _ in range(3)]

        wcs = make_wcs(shape=obj.drizzle_output_shape_hw)
        data = np.stack([
            np.full(obj.drizzle_output_shape_hw, c + 1, dtype=np.float32) for c in range(3)
        ], axis=0)
        header = wcs.to_header()
        header["EXPTIME"] = 1.0
        suffix = "ovr" if weight is not None else "def"
        path = tmp_path / f"tmp_{suffix}.fits"
        fits.writeto(path, data, header, overwrite=True)

        wht_before = [np.sum(d.out_wht) for d in obj.incremental_drizzle_objects]
        qm.SeestarQueuedStacker._process_incremental_drizzle_batch(
            obj, [str(path)], current_batch_num=1, total_batches_est=1, weight_map_override=weight
        )
        wht_after = [np.sum(d.out_wht) for d in obj.incremental_drizzle_objects]
        for b_val, a_val in zip(wht_before, wht_after):
            assert a_val >= b_val - 1e-6
        return wht_after

    baseline = run_batch(None)
    overridden = run_batch(np.full((5, 5), 0.5, dtype=np.float32))

    for b, o in zip(baseline, overridden):
        assert o < b and np.isclose(o, b * 0.5, rtol=0.1)


def test_incremental_drizzle_batch_weight_accumulates(tmp_path):
    obj = _make_obj(tmp_path, True)
    obj.drizzle_active_session = True
    obj.drizzle_mode = "Incremental"
    obj.preserve_linear_output = True
    obj.stop_processing = False
    obj.perform_cleanup = False
    obj.preview_callback = None
    obj._update_preview_incremental_drizzle = lambda: None
    obj.reproject_between_batches = False
    obj.reference_wcs_object = None
    obj.drizzle_output_shape_hw = (5, 5)
    obj.drizzle_output_wcs = make_wcs(shape=obj.drizzle_output_shape_hw)
    obj.drizzle_scale = 1.0
    obj.drizzle_pixfrac = 1.0
    obj.drizzle_kernel = "square"
    obj.images_in_cumulative_stack = 0
    obj.failed_stack_count = 0
    obj.current_stack_header = None

    from drizzle.resample import Drizzle

    obj.incremental_drizzle_objects = [Drizzle(out_shape=obj.drizzle_output_shape_hw) for _ in range(3)]

    wcs = make_wcs(shape=obj.drizzle_output_shape_hw)
    data = np.stack(
        [np.full(obj.drizzle_output_shape_hw, c + 1, dtype=np.float32) for c in range(3)],
        axis=0,
    )
    header = wcs.to_header()
    header["EXPTIME"] = 1.0

    path1 = tmp_path / "tmp1.fits"
    path2 = tmp_path / "tmp2.fits"
    fits.writeto(path1, data, header, overwrite=True)
    fits.writeto(path2, data, header, overwrite=True)

    wht_before = [np.sum(d.out_wht) for d in obj.incremental_drizzle_objects]
    qm.SeestarQueuedStacker._process_incremental_drizzle_batch(
        obj, [str(path1)], current_batch_num=1, total_batches_est=2
    )
    wht_mid = [np.sum(d.out_wht) for d in obj.incremental_drizzle_objects]
    for b_val, a_val in zip(wht_before, wht_mid):
        assert a_val >= b_val - 1e-6

    qm.SeestarQueuedStacker._process_incremental_drizzle_batch(
        obj, [str(path2)], current_batch_num=2, total_batches_est=2
    )
    wht_after = [np.sum(d.out_wht) for d in obj.incremental_drizzle_objects]
    for b_val, a_val in zip(wht_mid, wht_after):
        assert a_val >= b_val - 1e-6


def test_save_final_stack_classic_reproject(tmp_path):
    obj = _make_obj(tmp_path, True)
    obj.reproject_between_batches = True
    obj.preserve_linear_output = True
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    wht = np.ones_like(data, dtype=np.float32)

    qm.SeestarQueuedStacker._save_final_stack(
        obj,
        output_filename_suffix="_classic_reproject",
        drizzle_final_sci_data=data,
        drizzle_final_wht_data=wht,
    )

    saved = fits.getdata(obj.final_stacked_path)
    assert saved.dtype.kind == "f"
    assert np.allclose(saved.astype(np.float32), data)


def test_save_final_stack_classic_reproject_crop(tmp_path):
    obj = _make_obj(tmp_path, True)
    obj.reproject_between_batches = True
    obj.preserve_linear_output = True

    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    wht = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    obj.current_stack_header["CRPIX1"] = 2.0
    obj.current_stack_header["CRPIX2"] = 2.0

    qm.SeestarQueuedStacker._save_final_stack(
        obj,
        output_filename_suffix="_classic_reproject",
        drizzle_final_sci_data=data,
        drizzle_final_wht_data=wht,
    )

    saved = fits.getdata(obj.final_stacked_path)
    header = fits.getheader(obj.final_stacked_path)

    assert saved.shape == (2, 2)
    assert np.array_equal(saved.astype(np.float32), data[1:3, 1:3])
    assert header["CRPIX1"] == 1.0
    assert header["CRPIX2"] == 1.0


def test_save_final_stack_adds_radec(tmp_path):
    obj = _make_obj(tmp_path, True)
    obj.drizzle_active_session = True
    obj.drizzle_mode = "Final"
    obj.preserve_linear_output = True
    obj.drizzle_output_wcs = make_wcs()

    data = np.ones((2, 2), dtype=np.float32)
    wht = np.ones_like(data, dtype=np.float32)

    qm.SeestarQueuedStacker._save_final_stack(
        obj,
        drizzle_final_sci_data=data,
        drizzle_final_wht_data=wht,
        preserve_linear_output=True,
    )

    hdr = fits.getheader(obj.final_stacked_path)
    assert "RA" in hdr and "DEC" in hdr
    assert np.isclose(hdr["RA"], hdr["CRVAL1"])
    assert np.isclose(hdr["DEC"], hdr["CRVAL2"])



def test_save_final_stack_radec_from_reference_header(tmp_path):
    obj = _make_obj(tmp_path, True)
    obj.cumulative_sum_memmap = np.ones((2, 2, 3), dtype=np.float32)
    obj.cumulative_wht_memmap = np.ones((2, 2), dtype=np.float32)
    obj.reference_header_for_wcs = fits.Header()
    obj.reference_header_for_wcs["RA"] = 12.34
    obj.reference_header_for_wcs["DEC"] = 56.78

    qm.SeestarQueuedStacker._save_final_stack(obj)

    hdr = fits.getheader(obj.final_stacked_path)
    assert np.isclose(hdr["RA"], 12.34)
    assert np.isclose(hdr["DEC"], 56.78)

