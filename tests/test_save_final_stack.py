import importlib
import sys
import types
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

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
    # M3: the single-accumulator save path validates the science before writing;
    # bind the real method onto the lightweight test double.
    obj._validate_drizzle_science = types.MethodType(
        qm.SeestarQueuedStacker._validate_drizzle_science, obj
    )
    return obj


def test_save_final_stack_preserve_linear_float32(tmp_path):
    obj = _make_obj(tmp_path, True)
    obj.finalization_mode = qm.FINALIZATION_MODE_MOSAIC
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
    obj.finalization_mode = qm.FINALIZATION_MODE_MOSAIC
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
    # M3: the old ``incremental_drizzle_objects`` mode is gone; the final stack
    # is now read from the single per-channel accumulators.
    obj = _make_obj(tmp_path, True)
    obj.drizzle_active_session = True
    obj.preserve_linear_output = True

    shape = (2, 2)
    from seestar.core.drizzle_core import DrizzleAccumulator

    obj.drizzle_accumulators = [DrizzleAccumulator(shape) for _ in range(3)]
    obj.drizzle_accumulators[0]._out_img[:] = 1.0
    obj.drizzle_accumulators[1]._out_img[:] = 2.0
    obj.drizzle_accumulators[2]._out_img[:] = 3.0
    for acc in obj.drizzle_accumulators:
        acc._out_wht[:] = 1.0

    qm.SeestarQueuedStacker._save_final_stack(
        obj,
        output_filename_suffix="_drizzle_incr_true",
        preserve_linear_output=True,
    )

    saved = fits.getdata(obj.final_stacked_path)
    assert saved.dtype.kind == "f"
    assert saved.shape == (3, 2, 2)
    assert np.any(saved != 0)


# Removed test_save_final_stack_incremental_drizzle_batch: it exercised the
# legacy `_process_incremental_drizzle_batch` incremental path, which M3 has
# removed from the worker (single per-channel accumulator instead).  Batch
# invariance is now covered by test_batch_invariance_at_qm_level and the
# accumulator save path by test_save_final_stack_incremental_drizzle_objects.

def test_save_final_stack_zero_weights_abort(tmp_path):
    obj = _make_obj(tmp_path, True)
    obj.drizzle_active_session = True
    obj.preserve_linear_output = True

    shape = (2, 2)
    from seestar.core.drizzle_core import DrizzleAccumulator

    # M3: single per-channel accumulator; all-zero weights must abort cleanly
    # (no SUM/W memmap fallback).
    obj.drizzle_accumulators = [DrizzleAccumulator(shape) for _ in range(3)]
    for acc in obj.drizzle_accumulators:
        acc._out_img[:] = 1.0
        acc._out_wht[:] = 0.0

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
    obj.reproject_coadd_final = True
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
    obj.reproject_coadd_final = True
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
    # M3: drizzle_mode="Final" (batch data) is gone; the final stack is built
    # from the single per-channel accumulators, and its header must still carry
    # RA/DEC propagated from the output WCS.
    obj = _make_obj(tmp_path, True)
    obj.drizzle_active_session = True
    obj.preserve_linear_output = True
    obj.drizzle_output_wcs = make_wcs()

    from seestar.core.drizzle_core import DrizzleAccumulator

    shape = (2, 2)
    obj.drizzle_accumulators = [DrizzleAccumulator(shape) for _ in range(3)]
    for acc in obj.drizzle_accumulators:
        acc._out_img[:] = 1.0
        acc._out_wht[:] = 1.0

    qm.SeestarQueuedStacker._save_final_stack(
        obj,
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


def test_save_final_stack_batch1_negative_int16(tmp_path):
    obj = _make_obj(tmp_path, False)
    obj.batch_size = 1
    obj.reproject_coadd_final = True
    obj.preserve_linear_output = True

    data = np.array([[-2.0, -1.0], [-1.5, 0.0]], dtype=np.float32)
    wht = np.ones_like(data, dtype=np.float32)

    qm.SeestarQueuedStacker._save_final_stack(
        obj,
        output_filename_suffix="_classic_reproject",
        drizzle_final_sci_data=data,
        drizzle_final_wht_data=wht,
        preserve_linear_output=True,
    )

    saved = fits.getdata(obj.final_stacked_path)
    assert np.max(saved) > 0


# ---------------------------------------------------------------------------
# ZSSS-DPIC-01 R1: relative WHT threshold is M3-only; companion WHT lifecycle
# ---------------------------------------------------------------------------


def _make_m3_obj(tmp_path, threshold=0.0, shape=(8, 8)):
    obj = _make_obj(tmp_path, True)
    obj.drizzle_active_session = True
    obj.preserve_linear_output = True
    obj.drizzle_wht_threshold = threshold
    obj.drizzle_output_wcs = make_wcs(shape=shape)
    from seestar.core.drizzle_core import DrizzleAccumulator

    obj.drizzle_accumulators = [DrizzleAccumulator(shape) for _ in range(3)]
    for acc in obj.drizzle_accumulators:
        acc._out_img[:] = 1.0
        acc._out_wht[:] = 1.0
    obj.current_stack_header = obj.drizzle_output_wcs.to_header()
    # ZSSS-OPTIONAL-WHT-01: the companion WHT export is opt-in (default False
    # on a real queue manager).  The M3 companion tests below explicitly opt in.
    obj.save_drizzle_wht = True
    return obj


def test_companion_wht_off_by_default(tmp_path):
    """Default (flag absent) writes NO companion WHT, only the primary SCI."""
    obj = _make_m3_obj(tmp_path, threshold=0.5)
    del obj.save_drizzle_wht  # simulate a fresh queue manager default
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_m3_off_default", preserve_linear_output=True
    )
    assert obj._companion_wht_path is None
    assert not list(tmp_path.glob("*_wht.fit"))
    # primary SCI output still written
    assert obj.final_stacked_path is not None
    assert Path(obj.final_stacked_path).exists()
    assert fits.getdata(obj.final_stacked_path).shape == (3, 8, 8)


def test_companion_wht_false_no_write(tmp_path):
    """Explicit False keeps the companion OFF while the primary is written."""
    obj = _make_m3_obj(tmp_path, threshold=0.5)
    obj.save_drizzle_wht = False
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_m3_off", preserve_linear_output=True
    )
    assert obj._companion_wht_path is None
    assert not list(tmp_path.glob("*_wht.fit"))
    assert obj.final_stacked_path is not None
    assert Path(obj.final_stacked_path).exists()


def test_companion_wht_true_write(tmp_path):
    """Opt-in True writes the companion with the existing metadata contract."""
    obj = _make_m3_obj(tmp_path, threshold=0.5)
    assert obj.save_drizzle_wht is True
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_m3_on", preserve_linear_output=True
    )
    assert obj._companion_wht_path is not None
    assert Path(obj._companion_wht_path).exists()
    wht_data = fits.getdata(obj._companion_wht_path)
    wht_header = fits.getheader(obj._companion_wht_path)
    primary_data = fits.getdata(obj.final_stacked_path)
    assert wht_data.shape == primary_data.shape
    assert wht_header["EXTNAME"] == "WHT"
    for key in ("WHTMIN", "WHTMAX", "WHTMEAN", "COVFRAC"):
        assert key in wht_header


def test_companion_wht_flag_primary_sci_and_png_unchanged(tmp_path):
    """False vs True produce identical primary FITS science and preview PNG."""

    def run(enable):
        obj = _make_m3_obj(tmp_path, threshold=0.5)
        obj.save_drizzle_wht = enable
        qm.SeestarQueuedStacker._save_final_stack(
            obj, output_filename_suffix="_m3_cmp", preserve_linear_output=True
        )
        primary = fits.getdata(obj.final_stacked_path)
        png_path = Path(str(obj.final_stacked_path)).with_suffix(".png")
        png_bytes = png_path.read_bytes() if png_path.exists() else None
        return primary, png_bytes, obj._companion_wht_path

    off_sci, off_png, off_comp = run(False)
    on_sci, on_png, on_comp = run(True)

    # primary science (and preview PNG) are byte-for-byte identical
    assert np.array_equal(off_sci, on_sci)
    assert off_png is not None and on_png is not None
    assert off_png == on_png
    # companion only exists in the opt-in run
    assert off_comp is None
    assert on_comp is not None


def test_wht_threshold_m3_sets_relative_policy(tmp_path):
    obj = _make_m3_obj(tmp_path, threshold=0.5)
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_m3_rel", preserve_linear_output=True
    )
    policy = obj._drizzle_wht_policy_result
    assert policy is not None
    assert policy.reference_support > 0
    assert hasattr(policy, "mask")
    assert abs(policy.fraction - 0.5) < 1e-9


def test_wht_threshold_non_m3_keeps_raw_and_no_policy(tmp_path):
    obj = _make_obj(tmp_path, True)
    obj.finalization_mode = qm.FINALIZATION_MODE_MOSAIC
    obj.drizzle_wht_threshold = 0.5
    data = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    wht = np.array([[1.0, 0.3], [1.0, 0.8]], dtype=np.float32)
    qm.SeestarQueuedStacker._save_final_stack(
        obj,
        output_filename_suffix="_mosaic_raw",
        drizzle_final_sci_data=data,
        drizzle_final_wht_data=wht,
        preserve_linear_output=True,
    )
    # non-M3 never receives an M3 policy object
    assert obj._drizzle_wht_policy_result is None
    saved = fits.getdata(obj.final_stacked_path)
    # raw-absolute threshold semantics: wht 0.3 < 0.5 masked -> 0.0, 0.8 kept
    assert saved[0, 0] == 1.0
    assert saved[0, 1] == 0.0
    assert saved[1, 1] == 1.0


def test_companion_wht_written_for_m3(tmp_path):
    obj = _make_m3_obj(tmp_path, threshold=0.5)
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_m3_comp", preserve_linear_output=True
    )
    assert obj._companion_wht_path is not None
    assert Path(obj._companion_wht_path).exists()
    wht_data = fits.getdata(obj._companion_wht_path)
    wht_header = fits.getheader(obj._companion_wht_path)
    primary_data = fits.getdata(obj.final_stacked_path)
    # CHW layout mirrors the primary spatial shape
    assert wht_data.shape == primary_data.shape
    # WCS/CRPIX carried from the cropped final header
    assert "CRPIX1" in wht_header and "CRPIX2" in wht_header
    assert wht_header["EXTNAME"] == "WHT"
    for key in ("WHTMIN", "WHTMAX", "WHTMEAN", "COVFRAC"):
        assert key in wht_header


def test_companion_wht_threshold_metadata(tmp_path):
    obj = _make_m3_obj(tmp_path, threshold=0.5)
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_m3_threshmeta", preserve_linear_output=True
    )
    wht_header = fits.getheader(obj._companion_wht_path)
    policy = obj._drizzle_wht_policy_result
    assert policy is not None
    assert abs(wht_header["WHTFRAC"] - policy.fraction) < 1e-6
    assert abs(wht_header["WHTREF"] - policy.reference_support) < 1e-6
    assert abs(wht_header["WHTCUT"] - policy.cutoff) < 1e-6
    assert abs(wht_header["WHTMASK"] - policy.masked_fraction) < 1e-6
    # spatial block-support parameters are recorded (named + scale-invariant)
    assert wht_header["WHTTILE"] == policy.tile_size
    assert wht_header["WHTSUPP"] == policy.tile_support_min
    assert wht_header["WHTNPH"] == policy.n_phase_offsets


def test_companion_wht_cropped_with_primary_identical(tmp_path):
    """M3 save witness: real zero-WHT border -> primary & companion crop alike."""
    obj = _make_m3_obj(tmp_path, threshold=0.0, shape=(8, 8))
    # Real zero-WHT border with a nonzero interior [2:6, 2:6].
    for acc in obj.drizzle_accumulators:
        acc._out_img[:] = 0.0
        acc._out_wht[:] = 0.0
        acc._out_img[2:6, 2:6] = 100.0
        acc._out_wht[2:6, 2:6] = 1.0

    crpix1_before = obj.drizzle_output_wcs.wcs.crpix[0]
    crpix2_before = obj.drizzle_output_wcs.wcs.crpix[1]

    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_m3_crop", preserve_linear_output=True
    )

    primary = fits.getdata(obj.final_stacked_path)
    wht_data = fits.getdata(obj._companion_wht_path)
    wht_header = fits.getheader(obj._companion_wht_path)

    # primary (CHW) and companion (CHW) share the exact cropped spatial extent
    assert primary.shape == wht_data.shape
    # cropped to the nonzero interior [2:6, 2:6] -> (3, 4, 4)
    assert primary.shape == (3, 4, 4)
    # CRPIX shifted by the exact x0/y0 (=2, 2)
    assert wht_header["CRPIX1"] == crpix1_before - 2.0
    assert wht_header["CRPIX2"] == crpix2_before - 2.0


def test_companion_wht_float32_no_bzero_integer_primary(tmp_path):
    """Integer primary -> companion stays float32 with no BZERO/BSCALE."""
    obj = _make_m3_obj(tmp_path, threshold=0.0, shape=(8, 8))
    obj.save_final_as_float32 = False  # force the integer (uint16/int16) primary
    for acc in obj.drizzle_accumulators:
        acc._out_img[:] = 0.0
        acc._out_wht[:] = 0.0
        acc._out_img[2:6, 2:6] = 200.0
        acc._out_wht[2:6, 2:6] = 3.0

    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_m3_intcomp", preserve_linear_output=True
    )

    wht_data = fits.getdata(obj._companion_wht_path)
    wht_header = fits.getheader(obj._companion_wht_path)

    # companion is always a 32-bit float WHT, never pseudo-unsigned int
    # semantics (FITS stores float32 as big-endian '>f4', so assert on the
    # kind/itemsize rather than the native byte order)
    assert wht_data.dtype.kind == "f"
    assert wht_data.dtype.itemsize == 4
    assert wht_header["BITPIX"] == -32
    assert "BZERO" not in wht_header
    assert "BSCALE" not in wht_header
    # numerical weights unchanged: interior coverage value 3.0 survives verbatim
    assert np.allclose(np.unique(wht_data), [3.0], atol=1e-6)
    # the primary is genuinely an integer product (pseudo-unsigned int16)
    primary = fits.getdata(obj.final_stacked_path)
    assert primary.dtype.kind in ("i", "u")


def test_companion_failure_leaves_primary_intact(tmp_path, monkeypatch):
    obj = _make_m3_obj(tmp_path, threshold=0.5)

    def _raise_writeto(*args, **kwargs):
        raise OSError("simulated companion write failure")

    # the primary uses HDUList.writeto; the companion uses PrimaryHDU.writeto,
    # so patching PrimaryHDU.writeto only breaks the companion.
    monkeypatch.setattr(fits.PrimaryHDU, "writeto", _raise_writeto)
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_m3_compfail", preserve_linear_output=True
    )
    # primary intact
    assert obj.final_stacked_path is not None
    assert Path(obj.final_stacked_path).exists()
    # no companion left behind
    assert obj._companion_wht_path is None


def test_primary_failure_no_companion(tmp_path, monkeypatch):
    obj = _make_m3_obj(tmp_path, threshold=0.5)

    def _raise_hdulist_writeto(*args, **kwargs):
        raise OSError("simulated primary write failure")

    monkeypatch.setattr(fits.HDUList, "writeto", _raise_hdulist_writeto)
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_m3_primfail", preserve_linear_output=True
    )
    # primary failed -> no companion, path unset
    assert obj._companion_wht_path is None
    assert obj.final_stacked_path is None or not Path(obj.final_stacked_path).exists()


def test_non_m3_no_companion(tmp_path):
    obj = _make_obj(tmp_path, True)
    obj.finalization_mode = qm.FINALIZATION_MODE_MOSAIC
    obj.drizzle_wht_threshold = 0.0
    data = np.array([[0.2, 0.5], [0.3, 0.4]], dtype=np.float32)
    wht = np.ones_like(data, dtype=np.float32)
    qm.SeestarQueuedStacker._save_final_stack(
        obj,
        output_filename_suffix="_mosaic_nocomp",
        drizzle_final_sci_data=data,
        drizzle_final_wht_data=wht,
        preserve_linear_output=True,
    )
    assert obj._companion_wht_path is None


# ---------------------------------------------------------------------------
# ZSSS-DNOW-01 R1: effective Drizzle provenance + native signed WHT + support gate
# ---------------------------------------------------------------------------


def _make_m3_provenance_obj(tmp_path, kernel="square", pixfrac_eff=1.0,
                            scale=2.0, wht_thr_eff=0.0,
                            pixfrac_req=None, wht_thr_req=None):
    obj = _make_m3_obj(tmp_path, threshold=0.0, shape=(8, 8))
    obj.drizzle_kernel = kernel
    obj.drizzle_pixfrac = pixfrac_eff
    obj.drizzle_scale = scale
    obj.drizzle_wht_threshold_effective = wht_thr_eff
    obj.drizzle_wht_threshold = wht_thr_eff
    obj.drizzle_pixfrac_requested = pixfrac_req if pixfrac_req is not None else pixfrac_eff
    obj.drizzle_wht_threshold_requested = (
        wht_thr_req if wht_thr_req is not None else wht_thr_eff
    )
    return obj


def test_m3_primary_and_companion_effective_drz_provenance(tmp_path):
    obj = _make_m3_provenance_obj(
        tmp_path,
        kernel="lanczos2",
        pixfrac_eff=1.0,
        scale=2.0,
        wht_thr_eff=0.0,
        pixfrac_req=0.6,
        wht_thr_req=0.4,
    )
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_m3_prov", preserve_linear_output=True
    )

    hdr = fits.getheader(obj.final_stacked_path)
    assert hdr["DRZKERNEL"] == "lanczos2"
    assert abs(hdr["DRZPIXFR"] - 1.0) < 1e-9   # effective (Lanczos -> 1.0)
    assert abs(hdr["DRZSCALE"] - 2.0) < 1e-9
    assert hdr["DRZMODE"] == "M3"
    assert abs(hdr["DRZWTHT"] - 0.0) < 1e-9   # effective threshold (Lanczos -> 0)
    # requested values recorded separately, never confused with effective
    assert abs(hdr["DRZPFREQ"] - 0.6) < 1e-9
    assert abs(hdr["DRZWTHRQ"] - 0.4) < 1e-9

    # companion inherits the same effective provenance
    whdr = fits.getheader(obj._companion_wht_path)
    for key in ("DRZKERNEL", "DRZPIXFR", "DRZSCALE", "DRZMODE", "DRZWTHT"):
        assert key in whdr, key
    assert abs(whdr["DRZPIXFR"] - 1.0) < 1e-9
    assert abs(whdr["DRZWTHT"] - 0.0) < 1e-9


def test_m3_square_provenance_no_requested_override_keys(tmp_path):
    obj = _make_m3_provenance_obj(
        tmp_path,
        kernel="square",
        pixfrac_eff=0.8,
        scale=1.0,
        wht_thr_eff=0.3,
    )
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_m3_provsq", preserve_linear_output=True
    )
    hdr = fits.getheader(obj.final_stacked_path)
    assert hdr["DRZKERNEL"] == "square"
    assert abs(hdr["DRZPIXFR"] - 0.8) < 1e-9
    assert abs(hdr["DRZSCALE"] - 1.0) < 1e-9
    assert abs(hdr["DRZWTHT"] - 0.3) < 1e-9
    # requested == effective -> no separate requested keys written
    assert "DRZPFREQ" not in hdr
    assert "DRZWTHRQ" not in hdr


def test_companion_retains_signed_native_wht(tmp_path):
    obj = _make_m3_obj(tmp_path, threshold=0.0, shape=(8, 8))
    for acc in obj.drizzle_accumulators:
        acc._out_wht[:] = 1.0
        acc._out_img[:] = 100.0
    # native negative WHT in the interior (survives the positive-support crop)
    obj.drizzle_accumulators[0]._out_wht[3, 3] = -0.5
    obj.drizzle_accumulators[0]._out_wht[4, 4] = -2.0

    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_m3_signed", preserve_linear_output=True
    )

    wht_data = fits.getdata(obj._companion_wht_path)
    whdr = fits.getheader(obj._companion_wht_path)

    # native signed WHT exported verbatim (not max(raw_wht, 0))
    assert float(np.min(wht_data)) <= -1.9
    assert whdr["EXTNAME"] == "WHT"
    assert whdr["WHTMIN"] <= -1.9
    assert whdr["WHTNEG"] >= 2
    assert whdr["WHTZERO"] >= 0
    assert whdr["WHTPOS"] > 0


def test_support_gate_rejects_injected_mismatch(tmp_path, monkeypatch):
    obj = _make_m3_obj(tmp_path, threshold=0.0, shape=(8, 8))
    for acc in obj.drizzle_accumulators:
        acc._out_wht[:] = 1.0
        acc._out_img[:] = 100.0
    # inject a mismatch: invalid support (wht=0) with nonzero native science
    obj.drizzle_accumulators[0]._out_wht[3, 3] = 0.0

    # monkeypatch finalize to return raw native out_img WITHOUT the support
    # gate, simulating a future regression that leaks nonzero science onto
    # invalid support.
    from seestar.core.drizzle_core import DrizzleAccumulator

    def raw_finalize(self, mode="divide"):
        return np.array(self._out_img, dtype=np.float32, copy=True)

    monkeypatch.setattr(DrizzleAccumulator, "finalize", raw_finalize)

    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_m3_gate", preserve_linear_output=True
    )

    # fail-closed: no final stack written, clear diagnostic recorded
    assert obj.final_stacked_path is None or not Path(obj.final_stacked_path).exists()
    assert "support-integrity" in (obj.processing_error or "")


# ---------------------------------------------------------------------------
# ZSSS-DNOW-01 R2: threshold provenance, WHTTYPE discriminator, no cube
# ---------------------------------------------------------------------------


def test_companion_no_threshold_cards_when_effective_zero(tmp_path):
    """R2.1: requested threshold > 0 with effective 0 (Lanczos policy) must not
    emit WHTFRAC/policy cards for a relative threshold that never ran."""
    obj = _make_m3_obj(tmp_path, threshold=0.0, shape=(8, 8))
    obj.drizzle_kernel = "lanczos2"
    obj.drizzle_pixfrac = 1.0
    obj.drizzle_wht_threshold = 0.4           # requested
    obj.drizzle_wht_threshold_requested = 0.4
    obj.drizzle_wht_threshold_effective = 0.0  # effective (signed WHT policy)
    for acc in obj.drizzle_accumulators:
        acc._out_wht[:] = 1.0
        acc._out_img[:] = 100.0
    obj.drizzle_accumulators[0]._out_wht[3, 3] = -0.5

    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_m3_effzero", preserve_linear_output=True
    )

    hdr = fits.getheader(obj.final_stacked_path)
    whdr = fits.getheader(obj._companion_wht_path)

    # effective threshold is 0 on both primary and companion
    assert abs(hdr["DRZWTHT"] - 0.0) < 1e-9
    assert abs(whdr["DRZWTHT"] - 0.0) < 1e-9
    # requested is recorded separately, never confused with effective
    assert abs(hdr["DRZWTHRQ"] - 0.4) < 1e-9
    # no threshold/policy cards when the effective policy did not run
    for key in ("WHTFRAC", "WHTREF", "WHTCUT", "WHTMASK",
                "WHTTILE", "WHTSUPP", "WHTNPH"):
        assert key not in whdr, key
    # signed WHT still exported verbatim
    assert float(np.min(fits.getdata(obj._companion_wht_path))) <= -0.4


def test_companion_whttype_native_discriminator(tmp_path):
    """R2.3: the companion carries an explicit WHTTYPE discriminator, with
    EXTNAME=WHT preserved for compatibility."""
    obj = _make_m3_obj(tmp_path, threshold=0.0, shape=(8, 8))
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_m3_whttype", preserve_linear_output=True
    )
    whdr = fits.getheader(obj._companion_wht_path)
    assert whdr["EXTNAME"] == "WHT"
    assert whdr["WHTTYPE"] == "NATIVE"


def test_m3_finalization_no_full_positive_wht_cube(tmp_path, monkeypatch):
    """R2.2: M3 finalization builds its HWC outputs channel-by-channel and never
    stacks three float32 2-D channels into a full-frame HWC cube."""
    obj = _make_m3_obj(tmp_path, threshold=0.0, shape=(8, 8))
    for acc in obj.drizzle_accumulators:
        acc._out_wht[:] = 0.0
        acc._out_img[:] = 0.0
        acc._out_wht[2:6, 2:6] = 2.0
        acc._out_img[2:6, 2:6] = 100.0
    obj.drizzle_accumulators[0]._out_wht[3, 3] = -0.5

    stacked_cubes = []
    orig_stack = np.stack

    def spy_stack(arrays, *args, **kwargs):
        arrs = list(arrays)
        if (
            len(arrs) == 3
            and all(np.asarray(a).ndim == 2 for a in arrs)
            and all(np.asarray(a).dtype == np.float32 for a in arrs)
        ):
            stacked_cubes.append([np.asarray(a).shape for a in arrs])
        return orig_stack(arrays, *args, **kwargs)

    monkeypatch.setattr(np, "stack", spy_stack)

    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_m3_nocube", preserve_linear_output=True
    )

    # correctness: companion exported as native signed CHW (not clipped)
    wht_data = fits.getdata(obj._companion_wht_path)
    assert float(np.min(wht_data)) <= -0.4
    # allocation boundary: no full HWC float32 cube was ever stacked
    assert stacked_cubes == []
