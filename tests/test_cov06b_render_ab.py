"""COV-06B deterministic render OFF/ON scientific-equivalence witness."""

from pathlib import Path
import inspect
import types

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from seestar.queuep import queue_manager as qm


class _Finalizer:
    pass


def _make_finalizer(output_dir: Path, *, render: bool):
    obj = _Finalizer()
    obj.update_progress = lambda *args, **kwargs: None
    obj._close_memmaps = lambda: None
    obj.save_final_as_float32 = True
    obj.preserve_linear_output = True
    obj.drizzle_wht_threshold = 0.0
    obj.images_in_cumulative_stack = 8
    obj.total_exposure_seconds = 80.0
    obj.output_folder = str(output_dir)
    obj.output_filename = "cov06b_ab.fit"
    obj.drizzle_active_session = False
    obj.is_mosaic_run = True
    obj.drizzle_mode = "Final"
    obj.drizzle_output_wcs = None
    obj.drizzle_fillval = "0.0"
    obj.reproject_between_batches = False
    obj.reproject_coadd_final = False
    obj.cumulative_sum_memmap = None
    obj.cumulative_wht_memmap = None
    obj.finalization_mode = qm.FINALIZATION_MODE_MOSAIC
    obj.apply_feathering = False
    obj.apply_low_wht_mask = False
    obj.apply_coverage_render = render
    obj.coverage_render_n_ref = 32.0
    obj.coverage_render_applied_in_session = False
    obj.processing_error = None

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [48.0, 48.0]
    wcs.wcs.cdelt = np.array([-0.01, 0.01])
    wcs.wcs.crval = [275.0, -13.7]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    obj.current_stack_header = wcs.to_header()

    # SUP_W1=4, SUP_W2=4 -> N_eff_support=4 everywhere: low enough for the
    # renderer to produce a deterministic cosmetic difference.
    obj.coverage_sup_w1_memmap = np.full((96, 96), 4.0, dtype=np.float64)
    obj.coverage_sup_w2_memmap = np.full((96, 96), 4.0, dtype=np.float64)
    obj._derive_neff_support_for_render = types.MethodType(
        qm.SeestarQueuedStacker._derive_neff_support_for_render, obj
    )
    return obj


def _run(output_dir: Path, *, render: bool, sci, wht):
    output_dir.mkdir()
    obj = _make_finalizer(output_dir, render=render)
    support_before = (
        obj.coverage_sup_w1_memmap.copy(),
        obj.coverage_sup_w2_memmap.copy(),
    )
    header_before = obj.current_stack_header.copy()
    sci_before = sci.copy()
    wht_before = wht.copy()

    qm.SeestarQueuedStacker._save_final_stack(
        obj,
        drizzle_final_sci_data=sci,
        drizzle_final_wht_data=wht,
        preserve_linear_output=True,
        finalization_mode=qm.FINALIZATION_MODE_MOSAIC,
    )

    return {
        "fits_sci": fits.getdata(obj.final_stacked_path),
        "fits_header": fits.getheader(obj.final_stacked_path),
        "preview": obj.last_saved_data_for_preview.copy(),
        "sci_after": sci.copy(),
        "wht_after": wht.copy(),
        "sci_before": sci_before,
        "wht_before": wht_before,
        "support_before": support_before,
        "support_after": (
            obj.coverage_sup_w1_memmap.copy(),
            obj.coverage_sup_w2_memmap.copy(),
        ),
        "header_before": header_before,
        "render_applied": obj.coverage_render_applied_in_session,
    }


def test_render_on_changes_only_cosmetic_product(tmp_path):
    rng = np.random.default_rng(606)
    sci = (1000.0 + rng.normal(0.0, 80.0, (96, 96))).astype(np.float32)
    wht = np.linspace(0.6, 158.95, 96 * 96, dtype=np.float32).reshape(96, 96)

    off = _run(tmp_path / "off", render=False, sci=sci.copy(), wht=wht.copy())
    on = _run(tmp_path / "on", render=True, sci=sci.copy(), wht=wht.copy())

    # Strong scientific FITS contract: exact SCI and complete header equality.
    assert np.array_equal(off["fits_sci"], on["fits_sci"])
    assert off["fits_header"].tostring() == on["fits_header"].tostring()

    # The source SCI/WHT and positive support state are untouched in both runs.
    for result in (off, on):
        assert np.array_equal(result["sci_before"], result["sci_after"])
        assert np.array_equal(result["wht_before"], result["wht_after"])
        assert np.array_equal(result["support_before"][0], result["support_after"][0])
        assert np.array_equal(result["support_before"][1], result["support_after"][1])
        for key in ("CTYPE1", "CTYPE2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2"):
            assert result["fits_header"][key] == result["header_before"][key]

    # Only the downstream cosmetic render product differs.
    assert off["render_applied"] is False
    assert on["render_applied"] is True
    assert not np.array_equal(off["preview"], on["preview"])


def test_drizzle_render_confidence_uses_positive_support_not_signed_wht():
    obj = _Finalizer()
    obj.coverage_sup_w1_memmap = None
    obj.coverage_sup_w2_memmap = None
    obj._drizzle_support_available = True
    positive_neff = np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float32)
    obj._drizzle_support_n_eff = lambda: positive_neff.copy()
    # A signed native WHT is deliberately present but must never be consulted.
    obj.native_wht = np.array([[-2.0, 0.0], [1.0, 3.0]], dtype=np.float32)

    result = qm.SeestarQueuedStacker._derive_neff_support_for_render(obj)
    assert np.array_equal(result, positive_neff)
    assert np.all(result >= 0.0)


def test_start_processing_backend_binding_is_explicit_and_safe_default():
    method = qm.SeestarQueuedStacker.start_processing
    parameter = inspect.signature(method).parameters["apply_coverage_render"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False
    source = inspect.getsource(method)
    assert "self.apply_coverage_render = bool(apply_coverage_render)" in source
