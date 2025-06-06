import importlib
import logging
import sys
from pathlib import Path

import pytest

import numpy as np
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zemosaic.zemosaic_worker as worker


def make_wcs(ra, dec, shape=(100, 100)):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = np.array([-0.001, 0.001])
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (shape[1], shape[0])
    return w


def test_fallback_warning(caplog):
    importlib.reload(worker)
    worker.find_optimal_celestial_wcs = None

    w1 = make_wcs(0, 0)
    w2 = make_wcs(1, 0)

    def dummy_opt(panel_wcs_list, panel_shapes_hw_list, drizzle_scale_factor):
        return make_wcs(0, 0), (100, 100)

    worker._calculate_final_mosaic_grid_optimized = dummy_opt
    worker.CALC_GRID_OPTIMIZED_AVAILABLE = True

    caplog.set_level(logging.WARNING, logger="ZeMosaicWorker")
    out_wcs, out_shape = worker._calculate_final_mosaic_grid(
        [w1, w2], [(100, 100), (100, 100)], drizzle_scale_factor=1.0, progress_callback=None
    )

    assert out_wcs is not None
    assert out_shape is not None
    assert any(
        "find_optimal_celestial_wcs" in rec.getMessage() and rec.levelno >= logging.WARNING
        for rec in caplog.records
    )


def test_crop_pixel_shape_passed(monkeypatch, tmp_path):
    importlib.reload(worker)

    monkeypatch.setattr(worker, "REPROJECT_AVAILABLE", True)
    monkeypatch.setattr(worker, "reproject_interp", lambda input_data, output_projection, shape_out=None, order='bilinear', parallel=False: (input_data[0], np.ones(shape_out)))
    monkeypatch.setattr(worker, "ZEMOSAIC_UTILS_AVAILABLE", True)
    monkeypatch.setattr(worker, "ASTROMETRY_SOLVER_AVAILABLE", True)
    monkeypatch.setattr(worker, "ASTROMETRY_SOLVER_AVAILABLE", True)

    class DummyZU:
        @staticmethod
        def crop_image_and_wcs(image_data, wcs_obj, crop_fraction, progress_callback=None):
            h, w = image_data.shape[:2]
            dh = int(h * crop_fraction)
            dw = int(w * crop_fraction)
            if image_data.ndim == 3:
                cropped = image_data[dh:h-dh, dw:w-dw, :]
            else:
                cropped = image_data[dh:h-dh, dw:w-dw]
            new_wcs = wcs_obj.copy()
            new_wcs.pixel_shape = (cropped.shape[1], cropped.shape[0])
            return cropped, new_wcs

    monkeypatch.setattr(worker, "zemosaic_utils", DummyZU)

    data = np.ones((1, 100, 100), dtype=np.float32)
    fits_path = tmp_path / "tile.fits"
    from astropy.io import fits
    fits.writeto(fits_path, data, overwrite=True)

    wcs_in = make_wcs(0, 0, shape=(100, 100))

    captured = {}

    def dummy_reproject_and_coadd(input_data, output_projection, shape_out, reproject_function=None, combine_function="mean", match_background=True, **kwargs):
        captured["pixel_shapes"] = [w.pixel_shape for _, w in input_data]
        return np.zeros(shape_out, dtype=np.float32), np.zeros(shape_out, dtype=np.float32)

    monkeypatch.setattr(worker, "reproject_and_coadd", dummy_reproject_and_coadd)

    final_wcs = make_wcs(0, 0, shape=(80, 80))
    final_shape = (80, 80)

    worker.assemble_final_mosaic_with_reproject_coadd(
        [(str(fits_path), wcs_in)],
        final_wcs,
        final_shape,
        progress_callback=None,
        n_channels=1,
        match_bg=False,
        apply_crop=True,
        crop_percent=10.0,
    )

    assert captured.get("pixel_shapes") == [(80, 80)]


def test_resolve_after_crop(monkeypatch, tmp_path):
    importlib.reload(worker)

    monkeypatch.setattr(worker, "REPROJECT_AVAILABLE", True)
    monkeypatch.setattr(
        worker,
        "reproject_interp",
        lambda input_data, output_projection, shape_out=None, order="bilinear", parallel=False: (input_data[0], np.ones(shape_out)),
    )
    monkeypatch.setattr(worker, "ZEMOSAIC_UTILS_AVAILABLE", True)
    monkeypatch.setattr(worker, "ASTROMETRY_SOLVER_AVAILABLE", True)

    class DummyZU:
        @staticmethod
        def crop_image_and_wcs(image_data, wcs_obj, crop_fraction, progress_callback=None):
            h, w = image_data.shape[:2]
            dh = int(h * crop_fraction)
            dw = int(w * crop_fraction)
            cropped = image_data[dh:h-dh, dw:w-dw, :]
            new_wcs = wcs_obj.copy()
            new_wcs.pixel_shape = (cropped.shape[1], cropped.shape[0])
            return cropped, new_wcs

    monkeypatch.setattr(worker, "zemosaic_utils", DummyZU)

    data = np.ones((1, 100, 100), dtype=np.float32)
    fits_path = tmp_path / "tile.fits"
    from astropy.io import fits
    fits.writeto(fits_path, data, overwrite=True)

    wcs_in = make_wcs(0, 0, shape=(100, 100))

    class DummySolver:
        def __init__(self):
            self.called = False
            self.ra = None
            self.dec = None

        def solve(self, image_path, fits_header, settings, update_header_with_solution=True):
            self.called = True
            self.ra = fits_header.get("RA")
            self.dec = fits_header.get("DEC")
            return make_wcs(1, 1, shape=(80, 80))

    dummy_solver = DummySolver()

    captured = {}

    def dummy_reproject_and_coadd(input_data, output_projection, shape_out, reproject_function=None, combine_function="mean", match_background=True, **kwargs):
        captured["pixel_shapes"] = [w.pixel_shape for _, w in input_data]
        return np.zeros(shape_out, dtype=np.float32), np.zeros(shape_out, dtype=np.float32)

    monkeypatch.setattr(worker, "reproject_and_coadd", dummy_reproject_and_coadd)

    final_wcs = make_wcs(0, 0, shape=(80, 80))
    final_shape = (80, 80)

    worker.assemble_final_mosaic_with_reproject_coadd(
        [(str(fits_path), wcs_in)],
        final_wcs,
        final_shape,
        progress_callback=None,
        n_channels=1,
        match_bg=False,
        apply_crop=True,
        crop_percent=10.0,
        re_solve_cropped_tiles=True,
        solver_settings={},
        solver_instance=dummy_solver,
    )

    assert dummy_solver.called
    assert dummy_solver.ra is not None
    assert dummy_solver.dec is not None
    assert captured.get("pixel_shapes") == [(80, 80)]
