import importlib
import logging
import sys
from pathlib import Path

import pytest

import numpy as np
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import types

# Create minimal stub for seestar package to avoid heavy dependencies
if "seestar" not in sys.modules:
    seestar_pkg = types.ModuleType("seestar")
    seestar_pkg.__path__ = []
    enhancement_pkg = types.ModuleType("seestar.enhancement")
    enhancement_pkg.__path__ = []
    reproj_mod = types.ModuleType("seestar.enhancement.reproject_utils")
    alignment_pkg = types.ModuleType("seestar.alignment")
    alignment_pkg.__path__ = []
    solver_mod = types.ModuleType("seestar.alignment.astrometry_solver")

    class _DummySolver:
        def __init__(self, *a, **k):
            pass

        def solve(self, *a, **k):
            return None

    solver_mod.AstrometrySolver = _DummySolver

    def _missing(*_a, **_k):
        raise ImportError("reproject not available")

    reproj_mod.reproject_and_coadd = _missing
    reproj_mod.reproject_interp = _missing

    enhancement_pkg.reproject_utils = reproj_mod
    seestar_pkg.enhancement = enhancement_pkg
    seestar_pkg.alignment = alignment_pkg
    alignment_pkg.astrometry_solver = solver_mod
    sys.modules["seestar"] = seestar_pkg
    sys.modules["seestar.enhancement"] = enhancement_pkg
    sys.modules["seestar.enhancement.reproject_utils"] = reproj_mod
    sys.modules["seestar.alignment"] = alignment_pkg
    sys.modules["seestar.alignment.astrometry_solver"] = solver_mod

import zemosaic.zemosaic_worker as worker

pytestmark = pytest.mark.skipif(
    not getattr(worker, "PSUTIL_AVAILABLE", True),
    reason="psutil not available"
)


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
            if hasattr(new_wcs.wcs, "crpix"):
                new_wcs.wcs.crpix = [wcs_obj.wcs.crpix[0] - dw, wcs_obj.wcs.crpix[1] - dh]
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
    monkeypatch.setattr(
        worker,
        "reproject_and_coadd",
        lambda input_data, output_projection, shape_out, reproject_function=None, combine_function="mean", match_background=True, **kwargs: (np.zeros(shape_out, dtype=np.float32), np.zeros(shape_out, dtype=np.float32)),
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
            if hasattr(new_wcs.wcs, "crpix"):
                new_wcs.wcs.crpix = [wcs_obj.wcs.crpix[0] - dw, wcs_obj.wcs.crpix[1] - dh]
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
        solver_settings={"use_radec_hints": True},
        solver_instance=dummy_solver,
    )

    expected_ra, expected_dec = wcs_in.wcs_pix2world([[40, 40]], 0)[0]
    assert dummy_solver.called
    assert pytest.approx(dummy_solver.ra, abs=1e-6) == expected_ra
    assert pytest.approx(dummy_solver.dec, abs=1e-6) == expected_dec
    assert captured.get("pixel_shapes") == [(80, 80)]


def test_resolve_after_crop_no_hints(monkeypatch, tmp_path):
    importlib.reload(worker)

    monkeypatch.setattr(worker, "REPROJECT_AVAILABLE", True)
    monkeypatch.setattr(
        worker,
        "reproject_interp",
        lambda input_data, output_projection, shape_out=None, order="bilinear", parallel=False: (input_data[0], np.ones(shape_out)),
    )
    monkeypatch.setattr(
        worker,
        "reproject_and_coadd",
        lambda input_data, output_projection, shape_out, reproject_function=None, combine_function="mean", match_background=True, **kwargs: (np.zeros(shape_out, dtype=np.float32), np.zeros(shape_out, dtype=np.float32)),
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
            if hasattr(new_wcs.wcs, "crpix"):
                new_wcs.wcs.crpix = [wcs_obj.wcs.crpix[0] - dw, wcs_obj.wcs.crpix[1] - dh]
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

    solver = DummySolver()

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
        solver_settings={"use_radec_hints": False},
        solver_instance=solver,
    )

    assert solver.called
    assert solver.ra is None
    assert solver.dec is None
    assert captured.get("pixel_shapes") == [(80, 80)]


def test_solver_header_values_no_wcs(monkeypatch, tmp_path):

    importlib.reload(worker)

    monkeypatch.setattr(worker, "REPROJECT_AVAILABLE", True)
    monkeypatch.setattr(
        worker,
        "reproject_interp",
        lambda input_data, output_projection, shape_out=None, order="bilinear", parallel=False: (input_data[0], np.ones(shape_out)),
    )
    monkeypatch.setattr(
        worker,
        "reproject_and_coadd",
        lambda input_data, output_projection, shape_out, reproject_function=None, combine_function="mean", match_background=True, **kwargs: (np.zeros(shape_out, dtype=np.float32), np.zeros(shape_out, dtype=np.float32)),
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
            if hasattr(new_wcs.wcs, "crpix"):
                new_wcs.wcs.crpix = [wcs_obj.wcs.crpix[0] - dw, wcs_obj.wcs.crpix[1] - dh]
            return cropped, new_wcs

    monkeypatch.setattr(worker, "zemosaic_utils", DummyZU)

    data = np.ones((1, 100, 100), dtype=np.float32)
    fits_path = tmp_path / "tile_no_wcs.fits"
    from astropy.io import fits
    header = fits.Header()
    header["RA"] = 11.1
    header["DEC"] = -22.2
    fits.writeto(fits_path, data, header, overwrite=True)

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

    solver = DummySolver()

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
        solver_settings={"use_radec_hints": True},
        solver_instance=solver,
    )

    expected_ra, expected_dec = wcs_in.wcs_pix2world([[40, 40]], 0)[0]
    assert solver.called
    assert pytest.approx(solver.ra, abs=1e-6) == expected_ra
    assert pytest.approx(solver.dec, abs=1e-6) == expected_dec
    assert captured.get("pixel_shapes") == [(80, 80)]


def test_solver_header_values_no_wcs_no_hints(monkeypatch, tmp_path):

    importlib.reload(worker)

    monkeypatch.setattr(worker, "REPROJECT_AVAILABLE", True)
    monkeypatch.setattr(
        worker,
        "reproject_interp",
        lambda input_data, output_projection, shape_out=None, order="bilinear", parallel=False: (input_data[0], np.ones(shape_out)),
    )
    monkeypatch.setattr(
        worker,
        "reproject_and_coadd",
        lambda input_data, output_projection, shape_out, reproject_function=None, combine_function="mean", match_background=True, **kwargs: (np.zeros(shape_out, dtype=np.float32), np.zeros(shape_out, dtype=np.float32)),
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
            if hasattr(new_wcs.wcs, "crpix"):
                new_wcs.wcs.crpix = [wcs_obj.wcs.crpix[0] - dw, wcs_obj.wcs.crpix[1] - dh]
            return cropped, new_wcs

    monkeypatch.setattr(worker, "zemosaic_utils", DummyZU)

    data = np.ones((1, 100, 100), dtype=np.float32)
    fits_path = tmp_path / "tile_no_wcs.fits"
    from astropy.io import fits
    header = fits.Header()
    header["RA"] = 11.1
    header["DEC"] = -22.2
    fits.writeto(fits_path, data, header, overwrite=True)

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

    solver = DummySolver()

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
        solver_settings={"use_radec_hints": False},
        solver_instance=solver,
    )

    assert solver.called
    assert solver.ra is None
    assert solver.dec is None
    assert captured.get("pixel_shapes") == [(80, 80)]


def test_temp_header_clean(monkeypatch, tmp_path):
    import importlib
    importlib.reload(worker)

    monkeypatch.setattr(worker, "REPROJECT_AVAILABLE", True)
    monkeypatch.setattr(
        worker,
        "reproject_interp",
        lambda input_data, output_projection, shape_out=None, order="bilinear", parallel=False: (input_data[0], np.ones(shape_out)),
    )
    monkeypatch.setattr(
        worker,
        "reproject_and_coadd",
        lambda input_data, output_projection, shape_out, reproject_function=None, combine_function="mean", match_background=True, **kwargs: (np.zeros(shape_out, dtype=np.float32), np.zeros(shape_out, dtype=np.float32)),
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
            if hasattr(new_wcs.wcs, "crpix"):
                new_wcs.wcs.crpix = [wcs_obj.wcs.crpix[0] - dw, wcs_obj.wcs.crpix[1] - dh]
            return cropped, new_wcs

    monkeypatch.setattr(worker, "zemosaic_utils", DummyZU)

    from astropy.io import fits
    data = np.ones((1, 100, 100), dtype=np.float32)
    header = fits.Header()
    header["BSCALE"] = 2.0
    header["BZERO"] = 1000.0
    header["BITPIX"] = 16
    fits_path = tmp_path / "tile.fits"
    fits.writeto(fits_path, data, header, overwrite=True)

    wcs_in = make_wcs(0, 0, shape=(100, 100))

    class DummySolver:
        def __init__(self):
            self.header = None

        def solve(self, image_path, fits_header, settings, update_header_with_solution=True):
            self.header = fits_header.copy()
            return make_wcs(1, 1, shape=(80, 80))

    solver = DummySolver()

    captured = {}
    original_writeto = fits.writeto

    def fake_writeto(path, data, header=None, overwrite=True, **kw):
        captured["header"] = header.copy()
        original_writeto(path, data, header=header, overwrite=overwrite, **kw)

    monkeypatch.setattr(worker.fits, "writeto", fake_writeto)

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
        solver_instance=solver,
    )

    assert solver.header is not None
    assert "BSCALE" not in solver.header
    assert "BZERO" not in solver.header
    assert solver.header.get("BITPIX") == -32
    assert captured.get("header") is not None
    assert "BSCALE" not in captured["header"]
    assert "BZERO" not in captured["header"]
    assert captured["header"].get("BITPIX") == -32


def test_use_sidecar_wcs(monkeypatch, tmp_path):
    import importlib
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
        def load_and_validate_fits(filepath, normalize_to_float32=False, attempt_fix_nonfinite=True, progress_callback=None):
            from astropy.io import fits
            data = np.ones((2, 2), dtype=np.float32)
            hdr = fits.Header()
            hdr["BITPIX"] = 16
            return data, hdr

        @staticmethod
        def debayer_image(img, pattern, progress_callback=None):
            return np.stack([img] * 3, axis=-1)

        @staticmethod
        def detect_and_correct_hot_pixels(img, thr, size, progress_callback=None):
            return img

    monkeypatch.setattr(worker, "zemosaic_utils", DummyZU)

    from astropy.io import fits
    fits_path = tmp_path / "tile.fits"
    fits.writeto(fits_path, np.ones((2, 2), dtype=np.uint16))

    wcs_obj = make_wcs(10, 20, shape=(2, 2))
    sidecar_path = fits_path.with_suffix(".wcs")
    with open(sidecar_path, "w") as f:
        f.write(wcs_obj.to_header(relax=True).tostring(sep="\n"))

    class DummySolver:
        def __init__(self):
            self.called = False

        def solve(self, *args, **kwargs):
            self.called = True
            return make_wcs(0, 0, shape=(2, 2))

    solver = DummySolver()

    img, wcs_out, hdr, _ = worker.get_wcs_and_pretreat_raw_file(
        str(fits_path),
        "",  # astap_exe_path
        "",  # astap_data_dir
        0.0,  # astap_search_radius
        0,    # astap_downsample
        0,    # astap_sensitivity
        10,   # astap_timeout_seconds
        lambda *a, **k: None,
    )

    assert wcs_out is None
    assert not solver.called


def test_output_scale_warning_and_adjust(monkeypatch, caplog):
    import importlib
    importlib.reload(worker)

    monkeypatch.setattr(worker, "REPROJECT_AVAILABLE", True)
    monkeypatch.setattr(
        worker,
        "reproject_interp",
        lambda input_data, output_projection, shape_out=None, order="bilinear", parallel=False: (input_data[0], np.ones(shape_out)),
    )
    monkeypatch.setattr(worker, "ZEMOSAIC_UTILS_AVAILABLE", True)
    monkeypatch.setattr(worker, "ASTROMETRY_SOLVER_AVAILABLE", True)

    monkeypatch.setattr(worker, "CALC_GRID_OPTIMIZED_AVAILABLE", False)

    def dummy_focw(inputs, resolution, auto_rotate=True, projection='TAN', reference=None, frame='icrs'):
        w = make_wcs(0, 0)
        w.wcs.cdelt = np.array([-0.002, 0.002])
        w.pixel_shape = (100, 100)
        return w, (100, 100)

    monkeypatch.setattr(worker, "find_optimal_celestial_wcs", dummy_focw)

    w1 = make_wcs(0, 0)
    w2 = make_wcs(1, 0)

    caplog.set_level(logging.WARNING, logger="ZeMosaicWorker")
    out_wcs, out_shape = worker._calculate_final_mosaic_grid(
        [w1, w2], [(100, 100), (100, 100)], drizzle_scale_factor=1.0, progress_callback=None
    )

    assert out_wcs is not None
    pix_scale = np.mean(np.abs(out_wcs.wcs.cdelt))
    assert np.isclose(pix_scale, 0.002)


def test_grid_uses_resolved_wcs(monkeypatch):
    import importlib
    import os
    importlib.reload(worker)

    captured = {}

    def dummy_calc_grid(wcs_list, shape_list, drizzle_scale_factor=1.0, progress_callback=None):
        captured["wcs"] = wcs_list
        return make_wcs(0, 0), (50, 50)

    monkeypatch.setattr(worker, "_calculate_final_mosaic_grid", dummy_calc_grid)
    monkeypatch.setattr(worker, "ASTROMETRY_SOLVER_AVAILABLE", True)

    class DummySolver:
        def solve(self, image_path, fits_header, settings, update_header_with_solution=True):
            return make_wcs(5, 5, shape=(80, 80))

    in_wcs = make_wcs(0, 0, shape=(100, 100))
    import tempfile
    from astropy.io import fits
    tmpf = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    tmpf.close()
    dummy_path = tmpf.name
    fits.writeto(dummy_path, np.ones((2, 2), dtype=np.float32), overwrite=True)

    worker.prepare_tiles_and_calc_grid(
        [(dummy_path, in_wcs)],
        crop_percent=10.0,
        re_solve_cropped_tiles=True,
        solver_settings={},
        solver_instance=DummySolver(),
        drizzle_scale_factor=1.0,
        progress_callback=None,
    )

    os.remove(dummy_path)

    assert captured.get("wcs")
    assert np.allclose(captured["wcs"][0].wcs.crval, [5, 5])


def test_astrometry_fallback_to_astap(monkeypatch, tmp_path):
    import importlib
    import types
    importlib.reload(worker)

    monkeypatch.setattr(worker, "ASTROMETRY_SOLVER_AVAILABLE", True)
    monkeypatch.setattr(worker, "ZEMOSAIC_ASTROMETRY_AVAILABLE", True)

    monkeypatch.setattr(worker, "solve_with_astrometry", lambda *a, **k: None)

    dummy_wcs = make_wcs(1, 1, shape=(2, 2))

    monkeypatch.setattr(
        worker,
        "zemosaic_astrometry",
        types.SimpleNamespace(solve_with_astap=lambda *a, **k: dummy_wcs),
    )

    astap_exe = tmp_path / "astap.exe"
    astap_exe.write_text(" ")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    from astropy.io import fits

    fits_path = tmp_path / "img.fits"
    fits.writeto(fits_path, np.ones((2, 2), dtype=np.float32), overwrite=True)

    img, wcs_out, hdr, _ = worker.get_wcs_and_pretreat_raw_file(
        str(fits_path),
        str(astap_exe),
        str(data_dir),
        3.0,
        0,
        0,
        10,
        lambda *a, **k: None,
        solver_settings={"solver_choice": "ASTROMETRY"},
    )

    assert wcs_out is not None
    assert np.allclose(wcs_out.wcs.crval, dummy_wcs.wcs.crval)


