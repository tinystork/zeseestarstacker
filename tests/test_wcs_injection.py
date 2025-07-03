import importlib
import types
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import zemosaic.zemosaic_worker as worker


def test_wcs_written_back(monkeypatch, tmp_path):
    importlib.reload(worker)

    monkeypatch.setattr(worker, "ZEMOSAIC_UTILS_AVAILABLE", True)
    monkeypatch.setattr(worker, "ZEMOSAIC_ASTROMETRY_AVAILABLE", True)

    def dummy_load(filepath, normalize_to_float32=False, attempt_fix_nonfinite=True, progress_callback=None):
        return np.ones((2, 2, 3), dtype=np.float32), fits.Header(), {}

    dummy_utils = types.SimpleNamespace(
        load_and_validate_fits=dummy_load,
        debayer_image=lambda img, pattern, progress_callback=None: img,
        detect_and_correct_hot_pixels=lambda img, *a, **k: img,
        save_numpy_to_fits=lambda data, header, path, axis_order="CHW": fits.writeto(path, data, header, overwrite=True),
    )
    monkeypatch.setattr(worker, "zemosaic_utils", dummy_utils)

    dummy_wcs = WCS(naxis=2)
    dummy_wcs.wcs.crpix = [1, 1]
    dummy_wcs.wcs.cdelt = [-0.1, 0.1]
    dummy_wcs.wcs.crval = [12.3, 45.6]
    dummy_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    dummy_wcs.pixel_shape = (2, 2)

    def dummy_solver(**kwargs):
        header = kwargs.get("original_fits_header")
        if header is not None:
            header.update(dummy_wcs.to_header(relax=True))
        return dummy_wcs

    monkeypatch.setattr(worker, "zemosaic_astrometry", types.SimpleNamespace(solve_with_astap=dummy_solver))
    monkeypatch.setattr(worker, "astap_paths_valid", lambda *a, **k: True)

    fits_path = tmp_path / "img.fits"
    fits.writeto(fits_path, np.ones((2, 2), dtype=np.float32), overwrite=True)

    img, wcs_out, hdr, _ = worker.get_wcs_and_pretreat_raw_file(
        str(fits_path), "astap", "data", 0.0, 0, 0, 10, lambda *a, **k: None
    )

    assert wcs_out is not None
    with fits.open(fits_path) as hdul:
        assert np.isclose(hdul[0].header["CRVAL1"], 12.3)
        assert np.isclose(hdul[0].header["CRVAL2"], 45.6)
