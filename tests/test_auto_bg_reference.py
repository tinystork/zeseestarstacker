import sys
from pathlib import Path

import importlib
import logging
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

mosaic_utils = importlib.import_module("seestar.enhancement.mosaic_utils")


def make_wcs(shape=(10, 10)):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = np.array([-0.01, 0.01])
    w.wcs.crval = [0, 0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (shape[1], shape[0])
    return w


def test_auto_background_reference(monkeypatch, tmp_path, caplog):
    wcs = make_wcs()
    data0 = np.zeros((10, 10), dtype=np.float32)
    data1 = np.zeros((10, 10), dtype=np.float32) + 100
    data2 = np.zeros((10, 10), dtype=np.float32) + 200
    paths = []
    for i, data in enumerate([data0, data1, data2]):
        path = tmp_path / f"img{i}.fits"
        fits.writeto(path, data, overwrite=True)
        paths.append((str(path), wcs))

    monkeypatch.setattr(mosaic_utils, "reproject_interp", lambda input_tuple, output_projection, shape_out, **kwargs: (input_tuple[0], np.ones(shape_out, dtype=np.float32)))
    monkeypatch.setattr(mosaic_utils, "detect_stars", lambda img: np.zeros_like(img, dtype=bool))

    caplog.set_level(logging.DEBUG, logger=mosaic_utils.logger.name)
    mosaic_utils.assemble_final_mosaic_with_reproject_coadd(paths, wcs, (10, 10), match_bg=True)

    assert any("Selected background reference index: 1" in rec.getMessage() for rec in caplog.records)
    offsets = [rec.getMessage() for rec in caplog.records if rec.getMessage().startswith("BG_OFFSET idx")]
    assert "[100.0]" in offsets[0]
    assert "[0.0]" in offsets[1]
    assert "[-100.0]" in offsets[2]
