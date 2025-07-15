import sys
from pathlib import Path

import importlib
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


def test_auto_background_reference(monkeypatch, tmp_path):
    wcs = make_wcs()
    data0 = np.zeros((10, 10), dtype=np.float32)
    data1 = np.zeros((10, 10), dtype=np.float32) + 100
    data2 = np.zeros((10, 10), dtype=np.float32) + 200
    paths = []
    for i, data in enumerate([data0, data1, data2]):
        path = tmp_path / f"img{i}.fits"
        fits.writeto(path, data, overwrite=True)
        paths.append((str(path), wcs))

    captured = {}

    def dummy_reproject_and_coadd(input_pairs, output_projection, shape_out, match_background=True, background_reference=None, **kwargs):
        meds = [np.median(img) for img, _ in input_pairs]
        ref = meds[background_reference] if background_reference is not None else np.median(meds)
        offsets = [ref - m for m in meds]
        captured["reference"] = background_reference
        captured["offsets"] = offsets
        return np.zeros(shape_out, dtype=np.float32), np.ones(shape_out, dtype=np.float32)

    def dummy_wrapper(*args, **kwargs):
        return dummy_reproject_and_coadd(list(zip(kwargs["data_list"], kwargs["wcs_list"])), kwargs.get("output_projection"), kwargs.get("shape_out"), match_background=kwargs.get("match_background", True), background_reference=kwargs.get("background_reference"))

    monkeypatch.setattr(mosaic_utils, "reproject_and_coadd", dummy_reproject_and_coadd)
    monkeypatch.setattr(mosaic_utils.zemosaic_utils, "reproject_and_coadd_wrapper", dummy_wrapper)

    mosaic_utils.assemble_final_mosaic_with_reproject_coadd(paths, wcs, (10, 10), match_bg=True)

    assert captured["reference"] == 1
    assert np.allclose(captured["offsets"], [100, 0, -100])
