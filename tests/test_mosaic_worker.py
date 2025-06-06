import importlib
import logging
import sys
from pathlib import Path

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
