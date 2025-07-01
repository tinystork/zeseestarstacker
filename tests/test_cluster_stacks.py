import importlib

import numpy as np
from astropy.wcs import WCS

import zemosaic.zemosaic_worker as worker


def make_wcs(ra, dec, shape=(100, 100)):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = np.array([-0.001, 0.001])
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (shape[1], shape[0])
    return w


def test_basic_clustering():
    importlib.reload(worker)

    infos = [
        {"wcs": make_wcs(0.0, 0.0)},
        {"wcs": make_wcs(0.05, 0.05)},
        {"wcs": make_wcs(1.0, 1.0)},
    ]

    groups = worker.cluster_seestar_stacks(
        infos, stack_threshold_deg=0.1, progress_callback=lambda *a, **k: None
    )

    assert len(groups) == 2
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2]
