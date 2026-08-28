import sys
from pathlib import Path

import numpy as np
from astropy.wcs import WCS
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seestar.core.incremental_reprojection import reproject_and_coadd_batch


def make_wcs(shape=(4, 4)):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = np.array([-0.01, 0.01])
    w.wcs.crval = [0, 0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (shape[1], shape[0])
    return w


def test_reproject_and_coadd_batch_rgb():
    wcs_in = make_wcs()
    hdr = wcs_in.to_header()
    img = np.random.random((4, 4, 3)).astype(np.float32)
    out, cov = reproject_and_coadd_batch([img], [hdr], wcs_in, (4, 4))
    assert out.shape == (4, 4, 3)
    assert cov.shape == (4, 4)

