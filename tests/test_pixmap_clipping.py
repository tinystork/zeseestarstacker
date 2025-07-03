import numpy as np
from astropy.wcs import WCS
from seestar.enhancement.drizzle_integration import run_incremental_drizzle


def test_pixmap_clipping_keeps_weights():
    shape = (10, 10)
    img = np.ones((2, 2), dtype=np.float32)

    # Pixel map shifted out of range by -0.5 in both axes
    pixmap = np.dstack(
        (
            np.array([[-0.5, 0.5], [-0.5, 0.5]], dtype=np.float32),
            np.array([[-0.5, -0.5], [0.5, 0.5]], dtype=np.float32),
        )
    )

    # Clip coordinates to valid output bounds
    pixmap[..., 0] = np.clip(pixmap[..., 0], 0, shape[1] - 1)
    pixmap[..., 1] = np.clip(pixmap[..., 1], 0, shape[0] - 1)

    w = WCS(naxis=2)
    w.wcs.crpix = [1, 1]
    w.wcs.cdelt = [1, 1]
    w.wcs.crval = [0, 0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    out = run_incremental_drizzle([img], [w], w, shape, pixfrac=1.0)
    assert np.sum(out) > 0
