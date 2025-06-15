import numpy as np
from drizzle.resample import Drizzle


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

    driz = Drizzle(out_shape=shape, fillval=0.0)
    driz.add_image(img, pixmap=pixmap, exptime=1.0, weight_map=np.ones_like(img))

    assert np.sum(driz.out_wht) > 0
