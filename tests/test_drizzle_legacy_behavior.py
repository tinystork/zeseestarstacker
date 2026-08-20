"""Documentary test of the historical (defective) double-pass drizzle.

The historical "Final" mode drizzled ``N`` frames in batches onto the final
grid, saved the *raw* ``out_img``/``out_wht`` of each batch (no division), and
then re-drizzled those batch results (``data=sci_batch``,
``weight_map=wht_batch``, ``in_units="cps"``) with the pixmap clipped to
``[0, W-1]``.

This test reconstructs that double-pass minimally and demonstrates that its
result depends on the batch size and folds edge flux, unlike a single
accumulator.

.. note::
    ``drizzle`` 2.2.0 stores the *weighted mean* in ``out_img`` and the total
    weight in ``out_wht``.  The historical code believed ``out_img`` held
    ``SCI*WHT`` (the un-normalised weighted flux).  That quantity is
    ``out_img * out_wht``, which is what the historical double-pass therefore
    fed into the second pass.
"""

import numpy as np
from drizzle.resample import Drizzle

from seestar.core.drizzle_core import DrizzleAccumulator

H = W = 32
N_FRAMES = 8


def _make_frames():
    """8 constant frames with varied amplitude and sub-pixel shifts."""
    amps = [10.0, 20.0, 15.0, 25.0, 12.0, 18.0, 22.0, 16.0]
    shifts = [(0.0, 0.0), (0.2, -0.3), (-0.4, 0.1), (0.3, 0.2),
              (-0.1, -0.2), (0.5, -0.1), (-0.2, 0.4), (0.1, 0.3)]
    yy, xx = np.indices((H, W), dtype=np.float64)
    frames = [np.full((H, W), a, np.float32) for a in amps]
    pixmaps = [np.dstack((xx + sx, yy + sy)) for sx, sy in shifts]
    return frames, pixmaps


def _batch_drizzle(frames, pixmaps):
    """First pass: drizzle a group of frames onto the final grid (raw)."""
    d = Drizzle(out_img=np.zeros((H, W), np.float32),
                out_wht=np.zeros((H, W), np.float32),
                kernel="square", fillval="0.0")
    for f, p in zip(frames, pixmaps):
        d.add_image(f, exptime=1.0, pixmap=p,
                    weight_map=np.ones_like(f), in_units="counts", pixfrac=1.0)
    # historical "SCI*WHT" quantity = weighted flux = out_img * out_wht
    return d.out_img * d.out_wht, d.out_wht.copy()


def _legacy_combine(batches, offset, clip):
    """Second pass: re-drizzle raw batch results with a clipped pixmap."""
    yy, xx = np.indices((H, W), dtype=np.float64)
    pixmap = np.dstack((xx + offset, yy + offset))
    if clip:
        pixmap = np.clip(pixmap, 0, W - 1)

    d = Drizzle(out_img=np.zeros((H, W), np.float32),
                out_wht=np.zeros((H, W), np.float32),
                kernel="square", fillval="0.0")
    for sci_b, wht_b in batches:
        d.add_image(sci_b, exptime=1.0, pixmap=pixmap,
                    weight_map=wht_b, in_units="cps", pixfrac=1.0)
    return d.out_img.copy()


def _legacy_double_pass(batch_size, offset=0.0, clip=True):
    frames, pixmaps = _make_frames()
    batches = []
    for i in range(0, N_FRAMES, batch_size):
        batches.append(_batch_drizzle(frames[i:i + batch_size],
                                      pixmaps[i:i + batch_size]))
    return _legacy_combine(batches, offset, clip)


def _single_accumulator():
    frames, pixmaps = _make_frames()
    acc = DrizzleAccumulator((H, W))
    for f, p in zip(frames, pixmaps):
        acc.add(f, np.ones_like(f), p, exptime=1.0, in_units="counts")
    return acc.finalize()


def test_legacy_double_pass_is_batch_dependent():
    l2 = _legacy_double_pass(2)
    l4 = _legacy_double_pass(4)
    l8 = _legacy_double_pass(8)

    # different batch sizes give different results
    assert np.abs(l2 - l4).max() > 1e-3
    assert np.abs(l4 - l8).max() > 1e-3
    assert np.abs(l2 - l8).max() > 1e-3

    # and they all differ from the single-accumulator result
    single = _single_accumulator()
    for legacy in (l2, l4, l8):
        assert np.abs(single - legacy).max() > 1e-3


def test_legacy_pixmap_clip_folds_edge_flux():
    # With a half-pixel offset in the second pass, clipping the pixmap folds
    # the out-of-grid edge column back onto the border; without clipping that
    # flux is dropped.
    clipped = _legacy_double_pass(8, offset=0.5, clip=True)
    unclipped = _legacy_double_pass(8, offset=0.5, clip=False)

    assert np.abs(clipped - unclipped).max() > 1e-3
    # the difference is localised at the border, not in the interior
    assert np.abs(clipped[1:-1, 1:-1] - unclipped[1:-1, 1:-1]).max() < 1e-3
