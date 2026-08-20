"""Tests for the scientific drizzle core (:mod:`seestar.core.drizzle_core`).

Small, fast, synthetic tests (2D gaussians / constant fields).  Tolerances are
calibrated against the real ``drizzle`` 2.2.0 library (sub-pixel centroid
error ~0.0000 px, flux conservation ratio 1.000000, exact interior mean).
"""

import math

import numpy as np
from astropy.wcs import WCS

from seestar.core.drizzle_core import (
    DrizzleAccumulator,
    build_output_grid,
    drizzle_stream,
    pixmap_from_alignment,
)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def make_wcs(shape_hw, crval=(10.0, 20.0), cdelt=(-0.001, 0.001)):
    """Build a simple TAN WCS whose reference pixel is the grid centre."""
    h, w = shape_hw
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [w / 2.0 + 0.5, h / 2.0 + 0.5]
    wcs.wcs.crval = list(crval)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cdelt = np.array(list(cdelt))
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.pc = np.array([[1.0, 0.0], [0.0, 1.0]])
    wcs.array_shape = (h, w)
    return wcs


def gauss2d(shape_hw, amp, sig, pos):
    """A centred 2D Gaussian (``(x, y)`` position, ``(H, W)`` shape)."""
    h, w = shape_hw
    yy, xx = np.indices((h, w))
    return (amp * np.exp(-((xx - pos[0]) ** 2 + (yy - pos[1]) ** 2) / (2.0 * sig ** 2))).astype(
        np.float32
    )


def identity_tf():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def translation_tf(dx, dy):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)


def rotation_tf(angle_deg, centre):
    """2x3 affine rotating by ``angle_deg`` around ``centre=(cx, cy)``."""
    a = math.radians(angle_deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = np.array(centre, dtype=np.float64)
    t = c - r @ c
    return np.array([[r[0, 0], r[0, 1], t[0]], [r[1, 0], r[1, 1], t[1]]], dtype=np.float64)


def apply_tf(tf, xy):
    """Apply a 2x3 affine to an ``(N, 2)`` array of ``(x, y)`` points."""
    xy = np.asarray(xy, dtype=np.float64)
    ones = np.ones((xy.shape[0], 1))
    return (tf @ np.hstack([xy, ones]).T).T


def centroid(img):
    h, w = img.shape
    yy, xx = np.indices((h, w))
    total = img.sum()
    return (img * xx).sum() / total, (img * yy).sum() / total


# --------------------------------------------------------------------------
# 1. sub-pixel translation centroid
# --------------------------------------------------------------------------


def test_translation_subpixel_centroid():
    shape = (32, 32)
    ref = make_wcs(shape)
    out_wcs, _ = build_output_grid(ref, shape, 1.0)

    tf = translation_tf(0.3, -0.7)
    pixmap, mask = pixmap_from_alignment(shape, tf, ref, out_wcs)

    frame = gauss2d(shape, 100.0, 1.5, (16.0, 16.0))
    acc = DrizzleAccumulator(shape)
    acc.add(frame, np.ones(shape, np.float32), pixmap, in_grid_mask=mask)
    sci = acc.finalize()

    cx, cy = centroid(sci)
    assert abs(cx - 16.3) < 0.05
    assert abs(cy - 15.3) < 0.05


# --------------------------------------------------------------------------
# 2. significant rotation
# --------------------------------------------------------------------------


def test_rotation_significant():
    shape = (40, 40)
    ref = make_wcs(shape)
    out_wcs, _ = build_output_grid(ref, shape, 1.0)

    sources = [(14.0, 14.0), (26.0, 15.0), (19.0, 26.0)]
    frame = np.zeros(shape, np.float32)
    for sx, sy in sources:
        frame += gauss2d(shape, 100.0, 1.5, (sx, sy))

    centre = (20.0, 20.0)
    tf = rotation_tf(25.0, centre)
    pixmap, mask = pixmap_from_alignment(shape, tf, ref, out_wcs)

    acc = DrizzleAccumulator(shape)
    acc.add(frame, np.ones(shape, np.float32), pixmap, in_grid_mask=mask)
    sci = acc.finalize()

    expected = apply_tf(tf, np.array(sources))
    measured = np.array(centroid(sci))
    # The three sources are symmetric enough that the combined centroid is the
    # centroid of the rotated source positions.
    assert np.allclose(measured, expected.mean(axis=0), atol=0.1)


# --------------------------------------------------------------------------
# 3. out-of-grid pixels are excluded (no edge folding)
# --------------------------------------------------------------------------


def test_out_of_grid_excluded():
    shape = (16, 16)
    h, w = shape
    frame = np.full(shape, 10.0, np.float32)

    # Shift the whole frame ~80% out of the grid (+13 px in x).  A direct
    # pixmap (rather than a WCS round-trip) is used so the boundary pixel maps
    # to *exactly* 16.0 and is deterministically out of the grid.
    yy, xx = np.indices(shape, dtype=np.float64)
    pixmap = np.dstack((xx + 13.0, yy))
    in_grid = (
        (pixmap[..., 0] >= 0.0) & (pixmap[..., 0] < w)
        & (pixmap[..., 1] >= 0.0) & (pixmap[..., 1] < h)
    )

    # Only the 3 leftmost frame columns (x+13 < 16) stay in the grid.
    assert in_grid.sum() == 3 * h

    acc = DrizzleAccumulator(shape)
    acc.add(frame, np.ones(shape, np.float32), pixmap, in_grid_mask=in_grid)
    sci = acc.finalize()

    # The far-left band is never reached by the shifted frame: exactly zero
    # flux there (the out-of-grid pixels are masked out, never folded back).
    assert sci[:, :12].sum() == 0.0
    # Only the in-grid 3/16 of the frame contributes to the output.
    assert abs(sci.sum() - 3.0 * h * 10.0) < 1e-3

    # Note: drizzle 2.2.0 natively drops out-of-grid flux (it does not fold it
    # back onto the borders, unlike the historical pixmap *clipping*), so the
    # mask is a safety net; without it the total flux is identical here.
    acc2 = DrizzleAccumulator(shape)
    acc2.add(frame, np.ones(shape, np.float32), pixmap)
    assert np.allclose(sci, acc2.finalize(), atol=1e-6)


# --------------------------------------------------------------------------
# 4. non-uniform weights -> exact weighted mean
# --------------------------------------------------------------------------


def test_nonuniform_weights():
    shape = (16, 16)
    ref = make_wcs(shape)
    out_wcs, _ = build_output_grid(ref, shape, 1.0)
    pixmap, mask = pixmap_from_alignment(shape, identity_tf(), ref, out_wcs)

    val1 = np.full(shape, 10.0, np.float32)
    val2 = np.full(shape, 20.0, np.float32)
    w1 = np.ones(shape, np.float32)
    w2 = np.full(shape, 0.5, np.float32)

    acc = DrizzleAccumulator(shape)
    acc.add(val1, w1, pixmap, in_grid_mask=mask)
    acc.add(val2, w2, pixmap, in_grid_mask=mask)

    sci = acc.finalize()
    wht = acc.wht

    # weighted mean (2*val1 + 1*val2) / 3 = (20 + 20)/3 = 13.333...
    assert np.allclose(sci[8, 8], 40.0 / 3.0, atol=1e-4)
    # interior weight = 1.0 + 0.5
    assert abs(wht[8, 8] - 1.5) < 1e-4

    # a zero-weight region in frame 2 keeps only frame 1 there
    w2_zero = w2.copy()
    w2_zero[:, :8] = 0.0
    acc2 = DrizzleAccumulator(shape)
    acc2.add(val1, w1, pixmap, in_grid_mask=mask)
    acc2.add(val2, w2_zero, pixmap, in_grid_mask=mask)
    sci2 = acc2.finalize()
    assert np.allclose(sci2[8, 0], 10.0, atol=1e-4)  # left half: only frame 1
    assert np.allclose(sci2[8, 12], 40.0 / 3.0, atol=1e-4)  # right half: weighted mean


# --------------------------------------------------------------------------
# 5. batch invariance (group_size must not change the result)
# --------------------------------------------------------------------------


def _synthetic_frames(shape, ref_wcs, out_wcs):
    shifts = [(0.0, 0.0), (0.3, -0.7), (-0.5, 0.2), (0.8, 0.4),
              (-0.2, -0.4), (0.6, -0.1), (-0.7, 0.9), (0.1, 0.5)]
    amps = [100.0, 80.0, 120.0, 60.0, 90.0, 110.0, 70.0, 130.0]
    frames = []
    for i, (sx, sy) in enumerate(shifts):
        if i == 7:
            tf = rotation_tf(8.0, (16.0, 16.0))
        else:
            tf = translation_tf(sx, sy)
        pm, m = pixmap_from_alignment(shape, tf, ref_wcs, out_wcs)
        frames.append((gauss2d(shape, amps[i], 1.5, (16.0 + sx, 16.0 + sy)), pm, m))
    return frames


def test_batch_invariance():
    shape = (32, 32)
    ref = make_wcs(shape)
    out_wcs, _ = build_output_grid(ref, shape, 1.0)
    frames = _synthetic_frames(shape, ref, out_wcs)

    results = {}
    for gs in (1, 2, 5, 8):
        accs = [DrizzleAccumulator(shape) for _ in range(1)]
        gen = (
            (frame[None], np.ones((1, *shape), np.float32), pm, 1.0, "counts")
            for frame, pm, m in frames
        )
        final_sci, final_wht = drizzle_stream(accs, gen, group_size=gs)
        results[gs] = (final_sci.copy(), final_wht.copy())

    base_sci, base_wht = results[1]
    for gs in (2, 5, 8):
        sci, wht = results[gs]
        assert np.allclose(sci, base_sci, atol=1e-6, rtol=1e-5)
        assert np.allclose(wht, base_wht, atol=1e-6, rtol=1e-5)


# --------------------------------------------------------------------------
# 6. flux conservation
# --------------------------------------------------------------------------


def test_flux_conservation():
    shape = (32, 32)
    ref = make_wcs(shape)
    out_wcs, _ = build_output_grid(ref, shape, 1.0)
    pixmap, mask = pixmap_from_alignment(shape, identity_tf(), ref, out_wcs)

    data = np.full(shape, 5.0, np.float32)
    acc = DrizzleAccumulator(shape)
    acc.add(data, np.ones(shape, np.float32), pixmap, in_grid_mask=mask)
    sci = acc.finalize()

    assert np.isclose(sci.sum(), data.sum(), rtol=1e-5)
    assert np.allclose(sci[1:-1, 1:-1], 5.0, atol=1e-4)


# --------------------------------------------------------------------------
# 7. output grid WCS for scale=2
# --------------------------------------------------------------------------


def test_output_grid_wcs():
    shape = (32, 40)
    ref = make_wcs(shape)
    out_wcs, out_shape = build_output_grid(ref, shape, 2.0)

    assert out_shape == (64, 80)
    assert np.allclose(out_wcs.wcs.cdelt, ref.wcs.cdelt / 2.0)
    assert np.allclose(out_wcs.wcs.crval, ref.wcs.crval)
    assert list(out_wcs.wcs.ctype) == list(ref.wcs.ctype)

    # CRVAL (the reference sky) maps to the output reference pixel
    # (CRPIX_out - 1 in 0-based coordinates).
    ra, dec = ref.wcs.crval
    out_x, out_y = out_wcs.all_world2pix(np.array([ra]), np.array([dec]), 0)
    assert abs(out_x[0] - (out_wcs.wcs.crpix[0] - 1.0)) < 1e-6
    assert abs(out_y[0] - (out_wcs.wcs.crpix[1] - 1.0)) < 1e-6

    # a round-trip through the two WCS objects maps a reference pixel to
    # ``scale * p + (scale - 1)`` in the output grid
    h, w = shape
    sky = ref.all_pix2world(np.array([w / 2.0]), np.array([h / 2.0]), 0)
    out_x, out_y = out_wcs.all_world2pix(sky[0], sky[1], 0)
    assert abs(out_x[0] - (2.0 * (w / 2.0) + 1.0)) < 1e-6
    assert abs(out_y[0] - (2.0 * (h / 2.0) + 1.0)) < 1e-6


# --------------------------------------------------------------------------
# 8. frame half out of the field
# --------------------------------------------------------------------------


def test_edges_half_out():
    shape = (16, 16)
    ref = make_wcs(shape)
    out_wcs, _ = build_output_grid(ref, shape, 1.0)

    tf = translation_tf(8.0, 0.0)  # W/2
    pixmap, mask = pixmap_from_alignment(shape, tf, ref, out_wcs)

    frame = np.full(shape, 10.0, np.float32)
    acc = DrizzleAccumulator(shape)
    acc.add(frame, np.ones(shape, np.float32), pixmap, in_grid_mask=mask)
    sci = acc.finalize()

    # half the frame is inside the field -> half the flux, within 1e-2
    assert abs(sci.sum() - 0.5 * frame.sum()) / frame.sum() < 1e-2
    # nothing on the opposite (left) border
    assert sci[:, :7].sum() == 0.0


# --------------------------------------------------------------------------
# 9. exposure-time / units invariance
# --------------------------------------------------------------------------


def test_exptime_units_invariance():
    shape = (16, 16)
    ref = make_wcs(shape)
    out_wcs, _ = build_output_grid(ref, shape, 1.0)
    pixmap, mask = pixmap_from_alignment(shape, identity_tf(), ref, out_wcs)

    data = np.full(shape, 10.0, np.float32)

    a = DrizzleAccumulator(shape)
    a.add(data, np.ones(shape, np.float32), pixmap, exptime=10.0, in_units="counts",
          in_grid_mask=mask)

    b = DrizzleAccumulator(shape)
    b.add((data / 10.0).astype(np.float32), np.ones(shape, np.float32), pixmap,
          exptime=1.0, in_units="cps", in_grid_mask=mask)

    assert np.allclose(a.finalize(), b.finalize(), atol=1e-5)
    assert np.allclose(a.wht, 10.0 * b.wht, atol=1e-4)


# --------------------------------------------------------------------------
# 10. three RGB channels stay separate
# --------------------------------------------------------------------------


def test_rgb_three_channels():
    shape = (16, 16)
    ref = make_wcs(shape)
    out_wcs, _ = build_output_grid(ref, shape, 1.0)
    pixmap, mask = pixmap_from_alignment(shape, identity_tf(), ref, out_wcs)

    r = np.full(shape, 1.0, np.float32)
    g = np.full(shape, 2.0, np.float32)
    b = np.full(shape, 3.0, np.float32)
    data_cxhxw = np.stack([r, g, b], axis=0)
    weight_cxhxw = np.ones_like(data_cxhxw)

    accs = [DrizzleAccumulator(shape) for _ in range(3)]
    frames = [(data_cxhxw, weight_cxhxw, pixmap, 1.0, "counts")]
    final_sci, final_wht = drizzle_stream(accs, iter(frames), group_size=1)

    assert final_sci.shape == (16, 16, 3)
    assert np.allclose(final_sci[8, 8, 0], 1.0, atol=1e-4)
    assert np.allclose(final_sci[8, 8, 1], 2.0, atol=1e-4)
    assert np.allclose(final_sci[8, 8, 2], 3.0, atol=1e-4)
