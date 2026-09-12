"""GAR-06 Phase 2 acceptance: REFERENCE_ORIENTATION_GRID_ACCEPT.

The Standard output grid must be an exact scaled copy of the frozen reference
grid: projection / celestial frame / orientation / handedness / effective
pixel-scale matrix preserved (matrix divided by the scale), FITS
edge/centre-preserving CRPIX, public shape metadata equal to the returned
shape, and reference centres / perimeter contained for identity alignment.
"""

import math

import numpy as np
import pytest
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from seestar.core.drizzle_core import (
    DrizzleAccumulator,
    build_output_grid,
    pixmap_from_alignment,
)

REF_SHAPE = (60, 100)  # (H, W), deliberately non-square
ENCODINGS = ["pc", "cd"]
SCALES = [1, 2, 3, 4]


def make_reference(shape=REF_SHAPE, angle=37.0, encoding="pc", crpix=None,
                   crval=(275.0, 30.0), ctype=("RA---TAN", "DEC--TAN"),
                   handedness=-1.0):
    h, w = shape
    ref = WCS(naxis=2)
    if crpix is None:
        crpix = [(w + 1) / 2.0, (h + 1) / 2.0]
    ref.wcs.crpix = list(crpix)
    ref.wcs.crval = list(crval)
    ref.wcs.ctype = list(ctype)
    ref.wcs.cunit = ["deg", "deg"]
    a = math.radians(angle)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    if encoding == "cd":
        ref.wcs.cd = np.diag([handedness * 0.001, 0.001]) @ rot
    else:
        ref.wcs.pc = rot
        ref.wcs.cdelt = [handedness * 0.001, 0.001]
    ref.array_shape = shape
    ref.pixel_shape = (w, h)
    return ref


def _det_sign(wcs):
    return int(np.sign(np.linalg.det(wcs.pixel_scale_matrix)))


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("encoding", ENCODINGS)
def test_output_grid_is_exact_scaled_copy(encoding, scale):
    ref = make_reference(encoding=encoding)
    ref_crpix = np.asarray(ref.wcs.crpix, dtype=float).copy()
    out, out_shape = build_output_grid(ref, REF_SHAPE, scale)

    assert out_shape == (REF_SHAPE[0] * scale, REF_SHAPE[1] * scale)
    # public shape metadata equals the returned shape
    assert out.array_shape == out_shape
    assert out.pixel_shape == (out_shape[1], out_shape[0])
    # effective pixel-scale matrix divided by the scale
    assert np.allclose(
        out.pixel_scale_matrix, ref.pixel_scale_matrix / scale,
        rtol=1e-12, atol=1e-18,
    )
    # orientation and handedness preserved
    assert _det_sign(out) == _det_sign(ref)
    # projection / frame / anchor preserved
    assert list(out.wcs.ctype) == list(ref.wcs.ctype)
    assert np.allclose(out.wcs.crval, ref.wcs.crval)
    # FITS edge/centre-preserving CRPIX
    assert np.allclose(
        out.wcs.crpix, scale * (ref_crpix - 0.5) + 0.5, rtol=0, atol=1e-12
    )
    # reference object untouched (independent copy)
    assert np.allclose(ref.wcs.crpix, ref_crpix)
    # angular scale ratio is exactly 1/scale (independent of the matrix values)
    ratio = proj_plane_pixel_scales(out) / proj_plane_pixel_scales(ref)
    assert np.allclose(ratio, 1.0 / scale, rtol=1e-9)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("angle", [0.0, 37.0])
@pytest.mark.parametrize("handedness", [-1.0, 1.0])
def test_orientation_northup_and_handedness(scale, angle, handedness):
    ref = make_reference(angle=angle, handedness=handedness)
    out, _ = build_output_grid(ref, REF_SHAPE, scale)
    assert _det_sign(out) == int(np.sign(handedness))
    assert np.allclose(
        out.pixel_scale_matrix, ref.pixel_scale_matrix / scale, rtol=1e-12, atol=1e-18
    )


@pytest.mark.parametrize("scale", SCALES)
def test_edge_centre_mapping_with_offcentre_crpix(scale):
    ref = make_reference(crpix=[31.0, 17.0], encoding="pc")
    out, _ = build_output_grid(ref, REF_SHAPE, scale)
    # CRVAL anchor maps to CRPIX_out (1-based) == CRPIX_out - 1 (0-based)
    ra_dec = np.array(ref.wcs.crval)
    ox, oy = out.all_world2pix(np.array([ra_dec[0]]), np.array([ra_dec[1]]), 0)
    assert abs(ox[0] - (out.wcs.crpix[0] - 1.0)) < 1e-9
    assert abs(oy[0] - (out.wcs.crpix[1] - 1.0)) < 1e-9
    # the four reference pixel *edges* map exactly onto the scaled edges
    h, w = REF_SHAPE
    for ex, ey in [(-0.5, -0.5), (w - 0.5, -0.5), (w - 0.5, h - 0.5), (-0.5, h - 0.5)]:
        sky = ref.all_pix2world([[ex, ey]], 0)
        px = out.all_world2pix(sky, 0)[0]
        assert abs(px[0] - (scale * (ex + 0.5) - 0.5)) < 1e-6
        assert abs(px[1] - (scale * (ey + 0.5) - 0.5)) < 1e-6


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("encoding", ENCODINGS)
def test_reference_centres_and_perimeter_contained(encoding, scale):
    ref = make_reference(encoding=encoding)
    h, w = REF_SHAPE
    out, out_shape = build_output_grid(ref, REF_SHAPE, scale)

    # production pixmap: identity alignment must keep every reference centre
    # in-grid (this also proves the public array_shape metadata is correct).
    tf = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    _pixmap, mask = pixmap_from_alignment((h, w), tf, ref, out)
    assert mask.all()

    # independent perimeter/edge projection into the output canvas bounds
    edges = []
    xs = np.linspace(-0.5, w - 0.5, 25)
    ys = np.linspace(-0.5, h - 0.5, 25)
    edges += [(x, -0.5) for x in xs] + [(x, h - 0.5) for x in xs]
    edges += [(-0.5, y) for y in ys] + [(w - 0.5, y) for y in ys]
    sky = ref.all_pix2world(np.array(edges), 0)
    px = out.all_world2pix(sky, 0)
    assert px[:, 0].min() >= -0.5 - 1e-6
    assert px[:, 0].max() <= out_shape[1] - 0.5 + 1e-6
    assert px[:, 1].min() >= -0.5 - 1e-6
    assert px[:, 1].max() <= out_shape[0] - 0.5 + 1e-6


def test_supported_non_tan_projection_preserved():
    ref = make_reference(ctype=("RA---SIN", "DEC--SIN"))
    out, _ = build_output_grid(ref, REF_SHAPE, 3)
    assert list(out.wcs.ctype) == ["RA---SIN", "DEC--SIN"]
    assert np.allclose(
        out.pixel_scale_matrix, ref.pixel_scale_matrix / 3, rtol=1e-12, atol=1e-18
    )


def test_unsupported_distortion_is_refused():
    class _Distorted:
        is_celestial = True
        pixel_shape = (100, 60)
        sip = object()

    with pytest.raises(ValueError):
        build_output_grid(_Distorted(), REF_SHAPE, 2)


def test_shapeless_reference_with_explicit_shape_accepted():
    """F5: the caller supplies the authoritative shape; a shapeless reference
    WCS must not be refused, and the OUTPUT metadata must still match."""
    ref = make_reference()
    ref.pixel_shape = None
    try:
        ref.array_shape = None
    except Exception:  # noqa: BLE001
        pass
    out, out_shape = build_output_grid(ref, REF_SHAPE, 2)
    assert out_shape == (REF_SHAPE[0] * 2, REF_SHAPE[1] * 2)
    assert out.array_shape == out_shape
    assert out.pixel_shape == (out_shape[1], out_shape[0])
    assert np.allclose(
        out.pixel_scale_matrix, ref.pixel_scale_matrix / 2, rtol=1e-12, atol=1e-18
    )


@pytest.mark.parametrize(
    "bad_shape",
    [(0, 10), (10, 0), (-1, 5), (2.5, 10), (10, None), (True, 10), (10,), (1, 2, 3)],
)
def test_invalid_reference_shape_is_rejected(bad_shape):
    ref = make_reference()
    with pytest.raises(ValueError):
        build_output_grid(ref, bad_shape, 2)


def test_invalid_scale_and_non_celestial_refused():
    ref = make_reference()
    with pytest.raises(ValueError):
        build_output_grid(ref, REF_SHAPE, 0.5)
    with pytest.raises(ValueError):
        build_output_grid(ref, REF_SHAPE, float("nan"))
    with pytest.raises(ValueError):
        build_output_grid(None, REF_SHAPE, 2)


@pytest.mark.parametrize("kernel", ["square", "lanczos2", "lanczos3"])
def test_tiny_reference_only_deposition(kernel):
    ref = make_reference(shape=(8, 10))
    out, out_shape = build_output_grid(ref, (8, 10), 2)
    tf = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    pixmap, mask = pixmap_from_alignment((8, 10), tf, ref, out)
    assert mask.all()

    data = np.full((8, 10), 4.0, np.float32)
    acc = DrizzleAccumulator(out_shape, kernel=kernel, pixfrac=1.0)
    acc.add(data, np.ones((8, 10), np.float32), pixmap, in_grid_mask=mask)
    sci = acc.finalize()
    assert sci.shape == out_shape
    assert np.isfinite(sci).all()
    # geometry support is proven independently of the signed-WHT magnitude
    assert sci.sum() > 0.0
