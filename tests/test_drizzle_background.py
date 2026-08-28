"""Unit tests for M3 Drizzle photometric integrity (DPIC-01).

Covers the immutable additive background matching module
(:mod:`seestar.core.drizzle_background`) and the relative WHT threshold +
kernel/pixfrac validation helpers (:mod:`seestar.core.drizzle_core`).

All tests are deterministic, synthetic and fast (no real FITS / GPU).
"""

import numpy as np
import pytest
from astropy.wcs import WCS

from seestar.core.drizzle_background import (
    ANCHOR_VERSION,
    DEFAULT_MAX_SAMPLES,
    BackgroundAnchor,
    apply_background_offsets,
    estimate_background_offsets,
    invert_affine_2x3,
    native_wcs_to_reference_coords,
    rescale_01_to_adu,
    sample_bilinear,
)
from seestar.core.drizzle_core import (
    VALID_DRIZZLE_KERNELS,
    DrizzleAccumulator,
    validate_drizzle_kernel,
    validate_drizzle_pixfrac,
    wht_relative_threshold,
)


def _identity_tf():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def _translation_tf(dx, dy):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)


def _ones_weight(shape):
    return np.ones(shape, dtype=np.float32)


# ---------------------------------------------------------------------------
# BackgroundAnchor / estimator
# ---------------------------------------------------------------------------


def test_identical_background_zero_correction():
    shape = (32, 32)
    frame = np.full((*shape, 3), 100.0, dtype=np.float32)
    anchor = BackgroundAnchor(frame, reference_shape_hw=shape)
    offsets, diag = estimate_background_offsets(
        frame, _ones_weight(shape), _identity_tf(), anchor
    )
    assert diag["reason"] == "accepted"
    assert np.allclose(offsets, 0.0, atol=1e-3)


def test_constant_offset_recovered():
    shape = (32, 32)
    anchor = BackgroundAnchor(np.full((*shape, 3), 100.0, np.float32), reference_shape_hw=shape)
    frame = np.full((*shape, 3), 104.0, np.float32)
    offsets, diag = estimate_background_offsets(
        frame, _ones_weight(shape), _identity_tf(), anchor
    )
    assert np.allclose(offsets, 4.0, atol=0.05)
    # corrected background matches the anchor
    corrected = apply_background_offsets(frame, offsets)
    assert np.allclose(np.median(corrected, axis=(0, 1)), 100.0, atol=0.05)


def test_independent_rgb_offsets_recovered():
    shape = (32, 32)
    anchor = BackgroundAnchor(
        np.stack([np.full(shape, 100.0), np.full(shape, 110.0), np.full(shape, 90.0)], -1).astype(np.float32),
        reference_shape_hw=shape,
    )
    frame = np.stack([np.full(shape, 104.0), np.full(shape, 116.0), np.full(shape, 87.0)], -1).astype(np.float32)
    offsets, diag = estimate_background_offsets(
        frame, _ones_weight(shape), _identity_tf(), anchor
    )
    assert np.allclose(offsets, [4.0, 6.0, -3.0], atol=0.05)


def test_stars_do_not_bias_estimate():
    shape = (48, 48)
    rng = np.random.default_rng(0)
    anchor = BackgroundAnchor(np.full((*shape, 3), 100.0, np.float32), reference_shape_hw=shape)
    frame = np.full((*shape, 3), 104.0, np.float32)
    # a handful of bright "stars" (positive outliers) must not move the median
    for _ in range(30):
        x, y = rng.integers(0, shape[1]), rng.integers(0, shape[0])
        frame[y, x, :] += 5000.0
    offsets, diag = estimate_background_offsets(
        frame, _ones_weight(shape), _identity_tf(), anchor
    )
    assert np.allclose(offsets, 4.0, atol=0.2)


def test_nan_and_invalid_excluded():
    shape = (32, 32)
    anchor = BackgroundAnchor(np.full((*shape, 3), 100.0, np.float32), reference_shape_hw=shape)
    frame = np.full((*shape, 3), 104.0, np.float32)
    frame[0:4, 0:4, :] = np.nan
    weight = _ones_weight(shape)
    weight[0:4, 0:4] = 0.0  # invalid weight mask region
    offsets, diag = estimate_background_offsets(frame, weight, _identity_tf(), anchor)
    assert diag["reason"] == "accepted"
    assert np.allclose(offsets, 4.0, atol=0.1)


def test_insufficient_overlap_neutral_fallback():
    shape = (32, 32)
    anchor = BackgroundAnchor(np.full((*shape, 3), 100.0, np.float32), reference_shape_hw=shape)
    frame = np.full((*shape, 3), 104.0, np.float32)
    # shift the frame almost entirely out of the reference grid
    offsets, diag = estimate_background_offsets(
        frame, _ones_weight(shape), _translation_tf(500.0, 500.0), anchor
    )
    assert diag["reason"] == "insufficient_overlap"
    assert np.allclose(offsets, 0.0)
    assert diag["n_overlap"] < 200


def test_correction_is_pure_and_state_does_not_evolve():
    shape = (32, 32)
    anchor = BackgroundAnchor(np.full((*shape, 3), 100.0, np.float32), reference_shape_hw=shape)
    meta_before = anchor.to_metadata()
    bg_before = anchor.background.copy()

    fa = np.full((*shape, 3), 104.0, np.float32)
    fb = np.full((*shape, 3), 97.0, np.float32)

    off_a1, _ = estimate_background_offsets(fa, _ones_weight(shape), _identity_tf(), anchor)
    off_b, _ = estimate_background_offsets(fb, _ones_weight(shape), _identity_tf(), anchor)
    off_a2, _ = estimate_background_offsets(fa, _ones_weight(shape), _identity_tf(), anchor)

    # deterministic: same frame -> same offsets; different frame -> different offsets
    assert np.allclose(off_a1, off_a2, atol=1e-12)
    assert not np.allclose(off_a1, off_b, atol=1e-6)
    # the anchor state never mutates across calls
    assert np.array_equal(anchor.background, bg_before)
    assert anchor.to_metadata() == meta_before


def test_anchor_metadata_roundtrip_scalar_contract():
    shape = (24, 24)
    tf = _translation_tf(0.5, -0.25)
    data = np.stack([
        np.full(shape, 100.0), np.full(shape, 110.0), np.full(shape, 90.0),
    ], -1).astype(np.float32)
    anchor = BackgroundAnchor(data, tf=tf, reference_shape_hw=shape, provenance="reference:frame_001.fit")

    meta = anchor.to_metadata()
    assert meta["version"] == ANCHOR_VERSION
    assert meta["provenance"] == "reference:frame_001.fit"
    assert meta["reference_shape_hw"] == [24, 24]
    assert np.allclose(meta["background_per_channel"], [100.0, 110.0, 90.0], atol=1e-6)

    restored = BackgroundAnchor.from_metadata(meta)
    assert restored.provenance == anchor.provenance
    assert restored.version == anchor.version
    assert restored.reference_shape_hw == anchor.reference_shape_hw
    assert np.allclose(restored.background, anchor.background, atol=1e-6)
    assert np.allclose(restored.tf, anchor.tf, atol=1e-12)
    # the reconstructed anchor keeps its documented shape inspectable even
    # though it carries no pixel data
    assert restored.shape == anchor.shape
    # the reconstructed anchor is a scalar contract only (no pixel data)
    with pytest.raises(RuntimeError):
        restored.sample(np.array([1.0]), np.array([1.0]))


def test_invert_affine_roundtrip():
    tf = np.array([[0.9, -0.1, 3.0], [0.1, 0.9, -2.0]], dtype=np.float64)
    inv = invert_affine_2x3(tf)

    def apply(m, xy):
        return m @ np.array([xy[0], xy[1], 1.0])

    p = np.array([12.0, 7.0])
    q = apply(inv, apply(tf, p))
    assert np.allclose(q, p, atol=1e-9)


def test_sample_bilinear_exact_and_bounds():
    data = np.arange(16, dtype=np.float64).reshape(4, 4, 1)
    # exact pixel centre
    out = sample_bilinear(data, np.array([2.0]), np.array([1.0]))
    assert np.allclose(out[0, 0], data[1, 2, 0])
    # out of bounds -> NaN
    out2 = sample_bilinear(data, np.array([3.5]), np.array([1.0]))
    assert np.isnan(out2[0, 0])


# ---------------------------------------------------------------------------
# WHT relative threshold
# ---------------------------------------------------------------------------


def test_wht_relative_reference_support_and_cutoff():
    wht = np.full((20, 20), 5.0, dtype=np.float32)
    r = wht_relative_threshold(wht, 0.5)
    assert np.isclose(r.reference_support, 5.0, atol=1e-6)
    assert np.isclose(r.cutoff, 2.5, atol=1e-6)
    assert r.mask.all()
    assert r.masked_fraction == 0.0


def test_wht_exposure_scaling_invariant_mask():
    wht = np.full((20, 20), 5.0, dtype=np.float32)
    wht[0:2, 0:2] = 0.5
    r1 = wht_relative_threshold(wht, 0.5)
    r10 = wht_relative_threshold(wht * 10.0, 0.5)
    assert np.isclose(r10.reference_support, r1.reference_support * 10.0, rtol=1e-6)
    assert np.isclose(r10.cutoff, r1.cutoff * 10.0, rtol=1e-6)
    assert np.array_equal(r1.mask, r10.mask)


def test_wht_zeros_always_invalid():
    wht = np.zeros((10, 10), dtype=np.float32)
    wht[2:8, 2:8] = 1.0
    r = wht_relative_threshold(wht, 0.5)
    # zero-weight border pixels are always invalid
    assert not r.mask[0, 0]
    assert r.mask[2:8, 2:8].all()
    assert r.masked_fraction == 0.0


def test_wht_outlier_max_does_not_dominate_reference():
    wht = np.ones((40, 40), dtype=np.float32)
    wht[0, 0] = 10000.0  # a single pathological max
    r = wht_relative_threshold(wht, 0.5)
    # robust upper-tail reference ~ 1.0, not 10000
    assert r.reference_support < 2.0
    assert np.isclose(r.reference_support, 1.0, atol=1e-3)


def test_wht_near_threshold_classification():
    wht = np.ones((30, 30), dtype=np.float32)
    wht[0, 0] = 0.79
    wht[0, 1] = 0.80
    r = wht_relative_threshold(wht, 0.8)
    assert np.isclose(r.reference_support, 1.0, atol=1e-6)
    assert np.isclose(r.cutoff, 0.8, atol=1e-6)
    # exactly at the cutoff is valid; just below is invalid
    assert r.mask[0, 1]
    assert not r.mask[0, 0]


def test_wht_channel_reduction_semantics():
    # 3-D input reduces per-pixel over channels via mean.
    wht = np.zeros((8, 8, 3), dtype=np.float32)
    wht[..., 0] = 2.0
    wht[..., 1] = 4.0
    wht[..., 2] = 0.0
    r = wht_relative_threshold(wht, 0.5)
    # reference support from the mean map (2+4+0)/3 = 2.0
    assert np.isclose(r.reference_support, 2.0, atol=1e-6)
    assert np.isclose(r.cutoff, 1.0, atol=1e-6)


def test_wht_no_positive_weight_reason():
    wht = np.zeros((10, 10), dtype=np.float32)
    r = wht_relative_threshold(wht, 0.5)
    assert r.reason == "no_positive_weight"
    assert r.reference_support == 0.0
    assert not r.mask.any()


# ---------------------------------------------------------------------------
# kernel / pixfrac validation + wiring
# ---------------------------------------------------------------------------


def test_validate_kernel_defaults_and_aliases():
    assert validate_drizzle_kernel("square") == ("square", None)
    assert validate_drizzle_kernel("GAUSSIAN") == ("gaussian", None)
    kernel, reason = validate_drizzle_kernel("bogus")
    assert kernel == "square"
    assert reason is not None


def test_validate_pixfrac_range():
    assert validate_drizzle_pixfrac(1.0) == (1.0, None)
    assert validate_drizzle_pixfrac(0.8) == (0.8, None)
    p, reason = validate_drizzle_pixfrac(np.nan)
    assert p == 1.0 and reason is not None
    p, reason = validate_drizzle_pixfrac(3.0)
    assert p == 1.0 and reason is not None
    p, reason = validate_drizzle_pixfrac("abc")
    assert p == 1.0 and reason is not None


def test_valid_kernel_set_matches_engine():
    # The accepted kernel names are exactly the drizzle engine's documented set.
    assert VALID_DRIZZLE_KERNELS == frozenset(
        {"square", "gaussian", "point", "turbo", "lanczos2", "lanczos3"}
    )


def test_validate_kernel_tophat_gui_name_falls_back():
    # The Qt/Tk GUI and settings still list "tophat", but drizzle 2.2.0 rejects
    # it.  The runtime boundary must coerce it deterministically to "square"
    # (never claim every GUI name is engine-supported).
    kernel, reason = validate_drizzle_kernel("tophat")
    assert kernel == "square"
    assert reason is not None


def test_accumulator_stores_kernel_and_pixfrac(monkeypatch):
    acc = DrizzleAccumulator((16, 16), kernel="gaussian", pixfrac=0.8)
    assert acc.kernel == "gaussian"
    assert acc.pixfrac == 0.8

    # spy the underlying engine add path: pixfrac must reach add_image verbatim
    captured = {}
    orig_add_image = acc._drizzle.add_image

    def spy_add_image(**kwargs):
        captured.update(kwargs)
        return orig_add_image(**kwargs)

    monkeypatch.setattr(acc._drizzle, "add_image", spy_add_image)
    frame = np.ones((16, 16), dtype=np.float32)
    yy, xx = np.indices((16, 16), dtype=np.float64)
    pixmap = np.dstack((xx, yy))
    acc.add(frame, np.ones((16, 16), np.float32), pixmap)
    assert captured.get("pixfrac") == 0.8


# ---------------------------------------------------------------------------
# WHT spatially supported (block) robust maximum (R2.2)
# ---------------------------------------------------------------------------


def test_wht_block_supported_plateau_recovers_small_full_support():
    # 98% background support 20, 2% coherent plateau 100, one isolated 10000.
    shape = (100, 100)
    wht = np.full(shape, 20.0, dtype=np.float32)
    wht[40:54, 40:54] = 100.0  # 14x14 = 196 px (~2%)
    wht[0, 0] = 10000.0
    r = wht_relative_threshold(wht, 0.7)
    # reference near 100, cutoff near 70
    assert r.reference_support == pytest.approx(100.0, rel=0.05)
    assert r.cutoff == pytest.approx(70.0, rel=0.05)
    # WHT=20 masked, plateau kept, outlier does not define the reference
    assert not r.mask[50, 5]  # background support pixel
    assert r.mask[45, 45]  # plateau pixel kept
    assert r.reference_support < 500.0  # not the 10000 outlier
    # scale whole map by 10 -> identical mask
    r10 = wht_relative_threshold(wht * 10.0, 0.7)
    assert np.array_equal(r.mask, r10.mask)


def test_wht_outlier_only_does_not_define_reference():
    # Uniform support plus a single isolated outlier (existing behaviour).
    wht = np.ones((40, 40), dtype=np.float32)
    wht[0, 0] = 10000.0
    r = wht_relative_threshold(wht, 0.5)
    assert np.isclose(r.reference_support, 1.0, atol=1e-3)
    assert r.reference_support < 2.0


def test_wht_spatial_scattered_outliers_do_not_define_reference():
    # >0.5% high outliers (8 px of 1600), spatially scattered so no 8x8 tile
    # holds the minimum supported population (4): the reference must stay at
    # the dense background level (1.0), never at the scattered outliers (100).
    shape = (40, 40)
    wht = np.ones(shape, dtype=np.float32)
    spots = [(4, 4), (4, 20), (4, 36), (20, 4), (20, 20), (20, 36), (36, 4), (36, 20)]
    for y, x in spots:
        wht[y, x] = 100.0
    r = wht_relative_threshold(wht, 0.5)
    assert r.reference_support == pytest.approx(1.0, rel=1e-3)
    assert r.reference_support < 2.0


def test_wht_spatial_compact_cluster_defines_reference():
    # The same count of high pixels (8), compactly clustered -> defines reference.
    shape = (40, 40)
    wht = np.ones(shape, dtype=np.float32)
    wht[20:22, 20:24] = 100.0  # 2x4 compact block (8 px)
    r = wht_relative_threshold(wht, 0.5)
    assert r.reference_support == pytest.approx(100.0, rel=1e-3)


def test_wht_spatial_phase_offsets_resolve_boundary_straddle():
    # A compact 2x2 block of exactly the minimum supported population (4 px)
    # straddling the corner of four base tiles: the base grid splits it
    # 1+1+1+1 (< 4 in every tile), but the half-tile phase offset contains all
    # four -> supported (phase offsets remove boundary dependence).
    shape = (32, 32)
    wht = np.ones(shape, dtype=np.float32)
    wht[7:9, 7:9] = 100.0  # straddles x=8 and y=8 base boundaries
    r = wht_relative_threshold(wht, 0.5)
    assert r.reference_support == pytest.approx(100.0, rel=1e-3)


def test_wht_sparse_positive_layout_meaningful():
    # Sparse-positive (checkerboard) WHT with a locally dense high-coverage
    # region: block support operates on positive pixels within tiles and never
    # requires geometric neighbours to be positive.
    shape = (40, 40)
    yy, xx = np.indices(shape)
    checker = ((yy + xx) % 2 == 0)
    wht = np.where(checker, 10.0, 0.0).astype(np.float32)
    wht[16:24, 16:24] = np.where(checker[16:24, 16:24], 100.0, 0.0).astype(np.float32)
    r = wht_relative_threshold(wht, 0.5)
    assert r.reference_support == pytest.approx(100.0, rel=1e-3)


def test_wht_tiny_footprint_deterministic():
    # A footprint of exactly the minimum supported population is supported.
    wht = np.zeros((6, 6), dtype=np.float32)
    wht[2:4, 2:4] = 7.0  # 4 positive pixels == tile_support_min
    r = wht_relative_threshold(wht, 0.5)
    assert r.reference_support == pytest.approx(7.0, rel=1e-6)
    assert r.reason == "applied"

    # Fewer than the minimum supported population -> deterministic degenerate
    # keep-everything fallback (reference = minimum positive value).
    wht2 = np.zeros((6, 6), dtype=np.float32)
    wht2[2, 2] = 7.0
    wht2[3, 3] = 9.0
    r2 = wht_relative_threshold(wht2, 0.5)
    assert r2.reason == "no_supported_tile"
    assert r2.reference_support == pytest.approx(7.0, rel=1e-6)
    assert r2.mask[2, 2] and r2.mask[3, 3]
    assert r2.masked_fraction == 0.0


# ---------------------------------------------------------------------------
# native WCS measurement mapping (R1.2)
# ---------------------------------------------------------------------------


def _tan_wcs(shape_hw, crval=(10.0, 20.0), cdelt=(-0.001, 0.001), crpix=None):
    h, w = shape_hw
    wcs = WCS(naxis=2)
    if crpix is None:
        crpix = [w / 2.0 + 0.5, h / 2.0 + 0.5]
    wcs.wcs.crpix = list(crpix)
    wcs.wcs.crval = list(crval)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cdelt = np.array(list(cdelt))
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.pc = np.array([[1.0, 0.0], [0.0, 1.0]])
    wcs.array_shape = (h, w)
    return wcs


class _BadWCS:
    def all_pix2world(self, x, y, o):
        raise RuntimeError("pathological WCS")

    def all_world2pix(self, ra, dec, o):
        raise RuntimeError("pathological WCS")


def test_native_wcs_recovers_correct_offset_not_identity():
    shape = (64, 64)
    dx, dy = 10.0, 5.0
    cx, cy = 32.5, 32.5
    ref_wcs = _tan_wcs(shape, crpix=[cx, cy])
    native_wcs = _tan_wcs(shape, crpix=[cx - dx, cy - dy])

    # self-check the mapping direction for a sample pixel
    sx, sy = np.array([5.0]), np.array([7.0])
    rx, ry = native_wcs_to_reference_coords(native_wcs, ref_wcs, sx, sy)
    assert np.allclose(rx, sx + dx, atol=1e-4)
    assert np.allclose(ry, sy + dy, atol=1e-4)

    base = np.array([100.0, 110.0, 90.0])
    off = np.array([4.0, 6.0, -3.0])
    grad = 0.5
    yy, xx = np.indices(shape, dtype=np.float64)
    ref = np.empty((*shape, 3), np.float32)
    for c in range(3):
        ref[..., c] = base[c] + grad * (xx + yy)
    anchor = BackgroundAnchor(ref, reference_shape_hw=shape)

    # frame shows the same sky shifted by (dx, dy) plus an additive offset
    frame = np.empty((*shape, 3), np.float32)
    for c in range(3):
        frame[..., c] = base[c] + off[c] + grad * ((xx + dx) + (yy + dy))

    weight = np.ones(shape, np.float32)

    # WCS correspondence recovers the true offset
    off_wcs, diag_wcs = estimate_background_offsets(
        frame, weight, None, anchor,
        native_wcs=native_wcs, reference_wcs=ref_wcs,
    )
    assert diag_wcs["reason"] == "accepted"
    assert np.allclose(off_wcs, off, atol=0.1)

    # identity geometry would produce the wrong offset (includes the gradient)
    off_id, _ = estimate_background_offsets(frame, weight, _identity_tf(), anchor)
    assert not np.allclose(off_id, off, atol=0.1)
    assert np.allclose(off_id, off + grad * (dx + dy), atol=0.2)


def test_native_wcs_rotated_recovers_offset():
    # A rotated native WCS: identity would compare unrelated sky positions.
    shape = (48, 48)
    ref_wcs = _tan_wcs(shape)
    native_wcs = _tan_wcs(shape)
    # rotate the native PC matrix by 15 degrees
    a = np.radians(15.0)
    pc = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    native_wcs.wcs.pc = pc

    # Constant per-channel backgrounds; with a constant field the offset is
    # identical everywhere, so both identity and WCS mapping agree here.  This
    # test only asserts the WCS path is *accepted* (no fallback) and exact.
    base = np.array([100.0, 110.0, 90.0])
    off = np.array([3.0, 5.0, -2.0])
    anchor = BackgroundAnchor(
        np.stack([np.full(shape, base[c], np.float32) for c in range(3)], -1),
        reference_shape_hw=shape,
    )
    frame = np.stack([np.full(shape, base[c] + off[c], np.float32) for c in range(3)], -1)
    off_wcs, diag_wcs = estimate_background_offsets(
        frame, np.ones(shape, np.float32), None, anchor,
        native_wcs=native_wcs, reference_wcs=ref_wcs,
    )
    assert diag_wcs["reason"] == "accepted"
    assert np.allclose(off_wcs, off, atol=0.1)


def test_invalid_wcs_mapping_neutral_fallback():
    shape = (32, 32)
    anchor = BackgroundAnchor(
        np.full((*shape, 3), 100.0, np.float32), reference_shape_hw=shape
    )
    frame = np.full((*shape, 3), 104.0, np.float32)
    offsets, diag = estimate_background_offsets(
        frame, np.ones(shape, np.float32), None, anchor,
        native_wcs=_BadWCS(), reference_wcs=_tan_wcs(shape),
    )
    assert diag["reason"] == "invalid_wcs"
    assert np.allclose(offsets, 0.0)


def test_degenerate_geometry_reason_emitted():
    shape = (32, 32)
    anchor = BackgroundAnchor(
        np.full((*shape, 3), 100.0, np.float32), reference_shape_hw=shape
    )
    frame = np.full((*shape, 3), 104.0, np.float32)
    weight = np.ones(shape, np.float32)
    # singular affine
    tf_sing = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    off_sing, diag_sing = estimate_background_offsets(frame, weight, tf_sing, anchor)
    assert diag_sing["reason"] == "degenerate_geometry"
    assert np.allclose(off_sing, 0.0)
    # no geometry at all
    off_none, diag_none = estimate_background_offsets(frame, weight, None, anchor)
    assert diag_none["reason"] == "degenerate_geometry"
    assert np.allclose(off_none, 0.0)


# ---------------------------------------------------------------------------
# bounded sampling / memory (R1.6)
# ---------------------------------------------------------------------------


def test_estimator_bounded_sampling_large_shape():
    # 4M pixels with a 250k budget: the estimator must sample, not grid the
    # full frame, and still recover a constant offset accurately.
    shape = (2000, 2000)
    anchor = BackgroundAnchor(
        np.full((*shape, 3), 100.0, np.float32), reference_shape_hw=shape
    )
    frame = np.full((*shape, 3), 104.0, np.float32)
    offsets, diag = estimate_background_offsets(
        frame, np.ones(shape, np.float32), _identity_tf(), anchor
    )
    assert diag["n_candidate"] <= DEFAULT_MAX_SAMPLES
    assert diag["stride"] > 1
    assert diag["reason"] == "accepted"
    assert np.allclose(offsets, 4.0, atol=0.05)


def test_estimator_sampling_deterministic():
    shape = (256, 256)
    anchor = BackgroundAnchor(
        np.full((*shape, 3), 100.0, np.float32), reference_shape_hw=shape
    )
    frame = np.full((*shape, 3), 103.0, np.float32)
    weight = np.ones(shape, np.float32)
    off1, d1 = estimate_background_offsets(
        frame, weight, _identity_tf(), anchor, max_samples=1000
    )
    off2, d2 = estimate_background_offsets(
        frame, weight, _identity_tf(), anchor, max_samples=1000
    )
    assert np.allclose(off1, off2, atol=1e-12)
    assert d1["n_candidate"] == d2["n_candidate"]
    assert d1["n_candidate"] <= 1000


def test_estimator_no_cumulative_state_growth():
    shape = (64, 64)
    anchor = BackgroundAnchor(
        np.full((*shape, 3), 100.0, np.float32), reference_shape_hw=shape
    )
    meta_before = anchor.to_metadata()
    frame = np.full((*shape, 3), 102.0, np.float32)
    for _ in range(50):
        estimate_background_offsets(
            frame, np.ones(shape, np.float32), _identity_tf(), anchor
        )
    # anchor state never accumulates anything across calls
    assert anchor.to_metadata() == meta_before


# ---------------------------------------------------------------------------
# [0,1] -> ADU source-preparation drift guard (R1.3)
# ---------------------------------------------------------------------------


def test_rescale_01_to_adu_matches_source_preparation():
    # The shared helper must reproduce the exact Drizzle/Mosaic source prep:
    # * 65535 when the max is in [0,1], else unchanged; always clipped >= 0.
    in_01 = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    out = rescale_01_to_adu(in_01)
    assert np.allclose(out, in_01 * 65535.0)
    assert out.dtype == np.float32

    # already ADU (max > 1): only clipped, not rescaled
    in_adu = np.array([[-2.0, 0.0], [100.0, 65535.0]], dtype=np.float32)
    out_adu = rescale_01_to_adu(in_adu)
    assert np.array_equal(out_adu, np.clip(in_adu, 0.0, None).astype(np.float32))


def test_anchor_stores_float32_pixels():
    shape = (32, 32)
    anchor = BackgroundAnchor(
        np.full((*shape, 3), 100.0, np.float32), reference_shape_hw=shape
    )
    assert anchor._data.dtype == np.float32


def test_anchor_owns_private_copy_does_not_make_caller_readonly():
    """Constructing the anchor must not alias or freeze the caller's array."""
    data = np.full((16, 16, 3), 100.0, np.float32)
    assert data.flags.writeable
    anchor = BackgroundAnchor(data, reference_shape_hw=(16, 16))

    # caller's array stays writable (not made read-only by setflags(write=False))
    assert data.flags.writeable

    # anchor owns a private copy: mutating the caller does not change the anchor
    data[:] = 999.0
    assert np.all(anchor._data == 100.0)

    # the private copy itself is frozen (immutable anchor contract)
    assert not anchor._data.flags.writeable


# ---------------------------------------------------------------------------
# R2.3 — float32, non-mutating, bounded-memory offset application
# ---------------------------------------------------------------------------


def test_apply_background_offsets_float32_dtype_and_immutability():
    frame = np.full((16, 16, 3), 104.0, np.float32)
    original = frame.copy()
    offsets = np.array([4.0, -2.0, 0.5], np.float64)
    corrected = apply_background_offsets(frame, offsets)

    assert corrected.dtype == np.float32
    # caller input never mutated
    assert np.array_equal(frame, original)
    # numerical correctness (float32 tolerance)
    assert np.allclose(corrected[..., 0], 100.0, atol=1e-3)
    assert np.allclose(corrected[..., 1], 106.0, atol=1e-3)
    assert np.allclose(corrected[..., 2], 103.5, atol=1e-3)


def test_apply_background_offsets_2d():
    frame = np.full((8, 8), 50.0, np.float32)
    corrected = apply_background_offsets(frame, np.array([7.0]))
    assert corrected.dtype == np.float32
    assert np.allclose(corrected, 43.0, atol=1e-3)


def test_apply_background_offsets_no_float64_full_frame():
    # Direct source assertion (supplement to the dtype/immutability behaviour
    # tests above): the correction must not upcast the frame to a full-frame
    # float64 copy; it is performed on a private float32 copy.
    import inspect

    import seestar.core.drizzle_background as dbg

    src = inspect.getsource(dbg.apply_background_offsets)
    assert "np.asarray(frame, dtype=np.float64)" not in src
    assert "np.array(frame, dtype=np.float32" in src
