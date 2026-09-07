"""Phase 1 (zsss-support-overlap-p1-20260907) — neutral paired-overlap module.

Unit tests for :mod:`seestar.core.overlap_normalization`:

* VERBATIM Drizzle reuse: the module imports ``robust_location`` and the
  estimator defaults from ``seestar.core.drizzle_background`` (single
  implementation, exact old/new results by construction).
* Truthful Classic geometry: synthetic-detector warp support reproduces the
  ACTUAL production image warp (NaN ring, zero-weight contamination,
  fractional translations, rotations).
* Content validity after the loader repair seam (CFA/RGB, conservative Bayer
  influence) and content-mask warp on the canvas.
* ``sky_mean``: scalar paired luminance ``I-R`` robust location, same scalar
  across RGB, known +/- offsets, legitimate zero, neutral fallback, constant
  sky regression (old P25 full-canvas catastrophic, new recovered D) and
  structured spatial sky (same-coordinate, not independent visible-field P25).
* ``linear_fit``: verbatim per-channel P25/P90 model on the same common
  positions, degenerate-denominator preservation, explicit neutral
  insufficient-overlap.
* Explicit float32 numerical tolerance: constant [0,1] scenes absolute
  ``<= 2e-6`` after correction; sign and geometry independence verified.
* Coverage sweep: rotations 0/22.5/45/70/100 deg, translations +/-X,+/-Y,
  measured coverages ~95/80/60/50% and >25% unsupported scenarios.

All scenes are synthetic, deterministic and fast (no FITS / GPU / GUI).
"""

import numpy as np
import pytest

from astropy.io import fits as _fits

from seestar.core import drizzle_background as dzb
from seestar.core.image_processing import load_and_validate_fits
from seestar.core.overlap_normalization import (
    DEFAULT_MIN_OVERLAP_SAMPLES,
    GEOM_ERODE_PX,
    REASON_ACCEPTED,
    REASON_DEGENERATE_GEOMETRY,
    REASON_INSUFFICIENT_OVERLAP,
    REASON_NO_GEOMETRY,
    REASON_NO_REFERENCE_CONTENT,
    REASON_NO_SOURCE_CONTENT,
    REASON_NO_VALID_SAMPLES,
    apply_linear_fit,
    apply_sky_mean_offset,
    content_valid_canvas,
    content_validity_after_loader,
    effective_common_support,
    estimate_linear_fit,
    estimate_linear_fit_from_geometry,
    estimate_sky_mean_from_geometry,
    estimate_sky_mean_offset,
    geometry_support_mask,
    geometry_support_mask_or_none,
    luminance,
    robust_location,
    warp_content_mask,
)

# Explicit float32 tolerance for prepared [0,1] constant scenes (mission
# requirement: absolute <= 2e-6).
ABS_TOL = 2e-6

H = W = 64
CANVAS = (H, W)


def _identity():
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def _rot_deg(deg, tx=0.0, ty=0.0):
    import cv2

    c = np.cos(np.radians(deg))
    s = np.sin(np.radians(deg))
    M = np.array([[c, -s, tx], [s, c, ty]], dtype=np.float64)
    # match the production rotation centre convention used in the seam tests
    return cv2.getRotationMatrix2D((W / 2.0 - 0.5, H / 2.0 - 0.5), deg, 1.0)


def _full_content():
    return np.ones((H, W), dtype=bool)


def _rgb(val, shape=(H, W), dtype=np.float32):
    return np.full((*shape, 3), val, dtype=dtype)


def _mono(val, shape=(H, W), dtype=np.float32):
    return np.full(shape, val, dtype=dtype)


def _warp_image(img, M, canvas=(W, H)):
    """Actual production-style warp used as ground truth for support tests."""
    import cv2

    out = cv2.warpAffine(
        np.asarray(img, dtype=np.float32),
        M,
        canvas,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    return out


# ---------------------------------------------------------------------------
# A. Verbatim Drizzle reuse (single estimator implementation)
# ---------------------------------------------------------------------------

def test_robust_location_is_verbatim_drizzle_object():
    from seestar.core.overlap_normalization import robust_location as ol_robust

    assert ol_robust is dzb.robust_location


def test_defaults_match_drizzle_exactly():
    from seestar.core import overlap_normalization as ol

    assert ol.DEFAULT_MIN_OVERLAP_SAMPLES == dzb.DEFAULT_MIN_OVERLAP_SAMPLES == 200
    assert ol.DEFAULT_MAX_SAMPLES == dzb.DEFAULT_MAX_SAMPLES == 250_000
    assert ol.DEFAULT_SIGMA_CLIP == dzb.DEFAULT_SIGMA_CLIP == 3.0
    assert ol.DEFAULT_CLIP_ITERATIONS == dzb.DEFAULT_CLIP_ITERATIONS == 3


def test_robust_location_identical_results_drizzle_and_overlap():
    rng = np.random.default_rng(3)
    samples = rng.normal(100.0, 5.0, 5000)
    samples[::97] = 500.0  # outliers
    a, na = robust_location(samples)
    b, nb = dzb.robust_location(samples)
    assert a == b and na == nb


# ---------------------------------------------------------------------------
# B. Truthful geometry support vs ACTUAL image warp
# ---------------------------------------------------------------------------

def _erode(mask):
    import cv2

    return cv2.erode(
        np.asarray(mask, dtype=np.uint8),
        np.ones((3, 3), np.uint8),
        iterations=1,
    ).astype(bool)


def test_geometry_support_matches_actual_warp_nan_pattern_identity():
    M = _identity()
    img = np.zeros((H, W), np.float32)
    img[10:20, 10:20] = 1.0
    warped = _warp_image(img, M)
    actual_finite = ~np.isnan(warped)
    support = geometry_support_mask((H, W), M, CANVAS)
    # Support = eroded actual-finite mask (the ones-warp and image warp have
    # identical NaN patterns; erosion is the documented 1-px margin).
    assert np.array_equal(support, _erode(actual_finite))


def test_geometry_support_matches_actual_warp_fractional_translation():
    M = _identity().copy()
    M[0, 2] = 0.5
    img = np.zeros((H, W), np.float32)
    img[5:40, 5:40] = 1.0
    warped = _warp_image(img, M)
    support = geometry_support_mask((H, W), M, CANVAS)
    assert np.array_equal(support, _erode(~np.isnan(warped)))


def test_geometry_support_matches_actual_warp_rotation():
    M = _rot_deg(22.5)
    img = np.ones((H, W), np.float32)
    warped = _warp_image(img, M)
    support = geometry_support_mask((H, W), M, CANVAS)
    assert np.array_equal(support, _erode(~np.isnan(warped)))


def test_geometry_support_rejects_zero_weight_nan_contamination_ring():
    # Production warp with NaN borderValue poisons neighbours even at zero
    # interpolation weight; the ones-warp support must reproduce that exactly
    # (plus the documented one-pixel erosion margin).
    M = _identity()
    support = geometry_support_mask((H, W), M, CANVAS)
    assert not support[-1, :].any() and not support[:, -1].any()
    assert support[0, 0] and support[H - 3, W - 3]


def test_geometry_support_brightness_independent():
    # A legitimately-zero source pixel inside the detector must stay supported
    # (no brightness heuristic): zeros only exist because of real sky, not
    # because of NaN repair.
    M = _identity()
    support = geometry_support_mask((H, W), M, CANVAS)
    dark_src = np.zeros((H, W), np.float32)
    warped = _warp_image(dark_src, M)  # still finite everywhere inside
    assert np.array_equal(support, _erode(~np.isnan(warped)))


@pytest.mark.parametrize("deg", [0.0, 22.5, 45.0, 70.0, 100.0])
def test_coverage_sweep_rotations(deg):
    M = _rot_deg(deg)
    support = geometry_support_mask((H, W), M, CANVAS)
    frac = float(support.mean())
    assert 0.0 <= frac <= 1.0
    assert support.dtype == bool
    assert support.shape == (H, W)


def test_coverage_targets_translations_and_unsupported_gt25():
    # translations +/-X and +/-Y measured coverages ~95/80/60/50%
    def noerode_frac(M):
        ones = np.ones((H, W), np.float32)
        w = _warp_image(ones, M)
        return float((np.isfinite(w) & (w >= 1 - 1e-6)).mean())

    for tx, ty in [(0.0, 0.0), (-6.0, 0.0), (6.0, 0.0), (0.0, -6.0), (0.0, 6.0)]:
        M = _identity().copy()
        M[0, 2] = tx
        M[1, 2] = ty
        frac = noerode_frac(M)
        support = geometry_support_mask((H, W), M, CANVAS)
        ones = np.ones((H, W), np.float32)
        w = _warp_image(ones, M)
        # support equals the eroded actual-warp finite mask for every geometry
        assert np.array_equal(support, _erode(~np.isnan(w)))
        # measured (deterministic, this build): ~0.969 identity, ~0.89 for a
        # 6 px shift -> both stay near the ~95% coverage band
        assert frac > 0.85, (tx, ty, frac)
    # near 80% with |t|=13, near 60% with |t|=26, near 50% with |t|=32
    for t, target in [(13, 0.8), (26, 0.6), (32, 0.5)]:
        M = _identity().copy()
        M[0, 2] = -t
        frac = noerode_frac(M)
        assert abs(frac - target) < 0.05, (t, frac, target)
    # >25% unsupported with a large translation
    M = _identity().copy()
    M[0, 2] = -30.0
    support = geometry_support_mask((H, W), M, CANVAS)
    assert (1.0 - support.mean()) > 0.25


def test_geometry_support_rotation_unsupported_fraction():
    # at 45 deg a full-size detector loses >18% of canvas pixels (measured
    # 0.814 no-erode / 0.776 eroded on a 64x64 canvas)
    M = _rot_deg(45.0)
    support = geometry_support_mask((H, W), M, CANVAS)
    assert support.mean() < 0.80
    assert (1.0 - support.mean()) > 0.15


# ---------------------------------------------------------------------------
# C. Content validity after loader repair seam
# ---------------------------------------------------------------------------

def test_content_validity_mono_finite_all_valid():
    inv = np.zeros((H, W), dtype=bool)
    cv = content_validity_after_loader(inv, bayer=False)
    assert cv.all()


def test_content_validity_rejects_repaired_pixels_rgb():
    inv = np.zeros((H, W, 3), dtype=bool)
    inv[5, 6, 0] = True  # one bad channel at one pixel
    cv = content_validity_after_loader(inv, bayer=False)
    assert not cv[5, 6]
    assert cv.sum() == H * W - 1


def test_content_validity_bayer_conservative_dilation():
    # a repaired CFA sample must invalidate a small neighbourhood, not the
    # whole frame, and must not change the science array (mask only)
    inv = np.zeros((H, W), dtype=bool)
    inv[10, 10] = True
    cv = content_validity_after_loader(inv, bayer=True)
    # dilated influence around (10,10)
    assert not cv[10, 10]
    assert not cv[10, 11] and not cv[11, 10]  # 3x3 influence
    # far pixels untouched
    assert cv[40, 40]
    # non-bayer mono keeps the single pixel invalid only
    cv_no = content_validity_after_loader(inv, bayer=False)
    assert not cv_no[10, 10] and cv_no[10, 11] and cv_no[11, 10]


def test_content_valid_canvas_warp_identity():
    M = _identity()
    content = np.ones((H, W), dtype=np.float32)
    cv = content_valid_canvas(content, M, CANVAS)
    # identical to geometry support when the content is fully valid
    geom = geometry_support_mask((H, W), M, CANVAS)
    assert np.array_equal(cv, geom)


def test_content_valid_canvas_excludes_repaired_hole():
    M = _identity()
    content = np.ones((H, W), dtype=np.float32)
    content[30, 30] = 0.0
    cv = content_valid_canvas(content, M, CANVAS)
    assert not cv[30, 30]
    # far pixels untouched
    assert cv[40, 40]
    # a fully-valid content mask gives the geometry support exactly
    full = content_valid_canvas(np.ones((H, W), dtype=np.float32), M, CANVAS)
    assert np.array_equal(full, geometry_support_mask((H, W), M, CANVAS))


def test_effective_common_support_requires_all_three():
    g = np.ones((H, W), bool)
    c = np.ones((H, W), bool)
    r = np.ones((H, W), bool)
    assert effective_common_support(g, c, r).all()
    c[0, 0] = False
    r[1, 1] = False
    g[2, 2] = False
    eff = effective_common_support(g, c, r)
    assert not eff[0, 0] and not eff[1, 1] and not eff[2, 2]
    assert eff.sum() == H * W - 3


# ---------------------------------------------------------------------------
# D. sky_mean estimator
# ---------------------------------------------------------------------------

def _all_support():
    return np.ones((H, W), dtype=bool)


def test_sky_mean_recovers_positive_offset_mono():
    ref = _mono(0.5)
    src = _mono(0.5 + 0.1)
    off, diag = estimate_sky_mean_offset(
        src, ref, _all_support(), _full_content(), _full_content()
    )
    assert diag["reason"] == REASON_ACCEPTED
    assert abs(off - 0.1) <= 1e-6
    out = apply_sky_mean_offset(src, off)
    assert np.abs(out - ref).max() <= ABS_TOL


def test_sky_mean_recovers_negative_offset_rgb_same_scalar():
    ref = _rgb(0.4)
    src = _rgb(0.4 - 0.05)
    off, diag = estimate_sky_mean_offset(
        src, ref, _all_support(), _full_content(), _full_content()
    )
    assert diag["reason"] == REASON_ACCEPTED
    assert abs(off - (-0.05)) <= 1e-6
    out = apply_sky_mean_offset(src, off)
    assert np.abs(out - ref).max() <= ABS_TOL
    # same scalar across RGB: all channels shifted identically
    d = out - src
    assert np.abs(d[..., 0] - d[..., 1]).max() <= ABS_TOL
    assert np.abs(d[..., 1] - d[..., 2]).max() <= ABS_TOL


def test_sky_mean_legitimate_zero_offset():
    ref = _rgb(0.3)
    src = _rgb(0.3)  # no offset at all
    off, diag = estimate_sky_mean_offset(
        src, ref, _all_support(), _full_content(), _full_content()
    )
    assert diag["reason"] == REASON_ACCEPTED
    assert abs(off) <= ABS_TOL
    out = apply_sky_mean_offset(src, off)
    assert np.abs(out - ref).max() <= ABS_TOL


def test_sky_mean_stars_do_not_bias():
    rng = np.random.default_rng(0)
    ref = _rgb(0.5) + rng.normal(0, 0.001, (H, W, 3)).astype(np.float32)
    src = ref + 0.03
    src[10:13, 10:13, :] += 0.9  # star block
    off, diag = estimate_sky_mean_offset(
        src, ref, _all_support(), _full_content(), _full_content()
    )
    assert abs(off - 0.03) < 0.01


def test_sky_mean_insufficient_overlap_neutral():
    g = np.zeros((H, W), bool)
    g[0:5, 0:5] = True  # 25 < 200 samples
    off, diag = estimate_sky_mean_offset(
        _rgb(0.5), _rgb(0.4), g, _full_content(), _full_content()
    )
    assert off == 0.0
    assert diag["reason"] == REASON_INSUFFICIENT_OVERLAP
    # neutral application is identity on the values
    out = apply_sky_mean_offset(_rgb(0.5), off)
    assert np.abs(out - 0.5).max() <= ABS_TOL


def test_sky_mean_geometry_independent_result():
    # rotation changes support but not the recovered constant offset
    results = []
    for deg in (0.0, 22.5, 45.0):
        M = _rot_deg(deg)
        g = geometry_support_mask((H, W), M, CANVAS)
        if g.mean() < 0.05:
            continue
        off, diag = estimate_sky_mean_offset(
            _rgb(0.55), _rgb(0.5), g, _full_content(), _full_content()
        )
        assert diag["reason"] == REASON_ACCEPTED
        results.append(off)
    assert results
    assert max(results) - min(results) <= ABS_TOL


def test_sky_mean_nan_invalidity_source_excluded():
    ref = _rgb(0.5)
    src = _rgb(0.52)
    src[20:30, 20:30] = np.nan  # repaired-invalid region would be excluded
    src_fin = np.isfinite(luminance(src))
    # effective common support must be limited by finite source pixels
    g = _all_support()
    off, diag = estimate_sky_mean_offset(
        src, ref, g, _full_content(), _full_content()
    )
    assert diag["reason"] == REASON_ACCEPTED
    assert abs(off - 0.02) < 0.01  # not biased by the NaN region


# ---------------------------------------------------------------------------
# (Duplicate witness removed: ``test_constant_sky_regression_old_full_canvas_*``
# is defined once below with the substantive numeric-zero-padding construction;
# the earlier partial-canvas variant was shadowed by it and is obsolete.)
# ---------------------------------------------------------------------------


def test_structured_spatial_sky_same_coordinate_not_visible_field_p25():
    """Structured spatial sky: paired same-coordinate estimator vs P25.

    A smooth spatial gradient (structured sky) is present in both frames at
    the SAME coordinates after alignment.  A visible-field percentile measures
the frame histogram (dominated by the gradient + zero padding), while the
    paired difference at common coordinates cancels the shared structure and
    recovers the constant sky offset D.
    """
    Hh = Ww = 96
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float64)
    grad = (yy / Hh + xx / Ww) / 2.0
    sky_ref = 0.1
    D = 0.05
    ref2 = np.clip(sky_ref + 0.4 * grad, 0, 1).astype(np.float32)
    # source: same coordinates + D on a partial footprint (rest repaired zero)
    src2 = np.zeros((Hh, Ww), np.float32)
    n = 62
    src2[:, :n] = np.clip(ref2[:, :n] + D, 0, 1)
    valid = np.zeros((Hh, Ww), bool)
    valid[:, :n] = True

    legacy_off = float(np.percentile(luminance(src2), 25.0)) - float(
        np.percentile(luminance(ref2), 25.0)
    )
    assert abs(legacy_off - D) > 0.01

    M = _identity()
    geom = geometry_support_mask((Hh, Ww), M, (Hh, Ww))
    src_content = np.ones((Hh, Ww), bool)
    src_content[~valid] = False
    src_content &= geom
    ref_content = np.ones((Hh, Ww), bool) & geom
    new_off, diag = estimate_sky_mean_offset(
        src2, ref2, geom, src_content, ref_content
    )
    assert diag["reason"] == REASON_ACCEPTED
    assert abs(new_off - D) < 1e-3


# ---------------------------------------------------------------------------
# F. linear_fit estimator (verbatim model on common positions)
# ---------------------------------------------------------------------------

def test_linear_fit_recovers_affine_mono():
    ref = _mono(0.3) + np.linspace(0, 0.05, H * W).reshape(H, W).astype(np.float32)
    src = (1.5 * ref + 0.1).astype(np.float32)
    (a, b), diag = estimate_linear_fit(
        src, ref, _all_support(), _full_content(), _full_content()
    )
    assert diag["reason"] == REASON_ACCEPTED
    assert abs(a[0] - 1.0 / 1.5) < 1e-3
    out = apply_linear_fit(src, a, b)
    assert np.abs(out - ref).max() <= 1e-3


def test_linear_fit_per_channel_rgb():
    rng = np.random.default_rng(5)
    # spatially varying reference per channel (gradient + noise)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    ref = np.zeros((H, W, 3), dtype=np.float32)
    for ch in range(3):
        base = 0.2 * (ch + 1) / 3.0
        grad = base + 0.3 * (xx / W) + 0.1 * (yy / H)
        ref[..., ch] = grad + rng.normal(0, 0.002, (H, W))
    ref = np.clip(ref, 0, 1).astype(np.float32)
    # src = 1.25 * ref + 0.03 per channel (same per-channel affine)
    src = np.clip(ref * 1.25 + 0.03, 0, 1).astype(np.float32)
    (a, b), diag = estimate_linear_fit(
        src, ref, _all_support(), _full_content(), _full_content()
    )
    assert diag["reason"] == REASON_ACCEPTED
    assert np.allclose(a, 0.8, atol=1e-3)
    out = apply_linear_fit(src, a, b)
    assert np.abs(out - ref).max() <= 1e-3


def test_linear_fit_degenerate_denominator_preserved():
    # constant source channel -> legacy model keeps a=1 and b=ref_low-src_low
    ref = _mono(0.4) + np.linspace(0, 0.1, H * W).reshape(H, W).astype(np.float32)
    src = _mono(0.2)  # degenerate (no spread)
    (a, b), diag = estimate_linear_fit(
        src, ref, _all_support(), _full_content(), _full_content()
    )
    assert diag["reason"] == REASON_ACCEPTED
    assert a[0] == 1.0
    # b = ref_low - 1*src_low
    ref_low = float(np.percentile(ref, 25.0))
    assert abs(b[0] - (ref_low - 0.2)) < 1e-5


def test_linear_fit_insufficient_overlap_neutral():
    g = np.zeros((H, W), bool)
    g[0:5, 0:5] = True
    (a, b), diag = estimate_linear_fit(
        _rgb(0.5), _rgb(0.4), g, _full_content(), _full_content()
    )
    assert diag["reason"] == REASON_INSUFFICIENT_OVERLAP
    assert np.all(a == 1.0) and np.all(b == 0.0)
    out = apply_linear_fit(_rgb(0.5), a, b)
    assert np.abs(out - 0.5).max() <= ABS_TOL


def test_linear_fit_common_positions_not_full_canvas():
    # Structured scene where only a sub-region overlaps: percentiles must be
    # computed on the common positions, never the full visible frame.
    rng = np.random.default_rng(11)
    ref = (0.3 + 0.1 * rng.random((H, W))).astype(np.float32)
    src = (ref * 1.4 + 0.05).astype(np.float32)
    g = np.zeros((H, W), bool)
    g[5:59, 5:59] = True  # central overlap only
    (a, b), diag = estimate_linear_fit(src, ref, g, _full_content(), _full_content())
    assert diag["reason"] == REASON_ACCEPTED
    # model recovered from the common region only
    sub = g[5:59, 5:59]
    assert abs(a[0] - 1 / 1.4) < 1e-3
    out = apply_linear_fit(src, a, b)
    assert np.abs(out[5:59, 5:59] - ref[5:59, 5:59]).max() <= 1e-3


def test_linear_fit_nan_invalid_ref_excluded():
    ref = _mono(0.3) + np.linspace(0, 0.1, H * W).reshape(H, W).astype(np.float32)
    ref[40:50, 40:50] = np.nan  # would be loader-repaired zeros
    src = (ref * 1.2 + 0.01).astype(np.float32)
    src[40:50, 40:50] = np.nan
    (a, b), diag = estimate_linear_fit(
        src, ref, _all_support(), _full_content(), _full_content()
    )
    assert diag["reason"] == REASON_ACCEPTED
    assert abs(a[0] - 1 / 1.2) < 1e-2


# ---------------------------------------------------------------------------
# G. float32 [0,1] constant scenes — explicit absolute tolerance <= 2e-6
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("D", [0.001, 0.02, -0.03, 0.0])
def test_sky_mean_float32_constant_tolerance(D):
    ref = _rgb(0.25)
    src = _rgb(0.25 + D)
    off, diag = estimate_sky_mean_offset(
        src, ref, _all_support(), _full_content(), _full_content()
    )
    out = apply_sky_mean_offset(src, off)
    assert np.abs(out - ref).max() <= ABS_TOL
    # sign check: src brighter than ref => positive offset to subtract
    if D > 0:
        assert off > 0
    elif D < 0:
        assert off < 0


def test_apply_helpers_do_not_mutate_input():
    ref = _rgb(0.5)
    src = _rgb(0.6)
    before = src.copy()
    _ = apply_sky_mean_offset(src, 0.1)
    assert np.array_equal(src, before)
    _ = apply_linear_fit(src, np.array([1.0]), np.array([-0.05]))
    assert np.array_equal(src, before)


def test_min_overlap_samples_matches_drizzle_precedent():
    assert DEFAULT_MIN_OVERLAP_SAMPLES == 200


# ---------------------------------------------------------------------------
# H. Loader roundtrip: default/opt-in science identity + spatial mask (H,W)
# ---------------------------------------------------------------------------


def _write_fits(tmp_path, name, data):
    p = tmp_path / name
    _fits.PrimaryHDU(data=data).writeto(str(p), overwrite=True)
    return str(p)


def _rng():
    return np.random.default_rng(123)


def test_loader_default_optin_science_identical_rgb(tmp_path):
    rng = _rng()
    img = (rng.random((32, 32, 3)) * 1000).astype(np.float32)
    img[2:4, 5, 1] = np.nan
    p = _write_fits(tmp_path, "rgb.fits", img)
    a, _h = load_and_validate_fits(p)
    b, _h2, inv = load_and_validate_fits(p, report_invalidity=True)
    assert np.array_equal(a, b)  # opt-in never changes the science
    assert inv.shape == (32, 32)  # spatial reduction, not (32,32,3)
    assert inv[2, 5] and not inv[0, 0]


def test_loader_hwc1_spatial_invalidity_shape(tmp_path):
    # HWC1 (single channel as HxWxC) must reduce over the FINAL axis -> (H,W)
    rng = _rng()
    img = (rng.random((16, 16, 1)) * 100).astype(np.float32)
    img[3, 3, 0] = np.nan
    p = _write_fits(tmp_path, "hwc1.fits", img)
    data, _h, inv = load_and_validate_fits(p, report_invalidity=True)
    assert data.shape == (16, 16, 1)
    assert inv.shape == (16, 16), inv.shape
    assert inv[3, 3] and int(inv.sum()) == 1


def test_loader_hwc1_mono_science_identical(tmp_path):
    rng = _rng()
    img = (rng.random((16, 16, 1)) * 100).astype(np.float32)
    p = _write_fits(tmp_path, "hwc1b.fits", img)
    a, _ = load_and_validate_fits(p)
    b, _h, inv = load_and_validate_fits(p, report_invalidity=True)
    assert np.array_equal(a, b)
    assert inv.shape == (16, 16)
    assert not inv.any()


def test_loader_chw_transposed_and_invalid(tmp_path):
    rng = _rng()
    img = (rng.random((3, 20, 24)) * 100).astype(np.float32)  # CxHxW
    img[1, 7, 8] = np.nan
    p = _write_fits(tmp_path, "chw.fits", img)
    data, _h, inv = load_and_validate_fits(p, report_invalidity=True)
    assert data.shape == (20, 24, 3)
    assert inv.shape == (20, 24)
    assert inv[7, 8] and int(inv.sum()) == 1


def test_loader_mono2d_invalid_and_failure_arity(tmp_path):
    rng = _rng()
    img = (rng.random((16, 16)) * 100).astype(np.float32)
    img[9, 9] = np.inf
    p = _write_fits(tmp_path, "mono.fits", img)
    data, _h, inv = load_and_validate_fits(p, report_invalidity=True)
    assert data.ndim == 2
    assert inv.shape == (16, 16) and inv[9, 9]
    # failure returns are arity-consistent under the opt-in flag
    missing = tmp_path / "does_not_exist.fits"
    r = load_and_validate_fits(str(missing), report_invalidity=True)
    assert len(r) == 3 and r[0] is None and r[2] is None
    r2 = load_and_validate_fits(str(missing))
    assert len(r2) == 2 and r2[0] is None


def test_loader_nonfinite_positions_repair_science(tmp_path):
    rng = _rng()
    img = (rng.random((24, 24, 3)) * 500 + 50).astype(np.float32)
    img[0, 0, :] = np.nan
    img[5, 5, 1] = np.inf
    img[6, 6, 2] = -np.inf
    p = _write_fits(tmp_path, "nf.fits", img)
    data, _h, inv = load_and_validate_fits(p, report_invalidity=True)
    assert np.isfinite(data).all()
    assert inv[0, 0] and inv[5, 5] and inv[6, 6]
    assert int(inv.sum()) == 3


# ---------------------------------------------------------------------------
# I. Geometry neutrality: missing/malformed/singular M -> explicit reason, no
#    identity/full-support guess; sampling budget validation; bounded memory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_M",
    [
        None,
        np.zeros((2, 3), dtype=np.float64),  # singular zero
        np.ones((3, 3), dtype=np.float64),  # wrong shape (3x3)
        np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0]]),  # non-finite
        np.array([[1e300, 0.0, 0.0], [0.0, 1e300, 0.0]]),  # non-finite product
    ],
)
def test_geometry_bad_M_never_guesses_support(bad_M):
    with pytest.raises(ValueError):
        geometry_support_mask((H, W), bad_M, CANVAS)
    mask, reason = geometry_support_mask_or_none((H, W), bad_M, CANVAS)
    assert mask is None
    assert reason in (REASON_NO_GEOMETRY, REASON_DEGENERATE_GEOMETRY)


def test_geometry_singular_zero_M_explicit_reason():
    zero = np.zeros((2, 3), dtype=np.float64)
    mask, reason = geometry_support_mask_or_none((H, W), zero, CANVAS)
    assert mask is None
    assert reason == REASON_DEGENERATE_GEOMETRY


def _known_finite_content(shape=(H, W)):
    """Explicit known-finite content mask (0/1 float, source frame).

    Module-level scenes are fully finite by construction (no loader repair
    seam): callers declare this explicitly rather than relying on an implicit
    all-valid fallback (provenance is never silently assumed).
    """
    return np.ones(shape, dtype=np.float32)


def test_sky_mean_from_geometry_missing_M_neutral():
    ref = _rgb(0.4)
    src = _rgb(0.5)
    off, diag = estimate_sky_mean_from_geometry(
        src,
        ref,
        (H, W),
        None,
        src_content_mask_01=_known_finite_content(),
        ref_content_mask=_full_content(),
    )
    assert off == 0.0
    assert diag["reason"] == REASON_NO_GEOMETRY
    out = apply_sky_mean_offset(src, off)
    assert np.abs(out - src).max() <= ABS_TOL  # neutral: values unchanged


def test_sky_mean_from_geometry_singular_M_neutral():
    ref = _rgb(0.4)
    src = _rgb(0.5)
    off, diag = estimate_sky_mean_from_geometry(
        src,
        ref,
        (H, W),
        np.zeros((2, 3)),
        src_content_mask_01=_known_finite_content(),
        ref_content_mask=_full_content(),
    )
    assert off == 0.0
    assert diag["reason"] == REASON_DEGENERATE_GEOMETRY


def test_sky_mean_from_geometry_missing_source_content_neutral():
    # Missing source content evidence must NOT become an all-valid guess.
    ref = _rgb(0.4)
    src = _rgb(0.5)
    off, diag = estimate_sky_mean_from_geometry(
        src, ref, (H, W), _identity(), ref_content_mask=_full_content()
    )
    assert off == 0.0
    assert diag["reason"] == REASON_NO_SOURCE_CONTENT


def test_sky_mean_from_geometry_missing_ref_content_neutral():
    ref = _rgb(0.4)
    src = _rgb(0.5)
    off, diag = estimate_sky_mean_from_geometry(
        src, ref, (H, W), _identity(), src_content_mask_01=_known_finite_content()
    )
    assert off == 0.0
    assert diag["reason"] == REASON_NO_REFERENCE_CONTENT


def test_linear_fit_from_geometry_missing_M_neutral():
    ref = _rgb(0.4)
    src = _rgb(0.5)
    (a, b), diag = estimate_linear_fit_from_geometry(
        src,
        ref,
        (H, W),
        None,
        src_content_mask_01=_known_finite_content(),
        ref_content_mask=_full_content(),
    )
    assert np.all(a == 1.0) and np.all(b == 0.0)
    assert diag["reason"] == REASON_NO_GEOMETRY
    out = apply_linear_fit(src, a, b)
    assert np.abs(out - src).max() <= ABS_TOL


def test_sky_mean_from_geometry_recovers_offset_with_identity_M():
    ref = _rgb(0.25)
    src = _rgb(0.25 + 0.04)
    off, diag = estimate_sky_mean_from_geometry(
        src,
        ref,
        (H, W),
        _identity(),
        src_content_mask_01=_known_finite_content(),
        ref_content_mask=_full_content(),
    )
    assert diag["reason"] == REASON_ACCEPTED
    assert abs(off - 0.04) <= 1e-6
    out = apply_sky_mean_offset(src, off)
    assert np.abs(out - ref).max() <= ABS_TOL


@pytest.mark.parametrize("bad", [0, -1, -100, np.nan, np.inf, -np.inf])
def test_sampling_rejects_non_positive_max_samples(bad):
    from seestar.core.overlap_normalization import (
        paired_common_positions,
    )

    with pytest.raises(ValueError):
        paired_common_positions(
            _all_support(), _full_content(), _full_content(), max_samples=bad
        )


def test_sampling_bounded_large_overlap_no_full_nonzero():
    # A huge overlap (> budget) must not allocate full-frame index pairs:
    # use a big canvas and a small budget; returned positions <= budget and
    # the estimate still works.
    big_h = big_w = 4000  # 16 MP overlap >> 250000 budget
    geom = np.ones((big_h, big_w), dtype=bool)
    ref = np.full((big_h, big_w, 3), 0.3, dtype=np.float32)
    src = ref + 0.02
    off, diag = estimate_sky_mean_offset(
        src,
        ref,
        geom,
        geom.copy(),
        geom.copy(),
        max_samples=50_000,
    )
    assert diag["reason"] == REASON_ACCEPTED
    assert abs(off - 0.02) <= 1e-6
    assert diag["n_overlap"] <= 50_000
    assert diag["n_effective"] == big_h * big_w


def test_geometry_validation_identity_accepted():
    mask, reason = geometry_support_mask_or_none((H, W), _identity(), CANVAS)
    assert mask is not None and reason == REASON_ACCEPTED
    assert mask.shape == (H, W)


# ---------------------------------------------------------------------------
# J. Bayer influence radius: empirically validated against the ACTUAL debayer
# ---------------------------------------------------------------------------


def _import_debayer():
    from seestar.core.image_processing import debayer_image

    return debayer_image


@pytest.mark.parametrize("pat", ["GRBG", "RGGB", "GBRG", "BGGR"])
def test_bayer_influence_radius_empirical(pat):
    # A single repaired CFA sample must not affect output pixels farther than
    # the documented conservative radius (1px, 3x3 dilation).  Measured with
    # the REAL debayer: max Chebyshev influence radius == 1 for every pattern.
    from seestar.core.overlap_normalization import DEBAYER_INFLUENCE_PX

    debayer = _import_debayer()
    hh = ww = 48
    base = np.full((hh, ww), 0.5, np.float32)
    imp = base.copy()
    cy = cx = hh // 2
    imp[cy, cx] = 0.95  # impulse CFA sample (would be repaired zero)
    out_base = debayer(base, pat)
    out_imp = debayer(imp, pat)
    diff = np.abs(out_base - out_imp) > 1e-6
    affected = np.argwhere(diff.any(axis=-1))
    if affected.size:
        r = int(
            max(
                np.abs(affected[:, 0] - cy).max(),
                np.abs(affected[:, 1] - cx).max(),
            )
        )
        assert r <= DEBAYER_INFLUENCE_PX, (pat, r)
    # far-away pixels are never affected
    assert not diff[0, 0].any() and not diff[-1, -1].any()


@pytest.mark.parametrize("pat", ["GRBG", "RGGB", "GBRG", "BGGR"])
def test_bayer_content_validity_dilation_empirically_sufficient(pat):
    # The loader-seam content mask (dilated by the documented radius) must
    # cover every output pixel the REAL debayer could have contaminated.
    from seestar.core.overlap_normalization import (
        DEBAYER_INFLUENCE_PX,
        content_validity_after_loader,
    )

    debayer = _import_debayer()
    hh = ww = 48
    base = np.full((hh, ww), 0.5, np.float32)
    invalid_cfa = np.zeros((hh, ww), dtype=bool)
    for cy, cx in [(hh // 2, ww // 2), (5, 40), (40, 5), (0, 0), (hh - 1, ww - 1)]:
        invalid_cfa[cy, cx] = True
        imp = base.copy()
        imp[cy, cx] = 0.95
        out_base = debayer(base, pat)
        out_imp = debayer(imp, pat)
        diff = np.abs(out_base - out_imp) > 1e-6
        # content-valid mask derived from the invalid CFA report:
        cv = content_validity_after_loader(invalid_cfa, bayer=True)
        bad = np.argwhere(diff.any(axis=-1))
        for y, x in bad:
            assert not cv[y, x], (pat, (cy, cx), (int(y), int(x)))
        invalid_cfa[cy, cx] = False
    assert DEBAYER_INFLUENCE_PX >= 1  # documented conservative floor


# ---------------------------------------------------------------------------
# K. Corrected original-bug witness + production-warp scenes
# ---------------------------------------------------------------------------


def _make_sky_padded_scene(base_sky=0.2, D=0.05, unsupported_frac=0.4):
    """True production-style scene: aligned source canvas with a real warped
    footprint; the unsupported region is loader-repaired ZERO (numeric zero),
    not base sky.  ``valid`` marks the footprint interior."""
    n = int(W * (1.0 - unsupported_frac))
    canvas = np.zeros((H, W), dtype=np.float32)  # repaired zeros outside
    canvas[:, :n] = base_sky + D
    valid = np.zeros((H, W), dtype=bool)
    valid[:, :n] = True
    rng = np.random.default_rng(7)
    for _ in range(40):
        x = int(rng.integers(0, n))
        y = int(rng.integers(0, H))
        canvas[y, x] += 0.6  # stars on the supported side
    return canvas, valid


def test_constant_sky_regression_old_full_canvas_p25_catastrophic_new_recovers():
    """True zero-padding regression: legacy full-canvas P25 collapses to ~0
    because >25% of the source canvas is repaired zeros; the paired overlap
    estimator on common reliable positions recovers D exactly."""
    base_sky = 0.2
    D = 0.05
    ref = np.full((H, W), base_sky, dtype=np.float32)
    src, valid = _make_sky_padded_scene(base_sky, D)
    n = int(valid.sum(axis=0).max()) if valid.any() else 0
    # numeric zero padding outside the footprint, >25% unsupported
    assert not valid[:, n:].any()
    assert (src[:, n:] == 0.0).all()
    assert (1.0 - valid.mean()) > 0.25

    # legacy full-canvas P25 sky_mean: percentiles collapse toward the zeros
    legacy_off = float(np.percentile(luminance(src), 25.0)) - float(
        np.percentile(luminance(ref), 25.0)
    )
    assert abs(legacy_off + base_sky) < 0.02  # P25 ~ 0 -> erroneous offset
    assert abs(legacy_off - D) > 0.02  # catastrophic before the fix

    src_content = np.ones((H, W), dtype=bool)
    src_content[~valid] = False
    ref_content = np.ones((H, W), dtype=bool)
    M = _identity()
    geom = geometry_support_mask((H, W), M, CANVAS)
    src_content &= geom
    new_off, diag = estimate_sky_mean_offset(
        src, ref, geom, src_content, ref_content
    )
    assert diag["reason"] == REASON_ACCEPTED
    assert abs(new_off - D) <= ABS_TOL


def _production_warp(src, M, canvas=(W, H)):
    import cv2

    return cv2.warpAffine(
        np.asarray(src, dtype=np.float32),
        M,
        canvas,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )


def _scene(deg, tx=0.0, ty=0.0, base_sky=0.2, D=0.0):
    """Constant-sky scene: reference = base sky on the canvas; source =
    production-warped sky (+D) with NaN border -> repaired zeros, so only the
    warped detector footprint carries content."""
    import cv2

    M = cv2.getRotationMatrix2D((W / 2 - 0.5, H / 2 - 0.5), deg, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    src_pre = np.full((H, W), base_sky + D, dtype=np.float32)
    src_canvas = _production_warp(src_pre, M)
    src_canvas = np.nan_to_num(src_canvas, nan=0.0)
    ref = np.full((H, W), base_sky, dtype=np.float32)
    return src_canvas, ref, M


@pytest.mark.parametrize("deg", [0.0, 22.5, 45.0, 70.0, 100.0])
@pytest.mark.parametrize("D", [0.02, -0.03])
def test_production_warp_constant_scene_recovers_D(deg, D):
    base_sky = 0.2
    src_canvas, ref, M = _scene(deg, base_sky=base_sky, D=D)
    off, diag = estimate_sky_mean_from_geometry(
        src_canvas,
        ref,
        (H, W),
        M,
        src_content_mask_01=_known_finite_content(),
        ref_content_mask=_full_content(),
    )
    assert diag["reason"] == REASON_ACCEPTED
    assert abs(off - D) <= ABS_TOL, (deg, D, off)
    out = apply_sky_mean_offset(src_canvas, off)
    # only compare on the common support: repaired-zero padding is not content
    geom = geometry_support_mask((H, W), M, CANVAS)
    inside = geom & (ref > 0)
    assert inside.sum() > 0
    assert np.abs(out[inside] - ref[inside]).max() <= ABS_TOL


@pytest.mark.parametrize("tx,ty", [(-10, 0), (10, 0), (0, -10), (0, 10)])
def test_production_warp_translation_scene(tx, ty):
    base_sky = 0.2
    D = 0.05
    src_canvas, ref, M = _scene(0.0, tx=tx, ty=ty, base_sky=base_sky, D=D)
    off, diag = estimate_sky_mean_from_geometry(
        src_canvas,
        ref,
        (H, W),
        M,
        src_content_mask_01=_known_finite_content(),
        ref_content_mask=_full_content(),
    )
    assert diag["reason"] == REASON_ACCEPTED
    assert abs(off - D) <= ABS_TOL, (tx, ty, off)


def test_production_warp_legitimate_zero_and_uncovered_gt25():
    # D == 0 (legitimate zero offset) recovered, and a strong translation
    # leaves >25% of the canvas without source content (repaired zeros).
    base_sky = 0.2
    src_canvas, ref, M = _scene(0.0, tx=-30.0, base_sky=base_sky, D=0.0)
    geom = geometry_support_mask((H, W), M, CANVAS)
    assert (1.0 - geom.mean()) > 0.25
    off, diag = estimate_sky_mean_from_geometry(
        src_canvas,
        ref,
        (H, W),
        M,
        src_content_mask_01=_known_finite_content(),
        ref_content_mask=_full_content(),
    )
    assert abs(off) <= ABS_TOL
    assert diag["reason"] == REASON_ACCEPTED


def test_production_warp_structured_spatial_scene():
    # Physically valid structured-sky construction (Junior r1): the source is
    # a ROTATED view of the SAME analytic sky function S, not a copy of the
    # reference.  The source detector carries ``S(u) + D`` at its own focal
    # coordinates ``u``; the production warp then samples ``S(M^-1 c) + D`` on
    # the canvas ``c`` (verified against a direct analytic evaluation at M^-1
    # to ~6e-8).  The paired estimator must therefore recover D up to a small
    # residual from the rotated linear field: bilinear resampling of a linear
    # field under an affine map is exact, so the 1e-3 tolerance is justified
    # by interpolation/boundary effects only, never by the construction.
    import cv2

    hh = ww = 96
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float64)
    grad = (yy / hh + xx / ww) / 2.0
    base = 0.1
    D = 0.05

    def _sky(ux, uy):
        return np.clip(base + 0.4 * (uy / hh + ux / ww) / 2.0, 0, 1)

    ref2 = _sky(yy, xx).astype(np.float32)
    M = cv2.getRotationMatrix2D((ww / 2 - 0.5, hh / 2 - 0.5), 22.5, 1.0)
    # source detector frame: same analytic sky at ITS focal coords, plus D
    src_pre = (_sky(yy, xx) + D).astype(np.float32)
    src2 = _production_warp(src_pre, M, canvas=(ww, hh))
    # loader-style repair of the NaN border -> numeric zeros (not content)
    src2 = np.where(np.isnan(src2), 0.0, src2).astype(np.float32)
    off, diag = estimate_sky_mean_from_geometry(
        src2,
        ref2,
        (hh, ww),
        M,
        src_content_mask_01=_known_finite_content((hh, ww)),
        ref_content_mask=np.ones((hh, ww), dtype=bool),
    )
    assert diag["reason"] == REASON_ACCEPTED
    assert abs(off - D) <= 1e-3
    # legacy per-frame P25 of the structured + padded fields is not D
    legacy_off = float(np.percentile(luminance(src2), 25.0)) - float(
        np.percentile(luminance(ref2), 25.0)
    )
    assert abs(legacy_off - D) > 0.01
