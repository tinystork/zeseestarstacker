"""Deterministic tests for the corrected M16 scale witness helpers.

These pin the *corrected statistics* (defect #1) and the *faithful
preprocessing basis* (defect #2 of the M16 witness: header BAYERPAT + hot-pixel
correction).  They do not require the M16 dataset for the pure-helper tests;
the full-witness integration test is skipped when the data is absent.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "research", "registration_field_rotation"),
)

import m16_scale_witness as w  # noqa: E402


# --------------------------------------------------------------------------
# Defect #1 — the corrected scale statistics
# --------------------------------------------------------------------------


def test_scale_statistics_three_distinct_quantities():
    # skewed example where the three quantities differ, matching Jarvis's point
    scales = np.array([0.9999, 0.99995, 1.0000, 1.0001, 1.0002])
    stats, corner = w.scale_statistics(scales, rmax=1101.5)
    # median scale = 1.0000 -> |median-1| = 0 ppm
    assert abs(stats["abs_median_minus_1"] - 0.0) < 1e-9
    # median(|scale-1|) = median([100,50,0,100,200]) ppm = 100 ppm
    assert abs(stats["median_abs"] - 100.0) < 1e-9
    # mean(|scale-1|) = (100+50+0+100+200)/5 = 90 ppm
    assert abs(stats["mean_abs"] - 90.0) < 1e-9
    # corner displacement = ppm * 1e-6 * rmax
    assert abs(corner["median_abs"] - 100.0e-6 * 1101.5) < 1e-9


def test_scale_statistics_distinguishes_median_offset_from_median_abs():
    # A symmetric-ish distribution: |median-1| and median(|s-1|) are different.
    scales = np.array([0.99999, 0.999995, 1.000005, 1.00001])
    stats, _ = w.scale_statistics(scales, rmax=1101.5)
    assert stats["abs_median_minus_1"] != stats["median_abs"]
    # all three are non-negative
    assert all(v >= 0 for v in stats.values())


# --------------------------------------------------------------------------
# Defect #2 — faithful preparation basis
# --------------------------------------------------------------------------


def test_bayer_pattern_read_from_header():
    from astropy.io import fits

    hdr = fits.Header()
    hdr["BAYERPAT"] = "RGGB"
    assert w._bayer_pattern_from_header(hdr) == "RGGB"
    assert w._bayer_pattern_from_header(fits.Header()) == "GRBG"  # fallback
    assert w._bayer_pattern_from_header(None) == "GRBG"
    # invalid pattern -> fallback
    hdr2 = fits.Header()
    hdr2["BAYERPAT"] = "XYZW"
    assert w._bayer_pattern_from_header(hdr2) == "GRBG"


def test_hot_pixel_correction_replaces_isolated_hot_pixel():
    img = np.full((21, 21), 0.1, dtype=np.float32)
    img[10, 10] = 1.0  # isolated hot pixel (10x brighter than neighbourhood)
    out = w._detect_and_correct_hot_pixels_cpu(img, threshold=3.0, neighborhood_size=5)
    # the hot pixel is pulled back toward the local median (no longer the outlier)
    assert out[10, 10] < 0.5
    # flat background is preserved
    assert np.allclose(out[img != 1.0], 0.1, atol=1e-6)
    # dtype preserved
    assert out.dtype == img.dtype


def test_hot_pixel_correction_color_channels():
    rgb = np.full((21, 21, 3), 0.1, dtype=np.float32)
    rgb[10, 10, 0] = 1.0  # hot pixel only in the red channel
    out = w._detect_and_correct_hot_pixels_cpu(rgb, threshold=3.0, neighborhood_size=5)
    assert out[10, 10, 0] < 0.5
    assert np.allclose(out[10, 10, 1:], 0.1, atol=1e-6)


def test_debayer_matches_grbg_for_synthetic():
    # sanity: debayer of a constant raw image stays bounded in [0,1]
    raw = np.full((16, 16), 0.5, dtype=np.float32)
    rgb = w._debayer(raw, "GRBG")
    assert rgb.shape == (16, 16, 3)
    assert rgb.dtype == np.float32
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


# --------------------------------------------------------------------------
# Full-witness integration (skipped when M16 data is absent)
# --------------------------------------------------------------------------

M16_FOLDER = "/home/tristan/M16/quick"


@pytest.mark.skipif(not os.path.isdir(M16_FOLDER), reason="M16 dataset not present")
def test_full_witness_scale_within_noise():
    data = w.run(M16_FOLDER, seed=0)
    ok = [r for r in data["rows"] if r["status"] == "ok"]
    assert len(ok) == 19, f"expected 19 aligned frames, got {len(ok)}"
    scales = np.array([r["astroalign_scale"] for r in ok])
    stats, corner = w.scale_statistics(scales, data["rmax"])
    # scale is consistent with 1 within noise: median |scale-1| far below the
    # ~0.15 px held-out centroid noise at the frame corner
    assert stats["median_abs"] < 300, f"median |scale-1| {stats['median_abs']:.1f} ppm"
    assert corner["median_abs"] < 0.15, f"corner deviation {corner['median_abs']:.3f} px"
    # held-out improvement from retaining scale is negligible
    imp = np.array(
        [r["euclidean"]["hold_rms"] - r["similarity"]["hold_rms"] for r in ok]
    )
    assert abs(np.median(imp)) < 0.05, f"hold RMS improvement median {np.median(imp):.4f}"
