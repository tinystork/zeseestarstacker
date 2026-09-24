"""CFA-domain hot-pixel correction tests (BEFORE debayer) — rework-2.

Mission ``ZSSS-HOT-PIXEL-SMALL-N-ROBUSTNESS-20260923``.

Covers ``seestar.core.hot_pixels.detect_and_correct_hot_pixels_cfa`` and the
``is_bayer_pattern`` recognition gate.  Rework-2 semantics:

* SAME-COLOR replacement only (cross-phase neighbours never used for the
  replacement value);
* conservative isolation/coherent-PSF gate (a candidate is corrected only if
  its immediate full-resolution neighbourhood is NOT bright — a healthy star
  core is never corrected);
* border band (kernel radius) never corrected (no edge candidates).
"""

from __future__ import annotations

import numpy as np
import pytest

from seestar.core.hot_pixels import (
    detect_and_correct_hot_pixels_cfa,
    is_bayer_pattern,
)


def _img(shape=(16, 16), bg=100.0):
    return np.full(shape, float(bg), dtype=np.float32)


def _psf_cfa(size, fwhm, amp, bg, dy, dx, noise_sigma=0.0, noise_seed=0):
    """Deterministic Gaussian PSF sampled on the CFA grid, plus optional
    fixed-seed Gaussian noise (order-independent: seed derived per case)."""
    img = np.full((size, size), float(bg), dtype=np.float32)
    cy, cx = (size - 1) / 2.0 + dy, (size - 1) / 2.0 + dx
    sigma = fwhm / 2.355
    for y in range(size):
        for x in range(size):
            py, px = y + 0.5, x + 0.5
            r2 = (py - cy) ** 2 + (px - cx) ** 2
            img[y, x] = bg + amp * np.exp(-r2 / (2.0 * sigma * sigma))
    if noise_sigma > 0:
        rng = np.random.default_rng(noise_seed)
        img = img + rng.normal(0.0, noise_sigma, size=(size, size)).astype(np.float32)
    return img


# ---------------------------------------------------------------------------
# Recognition gate
# ---------------------------------------------------------------------------
def test_is_bayer_pattern_recognition():
    assert is_bayer_pattern("RGGB") is True
    assert is_bayer_pattern("rggb") is True
    assert is_bayer_pattern("GRBG") is True
    assert is_bayer_pattern("GBRG") is True
    assert is_bayer_pattern("BGGR") is True
    assert is_bayer_pattern("MONO") is False
    assert is_bayer_pattern("") is False
    assert is_bayer_pattern(None) is False
    assert is_bayer_pattern(42) is False


# ---------------------------------------------------------------------------
# Isolated defect detection / correction
# ---------------------------------------------------------------------------
def test_isolated_hot_pixel_corrected():
    img = _img()
    img[6, 6] = 63000.0
    corr, diag = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)
    assert diag["enabled"] is True
    assert diag["pattern"] == "RGGB"
    assert diag["candidates"] == 1
    assert diag["corrected"] == 1
    assert corr[6, 6] == pytest.approx(100.0, abs=1.0)


def test_multiple_isolated_hot_pixels_corrected():
    img = _img((16, 16))
    img[4, 4] = 63000.0
    img[4, 8] = 50000.0
    img[8, 8] = 63000.0
    corr, diag = detect_and_correct_hot_pixels_cfa(img, "GRBG", threshold=3.0)
    assert diag["corrected"] >= 3
    assert corr[4, 4] < 200.0
    assert corr[4, 8] < 200.0
    assert corr[8, 8] < 200.0


def test_lower_amplitude_isolated_corrected():
    img = _img()
    img[6, 6] = 5000.0  # 50x background, clearly isolated (above the ~16x star spike)
    corr, diag = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)
    assert diag["corrected"] == 1
    assert corr[6, 6] == pytest.approx(100.0, abs=1.0)


def test_near_saturation_hot_pixel():
    img = _img()
    img[6, 6] = 65000.0
    corr, diag = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)
    assert diag["corrected"] == 1
    assert corr[6, 6] < 200.0


def test_low_background_hot_pixel():
    img = _img(bg=1.0)
    img[6, 6] = 60000.0
    corr, diag = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)
    assert diag["corrected"] == 1
    assert corr[6, 6] < 10.0


# ---------------------------------------------------------------------------
# Spike-factor boundary regression (production factor 18)
# ---------------------------------------------------------------------------
def test_isolated_19x_neighbour_ratio_corrected():
    """A candidate ~19x its brightest immediate neighbour is corrected.

    The production spike factor is 18: a photosite 19x above its immediate
    full-resolution neighbourhood is a genuine isolated defect (no coherent
    PSF is that peaked) and must be corrected.  This is the behavioral pin
    that distinguishes factor 18 from factor 20 (at 20x this photosite would
    NOT be corrected).  The photosite also satisfies the same-color candidate
    gate (bright relative to its own colour plane).
    """
    img = _img()  # 16x16, bg=100.0
    img[6, 6] = 19.0 * 100.0  # 19x the background / immediate neighbours
    corr, diag = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)
    assert diag["corrected"] == 1
    assert corr[6, 6] == pytest.approx(100.0, abs=1.0)


def test_isolated_at_18x_not_corrected_strict():
    """A candidate exactly at the 18x boundary is NOT corrected (strict >).

    The spike test uses a STRICT ``img > factor * near_max`` comparison, so an
    isolated photosite exactly 18x its brightest immediate neighbour is NOT a
    spike.  This protects the conservative boundary: the factor must stay
    strictly above the ~16x coherent-PSF peak, and nothing AT the boundary is
    corrected.
    """
    img = _img()
    img[6, 6] = 18.0 * 100.0  # exactly at the boundary
    corr, diag = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)
    assert diag["corrected"] == 0
    assert corr[6, 6] == pytest.approx(18.0 * 100.0)


# ---------------------------------------------------------------------------
# No cross-color replacement
# ---------------------------------------------------------------------------
def test_no_cross_color_neighbours():
    """A hot photosite is replaced with its own color plane only; the
    orthogonally adjacent (cross-color) photosites are never used/touched."""
    img = _img((16, 16))
    img[6, 6] = 63000.0
    corr, _ = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)
    assert corr[6, 6] == pytest.approx(100.0, abs=1.0)  # same-color R median
    assert corr[6, 5] == pytest.approx(100.0)  # cross-color untouched
    assert corr[6, 7] == pytest.approx(100.0)
    assert corr[5, 6] == pytest.approx(100.0)
    assert corr[7, 6] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Healthy structures are NEVER corrected
# ---------------------------------------------------------------------------
def test_clean_background_untouched():
    img = _img()
    corr, diag = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)
    assert diag["corrected"] == 0
    np.testing.assert_array_equal(corr, img)


def test_fixed_noise_background_no_false_correction():
    # Deterministic "fixed" noise (no random tail): no false correction.
    y, x = np.mgrid[0:16, 0:16]
    img = _img() + (np.sin(x) + np.cos(y)).astype(np.float32)
    corr, diag = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)
    assert diag["corrected"] == 0


def test_bright_star_preserved():
    """A bright but spatially coherent star must NOT be corrected."""
    img = _img((16, 16))
    for y in range(16):
        for x in range(16):
            r2 = (y - 8) ** 2 + (x - 8) ** 2
            img[y, x] = 100.0 + 2000.0 * np.exp(-r2 / 4.0)
    corr, diag = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)
    assert diag["corrected"] == 0
    assert corr[8, 8] > 1000.0


def test_gradient_preserved():
    img = np.fromfunction(
        lambda y, x: 100.0 + 5.0 * (x + y), (16, 16), dtype=np.float32
    )
    corr, diag = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)
    assert diag["corrected"] == 0


# ---------------------------------------------------------------------------
# Healthy PSF matrix (the R1 defect-1 witness)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fwhm", [1.0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0])
def test_psf_matrix_corrected_zero(fwhm):
    """A healthy Gaussian PSF (amplitude +10000, bg 100, 33x33) with
    fixed-seed Gaussian noise (sigma 2) is never corrected, across 4 subpixel
    offsets x 4 CFA phase placements; output is exactly unchanged."""
    offsets = [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.25, 0.75)]
    phases = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    for i, off in enumerate(offsets):
        for j, ph in enumerate(phases):
            # Stable per-case seed (order-independent, deterministic).
            seed = int(round(fwhm * 100)) * 10000 + i * 100 + j
            img = _psf_cfa(
                33, fwhm, 10000.0, 100.0, off[0] + ph[1], off[1] + ph[0],
                noise_sigma=2.0, noise_seed=seed,
            )
            corr, diag = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)
            assert diag["corrected"] == 0, (fwhm, off, ph)
            # Exact output unchanged for every photosite.
            np.testing.assert_array_equal(corr, img)


# ---------------------------------------------------------------------------
# Border band + phase-coded parity
# ---------------------------------------------------------------------------
def _phase_coded(size, pattern="RGGB"):
    """R/G1/G2/B = 100/200/300/400 coded CFA mosaic (RGGB layout)."""
    values = {
        "RGGB": {(0, 0): 100.0, (0, 1): 200.0, (1, 0): 300.0, (1, 1): 400.0},
        "GRBG": {(0, 0): 200.0, (0, 1): 100.0, (1, 0): 400.0, (1, 1): 300.0},
        "GBRG": {(0, 0): 300.0, (0, 1): 400.0, (1, 0): 100.0, (1, 1): 200.0},
        "BGGR": {(0, 0): 400.0, (0, 1): 300.0, (1, 0): 200.0, (1, 1): 100.0},
    }[pattern]
    img = np.zeros((size, size), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            img[y, x] = values[(y % 2, x % 2)]
    return img


def test_border_phase_parity_no_cross_phase_replacement():
    """Phase-coded R/G1/G2/B=100/200/300/400: a corrected photosite is
    replaced with its OWN phase value, and border photosites are never
    corrected (no reflect-mode cross-phase mixing)."""
    img = _phase_coded(12, "RGGB")
    # interior R (6,6): hot -> corrected to R value 100.
    img[6, 6] = 63000.0
    # interior B (9,9): hot -> corrected to B value 400 (separated, not
    # adjacent to (6,6), so each is an isolated defect).
    img[9, 9] = 63000.0
    # corners + edges: hot -> NOT corrected (border band).
    img[0, 0] = 63000.0
    img[0, 11] = 63000.0
    img[11, 0] = 63000.0
    img[11, 11] = 63000.0
    img[0, 5] = 63000.0
    img[5, 0] = 63000.0

    corr, diag = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)

    assert corr[6, 6] == pytest.approx(100.0, abs=1.0)  # R -> R value (not 200/300/400)
    assert corr[9, 9] == pytest.approx(400.0, abs=1.0)  # B -> B value
    # border hot pixels unchanged (no edge correction candidates).
    assert corr[0, 0] == 63000.0
    assert corr[0, 11] == 63000.0
    assert corr[11, 0] == 63000.0
    assert corr[11, 11] == 63000.0
    assert corr[0, 5] == 63000.0
    assert corr[5, 0] == 63000.0
    # Only the two interior photosites corrected.
    assert diag["corrected"] == 2


def test_border_hot_pixels_not_corrected():
    """A hot photosite inside the kernel-radius border band is never a
    correction candidate."""
    for pos in [(0, 0), (1, 1), (0, 7), (7, 0), (15, 15), (15, 8), (8, 15)]:
        img = _img((16, 16))
        img[pos] = 63000.0
        corr, diag = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)
        assert diag["corrected"] == 0, pos
        assert corr[pos] == 63000.0


# ---------------------------------------------------------------------------
# Conservative ambiguity: hot near / over stellar environment
# ---------------------------------------------------------------------------
def test_hot_near_star_conservatively_uncorrected():
    """A hot photosite adjacent to a bright star core is conservatively NOT
    corrected (it is not a clear spike relative to the star), while the star
    itself is preserved."""
    img = _img((16, 16))
    for y in range(16):
        for x in range(16):
            r2 = (y - 8) ** 2 + (x - 8) ** 2
            img[y, x] = 100.0 + 10000.0 * np.exp(-r2 / 4.0)
    img[8, 9] = 63000.0  # hot photosite adjacent to the bright star core
    corr, diag = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)
    # Conservative: the near-star hot photosite is not a clear spike (only ~6x
    # the star core), so it is not corrected.
    assert corr[8, 9] == pytest.approx(63000.0)
    assert corr[8, 8] > 1000.0  # star core preserved


# ---------------------------------------------------------------------------
# dtype / contract
# ---------------------------------------------------------------------------
def test_dtype_preserved_float32():
    img = _img()
    img[6, 6] = 63000.0
    corr, _ = detect_and_correct_hot_pixels_cfa(img, "RGGB")
    assert corr.dtype == np.float32
    assert corr.shape == img.shape


def test_integer_dtype_roundtrip():
    img = np.full((16, 16), 100, dtype=np.int32)
    img[6, 6] = 60000
    corr, diag = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)
    assert corr.dtype == np.int32
    assert diag["corrected"] == 1
    assert corr[6, 6] == 100


def test_no_valid_pattern_raises():
    img = _img()
    with pytest.raises(ValueError):
        detect_and_correct_hot_pixels_cfa(img, "MONO")


def test_2d_only():
    img = np.full((16, 16, 3), 100.0, dtype=np.float32)
    with pytest.raises(ValueError):
        detect_and_correct_hot_pixels_cfa(img, "RGGB")


# ---------------------------------------------------------------------------
# R3: positive-multiplicative-scaling invariance (normalized witnesses)
# ---------------------------------------------------------------------------

_CFA_SCALES = [1.0, 1e-3, 1e-6, 1e3, 65535.0]


def test_normalized_cfa_witness_detects_isolated_defect():
    """Explicit normalized witness (R3 defect): an isolated .998 photosite
    over a .04 background is detected and corrected to the background (the
    pre-fix code corrected 0 photosites because of the absolute near_max
    unit floor)."""
    img = np.full((16, 16), 0.04, dtype=np.float32)
    img[6, 6] = 0.998
    corr, diag = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)
    assert diag["corrected"] == 1
    assert corr[6, 6] == pytest.approx(0.04, abs=1e-4)


@pytest.mark.parametrize("scale", _CFA_SCALES)
def test_cfa_scale_invariance_normalized_structure(scale):
    """bg=.04 / hot=.998 at scales 1, 1e-3, 1e-6, 1e3, 65535: identical
    classification (exactly one correction at the same photosite) and
    identical correction semantics (corrected to the scaled same-color
    median)."""
    bg = 0.04 * scale
    img = np.full((16, 16), bg, dtype=np.float32)
    img[6, 6] = 0.998 * scale
    corr, diag = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)
    assert diag["corrected"] == 1
    expected = np.full((16, 16), bg, dtype=np.float32)
    np.testing.assert_array_equal(corr, expected)


def test_cfa_scale_invariance_mask_identical_across_scales():
    """The correction mask (which photosites are corrected) is identical
    across every scale."""
    masks = []
    for scale in _CFA_SCALES:
        img = np.full((16, 16), 0.04 * scale, dtype=np.float32)
        img[6, 6] = 0.998 * scale
        corr, _ = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)
        masks.append(corr != img)
    for m in masks[1:]:
        np.testing.assert_array_equal(m, masks[0])
    assert int(np.count_nonzero(masks[0])) == 1


@pytest.mark.parametrize("fwhm", [1.0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0])
@pytest.mark.parametrize("scale", [1e-3, 65535.0])
def test_psf_matrix_corrected_zero_scaled(fwhm, scale):
    """A healthy Gaussian PSF is never corrected at numeric scales 1e-3 and
    65535 (scale 1.0 is covered by ``test_psf_matrix_corrected_zero``) across
    all phases/offsets; output exactly unchanged."""
    offsets = [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.25, 0.75)]
    phases = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    for i, off in enumerate(offsets):
        for j, ph in enumerate(phases):
            seed = int(round(fwhm * 100)) * 10000 + i * 100 + j
            img = _psf_cfa(
                33, fwhm, 10000.0 * scale, 100.0 * scale,
                off[0] + ph[1], off[1] + ph[0],
                noise_sigma=2.0 * scale, noise_seed=seed,
            )
            corr, diag = detect_and_correct_hot_pixels_cfa(img, "RGGB", threshold=3.0)
            assert diag["corrected"] == 0, (fwhm, scale, off, ph)
            np.testing.assert_array_equal(corr, img)
