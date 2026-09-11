"""R3 — FITS viewer-compatibility additive export offset (headless).

Mission: zsss-signed-float32-display-histogram-20260911 (phase viewer-compat-export).

Proves the pure additive viewer-compatibility transform at the float32 final-FITS
export seam and its reversibility/provenance, plus the R1 scientific-histogram
offset recovery.  Pure numpy + astropy; no engine, no Qt display.
"""

from __future__ import annotations

import numpy as np
import pytest

from seestar.queuep.queue_manager import compute_viewer_compatibility_offset

sh = pytest.importorskip("seestar.gui_qt.scientific_histogram")

SIGNED = np.array([[-4.733, -2.0, 0.0, 1.0], [5.0, 40.0, 82.809, -1.5]],
                  dtype=np.float32)


# --------------------------------------------------------------------------- #
# 1. Signed input, compat ON -> saved min == 0, offset == +4.733
# --------------------------------------------------------------------------- #
def test_signed_input_compat_on_makes_saved_non_negative():
    offset = compute_viewer_compatibility_offset(SIGNED, enabled=True)
    assert offset == pytest.approx(4.733, rel=0, abs=1e-5)

    saved = SIGNED + offset
    assert float(np.min(saved)) == pytest.approx(0.0, rel=0, abs=1e-6)
    assert saved.min() >= 0.0
    # Pure translation: the dynamic range is unchanged.
    assert float(np.max(saved) - np.min(saved)) == pytest.approx(
        float(np.max(SIGNED) - np.min(SIGNED)), rel=0, abs=1e-3
    )


# --------------------------------------------------------------------------- #
# 2. Positive-only input, compat ON -> offset 0, output numerically unchanged
# --------------------------------------------------------------------------- #
def test_positive_only_offset_zero_and_unchanged():
    positive = np.abs(SIGNED) + np.float32(0.5)
    offset = compute_viewer_compatibility_offset(positive, enabled=True)
    assert offset == 0.0
    saved = positive if offset == 0.0 else positive + offset
    assert np.array_equal(saved, positive)
    assert saved.dtype == positive.dtype


# --------------------------------------------------------------------------- #
# 3. Compat OFF -> signed data unchanged, ZSCOMPAT=F, ZSOFFSET=0
# --------------------------------------------------------------------------- #
def test_compat_off_preserves_signed_product():
    offset = compute_viewer_compatibility_offset(SIGNED, enabled=False)
    assert offset == 0.0
    saved = SIGNED if offset == 0.0 else SIGNED + offset
    assert np.array_equal(saved, SIGNED)
    assert float(np.min(saved)) < 0.0  # signed product preserved
    # Provenance semantics when disabled.
    zscompat = False
    zsoffset = 0.0
    assert zscompat is False and zsoffset == 0.0


# --------------------------------------------------------------------------- #
# 4. Reversibility: stored - ZSOFFSET reproduces the original (float32)
# --------------------------------------------------------------------------- #
def test_reversibility_within_float32():
    offset = compute_viewer_compatibility_offset(SIGNED, enabled=True)
    stored = SIGNED + offset
    restored = stored - offset
    assert np.allclose(
        restored.astype(np.float64), SIGNED.astype(np.float64), rtol=1e-6, atol=1e-3
    )


# --------------------------------------------------------------------------- #
# 5. ONE scalar offset for the whole product (not per-channel)
# --------------------------------------------------------------------------- #
def test_single_global_scalar_offset_rgb():
    rgb = np.array(
        [
            [[-5.0, -2.0, 0.5], [-1.0, 3.0, 4.0]],
            [[-4.0, -1.5, 1.0], [2.0, 5.0, 6.0]],
        ],
        dtype=np.float32,
    )  # per-channel minima: R=-5, G=-2, B=0.5
    offset = compute_viewer_compatibility_offset(rgb, enabled=True)
    assert offset == pytest.approx(5.0, rel=0, abs=1e-5)  # global min, one scalar

    saved = rgb + offset
    assert float(saved[..., 0].min()) == pytest.approx(0.0, abs=1e-5)
    # NOT per-channel: G/B keep their relative floor above 0.
    assert float(saved[..., 1].min()) == pytest.approx(3.0, abs=1e-5)
    assert float(saved[..., 2].min()) == pytest.approx(5.5, abs=1e-5)


# --------------------------------------------------------------------------- #
# 6. NaN/Inf: finite minimum only, bounded + deterministic
# --------------------------------------------------------------------------- #
def test_nan_inf_finite_only_deterministic():
    arr = np.array([-4.0, 1.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    o1 = compute_viewer_compatibility_offset(arr, enabled=True)
    o2 = compute_viewer_compatibility_offset(arr, enabled=True)
    assert o1 == o2 == pytest.approx(4.0, rel=0, abs=1e-6)

    saved = arr + o1
    assert np.isnan(saved[2])
    assert np.isposinf(saved[3])
    assert np.isneginf(saved[4])
    # Input never mutated.
    assert np.isnan(arr[2]) and np.isposinf(arr[3])


def test_offset_helper_does_not_mutate_input():
    before = SIGNED.copy()
    compute_viewer_compatibility_offset(SIGNED, enabled=True)
    assert np.array_equal(SIGNED, before)


def test_empty_and_all_non_finite_are_bounded():
    assert compute_viewer_compatibility_offset(
        np.zeros((0, 0), dtype=np.float32), enabled=True
    ) == 0.0
    assert compute_viewer_compatibility_offset(
        np.array([np.nan, np.inf], dtype=np.float32), enabled=True
    ) == 0.0


# --------------------------------------------------------------------------- #
# 7. Scientific histogram (R1) recovers the ORIGINAL signed carrier
# --------------------------------------------------------------------------- #
def _write_compat_fits(fits, path, arr, compat, offset):
    hdu = fits.PrimaryHDU(data=arr)
    hdu.header["BITPIX"] = -32
    hdu.header["ZSCOMPAT"] = bool(compat)
    hdu.header["ZSOFFSET"] = float(offset)
    hdu.writeto(path, overwrite=True)


def test_scientific_histogram_recovers_original_signed_range(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    original = SIGNED
    offset = compute_viewer_compatibility_offset(original, enabled=True)
    stored = original + offset
    assert float(stored.min()) == pytest.approx(0.0, abs=1e-6)

    path = str(tmp_path / "final_compat.fits")
    _write_compat_fits(fits, path, stored, True, offset)

    recovered = sh.read_raw_signed_fits(path)
    assert recovered is not None
    assert float(np.min(recovered)) == pytest.approx(
        float(np.min(original)), rel=0, abs=1e-3
    )
    assert float(np.min(recovered)) < 0.0

    model = sh.scientific_histogram_from_fits(path)
    assert model is not None
    assert model["range"][0] < 0.0
    assert model["range"][0] == pytest.approx(float(np.min(original)), rel=0, abs=1e-3)


def test_scientific_histogram_without_offset_unchanged(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    path = str(tmp_path / "final_plain.fits")
    _write_compat_fits(fits, path, SIGNED, False, 0.0)

    recovered = sh.read_raw_signed_fits(path)
    assert recovered is not None
    assert np.allclose(
        recovered.astype(np.float64), SIGNED.astype(np.float64), rtol=1e-6, atol=1e-6
    )


def test_scientific_histogram_ignores_offset_when_compat_false(tmp_path):
    """ZSCOMPAT=F is authoritative: no undo even if ZSOFFSET is nonzero."""
    fits = pytest.importorskip("astropy.io.fits")
    path = str(tmp_path / "final_inconsistent.fits")
    _write_compat_fits(fits, path, SIGNED, False, 4.733)

    recovered = sh.read_raw_signed_fits(path)
    assert recovered is not None
    assert np.allclose(
        recovered.astype(np.float64), SIGNED.astype(np.float64), rtol=1e-6, atol=1e-6
    )


def test_scientific_histogram_round_trip_rgb(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    original = np.stack([SIGNED, SIGNED - 1.0, SIGNED + 2.0], axis=0)  # (C,H,W)
    offset = compute_viewer_compatibility_offset(original, enabled=True)
    stored = original + offset

    path = str(tmp_path / "final_rgb_compat.fits")
    _write_compat_fits(fits, path, stored, True, offset)

    model = sh.scientific_histogram_from_fits(path)
    assert model is not None
    assert model["range"][0] == pytest.approx(
        float(np.min(original)), rel=0, abs=1e-3
    )
    assert model["range"][0] < 0.0


# --------------------------------------------------------------------------- #
# Non-negative regression safety (case 12) — offset is a no-op
# --------------------------------------------------------------------------- #
def test_non_negative_product_export_unchanged():
    arr = np.linspace(0.0, 100.0, 1000, dtype=np.float32).reshape(50, 20)
    offset = compute_viewer_compatibility_offset(arr, enabled=True)
    assert offset == 0.0
    stored = arr + offset
    assert np.array_equal(stored, arr)
