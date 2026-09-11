"""R1 — FINAL SCIENTIFIC vs DISPLAY histogram separation (pure numpy, headless).

Mission: zsss-signed-float32-display-histogram-20260911 (phase r1).

Covers:

* (a) truthful signed min/max/stats for a signed float32 sample (min < 0,
  max > 0, no clip);
* (b) bounded deterministic sampling (an input larger than the sample cap still
  yields a capped, deterministic result — and the bounded SAMPLE is retained
  for R2 re-binning);
* (c) the display-domain ``compute_histogram_float`` is bit-identical to the
  ratified display algorithm (unchanged by R1);
* (d) the display-vs-scientific distinction is represented (domain tag + label).

No Qt window/display is required.
"""

from __future__ import annotations

import numpy as np
import pytest

sh = pytest.importorskip("seestar.gui_qt.scientific_histogram")
pa = pytest.importorskip("seestar.gui_qt.preview_analysis")

SIGNED = np.float32([[-4.733, -2.0, 0.0, 1.0], [5.0, 40.0, 82.809, -1.5]])


# --------------------------------------------------------------------------- #
# (a) truthful signed min/max/stats
# --------------------------------------------------------------------------- #
def test_scientific_histogram_truthful_signed_minmax():
    model = sh.compute_scientific_histogram(SIGNED)

    assert model is not None
    assert model["domain"] == "scientific"
    lo, hi = model["range"]
    assert lo < 0.0, f"negative floor lost: {lo!r}"
    assert hi > 0.0
    assert lo == pytest.approx(float(np.min(SIGNED)), rel=0, abs=1e-6)
    assert hi == pytest.approx(float(np.max(SIGNED)), rel=0, abs=1e-6)

    st = model["stats"]["L"]
    assert st["min"] == pytest.approx(float(np.min(SIGNED)), rel=0, abs=1e-6)
    assert st["max"] == pytest.approx(float(np.max(SIGNED)), rel=0, abs=1e-6)
    assert st["min"] < 0.0 and st["max"] > 0.0
    # Bounded sample drives the in-domain statistics.
    assert st["median"] == pytest.approx(float(np.median(SIGNED)), rel=0, abs=1e-6)


def test_scientific_histogram_no_clip_counts_all_pixels():
    model = sh.compute_scientific_histogram(SIGNED)
    total = int(sum(int(np.sum(model["counts"][ch])) for ch in model["channels"]))
    # Every finite pixel is binned inside the signed carrier range.
    assert total == int(np.isfinite(SIGNED).sum())
    # Bin edges span the signed domain (negatives included).
    edges = model["edges"]["L"]
    assert edges[0] == pytest.approx(model["range"][0], rel=0, abs=1e-6)
    assert edges[-1] == pytest.approx(model["range"][1], rel=0, abs=1e-6)
    assert edges[0] < 0.0


def test_scientific_histogram_rgb_hwc_and_chw_layouts():
    hwc = np.stack(
        [
            SIGNED,
            SIGNED + np.float32(1.0),
            SIGNED - np.float32(1.0),
        ],
        axis=-1,
    )
    model_hwc = sh.compute_scientific_histogram(hwc)
    assert model_hwc["channels"] == ["R", "G", "B"]

    chw = np.moveaxis(hwc, -1, 0)  # (C, H, W)
    model_chw = sh.compute_scientific_histogram(chw)
    assert model_chw["channels"] == ["R", "G", "B"]
    # Same carrier -> same truthful global range regardless of axis order.
    assert model_hwc["range"] == model_chw["range"]
    for ch in ("R", "G", "B"):
        assert np.array_equal(model_hwc["counts"][ch], model_chw["counts"][ch])


# --------------------------------------------------------------------------- #
# (b) bounded deterministic sampling (+ retained sample for R2)
# --------------------------------------------------------------------------- #
def test_scientific_histogram_bounded_sampling_deterministic():
    rng = np.random.default_rng(3)
    n = pa.MAX_SAMPLE_PIXELS + 400_000  # > cap
    big = rng.normal(-2.0, 5.0, size=n).astype(np.float32).reshape(-1, 1000)

    model1 = sh.compute_scientific_histogram(big)
    model2 = sh.compute_scientific_histogram(big)

    sample = model1["samples"]["L"]
    assert sample.size <= pa.MAX_SAMPLE_PIXELS
    assert model1["sample_pixels"] <= pa.MAX_SAMPLE_PIXELS
    assert model1["total_finite"] == n
    # Deterministic: the exact capped sample is reproducible.
    assert np.array_equal(model1["samples"]["L"], model2["samples"]["L"])
    assert np.array_equal(model1["counts"]["L"], model2["counts"]["L"])


def test_scientific_histogram_retains_sample_for_r2_rebinning():
    model = sh.compute_scientific_histogram(SIGNED)
    sample = model["samples"]["L"]
    assert isinstance(sample, np.ndarray) and sample.ndim == 1
    # R2 can re-bin a narrow visible sub-range from the retained sample alone.
    narrow = sample[(sample >= -1.0) & (sample <= 2.0)]
    rebinned, _ = np.histogram(narrow, bins=64, range=(-1.0, 2.0))
    assert int(rebinned.sum()) == int(narrow.size)


def test_scientific_histogram_fail_closed_shapes():
    assert sh.compute_scientific_histogram(np.zeros((0, 0), dtype=np.float32)) is None
    assert sh.compute_scientific_histogram(np.full((4, 4), np.nan, dtype=np.float32)) is None
    # A 4-D array is not a supported carrier shape.
    assert sh.compute_scientific_histogram(np.zeros((2, 2, 5, 1), dtype=np.float32)) is None


# --------------------------------------------------------------------------- #
# FITS carrier read (raw signed, no BSCALE/BZERO rescale)
# --------------------------------------------------------------------------- #
def test_read_raw_signed_fits_preserves_negative_floor(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    arr = SIGNED.astype(np.float32)
    path = str(tmp_path / "final.fits")
    fits.PrimaryHDU(data=arr).writeto(path, overwrite=True)

    model = sh.scientific_histogram_from_fits(path)
    assert model is not None
    assert model["domain"] == "scientific"
    assert model["source"] == "final_fits"
    assert model["range"][0] == pytest.approx(float(np.min(arr)), rel=0, abs=1e-6)
    assert model["range"][0] < 0.0

    assert sh.scientific_histogram_from_fits(str(tmp_path / "missing.fits")) is None


# --------------------------------------------------------------------------- #
# (c) display-domain compute_histogram_float unchanged (ratified algorithm)
# --------------------------------------------------------------------------- #
def test_display_histogram_float_bit_identical_to_ratified_algorithm():
    arr = np.array([[0.0, 0.1], [0.25, 0.5]], dtype=np.float32)
    model = pa.compute_histogram_float(arr)

    # Independently re-derive the ratified display model for this dense/small
    # non-negative input (sample <= 512 -> bin high == analysis upper).  The
    # module casts to float64 first, so the reference uses float64 too.
    vals = np.asarray(arr, dtype=np.float64).ravel()[np.isfinite(arr.ravel())]
    vals = vals[vals >= pa.ANALYSIS_DOMAIN_FLOOR]
    upper = max(pa.HISTOGRAM_UPPER_FLOOR, float(np.max(vals)))
    in_domain = vals
    counts, _ = np.histogram(in_domain, bins=pa.HISTOGRAM_BINS, range=(0.0, upper))
    counts = counts.astype(np.int64)

    assert model["range"] == (pa.ANALYSIS_DOMAIN_FLOOR, upper)
    assert model["full_range"] == (pa.ANALYSIS_DOMAIN_FLOOR, upper)
    assert model["bin_range"] == (pa.ANALYSIS_DOMAIN_FLOOR, upper)
    assert np.array_equal(model["counts"]["L"], counts)
    assert np.array_equal(model["log_counts"]["L"], np.log1p(counts.astype(np.float64)))
    st = model["stats"]["L"]
    assert st["min"] == pytest.approx(float(np.min(in_domain)), rel=0, abs=1e-9)
    assert st["max"] == pytest.approx(float(np.max(in_domain)), rel=0, abs=1e-9)
    assert st["median"] == pytest.approx(float(np.median(in_domain)), rel=0, abs=1e-9)
    assert st["mean"] == pytest.approx(float(np.mean(in_domain)), rel=0, abs=1e-9)
    assert st["std"] == pytest.approx(float(np.std(in_domain)), rel=0, abs=1e-9)


# --------------------------------------------------------------------------- #
# (d) display-vs-scientific distinction represented
# --------------------------------------------------------------------------- #
def test_scientific_and_display_models_are_distinguishable():
    sci = sh.compute_scientific_histogram(SIGNED)
    display = pa.compute_histogram_float(np.abs(SIGNED))

    assert sci["domain"] == "scientific"
    # The display model is NOT tagged as the scientific space.
    assert display.get("domain") != "scientific"

    text = sh.format_scientific_histogram_status(sci)
    assert "Scientific" in text
    assert "-4.733" in text  # truthful signed floor is surfaced
    # Absent model -> empty label (no accidental scientific claim).
    assert sh.format_scientific_histogram_status(None) == ""
