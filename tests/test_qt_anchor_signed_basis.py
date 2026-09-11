"""Signed-aware Qt anchor-basis regression tests (pure numpy, headless).

Mission: zsss-signed-float32-display-histogram-20260911 (task 2 of 3)

Defect (production Qt path): ``compute_anchors`` / ``adapt_anchors_for_drift``
selected their percentile basis from a strictly-positive sample only, so on a
signed float32 frame (reference range approx. -4.733 ... +82.809) with a
dominant negative/zero background the p0.5 black anchor landed *inside* the
faint positive signal, collapsing the background and faint signal to black.

Fix mirrors ``stretch_display_data`` (D4): the basis is the FULL finite signed
sample when there is signed content, a significant exact-zero population
(>= 2%) or too few positive pixels (< 20); otherwise the legacy positive-only
basis is kept so clean non-negative inputs stay byte-identical.

No Qt widgets/display: only the pure anchor/mapping functions are exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

pa = pytest.importorskip("seestar.gui_qt.preview_analysis")

ANCHOR_LO_PCT = pa.ANCHOR_LO_PCT
ANCHOR_HI_PCT = pa.ANCHOR_HI_PCT
ANCHOR_SEP = pa.ANCHOR_SEP
ANCHOR_DRIFT_HYSTERESIS = pa.ANCHOR_DRIFT_HYSTERESIS

compute_anchors = pa.compute_anchors
adapt_anchors_for_drift = pa.adapt_anchors_for_drift
map_raw_linear = pa.map_raw_linear
_finite_positive_sample = pa._finite_positive_sample


def _signed_reference_frame(seed: int = 7):
    """Signed float32 raw_linear with a dominant negative/zero background,
    a faint positive region and a couple of bright pixels.  Roughly spans the
    reference range -4.733 ... +82.809."""
    rng = np.random.default_rng(seed)
    arr = rng.normal(-2.0, 0.5, size=(64, 64)).astype(np.float32)
    arr[:16, :] = 0.0  # ~25% exact-zero background (no-data support)
    arr[40:56, 40:56] = rng.uniform(0.1, 1.5, size=(16, 16)).astype(np.float32)
    arr[0, 0] = np.float32(82.809)
    arr[0, 1] = np.float32(40.0)
    return arr


# --------------------------------------------------------------------------- #
# (a) signed input: black anchor at/under the negative floor + separation
# --------------------------------------------------------------------------- #
def test_signed_anchors_black_point_at_or_under_negative_floor():
    arr = _signed_reference_frame()

    n_finite = int(np.isfinite(arr).sum())
    n_neg = int(np.sum(arr[np.isfinite(arr)] < 0))
    n_zero = int(np.sum(arr[np.isfinite(arr)] == 0))
    assert n_neg > 0 and n_zero / n_finite >= 0.02  # inclusive basis applies

    lo, hi = compute_anchors(arr)

    assert np.isfinite(lo) and np.isfinite(hi)
    assert hi > lo + ANCHOR_SEP
    # The black anchor must sit at/under the negative background floor, not
    # inside the faint positive signal.
    assert lo <= 0.0, f"black anchor inside signal: lo={lo!r}"


def test_signed_mapping_separates_background_from_faint_signal():
    arr = _signed_reference_frame()
    lo, hi = compute_anchors(arr)

    mapped = map_raw_linear(arr, lo, hi)
    finite_mapped = mapped[np.isfinite(mapped)]

    # No giant zero wall: only the sub-black tail (below the p0.5 floor) should
    # collapse to 0.
    frac_zero = float((finite_mapped == 0.0).mean())
    assert frac_zero < 0.05, f"artificial zero wall remains: frac0={frac_zero}"

    # The faint positive region must map strictly above the negative background.
    bg = mapped[:16, :]  # exact-zero background (arr == 0.0 > negative floor)
    faint = mapped[40:56, 40:56]
    assert float(np.median(faint)) > float(np.median(bg))

    # Bright pixel retains > 1 headroom (not silently clipped at the mapping).
    assert float(mapped[0, 0]) > 1.0


def test_old_positive_only_basis_would_have_collapsed_background():
    """Contrast witness: the pre-change positive-only basis produces the
    artificial zero wall this fix removes."""
    arr = _signed_reference_frame()
    old_sample = _finite_positive_sample(np, arr)
    assert old_sample is not None and old_sample.size > 0
    old_lo = float(np.percentile(old_sample, ANCHOR_LO_PCT))
    old_hi = float(np.percentile(old_sample, ANCHOR_HI_PCT))
    assert old_lo > 0.0  # black point inside the faint positive signal

    old_mapped = map_raw_linear(arr, old_lo, old_hi)
    old_frac_zero = float((old_mapped == 0.0).mean())
    assert old_frac_zero > 0.5, f"expected collapse, frac0={old_frac_zero}"


# --------------------------------------------------------------------------- #
# (b) clean non-negative inputs: byte-identical anchors (pre-change behaviour)
# --------------------------------------------------------------------------- #
def _prechange_compute_anchors(np, arr, sep=ANCHOR_SEP):
    """Replicates the pre-change compute_anchors (positive-only basis)."""
    arr = np.asarray(arr, dtype=np.float64)
    sample = _finite_positive_sample(np, arr)
    if sample is not None and sample.size > 0:
        lo = float(np.percentile(sample, ANCHOR_LO_PCT))
        hi = float(np.percentile(sample, ANCHOR_HI_PCT))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo + sep:
            return (lo, hi)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (0.0, 1.0)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi - lo > sep:
        return (lo, hi)
    mid = 0.5 * (lo + hi)
    return (mid - sep, mid + sep)


def _prechange_adapt(np, anchor_lo, anchor_hi, arr, hysteresis=ANCHOR_DRIFT_HYSTERESIS, sep=ANCHOR_SEP):
    """Replicates the pre-change adapt_anchors_for_drift (positive-only basis)."""
    arr = np.asarray(arr, dtype=np.float64)
    lo, hi = float(anchor_lo), float(anchor_hi)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return _prechange_compute_anchors(np, arr, sep=sep)
    sample = _finite_positive_sample(np, arr)
    if sample is None or sample.size == 0:
        return (lo, hi)
    cur_lo = float(np.percentile(sample, ANCHOR_LO_PCT))
    cur_hi = float(np.percentile(sample, ANCHOR_HI_PCT))
    if not (np.isfinite(cur_lo) and np.isfinite(cur_hi)):
        return (lo, hi)
    span = hi - lo
    if span <= 0.0:
        span = sep
    band = float(hysteresis) * span
    new_lo, new_hi = lo, hi
    if cur_hi > hi + band:
        new_hi = cur_hi
    if cur_lo < lo - band:
        new_lo = cur_lo
    if new_hi - new_lo <= sep:
        mid = 0.5 * (new_lo + new_hi)
        new_lo = mid - sep
        new_hi = mid + sep
    return (new_lo, new_hi)


def test_clean_non_negative_compute_anchors_byte_identical():
    rng = np.random.default_rng(0)
    arr = rng.uniform(1.0, 100.0, size=(64, 64)).astype(np.float32)

    assert pa._anchor_basis(np, arr) is not None
    assert compute_anchors(arr) == _prechange_compute_anchors(np, arr)


def test_clean_non_negative_adapt_anchors_byte_identical():
    rng = np.random.default_rng(13)
    frame1 = rng.uniform(100.0, 200.0, size=(64, 64, 3)).astype(np.float32)
    lo, hi = compute_anchors(frame1)

    for scale in (1.0, 1.10, 2.0, 3.0, 0.25):
        frame = frame1 * scale
        assert adapt_anchors_for_drift(lo, hi, frame) == _prechange_adapt(
            np, lo, hi, frame
        ), f"adapt anchors changed for clean non-negative scale={scale}"


def test_clean_non_negative_degenerate_inputs_unchanged():
    rng = np.random.default_rng(16)
    frame1 = rng.uniform(1.0, 10.0, size=(16, 16)).astype(np.float32)
    lo, hi = compute_anchors(frame1)

    assert adapt_anchors_for_drift(lo, hi, np.full((8, 8), np.nan)) == (lo, hi)
    assert adapt_anchors_for_drift(lo, hi, np.zeros((0, 0))) == (lo, hi)
    assert compute_anchors(np.full((8, 8), np.nan, dtype=np.float32)) == (0.0, 1.0)
    assert compute_anchors(np.zeros((0, 0), dtype=np.float32)) == (0.0, 1.0)


# --------------------------------------------------------------------------- #
# Degenerate / basis-selection edge cases
# --------------------------------------------------------------------------- #
def test_anchor_basis_inclusive_when_signed_content_present():
    arr = np.array([[-1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
    basis = pa._anchor_basis(np, arr)
    assert basis is not None
    assert float(np.min(basis)) < 0.0  # negatives retained in the basis


def test_anchor_basis_inclusive_on_large_zero_population():
    """A non-negative frame with a significant exact-zero population uses the
    inclusive basis (D4 mirror) — zeros pull the floor to 0."""
    arr = np.concatenate([np.zeros(200, dtype=np.float32), np.linspace(1.0, 5.0, 800, dtype=np.float32)])
    basis = pa._anchor_basis(np, arr)
    assert basis is not None
    assert float(np.min(basis)) == 0.0
    assert float((basis == 0.0).sum()) > 0


def test_anchor_basis_none_when_no_finite():
    arr = np.full((4, 4), np.nan, dtype=np.float32)
    assert pa._anchor_basis(np, arr) is None
