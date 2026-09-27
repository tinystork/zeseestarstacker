"""C6 DQ combination tests: DQ-primary invalidation into the valid/support mask.

These prove the frozen DQ contract (§5): ``invalid_final = (mask != 0) OR
invalidité_existante``, DQ primary (never overridden by a luminance mask), and
the explicit §43 test that FAILS if ``CalibrationResult.mask`` is dropped.
"""

from __future__ import annotations

import numpy as np

from seestar.queuep.queue_manager import SeestarQueuedStacker

# ZeCalibrator DQ bit categories (zecalibrator/core/dq.py).
DQ = {
    "INPUT_INVALID": 0x0001,
    "ADDITIVE_INVALID": 0x0002,
    "FLAT_INVALID": 0x0004,
    "SATURATED": 0x0008,
    "ARITH_NONFINITE": 0x0010,
}


def _stub():
    s = object.__new__(SeestarQueuedStacker)
    s._calibration_masks = {}
    s._dq_invalidated_count = 0
    return s


def test_dq_invalid_pixels_removed_from_valid_mask():
    s = _stub()
    mask = np.zeros((3, 3), dtype=np.uint16)
    mask[0, 1] = DQ["INPUT_INVALID"]
    mask[1, 1] = DQ["SATURATED"]
    mask[2, 2] = DQ["FLAT_INVALID"] | DQ["ARITH_NONFINITE"]
    s._calibration_masks["f.fits"] = mask
    valid = np.ones((3, 3), dtype=bool)
    combined = s._combine_dq_into_valid_mask("f.fits", valid, None, True)
    assert not combined[0, 1]
    assert not combined[1, 1]
    assert not combined[2, 2]
    assert combined[0, 0] and combined[2, 0] and combined[2, 1]


def test_dq_is_primary_not_overridden_by_luminance():
    # The luminance mask says "valid" everywhere; DQ still wins.
    s = _stub()
    mask = np.zeros((2, 2), dtype=np.uint16)
    mask[0, 0] = DQ["INPUT_INVALID"]
    s._calibration_masks["f.fits"] = mask
    valid = np.ones((2, 2), dtype=bool)  # luminance says all valid
    combined = s._combine_dq_into_valid_mask("f.fits", valid, None, True)
    assert not combined[0, 0]
    assert combined[0, 1] and combined[1, 0] and combined[1, 1]


def test_no_dq_zero_change_by_default():
    # No calibration DQ present -> the valid mask is untouched (zero change).
    s = _stub()
    valid = np.ones((2, 2), dtype=bool)
    combined = s._combine_dq_into_valid_mask("f.fits", valid, None, True)
    assert np.array_equal(combined, valid)
    assert s._dq_invalidated_count == 0


def test_audit_counter_increments():
    s = _stub()
    mask = np.zeros((2, 2), dtype=np.uint16)
    mask[0, 0] = DQ["SATURATED"]
    mask[1, 1] = DQ["ADDITIVE_INVALID"]
    s._calibration_masks["f.fits"] = mask
    valid = np.ones((2, 2), dtype=bool)
    s._combine_dq_into_valid_mask("f.fits", valid, None, True)
    assert s._dq_invalidated_count == 2


def test_dq_dropped_would_fail_explicit():
    # §43: if the integration simply dropped CalibrationResult.mask, this
    # assertion (DQ-invalid pixel excluded from support) would fail.
    s = _stub()
    mask = np.zeros((2, 2), dtype=np.uint16)
    mask[0, 0] = DQ["FLAT_INVALID"]
    s._calibration_masks["f.fits"] = mask
    valid = np.ones((2, 2), dtype=bool)
    combined = s._combine_dq_into_valid_mask("f.fits", valid, None, True)
    assert not combined[0, 0]  # would be True if the mask were dropped


def test_classic_aligned_grid_warps_dq_by_M():
    # Classic: the valid mask lives on the ALIGNED grid; the DQ mask is warped
    # by M (nearest-neighbour) into that grid before the AND.
    s = _stub()
    mask = np.zeros((2, 2), dtype=np.uint16)
    mask[0, 0] = DQ["INPUT_INVALID"]
    s._calibration_masks["f.fits"] = mask
    valid = np.ones((3, 3), dtype=bool)  # aligned grid (different shape)
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)  # identity
    combined = s._combine_dq_into_valid_mask("f.fits", valid, M, False)
    assert not combined[0, 0]
    assert combined[0, 1] and combined[2, 2]


def test_malformed_dq_never_degrades_support():
    # A DQ mask whose warp fails must never make the support worse.
    s = _stub()
    s._calibration_masks["f.fits"] = np.zeros((2, 2), dtype=np.uint16)
    valid = np.ones((3, 3), dtype=bool)
    combined = s._combine_dq_into_valid_mask("f.fits", valid, None, False)
    assert np.array_equal(combined, valid)
