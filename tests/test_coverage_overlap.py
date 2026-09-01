"""COV-03 focused tests: coverage-aware reliable-overlap normalization.

The overlap used to estimate photometric scale/offset is restricted to the
reliable (high-support) region, excluding low-support peripheral contamination.
"""

import numpy as np

from seestar.enhancement.reproject_utils import compute_overlap_median_ratio


def _disk_weight(H, W, cx, cy, r_high, r_low, high=1.0, low=0.01):
    Y, X = np.ogrid[:H, :W]
    r = np.hypot(Y - cy, X - cx)
    return np.where(r < r_high, high, np.where(r < r_low, low, 0.0)).astype(np.float32)


def test_reliable_overlap_excludes_low_support_periphery():
    H = W = 64
    ref = np.full((H, W), 1000.0, np.float32)
    new = np.full((H, W), 1000.0, np.float32)
    w = _disk_weight(H, W, 31.5, 31.5, r_high=20, r_low=30, high=1.0, low=0.01)
    Y, X = np.ogrid[:H, :W]
    r = np.hypot(Y - 31.5, X - 31.5)
    periphery = (r >= 20) & (r < 30)
    new_contam = new.copy()
    new_contam[periphery] = 5000.0

    s0, o0, ov0, rm0, bm0 = compute_overlap_median_ratio(
        ref, new_contam, w, w, reliable_fraction=0.0
    )
    s1, o1, ov1, rm1, bm1 = compute_overlap_median_ratio(
        ref, new_contam, w, w, reliable_fraction=0.02
    )

    assert s1 is not None
    assert abs(s1 - 1.0) < 0.05
    assert ov1 < ov0


def test_reliable_fraction_zero_is_backward_compatible():
    # scale is ref/new (new * scale ~= ref); large enough to exceed min_overlap
    ref = np.full((40, 40), 10.0, np.float32)
    new = np.full((40, 40), 20.0, np.float32)
    w = np.ones((40, 40), np.float32)
    s0, o0, ov0, rm0, bm0 = compute_overlap_median_ratio(
        ref, new, w, w, reliable_fraction=0.0
    )
    assert ov0 == 40 * 40
    assert s0 is not None and abs(s0 - 0.5) < 0.05


def test_reliable_fraction_without_weights_is_noop():
    # no weights -> reliable_fraction cannot restrict; scale = ref/new
    ref = np.full((40, 40), 5.0, np.float32)
    new = np.full((40, 40), 15.0, np.float32)
    s, o, ov, rm, bm = compute_overlap_median_ratio(ref, new, None, None, reliable_fraction=0.1)
    assert ov == 40 * 40
    assert s is not None and abs(s - 1.0 / 3.0) < 0.05
