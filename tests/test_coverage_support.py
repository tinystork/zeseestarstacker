"""Tests for the backend-neutral positive support accumulator (COV-01A).

Synthetic, fast, deterministic.  These pin the core state/math contract only:
positive per-exposure support, SUP_W1/SUP_W2 accumulation, derived N_eff_support,
atomic fail-before-mutation (including cumulative overflow), snapshot/restore
exactness, decomposition invariance, unit-weight / 100k-exposure dtype boundary,
and the exact-2D shape contract.
"""

import numpy as np
import pytest

from seestar.core.coverage_support import (
    PositiveSupportAccumulator,
    SUPPORT_STATE_VERSION,
    SUPPORT_DTYPES,
)


def _acc(shape=(2, 3), dtype=np.float64):
    return PositiveSupportAccumulator(shape, dtype=dtype)


def _s(shape, value):
    """Full-shape support array of constant positive value."""
    return np.full(shape, value, dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. Exact W1/W2 known-weight witness
# ---------------------------------------------------------------------------
def test_exact_w1_w2_known_weights():
    acc = _acc((2, 2))
    s0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    s1 = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    acc.add(s0)
    acc.add(s1)
    expected_w1 = s0 + s1
    expected_w2 = s0 * s0 + s1 * s1
    assert np.array_equal(acc.support_w1, expected_w1)
    assert np.array_equal(acc.support_w2, expected_w2)


# ---------------------------------------------------------------------------
# 2. Derived N_eff witness
# ---------------------------------------------------------------------------
def test_n_eff_witness():
    acc = _acc((2, 2))
    s0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    s1 = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    acc.add(s0)
    acc.add(s1)
    w1 = s0 + s1
    w2 = s0 * s0 + s1 * s1
    expected = w1 * w1 / w2
    result = acc.n_eff_support
    assert result.shape == (2, 2)
    assert np.allclose(result, expected, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# 3. Zero / undefined support
# ---------------------------------------------------------------------------
def test_zero_undefined_support():
    acc = _acc((2, 2))
    # No support added: N_eff must be 0.0 everywhere (documented neutral value).
    result = acc.n_eff_support
    assert np.all(result == 0.0)
    assert np.all(result >= 0.0)
    assert np.all(np.isfinite(result))


def test_partial_support_is_zero_where_undefined():
    acc = _acc((2, 2))
    s = np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float64)
    acc.add(s)
    result = acc.n_eff_support
    # Pixels with support: single exposure => N_eff == 1.0 (W1==s, W2==s**2).
    assert result[0, 1] == 1.0
    assert result[1, 1] == 1.0
    # Pixels with no support: neutral 0.0.
    assert result[0, 0] == 0.0
    assert result[1, 0] == 0.0


# ---------------------------------------------------------------------------
# 4. Invalid inputs fail before mutation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    np.array([[1.0, -1.0], [1.0, 1.0]]),  # negative
    np.array([[1.0, np.nan], [1.0, 1.0]]),  # NaN
    np.array([[1.0, np.inf], [1.0, 1.0]]),  # +Inf
])
def test_invalid_negative_nan_inf_fail_before_mutation(bad):
    acc = _acc((2, 2))
    acc.add(np.ones((2, 2)))
    w1_before = acc.support_w1
    w2_before = acc.support_w2
    with pytest.raises(ValueError):
        acc.add(bad)
    assert np.array_equal(acc.support_w1, w1_before)
    assert np.array_equal(acc.support_w2, w2_before)


def test_shape_mismatch_fail_before_mutation():
    acc = _acc((2, 2))
    acc.add(np.ones((2, 2)))
    w1_before = acc.support_w1
    w2_before = acc.support_w2
    with pytest.raises(ValueError):
        acc.add(np.ones((3, 3)))
    assert np.array_equal(acc.support_w1, w1_before)
    assert np.array_equal(acc.support_w2, w2_before)


def test_overflow_squared_fail_before_mutation():
    acc = _acc((1, 1))
    acc.add(np.ones((1, 1)))
    w1_before = acc.support_w1
    w2_before = acc.support_w2
    # Finite but s**2 overflows to +Inf in float64.
    huge = np.array([[1e200]], dtype=np.float64)
    with pytest.raises(ValueError):
        acc.add(huge)
    assert np.array_equal(acc.support_w1, w1_before)
    assert np.array_equal(acc.support_w2, w2_before)


def test_shape_validation_rejects_bad_shapes():
    with pytest.raises(ValueError):
        PositiveSupportAccumulator(())
    with pytest.raises(ValueError):
        PositiveSupportAccumulator((5,))  # 1-D
    with pytest.raises(ValueError):
        PositiveSupportAccumulator((0, 5))
    with pytest.raises(ValueError):
        PositiveSupportAccumulator((5, -1))


def test_3d_shape_rejected():
    with pytest.raises(ValueError):
        PositiveSupportAccumulator((2, 2, 3))


def test_dtype_validation_rejects_unknown():
    with pytest.raises(TypeError):
        PositiveSupportAccumulator((2, 2), dtype=np.int64)
    with pytest.raises(TypeError):
        PositiveSupportAccumulator((2, 2), dtype=np.float16)


# ---------------------------------------------------------------------------
# 5. Import/restore invalid state fails before visible state
# ---------------------------------------------------------------------------
def test_restore_invalid_state_fails_closed():
    acc = _acc((2, 2))
    acc.add(np.ones((2, 2)))
    good = acc.to_state()
    cases = []
    for key in ("support_w1", "support_w2"):
        c = dict(good)
        c[key] = np.array([[1.0, 1.0], [1.0, np.nan]])
        cases.append(c)
        c2 = dict(good)
        c2[key] = np.array([[1.0, -1.0], [1.0, 1.0]])
        cases.append(c2)
    bad_version = dict(good)
    bad_version["version"] = 999
    cases.append(bad_version)
    bad_dtype = dict(good)
    bad_dtype["dtype"] = "int64"
    cases.append(bad_dtype)
    bad_shape = dict(good)
    bad_shape["shape"] = [3, 3]
    cases.append(bad_shape)
    for c in cases:
        with pytest.raises((ValueError, TypeError)):
            PositiveSupportAccumulator.from_state(c)


def test_restore_rejects_non_dict():
    with pytest.raises(TypeError):
        PositiveSupportAccumulator.from_state([1, 2, 3])


# ---------------------------------------------------------------------------
# 6. Snapshot/restore exactness + no aliasing
# ---------------------------------------------------------------------------
def test_snapshot_restore_exactness_and_no_aliasing():
    acc = _acc((2, 2))
    acc.add(np.array([[1.0, 2.0], [3.0, 4.0]]))
    acc.add(np.array([[0.5, 0.25], [1.5, 2.0]]))
    state = acc.to_state()
    assert state["version"] == SUPPORT_STATE_VERSION
    assert state["shape"] == [2, 2]
    assert state["dtype"] == "float64"
    # to_state must return owned copies: mutating state does not affect acc.
    w1_snap = state["support_w1"].copy()
    w2_snap = state["support_w2"].copy()
    state["support_w1"][0, 0] = 999.0
    state["support_w2"][0, 0] = 999.0
    assert acc.support_w1[0, 0] != 999.0
    assert acc.support_w2[0, 0] != 999.0
    # Restore from pristine copies.
    restored = PositiveSupportAccumulator.from_state(
        {"version": state["version"], "shape": state["shape"],
         "dtype": state["dtype"],
         "support_w1": w1_snap, "support_w2": w2_snap}
    )
    assert restored.shape == acc.shape
    assert np.array_equal(restored.support_w1, acc.support_w1)
    assert np.array_equal(restored.support_w2, acc.support_w2)
    # No aliasing: mutate the state arrays again, restored is unaffected.
    w1_snap[0, 0] = 12345.0
    assert restored.support_w1[0, 0] != 12345.0


# ---------------------------------------------------------------------------
# 7. Decomposition invariance across partition markers
# ---------------------------------------------------------------------------
def test_decomposition_partitions():
    rng = np.random.default_rng(12345)
    shape = (4, 4)
    n = 61
    supports = [rng.uniform(0.0, 1.0, size=shape).astype(np.float64) for _ in range(n)]

    def run_all_in_order():
        a = _acc(shape)
        for s in supports:
            a.add(s)
        return a

    def run_partitioned(parts):
        a = _acc(shape)
        i = 0
        for k in parts:
            for _ in range(k):
                a.add(supports[i])
                i += 1
        return a

    all_in_one = run_all_in_order()
    part_3_17_41 = run_partitioned([3, 17, 41])
    singletons = run_partitioned([1] * n)
    assert np.array_equal(all_in_one.support_w1, part_3_17_41.support_w1)
    assert np.array_equal(all_in_one.support_w2, part_3_17_41.support_w2)
    assert np.array_equal(all_in_one.support_w1, singletons.support_w1)
    assert np.array_equal(all_in_one.support_w2, singletons.support_w2)
    assert np.array_equal(all_in_one.n_eff_support, singletons.n_eff_support)


# ---------------------------------------------------------------------------
# 8. Unit-weight restricted case reduces to count
# ---------------------------------------------------------------------------
def test_unit_weight_reduces_to_count():
    shape = (3, 3)
    acc = _acc(shape)
    n = 7
    for _ in range(n):
        acc.add(np.ones(shape))
    # Unit weight: SUP_W1 == SUP_W2 == n, N_eff == n (the count).
    assert np.all(acc.support_w1 == n)
    assert np.all(acc.support_w2 == n)
    assert np.all(acc.n_eff_support == n)


# ---------------------------------------------------------------------------
# 9. Dtype / overflow boundary at 100k contributions (float64 exact)
# ---------------------------------------------------------------------------
def test_100k_unit_contributions_float64_exact():
    shape = (2, 2)
    acc = _acc(shape, dtype=np.float64)
    n = 100_000
    for _ in range(n):
        acc.add(np.ones(shape))
    assert np.all(acc.support_w1 == float(n))
    assert np.all(acc.support_w2 == float(n))
    assert np.all(acc.n_eff_support == float(n))


def test_float32_dtype_is_supported():
    acc = _acc((2, 2), dtype=np.float32)
    acc.add(np.ones((2, 2), dtype=np.float32))
    assert acc.dtype == np.dtype(np.float32)
    assert acc.support_w1.dtype == np.float32
    assert np.all(acc.support_w1 == 1.0)
    assert SUPPORT_DTYPES == (np.dtype(np.float32), np.dtype(np.float64))


# ---------------------------------------------------------------------------
# 10. N_eff never mutates state
# ---------------------------------------------------------------------------
def test_n_eff_does_not_mutate_state():
    acc = _acc((2, 2))
    acc.add(np.ones((2, 2)))
    w1_before = acc.support_w1
    w2_before = acc.support_w2
    _ = acc.n_eff_support
    assert np.array_equal(acc.support_w1, w1_before)
    assert np.array_equal(acc.support_w2, w2_before)


# ---------------------------------------------------------------------------
# 11. Cumulative-overflow regression (float64 and float32)
# ---------------------------------------------------------------------------
def test_cumulative_overflow_fails_before_mutation_float64():
    acc = _acc((1, 1), dtype=np.float64)
    s = np.array([[1e154]], dtype=np.float64)
    acc.add(s)  # W1=1e154, W2=1e308 (both finite)
    w1_before = acc.support_w1
    w2_before = acc.support_w2
    assert np.all(np.isfinite(w1_before))
    assert np.all(np.isfinite(w2_before))
    # Second add would push W2 (1e308) to 2e308 = +Inf.
    with pytest.raises(ValueError):
        acc.add(s)
    assert np.array_equal(acc.support_w1, w1_before)
    assert np.array_equal(acc.support_w2, w2_before)


def test_cumulative_overflow_fails_before_mutation_float32():
    acc = _acc((1, 1), dtype=np.float32)
    s = np.array([[1.5e19]], dtype=np.float32)
    acc.add(s)  # W1~1.5e19, W2~2.25e38 (both finite in float32)
    w1_before = acc.support_w1
    w2_before = acc.support_w2
    assert np.all(np.isfinite(w1_before))
    assert np.all(np.isfinite(w2_before))
    # Second add would push W2 (~2.25e38) past float32 max (~3.4e38).
    with pytest.raises(ValueError):
        acc.add(s)
    assert np.array_equal(acc.support_w1, w1_before)
    assert np.array_equal(acc.support_w2, w2_before)


# ---------------------------------------------------------------------------
# 12. Huge-but-valid N_eff must not silently overflow to 0
# ---------------------------------------------------------------------------
def test_huge_valid_n_eff_not_overflow():
    acc = _acc((1, 1), dtype=np.float64)
    s = np.array([[7e153]], dtype=np.float64)
    acc.add(s)
    acc.add(s)
    # W1 = 1.4e154, W2 = 9.8e307 (both finite); naive W1**2 would overflow.
    assert np.all(np.isfinite(acc.support_w1))
    assert np.all(np.isfinite(acc.support_w2))
    n_eff = float(acc.n_eff_support[0, 0])
    assert np.isfinite(n_eff)
    # Two equal supports => effective support ~2.0, not 0.0.
    assert np.isclose(n_eff, 2.0, rtol=1e-3, atol=1e-3)
