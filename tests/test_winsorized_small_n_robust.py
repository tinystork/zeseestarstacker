"""Small-N gross-outlier guard + per-column zero-rank Winsorized Sigma tests.

Mission ``ZSSS-HOT-PIXEL-SMALL-N-ROBUSTNESS-20260923`` (rework-3).

Validates the ONE-PASS translation/scale-conservative extreme-gap guard that
replaced the R2 median/MAD rule for zero-rank columns, and the per-column
zero-rank classification (``floor(low * n_valid) == floor(high * n_valid) == 0``
from the ORIGINAL valid population, never the nominal N_batch).

R1 rejected 87316/500000 samples on a 100,000-column N=5 Gaussian witness
(seed 20260924, mean 100, sigma 5); R2's median/MAD rule rejected 5029 at
mean 0, sigma 5.  The rework-3 extreme-gap rule rejects 0 at both.
"""

from __future__ import annotations

import numpy as np
import pytest

from seestar.core.stack_methods import (
    _gross_outlier_keep,
    _stack_winsorized_sigma_iter,
    _winsor_zero_rank_cols,
)

DEFAULT_LIMITS = (0.05, 0.05)


def _run(values, **kw):
    images = [np.full((1, 1), float(v), dtype=np.float32) for v in values]
    kw.setdefault("kappa", 3.0)
    kw.setdefault("winsor_limits", DEFAULT_LIMITS)
    return _stack_winsorized_sigma_iter(images, None, **kw)


# ---------------------------------------------------------------------------
# Synthetic reducer witness
# ---------------------------------------------------------------------------
def test_witness_outlier_rejected():
    result, pct = _run([100, 101, 99, 102, 63000])
    assert pct > 0.0, "the 63000 gross outlier must be rejected"
    assert float(result[0, 0]) < 200.0  # never dominated by the 63000


def test_witness_coherent_no_artificial_rejection():
    result, pct = _run([100, 101, 99, 102, 98])
    assert pct == 0.0
    assert float(result[0, 0]) == pytest.approx(100.0, abs=1e-6)


def test_witness_apply_rewinsor_false_excludes_outlier():
    result, pct = _run([100, 101, 99, 102, 63000], apply_rewinsor=False)
    assert pct > 0.0
    assert float(result[0, 0]) == pytest.approx(100.5, abs=1e-6)  # mean of 4


# ---------------------------------------------------------------------------
# Explicit N=1..5 behavior
# ---------------------------------------------------------------------------
def test_n1_kept():
    result, pct = _run([100.0])
    assert pct == 0.0
    assert float(result[0, 0]) == pytest.approx(100.0)


def test_n2_no_safe_rejection():
    # n_valid <= 2 -> no safe rejection (both samples kept).
    result, pct = _run([100.0, 105.0])
    assert pct == 0.0
    assert float(result[0, 0]) == pytest.approx(102.5)


def test_n3_rejects_huge_outlier():
    result, pct = _run([100.0, 101.0, 63000.0])
    assert pct > 0.0
    assert float(result[0, 0]) < 200.0


def test_n4_rejects_huge_outlier():
    result, pct = _run([100.0, 101.0, 99.0, 63000.0])
    assert pct > 0.0
    assert float(result[0, 0]) < 200.0


# ---------------------------------------------------------------------------
# Amplitude witnesses (parameterized [100,101,99,102,X])
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("x", [500.0, 1000.0, 5000.0, 20000.0, 63000.0])
def test_isolated_outlier_amplitude_rejected(x):
    """[100,101,99,102,X]: the isolated X is rejected for every amplitude,
    leaving the 4 inlier survivors (result 100.8 with apply_rewinsor)."""
    result, pct = _run([100, 101, 99, 102, x])
    assert pct == pytest.approx(20.0)  # exactly the single X rejected
    assert float(result[0, 0]) == pytest.approx(100.8, abs=1e-4)


def test_ramp_characterized_additional():
    """[500,1000,5000,20000,63000] (ramp): no isolated extreme — every value
    is part of the spread, so the guard rejects nothing (conservative)."""
    result, pct = _run([500, 1000, 5000, 20000, 63000])
    assert pct == 0.0
    assert float(result[0, 0]) == pytest.approx(17900.0, rel=1e-3)


def test_flat_bright_cluster_rejects_huge_outlier():
    result, pct = _run([100.0, 100.0, 100.0, 100.0, 63000.0])
    assert pct == pytest.approx(20.0)
    assert float(result[0, 0]) == pytest.approx(100.0, abs=1e-6)


def test_flat_bright_cluster_keeps_tiny_variation():
    # All-equal survivors with a tiny variation (101 among 100s): only the
    # 63000 extreme is rejected.
    result, pct = _run([100.0, 100.0, 100.0, 101.0, 63000.0])
    assert pct == pytest.approx(20.0)
    assert float(result[0, 0]) < 200.0


# ---------------------------------------------------------------------------
# Deterministic Monte Carlo (the R1 87316 / R2 mean0-sigma5 rejection witness)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mean,sigma",
    [(100.0, 5.0), (0.0, 1.0), (0.0, 5.0), (1.0, 1.0), (10.0, 5.0), (-100.0, 5.0)],
)
def test_gaussian_monte_carlo_zero_rejection(mean, sigma):
    """100,000 N=5 Gaussian columns (seed 20260924): the one-pass
    translation/scale-conservative guard rejects exactly 0 samples (0 columns)
    for every mean/sigma.  R1 rejected 87316 at (100,5); R2's median/MAD rule
    rejected 5029 at (0,5) — this rule rejects 0 at both."""
    rng = np.random.default_rng(20260924)
    N = 100000
    cols = rng.normal(mean, sigma, size=(5, N)).astype(np.float32)
    arr = cols.reshape(5, N, 1)
    images = [arr[i] for i in range(5)]
    result, pct = _stack_winsorized_sigma_iter(
        images, None, kappa=3.0, winsor_limits=DEFAULT_LIMITS
    )
    assert pct == 0.0  # 0 rejected samples => 0 affected columns


# ---------------------------------------------------------------------------
# One-pass proof
# ---------------------------------------------------------------------------
def test_one_pass_guard_does_not_invoke_winsor_body(monkeypatch):
    """The all-zero-rank guard never invokes the iterative Winsor body."""
    import seestar.core.stack_methods as sm

    calls = {"n": 0}
    orig = sm._winsorized_sigma_iteration_body

    def spy(arr, mask, kappa_iter, winsor_limits):
        calls["n"] += 1
        return orig(arr, mask, kappa_iter, winsor_limits)

    monkeypatch.setattr(sm, "_winsorized_sigma_iteration_body", spy)
    _run([100, 101, 99, 102, 63000])
    assert calls["n"] == 0  # all-zero-rank: no Winsor iterations


def test_one_pass_guard_invariant_to_max_iters_and_kappa_decay():
    vals = [100, 101, 99, 102, 63000]
    base, base_pct = _run(vals)
    r1, p1 = _run(vals, max_iters=1)
    r2, p2 = _run(vals, max_iters=10, kappa_decay=0.5)
    r3, p3 = _run(vals, max_iters=5, kappa_decay=1.0)
    assert base_pct == p1 == p2 == p3 == pytest.approx(20.0)
    for r in (r1, r2, r3):
        assert float(r[0, 0]) == float(base[0, 0])


# ---------------------------------------------------------------------------
# Per-column zero-rank classification + mixed populations
# ---------------------------------------------------------------------------
def test_zero_rank_cols_boundary():
    n = np.array([1, 2, 5, 10, 19, 20, 44], dtype=np.int64)
    cols = _winsor_zero_rank_cols(DEFAULT_LIMITS, n)
    assert cols.tolist() == [True, True, True, True, True, False, False]


def test_mixed_n_valid_populations():
    """One cube with columns of n_valid 5,10,19 (zero-rank -> guard) and
    20,44 (rank-sufficient -> historical Winsor); the guard is per-column."""
    rng = np.random.default_rng(0)
    N = 44
    H, W = 1, 5
    arr = rng.normal(100.0, 5.0, size=(N, H, W)).astype(np.float32)
    # column valid counts: 5, 10, 19, 20, 44
    for i in range(5, N):
        arr[i, 0, 0] = np.nan
    for i in range(10, N):
        arr[i, 0, 1] = np.nan
    for i in range(19, N):
        arr[i, 0, 2] = np.nan
    for i in range(20, N):
        arr[i, 0, 3] = np.nan
    # inject a gross outlier in the n_valid=5 column (index 0)
    arr[0, 0, 0] = 63000.0
    images = [arr[i] for i in range(N)]

    result, pct = _stack_winsorized_sigma_iter(
        images, None, kappa=3.0, winsor_limits=DEFAULT_LIMITS
    )
    # The gross outlier in the n_valid=5 column is rejected -> its mean is
    # near 100 (not pulled to ~12600).
    assert float(result[0, 0]) < 200.0
    # The rank-sufficient columns are unaffected (mean ~100).
    assert float(result[0, 3]) == pytest.approx(100.0, abs=3.0)
    assert float(result[0, 4]) == pytest.approx(100.0, abs=3.0)


def test_rank_sufficient_column_unaffected_by_guard_in_same_cube():
    """A rank-sufficient column keeps the historical iterative Winsor even in
    a cube that also contains a zero-rank gross-outlier column: the guard does
    NOT consume kappa-decay iterations for the rank-sufficient column."""
    rng = np.random.default_rng(1)
    N = 20
    arr = rng.normal(100.0, 5.0, size=(N, 1, 2)).astype(np.float32)
    # column 0: rank-sufficient (n_valid=20), clean-ish with a mild outlier.
    arr[0, 0, 0] = 100.0 + 40.0  # mild outlier within winsor's reach
    # column 1: zero-rank (n_valid=5), with a gross outlier.
    for i in range(5, N):
        arr[i, 0, 1] = np.nan
    arr[0, 0, 1] = 63000.0
    images = [arr[i] for i in range(N)]

    # Reference: rank-sufficient column alone.
    ref_col0 = [arr[i, :, 0:1] for i in range(N)]
    ref_res, ref_pct = _stack_winsorized_sigma_iter(
        ref_col0, None, kappa=3.0, winsor_limits=DEFAULT_LIMITS
    )

    result, pct = _stack_winsorized_sigma_iter(
        images, None, kappa=3.0, winsor_limits=DEFAULT_LIMITS
    )
    # The rank-sufficient column result is bitwise identical to running alone.
    assert float(result[0, 0]) == float(ref_res[0, 0])
    # The gross outlier in the zero-rank column is rejected.
    assert float(result[0, 1]) < 200.0


# ---------------------------------------------------------------------------
# Helper-level: gross guard keep mask
# ---------------------------------------------------------------------------
def test_gross_outlier_keep_n2_no_rejection():
    col = np.array([100, 105, np.nan, np.nan, np.nan], dtype=np.float32).reshape(5, 1, 1)
    valid = ~np.isnan(col)
    keep = _gross_outlier_keep(col, valid)
    # n_valid == 2 -> everything kept.
    assert int(np.count_nonzero(keep)) == 2


def test_gross_outlier_keep_rejects_isolated():
    col = np.array([100, 101, 99, 102, 63000], dtype=np.float32).reshape(5, 1, 1)
    valid = ~np.isnan(col)
    keep = _gross_outlier_keep(col, valid)
    assert int(np.count_nonzero(keep)) == 4  # the 63000 rejected


# ---------------------------------------------------------------------------
# R3: positive-multiplicative-scaling invariance (normalized witnesses)
# ---------------------------------------------------------------------------
def test_normalized_small_n_rejects_only_outlier():
    """[.0400,.0401,.0399,.0402,.9978]: rejects only the .9978 (the R2/R3
    defect where the absolute ``3*max(|median|,1)`` floor let the outlier
    through, yielding a result of .2316)."""
    result, pct = _run([0.0400, 0.0401, 0.0399, 0.0402, 0.9978])
    assert pct == pytest.approx(20.0)  # only the single .9978 rejected
    assert float(result[0, 0]) < 0.1  # never dominated by the .9978


def test_normalized_small_n_clean_rejects_none():
    """[.0400,.0401,.0399,.0402,.0398]: no rejection (clean tight cluster)."""
    result, pct = _run([0.0400, 0.0401, 0.0399, 0.0402, 0.0398])
    assert pct == 0.0
    assert float(result[0, 0]) == pytest.approx(0.0400, abs=1e-5)


def test_small_n_scale_invariance_equivalent_results():
    """Scaling the whole column by 1, 1000, 65535, 1e-3 keeps the identical
    rejection decision and scales the result proportionally (relative to the
    scale-1 reference)."""
    base = [0.0400, 0.0401, 0.0399, 0.0402, 0.9978]
    clean = [0.0400, 0.0401, 0.0399, 0.0402, 0.0398]
    ref_res, ref_pct = _run(base)
    ref_clean, ref_clean_pct = _run(clean)
    assert ref_pct == pytest.approx(20.0)
    assert ref_clean_pct == 0.0
    for scale in (1000.0, 65535.0, 1e-3):
        result, pct = _run([v * scale for v in base])
        assert pct == pytest.approx(20.0)
        assert float(result[0, 0]) == pytest.approx(
            float(ref_res[0, 0]) * scale, rel=1e-3
        )
        result_c, pct_c = _run([v * scale for v in clean])
        assert pct_c == 0.0
        assert float(result_c[0, 0]) == pytest.approx(
            float(ref_clean[0, 0]) * scale, rel=1e-3
        )


def test_pure_noise_zero_rejection_scaled_equivalent():
    """100,000 N=5 Gaussian columns (seed 20260924, mean .04, sigma .002)
    generated ONCE, then scaled by 1, 1e3, 1e6: 0 false rejection at every
    scale and scale-equivalent (all-keep) weight maps."""
    rng = np.random.default_rng(20260924)
    N = 100000
    base = rng.normal(0.04, 0.002, size=(5, N)).astype(np.float32)
    ref_w = None
    for scale in (1.0, 1e3, 1e6):
        arr = (base * scale).reshape(5, N, 1)
        images = [arr[i] for i in range(5)]
        result, sum_w, pct = _stack_winsorized_sigma_iter(
            images, None, kappa=3.0, winsor_limits=DEFAULT_LIMITS,
            return_weights=True,
        )
        assert pct == 0.0
        assert np.all(sum_w == 5.0)  # every column keeps all 5 samples
        if ref_w is None:
            ref_w = sum_w
        else:
            np.testing.assert_array_equal(sum_w, ref_w)


# ---------------------------------------------------------------------------
# R3 rework-2: sorted-values-derived median through the guard (odd/even/NaN/edges)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "values,expected_keep",
    [
        # even n_valid (4), gross HIGH outlier -> reject the high extreme
        ([100.0, 101.0, 99.0, 63000.0], [True, True, True, False]),
        # even n_valid (4), gross LOW outlier (negative) -> reject the low extreme
        ([-100.0, -100.0, -100.0, -63000.0], [True, True, True, False]),
        # odd n_valid (5), gross high outlier
        ([100.0, 101.0, 99.0, 102.0, 63000.0], [True, True, True, True, False]),
        # NaN partial (n_valid=4 over 5): NaN stays excluded, high outlier rejected
        ([100.0, 101.0, 99.0, np.nan, 63000.0], [True, True, True, False, False]),
        # zero median: no finite relative scale -> conservative keep (no rejection)
        ([0.0, 0.0, 0.0, 0.0, 5.0], [True, True, True, True, True]),
        # ties: no unique extreme -> no rejection
        ([100.0, 100.0, 100.0, 100.0, 100.0], [True, True, True, True, True]),
    ],
)
def test_gross_outlier_keep_median_extraction_edges(values, expected_keep):
    """The guard's per-column median (derived from the sorted valid values) must
    reproduce np.nanmedian semantics for odd/even valid counts, NaN-partial
    populations, zero median, negative values and ties."""
    col = np.asarray(values, dtype=np.float32).reshape(len(values), 1, 1)
    valid = ~np.isnan(col)
    keep = _gross_outlier_keep(col, valid)
    got = [bool(keep[i, 0, 0]) for i in range(len(values))]
    assert got == expected_keep, (values, got)


def test_gross_outlier_keep_all_invalid_column_contributes_nothing():
    """A fully-invalid column (n_valid=0) never rejects and contributes nothing."""
    col = np.full((5, 1, 1), np.nan, dtype=np.float32)
    valid = ~np.isnan(col)
    keep = _gross_outlier_keep(col, valid)
    assert int(np.count_nonzero(keep)) == 0
