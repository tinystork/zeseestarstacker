"""Deterministic property tests for the batching-dependence POC.

These pin the *measured* conclusions of
``research/registration_field_rotation/batch_dependence_poc.py`` so a future
refactor cannot silently flip the architecture finding.  Deterministic (fixed
seeds, closed-form fits); no ``seestar`` import.

The conclusions pinned here are the POC's *behavioural* evidence for the
global-reference audit (evolving vs immutable target), complementing the
structural AST guard in ``tests/test_global_reference_audit.py``.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "research", "registration_field_rotation"),
)

import batch_dependence_poc as poc  # noqa: E402


def _zero_mean():
    return poc.build_experiment(30, 200, 7, sigma=poc.SIGMA)


def _radial():
    return poc.build_experiment(30, 200, 7, sigma=poc.SIGMA, bias=poc.radial_bias)


# --------------------------------------------------------------------------
# zero-mean noise: no bias propagation
# --------------------------------------------------------------------------


def test_zero_mean_no_bias_propagation():
    P, fit, hold, T, src, biased = _zero_mean()
    A = poc.simulate("A", P, fit, hold, T, src, batch_size=1)
    B = poc.simulate("B", P, fit, hold, T, src, batch_size=1)
    # both reach the noise floor in true-global error
    assert A["hold_resid_true"].mean() < 0.1
    assert B["hold_resid_true"].mean() < 0.1
    # the evolving reference converges to truth (drift decreases), while the
    # immutable reference keeps its fixed noise
    assert B["ref_bias"][-1] < B["ref_bias"][0]
    assert B["ref_bias"][-1] < A["ref_bias"][-1]


def test_zero_mean_batch_size_30_equals_immutable():
    P, fit, hold, T, src, biased = _zero_mean()
    A = poc.simulate("A", P, fit, hold, T, src, batch_size=30)
    B = poc.simulate("B", P, fit, hold, T, src, batch_size=30)
    # batch_size == N means the reference is never rebuilt during processing:
    # B degenerates to A
    np.testing.assert_allclose(B["ref_bias"], A["ref_bias"], rtol=1e-12)
    np.testing.assert_allclose(B["hold_resid_true"], A["hold_resid_true"], rtol=1e-12)


# --------------------------------------------------------------------------
# adversarial radial bias: evolving target absorbs & hides it
# --------------------------------------------------------------------------


def test_radial_immutable_exposes_bias_in_fit_residual():
    P, fit, hold, T, src, biased = _radial()
    A = poc.simulate("A", P, fit, hold, T, src, batch_size=1)
    # the immutable target's fit residual is clearly above the noise floor
    # (the similarity fit cannot absorb the radial bias)
    assert A["hold_resid_target"].mean() > 0.2
    # and its reference stays unbiased (fixed noise realization)
    assert A["ref_bias"].mean() < 0.1


def test_radial_evolving_hides_bias_in_fit_residual_but_true_error_unchanged():
    P, fit, hold, T, src, biased = _radial()
    A = poc.simulate("A", P, fit, hold, T, src, batch_size=1)
    B = poc.simulate("B", P, fit, hold, T, src, batch_size=1)
    # the evolving target's fit residual collapses toward the noise floor ...
    assert B["hold_resid_target"][-1] < A["hold_resid_target"][-1] / 2
    # ... while the source->true-global error is unchanged (bias hidden, not removed)
    assert abs(B["hold_resid_true"].mean() - A["hold_resid_true"].mean()) < 0.01
    # and the reference catalogue drifts away from ground truth
    assert B["ref_bias"][-1] > B["ref_bias"][0] * 3


def test_radial_batch_size_dependence():
    P, fit, hold, T, src, biased = _radial()
    B1 = poc.simulate("B", P, fit, hold, T, src, batch_size=1)
    B30 = poc.simulate("B", P, fit, hold, T, src, batch_size=30)
    # updating every frame drifts the reference much more than never updating
    assert B1["ref_bias"].mean() > B30["ref_bias"].mean() * 3
    # but the final true-global error is identical (batch history affects the
    # reference, not the source->true mapping)
    assert abs(B1["hold_resid_true"].mean() - B30["hold_resid_true"].mean()) < 0.01


def test_radial_residual_spatially_structured_above_noise():
    P, fit, hold, T, src, biased = _radial()
    A = poc.simulate("A", P, fit, hold, T, src, batch_size=1)
    # the unmodelled radial bias leaves a spatially structured residual clearly
    # above the 0.05 px noise floor in every region (the exact centre/edge/
    # corner ordering depends on the bias-minus-best-similarity shape, so we
    # only pin the "clearly above noise" fact, not the ordering)
    assert A["corner"][-1] > 0.2
    assert A["edge"][-1] > 0.2
    assert A["centre"][-1] > 0.2


# --------------------------------------------------------------------------
# order dependence: a transient first-batch bias matters only for B
# --------------------------------------------------------------------------


def test_first_batch_contamination_order_dependence():
    P, fit, hold, T, src, biased = poc.build_experiment(
        30, 200, 7, sigma=poc.SIGMA, bias=poc.radial_bias, bias_frames=list(range(10))
    )
    B_nat = poc.simulate("B", P, fit, hold, T, src, batch_size=10)

    P2, fit2, hold2, T2, src2, biased2 = poc.build_experiment(
        30, 200, 7, sigma=poc.SIGMA, bias=poc.radial_bias,
        bias_frames=list(range(10)), order=np.arange(30)[::-1],
    )
    B_rev = poc.simulate("B", P2, fit2, hold2, T2, src2, batch_size=10)

    # same biased frames, different order -> different reference outcome (order
    # dependence) for the evolving target
    assert B_nat["ref_bias"][-1] > B_rev["ref_bias"][-1] * 2


def test_immutable_order_independent():
    P, fit, hold, T, src, biased = poc.build_experiment(
        30, 200, 7, sigma=poc.SIGMA, bias=poc.radial_bias, bias_frames=list(range(10))
    )
    A_nat = poc.simulate("A", P, fit, hold, T, src, batch_size=10)

    P2, fit2, hold2, T2, src2, biased2 = poc.build_experiment(
        30, 200, 7, sigma=poc.SIGMA, bias=poc.radial_bias,
        bias_frames=list(range(10)), order=np.arange(30)[::-1],
    )
    A_rev = poc.simulate("A", P2, fit2, hold2, T2, src2, batch_size=10)

    # the immutable reference is order-independent: its reference never drifts
    assert A_nat["ref_bias"][-1] < 0.1
    assert A_rev["ref_bias"][-1] < 0.1
    # and the mean true error is identical (same set of biased frames)
    assert abs(A_nat["hold_resid_true"].mean() - A_rev["hold_resid_true"].mean()) < 1e-9


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_deterministic_repeatable():
    P, fit, hold, T, src, biased = _radial()
    a = poc.simulate("B", P, fit, hold, T, src, batch_size=5)
    b = poc.simulate("B", P, fit, hold, T, src, batch_size=5)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
