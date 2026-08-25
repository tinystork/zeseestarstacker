"""Deterministic property tests for the RF-2 registration-reference lifecycle POC.

These pin the *measured* conclusions of
``research/registration_reference_rf2/registration_lifecycle.py`` so a future
refactor cannot silently flip the architecture finding.  Deterministic (fixed
seeds, closed-form fits); no ``seestar`` import.

Corrective C1 framing: the primary stable candidate is the **immutable
initially-selected reference** (a single frame held constant).  The previous
"first-batch freeze" candidate is now ``freeze_first_batch`` and is tested as an
**explored, rejected** candidate — it must *fail* the organization-invariance
contract, while the immutable target must *pass* it.
"""

import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "research", "registration_reference_rf2"),
)

import registration_lifecycle as poc  # noqa: E402


def _zero_mean():
    return poc.build_observations(30, 200, 7)


def _translation():
    return poc.build_observations(30, 200, 7, bias=poc.translation_bias)


def _radial():
    return poc.build_observations(30, 200, 7, bias=poc.radial_bias)


def _transient_radial(order=None):
    return poc.build_observations(
        30, 200, 7, bias=poc.radial_bias, bias_frames=list(range(10)), order=order
    )


# --------------------------------------------------------------------------
# zero-mean noise: no bias propagation; stacked target has SNR benefit only
# --------------------------------------------------------------------------


def test_zero_mean_no_propagation_all_strategies():
    P, fit, hold, T, src, _, _ = _zero_mean()
    ref = poc.build_reference(P, seed=0)
    im = poc.simulate("immutable", P, fit, hold, T, src, batch_size=1, reference=ref)
    st = poc.simulate("freeze_first_batch", P, fit, hold, T, src, batch_size=10, reference=ref)
    ev = poc.simulate("evolving", P, fit, hold, T, src, batch_size=1, reference=ref)
    for r in (im, st, ev):
        assert r["true_err"].mean() < 0.1, "true-global error must stay at the noise floor"


def test_zero_mean_stacked_target_lower_fit_residual_but_same_true_error():
    P, fit, hold, T, src, _, _ = _zero_mean()
    ref = poc.build_reference(P, seed=0)
    im = poc.simulate("immutable", P, fit, hold, T, src, batch_size=1, reference=ref)
    st = poc.simulate("freeze_first_batch", P, fit, hold, T, src, batch_size=10, reference=ref)
    # a stacked (higher-SNR) reference has a lower target-fit residual than a
    # single-frame reference ...
    assert st["fit_resid"].mean() < im["fit_resid"].mean()
    # ... but the true-global error is unchanged (the lower residual is a
    # target-SNR effect, not an accuracy gain)
    assert abs(st["true_err"].mean() - im["true_err"].mean()) < 1e-3


def test_zero_mean_evolving_reference_converges():
    P, fit, hold, T, src, _, _ = _zero_mean()
    ref = poc.build_reference(P, seed=0)
    ev = poc.simulate("evolving", P, fit, hold, T, src, batch_size=1, reference=ref)
    assert ev["ref_bias"][-1] < ev["ref_bias"][0], "evolving reference converges to truth"


# --------------------------------------------------------------------------
# representable bias (translation) is CORRECTED, not a risk
# --------------------------------------------------------------------------


def test_representable_translation_bias_is_corrected():
    P, fit, hold, T, src, _, _ = _translation()
    ref = poc.build_reference(P, seed=0)
    for strategy in ("immutable", "freeze_first_batch", "evolving"):
        r = poc.simulate(strategy, P, fit, hold, T, src, batch_size=1, reference=ref)
        # the 0.5 px translation is absorbed into the per-frame transform, so
        # the true-global error stays at the noise floor (NOT ~0.5 px)
        assert r["true_err"].mean() < 0.1, f"{strategy} true error should be at noise floor"


# --------------------------------------------------------------------------
# non-representable bias (radial): hidden by drifting target, exposed by immutable
# --------------------------------------------------------------------------


def test_radial_immutable_exposes_bias():
    P, fit, hold, T, src, _, _ = _radial()
    ref = poc.build_reference(P, seed=0)
    im = poc.simulate("immutable", P, fit, hold, T, src, batch_size=1, reference=ref)
    # fit residual == true error == bias magnitude (well above noise)
    assert im["fit_resid"].mean() > 2.0
    assert abs(im["fit_resid"].mean() - im["true_err"].mean()) < 0.01
    assert im["ref_bias"].mean() < 0.1  # reference stays unbiased


def test_radial_evolving_hides_bias_but_true_error_unchanged():
    P, fit, hold, T, src, _, _ = _radial()
    ref = poc.build_reference(P, seed=0)
    im = poc.simulate("immutable", P, fit, hold, T, src, batch_size=1, reference=ref)
    ev = poc.simulate("evolving", P, fit, hold, T, src, batch_size=1, reference=ref)
    # the evolving target's fit residual collapses to the noise floor ...
    assert ev["fit_resid"][-1] < im["fit_resid"][-1] / 2
    # ... while the true-global error is unchanged (bias hidden, not removed)
    assert abs(ev["true_err"].mean() - im["true_err"].mean()) < 0.01
    # and the reference drifts away from ground truth
    assert ev["ref_bias"][-1] > ev["ref_bias"][0] * 3


def test_radial_freeze_first_batch_also_hides_bias_after_freeze():
    P, fit, hold, T, src, _, _ = _radial()
    ref = poc.build_reference(P, seed=0)
    st = poc.simulate("freeze_first_batch", P, fit, hold, T, src, batch_size=1, reference=ref)
    # freezing the first batch bakes its bias into the reference: the fit
    # residual collapses and the reference drifts, while the true error stays
    assert st["ref_bias"][-1] > st["ref_bias"][0] * 3
    assert st["true_err"].mean() > 2.0


# --------------------------------------------------------------------------
# batch-size behaviour
# --------------------------------------------------------------------------


def test_freeze_first_batch_reference_constant_after_freeze():
    P, fit, hold, T, src, _, _ = _zero_mean()
    ref = poc.build_reference(P, seed=0)
    bs = 10
    st = poc.simulate("freeze_first_batch", P, fit, hold, T, src, batch_size=bs, reference=ref)
    # the reference IS constant after the freeze point (a true but insufficient
    # property — it does NOT imply batch-size invariance of the *identity*)
    frozen = st["ref_bias"][bs:]
    assert np.allclose(frozen, frozen[0], rtol=1e-12, atol=1e-12)


def test_evolving_reference_batch_size_dependent():
    P, fit, hold, T, src, _, _ = _radial()
    ref = poc.build_reference(P, seed=0)
    ev1 = poc.simulate("evolving", P, fit, hold, T, src, batch_size=1, reference=ref)
    ev30 = poc.simulate("evolving", P, fit, hold, T, src, batch_size=30, reference=ref)
    # updating every frame drifts the reference much more than never updating
    assert ev1["ref_bias"].mean() > ev30["ref_bias"].mean() * 3
    # the true-global error is batch-size independent
    assert abs(ev1["true_err"].mean() - ev30["true_err"].mean()) < 0.01


# --------------------------------------------------------------------------
# order invariance of the reference
# --------------------------------------------------------------------------


def test_evolving_reference_order_dependent():
    Pn, fn, hn, Tn, sn, _, _ = _transient_radial(None)
    refn = poc.build_reference(Pn, seed=0)
    A_nat = poc.simulate("evolving", Pn, fn, hn, Tn, sn, batch_size=10, reference=refn)
    Pr, fr, hr, Tr, sr, _, _ = _transient_radial(np.arange(30)[::-1])
    refr = poc.build_reference(Pr, seed=0)
    A_rev = poc.simulate("evolving", Pr, fr, hr, Tr, sr, batch_size=10, reference=refr)
    # biased-first contaminates the reference; biased-last does not
    assert A_nat["ref_bias"][-1] > A_rev["ref_bias"][-1] * 2


def test_freeze_first_batch_reference_order_dependent_via_first_batch():
    Pn, fn, hn, Tn, sn, _, _ = _transient_radial(None)
    refn = poc.build_reference(Pn, seed=0)
    B_nat = poc.simulate("freeze_first_batch", Pn, fn, hn, Tn, sn, batch_size=10, reference=refn)
    Pr, fr, hr, Tr, sr, _, _ = _transient_radial(np.arange(30)[::-1])
    refr = poc.build_reference(Pr, seed=0)
    B_rev = poc.simulate("freeze_first_batch", Pr, fr, hr, Tr, sr, batch_size=10, reference=refr)
    assert B_nat["ref_bias"][-1] > B_rev["ref_bias"][-1] * 2


def test_immutable_reference_order_independent():
    Pn, fn, hn, Tn, sn, _, _ = _transient_radial(None)
    refn = poc.build_reference(Pn, seed=0)
    A0_nat = poc.simulate("immutable", Pn, fn, hn, Tn, sn, batch_size=10, reference=refn)
    Pr, fr, hr, Tr, sr, _, _ = _transient_radial(np.arange(30)[::-1])
    refr = poc.build_reference(Pr, seed=0)
    A0_rev = poc.simulate("immutable", Pr, fr, hr, Tr, sr, batch_size=10, reference=refr)
    assert A0_nat["ref_bias"][-1] < 0.1
    assert A0_rev["ref_bias"][-1] < 0.1
    # mean true error identical across orders (same set of biased frames)
    assert abs(A0_nat["true_err"].mean() - A0_rev["true_err"].mean()) < 1e-9


# --------------------------------------------------------------------------
# geometry invariance measured on TRANSFORMS (the corrective-C1 acceptance gate)
# --------------------------------------------------------------------------


def test_immutable_transforms_batch_invariant():
    """Same preselected reference under >=3 batch sizes -> per-frame transforms
    are identical (after reindexing by frame ID)."""
    P, fit, hold, T, src, _, fids = _radial()
    ref = poc.build_reference(P, seed=0)
    runs = [
        poc.simulate("immutable", P, fit, hold, T, src, batch_size=bs, reference=ref, frame_ids=fids)
        for bs in (1, 5, 10)
    ]
    assert poc.transforms_max_abs_diff(runs, fids) == 0.0
    # the reference identity is unchanged by construction
    assert all(r["ref_constant"] for r in runs)


def test_immutable_transforms_order_invariant():
    """Same preselected reference under >=2 orders -> per-frame transforms keyed
    by frame ID are identical."""
    Pn, fn, hn, Tn, sn, _, fidsn = _radial()
    refn = poc.build_reference(Pn, seed=0)
    nat = poc.simulate("immutable", Pn, fn, hn, Tn, sn, batch_size=10, reference=refn, frame_ids=fidsn)
    Pr, fr, hr, Tr, sr, _, fidsr = poc.build_observations(
        30, 200, 7, bias=poc.radial_bias, order=np.arange(30)[::-1]
    )
    refr = poc.build_reference(Pr, seed=0)
    rev = poc.simulate("immutable", Pr, fr, hr, Tr, sr, batch_size=10, reference=refr, frame_ids=fidsr)
    assert poc._order_diff(nat, rev) == 0.0


def test_freeze_first_batch_fails_batch_invariance():
    """The first-batch-freeze target's identity depends on batch_size, so its
    per-frame transforms differ across batch sizes."""
    P, fit, hold, T, src, _, fids = _radial()
    ref = poc.build_reference(P, seed=0)
    runs = [
        poc.simulate("freeze_first_batch", P, fit, hold, T, src, batch_size=bs, reference=ref, frame_ids=fids)
        for bs in (1, 5, 10)
    ]
    assert poc.transforms_max_abs_diff(runs, fids) > 1e-9


def test_freeze_first_batch_fails_order_invariance():
    """The first-batch-freeze target's identity depends on which frames are
    first, so per-frame transforms differ across orders."""
    Pn, fn, hn, Tn, sn, _, fidsn = _radial()
    refn = poc.build_reference(Pn, seed=0)
    nat = poc.simulate("freeze_first_batch", Pn, fn, hn, Tn, sn, batch_size=10, reference=refn, frame_ids=fidsn)
    Pr, fr, hr, Tr, sr, _, fidsr = poc.build_observations(
        30, 200, 7, bias=poc.radial_bias, order=np.arange(30)[::-1]
    )
    refr = poc.build_reference(Pr, seed=0)
    rev = poc.simulate("freeze_first_batch", Pr, fr, hr, Tr, sr, batch_size=10, reference=refr, frame_ids=fidsr)
    assert poc._order_diff(nat, rev) > 1e-9


def test_immutable_reference_identity_constant_across_runs():
    """The immutable target's identity is the same array across runs (not just
    the same seed)."""
    P, fit, hold, T, src, _, fids = _radial()
    ref = poc.build_reference(P, seed=0)
    r = poc.simulate("immutable", P, fit, hold, T, src, batch_size=10, reference=ref, frame_ids=fids)
    assert r["ref_constant"] is True
    assert np.array_equal(r["ref_final"], ref)


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_deterministic_repeatable():
    P, fit, hold, T, src, _, fids = _radial()
    ref = poc.build_reference(P, seed=0)
    a = poc.simulate("evolving", P, fit, hold, T, src, batch_size=5, reference=ref, frame_ids=fids)
    b = poc.simulate("evolving", P, fit, hold, T, src, batch_size=5, reference=ref, frame_ids=fids)
    for k in a:
        if k in ("failure_rate", "runtime_s", "transforms", "frame_ids", "ref_final", "ref_constant"):
            continue
        np.testing.assert_array_equal(a[k], b[k])
    for fid in fids:
        np.testing.assert_array_equal(a["transforms"][int(fid)], b["transforms"][int(fid)])
