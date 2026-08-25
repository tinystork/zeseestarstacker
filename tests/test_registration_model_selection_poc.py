"""Deterministic property tests for the RF-1 model-selection POC.

These tests pin the *quantitative conclusions* of
``research/registration_field_rotation/model_selection_poc.py`` so that a
future refactor cannot silently change the measured facts.  They are
deterministic (fixed seeds, closed-form fits) and do **not** import
``seestar``.

They deliberately do **not** embed a model *decision* (e.g. "current model is
sufficient").  The decision — ``FURTHER DATA REQUIRED`` — follows from the
*measured* facts (below) combined with the unmeasured real cross-session scale
drift, and lives in ``docs/registration_field_rotation_research.md``.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "research",
        "registration_field_rotation",
    ),
)

import model_selection_poc as poc  # noqa: E402

NOISE_FLOOR = 0.30  # px — generous bound for the 0.05 px centroid noise floor


def _hold_rms(run, model):
    return run["results"][model]["hold_rms"]


def _region_mean(run, model, region):
    return run["results"][model][region]


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_deterministic_repeatable():
    a = poc.run_all()
    b = poc.run_all()
    for ra, rb in zip(a, b):
        assert ra["name"] == rb["name"]
        for m in poc.MODELS:
            for k in ("hold_rms", "hold_p50", "hold_p95", "fit_rms"):
                va, vb = ra["results"][m][k], rb["results"][m][k]
                assert (np.isnan(va) and np.isnan(vb)) or va == vb


# --------------------------------------------------------------------------
# translation: every model reaches the noise floor (no overfitting penalty)
# --------------------------------------------------------------------------


def test_translation_all_models_at_noise_floor():
    run = poc.run_scenario("translation", poc.SCENARIOS["translation"])
    for m in poc.MODELS:
        assert _hold_rms(run, m) < NOISE_FLOOR, f"{m} over/under-fits translation"


# --------------------------------------------------------------------------
# rotation / rotation+translation / large rotation / partial overlap:
# translation is insufficient; euclidean (current model) reaches the noise floor
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["rotation", "rotation_translation", "large_rotation", "partial_overlap"]
)
def test_rotation_scenarios_euclidean_reaches_noise_floor(name):
    run = poc.run_scenario(name, poc.SCENARIOS[name])
    # translation cannot represent rotation
    assert _hold_rms(run, "translation") > 10.0, name
    # the current euclidean model reaches the noise floor
    assert _hold_rms(run, "euclidean") < NOISE_FLOOR, name
    # and no more-flexible model is meaningfully better on rigid geometry
    for m in ("similarity", "affine", "projective", "poly3"):
        assert _hold_rms(run, m) < NOISE_FLOOR, f"{name}/{m}"


# --------------------------------------------------------------------------
# scale: euclidean (scale=1) leaves a corner residual; similarity recovers it
# --------------------------------------------------------------------------


def test_scale_scenario_similarity_recovers_scale():
    run = poc.run_scenario("scale", poc.SCENARIOS["scale"])
    # the current model forces scale=1 and leaves an edge/corner residual
    assert _region_mean(run, "euclidean", "corner") > 1.5
    assert _region_mean(run, "euclidean", "centre") < _region_mean(
        run, "euclidean", "corner"
    )
    # similarity removes the scale residual
    assert _hold_rms(run, "similarity") < NOISE_FLOOR
    # and the recovered scale matches the injected 1.003
    assert abs(poc.recovered_similarity_scale() - 1.003) < 1e-4


def test_scale_corner_error_is_quantified():
    """Pin the quantitative fact behind FURTHER DATA REQUIRED: a 0.3% uniform
    scale drift, if it were real, would leave a ~2-3 px corner residual under
    the current scale=1 model.  (This is a measurement, not a decision.)"""
    run = poc.run_scenario("scale", poc.SCENARIOS["scale"])
    corner = _region_mean(run, "euclidean", "corner")
    assert 2.0 < corner < 3.5, f"corner residual changed: {corner}"


# --------------------------------------------------------------------------
# affine / projective: only the matching-or-more-complex model recovers it
# --------------------------------------------------------------------------


def test_affine_scenario_needs_affine():
    run = poc.run_scenario("affine", poc.SCENARIOS["affine"])
    assert _hold_rms(run, "similarity") > 0.5  # shear/anisotropic scale unmodeled
    assert _hold_rms(run, "affine") < NOISE_FLOOR
    assert _hold_rms(run, "projective") < NOISE_FLOOR
    assert _hold_rms(run, "poly3") < NOISE_FLOOR


def test_projective_scenario_needs_projective():
    run = poc.run_scenario("projective", poc.SCENARIOS["projective"])
    assert _hold_rms(run, "affine") > 0.3  # perspective unmodeled
    assert _hold_rms(run, "projective") < NOISE_FLOOR
    assert _hold_rms(run, "poly3") < NOISE_FLOOR


# --------------------------------------------------------------------------
# smooth radial distortion: a degree-3 (cubic) polynomial is required; the
# current model leaves a structured (radial) residual
# --------------------------------------------------------------------------


def test_smooth_local_radial_residual_structure():
    run = poc.run_scenario("smooth_local", poc.SCENARIOS["smooth_local"])
    # the current model leaves the largest residual, and it is corner-heavy
    assert _hold_rms(run, "euclidean") > _hold_rms(run, "similarity")
    assert _region_mean(run, "euclidean", "corner") > _region_mean(
        run, "euclidean", "centre"
    )
    # the order-3 polynomial (which spans the cubic monomials x^3, x*y^2) is
    # the smooth candidate that represents the injected r^3 field
    assert _hold_rms(run, "poly3") < _hold_rms(run, "euclidean")


def test_smooth_local_poly3_reaches_noise_floor():
    """The corrected smooth candidate (degree-3 polynomial) represents the
    injected radial r^3 distortion, which a degree-2 polynomial cannot."""
    run = poc.run_scenario("smooth_local", poc.SCENARIOS["smooth_local"])
    assert _hold_rms(run, "poly3") < NOISE_FLOOR, "poly3 should span the cubic field"


# --------------------------------------------------------------------------
# outliers: the current model + deterministic MAD rejection is robust
# --------------------------------------------------------------------------


def test_outliers_robustness():
    run = poc.run_scenario("outliers", poc.SCENARIOS["outliers"])
    assert run["n_outliers"] > 0
    # translation cannot recover the (rotated) geometry at all
    assert _hold_rms(run, "translation") > 10.0
    # rigid and low-order models recover to the noise floor after rejection
    for m in ("euclidean", "similarity", "affine", "projective"):
        assert _hold_rms(run, m) < NOISE_FLOOR, f"{m} not robust to outliers"
        assert run["results"][m]["n_inliers"] is not None


# --------------------------------------------------------------------------
# degenerate: minimum-point requirements are respected (failure behaviour)
# --------------------------------------------------------------------------


def test_degenerate_min_points():
    run = poc.run_scenario("degenerate", poc.SCENARIOS["degenerate"])
    # 3 stars -> 2 in the fit set
    assert run["n_fit"] == 2
    # translation fits with 2 points; affine/projective/poly3 cannot
    assert run["results"]["translation"]["fit_ok"] is True
    for m in ("affine", "projective", "poly3"):
        assert run["results"][m]["fit_ok"] is False, m


# --------------------------------------------------------------------------
# smooth-model metadata: dof / min points are stated correctly
# --------------------------------------------------------------------------


def test_poly3_metadata():
    assert "poly2" not in poc.MODELS  # the inadequate degree-2 candidate is gone
    assert poc.MODEL_INFO["poly3"]["dof"] == 20
    assert poc.MIN_POINTS["poly3"] == 10
