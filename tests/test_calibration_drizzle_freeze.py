"""C19 — Drizzle calibration freeze: config section + resume refusal.

Verifies that the Drizzle canonical config carries the frozen calibration
signature into the ``calibration`` section (absent == disabled), and that the
resume comparison names the diverging field (hard refusal).
"""

from __future__ import annotations

import importlib

import pytest

rc = importlib.import_module("seestar.run_contract")
resume = importlib.import_module("seestar.calibration.resume")
drizzle_checkpoint = importlib.import_module("seestar.core.drizzle_checkpoint")


def _fake_qm(calibration_freeze):
    class _Qm:
        pass

    qm = _Qm()
    qm.weighting_method = "none"
    qm.use_quality_weighting = False
    qm.weight_by_snr = True
    qm.weight_by_stars = True
    qm.snr_exponent = 1.0
    qm.stars_exponent = 0.5
    qm.min_weight = 0.01
    qm.correct_hot_pixels = True
    qm.hot_pixel_threshold = 3.0
    qm.neighborhood_size = 5
    qm.bayer_pattern = "GRBG"
    qm.drizzle_scale = 1.0
    qm.drizzle_kernel = "square"
    qm.drizzle_pixfrac = 1.0
    qm.drizzle_wht_threshold_effective = 0.0
    qm.drizzle_fillval = "0.0"
    qm._calibration_freeze = calibration_freeze
    return qm


def _freeze(**kw) -> dict:
    base = {
        "calibration_enabled": True,
        "calibration_provider": "zecalibrator",
        "calibration_api_version": "1.0",
        "calibration_product_version": "1.2.2",
        "calibration_library_fingerprint": "a" * 64,
        "calibration_contract_versions": {},
        "calibration_plan_map": {},
        "calibration_orientation_declaration": "identity",
    }
    base.update(kw)
    return base


def test_drizzle_config_carries_calibration_section():
    freeze = _freeze(
        calibration_plan_map={
            "sig": {"plan_id": "p1", "composition": {"applied_roles": ["dark"]}}
        }
    )
    cfg = drizzle_checkpoint.build_drizzle_canonical_config(
        _fake_qm(freeze), product_version="8.2.0"
    )
    cal = cfg.calibration
    assert cal.get("calibration_enabled") is True
    assert cal.get("calibration_orientation_declaration") == "identity"
    assert cal.get("calibration_library_fingerprint") == "a" * 64
    assert cal.get("calibration_plan_map") == freeze["calibration_plan_map"]
    # round-trips through the canonical dict (schema v3 keeps the section)
    assert rc.Section.CALIBRATION in cfg.to_canonical_dict()


def test_drizzle_config_omits_calibration_when_disabled():
    cfg = drizzle_checkpoint.build_drizzle_canonical_config(
        _fake_qm({}), product_version="8.2.0"
    )
    assert cfg.calibration == {}
    assert rc.Section.CALIBRATION not in cfg.to_canonical_dict()


def test_drizzle_resume_refusal_names_orientation_field():
    frozen = _freeze()
    current = _freeze(calibration_orientation_declaration="flip")  # divergence
    ok, field = resume.compare_calibration_freeze(frozen, current)
    assert ok is False
    assert field == "calibration_orientation_declaration"
    msg = resume.calibration_refusal_reason(field)
    assert "calibration_orientation_declaration" in msg
    assert "refusing resume" in msg


def test_drizzle_resume_refusal_names_fingerprint_field():
    frozen = _freeze()
    current = _freeze(calibration_library_fingerprint="b" * 64)  # masters changed
    ok, field = resume.compare_calibration_freeze(frozen, current)
    assert ok is False
    assert field == "calibration_library_fingerprint"


def test_drizzle_resume_identical_accepted():
    frozen = _freeze()
    ok, field = resume.compare_calibration_freeze(frozen, dict(frozen))
    assert ok is True
    assert field is None


def test_drizzle_resume_both_disabled_accepted():
    ok, field = resume.compare_calibration_freeze({}, {})
    assert ok is True
    assert field is None
