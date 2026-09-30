"""C7 resume/provenance tests: freeze comparison + legacy resume + provenance dump."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

resume = importlib.import_module("seestar.calibration.resume")
rc = importlib.import_module("seestar.run_contract")


def _freeze(**kw) -> dict:
    return dict(kw)


def _enabled_freeze(fingerprint="a" * 64):
    return _freeze(
        calibration_enabled=True,
        calibration_provider="zecalibrator",
        calibration_api_version="1.1",
        calibration_product_version="0.0.5",
        calibration_library_fingerprint=fingerprint,
        calibration_contract_versions={"science": 1, "matching": 1, "provenance": 1},
        calibration_plan_map={
            "sig1": {"plan_id": "p1", "composition": {"level": "COMPLETE"}}
        },
    )


# ---------------------------------------------------------------------------
# (a) freeze identical -> resume accepted
# ---------------------------------------------------------------------------
def test_freeze_identical_resume_accepted():
    frozen = _enabled_freeze()
    ok, field = resume.compare_calibration_freeze(frozen, dict(frozen))
    assert ok is True and field is None


# ---------------------------------------------------------------------------
# (b) masters modified -> resume refused, naming the field
# ---------------------------------------------------------------------------
def test_masters_modified_resume_refused_naming_field():
    frozen = _enabled_freeze(fingerprint="a" * 64)
    current = _enabled_freeze(fingerprint="b" * 64)  # masters changed
    ok, field = resume.compare_calibration_freeze(frozen, current)
    assert ok is False
    assert field == "calibration_library_fingerprint"
    msg = resume.calibration_refusal_reason(field)
    assert "calibration_library_fingerprint" in msg
    assert "refusing resume" in msg


def test_provider_changed_refused_naming_field():
    frozen = _enabled_freeze()
    current = _enabled_freeze()
    current["calibration_provider"] = "other-provider"
    ok, field = resume.compare_calibration_freeze(frozen, current)
    assert ok is False and field == "calibration_provider"


def test_plan_map_changed_refused():
    frozen = _enabled_freeze()
    current = _enabled_freeze()
    current["calibration_plan_map"] = {"sig1": {"plan_id": "p2"}}
    ok, field = resume.compare_calibration_freeze(frozen, current)
    assert ok is False and field == "calibration_plan_map"


def test_both_disabled_accepted():
    ok, field = resume.compare_calibration_freeze({}, {})
    assert ok is True and field is None


def test_enabled_vs_disabled_refused():
    ok, field = resume.compare_calibration_freeze({"calibration_enabled": True}, {})
    assert ok is False and field == "calibration_enabled"


# ---------------------------------------------------------------------------
# (c) legacy v2 checkpoint -> interpreted, disabled, no exception
# ---------------------------------------------------------------------------
def test_legacy_v2_checkpoint_interpreted(tmp_path):
    v2 = {
        "schema_version": 2,
        "product_version": "8.5.4",
        "scientific_config": {},
        "execution_config": {},
        "provenance": {},
    }
    p = tmp_path / "v2.cfg"
    p.write_text(json.dumps(v2), encoding="utf-8")
    report = rc.read_cfg(str(p))  # auto-migrates v2 -> v3, no exception
    assert report.config.calibration == {}  # absent == "calibration disabled"
    ok, field = resume.compare_calibration_freeze(report.config.calibration, {})
    assert ok is True  # legacy (disabled) == disabled -> resume allowed


# ---------------------------------------------------------------------------
# (d) provenance dump (answers the §3 questions)
# ---------------------------------------------------------------------------
def test_provenance_dump_answers_questions():
    freeze = {
        "calibration_enabled": True,
        "calibration_provider": "zecalibrator",
        "calibration_api_version": "1.1",
        "calibration_product_version": "0.0.5",
        "calibration_library_fingerprint": "a" * 64,
        "calibration_contract_versions": {},
        "calibration_plan_map": {
            "sig1": {
                "plan_id": "p1",
                "composition": {
                    "applied_roles": ["dark"],
                    "level": "COMPLETE",
                    "additive_state": "dark_incl_bias",
                    "flat_applied": False,
                },
            }
        },
    }
    admissions = [
        {"role": "dark", "content_sha256": "deadbeef" * 8, "path": "/m/d.fits", "size_bytes": 1},
        {"role": "bias", "content_sha256": "cafebabe" * 8, "path": "/m/b.fits", "size_bytes": 1},
    ]
    text = resume.render_calibration_provenance(freeze, admissions)
    assert "calibration enabled: True" in text
    assert "provider: zecalibrator 0.0.5 (API 1.1)" in text
    assert "library fingerprint:" in text
    assert "dark:deadbeef" in text
    assert "bias:cafebabe" in text
    assert "flat applied: False" in text
    assert "additive applied: dark_incl_bias" in text
    assert "applied roles: dark" in text


def test_provenance_disabled():
    text = resume.render_calibration_provenance({}, [])
    assert "calibration enabled: False" in text
    assert "disabled" in text
