"""C4 run-contract tests: schema v3 + calibration freeze + bounded v2->v3 migration."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

rc = importlib.import_module("seestar.run_contract")


def test_schema_version_is_three():
    assert rc.SCHEMA_VERSION == 3
    assert rc.Section.CALIBRATION == "calibration"
    assert rc.Section.CALIBRATION in rc.Section.ALL
    assert rc.Section.CALIBRATION not in rc.Section.REQUIRED


def test_calibration_fields_defined():
    names = {
        "calibration_enabled",
        "calibration_provider",
        "calibration_api_version",
        "calibration_product_version",
        "calibration_library_fingerprint",
        "calibration_contract_versions",
        "calibration_plan_map",
    }
    for name in names:
        fd = rc.field_def(name)
        assert fd.section == rc.Section.CALIBRATION
        assert fd.presence == rc.PRESENCE_OPTIONAL
        assert fd.qt_source is None  # runtime freeze, never settings-sourced
        assert not fd.backend_mapped  # never mapped to the engine


def test_migrate_v2_to_v3_bounded():
    v2 = {
        "schema_version": 2,
        "product_version": "8.5.4",
        "scientific_config": {},
        "execution_config": {},
        "provenance": {},
    }
    v3 = rc.migrate_v2_to_v3(v2)
    assert v3["schema_version"] == 3
    assert "calibration" not in v3  # absent == "calibration disabled", never invented

    with pytest.raises(rc.ValidationError):
        rc.migrate_v2_to_v3({"schema_version": 1})


def test_run_config_with_calibration_freeze_roundtrips():
    cfg = rc.RunConfig.from_sections(
        product_version="8.5.4",
        scientific={"stacking_mode": "mean"},
        calibration={
            "calibration_enabled": True,
            "calibration_provider": "zecalibrator",
            "calibration_api_version": "1.1",
            "calibration_product_version": "0.0.5",
            "calibration_library_fingerprint": "a" * 64,
            "calibration_contract_versions": {"science": 1},
            "calibration_plan_map": {
                "sig1": {"plan_id": "p1", "composition": {"level": "COMPLETE"}}
            },
        },
    )
    d = cfg.to_canonical_dict()
    assert d["schema_version"] == 3
    assert d["calibration"]["calibration_enabled"] is True
    assert "calibration_plan_map" in d["calibration"]


def test_run_config_without_calibration_omits_section():
    cfg = rc.RunConfig.from_sections(
        product_version="8.5.4", scientific={"stacking_mode": "mean"}
    )
    assert "calibration" not in cfg.to_canonical_dict()


def test_collect_from_settings_does_not_emit_calibration():
    # A bare settings object: no calibration fields may ever be collected from it.
    class _Bare:
        pass

    cfg = rc.collect_from_settings(_Bare(), product_version="8.5.4")
    assert cfg.calibration == {}
    assert "calibration" not in cfg.to_canonical_dict()


def test_read_cfg_auto_migrates_v2(tmp_path):
    v2 = {
        "schema_version": 2,
        "product_version": "8.5.4",
        "scientific_config": {},
        "execution_config": {},
        "provenance": {},
    }
    p = tmp_path / "v2.cfg"
    p.write_text(json.dumps(v2), encoding="utf-8")
    report = rc.read_cfg(str(p))
    assert report.config.calibration == {}


def test_read_cfg_rejects_altered_v2(tmp_path):
    # A corrupt/altered document (missing schema_version) fails cleanly.
    p = tmp_path / "bad.cfg"
    p.write_text(json.dumps({"product_version": "8.5.4"}), encoding="utf-8")
    with pytest.raises(rc.ValidationError):
        rc.read_cfg(str(p))
