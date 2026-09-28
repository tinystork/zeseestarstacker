"""C16 — orientation declaration in the freeze + resume comparison."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _install_stub(name):
    m = types.ModuleType(name)
    m.__path__ = []
    sys.modules[name] = m


def _load_by_path(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_PRE = dict(sys.modules)
for _n in ("seestar", "seestar.core", "seestar.calibration"):
    if _n not in sys.modules:
        _install_stub(_n)
port = _load_by_path(
    "seestar.core.calibration_port", "seestar/core/calibration_port.py"
)
adapter = _load_by_path(
    "seestar.calibration.zecalibrator_adapter",
    "seestar/calibration/zecalibrator_adapter.py",
)
resume = _load_by_path(
    "seestar.calibration.resume", "seestar/calibration/resume.py"
)
# Hermetic teardown: remove modules added by this loader AND restore any module
# the loader overwrote (so a second ``ProviderState`` enum never leaks into
# ``sys.modules`` and breaks the GUI tab probe's ``is`` identity check).
for _k in list(sys.modules):
    if _k not in _PRE:
        del sys.modules[_k]
    elif sys.modules[_k] is not _PRE[_k]:
        sys.modules[_k] = _PRE[_k]


def _freeze(orientation):
    return {
        "calibration_enabled": True,
        "calibration_provider": "zecalibrator",
        "calibration_api_version": "1.1",
        "calibration_product_version": "0.1.0",
        "calibration_library_fingerprint": "a" * 64,
        "calibration_contract_versions": {},
        "calibration_plan_map": {"sig": {"plan_id": "p1", "composition": {}}},
        "calibration_orientation_declaration": orientation,
    }


def test_orientation_declaration_divergence_refused():
    ok, field = resume.compare_calibration_freeze(
        _freeze("identity"), _freeze(None)
    )
    assert ok is False
    assert field == "calibration_orientation_declaration"


def test_orientation_declaration_identical_accepted():
    ok, field = resume.compare_calibration_freeze(
        _freeze("identity"), dict(_freeze("identity"))
    )
    assert ok is True and field is None


def test_provenance_renders_orientation_declaration():
    text = resume.render_calibration_provenance(_freeze("identity"), ())
    assert "orientation declaration: identity" in text


def test_provenance_omits_absent_declaration():
    text = resume.render_calibration_provenance(_freeze(None), ())
    assert "orientation declaration:" not in text
