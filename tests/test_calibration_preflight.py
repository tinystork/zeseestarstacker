"""C4 preflight unit tests: flat policy, freeze payload, light signature.

Pure functions only (no ZeCalibrator, no Qt): the preflight module is loaded by
file path under package stubs, mirroring ``tests/test_zecalibrator_adapter.py``.
The full end-to-end chain (real ZeCalibrator) is exercised by the C4 demo.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_PREEXISTING = dict(sys.modules)


def _install_package_stub(name: str) -> None:
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod


def _load_by_path(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for _name in ("seestar", "seestar.core", "seestar.calibration"):
    if _name not in sys.modules:
        _install_package_stub(_name)

port = _load_by_path(
    "seestar.core.calibration_port", "seestar/core/calibration_port.py"
)
adapter = _load_by_path(
    "seestar.calibration.zecalibrator_adapter",
    "seestar/calibration/zecalibrator_adapter.py",
)
preflight = _load_by_path(
    "seestar.calibration.preflight", "seestar/calibration/preflight.py"
)

for _key in list(sys.modules.keys()):
    if _key not in _PREEXISTING:
        del sys.modules[_key]
    elif sys.modules[_key] is not _PREEXISTING[_key]:
        sys.modules[_key] = _PREEXISTING[_key]


# ---------------------------------------------------------------------------
# Flat policy (mission §6)
# ---------------------------------------------------------------------------
def test_flat_policy_all_compatible_applies():
    assert preflight.flat_policy([True, True]) == (True, "")


def test_flat_policy_empty():
    assert preflight.flat_policy([]) == (False, "")


def test_flat_policy_mixed_turns_flat_off_with_message():
    flat, msg = preflight.flat_policy([True, False])
    assert flat is False
    assert "No common compatible flat" in msg


def test_flat_policy_none_compatible_turns_flat_off_with_message():
    flat, msg = preflight.flat_policy([False, False])
    assert flat is False
    assert "No common compatible flat" in msg


# ---------------------------------------------------------------------------
# Freeze payload (JSON-safe, 7 run-contract fields)
# ---------------------------------------------------------------------------
def test_build_freeze_empty_when_unavailable():
    assert preflight.build_freeze(preflight.PreflightResult(available=False)) == {}


def test_build_freeze_with_plan_entries():
    entries = (
        preflight.LightPlanEntry(
            signature="a" * 64,
            plan_id="plan-1",
            composition={
                "applied_roles": ["dark"],
                "level": "COMPLETE",
                "additive_state": "dark_incl_bias",
                "flat_applied": False,
            },
        ),
    )
    result = preflight.PreflightResult(
        available=True,
        fingerprint="f" * 64,
        provider_id="zecalibrator",
        product_version="0.0.5",
        api_version="1.1",
        plan_entries=entries,
    )
    freeze = preflight.build_freeze(result)
    assert freeze["calibration_enabled"] is True
    assert freeze["calibration_provider"] == "zecalibrator"
    assert freeze["calibration_library_fingerprint"] == "f" * 64
    plan_map = freeze["calibration_plan_map"]
    assert plan_map["a" * 64]["plan_id"] == "plan-1"
    assert plan_map["a" * 64]["composition"]["additive_state"] == "dark_incl_bias"
    # JSON-safe (no provider objects).
    import json

    json.dumps(freeze)


# ---------------------------------------------------------------------------
# Light signature (stable content hash, never a filename heuristic)
# ---------------------------------------------------------------------------
def test_light_signature_stable(tmp_path):
    p = tmp_path / "light.fits"
    p.write_bytes(b"some light bytes")
    assert preflight.light_signature(str(p)) == preflight.light_signature(str(p))
    assert len(preflight.light_signature(str(p))) == 64


def test_light_signature_differs_by_content(tmp_path):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    a.write_bytes(b"aaaa")
    b.write_bytes(b"bbbb")
    assert preflight.light_signature(str(a)) != preflight.light_signature(str(b))


# ---------------------------------------------------------------------------
# Preflight with an unavailable provider -> available=False (never raises)
# ---------------------------------------------------------------------------
def test_preflight_unavailable_provider():
    class _FakeProvider:
        def probe(self):
            return port.ProviderInfo(
                state=port.ProviderState.NOT_INSTALLED, message="absent"
            )

    result = preflight.preflight_calibration("/masters", ["/l.fits"], provider=_FakeProvider())
    assert result.available is False
