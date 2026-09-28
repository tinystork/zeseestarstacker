"""C11 — acquisition-signature plan map: per-class freeze, not per-frame.

Verifies the key change: the freeze is keyed by the **acquisition signature**
(header-only), the preflight resolves ONE representative per class (1 decode per
class, not one per frame), and a composition divergence at identical masters is
actually detected (the C10 gap).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

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
resume = _load_by_path(
    "seestar.calibration.resume", "seestar/calibration/resume.py"
)

for _key in list(sys.modules.keys()):
    if _key not in _PREEXISTING:
        del sys.modules[_key]
    elif sys.modules[_key] is not _PREEXISTING[_key]:
        sys.modules[_key] = _PREEXISTING[_key]


# ---------------------------------------------------------------------------
# acquisition signature (header-only key)
# ---------------------------------------------------------------------------
def _write_light(path, *, exptime=60.0, gain=120.0, bayer="MONO",
                 instrume="Seestar S50", shape=(40, 32)):
    hdu = fits.PrimaryHDU(np.zeros(shape, dtype=np.int16))
    hdu.header["IMAGETYP"] = "Light"
    hdu.header["EXPTIME"] = float(exptime)
    hdu.header["GAIN"] = float(gain)
    hdu.header["XBINNING"] = 1
    hdu.header["YBINNING"] = 1
    hdu.header["BAYERPAT"] = bayer
    hdu.header["INSTRUME"] = instrume
    hdu.header["NAXIS"] = 2
    hdu.writeto(str(path), overwrite=True)


def test_acquisition_signature_stable(tmp_path):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    _write_light(a, exptime=60.0)
    _write_light(b, exptime=60.0)
    assert preflight.acquisition_signature(str(a)) == preflight.acquisition_signature(str(b))


def test_acquisition_signature_discriminates_exposure(tmp_path):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    _write_light(a, exptime=60.0)
    _write_light(b, exptime=180.0)
    assert preflight.acquisition_signature(str(a)) != preflight.acquisition_signature(str(b))


def test_acquisition_signature_discriminates_binning(tmp_path):
    a = tmp_path / "a.fits"
    _write_light(a, exptime=60.0)
    sig = preflight.acquisition_signature(str(a))
    # change binning -> different signature
    with fits.open(a) as hdul:
        hdul[0].header["XBINNING"] = 2
        hdul[0].header["YBINNING"] = 2
        hdul.writeto(str(a), overwrite=True)
    assert preflight.acquisition_signature(str(a)) != sig


# ---------------------------------------------------------------------------
# composition divergence -> resume refused (the C10 gap, now closed)
# ---------------------------------------------------------------------------
def _freeze(plan_map):
    return {
        "calibration_enabled": True,
        "calibration_provider": "zecalibrator",
        "calibration_api_version": "1.1",
        "calibration_product_version": "0.0.5",
        "calibration_library_fingerprint": "a" * 64,
        "calibration_contract_versions": {},
        "calibration_plan_map": plan_map,
    }


def test_composition_divergence_refused_naming_plan_map():
    # Same masters fingerprint, but the effective composition changed (e.g. a
    # flat became admissible) -> the resume must refuse naming the plan_map.
    frozen = _freeze({
        "60.0|120.0|1x1|MONO|SEESTAR S50|32x40": {
            "plan_id": "p1",
            "composition": {"applied_roles": ["dark"], "additive_state": "dark_incl_bias",
                            "flat_applied": False, "level": "COMPLETE"},
        }
    })
    current = _freeze({
        "60.0|120.0|1x1|MONO|SEESTAR S50|32x40": {
            "plan_id": "p2",
            "composition": {"applied_roles": ["dark", "flat"], "additive_state": "dark_incl_bias",
                            "flat_applied": True, "level": "COMPLETE"},
        }
    })
    ok, field = resume.compare_calibration_freeze(frozen, current)
    assert ok is False
    assert field == "calibration_plan_map"


def test_identical_plan_map_accepted():
    frozen = _freeze({"sig": {"plan_id": "p1", "composition": {"flat_applied": True}}})
    ok, field = resume.compare_calibration_freeze(frozen, dict(frozen))
    assert ok is True and field is None


# ---------------------------------------------------------------------------
# preflight resolves ONE representative per acquisition class (1 decode/class)
# ---------------------------------------------------------------------------
def test_preflight_resolves_one_representative_per_class(tmp_path):
    # 4 lights across 2 acquisition classes (60s and 180s).
    lights = []
    for i in range(2):
        p = tmp_path / f"l60_{i}.fits"
        _write_light(p, exptime=60.0)
        lights.append(str(p))
    for i in range(2):
        p = tmp_path / f"l180_{i}.fits"
        _write_light(p, exptime=180.0)
        lights.append(str(p))

    calls = {"n": 0}

    class _Plan:
        plan_id = "plan-1"

    class _Comp:
        def __init__(self):
            self.applied_roles = ("dark",)
            self.additive_state = "dark_incl_bias"
            self.flat_applied = False
            self.level = "COMPLETE"

        def to_dict(self):
            return {"applied_roles": ["dark"], "additive_state": "dark_incl_bias",
                    "flat_applied": False, "level": "COMPLETE"}

    class _Session:
        def resolve_light(self, source):
            calls["n"] += 1
            return port.RouteResolution(
                state=port.CalibrationState.COMPLETED,
                outcome="MATCHED",
                plan=port.CalibrationPlan("plan-1", object()),
                composition=port.CalibrationComposition(
                    applied_roles=("dark",), level="COMPLETE",
                    additive_state="dark_incl_bias", flat_applied=False,
                ),
            )

    class _Provider:
        name = "fake"

        def probe(self):
            return port.ProviderInfo(
                state=port.ProviderState.AVAILABLE,
                provider_id="fake", api_version="1.1", product_version="0.0.5",
            )

        def route_key(self, path):
            # Fake canonical key: header-derived (exposure discriminates the two
            # classes).  Stands in for the C26 light_route_key helper.
            return preflight.acquisition_signature(path)

        def open_session(self, root, *, cancel=None):
            return port.SessionResult(
                state=port.CalibrationState.COMPLETED,
                session=_Session(), fingerprint="f" * 64,
            )

    result = preflight.preflight_calibration("/masters", lights, provider=_Provider())
    assert result.available
    # 2 acquisition classes -> exactly 2 decodes, never 4 (one per light).
    assert calls["n"] == 2
    # plan_map keyed by acquisition signature, one entry per class.
    assert len(result.plan_entries) == 2
    sigs = {e.signature for e in result.plan_entries}
    assert len(sigs) == 2
