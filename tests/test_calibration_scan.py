"""C12 — recursive masters scan + coarse-night acquisition signature."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

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


_PRE = set(sys.modules)
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
scan = _load_by_path("seestar.calibration.scan", "seestar/calibration/scan.py")
preflight = _load_by_path(
    "seestar.calibration.preflight", "seestar/calibration/preflight.py"
)
for _k in list(sys.modules):
    if _k not in _PRE:
        del sys.modules[_k]


# ---------------------------------------------------------------------------
# recursive scan
# ---------------------------------------------------------------------------
def _write_fits(path):
    fits.PrimaryHDU(np.zeros((8, 8), dtype=np.int16)).writeto(str(path), overwrite=True)


def test_scan_recursive_bounded_excludes_output_dirs(tmp_path):
    root = tmp_path / "masters"
    (root / "dark" / "nested" / "deep").mkdir(parents=True)
    (root / "stacked").mkdir(parents=True)
    (root / "calibrated").mkdir(parents=True)
    (root / "Flat").mkdir(parents=True)
    _write_fits(root / "dark" / "nested" / "deep" / "master.fits")
    _write_fits(root / "dark" / "nested" / "deep" / "master2.fits")
    _write_fits(root / "stacked" / "output.fits")     # excluded
    _write_fits(root / "calibrated" / "out.fits")     # excluded
    _write_fits(root / "Flat" / "flat.fits")
    (root / "dark" / "nested" / "deep" / "thumb.jpg").write_text("x")  # non-FITS
    out = scan.scan_masters_recursive(str(root), max_depth=4)
    names = {Path(p).name for p in out}
    assert names == {"master.fits", "master2.fits", "flat.fits"}


def test_scan_respects_max_depth(tmp_path):
    root = tmp_path / "m"
    (root / "a" / "b" / "c" / "d" / "e").mkdir(parents=True)
    _write_fits(root / "a" / "b" / "c" / "d" / "e" / "deep.fits")  # depth 5
    _write_fits(root / "a" / "shallow.fits")  # depth 2
    out = scan.scan_masters_recursive(str(root), max_depth=4)
    names = {Path(p).name for p in out}
    assert names == {"shallow.fits"}  # deep.fits at depth 5 is skipped


def test_flatten_disambiguates_collisions(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    (a / "x").mkdir(parents=True)
    (b / "x").mkdir(parents=True)
    _write_fits(a / "x" / "m.fits")
    _write_fits(b / "x" / "m.fits")
    flat = tmp_path / "flat"
    scan.flatten_masters([str(a / "x" / "m.fits"), str(b / "x" / "m.fits")], flat)
    names = sorted(p.name for p in flat.iterdir())
    assert len(names) == 2  # both copied, disambiguated


# ---------------------------------------------------------------------------
# coarse-night acquisition signature
# ---------------------------------------------------------------------------
def _write_light(path, *, date_obs, exptime=60.0, gain=120.0, filter_="irct",
                 bayer="RGGB", instrume="ZWO ASI294MC Pro"):
    hdu = fits.PrimaryHDU(np.zeros((16, 16), dtype=np.int16))
    hdu.header["IMAGETYP"] = "Light"
    hdu.header["EXPTIME"] = exptime
    hdu.header["GAIN"] = gain
    hdu.header["XBINNING"] = 1
    hdu.header["YBINNING"] = 1
    hdu.header["BAYERPAT"] = bayer
    hdu.header["INSTRUME"] = instrume
    hdu.header["FILTER"] = filter_
    hdu.header["DATE-OBS"] = date_obs
    hdu.header["NAXIS"] = 2
    hdu.writeto(str(path), overwrite=True)


def test_night_signature_same_night_across_midnight(tmp_path):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    _write_light(a, date_obs="2026-09-14T23:41:23")
    _write_light(b, date_obs="2026-09-15T00:30:00")  # after midnight -> same night
    assert preflight.acquisition_signature(str(a)) == preflight.acquisition_signature(str(b))


def test_night_signature_discriminates_different_night(tmp_path):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    _write_light(a, date_obs="2026-09-14T23:41:23")
    _write_light(b, date_obs="2026-09-16T01:00:00")  # a different night
    assert preflight.acquisition_signature(str(a)) != preflight.acquisition_signature(str(b))


def test_night_signature_includes_filter(tmp_path):
    a = tmp_path / "a.fits"
    b = tmp_path / "b.fits"
    _write_light(a, date_obs="2026-09-14T23:41:23", filter_="irct")
    _write_light(b, date_obs="2026-09-14T23:41:23", filter_="L")
    assert preflight.acquisition_signature(str(a)) != preflight.acquisition_signature(str(b))
