"""Tests for seestar.core.geometry_reference (GAR-04 M2)."""

import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from seestar.core.geometry_reference import (
    ORIGIN_AUTO_GEOMETRY,
    ORIGIN_AUTO_LEGACY,
    ORIGIN_RESUME,
    ORIGIN_USER,
    ORIGIN_ZEANALYSER,
    canonical_session_sources,
    reference_quality_metric,
    resolve_reference_precedence,
    select_geometry_reference,
)


def _write_tan_fits(path, ra, dec=0.0, scale=0.01, width=64, height=64):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [(width + 1) / 2.0, (height + 1) / 2.0]
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [-scale, scale]
    w.wcs.radesys = "ICRS"
    header = w.to_header()
    data = np.zeros((height, width), dtype=np.float32)
    fits.writeto(str(path), data, header=header, overwrite=True)
    return str(path)


def _write_no_wcs_fits(path, width=64, height=64):
    header = fits.Header()
    header["NAXIS1"] = width
    header["NAXIS2"] = height
    data = np.zeros((height, width), dtype=np.float32)
    fits.writeto(str(path), data, header=header, overwrite=True)
    return str(path)


def _write_rgb_fits(path, width=16, height=16, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(0.5, 0.1, size=(height, width, 3)).astype(np.float32)
    header = fits.Header()
    header["NAXIS1"] = width
    header["NAXIS2"] = height
    header["NAXIS3"] = 3
    fits.writeto(str(path), data, header=header, overwrite=True)
    return str(path)


# ---- quality metric -----------------------------------------------------

def test_reference_quality_metric_formula(tmp_path):
    from seestar.core.image_processing import load_and_validate_fits
    path = _write_rgb_fits(tmp_path / "rgb.fit")
    got = reference_quality_metric(path, correct_hot_pixels=False)
    image, _ = load_and_validate_fits(path)
    median = float(np.median(image))
    mad = float(np.median(np.abs(image - median)))
    expected = median / (1.4826 * mad + 1e-9)
    assert got == pytest.approx(expected, rel=1e-6)


def test_reference_quality_metric_rejects_flat(tmp_path):
    flat = np.full((16, 16, 3), 0.5, dtype=np.float32)
    header = fits.Header()
    header["NAXIS1"] = 16
    header["NAXIS2"] = 16
    header["NAXIS3"] = 3
    path = str(tmp_path / "flat.fit")
    fits.writeto(path, flat, header=header, overwrite=True)
    assert reference_quality_metric(path, correct_hot_pixels=False) is None


# ---- canonical sources --------------------------------------------------

def test_canonical_multi_root_and_dedup(tmp_path):
    current = tmp_path / "current"
    extra = tmp_path / "extra"
    current.mkdir()
    extra.mkdir()
    _write_tan_fits(current / "b.fit", 10.0)
    _write_tan_fits(extra / "a.fit", 10.01)
    os.symlink(current / "b.fit", extra / "same.fit")
    sources = canonical_session_sources(str(current), [str(extra)])
    assert [os.path.basename(p) for p in sources] == ["a.fit", "b.fit"]


def test_canonical_stack_plan_authoritative(tmp_path):
    import csv
    current = tmp_path / "current"
    current.mkdir()
    _write_tan_fits(current / "outside.fit", 10.0)
    second = _write_tan_fits(current / "second.fit", 10.01)
    first = _write_tan_fits(current / "first.fit", 10.02)
    plan = current / "stack_plan.csv"
    with open(plan, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["batch_id", "file_path"])
        writer.writeheader()
        writer.writerow({"batch_id": "0", "file_path": second})
        writer.writerow({"batch_id": "1", "file_path": first})
        writer.writerow({"batch_id": "1", "file_path": second})
    sources = canonical_session_sources(str(current), stack_plan_path=str(plan))
    assert sources == [os.path.realpath(second), os.path.realpath(first)]


def test_canonical_name_filter(tmp_path):
    _write_tan_fits(tmp_path / "Light_001.fit", 10.0)
    _write_tan_fits(tmp_path / "stack_002.fit", 10.0)
    sources = canonical_session_sources(str(tmp_path), apply_name_filter=True)
    assert [os.path.basename(p) for p in sources] == ["Light_001.fit"]


# ---- precedence ---------------------------------------------------------

def test_reference_precedence_order(tmp_path):
    resume = tmp_path / "resume.fit"
    user = tmp_path / "user.fit"
    za = tmp_path / "za.fit"
    for p in (resume, user, za):
        p.write_bytes(b"x")
    assert resolve_reference_precedence(
        resume_path=str(resume), user_path=str(user), zeanalyser_path=str(za),
        geometry_path="/g.fit", legacy_path="/l.fit",
    ).origin == ORIGIN_RESUME
    assert resolve_reference_precedence(
        user_path=str(user), zeanalyser_path=str(za),
        geometry_path="/g.fit", legacy_path="/l.fit",
    ).origin == ORIGIN_USER
    assert resolve_reference_precedence(
        zeanalyser_path=str(za), geometry_path="/g.fit", legacy_path="/l.fit",
    ).origin == ORIGIN_ZEANALYSER


def test_invalid_external_falls_to_geometry_then_legacy():
    assert resolve_reference_precedence(
        user_path="/missing", zeanalyser_path="/missing",
        geometry_path="/g.fit", legacy_path="/l.fit",
    ).origin == ORIGIN_AUTO_GEOMETRY
    assert resolve_reference_precedence(legacy_path="/l.fit").origin == ORIGIN_AUTO_LEGACY


# ---- selector -----------------------------------------------------------

def test_selector_selects_best_central_candidate(tmp_path):
    paths = [_write_tan_fits(tmp_path / ("Light_%03d.fit" % i), ra=10.0 + i * 0.01) for i in range(20)]

    def quality_fn(p):
        return 100.0 if os.path.basename(p) == "Light_007.fit" else 1.0

    result = select_geometry_reference(paths, quality_fn=quality_fn)
    assert result.resolved.origin == ORIGIN_AUTO_GEOMETRY
    assert os.path.basename(result.resolved.path) == "Light_007.fit"
    assert result.candidate_count == 20


def test_selector_falls_back_legacy_on_multi_target(tmp_path):
    paths = []
    for i in range(10):
        paths.append(_write_tan_fits(tmp_path / ("A_%d.fit" % i), ra=10.0))
    for i in range(10):
        paths.append(_write_tan_fits(tmp_path / ("B_%d.fit" % i), ra=185.0))
    result = select_geometry_reference(paths, quality_fn=lambda p: 1.0)
    assert result.resolved.origin == ORIGIN_AUTO_LEGACY
    assert result.resolved.path is None


def test_selector_falls_back_legacy_on_no_geometry(tmp_path):
    paths = [_write_no_wcs_fits(tmp_path / ("N_%d.fit" % i)) for i in range(5)]
    result = select_geometry_reference(paths, quality_fn=lambda p: 1.0)
    assert result.resolved.origin == ORIGIN_AUTO_LEGACY


def test_selector_respects_stop(tmp_path):
    paths = [_write_tan_fits(tmp_path / ("Light_%03d.fit" % i), ra=10.0 + i * 0.01) for i in range(20)]
    state = {"n": 0}

    def stop():
        state["n"] += 1
        return state["n"] > 3

    result = select_geometry_reference(paths, quality_fn=lambda p: 1.0, stop_requested=stop)
    assert result.resolved.origin == ORIGIN_AUTO_LEGACY
    assert state["n"] <= 4

