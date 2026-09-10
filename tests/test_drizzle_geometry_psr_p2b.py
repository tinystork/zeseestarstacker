"""P2-B focused tests: WCS-derived kernel pixel-scale factor (geometry only).

Covers the additive production groundwork kept by this bounded phase:

* WCS ratio derivation (nontrivial WCS, not merely ``1/scale``);
* fail-closed on unresolvable geometry;
* accumulator threading of the factor into the installed engine;
* validation of the factor (finite/positive);
* geometry-only A/B behaviour (Square unaffected; corrected Lanczos does not
  collapse to a point kernel; the disproven ``psr == scale`` control is not the
  corrected value).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS

from seestar.core.drizzle_core import (
    DrizzleAccumulator,
    DrizzleGeometryError,
    PIXEL_SCALE_RATIO_SOURCE,
    angular_pixel_scale_deg,
    derive_pixel_scale_ratio,
)

N = 48


def _wcs(plate_deg, crval=(10.0, 20.0), shape=(N, N)):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = [-plate_deg, plate_deg]
    w.wcs.crval = list(crval)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.array_shape = shape
    return w


def _pair(scale, plate_deg=2.4e-4):
    ref = _wcs(plate_deg)
    out = ref.deepcopy()
    out.wcs.crpix = np.asarray(ref.wcs.crpix) * scale
    out.wcs.cdelt = np.asarray(ref.wcs.cdelt) / scale
    out.array_shape = (int(round(N * scale)), int(round(N * scale)))
    return ref, out


def test_ratio_is_output_over_input_on_regular_grid():
    ref, out = _pair(3.0)
    ratio = derive_pixel_scale_ratio(ref, out)
    assert ratio == pytest.approx(1.0 / 3.0, rel=1e-9)
    # it is the angular ratio, not a hard-coded reciprocal of the scale scalar
    assert angular_pixel_scale_deg(out) == pytest.approx(
        angular_pixel_scale_deg(ref) / 3.0, rel=1e-9
    )


def test_ratio_is_nontrivial_for_anisotropic_wcs():
    ref = _wcs(2.4e-4)
    ref.wcs.cdelt = [-2.4e-4, 1.2e-4]  # anisotropic input pixels
    out = _wcs(1.2e-4)
    ratio = derive_pixel_scale_ratio(ref, out)
    assert math.isfinite(ratio) and ratio > 0.0
    assert ratio == pytest.approx(
        angular_pixel_scale_deg(out) / angular_pixel_scale_deg(ref), rel=1e-9
    )
    # NOT the reciprocal of any single scale scalar
    assert ratio != pytest.approx(1.0 / 3.0, rel=1e-9)


def test_fail_closed_on_unresolvable_geometry():
    ref, out = _pair(2.0)
    with pytest.raises(DrizzleGeometryError):
        derive_pixel_scale_ratio(None, out)
    with pytest.raises(DrizzleGeometryError):
        derive_pixel_scale_ratio(ref, None)
    degenerate = WCS(naxis=2)
    degenerate.wcs.cdelt = [0.0, 0.0]  # singular pixel scale
    with pytest.raises(DrizzleGeometryError):
        derive_pixel_scale_ratio(ref, degenerate)
    with pytest.raises(DrizzleGeometryError):
        derive_pixel_scale_ratio(degenerate, ref)


def test_accumulator_rejects_invalid_factor():
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            DrizzleAccumulator((4, 4), kernel="lanczos2", pixel_scale_ratio=bad)


def test_factor_is_threaded_to_the_engine(monkeypatch):
    captured = {}

    class _FakeEngine:
        _texptime = 0.0

        def add_image(self, **kwargs):
            captured.update(kwargs)

    acc = DrizzleAccumulator((4, 4), kernel="lanczos2", pixel_scale_ratio=0.5)
    acc._drizzle = _FakeEngine()
    acc.add(np.ones((2, 2), np.float32), np.ones((2, 2), np.float32),
            np.zeros((2, 2, 2), np.float64))
    assert captured["pixel_scale_ratio"] == 0.5
    assert "iscale" not in captured  # iscale contract unchanged


def test_absent_factor_is_not_passed_and_is_legacy_default(monkeypatch):
    captured = {}

    class _FakeEngine:
        _texptime = 0.0

        def add_image(self, **kwargs):
            captured.update(kwargs)

    acc = DrizzleAccumulator((4, 4), kernel="square")
    assert acc.pixel_scale_ratio is None
    acc._drizzle = _FakeEngine()
    acc.add(np.ones((2, 2), np.float32), np.ones((2, 2), np.float32),
            np.zeros((2, 2, 2), np.float64))
    assert "pixel_scale_ratio" not in captured


def test_from_native_state_carries_the_factor():
    img = np.zeros((4, 4), np.float32)
    wht = np.ones((4, 4), np.float32)
    acc = DrizzleAccumulator.from_native_state(
        (4, 4), img, wht, kernel="lanczos2", total_exptime=1.0,
        pixel_scale_ratio=0.25,
    )
    assert acc.pixel_scale_ratio == 0.25


def _ab_artifact():
    path = (
        Path(__file__).resolve().parents[1]
        / "research/drizzle_scientific_closure_p2/p2b/artifacts/p2b_geometry_ab.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_geometry_only_ab_qualification():
    art = _ab_artifact()
    rows = {(r["kernel"], r["scale"]): r for r in art["rows"]}
    # Square is unaffected by the upstream kernel-scale factor
    for scale in (1.0, 2.0, 3.0, 4.0):
        r = rows[("square", scale)]
        assert r["baseline_psr_absent"] == r["corrected_wcs_ratio"]
    # discriminating metrics for the physical-relevant kernels and scales
    for kernel in ("lanczos2", "lanczos3"):
        for scale in (2.0, 3.0, 4.0):
            r = rows[(kernel, scale)]
            base, corr, ctrl = (
                r["baseline_psr_absent"],
                r["corrected_wcs_ratio"],
                r["control_psr_equals_scale"],
            )
            assert r["wcs_derived_ratio"] == pytest.approx(1.0 / scale, rel=1e-9)
            # corrected geometry is NOT the disproven psr=scale collapse
            assert r["wcs_derived_ratio"] != pytest.approx(scale, rel=1e-6)
            assert ctrl["nonzero"] < corr["nonzero"]
            assert ctrl["sum_abs_wht"] < 0.5 * corr["sum_abs_wht"]
            # the corrected kernel is not a point kernel
            assert corr["nonzero"] > 100
            assert corr["rms_radius_out_px"] > 1.0
            # signed-weight discrimination (whole-impulse mass, not per-pixel
            # noise): baseline retains signed cancellation, the corrected WCS
            # ratio produces strictly positive weights (signed_fraction == 0)
            assert base["signed_fraction"] >= corr["signed_fraction"]
            assert corr["signed_fraction"] == pytest.approx(0.0, abs=1e-12)
    # the physical configuration (x3) starts from real signed cancellation
    for kernel in ("lanczos2", "lanczos3"):
        r3 = rows[(kernel, 3.0)]
        assert r3["baseline_psr_absent"]["signed_fraction"] > 0.0
        assert r3["corrected_wcs_ratio"]["signed_fraction"] == pytest.approx(0.0, abs=1e-12)
