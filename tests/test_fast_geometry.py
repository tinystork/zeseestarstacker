"""Tests for seestar.core.fast_geometry (GAR-04 M1)."""

import math

import pytest

from seestar.core.fast_geometry import (
    evaluate_footprint_gate,
    fast_geometry,
)


def _header(representation="PC+CDELT", width=1080, height=1920, ra=274.68,
            dec=-13.74, scale=0.00066, rotation_deg=0.0, crpix=None):
    header = {
        "NAXIS": 2, "NAXIS1": width, "NAXIS2": height, "WCSAXES": 2,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg", "CUNIT2": "deg",
        "CRPIX1": (width + 1) / 2.0, "CRPIX2": (height + 1) / 2.0,
        "CRVAL1": ra, "CRVAL2": dec,
    }
    if crpix is not None:
        header["CRPIX1"], header["CRPIX2"] = crpix
    theta = math.radians(rotation_deg)
    c, s = math.cos(theta), math.sin(theta)
    if representation == "CD":
        header.update({
            "CD1_1": -scale * c, "CD1_2": scale * s,
            "CD2_1": scale * s, "CD2_2": scale * c,
        })
    elif representation == "PC+CDELT":
        header.update({
            "CDELT1": -scale, "CDELT2": scale,
            "PC1_1": c, "PC1_2": -s, "PC2_1": s, "PC2_2": c,
        })
    else:
        raise ValueError(representation)
    return header


def test_cd_center_parity():
    geometry = fast_geometry(_header("CD", ra=274.68, dec=-13.74))
    assert geometry is not None
    assert geometry.representation == "CD"
    assert geometry.center_ra_deg == pytest.approx(274.68, abs=1e-9)
    assert geometry.center_dec_deg == pytest.approx(-13.74, abs=1e-9)


def test_pc_cdelt_center_parity():
    geometry = fast_geometry(_header("PC+CDELT", ra=274.68, dec=-13.74))
    assert geometry is not None
    assert geometry.representation == "PC+CDELT"
    assert geometry.center_ra_deg == pytest.approx(274.68, abs=1e-9)
    assert geometry.center_dec_deg == pytest.approx(-13.74, abs=1e-9)


def test_cd_and_pc_cdelt_equivalent_for_rotated_offset():
    crpix = (220.4793, 1171.0997)
    cd = fast_geometry(_header("CD", rotation_deg=1.7, crpix=crpix))
    pc = fast_geometry(_header("PC+CDELT", rotation_deg=1.7, crpix=crpix))
    assert cd is not None and pc is not None
    assert cd.center_ra_deg == pytest.approx(pc.center_ra_deg, abs=1e-10)
    assert cd.center_dec_deg == pytest.approx(pc.center_dec_deg, abs=1e-10)
    assert cd.footprint_diag_deg == pytest.approx(pc.footprint_diag_deg, abs=1e-10)


def test_center_moves_away_from_crval_when_crpix_off_center():
    geometry = fast_geometry(_header(crpix=(1.0, 1.0), ra=10.0, dec=10.0, scale=1.0))
    assert geometry is not None
    assert abs(geometry.center_ra_deg - 10.0) > 0.1
    assert abs(geometry.center_dec_deg - 10.0) > 0.1


def test_footprint_diagonal_magnitude():
    geometry = fast_geometry(_header(scale=0.00066))
    assert geometry is not None
    assert 1.40 < geometry.footprint_diag_deg < 1.50


@pytest.mark.parametrize("mutate", [
    lambda h: h.update({"CTYPE1": "RA---SIN"}),
    lambda h: h.update({"A_ORDER": 2}),
    lambda h: h.update({"PV1_1": 0.0}),
    lambda h: h.update({"WCSAXES": 3}),
    lambda h: h.update({"NAXIS3": 3}),
    lambda h: h.update({"CUNIT1": "rad"}),
    lambda h: h.update({"CTYPE1": "DEC--TAN", "CTYPE2": "RA---TAN"}),
    lambda h: h.update({"LONPOLE": 0.0}),
    lambda h: h.update({"CRVAL2": 120.0}),
])
def test_unsupported_wcs_rejected(mutate):
    header = _header()
    mutate(header)
    assert fast_geometry(header) is None


def test_incomplete_cd_rejected():
    header = _header("CD")
    del header["CD2_2"]
    assert fast_geometry(header) is None


def test_degenerate_transform_rejected():
    header = _header("PC+CDELT")
    header["CDELT1"] = 0.0
    assert fast_geometry(header) is None


def _geometries(ras, scale=0.00066):
    out = []
    for ra in ras:
        geometry = fast_geometry(_header(ra=ra, scale=scale))
        assert geometry is not None
        out.append(geometry)
    return out


def test_coherent_panel_accepted():
    result = evaluate_footprint_gate(_geometries([10.0 + i * 0.01 for i in range(20)]))
    assert result.accepted is True
    assert result.coherent_fraction == 1.0
    assert result.representative_diagonal_deg is not None


def test_multi_target_rejected_without_fixed_degree_radius():
    geometries = _geometries([10.0] * 10) + _geometries([185.0] * 10)
    result = evaluate_footprint_gate(geometries)
    assert result.accepted is False
    assert "frame diagonals" in result.reason


def test_gate_scale_invariant():
    small = _geometries([20.0 + i * 0.002 for i in range(10)], scale=0.0001)
    large = _geometries([20.0 + i * 0.02 for i in range(10)], scale=0.001)
    assert evaluate_footprint_gate(small).accepted == evaluate_footprint_gate(large).accepted


def test_insufficient_geometry_fraction_rejected():
    assert evaluate_footprint_gate(_geometries([10.0] * 7), dataset_size=10).accepted is False
    assert evaluate_footprint_gate(_geometries([10.0] * 8), dataset_size=10).accepted is True


def test_no_geometry_rejected():
    result = evaluate_footprint_gate([], dataset_size=20)
    assert result.accepted is False
    assert "insufficient fast geometry" in result.reason

