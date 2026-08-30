import os

import pytest

from seestar.core.geometry_reference import (
    GEOMETRY_SOURCE_NATIVE,
    GEOMETRY_SOURCE_UNAVAILABLE,
    GEOMETRY_SOURCE_WCS,
    ORIGIN_AUTO_GEOMETRY,
    ORIGIN_AUTO_LEGACY,
    select_geometry_reference,
)
from seestar.core.native_pointing import (
    NativePointing,
    evaluate_native_pointing_gate,
    parse_native_pointing,
    parse_native_pointing_result,
    robust_spherical_center,
)


def _native(ra, dec=-13.7):
    return {"RA": ra, "DEC": dec}


def _wcs(ra, dec=-13.7):
    return {
        "NAXIS1": 1920,
        "NAXIS2": 1080,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "WCSAXES": 2,
        "CRPIX1": 960.5,
        "CRPIX2": 540.5,
        "CRVAL1": ra,
        "CRVAL2": dec,
        "CDELT1": -0.0006,
        "CDELT2": 0.0006,
    }


def _run(monkeypatch, headers, quality=None, progress=None):
    paths = [f"/dataset/frame_{index:04d}.fit" for index in range(len(headers))]
    by_path = dict(zip(paths, headers))
    monkeypatch.setattr(
        "seestar.core.geometry_reference.fits.getheader", lambda path: by_path[path]
    )
    if quality is None:
        quality = lambda path: float(int(os.path.basename(path)[6:10]))
    return paths, select_geometry_reference(paths, quality_fn=quality, progress=progress)


def test_parser_accepts_decimal_degrees_and_normalizes_360():
    pointing = parse_native_pointing({"RA": 360.0, "DEC": "-13.75deg"})
    assert pointing == NativePointing(0.0, -13.75, ra_key="RA", dec_key="DEC")


@pytest.mark.parametrize(
    "header, reason_fragment",
    [
        ({"RA": "18:20:16", "DEC": "-13:45:00"}, "ambiguous"),
        ({"RA": 275.0}, "incomplete"),
        ({"RA": 275.0, "DEC": -91.0}, "outside"),
        ({"RA": float("nan"), "DEC": 0.0}, "non-finite"),
    ],
)
def test_parser_rejects_invalid_or_ambiguous_values(header, reason_fragment):
    result = parse_native_pointing_result(header)
    assert result.pointing is None
    assert reason_fragment in result.reason


def test_parser_never_cross_pairs_keyword_families():
    result = parse_native_pointing_result({"RA": 10.0, "OBJCTDEC": 20.0})
    assert result.pointing is None
    assert "incomplete RA/DEC" in result.reason


def test_robust_spherical_center_handles_ra_wrap():
    center = robust_spherical_center(
        [NativePointing(359.9, 10.0), NativePointing(0.0, 10.0), NativePointing(0.1, 10.0)]
    )
    assert min(abs(center[0]), abs(center[0] - 360.0)) < 1e-6
    assert center[1] == pytest.approx(10.0, abs=1e-5)


def test_native_gate_accepts_realistic_m16_drift():
    points = [
        NativePointing(275.069320 - 0.077085 + index * 0.154170 / 99.0,
                       -13.73 - 0.049444 + index * 0.098888 / 99.0)
        for index in range(100)
    ]
    result = evaluate_native_pointing_gate(points, dataset_size=100)
    assert result.accepted
    assert result.pointing_fraction == 1.0
    assert result.coherent_fraction == 1.0


def test_native_gate_rejects_disconnected_m16_m31_population():
    points = [NativePointing(275.07 + index * 1e-4, -13.73) for index in range(20)]
    points += [NativePointing(10.68 + index * 1e-4, 41.27) for index in range(20)]
    result = evaluate_native_pointing_gate(points, dataset_size=40)
    assert not result.accepted
    assert "incoherent native pointing" in result.reason


def test_a_native_without_wcs_selects_auto_geometry(monkeypatch):
    paths, result = _run(monkeypatch, [_native(275.0 + index * 0.001) for index in range(40)])
    assert result.geometry_source == GEOMETRY_SOURCE_NATIVE
    assert result.resolved.origin == ORIGIN_AUTO_GEOMETRY
    assert result.pointing_count == 40
    assert result.geometry_count == 0
    selected_index = int(os.path.basename(result.resolved.path)[6:10])
    assert 4 <= selected_index <= 35


def test_b_partial_wcs_enrichment_does_not_change_native_population(monkeypatch):
    native_headers = [_native(275.0 + index * 0.001) for index in range(40)]
    paths, plain = _run(monkeypatch, native_headers)
    enriched = []
    for index, header in enumerate(native_headers):
        merged = dict(header)
        if index < 17:
            merged.update(_wcs(header["RA"], header["DEC"]))
        enriched.append(merged)
    _paths, mixed = _run(monkeypatch, enriched)
    assert mixed.geometry_source == plain.geometry_source == GEOMETRY_SOURCE_NATIVE
    assert mixed.pointing_count == plain.pointing_count == 40
    assert mixed.geometry_count == 17
    assert mixed.resolved.path == plain.resolved.path


def test_c_central_shortlist_excludes_high_quality_first_20_edge_frames(monkeypatch):
    headers = [_native(275.8 + index * 1e-4) for index in range(20)]
    headers += [_native(275.0 + (index - 20) * 1e-4) for index in range(20, 72)]

    def quality(path):
        index = int(os.path.basename(path)[6:10])
        return 1000.0 if index < 20 else float(index)

    paths, result = _run(monkeypatch, headers, quality=quality)
    selected_index = int(os.path.basename(result.resolved.path)[6:10])
    assert result.geometry_source == GEOMETRY_SOURCE_NATIVE
    assert selected_index >= 20
    assert result.resolved.path != os.path.realpath(paths[19])


def test_d_quality_can_beat_mathematically_closest_candidate(monkeypatch):
    headers = [_native(10.0), _native(10.001), _native(9.999), _native(10.002)]
    paths, result = _run(
        monkeypatch,
        headers,
        quality=lambda path: 50.0 if path.endswith("0001.fit") else 1.0,
    )
    assert result.resolved.path == os.path.realpath(paths[1])
    assert result.selected_offset_deg > 0.0


def test_e_generic_solved_dataset_uses_strict_wcs_fallback(monkeypatch):
    _paths, result = _run(monkeypatch, [_wcs(120.0 + index * 0.001) for index in range(40)])
    assert result.geometry_source == GEOMETRY_SOURCE_WCS
    assert result.resolved.origin == ORIGIN_AUTO_GEOMETRY
    assert result.pointing_count == 0
    assert result.geometry_count == 40


def test_f_no_geometry_is_explicit_legacy_fallback(monkeypatch):
    _paths, result = _run(monkeypatch, [{} for _index in range(40)])
    assert result.geometry_source == GEOMETRY_SOURCE_UNAVAILABLE
    assert result.resolved.origin == ORIGIN_AUTO_LEGACY
    assert "native:" in result.fallback_reason
    assert "wcs:" in result.fallback_reason


def test_h_scan_progress_is_bounded_for_1600_inputs(monkeypatch):
    events = []

    def progress(message, percent=None, level=None):
        events.append((message, percent, level))

    _paths, result = _run(
        monkeypatch,
        [_native(275.0 + (index % 100) * 1e-5) for index in range(1600)],
        progress=progress,
    )
    scan_events = [event for event in events if event[0].startswith("Scanning dataset pointing")]
    assert len(scan_events) == 11
    assert len(events) <= 20
    assert scan_events[0][1] == 0
    assert scan_events[-1][1] == 100
    assert all(event[2] == "INFO" for event in events)
    assert result.progress_event_count == len(events)
