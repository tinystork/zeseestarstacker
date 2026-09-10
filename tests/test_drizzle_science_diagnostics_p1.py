"""Phase-1 / rework-1 / rework-2 tests for passive Drizzle science diagnostics.

ZSSS-DRIZZLE-CLOSURE-P1.  Proves the instrumentation contract:

* read-only helpers (SCI/WHT/SUP bitwise unchanged, incl. signed + non-finite);
* a diagnostic calculation or I/O failure can neither change science nor abort
  the run;
* the persisted artifact contains terminal lifecycle events, with explicit
  terminal outcome on early-return/failure routes (F1, R3);
* artifact coordinates/support align to the final post-crop /
  post-effective-WHT-policy SCI grid (F2);
* memory-bounded redesign: no full EDT map, no `.wht` support copies, no full
  positive fancy-index selection, bounded sample/top-k caps (R1);
* distance-map edge semantics are symmetric and edge-aware (F4);
* physical support (`SUP_W1>0`) is never conflated with positive native WHT;
  fallback labelling is explicit (F5, R2);
* N_eff statistics are computed over valid support only (F6);
* lifecycle retention coalesces repetitive checkpoint events while always
  keeping terminal evidence (F7);
* cleanup provenance is truthful (F8) and first-deposit timing is split (F9);
* atomic writer removes its temp file on `os.replace` failure (R4).

All scenes are synthetic, deterministic and fast (no GPU / GUI / network).
"""

import inspect
import json
import types

import numpy as np
import pytest

from astropy.wcs import WCS

from seestar.core import drizzle_science_diagnostics as dsd
from seestar.core.drizzle_core import DrizzleAccumulator, build_output_grid
from seestar.queuep import queue_manager as qm
from seestar.queuep.queue_manager import SeestarQueuedStacker


def make_wcs(shape_hw, cdelt=(-0.001, 0.001)):
    h, w = shape_hw
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [w / 2.0 + 0.5, h / 2.0 + 0.5]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cdelt = np.array(list(cdelt))
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.pc = np.array([[1.0, 0.0], [0.0, 1.0]])
    wcs.array_shape = (h, w)
    return wcs


def _bitwise_equal(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


class Stub:
    def __init__(self, diag=None):
        if diag is not None:
            self._drizzle_science_diag = diag


# ---------------------------------------------------------------------------
# 1. read-only / passive
# ---------------------------------------------------------------------------


def test_helpers_are_read_only_bitwise_including_signed_and_nonfinite():
    rng = np.random.default_rng(1234)
    sci = rng.normal(size=(24, 24, 3)).astype(np.float32)
    wht = rng.normal(size=(24, 24, 3)).astype(np.float32)
    sci[0, 0, 0] = np.nan
    wht[1, 1, 1] = np.inf
    wht[2, 2, 2] = -np.inf
    w1 = np.abs(rng.random((24, 24))).astype(np.float32)
    w2 = (w1 * w1 + 1e-3).astype(np.float32)
    mask = np.isfinite(w1) & (w1 > 0)

    snaps = [np.array(x, copy=True) for x in (sci, wht, w1, w2)]
    sections = dsd.summarize_run(sci, wht, sup_w1=w1, sup_w2=w2, support_mask=mask)
    for c in range(3):
        dsd.sci_channel_stats(sci[..., c])
        dsd.wht_channel_diagnostics(wht[..., c], sci[..., c])
        dsd.threshold_sweep(wht[..., c], sci[..., c], mask)
    dsd.support_conditioning(w1, w2)
    dsd.conditioning_candidates(sci, wht, mask, w1, w2)
    dsd.spatial_boundary_diagnostics(sci, wht, mask, sup_w1=w1, sup_w2=w2)
    for snap, arr in zip(snaps, (sci, wht, w1, w2)):
        assert _bitwise_equal(snap, arr), "diagnostic helper mutated its input"
    assert sections["sci_stats"][0]["nonfinite_fraction"] > 0


def test_collector_to_dict_versioned_and_deterministic():
    diag = dsd.DrizzleScienceDiagnostics(run_token="tok")
    diag.set_run_config(kernel="lanczos2", scale=2.0, pixfrac_requested=0.8,
                        pixfrac_effective=1.0)
    diag.set_geometry(dsd.geometry_diagnostic(make_wcs((8, 8)), make_wcs((8, 8))))
    diag.add_lifecycle({"stage": "x", "ts": 1.0, "support_available": True})
    obj = json.loads(json.dumps(diag.to_dict(), sort_keys=True, allow_nan=False))
    assert obj["schema_version"] == dsd.SCHEMA_VERSION
    assert obj["kernel"] == "lanczos2"
    assert obj["pixel_scale_ratio_current"] == 1.0
    assert obj["pixel_scale_ratio_source"] == "upstream_add_image_default"
    assert obj["diagnostic_only"] is True


def test_collector_write_atomic_and_fail_open(tmp_path):
    diag = dsd.DrizzleScienceDiagnostics(run_token="r", output_folder=str(tmp_path))
    diag.set_run_config(kernel="square", scale=1.0, pixfrac_requested=1.0,
                        pixfrac_effective=1.0)
    assert diag.write() is True
    json.loads((tmp_path / dsd.ARTIFACT_FILENAME).read_text())
    assert not list(tmp_path.glob("*.tmp.*"))
    assert dsd.DrizzleScienceDiagnostics(run_token="r").write("/nonexistent-\x00/x.json") is False


def test_writer_temp_cleanup_on_replace_failure(tmp_path, monkeypatch):
    """R4: a failing os.replace must not leak a *.tmp.* sibling."""
    diag = dsd.DrizzleScienceDiagnostics(run_token="r", output_folder=str(tmp_path))
    diag.set_run_config(kernel="square", scale=1.0, pixfrac_requested=1.0,
                        pixfrac_effective=1.0)

    def boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(dsd.os, "replace", boom)
    assert diag.write() is False
    assert not list(tmp_path.glob("*.tmp.*")), "temporary file leaked"
    # the failure is fail-open and does not corrupt a later successful write
    monkeypatch.undo()
    assert diag.write() is True


def test_setters_and_write_fail_open_on_garbage():
    diag = dsd.DrizzleScienceDiagnostics(run_token="r")
    diag.set_geometry(object())
    diag.set_sci_stats(object())
    diag.set_boundary_bins(None)
    diag.note("ok")
    assert json.dumps(diag.to_dict(), sort_keys=True, allow_nan=False)
    assert diag.write() is False


# ---------------------------------------------------------------------------
# 2. geometry + real add_image contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale,expected",
                         [(1, 1.0), (2, 0.5), (3, 1.0 / 3.0), (4, 0.25)])
def test_geometry_candidate_is_wcs_ratio(scale, expected):
    ref = make_wcs((32, 32))
    out, _shape = build_output_grid(ref, (32, 32), scale)
    rec = dsd.geometry_diagnostic(ref, out, kernel="square", scale=scale)
    assert rec["available"] is True
    assert rec["pixel_scale_ratio_current"] == 1.0
    assert rec["candidate_source"] == "wcs_ratio"
    assert rec["pixel_scale_ratio_candidate"] == pytest.approx(expected, rel=1e-6)


def test_geometry_fail_open_on_missing_wcs():
    rec = dsd.geometry_diagnostic(None, None)
    assert rec["available"] is False
    assert rec["pixel_scale_ratio_candidate"] is None and rec["reason"]


def test_init_diagnostics_emits_geometry_token():
    lines = []
    ref = make_wcs((16, 16))
    out, _shape = build_output_grid(ref, (16, 16), 2)
    obj = types.SimpleNamespace(
        reference_wcs_object=ref, drizzle_output_wcs=out, output_folder=None,
        update_progress=lambda msg, *a, **k: lines.append(str(msg)),
    )
    SeestarQueuedStacker._init_drizzle_science_diagnostics(obj, "square", 1.0, 1.0, 2.0, None)
    token = [x for x in lines if x.startswith("DRIZZLE_GEOMETRY_DIAGNOSTIC")][0]
    assert "kernel=square" in token and "scale=2.0" in token
    assert "pixel_scale_ratio_current=1.0" in token
    assert "candidate_source=wcs_ratio" in token
    assert "fresh_m3_init" in obj._drizzle_science_diag.lifecycle_stages()


def test_init_diagnostics_geometry_token_fail_open_without_wcs():
    lines = []
    obj = types.SimpleNamespace(
        reference_wcs_object=None, drizzle_output_wcs=None, output_folder=None,
        update_progress=lambda msg, *a, **k: lines.append(str(msg)),
    )
    SeestarQueuedStacker._init_drizzle_science_diagnostics(obj, "square", 1.0, 1.0, 1.0, None)
    token = [x for x in lines if x.startswith("DRIZZLE_GEOMETRY_DIAGNOSTIC")][0]
    assert "pixel_scale_ratio_candidate=unavailable" in token and "reason=" in token


def test_real_add_image_omits_iscale_and_pixel_scale_ratio(monkeypatch):
    captured = {}
    import drizzle.resample as dr
    real = dr.Drizzle.add_image

    def spy(self, *args, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(dr.Drizzle, "add_image", spy)
    acc = DrizzleAccumulator((16, 16), kernel="square", pixfrac=0.7)
    data = np.ones((8, 8), dtype=np.float32)
    wt = np.ones((8, 8), dtype=np.float32)
    yy, xx = np.indices((8, 8), dtype=np.float64)
    pixmap = np.dstack((xx + 4.0, yy + 4.0))
    acc.add(data, wt, pixmap, exptime=2.0, in_units="counts")
    keys = set(captured["kwargs"])
    assert keys == {"data", "exptime", "pixmap", "weight_map", "in_units",
                    "pixfrac", "wht_scale"}
    c = dsd.contract_diagnostic(acc.kernel, acc.pixfrac, exptime=2.0,
                                in_units="counts", fillval=acc.fillval)
    assert c["pixel_scale_ratio_source"] == "upstream_add_image_default"
    assert c["pixel_scale_ratio_effective"] == 1.0
    assert c["iscale_source"] == "upstream_add_image_default"
    assert c["iscale_effective"] == 1.0
    assert c["kernel_source"] == "drizzle_construction_explicit"
    assert c["fillval_source"] == "drizzle_construction_default"
    assert c["wht_scale_effective"] == 2.0


# ---------------------------------------------------------------------------
# 3. threshold / support / N_eff / R2 labelling
# ---------------------------------------------------------------------------


def test_threshold_sweep_separates_support_from_positive_wht():
    wht = np.array([[1.0, 0.05, 0.0], [2e-3, 5e-5, -1.0]], dtype=np.float32)
    sci = np.arange(6, dtype=np.float32).reshape(2, 3) + 1.0
    support = np.array([[True, True, True], [True, True, False]])
    sweep = dsd.threshold_sweep(wht, sci, support, abs_candidates=(1e-4, 1e-3))
    assert sweep["support_source"] == dsd.SUPPORT_SOURCE_PHYSICAL
    assert sweep["physical_support_pixels"] == int(support.sum())
    assert sweep["currently_valid_positive_native_wht_pixels"] == 4
    by = {c["name"]: c for c in sweep["candidates"]}
    assert by["abs_0.001"]["newly_removed_from_current_valid"] == 1
    assert by["abs_0.0001"]["newly_removed_from_current_valid"] == 1


def test_threshold_sweep_fallback_label_does_not_claim_physical_support():
    """R2: a native-WHT-derived fallback must not be labelled physical support."""
    wht = np.ones((4, 4), dtype=np.float32)
    sci = np.ones((4, 4), dtype=np.float32)
    support = np.ones((4, 4), dtype=bool)
    sweep = dsd.threshold_sweep(wht, sci, support,
                                support_source="native_positive_wht_fallback")
    assert sweep["physical_support_pixels"] is None
    assert sweep["fallback_support_pixels"] == 16
    assert sweep["support_population_pixels"] == 16
    assert sweep["support_denominator_label"] == "fallback_support_pixels"


def test_summarize_fallback_support_source_label():
    sci = np.ones((8, 8, 3), dtype=np.float32)
    wht = np.ones((8, 8, 3), dtype=np.float32)
    sec = dsd.summarize_run(sci, wht, sup_w1=None, sup_w2=None)
    assert sec["support_source"] == dsd.SUPPORT_SOURCE_FALLBACK
    assert sec["boundary_bins"]["support_source"] == dsd.SUPPORT_SOURCE_FALLBACK
    assert sec["threshold_sweep"][0]["physical_support_pixels"] is None
    assert sec["threshold_sweep"][0]["support_source"] == "native_positive_wht_fallback"


def test_support_conditioning_island_not_diluted_by_off_support_zeros():
    w1 = np.zeros((8, 8), dtype=np.float32)
    w2 = np.zeros((8, 8), dtype=np.float32)
    w1[3:5, 3:5] = 2.0
    w2[3:5, 3:5] = 4.0
    rec = dsd.support_conditioning(w1, w2)
    assert rec["support_pixels"] == 4
    assert rec["n_eff_min"] == rec["n_eff_median"] == rec["n_eff_max"] == pytest.approx(1.0)
    assert rec["n_eff_mean"] == pytest.approx(1.0)
    assert rec["off_support_fraction"] == pytest.approx(60 / 64)


def test_boundary_bins_all_true_symmetric_and_edge_aware():
    support = np.ones((7, 7), dtype=bool)
    d, meta = dsd.support_distance_map(support)
    assert meta["available"] is True and meta["edge_padding"] is True
    assert d[0, 0] == pytest.approx(1.0)
    assert d[3, 3] == pytest.approx(4.0)
    assert np.allclose(d, d[::-1, :]) and np.allclose(d, d[:, ::-1])


def test_boundary_bins_edge_touching_left_right_symmetric():
    left = np.zeros((7, 7), dtype=bool)
    left[:, :3] = True
    right = np.zeros((7, 7), dtype=bool)
    right[:, 4:] = True
    dl, _ = dsd.support_distance_map(left)
    dr, _ = dsd.support_distance_map(right)
    assert dl[3, 0] == pytest.approx(1.0)
    assert dl[3, 1] == pytest.approx(2.0)
    assert dl[3, 2] == pytest.approx(1.0)
    assert np.allclose(dl, dr[:, ::-1])


def test_tiled_boundary_matches_full_map_on_bounded_grid():
    """The streaming tiled boundary must agree with the full EDT reference."""
    rng = np.random.default_rng(5)
    support = rng.random((48, 64)) > 0.4
    ref, _meta = dsd.support_distance_map(support)
    sci = rng.normal(size=(48, 64, 1)).astype(np.float32)
    wht = np.ones((48, 64, 1), dtype=np.float32)
    rec = dsd.spatial_boundary_diagnostics(sci, wht, support)
    assert rec["available"] is True
    assert rec["distance_algorithm"] == "row_tile_edt_halo"
    assert rec["halo"] >= 17
    # per-pixel bin membership derived from the reference map
    edges = [hi for _lo, hi in dsd.DISTANCE_BIN_EDGES[:-1]]
    bi = np.searchsorted(np.asarray(edges), ref[support], side="left")
    expected = np.bincount(bi, minlength=5)
    got = np.array([b["pixels"] for b in rec["per_channel"][0]["bins"]])
    assert np.array_equal(got, expected)


def test_boundary_bins_per_channel_and_shape():
    support = np.ones((16, 16), dtype=bool)
    sci = np.stack([np.full((16, 16), c + 1.0, dtype=np.float32) for c in range(3)], axis=-1)
    wht = np.full((16, 16, 3), 0.5, dtype=np.float32)
    rec = dsd.spatial_boundary_diagnostics(sci, wht, support)
    assert rec["available"] is True
    assert len(rec["per_channel"]) == 3
    labels = [b["label"] for b in rec["per_channel"][0]["bins"]]
    assert labels == list(dsd.DISTANCE_BIN_LABELS)
    assert sum(b["pixels"] for b in rec["per_channel"][0]["bins"]) == int(support.sum())
    rec2 = dsd.spatial_boundary_diagnostics(sci, wht, support)
    assert json.dumps(rec, sort_keys=True, allow_nan=False) == json.dumps(
        rec2, sort_keys=True, allow_nan=False
    )


def test_bounded_extrema_only_over_support_and_coordinates():
    sci = np.arange(100, dtype=np.float32).reshape(10, 10)
    wht = np.ones((10, 10), dtype=np.float32)
    support = np.zeros((10, 10), dtype=bool)
    support[2:5, 2:5] = True
    recs = dsd.bounded_extrema_records(sci, wht, support, n=3)
    for r in recs:
        assert support.ravel()[r["index"]], "extrema must lie on physical support"
        assert r["row"] * 10 + r["col"] == r["index"]


# ---------------------------------------------------------------------------
# 3b. D1/D2/D3 — per-extreme N_eff, per-extreme distance, canonical labels
# ---------------------------------------------------------------------------


def _heterogeneous_support_fixture():
    h = w = 16
    sci = np.zeros((h, w, 3), dtype=np.float32)
    wht = np.ones((h, w, 3), dtype=np.float32)
    w1 = np.zeros((h, w), dtype=np.float32)
    w2 = np.zeros((h, w), dtype=np.float32)
    # (row, col, sup_w1, sup_w2) -> N_eff = (w1/sqrt(w2))**2
    spec = [(2, 2, 2.0, 4.0, 1.0), (2, 13, 6.0, 9.0, 4.0),
            (13, 2, 3.0, 3.0, 3.0)]
    for i, (r, c, a, b, _ne) in enumerate(spec):
        w1[r, c] = a
        w2[r, c] = b
        sci[r, c, :] = float(i + 1)
    return sci, wht, w1, w2, spec


def test_d1_per_extreme_n_eff_exact_values_support_extrema():
    """D1: support extrema must carry exact per-point N_eff (not null)."""
    sci, wht, w1, w2, spec = _heterogeneous_support_fixture()
    sec = dsd.summarize_run(sci, wht, sup_w1=w1, sup_w2=w2,
                            support_mask=(w1 > 0), n_extrema=4)
    expected = {(r, c): ne for r, c, _a, _b, ne in spec}
    seen = {}
    for recs in sec["support_extrema"]:
        for r in recs:
            key = (r["row"], r["col"])
            assert r["n_eff"] is not None and r["n_eff_valid"] is True
            assert r["n_eff"] == pytest.approx(expected[key], rel=1e-6)
            seen[key] = r["n_eff"]
    assert set(seen) == set(expected)
    # conditioning extrema carry the same exact N_eff
    for ch in sec["conditioning_candidates"]["per_channel"]:
        for r in ch["extrema"]:
            key = (r["row"], r["col"])
            assert r["n_eff"] == pytest.approx(expected[key], rel=1e-6)


def test_d1_n_eff_null_when_invalid_support():
    """D1: N_eff is null (with validity flag) when the support pair is invalid."""
    sci = np.zeros((6, 6, 3), dtype=np.float32)
    sci[2, 2, :] = 1.0
    wht = np.ones((6, 6, 3), dtype=np.float32)
    w1 = np.zeros((6, 6), dtype=np.float32)
    w2 = np.zeros((6, 6), dtype=np.float32)
    w1[2, 2] = 1.0
    w2[2, 2] = 0.0  # invalid: SUP_W2 must be > 0
    recs = dsd.bounded_extrema_records(sci[..., 0], wht[..., 0], w1 > 0,
                                       w1, w2, None, None, n=2)
    assert any(r["n_eff"] is None and r["n_eff_valid"] is False for r in recs)


def test_d2_local_support_distance_reference_and_gt16():
    """D2: bounded local distance matches the reference and reports >16 truthfully."""
    support = np.zeros((60, 60), dtype=bool)
    support[5:55, 5:55] = True
    ref, _meta = dsd.support_distance_map(support)
    assert dsd._local_support_distance(support, 5, 5) == pytest.approx(ref[5, 5])
    assert ref[5, 5] == pytest.approx(1.0)
    for r, c in [(12, 12), (20, 20), (5, 30)]:
        if ref[r, c] <= 16:
            assert dsd._local_support_distance(support, r, c) == pytest.approx(ref[r, c])
    # global edge is a boundary
    edge = np.ones((10, 10), dtype=bool)
    assert dsd._local_support_distance(edge, 0, 0) == pytest.approx(1.0)
    assert dsd._local_support_distance(edge, 0, 5) == pytest.approx(1.0)
    # all-true large grid: centre distance > 16 -> truthful None
    big = np.ones((60, 60), dtype=bool)
    ref_big, _m = dsd.support_distance_map(big)
    assert ref_big[30, 30] > 16
    assert dsd._local_support_distance(big, 30, 30) is None


def test_d2_support_extrema_carry_distance():
    """D2: support extrema carry the boundary distance (bounded, no full map)."""
    support = np.zeros((40, 40), dtype=bool)
    support[4:36, 4:36] = True
    sci = np.zeros((40, 40), dtype=np.float32)
    sci[20, 20] = 9.0  # interior maximum
    sci[4, 4] = -3.0   # boundary minimum
    wht = np.ones((40, 40), dtype=np.float32)
    ref, _m = dsd.support_distance_map(support)
    recs = dsd.bounded_extrema_records(sci, wht, support, n=2)
    by = {(r["row"], r["col"]): r for r in recs}
    assert by[(4, 4)]["distance"] == pytest.approx(1.0)
    assert by[(4, 4)]["distance_resolved"] is True
    assert by[(20, 20)]["distance"] == pytest.approx(ref[20, 20])
    assert by[(20, 20)]["distance_resolved"] is True


def test_d2_gt16_extrema_distance_is_null_not_fabricated():
    support = np.ones((60, 60), dtype=bool)
    sci = np.zeros((60, 60), dtype=np.float32)
    sci[30, 30] = 5.0
    wht = np.ones((60, 60), dtype=np.float32)
    recs = dsd.bounded_extrema_records(sci, wht, support, n=2)
    centre = [r for r in recs if (r["row"], r["col"]) == (30, 30)]
    assert centre
    assert centre[0]["distance"] is None
    assert centre[0]["distance_resolved"] is False
    assert centre[0]["distance_bound"] == pytest.approx(17.0)


def test_d3_fallback_label_canonical_everywhere():
    """D3: without a SUP pair every support_source field uses the canonical string."""
    sci = np.ones((8, 8, 3), dtype=np.float32)
    wht = np.ones((8, 8, 3), dtype=np.float32)
    sec = dsd.summarize_run(sci, wht)
    canon = dsd.SUPPORT_SOURCE_FALLBACK
    assert canon == "native_positive_wht_fallback"
    assert sec["support_source"] == canon
    assert sec["boundary_bins"]["support_source"] == canon
    assert sec["conditioning_candidates"]["support_source"] == canon
    for sw in sec["threshold_sweep"]:
        assert sw["support_source"] == canon
        assert sw["physical_support_pixels"] is None
        assert sw["fallback_support_pixels"] is not None


# ---------------------------------------------------------------------------
# 4. memory-boundedness: R1 structural + behavioral caps
# ---------------------------------------------------------------------------


def test_bounded_constants_exist_and_sane():
    assert dsd.ROW_CHUNK >= 1
    assert 10_000 <= dsd.MAX_SAMPLE_COUNT <= 1_000_000
    assert 64 * 1024 * 1024 <= dsd.MAX_BOUNDARY_WORK_BYTES <= 2 * 1024 ** 3
    assert dsd.MAX_LIFECYCLE_EVENTS <= 4096
    assert dsd.BOUNDARY_HALO >= 17
    assert dsd.BOUNDARY_TILE_ROWS >= 1


def test_summarize_does_not_call_full_distance_map():
    """R1 structural guard: the summary path must not build a full EDT map."""
    src = inspect.getsource(dsd.summarize_run)
    assert "support_distance_map" not in src
    assert "distance_transform_edt" not in src


def test_module_has_no_full_argsort_and_no_hwc_float64_conversion():
    """Structural guard (documentation): no full argsort / HWC float64 casts."""
    src = inspect.getsource(dsd)
    assert "np.argsort" not in src
    assert "asarray(sci_hwc, dtype=np.float64" not in src
    assert "asarray(wht_hwc, dtype=np.float64" not in src
    assert "_channel_mean" not in src
    assert "distance_transform_edt" in src  # tiled EDT still present


def test_crop_sup_views_never_call_copying_wht_property():
    """R1 behavioral guard: support views must not use the copying .wht."""
    class NoCopyAcc:
        def __init__(self, arr):
            self._out_wht = arr

        @property
        def wht(self):
            raise AssertionError("copying .wht property must not be used")

    a = np.arange(64, dtype=np.float32).reshape(8, 8)
    b = (a + 1.0).astype(np.float32)
    obj = types.SimpleNamespace(
        drizzle_sup_w1=NoCopyAcc(a), drizzle_sup_w2=NoCopyAcc(b),
        _drizzle_support_available=True,
    )
    w1, w2 = qm._drizzle_crop_sup_views(obj, {"x0": 1, "y0": 1, "x1": 4, "y1": 4})
    assert w1.shape == (3, 3) and w2.shape == (3, 3)
    # zero-copy view (shares memory with the resident support buffer)
    assert w1.base is not None and np.shares_memory(w1, a)
    # missing private view degrades fail-open to unavailable
    assert qm._drizzle_crop_sup_views(
        types.SimpleNamespace(drizzle_sup_w1=object(),
                              drizzle_sup_w2=object(),
                              _drizzle_support_available=True), {}
    ) == (None, None)


def test_sample_cap_is_exercised_by_population_above_cap():
    """R5: a population above the cap must produce an exactly capped sample."""
    n = 800
    total = n * n  # 640_000 > MAX_SAMPLE_COUNT
    st = dsd.sci_channel_stats(np.zeros((n, n), dtype=np.float32))
    assert st["count"] == total
    assert st["sample_stride"] == total // dsd.MAX_SAMPLE_COUNT
    assert st["sample_count"] == dsd.MAX_SAMPLE_COUNT


def test_summarize_extrema_topk_and_meta_caps():
    rng = np.random.default_rng(3)
    h, w = 64, 96
    sci = rng.normal(size=(h, w, 3)).astype(np.float32)
    wht = rng.normal(size=(h, w, 3)).astype(np.float32)
    w1 = np.abs(rng.random((h, w))).astype(np.float32)
    w2 = (w1 * w1 + 1e-3).astype(np.float32)
    mask = np.isfinite(w1) & (w1 > 0)
    sec = dsd.summarize_run(sci, wht, sup_w1=w1, sup_w2=w2, support_mask=mask)
    for st in sec["sci_stats"]:
        assert st["sample_count"] <= dsd.MAX_SAMPLE_COUNT
        assert st["sample_stride"] >= 1
    for recs in sec["support_extrema"]:
        assert len(recs) <= 2 * (dsd.EXTREMA_COUNT + 1)
    meta = sec["meta"]
    assert meta["boundary_algorithm"] == "row_tile_edt_halo"
    assert meta["boundary_halo"] >= 17
    assert meta["max_temporary_bytes"] < 8 * 1024 * 1024
    assert meta["degraded_sections"] == []
    assert sec["conditioning_candidates"]["local_reference_method"] == "bounded_tile_sample"


def test_summarize_artifact_is_bounded_and_json_safe():
    rng = np.random.default_rng(11)
    sci = rng.normal(size=(128, 128, 3)).astype(np.float32)
    wht = rng.normal(size=(128, 128, 3)).astype(np.float32)
    w1 = np.abs(rng.random((128, 128))).astype(np.float32)
    w2 = (w1 * w1 + 1e-3).astype(np.float32)
    mask = np.isfinite(w1) & (w1 > 0)
    sec = dsd.summarize_run(sci, wht, sup_w1=w1, sup_w2=w2, support_mask=mask)
    text = json.dumps({"sections": sec}, sort_keys=True, allow_nan=False)
    assert "array(" not in text
    assert len(text) < 500_000


# ---------------------------------------------------------------------------
# 5. lifecycle retention + seams
# ---------------------------------------------------------------------------


def test_lifecycle_retention_coalesces_checkpoints_and_keeps_terminal():
    diag = dsd.DrizzleScienceDiagnostics(run_token="r")
    for i in range(6000):
        diag.add_lifecycle({"stage": "checkpoint_save", "ts": float(i),
                            "generation": i, "frame_count": i})
    for stage in ("stop_requested", "drizzle_finalization_entered",
                  "fits_save", "memmap_cleanup_returned", "artifact_final"):
        diag.add_lifecycle({"stage": stage, "ts": 9.0})
    stages = diag.lifecycle_stages()
    assert "checkpoint_save" not in stages
    for terminal in ("stop_requested", "drizzle_finalization_entered",
                     "fits_save", "memmap_cleanup_returned", "artifact_final"):
        assert terminal in stages
    assert diag.coalesced["checkpoint_save"]["count"] == 6000
    payload = json.dumps(diag.to_dict(), sort_keys=True, allow_nan=False)
    assert len(payload) < 60_000
    assert len(diag.lifecycle) <= dsd.MAX_LIFECYCLE_EVENTS


def test_support_lifecycle_failopen_records_and_noop():
    diag = dsd.DrizzleScienceDiagnostics(run_token="r")
    stub = Stub(diag)
    stub.drizzle_sup_w1 = object()
    stub._drizzle_support_available = True
    qm._support_lifecycle_failopen(stub, "unit_stage", extra=1)
    assert "unit_stage" in diag.lifecycle_stages()
    qm._support_lifecycle_failopen(Stub(None), "unit_stage")


def test_stop_real_seam_records_lifecycle():
    diag = dsd.DrizzleScienceDiagnostics(run_token="r")

    class StopStub:
        def __init__(self):
            self._drizzle_science_diag = diag
            self.processing_active = True
            self.aligner = types.SimpleNamespace(stop_processing=False)
            self.autotuner = None
            self.quality_executor = None

        def update_progress(self, *a, **k):
            return None

        def stop_processing(self):
            return None

    obj = StopStub()
    SeestarQueuedStacker.stop(obj)
    assert "stop_requested" in diag.lifecycle_stages()


def test_coverage_render_lifecycle_entered_and_exited():
    diag = dsd.DrizzleScienceDiagnostics(run_token="r")
    obj = Stub(diag)
    obj.apply_coverage_render = False
    obj.update_progress = lambda *a, **k: None
    obj._emit_lifecycle = lambda *a, **k: None
    neff, status = qm._prepare_coverage_render(obj)
    assert status == "NOT_REQUESTED"
    stages = diag.lifecycle_stages()
    assert "coverage_render_entered" in stages and "coverage_render_exited" in stages


def test_memmap_cleanup_naming_truthful_and_returned():
    diag = dsd.DrizzleScienceDiagnostics(run_token="r")
    obj = types.SimpleNamespace(
        _drizzle_science_diag=diag,
        cumulative_sum_memmap=None, cumulative_wht_memmap=None,
        coverage_sup_w1_memmap=None, coverage_sup_w2_memmap=None,
        drizzle_sup_w1=object(), drizzle_sup_w2=object(),
    )
    SeestarQueuedStacker._close_memmaps(obj)
    stages = diag.lifecycle_stages()
    assert "memmap_cleanup_entered" in stages and "memmap_cleanup_returned" in stages
    assert "cleanup_reset" not in stages
    rec = next(r for r in diag.lifecycle if r["stage"] == "memmap_cleanup_returned")
    assert rec["drizzle_support_released"] is False
    assert rec["drizzle_sup_w1_present"] is True


def test_checkpoint_restore_production_ordering_records_lifecycle():
    restored = types.SimpleNamespace(
        counters={"frame_count": 7, "stacked_batches_count": 1,
                  "total_exposure_seconds": 70.0, "exposure_unknown_count": 0,
                  "exposure_min": 10.0, "exposure_max": 10.0},
        accumulators=[DrizzleAccumulator((4, 4)) for _ in range(3)],
        support_accumulators=(DrizzleAccumulator((4, 4), kernel="square"),
                              DrizzleAccumulator((4, 4), kernel="square")),
        completed_sources=[],
        session={"plan": {}, "input_roots": [], "reference": {}},
    )
    obj = types.SimpleNamespace(drizzle_group_size=50)
    SeestarQueuedStacker._restore_drizzle_checkpoint_runtime(obj, restored)
    assert getattr(obj, "_drizzle_science_diag", None) is None
    ref = make_wcs((4, 4))
    obj.reference_wcs_object = ref
    obj.drizzle_output_wcs = ref
    obj.output_folder = None
    obj.update_progress = lambda *a, **k: None
    SeestarQueuedStacker._init_drizzle_science_diagnostics(
        obj, "square", 1.0, 1.0, 1.0, restored
    )
    SeestarQueuedStacker._restore_drizzle_checkpoint_runtime(obj, restored)
    assert "checkpoint_restore" in obj._drizzle_science_diag.lifecycle_stages()


# ---------------------------------------------------------------------------
# 6. deposit timing (F9)
# ---------------------------------------------------------------------------


def _make_deposit_stub(diag, shape=(16, 16)):
    ref = make_wcs(shape)
    out_wcs, out_shape = build_output_grid(ref, shape, 2)
    obj = types.SimpleNamespace(
        _drizzle_science_diag=diag, reference_wcs_object=ref,
        drizzle_output_wcs=out_wcs,
        drizzle_accumulators=[DrizzleAccumulator(out_shape) for _ in range(3)],
        drizzle_sup_w1=DrizzleAccumulator(out_shape, kernel="square"),
        drizzle_sup_w2=DrizzleAccumulator(out_shape, kernel="square"),
        _drizzle_support_available=True, _drizzle_support_unavailable_reason=None,
        _drizzle_bg_anchor=None, _drizzle_frame_count=0,
    )
    return obj


def test_first_deposit_split_science_and_support():
    from astropy.io import fits

    diag = dsd.DrizzleScienceDiagnostics(run_token="r")
    obj = _make_deposit_stub(diag)
    data = np.ones((16, 16, 3), dtype=np.float32)
    wt = np.ones((16, 16), dtype=np.float32)
    header = fits.Header()
    tf = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ok = SeestarQueuedStacker._add_frame_to_drizzle_accumulators(obj, data, header, tf, wt)
    assert ok is True
    stages = diag.lifecycle_stages()
    assert "first_science_deposit" in stages and "first_support_deposit" in stages
    assert "first_deposit" not in stages
    sup_rec = next(r for r in diag.lifecycle if r["stage"] == "first_support_deposit")
    assert sup_rec["succeeded"] is True
    assert diag.contract["pixel_scale_ratio_source"] == "upstream_add_image_default"


# ---------------------------------------------------------------------------
# 7. finalization / crop alignment / persisted terminal events
# ---------------------------------------------------------------------------


def _make_save_obj(tmp_path, diag, kernel="lanczos2", pixfrac=1.0,
                   interior_wht=False, threshold=0.0, gradient_wht=False,
                   zero_wht=False):
    from astropy.io import fits

    obj = types.SimpleNamespace()
    obj.update_progress = lambda *a, **k: None
    obj._close_memmaps = types.MethodType(
        lambda self: SeestarQueuedStacker._close_memmaps(self), obj
    )
    obj._emit_lifecycle = lambda *a, **k: None
    obj._drizzle_science_diag = diag
    obj.save_final_as_float32 = True
    obj.preserve_linear_output = True
    obj.drizzle_wht_threshold = 0.0
    obj.drizzle_wht_threshold_effective = threshold
    obj.images_in_cumulative_stack = 1
    obj.total_exposure_seconds = 10.0
    obj.output_folder = str(tmp_path)
    obj.output_filename = "out.fit"
    obj.current_stack_header = fits.Header()
    obj.reference_header_for_wcs = None
    obj.drizzle_active_session = True
    obj.is_mosaic_run = False
    obj.drizzle_mode = "Final"
    obj.drizzle_kernel = kernel
    obj.drizzle_pixfrac = pixfrac
    obj.drizzle_scale = 2.0
    obj.drizzle_output_wcs = make_wcs((16, 16))
    obj.drizzle_fillval = "0.0"
    obj.finalization_mode = qm.FINALIZATION_MODE_DRIZZLE
    obj.reproject_between_batches = False
    obj.cumulative_sum_memmap = None
    obj.cumulative_wht_memmap = None
    obj.apply_coverage_render = False
    obj.apply_feathering = False
    obj.apply_low_wht_mask = False
    obj.processing_error = None
    obj.save_drizzle_wht = False
    obj._validate_drizzle_science = types.MethodType(
        qm.SeestarQueuedStacker._validate_drizzle_science, obj
    )

    accs = [DrizzleAccumulator((16, 16), kernel=kernel, pixfrac=pixfrac)
            for _ in range(3)]
    rng = np.random.default_rng(7)
    for c, acc in enumerate(accs):
        acc._out_img[:] = rng.normal(size=(16, 16)).astype(np.float32) + 2.0 + c
        if zero_wht:
            acc._out_wht[:] = 0.0
        elif gradient_wht:
            w = np.zeros((16, 16), dtype=np.float32)
            w[4:8, 4:12] = 0.2
            w[8:12, 4:12] = 2.0
            acc._out_wht[:] = w
        elif interior_wht:
            w = np.zeros((16, 16), dtype=np.float32)
            w[4:12, 4:12] = 1.0
            acc._out_wht[:] = w
        else:
            acc._out_wht[:] = 1.0
    obj.drizzle_accumulators = accs
    obj.drizzle_sup_w1 = DrizzleAccumulator((16, 16), kernel="square")
    obj.drizzle_sup_w2 = DrizzleAccumulator((16, 16), kernel="square")
    obj.drizzle_sup_w1._out_img[:] = 3.0
    obj.drizzle_sup_w2._out_img[:] = 9.0
    obj.drizzle_sup_w1._out_wht[:] = 1.0
    obj.drizzle_sup_w2._out_wht[:] = 1.0
    obj._drizzle_support_available = True
    obj._drizzle_support_unavailable_reason = None
    return obj


def test_artifact_persists_terminal_lifecycle_events(tmp_path):
    diag = dsd.DrizzleScienceDiagnostics(run_token="run1", output_folder=str(tmp_path))
    diag.set_run_config(kernel="lanczos2", scale=2.0, pixfrac_requested=1.0,
                        pixfrac_effective=1.0)
    diag.set_geometry(dsd.geometry_diagnostic(make_wcs((16, 16)), make_wcs((16, 16))))
    obj = _make_save_obj(tmp_path, diag)
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_drizzle", preserve_linear_output=True
    )
    payload = json.loads((tmp_path / dsd.ARTIFACT_FILENAME).read_text())
    persisted = [r["stage"] for r in payload["lifecycle"]]
    for terminal in ("memmap_cleanup_returned", "drizzle_finalization_returned",
                     "fits_save", "artifact_final"):
        assert terminal in persisted, terminal
    assert payload["finalization_state"]["fits_saved"] is True
    assert payload["finalization_state"]["outcome"] == "success"
    assert len(payload["sci_stats"]) == 3
    assert payload["crop"]["width"] == 16


def test_fits_failure_path_still_persists_terminal_events(tmp_path, monkeypatch):
    diag = dsd.DrizzleScienceDiagnostics(run_token="runf", output_folder=str(tmp_path))
    diag.set_run_config(kernel="square", scale=2.0, pixfrac_requested=1.0,
                        pixfrac_effective=1.0)
    obj = _make_save_obj(tmp_path, diag, kernel="square")

    def boom(self, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(qm.fits.HDUList, "writeto", boom)
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_drizzle", preserve_linear_output=True
    )
    payload = json.loads((tmp_path / dsd.ARTIFACT_FILENAME).read_text())
    persisted = [r["stage"] for r in payload["lifecycle"]]
    assert "fits_save" in persisted and "artifact_final" in persisted
    rec = next(r for r in payload["lifecycle"] if r["stage"] == "fits_save")
    assert rec.get("success") is False
    assert payload["finalization_state"]["outcome"] == "fits_failed"


def test_early_return_marks_artifact_final_failed(tmp_path):
    """R3: a terminal no-support route persists artifact_final with outcome."""
    diag = dsd.DrizzleScienceDiagnostics(run_token="runz", output_folder=str(tmp_path))
    diag.set_run_config(kernel="square", scale=2.0, pixfrac_requested=1.0,
                        pixfrac_effective=1.0)
    obj = _make_save_obj(tmp_path, diag, kernel="square", zero_wht=True)
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_drizzle", preserve_linear_output=True
    )
    payload = json.loads((tmp_path / dsd.ARTIFACT_FILENAME).read_text())
    stages = [r["stage"] for r in payload["lifecycle"]]
    assert "artifact_final" in stages
    rec = next(r for r in payload["lifecycle"] if r["stage"] == "artifact_final")
    assert rec["outcome"] == "failed" and rec["reason"] == "no_support"
    assert rec["success"] is False


def test_crop_alignment_no_threshold_matches_final_grid(tmp_path):
    diag = dsd.DrizzleScienceDiagnostics(run_token="runc", output_folder=str(tmp_path))
    diag.set_run_config(kernel="lanczos2", scale=2.0, pixfrac_requested=1.0,
                        pixfrac_effective=1.0)
    obj = _make_save_obj(tmp_path, diag, interior_wht=True, threshold=0.0)
    expected = [acc.finalize("divide")[4:12, 4:12] for acc in obj.drizzle_accumulators]
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_drizzle", preserve_linear_output=True
    )
    payload = json.loads((tmp_path / dsd.ARTIFACT_FILENAME).read_text())
    assert payload["crop"] == {"x0": 4, "y0": 4, "x1": 12, "y1": 12,
                               "width": 8, "height": 8}
    assert payload["sections_meta"]["shape_hwc"] == [8, 8, 3]
    for c in range(3):
        fin = expected[c][np.isfinite(expected[c])]
        assert payload["sci_stats"][c]["min"] == pytest.approx(float(fin.min()), rel=1e-5)
        assert payload["sci_stats"][c]["max"] == pytest.approx(float(fin.max()), rel=1e-5)
        for r in payload["wht_diagnostics"][c]["extrema"]:
            assert 0 <= r["row"] < 8 and 0 <= r["col"] < 8


def test_crop_square_threshold_masking_changes_finite_count(tmp_path):
    """R5: compare the same fixture with threshold 0 vs nonzero, non-trivially."""
    def run(tag, thr):
        tmp = tmp_path / tag
        tmp.mkdir()
        diag = dsd.DrizzleScienceDiagnostics(run_token=tag, output_folder=str(tmp))
        diag.set_run_config(kernel="square", scale=2.0, pixfrac_requested=1.0,
                            pixfrac_effective=1.0)
        obj = _make_save_obj(tmp, diag, kernel="square", gradient_wht=True,
                             threshold=thr)
        qm.SeestarQueuedStacker._save_final_stack(
            obj, output_filename_suffix="_drizzle", preserve_linear_output=True
        )
        return json.loads((tmp / dsd.ARTIFACT_FILENAME).read_text())

    off = run("off", 0.0)
    on = run("on", 0.5)
    assert off["effective_wht_policy"].get("fraction", 0.0) == 0.0
    assert on["effective_wht_policy"].get("fraction") == pytest.approx(0.5)
    assert off["crop"]["width"] == on["crop"]["width"] == 8
    assert off["sections_meta"]["shape_hwc"] == on["sections_meta"]["shape_hwc"] == [8, 8, 3]
    # the nonzero threshold genuinely removed finite science pixels
    for c in range(3):
        assert on["sci_stats"][c]["finite_count"] < off["sci_stats"][c]["finite_count"]
        assert on["sci_stats"][c]["finite_count"] < 64
        for r in on["wht_diagnostics"][c]["extrema"]:
            assert 0 <= r["row"] < 8 and 0 <= r["col"] < 8


def test_diagnostic_failure_cannot_change_science_or_abort(tmp_path, monkeypatch):
    diag = dsd.DrizzleScienceDiagnostics(run_token="run2", output_folder=str(tmp_path))
    diag.set_run_config(kernel="square", scale=2.0, pixfrac_requested=1.0,
                        pixfrac_effective=1.0)
    obj = _make_save_obj(tmp_path, diag, kernel="square")

    def boom(*a, **k):
        raise RuntimeError("diagnostic calculation exploded")

    monkeypatch.setattr(dsd, "summarize_run", boom)
    monkeypatch.setattr(diag, "write", boom)
    snaps = [np.array(obj.drizzle_accumulators[1]._out_img, copy=True),
             np.array(obj.drizzle_accumulators[1]._out_wht, copy=True)]
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_drizzle", preserve_linear_output=True
    )
    assert _bitwise_equal(snaps[0], obj.drizzle_accumulators[1]._out_img)
    assert _bitwise_equal(snaps[1], obj.drizzle_accumulators[1]._out_wht)
    assert obj.final_stacked_path is not None


def test_science_bitwise_identical_with_diagnostics_enabled_and_failing(tmp_path, monkeypatch):
    def run(diag_obj, tag):
        obj = _make_save_obj(tmp_path, diag_obj, kernel="square")
        return obj, [np.array(obj.drizzle_accumulators[c]._out_img, copy=True)
                     for c in range(3)]

    diag_ok = dsd.DrizzleScienceDiagnostics(run_token="ok", output_folder=str(tmp_path))
    diag_ok.set_run_config(kernel="square", scale=2.0, pixfrac_requested=1.0,
                           pixfrac_effective=1.0)
    obj_a, snaps_a = run(diag_ok, "a")
    qm.SeestarQueuedStacker._save_final_stack(
        obj_a, output_filename_suffix="_drizzle", preserve_linear_output=True
    )

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(dsd, "summarize_run", boom)
    diag_bad = dsd.DrizzleScienceDiagnostics(run_token="bad", output_folder=str(tmp_path))
    diag_bad.set_run_config(kernel="square", scale=2.0, pixfrac_requested=1.0,
                            pixfrac_effective=1.0)
    obj_b, snaps_b = run(diag_bad, "b")
    qm.SeestarQueuedStacker._save_final_stack(
        obj_b, output_filename_suffix="_drizzle", preserve_linear_output=True
    )
    for a, b in zip(snaps_a, snaps_b):
        assert _bitwise_equal(a, b)


def test_persist_noop_without_collector():
    obj = types.SimpleNamespace(final_stacked_path=None)
    qm._persist_drizzle_science_diagnostics(obj)
    qm._compute_drizzle_science_diagnostics(obj, None, None)
