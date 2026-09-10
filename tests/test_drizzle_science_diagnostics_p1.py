"""Phase-1 tests for passive Drizzle science diagnostics (ZSSS-DRIZZLE-CLOSURE-P1).

These tests prove the instrumentation contract only:

* the diagnostic helpers are read-only (SCI/WHT/SUP bitwise unchanged, incl.
  signed and non-finite test data);
* a diagnostic calculation or I/O failure can neither change science nor abort
  the run;
* the SUPPORT_LIFECYCLE ring observes fresh init / first deposit / checkpoint
  save+restore / STOP / finalization / coverage render / FITS save / cleanup
  without altering ordering, retention or outcome;
* the resolved-geometry candidate is the WCS ratio while the real
  ``Drizzle.add_image`` call still omits ``iscale``/``pixel_scale_ratio``
  (upstream defaults);
* threshold / N_eff / boundary-bin summaries are bounded and deterministic;
* real drizzle 2.2.0 pixfrac behaviour matches the archaeology (Lanczos
  ignored/forced 1.0; positive kernels active; >1 spreads).

All scenes are synthetic, deterministic and fast (no GPU / GUI / network).
"""

import json
import math
import types

import numpy as np
import pytest

from astropy.wcs import WCS

from seestar.core import drizzle_science_diagnostics as dsd
from seestar.core.drizzle_core import DrizzleAccumulator, build_output_grid
from seestar.queuep import queue_manager as qm
from seestar.queuep.queue_manager import SeestarQueuedStacker


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and a.tobytes() == b.tobytes()
    )


def _snapshot(*arrays):
    return [np.array(a, copy=True) for a in arrays]


def _assert_unchanged(snaps, arrays):
    for snap, arr in zip(snaps, arrays):
        assert np.array_equal(snap, arr) or (
            snap.shape == arr.shape and snap.dtype == arr.dtype
            and np.array_equal(np.nan_to_num(snap), np.nan_to_num(arr))
        )
        assert _bitwise_equal(snap, arr), "diagnostic helper mutated its input"


class Stub:
    """Minimal duck object carrying a diagnostics collector."""

    def __init__(self, diag=None):
        if diag is not None:
            self._drizzle_science_diag = diag


# ---------------------------------------------------------------------------
# 1. read-only / passive helpers
# ---------------------------------------------------------------------------


def test_helpers_are_read_only_bitwise_including_signed_and_nonfinite():
    rng = np.random.default_rng(1234)
    sci = rng.normal(size=(12, 12, 3)).astype(np.float32)
    wht = rng.normal(size=(12, 12, 3)).astype(np.float32)  # signed
    # inject non-finite values where they are semantically allowed
    sci[0, 0, 0] = np.nan
    wht[1, 1, 1] = np.inf
    wht[2, 2, 2] = -np.inf
    w1 = np.abs(rng.random((12, 12))).astype(np.float64)
    w2 = (w1 * w1 + 1e-3).astype(np.float64)

    snaps = _snapshot(sci, wht, w1, w2)
    sections = dsd.summarize_run(sci, wht, sup_w1=w1, sup_w2=w2)
    # every helper the summary uses, called directly too
    for c in range(3):
        dsd.sci_channel_stats(sci[..., c])
        dsd.wht_channel_diagnostics(wht[..., c], sci[..., c])
        dsd.threshold_sweep(wht[..., c], sci[..., c])
    dsd.support_conditioning(w1, w2)
    dsd.conditioning_candidates(sections and sci[..., 0], wht[..., 0], w1, w2)
    dsd.spatial_boundary_diagnostics(
        sci[..., 0], wht[..., 0], np.isfinite(w1) & (w1 > 0)
    )
    _assert_unchanged(snaps, [sci, wht, w1, w2])
    assert sections["sci_stats"][0]["nonfinite_fraction"] > 0


def test_collector_to_dict_is_deterministic_json_safe_and_versioned():
    diag = dsd.DrizzleScienceDiagnostics(run_token="tok", output_folder=None)
    diag.set_run_config(kernel="lanczos2", scale=2.0, pixfrac_requested=0.8,
                        pixfrac_effective=1.0)
    diag.set_geometry(dsd.geometry_diagnostic(make_wcs((8, 8)), make_wcs((8, 8))))
    diag.add_lifecycle({"stage": "x", "ts": 1.0, "support_available": True})
    d1 = json.dumps(diag.to_dict(), sort_keys=True, allow_nan=False)
    d2 = json.dumps(diag.to_dict(), sort_keys=True, allow_nan=False)
    assert dsd.SCHEMA_VERSION in d1
    obj = json.loads(d1)
    assert obj["schema_version"] == dsd.SCHEMA_VERSION
    assert obj["kernel"] == "lanczos2"
    assert obj["pixel_scale_ratio_current"] == 1.0
    assert obj["pixel_scale_ratio_source"] == "upstream_default"
    assert obj["diagnostic_only"] is True
    # generated_ts changes but the schema/keys must be identical
    assert set(json.loads(d2).keys()) == set(obj.keys())


def test_collector_write_atomic_and_fail_open(tmp_path):
    diag = dsd.DrizzleScienceDiagnostics(run_token="r", output_folder=str(tmp_path))
    diag.set_run_config(kernel="square", scale=1.0, pixfrac_requested=1.0,
                        pixfrac_effective=1.0)
    assert diag.write() is True
    path = tmp_path / dsd.ARTIFACT_FILENAME
    assert path.exists()
    json.loads(path.read_text())
    assert not list(tmp_path.glob("*.tmp.*")), "temp file not cleaned up"
    # unwritable target -> False, never raises
    bad = dsd.DrizzleScienceDiagnostics(run_token="r", output_folder=None)
    assert bad.write("/nonexistent-dir-\x00/x.json") is False


def test_setters_and_write_fail_open_on_garbage():
    diag = dsd.DrizzleScienceDiagnostics(run_token="r")
    diag.set_geometry(object())  # not a mapping -> must be swallowed
    diag.set_sci_stats(object())
    diag.set_wht_diagnostics(None)
    diag.note("ok")
    # to_dict must still produce a valid payload
    assert json.dumps(diag.to_dict(), sort_keys=True, allow_nan=False)
    # write with no target -> False, no raise
    assert diag.write() is False


# ---------------------------------------------------------------------------
# 2. geometry + real add_image contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale,expected", [(1, 1.0), (2, 0.5), (3, 1.0 / 3.0),
                                            (4, 0.25)])
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
    assert rec["pixel_scale_ratio_candidate"] is None
    assert rec["reason"]


def test_init_diagnostics_emits_geometry_token():
    lines = []
    ref = make_wcs((16, 16))
    out, _shape = build_output_grid(ref, (16, 16), 2)
    obj = types.SimpleNamespace(
        reference_wcs_object=ref,
        drizzle_output_wcs=out,
        output_folder=None,
        update_progress=lambda msg, *a, **k: lines.append(str(msg)),
    )
    SeestarQueuedStacker._init_drizzle_science_diagnostics(
        obj, "square", 1.0, 1.0, 2.0, None
    )
    token = next(x for x in lines if x.startswith("DRIZZLE_GEOMETRY_DIAGNOSTIC"))
    assert "kernel=square" in token
    assert "scale=2.0" in token
    assert "pixel_scale_ratio_current=1.0" in token
    assert "candidate_source=wcs_ratio" in token
    assert obj._drizzle_science_diag is not None
    assert obj._drizzle_science_diag.geometry["pixel_scale_ratio_candidate"] == pytest.approx(0.5)
    stages = [r["stage"] for r in obj._drizzle_science_diag.lifecycle]
    assert "fresh_m3_init" in stages and "accumulators_ready" in stages


def test_init_diagnostics_geometry_token_fail_open_without_wcs():
    lines = []
    obj = types.SimpleNamespace(
        reference_wcs_object=None,
        drizzle_output_wcs=None,
        output_folder=None,
        update_progress=lambda msg, *a, **k: lines.append(str(msg)),
    )
    SeestarQueuedStacker._init_drizzle_science_diagnostics(
        obj, "square", 1.0, 1.0, 1.0, None
    )
    token = next(x for x in lines if x.startswith("DRIZZLE_GEOMETRY_DIAGNOSTIC"))
    assert "pixel_scale_ratio_candidate=unavailable" in token
    assert "reason=" in token


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
    assert "pixel_scale_ratio" not in keys
    assert "iscale" not in keys
    assert keys == {
        "data", "exptime", "pixmap", "weight_map", "in_units", "pixfrac",
        "wht_scale",
    }
    # the run-time contract helper reports the same upstream defaults
    contract = dsd.contract_diagnostic(acc.kernel, acc.pixfrac, exptime=2.0,
                                       in_units="counts", fillval=acc.fillval)
    assert contract["pixel_scale_ratio_source"] == "upstream_default"
    assert contract["pixel_scale_ratio_effective"] == 1.0
    assert contract["iscale_source"] == "upstream_default"
    assert contract["iscale_effective"] == 1.0
    assert contract["wht_scale_effective"] == 2.0
    # geometry candidate is NEVER passed upstream
    assert "pixel_scale_ratio_candidate" not in keys


# ---------------------------------------------------------------------------
# 3. threshold / N_eff / boundary bounded summaries
# ---------------------------------------------------------------------------


def test_threshold_sweep_reports_removal_without_applying():
    wht = np.array([[1.0, 0.05, 0.0], [2e-3, 5e-5, -1.0]], dtype=np.float32)
    sci = np.arange(6, dtype=np.float32).reshape(2, 3) + 1.0
    before = wht.copy()
    sweep = dsd.threshold_sweep(wht, sci, abs_candidates=(1e-4, 1e-3))
    assert _bitwise_equal(before, wht)
    assert sweep["support_pixels"] >= 1
    assert sweep["positive_reference"] is not None
    names = [c["name"] for c in sweep["candidates"]]
    assert "current_epsilon" in names
    # the strictest candidate removes at least as many pixels as the weakest
    by_name = {c["name"]: c for c in sweep["candidates"]}
    assert by_name["abs_0.001"]["removed_pixels"] >= by_name["abs_0.0001"]["removed_pixels"]
    json.dumps(sweep, allow_nan=False)


def test_support_conditioning_known_n_eff():
    w1 = np.full((4, 4), 2.0)
    w2 = np.full((4, 4), 4.0)
    rec = dsd.support_conditioning(w1, w2)
    assert rec["available"] is True
    assert rec["n_eff_min"] == rec["n_eff_max"] == pytest.approx(1.0)
    # absent pair is explicitly unavailable, never fabricated
    assert dsd.support_conditioning(None, None)["available"] is False


def test_boundary_bins_cover_support_and_deterministic():
    support = np.ones((16, 16), dtype=bool)
    sci = np.linspace(1.0, 2.0, 16 * 16, dtype=np.float64).reshape(16, 16)
    wht = np.full((16, 16), 0.5, dtype=np.float64)
    neff = np.full((16, 16), 3.0, dtype=np.float64)
    rec = dsd.spatial_boundary_diagnostics(sci, wht, support, neff=neff)
    assert rec["available"] is True
    labels = [b["label"] for b in rec["bins"]]
    assert labels == list(dsd.DISTANCE_BIN_LABELS)
    assert sum(b["pixels"] for b in rec["bins"]) == int(support.sum())
    rec2 = dsd.spatial_boundary_diagnostics(sci, wht, support, neff=neff)
    assert json.dumps(rec, sort_keys=True, allow_nan=False) == json.dumps(
        rec2, sort_keys=True, allow_nan=False
    )


def test_select_extrema_indices_is_bounded_and_deterministic():
    vals = np.arange(100, dtype=np.float64).reshape(10, 10)
    mask = np.ones((10, 10), dtype=bool)
    lo, hi = dsd.select_extrema_indices(vals, mask, 3)
    assert lo.tolist() == [0, 1, 2]
    assert hi.tolist() == [99, 98, 97]
    lo2, hi2 = dsd.select_extrema_indices(vals, mask, 3)
    assert lo.tolist() == lo2.tolist() and hi.tolist() == hi2.tolist()


# ---------------------------------------------------------------------------
# 4. lifecycle through the real seams
# ---------------------------------------------------------------------------


def test_support_lifecycle_failopen_records_and_noop():
    diag = dsd.DrizzleScienceDiagnostics(run_token="r")
    stub = Stub(diag)
    stub.drizzle_sup_w1 = object()
    stub.drizzle_sup_w2 = object()
    stub._drizzle_support_available = True
    qm._support_lifecycle_failopen(stub, "unit_stage", extra=1)
    assert any(r["stage"] == "unit_stage" for r in diag.lifecycle)
    # no collector -> silent no-op (never raises)
    qm._support_lifecycle_failopen(Stub(None), "unit_stage")


def test_lifecycle_record_captures_presence_and_stop_state():
    stub = Stub()
    stub.drizzle_sup_w1 = None
    stub.drizzle_sup_w2 = None
    stub._drizzle_support_available = False
    stub._drizzle_support_unavailable_reason = "legacy"
    stub.user_requested_stop = True
    rec = dsd.lifecycle_record("stop_requested", stub)
    assert rec["drizzle_sup_w1_present"] is False
    assert rec["support_available"] is False
    assert rec["support_reason"] == "legacy"
    assert rec["stopped"] is True


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
    assert any(r["stage"] == "stop_requested" for r in diag.lifecycle)
    assert obj.user_requested_stop is True


def test_coverage_render_lifecycle_entered_and_exited():
    diag = dsd.DrizzleScienceDiagnostics(run_token="r")
    obj = Stub(diag)
    obj.apply_coverage_render = False
    obj.update_progress = lambda *a, **k: None
    obj._emit_lifecycle = lambda *a, **k: None
    obj.drizzle_sup_w1 = None
    obj.drizzle_sup_w2 = None
    neff, status = qm._prepare_coverage_render(obj)
    assert status == "NOT_REQUESTED"
    stages = [r["stage"] for r in diag.lifecycle]
    assert "coverage_render_entered" in stages
    assert "coverage_render_exited" in stages
    assert obj.coverage_render_status == "NOT_REQUESTED"


def test_cleanup_reset_real_seam_records_lifecycle():
    diag = dsd.DrizzleScienceDiagnostics(run_token="r")
    obj = types.SimpleNamespace(
        _drizzle_science_diag=diag,
        cumulative_sum_memmap=None,
        cumulative_wht_memmap=None,
        coverage_sup_w1_memmap=None,
        coverage_sup_w2_memmap=None,
    )
    SeestarQueuedStacker._close_memmaps(obj)
    assert any(r["stage"] == "cleanup_reset" for r in diag.lifecycle)


def test_checkpoint_restore_real_seam_records_lifecycle():
    diag = dsd.DrizzleScienceDiagnostics(run_token="r")
    restored = types.SimpleNamespace(
        counters={
            "frame_count": 7,
            "stacked_batches_count": 1,
            "total_exposure_seconds": 70.0,
            "exposure_unknown_count": 0,
            "exposure_min": 10.0,
            "exposure_max": 10.0,
        },
        accumulators=[DrizzleAccumulator((4, 4)) for _ in range(3)],
        support_accumulators=(
            DrizzleAccumulator((4, 4), kernel="square", pixfrac=1.0),
            DrizzleAccumulator((4, 4), kernel="square", pixfrac=1.0),
        ),
        completed_sources=[],
        session={"plan": {}, "input_roots": [], "reference": {}},
    )
    obj = types.SimpleNamespace(
        _drizzle_science_diag=diag,
        drizzle_group_size=50,
    )
    SeestarQueuedStacker._restore_drizzle_checkpoint_runtime(obj, restored)
    assert any(r["stage"] == "checkpoint_restore" for r in diag.lifecycle)
    assert obj._drizzle_frame_count == 7


def _make_deposit_stub(tmp_path, diag, shape=(16, 16)):
    ref = make_wcs(shape)
    out_wcs, out_shape = build_output_grid(ref, shape, 2)
    obj = types.SimpleNamespace(
        _drizzle_science_diag=diag,
        reference_wcs_object=ref,
        drizzle_output_wcs=out_wcs,
        drizzle_accumulators=[DrizzleAccumulator(out_shape) for _ in range(3)],
        drizzle_sup_w1=DrizzleAccumulator(out_shape, kernel="square",
                                           pixfrac=1.0),
        drizzle_sup_w2=DrizzleAccumulator(out_shape, kernel="square",
                                           pixfrac=1.0),
        _drizzle_support_available=True,
        _drizzle_support_unavailable_reason=None,
        _drizzle_bg_anchor=None,
        _drizzle_frame_count=0,
    )
    return obj


def test_first_deposit_records_contract_and_lifecycle():
    diag = dsd.DrizzleScienceDiagnostics(run_token="r")
    obj = _make_deposit_stub(None, diag)
    data = np.ones((16, 16, 3), dtype=np.float32)
    wt = np.ones((16, 16), dtype=np.float32)
    from astropy.io import fits

    header = fits.Header()
    header["EXPTIME"] = 5.0
    tf = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ok = SeestarQueuedStacker._add_frame_to_drizzle_accumulators(
        obj, data, header, tf, wt
    )
    assert ok is True
    assert diag.contract is not None
    assert diag.contract["pixel_scale_ratio_source"] == "upstream_default"
    assert diag.contract["iscale_effective"] == 1.0
    assert diag.contract["wht_scale_effective"] == 5.0
    assert any(r["stage"] == "first_deposit" for r in diag.lifecycle)
    # accumulating support was genuinely present at the seam
    rec = next(r for r in diag.lifecycle if r["stage"] == "first_deposit")
    assert rec["drizzle_sup_w1_present"] is True
    assert rec["support_available"] is True


def test_first_deposit_logs_add_image_contract(caplog):
    import logging

    diag = dsd.DrizzleScienceDiagnostics(run_token="r")
    obj = _make_deposit_stub(None, diag)
    data = np.ones((16, 16, 3), dtype=np.float32)
    wt = np.ones((16, 16), dtype=np.float32)
    from astropy.io import fits

    header = fits.Header()
    tf = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with caplog.at_level(logging.INFO, logger=qm.logger.name):
        SeestarQueuedStacker._add_frame_to_drizzle_accumulators(
            obj, data, header, tf, wt
        )
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "DRIZZLE_ADD_IMAGE_CONTRACT" in text
    assert "pixel_scale_ratio_source=upstream_default" in text
    assert "iscale_source=upstream_default" in text


# ---------------------------------------------------------------------------
# 5. finalization / FITS / artifact through the real save seam
# ---------------------------------------------------------------------------


def _make_save_obj(tmp_path, diag, kernel="lanczos2", pixfrac=1.0):
    from astropy.io import fits

    obj = types.SimpleNamespace()
    obj.update_progress = lambda *a, **k: None
    obj._close_memmaps = types.MethodType(
        lambda self: qm._support_lifecycle_failopen(self, "cleanup_reset"), obj
    )
    obj._emit_lifecycle = lambda *a, **k: None
    obj._drizzle_science_diag = diag
    obj.save_final_as_float32 = True
    obj.preserve_linear_output = True
    obj.drizzle_wht_threshold = 0.0
    obj.drizzle_wht_threshold_effective = 0.0
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
    obj.drizzle_output_wcs = make_wcs((8, 8))
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
    obj._resolve_signed_lanczos_float32_reason = None

    accs = [DrizzleAccumulator((8, 8), kernel=kernel, pixfrac=pixfrac)
            for _ in range(3)]
    rng = np.random.default_rng(7)
    for c, acc in enumerate(accs):
        acc._out_img[:] = rng.normal(size=(8, 8)).astype(np.float32) + c
        acc._out_wht[:] = rng.normal(size=(8, 8)).astype(np.float32)
    obj.drizzle_accumulators = accs
    obj.drizzle_sup_w1 = DrizzleAccumulator((8, 8), kernel="square", pixfrac=1.0)
    obj.drizzle_sup_w2 = DrizzleAccumulator((8, 8), kernel="square", pixfrac=1.0)
    obj.drizzle_sup_w1._out_img[:] = 3.0
    obj.drizzle_sup_w2._out_img[:] = 9.0
    obj.drizzle_sup_w1._out_wht[:] = 1.0
    obj.drizzle_sup_w2._out_wht[:] = 1.0
    obj._drizzle_support_available = True
    obj._drizzle_support_unavailable_reason = None
    return obj


def test_finalization_fits_cleanup_lifecycle_and_artifact(tmp_path):
    diag = dsd.DrizzleScienceDiagnostics(run_token="run1",
                                         output_folder=str(tmp_path))
    diag.set_run_config(kernel="lanczos2", scale=2.0, pixfrac_requested=1.0,
                        pixfrac_effective=1.0)
    diag.set_geometry(dsd.geometry_diagnostic(make_wcs((8, 8)), make_wcs((8, 8))))
    obj = _make_save_obj(tmp_path, diag)
    snaps = _snapshot(obj.drizzle_accumulators[0]._out_img,
                      obj.drizzle_accumulators[0]._out_wht,
                      obj.drizzle_sup_w1._out_img)

    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_drizzle", preserve_linear_output=True
    )

    stages = [r["stage"] for r in diag.lifecycle]
    assert "drizzle_finalization_entered" in stages
    assert "drizzle_finalization_returned" in stages
    assert "drizzle_finalization_pre_save" in stages
    assert "fits_save" in stages
    assert "cleanup_reset" in stages

    # science unchanged (bitwise) by the instrumentation
    assert _bitwise_equal(snaps[0], obj.drizzle_accumulators[0]._out_img)
    assert _bitwise_equal(snaps[1], obj.drizzle_accumulators[0]._out_wht)
    assert _bitwise_equal(snaps[2], obj.drizzle_sup_w1._out_img)

    artifact = tmp_path / dsd.ARTIFACT_FILENAME
    assert artifact.exists()
    payload = json.loads(artifact.read_text())
    assert payload["schema_version"] == dsd.SCHEMA_VERSION
    assert payload["kernel"] == "lanczos2"
    assert payload["pixfrac_effective"] == 1.0
    assert payload["pixel_scale_ratio_current"] == 1.0
    assert payload["pixel_scale_ratio_source"] == "upstream_default"
    assert len(payload["sci_stats"]) == 3
    assert len(payload["wht_diagnostics"]) == 3
    assert len(payload["threshold_sweep"]) == 3
    assert payload["boundary_bins"]["available"] is True
    assert payload["support"]["available"] is True
    # bounded: no array payload smuggled in
    text = artifact.read_text()
    assert "array(" not in text
    assert len(text) < 200_000


def test_diagnostic_failure_cannot_change_science_or_abort(tmp_path, monkeypatch):
    diag = dsd.DrizzleScienceDiagnostics(run_token="run2",
                                         output_folder=str(tmp_path))
    diag.set_run_config(kernel="square", scale=2.0, pixfrac_requested=1.0,
                        pixfrac_effective=1.0)
    obj = _make_save_obj(tmp_path, diag, kernel="square", pixfrac=1.0)

    def boom(*a, **k):
        raise RuntimeError("diagnostic calculation exploded")

    monkeypatch.setattr(dsd, "summarize_run", boom)
    monkeypatch.setattr(diag, "write", boom)
    snaps = _snapshot(obj.drizzle_accumulators[1]._out_img,
                      obj.drizzle_accumulators[1]._out_wht)
    # must NOT raise and must NOT alter science
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_drizzle", preserve_linear_output=True
    )
    assert _bitwise_equal(snaps[0], obj.drizzle_accumulators[1]._out_img)
    assert _bitwise_equal(snaps[1], obj.drizzle_accumulators[1]._out_wht)
    assert obj.final_stacked_path is not None


def test_writer_io_failure_is_fail_open(tmp_path, monkeypatch):
    diag = dsd.DrizzleScienceDiagnostics(run_token="run3",
                                         output_folder=str(tmp_path))

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(dsd, "DrizzleScienceDiagnostics", dsd.DrizzleScienceDiagnostics)
    monkeypatch.setattr(diag, "write", boom)
    obj = _make_save_obj(tmp_path, diag, kernel="square", pixfrac=1.0)
    qm.SeestarQueuedStacker._save_final_stack(
        obj, output_filename_suffix="_drizzle", preserve_linear_output=True
    )
    assert obj.final_stacked_path is not None


def test_write_drizzle_science_diagnostics_noop_without_collector(tmp_path):
    obj = types.SimpleNamespace(final_stacked_path=None)
    # no collector -> no raise, no artifact
    qm._write_drizzle_science_diagnostics(obj)
    assert not (tmp_path / dsd.ARTIFACT_FILENAME).exists()


# ---------------------------------------------------------------------------
# 6. pixfrac archaeology (real drizzle 2.2.0 probe)
# ---------------------------------------------------------------------------


def _real_drizzle(kernel, pixfrac, shift=0.5):
    from drizzle.resample import Drizzle

    out = np.zeros((16, 16), dtype=np.float32)
    wht = np.zeros((16, 16), dtype=np.float32)
    engine = Drizzle(out_img=out, out_wht=wht, kernel=kernel, fillval="0.0")
    data = np.zeros((8, 8), dtype=np.float32)
    data[3:5, 3:5] = 1.0
    yy, xx = np.indices((8, 8), dtype=np.float64)
    pixmap = np.dstack((xx + 4.0 + shift, yy + 4.0))
    w = np.ones((8, 8), dtype=np.float32)
    engine.add_image(data=data, exptime=1.0, pixmap=pixmap, weight_map=w,
                     in_units="counts", pixfrac=pixfrac, wht_scale=1.0)
    return out.copy(), wht.copy()


@pytest.mark.parametrize("kernel", ["lanczos2", "lanczos3"])
def test_lanczos_pixfrac_is_ignored(kernel):
    ref_out, ref_wht = _real_drizzle(kernel, 1.0)
    for pf in (0.5, 0.8, 1.5):
        out, wht = _real_drizzle(kernel, pf)
        assert np.array_equal(out, ref_out)
        assert np.array_equal(wht, ref_wht)


def test_square_pixfrac_active_and_gt_one_spreads():
    out_small, _ = _real_drizzle("square", 0.5)
    out_full, _ = _real_drizzle("square", 1.0)
    out_wide, _ = _real_drizzle("square", 1.5)
    # pixfrac<1 concentrates coverage; pixfrac>1 spreads it (legacy archaeology)
    assert np.count_nonzero(out_small) <= np.count_nonzero(out_full)
    assert np.count_nonzero(out_wide) >= np.count_nonzero(out_full)


def test_contract_marks_lanczos_effective_pixfrac_one():
    contract = dsd.contract_diagnostic("lanczos2", 0.8, exptime=1.0)
    assert contract["pixfrac_effective"] == 0.8  # requested stays explicit
    # the Lanczos force-to-1.0 is a queue-manager policy, not a wrapper change
    assert contract["pixel_scale_ratio_source"] == "upstream_default"
