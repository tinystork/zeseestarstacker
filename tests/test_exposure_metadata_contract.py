"""ZSSS-OTPUX-A-01 — exposure metadata truthfulness focused regression tests.

Ratified exposure contract (see ``docs/output_truthfulness_preview_audit.md``
§5.1 / §7-A):

* canonical per-frame parse: ``EXPTIME`` then ``EXPOSURE``, valid iff numeric
  finite ``> 0``, else *unknown*;
* ``NIMAGES`` counts *accepted* contributors (classic batch past the combine
  gates; Drizzle ``add`` returned ``True``) — never "aligned"/"attempted";
* ``TOTEXP`` is the exact nominal sum of *known* accepted exposures, else
  omitted with ``NEXPUNK`` = exact accepted-unknown count (never a fabricated
  ``0.0``/``1.0``);
* final ``EXPTIME`` is set only when every accepted exposure is known and
  uniform, and any inherited ``EXPTIME``/``EXPOSURE`` is deleted otherwise;
* the scientific pixel/accumulator output is bit-identical (bookkeeping only).

These tests exercise the real ``queue_manager`` functions/methods on bare
``SeestarQueuedStacker`` objects (no heavy ``__init__``), mirroring the
established ``tests/test_save_final_stack.py`` and ``tests/test_resume.py``
harness style.
"""

from __future__ import annotations

import importlib.util
import json
import queue
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import the real queue_manager directly.  It already degrades gracefully
# without the Tk GUI (its `from seestar.gui.settings import SettingsManager,
# TILE_HEIGHT` is wrapped in a try/except that falls back to
# `SettingsManager = object`), so no synthetic `seestar.gui` stubs are needed.
# Installing synthetic `seestar.gui`/`seestar.gui.settings`/`seestar.gui
# .histogram_widget` modules here used to poison process-global `sys.modules`
# at collection time (empty `__path__`), breaking later `seestar.gui_qt`
# imports when this file was collected first.  No synthetic module is
# installed, so nothing needs restoring.
from seestar.queuep.queue_manager import (  # noqa: E402
    FINALIZATION_MODE_CLASSIC_SUMW,
    FINALIZATION_MODE_DRIZZLE,
    FINALIZATION_MODE_REPROJECT_COADD,
    SeestarQueuedStacker,
    _apply_exposure_metadata,
    _batch_exposure_provenance,
    _frame_exposure_seconds,
)

# Reuse the resume test harness (module-level helpers) without re-collecting
# its tests: load it under a private module name.
_resume_spec = importlib.util.spec_from_file_location(
    "_zsss_exposure_resume_helpers", str(ROOT / "tests" / "test_resume.py")
)
_resume_helpers = importlib.util.module_from_spec(_resume_spec)
_resume_spec.loader.exec_module(_resume_helpers)

make_resume_stack = _resume_helpers.make_resume_stack
build_session = _resume_helpers.build_session
bind_session = _resume_helpers.bind_session
write_valid_checkpoint = _resume_helpers.write_valid_checkpoint
close_mm = _resume_helpers.close_mm


# ---------------------------------------------------------------------------
# 1. Canonical per-frame exposure parse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "header, expected",
    [
        ({"EXPTIME": 10.0}, 10.0),
        ({"EXPTIME": 10.0, "EXPOSURE": 20.0}, 10.0),  # EXPTIME wins
        ({"EXPOSURE": 20.0}, 20.0),  # fallback
        ({"EXPTIME": "bad", "EXPOSURE": 20.0}, 20.0),  # invalid EXPTIME falls through
        ({"EXPTIME": "10.5"}, 10.5),  # numeric string
        ({"EXPTIME": float("nan")}, None),
        ({"EXPTIME": float("inf")}, None),
        ({"EXPTIME": 0.0}, None),  # non-positive
        ({"EXPTIME": -5.0}, None),
        ({"EXPTIME": 0.0, "EXPOSURE": 30.0}, 30.0),  # non-positive EXPTIME falls through
        ({"EXPTIME": float("nan"), "EXPOSURE": float("nan")}, None),
        ({}, None),
    ],
)
def test_canonical_parse_order_and_validity(header, expected):
    result = _frame_exposure_seconds(header)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_canonical_parse_none_header():
    assert _frame_exposure_seconds(None) is None


# ---------------------------------------------------------------------------
# 2. Composable batch provenance
# ---------------------------------------------------------------------------

def test_batch_provenance_uniform_mixed_unknown():
    headers = [
        {"EXPTIME": 2.0},
        {"EXPTIME": 3.0},
        {},  # unknown
        {"EXPTIME": 2.0},
        {"EXPTIME": -1.0},  # non-positive -> unknown
    ]
    known_sum, unknown_count, mn, mx = _batch_exposure_provenance(headers)
    assert known_sum == pytest.approx(7.0)
    assert unknown_count == 2
    assert mn == pytest.approx(2.0)
    assert mx == pytest.approx(3.0)


def test_batch_provenance_all_unknown():
    known_sum, unknown_count, mn, mx = _batch_exposure_provenance([{}, {"EXPTIME": 0.0}])
    assert known_sum == 0.0
    assert unknown_count == 2
    assert mn is None
    assert mx is None


# ---------------------------------------------------------------------------
# 3. Admitted-frame aggregate (exactly once after admission)
# ---------------------------------------------------------------------------

def _bare_exposure_obj():
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.total_exposure_seconds = 0.0
    o._exposure_unknown_count = 0
    o._exposure_min = None
    o._exposure_max = None
    return o


def test_admit_exposure_accumulates_and_bounds():
    o = _bare_exposure_obj()
    o._admit_exposure(2.0, 0, 2.0, 2.0)
    o._admit_exposure(3.0, 0, 3.0, 3.0)
    o._admit_exposure(0.0, 1)
    assert o.total_exposure_seconds == pytest.approx(5.0)
    assert o._exposure_unknown_count == 1
    assert o._exposure_min == pytest.approx(2.0)
    assert o._exposure_max == pytest.approx(3.0)


def test_classic_vs_drizzle_aggregate_parity():
    # Classic: one batch of three 2.0 s frames.
    classic = _bare_exposure_obj()
    classic._admit_exposure(6.0, 0, 2.0, 2.0)
    # Drizzle: three individually-admitted 2.0 s frames.
    drizzle = _bare_exposure_obj()
    for _ in range(3):
        drizzle._admit_exposure(2.0, 0, 2.0, 2.0)
    for a, b in (
        (classic.total_exposure_seconds, drizzle.total_exposure_seconds),
        (classic._exposure_unknown_count, drizzle._exposure_unknown_count),
        (classic._exposure_min, drizzle._exposure_min),
        (classic._exposure_max, drizzle._exposure_max),
    ):
        assert a == pytest.approx(b) if a is not None else b is None


def test_hierarchical_grouping_invariance():
    headers = [
        {"EXPTIME": 2.0},
        {"EXPTIME": 3.0},
        {},
        {"EXPTIME": 2.0},
    ]
    whole = _batch_exposure_provenance(headers)

    # Partition into two groups and fold each through _admit_exposure.
    part_a = _batch_exposure_provenance(headers[:2])
    part_b = _batch_exposure_provenance(headers[2:])
    o = _bare_exposure_obj()
    o._admit_exposure(*part_a)
    o._admit_exposure(*part_b)

    assert o.total_exposure_seconds == pytest.approx(whole[0])
    assert o._exposure_unknown_count == whole[1]
    assert o._exposure_min == pytest.approx(whole[2])
    assert o._exposure_max == pytest.approx(whole[3])


# ---------------------------------------------------------------------------
# 4. Final-header exposure semantics (_apply_exposure_metadata)
# ---------------------------------------------------------------------------

def _header_obj(unknown_count=0, known_sum=0.0, mn=None, mx=None, accepted=0):
    o = _bare_exposure_obj()
    o.total_exposure_seconds = known_sum
    o._exposure_unknown_count = unknown_count
    o._exposure_min = mn
    o._exposure_max = mx
    return o, accepted


def test_apply_uniform_sets_totexp_and_uniform_exptime():
    o, accepted = _header_obj(known_sum=7.5, mn=2.5, mx=2.5, accepted=3)
    hdr = fits.Header()
    hdr["EXPTIME"] = 1.0  # stale inherited value must be replaced
    _apply_exposure_metadata(o, hdr, accepted)
    assert hdr["TOTEXP"] == pytest.approx(7.5)
    assert hdr["EXPTIME"] == pytest.approx(2.5)
    assert "EXPOSURE" not in hdr
    assert "NEXPUNK" not in hdr


def test_apply_mixed_sums_totexp_and_removes_exptime():
    o, accepted = _header_obj(known_sum=3.0, mn=1.0, mx=2.0, accepted=2)
    hdr = fits.Header()
    hdr["EXPTIME"] = 1.0
    hdr["EXPOSURE"] = 1.0
    _apply_exposure_metadata(o, hdr, accepted)
    assert hdr["TOTEXP"] == pytest.approx(3.0)
    assert "EXPTIME" not in hdr
    assert "EXPOSURE" not in hdr
    assert "NEXPUNK" not in hdr


def test_apply_unknown_omits_totexp_and_writes_nexpunk():
    o, accepted = _header_obj(unknown_count=1, known_sum=4.0, mn=2.0, mx=2.0, accepted=3)
    hdr = fits.Header()
    hdr["TOTEXP"] = 99.0  # stale value must be deleted
    hdr["EXPTIME"] = 2.0
    hdr["EXPOSURE"] = 2.0
    _apply_exposure_metadata(o, hdr, accepted)
    assert "TOTEXP" not in hdr
    assert hdr["NEXPUNK"] == 1
    assert "EXPTIME" not in hdr
    assert "EXPOSURE" not in hdr


def test_apply_no_accepted_frames_removes_all_scalars():
    o, accepted = _header_obj(accepted=0)
    hdr = fits.Header()
    hdr["TOTEXP"] = 1.0
    hdr["NEXPUNK"] = 1
    hdr["EXPTIME"] = 1.0
    hdr["EXPOSURE"] = 1.0
    _apply_exposure_metadata(o, hdr, accepted)
    for key in ("TOTEXP", "NEXPUNK", "EXPTIME", "EXPOSURE"):
        assert key not in hdr


# ---------------------------------------------------------------------------
# 5. End-to-end final-save wiring (Drizzle uses _drizzle_frame_count)
# ---------------------------------------------------------------------------

class _Dummy:
    pass


def _make_drizzle_save_obj(tmp_path, save_as_float32=True):
    obj = _Dummy()
    obj.update_progress = lambda *a, **k: None
    obj._close_memmaps = lambda: None
    obj._wait_drizzle_processes = lambda: None
    obj.save_final_as_float32 = save_as_float32
    obj.preserve_linear_output = True
    obj.drizzle_wht_threshold = 0
    obj.output_folder = str(tmp_path)
    obj.output_filename = "out.fit"
    obj.current_stack_header = fits.Header()
    obj.drizzle_active_session = True
    obj.is_mosaic_run = False
    obj.drizzle_mode = "Final"
    obj.drizzle_output_wcs = None
    obj.drizzle_fillval = "0.0"
    obj.reproject_between_batches = False
    obj.cumulative_sum_memmap = None
    obj.cumulative_wht_memmap = None
    obj.stop_processing = False
    obj._validate_drizzle_science = types.MethodType(
        SeestarQueuedStacker._validate_drizzle_science, obj
    )
    # Exposure metadata truthfulness state.
    obj.total_exposure_seconds = 0.0
    obj._exposure_unknown_count = 0
    obj._exposure_min = None
    obj._exposure_max = None
    obj._drizzle_frame_count = 0
    obj.images_in_cumulative_stack = 0  # must NOT be used for Drizzle NIMAGES
    obj.aligned_files_count = 999  # must NOT feed NIMAGES
    return obj


def _add_drizzle_science(obj, shape=(2, 2)):
    from seestar.core.drizzle_core import DrizzleAccumulator

    obj.drizzle_accumulators = [DrizzleAccumulator(shape) for _ in range(3)]
    for i, acc in enumerate(obj.drizzle_accumulators):
        acc._out_img[:] = float(i + 1)
        acc._out_wht[:] = 1.0


def test_save_final_stack_drizzle_uniform(tmp_path):
    obj = _make_drizzle_save_obj(tmp_path)
    _add_drizzle_science(obj)
    obj._drizzle_frame_count = 3
    obj.total_exposure_seconds = 6.0
    obj._exposure_unknown_count = 0
    obj._exposure_min = 2.0
    obj._exposure_max = 2.0
    obj.current_stack_header["EXPTIME"] = 1.0  # stale

    SeestarQueuedStacker._save_final_stack(obj, output_filename_suffix="_drizzle_u")

    hdr = fits.getheader(obj.final_stacked_path)
    assert hdr["NIMAGES"] == 3
    assert hdr["TOTEXP"] == pytest.approx(6.0)
    assert hdr["EXPTIME"] == pytest.approx(2.0)
    assert "EXPOSURE" not in hdr
    assert "NEXPUNK" not in hdr


def test_save_final_stack_drizzle_mixed_deletes_exptime(tmp_path):
    obj = _make_drizzle_save_obj(tmp_path)
    _add_drizzle_science(obj)
    obj._drizzle_frame_count = 2
    obj.total_exposure_seconds = 3.0
    obj._exposure_unknown_count = 0
    obj._exposure_min = 1.0
    obj._exposure_max = 2.0
    obj.current_stack_header["EXPTIME"] = 1.0
    obj.current_stack_header["EXPOSURE"] = 1.0

    SeestarQueuedStacker._save_final_stack(obj, output_filename_suffix="_drizzle_m")

    hdr = fits.getheader(obj.final_stacked_path)
    assert hdr["NIMAGES"] == 2
    assert hdr["TOTEXP"] == pytest.approx(3.0)
    assert "EXPTIME" not in hdr
    assert "EXPOSURE" not in hdr


def test_save_final_stack_drizzle_unknown_writes_nexpunk(tmp_path):
    obj = _make_drizzle_save_obj(tmp_path)
    _add_drizzle_science(obj)
    obj._drizzle_frame_count = 3
    obj.total_exposure_seconds = 4.0  # two known (2.0 each) + one unknown
    obj._exposure_unknown_count = 1
    obj._exposure_min = 2.0
    obj._exposure_max = 2.0
    obj.current_stack_header["TOTEXP"] = 99.0
    obj.current_stack_header["EXPTIME"] = 2.0

    SeestarQueuedStacker._save_final_stack(obj, output_filename_suffix="_drizzle_nk")

    hdr = fits.getheader(obj.final_stacked_path)
    assert hdr["NIMAGES"] == 3
    assert "TOTEXP" not in hdr
    assert hdr["NEXPUNK"] == 1
    assert "EXPTIME" not in hdr


def test_save_final_stack_classic_uniform_exptime(tmp_path):
    obj = _Dummy()
    obj.update_progress = lambda *a, **k: None
    obj.logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    obj._close_memmaps = lambda: None
    obj._wait_drizzle_processes = lambda: None
    obj.save_final_as_float32 = True
    obj.preserve_linear_output = True
    obj.drizzle_wht_threshold = 0
    obj.output_folder = str(tmp_path)
    obj.output_filename = "out.fit"
    obj.current_stack_header = fits.Header()
    obj.drizzle_active_session = False
    obj.is_mosaic_run = False
    obj.drizzle_mode = "Final"
    obj.drizzle_output_wcs = None
    obj.drizzle_fillval = "0.0"
    obj.reproject_between_batches = False
    obj.reproject_coadd_final = False
    obj.stop_processing = False
    obj.cumulative_sum_memmap = np.ones((2, 2, 3), dtype=np.float32)
    obj.cumulative_wht_memmap = np.ones((2, 2), dtype=np.float32)
    obj.images_in_cumulative_stack = 3
    obj.total_exposure_seconds = 6.0
    obj._exposure_unknown_count = 0
    obj._exposure_min = 2.0
    obj._exposure_max = 2.0
    obj.current_stack_header["EXPTIME"] = 0.5  # stale

    SeestarQueuedStacker._save_final_stack(obj, output_filename_suffix="_classic_u")

    hdr = fits.getheader(obj.final_stacked_path)
    assert hdr["NIMAGES"] == 3
    assert hdr["TOTEXP"] == pytest.approx(6.0)
    assert hdr["EXPTIME"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 6. Resume persistence / restore (fail-closed on old/incomplete state)
# ---------------------------------------------------------------------------

def _manifest_path(out):
    return Path(out) / "memmap_accumulators" / "resume_manifest.json"


def test_resume_exposure_state_roundtrip(tmp_path):
    shape = (2, 2, 3)
    session = build_session(tmp_path, n_sources=2)
    ledger = session["sources"]
    out = tmp_path / "out"
    out.mkdir()
    write_valid_checkpoint(
        out,
        shape,
        count=2,
        ledger=ledger,
        session=session,
        images_in=2,
        total_exposure=4.0,
        header=fits.Header(),
    )
    # Mutate the manifest to carry non-default exposure aggregate state.
    mp = _manifest_path(out)
    m = json.loads(mp.read_text(encoding="utf-8"))
    m["exposure_unknown_count"] = 1
    m["exposure_min"] = 2.0
    m["exposure_max"] = 2.0
    m["total_exposure_seconds"] = 2.0
    mp.write_text(json.dumps(m), encoding="utf-8")

    s = make_resume_stack(out)
    bind_session(s, session)
    ok, reason = s._validate_and_open_resume(shape)
    assert ok is True, reason
    assert s.total_exposure_seconds == pytest.approx(2.0)
    assert s._exposure_unknown_count == 1
    assert s._exposure_min == pytest.approx(2.0)
    assert s._exposure_max == pytest.approx(2.0)
    close_mm(s.cumulative_sum_memmap)
    close_mm(s.cumulative_wht_memmap)


def test_resume_old_manifest_missing_exposure_fields_fails_closed(tmp_path):
    shape = (2, 2, 3)
    session = build_session(tmp_path, n_sources=2)
    ledger = session["sources"]
    out = tmp_path / "out"
    out.mkdir()
    write_valid_checkpoint(
        out, shape, count=2, ledger=ledger, session=session, images_in=2
    )
    # Simulate an old-format manifest that predates the exposure contract.
    mp = _manifest_path(out)
    m = json.loads(mp.read_text(encoding="utf-8"))
    m.pop("exposure_unknown_count", None)
    m.pop("exposure_min", None)
    m.pop("exposure_max", None)
    mp.write_text(json.dumps(m), encoding="utf-8")

    s = make_resume_stack(out)
    bind_session(s, session)
    ok, reason, _ = s._validate_resume_headless()
    assert ok is False
    assert "exposure truthfulness" in reason


def test_resume_continuation_no_double_count(tmp_path):
    """Checkpoint two 2.0 s batches, reopen, add a third: no double count."""
    shape = (2, 2, 3)
    session = build_session(tmp_path, n_sources=3)
    # Plan decomposition [1, 1, 1] so each single-source batch is a boundary.
    session["plan"] = {
        "sources": session["sources"],
        "decomposition": [1, 1, 1],
    }
    src_files = [s["path"] for s in session["sources"]]
    out = tmp_path / "out"
    out.mkdir()

    def _do_batch(stack, src_file):
        stack.stacked_batches_count += 1
        stack._current_batch_paths = [src_file]
        hdr = fits.Header()
        hdr["NIMAGES"] = 1
        hdr["TOTEXP"] = 2.0
        hdr["NEXPUNK"] = 0
        hdr["EXP_MIN"] = 2.0
        hdr["EXP_MAX"] = 2.0
        stack._combine_batch_result(
            np.ones(shape, dtype=np.float32),
            hdr,
            np.ones((shape[0], shape[1]), dtype=np.float32),
        )

    part = make_resume_stack(out)
    bind_session(part, session)
    assert part._initialize_classic_sumw_accumulators(shape) is True
    _do_batch(part, src_files[0])
    _do_batch(part, src_files[1])
    close_mm(part.cumulative_sum_memmap)
    close_mm(part.cumulative_wht_memmap)

    reopened = make_resume_stack(out)
    bind_session(reopened, session)
    ok, reason = reopened._validate_and_open_resume(shape)
    assert ok is True, reason
    assert reopened.images_in_cumulative_stack == 2
    assert reopened.total_exposure_seconds == pytest.approx(4.0)
    assert reopened._exposure_unknown_count == 0
    reopened.stacked_batches_count = reopened._resume_pending_count
    _do_batch(reopened, src_files[2])

    assert reopened.images_in_cumulative_stack == 3
    assert reopened.total_exposure_seconds == pytest.approx(6.0)
    assert reopened._exposure_unknown_count == 0
    assert reopened._exposure_min == pytest.approx(2.0)
    assert reopened._exposure_max == pytest.approx(2.0)
    close_mm(reopened.cumulative_sum_memmap)
    close_mm(reopened.cumulative_wht_memmap)


# ---------------------------------------------------------------------------
# 7. Pixel / accumulator witness: bookkeeping never mutates science
# ---------------------------------------------------------------------------

def test_pixel_witness_bookkeeping_does_not_mutate_accumulators():
    from seestar.core.drizzle_core import DrizzleAccumulator

    obj = _bare_exposure_obj()
    obj.drizzle_accumulators = [DrizzleAccumulator((4, 4)) for _ in range(3)]
    rng = np.random.default_rng(0)
    for acc in obj.drizzle_accumulators:
        acc._out_img[:] = rng.random((4, 4)).astype(np.float32)
        acc._out_wht[:] = rng.random((4, 4)).astype(np.float32)

    before_img = [acc._out_img.copy() for acc in obj.drizzle_accumulators]
    before_wht = [acc._out_wht.copy() for acc in obj.drizzle_accumulators]

    # The exact bookkeeping performed after a successful Drizzle admission.
    obj._admit_exposure(2.5, 0, 2.5, 2.5)
    obj._admit_exposure(0.0, 1)

    for acc, bi, bw in zip(
        obj.drizzle_accumulators, before_img, before_wht
    ):
        assert np.array_equal(acc._out_img, bi)
        assert np.array_equal(acc._out_wht, bw)
    assert obj.total_exposure_seconds == pytest.approx(2.5)
    assert obj._exposure_unknown_count == 1


# ---------------------------------------------------------------------------
# 8. Precision witness: fractional exposures survive production batch/header/
#    aggregation wiring; batch grouping does not change the nominal sum.
# ---------------------------------------------------------------------------

def _classic_wiring_stack(shape=(2, 2)):
    """Bare classic SUM/W stacker for the real ``_stack_batch`` +
    ``_combine_batch_result`` production wiring (no helper-only grouping)."""
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o.stacking_mode = "mean"
    o.normalize_method = "none"
    o.weighting_method = "none"
    o.use_quality_weighting = False
    o.weight_by_snr = False
    o.weight_by_stars = False
    o.snr_exponent = 1.0
    o.stars_exponent = 0.5
    o.min_weight = 0.0
    o.apply_batch_feathering = False
    o.reproject_between_batches = False
    o.reproject_coadd_final = False
    o.drizzle_active_session = False
    o.is_mosaic_run = False
    o.stack_kappa_low = 3.0
    o.stack_kappa_high = 3.0
    o.winsor_limits = (0.05, 0.05)
    o.stack_reject_algo = "none"
    o.max_hq_mem = 1_000_000_000
    o.batch_size = 10
    o.settings = None
    o.reference_header_for_wcs = None
    o.reference_wcs_object = None
    o.interbatch_norm_active = False
    o.max_stack_workers = 1
    o._current_batch_paths = []
    o._quality_reference_scale = 1.0
    o._checkpointing_enabled = False
    # ``_combine_batch_result`` accumulation state.
    o.memmap_shape = (shape[0], shape[1], 3)
    o.memmap_dtype_sum = np.float32
    o.memmap_dtype_wht = np.float32
    o.cumulative_sum_memmap = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    o.cumulative_wht_memmap = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    o.stacked_batches_count = 0
    o.images_in_cumulative_stack = 0
    o.total_exposure_seconds = 0.0
    o._exposure_unknown_count = 0
    o._exposure_min = None
    o._exposure_max = None
    o.failed_stack_count = 0
    o.current_stack_header = None
    o.logger = types.SimpleNamespace(warning=lambda *a, **k: None)
    o.correct_hot_pixels = False
    return o


def _exposure_item(exptime=None, shape=(2, 2)):
    """One batch item ``(img, header, scores, wcs, mask)``."""
    hdr = fits.Header()
    if exptime is not None:
        hdr["EXPTIME"] = exptime
    img = np.full(shape, 1.0, dtype=np.float32)
    mask = np.ones(shape, dtype=bool)
    return (img, hdr, {"snr": 1.0, "stars": 0.0}, None, mask)


def test_fractional_exposure_batch_grouping_invariance():
    # Three 0.0049 s frames: rounding at the batch level (round(x,2)=0.00)
    # used to collapse three singleton batches to a 0.0 aggregate instead of
    # 0.0147.  The contract requires the exact nominal sum internally.
    a = _exposure_item(0.0049)
    b = _exposure_item(0.0049)
    c = _exposure_item(0.0049)

    # Grouping 1: three singleton batches (each 0.0049 s).
    s1 = _classic_wiring_stack()
    for item in (a, b, c):
        V, hdr, W = s1._stack_batch([item], 1, 1)
        # The batch header must carry the full-precision nominal sum.
        assert hdr["EXP_SUM"] == pytest.approx(0.0049)
        s1._combine_batch_result(V, hdr, W)

    # Grouping 2: one three-frame batch.
    s2 = _classic_wiring_stack()
    V2, hdr2, W2 = s2._stack_batch([a, b, c], 1, 1)
    assert hdr2["EXP_SUM"] == pytest.approx(0.0147)
    s2._combine_batch_result(V2, hdr2, W2)

    # Both groupings must aggregate to the exact nominal sum 0.0147 (never 0.0).
    assert s1.total_exposure_seconds == pytest.approx(0.0147)
    assert s2.total_exposure_seconds == pytest.approx(0.0147)
    assert s1.total_exposure_seconds == pytest.approx(s2.total_exposure_seconds)

    # Rounding happens only on final FITS output: round(0.0147, 2) == 0.01.
    final1 = fits.Header()
    _apply_exposure_metadata(s1, final1, 3)
    assert final1["TOTEXP"] == pytest.approx(0.01)
    assert "NEXPUNK" not in final1


# ---------------------------------------------------------------------------
# 9. Classic reproject/coadd finalization (FINALIZATION_MODE_REPROJECT_COADD)
#    applies truthful TOTEXP/NEXPUNK/EXPTIME and deletes stale cards.
# ---------------------------------------------------------------------------

def _make_reproject_save_obj(tmp_path):
    obj = _Dummy()
    obj.update_progress = lambda *a, **k: None
    obj.logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    obj._close_memmaps = lambda: None
    obj._wait_drizzle_processes = lambda: None
    obj.save_final_as_float32 = True
    obj.preserve_linear_output = True
    obj.drizzle_wht_threshold = 0
    obj.output_folder = str(tmp_path)
    obj.output_filename = "out.fit"
    obj.current_stack_header = fits.Header()
    obj.drizzle_active_session = False
    obj.is_mosaic_run = False
    obj.drizzle_mode = "Final"
    obj.drizzle_output_wcs = None
    obj.drizzle_fillval = "0.0"
    obj.reproject_between_batches = False
    obj.reproject_coadd_final = True  # -> FINALIZATION_MODE_REPROJECT_COADD
    obj.finalization_mode = FINALIZATION_MODE_REPROJECT_COADD
    obj.stop_processing = False
    obj.cumulative_sum_memmap = None
    obj.cumulative_wht_memmap = None
    obj._validate_drizzle_science = types.MethodType(
        SeestarQueuedStacker._validate_drizzle_science, obj
    )
    obj.total_exposure_seconds = 0.0
    obj._exposure_unknown_count = 0
    obj._exposure_min = None
    obj._exposure_max = None
    obj.images_in_cumulative_stack = 0
    obj.aligned_files_count = 0
    return obj


def _run_reproject_save(obj, tmp_path, suffix):
    data = np.ones((2, 2, 3), dtype=np.float32)
    wht = np.ones((2, 2), dtype=np.float32)
    SeestarQueuedStacker._save_final_stack(
        obj,
        output_filename_suffix=suffix,
        drizzle_final_sci_data=data,
        drizzle_final_wht_data=wht,
        preserve_linear_output=True,
    )
    return fits.getheader(obj.final_stacked_path)


def test_reproject_coadd_uniform_sets_totexp_and_exptime(tmp_path):
    obj = _make_reproject_save_obj(tmp_path)
    obj.images_in_cumulative_stack = 3
    obj.total_exposure_seconds = 6.0
    obj._exposure_min = 2.0
    obj._exposure_max = 2.0
    obj.current_stack_header["EXPTIME"] = 1.0  # stale inherited
    hdr = _run_reproject_save(obj, tmp_path, "_rc_u")
    assert hdr["NIMAGES"] == 3
    assert hdr["TOTEXP"] == pytest.approx(6.0)
    assert hdr["EXPTIME"] == pytest.approx(2.0)
    assert "NEXPUNK" not in hdr


def test_reproject_coadd_mixed_sums_totexp_and_deletes_exptime(tmp_path):
    obj = _make_reproject_save_obj(tmp_path)
    obj.images_in_cumulative_stack = 2
    obj.total_exposure_seconds = 3.0
    obj._exposure_min = 1.0
    obj._exposure_max = 2.0
    obj.current_stack_header["EXPTIME"] = 1.0
    obj.current_stack_header["EXPOSURE"] = 1.0
    hdr = _run_reproject_save(obj, tmp_path, "_rc_m")
    assert hdr["NIMAGES"] == 2
    assert hdr["TOTEXP"] == pytest.approx(3.0)
    assert "EXPTIME" not in hdr
    assert "EXPOSURE" not in hdr


def test_reproject_coadd_unknown_omits_totexp_and_writes_nexpunk(tmp_path):
    obj = _make_reproject_save_obj(tmp_path)
    obj.images_in_cumulative_stack = 3
    obj.total_exposure_seconds = 4.0  # two known 2.0 + one unknown
    obj._exposure_unknown_count = 1
    obj._exposure_min = 2.0
    obj._exposure_max = 2.0
    obj.current_stack_header["TOTEXP"] = 99.0
    obj.current_stack_header["EXPTIME"] = 2.0
    hdr = _run_reproject_save(obj, tmp_path, "_rc_nk")
    assert hdr["NIMAGES"] == 3
    assert "TOTEXP" not in hdr
    assert hdr["NEXPUNK"] == 1
    assert "EXPTIME" not in hdr


# ---------------------------------------------------------------------------
# 10. Single-classic-batch finalizer seeds truthful aggregate from the batch
#     header and deletes the stale inherited per-frame EXPTIME.
# ---------------------------------------------------------------------------

def _make_classic_batch_file(tmp_path, header):
    obj = SeestarQueuedStacker()
    obj.output_folder = str(tmp_path)
    obj.update_progress = lambda *a, **k: None
    obj.reproject_coadd_final = True
    obj.solve_batches = False
    obj.reference_header_for_wcs = fits.Header()
    obj.reference_header_for_wcs["RA"] = 10.0
    obj.reference_header_for_wcs["DEC"] = 20.0
    data = np.ones((4, 4, 3), dtype=np.float32)
    wht = np.ones((4, 4), dtype=np.float32)
    sci, wht_paths = obj._save_and_solve_classic_batch(data, wht, header, 1)
    return obj, sci, wht_paths


def _finalize_single_batch_header(obj, sci, wht_paths):
    obj.finalization_mode = FINALIZATION_MODE_REPROJECT_COADD
    obj.output_filename = "final.fit"
    obj.preserve_linear_output = True
    obj.save_final_as_float32 = True
    obj._finalize_single_classic_batch((sci, wht_paths))
    return fits.getheader(obj.final_stacked_path)


def test_single_classic_batch_uniform_replaces_stale_exptime(tmp_path):
    hdr = fits.Header()
    hdr["EXPTIME"] = 99.0  # stale per-frame value inherited from the source
    hdr["NIMAGES"] = 3
    hdr["TOTEXP"] = 1.5
    hdr["NEXPUNK"] = 0
    hdr["EXP_MIN"] = 0.5
    hdr["EXP_MAX"] = 0.5
    hdr["EXP_SUM"] = 1.5
    obj, sci, wht_paths = _make_classic_batch_file(tmp_path, hdr)
    final = _finalize_single_batch_header(obj, sci, wht_paths)
    assert final["NIMAGES"] == 3
    assert final["TOTEXP"] == pytest.approx(1.5)
    assert final["EXPTIME"] == pytest.approx(0.5)
    assert "NEXPUNK" not in final


def test_single_classic_batch_mixed_deletes_exptime(tmp_path):
    hdr = fits.Header()
    hdr["EXPTIME"] = 99.0
    hdr["NIMAGES"] = 3
    hdr["TOTEXP"] = 2.0
    hdr["NEXPUNK"] = 0
    hdr["EXP_MIN"] = 0.5
    hdr["EXP_MAX"] = 1.0
    hdr["EXP_SUM"] = 2.0
    obj, sci, wht_paths = _make_classic_batch_file(tmp_path, hdr)
    final = _finalize_single_batch_header(obj, sci, wht_paths)
    assert final["NIMAGES"] == 3
    assert final["TOTEXP"] == pytest.approx(2.0)
    assert "EXPTIME" not in final
    assert "NEXPUNK" not in final


def test_single_classic_batch_unknown_writes_nexpunk(tmp_path):
    hdr = fits.Header()
    hdr["EXPTIME"] = 99.0
    hdr["NIMAGES"] = 3
    hdr["TOTEXP"] = 1.0
    hdr["NEXPUNK"] = 1
    hdr["EXP_MIN"] = 0.5
    hdr["EXP_MAX"] = 0.5
    hdr["EXP_SUM"] = 1.0
    obj, sci, wht_paths = _make_classic_batch_file(tmp_path, hdr)
    final = _finalize_single_batch_header(obj, sci, wht_paths)
    assert final["NIMAGES"] == 3
    assert "TOTEXP" not in final
    assert final["NEXPUNK"] == 1
    assert "EXPTIME" not in final


# ---------------------------------------------------------------------------
# 11. Drizzle: accepted count/exposure committed before fallible side effects;
#     a failed add increments neither.
# ---------------------------------------------------------------------------

def _worker_wcs(shape=(2, 2)):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = np.array([-0.01, 0.01])
    w.wcs.crval = [0, 0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (shape[1], shape[0])
    return w


def _make_drizzle_worker(tmp_path, exptime=None, add_result=True, move_raises=False):
    obj = SeestarQueuedStacker()
    obj.perform_cleanup = False
    obj.stop_processing = False
    obj.current_folder = str(tmp_path)
    obj.output_folder = str(tmp_path)
    obj.queue = queue.Queue()
    src = tmp_path / "Light_001.fit"
    fits.writeto(src, np.zeros((2, 2, 3), dtype=np.float32), overwrite=True)
    obj.queue.put(str(src))
    obj.additional_folders = []
    obj.files_in_queue = 1
    obj.batch_size = 1
    obj.drizzle_active_session = True
    obj.drizzle_mode = "Final"
    obj.stacked_batches_count = 0
    obj.total_batches_estimated = 1
    obj.mosaic_settings_dict = {}
    obj.local_solver_preference = "none"
    obj.astap_search_radius = 1.0
    obj.astap_downsample = 1
    obj.astap_sensitivity = 100
    obj.reference_pixel_scale_arcsec = 1.0
    obj.astap_path = ""
    obj.astap_data_dir = ""
    obj.local_ansvr_path = ""
    obj.api_key = None
    obj.ansvr_timeout_sec = 5
    obj.astap_timeout_sec = 5
    obj.astrometry_net_timeout_sec = 5
    obj.drizzle_fillval = "0.0"
    obj.update_progress = lambda *a, **k: None
    obj.move_stacked = False

    ref_path = tmp_path / "temp_processing" / "reference_image.fit"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(ref_path, np.zeros((2, 2), dtype=np.float32), overwrite=True)

    class _Aligner:
        def __init__(self):
            self.correct_hot_pixels = True
            self.hot_pixel_threshold = 3.0
            self.neighborhood_size = 5
            self.bayer_pattern = "GRBG"

        def _get_reference_image(self, folder, files, out_folder):
            return np.zeros((2, 2, 3), dtype=np.float32), fits.Header()

    obj.aligner = _Aligner()

    class _Solver:
        def solve(self, *a, **k):
            return _worker_wcs()

    obj.astrometry_solver = _Solver()
    obj._create_drizzle_output_wcs = lambda ref_wcs, shape, scale: (
        _worker_wcs(shape),
        shape,
    )

    dummy_data = np.zeros((2, 2, 3), dtype=np.float32)
    tf = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, -0.25]], dtype=np.float64)
    hdr = fits.Header()
    if exptime is not None:
        hdr["EXPTIME"] = exptime
    obj._process_file = lambda *a, **k: (
        dummy_data,
        hdr,
        None,
        None,
        tf,
        np.ones((2, 2), dtype=np.float32),
    )

    def fake_add_frame(original_data, header, tf_val, weight_map, native_wcs=None):
        obj.stop_processing = True
        return add_result

    obj._add_frame_to_drizzle_accumulators = fake_add_frame
    if move_raises:
        def _boom(*a, **k):
            raise RuntimeError("boom: move failed after admission")

        obj._move_to_stacked = _boom
    else:
        obj._move_to_stacked = lambda *a, **k: None
    obj._save_partial_stack = lambda *a, **k: None
    obj._update_batch_count_file = lambda *a, **k: None
    obj._send_eta_update = lambda *a, **k: None
    obj._save_final_stack = lambda *a, **k: None
    obj._process_incremental_drizzle_batch = (
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("incremental called"))
    )
    obj._start_drizzle_process = (
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("incremental start called"))
    )
    obj._process_and_save_drizzle_batch = (
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("final batch called"))
    )
    obj._process_completed_batch = (
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("classic called"))
    )
    obj.cleanup_temp_reference = lambda: None
    obj._cleanup_drizzle_temp_files = lambda: None
    obj._cleanup_drizzle_batch_outputs = lambda: None
    obj._cleanup_mosaic_panel_stacks_temp = lambda: None
    obj._wait_drizzle_processes = lambda: None
    return obj


def test_drizzle_post_admission_side_effect_failure_keeps_lockstep(tmp_path):
    obj = _make_drizzle_worker(tmp_path, exptime=2.5, add_result=True, move_raises=True)
    SeestarQueuedStacker._worker(obj)
    # The move side effect raised AFTER admission, but the accepted count and
    # exposure were committed first and stay in lockstep with the admitted frame.
    assert obj._drizzle_frame_count == 1
    assert obj.total_exposure_seconds == pytest.approx(2.5)
    assert obj._exposure_unknown_count == 0
    assert obj._exposure_min == pytest.approx(2.5)
    assert obj._exposure_max == pytest.approx(2.5)


def test_drizzle_failed_add_increments_neither(tmp_path):
    obj = _make_drizzle_worker(tmp_path, exptime=2.5, add_result=False)
    SeestarQueuedStacker._worker(obj)
    assert obj._drizzle_frame_count == 0
    assert obj.total_exposure_seconds == 0.0
    assert obj._exposure_unknown_count == 0
    assert obj._exposure_min is None
    assert obj._exposure_max is None
    assert obj.failed_stack_count == 1


# ---------------------------------------------------------------------------
# 12. Resume: malformed exposure provenance fails closed (no false final TOTEXP).
# ---------------------------------------------------------------------------

def _write_exposure_manifest(
    tmp_path,
    session,
    images_in=2,
    unknown=0,
    mn=2.0,
    mx=2.0,
    totexp=4.0,
    shape=(2, 2, 3),
):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    write_valid_checkpoint(
        out,
        shape,
        count=2,
        ledger=session["sources"],
        session=session,
        images_in=images_in,
        total_exposure=totexp,
        header=fits.Header(),
    )
    mp = _manifest_path(out)
    m = json.loads(mp.read_text(encoding="utf-8"))
    m["exposure_unknown_count"] = unknown
    m["exposure_min"] = mn
    m["exposure_max"] = mx
    m["total_exposure_seconds"] = totexp
    mp.write_text(json.dumps(m), encoding="utf-8")
    s = make_resume_stack(out)
    bind_session(s, session)
    return s


def test_resume_unknown_count_exceeds_accepted_fails_closed(tmp_path):
    session = build_session(tmp_path, n_sources=2)
    s = _write_exposure_manifest(tmp_path, session, images_in=2, unknown=3)
    ok, reason, _ = s._validate_resume_headless()
    assert ok is False
    assert "exposure_unknown_count" in reason


def test_resume_partial_bounds_fails_closed(tmp_path):
    session = build_session(tmp_path, n_sources=2)
    s = _write_exposure_manifest(tmp_path, session, images_in=2, mn=2.0, mx=None)
    ok, reason, _ = s._validate_resume_headless()
    assert ok is False
    assert "bounds" in reason


def test_resume_min_gt_max_fails_closed(tmp_path):
    session = build_session(tmp_path, n_sources=2)
    s = _write_exposure_manifest(tmp_path, session, images_in=2, mn=3.0, mx=2.0)
    ok, reason, _ = s._validate_resume_headless()
    assert ok is False
    assert "exposure_min > exposure_max" in reason


def test_resume_known_sum_out_of_bounds_fails_closed(tmp_path):
    session = build_session(tmp_path, n_sources=2)
    s = _write_exposure_manifest(
        tmp_path, session, images_in=2, mn=2.0, mx=2.0, totexp=100.0
    )
    ok, reason, _ = s._validate_resume_headless()
    assert ok is False
    assert "inconsistent with exposure bounds" in reason


def test_resume_all_unknown_nonzero_sum_fails_closed(tmp_path):
    session = build_session(tmp_path, n_sources=2)
    s = _write_exposure_manifest(
        tmp_path, session, images_in=2, unknown=2, mn=2.0, mx=2.0, totexp=5.0
    )
    ok, reason, _ = s._validate_resume_headless()
    assert ok is False
    assert "all-unknown" in reason
