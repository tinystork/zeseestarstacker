import importlib.util
import json
import os
import shutil
import sys
import threading
import types
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import pytest
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from seestar.queuep.queue_manager import (  # noqa: E402
    SeestarQueuedStacker,
    _BATCH_BREAK_TOKEN,
    _ResumeCheckpointError,
)
import seestar.run_contract as rc  # noqa: E402


def close_mm(mm):
    try:
        mm.flush()
        if hasattr(mm, "_mmap") and mm._mmap is not None:
            mm._mmap.close()
    except Exception:
        pass


def _stat_identity(path):
    """Mirror of the production source-identity evidence (size + mtime_ns)."""
    p = os.path.abspath(str(path))
    st = os.stat(p)
    return {
        "path": os.path.normcase(p),
        "name": os.path.basename(p),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _write_file(path, size):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00" * size)
    return str(p)


def _make_resume_stack(out_dir, **overrides):
    """Build a bare SeestarQueuedStacker (no __init__) with a fixed,
    deterministic scientific configuration for resume-contract tests."""
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o.logger = types.SimpleNamespace(
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    o.output_folder = str(out_dir)
    o.memmap_dtype_sum = np.float32
    o.memmap_dtype_wht = np.float32
    o.memmap_shape = None
    o.cumulative_sum_memmap = None
    o.cumulative_wht_memmap = None
    o.sum_memmap_path = None
    o.wht_memmap_path = None
    o.batch_count_path = None
    o.stacked_subdir_name = "stacked"
    o.aligner = types.SimpleNamespace(reference_image_path=None)
    o.queue = Queue()

    # mode flags
    o.is_mosaic_run = False
    o.drizzle_active_session = False
    o.reproject_between_batches = False
    o.reproject_coadd_final = False

    # scientific fingerprint attributes (see _RESUME_FINGERPRINT_ATTRS)
    o.stacking_mode = "kappa-sigma"
    o.kappa = 2.5
    o.stack_kappa_low = 2.5
    o.stack_kappa_high = 2.5
    o.winsor_limits = (0.05, 0.05)
    o.normalize_method = "none"
    o.weighting_method = "none"
    o.use_quality_weighting = False
    o.weight_by_snr = True
    o.weight_by_stars = True
    o.snr_exponent = 1.0
    o.stars_exponent = 0.5
    o.min_weight = 0.1
    o.correct_hot_pixels = True
    o.hot_pixel_threshold = 3.0
    o.neighborhood_size = 5
    o.bayer_pattern = "GRBG"
    o.batch_size = 10
    o.chunk_size = None
    o.apply_batch_feathering = True
    o.apply_feathering = False
    o.feather_blur_px = 256
    o.apply_master_tile_crop = False
    o.master_tile_crop_percent_decimal = 0.18
    o.apply_low_wht_mask = False
    o.low_wht_percentile = 5
    o.low_wht_soften_px = 128

    # checkpoint state
    o._resume_requested = False
    o._resume_active = False
    o._resume_completed_sources = []
    o._resume_pending_count = 0
    o._checkpointing_enabled = False

    # HSI-2B C1 session binding (deterministic default per out_dir)
    o.current_folder = None
    o.additional_folders = []
    o._resume_input_roots = [
        os.path.normcase(os.path.abspath(str(Path(out_dir) / "input")))
    ]
    o._resume_reference_identity = {
        "path": os.path.normcase(
            os.path.abspath(str(Path(out_dir) / "input" / "reference.fits"))
        ),
        "name": "reference.fits",
        "size": 4096,
        "mtime_ns": 123456789,
    }
    o._resume_plan = None
    o._resume_images_in_cumulative_stack = 0
    o._resume_total_exposure_seconds = 0.0
    o._resume_cumulative_header = None

    # accumulation state used by _combine_batch_result
    o.stacked_batches_count = 0
    o._current_batch_paths = []
    o.images_in_cumulative_stack = 0
    o.total_exposure_seconds = 0.0
    o.current_stack_header = None
    o.failed_stack_count = 0
    o.stop_processing = False
    o._last_classic_batch_solved = True

    for k, v in overrides.items():
        setattr(o, k, v)
    return o


def make_resume_stack(out_dir, **overrides):
    return _make_resume_stack(out_dir, **overrides)


def make_init_stack(out_dir, **overrides):
    """Bare stacker with enough state for ``initialize`` to run."""
    o = _make_resume_stack(out_dir, **overrides)
    o.finalization_mode = None
    o.enable_preview = False
    o.batch_size = 10
    o.aligned_temp_dir = None
    o.perform_cleanup = True
    o.unaligned_folder = None
    o.drizzle_temp_dir = None
    o.drizzle_batch_output_dir = None
    o.classic_batch_output_dir = None
    o.drizzle_mode = "Final"
    o.warned_unaligned_source_folders = set()
    o.queue = Queue()
    import threading

    o.folders_lock = threading.Lock()
    o.processed_files = set()
    o.additional_folders = []
    o.queue_prepared = False
    o.intermediate_drizzle_batch_files = []
    o.reference_wcs_object = None
    o.use_drizzle = False
    o.reference_shape = None
    o.fixed_output_wcs = None
    o.fixed_output_shape = None
    o.input_reference_shape_hw = None
    o.keep_input_size_for_reproject = False
    o._has_stack_plan = False
    return o


def build_session(tmp_path, n_sources=0, ref_size=4096):
    """Create an input root with a reference file + n sources; return a dict."""
    input_dir = tmp_path / "input"
    ref_path = _write_file(input_dir / "reference.fits", ref_size)
    ref_ident = _stat_identity(ref_path)
    roots = [os.path.normcase(os.path.abspath(str(input_dir)))]
    sources = []
    for i in range(n_sources):
        p = _write_file(input_dir / f"obs_{i:03d}.fits", 100 + i)
        sources.append(_stat_identity(p))
    plan = {
        "sources": list(sources),
        "decomposition": [1] * len(sources),
    }
    return {
        "input_dir": str(input_dir),
        "roots": roots,
        "reference": ref_ident,
        "ref_path": ref_path,
        "sources": sources,
        "plan": plan,
    }


def bind_session(stack, session):
    stack._resume_input_roots = list(session["roots"])
    stack._resume_reference_identity = dict(session["reference"])
    stack._resume_plan = dict(session["plan"])


def make_ledger(input_dir, n, start=0):
    """Create n real source files and return their identities (ordered)."""
    idents = []
    for i in range(n):
        p = _write_file(Path(input_dir) / f"obs_{(start + i):03d}.fits", 100 + start + i)
        idents.append(_stat_identity(p))
    return idents


def write_valid_checkpoint(
    out_dir,
    shape,
    count,
    ledger,
    sum_val=1.0,
    wht_val=2.0,
    session=None,
    images_in=0,
    total_exposure=0.0,
    header=None,
):
    """Create a valid versioned clean checkpoint (nonzero SUM/WHT + manifest)."""
    memdir = Path(out_dir) / "memmap_accumulators"
    memdir.mkdir(parents=True, exist_ok=True)
    sum_mm = np.lib.format.open_memmap(
        memdir / "cumulative_SUM.npy", mode="w+", dtype=np.float32, shape=shape
    )
    wht_mm = np.lib.format.open_memmap(
        memdir / "cumulative_WHT.npy", mode="w+", dtype=np.float32, shape=shape
    )
    sum_mm[:] = np.asarray(sum_val) if np.ndim(sum_val) else sum_val
    wht_mm[:] = np.asarray(wht_val) if np.ndim(wht_val) else wht_val
    sum_mm.flush()
    wht_mm.flush()
    close_mm(sum_mm)
    close_mm(wht_mm)

    stack = make_resume_stack(out_dir)
    if session is not None:
        bind_session(stack, session)
    stack.memmap_shape = tuple(shape)
    stack._resume_completed_sources = list(ledger)
    stack.stacked_batches_count = count
    stack.images_in_cumulative_stack = images_in
    stack.total_exposure_seconds = total_exposure
    stack.current_stack_header = header
    # Default plan for a "fully completed" checkpoint = the ledger itself.
    if stack._resume_plan is None:
        stack._resume_plan = {
            "sources": list(ledger),
            "decomposition": [len(ledger)] if ledger else [],
        }
    stack._write_resume_manifest(
        state="clean", completed_sources=list(ledger), stacked_batches_count=count
    )
    return stack


# ---------------------------------------------------------------------------
# 1. Legacy artifact set that old _can_resume accepted is now refused and left
#    unchanged.
# ---------------------------------------------------------------------------
def test_legacy_three_file_set_refused_and_unchanged(tmp_path):
    out = tmp_path
    memdir = out / "memmap_accumulators"
    memdir.mkdir()
    sum_mm = np.lib.format.open_memmap(
        memdir / "cumulative_SUM.npy", mode="w+", dtype=np.float32, shape=(2, 2, 3)
    )
    sum_mm[:] = 7.5
    wht_mm = np.lib.format.open_memmap(
        memdir / "cumulative_WHT.npy", mode="w+", dtype=np.float32, shape=(2, 2)
    )
    wht_mm[:] = 3.0
    sum_mm.flush()
    wht_mm.flush()
    close_mm(sum_mm)
    close_mm(wht_mm)
    (out / "batches_count.txt").write_text("2")

    sum_bytes = (memdir / "cumulative_SUM.npy").read_bytes()
    wht_bytes = (memdir / "cumulative_WHT.npy").read_bytes()

    s = make_resume_stack(out)
    assert s._can_resume(out) is True
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is False
    assert "manifest" in reason

    assert (memdir / "cumulative_SUM.npy").read_bytes() == sum_bytes
    assert (memdir / "cumulative_WHT.npy").read_bytes() == wht_bytes


# ---------------------------------------------------------------------------
# 2. Versioned clean checkpoint round-trip: nonzero HWC SUM/WHT + batch count
#    reopen byte/value-preserved.
# ---------------------------------------------------------------------------
def test_versioned_clean_checkpoint_roundtrip(tmp_path):
    out = tmp_path
    shape = (3, 4, 3)
    sum_expected = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 0.25
    wht_expected = np.full(shape, 2.0, dtype=np.float32)
    session = build_session(out, n_sources=4)
    ledger = session["sources"]

    write_valid_checkpoint(
        out, shape, count=4, ledger=ledger, session=session,
        sum_val=sum_expected, wht_val=wht_expected,
    )

    s = make_resume_stack(out)
    bind_session(s, session)
    ok, reason = s._validate_and_open_resume(shape)
    assert ok is True, reason
    assert s._resume_active is True
    assert s.memmap_shape == shape
    assert s._resume_pending_count == 4
    assert s._resume_completed_sources == ledger
    # value-preserved (not zeroed)
    assert np.array_equal(s.cumulative_sum_memmap, sum_expected)
    assert np.array_equal(s.cumulative_wht_memmap, wht_expected)
    close_mm(s.cumulative_sum_memmap)
    close_mm(s.cumulative_wht_memmap)


# ---------------------------------------------------------------------------
# 3. Scientific configuration mismatch refused (stacking method + rejection).
# ---------------------------------------------------------------------------
def test_scientific_config_mismatch_refused(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=2)
    write_valid_checkpoint(out, (2, 2, 3), count=2, ledger=session["sources"], session=session)

    s = make_resume_stack(out, stacking_mode="mean")
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is False
    assert "configuration mismatch" in reason

    s = make_resume_stack(out, kappa=4.0)
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is False
    assert "configuration mismatch" in reason

    s = make_resume_stack(out, snr_exponent=2.0)
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is False
    assert "configuration mismatch" in reason


# ---------------------------------------------------------------------------
# 4. Shape/dtype mismatch and 2-D WHT refused.
# ---------------------------------------------------------------------------
def test_shape_mismatch_refused(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=1)
    write_valid_checkpoint(out, (2, 2, 3), count=1, ledger=session["sources"], session=session)
    s = make_resume_stack(out)
    ok, reason = s._validate_and_open_resume((3, 3, 3))
    assert ok is False
    assert "shape" in reason


def test_2d_wht_refused(tmp_path):
    out = tmp_path
    memdir = out / "memmap_accumulators"
    memdir.mkdir(parents=True)
    shape = (2, 2, 3)
    sum_mm = np.lib.format.open_memmap(
        memdir / "cumulative_SUM.npy", mode="w+", dtype=np.float32, shape=shape
    )
    sum_mm[:] = 1.0
    wht_mm = np.lib.format.open_memmap(
        memdir / "cumulative_WHT.npy", mode="w+", dtype=np.float32, shape=(2, 2)
    )
    wht_mm[:] = 1.0
    sum_mm.flush()
    wht_mm.flush()
    close_mm(sum_mm)
    close_mm(wht_mm)
    session = build_session(out, n_sources=0)
    stack = make_resume_stack(out)
    bind_session(stack, session)
    stack.memmap_shape = shape
    stack._write_resume_manifest(state="clean", completed_sources=[], stacked_batches_count=0)

    s = make_resume_stack(out)
    bind_session(s, session)
    ok, reason = s._validate_and_open_resume(shape)
    assert ok is False
    assert "2-D" in reason or "3-D" in reason


def test_dtype_mismatch_refused(tmp_path):
    out = tmp_path
    memdir = out / "memmap_accumulators"
    memdir.mkdir(parents=True)
    shape = (2, 2, 3)
    sum_mm = np.lib.format.open_memmap(
        memdir / "cumulative_SUM.npy", mode="w+", dtype=np.float64, shape=shape
    )
    sum_mm[:] = 1.0
    wht_mm = np.lib.format.open_memmap(
        memdir / "cumulative_WHT.npy", mode="w+", dtype=np.float32, shape=shape
    )
    wht_mm[:] = 1.0
    sum_mm.flush()
    wht_mm.flush()
    close_mm(sum_mm)
    close_mm(wht_mm)
    session = build_session(out, n_sources=0)
    stack = make_resume_stack(out)
    bind_session(stack, session)
    stack.memmap_shape = shape
    stack._write_resume_manifest(state="clean", completed_sources=[], stacked_batches_count=0)

    s = make_resume_stack(out)
    bind_session(s, session)
    ok, reason = s._validate_and_open_resume(shape)
    assert ok is False
    assert "dtype" in reason


# ---------------------------------------------------------------------------
# 5. Corrupt / missing manifest and dirty state refused.
# ---------------------------------------------------------------------------
def test_missing_manifest_refused(tmp_path):
    out = tmp_path
    memdir = out / "memmap_accumulators"
    memdir.mkdir(parents=True)
    np.lib.format.open_memmap(
        memdir / "cumulative_SUM.npy", mode="w+", dtype=np.float32, shape=(2, 2, 3)
    )[:] = 1.0
    np.lib.format.open_memmap(
        memdir / "cumulative_WHT.npy", mode="w+", dtype=np.float32, shape=(2, 2, 3)
    )[:] = 1.0
    s = make_resume_stack(out)
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is False
    assert "manifest" in reason


def test_corrupt_manifest_refused(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=1)
    write_valid_checkpoint(out, (2, 2, 3), count=1, ledger=session["sources"], session=session)
    manifest_path = out / "memmap_accumulators" / "resume_manifest.json"
    manifest_path.write_text("{ not valid json", encoding="utf-8")
    s = make_resume_stack(out)
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is False
    assert "manifest" in reason


def test_dirty_state_refused(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=1)
    write_valid_checkpoint(out, (2, 2, 3), count=1, ledger=session["sources"], session=session)
    stack = make_resume_stack(out)
    bind_session(stack, session)
    stack.memmap_shape = (2, 2, 3)
    stack._write_resume_manifest(state="dirty", completed_sources=session["sources"], stacked_batches_count=1)
    s = make_resume_stack(out)
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is False
    assert "clean" in reason


# ---------------------------------------------------------------------------
# 6. Non-finite / negative WHT refused.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("wht_val", [float("nan"), float("inf"), -1.0])
def test_nonfinite_or_negative_wht_refused(tmp_path, wht_val):
    out = tmp_path
    shape = (2, 2, 3)
    memdir = out / "memmap_accumulators"
    memdir.mkdir(parents=True)
    sum_mm = np.lib.format.open_memmap(
        memdir / "cumulative_SUM.npy", mode="w+", dtype=np.float32, shape=shape
    )
    sum_mm[:] = 1.0
    wht_mm = np.lib.format.open_memmap(
        memdir / "cumulative_WHT.npy", mode="w+", dtype=np.float32, shape=shape
    )
    wht_mm[:] = 2.0
    wht_mm[0, 0, 0] = wht_val
    sum_mm.flush()
    wht_mm.flush()
    close_mm(sum_mm)
    close_mm(wht_mm)
    session = build_session(out, n_sources=0)
    stack = make_resume_stack(out)
    bind_session(stack, session)
    stack.memmap_shape = shape
    stack._write_resume_manifest(state="clean", completed_sources=[], stacked_batches_count=0)

    s = make_resume_stack(out)
    bind_session(s, session)
    ok, reason = s._validate_and_open_resume(shape)
    assert ok is False
    if np.isnan(wht_val) or np.isinf(wht_val):
        assert "finite" in reason
    else:
        assert "negative" in reason


# ---------------------------------------------------------------------------
# 7. Completed-source identity ledger: exact filtering, not count*batch_size.
# ---------------------------------------------------------------------------
def test_ledger_filters_exact_identities_not_count_times_batch(tmp_path):
    src_dir = tmp_path / "input"
    src_dir.mkdir()
    files = []
    for i in range(70):
        p = src_dir / f"obs_{i:03d}.fits"
        _write_file(p, i + 1)
        files.append(str(p))
    completed = files[:61]
    remaining = files[61:]

    stack = make_resume_stack(tmp_path)
    stack._resume_active = True
    stack._resume_completed_sources = [_stat_identity(p) for p in completed]
    stack.queue = Queue()
    for p in files:
        stack.queue.put(p)
    stack.files_in_queue = len(files)
    stack.use_batch_plan = False
    stack.batch_size = 10
    stack._has_stack_plan = False
    stack.total_batches_estimated = 0

    skipped = stack._filter_queue_by_resume_ledger()
    assert skipped == 61
    left = []
    while not stack.queue.empty():
        left.append(stack.queue.get_nowait())
    assert sorted(left) == sorted(remaining)


def test_ledger_rejects_same_path_different_identity(tmp_path):
    # A different observation masquerading under a completed path must raise
    # (hard refusal), never be kept as a "remaining file to stack".
    p = tmp_path / "obs.fits"
    _write_file(p, 4)
    stack = make_resume_stack(tmp_path)
    stack._resume_active = True
    stack._resume_completed_sources = [
        {
            "path": os.path.normcase(os.path.abspath(str(p))),
            "name": p.name,
            "size": 999,
            "mtime_ns": 12345,
        }
    ]
    stack.queue = Queue()
    stack.queue.put(str(p))
    stack.files_in_queue = 1
    stack.use_batch_plan = False
    stack.batch_size = 10
    stack._has_stack_plan = False
    stack.total_batches_estimated = 0

    with pytest.raises(_ResumeCheckpointError):
        stack._filter_queue_by_resume_ledger()


def test_ledger_matching_identity_is_skipped(tmp_path):
    p = tmp_path / "obs.fits"
    _write_file(p, 4)
    stack = make_resume_stack(tmp_path)
    ident = _stat_identity(p)
    stack._resume_active = True
    stack._resume_completed_sources = [ident]
    stack.queue = Queue()
    stack.queue.put(str(p))
    stack.files_in_queue = 1
    stack.use_batch_plan = False
    stack.batch_size = 10
    stack._has_stack_plan = False
    stack.total_batches_estimated = 0

    skipped = stack._filter_queue_by_resume_ledger()
    assert skipped == 1
    assert stack.queue.empty()


# ---------------------------------------------------------------------------
# 8. Controlled restart property: uninterrupted vs checkpoint + reopen +
#    remaining contributions agree.
# ---------------------------------------------------------------------------
def _do_batch(stack, idx, v, w, src_file, nimages=1, totexp=1.0):
    stack.stacked_batches_count += 1
    stack._current_batch_paths = [src_file]
    hdr = fits.Header()
    hdr["NIMAGES"] = nimages
    hdr["TOTEXP"] = totexp
    stack._combine_batch_result(v, hdr, w)


def test_controlled_restart_agrees_with_uninterrupted(tmp_path):
    shape = (4, 5, 3)
    rng = np.random.default_rng(0)
    batches = []
    for _ in range(3):
        v = (rng.random(shape) * 100.0).astype(np.float32)
        w = (rng.random(shape) * 3.0).astype(np.float32) + 0.5
        batches.append((v, w))

    src_files = []
    for i in range(3):
        f = tmp_path / f"src_{i}.fits"
        _write_file(f, i + 1)
        src_files.append(str(f))

    session = build_session(tmp_path, n_sources=0)
    session["sources"] = [_stat_identity(f) for f in src_files]
    # Each _do_batch commits exactly one source, so the persisted plan must
    # decompose the three observations as three single-source batches for the
    # completed ledger (2 sources after two batches) to land on a boundary.
    session["plan"] = {"sources": session["sources"], "decomposition": [1, 1, 1]}

    # Uninterrupted run: all three batches.
    full = make_resume_stack(tmp_path / "full")
    bind_session(full, session)
    assert full._initialize_classic_sumw_accumulators(shape) is True
    for i, (v, w) in enumerate(batches):
        _do_batch(full, i, v, w, src_files[i], totexp=2.0)
    full_sum = np.array(full.cumulative_sum_memmap, dtype=np.float32, copy=True)
    full_wht = np.array(full.cumulative_wht_memmap, dtype=np.float32, copy=True)
    full_images = full.images_in_cumulative_stack
    full_exposure = full.total_exposure_seconds

    # Checkpoint + reopen after two batches, then add the third.
    part = make_resume_stack(tmp_path / "part")
    bind_session(part, session)
    assert part._initialize_classic_sumw_accumulators(shape) is True
    for i in (0, 1):
        _do_batch(part, i, batches[i][0], batches[i][1], src_files[i], totexp=2.0)

    # Reopen the committed clean checkpoint (two batches).
    reopened = make_resume_stack(tmp_path / "part")
    bind_session(reopened, session)
    ok, reason = reopened._validate_and_open_resume(shape)
    assert ok is True, reason
    assert reopened._resume_pending_count == 2
    assert len(reopened._resume_completed_sources) == 2
    assert reopened.images_in_cumulative_stack == 2
    assert reopened.total_exposure_seconds == pytest.approx(4.0)
    # restored count, then add the remaining third batch.
    reopened.stacked_batches_count = reopened._resume_pending_count
    _do_batch(reopened, 2, batches[2][0], batches[2][1], src_files[2], totexp=2.0)

    reopened_sum = np.array(reopened.cumulative_sum_memmap, dtype=np.float32, copy=True)
    reopened_wht = np.array(reopened.cumulative_wht_memmap, dtype=np.float32, copy=True)

    assert np.allclose(reopened_sum, full_sum, rtol=1e-5, atol=1e-5)
    assert np.allclose(reopened_wht, full_wht, rtol=1e-5, atol=1e-5)
    assert reopened.images_in_cumulative_stack == full_images
    assert reopened.total_exposure_seconds == pytest.approx(full_exposure)

    with np.errstate(divide="ignore", invalid="ignore"):
        full_final = full_sum / full_wht
        reopened_final = reopened_sum / reopened_wht
    assert np.allclose(reopened_final, full_final, rtol=1e-4, atol=1e-4)

    close_mm(full.cumulative_sum_memmap)
    close_mm(full.cumulative_wht_memmap)
    close_mm(part.cumulative_sum_memmap)
    close_mm(part.cumulative_wht_memmap)
    close_mm(reopened.cumulative_sum_memmap)
    close_mm(reopened.cumulative_wht_memmap)


# ---------------------------------------------------------------------------
# 9. Dirty-before-mutation / clean-after-flush checkpoint behaviour.
# ---------------------------------------------------------------------------
def test_dirty_checkpoint_refused_and_commit_returns_clean(tmp_path):
    out = tmp_path
    stack = make_resume_stack(out)
    assert stack._initialize_classic_sumw_accumulators((2, 2, 3)) is True
    manifest_path = stack._resume_manifest_path()
    assert json.loads(manifest_path.read_text())["state"] == "clean"

    stack._checkpoint_mark_dirty()
    assert json.loads(manifest_path.read_text())["state"] == "dirty"

    other = make_resume_stack(out)
    ok, reason = other._validate_and_open_resume((2, 2, 3))
    assert ok is False
    assert "clean" in reason

    stack.stacked_batches_count = 1
    stack._current_batch_paths = []
    stack._checkpoint_commit_batch()
    assert json.loads(manifest_path.read_text())["state"] == "clean"

    close_mm(stack.cumulative_sum_memmap)
    close_mm(stack.cumulative_wht_memmap)


# ---------------------------------------------------------------------------
# 10. Unsupported drizzle / mosaic / reproject resume refused.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides,needle",
    [
        ({"drizzle_active_session": True}, "drizzle"),
        ({"is_mosaic_run": True, "drizzle_active_session": True}, "plain classic"),
        ({"reproject_between_batches": True}, "plain classic"),
        ({"reproject_coadd_final": True}, "plain classic"),
    ],
)
def test_unsupported_mode_resume_refused(tmp_path, overrides, needle):
    out = tmp_path
    session = build_session(out, n_sources=1)
    write_valid_checkpoint(out, (2, 2, 3), count=1, ledger=session["sources"], session=session)
    s = make_resume_stack(out, **overrides)
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is False
    assert needle in reason


# ---------------------------------------------------------------------------
# 11. Fresh initialization creates valid zero HWC SUM/WHT + manifest and does
#     not accidentally resume.
# ---------------------------------------------------------------------------
def test_initialize_fresh_creates_zeroed_and_manifest(tmp_path):
    out = tmp_path / "out"
    s = make_init_stack(out)
    assert s.initialize(str(out), (4, 5, 3)) is True
    assert s._resume_active is False
    assert s._checkpointing_enabled is True
    assert s.memmap_shape == (4, 5, 3)
    assert np.all(s.cumulative_sum_memmap == 0.0)
    assert np.all(s.cumulative_wht_memmap == 0.0)
    manifest = json.loads(s._resume_manifest_path().read_text())
    assert manifest["state"] == "clean"
    assert manifest["stacked_batches_count"] == 0
    assert manifest["completed_sources"] == []
    close_mm(s.cumulative_sum_memmap)
    close_mm(s.cumulative_wht_memmap)


def test_initialize_legacy_artifacts_fail_closed_unchanged(tmp_path):
    out = tmp_path / "out"
    memdir = out / "memmap_accumulators"
    memdir.mkdir(parents=True)
    sum_mm = np.lib.format.open_memmap(
        memdir / "cumulative_SUM.npy", mode="w+", dtype=np.float32, shape=(4, 5, 3)
    )
    sum_mm[:] = 9.0
    wht_mm = np.lib.format.open_memmap(
        memdir / "cumulative_WHT.npy", mode="w+", dtype=np.float32, shape=(4, 5)
    )
    wht_mm[:] = 2.0
    sum_mm.flush()
    wht_mm.flush()
    close_mm(sum_mm)
    close_mm(wht_mm)
    (out / "batches_count.txt").write_text("2")

    sum_bytes = (memdir / "cumulative_SUM.npy").read_bytes()
    wht_bytes = (memdir / "cumulative_WHT.npy").read_bytes()

    s = make_init_stack(out)
    assert s.initialize(str(out), (4, 5, 3)) is False
    assert (memdir / "cumulative_SUM.npy").read_bytes() == sum_bytes
    assert (memdir / "cumulative_WHT.npy").read_bytes() == wht_bytes


def test_initialize_resume_preserves_memmaps(tmp_path):
    out = tmp_path / "out"
    session = build_session(out, n_sources=3)
    write_valid_checkpoint(
        out, (4, 5, 3), count=3, ledger=session["sources"],
        session=session, sum_val=1.5, wht_val=2.5,
    )
    s = make_init_stack(out)
    bind_session(s, session)
    assert s.initialize(str(out), (4, 5, 3)) is True
    assert s._resume_active is True
    assert s.stacked_batches_count == 3
    assert np.all(s.cumulative_sum_memmap == 1.5)
    assert np.all(s.cumulative_wht_memmap == 2.5)
    assert s._resume_completed_sources == session["sources"]
    close_mm(s.cumulative_sum_memmap)
    close_mm(s.cumulative_wht_memmap)


# ---------------------------------------------------------------------------
# 12. Unrelated partial-save behaviour preserved (regression guard).
# ---------------------------------------------------------------------------
def test_save_partial_stack(tmp_path):
    out = tmp_path
    s = SeestarQueuedStacker()
    s.output_folder = str(out)
    s.output_filename = "stack"
    s.cumulative_sum_memmap = np.zeros((2, 2, 3), dtype=np.float32)
    s.cumulative_wht_memmap = np.ones((2, 2), dtype=np.float32)
    s.stacked_batches_count = 2
    s.partial_save_interval = 1

    class DummyVar:
        def set(self, value):
            self.value = value

    s.gui = types.SimpleNamespace(last_stack_path=DummyVar())

    prev = out / "stack_batch001.fit"
    prev.write_bytes(b"test")

    s._save_partial_stack()

    expected = out / "stack_batch002.fit"
    assert expected.exists()
    assert not prev.exists()


def test_save_partial_stack_failure_keeps_previous(tmp_path, monkeypatch):
    out = tmp_path
    s = SeestarQueuedStacker()
    s.output_folder = str(out)
    s.output_filename = "stack"
    s.cumulative_sum_memmap = np.zeros((2, 2, 3), dtype=np.float32)
    s.cumulative_wht_memmap = np.ones((2, 2), dtype=np.float32)
    s.stacked_batches_count = 2
    s.partial_save_interval = 1

    prev = out / "stack_batch001.fit"
    prev.write_bytes(b"test")

    def fail_replace(src, dst):
        raise RuntimeError("boom")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(RuntimeError):
        s._save_partial_stack()

    assert prev.exists()


# ===========================================================================
# HSI-2B C1 regressions: session/reference/plan binding and checkpoint protocol
# ===========================================================================

# ---------------------------------------------------------------------------
# 13. Missing completed source: refusal, artifacts unchanged.
# ---------------------------------------------------------------------------
def test_missing_completed_source_refused_artifacts_unchanged(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=2)
    # point one completed-source identity at a now-deleted file
    ledger = list(session["sources"])
    removed = Path(ledger[0]["path"])
    removed.unlink()
    write_valid_checkpoint(out, (2, 2, 3), count=2, ledger=ledger, session=session)

    manifest_path = out / "memmap_accumulators" / "resume_manifest.json"
    manifest_before = manifest_path.read_bytes()
    sum_before = (out / "memmap_accumulators" / "cumulative_SUM.npy").read_bytes()

    s = make_resume_stack(out)
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is False
    assert "missing or changed" in reason or "missing" in reason
    assert manifest_path.read_bytes() == manifest_before
    assert (out / "memmap_accumulators" / "cumulative_SUM.npy").read_bytes() == sum_before


# ---------------------------------------------------------------------------
# 14. Completed source moved to stacked/ with preserved identity: accepted.
# ---------------------------------------------------------------------------
def test_completed_source_moved_to_stacked_accepted(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=3)
    # move the first completed source to the default stacked/ location
    src0 = session["sources"][0]
    orig = Path(src0["path"])
    stacked_dir = orig.parent / "stacked"
    stacked_dir.mkdir()
    moved = stacked_dir / orig.name
    shutil.move(str(orig), str(moved))

    write_valid_checkpoint(out, (2, 2, 3), count=3, ledger=session["sources"], session=session)

    s = make_resume_stack(out)
    bind_session(s, session)
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is True, reason
    # the completed source is resolved to its moved-to-stacked counterpart
    resolved = s._resolve_source_path(src0)
    assert resolved == os.path.normcase(os.path.abspath(str(moved)))
    close_mm(s.cumulative_sum_memmap)
    close_mm(s.cumulative_wht_memmap)


# ---------------------------------------------------------------------------
# 15. Different input root/dataset with same shape/settings: refusal.
# ---------------------------------------------------------------------------
def test_different_input_root_refused(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=2)
    write_valid_checkpoint(out, (2, 2, 3), count=2, ledger=session["sources"], session=session)

    s = make_resume_stack(out)
    s._resume_input_roots = [os.path.normcase(os.path.abspath(str(tmp_path / "other" / "input")))]
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is False
    assert "input roots" in reason


# ---------------------------------------------------------------------------
# 16. Same path but changed size/mtime: refusal (validation level).
# ---------------------------------------------------------------------------
def test_same_path_changed_size_refused(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=1)
    ledger = list(session["sources"])
    # overwrite the same path with different content (size changes)
    p = Path(ledger[0]["path"])
    p.write_bytes(b"\x01" * 5000)
    write_valid_checkpoint(out, (2, 2, 3), count=1, ledger=ledger, session=session)

    s = make_resume_stack(out)
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is False
    assert "missing or changed" in reason


# ---------------------------------------------------------------------------
# 17. Extra / missing / reordered / regrouped planned source: refusal.
# ---------------------------------------------------------------------------
def _plan_checkpoint_and_queue(out, session, ledger, queue_sources):
    """Write a checkpoint and build a resume stacker whose queue is the given
    ordered remaining source list (with optional batch-break tokens)."""
    write_valid_checkpoint(out, (2, 2, 3), count=len(ledger), ledger=ledger, session=session)
    s = make_resume_stack(out)
    bind_session(s, session)
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is True, reason
    s._resume_completed_sources = list(ledger)
    s._resume_plan = dict(session["plan"])
    s.queue = Queue()
    for item in queue_sources:
        s.queue.put(item)
    s._checkpointing_enabled = True
    return s


def test_plan_extra_source_refused(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=3)
    ledger = session["sources"][:2]
    remaining = [session["sources"][2]["path"]]
    # add an extra file to the input and put it in the queue
    extra = _write_file(Path(session["input_dir"]) / "extra.fits", 777)
    s = _plan_checkpoint_and_queue(out, session, ledger, remaining + [extra])
    assert s._checkpoint_preflight() is False


def test_plan_missing_source_refused(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=4)
    ledger = session["sources"][:2]
    # two sources should remain; only one is present in the queue
    s = _plan_checkpoint_and_queue(out, session, ledger, [session["sources"][2]["path"]])
    assert s._checkpoint_preflight() is False


def test_plan_reordered_source_refused(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=3)
    ledger = session["sources"][:1]
    remaining = [session["sources"][2]["path"], session["sources"][1]["path"]]
    s = _plan_checkpoint_and_queue(out, session, ledger, remaining)
    assert s._checkpoint_preflight() is False


def test_plan_regrouped_source_refused(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=4)
    # plan decomposition is [2, 2] (two batches of two); a resume that
    # regroups the remaining two sources into [1, 1] must be refused.
    session["plan"] = {"sources": session["sources"], "decomposition": [2, 2]}
    ledger = session["sources"][:2]
    remaining = [s["path"] for s in session["sources"][2:]]
    s = _plan_checkpoint_and_queue(out, session, ledger, remaining)
    # insert a batch-break token to regroup the remaining into two batches
    s.queue = Queue()
    s.queue.put(session["sources"][2]["path"])
    s.queue.put(_BATCH_BREAK_TOKEN)
    s.queue.put(session["sources"][3]["path"])
    assert s._checkpoint_preflight() is False


def test_plan_exact_match_accepted(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=3)
    ledger = session["sources"][:2]
    remaining = [session["sources"][2]["path"]]
    s = _plan_checkpoint_and_queue(out, session, ledger, remaining)
    assert s._checkpoint_preflight() is True


# ---------------------------------------------------------------------------
# 18. Reference identity: moved-to-stacked resolved/reused; missing/replaced
#     reference refused.
# ---------------------------------------------------------------------------
def test_reference_moved_to_stacked_resolved(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=3)
    # move the reference to stacked/
    ref_orig = Path(session["ref_path"])
    stacked_dir = ref_orig.parent / "stacked"
    stacked_dir.mkdir()
    moved = stacked_dir / ref_orig.name
    shutil.move(str(ref_orig), str(moved))

    write_valid_checkpoint(out, (2, 2, 3), count=0, ledger=[], session=session)

    s = make_resume_stack(out)
    assert s._resolve_resume_reference() == os.path.normcase(os.path.abspath(str(moved)))


def test_reference_missing_refused(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=2)
    write_valid_checkpoint(out, (2, 2, 3), count=0, ledger=[], session=session)
    Path(session["ref_path"]).unlink()

    s = make_resume_stack(out)
    assert s._resolve_resume_reference() is None
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    # the reference itself is not a completed source, so validation of the
    # completed ledger still passes; the reference identity is only resolved by
    # the preflight seam (tested above).  Assert the session reference is not
    # silently replaced by a different frame.
    s._resume_reference_identity = None
    ok2, reason2 = s._validate_and_open_resume((2, 2, 3))
    assert ok2 is False
    assert "reference" in reason2


def test_reference_replaced_refused(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=2)
    write_valid_checkpoint(out, (2, 2, 3), count=0, ledger=[], session=session)
    # replace the reference with a different file (different size)
    Path(session["ref_path"]).write_bytes(b"\x02" * 9999)

    s = make_resume_stack(out)
    assert s._resolve_resume_reference() is None


# ---------------------------------------------------------------------------
# 19. Fingerprint/session contract changes when input/reference/plan changes.
# ---------------------------------------------------------------------------
def test_fingerprint_changes_with_session(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=2)
    write_valid_checkpoint(out, (2, 2, 3), count=2, ledger=session["sources"], session=session)

    manifest = json.loads((out / "memmap_accumulators" / "resume_manifest.json").read_text())
    # The scientific fingerprint is settings-only; the session carries the
    # roots/reference/plan.  Changing the reference identity must be refused.
    s = make_resume_stack(out)
    s._resume_reference_identity = dict(session["reference"])
    s._resume_reference_identity["mtime_ns"] = 999
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is False
    assert "reference" in reason


# ---------------------------------------------------------------------------
# 20. Dirty-write failure before mutation: raises/stops and SUM/W unchanged.
# ---------------------------------------------------------------------------
def test_dirty_write_failure_aborts_before_mutation(tmp_path, monkeypatch):
    out = tmp_path
    session = build_session(out, n_sources=1)
    stack = make_resume_stack(out)
    bind_session(stack, session)
    assert stack._initialize_classic_sumw_accumulators((2, 2, 3)) is True
    before_sum = np.array(stack.cumulative_sum_memmap, copy=True)
    before_wht = np.array(stack.cumulative_wht_memmap, copy=True)

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(stack, "_write_resume_manifest", fail_write)

    v = np.ones((2, 2, 3), dtype=np.float32)
    w = np.ones((2, 2, 3), dtype=np.float32)
    hdr = fits.Header()
    hdr["NIMAGES"] = 1
    hdr["TOTEXP"] = 1.0
    # The dirty mark itself is mandatory: it raises, never warn-and-continue.
    with pytest.raises(_ResumeCheckpointError):
        stack._checkpoint_mark_dirty()

    # _combine_batch_result aborts before either accumulator mutates.
    stack._combine_batch_result(v, hdr, w)

    assert stack.stop_processing is True
    assert np.array_equal(stack.cumulative_sum_memmap, before_sum)
    assert np.array_equal(stack.cumulative_wht_memmap, before_wht)
    close_mm(stack.cumulative_sum_memmap)
    close_mm(stack.cumulative_wht_memmap)


# ---------------------------------------------------------------------------
# 21. Clean-commit failure after mutation: manifest remains dirty and resume
#     refuses.
# ---------------------------------------------------------------------------
def test_clean_commit_failure_leaves_dirty(tmp_path, monkeypatch):
    out = tmp_path
    session = build_session(out, n_sources=1)
    stack = make_resume_stack(out)
    bind_session(stack, session)
    assert stack._initialize_classic_sumw_accumulators((2, 2, 3)) is True

    manifest_path = stack._resume_manifest_path()
    orig_write = stack._write_resume_manifest
    calls = {"n": 0}

    def flaky_write(*args, **kwargs):
        calls["n"] += 1
        if kwargs.get("state") == "clean":
            raise OSError("disk full")
        return orig_write(*args, **kwargs)

    monkeypatch.setattr(stack, "_write_resume_manifest", flaky_write)

    v = np.ones((2, 2, 3), dtype=np.float32)
    w = np.ones((2, 2, 3), dtype=np.float32)
    hdr = fits.Header()
    hdr["NIMAGES"] = 1
    hdr["TOTEXP"] = 1.0
    # _combine_batch_result surfaces the checkpoint failure (stops processing);
    # the exception is already exercised at the _checkpoint_mark_dirty level.
    stack._combine_batch_result(v, hdr, w)
    assert stack.stop_processing is True

    # manifest is still dirty (the clean commit failed), resume refuses
    assert json.loads(manifest_path.read_text())["state"] == "dirty"
    other = make_resume_stack(out)
    ok, reason = other._validate_and_open_resume((2, 2, 3))
    assert ok is False
    assert "clean" in reason
    close_mm(stack.cumulative_sum_memmap)
    close_mm(stack.cumulative_wht_memmap)


# ---------------------------------------------------------------------------
# 22. Clean checkpoint restores batch count, completed ledger,
#     images_in_cumulative_stack, total_exposure_seconds, and header
#     NIMAGES/TOTEXP.
# ---------------------------------------------------------------------------
def test_clean_checkpoint_restores_scientific_metadata(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=3)
    hdr = fits.Header()
    hdr["NIMAGES"] = 7
    hdr["TOTEXP"] = 12.5
    hdr["STACKTYP"] = "Classic SUM/W (kappa-sigma)"
    hdr["OBJECT"] = "M42"
    write_valid_checkpoint(
        out, (2, 2, 3), count=3, ledger=session["sources"], session=session,
        images_in=7, total_exposure=12.5, header=hdr,
    )

    s = make_resume_stack(out)
    bind_session(s, session)
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is True, reason
    assert s.stacked_batches_count == 3
    assert s.images_in_cumulative_stack == 7
    assert s.total_exposure_seconds == pytest.approx(12.5)
    assert s.current_stack_header["NIMAGES"] == 7
    assert s.current_stack_header["TOTEXP"] == pytest.approx(12.5)
    assert s.current_stack_header["OBJECT"] == "M42"
    close_mm(s.cumulative_sum_memmap)
    close_mm(s.cumulative_wht_memmap)


# ---------------------------------------------------------------------------
# 23. Ledger with missing/null/non-integer size/mtime, duplicate identities,
#     or invalid path fails closed.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad_entry",
    [
        {"name": "a.fits", "size": 100, "mtime_ns": 1},          # missing path
        {"path": "x", "name": "a.fits", "size": None, "mtime_ns": 1},   # null size
        {"path": "x", "name": "a.fits", "size": 100, "mtime_ns": None},  # null mtime
        {"path": "x", "name": "a.fits", "size": 100.5, "mtime_ns": 1},  # non-int size
        {"path": "x", "name": "a.fits", "size": 100, "mtime_ns": "1"},  # non-int mtime
        {"path": 123, "name": "a.fits", "size": 100, "mtime_ns": 1},    # invalid path type
    ],
)
def test_ledger_bad_entry_fails_closed(tmp_path, bad_entry):
    out = tmp_path
    memdir = out / "memmap_accumulators"
    memdir.mkdir(parents=True)
    shape = (2, 2, 3)
    for name in ("cumulative_SUM.npy", "cumulative_WHT.npy"):
        mm = np.lib.format.open_memmap(memdir / name, mode="w+", dtype=np.float32, shape=shape)
        mm[:] = 1.0
        mm.flush()
        close_mm(mm)
    stack = make_resume_stack(out)
    stack.memmap_shape = shape
    stack._resume_plan = {"sources": [], "decomposition": []}
    stack._write_resume_manifest(state="clean", completed_sources=[bad_entry], stacked_batches_count=1)

    s = make_resume_stack(out)
    ok, reason = s._validate_and_open_resume(shape)
    assert ok is False
    assert "ledger" in reason or "path" in reason or "size/mtime" in reason


def test_ledger_duplicate_identity_fails_closed(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=1)
    dup = [session["sources"][0], session["sources"][0]]
    write_valid_checkpoint(out, (2, 2, 3), count=2, ledger=dup, session=session)
    s = make_resume_stack(out)
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is False
    assert "duplicate" in reason


# ---------------------------------------------------------------------------
# 24. Production-flow seam: valid resume forces the original reference and
#     exact remaining queue; invalid session never starts.
# ---------------------------------------------------------------------------
def test_seam_valid_resume_forces_reference_and_queue(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=3)
    # move the reference and the first completed source to stacked/
    ref_orig = Path(session["ref_path"])
    stacked_dir = ref_orig.parent / "stacked"
    stacked_dir.mkdir()
    moved_ref = stacked_dir / ref_orig.name
    shutil.move(str(ref_orig), str(moved_ref))

    src0 = Path(session["sources"][0]["path"])
    shutil.move(str(src0), str(stacked_dir / src0.name))

    ledger = session["sources"][:2]
    write_valid_checkpoint(out, (2, 2, 3), count=2, ledger=ledger, session=session)

    s = make_resume_stack(out)
    bind_session(s, session)
    # the reference is resolved to its stacked counterpart and forced on the
    # aligner before any auto-selection.
    resolved_ref = s._resolve_resume_reference()
    assert resolved_ref == os.path.normcase(os.path.abspath(str(moved_ref)))
    s.aligner.reference_image_path = resolved_ref

    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is True, reason
    s._resume_completed_sources = list(ledger)
    s._resume_plan = dict(session["plan"])

    # remaining queue = the one un-consumed source
    s.queue = Queue()
    s.queue.put(session["sources"][2]["path"])
    skipped = s._filter_queue_by_resume_ledger()
    assert skipped == 0
    # preflight (plan validation) passes, so the worker would start
    assert s._checkpoint_preflight() is True
    close_mm(s.cumulative_sum_memmap)
    close_mm(s.cumulative_wht_memmap)


def test_seam_invalid_session_never_starts(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=2)
    write_valid_checkpoint(out, (2, 2, 3), count=2, ledger=session["sources"], session=session)

    # wrong input root -> validation refuses (worker never starts)
    s = make_resume_stack(out)
    s._resume_input_roots = [os.path.normcase(os.path.abspath(str(tmp_path / "elsewhere")))]
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is False
    assert "input roots" in reason


# ===========================================================================
# HSI-2B C2 regressions: automatic decomposition derivation + pre-reference
# fail-closed seam.
# ===========================================================================

# ---------------------------------------------------------------------------
# 25. Automatic batch decomposition: a persisted [2,2,2] plan with the first
#     two sources completed must derive [2,2] for the remaining four (not [4]).
# ---------------------------------------------------------------------------
def _auto_batch_resume_stack(out, session, ledger, queue_paths, batch_size):
    """Resume stacker with an automatic (break-token-free) remaining queue."""
    s = make_resume_stack(out, batch_size=batch_size)
    bind_session(s, session)
    s._resume_completed_sources = list(ledger)
    s._resume_plan = dict(session["plan"])
    s._resume_active = True
    s._checkpointing_enabled = True
    s.use_batch_plan = False
    s.queue = Queue()
    for p in queue_paths:
        s.queue.put(p)
    return s


def test_auto_batch_remaining_decomposition_counterexample(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=6)
    session["plan"] = {"sources": session["sources"], "decomposition": [2, 2, 2]}
    ledger = session["sources"][:2]
    remaining = [s_["path"] for s_ in session["sources"][2:]]

    s = _auto_batch_resume_stack(out, session, ledger, remaining, batch_size=2)

    sources, decomp, has_breaks = s._scan_queue_decomposition()
    assert decomp == [2, 2], decomp
    assert has_breaks is False
    assert s._validate_plan_against_manifest() == (True, None)
    assert s._checkpoint_preflight() is True


def test_auto_batch_partial_remaining(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=5)
    session["plan"] = {"sources": session["sources"], "decomposition": [2, 2, 1]}
    ledger = session["sources"][:2]
    remaining = [s_["path"] for s_ in session["sources"][2:]]

    s = _auto_batch_resume_stack(out, session, ledger, remaining, batch_size=2)

    sources, decomp, has_breaks = s._scan_queue_decomposition()
    assert decomp == [2, 1], decomp
    assert s._validate_plan_against_manifest() == (True, None)
    assert s._checkpoint_preflight() is True


# ---------------------------------------------------------------------------
# 26. Real regrouping (automatic and explicit break) still refused.
# ---------------------------------------------------------------------------
def test_auto_batch_regroup_refused(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=6)
    session["plan"] = {"sources": session["sources"], "decomposition": [2, 2, 2]}
    ledger = session["sources"][:2]
    remaining = [s_["path"] for s_ in session["sources"][2:]]

    # Regroup the remaining four into a single batch by raising batch_size.
    s = _auto_batch_resume_stack(out, session, ledger, remaining, batch_size=4)
    sources, decomp, has_breaks = s._scan_queue_decomposition()
    assert decomp == [4], decomp
    ok, reason = s._validate_plan_against_manifest()
    assert ok is False
    assert "decomposition" in reason


def test_auto_batch_explicit_break_regroup_refused(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=6)
    session["plan"] = {"sources": session["sources"], "decomposition": [2, 2, 2]}
    ledger = session["sources"][:2]

    s = _auto_batch_resume_stack(out, session, ledger, [], batch_size=2)
    # Insert explicit break tokens to regroup the remaining [2,2] into four
    # singletons [1,1,1,1] -> must refuse.
    s.queue = Queue()
    for s_ in session["sources"][2:]:
        s.queue.put(s_["path"])
        s.queue.put(_BATCH_BREAK_TOKEN)
    ok, reason = s._validate_plan_against_manifest()
    assert ok is False


# ---------------------------------------------------------------------------
# 27. Actual start_processing seam helpers.
# ---------------------------------------------------------------------------
class _SpyReferenceAligner:
    """Records _get_reference_image calls and the pinned reference path."""

    def __init__(self, shape=(4, 5, 3)):
        self.stop_processing = False
        self.reference_image_path = None
        self.calls = []
        self._shape = shape

    def _get_reference_image(self, folder, files, output_folder):
        self.calls.append(
            {
                "folder": folder,
                "files": list(files),
                "output_folder": output_folder,
                "reference_image_path": self.reference_image_path,
            }
        )
        return (np.zeros(self._shape, dtype=np.float32), fits.Header())


def _make_start_processing_stack(out_dir, input_dir, aligner=None, **overrides):
    """Bare stacker with enough state for a real start_processing call."""
    o = _make_resume_stack(out_dir)
    o.processing_active = False
    o.user_requested_stop = False
    o.autotuner = None
    o.current_folder = str(input_dir)
    o.output_folder = str(out_dir)
    o._has_stack_plan = False
    o._interbatch_start_session = lambda: None
    o._derive_drizzle_processing_policy = lambda: None
    o.freeze_reference_wcs = False
    o.stacked_subdir_name = "stacked"
    o.additional_folders = []
    o.reference_pixel_scale_arcsec = None
    o.stack_final_combine = "mean"
    o.use_batch_plan = False
    o.folders_lock = threading.Lock()
    o.processed_files = set()
    o._resume_resolved_reference = None
    if aligner is not None:
        o.aligner = aligner
    for k, v in overrides.items():
        setattr(o, k, v)
    return o


def _make_invalid_resume_scenario(tmp_path, scenario):
    """Build a resume checkpoint, apply the scenario mutation, return
    (out_dir, start_kwargs)."""
    out = tmp_path / "out"
    out.mkdir()
    session = build_session(out, n_sources=2)
    shape = (2, 2, 3)
    write_valid_checkpoint(
        out, shape, count=2, ledger=session["sources"], session=session
    )
    start_kwargs = dict(
        input_dir=session["input_dir"], output_dir=str(out), batch_size=10,
        resume_intent="resume",
    )
    if scenario == "dirty":
        stack = make_resume_stack(out)
        bind_session(stack, session)
        stack.memmap_shape = shape
        stack._write_resume_manifest(
            state="dirty",
            completed_sources=session["sources"],
            stacked_batches_count=2,
        )
    elif scenario == "corrupt":
        (out / "memmap_accumulators" / "resume_manifest.json").write_text(
            "{ not valid json", encoding="utf-8"
        )
    elif scenario == "fingerprint":
        start_kwargs["stacking_mode"] = "mean"
    elif scenario == "root":
        other = tmp_path / "other"
        other.mkdir()
        start_kwargs["input_dir"] = str(other)
    elif scenario == "reference_missing":
        Path(session["ref_path"]).unlink()
    elif scenario == "reference_replaced":
        Path(session["ref_path"]).write_bytes(b"\x02" * 9999)
    elif scenario == "session_missing":
        mp = out / "memmap_accumulators" / "resume_manifest.json"
        m = json.loads(mp.read_text(encoding="utf-8"))
        m.pop("session", None)
        mp.write_text(json.dumps(m), encoding="utf-8")
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    return out, start_kwargs


@pytest.mark.parametrize(
    "scenario",
    [
        "dirty",
        "corrupt",
        "fingerprint",
        "root",
        "reference_missing",
        "reference_replaced",
        "session_missing",
    ],
)
def test_start_processing_invalid_resume_fails_before_reference(tmp_path, scenario):
    out, start_kwargs = _make_invalid_resume_scenario(tmp_path, scenario)

    # Pre-existing sentinel reference artifacts (must stay byte-identical).
    temp_dir = out / "temp_processing"
    temp_dir.mkdir()
    fit_sentinel = temp_dir / "reference_image.fit"
    png_sentinel = temp_dir / "reference_image.png"
    fit_sentinel.write_bytes(b"FIT-SENTINEL")
    png_sentinel.write_bytes(b"PNG-SENTINEL")

    manifest_path = out / "memmap_accumulators" / "resume_manifest.json"
    sum_path = out / "memmap_accumulators" / "cumulative_SUM.npy"
    wht_path = out / "memmap_accumulators" / "cumulative_WHT.npy"
    before = {
        "fit": fit_sentinel.read_bytes(),
        "png": png_sentinel.read_bytes(),
        "manifest": manifest_path.read_bytes() if manifest_path.exists() else None,
        "sum": sum_path.read_bytes() if sum_path.exists() else None,
        "wht": wht_path.read_bytes() if wht_path.exists() else None,
    }

    aligner = _SpyReferenceAligner()
    s = _make_start_processing_stack(
        out, Path(start_kwargs["input_dir"]), aligner=aligner
    )

    result = s.start_processing(**start_kwargs)

    assert result is False
    # _get_reference_image never called -> no reference artifact write.
    assert aligner.calls == []
    assert not hasattr(s, "processing_thread")
    assert s.processing_active is False
    # Sentinel reference artifacts + manifest/SUM/WHT all untouched.
    assert fit_sentinel.read_bytes() == before["fit"]
    assert png_sentinel.read_bytes() == before["png"]
    assert manifest_path.read_bytes() == before["manifest"]
    assert sum_path.read_bytes() == before["sum"]
    assert wht_path.read_bytes() == before["wht"]


def test_start_processing_valid_resume_pins_reference_and_queue(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    session = build_session(out, n_sources=6)
    session["plan"] = {"sources": session["sources"], "decomposition": [2, 2, 2]}
    ledger = session["sources"][:2]

    # Move the reference to stacked/ (verified moved-to-stacked counterpart).
    ref_orig = Path(session["ref_path"])
    stacked_dir = ref_orig.parent / "stacked"
    stacked_dir.mkdir()
    moved_ref = stacked_dir / ref_orig.name
    shutil.move(str(ref_orig), str(moved_ref))

    # Write a valid clean checkpoint with batch_size=2 (matching start config).
    shape = (2, 2, 3)
    memdir = out / "memmap_accumulators"
    memdir.mkdir(parents=True, exist_ok=True)
    for name, val in (("cumulative_SUM.npy", 1.0), ("cumulative_WHT.npy", 2.0)):
        mm = np.lib.format.open_memmap(
            memdir / name, mode="w+", dtype=np.float32, shape=shape
        )
        mm[:] = val
        mm.flush()
        close_mm(mm)
    stack = make_resume_stack(out, batch_size=2)
    bind_session(stack, session)
    stack.memmap_shape = shape
    stack._resume_completed_sources = list(ledger)
    stack.stacked_batches_count = 2
    stack._write_resume_manifest(
        state="clean", completed_sources=list(ledger), stacked_batches_count=2
    )

    aligner = _SpyReferenceAligner()
    s = _make_start_processing_stack(out, session["input_dir"], aligner=aligner)

    def fake_initialize(self, output_dir, shape_hwc, enable_preview=False):
        self._resume_active = True
        self._resume_completed_sources = list(ledger)
        self._resume_plan = dict(session["plan"])
        self._checkpointing_enabled = True
        self.stacked_batches_count = 2
        return True

    all_paths = [s_["path"] for s_ in session["sources"]]

    def fake_add_files_to_queue(self, folder):
        self.files_in_queue = 0
        for p in all_paths:
            self.queue.put(p)
            self.files_in_queue += 1
        self.use_batch_plan = False
        return len(all_paths)

    s.initialize = types.MethodType(fake_initialize, s)
    s._add_files_to_queue = types.MethodType(fake_add_files_to_queue, s)
    s._worker = lambda: None

    result = s.start_processing(
        input_dir=session["input_dir"], output_dir=str(out), batch_size=2,
        resume_intent="resume",
    )

    assert result is True
    # _get_reference_image was called exactly once, with the moved-to-stacked
    # reference already pinned on the aligner before the call.
    assert len(aligner.calls) == 1
    assert os.path.normcase(os.path.abspath(aligner.calls[0]["reference_image_path"])) == os.path.normcase(
        os.path.abspath(str(moved_ref))
    )
    # The worker thread started.
    assert hasattr(s, "processing_thread")
    assert s.processing_active is True
    # The later queue preflight observed the exact remaining auto-batched
    # suffix [2,2].
    remaining = [p for p in list(s.queue.queue) if p != _BATCH_BREAK_TOKEN]
    assert len(remaining) == 4
    sources, decomp, has_breaks = s._scan_queue_decomposition()
    assert decomp == [2, 2], decomp
    assert has_breaks is False
    assert s._validate_plan_against_manifest() == (True, None)


def test_worker_normalization_reference_uses_pinned_original(tmp_path):
    """P1-FIX seam: the normalization reference captured by the worker is the
    pinned global classic-alignment reference (returned by
    ``_get_reference_image``), never a batch output or the first remaining
    source."""
    import inspect

    # The worker captures the reference from the real global reference data,
    # before that array can later be replaced by intermediate stacks.
    src = inspect.getsource(SeestarQueuedStacker._worker)
    assert "self._capture_normalization_reference(" in src
    assert "reference_image_data_for_global_alignment" in src

    # Behavioral: a distinctive pinned reference is what the normalization
    # reference resolves to, and a different source is normalized *against* it
    # (not adopted as the reference).
    orig_ref = np.full((4, 4, 3), 9.0, dtype=np.float32)

    class _PinnedRefAligner:
        def __init__(self):
            self.reference_image_path = None

        def _get_reference_image(self, folder, files, output_folder):
            # The pinned original reference (already moved to stacked/), not a
            # first remaining source, is what the aligner returns.
            return (orig_ref, fits.Header())

    stack = _make_resume_stack(tmp_path / "out")
    stack.aligner = _PinnedRefAligner()
    stack.aligner.reference_image_path = str(tmp_path / "stacked" / "ref.fits")
    stack.normalize_method = "linear_fit"

    ref_data, _ = stack.aligner._get_reference_image(None, None, None)
    stack._capture_normalization_reference(ref_data)
    assert np.array_equal(stack._norm_reference, orig_ref)

    first_source = np.full((4, 4, 3), 2.0, dtype=np.float32)
    normalized = stack._normalize_sources_against_reference(
        [np.array(first_source, dtype=np.float32, copy=True)]
    )
    # The reference is untouched, and the source was mapped onto it (linear_fit
    # of a constant source shifts it to the reference level 9.0), proving the
    # pinned original reference drives normalization — not the first source.
    assert np.array_equal(stack._norm_reference, orig_ref)
    assert np.allclose(normalized[0], orig_ref, atol=1e-3)
    assert not np.allclose(normalized[0], first_source, atol=1e-3)


# ===========================================================================
# HSI-2B C3 regressions: complete fail-closed schema + artifact validation
# before reference preparation.
# ===========================================================================

def _write_fits_reference(path, h, w, seed=0):
    data = (np.random.default_rng(seed).random((h, w)) * 100.0).astype(np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=data).writeto(path, overwrite=True)
    return str(path)


def _resume_requested_stack(out, **overrides):
    return make_resume_stack(out, _resume_requested=True, **overrides)


def _manifest_path(out):
    return Path(out) / "memmap_accumulators" / "resume_manifest.json"


def _mutate_manifest(out, mutator):
    mp = _manifest_path(out)
    m = json.loads(mp.read_text(encoding="utf-8"))
    mutator(m)
    mp.write_text(json.dumps(m), encoding="utf-8")
    return m


def _early_refuses(out, session, needle=None):
    s = _resume_requested_stack(out)
    bind_session(s, session)
    ok, detail = s._early_resume_preflight()
    assert ok is False
    if needle is not None:
        assert needle in detail, detail
    return detail


def _make_fits_seam_base(tmp_path, ref_hw=(2, 2), shape=(2, 2, 3), n_sources=2):
    """A valid clean checkpoint whose reference is a real FITS file."""
    out = tmp_path / "out"
    out.mkdir()
    input_dir = out / "input"
    ref_path = _write_fits_reference(
        input_dir / "reference.fits", ref_hw[0], ref_hw[1]
    )
    ref_ident = _stat_identity(ref_path)
    roots = [os.path.normcase(os.path.abspath(str(input_dir)))]
    sources = []
    for i in range(n_sources):
        p = _write_file(input_dir / f"obs_{i:03d}.fits", 100 + i)
        sources.append(_stat_identity(p))
    plan = {"sources": list(sources), "decomposition": [1] * n_sources}
    session = {
        "input_dir": str(input_dir),
        "roots": roots,
        "reference": ref_ident,
        "ref_path": ref_path,
        "sources": sources,
        "plan": plan,
    }
    write_valid_checkpoint(
        out, shape, count=n_sources, ledger=sources, session=session
    )
    return out, session


def _run_seam_assert_refused(tmp_path, mutate=None, out=None, session=None,
                             ref_hw=(2, 2), shape=(2, 2, 3)):
    if out is None:
        out, session = _make_fits_seam_base(
            tmp_path, ref_hw=ref_hw, shape=shape
        )
    if mutate is not None:
        mutate(out, session)

    temp_dir = out / "temp_processing"
    temp_dir.mkdir()
    fit_sentinel = temp_dir / "reference_image.fit"
    png_sentinel = temp_dir / "reference_image.png"
    fit_sentinel.write_bytes(b"FIT-SENTINEL")
    png_sentinel.write_bytes(b"PNG-SENTINEL")

    manifest_path = out / "memmap_accumulators" / "resume_manifest.json"
    sum_path = out / "memmap_accumulators" / "cumulative_SUM.npy"
    wht_path = out / "memmap_accumulators" / "cumulative_WHT.npy"
    before = {
        "fit": fit_sentinel.read_bytes(),
        "png": png_sentinel.read_bytes(),
        "manifest": manifest_path.read_bytes() if manifest_path.exists() else None,
        "sum": sum_path.read_bytes() if sum_path.exists() else None,
        "wht": wht_path.read_bytes() if wht_path.exists() else None,
    }

    aligner = _SpyReferenceAligner()
    s = _make_start_processing_stack(out, Path(session["input_dir"]), aligner=aligner)
    result = s.start_processing(
        input_dir=session["input_dir"], output_dir=str(out), batch_size=10,
        resume_intent="resume",
    )

    assert result is False
    assert aligner.calls == []
    assert not hasattr(s, "processing_thread")
    assert s.processing_active is False
    if fit_sentinel.exists():
        assert fit_sentinel.read_bytes() == before["fit"]
    if png_sentinel.exists():
        assert png_sentinel.read_bytes() == before["png"]
    if manifest_path.exists():
        assert manifest_path.read_bytes() == before["manifest"]
    if sum_path.exists():
        assert sum_path.read_bytes() == before["sum"]
    if wht_path.exists():
        assert wht_path.read_bytes() == before["wht"]
    return s


# ---------------------------------------------------------------------------
# 28. The two exact parent witnesses now refuse at the early preflight.
# ---------------------------------------------------------------------------
def test_parent_witness_nonint_decomposition_refused(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=2)
    write_valid_checkpoint(
        out, (2, 2, 3), count=2, ledger=session["sources"], session=session
    )
    _mutate_manifest(
        out, lambda m: m["session"]["plan"].update(decomposition=["not-an-int"])
    )
    _early_refuses(out, session, "decomposition")


def test_parent_witness_null_source_size_refused(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=2)
    write_valid_checkpoint(
        out, (2, 2, 3), count=2, ledger=session["sources"], session=session
    )
    _mutate_manifest(
        out,
        lambda m: m["session"]["plan"]["sources"].__setitem__(
            0, dict(m["session"]["plan"]["sources"][0], size=None)
        ),
    )
    _early_refuses(out, session, "size")


# ---------------------------------------------------------------------------
# 29. Malformed plan identity/decomposition: deterministic refusal.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mutator, needle",
    [
        (lambda m: m["session"]["plan"]["sources"].__setitem__(0, dict(m["session"]["plan"]["sources"][0], size=None)), "size"),
        (lambda m: m["session"]["plan"]["sources"].__setitem__(0, dict(m["session"]["plan"]["sources"][0], size="100")), "size"),
        (lambda m: m["session"]["plan"]["sources"].__setitem__(0, dict(m["session"]["plan"]["sources"][0], size=True)), "size"),
        (lambda m: m["session"]["plan"]["sources"].__setitem__(0, dict(m["session"]["plan"]["sources"][0], mtime_ns=None)), "mtime_ns"),
        (lambda m: m["session"]["plan"]["sources"].__setitem__(0, dict(m["session"]["plan"]["sources"][0], mtime_ns="1")), "mtime_ns"),
        (lambda m: m["session"]["plan"]["sources"].__setitem__(0, dict(m["session"]["plan"]["sources"][0], mtime_ns=True)), "mtime_ns"),
        (lambda m: m["session"]["plan"]["sources"].__setitem__(0, dict(m["session"]["plan"]["sources"][0], path="")), "path"),
        (lambda m: m["session"]["plan"]["sources"].__setitem__(0, dict(m["session"]["plan"]["sources"][0], path=123)), "path"),
        (lambda m: m["session"]["plan"]["sources"].__setitem__(1, dict(m["session"]["plan"]["sources"][0])), "duplicate"),
        (lambda m: m["session"]["plan"].update(decomposition="not-a-list"), "decomposition"),
        (lambda m: m["session"]["plan"].update(decomposition=[0]), "strictly positive"),
        (lambda m: m["session"]["plan"].update(decomposition=[-1, 3]), "strictly positive"),
        (lambda m: m["session"]["plan"].update(decomposition=[True, 1]), "integer"),
        (lambda m: m["session"]["plan"].update(decomposition=["1", 1]), "integer"),
        (lambda m: m["session"]["plan"].update(decomposition=[3]), "sum"),
    ],
)
def test_malformed_plan_refused_before_reference(tmp_path, mutator, needle):
    out = tmp_path
    session = build_session(out, n_sources=2)
    write_valid_checkpoint(
        out, (2, 2, 3), count=2, ledger=session["sources"], session=session
    )
    _mutate_manifest(out, mutator)
    _early_refuses(out, session, needle)


# ---------------------------------------------------------------------------
# 30. Malformed reference identity fields: clean refusal, no exception.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mutator, needle",
    [
        (lambda m: m["session"].update(reference=None), "reference"),
        (lambda m: m["session"].update(reference="not-an-object"), "reference"),
        (lambda m: m["session"].update(reference=dict(m["session"]["reference"], path="")), "path"),
        (lambda m: m["session"].update(reference=dict(m["session"]["reference"], size=None)), "size"),
        (lambda m: m["session"].update(reference=dict(m["session"]["reference"], size=True)), "size"),
        (lambda m: m["session"].update(reference=dict(m["session"]["reference"], mtime_ns="abc")), "mtime_ns"),
    ],
)
def test_malformed_reference_identity_refused(tmp_path, mutator, needle):
    out = tmp_path
    session = build_session(out, n_sources=2)
    write_valid_checkpoint(
        out, (2, 2, 3), count=2, ledger=session["sources"], session=session
    )
    _mutate_manifest(out, mutator)
    _early_refuses(out, session, needle)


# ---------------------------------------------------------------------------
# 31. Ledger prefix / boundary violations: refusal before reference.
# ---------------------------------------------------------------------------
def test_ledger_not_exact_prefix_refused_before_reference(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=3)
    session["plan"] = {"sources": session["sources"], "decomposition": [1, 1, 1]}
    ledger = [session["sources"][0], session["sources"][2]]
    write_valid_checkpoint(out, (2, 2, 3), count=2, ledger=ledger, session=session)
    _early_refuses(out, session, "prefix")


def test_ledger_inside_batch_refused_before_reference(tmp_path):
    out = tmp_path
    session = build_session(out, n_sources=4)
    session["plan"] = {"sources": session["sources"], "decomposition": [2, 2]}
    ledger = session["sources"][:1]
    write_valid_checkpoint(out, (2, 2, 3), count=1, ledger=ledger, session=session)
    _early_refuses(out, session, "boundar")


# ---------------------------------------------------------------------------
# 32. Read-only reference shape probe: matching accepted, mismatch refused.
# ---------------------------------------------------------------------------
def test_early_preflight_accepts_matching_reference_shape(tmp_path):
    out, session = _make_fits_seam_base(tmp_path, ref_hw=(2, 2), shape=(2, 2, 3))
    s = _resume_requested_stack(out)
    bind_session(s, session)
    ok, ref = s._early_resume_preflight()
    assert ok is True
    assert ref == os.path.normcase(os.path.abspath(str(session["ref_path"])))
    assert s._resume_preflight_passed is True
    # The full open reuses the preflight validation and restores state.
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is True, reason
    assert s._resume_active is True
    assert s.stacked_batches_count == 2
    assert len(s._resume_completed_sources) == 2
    close_mm(s.cumulative_sum_memmap)
    close_mm(s.cumulative_wht_memmap)


def test_early_preflight_refuses_mismatched_reference_shape(tmp_path):
    out, session = _make_fits_seam_base(tmp_path, ref_hw=(4, 4), shape=(2, 2, 3))
    s = _resume_requested_stack(out)
    bind_session(s, session)
    ok, detail = s._early_resume_preflight()
    assert ok is False
    assert "shape" in detail


# ---------------------------------------------------------------------------
# 33. Actual start_processing seam: invalid persisted artifacts never reach
#     reference preparation and never modify target artifacts.
# ---------------------------------------------------------------------------
def _mut_missing_sum(out, session):
    (out / "memmap_accumulators" / "cumulative_SUM.npy").unlink()


def _mut_missing_wht(out, session):
    (out / "memmap_accumulators" / "cumulative_WHT.npy").unlink()


def _mut_corrupt_sum(out, session):
    (out / "memmap_accumulators" / "cumulative_SUM.npy").write_bytes(b"garbage-not-an-npy")


def _mut_2d_wht(out, session):
    p = out / "memmap_accumulators" / "cumulative_WHT.npy"
    p.unlink()
    mm = np.lib.format.open_memmap(str(p), mode="w+", dtype=np.float32, shape=(2, 2))
    mm[:] = 1.0
    mm.flush()
    close_mm(mm)


def _mut_wrong_dtype(out, session):
    p = out / "memmap_accumulators" / "cumulative_SUM.npy"
    old = np.lib.format.open_memmap(str(p), mode="r")
    vals = np.array(old, copy=True)
    close_mm(old)
    p.unlink()
    mm = np.lib.format.open_memmap(str(p), mode="w+", dtype=np.float64, shape=vals.shape)
    mm[:] = vals
    mm.flush()
    close_mm(mm)


def _mut_nonfinite_sum(out, session):
    p = out / "memmap_accumulators" / "cumulative_SUM.npy"
    mm = np.lib.format.open_memmap(str(p), mode="r+")
    mm[0, 0, 0] = np.nan
    mm.flush()
    close_mm(mm)


def _mut_nonfinite_wht(out, session):
    p = out / "memmap_accumulators" / "cumulative_WHT.npy"
    mm = np.lib.format.open_memmap(str(p), mode="r+")
    mm[0, 0, 0] = np.nan
    mm.flush()
    close_mm(mm)


def _mut_negative_wht(out, session):
    p = out / "memmap_accumulators" / "cumulative_WHT.npy"
    mm = np.lib.format.open_memmap(str(p), mode="r+")
    mm[0, 0, 0] = -1.0
    mm.flush()
    close_mm(mm)


def _mut_bad_counter(out, session):
    _mutate_manifest(out, lambda m: m.update(stacked_batches_count="abc"))


def _mut_bad_counter_negative(out, session):
    _mutate_manifest(out, lambda m: m.update(images_in_cumulative_stack=-3))


def _mut_bad_exposure(out, session):
    _mutate_manifest(out, lambda m: m.update(total_exposure_seconds=float("nan")))


def _mut_bad_header(out, session):
    _mutate_manifest(out, lambda m: m.update(cumulative_header="not-a-dict"))


_SEAM_MUTATORS = {
    "missing_sum": _mut_missing_sum,
    "missing_wht": _mut_missing_wht,
    "corrupt_npy": _mut_corrupt_sum,
    "2d_wht": _mut_2d_wht,
    "wrong_dtype": _mut_wrong_dtype,
    "nonfinite_sum": _mut_nonfinite_sum,
    "nonfinite_wht": _mut_nonfinite_wht,
    "negative_wht": _mut_negative_wht,
    "bad_counter": _mut_bad_counter,
    "bad_counter_negative": _mut_bad_counter_negative,
    "bad_exposure": _mut_bad_exposure,
    "bad_header": _mut_bad_header,
}


@pytest.mark.parametrize("scenario", sorted(_SEAM_MUTATORS))
def test_start_processing_invalid_artifact_fails_before_reference(tmp_path, scenario):
    _run_seam_assert_refused(tmp_path, mutate=_SEAM_MUTATORS[scenario])


def test_start_processing_reference_shape_mismatch_fails_before_reference(tmp_path):
    _run_seam_assert_refused(tmp_path, ref_hw=(4, 4), shape=(2, 2, 3))


# ---------------------------------------------------------------------------
# 34. HSI-2B C4: fail-closed dtype contract.  A persisted checkpoint is only
#     resumable when its manifest SUM/WHT dtypes equal the runtime-configured
#     scientific accumulator dtypes (float32) AND the on-disk SUM/WHT dtypes
#     match.  Matching non-float32 artifacts (int64 / Unicode / complex /
#     bool) must be refused *before* reference preparation with a clean
#     (False, reason) — never a TypeError from np.isfinite / comparison.
# ---------------------------------------------------------------------------
def _rewrite_artifacts_dtype(out, sum_dtype, wht_dtype, sum_fill, wht_fill):
    """Rewrite the on-disk SUM/WHT artifacts with the given dtype, preserving
    the existing shape, and leave the manifest dtype fields to be mutated."""
    memdir = out / "memmap_accumulators"
    sum_path = memdir / "cumulative_SUM.npy"
    wht_path = memdir / "cumulative_WHT.npy"
    old = np.lib.format.open_memmap(str(sum_path), mode="r")
    shape = old.shape
    close_mm(old)
    for path, dtype, fill in (
        (sum_path, sum_dtype, sum_fill),
        (wht_path, wht_dtype, wht_fill),
    ):
        path.unlink()
        mm = np.lib.format.open_memmap(str(path), mode="w+", dtype=dtype, shape=shape)
        mm[:] = fill
        mm.flush()
        close_mm(mm)


def _mut_int64_dtype(out, session):
    # Architect reproduction 1: matching int64 manifest + int64 SUM/WHT.
    _rewrite_artifacts_dtype(out, np.int64, np.int64, 1, 2)
    _mutate_manifest(out, lambda m: m.update(dtype_sum="int64", dtype_wht="int64"))


def _mut_unicode_dtype(out, session):
    # Architect reproduction 2: matching Unicode manifest + Unicode SUM/WHT.
    _rewrite_artifacts_dtype(out, "<U1", "<U1", "1", "2")
    _mutate_manifest(out, lambda m: m.update(dtype_sum="<U1", dtype_wht="<U1"))


def _mut_complex_dtype(out, session):
    _rewrite_artifacts_dtype(out, np.complex64, np.complex64, 1 + 0j, 2 + 0j)
    _mutate_manifest(out, lambda m: m.update(dtype_sum="complex64", dtype_wht="complex64"))


def _mut_bool_dtype(out, session):
    _rewrite_artifacts_dtype(out, np.bool_, np.bool_, True, True)
    _mutate_manifest(out, lambda m: m.update(dtype_sum="bool", dtype_wht="bool"))


@pytest.mark.parametrize(
    "mutator",
    [_mut_int64_dtype, _mut_unicode_dtype, _mut_complex_dtype, _mut_bool_dtype],
)
def test_early_preflight_foreign_dtype_clean_refusal(tmp_path, mutator):
    """Matching non-float32 manifest+artifacts refuse at the headless dtype
    contract with a clean reason; the full open path also refuses without any
    np.isfinite/comparison TypeError escaping."""
    out, session = _make_fits_seam_base(tmp_path, ref_hw=(2, 2), shape=(2, 2, 3))
    mutator(out, session)
    s = _resume_requested_stack(out)
    bind_session(s, session)
    ok, detail = s._early_resume_preflight()
    assert ok is False
    assert "dtype" in detail, detail
    assert "configured scientific dtype" in detail, detail
    ok2, reason2 = s._validate_and_open_resume((2, 2, 3))
    assert ok2 is False
    assert "dtype" in reason2, reason2


@pytest.mark.parametrize(
    "mutator",
    [_mut_int64_dtype, _mut_unicode_dtype, _mut_complex_dtype, _mut_bool_dtype],
)
def test_start_processing_foreign_dtype_refused_before_reference(tmp_path, mutator):
    """The real start_processing seam refuses foreign dtypes before
    _get_reference_image, with no exception, no thread, and byte-identical
    sentinel/manifest/SUM/WHT artifacts."""
    _run_seam_assert_refused(tmp_path, mutate=mutator)


def test_valid_float32_checkpoint_still_passes_preflight_and_open(tmp_path):
    """The canonical float32 checkpoint (manifest dtype == runtime dtype) must
    still pass the early preflight and open, then restore state."""
    out, session = _make_fits_seam_base(tmp_path, ref_hw=(2, 2), shape=(2, 2, 3))
    m = json.loads(_manifest_path(out).read_text(encoding="utf-8"))
    assert m["dtype_sum"] == np.dtype(np.float32).name
    assert m["dtype_wht"] == np.dtype(np.float32).name

    s = _resume_requested_stack(out)
    bind_session(s, session)
    ok, ref = s._early_resume_preflight()
    assert ok is True
    assert s._resume_preflight_passed is True
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is True, reason
    assert s._resume_active is True
    assert s.stacked_batches_count == 2
    close_mm(s.cumulative_sum_memmap)
    close_mm(s.cumulative_wht_memmap)


# ---------------------------------------------------------------------------
# 35. P5-FIX: quality reference scale (q_ref) persistence and fail-closed
#     resume validation.
# ---------------------------------------------------------------------------
def _make_quality_weighted_base(tmp_path, q_ref=50.0, shape=(2, 2, 3), n_sources=2):
    """A valid quality-weighted clean checkpoint with a persisted q_ref."""
    out = tmp_path / "out"
    out.mkdir()
    input_dir = out / "input"
    ref_path = _write_fits_reference(input_dir / "reference.fits", 2, 2)
    ref_ident = _stat_identity(ref_path)
    roots = [os.path.normcase(os.path.abspath(str(input_dir)))]
    sources = []
    for i in range(n_sources):
        p = _write_file(input_dir / f"obs_{i:03d}.fits", 100 + i)
        sources.append(_stat_identity(p))
    plan = {"sources": list(sources), "decomposition": [1] * n_sources}
    session = {
        "input_dir": str(input_dir),
        "roots": roots,
        "reference": ref_ident,
        "ref_path": ref_path,
        "sources": sources,
        "plan": plan,
    }
    memdir = out / "memmap_accumulators"
    memdir.mkdir(parents=True, exist_ok=True)
    for name, val in (("cumulative_SUM.npy", 1.0), ("cumulative_WHT.npy", 2.0)):
        mm = np.lib.format.open_memmap(
            memdir / name, mode="w+", dtype=np.float32, shape=shape
        )
        mm[:] = val
        mm.flush()
        close_mm(mm)
    stack = make_resume_stack(out, use_quality_weighting=True)
    bind_session(stack, session)
    stack._quality_reference_scale = q_ref
    stack.memmap_shape = tuple(shape)
    stack._resume_completed_sources = list(sources)
    stack.stacked_batches_count = n_sources
    stack._write_resume_manifest(
        state="clean", completed_sources=list(sources), stacked_batches_count=n_sources
    )
    return out, session


def test_quality_weighted_checkpoint_roundtrip_restores_q_ref(tmp_path):
    out, session = _make_quality_weighted_base(tmp_path, q_ref=50.0)
    s = make_resume_stack(out, use_quality_weighting=True)
    bind_session(s, session)
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is True, reason
    assert s._resume_active is True
    assert s._quality_reference_scale == 50.0
    close_mm(s.cumulative_sum_memmap)
    close_mm(s.cumulative_wht_memmap)


def test_quality_weighted_missing_q_ref_refused_before_mutation(tmp_path):
    out, session = _make_quality_weighted_base(tmp_path, q_ref=50.0)
    mp = _manifest_path(out)
    m = json.loads(mp.read_text(encoding="utf-8"))
    del m["quality_reference_scale"]
    mp.write_text(json.dumps(m), encoding="utf-8")

    sum_path = out / "memmap_accumulators" / "cumulative_SUM.npy"
    wht_path = out / "memmap_accumulators" / "cumulative_WHT.npy"
    sum_before = sum_path.read_bytes()
    wht_before = wht_path.read_bytes()

    s = make_resume_stack(out, use_quality_weighting=True)
    bind_session(s, session)
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is False
    assert "quality reference scale" in reason
    # No artifact mutation: SUM/WHT memmaps are byte-identical.
    assert sum_path.read_bytes() == sum_before
    assert wht_path.read_bytes() == wht_before


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), "NaN", True])
def test_quality_weighted_malformed_q_ref_refused(tmp_path, bad):
    out, session = _make_quality_weighted_base(tmp_path, q_ref=50.0)
    _mutate_manifest(out, lambda m: m.__setitem__("quality_reference_scale", bad))
    s = make_resume_stack(out, use_quality_weighting=True)
    bind_session(s, session)
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is False
    assert "quality reference scale" in reason


def test_quality_weighting_disabled_does_not_require_q_ref(tmp_path):
    """A non-quality-weighted checkpoint is accepted even when the manifest
    carries a stray/malformed quality_reference_scale: the field is ignored
    when ``use_quality_weighting`` is False."""
    out, session = _make_fits_seam_base(tmp_path, ref_hw=(2, 2), shape=(2, 2, 3))
    _mutate_manifest(out, lambda m: m.__setitem__("quality_reference_scale", "garbage"))
    s = _resume_requested_stack(out)
    bind_session(s, session)
    ok, reason = s._validate_and_open_resume((2, 2, 3))
    assert ok is True, reason
    close_mm(s.cumulative_sum_memmap)
    close_mm(s.cumulative_wht_memmap)


# ---------------------------------------------------------------------------
# 36. F4: start_processing capture seam — the prepared immutable reference data
#     is passed to _capture_quality_reference exactly once on a fresh
#     quality-weighted run, before worker launch; a resume skips recomputation.
# ---------------------------------------------------------------------------
def _capture_spy_aligner(ref_data):
    """An aligner stub returning a fixed, identity-checkable reference array."""
    class _Aligner(_SpyReferenceAligner):
        def _get_reference_image(self, folder, files, output_folder):
            self.calls.append(
                {
                    "folder": folder,
                    "files": list(files),
                    "output_folder": output_folder,
                    "reference_image_path": self.reference_image_path,
                }
            )
            return ref_data, fits.Header()

    return _Aligner()


def test_start_processing_quality_weighted_captures_q_ref_once_fresh(tmp_path):
    """F4: a fresh quality-weighted run passes the actual prepared reference
    data to ``_capture_quality_reference`` exactly once, before the worker
    thread is launched."""
    out = tmp_path / "out"
    out.mkdir()
    input_dir = out / "input"
    input_dir.mkdir()
    _write_fits_reference(input_dir / "reference.fits", 4, 5)

    ref_data = np.full((4, 5, 3), 123.0, dtype=np.float32)
    aligner = _capture_spy_aligner(ref_data)
    s = _make_start_processing_stack(
        out, input_dir, aligner=aligner, use_quality_weighting=True
    )

    captured = []
    worker_launched = []
    real_capture = s._capture_quality_reference

    def spy_capture(data):
        captured.append(data)
        assert worker_launched == []  # capture must precede worker launch
        real_capture(data)

    s._capture_quality_reference = spy_capture

    def fake_initialize(self, output_dir, shape_hwc, enable_preview=False):
        self._checkpointing_enabled = False
        return True

    s.initialize = types.MethodType(fake_initialize, s)

    def fake_add_files_to_queue(self, folder):
        self.files_in_queue = 0
        self.use_batch_plan = False
        return 0

    s._add_files_to_queue = types.MethodType(fake_add_files_to_queue, s)

    def fake_worker():
        worker_launched.append(True)

    s._worker = fake_worker

    result = s.start_processing(
        input_dir=str(input_dir),
        output_dir=str(out),
        batch_size=10,
        use_weighting=True,
    )

    assert result is True
    assert len(captured) == 1
    assert captured[0] is ref_data  # the exact prepared immutable reference data
    # q_ref was pinned by the real capture (finite, positive).
    assert s._quality_reference_scale is not None
    assert np.isfinite(s._quality_reference_scale)
    assert s._quality_reference_scale > 0.0
    # The worker thread was launched after capture.
    s.processing_thread.join(timeout=5)
    assert worker_launched == [True]


def test_start_processing_quality_weighted_resume_skips_recomputation(tmp_path):
    """F4: a quality-weighted resume does NOT recompute q_ref (no
    ``_capture_quality_reference`` call); the persisted q_ref is restored."""
    out, session = _make_quality_weighted_base(tmp_path, q_ref=50.0)
    aligner = _capture_spy_aligner(np.full((2, 2, 3), 1.0, dtype=np.float32))
    s = _make_start_processing_stack(
        out,
        Path(session["input_dir"]),
        aligner=aligner,
        use_quality_weighting=True,
        _resume_resolved_reference=session["ref_path"],
    )

    captured = []
    s._capture_quality_reference = lambda data: captured.append(data)

    def fake_initialize(self, output_dir, shape_hwc, enable_preview=False):
        # Simulate the real resume restore: read the persisted q_ref verbatim.
        mp = Path(output_dir) / "memmap_accumulators" / "resume_manifest.json"
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        self._resume_active = True
        self._checkpointing_enabled = False
        self._resume_completed_sources = []
        self._quality_reference_scale = float(manifest["quality_reference_scale"])
        return True

    s.initialize = types.MethodType(fake_initialize, s)

    def fake_add_files_to_queue(self, folder):
        self.files_in_queue = 0
        self.use_batch_plan = False
        return 0

    s._add_files_to_queue = types.MethodType(fake_add_files_to_queue, s)
    s._worker = lambda: None

    result = s.start_processing(
        input_dir=session["input_dir"],
        output_dir=str(out),
        batch_size=10,
        use_weighting=True,
        resume_intent="resume",
    )

    assert result is True
    assert captured == []  # no recomputation on resume
    assert s._quality_reference_scale == 50.0  # restored verbatim


# ---------------------------------------------------------------------------
# 37. F5: quality-weighted fresh-vs-resume continuation parity with a binding
#     min_weight and q_ref.
# ---------------------------------------------------------------------------
def test_quality_weighted_continuation_parity(tmp_path):
    """F5: uninterrupted vs checkpoint+resume agree on final V, WHT and SUM for
    a quality-weighted run with binding min_weight and q_ref; q_ref is restored
    verbatim and a post-resume source whose floor binds is weighted identically.
    """
    shape = (4, 5, 3)
    q_ref = 50.0
    min_weight = 0.5
    snrs = [5.0, 50.0, 10.0]  # relative [0.1, 1.0, 0.2] -> floored [0.5, 1.0, 0.5]

    def quality_weights(snr_list):
        st = make_resume_stack(
            tmp_path,
            use_quality_weighting=True,
            min_weight=min_weight,
            weight_by_stars=False,
        )
        st._quality_reference_scale = q_ref
        return st._calculate_weights([{"snr": s, "stars": 0.0} for s in snr_list])

    weights = quality_weights(snrs)
    assert np.allclose(weights, [0.5, 1.0, 0.5], rtol=1e-6)

    src_files = []
    for i in range(3):
        f = tmp_path / f"src_{i}.fits"
        _write_file(f, i + 1)
        src_files.append(str(f))

    session = build_session(tmp_path, n_sources=0)
    session["sources"] = [_stat_identity(f) for f in src_files]
    session["plan"] = {"sources": session["sources"], "decomposition": [1, 1, 1]}

    def make_batch(i):
        v = np.full(shape, 100.0, dtype=np.float32)
        w = np.full(shape, float(weights[i]), dtype=np.float32)
        return v, w

    def make_qw_stack(rel):
        st = make_resume_stack(
            tmp_path / rel,
            use_quality_weighting=True,
            min_weight=min_weight,
            weight_by_stars=False,
        )
        st._quality_reference_scale = q_ref
        bind_session(st, session)
        return st

    # Uninterrupted run: all three batches.
    full = make_qw_stack("full")
    assert full._initialize_classic_sumw_accumulators(shape) is True
    for i in range(3):
        v, w = make_batch(i)
        _do_batch(full, i, v, w, src_files[i], totexp=2.0)
    full_sum = np.array(full.cumulative_sum_memmap, dtype=np.float32, copy=True)
    full_wht = np.array(full.cumulative_wht_memmap, dtype=np.float32, copy=True)

    # Checkpoint + reopen after two batches, then add the third (floor binds).
    part = make_qw_stack("part")
    assert part._initialize_classic_sumw_accumulators(shape) is True
    for i in (0, 1):
        v, w = make_batch(i)
        _do_batch(part, i, v, w, src_files[i], totexp=2.0)

    reopened = make_qw_stack("part")
    ok, reason = reopened._validate_and_open_resume(shape)
    assert ok is True, reason
    assert reopened._quality_reference_scale == q_ref  # restored verbatim
    # The post-resume source's floor binds against the restored q_ref.
    w3 = reopened._calculate_weights([{"snr": snrs[2], "stars": 0.0}])
    assert np.allclose(w3, [min_weight], rtol=1e-6)
    reopened.stacked_batches_count = reopened._resume_pending_count
    v3, w3arr = make_batch(2)
    _do_batch(reopened, 2, v3, w3arr, src_files[2], totexp=2.0)

    reopened_sum = np.array(
        reopened.cumulative_sum_memmap, dtype=np.float32, copy=True
    )
    reopened_wht = np.array(
        reopened.cumulative_wht_memmap, dtype=np.float32, copy=True
    )

    assert np.allclose(reopened_sum, full_sum, rtol=1e-5, atol=1e-5)
    assert np.allclose(reopened_wht, full_wht, rtol=1e-5, atol=1e-5)

    with np.errstate(divide="ignore", invalid="ignore"):
        full_final = full_sum / full_wht
        reopened_final = reopened_sum / reopened_wht
    assert np.allclose(reopened_final, full_final, rtol=1e-4, atol=1e-4)

    for mm in (
        full.cumulative_sum_memmap,
        full.cumulative_wht_memmap,
        part.cumulative_sum_memmap,
        part.cumulative_wht_memmap,
        reopened.cumulative_sum_memmap,
        reopened.cumulative_wht_memmap,
    ):
        close_mm(mm)


# ---------------------------------------------------------------------------
# 26. Schema-v2 manifest content / digest / config payload
# ---------------------------------------------------------------------------
def _read_manifest(out_dir):
    return json.loads(
        (Path(out_dir) / "memmap_accumulators" / "resume_manifest.json").read_text()
    )


def _write_v2_checkpoint(out_dir, shape, count, ledger, session=None):
    """Same as ``write_valid_checkpoint`` but asserts the v2 manifest shape."""
    stack = write_valid_checkpoint(
        out_dir, shape, count, ledger, session=session
    )
    manifest = _read_manifest(out_dir)
    assert manifest["schema_version"] == 2
    assert (Path(out_dir) / "run_config.cfg").is_file()
    return stack, manifest


def test_manifest_v2_content_digest_and_payload(tmp_path):
    out = tmp_path
    shape = (2, 2, 3)
    session = build_session(out, n_sources=2)
    stack, manifest = _write_v2_checkpoint(
        out, shape, count=2, ledger=session["sources"], session=session
    )

    # Required v2 keys present with the right shape/types.
    assert manifest["schema_version"] == 2
    assert manifest["mode"] == "classic_sumw"
    assert isinstance(manifest["scientific_config"], dict)
    assert isinstance(manifest["fingerprint"], str) and len(manifest["fingerprint"]) == 64
    assert isinstance(manifest["run_config_digest"], str) and len(manifest["run_config_digest"]) == 64

    # fingerprint is the authoritative classic hash of the current engine state.
    assert manifest["fingerprint"] == stack._scientific_fingerprint()

    # scientific_config is the canonical classic payload (percent + list winsor).
    sci = manifest["scientific_config"]
    assert sci["master_tile_crop_percent"] == 18.0
    assert sci["winsor_limits"] == [0.05, 0.05]

    # run_config_digest matches the exact canonical run_config.cfg model on disk.
    report = rc.read_cfg(str(Path(out) / "run_config.cfg"))
    assert report.config.full_digest() == manifest["run_config_digest"]
    assert report.config.scientific == sci
    # recomputed fingerprint from the stored payload agrees with the stored hash.
    assert rc.classic_fingerprint(
        rc.RunConfig.from_sections(scientific=sci)
    ) == manifest["fingerprint"]


def test_manifest_v2_all_existing_hsi_fields_unchanged(tmp_path):
    out = tmp_path
    shape = (2, 2, 3)
    session = build_session(out, n_sources=1)
    _, manifest = _write_v2_checkpoint(
        out, shape, count=1, ledger=session["sources"], session=session
    )
    for key in (
        "state", "mode", "semantics", "shape", "dtype_sum", "dtype_wht",
        "quality_reference_scale", "stacked_batches_count",
        "images_in_cumulative_stack", "total_exposure_seconds",
        "exposure_unknown_count", "exposure_min", "exposure_max",
        "cumulative_header", "session", "completed_sources",
    ):
        assert key in manifest, key


# ---------------------------------------------------------------------------
# 27. V2 tamper matrix: fail closed before memmap open, no mutation
# ---------------------------------------------------------------------------
def _assert_refused_no_mutation(tmp_path, mutate, needle):
    out = tmp_path
    shape = (2, 2, 3)
    session = build_session(out, n_sources=2)
    write_valid_checkpoint(
        out, shape, count=2, ledger=session["sources"], session=session
    )
    manifest_path = Path(out) / "memmap_accumulators" / "resume_manifest.json"
    run_cfg_path = Path(out) / "run_config.cfg"
    sum_path = Path(out) / "memmap_accumulators" / "cumulative_SUM.npy"
    wht_path = Path(out) / "memmap_accumulators" / "cumulative_WHT.npy"

    mutate(manifest_path, run_cfg_path)

    # Post-tamper snapshot: the read-only validation must not modify any
    # artifact further (SUM/WHT are never opened, manifest/run_config.cfg stay
    # exactly as the tamper left them).
    def _snapshot():
        return {
            "sum": sum_path.read_bytes(),
            "wht": wht_path.read_bytes(),
            "manifest": manifest_path.read_bytes(),
            "run_cfg": run_cfg_path.read_bytes() if run_cfg_path.exists() else None,
        }

    snapshot = _snapshot()

    s = make_resume_stack(out)
    bind_session(s, session)
    ok, reason, _ = s._validate_resume_headless()
    assert ok is False
    assert needle in reason

    assert _snapshot() == snapshot


def test_v2_tamper_scientific_config_payload(tmp_path):
    def mutate(mp, rp):
        m = json.loads(mp.read_text())
        m["scientific_config"]["kappa"] = 99.0
        mp.write_text(json.dumps(m), encoding="utf-8")

    _assert_refused_no_mutation(tmp_path, mutate, "does not match its fingerprint")


def test_v2_tamper_fingerprint(tmp_path):
    def mutate(mp, rp):
        m = json.loads(mp.read_text())
        m["fingerprint"] = "0" * 64
        mp.write_text(json.dumps(m), encoding="utf-8")

    _assert_refused_no_mutation(tmp_path, mutate, "configuration mismatch")


def test_v2_tamper_run_config_cfg_content(tmp_path):
    def mutate(mp, rp):
        data = json.loads(rp.read_text())
        data["scientific_config"]["kappa"] = 9.9
        rp.write_text(
            json.dumps(data, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    _assert_refused_no_mutation(tmp_path, mutate, "does not match its recorded digest")


def test_v2_tamper_run_config_digest_field(tmp_path):
    def mutate(mp, rp):
        m = json.loads(mp.read_text())
        m["run_config_digest"] = "f" * 64
        mp.write_text(json.dumps(m), encoding="utf-8")

    _assert_refused_no_mutation(tmp_path, mutate, "does not match its recorded digest")


def test_v2_corrupt_run_config_cfg(tmp_path):
    def mutate(mp, rp):
        rp.write_text("{ not valid json", encoding="utf-8")

    _assert_refused_no_mutation(tmp_path, mutate, "run_config.cfg invalid")


def test_v2_missing_run_config_cfg(tmp_path):
    def mutate(mp, rp):
        rp.unlink()

    _assert_refused_no_mutation(tmp_path, mutate, "run_config.cfg missing")


# ---------------------------------------------------------------------------
# 28. Schema-v1 compatible resume regression + mismatch refusal
# ---------------------------------------------------------------------------
def _write_v1_manifest(out_dir, fingerprint, session=None, shape=(2, 2, 3)):
    memdir = Path(out_dir) / "memmap_accumulators"
    memdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "state": "clean",
        "mode": "classic_sumw",
        "fingerprint": fingerprint,
        "shape": list(shape),
        "dtype_sum": "float32",
        "dtype_wht": "float32",
        "stacked_batches_count": 0,
        "images_in_cumulative_stack": 0,
        "total_exposure_seconds": 0.0,
        "exposure_unknown_count": 0,
        "exposure_min": None,
        "exposure_max": None,
        "cumulative_header": {},
        "quality_reference_scale": None,
        "completed_sources": [],
        "session": {
            "input_roots": session["roots"] if session else [],
            "reference": session["reference"] if session else None,
            "plan": session["plan"] if session else {"sources": [], "decomposition": []},
        },
    }
    (memdir / "resume_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_v1_memmaps(out_dir, shape):
    memdir = Path(out_dir) / "memmap_accumulators"
    memdir.mkdir(parents=True, exist_ok=True)
    sum_mm = np.lib.format.open_memmap(
        memdir / "cumulative_SUM.npy", mode="w+", dtype=np.float32, shape=shape
    )
    wht_mm = np.lib.format.open_memmap(
        memdir / "cumulative_WHT.npy", mode="w+", dtype=np.float32, shape=shape
    )
    sum_mm[:] = 1.0
    wht_mm[:] = 1.0
    sum_mm.flush()
    wht_mm.flush()
    close_mm(sum_mm)
    close_mm(wht_mm)


def test_v1_manifest_resume_compatible(tmp_path):
    """A schema-v1 manifest resumes under its exact fingerprint contract when
    the current effective fingerprint matches (no run_config.cfg required)."""
    out = tmp_path
    shape = (2, 2, 3)
    session = build_session(out, n_sources=0)
    s0 = make_resume_stack(out)
    bind_session(s0, session)
    _write_v1_memmaps(out, shape)
    _write_v1_manifest(out, s0._scientific_fingerprint(), session=session, shape=shape)

    s = make_resume_stack(out)
    bind_session(s, session)
    ok, reason = s._validate_and_open_resume(shape)
    assert ok is True, reason
    close_mm(s.cumulative_sum_memmap)
    close_mm(s.cumulative_wht_memmap)


def test_v1_manifest_mismatch_refused(tmp_path):
    """A schema-v1 manifest with a mismatching current fingerprint is refused
    exactly like the legacy behavior; never reconstructed from the hash."""
    out = tmp_path
    shape = (2, 2, 3)
    session = build_session(out, n_sources=0)
    s0 = make_resume_stack(out)
    bind_session(s0, session)
    _write_v1_memmaps(out, shape)
    _write_v1_manifest(out, s0._scientific_fingerprint(), session=session, shape=shape)

    s = make_resume_stack(out, stacking_mode="mean")
    bind_session(s, session)
    ok, reason = s._validate_and_open_resume(shape)
    assert ok is False
    assert "configuration mismatch" in reason


def test_v1_manifest_not_upgraded_on_disk(tmp_path):
    """Reading/resuming a v1 manifest must never rewrite it to v2 on disk."""
    out = tmp_path
    shape = (2, 2, 3)
    session = build_session(out, n_sources=0)
    s0 = make_resume_stack(out)
    bind_session(s0, session)
    _write_v1_memmaps(out, shape)
    _write_v1_manifest(out, s0._scientific_fingerprint(), session=session, shape=shape)
    manifest_path = Path(out) / "memmap_accumulators" / "resume_manifest.json"
    before = manifest_path.read_bytes()

    s = make_resume_stack(out)
    bind_session(s, session)
    ok, _ = s._validate_and_open_resume(shape)
    assert ok is True
    close_mm(s.cumulative_sum_memmap)
    close_mm(s.cumulative_wht_memmap)
    assert manifest_path.read_bytes() == before
    assert not (Path(out) / "run_config.cfg").exists()


# ---------------------------------------------------------------------------
# 29. CFG-alone: run_config.cfg is configuration evidence, never a checkpoint
# ---------------------------------------------------------------------------
def test_cfg_alone_fresh_refuses(tmp_path):
    """run_config.cfg alone is recognized prior-run state: a fresh run refuses."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "run_config.cfg").write_text("{}", encoding="utf-8")

    s = make_resume_stack(str(out))
    assert s._resume_artifacts_present(str(out)) is True
    assert s._can_resume(Path(str(out))) is True


def test_cfg_alone_resume_refuses_missing_checkpoint(tmp_path):
    """run_config.cfg alone cannot authorize a resume: the manifest/accumulators
    are required and their absence fails closed."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "run_config.cfg").write_text("{}", encoding="utf-8")

    s = make_resume_stack(str(out))
    s._resume_requested = True
    ok, reason, _ = s._validate_resume_headless()
    assert ok is False
    assert "manifest" in reason


# ---------------------------------------------------------------------------
# 30. RSM2-02B1 R1 corrections
# ---------------------------------------------------------------------------
class _StartRefAligner:
    """Returns a fixed HWC reference frame for ``start_processing`` shape prep."""

    def __init__(self, shape=(4, 5, 3)):
        self.stop_processing = False
        self.reference_image_path = None
        self._shape = shape

    def _get_reference_image(self, folder, files, output_folder):
        return (np.zeros(self._shape, dtype=np.float32), fits.Header())


def _make_start_stack(out_dir, input_dir):
    """Bare stacker with enough state to drive ``start_processing``'s fresh
    classic path to a real manifest write (fake aligner + fake worker)."""
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.update_progress = lambda *a, **k: None
    o.logger = types.SimpleNamespace(
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    o.processing_active = False
    o.stop_processing = False
    o.user_requested_stop = False
    o.startup_refusal = None
    o.aligner = _StartRefAligner()
    o.autotuner = None
    o.current_folder = str(input_dir)
    o.output_folder = str(out_dir)
    o.is_mosaic_run = False
    o.drizzle_active_session = False
    o.reproject_between_batches = False
    o.reproject_coadd_final = False
    o.freeze_reference_wcs = False
    o.reproject_output_wcs = None
    o.master_sum = None
    o.master_coverage = None
    o.reference_pixel_scale_arcsec = None
    o.reference_wcs_object = None
    o.fixed_output_wcs = None
    o.fixed_output_shape = None
    o.input_reference_shape_hw = None
    o.keep_input_size_for_reproject = False
    o._has_stack_plan = False
    o.cumulative_sum_memmap = None
    o.cumulative_wht_memmap = None
    o.cumulative_wht_path = None
    o.sum_memmap_path = None
    o.wht_memmap_path = None
    o.memmap_shape = None
    o.memmap_dtype_sum = np.float32
    o.memmap_dtype_wht = np.float32
    o._resume_active = False
    o._resume_completed_sources = []
    o._checkpointing_enabled = False
    o._norm_reference = None
    o.additional_folders = []
    o.folders_lock = threading.Lock()
    o.processed_files = set()
    o.queue = Queue()
    o.files_in_queue = 0
    o.queue_prepared = False
    o._resume_requested = False
    o.resume_source = None
    o._autotuner_started_this_attempt = False
    o._attempt_preexisting_state = None
    o.batch_count_path = None
    o.drizzle_mode = "Final"
    o.use_drizzle = False
    o.reference_shape = None
    o._resume_resolved_reference = None
    o._resume_input_roots = None
    o._resume_reference_identity = None
    o._resume_plan = None
    o.images_in_cumulative_stack = 0
    o.total_exposure_seconds = 0.0
    o.current_stack_header = None
    o.stacked_batches_count = 0
    o._exposure_unknown_count = 0
    o._exposure_min = None
    o._exposure_max = None
    return o


def test_repeated_start_binds_fresh_canonical_config(tmp_path):
    """A repeated Start with changed classic settings and a new output folder
    writes a CFG + manifest reflecting the second session, never the stale
    cached canonical config of the first (RSM2-02B1 R1 correction #1)."""
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "frame.fits").write_bytes(b"\x00" * 16)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    s = _make_start_stack(out1, inp)

    def fake_initialize(self, output_dir, shape_hwc, enable_preview=False):
        return self._initialize_classic_sumw_accumulators(tuple(shape_hwc))

    s.initialize = types.MethodType(fake_initialize, s)
    s._add_files_to_queue = lambda folder: 0
    s._checkpoint_preflight = lambda: True
    s._worker = lambda: None

    # Session 1: kappa=2.5
    assert s.start_processing(
        input_dir=str(inp), output_dir=str(out1), kappa=2.5,
        batch_size=10, resume_intent="fresh",
    ) is True
    rep1 = rc.read_cfg(str(out1 / "run_config.cfg"))
    assert rep1.config.scientific["kappa"] == 2.5
    assert _read_manifest(out1)["scientific_config"]["kappa"] == 2.5

    # Session 2 (new Start): changed classic settings + new output folder.
    s.processing_active = False
    s.stop_processing = False
    s.output_folder = str(out2)
    assert s.start_processing(
        input_dir=str(inp), output_dir=str(out2), kappa=3.0,
        batch_size=10, resume_intent="fresh",
    ) is True

    rep2 = rc.read_cfg(str(out2 / "run_config.cfg"))
    assert rep2.config.scientific["kappa"] == 3.0
    manifest2 = _read_manifest(out2)
    assert manifest2["scientific_config"]["kappa"] == 3.0


def test_malformed_effective_field_refuses_persistence(tmp_path):
    """A malformed/uncoercible effective classic field refuses checkpoint
    persistence (fail closed) and never writes a self-inconsistent manifest/CFG
    (RSM2-02B1 R1 correction #2)."""
    out = tmp_path
    shape = (2, 2, 3)
    session = build_session(out, n_sources=2)
    s = make_resume_stack(out)
    bind_session(s, session)
    s.memmap_shape = shape
    s.kappa = "not-a-float"

    with pytest.raises(rc.ConfigError):
        s._write_resume_manifest(
            state="clean", completed_sources=[], stacked_batches_count=0
        )

    assert not (out / "memmap_accumulators" / "resume_manifest.json").exists()
    assert not (out / "run_config.cfg").exists()


def test_canonical_engine_fingerprint_divergence_refuses(tmp_path):
    """A canonical/engine classic-fingerprint divergence (a coercible value that
    changes the canonical representation) refuses persistence before any write
    (RSM2-02B1 R1 correction #2 enforcement)."""
    out = tmp_path
    shape = (2, 2, 3)
    session = build_session(out, n_sources=2)
    s = make_resume_stack(out)
    bind_session(s, session)
    s.memmap_shape = shape
    # neighborhood_size is int-kind: 5.0 coerces to 5 canonically but hashes as
    # 5.0 in the engine fingerprint, so the two fingerprints must diverge.
    s.neighborhood_size = 5.0

    with pytest.raises(_ResumeCheckpointError):
        s._write_resume_manifest(
            state="clean", completed_sources=[], stacked_batches_count=0
        )

    assert not (out / "memmap_accumulators" / "resume_manifest.json").exists()
    assert not (out / "run_config.cfg").exists()


def test_v1_opened_session_keeps_v1_writes_no_cfg(tmp_path):
    """A session opened from a schema-v1 manifest keeps v1 write semantics on a
    subsequent manifest write: schema stays 1 and no run_config.cfg appears
    (RSM2-02B1 R1 correction #3)."""
    out = tmp_path
    shape = (2, 2, 3)
    session = build_session(out, n_sources=0)
    s0 = make_resume_stack(out)
    bind_session(s0, session)
    _write_v1_memmaps(out, shape)
    _write_v1_manifest(out, s0._scientific_fingerprint(), session=session, shape=shape)

    s = make_resume_stack(out)
    bind_session(s, session)
    ok, _ = s._validate_and_open_resume(shape)
    assert ok is True
    close_mm(s.cumulative_sum_memmap)
    close_mm(s.cumulative_wht_memmap)

    # A subsequent dirty/clean manifest write must preserve v1 semantics.
    s._write_resume_manifest(state="dirty")

    manifest = _read_manifest(out)
    assert manifest["schema_version"] == 1
    assert "scientific_config" not in manifest
    assert "run_config_digest" not in manifest
    assert not (out / "run_config.cfg").exists()


def test_preexisting_run_cfg_survives_failed_fresh_cleanup(tmp_path):
    """A failed fresh persistence removes only attempt-created artifacts; a
    pre-existing run_config.cfg (snapshot-captured) is left untouched
    (RSM2-02B1 R1 correction #4)."""
    out = tmp_path
    pre_cfg = out / "run_config.cfg"
    pre_cfg.write_text('{"schema_version":2}\n', encoding="utf-8")

    s = make_resume_stack(out)
    s._autotuner_started_this_attempt = False
    s._attempt_preexisting_state = s._snapshot_existing_state()
    s._resume_active = False

    memdir = out / "memmap_accumulators"
    memdir.mkdir()
    (memdir / "resume_manifest.json").write_text("{}", encoding="utf-8")
    (memdir / "resume_manifest.json.tmp").write_text("{}", encoding="utf-8")

    s._remove_attempt_created_state()

    assert not (memdir / "resume_manifest.json").exists()
    assert not (memdir / "resume_manifest.json.tmp").exists()
    assert pre_cfg.read_bytes() == b'{"schema_version":2}\n'
    assert not memdir.exists()
