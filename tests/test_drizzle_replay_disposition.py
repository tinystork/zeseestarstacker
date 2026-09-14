"""ZSSS-DRIZZLE-REJECTED-SOURCE REWORK 3 — canonical replay disposition.

Proves the native Drizzle Resume contract for observations that were accepted
but not durably committed and are replayed from the canonical ``stacked``
directory:

* exactly ONE canonical stacked location — replay + re-move is idempotent
  (never ``stacked/stacked``);
* the durable disposition identity is the AUTHORITATIVE session-plan
  identity, never the physical replay path;
* a rejection never requires a late ``stat()`` of a source that may already
  have been moved or become unavailable;
* a nested non-canonical stacked location is refused with a specific
  diagnostic.

Witnesses 1-6 + the current real Windows frontier (generation 473,
frame_count 4640, plan_cursor 4649, 9 rejected dispositions).
"""

import json
import os
import threading
from pathlib import Path
from queue import Queue

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from seestar.core.drizzle_checkpoint import (
    CHECKPOINT_DIRNAME,
    MANIFEST_FILENAME,
    DrizzleCheckpointError,
    DrizzleCheckpointWriter,
    SafeStackedSourceResolver,
    build_drizzle_canonical_config,
    read_drizzle_checkpoint,
)
from seestar.core.drizzle_core import DrizzleAccumulator
import seestar.queuep.queue_manager as queue_manager_module
from seestar.queuep.queue_manager import SeestarQueuedStacker

SHAPE = (8, 8)


def _wcs():
    w = WCS(naxis=2)
    w.wcs.crpix = [4.5, 4.5]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = [-0.001, 0.001]
    w.array_shape = SHAPE
    return w


def _reference_geometry():
    from seestar.core.drizzle_checkpoint import serialize_input_reference_geometry

    return serialize_input_reference_geometry(_wcs(), SHAPE, None)


def _identity(path):
    st = os.stat(path)
    return {
        "path": os.path.normcase(os.path.abspath(str(path))),
        "name": os.path.basename(str(path)),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _frame(index, exptime=1.0):
    yy, xx = np.indices(SHAPE, dtype=np.float64)
    data = (index * 3.0 + xx * 0.25 + yy * 0.5).astype(np.float32)
    weight = np.full(SHAPE, 0.7, dtype=np.float32)
    pixmap = np.dstack((xx + index * 0.07, yy - index * 0.04))
    in_grid = np.ones(SHAPE, dtype=bool)
    return data, weight, pixmap, in_grid, exptime


def _add(accs, frame):
    data, weight, pixmap, in_grid, exptime = frame
    for acc in accs:
        acc.add(
            data, weight, pixmap, exptime=exptime,
            in_units="counts", in_grid_mask=in_grid,
        )


def _configure(qm, output, inputs, kernel="square", group_size=1):
    qm.output_folder = str(output)
    qm._resume_input_roots = [str(inputs)]
    qm.drizzle_active_session = True
    qm.is_mosaic_run = False
    qm.reproject_between_batches = False
    qm.reproject_coadd_final = False
    qm.move_stacked = False
    qm.stacked_subdir_name = "stacked"
    qm.weighting_method = "none"
    qm.use_quality_weighting = False
    qm.weight_by_snr = True
    qm.weight_by_stars = True
    qm.snr_exponent = 1.0
    qm.stars_exponent = 0.5
    qm.min_weight = 0.01
    qm.correct_hot_pixels = True
    qm.hot_pixel_threshold = 3.0
    qm.neighborhood_size = 5
    qm.bayer_pattern = "GRBG"
    qm.drizzle_scale = 1.0
    qm.reference_wcs_object = _wcs()
    qm.drizzle_output_wcs = _wcs()
    qm.drizzle_output_shape_hw = SHAPE
    qm.input_reference_shape_hw = SHAPE
    qm.drizzle_kernel = kernel
    qm.drizzle_pixfrac = 1.0
    qm.drizzle_wht_threshold = 0.0
    qm.drizzle_wht_threshold_effective = 0.0
    qm.drizzle_fillval = "0.0"
    qm.drizzle_group_size = group_size
    qm.drizzle_processing_policy = "incremental"
    qm.preview_callback = None
    qm.update_progress = lambda *args, **kwargs: None
    qm.processing_error = None
    qm._drizzle_checkpoint_enabled = True
    qm._drizzle_frame_count = 0
    qm._drizzle_group_index = 0
    qm.stacked_batches_count = 0
    qm.total_exposure_seconds = 0.0
    qm._exposure_unknown_count = 0
    qm._exposure_min = None
    qm._exposure_max = None
    qm.failed_align_count = 0
    qm.failed_stack_count = 0
    qm.aligned_files_count = 0
    qm.processed_files_count = 0
    qm.processed_files = set()
    qm.warned_unaligned_source_folders = set()
    qm._drizzle_checkpoint_writer = None
    qm._drizzle_checkpoint_plan = None
    qm._drizzle_completed_sources = []
    qm._drizzle_plan_cursor = 0
    qm._drizzle_rejected_sources = []
    qm._drizzle_plan_path_map = {}
    qm._drizzle_checkpoint_last_committed_frames = 0
    qm._drizzle_resume_result = None
    qm._drizzle_resume_continuation = None
    qm._resume_requested = False
    qm._resume_active = False
    qm._resume_plan = None
    qm._resume_completed_sources = []
    qm._resume_reference_identity = None
    qm._normalize_effective_drizzle_config()


def _accept(qm, path, index):
    data, weight, pixmap, in_grid, exptime = _frame(index)
    ok = qm._add_frame_to_drizzle_accumulators(
        np.stack([data] * 3, axis=-1).astype(np.float32),
        fits.Header([("EXPTIME", 1.0)]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        weight,
        native_wcs=_wcs(),
    )
    assert ok is True
    qm._drizzle_group_tick()
    qm._admit_exposure(1.0, 0, 1.0, 1.0)
    qm.stacked_batches_count += 1
    qm._drizzle_checkpoint_after_frame(str(path))


def _reference_run_qm(tmp_path, accept_indices, n_sources):
    """Golden science: a second fresh QM run accepting exactly the given
    plan indices through the SAME production deposition seam."""
    tag = abs(hash(tuple(accept_indices))) % 10**9
    output = tmp_path / f"ref_out_{tag}"
    inputs = tmp_path / f"ref_inputs_{tag}"
    output.mkdir(exist_ok=True)
    inputs.mkdir(exist_ok=True)
    all_paths = [inputs / f"src_{i}.fit" for i in range(n_sources)]
    for i, p in enumerate(all_paths):
        fits.PrimaryHDU(np.full(SHAPE, i + 1, dtype=np.uint16)).writeto(p)
    ref_path = inputs / "reference.fit"
    fits.PrimaryHDU(np.full(SHAPE, 7, dtype=np.uint16)).writeto(ref_path)
    ref_paths = [all_paths[i] for i in accept_indices]
    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, kernel="square", group_size=1)
    idents = [_identity(p) for p in ref_paths]
    qm.drizzle_accumulators = [
        DrizzleAccumulator(SHAPE, kernel="square", pixfrac=1.0) for _ in range(3)
    ]
    qm._drizzle_checkpoint_plan = {
        "sources": idents,
        "decomposition": [len(idents)],
    }
    if hasattr(qm, "_drizzle_rebuild_plan_path_map"):
        qm._drizzle_rebuild_plan_path_map(idents, [str(p) for p in ref_paths])
    else:
        qm._drizzle_plan_path_map = {
            os.path.normcase(os.path.abspath(str(p))): ident
            for p, ident in zip(ref_paths, idents)
        }
    qm._resume_reference_identity = _identity(ref_path)
    cfg = build_drizzle_canonical_config(
        qm, product_version=qm._canonical_product_version()
    )
    qm._drizzle_checkpoint_writer = DrizzleCheckpointWriter(
        str(output), qm._canonical_product_version(), cfg, _wcs(), SHAPE
    )
    for plan_index in accept_indices:
        _accept(qm, all_paths[plan_index], plan_index)
    return qm.drizzle_accumulators


def _build_checkpoint(tmp_path, n_sources, committed_indices, rejected_indices,
                      generation=1, remaining_layout=None):
    """Build a real checkpoint with ``committed_indices`` accepted and
    ``rejected_indices`` rejected (both plan-ordered dispositions).

    ``remaining_layout`` maps plan index -> physical location
    ("original" | "stacked" | "unaligned") for sources beyond the cursor;
    defaults to original for all."""
    inputs = tmp_path / "inputs"
    out = tmp_path / "out"
    inputs.mkdir()
    out.mkdir()
    paths = []
    for i in range(n_sources):
        p = inputs / f"src_{i}.fit"
        fits.PrimaryHDU(np.full(SHAPE, i + 1, dtype=np.uint16)).writeto(p)
        paths.append(p)
    idents = [_identity(p) for p in paths]
    reference_path = inputs / "reference.fit"
    fits.PrimaryHDU(np.full(SHAPE, 7, dtype=np.uint16)).writeto(reference_path)
    reference_ident = _identity(reference_path)

    cursor = max(max(committed_indices, default=-1), max(rejected_indices, default=-1)) + 1
    committed_sorted = sorted(committed_indices)
    rejected_sorted = sorted(rejected_indices)
    # Plan-ordered disposition interleaving over plan[:cursor].
    completed = [idents[i] for i in committed_sorted]
    rejected = [idents[i] for i in rejected_sorted]
    for i in range(cursor):
        assert i in committed_indices or i in rejected_indices, (
            f"plan index {i} has no disposition in the fixture"
        )
    frame_count = len(committed_sorted)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, out, inputs, kernel="square", group_size=1)
    qm.drizzle_accumulators = [
        DrizzleAccumulator(SHAPE, kernel="square", pixfrac=1.0) for _ in range(3)
    ]
    qm._drizzle_checkpoint_plan = {
        "sources": idents,
        "decomposition": [n_sources],
    }
    if hasattr(qm, "_drizzle_rebuild_plan_path_map"):
        qm._drizzle_rebuild_plan_path_map(idents, [str(p) for p in paths])
    else:  # pre-fix tree: build the equivalent map directly
        qm._drizzle_plan_path_map = {
            os.path.normcase(os.path.abspath(str(p))): ident
            for p, ident in zip(paths, idents)
        }
    qm._resume_reference_identity = reference_ident
    cfg = build_drizzle_canonical_config(
        qm, product_version=qm._canonical_product_version()
    )
    writer = DrizzleCheckpointWriter(
        str(out), qm._canonical_product_version(), cfg, _wcs(), SHAPE
    )
    qm._drizzle_checkpoint_writer = writer
    # Seed science: exactly ONE frame (plan index 0) deposited through the
    # SAME production seam the resumed run uses (so witness-4 science
    # comparisons are bit-truthful).  The fixture requires plan index 0 to
    # be a committed disposition.
    assert 0 in committed_indices, "fixture requires plan index 0 committed"
    _accept(qm, paths[0], 0)
    # Forge the requested frontier: generation, counters, disposition
    # partition (new-format fields).
    ckpt_dir = out / CHECKPOINT_DIRNAME
    manifest = json.loads((ckpt_dir / MANIFEST_FILENAME).read_text())
    for ch in manifest["channels"]:
        for key in ("out_img", "out_wht"):
            old = ch[key]["file"]
            new = old.replace(f"gen-{1:08d}-", f"gen-{generation:08d}-")
            (ckpt_dir / old).rename(ckpt_dir / new)
            ch[key]["file"] = new
    manifest["generation"] = generation
    manifest["frame_count"] = frame_count
    manifest["stacked_batches_count"] = frame_count
    manifest["total_exposure_seconds"] = float(frame_count)
    manifest["exposure_unknown_count"] = 0
    manifest["exposure_min"] = 1.0 if frame_count else None
    manifest["exposure_max"] = 1.0 if frame_count else None
    manifest["session"]["plan"]["sources"] = idents
    manifest["session"]["plan"]["decomposition"] = [n_sources]
    manifest["completed_sources"] = completed
    if rejected:
        manifest["rejected_sources"] = rejected
        manifest["plan_cursor"] = cursor
    (ckpt_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    # Physical layout for the remaining sources (beyond the cursor).
    layout = remaining_layout or {}
    stacked = None
    unaligned = None
    for i in range(cursor, n_sources):
        where = layout.get(i, "original")
        if where == "stacked":
            if stacked is None:
                stacked = inputs / "stacked"
                stacked.mkdir()
            paths[i].rename(stacked / paths[i].name)
        elif where == "unaligned":
            if unaligned is None:
                unaligned = inputs / "unaligned_by_stacker"
                unaligned.mkdir()
            paths[i].rename(unaligned / paths[i].name)
    for i in rejected_sorted:
        if unaligned is None:
            unaligned = inputs / "unaligned_by_stacker"
            unaligned.mkdir()
        if paths[i].exists():
            paths[i].rename(unaligned / paths[i].name)
    for i in committed_sorted:
        if stacked is None:
            stacked = inputs / "stacked"
            stacked.mkdir()
        if paths[i].exists():
            paths[i].rename(stacked / paths[i].name)

    return inputs, out, paths, idents, reference_ident, cursor


def _headless_resume(output, inputs, move_stacked=True):
    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, kernel="square", group_size=1)
    qm.move_stacked = move_stacked
    qm._resume_requested = True
    ok, result, ref = qm._validate_drizzle_resume_headless()
    assert ok is True
    qm._drizzle_resume_result = result
    qm._restore_drizzle_checkpoint_runtime(result)
    qm.queue = Queue()
    for p in result.resolved_remaining_paths:
        qm.queue.put(p)
    assert qm._init_drizzle_checkpoint() is True
    return qm, result


# ---------------------------------------------------------------------------
# Witness 1 — replay accepted from stacked is idempotent
# ---------------------------------------------------------------------------

def test_witness1_replay_accepted_from_stacked_idempotent(tmp_path):
    inputs, out, paths, idents, reference_ident, cursor = _build_checkpoint(
        tmp_path, 5, committed_indices=[0], rejected_indices=[],
        remaining_layout={1: "stacked"},
    )
    qm, result = _headless_resume(out, inputs)
    stacked_path = inputs / "stacked" / "src_1.fit"
    assert stacked_path.exists()
    assert result.resolved_remaining_paths[0] == str(stacked_path)

    _accept(qm, stacked_path, 1)
    qm.move_stacked = True
    qm._move_to_stacked([str(stacked_path)])   # must be a NO-OP
    assert stacked_path.exists()
    assert not (inputs / "stacked" / "stacked").exists()

    manifest = json.loads((out / CHECKPOINT_DIRNAME / MANIFEST_FILENAME).read_text())
    # The durable ledger identity is the AUTHORITATIVE plan identity (the
    # original canonical path), never the stacked physical path.
    new_entry = manifest["completed_sources"][1]
    assert new_entry["path"] == idents[1]["path"]
    assert new_entry["name"] == idents[1]["name"]
    assert new_entry["size"] == idents[1]["size"]
    assert new_entry["mtime_ns"] == idents[1]["mtime_ns"]

    qm._drizzle_checkpoint_force_flush()
    result4 = read_drizzle_checkpoint(str(out), resolver=SafeStackedSourceResolver("stacked"))
    assert result4.completed_sources[1]["path"] == idents[1]["path"]
    assert result4.next_source_index == 2


# ---------------------------------------------------------------------------
# Witness 2 — replayed accepted sources then a rejection
# ---------------------------------------------------------------------------

def test_witness2_replayed_accepted_then_rejection(tmp_path):
    inputs, out, paths, idents, reference_ident, cursor = _build_checkpoint(
        tmp_path, 6, committed_indices=[0], rejected_indices=[],
        remaining_layout={1: "stacked", 2: "stacked", 3: "stacked", 4: "stacked"},
    )
    qm, result = _headless_resume(out, inputs)
    qm.move_stacked = True
    for i in (1, 2, 3, 4):
        stacked_path = inputs / "stacked" / f"src_{i}.fit"
        _accept(qm, stacked_path, i)
        qm._move_to_stacked([str(stacked_path)])
    assert not (inputs / "stacked" / "stacked").exists()

    # E at original -> rejection.
    qm._drizzle_checkpoint_after_rejection(str(paths[5]))
    qm._move_to_unaligned(str(paths[5]))
    manifest = json.loads((out / CHECKPOINT_DIRNAME / MANIFEST_FILENAME).read_text())
    assert manifest["plan_cursor"] == 6
    assert [x["path"] for x in manifest["completed_sources"]] == [
        idents[i]["path"] for i in (0, 1, 2, 3, 4)
    ]
    assert [x["path"] for x in manifest["rejected_sources"]] == [idents[5]["path"]]
    assert manifest["frame_count"] == 5

    result4 = read_drizzle_checkpoint(str(out), resolver=SafeStackedSourceResolver("stacked"))
    assert result4.next_source_index == 6
    assert [x["path"] for x in result4.completed_sources] == [
        idents[i]["path"] for i in (0, 1, 2, 3, 4)
    ]
    assert [x["path"] for x in result4.rejected_sources] == [idents[5]["path"]]
    # The four replayed files remain in canonical stacked (no recursion).
    for i in (1, 2, 3, 4):
        assert (inputs / "stacked" / f"src_{i}.fit").exists()


# ---------------------------------------------------------------------------
# Witness 3 — invalid FITS rejection must not late-stat a moved source
# ---------------------------------------------------------------------------

def test_witness3_invalid_fits_rejection_without_late_stat(tmp_path):
    inputs, out, paths, idents, reference_ident, cursor = _build_checkpoint(
        tmp_path, 4, committed_indices=[0], rejected_indices=[],
    )
    qm, result = _headless_resume(out, inputs)
    # The source has ALREADY been moved/removed before the rejection hook
    # runs (the architectural hazard: a late stat() at the original path).
    paths[1].rename(inputs / "gone_placeholder.fit")
    assert not paths[1].exists()

    # The rejection disposition must commit from the authoritative plan
    # identity — no stat required, no "cannot stat source for identity".
    qm._drizzle_checkpoint_after_rejection(str(paths[1]))
    manifest = json.loads((out / CHECKPOINT_DIRNAME / MANIFEST_FILENAME).read_text())
    assert [x["path"] for x in manifest["rejected_sources"]] == [idents[1]["path"]]
    assert manifest["plan_cursor"] == 2
    result4 = read_drizzle_checkpoint(str(out), resolver=SafeStackedSourceResolver("stacked"))
    assert result4.next_source_index == 2
    assert [x["path"] for x in result4.rejected_sources] == [idents[1]["path"]]


# ---------------------------------------------------------------------------
# Witness 4 — crash after accepted commit, before move
# ---------------------------------------------------------------------------

def test_witness4_crash_after_accepted_commit_before_move(tmp_path):
    inputs, out, paths, idents, reference_ident, cursor = _build_checkpoint(
        tmp_path, 4, committed_indices=[0], rejected_indices=[],
    )
    qm, result = _headless_resume(out, inputs)
    # Accept src_1 at its ORIGINAL path; commit; crash BEFORE the move.
    _accept(qm, paths[1], 1)
    qm._drizzle_checkpoint_force_flush()
    assert paths[1].exists()   # never moved

    # Next Resume: the disposition is already durable -> src_1 is NOT in the
    # remaining queue -> never double-added to science.
    result2 = read_drizzle_checkpoint(str(out), resolver=SafeStackedSourceResolver("stacked"))
    assert result2.next_source_index == 2
    assert [Path(p).name for p in result2.resolved_remaining_paths] == [
        "src_2.fit", "src_3.fit",
    ]
    # Science contains exactly src_0 + src_1 (one addition each).
    expected = _reference_run_qm(tmp_path, [0, 1], 4)
    for resumed, exp in zip(result2.accumulators, expected):
        assert np.array_equal(resumed._out_img, exp._out_img)
        assert np.array_equal(resumed._out_wht, exp._out_wht)


# ---------------------------------------------------------------------------
# Witness 5 — crash after rejection commit, before move
# ---------------------------------------------------------------------------

def test_witness5_crash_after_rejection_commit_before_move(tmp_path):
    inputs, out, paths, idents, reference_ident, cursor = _build_checkpoint(
        tmp_path, 4, committed_indices=[0], rejected_indices=[],
    )
    qm, result = _headless_resume(out, inputs)
    # Reject src_1; commit; crash BEFORE the move to unaligned_by_stacker.
    qm._drizzle_checkpoint_after_rejection(str(paths[1]))
    assert paths[1].exists()   # tolerated pending physical move

    # Next Resume: the rejected source is never replayed scientifically and
    # its stale physical location is tolerated.
    result2 = read_drizzle_checkpoint(str(out), resolver=SafeStackedSourceResolver("stacked"))
    assert result2.next_source_index == 2
    assert [Path(p).name for p in result2.resolved_remaining_paths] == [
        "src_2.fit", "src_3.fit",
    ]
    assert [x["path"] for x in result2.rejected_sources] == [idents[1]["path"]]


# ---------------------------------------------------------------------------
# Witness 6 — nested stacked is refused diagnostically
# ---------------------------------------------------------------------------

def test_witness6_nested_stacked_refused(tmp_path):
    inputs, out, paths, idents, reference_ident, cursor = _build_checkpoint(
        tmp_path, 3, committed_indices=[0], rejected_indices=[],
    )
    # Move a remaining source into a NESTED non-canonical location.
    nested_dir = inputs / "stacked" / "stacked"
    nested_dir.mkdir(parents=True)
    paths[1].rename(nested_dir / paths[1].name)

    with pytest.raises(DrizzleCheckpointError) as exc:
        read_drizzle_checkpoint(str(out), resolver=SafeStackedSourceResolver("stacked"))
    assert "non-canonical nested stacked location" in str(exc.value)
    assert "src_1.fit" in str(exc.value)


# ---------------------------------------------------------------------------
# Real Windows frontier — generation 473 / frame 4640 / cursor 4649
# ---------------------------------------------------------------------------

FRONTIER_PLAN_LEN = 5324
FRONTIER_FRAME = 4640
FRONTIER_CURSOR = 4649
FRONTIER_GEN = 473
FRONTIER_REJECTED_INDICES = list(range(4631, 4640))   # 9 rejected dispositions
FRONTIER_STACKED_REMAINING = [0, 1, 2, 3]             # 220234..220337 in stacked
FRONTIER_ORIGINAL_REJECT_TARGET = 4                   # 220458 at original


class _LifecycleAligner:
    def __init__(self):
        self.stop_processing = False
        self.reference_image_path = None
        self.calls = []

    def _get_reference_image(self, folder, files, output_folder):
        pinned = self.reference_image_path
        self.calls.append((folder, tuple(files), pinned))
        assert pinned is not None
        data = np.repeat(
            fits.getdata(pinned).astype(np.float32)[..., None], 3, axis=2
        )
        header = fits.getheader(pinned)
        header["HIERARCH SEESTAR REF SRCFILE"] = os.path.basename(pinned)
        temp_dir = Path(output_folder) / "temp_processing"
        temp_dir.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(data[..., 0], header=header).writeto(
            temp_dir / "reference_image.fit", overwrite=True
        )
        return data, header


class _NoopExecutor:
    def __init__(self, max_workers=1, **_kwargs):
        self._max_workers = max_workers

    def shutdown(self, *args, **kwargs):
        return None


def _build_frontier(tmp_path):
    inputs = tmp_path / "inputs"
    out = tmp_path / "out"
    inputs.mkdir()
    out.mkdir()
    names = [
        f"Light_SH2-101_20.0s_IRCUT_20260815-{215000 + i}.fit"
        for i in range(FRONTIER_PLAN_LEN)
    ]
    for i, name in enumerate(names):
        fits.PrimaryHDU(np.full(SHAPE, (i % 200) + 1, dtype=np.uint16)).writeto(
            inputs / name
        )
    paths = [inputs / name for name in names]
    idents = [_identity(p) for p in paths]
    ref_name = "Light_SH2-101_20.0s_IRCUT_20260707-003517.fit"
    fits.PrimaryHDU(np.full(SHAPE, 9, dtype=np.uint16)).writeto(inputs / ref_name)
    reference_ident = _identity(inputs / ref_name)

    accepted_idx = [
        i for i in range(FRONTIER_CURSOR) if i not in set(FRONTIER_REJECTED_INDICES)
    ]
    assert len(accepted_idx) == FRONTIER_FRAME
    completed = [idents[i] for i in accepted_idx]
    rejected = [idents[i] for i in FRONTIER_REJECTED_INDICES]
    # The real recovered 127-batch decomposition: the frontier cursor 4649
    # sits inside the first future batch (4620 + 37 survivors, residual 8).
    decomposition = [42] * 110 + [37] + [42] * 15 + [37]
    assert sum(decomposition) == FRONTIER_PLAN_LEN
    assert len(decomposition) == 127

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, out, inputs, kernel="square", group_size=1)
    qm._resume_reference_identity = reference_ident
    cfg = build_drizzle_canonical_config(
        qm, product_version=qm._canonical_product_version()
    )
    writer = DrizzleCheckpointWriter(
        str(out), qm._canonical_product_version(), cfg, _wcs(), SHAPE
    )
    accs = [
        DrizzleAccumulator(SHAPE, kernel="square", pixfrac=1.0) for _ in range(3)
    ]
    _add(accs, _frame(0))
    _add(accs, _frame(1))
    writer.commit(
        accs,
        session_binding={
            "input_roots": [str(inputs)],
            "reference": reference_ident,
            "plan": {"sources": idents, "decomposition": decomposition},
            "reference_geometry": _reference_geometry(),
        },
        counters={
            "frame_count": 2,
            "stacked_batches_count": 2,
            "total_exposure_seconds": 2.0,
            "exposure_unknown_count": 0,
            "exposure_min": 1.0,
            "exposure_max": 1.0,
        },
        completed_sources=idents[:2],
    )
    ckpt_dir = out / CHECKPOINT_DIRNAME
    manifest = json.loads((ckpt_dir / MANIFEST_FILENAME).read_text())
    for ch in manifest["channels"]:
        for key in ("out_img", "out_wht"):
            old = ch[key]["file"]
            new = old.replace(f"gen-{1:08d}-", f"gen-{FRONTIER_GEN:08d}-")
            (ckpt_dir / old).rename(ckpt_dir / new)
            ch[key]["file"] = new
    manifest["generation"] = FRONTIER_GEN
    manifest["frame_count"] = FRONTIER_FRAME
    manifest["stacked_batches_count"] = FRONTIER_FRAME
    manifest["total_exposure_seconds"] = float(FRONTIER_FRAME)
    manifest["exposure_unknown_count"] = 0
    manifest["exposure_min"] = 1.0
    manifest["exposure_max"] = 1.0
    manifest["session"]["plan"]["sources"] = idents
    manifest["session"]["plan"]["decomposition"] = decomposition
    manifest["completed_sources"] = completed
    manifest["rejected_sources"] = rejected
    manifest["plan_cursor"] = FRONTIER_CURSOR
    (ckpt_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    # Physical layout per the real Windows witness.
    stacked = inputs / "stacked"
    stacked.mkdir()
    for i in accepted_idx:
        paths[i].rename(stacked / paths[i].name)
    unaligned = inputs / "unaligned_by_stacker"
    unaligned.mkdir()
    for i in FRONTIER_REJECTED_INDICES:
        paths[i].rename(unaligned / paths[i].name)
    frontier = list(range(FRONTIER_CURSOR, FRONTIER_PLAN_LEN))
    for rel in FRONTIER_STACKED_REMAINING:
        i = frontier[rel]
        paths[i].rename(stacked / paths[i].name)

    return inputs, out, paths, idents, reference_ident, frontier


def test_windows_frontier_473_4640_4649_lifecycle(tmp_path, monkeypatch):
    inputs, out, paths, idents, reference_ident, frontier = _build_frontier(tmp_path)
    monkeypatch.setattr(
        queue_manager_module,
        "ProcessPoolExecutor",
        lambda **kwargs: _NoopExecutor(**kwargs),
    )
    qm = SeestarQueuedStacker(batch_size=2, autotune=False)
    qm.update_progress = lambda *args, **kwargs: None
    aligner = _LifecycleAligner()
    qm.aligner = aligner
    worker_started = threading.Event()
    worker_snapshot = {}

    def _worker():
        worker_snapshot["queue"] = list(qm.queue.queue)
        worker_snapshot["frame_count"] = qm._drizzle_frame_count
        worker_snapshot["plan_cursor"] = qm._drizzle_plan_cursor
        worker_snapshot["rejected"] = [
            x["name"] for x in qm._drizzle_rejected_sources
        ]
        worker_snapshot["continuation"] = qm._drizzle_resume_continuation
        worker_started.set()

    qm._worker = _worker
    qm._solve_astrometry_async = lambda *args, **kwargs: _wcs()
    started = qm.start_processing(
        input_dir=str(inputs),
        output_dir=str(out),
        use_drizzle=True,
        drizzle_scale=1.0,
        drizzle_kernel="square",
        drizzle_pixfrac=1.0,
        drizzle_wht_threshold=0.0,
        drizzle_group_size=1,
        batch_size=2,
        min_w=0.01,
        move_stacked=True,
        perform_cleanup=False,
        resume_intent="resume",
        reproject_between_batches=False,
        reproject_coadd_final=False,
    )
    if qm.processing_thread is not None:
        qm.processing_thread.join(timeout=5)
    assert started is True, f"start_processing refused: {qm.processing_error}"
    assert worker_started.is_set()
    assert worker_snapshot["frame_count"] == FRONTIER_FRAME
    assert worker_snapshot["plan_cursor"] == FRONTIER_CURSOR
    assert len(worker_snapshot["rejected"]) == 9

    queue_items = [
        q for q in worker_snapshot["queue"]
        if q != queue_manager_module._BATCH_BREAK_TOKEN
    ]
    assert len(queue_items) == FRONTIER_PLAN_LEN - FRONTIER_CURSOR
    # First queue source is the stacked 220234-equivalent.
    assert Path(queue_items[0]).name == paths[frontier[0]].name
    assert (inputs / "stacked" / Path(queue_items[0]).name).exists()

    # Replay the four stacked sources; moves must be idempotent no-ops.
    for rel in FRONTIER_STACKED_REMAINING:
        i = frontier[rel]
        stacked_path = inputs / "stacked" / paths[i].name
        _accept(qm, stacked_path, i)
        qm._move_to_stacked([str(stacked_path)])
    assert not (inputs / "stacked" / "stacked").exists()

    # The fifth frontier source (at original) is rejected.
    reject_target = frontier[FRONTIER_ORIGINAL_REJECT_TARGET]
    assert paths[reject_target].exists()
    qm._drizzle_checkpoint_after_rejection(str(paths[reject_target]))
    qm._move_to_unaligned(str(paths[reject_target]))

    manifest = json.loads((out / CHECKPOINT_DIRNAME / MANIFEST_FILENAME).read_text())
    assert manifest["generation"] >= FRONTIER_GEN + 1
    assert manifest["frame_count"] == FRONTIER_FRAME + 4
    assert manifest["plan_cursor"] == FRONTIER_CURSOR + 5
    assert len(manifest["rejected_sources"]) == 10
    # Accepted replay dispositions persist the AUTHORITATIVE plan identities.
    for rel in FRONTIER_STACKED_REMAINING:
        i = frontier[rel]
        assert idents[i] in manifest["completed_sources"]
    assert idents[reject_target] in manifest["rejected_sources"]

    result4 = read_drizzle_checkpoint(
        str(out), resolver=SafeStackedSourceResolver("stacked")
    )
    assert result4.counters["frame_count"] == FRONTIER_FRAME + 4
    assert result4.next_source_index == FRONTIER_CURSOR + 5
    completed_keys = {
        (s["path"], s["size"], s["mtime_ns"]) for s in result4.completed_sources
    }
    for rel in FRONTIER_STACKED_REMAINING:
        i = frontier[rel]
        assert (idents[i]["path"], idents[i]["size"], idents[i]["mtime_ns"]) in completed_keys
    rejected_keys = {
        (s["path"], s["size"], s["mtime_ns"]) for s in result4.rejected_sources
    }
    assert (idents[reject_target]["path"], idents[reject_target]["size"], idents[reject_target]["mtime_ns"]) in rejected_keys
    # Next pending source is correct (frontier[5]).
    assert Path(result4.resolved_remaining_paths[0]).name == paths[frontier[5]].name
    # No nested stacked directory anywhere.
    assert not (inputs / "stacked" / "stacked").exists()
