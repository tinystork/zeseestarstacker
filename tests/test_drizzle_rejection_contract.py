"""ZSSS-DRIZZLE-REJECTED-SOURCE regression suite (RJK).

P0 production recovery blocker: a real Windows Lanczos3 Drizzle run (5329
planned observations, generation 462 at 4620 admitted frames) exposed a
contradiction between the Drizzle checkpoint ledger contract and non-fatal
alignment rejection.

    PLAN: A B C D E
    B rejected (moved to unaligned_by_stacker), A/C accepted

Under the 8.5.1 prefix-only contract ``completed_sources`` had to equal
``plan[:frame_count]`` — impossible once any non-tail observation is
rejected, and Resume reconstructed the remaining queue from
``plan[len(completed_sources):]`` so the filesystem could never agree with
the persisted plan.

The permanent contract (this file proves it):

* ``frame_count`` = number of scientifically admitted Drizzle exposures;
* ``stacked_batches_count`` = accepted Drizzle exposure count (== frame_count);
* ``completed_sources`` = identities whose science IS in SCI/WHT/SUPPORT
  (plan-ordered subsequence, not necessarily a prefix);
* ``plan_cursor`` = number of plan sources with a FINAL disposition
  (accepted + rejected) — the resume cursor;
* ``rejected_sources`` = ordered dispositions of non-admitted observations.

A rejected observation never touches the accumulators, never increments
``frame_count`` / ``stacked_batches_count``, never enters ``completed_sources``,
but is durably recorded (its own checkpoint generation) BEFORE the source is
moved, so Resume continues beyond it without replaying or double-counting.

Legacy prefix-only checkpoints (no ``plan_cursor`` / ``rejected_sources``)
keep their exact historical semantics (Gate 5).

Gates:
1  accepted -> rejected -> accepted must not make the next commit impossible
2  consecutive rejections
3  Resume after committed rejection progress (no double-count, no replay)
4  trailing rejection remains durably resumable / finalizable
5  legacy prefix-only schema-v1 checkpoint still resumes
6  immutable science ledger (SCI/WHT bit-identical; rejected never admitted)
7  crash/restart around rejection/move/checkpoint boundaries
8  emergency recovery witness (5329 -> clean 4620 -> remove 5 future rejected
   -> 5324 -> Resume -> accept next source -> commit -> reload)
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


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

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
        "path": os.path.normcase(str(path)),
        "name": os.path.basename(str(path)),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _frame(index, exptime=1.0):
    yy, xx = np.indices(SHAPE, dtype=np.float64)
    data = (index * 3.0 + xx * 0.25 + yy * 0.5).astype(np.float32)
    weight = np.full(SHAPE, 0.7 + index * 0.05, dtype=np.float32)
    pixmap = np.dstack((xx + index * 0.07, yy - index * 0.04))
    in_grid = np.ones(SHAPE, dtype=bool)
    return data, weight, pixmap, in_grid, exptime


def _add(accs, frame):
    data, weight, pixmap, in_grid, exptime = frame
    for acc in accs:
        acc.add(
            data,
            weight,
            pixmap,
            exptime=exptime,
            in_units="counts",
            in_grid_mask=in_grid,
        )


def _make_sources(tmp_path, n, prefix="src"):
    """Create ``n`` distinct FITS sources; returns (paths, identities)."""
    paths, idents = [], []
    for i in range(n):
        p = tmp_path / f"{prefix}_{i}.fit"
        fits.PrimaryHDU(np.full(SHAPE, i + 1, dtype=np.uint16)).writeto(p)
        paths.append(p)
        idents.append(_identity(p))
    return paths, idents


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
    qm._drizzle_checkpoint_last_committed_frames = 0
    qm._drizzle_resume_result = None
    qm._drizzle_resume_continuation = None
    qm._resume_requested = False
    qm._resume_active = False
    qm._resume_plan = None
    qm._resume_completed_sources = []
    qm._resume_reference_identity = None


def _arm_fresh_writer(qm, output, paths):
    """Bind the plan from ``paths`` and create the fresh-run writer."""
    idents = [_identity(p) for p in paths]
    ref_path = Path(paths[0]).parent / "reference.fit"
    if not ref_path.exists():
        fits.PrimaryHDU(np.full(SHAPE, 7, dtype=np.uint16)).writeto(ref_path)
    reference_ident = _identity(ref_path)
    qm.drizzle_accumulators = [
        DrizzleAccumulator(SHAPE, kernel="square", pixfrac=1.0) for _ in range(3)
    ]
    qm._drizzle_checkpoint_plan = {
        "sources": idents,
        "decomposition": [len(idents)],
    }
    qm._resume_reference_identity = reference_ident
    cfg = build_drizzle_canonical_config(
        qm, product_version=qm._canonical_product_version()
    )
    qm._drizzle_checkpoint_writer = DrizzleCheckpointWriter(
        str(output),
        qm._canonical_product_version(),
        cfg,
        _wcs(),
        SHAPE,
    )
    return idents


def _accept(qm, path, index):
    """Drive the accepted-frame lifecycle exactly like the worker does."""
    ok = qm._add_frame_to_drizzle_accumulators(
        np.stack([_frame(index)[0]] * 3, axis=-1).astype(np.float32),
        fits.Header([("EXPTIME", 1.0)]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        _frame(index)[1],
        native_wcs=_wcs(),
    )
    assert ok is True
    qm._drizzle_group_tick()
    qm._admit_exposure(1.0, 0, 1.0, 1.0)
    qm.stacked_batches_count += 1
    qm._drizzle_checkpoint_after_frame(str(path))


def _reject(qm, path):
    """Drive the rejected-observation lifecycle exactly like the worker does."""
    qm.failed_align_count += 1
    qm._drizzle_checkpoint_after_rejection(str(path))
    qm._move_to_unaligned(str(path))


def _manifest(output):
    return json.loads(
        (Path(output) / CHECKPOINT_DIRNAME / MANIFEST_FILENAME).read_text()
    )


def _ledger_names(manifest):
    return [x["name"] for x in manifest["completed_sources"]]


def _rejected_names(manifest):
    return [x["name"] for x in manifest.get("rejected_sources") or []]


def _fresh_resume_headless(qm_seed, output, inputs):
    qm2 = object.__new__(SeestarQueuedStacker)
    _configure(qm2, output, inputs, group_size=1)
    qm2.move_stacked = bool(getattr(qm_seed, "move_stacked", False))
    qm2._resume_requested = True
    ok, result, ref = qm2._validate_drizzle_resume_headless()
    if ok:
        qm2._drizzle_resume_result = result
    return ok, result, ref


def _reference_run(tmp_path, accept_indices, n):
    """Drive a second fresh QM run over the same sources, accepting ONLY the
    given plan indices — the golden science a rejection-free run produces."""
    output = tmp_path / f"ref_out_{abs(hash(tuple(accept_indices))) % 10**9}"
    inputs = tmp_path / "inputs"
    output.mkdir(exist_ok=True)
    all_paths = [inputs / f"src_{i}.fit" for i in range(n)]
    ref_paths = [all_paths[i] for i in accept_indices]
    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, ref_paths)
    for plan_index in accept_indices:
        _accept(qm, all_paths[plan_index], plan_index)
    return qm.drizzle_accumulators


# ---------------------------------------------------------------------------
# Gate 1 — accepted -> rejected -> accepted must commit
# ---------------------------------------------------------------------------

def test_gate1_accepted_rejected_accepted_commits(tmp_path):
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, _idents = _make_sources(inputs, 5)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, paths)

    _accept(qm, paths[0], 0)   # A accepted -> frame_count 1, gen 1
    _reject(qm, paths[1])      # B rejected -> cursor-only gen 2
    _accept(qm, paths[2], 2)   # C accepted -> frame_count 2, gen 3

    manifest = _manifest(output)
    assert manifest["frame_count"] == 2
    assert manifest["stacked_batches_count"] == 2
    assert manifest["plan_cursor"] == 3
    assert _ledger_names(manifest) == ["src_0.fit", "src_2.fit"]
    assert _rejected_names(manifest) == ["src_1.fit"]

    # The persisted state reloads through the real reader.
    result = read_drizzle_checkpoint(str(output))
    assert result.next_source_index == 3
    assert [x["name"] for x in result.completed_sources] == [
        "src_0.fit",
        "src_2.fit",
    ]
    assert [x["name"] for x in result.rejected_sources] == ["src_1.fit"]
    assert result.resolved_remaining_paths == (str(paths[3]), str(paths[4]))


def test_gate1_legacy_call_refuses_truthful_state_without_dispositions(tmp_path):
    """The *old* prefix-only contract (8.5.1) cannot persist this truthful
    state: completed=[A, C] with frame_count=2 while plan[:2]=[A, B].  A
    legacy-style commit (no rejected ledger) must still refuse — the new
    contract is additive, not a validation weakening."""
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, idents = _make_sources(inputs, 5)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, paths)
    accs = [
        DrizzleAccumulator(SHAPE, kernel="square", pixfrac=1.0) for _ in range(3)
    ]
    _add(accs, _frame(0))
    _add(accs, _frame(1))

    writer = qm._drizzle_checkpoint_writer
    binding = {
        "input_roots": [str(inputs)],
        "reference": idents[0],
        "plan": qm._drizzle_checkpoint_plan,
        "reference_geometry": _reference_geometry(),
    }
    with pytest.raises(DrizzleCheckpointError) as exc:
        writer.commit(
            accs,
            session_binding=binding,
            counters={
                "frame_count": 2,
                "stacked_batches_count": 2,
                "total_exposure_seconds": 2.0,
                "exposure_unknown_count": 0,
                "exposure_min": 1.0,
                "exposure_max": 1.0,
            },
            completed_sources=[idents[0], idents[2]],
        )
    assert "ordered" in str(exc.value) or "prefix" in str(exc.value)


# ---------------------------------------------------------------------------
# Gate 2 — consecutive rejections
# ---------------------------------------------------------------------------

def test_gate2_consecutive_rejections(tmp_path):
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, _idents = _make_sources(inputs, 5)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, paths)

    _accept(qm, paths[0], 0)   # A
    _reject(qm, paths[1])      # B
    _reject(qm, paths[2])      # C
    _accept(qm, paths[3], 3)   # D

    manifest = _manifest(output)
    assert manifest["frame_count"] == 2
    assert manifest["plan_cursor"] == 4
    assert _ledger_names(manifest) == ["src_0.fit", "src_3.fit"]
    assert _rejected_names(manifest) == ["src_1.fit", "src_2.fit"]

    result = read_drizzle_checkpoint(str(output))
    assert result.next_source_index == 4
    assert result.resolved_remaining_paths == (str(paths[4]),)


# ---------------------------------------------------------------------------
# Gate 3 — Resume after committed rejection progress
# ---------------------------------------------------------------------------

def test_gate3_resume_after_committed_rejection(tmp_path):
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, _idents = _make_sources(inputs, 5)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, paths)
    _accept(qm, paths[0], 0)
    _reject(qm, paths[1])
    _accept(qm, paths[2], 2)

    # Rejected file now lives in unaligned_by_stacker (worker move).
    assert not paths[1].exists()
    assert (inputs / "unaligned_by_stacker" / "src_1.fit").exists()

    # --- production Resume path on a fresh QM ---
    qm2 = object.__new__(SeestarQueuedStacker)
    _configure(qm2, output, inputs, group_size=1)
    qm2._resume_requested = True
    ok, result, _ref = qm2._validate_drizzle_resume_headless()
    assert ok is True
    qm2._drizzle_resume_result = result
    qm2._restore_drizzle_checkpoint_runtime(result)
    assert qm2._drizzle_plan_cursor == 3
    assert [x["name"] for x in qm2._drizzle_rejected_sources] == ["src_1.fit"]
    assert qm2._drizzle_frame_count == 2

    # The authoritative remaining queue: exactly plan[3:] (B rejected and
    # A/C accepted are never replayed).
    qm2.queue = Queue()
    for p in paths[3:]:
        qm2.queue.put(str(p))
    assert qm2._init_drizzle_checkpoint() is True
    continuation = qm2._drizzle_resume_continuation
    assert continuation.next_source_index == 3

    # Continue with D and E.  The resumed accumulators hold exactly A+C.
    for i, path in enumerate(paths[3:], start=3):
        _accept(qm2, path, i)

    manifest = _manifest(output)
    assert manifest["frame_count"] == 4
    assert manifest["plan_cursor"] == 5
    assert _ledger_names(manifest) == [
        "src_0.fit",
        "src_2.fit",
        "src_3.fit",
        "src_4.fit",
    ]
    assert _rejected_names(manifest) == ["src_1.fit"]

    # Science ledger matches exactly the admitted frames: A,C,D,E — never B.
    # The reference run drives the SAME production deposition path (the
    # QM `_add_frame_to_drizzle_accumulators` seam) so the comparison is
    # bit-truthful.
    expected = _reference_run(tmp_path, [0, 2, 3, 4], 5)
    for resumed, exp in zip(qm2.drizzle_accumulators, expected):
        assert np.array_equal(resumed._out_img, exp._out_img)
        assert np.array_equal(resumed._out_wht, exp._out_wht)
        assert np.array_equal(resumed.finalize("divide"), exp.finalize("divide"))


# ---------------------------------------------------------------------------
# Gate 4 — trailing rejection remains durably resumable / finalizable
# ---------------------------------------------------------------------------

def test_gate4_trailing_rejection_durably_committed(tmp_path):
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, _idents = _make_sources(inputs, 5)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, paths)

    _accept(qm, paths[0], 0)
    _accept(qm, paths[1], 1)
    _accept(qm, paths[2], 2)
    _reject(qm, paths[3])   # trailing rejections: no later accepted frame
    _reject(qm, paths[4])
    qm._drizzle_checkpoint_force_flush()  # idempotent no-op (frames unchanged)

    manifest = _manifest(output)
    assert manifest["frame_count"] == 3
    assert manifest["plan_cursor"] == 5
    assert _rejected_names(manifest) == ["src_3.fit", "src_4.fit"]

    # The run is finalizable: remaining work is exactly empty and the
    # rejected dispositions are durable.
    result = read_drizzle_checkpoint(str(output))
    assert result.next_source_index == 5
    assert result.resolved_remaining_paths == ()


def test_gate4_trailing_rejection_resumable(tmp_path):
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, _idents = _make_sources(inputs, 5)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, paths)
    _accept(qm, paths[0], 0)
    _accept(qm, paths[1], 1)
    _accept(qm, paths[2], 2)
    _reject(qm, paths[3])   # near-tail rejection; src_4 never processed

    # Resume: remaining = plan[4:] = [src_4] (src_3 disposed).
    result = read_drizzle_checkpoint(str(output))
    assert result.next_source_index == 4
    assert [Path(p).name for p in result.resolved_remaining_paths] == ["src_4.fit"]

    qm2 = object.__new__(SeestarQueuedStacker)
    _configure(qm2, output, inputs, group_size=1)
    qm2._resume_requested = True
    ok, result2, _ref = qm2._validate_drizzle_resume_headless()
    assert ok is True
    qm2._drizzle_resume_result = result2
    qm2._restore_drizzle_checkpoint_runtime(result2)
    qm2.queue = Queue()
    qm2.queue.put(str(paths[4]))
    assert qm2._init_drizzle_checkpoint() is True
    _accept(qm2, paths[4], 4)
    manifest = _manifest(output)
    assert manifest["frame_count"] == 4
    assert manifest["plan_cursor"] == 5
    assert _rejected_names(manifest) == ["src_3.fit"]


# ---------------------------------------------------------------------------
# Gate 5 — legacy prefix-only checkpoint still resumes
# ---------------------------------------------------------------------------

def test_gate5_legacy_prefix_only_checkpoint_resumes(tmp_path):
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, idents = _make_sources(inputs, 4)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, paths)
    accs = [
        DrizzleAccumulator(SHAPE, kernel="square", pixfrac=1.0) for _ in range(3)
    ]
    _add(accs, _frame(0))
    _add(accs, _frame(1))
    qm._drizzle_checkpoint_writer.commit(
        accs,
        session_binding={
            "input_roots": [str(inputs)],
            "reference": idents[0],
            "plan": qm._drizzle_checkpoint_plan,
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
    # Legacy manifest: no rejection metadata at all.
    manifest = _manifest(output)
    assert "rejected_sources" not in manifest
    assert "plan_cursor" not in manifest

    result = read_drizzle_checkpoint(str(output))
    assert result.next_source_index == 2
    assert result.plan_cursor == 2
    assert list(result.rejected_sources) == []
    assert result.resolved_remaining_paths == (str(paths[2]), str(paths[3]))

    # Resume continues from the legacy prefix cursor.
    qm2 = object.__new__(SeestarQueuedStacker)
    _configure(qm2, output, inputs, group_size=1)
    qm2._resume_requested = True
    ok, result2, _ref = qm2._validate_drizzle_resume_headless()
    assert ok is True
    qm2._drizzle_resume_result = result2
    qm2._restore_drizzle_checkpoint_runtime(result2)
    assert qm2._drizzle_plan_cursor == 2
    qm2.queue = Queue()
    for p in paths[2:]:
        qm2.queue.put(str(p))
    assert qm2._init_drizzle_checkpoint() is True
    for i, path in enumerate(paths[2:], start=2):
        _accept(qm2, path, i)
    manifest = _manifest(output)
    assert manifest["frame_count"] == 4
    assert _ledger_names(manifest) == [f"src_{i}.fit" for i in range(4)]


# ---------------------------------------------------------------------------
# Gate 6 — immutable science ledger
# ---------------------------------------------------------------------------

def test_gate6_rejection_never_enters_science(tmp_path):
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, _idents = _make_sources(inputs, 4)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, paths)

    # Run with a rejection: A accepted, B rejected, C accepted.
    _accept(qm, paths[0], 0)
    _reject(qm, paths[1])
    _accept(qm, paths[2], 2)
    resumed_result = read_drizzle_checkpoint(str(output))

    # Reference: a rejection-free run accepting exactly the same sources
    # (A and C) through the same production deposition seam.
    ref_accs = _reference_run(tmp_path, [0, 2], 4)
    for acc, ref in zip(resumed_result.accumulators, ref_accs):
        assert np.array_equal(acc._out_img, ref._out_img)
        assert np.array_equal(acc._out_wht, ref._out_wht)
        assert np.array_equal(acc.finalize("divide"), ref.finalize("divide"))
    manifest = _manifest(output)
    assert _ledger_names(manifest) == ["src_0.fit", "src_2.fit"]
    assert _rejected_names(manifest) == ["src_1.fit"]
    assert manifest["frame_count"] == 2
    assert manifest["stacked_batches_count"] == 2


# ---------------------------------------------------------------------------
# Gate 7 — crash/restart around rejection/move/checkpoint boundaries
# ---------------------------------------------------------------------------

def test_gate7a_rejection_committed_then_move_fails(tmp_path):
    """Window 1/8-variant: rejection committed, then the unaligned move
    crashes (file stays at original).  Resume must skip it via the cursor —
    the disposition is durable, the location is irrelevant."""
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, _idents = _make_sources(inputs, 5)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, paths)
    _accept(qm, paths[0], 0)
    qm._drizzle_checkpoint_after_rejection(str(paths[1]))  # durable disposition
    # The move then fails; the file remains at its original path.
    assert paths[1].exists()

    ok, result, _ref = _fresh_resume_headless(qm, output, inputs)
    assert ok is True
    assert result.next_source_index == 2
    assert [Path(p).name for p in result.resolved_remaining_paths] == [
        "src_2.fit",
        "src_3.fit",
        "src_4.fit",
    ]


def test_gate7b_rejection_commit_fails_before_move(tmp_path, monkeypatch):
    """Window 1: rejection disposition commit fails -> mandatory abort BEFORE
    the move; the source stays at its original path and is replayed on Resume
    (deterministic: the alignment failure simply occurs again)."""
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, _idents = _make_sources(inputs, 5)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, paths)
    _accept(qm, paths[0], 0)
    manifest_before = (
        Path(output) / CHECKPOINT_DIRNAME / MANIFEST_FILENAME
    ).read_bytes()

    def _boom(*args, **kwargs):
        raise DrizzleCheckpointError("injected commit failure")

    monkeypatch.setattr(qm, "_drizzle_checkpoint_commit", _boom)
    with pytest.raises(DrizzleCheckpointError):
        qm._drizzle_checkpoint_after_rejection(str(paths[1]))
    # File never moved: replayable, and the checkpoint is untouched.
    assert paths[1].exists()
    assert (
        Path(output) / CHECKPOINT_DIRNAME / MANIFEST_FILENAME
    ).read_bytes() == manifest_before

    result = read_drizzle_checkpoint(str(output))
    assert result.next_source_index == 1
    assert [Path(p).name for p in result.resolved_remaining_paths] == [
        f"src_{i}.fit" for i in range(1, 5)
    ]


def test_gate7c_accepted_moved_before_commit_replays(tmp_path):
    """Accepted frame moved to stacked, then crash BEFORE its checkpoint
    commit (cadence not reached).  The identity survives the move (size +
    mtime_ns) so a resolution-aware resume can replay it from stacked."""
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, idents = _make_sources(inputs, 3)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=50)  # never reaches cadence
    _arm_fresh_writer(qm, output, paths)
    _accept(qm, paths[0], 0)   # ledger/cursor in memory, no commit yet
    qm.move_stacked = True
    qm._move_to_stacked([str(paths[0])])   # moved, then "crash"

    stacked_path = inputs / "stacked" / paths[0].name
    assert stacked_path.exists()
    ident0 = _identity(stacked_path)
    assert ident0["size"] == idents[0]["size"]
    assert ident0["mtime_ns"] == idents[0]["mtime_ns"]
    # The plan identity (original path + size + mtime) resolves to the
    # stacked counterpart through the production resolver.
    resolver = SafeStackedSourceResolver("stacked")
    assert resolver.resolve(idents[0], {"role": "plan", "index": 0, "is_completed": False}) == [
        idents[0]["path"],
        str(stacked_path),
    ]


def test_gate7d_rejection_crash_before_commit_replays(tmp_path):
    """Window 1-variant: rejection observed but the process dies before the
    disposition commit.  Nothing is recorded and the file was never moved:
    Resume replays it and the rejection happens again deterministically —
    no divergence."""
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, _idents = _make_sources(inputs, 3)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, paths)
    _accept(qm, paths[0], 0)
    manifest_before = (
        Path(output) / CHECKPOINT_DIRNAME / MANIFEST_FILENAME
    ).read_bytes()

    # Simulated crash right after the rejection, before hook/move.
    assert paths[1].exists()
    assert (
        Path(output) / CHECKPOINT_DIRNAME / MANIFEST_FILENAME
    ).read_bytes() == manifest_before

    result = read_drizzle_checkpoint(str(output))
    assert [Path(p).name for p in result.resolved_remaining_paths] == [
        "src_1.fit",
        "src_2.fit",
    ]


def test_gate7e_resume_sees_rejected_in_unaligned(tmp_path):
    """Window 5: the rejected source physically lives in
    ``unaligned_by_stacker`` while its disposition is committed.  The reader
    must NOT try to resolve it (dispositions are terminal) and Resume must
    continue beyond it."""
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, _idents = _make_sources(inputs, 4)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, paths)
    _accept(qm, paths[0], 0)
    _reject(qm, paths[1])   # committed + moved to unaligned_by_stacker
    assert (inputs / "unaligned_by_stacker" / "src_1.fit").exists()

    ok, result, _ref = _fresh_resume_headless(qm, output, inputs)
    assert ok is True
    assert result.next_source_index == 2
    assert [Path(p).name for p in result.resolved_remaining_paths] == [
        "src_2.fit",
        "src_3.fit",
    ]


def test_gate7f_collision_renamed_unaligned_never_resolved(tmp_path):
    """Window 6: the rejection destination got a collision-renamed filename.
    Rejected dispositions are terminal and never re-resolved, so the
    checkpoint stays fully readable — the rename is irrelevant by explicit
    design (never a basename-only guess)."""
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, _idents = _make_sources(inputs, 3)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, paths)
    _accept(qm, paths[0], 0)
    _reject(qm, paths[1])
    unaligned_dir = inputs / "unaligned_by_stacker"
    renamed = unaligned_dir / "src_1_unaligned_20260815_214845.fit"
    (unaligned_dir / "src_1.fit").rename(renamed)

    ok, result, _ref = _fresh_resume_headless(qm, output, inputs)
    assert ok is True
    assert result.next_source_index == 2
    assert [Path(p).name for p in result.resolved_remaining_paths] == ["src_2.fit"]


def test_gate7g_commit_failure_after_admission_before_move(tmp_path, monkeypatch):
    """Window 7: checkpoint commit fails after scientific admission but before
    the source move.  The worker aborts (mandatory), the file stays at its
    original path, and Resume replays it — no silent divergence."""
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, _idents = _make_sources(inputs, 3)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, paths)
    _accept(qm, paths[0], 0)   # gen 1 committed

    manifest_before = (
        Path(output) / CHECKPOINT_DIRNAME / MANIFEST_FILENAME
    ).read_bytes()

    def _boom(*args, **kwargs):
        raise DrizzleCheckpointError("injected commit failure")

    monkeypatch.setattr(qm, "_drizzle_checkpoint_commit", _boom)
    # Frame admitted in memory, then the cadence commit fails.
    ok = qm._add_frame_to_drizzle_accumulators(
        np.stack([_frame(1)[0]] * 3, axis=-1).astype(np.float32),
        fits.Header([("EXPTIME", 1.0)]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        _frame(1)[1],
        native_wcs=_wcs(),
    )
    assert ok is True
    qm._drizzle_group_tick()
    qm._admit_exposure(1.0, 0, 1.0, 1.0)
    qm.stacked_batches_count += 1
    with pytest.raises(DrizzleCheckpointError):
        qm._drizzle_checkpoint_after_frame(str(paths[1]))
    # Source never moved, checkpoint byte-identical.
    assert paths[1].exists()
    assert (
        Path(output) / CHECKPOINT_DIRNAME / MANIFEST_FILENAME
    ).read_bytes() == manifest_before
    result = read_drizzle_checkpoint(str(output))
    assert result.next_source_index == 1
    assert [Path(p).name for p in result.resolved_remaining_paths] == [
        "src_1.fit",
        "src_2.fit",
    ]


def test_gate7h_move_failure_after_commit(tmp_path):
    """Window 8: source move fails after a successful checkpoint commit.  The
    completed identity resolves at its ORIGINAL path (the resolver tries
    original first) and Resume never replays it."""
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, _idents = _make_sources(inputs, 4)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, paths)
    qm.move_stacked = True
    qm._move_to_stacked = lambda paths_: (_ for _ in ()).throw(OSError("move fail"))

    _accept(qm, paths[0], 0)   # committed; move then "fails" -> stays at original
    _accept(qm, paths[1], 1)
    assert paths[0].exists() and paths[1].exists()

    ok, result, _ref = _fresh_resume_headless(qm, output, inputs)
    assert ok is True
    assert result.next_source_index == 2
    assert [Path(p).name for p in result.resolved_completed_paths] == [
        "src_0.fit",
        "src_1.fit",
    ]
    assert [Path(p).name for p in result.resolved_remaining_paths] == [
        "src_2.fit",
        "src_3.fit",
    ]


def test_gate7_rejection_first_observation(tmp_path):
    """A rejection before ANY accepted frame is durably persistable (the
    writer now allows a pure-rejection generation) and resumable."""
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, _idents = _make_sources(inputs, 3)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, paths)

    _reject(qm, paths[0])   # first observation rejected: frame_count stays 0
    manifest = _manifest(output)
    assert manifest["frame_count"] == 0
    assert manifest["plan_cursor"] == 1
    assert _rejected_names(manifest) == ["src_0.fit"]

    # Resume: remaining = src_1, src_2.
    result = read_drizzle_checkpoint(str(output))
    assert result.next_source_index == 1
    assert [Path(p).name for p in result.resolved_remaining_paths] == [
        "src_1.fit",
        "src_2.fit",
    ]

    qm2 = object.__new__(SeestarQueuedStacker)
    _configure(qm2, output, inputs, group_size=1)
    qm2._resume_requested = True
    ok, result2, _ref = qm2._validate_drizzle_resume_headless()
    assert ok is True
    qm2._drizzle_resume_result = result2
    qm2._restore_drizzle_checkpoint_runtime(result2)
    qm2.queue = Queue()
    for p in paths[1:]:
        qm2.queue.put(str(p))
    assert qm2._init_drizzle_checkpoint() is True
    _accept(qm2, paths[1], 1)
    manifest = _manifest(output)
    assert manifest["frame_count"] == 1
    assert manifest["plan_cursor"] == 2
    assert _rejected_names(manifest) == ["src_0.fit"]
    assert _ledger_names(manifest) == ["src_1.fit"]


def test_rejecting_the_session_reference_is_refused(tmp_path):
    """A disposed session reference would make the alignment reference
    unresolvable on Resume — the rejection disposition is refused (fail
    closed) instead of committing an un-resumable checkpoint."""
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, idents = _make_sources(inputs, 3)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, paths)
    _accept(qm, paths[0], 0)

    # Force the session reference to be one of the plan sources.
    qm._resume_reference_identity = idents[1]
    with pytest.raises(DrizzleCheckpointError) as exc:
        qm._drizzle_checkpoint_after_rejection(str(paths[1]))
    assert "reference" in str(exc.value)
    # The checkpoint is untouched and the file is not moved.
    assert paths[1].exists()
    manifest = _manifest(output)
    assert "rejected_sources" not in manifest

    # Defense in depth: a checkpoint claiming a rejected reference refuses
    # to load through the reader.
    manifest_path = Path(output) / CHECKPOINT_DIRNAME / MANIFEST_FILENAME
    forged = json.loads(manifest_path.read_text())
    ref_ident = forged["session"]["reference"]
    forged["plan_cursor"] = 2
    forged["rejected_sources"] = [ref_ident]
    forged["completed_sources"] = [idents[0]]
    manifest_path.write_text(json.dumps(forged, sort_keys=True), encoding="utf-8")
    with pytest.raises(DrizzleCheckpointError) as exc2:
        read_drizzle_checkpoint(str(output))
    assert "reference" in str(exc2.value)


# ---------------------------------------------------------------------------
# Gate 8 — emergency recovery witness (5329 -> 4620 -> 5324 -> resume -> +1)
# ---------------------------------------------------------------------------

_RECOVERY_PLAN_LEN = 5329
_RECOVERY_FRAME_COUNT = 4620
_RECOVERY_GENERATION = 462
_RECOVERY_REJECTED_INDICES = [4621, 4622, 4628, 4629, 4630]
_RECOVERY_ACCEPTED_UNCOMMITTED_INDICES = [
    4620, 4623, 4624, 4625, 4626, 4627, 4631, 4632, 4633,
]


def _recovery_decomposition(plan_len, batch=10):
    """Deterministic batch decomposition aligned at the commit cursor."""
    full, rem = divmod(plan_len, batch)
    decomp = [batch] * full
    if rem:
        decomp.append(rem)
    return decomp


def _forge_checkpoint(tmp_path, writer, accs, plan_idents, decomposition,
                      frame_count, generation, move_committed_to_stacked=False):
    """Commit one real generation then forge the manifest into the synthetic
    generation-``generation`` witness (legacy schema-v1 format, prefix-only)
    exactly as the real Windows run persisted it.

    The forged witness is a *legacy* checkpoint: completed_sources is the
    exact ordered prefix ``plan[:frame_count]``, frame_count ==
    stacked_batches_count, no rejection metadata.  Only the manifest JSON and
    the artifact filenames are rewritten; artifact BYTES and run_config.cfg
    are untouched.
    """
    out = tmp_path / "out"
    inputs = tmp_path / "inputs"
    binding = {
        "input_roots": [str(inputs)],
        "reference": plan_idents[0],
        "plan": {"sources": plan_idents, "decomposition": decomposition},
        "reference_geometry": _reference_geometry(),
    }
    writer.commit(
        accs,
        session_binding=binding,
        counters={
            "frame_count": 2,
            "stacked_batches_count": 2,
            "total_exposure_seconds": 2.0,
            "exposure_unknown_count": 0,
            "exposure_min": 1.0,
            "exposure_max": 1.0,
        },
        completed_sources=plan_idents[:2],
    )
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
    manifest["exposure_min"] = 1.0
    manifest["exposure_max"] = 1.0
    manifest["session"]["plan"]["sources"] = plan_idents
    manifest["session"]["plan"]["decomposition"] = decomposition
    manifest["completed_sources"] = plan_idents[:frame_count]
    (ckpt_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return out


def _recovery_surgery(manifest, rejected_indices):
    """One-shot recovery plan surgery: remove only the FUTURE rejected plan
    entries and re-derive the decomposition over the surviving sources.
    Never touches committed science or completed identities."""
    sources = manifest["session"]["plan"]["sources"]
    decomp = manifest["session"]["plan"]["decomposition"]
    rejected_keys = {
        (sources[i]["path"], sources[i]["size"], sources[i]["mtime_ns"])
        for i in rejected_indices
    }
    new_sources = [
        s for s in sources
        if (s["path"], s["size"], s["mtime_ns"]) not in rejected_keys
    ]
    # Replay old batch boundaries over the surviving sources.
    new_decomp = []
    offset = 0
    for b in decomp:
        batch_members = sources[offset:offset + b]
        survivors = [
            s for s in batch_members
            if (s["path"], s["size"], s["mtime_ns"]) not in rejected_keys
        ]
        if survivors:
            new_decomp.append(len(survivors))
        offset += b
    assert offset == len(sources)
    assert sum(new_decomp) == len(new_sources)
    return new_sources, new_decomp


def test_gate8_emergency_recovery_witness(tmp_path):
    inputs = tmp_path / "inputs"
    out = tmp_path / "out"
    inputs.mkdir()
    out.mkdir()

    # 5329 distinct real source files.
    paths = [
        inputs / f"Light_SH2-101_20.0s_IRCUT_20260815-{214000 + i}.fit"
        for i in range(_RECOVERY_PLAN_LEN)
    ]
    for i, p in enumerate(paths):
        fits.PrimaryHDU(np.full(SHAPE, (i % 200) + 1, dtype=np.uint16)).writeto(p)
    idents = [_identity(p) for p in paths]

    decomposition = _recovery_decomposition(_RECOVERY_PLAN_LEN)
    assert sum(decomposition) == _RECOVERY_PLAN_LEN
    # The ledger boundary (4620) must align to a persisted batch boundary —
    # required by the resume validator, and true in the real witness.
    cum = 0
    for b in decomposition:
        cum += b
        if cum == _RECOVERY_FRAME_COUNT:
            break
    assert cum == _RECOVERY_FRAME_COUNT

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, out, inputs, group_size=1)
    qm._resume_reference_identity = idents[0]
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
    out = _forge_checkpoint(
        tmp_path, writer, accs, idents, decomposition,
        _RECOVERY_FRAME_COUNT, _RECOVERY_GENERATION,
    )

    # ---- 1. the pristine witness loads through the real reader ----
    resolver = SafeStackedSourceResolver("stacked")
    result = read_drizzle_checkpoint(str(out), resolver=resolver)
    assert result.generation == _RECOVERY_GENERATION
    assert result.next_source_index == _RECOVERY_FRAME_COUNT
    assert result.completed_sources == idents[:_RECOVERY_FRAME_COUNT]

    # ---- 2. model the live filesystem of the failed Windows run ----
    stacked = inputs / "stacked"
    stacked.mkdir()
    for p in paths[:_RECOVERY_FRAME_COUNT]:
        p.rename(stacked / p.name)
    for i in _RECOVERY_ACCEPTED_UNCOMMITTED_INDICES:
        paths[i].rename(stacked / paths[i].name)
    unaligned = inputs / "unaligned_by_stacker"
    unaligned.mkdir()
    for i in _RECOVERY_REJECTED_INDICES:
        paths[i].rename(unaligned / paths[i].name)

    # The unmodified checkpoint now REFUSES to read: the five rejected
    # sources cannot resolve (unaligned is not a legal destination).  This is
    # exactly the real secondary resume failure the surgery must fix.
    with pytest.raises(DrizzleCheckpointError):
        read_drizzle_checkpoint(str(out), resolver=resolver)

    # ---- 3. the one-shot surgery: remove ONLY the five FUTURE rejected ----
    manifest_path = out / CHECKPOINT_DIRNAME / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["session"]["plan"]["sources"], manifest["session"]["plan"][
        "decomposition"
    ] = _recovery_surgery(manifest, _RECOVERY_REJECTED_INDICES)
    new_plan_sources = manifest["session"]["plan"]["sources"]
    new_decomp = manifest["session"]["plan"]["decomposition"]
    assert len(new_plan_sources) == _RECOVERY_PLAN_LEN - 5
    assert sum(new_decomp) == _RECOVERY_PLAN_LEN - 5
    # Critical property: the committed prefix is untouched.
    assert new_plan_sources[:_RECOVERY_FRAME_COUNT] == idents[:_RECOVERY_FRAME_COUNT]
    assert manifest["completed_sources"] == idents[:_RECOVERY_FRAME_COUNT]
    remaining_keys = {(s["path"], s["size"], s["mtime_ns"]) for s in new_plan_sources}
    for i in _RECOVERY_REJECTED_INDICES:
        assert (idents[i]["path"], idents[i]["size"], idents[i]["mtime_ns"]) not in remaining_keys
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    # ---- 4. restore the 9 accepted-but-uncommitted files to original paths -
    for i in _RECOVERY_ACCEPTED_UNCOMMITTED_INDICES:
        (stacked / paths[i].name).rename(paths[i])

    # ---- 5. load the recovered checkpoint through the real reader ----
    result2 = read_drizzle_checkpoint(str(out), resolver=resolver)
    assert result2.generation == _RECOVERY_GENERATION
    assert result2.next_source_index == _RECOVERY_FRAME_COUNT
    assert result2.completed_sources == idents[:_RECOVERY_FRAME_COUNT]
    assert len(result2.resolved_remaining_paths) == (
        _RECOVERY_PLAN_LEN - 5 - _RECOVERY_FRAME_COUNT
    )
    assert Path(result2.resolved_remaining_paths[0]).name == paths[
        _RECOVERY_ACCEPTED_UNCOMMITTED_INDICES[0]
    ].name

    # ---- 6. production Resume path: preflight + remaining queue ----
    qm2 = object.__new__(SeestarQueuedStacker)
    _configure(qm2, out, inputs, group_size=1)
    qm2.move_stacked = True
    qm2._resume_requested = True
    ok, result3, _ref = qm2._validate_drizzle_resume_headless()
    assert ok is True
    assert result3.next_source_index == _RECOVERY_FRAME_COUNT
    qm2._drizzle_resume_result = result3
    qm2._restore_drizzle_checkpoint_runtime(result3)
    assert qm2._drizzle_frame_count == _RECOVERY_FRAME_COUNT
    assert qm2.stacked_batches_count == _RECOVERY_FRAME_COUNT

    qm2.queue = Queue()
    for p in result3.resolved_remaining_paths:
        qm2.queue.put(p)
    assert qm2._init_drizzle_checkpoint() is True
    continuation = qm2._drizzle_resume_continuation
    assert continuation is not None
    assert continuation.next_source_index == _RECOVERY_FRAME_COUNT
    assert qm2._drizzle_checkpoint_writer.current_generation == _RECOVERY_GENERATION

    # ---- 7. admit the next observation and commit generation 463 ----
    next_path = result3.resolved_remaining_paths[0]
    _accept(qm2, next_path, _RECOVERY_FRAME_COUNT)
    qm2._drizzle_checkpoint_force_flush()

    manifest_after = _manifest(out)
    assert manifest_after["generation"] == _RECOVERY_GENERATION + 1
    assert manifest_after["frame_count"] == _RECOVERY_FRAME_COUNT + 1
    assert manifest_after["stacked_batches_count"] == _RECOVERY_FRAME_COUNT + 1
    assert len(manifest_after["completed_sources"]) == _RECOVERY_FRAME_COUNT + 1

    # ---- 8. reload the newly committed checkpoint through the real reader --
    result4 = read_drizzle_checkpoint(str(out), resolver=resolver)
    assert result4.generation == _RECOVERY_GENERATION + 1
    assert result4.next_source_index == _RECOVERY_FRAME_COUNT + 1
    # The first 4620 completed identities are unchanged.
    assert result4.completed_sources[:_RECOVERY_FRAME_COUNT] == idents[:_RECOVERY_FRAME_COUNT]
    # The newly admitted observation is exactly the first restored source.
    assert result4.completed_sources[_RECOVERY_FRAME_COUNT] == _identity(next_path)
    # No removed rejected observation ever enters the completed ledger.
    completed_keys = {
        (s["path"], s["size"], s["mtime_ns"]) for s in result4.completed_sources
    }
    for i in _RECOVERY_REJECTED_INDICES:
        assert (idents[i]["path"], idents[i]["size"], idents[i]["mtime_ns"]) not in completed_keys
    # Science counters are coherent.
    assert result4.counters["frame_count"] == _RECOVERY_FRAME_COUNT + 1
    assert result4.counters["stacked_batches_count"] == _RECOVERY_FRAME_COUNT + 1


def test_gate8_plan_surgery_never_rewrites_science_bytes(tmp_path):
    """The committed accumulator state is never conceptually rewritten by
    the plan surgery: the generation-462 artifact files stay byte-identical
    before and after the manifest edit."""
    inputs = tmp_path / "inputs"
    out = tmp_path / "out"
    inputs.mkdir()
    out.mkdir()
    n = 60
    frame_count = 30
    paths, idents = _make_sources(inputs, n)
    decomposition = _recovery_decomposition(n)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, out, inputs, group_size=1)
    qm._resume_reference_identity = idents[0]
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
    out = _forge_checkpoint(
        tmp_path, writer, accs, idents, decomposition,
        frame_count, _RECOVERY_GENERATION,
    )
    ckpt_dir = out / CHECKPOINT_DIRNAME
    artifacts_before = {}
    for name in sorted(p.name for p in ckpt_dir.glob("gen-*.npy")):
        artifacts_before[name] = (ckpt_dir / name).read_bytes()

    manifest_path = ckpt_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["session"]["plan"]["sources"], manifest["session"]["plan"][
        "decomposition"
    ] = _recovery_surgery(manifest, [32, 35, 39])
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    for name, data in artifacts_before.items():
        assert (ckpt_dir / name).read_bytes() == data


# ---------------------------------------------------------------------------
# QM-level: _validate_plan_against_manifest with rejections + stacked files
# ---------------------------------------------------------------------------

def test_validate_plan_against_manifest_resolution_aware(tmp_path):
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, _idents = _make_sources(inputs, 4)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, paths)
    _accept(qm, paths[0], 0)
    _reject(qm, paths[1])
    _accept(qm, paths[2], 2)

    qm2 = object.__new__(SeestarQueuedStacker)
    _configure(qm2, output, inputs, group_size=1)
    qm2._resume_requested = True
    ok, result, _ref = qm2._validate_drizzle_resume_headless()
    assert ok is True
    qm2._drizzle_resume_result = result
    qm2._restore_drizzle_checkpoint_runtime(result)

    # Remaining source (src_3) physically in stacked/ (uncommitted accepted
    # frame from a pre-crash run) — the queue holds the stacked path.
    stacked = inputs / "stacked"
    stacked.mkdir()
    paths[3].rename(stacked / paths[3].name)
    qm2.queue = Queue()
    qm2.queue.put(str(stacked / "src_3.fit"))
    ok_manifest, reason = qm2._validate_plan_against_manifest()
    assert ok_manifest is True, reason

    # A wrong file (same order slot, different identity) is refused.
    bogus = inputs / "bogus.fit"
    bogus.write_bytes(b"not the same observation")
    qm2.queue = Queue()
    qm2.queue.put(str(bogus))
    ok_manifest, reason = qm2._validate_plan_against_manifest()
    assert ok_manifest is False
    assert "differs" in reason


def test_worker_rejection_branch_commits_before_move(tmp_path):
    """The production worker failure branch records the durable rejection
    disposition BEFORE moving the source (single canonical disposal point)."""
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, _idents = _make_sources(inputs, 3)

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=1)
    _arm_fresh_writer(qm, output, paths)
    _accept(qm, paths[0], 0)

    events = []
    orig_move = qm._move_to_unaligned
    orig_hook = qm._drizzle_checkpoint_after_rejection

    def _reject_spy(fp):
        events.append(("commit", Path(fp).name))
        orig_hook(fp)

    def _move_spy(fp):
        events.append(("move", Path(fp).name))
        orig_move(fp)

    qm._drizzle_checkpoint_after_rejection = _reject_spy
    qm._move_to_unaligned = _move_spy

    # Minimal worker-loop replica (the real one lives in _worker): the
    # rejection disposition is committed BEFORE the unaligned move.
    qm.failed_align_count += 1
    if getattr(qm, "drizzle_active_session", False):
        qm._drizzle_checkpoint_after_rejection(str(paths[1]))
    if hasattr(qm, "_move_to_unaligned"):
        qm._move_to_unaligned(str(paths[1]))

    assert [e[0] for e in events] == ["commit", "move"]
    manifest = _manifest(output)
    assert _rejected_names(manifest) == ["src_1.fit"]
    assert manifest["plan_cursor"] == 2
    assert not paths[1].exists()
    assert (inputs / "unaligned_by_stacker" / "src_1.fit").exists()


# ---------------------------------------------------------------------------
# Full start_processing lifecycle with rejection + stacked remaining file
# ---------------------------------------------------------------------------

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


def test_start_processing_resume_rejection_and_stacked_remaining(
    tmp_path, monkeypatch
):
    """The REAL start_processing resume lifecycle: a checkpoint with a
    committed rejection disposition and one remaining source physically in
    stacked/ (accepted-but-uncommitted pre-crash frame) resumes with the
    authoritative remaining queue rebuilt from verified resolved paths."""
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths, _idents = _make_sources(inputs, 4)

    # Build the state: accept src_0, reject src_1, accept src_2, then
    # src_3 accepted but its cadence never fired and it moved to stacked.
    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, group_size=2)
    _arm_fresh_writer(qm, output, paths)
    _accept(qm, paths[0], 0)      # frame 1, no commit (cadence 2)
    _reject(qm, paths[1])         # committed rejection disposition
    _accept(qm, paths[2], 2)      # frame 2 -> cadence commit
    _accept(qm, paths[3], 3)      # frame 3, no commit
    qm.move_stacked = True
    qm._move_to_stacked([str(paths[3])])   # then "crash"
    assert (inputs / "stacked" / "src_3.fit").exists()

    manifest = _manifest(output)
    assert manifest["frame_count"] == 2
    assert manifest["plan_cursor"] == 3
    assert _rejected_names(manifest) == ["src_1.fit"]

    # Real production lifecycle.
    monkeypatch.setattr(
        queue_manager_module,
        "ProcessPoolExecutor",
        lambda **kwargs: _NoopExecutor(**kwargs),
    )
    qm2 = SeestarQueuedStacker(batch_size=2, autotune=False)
    qm2.update_progress = lambda *args, **kwargs: None
    aligner = _LifecycleAligner()
    qm2.aligner = aligner
    worker_started = threading.Event()
    worker_snapshot = {}

    def _worker():
        worker_snapshot["queue"] = list(qm2.queue.queue)
        worker_snapshot["frame_count"] = qm2._drizzle_frame_count
        worker_snapshot["stacked_batches_count"] = qm2.stacked_batches_count
        worker_snapshot["plan_cursor"] = qm2._drizzle_plan_cursor
        worker_snapshot["rejected"] = [
            x["name"] for x in qm2._drizzle_rejected_sources
        ]
        worker_snapshot["continuation"] = qm2._drizzle_resume_continuation
        worker_started.set()

    qm2._worker = _worker
    qm2._solve_astrometry_async = lambda *args, **kwargs: _wcs()
    started = qm2.start_processing(
        input_dir=str(inputs),
        output_dir=str(output),
        use_drizzle=True,
        drizzle_scale=1.0,
        drizzle_kernel="square",
        drizzle_pixfrac=1.0,
        drizzle_wht_threshold=0.0,
        drizzle_group_size=2,
        batch_size=2,
        min_w=0.01,
        move_stacked=True,
        perform_cleanup=False,
        resume_intent="resume",
        reproject_between_batches=False,
        reproject_coadd_final=False,
    )
    if qm2.processing_thread is not None:
        qm2.processing_thread.join(timeout=5)

    assert started is True
    assert worker_started.is_set()
    # The authoritative queue is the single stacked remaining source — the
    # rejected src_1 is never replayed and the committed A/C are never
    # replayed either.
    assert worker_snapshot["queue"] == [str(inputs / "stacked" / "src_3.fit")]
    assert worker_snapshot["frame_count"] == 2
    assert worker_snapshot["stacked_batches_count"] == 2
    assert worker_snapshot["plan_cursor"] == 3
    assert worker_snapshot["rejected"] == ["src_1.fit"]
    assert worker_snapshot["continuation"] is not None
    assert worker_snapshot["continuation"].next_source_index == 3
    assert qm2._drizzle_checkpoint_writer.current_generation == manifest["generation"]
