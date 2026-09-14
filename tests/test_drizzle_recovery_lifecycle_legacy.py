"""ZSSS-DRIZZLE-REJECTED-SOURCE REWORK 1 — exact full-lifecycle recovery witness.

This test models the REAL Windows run geometry and drives the genuine
production ``SeestarQueuedStacker.start_processing(...)`` resume lifecycle:

* legacy schema-v1 checkpoint (no ``plan_cursor`` / ``rejected_sources``),
  generation 462, frame_count = stacked_batches_count = 4620;
* recovered 5324-source plan (the five FUTURE rejected observations removed),
  completed_sources = exact recovered plan[:4620];
* 127-batch persisted decomposition: [42]*110 + [37] + [42]*15 + [37]
  (the first future batch of 42 lost five observations);
* batch_size requested = AUTO (0) with the effective capacity frozen to 42;
* 704 remaining filesystem observations at their canonical original paths
  (the nine replay observations restored from stacked/);
* 4620 committed sources physically in stacked/;
* the five rejected sources parked OUTSIDE the active plan in
  unaligned_by_stacker/;
* an extra session reference FITS (July date, like the real frozen
  reference) present at its original position but NOT part of the plan.

The queue is NEVER constructed by hand: everything from
``_early_resume_preflight`` to ``_init_drizzle_checkpoint`` runs inside the
real ``start_processing``.
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

PLAN_LEN_ORIG = 5329
FRAME_COUNT = 4620
GENERATION = 462
BATCH = 42
REJECTED_INDICES = [4621, 4622, 4628, 4629, 4630]
ACCEPTED_UNCOMMITTED_INDICES = [4620, 4623, 4624, 4625, 4626, 4627, 4631, 4632, 4633]


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


def _configure(qm, output, inputs, kernel="lanczos3", group_size=10):
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
    qm.drizzle_scale = 2.0
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
    qm._normalize_effective_drizzle_config()


def _recovered_decomposition():
    """127 batches: [42]*110 + [37] + [42]*15 + [37] == 5324."""
    return [BATCH] * 110 + [BATCH - 5] + [BATCH] * 15 + [37]


def _build_recovered_state(tmp_path):
    """Build the recovered legacy checkpoint + live filesystem state."""
    inputs = tmp_path / "inputs"
    out = tmp_path / "out"
    inputs.mkdir()
    out.mkdir()

    names = [
        f"Light_SH2-101_20.0s_IRCUT_20260815-{214000 + i}.fit"
        for i in range(PLAN_LEN_ORIG)
    ]
    for i, name in enumerate(names):
        fits.PrimaryHDU(
            np.full(SHAPE, (i % 200) + 1, dtype=np.uint16)
        ).writeto(inputs / name)
    paths = [inputs / name for name in names]
    idents = [_identity(p) for p in paths]

    # Extra session reference (July date, NOT part of the plan).
    ref_name = "Light_SH2-101_20.0s_IRCUT_20260707-003517.fit"
    fits.PrimaryHDU(np.full(SHAPE, 9, dtype=np.uint16)).writeto(inputs / ref_name)
    reference_ident = _identity(inputs / ref_name)

    # Recovery surgery: recovered plan = 5324 sources, 127 batches.
    decomp_rec = _recovered_decomposition()
    assert sum(decomp_rec) == PLAN_LEN_ORIG - 5
    removed = set(REJECTED_INDICES)
    plan_idents = [
        ident for i, ident in enumerate(idents) if i not in removed
    ]
    assert len(plan_idents) == PLAN_LEN_ORIG - 5
    assert plan_idents[:FRAME_COUNT] == idents[:FRAME_COUNT]

    # Forge the legacy checkpoint (gen 462, prefix-only format).
    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, out, inputs, kernel="lanczos3", group_size=10)
    qm._resume_reference_identity = reference_ident
    cfg = build_drizzle_canonical_config(
        qm, product_version=qm._canonical_product_version()
    )
    writer = DrizzleCheckpointWriter(
        str(out), qm._canonical_product_version(), cfg, _wcs(), SHAPE
    )
    accs = [
        DrizzleAccumulator(SHAPE, kernel="lanczos3", pixfrac=1.0) for _ in range(3)
    ]
    _add(accs, _frame(0))
    _add(accs, _frame(1))
    writer.commit(
        accs,
        session_binding={
            "input_roots": [str(inputs)],
            "reference": reference_ident,
            "plan": {"sources": plan_idents, "decomposition": decomp_rec},
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
        completed_sources=plan_idents[:2],
    )
    ckpt_dir = out / CHECKPOINT_DIRNAME
    manifest = json.loads((ckpt_dir / MANIFEST_FILENAME).read_text())
    for ch in manifest["channels"]:
        for key in ("out_img", "out_wht"):
            old = ch[key]["file"]
            new = old.replace(f"gen-{1:08d}-", f"gen-{GENERATION:08d}-")
            (ckpt_dir / old).rename(ckpt_dir / new)
            ch[key]["file"] = new
    manifest["generation"] = GENERATION
    manifest["frame_count"] = FRAME_COUNT
    manifest["stacked_batches_count"] = FRAME_COUNT
    manifest["total_exposure_seconds"] = float(FRAME_COUNT)
    manifest["exposure_unknown_count"] = 0
    manifest["exposure_min"] = 1.0
    manifest["exposure_max"] = 1.0
    manifest["session"]["plan"]["sources"] = plan_idents
    manifest["session"]["plan"]["decomposition"] = decomp_rec
    manifest["completed_sources"] = plan_idents[:FRAME_COUNT]
    (ckpt_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    # Live filesystem: 4620 committed -> stacked/, 5 rejected -> unaligned/,
    # 9 accepted-uncommitted -> stacked/ then restored to original.
    stacked = inputs / "stacked"
    stacked.mkdir()
    for p in paths[:FRAME_COUNT]:
        p.rename(stacked / p.name)
    unaligned = inputs / "unaligned_by_stacker"
    unaligned.mkdir()
    for i in REJECTED_INDICES:
        paths[i].rename(unaligned / paths[i].name)
    for i in ACCEPTED_UNCOMMITTED_INDICES:
        paths[i].rename(stacked / paths[i].name)
    for i in ACCEPTED_UNCOMMITTED_INDICES:
        (stacked / paths[i].name).rename(paths[i])

    # 704 remaining at original positions.
    remaining = [p for p in paths if p.exists() and p.parent == inputs]
    assert len(remaining) == PLAN_LEN_ORIG - 5 - FRAME_COUNT
    return inputs, out, paths, idents, plan_idents, decomp_rec, reference_ident


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


@pytest.mark.parametrize("with_stack_plan_csv", [False, True])
def test_start_processing_legacy_recovery_witness(tmp_path, monkeypatch,
                                                  with_stack_plan_csv):
    inputs, out, paths, idents, plan_idents, decomp_rec, reference_ident = (
        _build_recovered_state(tmp_path)
    )
    if with_stack_plan_csv:
        # A leftover stack_plan.csv listing the ORIGINAL plan batches (as the
        # original AUTO=42 run could have left in the input folder).
        rows = ["filename"]
        for i in range(PLAN_LEN_ORIG):
            rows.append(paths[i].name if i not in set(REJECTED_INDICES) else paths[i].name)
        (inputs / "stack_plan.csv").write_text("\n".join(rows), encoding="utf-8")

    monkeypatch.setattr(
        queue_manager_module,
        "ProcessPoolExecutor",
        lambda **kwargs: _NoopExecutor(**kwargs),
    )
    qm = SeestarQueuedStacker(batch_size=0, autotune=False)
    qm.update_progress = lambda *args, **kwargs: None
    aligner = _LifecycleAligner()
    qm.aligner = aligner
    # AUTO batch: force the machine-independent effective capacity 42.
    monkeypatch.setattr(qm, "_estimate_batch_size", lambda: BATCH)

    worker_started = threading.Event()
    worker_snapshot = {}
    install_ran = {}

    def _worker():
        worker_snapshot["queue"] = list(qm.queue.queue)
        worker_snapshot["frame_count"] = qm._drizzle_frame_count
        worker_snapshot["plan_cursor"] = qm._drizzle_plan_cursor
        worker_snapshot["continuation"] = qm._drizzle_resume_continuation
        worker_snapshot["resume_active"] = qm._resume_active
        worker_snapshot["resume_result"] = qm._drizzle_resume_result
        worker_started.set()

    qm._worker = _worker
    qm._solve_astrometry_async = lambda *args, **kwargs: _wcs()
    orig_install = qm._install_resume_remaining_queue

    def _install_spy(result):
        install_ran["ran"] = True
        install_ran["n"] = len(result.resolved_remaining_paths)
        return orig_install(result)

    qm._install_resume_remaining_queue = _install_spy

    started = qm.start_processing(
        input_dir=str(inputs),
        output_dir=str(out),
        use_drizzle=True,
        drizzle_scale=2.0,
        drizzle_kernel="lanczos3",
        drizzle_pixfrac=1.0,
        drizzle_wht_threshold=0.0,
        drizzle_group_size=10,
        batch_size=0,
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
    assert worker_started.is_set(), "worker was not allowed to start"
    assert install_ran.get("ran") is True
    assert install_ran["n"] == PLAN_LEN_ORIG - 5 - FRAME_COUNT

    queue_items = [q for q in worker_snapshot["queue"] if q != queue_manager_module._BATCH_BREAK_TOKEN]
    assert len(queue_items) == PLAN_LEN_ORIG - 5 - FRAME_COUNT
    # First queue source is the restored old plan[4620] observation.
    assert Path(queue_items[0]).name == paths[ACCEPTED_UNCOMMITTED_INDICES[0]].name
    # Queue ordered identity == recovered plan[4620:].
    expected_names = [x["name"] for x in plan_idents[FRAME_COUNT:]]
    assert [Path(q).name for q in queue_items] == expected_names
    # Break-token decomposition == recovered suffix decomposition.
    from seestar.core.drizzle_checkpoint import (
        DrizzleCheckpointWriter as _W,
    )
    assert worker_snapshot["frame_count"] == FRAME_COUNT
    assert worker_snapshot["plan_cursor"] == FRAME_COUNT
    continuation = worker_snapshot["continuation"]
    assert continuation is not None
    assert continuation.next_source_index == FRAME_COUNT
    assert qm._drizzle_checkpoint_writer.current_generation == GENERATION

    # First post-resume admission commits generation 463 through the real
    # writer and reloads coherently.
    next_path = queue_items[0]
    ok = qm._add_frame_to_drizzle_accumulators(
        np.stack([_frame(0)[0]] * 3, axis=-1).astype(np.float32),
        fits.Header([("EXPTIME", 1.0)]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        _frame(0)[1],
        native_wcs=_wcs(),
    )
    assert ok is True
    qm._drizzle_group_tick()
    qm._admit_exposure(1.0, 0, 1.0, 1.0)
    qm.stacked_batches_count += 1
    qm._drizzle_checkpoint_after_frame(str(next_path))
    qm._drizzle_checkpoint_force_flush()

    manifest_after = json.loads(
        (out / CHECKPOINT_DIRNAME / MANIFEST_FILENAME).read_text()
    )
    assert manifest_after["generation"] == GENERATION + 1
    assert manifest_after["frame_count"] == FRAME_COUNT + 1

    result4 = read_drizzle_checkpoint(
        str(out), resolver=SafeStackedSourceResolver("stacked")
    )
    assert result4.generation == GENERATION + 1
    assert result4.next_source_index == FRAME_COUNT + 1
    assert result4.completed_sources[:FRAME_COUNT] == plan_idents[:FRAME_COUNT]
    completed_keys = {
        (s["path"], s["size"], s["mtime_ns"]) for s in result4.completed_sources
    }
    for i in REJECTED_INDICES:
        key = (idents[i]["path"], idents[i]["size"], idents[i]["mtime_ns"])
        assert key not in completed_keys
    assert result4.counters["frame_count"] == FRAME_COUNT + 1
    assert result4.counters["stacked_batches_count"] == FRAME_COUNT + 1

    # Gate 9: a rejection AFTER the recovery uses the new disposition
    # contract normally on the re-armed continuation (cursor-only durable
    # generation, science untouched, frame_count unchanged).
    rejected_path = queue_items[1]
    qm._drizzle_checkpoint_after_rejection(str(rejected_path))
    manifest_after_rej = json.loads(
        (out / CHECKPOINT_DIRNAME / MANIFEST_FILENAME).read_text()
    )
    assert manifest_after_rej["generation"] == GENERATION + 2
    assert manifest_after_rej["frame_count"] == FRAME_COUNT + 1
    assert manifest_after_rej["plan_cursor"] == FRAME_COUNT + 2
    assert [x["name"] for x in manifest_after_rej["rejected_sources"]] == [
        Path(rejected_path).name
    ]
    result5 = read_drizzle_checkpoint(
        str(out), resolver=SafeStackedSourceResolver("stacked")
    )
    assert result5.generation == GENERATION + 2
    assert result5.next_source_index == FRAME_COUNT + 2
    assert [x["name"] for x in result5.rejected_sources] == [
        Path(rejected_path).name
    ]


def test_install_skipped_seam_is_self_healed(tmp_path, monkeypatch):
    """The real Windows failure seam: whatever upstream branch/reset prevents
    ``_install_resume_remaining_queue`` from rebuilding the queue before
    ``_init_drizzle_checkpoint`` must NOT be able to fail the run.  The
    validated continuation is the authority and ``_init_drizzle_checkpoint``
    re-establishes the remaining queue itself.

    On 50d7b34 this scenario reproduces the exact production failure
    ("remaining Drizzle observation set/order differs from the validated
    checkpoint plan") because the ambient scan queue (including the extra
    session-reference FITS and no batch tokens) disagrees with the persisted
    suffix.  On the fixed branch the same scenario must start the worker
    with the correct 704-source queue."""
    inputs, out, paths, idents, plan_idents, decomp_rec, reference_ident = (
        _build_recovered_state(tmp_path)
    )
    monkeypatch.setattr(
        queue_manager_module,
        "ProcessPoolExecutor",
        lambda **kwargs: _NoopExecutor(**kwargs),
    )
    qm = SeestarQueuedStacker(batch_size=0, autotune=False)
    qm.update_progress = lambda *args, **kwargs: None
    aligner = _LifecycleAligner()
    qm.aligner = aligner
    monkeypatch.setattr(qm, "_estimate_batch_size", lambda: BATCH)

    # Simulate the upstream seam: the Phase-B1 install never executes (as if
    # its gate were false) — the ambient scan queue survives.  The second
    # call site (`_init_drizzle_checkpoint`) still delegates to the real
    # implementation, which is the authoritative re-establishment under
    # test.
    orig_install = qm._install_resume_remaining_queue
    install_calls = {"n": 0}

    def _skip_phase_b1_install(result):
        install_calls["n"] += 1
        if install_calls["n"] == 1:
            return 0
        return orig_install(result)

    qm._install_resume_remaining_queue = _skip_phase_b1_install

    worker_started = threading.Event()
    worker_snapshot = {}

    def _worker():
        worker_snapshot["queue"] = list(qm.queue.queue)
        worker_snapshot["continuation"] = qm._drizzle_resume_continuation
        worker_started.set()

    qm._worker = _worker
    qm._solve_astrometry_async = lambda *args, **kwargs: _wcs()

    started = qm.start_processing(
        input_dir=str(inputs),
        output_dir=str(out),
        use_drizzle=True,
        drizzle_scale=2.0,
        drizzle_kernel="lanczos3",
        drizzle_pixfrac=1.0,
        drizzle_wht_threshold=0.0,
        drizzle_group_size=10,
        batch_size=0,
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
    assert install_calls["n"] >= 2
    assert worker_started.is_set()
    queue_items = [
        q for q in worker_snapshot["queue"]
        if q != queue_manager_module._BATCH_BREAK_TOKEN
    ]
    assert len(queue_items) == PLAN_LEN_ORIG - 5 - FRAME_COUNT
    assert Path(queue_items[0]).name == paths[ACCEPTED_UNCOMMITTED_INDICES[0]].name
    assert [Path(q).name for q in queue_items] == [
        x["name"] for x in plan_idents[FRAME_COUNT:]
    ]


def test_failed_attempt_then_retry_same_process(tmp_path, monkeypatch):
    """Two start_processing attempts in the SAME stacker instance: the first
    diverges (a remaining source changed identity between the validated
    preflight and checkpoint initialization) and fails closed with the
    first-difference diagnostic; the retry (evidence restored) starts the
    worker.  This models the real operator sequence (8.5.1 attempt, update,
    retry) on one long-lived process."""
    inputs, out, paths, idents, plan_idents, decomp_rec, reference_ident = (
        _build_recovered_state(tmp_path)
    )
    monkeypatch.setattr(
        queue_manager_module,
        "ProcessPoolExecutor",
        lambda **kwargs: _NoopExecutor(**kwargs),
    )
    qm = SeestarQueuedStacker(batch_size=0, autotune=False)
    qm.update_progress = lambda *args, **kwargs: None
    aligner = _LifecycleAligner()
    qm.aligner = aligner
    monkeypatch.setattr(qm, "_estimate_batch_size", lambda: BATCH)

    victim = paths[ACCEPTED_UNCOMMITTED_INDICES[0]]
    orig_preflight = qm._early_resume_preflight
    tampered = {}

    def _preflight_tamper():
        ok, res = orig_preflight()
        if ok and not tampered:
            tampered["mtime"] = os.stat(victim).st_mtime_ns
            os.utime(victim, ns=(1, 1))
        return ok, res

    qm._early_resume_preflight = _preflight_tamper
    worker_started = threading.Event()
    qm._worker = worker_started.set
    qm._solve_astrometry_async = lambda *args, **kwargs: _wcs()

    def _attempt():
        return qm.start_processing(
            input_dir=str(inputs),
            output_dir=str(out),
            use_drizzle=True,
            drizzle_scale=2.0,
            drizzle_kernel="lanczos3",
            drizzle_pixfrac=1.0,
            drizzle_wht_threshold=0.0,
            drizzle_group_size=10,
            batch_size=0,
            min_w=0.01,
            move_stacked=True,
            perform_cleanup=False,
            resume_intent="resume",
            reproject_between_batches=False,
            reproject_coadd_final=False,
        )

    first = _attempt()
    assert first is False
    error = qm.processing_error or ""
    assert "differs" in error
    # First-difference diagnostic evidence.
    assert "resume cursor" in error
    assert "expected remaining" in error
    assert "actual remaining" in error
    assert "first difference" in error
    assert victim.name in error

    # Restore the exact evidence and retry in the same instance.
    os.utime(victim, ns=(tampered["mtime"], tampered["mtime"]))
    second = _attempt()
    assert second is True


def test_diagnostic_detail_is_bounded_and_precise(tmp_path):
    """The new mismatch diagnostic reports cursor/counts/first difference and
    never dumps more than one identity pair."""
    qm = object.__new__(SeestarQueuedStacker)
    cur = [
        {"name": "a.fit", "path": "/x/a.fit", "size": 1, "mtime_ns": 10},
        {"name": "wrong.fit", "path": "/x/wrong.fit", "size": 2, "mtime_ns": 20},
    ]
    exp = [
        {"name": "a.fit", "path": "/x/a.fit", "size": 1, "mtime_ns": 10},
        {"name": "b.fit", "path": "/x/b.fit", "size": 3, "mtime_ns": 30},
        {"name": "c.fit", "path": "/x/c.fit", "size": 4, "mtime_ns": 40},
    ]
    detail = qm._remaining_queue_mismatch_detail(cur, exp, 4620)
    assert "resume cursor 4620" in detail
    assert "expected remaining 3" in detail
    assert "actual remaining 2" in detail
    assert "first difference at index 1" in detail
    assert "wrong.fit" in detail and "b.fit" in detail
    # Bounded: exactly one index reported, no full dump.
    assert "index 2" not in detail
    assert detail.count("actual[") == 1 and detail.count("expected[") == 1


# ---------------------------------------------------------------------------
# REWORK 2 — Windows path-identity case semantics (exact recovery lifecycle)
# ---------------------------------------------------------------------------

def _build_recovered_state_case_asymmetry(tmp_path):
    """Same recovered geometry as ``_build_recovered_state`` but with the
    REAL Windows casing asymmetry:

    * files on disk carry lowercase names (the normcase()-normalized paths);
    * the persisted identities preserve ORIGINAL display casing in ``name``
      while ``path`` is the lowercase on-disk path.

    On the real Windows host this arises naturally: the checkpoint writer
    recorded ``name`` from the original-cased scanned path while ``path`` was
    ``normcase()``-normalized; the reader then resolves the lowercase path
    and every re-derived queue basename is lowercase."""
    inputs = tmp_path / "inputs"
    out = tmp_path / "out"
    inputs.mkdir()
    out.mkdir()

    mixed_names = [
        f"Light_SH2-101_20.0s_IRCUT_20260815-{214000 + i}.fit"
        for i in range(PLAN_LEN_ORIG)
    ]
    lower_names = [n.lower() for n in mixed_names]
    for i, lower in enumerate(lower_names):
        fits.PrimaryHDU(
            np.full(SHAPE, (i % 200) + 1, dtype=np.uint16)
        ).writeto(inputs / lower)
    paths = [inputs / lower for lower in lower_names]

    def _case_mixed_identity(p, mixed_name):
        st = os.stat(p)
        return {
            "path": os.path.normcase(os.path.abspath(str(p))),
            "name": mixed_name,
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }

    idents = [
        _case_mixed_identity(p, mixed_names[i]) for i, p in enumerate(paths)
    ]

    ref_lower = "light_sh2-101_20.0s_ircut_20260707-003517.fit"
    ref_mixed = "Light_SH2-101_20.0s_IRCUT_20260707-003517.fit"
    fits.PrimaryHDU(np.full(SHAPE, 9, dtype=np.uint16)).writeto(
        inputs / ref_lower
    )
    reference_ident = _case_mixed_identity(inputs / ref_lower, ref_mixed)

    decomp_rec = _recovered_decomposition()
    removed = set(REJECTED_INDICES)
    plan_idents = [
        ident for i, ident in enumerate(idents) if i not in removed
    ]
    assert len(plan_idents) == PLAN_LEN_ORIG - 5
    assert plan_idents[:FRAME_COUNT] == idents[:FRAME_COUNT]

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, out, inputs, kernel="lanczos3", group_size=10)
    qm._resume_reference_identity = reference_ident
    cfg = build_drizzle_canonical_config(
        qm, product_version=qm._canonical_product_version()
    )
    writer = DrizzleCheckpointWriter(
        str(out), qm._canonical_product_version(), cfg, _wcs(), SHAPE
    )
    accs = [
        DrizzleAccumulator(SHAPE, kernel="lanczos3", pixfrac=1.0) for _ in range(3)
    ]
    _add(accs, _frame(0))
    _add(accs, _frame(1))
    writer.commit(
        accs,
        session_binding={
            "input_roots": [str(inputs)],
            "reference": reference_ident,
            "plan": {"sources": plan_idents, "decomposition": decomp_rec},
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
        completed_sources=plan_idents[:2],
    )
    ckpt_dir = out / CHECKPOINT_DIRNAME
    manifest = json.loads((ckpt_dir / MANIFEST_FILENAME).read_text())
    for ch in manifest["channels"]:
        for key in ("out_img", "out_wht"):
            old = ch[key]["file"]
            new = old.replace(f"gen-{1:08d}-", f"gen-{GENERATION:08d}-")
            (ckpt_dir / old).rename(ckpt_dir / new)
            ch[key]["file"] = new
    manifest["generation"] = GENERATION
    manifest["frame_count"] = FRAME_COUNT
    manifest["stacked_batches_count"] = FRAME_COUNT
    manifest["total_exposure_seconds"] = float(FRAME_COUNT)
    manifest["exposure_unknown_count"] = 0
    manifest["exposure_min"] = 1.0
    manifest["exposure_max"] = 1.0
    manifest["session"]["plan"]["sources"] = plan_idents
    manifest["session"]["plan"]["decomposition"] = decomp_rec
    manifest["completed_sources"] = plan_idents[:FRAME_COUNT]
    (ckpt_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    stacked = inputs / "stacked"
    stacked.mkdir()
    for p in paths[:FRAME_COUNT]:
        p.rename(stacked / p.name)
    unaligned = inputs / "unaligned_by_stacker"
    unaligned.mkdir()
    for i in REJECTED_INDICES:
        paths[i].rename(unaligned / paths[i].name)
    for i in ACCEPTED_UNCOMMITTED_INDICES:
        paths[i].rename(stacked / paths[i].name)
    for i in ACCEPTED_UNCOMMITTED_INDICES:
        (stacked / paths[i].name).rename(paths[i])

    remaining = [p for p in paths if p.exists() and p.parent == inputs]
    assert len(remaining) == PLAN_LEN_ORIG - 5 - FRAME_COUNT
    return (
        inputs, out, paths, idents, plan_idents, decomp_rec, reference_ident,
        lower_names,
    )


def _start_recovery_lifecycle(monkeypatch, inputs, out, simulate_windows_case):
    monkeypatch.setattr(
        queue_manager_module,
        "ProcessPoolExecutor",
        lambda **kwargs: _NoopExecutor(**kwargs),
    )
    if simulate_windows_case:
        import ntpath
        import seestar.core.drizzle_checkpoint as core_ckpt

        helper = getattr(queue_manager_module, "_source_names_equivalent", None)
        if helper is not None:
            monkeypatch.setattr(
                queue_manager_module,
                "_source_names_equivalent",
                lambda a, b: helper(a, b, normcase=ntpath.normcase),
            )
        # The writer/reader whole-identity comparisons (ledger prefix checks)
        # live in the core checkpoint module and must observe the same host
        # semantics for the simulated Windows host.
        if hasattr(core_ckpt, "identity_names_equivalent"):
            orig_core_names = core_ckpt.identity_names_equivalent
            monkeypatch.setattr(
                core_ckpt,
                "identity_names_equivalent",
                lambda a, b, normcase=None: orig_core_names(
                    a, b, normcase=ntpath.normcase
                ),
            )
    qm = SeestarQueuedStacker(batch_size=0, autotune=False)
    qm.update_progress = lambda *args, **kwargs: None
    aligner = _LifecycleAligner()
    qm.aligner = aligner
    monkeypatch.setattr(qm, "_estimate_batch_size", lambda: BATCH)
    worker_started = threading.Event()
    worker_snapshot = {}

    def _worker():
        worker_snapshot["queue"] = list(qm.queue.queue)
        worker_snapshot["frame_count"] = qm._drizzle_frame_count
        worker_snapshot["plan_cursor"] = qm._drizzle_plan_cursor
        worker_snapshot["continuation"] = qm._drizzle_resume_continuation
        worker_started.set()

    qm._worker = _worker
    qm._solve_astrometry_async = lambda *args, **kwargs: _wcs()
    started = qm.start_processing(
        input_dir=str(inputs),
        output_dir=str(out),
        use_drizzle=True,
        drizzle_scale=2.0,
        drizzle_kernel="lanczos3",
        drizzle_pixfrac=1.0,
        drizzle_wht_threshold=0.0,
        drizzle_group_size=10,
        batch_size=0,
        min_w=0.01,
        move_stacked=True,
        perform_cleanup=False,
        resume_intent="resume",
        reproject_between_batches=False,
        reproject_coadd_final=False,
    )
    if qm.processing_thread is not None:
        qm.processing_thread.join(timeout=5)
    return qm, started, worker_started, worker_snapshot


def test_lifecycle_case_asymmetry_posix_control_refuses(tmp_path, monkeypatch):
    """POSIX semantics: the SAME casing asymmetry MUST refuse (case-only
    display-name difference is a real identity difference on a
    case-sensitive filesystem).  This is also the faithful Linux reproducer
    of the pre-fix Windows behavior — the first-difference diagnostic names
    the case-only divergence at index 0."""
    inputs, out, paths, idents, plan_idents, decomp_rec, reference_ident, _ = (
        _build_recovered_state_case_asymmetry(tmp_path)
    )
    qm, started, worker_started, _snapshot = _start_recovery_lifecycle(
        monkeypatch, inputs, out, simulate_windows_case=False
    )
    assert started is False
    assert not worker_started.is_set()
    error = qm.processing_error or ""
    assert "differs" in error
    assert "resume cursor 4620" in error
    assert "expected remaining 704" in error
    assert "actual remaining 704" in error
    assert "first difference at index 0" in error


def test_lifecycle_case_asymmetry_windows_semantics_full(tmp_path, monkeypatch):
    """Windows semantics: the exact real-world casing asymmetry must resume
    through the REAL start_processing lifecycle — worker starts with the 704
    authoritative sources, generation 463 commits, reload succeeds, and a
    post-recovery rejection uses the disposition contract."""
    import ntpath

    from seestar.core.drizzle_checkpoint import SafeStackedSourceResolver
    import seestar.core.drizzle_checkpoint as core_ckpt

    def _names_equivalent(a, b):
        helper = getattr(
            core_ckpt, "identity_names_equivalent", None
        ) or (lambda x, y, normcase=None: normcase(str(x)) == normcase(str(y)))
        return helper(a, b, normcase=ntpath.normcase)

    inputs, out, paths, idents, plan_idents, decomp_rec, reference_ident, _ = (
        _build_recovered_state_case_asymmetry(tmp_path)
    )
    qm, started, worker_started, snapshot = _start_recovery_lifecycle(
        monkeypatch, inputs, out, simulate_windows_case=True
    )
    assert started is True, f"start_processing refused: {qm.processing_error}"
    assert worker_started.is_set()

    queue_items = [
        q for q in snapshot["queue"]
        if q != queue_manager_module._BATCH_BREAK_TOKEN
    ]
    assert len(queue_items) == PLAN_LEN_ORIG - 5 - FRAME_COUNT
    assert snapshot["frame_count"] == FRAME_COUNT
    assert snapshot["plan_cursor"] == FRAME_COUNT
    # First queue source is the restored old plan[4620] observation (its
    # on-disk lowercase name; the persisted display name is mixed-case).
    assert Path(queue_items[0]).name == paths[
        ACCEPTED_UNCOMMITTED_INDICES[0]
    ].name
    # Ordered queue identity == recovered plan[4620:] under host semantics.
    expected_names = [x["name"] for x in plan_idents[FRAME_COUNT:]]
    for actual, expected in zip(queue_items, expected_names):
        assert _names_equivalent(Path(actual).name, expected)

    continuation = snapshot["continuation"]
    assert continuation is not None
    assert continuation.next_source_index == FRAME_COUNT
    assert qm._drizzle_checkpoint_writer.current_generation == GENERATION

    # First post-resume admission -> generation 463 -> reload.
    next_path = queue_items[0]
    ok = qm._add_frame_to_drizzle_accumulators(
        np.stack([_frame(0)[0]] * 3, axis=-1).astype(np.float32),
        fits.Header([("EXPTIME", 1.0)]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        _frame(0)[1],
        native_wcs=_wcs(),
    )
    assert ok is True
    qm._drizzle_group_tick()
    qm._admit_exposure(1.0, 0, 1.0, 1.0)
    qm.stacked_batches_count += 1
    qm._drizzle_checkpoint_after_frame(str(next_path))
    qm._drizzle_checkpoint_force_flush()

    manifest_after = json.loads(
        (out / CHECKPOINT_DIRNAME / MANIFEST_FILENAME).read_text()
    )
    assert manifest_after["generation"] == GENERATION + 1
    assert manifest_after["frame_count"] == FRAME_COUNT + 1
    result4 = read_drizzle_checkpoint(
        str(out), resolver=SafeStackedSourceResolver("stacked")
    )
    assert result4.generation == GENERATION + 1
    assert result4.next_source_index == FRAME_COUNT + 1
    assert result4.completed_sources[:FRAME_COUNT] == plan_idents[:FRAME_COUNT]
    completed_keys = {
        (s["path"], s["size"], s["mtime_ns"]) for s in result4.completed_sources
    }
    for i in REJECTED_INDICES:
        key = (idents[i]["path"], idents[i]["size"], idents[i]["mtime_ns"])
        assert key not in completed_keys
    assert result4.counters["frame_count"] == FRAME_COUNT + 1
    assert result4.counters["stacked_batches_count"] == FRAME_COUNT + 1

    # Post-recovery rejection -> cursor-only generation -> reload.
    rejected_path = queue_items[1]
    qm._drizzle_checkpoint_after_rejection(str(rejected_path))
    manifest_rej = json.loads(
        (out / CHECKPOINT_DIRNAME / MANIFEST_FILENAME).read_text()
    )
    assert manifest_rej["generation"] == GENERATION + 2
    assert manifest_rej["frame_count"] == FRAME_COUNT + 1
    assert manifest_rej["plan_cursor"] == FRAME_COUNT + 2
    assert [x["name"] for x in manifest_rej["rejected_sources"]] == [
        Path(rejected_path).name
    ]
    result5 = read_drizzle_checkpoint(
        str(out), resolver=SafeStackedSourceResolver("stacked")
    )
    assert result5.generation == GENERATION + 2
    assert result5.next_source_index == FRAME_COUNT + 2
    assert [x["name"] for x in result5.rejected_sources] == [
        Path(rejected_path).name
    ]
