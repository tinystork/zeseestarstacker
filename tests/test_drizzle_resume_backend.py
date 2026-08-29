"""RSM2-D2B2B backend/headless activation of native Drizzle Resume."""

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
    DrizzleCheckpointWriter,
    build_drizzle_canonical_config,
)
from seestar.core.drizzle_core import DrizzleAccumulator
import seestar.queuep.queue_manager as queue_manager_module
from seestar.queuep.queue_manager import SeestarQueuedStacker


SHAPE = (8, 8)


def _identity(path):
    st = os.stat(path)
    return {
        "path": os.path.normcase(str(path)),
        "name": os.path.basename(str(path)),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _wcs():
    w = WCS(naxis=2)
    w.wcs.crpix = [4.5, 4.5]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = [-0.001, 0.001]
    w.array_shape = SHAPE
    return w


def _configure(qm, output, input_root, kernel):
    qm.output_folder = str(output)
    qm._resume_input_roots = [str(input_root)]
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
    qm.drizzle_kernel = kernel
    qm.drizzle_pixfrac = 1.0
    qm.drizzle_wht_threshold = 0.0
    qm.drizzle_wht_threshold_effective = 0.0
    qm.drizzle_fillval = "0.0"
    qm.drizzle_group_size = 2
    qm.drizzle_processing_policy = "incremental"
    qm.preview_callback = None
    qm.update_progress = lambda *args, **kwargs: None
    qm.processing_error = None
    qm._drizzle_checkpoint_enabled = True
    qm._drizzle_checkpoint_writer = None
    qm._drizzle_resume_result = None
    qm._drizzle_resume_continuation = None
    qm._drizzle_checkpoint_last_committed_frames = 0
    qm._resume_requested = True
    qm._resume_active = False


def _frame(index):
    yy, xx = np.indices(SHAPE, dtype=np.float64)
    data = (index * 3.0 + xx * 0.25 + yy * 0.5).astype(np.float32)
    weight = np.full(SHAPE, 0.7 + index * 0.05, dtype=np.float32)
    pixmap = np.dstack((xx + index * 0.07, yy - index * 0.04))
    in_grid = np.ones(SHAPE, dtype=bool)
    return data, weight, pixmap, in_grid


def _add(accs, frame):
    data, weight, pixmap, in_grid = frame
    for acc in accs:
        acc.add(
            data,
            weight,
            pixmap,
            exptime=1.0,
            in_units="counts",
            in_grid_mask=in_grid,
        )


def _checkpoint(tmp_path, kernel="square", decomposition=None):
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths = []
    for i in range(4):
        p = inputs / f"src_{i}.fit"
        hdu = fits.PrimaryHDU(np.full(SHAPE, i + 1, dtype=np.uint16))
        hdu.header["EXPTIME"] = 1.0
        hdu.writeto(p)
        paths.append(p)
    idents = [_identity(p) for p in paths]

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, kernel)
    cfg = build_drizzle_canonical_config(
        qm, product_version=qm._canonical_product_version()
    )
    writer = DrizzleCheckpointWriter(
        output, qm._canonical_product_version(), cfg, _wcs(), SHAPE
    )
    accs = [DrizzleAccumulator(SHAPE, kernel=kernel, pixfrac=1.0) for _ in range(3)]
    for i in range(2):
        _add(accs, _frame(i))
    writer.commit(
        accs,
        session_binding={
            "input_roots": [str(inputs)],
            "reference": idents[0],
            "plan": {
                "sources": idents,
                "decomposition": list(decomposition or [4]),
            },
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
    return qm, output, inputs, paths, idents


class _NoopExecutor:
    def __init__(self, max_workers=1, **_kwargs):
        self._max_workers = max_workers

    def shutdown(self, *args, **kwargs):
        return None


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


def _start_resume_lifecycle(monkeypatch, output, inputs, *, batch_size):
    """Run the real start_processing lifecycle with bounded heavy-I/O doubles."""
    monkeypatch.setattr(
        queue_manager_module,
        "ProcessPoolExecutor",
        lambda **kwargs: _NoopExecutor(**kwargs),
    )
    qm = SeestarQueuedStacker(batch_size=batch_size, autotune=False)
    qm.update_progress = lambda *args, **kwargs: None
    aligner = _LifecycleAligner()
    qm.aligner = aligner
    worker_started = threading.Event()
    worker_snapshot = {}

    def _worker():
        worker_snapshot["queue"] = list(qm.queue.queue)
        worker_snapshot["frame_count"] = qm._drizzle_frame_count
        worker_snapshot["stacked_batches_count"] = qm.stacked_batches_count
        worker_snapshot["total_exposure_seconds"] = qm.total_exposure_seconds
        worker_snapshot["continuation"] = qm._drizzle_resume_continuation
        worker_snapshot["finalization_mode"] = qm.finalization_mode
        worker_started.set()

    qm._worker = _worker
    qm._solve_astrometry_async = lambda *args, **kwargs: _wcs()
    started = qm.start_processing(
        input_dir=str(inputs),
        output_dir=str(output),
        use_drizzle=True,
        drizzle_scale=1.0,
        drizzle_kernel="square",
        drizzle_pixfrac=1.0,
        drizzle_wht_threshold=0.0,
        drizzle_group_size=2,
        batch_size=batch_size,
        min_w=0.01,
        move_stacked=False,
        perform_cleanup=False,
        resume_intent="resume",
        reproject_between_batches=False,
        reproject_coadd_final=False,
    )
    if qm.processing_thread is not None:
        qm.processing_thread.join(timeout=5)
    return qm, aligner, worker_started, worker_snapshot, started


@pytest.mark.parametrize("kernel", ["square", "lanczos2"])
def test_stop_resume_backend_is_bit_identical_and_commits_n_plus_1(tmp_path, kernel):
    qm, output, _inputs, paths, _idents = _checkpoint(tmp_path, kernel)
    ok, resolved_ref = qm._early_resume_preflight()
    assert ok is True
    assert resolved_ref == str(paths[0])

    qm.queue = Queue()
    for path in paths[2:]:
        qm.queue.put(str(path))
    assert qm._init_drizzle_checkpoint() is True
    assert qm._drizzle_frame_count == 2
    assert qm.stacked_batches_count == 2
    assert qm.total_exposure_seconds == 2.0

    for i, path in enumerate(paths[2:], start=2):
        _add(qm.drizzle_accumulators, _frame(i))
        qm._drizzle_group_tick()
        qm.stacked_batches_count += 1
        qm.total_exposure_seconds += 1.0
        qm._drizzle_checkpoint_after_frame(str(path))

    continuous = [
        DrizzleAccumulator(SHAPE, kernel=kernel, pixfrac=1.0) for _ in range(3)
    ]
    for i in range(4):
        _add(continuous, _frame(i))
    for resumed, expected in zip(qm.drizzle_accumulators, continuous):
        assert np.array_equal(resumed._out_img, expected._out_img)
        assert np.array_equal(resumed._out_wht, expected._out_wht)
        assert np.array_equal(resumed.finalize("divide"), expected.finalize("divide"))

    manifest = json.loads(
        (output / ".m3d_checkpoint" / "checkpoint.json").read_text()
    )
    assert manifest["generation"] == 2
    assert manifest["frame_count"] == 4
    assert [x["name"] for x in manifest["completed_sources"]] == [
        f"src_{i}.fit" for i in range(4)
    ]


def test_suffix_mismatch_refuses_before_new_generation(tmp_path):
    qm, output, _inputs, paths, _idents = _checkpoint(tmp_path)
    ok, _ = qm._early_resume_preflight()
    assert ok is True
    manifest_path = output / ".m3d_checkpoint" / "checkpoint.json"
    before = manifest_path.read_bytes()
    qm.queue = Queue()
    qm.queue.put(str(paths[3]))
    qm.queue.put(str(paths[2]))
    assert qm._init_drizzle_checkpoint() is False
    assert manifest_path.read_bytes() == before
    assert json.loads(before)["generation"] == 1


def test_decomposition_suffix_accepts_exact_remaining_grouping(tmp_path):
    qm, output, _inputs, paths, _idents = _checkpoint(
        tmp_path, decomposition=[2, 2]
    )
    ok, _ = qm._early_resume_preflight()
    assert ok is True
    qm.batch_size = 2
    qm.queue = Queue()
    for path in paths[2:]:
        qm.queue.put(str(path))

    assert qm._init_drizzle_checkpoint() is True
    assert qm._drizzle_resume_continuation is not None
    assert qm._drizzle_checkpoint_plan["decomposition"] == [2, 2]


def test_decomposition_regroup_refuses_before_continuation_or_write(tmp_path):
    qm, output, _inputs, paths, _idents = _checkpoint(
        tmp_path, decomposition=[2, 2]
    )
    ok, _ = qm._early_resume_preflight()
    assert ok is True
    manifest_path = output / ".m3d_checkpoint" / "checkpoint.json"
    before = manifest_path.read_bytes()
    # Same exact source suffix, but regrouped as [1, 1] instead of [2].
    qm.batch_size = 1
    qm.queue = Queue()
    for path in paths[2:]:
        qm.queue.put(str(path))

    assert qm._init_drizzle_checkpoint() is False
    assert qm._drizzle_resume_continuation is None
    assert qm._drizzle_checkpoint_writer is None
    assert manifest_path.read_bytes() == before


def test_start_processing_resume_traverses_production_lifecycle_to_worker(
    tmp_path, monkeypatch
):
    _seed, output, inputs, paths, _idents = _checkpoint(
        tmp_path, decomposition=[2, 2]
    )

    qm, aligner, worker_started, snapshot, started = _start_resume_lifecycle(
        monkeypatch, output, inputs, batch_size=2
    )

    assert started is True
    assert worker_started.is_set()
    # The original persisted reference is pinned before reference preparation;
    # the first unprocessed source is never substituted for it.
    assert aligner.calls
    assert aligner.calls[0][2] == str(paths[0])
    # initialize restored native SCI/WHT/counters, then the real queue fill and
    # ledger filter left exactly the persisted suffix before worker launch.
    assert snapshot["queue"] == [str(paths[2]), str(paths[3])]
    assert snapshot["frame_count"] == 2
    assert snapshot["stacked_batches_count"] == 2
    assert snapshot["total_exposure_seconds"] == 2.0
    assert snapshot["continuation"] is not None
    assert snapshot["continuation"].generation == 1
    assert snapshot["finalization_mode"] == "drizzle"
    assert qm._drizzle_checkpoint_writer.current_generation == 1
    for restored in qm.drizzle_accumulators:
        assert np.any(restored._out_wht != 0)


def test_start_processing_invalid_decomposition_never_launches_worker_or_writes(
    tmp_path, monkeypatch
):
    _seed, output, inputs, _paths, _idents = _checkpoint(
        tmp_path, decomposition=[2, 2]
    )
    manifest_path = output / ".m3d_checkpoint" / "checkpoint.json"
    before = manifest_path.read_bytes()

    qm, _aligner, worker_started, _snapshot, started = _start_resume_lifecycle(
        monkeypatch, output, inputs, batch_size=1
    )

    assert started is False
    assert not worker_started.is_set()
    assert qm.processing_thread is None
    assert qm._drizzle_resume_continuation is None
    assert manifest_path.read_bytes() == before


def test_moved_completed_source_requires_exact_opt_in(tmp_path):
    qm, _output, inputs, paths, _idents = _checkpoint(tmp_path)
    stacked = inputs / "stacked"
    stacked.mkdir()
    paths[0].rename(stacked / paths[0].name)
    paths[1].rename(stacked / paths[1].name)

    ok, reason, _ = qm._validate_drizzle_resume_headless()
    assert ok is False
    assert "invalid Drizzle checkpoint" in reason

    qm.move_stacked = True
    ok, result, resolved_ref = qm._validate_drizzle_resume_headless()
    assert ok is True
    assert resolved_ref == str(stacked / paths[0].name)
    assert list(result.resolved_remaining_paths) == [str(p) for p in paths[2:]]


def test_drizzle_namespace_counts_as_state_even_without_run_config(tmp_path):
    qm = object.__new__(SeestarQueuedStacker)
    out = tmp_path / "out"
    (out / ".m3d_checkpoint").mkdir(parents=True)
    assert qm._resume_artifacts_present(out) is True


def test_runtime_restore_keeps_native_counters_and_ledger(tmp_path):
    qm, _output, _inputs, _paths, idents = _checkpoint(tmp_path)
    ok, result, _ = qm._validate_drizzle_resume_headless()
    assert ok is True
    # Simulate the common initialize reset that previously zeroed these fields.
    qm.stacked_batches_count = 0
    qm.total_exposure_seconds = 0.0
    qm._resume_completed_sources = []
    qm._restore_drizzle_checkpoint_runtime(result)
    assert qm._drizzle_frame_count == 2
    assert qm.stacked_batches_count == 2
    assert qm.total_exposure_seconds == 2.0
    assert qm._resume_completed_sources == idents[:2]
    assert qm._resume_plan["sources"] == idents


@pytest.mark.parametrize(
    "decomposition, completed, batch_size, expected_suffix",
    [
        ([4, 4, 2], 3, 4, [1, 4, 2]),  # mid-batch: residual of the first 4
        ([4, 4, 2], 4, 4, [4, 2]),     # aligned boundary
    ],
)
def test_mid_batch_resume_uses_authoritative_suffix(
    tmp_path, decomposition, completed, batch_size, expected_suffix
):
    n = sum(decomposition)
    output = tmp_path / "out"
    inputs = tmp_path / "inputs"
    output.mkdir()
    inputs.mkdir()
    paths = []
    for i in range(n):
        p = inputs / ("src_%d.fit" % i)
        hdu = fits.PrimaryHDU(np.full(SHAPE, i + 1, dtype=np.uint16))
        hdu.header["EXPTIME"] = 1.0
        hdu.writeto(p)
        paths.append(p)
    idents = [_identity(p) for p in paths]

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, output, inputs, "square")
    cfg = build_drizzle_canonical_config(
        qm, product_version=qm._canonical_product_version()
    )
    writer = DrizzleCheckpointWriter(
        output, qm._canonical_product_version(), cfg, _wcs(), SHAPE
    )
    accs = [DrizzleAccumulator(SHAPE, kernel="square", pixfrac=1.0) for _ in range(3)]
    for i in range(completed):
        _add(accs, _frame(i))
    writer.commit(
        accs,
        session_binding={
            "input_roots": [str(inputs)],
            "reference": idents[0],
            "plan": {"sources": idents, "decomposition": list(decomposition)},
        },
        counters={
            "frame_count": completed,
            "stacked_batches_count": completed,
            "total_exposure_seconds": float(completed),
            "exposure_unknown_count": 0,
            "exposure_min": 1.0,
            "exposure_max": 1.0,
        },
        completed_sources=idents[:completed],
    )

    ok, _ = qm._early_resume_preflight()
    assert ok is True
    qm.batch_size = batch_size
    qm.queue = Queue()
    for path in paths[completed:]:
        qm.queue.put(str(path))

    assert qm._init_drizzle_checkpoint() is True
    assert qm._drizzle_resume_continuation is not None
    _, decomp, _ = qm._scan_queue_decomposition()
    assert decomp == expected_suffix
