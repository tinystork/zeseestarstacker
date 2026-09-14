"""ZSSS-DRIZZLE-REJECTED-SOURCE Phase B reproducer (pristine 8.5.1).

Run against the untouched 8.5.1 tree (57a8a1d).  This script PROVES the two
production failure modes by exercising the real production code paths:

  FAILURE 1 — checkpoint contract contradiction
      PLAN = A B C D E
      B rejected (moved to unaligned_by_stacker), A and C accepted.
      8.5.1 then tries to persist the truthful state
      (completed_sources = [A, C], frame_count = 2) and must refuse with
      "completed_sources is not the exact ordered prefix of the session plan"
      because plan[:2] = [A, B].

  FAILURE 2 — secondary Resume failure
      A committed checkpoint with completed = [A] (frame_count 1) exists, but
      C was accepted and moved to stacked/ AFTER the commit and B was rejected
      into unaligned_by_stacker/.  The filesystem remaining suffix can no
      longer equal persisted_sources[len(completed_sources):], so
      _validate_plan_against_manifest refuses Resume.

Both demonstrations must succeed on stock 8.5.1 code.  If either does NOT
fail, the reproducer model does not match the production defect.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from queue import Queue

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from seestar.core.drizzle_checkpoint import (
    CHECKPOINT_DIRNAME,
    MANIFEST_FILENAME,
    DrizzleCheckpointError,
    build_drizzle_canonical_config,
)
from seestar.core.drizzle_core import DrizzleAccumulator
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


def _frame(index):
    yy, xx = np.indices(SHAPE, dtype=np.float64)
    data = (index * 3.0 + xx * 0.25 + yy * 0.5).astype(np.float32)
    weight = np.full(SHAPE, 0.7, dtype=np.float32)
    pixmap = np.dstack((xx + index * 0.07, yy - index * 0.04))
    in_grid = np.ones(SHAPE, dtype=bool)
    return data, weight, pixmap, in_grid


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
    qm._drizzle_checkpoint_last_committed_frames = 0
    qm._drizzle_resume_result = None
    qm._drizzle_resume_continuation = None
    qm._resume_requested = False
    qm._resume_active = False
    qm._resume_plan = None
    qm._resume_completed_sources = []
    qm._resume_reference_identity = None


def _bind(qm, output, inputs, paths):
    idents = []
    for p in paths:
        st = os.stat(p)
        idents.append({
            "path": os.path.normcase(str(p)),
            "name": os.path.basename(str(p)),
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        })
    ref_path = Path(inputs) / "reference.fit"
    ref_path.write_bytes(b"session-alignment-reference")
    st = os.stat(ref_path)
    qm._resume_reference_identity = {
        "path": os.path.normcase(str(ref_path)),
        "name": ref_path.name,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }
    qm.drizzle_accumulators = [
        DrizzleAccumulator(SHAPE, kernel="square", pixfrac=1.0) for _ in range(3)
    ]
    qm._drizzle_checkpoint_plan = {
        "sources": idents,
        "decomposition": [len(idents)],
    }
    cfg = build_drizzle_canonical_config(
        qm, product_version=qm._canonical_product_version()
    )
    from seestar.core.drizzle_checkpoint import DrizzleCheckpointWriter
    qm._drizzle_checkpoint_writer = DrizzleCheckpointWriter(
        str(output), qm._canonical_product_version(), cfg, _wcs(), SHAPE
    )
    return idents


def _accept(qm, path, index):
    data, weight, pixmap, in_grid = _frame(index)
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


def _manifest(output):
    return json.loads(
        (Path(output) / CHECKPOINT_DIRNAME / MANIFEST_FILENAME).read_text()
    )


def _build(tmpdir):
    tmp = Path(tmpdir)
    inputs = tmp / "inputs"
    out = tmp / "out"
    inputs.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(5):
        p = inputs / f"src_{i}.fit"
        fits.PrimaryHDU(np.full(SHAPE, i + 1, dtype=np.uint16)).writeto(p)
        paths.append(p)
    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, out, inputs, group_size=1)
    idents = _bind(qm, out, inputs, paths)
    return tmp, inputs, out, paths, idents, qm


def demonstrate_failure_1(tmpdir):
    print("== FAILURE 1: checkpoint contract contradiction (A, B rejected, C) ==")
    tmp, inputs, out, paths, idents, qm = _build(tmpdir)
    _accept(qm, paths[0], 0)          # A accepted -> gen 1 (frame_count 1)
    # B rejected: the 8.5.1 worker moves it away and continues (non-fatal).
    qm.failed_align_count += 1
    qm._move_to_unaligned(str(paths[1]))
    assert not paths[1].exists()
    assert (inputs / "unaligned_by_stacker" / "src_1.fit").exists()
    # C accepted.  Its cadence commit (group_size=1) runs inside the real
    # production hook `_drizzle_checkpoint_after_frame` — which is where the
    # production failure actually strikes.
    try:
        _accept(qm, paths[2], 2)
    except DrizzleCheckpointError as exc:
        print(f"8.5.1 production hook refuses with: {exc}")
        assert "exact ordered prefix" in str(exc)
        return True
    print("ERROR: 8.5.1 accepted the truthful rejected-state — reproducer invalid")
    return False


def demonstrate_failure_1b(tmpdir):
    """Direct writer-level proof: the truthful state cannot even be
    expressed at the contract boundary."""
    print()
    print("== FAILURE 1b: writer refuses completed=[A, C] / frame_count=2 ==")
    tmp, inputs, out, paths, idents, qm = _build(tmpdir)
    _accept(qm, paths[0], 0)          # A -> gen 1

    writer = qm._drizzle_checkpoint_writer
    binding = {
        "input_roots": [str(inputs)],
        "reference": qm._resume_reference_identity,
        "plan": qm._drizzle_checkpoint_plan,
    }
    from seestar.core.drizzle_checkpoint import serialize_input_reference_geometry
    binding["reference_geometry"] = serialize_input_reference_geometry(
        _wcs(), SHAPE, None
    )
    try:
        writer.commit(
            qm.drizzle_accumulators,
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
        print("ERROR: 8.5.1 accepted the truthful rejected-state — reproducer invalid")
        return False
    except DrizzleCheckpointError as exc:
        print(f"8.5.1 refuses with: {exc}")
        assert "exact ordered prefix" in str(exc)
        return True


def demonstrate_failure_2(tmpdir):
    print()
    print("== FAILURE 2: secondary Resume failure (stacked/uncommitted files) ==")
    tmp, inputs, out, paths, idents, qm = _build(tmpdir)
    _accept(qm, paths[0], 0)          # A -> gen 1 (frame_count 1, committed)

    # After the commit: B rejected (unaligned), C accepted but not committed
    # (its group cadence never fired), C moved to stacked, D/E untouched.
    qm.failed_align_count += 1
    qm._move_to_unaligned(str(paths[1]))
    qm2 = object.__new__(SeestarQueuedStacker)
    _configure(qm2, out, inputs, group_size=1)
    qm2._resume_requested = True
    qm2._resume_active = True
    # The resumed ledger (restored from the gen-1 checkpoint) = [A].
    st = os.stat(paths[0])
    qm2._resume_completed_sources = [idents[0]]
    qm2._resume_plan = qm._drizzle_checkpoint_plan
    qm2._drizzle_resume_result = None
    # C was accepted + moved to stacked AFTER the commit (real crash window).
    qm.move_stacked = True
    qm._move_to_stacked([str(paths[2])])
    assert (inputs / "stacked" / "src_2.fit").exists()
    # The folder scan sees only what is still at original positions.
    qm2.queue = Queue()
    for p in (paths[3], paths[4]):
        qm2.queue.put(str(p))

    ok, reason = qm2._validate_plan_against_manifest()
    print(f"_validate_plan_against_manifest -> ({ok}, {reason!r})")
    assert ok is False
    assert "differs" in reason or "prefix" in reason
    return True


def main():
    tmpdir = tempfile.mkdtemp(prefix="zsss-repro-851-")
    r1 = demonstrate_failure_1(Path(tmpdir) / "f1")
    r1b = demonstrate_failure_1b(Path(tmpdir) / "f1b")
    r2 = demonstrate_failure_2(Path(tmpdir) / "f2")
    print()
    print("RESULT: all 8.5.1 failure modes reproduced:", r1 and r1b and r2)
    return 0 if (r1 and r1b and r2) else 1


if __name__ == "__main__":
    sys.exit(main())
