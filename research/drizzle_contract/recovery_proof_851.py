"""ZSSS-DRIZZLE-REJECTED-SOURCE Phase C: emergency recovery proof.

This script proves the one-shot recovery model for the real Windows run
against the REAL production code path.  It is designed to run on BOTH:

* pristine 8.5.1 (57a8a1d) — the decisive gate: stock Resume must accept the
  recovered state and commit generation 463;
* the fixed branch — the permanent contract must behave identically on the
  legacy witness.

Witness (synthetic, equivalent to the audited Windows checkpoint):

    plan length 5329, generation 462, frame_count 4620,
    stacked_batches_count 4620, completed_sources = exact plan[0:4620],
    state = clean, decomposition [10]*532 + [9].

Live filesystem (as audited):

    4620 committed sources in stacked/,
    9 accepted-after-4620 sources in stacked/ (science NOT in gen 462),
    5 rejected future observations in unaligned_by_stacker/,
    the remaining 695 sources at their original positions.

One-shot recovery (no committed science modified):

    1. remove the five FUTURE rejected plan entries (5329 -> 5324) and
       re-derive the decomposition over the surviving sources;
    2. restore the nine accepted-but-uncommitted files from stacked/ to
       their canonical original locations (they must be replayed);
    3. leave the five rejected observations parked outside the active plan.

Proof obligations executed below:

    A. the pristine witness loads through the real checkpoint reader;
    B. after the live-filesystem moves, the unmodified checkpoint REFUSES to
       read (the real secondary failure) — the surgery is therefore required;
    C. the recovered checkpoint loads through the real reader;
    D. the production Resume path re-arms the continuation writer;
    E. the remaining queue validates against the recovered plan;
    F. one subsequent observation is admitted and generation 463 commits;
    G. the newly committed checkpoint reloads with coherent counters;
    H. the first 4620 completed identities are byte-unchanged and no removed
       rejected observation ever enters the completed ledger.

Exit code 0 only when every obligation holds.
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
    DrizzleCheckpointWriter,
    SafeStackedSourceResolver,
    build_drizzle_canonical_config,
    read_drizzle_checkpoint,
)
from seestar.core.drizzle_core import DrizzleAccumulator
from seestar.queuep.queue_manager import SeestarQueuedStacker

SHAPE = (8, 8)

PLAN_LEN = 5329
FRAME_COUNT = 4620
GENERATION = 462
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


def _configure(qm, output, inputs, group_size=1):
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
    qm.drizzle_kernel = "square"
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


def _forge(tmp_dir, inputs, out):
    """Forge the synthetic generation-462 witness (legacy prefix-only format).

    One real generation is committed with the production writer, then the
    manifest is rewritten: generation -> 462, plan -> the 5329 identities,
    frame_count/stacked_batches_count -> 4620, completed_sources ->
    plan[:4620], decomposition -> [10]*532 + [9].  Artifact BYTES and
    run_config.cfg stay untouched (only filenames are renumbered)."""
    paths = [
        inputs / f"Light_SH2-101_20.0s_IRCUT_20260815-{214000 + i}.fit"
        for i in range(PLAN_LEN)
    ]
    for i, p in enumerate(paths):
        fits.PrimaryHDU(np.full(SHAPE, (i % 200) + 1, dtype=np.uint16)).writeto(p)
    idents = [_identity(p) for p in paths]

    full, rem = divmod(PLAN_LEN, 10)
    decomposition = [10] * full + ([rem] if rem else [])

    qm = object.__new__(SeestarQueuedStacker)
    _configure(qm, out, inputs, group_size=1)
    qm._resume_reference_identity = idents[0]
    cfg = build_drizzle_canonical_config(
        qm, product_version=qm._canonical_product_version()
    )
    writer = DrizzleCheckpointWriter(
        str(out), qm._canonical_product_version(), cfg, _wcs(), SHAPE
    )
    accs = [DrizzleAccumulator(SHAPE, kernel="square", pixfrac=1.0) for _ in range(3)]
    _add(accs, _frame(0))
    _add(accs, _frame(1))
    writer.commit(
        accs,
        session_binding={
            "input_roots": [str(inputs)],
            "reference": idents[0],
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
    manifest["session"]["plan"]["sources"] = idents
    manifest["session"]["plan"]["decomposition"] = decomposition
    manifest["completed_sources"] = idents[:FRAME_COUNT]
    (ckpt_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    return paths, idents, decomposition


def _surgery(manifest, rejected_indices):
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


def main():
    tmp_dir = Path(tempfile.mkdtemp(prefix="zsss-recovery-proof-"))
    inputs = tmp_dir / "inputs"
    out = tmp_dir / "out"
    inputs.mkdir()
    out.mkdir()

    paths, idents, decomposition = _forge(tmp_dir, inputs, out)
    resolver = SafeStackedSourceResolver("stacked")
    ckpt_manifest = out / CHECKPOINT_DIRNAME / MANIFEST_FILENAME

    # ---- A. pristine witness loads through the real reader ----
    result = read_drizzle_checkpoint(str(out), resolver=resolver)
    assert result.generation == GENERATION
    assert result.next_source_index == FRAME_COUNT
    assert result.completed_sources == idents[:FRAME_COUNT]
    print(f"A. pristine witness loads: gen={result.generation}, "
          f"frame_count={result.counters['frame_count']}, "
          f"plan={len(result.session['plan']['sources'])}")

    # ---- B. live Windows filesystem state ----
    stacked = inputs / "stacked"
    stacked.mkdir()
    for p in paths[:FRAME_COUNT]:
        p.rename(stacked / p.name)
    for i in ACCEPTED_UNCOMMITTED_INDICES:
        paths[i].rename(stacked / paths[i].name)
    unaligned = inputs / "unaligned_by_stacker"
    unaligned.mkdir()
    for i in REJECTED_INDICES:
        paths[i].rename(unaligned / paths[i].name)

    try:
        read_drizzle_checkpoint(str(out), resolver=resolver)
        print("B. ERROR: unmodified checkpoint read — expected refusal")
        return 1
    except DrizzleCheckpointError as exc:
        print(f"B. unmodified checkpoint refuses (real secondary failure): {exc}")

    # ---- C. one-shot surgery ----
    manifest = json.loads(ckpt_manifest.read_text())
    new_sources, new_decomp = _surgery(manifest, REJECTED_INDICES)
    manifest["session"]["plan"]["sources"] = new_sources
    manifest["session"]["plan"]["decomposition"] = new_decomp
    assert len(new_sources) == PLAN_LEN - 5
    assert new_sources[:FRAME_COUNT] == idents[:FRAME_COUNT]
    ckpt_manifest.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    for i in ACCEPTED_UNCOMMITTED_INDICES:
        (stacked / paths[i].name).rename(paths[i])

    result2 = read_drizzle_checkpoint(str(out), resolver=resolver)
    assert result2.generation == GENERATION
    assert result2.next_source_index == FRAME_COUNT
    assert len(result2.resolved_remaining_paths) == PLAN_LEN - 5 - FRAME_COUNT
    assert result2.resolved_remaining_paths[0] == str(paths[ACCEPTED_UNCOMMITTED_INDICES[0]])
    print(f"C. recovered checkpoint loads: remaining={len(result2.resolved_remaining_paths)}, "
          f"first={Path(result2.resolved_remaining_paths[0]).name}")

    # ---- D/E. production Resume path + remaining queue validation ----
    qm2 = object.__new__(SeestarQueuedStacker)
    _configure(qm2, out, inputs, group_size=1)
    qm2.move_stacked = True
    qm2._resume_requested = True
    ok, result3, _ref = qm2._validate_drizzle_resume_headless()
    assert ok is True
    qm2._drizzle_resume_result = result3
    qm2.queue = Queue()
    for p in result3.resolved_remaining_paths:
        qm2.queue.put(p)
    assert qm2._init_drizzle_checkpoint() is True
    continuation = qm2._drizzle_resume_continuation
    assert continuation.next_source_index == FRAME_COUNT
    assert qm2._drizzle_checkpoint_writer.current_generation == GENERATION
    print(f"D/E. production Resume re-arms: writer_generation="
          f"{qm2._drizzle_checkpoint_writer.current_generation}, "
          f"frame_count={qm2._drizzle_frame_count}, "
          f"stacked_batches_count={qm2.stacked_batches_count}")

    # ---- F. admit the next observation; commit generation 463 ----
    next_path = result3.resolved_remaining_paths[0]
    _accept(qm2, next_path, FRAME_COUNT)
    qm2._drizzle_checkpoint_force_flush()
    manifest_after = json.loads(ckpt_manifest.read_text())
    assert manifest_after["generation"] == GENERATION + 1
    assert manifest_after["frame_count"] == FRAME_COUNT + 1
    assert manifest_after["stacked_batches_count"] == FRAME_COUNT + 1
    print(f"F. next observation admitted: generation={manifest_after['generation']}, "
          f"frame_count={manifest_after['frame_count']}")

    # ---- G/H. reload the newly committed checkpoint ----
    result4 = read_drizzle_checkpoint(str(out), resolver=resolver)
    assert result4.generation == GENERATION + 1
    assert result4.next_source_index == FRAME_COUNT + 1
    assert result4.completed_sources[:FRAME_COUNT] == idents[:FRAME_COUNT]
    assert result4.completed_sources[FRAME_COUNT] == _identity(next_path)
    completed_keys = {
        (s["path"], s["size"], s["mtime_ns"]) for s in result4.completed_sources
    }
    for i in REJECTED_INDICES:
        assert (idents[i]["path"], idents[i]["size"], idents[i]["mtime_ns"]) not in completed_keys
    assert result4.counters["frame_count"] == FRAME_COUNT + 1
    assert result4.counters["stacked_batches_count"] == FRAME_COUNT + 1
    print("G/H. reload coherent: 4620 committed identities unchanged, "
          "no rejected observation in the completed ledger, counters coherent")

    print()
    print("EMERGENCY RECOVERY PROOF: ALL OBLIGATIONS GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
