"""ZSSS-LIFECYCLE-01-R1: failed-start resource unwind regression tests.

These tests pin the two lifecycle corrections for the resume-contract slice:

* **B1 (autotuner leak):** a bare stacker with a fake autotuner and an invalid
  input must not start the tuner before failing (the tuner is started at the
  latest safe point, just before the worker thread).  A second Start on the
  same instance then reaches normal validation with no stale flags/resources.
* **B2 (post-initialize artifacts):** a fresh attempt that creates output-bound
  SUM/WHT memmaps and then fails must release the handles and remove *only* the
  artifacts it created, leaving preexisting sentinel bytes untouched.  The
  cleanup helper is idempotent and preserves the structured refusal carrier.

No real stacking is performed.  The engine is imported lazily so the rest of
the suite stays fast and import-hygiene-clean.
"""

from __future__ import annotations

import os
import types
from pathlib import Path
from queue import Queue

import numpy as np
import pytest
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _FakeTuner:
    """Minimal autotuner stand-in recording start/stop counts."""

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1


class _RefAligner:
    """Records ``_get_reference_image`` and returns a fixed HWC frame."""

    def __init__(self, shape=(4, 5, 3)) -> None:
        self.stop_processing = False
        self.reference_image_path = None
        self.calls = 0
        self._shape = shape

    def _get_reference_image(self, folder, files, output_folder):
        self.calls += 1
        return (np.zeros(self._shape, dtype=np.float32), fits.Header())


def _bare_stack(out_dir, input_dir, autotuner=None, aligner=None):
    """Bare ``SeestarQueuedStacker`` (no ``__init__``) with enough state to run
    the early ``start_processing`` / cleanup seams deterministically."""
    from seestar.queuep.queue_manager import SeestarQueuedStacker

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
    o.aligner = (
        aligner
        if aligner is not None
        else types.SimpleNamespace(stop_processing=False)
    )
    o.autotuner = autotuner
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
    o.cumulative_sum_memmap = None
    o.cumulative_wht_memmap = None
    # Legacy package-local memmap slots (inert until a real allocation opens
    # them; the constructor must never create a package file).
    o.cumulative_wht_path = None
    o.cumulative_wht_memmap = None
    o.sum_memmap_path = None
    o.wht_memmap_path = None
    o.memmap_shape = None
    o.memmap_dtype_sum = np.float32
    o.memmap_dtype_wht = np.float32
    o._resume_active = False
    o._norm_reference = None
    o.additional_folders = []
    o.folders_lock = __import__("threading").Lock()
    o.processed_files = set()
    o.queue = Queue()
    o.files_in_queue = 0
    o._has_stack_plan = False
    o.batch_count_path = None
    return o


# --------------------------------------------------------------------------
# B1: fake autotuner invalid-input sequence — tuner not started, no stale state
# --------------------------------------------------------------------------
def test_fake_autotuner_invalid_input_started_equals_stopped(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    missing_in = tmp_path / "missing_input"  # does not exist

    tuner = _FakeTuner()
    s = _bare_stack(out, missing_in, autotuner=tuner)

    result = s.start_processing(
        input_dir=str(missing_in), output_dir=str(out), resume_intent="fresh"
    )

    assert result is False
    # The autotuner is started only at the latest safe point (just before the
    # worker thread), so a failed input validation never leaks it.
    assert tuner.started == 0
    assert tuner.stopped == 0
    assert tuner.started == tuner.stopped
    assert s.processing_active is False

    # Second Start on the same instance reaches normal validation (past the
    # intent gate) with no stale flags or resources.
    tuner2 = _FakeTuner()
    s.autotuner = tuner2
    in2 = tmp_path / "in2"
    in2.mkdir()
    r2 = s.start_processing(
        input_dir=str(in2), output_dir=str(out), resume_intent="fresh"
    )
    # Fails later for an unrelated reason (no FITS file in the input folder),
    # never an intent refusal and never a stale-flag issue.
    assert r2 is False
    assert s.startup_refusal is None
    assert s.processing_active is False
    assert tuner2.started == tuner2.stopped == 0


# --------------------------------------------------------------------------
# B1 helper: the cleanup helper stops an attempt-owned tuner (idempotently)
# --------------------------------------------------------------------------
def test_cleanup_helper_stops_owned_tuner_idempotently(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    tuner = _FakeTuner()
    s = _bare_stack(out, tmp_path / "in", autotuner=tuner)

    # Simulate that this attempt started the tuner.
    s._autotuner_started_this_attempt = True
    tuner.started += 1

    s._cleanup_failed_start()
    assert tuner.stopped == 1
    assert s.processing_active is False

    # Idempotent: a second cleanup never stops it again.
    s._cleanup_failed_start()
    assert tuner.stopped == 1


# --------------------------------------------------------------------------
# B2: post-initialize startup failure releases handles + only attempt artifacts
# --------------------------------------------------------------------------
def test_cleanup_helper_releases_memmaps_and_preserves_preexisting(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "sentinel.txt"
    sentinel.write_bytes(b"PREEXISTING-BYTES")
    before = sentinel.read_bytes()

    s = _bare_stack(out, tmp_path / "in")
    # Record pre-existing state exactly as start_processing does.
    s._autotuner_started_this_attempt = False
    s._attempt_preexisting_state = s._snapshot_existing_state()

    # Simulate a fresh post-initialize state: attempt-created SUM/WHT memmaps.
    memdir = out / "memmap_accumulators"
    memdir.mkdir()
    s.cumulative_sum_memmap = np.lib.format.open_memmap(
        str(memdir / "cumulative_SUM.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(2, 2, 3),
    )
    s.cumulative_wht_memmap = np.lib.format.open_memmap(
        str(memdir / "cumulative_WHT.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(2, 2, 3),
    )
    s.cumulative_sum_memmap[:] = 0.0
    s.cumulative_wht_memmap[:] = 0.0
    (out / "batches_count.txt").write_text("0", encoding="utf-8")
    sum_ref = s.cumulative_sum_memmap
    wht_ref = s.cumulative_wht_memmap
    s._resume_active = False

    s._cleanup_failed_start()

    # Handles closed + references dropped.
    assert s.cumulative_sum_memmap is None
    assert s.cumulative_wht_memmap is None
    assert sum_ref._mmap.closed is True
    assert wht_ref._mmap.closed is True

    # Only attempt-created artifacts removed; preexisting sentinel untouched.
    assert not (out / "memmap_accumulators").exists()
    assert not (out / "batches_count.txt").exists()
    assert sentinel.read_bytes() == before
    assert sorted(p.name for p in out.iterdir()) == ["sentinel.txt"]


def test_cleanup_helper_is_idempotent(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    s = _bare_stack(out, tmp_path / "in")
    s._autotuner_started_this_attempt = False
    s._attempt_preexisting_state = s._snapshot_existing_state()
    s._resume_active = False

    # No artifacts present at all: cleanup must be a no-op and never raise.
    s._cleanup_failed_start()
    s._cleanup_failed_start()
    assert s.processing_active is False
    assert s.stop_processing is False
    assert s.user_requested_stop is False


# --------------------------------------------------------------------------
# B2 integration: real start_processing fails *after* initialize creates memmaps
# --------------------------------------------------------------------------
def test_post_initialize_startup_failure_cleans_attempt_artifacts(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "sentinel.txt"
    sentinel.write_bytes(b"KEEP-ME")
    before = sentinel.read_bytes()

    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "frame.fits").write_bytes(b"\x00" * 16)

    s = _bare_stack(out, inp, aligner=_RefAligner())

    # Real initialize creates attempt-owned SUM/WHT memmaps + manifest on a
    # fresh run; here we reproduce exactly that artifact set.
    def fake_initialize(self, output_dir, shape_hwc, enable_preview=False):
        memdir = Path(output_dir) / "memmap_accumulators"
        memdir.mkdir(parents=True, exist_ok=True)
        self.cumulative_sum_memmap = np.lib.format.open_memmap(
            str(memdir / "cumulative_SUM.npy"),
            mode="w+",
            dtype=np.float32,
            shape=shape_hwc,
        )
        self.cumulative_wht_memmap = np.lib.format.open_memmap(
            str(memdir / "cumulative_WHT.npy"),
            mode="w+",
            dtype=np.float32,
            shape=shape_hwc,
        )
        self.cumulative_sum_memmap[:] = 0.0
        self.cumulative_wht_memmap[:] = 0.0
        self.memmap_shape = tuple(shape_hwc)
        self._resume_active = False
        (Path(output_dir) / "batches_count.txt").write_text("0", encoding="utf-8")
        return True

    s.initialize = types.MethodType(fake_initialize, s)
    s._add_files_to_queue = lambda folder: 0
    s._checkpoint_preflight = lambda: False  # post-initialize startup failure
    s._worker = lambda: None

    result = s.start_processing(
        input_dir=str(inp), output_dir=str(out), batch_size=10, resume_intent="fresh"
    )

    assert result is False
    # References dropped (memmap handles closed by _cleanup_failed_start).
    assert s.cumulative_sum_memmap is None
    assert s.cumulative_wht_memmap is None
    # Attempt-created artifacts removed; preexisting sentinel unchanged.
    assert not (out / "memmap_accumulators").exists()
    assert not (out / "batches_count.txt").exists()
    assert sentinel.read_bytes() == before
    assert not hasattr(s, "processing_thread")
    assert s.processing_active is False


# --------------------------------------------------------------------------
# No package-local filesystem side effect / no Errno-22 analogue
# --------------------------------------------------------------------------
def test_failed_start_has_no_package_filesystem_side_effect(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    missing_in = tmp_path / "missing_input"

    s = _bare_stack(out, missing_in, autotuner=_FakeTuner())
    result = s.start_processing(
        input_dir=str(missing_in), output_dir=str(out), resume_intent="fresh"
    )
    assert result is False

    pkg_dir = ROOT / "seestar"
    # The obsolete package-local cumulative_wht.dat must never be created (the
    # historical Errno-22 analogue: opening a 0-byte package memmap on Windows).
    assert not (pkg_dir / "cumulative_wht.dat").exists()
    assert s.cumulative_wht_path is None
    assert s.cumulative_wht_memmap is None
