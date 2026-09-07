"""Stage D — transactional Classic batch lifecycle (failure-safe commit).

Contract (see ``docs/8.4.0_state.md`` stage D and
``.a2a-reports/.../stage-d-contract.md``): a source exposure may be marked
consumed (moved to ``stacked/`` / ledgered / checkpointed / counted) ONLY
after its scientific contribution is durably committed.  A failed reduction
is fatal and truthful, never interpreted as a successful (or ignorable)
batch, never moving sources or advancing committed state.

These tests drive the REAL transactional completion helper
(``SeestarQueuedStacker._process_completed_batch``) and the REAL Classic
mean reducer / ``_combine_batch_result`` on production-style bare stackers
(mirroring ``tests/test_coverage_support_classic.py`` and
``tests/test_exposure_metadata_contract.py`` harness style):

* failure matrix — reducer failures of every required family raise and
  leave sources / committed counter / count file / meta / ledger / partial
  cumulative untouched;
* on-disk resume witness — batch 1 commits (sources under ``stacked/``,
  committed count == 1), batch 2 fails (sources remain in their input
  location, cumulative equals exactly batch 1) and a legitimate re-run does
  NOT skip batch 2;
* counter semantics — ``stacked_batches_count`` equals committed batches,
  never attempted;
* empty / no-reducer-output (reproject missing-WCS family) batches are
  NON-fatal no-commit conditions: nothing consumed, nothing advanced;
* ``_combine_batch_result`` failure signalling — checkpoint / memory /
  accumulation failures inside the commit are surfaced (strict mode raises
  ``BatchReductionError``) and the helper consumes nothing afterwards.

Seam notes (honesty): SPATIAL_TILED_CPU and GPU->CPU-fallback wiring into
``_stack_batch`` dispatch belongs to stage E; today the winsorized CPU
reducer that production reaches is ``_stack_winsorized_sigma`` through
``_gpu_reduce_winsorized``, and every reducer exception that escapes a
reduction is converted by ``_stack_batch`` into ``BatchReductionError``
(the stage D conversion under test in the real-pipeline row).  The matrix
therefore injects the canonical exception class of each reducer family at
the ``_stack_batch`` boundary (post-conversion, exactly what the helper can
observe) AND one REAL in-pipeline conversion row.
"""

from __future__ import annotations

import os
import types
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from seestar.core.cpu_winsor_exact_n import CpuWinsorMemoryRefused
from seestar.queuep.queue_manager import (
    BatchReductionError,
    SeestarQueuedStacker,
    _ResumeCheckpointError,
)


# ---------------------------------------------------------------------------
# Harness: production-style bare stacker with real memmaps + transactional
# extras (identical science wiring to test_coverage_support_classic).
# ---------------------------------------------------------------------------

def _tx_stack(tmp_path, name, shape=(4, 5, 3), stacking_mode="mean"):
    o = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    o.output_folder = str(tmp_path / name)
    os.makedirs(o.output_folder, exist_ok=True)
    o.update_progress = lambda *a, **k: None
    o.logger = types.SimpleNamespace(
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    o.stacking_mode = stacking_mode
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
    o.batch_size = 1
    o.settings = None
    o.reference_header_for_wcs = None
    o.reference_wcs_object = None
    o.interbatch_norm_active = False
    o.max_stack_workers = 1
    o._current_batch_paths = []
    o._indices_cache = {}
    o._norm_reference = None
    o._is_plain_classic = lambda: True
    o._support_state_available = True
    o._support_unavailable_reason = None
    o.coverage_sup_w1_memmap = None
    o.coverage_sup_w2_memmap = None
    o._create_support_memmaps(shape[:2])
    o.apply_batch_feathering = False
    # Transactional helpers / consumption state.
    o._last_classic_batch_solved = True
    o._send_eta_update = lambda *a, **k: None
    o._update_preview_sum_w = lambda *a, **k: None
    o._solve_cumulative_stack = lambda *a, **k: (None, None)
    o.stop_processing = False
    o.processing_error = None
    o.failed_stack_count = 0
    o.stacked_batches_count = 0
    o.images_in_cumulative_stack = 0
    o.total_exposure_seconds = 0.0
    o._exposure_unknown_count = 0
    o._exposure_min = None
    o._exposure_max = None
    o.align_on_disk = False
    o.aligned_temp_paths = []
    o.move_stacked = True
    o.stacked_subdir_name = "stacked"
    o.output_filename = "stack"
    o.partial_save_interval = 0
    o.batch_count_path = os.path.join(o.output_folder, "batch_count.txt")
    memdir = os.path.join(o.output_folder, "memmap_accumulators")
    os.makedirs(memdir, exist_ok=True)
    o.sum_memmap_path = os.path.join(memdir, "cumulative_SUM.npy")
    o.wht_memmap_path = os.path.join(memdir, "cumulative_WHT.npy")
    o.cumulative_sum_memmap = np.lib.format.open_memmap(
        o.sum_memmap_path, mode="w+", dtype=np.float32, shape=shape
    )
    o.cumulative_wht_memmap = np.lib.format.open_memmap(
        o.wht_memmap_path, mode="w+", dtype=np.float32, shape=shape
    )
    o.cumulative_sum_memmap[:] = 0.0
    o.cumulative_wht_memmap[:] = 0.0
    o.memmap_shape = shape
    o.memmap_dtype_sum = np.float32
    o.memmap_dtype_wht = np.float32
    o.current_stack_header = None
    o.correct_hot_pixels = False
    return o


def _snapshot(o):
    """Byte-identical snapshot of the committed cumulative state."""
    return (
        np.array(o.cumulative_sum_memmap, copy=True),
        np.array(o.cumulative_wht_memmap, copy=True),
    )


def _assert_zero_cumulative(o):
    assert not np.any(o.cumulative_sum_memmap)
    assert not np.any(o.cumulative_wht_memmap)


def _item(img, mask=None):
    if mask is None:
        mask = np.ones(img.shape[:2], dtype=bool)
    return (img, fits.Header(), {"snr": 1.0, "stars": 0.0}, None, mask)


def _write_sources(tmp_path, names):
    """Write real placeholder source files under an input/ folder."""
    src_dir = tmp_path / "input"
    src_dir.mkdir(exist_ok=True)
    paths = []
    for n in names:
        p = src_dir / n
        p.write_bytes(b"\x00" * 32)
        paths.append(str(p))
    return src_dir, paths


def _images(shape=(4, 5, 3), base=10.0, n=2, seed=0):
    rng = np.random.default_rng(seed)
    return [
        (base + rng.normal(0.0, 1.0, shape)).astype(np.float32) for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# 8. REWORK-1: the six silent early-return failure paths of
#    ``_combine_batch_result`` (coverage shape / colour shape / gray shape /
#    N&B dims / near-zero coverage / non-finite zero-weight) must raise under
#    strict_commit=True and stay silently-returning under the legacy default.
# ---------------------------------------------------------------------------

SIX_SILENT_KINDS = [
    "coverage_shape",
    "color_shape",
    "gray_shape",
    "nb_dims",
    "near_zero_coverage",
    "nonfinite_zero_weight",
]


def _crafted_batch(kind):
    """A (stacked, hdr, cov) triple that drives the REAL combine into the
    named silent guard (memmap shape is (4, 5, 3))."""
    hdr = fits.Header()
    hdr["NIMAGES"] = 2
    ones = np.ones((4, 5, 3), dtype=np.float32)
    cov = np.ones((4, 5), dtype=np.float32)
    if kind == "coverage_shape":
        return ones, hdr, np.ones((6, 6), dtype=np.float32)
    if kind == "color_shape":
        return np.ones((3, 5, 3), dtype=np.float32), hdr, cov
    if kind == "gray_shape":
        # Smaller than the memmap (no resize branch): pure gray-shape guard.
        return np.ones((3, 5), dtype=np.float32), hdr, cov
    if kind == "nb_dims":
        return np.ones((4, 5, 1), dtype=np.float32), hdr, cov
    if kind == "near_zero_coverage":
        return ones, hdr, np.zeros((4, 5), dtype=np.float32)
    if kind == "nonfinite_zero_weight":
        return np.full((4, 5, 3), np.nan, dtype=np.float32), hdr, cov
    raise KeyError(kind)


@pytest.mark.parametrize("kind", SIX_SILENT_KINDS)
def test_six_silent_paths_strict_combine_raises(tmp_path, kind):
    """Direct strict_commit=True: each of the six paths raises
    BatchReductionError, sets the terminal state and counts the failed
    frames once (legacy accounting preserved before the raise)."""
    o = _tx_stack(tmp_path, f"strict_{kind}")
    o._support_state_available = False
    stacked, hdr, cov = _crafted_batch(kind)
    with pytest.raises(BatchReductionError) as ei:
        o._combine_batch_result(stacked, hdr, cov, strict_commit=True)
    assert "batch commit refused" in str(ei.value)
    assert o.stop_processing is True
    assert o.processing_error
    # Legacy accounting still ran exactly once before the truthful raise.
    assert o.failed_stack_count == 2
    # No partial-cumulative mutation.
    _assert_zero_cumulative(o)
    assert o.images_in_cumulative_stack == 0


@pytest.mark.parametrize("kind", SIX_SILENT_KINDS)
def test_six_silent_paths_strict_via_helper_consumes_nothing(tmp_path, kind):
    """Via the transactional helper: the strict commit failure propagates as
    BatchReductionError and NOTHING is consumed or advanced — no move, no
    committed counter, no count-file/meta, no partial save, cumulative
    unchanged, terminal state set."""
    o = _tx_stack(tmp_path, f"helper_{kind}")
    o._support_state_available = False
    src_dir, src_paths = _write_sources(tmp_path, ["a.fit", "b.fit"])
    imgs = _images(n=2)
    items = [_item(im) for im in imgs]
    o._current_batch_paths = list(src_paths)

    crafted = _crafted_batch(kind)
    o._stack_batch = lambda *a, **k: crafted

    calls = {"move": 0, "partial": 0, "count": 0, "meta": 0}

    def _spy(name):
        def _f(*a, **k):
            calls[name] += 1

        return _f

    o._move_to_stacked = _spy("move")
    o._save_partial_stack = _spy("partial")
    o._update_batch_count_file = _spy("count")
    o._update_batches_meta = _spy("meta")

    with pytest.raises(BatchReductionError):
        o._process_completed_batch(items, 1, 1, None)

    assert calls == {"move": 0, "partial": 0, "count": 0, "meta": 0}
    # Committed counter rolled back (never "attempted" 1).
    assert o.stacked_batches_count == 0
    assert not os.path.exists(o.batch_count_path)
    for p in src_paths:
        assert os.path.exists(p), f"{kind}: source consumed: {p}"
    assert not (src_dir / "stacked").exists()
    _assert_zero_cumulative(o)
    assert o.images_in_cumulative_stack == 0
    assert len(items) == 2
    assert len(o._current_batch_paths) == 2
    # Terminal failure state is set truthfully.
    assert o.stop_processing is True
    assert o.processing_error


@pytest.mark.parametrize("kind", SIX_SILENT_KINDS)
def test_six_silent_paths_legacy_default_unchanged(tmp_path, kind):
    """strict_commit=False (default, out-of-scope callers) keeps the
    historical silent behaviour byte-identical: no raise, no terminal state,
    failed frames counted."""
    o = _tx_stack(tmp_path, f"legacy_{kind}")
    o._support_state_available = False
    stacked, hdr, cov = _crafted_batch(kind)
    # No exception under the legacy default.
    o._combine_batch_result(stacked, hdr, cov)
    assert o.stop_processing is False
    assert o.processing_error is None
    assert o.failed_stack_count == 2
    _assert_zero_cumulative(o)
    assert o.images_in_cumulative_stack == 0




# ---------------------------------------------------------------------------
# 9. CLOSURE REWORK-1 (Nono C-2.1): a never-committed batch's source paths
#    must never be swept (moved/ledgered) by a later commit — the helper
#    scopes the run-level source ledger to the actually-committed batch.
# ---------------------------------------------------------------------------

def _valid_stack_result(shape=(4, 5, 3), n=2):
    hdr = fits.Header()
    hdr["NIMAGES"] = n
    return (
        np.ones(shape, dtype=np.float32),
        hdr,
        np.ones((shape[0], shape[1]), dtype=np.float32),
    )


def test_reproject_no_commit_paths_never_swept_by_later_commit(tmp_path):
    """Nono C-2.1 reproject scenario: a mid-run reproject flush that
    no-commits (unsolved reproject skip, batch dropped) must NOT have its
    source paths consumed by a later successful commit (move_sources=True).
    The never-committed sources stay in their input location, retryable; the
    committed counter / count-file / cumulative reflect ONLY the committed
    batch."""
    o = _tx_stack(tmp_path, "nono_reproj")
    o.reproject_between_batches = True
    o.reproject_coadd_final = False
    o._support_state_available = False
    o.solve_batches = False
    o._reproject_support_tracking_enabled = False
    o.intermediate_classic_batch_files = []
    o.unsolved_classic_batch_files = set()
    src_dir = tmp_path / "input"
    src_dir.mkdir(exist_ok=True)

    def _write(name):
        p = src_dir / name
        p.write_bytes(b"\x00" * 32)
        return str(p)

    a_paths = [_write(f"a{i}.fit") for i in range(2)]
    b_paths = [_write(f"b{i}.fit") for i in range(2)]
    state = {"solved": False}

    def fake_save_and_solve(*a, **k):
        # Reproject solve outcome: mid-run batch unsolved (skip), later batch
        # solved (commit).
        o._last_classic_batch_solved = state["solved"]
        return (None, None)

    o._save_and_solve_classic_batch = fake_save_and_solve
    o._stack_batch = lambda *a, **k: _valid_stack_result(n=2)

    imgs = _images(n=2)
    # Phase 1 — mid-run reproject flush: unsolved -> non-fatal no-commit; the
    # caller (reproject in-loop) drops the in-memory batch.
    items_a = [_item(im) for im in imgs]
    o._current_batch_paths = list(a_paths)
    committed = o._process_completed_batch(
        items_a, 1, 1, None,
        reproject_batch=True, move_sources=False, save_partial=True,
        update_count_file=False, clear_paths=True,
    )
    assert committed is False
    items_a.clear()  # historical in-loop drop of the in-memory batch
    assert o.stacked_batches_count == 0
    for p in a_paths:
        assert os.path.exists(p)  # still in input, never moved

    # Phase 2 — end-of-run reproject final-partial flush: a NEW batch commits
    # with move_sources=True.  The stale never-committed paths must NOT be
    # swept.
    state["solved"] = True
    o._current_batch_paths = list(a_paths) + list(b_paths)  # stale + new
    items_b = [_item(im) for im in _images(n=2, seed=5)]
    committed2 = o._process_completed_batch(
        items_b, 2, 2, None,
        reproject_batch=True, move_sources=True, save_partial=True,
        update_count_file=True, update_meta=True, clear_paths=True,
    )
    assert committed2 is True
    assert o.stacked_batches_count == 1
    assert Path(o.batch_count_path).read_text(encoding="utf-8").strip() == "1"
    assert o.images_in_cumulative_stack == 2  # only the committed batch
    # Committed batch sources moved; never-committed sources REMAIN in input.
    for p in b_paths:
        assert not os.path.exists(p), "committed source must be consumed"
        assert (src_dir / "stacked" / os.path.basename(p)).exists()
    for p in a_paths:
        assert os.path.exists(p), "never-committed source must NOT be moved"
        assert not (src_dir / "stacked" / os.path.basename(p)).exists()
    # Ledger was scoped to the committed batch then cleared (no stale residue).
    assert o._current_batch_paths == []


def test_classic_no_commit_retry_keeps_legitimate_consumption(tmp_path):
    """Nono C-2.1 classic counterpart: a classic no-commit batch that is
    RETAINED for retry must still be consumed by the later successful retry
    commit (legitimate retry — ledger == items, no stale sweep)."""
    o = _tx_stack(tmp_path, "nono_classic")
    o._support_state_available = False
    src_dir = tmp_path / "input"
    src_dir.mkdir(exist_ok=True)

    def _write(name):
        p = src_dir / name
        p.write_bytes(b"\x00" * 32)
        return str(p)

    a_path = _write("a.fit")
    b_path = _write("b.fit")
    a_img, b_img = _images(n=2)

    # Batch 1: no reducer output (all filtered) -> False; classic keeps the
    # in-memory batch AND its ledger paths for retry.
    o._current_batch_paths = [a_path]
    items = [_item(a_img)]
    o._stack_batch = lambda *a, **k: (None, None, None)
    assert o._process_completed_batch(items, 1, 2, None) is False
    assert o.stacked_batches_count == 0
    assert os.path.exists(a_path)

    # Caller keeps the item; a second frame is appended; the retry flush now
    # commits BOTH (ledger == items -> legitimate retry consumption).
    items.append(_item(b_img))
    o._current_batch_paths = [a_path, b_path]
    o._stack_batch = lambda *a, **k: _valid_stack_result(n=2)
    assert o._process_completed_batch(items, 2, 2, None) is True
    assert o.stacked_batches_count == 1
    assert o.images_in_cumulative_stack == 2
    assert (src_dir / "stacked" / "a.fit").exists()
    assert (src_dir / "stacked" / "b.fit").exists()
    assert not os.path.exists(a_path)
    assert not os.path.exists(b_path)




# ---------------------------------------------------------------------------
# 1. Real-pipeline conversion: a reducer-stage exception inside the REAL
#    ``_stack_batch`` (mean path, support staging) is wrapped into
#    BatchReductionError — never an ambiguous (None, None, None).
# ---------------------------------------------------------------------------

def test_stack_batch_converts_in_pipeline_reducer_failure(tmp_path):
    o = _tx_stack(tmp_path, "conv")
    imgs = _images(n=2)
    items = [_item(im) for im in imgs]

    def boom(*a, **k):
        raise MemoryError("mean reducer allocation failure")

    o._stage_batch_support = boom
    with pytest.raises(BatchReductionError) as ei:
        o._stack_batch(items, 1, 1)
    assert "MemoryError" in str(ei.value)


# ---------------------------------------------------------------------------
# 2. Failure matrix — no consume / no advance / terminal truthful raise.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kind,exc",
    [
        # FULL_CPU reducer OOM.
        ("FULL_CPU reducer", MemoryError("FULL_CPU reducer allocation failure")),
        # SPATIAL_TILED_CPU reducer OOM (canonical class of a tiled CPU run).
        ("SPATIAL_TILED_CPU reducer", MemoryError("SPATIAL_TILED_CPU tile allocation failure")),
        # GPU kernel failure whose CPU-fallback reducer also raises.
        ("GPU->CPU fallback reduction", RuntimeError("GPU kernel failed; CPU fallback reducer raised")),
        # Minimum-tile CPU refusal (stage C refusal class).
        ("minimum-tile CPU refusal", CpuWinsorMemoryRefused("cpu_min_tile_exceeds_budget")),
        # Generic reducer exception.
        ("generic reducer exception", ValueError("reducer boom")),
    ],
)
def test_failure_matrix_no_consume_no_advance(tmp_path, kind, exc):
    o = _tx_stack(tmp_path, "mx")
    src_dir, src_paths = _write_sources(tmp_path, ["a.fit", "b.fit"])
    imgs = _images(n=2)
    items = [_item(im) for im in imgs]
    o._current_batch_paths = list(src_paths)
    # Reducer failure surface: the stage D conversion inside ``_stack_batch``
    # already turned the reducer exception into a BatchReductionError (see
    # the real-pipeline conversion test); the helper must treat it as fatal.
    def failing_stack(*a, **k):
        raise BatchReductionError(
            f"{kind}: {type(exc).__name__}: {exc}", cause=exc
        ) from exc

    o._stack_batch = failing_stack
    with pytest.raises(BatchReductionError):
        o._process_completed_batch(items, 1, 1, None)

    # No committed-counter advance.
    assert o.stacked_batches_count == 0
    # No count-file / meta advance.
    assert not os.path.exists(o.batch_count_path)
    # No source movement: both files remain in their input location.
    for p in src_paths:
        assert os.path.exists(p), f"{kind}: source was consumed: {p}"
    assert not (src_dir / "stacked").exists()
    # No resume-ledger / checkpoint advance (checkpoint disabled here and the
    # batch never reached a commit), no partial-cumulative mutation.
    assert o.images_in_cumulative_stack == 0
    _assert_zero_cumulative(o)
    # Batch list is not cleared by a fatal path (nothing was consumed).
    assert len(items) == 2
    assert len(o._current_batch_paths) == 2


@pytest.mark.parametrize(
    "exc",
    [
        MemoryError("raw reducer OOM escaping a seam"),
        RuntimeError("raw reducer crash"),
    ],
)
def test_raw_reducer_exception_escape_still_consumes_nothing(tmp_path, exc):
    """Even a raw (unconverted) reducer exception that reaches the helper
    directly must never consume sources or advance committed state — the
    worker's outer fatal handler turns it into a truthful FAILED run."""
    o = _tx_stack(tmp_path, "raw")
    src_dir, src_paths = _write_sources(tmp_path, ["a.fit", "b.fit"])
    imgs = _images(n=2)
    items = [_item(im) for im in imgs]
    o._current_batch_paths = list(src_paths)

    def failing_stack(*a, **k):
        raise exc

    o._stack_batch = failing_stack
    with pytest.raises(type(exc)):
        o._process_completed_batch(items, 1, 1, None)
    assert o.stacked_batches_count == 0
    assert not os.path.exists(o.batch_count_path)
    for p in src_paths:
        assert os.path.exists(p)
    assert not (src_dir / "stacked").exists()
    _assert_zero_cumulative(o)


# ---------------------------------------------------------------------------
# 3. Success-path control: commit happens once, then sources move.
# ---------------------------------------------------------------------------

def test_successful_batch_commits_then_consumes(tmp_path):
    o = _tx_stack(tmp_path, "ok")
    src_dir, src_paths = _write_sources(tmp_path, ["a.fit", "b.fit"])
    imgs = _images(n=2, base=100.0)
    items = [_item(im) for im in imgs]
    o._current_batch_paths = list(src_paths)

    ok = o._process_completed_batch(items, 1, 1, None)
    assert ok is True
    assert o.stacked_batches_count == 1
    assert Path(o.batch_count_path).read_text(encoding="utf-8").strip() == "1"
    assert o.images_in_cumulative_stack == 2
    # Sources consumed ONLY after the durable commit: both moved under
    # input/stacked/.
    assert (src_dir / "stacked" / "a.fit").exists()
    assert (src_dir / "stacked" / "b.fit").exists()
    for p in src_paths:
        assert not os.path.exists(p)
    # Batch ledger cleared after commit.
    assert len(items) == 0
    assert o._current_batch_paths == []
    # Cumulative state is exactly the committed batch.
    assert not np.all(o.cumulative_wht_memmap == 0)


# ---------------------------------------------------------------------------
# 3b. Reproject-flavoured completion (mechanical): the same transactional
#     helper serves the reproject batch callers (persist+solve then commit
#     then consume) without touching Reproject science.
# ---------------------------------------------------------------------------

def test_reproject_flavoured_completion_commits_then_consumes(tmp_path):
    o = _tx_stack(tmp_path, "reproj")
    o.solve_batches = False
    o._reproject_support_tracking_enabled = False
    o.intermediate_classic_batch_files = []
    o.unsolved_classic_batch_files = set()
    src_dir, src_paths = _write_sources(tmp_path, ["a.fit", "b.fit"])
    imgs = _images(n=2, base=100.0)
    items = [_item(im) for im in imgs]
    o._current_batch_paths = list(src_paths)

    ok = o._process_completed_batch(
        items,
        1,
        1,
        None,
        reproject_batch=True,
        move_sources=True,
        save_partial=True,
        update_count_file=True,
        update_meta=True,
        clear_paths=True,
    )
    assert ok is True
    assert o.stacked_batches_count == 1
    assert Path(o.batch_count_path).read_text(encoding="utf-8").strip() == "1"
    assert (src_dir / "stacked" / "a.fit").exists()
    assert (src_dir / "stacked" / "b.fit").exists()
    assert len(items) == 0
    assert o._current_batch_paths == []
    # The batch was persisted (reproject batch file) before consumption.
    batch_dir = Path(o.output_folder) / "classic_batch_outputs"
    assert (batch_dir / "classic_batch_001.fits").exists()


def test_reproject_flavoured_failure_consumes_nothing(tmp_path):
    o = _tx_stack(tmp_path, "reprojfail")
    o.solve_batches = False
    o.intermediate_classic_batch_files = []
    o.unsolved_classic_batch_files = set()
    src_dir, src_paths = _write_sources(tmp_path, ["a.fit"])
    imgs = _images(n=1)
    items = [_item(im) for im in imgs]
    o._current_batch_paths = list(src_paths)

    def failing_stack(*a, **k):
        raise BatchReductionError(
            "reproject reduce failure", cause=MemoryError("oom")
        )

    o._stack_batch = failing_stack
    with pytest.raises(BatchReductionError):
        o._process_completed_batch(
            items,
            1,
            1,
            None,
            reproject_batch=True,
            move_sources=True,
            save_partial=True,
            update_count_file=True,
            update_meta=True,
            clear_paths=True,
        )
    assert o.stacked_batches_count == 0
    assert not os.path.exists(o.batch_count_path)
    assert os.path.exists(src_paths[0])
    assert not (src_dir / "stacked").exists()
    _assert_zero_cumulative(o)
    assert len(items) == 1
    assert o._current_batch_paths == [src_paths[0]]


# ---------------------------------------------------------------------------
# 4. Counter semantics: committed, never attempted.
# ---------------------------------------------------------------------------

def test_counter_equals_committed_batches_after_midrun_failure(tmp_path):
    o = _tx_stack(tmp_path, "cnt")
    src_dir, src_paths = _write_sources(tmp_path, ["a.fit", "b.fit"])
    imgs = _images(n=2, base=100.0)
    # Batch 1 commits for real.
    items1 = [_item(imgs[0])]
    o._current_batch_paths = [src_paths[0]]
    assert o._process_completed_batch(items1, 1, 2, None) is True
    after_b1 = _snapshot(o)
    assert o.stacked_batches_count == 1

    # Batch 2 reducer fails (post-conversion BatchReductionError).
    items2 = [_item(imgs[1])]
    o._current_batch_paths = [src_paths[1]]

    def failing_stack(*a, **k):
        raise BatchReductionError(
            "batch 2 reducer failure: MemoryError: boom", cause=MemoryError("boom")
        )

    o._stack_batch = failing_stack
    with pytest.raises(BatchReductionError):
        o._process_completed_batch(items2, 2, 2, None)

    # Committed counter == committed batches (1), not attempted (2).
    assert o.stacked_batches_count == 1
    assert Path(o.batch_count_path).read_text(encoding="utf-8").strip() == "1"
    # Batch-2 source untouched; batch-1 source consumed.
    assert os.path.exists(src_paths[1])
    assert (src_dir / "stacked" / "a.fit").exists()
    # Cumulative equals exactly batch 1 (byte-identical before/after failure).
    after_fail = _snapshot(o)
    assert np.array_equal(after_b1[0], after_fail[0])
    assert np.array_equal(after_b1[1], after_fail[1])


# ---------------------------------------------------------------------------
# 5. On-disk resume witness: batch 2 physically remains retryable and a
#    legitimate re-run does NOT skip it.
# ---------------------------------------------------------------------------

def test_on_disk_resume_witness_batch2_retryable(tmp_path):
    o = _tx_stack(tmp_path, "rw")
    src_dir, src_paths = _write_sources(tmp_path, ["a.fit", "b.fit"])
    imgs = _images(n=2, base=100.0)

    # Batch 1 commits.
    o._current_batch_paths = [src_paths[0]]
    assert o._process_completed_batch([_item(imgs[0])], 1, 2, None) is True
    # Batch 2 fails truthfully.
    o._current_batch_paths = [src_paths[1]]

    def failing_stack(*a, **k):
        raise BatchReductionError(
            "batch 2 reducer failure", cause=MemoryError("oom")
        )

    o._stack_batch = failing_stack
    with pytest.raises(BatchReductionError):
        o._process_completed_batch([_item(imgs[1])], 2, 2, None)

    # On disk: batch-1 sources under stacked/, batch-2 source in place.
    assert (src_dir / "stacked" / "a.fit").exists()
    assert os.path.exists(src_paths[1])
    assert not (src_dir / "stacked" / "b.fit").exists()
    assert Path(o.batch_count_path).read_text(encoding="utf-8").strip() == "1"

    # A later legitimate re-run (same output state, batch 2 reducer healthy)
    # must NOT skip batch 2 because of the failed attempt.
    o2 = _tx_stack(tmp_path, "rw2")
    # Carry the committed on-disk state forward as a resume would: batch-1
    # cumulative already committed on disk under rw2, count == 1.
    shutil_memmaps(o, o2)
    Path(o2.batch_count_path).write_text("1", encoding="utf-8")
    o2.stacked_batches_count = 1
    o2.images_in_cumulative_stack = 1
    o2._current_batch_paths = [src_paths[1]]
    assert o2._process_completed_batch([_item(imgs[1])], 2, 2, None) is True
    assert o2.stacked_batches_count == 2
    assert Path(o2.batch_count_path).read_text(encoding="utf-8").strip() == "2"
    assert (src_dir / "stacked" / "b.fit").exists()
    assert not os.path.exists(src_paths[1])
    # Batch-1 science was NOT double-counted by the re-run.
    assert o2.images_in_cumulative_stack == 2


def shutil_memmaps(o_src, o_dst):
    """Copy committed on-disk memmaps from one stack to another."""
    import shutil

    for attr in ("sum_memmap_path", "wht_memmap_path"):
        shutil.copyfile(getattr(o_src, attr), getattr(o_dst, attr))
    o_dst.cumulative_sum_memmap.flush()
    o_dst.cumulative_wht_memmap.flush()


# ---------------------------------------------------------------------------
# 6. Non-fatal no-commit conditions: empty batch and no-reducer-output
#    batches (all items filtered / reproject missing-WCS family) must NOT
#    advance committed state and must NOT move sources.
# ---------------------------------------------------------------------------

def test_empty_batch_nonfatal_no_commit(tmp_path):
    o = _tx_stack(tmp_path, "empty")
    src_dir, src_paths = _write_sources(tmp_path, ["a.fit"])
    o._current_batch_paths = list(src_paths)
    ok = o._process_completed_batch([], 1, 1, None)
    assert ok is False
    assert o.stacked_batches_count == 0
    assert not os.path.exists(o.batch_count_path)
    assert os.path.exists(src_paths[0])
    assert not (src_dir / "stacked").exists()
    _assert_zero_cumulative(o)


def test_no_reducer_output_batch_nonfatal_no_commit(tmp_path):
    """(None, None, None) from _stack_batch == no reducer ran (all items
    filtered out / reproject missing-WCS family): non-fatal, no commit, no
    source movement, batch left retryable."""
    o = _tx_stack(tmp_path, "noc")
    src_dir, src_paths = _write_sources(tmp_path, ["a.fit", "b.fit"])
    imgs = _images(n=2)
    items = [_item(im) for im in imgs]
    o._current_batch_paths = list(src_paths)
    o._stack_batch = lambda *a, **k: (None, None, None)
    ok = o._process_completed_batch(items, 1, 1, None)
    assert ok is False
    assert o.stacked_batches_count == 0
    assert not os.path.exists(o.batch_count_path)
    for p in src_paths:
        assert os.path.exists(p)
    assert not (src_dir / "stacked").exists()
    _assert_zero_cumulative(o)
    assert o.images_in_cumulative_stack == 0
    # Batch list left intact (retryable).
    assert len(items) == 2
    assert len(o._current_batch_paths) == 2


# ---------------------------------------------------------------------------
# 7. _combine_batch_result failure signalling: strict commit failures are
#    surfaced and nothing is consumed afterwards.
# ---------------------------------------------------------------------------

def test_combine_internal_checkpoint_failure_is_terminal(tmp_path):
    """A real combine whose checkpoint dirty-persist fails (the production
    strict path, fail BEFORE any accumulator mutation) must surface as a
    BatchReductionError; the helper rolls back its transient counter advance
    and consumes nothing — the cumulative stays untouched."""
    o = _tx_stack(tmp_path, "ckptfail")
    # This scenario targets the checkpoint guard of the real combine;
    # support-payload tracking is disabled so the COV-01B preflight does not
    # mask the injected checkpoint failure (support wiring is exercised by
    # the success / counter / witness tests above).
    o._support_state_available = False
    src_dir, src_paths = _write_sources(tmp_path, ["a.fit"])
    imgs = _images(n=1)
    o._current_batch_paths = [src_paths[0]]

    def failing_dirty(*a, **k):
        raise _ResumeCheckpointError("checkpoint dirty persist failed")

    o._checkpoint_mark_dirty = failing_dirty
    with pytest.raises(BatchReductionError):
        o._process_completed_batch([_item(imgs[0])], 1, 1, None)
    # Rolled back to committed count (0) — never "attempted" (1).
    assert o.stacked_batches_count == 0
    assert not os.path.exists(o.batch_count_path)
    assert os.path.exists(src_paths[0])
    assert not (src_dir / "stacked").exists()
    # No partial-cumulative mutation from the failed batch.
    _assert_zero_cumulative(o)
    assert o.images_in_cumulative_stack == 0
    # The strict commit path set the terminal failure state truthfully.
    assert o.stop_processing is True
    assert o.processing_error


def test_combine_memory_failure_is_terminal(tmp_path):
    o = _tx_stack(tmp_path, "memfail")
    src_dir, src_paths = _write_sources(tmp_path, ["a.fit"])
    imgs = _images(n=1)
    o._current_batch_paths = [src_paths[0]]

    def mem_commit(*a, **k):
        raise MemoryError("accumulation OOM")

    o._combine_batch_result = mem_commit
    with pytest.raises(BatchReductionError):
        o._process_completed_batch([_item(imgs[0])], 1, 1, None)
    assert o.stacked_batches_count == 0
    assert not os.path.exists(o.batch_count_path)
    assert os.path.exists(src_paths[0])
    assert not (src_dir / "stacked").exists()
    assert o.images_in_cumulative_stack == 0
    _assert_zero_cumulative(o)


def test_combine_generic_accumulation_failure_is_terminal(tmp_path):
    o = _tx_stack(tmp_path, "accfail")
    src_dir, src_paths = _write_sources(tmp_path, ["a.fit"])
    imgs = _images(n=1)
    o._current_batch_paths = [src_paths[0]]

    def boom_commit(*a, **k):
        raise RuntimeError("accumulation exploded")

    o._combine_batch_result = boom_commit
    with pytest.raises(BatchReductionError):
        o._process_completed_batch([_item(imgs[0])], 1, 1, None)
    assert o.stacked_batches_count == 0
    assert not os.path.exists(o.batch_count_path)
    assert os.path.exists(src_paths[0])
    assert not (src_dir / "stacked").exists()
    assert o.images_in_cumulative_stack == 0
    _assert_zero_cumulative(o)
    assert o.stop_processing is True
    assert o.processing_error


def test_combine_bre_direct_raise_rolls_back_and_reraises(tmp_path):
    o = _tx_stack(tmp_path, "bredir")
    src_dir, src_paths = _write_sources(tmp_path, ["a.fit"])
    imgs = _images(n=1)
    o._current_batch_paths = [src_paths[0]]

    def bre_commit(*a, **k):
        raise BatchReductionError("durable commit refused")

    o._combine_batch_result = bre_commit
    with pytest.raises(BatchReductionError):
        o._process_completed_batch([_item(imgs[0])], 1, 1, None)
    assert o.stacked_batches_count == 0
    assert os.path.exists(src_paths[0])
    assert not (src_dir / "stacked").exists()
    _assert_zero_cumulative(o)


def test_combine_legacy_non_strict_default_unchanged(tmp_path):
    """Out-of-scope callers (Drizzle cached reprojection, direct unit calls)
    keep the historical swallow-and-stop behaviour by default."""
    o = _tx_stack(tmp_path, "legacy")
    # Same masking note as above: this scenario targets the memmap guard.
    o._support_state_available = False
    o.cumulative_sum_memmap[:] = 0.0
    # Force the memmap-missing guard (legacy non-strict swallow).
    o.cumulative_sum_memmap = None
    o.cumulative_wht_memmap = None
    hdr = fits.Header()
    hdr["NIMAGES"] = 1
    o._combine_batch_result(
        np.ones((4, 5, 3), dtype=np.float32), hdr, np.ones((4, 5), dtype=np.float32)
    )
    assert o.stop_processing is True
    assert o.processing_error == "Memmap non initialisé"
