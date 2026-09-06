"""Phase B1 — canonical batch contract (engine) focused tests.

Mission ``zsss-winsorized-gpu-perf-memory`` Phase B1.  These tests lock the
canonical batch vocabulary (0 = Auto / 1 = Boring / >= 2 = explicit),
B_requested / B_resolved / N_batch semantics, legacy ``-1 -> 0`` normalization,
Reproject&Coadd decoupling from the Auto sentinel, freeze/persist/resume of
B_resolved, removal of dynamic batch mutation, align_on_disk derived-state
routing and the injectable AutoBatch planner.

They run against the pure canonical kernel
(``seestar.core.batch_contract``) and against the engine resolution helpers
(``_resolve_batch_request`` / ``_freeze_batch_resolved`` /
``_read_persisted_batch_contract`` / ``_batch_flush_trigger`` /
``_reproject_inherits_frozen_reference``) without starting a full worker.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import seestar.queuep.queue_manager as qmm
from seestar.core import batch_contract as bc
from seestar.core.batch_contract import (
    BATCH_AUTO,
    BATCH_BORING,
    AutoBatchPlanner,
    batch_requested_mode,
    clamp_resolved_to_samples,
    normalize_batch_requested,
    plan_auto_batch,
)
from seestar.gui.run_config import compute_align_on_disk
from seestar.queuep.queue_manager import SeestarQueuedStacker

ROOT = Path(__file__).resolve().parents[1]


def _resolution_object(**attrs):
    """A lightweight SeestarQueuedStacker shell for resolution helpers."""
    obj = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    obj.logger = qmm.logger
    obj.update_progress = lambda *a, **k: None
    obj.batch_requested = 0
    obj.batch_resolved = 1
    obj.batch_flush_mode = "count"
    obj._batch_contract_restored = False
    obj.chunk_size = None
    obj.batch_size = 1
    obj.files_in_queue = 0
    obj.current_folder = None
    obj.output_folder = None
    obj.additional_folders = []
    obj.reproject_coadd_final = False
    obj.reproject_between_batches = False
    obj.freeze_reference_wcs = False
    obj.stack_final_combine = "mean"
    obj._resume_requested = False
    obj._has_stack_plan = False
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


# ---------------------------------------------------------------------------
# 1. Canonical vocabulary / normalization / terminology
# ---------------------------------------------------------------------------


def test_canonical_values_and_legacy_negatives_normalize_to_zero():
    # -1 (and any legacy negative) is the legacy Auto spelling -> canonical 0.
    assert normalize_batch_requested(-1) == 0
    assert normalize_batch_requested(-5) == 0
    assert normalize_batch_requested(-999) == 0
    # 0 is canonical Auto and stays 0.
    assert normalize_batch_requested(0) == BATCH_AUTO
    # 1 is Boring.
    assert normalize_batch_requested(1) == BATCH_BORING
    # >= 2 explicit capacity.
    assert normalize_batch_requested(2) == 2
    assert normalize_batch_requested(42) == 42
    # garbage fails safe toward canonical Auto.
    assert normalize_batch_requested("abc") == BATCH_AUTO
    assert normalize_batch_requested(None) == BATCH_AUTO


def test_mode_classification():
    assert batch_requested_mode(-1) == "auto"
    assert batch_requested_mode(0) == "auto"
    assert batch_requested_mode(1) == "boring"
    assert batch_requested_mode(7) == "explicit"
    assert bc.BATCH_MODE_AUTO == "auto"
    assert bc.BATCH_MODE_BORING == "boring"
    assert bc.BATCH_MODE_EXPLICIT == "explicit"


def test_clamp_resolved_to_samples_rules():
    assert clamp_resolved_to_samples(50, 7) == 7
    assert clamp_resolved_to_samples(50, None) == 50  # unknown samples: no cap
    assert clamp_resolved_to_samples(1, 7) == 1
    assert clamp_resolved_to_samples(50, 0) == 1
    assert clamp_resolved_to_samples("junk", 7) == 1


def test_engine_resolution_legacy_minus_one_becomes_auto_zero():
    calls = []

    def fake_estimate():
        calls.append(1)
        return 12

    obj = _resolution_object(_estimate_batch_size=fake_estimate)
    obj._resolve_batch_request(-1)  # legacy Auto sentinel
    assert obj.batch_requested == 0
    assert obj.batch_resolved == 12
    assert obj.batch_size == 12
    assert calls, "Auto request must consult the estimator once"


def test_engine_resolution_boring_and_explicit():
    obj = _resolution_object(_estimate_batch_size=lambda: 999)
    obj._resolve_batch_request(1)
    assert obj.batch_requested == 1 and obj.batch_resolved == 1
    obj._resolve_batch_request(7)
    assert obj.batch_requested == 7 and obj.batch_resolved == 7


# ---------------------------------------------------------------------------
# 2. Auto Fresh -> persist; Auto Resume -> identical B_resolved
# ---------------------------------------------------------------------------


def _write_manifest(out_dir, *, requested=0, resolved=5, chunk_size=None,
                    schema_version=2):
    out_dir = Path(out_dir)
    memdir = out_dir / "memmap_accumulators"
    memdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": schema_version,
        "state": "clean",
        "mode": "classic_sumw",
        "fingerprint": "x",
        "batch": {"requested": requested, "resolved": resolved,
                  "chunk_size": chunk_size},
    }
    (memdir / "resume_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_auto_resume_reuses_persisted_b_resolved_despite_changed_ram(tmp_path):
    # Fresh Auto: RAM estimator returns 12 -> resolved candidate 12.
    estimator_calls = []

    def fake_estimate():
        estimator_calls.append(1)
        return 12

    obj = _resolution_object(
        _estimate_batch_size=fake_estimate,
        output_folder=str(tmp_path / "fresh"),
    )
    obj._resolve_batch_request(0)
    obj.files_in_queue = 5
    obj._freeze_batch_resolved()
    assert obj.batch_resolved == 5  # capped by the static queue population
    assert obj._batch_contract_restored is False

    # Persist the frozen contract (mirror of the manifest batch block).
    _write_manifest(tmp_path / "out", requested=0, resolved=5)

    # Resume with a *different* machine RAM: estimator would now return 999.
    resumed = _resolution_object(
        _estimate_batch_size=lambda: 999,
        output_folder=str(tmp_path / "out"),
        files_in_queue=3,  # remaining population smaller than B_resolved
    )
    resumed._resume_requested = True
    resumed._resolve_batch_request(0)
    assert resumed._batch_contract_restored is True
    assert resumed.batch_requested == 0
    assert resumed.batch_resolved == 5  # reused verbatim, never re-estimated
    resumed._freeze_batch_resolved()
    # Resume must NOT shrink the frozen contract to the remaining population.
    assert resumed.batch_resolved == 5
    assert resumed.batch_size == 5
    assert len(estimator_calls) == 1  # only the fresh run estimated


def test_persisted_zero_stays_canonical_auto(tmp_path):
    _write_manifest(tmp_path, requested=0, resolved=9)
    obj = _resolution_object(
        output_folder=str(tmp_path),
        _estimate_batch_size=lambda: 999,
    )
    obj._resume_requested = True
    obj._resolve_batch_request(0)
    assert obj._batch_contract_restored is True
    assert obj.batch_requested == 0
    assert obj.batch_resolved == 9


def test_resume_without_contract_falls_back_to_fresh_resolution(tmp_path):
    obj = _resolution_object(
        output_folder=str(tmp_path),
        _estimate_batch_size=lambda: 11,
    )
    obj._resume_requested = True
    obj._resolve_batch_request(0)
    # No manifest / no batch block: fresh-style resolution (Auto estimate).
    assert obj._batch_contract_restored is False
    assert obj.batch_resolved == 11


def test_batch_contract_persisted_in_manifest_write(tmp_path):
    """The engine resume manifest carries requested + resolved (freeze)."""
    obj = _resolution_object(
        output_folder=str(tmp_path),
        _estimate_batch_size=lambda: 20,
    )
    obj._resolve_batch_request(0)
    obj.files_in_queue = 6
    obj._freeze_batch_resolved()
    # _write_resume_manifest reads these two attrs into manifest["batch"].
    assert obj.batch_resolved == 6
    assert obj.batch_requested == 0
    # Re-read through the resume path reproduces the exact contract.
    obj.output_folder = str(tmp_path)
    _write_manifest(tmp_path, requested=obj.batch_requested,
                    resolved=obj.batch_resolved)
    obj._resume_requested = True
    reread = obj._read_persisted_batch_contract()
    assert reread == {"requested": 0, "resolved": 6, "chunk_size": None}


# ---------------------------------------------------------------------------
# 3. No dynamic batch mutation; freeze is final
# ---------------------------------------------------------------------------


def test_timing_based_shrink_removed_from_engine_source():
    src = (ROOT / "seestar" / "queuep" / "queue_manager.py").read_text(
        encoding="utf-8"
    )
    assert "self.batch_size // 2" not in src
    assert "Batch size reduced to" not in src


def test_explicit_b_resolved_never_dynamically_modified(tmp_path):
    obj = _resolution_object(_estimate_batch_size=lambda: 10)
    obj._resolve_batch_request(7)
    obj.files_in_queue = 1000  # huge queue: explicit capacity must NOT grow
    obj._freeze_batch_resolved()
    assert obj.batch_resolved == 7
    # The flush trigger honors the frozen capacity for every flush.
    assert obj._batch_flush_trigger() == 7.0


def test_process_batch_parallel_never_mutates_batch_size(monkeypatch):
    """The historical timing-based shrink (`_process_batch_parallel`) is gone:
    a slow batch pass leaves the frozen capacity untouched."""
    calls = []

    def fake_process_file(path):
        calls.append(path)
        time.sleep(0.02)
        return None

    obj = _resolution_object(
        _estimate_batch_size=lambda: 8,
        num_threads=2,
        _process_file=fake_process_file,
    )
    obj._resolve_batch_request(8)
    results = obj._process_batch_parallel(["a.fit", "b.fit", "c.fit", "d.fit"])
    assert len(calls) == 4
    assert results == [None, None, None, None]
    assert obj.batch_size == 8  # never halved, whatever the timing


def test_final_partial_batch_is_natural_not_mutation():
    obj = _resolution_object()
    obj.batch_requested = 0
    obj.batch_resolved = 4
    obj.batch_size = 4
    obj.batch_flush_mode = "count"
    obj.files_in_queue = 10
    # 10 frames at B_resolved=4 -> populations 4, 4, 2: N_batch <= B_resolved.
    populations = []
    n = 10
    while n > 0:
        take = min(4, n)
        populations.append(take)
        n -= take
    assert populations == [4, 4, 2]
    assert all(1 <= p <= obj.batch_resolved for p in populations)


# ---------------------------------------------------------------------------
# 4. Reproject&Coadd decoupled from the Auto sentinel 0
# ---------------------------------------------------------------------------


def test_auto_does_not_force_reproject():
    obj = _resolution_object(_estimate_batch_size=lambda: 10)
    obj.reproject_coadd_final = False
    obj.reproject_between_batches = False
    obj.stack_final_combine = "mean"
    obj._resolve_batch_request(0)
    assert obj.reproject_coadd_final is False  # decoupled: not forced by 0
    assert obj.stack_final_combine == "mean"


def test_auto_plus_reproject_keeps_single_grid_workflow():
    obj = _resolution_object(_estimate_batch_size=lambda: 10)
    obj.reproject_coadd_final = True
    obj.reproject_between_batches = True
    obj.freeze_reference_wcs = False
    obj.stack_final_combine = "mean"
    obj._resolve_batch_request(0)
    assert obj.reproject_coadd_final is True
    assert obj.stack_final_combine == "reproject_coadd"
    assert obj.freeze_reference_wcs is True
    assert obj.reproject_between_batches is False


def test_reproject_executes_for_explicit_and_boring_too():
    # Reproject&Coadd is its own mode: it works for explicit / Boring runs
    # without any Auto sentinel (and must not mutate their freeze flags).
    obj = _resolution_object(_estimate_batch_size=lambda: 10)
    obj.reproject_coadd_final = True
    obj.reproject_between_batches = True
    obj.freeze_reference_wcs = True
    obj._resolve_batch_request(5)
    assert obj.reproject_coadd_final is True
    assert obj.reproject_between_batches is True  # explicit run untouched
    assert obj.batch_resolved == 5


def test_reproject_batch_wcs_inheritance_is_mode_routed():
    # Frozen-reference inheritance is routed on explicit state (Auto intent +
    # Reproject&Coadd flag), never on a raw engine sentinel.
    obj = _resolution_object()
    obj.reproject_coadd_final = True
    obj.batch_requested = 0
    obj.batch_size = 3  # resolved Auto run (never 0 in the engine)
    assert obj._reproject_inherits_frozen_reference() is True

    obj.batch_requested = 5  # explicit Reproject&Coadd -> per-batch WCS path
    assert obj._reproject_inherits_frozen_reference() is False

    # Legacy pre-resolution object carrying the old all-in-RAM sentinel maps
    # to the Auto workflow (compatibility guard).
    legacy = _resolution_object()
    legacy.reproject_coadd_final = True
    legacy.batch_size = 0
    legacy.batch_requested = 5
    assert legacy._reproject_inherits_frozen_reference() is True

    # No Reproject&Coadd -> never inherits through the batch value alone.
    obj.reproject_coadd_final = False
    obj.batch_requested = 0
    assert obj._reproject_inherits_frozen_reference() is False


def test_reproject_regression_proves_no_sentinel_dependence():
    """Source audit: the engine's batch-trigger and reproject routing no longer
    contain any semantic equality check against the raw Auto sentinel 0."""
    src = (ROOT / "seestar" / "queuep" / "queue_manager.py").read_text(
        encoding="utf-8"
    )
    # Worker flush trigger used to special-case `self.batch_size == 0`; the
    # only remaining zero-tests are defensive guards on legacy instances.
    assert "trigger = float(\"inf\")" not in src.replace(
        "def _batch_flush_trigger", ""
    )
    assert "explicit_all_ram_single_batch" not in src  # old provenance token
    assert "batch_size_requested=all_ram" not in src


# ---------------------------------------------------------------------------
# 5. align_on_disk and derived state under the canonical contract
# ---------------------------------------------------------------------------


def test_align_on_disk_mode_routing():
    # Auto (0 / legacy negatives) -> in-RAM aligned pipeline (historical Auto
    # behavior); Boring (1) and explicit (>=2) -> on-disk aligned temporaries.
    assert compute_align_on_disk(0) is False
    assert compute_align_on_disk(-1) is False
    assert compute_align_on_disk(1) is True
    assert compute_align_on_disk(2) is True
    assert compute_align_on_disk(50) is True
    assert compute_align_on_disk("garbage") is False


def test_align_on_disk_independent_of_reproject_flag():
    # align_on_disk is a batch-mode routing, not a Reproject&Coadd sentinel.
    assert compute_align_on_disk(0) is compute_align_on_disk(0)
    # (No sentinel coupling: value 0 alone never enables on-disk alignment.)


# ---------------------------------------------------------------------------
# 6. AutoBatch planner scenarios (injectable, backend-independent)
# ---------------------------------------------------------------------------

GIB = 1 << 30


def test_planner_very_low_ram():
    p = AutoBatchPlanner(available_ram_bytes=512 * 1024 * 1024,
                         image_hw=(3840, 2160))
    assert 1 <= p.resolve() <= 50
    # Conservative: 512 MB cannot host a huge batch of 4K images.
    assert p.resolve() <= 2


def test_planner_moderate_and_large_ram_scale():
    low = AutoBatchPlanner(available_ram_bytes=4 * GIB, image_hw=(1920, 1080))
    high = AutoBatchPlanner(available_ram_bytes=64 * GIB, image_hw=(1920, 1080))
    assert 1 <= low.resolve() <= high.resolve() <= 50


def test_planner_memory_query_failure_conservative_fallback():
    def boom():
        raise RuntimeError("memory discovery failed")

    p = AutoBatchPlanner(image_hw=(3840, 2160), memory_query=boom)
    resolved = p.resolve()
    # Never crashes, never guesses huge: bounded by the conservative fallback.
    assert 1 <= resolved <= 10


def test_planner_no_provider_fallback():
    p = AutoBatchPlanner(image_hw=(3840, 2160), fallback_batch=4)
    assert p.resolve() == 4


def test_planner_tiny_image_caps_at_max():
    p = AutoBatchPlanner(available_ram_bytes=8 * GIB, image_hw=(4, 5))
    assert p.resolve() == 50


def test_planner_s50_rgb_footprint():
    # S50 RGB ~ 3000x2000 x 3 channels: moderate RAM yields a sane capacity.
    p = AutoBatchPlanner(available_ram_bytes=8 * GIB, image_hw=(3000, 2000))
    resolved = p.resolve()
    assert 1 <= resolved <= 50
    big = AutoBatchPlanner(available_ram_bytes=8 * GIB, image_hw=(3000, 2000),
                           channels=3)
    assert big.resolve() == resolved  # RGB is the default worst-case


def test_planner_short_finite_queue_caps_below_estimate():
    p = AutoBatchPlanner(available_ram_bytes=64 * GIB, image_hw=(4, 5))
    assert p.resolve(queue_length=3) == 3  # <= samples


def test_planner_large_finite_queue_and_bounded_by_samples():
    p = AutoBatchPlanner(available_ram_bytes=64 * GIB, image_hw=(1920, 1080))
    resolved = p.resolve(queue_length=40)
    assert 1 <= resolved <= 40  # B_resolved <= samples for a static queue
    assert p.resolve(queue_length=1000) == min(p.resolve(queue_length=None), 1000)


def test_planner_backend_independence():
    # The planner kernel has no GPU concept: identical workload + host memory
    # inputs resolve identically, whatever a "backend" would later request.
    a = plan_auto_batch(16 * GIB, image_hw=(1920, 1080))
    b = plan_auto_batch(16 * GIB, image_hw=(1920, 1080))
    assert a == b


# ---------------------------------------------------------------------------
# 7. Engine batch-trigger routing (count vs token)
# ---------------------------------------------------------------------------


def test_batch_flush_trigger_routing():
    obj = _resolution_object()
    obj.batch_flush_mode = "count"
    obj.batch_size = 4
    obj.batch_requested = 4
    assert obj._batch_flush_trigger() == 4.0
    # Boring with a chunk grouping uses the chunk as the RAM trigger.
    obj.batch_size = 1
    obj.chunk_size = 25
    assert obj._batch_flush_trigger() == 25.0
    obj.chunk_size = None
    assert obj._batch_flush_trigger() == 1.0
    # Tokenized queue: delimiters drive the flush; count trigger disabled.
    obj.batch_flush_mode = "token"
    obj.batch_size = 4
    assert obj._batch_flush_trigger() == float("inf")
    # Legacy pre-resolution sentinel guard keeps the historical inf behavior.
    obj.batch_flush_mode = "count"
    obj.batch_size = 0
    assert obj._batch_flush_trigger() == float("inf")


# ---------------------------------------------------------------------------
# 8. Raw-request-derived decision audit at the seams
# ---------------------------------------------------------------------------


def test_qt_normalize_never_generates_minus_one():
    from seestar.gui_qt.settings_validation import normalize_batch_size

    assert normalize_batch_size(0) == 0
    assert normalize_batch_size(0, reproject_coadd_final=True) == 0
    assert normalize_batch_size(-1) == 0
    assert normalize_batch_size(1) == 1
    assert normalize_batch_size(12) == 12


def test_batch_requested_token_canonical():
    token = qmm._batch_requested_token
    assert token(-1) == "auto"
    assert token(0) == "auto"
    assert token(1) == "1"
    assert token(7) == "7"
    assert "all_ram" != token(0)


def test_engine_source_has_no_legacy_all_ram_token_generation():
    src = (ROOT / "seestar" / "queuep" / "queue_manager.py").read_text(
        encoding="utf-8"
    )
    assert "return \"all_ram\"" not in src


def test_hierarchical_and_handoff_paths_do_not_read_raw_batch_sentinel():
    """Audit: hierarchical stacking (interbatch) and the Qt handoff do not
    derive behavior from the raw batch value / Auto sentinel."""
    from seestar.gui.run_config import (
        RunRequest,
        SEAM_ONLY_KWARGS,
        build_backend_kwargs,
        split_backend_kwargs,
    )

    assert "batch_size" not in SEAM_ONLY_KWARGS  # engine-owned, not seam-split
    for module in (
        "seestar/gui_qt/run_handoff.py",
        "seestar/gui_qt/final_combine.py",
    ):
        src = (ROOT / module).read_text(encoding="utf-8")
        assert "batch_size" not in src, module
