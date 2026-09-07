"""Phase H (P6) — truthful GPU execution provenance: focused unit tests.

Covers the execution-truth telemetry store, the fallback-reason catalog
mapping, the batch-provenance tokens, and the aggregate summary — the
provenance layer added at the reduction-dispatch seam.  These are pure
host-side unit tests: no device, no real dispatch.
"""

from __future__ import annotations

import pytest

from seestar.queuep.queue_manager import (
    GPU_EXEC_REASON_GPU_KERNEL,
    GPU_EXEC_REASON_POLICY_CPU,
    GPU_EXEC_REASON_VRAM_NO_VALID_TILE,
    SeestarQueuedStacker,
    _gpu_execution_reason_token,
)


def _mk_stacker(**attrs) -> SeestarQueuedStacker:
    """Duck-typed stacker (no __init__): the telemetry methods use getattr
    defensively so a __new__-constructed instance is enough."""
    s = SeestarQueuedStacker.__new__(SeestarQueuedStacker)
    for k, v in attrs.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------------------
# fallback-reason catalog mapping
# ---------------------------------------------------------------------------


def test_reason_token_maps_legacy_codes():
    assert _gpu_execution_reason_token(None) is None
    assert _gpu_execution_reason_token("policy_cpu") == "policy_cpu"
    assert _gpu_execution_reason_token("backend_unavailable") == "backend_unavailable"
    assert _gpu_execution_reason_token("cupy_import") == "cupy_import_failure"
    assert _gpu_execution_reason_token("vram_reject") == "vram_no_valid_tile"
    assert _gpu_execution_reason_token("planner_failure") == "planner_failure"
    assert _gpu_execution_reason_token("meminfo_failure") == "runtime_memory_failure"
    # unknown codes pass through untouched
    assert _gpu_execution_reason_token("weird_code") == "weird_code"


# ---------------------------------------------------------------------------
# batch provenance tokens (B_requested / B_resolved / N_batch separation)
# ---------------------------------------------------------------------------


def test_batch_provenance_auto():
    s = _mk_stacker(batch_requested=0, batch_resolved=37)
    t = s._gpu_batch_provenance_tokens()
    assert t["batch_size_requested"] == 0
    assert t["batch_mode"] == "auto"
    assert t["batch_size_resolved"] == 37
    assert t["batch_size_reason"] == "auto_resolved_frozen"


def test_batch_provenance_boring():
    s = _mk_stacker(batch_requested=1, batch_resolved=1)
    t = s._gpu_batch_provenance_tokens()
    assert t["batch_size_requested"] == 1
    assert t["batch_mode"] == "boring"
    assert t["batch_size_resolved"] == 1
    assert t["batch_size_reason"] == "requested_frozen"


def test_batch_provenance_explicit():
    s = _mk_stacker(batch_requested=50, batch_resolved=50)
    t = s._gpu_batch_provenance_tokens()
    assert t["batch_size_requested"] == 50
    assert t["batch_mode"] == "explicit"
    assert t["batch_size_resolved"] == 50
    assert t["batch_size_reason"] == "requested_frozen"


# ---------------------------------------------------------------------------
# per-reduction execution-truth record
# ---------------------------------------------------------------------------


def _record(s, **kw):
    defaults = dict(
        operation="winsorized_sigma_clip",
        scientific_N_batch=12,
        workload_shape=(12, 480, 270),
        backend_requested="cupy",
        eligible=True,
        attempted=True,
        executed="gpu",
        gpu_memory_mode="full",
    )
    defaults.update(kw)
    s._record_gpu_execution(**defaults)


def test_record_gpu_execution_success_truth():
    s = _mk_stacker()
    _record(s)
    (ev,) = s._gpu_execution_events
    assert ev["executed"] == "gpu"
    assert ev["gpu_memory_mode"] == "full"
    assert ev["fallback"] is False
    assert ev["fallback_reason"] == "none"
    assert ev["scientific_N_batch"] == 12
    assert ev["workload_shape"] == (12, 480, 270)


def test_record_cpu_fallback_never_claims_gpu():
    s = _mk_stacker()
    _record(s, attempted=True, executed="cpu", gpu_memory_mode="fallback",
            fallback_reason=GPU_EXEC_REASON_GPU_KERNEL)
    (ev,) = s._gpu_execution_events
    assert ev["executed"] == "cpu"
    assert ev["fallback"] is True
    assert ev["fallback_reason"] == "gpu_kernel_failure"


def test_record_tiled_carries_tile_shape():
    s = _mk_stacker()
    _record(s, executed="gpu", gpu_memory_mode="tiled", planner_mode="tiled",
            tile_shape=(720,), n_tiles=2)
    (ev,) = s._gpu_execution_events
    assert ev["gpu_memory_mode"] == "tiled"
    assert ev["tile_shape"] == (720,)
    assert ev["n_tiles"] == 2


# ---------------------------------------------------------------------------
# aggregate GPU_EXECUTION_SUMMARY
# ---------------------------------------------------------------------------


def test_summary_aggregates_full_tiled_fallback():
    s = _mk_stacker()
    _record(s, executed="gpu", gpu_memory_mode="full")
    _record(s, executed="gpu", gpu_memory_mode="tiled", tile_shape=(720,))
    _record(s, executed="cpu", gpu_memory_mode="fallback",
            fallback_reason=GPU_EXEC_REASON_VRAM_NO_VALID_TILE)
    _record(s, executed="cpu", gpu_memory_mode="fallback",
            fallback_reason=GPU_EXEC_REASON_VRAM_NO_VALID_TILE)
    tok = s._gpu_execution_summary_tokens()
    assert tok is not None
    assert tok["reductions"] == 4
    assert tok["gpu_full"] == 1
    assert tok["gpu_tiled"] == 1
    assert tok["cpu_fallback"] == 2
    assert tok["max_N_batch"] == 12
    # fallback reasons are dedup'd and ordered
    assert tok["fallback_reasons"] == ["vram_no_valid_tile"]


def test_summary_max_n_batch_and_peak_vram():
    s = _mk_stacker()
    _record(s, scientific_N_batch=4, estimated_peak_vram_bytes=100)
    _record(s, scientific_N_batch=50, estimated_peak_vram_bytes=700)
    _record(s, scientific_N_batch=12, estimated_peak_vram_bytes=300)
    tok = s._gpu_execution_summary_tokens()
    assert tok["max_N_batch"] == 50
    assert tok["peak_vram"] == 700


def test_summary_none_when_no_dispatch():
    s = _mk_stacker()
    assert s._gpu_execution_summary_tokens() is None


def test_emit_summary_once_per_run(monkeypatch):
    s = _mk_stacker()
    emitted = []

    def _capture(prefix, tokens):
        emitted.append((prefix, tokens))

    monkeypatch.setattr(s, "_emit_provenance_block", _capture)
    _record(s, executed="gpu", gpu_memory_mode="full")
    s._emit_gpu_execution_summary()
    s._emit_gpu_execution_summary()  # second call is a no-op
    assert len(emitted) == 1
    assert emitted[0][0] == "GPU_EXECUTION_SUMMARY"
    assert emitted[0][1]["gpu_full"] == 1


def test_emit_summary_emits_nothing_when_empty(monkeypatch):
    s = _mk_stacker()
    emitted = []
    monkeypatch.setattr(s, "_emit_provenance_block",
                        lambda prefix, tokens: emitted.append((prefix, tokens)))
    s._emit_gpu_execution_summary()
    assert emitted == []
    assert s._gpu_execution_summary_emitted is True
