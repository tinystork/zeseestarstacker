"""D1 tests: execution-aware Drizzle provenance semantics (D1.3 / D1.4 / D3.6).

Proves, against the REAL engine emission gates used at run start:

* ``RUN_EFFECTIVE`` reports ``stacking_mode_effective=drizzle_direct_accumulation``
  plus ``stacking_mode_substitution_reason=classic_reducer_not_used_by_drizzle_path``
  when the run is a Drizzle direct accumulation (``drizzle_active_session`` /
  ``FINALIZATION_MODE_DRIZZLE``), even when the requested stacking mode was a
  Classic reducer (e.g. winsorized-sigma-clip),
* the same Classic run reports the real reducer (``winsorized_sigma_clip``)
  with NO substitution reason,
* ``GPU_DECISION`` reports ``execution=not_executed`` with
  ``fallback_reason=reducer_bypassed_by_drizzle_path`` on a Drizzle run even
  when GPU was requested AND cupy-ready -- never ``used`` (D1.4),
* the cfg evidence helpers mirror both semantics,
* D3.6 output-serialization requested tokens reach ``RUN_REQUEST`` and the
  persisted ``run_config.cfg`` (real accepted-run lifecycle), and the
  finalization-time effective tokens surface through the cfg-value helper.

The emission gates are exercised with the exact instance flags the engine
resolves before calling them in ``start_processing`` (``drizzle_active_session``
is fixed before ``RUN_EFFECTIVE`` is emitted at the accepted-run seam).
"""

from __future__ import annotations

import os
import threading

import numpy as np
import pytest
from astropy.io import fits

import seestar.queuep.queue_manager as queue_manager_module
import seestar.run_contract as run_contract
from seestar.core.gpu import GpuCapabilities
from seestar.queuep.queue_manager import SeestarQueuedStacker

# Reuse the bounded real-lifecycle helpers of the M2 diagnostics suite.
import tests.test_run_provenance_diagnostics as m2

_write_source = m2._write_source
_run_lifecycle = m2._run_lifecycle


def _ready_caps():
    return GpuCapabilities(
        gpu_detected=True,
        cuda_runtime_ready=True,
        cupy_ready=True,
        opencv_cuda_ready=False,
        backend_ready=True,
        device_name="Fake GPU",
        device_vram_mb=2048,
        compute_capability="6.1",
        failure_reason=None,
        state="ready",
    )


def _unavailable_caps():
    return GpuCapabilities(
        gpu_detected=False,
        cuda_runtime_ready=False,
        cupy_ready=False,
        opencv_cuda_ready=False,
        backend_ready=False,
        device_name=None,
        device_vram_mb=None,
        compute_capability=None,
        failure_reason="cupy unavailable",
        state="cuda_no_backend",
    )


def _emission_stack(
    *,
    use_drizzle,
    stacking_mode="winsorized-sigma-clip",
    request_gpu=False,
    caps=None,
):
    """Real stacker configured with the exact flags ``start_processing`` sets
    before emitting RUN_EFFECTIVE / GPU_DECISION (no worker, no FITS run)."""
    s = SeestarQueuedStacker(batch_size=1, autotune=False)
    s.drizzle_active_session = bool(use_drizzle)
    s.finalization_mode = (
        queue_manager_module.FINALIZATION_MODE_DRIZZLE
        if use_drizzle
        else queue_manager_module.FINALIZATION_MODE_CLASSIC_SUMW
    )
    s.stacking_mode = stacking_mode
    s.request_gpu = bool(request_gpu)
    s._gpu_capabilities = caps if caps is not None else _unavailable_caps()
    s._run_prov_requested = {"stacking_mode_requested": stacking_mode}
    events = []
    s.update_progress = lambda message, progress=None, level=None: events.append(
        str(message)
    )
    return s, events


# ---------------------------------------------------------------------------
# D1.3: execution-aware stacking_mode_effective
# ---------------------------------------------------------------------------


def test_drizzle_effective_reports_direct_accumulation_with_substitution_reason():
    s, events = _emission_stack(use_drizzle=True)
    eff = s._run_provenance_effective()
    assert eff["stacking_mode_effective"] == "drizzle_direct_accumulation"
    assert (
        eff["stacking_mode_substitution_reason"]
        == "classic_reducer_not_used_by_drizzle_path"
    )
    # The durable RUN_EFFECTIVE block carries both tokens.
    s._emit_run_provenance_effective()
    eff_lines = [e for e in events if e.startswith("RUN_EFFECTIVE ")]
    assert len(eff_lines) == 1, events
    line = eff_lines[0]
    assert "stacking_mode_effective=drizzle_direct_accumulation" in line
    assert (
        "stacking_mode_substitution_reason=classic_reducer_not_used_by_drizzle_path"
        in line
    )
    # The requested Classic reducer was NOT claimed to have executed anywhere.
    assert "stacking_mode_effective=winsorized_sigma_clip" not in line


def test_classic_effective_reports_real_reducer_without_substitution():
    s, events = _emission_stack(use_drizzle=False)
    eff = s._run_provenance_effective()
    assert eff["stacking_mode_effective"] == "winsorized_sigma_clip"
    assert "stacking_mode_substitution_reason" not in eff
    s._emit_run_provenance_effective()
    eff_lines = [e for e in events if e.startswith("RUN_EFFECTIVE ")]
    assert len(eff_lines) == 1
    assert "stacking_mode_effective=winsorized_sigma_clip" in eff_lines[0]


def test_cfg_values_drizzle_safe_mirror_direct_accumulation():
    s, _events = _emission_stack(
        use_drizzle=True, request_gpu=True, caps=_ready_caps()
    )
    out = s.run_provenance_cfg_values_drizzle_safe()
    assert out["stacking_mode_effective"] == "drizzle_direct_accumulation"
    assert (
        out["stacking_mode_substitution_reason"]
        == "classic_reducer_not_used_by_drizzle_path"
    )
    assert out["gpu_execution"] == "not_executed"
    assert out["gpu_fallback_reason"] == "reducer_bypassed_by_drizzle_path"


# ---------------------------------------------------------------------------
# D1.4: GPU_DECISION never claims a Classic GPU reducer ran under Drizzle
# ---------------------------------------------------------------------------


def test_gpu_decision_not_executed_under_drizzle_even_when_cupy_ready():
    s, events = _emission_stack(
        use_drizzle=True, request_gpu=True, caps=_ready_caps()
    )
    s._emit_run_provenance_effective()
    decisions = [e for e in events if e.startswith("GPU_DECISION ")]
    assert len(decisions) == 1, events
    line = decisions[0]
    assert "requested=true" in line
    # operation names the REQUESTED Classic family (never executed).
    assert "operation=stacking_reduction:winsorized_sigma_clip" in line
    assert "effective_backend=cupy" in line
    assert "execution=not_executed" in line
    assert "fallback_reason=reducer_bypassed_by_drizzle_path" in line
    assert "execution=used" not in line


def test_gpu_decision_eligible_for_classic_winsorized():
    s, events = _emission_stack(
        use_drizzle=False, request_gpu=True, caps=_ready_caps()
    )
    s._emit_run_provenance_effective()
    decisions = [e for e in events if e.startswith("GPU_DECISION ")]
    assert len(decisions) == 1, events
    line = decisions[0]
    assert "operation=stacking_reduction:winsorized_sigma_clip" in line
    assert "execution=eligible" in line
    assert "fallback_reason=none" in line


def test_gpu_decision_not_eligible_for_mean_unchanged():
    # Classic run with a mode that has no GPU reducer keeps not_eligible.
    s, events = _emission_stack(
        use_drizzle=False,
        stacking_mode="mean",
        request_gpu=True,
        caps=_ready_caps(),
    )
    s._emit_run_provenance_effective()
    decisions = [e for e in events if e.startswith("GPU_DECISION ")]
    assert len(decisions) == 1, events
    line = decisions[0]
    assert "operation=stacking_reduction:mean" in line
    assert "execution=not_eligible" in line
    assert "fallback_reason=no_gpu_reducer_for_mode:mean" in line


# ---------------------------------------------------------------------------
# D3.6: requested tokens reach RUN_REQUEST + run_config.cfg (real lifecycle)
# ---------------------------------------------------------------------------


def test_run_request_and_cfg_capture_output_serialization_requested(
    monkeypatch, tmp_path
):
    stacker, events, output_dir = _run_lifecycle(
        monkeypatch,
        tmp_path,
        stacking_mode="winsorized-sigma-clip",
        save_as_float32=True,
        preserve_linear_output=True,
    )
    req = [e for e in events if e.startswith("RUN_REQUEST ")]
    assert len(req) == 1, events
    assert "save_as_float32_requested=true" in req[0]
    assert "preserve_linear_output_requested=true" in req[0]

    cfg_path = output_dir / "run_config.cfg"
    assert cfg_path.is_file()
    report = run_contract.read_cfg(str(cfg_path))
    ex = report.config.execution
    assert ex.get("save_as_float32_requested") is True
    assert ex.get("preserve_linear_output_requested") is True
    # The run is Classic here, so the effective stacking mode stays the real
    # reducer (no drizzle substitution) -- regression guard on the cfg path.
    assert report.config.scientific.get("stacking_mode_effective") == (
        "winsorized_sigma_clip"
    )


def test_cfg_values_surface_requested_and_recorded_effective_tokens():
    s, _events = _emission_stack(use_drizzle=False)
    s._run_prov_requested = {
        "stacking_mode_requested": "winsorized-sigma-clip",
        "save_as_float32_requested": True,
        "preserve_linear_output_requested": True,
        "batch_size_requested": "auto",
    }
    s._serialization_effective = {
        "save_as_float32_effective": True,
        "preserve_linear_output_effective": True,
        "output_dtype_effective": "float32",
        "scientific_domain_before_serialization": "signed_float32",
        "scientific_domain_written": "signed_float32",
    }
    out = s.run_provenance_cfg_values()
    assert out["save_as_float32_requested"] is True
    assert out["preserve_linear_output_requested"] is True
    assert out["save_as_float32_effective"] is True
    assert out["preserve_linear_output_effective"] is True
    assert out["output_dtype_effective"] == "float32"
    assert out["scientific_domain_before_serialization"] == "signed_float32"
    assert out["scientific_domain_written"] == "signed_float32"

    # Before finalization the effective tokens are absent (deterministic
    # mid-run cfg writes never change digest).
    s._serialization_effective = None
    out2 = s.run_provenance_cfg_values()
    assert out2["save_as_float32_requested"] is True
    assert "save_as_float32_effective" not in out2
    assert "output_dtype_effective" not in out2
