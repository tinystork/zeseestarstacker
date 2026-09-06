"""M2 tests: A2 requested/effective contract + A3 durable run diagnostics +
A4 reproducible run_config.cfg (real engine path).

Proves:

* the new A2 canonical fields (batch_size_requested / batch_size_effective,
  stacking_mode_effective, normalize/weighting/quality effective aliases,
  GPU decision diagnostics, reference provenance) round-trip through
  ``run_contract`` write/read,
* ``RUN_REQUEST`` / ``RUN_EFFECTIVE`` / ``GPU_DECISION`` blocks are emitted by
  the REAL engine at an accepted run and carry the requested + effective
  values,
* ``GPU_DECISION`` carries an explicit ``fallback_reason`` when GPU was
  requested but not used, and ``not_eligible`` with the Track-B reason for
  winsorized-sigma (not yet GPU-qualified),
* ``run_config.cfg`` records ``batch_size_requested=auto`` next to a concrete
  ``batch_size_effective=<N>`` (no unexplained -1 sentinel).

The engine is exercised through the same bounded real-lifecycle harness used
by the M1 E2E tests (real ``SeestarQueuedStacker`` + real ``start_processing``
with a stub worker; no real stacking).
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


class _NoopExecutor:
    def __init__(self, max_workers=1, **_kwargs):
        self._max_workers = max_workers

    def shutdown(self, *_args, **_kwargs):
        return None


def _write_source(path, value=0):
    yy, xx = np.indices((32, 32))
    data = ((xx + yy + value) % 17).astype(np.uint16) * 100
    header = fits.Header()
    header["RA"] = 275.0
    header["DEC"] = -13.7
    header["BAYERPAT"] = "GRBG"
    header["EXPTIME"] = 10.0
    fits.PrimaryHDU(data=data, header=header).writeto(path)


def _run_lifecycle(
    monkeypatch,
    tmp_path,
    *,
    batch_size=-1,
    stacking_mode="kappa-sigma",
    request_gpu=False,
    gpu_caps=None,
):
    """Run the real start_processing lifecycle to the worker seam.

    Mirrors the M1 E2E harness: real engine, real reference materialisation,
    no-op process pool, stub worker that stops immediately.  Returns
    ``(stacker, progress_events, output_dir)``.
    """
    monkeypatch.setattr(queue_manager_module, "ProcessPoolExecutor", _NoopExecutor)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    source = input_dir / "central.fit"
    _write_source(source)

    stacker = SeestarQueuedStacker(batch_size=1, autotune=False)
    if gpu_caps is not None:
        stacker._gpu_capabilities = gpu_caps
    stacker.request_gpu = bool(request_gpu)
    events = []
    stacker.update_progress = lambda message, progress=None, level=None: events.append(
        str(message)
    )
    worker_done = threading.Event()

    def worker():
        stacker.processing_active = False
        worker_done.set()

    stacker._worker = worker
    started = stacker.start_processing(
        str(input_dir),
        str(output_dir),
        reference_path_ui=str(source),
        stacking_mode=stacking_mode,
        batch_size=batch_size,
        correct_hot_pixels=False,
        perform_cleanup=False,
        move_stacked=False,
        reproject_between_batches=False,
        reproject_coadd_final=False,
    )
    if stacker.processing_thread is not None:
        stacker.processing_thread.join(timeout=10)
    stacker.quality_executor.shutdown()
    assert started is True
    return stacker, events, output_dir


# ---------------------------------------------------------------------------
# 1. A2 fields round-trip through run_contract read/write
# ---------------------------------------------------------------------------


def test_a2_provenance_fields_round_trip(tmp_path):
    cfg = run_contract.RunConfig.from_sections(
        product_version="8.3.0",
        scientific={
            "stacking_mode_effective": "winsorized_sigma_clip",
            "normalize_method_effective": "linear_fit",
            "weighting_method_effective": "noise_variance",
            "use_quality_weighting_effective": True,
            "batch_size_requested": "auto",
            "batch_size_effective": 7,
        },
        execution={
            "gpu_capability_state": "no_gpu",
            "gpu_device_name": None,
            "gpu_requested_backend": "cupy",
            "gpu_effective_backend": "cpu",
            "gpu_operation": "stacking_reduction:kappa_sigma",
            "gpu_execution": "fallback",
            "gpu_fallback_reason": "no_gpu",
            "reference_policy_requested": "user",
            "reference_origin_effective": "USER",
            "reference_path_effective": "/data/ref.fit",
        },
    )
    target = tmp_path / "run_config.cfg"
    run_contract.write_cfg(cfg, str(target))
    report = run_contract.read_cfg(str(target))
    reloaded = report.config
    assert report.unknown_keys == ()
    assert reloaded.scientific["batch_size_requested"] == "auto"
    assert reloaded.scientific["batch_size_effective"] == 7
    assert reloaded.scientific["stacking_mode_effective"] == "winsorized_sigma_clip"
    assert reloaded.execution["gpu_effective_backend"] == "cpu"
    assert reloaded.execution["gpu_execution"] == "fallback"
    assert reloaded.execution["gpu_fallback_reason"] == "no_gpu"
    assert reloaded.execution["reference_origin_effective"] == "USER"
    assert reloaded.to_canonical_bytes() == cfg.to_canonical_bytes()


def test_a2_provenance_fields_do_not_leak_into_classic_fingerprint():
    base = run_contract.RunConfig.from_sections(
        product_version="8.3.0",
        scientific={"stacking_mode": "kappa-sigma", "kappa": 2.5},
    )
    with_prov = run_contract.RunConfig.from_sections(
        product_version="8.3.0",
        scientific={
            "stacking_mode": "kappa-sigma",
            "kappa": 2.5,
            "stacking_mode_effective": "kappa_sigma",
            "batch_size_requested": "auto",
            "batch_size_effective": 12,
        },
        execution={"gpu_effective_backend": "cpu", "gpu_execution": "not_requested"},
    )
    # The classic fingerprint is byte-identical: provenance fields are not
    # part of the scientific fingerprint domain.
    assert base.classic_fingerprint() == with_prov.classic_fingerprint()
    assert base.full_digest() != with_prov.full_digest()


# ---------------------------------------------------------------------------
# 2/3. RUN_REQUEST / RUN_EFFECTIVE / GPU_DECISION at an accepted run
# ---------------------------------------------------------------------------


def test_accepted_run_emits_run_request_and_run_effective(monkeypatch, tmp_path):
    stacker, events, _out = _run_lifecycle(
        monkeypatch, tmp_path, batch_size=-1, stacking_mode="winsorized-sigma-clip"
    )
    req = [e for e in events if e.startswith("RUN_REQUEST ")]
    eff = [e for e in events if e.startswith("RUN_EFFECTIVE ")]
    assert len(req) == 1, events
    assert len(eff) == 1, events
    req_line = req[0]
    eff_line = eff[0]
    # Requested tokens present.
    assert "batch_size_requested=auto" in req_line
    assert "gpu_requested=false" in req_line
    assert "stacking_mode_requested=winsorized-sigma-clip" in req_line
    assert "normalization_requested=none" in req_line
    # Effective tokens present.
    assert "batch_size_effective=" in eff_line
    assert "gpu_backend_effective=" in eff_line
    assert "stacking_mode_effective=winsorized_sigma_clip" in eff_line
    assert "reference_effective=origin=USER" in eff_line
    # No GPU_DECISION when GPU was not requested.
    assert not any(e.startswith("GPU_DECISION") for e in events)


def test_gpu_decision_fallback_reason_when_requested_but_unavailable(
    monkeypatch, tmp_path
):
    caps = GpuCapabilities(
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
    stacker, events, _out = _run_lifecycle(
        monkeypatch, tmp_path, request_gpu=True, gpu_caps=caps
    )
    decisions = [e for e in events if e.startswith("GPU_DECISION ")]
    assert len(decisions) == 1, events
    line = decisions[0]
    assert "requested=true" in line
    assert "operation=stacking_reduction:kappa_sigma" in line
    assert "effective_backend=cpu" in line
    assert "execution=fallback" in line
    assert "fallback_reason=cupy unavailable" in line


def test_gpu_decision_not_eligible_for_winsorized(monkeypatch, tmp_path):
    # GPU available (cupy ready) but winsorized-sigma has no GPU reducer yet:
    # Track B qualification pending -> not_eligible with explicit reason.
    caps = GpuCapabilities(
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
    stacker, events, _out = _run_lifecycle(
        monkeypatch,
        tmp_path,
        request_gpu=True,
        gpu_caps=caps,
        stacking_mode="winsorized-sigma-clip",
    )
    decisions = [e for e in events if e.startswith("GPU_DECISION ")]
    assert len(decisions) == 1, events
    line = decisions[0]
    assert "requested=true" in line
    assert "effective_backend=cupy" in line
    assert "execution=not_eligible" in line
    assert (
        "fallback_reason=winsorized_gpu_qualification_pending_track_b" in line
    )


# ---------------------------------------------------------------------------
# 4. run_config.cfg: requested vs effective batch size (real engine)
# ---------------------------------------------------------------------------


def test_run_config_cfg_records_requested_auto_and_concrete_effective(
    monkeypatch, tmp_path
):
    stacker, _events, output_dir = _run_lifecycle(monkeypatch, tmp_path, batch_size=-1)
    cfg_path = output_dir / "run_config.cfg"
    assert cfg_path.is_file()
    report = run_contract.read_cfg(str(cfg_path))
    sci = report.config.scientific
    # The engine auto-resolved a concrete batch size and recorded it next to
    # the requested semantics ("auto"), never an unexplained -1 sentinel.
    assert sci.get("batch_size_requested") == "auto"
    effective = sci.get("batch_size_effective")
    assert isinstance(effective, int) and effective >= 1
    assert sci.get("batch_size") == effective
    # The effective stacking kernel is also recorded on disk.
    assert sci.get("stacking_mode_effective") == "kappa_sigma"


def test_run_config_cfg_requested_all_ram_stays_explicit(monkeypatch, tmp_path):
    # batch_size=0 is the special all-in-RAM single batch mode; it must be
    # recorded as the explicit 'all_ram' semantics, not as a bare 0 sentinel.
    stacker, events, output_dir = _run_lifecycle(
        monkeypatch, tmp_path, batch_size=0, stacking_mode="winsorized-sigma-clip"
    )
    req = [e for e in events if e.startswith("RUN_REQUEST ")]
    assert req and "batch_size_requested=all_ram" in req[0]
    cfg_path = output_dir / "run_config.cfg"
    if cfg_path.is_file():
        report = run_contract.read_cfg(str(cfg_path))
        assert report.config.scientific.get("batch_size_requested") == "all_ram"
