"""COV-06C runtime activation, support lifecycle and explicit outcomes."""

from pathlib import Path
import types

import numpy as np
from astropy.io import fits

from seestar.queuep import queue_manager as qm


def _classic_finalizer(output_dir: Path, *, requested: bool, support: bool):
    """Build the real Classic final-save seam with close-on-finalize support."""
    obj = types.SimpleNamespace()
    obj.output_folder = str(output_dir)
    obj.output_filename = "cov06c.fit"
    obj.finalization_mode = qm.FINALIZATION_MODE_CLASSIC_SUMW
    obj.save_final_as_float32 = True
    obj.preserve_linear_output = True
    obj.drizzle_wht_threshold = 0.0
    obj.drizzle_output_wcs = None
    obj.drizzle_active_session = False
    obj.is_mosaic_run = False
    obj.reproject_between_batches = False
    obj.reproject_coadd_final = False
    obj.drizzle_mode = "Final"
    obj.drizzle_fillval = "0.0"
    obj.batch_size = 2
    obj.images_in_cumulative_stack = 2
    obj.total_exposure_seconds = 20.0
    obj.current_stack_header = fits.Header()
    obj.processing_error = None
    obj.apply_feathering = False
    obj.apply_low_wht_mask = False
    obj.apply_coverage_render = requested
    obj.coverage_render_n_ref = 32.0
    obj._support_state_available = support
    obj._support_unavailable_reason = None if support else "support not initialized"
    obj._drizzle_support_available = False
    obj._drizzle_support_unavailable_reason = "not a Drizzle run"
    obj.logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    obj.update_progress_messages = []
    obj.update_progress = lambda *args, **kwargs: obj.update_progress_messages.append(
        (args, kwargs)
    )
    obj.lifecycle_events = []
    obj._emit_lifecycle = lambda event, **fields: obj.lifecycle_events.append(
        (event, fields)
    )

    y, x = np.mgrid[:48, :48]
    signal = (500.0 + 0.2 * x + 0.1 * y).astype(np.float32)
    obj.cumulative_sum_memmap = np.repeat((signal * 2.0)[..., None], 3, axis=2)
    obj.cumulative_wht_memmap = np.full((48, 48), 2.0, dtype=np.float32)
    if support:
        # SUP_W1=2, SUP_W2=2 -> real N_eff derivation gives 2 everywhere.
        obj.coverage_sup_w1_memmap = np.full((48, 48), 2.0, dtype=np.float64)
        obj.coverage_sup_w2_memmap = np.full((48, 48), 2.0, dtype=np.float64)
    else:
        obj.coverage_sup_w1_memmap = None
        obj.coverage_sup_w2_memmap = None
    obj._derive_neff_support_for_render = types.MethodType(
        qm.SeestarQueuedStacker._derive_neff_support_for_render, obj
    )

    # Reproduce the production lifecycle bug seam: Classic copies SCI/WHT and
    # then closes every memmap, including support.  COV-06C must have derived
    # N_eff before this callback runs.
    def close_memmaps():
        obj.cumulative_sum_memmap = None
        obj.cumulative_wht_memmap = None
        obj.coverage_sup_w1_memmap = None
        obj.coverage_sup_w2_memmap = None

    obj._close_memmaps = close_memmaps
    return obj


def _run_finalizer(obj):
    Path(obj.output_folder).mkdir()
    qm.SeestarQueuedStacker._save_final_stack(
        obj,
        preserve_linear_output=True,
        finalization_mode=qm.FINALIZATION_MODE_CLASSIC_SUMW,
    )


def _event(obj, name):
    return [fields for event, fields in obj.lifecycle_events if event == name]


def test_classic_requested_uses_real_support_derivation_and_applies_once(
    tmp_path, monkeypatch
):
    obj = _classic_finalizer(tmp_path / "applied", requested=True, support=True)
    original_renderer = qm.coverage_aware_render
    calls = []

    def counting_renderer(sci, neff, **kwargs):
        calls.append(neff.copy())
        return original_renderer(sci, neff, **kwargs)

    monkeypatch.setattr(qm, "coverage_aware_render", counting_renderer)
    _run_finalizer(obj)

    assert len(calls) == 1
    assert np.allclose(calls[0], 2.0, rtol=0.0, atol=2e-7)
    assert obj.coverage_render_applied_in_session is True
    assert obj.coverage_render_status == "APPLIED"
    assert Path(obj.final_stacked_path).is_file()
    support_event = _event(obj, "COVERAGE_RENDER_SUPPORT")
    assert support_event == [
        {
            "classic_available": True,
            "sup_w1_present": True,
            "sup_w2_present": True,
            "drizzle_available": False,
            "classic_reason": None,
            "drizzle_reason": "not a Drizzle run",
            "reason": None,
        }
    ]
    result = _event(obj, "COVERAGE_RENDER_RESULT")
    assert len(result) == 1
    assert result[0]["requested"] is True
    assert result[0]["status"] == "APPLIED"
    assert result[0]["positive_fraction"] == 1.0


def test_requested_without_support_skips_explicitly_and_preserves_fits(
    tmp_path, monkeypatch
):
    obj = _classic_finalizer(tmp_path / "skipped", requested=True, support=False)
    monkeypatch.setattr(
        qm,
        "coverage_aware_render",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not render")),
    )

    _run_finalizer(obj)

    assert obj.coverage_render_applied_in_session is False
    assert obj.coverage_render_status == "SKIPPED_NO_SUPPORT"
    assert Path(obj.final_stacked_path).is_file()
    result = _event(obj, "COVERAGE_RENDER_RESULT")
    assert result == [
        {
            "requested": True,
            "status": "SKIPPED_NO_SUPPORT",
            "reason": "support not initialized",
        }
    ]
    assert any(
        "SKIPPED_NO_SUPPORT" in str(args[0])
        for args, _kwargs in obj.update_progress_messages
        if args
    )


def test_render_off_reports_not_requested_without_support_work(tmp_path, monkeypatch):
    obj = _classic_finalizer(tmp_path / "off", requested=False, support=True)
    derive_calls = []
    obj._derive_neff_support_for_render = lambda: derive_calls.append(True)
    monkeypatch.setattr(
        qm,
        "coverage_aware_render",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not render")),
    )

    _run_finalizer(obj)

    assert derive_calls == []
    assert obj.coverage_render_status == "NOT_REQUESTED"
    assert obj.coverage_render_applied_in_session is False
    assert Path(obj.final_stacked_path).is_file()
    assert _event(obj, "COVERAGE_RENDER_SUPPORT") == []
    assert _event(obj, "COVERAGE_RENDER_RESULT") == [
        {"requested": False, "status": "NOT_REQUESTED"}
    ]


def test_canonical_runtime_config_persists_cosmetic_request():
    obj = types.SimpleNamespace(
        _run_config_canonical=None,
        apply_coverage_render=True,
        current_folder="/input",
        output_folder="/output",
        output_filename="stack.fit",
    )
    obj._canonical_product_version = lambda: "test"

    cfg = qm.SeestarQueuedStacker._canonical_run_config(obj)

    assert cfg.execution["apply_coverage_render"] is True
