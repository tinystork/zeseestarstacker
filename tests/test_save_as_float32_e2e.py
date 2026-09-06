"""D3.3 tests: ``save_final_as_float32`` end-to-end propagation.

Pins the FULL production chain for the output-dtype option, independently for
the CLASSIC and the DRIZZLE route, mirroring the Lanczos-transmission e2e
pattern (real wiring, no display, no long engine runs):

* Qt settings widget -> ``QtSettingsState.save_final_as_float32``
  (``collect_settings_state``),
* state -> ``RunRequest.backend_kwargs`` via the canonical builder
  (``build_backend_kwargs`` -> key ``save_as_float32``),
* backend kwargs -> ``start_processing`` surface via ``split_backend_kwargs``
  (the exact seam ``SeestarQueuedStackerBackend`` uses),
* the real engine argument-receipt seam: the instance attribute
  ``save_final_as_float32`` is set from the request boolean exactly as
  ``start_processing`` does, and the real ``_capture_run_provenance_requested``
  records the ``save_as_float32_requested`` token,
* the real finalizer + FITS writer: ``_save_final_stack`` runs on a real
  ``SeestarQueuedStacker`` skeleton in ``FINALIZATION_MODE_CLASSIC_SUMW``
  (Classic SUM/W) and ``FINALIZATION_MODE_DRIZZLE`` (single accumulator)
  respectively; the reopened FITS dtype matches the requested float32/uint16,
  the saved content is faithful, and the provenance tokens agree:
  ``save_as_float32_requested == save_as_float32_effective == requested``.

No stack of real FITS is executed: the engine data are deterministic
synthetic arrays fed to the real production finalizer boundaries.
"""

from __future__ import annotations

import logging
import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from astropy.io import fits
from PySide6.QtWidgets import QApplication, QCheckBox

from seestar.core.drizzle_core import DrizzleAccumulator
from seestar.gui_qt import MainWindow, create_application

# Reuse the Dummy harness of test_save_final_stack (real finalizer binding).
import tests.test_save_final_stack as tsf


@pytest.fixture(scope="session")
def qapp():
    app = create_application([])
    assert app is QApplication.instance()
    return app


@pytest.fixture()
def window(qapp):
    win = MainWindow()
    yield win
    win.shutdown()


# ---------------------------------------------------------------------------
# 1. Qt UI -> settings state -> RunRequest -> start_processing surface
# ---------------------------------------------------------------------------


def _float32_widget(window):
    w = window._settings_widgets["save_final_as_float32"]
    assert isinstance(w, QCheckBox)
    return w


def _assert_request_carries_float32(window, requested: bool):
    state = window.collect_settings_state()
    assert state.save_final_as_float32 is requested

    request = window.build_run_request()
    assert request.backend_kwargs["save_as_float32"] is requested

    # The exact seam the SeestarQueuedStackerBackend adapter uses: the option
    # is a start_processing argument (never a seam-only snapshot field).
    from seestar.gui_qt.run_bridge import split_backend_kwargs

    start_kwargs, seam_kwargs = split_backend_kwargs(request.backend_kwargs)
    assert "save_as_float32" not in seam_kwargs
    assert start_kwargs["save_as_float32"] is requested
    return state


def test_ui_to_request_classic_route_float32_and_uint16(window):
    """Classic route (drizzle off): the checkbox reaches the request."""
    assert window.drizzle_check.isChecked() is False
    widget = _float32_widget(window)
    assert widget.isChecked() is False  # default: uint16

    widget.setChecked(True)
    _assert_request_carries_float32(window, True)

    widget.setChecked(False)
    _assert_request_carries_float32(window, False)


def test_ui_to_request_drizzle_route_float32_and_uint16(window):
    """Drizzle route (independent): same propagation with drizzle enabled."""
    window.drizzle_check.setChecked(True)
    window.drizzle_kernel_combo.setCurrentText("lanczos3")

    widget = _float32_widget(window)
    widget.setChecked(True)
    state = _assert_request_carries_float32(window, True)
    assert state.use_drizzle is True
    assert state.drizzle_kernel == "lanczos3"

    widget.setChecked(False)
    _assert_request_carries_float32(window, False)


# ---------------------------------------------------------------------------
# 2. Real engine seam: attribute ingestion + requested-token capture
# ---------------------------------------------------------------------------


def _engine_skeleton(tmp_path):
    """A real ``SeestarQueuedStacker`` skeleton with the finalizer-surface
    attributes the production save path requires."""
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    obj = tsf._make_obj(tmp_path, False)
    obj.logger = logging.getLogger("zsss.d33")
    obj.batch_size = 0
    obj.aligned_temp_paths = []
    obj.preserve_linear_output = False
    obj.processing_error = None
    obj.save_final_as_float32 = False
    # Bind the REAL provenance capture seam onto the harness (the same
    # pattern the suite uses for ``_validate_drizzle_science``): the token
    # bookkeeping and emission run production code on the test object.
    for name in (
        "_capture_run_provenance_requested",
        "_emit_provenance_block",
        "_provenance_line",
    ):
        setattr(
            obj,
            name,
            types.MethodType(getattr(SeestarQueuedStacker, name), obj),
        )
    obj._validate_drizzle_science = types.MethodType(
        SeestarQueuedStacker._validate_drizzle_science, obj
    )
    return obj


def test_engine_receipt_and_requested_token_agree(window):
    """The exact start_processing receipt: request bool -> instance attr and
    the ``save_as_float32_requested`` provenance token (real capture seam)."""
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    obj = _engine_skeleton("/tmp")
    # Production receipt (start_processing lines): the argument boolean is
    # assigned to the instance attribute, then the accepted-run provenance
    # snapshot records the same boolean as the requested token.
    for requested in (True, False):
        obj.save_final_as_float32 = bool(requested)
        obj._capture_run_provenance_requested(
            stacking_mode_requested="kappa-sigma",
            save_as_float32_requested=bool(requested),
            preserve_linear_output_requested=False,
            drizzle_requested=False,
        )
        assert obj.save_final_as_float32 is requested
        assert (
            obj._run_prov_requested["save_as_float32_requested"] is requested
        )
        # The start_processing signature really accepts the option.
        import inspect

        sig = inspect.signature(
            SeestarQueuedStacker.start_processing
        )
        assert "save_as_float32" in sig.parameters


# ---------------------------------------------------------------------------
# 3. Real finalizer + FITS writer: CLASSIC and DRIZZLE, float32 and uint16
# ---------------------------------------------------------------------------


def _expected_uint16(raw):
    """Reproduce the production uint16 conversion (full-scale to 65535)."""
    raw = np.clip(raw, 0.0, None)
    mx = float(np.max(raw))
    assert np.isfinite(mx) and mx > 0
    effective_max = max(mx, 1.0 / 65535.0)
    return (raw * (65535.0 / effective_max)).astype(np.uint16)


def _classic_save(obj, final_sum, final_wht, suffix):
    """Run the REAL ``_save_final_stack`` CLASSIC SUM/W finalization."""
    from seestar.queuep.queue_manager import (
        FINALIZATION_MODE_CLASSIC_SUMW,
        SeestarQueuedStacker,
    )

    obj.finalization_mode = FINALIZATION_MODE_CLASSIC_SUMW
    obj.drizzle_active_session = False
    obj.cumulative_sum_memmap = final_sum
    obj.cumulative_wht_memmap = final_wht
    obj.images_in_cumulative_stack = int(np.max(final_wht))
    obj.total_exposure_seconds = float(np.max(final_wht))
    SeestarQueuedStacker._save_final_stack(obj, output_filename_suffix=suffix)
    return fits.getdata(obj.final_stacked_path)


@pytest.mark.parametrize("requested_float32", [True, False])
def test_classic_route_e2e_dtype_and_tokens(tmp_path, requested_float32):
    """Classic (SUM/W) route: UI-independent engine-level end-to-end."""
    from seestar.queuep.queue_manager import SeestarQueuedStacker

    # Deterministic classic science: per-channel mean M (HWC), N contributors.
    h, w, n = 5, 9, 3
    base = np.zeros((h, w), dtype=np.float64)
    base[1:-1, 1:-1] = np.array(
        [
            [0.25, 0.5, 0.75, 0.125, 0.625, 0.375, 0.875],
            [0.5, 0.75, 0.25, 0.625, 0.125, 0.875, 0.375],
            [0.75, 0.25, 0.5, 0.375, 0.875, 0.625, 0.125],
        ],
        dtype=np.float64,
    )
    mean_rgb = np.stack([base, base * 2.0, base * 0.5], axis=-1)  # HWC
    wht = np.full((h, w), float(n), dtype=np.float64)
    final_sum = mean_rgb * n

    obj = _engine_skeleton(tmp_path)
    obj.save_final_as_float32 = requested_float32
    obj._capture_run_provenance_requested(
        stacking_mode_requested="mean",
        save_as_float32_requested=requested_float32,
        preserve_linear_output_requested=False,
        drizzle_requested=False,
    )
    saved = _classic_save(obj, final_sum, wht, "_c33")

    # FITS dtype matches the requested option (reopened with astropy).
    if requested_float32:
        assert saved.dtype.kind == "f" and saved.dtype.itemsize == 4
    else:
        assert saved.dtype == np.uint16

    # Content faithful: CHW == per-channel mean (float32) or full-scale uint16.
    saved_c = np.moveaxis(np.asarray(saved, dtype=np.float64), 0, -1)  # HWC
    if requested_float32:
        assert np.allclose(saved_c, mean_rgb, rtol=1e-5, atol=1e-6)
    else:
        expected = _expected_uint16(mean_rgb)
        assert saved.shape == (3, h, w)
        for c in range(3):
            assert np.array_equal(
                saved[c], expected[..., c]
            ), f"channel {c} uint16 content mismatch"

    # Provenance tokens agree with the request.
    assert obj._run_prov_requested["save_as_float32_requested"] is requested_float32
    assert (
        obj._serialization_effective["save_as_float32_effective"]
        is requested_float32
    )
    assert obj._serialization_effective["output_dtype_effective"] == (
        "float32" if requested_float32 else "uint16"
    )


def _drizzle_sci(shape=(12, 13)):
    """Deterministic positive HWC drizzle science (per-channel, exact values)."""
    h, w = shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    r = np.hypot(xx - w / 2.0, yy - h / 2.0)
    core = np.exp(-(r**2) / (2.0 * 2.5**2)).astype(np.float32)
    return np.stack([core, core * 2.0, core * 0.5], axis=-1).astype(np.float32)


def _drizzle_save(obj, sci, suffix):
    """Run the REAL ``_save_final_stack`` Drizzle finalization over 3 real
    accumulators (identical pattern to the D3.5 witness)."""
    from seestar.queuep.queue_manager import (
        FINALIZATION_MODE_DRIZZLE,
        SeestarQueuedStacker,
    )

    h, w = sci.shape[:2]
    obj.finalization_mode = FINALIZATION_MODE_DRIZZLE
    obj.drizzle_active_session = True
    obj.drizzle_accumulators = []
    for c in range(3):
        acc = DrizzleAccumulator((h, w), kernel="square", pixfrac=1.0)
        acc._out_img[:] = sci[..., c]
        acc._out_wht[:] = np.ones((h, w), dtype=np.float32)
        obj.drizzle_accumulators.append(acc)
    obj.drizzle_output_wcs = None
    obj.images_in_cumulative_stack = 1
    obj.total_exposure_seconds = 1.0
    SeestarQueuedStacker._save_final_stack(obj, output_filename_suffix=suffix)
    return fits.getdata(obj.final_stacked_path)


@pytest.mark.parametrize("requested_float32", [True, False])
def test_drizzle_route_e2e_dtype_and_tokens(tmp_path, requested_float32):
    """Drizzle route (independent): real accumulator finalization e2e."""
    sci = _drizzle_sci()
    obj = _engine_skeleton(tmp_path)
    obj.save_final_as_float32 = requested_float32
    obj._capture_run_provenance_requested(
        stacking_mode_requested="winsorized-sigma-clip",
        save_as_float32_requested=requested_float32,
        preserve_linear_output_requested=False,
        drizzle_requested=True,
        drizzle_kernel_requested="square",
    )
    saved = _drizzle_save(obj, sci, "_d33")

    if requested_float32:
        assert saved.dtype.kind == "f" and saved.dtype.itemsize == 4
        # The float32 FITS primary is CHW and matches the drizzle science.
        assert saved.shape == (3,) + sci.shape[:2]
        for c in range(3):
            assert np.allclose(
                np.asarray(saved[c], dtype=np.float32),
                sci[..., c],
                rtol=1e-6,
                atol=1e-6,
            )
    else:
        assert saved.dtype == np.uint16
        assert np.min(saved) >= 0
        # Full-scale conversion (float rounding can land on 65534/65535).
        assert np.max(saved) >= 65534
        # Channels scaled by the shared maximum (HWC max over all channels).
        expected_c = _expected_uint16(sci)
        for c in range(3):
            assert np.array_equal(
                saved[c], expected_c[..., c]
            ), f"channel {c} uint16 content mismatch"

    assert obj._run_prov_requested["save_as_float32_requested"] is requested_float32
    assert (
        obj._serialization_effective["save_as_float32_effective"]
        is requested_float32
    )
    assert obj._serialization_effective["output_dtype_effective"] == (
        "float32" if requested_float32 else "uint16"
    )
    # Drizzle header records the finalization mode truthfully.
    hdr = fits.getheader(obj.final_stacked_path)
    assert hdr.get("DRZMODE") == "M3"
