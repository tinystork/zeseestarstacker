"""REWORK-2 C/D — integration tests for the REAL ``_calibrate_frame_to_working``.

These drive the real engine method with provider/session doubles (a real
``CalibrationIntegrator`` over a fake provider + fake session) so the counters
increment *automatically* (not via manual ``record_*`` calls), and then re-read
the durable final proof (``_calibration_provenance`` + the versioned
``calibration_provenance.json`` artifact written by ``_close_calibration_session``).

Four outcomes are covered: success / no-plan / calibrate-none / load-fail.
No ZeCalibrator, no Qt, no GPU: a tiny synthetic FITS file feeds the real
``load_and_validate_fits``.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from seestar.calibration.streaming import CalibrationIntegrator
from seestar.core.calibration_port import (
    CalibrationComposition,
    CalibrationPlan,
    CalibrationResult,
    CalibrationState,
    ProviderInfo,
    ProviderState,
    RouteResolution,
    SessionResult,
)
from seestar.queuep.queue_manager import SeestarQueuedStacker


class _FakeSession:
    """Configurable fake session: resolve_light / calibrate outcomes."""
    fingerprint = "f" * 64
    context_preparation_count = 0

    def __init__(self, *, resolve="match", calibrate="ok"):
        self._resolve = resolve
        self._calibrate = calibrate

    def resolve_light(self, source, *, cancel=None):
        if self._resolve == "no_match":
            return RouteResolution(
                state=CalibrationState.COMPLETED,
                outcome="NO_MATCH",
                reasons=("EXPOSURE_MISMATCH",),
            )
        if self._resolve == "failed":
            return RouteResolution(state=CalibrationState.FAILED)
        return RouteResolution(
            state=CalibrationState.COMPLETED,
            outcome="MATCHED",
            plan=CalibrationPlan("plan-1", object()),
            composition=CalibrationComposition(
                applied_roles=("dark", "flat"),
                level="COMPLETE",
                additive_state="dark_incl_bias",
                flat_applied=True,
            ),
        )

    def calibrate(self, source, plan, *, cancel=None):
        if self._calibrate == "none":
            return CalibrationResult(state=CalibrationState.FAILED)
        data = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        mask = np.zeros((2, 2), dtype=np.uint16)
        return CalibrationResult(state=CalibrationState.COMPLETED, data=data, mask=mask)

    def close(self):
        pass


class _FakeProvider:
    name = "fake"

    def probe(self):
        return ProviderInfo(
            state=ProviderState.AVAILABLE,
            provider_id="fake",
            api_version="1.1",
            product_version="0.0.5",
        )

    def route_key(self, path):
        return "route-key"

    def route_key_result(self, path):
        from seestar.core.calibration_port import RouteKeyResult

        return RouteKeyResult(key="route-key")

    def open_session(self, root, *, cancel=None, sensor_orientation=None):
        return SessionResult(
            state=CalibrationState.COMPLETED,
            session=_FakeSession(resolve=self._resolve, calibrate=self._calibrate),
            fingerprint="f" * 64,
        )

    def __init__(self, *, resolve="match", calibrate="ok"):
        self._resolve = resolve
        self._calibrate = calibrate


def _make_fits(tmp_path) -> str:
    """Write a tiny 2x2 float32 FITS the real loader accepts."""
    from astropy.io import fits

    path = os.path.join(str(tmp_path), "light.fits")
    hdu = fits.PrimaryHDU(np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32))
    hdu.writeto(path, overwrite=True)
    return path


def _stacker(tmp_path, *, resolve="match", calibrate="ok"):
    """Bare engine instance wired with a real integrator over a fake provider."""
    s = object.__new__(SeestarQueuedStacker)
    s.output_folder = str(tmp_path)
    s._calibration_masks = {}
    s._calibration_provenance = {}
    s.update_progress = lambda *a, **k: None
    integ = CalibrationIntegrator(
        str(tmp_path), provider=_FakeProvider(resolve=resolve, calibrate=calibrate)
    )
    assert integ.open() is True
    s._calibration_integrator = integ
    return s


def _read_artifact(tmp_path):
    path = os.path.join(str(tmp_path), "calibration_provenance.json")
    assert os.path.exists(path), f"durable artifact missing: {path}"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_frame_success_increments_applied_and_persists_durable_proof(tmp_path):
    s = _stacker(tmp_path)
    light = _make_fits(tmp_path)

    out = s._calibrate_frame_to_working(light)
    assert out is not None
    working, header, mask = out
    assert working.dtype == np.float32
    assert mask.dtype == np.uint16  # DQ carried

    prov = s._calibration_integrator.provenance_snapshot()
    assert prov["calibration_requested"] == 1
    assert prov["calibration_frames_applied"] == 1
    assert prov["calibration_frames_skipped"] == 0
    assert prov["calibration_frames_failed"] == 0
    assert prov["calibration_dq_present"] is True
    assert "dark" in prov["calibration_effective_roles"]

    # Close + re-read the durable final proof.
    s._close_calibration_session()
    artifact = _read_artifact(tmp_path)
    assert artifact["schema_version"] == 1
    assert artifact["calibration_frames_applied"] == 1
    assert artifact["calibration_frames_skipped"] == 0


def test_frame_no_plan_increments_skipped_with_real_reason(tmp_path):
    s = _stacker(tmp_path, resolve="no_match")
    light = _make_fits(tmp_path)

    out = s._calibrate_frame_to_working(light)
    assert out is None

    prov = s._calibration_integrator.provenance_snapshot()
    assert prov["calibration_frames_skipped"] == 1
    assert prov["calibration_frames_applied"] == 0
    assert prov["calibration_skip_reasons"] == {"resolve:no_match": 1}

    s._close_calibration_session()
    artifact = _read_artifact(tmp_path)
    assert artifact["calibration_frames_skipped"] == 1
    assert artifact["calibration_skip_reasons"] == {"resolve:no_match": 1}


def test_frame_calibrate_none_increments_failed_with_real_reason(tmp_path):
    s = _stacker(tmp_path, calibrate="none")
    light = _make_fits(tmp_path)

    out = s._calibrate_frame_to_working(light)
    assert out is None

    prov = s._calibration_integrator.provenance_snapshot()
    assert prov["calibration_frames_failed"] == 1
    assert prov["calibration_failure_reasons"] == {"failed": 1}

    s._close_calibration_session()
    artifact = _read_artifact(tmp_path)
    assert artifact["calibration_frames_failed"] == 1


def test_frame_load_fail_increments_failed(tmp_path):
    s = _stacker(tmp_path)
    missing = os.path.join(str(tmp_path), "does_not_exist.fits")

    out = s._calibrate_frame_to_working(missing)
    assert out is None

    prov = s._calibration_integrator.provenance_snapshot()
    assert prov["calibration_frames_failed"] == 1
    assert prov["calibration_failure_reasons"] == {"light_load_failed": 1}

    s._close_calibration_session()
    artifact = _read_artifact(tmp_path)
    assert artifact["calibration_frames_failed"] == 1


def test_open_failure_surfaces_real_reason_not_generic(tmp_path):
    """INCOMPATIBLE probe must be surfaced actionably (REWORK defect #4)."""
    from seestar.core.calibration_port import ProviderInfo, ProviderState

    class _IncompatibleProvider(_FakeProvider):
        def probe(self):
            return ProviderInfo(
                state=ProviderState.INCOMPATIBLE,
                message="missing public API symbol(s): ['light_route_key']",
            )

        def open_session(self, root, *, cancel=None, sensor_orientation=None):
            from seestar.core.calibration_port import (
                CalibrationError,
                ErrorKind,
                SessionResult,
            )

            return SessionResult(
                state=CalibrationState.UNAVAILABLE,
                error=CalibrationError(
                    ErrorKind.UNAVAILABLE,
                    "missing public API symbol(s): ['light_route_key']",
                ),
            )

    integ = CalibrationIntegrator(str(tmp_path), provider=_IncompatibleProvider())
    info = integ.probe_info()
    assert info.available is False
    assert "light_route_key" in info.message

    # The engine seam must surface this (not "no usable masters").
    s = object.__new__(SeestarQueuedStacker)
    s._calibration_enabled = True
    s._calibration_master_folder = str(tmp_path)
    s._calibration_orientation = ""
    s._calibration_integrator = None
    s._calibration_unavailable_reason = None
    messages = []
    s.update_progress = lambda m, level=None: messages.append(m)

    import seestar.calibration.streaming as streaming_mod

    # Patch the integrator the seam constructs to use our incompatible provider.
    orig = streaming_mod.CalibrationIntegrator

    def _fake_integrator(folder, *, provider=None, plan_map=None, sensor_orientation=None):
        return integ

    streaming_mod.CalibrationIntegrator = _fake_integrator
    try:
        s._open_calibration_session()
    finally:
        streaming_mod.CalibrationIntegrator = orig

    assert s._calibration_integrator is None
    assert "light_route_key" in s._calibration_unavailable_reason
    assert any("light_route_key" in m for m in messages)


def test_e2e_backend_runner_seam_propagates_and_counts(tmp_path, monkeypatch):
    """REWORK D — E2E: GUI seam -> engine -> counters -> durable proof.

    Goes through the REAL ``SeestarQueuedStackerBackend`` seam
    (``split_backend_kwargs`` + ``_apply_seam_kwargs``) and the REAL engine
    calibration lifecycle (``_open_calibration_session`` /
    ``_calibrate_frame_to_working`` / ``_close_calibration_session``) with a
    fake-backed integrator injected via monkeypatch — no manual ``_calibration_*``
    assignment.  Synthetic tiny FITS, no GPU, no full run.
    """
    import seestar.calibration.streaming as streaming_mod
    from seestar.gui_qt.backend_runner import SeestarQueuedStackerBackend
    from seestar.gui_qt.run_bridge import build_run_request, split_backend_kwargs
    from seestar.gui_qt.run_handoff import attach_run_settings
    from seestar.gui_qt.settings_state import QtSettingsState

    real_integ = streaming_mod.CalibrationIntegrator

    def _fake_integrator(folder, *, provider=None, plan_map=None, sensor_orientation=None):
        return real_integ(
            folder,
            provider=_FakeProvider(resolve="match", calibrate="ok"),
            plan_map=plan_map,
            sensor_orientation=sensor_orientation,
        )

    monkeypatch.setattr(streaming_mod, "CalibrationIntegrator", _fake_integrator)

    # Build a real RunRequest through the GUI builders.
    state = QtSettingsState(
        calibration_enabled=True,
        calibration_master_folder=str(tmp_path),
        calibration_orientation="identity",
    )
    request = attach_run_settings(
        build_run_request(state),
        use_gpu=False,
        reference_origin_hint=None,
        calibration_enabled=True,
        calibration_master_folder=str(tmp_path),
        calibration_orientation="identity",
    )
    start_kwargs, seam_kwargs = split_backend_kwargs(request.backend_kwargs)
    assert "calibration_enabled" not in start_kwargs  # seam, never a start kwarg

    # Real engine instance + REAL seam application (never manual private).
    s = object.__new__(SeestarQueuedStacker)
    s.output_folder = str(tmp_path)
    s._calibration_masks = {}
    s._calibration_provenance = {}
    s.update_progress = lambda *a, **k: None
    SeestarQueuedStackerBackend._apply_seam_kwargs(s, seam_kwargs)

    # The seam set the instance fields.
    assert s._calibration_enabled is True
    assert s._calibration_master_folder == str(tmp_path)
    assert s._calibration_orientation == "identity"

    # Real engine lifecycle.
    s._open_calibration_session()
    assert s._calibration_integrator is not None

    light = _make_fits(tmp_path)
    out = s._calibrate_frame_to_working(light)
    assert out is not None

    s._close_calibration_session()
    artifact = _read_artifact(tmp_path)
    assert artifact["calibration_frames_applied"] == 1
    assert artifact["calibration_session_open"] is True
    assert artifact["calibration_classes_planned"] >= 0


def _engine_stackers(tmp_path, *, enabled=True, folder="", orientation=""):
    """Bare engine with the C5 seam fields set + a message capture sink."""
    s = object.__new__(SeestarQueuedStacker)
    s.output_folder = str(tmp_path)
    s._calibration_masks = {}
    s._calibration_provenance = {}
    s._calibration_enabled = enabled
    s._calibration_master_folder = folder
    s._calibration_orientation = orientation
    s._calibration_integrator = None
    s._calibration_unavailable_reason = None
    messages = []
    s.update_progress = lambda m, level=None: messages.append(m)
    return s, messages


def test_f1_freeze_caches_roles_then_frame_applies(tmp_path):
    """F1: freeze_snapshot(light) must record effective roles at cache time, so
    the later cached resolve() + _calibrate_frame_to_working still reports
    dark+flat in the final artifact."""
    s = _stacker(tmp_path)
    light = _make_fits(tmp_path)
    # Preflight: freeze the plan (caches plan + records roles).
    freeze = s._calibration_integrator.freeze_snapshot([light])
    assert freeze["calibration_plan_map"]
    assert "dark" in s._calibration_integrator.provenance_snapshot()["calibration_effective_roles"]

    # Real frame path hits the cache (no re-decode).
    out = s._calibrate_frame_to_working(light)
    assert out is not None

    s._close_calibration_session()
    artifact = _read_artifact(tmp_path)
    assert artifact["calibration_frames_applied"] == 1
    assert "dark" in artifact["calibration_effective_roles"]
    assert "flat" in artifact["calibration_effective_roles"]


def test_f3_incompatible_provider_writes_minimal_artifact(tmp_path):
    """F3: calibration requested but provider INCOMPATIBLE must still write a
    durable minimal artifact at close (session_open=false + open_reason)."""
    s, messages = _engine_stackers(tmp_path, folder=str(tmp_path))

    import seestar.calibration.streaming as streaming_mod

    class _Incompat(_FakeProvider):
        def probe(self):
            return ProviderInfo(
                state=ProviderState.INCOMPATIBLE,
                message="missing public API symbol(s): ['light_route_key']",
            )

    integ = CalibrationIntegrator(str(tmp_path), provider=_Incompat())
    orig = streaming_mod.CalibrationIntegrator
    streaming_mod.CalibrationIntegrator = lambda *a, **k: integ
    try:
        s._open_calibration_session()
    finally:
        streaming_mod.CalibrationIntegrator = orig

    assert s._calibration_integrator is None
    s._close_calibration_session()

    artifact = _read_artifact(tmp_path)
    assert artifact["calibration_enabled_requested"] is True
    assert artifact["calibration_session_open"] is False
    assert artifact["calibration_frames_applied"] == 0
    assert "light_route_key" in artifact["calibration_open_reason"]
    # A durable log block was also emitted.
    assert any("CALIBRATION_PROVENANCE" in m for m in messages)


def test_f3_missing_folder_writes_minimal_artifact(tmp_path):
    """F3: enabled but no master folder -> minimal artifact (no session)."""
    s, _ = _engine_stackers(tmp_path, folder="")
    s._open_calibration_session()
    s._close_calibration_session()
    artifact = _read_artifact(tmp_path)
    assert artifact["calibration_session_open"] is False
    assert artifact["calibration_open_reason"] == "no_master_folder"
    assert artifact["calibration_frames_applied"] == 0


def test_f3_disabled_writes_no_artifact(tmp_path):
    """Calibration disabled -> no artifact (absence == disabled)."""
    s, _ = _engine_stackers(tmp_path, enabled=False)
    s._open_calibration_session()
    s._close_calibration_session()
    assert not os.path.exists(os.path.join(str(tmp_path), "calibration_provenance.json"))


def test_close_exception_still_writes_artifact(tmp_path):
    """Robustness: a raising session.close() must not break the finalize path nor
    lose the durable proof; a bounded close_failed reason is aggregated."""
    class _CloseRaises(_FakeSession):
        def close(self):
            raise RuntimeError("close boom")

    class _Provider(_FakeProvider):
        def open_session(self, root, *, cancel=None, sensor_orientation=None):
            return SessionResult(
                state=CalibrationState.COMPLETED,
                session=_CloseRaises(),
                fingerprint="f" * 64,
            )

    s = _stacker(tmp_path)
    # Replace the integrator's session with the raising one via a fresh integrator.
    integ = CalibrationIntegrator(str(tmp_path), provider=_Provider())
    assert integ.open() is True
    s._calibration_integrator = integ
    light = _make_fits(tmp_path)
    assert s._calibrate_frame_to_working(light) is not None

    s._close_calibration_session()
    artifact = _read_artifact(tmp_path)
    assert artifact["calibration_frames_applied"] == 1
    assert artifact["calibration_close_reason"] == "close_failed:RuntimeError"


def test_write_failure_emits_durable_event(tmp_path):
    """Robustness: a failed artifact write is fail-open for science but emits a
    bounded durable CALIBRATION_PROVENANCE_WRITE_FAILURE block (never silent)."""
    s, messages = _engine_stackers(tmp_path, folder=str(tmp_path))
    # enabled + a fake integrator so _close has a snapshot to write.
    import seestar.calibration.streaming as streaming_mod

    integ = CalibrationIntegrator(str(tmp_path), provider=_FakeProvider())
    assert integ.open() is True
    s._calibration_integrator = integ

    # Force the artifact write to fail: output_folder points into a file path.
    blocker = os.path.join(str(tmp_path), "blocked")
    with open(blocker, "w") as fh:
        fh.write("x")
    s.output_folder = blocker  # os.path.join(blocker, filename) will fail

    s._close_calibration_session()
    assert any("CALIBRATION_PROVENANCE_WRITE_FAILURE" in m for m in messages)


def test_close_is_idempotent(tmp_path):
    """Robustness: worker finally + explicit stop both call close; the second is a no-op."""
    s = _stacker(tmp_path)
    light = _make_fits(tmp_path)
    assert s._calibrate_frame_to_working(light) is not None
    s._close_calibration_session()
    first = _read_artifact(tmp_path)
    s._close_calibration_session()  # must not raise nor overwrite
    second = _read_artifact(tmp_path)
    assert first == second


def test_calibrate_cancelled_records_failure_reason(tmp_path):
    """Lifecycle: a CANCELLED calibration maps to a stable failure reason."""
    class _CancelSession(_FakeSession):
        def calibrate(self, source, plan, *, cancel=None):
            return CalibrationResult(state=CalibrationState.CANCELLED)

    class _Provider(_FakeProvider):
        def open_session(self, root, *, cancel=None, sensor_orientation=None):
            return SessionResult(
                state=CalibrationState.COMPLETED,
                session=_CancelSession(),
                fingerprint="f" * 64,
            )

    s = _stacker(tmp_path)
    integ = CalibrationIntegrator(str(tmp_path), provider=_Provider())
    assert integ.open() is True
    s._calibration_integrator = integ
    light = _make_fits(tmp_path)
    out = s._calibrate_frame_to_working(light)
    assert out is None
    prov = integ.provenance_snapshot()
    assert prov["calibration_frames_failed"] == 1
    assert prov["calibration_failure_reasons"] == {"cancelled": 1}
