"""C-provenance — real-reason preservation + bounded runtime counters (LOT A).

Unit-level checks of the ``CalibrationIntegrator`` provenance: the counters are
incremented through the **real** ``resolve()``/``calibrate()`` code paths (never
manual ``record_*`` calls), the skip/failure reasons are the provider's **real**
stable tokens (not fabricated text), and ``session_open`` is kept distinct from
``classes_planned`` and from ``frames applied/skipped/failed``.

No ZeCalibrator / Qt required: a fake provider + a fake session drive the
``CalibrationIntegrator`` directly.  The engine-boundary integration tests live
in ``tests/test_calibration_frame_integration.py`` (real ``_calibrate_frame_to_working``).
"""

from __future__ import annotations

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


class _FakeProvider:
    name = "fake"

    def __init__(self, *, resolve="match", calibrate="ok", key="route-key"):
        self._resolve = resolve
        self._calibrate = calibrate
        self._key = key

    def probe(self):
        return ProviderInfo(
            state=ProviderState.AVAILABLE,
            provider_id="fake",
            api_version="1.1",
            product_version="0.0.5",
        )

    def route_key(self, path):
        return self._key

    def open_session(self, root, *, cancel=None, sensor_orientation=None):
        return SessionResult(
            state=CalibrationState.COMPLETED,
            session=_FakeSession(resolve=self._resolve, calibrate=self._calibrate),
            fingerprint="f" * 64,
        )


class _FakeSession:
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
        if self._resolve == "multi_no_match":
            return RouteResolution(
                state=CalibrationState.COMPLETED,
                outcome="NO_MATCH",
                reasons=("EXPOSURE_MISMATCH", "TEMPERATURE_MISMATCH"),
            )
        if self._resolve == "ambiguous":
            return RouteResolution(
                state=CalibrationState.COMPLETED,
                outcome="AMBIGUOUS",
                reasons=("AMBIGUOUS",),
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
        return CalibrationResult(
            state=CalibrationState.COMPLETED,
            data=object(),
            mask=object(),
        )

    def close(self):
        pass


def _integrator(*, resolve="match", calibrate="ok", key="route-key"):
    return CalibrationIntegrator(
        "/masters", provider=_FakeProvider(resolve=resolve, calibrate=calibrate, key=key)
    )


def test_resolve_and_calibrate_increment_counters_through_real_path():
    integ = _integrator()
    assert integ.open() is True

    plan = integ.resolve("/l.fits")
    assert plan is not None
    cal = integ.calibrate("/l.fits", plan)
    assert cal is not None

    prov = integ.provenance_snapshot()
    assert prov["calibration_session_open"] is True
    assert prov["calibration_classes_planned"] == 1
    assert prov["calibration_dq_present"] is True
    assert "dark" in prov["calibration_effective_roles"]
    assert "flat" in prov["calibration_effective_roles"]
    integ.close()


def test_resolve_no_match_records_real_reason_not_fabricated_text():
    integ = _integrator(resolve="no_match")
    assert integ.open() is True

    plan = integ.resolve("/l.fits")
    assert plan is None

    prov = integ.provenance_snapshot()
    assert prov["calibration_frames_skipped"] == 1
    assert prov["calibration_skip_reasons"] == {"resolve:no_match": 1}
    integ.close()


def test_route_key_unavailable_records_real_reason():
    integ = _integrator(key=None)
    assert integ.open() is True

    plan = integ.resolve("/l.fits")
    assert plan is None

    prov = integ.provenance_snapshot()
    assert prov["calibration_frames_skipped"] == 1
    assert prov["calibration_skip_reasons"] == {"route_key:unavailable": 1}
    integ.close()


def test_calibrate_failed_records_real_reason():
    integ = _integrator(calibrate="none")
    assert integ.open() is True

    plan = integ.resolve("/l.fits")
    assert plan is not None
    cal = integ.calibrate("/l.fits", plan)
    assert cal is None

    prov = integ.provenance_snapshot()
    assert prov["calibration_frames_failed"] == 1
    assert prov["calibration_failure_reasons"] == {"failed": 1}
    integ.close()


def test_provenance_distinguishes_session_open_from_plan_resolved():
    integ = _integrator()
    assert integ.open() is True
    prov = integ.provenance_snapshot()
    assert prov["calibration_session_open"] is True
    assert prov["calibration_classes_planned"] == 0
    assert prov["calibration_frames_applied"] == 0
    integ.close()


def test_freeze_plan_reason_recorded_when_no_plan_resolved():
    # A provider whose route_key always returns None -> freeze has an empty plan
    # map AND a durable reason (never a silent empty map).
    integ = _integrator(key=None)
    assert integ.open() is True
    freeze = integ.freeze_snapshot(["/l.fits"])
    assert freeze["calibration_plan_map"] == {}
    assert integ.provenance_snapshot()["calibration_plan_reason"] is not None
    integ.close()


def test_freeze_route_key_failure_is_aggregated():
    # F2: a light that cannot be route-classed is a real freeze cause; the
    # plan_reason must reflect it (never a bare "no representatives resolved").
    integ = _integrator(key=None)
    assert integ.open() is True
    integ.freeze_snapshot(["/l.fits"])
    prov = integ.provenance_snapshot()
    assert "route_key:unavailable" in prov["calibration_resolve_reasons"]
    assert prov["calibration_plan_reason"] == "route_key:unavailable:1"
    integ.close()


def test_freeze_resets_resolve_reasons_to_avoid_double_count():
    # F2: repeated freeze reconstruction must not double-count the aggregate.
    integ = _integrator(key=None)
    assert integ.open() is True
    integ.freeze_snapshot(["/l.fits"])
    integ.freeze_snapshot(["/l.fits"])
    prov = integ.provenance_snapshot()
    assert prov["calibration_resolve_reasons"] == {"route_key:unavailable": 1}
    integ.close()


def test_freeze_caches_effective_roles_for_cached_resolve():
    # F1: roles must be recorded at freeze/cache time; a cached resolve() then
    # returns the plan without re-reading the composition but the roles persist.
    integ = _integrator()
    assert integ.open() is True
    freeze = integ.freeze_snapshot(["/l.fits"])
    assert freeze["calibration_plan_map"]  # one class planned
    prov = integ.provenance_snapshot()
    assert "dark" in prov["calibration_effective_roles"]
    assert "flat" in prov["calibration_effective_roles"]
    # resolve() now hits the cache and must NOT lose the roles.
    plan = integ.resolve("/l.fits")
    assert plan is not None
    assert "dark" in integ.provenance_snapshot()["calibration_effective_roles"]
    integ.close()


def test_resolve_preserves_provider_reason_codes():
    # F2: NO_MATCH with multiple provider codes must keep the stable outcome AND
    # the real codes in the bounded resolve-reasons aggregate.
    class _MultiProvider(_FakeProvider):
        pass

    integ = CalibrationIntegrator("/masters", provider=_MultiProvider(resolve="multi_no_match"))
    assert integ.open() is True
    plan = integ.resolve("/l.fits")
    assert plan is None
    prov = integ.provenance_snapshot()
    assert prov["calibration_skip_reasons"] == {"resolve:no_match": 1}
    rr = prov["calibration_resolve_reasons"]
    assert rr["no_match"] == 1
    assert "code:EXPOSURE_MISMATCH" in rr
    assert "code:TEMPERATURE_MISMATCH" in rr
    integ.close()


def test_resolve_ambiguous_records_stable_outcome():
    # F2: AMBIGUOUS must map to a stable token (never a traceback/code leak).
    integ = _integrator(resolve="ambiguous")
    assert integ.open() is True
    plan = integ.resolve("/l.fits")
    assert plan is None
    prov = integ.provenance_snapshot()
    assert prov["calibration_skip_reasons"] == {"resolve:ambiguous": 1}
    assert prov["calibration_resolve_reasons"]["ambiguous"] == 1
    integ.close()


def test_probe_exception_returns_unhealthy():
    # F4: a raising provider.probe() must map to UNHEALTHY (never raise).
    class _RaisingProvider(_FakeProvider):
        def probe(self):
            raise RuntimeError("boom")

    integ = CalibrationIntegrator("/masters", provider=_RaisingProvider())
    info = integ.probe_info()
    assert info.state == ProviderState.UNHEALTHY
    assert info.available is False
    assert "RuntimeError" in info.message
