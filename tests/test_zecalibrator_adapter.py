"""Focused tests for the optional ZeCalibrator adapter (public-import-only).

These tests never require a real ZeCalibrator installation: they inject a fake
``zecalibrator.api.v1`` module into ``sys.modules`` via monkeypatch and exercise
the probe, capability negotiation, result mapping and failure handling.

Both the port and the adapter are loaded directly by file path (mirroring
``tests/test_zesolver_adapter.py``) so the tests never pull the heavy ``seestar``
package tree.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from collections.abc import Mapping
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_PREEXISTING_MODULE_KEYS = set(sys.modules.keys())


def _install_package_stub(name: str) -> None:
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod


def _load_by_path(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# The adapter does ``from seestar.core.calibration_port import ...``, so
# ``seestar`` / ``seestar.core`` / ``seestar.calibration`` must resolve while the
# adapter executes.  Install temporary stubs, then restore sys.modules afterwards
# (hermetic: no stub outlives this module's import).
for _name in ("seestar", "seestar.core", "seestar.calibration"):
    if _name not in sys.modules:
        _install_package_stub(_name)

port = _load_by_path(
    "seestar.core.calibration_port", "seestar/core/calibration_port.py"
)
adapter = _load_by_path(
    "seestar.calibration.zecalibrator_adapter",
    "seestar/calibration/zecalibrator_adapter.py",
)

for _key in list(sys.modules.keys()):
    if _key not in _PREEXISTING_MODULE_KEYS:
        del sys.modules[_key]


# ---------------------------------------------------------------------------
# Fake ``zecalibrator.api.v1``
# ---------------------------------------------------------------------------
class _FakeApiInfo:
    def __init__(self, api_version, capabilities, product_version="0.0.5"):
        self.api_version = api_version
        self.capabilities = tuple(capabilities)
        self.product_version = product_version


class _FakeCancellationToken:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def is_cancelled(self):
        return self.cancelled


class _FakeOperationCancelled(Exception):
    pass


class _FakeInvalidRequestError(Exception):
    pass


class _FakePlanSourceMismatchError(_FakeInvalidRequestError):
    pass


class _FakeLibraryClosedError(Exception):
    pass


class _FakeImportDeclaration:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeFitsFrameSource:
    def __init__(self, path, declaration=None):
        self.path = path
        self.declaration = declaration


class _FakeAdmission:
    __module__ = "zecalibrator.api.v1"

    def __init__(self, role, path, content_sha256, size_bytes):
        self.role = role
        self.path = path
        self.content_sha256 = content_sha256
        self.size_bytes = size_bytes


class _FakeRejection:
    __module__ = "zecalibrator.api.v1"

    def __init__(self, path, reason_code, detail=""):
        self.path = path
        self.reason_code = reason_code
        self.detail = detail


class _FakePlan:
    __module__ = "zecalibrator.api.v1"

    def __init__(self, plan_id):
        self.plan_id = plan_id
        self.composition = _FakeComposition()


class _FakeComposition:
    __module__ = "zecalibrator.api.v1"

    def __init__(self):
        self.applied_roles = ["dark"]
        self.skipped_roles = []
        self.level = "COMPLETE"
        self.additive_state = "dark_incl_bias"
        self.flat_applied = False
        self.no_candidate_roles = []
        self.rejected_masters = []

    def to_dict(self):
        return {
            "applied_roles": self.applied_roles,
            "skipped_roles": self.skipped_roles,
            "level": self.level,
            "additive_state": self.additive_state,
            "flat_applied": self.flat_applied,
            "no_candidate_roles": self.no_candidate_roles,
            "rejected_masters": self.rejected_masters,
        }


class _FakeRouteResolution:
    def __init__(self, operation_status="COMPLETED", outcome="MATCHED", plan=None):
        self.operation_status = operation_status
        self.outcome = outcome
        self.plan = plan
        self.reasons = []
        self.unverified = []
        self.details = ""


class _FakeProvenance:
    __module__ = "zecalibrator.api.v1"

    def to_dict(self):
        return {"backend": "cpu", "operation_id": "fake"}


class _FakeCalibrationResult:
    __module__ = "zecalibrator.api.v1"

    def __init__(self, status="COMPLETED", data=None, mask=None, reason_code=None):
        self.status = status
        self.data = data
        self.mask = mask
        self.reason_code = reason_code
        self.warnings = ()
        self.provenance = _FakeProvenance()


class _FakeSessionLibrary:
    __module__ = "zecalibrator.api.v1"

    def __init__(self, fingerprint="fp-1", calibrate_raises=None):
        self.fingerprint = fingerprint
        self._calibrate_raises = calibrate_raises
        self.closed = False

    def resolve_light(self, source, *, cancel=None):
        return _FakeRouteResolution(
            operation_status="COMPLETED", outcome="MATCHED", plan=_FakePlan("plan-1")
        )

    def calibrate(self, source, plan, *, cancel=None):
        if self._calibrate_raises is not None:
            raise self._calibrate_raises("foreign plan")
        return _FakeCalibrationResult(
            status="COMPLETED", data="float32-data", mask="dq-mask"
        )

    def close(self):
        self.closed = True


class _FakeSessionLibraryResult:
    def __init__(self, operation_status="COMPLETED", handle=None, fingerprint="", admissions=(), rejected=(), counts_by_role=None, warnings=()):
        self.operation_status = operation_status
        self.handle = handle
        self.fingerprint = fingerprint
        self.admissions = admissions
        self.rejected = rejected
        self.counts_by_role = counts_by_role or {}
        self.warnings = warnings


def _make_v1(*, api_version="1.1", capabilities=None, handle=None):
    caps = capabilities if capabilities is not None else (
        "session_library", "auto_route", "calibrate_frame", "cancel",
    )
    v1 = types.ModuleType("zecalibrator.api.v1")
    v1.get_api_info = lambda: _FakeApiInfo(api_version, caps)
    v1.CancellationToken = _FakeCancellationToken
    v1.OperationCancelled = _FakeOperationCancelled
    v1.InvalidRequestError = _FakeInvalidRequestError
    v1.PlanSourceMismatchError = _FakePlanSourceMismatchError
    v1.LibraryClosedError = _FakeLibraryClosedError
    v1.ImportDeclaration = _FakeImportDeclaration
    v1.FitsFrameSource = _FakeFitsFrameSource

    def open_session_library(root, *, cancel=None):
        return _FakeSessionLibraryResult(
            operation_status="COMPLETED",
            handle=handle if handle is not None else _FakeSessionLibrary(),
            fingerprint="fp-1",
            admissions=(_FakeAdmission("dark", "dark.fits", "a" * 64, 123),),
            rejected=(_FakeRejection("flat.fits", "FLAT_QUALITY_EVIDENCE_INSUFFICIENT"),),
            counts_by_role={"dark": 1},
        )

    v1.open_session_library = open_session_library
    return v1


def _install_zecalibrator(monkeypatch, v1: types.ModuleType) -> None:
    zc = types.ModuleType("zecalibrator")
    zc.__path__ = []
    zc_api = types.ModuleType("zecalibrator.api")
    zc_api.__path__ = []
    monkeypatch.setitem(sys.modules, "zecalibrator", zc)
    monkeypatch.setitem(sys.modules, "zecalibrator.api", zc_api)
    monkeypatch.setitem(sys.modules, "zecalibrator.api.v1", v1)


def _remove_zecalibrator(monkeypatch) -> None:
    for key in list(sys.modules):
        if key == "zecalibrator" or key.startswith("zecalibrator."):
            monkeypatch.delitem(sys.modules, key, raising=False)


ALL_CAPS = ("session_library", "auto_route", "calibrate_frame", "cancel")


# ---------------------------------------------------------------------------
# Case 1 — provider absent -> UNAVAILABLE, historical path intact
# ---------------------------------------------------------------------------
def test_probe_absent_provider_returns_not_installed(monkeypatch):
    _remove_zecalibrator(monkeypatch)
    info = adapter.probe()
    assert info.state is port.ProviderState.NOT_INSTALLED
    assert info.available is False


def test_open_session_absent_provider_returns_unavailable_no_raise(monkeypatch):
    _remove_zecalibrator(monkeypatch)
    provider = adapter.ZeCalibratorProvider()
    result = provider.open_session("/does/not/matter")
    assert result.state is port.CalibrationState.UNAVAILABLE
    assert result.session is None
    assert result.error is not None
    assert result.error.kind is port.ErrorKind.UNAVAILABLE


# ---------------------------------------------------------------------------
# Case 3 — broken import -> UNAVAILABLE, no user traceback
# ---------------------------------------------------------------------------
def test_probe_broken_import_returns_unhealthy_no_traceback(monkeypatch):
    def broken_import():
        raise RuntimeError("synthetic broken import")

    monkeypatch.setattr(adapter, "_import_api", broken_import)
    info = adapter.probe()  # must NOT raise
    assert info.state is port.ProviderState.UNHEALTHY
    assert info.available is False


def test_open_session_broken_import_returns_unavailable_no_traceback(monkeypatch):
    def broken_import():
        raise RuntimeError("synthetic broken import")

    monkeypatch.setattr(adapter, "_import_api", broken_import)
    result = adapter.ZeCalibratorProvider().open_session("/x")
    assert result.state is port.CalibrationState.UNAVAILABLE


# ---------------------------------------------------------------------------
# Case 4 — incompatible major "2.0" -> UNAVAILABLE
# ---------------------------------------------------------------------------
def test_probe_incompatible_major(monkeypatch):
    _install_zecalibrator(monkeypatch, _make_v1(api_version="2.0"))
    info = adapter.probe()
    assert info.state is port.ProviderState.INCOMPATIBLE
    assert info.available is False
    assert info.api_major == "2"
    assert "major" in info.message


def test_open_session_incompatible_major_returns_unavailable(monkeypatch):
    _install_zecalibrator(monkeypatch, _make_v1(api_version="2.0"))
    result = adapter.ZeCalibratorProvider().open_session("/x")
    assert result.state is port.CalibrationState.UNAVAILABLE
    assert result.error.kind is port.ErrorKind.UNAVAILABLE


# ---------------------------------------------------------------------------
# Case 5 — missing capability -> UNAVAILABLE naming the capability
# ---------------------------------------------------------------------------
def test_probe_missing_capability_names_it(monkeypatch):
    _install_zecalibrator(
        monkeypatch, _make_v1(capabilities=("session_library", "calibrate_frame", "cancel"))
    )
    info = adapter.probe()
    assert info.state is port.ProviderState.INCOMPATIBLE
    assert info.available is False
    assert "auto_route" in info.message


def test_open_session_missing_capability_returns_unavailable(monkeypatch):
    _install_zecalibrator(
        monkeypatch, _make_v1(capabilities=("session_library", "calibrate_frame", "cancel"))
    )
    result = adapter.ZeCalibratorProvider().open_session("/x")
    assert result.state is port.CalibrationState.UNAVAILABLE


# ---------------------------------------------------------------------------
# Case 2 — compatible provider -> adapter works end-to-end (neutral types)
# ---------------------------------------------------------------------------
def test_probe_compatible_provider(monkeypatch):
    _install_zecalibrator(monkeypatch, _make_v1(api_version="1.1"))
    info = adapter.probe()
    assert info.state is port.ProviderState.AVAILABLE
    assert info.available is True
    assert info.provider_id == "zecalibrator"
    assert info.api_version == "1.1"
    assert info.api_major == "1"
    assert set(info.capabilities) >= set(ALL_CAPS)


def test_open_session_maps_to_neutral_result(monkeypatch):
    _install_zecalibrator(monkeypatch, _make_v1(api_version="1.1"))
    result = adapter.ZeCalibratorProvider().open_session("/masters")
    assert result.state is port.CalibrationState.COMPLETED
    assert result.session is not None
    assert result.fingerprint == "fp-1"
    assert result.counts_by_role == {"dark": 1}
    assert result.admissions[0].role == "dark"
    assert result.admissions[0].path == "dark.fits"
    assert result.admissions[0].content_sha256 == "a" * 64
    assert result.rejected[0].reason_code == "FLAT_QUALITY_EVIDENCE_INSUFFICIENT"


def test_resolve_light_and_calibrate_map_to_neutral_types(monkeypatch):
    _install_zecalibrator(monkeypatch, _make_v1(api_version="1.1"))
    result = adapter.ZeCalibratorProvider().open_session("/masters")
    session = result.session

    light = port.LightSource(path="light.fits")
    rr = session.resolve_light(light)
    assert rr.state is port.CalibrationState.COMPLETED
    assert rr.outcome == "MATCHED"
    assert rr.plan.plan_id == "plan-1"

    cr = session.calibrate(light, rr.plan)
    assert cr.state is port.CalibrationState.COMPLETED
    assert cr.data == "float32-data"
    assert cr.mask == "dq-mask"
    assert cr.provenance == {"backend": "cpu", "operation_id": "fake"}

    session.close()
    assert session._handle.closed is True


def test_foreign_plan_error_is_translated_to_neutral_failure(monkeypatch):
    handle = _FakeSessionLibrary(calibrate_raises=_FakePlanSourceMismatchError)
    _install_zecalibrator(monkeypatch, _make_v1(api_version="1.1", handle=handle))
    result = adapter.ZeCalibratorProvider().open_session("/masters")
    session = result.session

    plan = session.resolve_light(port.LightSource(path="light.fits")).plan
    cr = session.calibrate(port.LightSource(path="light.fits"), plan)
    assert cr.state is port.CalibrationState.FAILED
    assert cr.error.kind is port.ErrorKind.FAILED
    # no ZeCalibrator type leaks: the error carries only a message string.
    assert "foreign plan" in cr.error.message


def test_plan_is_required_and_typed(monkeypatch):
    _install_zecalibrator(monkeypatch, _make_v1(api_version="1.1"))
    session = adapter.ZeCalibratorProvider().open_session("/masters").session
    cr = session.calibrate(port.LightSource(path="light.fits"), plan=None)
    assert cr.state is port.CalibrationState.FAILED
    assert cr.error is not None


# ---------------------------------------------------------------------------
# Case 6 — import hygiene (separate process)
# ---------------------------------------------------------------------------
def test_adapter_import_does_not_pull_zecalibrator_subprocess():
    code = (
        "import importlib.util, sys\n"
        "from pathlib import Path\n"
        f"ROOT = Path({str(ROOT)!r})\n"
        "def stub(name):\n"
        "    m = __import__('types').ModuleType(name); m.__path__ = []; sys.modules[name] = m\n"
        "for n in ('seestar', 'seestar.core', 'seestar.calibration'):\n"
        "    stub(n)\n"
        "spec = importlib.util.spec_from_file_location('seestar.core.calibration_port', ROOT / 'seestar/core/calibration_port.py')\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['seestar.core.calibration_port'] = m; spec.loader.exec_module(m)\n"
        "spec2 = importlib.util.spec_from_file_location('seestar.calibration.zecalibrator_adapter', ROOT / 'seestar/calibration/zecalibrator_adapter.py')\n"
        "m2 = importlib.util.module_from_spec(spec2); sys.modules['seestar.calibration.zecalibrator_adapter'] = m2; spec2.loader.exec_module(m2)\n"
        "assert 'zecalibrator' not in sys.modules, 'zecalibrator was imported eagerly'\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_adapter_source_references_only_public_module():
    src = (ROOT / "seestar/calibration/zecalibrator_adapter.py").read_text(encoding="utf-8")
    import_lines = [
        line.strip()
        for line in src.splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    # The adapter must never import zecalibrator at module level (lazy import
    # only, via a string literal) and never import private/underscore modules.
    for line in import_lines:
        assert "zecalibrator" not in line, line
    # The only zecalibrator reference is the public module string (no private
    # or GUI module is ever imported).
    assert adapter._API_MODULE == "zecalibrator.api.v1"
    assert "zecalibrator.api.v1._" not in adapter._API_MODULE
    assert "zecalibrator.gui" not in adapter._API_MODULE


# ---------------------------------------------------------------------------
# RW-4 — in-flight cancellation is propagated (live token mapping)
# ---------------------------------------------------------------------------
def test_inflight_cancellation_propagates_RW4(monkeypatch):
    # A neutral handle that flips to cancelled after N polls (mid-operation).
    class _FlipHandle:
        def __init__(self, flip_after):
            self.calls = 0
            self.flip_after = flip_after
            self.cancelled = False

        def is_cancelled(self):
            self.calls += 1
            if self.calls > self.flip_after:
                self.cancelled = True
            return self.cancelled

    handle = _FlipHandle(flip_after=2)

    v1 = _make_v1(api_version="1.1")

    def open_session_library(root, *, cancel=None):
        for _ in range(5):
            cancel.raise_if_cancelled()  # provider cooperative checkpoint
        return _FakeSessionLibraryResult(
            operation_status="COMPLETED", handle=_FakeSessionLibrary(), fingerprint="fp-1",
        )

    v1.open_session_library = open_session_library
    _install_zecalibrator(monkeypatch, v1)

    result = adapter.ZeCalibratorProvider().open_session("/x", cancel=handle)
    assert result.state is port.CalibrationState.CANCELLED
    assert handle.cancelled is True  # the flip happened during the operation


def test_never_cancelling_handle_completes_RW4(monkeypatch):
    class _NeverCancel:
        def is_cancelled(self):
            return False

    v1 = _make_v1(api_version="1.1")

    def open_session_library(root, *, cancel=None):
        for _ in range(5):
            cancel.raise_if_cancelled()
        return _FakeSessionLibraryResult(
            operation_status="COMPLETED", handle=_FakeSessionLibrary(), fingerprint="fp-1",
        )

    v1.open_session_library = open_session_library
    _install_zecalibrator(monkeypatch, v1)

    result = adapter.ZeCalibratorProvider().open_session("/x", cancel=_NeverCancel())
    assert result.state is port.CalibrationState.COMPLETED


# ---------------------------------------------------------------------------
# RW-5 — effective composition carried by RouteResolution (preflight, JSON-safe)
# ---------------------------------------------------------------------------
def test_route_resolution_carries_json_safe_composition_RW5(monkeypatch):
    _install_zecalibrator(monkeypatch, _make_v1(api_version="1.1"))
    session = adapter.ZeCalibratorProvider().open_session("/masters").session

    rr = session.resolve_light(port.LightSource(path="light.fits"))
    assert rr.state is port.CalibrationState.COMPLETED
    comp = rr.composition
    assert comp is not None
    assert comp.level == "COMPLETE"
    assert comp.additive_state == "dark_incl_bias"
    assert comp.applied_roles == ("dark",)
    assert comp.flat_applied is False

    d = comp.to_dict()
    assert d["applied_roles"] == ["dark"]
    json.dumps(d)  # JSON-safe (no exception)


# ---------------------------------------------------------------------------
# RW-6 — no provider type leaks across the neutral boundary
# ---------------------------------------------------------------------------
def _assert_no_provider_type(value, path):
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, Mapping):
        for k, v in value.items():
            _assert_no_provider_type(v, f"{path}.{k}")
        return
    if isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            _assert_no_provider_type(v, f"{path}[{i}]")
        return
    mod = type(value).__module__
    assert not mod.startswith("zecalibrator"), (path, type(value).__name__, mod)
    if hasattr(value, "__dataclass_fields__"):
        for fname in value.__dataclass_fields__:
            _assert_no_provider_type(getattr(value, fname), f"{path}.{fname}")


def test_no_provider_type_leak_RW6(monkeypatch):
    # The fake provider objects carry __module__ == "zecalibrator.api.v1" so this
    # walk genuinely fails if the adapter ever passes one through as a field value.
    _install_zecalibrator(monkeypatch, _make_v1(api_version="1.1"))
    res = adapter.ZeCalibratorProvider().open_session("/masters")
    session = res.session
    rr = session.resolve_light(port.LightSource(path="light.fits"))
    cr = session.calibrate(port.LightSource(path="light.fits"), rr.plan)

    for obj, name in (
        (res, "SessionResult"),
        (rr, "RouteResolution"),
        (cr, "CalibrationResult"),
    ):
        for fname in obj.__dataclass_fields__:
            _assert_no_provider_type(getattr(obj, fname), f"{name}.{fname}")

    # The ONLY opaque provider retention is CalibrationPlan._provider_plan
    # (a private attribute of a non-dataclass, never a dataclass field).
    assert not hasattr(port.CalibrationPlan, "__dataclass_fields__")
