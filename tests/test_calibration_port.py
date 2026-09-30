"""Tests for the transport-neutral calibration port (no ZeCalibrator, no Qt).

The port is loaded directly by file path so these tests never pull the heavy
``seestar`` package tree (OpenCV, NumPy, Astropy, …).  A fake provider proves the
port and its protocol are decoupled from any concrete calibration provider.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT_RELPATH = "seestar/core/calibration_port.py"


def _load_by_path(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load the transport-neutral port under a NON-canonical module name.  Loading it
# under ``seestar.core.calibration_port`` would shadow the real module in
# ``sys.modules`` and create a SECOND ``ProviderState`` enum, breaking the
# ``is`` identity check in the GUI tab probe (order-dependent test pollution).
# The port is stdlib-only and never self-references, so the name is arbitrary.
port = _load_by_path("seestar.core.calibration_port__under_test", PORT_RELPATH)


# ---------------------------------------------------------------------------
# Neutral types + error taxonomy
# ---------------------------------------------------------------------------
def test_error_kinds_distinguish_unavailable_from_failed():
    unavailable = port.CalibrationError(port.ErrorKind.UNAVAILABLE, "not installed")
    failed = port.CalibrationError(port.ErrorKind.FAILED, "bad master")
    cancelled = port.CalibrationError(port.ErrorKind.CANCELLED, "cancelled")

    assert unavailable.kind is port.ErrorKind.UNAVAILABLE
    assert failed.kind is port.ErrorKind.FAILED
    assert cancelled.kind is port.ErrorKind.CANCELLED
    # "unavailable" is a distinct notion from "failure".
    assert unavailable.kind is not port.ErrorKind.FAILED
    # CalibrationError is a normal Exception.
    assert isinstance(failed, Exception)


def test_states_are_distinct_and_stable():
    states = {
        port.CalibrationState.COMPLETED,
        port.CalibrationState.CANCELLED,
        port.CalibrationState.FAILED,
        port.CalibrationState.UNAVAILABLE,
    }
    assert len(states) == 4
    assert port.CalibrationState.UNAVAILABLE.value == "UNAVAILABLE"


def test_plan_is_opaque_with_stable_plan_id():
    provider_plan = object()
    plan = port.CalibrationPlan("deadbeef", provider_plan)
    assert plan.plan_id == "deadbeef"
    assert plan._provider_plan is provider_plan  # opaque carrier, never exposed publicly


def test_result_envelopes_default_to_unavailable():
    assert port.SessionResult().state is port.CalibrationState.UNAVAILABLE
    assert port.RouteResolution().state is port.CalibrationState.UNAVAILABLE
    assert port.CalibrationResult().state is port.CalibrationState.UNAVAILABLE


def test_provider_info_available_flag():
    avail = port.ProviderInfo(
        state=port.ProviderState.AVAILABLE,
        provider_id="zecalibrator",
        api_version="1.1",
        api_major="1",
        capabilities=("session_library", "auto_route"),
    )
    assert avail.available is True

    missing = port.ProviderInfo(
        state=port.ProviderState.NOT_INSTALLED, message="no module"
    )
    assert missing.available is False


# ---------------------------------------------------------------------------
# The port has no ZeCalibrator dependency: a fake provider satisfies it.
# ---------------------------------------------------------------------------
class _FakeSession:
    fingerprint = "fp-123"

    def resolve_light(self, source, *, cancel=None):
        return port.RouteResolution(
            state=port.CalibrationState.COMPLETED,
            outcome="MATCHED",
            plan=port.CalibrationPlan("plan-1", object()),
        )

    def calibrate(self, source, plan, *, cancel=None):
        return port.CalibrationResult(
            state=port.CalibrationState.COMPLETED, data="float32-data", mask="dq-mask"
        )

    def close(self):
        self.closed = True


class _FakeProvider:
    name = "fake"

    def probe(self):
        return port.ProviderInfo(
            state=port.ProviderState.AVAILABLE,
            provider_id="fake",
            api_version="1.1",
            api_major="1",
            capabilities=("session_library", "auto_route", "calibrate_frame", "cancel"),
        )

    def open_session(self, root, *, cancel=None):
        return port.SessionResult(
            state=port.CalibrationState.COMPLETED,
            session=_FakeSession(),
            fingerprint="fp-123",
            admissions=(port.MasterAdmission("dark", "dark.fits", "a" * 64, 123),),
            rejected=(),
            counts_by_role={"dark": 1},
        )


def test_port_runs_against_fake_provider_without_zecalibrator():
    # The whole port contract is exercisable with a fake provider: no
    # zecalibrator import is required anywhere in this test.
    assert "zecalibrator" not in sys.modules

    provider = _FakeProvider()
    info = provider.probe()
    assert info.available is True

    result = provider.open_session("/some/folder")
    assert result.state is port.CalibrationState.COMPLETED
    assert result.session is not None
    assert result.session.fingerprint == "fp-123"
    assert result.counts_by_role == {"dark": 1}
    assert result.admissions[0].role == "dark"

    light = port.LightSource(path="light.fits")
    rr = result.session.resolve_light(light)
    assert rr.state is port.CalibrationState.COMPLETED
    assert rr.outcome == "MATCHED"
    assert rr.plan.plan_id == "plan-1"

    cr = result.session.calibrate(light, rr.plan)
    assert cr.state is port.CalibrationState.COMPLETED
    assert cr.data == "float32-data"
    assert cr.mask == "dq-mask"

    result.session.close()
    assert getattr(result.session, "closed", False) is True


# ---------------------------------------------------------------------------
# Import hygiene — measured in a separate process.
# ---------------------------------------------------------------------------
def test_port_source_is_transport_neutral_static():
    src = (ROOT / PORT_RELPATH).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    # The port's own imports must stay stdlib-only (``__future__`` has module=None).
    assert imported <= {"__future__", "dataclasses", "enum", "typing"}, sorted(imported)
    for name in imported:
        assert not name.startswith(("numpy", "astropy", "zecalibrator", "seestar")), name


def test_port_import_pulls_no_heavy_deps_subprocess():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('p', {str(ROOT / PORT_RELPATH)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['p'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "for m in ('zecalibrator', 'numpy', 'astropy', 'PySide6', 'tkinter'):\n"
        "    assert m not in sys.modules, m\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
