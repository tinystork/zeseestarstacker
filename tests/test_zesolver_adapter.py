"""Focused tests for the optional ZeSolver API v1 adapter.

These tests never require a real ZeSolver installation: they inject a fake
``zesolver.api.v1`` module into ``sys.modules`` via monkeypatch and exercise the
discovery, request mapping, result conversion, lifecycle and failure handling
logic.

Both modules are loaded directly by file path (mirroring the pattern used by
``tests/test_solver_config.py``) so the tests do not pull in the heavy ``seestar``
package tree, which requires optional dependencies (OpenCV, etc.) that are absent
from this test environment.
"""

from __future__ import annotations

import enum
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# -- load solver_port + adapter by file path under their real package names ----

def _install_package_stub(name: str) -> None:
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod


# Snapshot the modules present before we touch ``sys.modules``. This test module
# must stay hermetic: leaking a hollow ``seestar``/``seestar.alignment`` stub
# into ``sys.modules`` would shadow the real package for any test collected
# after this one, making the whole suite order-dependent.
_PREEXISTING_MODULE_KEYS = set(sys.modules.keys())


def _load_by_path(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ``solver_port`` has no intra-package imports, but the adapter performs
# ``from .solver_port import ...``, so it needs ``seestar.alignment`` to be a
# resolvable package while it executes. We install a temporary stub for exactly
# that purpose, then remove it (and every other module we introduced) straight
# afterwards so no stub outlives this module's import.
if "seestar.alignment" not in sys.modules:
    _install_package_stub("seestar.alignment")

solver_port = _load_by_path(
    "seestar.alignment.solver_port", "seestar/alignment/solver_port.py"
)
adapter = _load_by_path(
    "seestar.alignment.zesolver_adapter", "seestar/alignment/zesolver_adapter.py"
)

# Restore ``sys.modules`` to its pre-import state.
for _key in list(sys.modules.keys()):
    if _key not in _PREEXISTING_MODULE_KEYS:
        del sys.modules[_key]

DiscoveryState = solver_port.DiscoveryState
SolveStatus = solver_port.SolveStatus


# ---------------------------------------------------------------------------
# Fake ``zesolver.api.v1`` helpers
# ---------------------------------------------------------------------------


class _FakeNetworkPolicy(enum.Enum):
    DISABLED = "disabled"
    ALLOWED = "allowed"


class _FakeGpuPolicy(enum.Enum):
    AUTO = "auto"
    DISABLED = "disabled"
    REQUIRED = "required"


class _FakeBackendPolicy(enum.Enum):
    AUTO = "auto"
    NEAR_ONLY = "near_only"
    BLIND_ONLY = "blind_only"


class _FakeWritePolicy(enum.Enum):
    OVERWRITE_INPUT = "overwrite_input"


class _FakeWritePolicyRich(enum.Enum):
    OVERWRITE_INPUT = "overwrite_input"
    WRITE_NONE = "write_none"
    READ_ONLY = "read_only"


class _FakeSolveStatus(enum.Enum):
    SOLVED = "solved"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class _FakeFailureCode(enum.Enum):
    NO_SOLUTION = "no_solution"
    SKIPPED_EXISTING_WCS = "skipped_existing_wcs"


class _FakeAvailability(enum.Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_CHECKED = "not_checked"


class _FakeCancellationToken:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


VALID_WCS_CARDS = (
    "CRVAL1  =                 10.0",
    "CRVAL2  =                 20.0",
    "CRPIX1  =                512.0",
    "CRPIX2  =                384.0",
    "CDELT1  =  -0.000277777777778",
    "CDELT2  =   0.000277777777778",
    "CTYPE1  = 'RA---TAN'",
    "CTYPE2  = 'DEC--TAN'",
)


def _install_package_stubs(monkeypatch, v1: types.ModuleType) -> None:
    pkg = types.ModuleType("zesolver")
    pkg.__path__ = []
    api = types.ModuleType("zesolver.api")
    api.__path__ = []
    monkeypatch.setitem(sys.modules, "zesolver", pkg)
    monkeypatch.setitem(sys.modules, "zesolver.api", api)
    monkeypatch.setitem(sys.modules, "zesolver.api.v1", v1)


def _remove_zesolver(monkeypatch) -> None:
    for key in list(sys.modules):
        if key == "zesolver" or key.startswith("zesolver."):
            monkeypatch.delitem(sys.modules, key, raising=False)


def _make_v1(
    *,
    api_version="1.0",
    api_major=1,
    probe=None,
    product_version="1.2.3",
    supported_capabilities=("near_solve", "blind_solve", "wcs_write", "gpu", "cancel"),
):
    v1 = types.ModuleType("zesolver.api.v1")
    v1.API_VERSION = api_version
    v1.API_MAJOR = api_major
    v1.probe = probe if probe is not None else (lambda **kw: None)

    def get_api_info():
        return types.SimpleNamespace(
            product_version=product_version,
            supported_capabilities=supported_capabilities,
        )

    v1.get_api_info = get_api_info
    return v1


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discover_not_installed(monkeypatch):
    _remove_zesolver(monkeypatch)
    discovery = adapter.discover_zesolver()
    assert discovery.state is DiscoveryState.NOT_INSTALLED


def test_discover_unhealthy_when_internal_import_fails(monkeypatch):
    _remove_zesolver(monkeypatch)

    def boom(name):
        raise ModuleNotFoundError("missing internal dep", name="internal_dep")

    monkeypatch.setattr(adapter.importlib, "import_module", boom)
    discovery = adapter.discover_zesolver()
    assert discovery.state is DiscoveryState.UNHEALTHY


def test_discover_unhealthy_when_import_raises_other_error(monkeypatch):
    _remove_zesolver(monkeypatch)

    def boom(name):
        raise RuntimeError("boom")

    monkeypatch.setattr(adapter.importlib, "import_module", boom)
    discovery = adapter.discover_zesolver()
    assert discovery.state is DiscoveryState.UNHEALTHY


def test_discover_incompatible_v2(monkeypatch):
    v1 = _make_v1(api_version="2.0", api_major=2)
    _install_package_stubs(monkeypatch, v1)
    discovery = adapter.discover_zesolver()
    assert discovery.state is DiscoveryState.INCOMPATIBLE


def test_discover_incompatible_major_parsed_from_version(monkeypatch):
    v1 = _make_v1(api_version="3.1")
    del v1.API_MAJOR
    _install_package_stubs(monkeypatch, v1)
    discovery = adapter.discover_zesolver()
    assert discovery.state is DiscoveryState.INCOMPATIBLE


def test_discover_incompatible_missing_wcs_write(monkeypatch):
    v1 = _make_v1(supported_capabilities=("near_solve", "blind_solve"))
    _install_package_stubs(monkeypatch, v1)
    discovery = adapter.discover_zesolver()
    assert discovery.state is DiscoveryState.INCOMPATIBLE


def test_discover_incompatible_no_solve_backend(monkeypatch):
    v1 = _make_v1(supported_capabilities=("wcs_write", "gpu", "cancel"))
    _install_package_stubs(monkeypatch, v1)
    discovery = adapter.discover_zesolver()
    assert discovery.state is DiscoveryState.INCOMPATIBLE


def test_discover_unhealthy_when_probe_raises(monkeypatch):
    def probe(**kwargs):
        raise RuntimeError("boom")

    v1 = _make_v1(probe=probe)
    _install_package_stubs(monkeypatch, v1)
    discovery = adapter.discover_zesolver()
    assert discovery.state is DiscoveryState.UNHEALTHY


def test_discover_unhealthy_required_capability_unavailable(monkeypatch):
    def probe(**kwargs):
        return types.SimpleNamespace(
            capabilities=[
                types.SimpleNamespace(
                    id="wcs_write", availability=_FakeAvailability.UNAVAILABLE
                ),
            ]
        )

    v1 = _make_v1(probe=probe)
    _install_package_stubs(monkeypatch, v1)
    discovery = adapter.discover_zesolver()
    assert discovery.state is DiscoveryState.UNHEALTHY


def test_discover_available(monkeypatch):
    v1 = _make_v1()
    _install_package_stubs(monkeypatch, v1)
    discovery = adapter.discover_zesolver()
    assert discovery.state is DiscoveryState.AVAILABLE
    assert discovery.api_version == "1.0"
    assert discovery.product_version == "1.2.3"


# ---------------------------------------------------------------------------
# Adapter request mapping + result conversion (full fake v1)
# ---------------------------------------------------------------------------


def _make_full_v1():
    rec = types.SimpleNamespace(
        solve_options_kwargs=None,
        solve_hints_kwargs=None,
        solve_request_input=None,
        create_runtime_kwargs=None,
        create_runtime_calls=0,
        create_session_calls=0,
        runtime_close_calls=0,
        session_close_calls=0,
        solve_calls=0,
        result=None,
        solve_exc=None,
        last_cancellation=None,
        last_progress=None,
        runtime=None,
    )

    class FakeSolveHints:
        def __init__(self, **kwargs):
            rec.solve_hints_kwargs = kwargs

    class FakeSolveOptions:
        def __init__(self, **kwargs):
            rec.solve_options_kwargs = kwargs

    class FakeSolveRequest:
        def __init__(self, input_path, hints=None, options=None):
            rec.solve_request_input = input_path
            rec.solve_request_hints = hints
            rec.solve_request_options = options

    class FakeCanonicalWcsHeader:
        def __init__(self, cards):
            self.format = "fits-header-cards-v1"
            self.cards = cards

    class FakeSolveResult:
        def __init__(self, status, wcs_header=None, failure_code=None, message=None):
            self.status = status
            self.wcs_header = wcs_header
            self.failure_code = failure_code
            self.message = message

    class FakeSession:
        def solve(self, request, cancellation=None, progress=None):
            rec.solve_calls += 1
            rec.last_cancellation = cancellation
            rec.last_progress = progress
            if rec.solve_exc is not None:
                raise rec.solve_exc
            return rec.result

        def close(self):
            rec.session_close_calls += 1

    class FakeRuntime:
        def __init__(self):
            self._session = FakeSession()

        def create_session(self):
            rec.create_session_calls += 1
            return self._session

        def close(self):
            rec.runtime_close_calls += 1

    def create_solver_runtime(**kwargs):
        rec.create_runtime_calls += 1
        rec.create_runtime_kwargs = kwargs
        runtime = FakeRuntime()
        rec.runtime = runtime
        return runtime

    v1 = types.ModuleType("zesolver.api.v1")
    v1.API_VERSION = "1.0"
    v1.API_MAJOR = 1
    v1.get_api_info = lambda: types.SimpleNamespace(
        product_version="1.2.3",
        supported_capabilities=("near_solve", "blind_solve", "wcs_write", "gpu", "cancel"),
    )
    v1.probe = lambda **kw: None
    v1.SolveHints = FakeSolveHints
    v1.SolveOptions = FakeSolveOptions
    v1.SolveRequest = FakeSolveRequest
    v1.CanonicalWcsHeader = FakeCanonicalWcsHeader
    v1.SolveResult = FakeSolveResult
    v1.SolveStatus = _FakeSolveStatus
    v1.FailureCode = _FakeFailureCode
    v1.NetworkPolicy = _FakeNetworkPolicy
    v1.GpuPolicy = _FakeGpuPolicy
    v1.BackendPolicy = _FakeBackendPolicy
    v1.WritePolicy = _FakeWritePolicy
    v1.CancellationToken = _FakeCancellationToken
    v1.create_solver_runtime = create_solver_runtime
    return v1, rec


def test_adapter_solve_solved_with_wcs(monkeypatch):
    v1, rec = _make_full_v1()
    rec.result = v1.SolveResult(
        _FakeSolveStatus.SOLVED, wcs_header=v1.CanonicalWcsHeader(VALID_WCS_CARDS)
    )
    _install_package_stubs(monkeypatch, v1)

    outcome = adapter.ZeSolverAdapter().solve(
        image_fits_path="/tmp/img.fits",
        fits_header={"NAXIS1": 100},
        settings={"scale_est_arcsec_per_pix": 1.2},
        progress_callback=None,
    )

    assert outcome.status is SolveStatus.SOLVED
    assert outcome.wcs is not None
    assert list(outcome.wcs.wcs.ctype) == ["RA---TAN", "DEC--TAN"]
    assert outcome.header is not None
    assert outcome.should_write_header_back is True
    assert outcome.backend_used == "zesolver"

    # Request mapping.
    assert rec.solve_request_input == Path("/tmp/img.fits")
    assert rec.solve_hints_kwargs == {"pixel_scale_arcsec": 1.2}
    assert rec.solve_options_kwargs["network_policy"] is _FakeNetworkPolicy.DISABLED
    assert rec.solve_options_kwargs["write_policy"] is _FakeWritePolicy.OVERWRITE_INPUT
    # Timeout default.
    assert rec.solve_options_kwargs["timeout_s"] == 120.0


def test_adapter_solve_maps_radec_hints_and_timeout(monkeypatch):
    v1, rec = _make_full_v1()
    rec.result = v1.SolveResult(
        _FakeSolveStatus.SOLVED, wcs_header=v1.CanonicalWcsHeader(VALID_WCS_CARDS)
    )
    _install_package_stubs(monkeypatch, v1)

    adapter.ZeSolverAdapter().solve(
        image_fits_path="/tmp/img.fits",
        fits_header={"RA": 12.5, "DEC": 34.5},
        settings={"use_radec_hints": True, "astap_timeout_sec": 300},
        progress_callback=None,
    )

    assert rec.solve_hints_kwargs == {"ra_deg": 12.5, "dec_deg": 34.5}
    assert rec.solve_options_kwargs["timeout_s"] == 300.0


def test_adapter_solve_failed(monkeypatch):
    v1, rec = _make_full_v1()
    rec.result = v1.SolveResult(
        _FakeSolveStatus.FAILED,
        failure_code=_FakeFailureCode.NO_SOLUTION,
        message="no stars",
    )
    _install_package_stubs(monkeypatch, v1)

    outcome = adapter.ZeSolverAdapter().solve(
        image_fits_path="/tmp/img.fits",
        fits_header={},
        settings={},
        progress_callback=None,
    )
    assert outcome.status is SolveStatus.FAILED
    assert outcome.failure_code == "no_solution"
    assert outcome.message == "no stars"
    assert outcome.wcs is None
    assert outcome.should_write_header_back is False


def test_adapter_solve_wcs_conversion_failed(monkeypatch):
    # A canonical header without usable cards cannot be converted to a WCS.
    v1, rec = _make_full_v1()
    rec.result = v1.SolveResult(
        _FakeSolveStatus.SOLVED, wcs_header=v1.CanonicalWcsHeader(None)
    )
    _install_package_stubs(monkeypatch, v1)

    outcome = adapter.ZeSolverAdapter().solve(
        image_fits_path="/tmp/img.fits",
        fits_header={},
        settings={},
        progress_callback=None,
    )
    assert outcome.status is SolveStatus.FAILED
    assert outcome.failure_code == "wcs_conversion_failed"


def test_adapter_solve_unavailable_when_not_installed(monkeypatch):
    _remove_zesolver(monkeypatch)
    outcome = adapter.ZeSolverAdapter().solve(
        image_fits_path="/tmp/img.fits",
        fits_header={},
        settings={},
        progress_callback=None,
    )
    assert outcome.status is SolveStatus.UNAVAILABLE
    assert outcome.failure_code == "not_installed"


def test_adapter_solve_unavailable_when_incompatible(monkeypatch):
    v1 = _make_v1(api_version="2.0", api_major=2)
    _install_package_stubs(monkeypatch, v1)
    outcome = adapter.ZeSolverAdapter().solve(
        image_fits_path="/tmp/img.fits",
        fits_header={},
        settings={},
        progress_callback=None,
    )
    assert outcome.status is SolveStatus.UNAVAILABLE
    assert outcome.failure_code == "incompatible"


def test_adapter_unexpected_exception_returns_failed(monkeypatch):
    v1, rec = _make_full_v1()
    rec.solve_exc = RuntimeError("engine exploded")
    _install_package_stubs(monkeypatch, v1)

    outcome = adapter.ZeSolverAdapter().solve(
        image_fits_path="/tmp/img.fits",
        fits_header={},
        settings={},
        progress_callback=None,
    )
    assert outcome.status is SolveStatus.FAILED
    assert outcome.failure_code == "unexpected_error"
    assert "engine exploded" in outcome.message


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_adapter_runtime_and_session_shared_per_thread(monkeypatch):
    v1, rec = _make_full_v1()
    rec.result = v1.SolveResult(
        _FakeSolveStatus.SOLVED, wcs_header=v1.CanonicalWcsHeader(VALID_WCS_CARDS)
    )
    _install_package_stubs(monkeypatch, v1)

    a = adapter.ZeSolverAdapter(gpu_policy="disabled")
    for _ in range(3):
        a.solve(
            image_fits_path="/tmp/img.fits",
            fits_header={},
            settings={},
            progress_callback=None,
        )

    # One runtime per adapter instance, one session per thread.
    assert rec.create_runtime_calls == 1
    assert rec.create_runtime_kwargs["gpu_policy"] is _FakeGpuPolicy.DISABLED
    assert rec.create_session_calls == 1
    assert rec.solve_calls == 3


def test_adapter_close_is_idempotent_and_recreates(monkeypatch):
    v1, rec = _make_full_v1()
    rec.result = v1.SolveResult(
        _FakeSolveStatus.SOLVED, wcs_header=v1.CanonicalWcsHeader(VALID_WCS_CARDS)
    )
    _install_package_stubs(monkeypatch, v1)

    a = adapter.ZeSolverAdapter()
    a.solve(
        image_fits_path="/tmp/a.fits",
        fits_header={},
        settings={},
        progress_callback=None,
    )
    assert rec.create_runtime_calls == 1
    assert rec.create_session_calls == 1
    assert rec.runtime_close_calls == 0

    a.close()
    assert rec.runtime_close_calls == 1
    assert rec.session_close_calls == 1

    # Idempotent: second close is a no-op on the already-closed runtime.
    a.close()
    assert rec.runtime_close_calls == 1
    assert rec.session_close_calls == 1

    # Subsequent solve creates a fresh runtime + session.
    a.solve(
        image_fits_path="/tmp/b.fits",
        fits_header={},
        settings={},
        progress_callback=None,
    )
    assert rec.create_runtime_calls == 2
    assert rec.create_session_calls == 2
    assert rec.runtime_close_calls == 1


def test_adapter_close_without_runtime_is_noop(monkeypatch):
    v1, rec = _make_full_v1()
    _install_package_stubs(monkeypatch, v1)

    a = adapter.ZeSolverAdapter()
    a.close()
    a.close()
    assert rec.create_runtime_calls == 0
    assert rec.runtime_close_calls == 0
    assert rec.session_close_calls == 0


def test_adapter_cancel_active_solve_best_effort(monkeypatch):
    v1, rec = _make_full_v1()
    _install_package_stubs(monkeypatch, v1)

    a = adapter.ZeSolverAdapter()
    # No active solve -> cancel is a no-op, must not raise.
    a.cancel_active_solve()

    # With an active token registered, cancel must fire (best-effort).
    token = _FakeCancellationToken()
    a._register_active_token(token)
    a.cancel_active_solve()
    assert token.cancelled is True


# ---------------------------------------------------------------------------
# Public-import-only contract (static source inspection)
# ---------------------------------------------------------------------------


def test_sources_have_no_top_level_zesolver_import():
    port_src = (ROOT / "seestar" / "alignment" / "solver_port.py").read_text(
        encoding="utf-8"
    )
    adapter_src = (ROOT / "seestar" / "alignment" / "zesolver_adapter.py").read_text(
        encoding="utf-8"
    )

    assert "import zesolver" not in port_src
    assert "from zesolver" not in port_src

    # Adapter references the public module only as a string constant (never as
    # a top-level import statement).
    assert "import zesolver" not in adapter_src
    assert "from zesolver" not in adapter_src
    assert "zesolver.api.v1" in adapter_src


# ---------------------------------------------------------------------------
# M2a additions: SKIPPED mapping, write-policy selection, timeout source
# ---------------------------------------------------------------------------


def test_adapter_skipped_status_maps_to_skipped(monkeypatch):
    v1, rec = _make_full_v1()
    rec.result = v1.SolveResult(_FakeSolveStatus.SKIPPED, message="already solved")
    _install_package_stubs(monkeypatch, v1)

    outcome = adapter.ZeSolverAdapter().solve(
        image_fits_path="/tmp/img.fits",
        fits_header={},
        settings={},
        progress_callback=None,
    )
    assert outcome.status is SolveStatus.SKIPPED
    assert outcome.failure_code == "skipped_existing_wcs"
    assert outcome.wcs is None
    assert outcome.should_write_header_back is False


def test_adapter_skipped_via_failure_code(monkeypatch):
    v1, rec = _make_full_v1()
    rec.result = v1.SolveResult(
        _FakeSolveStatus.FAILED,
        failure_code=_FakeFailureCode.SKIPPED_EXISTING_WCS,
        message="file already has WCS",
    )
    _install_package_stubs(monkeypatch, v1)

    outcome = adapter.ZeSolverAdapter().solve(
        image_fits_path="/tmp/img.fits",
        fits_header={},
        settings={},
        progress_callback=None,
    )
    assert outcome.status is SolveStatus.SKIPPED
    assert outcome.failure_code == "skipped_existing_wcs"


def test_adapter_skipped_keeps_existing_wcs_header(monkeypatch):
    v1, rec = _make_full_v1()
    rec.result = v1.SolveResult(
        _FakeSolveStatus.SKIPPED,
        wcs_header=v1.CanonicalWcsHeader(VALID_WCS_CARDS),
        message="already solved",
    )
    _install_package_stubs(monkeypatch, v1)

    outcome = adapter.ZeSolverAdapter().solve(
        image_fits_path="/tmp/img.fits",
        fits_header={"NAXIS1": 100},
        settings={},
        progress_callback=None,
    )
    assert outcome.status is SolveStatus.SKIPPED
    assert outcome.wcs is not None
    assert list(outcome.wcs.wcs.ctype) == ["RA---TAN", "DEC--TAN"]
    assert outcome.should_write_header_back is False


def test_write_policy_least_destructive_selection():
    v1 = types.SimpleNamespace(WritePolicy=_FakeWritePolicyRich)
    chosen = adapter.ZeSolverAdapter._select_least_destructive_write_policy(v1)
    assert chosen is _FakeWritePolicyRich.WRITE_NONE


def test_write_policy_falls_back_to_overwrite_input():
    v1 = types.SimpleNamespace(WritePolicy=_FakeWritePolicy)
    chosen = adapter.ZeSolverAdapter._select_least_destructive_write_policy(v1)
    assert chosen is _FakeWritePolicy.OVERWRITE_INPUT


def test_resolve_timeout_only_astap_key():
    assert adapter.ZeSolverAdapter._resolve_timeout({}) == 120.0
    assert adapter.ZeSolverAdapter._resolve_timeout({"astap_timeout_sec": 45}) == 45.0
    # ansvr_timeout_sec must no longer be consulted as a fallback.
    assert adapter.ZeSolverAdapter._resolve_timeout({"ansvr_timeout_sec": 999}) == 120.0
