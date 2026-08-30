"""Dispatch tests for the reworked ``AstrometrySolver.solve()``.

These tests load ``astrometry_solver.py`` directly by file path (mirroring
``tests/test_astrometry_solver.py``) so they do not pull in the heavy
``seestar`` package tree beyond what the solver itself needs.  They inject a
fake ZeSolver adapter by monkeypatching the ``_zesolver_adapter_class`` class
attribute and a fake ASTAP solver by monkeypatching the ``_try_solve_astap``
instance method, then assert the ZeSolver -> ASTAP dispatch (including the
SKIPPED-existing-WCS and legacy-preference migration paths).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "astrometry_solver",
    ROOT / "seestar" / "alignment" / "astrometry_solver.py",
)
astrometry_solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(astrometry_solver)
AstrometrySolver = astrometry_solver.AstrometrySolver

# The real solver_port module is imported as a side-effect of loading
# astrometry_solver (via the adapter); reuse it to build outcomes.
import seestar.alignment.solver_port as solver_port


def make_celestial_wcs():
    w = WCS(naxis=2)
    w.wcs.crpix = [5.0, 5.0]
    w.wcs.cdelt = [-0.001, 0.001]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def recording_adapter_class(outcome):
    """Return a fake adapter class and a dict tracking its instantiation/use."""
    state = {"created": 0, "solve_calls": 0}

    class FakeAdapter:
        def __init__(self):
            state["created"] += 1

        def solve(self, **kwargs):
            state["solve_calls"] += 1
            return outcome

    return FakeAdapter, state


def solved_outcome(wcs=None):
    return solver_port.SolverOutcome(
        status=solver_port.SolveStatus.SOLVED,
        wcs=wcs if wcs is not None else make_celestial_wcs(),
        should_write_header_back=True,
        backend_used="zesolver",
    )


def make_dummy_astap(tmp_path):
    dummy_exe = tmp_path / "astap"
    dummy_exe.write_text("")
    return str(dummy_exe)


# ---------------------------------------------------------------------------


def test_zesolver_solved_writes_header_only_when_requested(tmp_path, monkeypatch):
    w = make_celestial_wcs()
    fake_cls, state = recording_adapter_class(solved_outcome(w))
    monkeypatch.setattr(AstrometrySolver, "_zesolver_adapter_class", fake_cls)

    solver = AstrometrySolver()
    settings = {"local_solver_preference": "zesolver"}

    # update_header_with_solution=True -> header written back.
    header_true = fits.Header()
    header_true["SIMPLE"] = True
    result = solver.solve(
        "dummy.fits", header_true, settings, update_header_with_solution=True
    )
    assert result is not None and result.is_celestial
    assert state["solve_calls"] == 1
    assert "ZESOLVER_SOLVED" in header_true
    assert header_true["CTYPE1"] == "RA---TAN"

    # update_header_with_solution=False -> no write back.
    header_false = fits.Header()
    header_false["SIMPLE"] = True
    result2 = solver.solve(
        "dummy.fits", header_false, settings, update_header_with_solution=False
    )
    assert result2 is not None and result2.is_celestial
    assert "ZESOLVER_SOLVED" not in header_false
    assert "CTYPE1" not in header_false


def test_zesolver_unavailable_falls_back_to_astap(tmp_path, monkeypatch):
    outcome = solver_port.SolverOutcome(
        status=solver_port.SolveStatus.UNAVAILABLE,
        message="not installed",
        backend_used="zesolver",
    )
    fake_cls, state = recording_adapter_class(outcome)
    monkeypatch.setattr(AstrometrySolver, "_zesolver_adapter_class", fake_cls)

    astap_wcs = make_celestial_wcs()
    solver = AstrometrySolver()
    calls = []

    def fake_astap(*args, **kwargs):
        calls.append(1)
        return astap_wcs

    monkeypatch.setattr(solver, "_try_solve_astap", fake_astap)

    settings = {
        "local_solver_preference": "zesolver",
        "astap_path": make_dummy_astap(tmp_path),
    }
    result = solver.solve("dummy.fits", fits.Header(), settings)

    assert result is astap_wcs
    assert state["solve_calls"] == 1
    assert len(calls) == 1


def test_zesolver_failed_falls_back_to_astap(tmp_path, monkeypatch):
    outcome = solver_port.SolverOutcome(
        status=solver_port.SolveStatus.FAILED,
        failure_code="no_solution",
        backend_used="zesolver",
    )
    fake_cls, state = recording_adapter_class(outcome)
    monkeypatch.setattr(AstrometrySolver, "_zesolver_adapter_class", fake_cls)

    astap_wcs = make_celestial_wcs()
    solver = AstrometrySolver()
    calls = []

    def fake_astap(*args, **kwargs):
        calls.append(1)
        return astap_wcs

    monkeypatch.setattr(solver, "_try_solve_astap", fake_astap)

    settings = {
        "local_solver_preference": "zesolver",
        "astap_path": make_dummy_astap(tmp_path),
    }
    result = solver.solve("dummy.fits", fits.Header(), settings)

    assert result is astap_wcs
    assert state["solve_calls"] == 1
    assert len(calls) == 1


def test_zesolver_skipped_with_existing_wcs_no_astap(tmp_path, monkeypatch):
    existing = make_celestial_wcs()
    header = existing.to_header()
    outcome = solver_port.SolverOutcome(
        status=solver_port.SolveStatus.SKIPPED,
        failure_code="skipped_existing_wcs",
        backend_used="zesolver",
    )
    fake_cls, state = recording_adapter_class(outcome)
    monkeypatch.setattr(AstrometrySolver, "_zesolver_adapter_class", fake_cls)

    solver = AstrometrySolver()
    astap_calls = []
    monkeypatch.setattr(
        solver,
        "_try_solve_astap",
        lambda *a, **k: astap_calls.append(1) or None,
    )

    settings = {
        "local_solver_preference": "zesolver",
        "astap_path": make_dummy_astap(tmp_path),
    }
    result = solver.solve("dummy.fits", header, settings)

    assert result is not None and result.is_celestial
    assert state["solve_calls"] == 1
    assert astap_calls == []


def test_zesolver_skipped_without_wcs_falls_back_to_astap(tmp_path, monkeypatch):
    outcome = solver_port.SolverOutcome(
        status=solver_port.SolveStatus.SKIPPED,
        failure_code="skipped_existing_wcs",
        backend_used="zesolver",
    )
    fake_cls, state = recording_adapter_class(outcome)
    monkeypatch.setattr(AstrometrySolver, "_zesolver_adapter_class", fake_cls)

    astap_wcs = make_celestial_wcs()
    solver = AstrometrySolver()
    calls = []

    def fake_astap(*args, **kwargs):
        calls.append(1)
        return astap_wcs

    monkeypatch.setattr(solver, "_try_solve_astap", fake_astap)

    settings = {
        "local_solver_preference": "zesolver",
        "astap_path": make_dummy_astap(tmp_path),
    }
    result = solver.solve("dummy.fits", fits.Header(), settings)

    assert result is astap_wcs
    assert state["solve_calls"] == 1
    assert len(calls) == 1


def test_astap_only_never_creates_adapter(tmp_path, monkeypatch):
    fake_cls, state = recording_adapter_class(None)
    monkeypatch.setattr(AstrometrySolver, "_zesolver_adapter_class", fake_cls)

    w = make_celestial_wcs()
    solver = AstrometrySolver()
    monkeypatch.setattr(solver, "_try_solve_astap", lambda *a, **k: w)

    settings = {
        "local_solver_preference": "astap",
        "astap_path": make_dummy_astap(tmp_path),
    }
    result = solver.solve("dummy.fits", fits.Header(), settings)

    assert result is w
    assert state["created"] == 0
    assert state["solve_calls"] == 0


def test_none_preference_does_no_solving(tmp_path, monkeypatch):
    fake_cls, state = recording_adapter_class(None)
    monkeypatch.setattr(AstrometrySolver, "_zesolver_adapter_class", fake_cls)

    solver = AstrometrySolver()
    astap_calls = []
    monkeypatch.setattr(
        solver,
        "_try_solve_astap",
        lambda *a, **k: astap_calls.append(1) or None,
    )

    settings = {"local_solver_preference": "none"}
    result = solver.solve("dummy.fits", fits.Header(), settings)

    assert result is None
    assert state["created"] == 0
    assert state["solve_calls"] == 0
    assert astap_calls == []


def test_legacy_preference_migrates_to_zesolver(tmp_path, monkeypatch):
    w = make_celestial_wcs()
    fake_cls, state = recording_adapter_class(solved_outcome(w))
    monkeypatch.setattr(AstrometrySolver, "_zesolver_adapter_class", fake_cls)
    monkeypatch.setattr(astrometry_solver, "_legacy_preference_warned", False)

    messages = []
    solver = AstrometrySolver(progress_callback=lambda m, p: messages.append(m))

    for pref in ("ansvr", "astrometry"):
        header = fits.Header()
        settings = {"local_solver_preference": pref}
        result = solver.solve("dummy.fits", header, settings)
        assert result is not None and result.is_celestial

    assert state["solve_calls"] == 2
    assert any("migrée" in m for m in messages)


def test_no_top_level_zesolver_import():
    src = (ROOT / "seestar" / "alignment" / "astrometry_solver.py").read_text(
        encoding="utf-8"
    )
    assert "import zesolver" not in src
    assert "from zesolver" not in src



# ---------------------------------------------------------------------------
# ZESOLVER-WCS-CANONICAL: full AstrometrySolver chain under threading
# ---------------------------------------------------------------------------


def _naxis3_celestial_cards():
    """FITS cards for a solved RGB cube (NAXIS=3, celestial 2D WCS)."""
    h = fits.Header()
    h["SIMPLE"] = True
    h["BITPIX"] = -32
    h["NAXIS"] = 3
    h["NAXIS1"] = 1920
    h["NAXIS2"] = 1080
    h["NAXIS3"] = 3
    h["CTYPE1"] = "RA---TAN"
    h["CTYPE2"] = "DEC--TAN"
    h["CRVAL1"] = 275.037495
    h["CRVAL2"] = -13.730556
    h["CRPIX1"] = 960.5
    h["CRPIX2"] = 540.5
    h["CD1_1"] = -6.6666667e-05
    h["CD1_2"] = 0.0
    h["CD2_1"] = 0.0
    h["CD2_2"] = 6.6666667e-05
    return tuple(c.image for c in h.cards)


def _install_fake_v1_rgb(monkeypatch, cards):
    """Install a minimal fake zesolver.api.v1 returning a NAXIS=3 SOLVED."""
    import enum
    import types

    class _E(enum.Enum):
        pass

    class NetworkPolicy(_E):
        DISABLED = "disabled"

    class GpuPolicy(_E):
        AUTO = "auto"
        DISABLED = "disabled"
        REQUIRED = "required"

    class BackendPolicy(_E):
        AUTO = "auto"
        NEAR_ONLY = "near_only"
        BLIND_ONLY = "blind_only"

    class WritePolicy(_E):
        OVERWRITE_INPUT = "overwrite_input"
        WRITE_NONE = "write_none"

    class SolveStatus(_E):
        SOLVED = "solved"
        FAILED = "failed"
        SKIPPED_EXISTING_WCS = "skipped_existing_wcs"
        CANCELLED = "cancelled"

    class FailureCode(_E):
        NO_SOLUTION = "no_solution"

    class SolveHints:
        def __init__(self, **kw):
            pass

    class SolveOptions:
        def __init__(self, **kw):
            pass

    class SolveRequest:
        def __init__(self, input_path, hints=None, options=None):
            pass

    class CancellationToken:
        def __init__(self):
            pass

        def cancel(self):
            pass

    class CanonicalWcsHeader:
        def __init__(self, cards):
            self.format = "fits-header-cards-v1"
            self.cards = cards

    class SolveResult:
        def __init__(self, status, wcs_header=None, failure_code=None, message=None):
            self.status = status
            self.wcs_header = wcs_header
            self.failure_code = failure_code
            self.message = message

    class _Session:
        def solve(self, request, cancellation=None, progress=None):
            return SolveResult(SolveStatus.SOLVED, wcs_header=CanonicalWcsHeader(cards))

        def close(self):
            pass

    class _Runtime:
        def create_session(self):
            return _Session()

        def close(self):
            pass

    v1 = types.ModuleType("zesolver.api.v1")
    v1.API_VERSION = "1.2"
    v1.API_MAJOR = 1
    v1.get_api_info = lambda: types.SimpleNamespace(
        product_version="1.2.1",
        supported_capabilities=("near_solve", "blind_solve", "wcs_write", "gpu", "cancel"),
    )
    v1.probe = lambda **kw: None
    v1.SolveHints = SolveHints
    v1.SolveOptions = SolveOptions
    v1.SolveRequest = SolveRequest
    v1.CanonicalWcsHeader = CanonicalWcsHeader
    v1.SolveResult = SolveResult
    v1.SolveStatus = SolveStatus
    v1.FailureCode = FailureCode
    v1.NetworkPolicy = NetworkPolicy
    v1.GpuPolicy = GpuPolicy
    v1.BackendPolicy = BackendPolicy
    v1.WritePolicy = WritePolicy
    v1.CancellationToken = CancellationToken
    v1.create_solver_runtime = lambda **kw: _Runtime()

    for name in ("zesolver", "zesolver.api"):
        mod = types.ModuleType(name)
        mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setitem(sys.modules, "zesolver.api.v1", v1)


def test_zesolver_rgb_cube_threaded_witness(monkeypatch):
    import concurrent.futures

    _install_fake_v1_rgb(monkeypatch, _naxis3_celestial_cards())
    settings = {"local_solver_preference": "zesolver"}

    def _solve():
        msgs = []
        hdr = fits.Header()
        hdr["NAXIS"] = 3
        wcs = AstrometrySolver(progress_callback=lambda m, p: msgs.append(m)).solve(
            "/tmp/rgb.fits", hdr, settings, update_header_with_solution=False
        )
        return wcs, msgs

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        results = [f.result() for f in [ex.submit(_solve) for _ in range(2)]]

    for wcs, msgs in results:
        assert wcs is not None
        assert wcs.is_celestial
        joined = "\n".join(msgs)
        assert "SOLVED mais WCS absent/non" not in joined
        assert "Fallback ASTAP" not in joined
