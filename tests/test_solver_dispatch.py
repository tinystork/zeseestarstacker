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
