from zemosaic.solver_settings import SolverSettings
import tempfile
import json
import pathlib


def test_roundtrip():
    tmp = pathlib.Path(tempfile.gettempdir()) / "solver.json"
    s0 = SolverSettings(solver_choice="ASTROMETRY", api_key="ABC", timeout=42, downsample=3, force_lum=True)
    s0.save(tmp)
    s1 = SolverSettings.load(tmp)
    assert s0 == s1
    tmp.unlink(missing_ok=True)
