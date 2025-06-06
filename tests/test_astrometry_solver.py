import subprocess
from pathlib import Path
import sys
import importlib.util
import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "astrometry_solver",
    ROOT / "seestar" / "alignment" / "astrometry_solver.py",
)
astrometry_solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(astrometry_solver)
AstrometrySolver = astrometry_solver.AstrometrySolver


def test_astap_command_uses_gui_parameters(tmp_path, monkeypatch):
    img = np.ones((10, 10), dtype=np.float32)
    fits_path = tmp_path / "img.fits"
    fits.writeto(fits_path, img, overwrite=True)

    solver = AstrometrySolver()

    captured = {}

    def fake_run(cmd, capture_output, text, timeout, check, cwd):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 1, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    header = fits.getheader(fits_path)

    solver._try_solve_astap(
        str(fits_path),
        header,
        "astap.exe",
        str(tmp_path),
        3.0,
        None,
        None,
        5,
        False,
        astap_downsample=4,
        astap_sensitivity=55,
    )

    assert "cmd" in captured
    cmd = captured["cmd"]
    assert "-z" in cmd and cmd[cmd.index("-z") + 1] == "4"
    assert "-sens" in cmd and cmd[cmd.index("-sens") + 1] == "55"
