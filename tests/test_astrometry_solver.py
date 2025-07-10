import subprocess
from pathlib import Path
import sys
import importlib.util
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
    assert "-r" in cmd and cmd[cmd.index("-r") + 1] == "3.00"


def test_parse_wcs_with_nonstandard_keywords(tmp_path):
    solver = AstrometrySolver()

    w = WCS(naxis=2)
    w.wcs.crpix = [5, 5]
    w.wcs.cdelt = [-0.0001, 0.0001]
    w.wcs.crval = [10.0, 20.0]
    w.wcs.ctype = ["RA---TPV", "DEC--TPV"]

    hdr = w.to_header(relax=True)
    hdr["FOO"] = "BAR"  # mot-clé non standard

    wcs_path = tmp_path / "test.wcs"
    with open(wcs_path, "w") as f:
        f.write(hdr.tostring(sep="\n"))

    parsed = solver._parse_wcs_file_content(str(wcs_path), (10, 10))

    assert parsed is not None
    assert parsed.is_celestial
    assert parsed.pixel_shape == (10, 10)


def test_parse_astap_wcs_file(tmp_path):
    solver = AstrometrySolver()

    wcs_header = """
NAXIS   =                    2
NAXIS1  =                   10
NAXIS2  =                   10
CRPIX1  =                    5
CRPIX2  =                    5
CD1_1   =        -1.0E-04
CD1_2   =         0.0E0
CD2_1   =         0.0E0
CD2_2   =         1.0E-04
CRVAL1  =                10.0
CRVAL2  =               -10.0
CTYPE1  = 'RA---TAN'
CTYPE2  = 'DEC--TAN'
"""

    wcs_path = tmp_path / "sample_astap.wcs"
    wcs_path.write_text(wcs_header.strip() + "\n")

    parsed = solver._parse_wcs_file_content(str(wcs_path), (10, 10))

    assert parsed is not None
    assert parsed.is_celestial
    assert parsed.pixel_shape == (10, 10)


def test_default_radius_used_when_missing(tmp_path, monkeypatch):
    """solve() should fall back to ASTAP_DEFAULT_SEARCH_RADIUS."""
    img = np.ones((10, 10), dtype=np.float32)
    fits_path = tmp_path / "img.fits"
    fits.writeto(fits_path, img, overwrite=True)

    solver = AstrometrySolver()

    captured = {}

    def fake_try_solve_astap(image, header, exe, data, radius, *args, **kw):
        captured["radius"] = radius
        return None

    monkeypatch.setattr(solver, "_try_solve_astap", fake_try_solve_astap)

    dummy_exe = tmp_path / "astap.exe"
    dummy_exe.write_text("")

    settings = {
        "local_solver_preference": "astap",
        "astap_path": str(dummy_exe),
    }

    header = fits.getheader(fits_path)
    solver.solve(str(fits_path), header, settings)

    assert "radius" in captured
    assert captured["radius"] == astrometry_solver.ASTAP_DEFAULT_SEARCH_RADIUS


def test_use_radec_hints_toggle(tmp_path, monkeypatch):
    img = np.ones((5, 5), dtype=np.float32)
    hdr = fits.Header()
    hdr["RA"] = 123.4
    hdr["DEC"] = -50.1
    fits_path = tmp_path / "hint.fits"
    fits.writeto(fits_path, img, hdr, overwrite=True)

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
        1.0,
        None,
        None,
        5,
        False,
        use_radec_hints=False,
    )
    assert "-ra" not in captured["cmd"]

    solver._try_solve_astap(
        str(fits_path),
        header,
        "astap.exe",
        str(tmp_path),
        1.0,
        None,
        None,
        5,
        False,
        use_radec_hints=True,
    )
    assert "-ra" in captured["cmd"]


def test_astap_uses_pxscale_from_header(tmp_path, monkeypatch):
    img = np.ones((10, 10), dtype=np.float32)
    hdr = fits.Header()
    hdr["XPIXSZ"] = 2.4
    hdr["FOCALLEN"] = 240.0
    fits_path = tmp_path / "px.fits"
    fits.writeto(fits_path, img, hdr, overwrite=True)

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
        1.0,
        None,
        None,
        5,
        False,
    )

    assert "-pxscale" in captured["cmd"]
    idx = captured["cmd"].index("-pxscale") + 1
    val = float(captured["cmd"][idx])
    assert abs(val - ((hdr["XPIXSZ"] / hdr["FOCALLEN"]) * 206.265)) < 0.01


def test_astap_adds_fov_when_scale_missing(tmp_path, monkeypatch):
    img = np.ones((10, 10), dtype=np.float32)
    fits_path = tmp_path / "nofov.fits"
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
        2.0,
        None,
        None,
        5,
        False,
    )

    assert "-fov" in captured["cmd"]


def test_resolve_astap_app(monkeypatch, tmp_path):
    app_dir = tmp_path / "ASTAP.app"
    bin_dir = app_dir / "Contents" / "MacOS"
    bin_dir.mkdir(parents=True)
    exe = bin_dir / "astap"
    exe.write_text("")

    monkeypatch.setattr(astrometry_solver.platform, "system", lambda: "Darwin")

    resolved = astrometry_solver.resolve_astap_executable(str(app_dir))
    assert resolved == str(exe)
