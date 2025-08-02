import importlib.util
import sys
import types
from pathlib import Path

from astropy.io import fits


def _load_sanitize(monkeypatch):
    if "seestar.gui" not in sys.modules:
        gui_pkg = types.ModuleType("seestar.gui")
        settings_mod = types.ModuleType("seestar.gui.settings")
        monkeypatch.setitem(sys.modules, "seestar.gui", gui_pkg)
        monkeypatch.setitem(sys.modules, "seestar.gui.settings", settings_mod)

    spec = importlib.util.spec_from_file_location(
        "seestar.gui.boring_stack",
        Path(__file__).resolve().parents[1] / "seestar" / "gui" / "boring_stack.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._sanitize_astap_wcs


def test_sanitize_astap_wcs_text_header(monkeypatch, tmp_path):
    """Ensure non-padded ASTAP ``.wcs`` headers are parsed correctly."""

    _sanitize_astap_wcs = _load_sanitize(monkeypatch)

    from astropy.io.fits import Card

    cards = [
        Card("SIMPLE", True).image,
        "END".ljust(80),
    ]

    wcs_path = tmp_path / "sample.wcs"
    with open(wcs_path, "w", newline="\n") as f:
        # intentionally omit 2880-byte padding
        f.write("\n".join(cards))

    _sanitize_astap_wcs(str(wcs_path))

    hdr = fits.Header.fromfile(wcs_path, sep="\n", padding=False, endcard=False)
    assert hdr["SIMPLE"] is True
