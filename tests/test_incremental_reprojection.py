import importlib
import sys
from pathlib import Path

import numpy as np
from astropy.wcs import WCS
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if "seestar.gui" not in sys.modules:
    import types

    seestar_pkg = types.ModuleType("seestar")
    seestar_pkg.__path__ = [str(ROOT / "seestar")]
    gui_pkg = types.ModuleType("seestar.gui")
    gui_pkg.__path__ = []
    settings_mod = types.ModuleType("seestar.gui.settings")

    class DummySettingsManager:
        pass

    settings_mod.SettingsManager = DummySettingsManager
    hist_mod = types.ModuleType("seestar.gui.histogram_widget")
    hist_mod.HistogramWidget = object
    gui_pkg.settings = settings_mod
    gui_pkg.histogram_widget = hist_mod
    seestar_pkg.gui = gui_pkg
    sys.modules["seestar"] = seestar_pkg
    sys.modules["seestar.gui"] = gui_pkg
    sys.modules["seestar.gui.settings"] = settings_mod
    sys.modules["seestar.gui.histogram_widget"] = hist_mod

spec = importlib.util.spec_from_file_location(
    "seestar.enhancement.reproject_utils",
    ROOT / "seestar" / "enhancement" / "reproject_utils.py",
)
reproj_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproj_module)
sys.modules["seestar.enhancement.reproject_utils"] = reproj_module
if "seestar.core.incremental_reprojection" in sys.modules:
    import importlib as _importlib
    _importlib.reload(sys.modules["seestar.core.incremental_reprojection"])

mod = importlib.import_module("seestar.core.incremental_reprojection")
reproject_and_coadd_batch = mod.reproject_and_coadd_batch


def make_wcs(shape=(4, 4)):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = np.array([-0.01, 0.01])
    w.wcs.crval = [0, 0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (shape[1], shape[0])
    return w


def test_reproject_and_coadd_batch_rgb():
    # reload real reproject_utils in case another test replaced it
    spec = importlib.util.spec_from_file_location(
        "seestar.enhancement.reproject_utils",
        ROOT / "seestar" / "enhancement" / "reproject_utils.py",
    )
    reproj_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reproj_module)
    sys.modules["seestar.enhancement.reproject_utils"] = reproj_module

    wcs_in = make_wcs()
    hdr = wcs_in.to_header()
    img = np.random.random((4, 4, 3)).astype(np.float32)
    out, cov = reproject_and_coadd_batch([img], [hdr], wcs_in, (4, 4))
    assert out.shape == (4, 4, 3)
    assert cov.shape == (4, 4)


