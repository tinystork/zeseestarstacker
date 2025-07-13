import importlib.util
import sys
from pathlib import Path
import glob

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

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

spec2 = importlib.util.spec_from_file_location(
    "seestar.enhancement.stack_enhancement",
    ROOT / "seestar" / "enhancement" / "stack_enhancement.py",
)
stack_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(stack_module)
sys.modules["seestar.enhancement.stack_enhancement"] = stack_module

spec3 = importlib.util.spec_from_file_location(
    "seestar.core.incremental_reprojection",
    ROOT / "seestar" / "core" / "incremental_reprojection.py",
)
inc_module = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(inc_module)
sys.modules["seestar.core.incremental_reprojection"] = inc_module

from seestar.core.incremental_reprojection import reproject_and_coadd_batch
from seestar.enhancement.stack_enhancement import apply_edge_crop

SAMPLE_DIR = ROOT / "seestar" / "sample"
FILES = sorted(glob.glob(str(SAMPLE_DIR / '*2244*.fit')))[:3]


def load_data(paths, crop=False):
    images = []
    headers = []
    for f in paths:
        with fits.open(f) as hdul:
            img = hdul[0].data.astype(np.float32)
            hdr = hdul[0].header
            if crop:
                img, hdr = apply_edge_crop(img, 0.02, header=hdr)
            images.append(img)
            headers.append(hdr)
    return images, headers


def test_reproject_and_coadd_batch_sample():
    # reload modules in case previous tests monkeypatched them
    spec = importlib.util.spec_from_file_location(
        "seestar.enhancement.reproject_utils",
        ROOT / "seestar" / "enhancement" / "reproject_utils.py",
    )
    reproj_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reproj_module)
    sys.modules["seestar.enhancement.reproject_utils"] = reproj_module

    spec3 = importlib.util.spec_from_file_location(
        "seestar.core.incremental_reprojection",
        ROOT / "seestar" / "core" / "incremental_reprojection.py",
    )
    inc_module = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(inc_module)
    sys.modules["seestar.core.incremental_reprojection"] = inc_module
    from seestar.core.incremental_reprojection import reproject_and_coadd_batch

    images, headers = load_data(FILES, crop=False)
    ref_wcs = WCS(headers[0])
    if ref_wcs.pixel_shape is None:
        ref_wcs.pixel_shape = (headers[0]['NAXIS1'], headers[0]['NAXIS2'])
    shape = (ref_wcs.pixel_shape[1], ref_wcs.pixel_shape[0])
    result, cov = reproject_and_coadd_batch(images, headers, ref_wcs, shape)
    assert result.shape == (1920, 1080)
    assert cov.shape == (1920, 1080)

    images_c, headers_c = load_data(FILES, crop=True)
    ref_wcs_c = WCS(headers_c[0])
    if ref_wcs_c.pixel_shape is None:
        ref_wcs_c.pixel_shape = (headers_c[0]['NAXIS1'], headers_c[0]['NAXIS2'])
    shape_c = (ref_wcs_c.pixel_shape[1], ref_wcs_c.pixel_shape[0])
    result_c, cov_c = reproject_and_coadd_batch(images_c, headers_c, ref_wcs_c, shape_c)
    assert result_c.shape == (1844, 1038)
    assert cov_c.shape == (1844, 1038)

