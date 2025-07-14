import importlib.util
import sys
import types
from pathlib import Path
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

seestar_pkg = sys.modules.get("seestar", types.ModuleType("seestar"))
seestar_pkg.__path__ = [str(ROOT / "seestar")]
sys.modules["seestar"] = seestar_pkg

enhancement_pkg = sys.modules.get(
    "seestar.enhancement", types.ModuleType("seestar.enhancement")
)
enhancement_pkg.__path__ = [str(ROOT / "seestar" / "enhancement")]
sys.modules["seestar.enhancement"] = enhancement_pkg

spec_rp = importlib.util.spec_from_file_location(
    "seestar.enhancement.reproject_utils",
    ROOT / "seestar" / "enhancement" / "reproject_utils.py",
)
reproject_utils = importlib.util.module_from_spec(spec_rp)
spec_rp.loader.exec_module(reproject_utils)
sys.modules["seestar.enhancement.reproject_utils"] = reproject_utils

spec = importlib.util.spec_from_file_location(
    "seestar.enhancement.mosaic_utils",
    ROOT / "seestar" / "enhancement" / "mosaic_utils.py",
)
mosaic_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mosaic_utils)

spec2 = importlib.util.spec_from_file_location(
    "seestar.core.reprojection", ROOT / "seestar" / "core" / "reprojection.py"
)
core_reproj = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(core_reproj)
sys.modules["seestar.core.reprojection"] = core_reproj
reproject_to_reference_wcs = core_reproj.reproject_to_reference_wcs


def make_wcs(shape=(4, 4)):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    w.wcs.cdelt = np.array([-0.01, 0.01])
    w.wcs.crval = [0, 0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (shape[1], shape[0])
    return w


def dummy_reproj(data_wcs, output_projection=None, shape_out=None, **kwargs):
    data, _ = data_wcs
    if data.ndim == 3:
        result = data[: shape_out[0], : shape_out[1], :]
    else:
        result = data[: shape_out[0], : shape_out[1]]
    return result, np.ones(shape_out, dtype=float)


def dummy_reproject_and_coadd(input_pairs, output_projection, shape_out, **kwargs):
    return dummy_reproj(input_pairs[0], output_projection, shape_out)


def test_mosaic_shape_matches_simple_reproj(tmp_path, monkeypatch):
    monkeypatch.setattr(mosaic_utils, "reproject_and_coadd", dummy_reproject_and_coadd)
    monkeypatch.setattr(mosaic_utils, "reproject_interp", dummy_reproj)
    monkeypatch.setattr(core_reproj, "reproject_interp", dummy_reproj)

    data = np.ones((4, 4, 3), dtype=np.float32)
    fits_path = tmp_path / "tile.fits"
    fits.writeto(fits_path, data, overwrite=True)

    wcs_in = make_wcs(shape=(4, 4))
    final_wcs = make_wcs(shape=(4, 4))

    final_shape = (4, 4)

    mosaic, _ = mosaic_utils.assemble_final_mosaic_with_reproject_coadd(
        [(str(fits_path), wcs_in)], final_wcs, final_shape, match_bg=False
    )

    simple = reproject_to_reference_wcs(data, wcs_in, final_wcs, final_shape)

    assert mosaic.shape == simple.shape
