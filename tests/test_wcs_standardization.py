import importlib.util
from pathlib import Path
import numpy as np
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "reproj_utils",
    ROOT / "seestar" / "core" / "reprojection_utils.py",
)
reproj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproj)
compute_final_output_grid = reproj.compute_final_output_grid


def make_wcs(radesys="ICRS"):
    w = WCS(naxis=2)
    w.wcs.crpix = [5, 5]
    w.wcs.cd = np.array([[-0.01, 0.0], [0.0, 0.01]])
    w.wcs.crval = [10.0, 20.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.radesys = radesys
    w.wcs.equinox = 2000.0
    w.pixel_shape = (10, 10)
    return w


def test_compute_final_output_grid_consistent():
    w1 = make_wcs("ICRS")
    w2 = make_wcs("FK5")
    headers = [((10, 10), w1), ((10, 10), w2)]
    out_wcs, out_shape = compute_final_output_grid(headers, scale=1.0)
    assert out_shape is not None
    assert out_wcs.wcs.radesys == "ICRS"

