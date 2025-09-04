import sys
import types
from pathlib import Path
import numpy as np
from types import SimpleNamespace
from astropy.io import fits
from astropy.wcs import WCS

# Ensure repository root on path and stub ``cv2`` to avoid heavy dependency
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("cv2", types.SimpleNamespace(cuda=types.SimpleNamespace(getCudaEnabledDeviceCount=lambda: 0)))

from seestar.queuep.queue_manager import SeestarQueuedStacker


def test_merge_reference_wcs(tmp_path):
    ref_wcs = WCS(naxis=2)
    ref_wcs.wcs.crpix = [1.0, 2.0]
    ref_wcs.wcs.cdelt = np.array([0.1, 0.1])
    ref_wcs.wcs.crval = [30.0, 40.0]
    ref_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    header_orig = fits.Header()
    header_orig["CRPIX1"] = 10.0
    header_orig["CRPIX2"] = 20.0
    header_orig["CRVAL1"] = 1.0
    header_orig["CRVAL2"] = 2.0
    header_orig["CD1_1"] = 0.2
    header_orig["CD1_2"] = 0.0
    header_orig["CD2_1"] = 0.0
    header_orig["CD2_2"] = 0.2
    header_orig["CTYPE1"] = "RA---TAN"
    header_orig["CTYPE2"] = "DEC--TAN"

    dummy = SimpleNamespace(
        aligned_temp_dir=str(tmp_path),
        aligned_files_count=0,
        reference_wcs_object=ref_wcs,
        reference_shape=None,
        update_progress=lambda *a, **k: None,
    )

    img = np.zeros((5, 5, 3), dtype=np.float32)
    mask = np.ones((5, 5), dtype=np.uint8)

    img_path, _ = SeestarQueuedStacker._save_aligned_temp(dummy, img, mask)
    hdr = SeestarQueuedStacker._merge_reference_wcs(dummy, header_orig)

    hdr_path = tmp_path / "aligned_00000.hdr"
    with open(hdr_path, "w", encoding="utf-8") as f:
        f.write(hdr.tostring(sep="\n"))

    with fits.open(img_path) as hdul:
        h = hdul[0].header
    assert h["CRVAL1"] == ref_wcs.wcs.crval[0]
    assert h["CRVAL2"] == ref_wcs.wcs.crval[1]

    hdr_loaded = fits.Header.fromtextfile(hdr_path)
    assert hdr_loaded["CRVAL1"] == ref_wcs.wcs.crval[0]
    assert hdr_loaded["CRVAL1"] != header_orig["CRVAL1"]

