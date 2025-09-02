import sys
import types
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


# Ensure local package is importable and stub heavy dependencies
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if "seestar.gui" not in sys.modules:
    seestar_pkg = types.ModuleType("seestar")
    seestar_pkg.__path__ = [str(ROOT / "seestar")]
    gui_pkg = types.ModuleType("seestar.gui")
    gui_pkg.__path__ = [str(ROOT / "seestar" / "gui")]
    # Stub modules that boring_stack imports but aren't needed here
    settings_mod = types.ModuleType("seestar.gui.settings")
    settings_mod.SettingsManager = object
    queue_mod = types.ModuleType("seestar.queuep.queue_manager")
    queue_mod.SeestarQueuedStacker = object
    queue_mod._quality_metrics_worker = object
    alignment_mod = types.ModuleType("seestar.core.alignment")
    alignment_mod.SeestarAligner = object
    norm_mod = types.ModuleType("seestar.core.normalization")
    norm_mod._calc_linear_fit = lambda *args, **kwargs: None
    norm_mod._normalize_images = lambda *args, **kwargs: None
    norm_mod._calc_multiplicative = lambda *args, **kwargs: None
    norm_mod._normalize_images_linear_fit = lambda *args, **kwargs: None
    norm_mod._normalize_images_sky_mean = lambda *args, **kwargs: None
    streaming_mod = types.ModuleType("seestar.core.streaming_stack")
    streaming_mod.stack_disk_streaming = lambda *args, **kwargs: None
    image_proc_mod = types.ModuleType("seestar.core.image_processing")
    image_proc_mod.load_and_validate_fits = lambda *args, **kwargs: None
    image_proc_mod.save_fits_image = lambda *args, **kwargs: None
    image_proc_mod.sanitize_header_for_wcs = lambda *args, **kwargs: None
    astro_solver_mod = types.ModuleType("seestar.alignment.astrometry_solver")
    astro_solver_mod.AstrometrySolver = object
    reproject_utils_mod = types.ModuleType("seestar.reproject_utils")
    wcs_utils_mod = types.SimpleNamespace(_sanitize_continue_as_string=lambda hdr: None)
    sys.modules.update(
        {
            "seestar": seestar_pkg,
            "seestar.gui": gui_pkg,
            "seestar.gui.settings": settings_mod,
            "seestar.queuep.queue_manager": queue_mod,
            "seestar.core.alignment": alignment_mod,
            "seestar.core.normalization": norm_mod,
            "seestar.core.streaming_stack": streaming_mod,
            "seestar.core.image_processing": image_proc_mod,
            "seestar.alignment.astrometry_solver": astro_solver_mod,
            "seestar.reproject_utils": reproject_utils_mod,
            "seestar.utils.wcs_utils": wcs_utils_mod,
        }
    )

from seestar.gui.boring_stack import _load_wcs_header_only


def _create_fits_without_simple(path: Path):
    """Create a minimal FITS file missing the ``SIMPLE`` card."""
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 2
    header["NAXIS2"] = 2
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CD1_1"] = 1.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 1.0

    data = np.zeros((2, 2), dtype=np.float32)
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(path)

    # Replace the initial ``SIMPLE`` keyword with a non-standard one while
    # keeping the rest of the header intact so that only the absence of
    # ``SIMPLE`` triggers the relaxed loader.
    with open(path, "r+b") as f:
        first = f.read(80)
        f.seek(0)
        f.write(b"NOSIMPLE" + first[8:])


def test_load_wcs_header_only_handles_missing_simple(tmp_path):
    fp = tmp_path / "no_simple.fits"
    _create_fits_without_simple(fp)

    w = _load_wcs_header_only(str(fp))
    assert isinstance(w, WCS)
