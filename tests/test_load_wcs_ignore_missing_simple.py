"""Coverage for ``_load_wcs_header_only`` tolerating a FITS file whose primary
header is missing the ``SIMPLE`` card.

The previous version of this test installed a large synthetic ``seestar``
package tree into ``sys.modules`` (including a minimal
``seestar.alignment.astrometry_solver`` with only ``AstrometrySolver``) so it
could import ``seestar.gui.boring_stack`` without pulling the heavy
scientific/GUI tree.  That synthetic tree persisted globally and later broke
``tests/test_reproject_zm_wcs_fix.py`` (which imports
``_canonicalize_wcs_scale`` from the real ``astrometry_solver``).  The real
``_load_wcs_header_only`` imports cleanly, so we import it directly.
"""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

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
