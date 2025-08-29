from pathlib import Path
import importlib.util

import pytest
from astropy.io import fits
from astropy.wcs import WCS

# Import the sanitize_header_for_wcs function without triggering the
# heavy seestar package initialisation.
spec = importlib.util.spec_from_file_location(
    "image_processing", Path(__file__).resolve().parents[1] / "seestar" / "core" / "image_processing.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sanitize_header_for_wcs = module.sanitize_header_for_wcs

def test_sanitize_header_for_wcs_removes_non_string_continue():
    header = fits.Header()
    header['SIMPLE'] = True
    header['BITPIX'] = 8
    header['NAXIS'] = 2
    header['NAXIS1'] = 10
    header['NAXIS2'] = 10
    header['CRPIX1'] = 5
    header['CRPIX2'] = 5
    header['CRVAL1'] = 0
    header['CRVAL2'] = 0
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CD1_1'] = -0.0002777777778
    header['CD2_2'] = 0.0002777777778
    header.append(fits.Card('CONTINUE', 123))

    # Ensure WCS construction fails before sanitisation
    with pytest.raises(Exception):
        WCS(header, naxis=2)

    sanitize_header_for_wcs(header)

    # After sanitisation the invalid CONTINUE card should be gone
    wcs = WCS(header, naxis=2)
    assert wcs is not None
    assert all(card.keyword != 'CONTINUE' for card in header.cards)
