import logging
import os
from astropy.io import fits
from astropy.wcs import WCS

from seestar.core.image_processing import sanitize_header_for_wcs

logger = logging.getLogger(__name__)


def load_wcs_header_only(path: str) -> WCS:
    """Return ``WCS`` loaded from ``path`` without touching image data.

    The header is read with ``ignore_missing_simple=True`` so truncated files
    raise ``ValueError``. If the file is smaller than 2880 bytes or lacks the
    ``SIMPLE`` card a ``ValueError`` is raised.
    """
    size = os.path.getsize(path)
    if size < 2880:
        raise ValueError(f"file too small ({size} bytes)")
    hdr = fits.getheader(path, ignore_missing_simple=True, memmap=False)
    if "SIMPLE" not in hdr:
        raise ValueError("missing SIMPLE")
    sanitize_header_for_wcs(hdr)
    return WCS(hdr, naxis=2, relax=True)
