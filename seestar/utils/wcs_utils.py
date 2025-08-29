from astropy.io import fits
from astropy.wcs import WCS

WCS_KEYS_PREFIXES = (
    "WCSAXES", "CTYPE", "CUNIT", "CRVAL", "CRPIX",
    "CD", "PC", "CDELT", "CROTA", "LONPOLE", "LATPOLE",
    "MJDREF", "EQUINOX", "RADESYS", "A_ORDER", "B_ORDER", "A_", "B_", "AP_", "BP_",
    "PV", "PROJP"
)

def _strip_wcs_cards(header: fits.Header) -> None:
    """Remove existing WCS-related cards from ``header`` in-place."""
    to_del = []
    for k in list(header.keys()):
        up = k.upper()
        if up == "COMMENT" or up == "HISTORY":
            continue
        if any(up.startswith(p) for p in WCS_KEYS_PREFIXES):
            to_del.append(k)
    for k in to_del:
        try:
            del header[k]
        except Exception:
            pass

def _sanitize_continue_as_string(header: fits.Header) -> None:
    """Force ``CONTINUE`` card values to string type."""
    for k, v in list(header.items()):
        if k == "CONTINUE":
            header[k] = str(v)

def write_wcs_to_fits_inplace(fits_path: str, wcs_obj: WCS) -> None:
    """Persist ``wcs_obj`` into ``fits_path`` updating header only.

    The FITS file is opened with ``memmap=True`` and data are left untouched.
    Any previous WCS cards are removed before injecting the new solution.
    """
    with fits.open(fits_path, mode="update", memmap=True) as hdul:
        h = hdul[0].header
        _strip_wcs_cards(h)
        h_wcs = wcs_obj.to_header(relax=True)
        for k, v in h_wcs.items():
            h[k] = v
        _sanitize_continue_as_string(h)
        hdul.flush()
