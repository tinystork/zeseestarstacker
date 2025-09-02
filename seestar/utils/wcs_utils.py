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


def inject_sanitized_wcs(header: fits.Header, wcs_obj: WCS) -> int:
    """Inject a sanitized WCS solution into ``header``.

    Existing WCS cards and verbose HISTORY/COMMENT entries are removed.
    Only ASCII values are written, keys are truncated to 8 characters and
    ``CONTINUE`` cards are skipped.  Returns the number of WCS keywords
    written to ``header``.
    """

    _strip_wcs_cards(header)
    # Purge existing HISTORY/COMMENT
    for key in ["HISTORY", "COMMENT"]:
        while key in header:
            del header[key]

    h_wcs = wcs_obj.to_header(relax=True)
    count = 0
    for k, v in h_wcs.items():
        upk = k.upper()
        if upk in ("HISTORY", "COMMENT", "CONTINUE"):
            continue
        k8 = k[:8]
        if isinstance(v, str):
            try:
                v = v.encode("ascii", "ignore").decode("ascii")
            except Exception:
                continue
        try:
            header[k8] = v
            count += 1
        except Exception:
            continue

    _sanitize_continue_as_string(header)
    return count


def write_wcs_to_fits_inplace(fits_path: str, wcs_obj: WCS) -> int:
    """Persist ``wcs_obj`` into ``fits_path`` updating header only.

    The FITS file is opened with ``memmap=True`` and data are left untouched.
    Any previous WCS cards are removed before injecting the new solution.
    Returns the number of WCS keywords written.
    """
    with fits.open(fits_path, mode="update", memmap=True) as hdul:
        h = hdul[0].header
        count = inject_sanitized_wcs(h, wcs_obj)
        hdul.flush()
    return count
