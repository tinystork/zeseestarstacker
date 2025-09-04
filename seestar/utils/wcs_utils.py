import os
from astropy.io import fits
from astropy.wcs import WCS

WCS_KEYS_PREFIXES = (
    "WCSAXES",
    "CTYPE",
    "CUNIT",
    "CRVAL",
    "CRPIX",
    "CD",
    "PC",
    "CDELT",
    "CROTA",
    "LONPOLE",
    "LATPOLE",
    "MJDREF",
    "EQUINOX",
    "RADESYS",
    "A_ORDER",
    "B_ORDER",
    "A_",
    "B_",
    "AP_",
    "BP_",
    "PV",
    "PROJP",
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


def inject_sanitized_wcs(header: fits.Header, src) -> fits.Header | None:
    """Inject a sanitized WCS into ``header`` from various sources.

    Parameters
    ----------
    header : fits.Header
        Destination header updated in-place.
    src : astropy.wcs.WCS | fits.Header | str
        Source of the WCS information.  If a :class:`~astropy.wcs.WCS` is
        provided, its serialized header is merged.  If ``src`` is a
        :class:`~astropy.io.fits.Header`, it is used directly.  When ``src``
        is a string, it is interpreted as a path to a FITS file or to a
        ``.wcs``/``.hdr`` sidecar containing WCS keywords.

    Returns
    -------
    fits.Header or None
        The updated ``header`` on success, otherwise ``None``.
    """

    try:
        if isinstance(src, WCS):
            src_hdr = src.to_header(relax=True)
        elif isinstance(src, fits.Header):
            src_hdr = src
        else:  # assume path
            path = str(src)
            base, ext = os.path.splitext(path)
            sidecar = base + ".wcs"
            if os.path.isfile(sidecar):
                src_hdr = fits.Header.fromtextfile(sidecar)
            elif os.path.isfile(path):
                src_hdr = fits.getheader(path, memmap=False)
            else:
                return None

        _strip_wcs_cards(header)
        for key in ["HISTORY", "COMMENT"]:
            while key in header:
                del header[key]

        for k, v in src_hdr.items():
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
            except Exception:
                continue

        _sanitize_continue_as_string(header)
        return header
    except Exception:
        return None


def write_wcs_to_fits_inplace(fits_path: str, wcs_obj) -> int:
    """Persist a WCS solution or header into ``fits_path`` updating header only.

    Parameters
    ----------
    fits_path : str
        Path to the FITS file to update.
    wcs_obj : :class:`~astropy.wcs.WCS` or :class:`~astropy.io.fits.Header`
        WCS object or ready-to-write header containing WCS keywords.

    The FITS file is opened with ``memmap=True`` and data are left untouched.
    Any previous WCS cards are removed before injecting the new solution.
    Returns the number of WCS keywords written.
    """
    with fits.open(fits_path, mode="update", memmap=True) as hdul:
        h = hdul[0].header
        inject_sanitized_wcs(h, wcs_obj)
        count = sum(1 for k in h.keys() if any(k.startswith(p) for p in WCS_KEYS_PREFIXES))
        hdul.flush()
    return count
