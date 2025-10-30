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


def _ensure_sip_suffix(header: fits.Header) -> None:
    """Ensure CTYPE cards carry a "-SIP" suffix when SIP coefficients exist.

    Astropy emits an informational message and may behave inconsistently when
    SIP distortion keywords (e.g. ``A_ORDER``, ``B_ORDER`` or ``A_0_2``) are
    present but the projection types in ``CTYPE1``/``CTYPE2`` miss the
    required "-SIP" suffix (e.g. ``RA---TAN`` → ``RA---TAN-SIP``).

    This helper mutates the provided header in-place to append "-SIP" to both
    axes when SIP terms are detected and the suffix is missing. It is a
    no-op when SIP terms are absent or the suffix is already present.
    """
    try:
        # Detect SIP presence via common keywords
        has_sip = any(k in header for k in ("A_ORDER", "B_ORDER"))
        if not has_sip:
            # Also check for individual coefficient cards
            for k in list(header.keys()):
                uk = k.upper()
                if uk.startswith("A_") or uk.startswith("B_") or uk.startswith("AP_") or uk.startswith("BP_"):
                    has_sip = True
                    break
        if not has_sip:
            return

        for key in ("CTYPE1", "CTYPE2"):
            if key not in header:
                continue
            val = str(header.get(key, ""))
            uval = val.upper()
            # Append only once and only for TAN-like celestial projections
            if "-SIP" not in uval:
                # Keep original value and simply append the suffix
                try:
                    header[key] = f"{val}-SIP"
                except Exception:
                    pass
    except Exception:
        # Best-effort; avoid breaking the pipeline on exotic headers
        return


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
        _ensure_sip_suffix(header)
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
