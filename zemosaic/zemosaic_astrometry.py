# zemosaic_astrometry.py
"""Convenience ASTAP solving wrapper for ZeMosaic."""

from __future__ import annotations

from seestar.alignment.astrometry_solver import (
    AstrometrySolver,
    ASTAP_DEFAULT_SEARCH_RADIUS,
)
from astropy.io import fits


def solve_with_astap(
    image_fits_path: str,
    original_fits_header,
    astap_exe_path: str,
    astap_data_dir: str,
    search_radius_deg: float | None = None,
    downsample_factor: int | None = None,
    sensitivity: int | None = None,
    use_radec_hints: bool = False,
    timeout_sec: int = 180,
    update_original_header_in_place: bool = True,
    progress_callback=None,
    solver_instance: AstrometrySolver | None = None,
):
    """Solve WCS for an image using ASTAP via :class:`AstrometrySolver`.

    The function performs several attempts with gradually fewer hints
    (RA/DEC and pixel scale) if the initial attempt fails. This helps
    when FITS headers contain inaccurate metadata.
    """

    if solver_instance is None:
        solver_instance = AstrometrySolver(progress_callback=progress_callback)

    base_settings = {
        "local_solver_preference": "astap",
        "astap_path": astap_exe_path,
        "astap_data_dir": astap_data_dir,
        "astap_search_radius": search_radius_deg
        if search_radius_deg is not None
        else ASTAP_DEFAULT_SEARCH_RADIUS,
        "astap_downsample": downsample_factor,
        "astap_sensitivity": sensitivity,
        "astap_timeout_sec": timeout_sec,
        # Defaults for other solver parameters
        "local_ansvr_path": None,
        "api_key": None,
        "ansvr_timeout_sec": 120,
        "astrometry_net_timeout_sec": 300,
        "scale_est_arcsec_per_pix": None,
        "scale_tolerance_percent": 20,
    }

    header_orig = original_fits_header or fits.Header()
    header_no_hints = header_orig.copy()
    for key in ("RA", "DEC", "CRVAL1", "CRVAL2"):
        if key in header_no_hints:
            del header_no_hints[key]
    header_no_scale = header_no_hints.copy()
    for key in ("XPIXSZ", "PIXSIZE1", "FOCALLEN"):
        if key in header_no_scale:
            del header_no_scale[key]

    attempts = []
    if use_radec_hints:
        attempts.append(("hints", header_orig, True, search_radius_deg))
    attempts.append(("no_hints", header_orig if not use_radec_hints else header_no_hints, False, search_radius_deg))
    attempts.append(("blind", header_no_scale, False, search_radius_deg))
    if search_radius_deg and search_radius_deg > 5:
        attempts.append(("blind_small_radius", header_no_scale, False, 5.0))

    for idx, (label, hdr, hints_flag, radius_val) in enumerate(attempts, 1):
        try_settings = base_settings.copy()
        try_settings["use_radec_hints"] = hints_flag
        if radius_val is not None:
            try_settings["astap_search_radius"] = radius_val
        solver_instance._log(
            f"ASTAP attempt {idx}: {label} (radius={try_settings['astap_search_radius']})",
            "DEBUG",
        )
        wcs = solver_instance.solve(
            image_fits_path,
            hdr,
            try_settings,
            update_header_with_solution=False,
        )
        if wcs:
            if update_original_header_in_place and original_fits_header is not None:
                solver_instance._update_fits_header_with_wcs(
                    original_fits_header, wcs, solver_name="ASTAP"
                )
            return wcs

    return None
