# zemosaic_astrometry.py
"""Convenience ASTAP solving wrapper for ZeMosaic."""

from __future__ import annotations

from seestar.alignment.astrometry_solver import (
    AstrometrySolver,
    ASTAP_DEFAULT_SEARCH_RADIUS,
)


def solve_with_astap(
    image_fits_path: str,
    original_fits_header,
    astap_exe_path: str,
    astap_data_dir: str,
    search_radius_deg: float | None = None,
    downsample_factor: int | None = None,
    sensitivity: int | None = None,
    timeout_sec: int = 180,
    update_original_header_in_place: bool = True,
    progress_callback=None,
    solver_instance: AstrometrySolver | None = None,
):
    """Solve WCS for an image using ASTAP via :class:`AstrometrySolver`."""

    if solver_instance is None:
        solver_instance = AstrometrySolver(progress_callback=progress_callback)

    settings = {
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
        "use_radec_hints": False,
    }

    return solver_instance.solve(
        image_fits_path,
        original_fits_header,
        settings,
        update_header_with_solution=update_original_header_in_place,
    )
