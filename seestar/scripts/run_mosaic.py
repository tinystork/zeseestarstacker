#!/usr/bin/env python3
"""Simple command-line interface to run ZeMosaic's hierarchical mosaic worker.

This script is meant to be invoked from the project root or installed as
``python -m seestar.scripts.run_mosaic``. It loads default values from
``zemosaic_config`` and allows overriding key astrometric solver options
via command-line arguments.
"""
import argparse
import logging

from zemosaic import zemosaic_config, zemosaic_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the hierarchical mosaic process on a folder of FITS files. "
            "Solver options are gathered into a solver_settings dict and passed "
            "to run_hierarchical_mosaic."
        ),
    )
    parser.add_argument("input_folder", help="Directory containing FITS tiles")
    parser.add_argument("output_folder", help="Directory where results are written")
    parser.add_argument("--astap-path", dest="astap_path", help="Path to ASTAP executable")
    parser.add_argument("--astap-data-dir", dest="astap_data_dir", help="Directory with ASTAP star catalogs")
    parser.add_argument("--search-radius", dest="search_radius", type=float, help="ASTAP search radius in degrees")
    parser.add_argument("--ansvr-config", dest="local_ansvr_path", help="Path to local ansvr.cfg")
    parser.add_argument("--api-key", dest="api_key", help="API key for astrometry.net")
    parser.add_argument("--local-solver-preference", choices=["none", "astap", "ansvr"],
                        dest="local_solver_preference", help="Preferred local solver")
    parser.add_argument("--astrometry-method", choices=["astap", "astrometry", "astrometry.net"],
                        dest="astrometry_method", help="Solver method to use")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def cli_progress(message: str, progress: float | None = None) -> None:
    if progress is not None:
        print(f"[{progress:.1f}%] {message}")
    else:
        print(message)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    config = zemosaic_config.load_config()

    solver_settings = {
        "astap_path": args.astap_path,
        "astap_data_dir": args.astap_data_dir,
        "astap_search_radius": args.search_radius,
        "local_ansvr_path": args.local_ansvr_path,
        "api_key": args.api_key,
        "local_solver_preference": args.local_solver_preference,
        "astrometry_method": args.astrometry_method,
    }
    # Remove None entries
    solver_settings = {k: v for k, v in solver_settings.items() if v is not None}

    winsor_str = config.get("stacking_winsor_limits", "0.05,0.05")
    try:
        winsor_limits = tuple(float(x) for x in winsor_str.split(","))
    except Exception:
        winsor_limits = (0.05, 0.05)

    zemosaic_worker.run_hierarchical_mosaic(
        args.input_folder,
        args.output_folder,
        solver_settings,
        config.get("cluster_panel_threshold", 0.5),
        cli_progress,
        config.get("stacking_normalize_method", "none"),
        config.get("stacking_weighting_method", "none"),
        config.get("stacking_rejection_algorithm", "kappa_sigma"),
        config.get("stacking_kappa_low", 3.0),
        config.get("stacking_kappa_high", 3.0),
        winsor_limits,
        config.get("stacking_final_combine_method", "mean"),
        config.get("apply_radial_weight", False),
        config.get("radial_feather_fraction", 0.8),
        config.get("radial_shape_power", 2.0),
        config.get("min_radial_weight_floor", 0.0),
        config.get("final_assembly_method", "reproject_coadd"),
        config.get("num_processing_workers", 0),
        config.get("apply_master_tile_crop", False),
        config.get("master_tile_crop_percent", 10.0),
        config.get("save_final_as_uint16", False),
        config.get("re_solve_cropped_tiles", False),
    )


if __name__ == "__main__":
    main()
