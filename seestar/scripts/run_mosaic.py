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
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        astap_exe_path=args.astap_path or config.get("astap_executable_path", ""),
        astap_data_dir_param=args.astap_data_dir or config.get("astap_data_directory_path", ""),
        astap_search_radius_config=args.search_radius if args.search_radius is not None else config.get("astap_default_search_radius", 3.0),
        astap_downsample_config=config.get("astap_default_downsample", 2),
        astap_sensitivity_config=config.get("astap_default_sensitivity", 100),
        cluster_threshold_config=config.get("cluster_panel_threshold", 0.5),
        progress_callback=cli_progress,
        stack_norm_method=config.get("stacking_normalize_method", "none"),
        stack_weight_method=config.get("stacking_weighting_method", "none"),
        stack_reject_algo=config.get("stacking_rejection_algorithm", "kappa_sigma"),
        stack_kappa_low=config.get("stacking_kappa_low", 3.0),
        stack_kappa_high=config.get("stacking_kappa_high", 3.0),
        parsed_winsor_limits=winsor_limits,
        stack_final_combine=config.get("stacking_final_combine_method", "mean"),
        apply_radial_weight_config=config.get("apply_radial_weight", False),
        radial_feather_fraction_config=config.get("radial_feather_fraction", 0.8),
        radial_shape_power_config=config.get("radial_shape_power", 2.0),
        min_radial_weight_floor_config=config.get("min_radial_weight_floor", 0.0),
        final_assembly_method_config=config.get("final_assembly_method", "reproject_coadd"),
        num_base_workers_config=config.get("num_processing_workers", 0),
        apply_master_tile_crop_config=config.get("apply_master_tile_crop", False),
        master_tile_crop_percent_config=config.get("master_tile_crop_percent", 10.0),
        save_final_as_uint16_config=config.get("save_final_as_uint16", False),
        coadd_use_memmap_config=config.get("coadd_use_memmap", True),
        coadd_memmap_dir_config=config.get("coadd_memmap_dir", ""),
        coadd_cleanup_memmap_config=config.get("coadd_cleanup_memmap", True),
        assembly_process_workers_config=config.get("assembly_process_workers", 0),
        auto_limit_frames_per_master_tile_config=config.get("auto_limit_frames_per_master_tile", True),
        auto_limit_memory_fraction_config=config.get("auto_limit_memory_fraction", 0.1),
        winsor_worker_limit_config=config.get("winsor_worker_limit", 4),
        max_raw_per_master_tile_config=config.get("max_raw_per_master_tile", 0),
        solver_settings=solver_settings,
    )


if __name__ == "__main__":
    main()
