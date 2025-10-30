"""
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘ ZeMosaic / ZeSeestarStacker Project                                  â•‘
â•‘                                                                      â•‘
â•‘ Auteur  : Tinystork, seigneur des couteaux Ã  beurre (aka Tristan Nauleau)  
â•‘ Partenaire : J.A.R.V.I.S. (/ËˆdÊ’É‘ËrvÉªs/) â€” Just a Rather Very Intelligent System  
â•‘              (aka ChatGPT, Grand MaÃ®tre du ciselage de code)         â•‘
â•‘                                                                      â•‘
â•‘ Licence : GNU General Public License v3.0 (GPL-3.0)                  â•‘
â•‘                                                                      â•‘
â•‘ Description :                                                        â•‘
â•‘   Ce programme a Ã©tÃ© forgÃ© Ã  la lueur des pixels et de la cafÃ©ine,   â•‘
â•‘   dans le but noble de transformer des nuages de photons en art      â•‘
â•‘   astronomique. Si vous lâ€™utilisez, pensez Ã  dire â€œmerciâ€,           â•‘
â•‘   Ã  lever les yeux vers le ciel, ou Ã  citer Tinystork et J.A.R.V.I.S.â•‘
â•‘   (le karma des dÃ©veloppeurs en dÃ©pend).                             â•‘
â•‘                                                                      â•‘
â•‘ Avertissement :                                                      â•‘
â•‘   Aucune IA ni aucun couteau Ã  beurre nâ€™a Ã©tÃ© blessÃ© durant le       â•‘
â•‘   dÃ©veloppement de ce code.                                          â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘ ZeMosaic / ZeSeestarStacker Project                                  â•‘
â•‘                                                                      â•‘
â•‘ Author  : Tinystork, Lord of the Butter Knives (aka Tristan Nauleau) â•‘
â•‘ Partner : J.A.R.V.I.S. (/ËˆdÊ’É‘ËrvÉªs/) â€” Just a Rather Very Intelligent System  
â•‘           (aka ChatGPT, Grand Master of Code Chiseling)              â•‘
â•‘                                                                      â•‘
â•‘ License : GNU General Public License v3.0 (GPL-3.0)                  â•‘
â•‘                                                                      â•‘
â•‘ Description:                                                         â•‘
â•‘   This program was forged under the sacred light of pixels and       â•‘
â•‘   caffeine, with the noble intent of turning clouds of photons into  â•‘
â•‘   astronomical art. If you use it, please consider saying â€œthanks,â€  â•‘
â•‘   gazing at the stars, or crediting Tinystork and J.A.R.V.I.S. â€”     â•‘
â•‘   developer karma depends on it.                                     â•‘
â•‘                                                                      â•‘
â•‘ Disclaimer:                                                          â•‘
â•‘   No AIs or butter knives were harmed in the making of this code.    â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

Optional GUI filter for ZeMosaic Phase 1 results.

This module exposes a single function:

    launch_filter_interface(
        raw_files_with_wcs: list[dict],
        initial_overrides: dict | None = None,
    ) -> tuple[list[dict], bool, dict | None]

It opens a small Tkinter window with a schematic sky map (RA/Dec, in degrees)
showing the footprint of each WCS-resolved image as a polygon and a checkbox
list to include/exclude individual images. A helper can also exclude images
farther than X degrees from the global center.

Safety requirements:
- If this module is missing or raises, the caller should continue unchanged.
- If the window is closed or any error occurs, the original input list is
  returned unchanged.
  In that case this function returns (original_list, False) so the caller can
  decide to abort processing.

Dependencies limited to tkinter, astropy, matplotlib, numpy; all optional.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from collections.abc import Iterable
from dataclasses import asdict
import os
import sys
import shutil
import datetime
import importlib
import time
import traceback
import threading
import queue


def _group_center_deg(group: list[dict]) -> Optional[tuple[float, float]]:
    """Return the average RA/DEC (degrees) of a group if available."""

    ras: list[float] = []
    decs: list[float] = []
    for info in group:
        ra = info.get("RA")
        dec = info.get("DEC")
        if ra is not None and dec is not None:
            try:
                ras.append(float(ra))
                decs.append(float(dec))
            except Exception:
                continue
    if not ras:
        return None
    return (sum(ras) / len(ras), sum(decs) / len(decs))


def _angular_sep_deg(a: Optional[tuple[float, float]], b: Optional[tuple[float, float]]) -> float:
    """Approximate angular separation in degrees between two (RA, DEC) tuples."""

    if not a or not b:
        return 9999.0
    dra = abs(a[0] - b[0])
    ddec = abs(a[1] - b[1])
    return (dra ** 2 + ddec ** 2) ** 0.5


def _merge_small_groups(groups: list[list[dict]], min_size: int, cap: int) -> list[list[dict]]:
    """
    Merge groups smaller than ``min_size`` with their nearest neighbour when the
    resulting size stays below ``cap`` (allowing a 10% margin).
    """

    if not groups or min_size <= 0 or cap <= 0:
        return groups

    merged_flags = [False] * len(groups)
    centers = [_group_center_deg(g) for g in groups]

    for i, group in enumerate(groups):
        if merged_flags[i] or len(group) >= min_size:
            continue

        best_j: Optional[int] = None
        best_dist = float("inf")
        for j, neighbour in enumerate(groups):
            if i == j or merged_flags[j]:
                continue
            dist = _angular_sep_deg(centers[i], centers[j])
            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j is not None:
            candidate_size = len(groups[best_j]) + len(group)
            if candidate_size <= int(cap * 1.1):
                groups[best_j].extend(group)
                merged_flags[i] = True
                print(
                    f"[AutoMerge] Group {i} ({len(group)} imgs) merged into {best_j} "
                    f"(now {len(groups[best_j])})"
                )

    return [grp for idx, grp in enumerate(groups) if not merged_flags[idx]]


def launch_filter_interface(
    raw_files_with_wcs_or_dir,
    initial_overrides: Optional[Dict[str, Any]] = None,
    *,
    stream_scan: bool = False,
    scan_recursive: bool = True,
    batch_size: int = 100,
    preview_cap: int = 200,
    solver_settings_dict: Optional[Dict[str, Any]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """
    Display an optional Tkinter GUI to filter WCS-resolved images.

    Parameters
    ----------
    raw_files_with_wcs_or_dir : list[dict] | str
        Either the legacy list of metadata dictionaries or, when ``stream_scan``
        is True, a directory path to crawl lazily for FITS files.
        Each dict should ideally include:
          - 'path' (or 'path_raw') : absolute path to the FITS file
          - 'wcs' : astropy.wcs.WCS object
          - 'shape' (optional) : (H, W)
          - 'index' (optional) : processing index
          - 'center' (optional) : astropy.coordinates.SkyCoord
          - 'header' (optional) : astropy.io.fits.Header (used to infer shape)

    stream_scan : bool, optional
        Enable lazy directory crawling instead of receiving a pre-computed
        metadata list.  When enabled, ``raw_files_with_wcs_or_dir`` must be a
        directory path.
    scan_recursive : bool, optional
        When ``stream_scan`` is active, decide whether to descend into
        sub-directories (default: True).
    batch_size : int, optional
        Number of files to accumulate before pushing a batch to the UI while
        streaming (default: 100).
    preview_cap : int, optional
        Maximum number of footprints drawn on the Matplotlib preview to avoid
        locking the UI for very large directories (default: 200).
    solver_settings_dict : dict, optional
        Dictionary of solver configuration selected in the main GUI. When
        provided, values such as ASTAP paths and search radius override the
        defaults loaded from configuration files.
    config_overrides : dict, optional
        Additional configuration values collected from the main GUI (for
        example the ASTAP data directory or clustering caps). These override
        the defaults detected by this module.

    Returns
    -------
    tuple[list[dict], bool, dict | None]
        (filtered_list, accepted, overrides) where accepted is True only when
        Validate is clicked; False when Cancel is clicked or the window is closed.
        overrides can contain optional metadata collected in the filter GUI, such
        as pre-computed master tile groupings or counters for resolved WCS files:
          {
             "preplan_master_groups": list[list[dict]],
             "autosplit_cap": int,
             "filter_excluded_indices": list[int],
             "resolved_wcs_count": int,
          }
        Keys are included only when relevant actions were triggered.
    """
    # Early validation and fail-safe behavior
    stream_mode = False
    input_dir: Optional[str] = None

    if stream_scan and isinstance(raw_files_with_wcs_or_dir, str):
        candidate = raw_files_with_wcs_or_dir
        if os.path.isdir(candidate):
            stream_mode = True
            input_dir = candidate
        else:
            return [], False, None

    if not stream_mode:
        if not isinstance(raw_files_with_wcs_or_dir, list) or not raw_files_with_wcs_or_dir:
            return raw_files_with_wcs_or_dir, False, None

    raw_items_input = raw_files_with_wcs_or_dir

    try:
        # --- Optional localization support (autonomous fallback) ---
        # If running inside the ZeMosaic project folder structure, try to use
        # the existing localization system and the language set in main GUI.
        localizer = None
        cfg_defaults: Dict[str, Any] = {
            "astap_executable_path": "",
            "astap_data_directory_path": "",
            "astap_default_search_radius": 0.0,
            "astap_default_downsample": 0,
            "astap_default_sensitivity": 100,
            "auto_limit_frames_per_master_tile": True,
            "max_raw_per_master_tile": 0,
            "apply_master_tile_crop": False,
            "master_tile_crop_percent": 0.0,
        }
        cfg: Dict[str, Any] | None = None
        solver_settings_payload: Dict[str, Any] = {}
        lang_code = "en"

        # Ensure project directory is on sys.path to import project modules
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if base_dir not in sys.path:
            sys.path.insert(0, base_dir)

        # Try to load localization support from either legacy or packaged paths
        localizer_cls = None
        localization_errors: list[str] = []
        for mod_name in ("zemosaic_localization", "locales.zemosaic_localization"):
            try:
                module = importlib.import_module(mod_name)
                candidate = getattr(module, "ZeMosaicLocalization", None)
                if candidate is not None:
                    localizer_cls = candidate
                    break
            except Exception as exc:
                localization_errors.append(str(exc))

        # Load persistent configuration if available
        zconfig_module = None
        try:
            zconfig_module = importlib.import_module("zemosaic_config")
        except Exception:
            try:
                pkg_prefix = globals().get("__package__") or ""
                if pkg_prefix:
                    zconfig_module = importlib.import_module(f"{pkg_prefix}.zemosaic_config")
            except Exception as exc:
                print(f"WARNING (Filter GUI): failed to import configuration module: {exc}")
                zconfig_module = None

        if zconfig_module is not None:
            try:
                cfg = zconfig_module.load_config()
                if isinstance(cfg, dict):
                    lang_code = str(cfg.get("language", lang_code))
                    cfg_defaults.update({
                        "astap_executable_path": cfg.get("astap_executable_path", cfg_defaults["astap_executable_path"]),
                        "astap_data_directory_path": cfg.get("astap_data_directory_path", cfg_defaults["astap_data_directory_path"]),
                        "astap_default_search_radius": cfg.get("astap_default_search_radius", cfg_defaults["astap_default_search_radius"]),
                        "astap_default_downsample": cfg.get("astap_default_downsample", cfg_defaults["astap_default_downsample"]),
                        "astap_default_sensitivity": cfg.get("astap_default_sensitivity", cfg_defaults["astap_default_sensitivity"]),
                        "auto_limit_frames_per_master_tile": cfg.get("auto_limit_frames_per_master_tile", cfg_defaults["auto_limit_frames_per_master_tile"]),
                        "max_raw_per_master_tile": cfg.get("max_raw_per_master_tile", cfg_defaults["max_raw_per_master_tile"]),
                        "apply_master_tile_crop": cfg.get("apply_master_tile_crop", cfg_defaults["apply_master_tile_crop"]),
                        "master_tile_crop_percent": cfg.get("master_tile_crop_percent", cfg_defaults["master_tile_crop_percent"]),
                    })
            except Exception as exc:
                print(f"WARNING (Filter GUI): failed to load configuration: {exc}")

        # Load solver settings (either provided by caller or defaults)
        solver_cls = None
        solver_module = None
        try:
            solver_module = importlib.import_module("solver_settings")
        except Exception:
            try:
                pkg_prefix = globals().get("__package__") or ""
                if pkg_prefix:
                    solver_module = importlib.import_module(f"{pkg_prefix}.solver_settings")
            except Exception as exc:
                print(f"WARNING (Filter GUI): failed to import solver settings: {exc}")
                solver_module = None

        if solver_module is not None:
            solver_cls = getattr(solver_module, "SolverSettings", None)

        if isinstance(solver_settings_dict, dict):
            solver_settings_payload.update(solver_settings_dict)
        elif solver_cls is not None:
            try:
                solver_settings = solver_cls.load_default()
            except Exception:
                solver_settings = solver_cls()
            try:
                solver_settings_payload.update(asdict(solver_settings))
            except Exception:
                pass

        if solver_settings_payload:
            exe_path = solver_settings_payload.get("astap_executable_path")
            data_path = solver_settings_payload.get("astap_data_directory_path")
            search_radius = solver_settings_payload.get("astap_search_radius_deg")
            downsample = solver_settings_payload.get("astap_downsample")
            sensitivity = solver_settings_payload.get("astap_sensitivity")

            if isinstance(exe_path, str) and exe_path:
                cfg_defaults["astap_executable_path"] = exe_path
            if isinstance(data_path, str) and data_path:
                cfg_defaults["astap_data_directory_path"] = data_path
            if search_radius is not None:
                cfg_defaults["astap_default_search_radius"] = search_radius
            if downsample is not None:
                cfg_defaults["astap_default_downsample"] = downsample
            if sensitivity is not None:
                cfg_defaults["astap_default_sensitivity"] = sensitivity

        if localizer_cls is not None:
            try:
                localizer = localizer_cls(language_code=lang_code)
            except Exception as exc:
                print(f"WARNING (Filter GUI): failed to initialise localization: {exc}")
                localizer = None
        elif localization_errors:
            print(
                "WARNING (Filter GUI): localization module not available; proceeding with defaults. "
                f"Details: {localization_errors[-1]}"
            )

        if isinstance(config_overrides, dict):
            try:
                cfg_defaults.update(config_overrides)
            except Exception:
                pass

        def _tr(key: str, default_text: Optional[str] = None, **kwargs) -> str:
            if localizer is not None:
                try:
                    # Pass default_text so JSON keys are optional
                    return localizer.get(key, default_text, **kwargs)
                except Exception:
                    pass
            # Fallback: return default_text or key placeholder
            return default_text if default_text is not None else f"_{key}_"

        # Imports kept inside to avoid import-time errors affecting pipeline
        import tkinter as tk
        from tkinter import ttk, messagebox, scrolledtext

        from core.tk_safe import patch_tk_variables

        patch_tk_variables()
        import numpy as np
        from astropy.coordinates import SkyCoord
        from astropy.io import fits
        from astropy.wcs import WCS
        from astropy.wcs.utils import pixel_to_skycoord
        import astropy.units as u
        from matplotlib.figure import Figure
        from matplotlib.patches import Polygon
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        def footprint_wh_deg(wcs_obj: Any) -> tuple[float, float]:
            """Return footprint width/height in degrees for a given WCS."""

            try:
                ny: Optional[int] = None
                nx: Optional[int] = None
                if getattr(wcs_obj, "array_shape", None):
                    ny, nx = wcs_obj.array_shape  # type: ignore[attr-defined]
                elif getattr(wcs_obj, "pixel_shape", None):
                    nx, ny = wcs_obj.pixel_shape  # type: ignore[attr-defined]
                if ny is None or nx is None:
                    return (float("nan"), float("nan"))

                px = np.array([[0, 0], [nx, 0], [nx, ny], [0, ny]], dtype=float)
                sky = pixel_to_skycoord(px[:, 0], px[:, 1], wcs_obj)
                ra = sky.ra.deg
                dec = sky.dec.deg
                ra0 = float(np.nanmedian(ra))
                dec0 = float(np.nanmedian(dec))
                cos_dec0 = float(np.cos(np.deg2rad(dec0))) if np.isfinite(dec0) else 1.0
                if abs(cos_dec0) < 1e-6:
                    cos_dec0 = 1e-6 if cos_dec0 >= 0 else -1e-6
                x = (ra - ra0) * cos_dec0
                y = dec - dec0
                width = float(np.nanmax(x) - np.nanmin(x))
                height = float(np.nanmax(y) - np.nanmin(y))
                return width, height
            except Exception:
                return (float("nan"), float("nan"))

        astrometry_mod = None
        worker_mod = None
        _import_failures: Dict[str, list[str]] = {}

        def _record_import_failure(key: str, module_name: str, exc: BaseException) -> None:
            message = f"{module_name}: {exc}"
            try:
                tb = traceback.format_exception_only(type(exc), exc)
                if tb:
                    message = f"{module_name}: {tb[-1].strip()}"
            except Exception:
                pass
            _import_failures.setdefault(key, []).append(message)

        def _import_module_with_fallback(mod_name: str):
            """Attempt to import a module regardless of package layout."""

            candidates = [mod_name]
            pkg = globals().get("__package__") or ""
            if pkg:
                candidates.append(f"{pkg}.{mod_name}")
            if not mod_name.startswith("zemosaic"):
                candidates.append(f"zemosaic.{mod_name}")

            for candidate in candidates:
                try:
                    return importlib.import_module(candidate)
                except Exception as exc:
                    _record_import_failure(mod_name, candidate, exc)
            return None

        astrometry_mod = _import_module_with_fallback('zemosaic_astrometry')
        worker_mod = _import_module_with_fallback('zemosaic_worker')
        worker_import_failure_summary = "; ".join(_import_failures.get('zemosaic_worker', []))

        solve_with_astap = getattr(astrometry_mod, 'solve_with_astap', None) if astrometry_mod else None
        extract_center_from_header_fn = getattr(astrometry_mod, 'extract_center_from_header', None) if astrometry_mod else None
        astap_fits_module = getattr(astrometry_mod, 'fits', None) if astrometry_mod else None
        astap_astropy_available = bool(getattr(astrometry_mod, 'ASTROPY_AVAILABLE_ASTROMETRY', False)) if astrometry_mod else False
        cluster_func = getattr(worker_mod, 'cluster_seestar_stacks_connected', None) if worker_mod else None
        autosplit_func = getattr(worker_mod, '_auto_split_groups', None) if worker_mod else None
        compute_dispersion_func = getattr(worker_mod, '_compute_max_angular_separation_deg', None) if worker_mod else None

        MAX_FOOTPRINTS = int(preview_cap or 0) or 3000
        cache_csv_path: Optional[str] = None
        if stream_mode and input_dir:
            cache_csv_path = os.path.join(input_dir, "headers_cache.csv")

        stream_queue: Optional[queue.Queue] = None
        # Stop flag for the streaming worker to support instant cancel/close
        stream_stop_event: Optional[threading.Event] = None
        stream_state = {
            "done": not stream_mode,
            "running": False,
            "pending_start": False,
            "status_message": None,
            "spawn_worker": None,
            "csv_loaded": False,
        }
        if cache_csv_path:
            stream_state["csv_path"] = cache_csv_path

        def _iter_fits_paths(root: str, recursive: bool = True):
            if not recursive:
                try:
                    with os.scandir(root) as it:
                        for entry in it:
                            if entry.is_file() and entry.name.lower().endswith((".fit", ".fits")):
                                yield entry.path
                except Exception:
                    return
            else:
                for r, _dirs, files in os.walk(root):
                    for fn in files:
                        if fn.lower().endswith((".fit", ".fits")):
                            yield os.path.join(r, fn)

        def _minimal_header_payload(fpath: str) -> Dict[str, Any]:
            payload: Dict[str, Any] = {"path": fpath}
            try:
                hdr = fits.getheader(fpath, 0)
            except Exception:
                return payload

            nax1 = hdr.get("NAXIS1")
            nax2 = hdr.get("NAXIS2")
            if isinstance(nax1, (int, float)) and isinstance(nax2, (int, float)):
                payload["shape"] = (int(nax2), int(nax1))

            ra = hdr.get("CRVAL1")
            dec = hdr.get("CRVAL2")
            if ra is not None and dec is not None:
                try:
                    payload["center"] = (float(ra), float(dec))
                except Exception:
                    pass

            try:
                w = WCS(hdr, naxis=2, relax=True)
                if w is not None and getattr(w, "is_celestial", False):
                    payload["wcs"] = w
            except Exception:
                pass

            keep_keys = {
                key: hdr.get(key)
                for key in (
                    "NAXIS1",
                    "NAXIS2",
                    "CRVAL1",
                    "CRVAL2",
                    "CTYPE1",
                    "CTYPE2",
                    "CDELT1",
                    "CDELT2",
                    "CD1_1",
                    "CD1_2",
                    "CD2_1",
                    "CD2_2",
                    "PC1_1",
                    "PC1_2",
                    "PC2_1",
                    "PC2_2",
                    "DATE-OBS",
                    "EXPTIME",
                    "FILTER",
                    "OBJECT",
                )
                if key in hdr
            }
            if keep_keys:
                payload["header"] = keep_keys

            return payload

        def _load_csv_bootstrap(path_csv: str) -> list[Dict[str, Any]]:
            rows: list[Dict[str, Any]] = []
            try:
                import csv

                with open(path_csv, "r", newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        entry: Dict[str, Any] = {"path": row.get("path", "")}
                        nax1 = row.get("NAXIS1")
                        nax2 = row.get("NAXIS2")
                        if nax1 and nax2:
                            try:
                                entry["shape"] = (int(nax2), int(nax1))
                            except Exception:
                                pass
                        ra_val = row.get("CRVAL1")
                        dec_val = row.get("CRVAL2")
                        if ra_val and dec_val:
                            try:
                                entry["center"] = (float(ra_val), float(dec_val))
                            except Exception:
                                pass
                        # Try to read persisted WCS footprint corners (if present)
                        try:
                            fp_ra_vals: list[float] = []
                            fp_dec_vals: list[float] = []
                            for k in ("1", "2", "3", "4"):
                                v_ra = row.get(f"FP_RA{k}")
                                v_dec = row.get(f"FP_DEC{k}")
                                if v_ra is not None and v_dec is not None and str(v_ra) != "" and str(v_dec) != "":
                                    fp_ra_vals.append(float(v_ra))
                                    fp_dec_vals.append(float(v_dec))
                            if fp_ra_vals and len(fp_ra_vals) == len(fp_dec_vals) and len(fp_ra_vals) >= 3:
                                entry["footprint_radec"] = [
                                    (fp_ra_vals[i], fp_dec_vals[i]) for i in range(len(fp_ra_vals))
                                ]
                        except Exception:
                            pass
                        header_subset = {
                            key: row.get(key)
                            for key in (
                                "NAXIS1",
                                "NAXIS2",
                                "CRVAL1",
                                "CRVAL2",
                                "DATE-OBS",
                                "EXPTIME",
                                "FILTER",
                                "OBJECT",
                            )
                            if row.get(key) is not None
                        }
                        if header_subset:
                            entry["header"] = header_subset
                        rows.append(entry)
            except Exception:
                return []
            return rows

        def _export_csv(path_csv: str, items_for_csv: list[Dict[str, Any]]) -> None:
            try:
                import csv

                fieldnames = [
                    "path",
                    "NAXIS1",
                    "NAXIS2",
                    "CRVAL1",
                    "CRVAL2",
                    # Persist up to 4 WCS footprint corners (deg)
                    "FP_RA1", "FP_DEC1",
                    "FP_RA2", "FP_DEC2",
                    "FP_RA3", "FP_DEC3",
                    "FP_RA4", "FP_DEC4",
                    "DATE-OBS",
                    "EXPTIME",
                    "FILTER",
                    "OBJECT",
                ]
                with open(path_csv, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for item in items_for_csv:
                        header_payload = item.get("header") or {}
                        shape_payload = item.get("shape")
                        nax1 = header_payload.get("NAXIS1")
                        nax2 = header_payload.get("NAXIS2")
                        if (nax1 is None or nax2 is None) and isinstance(shape_payload, (list, tuple)) and len(shape_payload) >= 2:
                            nax2 = shape_payload[0]
                            nax1 = shape_payload[1]
                        center_payload = item.get("center")
                        if hasattr(center_payload, "ra") and hasattr(center_payload, "dec"):
                            try:
                                center_tuple = (
                                    float(center_payload.ra.deg),
                                    float(center_payload.dec.deg),
                                )
                            except Exception:
                                center_tuple = (None, None)
                        else:
                            center_tuple = None
                            if isinstance(center_payload, (list, tuple)) and len(center_payload) >= 2:
                                try:
                                    center_tuple = (float(center_payload[0]), float(center_payload[1]))
                                except Exception:
                                    center_tuple = None
                        # Determine footprint corners, if available
                        fp_cols = {f"FP_RA{i}": "" for i in range(1, 5)}
                        fp_cols.update({f"FP_DEC{i}": "" for i in range(1, 5)})
                        fp_src = None
                        try:
                            pre = item.get("footprint_radec") or item.get("_precomp_fp")
                            if isinstance(pre, (list, tuple)) and len(pre) >= 3:
                                fp_src = [(float(p[0]), float(p[1])) for p in pre]
                        except Exception:
                            fp_src = None
                        if fp_src is None:
                            # Try computing from WCS if present
                            wcs_obj = item.get("wcs") if isinstance(item, dict) else None
                            if wcs_obj is not None and isinstance(shape_payload, (list, tuple)) and len(shape_payload) >= 2:
                                try:
                                    h = int(shape_payload[0]); w = int(shape_payload[1])
                                    corners = [
                                        (0.0, 0.0),
                                        (w - 1.0, 0.0),
                                        (w - 1.0, h - 1.0),
                                        (0.0, h - 1.0),
                                    ]
                                    fp_tmp: list[tuple[float, float]] = []
                                    for (x, y) in corners:
                                        sc = wcs_obj.pixel_to_world(x, y)
                                        ra = float(sc.ra.to(u.deg).value)
                                        dec = float(sc.dec.to(u.deg).value)
                                        fp_tmp.append((ra, dec))
                                    fp_src = fp_tmp
                                except Exception:
                                    fp_src = None
                        if isinstance(fp_src, list) and fp_src:
                            for i in range(min(4, len(fp_src))):
                                fp_cols[f"FP_RA{i+1}"] = fp_src[i][0]
                                fp_cols[f"FP_DEC{i+1}"] = fp_src[i][1]
                        writer.writerow(
                            {
                                "path": item.get("path", ""),
                                "NAXIS1": nax1 if nax1 is not None else "",
                                "NAXIS2": nax2 if nax2 is not None else "",
                                "CRVAL1": center_tuple[0] if center_tuple else "",
                                "CRVAL2": center_tuple[1] if center_tuple else "",
                                **fp_cols,
                                "DATE-OBS": header_payload.get("DATE-OBS", ""),
                                "EXPTIME": header_payload.get("EXPTIME", ""),
                                "FILTER": header_payload.get("FILTER", ""),
                                "OBJECT": header_payload.get("OBJECT", ""),
                            }
                        )
            except Exception as exc:
                _log_message(f"[CSV] Export failed: {exc}", level="WARN")

        initial_batches: list[list[Dict[str, Any]]] = []

        if stream_mode and input_dir:

            def _crawl_worker(target_queue: "queue.Queue[list[Dict[str, Any]] | None]",
                               stop_event: threading.Event) -> None:
                batch: list[Dict[str, Any]] = []
                minimum_batch = max(1, int(batch_size) if isinstance(batch_size, int) else 100)
                for idx, fpath in enumerate(_iter_fits_paths(input_dir, recursive=scan_recursive)):
                    # Allow cooperative, responsive cancellation
                    if stop_event.is_set():
                        break
                    item = _minimal_header_payload(fpath)
                    item["index"] = idx
                    batch.append(item)
                    if len(batch) >= minimum_batch:
                        if stop_event.is_set():
                            break
                        target_queue.put(batch)
                        batch = []
                if not stop_event.is_set():
                    if batch:
                        target_queue.put(batch)
                # Always signal completion to unblock consumers
                target_queue.put(None)
                stream_state["running"] = False

            def _spawn_worker() -> None:
                nonlocal stream_queue
                nonlocal stream_stop_event
                if stream_state.get("running"):
                    return
                stream_queue = queue.Queue()
                stream_stop_event = threading.Event()
                stream_state["running"] = True
                stream_state["done"] = False
                stream_state["status_message"] = None
                threading.Thread(
                    target=_crawl_worker,
                    args=(stream_queue, stream_stop_event),
                    daemon=True,
                ).start()

            stream_state["spawn_worker"] = _spawn_worker
            stream_state["done"] = True
            if cache_csv_path and os.path.isfile(cache_csv_path):
                bootstrap_rows = _load_csv_bootstrap(cache_csv_path)
                if bootstrap_rows:
                    initial_batches.append(bootstrap_rows)
                    stream_state["csv_loaded"] = True
                    stream_state["status_message"] = _tr(
                        "filter_status_ready_csv",
                        "Loaded from CSV. Click Analyse to refresh.",
                    )
                else:
                    stream_state["pending_start"] = True
            else:
                stream_state["pending_start"] = True
        else:
            normalized: list[Dict[str, Any]] = []
            for i, entry in enumerate(raw_items_input or []):
                try:
                    data = dict(entry)
                except Exception:
                    data = {"path": entry}
                data.setdefault("index", i)
                normalized.append(data)
            if not normalized:
                return raw_items_input, False, None
            initial_batches.append(normalized)

        # Attempt to read astropy WCS only when needed
        # (objects are provided by caller; we don't import WCS explicitly here)

        # Normalize entries to a consistent structure the GUI can use
        class Item:
            def __init__(self, src: Dict[str, Any], idx: int):
                self.src = src
                self.index: int = int(src.get("index", idx))
                # Prefer 'path', then 'path_raw', then fallback to cached path
                self.path: str = (
                    src.get("path")
                    or src.get("path_raw")
                    or src.get("path_preprocessed_cache")
                    or f"<unknown_{idx}>"
                )
                self.wcs = src.get("wcs")
                header_payload = src.get("header")
                if header_payload is None and src.get("header_subset") is not None:
                    header_payload = src.get("header_subset")
                self.header = header_payload
                self.shape: Optional[tuple[int, int]] = None
                self.center: Optional[SkyCoord] = None
                self.phase0_center: Optional[SkyCoord] = None
                # Footprint computations are quite expensive for thousands of
                # entries.  Compute them lazily only when the UI actually needs
                # the polygon to be drawn.
                self._footprint_cache: Optional[np.ndarray] = None
                self._footprint_ready: bool = False
                self.refresh_geometry()

            def _coerce_skycoord(self, value: Any) -> Optional[SkyCoord]:
                if value is None:
                    return None
                if isinstance(value, SkyCoord):
                    return value
                try:
                    if isinstance(value, (list, tuple)) and len(value) >= 2:
                        ra_deg = float(value[0])
                        dec_deg = float(value[1])
                        return SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
                    if isinstance(value, dict):
                        ra_val = value.get("ra") or value.get("RA")
                        dec_val = value.get("dec") or value.get("DEC")
                        if ra_val is not None and dec_val is not None:
                            return SkyCoord(ra=float(ra_val) * u.deg, dec=float(dec_val) * u.deg, frame="icrs")
                except Exception:
                    return None
                return None

            def _infer_shape(self) -> Optional[tuple[int, int]]:
                shp = self.src.get("shape")
                if isinstance(shp, (list, tuple)) and len(shp) >= 2:
                    try:
                        h = int(shp[0])
                        w = int(shp[1])
                        if h > 0 and w > 0:
                            return (h, w)
                    except Exception:
                        pass
                header_obj = self.header
                if header_obj is not None:
                    try:
                        getter = header_obj.get if hasattr(header_obj, "get") else header_obj.__getitem__
                        naxis1 = int(getter("NAXIS1"))
                        naxis2 = int(getter("NAXIS2"))
                        if naxis1 > 0 and naxis2 > 0:
                            return (naxis2, naxis1)
                    except Exception:
                        pass
                if getattr(self.wcs, "pixel_shape", None):
                    try:
                        nx, ny = self.wcs.pixel_shape  # type: ignore[attr-defined]
                        h = int(ny)
                        w = int(nx)
                        if h > 0 and w > 0:
                            return (h, w)
                    except Exception:
                        pass
                if getattr(self.wcs, "array_shape", None):
                    try:
                        ny, nx = self.wcs.array_shape  # type: ignore[attr-defined]
                        h = int(ny)
                        w = int(nx)
                        if h > 0 and w > 0:
                            return (h, w)
                    except Exception:
                        pass
                return None

            def _center_from_wcs(self, shape_hw: Optional[tuple[int, int]]) -> Optional[SkyCoord]:
                if self.wcs is None:
                    return None
                try:
                    if shape_hw is not None:
                        h, w = shape_hw
                        xc = (w - 1) / 2.0
                        yc = (h - 1) / 2.0
                    else:
                        crpix = getattr(self.wcs.wcs, "crpix", None)
                        if crpix is not None and len(crpix) >= 2:
                            xc, yc = float(crpix[0]), float(crpix[1])
                        else:
                            xc, yc = 1023.5, 1023.5
                    sky = self.wcs.pixel_to_world(xc, yc)
                    ra = float(sky.ra.to(u.deg).value)
                    dec = float(sky.dec.to(u.deg).value)
                    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
                except Exception:
                    return None

            def _center_from_header(self) -> Optional[SkyCoord]:
                if extract_center_from_header_fn and self.header is not None:
                    try:
                        return extract_center_from_header_fn(self.header)
                    except Exception:
                        return None
                return None

            def _build_footprint(self, shape_hw: Optional[tuple[int, int]]) -> Optional[np.ndarray]:
                if self.wcs is None or shape_hw is None:
                    return None
                try:
                    h, w = shape_hw
                    corners = [
                        (0.0, 0.0),
                        (w - 1.0, 0.0),
                        (w - 1.0, h - 1.0),
                        (0.0, h - 1.0),
                    ]
                    ras = []
                    decs = []
                    for (x, y) in corners:
                        sc = self.wcs.pixel_to_world(x, y)
                        ras.append(float(sc.ra.to(u.deg).value))
                        decs.append(float(sc.dec.to(u.deg).value))
                    return np.column_stack([np.array(ras), np.array(decs)])
                except Exception:
                    return None

            def _apply_precomputed_footprint_if_available(self) -> None:
                """Load a precomputed footprint polygon from the source dict.

                The CSV bootstrap can store footprint corners as a list of
                (RA, Dec) tuples under the ``footprint_radec`` key. When this
                data is present, use it directly so the preview can display the
                blue frames without requiring a WCS object.
                """
                try:
                    pts = self.src.get("footprint_radec")
                    if isinstance(pts, (list, tuple)) and len(pts) >= 3:
                        arr = []
                        for p in pts:
                            try:
                                arr.append([float(p[0]), float(p[1])])
                            except Exception:
                                arr = []
                                break
                        if arr:
                            self._footprint_cache = np.array(arr, dtype=float)
                            self._footprint_ready = True
                            return
                except Exception:
                    pass
                # Optional fallback: FP_RA*/FP_DEC* scalars if present
                try:
                    vals: list[list[float]] = []
                    for k in ("1", "2", "3", "4"):
                        ra = self.src.get(f"FP_RA{k}")
                        dec = self.src.get(f"FP_DEC{k}")
                        if ra is None or dec is None or str(ra) == "" or str(dec) == "":
                            continue
                        vals.append([float(ra), float(dec)])
                    if len(vals) >= 3:
                        self._footprint_cache = np.array(vals, dtype=float)
                        self._footprint_ready = True
                        return
                except Exception:
                    pass

            def get_cached_footprint(self) -> Optional[np.ndarray]:
                """Return footprint if it has already been computed."""

                if self._footprint_ready:
                    return self._footprint_cache
                return None

            def ensure_footprint(self) -> Optional[np.ndarray]:
                """Compute and cache the footprint when needed."""

                if not self._footprint_ready:
                    self._footprint_cache = self._build_footprint(self.shape)
                    # Mark as ready even when None so we don't retry endlessly
                    self._footprint_ready = True
                return self._footprint_cache

            def refresh_geometry(self) -> None:
                self.shape = self._infer_shape()
                if self.shape and self.wcs is not None and getattr(self.wcs, "pixel_shape", None) is None:
                    try:
                        self.wcs.pixel_shape = (self.shape[1], self.shape[0])
                    except Exception:
                        pass
                self.phase0_center = self._coerce_skycoord(self.src.get("phase0_center"))
                direct_center = self._coerce_skycoord(self.src.get("center"))
                center = direct_center or self.phase0_center or self._center_from_wcs(self.shape) or self._center_from_header()
                self.center = center
                if self.center is not None:
                    self.src["center"] = self.center
                if self.shape is not None:
                    self.src["shape"] = self.shape
                # Heavy footprint computations are deferred until explicitly
                # requested by the UI via ``ensure_footprint``.
                self._footprint_cache = None
                self._footprint_ready = False
                # But if a precomputed polygon is available (from CSV), use it
                # so the preview can immediately display blue frames.
                self._apply_precomputed_footprint_if_available()

        raw_files_with_wcs: list[Dict[str, Any]] = []
        items: list[Item] = []

        def _materialize_batch(batch: list[Dict[str, Any]]) -> None:
            for entry in batch:
                raw_files_with_wcs.append(entry)
                items.append(Item(entry, len(raw_files_with_wcs) - 1))

        for batch in initial_batches:
            _materialize_batch(batch)
        overrides_state: Dict[str, Any] = {}
        if isinstance(initial_overrides, dict):
            try:
                overrides_state.update(initial_overrides)
            except Exception:
                overrides_state = {}

        # Determine available sky coordinates.  Older projects or raw header
        # scans may not provide any RA/Dec information, in which case we still
        # want to display the file list even if the sky preview cannot show any
        # overlay.  Fall back to a neutral reference center so downstream logic
        # continues to work (thresholds, wrapping helpers, etc.).
        has_explicit_centers = False

        # Compute robust global center via unit-vector average
        def average_skycoord(coords: list[SkyCoord]) -> SkyCoord:
            arr = np.array([c.cartesian.xyz.value for c in coords])
            vec = arr.mean(axis=0)
            vec_norm = vec / np.linalg.norm(vec)
            sc = SkyCoord(x=vec_norm[0] * u.one, y=vec_norm[1] * u.one, z=vec_norm[2] * u.one, frame="icrs", representation_type="cartesian").spherical
            return SkyCoord(ra=sc.lon.to(u.deg), dec=sc.lat.to(u.deg), frame="icrs")

        global_center: SkyCoord = SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg, frame="icrs")
        ref_ra = float(global_center.ra.to(u.deg).value)
        ref_dec = float(global_center.dec.to(u.deg).value)

        # RA wrapping helper around a reference RA
        def wrap_ra_deg(ra_deg: float, ref_deg: float) -> float:
            x = ra_deg
            r = ref_deg
            d = x - r
            # map difference to [-180, 180)
            d = (d + 180.0) % 360.0 - 180.0
            return r + d

        # Prepare Tkinter window (use Toplevel if a main Tk exists)
        parent_root = getattr(tk, "_default_root", None)
        root_is_toplevel = False
        if parent_root is not None:
            try:
                root = tk.Toplevel(parent_root)
                root_is_toplevel = True
                try:
                    root.transient(parent_root)
                except Exception:
                    pass
                try:
                    root.grab_set()  # modal
                except Exception:
                    pass
            except Exception:
                root = tk.Tk()
        else:
            root = tk.Tk()

        root.title(_tr(
            "filter_window_title",
            "ZeMosaic - Filtrer les images WCS (optionnel)" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "ZeMosaic - Filter WCS images (optional)"
        ))
        # S'assurer que la fenÃªtre apparaÃ®t au premier plan et prend le focus
        try:
            root.lift()
            root.attributes("-topmost", True)
            root.after(200, lambda: root.attributes("-topmost", False))
            root.focus_force()
        except Exception:
            pass
        # Top-level layout: left plot, right checkboxes/actions
        main = ttk.Frame(root)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(1, weight=1)

        # Status strip (top): crawling indicator + CSV helpers
        status = ttk.Frame(main)
        status.grid(row=0, column=0, columnspan=2, sticky="ew")
        status.columnconfigure(1, weight=1)
        status_var = tk.StringVar(master=root, value=_tr("filter_status_crawling", "Crawling filesâ€¦ please wait"))
        ttk.Label(status, textvariable=status_var).grid(row=0, column=0, padx=6, pady=4, sticky="w")
        pb = ttk.Progressbar(status, mode="indeterminate", length=180)
        pb.grid(row=0, column=1, padx=6, pady=4, sticky="e")
        btn_bar = ttk.Frame(status)
        btn_bar.grid(row=0, column=2, padx=(0, 6), pady=4, sticky="e")
        analyse_btn = ttk.Button(
            btn_bar,
            text=_tr("filter_btn_analyse", "Analyse"),
            width=10,
        )
        export_btn = ttk.Button(
            btn_bar,
            text=_tr("filter_btn_export_csv", "Export CSV"),
            width=12,
        )
        analyse_btn.grid(row=0, column=0, padx=(0, 6))
        export_btn.grid(row=0, column=1)
        total_initial_entries = len(items)

        if stream_mode:
            if stream_state.get("status_message"):
                status_var.set(stream_state["status_message"])
                try:
                    pb.stop()
                except Exception:
                    pass
            elif stream_state.get("pending_start"):
                # Idle: wait for explicit click on Analyse
                try:
                    pb.stop()
                except Exception:
                    pass
                try:
                    status_var.set(
                        _tr(
                            "filter_status_click_analyse",
                            "Cliquez sur Analyse pour dÃ©marrer l'exploration." if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Ready â€” click Analyse to scan.",
                        )
                    )
                except Exception:
                    pass
            else:
                try:
                    pb.stop()
                except Exception:
                    pass
            if not cache_csv_path:
                try:
                    export_btn.state(["disabled"])
                except Exception:
                    pass
        else:
            if total_initial_entries > 400:
                try:
                    pb.start(80)
                except Exception:
                    pass
                status_var.set(
                    _tr(
                        "filter_status_populating",
                        "Preparing listâ€¦ {current}/{total}",
                    ).format(current=0, total=total_initial_entries)
                )
            else:
                try:
                    pb.stop()
                except Exception:
                    pass
                status_var.set(_tr("filter_status_ready", "Crawling done."))
            try:
                analyse_btn.state(["disabled"])
            except Exception:
                pass
            try:
                export_btn.state(["disabled"])
            except Exception:
                pass

        # Matplotlib figure
        # Use constrained_layout to reduce internal padding and let the
        # figure adapt to the available space.
        fig = Figure(figsize=(7.0, 5.0), dpi=100, constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.set_xlabel(_tr(
            "filter_axis_ra_deg",
            "AD [deg]" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "RA [deg]"
        ))
        ax.set_ylabel(_tr(
            "filter_axis_dec_deg",
            "Dec [deg]"
        ))
        ax.set_aspect("equal", adjustable="datalim")
        # For sky-like view, invert RA axis (optional)
        ax.invert_xaxis()

        canvas = FigureCanvasTkAgg(fig, master=main)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=1, column=0, sticky="nsew")

        # Make the Matplotlib figure follow the widget size to avoid
        # large empty borders around the plot.
        def _apply_resize():
            try:
                w = max(50, int(canvas_widget.winfo_width()))
                h = max(50, int(canvas_widget.winfo_height()))
                # Resize figure in pixels -> inches
                fig.set_size_inches(w / fig.dpi, h / fig.dpi, forward=True)
                # Maximize axes area inside the figure when compatible with the
                # active Matplotlib layout engine.  Newer Matplotlib versions
                # (>=3.8) expose layout engines that reject ``subplots_adjust``
                # calls.  Skip the adjustment in that case to avoid runtime
                # warnings.
                try:
                    layout_engine = None
                    if hasattr(fig, "get_layout_engine"):
                        layout_engine = fig.get_layout_engine()
                    if layout_engine is None:
                        fig.subplots_adjust(left=0.06, right=0.995, bottom=0.08, top=0.98)
                except Exception:
                    pass
                canvas.draw_idle()
            except Exception:
                pass

        _resize_job = {"id": None}

        def _on_canvas_configure(_event=None):
            # Throttle frequent resizes during interactive dragging
            try:
                if _resize_job["id"] is not None:
                    canvas_widget.after_cancel(_resize_job["id"])  # type: ignore[arg-type]
                _resize_job["id"] = canvas_widget.after(60, lambda: (_apply_resize(), _resize_job.update({"id": None})))
            except Exception:
                pass

        # Bind and trigger once to sync initial size
        try:
            canvas_widget.bind("<Configure>", _on_canvas_configure)
        except Exception:
            pass
        try:
            root.update_idletasks()
            _apply_resize()
        except Exception:
            pass

        # Enable mouse-wheel zoom on the plot for easier selection
        def _setup_wheel_zoom(ax):
            base_scale = 1.2  # zoom factor per wheel notch

            def _orient_limits(lims):
                a, b = lims
                inv = a > b
                mn, mx = (b, a) if inv else (a, b)
                return mn, mx, inv

            def _apply_limits(ax, xmin, xmax, xinv, ymin, ymax, yinv):
                if xinv:
                    ax.set_xlim(xmax, xmin)
                else:
                    ax.set_xlim(xmin, xmax)
                if yinv:
                    ax.set_ylim(ymax, ymin)
                else:
                    ax.set_ylim(ymin, ymax)

            def on_scroll(event):
                try:
                    if event is None or event.inaxes is None:
                        return
                    ax_ = event.inaxes
                    xdata = event.xdata if event.xdata is not None else sum(ax_.get_xlim()) / 2.0
                    ydata = event.ydata if event.ydata is not None else sum(ax_.get_ylim()) / 2.0

                    xmin0, xmax0, xinv = _orient_limits(ax_.get_xlim())
                    ymin0, ymax0, yinv = _orient_limits(ax_.get_ylim())
                    width = max(1e-9, (xmax0 - xmin0))
                    height = max(1e-9, (ymax0 - ymin0))

                    # Choose scale direction
                    if getattr(event, 'button', 'up') in ('up', 4):
                        # zoom in
                        scale = 1.0 / base_scale
                    else:
                        # zoom out
                        scale = base_scale

                    new_w = width * scale
                    new_h = height * scale

                    # Compute relative position of mouse within current view
                    # using oriented (min->max) extents
                    relx = (xdata - xmin0) / max(1e-12, width)
                    rely = (ydata - ymin0) / max(1e-12, height)
                    relx = min(max(relx, 0.0), 1.0)
                    rely = min(max(rely, 0.0), 1.0)

                    xmin = xdata - relx * new_w
                    xmax = xdata + (1.0 - relx) * new_w
                    ymin = ydata - rely * new_h
                    ymax = ydata + (1.0 - rely) * new_h

                    # Avoid zero-span
                    if (xmax - xmin) < 1e-9:
                        pad = 5e-10
                        xmin -= pad; xmax += pad
                    if (ymax - ymin) < 1e-9:
                        pad = 5e-10
                        ymin -= pad; ymax += pad

                    _apply_limits(ax_, xmin, xmax, xinv, ymin, ymax, yinv)
                    canvas.draw_idle()
                except Exception:
                    pass

            try:
                canvas.mpl_connect('scroll_event', on_scroll)
            except Exception:
                pass

        _setup_wheel_zoom(ax)

        # Lazy recomputation of global center once we have some items
        _center_ready = {"ok": False}

        def _maybe_update_global_center():
            nonlocal global_center, has_explicit_centers, ref_ra, ref_dec
            coords = [it.center for it in items if it.center is not None]
            has_explicit_centers = bool(coords)
            if not coords:
                return
            global_center = coords[0] if len(coords) == 1 else average_skycoord(coords)
            ref_ra = float(global_center.ra.to(u.deg).value)
            ref_dec = float(global_center.dec.to(u.deg).value)
            _center_ready["ok"] = True

        _maybe_update_global_center()

        # Right panel with controls
        right = ttk.Frame(main)
        right.grid(row=1, column=1, sticky="nsew")
        # The scrollable list lives at row=2 (row=1 reserved for clustering params)
        try:
            right.rowconfigure(2, weight=1)
        except Exception:
            right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # Threshold controls
        thresh_frame = ttk.LabelFrame(
            right,
            text=_tr(
                "filter_exclude_by_distance_title",
                "Exclure par distance au centre" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Exclude by distance to center",
            ),
        )
        thresh_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        ttk.Label(
            thresh_frame,
            text=_tr(
                "filter_distance_label",
                "Distance (deg) :" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Distance (deg):",
            ),
        ).grid(row=0, column=0, padx=4, pady=4)
        thresh_var = tk.StringVar(master=root, value="5.0")
        thresh_entry = ttk.Entry(thresh_frame, textvariable=thresh_var, width=8)
        thresh_entry.grid(row=0, column=1, padx=4, pady=4)
        def apply_threshold():
            try:
                thr = float(thresh_var.get())
            except Exception:
                messagebox.showwarning(
                    _tr(
                        "filter_invalid_value_title",
                        "Valeur invalide" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Invalid value",
                    ),
                    _tr(
                        "filter_invalid_value_message",
                        "Veuillez entrer une distance en degrÃ©s (nombre)." if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Please enter a distance in degrees (number).",
                    ),
                )
                return
            for idx, it in enumerate(items):
                if it.center is None:
                    continue
                sep = it.center.separation(global_center).to(u.deg).value
                if sep > thr:
                    _set_selected(idx, False)
            update_visuals()
        thresh_button = ttk.Button(
            thresh_frame,
            text=_tr(
                "filter_apply_threshold_button",
                "Exclure > XÂ°" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Exclude > XÂ°",
            ),
            command=apply_threshold,
        )
        thresh_button.grid(row=0, column=2, padx=4, pady=4)

        if not has_explicit_centers:
            try:
                thresh_entry.state(["disabled"])
            except Exception:
                thresh_entry.configure(state=tk.DISABLED)
            try:
                thresh_button.state(["disabled"])
            except Exception:
                pass

        # Logging pane
        log_frame = ttk.LabelFrame(
            right,
            text=_tr(
                "filter_log_panel_title",
                "Activity log" if 'fr' not in str(locals().get('lang_code', 'en')).lower() else "Journal d'activitÃ©",
            ),
        )
        log_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        log_widget = scrolledtext.ScrolledText(log_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        log_widget.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        async_events: "queue.Queue[tuple[Any, ...]]" = queue.Queue()
        resolve_state = {"running": False}

        def _enqueue_event(kind: str, *payload: Any) -> None:
            try:
                async_events.put_nowait((kind, *payload))
            except Exception:
                pass

        def _log_message(message: str, level: str = "INFO") -> None:
            try:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                log_widget.configure(state=tk.NORMAL)
                log_widget.insert(tk.END, f"[{timestamp}] [{level.upper()}] {message}\n")
                log_widget.configure(state=tk.DISABLED)
                log_widget.see(tk.END)
            except Exception:
                pass

        if not has_explicit_centers:
            _log_message(
                _tr(
                    "filter_warn_no_displayable_wcs",
                    "Aucune information WCS/centre disponible ; l'aperÃ§u restera vide mais vous pouvez sÃ©lectionner les fichiers." if 'fr' in str(locals().get('lang_code', 'en')).lower() else "No WCS/center information available; the sky preview will remain empty but you can still select files.",
                ),
                level="WARN",
            )

        def _progress_callback(
            msg: Any,
            progress: Any = None,
            lvl: str | None = None,
            **kwargs: Any,
        ) -> None:
            """Forward worker log messages without assuming a strict signature.

            Older worker callbacks only forwarded ``(message, progress, level)``
            while newer worker builds may pass additional keyword arguments used
            for GUI formatting.  Accepting ``**kwargs`` prevents ``TypeError``
            exceptions from background threads and keeps the logging pane
            functional.
            """

            if msg is None:
                return

            level = (lvl or "INFO")
            text = str(msg)

            detail_parts: list[str] = []
            if progress not in (None, ""):
                detail_parts.append(f"progress={progress}")
            if kwargs:
                detail_parts.extend(f"{key}={value}" for key, value in kwargs.items())

            if detail_parts:
                text = f"{text} ({', '.join(detail_parts)})"

            _enqueue_event("log", text, level)

        def _sanitize_path(value: Any) -> str:
            """Normalize user-provided filesystem paths.

            Configuration values may include surrounding quotes, unresolved
            environment variables (e.g. ``%PROGRAMFILES%`` on Windows) or
            user-home shortcuts.  Normalising here avoids false negatives
            when checking for ASTAP availability.
            """

            try:
                if value is None:
                    return ""
                path_str = str(value).strip()
            except Exception:
                return ""

            # Drop wrapping quotes that Windows file dialogs can preserve
            if path_str.startswith(('"', "'")) and path_str.endswith(('"', "'")) and len(path_str) >= 2:
                path_str = path_str[1:-1]

            expanded = os.path.expanduser(os.path.expandvars(path_str))
            return expanded

        astap_exe_path_raw = cfg_defaults.get('astap_executable_path', '')
        astap_data_dir_raw = cfg_defaults.get('astap_data_directory_path', '')
        astap_exe_path = _sanitize_path(astap_exe_path_raw)
        astap_data_dir = _sanitize_path(astap_data_dir_raw)

        # Keep the sanitized values in the defaults so downstream callers (log
        # messages, resolver invocations) see consistent paths.
        if astap_exe_path:
            cfg_defaults['astap_executable_path'] = astap_exe_path
        if astap_data_dir:
            cfg_defaults['astap_data_directory_path'] = astap_data_dir
        search_radius_default = cfg_defaults.get('astap_default_search_radius', 0.0)
        downsample_default = cfg_defaults.get('astap_default_downsample', 0)
        autosplit_cap_cfg = cfg_defaults.get('max_raw_per_master_tile', 0)
        try:
            autosplit_cap_cfg_int = int(autosplit_cap_cfg)
        except Exception:
            autosplit_cap_cfg_int = 0
        autosplit_cap = autosplit_cap_cfg_int if autosplit_cap_cfg_int > 0 else 50
        autosplit_cap = max(1, min(50, autosplit_cap))
        autosplit_min_cap = min(8, autosplit_cap)

        def _astap_path_available(path: str) -> bool:
            """Return True when the configured ASTAP location looks valid."""

            if not path:
                return False

            # Direct file check
            if os.path.isfile(path):
                return True

            # macOS packages are directories ending with ``.app``
            if sys.platform == "darwin" and path.lower().endswith(".app") and os.path.isdir(path):
                return True

            # Accept directories that contain the ASTAP binary
            if os.path.isdir(path):
                exe_name = "astap.exe" if os.name == "nt" else "astap"
                candidate = os.path.join(path, exe_name)
                if os.path.isfile(candidate):
                    return True

            # As a generic fallback try resolving via PATH
            resolved = shutil.which(path) or shutil.which(os.path.basename(path))
            if resolved:
                return True

            # As a generic fallback accept any existing executable entry.
            try:
                return os.path.exists(path) and os.access(path, os.X_OK)
            except Exception:
                return False

        astap_available = bool(
            solve_with_astap is not None
            and astap_astropy_available
            and _astap_path_available(astap_exe_path)
        )

        # Scrollable selection list (checkboxes for small sets, listbox for large)
        list_frame = ttk.LabelFrame(
            right,
            text=_tr(
                "filter_images_check_to_keep",
                "Images (cocher pour garder)" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Images (check to keep)",
            ),
        )
        list_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        CHECKBOX_HARD_LIMIT = 2000
        use_listbox_mode = stream_mode or total_initial_entries > CHECKBOX_HARD_LIMIT
        listbox_widget: Optional[tk.Listbox] = None
        canvas_list: Optional[tk.Canvas] = None
        inner: Optional[ttk.Frame] = None
        list_scroll_target: Optional[Any] = None

        if use_listbox_mode:
            listbox_widget = tk.Listbox(
                list_frame,
                selectmode=tk.MULTIPLE,
                exportselection=False,
                activestyle="none",
            )
            listbox_widget.grid(row=0, column=0, sticky="nsew")
            vsb = ttk.Scrollbar(list_frame, orient="vertical", command=listbox_widget.yview)
            vsb.grid(row=0, column=1, sticky="ns")
            listbox_widget.configure(yscrollcommand=vsb.set)
            list_scroll_target = listbox_widget
        else:
            canvas_list = tk.Canvas(list_frame, borderwidth=0, highlightthickness=0)
            vsb = ttk.Scrollbar(list_frame, orient="vertical", command=canvas_list.yview)
            inner = ttk.Frame(canvas_list)
            inner.bind("<Configure>", lambda e: canvas_list.configure(scrollregion=canvas_list.bbox("all")))
            canvas_list.create_window((0, 0), window=inner, anchor="nw")
            canvas_list.configure(yscrollcommand=vsb.set)
            canvas_list.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            list_scroll_target = canvas_list

        # Enable mouse-wheel scrolling over the right list (Windows/Linux/macOS)
        def _on_list_mousewheel(event):
            target = list_scroll_target
            if target is None:
                return
            try:
                if getattr(event, 'num', None) == 4:  # Linux scroll up
                    target.yview_scroll(-1, "units")
                elif getattr(event, 'num', None) == 5:  # Linux scroll down
                    target.yview_scroll(1, "units")
                else:  # Windows / macOS
                    delta = int(-1 * (event.delta / 120)) if getattr(event, 'delta', 0) else 0
                    if delta != 0:
                        target.yview_scroll(delta, "units")
            except Exception:
                pass

        def _bind_list_mousewheel(event):
            target = list_scroll_target if list_scroll_target is not None else getattr(event, "widget", None)
            if target is None:
                return
            try:
                target.bind_all("<MouseWheel>", _on_list_mousewheel)
                target.bind_all("<Button-4>", _on_list_mousewheel)
                target.bind_all("<Button-5>", _on_list_mousewheel)
            except Exception:
                pass

        def _unbind_list_mousewheel(event):
            target = list_scroll_target if list_scroll_target is not None else getattr(event, "widget", None)
            if target is None:
                return
            try:
                target.unbind_all("<MouseWheel>")
                target.unbind_all("<Button-4>")
                target.unbind_all("<Button-5>")
            except Exception:
                pass

        if list_scroll_target is not None:
            list_scroll_target.bind("<Enter>", _bind_list_mousewheel)
            list_scroll_target.bind("<Leave>", _unbind_list_mousewheel)

        selection_state: list[bool] = []
        listbox_selection_cache: set[int] = set()
        listbox_programmatic = {"active": False}

        summary_var = tk.StringVar(master=root, value="")
        resolved_counter = {"count": int(overrides_state.get("resolved_wcs_count", 0) or 0)}

        if not has_explicit_centers:
            summary_var.set(
                _tr(
                    "filter_summary_no_centers",
                    "Centres WCS indisponibles â€” sÃ©lection manuelle uniquement." if 'fr' in str(locals().get('lang_code', 'en')).lower() else "WCS centers unavailable â€” manual selection only.",
                )
            )

        operations = ttk.Frame(right)
        operations.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        operations.columnconfigure(2, weight=1)

        draw_footprints_var = tk.BooleanVar(master=root, value=True)
        write_wcs_var = tk.BooleanVar(master=root, value=True)

        resolve_btn = ttk.Button(
            operations,
            text=_tr("filter_btn_resolve_wcs", "Resolve missing WCS"),
        )
        resolve_btn.grid(row=0, column=0, padx=4, pady=2, sticky="w")

        auto_btn = ttk.Button(
            operations,
            text=_tr("filter_btn_auto_group", "Auto-organize Master Tiles"),
        )
        auto_btn.grid(row=0, column=1, padx=4, pady=2, sticky="w")

        ttk.Label(
            operations,
            textvariable=summary_var,
            anchor="w",
            justify="left",
            wraplength=260,
        ).grid(row=0, column=2, padx=4, pady=2, sticky="w")

        footprints_chk = ttk.Checkbutton(
            operations,
            text=_tr("filter_chk_draw_footprints", "Draw WCS footprints"),
            variable=draw_footprints_var,
        )
        footprints_chk.grid(row=1, column=0, columnspan=2, padx=4, pady=(2, 0), sticky="w")

        write_wcs_chk = ttk.Checkbutton(
            operations,
            text=_tr("filter_chk_write_wcs", "Write WCS to file"),
            variable=write_wcs_var,
        )
        write_wcs_chk.grid(row=1, column=2, padx=4, pady=(2, 0), sticky="w")

        if not astap_available:
            resolve_btn.state(["disabled"])
            try:
                write_wcs_chk.state(["disabled"])
            except Exception:
                pass
            _log_message(
                _tr("filter_warn_astap_missing", "ASTAP executable not configured; skipping resolution."),
                level="WARN",
            )

        if not (cluster_func and autosplit_func):
            auto_btn.state(["disabled"])
            if worker_import_failure_summary:
                _log_message(
                    _tr(
                        "filter_warn_worker_missing",
                        "Auto-grouping disabled because the processing worker module failed to load: {error}",
                        error=worker_import_failure_summary,
                    ),
                    level="ERROR",
                )

        def _resolve_missing_wcs_inplace() -> None:
            if resolve_state["running"]:
                return
            if not astap_available or solve_with_astap is None:
                _log_message(
                    _tr("filter_warn_astap_missing", "ASTAP executable not configured; skipping resolution."),
                    level="WARN",
                )
                return

            pending = [
                (idx, item)
                for idx, item in enumerate(items)
                if item.wcs is None and isinstance(item.path, str) and os.path.isfile(item.path)
            ]
            if not pending:
                msg = _tr("filter_log_no_missing_wcs", "All listed files already include a WCS solution.")
                summary_var.set(msg)
                _log_message(msg, level="INFO")
                return

            write_inplace = bool(write_wcs_var.get())
            resolve_state["running"] = True
            resolve_btn.state(["disabled"])
            summary_var.set(_tr("filter_log_resolving", "Resolving missing WCS..."))
            _log_message(_tr("filter_log_resolving", "Resolving missing WCS..."), level="INFO")

            try:
                try:
                    srch_radius = float(search_radius_default)
                    if srch_radius <= 0:
                        srch_radius = None
                except Exception:
                    srch_radius = None
                try:
                    downsample_val = int(downsample_default)
                    if downsample_val < 0:
                        downsample_val = None
                except Exception:
                    downsample_val = None

                def _resolve_worker(pairs: list[tuple[int, Any]]) -> None:
                    resolved_now = 0
                    try:
                        for idx, item in pairs:
                            path = item.path
                            header_obj = item.header
                            if header_obj is not None:
                                _enqueue_event("header_loaded", idx, header_obj)
                            elif astap_fits_module is not None and astap_astropy_available:
                                try:
                                    with astap_fits_module.open(path) as hdul_hdr:
                                        header_obj = hdul_hdr[0].header
                                        if header_obj is not None:
                                            _enqueue_event("header_loaded", idx, header_obj)
                                except Exception as exc:
                                    _enqueue_event("log", f"Failed to load FITS header for '{path}': {exc}", "ERROR")
                                    header_obj = item.header
                            try:
                                wcs_obj = solve_with_astap(
                                    path,
                                    header_obj,
                                    astap_exe_path,
                                    astap_data_dir,
                                    search_radius_deg=srch_radius,
                                    downsample_factor=downsample_val,
                                    sensitivity=None,
                                    timeout_sec=60,
                                    update_original_header_in_place=write_inplace,
                                    progress_callback=_progress_callback,
                                )
                            except Exception as exc:
                                _enqueue_event("log", f"ASTAP resolve exception: {exc}", "ERROR")
                                wcs_obj = None
                            if wcs_obj and getattr(wcs_obj, "is_celestial", False):
                                resolved_now += 1
                                _enqueue_event("resolved_item", idx, header_obj, wcs_obj)
                    finally:
                        _enqueue_event("resolve_done", resolved_now)

                threading.Thread(target=_resolve_worker, args=(pending,), daemon=True).start()
            except Exception:
                resolve_state["running"] = False
                resolve_btn.state(["!disabled"])
                raise

        def _auto_organize_master_tiles() -> None:
            if not (cluster_func and autosplit_func):
                return
            auto_btn.state(["disabled"])
            try:
                selected_indices = _get_selected_indices()
                if not selected_indices:
                    summary_text = _tr(
                        "filter_log_groups_summary",
                        "Prepared {g} group(s), sizes: {sizes}.",
                        g=0,
                        sizes="[]",
                    )
                    summary_var.set(summary_text)
                    _log_message(summary_text, level="INFO")
                    return

                class _FallbackWCS:
                    is_celestial = True

                    def __init__(self, center_coord: SkyCoord):
                        self._center = center_coord
                        self.pixel_shape = (1, 1)
                        self.array_shape = (1, 1)

                        class _Inner:
                            def __init__(self, center: SkyCoord):
                                self.crval = (
                                    float(center.ra.to(u.deg).value),
                                    float(center.dec.to(u.deg).value),
                                )
                                self.crpix = (0.5, 0.5)

                        self.wcs = _Inner(center_coord)

                    def pixel_to_world(self, _x: float, _y: float):
                        return self._center

                candidate_infos: list[dict] = []
                coord_samples: list[tuple[float, float]] = []
                for idx in selected_indices:
                    item = items[idx]
                    entry = dict(item.src)
                    if "path" not in entry:
                        entry["path"] = item.path
                    if "path_raw" not in entry and item.src.get("path_raw"):
                        entry["path_raw"] = item.src.get("path_raw")
                    if "header" not in entry:
                        entry["header"] = item.header
                    if item.shape and "shape" not in entry:
                        entry["shape"] = item.shape
                    center_obj = item.center or item.phase0_center
                    if center_obj is None and extract_center_from_header_fn and item.header is not None:
                        try:
                            center_obj = extract_center_from_header_fn(item.header)
                        except Exception:
                            center_obj = None
                    if item.wcs is None and center_obj is not None:
                        entry["wcs"] = _FallbackWCS(center_obj)
                        entry["_fallback_wcs_used"] = True
                        entry.setdefault("phase0_center", center_obj)
                        entry.setdefault("center", center_obj)
                    elif item.wcs is not None:
                        entry["wcs"] = item.wcs
                        if center_obj is not None:
                            entry.setdefault("center", center_obj)
                    if center_obj is not None:
                        coord_samples.append(
                            (
                                float(center_obj.ra.to(u.deg).value),
                                float(center_obj.dec.to(u.deg).value),
                            )
                        )
                        entry.setdefault("RA", coord_samples[-1][0])
                        entry.setdefault("DEC", coord_samples[-1][1])
                    candidate_infos.append(entry)

                if not candidate_infos:
                    summary_text = _tr(
                        "filter_log_groups_summary",
                        "Prepared {g} group(s), sizes: {sizes}.",
                        g=0,
                        sizes="[]",
                    )
                    summary_var.set(summary_text)
                    _log_message(summary_text, level="WARN")
                    return

                if coord_samples:
                    if compute_dispersion_func:
                        try:
                            dispersion_deg = float(compute_dispersion_func(coord_samples))
                        except Exception:
                            dispersion_deg = 0.0
                    else:
                        dispersion_deg = 0.0
                else:
                    dispersion_deg = 0.0

                if dispersion_deg <= 0.12:
                    threshold_deg = 0.10
                elif dispersion_deg <= 0.30:
                    threshold_deg = 0.15
                else:
                    threshold_deg = 0.05 if dispersion_deg <= 0.60 else 0.20
                threshold_deg = min(0.20, max(0.08, threshold_deg))

                groups = cluster_func(
                    candidate_infos,
                    float(threshold_deg),
                    _progress_callback,
                    orientation_split_threshold_deg=0.0,
                )
                if not groups:
                    summary_text = _tr(
                        "filter_log_groups_summary",
                        "Prepared {g} group(s), sizes: {sizes}.",
                        g=0,
                        sizes="[]",
                    )
                    summary_var.set(summary_text)
                    _log_message(summary_text, level="WARN")
                    return

                final_groups = autosplit_func(
                    groups,
                    cap=int(autosplit_cap),
                    min_cap=int(autosplit_min_cap),
                    progress_callback=_progress_callback,
                )
                final_groups = _merge_small_groups(
                    final_groups,
                    min_size=int(autosplit_min_cap),
                    cap=int(autosplit_cap),
                )
                for grp in final_groups:
                    for info in grp:
                        if info.pop("_fallback_wcs_used", False):
                            info.pop("wcs", None)
                overrides_state["preplan_master_groups"] = final_groups
                overrides_state["autosplit_cap"] = int(autosplit_cap)
                sizes = [len(gr) for gr in final_groups]
                sizes_str = ", ".join(str(s) for s in sizes) if sizes else "[]"
                summary_text = _tr(
                    "filter_log_groups_summary",
                    "Prepared {g} group(s), sizes: {sizes}.",
                    g=len(final_groups),
                    sizes=sizes_str,
                )
                summary_var.set(summary_text)
                _log_message(summary_text, level="INFO")
                try:
                    _draw_group_outlines(final_groups)
                except Exception:
                    pass
            finally:
                auto_btn.state(["!disabled"])

        resolve_btn.configure(command=_resolve_missing_wcs_inplace)
        auto_btn.configure(command=_auto_organize_master_tiles)

        # If initial overrides already include preplanned groups, draw them now
        try:
            preplanned = overrides_state.get("preplan_master_groups") if isinstance(overrides_state, dict) else None
            if preplanned:
                _draw_group_outlines(preplanned)
        except Exception:
            pass

        # Selection helpers
        actions = ttk.Frame(right)
        actions.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        def select_all():
            if use_listbox_mode:
                if selection_state:
                    listbox_programmatic["active"] = True
                    try:
                        selection_state[:] = [True] * len(selection_state)
                        listbox_selection_cache.clear()
                        listbox_selection_cache.update(range(len(selection_state)))
                        if listbox_widget is not None:
                            listbox_widget.selection_set(0, tk.END)
                    finally:
                        listbox_programmatic["active"] = False
                else:
                    pass
            else:
                for v in check_vars:
                    v.set(True)
            update_visuals()

        def select_none():
            if use_listbox_mode:
                if selection_state:
                    listbox_programmatic["active"] = True
                    try:
                        selection_state[:] = [False] * len(selection_state)
                        listbox_selection_cache.clear()
                        if listbox_widget is not None:
                            listbox_widget.selection_clear(0, tk.END)
                    finally:
                        listbox_programmatic["active"] = False
            else:
                for v in check_vars:
                    v.set(False)
            update_visuals()
        ttk.Button(
            actions,
            text=_tr(
                "filter_select_all",
                "Tout sÃ©lectionner" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Select all",
            ),
            command=select_all,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            actions,
            text=_tr(
                "filter_select_none",
                "Tout dÃ©sÃ©lectionner" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Deselect all",
            ),
            command=select_none,
        ).pack(side=tk.LEFT, padx=4)

        # Confirm/cancel buttons
        bottom = ttk.Frame(right)
        bottom.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        result: dict[str, Any] = {
            "accepted": None,
            "selected_indices": None,
            "overrides": None,
            "cancelled": False,
        }

        def _cancel_overrides_payload() -> dict[str, Any]:
            payload: dict[str, Any] = {}
            if isinstance(overrides_state, dict) and overrides_state:
                payload.update(overrides_state)
            current = result.get("overrides")
            if isinstance(current, dict) and current:
                payload.update(current)
            payload["filter_cancelled"] = True
            return payload

        def on_validate():
            _drain_stream_queue_non_blocking(mark_done=True)
            sel = _get_selected_indices()
            result["accepted"] = True
            result["selected_indices"] = sel
            result["overrides"] = overrides_state if overrides_state else None
            result["cancelled"] = False
            try:
                root.quit()
            except Exception:
                pass
            root.destroy()
        def on_cancel():
            _drain_stream_queue_non_blocking(mark_done=True)
            result["accepted"] = False
            result["selected_indices"] = None
            result["cancelled"] = True
            result["overrides"] = _cancel_overrides_payload()
            try:
                root.quit()
            except Exception:
                pass
            root.destroy()
        ttk.Button(
            bottom,
            text=_tr(
                "filter_validate",
                "Valider" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Validate",
            ),
            command=on_validate,
        ).pack(side=tk.RIGHT, padx=4)
        ttk.Button(
            bottom,
            text=_tr(
                "filter_cancel",
                "Annuler" if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Cancel",
            ),
            command=on_cancel,
        ).pack(side=tk.RIGHT, padx=4)

        # Populate checkboxes and draw footprints (supports streaming additions)
        check_vars: list[tk.BooleanVar] = []
        checkbuttons: list[Any] = []
        item_labels: list[str] = []
        patches: list[Any] = []
        center_pts: list[Any] = []  # matplotlib line2D handles
        # Master Tile outlines (group-level overlays)
        group_outline_patches: list[Any] = []

        def _clear_group_outlines() -> None:
            try:
                for art in group_outline_patches:
                    try:
                        art.remove()
                    except Exception:
                        pass
            finally:
                group_outline_patches.clear()

        def _draw_group_outlines(groups_payload: Optional[list[list[dict]]]) -> None:
            """Draw red Master Tile contours using celestial coordinates.

            For each group (list of dict entries similar to items), estimate a
            representative footprint size in degrees and render a dashed red
            rectangle centred on every available RA/Dec sample.  The rectangles
            are computed in the sky reference frame so their proportions remain
            consistent regardless of the current Matplotlib canvas scaling.
            """
            if not groups_payload:
                _clear_group_outlines()
                try:
                    ax.relim()
                except Exception:
                    pass
                try:
                    ax.autoscale_view()
                except Exception:
                    pass
                try:
                    canvas.draw_idle()
                except Exception:
                    pass
                return

            _clear_group_outlines()
            try:
                def _wcs_corners_deg(wcs_obj: Any) -> Optional[np.ndarray]:
                    try:
                        ny: Optional[int] = None
                        nx: Optional[int] = None
                        if getattr(wcs_obj, "array_shape", None):
                            ny, nx = wcs_obj.array_shape  # type: ignore[attr-defined]
                        elif getattr(wcs_obj, "pixel_shape", None):
                            nx, ny = wcs_obj.pixel_shape  # type: ignore[attr-defined]
                        if ny is None or nx is None:
                            return None
                        px = np.array(
                            [[0.0, 0.0], [float(nx), 0.0], [float(nx), float(ny)], [0.0, float(ny)]],
                            dtype=float,
                        )
                        sky = pixel_to_skycoord(px[:, 0], px[:, 1], wcs_obj)
                        ra = np.array(sky.ra.deg, dtype=float)
                        dec = np.array(sky.dec.deg, dtype=float)
                        if ra.size != 4 or dec.size != 4:
                            return None
                        return np.column_stack((ra, dec))
                    except Exception:
                        return None

                def _extend_from_points(
                    points: Any,
                    widths_acc: list[float],
                    heights_acc: list[float],
                    ra_acc: list[float],
                    dec_acc: list[float],
                ) -> None:
                    try:
                        arr = np.asarray(points, dtype=float)
                    except Exception:
                        return
                    if arr.ndim != 2 or arr.shape[1] < 2:
                        return
                    ra_vals = arr[:, 0]
                    dec_vals = arr[:, 1]
                    if ra_vals.size == 0 or dec_vals.size == 0:
                        return
                    mask = np.isfinite(ra_vals) & np.isfinite(dec_vals)
                    if not np.any(mask):
                        return
                    ra_vals = ra_vals[mask]
                    dec_vals = dec_vals[mask]
                    ra_acc.extend(float(v) for v in ra_vals.tolist())
                    dec_acc.extend(float(v) for v in dec_vals.tolist())

                    ref_ra_local = float(np.nanmedian(ra_vals))
                    ref_dec_local = float(np.nanmedian(dec_vals))
                    if not np.isfinite(ref_ra_local) or not np.isfinite(ref_dec_local):
                        return
                    cos_local = float(np.cos(np.deg2rad(ref_dec_local))) if np.isfinite(ref_dec_local) else 1.0
                    if abs(cos_local) < 1e-6:
                        cos_local = 1e-6 if cos_local >= 0 else -1e-6

                    try:
                        x_vals = [
                            (wrap_ra_deg(float(rv), ref_ra_local) - ref_ra_local) * cos_local for rv in ra_vals
                        ]
                        y_vals = [float(dv) - ref_dec_local for dv in dec_vals]
                    except Exception:
                        return

                    if x_vals:
                        span_x = float(np.nanmax(x_vals) - np.nanmin(x_vals))
                        if np.isfinite(span_x) and span_x > 0:
                            widths_acc.append(span_x)
                    if y_vals:
                        span_y = float(np.nanmax(y_vals) - np.nanmin(y_vals))
                        if np.isfinite(span_y) and span_y > 0:
                            heights_acc.append(span_y)

                for grp in groups_payload:
                    centers_wrapped: list[float] = []
                    centers_dec: list[float] = []
                    widths: list[float] = []
                    heights: list[float] = []
                    corner_ra_samples: list[float] = []
                    corner_dec_samples: list[float] = []

                    for info in (grp or []):
                        # --- collect W×H en degrés (médiane servira de taille de tuile)
                        wcs_candidates = []
                        idx_mapped = None
                        path_val = (
                            (info.get("path") or info.get("path_raw") or info.get("path_preprocessed_cache"))
                            if isinstance(info, dict)
                            else None
                        )
                        if path_val:
                            key = _path_key(path_val)
                            idx_mapped = known_path_index.get(key)
                        if idx_mapped is not None and 0 <= idx_mapped < len(items) and getattr(items[idx_mapped], "wcs", None) is not None:
                            wcs_candidates.append(items[idx_mapped].wcs)
                        if isinstance(info, dict) and info.get("wcs") is not None:
                            wcs_candidates.append(info["wcs"])
                        for wcs_obj in wcs_candidates:
                            w_deg, h_deg = footprint_wh_deg(wcs_obj)
                            if np.isfinite(w_deg) and np.isfinite(h_deg) and w_deg > 0 and h_deg > 0:
                                widths.append(w_deg)
                                heights.append(h_deg)
                            corners = _wcs_corners_deg(wcs_obj)
                            if corners is not None:
                                _extend_from_points(
                                    corners,
                                    widths,
                                    heights,
                                    corner_ra_samples,
                                    corner_dec_samples,
                                )
                            if widths and heights:
                                break

                        # --- collect centre RA/Dec
                        ra_c = None
                        dec_c = None
                        if idx_mapped is not None and 0 <= idx_mapped < len(items) and items[idx_mapped].center is not None:
                            try:
                                ra_c = float(items[idx_mapped].center.ra.to(u.deg).value)
                                dec_c = float(items[idx_mapped].center.dec.to(u.deg).value)
                            except Exception:
                                ra_c = None
                                dec_c = None
                        if (ra_c is None or dec_c is None) and isinstance(info, dict):
                            c = info.get("center")
                            try:
                                if c is not None and hasattr(c, "ra") and hasattr(c, "dec"):
                                    ra_c = float(c.ra.to(u.deg).value)
                                    dec_c = float(c.dec.to(u.deg).value)
                                elif isinstance(c, (list, tuple)) and len(c) >= 2:
                                    ra_c = float(c[0])
                                    dec_c = float(c[1])
                                elif isinstance(c, dict):
                                    ra_v = c.get("ra") or c.get("RA")
                                    dec_v = c.get("dec") or c.get("DEC")
                                    if ra_v is not None and dec_v is not None:
                                        ra_c = float(ra_v)
                                        dec_c = float(dec_v)
                            except Exception:
                                ra_c = None
                                dec_c = None
                        if (ra_c is None or dec_c is None) and isinstance(info, dict):
                            try:
                                ra_v = info.get("RA")
                                dec_v = info.get("DEC")
                                if ra_v is not None and dec_v is not None:
                                    ra_c = float(ra_v)
                                    dec_c = float(dec_v)
                            except Exception:
                                ra_c = None
                                dec_c = None

                        if ra_c is not None and dec_c is not None:
                            centers_wrapped.append(wrap_ra_deg(ra_c, ref_ra))
                            centers_dec.append(float(dec_c))

                        # --- integrate precomputed footprint polygons when WCS is absent
                        footprint_points: Optional[Any] = None
                        if idx_mapped is not None and 0 <= idx_mapped < len(items):
                            try:
                                fp_cached = items[idx_mapped].get_cached_footprint()
                                if fp_cached is None:
                                    fp_cached = items[idx_mapped].ensure_footprint()
                                footprint_points = fp_cached
                            except Exception:
                                footprint_points = None
                        if footprint_points is None and isinstance(info, dict):
                            footprint_points = info.get("footprint_radec") or info.get("_precomp_fp")
                            if footprint_points is None:
                                vals: list[tuple[float, float]] = []
                                try:
                                    for k in ("1", "2", "3", "4"):
                                        ra_val = info.get(f"FP_RA{k}")
                                        dec_val = info.get(f"FP_DEC{k}")
                                        if ra_val is None or dec_val is None:
                                            continue
                                        if str(ra_val) == "" or str(dec_val) == "":
                                            continue
                                        vals.append((float(ra_val), float(dec_val)))
                                except Exception:
                                    vals = []
                                if vals:
                                    footprint_points = vals
                        if footprint_points is not None:
                            _extend_from_points(
                                footprint_points,
                                widths,
                                heights,
                                corner_ra_samples,
                                corner_dec_samples,
                            )

                    # --- si pas de centre: passer au groupe suivant
                    if not centers_wrapped:
                        continue

                    # Taille de tuile = médiane des footprints du groupe (fallback 0.2°)
                    tile_w = float(np.median(widths)) if widths else 0.2
                    tile_h = float(np.median(heights)) if heights else 0.2

                    # Centre représentatif = médiane des centres du groupe
                    grp_ra = float(np.median(centers_wrapped))
                    grp_dec = float(np.median(centers_dec))

                    cos_dec = float(np.cos(np.deg2rad(grp_dec))) if np.isfinite(grp_dec) else 1.0
                    if abs(cos_dec) < 1e-6:  # éviter la division quasi nulle
                        cos_dec = 1e-6 if cos_dec >= 0 else -1e-6

                    if corner_ra_samples and corner_dec_samples:
                        try:
                            x_offsets: list[float] = []
                            y_offsets: list[float] = []
                            for ra_val, dec_val in zip(corner_ra_samples, corner_dec_samples):
                                delta_ra = wrap_ra_deg(float(ra_val), grp_ra) - grp_ra
                                x_offsets.append(delta_ra * cos_dec)
                                y_offsets.append(float(dec_val) - grp_dec)
                            if x_offsets and y_offsets:
                                x_min = float(np.nanmin(x_offsets))
                                x_max = float(np.nanmax(x_offsets))
                                y_min = float(np.nanmin(y_offsets))
                                y_max = float(np.nanmax(y_offsets))
                                if np.isfinite(x_min) and np.isfinite(x_max) and np.isfinite(y_min) and np.isfinite(y_max):
                                    # recentrer sur le centre du canvas combiné
                                    x_center = 0.5 * (x_min + x_max)
                                    y_center = 0.5 * (y_min + y_max)
                                    if abs(x_center) > 1e-9:
                                        grp_ra = wrap_ra_deg(grp_ra + (x_center / cos_dec), ref_ra)
                                    if abs(y_center) > 1e-9:
                                        grp_dec = grp_dec + y_center
                                    cos_dec = float(np.cos(np.deg2rad(grp_dec))) if np.isfinite(grp_dec) else 1.0
                                    if abs(cos_dec) < 1e-6:
                                        cos_dec = 1e-6 if cos_dec >= 0 else -1e-6
                                    x_offsets = []
                                    y_offsets = []
                                    for ra_val, dec_val in zip(corner_ra_samples, corner_dec_samples):
                                        delta_ra = wrap_ra_deg(float(ra_val), grp_ra) - grp_ra
                                        x_offsets.append(delta_ra * cos_dec)
                                        y_offsets.append(float(dec_val) - grp_dec)
                                    x_span = float(np.nanmax(x_offsets) - np.nanmin(x_offsets)) if x_offsets else 0.0
                                    y_span = float(np.nanmax(y_offsets) - np.nanmin(y_offsets)) if y_offsets else 0.0
                                    if np.isfinite(x_span) and x_span > 0:
                                        tile_w = max(tile_w, x_span)
                                    if np.isfinite(y_span) and y_span > 0:
                                        tile_h = max(tile_h, y_span)
                        except Exception:
                            pass

                    # Construire 1 rectangle rouge pour CE groupe
                    dx = tile_w / 2.0 / cos_dec
                    dy = tile_h / 2.0
                    ra_corners = [
                        wrap_ra_deg(grp_ra - dx, ref_ra),
                        wrap_ra_deg(grp_ra + dx, ref_ra),
                        wrap_ra_deg(grp_ra + dx, ref_ra),
                        wrap_ra_deg(grp_ra - dx, ref_ra),
                    ]
                    dec_corners = [grp_dec - dy, grp_dec - dy, grp_dec + dy, grp_dec + dy]
                    rect_pts = list(zip(ra_corners, dec_corners))

                    poly = Polygon(
                        rect_pts,
                        closed=True,
                        fill=False,
                        edgecolor="red",
                        linewidth=1.6,
                        alpha=0.9,
                        linestyle="--",
                        zorder=5,
                    )
                    ax.add_patch(poly)
                    group_outline_patches.append(poly)

            except Exception:
                _clear_group_outlines()
            finally:
                try:
                    ax.relim()
                except Exception:
                    pass
                try:
                    ax.autoscale_view()
                except Exception:
                    pass
                try:
                    canvas.draw_idle()
                except Exception:
                    pass
        # Map matplotlib artists back to item indices for click-to-select
        artist_to_index: dict[Any, int] = {}
        known_path_index: dict[str, int] = {}

        def _is_selected(idx: int) -> bool:
            if idx < 0:
                return False
            if use_listbox_mode:
                if idx < len(selection_state):
                    return bool(selection_state[idx])
                return False
            if idx < len(check_vars):
                try:
                    return bool(check_vars[idx].get())
                except Exception:
                    return False
            return False

        def _set_selected(idx: int, value: bool, *, sync_widget: bool = True) -> None:
            val = bool(value)
            if use_listbox_mode:
                if idx < 0:
                    return
                while idx >= len(selection_state):
                    selection_state.append(True)
                    listbox_selection_cache.add(len(selection_state) - 1)
                selection_state[idx] = val
                if not sync_widget:
                    if val:
                        listbox_selection_cache.add(idx)
                    else:
                        listbox_selection_cache.discard(idx)
                    return
                if listbox_widget is None:
                    if val:
                        listbox_selection_cache.add(idx)
                    else:
                        listbox_selection_cache.discard(idx)
                    return
                listbox_programmatic["active"] = True
                try:
                    if val:
                        listbox_widget.selection_set(idx)
                        listbox_selection_cache.add(idx)
                    else:
                        listbox_widget.selection_clear(idx)
                        listbox_selection_cache.discard(idx)
                except Exception:
                    pass
                finally:
                    listbox_programmatic["active"] = False
            else:
                if 0 <= idx < len(check_vars):
                    try:
                        check_vars[idx].set(val)
                    except Exception:
                        pass

        def _get_selected_indices() -> list[int]:
            if use_listbox_mode:
                return [i for i, flag in enumerate(selection_state) if flag]
            return [i for i, var in enumerate(check_vars) if var.get()]

        def _update_item_label(idx: int, text: str) -> None:
            if idx < 0 or idx >= len(item_labels):
                return
            item_labels[idx] = text
            if use_listbox_mode:
                if listbox_widget is None:
                    return
                listbox_programmatic["active"] = True
                try:
                    listbox_widget.delete(idx)
                    listbox_widget.insert(idx, text)
                    if _is_selected(idx):
                        listbox_widget.selection_set(idx)
                        listbox_selection_cache.add(idx)
                    else:
                        listbox_widget.selection_clear(idx)
                        listbox_selection_cache.discard(idx)
                except Exception:
                    pass
                finally:
                    listbox_programmatic["active"] = False
            else:
                if 0 <= idx < len(checkbuttons):
                    try:
                        checkbuttons[idx].configure(text=text)
                    except Exception:
                        pass

        def _path_key(value: Any) -> str:
            try:
                if isinstance(value, str) and value:
                    return os.path.normcase(os.path.abspath(value))
            except Exception:
                pass
            try:
                if isinstance(value, str) and value:
                    return os.path.normcase(value)
            except Exception:
                pass
            return str(value) if value is not None else ""

        def _should_draw_footprints() -> bool:
            try:
                return bool(draw_footprints_var.get())
            except Exception:
                return False

        all_ra_vals: list[float] = []
        all_dec_vals: list[float] = []
        drawn_footprints = {"count": 0}
        axes_update_pending = {"pending": False}
        population_state = {
            "next_index": 0,
            "total": total_initial_entries,
            "finalized": False,
        }
        populate_indicator = {"running": False}

        def _update_population_indicator(processed: int, *, done: bool = False) -> None:
            if stream_mode:
                return
            if done:
                try:
                    pb.stop()
                except Exception:
                    pass
                populate_indicator["running"] = False
                status_var.set(_tr("filter_status_ready", "Crawling done."))
                return
            if population_state["total"] <= 0:
                return
            if not populate_indicator["running"]:
                try:
                    pb.start(80)
                except Exception:
                    pass
                populate_indicator["running"] = True
            status_var.set(
                _tr(
                    "filter_status_populating",
                    "Preparing listâ€¦ {current}/{total}",
                ).format(current=min(processed, population_state["total"]), total=population_state["total"])
            )

        def _add_item_row(item: Item) -> None:
            idx = len(item_labels)
            base = os.path.basename(item.path)
            sep_txt = ""
            if item.center is not None:
                try:
                    sep_deg = item.center.separation(global_center).to(u.deg).value
                    sep_txt = f"  ({sep_deg:.2f}Â°)"
                except Exception:
                    sep_txt = ""

            display_text = base + sep_txt
            item_labels.append(display_text)

            if use_listbox_mode:
                selection_state.append(True)
                listbox_selection_cache.add(idx)
                if listbox_widget is not None:
                    listbox_programmatic["active"] = True
                    try:
                        listbox_widget.insert(tk.END, display_text)
                        listbox_widget.selection_set(idx)
                    except Exception:
                        pass
                    finally:
                        listbox_programmatic["active"] = False
            else:
                var = tk.BooleanVar(master=root, value=True)
                check_vars.append(var)
                if inner is not None:
                    cb = ttk.Checkbutton(inner, text=display_text, variable=var, command=lambda i=idx: update_visuals(i))
                    cb.pack(anchor="w", fill="x", pady=1)
                    checkbuttons.append(cb)
                else:
                    checkbuttons.append(None)

            path_key = _path_key(item.path)
            if path_key:
                known_path_index[path_key] = idx

            # Ensure placeholders exist for visuals
            patches.append(None)
            center_pts.append(None)

            color_sel = "tab:blue"

            local_ra_vals: list[float] = []
            local_dec_vals: list[float] = []

            should_draw = _should_draw_footprints()
            footprint_to_draw: Optional[np.ndarray] = None
            if should_draw and drawn_footprints["count"] < MAX_FOOTPRINTS:
                footprint_to_draw = item.ensure_footprint()
            elif should_draw:
                footprint_to_draw = item.get_cached_footprint()

            if should_draw and footprint_to_draw is not None and drawn_footprints["count"] < MAX_FOOTPRINTS:
                try:
                    ra_wrapped = [wrap_ra_deg(float(ra), ref_ra) for ra in footprint_to_draw[:, 0].tolist()]
                    decs = footprint_to_draw[:, 1].tolist()
                    poly = Polygon(
                        list(zip(ra_wrapped, decs)),
                        closed=True,
                        fill=False,
                        edgecolor=color_sel,
                        linewidth=1.0,
                        alpha=0.9,
                    )
                    try:
                        poly.set_picker(True)
                    except Exception:
                        pass
                    ax.add_patch(poly)
                    patches[idx] = poly
                    artist_to_index[poly] = idx
                    local_ra_vals.extend(ra_wrapped)
                    local_dec_vals.extend(decs)
                    drawn_footprints["count"] += 1
                except Exception:
                    patches[idx] = None
            elif item.center is not None:
                try:
                    ra_c = wrap_ra_deg(float(item.center.ra.to(u.deg).value), ref_ra)
                    dec_c = float(item.center.dec.to(u.deg).value)
                    ln, = ax.plot(
                        [ra_c],
                        [dec_c],
                        marker="o",
                        markersize=3,
                        color=color_sel,
                        alpha=0.9,
                        picker=8,
                    )
                    center_pts[idx] = ln
                    artist_to_index[ln] = idx
                    local_ra_vals.append(ra_c)
                    local_dec_vals.append(dec_c)
                except Exception:
                    center_pts[idx] = None

            all_ra_vals.extend(local_ra_vals)
            all_dec_vals.extend(local_dec_vals)
        ax.grid(True, which="both", linestyle=":", linewidth=0.6)

        def update_visuals(changed_index: Optional[Iterable[int] | int] = None) -> None:
            """Refresh matplotlib artists to match the checkbox selection state."""

            if changed_index is None:
                target_indices: Iterable[int] = range(len(items))
            elif isinstance(changed_index, Iterable) and not isinstance(changed_index, (str, bytes)):
                target_indices = changed_index
            else:
                target_indices = (changed_index,)

            any_updated = False
            for raw_idx in target_indices:
                try:
                    i = int(raw_idx)
                except (TypeError, ValueError):
                    continue
                if i < 0 or i >= len(items):
                    continue
                selected = _is_selected(i)
                col = "tab:blue" if selected else "0.7"
                alp = 0.9 if selected else 0.3
                if i < len(patches) and patches[i] is not None:
                    patches[i].set_edgecolor(col)
                    patches[i].set_alpha(alp)
                    any_updated = True
                if i < len(center_pts) and center_pts[i] is not None:
                    center_pts[i].set_color(col)
                    center_pts[i].set_alpha(alp)
                    any_updated = True
            if any_updated:
                canvas.draw_idle()

        if use_listbox_mode and listbox_widget is not None:
            def _on_listbox_select(_event: Any) -> None:
                if listbox_programmatic.get("active"):
                    return
                try:
                    current = {int(idx) for idx in listbox_widget.curselection()}
                except Exception:
                    return
                added = current - listbox_selection_cache
                removed = listbox_selection_cache - current
                if not added and not removed:
                    return
                for idx in added:
                    if 0 <= idx < len(selection_state):
                        selection_state[idx] = True
                for idx in removed:
                    if 0 <= idx < len(selection_state):
                        selection_state[idx] = False
                listbox_selection_cache.clear()
                listbox_selection_cache.update(current)
                update_visuals(list(added | removed))

            listbox_widget.bind("<<ListboxSelect>>", _on_listbox_select)

        def _apply_initial_axes() -> None:
            if all_ra_vals and all_dec_vals:
                ra_min, ra_max = min(all_ra_vals), max(all_ra_vals)
                dec_min, dec_max = min(all_dec_vals), max(all_dec_vals)
                ra_pad = max(1e-3, (ra_max - ra_min) * 0.05 + 0.2)
                dec_pad = max(1e-3, (dec_max - dec_min) * 0.05 + 0.2)
                ax.set_xlim(ra_max + ra_pad, ra_min - ra_pad)
                ax.set_ylim(dec_min - dec_pad, dec_max + dec_pad)
                canvas.draw_idle()

        def _finalize_initial_population() -> None:
            if population_state["finalized"]:
                return
            population_state["finalized"] = True
            update_visuals()
            _apply_initial_axes()
            _update_population_indicator(population_state["total"], done=True)

        def _populate_initial_chunk() -> None:
            total = population_state["total"]
            if total <= 0:
                _finalize_initial_population()
                return
            start = population_state["next_index"]
            if start >= total:
                _finalize_initial_population()
                return
            # Load a larger synchronous chunk first, then smaller async batches
            chunk = 0
            if start == 0:
                chunk = min(total, 400)
            else:
                chunk = min(total - start, 200)
            end = start + chunk
            for idx in range(start, end):
                _add_item_row(items[idx])
            population_state["next_index"] = end
            _update_population_indicator(end, done=end >= total)
            try:
                canvas.draw_idle()
            except Exception:
                pass
            if end >= total:
                _finalize_initial_population()
            else:
                # Yield to Tkinter before continuing with the next batch
                try:
                    root.after(15, _populate_initial_chunk)
                except Exception:
                    # If scheduling fails, continue synchronously to avoid losing items
                    _populate_initial_chunk()

        _populate_initial_chunk()

        def _recompute_axes_limits() -> None:
            ra_vals: list[float] = []
            dec_vals: list[float] = []
            for it in items:
                footprint = it.get_cached_footprint()
                if footprint is not None:
                    for ra in footprint[:, 0].tolist():
                        ra_vals.append(wrap_ra_deg(float(ra), ref_ra))
                    dec_vals.extend(footprint[:, 1].tolist())
                elif it.center is not None:
                    ra_vals.append(wrap_ra_deg(float(it.center.ra.to(u.deg).value), ref_ra))
                    dec_vals.append(float(it.center.dec.to(u.deg).value))
            if ra_vals and dec_vals:
                ra_min, ra_max = min(ra_vals), max(ra_vals)
                dec_min, dec_max = min(dec_vals), max(dec_vals)
                ra_pad = max(1e-3, (ra_max - ra_min) * 0.05 + 0.2)
                dec_pad = max(1e-3, (dec_max - dec_min) * 0.05 + 0.2)
                ax.set_xlim(ra_max + ra_pad, ra_min - ra_pad)
                ax.set_ylim(dec_min - dec_pad, dec_max + dec_pad)
                canvas.draw_idle()

        def _schedule_axes_update() -> None:
            if axes_update_pending["pending"]:
                return

            def _do_update() -> None:
                axes_update_pending["pending"] = False
                _recompute_axes_limits()

            axes_update_pending["pending"] = True
            try:
                root.after_idle(_do_update)
            except Exception:
                _do_update()

        def _ingest_batch(batch: list[Dict[str, Any]]) -> None:
            if not batch:
                return
            new_indices: list[int] = []
            for entry in batch:
                candidate_path = entry.get("path") or entry.get("path_raw") or entry.get("path_preprocessed_cache")
                key = _path_key(candidate_path)
                if key and key in known_path_index:
                    existing_idx = known_path_index[key]
                    entry.setdefault("index", existing_idx)
                    try:
                        raw_files_with_wcs[existing_idx].update(entry)
                    except Exception:
                        raw_files_with_wcs[existing_idx] = dict(entry)
                    try:
                        items[existing_idx].src.update(entry)
                    except Exception:
                        items[existing_idx].src = dict(entry)
                    try:
                        items[existing_idx].index = existing_idx
                        items[existing_idx].refresh_geometry()
                    except Exception:
                        pass
                    try:
                        base_name = os.path.basename(items[existing_idx].path)
                        sep_txt = ""
                        if items[existing_idx].center is not None:
                            sep_deg = items[existing_idx].center.separation(global_center).to(u.deg).value
                            sep_txt = f"  ({sep_deg:.2f}Â°)"
                        _update_item_label(existing_idx, base_name + sep_txt)
                    except Exception:
                        pass
                    try:
                        _refresh_item_visual(existing_idx)
                    except Exception:
                        pass
                    continue
                raw_files_with_wcs.append(entry)
                new_index = len(raw_files_with_wcs) - 1
                entry.setdefault("index", new_index)
                new_item = Item(entry, new_index)
                items.append(new_item)
                _add_item_row(new_item)
                new_indices.append(new_index)
            if new_indices:
                update_visuals(new_indices)
            if not _center_ready["ok"]:
                _maybe_update_global_center()
                if _center_ready["ok"] and has_explicit_centers:
                    try:
                        thresh_entry.state(["!disabled"])
                    except Exception:
                        thresh_entry.configure(state=tk.NORMAL)
                    try:
                        thresh_button.state(["!disabled"])
                    except Exception:
                        pass
            if new_indices:
                _schedule_axes_update()

        def _consume_ui_queue() -> None:
            if stream_queue is None or stream_state.get("done"):
                return
            try:
                while True:
                    batch = stream_queue.get_nowait()
                    if batch is None:
                        stream_state["done"] = True
                        stream_state["running"] = False
                        try:
                            pb.stop()
                            status_var.set(_tr("filter_status_ready", "Crawling done."))
                        except Exception:
                            pass
                        try:
                            _log_message(
                                _tr(
                                    "filter_log_analysis_complete",
                                    "Analyse terminee." if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Analysis complete.",
                                ),
                                level="INFO",
                            )
                        except Exception:
                            pass
                        stream_state["status_message"] = _tr("filter_status_ready", "Crawling done.")
                        try:
                            analyse_btn.state(["!disabled"])
                        except Exception:
                            pass
                        break
                    if batch:
                        _ingest_batch(batch)
            except queue.Empty:
                pass
            finally:
                if not stream_state.get("done"):
                    try:
                        root.after(40, _consume_ui_queue)
                    except Exception:
                        stream_state["done"] = True

        if stream_queue is not None and not stream_state.get("done"):
            try:
                root.after(40, _consume_ui_queue)
            except Exception:
                stream_state["done"] = True

        def _drain_stream_queue_non_blocking(mark_done: bool = False) -> None:
            """Drain any queued batches without waiting.

            If ``mark_done`` is True, also request the worker to stop and
            prevent re-scheduling of consumers. This makes Cancel/close
            instantaneous even on slow USB drives.
            """
            nonlocal stream_stop_event
            if stream_queue is None:
                return
            if mark_done:
                stream_state["done"] = True
                if stream_stop_event is not None:
                    try:
                        stream_stop_event.set()
                    except Exception:
                        pass
            drained_any = False
            while True:
                try:
                    batch = stream_queue.get_nowait()
                except queue.Empty:
                    break
                drained_any = True
                if batch is None:
                    stream_state["done"] = True
                    stream_state["running"] = False
                    try:
                        pb.stop()
                        status_var.set(_tr("filter_status_ready", "Crawling done."))
                    except Exception:
                        pass
                    try:
                        _log_message(
                            _tr(
                                "filter_log_analysis_complete",
                                "Analyse terminee." if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Analysis complete.",
                            ),
                            level="INFO",
                        )
                    except Exception:
                        pass
                    stream_state["status_message"] = _tr("filter_status_ready", "Crawling done.")
                    try:
                        analyse_btn.state(["!disabled"])
                    except Exception:
                        pass
                    break
                if batch:
                    _ingest_batch(batch)
            # If we consumed something and we're not marked done, schedule a light pass
            if drained_any and not stream_state.get("done"):
                try:
                    root.after(40, _consume_ui_queue)
                except Exception:
                    stream_state["done"] = True

        def _refresh_item_visual(idx: int) -> None:
            if idx < 0 or idx >= len(items):
                return
            item = items[idx]
            prev_patch = patches[idx] if idx < len(patches) else None
            prev_point = center_pts[idx] if idx < len(center_pts) else None
            if prev_patch is not None:
                try:
                    prev_patch.remove()
                except Exception:
                    pass
                artist_to_index.pop(prev_patch, None)
                patches[idx] = None
                drawn_footprints["count"] = max(0, drawn_footprints["count"] - 1)
            if prev_point is not None:
                try:
                    prev_point.remove()
                except Exception:
                    pass
                artist_to_index.pop(prev_point, None)
                center_pts[idx] = None
            selected = _is_selected(idx)
            color_sel = "tab:blue" if selected else "0.7"
            alpha_val = 0.9 if selected else 0.3
            new_patch = None
            new_point = None
            should_draw = _should_draw_footprints()
            footprint = item.get_cached_footprint() if should_draw else None
            allow_draw = should_draw and drawn_footprints["count"] < MAX_FOOTPRINTS
            if allow_draw and (footprint is not None or item.wcs is not None):
                try:
                    if footprint is None:
                        footprint = item.ensure_footprint()
                    if footprint is None:
                        raise ValueError("no footprint")
                    ra_wrapped = [wrap_ra_deg(float(ra), ref_ra) for ra in footprint[:, 0].tolist()]
                    decs = footprint[:, 1].tolist()
                    new_patch = Polygon(
                        list(zip(ra_wrapped, decs)),
                        closed=True,
                        fill=False,
                        edgecolor=color_sel,
                        linewidth=1.0,
                        alpha=alpha_val,
                    )
                    new_patch.set_picker(True)
                    ax.add_patch(new_patch)
                    artist_to_index[new_patch] = idx
                    patches[idx] = new_patch
                    drawn_footprints["count"] += 1
                except Exception:
                    new_patch = None
            elif item.center is not None:
                try:
                    ra_c = wrap_ra_deg(float(item.center.ra.to(u.deg).value), ref_ra)
                    dec_c = float(item.center.dec.to(u.deg).value)
                    new_point, = ax.plot([ra_c], [dec_c], marker="o", markersize=3, color=color_sel, alpha=alpha_val, picker=8)
                    artist_to_index[new_point] = idx
                    center_pts[idx] = new_point
                except Exception:
                    new_point = None
            update_visuals(idx)
            _schedule_axes_update()

        def _on_draw_mode_change() -> None:
            for index in range(len(items)):
                try:
                    _refresh_item_visual(index)
                except Exception:
                    pass

        try:
            footprints_chk.configure(command=_on_draw_mode_change)
        except Exception:
            pass

        def _trigger_stream_start(force: bool = False) -> None:
            if not stream_mode:
                return
            spawn_callable = stream_state.get("spawn_worker")
            if not callable(spawn_callable):
                return
            if stream_state.get("running"):
                return
            if not force and not stream_state.get("pending_start"):
                return
            stream_state["pending_start"] = False
            stream_state["status_message"] = _tr("filter_status_crawling", "Crawling filesâ€¦ please wait")
            try:
                status_var.set(stream_state["status_message"])
            except Exception:
                pass
            try:
                pb.start(80)
            except Exception:
                pass
            try:
                analyse_btn.state(["disabled"])
            except Exception:
                pass
            spawn_callable()
            try:
                root.after(40, _consume_ui_queue)
            except Exception:
                stream_state["done"] = True

        def _on_analyse() -> None:
            if not stream_mode:
                return
            _trigger_stream_start(force=True)

        def _on_export() -> None:
            path_csv = stream_state.get("csv_path") if isinstance(stream_state.get("csv_path"), str) else cache_csv_path
            if not path_csv:
                return
            try:
                os.makedirs(os.path.dirname(path_csv), exist_ok=True)
            except Exception:
                pass
            payload: list[Dict[str, Any]] = []
            for it in items:
                data = dict(it.src)
                if it.shape is not None:
                    data.setdefault("shape", it.shape)
                if it.center is not None:
                    data.setdefault("center", it.center)
                # If we have a computed footprint (from WCS or CSV), pass it
                # along explicitly so the exporter can persist it.
                try:
                    fp = it.get_cached_footprint() or it.ensure_footprint()
                    if fp is not None:
                        # Convert ndarray to list of tuples for serialization
                        data.setdefault("_precomp_fp", [(float(r), float(d)) for r, d in fp.tolist()])
                except Exception:
                    pass
                payload.append(data)
            _export_csv(path_csv, payload)
            stream_state["csv_loaded"] = True
            stream_state["status_message"] = _tr(
                "filter_status_ready_csv",
                "Loaded from CSV. Click Analyse to refresh.",
            )
            try:
                status_var.set(stream_state["status_message"])
            except Exception:
                pass
            _log_message(f"[CSV] Exported {len(payload)} items -> {path_csv}", level="INFO")

        analyse_btn.configure(command=_on_analyse)
        export_btn.configure(command=_on_export)
        if stream_mode:
            try:
                analyse_btn.state(["!disabled"])
            except Exception:
                pass
            if stream_state.get("pending_start"):
                # Do not auto-start; wait for user to click Analyse
                try:
                    pb.stop()
                    status_var.set(
                        _tr(
                            "filter_status_click_analyse",
                            "Cliquez sur Analyse pour dÃ©marrer l'exploration." if 'fr' in str(locals().get('lang_code', 'en')).lower() else "Ready â€” click Analyse to scan.",
                        )
                    )
                except Exception:
                    pass
            elif stream_state.get("status_message"):
                try:
                    analyse_btn.state(["!disabled"])
                except Exception:
                    pass

        def _process_async_events() -> None:
            try:
                while True:
                    event = async_events.get_nowait()
                    kind = event[0]
                    if kind == "log":
                        _, message, level = event
                        _log_message(str(message), level=str(level))
                    elif kind == "header_loaded":
                        _, idx, header_obj = event
                        if (
                            isinstance(idx, int)
                            and 0 <= idx < len(items)
                            and header_obj is not None
                        ):
                            item = items[idx]
                            item.header = header_obj
                            item.src["header"] = header_obj
                    elif kind == "resolved_item":
                        _, idx, header_obj, wcs_obj = event
                        if isinstance(idx, int) and 0 <= idx < len(items):
                            item = items[idx]
                            if header_obj is not None:
                                item.header = header_obj
                                item.src["header"] = header_obj
                            if wcs_obj is not None and getattr(wcs_obj, "is_celestial", False):
                                item.src["wcs"] = wcs_obj
                                item.wcs = wcs_obj
                                try:
                                    item.refresh_geometry()
                                except Exception:
                                    pass
                                _refresh_item_visual(idx)
                    elif kind == "resolve_done":
                        _, resolved_now = event
                        try:
                            resolved_count = int(resolved_now)
                        except Exception:
                            resolved_count = 0
                        if resolved_count >= 0:
                            resolved_counter["count"] += resolved_count
                            if resolved_count > 0:
                                overrides_state["resolved_wcs_count"] = resolved_counter["count"]
                        summary_msg = _tr(
                            "filter_log_resolved_n",
                            "Resolved WCS for {n} files.",
                            n=resolved_count,
                        )
                        summary_var.set(summary_msg)
                        _log_message(summary_msg, level="INFO")
                        if resolved_count > 0:
                            _schedule_axes_update()
                        resolve_state["running"] = False
                        try:
                            resolve_btn.state(["!disabled"])
                        except Exception:
                            pass
                    else:
                        pass
            except queue.Empty:
                pass
            finally:
                try:
                    root.after(150, _process_async_events)
                except Exception:
                    pass

        _process_async_events()

        # Click-to-select/deselect via matplotlib pick events
        def _on_pick(event):
            try:
                artist = getattr(event, 'artist', None)
                if artist is None:
                    return
                i = artist_to_index.get(artist)
                if i is None:
                    return
                # Toggle associated checkbox and refresh colors
                curr = _is_selected(i)
                _set_selected(i, not curr)
                update_visuals(i)
            except Exception:
                pass

        try:
            canvas.mpl_connect('pick_event', _on_pick)
        except Exception:
            pass

        # On window close: treat as cancel (keep all)
        def on_close():
            _drain_stream_queue_non_blocking(mark_done=True)
            if result.get("accepted") is None:
                result["accepted"] = False
            result["selected_indices"] = None
            result["cancelled"] = True
            result["overrides"] = _cancel_overrides_payload()
            root.destroy()
        root.protocol("WM_DELETE_WINDOW", on_close)

        # Start modal loop (use wait_window for Toplevel)
        try:
            if root_is_toplevel:
                parent = parent_root if parent_root is not None else root
                try:
                    parent.wait_window(root)
                except Exception:
                    # Fallback to simple update loop
                    try:
                        root.update()
                        root.update_idletasks()
                    except Exception:
                        pass
            else:
                root.mainloop()
        except KeyboardInterrupt:
            # If interrupted, keep default behavior (keep all)
            pass
        except Exception:
            # Some environments (notably on Windows when running from a
            # multiprocessing daemon) occasionally fail to enter Tk's
            # mainloop, leaving the window unresponsive.  Fall back to a
            # manual event pump so the user can still interact with the UI.
            try:
                deadline = time.monotonic() + 3600.0  # 1 hour safety cap
                while True:
                    try:
                        root.update_idletasks()
                        root.update()
                    except Exception:
                        break
                    # Exit when the window is destroyed or a decision was made
                    if not root.winfo_exists() or result.get("accepted") is not None:
                        break
                    if time.monotonic() > deadline:
                        break
                    time.sleep(0.05)
                try:
                    if root.winfo_exists():
                        root.destroy()
                except Exception:
                    pass
            except Exception:
                pass

        # Return selection (and move unselected files into 'filtered_by_user')
        accepted_flag = result.get("accepted")
        if accepted_flag is None:
            accepted_flag = False
        cancelled_flag = bool(result.get("cancelled"))
        if cancelled_flag:
            overrides_payload: dict[str, Any] | None
            overrides_payload = None
            overrides_result = result.get("overrides")
            if isinstance(overrides_result, dict):
                overrides_payload = dict(overrides_result)
            elif isinstance(overrides_state, dict) and overrides_state:
                overrides_payload = dict(overrides_state)
            if overrides_payload is None:
                overrides_payload = {}
            overrides_payload["filter_cancelled"] = True
            return raw_files_with_wcs, False, overrides_payload
        if accepted_flag and isinstance(result.get("selected_indices"), list):
            sel = result["selected_indices"]  # type: ignore[assignment]

            # Compute unselected indices
            total_n = len(raw_files_with_wcs)
            unselected_indices = [i for i in range(total_n) if i not in sel]
            if unselected_indices:
                overrides_state["filter_excluded_indices"] = unselected_indices

            # Prepare destination folder under the common input directory
            def _preferred_src_path(entry: Dict[str, Any]) -> Optional[str]:
                # Prefer raw/original paths if available
                p = entry.get("path_raw") or entry.get("path")
                if isinstance(p, str):
                    return p
                return None

            excluded_paths: list[str] = []
            all_src_dirs: list[str] = []
            for i in unselected_indices:
                p = _preferred_src_path(raw_files_with_wcs[i])
                if p and os.path.isfile(p):
                    excluded_paths.append(p)
                    all_src_dirs.append(os.path.dirname(p))

            dest_base: Optional[str] = None
            if all_src_dirs:
                try:
                    dest_base = os.path.commonpath(all_src_dirs)
                except Exception:
                    dest_base = all_src_dirs[0]

            # If we have a base, move excluded files to '<base>/filtered_by_user'
            if dest_base is not None and excluded_paths:
                dest_dir = os.path.join(dest_base, "filtered_by_user")
                try:
                    os.makedirs(dest_dir, exist_ok=True)
                except Exception:
                    dest_dir = None  # will fallback to per-file folder

                def _unique_dest(path_dir: str, filename: str) -> str:
                    base, ext = os.path.splitext(filename)
                    candidate = os.path.join(path_dir, filename)
                    if not os.path.exists(candidate):
                        return candidate
                    # Append timestamp, then counter if still colliding
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    candidate = os.path.join(path_dir, f"{base}_{ts}{ext}")
                    if not os.path.exists(candidate):
                        return candidate
                    k = 1
                    while True:
                        candidate = os.path.join(path_dir, f"{base}_{ts}_{k}{ext}")
                        if not os.path.exists(candidate):
                            return candidate
                        k += 1

                # Perform moves
                for src_path in excluded_paths:
                    try:
                        target_dir = dest_dir
                        if target_dir is None:
                            # As fallback, move next to its source directory under a local 'filtered_by_user'
                            local_dir = os.path.join(os.path.dirname(src_path), "filtered_by_user")
                            os.makedirs(local_dir, exist_ok=True)
                            target_dir = local_dir
                        dest_path = _unique_dest(target_dir, os.path.basename(src_path))
                        shutil.move(src_path, dest_path)
                    except Exception as e:
                        # Non-fatal: keep going
                        print(f"WARN filter_gui: Failed to move '{src_path}' -> filtered_by_user: {e}")

            return [raw_files_with_wcs[i] for i in sel], True, (overrides_state if overrides_state else None)
        else:
            # Keep all if canceled or closed
            return raw_files_with_wcs, False, None

    except ImportError:
        # Any optional dependency missing â€” silently keep all
        return raw_files_with_wcs, False, None
    except Exception:
        # Any unexpected error â€” fail safe and keep all
        return raw_files_with_wcs, False, None


__all__ = ["launch_filter_interface"]

