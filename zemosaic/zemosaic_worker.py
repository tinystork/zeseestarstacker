# zemosaic_worker.py

import os
import shutil
import time
import traceback
import gc
import logging
import inspect  # Pas utilisé directement ici, mais peut être utile pour des introspections futures
import psutil
import tempfile
import glob
import uuid
import multiprocessing
from typing import Callable
from types import SimpleNamespace

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
# BrokenProcessPool moved under concurrent.futures.process in modern Python
from concurrent.futures.process import BrokenProcessPool


# --- Configuration du Logging ---
logger = logging.getLogger("ZeMosaicWorker")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    try:
        log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zemosaic_worker.log")
    except NameError: 
        log_file_path = "zemosaic_worker.log"
    fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
logger.info("Logging pour ZeMosaicWorker initialisé. Logs écrits dans: %s", log_file_path)

# --- Third-Party Library Imports ---
import numpy as np
import zarr
from packaging.version import Version

try:
    from zarr.storage import LRUStoreCache
    if Version(zarr.__version__).major >= 3:
        # In zarr>=3 LRUStoreCache was removed. Use a no-op wrapper
        raise ImportError
except Exception:  # pragma: no cover - fallback for zarr>=3 or missing cache
    class LRUStoreCache:
        """Simple pass-through wrapper used when LRUStoreCache is unavailable."""

        def __init__(self, store, max_size=None):
            self.store = store

        def __getattr__(self, name):
            return getattr(self.store, name)

try:
    # Prefer storage module first (zarr < 3)
    from zarr.storage import DirectoryStore
except Exception:
    try:  # pragma: no cover - zarr >= 3 uses LocalStore
        from zarr.storage import LocalStore as DirectoryStore
    except Exception:
        try:
            from zarr.storage import FsspecStore
            import fsspec

            def DirectoryStore(path):
                return FsspecStore(fsspec.filesystem("file").get_mapper(path))
        except Exception:  # pragma: no cover - ultimate fallback
            DirectoryStore = None

# now LRUStoreCache and DirectoryStore are defined


# --- Astropy (critique) ---
ASTROPY_AVAILABLE = False
WCS, SkyCoord, Angle, fits, u = None, None, None, None, None
try:
    from astropy.io import fits as actual_fits
    from astropy.wcs import WCS as actual_WCS
    from astropy.coordinates import SkyCoord as actual_SkyCoord, Angle as actual_Angle
    from astropy import units as actual_u
    fits, WCS, SkyCoord, Angle, u = actual_fits, actual_WCS, actual_SkyCoord, actual_Angle, actual_u
    ASTROPY_AVAILABLE = True
    logger.info("Bibliothèque Astropy importée.")
except ImportError as e_astro_imp: logger.critical(f"Astropy non trouvée: {e_astro_imp}.")
except Exception as e_astro_other_imp: logger.critical(f"Erreur import Astropy: {e_astro_other_imp}", exc_info=True)

# --- Reproject (critique pour la mosaïque) ---
REPROJECT_AVAILABLE = False
find_optimal_celestial_wcs, reproject_and_coadd, reproject_interp = None, None, None
try:
    from reproject.mosaicking import find_optimal_celestial_wcs as actual_find_optimal_wcs
    from reproject.mosaicking import reproject_and_coadd as actual_reproject_coadd
    from reproject import reproject_interp as actual_reproject_interp
    find_optimal_celestial_wcs, reproject_and_coadd, reproject_interp = actual_find_optimal_wcs, actual_reproject_coadd, actual_reproject_interp
    REPROJECT_AVAILABLE = True
    logger.info("Bibliothèque 'reproject' importée.")
except ImportError as e_reproject_final: logger.critical(f"Échec import reproject: {e_reproject_final}.")
except Exception as e_reproject_other_final: logger.critical(f"Erreur import 'reproject': {e_reproject_other_final}", exc_info=True)

# --- Local Project Module Imports ---
zemosaic_utils, ZEMOSAIC_UTILS_AVAILABLE = None, False
zemosaic_astrometry, ZEMOSAIC_ASTROMETRY_AVAILABLE = None, False
zemosaic_align_stack, ZEMOSAIC_ALIGN_STACK_AVAILABLE = None, False
CALC_GRID_OPTIMIZED_AVAILABLE = False
_calculate_final_mosaic_grid_optimized = None

try:
    import zemosaic_utils
    from zemosaic_utils import (
        gpu_assemble_final_mosaic_reproject_coadd,
        gpu_assemble_final_mosaic_incremental,
        reproject_and_coadd_wrapper,
    )
    ZEMOSAIC_UTILS_AVAILABLE = True
    logger.info("Module 'zemosaic_utils' importé.")
except ImportError as e: logger.error(f"Import 'zemosaic_utils.py' échoué: {e}.")
try: import zemosaic_astrometry; ZEMOSAIC_ASTROMETRY_AVAILABLE = True; logger.info("Module 'zemosaic_astrometry' importé.")
except ImportError as e: logger.error(f"Import 'zemosaic_astrometry.py' échoué: {e}.")
try: import zemosaic_align_stack; ZEMOSAIC_ALIGN_STACK_AVAILABLE = True; logger.info("Module 'zemosaic_align_stack' importé.")
except ImportError as e: logger.error(f"Import 'zemosaic_align_stack.py' échoué: {e}.")
from .solver_settings import SolverSettings

# Optional configuration import for GPU toggle
try:
    import zemosaic_config
    ZEMOSAIC_CONFIG_AVAILABLE = True
except Exception:
    zemosaic_config = None  # type: ignore
    ZEMOSAIC_CONFIG_AVAILABLE = False

import importlib.util

def gpu_is_available() -> bool:
    """Return True if CuPy and a CUDA device are available."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return cupy.is_available()
    except Exception:
        return False

# Exposed compatibility flag expected by some tests
ASTROMETRY_SOLVER_AVAILABLE = ZEMOSAIC_ASTROMETRY_AVAILABLE

# progress_callback(stage: str, current: int, total: int)







# DANS zemosaic_worker.py

# ... (imports et logger configuré comme avant) ...

# --- Helper pour log et callback ---
def _log_and_callback(
    message_key_or_raw,
    progress_value=None,
    level="INFO",
    callback=None,
    **kwargs,
):
    """
    Helper pour loguer un message et appeler le callback GUI.
    - Si level est INFO, WARN, ERROR, SUCCESS, message_key_or_raw est traité comme une clé.
    - Sinon (DEBUG, ETA_LEVEL, etc.), message_key_or_raw est loggué tel quel.
    - Les **kwargs sont passés pour le formatage si message_key_or_raw est une clé.
    """
    # Support backwards compatibility for lvl/prog keyword aliases
    if "lvl" in kwargs and level == "INFO":
        level = kwargs.pop("lvl")
    elif "lvl" in kwargs:
        level = kwargs.pop("lvl")
    if "prog" in kwargs and progress_value is None:
        progress_value = kwargs.pop("prog")
    elif "prog" in kwargs:
        progress_value = kwargs.pop("prog")
    log_level_map = {
        "INFO": logging.INFO, "DEBUG": logging.DEBUG, "DEBUG_DETAIL": logging.DEBUG,
        "WARN": logging.WARNING, "ERROR": logging.ERROR, "SUCCESS": logging.INFO,
        "INFO_DETAIL": logging.DEBUG, 
        "ETA_LEVEL": logging.DEBUG, # Pour les messages ETA spécifiques
        "CHRONO_LEVEL": logging.DEBUG # Pour les commandes de chrono
    }
    
    level_str = "INFO" # Défaut
    if isinstance(level, str):
        level_str = level.upper()
    elif level is not None:
        logger.warning(f"_log_and_callback: Argument 'level' inattendu (type: {type(level)}, valeur: {level}). Utilisation de INFO par défaut.")

    # Préparer le message pour le logger Python interne
    final_message_for_py_logger = ""
    user_facing_log_levels = ["INFO", "WARN", "ERROR", "SUCCESS"]

    if level_str in user_facing_log_levels:
        # Pour ces niveaux, on s'attend à une clé. Logguer la clé et les args pour le debug interne.
        final_message_for_py_logger = f"[CLÉ_POUR_GUI: {message_key_or_raw}]"
        if kwargs:
            final_message_for_py_logger += f" (Args: {kwargs})"
    else: 
        # Pour les niveaux DEBUG, ETA, CHRONO, on loggue le message brut.
        # Si des kwargs sont passés avec un message brut (ex: debug), on peut essayer de le formater.
        final_message_for_py_logger = str(message_key_or_raw)
        if kwargs:
            try:
                final_message_for_py_logger = final_message_for_py_logger.format(**kwargs)
            except (KeyError, ValueError, IndexError) as fmt_err:
                logger.debug(f"Échec formatage message brut '{message_key_or_raw}' avec kwargs {kwargs} pour logger interne: {fmt_err}")
                # Garder le message brut si le formatage échoue

    logger.log(log_level_map.get(level_str, logging.INFO), final_message_for_py_logger)
    
    # Appel au callback GUI
    if callback and callable(callback):
        try:
            # On envoie la clé (ou le message brut) et les kwargs au callback GUI.
            # La GUI (sa méthode _log_message) sera responsable de faire la traduction
            # et le formatage final en utilisant ces kwargs si message_key_or_raw est une clé.
            #
            # La signature de _log_message dans la GUI doit être :
            # def _log_message(self, message_key_or_raw, progress_value=None, level="INFO", **kwargs):
            callback(message_key_or_raw, progress_value, level if isinstance(level, str) else "INFO", **kwargs)
        except Exception as e_cb:
            # Logguer l'erreur du callback, mais ne pas planter le worker pour ça
            logger.warning(f"Erreur dans progress_callback lors de l'appel depuis _log_and_callback: {e_cb}", exc_info=False)
            # Peut-être afficher la trace pour le debug du callback lui-même
            # logger.debug("Traceback de l'erreur du callback:", exc_info=True)




def _log_memory_usage(progress_callback: callable, context_message: str = ""): # Fonction helper définie ici ou globalement dans le module
    """Logue l'utilisation actuelle de la mémoire du processus et du système."""
    if not progress_callback or not callable(progress_callback):
        return
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / (1024 * 1024)
        vms_mb = mem_info.vms / (1024 * 1024)

        virtual_mem = psutil.virtual_memory()
        available_ram_mb = virtual_mem.available / (1024 * 1024)
        total_ram_mb = virtual_mem.total / (1024 * 1024)
        percent_ram_used = virtual_mem.percent

        swap_mem = psutil.swap_memory()
        used_swap_mb = swap_mem.used / (1024 * 1024)
        total_swap_mb = swap_mem.total / (1024 * 1024)
        percent_swap_used = swap_mem.percent
        
        log_msg = (
            f"Memory Usage ({context_message}): "
            f"Proc RSS: {rss_mb:.1f}MB, VMS: {vms_mb:.1f}MB. "
            f"Sys RAM: Avail {available_ram_mb:.0f}MB / Total {total_ram_mb:.0f}MB ({percent_ram_used}%% used). "
            f"Sys Swap: Used {used_swap_mb:.0f}MB / Total {total_swap_mb:.0f}MB ({percent_swap_used}%% used)."
        )
        _log_and_callback(log_msg, prog=None, lvl="DEBUG", callback=progress_callback)
        
    except Exception as e_mem_log:
        _log_and_callback(f"Erreur lors du logging mémoire ({context_message}): {e_mem_log}", prog=None, lvl="WARN", callback=progress_callback)


def _wait_for_memmap_files(prefixes, timeout=10.0):
    """Poll until each prefix.dat and prefix.npy exist and are non-empty."""
    import time, os
    start = time.time()
    while True:
        all_ready = True
        for prefix in prefixes:
            dat_f = prefix + '.dat'
            npy_f = prefix + '.npy'
            if not (os.path.exists(dat_f) and os.path.getsize(dat_f) > 0 and os.path.exists(npy_f) and os.path.getsize(npy_f) > 0):
                all_ready = False
                break
        if all_ready:
            return
        if time.time() - start > timeout:
            raise RuntimeError(f"Memmap file not ready after {timeout}s: {prefix}")


def astap_paths_valid(astap_exe_path: str, astap_data_dir: str) -> bool:
    """Return True if ASTAP executable and data directory look valid."""
    return (
        astap_exe_path
        and os.path.isfile(astap_exe_path)
        and astap_data_dir
        and os.path.isdir(astap_data_dir)
    )


def _write_header_to_fits(file_path: str, header_obj, pcb=None):
    """Safely update ``file_path`` FITS header with ``header_obj`` if possible."""
    if not (ASTROPY_AVAILABLE and fits):
        return
    try:
        with fits.open(file_path, mode="update", memmap=False) as hdul:
            hdul[0].header.update(header_obj)
            hdul.flush()
        if pcb:
            pcb("getwcs_info_header_written", lvl="DEBUG_DETAIL", filename=os.path.basename(file_path))
    except Exception as e_update:
        if pcb:
            pcb("getwcs_warn_header_write_failed", lvl="WARN", filename=os.path.basename(file_path), error=str(e_update))


def solve_with_astrometry(
    image_fits_path: str,
    fits_header,
    settings: dict | None,
    progress_callback=None,
):
    """Attempt plate solving via the Astrometry.net service."""

    if not ASTROMETRY_SOLVER_AVAILABLE:
        return None

    try:
        from . import zemosaic_astrometry
    except Exception:
        return None

    solver_dict = settings or {}
    api_key = solver_dict.get("api_key", "")
    timeout = solver_dict.get("timeout")
    down = solver_dict.get("downsample")

    try:
        return zemosaic_astrometry.solve_with_astrometry_net(
            image_fits_path,
            fits_header,
            api_key=api_key,
            timeout_sec=timeout or 60,
            downsample_factor=down,
            update_original_header_in_place=True,
            progress_callback=progress_callback,
        )
    except Exception as e:
        _log_and_callback(
            f"Astrometry solve error: {e}", prog=None, lvl="WARN", callback=progress_callback
        )
        return None


def solve_with_ansvr(
    image_fits_path: str,
    fits_header,
    settings: dict | None,
    progress_callback=None,
):
    """Attempt plate solving using a local ansvr installation."""

    if not ASTROMETRY_SOLVER_AVAILABLE:
        return None

    try:
        from . import zemosaic_astrometry
    except Exception:
        return None

    solver_dict = settings or {}
    path = solver_dict.get("ansvr_path") or solver_dict.get("astrometry_local_path") or solver_dict.get("local_ansvr_path")
    timeout = solver_dict.get("ansvr_timeout") or solver_dict.get("timeout")

    try:
        return zemosaic_astrometry.solve_with_ansvr(
            image_fits_path,
            fits_header,
            ansvr_config_path=path or "",
            timeout_sec=timeout or 120,
            update_original_header_in_place=True,
            progress_callback=progress_callback,
        )
    except Exception as e:
        _log_and_callback(
            f"Ansvr solve error: {e}", prog=None, lvl="WARN", callback=progress_callback
        )
        return None


def _prepare_image_for_astap(data: np.ndarray, force_lum: bool = False) -> np.ndarray:
    """Normalize image layout for ASTAP.

    Parameters
    ----------
    data : np.ndarray
        Raw image data.
    force_lum : bool
        If True and ``data`` is 3-D, the channels are averaged to produce a mono image.

    Returns
    -------
    np.ndarray
        Array formatted in CHW order or 2-D monochrome.
    """

    if force_lum and data.ndim == 3:
        return data.mean(axis=0).astype(data.dtype)
    if data.ndim == 2:
        return data
    if data.ndim == 3 and data.shape[-1] in (3, 4):
        return np.moveaxis(data, -1, 0)
    if data.ndim == 3 and data.shape[0] in (3, 4):
        return data
    if data.ndim == 3:
        return data.sum(axis=0)
    return data


def reproject_tile_to_mosaic(tile_path: str, tile_wcs, mosaic_wcs, mosaic_shape_hw,
                             feather: bool = True,
                             apply_crop: bool = False,
                             crop_percent: float = 0.0):
    """Reprojecte une tuile sur la grille finale et renvoie l'image et sa carte
    de poids ainsi que la bounding box utile.

    Les bornes sont retournées dans l'ordre ``(xmin, xmax, ymin, ymax)`` afin
    de correspondre aux indices ``[ligne, colonne]`` lors de l'incrémentation
    sur la mosaïque.

    ``tile_wcs`` et ``mosaic_wcs`` peuvent être soit des objets :class:`WCS`
    directement, soit des en-têtes FITS (``dict`` ou :class:`~astropy.io.fits.Header``).
    Cela permet d'utiliser cette fonction avec :class:`concurrent.futures.ProcessPoolExecutor`
    où les arguments doivent être sérialisables.
    """
    if not (REPROJECT_AVAILABLE and reproject_interp and ASTROPY_AVAILABLE and fits):
        return None, None, (0, 0, 0, 0)

    # Les objets WCS ne sont pas toujours sérialisables via multiprocessing.
    # Si on reçoit des en-têtes (dict ou fits.Header), reconstruire les WCS ici.
    if ASTROPY_AVAILABLE and WCS:
        if not isinstance(tile_wcs, WCS):
            try:
                tile_wcs = WCS(tile_wcs)
            except Exception:
                return None, None, (0, 0, 0, 0)
        if not isinstance(mosaic_wcs, WCS):
            try:
                mosaic_wcs = WCS(mosaic_wcs)
            except Exception:
                return None, None, (0, 0, 0, 0)

    with fits.open(tile_path, memmap=False) as hdul:
        data = hdul[0].data.astype(np.float32)

    # Les master tiles sauvegardées via ``save_fits_image`` utilisent l'ordre
    # d'axes ``CxHxW``.  Pour l'assemblage incrémental nous attendons
    # ``H x W x C``.  Effectuer la conversion si nécessaire.
    if data.ndim == 3 and data.shape[0] == 3 and data.shape[-1] != 3:
        data = np.moveaxis(data, 0, -1)

    if data.ndim == 2:
        data = data[..., np.newaxis]
    n_channels = data.shape[-1]

    # Optional cropping of the tile before reprojection
    if apply_crop and crop_percent > 1e-3 and ZEMOSAIC_UTILS_AVAILABLE \
            and hasattr(zemosaic_utils, "crop_image_and_wcs"):
        try:
            cropped, cropped_wcs = zemosaic_utils.crop_image_and_wcs(
                data,
                tile_wcs,
                crop_percent / 100.0,
                progress_callback=None,
            )
            if cropped is not None and cropped_wcs is not None:
                data = cropped
                tile_wcs = cropped_wcs
                n_channels = data.shape[-1]
        except Exception:
            pass

    base_weight = np.ones(data.shape[:2], dtype=np.float32)
    if (
        feather
        and ZEMOSAIC_UTILS_AVAILABLE
        and hasattr(zemosaic_utils, "make_radial_weight_map")
    ):
        try:
            base_weight = zemosaic_utils.make_radial_weight_map(
                data.shape[0],
                data.shape[1],
                feather_fraction=0.92,
                min_weight_floor=0.10,
            )
            logger.debug("Feather applied with min_weight_floor=0.10")
        except Exception:
            base_weight = np.ones(data.shape[:2], dtype=np.float32)

    # --- Determine bounding box covered by the tile on the mosaic
    footprint_full, _ = reproject_interp(
        (base_weight, tile_wcs),
        mosaic_wcs,
        shape_out=mosaic_shape_hw,
        order='nearest-neighbor',  # suffit, c'est binaire
        parallel=False,
    )

    j_idx, i_idx = np.where(footprint_full > 0)
    if j_idx.size == 0:
        return None, None, (0, 0, 0, 0)

    j0, j1 = int(j_idx.min()), int(j_idx.max()) + 1
    i0, i1 = int(i_idx.min()), int(i_idx.max()) + 1
    h, w = j1 - j0, i1 - i0

    # Create a WCS for the sub-region
    try:
        sub_wcs = mosaic_wcs.deepcopy()
        sub_wcs.wcs.crpix = [mosaic_wcs.wcs.crpix[0] - i0, mosaic_wcs.wcs.crpix[1] - j0]
    except Exception:
        sub_wcs = mosaic_wcs

    # Allocate arrays only for the useful area
    reproj_img = np.zeros((h, w, n_channels), dtype=np.float32)
    reproj_weight = np.zeros((h, w), dtype=np.float32)

    for c in range(n_channels):
        reproj_c, footprint = reproject_interp(
            (data[..., c], tile_wcs),
            sub_wcs,
            shape_out=(h, w),
            order='bilinear',
            parallel=False,
        )

        w_reproj, _ = reproject_interp(
            (base_weight, tile_wcs),
            sub_wcs,
            shape_out=(h, w),
            order='bilinear',
            parallel=False,
        )

        total_w = footprint * w_reproj
        reproj_img[..., c] = reproj_c.astype(np.float32)
        reproj_weight += total_w.astype(np.float32)

    valid = reproj_weight > 0
    if not np.any(valid):
        return None, None, (0, 0, 0, 0)

    # Soustraire un fond médian par canal pour imiter match_background=True
    try:
        for c in range(n_channels):
            med_c = np.nanmedian(reproj_img[..., c][valid])
            if np.isfinite(med_c):
                reproj_img[..., c] -= med_c
        reproj_img = np.clip(reproj_img, 0, None)
    except Exception:
        pass

    # Les indices sont retournés dans l'ordre (xmin, xmax, ymin, ymax)
    return reproj_img, reproj_weight, (i0, i1, j0, j1)




# --- Fonctions Utilitaires Internes au Worker ---
def _calculate_final_mosaic_grid(panel_wcs_list: list, panel_shapes_hw_list: list,
                                 drizzle_scale_factor: float = 1.0, progress_callback: callable = None):
    num_initial_inputs = len(panel_wcs_list)
    # Utilisation de clés pour les messages utilisateur
    _log_and_callback("calcgrid_info_start_calc", num_wcs_shapes=num_initial_inputs, scale_factor=drizzle_scale_factor, level="DEBUG_DETAIL", callback=progress_callback)
    
    if not REPROJECT_AVAILABLE:
        _log_and_callback("calcgrid_error_reproject_unavailable", level="ERROR", callback=progress_callback)
        return None, None
    if find_optimal_celestial_wcs is None:
        if CALC_GRID_OPTIMIZED_AVAILABLE and _calculate_final_mosaic_grid_optimized:
            _log_and_callback(
                "calcgrid_warn_find_optimal_celestial_wcs_missing",
                level="WARN",
                callback=progress_callback,
            )
            return _calculate_final_mosaic_grid_optimized(
                panel_wcs_list, panel_shapes_hw_list, drizzle_scale_factor
            )
        _log_and_callback("calcgrid_error_reproject_unavailable", level="ERROR", callback=progress_callback)
        return None, None
    if not (ASTROPY_AVAILABLE and u and Angle):
        _log_and_callback("calcgrid_error_astropy_unavailable", level="ERROR", callback=progress_callback); return None, None
    if num_initial_inputs == 0:
        _log_and_callback("calcgrid_error_no_wcs_shape", level="ERROR", callback=progress_callback); return None, None

    valid_wcs_inputs = []; valid_shapes_inputs_hw = []
    for idx_filt, wcs_filt in enumerate(panel_wcs_list):
        if isinstance(wcs_filt, WCS) and wcs_filt.is_celestial:
            if idx_filt < len(panel_shapes_hw_list):
                shape_filt = panel_shapes_hw_list[idx_filt]
                if isinstance(shape_filt, tuple) and len(shape_filt) == 2 and isinstance(shape_filt[0], int) and shape_filt[0] > 0 and isinstance(shape_filt[1], int) and shape_filt[1] > 0:
                    valid_wcs_inputs.append(wcs_filt); valid_shapes_inputs_hw.append(shape_filt)
                else: _log_and_callback("calcgrid_warn_invalid_shape_skipped", shape=shape_filt, wcs_index=idx_filt, level="WARN", callback=progress_callback)
            else: _log_and_callback("calcgrid_warn_no_shape_for_wcs_skipped", wcs_index=idx_filt, level="WARN", callback=progress_callback)
        else: _log_and_callback("calcgrid_warn_invalid_wcs_skipped", wcs_index=idx_filt, level="WARN", callback=progress_callback)
    
    if not valid_wcs_inputs:
        _log_and_callback("calcgrid_error_no_valid_wcs_shape_after_filter", level="ERROR", callback=progress_callback); return None, None

    panel_wcs_list_to_use = valid_wcs_inputs; panel_shapes_hw_list_to_use = valid_shapes_inputs_hw
    num_valid_inputs = len(panel_wcs_list_to_use)
    _log_and_callback(f"CalcGrid: {num_valid_inputs} WCS/Shapes valides pour calcul.", None, "DEBUG", progress_callback) # Log technique

    inputs_for_optimal_wcs_calc = []
    for i in range(num_valid_inputs):
        wcs_in = panel_wcs_list_to_use[i]
        shape_in_hw = panel_shapes_hw_list_to_use[i] # shape (height, width)
        shape_in_wh_for_wcs_pixel_shape = (shape_in_hw[1], shape_in_hw[0]) # (width, height) for WCS.pixel_shape

        # Ensure WCS.pixel_shape is set for reproject, it might use it internally.
        if wcs_in.pixel_shape is None or wcs_in.pixel_shape != shape_in_wh_for_wcs_pixel_shape:
            try: 
                wcs_in.pixel_shape = shape_in_wh_for_wcs_pixel_shape
                _log_and_callback(f"CalcGrid: WCS {i} pixel_shape set to {shape_in_wh_for_wcs_pixel_shape}", None, "DEBUG_DETAIL", progress_callback)
            except Exception as e_pshape_set: 
                _log_and_callback("calcgrid_warn_set_pixel_shape_failed", wcs_index=i, error=str(e_pshape_set), level="WARN", callback=progress_callback)
        
        # **** LA CORRECTION EST ICI ****
        # find_optimal_celestial_wcs expects a list of (shape, wcs) tuples or HDU objects.
        # The shape should be (height, width).
        inputs_for_optimal_wcs_calc.append((shape_in_hw, wcs_in))
        # *****************************

    if not inputs_for_optimal_wcs_calc:
        _log_and_callback("calcgrid_error_no_wcs_for_optimal_calc", level="ERROR", callback=progress_callback); return None, None
        
    try:
        sum_of_pixel_scales_deg = 0.0; count_of_valid_scales = 0
        # For calculating average input pixel scale, we use panel_wcs_list_to_use (which are just WCS objects)
        for wcs_obj_scale in panel_wcs_list_to_use: 
            if not (wcs_obj_scale and wcs_obj_scale.is_celestial): continue
            try:
                current_pixel_scale_deg = 0.0
                if hasattr(wcs_obj_scale, 'proj_plane_pixel_scales') and callable(wcs_obj_scale.proj_plane_pixel_scales):
                    pixel_scales_angle_tuple = wcs_obj_scale.proj_plane_pixel_scales(); current_pixel_scale_deg = np.mean(np.abs([s.to_value(u.deg) for s in pixel_scales_angle_tuple]))
                elif hasattr(wcs_obj_scale, 'pixel_scale_matrix'): current_pixel_scale_deg = np.sqrt(np.abs(np.linalg.det(wcs_obj_scale.pixel_scale_matrix)))
                else: continue
                if np.isfinite(current_pixel_scale_deg) and current_pixel_scale_deg > 1e-10: sum_of_pixel_scales_deg += current_pixel_scale_deg; count_of_valid_scales += 1
            except Exception: pass # Ignore errors in calculating scale for one WCS
        
        avg_input_pixel_scale_deg = (2.0 / 3600.0) # Fallback 2 arcsec/pix
        if count_of_valid_scales > 0: avg_input_pixel_scale_deg = sum_of_pixel_scales_deg / count_of_valid_scales
        elif num_valid_inputs > 0 : _log_and_callback("calcgrid_warn_scale_fallback", level="WARN", callback=progress_callback)
        
        target_resolution_deg_per_pixel = avg_input_pixel_scale_deg / drizzle_scale_factor
        target_resolution_angle = Angle(target_resolution_deg_per_pixel, unit=u.deg)
        _log_and_callback("calcgrid_info_scales", avg_input_scale_arcsec=avg_input_pixel_scale_deg*3600, target_scale_arcsec=target_resolution_angle.arcsec, level="INFO", callback=progress_callback)
        
        # Now call with inputs_for_optimal_wcs_calc which is a list of (shape_hw, wcs) tuples
        optimal_wcs_out, optimal_shape_hw_out = find_optimal_celestial_wcs(
            inputs_for_optimal_wcs_calc, # This is now a list of (shape_hw, WCS) tuples
            resolution=target_resolution_angle, 
            auto_rotate=True, 
            projection='TAN', 
            reference=None, 
            frame='icrs'
        )
        
        if optimal_wcs_out and optimal_shape_hw_out:
            expected_pixel_shape_wh_for_wcs_out = (optimal_shape_hw_out[1], optimal_shape_hw_out[0])
            if optimal_wcs_out.pixel_shape is None or optimal_wcs_out.pixel_shape != expected_pixel_shape_wh_for_wcs_out:
                try: optimal_wcs_out.pixel_shape = expected_pixel_shape_wh_for_wcs_out
                except Exception: pass
            if not (hasattr(optimal_wcs_out.wcs, 'naxis1') and hasattr(optimal_wcs_out.wcs, 'naxis2')) or not (optimal_wcs_out.wcs.naxis1 > 0 and optimal_wcs_out.wcs.naxis2 > 0) :
                try: optimal_wcs_out.wcs.naxis1 = expected_pixel_shape_wh_for_wcs_out[0]; optimal_wcs_out.wcs.naxis2 = expected_pixel_shape_wh_for_wcs_out[1]
                except Exception: pass
        
        _log_and_callback("calcgrid_info_optimal_grid_calculated", shape=optimal_shape_hw_out, crval=optimal_wcs_out.wcs.crval if optimal_wcs_out and optimal_wcs_out.wcs else 'N/A', level="INFO", callback=progress_callback)
        return optimal_wcs_out, optimal_shape_hw_out
    except ImportError: _log_and_callback("calcgrid_error_find_optimal_wcs_unavailable", level="ERROR", callback=progress_callback); return None, None
    except Exception as e_optimal_wcs_call: 
        _log_and_callback("calcgrid_error_find_optimal_wcs_call", error=str(e_optimal_wcs_call), level="ERROR", callback=progress_callback)
        logger.error("Traceback find_optimal_celestial_wcs:", exc_info=True)
        return None, None


def cluster_seestar_stacks(all_raw_files_with_info: list, stack_threshold_deg: float, progress_callback: callable):
    """Group raw files captured by the Seestar based on their WCS position."""

    if not (ASTROPY_AVAILABLE and SkyCoord and u):
        _log_and_callback("clusterstacks_error_astropy_unavailable", level="ERROR", callback=progress_callback)
        return []

    if not all_raw_files_with_info:
        _log_and_callback("clusterstacks_warn_no_raw_info", level="WARN", callback=progress_callback)
        return []

    _log_and_callback(
        "clusterstacks_info_start",
        num_files=len(all_raw_files_with_info),
        threshold=stack_threshold_deg,
        level="INFO",
        callback=progress_callback,
    )

    panel_centers_sky = []
    panel_data_for_clustering = []

    for i, info in enumerate(all_raw_files_with_info):
        wcs_obj = info["wcs"]
        if not (wcs_obj and wcs_obj.is_celestial):
            continue
        try:
            if wcs_obj.pixel_shape:
                center_world = wcs_obj.pixel_to_world(
                    wcs_obj.pixel_shape[0] / 2.0,
                    wcs_obj.pixel_shape[1] / 2.0,
                )
            elif hasattr(wcs_obj.wcs, "crval"):
                center_world = SkyCoord(
                    ra=wcs_obj.wcs.crval[0] * u.deg,
                    dec=wcs_obj.wcs.crval[1] * u.deg,
                    frame="icrs",
                )
            else:
                continue
            panel_centers_sky.append(center_world)
            panel_data_for_clustering.append(info)
        except Exception:
            continue

    if not panel_centers_sky:
        _log_and_callback("clusterstacks_warn_no_centers", level="WARN", callback=progress_callback)
        return []

    groups = []
    assigned_mask = [False] * len(panel_centers_sky)

    for i in range(len(panel_centers_sky)):
        if assigned_mask[i]:
            continue
        current_group_infos = [panel_data_for_clustering[i]]
        assigned_mask[i] = True
        current_group_center_seed = panel_centers_sky[i]
        for j in range(i + 1, len(panel_centers_sky)):
            if assigned_mask[j]:
                continue
            if current_group_center_seed.separation(panel_centers_sky[j]).deg < stack_threshold_deg:
                current_group_infos.append(panel_data_for_clustering[j])
                assigned_mask[j] = True
        groups.append(current_group_infos)

    _log_and_callback("clusterstacks_info_finished", num_groups=len(groups), level="INFO", callback=progress_callback)
    return groups

def get_wcs_and_pretreat_raw_file(
    file_path: str,
    astap_exe_path: str,
    astap_data_dir: str,
    astap_search_radius: float,
    astap_downsample: int,
    astap_sensitivity: int,
    astap_timeout_seconds: int,
    progress_callback: callable,
    hotpix_mask_dir: str | None = None,
    solver_settings: dict | None = None,
):
    filename = os.path.basename(file_path)
    # Utiliser une fonction helper pour les logs internes à cette fonction si _log_and_callback
    # est trop lié à la structure de run_hierarchical_mosaic
    _pcb_local = lambda msg_key, lvl="DEBUG", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else print(f"GETWCS_LOG {lvl}: {msg_key} {kwargs}")

    if solver_settings is None:
        solver_settings = {}

    _pcb_local(f"GetWCS_Pretreat: Début pour '{filename}'.", lvl="DEBUG_DETAIL") # Niveau DEBUG_DETAIL pour être moins verbeux

    hp_mask_path = None

    if not (ZEMOSAIC_UTILS_AVAILABLE and zemosaic_utils):
        _pcb_local("getwcs_error_utils_unavailable", lvl="ERROR")
        return None, None, None, None
        
    res_load = zemosaic_utils.load_and_validate_fits(
        file_path,
        normalize_to_float32=False,
        attempt_fix_nonfinite=True,
        progress_callback=progress_callback,
    )
    if isinstance(res_load, tuple):
        img_data_raw_adu = res_load[0]
        header_orig = res_load[1] if len(res_load) > 1 else None
    else:
        img_data_raw_adu = res_load
        header_orig = None

    if img_data_raw_adu is None or header_orig is None:
        _pcb_local("getwcs_error_load_failed", lvl="ERROR", filename=filename)
        # Le fichier n'a pas pu être chargé, on ne peut pas le déplacer car on ne sait pas s'il existe ou est corrompu.
        # Ou on pourrait essayer de le déplacer s'il existe. Pour l'instant, on retourne None.
        return None, None, None, None

    # ... (log de post-load) ...
    _pcb_local(f"  Post-Load: '{filename}' - Shape: {img_data_raw_adu.shape}, Dtype: {img_data_raw_adu.dtype}", lvl="DEBUG_VERY_DETAIL")

    img_data_processed_adu = img_data_raw_adu.astype(np.float32, copy=True)

    # --- Débayerisation ---
    if img_data_processed_adu.ndim == 2:
        _pcb_local(f"  Monochrome détecté pour '{filename}'. Débayerisation...", lvl="DEBUG_DETAIL")
        bayer_pattern = header_orig.get('BAYERPAT', header_orig.get('CFAIMAGE', 'GRBG'))
        if not isinstance(bayer_pattern, str) or bayer_pattern.upper() not in ['GRBG','RGGB','GBRG','BGGR']: bayer_pattern = 'GRBG'
        else: bayer_pattern = bayer_pattern.upper()
        
        bitpix = header_orig.get('BITPIX', 16)
        # ... (logique de max_val_for_norm_before_debayer inchangée) ...
        max_val_for_norm_before_debayer = (2**abs(bitpix))-1. if bitpix!=0 and np.issubdtype(img_data_processed_adu.dtype,np.integer) else (65535. if np.issubdtype(img_data_processed_adu.dtype,np.unsignedinteger) else 1.)
        if abs(bitpix)>16 and np.issubdtype(img_data_processed_adu.dtype,np.integer): max_val_for_norm_before_debayer=(2**16)-1.
        if max_val_for_norm_before_debayer<=0: max_val_for_norm_before_debayer=1.

        img_norm_for_debayer = np.zeros_like(img_data_processed_adu,dtype=np.float32)
        min_adu_pre_debayer,max_adu_pre_debayer=np.nanmin(img_data_processed_adu),np.nanmax(img_data_processed_adu)
        range_adu_pre_debayer=max_adu_pre_debayer-min_adu_pre_debayer
        if range_adu_pre_debayer>1e-9: img_norm_for_debayer=(img_data_processed_adu-min_adu_pre_debayer)/range_adu_pre_debayer
        elif np.any(np.isfinite(img_data_processed_adu)): img_norm_for_debayer=np.full_like(img_data_processed_adu,0.5)
        img_norm_for_debayer=np.clip(img_norm_for_debayer,0.,1.)
        
        try:
            img_rgb_norm_01 = zemosaic_utils.debayer_image(img_norm_for_debayer, bayer_pattern, progress_callback=progress_callback)
            if range_adu_pre_debayer>1e-9: img_data_processed_adu=(img_rgb_norm_01*range_adu_pre_debayer)+min_adu_pre_debayer
            else: img_data_processed_adu=np.full_like(img_rgb_norm_01,min_adu_pre_debayer if np.isfinite(min_adu_pre_debayer) else 0.)
        except Exception as e_debayer: 
            _pcb_local("getwcs_warn_debayer_failed", lvl="WARN", filename=filename, error=str(e_debayer))
            img_data_processed_adu = np.stack([img_data_processed_adu]*3, axis=-1) # Fallback stack
    
    if img_data_processed_adu.ndim == 2: # Toujours monochrome après tentative de débayerisation
        _pcb_local("getwcs_warn_still_2d_after_debayer_attempt", lvl="WARN", filename=filename)
        img_data_processed_adu = np.stack([img_data_processed_adu]*3, axis=-1)
    
    if img_data_processed_adu.ndim != 3 or img_data_processed_adu.shape[-1] != 3:
        _pcb_local("getwcs_error_shape_after_debayer_final_check", lvl="ERROR", filename=filename, shape=str(img_data_processed_adu.shape))
        return None, None, None, None

    # --- Correction Hot Pixels ---
    _pcb_local(f"  Correction HP pour '{filename}'...", lvl="DEBUG_DETAIL")
    if hotpix_mask_dir:
        os.makedirs(hotpix_mask_dir, exist_ok=True)
        hp_mask_path = os.path.join(hotpix_mask_dir, f"hp_mask_{os.path.splitext(filename)[0]}_{uuid.uuid4().hex}.npy")
    if 'save_mask_path' in zemosaic_utils.detect_and_correct_hot_pixels.__code__.co_varnames:
        img_data_hp_corrected_adu = zemosaic_utils.detect_and_correct_hot_pixels(
            img_data_processed_adu,
            3.0,
            5,
            progress_callback=progress_callback,
            save_mask_path=hp_mask_path,
        )
    else:
        img_data_hp_corrected_adu = zemosaic_utils.detect_and_correct_hot_pixels(
            img_data_processed_adu,
            3.0,
            5,
            progress_callback=progress_callback,
        )
    if img_data_hp_corrected_adu is not None: 
        img_data_processed_adu = img_data_hp_corrected_adu
    else: _pcb_local("getwcs_warn_hp_returned_none_using_previous", lvl="WARN", filename=filename)

    # --- Résolution WCS ---
    _pcb_local(f"  Résolution WCS pour '{filename}'...", lvl="DEBUG_DETAIL")
    wcs_brute = None
    if ASTROPY_AVAILABLE and WCS: # S'assurer que WCS est bien l'objet d'Astropy
        try:
            wcs_from_header = WCS(header_orig, naxis=2, relax=True) # Utiliser WCS d'Astropy
            if wcs_from_header.is_celestial and hasattr(wcs_from_header.wcs,'crval') and \
               (hasattr(wcs_from_header.wcs,'cdelt') or hasattr(wcs_from_header.wcs,'cd') or hasattr(wcs_from_header.wcs,'pc')):
                wcs_brute = wcs_from_header
                _pcb_local(f"    WCS trouvé dans header FITS de '{filename}'.", lvl="DEBUG_DETAIL")
        except Exception as e_wcs_hdr:
            _pcb_local("getwcs_warn_header_wcs_read_failed", lvl="WARN", filename=filename, error=str(e_wcs_hdr))
            wcs_brute = None
            
    solver_choice_effective = (solver_settings or {}).get("solver_choice", "ASTAP")
    api_key_len = len((solver_settings or {}).get("api_key", ""))
    _pcb_local(
        f"Solver choice effective={solver_choice_effective}",
        lvl="DEBUG_DETAIL",
    )
    if wcs_brute is None and ZEMOSAIC_ASTROMETRY_AVAILABLE and zemosaic_astrometry:
        tempdir_solver = tempfile.mkdtemp(prefix="solver_")
        basename = os.path.splitext(filename)[0]
        temp_fits = os.path.join(tempdir_solver, f"{basename}_minimal.fits")
        try:
            force_lum = bool((solver_settings or {}).get("force_lum", False))
            img_norm = _prepare_image_for_astap(img_data_raw_adu, force_lum=force_lum)
            zemosaic_utils.save_numpy_to_fits(img_norm, header_orig, temp_fits, axis_order="CHW")
        except Exception as e_tmp_write:
            _pcb_local("getwcs_error_astap_tempfile_write_failed", lvl="ERROR", filename=filename, error=str(e_tmp_write))
            logger.error(f"Erreur écriture FITS temporaire solver pour {filename}:", exc_info=True)
            del img_data_raw_adu
            gc.collect()
            try:
                shutil.rmtree(tempdir_solver)
            except Exception:
                pass
            wcs_brute = None
        else:
            try:
                if solver_choice_effective == "ASTROMETRY":
                    _pcb_local("GetWCS: using ASTROMETRY", lvl="DEBUG")
                    wcs_brute = solve_with_astrometry(
                        temp_fits,
                        header_orig,
                        solver_settings or {},
                        progress_callback,
                    )
                    if not wcs_brute and astap_paths_valid(astap_exe_path, astap_data_dir):
                        _pcb_local("Astrometry failed; fallback to ASTAP", lvl="INFO")
                        _pcb_local("GetWCS: using ASTAP (fallback)", lvl="DEBUG")
                        wcs_brute = zemosaic_astrometry.solve_with_astap(
                            image_fits_path=temp_fits,
                            original_fits_header=header_orig,
                            astap_exe_path=astap_exe_path,
                            astap_data_dir=astap_data_dir,
                            search_radius_deg=astap_search_radius,
                            downsample_factor=astap_downsample,
                            sensitivity=astap_sensitivity,
                            timeout_sec=astap_timeout_seconds,
                            update_original_header_in_place=True,
                            progress_callback=progress_callback,
                        )
                    if wcs_brute:
                        _pcb_local("getwcs_info_astrometry_solved", lvl="INFO_DETAIL", filename=filename)
                elif solver_choice_effective == "ANSVR":
                    _pcb_local("GetWCS: using ANSVR", lvl="DEBUG")
                    wcs_brute = solve_with_ansvr(
                        temp_fits,
                        header_orig,
                        solver_settings or {},
                        progress_callback,
                    )
                    if not wcs_brute and astap_paths_valid(astap_exe_path, astap_data_dir):
                        _pcb_local("Ansvr failed; fallback to ASTAP", lvl="INFO")
                        _pcb_local("GetWCS: using ASTAP (fallback)", lvl="DEBUG")
                        wcs_brute = zemosaic_astrometry.solve_with_astap(
                            image_fits_path=temp_fits,
                            original_fits_header=header_orig,
                            astap_exe_path=astap_exe_path,
                            astap_data_dir=astap_data_dir,
                            search_radius_deg=astap_search_radius,
                            downsample_factor=astap_downsample,
                            sensitivity=astap_sensitivity,
                            timeout_sec=astap_timeout_seconds,
                            update_original_header_in_place=True,
                            progress_callback=progress_callback,
                        )
                    if wcs_brute:
                        _pcb_local("getwcs_info_astrometry_solved", lvl="INFO_DETAIL", filename=filename)
                else:
                    _pcb_local("GetWCS: using ASTAP", lvl="DEBUG")
                    wcs_brute = zemosaic_astrometry.solve_with_astap(
                        image_fits_path=temp_fits,
                        original_fits_header=header_orig,
                        astap_exe_path=astap_exe_path,
                        astap_data_dir=astap_data_dir,
                        search_radius_deg=astap_search_radius,
                        downsample_factor=astap_downsample,
                        sensitivity=astap_sensitivity,
                        timeout_sec=astap_timeout_seconds,
                        update_original_header_in_place=True,
                        progress_callback=progress_callback,
                    )
                    if wcs_brute:
                        _pcb_local("getwcs_info_astap_solved", lvl="INFO_DETAIL", filename=filename)
                    else:
                        _pcb_local("getwcs_warn_astap_failed", lvl="WARN", filename=filename)
            except Exception as e_solver_call:
                _pcb_local("getwcs_error_astap_exception", lvl="ERROR", filename=filename, error=str(e_solver_call))
                logger.error(f"Erreur solver pour {filename}", exc_info=True)
                wcs_brute = None
            finally:
                del img_data_raw_adu
                gc.collect()
                try:
                    os.remove(temp_fits)
                    os.rmdir(tempdir_solver)
                except Exception:
                    pass
    elif wcs_brute is None: # Ni header, ni ASTAP n'a fonctionné ou n'était dispo
        _pcb_local("getwcs_warn_no_wcs_source_available_or_failed", lvl="WARN", filename=filename)
        # Action de déplacement sera gérée par le check suivant

    # --- Vérification finale du WCS et action de déplacement si échec ---
    if wcs_brute and wcs_brute.is_celestial:
        # Mettre à jour pixel_shape si nécessaire
        if wcs_brute.pixel_shape is None or not (wcs_brute.pixel_shape[0]>0 and wcs_brute.pixel_shape[1]>0):
            n1_final = header_orig.get('NAXIS1', img_data_processed_adu.shape[1])
            n2_final = header_orig.get('NAXIS2', img_data_processed_adu.shape[0])
            if n1_final > 0 and n2_final > 0:
                try: wcs_brute.pixel_shape = (int(n1_final), int(n2_final))
                except Exception as e_ps_final: 
                    _pcb_local("getwcs_error_set_pixel_shape_final_wcs_invalid", lvl="ERROR", filename=filename, error=str(e_ps_final))
                    # WCS devient invalide ici
                    wcs_brute = None # Forcer le déplacement
            else:
                _pcb_local("getwcs_error_invalid_naxis_for_pixel_shape_wcs_invalid", lvl="ERROR", filename=filename)
                wcs_brute = None # Forcer le déplacement
        
        if wcs_brute and wcs_brute.is_celestial: # Re-vérifier après la tentative de set_pixel_shape
            _pcb_local("getwcs_info_pretreatment_wcs_ok", lvl="DEBUG", filename=filename)
            _write_header_to_fits(file_path, header_orig, _pcb_local)
            return img_data_processed_adu, wcs_brute, header_orig, hp_mask_path
        # else: tombe dans le bloc de déplacement ci-dessous

    # Si on arrive ici, c'est que wcs_brute est None ou non céleste
    _pcb_local("getwcs_action_moving_unsolved_file", lvl="WARN", filename=filename)
    try:
        original_file_dir = os.path.dirname(file_path)
        unaligned_dir_name = "unaligned_by_zemosaic"
        unaligned_path = os.path.join(original_file_dir, unaligned_dir_name)
        
        if not os.path.exists(unaligned_path):
            os.makedirs(unaligned_path)
            _pcb_local(f"  Création dossier: '{unaligned_path}'", lvl="INFO_DETAIL")
        
        destination_path = os.path.join(unaligned_path, filename)
        
        if os.path.exists(destination_path):
            base, ext = os.path.splitext(filename)
            timestamp_suffix = time.strftime("_%Y%m%d%H%M%S")
            destination_path = os.path.join(unaligned_path, f"{base}{timestamp_suffix}{ext}")
            _pcb_local(f"  Fichier de destination '{filename}' existe déjà. Renommage en '{os.path.basename(destination_path)}'", lvl="DEBUG_DETAIL")

        shutil.move(file_path, destination_path) # shutil.move écrase si la destination existe et est un fichier
                                                  # mais notre renommage ci-dessus gère le cas.
        _pcb_local(f"  Fichier '{filename}' déplacé vers '{unaligned_path}'.", lvl="INFO")

    except Exception as e_move:
        _pcb_local(f"getwcs_error_moving_unaligned_file", lvl="ERROR", filename=filename, error=str(e_move))
        logger.error(f"Erreur déplacement fichier {filename} vers dossier unaligned:", exc_info=True)
            
    if img_data_processed_adu is not None: del img_data_processed_adu 
    gc.collect()
    return None, None, None, None








# Dans zemosaic_worker.py

# ... (vos imports existants : os, shutil, time, traceback, gc, logging, np, astropy, reproject, et les modules zemosaic_...)

def create_master_tile(
    seestar_stack_group_info: list[dict], 
    tile_id: int, 
    output_temp_dir: str,
    # Paramètres de stacking existants
    stack_norm_method: str,
    stack_weight_method: str, # Ex: "none", "noise_variance", "noise_fwhm", "noise_plus_fwhm"
    stack_reject_algo: str,
    stack_kappa_low: float,
    stack_kappa_high: float,
    parsed_winsor_limits: tuple[float, float],
    stack_final_combine: str,
    # --- NOUVEAUX PARAMÈTRES POUR LA PONDÉRATION RADIALE ---
    apply_radial_weight: bool,             # Vient de la GUI/config
    radial_feather_fraction: float,      # Vient de la GUI/config
    radial_shape_power: float,           # Pourrait être une constante ou configurable
    min_radial_weight_floor: float,
    # --- FIN NOUVEAUX PARAMÈTRES ---
    # Paramètres ASTAP (pourraient être enlevés si plus du tout utilisés ici)
    astap_exe_path_global: str, 
    astap_data_dir_global: str, 
    astap_search_radius_global: float,
    astap_downsample_global: int,
    astap_sensitivity_global: int,
    astap_timeout_seconds_global: int,
    winsor_pool_workers: int,
    progress_callback: callable
):
    """
    Crée une "master tuile" à partir d'un groupe d'images.
    Lit les données image prétraitées depuis un cache disque (.npy).
    Utilise les WCS et Headers déjà résolus et stockés en mémoire.
    Transmet toutes les options de stacking, y compris la pondération radiale.
    """
    pcb_tile = lambda msg_key, prog=None, lvl="INFO_DETAIL", **kwargs: _log_and_callback(msg_key, prog, lvl, callback=progress_callback, **kwargs)
    # Load persistent configuration to forward GPU preference
    if ZEMOSAIC_CONFIG_AVAILABLE and zemosaic_config:
        try:
            zconfig = SimpleNamespace(**zemosaic_config.load_config())
        except Exception:
            zconfig = SimpleNamespace()
    else:
        zconfig = SimpleNamespace()
    func_id_log_base = "mastertile"

    pcb_tile(f"{func_id_log_base}_info_creation_started_from_cache", prog=None, lvl="INFO", 
             num_raw=len(seestar_stack_group_info), tile_id=tile_id)
    pcb_tile(f"    {func_id_log_base}_{tile_id}: Options Stacking - Norm='{stack_norm_method}', "
             f"Weight='{stack_weight_method}' (RadialWeight={apply_radial_weight}), "
             f"Reject='{stack_reject_algo}', Combine='{stack_final_combine}'", prog=None, lvl="DEBUG")

    if not (ZEMOSAIC_UTILS_AVAILABLE and zemosaic_utils and ZEMOSAIC_ALIGN_STACK_AVAILABLE and zemosaic_align_stack and ASTROPY_AVAILABLE and fits): # Ajout de 'fits' pour header_mt_save
        # ... (votre gestion d'erreur de dépendances existante) ...
        if not ZEMOSAIC_UTILS_AVAILABLE: pcb_tile(f"{func_id_log_base}_error_utils_unavailable", prog=None, lvl="ERROR", tile_id=tile_id)
        if not ZEMOSAIC_ALIGN_STACK_AVAILABLE: pcb_tile(f"{func_id_log_base}_error_alignstack_unavailable", prog=None, lvl="ERROR", tile_id=tile_id)
        if not ASTROPY_AVAILABLE or not fits: pcb_tile(f"{func_id_log_base}_error_astropy_unavailable", prog=None, lvl="ERROR", tile_id=tile_id)
        return None, None
        
    if not seestar_stack_group_info: 
        pcb_tile(f"{func_id_log_base}_error_no_images_provided", prog=None, lvl="ERROR", tile_id=tile_id)
        return None,None
    
    # Choix de l'image de référence (généralement la première du groupe après tri ou la plus centrale)
    reference_image_index_in_group = 0 # Pourrait être plus sophistiqué à l'avenir
    if not (0 <= reference_image_index_in_group < len(seestar_stack_group_info)): 
        pcb_tile(f"{func_id_log_base}_error_invalid_ref_index", prog=None, lvl="ERROR", tile_id=tile_id, ref_idx=reference_image_index_in_group, group_size=len(seestar_stack_group_info))
        return None,None
    
    ref_info_for_tile = seestar_stack_group_info[reference_image_index_in_group]
    wcs_for_master_tile = ref_info_for_tile.get('wcs')
    # Le header est un dict venant du cache, il faut le convertir en objet fits.Header si besoin
    header_dict_for_master_tile_base = ref_info_for_tile.get('header') 

    if not (wcs_for_master_tile and wcs_for_master_tile.is_celestial and header_dict_for_master_tile_base):
        pcb_tile(f"{func_id_log_base}_error_invalid_ref_wcs_header", prog=None, lvl="ERROR", tile_id=tile_id)
        return None,None
    
    # Conversion du dict en objet astropy.io.fits.Header pour la sauvegarde
    header_for_master_tile_base = fits.Header(header_dict_for_master_tile_base.cards if hasattr(header_dict_for_master_tile_base,'cards') else header_dict_for_master_tile_base)
    
    ref_path_raw = ref_info_for_tile.get('path_raw', 'UnknownRawRef')
    pcb_tile(f"{func_id_log_base}_info_reference_set", prog=None, lvl="DEBUG_DETAIL", ref_index=reference_image_index_in_group, ref_filename=os.path.basename(ref_path_raw), tile_id=tile_id)

    pcb_tile(f"{func_id_log_base}_info_loading_from_cache_started", prog=None, lvl="DEBUG_DETAIL", num_images=len(seestar_stack_group_info), tile_id=tile_id)
    
    tile_images_data_HWC_adu = []
    tile_original_raw_headers = [] # Liste des dictionnaires de header originaux

    for i, raw_file_info in enumerate(seestar_stack_group_info):
        cached_image_file_path = raw_file_info.get('path_preprocessed_cache')
        original_raw_path = raw_file_info.get('path_raw', 'UnknownRawPathForTileImg') # Plus descriptif

        if not (cached_image_file_path and os.path.exists(cached_image_file_path)):
            pcb_tile(f"{func_id_log_base}_warn_cache_file_missing", prog=None, lvl="WARN", filename=os.path.basename(original_raw_path), cache_path=cached_image_file_path, tile_id=tile_id)
            continue
        
        # pcb_tile(f"    {func_id_log_base}_{tile_id}_Img{i}: Lecture cache '{os.path.basename(cached_image_file_path)}'", prog=None, lvl="DEBUG_VERY_DETAIL")
        
        try:
            img_data_adu = np.load(cached_image_file_path) 
            if not (isinstance(img_data_adu, np.ndarray) and img_data_adu.dtype == np.float32 and img_data_adu.ndim == 3 and img_data_adu.shape[-1] == 3):
                pcb_tile(f"{func_id_log_base}_warn_invalid_cached_data", prog=None, lvl="WARN", filename=os.path.basename(cached_image_file_path), 
                         shape=img_data_adu.shape if hasattr(img_data_adu, 'shape') else 'N/A', 
                         dtype=img_data_adu.dtype if hasattr(img_data_adu, 'dtype') else 'N/A', tile_id=tile_id)
                del img_data_adu; gc.collect(); continue
            
            tile_images_data_HWC_adu.append(img_data_adu)
            # Stocker le dict de header, pas l'objet fits.Header, car c'est ce qui est dans raw_file_info
            tile_original_raw_headers.append(raw_file_info.get('header')) 
        except MemoryError as e_mem_load_cache:
             pcb_tile(f"{func_id_log_base}_error_memory_loading_cache", prog=None, lvl="ERROR", filename=os.path.basename(cached_image_file_path), error=str(e_mem_load_cache), tile_id=tile_id)
             del tile_images_data_HWC_adu, tile_original_raw_headers; gc.collect(); return None, None
        except Exception as e_load_cache:
            pcb_tile(f"{func_id_log_base}_error_loading_cache", prog=None, lvl="ERROR", filename=os.path.basename(cached_image_file_path), error=str(e_load_cache), tile_id=tile_id)
            logger.error(f"Erreur chargement cache {cached_image_file_path} pour tuile {tile_id}", exc_info=True)
            continue
            
    if not tile_images_data_HWC_adu: 
        pcb_tile(f"{func_id_log_base}_error_no_valid_images_from_cache", prog=None, lvl="ERROR", tile_id=tile_id)
        return None,None
    # pcb_tile(f"{func_id_log_base}_info_loading_from_cache_finished", prog=None, lvl="DEBUG_DETAIL", num_loaded=len(tile_images_data_HWC_adu), tile_id=tile_id)

    # pcb_tile(f"{func_id_log_base}_info_intra_tile_alignment_started", prog=None, lvl="DEBUG_DETAIL", num_to_align=len(tile_images_data_HWC_adu), tile_id=tile_id)
    aligned_images_for_stack = zemosaic_align_stack.align_images_in_group(
        image_data_list=tile_images_data_HWC_adu, 
        reference_image_index=reference_image_index_in_group, 
        progress_callback=progress_callback
    )
    del tile_images_data_HWC_adu; gc.collect()
    
    valid_aligned_images = [img for img in aligned_images_for_stack if img is not None]
    if aligned_images_for_stack: del aligned_images_for_stack # Libérer la liste originale après filtrage

    num_actually_aligned_for_header = len(valid_aligned_images)
    # pcb_tile(f"{func_id_log_base}_info_intra_tile_alignment_finished", prog=None, lvl="DEBUG_DETAIL", num_aligned=num_actually_aligned_for_header, tile_id=tile_id)
    
    if not valid_aligned_images: 
        pcb_tile(f"{func_id_log_base}_error_no_images_after_alignment", prog=None, lvl="ERROR", tile_id=tile_id)
        return None,None
    
    pcb_tile(f"{func_id_log_base}_info_stacking_started", prog=None, lvl="DEBUG_DETAIL", 
             num_to_stack=len(valid_aligned_images), tile_id=tile_id) # Les options sont loggées au début
    
    if stack_reject_algo == "winsorized_sigma_clip":
        master_tile_stacked_HWC, _ = zemosaic_align_stack.stack_winsorized_sigma_clip(
            valid_aligned_images,
            zconfig=zconfig,
            kappa=stack_kappa_low,
            winsor_limits=parsed_winsor_limits,
            apply_rewinsor=True,
        )
    elif stack_reject_algo == "kappa_sigma":
        master_tile_stacked_HWC, _ = zemosaic_align_stack.stack_kappa_sigma_clip(
            valid_aligned_images,
            zconfig=zconfig,
            sigma_low=stack_kappa_low,
            sigma_high=stack_kappa_high,
        )
    elif stack_reject_algo == "linear_fit_clip":
        master_tile_stacked_HWC, _ = zemosaic_align_stack.stack_linear_fit_clip(
            valid_aligned_images,
            zconfig=zconfig,
            sigma=stack_kappa_high,
        )
    else:
        master_tile_stacked_HWC = zemosaic_align_stack.stack_aligned_images(
            aligned_image_data_list=valid_aligned_images,
            normalize_method=stack_norm_method,
            weighting_method=stack_weight_method,
            rejection_algorithm=stack_reject_algo,
            final_combine_method=stack_final_combine,
            sigma_clip_low=stack_kappa_low,
            sigma_clip_high=stack_kappa_high,
            winsor_limits=parsed_winsor_limits,
            minimum_signal_adu_target=0.0,
            apply_radial_weight=apply_radial_weight,
            radial_feather_fraction=radial_feather_fraction,
            radial_shape_power=radial_shape_power,
            winsor_max_workers=winsor_pool_workers,
            progress_callback=progress_callback,
        )
    
    del valid_aligned_images; gc.collect() # valid_aligned_images a été passé par valeur (copie de la liste)
                                          # mais les arrays NumPy à l'intérieur sont passés par référence.
                                          # stack_aligned_images travaille sur ces arrays.
                                          # Il est bon de del ici.

    if master_tile_stacked_HWC is None: 
        pcb_tile(f"{func_id_log_base}_error_stacking_failed", prog=None, lvl="ERROR", tile_id=tile_id)
        return None,None
    
    pcb_tile(f"{func_id_log_base}_info_stacking_finished", prog=None, lvl="DEBUG_DETAIL", tile_id=tile_id, 
             shape=master_tile_stacked_HWC.shape)
             # min_val=np.nanmin(master_tile_stacked_HWC), # Peut être verbeux
             # max_val=np.nanmax(master_tile_stacked_HWC), 
             # mean_val=np.nanmean(master_tile_stacked_HWC))

    # pcb_tile(f"{func_id_log_base}_info_saving_started", prog=None, lvl="DEBUG_DETAIL", tile_id=tile_id)
    temp_fits_filename = f"master_tile_{tile_id:03d}.fits"
    temp_fits_filepath = os.path.join(output_temp_dir,temp_fits_filename)
    
    try:
        # Créer un nouvel objet Header pour la sauvegarde
        header_mt_save = fits.Header()
        if wcs_for_master_tile:
            try: 
                # S'assurer que wcs_for_master_tile a les NAXIS bien définis pour to_header
                # La shape de master_tile_stacked_HWC est (H, W, C)
                # Pour le WCS 2D, on a besoin de (W, H)
                if master_tile_stacked_HWC.ndim >= 2:
                    h_final, w_final = master_tile_stacked_HWC.shape[:2]
                    # Mettre à jour les attributs NAXIS du WCS si nécessaire,
                    # car to_header les utilise.
                    # wcs_for_master_tile.wcs.naxis1 = w_final # Ne pas modifier l'objet WCS original directement ici
                    # wcs_for_master_tile.wcs.naxis2 = h_final # car il est partagé/réutilisé.
                    # Créer une copie du WCS pour modification locale avant to_header si besoin.
                    # Cependant, save_fits_image devrait gérer les NAXIS en fonction des données.
                    pass

                header_mt_save.update(wcs_for_master_tile.to_header(relax=True))
            except Exception as e_wcs_hdr: 
                pcb_tile(f"{func_id_log_base}_warn_wcs_header_error_saving", prog=None, lvl="WARN", tile_id=tile_id, error=str(e_wcs_hdr))
        
        
        
        header_mt_save['ZMT_TYPE']=('Master Tile','ZeMosaic Processed Tile'); header_mt_save['ZMT_ID']=(tile_id,'Master Tile ID')
        header_mt_save['ZMT_NRAW']=(len(seestar_stack_group_info),'Raw frames in this tile group')
        header_mt_save['ZMT_NALGN']=(num_actually_aligned_for_header,'Successfully aligned frames for stack')
        header_mt_save['ZMT_NORM'] = (str(stack_norm_method), 'Normalization method')
        header_mt_save['ZMT_WGHT'] = (str(stack_weight_method), 'Weighting method')
        if apply_radial_weight: # Log des paramètres radiaux
            header_mt_save['ZMT_RADW'] = (True, 'Radial weighting applied')
            header_mt_save['ZMT_RADF'] = (radial_feather_fraction, 'Radial feather fraction')
            header_mt_save['ZMT_RADP'] = (radial_shape_power, 'Radial shape power')
        else:
            header_mt_save['ZMT_RADW'] = (False, 'Radial weighting applied')

        header_mt_save['ZMT_REJ'] = (str(stack_reject_algo), 'Rejection algorithm')
        if stack_reject_algo == "kappa_sigma":
            header_mt_save['ZMT_KAPLO'] = (stack_kappa_low, 'Kappa Sigma Low threshold')
            header_mt_save['ZMT_KAPHI'] = (stack_kappa_high, 'Kappa Sigma High threshold')
        elif stack_reject_algo == "winsorized_sigma_clip":
            header_mt_save['ZMT_WINLO'] = (parsed_winsor_limits[0], 'Winsor Lower limit %')
            header_mt_save['ZMT_WINHI'] = (parsed_winsor_limits[1], 'Winsor Upper limit %')
            # Les paramètres Kappa sont aussi pertinents pour Winsorized
            header_mt_save['ZMT_KAPLO'] = (stack_kappa_low, 'Kappa Low for Winsorized')
            header_mt_save['ZMT_KAPHI'] = (stack_kappa_high, 'Kappa High for Winsorized')
        header_mt_save['ZMT_COMB'] = (str(stack_final_combine), 'Final combine method')
        
        if header_for_master_tile_base: # C'est déjà un objet fits.Header
            ref_path_raw_for_hdr = seestar_stack_group_info[reference_image_index_in_group].get('path_raw', 'UnknownRef')
            header_mt_save['ZMT_REF'] = (os.path.basename(ref_path_raw_for_hdr), 'Reference raw frame for this tile WCS')
            keys_from_ref = ['OBJECT','DATE-AVG','FILTER','INSTRUME','FOCALLEN','XPIXSZ','YPIXSZ', 'GAIN', 'OFFSET'] # Ajout GAIN, OFFSET
            for key_h in keys_from_ref:
                if key_h in header_for_master_tile_base:
                    try: 
                        # Tenter d'obtenir la valeur et le commentaire
                        card = header_for_master_tile_base.cards[key_h]
                        header_mt_save[key_h] = (card.value, card.comment)
                    except (KeyError, AttributeError): # Si la carte n'a pas de commentaire ou si ce n'est pas un objet CardImage
                        header_mt_save[key_h] = header_for_master_tile_base[key_h]
            
            total_exposure_tile = 0.
            num_exposure_summed = 0
            for hdr_raw_item_dict in tile_original_raw_headers: # Ce sont des dicts
                if hdr_raw_item_dict is None: continue
                try: 
                    exposure_val = hdr_raw_item_dict.get('EXPTIME', hdr_raw_item_dict.get('EXPOSURE', 0.0))
                    total_exposure_tile += float(exposure_val if exposure_val is not None else 0.0)
                    num_exposure_summed +=1
                except (TypeError, ValueError) : pass
            header_mt_save['EXPTOTAL']=(round(total_exposure_tile,2),'[s] Sum of EXPTIME for this tile')
            header_mt_save['NEXP_SUM']=(num_exposure_summed,'Number of exposures summed for EXPTOTAL')


        zemosaic_utils.save_fits_image(
            image_data=master_tile_stacked_HWC,
            output_path=temp_fits_filepath,
            header=header_mt_save,
            overwrite=True,
            save_as_float=True,
            progress_callback=progress_callback,
            axis_order="HWC",
        )
        pcb_tile(f"{func_id_log_base}_info_saved", prog=None, lvl="INFO_DETAIL", tile_id=tile_id, format_type='float32', filename=os.path.basename(temp_fits_filepath))
        # pcb_tile(f"{func_id_log_base}_info_saving_finished", prog=None, lvl="DEBUG_DETAIL", tile_id=tile_id)
        return temp_fits_filepath, wcs_for_master_tile
        
    except Exception as e_save_mt:
        pcb_tile(f"{func_id_log_base}_error_saving", prog=None, lvl="ERROR", tile_id=tile_id, error=str(e_save_mt))
        logger.error(f"Traceback pour {func_id_log_base}_{tile_id} sauvegarde:", exc_info=True)
        return None,None
    finally:
        if 'master_tile_stacked_HWC' in locals() and master_tile_stacked_HWC is not None: 
            del master_tile_stacked_HWC
        gc.collect()



# Dans zemosaic_worker.py

# ... (s'assurer que zemosaic_utils est importé et ZEMOSAIC_UTILS_AVAILABLE est défini)
# ... (s'assurer que WCS, fits d'Astropy sont importés, ainsi que reproject_interp)
# ... (définition de logger, _log_and_callback, etc.)



def assemble_final_mosaic_incremental(
    master_tile_fits_with_wcs_list: list,
    final_output_wcs: WCS,
    final_output_shape_hw: tuple,
    progress_callback: callable,
    n_channels: int = 3,
    dtype_accumulator: np.dtype = np.float64,
    dtype_norm: np.dtype = np.float32,
    apply_crop: bool = False,
    crop_percent: float = 0.0,
    processing_threads: int = 0,
    memmap_dir: str | None = None,
    cleanup_memmap: bool = True,
):
    """Assemble les master tiles par co-addition sur disque."""
    FLUSH_BATCH_SIZE = 10  # nombre de tuiles entre chaque flush sur le memmap
    use_feather = False  # Désactivation du feathering par défaut
    pcb_asm = lambda msg_key, prog=None, lvl="INFO_DETAIL", **kwargs: _log_and_callback(
        msg_key, prog, lvl, callback=progress_callback, **kwargs
    )

    pcb_asm(
        f"ASM_INC: Début. Options rognage - Appliquer: {apply_crop}, %: {crop_percent if apply_crop else 'N/A'}",
        lvl="DEBUG_DETAIL",
    )

    if not (REPROJECT_AVAILABLE and reproject_interp and ASTROPY_AVAILABLE and fits):
        missing_deps = []
        if not REPROJECT_AVAILABLE or not reproject_interp:
            missing_deps.append("Reproject (reproject_interp)")
        if not ASTROPY_AVAILABLE or not fits:
            missing_deps.append("Astropy (fits)")
        pcb_asm(
            "assemble_error_core_deps_unavailable_incremental",
            prog=None,
            lvl="ERROR",
            missing=", ".join(missing_deps),
        )
        return None, None

    if not master_tile_fits_with_wcs_list:
        pcb_asm("assemble_error_no_tiles_provided_incremental", prog=None, lvl="ERROR")
        return None, None

    # ``final_output_shape_hw`` MUST be provided in ``(height, width)`` order.
    if (
        not isinstance(final_output_shape_hw, (tuple, list))
        or len(final_output_shape_hw) != 2
    ):
        pcb_asm(
            "assemble_error_invalid_final_shape_inc",
            prog=None,
            lvl="ERROR",
            shape=str(final_output_shape_hw),
        )
        return None, None

    h, w = map(int, final_output_shape_hw)

    # --- Extra validation to help catch swapped width/height ---
    try:
        w_wcs = int(getattr(final_output_wcs, "pixel_shape", (w, h))[0])
        h_wcs = int(getattr(final_output_wcs, "pixel_shape", (w, h))[1])
    except Exception:
        w_wcs = int(getattr(final_output_wcs.wcs, "naxis1", w)) if hasattr(final_output_wcs, "wcs") else w
        h_wcs = int(getattr(final_output_wcs.wcs, "naxis2", h)) if hasattr(final_output_wcs, "wcs") else h

    expected_hw = (h_wcs, w_wcs)
    if (h, w) != expected_hw:
        if (w, h) == expected_hw:
            pcb_asm(
                "assemble_warn_swapped_final_shape_inc",
                prog=None,
                lvl="WARN",
                provided=str(final_output_shape_hw),
                expected=str(expected_hw),
            )
            h, w = expected_hw
        else:
            pcb_asm(
                "assemble_error_mismatch_final_shape_inc",
                prog=None,
                lvl="ERROR",
                provided=str(final_output_shape_hw),
                expected=str(expected_hw),
            )
            return None, None

    sum_shape = (h, w, n_channels)
    weight_shape = (h, w)


    internal_temp_dir = False
    if memmap_dir is None:
        memmap_dir = tempfile.mkdtemp(prefix="zemosaic_memmap_")
        internal_temp_dir = True
    else:
        os.makedirs(memmap_dir, exist_ok=True)
    sum_path = os.path.join(memmap_dir, "SOMME.fits")
    weight_path = os.path.join(memmap_dir, "WEIGHT.fits")

    try:
        fits.writeto(sum_path, np.zeros(sum_shape, dtype=dtype_accumulator), overwrite=True)
        fits.writeto(weight_path, np.zeros(weight_shape, dtype=dtype_norm), overwrite=True)
    except Exception as e_create:
        pcb_asm("assemble_error_memmap_write_failed_inc", prog=None, lvl="ERROR", error=str(e_create))
        logger.error("Failed to create memmap FITS", exc_info=True)
        return None, None


    try:
        req_workers = int(processing_threads)
    except Exception:
        req_workers = 0
    if req_workers > 0:
        max_procs = req_workers
    else:
        max_procs = min(os.cpu_count() or 1, len(master_tile_fits_with_wcs_list))
    pcb_asm(f"ASM_INC: Using {max_procs} process workers", lvl="DEBUG_DETAIL")

    parent_is_daemon = multiprocessing.current_process().daemon
    Executor = ThreadPoolExecutor if parent_is_daemon else ProcessPoolExecutor


    try:
        with Executor(max_workers=max_procs) as ex, \
                fits.open(sum_path, mode="update", memmap=True) as hsum, \
                fits.open(weight_path, mode="update", memmap=True) as hwei:
            fsum = hsum[0].data
            fwei = hwei[0].data

            tiles_since_flush = 0

            future_map = {}
            for tile_idx, (tile_path, tile_wcs) in enumerate(master_tile_fits_with_wcs_list, 1):
                pcb_asm(
                    "assemble_info_processing_tile",
                    prog=None,
                    lvl="INFO_DETAIL",
                    tile_num=tile_idx,
                    total_tiles=len(master_tile_fits_with_wcs_list),
                    filename=os.path.basename(tile_path),
                )
                # Les objets WCS peuvent poser problème lors de la sérialisation.
                # On transmet donc leurs en-têtes et ils seront reconstruits dans le worker.
                tile_wcs_hdr = tile_wcs.to_header() if hasattr(tile_wcs, "to_header") else tile_wcs
                output_wcs_hdr = final_output_wcs.to_header() if hasattr(final_output_wcs, "to_header") else final_output_wcs
                future = ex.submit(
                    reproject_tile_to_mosaic,
                    tile_path,
                    tile_wcs_hdr,
                    output_wcs_hdr,
                    final_output_shape_hw,
                    feather=use_feather,
                    apply_crop=apply_crop,
                    crop_percent=crop_percent,
                )
                future_map[future] = tile_idx

            processed = 0
            total_steps = len(future_map)
            start_time_iter = time.time()
            last_time = start_time_iter
            step_times = []
            for fut in as_completed(future_map):
                idx = future_map[fut]
                try:
                    # reproject_tile_to_mosaic renvoie les bornes de la tuile
                    # sous la forme (xmin, xmax, ymin, ymax) afin de
                    # correspondre aux indices de colonne puis de ligne.
                    I_tile, W_tile, (xmin, xmax, ymin, ymax) = fut.result()
                except MemoryError as e_mem:
                    pcb_asm(
                        "assemble_error_memory_tile_reprojection_inc",
                        prog=None,
                        lvl="ERROR",
                        tile_num=idx,
                        error=str(e_mem),
                    )
                    logger.error(
                        f"MemoryError reproject_tile_to_mosaic tuile {idx}",
                        exc_info=True,
                    )
                    processed += 1
                    continue
                except BrokenProcessPool as bpp:
                    pcb_asm(
                        "assemble_error_broken_process_pool_incremental",
                        prog=None,
                        lvl="ERROR",
                        tile_num=idx,
                        error=str(bpp),
                    )
                    logger.error(
                        "BrokenProcessPool during tile reprojection",
                        exc_info=True,
                    )
                    return None, None
                except Exception as e_reproj:
                    pcb_asm(
                        "assemble_error_tile_reprojection_failed_inc",
                        prog=None,
                        lvl="ERROR",
                        tile_num=idx,
                        error=str(e_reproj),
                    )
                    logger.error(
                        f"Erreur reproject_tile_to_mosaic tuile {idx}",
                        exc_info=True,
                    )
                    processed += 1
                    continue

                if I_tile is not None and W_tile is not None:
                    mask = W_tile > 0
                    tgt_sum = fsum[ymin:ymax, xmin:xmax]
                    tgt_wgt = fwei[ymin:ymax, xmin:xmax]
                    for c in range(n_channels):
                        tgt_sum[..., c][mask] += I_tile[..., c][mask] * W_tile[mask]
                    tgt_wgt[mask] += W_tile[mask]
                    tiles_since_flush += 1
                    if tiles_since_flush >= FLUSH_BATCH_SIZE:
                        hsum.flush()
                        hwei.flush()
                        tiles_since_flush = 0

                processed += 1
                now = time.time()
                step_times.append(now - last_time)
                last_time = now
                if progress_callback:
                    try:
                        progress_callback("phase5_incremental", processed, total_steps)
                    except Exception:
                        pass
                if processed % 10 == 0 or processed == len(master_tile_fits_with_wcs_list):
                    pcb_asm(
                        "assemble_progress_tiles_processed_inc",
                        prog=None,
                        lvl="INFO_DETAIL",
                        num_done=processed,
                        total_num=len(master_tile_fits_with_wcs_list),
                    )

            if tiles_since_flush > 0:
                hsum.flush()
                hwei.flush()
                tiles_since_flush = 0
    except Exception as e_pool:
        pcb_asm("assemble_error_incremental_pool_failed", prog=None, lvl="ERROR", error=str(e_pool))
        logger.error("Error during incremental assembly", exc_info=True)
        return None, None

    with fits.open(sum_path, memmap=True) as hsum, fits.open(weight_path, memmap=True) as hwei:
        sum_data = hsum[0].data.astype(np.float32)
        weight_data = hwei[0].data.astype(np.float32)
        mosaic = np.zeros_like(sum_data, dtype=np.float32)
        np.divide(sum_data, weight_data[..., None], out=mosaic, where=weight_data[..., None] > 0)

    if step_times:
        avg_step = sum(step_times) / len(step_times)
        total_elapsed = time.time() - start_time_iter
        pcb_asm(
            "assemble_debug_incremental_timing",
            prog=None,
            lvl="DEBUG_DETAIL",
            avg=f"{avg_step:.2f}",
            total=f"{total_elapsed:.2f}",
        )

    pcb_asm("assemble_info_finished_incremental", prog=None, lvl="INFO", shape=str(mosaic.shape))

    if cleanup_memmap:
        for p in (sum_path, weight_path):
            try:
                os.remove(p)
            except OSError:
                pass

        if internal_temp_dir:
            try:
                os.rmdir(memmap_dir)
            except OSError:
                pass


    return mosaic, weight_data

def _reproject_and_coadd_channel_worker(channel_data_list, output_wcs_header, output_shape_hw, match_bg, mm_sum_prefix=None, mm_cov_prefix=None):
    """Worker function to run reproject_and_coadd in a separate process."""
    from astropy.wcs import WCS
    from reproject import reproject_interp
    import numpy as np

    final_wcs = WCS(output_wcs_header)
    data_list = []
    wcs_list = []
    for arr, hdr in channel_data_list:
        data_list.append(arr)
        wcs_list.append(WCS(hdr))




    # The memmap prefixes are produced by other workers. Ensure they exist before
    # reading if provided. Wait here until both files are fully written.

    import inspect
    sig = inspect.signature(reproject_and_coadd)
    bg_kw = "match_background" if "match_background" in sig.parameters else (
        "match_bg" if "match_bg" in sig.parameters else None
    )

    kwargs = {
        "output_projection": final_wcs,
        "shape_out": output_shape_hw,
        "reproject_function": reproject_interp,
        "combine_function": "mean",
    }
    if bg_kw:
        kwargs[bg_kw] = match_bg

    stacked, coverage = reproject_and_coadd_wrapper(
        data_list=data_list,
        wcs_list=wcs_list,
        shape_out=output_shape_hw,
        output_projection=final_wcs,
        use_gpu=False,
        cpu_func=reproject_and_coadd,
        **kwargs,
    )

    if mm_sum_prefix and mm_cov_prefix:
        _wait_for_memmap_files([mm_sum_prefix, mm_cov_prefix])
    return stacked.astype(np.float32), coverage.astype(np.float32)


def assemble_final_mosaic_reproject_coadd(
    master_tile_fits_with_wcs_list: list,
    final_output_wcs: WCS,
    final_output_shape_hw: tuple,
    progress_callback: callable,
    n_channels: int = 3,
    match_bg: bool = True,
    apply_crop: bool = False,
    crop_percent: float = 0.0,
    use_memmap: bool = False,
    memmap_dir: str | None = None,
    cleanup_memmap: bool = True,
    assembly_process_workers: int = 0,
    re_solve_cropped_tiles: bool = False,
    solver_settings: dict | None = None,
    solver_instance=None,
    use_gpu: bool = False,


):
    """Assemble les master tiles en utilisant ``reproject_and_coadd``."""
    _pcb = lambda msg_key, prog=None, lvl="INFO_DETAIL", **kwargs: _log_and_callback(
        msg_key, prog, lvl, callback=progress_callback, **kwargs
    )

    _log_memory_usage(progress_callback, "Début assemble_final_mosaic_reproject_coadd")
    _pcb(
        f"ASM_REPROJ_COADD: Options de rognage - Appliquer: {apply_crop}, Pourcentage: {crop_percent if apply_crop else 'N/A'}",
        lvl="DEBUG_DETAIL",
    )

    # Ensure wrapper uses the possibly monkeypatched CPU implementation
    try:
        zemosaic_utils.cpu_reproject_and_coadd = reproject_and_coadd
    except Exception:
        pass


    if not (REPROJECT_AVAILABLE and reproject_and_coadd and ASTROPY_AVAILABLE and fits):
        missing_deps = []
        if not REPROJECT_AVAILABLE or not reproject_and_coadd:
            missing_deps.append("Reproject")
        if not ASTROPY_AVAILABLE or not fits:
            missing_deps.append("Astropy (fits)")
        _pcb(
            "assemble_error_core_deps_unavailable_reproject_coadd",
            prog=None,
            lvl="ERROR",
            missing=", ".join(missing_deps),
        )
        return None, None

    if not master_tile_fits_with_wcs_list:
        _pcb("assemble_error_no_tiles_provided_reproject_coadd", prog=None, lvl="ERROR")
        return None, None

    if (
        not isinstance(final_output_shape_hw, (tuple, list))
        or len(final_output_shape_hw) != 2
    ):
        _pcb(
            "assemble_error_invalid_final_shape_reproj_coadd",
            prog=None,
            lvl="ERROR",
            shape=str(final_output_shape_hw),
        )
        return None, None

    h, w = map(int, final_output_shape_hw)

    try:
        w_wcs = int(getattr(final_output_wcs, "pixel_shape", (w, h))[0])
        h_wcs = int(getattr(final_output_wcs, "pixel_shape", (w, h))[1])
    except Exception:
        w_wcs = int(getattr(final_output_wcs.wcs, "naxis1", w)) if hasattr(final_output_wcs, "wcs") else w
        h_wcs = int(getattr(final_output_wcs.wcs, "naxis2", h)) if hasattr(final_output_wcs, "wcs") else h

    expected_hw = (h_wcs, w_wcs)
    if (h, w) != expected_hw:
        if (w, h) == expected_hw:
            _pcb(
                "assemble_warn_swapped_final_shape_reproj_coadd",
                prog=None,
                lvl="WARN",
                provided=str(final_output_shape_hw),
                expected=str(expected_hw),
            )
            h, w = expected_hw
            final_output_shape_hw = (h, w)
        else:
            _pcb(
                "assemble_error_mismatch_final_shape_reproj_coadd",
                prog=None,
                lvl="ERROR",
                provided=str(final_output_shape_hw),
                expected=str(expected_hw),
            )
            return None, None

    # Convertir la sortie WCS en header FITS si possible une seule fois
    output_header = (
        final_output_wcs.to_header()
        if hasattr(final_output_wcs, "to_header")
        else final_output_wcs
    )


    input_data_all_tiles_HWC_processed = []
    hdr_for_output = None
    for idx, (tile_path, tile_wcs) in enumerate(master_tile_fits_with_wcs_list, 1):
        with fits.open(tile_path, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float32)

        # Master tiles saved via ``save_fits_image`` use the ``HWC`` axis order
        # which stores color images in ``C x H x W`` within the FITS file.  When
        # reading them back for final assembly we expect ``H x W x C``.
        # If the first axis has length 3 and differs from the last axis we
        # convert back to ``HWC``.  This avoids passing arrays of shape
        # ``(3, H, W)`` to ``reproject_and_coadd`` which would produce an
        # invalid coverage map consisting of thin lines only.
        if data.ndim == 3 and data.shape[0] in (1, 3) and data.shape[-1] != data.shape[0]:
            data = np.moveaxis(data, 0, -1)
        if data.ndim == 2:
            data = data[..., np.newaxis]

        if (
            apply_crop
            and crop_percent > 1e-3
            and ZEMOSAIC_UTILS_AVAILABLE
            and hasattr(zemosaic_utils, "crop_image_and_wcs")
        ):
            try:
                cropped, cropped_wcs = zemosaic_utils.crop_image_and_wcs(
                    data,
                    tile_wcs,
                    crop_percent / 100.0,
                    progress_callback=None,
                )
                if cropped is not None and cropped_wcs is not None:
                    data = cropped
                    tile_wcs = cropped_wcs
            except Exception:
                pass

        if re_solve_cropped_tiles and solver_instance is not None:
            try:
                hdr = fits.Header()
                hdr['BITPIX'] = -32
                if 'BSCALE' in hdr:
                    del hdr['BSCALE']
                if 'BZERO' in hdr:
                    del hdr['BZERO']
                use_hints = solver_settings.get("use_radec_hints", False) if solver_settings else False
                if use_hints and hasattr(tile_wcs, "wcs"):
                    cx = tile_wcs.pixel_shape[0] / 2
                    cy = tile_wcs.pixel_shape[1] / 2
                    ra_dec = tile_wcs.wcs_pix2world([[cx, cy]], 0)[0]
                    hdr["RA"] = ra_dec[0]
                    hdr["DEC"] = ra_dec[1]
                solver_instance.solve(
                    str(tile_path), hdr, solver_settings or {}, update_header_with_solution=True
                )
                hdr_for_output = hdr
            except Exception:
                pass

        input_data_all_tiles_HWC_processed.append((data, tile_wcs))

        if idx % 10 == 0 or idx == len(master_tile_fits_with_wcs_list):
            _pcb(
                "assemble_progress_tiles_processed_inc",
                prog=None,
                lvl="INFO_DETAIL",
                num_done=idx,
                total_num=len(master_tile_fits_with_wcs_list),
            )



    # Build kwargs dynamically to remain compatible with different reproject versions
    reproj_kwargs = {}
    try:
        import inspect
        sig = inspect.signature(reproject_and_coadd)
        if "match_background" in sig.parameters:
            reproj_kwargs["match_background"] = match_bg
        elif "match_bg" in sig.parameters:
            reproj_kwargs["match_bg"] = match_bg
        if "process_workers" in sig.parameters:
            reproj_kwargs["process_workers"] = assembly_process_workers
        if "use_memmap" in sig.parameters:
            reproj_kwargs["use_memmap"] = use_memmap
        elif "intermediate_memmap" in sig.parameters:
            reproj_kwargs["intermediate_memmap"] = use_memmap
        if "memmap_dir" in sig.parameters:
            reproj_kwargs["memmap_dir"] = memmap_dir
        if "cleanup_memmap" in sig.parameters:
            reproj_kwargs["cleanup_memmap"] = False
    except Exception:
        # If introspection fails just fall back to basic arguments
        reproj_kwargs = {"match_background": match_bg}


    mosaic_channels = []
    coverage = None
    try:
        total_steps = n_channels
        start_time_loop = time.time()
        last_time = start_time_loop
        step_times = []
        for ch in range(n_channels):

            data_list = [arr[..., ch] for arr, _w in input_data_all_tiles_HWC_processed]
            wcs_list = [wcs for _arr, wcs in input_data_all_tiles_HWC_processed]

            chan_mosaic, chan_cov = reproject_and_coadd_wrapper(
                data_list=data_list,
                wcs_list=wcs_list,
                shape_out=final_output_shape_hw,

                output_projection=output_header,
                use_gpu=use_gpu,
                cpu_func=reproject_and_coadd,

                reproject_function=reproject_interp,
                combine_function="mean",
                **reproj_kwargs,
            )
            mosaic_channels.append(chan_mosaic.astype(np.float32))
            if coverage is None:
                coverage = chan_cov.astype(np.float32)
            now = time.time()
            step_times.append(now - last_time)
            last_time = now
            if progress_callback:
                try:
                    progress_callback("phase5_reproject", ch + 1, total_steps)
                except Exception:
                    pass
    except Exception as e_reproject:
        _pcb("assemble_error_reproject_coadd_call_failed", lvl="ERROR", error=str(e_reproject))
        logger.error(
            "Erreur fatale lors de l'appel à reproject_and_coadd:",
            exc_info=True,
        )
        return None, None

    mosaic_data = np.stack(mosaic_channels, axis=-1)
    if step_times:
        avg_step = sum(step_times) / len(step_times)
        total_elapsed = time.time() - start_time_loop
        _pcb(
            "assemble_debug_reproject_timing",
            prog=None,
            lvl="DEBUG_DETAIL",
            avg=f"{avg_step:.2f}",
            total=f"{total_elapsed:.2f}",
        )
    if re_solve_cropped_tiles and solver_instance is not None and hdr_for_output is not None:
        try:
            fits.writeto("final_mosaic.fits", mosaic_data.astype(np.float32), hdr_for_output, overwrite=True)
        except Exception:
            pass

    if use_memmap and cleanup_memmap and memmap_dir:
        try:
            shutil.rmtree(memmap_dir)
        except OSError:
            pass

    _log_memory_usage(progress_callback, "Fin assemble_final_mosaic_reproject_coadd")
    _pcb(
        "assemble_info_finished_reproject_coadd",
        prog=None,
        lvl="INFO",
        shape=mosaic_data.shape if mosaic_data is not None else "N/A",
    )

    return mosaic_data.astype(np.float32), coverage.astype(np.float32)

# Backwards compatibility alias expected by tests
assemble_final_mosaic_with_reproject_coadd = assemble_final_mosaic_reproject_coadd


def prepare_tiles_and_calc_grid(
    tiles_with_wcs: list,
    crop_percent: float = 0.0,
    re_solve_cropped_tiles: bool = False,
    solver_settings: dict | None = None,
    solver_instance=None,
    drizzle_scale_factor: float = 1.0,
    progress_callback: Callable | None = None,
):
    wcs_list = []
    shape_list = []
    for path, w in tiles_with_wcs:
        current_wcs = w
        if re_solve_cropped_tiles and solver_instance is not None:
            try:
                solved = solver_instance.solve(path, w.to_header(), solver_settings or {}, update_header_with_solution=True)
                if solved:
                    current_wcs = solved
            except Exception:
                pass
        wcs_list.append(current_wcs)
        if hasattr(current_wcs, "pixel_shape"):
            shape_list.append((current_wcs.pixel_shape[1], current_wcs.pixel_shape[0]))
        else:
            shape_list.append((0, 0))
    return _calculate_final_mosaic_grid(wcs_list, shape_list, drizzle_scale_factor, progress_callback)




def run_hierarchical_mosaic(
    input_folder: str,
    output_folder: str,
    astap_exe_path: str,
    astap_data_dir_param: str,
    astap_search_radius_config: float,
    astap_downsample_config: int,
    astap_sensitivity_config: int,
    cluster_threshold_config: float,
    progress_callback: callable,
    stack_norm_method: str,
    stack_weight_method: str,
    stack_reject_algo: str,
    stack_kappa_low: float,
    stack_kappa_high: float,
    parsed_winsor_limits: tuple[float, float],
    stack_final_combine: str,
    apply_radial_weight_config: bool,
    radial_feather_fraction_config: float,
    radial_shape_power_config: float,
    min_radial_weight_floor_config: float,
    final_assembly_method_config: str,
    num_base_workers_config: int,
        # --- ARGUMENTS POUR LE ROGNAGE ---
    apply_master_tile_crop_config: bool,
    master_tile_crop_percent_config: float,
    save_final_as_uint16_config: bool,

    coadd_use_memmap_config: bool,
    coadd_memmap_dir_config: str,
    coadd_cleanup_memmap_config: bool,
    assembly_process_workers_config: int,
    auto_limit_frames_per_master_tile_config: bool,
    winsor_worker_limit_config: int,
    max_raw_per_master_tile_config: int,
    use_gpu_phase5: bool = False,
    gpu_id_phase5: int | None = None,
    solver_settings: dict | None = None
):
    """
    Orchestre le traitement de la mosaïque hiérarchique.

    Parameters
    ----------
    winsor_worker_limit_config : int
        Nombre maximal de workers pour la phase de rejet Winsorized.
    """
    pcb = lambda msg_key, prog=None, lvl="INFO", **kwargs: _log_and_callback(msg_key, prog, lvl, callback=progress_callback, **kwargs)
    
    def update_gui_eta(eta_seconds_total):
        if progress_callback and callable(progress_callback):
            eta_str = "--:--:--"
            if eta_seconds_total is not None and eta_seconds_total >= 0:
                h, rem = divmod(int(eta_seconds_total), 3600); m, s = divmod(rem, 60)
                eta_str = f"{h:02d}:{m:02d}:{s:02d}"
            pcb(f"ETA_UPDATE:{eta_str}", prog=None, lvl="ETA_LEVEL") 


    # Seuil de clustering : valeur de repli à 0.08° si l'option est absente ou non positive
    try:
        cluster_threshold = float(cluster_threshold_config or 0)
    except (TypeError, ValueError):
        cluster_threshold = 0
    SEESTAR_STACK_CLUSTERING_THRESHOLD_DEG = (
        cluster_threshold if cluster_threshold > 0 else 0.08

    )
    PROGRESS_WEIGHT_PHASE1_RAW_SCAN = 30; PROGRESS_WEIGHT_PHASE2_CLUSTERING = 5
    PROGRESS_WEIGHT_PHASE3_MASTER_TILES = 35; PROGRESS_WEIGHT_PHASE4_GRID_CALC = 5
    PROGRESS_WEIGHT_PHASE5_ASSEMBLY = 15; PROGRESS_WEIGHT_PHASE6_SAVE = 8
    PROGRESS_WEIGHT_PHASE7_CLEANUP = 2

    DEFAULT_PHASE_WORKER_RATIO = 1.0
    ALIGNMENT_PHASE_WORKER_RATIO = 0.5  # Limit aggressive phases to 50% of base workers

    if use_gpu_phase5 and gpu_id_phase5 is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id_phase5)
        try:
            import cupy
            cupy.cuda.Device(0).use()
        except Exception as e:
            pcb(
                "run_error_gpu_init_failed",
                prog=None,
                lvl="ERROR",
                error=str(e),
            )
            use_gpu_phase5 = False
    else:
        for v in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
            os.environ.pop(v, None)

    # Determine final GPU usage flag only if a valid NVIDIA GPU is selected
    use_gpu_phase5_flag = (
        use_gpu_phase5 and gpu_id_phase5 is not None and gpu_is_available()
    )
    def _compute_phase_workers(base_workers: int, num_tasks: int, ratio: float = DEFAULT_PHASE_WORKER_RATIO) -> int:
        workers = max(1, int(base_workers * ratio))
        if num_tasks > 0:
            workers = min(workers, num_tasks)
        return max(1, workers)
    current_global_progress = 0
    
    error_messages_deps = []
    if not (ASTROPY_AVAILABLE and WCS and SkyCoord and Angle and fits and u): error_messages_deps.append("Astropy")
    if not (REPROJECT_AVAILABLE and find_optimal_celestial_wcs and reproject_and_coadd and reproject_interp): error_messages_deps.append("Reproject")
    if not (ZEMOSAIC_UTILS_AVAILABLE and zemosaic_utils): error_messages_deps.append("zemosaic_utils")
    if not (ZEMOSAIC_ASTROMETRY_AVAILABLE and zemosaic_astrometry): error_messages_deps.append("zemosaic_astrometry")
    if not (ZEMOSAIC_ALIGN_STACK_AVAILABLE and zemosaic_align_stack): error_messages_deps.append("zemosaic_align_stack")
    try: import psutil
    except ImportError: error_messages_deps.append("psutil")
    if error_messages_deps:
        pcb("run_error_critical_deps_missing", prog=None, lvl="ERROR", modules=", ".join(error_messages_deps)); return

    start_time_total_run = time.monotonic()
    pcb("CHRONO_START_REQUEST", prog=None, lvl="CHRONO_LEVEL")
    _log_memory_usage(progress_callback, "Début Run Hierarchical Mosaic")
    pcb("run_info_processing_started", prog=current_global_progress, lvl="INFO")
    pcb(f"  Config ASTAP: Exe='{os.path.basename(astap_exe_path) if astap_exe_path else 'N/A'}', Data='{os.path.basename(astap_data_dir_param) if astap_data_dir_param else 'N/A'}', Radius={astap_search_radius_config}deg, Downsample={astap_downsample_config}, Sens={astap_sensitivity_config}", prog=None, lvl="DEBUG_DETAIL")
    pcb(f"  Config Workers (GUI): Base demandé='{num_base_workers_config}' (0=auto)", prog=None, lvl="DEBUG_DETAIL")
    pcb(f"  Options Stacking (Master Tuiles): Norm='{stack_norm_method}', Weight='{stack_weight_method}', Reject='{stack_reject_algo}', Combine='{stack_final_combine}', RadialWeight={apply_radial_weight_config} (Feather={radial_feather_fraction_config if apply_radial_weight_config else 'N/A'}, Power={radial_shape_power_config if apply_radial_weight_config else 'N/A'}, Floor={min_radial_weight_floor_config if apply_radial_weight_config else 'N/A'})", prog=None, lvl="DEBUG_DETAIL")
    pcb(f"  Options Assemblage Final: Méthode='{final_assembly_method_config}'", prog=None, lvl="DEBUG_DETAIL")

    time_per_raw_file_wcs = None; time_per_master_tile_creation = None
    cache_dir_name = ".zemosaic_img_cache"; temp_image_cache_dir = os.path.join(output_folder, cache_dir_name)
    try:
        if os.path.exists(temp_image_cache_dir): shutil.rmtree(temp_image_cache_dir)
        os.makedirs(temp_image_cache_dir, exist_ok=True)
    except OSError as e_mkdir_cache: 
        pcb("run_error_cache_dir_creation_failed", prog=None, lvl="ERROR", directory=temp_image_cache_dir, error=str(e_mkdir_cache)); return

# --- Phase 1 (Prétraitement et WCS) ---
    base_progress_phase1 = current_global_progress
    _log_memory_usage(progress_callback, "Début Phase 1 (Prétraitement)")
    pcb("run_info_phase1_started_cache", prog=base_progress_phase1, lvl="INFO")
    
    fits_file_paths = []
    # Scan des fichiers FITS dans le dossier d'entrée et ses sous-dossiers
    for root_dir_iter, _, files_in_dir_iter in os.walk(input_folder):
        for file_name_iter in files_in_dir_iter:
            if file_name_iter.lower().endswith((".fit", ".fits")): 
                fits_file_paths.append(os.path.join(root_dir_iter, file_name_iter))
    
    if not fits_file_paths: 
        pcb("run_error_no_fits_found_input", prog=current_global_progress, lvl="ERROR")
        return # Sortie anticipée si aucun fichier FITS n'est trouvé

    num_total_raw_files = len(fits_file_paths)
    pcb("run_info_found_potential_fits", prog=base_progress_phase1, lvl="INFO_DETAIL", num_files=num_total_raw_files)
    
    # --- Détermination du nombre de workers de BASE ---
    effective_base_workers = 0
    num_logical_processors = os.cpu_count() or 1 
    
    if num_base_workers_config <= 0: # Mode automatique (0 de la GUI)
        desired_auto_ratio = 0.75
        effective_base_workers = max(1, int(np.ceil(num_logical_processors * desired_auto_ratio)))
        pcb(f"WORKERS_CONFIG: Mode Auto. Base de workers calculée: {effective_base_workers} ({desired_auto_ratio*100:.0f}% de {num_logical_processors} processeurs logiques)", prog=None, lvl="INFO_DETAIL")
    else: # Mode manuel
        effective_base_workers = min(num_base_workers_config, num_logical_processors)
        if effective_base_workers < num_base_workers_config:
             pcb(f"WORKERS_CONFIG: Demande GUI ({num_base_workers_config}) limitée à {effective_base_workers} (total processeurs logiques: {num_logical_processors}).", prog=None, lvl="WARN")
        pcb(f"WORKERS_CONFIG: Mode Manuel. Base de workers: {effective_base_workers}", prog=None, lvl="INFO_DETAIL")
    
    if effective_base_workers <= 0: # Fallback
        effective_base_workers = 1
        pcb(f"WORKERS_CONFIG: AVERT - effective_base_workers était <= 0, forcé à 1.", prog=None, lvl="WARN")

    # Calcul du nombre de workers pour la Phase 1
    actual_num_workers_ph1 = _compute_phase_workers(
        effective_base_workers,
        num_total_raw_files,
        DEFAULT_PHASE_WORKER_RATIO,
    )
    pcb(
        f"WORKERS_PHASE1: Utilisation de {actual_num_workers_ph1} worker(s). (Base: {effective_base_workers}, Fichiers: {num_total_raw_files})",
        prog=None,
        lvl="INFO",
    )  # Log mis à jour pour plus de clarté
    
    start_time_phase1 = time.monotonic()
    all_raw_files_processed_info_dict = {} # Pour stocker les infos des fichiers traités avec succès
    files_processed_count_ph1 = 0      # Compteur pour les fichiers soumis au ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=actual_num_workers_ph1, thread_name_prefix="ZeMosaic_Ph1_") as executor_ph1:
        batch_size = 200
        for i in range(0, len(fits_file_paths), batch_size):
            batch = fits_file_paths[i:i+batch_size]
            future_to_filepath_ph1 = {
                executor_ph1.submit(
                    get_wcs_and_pretreat_raw_file,
                    f_path,
                    astap_exe_path,
                    astap_data_dir_param,
                    astap_search_radius_config,
                    astap_downsample_config,
                    astap_sensitivity_config,
                    180,
                    progress_callback,
                    temp_image_cache_dir,
                    solver_settings
                ): f_path for f_path in batch
            }

            for future in as_completed(future_to_filepath_ph1):
                file_path_original = future_to_filepath_ph1[future]
                files_processed_count_ph1 += 1  # Incrémenter pour chaque future terminée

                prog_step_phase1 = base_progress_phase1 + int(
                    PROGRESS_WEIGHT_PHASE1_RAW_SCAN * (files_processed_count_ph1 / max(1, num_total_raw_files))
                )

                try:
                    # Récupérer le résultat de la tâche
                    img_data_adu, wcs_obj_solved, header_obj_updated, hp_mask_path = future.result()

                    # Si la tâche a réussi (ne retourne pas que des None)
                    if (
                        img_data_adu is not None
                        and wcs_obj_solved is not None
                        and header_obj_updated is not None
                    ):
                        # Sauvegarder les données prétraitées en .npy
                        cache_file_basename = f"preprocessed_{os.path.splitext(os.path.basename(file_path_original))[0]}_{files_processed_count_ph1}.npy"
                        cached_image_path = os.path.join(temp_image_cache_dir, cache_file_basename)
                        try:
                            np.save(cached_image_path, img_data_adu)
                            # Stocker les informations pour les phases suivantes
                            all_raw_files_processed_info_dict[file_path_original] = {
                                'path_raw': file_path_original,
                                'path_preprocessed_cache': cached_image_path,
                                'path_hotpix_mask': hp_mask_path,
                                'wcs': wcs_obj_solved,
                                'header': header_obj_updated,
                            }
                        except Exception as e_save_npy:
                            pcb(
                                "run_error_phase1_save_npy_failed",
                                prog=prog_step_phase1,
                                lvl="ERROR",
                                filename=os.path.basename(file_path_original),
                                error=str(e_save_npy),
                            )
                            logger.error(f"Erreur sauvegarde NPY pour {file_path_original}:", exc_info=True)
                        finally:
                            # Libérer la mémoire des données image dès que possible
                            del img_data_adu
                            gc.collect()
                    else:
                        # Le fichier a échoué (ex: WCS non résolu et déplacé)
                        # get_wcs_and_pretreat_raw_file a déjà loggué l'échec spécifique.
                        pcb(
                            "run_warn_phase1_wcs_pretreat_failed_or_skipped_thread",
                            prog=prog_step_phase1,
                            lvl="WARN",
                            filename=os.path.basename(file_path_original),
                        )
                        if img_data_adu is not None:
                            del img_data_adu
                            gc.collect()

                except Exception as exc_thread:
                    # Erreur imprévue dans la future elle-même
                    pcb(
                        "run_error_phase1_thread_exception",
                        prog=prog_step_phase1,
                        lvl="ERROR",
                        filename=os.path.basename(file_path_original),
                        error=str(exc_thread),
                    )
                    logger.error(
                        f"Exception non gérée dans le thread Phase 1 pour {file_path_original}:",
                        exc_info=True,
                    )

                # Log de mémoire et ETA
                if (
                    files_processed_count_ph1 % max(1, num_total_raw_files // 10) == 0
                    or files_processed_count_ph1 == num_total_raw_files
                ):
                    _log_memory_usage(
                        progress_callback,
                        f"Phase 1 - Traité {files_processed_count_ph1}/{num_total_raw_files}",
                    )

                elapsed_phase1 = time.monotonic() - start_time_phase1
                if files_processed_count_ph1 > 0:
                    time_per_raw_file_wcs = elapsed_phase1 / files_processed_count_ph1
                    eta_phase1_sec = (num_total_raw_files - files_processed_count_ph1) * time_per_raw_file_wcs
                    current_progress_in_run_percent = base_progress_phase1 + (
                        files_processed_count_ph1 / max(1, num_total_raw_files)
                    ) * PROGRESS_WEIGHT_PHASE1_RAW_SCAN
                    time_per_percent_point_global = (
                        (time.monotonic() - start_time_total_run) / max(1, current_progress_in_run_percent)
                        if current_progress_in_run_percent > 0
                        else (time.monotonic() - start_time_total_run)
                    )
                    total_eta_sec = eta_phase1_sec + (
                        100 - current_progress_in_run_percent
                    ) * time_per_percent_point_global
                    update_gui_eta(total_eta_sec)

    # Construire la liste finale des informations des fichiers traités avec succès
    all_raw_files_processed_info = [
        all_raw_files_processed_info_dict[fp] 
        for fp in fits_file_paths 
        if fp in all_raw_files_processed_info_dict
    ]
    
    if not all_raw_files_processed_info: 
        pcb("run_error_phase1_no_valid_raws_after_cache", prog=(base_progress_phase1 + PROGRESS_WEIGHT_PHASE1_RAW_SCAN), lvl="ERROR")
        return # Sortie anticipée si aucun fichier n'a pu être traité avec succès

    current_global_progress = base_progress_phase1 + PROGRESS_WEIGHT_PHASE1_RAW_SCAN
    _log_memory_usage(progress_callback, "Fin Phase 1 (Prétraitement)")
    pcb("run_info_phase1_finished_cache", prog=current_global_progress, lvl="INFO", num_valid_raws=len(all_raw_files_processed_info))
    if time_per_raw_file_wcs: 
        pcb(f"    Temps moyen/brute (P1): {time_per_raw_file_wcs:.2f}s", prog=None, lvl="DEBUG")

    # --- Phase 2 (Clustering) ---
    base_progress_phase2 = current_global_progress
    _log_memory_usage(progress_callback, "Début Phase 2 (Clustering)")
    pcb("run_info_phase2_started", prog=base_progress_phase2, lvl="INFO")
    seestar_stack_groups = cluster_seestar_stacks(all_raw_files_processed_info, SEESTAR_STACK_CLUSTERING_THRESHOLD_DEG, progress_callback)
    if not seestar_stack_groups:
        pcb("run_error_phase2_no_groups", prog=(base_progress_phase2 + PROGRESS_WEIGHT_PHASE2_CLUSTERING), lvl="ERROR")
        return
    if max_raw_per_master_tile_config and max_raw_per_master_tile_config > 0:
        new_groups = []
        for g in seestar_stack_groups:
            for i in range(0, len(g), max_raw_per_master_tile_config):
                new_groups.append(g[i:i + max_raw_per_master_tile_config])
        if len(new_groups) != len(seestar_stack_groups):
            pcb(
                "clusterstacks_info_groups_split_manual_limit",
                prog=None,
                lvl="INFO_DETAIL",
                original=len(seestar_stack_groups),
                new=len(new_groups),
                limit=max_raw_per_master_tile_config,
            )
        seestar_stack_groups = new_groups
    cpu_total = os.cpu_count() or 1
    winsor_worker_limit = max(1, min(int(winsor_worker_limit_config), cpu_total))
    pcb(
        f"Winsor worker limit set to {winsor_worker_limit}" + (
            " (ProcessPoolExecutor enabled)" if winsor_worker_limit > 1 else ""
        ),
        prog=None,
        lvl="INFO",
    )
    manual_limit = max_raw_per_master_tile_config
    if auto_limit_frames_per_master_tile_config:
        try:
            sample_path = seestar_stack_groups[0][0].get('path_preprocessed_cache')
            sample_arr = np.load(sample_path, mmap_mode='r')
            bytes_per_frame = sample_arr.nbytes
            sample_shape = sample_arr.shape
            sample_arr = None
            available_bytes = psutil.virtual_memory().available
            expected_workers = max(1, int(effective_base_workers * ALIGNMENT_PHASE_WORKER_RATIO))
            limit = max(
                1,
                int(
                    available_bytes // (expected_workers * bytes_per_frame * 6)
                ),
            )
            if manual_limit > 0:
                limit = min(limit, manual_limit)
            winsor_worker_limit = min(winsor_worker_limit, limit)
            new_groups = []
            for g in seestar_stack_groups:
                for i in range(0, len(g), limit):
                    new_groups.append(g[i:i+limit])
            if len(new_groups) != len(seestar_stack_groups):
                pcb(
                    "clusterstacks_info_groups_split_auto_limit",
                    prog=None,
                    lvl="INFO_DETAIL",
                    original=len(seestar_stack_groups),
                    new=len(new_groups),
                    limit=limit,
                    shape=str(sample_shape),
                )
            seestar_stack_groups = new_groups
            if manual_limit > 0 and limit != manual_limit:
                logger.info(
                    "Manual frame limit (%d) is lower than auto limit, using manual value.",
                    manual_limit,
                )
        except Exception as e_auto:
            pcb("clusterstacks_warn_auto_limit_failed", prog=None, lvl="WARN", error=str(e_auto))
    current_global_progress = base_progress_phase2 + PROGRESS_WEIGHT_PHASE2_CLUSTERING
    num_seestar_stacks_to_process = len(seestar_stack_groups)
    _log_memory_usage(progress_callback, "Fin Phase 2"); pcb("run_info_phase2_finished", prog=current_global_progress, lvl="INFO", num_groups=num_seestar_stacks_to_process)



    # --- Phase 3 (Création Master Tuiles) ---
    base_progress_phase3 = current_global_progress
    _log_memory_usage(progress_callback, "Début Phase 3 (Master Tuiles)")
    pcb("run_info_phase3_started_from_cache", prog=base_progress_phase3, lvl="INFO")
    temp_master_tile_storage_dir = os.path.join(output_folder, "zemosaic_temp_master_tiles")
    try:
        if os.path.exists(temp_master_tile_storage_dir): shutil.rmtree(temp_master_tile_storage_dir)
        os.makedirs(temp_master_tile_storage_dir, exist_ok=True)
    except OSError as e_mkdir_mt: 
        pcb("run_error_phase3_mkdir_failed", prog=current_global_progress, lvl="ERROR", directory=temp_master_tile_storage_dir, error=str(e_mkdir_mt)); return
        
    master_tiles_results_list_temp = {}
    start_time_phase3 = time.monotonic()
    
    # Calcul des workers pour la Phase 3 (alignement/stacking des groupes)
    actual_num_workers_ph3 = _compute_phase_workers(
        effective_base_workers,
        num_seestar_stacks_to_process,
        ALIGNMENT_PHASE_WORKER_RATIO,
    )
    pcb(
        f"WORKERS_PHASE3: Utilisation de {actual_num_workers_ph3} worker(s). (Base: {effective_base_workers}, Ratio {ALIGNMENT_PHASE_WORKER_RATIO*100:.0f}%, Groupes: {num_seestar_stacks_to_process})",
        prog=None,
        lvl="INFO",
    )  # Log mis à jour pour clarté

    tiles_processed_count_ph3 = 0
    # Envoyer l'info initiale avant la boucle
    if num_seestar_stacks_to_process > 0:
        pcb(f"MASTER_TILE_COUNT_UPDATE:{tiles_processed_count_ph3}/{num_seestar_stacks_to_process}", prog=None, lvl="ETA_LEVEL")
    
    executor_ph3 = ThreadPoolExecutor(max_workers=actual_num_workers_ph3, thread_name_prefix="ZeMosaic_Ph3_")

    future_to_group_index = {
        executor_ph3.submit(
            create_master_tile,
            sg_info_list,
            i_stk,  # tile_id
            temp_master_tile_storage_dir,
            stack_norm_method, stack_weight_method, stack_reject_algo,
            stack_kappa_low, stack_kappa_high, parsed_winsor_limits,
            stack_final_combine,
            apply_radial_weight_config, radial_feather_fraction_config,
            radial_shape_power_config, min_radial_weight_floor_config,
            astap_exe_path, astap_data_dir_param, astap_search_radius_config,
            astap_downsample_config, astap_sensitivity_config, 180,  # timeout ASTAP
            winsor_worker_limit,
            progress_callback
        ): i_stk for i_stk, sg_info_list in enumerate(seestar_stack_groups)
    }

    start_time_loop_ph3 = time.time()
    last_time_loop_ph3 = start_time_loop_ph3
    step_times_ph3 = []

    for future in as_completed(future_to_group_index):
            
            group_index_original = future_to_group_index[future]
            tiles_processed_count_ph3 += 1
            
            # --- ENVOYER LA MISE À JOUR DU COMPTEUR DE TUILES ---
            pcb(f"MASTER_TILE_COUNT_UPDATE:{tiles_processed_count_ph3}/{num_seestar_stacks_to_process}", prog=None, lvl="ETA_LEVEL")
            # --- FIN ENVOI MISE À JOUR ---
            
            prog_step_phase3 = base_progress_phase3 + int(PROGRESS_WEIGHT_PHASE3_MASTER_TILES * (tiles_processed_count_ph3 / max(1, num_seestar_stacks_to_process)))
            if progress_callback:
                try:
                    progress_callback("phase3_master_tiles", tiles_processed_count_ph3, num_seestar_stacks_to_process)
                except Exception:
                    pass

            now = time.time()
            step_times_ph3.append(now - last_time_loop_ph3)
            last_time_loop_ph3 = now
            try:
                mt_result_path, mt_result_wcs = future.result()
                if mt_result_path and mt_result_wcs: 
                    master_tiles_results_list_temp[group_index_original] = (mt_result_path, mt_result_wcs)
                else: 
                    pcb("run_warn_phase3_master_tile_creation_failed_thread", prog=prog_step_phase3, lvl="WARN", stack_num=group_index_original + 1)
            except Exception as exc_thread_ph3: 
                pcb("run_error_phase3_thread_exception", prog=prog_step_phase3, lvl="ERROR", stack_num=group_index_original + 1, error=str(exc_thread_ph3))
                logger.error(f"Exception Phase 3 pour stack {group_index_original + 1}:", exc_info=True)
            
            if tiles_processed_count_ph3 % max(1, num_seestar_stacks_to_process // 5) == 0 or tiles_processed_count_ph3 == num_seestar_stacks_to_process : 
                 _log_memory_usage(progress_callback, f"Phase 3 - Traité {tiles_processed_count_ph3}/{num_seestar_stacks_to_process} tuiles")
            
            elapsed_phase3 = time.monotonic() - start_time_phase3
            time_per_master_tile_creation = elapsed_phase3 / max(1, tiles_processed_count_ph3)
            eta_phase3_sec = (num_seestar_stacks_to_process - tiles_processed_count_ph3) * time_per_master_tile_creation
            current_progress_in_run_percent_ph3 = base_progress_phase3 + (tiles_processed_count_ph3 / max(1, num_seestar_stacks_to_process)) * PROGRESS_WEIGHT_PHASE3_MASTER_TILES
            time_per_percent_point_global_ph3 = (time.monotonic() - start_time_total_run) / max(1, current_progress_in_run_percent_ph3) if current_progress_in_run_percent_ph3 > 0 else (time.monotonic() - start_time_total_run)
            total_eta_sec_ph3 = eta_phase3_sec + (100 - current_progress_in_run_percent_ph3) * time_per_percent_point_global_ph3
            update_gui_eta(total_eta_sec_ph3)

    # Toutes les futures sont terminées → fermeture propre
    executor_ph3.shutdown(wait=True)

    master_tiles_results_list = [master_tiles_results_list_temp[i] for i in sorted(master_tiles_results_list_temp.keys())]
    del master_tiles_results_list_temp; gc.collect()
    if not master_tiles_results_list:
        pcb("run_error_phase3_no_master_tiles_created", prog=(base_progress_phase3 + PROGRESS_WEIGHT_PHASE3_MASTER_TILES), lvl="ERROR"); return

    current_global_progress = base_progress_phase3 + PROGRESS_WEIGHT_PHASE3_MASTER_TILES
    _log_memory_usage(progress_callback, "Fin Phase 3");
    if step_times_ph3:
        avg_step = sum(step_times_ph3) / len(step_times_ph3)
        total_elapsed = time.time() - start_time_loop_ph3
        pcb(
            "phase3_debug_timing",
            prog=None,
            lvl="DEBUG_DETAIL",
            avg=f"{avg_step:.2f}",
            total=f"{total_elapsed:.2f}",
        )
    pcb("run_info_phase3_finished_from_cache", prog=current_global_progress, lvl="INFO", num_master_tiles=len(master_tiles_results_list))
    
    # Assurer que le compteur final est bien affiché (au cas où la dernière itération n'aurait pas été exactement le total)
    # Bien que la logique dans la boucle devrait déjà le faire. Peut être redondant mais ne fait pas de mal.
    pcb(f"MASTER_TILE_COUNT_UPDATE:{tiles_processed_count_ph3}/{num_seestar_stacks_to_process}", prog=None, lvl="ETA_LEVEL")

    logger.info("All master tiles complete, entering Phase 5 (reproject & coadd)")
    if progress_callback:
        try:
            progress_callback("run_info_phase3_finished", None, "INFO", num_master_tiles=len(master_tiles_results_list))
        except Exception:
            logger.warning("progress_callback failed for phase3 finished", exc_info=True)




    
    
    # --- Phase 4 (Calcul Grille Finale) ---
    base_progress_phase4 = current_global_progress
    _log_memory_usage(progress_callback, "Début Phase 4 (Calcul Grille)")
    pcb("run_info_phase4_started", prog=base_progress_phase4, lvl="INFO")
    wcs_list_for_final_grid = []; shapes_list_for_final_grid_hw = []
    start_time_loop_ph4 = time.time(); last_time_loop_ph4 = start_time_loop_ph4; step_times_ph4 = []
    total_steps_ph4 = len(master_tiles_results_list)
    for idx_loop, (mt_path_iter,mt_wcs_iter) in enumerate(master_tiles_results_list, 1):
        # ... (logique de récupération shape, inchangée) ...
        if not (mt_path_iter and os.path.exists(mt_path_iter) and mt_wcs_iter and mt_wcs_iter.is_celestial): pcb("run_warn_phase4_invalid_master_tile_for_grid", prog=None, lvl="WARN", path=os.path.basename(mt_path_iter if mt_path_iter else "N/A_path")); continue
        try:
            h_mt_loc,w_mt_loc=0,0
            if mt_wcs_iter.pixel_shape and mt_wcs_iter.pixel_shape[0] > 0 and mt_wcs_iter.pixel_shape[1] > 0 : h_mt_loc,w_mt_loc=mt_wcs_iter.pixel_shape[1],mt_wcs_iter.pixel_shape[0] 
            else: 
                with fits.open(mt_path_iter,memmap=True, do_not_scale_image_data=True) as hdul_mt_s:
                    if hdul_mt_s[0].data is None: pcb("run_warn_phase4_no_data_in_tile_fits", prog=None, lvl="WARN", path=os.path.basename(mt_path_iter)); continue
                    data_shape = hdul_mt_s[0].shape
                    if len(data_shape) == 3:
                        # data_shape == (height, width, channels)
                        h_mt_loc,w_mt_loc = data_shape[0],data_shape[1]
                    elif len(data_shape) == 2: h_mt_loc,w_mt_loc = data_shape[0],data_shape[1]
                    else: pcb("run_warn_phase4_unhandled_tile_shape", prog=None, lvl="WARN", path=os.path.basename(mt_path_iter), shape=data_shape); continue 
                    if mt_wcs_iter and mt_wcs_iter.is_celestial and mt_wcs_iter.pixel_shape is None:
                        try: mt_wcs_iter.pixel_shape=(w_mt_loc,h_mt_loc)
                        except Exception as e_set_ps: pcb("run_warn_phase4_failed_set_pixel_shape", prog=None, lvl="WARN", path=os.path.basename(mt_path_iter), error=str(e_set_ps))
            if h_mt_loc > 0 and w_mt_loc > 0: shapes_list_for_final_grid_hw.append((int(h_mt_loc),int(w_mt_loc))); wcs_list_for_final_grid.append(mt_wcs_iter)
            else: pcb("run_warn_phase4_zero_dimensions_tile", prog=None, lvl="WARN", path=os.path.basename(mt_path_iter))
            now = time.time(); step_times_ph4.append(now - last_time_loop_ph4); last_time_loop_ph4 = now
            if progress_callback:
                try:
                    progress_callback("phase4_grid", idx_loop, total_steps_ph4)
                except Exception:
                    pass
        except Exception as e_read_tile_shape: pcb("run_error_phase4_reading_tile_shape", prog=None, lvl="ERROR", path=os.path.basename(mt_path_iter), error=str(e_read_tile_shape)); logger.error(f"Erreur lecture shape tuile {os.path.basename(mt_path_iter)}:", exc_info=True); continue
    if not wcs_list_for_final_grid or not shapes_list_for_final_grid_hw or len(wcs_list_for_final_grid) != len(shapes_list_for_final_grid_hw): pcb("run_error_phase4_insufficient_tile_info", prog=(base_progress_phase4 + PROGRESS_WEIGHT_PHASE4_GRID_CALC), lvl="ERROR"); return
    final_mosaic_drizzle_scale = 1.0 
    final_output_wcs, final_output_shape_hw = _calculate_final_mosaic_grid(wcs_list_for_final_grid, shapes_list_for_final_grid_hw, final_mosaic_drizzle_scale, progress_callback)
    if not final_output_wcs or not final_output_shape_hw: pcb("run_error_phase4_grid_calc_failed", prog=(base_progress_phase4 + PROGRESS_WEIGHT_PHASE4_GRID_CALC), lvl="ERROR"); return
    current_global_progress = base_progress_phase4 + PROGRESS_WEIGHT_PHASE4_GRID_CALC
    _log_memory_usage(progress_callback, "Fin Phase 4");
    if step_times_ph4:
        avg_step = sum(step_times_ph4) / len(step_times_ph4)
        total_elapsed = time.time() - start_time_loop_ph4
        pcb(
            "phase4_debug_timing",
            prog=None,
            lvl="DEBUG_DETAIL",
            avg=f"{avg_step:.2f}",
            total=f"{total_elapsed:.2f}",
        )
    pcb("run_info_phase4_finished", prog=current_global_progress, lvl="INFO", shape=final_output_shape_hw, crval=final_output_wcs.wcs.crval if final_output_wcs.wcs else 'N/A')

# --- Phase 5 (Assemblage Final) ---
    base_progress_phase5 = current_global_progress
    USE_INCREMENTAL_ASSEMBLY = (final_assembly_method_config == "incremental")
    _log_memory_usage(progress_callback, f"Début Phase 5 (Méthode: {final_assembly_method_config}, Rognage MT Appliqué: {apply_master_tile_crop_config}, %Rognage: {master_tile_crop_percent_config if apply_master_tile_crop_config else 'N/A'})") # Log mis à jour
    
    valid_master_tiles_for_assembly = []
    for mt_p, mt_w in master_tiles_results_list:
        if mt_p and os.path.exists(mt_p) and mt_w and mt_w.is_celestial: 
            valid_master_tiles_for_assembly.append((mt_p, mt_w))
        else:
            pcb("run_warn_phase5_invalid_tile_skipped_for_assembly", prog=None, lvl="WARN", filename=os.path.basename(mt_p if mt_p else 'N/A')) # Clé de log plus spécifique
            
    if not valid_master_tiles_for_assembly: 
        pcb("run_error_phase5_no_valid_tiles_for_assembly", prog=(base_progress_phase5 + PROGRESS_WEIGHT_PHASE5_ASSEMBLY), lvl="ERROR")
        # Nettoyage optionnel ici avant de retourner si besoin
        return

    final_mosaic_data_HWC, final_mosaic_coverage_HW = None, None
    log_key_phase5_failed, log_key_phase5_finished = "", ""

    # Vérification de la disponibilité des fonctions d'assemblage
    # (Tu pourrais les importer en haut du module pour éviter le check 'in globals()' à chaque fois)
    reproject_coadd_available = ('assemble_final_mosaic_reproject_coadd' in globals() and callable(assemble_final_mosaic_reproject_coadd))
    incremental_available = ('assemble_final_mosaic_incremental' in globals() and callable(assemble_final_mosaic_incremental))

    if USE_INCREMENTAL_ASSEMBLY:
        if not incremental_available: 
            pcb("run_error_phase5_inc_func_missing", prog=None, lvl="CRITICAL"); return
        pcb("run_info_phase5_started_incremental", prog=base_progress_phase5, lvl="INFO")
        inc_memmap_dir = temp_master_tile_storage_dir or output_folder
        if use_gpu_phase5_flag:
            try:
                import cupy
                cupy.cuda.Device(0).use()
                final_mosaic_data_HWC, final_mosaic_coverage_HW = zemosaic_utils.gpu_assemble_final_mosaic_incremental(
                    master_tile_fits_with_wcs_list=valid_master_tiles_for_assembly,
                    final_output_wcs=final_output_wcs,
                    final_output_shape_hw=final_output_shape_hw,
                    progress_callback=progress_callback,
                    n_channels=3,
                    apply_crop=apply_master_tile_crop_config,
                    crop_percent=master_tile_crop_percent_config,
                    processing_threads=assembly_process_workers_config,
                    memmap_dir=inc_memmap_dir,
                    cleanup_memmap=True,
                )
            except Exception as e_gpu:
                logger.warning("GPU incremental assembly failed, falling back to CPU: %s", e_gpu)
                final_mosaic_data_HWC, final_mosaic_coverage_HW = assemble_final_mosaic_incremental(
                    master_tile_fits_with_wcs_list=valid_master_tiles_for_assembly,
                    final_output_wcs=final_output_wcs,
                    final_output_shape_hw=final_output_shape_hw,
                    progress_callback=progress_callback,
                    n_channels=3,
                    apply_crop=apply_master_tile_crop_config,
                    crop_percent=master_tile_crop_percent_config,
                    processing_threads=assembly_process_workers_config,
                    memmap_dir=inc_memmap_dir,
                    cleanup_memmap=True,
                )
        else:
            final_mosaic_data_HWC, final_mosaic_coverage_HW = assemble_final_mosaic_incremental(
                master_tile_fits_with_wcs_list=valid_master_tiles_for_assembly,
                final_output_wcs=final_output_wcs,
                final_output_shape_hw=final_output_shape_hw,
                progress_callback=progress_callback,
                n_channels=3,
                apply_crop=apply_master_tile_crop_config,
                crop_percent=master_tile_crop_percent_config,
                processing_threads=assembly_process_workers_config,
                memmap_dir=inc_memmap_dir,
                cleanup_memmap=True,
            )
        log_key_phase5_failed = "run_error_phase5_assembly_failed_incremental"
        log_key_phase5_finished = "run_info_phase5_finished_incremental"
    else: # Méthode Reproject & Coadd
        if not reproject_coadd_available: 
            pcb("run_error_phase5_reproject_coadd_func_missing", prog=None, lvl="CRITICAL"); return
        pcb("run_info_phase5_started_reproject_coadd", prog=base_progress_phase5, lvl="INFO")

        if use_gpu_phase5_flag:
            try:
                import cupy
                cupy.cuda.Device(0).use()
                final_mosaic_data_HWC, final_mosaic_coverage_HW = zemosaic_utils.gpu_assemble_final_mosaic_reproject_coadd(
                    master_tile_fits_with_wcs_list=valid_master_tiles_for_assembly,
                    final_output_wcs=final_output_wcs,
                    final_output_shape_hw=final_output_shape_hw,
                    progress_callback=progress_callback,
                    n_channels=3,
                    match_bg=True,
                    apply_crop=apply_master_tile_crop_config,
                    crop_percent=master_tile_crop_percent_config,
                )
            except Exception as e_gpu:
                logger.warning("GPU reproject_coadd failed, falling back to CPU: %s", e_gpu)
                final_mosaic_data_HWC, final_mosaic_coverage_HW = assemble_final_mosaic_reproject_coadd(
                    master_tile_fits_with_wcs_list=valid_master_tiles_for_assembly,
                    final_output_wcs=final_output_wcs,
                    final_output_shape_hw=final_output_shape_hw,
                    progress_callback=progress_callback,
                    n_channels=3,
                    match_bg=True,
                    apply_crop=apply_master_tile_crop_config,
                    crop_percent=master_tile_crop_percent_config,
                    use_gpu=False,
                )
        else:
            final_mosaic_data_HWC, final_mosaic_coverage_HW = assemble_final_mosaic_reproject_coadd(
                master_tile_fits_with_wcs_list=valid_master_tiles_for_assembly,
                final_output_wcs=final_output_wcs,
                final_output_shape_hw=final_output_shape_hw,
                progress_callback=progress_callback,
                n_channels=3,
                match_bg=True,
                apply_crop=apply_master_tile_crop_config,
                crop_percent=master_tile_crop_percent_config,
                use_gpu=use_gpu_phase5_flag,
            )

        log_key_phase5_failed = "run_error_phase5_assembly_failed_reproject_coadd"
        log_key_phase5_finished = "run_info_phase5_finished_reproject_coadd"

    if final_mosaic_data_HWC is None: 
        pcb(log_key_phase5_failed, prog=(base_progress_phase5 + PROGRESS_WEIGHT_PHASE5_ASSEMBLY), lvl="ERROR")
        # Nettoyage optionnel ici
        return
        
    current_global_progress = base_progress_phase5 + PROGRESS_WEIGHT_PHASE5_ASSEMBLY
    _log_memory_usage(progress_callback, "Fin Phase 5 (Assemblage)")
    pcb(log_key_phase5_finished, prog=current_global_progress, lvl="INFO", 
        shape=final_mosaic_data_HWC.shape if final_mosaic_data_HWC is not None else "N/A")
    

    # --- Phase 6 (Sauvegarde) ---
    base_progress_phase6 = current_global_progress
    _log_memory_usage(progress_callback, "Début Phase 6 (Sauvegarde)")
    pcb("run_info_phase6_started", prog=base_progress_phase6, lvl="INFO")
    output_base_name = f"zemosaic_MT{len(master_tiles_results_list)}_R{len(all_raw_files_processed_info)}"
    final_fits_path = os.path.join(output_folder, f"{output_base_name}.fits")
    
    final_header = fits.Header() 
    if final_output_wcs:
        try: final_header.update(final_output_wcs.to_header(relax=True))
        except Exception as e_hdr_wcs: pcb("run_warn_phase6_wcs_to_header_failed", error=str(e_hdr_wcs), lvl="WARN")
    
    final_header['SOFTWARE']=('ZeMosaic v0.9.4','Mosaic Software') # Incrémente la version si tu le souhaites
    final_header['NMASTILE']=(len(master_tiles_results_list),"Master Tiles combined")
    final_header['NRAWINIT']=(num_total_raw_files,"Initial raw images found")
    final_header['NRAWPROC']=(len(all_raw_files_processed_info),"Raw images with WCS processed")
    # ... (autres clés de config comme ASTAP, Stacking, etc.) ...
    final_header['STK_NORM'] = (str(stack_norm_method), 'Stacking: Normalization Method')
    final_header['STK_WGHT'] = (str(stack_weight_method), 'Stacking: Weighting Method')
    if apply_radial_weight_config:
        final_header['STK_RADW'] = (True, 'Stacking: Radial Weighting Applied')
        final_header['STK_RADFF'] = (radial_feather_fraction_config, 'Stacking: Radial Feather Fraction')
        final_header['STK_RADPW'] = (radial_shape_power_config, 'Stacking: Radial Weight Shape Power')
        final_header['STK_RADFLR'] = (min_radial_weight_floor_config, 'Stacking: Min Radial Weight Floor')
    else:
        final_header['STK_RADW'] = (False, 'Stacking: Radial Weighting Applied')
    final_header['STK_REJ'] = (str(stack_reject_algo), 'Stacking: Rejection Algorithm')
    # ... (kappa, winsor si pertinent pour l'algo de rejet) ...
    final_header['STK_COMB'] = (str(stack_final_combine), 'Stacking: Final Combine Method')
    final_header['ZMASMBMTH'] = (final_assembly_method_config, 'Final Assembly Method')
    final_header['ZM_WORKERS'] = (num_base_workers_config, 'GUI: Base workers config (0=auto)')

    try:
        if not (ZEMOSAIC_UTILS_AVAILABLE and zemosaic_utils): 
            raise RuntimeError("zemosaic_utils non disponible pour sauvegarde FITS.")
        zemosaic_utils.save_fits_image(
            image_data=final_mosaic_data_HWC,
            output_path=final_fits_path,
            header=final_header,
            overwrite=True,
            save_as_float=not save_final_as_uint16_config,
            progress_callback=progress_callback,
            axis_order="HWC",
        )
        
        if final_mosaic_coverage_HW is not None and np.any(final_mosaic_coverage_HW):
            coverage_path = os.path.join(output_folder, f"{output_base_name}_coverage.fits")
            cov_hdr = fits.Header() 
            if ASTROPY_AVAILABLE and final_output_wcs: 
                try: cov_hdr.update(final_output_wcs.to_header(relax=True))
                except: pass 
            cov_hdr['EXTNAME']=('COVERAGE','Coverage Map') 
            cov_hdr['BUNIT']=('count','Pixel contributions or sum of weights')
            zemosaic_utils.save_fits_image(
                final_mosaic_coverage_HW,
                coverage_path,
                header=cov_hdr,
                overwrite=True,
                save_as_float=True,
                progress_callback=progress_callback,
                axis_order="HWC",
            )
            pcb("run_info_coverage_map_saved", prog=None, lvl="INFO_DETAIL", filename=os.path.basename(coverage_path))
        
        current_global_progress = base_progress_phase6 + PROGRESS_WEIGHT_PHASE6_SAVE
        pcb("run_success_mosaic_saved", prog=current_global_progress, lvl="SUCCESS", filename=os.path.basename(final_fits_path))
    except Exception as e_save_m: 
        pcb("run_error_phase6_save_failed", prog=(base_progress_phase6 + PROGRESS_WEIGHT_PHASE6_SAVE), lvl="ERROR", error=str(e_save_m))
        logger.error("Erreur sauvegarde FITS final:", exc_info=True)
        # En cas d'échec de sauvegarde, on ne peut pas générer de preview car final_mosaic_data_HWC pourrait être le problème.
        # On essaie quand même de nettoyer avant de retourner.
        if 'final_mosaic_data_HWC' in locals() and final_mosaic_data_HWC is not None: del final_mosaic_data_HWC
        if 'final_mosaic_coverage_HW' in locals() and final_mosaic_coverage_HW is not None: del final_mosaic_coverage_HW
        gc.collect()
        return

    _log_memory_usage(progress_callback, "Fin Sauvegarde FITS (avant preview)")

    # --- MODIFIÉ : Génération de la Preview PNG avec stretch_auto_asifits_like ---
    if final_mosaic_data_HWC is not None and ZEMOSAIC_UTILS_AVAILABLE and zemosaic_utils:
        pcb("run_info_preview_stretch_started_auto_asifits", prog=None, lvl="INFO_DETAIL") # Log mis à jour
        try:
            # Vérifier si la fonction stretch_auto_asifits_like existe dans zemosaic_utils
            if hasattr(zemosaic_utils, 'stretch_auto_asifits_like') and callable(zemosaic_utils.stretch_auto_asifits_like):
                
                # Paramètres pour stretch_auto_asifits_like (à ajuster si besoin)
                # Ces valeurs sont des exemples, tu devras peut-être les affiner
                # ou les rendre configurables plus tard.
                preview_p_low = 2.5  # Percentile pour le point noir (plus élevé que pour asinh seul)
                preview_p_high = 99.8 # Percentile pour le point blanc initial
                                      # Facteur 'a' pour le stretch asinh après la normalisation initiale
                                      # Pour un stretch plus "doux" similaire à ASIFitsView, 'a' peut être plus grand.
                                      # ASIFitsView utilise souvent un 'midtones balance' (gamma-like) aussi.
                                      # Un 'a' de 10 comme dans ton code de test est très doux. Essayons 0.5 ou 1.0.
                preview_asinh_a = 20.0 # Test avec une valeur plus douce pour le 'a' de asinh

                m_stretched = zemosaic_utils.stretch_auto_asifits_like(
                    final_mosaic_data_HWC,
                    p_low=preview_p_low,
                    p_high=preview_p_high,
                    asinh_a=preview_asinh_a,
                    apply_wb=True  # Applique une balance des blancs automatique
                )

                if m_stretched is not None:
                    img_u8 = (
                        np.nan_to_num(
                            np.clip(m_stretched.astype(np.float32), 0, 1)
                        )
                        * 255
                    ).astype(np.uint8)
                    png_path = os.path.join(output_folder, f"{output_base_name}_preview.png")
                    try: 
                        import cv2 # Importer cv2 seulement si nécessaire
                        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
                        if cv2.imwrite(png_path, img_bgr): 
                            pcb("run_success_preview_saved_auto_asifits", prog=None, lvl="SUCCESS", filename=os.path.basename(png_path))
                        else: 
                            pcb("run_warn_preview_imwrite_failed_auto_asifits", prog=None, lvl="WARN", filename=os.path.basename(png_path))
                    except ImportError: 
                        pcb("run_warn_preview_opencv_missing_for_auto_asifits", prog=None, lvl="WARN")
                    except Exception as e_cv2_prev: 
                        pcb("run_error_preview_opencv_failed_auto_asifits", prog=None, lvl="ERROR", error=str(e_cv2_prev))
                else:
                    pcb("run_error_preview_stretch_auto_asifits_returned_none", prog=None, lvl="ERROR")
            else:
                pcb("run_warn_preview_stretch_auto_asifits_func_missing", prog=None, lvl="WARN")
                # Fallback sur l'ancienne méthode si stretch_auto_asifits_like n'est pas trouvée
                # (Tu peux supprimer ce fallback si tu es sûr que la fonction existe)
                pcb("run_info_preview_fallback_to_simple_asinh", prog=None, lvl="DEBUG_DETAIL")
                if hasattr(zemosaic_utils, 'stretch_percentile_rgb') and zemosaic_utils.ASTROPY_VISUALIZATION_AVAILABLE:
                     m_stretched_fallback = zemosaic_utils.stretch_percentile_rgb(final_mosaic_data_HWC, p_low=0.5, p_high=99.9, independent_channels=False, asinh_a=0.01 )
                     if m_stretched_fallback is not None:
                        img_u8_fb = (np.clip(m_stretched_fallback.astype(np.float32), 0, 1) * 255).astype(np.uint8)
                        png_path_fb = os.path.join(output_folder, f"{output_base_name}_preview_fallback.png")
                        try:
                            import cv2
                            img_bgr_fb = cv2.cvtColor(img_u8_fb, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(png_path_fb, img_bgr_fb)
                            pcb("run_success_preview_saved_fallback", prog=None, lvl="INFO_DETAIL", filename=os.path.basename(png_path_fb))
                        except: pass # Ignorer erreur fallback

        except Exception as e_stretch_main: 
            pcb("run_error_preview_stretch_unexpected_main", prog=None, lvl="ERROR", error=str(e_stretch_main))
            logger.error("Erreur imprévue lors de la génération de la preview:", exc_info=True)
            
    if 'final_mosaic_data_HWC' in locals() and final_mosaic_data_HWC is not None: del final_mosaic_data_HWC
    if 'final_mosaic_coverage_HW' in locals() and final_mosaic_coverage_HW is not None: del final_mosaic_coverage_HW
    gc.collect()



    # --- Phase 7 (Nettoyage) ---
    # ... (contenu Phase 7 inchangé) ...
    base_progress_phase7 = current_global_progress
    _log_memory_usage(progress_callback, "Début Phase 7 (Nettoyage)")
    pcb("run_info_phase7_cleanup_starting", prog=base_progress_phase7, lvl="INFO")
    try:
        if os.path.exists(temp_image_cache_dir): shutil.rmtree(temp_image_cache_dir); pcb("run_info_temp_preprocessed_cache_cleaned", prog=None, lvl="INFO_DETAIL", directory=temp_image_cache_dir)
        if os.path.exists(temp_master_tile_storage_dir): shutil.rmtree(temp_master_tile_storage_dir); pcb("run_info_temp_master_tiles_fits_cleaned", prog=None, lvl="INFO_DETAIL", directory=temp_master_tile_storage_dir)
    except Exception as e_clean_final: pcb("run_warn_phase7_cleanup_failed", prog=None, lvl="WARN", error=str(e_clean_final))
    current_global_progress = base_progress_phase7 + PROGRESS_WEIGHT_PHASE7_CLEANUP; current_global_progress = min(100, current_global_progress)
    _log_memory_usage(progress_callback, "Fin Phase 7"); pcb("CHRONO_STOP_REQUEST", prog=None, lvl="CHRONO_LEVEL"); update_gui_eta(0)
    total_duration_sec = time.monotonic() - start_time_total_run
    pcb("run_success_processing_completed", prog=current_global_progress, lvl="SUCCESS", duration=f"{total_duration_sec:.2f}")
    gc.collect(); _log_memory_usage(progress_callback, "Fin Run Hierarchical Mosaic (après GC final)")
    logger.info(f"===== Run Hierarchical Mosaic COMPLETED in {total_duration_sec:.2f}s =====")
################################################################################
################################################################################
####

def run_hierarchical_mosaic_process(
    progress_queue,
    *args,
    solver_settings_dict=None,
    **kwargs,
):
    """Wrapper for running :func:`run_hierarchical_mosaic` in a separate process."""

    # progress_callback(stage: str, current: int, total: int)

    def queue_callback(*cb_args, **cb_kwargs):
        """Proxy callback used inside the worker process.

        It supports both legacy logging calls and the new progress
        reporting style ``progress_callback(stage, current, total)``.

        Legacy calls are forwarded unchanged as
        ``(message_key_or_raw, progress_value, level, kwargs)`` tuples.
        Stage updates are sent with ``"STAGE_PROGRESS"`` as the message key.
        """
        if (
            len(cb_args) == 3
            and not cb_kwargs
            and isinstance(cb_args[0], str)
            and isinstance(cb_args[1], int)
            and isinstance(cb_args[2], int)
        ):
            stage, current, total = cb_args
            progress_queue.put(("STAGE_PROGRESS", stage, current, {"total": total}))
            return

        message_key_or_raw = cb_args[0] if cb_args else ""
        progress_value = cb_args[1] if len(cb_args) > 1 else None
        level = cb_args[2] if len(cb_args) > 2 else cb_kwargs.pop("level", "INFO")
        if "lvl" in cb_kwargs:
            level = cb_kwargs.pop("lvl")
        progress_queue.put((message_key_or_raw, progress_value, level, cb_kwargs))

    full_args = args[:8] + (queue_callback,) + args[8:]
    try:
        run_hierarchical_mosaic(*full_args, solver_settings=solver_settings_dict, **kwargs)
    except Exception as e_proc:
        progress_queue.put(("PROCESS_ERROR", None, "ERROR", {"error": str(e_proc)}))
    finally:
        progress_queue.put(("PROCESS_DONE", None, "INFO", {}))

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="ZeMosaic worker")
    parser.add_argument("input_folder", help="Folder with input FITS")
    parser.add_argument("output_folder", help="Destination folder")
    parser.add_argument("--config", default=None, help="Optional config JSON")
    parser.add_argument("--coadd_use_memmap", action="store_true",
                        help="Write sum/cov arrays to disk via numpy.memmap")
    parser.add_argument("--coadd_memmap_dir", default=None,
                        help="Directory to store *.dat blocks")
    parser.add_argument("--coadd_cleanup_memmap", action="store_true",
                        default=True,
                        help="Delete *.dat blocks when the run finishes")
    parser.add_argument("--no_auto_limit_frames", action="store_true",
                        help="Disable automatic frame limit per master tile")
    parser.add_argument("--assembly_process_workers", type=int, default=None,
                        help="Number of processes for final assembly (0=auto)")
    parser.add_argument("-W", "--winsor-workers", type=int, default=None,
                        help="Process workers for Winsorized rejection (1-16)")
    parser.add_argument("--max-raw-per-master-tile", type=int, default=None,
                        help="Cap raw frames per master tile (0=auto)")
    parser.add_argument("--solver-settings", default=None,
                        help="Path to solver settings JSON")
    args = parser.parse_args()

    cfg = {}
    if ZEMOSAIC_CONFIG_AVAILABLE and zemosaic_config:
        cfg.update(zemosaic_config.load_config())
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass

    solver_cfg = {}
    if args.solver_settings:
        try:
            solver_cfg = SolverSettings.load(args.solver_settings).__dict__
        except Exception:
            solver_cfg = {}
    else:
        try:
            solver_cfg = SolverSettings.load_default().__dict__
        except Exception:
            solver_cfg = SolverSettings().__dict__

  
