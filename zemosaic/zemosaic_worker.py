# zemosaic_worker.py

import os
import shutil
import time
import traceback
import gc
import logging
import inspect # Pas utilisé directement ici, mais peut être utile pour des introspections futures
import psutil

from concurrent.futures import ThreadPoolExecutor, as_completed

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

try: import zemosaic_utils; ZEMOSAIC_UTILS_AVAILABLE = True; logger.info("Module 'zemosaic_utils' importé.")
except ImportError as e: logger.error(f"Import 'zemosaic_utils.py' échoué: {e}.")
try: import zemosaic_astrometry; ZEMOSAIC_ASTROMETRY_AVAILABLE = True; logger.info("Module 'zemosaic_astrometry' importé.")
except ImportError as e: logger.error(f"Import 'zemosaic_astrometry.py' échoué: {e}.")
try: import zemosaic_align_stack; ZEMOSAIC_ALIGN_STACK_AVAILABLE = True; logger.info("Module 'zemosaic_align_stack' importé.")
except ImportError as e: logger.error(f"Import 'zemosaic_align_stack.py' échoué: {e}.")







# DANS zemosaic_worker.py

# ... (imports et logger configuré comme avant) ...

# --- Helper pour log et callback ---
def _log_and_callback(message_key_or_raw, progress_value=None, level="INFO", callback=None, **kwargs):
    """
    Helper pour loguer un message et appeler le callback GUI.
    - Si level est INFO, WARN, ERROR, SUCCESS, message_key_or_raw est traité comme une clé.
    - Sinon (DEBUG, ETA_LEVEL, etc.), message_key_or_raw est loggué tel quel.
    - Les **kwargs sont passés pour le formatage si message_key_or_raw est une clé.
    """
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




# --- Fonctions Utilitaires Internes au Worker ---
def _calculate_final_mosaic_grid(panel_wcs_list: list, panel_shapes_hw_list: list,
                                 drizzle_scale_factor: float = 1.0, progress_callback: callable = None):
    num_initial_inputs = len(panel_wcs_list)
    # Utilisation de clés pour les messages utilisateur
    _log_and_callback("calcgrid_info_start_calc", num_wcs_shapes=num_initial_inputs, scale_factor=drizzle_scale_factor, level="DEBUG_DETAIL", callback=progress_callback)
    
    if not (REPROJECT_AVAILABLE and find_optimal_celestial_wcs):
        _log_and_callback("calcgrid_error_reproject_unavailable", level="ERROR", callback=progress_callback); return None, None
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
    if not (ASTROPY_AVAILABLE and SkyCoord and u): _log_and_callback("clusterstacks_error_astropy_unavailable", level="ERROR", callback=progress_callback); return []
    if not all_raw_files_with_info: _log_and_callback("clusterstacks_warn_no_raw_info", level="WARN", callback=progress_callback); return []
    _log_and_callback("clusterstacks_info_start", num_files=len(all_raw_files_with_info), threshold=stack_threshold_deg, level="INFO", callback=progress_callback)
    panel_centers_sky = []; panel_data_for_clustering = []
    for i, info in enumerate(all_raw_files_with_info):
        wcs_obj = info['wcs']
        if not (wcs_obj and wcs_obj.is_celestial): continue
        try:
            if wcs_obj.pixel_shape: center_world = wcs_obj.pixel_to_world(wcs_obj.pixel_shape[0]/2.0, wcs_obj.pixel_shape[1]/2.0)
            elif hasattr(wcs_obj.wcs, 'crval'): center_world = SkyCoord(ra=wcs_obj.wcs.crval[0]*u.deg, dec=wcs_obj.wcs.crval[1]*u.deg, frame='icrs')
            else: continue
            panel_centers_sky.append(center_world); panel_data_for_clustering.append(info)
        except Exception: continue
    if not panel_centers_sky: _log_and_callback("clusterstacks_warn_no_centers", level="WARN", callback=progress_callback); return []
    groups = []; assigned_mask = [False] * len(panel_centers_sky)
    for i in range(len(panel_centers_sky)):
        if assigned_mask[i]: continue
        current_group_infos = [panel_data_for_clustering[i]]; assigned_mask[i] = True
        current_group_center_seed = panel_centers_sky[i]
        for j in range(i + 1, len(panel_centers_sky)):
            if assigned_mask[j]: continue
            if current_group_center_seed.separation(panel_centers_sky[j]).deg < stack_threshold_deg:
                current_group_infos.append(panel_data_for_clustering[j]); assigned_mask[j] = True
        groups.append(current_group_infos)
    _log_and_callback("clusterstacks_info_finished", num_groups=len(groups), level="INFO", callback=progress_callback)
    return groups

def get_wcs_and_pretreat_raw_file(file_path: str, astap_exe_path: str, astap_data_dir: str, 
                                  astap_search_radius: float, astap_downsample: int, 
                                  astap_sensitivity: int, astap_timeout_seconds: int, 
                                  progress_callback: callable):
    filename = os.path.basename(file_path)
    # Utiliser une fonction helper pour les logs internes à cette fonction si _log_and_callback
    # est trop lié à la structure de run_hierarchical_mosaic
    _pcb_local = lambda msg_key, lvl="DEBUG", **kwargs: \
        progress_callback(msg_key, None, lvl, **kwargs) if progress_callback else print(f"GETWCS_LOG {lvl}: {msg_key} {kwargs}")

    _pcb_local(f"GetWCS_Pretreat: Début pour '{filename}'.", lvl="DEBUG_DETAIL") # Niveau DEBUG_DETAIL pour être moins verbeux

    if not (ZEMOSAIC_UTILS_AVAILABLE and zemosaic_utils):
        _pcb_local("getwcs_error_utils_unavailable", lvl="ERROR")
        return None, None, None
        
    img_data_raw_adu, header_orig = zemosaic_utils.load_and_validate_fits(
        file_path, 
        normalize_to_float32=False, 
        attempt_fix_nonfinite=True, 
        progress_callback=progress_callback
    )

    if img_data_raw_adu is None or header_orig is None:
        _pcb_local("getwcs_error_load_failed", lvl="ERROR", filename=filename)
        # Le fichier n'a pas pu être chargé, on ne peut pas le déplacer car on ne sait pas s'il existe ou est corrompu.
        # Ou on pourrait essayer de le déplacer s'il existe. Pour l'instant, on retourne None.
        return None, None, None

    # ... (log de post-load) ...
    _pcb_local(f"  Post-Load: '{filename}' - Shape: {img_data_raw_adu.shape}, Dtype: {img_data_raw_adu.dtype}", lvl="DEBUG_VERY_DETAIL")

    img_data_processed_adu = img_data_raw_adu.astype(np.float32, copy=True)
    del img_data_raw_adu; gc.collect()

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
        return None, None, None

    # --- Correction Hot Pixels ---
    _pcb_local(f"  Correction HP pour '{filename}'...", lvl="DEBUG_DETAIL")
    img_data_hp_corrected_adu = zemosaic_utils.detect_and_correct_hot_pixels(img_data_processed_adu,3.,5,progress_callback=progress_callback)
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
            
    if wcs_brute is None and ZEMOSAIC_ASTROMETRY_AVAILABLE and zemosaic_astrometry:
        _pcb_local(f"    WCS non trouvé/valide dans header. Appel solve_with_astap pour '{filename}'.", lvl="DEBUG_DETAIL")
        wcs_brute = zemosaic_astrometry.solve_with_astap(
            image_fits_path=file_path, original_fits_header=header_orig, 
            astap_exe_path=astap_exe_path, astap_data_dir=astap_data_dir, 
            search_radius_deg=astap_search_radius, downsample_factor=astap_downsample, 
            sensitivity=astap_sensitivity, timeout_sec=astap_timeout_seconds, 
            update_original_header_in_place=True, # Important que le header soit mis à jour
            progress_callback=progress_callback
        )
        if wcs_brute: _pcb_local("getwcs_info_astap_solved", lvl="INFO_DETAIL", filename=filename)
        else: _pcb_local("getwcs_warn_astap_failed", lvl="WARN", filename=filename)
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
            return img_data_processed_adu, wcs_brute, header_orig # header_orig peut avoir été mis à jour par ASTAP
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
    return None, None, None # Indique l'échec pour ce fichier








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
    progress_callback: callable
):
    """
    Crée une "master tuile" à partir d'un groupe d'images.
    Lit les données image prétraitées depuis un cache disque (.npy).
    Utilise les WCS et Headers déjà résolus et stockés en mémoire.
    Transmet toutes les options de stacking, y compris la pondération radiale.
    """
    pcb_tile = lambda msg_key, prog=None, lvl="INFO_DETAIL", **kwargs: _log_and_callback(msg_key, prog, lvl, callback=progress_callback, **kwargs)
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
        # --- TRANSMISSION DES NOUVEAUX PARAMÈTRES ---
        apply_radial_weight=apply_radial_weight,
        radial_feather_fraction=radial_feather_fraction,
        radial_shape_power=radial_shape_power,
        # --- FIN TRANSMISSION ---
        progress_callback=progress_callback
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
    crop_percent: float = 0.0
):
    """
    Assemble les master tuiles en une mosaïque finale de manière incrémentale.
    Peut optionnellement rogner les master tuiles avant assemblage.
    """
    pcb_asm = lambda msg_key, prog=None, lvl="INFO_DETAIL", **kwargs: \
        _log_and_callback(msg_key, prog, lvl, callback=progress_callback, **kwargs)

    pcb_asm(f"ASM_INC: Début. Options rognage - Appliquer: {apply_crop}, %: {crop_percent if apply_crop else 'N/A'}", lvl="DEBUG_DETAIL")

    if not (REPROJECT_AVAILABLE and reproject_interp and ASTROPY_AVAILABLE and fits):
        missing_deps = []
        if not REPROJECT_AVAILABLE or not reproject_interp: missing_deps.append("Reproject (reproject_interp)")
        if not ASTROPY_AVAILABLE or not fits : missing_deps.append("Astropy (fits)")
        pcb_asm("assemble_error_core_deps_unavailable_incremental", prog=None, lvl="ERROR", missing=", ".join(missing_deps)); return None, None

    num_master_tiles = len(master_tile_fits_with_wcs_list)
    pcb_asm("assemble_info_start_incremental", prog=None, lvl="INFO", num_tiles=num_master_tiles)
    if not master_tile_fits_with_wcs_list:
        pcb_asm("assemble_error_no_tiles_provided_incremental", prog=None, lvl="ERROR"); return None, None

    final_shape_for_accumulators_hwc = (final_output_shape_hw[0], final_output_shape_hw[1], n_channels)
    pcb_asm("assemble_info_allocating_accumulators", prog=None, lvl="DEBUG_DETAIL", shape=str(final_shape_for_accumulators_hwc), dtype_sum=str(dtype_accumulator), dtype_norm=str(dtype_norm))
    try:
        running_sum_accumulator = np.zeros(final_shape_for_accumulators_hwc, dtype=dtype_accumulator)
        running_norm_accumulator = np.zeros(final_shape_for_accumulators_hwc, dtype=dtype_norm)
        final_pixel_contributions = np.zeros(final_output_shape_hw, dtype=np.float32) 
    except MemoryError as e_mem_acc: 
        pcb_asm("assemble_error_memory_allocating_accumulators", prog=None, lvl="ERROR", error=str(e_mem_acc)); logger.error("MemoryError allocation accumulateurs (incrémental).", exc_info=True); return None, None
    except Exception as e_acc: 
        pcb_asm("assemble_error_allocating_accumulators", prog=None, lvl="ERROR", error=str(e_acc)); logger.error("Erreur allocation accumulateurs (incrémental).", exc_info=True); return None, None
    pcb_asm("assemble_info_accumulators_allocated", prog=None, lvl="DEBUG_DETAIL")

    for tile_idx, (tile_path, mt_wcs_obj_original) in enumerate(master_tile_fits_with_wcs_list, 1):
        pcb_asm("assemble_info_processing_tile", prog=None, lvl="INFO_DETAIL", tile_num=tile_idx, total_tiles=num_master_tiles, filename=os.path.basename(tile_path))
        
        # Initialisation des variables pour ce scope de boucle
        current_tile_data_hwc = None
        data_to_use_for_reproject = None
        wcs_to_use_for_reproject = None
        tile_processed_successfully_this_iteration = False

        try:
            with fits.open(tile_path, memmap=False, do_not_scale_image_data=True) as hdul: # memmap=False est plus sûr pour éviter problèmes de fichiers ouverts
                if not hdul or not hasattr(hdul[0], 'data') or hdul[0].data is None: 
                    pcb_asm("assemble_warn_tile_empty_or_no_data_inc", prog=None, lvl="WARN", filename=os.path.basename(tile_path))
                    continue 
                
                data_tile_cxhxw = hdul[0].data.astype(np.float32)
                if data_tile_cxhxw.ndim == 3 and data_tile_cxhxw.shape[0] == n_channels:
                    current_tile_data_hwc = np.moveaxis(data_tile_cxhxw, 0, -1)
                elif data_tile_cxhxw.ndim == 2 and n_channels == 1:
                    current_tile_data_hwc = data_tile_cxhxw[..., np.newaxis]
                else:
                    pcb_asm("assemble_warn_tile_shape_mismatch_inc", prog=None, lvl="WARN", filename=os.path.basename(tile_path), shape=str(data_tile_cxhxw.shape), expected_channels=n_channels)
                    del data_tile_cxhxw; gc.collect(); continue
            del data_tile_cxhxw; gc.collect()

            data_to_use_for_reproject = current_tile_data_hwc
            wcs_to_use_for_reproject = mt_wcs_obj_original

            if apply_crop and crop_percent > 1e-3: # Appliquer si crop_percent significatif
                if ZEMOSAIC_UTILS_AVAILABLE and hasattr(zemosaic_utils, 'crop_image_and_wcs'):
                    pcb_asm(f"  ASM_INC: Rognage {crop_percent:.1f}% pour tuile {os.path.basename(tile_path)}", lvl="DEBUG_DETAIL")
                    cropped_data, cropped_wcs = zemosaic_utils.crop_image_and_wcs(
                        current_tile_data_hwc, mt_wcs_obj_original, crop_percent / 100.0, progress_callback
                    )
                    if cropped_data is not None and cropped_wcs is not None:
                        data_to_use_for_reproject = cropped_data
                        wcs_to_use_for_reproject = cropped_wcs
                        pcb_asm(f"    Nouvelle shape après rognage: {data_to_use_for_reproject.shape[:2]}", lvl="DEBUG_VERY_DETAIL")
                    else:
                        pcb_asm(f"  ASM_INC: AVERT - Rognage a échoué pour tuile {os.path.basename(tile_path)}. Utilisation tuile non rognée.", lvl="WARN")
                else:
                    pcb_asm(f"  ASM_INC: AVERT - Option rognage activée mais zemosaic_utils.crop_image_and_wcs non dispo.", lvl="WARN")
            
            if data_to_use_for_reproject is None or wcs_to_use_for_reproject is None: 
                pcb_asm(f"  ASM_INC: Données ou WCS pour reprojection sont None pour tuile {os.path.basename(tile_path)}, ignorée.", lvl="WARN")
                continue

            tile_footprint_combined_for_coverage = np.zeros(final_output_shape_hw, dtype=bool)
            for i_channel in range(n_channels):
                channel_data_to_reproject = None
                if data_to_use_for_reproject.ndim == 3 and data_to_use_for_reproject.shape[-1] > i_channel:
                    channel_data_to_reproject = data_to_use_for_reproject[..., i_channel]
                elif data_to_use_for_reproject.ndim == 2 and i_channel == 0:
                    channel_data_to_reproject = data_to_use_for_reproject
                
                if channel_data_to_reproject is None:
                    pcb_asm(f"  ASM_INC: Canal {i_channel} non trouvé pour reproj (tuile {tile_idx}), shape: {data_to_use_for_reproject.shape}", lvl="WARN"); continue

                input_for_reproject = (channel_data_to_reproject, wcs_to_use_for_reproject)
                reprojected_channel_data, footprint_channel = reproject_interp(input_for_reproject, final_output_wcs, shape_out=final_output_shape_hw, order='bilinear', parallel=False)
                valid_pixels_mask_footprint = footprint_channel > 0.01
                if np.any(valid_pixels_mask_footprint):
                    weights_for_channel = footprint_channel[valid_pixels_mask_footprint]; data_to_add = reprojected_channel_data[valid_pixels_mask_footprint]
                    running_sum_accumulator[..., i_channel][valid_pixels_mask_footprint] += data_to_add * weights_for_channel
                    running_norm_accumulator[..., i_channel][valid_pixels_mask_footprint] += weights_for_channel
                    if i_channel == 0: tile_footprint_combined_for_coverage |= valid_pixels_mask_footprint
                del reprojected_channel_data, footprint_channel, valid_pixels_mask_footprint
                if 'weights_for_channel' in locals(): del weights_for_channel
                if 'data_to_add' in locals(): del data_to_add
                gc.collect()
            
            if np.any(tile_footprint_combined_for_coverage):
                final_pixel_contributions[tile_footprint_combined_for_coverage] += 1.0
            tile_processed_successfully_this_iteration = True # Marquer comme succès pour cette itération

        except MemoryError as e_mem_tile:
             pcb_asm("assemble_error_memory_processing_tile_inc", prog=None, lvl="ERROR", filename=os.path.basename(tile_path), error=str(e_mem_tile)); logger.error(f"MemoryError traitement tuile {tile_path} (incrémental).", exc_info=True)
        except Exception as e_tile:
            pcb_asm("assemble_error_processing_tile_inc", prog=None, lvl="WARN", filename=os.path.basename(tile_path), error=str(e_tile)); logger.error(f"Erreur traitement tuile {tile_path} (incrémental).", exc_info=True)
        finally:
            # current_tile_data_hwc a été chargé du FITS
            if current_tile_data_hwc is not None:
                del current_tile_data_hwc
            
            # data_to_use_for_reproject peut être le même objet que current_tile_data_hwc (pas de rognage)
            # ou un nouvel objet (rognage appliqué). S'il est différent, il faut le supprimer aussi.
            if 'data_to_use_for_reproject' in locals() and \
               data_to_use_for_reproject is not None and \
               (locals().get('current_tile_data_hwc_exists_before_del', False) and \
                data_to_use_for_reproject is not current_tile_data_hwc): # current_tile_data_hwc n'existe plus si déjà supprimé
                 del data_to_use_for_reproject
            elif 'data_to_use_for_reproject' in locals() and data_to_use_for_reproject is not None and not locals().get('current_tile_data_hwc_exists_before_del', False):
                 # Si current_tile_data_hwc a été del mais data_to_use_for_reproject est toujours là (devrait être le même objet dans ce cas)
                 # cette condition est complexe, simplifions en s'assurant qu'ils sont initialisés à None.
                 pass # La suppression de current_tile_data_hwc (si non None) devrait suffire si pas de rognage.

            # Simplification du finally pour le nettoyage des données de tuile
            # Les variables sont initialisées à None au début de la boucle.
            # On s'assure juste de supprimer si elles ont été peuplées.
            # gc.collect() est appelé à la fin.

        # ... (log de progression) ...
        if tile_idx % 10 == 0 or tile_idx == num_master_tiles : 
            pcb_asm("assemble_progress_tiles_processed_inc", prog=None, lvl="INFO_DETAIL", num_done=tile_idx, total_num=num_master_tiles)

    pcb_asm("assemble_info_final_normalization_inc", prog=None, lvl="DEBUG_DETAIL")
    epsilon = 1e-9; final_mosaic_hwc = np.zeros_like(running_sum_accumulator, dtype=np.float32)
    for i_channel in range(n_channels):
        norm_channel_values = running_norm_accumulator[..., i_channel]; sum_channel_values = running_sum_accumulator[..., i_channel]
        valid_norm_mask = norm_channel_values > epsilon
        np.divide(sum_channel_values, norm_channel_values, out=final_mosaic_hwc[..., i_channel], where=valid_norm_mask)
    del running_sum_accumulator, running_norm_accumulator; gc.collect()
    pcb_asm("assemble_info_finished_incremental", prog=None, lvl="INFO", shape=str(final_mosaic_hwc.shape if final_mosaic_hwc is not None else "N/A"))
    return final_mosaic_hwc, final_pixel_contributions.astype(np.float32)



def assemble_final_mosaic_with_reproject_coadd(
    master_tile_fits_with_wcs_list: list,
    final_output_wcs: WCS, # Type hint pour WCS d'Astropy
    final_output_shape_hw: tuple,
    progress_callback: callable,
    n_channels: int = 3, 
    match_bg: bool = True,
    # --- NOUVEAUX PARAMÈTRES POUR LE ROGNAGE ---
    apply_crop: bool = False,
    crop_percent: float = 0.0 # Pourcentage par côté, 0.0 = pas de rognage par défaut
    # --- FIN NOUVEAUX PARAMÈTRES ---
):
    """
    Assemble les master tuiles en une mosaïque finale en utilisant reproject_and_coadd.
    Peut optionnellement rogner les master tuiles avant assemblage.
    """
    _pcb = lambda msg_key, prog=None, lvl="INFO_DETAIL", **kwargs: \
        _log_and_callback(msg_key, prog, lvl, callback=progress_callback, **kwargs)

    _log_memory_usage(progress_callback, "Début assemble_final_mosaic_with_reproject_coadd")
    _pcb(f"ASM_REPROJ_COADD: Options de rognage - Appliquer: {apply_crop}, Pourcentage: {crop_percent if apply_crop else 'N/A'}", lvl="DEBUG_DETAIL") # Log des options de rognage

    # ... (Vérification des dépendances REPROJECT_AVAILABLE, ASTROPY_AVAILABLE - inchangée) ...
    if not (REPROJECT_AVAILABLE and reproject_and_coadd and reproject_interp and ASTROPY_AVAILABLE and fits):
        missing_deps = []; # ...
        if not REPROJECT_AVAILABLE or not reproject_and_coadd or not reproject_interp: missing_deps.append("Reproject")
        if not ASTROPY_AVAILABLE or not fits : missing_deps.append("Astropy (fits)")
        _pcb("assemble_error_core_deps_unavailable_reproject_coadd", prog=None, lvl="ERROR", missing=", ".join(missing_deps)); return None, None

    num_master_tiles = len(master_tile_fits_with_wcs_list)
    _pcb("assemble_info_start_reproject_coadd", prog=None, lvl="INFO", num_tiles=num_master_tiles, match_bg=match_bg)
    if not master_tile_fits_with_wcs_list:
        _pcb("assemble_error_no_tiles_provided_reproject_coadd", prog=None, lvl="ERROR"); return None, None

    _pcb("assemble_info_reading_all_master_tiles_for_reproject_coadd", prog=None, lvl="DEBUG_DETAIL")
    
    # input_data_all_tiles_HWC va stocker des tuples (données_image_HWC, wcs_objet_correspondant)
    # Ces données et WCS seront potentiellement ceux des images rognées.
    input_data_all_tiles_HWC_processed = [] 
    
    for i_tile_load, (mt_path, mt_wcs_obj_original) in enumerate(master_tile_fits_with_wcs_list):
        try:
            _pcb(f"  ASM_REPROJ_COADD: Lecture et prétraitement (rognage si actif) Master Tile {i_tile_load+1}/{num_master_tiles} '{os.path.basename(mt_path)}'", prog=None, lvl="DEBUG_VERY_DETAIL")
            
            with fits.open(mt_path, memmap=False, do_not_scale_image_data=True) as hdul: # Garder do_not_scale
                if not hdul or hdul[0].data is None:
                    _pcb("assemble_warn_tile_empty_reproject_coadd", prog=None, lvl="WARN", filename=os.path.basename(mt_path))
                    continue
                
                # Charger les données brutes de la master tuile
                mt_data_cxhxw_adu = hdul[0].data.astype(np.float32)
                
                current_tile_data_hwc = None
                if mt_data_cxhxw_adu.ndim == 3 and mt_data_cxhxw_adu.shape[0] == n_channels:
                    current_tile_data_hwc = np.moveaxis(mt_data_cxhxw_adu, 0, -1)
                elif mt_data_cxhxw_adu.ndim == 2 and n_channels == 1:
                     current_tile_data_hwc = mt_data_cxhxw_adu[..., np.newaxis]
                else:
                    _pcb("assemble_warn_tile_shape_mismatch_reproject_coadd", prog=None, lvl="WARN", filename=os.path.basename(mt_path), shape=str(mt_data_cxhxw_adu.shape), expected_channels=n_channels)
                    continue
            
            # --- APPLICATION DU ROGNAGE SI ACTIVÉ ---
            data_to_use_for_assembly = current_tile_data_hwc
            wcs_to_use_for_assembly = mt_wcs_obj_original

            if apply_crop and crop_percent > 1e-3: # Appliquer si crop_percent > 0 (avec une petite tolérance)
                if ZEMOSAIC_UTILS_AVAILABLE and hasattr(zemosaic_utils, 'crop_image_and_wcs'):
                    _pcb(f"    ASM_REPROJ_COADD: Rognage {crop_percent:.1f}% pour tuile {os.path.basename(mt_path)}", lvl="DEBUG_DETAIL")
                    cropped_data, cropped_wcs = zemosaic_utils.crop_image_and_wcs(
                        current_tile_data_hwc, 
                        mt_wcs_obj_original, 
                        crop_percent / 100.0, # La fonction attend une fraction (0.0 à 1.0)
                        progress_callback=progress_callback
                    )
                    if cropped_data is not None and cropped_wcs is not None:
                        data_to_use_for_assembly = cropped_data
                        wcs_to_use_for_assembly = cropped_wcs
                        _pcb(f"      Nouvelle shape après rognage: {data_to_use_for_assembly.shape[:2]}", lvl="DEBUG_VERY_DETAIL")
                    else:
                        _pcb(f"    ASM_REPROJ_COADD: AVERT - Rognage a échoué pour tuile {os.path.basename(mt_path)}. Utilisation de la tuile non rognée.", lvl="WARN")
                else:
                    _pcb(f"    ASM_REPROJ_COADD: AVERT - Option de rognage activée mais zemosaic_utils.crop_image_and_wcs non disponible.", lvl="WARN")
            # --- FIN APPLICATION DU ROGNAGE ---

            input_data_all_tiles_HWC_processed.append((data_to_use_for_assembly, wcs_to_use_for_assembly))

        except MemoryError as e_mem_read: # ... (gestion MemoryError comme avant) ...
            _pcb("assemble_error_memory_reading_all_tiles", prog=None, lvl="ERROR", filename=os.path.basename(mt_path), error=str(e_mem_read)); logger.error(f"MemoryError lecture tuile {os.path.basename(mt_path)}:", exc_info=True); _log_memory_usage(progress_callback, f"MemoryError lecture tuile {i_tile_load+1}"); del input_data_all_tiles_HWC_processed; gc.collect(); return None, None 
        except Exception as e_read_mt: # ... (gestion autre Exception comme avant) ...
            _pcb("assemble_error_read_master_tile_reproject_coadd", prog=None, lvl="WARN", filename=os.path.basename(mt_path), error=str(e_read_mt)); logger.error(f"Erreur lecture tuile {os.path.basename(mt_path)}:", exc_info=True); continue
    
    if not input_data_all_tiles_HWC_processed: # Vérifier la liste traitée
        _pcb("assemble_error_no_valid_tiles_after_read_and_crop_reproject_coadd", prog=None, lvl="ERROR"); return None, None
    
    _log_memory_usage(progress_callback, "Phase 5 (reproject_coadd) - Après chargement/rognage de toutes les master tuiles")
    _pcb("assemble_info_all_tiles_loaded_and_processed_reproject_coadd", prog=None, lvl="DEBUG", num_loaded_tiles=len(input_data_all_tiles_HWC_processed))


    final_mosaic_stacked_channels_list = [] 
    final_mosaic_coverage_map = None 
    
    for i_channel in range(n_channels):
        # ... (log memory et info canal) ...
        _log_memory_usage(progress_callback, f"Phase 5 (reproject_coadd) - Début canal {i_channel+1}")
        _pcb("assemble_info_channel_processing_reproject_coadd", prog=None, lvl="INFO_DETAIL", channel_num=i_channel + 1, total_channels=n_channels)
        
        current_channel_input_data = []
        # UTILISER input_data_all_tiles_HWC_processed ICI
        for tile_data_hwc_processed, tile_wcs_processed in input_data_all_tiles_HWC_processed:
            try:
                # S'assurer que tile_data_hwc_processed a bien la dimension de canal
                if tile_data_hwc_processed.ndim == 3 and tile_data_hwc_processed.shape[-1] > i_channel :
                    channel_data_hw = tile_data_hwc_processed[..., i_channel].copy() # .copy() est important pour reproject
                    current_channel_input_data.append((channel_data_hw, tile_wcs_processed))
                elif tile_data_hwc_processed.ndim == 2 and i_channel == 0 : # Cas monochrome
                    current_channel_input_data.append((tile_data_hwc_processed.copy(), tile_wcs_processed))
                else:
                    _pcb("assemble_error_channel_index_reproject_coadd_processed", lvl="ERROR", tile_shape=str(tile_data_hwc_processed.shape), channel_idx=i_channel)
            except IndexError: # Devrait être attrapé par la condition de shape ci-dessus
                _pcb("assemble_error_channel_index_reproject_coadd_indexerror", lvl="ERROR", tile_shape=str(tile_data_hwc_processed.shape), channel_idx=i_channel)
                pass # Ne pas ajouter si le canal n'existe pas
        
        if not current_channel_input_data: # ... (gestion canal vide comme avant) ...
            _pcb("assemble_warn_no_data_for_channel_reproject_coadd", lvl="WARN", channel_num=i_channel+1); black_channel = np.zeros(final_output_shape_hw, dtype=np.float32); final_mosaic_stacked_channels_list.append(black_channel)
            if i_channel == 0 and final_mosaic_coverage_map is None: final_mosaic_coverage_map = np.zeros(final_output_shape_hw, dtype=np.float32)
            _log_memory_usage(progress_callback, f"Phase 5 (reproject_coadd) - Fin canal {i_channel+1} (données manquantes)"); continue

        try:
            _pcb(f"  Appel de reproject_and_coadd pour canal {i_channel+1} avec {len(current_channel_input_data)} images (potentiellement rognées). match_background={match_bg}", prog=None, lvl="DEBUG_DETAIL")
            
            stacked_channel_output, coverage_channel_output = reproject_and_coadd(
                current_channel_input_data, # Contient (données_rognées_canal, wcs_rogné)
                output_projection=final_output_wcs,
                shape_out=final_output_shape_hw, 
                reproject_function=reproject_interp, 
                combine_function='mean', 
                match_background=match_bg,
                # block_size=(512,512) # Optionnel: pour tester si ça aide avec la mémoire, peut ralentir
            )
            final_mosaic_stacked_channels_list.append(stacked_channel_output.astype(np.float32))
            if i_channel == 0: final_mosaic_coverage_map = coverage_channel_output.astype(np.float32)
            _pcb("assemble_info_channel_processed_reproject_coadd", prog=None, lvl="INFO_DETAIL", channel_num=i_channel + 1)
        
        except MemoryError as e_mem_reproject: # ... (gestion MemoryError comme avant) ...
            _pcb("assemble_error_memory_channel_reprojection_reproject_coadd", prog=None, lvl="ERROR", channel_num=i_channel + 1, error=str(e_mem_reproject)); logger.error(f"MemoryError reproject_and_coadd canal {i_channel + 1}:", exc_info=True); _log_memory_usage(progress_callback, f"MemoryError reproject_and_coadd canal {i_channel+1}"); del input_data_all_tiles_HWC_processed, current_channel_input_data, final_mosaic_stacked_channels_list, final_mosaic_coverage_map; gc.collect(); return None, None 
        except Exception as e_reproject_ch: # ... (gestion autre Exception comme avant) ...
            _pcb("assemble_error_channel_reprojection_failed_reproject_coadd", prog=None, lvl="ERROR", channel_num=i_channel + 1, error=str(e_reproject_ch)); logger.error(f"Erreur reproject_and_coadd canal {i_channel+1}:", exc_info=True); del input_data_all_tiles_HWC_processed, current_channel_input_data, final_mosaic_stacked_channels_list, final_mosaic_coverage_map; gc.collect(); return None, None
        finally:
            _log_memory_usage(progress_callback, f"Phase 5 (reproject_coadd) - Fin canal {i_channel+1} (avant del current_channel_input_data)")
            del current_channel_input_data; gc.collect()
            _log_memory_usage(progress_callback, f"Phase 5 (reproject_coadd) - Fin canal {i_channel+1} (après del current_channel_input_data)")

    _log_memory_usage(progress_callback, "Phase 5 (reproject_coadd) - Après traitement de tous les canaux")
    del input_data_all_tiles_HWC_processed # Supprimer la liste des données chargées
    gc.collect()
    _log_memory_usage(progress_callback, "Phase 5 (reproject_coadd) - Après del input_data_all_tiles_HWC_processed")

    # ... (Fin de la fonction : gestion des canaux manquants, stack final, return - inchangé) ...
    if len(final_mosaic_stacked_channels_list) != n_channels: # ...
        _pcb("assemble_error_stacking_failed_missing_channels_reproject_coadd", prog=None, lvl="ERROR", num_expected=n_channels, num_actual=len(final_mosaic_stacked_channels_list))
        while len(final_mosaic_stacked_channels_list) < n_channels: _pcb(f"assemble_warn_padding_missing_channel_reproject_coadd", lvl="WARN", channel_num_padded=len(final_mosaic_stacked_channels_list)+1); final_mosaic_stacked_channels_list.append(np.zeros(final_output_shape_hw, dtype=np.float32))
        if final_mosaic_coverage_map is None and n_channels > 0 : final_mosaic_coverage_map = np.zeros(final_output_shape_hw, dtype=np.float32)
    try: final_mosaic_data_HWC = np.stack(final_mosaic_stacked_channels_list, axis=-1)
    except ValueError as e_stack_final: _pcb("assemble_error_final_channel_stack_failed_reproject_coadd", prog=None, lvl="ERROR", error=str(e_stack_final)); logger.error(f"Erreur stack final canaux. Shapes: {[ch.shape for ch in final_mosaic_stacked_channels_list if hasattr(ch, 'shape')]}", exc_info=True); return None, None
    finally: del final_mosaic_stacked_channels_list; gc.collect()
    _log_memory_usage(progress_callback, "Fin assemble_final_mosaic_with_reproject_coadd")
    _pcb("assemble_info_finished_reproject_coadd", prog=None, lvl="INFO", shape=final_mosaic_data_HWC.shape if final_mosaic_data_HWC is not None else "N/A")
    return final_mosaic_data_HWC, final_mosaic_coverage_map




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
    save_final_as_uint16_config: bool

):
    """
    Orchestre le traitement de la mosaïque hiérarchique.
    """
    pcb = lambda msg_key, prog=None, lvl="INFO", **kwargs: _log_and_callback(msg_key, prog, lvl, callback=progress_callback, **kwargs)
    
    def update_gui_eta(eta_seconds_total):
        if progress_callback and callable(progress_callback):
            eta_str = "--:--:--"
            if eta_seconds_total is not None and eta_seconds_total >= 0:
                h, rem = divmod(int(eta_seconds_total), 3600); m, s = divmod(rem, 60)
                eta_str = f"{h:02d}:{m:02d}:{s:02d}"
            pcb(f"ETA_UPDATE:{eta_str}", prog=None, lvl="ETA_LEVEL") 

    SEESTAR_STACK_CLUSTERING_THRESHOLD_DEG = 0.08
    PROGRESS_WEIGHT_PHASE1_RAW_SCAN = 30; PROGRESS_WEIGHT_PHASE2_CLUSTERING = 5
    PROGRESS_WEIGHT_PHASE3_MASTER_TILES = 35; PROGRESS_WEIGHT_PHASE4_GRID_CALC = 5
    PROGRESS_WEIGHT_PHASE5_ASSEMBLY = 15; PROGRESS_WEIGHT_PHASE6_SAVE = 8
    PROGRESS_WEIGHT_PHASE7_CLEANUP = 2
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
    actual_num_workers_ph1 = max(1, min(effective_base_workers, num_total_raw_files if num_total_raw_files > 0 else 1))
    pcb(f"WORKERS_PHASE1: Utilisation de {actual_num_workers_ph1} worker(s). (Base: {effective_base_workers}, Fichiers: {num_total_raw_files})", prog=None, lvl="INFO") # Log mis à jour pour plus de clarté
    
    start_time_phase1 = time.monotonic()
    all_raw_files_processed_info_dict = {} # Pour stocker les infos des fichiers traités avec succès
    files_processed_count_ph1 = 0      # Compteur pour les fichiers soumis au ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=actual_num_workers_ph1, thread_name_prefix="ZeMosaic_Ph1_") as executor_ph1:
        future_to_filepath_ph1 = { 
            executor_ph1.submit(
                get_wcs_and_pretreat_raw_file, 
                f_path, 
                astap_exe_path, 
                astap_data_dir_param, 
                astap_search_radius_config, 
                astap_downsample_config, 
                astap_sensitivity_config, 
                180, # astap_timeout_seconds
                progress_callback
            ): f_path for f_path in fits_file_paths 
        }
        
        for future in as_completed(future_to_filepath_ph1):
            file_path_original = future_to_filepath_ph1[future]
            files_processed_count_ph1 += 1 # Incrémenter pour chaque future terminée
            
            prog_step_phase1 = base_progress_phase1 + int(PROGRESS_WEIGHT_PHASE1_RAW_SCAN * (files_processed_count_ph1 / max(1, num_total_raw_files)))
            
            try:
                # Récupérer le résultat de la tâche
                img_data_adu, wcs_obj_solved, header_obj_updated = future.result()
                
                # Si la tâche a réussi (ne retourne pas que des None)
                if img_data_adu is not None and wcs_obj_solved is not None and header_obj_updated is not None:
                    # Sauvegarder les données prétraitées en .npy
                    cache_file_basename = f"preprocessed_{os.path.splitext(os.path.basename(file_path_original))[0]}_{files_processed_count_ph1}.npy"
                    cached_image_path = os.path.join(temp_image_cache_dir, cache_file_basename)
                    try:
                        np.save(cached_image_path, img_data_adu)
                        # Stocker les informations pour les phases suivantes
                        all_raw_files_processed_info_dict[file_path_original] = {
                            'path_raw': file_path_original, 
                            'path_preprocessed_cache': cached_image_path, 
                            'wcs': wcs_obj_solved, 
                            'header': header_obj_updated 
                        }
                        # pcb(f"Phase 1: Fichier '{os.path.basename(file_path_original)}' traité et mis en cache.", prog=prog_step_phase1, lvl="DEBUG_VERY_DETAIL") # Optionnel
                    except Exception as e_save_npy:
                        pcb("run_error_phase1_save_npy_failed", prog=prog_step_phase1, lvl="ERROR", filename=os.path.basename(file_path_original), error=str(e_save_npy))
                        logger.error(f"Erreur sauvegarde NPY pour {file_path_original}:", exc_info=True)
                    finally: 
                        # Libérer la mémoire des données image dès que possible
                        del img_data_adu; gc.collect() 
                else: 
                    # Le fichier a échoué (ex: WCS non résolu et déplacé)
                    # get_wcs_and_pretreat_raw_file a déjà loggué l'échec spécifique.
                    pcb("run_warn_phase1_wcs_pretreat_failed_or_skipped_thread", prog=prog_step_phase1, lvl="WARN", filename=os.path.basename(file_path_original))
                    # S'assurer que img_data_adu est bien None si le retour était None,None,None pour éviter del sur None
                    if img_data_adu is not None: del img_data_adu; gc.collect()

            except Exception as exc_thread: 
                # Erreur imprévue dans la future elle-même
                pcb("run_error_phase1_thread_exception", prog=prog_step_phase1, lvl="ERROR", filename=os.path.basename(file_path_original), error=str(exc_thread))
                logger.error(f"Exception non gérée dans le thread Phase 1 pour {file_path_original}:", exc_info=True)
            
            # Log de mémoire et ETA
            if files_processed_count_ph1 % max(1, num_total_raw_files // 10) == 0 or files_processed_count_ph1 == num_total_raw_files: 
                _log_memory_usage(progress_callback, f"Phase 1 - Traité {files_processed_count_ph1}/{num_total_raw_files}")
            
            elapsed_phase1 = time.monotonic() - start_time_phase1
            if files_processed_count_ph1 > 0 : # Eviter division par zéro si aucun fichier traité (ne devrait pas arriver ici)
                time_per_raw_file_wcs = elapsed_phase1 / files_processed_count_ph1
                eta_phase1_sec = (num_total_raw_files - files_processed_count_ph1) * time_per_raw_file_wcs
                current_progress_in_run_percent = base_progress_phase1 + (files_processed_count_ph1 / max(1, num_total_raw_files)) * PROGRESS_WEIGHT_PHASE1_RAW_SCAN
                time_per_percent_point_global = (time.monotonic() - start_time_total_run) / max(1, current_progress_in_run_percent) if current_progress_in_run_percent > 0 else (time.monotonic() - start_time_total_run)
                total_eta_sec = eta_phase1_sec + (100 - current_progress_in_run_percent) * time_per_percent_point_global
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
    if not seestar_stack_groups: pcb("run_error_phase2_no_groups", prog=(base_progress_phase2 + PROGRESS_WEIGHT_PHASE2_CLUSTERING), lvl="ERROR"); return
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
    
    # Calcul des workers pour la Phase 3 (déjà fait, en utilisant num_seestar_stacks_to_process de la Phase 2)
    reduction_ph3 = 4 # Ta valeur
    actual_num_workers_ph3_candidate = max(1, effective_base_workers - reduction_ph3)
    actual_num_workers_ph3 = max(1, min(actual_num_workers_ph3_candidate, num_seestar_stacks_to_process if num_seestar_stacks_to_process > 0 else 1))
    pcb(f"WORKERS_PHASE3: Utilisation de {actual_num_workers_ph3} worker(s). (Base: {effective_base_workers}, Réduc Candidat (-{reduction_ph3}): {actual_num_workers_ph3_candidate}, Groupes: {num_seestar_stacks_to_process})", prog=None, lvl="INFO") # Log mis à jour pour clarté

    tiles_processed_count_ph3 = 0
    # Envoyer l'info initiale avant la boucle
    if num_seestar_stacks_to_process > 0:
        pcb(f"MASTER_TILE_COUNT_UPDATE:{tiles_processed_count_ph3}/{num_seestar_stacks_to_process}", prog=None, lvl="ETA_LEVEL")
    
    with ThreadPoolExecutor(max_workers=actual_num_workers_ph3, thread_name_prefix="ZeMosaic_Ph3_") as executor_ph3:
        future_to_group_index = { 
            executor_ph3.submit(
                create_master_tile,
                sg_info_list, 
                i_stk, # tile_id
                temp_master_tile_storage_dir,
                stack_norm_method, stack_weight_method, stack_reject_algo,
                stack_kappa_low, stack_kappa_high, parsed_winsor_limits,
                stack_final_combine,
                apply_radial_weight_config, radial_feather_fraction_config,
                radial_shape_power_config, min_radial_weight_floor_config, 
                astap_exe_path, astap_data_dir_param, astap_search_radius_config, 
                astap_downsample_config, astap_sensitivity_config, 180, # timeout ASTAP         
                progress_callback
            ): i_stk for i_stk, sg_info_list in enumerate(seestar_stack_groups) 
        }
        for future in as_completed(future_to_group_index):
            group_index_original = future_to_group_index[future]
            tiles_processed_count_ph3 += 1
            
            # --- ENVOYER LA MISE À JOUR DU COMPTEUR DE TUILES ---
            pcb(f"MASTER_TILE_COUNT_UPDATE:{tiles_processed_count_ph3}/{num_seestar_stacks_to_process}", prog=None, lvl="ETA_LEVEL")
            # --- FIN ENVOI MISE À JOUR ---
            
            prog_step_phase3 = base_progress_phase3 + int(PROGRESS_WEIGHT_PHASE3_MASTER_TILES * (tiles_processed_count_ph3 / max(1, num_seestar_stacks_to_process)))
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
            
    master_tiles_results_list = [master_tiles_results_list_temp[i] for i in sorted(master_tiles_results_list_temp.keys())]
    del master_tiles_results_list_temp; gc.collect() 
    if not master_tiles_results_list: 
        pcb("run_error_phase3_no_master_tiles_created", prog=(base_progress_phase3 + PROGRESS_WEIGHT_PHASE3_MASTER_TILES), lvl="ERROR"); return
    
    current_global_progress = base_progress_phase3 + PROGRESS_WEIGHT_PHASE3_MASTER_TILES
    _log_memory_usage(progress_callback, "Fin Phase 3"); 
    pcb("run_info_phase3_finished_from_cache", prog=current_global_progress, lvl="INFO", num_master_tiles=len(master_tiles_results_list))
    
    # Assurer que le compteur final est bien affiché (au cas où la dernière itération n'aurait pas été exactement le total)
    # Bien que la logique dans la boucle devrait déjà le faire. Peut être redondant mais ne fait pas de mal.
    pcb(f"MASTER_TILE_COUNT_UPDATE:{tiles_processed_count_ph3}/{num_seestar_stacks_to_process}", prog=None, lvl="ETA_LEVEL")




    
    
    # --- Phase 4 (Calcul Grille Finale) ---
    base_progress_phase4 = current_global_progress
    _log_memory_usage(progress_callback, "Début Phase 4 (Calcul Grille)")
    pcb("run_info_phase4_started", prog=base_progress_phase4, lvl="INFO")
    wcs_list_for_final_grid = []; shapes_list_for_final_grid_hw = []
    for mt_path_iter,mt_wcs_iter in master_tiles_results_list:
        # ... (logique de récupération shape, inchangée) ...
        if not (mt_path_iter and os.path.exists(mt_path_iter) and mt_wcs_iter and mt_wcs_iter.is_celestial): pcb("run_warn_phase4_invalid_master_tile_for_grid", prog=None, lvl="WARN", path=os.path.basename(mt_path_iter if mt_path_iter else "N/A_path")); continue
        try:
            h_mt_loc,w_mt_loc=0,0
            if mt_wcs_iter.pixel_shape and mt_wcs_iter.pixel_shape[0] > 0 and mt_wcs_iter.pixel_shape[1] > 0 : h_mt_loc,w_mt_loc=mt_wcs_iter.pixel_shape[1],mt_wcs_iter.pixel_shape[0] 
            else: 
                with fits.open(mt_path_iter,memmap=True, do_not_scale_image_data=True) as hdul_mt_s:
                    if hdul_mt_s[0].data is None: pcb("run_warn_phase4_no_data_in_tile_fits", prog=None, lvl="WARN", path=os.path.basename(mt_path_iter)); continue
                    data_shape = hdul_mt_s[0].shape 
                    if len(data_shape) == 3: h_mt_loc,w_mt_loc = data_shape[1],data_shape[2]
                    elif len(data_shape) == 2: h_mt_loc,w_mt_loc = data_shape[0],data_shape[1]
                    else: pcb("run_warn_phase4_unhandled_tile_shape", prog=None, lvl="WARN", path=os.path.basename(mt_path_iter), shape=data_shape); continue 
                    if mt_wcs_iter and mt_wcs_iter.is_celestial and mt_wcs_iter.pixel_shape is None:
                        try: mt_wcs_iter.pixel_shape=(w_mt_loc,h_mt_loc)
                        except Exception as e_set_ps: pcb("run_warn_phase4_failed_set_pixel_shape", prog=None, lvl="WARN", path=os.path.basename(mt_path_iter), error=str(e_set_ps))
            if h_mt_loc > 0 and w_mt_loc > 0: shapes_list_for_final_grid_hw.append((int(h_mt_loc),int(w_mt_loc))); wcs_list_for_final_grid.append(mt_wcs_iter)
            else: pcb("run_warn_phase4_zero_dimensions_tile", prog=None, lvl="WARN", path=os.path.basename(mt_path_iter))
        except Exception as e_read_tile_shape: pcb("run_error_phase4_reading_tile_shape", prog=None, lvl="ERROR", path=os.path.basename(mt_path_iter), error=str(e_read_tile_shape)); logger.error(f"Erreur lecture shape tuile {os.path.basename(mt_path_iter)}:", exc_info=True); continue
    if not wcs_list_for_final_grid or not shapes_list_for_final_grid_hw or len(wcs_list_for_final_grid) != len(shapes_list_for_final_grid_hw): pcb("run_error_phase4_insufficient_tile_info", prog=(base_progress_phase4 + PROGRESS_WEIGHT_PHASE4_GRID_CALC), lvl="ERROR"); return
    final_mosaic_drizzle_scale = 1.0 
    final_output_wcs, final_output_shape_hw = _calculate_final_mosaic_grid(wcs_list_for_final_grid, shapes_list_for_final_grid_hw, final_mosaic_drizzle_scale, progress_callback)
    if not final_output_wcs or not final_output_shape_hw: pcb("run_error_phase4_grid_calc_failed", prog=(base_progress_phase4 + PROGRESS_WEIGHT_PHASE4_GRID_CALC), lvl="ERROR"); return
    current_global_progress = base_progress_phase4 + PROGRESS_WEIGHT_PHASE4_GRID_CALC
    _log_memory_usage(progress_callback, "Fin Phase 4"); pcb("run_info_phase4_finished", prog=current_global_progress, lvl="INFO", shape=final_output_shape_hw, crval=final_output_wcs.wcs.crval if final_output_wcs.wcs else 'N/A')

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
    reproject_coadd_available = ('assemble_final_mosaic_with_reproject_coadd' in globals() and callable(assemble_final_mosaic_with_reproject_coadd))
    incremental_available = ('assemble_final_mosaic_incremental' in globals() and callable(assemble_final_mosaic_incremental))

    if USE_INCREMENTAL_ASSEMBLY:
        if not incremental_available: 
            pcb("run_error_phase5_inc_func_missing", prog=None, lvl="CRITICAL"); return
        pcb("run_info_phase5_started_incremental", prog=base_progress_phase5, lvl="INFO")
        final_mosaic_data_HWC, final_mosaic_coverage_HW = assemble_final_mosaic_incremental(
            master_tile_fits_with_wcs_list=valid_master_tiles_for_assembly, 
            final_output_wcs=final_output_wcs, 
            final_output_shape_hw=final_output_shape_hw,
            progress_callback=progress_callback,
            n_channels=3,
            # --- PASSAGE DES PARAMÈTRES DE ROGNAGE ---
            apply_crop=apply_master_tile_crop_config,
            crop_percent=master_tile_crop_percent_config
            # --- FIN PASSAGE ---
        )
        log_key_phase5_failed = "run_error_phase5_assembly_failed_incremental"
        log_key_phase5_finished = "run_info_phase5_finished_incremental"
    else: # Méthode Reproject & Coadd
        if not reproject_coadd_available: 
            pcb("run_error_phase5_reproject_coadd_func_missing", prog=None, lvl="CRITICAL"); return
        pcb("run_info_phase5_started_reproject_coadd", prog=base_progress_phase5, lvl="INFO")
        final_mosaic_data_HWC, final_mosaic_coverage_HW = assemble_final_mosaic_with_reproject_coadd(
            master_tile_fits_with_wcs_list=valid_master_tiles_for_assembly, 
            final_output_wcs=final_output_wcs, 
            final_output_shape_hw=final_output_shape_hw,
            progress_callback=progress_callback,
            n_channels=3, 
            match_bg=True,
            # --- PASSAGE DES PARAMÈTRES DE ROGNAGE ---
            apply_crop=apply_master_tile_crop_config,
            crop_percent=master_tile_crop_percent_config
            # --- FIN PASSAGE ---
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
                preview_asinh_a = 0.1 # Facteur 'a' pour le stretch asinh après la normalisation initiale
                                      # Pour un stretch plus "doux" similaire à ASIFitsView, 'a' peut être plus grand.
                                      # ASIFitsView utilise souvent un 'midtones balance' (gamma-like) aussi.
                                      # Un 'a' de 10 comme dans ton code de test est très doux. Essayons 0.5 ou 1.0.
                preview_asinh_a = 1.0 # Test avec une valeur plus douce pour le 'a' de asinh

                m_stretched = zemosaic_utils.stretch_auto_asifits_like(
                    final_mosaic_data_HWC,
                    p_low=preview_p_low, 
                    p_high=preview_p_high,
                    asinh_a_factor=preview_asinh_a, # Renommé pour correspondre à une signature possible
                    # ou simplement asinh_a=preview_asinh_a si la fonction s'appelle ainsi
                    apply_wb=True # Supposons que tu veuilles la balance des blancs auto
                )

                if m_stretched is not None:
                    img_u8 = (np.clip(m_stretched.astype(np.float32), 0, 1) * 255).astype(np.uint8)
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

####################################################################################################################################################################


