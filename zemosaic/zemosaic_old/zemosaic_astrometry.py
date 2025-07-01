# zemosaic_astrometry.py

import os
import numpy as np
import warnings
import time
# import tempfile # Plus utilisé directement si on nettoie manuellement
import traceback
import subprocess
# import shutil # Plus utilisé directement si on nettoie manuellement
import gc
import logging
import psutil
from concurrent.futures import ProcessPoolExecutor

import multiprocessing


logger = logging.getLogger("ZeMosaicAstrometry")
# ... (pas besoin de reconfigurer le logger ici s'il hérite du worker)

try:
    from astropy.io import fits
    from astropy.wcs import WCS as AstropyWCS, FITSFixedWarning 
    from astropy.utils.exceptions import AstropyWarning
    from astropy import units as u # Nécessaire pour _update_fits_header_with_wcs_za
    ASTROPY_AVAILABLE_ASTROMETRY = True
    warnings.filterwarnings('ignore', category=FITSFixedWarning)
    warnings.filterwarnings('ignore', category=AstropyWarning)
except ImportError:
    logger.error("Astropy non installée. Certaines fonctionnalités de zemosaic_astrometry seront limitées.")
    ASTROPY_AVAILABLE_ASTROMETRY = False
    class AstropyWCS: pass 
    class FITSFixedWarning(Warning): pass
    u = None


def _log_memory_usage(progress_callback: callable, context_message: str = ""):
    """Logue l'utilisation mémoire du processus courant."""
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
            f"Sys RAM: Avail {available_ram_mb:.0f}MB / Total {total_ram_mb:.0f}MB ({percent_ram_used}% used). "
            f"Sys Swap: Used {used_swap_mb:.0f}MB / Total {total_swap_mb:.0f}MB ({percent_swap_used}% used)."
        )
        progress_callback(log_msg, None, "DEBUG")
    except Exception as e_mem_log:
        progress_callback(f"Erreur lors du logging mémoire ({context_message}): {e_mem_log}", None, "WARN")


def _run_astap_subprocess(cmd_list: list, cwd: str, timeout_sec: int):
    """Fonction exécutée dans un ProcessPoolExecutor pour lancer ASTAP."""
    return subprocess.run(
        cmd_list,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout_sec,
        check=False,
        cwd=cwd,
    )


def _calculate_pixel_scale_from_header(header: fits.Header, progress_callback: callable = None) -> float | None:
    # ... (corps de la fonction inchangé, il semble correct)
    if not header:
        return None
    focal_len_mm = None
    pixel_size_um = None
    focal_keys = ['FOCALLEN', 'FOCAL', 'FLENGTH']
    for key in focal_keys:
        if key in header and isinstance(header[key], (int, float)) and header[key] > 0:
            focal_len_mm = float(header[key])
            if progress_callback: progress_callback(f"  ASTAP ScaleCalc: Trouvé {key}={focal_len_mm} mm", None, "DEBUG_DETAIL")
            break
    if focal_len_mm is None:
        if progress_callback: progress_callback("  ASTAP ScaleCalc: FOCALLEN non trouvée ou invalide dans le header.", None, "DEBUG_DETAIL")
        return None
    pix_size_keys = ['XPIXSZ', 'PIXSIZE', 'PIXELSIZE', 'PIXSCAL1', 'SCALE']
    for key in pix_size_keys:
        if key in header and isinstance(header[key], (int, float)) and header[key] > 0:
            if key.upper() == 'PIXSCAL1':
                unit_key = f"CUNIT{key[-1]}" if key[-1].isdigit() else None
                if unit_key and unit_key in header and str(header[unit_key]).lower() in ['arcsec', 'asec', '"']:
                    if progress_callback: progress_callback(f"  ASTAP ScaleCalc: Trouvé {key}={header[key]} arcsec/pix directement.", None, "DEBUG_DETAIL")
                    return float(header[key])
            pixel_size_um = float(header[key])
            if progress_callback: progress_callback(f"  ASTAP ScaleCalc: Trouvé {key}={pixel_size_um} µm", None, "DEBUG_DETAIL")
            break
    if pixel_size_um is None:
        if progress_callback: progress_callback("  ASTAP ScaleCalc: XPIXSZ (ou équivalent) non trouvé ou invalide.", None, "DEBUG_DETAIL")
        return None
    try:
        calculated_scale_arcsec_pix = (pixel_size_um / focal_len_mm) * 206.264806
        if progress_callback: progress_callback(f"  ASTAP ScaleCalc: Échelle calculée: {calculated_scale_arcsec_pix:.3f} arcsec/pix", None, "INFO_DETAIL")
        return calculated_scale_arcsec_pix
    except ZeroDivisionError:
        if progress_callback: progress_callback("  ASTAP ScaleCalc ERREUR: Division par zéro (FOCALLEN nulle ?).", None, "WARN")
        return None

def _parse_wcs_file_content_za(wcs_file_path, image_shape_hw, progress_callback=None):
    # ... (corps de la fonction inchangé, il semble correct)
    filename_log = os.path.basename(wcs_file_path)
    if progress_callback: progress_callback(f"  ASTAP WCS Parse: Tentative parsing '{filename_log}' pour shape {image_shape_hw}", None, "DEBUG_DETAIL")
    if not (os.path.exists(wcs_file_path) and os.path.getsize(wcs_file_path) > 0):
        if progress_callback: progress_callback(f"    ASTAP WCS Parse ERREUR: Fichier WCS '{filename_log}' non trouvé ou vide.", None, "WARN")
        return None
    if not ASTROPY_AVAILABLE_ASTROMETRY:
        if progress_callback: progress_callback("    ASTAP WCS Parse ERREUR: Astropy non disponible pour parser WCS.", None, "ERROR")
        return None
    try:
        with open(wcs_file_path, 'r', errors='replace') as f: wcs_text = f.read()
        wcs_hdr_from_text = fits.Header.fromstring(wcs_text.replace('\r\n', '\n').replace('\r', '\n'), sep='\n')
        if 'NAXIS1' not in wcs_hdr_from_text and image_shape_hw:
            wcs_hdr_from_text['NAXIS1'] = image_shape_hw[1]
        if 'NAXIS2' not in wcs_hdr_from_text and image_shape_hw:
            wcs_hdr_from_text['NAXIS2'] = image_shape_hw[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            wcs_obj = AstropyWCS(wcs_hdr_from_text, naxis=2, relax=True)
        if wcs_obj and wcs_obj.is_celestial:
            if image_shape_hw and image_shape_hw[0] > 0 and image_shape_hw[1] > 0:
                try:
                    wcs_obj.pixel_shape = (image_shape_hw[1], image_shape_hw[0])
                except Exception as e_ps_parse:
                    if progress_callback: progress_callback(f"    ASTAP WCS Parse AVERT: Échec set pixel_shape sur WCS parsé: {e_ps_parse}", None, "WARN")
            if progress_callback: progress_callback(f"    ASTAP WCS Parse: Objet WCS parsé avec succès depuis '{filename_log}'.", None, "DEBUG_DETAIL")
            return wcs_obj
        else:
            if progress_callback: progress_callback(f"    ASTAP WCS Parse ERREUR: Échec création WCS valide/céleste depuis '{filename_log}'.", None, "WARN")
            return None
    except Exception as e:
        if progress_callback: progress_callback(f"    ASTAP WCS Parse ERREUR: Exception lors du parsing WCS '{filename_log}': {e}", None, "ERROR")
        logger.error(f"Erreur parsing WCS '{wcs_file_path}': {e}", exc_info=True)
        return None


def _update_fits_header_with_wcs_za(fits_header_to_update: fits.Header, 
                                   wcs_object_solution: AstropyWCS, 
                                   solver_name="ASTAP_ZeMosaic", 
                                   progress_callback=None):
    if not (fits_header_to_update is not None and wcs_object_solution and wcs_object_solution.is_celestial):
        if progress_callback: progress_callback("  ASTAP HeaderUpdate: MàJ header annulée: header/WCS invalide.", None, "WARN")
        return False 
    if progress_callback: progress_callback(f"  ASTAP HeaderUpdate: MàJ header FITS avec solution WCS de {solver_name}...", None, "DEBUG_DETAIL")
    if not ASTROPY_AVAILABLE_ASTROMETRY:
        if progress_callback: progress_callback("  ASTAP HeaderUpdate ERREUR: Astropy non disponible pour MàJ header.", None, "ERROR")
        return False
    try:
        wcs_keys_to_remove = [
            'WCSAXES', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 
            'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
            'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
            'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
            'CDELT1', 'CDELT2', 'CROTA1', 'CROTA2', 
            'LONPOLE', 'LATPOLE', 'EQUINOX', 'RADESYS',
            'PV1_0', 'PV1_1', 'PV1_2', 'PV2_0', 'PV2_1', 'PV2_2' 
        ]
        for key_del in wcs_keys_to_remove:
            if key_del in fits_header_to_update:
                try:
                    del fits_header_to_update[key_del]
                except KeyError:
                    pass
        
        # Correction de la coquille ici :
        new_wcs_header_cards = wcs_object_solution.to_header(relax=True) # Utiliser relax=True est plus simple et robuste
        
        fits_header_to_update.update(new_wcs_header_cards)
        fits_header_to_update[f'{solver_name.upper()}_SOLVED'] = (True, f'{solver_name} solution')
        
        if u is not None: # S'assurer que astropy.units est importé
            try:
                if hasattr(wcs_object_solution, 'proj_plane_pixel_scales') and callable(wcs_object_solution.proj_plane_pixel_scales):
                    scales_deg = wcs_object_solution.proj_plane_pixel_scales()
                    pixscale_arcsec = np.mean(np.abs(scales_deg.to_value(u.arcsec)))
                    fits_header_to_update[f'{solver_name.upper()}_PSCALE'] = (float(f"{pixscale_arcsec:.4f}"), f'[asec/pix] Scale from {solver_name} WCS')
            except Exception:
                pass
            
        if progress_callback: progress_callback("  ASTAP HeaderUpdate: Header FITS MàJ avec WCS.", None, "DEBUG_DETAIL")
        return True
    except Exception as e_upd:
        if progress_callback: progress_callback(f"  ASTAP HeaderUpdate ERREUR: {e_upd}", None, "ERROR")
        logger.error(f"Erreur MàJ header FITS avec WCS: {e_upd}", exc_info=True) # Log le traceback complet
        return False



# DANS zemosaic_astrometry.py

def solve_with_astap(image_fits_path: str,
                     original_fits_header: fits.Header,
                     astap_exe_path: str,
                     astap_data_dir: str,
                     search_radius_deg: float | None = None,    # Depuis GUI
                     downsample_factor: int | None = None,      # Depuis GUI (pour -z)
                     sensitivity: int | None = None,            # Depuis GUI (pour -sens)
                     timeout_sec: int = 120,
                     update_original_header_in_place: bool = False,
                     progress_callback: callable = None):

    if not ASTROPY_AVAILABLE_ASTROMETRY:
        if progress_callback: progress_callback("ASTAP Solve ERREUR: Astropy non disponible, ASTAP solve annulé.", None, "ERROR")
        return None

    img_basename_log = os.path.basename(image_fits_path)
    if progress_callback: progress_callback(f"ASTAP Solve: Début pour '{img_basename_log}'", None, "INFO_DETAIL")
    logger.debug(f"ASTAP Solve params (entrée fonction): image='{img_basename_log}', radius={search_radius_deg}, "
                 f"downsample={downsample_factor}, sensitivity={sensitivity}")

    if not (astap_exe_path and os.path.isfile(astap_exe_path)):
        if progress_callback: progress_callback(f"ASTAP Solve ERREUR: Chemin ASTAP exe invalide: '{astap_exe_path}'.", None, "ERROR")
        return None
    if not (astap_data_dir and os.path.isdir(astap_data_dir)):
        if progress_callback: progress_callback(f"ASTAP Solve AVERT: Chemin ASTAP data non spécifié ou invalide: '{astap_data_dir}'. ASTAP pourrait ne pas trouver ses bases.", None, "WARN")
    if not (image_fits_path and os.path.isfile(image_fits_path)):
        if progress_callback: progress_callback(f"ASTAP Solve ERREUR: Chemin image FITS invalide: '{image_fits_path}'.", None, "ERROR")
        return None
    if original_fits_header is None: # Should not happen if called from worker
        if progress_callback: progress_callback("ASTAP Solve ERREUR: Header FITS original non fourni.", None, "ERROR")
        return None

    current_image_dir = os.path.dirname(image_fits_path)
    base_image_name_no_ext = os.path.splitext(os.path.basename(image_fits_path))[0]
    expected_wcs_file_path = os.path.join(current_image_dir, base_image_name_no_ext + ".wcs")
    expected_ini_file_path = os.path.join(current_image_dir, base_image_name_no_ext + ".ini")
    astap_log_file_path = os.path.join(current_image_dir, base_image_name_no_ext + ".log")
    files_to_cleanup_by_astap = [expected_wcs_file_path, expected_ini_file_path]

    for f_to_clean in files_to_cleanup_by_astap:
        if os.path.exists(f_to_clean):
            try: os.remove(f_to_clean)
            except Exception as e_del_pre:
                if progress_callback: progress_callback(f"  ASTAP Solve AVERT: Échec nettoyage pré-ASTAP '{os.path.basename(f_to_clean)}': {e_del_pre}", None, "WARN")
    if os.path.exists(astap_log_file_path):
        try: os.remove(astap_log_file_path)
        except Exception as e_del_log_pre:
            if progress_callback: progress_callback(f"  ASTAP Solve AVERT: Échec nettoyage pré-ASTAP log '{os.path.basename(astap_log_file_path)}': {e_del_log_pre}", None, "WARN")

    cmd_list_astap = [astap_exe_path, "-f", image_fits_path, "-log"]
    if astap_data_dir and os.path.isdir(astap_data_dir):
         cmd_list_astap.extend(["-d", astap_data_dir])

    # --- MODIFICATION: Option -pxscale / -fov 0 temporairement enlevée pour test ---
    # calculated_px_scale = _calculate_pixel_scale_from_header(original_fits_header, progress_callback)
    # if calculated_px_scale and calculated_px_scale > 0.01 and calculated_px_scale < 50.0:
    #     cmd_list_astap.extend(["-pxscale", f"{calculated_px_scale:.4f}"])
    #     if progress_callback: progress_callback(f"  ASTAP Solve: Utilisation -pxscale {calculated_px_scale:.4f} arcsec/pix (calculé du header).", None, "DEBUG")
    # else:
    #     cmd_list_astap.extend(["-fov", "0"])
    #     if progress_callback: progress_callback("  ASTAP Solve: -pxscale non calculable/utilisé. Utilisation -fov 0.", None, "DEBUG")
    if progress_callback: progress_callback("  ASTAP Solve: Option -pxscale / -fov 0 non ajoutée explicitement (test). ASTAP estimera.", None, "DEBUG")


    # Gestion du downsampling (-z)
    if downsample_factor is not None and isinstance(downsample_factor, int) and downsample_factor >= 0:
        cmd_list_astap.extend(["-z", str(downsample_factor)])
        if progress_callback: progress_callback(f"  ASTAP Solve: Utilisation -z {downsample_factor} (configuré).", None, "DEBUG")
    else:
        if progress_callback: progress_callback(f"  ASTAP Solve: Downsample non spécifié ou invalide ({downsample_factor}). ASTAP utilisera son défaut pour -z.", None, "DEBUG_DETAIL")

    # Gestion de la sensibilité (-sens)
    if sensitivity is not None and isinstance(sensitivity, int): # Typiquement positif pour -sens 100
        cmd_list_astap.extend(["-sens", str(sensitivity)])
        if progress_callback: progress_callback(f"  ASTAP Solve: Utilisation -sens {sensitivity} (configuré).", None, "DEBUG")
    else:
        if progress_callback: progress_callback(f"  ASTAP Solve: Sensibilité non spécifiée ou invalide ({sensitivity}). ASTAP utilisera son défaut pour -sens.", None, "DEBUG_DETAIL")

    # --- MODIFICATION: Gestion des hints RA/Dec et du rayon de recherche ---
    # On n'ajoute plus -ra / -spd explicitement pour ce test, mais on ajoute -r si configuré
    if search_radius_deg is not None and search_radius_deg > 0:
        cmd_list_astap.extend(["-r", str(search_radius_deg)])
        if progress_callback: progress_callback(f"  ASTAP Solve: Utilisation rayon de recherche -r {search_radius_deg}° (pas de hints RA/Dec explicites).", None, "DEBUG")
    else:
        # Si aucun rayon n'est spécifié, et qu'on n'utilise pas -fov 0 explicitement, ASTAP a son propre comportement.
        # Si l'objectif est de forcer ASTAP à utiliser son mécanisme "blind" sans rayon, on ne met rien.
        # Si on voulait forcer une recherche "full-sky" mais limitée par le fov, on aurait besoin de -fov 0 et -r (grand).
        # Pour l'instant, on laisse ASTAP décider si -r n'est pas fourni.
        if progress_callback: progress_callback(f"  ASTAP Solve: Pas de rayon de recherche explicite spécifié via -r. ASTAP utilisera ses valeurs par défaut pour la zone de recherche.", None, "DEBUG_DETAIL")


    if progress_callback: progress_callback(f"  ASTAP Solve: Commande: {' '.join(cmd_list_astap)}", None, "DEBUG")
    logger.info(f"Executing ASTAP for {img_basename_log}: {' '.join(cmd_list_astap)}")
    wcs_solved_obj = None
    astap_success = False

    try:

        astap_process_result = None
        try:
            if not multiprocessing.current_process().daemon:
                with ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_run_astap_subprocess, cmd_list_astap, current_image_dir, timeout_sec)
                    astap_process_result = future.result()
            else:
                raise RuntimeError("daemon process")
        except (AssertionError, RuntimeError) as e_pool:
            if progress_callback:
                progress_callback(f"  ASTAP Solve: ProcessPoolExecutor indisponible ({e_pool}). Lancement direct.", None, "DEBUG_DETAIL")
            astap_process_result = _run_astap_subprocess(cmd_list_astap, current_image_dir, timeout_sec)

        logger.debug(f"ASTAP return code: {astap_process_result.returncode}")

        rc_astap = astap_process_result.returncode
        del astap_process_result
        gc.collect()
        _log_memory_usage(progress_callback, "Après GC post-ASTAP")

        if rc_astap == 0:
            if os.path.exists(expected_wcs_file_path) and os.path.getsize(expected_wcs_file_path) > 0:
                if progress_callback: progress_callback(f"  ASTAP Solve: Résolution OK (code 0). Fichier WCS '{os.path.basename(expected_wcs_file_path)}' trouvé.", None, "INFO_DETAIL")
                img_height = original_fits_header.get('NAXIS2', 0)
                img_width = original_fits_header.get('NAXIS1', 0)
                if img_height == 0 or img_width == 0:
                    try:
                        with fits.open(image_fits_path) as hdul_shape:
                            shape_from_file = hdul_shape[0].shape
                            if len(shape_from_file) >=2 :
                                img_height = shape_from_file[-2]
                                img_width = shape_from_file[-1]
                    except Exception as e_shape_read:
                         if progress_callback: progress_callback(f"  ASTAP Solve AVERT: Impossible de lire NAXIS1/2 du header ou du fichier FITS: {e_shape_read}. WCS parsing pourrait échouer.", None, "WARN")
                if img_height > 0 and img_width > 0:
                    wcs_solved_obj = _parse_wcs_file_content_za(expected_wcs_file_path, (img_height, img_width), progress_callback)
                else:
                    if progress_callback: progress_callback(f"  ASTAP Solve ERREUR: Dimensions image (NAXIS1/2) non trouvées pour '{img_basename_log}'. WCS non parsé.", None, "ERROR")
                if wcs_solved_obj and wcs_solved_obj.is_celestial:
                    astap_success = True
                    if progress_callback: progress_callback(f"  ASTAP Solve: Objet WCS créé et céleste pour '{img_basename_log}'.", None, "INFO")
                    if update_original_header_in_place and original_fits_header is not None:
                        if _update_fits_header_with_wcs_za(original_fits_header, wcs_solved_obj, progress_callback=progress_callback):
                             if progress_callback: progress_callback(f"  ASTAP Solve: Header FITS original mis à jour avec WCS pour '{img_basename_log}'.", None, "DEBUG_DETAIL")
                        else:
                             if progress_callback: progress_callback(f"  ASTAP Solve AVERT: Échec MàJ header FITS original avec WCS pour '{img_basename_log}'.", None, "WARN")
                else:
                    if progress_callback: progress_callback(f"  ASTAP Solve ERREUR: WCS parsé non valide ou non céleste pour '{img_basename_log}'.", None, "ERROR")
                    wcs_solved_obj = None
            else:
                if progress_callback: progress_callback(f"  ASTAP Solve ERREUR: Code 0 mais fichier .wcs manquant/vide ('{os.path.basename(expected_wcs_file_path)}').", None, "ERROR")
        else:
            error_msg = f"ASTAP Solve Échec (code {rc_astap}) pour '{img_basename_log}'."
            if rc_astap == 1: error_msg += " (No solution found)."
            elif rc_astap == 2: error_msg += " (ASTAP FITS read error - vérifiez format/corruption)."
            elif rc_astap == 10: error_msg += " (ASTAP database not found - vérifiez -d)."
            if progress_callback: progress_callback(f"  {error_msg}", None, "WARN")
            logger.warning(error_msg)

    except subprocess.TimeoutExpired:
        if progress_callback: progress_callback(f"ASTAP Solve ERREUR: Timeout ({timeout_sec}s) pour '{img_basename_log}'.", None, "ERROR")
        logger.error(f"ASTAP command timed out for {img_basename_log}", exc_info=False)
    except FileNotFoundError:
        if progress_callback: progress_callback(f"ASTAP Solve ERREUR: Exécutable ASTAP '{astap_exe_path}' non trouvé.", None, "ERROR")
        logger.error(f"ASTAP executable not found at '{astap_exe_path}'.", exc_info=False)
    except Exception as e_astap_glob:
        if progress_callback: progress_callback(f"ASTAP Solve ERREUR Inattendue: {e_astap_glob}", None, "ERROR")
        logger.error(f"Unexpected error during ASTAP execution for {img_basename_log}: {e_astap_glob}", exc_info=True)
    finally:
        if progress_callback: progress_callback(f"  ASTAP Solve: Nettoyage post-exécution (sauf log si échec) pour '{img_basename_log}'...", None, "DEBUG_DETAIL")
        for f_clean_post in files_to_cleanup_by_astap: # .wcs, .ini
            if os.path.exists(f_clean_post):
                try:
                    if f_clean_post == expected_wcs_file_path and astap_success and not update_original_header_in_place:
                        if progress_callback: progress_callback(f"    ASTAP Clean: Conservation du .wcs: {os.path.basename(f_clean_post)} (succès, pas de MàJ header en place)", None, "DEBUG_DETAIL")
                        continue
                    os.remove(f_clean_post)
                    if progress_callback: progress_callback(f"    ASTAP Clean: Fichier '{os.path.basename(f_clean_post)}' supprimé.", None, "DEBUG_DETAIL")
                except Exception as e_del_post:
                    if progress_callback: progress_callback(f"    ASTAP Clean AVERT: Échec nettoyage '{os.path.basename(f_clean_post)}': {e_del_post}", None, "WARN")

        if astap_success and os.path.exists(astap_log_file_path):
            try:
                os.remove(astap_log_file_path)
                if progress_callback: progress_callback(f"    ASTAP Clean: Fichier log ASTAP '{os.path.basename(astap_log_file_path)}' supprimé (succès).", None, "DEBUG_DETAIL")
            except Exception as e_del_log_succ:
                if progress_callback: progress_callback(f"    ASTAP Clean AVERT: Échec nettoyage log ASTAP (succès) '{os.path.basename(astap_log_file_path)}': {e_del_log_succ}", None, "WARN")
        elif not astap_success and os.path.exists(astap_log_file_path):
             if progress_callback: progress_callback(f"    ASTAP Clean: CONSERVATION du log ASTAP '{os.path.basename(astap_log_file_path)}' (échec solve).", None, "INFO_DETAIL")

        gc.collect()

    if wcs_solved_obj:
        if progress_callback: progress_callback(f"ASTAP Solve: WCS trouvé pour {img_basename_log}.", None, "INFO_DETAIL")
    else:
        if progress_callback: progress_callback(f"ASTAP Solve: Pas de WCS final pour {img_basename_log}.", None, "WARN")
    return wcs_solved_obj



