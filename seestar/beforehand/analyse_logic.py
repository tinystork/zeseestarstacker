# -----------------------------------------------------------------------------
# Auteur       : TRISTAN NAULEAU 
# Date         : 2025-07-12
# Licence      : GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
#
# Ce travail est distribué librement en accord avec les termes de la
# GNU GPL v3 (https://www.gnu.org/licenses/gpl-3.0.html).
# Vous êtes libre de redistribuer et de modifier ce code, à condition
# de conserver cette notice et de mentionner que je suis l’auteur
# de tout ou partie du code si vous le réutilisez.
# -----------------------------------------------------------------------------
# Author       : TRISTAN NAULEAU
# Date         : 2025-07-12
# License      : GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
#
# This work is freely distributed under the terms of the
# GNU GPL v3 (https://www.gnu.org/licenses/gpl-3.0.html).
# You are free to redistribute and modify this code, provided that
# you keep this notice and mention that I am the author
# of all or part of the code if you reuse it.
# -----------------------------------------------------------------------------
# (Imports and write_log_summary remain the same)
import os
import shutil
import time
import datetime
import traceback
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import warnings
import json
import concurrent.futures
import threading
import zipfile
import xml.etree.ElementTree as ET
import csv

from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

import bortle_utils


def artif_ratio_to_sqm(ratio_artif, mag_naturel=21.6):
    """Convert a ratio artificial/natural sky brightness to SQM."""
    return mag_naturel - 2.5 * np.log10(1 + ratio_artif)

NATURAL_SKY = 174.0  # µcd/m² ≈ 22 mag/arcsec²


def _load_bortle_raster(path):
    """Load a Bortle raster dataset, supporting KMZ ground overlays."""
    if path.lower().endswith('.kmz'):
        with zipfile.ZipFile(path, 'r') as zf:
            kml_name = next((n for n in zf.namelist() if n.lower().endswith('.kml')), None)
            if not kml_name:
                raise ValueError('KMZ missing KML file')
            kml_bytes = zf.read(kml_name)
            root = ET.fromstring(kml_bytes)
            ns = ''
            if root.tag.startswith('{'):
                ns = root.tag[: root.tag.index('}') + 1]
            go = root.find('.//' + ns + 'GroundOverlay')
            if go is None:
                raise ValueError('KMZ missing GroundOverlay')
            href_el = go.find('.//' + ns + 'href')
            if href_el is None or not href_el.text:
                raise ValueError('KMZ GroundOverlay missing href')
            image_name = href_el.text.strip()
            if image_name not in zf.namelist():
                raise ValueError('Referenced image not found in KMZ')
            llbox = go.find('.//' + ns + 'LatLonBox')
            if llbox is None:
                raise ValueError('KMZ GroundOverlay missing LatLonBox')

            def _read_val(tag):
                el = llbox.find(ns + tag)
                if el is None or el.text is None:
                    raise ValueError(f'LatLonBox missing {tag}')
                return float(el.text)

            north = _read_val('north')
            south = _read_val('south')
            east = _read_val('east')
            west = _read_val('west')

            img_bytes = zf.read(image_name)

        with MemoryFile(img_bytes) as mem_src:
            with mem_src.open() as src:
                data = src.read()
                height = src.height
                width = src.width
                count = src.count
                dtype = src.dtypes[0]

        transform = from_bounds(west, south, east, north, width, height)
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': count,
            'dtype': dtype,
            'crs': 'EPSG:4326',
            'transform': transform,
        }
        mem_out = MemoryFile()
        ds = mem_out.open(**profile)
        ds.write(data)
        return ds
    else:
        return bortle_utils.load_bortle_raster(path)

try:
    import starcount_module
except ImportError:
    print("AVERTISSEMENT (analyse_logic): starcount_module.py introuvable. Le comptage d'etoiles sera désactivé.")
    starcount_module = None

try:
    import ecc_module
except ImportError:
    print("AVERTISSEMENT (analyse_logic): ecc_module.py introuvable. FWHM/Ecc ne seront pas calculés.")
    ecc_module = None

try:
    import snr_module
except ImportError:
    print("ERREUR CRITIQUE (analyse_logic): snr_module.py introuvable.")
    raise ImportError("Module SNR manquant.")

TRAIL_MODULE_LOADED = False
SATDET_AVAILABLE = False
SATDET_USES_SEARCHPATTERN = False
SATDET_ACCEPTS_LIST = False

try:
    import trail_module
    TRAIL_MODULE_LOADED = True
    SATDET_AVAILABLE = getattr(trail_module, 'SATDET_AVAILABLE', False)
    SATDET_USES_SEARCHPATTERN = getattr(trail_module, 'SATDET_USES_SEARCHPATTERN', False)
    SATDET_ACCEPTS_LIST = getattr(trail_module, 'SATDET_ACCEPTS_LIST', False)
    print(f"INFO (analyse_logic): trail_module chargé. SATDET_AVAILABLE={SATDET_AVAILABLE}, SATDET_ACCEPTS_LIST={SATDET_ACCEPTS_LIST}")
except ImportError:
    print("AVERTISSEMENT (analyse_logic): trail_module.py introuvable. La détection de traînées sera désactivée.")
except Exception as e:
    print(f"ERREUR (analyse_logic): Erreur lors de l'import ou de la lecture de trail_module: {e}")


def sanitize_for_json(obj):
    """Convertit récursivement les objets pour compatibilité JSON."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(elem) for elem in obj]
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.int32, np.int64, np.int_)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj



def write_log_summary(log_file_path, input_dir, options,
                      analysis_config=None,
                      sat_errors=None, results_list=None,
                      selection_stats=None,
                      skipped_marker_dirs_count=0):
    """Écrit le résumé ET LES DONNÉES DE VISUALISATION dans le fichier log."""
    try:
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write("\n" + "="*80 + "\n")
            log_file.write("Résumé de l'analyse:\n")
            log_file.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Dossier analysé: {input_dir}\n")
            log_file.write(f"Inclure sous-dossiers: {'Oui' if options.get('include_subfolders') else 'Non'}\n")
            log_file.write(f"Dossiers ignorés (marqueur trouvé): {skipped_marker_dirs_count}\n")
            log_file.write(f"Analyse SNR effectuée: {'Oui' if options.get('analyze_snr') else 'Non'}\n")
            log_file.write(f"Détection Traînées effectuée: {'Oui' if options.get('detect_trails') and SATDET_AVAILABLE else 'Non'}\n")

            # Log des paramètres de détection si effectuée
            if options.get('detect_trails') and SATDET_AVAILABLE and analysis_config:
                 log_file.write("Paramètres détection traînées (Hough):\n")
                 log_file.write(f"  sigma={analysis_config.get('sigma', 'N/A')}, low_thresh={analysis_config.get('low_thresh', 'N/A')}, h_thresh={analysis_config.get('h_thresh', 'N/A')}\n")
                 log_file.write(f"  line_len={analysis_config.get('line_len', 'N/A')}, small_edge={analysis_config.get('small_edge', 'N/A')}, line_gap={analysis_config.get('line_gap', 'N/A')}\n")

            # Log des erreurs satdet si présentes
            if sat_errors:
                log_file.write("Erreurs spécifiques reportées par la détection de traînées:\n"); count = 0
                global_errors = {k: v for k, v in sat_errors.items() if not isinstance(k, tuple) or len(k) != 2}
                file_errors = {k: v for k, v in sat_errors.items() if isinstance(k, tuple) and len(k) == 2}
                for key, msg in global_errors.items(): log_file.write(f"  - ERREUR GLOBALE ({key}): {msg}\n"); count += 1
                for (fname, ext), msg in file_errors.items():
                    try: rel_path = os.path.relpath(fname, input_dir)
                    except ValueError: rel_path = fname
                    if "is not a valid science extension" not in str(msg): log_file.write(f"  - {rel_path} (ext {ext}): {msg}\n"); count += 1
                if count == 0: log_file.write("  (Aucune erreur pertinente)\n")

            # Log des critères de sélection SNR si analyse SNR effectuée
            if options.get('analyze_snr') and selection_stats:
                log_file.write("Critères de sélection SNR appliqués:\n"); mode = selection_stats.get('mode'); value = selection_stats.get('value'); threshold = selection_stats.get('threshold')
                log_file.write(f"  Mode: {mode or 'N/A'}\n")
                if mode == 'percent':
                     log_file.write(f"  Pourcentage à garder: {value}%\n")
                     if threshold is not None: log_file.write(f"  Seuil SNR correspondant: {threshold:.2f}\n")
                     else: log_file.write("  Seuil SNR correspondant: N/A\n")
                elif mode == 'threshold': log_file.write(f"  Seuil SNR minimum: {value}\n")
                log_file.write(f"  Images initialement éligibles (OK): {selection_stats.get('initial_count', 'N/A')}\n")
                log_file.write(f"  Images sélectionnées/conservées par SNR (avant filtre trail): {selection_stats.get('selected_count', 'N/A')}\n")
                log_file.write(f"  Images rejetées pour faible SNR (avant filtre trail): {selection_stats.get('rejected_count', 'N/A')}\n")

            # Log des stats globales et actions (uses FINAL state from results_list)
            if results_list is None: 
                log_file.write("Aucune analyse individuelle de fichier effectuée pour ce résumé.\n")
            else:
                total_processed = len(results_list); analyzed_count = sum(1 for r in results_list if r.get('status') != 'pending'); errors_count = sum(1 for r in results_list if r.get('status') == 'error')
                rejected_trails = sum(1 for r in results_list if r.get('rejected_reason') == 'trail'); rejected_low_snr = sum(1 for r in results_list if r.get('rejected_reason') == 'low_snr'); kept_count = sum(1 for r in results_list if r.get('status') == 'ok' and r.get('rejected_reason') is None)
                moved_trails = sum(1 for r in results_list if r.get('action') == 'moved_trail'); moved_low_snr = sum(1 for r in results_list if r.get('action') == 'moved_snr'); deleted_trails = sum(1 for r in results_list if r.get('action') == 'deleted_trail'); deleted_low_snr = sum(1 for r in results_list if r.get('action') == 'deleted_snr')
                log_file.write(f"Nombre total de fichiers FITS trouvés (hors dossiers ignorés): {total_processed}\n"); log_file.write(f"  Fichiers analysés (ou tentative): {analyzed_count}\n"); log_file.write(f"  Images conservées dans le dossier source/sous-dossiers: {kept_count}\n"); log_file.write(f"  Images marquées pour rejet (traînées): {rejected_trails}\n"); log_file.write(f"  Images marquées pour rejet (faible SNR): {rejected_low_snr}\n"); log_file.write(f"  Erreurs d'analyse fichier: {errors_count}\n")
                snr_reject_path_base = options.get('snr_reject_dir','N/A'); trail_reject_path_base = options.get('trail_reject_dir','N/A')
                if options.get('move_rejected'): 
                    log_file.write(f"Actions (Déplacement activé):\n")
                    if os.path.basename(trail_reject_path_base): log_file.write(f"  Déplacées vers '{os.path.basename(trail_reject_path_base)}' (traînées): {moved_trails}\n")
                    if os.path.basename(snr_reject_path_base): log_file.write(f"  Déplacées vers '{os.path.basename(snr_reject_path_base)}' (faible SNR): {moved_low_snr}\n")
                elif options.get('delete_rejected'): 
                    log_file.write(f"Actions (Suppression activée):\n")
                    log_file.write(f"  Supprimées (traînées): {deleted_trails}\n")
                    log_file.write(f"  Supprimées (faible SNR): {deleted_low_snr}\n")
                else: 
                    log_file.write(f"Actions: Aucune (fichiers rejetés non déplacés/supprimés)\n")
                
                if options.get('analyze_snr'):
                     all_valid_snrs = [r['snr'] for r in results_list if r.get('status') == 'ok' and 'snr' in r and r['snr'] is not None and np.isfinite(r['snr'])]
                     if all_valid_snrs:
                         log_file.write(f"Statistiques SNR (sur {len(all_valid_snrs)} images valides):\n")
                         mean_snr = np.mean(all_valid_snrs); median_snr = np.median(all_valid_snrs)
                         min_snr = min(all_valid_snrs); max_snr = max(all_valid_snrs)
                         log_file.write(f"  Moy: {mean_snr:.2f}, Med: {median_snr:.2f}, Min: {min_snr:.2f}, Max: {max_snr:.2f}\n")
                     else:
                         log_file.write("Statistiques SNR: Aucune donnée SNR valide calculée.\n")

                starcounts = [r['starcount'] for r in results_list if r.get('status') == 'ok' and r.get('starcount') is not None]
                if starcounts:
                    log_file.write(f"Statistiques Starcount (sur {len(starcounts)} images valides):\n")
                    mean_sc = np.mean(starcounts); median_sc = np.median(starcounts)
                    min_sc = min(starcounts); max_sc = max(starcounts)
                    log_file.write(f"  Moy: {mean_sc:.2f}, Med: {median_sc:.2f}, Min: {min_sc:.2f}, Max: {max_sc:.2f}\n")
                else:
                    log_file.write("Statistiques Starcount: Aucune donnée valide.\n")
            
            if results_list is not None:
                log_file.write("\n--- BEGIN VISUALIZATION DATA ---\n")
                json.dump(sanitize_for_json(results_list), log_file, indent=4)
                log_file.write("\n--- END VISUALIZATION DATA ---\n")

    except Exception as e:
        print(f"ERREUR CRITIQUE lors de l'écriture du résumé du log ({log_file_path}): {e}"); traceback.print_exc()
        # Essayer d'écrire l'erreur dans le log lui-même si la section principale a échoué
        try:
            with open(log_file_path, 'a', encoding='utf-8') as log_file_err: # 'a' pour ne pas écraser
                 log_file_err.write(f"\nERREUR CRITIQUE lors de l'écriture de ce résumé: {e}\n{traceback.format_exc()}");
        except Exception: 
            pass # Si même ça échoue, on ne peut plus rien faire ici



# --- DANS analyse_logic.py ---
# (Placez cette fonction au même niveau que perform_analysis, par exemple après)

def apply_pending_snr_actions(results_list, snr_reject_abs_path,
                              delete_rejected_flag, move_rejected_flag,
                              log_callback, status_callback, progress_callback,
                              input_dir_abs): # Ajout input_dir_abs pour rel_path
    """
    Applique les actions de déplacement ou suppression pour les fichiers marqués 
    avec 'low_snr_pending_action'.
    Modifie results_list en place.
    Retourne le nombre d'actions effectuées.
    """
    actions_count = 0
    if not results_list:
        return actions_count

    # S'assurer que les callbacks sont utilisables
    _log = log_callback if callable(log_callback) else lambda k, **kw: print(f"LOGIC_APPLY_SNR_LOG: {k} {kw}")
    _status = status_callback if callable(status_callback) else lambda k, **kw: print(f"LOGIC_APPLY_SNR_STATUS: {k} {kw}")
    _progress = progress_callback if callable(progress_callback) else lambda v: print(f"LOGIC_APPLY_SNR_PROGRESS: {v}")


    files_to_process_action = [r for r in results_list if r.get('rejected_reason') == 'low_snr_pending_action' and r.get('status') == 'ok']
    total_pending_files = len(files_to_process_action)
    
    if total_pending_files == 0:
        _log("logic_info_prefix", text="Aucune action SNR en attente à appliquer.")
        return 0

    _status("status_custom", text=f"Application des actions SNR différées sur {total_pending_files} fichiers...")
    _progress(0) # Démarrer la progression pour cette action

    for i, r in enumerate(files_to_process_action):
        # Mettre à jour la progression spécifique à cette tâche
        current_progress = ((i + 1) / total_pending_files) * 100
        _progress(current_progress)
        
        # Obtenir rel_path pour les logs et statuts
        try:
            rel_path = os.path.relpath(r.get('path'), input_dir_abs) if r.get('path') and input_dir_abs else r.get('file', 'Fichier inconnu')
        except ValueError: # Peut arriver si les chemins sont sur des lecteurs différents
            rel_path = r.get('file', 'Fichier inconnu')

        _status("status_custom", text=f"Action SNR sur {rel_path} ({i+1}/{total_pending_files})")

        current_path = r.get('path')
        if not current_path or not os.path.exists(current_path):
            _log("logic_move_skipped", file=rel_path, e="Fichier source non trouvé pour action SNR différée.")
            r['action_comment'] += " Source non trouvée pour action différée."
            r['action'] = 'error_action_deferred'
            r['status'] = 'error' # Marquer comme erreur si le fichier a disparu
            continue

        action_taken_this_file = False
        original_rejected_reason = r['rejected_reason'] # Sauvegarder au cas où

        if delete_rejected_flag:
            try:
                os.remove(current_path)
                _log("logic_info_prefix", text=f"Fichier supprimé (SNR différé): {rel_path}")
                r['path'] = None # Marquer le chemin comme nul
                r['action'] = 'deleted_snr'
                r['rejected_reason'] = 'low_snr' # Finaliser la raison
                r['status'] = 'processed_action' # ou un statut plus spécifique
                actions_count += 1
                action_taken_this_file = True
            except Exception as del_e:
                _log("logic_error_prefix", text=f"Erreur suppression SNR différé {rel_path}: {del_e}")
                r['action_comment'] += f" Erreur suppression différée: {del_e}"
                r['action'] = 'error_delete'
                r['rejected_reason'] = original_rejected_reason # Restaurer
        
        elif move_rejected_flag and snr_reject_abs_path:
            if not os.path.isdir(snr_reject_abs_path):
                try:
                    os.makedirs(snr_reject_abs_path)
                    _log("logic_dir_created", path=snr_reject_abs_path)
                except OSError as e_mkdir:
                    _log("logic_dir_create_error", path=snr_reject_abs_path, e=e_mkdir)
                    r['action_comment'] += f" Dossier rejet SNR inaccessible: {e_mkdir}"
                    r['action'] = 'error_move'
                    r['rejected_reason'] = original_rejected_reason
                    continue 

            dest_path = os.path.join(snr_reject_abs_path, os.path.basename(current_path))
            try:
                if os.path.normpath(current_path) != os.path.normpath(dest_path):
                    shutil.move(current_path, dest_path)
                    _log("logic_moved_info", folder=os.path.basename(snr_reject_abs_path), 
                         text_key_suffix="_deferred_snr", # Pour un message log plus spécifique
                         file_rel_path=rel_path)
                    r['path'] = dest_path # Mettre à jour le chemin
                    r['action'] = 'moved_snr'
                    r['rejected_reason'] = 'low_snr' # Finaliser la raison
                    r['status'] = 'processed_action'
                    actions_count += 1
                    action_taken_this_file = True
                else:
                    r['action_comment'] += " Déjà dans dossier cible (différé)?"
                    r['action'] = 'kept' 
                    r['rejected_reason'] = 'low_snr' 
                    r['status'] = 'processed_action' # Consideré comme actionné car déjà à destination
                    action_taken_this_file = True # On le compte comme une "action"
            except Exception as move_e:
                _log("logic_move_error", file=rel_path, e=move_e)
                r['action_comment'] += f" Erreur déplacement différé: {move_e}"
                r['action'] = 'error_move'
                r['rejected_reason'] = original_rejected_reason
        
        if not action_taken_this_file and not delete_rejected_flag and not move_rejected_flag:
            # Si aucune action de déplacement/suppression n'était configurée, 
            # on finalise juste le statut.
            r['action'] = 'kept_pending_snr_no_action' # Statut spécifique
            r['rejected_reason'] = 'low_snr' # Finaliser la raison
            r['status'] = 'processed_action'
            r['action_comment'] += " Action SNR différée mais ni suppression ni déplacement activé."
            # On ne compte pas cela comme une "action" de déplacement/suppression.
    
    _progress(100) # Fin de la progression pour cette tâche
    _status("status_custom", text=f"{actions_count} actions SNR différées appliquées.")
    _log("logic_info_prefix", text=f"{actions_count} actions SNR différées appliquées.")
    return actions_count


def apply_pending_trail_actions(results_list, trail_reject_abs_path,
                                delete_rejected_flag, move_rejected_flag,
                                log_callback, status_callback, progress_callback,
                                input_dir_abs):
    """Apply deferred actions for trail-rejected files."""
    actions_count = 0
    if not results_list:
        return actions_count

    _log = log_callback if callable(log_callback) else lambda k, **kw: None
    _status = status_callback if callable(status_callback) else lambda k, **kw: None
    _progress = progress_callback if callable(progress_callback) else lambda v: None

    to_process = [r for r in results_list if r.get('rejected_reason') == 'trail_pending_action' and r.get('status') == 'ok']
    total = len(to_process)
    if total == 0:
        _log('logic_info_prefix', text='Aucune action Traînées en attente.')
        return 0

    _status('status_custom', text=f'Application des actions Traînées différées sur {total} fichiers...')
    _progress(0)

    for i, r in enumerate(to_process):
        _progress(((i + 1) / total) * 100)
        try:
            rel_path = os.path.relpath(r.get('path'), input_dir_abs) if r.get('path') and input_dir_abs else r.get('file', 'N/A')
        except ValueError:
            rel_path = r.get('file', 'N/A')

        _status('status_custom', text=f'Action Traînées sur {rel_path} ({i+1}/{total})')

        current_path = r.get('path')
        if not current_path or not os.path.exists(current_path):
            _log('logic_move_skipped', file=rel_path, e='Fichier source introuvable pour action Traînées différée.')
            r['action_comment'] = r.get('action_comment', '') + ' Source non trouvée pour action différée.'
            r['action'] = 'error_action_deferred'
            r['status'] = 'error'
            continue

        action_done = False
        original_reason = r['rejected_reason']

        if delete_rejected_flag:
            try:
                os.remove(current_path)
                _log('logic_info_prefix', text=f'Fichier supprimé (Traînées différé): {rel_path}')
                r['path'] = None
                r['action'] = 'deleted_trail'
                r['rejected_reason'] = 'trail'
                r['status'] = 'processed_action'
                actions_count += 1
                action_done = True
            except Exception as del_e:
                _log('logic_error_prefix', text=f'Erreur suppression Traînées différé {rel_path}: {del_e}')
                r['action_comment'] = r.get('action_comment', '') + f' Erreur suppression différée: {del_e}'
                r['action'] = 'error_delete'
                r['rejected_reason'] = original_reason
        elif move_rejected_flag and trail_reject_abs_path:
            if not os.path.isdir(trail_reject_abs_path):
                try:
                    os.makedirs(trail_reject_abs_path)
                    _log('logic_dir_created', path=trail_reject_abs_path)
                except OSError as e_mkdir:
                    _log('logic_dir_create_error', path=trail_reject_abs_path, e=e_mkdir)
                    r['action_comment'] = r.get('action_comment', '') + f' Dossier rejet Traînées inaccessible: {e_mkdir}'
                    r['action'] = 'error_move'
                    r['rejected_reason'] = original_reason
                    continue

            dest_path = os.path.join(trail_reject_abs_path, os.path.basename(current_path))
            try:
                if os.path.normpath(current_path) != os.path.normpath(dest_path):
                    shutil.move(current_path, dest_path)
                    _log('logic_moved_info', folder=os.path.basename(trail_reject_abs_path), text_key_suffix='_deferred_trail', file_rel_path=rel_path)
                    r['path'] = dest_path
                    r['action'] = 'moved_trail'
                    r['rejected_reason'] = 'trail'
                    r['status'] = 'processed_action'
                    actions_count += 1
                    action_done = True
                else:
                    r['action_comment'] = r.get('action_comment', '') + ' Déjà dans dossier cible (différé)?'
                    r['action'] = 'kept'
                    r['rejected_reason'] = 'trail'
                    r['status'] = 'processed_action'
                    action_done = True
            except Exception as move_e:
                _log('logic_move_error', file=rel_path, e=move_e)
                r['action_comment'] = r.get('action_comment', '') + f' Erreur déplacement différé: {move_e}'
                r['action'] = 'error_move'
                r['rejected_reason'] = original_reason

        if not action_done and not delete_rejected_flag and not move_rejected_flag:
            r['action'] = 'kept_pending_trail_no_action'
            r['rejected_reason'] = 'trail'
            r['status'] = 'processed_action'
            r['action_comment'] = r.get('action_comment', '') + ' Action Traînées différée mais aucune opération configurée.'

    _progress(100)
    _status('status_custom', text=f'{actions_count} actions Traînées différées appliquées.')
    _log('logic_info_prefix', text=f'{actions_count} actions Traînées différées appliquées.')
    return actions_count


def apply_pending_reco_actions(results_list, reject_abs_path,
                               delete_rejected_flag, move_rejected_flag,
                               log_callback, status_callback, progress_callback,
                               input_dir_abs):
    """Apply actions for images not in recommendations."""
    actions_count = 0
    if not results_list:
        return actions_count

    _log = log_callback if callable(log_callback) else lambda k, **kw: None
    _status = status_callback if callable(status_callback) else lambda k, **kw: None
    _progress = progress_callback if callable(progress_callback) else lambda v: None

    to_process = [r for r in results_list if r.get('rejected_reason') == 'not_in_recommendation' and r.get('action') == 'pending_reco_action' and r.get('status') == 'ok']
    total = len(to_process)
    if total == 0:
        _log('logic_info_prefix', text="Aucune action recommandation en attente.")
        return 0

    _status('status_custom', text=f'Application des actions recommandation sur {total} fichiers...')
    _progress(0)

    for i, r in enumerate(to_process):
        _progress(((i + 1) / total) * 100)
        try:
            rel_path = os.path.relpath(r.get('path'), input_dir_abs) if r.get('path') and input_dir_abs else r.get('file', 'N/A')
        except ValueError:
            rel_path = r.get('file', 'N/A')

        _status('status_custom', text=f'Action reco sur {rel_path} ({i+1}/{total})')

        current_path = r.get('path')
        if not current_path or not os.path.exists(current_path):
            _log('logic_move_skipped', file=rel_path, e='Fichier source introuvable pour action recommandation.')
            r['action_comment'] = r.get('action_comment', '') + ' Source non trouvée pour action différée.'
            r['action'] = 'error_action_deferred'
            r['status'] = 'error'
            continue

        action_done = False
        original_reason = r['rejected_reason']
        if delete_rejected_flag:
            try:
                os.remove(current_path)
                _log('logic_info_prefix', text=f'Fichier supprimé (reco): {rel_path}')
                r['path'] = None
                r['action'] = 'deleted_reco'
                r['rejected_reason'] = 'not_in_recommendation'
                r['status'] = 'processed_action'
                actions_count += 1
                action_done = True
            except Exception as del_e:
                _log('logic_error_prefix', text=f'Erreur suppression reco {rel_path}: {del_e}')
                r['action_comment'] = r.get('action_comment', '') + f' Erreur suppression différée: {del_e}'
                r['action'] = 'error_delete'
                r['rejected_reason'] = original_reason
        elif move_rejected_flag and reject_abs_path:
            if not os.path.isdir(reject_abs_path):
                try:
                    os.makedirs(reject_abs_path)
                    _log('logic_dir_created', path=reject_abs_path)
                except OSError as e_mkdir:
                    _log('logic_dir_create_error', path=reject_abs_path, e=e_mkdir)
                    r['action_comment'] = r.get('action_comment', '') + f' Dossier rejet reco inaccessible: {e_mkdir}'
                    r['action'] = 'error_move'
                    r['rejected_reason'] = original_reason
                    continue

            dest_path = os.path.join(reject_abs_path, os.path.basename(current_path))
            try:
                if os.path.normpath(current_path) != os.path.normpath(dest_path):
                    shutil.move(current_path, dest_path)
                    _log('logic_moved_info', folder=os.path.basename(reject_abs_path), text_key_suffix='_deferred_reco', file_rel_path=rel_path)
                    r['path'] = dest_path
                    r['action'] = 'moved_reco'
                    r['rejected_reason'] = 'not_in_recommendation'
                    r['status'] = 'processed_action'
                    actions_count += 1
                    action_done = True
                else:
                    r['action_comment'] = r.get('action_comment', '') + ' Déjà dans dossier cible (différé)?'
                    r['action'] = 'kept'
                    r['rejected_reason'] = 'not_in_recommendation'
                    r['status'] = 'processed_action'
                    action_done = True
            except Exception as move_e:
                _log('logic_move_error', file=rel_path, e=move_e)
                r['action_comment'] = r.get('action_comment', '') + f' Erreur déplacement différé: {move_e}'
                r['action'] = 'error_move'
                r['rejected_reason'] = original_reason

        if not action_done and not delete_rejected_flag and not move_rejected_flag:
            r['action'] = 'kept_pending_reco_no_action'
            r['rejected_reason'] = 'not_in_recommendation'
            r['status'] = 'processed_action'
            r['action_comment'] = r.get('action_comment', '') + ' Action recommandation différée mais aucune opération configurée.'

    _progress(100)
    _status('status_custom', text=f'{actions_count} actions recommandation appliquées.')
    _log('logic_info_prefix', text=f'{actions_count} actions recommandation appliquées.')
    return actions_count


def build_recommended_images(results):
    """Return list of files meeting SNR/FWHM/Ecc criteria."""
    kept = [r for r in results if r.get('action') == 'kept']

    snrs = np.array([
        r.get('snr') if r.get('snr') is not None else np.nan
        for r in kept
    ], dtype=float)
    fwhms = np.array([
        r.get('fwhm') if r.get('fwhm') is not None else np.nan
        for r in kept
    ], dtype=float)
    eccs = np.array([
        r.get('ecc') if r.get('ecc') is not None
        else (r.get('e') if r.get('e') is not None else np.nan)
        for r in kept
    ], dtype=float)

    if kept:
        snr_min = float(np.nanpercentile(snrs, 25))
        fwhm_max = float(np.nanpercentile(fwhms, 75))
        ecc_max = float(np.nanpercentile(eccs, 75))
    else:
        snr_min = fwhm_max = ecc_max = float('nan')

    reco = [
        r for r in kept
        if (
            r.get('snr') is not None
            and float(r.get('snr') if r.get('snr') is not None else np.nan) >= snr_min
        )
        and (
            float(r.get('fwhm') if r.get('fwhm') is not None else np.nan)
            <= fwhm_max
        )
        and (
            float(
                r.get('ecc') if r.get('ecc') is not None
                else (r.get('e') if r.get('e') is not None else np.nan)
            )
            <= ecc_max
        )
    ]
    return reco, snr_min, fwhm_max, ecc_max


def debug_counts(results):
    total = len(results)
    low_snr = sum(r.get('rejected_reason') == 'low_snr' for r in results)
    high_fwhm = sum(r.get('rejected_reason') == 'high_fwhm' for r in results)
    starcount = sum(r.get('rejected_reason') == 'starcount_out' for r in results)
    ecc = sum(r.get('rejected_reason') == 'high_eccentricity' for r in results)
    pending = sum(str(r.get('action', '')).startswith('pending') for r in results)
    print(f"total={total} | snr={low_snr} | fwhm={high_fwhm} | stars={starcount} | e={ecc} | pending={pending}")


def write_telescope_pollution_csv(csv_path, results_list, bortle_dataset=None):
    """Write per-telescope light pollution info to a CSV file."""
    telescopes = {}
    for r in results_list:
        if r.get('status') != 'ok':
            continue
        tele = r.get('telescope') or 'Unknown'
        if tele in telescopes:
            continue
        lon = r.get('sitelong')
        lat = r.get('sitelat')
        try:
            lon = float(lon) if lon is not None else None
            lat = float(lat) if lat is not None else None
        except Exception:
            lon = lat = None
        l_ucd = None
        sqm = None
        if bortle_dataset and lon is not None and lat is not None:
            try:
                l_ucd = bortle_utils.sample_bortle_dataset(bortle_dataset, lon, lat)
                sqm = artif_ratio_to_sqm(l_ucd)
            except Exception:
                l_ucd = None
                sqm = None
        telescopes[tele] = {
            'lon': lon,
            'lat': lat,
            'l_ucd_artif': l_ucd,
            'sqm': sqm,
        }

    try:
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['telescope', 'longitude', 'latitude', 'l_ucd_artif', 'sqm'])
            for tele, data in telescopes.items():
                writer.writerow([
                    tele,
                    '' if data['lon'] is None else data['lon'],
                    '' if data['lat'] is None else data['lat'],
                    '' if data['l_ucd_artif'] is None else f"{data['l_ucd_artif']:.2f}",
                    '' if data['sqm'] is None else f"{data['sqm']:.2f}",
                ])
    except Exception:
        raise


def apply_pending_organization(results_list, log_callback=None,
                               status_callback=None, progress_callback=None,
                               input_dir_abs=None):
    """Move kept images to their destination folders."""
    actions_count = 0
    if not results_list:
        return actions_count

    _log = log_callback if callable(log_callback) else lambda k, **kw: None
    _status = status_callback if callable(status_callback) else lambda k, **kw: None
    _progress = progress_callback if callable(progress_callback) else lambda v: None

    to_process = [
        r for r in results_list
        if r.get('status') == 'ok' and r.get('action') == 'kept'
        and r.get('path') and r.get('filepath_dst')
    ]
    total = len(to_process)
    if total == 0:
        _log('logic_info_prefix', text='Aucun fichier à organiser.')
        return 0

    _status('status_custom', text=f'Organisation de {total} fichiers...')
    for i, r in enumerate(to_process):
        _progress(((i + 1) / total) * 100)
        current_path = r.get('path')
        dest_path = r.get('filepath_dst')
        try:
            rel_path = os.path.relpath(current_path, input_dir_abs) if current_path and input_dir_abs else os.path.basename(current_path)
        except ValueError:
            rel_path = os.path.basename(current_path)

        if not dest_path or os.path.normpath(current_path) == os.path.normpath(dest_path):
            continue
        try:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.move(current_path, dest_path)
            r['path'] = dest_path
            actions_count += 1
            _log('logic_moved_info', folder=os.path.basename(os.path.dirname(dest_path)), text_key_suffix='_organize', file_rel_path=rel_path)
        except Exception as e:
            _log('logic_move_error', file=rel_path, e=e)
    _progress(100)
    _status('status_custom', text=f'{actions_count} fichiers organisés.')
    _log('logic_info_prefix', text=f'{actions_count} fichiers organisés.')
    return actions_count



# --- Helpers for parallel processing ---

def _snr_worker(path):
    """Worker to compute SNR and star count for a FITS file."""

    result = {
        'path': path,
        'snr': np.nan,
        'sky_bg': np.nan,
        'sky_noise': np.nan,
        'signal_pixels': 0,
        'starcount': None,
        'exposure': 'N/A',
        'filter': 'N/A',
        'temperature': 'N/A',
        'eqmode': 2,
        'sitelong': None,
        'sitelat': None,
        'telescope': None,
        'date_obs': None,
        'error': None,
        'fwhm': np.nan,
        'ecc': np.nan,
        'n_star_ecc': 0,
    }
    hdul = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            hdul = fits.open(path, memmap=False, lazy_load_hdus=True)
            if hdul and len(hdul) > 0 and hasattr(hdul[0], 'data') and hdul[0].data is not None:
                data = hdul[0].data
                header = hdul[0].header
                result['eqmode'] = header.get('EQMODE', 2)
                result['sitelong'] = header.get('SITELONG')
                result['sitelat'] = header.get('SITELAT')
                result['telescope'] = header.get('TELESCOP', header.get('TELESCOPE'))
                result['date_obs'] = header.get('DATE-OBS')
                result['exposure'] = header.get('EXPTIME', header.get('EXPOSURE', 'N/A'))
                result['filter'] = header.get('FILTER', 'N/A')
                result['temperature'] = header.get('CCD-TEMP', header.get('TEMPERAT', 'N/A'))
                snr, sky_bg, sky_noise, signal_pixels = snr_module.calculate_snr(data)
                result.update({'snr': snr, 'sky_bg': sky_bg, 'sky_noise': sky_noise, 'signal_pixels': signal_pixels})

                if starcount_module is not None:

                    try:
                        result['starcount'] = starcount_module.calculate_starcount(data)
                    except Exception:
                        result['starcount'] = None

                if ecc_module is not None:

                    try:
                        fwhm_val, ecc_val, n_det = ecc_module.calculate_fwhm_ecc(data)
                        result['fwhm'] = fwhm_val
                        result['ecc'] = ecc_val
                        result['n_star_ecc'] = n_det
                    except Exception:
                        result['fwhm'] = np.nan
                        result['ecc'] = np.nan
                        result['n_star_ecc'] = 0
            else:
                result['error'] = 'Pas de données image valides dans HDU 0.'
    except Exception as e:
        result['error'] = str(e)
    finally:
        if hdul:
            try:
                hdul.close()
            except Exception:
                pass
    return result


def _trail_worker(args):
    """Worker to run trail detection on a chunk."""
    chunk, params = args
    return trail_module.run_trail_detection(chunk, params)


# --- Fonction principale d'analyse (Orchestrateur) ---
def perform_analysis(input_dir, output_log, options, callbacks):
    """
    Fonction principale d'orchestration de l'analyse.
    ORDRE: Découverte Fichiers (avec skip marqueur) -> SNR -> SNR Rejection/Action -> Trail Detection -> Trail Rejection/Action -> Création Marqueur -> Logging
    Gère la récursion et l'exclusion des dossiers de rejet.
    """
    _status = callbacks.get('status', lambda k, **kw: None)
    _progress = callbacks.get('progress', lambda v: None)
    _log = callbacks.get('log', lambda k, **kw: print(f"LOGIC_LOG: {k} {kw}"))

    _status("status_analysis_prep")
    start_time = time.time()

    import multiprocessing
    n_cpus = multiprocessing.cpu_count()
    n_workers = max(1, int(n_cpus * 0.75))

    bortle_dataset = None
    bortle_lock = threading.Lock()
    if options.get('use_bortle') and options.get('bortle_path'):
        try:
            bortle_dataset = _load_bortle_raster(options['bortle_path'])
        except Exception as e:
            _log('logic_warn_prefix', text=f'Bortle raster load error: {e}')

    # --- NOUVEAU : Extraire les options pour l'application immédiate des actions ---
    apply_snr_action_immediately = options.get('apply_snr_action_immediately', True)
    apply_trail_action_immediately = options.get('apply_trail_action_immediately', True)
    # --- FIN NOUVEAU ---

    # --- Validation chemins & Création dossiers ---
    if not input_dir or not os.path.isdir(input_dir): 
        _log("logic_error_prefix", clear=True, text=f"Dossier d'entrée invalide: {input_dir}")
        _status("status_dir_create_error", e=f"Input folder invalid: {input_dir}")
        return []
    if not output_log: 
        _log("logic_error_prefix", clear=True, text="Fichier log non spécifié.")
        _status("msg_log_file_missing")
        return []
    
    abs_input_dir = os.path.abspath(input_dir)
    output_root = options.get('output_root', abs_input_dir)
    reject_dirs_to_exclude_abs = []
    snr_reject_abs = None
    trail_reject_abs = None

    # Dossier de rejet SNR (même si l'action n'est pas immédiate, on a besoin du chemin pour plus tard)
    if options.get('analyze_snr') and options.get('snr_selection_mode') != 'none':
        snr_reject_rel = options.get('snr_reject_dir')
        if options.get('move_rejected', False) and not snr_reject_rel : # Vérifier si move est activé ET que le chemin est manquant
            _log("logic_error_prefix", clear=True, text="Chemin dossier rejet SNR non spécifié mais déplacement activé.")
            return []
        if snr_reject_rel: # Si un chemin est fourni (même si move_rejected est False, on le prépare)
            snr_reject_abs = os.path.abspath(snr_reject_rel)
            reject_dirs_to_exclude_abs.append(snr_reject_abs)
            if not os.path.exists(snr_reject_abs):
                try: 
                    os.makedirs(snr_reject_abs)
                    _log("logic_dir_created", path=snr_reject_abs)
                except OSError as e: 
                    _log("logic_dir_create_error", path=snr_reject_abs, e=e)
                    _status("status_dir_create_error", e=e)
                    return []
            elif not os.path.isdir(snr_reject_abs): 
                _log("logic_error_prefix", text=f"Chemin rejet SNR n'est pas un dossier: {snr_reject_abs}")
                _status("status_dir_create_error", e="SNR Reject path is not a directory")
                return []

    # Dossier de rejet Trail (logique inchangée, appliqué immédiatement)
    if options.get('detect_trails') and SATDET_AVAILABLE:
        trail_reject_rel = options.get('trail_reject_dir')
        if options.get('move_rejected', False) and not trail_reject_rel:
            _log("logic_error_prefix", clear=True, text="Chemin dossier rejet Trail non spécifié mais déplacement activé.")
            return []
        if trail_reject_rel:
            trail_reject_abs = os.path.abspath(trail_reject_rel)
            reject_dirs_to_exclude_abs.append(trail_reject_abs)
            if not os.path.exists(trail_reject_abs):
                try: 
                    os.makedirs(trail_reject_abs)
                    _log("logic_dir_created", path=trail_reject_abs)
                except OSError as e: 
                    _log("logic_dir_create_error", path=trail_reject_abs, e=e)
                    _status("status_dir_create_error", e=e)
                    return []
            elif not os.path.isdir(trail_reject_abs): 
                _log("logic_error_prefix", text=f"Chemin rejet Trail n'est pas un dossier: {trail_reject_abs}")
                _status("status_dir_create_error", e="Trail Reject path is not a directory")
                return []

    # Initialiser le log
    try:
        with open(output_log, 'w', encoding='utf-8') as f:
             f.write(f"Début de l'analyse: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
             f.write(f"Dossier d'entrée: {abs_input_dir}\n")
             f.write("Options d'analyse:\n")
             for key, value in options.items():
                 if key != 'trail_params' and key != 'apply_snr_action_immediately' and not callable(value) : 
                     f.write(f"  {key}: {value}\n")
                 elif key == 'trail_params': 
                     f.write(f"  trail_params: {value}\n")
             # Loguer la nouvelle option
             f.write(f"  apply_snr_action_immediately: {apply_snr_action_immediately}\n")
             f.write("="*80 + "\n")
    except IOError as e: 
        _log("logic_log_init_error", clear=True, path=output_log, e=e)
        _status("status_log_error")
        return []

    # --- Étape 1: Découverte des fichiers FITS (avec skip marqueur) ---
    _progress(0.0); _status("status_discovery_start")
    fits_files_to_process = []
    processed_directories = set() 
    skipped_marker_dirs_count = 0 # Correctement initialisé ici
    include_subfolders = options.get('include_subfolders', False)
    fits_extensions = ('.fit', '.fits', '.fts')
    marker_filename = ".astro_analyzer_run_complete" 

    try:
        for dirpath, dirnames, filenames in os.walk(abs_input_dir, topdown=True):
            current_dir_abs = os.path.abspath(dirpath)
            dirs_to_remove = [d for d in dirnames if os.path.abspath(os.path.join(current_dir_abs, d)) in reject_dirs_to_exclude_abs]
            if dirs_to_remove:
                for dname in dirs_to_remove:
                    _log("logic_info_prefix", text=f"Exclusion du sous-dossier de rejet: {os.path.relpath(os.path.join(current_dir_abs, dname), abs_input_dir)}")
                    dirnames.remove(dname)
            marker_file_path = os.path.join(current_dir_abs, marker_filename)
            if os.path.exists(marker_file_path):
                _log("logic_info_prefix", text=f"Ignoré (marqueur trouvé): {os.path.relpath(current_dir_abs, abs_input_dir)}")
                skipped_marker_dirs_count += 1
                dirnames[:] = [] 
                continue 
            found_in_this_dir = False
            for filename in filenames:
                if filename.lower().endswith(fits_extensions):
                    full_path = os.path.join(current_dir_abs, filename)
                    fits_files_to_process.append(full_path)
                    found_in_this_dir = True
            if found_in_this_dir:
                processed_directories.add(current_dir_abs)
            if not include_subfolders and dirpath == abs_input_dir: 
                dirnames[:] = []
    except OSError as e: 
        _log("logic_error_prefix", text=f"Erreur lors du parcours du dossier {abs_input_dir}: {e}")
        _status("status_dir_create_error", e=f"Error walking directory: {e}")

        write_log_summary(output_log, abs_input_dir, options, None, None, [], None, skipped_marker_dirs_count)

        write_log_summary(output_log, abs_input_dir, options, None, None, [], None, skipped_marker_dirs_count)
        return []

    fits_files_to_process = sorted(list(set(fits_files_to_process)))
    total_files = len(fits_files_to_process)
    if total_files == 0:
        _log("logic_no_fits_snr"); _status("status_analysis_done_no_valid")
        write_log_summary(output_log, abs_input_dir, options, None, None, [], None, skipped_marker_dirs_count)
        # Création marqueur même si pas de fichiers, car le dossier a été "visité"
        analysis_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for directory_to_mark in processed_directories: # processed_directories sera vide si aucun FITS, mais on garde la logique
            if not os.path.exists(os.path.join(directory_to_mark, marker_filename)): # Vérifier avant de marquer
                 marker_file_path_empty = os.path.join(directory_to_mark, marker_filename)
                 try:
                     with open(marker_file_path_empty, 'w', encoding='utf-8') as mf:
                         mf.write(f"Analyse Astro Analyzer terminée pour ce dossier le: {analysis_datetime}\n")
                         mf.write(f"  (Aucun fichier FITS traitable trouvé lors de cette analyse)\n")
                     _log("logic_info_prefix", text=f"Marqueur créé (aucun FITS): {os.path.relpath(marker_file_path_empty, abs_input_dir)}")
                 except IOError as e_marker:
                     _log("logic_error_prefix", text=f"Impossible de créer le fichier marqueur (aucun FITS) dans {directory_to_mark}: {e_marker}")
        return []
    _log("logic_info_prefix", text=f"{total_files} fichiers FITS trouvés pour analyse (hors dossiers ignorés).")


    # --- Étape 2: Boucle Analyse SNR ---
    # ... (cette section reste identique) ...
    all_results_list = []
    _log("logic_snr_start")
    snr_loop_errors = 0
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:

            future_map = {ex.submit(_snr_worker, p): p for p in fits_files_to_process}

            for idx, future in enumerate(concurrent.futures.as_completed(future_map)):
                fits_file_path = future_map[future]
                progress = ((idx + 1) / total_files) * 50
                try:
                    rel_path_for_status = os.path.relpath(fits_file_path, abs_input_dir)
                except ValueError:
                    rel_path_for_status = os.path.basename(fits_file_path)
                _progress(progress)
                _status("status_snr_start", file=rel_path_for_status, i=idx+1, total=total_files)
                result_base = {
                    'file': os.path.basename(fits_file_path),
                    'path': fits_file_path,
                    'rel_path': rel_path_for_status,
                    'status': 'pending',
                    'action': 'kept',
                    'rejected_reason': None,
                    'action_comment': '',
                    'error_message': '',
                    'has_trails': False,
                    'num_trails': 0,
                    'starcount': None,
                    'fwhm': np.nan,
                    'ecc': np.nan,
                    'n_star_ecc': 0,
                    'eqmode': 2,
                    'sitelong': None,
                    'sitelat': None,
                    'telescope': None,
                    'date_obs': None,
                }
                try:
                    worker_res = future.result()
                    if options.get('analyze_snr'):
                        if worker_res.get('error'):
                            raise Exception(worker_res['error'])
                        snr_data = {
                            'snr': worker_res['snr'],
                            'sky_bg': worker_res['sky_bg'],
                            'sky_noise': worker_res['sky_noise'],
                            'signal_pixels': worker_res['signal_pixels'],
                            'exposure': worker_res['exposure'],
                            'filter': worker_res['filter'],
                            'temperature': worker_res['temperature'],
                        }
                        result_base.update(snr_data)
                        result_base['eqmode'] = worker_res.get('eqmode', 2)
                        result_base['sitelong'] = worker_res.get('sitelong')
                        result_base['sitelat'] = worker_res.get('sitelat')
                        result_base['telescope'] = worker_res.get('telescope')
                        result_base['date_obs'] = worker_res.get('date_obs')
                        if 'starcount' in worker_res:
                            result_base['starcount'] = worker_res['starcount']
                        if 'fwhm' in worker_res:
                            result_base['fwhm'] = worker_res['fwhm']
                        if 'ecc' in worker_res:
                            result_base['ecc'] = worker_res['ecc']
                        if 'n_star_ecc' in worker_res:
                            result_base['n_star_ecc'] = worker_res['n_star_ecc']
                        result_base['status'] = 'ok'
                        _log("logic_snr_info", file=result_base['rel_path'], snr=worker_res['snr'], bg=worker_res['sky_bg'])
                    else:
                        if result_base['status'] == 'pending':
                            result_base['status'] = 'ok'
                except Exception as snr_e:
                    err_msg = f"Erreur analyse SNR/FITS: {snr_e}"
                    result_base['status'] = 'error'
                    result_base['error_message'] = err_msg
                    snr_loop_errors += 1
                    _log("logic_file_error", file=result_base['rel_path'], e=err_msg)
                all_results_list.append(result_base)
    except Exception as pool_e:
        _log("logic_warn_prefix", text=f"Echec pool SNR ({pool_e}), fallback séquentiel")
        all_results_list = []
        snr_loop_errors = 0

        for i, fits_file_path in enumerate(fits_files_to_process):
            progress = ((i + 1) / total_files) * 50
            try:
                rel_path_for_status = os.path.relpath(fits_file_path, abs_input_dir)
            except ValueError:
                rel_path_for_status = os.path.basename(fits_file_path)
            _progress(progress)
            _status("status_snr_start", file=rel_path_for_status, i=i+1, total=total_files)
            result = {
                'file': os.path.basename(fits_file_path),
                'path': fits_file_path,
                'rel_path': rel_path_for_status,
                'status': 'pending',
                'action': 'kept',
                'rejected_reason': None,
                'action_comment': '',
                'error_message': '',
                'has_trails': False,
                'num_trails': 0,
                'starcount': None,
                'eqmode': 2,
                'sitelong': None,
                'sitelat': None,
                'telescope': None,
                'date_obs': None,
            }
            try:
                if options.get('analyze_snr'):
                    hdul = None
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', AstropyWarning)
                        try:
                            hdul = fits.open(fits_file_path, memmap=False, lazy_load_hdus=True)
                            if hdul and len(hdul) > 0 and hasattr(hdul[0], 'data') and hdul[0].data is not None:
                                data = hdul[0].data
                                header = hdul[0].header
                                result['eqmode'] = header.get('EQMODE', 2)
                                result['sitelong'] = header.get('SITELONG')
                                result['sitelat'] = header.get('SITELAT')
                                result['telescope'] = header.get('TELESCOP', header.get('TELESCOPE'))
                                result['date_obs'] = header.get('DATE-OBS')
                                exposure = header.get('EXPTIME', header.get('EXPOSURE', 'N/A'))
                                filter_name = header.get('FILTER', 'N/A')
                                temperature = header.get('CCD-TEMP', header.get('TEMPERAT', 'N/A'))
                                snr, sky_bg, sky_noise, signal_pixels = snr_module.calculate_snr(data)
                                if np.isfinite(snr) and np.isfinite(sky_bg) and np.isfinite(sky_noise):
                                    snr_data = {
                                        'snr': snr,
                                        'sky_bg': sky_bg,
                                        'sky_noise': sky_noise,
                                        'signal_pixels': signal_pixels,
                                        'exposure': exposure,
                                        'filter': filter_name,
                                        'temperature': temperature,
                                    }
                                    result.update(snr_data)

                                    if starcount_module is not None:

                                        try:
                                            result['starcount'] = starcount_module.calculate_starcount(data)
                                        except Exception:
                                            result['starcount'] = None

                                    if ecc_module is not None:

                                        try:
                                            fwhm_val, ecc_val, n_det = ecc_module.calculate_fwhm_ecc(data)
                                            result['fwhm'] = fwhm_val
                                            result['ecc'] = ecc_val
                                            result['n_star_ecc'] = n_det
                                        except Exception:
                                            result['fwhm'] = np.nan
                                            result['ecc'] = np.nan
                                            result['n_star_ecc'] = 0
                                    result['status'] = 'ok'
                                    _log("logic_snr_info", file=result['rel_path'], snr=snr, bg=sky_bg)
                                else:
                                    raise ValueError("Calcul SNR a retourné des valeurs non finies.")
                            else:
                                raise ValueError("Pas de données image valides dans HDU 0.")
                        except Exception as snr_e:
                            err_msg = f"Erreur analyse SNR/FITS: {snr_e}"
                            result['status'] = 'error'
                            result['error_message'] = err_msg
                            snr_loop_errors += 1
                            _log("logic_file_error", file=result['rel_path'], e=err_msg)
                        finally:
                            if hdul:
                                try:
                                    hdul.close()
                                except Exception as e_close:
                                    print(f"WARN: Erreur fermeture FITS {result['rel_path']}: {e_close}")
                else:
                    if result['status'] == 'pending':
                        result['status'] = 'ok'
            except Exception as file_e:
                err_msg = f"Erreur traitement général: {file_e}"
                result['status'] = 'error'
                result['error_message'] = err_msg
                snr_loop_errors += 1
                _log("logic_file_error", file=result['rel_path'], e=err_msg)
                print(f"\n--- Traceback Erreur Fichier (Logic) {result['rel_path']} ---")
                traceback.print_exc()
                print("---------------------------------------------------\n")
            finally:
                all_results_list.append(result)
    if snr_loop_errors > 0:
        _log("logic_warn_prefix", text=f"{snr_loop_errors} erreur(s) rencontrée(s) pendant l'analyse SNR initiale.")


    # --- Étape 3: Calcul Seuil SNR ---
    # ... (cette section reste identique) ...
    snr_threshold = -np.inf
    selection_stats = None
    starcount_threshold = options.get('starcount_threshold')
    fwhm_max_slider = options.get('fwhm_max')
    fwhm_min_slider = options.get('fwhm_min')
    ecc_max_slider = options.get('ecc_max')
    ecc_min_slider = options.get('ecc_min')
    if options.get('analyze_snr') and options.get('snr_selection_mode') != 'none':
        mode = options.get('snr_selection_mode')
        value_str = options.get('snr_selection_value')
        selection_stats = {'mode': mode, 'value': value_str, 'threshold': None, 'initial_count': 0, 'selected_count': 0, 'rejected_count': 0}
        eligible_snr_results = [(r['snr'], idx) for idx, r in enumerate(all_results_list) if r['status'] == 'ok' and 'snr' in r and np.isfinite(r['snr'])]
        selection_stats['initial_count'] = len(eligible_snr_results)
        if selection_stats['initial_count'] > 0:
            eligible_snrs = [res[0] for res in eligible_snr_results]; local_snr_threshold = -np.inf
            if mode == 'threshold':
                try: local_snr_threshold = float(value_str); selection_stats['threshold'] = local_snr_threshold
                except (ValueError, TypeError): _log("logic_warn_prefix", text=f"Seuil SNR invalide: '{value_str}'.")
            elif mode == 'percent':
                try:
                    percentile_to_keep = float(value_str); assert 0 < percentile_to_keep <= 100
                    percentile_val = 100.0 - percentile_to_keep; local_snr_threshold = np.nanpercentile(eligible_snrs, percentile_val); selection_stats['threshold'] = local_snr_threshold
                except Exception as e: _log("logic_warn_prefix", text=f"Valeur pourcentage invalide: '{value_str}' ({e}).")
            elif mode == 'none': selection_stats['threshold'] = None 
            snr_threshold = local_snr_threshold


    # --- Étape 4: Rejet SNR et Actions Associées ---
    _log("logic_info_prefix", text="Application du marquage/rejet SNR et actions...")
    kept_by_snr_initial = 0; rejected_by_snr_initial = 0; files_kept_for_trails = []
    snr_filter_active = options.get('analyze_snr') and options.get('snr_selection_mode') != 'none' and np.isfinite(snr_threshold) and snr_threshold > -np.inf
    
    if snr_filter_active: _log("logic_info_prefix", text=f"Seuil SNR appliqué: {snr_threshold:.3f}")
    
    for r_idx, r in enumerate(all_results_list):
        progress_snr_action = 50 + ((r_idx + 1) / total_files) * 5 # Petite progression pour cette étape
        _progress(progress_snr_action)

        process_for_trails = True # Par défaut, on traite pour les traînées
        if r['status'] == 'ok' and r.get('rejected_reason') is None:
            if snr_filter_active and 'snr' in r and np.isfinite(r['snr']):
                if r['snr'] < snr_threshold:
                    # --- MODIFIÉ : Logique pour action immédiate ou différée ---
                    if apply_snr_action_immediately:
                        r['rejected_reason'] = 'low_snr' # Action immédiate
                        _log("logic_info_prefix", text=f"SNR Rejet (immédiat): {r['rel_path']} (SNR: {r['snr']:.2f} < {snr_threshold:.2f})")
                    else:
                        r['rejected_reason'] = 'low_snr_pending_action' # Marquer pour action différée
                        r['action'] = 'pending_snr_action' # Statut d'action en attente
                        _log("logic_info_prefix", text=f"SNR Marqué pour rejet (différé): {r['rel_path']} (SNR: {r['snr']:.2f} < {snr_threshold:.2f})")
                    # --- FIN MODIFICATION ---
                    rejected_by_snr_initial += 1
                    
                    # --- MODIFIÉ : Conditionner l'action de déplacement/suppression ---
                    if apply_snr_action_immediately:
                        action_to_take = 'kept' # Par défaut on garde (si pas de delete/move)
                        reject_dir_option = options.get('snr_reject_dir') 
                        
                        if options.get('delete_rejected'): 
                            action_to_take = 'deleted_snr'
                        elif options.get('move_rejected') and reject_dir_option and snr_reject_abs: # Vérifier aussi snr_reject_abs
                            action_to_take = 'moved_snr'

                        if action_to_take != 'kept':
                            current_path = r['path']
                            if current_path and os.path.exists(current_path):
                                process_for_trails = False # Ne pas analyser les traînées si rejeté par SNR et actionné
                                if action_to_take == 'moved_snr':
                                    dest_path = os.path.join(snr_reject_abs, os.path.basename(current_path))
                                    try:
                                         if os.path.normpath(current_path) != os.path.normpath(dest_path): 
                                             shutil.move(current_path, dest_path)
                                             _log("logic_moved_info", folder=os.path.basename(snr_reject_abs), text_key_suffix="_snr", file_rel_path=r['rel_path'])
                                             r['path'] = dest_path; r['action'] = 'moved_snr'
                                         else: 
                                             r['action_comment'] += " Déjà dans dossier cible SNR?"; r['action'] = 'kept'; process_for_trails = True
                                    except Exception as move_e: 
                                        _log("logic_move_error", file=r['rel_path'], e=move_e)
                                        r['action_comment'] += f" Erreur déplacement SNR: {move_e}"; r['action'] = 'error_move'; r['rejected_reason'] = None; process_for_trails = True
                                elif action_to_take == 'deleted_snr':
                                    try: 
                                        os.remove(current_path)
                                        _log("logic_info_prefix", text=f"Fichier supprimé (SNR): {r['rel_path']}")
                                        r['path'] = None; r['action'] = 'deleted_snr'
                                    except Exception as del_e: 
                                        _log("logic_error_prefix", text=f"Erreur suppression SNR {r['rel_path']}: {del_e}")
                                        r['action_comment'] += f" Erreur suppression SNR: {del_e}"; r['action'] = 'error_delete'; r['rejected_reason'] = None; process_for_trails = True
                            else: 
                                _log("logic_move_skipped", file=r['rel_path'], e="Fichier source non trouvé pour action SNR.")
                                r['action_comment'] += " Ignoré action SNR (source non trouvée)."; r['action'] = 'error_action'; r['rejected_reason'] = None; process_for_trails = True
                        else: # action_to_take == 'kept' (donc ni delete ni move activé pour SNR)
                            r['action'] = 'kept' # Même si raison rejet = low_snr
                            r['action_comment'] += " Rejeté (faible SNR) mais action=none."
                            process_for_trails = False # On ne traite pas les traînées si marqué rejeté SNR et aucune action fichier
                    else: # apply_snr_action_immediately is False
                        # Le fichier est marqué 'low_snr_pending_action'.
                        # Il ne sera pas déplacé/supprimé ici.
                        # Il sera toujours considéré pour l'analyse des traînées car process_for_trails est True.
                        pass 
                    # --- FIN MODIFICATION ---
                else:  # r['snr'] >= snr_threshold
                    kept_by_snr_initial += 1
            # else: Fichier OK mais pas de filtre SNR actif, ou SNR non valide (process_for_trails reste True)

            if starcount_threshold is not None and r.get('starcount') is not None:
                if r['starcount'] < starcount_threshold:
                    r['rejected_reason'] = 'starcount_pending_action'
                    r['action'] = 'pending_starcount_action'
                    process_for_trails = False

            if options.get('analyse_fwhm') and np.isfinite(r.get('fwhm', np.nan)):
                if (fwhm_max_slider is not None and r['fwhm'] > fwhm_max_slider) or \
                   (fwhm_min_slider is not None and r['fwhm'] < fwhm_min_slider):
                    r['rejected_reason'] = 'high_fwhm_pending_action'
                    r['action'] = 'pending_fwhm_action'
                    process_for_trails = False

            if options.get('analyse_ecc') and np.isfinite(r.get('ecc', np.nan)):
                if (ecc_max_slider is not None and r['ecc'] > ecc_max_slider) or \
                   (ecc_min_slider is not None and r['ecc'] < ecc_min_slider):
                    r['rejected_reason'] = 'high_ecc_pending_action'
                    r['action'] = 'pending_ecc_action'
                    process_for_trails = False
        
        # Ajouter à la liste pour analyse des traînées si applicable
        # Un fichier marqué 'low_snr_pending_action' EST toujours candidat pour l'analyse des traînées.
        # Il ne sera retiré de la liste que si apply_snr_action_immediately est True ET qu'une action de déplacement/suppression a eu lieu.
        if r['status'] == 'ok' and \
           (r.get('rejected_reason') is None or r.get('rejected_reason') == 'low_snr_pending_action') and \
           r['path'] is not None and \
           process_for_trails:
            files_kept_for_trails.append(r['path'])
            
    if selection_stats: 
        selection_stats['selected_count'] = kept_by_snr_initial
        selection_stats['rejected_count'] = rejected_by_snr_initial


    # --- Étape 5: Détection des traînées ---
    # ... (cette section reste identique, elle utilise files_kept_for_trails
    #      qui contient maintenant aussi les fichiers marqués 'low_snr_pending_action') ...
    #      Progression de 55% à 90%
    trail_results = {}
    trail_errors = {}
    trail_analysis_config = None
    if options.get('detect_trails') and SATDET_AVAILABLE:
        if not TRAIL_MODULE_LOADED or not hasattr(trail_module, 'run_trail_detection'):
            _log("logic_error_prefix", text="Erreur: trail_module n'est pas chargé ou manque run_trail_detection.")
            options['detect_trails'] = False
        elif not files_kept_for_trails:
            _log("logic_info_prefix", text="Aucun fichier éligible pour la détection de traînées après filtre SNR.")
        else:
            input_for_trail_module = files_kept_for_trails
            trail_params = options.get('trail_params', {})
            trail_analysis_config = trail_params.copy()
            chunks = [input_for_trail_module[i::n_workers] for i in range(n_workers)]
            _progress('indeterminate')
            _status("status_custom", text="Lancement détection traînées...")
            completed = 0
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futures = {ex.submit(_trail_worker, (chunk, trail_params)): chunk for chunk in chunks if chunk}
                    total_chunks = len(futures)
                    for future in concurrent.futures.as_completed(futures):
                        res, err = future.result()
                        trail_results.update(res or {})
                        trail_errors.update(err or {})
                        completed += 1
                        prog = 55 + ((completed / total_chunks) * 35)
                        _progress(min(prog, 90.0))
            except Exception as trail_e:
                _log("logic_error_prefix", text=f"Erreur pool trail: {trail_e}")
                traceback.print_exc()
                trail_errors[('FATAL_CALL_ERROR', 0)] = str(trail_e)
            _progress(90.0)
    else:
        _progress(90.0)


    # --- Étape 6: Rejet Traînées et Actions Associées ---
    # ... (cette section reste identique, elle modifie les items dans all_results_list) ...
    #      Progression de 90% à 95%
    _log("logic_info_prefix", text="Application du rejet Traînées et actions...")
    if options.get('detect_trails') and SATDET_AVAILABLE: # SATDET_AVAILABLE vérifié à nouveau au cas où désactivé
        for r_idx, r in enumerate(all_results_list):
            progress_trail_action = 90 + ((r_idx + 1) / total_files) * 5 # 5% pour cette étape
            _progress(progress_trail_action)

            # On ne traite que les fichiers qui sont encore OK et qui n'ont pas été actionnés par SNR (si action immédiate)
            # OU les fichiers qui sont en attente d'action SNR (car ils doivent quand même être vérifiés pour les traînées)
            if r['status'] == 'ok' and \
               (r.get('rejected_reason') is None or r.get('rejected_reason') == 'low_snr_pending_action') and \
               r.get('action') not in ['moved_snr', 'deleted_snr', 'moved_trail', 'deleted_trail']: # Ne pas retraiter si déjà actionné
                
                current_path = r['path']
                if not current_path: # Si le chemin est None (ex: déjà supprimé par SNR immédiat - ne devrait pas arriver ici)
                    continue
                
                # Trouver les résultats de trail_module pour ce fichier
                abs_file_path_for_trail = os.path.abspath(current_path)
                found_key_trail = None
                # trail_results peut avoir des clés comme ('/abs/path/to/file.fits', 0)
                # ou pour les listes, directement le chemin '/abs/path/to/file.fits'
                if isinstance(input_for_trail_module, list): # Si trail_module a pris une liste
                    if abs_file_path_for_trail in trail_results:
                        found_key_trail = abs_file_path_for_trail
                else: # Si trail_module a pris un pattern
                    for key_tuple in trail_results.keys():
                        if isinstance(key_tuple, tuple) and len(key_tuple) == 2:
                            try:
                                res_abs_path = os.path.abspath(key_tuple[0])
                                if os.path.normcase(res_abs_path) == os.path.normcase(abs_file_path_for_trail) and key_tuple[1] == 0: # Extension 0
                                    found_key_trail = key_tuple
                                    break
                            except Exception: pass # Ignorer erreurs de normalisation de chemin

                if found_key_trail:
                    trail_segments = trail_results[found_key_trail]
                    if isinstance(trail_segments, (list, np.ndarray)) and len(trail_segments) > 0:
                        r['has_trails'] = True
                        r['num_trails'] = len(trail_segments)
                        _log("logic_info_prefix", text=f"Trail Rejet: {r['rel_path']} ({len(trail_segments)} segments)")

                        if apply_trail_action_immediately:
                            r['rejected_reason'] = 'trail'
                            action_to_take_trail = 'kept'
                            reject_dir_trail_option = options.get('trail_reject_dir')
                            if options.get('delete_rejected'):
                                action_to_take_trail = 'deleted_trail'
                            elif options.get('move_rejected') and reject_dir_trail_option and trail_reject_abs:
                                action_to_take_trail = 'moved_trail'

                            if action_to_take_trail != 'kept':
                                if os.path.exists(current_path):
                                    if action_to_take_trail == 'moved_trail':
                                        dest_path_trail = os.path.join(trail_reject_abs, os.path.basename(current_path))
                                        try:
                                            if os.path.normpath(current_path) != os.path.normpath(dest_path_trail):
                                                shutil.move(current_path, dest_path_trail)
                                                _log("logic_moved_info", folder=os.path.basename(trail_reject_abs), text_key_suffix="_trail", file_rel_path=r['rel_path'])
                                                r['path'] = dest_path_trail
                                                r['action'] = 'moved_trail'
                                            else:
                                                r['action_comment'] += " Déjà dans dossier cible Trail?"
                                                r['action'] = 'kept'
                                        except Exception as move_e_tr:
                                            _log("logic_move_error", file=r['rel_path'], e=move_e_tr)
                                            r['action_comment'] += f" Erreur déplacement Trail: {move_e_tr}"
                                            r['action'] = 'error_move'
                                            r['rejected_reason'] = None
                                    elif action_to_take_trail == 'deleted_trail':
                                        try:
                                            os.remove(current_path)
                                            _log("logic_info_prefix", text=f"Fichier supprimé (Trail): {r['rel_path']}")
                                            r['path'] = None
                                            r['action'] = 'deleted_trail'
                                        except Exception as del_e_tr:
                                            _log("logic_error_prefix", text=f"Erreur suppression Trail {r['rel_path']}: {del_e_tr}")
                                            r['action_comment'] += f" Erreur suppression Trail: {del_e_tr}"
                                            r['action'] = 'error_delete'
                                            r['rejected_reason'] = None
                                else:
                                    _log("logic_move_skipped", file=r['rel_path'], e="Fichier source non trouvé pour action Trail.")
                                    r['action_comment'] += " Ignoré action Trail (source non trouvée)."
                                    r['action'] = 'error_action'
                                    r['rejected_reason'] = None
                            else:
                                r['action'] = 'kept'
                                r['action_comment'] += " Rejeté (traînée) mais action=none."
                        else:
                            r['rejected_reason'] = 'trail_pending_action'
                            r['action'] = 'pending_trail_action'
                            r['action_comment'] += ' Action Trail différée.'
                    else: # Pas de traînées trouvées pour ce fichier
                        r['has_trails'] = False; r['num_trails'] = 0
                else:  # Pas de résultat de trail_module pour ce fichier, vérifier les erreurs
                    file_had_trail_error = False
                    if trail_errors:
                        # Chercher une erreur spécifique à ce fichier
                        err_key_to_check = None
                        if isinstance(input_for_trail_module, list): err_key_to_check = abs_file_path_for_trail
                        else: err_key_to_check = (abs_file_path_for_trail, 0) # Tuple pour les patterns

                        # Adapter la recherche d'erreur selon le type d'input_for_trail_module
                        error_message_for_file = None
                        if isinstance(input_for_trail_module, list):
                            error_message_for_file = trail_errors.get(err_key_to_check)
                        else: # Pattern
                            for key_tuple_err, msg_err in trail_errors.items():
                                if isinstance(key_tuple_err, tuple) and len(key_tuple_err) == 2:
                                    try:
                                        err_abs_path = os.path.abspath(key_tuple_err[0])
                                        if os.path.normcase(err_abs_path) == os.path.normcase(abs_file_path_for_trail) and key_tuple_err[1] == 0:
                                            error_message_for_file = msg_err
                                            break
                                    except: pass
                        
                        if error_message_for_file and "is not a valid science extension" not in str(error_message_for_file):
                            r['action_comment'] += f" Erreur détection trail ({str(error_message_for_file)[:50]}...). "
                            file_had_trail_error = True
                    if not file_had_trail_error:
                         r['has_trails'] = False; r['num_trails'] = 0 # Marquer comme non-traînée si pas d'erreur spécifique

    # --- Tri Bortle et organisation ---
    for r in all_results_list:
        r['mount'] = 'ALTZ'
        r['bortle'] = 'Unknown'
        r['filepath_dst'] = r.get('path')
        if r.get('status') == 'ok' and r.get('action') == 'kept' and r.get('path'):
            group = 'EQ' if str(r.get('eqmode', 2)) == '1' else 'ALTZ'
            r['mount'] = group
            lon = r.get('sitelong'); lat = r.get('sitelat')
            bortle_class = 'Unknown'
            if options.get('use_bortle') and bortle_dataset and lon is not None and lat is not None:
                try:
                    with bortle_lock:
                        l_ucd = bortle_utils.sample_bortle_dataset(bortle_dataset, float(lon), float(lat))
                    sqm = artif_ratio_to_sqm(float(l_ucd))
                    bortle_class = str(bortle_utils.sqm_to_bortle(float(sqm)))
                except Exception:
                    bortle_class = 'Unknown'
                    l_ucd = None
                    sqm = None
            else:
                l_ucd = None
                sqm = None
            r['bortle'] = bortle_class
            r['l_ucd_artif'] = l_ucd
            r['sqm'] = sqm
            tele = r.get('telescope') or 'Unknown'
            date_obs = r.get('date_obs')
            date_obj = None
            if date_obs:
                try:
                    date_obj = datetime.datetime.fromisoformat(str(date_obs).split('T')[0])
                except Exception:
                    pass
            filename_lower = os.path.basename(r['path']).lower()
            if 'mosaic' in filename_lower:
                parts = [output_root, 'mosaic', group]
            else:
                parts = [output_root, group]
            if options.get('use_bortle'):
                parts.append(f"Bortle_{bortle_class}")
            parts.append(tele)
            if date_obj:
                parts.append(date_obj.strftime('%Y-%m-%d'))
            parts.append(f"Filter_{r.get('filter') or 'Unknown'}")
            dest_dir = os.path.join(*filter(None, parts))
            dest_path = os.path.join(dest_dir, os.path.basename(r['path']))
            r['filepath_dst'] = dest_path
    

    # --- Étape 7: Création des fichiers marqueurs ---
    # ... (cette section reste identique) ...
    #      Progression de 95% à 98%
    _progress(95.0) # Avant de commencer les marqueurs
    _log("logic_info_prefix", text="Création des fichiers marqueurs...")
    analysis_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    num_processed_dirs = len(processed_directories)
    for dir_idx, directory in enumerate(processed_directories):
        marker_progress = 95 + ((dir_idx + 1) / num_processed_dirs if num_processed_dirs > 0 else 1) * 3 # 3% pour les marqueurs
        _progress(marker_progress)
        marker_file_path = os.path.join(directory, marker_filename)
        try:
            with open(marker_file_path, 'w', encoding='utf-8') as mf:
                mf.write(f"Analyse Astro Analyzer terminée pour ce dossier le: {analysis_datetime}\n")
            _log("logic_info_prefix", text=f"Marqueur créé: {os.path.relpath(marker_file_path, abs_input_dir)}")
        except IOError as e_marker:
            _log("logic_error_prefix", text=f"Impossible de créer le fichier marqueur dans {directory}: {e_marker}")
    _progress(98.0) # Après les marqueurs


    # --- Étape 8: Écrire les résultats détaillés FINALS dans le log ---
    try:
        with open(output_log, 'a', encoding='utf-8') as log_file:  # Mode 'a' pour ajouter au log existant
            log_file.write("\n--- Analyse individuelle des fichiers (État final après actions) ---\n")

            header_parts = ["Fichier (Relatif)", "Statut", "SNR", "Fond", "Bruit", "PixSig", "Starcount"]
            if options.get('detect_trails') and SATDET_AVAILABLE:
                header_parts.extend(["Traînée", "NbSeg"])
            header_parts.extend(["Expo", "Filtre", "Temp", "Monture", "Bortle", "Dest", "Action Finale", "Rejet", "Commentaire"])

            header = "\t".join(header_parts) + "\n"
            log_file.write(header)

            for r in all_results_list:
                log_line_parts = [
                    str(r.get('rel_path', '?')),
                    str(r.get('status', '?')),
                    f"{r.get('snr', np.nan):.2f}",
                    f"{r.get('sky_bg', np.nan):.2f}",
                    f"{r.get('sky_noise', np.nan):.2f}",
                    str(r.get('signal_pixels', 0)),
                    str(r.get('starcount', 'N/A'))
                ]
                if options.get('detect_trails') and SATDET_AVAILABLE:
                    trail_status = 'N/A'
                    if 'has_trails' in r:
                        trail_status = 'Oui' if r['has_trails'] else 'Non'
                    log_line_parts.extend([trail_status, str(r.get('num_trails', 0))])

                log_line_parts.extend([
                    str(r.get('exposure', 'N/A')),
                    str(r.get('filter', 'N/A')),
                    str(r.get('temperature', 'N/A')),
                    str(r.get('mount', '')),
                    str(r.get('bortle', '')),
                    str(r.get('filepath_dst', '')),
                    str(r.get('action', '?')),
                    str(r.get('rejected_reason') or ''),
                    (str(r.get('error_message') or '') + " " + str(r.get('action_comment') or '')).strip()
                ])
                log_line = "\t".join(log_line_parts) + "\n"
                log_file.write(log_line.replace('\tnan', '\tN/A').replace('\tN/A', '\t-'))

    except IOError as e: 
        # Utiliser le callback _log s'il est disponible, sinon print
        _log_func = _log if callable(_log) else print
        _log_func("logic_log_init_error", path=output_log, e=e) # Ou une clé d'erreur plus générique pour le log
    _progress(99.0)


    # --- Étape 9: Écriture du Résumé Final et Retour ---
    _progress(100); end_time = time.time(); duration = end_time - start_time
    write_log_summary(output_log, abs_input_dir, options, trail_analysis_config, trail_errors, all_results_list, selection_stats, skipped_marker_dirs_count)
    try:
        with open(output_log, 'a', encoding='utf-8') as log_file:
            log_file.write(f"\nDurée totale de l'analyse: {duration:.2f} secondes\n")
            log_file.write("="*80 + "\nFin du log.\n")
    except IOError: pass # Ignorer si erreur ici, le principal est déjà écrit

    csv_path = os.path.join(os.path.dirname(output_log), 'telescopes_pollution.csv')
    try:
        write_telescope_pollution_csv(csv_path, all_results_list, bortle_dataset if options.get('use_bortle') else None)
        _log('logic_info_prefix', text=f"CSV pollution écrit: {os.path.basename(csv_path)}")
    except Exception as csv_e:
        _log('logic_error_prefix', text=f"Erreur écriture CSV pollution: {csv_e}")

    if bortle_dataset:
        try:
            bortle_dataset.close()
        except Exception:
            pass

    _status("status_analysis_done") # Statut final générique
    return all_results_list
