# --- START OF FILE analyse_logic.py ---
# (Imports and write_log_summary remain the same)
import os
import glob
import shutil
import time
import datetime
import traceback
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
import warnings
import json

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
            
            # --- AJOUT : Section pour sauvegarder les données de visualisation ---
            if results_list is not None:
                log_file.write("\n" + "--- BEGIN VISUALIZATION DATA ---" + "\n")
                try:
                    # Remplacer les NaN par None pour une sérialisation JSON valide
                    # et convertir les booléens numpy en booléens python
                    def sanitize_for_json(obj):
                        if isinstance(obj, dict):
                            return {k: sanitize_for_json(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [sanitize_for_json(elem) for elem in obj]
                        elif isinstance(obj, (np.float32, np.float64)):
                            return float(obj) if np.isfinite(obj) else None
                        elif isinstance(obj, (np.int32, np.int64, np.int_)):
                            return int(obj)
                        elif isinstance(obj, np.bool_):
                            return bool(obj)
                        elif isinstance(obj, float) and not np.isfinite(obj): # Gérer NaN/inf float Python
                            return None
                        return obj

                    sanitized_results = sanitize_for_json(results_list)
                    json.dump(sanitized_results, log_file, indent=4) # indent pour lisibilité
                    log_file.write("\n" + "--- END VISUALIZATION DATA ---" + "\n")
                except Exception as e_json:
                    log_file.write(f"ERREUR: Impossible de sauvegarder les données de visualisation en JSON: {e_json}\n")
                    log_file.write("--- END VISUALIZATION DATA (ERROR) ---" + "\n")
            # --- FIN AJOUT ---

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

    # --- NOUVEAU : Extraire l'option pour l'application immédiate des actions SNR ---
    apply_snr_action_immediately = options.get('apply_snr_action_immediately', True)
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
    all_results_list = []; _log("logic_snr_start"); snr_loop_errors = 0
    for i, fits_file_path in enumerate(fits_files_to_process):
        progress = ((i + 1) / total_files) * 50 # SNR jusqu'à 50%
        try: rel_path_for_status = os.path.relpath(fits_file_path, abs_input_dir)
        except ValueError: rel_path_for_status = os.path.basename(fits_file_path)
        _progress(progress); _status("status_snr_start", file=rel_path_for_status, i=i+1, total=total_files)
        result = {'file': os.path.basename(fits_file_path), 'path': fits_file_path, 'rel_path': rel_path_for_status, 
                  'status': 'pending', 'action': 'kept', 'rejected_reason': None, 
                  'action_comment': '', 'error_message': '', 'has_trails': False, 'num_trails': 0}
        try:
            if options.get('analyze_snr'):
                exposure, filter_name, temperature = 'N/A', 'N/A', 'N/A'; snr, sky_bg, sky_noise, signal_pixels = np.nan, np.nan, np.nan, 0
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', AstropyWarning)
                    hdul = None
                    try:
                        hdul = fits.open(fits_file_path, memmap=False, lazy_load_hdus=True)
                        if hdul and len(hdul) > 0 and hasattr(hdul[0], 'data') and hdul[0].data is not None:
                            data = hdul[0].data; header = hdul[0].header
                            exposure = header.get('EXPTIME', header.get('EXPOSURE', 'N/A')); filter_name = header.get('FILTER', 'N/A'); temperature = header.get('CCD-TEMP', header.get('TEMPERAT', 'N/A'))
                            snr, sky_bg, sky_noise, signal_pixels = snr_module.calculate_snr(data)
                            if np.isfinite(snr) and np.isfinite(sky_bg) and np.isfinite(sky_noise):
                                snr_data = {'snr': snr, 'sky_bg': sky_bg, 'sky_noise': sky_noise, 'signal_pixels': signal_pixels, 'exposure': exposure, 'filter': filter_name, 'temperature': temperature}
                                result.update(snr_data); result['status'] = 'ok'; _log("logic_snr_info", file=result['rel_path'], snr=snr, bg=sky_bg)
                            else: raise ValueError("Calcul SNR a retourné des valeurs non finies.")
                        else: raise ValueError("Pas de données image valides dans HDU 0.")
                    except Exception as snr_e: result['status'] = 'error'; err_msg = f"Erreur analyse SNR/FITS: {snr_e}"; result['error_message'] = err_msg; _log("logic_file_error", file=result['rel_path'], e=err_msg); snr_loop_errors += 1
                    finally:
                        if hdul: 
                            try: hdul.close()
                            except Exception as e_close: print(f"WARN: Erreur fermeture FITS {result['rel_path']}: {e_close}")
            else: # Si pas d'analyse SNR, marquer comme OK pour la suite
                if result['status'] == 'pending': result['status'] = 'ok' 
        except Exception as file_e: result['status'] = 'error'; err_msg = f"Erreur traitement général: {file_e}"; result['error_message'] = err_msg; _log("logic_file_error", file=result['rel_path'], e=err_msg); snr_loop_errors += 1; print(f"\n--- Traceback Erreur Fichier (Logic) {result['rel_path']} ---"); traceback.print_exc(); print("---------------------------------------------------\n")
        finally: all_results_list.append(result)
    if snr_loop_errors > 0: _log("logic_warn_prefix", text=f"{snr_loop_errors} erreur(s) rencontrée(s) pendant l'analyse SNR initiale.")


    # --- Étape 3: Calcul Seuil SNR ---
    # ... (cette section reste identique) ...
    snr_threshold = -np.inf
    selection_stats = None
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
                else: # r['snr'] >= snr_threshold
                    kept_by_snr_initial += 1 
            # else: Fichier OK mais pas de filtre SNR actif, ou SNR non valide (process_for_trails reste True)
        
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
    trail_results = {}; trail_errors = {}; trail_analysis_config = None
    if options.get('detect_trails') and SATDET_AVAILABLE:
        if not TRAIL_MODULE_LOADED or not hasattr(trail_module, 'run_trail_detection'): 
            _log("logic_error_prefix", text="Erreur: trail_module n'est pas chargé ou manque run_trail_detection.")
            options['detect_trails'] = False # Désactiver pour la suite
        elif not files_kept_for_trails: # Si aucun fichier n'est éligible pour satdet
            _log("logic_info_prefix", text="Aucun fichier éligible pour la détection de traînées après filtre SNR.")
        else:
            # trail_module s'attend à un search_pattern. On va lui donner le dossier parent
            # et il filtrera en interne si besoin, ou on lui passe la liste des fichiers éligibles
            # si SATDET_ACCEPTS_LIST est True.
            input_for_trail_module = None
            if SATDET_ACCEPTS_LIST:
                _log("logic_info_prefix", text=f"Détection traînées sur {len(files_kept_for_trails)} fichiers spécifiques.")
                input_for_trail_module = files_kept_for_trails
            elif SATDET_USES_SEARCHPATTERN:
                # Utiliser le dossier parent principal si pas de liste acceptée
                # On pourrait essayer de construire un pattern plus complexe, mais pour l'instant on garde simple.
                # Le module trail devrait gérer les fichiers non-FITS ou les erreurs individuellement.
                # On prend le dossier du premier fichier éligible comme base, ou l'input_dir.
                search_base_dir = os.path.dirname(files_kept_for_trails[0]) if files_kept_for_trails else abs_input_dir
                # Essayer de trouver une extension commune
                common_ext = None
                if any(f.lower().endswith('.fit') for f in files_kept_for_trails): common_ext = "*.fit"
                elif any(f.lower().endswith('.fits') for f in files_kept_for_trails): common_ext = "*.fits"
                elif any(f.lower().endswith('.fts') for f in files_kept_for_trails): common_ext = "*.fts"
                
                if common_ext:
                    search_pattern_to_use = os.path.join(search_base_dir, common_ext)
                    # Si récursif, on pourrait utiliser un glob plus complexe, mais satdet ne le gère pas forcément.
                    # Pour l'instant, on se base sur le dossier d'entrée principal si récursif.
                    if include_subfolders and search_base_dir != abs_input_dir : # Si sous-dossier et récursif
                         # On pourrait devoir lancer satdet par sous-dossier ou lui donner une liste.
                         # Pour l'instant, on avertit et on utilise le dossier racine.
                         _log("logic_warn_prefix", text=f"Détection traînées en mode récursif utilise le dossier racine '{abs_input_dir}' pour satdet.")
                         search_pattern_to_use = os.path.join(abs_input_dir, common_ext)

                    input_for_trail_module = search_pattern_to_use
                else: # Si pas d'extension commune, on ne peut pas créer de pattern simple
                    _log("logic_warn_prefix", text="Aucune extension FITS commune trouvée pour le pattern satdet, détection traînées sur tous les fichiers du dossier.")
                    input_for_trail_module = os.path.join(abs_input_dir, "*.*") # Fallback large

            if input_for_trail_module:
                 trail_params = options.get('trail_params', {}); trail_analysis_config = trail_params.copy()
                 _progress('indeterminate') # Mettre la barre en mode indéterminé pour satdet
                 status_msg_trail = f"Lancement détection traînées ({'pattern' if isinstance(input_for_trail_module, str) else 'liste'})..."
                 _status("status_custom", text=status_msg_trail)
                 _log("logic_info_prefix", text=f"{status_msg_trail} avec entrée: {input_for_trail_module if isinstance(input_for_trail_module, str) else str(len(input_for_trail_module)) + ' fichiers'}")
                 
                 try: 
                     # Début de la progression pour satdet (de 55 à 90)
                     def satdet_status_callback(key, **kwargs):
                         if key == 'satdet_progress_file': # Supposons que trail_module envoie ce statut
                             file_idx = kwargs.get('current', 0)
                             total_s_files = kwargs.get('total', len(files_kept_for_trails) if files_kept_for_trails else 1)
                             satdet_prog = 55 + ( (file_idx / total_s_files) * 35 ) # 35% de la barre totale
                             _progress(min(satdet_prog, 90.0)) # Capper à 90%
                         elif key == 'status_satdet_wait': # Le trail_module peut envoyer ça
                             _progress('indeterminate')
                         elif key == 'status_satdet_done': # Le trail_module peut envoyer ça
                             _progress(90.0) # Fin de satdet
                         _status(key, **kwargs) # Relayer les autres statuts

                     trail_results, trail_errors = trail_module.run_trail_detection(input_for_trail_module, trail_params, status_callback=satdet_status_callback, log_callback=_log)
                 except Exception as trail_e: 
                     _log("logic_error_prefix", text=f"Erreur lors de l'appel à trail_module.run_trail_detection: {trail_e}"); traceback.print_exc()
                     trail_errors[('FATAL_CALL_ERROR', 0)] = str(trail_e)
                 
                 _progress(90.0) # Assurer que la progression est à 90% après satdet
                 if ('DEPENDENCY_ERROR', 0) in trail_errors or ('FATAL_ERROR', 0) in trail_errors or \
                    ('IMPORT_ERROR', 0) in trail_errors or ('FATAL_CALL_ERROR', 0) in trail_errors or \
                    ('CONFIG_ERROR', 0) in trail_errors:
                     _log("logic_error_prefix", text="Échec critique détection traînées. Arrêt de l'analyse."); 
                     write_log_summary(output_log, abs_input_dir, options, trail_analysis_config, trail_errors, all_results_list, selection_stats, skipped_marker_dirs_count)
                     # Création marqueur même en cas d'erreur trail critique
                     analysis_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                     for directory in processed_directories:
                        marker_file_path = os.path.join(directory, marker_filename)
                        try:
                            with open(marker_file_path, 'w', encoding='utf-8') as mf:
                                mf.write(f"Analyse Astro Analyzer (partielle - erreur trail) terminée pour ce dossier le: {analysis_datetime}\n")
                            _log("logic_info_prefix", text=f"Marqueur créé (erreur trail): {os.path.relpath(marker_file_path, abs_input_dir)}")
                        except IOError as e_marker: _log("logic_error_prefix", text=f"Impossible de créer le marqueur (erreur trail) dans {directory}: {e_marker}")
                     return [] # Arrêter l'analyse ici
            else: 
                _log("logic_no_fits_satdet", path=abs_input_dir); _status("status_satdet_no_file")
                _progress(90.0) # Si pas d'analyse trail, sauter à 90%
    else: # Si pas de détection de traînées activée
        _progress(90.0) # Sauter la progression de satdet


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
                        # Si le fichier était en attente pour SNR, le rejet trail prend priorité
                        r['rejected_reason'] = 'trail' 
                        _log("logic_info_prefix", text=f"Trail Rejet: {r['rel_path']} ({len(trail_segments)} segments)")
                        
                        action_to_take_trail = 'kept'
                        reject_dir_trail_option = options.get('trail_reject_dir')
                        if options.get('delete_rejected'): action_to_take_trail = 'deleted_trail'
                        elif options.get('move_rejected') and reject_dir_trail_option and trail_reject_abs: action_to_take_trail = 'moved_trail'
                        
                        if action_to_take_trail != 'kept':
                            if os.path.exists(current_path): # Vérifier à nouveau si le fichier existe
                                if action_to_take_trail == 'moved_trail':
                                    dest_path_trail = os.path.join(trail_reject_abs, os.path.basename(current_path))
                                    try:
                                        if os.path.normpath(current_path) != os.path.normpath(dest_path_trail):
                                            shutil.move(current_path, dest_path_trail)
                                            _log("logic_moved_info", folder=os.path.basename(trail_reject_abs), text_key_suffix="_trail", file_rel_path=r['rel_path'])
                                            r['path'] = dest_path_trail; r['action'] = 'moved_trail'
                                        else: r['action_comment'] += " Déjà dans dossier cible Trail?"; r['action'] = 'kept'
                                    except Exception as move_e_tr:
                                        _log("logic_move_error", file=r['rel_path'], e=move_e_tr)
                                        r['action_comment'] += f" Erreur déplacement Trail: {move_e_tr}"; r['action'] = 'error_move'; r['rejected_reason'] = None 
                                elif action_to_take_trail == 'deleted_trail':
                                    try:
                                        os.remove(current_path)
                                        _log("logic_info_prefix", text=f"Fichier supprimé (Trail): {r['rel_path']}")
                                        r['path'] = None; r['action'] = 'deleted_trail'
                                    except Exception as del_e_tr:
                                        _log("logic_error_prefix", text=f"Erreur suppression Trail {r['rel_path']}: {del_e_tr}")
                                        r['action_comment'] += f" Erreur suppression Trail: {del_e_tr}"; r['action'] = 'error_delete'; r['rejected_reason'] = None
                            else:
                                _log("logic_move_skipped", file=r['rel_path'], e="Fichier source non trouvé pour action Trail.")
                                r['action_comment'] += " Ignoré action Trail (source non trouvée)."; r['action'] = 'error_action'; r['rejected_reason'] = None
                        else: # action_to_take_trail == 'kept'
                            r['action'] = 'kept' # Même si raison rejet = trail
                            r['action_comment'] += " Rejeté (traînée) mais action=none."
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
         with open(output_log, 'a', encoding='utf-8') as log_file: # Mode 'a' pour ajouter au log existant
            log_file.write("\n--- Analyse individuelle des fichiers (État final après actions) ---\n")
            header_parts = ["Fichier (Relatif)", "Statut", "SNR", "Fond", "Bruit", "PixSig"]
            if options.get('detect_trails') and SATDET_AVAILABLE: header_parts.extend(["Traînée", "NbSeg"])
            header_parts.extend(["Expo", "Filtre", "Temp", "Action Finale", "Rejet", "Commentaire"])
            header = "\t".join(header_parts) + "\n"; log_file.write(header)
            
            for r in all_results_list:
                 log_line_parts = [
                     str(r.get('rel_path','?')),
                     str(r.get('status','?')),
                     f"{r.get('snr', np.nan):.2f}",
                     f"{r.get('sky_bg', np.nan):.2f}",
                     f"{r.get('sky_noise', np.nan):.2f}",
                     str(r.get('signal_pixels',0))
                 ]
                 if options.get('detect_trails') and SATDET_AVAILABLE:
                     trail_status = 'N/A'
                     if 'has_trails' in r: 
                         # --- CORRECTION ICI ---
                         trail_status = 'Oui' if r['has_trails'] else 'Non' 
                         # --- FIN CORRECTION ---
                     log_line_parts.extend([trail_status, str(r.get('num_trails',0))])
                 
                 log_line_parts.extend([
                     str(r.get('exposure','N/A')),
                     str(r.get('filter','N/A')),
                     str(r.get('temperature','N/A')),
                     str(r.get('action','?')),
                     str(r.get('rejected_reason') or ''),
                     (str(r.get('error_message') or '') + " " + str(r.get('action_comment') or '')).strip()
                 ])
                 log_line = "\t".join(log_line_parts) + "\n"
                 log_file.write(log_line.replace('\tnan','\tN/A').replace('\tN/A','\t-'))
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
    
    _status("status_analysis_done") # Statut final générique
    return all_results_list
