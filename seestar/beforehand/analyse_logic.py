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

# --- Fonctions d'aide ---
def write_log_summary(log_file_path, input_dir, options,
                      analysis_config=None,
                      sat_errors=None, results_list=None,
                      selection_stats=None,
                      skipped_dirs_count=0): # NOUVEAU paramètre
    """Écrit le résumé dans le fichier log."""
    try:
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write("\n" + "="*80 + "\n")
            log_file.write("Résumé de l'analyse:\n")
            log_file.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Dossier analysé: {input_dir}\n")
            log_file.write(f"Inclure sous-dossiers: {'Oui' if options.get('include_subfolders') else 'Non'}\n")
            # --- NOUVEAU: Log dossiers skippés ---
            log_file.write(f"Dossiers ignorés (marqueur trouvé): {skipped_dirs_count}\n")
            # --- FIN NOUVEAU ---
            log_file.write(f"Analyse SNR effectuée: {'Oui' if options.get('analyze_snr') else 'Non'}\n")
            log_file.write(f"Détection Traînées effectuée: {'Oui' if options.get('detect_trails') and SATDET_AVAILABLE else 'Non'}\n")

            # ... (le reste de la fonction write_log_summary reste identique à la version précédente) ...
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
            if results_list is None: log_file.write("Aucune analyse individuelle de fichier effectuée.\n")
            else:
                total_processed = len(results_list); analyzed_count = sum(1 for r in results_list if r.get('status') != 'pending'); errors_count = sum(1 for r in results_list if r.get('status') == 'error')
                rejected_trails = sum(1 for r in results_list if r.get('rejected_reason') == 'trail'); rejected_low_snr = sum(1 for r in results_list if r.get('rejected_reason') == 'low_snr'); kept_count = sum(1 for r in results_list if r.get('status') == 'ok' and r.get('rejected_reason') is None)
                moved_trails = sum(1 for r in results_list if r.get('action') == 'moved_trail'); moved_low_snr = sum(1 for r in results_list if r.get('action') == 'moved_snr'); deleted_trails = sum(1 for r in results_list if r.get('action') == 'deleted_trail'); deleted_low_snr = sum(1 for r in results_list if r.get('action') == 'deleted_snr')
                log_file.write(f"Nombre total de fichiers FITS trouvés (hors dossiers ignorés): {total_processed}\n"); log_file.write(f"  Fichiers analysés (ou tentative): {analyzed_count}\n"); log_file.write(f"  Images conservées dans le dossier source/sous-dossiers: {kept_count}\n"); log_file.write(f"  Images marquées pour rejet (traînées): {rejected_trails}\n"); log_file.write(f"  Images marquées pour rejet (faible SNR): {rejected_low_snr}\n"); log_file.write(f"  Erreurs d'analyse fichier: {errors_count}\n")
                snr_reject_path_base = options.get('snr_reject_dir','N/A'); trail_reject_path_base = options.get('trail_reject_dir','N/A')
                if options.get('move_rejected'): log_file.write(f"Actions (Déplacement activé):\n"); log_file.write(f"  Déplacées vers '{os.path.basename(trail_reject_path_base)}' (traînées): {moved_trails}\n"); log_file.write(f"  Déplacées vers '{os.path.basename(snr_reject_path_base)}' (faible SNR): {moved_low_snr}\n")
                elif options.get('delete_rejected'): log_file.write(f"Actions (Suppression activée):\n"); log_file.write(f"  Supprimées (traînées): {deleted_trails}\n"); log_file.write(f"  Supprimées (faible SNR): {deleted_low_snr}\n")
                else: log_file.write(f"Actions: Aucune (fichiers rejetés non déplacés/supprimés)\n")
                if options.get('analyze_snr'):
                     all_valid_snrs = [r['snr'] for r in results_list if r.get('status') == 'ok' and 'snr' in r and np.isfinite(r['snr'])]
                     if all_valid_snrs: log_file.write(f"Statistiques SNR (sur {len(all_valid_snrs)} images valides):\n"); mean_snr = np.mean(all_valid_snrs); median_snr = np.median(all_valid_snrs); min_snr = min(all_valid_snrs); max_snr = max(all_valid_snrs); log_file.write(f"  Moy: {mean_snr:.2f}, Med: {median_snr:.2f}, Min: {min_snr:.2f}, Max: {max_snr:.2f}\n")
                     else: log_file.write("Statistiques SNR: Aucune donnée SNR valide calculée.\n")
    except Exception as e:
        print(f"ERREUR CRITIQUE lors de l'écriture du résumé du log ({log_file_path}): {e}"); traceback.print_exc()
        try:
            with open(log_file_path, 'a', encoding='utf-8') as log_file: log_file.write(f"\nERREUR CRITIQUE lors de l'écriture de ce résumé: {e}\n{traceback.format_exc()}");
        except Exception: pass


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

    # --- Validation chemins & Création dossiers ---
    # (Identique à la version précédente)
    if not input_dir or not os.path.isdir(input_dir): _log("logic_error_prefix", clear=True, text=f"Dossier d'entrée invalide: {input_dir}"); _status("status_dir_create_error", e=f"Input folder invalid: {input_dir}"); return []
    if not output_log: _log("logic_error_prefix", clear=True, text="Fichier log non spécifié."); _status("msg_log_file_missing"); return []
    abs_input_dir = os.path.abspath(input_dir)
    reject_dirs_to_exclude_abs = []; snr_reject_abs = None; trail_reject_abs = None
    if options.get('move_rejected', False):
        snr_reject_rel = options.get('snr_reject_dir'); trail_reject_rel = options.get('trail_reject_dir')
        if options.get('analyze_snr') and options.get('snr_selection_mode') != 'none':
            if not snr_reject_rel: _log("logic_error_prefix", clear=True, text="Chemin dossier rejet SNR non spécifié."); return []
            snr_reject_abs = os.path.abspath(snr_reject_rel); reject_dirs_to_exclude_abs.append(snr_reject_abs)
            if not os.path.exists(snr_reject_abs):
                try: os.makedirs(snr_reject_abs); _log("logic_dir_created", path=snr_reject_abs)
                except OSError as e: _log("logic_dir_create_error", path=snr_reject_abs, e=e); _status("status_dir_create_error", e=e); return []
            elif not os.path.isdir(snr_reject_abs): _log("logic_error_prefix", text=f"Chemin rejet SNR n'est pas un dossier: {snr_reject_abs}"); _status("status_dir_create_error", e="SNR Reject path is not a directory"); return []
        if options.get('detect_trails') and SATDET_AVAILABLE:
            if not trail_reject_rel: _log("logic_error_prefix", clear=True, text="Chemin dossier rejet Trail non spécifié."); return []
            trail_reject_abs = os.path.abspath(trail_reject_rel); reject_dirs_to_exclude_abs.append(trail_reject_abs)
            if not os.path.exists(trail_reject_abs):
                try: os.makedirs(trail_reject_abs); _log("logic_dir_created", path=trail_reject_abs)
                except OSError as e: _log("logic_dir_create_error", path=trail_reject_abs, e=e); _status("status_dir_create_error", e=e); return []
            elif not os.path.isdir(trail_reject_abs): _log("logic_error_prefix", text=f"Chemin rejet Trail n'est pas un dossier: {trail_reject_abs}"); _status("status_dir_create_error", e="Trail Reject path is not a directory"); return []

    # Initialiser le log
    try:
        with open(output_log, 'w', encoding='utf-8') as f:
             f.write(f"Début de l'analyse: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"); f.write(f"Dossier d'entrée: {abs_input_dir}\n"); f.write("Options d'analyse:\n")
             for key, value in options.items():
                 if key != 'trail_params' and not callable(value): f.write(f"  {key}: {value}\n")
                 elif key == 'trail_params': f.write(f"  trail_params: {value}\n")
             f.write("="*80 + "\n")
    except IOError as e: _log("logic_log_init_error", clear=True, path=output_log, e=e); _status("status_log_error"); return []

    # --- Étape 1: Découverte des fichiers FITS (avec skip marqueur) ---
    _progress(0.0); _status("status_discovery_start")
    fits_files_to_process = []
    processed_directories = set() # NOUVEAU: Garder trace des dossiers où on a traité des fichiers
    skipped_marker_dirs_count = 0 # NOUVEAU: Compter les dossiers skippés par marqueur
    include_subfolders = options.get('include_subfolders', False)
    fits_extensions = ('.fit', '.fits', '.fts')
    marker_filename = ".astro_analyzer_run_complete" # NOUVEAU: Nom du fichier marqueur

    _log("logic_info_prefix", text=f"Recherche de fichiers FITS dans {abs_input_dir} (Récursion: {include_subfolders})...")
    try:
        for dirpath, dirnames, filenames in os.walk(abs_input_dir, topdown=True):
            current_dir_abs = os.path.abspath(dirpath)

            # --- Logique d'exclusion des sous-dossiers de REJET ---
            dirs_to_remove = [d for d in dirnames if os.path.abspath(os.path.join(current_dir_abs, d)) in reject_dirs_to_exclude_abs]
            if dirs_to_remove:
                for dname in dirs_to_remove:
                    _log("logic_info_prefix", text=f"Exclusion du sous-dossier de rejet: {os.path.relpath(os.path.join(current_dir_abs, dname), abs_input_dir)}")
                    dirnames.remove(dname) # Modifie dirnames in-place pour os.walk

            # --- NOUVEAU: Logique de skip par MARQUEUR ---
            marker_file_path = os.path.join(current_dir_abs, marker_filename)
            if os.path.exists(marker_file_path):
                _log("logic_info_prefix", text=f"Ignoré (marqueur trouvé): {os.path.relpath(current_dir_abs, abs_input_dir)}")
                skipped_marker_dirs_count += 1
                dirnames[:] = [] # Ne pas descendre dans les sous-dossiers de ce dossier marqué
                continue # Passer au prochain dossier dans os.walk
            # --- FIN NOUVEAU ---

            # Ajouter les fichiers FITS du dossier courant (dirpath)
            found_in_this_dir = False
            for filename in filenames:
                if filename.lower().endswith(fits_extensions):
                    full_path = os.path.join(current_dir_abs, filename)
                    fits_files_to_process.append(full_path)
                    found_in_this_dir = True

            # Si on a trouvé des fichiers FITS à traiter dans ce dossier, on le note
            if found_in_this_dir:
                processed_directories.add(current_dir_abs)

            # Arrêt si non récursif
            if not include_subfolders and dirpath == abs_input_dir: # S'assurer qu'on est au niveau racine
                dirnames[:] = []

    except OSError as e: _log("logic_error_prefix", text=f"Erreur lors du parcours du dossier {abs_input_dir}: {e}"); _status("status_dir_create_error", e=f"Error walking directory: {e}"); write_log_summary(output_log, abs_input_dir, options, None, None, [], None, skipped_marker_dirs_count); return []

    fits_files_to_process = sorted(list(set(fits_files_to_process)))
    total_files = len(fits_files_to_process)

    if total_files == 0:
        _log("logic_no_fits_snr"); _status("status_analysis_done_no_valid")
        write_log_summary(output_log, abs_input_dir, options, None, None, [], None, skipped_marker_dirs_count)
        return []
    _log("logic_info_prefix", text=f"{total_files} fichiers FITS trouvés pour analyse (hors dossiers ignorés).")

    # --- Étape 2: Boucle Analyse SNR ---
    # (Identique à la version précédente, utilise fits_files_to_process)
    all_results_list = []; _log("logic_snr_start"); snr_loop_errors = 0
    for i, fits_file_path in enumerate(fits_files_to_process):
        progress = ((i + 1) / total_files) * 50
        try: rel_path_for_status = os.path.relpath(fits_file_path, abs_input_dir)
        except ValueError: rel_path_for_status = os.path.basename(fits_file_path)
        _progress(progress); _status("status_snr_start", file=rel_path_for_status, i=i+1, total=total_files)
        result = {'file': os.path.basename(fits_file_path), 'path': fits_file_path, 'rel_path': rel_path_for_status, 'status': 'pending', 'action': 'kept', 'rejected_reason': None, 'action_comment': '', 'error_message': '', 'has_trails': False, 'num_trails': 0}
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
            else:
                if result['status'] == 'pending': result['status'] = 'ok'
        except Exception as file_e: result['status'] = 'error'; err_msg = f"Erreur traitement général: {file_e}"; result['error_message'] = err_msg; _log("logic_file_error", file=result['rel_path'], e=err_msg); snr_loop_errors += 1; print(f"\n--- Traceback Erreur Fichier (Logic) {result['rel_path']} ---"); traceback.print_exc(); print("---------------------------------------------------\n")
        finally: all_results_list.append(result)
    if snr_loop_errors > 0: _log("logic_warn_prefix", text=f"{snr_loop_errors} erreur(s) rencontrée(s) pendant l'analyse SNR initiale.")

# --- Étape 3: Calcul Seuil SNR (Identique à la version précédente) ---
    snr_threshold = -np.inf
    selection_stats = None
    if options.get('analyze_snr') and options.get('snr_selection_mode') != 'none':
        mode = options.get('snr_selection_mode')
        value_str = options.get('snr_selection_value')
        selection_stats = {'mode': mode, 'value': value_str, 'threshold': None, 'initial_count': 0,
                           'selected_count': 0, 'rejected_count': 0}
        eligible_snr_results = [(r['snr'], idx) for idx, r in enumerate(all_results_list)
                                if r['status'] == 'ok' and 'snr' in r and np.isfinite(r['snr'])]
        selection_stats['initial_count'] = len(eligible_snr_results)
        if selection_stats['initial_count'] > 0:
            eligible_snrs = [res[0] for res in eligible_snr_results]
            local_snr_threshold = -np.inf
            if mode == 'threshold':
                try:
                    local_snr_threshold = float(value_str)
                    selection_stats['threshold'] = local_snr_threshold
                except (ValueError, TypeError):
                    _log("logic_warn_prefix", text=f"Seuil SNR invalide: '{value_str}'.")
            elif mode == 'percent':
                try:
                    percentile_to_keep = float(value_str)
                    assert 0 < percentile_to_keep <= 100
                    percentile_val = 100.0 - percentile_to_keep
                    local_snr_threshold = np.nanpercentile(eligible_snrs, percentile_val)
                    selection_stats['threshold'] = local_snr_threshold
                except Exception as e:
                    _log("logic_warn_prefix", text=f"Valeur pourcentage invalide: '{value_str}' ({e}).")
            elif mode == 'none':
                selection_stats['threshold'] = None
            snr_threshold = local_snr_threshold


    # --- Étape 4: Rejet SNR et Actions Associées ---
    # (Identique à la version précédente, utilise snr_reject_abs)
    _log("logic_info_prefix", text="Application du rejet SNR et actions...")
    kept_by_snr_initial = 0; rejected_by_snr_initial = 0; files_kept_for_trails = []
    snr_filter_active = options.get('analyze_snr') and options.get('snr_selection_mode') != 'none' and np.isfinite(snr_threshold) and snr_threshold > -np.inf
    if snr_filter_active: _log("logic_info_prefix", text=f"Seuil SNR appliqué: {snr_threshold:.3f}")
    for r in all_results_list:
        process_for_trails = True
        if r['status'] == 'ok' and r.get('rejected_reason') is None:
            if snr_filter_active and 'snr' in r and np.isfinite(r['snr']):
                if r['snr'] < snr_threshold:
                    r['rejected_reason'] = 'low_snr'; rejected_by_snr_initial += 1
                    action_to_take = 'kept'; reject_dir = options.get('snr_reject_dir') # Relative path for option check
                    if options.get('delete_rejected'): action_to_take = 'deleted_snr'
                    elif options.get('move_rejected') and reject_dir: action_to_take = 'moved_snr'
                    if action_to_take != 'kept':
                        current_path = r['path']
                        if current_path and os.path.exists(current_path):
                            process_for_trails = False
                            if action_to_take == 'moved_snr':
                                dest_path = os.path.join(snr_reject_abs, os.path.basename(current_path)) # Use absolute path for destination
                                try:
                                     if os.path.normpath(current_path) != os.path.normpath(dest_path): shutil.move(current_path, dest_path); _log("logic_moved_info", folder=os.path.basename(snr_reject_abs)); r['path'] = dest_path; r['action'] = 'moved_snr'
                                     else: r['action_comment'] += " Déjà dans dossier cible?"; r['action'] = 'kept'; process_for_trails = True
                                except Exception as move_e: _log("logic_move_error", file=r['rel_path'], e=move_e); r['action_comment'] += f" Erreur déplacement SNR: {move_e}"; r['action'] = 'error_move'; r['rejected_reason'] = None; process_for_trails = True
                            elif action_to_take == 'deleted_snr':
                                try: os.remove(current_path); _log("logic_info_prefix", text=f"Fichier supprimé (SNR): {r['rel_path']}"); r['path'] = None; r['action'] = 'deleted_snr'
                                except Exception as del_e: _log("logic_error_prefix", text=f"Erreur suppression SNR {r['rel_path']}: {del_e}"); r['action_comment'] += f" Erreur suppression SNR: {del_e}"; r['action'] = 'error_delete'; r['rejected_reason'] = None; process_for_trails = True
                        else: _log("logic_move_skipped", file=r['rel_path']); r['action_comment'] += " Ignoré action SNR (non trouvé source)."; r['action'] = 'error'; r['rejected_reason'] = None; process_for_trails = True
                    else: r['action'] = 'kept'; r['action_comment'] += " Rejeté (faible SNR) mais action=none."; process_for_trails = False
                else: kept_by_snr_initial += 1 # Passed SNR or SNR invalid
            # else: File OK but no SNR filter applied
        # Add to trail list if applicable
        if r['status'] == 'ok' and r.get('rejected_reason') is None and r['path'] is not None and process_for_trails: files_kept_for_trails.append(r['path'])
    # Update stats
    if selection_stats: selection_stats['selected_count'] = kept_by_snr_initial; selection_stats['rejected_count'] = rejected_by_snr_initial

    # --- Étape 5: Détection des traînées ---
    # (Identique à la version précédente - utilise search_pattern basé sur input_dir)
    trail_results = {}; trail_errors = {}; trail_analysis_config = None
    if options.get('detect_trails') and SATDET_AVAILABLE:
        if not TRAIL_MODULE_LOADED or not hasattr(trail_module, 'run_trail_detection'): _log("logic_error_prefix", text="Erreur: trail_module n'est pas chargé ou manque run_trail_detection."); options['detect_trails'] = False
        else:
            # Find search pattern based on files currently in input_dir (and subdirs if recursive)
            fits_files_for_trails_pattern = []
            try:
                for dirpath, dirnames, filenames in os.walk(abs_input_dir, topdown=True):
                    current_dir_abs = os.path.abspath(dirpath)
                    dirs_to_remove = [d for d in dirnames if os.path.abspath(os.path.join(current_dir_abs, d)) in reject_dirs_to_exclude_abs]
                    for dname in dirs_to_remove: dirnames.remove(dname)
                    for filename in filenames:
                        if filename.lower().endswith(fits_extensions): fits_files_for_trails_pattern.append(os.path.join(current_dir_abs, filename))
                    if not include_subfolders: dirnames[:] = []
            except OSError as e: _log("logic_error_prefix", text=f"Erreur re-parcours dossier {abs_input_dir} avant SatDet: {e}"); fits_files_for_trails_pattern = []

            search_pattern_to_use = None
            top_level_files = [f for f in fits_files_for_trails_pattern if os.path.dirname(f) == abs_input_dir]
            if not top_level_files: top_level_files = fits_files_for_trails_pattern
            if any(f.lower().endswith('.fit') for f in top_level_files): search_pattern_to_use = os.path.join(abs_input_dir, "*.fit")
            if any(f.lower().endswith('.fits') for f in top_level_files): search_pattern_to_use = os.path.join(abs_input_dir, "*.fits")
            if any(f.lower().endswith('.fts') for f in top_level_files): search_pattern_to_use = os.path.join(abs_input_dir, "*.fts")

            if search_pattern_to_use:
                 trail_params = options.get('trail_params', {}); trail_analysis_config = trail_params.copy()
                 _progress('indeterminate'); _log("logic_info_prefix", text=f"Lancement détection traînées avec pattern: {search_pattern_to_use}")
                 try: trail_results, trail_errors = trail_module.run_trail_detection(search_pattern_to_use, trail_params, status_callback=_status, log_callback=_log)
                 except Exception as trail_e: _log("logic_error_prefix", text=f"Erreur lors de l'appel à trail_module.run_trail_detection: {trail_e}"); traceback.print_exc(); trail_errors[('FATAL_CALL_ERROR', 0)] = str(trail_e)
                 _progress(90.0)
                 if ('DEPENDENCY_ERROR', 0) in trail_errors or ('FATAL_ERROR', 0) in trail_errors or ('IMPORT_ERROR', 0) in trail_errors or ('FATAL_CALL_ERROR', 0) in trail_errors: _log("logic_error_prefix", text="Échec critique détection traînées. Arrêt de l'analyse."); write_log_summary(output_log, abs_input_dir, options, trail_analysis_config, trail_errors, all_results_list, selection_stats); return []
            else: _log("logic_no_fits_satdet", path=abs_input_dir); _status("status_satdet_no_file"); _progress(90.0)
    if not options.get('detect_trails'): _progress(90.0)
# --- Étape 6: Rejet Traînées et Actions Associées ---
    # (Identique à la version précédente, utilise trail_reject_abs)
    _log("logic_info_prefix", text="Application du rejet Traînées et actions...")
    if options.get('detect_trails') and SATDET_AVAILABLE:
        for r in all_results_list:
            if r['status'] == 'ok' and r.get('rejected_reason') is None and r.get('action') == 'kept':
                current_path = r['path']
                if not current_path:
                    continue
                abs_file_path = os.path.abspath(current_path)
                found_key = None
                for key_tuple in trail_results.keys():
                    if isinstance(key_tuple, tuple) and len(key_tuple) == 2:
                        try:
                            res_abs_path = os.path.abspath(key_tuple[0])
                            if os.path.normcase(res_abs_path) == os.path.normcase(abs_file_path) and key_tuple[1] == 0:
                                found_key = key_tuple
                                break
                        except Exception:
                            pass
                if found_key:
                    trail_segments = trail_results[found_key]
                    if isinstance(trail_segments, (list, np.ndarray)) and len(trail_segments) > 0:
                        r['has_trails'] = True
                        r['num_trails'] = len(trail_segments)
                        r['rejected_reason'] = 'trail'
                        action_to_take = 'kept'
                        reject_dir = options.get('trail_reject_dir')
                        if options.get('delete_rejected'):
                            action_to_take = 'deleted_trail'
                        elif options.get('move_rejected') and reject_dir:
                            action_to_take = 'moved_trail'
                        if action_to_take != 'kept':
                            if os.path.exists(current_path):
                                if action_to_take == 'moved_trail':
                                    dest_path = os.path.join(trail_reject_abs, os.path.basename(current_path))  # Use absolute path
                                    try:
                                        if os.path.normpath(current_path) != os.path.normpath(dest_path):
                                            shutil.move(current_path, dest_path)
                                            _log("logic_moved_info", folder=os.path.basename(trail_reject_abs))
                                            r['path'] = dest_path
                                            r['action'] = 'moved_trail'
                                        else:
                                            r['action_comment'] += " Déjà dans dossier cible?"
                                            r['action'] = 'kept'
                                    except Exception as move_e:
                                        _log("logic_move_error", file=r['rel_path'], e=move_e)
                                        r['action_comment'] += f" Erreur déplacement Trail: {move_e}"
                                        r['action'] = 'error_move'
                                        r['rejected_reason'] = None
                                elif action_to_take == 'deleted_trail':
                                    try:
                                        os.remove(current_path)
                                        _log("logic_info_prefix", text=f"Fichier supprimé (Trail): {r['rel_path']}")
                                        r['path'] = None
                                        r['action'] = 'deleted_trail'
                                    except Exception as del_e:
                                        _log("logic_error_prefix", text=f"Erreur suppression Trail {r['rel_path']}: {del_e}")
                                        r['action_comment'] += f" Erreur suppression Trail: {del_e}"
                                        r['action'] = 'error_delete'
                                        r['rejected_reason'] = None
                            else:
                                _log("logic_move_skipped", file=r['rel_path'])
                                r['action_comment'] += " Ignoré action Trail (non trouvé source)."
                                r['action'] = 'error'
                                r['rejected_reason'] = None
                        else:
                            r['action'] = 'kept'
                            r['action_comment'] += " Rejeté (traînée) mais action=none."
                    else:
                        r['has_trails'] = False
                        r['num_trails'] = 0
                else:  # Check errors
                    file_had_error = False
                    if trail_errors:
                        for key_tuple, msg in trail_errors.items():
                            if isinstance(key_tuple, tuple) and len(key_tuple) == 2:
                                try:
                                    err_abs_path = os.path.abspath(key_tuple[0])
                                    if os.path.normcase(err_abs_path) == os.path.normcase(abs_file_path) and key_tuple[1] == 0:
                                        if "is not a valid science extension" not in str(msg):
                                            r['action_comment'] += f" Erreur détection trail ({msg[:50]}...). "
                                            file_had_error = True
                                            break
                                except Exception:
                                    pass
    # --- NOUVEAU: Étape 7: Création des fichiers marqueurs ---
    _log("logic_info_prefix", text="Création des fichiers marqueurs...")
    analysis_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for directory in processed_directories:
        marker_file_path = os.path.join(directory, marker_filename)
        try:
            with open(marker_file_path, 'w', encoding='utf-8') as mf:
                mf.write(f"Analyse Astro Analyzer terminée pour ce dossier le: {analysis_datetime}\n")
                # Optionnel: Ajouter plus d'infos comme les paramètres utilisés
                # mf.write(f"Options: {options}\n")
            _log("logic_info_prefix", text=f"Marqueur créé: {os.path.relpath(marker_file_path, abs_input_dir)}")
        except IOError as e_marker:
            _log("logic_error_prefix", text=f"Impossible de créer le fichier marqueur dans {directory}: {e_marker}")
            # Continuer même si un marqueur ne peut être créé

    # --- Étape 8: Écrire les résultats détaillés FINALS dans le log ---
    # (Identique à la version précédente)
    try:
         with open(output_log, 'a', encoding='utf-8') as log_file:
            log_file.write("\n--- Analyse individuelle des fichiers (État final après actions) ---\n")
            header = "Fichier (Relatif)\tStatut\tSNR\tFond\tBruit\tPixSig";
            if options.get('detect_trails') and SATDET_AVAILABLE: header += "\tTraînée\tNbSeg"
            header += "\tExpo\tFiltre\tTemp\tAction Finale\tRejet\tCommentaire\n"; log_file.write(header)
            for r in all_results_list:
                 log_line = f"{r.get('rel_path','?')}\t{r.get('status','?')}\t"; log_line += f"{r.get('snr', np.nan):.2f}\t{r.get('sky_bg', np.nan):.2f}\t{r.get('sky_noise', np.nan):.2f}\t{r.get('signal_pixels',0)}\t"
                 if options.get('detect_trails') and SATDET_AVAILABLE: trail_status = 'N/A';
                 if 'has_trails' in r: trail_status = 'Oui' if r['has_trails'] else 'Non'; log_line += f"{trail_status}\t{r.get('num_trails',0)}\t"
                 log_line += f"{r.get('exposure','N/A')}\t{r.get('filter','N/A')}\t{r.get('temperature','N/A')}\t"; log_line += f"{r.get('action','?')}\t{r.get('rejected_reason') or ''}\t{r.get('error_message') or ''} {r.get('action_comment') or ''}\n"
                 log_file.write(log_line.replace('\tnan','\t').replace('\tN/A','\t'))
    except IOError as e: _log("logic_log_init_error", path=output_log, e=e)

    # --- Étape 9: Écriture du Résumé Final et Retour ---
    _progress(100); end_time = time.time(); duration = end_time - start_time
    # Passer le compte des dossiers skippés au résumé
    write_log_summary(output_log, abs_input_dir, options, trail_analysis_config, trail_errors, all_results_list, selection_stats, skipped_marker_dirs_count)
    try:
        with open(output_log, 'a', encoding='utf-8') as log_file: log_file.write(f"\nDurée totale de l'analyse: {duration:.2f} secondes\n"); log_file.write("="*80 + "\nFin du log.\n")
    except IOError: pass
    _status("status_analysis_done")
    return all_results_list

# --- FIN DU FICHIER analyse_logic.py ---