# --- START OF FILE trail_module.py ---

import os
import numpy as np
import warnings
import inspect
import traceback

# --- Gestion acstools ---
SATDET_AVAILABLE = False
SATDET_USES_SEARCHPATTERN = False # Garder pour info
# SATDET_ACCEPTS_LIST = False     # Plus nécessaire de tracker spécifiquement

try:
    from acstools import satdet
    if hasattr(satdet, 'detsat') and callable(satdet.detsat):
        sig = inspect.signature(satdet.detsat)
        params = list(sig.parameters)
        first_param_name = params[0] if params else None
        # On vérifie juste si le premier paramètre est celui attendu pour un pattern
        if first_param_name in ['searchpattern', 'input_obj']: # Accepter les deux noms courants
            SATDET_AVAILABLE = True
            SATDET_USES_SEARCHPATTERN = True # On suppose qu'il prend un pattern
            print(f"INFO (trail_module): acstools.satdet.detsat détecté (premier param: '{first_param_name}').")
            # On ne vérifie plus explicitement SATDET_ACCEPTS_LIST ici
        else:
            print(f"AVERTISSEMENT (trail_module): acstools.satdet.detsat trouvé, mais signature ({params}) non attendue. Détection désactivée.")
            SATDET_AVAILABLE = False
    else:
         print("AVERTISSEMENT (trail_module): acstools importé, mais satdet.detsat non trouvé/callable. Détection désactivée.")
except ImportError:
    print("INFO (trail_module): acstools non trouvé. Détection de traînées désactivée.")
except Exception as e:
    print(f"ERREUR (trail_module): Erreur lors de l'import/inspection de acstools.satdet: {e}"); traceback.print_exc()

# --- Dépendances Optionnelles ---
SCIPY_AVAILABLE = False; SKIMAGE_AVAILABLE = False
if SATDET_AVAILABLE:
    try: import scipy; SCIPY_AVAILABLE = True
    except ImportError: print("AVERTISSEMENT (trail_module): scipy non trouvé.")
    try: import skimage; SKIMAGE_AVAILABLE = True
    except ImportError: print("AVERTISSEMENT (trail_module): scikit-image non trouvé.")

# --- REVERTED FUNCTION ---
def run_trail_detection(search_pattern, sat_params_input, status_callback=None, log_callback=None):
    """
    Exécute acstools.satdet.detsat pour détecter les traînées en utilisant un search_pattern.
    (Version revertie pour n'accepter que les patterns)
    """
    _status_callback = status_callback if callable(status_callback) else lambda k, **kw: print(f"TRAIL_STATUS: {k} {kw}")
    _log_callback = log_callback if callable(log_callback) else lambda k, **kw: print(f"TRAIL_LOG: {k} {kw}")

    # --- Vérifications Préalables ---
    if not SATDET_AVAILABLE or not SATDET_USES_SEARCHPATTERN: # Vérifier si compatible pattern
        err_msg = "Détection traînées non disponible ou incompatible avec les search patterns."
        _log_callback("logic_error_prefix", text=err_msg)
        return {}, {('FATAL_ERROR', 0): err_msg}

    if not isinstance(search_pattern, str): # Doit être un string maintenant
         err_msg = f"Type d'entrée invalide pour run_trail_detection (version pattern): {type(search_pattern)}. Attendu str."
         _log_callback("logic_error_prefix", text=err_msg)
         return {}, {('CONFIG_ERROR', 0): err_msg}

    if not SCIPY_AVAILABLE or not SKIMAGE_AVAILABLE:
         missing_deps = [dep for dep, avail in [("scipy", SCIPY_AVAILABLE), ("scikit-image", SKIMAGE_AVAILABLE)] if not avail]
         err_msg = f"Dépendances manquantes pour satdet: {', '.join(missing_deps)}"
         _log_callback("logic_error_prefix", text=err_msg); _status_callback("status_satdet_dep_error")
         return {}, {('DEPENDENCY_ERROR', 0): err_msg}

    # --- Récupérer et valider les paramètres (inchangé) ---
    defaults = {'sigma': 2.0, 'low_thresh': 0.1, 'h_thresh': 0.5, 'line_len': 150, 'small_edge': 60, 'line_gap': 75}
    params = {}
    for key, default_val in defaults.items():
        value_str = sat_params_input.get(key); current_val = default_val
        if value_str is not None:
            try:
                if key in ['sigma', 'low_thresh', 'h_thresh']: current_val = float(value_str)
                else: current_val = int(value_str)
                if key == 'sigma' and current_val <= 0: raise ValueError("doit être > 0")
                if key == 'low_thresh' and not (0 <= current_val <= 1): raise ValueError("doit être entre 0 et 1")
                if key == 'h_thresh' and not (0 <= current_val <= 1): raise ValueError("doit être entre 0 et 1")
                if key == 'line_len' and current_val <= 0: raise ValueError("doit être > 0")
                if key == 'small_edge' and current_val < 0: raise ValueError("doit être >= 0")
                if key == 'line_gap' and current_val <= 0: raise ValueError("doit être > 0")
                params[key] = current_val
            except (ValueError, TypeError) as e: log_key = f"logic_{key.replace('_thresh','thr')}_invalid"; _log_callback(log_key, e=e, default=default_val); params[key] = default_val
        else: params[key] = default_val
    if params['h_thresh'] < params['low_thresh']: original_high = params['h_thresh']; params['h_thresh'] = params['low_thresh']; _log_callback("logic_warn_prefix", text=f"High Thresh ({original_high}) < Low Thresh ({params['low_thresh']}). Ajustement High Thresh à {params['h_thresh']}.")

    # --- Paramètres fixes (inchangé) ---
    chips_to_use = [0]; n_processes = 1; verbose_det = False; plot_det = False

    # --- Exécution de satdet ---
    _status_callback("status_satdet_wait"); _log_callback("logic_satdet_params", **params, chips=chips_to_use)
    results = {}; errors = {}
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message=r'.*is not a valid science extension.*', category=UserWarning)

            # --- REVERTED: Appel à satdet avec searchpattern ---
            # Déterminer le nom du premier argument attendu par detsat
            sig = inspect.signature(satdet.detsat)
            first_param_name = list(sig.parameters)[0]

            satdet_kwargs = {
                'chips': chips_to_use, 'n_processes': n_processes, 'sigma': params['sigma'],
                'low_thresh': params['low_thresh'], 'h_thresh': params['h_thresh'],
                'line_len': params['line_len'], 'small_edge': params['small_edge'],
                'line_gap': params['line_gap'], 'plot': plot_det, 'verbose': verbose_det
            }
            # Ajouter le search_pattern avec le bon nom de paramètre
            satdet_kwargs[first_param_name] = search_pattern

            results, errors = satdet.detsat(**satdet_kwargs)
            # --- FIN REVERT ---

        _status_callback("status_satdet_done")

        # Logguer les erreurs spécifiques (inchangé)
        if errors:
            _log_callback("logic_satdet_errors_title"); count = 0
            for key, msg in errors.items():
                if isinstance(key, tuple) and len(key) == 2: fname, ext = key;
                if "is not a valid science extension" not in str(msg): _log_callback("logic_satdet_errors_item", fname=os.path.basename(fname), ext=ext, msg=msg); count += 1
                else: _log_callback("logic_error_prefix", text=f"Erreur Satdet non liée à un fichier ({key}): {msg}"); count += 1
            if count == 0: _log_callback("logic_satdet_errors_none")
        return results, errors

    except ImportError as imp_err: err_msg = f"Erreur d'importation interne lors de l'appel à satdet: {imp_err}."; _log_callback("logic_satdet_import_error", e=imp_err); _status_callback("status_satdet_dep_error"); return {}, {('IMPORT_ERROR', 0): err_msg}
    except Exception as e:
         err_msg = f"Erreur majeure lors de l'appel à acstools.satdet.detsat: {e}"; _log_callback("logic_satdet_major_error", e=e); print("\n--- Traceback Erreur Trail Module ---"); traceback.print_exc(); print("-----------------------------------\n"); _status_callback("status_satdet_error"); return {}, {('FATAL_ERROR', 0): str(e)}

# --- FIN DU FICHIER trail_module.py ---