# zemosaic_config.py
import json
import os
import tkinter.filedialog as fd
import tkinter.messagebox as mb

CONFIG_FILE_NAME = "zemosaic_config.json"
DEFAULT_CONFIG = {
    "astap_executable_path": "",
    "astap_data_directory_path": "", 
    "astap_default_search_radius": 3.0, 
    "astap_default_downsample": 2, 
    "astap_default_sensitivity": 100,
    "language": "en",
    "num_processing_workers": -1, # -1 pour auto
    "stacking_normalize_method": "linear_fit",
    "stacking_weighting_method": "noise_variance",
    "stacking_rejection_algorithm": "winsorized_sigma_clip", 
    "stacking_kappa_low": 3.0,
    "stacking_kappa_high": 3.0,
    "stacking_winsor_limits": "0.05,0.05", # String, sera parsé
    "stacking_final_combine_method": "mean",
    "apply_radial_weight": False,
    "radial_feather_fraction": 0.8,
    "radial_shape_power": 2.0,
    "final_assembly_method": "reproject_coadd", # Options: "reproject_coadd", "incremental",
    "save_final_as_uint16": False,
    # --- CLES POUR LE ROGNAGE DES MASTER TUILES ---
    "apply_master_tile_crop": True,       # Désactivé par défaut
    "master_tile_crop_percent": 18.0      # Pourcentage par côté si activé (ex: 10%)
    # --- FIN CLES POUR LE ROGNAGE --- 
}

def get_config_path():
    """
    Retourne le chemin du fichier de configuration.
    Le fichier sera situé dans le même dossier que ce script (zemosaic_config.py).
    """
    # __file__ est le chemin du script actuel (zemosaic_config.py)
    # os.path.dirname(__file__) donne le dossier contenant ce script
    # os.path.abspath() assure que le chemin est absolu
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, CONFIG_FILE_NAME)

def load_config():
    config_path = get_config_path()
    current_config = DEFAULT_CONFIG.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f: # Spécifier encoding
                loaded_config = json.load(f)
                for key, default_value in DEFAULT_CONFIG.items():
                    current_config[key] = loaded_config.get(key, default_value)
                # Gérer les clés obsolètes ou nouvelles non présentes dans DEFAULT_CONFIG
                # Par exemple, on pourrait choisir de ne garder que les clés de DEFAULT_CONFIG
                # ou d'ajouter les nouvelles clés de loaded_config qui ne sont pas dans DEFAULT_CONFIG.
                # Pour l'instant, la boucle ci-dessus s'assure que toutes les clés de DEFAULT_CONFIG
                # sont présentes dans current_config, en prenant la valeur chargée si elle existe.
        except json.JSONDecodeError:
            # Utiliser mb (messagebox) si disponible, sinon print
            msg_title = "Config Error"
            msg_text = f"Error reading {config_path}. Using default configuration."
            try:
                if mb: mb.showwarning(msg_title, msg_text)
                else: print(f"WARNING: {msg_title} - {msg_text}")
            except Exception: print(f"WARNING: {msg_title} - {msg_text} (messagebox error)")
        except Exception as e:
            msg_title = "Config Error"
            msg_text = f"Unexpected error reading {config_path}: {e}. Using defaults."
            try:
                if mb: mb.showerror(msg_title, msg_text)
                else: print(f"ERROR: {msg_title} - {msg_text}")
            except Exception: print(f"ERROR: {msg_title} - {msg_text} (messagebox error)")
    # else:
        # print(f"Config file not found at {config_path}. Using default configuration.")
    return current_config

def save_config(config_data):
    config_path = get_config_path()
    try:
        # Avant de sauvegarder, s'assurer que config_data ne contient que les clés attendues
        # pour éviter d'écrire des clés temporaires ou obsolètes.
        # On ne garde que les clés qui sont dans DEFAULT_CONFIG.
        config_to_save = {}
        for key in DEFAULT_CONFIG.keys():
            if key in config_data:
                config_to_save[key] = config_data[key]
            # else: # Optionnel: si une clé par défaut manque dans config_data, la remettre
            #     config_to_save[key] = DEFAULT_CONFIG[key]


        # Si config_to_save est vide (si config_data n'avait aucune clé de DEFAULT_CONFIG),
        # on pourrait choisir de sauvegarder DEFAULT_CONFIG à la place, ou rien.
        # Pour l'instant, on sauvegarde ce qui a été filtré.
        # S'il est vide, cela pourrait indiquer un problème en amont.
        if not config_to_save and config_data: # Si config_data n'était pas vide mais qu'aucune clé n'a matché
            print(f"AVERT (save_config): Aucune clé de DEFAULT_CONFIG trouvée dans config_data. Sauvegarde de config_data tel quel.")
            config_to_save = config_data # Sauvegarder ce qu'on a reçu pour ne pas perdre d'info, mais c'est suspect
        elif not config_to_save and not config_data: # Si config_data était vide
             print(f"AVERT (save_config): config_data est vide, rien à sauvegarder pour {config_path}.")
             return False # Ne pas créer un fichier vide


        with open(config_path, 'w', encoding='utf-8') as f: # Spécifier encoding
            json.dump(config_to_save, f, indent=4, ensure_ascii=False) # ensure_ascii=False pour les caractères non-ASCII
        print(f"Configuration sauvegardée vers {config_path}")
        return True
    except IOError as e:
        msg_title = "Config Error"
        msg_text = f"Unable to save configuration to {config_path}:\n{e}"
        try:
            if mb: mb.showerror(msg_title, msg_text)
            else: print(f"ERROR: {msg_title} - {msg_text}")
        except Exception: print(f"ERROR: {msg_title} - {msg_text} (messagebox error)")
        return False

# Les fonctions ask_and_set_... et get_... restent les mêmes,
# elles utiliseront le nouveau chemin via get_config_path().
# Assurez-vous que tkinter.filedialog (fd) est importé si vous l'utilisez dans ces fonctions.
# Par exemple :
# import tkinter.filedialog as fd # Au début du fichier si ce n'est pas déjà fait globalement
# ... (vos fonctions ask_and_set_astap_path, etc.)

def ask_and_set_astap_path(current_config):
    """Demande à l'utilisateur le chemin de l'exécutable ASTAP et met à jour la config."""
    astap_path = fd.askopenfilename(
        title="Sélectionner l'exécutable ASTAP",
        filetypes=(("Fichiers exécutables", "*.exe"), ("Tous les fichiers", "*.*"))
    )
    if astap_path:
        current_config["astap_executable_path"] = astap_path
        if save_config(current_config):
            mb.showinfo("Chemin ASTAP Défini", f"Chemin ASTAP défini à : {astap_path}", parent=None) # Spécifier parent si possible
        return astap_path
    return current_config.get("astap_executable_path", "")


def ask_and_set_astap_data_dir_path(current_config):
    """Demande à l'utilisateur le chemin du dossier de données ASTAP et met à jour la config."""
    astap_data_dir = fd.askdirectory(
        title="Sélectionner le dossier de données ASTAP (contenant G17, H17, etc.)"
    )
    if astap_data_dir:
        current_config["astap_data_directory_path"] = astap_data_dir
        if save_config(current_config):
            mb.showinfo("Dossier Données ASTAP Défini", f"Dossier de données ASTAP défini à : {astap_data_dir}", parent=None)
        return astap_data_dir
    return current_config.get("astap_data_directory_path", "")


def get_astap_executable_path():
    config = load_config()
    return config.get("astap_executable_path", "")

def get_astap_data_directory_path():
    config = load_config()
    return config.get("astap_data_directory_path", "") # Retourne une chaîne vide si non défini

def get_astap_default_search_radius():
    config = load_config()
    return config.get("astap_default_search_radius", DEFAULT_CONFIG["astap_default_search_radius"])

def get_astap_default_downsample():
    config = load_config()
    return config.get("astap_default_downsample", DEFAULT_CONFIG["astap_default_downsample"])

def get_astap_default_sensitivity():
    config = load_config()
    return config.get("astap_default_sensitivity", DEFAULT_CONFIG["astap_default_sensitivity"])