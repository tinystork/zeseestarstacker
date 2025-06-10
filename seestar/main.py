#!/usr/bin/env python3
"""
Script principal pour lancer l'application Seestar Stacker GUI.
Ce script est destiné à être lancé depuis l'intérieur du package 'seestar'.
"""

import os
import sys
import tkinter as tk
import traceback 
import warnings
from astropy.io.fits.verify import VerifyWarning
import argparse
import logging

logger = logging.getLogger(__name__)

# --- MODIFIED Robust PYTHONPATH Modification ---
try:
    current_script_path = os.path.abspath(__file__)
    seestar_package_dir = os.path.dirname(current_script_path)  # Chemin vers F:\...\seestar\
    project_root_dir = os.path.dirname(seestar_package_dir)     # Chemin vers F:\...\zeseestarstacker

    # 1. S'assurer que la RACINE DU PROJET est dans sys.path et en PREMIÈRE position.
    #    Ceci est crucial pour que 'import seestar.xxx' fonctionne correctement.
    if project_root_dir in sys.path:
        sys.path.remove(project_root_dir) # L'enlever s'il est déjà là pour le remettre en tête
    sys.path.insert(0, project_root_dir)
    logger.debug("Project root '%s' mis en tête de sys.path.", project_root_dir)

    # 2. S'assurer que le DOSSIER DU SCRIPT LUI-MÊME (seestar/) N'EST PAS dans sys.path
    #    s'il a été ajouté automatiquement parce qu'on lance le script depuis ce dossier.
    #    Avoir à la fois la racine et le dossier du package peut causer des conflits.
    #    Attention : ne le supprimer que s'il n'est pas identique à project_root_dir (ne devrait pas arriver ici)
    if seestar_package_dir in sys.path and seestar_package_dir != project_root_dir:
        try:
            sys.path.remove(seestar_package_dir)
            logger.debug(
                "Supprimé seestar_package_dir '%s' de sys.path pour éviter ambiguïté.",
                seestar_package_dir,
            )
        except ValueError:
            pass # N'était pas là, c'est bien.
            
    # 3. Définir __package__ pour aider les imports relatifs dans le package
    #    Seulement si ce script est le point d'entrée.
    if __name__ == "__main__" and (__package__ is None or __package__ == ""):
        # Le nom du package est le nom du dossier parent du script main.py,
        # qui est 'seestar' dans ce cas.
        package_name = os.path.basename(seestar_package_dir)
        if package_name: # S'assurer que le nom n'est pas vide
            __package__ = package_name
            logger.debug("__package__ défini à '%s'.", __package__)
        else:
            logger.warning(
                "Impossible de déterminer le nom du package depuis %s.",
                seestar_package_dir,
            )


except Exception as path_e:
    logger.error("Erreur configuration sys.path/package: %s", path_e)
    traceback.print_exc()
# --- FIN MODIFIED ---


# --- Bloc de débogage des imports (gardé) ---
# ... (identique à avant) ...
logger.debug("--------------------")
logger.debug("__package__ est maintenant: %s", __package__)
logger.debug("sys.path au moment de l'import de SeestarStackerGUI:")
for p_idx, p_path in enumerate(sys.path):
    logger.debug("  [%s] %s", p_idx, p_path)
logger.debug("--------------------")
try:
    import queuep.queue_manager
    logger.debug(
        "Chemin module seestar.queuep.queue_manager CHARGÉ : %s",
        queuep.queue_manager.__file__,
    )
    import gui.main_window
    logger.debug(
        "Chemin module seestar.gui.main_window CHARGÉ : %s",
        gui.main_window.__file__,
    )
except Exception as e_qm_debug:
    logger.debug("ERREUR import pour debug: %s", e_qm_debug)
logger.debug("--------------------")
# --- FIN Bloc de débogage ---



# --- NOUVEAU BLOC DE DÉBOGAGE POUR LES IMPORTS ---
logger.debug("--------------------")
logger.debug("sys.path au moment de l'import de SeestarStackerGUI:")
for p_idx, p_path in enumerate(sys.path):
    logger.debug("  [%s] %s", p_idx, p_path)
logger.debug("--------------------")

try:
    # Tentative d'import pour vérifier le chemin
    # Maintenant, on importe directement car project_root_dir est dans sys.path
    import queuep.queue_manager
    logger.debug(
        "Chemin module seestar.queuep.queue_manager CHARGÉ : %s",
        queuep.queue_manager.__file__,
    )
    import gui.main_window
    logger.debug(
        "Chemin module seestar.gui.main_window CHARGÉ : %s",
        gui.main_window.__file__,
    )

except ImportError as e_qm_debug:
    logger.debug(
        "ERREUR import seestar.queuep.queue_manager (ou autre) pour debug: %s",
        e_qm_debug,
    )
    logger.debug("  Cela peut indiquer un problème avec sys.path ou la structure du package.")
except AttributeError:
    logger.debug(
        "Module seestar importé mais pas d'attribut __file__."
    )
except Exception as e_debug_other:
    logger.debug(
        "Erreur inattendue vérification imports: %s",
        e_debug_other,
    )
logger.debug("--------------------")
# --- FIN NOUVEAU BLOC DE DÉBOGAGE ---


# --- Filter specific Astropy warnings globally ---
warnings.filterwarnings(
    'ignore',
    category=VerifyWarning,
    message="Keyword name.*is greater than 8 characters.*"
)
warnings.filterwarnings(
    'ignore',
    category=VerifyWarning,
    message="Keyword name.*contains characters not allowed.*"
)


# --- NOUVELLE VÉRIFICATION RADICALE ---
logger.debug("--------------------")
logger.debug(
    "DEBUG [seestar/main.py RADICAL CHECK]: Vérification du contenu de queue_manager.py..."
)
try:
    import importlib.util
    spec = importlib.util.find_spec("seestar.queuep.queue_manager")
    if spec and spec.origin:
        queue_manager_path = spec.origin
        logger.debug(
            "  Chemin trouvé pour seestar.queuep.queue_manager: %s",
            queue_manager_path,
        )
        if os.path.exists(queue_manager_path):
            logger.debug("  Le fichier %s EXISTE.", queue_manager_path)
            with open(queue_manager_path, 'r', encoding='utf-8') as f_qm_content:
                content = f_qm_content.read()
                # Chercher la définition de set_preview_callback
                idx_def = content.find("def set_preview_callback(self, callback):")
                if idx_def != -1:
                    # Extraire environ 10 lignes après la définition
                    snippet_end = content.find("\n", idx_def + 800) # Cherche la fin de ligne dans les ~20 lignes suivantes
                    if snippet_end == -1: snippet_end = idx_def + 800 # Si pas de \n, prendre une tranche fixe
                    snippet = content[idx_def:snippet_end]
                    logger.debug(
                        "  Extrait de set_preview_callback dans %s (lignes ~%s):",
                        queue_manager_path,
                        content.count(os.linesep, 0, idx_def) + 1,
                    )
                    logger.debug("    ---- SNIPPET START ----")
                    for line_in_snippet in snippet.splitlines()[:10]: # Afficher les 10 premières lignes du snippet
                        logger.debug("    | %s", line_in_snippet)
                    logger.debug("    ---- SNIPPET END ----")
                    if "_cleanup_mosaic_panel_stacks_temp()" in snippet or \
                       "_cleanup_drizzle_batch_outputs()" in snippet or \
                       "cleanup_unaligned_files()" in snippet:
                        logger.debug(
                            "  ALERTE RADICAL CHECK: Un appel _cleanup_ semble être présent dans le snippet de set_preview_callback !"
                        )
                    else:
                        logger.debug(
                            "  INFO RADICAL CHECK: Aucun appel _cleanup_ évident dans les premières lignes du snippet de set_preview_callback."
                        )
                else:
                    logger.debug(
                        "  ERREUR RADICAL CHECK: 'def set_preview_callback' non trouvée dans %s",
                        queue_manager_path,
                    )
        else:
            logger.debug(
                "  ERREUR RADICAL CHECK: Le fichier %s N'EXISTE PAS (problème find_spec).",
                queue_manager_path,
            )
    else:
        logger.debug(
            "  ERREUR RADICAL CHECK: Spec pour seestar.queuep.queue_manager non trouvé."
        )
except Exception as e_radical:
    logger.debug("  ERREUR pendant la vérification radicale: %s", e_radical)
logger.debug("--------------------")
# --- FIN NOUVELLE VÉRIFICATION RADICALE ---

# --- Import Application and Version ---
# Ces imports devraient maintenant fonctionner car la racine du projet est dans sys.path
try:
    # --- MODIFICATION ICI ---
    # Utiliser des imports relatifs explicites puisque main.py est DANS le package seestar
    # et __package__ est défini.
    from .gui import SeestarStackerGUI  # '.' signifie le package courant (seestar)
    from . import __version__ as SEESTAR_VERSION # Accéder à __version__ via __init__.py du package courant
    # Assure-toi que seestar/__init__.py définit bien __version__
    logger.debug(
        "Import de .gui.SeestarStackerGUI et .__version__ réussi.")

except ImportError as e:
    logger.error("--- Import Error ---")
    logger.error("Error: %s", e)
    logger.error("__package__ au moment de l'erreur: %s", __package__)
    logger.error("Could not import the Seestar application components.")
    logger.error("Suggestions:")
    logger.error("1. Vérifiez que la racine du projet (contenant 'seestar/') est dans sys.path.")
    logger.error("2. Vérifiez que __package__ est correctement défini si ce script est lancé directement.")
    logger.error("3. Assurez-vous que les sous-modules (gui, core, etc.) ont des __init__.py.")
    logger.error("4. Vérifiez que __version__ est défini dans seestar/__init__.py.")

    try:
        logger.error("Script path: %s", current_script_path)
        logger.error("Project root added to path: %s", project_root_dir)
    except NameError:
         logger.error("Path variables not set due to earlier error.")
    logger.error("Current sys.path (at import error):")
    for p_path in sys.path:
        logger.error("  %s", p_path)
    logger.error("-" * 20)
    try:
        input("Appuyez sur Entrée pour quitter...")
    except EOFError: pass
    sys.exit(1)
except Exception as e:
    logger.error("Unexpected error during initial import: %s", e)
    traceback.print_exc()
    try:
        input("Appuyez sur Entrée pour quitter...")
    except EOFError: pass
    sys.exit(1)

# ... (le reste de ton fichier seestar/main.py : check_dependencies, main(), if __name__ == "__main__":)
# ... (AUCUN CHANGEMENT NÉCESSAIRE DANS CES FONCTIONS PAR RAPPORT À CETTE MODIFICATION DE SYS.PATH)

def check_dependencies():
    """
    Vérifie que toutes les dépendances sont installées.

    Returns:
        bool: True si toutes les dépendances sont installées, False sinon
    """
    # Dependencies required - should match requirements.txt
    dependencies = [
        ('numpy', 'numpy'),
        ('cv2', 'opencv-python'), # opencv-contrib-python est recommandé pour CUDA
        ('astropy', 'astropy'),
        ('astroalign', 'astroalign'),
        ('tqdm', 'tqdm'),
        ('matplotlib', 'matplotlib'), 
        ('PIL', 'Pillow'), 
        ('skimage', 'scikit-image'), 
        ('colour_demosaicing', 'colour-demosaicing'),
    ]

    missing_deps = []
    import_errors = {}

    print("Vérification des dépendances...")
    all_ok = True
    for module_name, package_name in dependencies:
        try:
            print(f"  Checking for {module_name}...", end='', flush=True)
            __import__(module_name)
            print(" OK")
        except ImportError as ie:
            print(" MANQUANT")
            missing_deps.append(package_name)
            import_errors[package_name] = str(ie)
            all_ok = False
        except Exception as e_dep_check: # Renommer la variable d'exception ici
             print(f" ERREUR ({type(e_dep_check).__name__})")
             missing_deps.append(package_name) 
             import_errors[package_name] = str(e_dep_check)
             all_ok = False

    psutil_ok = False
    try:
        print("  Checking for psutil (optional)...", end='', flush=True)
        __import__('psutil')
        print(" OK")
        psutil_ok = True
    except ImportError:
         print(" MANQUANT")
    except Exception as e_psutil: # Renommer la variable d'exception ici
        print(f" ERREUR ({type(e_psutil).__name__})")

    print("Vérification terminée.")

    if not psutil_ok:
         print("\nAVERTISSEMENT: Dépendance optionnelle 'psutil' manquante.")
         print("L'estimation automatique de la taille des lots (batch size) ne fonctionnera pas.")
         print("Pour l'activer : pip install psutil\n")

    if not all_ok:
        if 'Pillow' not in missing_deps and any(p == 'Pillow' for _, p in dependencies if _ == 'PIL'):
             missing_deps.append('Pillow')
        unique_missing = sorted(list(set(missing_deps)))
        print("\n--- ERREUR: Dépendances Manquantes ---")
        print("Veuillez installer ou vérifier les dépendances suivantes:")
        print(f"  pip install {' '.join(unique_missing)}")
        print("\nExemple de commande complète (ajustez si nécessaire):")
        print(f"  pip install numpy opencv-python astropy astroalign tqdm matplotlib Pillow scikit-image colour-demosaicing")
        if not psutil_ok: print(f"  pip install psutil  (Optionnel)")
        print("\nDétails des erreurs d'importation rencontrées:")
        for pkg in unique_missing:
            if pkg in import_errors: print(f"  - {pkg}: {import_errors[pkg]}")
        print("-" * 30)
        return False
    return True


def main():
    """Point d'entrée principal de l'application."""
    print(f"\n--- Démarrage de Seestar Stacker v{SEESTAR_VERSION} ---")
    print(f"DEBUG (seestar/main.py): Lancement de la fonction main().") # Modifié pour indiquer quel main

    if not check_dependencies():
        print("\nOpération annulée en raison de dépendances manquantes.")
        try: input("Appuyez sur Entrée pour quitter...")
        except EOFError: pass
        sys.exit(1)

    print("DEBUG (seestar/main.py): Configuration du parser d'arguments...")
    parser = argparse.ArgumentParser(description="Seestar Stacker GUI")
    parser.add_argument("--input-dir", type=str, help="Optional: Pre-fill the input directory.")
    parser.add_argument("--stack-from-analyzer", type=str, metavar="ANALYZED_DIR", help="Internal: Launch stacking directly from analyzer for the specified directory.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    print("DEBUG (seestar/main.py): Parsing des arguments fournis...")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print(f"DEBUG (seestar/main.py): Arguments parsés: {args}")

    input_dir_from_args = None
    stack_immediately_path = None

    if args.stack_from_analyzer:
        abs_path_analyzer = os.path.abspath(args.stack_from_analyzer)
        if os.path.isdir(abs_path_analyzer):
            stack_immediately_path = abs_path_analyzer
            input_dir_from_args = abs_path_analyzer
            print(f"INFO (seestar/main.py): Lancement auto stacking demandé par analyseur pour: {stack_immediately_path}")
        else:
            print(f"AVERTISSEMENT (seestar/main.py): Chemin --stack-from-analyzer '{args.stack_from_analyzer}' invalide. Ignoré.")
    elif args.input_dir:
        abs_path_arg = os.path.abspath(args.input_dir)
        if os.path.isdir(abs_path_arg):
            input_dir_from_args = abs_path_arg
            print(f"INFO (seestar/main.py): Dossier d'entrée pré-rempli depuis --input-dir: {input_dir_from_args}")
        else:
            print(f"AVERTISSEMENT (seestar/main.py): Chemin --input-dir '{args.input_dir}' invalide. Ignoré.")

    print("\nLancement de l'interface graphique...")
    root = None
    try:
        print(f"DEBUG (seestar/main.py): Instanciation SeestarStackerGUI avec initial_input_dir='{input_dir_from_args}', stack_immediately_from='{stack_immediately_path}'")
        app = SeestarStackerGUI(
            initial_input_dir=input_dir_from_args,
            stack_immediately_from=stack_immediately_path 
        )
        root = app.root
        print("DEBUG (seestar/main.py): Entrée dans root.mainloop().")
        root.mainloop()
        print("DEBUG (seestar/main.py): Sortie de root.mainloop().")

    except tk.TclError as e_tk: 
        err_str = str(e_tk).lower()
        print("\n--- ERREUR Tkinter/Tcl ---")
        if "display" in err_str or "no display name" in err_str or "couldn't connect to display" in err_str:
            print("Impossible d'ouvrir l'affichage graphique. Vérifiez l'environnement X11/Wayland/Aqua.")
        elif "invalid command name" in err_str: print("Commande Tkinter invalide. Problème d'installation Tk/Tcl ou incompatibilité.")
        else: print("Erreur Tkinter/Tcl inattendue.")
        print(f"\nErreur détaillée: {e_tk}"); traceback.print_exc()
        print("-" * 25)
        try: input("Appuyez sur Entrée pour quitter...")
        except EOFError: pass
        sys.exit(1)

    except Exception as e_generic: 
        print(f"\n--- ERREUR INATTENDUE ---"); print(f"Erreur lors du lancement/exécution:"); print(f"Type: {type(e_generic).__name__}"); print(f"Erreur: {e_generic}")
        print("\n--- Traceback ---"); traceback.print_exc(); print("-" * 20)
        try: input("Appuyez sur Entrée pour quitter...")
        except EOFError: pass; sys.exit(1)

    print("\n--- Seestar Stacker Terminé ---")


if __name__ == "__main__":
    try:
        main()
    except SystemExit: pass
    except Exception as e_fatal: 
        print(f"\n--- ERREUR FATALE NON INTERCEPTÉE ---"); print(f"Erreur critique dans main():"); print(f"Type: {type(e_fatal).__name__}"); print(f"Erreur: {e_fatal}")
        print("\n--- Traceback ---"); traceback.print_exc(); print("-" * 30)
        try: input("Appuyez sur Entrée pour quitter...")
        except EOFError: pass; sys.exit(1)

# --- END OF FILE seestar/main.py ---
