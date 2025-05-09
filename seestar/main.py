# --- START OF FILE seestar/main.py ---
#!/usr/bin/env python3
"""
Script principal pour lancer l'application Seestar Stacker GUI.
"""

import os
import sys
import tkinter as tk
import traceback # Keep for detailed error reporting
import warnings
from astropy.io.fits.verify import VerifyWarning
import argparse

# --- Robust PYTHONPATH Modification ---
# Goal: Ensure the directory *containing* the 'seestar' package is in sys.path
try:
    # Path to the directory containing this script (main.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assumed project root (the 'seestar' package directory itself)
    project_root = script_dir
    # Parent directory of the 'seestar' package
    parent_of_project_root = os.path.dirname(project_root)

    # Add the parent directory to sys.path if it's not already there
    if parent_of_project_root not in sys.path:
        sys.path.insert(0, parent_of_project_root)
        print(f"DEBUG: Added to sys.path: {parent_of_project_root}")

    # Debug: Print relevant paths
    # print(f"DEBUG: Script directory: {script_dir}")
    # print(f"DEBUG: Project root (seestar package): {project_root}")
    # print(f"DEBUG: Parent added to path: {parent_of_project_root}")
    # print(f"DEBUG: Current sys.path: {sys.path}")

except Exception as path_e:
    print(f"Error setting up sys.path: {path_e}")
    # Attempt to continue, import might still work if installed

# --- Filter specific Astropy warnings globally --- ## THIS IS THE FIX ##
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
# --- End global filter --- ## END OF FIX ##
# --- Import Application and Version ---
try:
    # Now import using the package structure (e.g., from seestar.gui...)
    from seestar.gui import SeestarStackerGUI
    from seestar import __version__ as SEESTAR_VERSION

except ImportError as e:
    print(f"\n--- Import Error ---")
    print(f"Error: {e}")
    print("\nCould not import the Seestar application components.")
    print("Suggestions:")
    print("1. Ensure you are running this script from within the main 'seestar' directory")
    print("   OR that the 'seestar' directory is visible in your Python environment.")
    print("2. Make sure all required dependencies are installed (see 'requirements.txt').")
    print("   Run: pip install -r requirements.txt")
    print("\n--- Path Information ---")
    try:
        print(f"Script directory: {script_dir}")
        print(f"Parent added to path: {parent_of_project_root}")
    except NameError:
         print("Path variables not set due to earlier error.")
    print(f"Current sys.path: {sys.path}")
    print("-" * 20)
    try: input("Appuyez sur Entrée pour quitter...")
    except EOFError: pass
    sys.exit(1)
except Exception as e: # Catch other potential import errors
    print(f"Unexpected error during initial import: {e}")
    traceback.print_exc()
    try: input("Appuyez sur Entrée pour quitter...")
    except EOFError: pass
    sys.exit(1)


def check_dependencies():
    """
    Vérifie que toutes les dépendances sont installées.

    Returns:
        bool: True si toutes les dépendances sont installées, False sinon
    """
    # Dependencies required - should match requirements.txt
    dependencies = [
        ('numpy', 'numpy'),
        ('cv2', 'opencv-python'),
        ('astropy', 'astropy'),
        ('astroalign', 'astroalign'),
        ('tqdm', 'tqdm'),
        ('matplotlib', 'matplotlib'), # Needed for histogram widget
        ('PIL', 'Pillow'), # Needed for preview image handling
        ('skimage', 'scikit-image'), # Used by astroalign internally? Check if truly needed. Keeping for now.
        ('colour_demosaicing', 'colour-demosaicing'), # Needed for debayering in new drizzle code
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
        except Exception as e:
             print(f" ERREUR ({type(e).__name__})")
             missing_deps.append(package_name) # Assume missing if other error
             import_errors[package_name] = str(e)
             all_ok = False


    # Check optional psutil separately
    psutil_ok = False
    try:
        print("  Checking for psutil (optional)...", end='', flush=True)
        __import__('psutil')
        print(" OK")
        psutil_ok = True
    except ImportError:
         print(" MANQUANT")
    except Exception as e:
        print(f" ERREUR ({type(e).__name__})")

    print("Vérification terminée.")

    if not psutil_ok:
         print("\nAVERTISSEMENT: Dépendance optionnelle 'psutil' manquante.")
         print("L'estimation automatique de la taille des lots (batch size) ne fonctionnera pas.")
         print("Pour l'activer : pip install psutil\n")

    if not all_ok:
        # Ensure Pillow is mentioned if PIL failed
        if 'Pillow' not in missing_deps and any(p == 'Pillow' for _, p in dependencies if _ == 'PIL'):
             missing_deps.append('Pillow') # Add Pillow if PIL import failed

        # Remove duplicates just in case
        unique_missing = sorted(list(set(missing_deps)))

        print("\n--- ERREUR: Dépendances Manquantes ---")
        print("Veuillez installer ou vérifier les dépendances suivantes:")
        print(f"  pip install {' '.join(unique_missing)}")
        print("\nExemple de commande complète (ajustez si nécessaire):")
        print(f"  pip install numpy opencv-python astropy astroalign tqdm matplotlib Pillow scikit-image")
        if not psutil_ok: print(f"  pip install psutil  (Optionnel)")

        print("\nDétails des erreurs d'importation rencontrées:")
        for pkg in unique_missing:
            if pkg in import_errors:
                 print(f"  - {pkg}: {import_errors[pkg]}")
        print("-" * 30)

        return False

    return True



def main():
    """Point d'entrée principal de l'application."""
    print(f"\n--- Démarrage de Seestar Stacker v{SEESTAR_VERSION} ---")
    print(f"DEBUG (main.py): Lancement du script principal.") # <-- AJOUTÉ DEBUG

    # Vérifier les dépendances
    if not check_dependencies():
        print("\nOpération annulée en raison de dépendances manquantes.")
        try: input("Appuyez sur Entrée pour quitter...")
        except EOFError: pass
        sys.exit(1)

    # --- MODIFIÉ : Parser les arguments de ligne de commande ---
    print("DEBUG (main.py): Configuration du parser d'arguments...") # <-- AJOUTÉ DEBUG
    parser = argparse.ArgumentParser(description="Seestar Stacker GUI")
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Optional: Pre-fill the input directory."
    )
    # --- NOUVEL ARGUMENT AJOUTÉ ICI ---
    parser.add_argument(
        "--stack-from-analyzer", # <-- NOUVEAU
        type=str,
        metavar="ANALYZED_DIR", # Nom affiché dans l'aide
        help="Internal: Launch stacking directly from analyzer for the specified directory."
    )
    # Ajouter d'autres arguments ici si nécessaire
    print("DEBUG (main.py): Parsing des arguments fournis...") # <-- AJOUTÉ DEBUG
    args = parser.parse_args()
    print(f"DEBUG (main.py): Arguments parsés: {args}") # <-- AJOUTÉ DEBUG

    input_dir_from_args = None
    stack_immediately_path = None # <-- NOUVEAU: Variable pour stocker chemin si lancé par analyseur

    # --- NOUVEAU: Logique pour --stack-from-analyzer ---
    if args.stack_from_analyzer:
        abs_path_analyzer = os.path.abspath(args.stack_from_analyzer)
        if os.path.isdir(abs_path_analyzer):
            stack_immediately_path = abs_path_analyzer
            input_dir_from_args = abs_path_analyzer # Pré-remplir aussi le champ input
            print(f"INFO (main.py): Lancement automatique stacking demandé par analyseur pour: {stack_immediately_path}") # <-- AJOUTÉ INFO
        else:
            print(f"AVERTISSEMENT (main.py): Le chemin --stack-from-analyzer '{args.stack_from_analyzer}' n'est pas un dossier valide. Empilage automatique ignoré.")
            # Optionnel: Mettre input_dir_from_args à None pour ne pas pré-remplir si invalide ?
            # input_dir_from_args = None
    # --- FIN NOUVEAU ---
    # --- Gérer --input-dir (SEULEMENT si --stack-from-analyzer n'a pas été utilisé) ---
    elif args.input_dir: # <-- AJOUTÉ 'elif' pour prioriser --stack-from-analyzer
        abs_path_arg = os.path.abspath(args.input_dir)
        if os.path.isdir(abs_path_arg):
            input_dir_from_args = abs_path_arg
            print(f"INFO (main.py): Dossier d'entrée pré-rempli depuis argument --input-dir: {input_dir_from_args}")
        else:
            print(f"AVERTISSEMENT (main.py): Le chemin --input-dir '{args.input_dir}' n'est pas un dossier valide. Ignoré.")
    # --- FIN MODIFICATION ---

    print("\nLancement de l'interface graphique...")
    # Lancer l'interface graphique
    root = None
    try:
        # --- MODIFIÉ: Passer le chemin pour le stacking immédiat à SeestarStackerGUI ---
        print(f"DEBUG (main.py): Instanciation SeestarStackerGUI avec initial_input_dir='{input_dir_from_args}', stack_immediately_from='{stack_immediately_path}'") # <-- AJOUTÉ DEBUG
        app = SeestarStackerGUI(
            initial_input_dir=input_dir_from_args,
            stack_immediately_from=stack_immediately_path 
        )
        # --- FIN MODIFICATION ---

        root = app.root
        print("DEBUG (main.py): Entrée dans root.mainloop().") # <-- AJOUTÉ DEBUG
        root.mainloop()
        print("DEBUG (main.py): Sortie de root.mainloop().") # <-- AJOUTÉ DEBUG

    except tk.TclError as e:
        # ... (Gestion erreurs Tkinter/Tcl identique) ...
        err_str = str(e).lower()
        print("\n--- ERREUR Tkinter/Tcl ---")
        if "display" in err_str or "no display name" in err_str or "couldn't connect to display" in err_str:
            print("Impossible d'ouvrir l'affichage graphique.")
            print("Assurez-vous qu'un environnement graphique (X11, Wayland, Aqua, Bureau à Distance)")
            print("est disponible et correctement configuré.")
            print("Si vous utilisez SSH, vérifiez le transfert X11 (ssh -X).")
        elif "invalid command name" in err_str:
             print("Commande Tkinter invalide.")
             print("Cela peut indiquer un problème avec l'installation de Tk/Tcl,")
             print("une incompatibilité de version, ou une erreur interne.")
        else:
            print("Erreur Tkinter/Tcl inattendue.")
        print(f"\nErreur détaillée: {e}")
        traceback.print_exc()
        print("-" * 25)
        try: input("Appuyez sur Entrée pour quitter...")
        except EOFError: pass
        sys.exit(1)

    except Exception as e:
        # ... (Gestion erreurs générales identique) ...
        print(f"\n--- ERREUR INATTENDUE ---")
        print(f"Une erreur s'est produite lors du lancement ou de l'exécution de l'application:")
        print(f"Type: {type(e).__name__}")
        print(f"Erreur: {e}")
        print("\n--- Traceback ---")
        traceback.print_exc()
        print("-" * 20)
        try: input("Appuyez sur Entrée pour quitter...")
        except EOFError: pass
        sys.exit(1)

    print("\n--- Seestar Stacker Terminé ---")


if __name__ == "__main__":
    # Basic error handling around main() itself
    try:
        main()
    except SystemExit:
         pass # Allow sys.exit() to function normally
    except Exception as e:
        print(f"\n--- ERREUR FATALE NON INTERCEPTÉE ---")
        print(f"Une erreur critique s'est produite dans main():")
        print(f"Type: {type(e).__name__}")
        print(f"Erreur: {e}")
        print("\n--- Traceback ---")
        traceback.print_exc()
        print("-" * 30)
        try: input("Appuyez sur Entrée pour quitter...")
        except EOFError: pass
        sys.exit(1)
# --- END OF FILE seestar/main.py ---