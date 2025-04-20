#!/usr/bin/env python3
"""
Script principal pour lancer l'application Seestar Stacker.
"""

import os
import sys
import tkinter as tk # <--- Added this import
import traceback # Keep for detailed error reporting

# Ajouter le chemin du projet au PYTHONPATH
# This assumes main.py is inside the 'seestar' directory, and you run it from there.
# If 'seestar' is installed as a package, this might not be necessary.
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import using the package structure
try:
    from seestar.gui import SeestarStackerGUI
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure the 'seestar' package is in your PYTHONPATH or installed.")
    print(f"Project root added to path: {project_root}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)
except Exception as e: # Catch other potential import errors
    print(f"Unexpected error during initial import: {e}")
    traceback.print_exc()
    sys.exit(1)


def check_dependencies():
    """
    Vérifie que toutes les dépendances sont installées.

    Returns:
        bool: True si toutes les dépendances sont installées, False sinon
    """
    # Dependencies required by the corrected code
    dependencies = [
        ('numpy', 'numpy'),
        ('astropy', 'astropy'),
        ('cv2', 'opencv-python'),
        ('astroalign', 'astroalign'),
        ('tqdm', 'tqdm'),
        ('matplotlib', 'matplotlib'), # Added for tools.stretch
        ('PIL', 'Pillow'), # Added for GUI preview
    ]

    missing_deps = []

    print("Vérification des dépendances...")
    for module_name, package_name in dependencies:
        try:
            print(f"  Checking for {module_name}...", end='')
            __import__(module_name)
            print(" OK")
        except ImportError:
            print(" MANQUANT")
            missing_deps.append(package_name)
        except Exception as e:
             print(f" ERREUR ({e})")
             missing_deps.append(package_name) # Assume missing if other error

    # Check optional psutil separately
    psutil_ok = False
    try:
        print("  Checking for psutil (optional)...", end='')
        __import__('psutil')
        print(" OK")
        psutil_ok = True
    except ImportError:
         print(" MANQUANT")
    except Exception as e:
        print(f" ERREUR ({e})")

    print("Vérification terminée.")

    if not psutil_ok:
         print("\nAVERTISSEMENT: Dépendance optionnelle 'psutil' manquante.")
         print("L'estimation automatique de la taille des lots ne fonctionnera pas.")
         print("Pour l'activer : pip install psutil\n")

    if missing_deps:
        # Ensure Pillow is specifically mentioned if PIL failed
        if 'Pillow' not in missing_deps and any(p == 'Pillow' for _, p in dependencies if _ == 'PIL'):
             missing_deps.append('Pillow') # Add Pillow if PIL import failed

        # Remove duplicates just in case
        unique_missing = sorted(list(set(missing_deps)))

        print("\nERREUR: Dépendances manquantes!")
        print("Veuillez installer les dépendances suivantes:")
        print(f"pip install {' '.join(unique_missing)}")
        print("\nExemple de commande complète:")
        print(f"pip install numpy astropy opencv-python astroalign tqdm matplotlib Pillow")

        return False

    return True


def main():
    """Point d'entrée principal de l'application."""
    print("\nDémarrage de Seestar Stacker...")

    # Vérifier les dépendances
    if not check_dependencies():
        print("\nVeuillez installer les dépendances manquantes et réessayer.")
        sys.exit(1)

    print("\nLancement de l'interface graphique...")
    # Lancer l'interface graphique
    try:
        app = SeestarStackerGUI()
        app.root.mainloop()
    except tk.TclError as e:
        # Handle common Tkinter errors more gracefully
        if "display" in str(e).lower() or "no display name" in str(e).lower():
            print("\nERREUR: Impossible d'ouvrir l'affichage graphique.")
            print("Assurez-vous qu'un environnement graphique est disponible")
            print("(par exemple, X11, Wayland, Aqua, ou une session Bureau à distance).")
            print(f"Erreur Tcl détaillée: {e}")
        elif "Invalid command name" in str(e):
             print("\nERREUR Tkinter: Commande invalide.")
             print("Cela peut indiquer un problème avec l'installation de Tk/Tcl ou une erreur interne.")
             print(f"Erreur Tcl détaillée: {e}")
        else:
            print(f"\nERREUR Tkinter/Tcl inattendue: {e}")
            traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected error during GUI setup or run
        print(f"\nERREUR inattendue lors du lancement ou de l'exécution de l'application: {e}")
        traceback.print_exc()
        sys.exit(1)
    print("\nSeestar Stacker terminé.")


if __name__ == "__main__":
    # Basic error handling around main() itself
    try:
        main()
    except Exception as e:
        print(f"\nERREUR fatale non interceptée dans main(): {e}")
        traceback.print_exc()
        sys.exit(1)