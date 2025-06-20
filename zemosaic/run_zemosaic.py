# zemosaic/run_zemosaic.py
import os
import sys  # Ajout pour sys.path et sys.modules
import tkinter as tk
from tkinter import messagebox
import logging
import traceback

logger = logging.getLogger(__name__)

# --- Robust PYTHONPATH Modification (mirrors seestar/main.py logic) ---
try:
    current_script_path = os.path.abspath(__file__)
    zemosaic_package_dir = os.path.dirname(current_script_path)
    project_root_dir = os.path.dirname(zemosaic_package_dir)

    if project_root_dir in sys.path:
        sys.path.remove(project_root_dir)
    sys.path.insert(0, project_root_dir)

    if zemosaic_package_dir in sys.path and zemosaic_package_dir != project_root_dir:
        try:
            sys.path.remove(zemosaic_package_dir)
        except ValueError:
            pass

    if __name__ in ("__main__", "__mp_main__") and (
        __package__ is None or __package__ == ""
    ):
        __package__ = os.path.basename(zemosaic_package_dir)
except Exception as path_e:
    logger.error("Erreur configuration sys.path/package: %s", path_e)
    traceback.print_exc()
# --- FIN modification PYTHONPATH ---

from .core.cuda_utils import enforce_nvidia_gpu


# --- Impression de débogage initiale ---
print("--- run_zemosaic.py: DÉBUT DES IMPORTS ---")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"Chemin de travail actuel (CWD): {sys.path[0]}") # sys.path[0] est généralement le dossier du script


if enforce_nvidia_gpu():
    print(f"GPU NVIDIA forcée (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
else:
    print("Aucun GPU NVIDIA détecté ou 'nvidia-smi' indisponible.")


# Essayer d'importer la classe GUI et la variable de disponibilité du worker
try:
    from zemosaic_gui import ZeMosaicGUI, ZEMOSAIC_WORKER_AVAILABLE
    print("--- run_zemosaic.py: Import de zemosaic_gui RÉUSSI ---")

    # Vérifier le module zemosaic_worker si la GUI dit qu'il est disponible
    if ZEMOSAIC_WORKER_AVAILABLE:
        try:
            # Tenter d'importer zemosaic_worker directement pour inspecter son chemin
            # Note: Il est déjà importé par zemosaic_gui si ZEMOSAIC_WORKER_AVAILABLE est True
            import zemosaic_worker 
            print(f"DEBUG (run_zemosaic): zemosaic_worker chargé depuis: {zemosaic_worker.__file__}")
            if 'zemosaic_worker' in sys.modules:
                 print(f"DEBUG (run_zemosaic): sys.modules['zemosaic_worker'] pointe vers: {sys.modules['zemosaic_worker'].__file__}")
            else:
                print("DEBUG (run_zemosaic): zemosaic_worker n'est pas dans sys.modules après import direct (étrange).")
        except ImportError as e_worker_direct:
            print(f"ERREUR (run_zemosaic): Échec de l'import direct de zemosaic_worker pour débogage: {e_worker_direct}")
        except AttributeError:
            print(f"ERREUR (run_zemosaic): zemosaic_worker importé mais n'a pas d'attribut __file__ (très étrange).")

except ImportError as e:
    print(f"ERREUR CRITIQUE (run_zemosaic): Impossible d'importer ZeMosaicGUI depuis zemosaic_gui.py: {e}")
    print("  Veuillez vérifier que zemosaic_gui.py est présent et que toutes ses dépendances Python sont installées.")
    
    try:
        root_err = tk.Tk()
        root_err.withdraw()
        messagebox.showerror("Erreur de Lancement Fatale",
                             f"Impossible d'importer le module GUI principal (zemosaic_gui.py).\n"
                             f"Erreur: {e}\n\n"
                             "Veuillez vérifier les logs console pour plus de détails.")
        root_err.destroy()
    except Exception as tk_err:
        print(f"  Erreur Tkinter lors de la tentative d'affichage de la messagebox: {tk_err}")
    
    ZEMOSAIC_WORKER_AVAILABLE = False 
    ZeMosaicGUI = None

print("--- run_zemosaic.py: FIN DES IMPORTS ---")
print("DEBUG (run_zemosaic): sys.path complet:\n" + "\n".join(sys.path))
print("-" * 50)


def main():
    """Fonction principale pour lancer l'application ZeMosaic."""
    print("--- run_zemosaic.py: Entrée dans main() ---")

    # Vérification de sys.modules au début de main
    if 'zemosaic_worker' in sys.modules:
        print(f"DEBUG (main): 'zemosaic_worker' EST dans sys.modules. Chemin: {sys.modules['zemosaic_worker'].__file__}")
    else:
        print("DEBUG (main): 'zemosaic_worker' N'EST PAS dans sys.modules au début de main.")


    if not ZeMosaicGUI: 
        print("ZeMosaic ne peut pas démarrer car la classe GUI (ZeMosaicGUI) n'a pas pu être chargée.")
        return

    if not ZEMOSAIC_WORKER_AVAILABLE:
        print("Avertissement (run_zemosaic main): Le module worker (zemosaic_worker.py) n'est pas disponible ou n'a pas pu être importé correctement par zemosaic_gui.py.")
        
        root_temp_err_worker = tk.Tk()
        root_temp_err_worker.withdraw() 
        messagebox.showerror("Erreur de Lancement Critique (Worker)",
                             "Le module 'zemosaic_worker.py' est introuvable ou contient une erreur d'importation.\n"
                             "L'application ZeMosaic ne peut pas démarrer correctement.\n\n"
                             "Veuillez vérifier les logs console pour plus de détails.")
        root_temp_err_worker.destroy()
        return 

    print("DEBUG (main): ZEMOSAIC_WORKER_AVAILABLE est True. Tentative de création de l'interface graphique.")
    root = tk.Tk()
    app = ZeMosaicGUI(root)
    root.mainloop()
    print("--- run_zemosaic.py: mainloop() terminée ---")

if __name__ == "__main__":
    print("Lancement de ZeMosaic via run_zemosaic.py (__name__ == '__main__')...")
    main()
    print("ZeMosaic terminé (sortie de __main__).")