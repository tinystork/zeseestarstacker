# zemosaic/run_zemosaic.py
import sys  # Ajout pour sys.path et sys.modules
# import reproject # L'import direct ici n'est pas crucial, mais ne fait pas de mal
import tkinter as tk
from tkinter import messagebox  # Nécessaire pour la messagebox d'erreur critique
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ZeMosaicRunner")

# --- Impression de débogage initiale ---
logger.debug("--- run_zemosaic.py: DÉBUT DES IMPORTS ---")
logger.debug(f"Python Executable: {sys.executable}")
logger.debug(f"Python Version: {sys.version}")
logger.debug(f"Chemin de travail actuel (CWD): {sys.path[0]}")  # sys.path[0] est généralement le dossier du script

# Essayer d'importer la classe GUI et la variable de disponibilité du worker
try:
    from .zemosaic_gui import ZeMosaicGUI, ZEMOSAIC_WORKER_AVAILABLE
    logger.debug("--- run_zemosaic.py: Import de zemosaic_gui RÉUSSI (relatif) ---")

    # Vérifier le module zemosaic_worker si la GUI dit qu'il est disponible
    if ZEMOSAIC_WORKER_AVAILABLE:
        try:
            # Tenter d'importer zemosaic_worker directement pour inspecter son chemin
            # Note: Il est déjà importé par zemosaic_gui si ZEMOSAIC_WORKER_AVAILABLE est True
            import zemosaic_worker
            logger.debug(f"zemosaic_worker chargé depuis: {zemosaic_worker.__file__}")
            if 'zemosaic_worker' in sys.modules:
                logger.debug(
                    f"sys.modules['zemosaic_worker'] pointe vers: {sys.modules['zemosaic_worker'].__file__}"
                )
            else:
                logger.debug("zemosaic_worker n'est pas dans sys.modules après import direct (étrange).")
        except ImportError as e_worker_direct:
            logger.error(
                f"Échec de l'import direct de zemosaic_worker pour débogage: {e_worker_direct}"
            )
        except AttributeError:
            logger.error(
                "zemosaic_worker importé mais n'a pas d'attribut __file__ (très étrange)."
            )

except ImportError as e:
    try:
        from zemosaic_gui import ZeMosaicGUI, ZEMOSAIC_WORKER_AVAILABLE
        logger.debug("--- run_zemosaic.py: Import de zemosaic_gui RÉUSSI (absolu) ---")
    except ImportError as e2:
        logger.critical(
            f"Impossible d'importer ZeMosaicGUI depuis zemosaic_gui.py: {e2}"
        )
        logger.critical(
            "  Veuillez vérifier que zemosaic_gui.py est présent et que toutes ses dépendances Python sont installées."
        )

        try:
            root_err = tk.Tk()
            root_err.withdraw()
            messagebox.showerror("Erreur de Lancement Fatale",
                                 f"Impossible d'importer le module GUI principal (zemosaic_gui.py).\n"
                                 f"Erreur: {e2}\n\n"
                                 "Veuillez vérifier les logs console pour plus de détails.")
            root_err.destroy()
        except Exception as tk_err:
            logger.error(
                f"Erreur Tkinter lors de la tentative d'affichage de la messagebox: {tk_err}"
            )

        ZEMOSAIC_WORKER_AVAILABLE = False
        ZeMosaicGUI = None

logger.debug("--- run_zemosaic.py: FIN DES IMPORTS ---")
import os
logger.debug(
    f"DEBUG (run_zemosaic): sys.path complet: {os.linesep}{os.linesep.join(sys.path)}"
)
logger.debug("-" * 50)


def main():
    """Fonction principale pour lancer l'application ZeMosaic."""
    logger.debug("--- run_zemosaic.py: Entrée dans main() ---")

    # Vérification de sys.modules au début de main
    if 'zemosaic_worker' in sys.modules:
        logger.debug(
            f"'zemosaic_worker' EST dans sys.modules. Chemin: {sys.modules['zemosaic_worker'].__file__}"
        )
    else:
        logger.debug("'zemosaic_worker' N'EST PAS dans sys.modules au début de main.")


    if not ZeMosaicGUI: 
        logger.error("ZeMosaic ne peut pas démarrer car la classe GUI (ZeMosaicGUI) n'a pas pu être chargée.")
        return

    if not ZEMOSAIC_WORKER_AVAILABLE:
        logger.error(
            "Avertissement (run_zemosaic main): Le module worker (zemosaic_worker.py) n'est pas disponible ou n'a pas pu être importé correctement par zemosaic_gui.py."
        )
        
        root_temp_err_worker = tk.Tk()
        root_temp_err_worker.withdraw() 
        messagebox.showerror("Erreur de Lancement Critique (Worker)",
                             "Le module 'zemosaic_worker.py' est introuvable ou contient une erreur d'importation.\n"
                             "L'application ZeMosaic ne peut pas démarrer correctement.\n\n"
                             "Veuillez vérifier les logs console pour plus de détails.")
        root_temp_err_worker.destroy()
        return 

    logger.debug("ZEMOSAIC_WORKER_AVAILABLE est True. Tentative de création de l'interface graphique.")
    root = tk.Tk()
    app = ZeMosaicGUI(root)
    root.mainloop()
    logger.debug("--- run_zemosaic.py: mainloop() terminée ---")

if __name__ == "__main__":
    logger.info("Lancement de ZeMosaic via run_zemosaic.py (__name__ == '__main__')...")
    main()
    logger.info("ZeMosaic terminé (sortie de __main__).")
