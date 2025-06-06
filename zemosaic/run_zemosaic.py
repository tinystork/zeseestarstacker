# zemosaic/run_zemosaic.py
import sys  # Ajout pour sys.path et sys.modules
# import reproject # L'import direct ici n'est pas crucial, mais ne fait pas de mal
import argparse
import tkinter as tk
from tkinter import messagebox  # Nécessaire pour la messagebox d'erreur critique
import os
import logging

# Determine verbosity from environment variable or command-line flag
_verbose_env = os.getenv("SEESTAR_VERBOSE", "")
_verbose_flag = "-v" in sys.argv or "--verbose" in sys.argv
if _verbose_flag:
    # Remove the flag so Tkinter doesn't see it
    sys.argv = [a for a in sys.argv if a not in ("-v", "--verbose")]
log_level = logging.DEBUG if (
    _verbose_flag or str(_verbose_env).lower() in ("1", "true", "yes")
) else logging.INFO
logging.basicConfig(level=log_level)
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


def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run ZeMosaic either via the GUI (default) or headless CLI. "
            "Solver parameters are collected into a solver_settings dictionary "
            "and forwarded to run_hierarchical_mosaic."
        ),
        add_help=True,
    )
    parser.add_argument("input_folder", nargs="?", help="Folder with FITS images")
    parser.add_argument("output_folder", nargs="?", help="Destination folder")
    parser.add_argument("--cli", action="store_true", help="Run headless without GUI")
    parser.add_argument("--astap-path", dest="astap_path", help="Path to ASTAP executable")
    parser.add_argument(
        "--astap-data-dir", dest="astap_data_dir", help="Directory with ASTAP catalogs"
    )
    parser.add_argument(
        "--astrometry-method",
        dest="astrometry_method",
        choices=["astap", "astrometry", "astrometry.net"],
        help="Solver method",
    )
    return parser.parse_args()


def _cli_progress(msg: str, progress: float | None = None) -> None:
    if progress is not None:
        print(f"[{progress:.1f}%] {msg}")
    else:
        print(msg)


def _run_cli(args: argparse.Namespace) -> None:
    from . import zemosaic_config, zemosaic_worker

    config = zemosaic_config.load_config()

    solver_settings = {
        "astap_path": args.astap_path or config.get("astap_path"),
        "astap_data_dir": args.astap_data_dir or config.get("astap_data_dir"),
        "astap_search_radius": config.get("astap_default_search_radius", 3.0),
        "astap_downsample": config.get("astap_default_downsample", 2),
        "astap_sensitivity": config.get("astap_default_sensitivity", 100),
        "astrometry_method": args.astrometry_method or config.get("astrometry_method"),
    }

    winsor_str = config.get("stacking_winsor_limits", "0.05,0.05")
    try:
        winsor_limits = tuple(float(x) for x in winsor_str.split(","))
    except Exception:
        winsor_limits = (0.05, 0.05)

    zemosaic_worker.run_hierarchical_mosaic(
        args.input_folder,
        args.output_folder,
        solver_settings,
        config.get("cluster_panel_threshold", 0.5),
        _cli_progress,
        config.get("stacking_normalize_method", "none"),
        config.get("stacking_weighting_method", "none"),
        config.get("stacking_rejection_algorithm", "kappa_sigma"),
        config.get("stacking_kappa_low", 3.0),
        config.get("stacking_kappa_high", 3.0),
        winsor_limits,
        config.get("stacking_final_combine_method", "mean"),
        config.get("apply_radial_weight", False),
        config.get("radial_feather_fraction", 0.8),
        config.get("radial_shape_power", 2.0),
        config.get("min_radial_weight_floor", 0.0),
        config.get("final_assembly_method", "reproject_coadd"),
        config.get("num_processing_workers", 0),
        config.get("apply_master_tile_crop", False),
        config.get("master_tile_crop_percent", 10.0),
        config.get("save_final_as_uint16", False),
        config.get("re_solve_cropped_tiles", False),
    )


def main():
    """Fonction principale pour lancer l'application ZeMosaic."""
    logger.debug("--- run_zemosaic.py: Entrée dans main() ---")
    args = _parse_cli_args()

    if args.cli:
        if not (args.input_folder and args.output_folder):
            print("Input and output folders are required in CLI mode")
            return
        _run_cli(args)
        return

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
