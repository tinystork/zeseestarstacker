"""
Module principal pour l'interface graphique de GSeestar.
Intègre la prévisualisation avancée et le traitement en file d'attente via QueueManager.
(Version Révisée: Ajout dossiers avant start, Log amélioré, Bouton Ouvrir Sortie)
"""

import gc
import logging
import math
import os
import csv
import platform  # NOUVEL import
import subprocess, shutil, sys

# Import psutil lazily for automatic chunk size estimation
try:
    import psutil
    _psutil_available = True
except Exception:
    psutil = None
    _psutil_available = False

# --- NOUVEAUX IMPORTS SPÉCIFIQUES POUR LE LANCEUR ---

# ----------------------------------------------------
import tempfile  # <-- AJOUTÉ
import threading
import time
import tkinter as tk
import traceback
import re
from pathlib import Path
from tkinter import font as tkFont
from tkinter import messagebox, ttk
from queue import Empty

import numpy as np
from astropy.io import fits
from PIL import Image, ImageTk

from zemosaic import zemosaic_config

from ..queuep.queue_manager import (
    GLOBAL_DRZ_BATCH_VERSION_STRING_ULTRA_DEBUG as APP_VERSION,
)

from .ui_utils import ToolTip
from .boring_stack import read_paths


logger = logging.getLogger(__name__)


def _to_slug(gui_value: str) -> str:
    m = {
        "Reproject and coadd": "reproject_coadd",
        "Reproject": "reproject",
        "Mean": "mean",
        "Reject": "reject",
        "None": "none",
    }
    return m.get(gui_value, gui_value.strip().lower())

logger.debug("-" * 20)
logger.debug("Tentative d'importation de SeestarQueuedStacker...")
try:
    # L'import que vous avez déjà
    from ..queuep.queue_manager import SeestarQueuedStacker

    logger.debug("Import de 'SeestarQueuedStacker' réussi.")
    logger.debug("Type de l'objet importé: %s", type(SeestarQueuedStacker))
    # Vérifier si l'attribut existe sur la CLASSE importée
    logger.debug(
        "La CLASSE importée a 'set_progress_callback'? %s",
        hasattr(SeestarQueuedStacker, "set_progress_callback"),
    )
    logger.debug("Attributs de la CLASSE importée: %s", dir(SeestarQueuedStacker))
except ImportError as imp_err:
    logger.error("ÉCHEC de l'import de SeestarQueuedStacker: %s", imp_err)
    traceback.print_exc()
    # Si l'import échoue, l'application ne peut pas continuer
    sys.exit("Échec de l'importation critique.")
except Exception as gen_err:
    logger.error(
        "Erreur INATTENDUE pendant l'import de SeestarQueuedStacker: %s",
        gen_err,
    )
    traceback.print_exc()
    sys.exit("Échec de l'importation critique.")
# Print separator to clearly show the start of queued stacker import logs
logger.debug("-" * 20)
# Seestar imports
from ..core.image_processing import debayer_image, load_and_validate_fits
from ..core.utils import downsample_image
from ..localization import Localization
from .local_solver_gui import LocalSolverSettingsWindow
from .mosaic_gui import MosaicSettingsWindow

try:
    # Import tools for preview adjustments and auto calculations
    from ..tools.stretch import ColorCorrection, StretchPresets
    from ..tools.stretch import apply_auto_stretch as calculate_auto_stretch
    from ..tools.stretch import apply_auto_white_balance as calculate_auto_wb

    _tools_available = True
except ImportError as tool_err:
    logger.warning("Could not import stretch/color tools: %s.", tool_err)
    _tools_available = False

    # Dummy implementations if tools are missing
    class StretchPresets:
        @staticmethod
        def linear(data, bp=0.0, wp=1.0):
            wp = max(wp, bp + 1e-6)
            return np.clip((data - bp) / (wp - bp), 0, 1)

        @staticmethod
        def asinh(data, scale=1.0, bp=0.0):
            data_s = data - bp
            data_c = np.maximum(data_s, 0.0)
            max_v = np.nanmax(data_c)
            den = np.arcsinh(scale * max_v)
            return np.arcsinh(scale * data_c) / den if den > 1e-6 else np.zeros_like(data)

        @staticmethod
        def logarithmic(data, scale=1.0, bp=0.0):
            data_s = data - bp
            data_c = np.maximum(data_s, 1e-10)
            max_v = np.nanmax(data_c)
            den = np.log1p(scale * max_v)
            return np.log1p(scale * data_c) / den if den > 1e-6 else np.zeros_like(data)

        @staticmethod
        def gamma(data, gamma=1.0):
            return np.power(np.maximum(data, 1e-10), gamma)

    class ColorCorrection:
        @staticmethod
        def white_balance(data, r=1.0, g=1.0, b=1.0):
            if data is None or data.ndim != 3:
                return data
            corr = data.astype(np.float32).copy()
            corr[..., 0] *= r
            corr[..., 1] *= g
            corr[..., 2] *= b
            return np.clip(corr, 0, 1)

    def calculate_auto_stretch(*args, **kwargs):
        return (0.0, 1.0)

    def calculate_auto_wb(*args, **kwargs):
        return (1.0, 1.0, 1.0)


# GUI Component Imports
from .file_handling import FileHandlingManager
from .histogram_widget import HistogramWidget
from .preview import PreviewManager
from .progress import ProgressManager
from .settings import SettingsManager


class SeestarStackerGUI:
    """GUI principale pour Seestar."""

    def __init__(self, initial_input_dir=None, stack_immediately_from=None):
        self.root = tk.Tk()
        self.app_version = APP_VERSION

        # --- DÉBUT CONFIGURATION LOGGER DE BASE (Option B) ---
        # Créer un nom de logger unique basé sur l'ID de l'objet pour éviter les conflits
        # si plusieurs instances étaient créées (bien que peu probable pour une GUI principale).
        self.logger = logging.getLogger(f"SeestarStackerGUI_{id(self)}")
        self.logger.setLevel(logging.DEBUG)  # Capturer tous les niveaux de logs pour ce logger

        # Déterminer le chemin du fichier log (à côté de main_window.py)
        try:
            # Chemin du module courant (main_window.py)
            gui_module_dir = os.path.dirname(os.path.abspath(__file__))
            log_file_name = "seestar_gui_debug.log"
            log_file_path = os.path.join(gui_module_dir, log_file_name)
        except NameError:  # Au cas où __file__ ne serait pas défini (très rare pour un module)
            log_file_path = "seestar_gui_debug.log"  # Fallback dans le dossier courant
            gui_module_dir = "."  # Pour le message de log

        # Vérifier si un FileHandler pour CE fichier log existe déjà pour CE logger
        # pour éviter d'ajouter plusieurs handlers au même logger écrivant dans le même fichier.
        # Ceci est utile si __init__ pouvait être appelé plusieurs fois sur la même instance (généralement non).
        handler_exists = False
        for handler in self.logger.handlers:
            if (
                isinstance(handler, logging.FileHandler)
                and hasattr(handler, "baseFilename")
                and os.path.normpath(handler.baseFilename) == os.path.normpath(log_file_path)
            ):
                handler_exists = True
                break

        if not handler_exists:
            try:
                # 'w' pour écraser le log à chaque lancement (facilite le débogage du dernier run)
                # 'a' pour ajouter au log existant
                fh = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
                fh.setLevel(logging.DEBUG)  # Le handler capture aussi tous les niveaux
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s"
                )
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)
                self.logger.propagate = (
                    False  # Empêcher les messages de remonter au logger root si d'autres configs existent
                )
                self.logger.info(f"Logger pour SeestarStackerGUI initialisé. Logs dans: {log_file_path}")
            except Exception as e_log_init:
                # Si la création du logger échoue, on ne peut pas utiliser self.logger, donc print.
                print(f"ERREUR CRITIQUE: Impossible d'initialiser le FileHandler pour le logger: {e_log_init}")
                print(f"  Les logs de SeestarStackerGUI ne seront pas écrits dans '{log_file_path}'.")
                # On peut créer un logger "nul" pour éviter des AttributeError plus tard,
                # mais les logs ne seront pas sauvegardés.
                self.logger = logging.getLogger("SeestarStackerGUI_Null")
                self.logger.addHandler(logging.NullHandler())
        else:
            self.logger.info(f"FileHandler pour {log_file_path} existe déjà pour ce logger. Pas de nouvel ajout.")
        # --- FIN CONFIGURATION LOGGER DE BASE ---

        self.logger.info("DEBUG (GUI __init__): Initialisation SeestarStackerGUI...")  # Maintenant self.logger existe
        self.logger.info(
            f"DEBUG (GUI __init__): Reçu initial_input_dir='{initial_input_dir}', stack_immediately_from='{stack_immediately_from}'"
        )

        try:
            icon_path = "icon/icon.png"
            if os.path.exists(icon_path):
                icon_image = Image.open(icon_path)
                self.tk_icon = ImageTk.PhotoImage(icon_image)
                self.root.iconphoto(True, self.tk_icon)
                self.logger.info(f"DEBUG (GUI __init__): Icone chargée depuis: {icon_path}")
            else:
                self.logger.warning(f"Warning: Icon file not found at: {icon_path}. Using default icon.")
        except Exception as e:
            self.logger.error(f"Error loading or setting window icon: {e}")

        self.astrometry_api_key_var = tk.StringVar()
        self.last_stack_path = tk.StringVar()
        self.temp_folder_path = tk.StringVar()
        self.localization = Localization("en")
        self.settings = SettingsManager()
        try:
            import inspect

            qs_init_params = inspect.signature(SeestarQueuedStacker.__init__).parameters
            if "settings" in qs_init_params:
                self.queued_stacker = SeestarQueuedStacker(settings=self.settings)
            else:
                self.logger.debug("SeestarQueuedStacker.__init__ ne supporte pas le param\u00e8tre 'settings'.")
                self.queued_stacker = SeestarQueuedStacker()
                # Tenter d'attacher les settings manuellement
                if hasattr(self.queued_stacker, "settings"):
                    self.queued_stacker.settings = self.settings
        except Exception as init_err:
            self.logger.error("Erreur lors de l'initialisation de SeestarQueuedStacker: %s", init_err)
            self.queued_stacker = SeestarQueuedStacker()
            if hasattr(self.queued_stacker, "settings"):
                self.queued_stacker.settings = self.settings
        self.processing = False
        self.thread = None
        self.current_preview_data = None
        self.current_preview_hist_data = None
        self.current_stack_header = None
        self.debounce_timer_id = None
        self.time_per_image = 0
        self.global_start_time = None
        self.additional_folders_to_process = []
        self.tooltips = {}
        self.logger.info("DEBUG (GUI __init__): Dictionnaire self.tooltips initialisé.")
        self.batches_processed_for_preview_refresh = 0
        self.preview_auto_refresh_batch_interval = 10
        # Track when histogram range should be refreshed from sliders
        self._hist_range_update_pending = False
        self.mosaic_mode_active = False
        self.logger.info("DEBUG (GUI __init__): Flag self.mosaic_mode_active initialisé à False.")
        self.mosaic_settings = {}
        self.logger.info("DEBUG (GUI __init__): Flag self.mosaic_mode_active et dict self.mosaic_settings initialisés.")

        # Load shared configuration used by mosaic and solver settings dialogs
        try:
            self.config = zemosaic_config.load_config()
            self.logger.info("DEBUG (GUI __init__): Configuration chargée depuis zemosaic_config.")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.config = {}

        self._trigger_immediate_stack = False
        self._folder_for_immediate_stack = None

        self.use_weighting_var = tk.BooleanVar(value=False)
        self.weight_snr_var = tk.BooleanVar(value=True)
        self.weight_stars_var = tk.BooleanVar(value=True)
        self.snr_exponent_var = tk.DoubleVar(value=1.0)
        self.stars_exponent_var = tk.DoubleVar(value=0.5)
        self.min_weight_var = tk.DoubleVar(value=0.1)

        self._final_stretch_set_by_processing_finished = False  # <--- C'EST LA LIGNE DE LA MÉTHODE 2

        self.init_variables()
        self.last_stack_path.trace_add("write", self._on_last_stack_changed)

        self.settings.load_settings()
        self.language_var.set(self.settings.language)
        self.localization.set_language(self.settings.language)
        self.logger.info(f"DEBUG (GUI __init__): Settings chargés, langue définie sur '{self.settings.language}'.")
        self.logger.info(
            f"DEBUG (GUI __init__): Valeur de self.settings.astrometry_api_key APRES load_settings: '{self.settings.astrometry_api_key}' (longueur: {len(self.settings.astrometry_api_key)})"
        )

        self._auto_stretch_after_id = None
        self._auto_wb_after_id = None
        self.auto_zoom_histogram_var = tk.BooleanVar(value=False)
        self.auto_zoom_histogram_var.trace_add("write", self._update_histogram_autozoom_state)
        self.initial_auto_stretch_done = False

        if stack_immediately_from and isinstance(stack_immediately_from, str) and os.path.isdir(stack_immediately_from):
            self.logger.info(f"INFO (GUI __init__): Stacking immédiat demandé pour: {stack_immediately_from}")
            self.input_path.set(stack_immediately_from)
            self._folder_for_immediate_stack = stack_immediately_from
            self._trigger_immediate_stack = True
            self.logger.info(f"DEBUG (GUI __init__): Flag _trigger_immediate_stack mis à True.")
        elif initial_input_dir and isinstance(initial_input_dir, str) and os.path.isdir(initial_input_dir):
            self.logger.info(
                f"INFO (GUI __init__): Pré-remplissage dossier entrée depuis argument: {initial_input_dir}"
            )
            self.input_path.set(initial_input_dir)

        self.file_handler = FileHandlingManager(self)
        self.create_layout()
        self.last_stack_path.trace_add("write", self._on_last_stack_changed)
        self.init_managers()
        self.logger.info("DEBUG (GUI __init__): Layout créé, managers initialisés.")

        self.settings.apply_to_ui(self)
        try:
            api_key_val_after_apply = self.astrometry_api_key_var.get()
            self.logger.info(
                f"DEBUG (GUI __init__): Valeur de self.astrometry_api_key_var APRES apply_to_ui: '{api_key_val_after_apply}' (longueur: {len(api_key_val_after_apply)})"
            )
        except Exception as e_get_var:
            self.logger.error(
                f"DEBUG (GUI __init__): Erreur lecture self.astrometry_api_key_var après apply_to_ui: {e_get_var}"
            )

        if hasattr(self, "_update_spinbox_from_float"):
            self._update_spinbox_from_float()
        self._update_drizzle_options_state()
        self._toggle_boring_thread()
        self._update_show_folders_button_state()
        self.update_ui_language()

        self.logger.info("--------------------")
        self.logger.info("DEBUG MW __init__: Vérification de self.queued_stacker.set_preview_callback AVANT appel...")
        if hasattr(self.queued_stacker, "set_preview_callback") and callable(self.queued_stacker.set_preview_callback):
            import inspect

            try:
                source_lines, start_line = inspect.getsourcelines(self.queued_stacker.set_preview_callback)
                self.logger.info(
                    f"  Source de self.queued_stacker.set_preview_callback (ligne de début: {start_line}):"
                )
                for i, line_content in enumerate(source_lines[:10]):
                    self.logger.info(f"    L{start_line + i}: {line_content.rstrip()}")
                source_code_str = "".join(source_lines)
                if (
                    "_cleanup_mosaic_panel_stacks_temp()" in source_code_str
                    or "_cleanup_drizzle_batch_outputs()" in source_code_str
                    or "cleanup_unaligned_files()" in source_code_str
                ):
                    self.logger.warning(
                        "  ALERTE MW DEBUG: Un appel _cleanup_ SEMBLE ÊTRE PRÉSENT dans le code source de la méthode set_preview_callback attachée à l'instance !"
                    )
                else:
                    self.logger.info(
                        "  INFO MW DEBUG: Aucun appel _cleanup_ évident dans le code source de la méthode set_preview_callback attachée à l'instance."
                    )
            except Exception as e_inspect:
                self.logger.error(f"  ERREUR MW DEBUG: Erreur inspect: {e_inspect}")
        else:
            self.logger.error(
                "  ERREUR MW DEBUG: self.queued_stacker n'a pas de méthode set_preview_callback ou elle n'est pas callable."
            )
        self.logger.info("--------------------")

        self.gui_event_queue = self.queued_stacker.gui_event_queue
        self.queued_stacker.set_progress_callback(
            lambda m, p=None: self.gui_event_queue.put(
                lambda mm=m, pp=p: self.update_progress_gui(mm, pp)
            )
        )
        self.queued_stacker.set_preview_callback(
            lambda *a, **k: self.gui_event_queue.put(
                lambda aa=a, kk=k: self.update_preview_from_stacker(*aa, **kk)
            )
        )
        self._poll_gui_events()
        self.logger.info("DEBUG (GUI __init__): Callbacks backend connectés.")

        self.root.title(f"{self.tr('title')}  –  {self.app_version}")
        try:
            self.root.geometry(self.settings.window_geometry)
        except tk.TclError:
            self.root.geometry("1200x750")
        self.root.minsize(1100, 650)
        self.root.bind("<Configure>", self._debounce_resize)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self._update_final_scnr_options_state()
        self._update_photutils_bn_options_state()
        self._update_feathering_options_state()
        self._update_low_wht_mask_options_state()

        self.preview_img_count = 0
        self.preview_total_imgs = 0
        self.preview_current_batch = 0
        self.preview_total_batches = 0

        self._try_show_first_input_image()
        self.update_additional_folders_display()

        if self._trigger_immediate_stack:
            self.logger.info("DEBUG (GUI __init__): Planification du lancement immédiat via after(500, ...).")
            self.root.after(500, self._start_immediate_stack)
        else:
            self.logger.info("DEBUG (GUI __init__): Pas de stacking immédiat demandé.")

        self.logger.info("DEBUG (GUI __init__): Initialisation SeestarStackerGUI terminée.")

    # --- DANS LA CLASSE SeestarStackerGUI ---
    # (Ajoutez cette méthode, par exemple après __init__ ou près de start_processing)

    def _start_immediate_stack(self):
        """
        Méthode appelée via 'after' pour démarrer le stacking automatiquement
        si demandé par l'analyseur.
        """
        print("DEBUG (GUI): Exécution de _start_immediate_stack().")  # <-- AJOUTÉ DEBUG
        # Double vérification que le flag est bien positionné et qu'un dossier est défini
        if self._trigger_immediate_stack and self._folder_for_immediate_stack:
            print(
                f"DEBUG (GUI): Conditions remplies. Tentative de lancement de start_processing pour: {self._folder_for_immediate_stack}"
            )  # <-- AJOUTÉ DEBUG
            # Assurer que le dossier d'entrée dans l'UI correspond bien
            # (Normalement déjà fait dans __init__, mais sécurité supplémentaire)
            current_ui_input = self.input_path.get()
            if os.path.normpath(current_ui_input) != os.path.normpath(self._folder_for_immediate_stack):
                print(
                    f"AVERTISSEMENT (GUI): Dossier UI ({current_ui_input}) ne correspond pas au dossier demandé ({self._folder_for_immediate_stack}). Mise à jour UI."
                )
                self.input_path.set(self._folder_for_immediate_stack)
                # On pourrait aussi mettre à jour self.settings.input_folder ici

            # Vérifier si le dossier de sortie est défini, sinon, suggérer un défaut
            if not self.output_path.get():
                default_output = os.path.join(self._folder_for_immediate_stack, "stack_output")
                print(f"INFO (GUI): Dossier de sortie non défini, utilisation défaut: {default_output}")
                self.output_path.set(default_output)
                # Il faudra peut-être créer ce dossier dans start_processing

            # Appeler la méthode start_processing normale
            self.start_processing()
        else:
            print(
                "DEBUG (GUI): _start_immediate_stack() appelé mais conditions non remplies (flag ou dossier manquant)."
            )  # <-- AJOUTÉ DEBUG

        # Réinitialiser le flag pour éviter déclenchement multiple
        self._trigger_immediate_stack = False
        self._folder_for_immediate_stack = None

    # --- DANS LA CLASSE SeestarStackerGUI DANS seestar/gui/main_window.py ---

    def init_variables(self):
        """
        Initialise les variables Tkinter.
        MODIFIED: Ajout de save_as_float32_var.
        """
        print("DEBUG (GUI init_variables V_SaveAsFloat32_1): Initialisation des variables Tkinter...")  # Version Log

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.output_filename_var = tk.StringVar()
        self.reference_image_path = tk.StringVar()
        self.last_stack_path = tk.StringVar()
        self.temp_folder_path = tk.StringVar()
        self.stacking_mode = tk.StringVar(value="kappa-sigma")
        self.kappa = tk.DoubleVar(value=2.5)
        # New unified stacking method variable
        self.stack_method_var = tk.StringVar(value="kappa_sigma")
        # Display variables for localized labels in the comboboxes
        self.stack_method_display_var = tk.StringVar()
        self.stack_norm_display_var = tk.StringVar()
        self.stack_weight_display_var = tk.StringVar()
        self.stack_final_display_var = tk.StringVar()
        # --- New stacking option variables ---
        self.stack_norm_method_var = tk.StringVar(value="none")
        self.stack_weight_method_var = tk.StringVar(value="none")
        self.stack_reject_algo_var = tk.StringVar(value="kappa_sigma")
        # Default final combine method
        self.stack_final_combine_var = tk.StringVar(value="mean")
        self.stacking_kappa_low_var = tk.DoubleVar(value=3.0)
        self.stacking_kappa_high_var = tk.DoubleVar(value=3.0)
        self.stacking_winsor_limits_str_var = tk.StringVar(value="0.05,0.05")
        self.max_hq_mem_var = tk.DoubleVar(value=8)
        self.batch_size = tk.IntVar(value=10)
        self.batch_size.trace_add("write", self._on_batch_size_changed)
        self.boring_thread_var = tk.BooleanVar(value=False)
        self.correct_hot_pixels = tk.BooleanVar(value=True)
        self.hot_pixel_threshold = tk.DoubleVar(value=3.0)
        self.neighborhood_size = tk.IntVar(value=5)
        self.cleanup_temp_var = tk.BooleanVar(value=True)

        self.use_drizzle_var = tk.BooleanVar(value=False)
        self.drizzle_scale_var = tk.StringVar(value="2")
        self.drizzle_wht_threshold_var = tk.DoubleVar(value=0.7)
        self.drizzle_wht_display_var = tk.StringVar(value="70")
        self.drizzle_mode_var = tk.StringVar(value="Final")
        self.drizzle_kernel_var = tk.StringVar(value="square")
        self.drizzle_pixfrac_var = tk.DoubleVar(value=1.0)
        self.use_gpu_var = tk.BooleanVar(value=False)

        self.preview_stretch_method = tk.StringVar(value="Asinh")
        self.preview_black_point = tk.DoubleVar(value=0.01)
        self.preview_white_point = tk.DoubleVar(value=0.99)
        self.preview_gamma = tk.DoubleVar(value=1.0)
        self.preview_r_gain = tk.DoubleVar(value=1.0)
        self.preview_g_gain = tk.DoubleVar(value=1.0)
        self.preview_b_gain = tk.DoubleVar(value=1.0)
        self.preview_brightness = tk.DoubleVar(value=1.0)
        self.preview_contrast = tk.DoubleVar(value=1.0)
        self.preview_saturation = tk.DoubleVar(value=1.0)

        self.language_var = tk.StringVar(value="en")
        self.remaining_files_var = tk.StringVar(value=self.tr("no_files_waiting", default="No files waiting"))
        self.additional_folders_var = tk.StringVar(value=self.tr("no_additional_folders", default="None"))
        default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
        self.aligned_files_var = tk.StringVar(value=default_aligned_fmt.format(count="--"))
        self.remaining_time_var = tk.StringVar(value="--:--:--")
        self.elapsed_time_var = tk.StringVar(value="00:00:00")
        self._after_id_resize = None

        self.apply_chroma_correction_var = tk.BooleanVar(value=True)
        print("DEBUG (GUI init_variables): Variable apply_chroma_correction_var créée.")

        self.apply_final_scnr_var = tk.BooleanVar(value=False)
        self.final_scnr_amount_var = tk.DoubleVar(value=0.8)
        self.final_scnr_preserve_lum_var = tk.BooleanVar(value=True)
        print("DEBUG (GUI init_variables): Variables SCNR Final créées.")

        self.bn_grid_size_str_var = tk.StringVar(value="16x16")
        self.bn_perc_low_var = tk.IntVar(value=5)
        self.bn_perc_high_var = tk.IntVar(value=30)
        self.bn_std_factor_var = tk.DoubleVar(value=1.0)
        self.bn_min_gain_var = tk.DoubleVar(value=0.2)
        self.bn_max_gain_var = tk.DoubleVar(value=7.0)

        self.cb_border_size_var = tk.IntVar(value=25)
        self.cb_blur_radius_var = tk.IntVar(value=8)
        self.cb_min_b_factor_var = tk.DoubleVar(value=0.4)
        self.cb_max_b_factor_var = tk.DoubleVar(value=1.5)

        self.final_edge_crop_percent_var = tk.DoubleVar(value=2.0)

        # Option to crop master tiles before stacking
        self.apply_master_tile_crop_var = tk.BooleanVar(value=False)
        self.master_tile_crop_percent_var = tk.DoubleVar(value=18.0)

        # --- nouveaux toggles Expert ---
        self.apply_bn_var = tk.BooleanVar(value=True)
        self.apply_cb_var = tk.BooleanVar(value=True)
        self.apply_final_crop_var = tk.BooleanVar(value=True)

        print("DEBUG (GUI init_variables): Variables Onglet Expert (BN, CB, Crop) créées.")

        self.apply_photutils_bn_var = tk.BooleanVar(value=False)
        self.photutils_bn_box_size_var = tk.IntVar(value=128)
        self.photutils_bn_filter_size_var = tk.IntVar(value=5)
        self.photutils_bn_sigma_clip_var = tk.DoubleVar(value=3.0)
        self.photutils_bn_exclude_percentile_var = tk.DoubleVar(value=98.0)

        self.apply_feathering_var = tk.BooleanVar(value=False)
        self.feather_blur_px_var = tk.IntVar(value=256)
        self.apply_batch_feathering_var = tk.BooleanVar(value=True)
        print("DEBUG (GUI init_variables): Variables Feathering créées (apply_feathering_var, feather_blur_px_var).")

        print("DEBUG (GUI init_variables): Variables pour Photutils Background Subtraction créées.")

        self.apply_low_wht_mask_var = tk.BooleanVar(value=False)
        self.low_wht_pct_var = tk.IntVar(value=5)
        self.low_wht_soften_px_var = tk.IntVar(value=128)
        print("DEBUG (GUI init_variables): Variables Low WHT Mask créées.")

        # --- NOUVELLE VARIABLE TKINTER POUR L'OPTION DE SAUVEGARDE ---
        self.save_as_float32_var = tk.BooleanVar(value=False)  # Défaut à False (donc uint16)
        print(
            f"DEBUG (GUI init_variables): Variable save_as_float32_var créée (valeur initiale: {self.save_as_float32_var.get()})."
        )
        self.preserve_linear_output_var = tk.BooleanVar(value=False)
        print(
            f"DEBUG (GUI init_variables): Variable preserve_linear_output_var créée (valeur initiale: {self.preserve_linear_output_var.get()})."
        )
        self.use_third_party_solver_var = tk.BooleanVar(value=True)
        print(
            f"DEBUG (GUI init_variables): Variable use_third_party_solver_var créée (valeur initiale: {self.use_third_party_solver_var.get()})."
        )
        self.reproject_between_batches_var = tk.BooleanVar(value=False)
        # Separate toggle for final reproject+coadd
        self.reproject_coadd_var = tk.BooleanVar(value=False)
        self.ansvr_host_port_var = tk.StringVar(value="127.0.0.1:8080")

        self.astrometry_solve_field_dir_var = tk.StringVar(value="")

        # --- FIN NOUVELLE VARIABLE ---

        print("DEBUG (GUI init_variables V_SaveAsFloat32_1): Fin initialisation variables Tkinter.")  # Version Log

    #######################################################################################################################

    def _update_drizzle_options_state(self):
        """Active ou désactive les options Drizzle (mode, échelle, seuil WHT)."""
        try:
            # Déterminer l'état souhaité (NORMAL ou DISABLED) basé sur la checkbox Drizzle
            global_drizzle_enabled = self.use_drizzle_var.get()
            state = tk.NORMAL if global_drizzle_enabled else tk.DISABLED

            # --- NOUVEAU : Désactiver aussi l'échelle/WHT si mode Incrémental ---
            # Certaines options peuvent ne pas être pertinentes en mode incrémental
            # Pour l'instant, on les laisse actives dans les deux modes si Drizzle est coché.
            # On pourrait ajouter une logique ici plus tard si nécessaire.
            # Par exemple:
            # is_incremental = self.drizzle_mode_var.get() == "Incremental"
            # scale_state = state if not is_incremental else tk.DISABLED # Désactiver échelle si incrémental?

            # Liste des widgets qui dépendent de l'activation GLOBALE de Drizzle
            widgets_to_toggle = [
                # Widgets pour le CHOIX DU MODE (Nouveau)
                getattr(self, "drizzle_mode_label", None),
                getattr(self, "drizzle_radio_final", None),
                getattr(self, "drizzle_radio_incremental", None),
                # Widgets pour l'ÉCHELLE Drizzle (Existant)
                getattr(self, "drizzle_scale_label", None),
                getattr(self, "drizzle_radio_2x", None),
                getattr(self, "drizzle_radio_3x", None),
                getattr(self, "drizzle_radio_4x", None),
                # getattr(self, 'drizzle_scale_combo', None), # Si Combobox
                # Widgets pour le SEUIL WHT (Existant)
                getattr(self, "drizzle_wht_label", None),
                getattr(self, "drizzle_wht_spinbox", None),
                # Kernel Drizzle
                getattr(self, "drizzle_kernel_label", None),
                getattr(self, "drizzle_kernel_combo", None),
                # Pixfrac Drizzle
                getattr(self, "drizzle_pixfrac_label", None),
                getattr(self, "drizzle_pixfrac_spinbox", None),
                getattr(self, "use_gpu_check", None),
                # --- FIN DES NOUVELLES LIGNES ---
            ]

            # Boucle pour appliquer l'état (NORMAL ou DISABLED) à chaque widget de la liste
            for widget in widgets_to_toggle:
                # Vérifier si le widget existe réellement avant de tenter de le configurer
                if widget and hasattr(widget, "winfo_exists") and widget.winfo_exists():
                    # Appliquer l'état global (activé/désactivé par la checkbox principale)
                    widget.config(state=state)

        except tk.TclError:
            # Ignorer les erreurs si un widget n'existe pas (peut arriver pendant l'init)
            pass
        except AttributeError:
            # Ignorer si un attribut n'existe pas encore (peut arriver pendant l'init)
            pass
        except Exception as e:
            # Loguer les erreurs inattendues
            print(f"Error in _update_drizzle_options_state: {e}")
            # traceback.print_exc(limit=1) # Décommenter pour débogage

    def _update_final_scnr_options_state(self, *args):
        """Active ou désactive les options SCNR détaillées (intensité, préserver lum)."""
        try:
            scnr_active = self.apply_final_scnr_var.get()
            new_state = tk.NORMAL if scnr_active else tk.DISABLED

            # Widgets SCNR à contrôler
            # Le groupe slider/spinbox pour amount
            if hasattr(self, "scnr_amount_ctrls"):
                amount_widgets = [
                    self.scnr_amount_ctrls.get("slider"),
                    self.scnr_amount_ctrls.get("spinbox"),
                    self.scnr_amount_ctrls.get("label"),  # Griser le label aussi
                ]
                for widget in amount_widgets:
                    if widget and hasattr(widget, "winfo_exists") and widget.winfo_exists():
                        widget.config(state=new_state)

            # Checkbox pour préserver la luminance
            if hasattr(self, "final_scnr_preserve_lum_check") and self.final_scnr_preserve_lum_check.winfo_exists():
                self.final_scnr_preserve_lum_check.config(state=new_state)

            # print(f"DEBUG: État options SCNR mis à jour vers: {'NORMAL' if scnr_active else 'DISABLED'}") # Debug
        except tk.TclError:
            # Peut arriver si les widgets sont détruits pendant l'appel
            pass
        except AttributeError:
            # Peut arriver si les attributs n'existent pas encore pendant l'initialisation
            # print("DEBUG: AttributeError dans _update_final_scnr_options_state (probablement pendant init)") # Debug
            pass
        except Exception as e:
            print(f"ERREUR inattendue dans _update_final_scnr_options_state: {e}")
            traceback.print_exc(limit=1)

    def _toggle_boring_thread(self):
        """Synchronize ``batch_size`` with the boring-thread checkbox."""
        try:
            if self.boring_thread_var.get():
                if self.batch_size.get() != 1:
                    self.batch_size.set(1)
                if hasattr(self, "batch_spinbox") and self.batch_spinbox.winfo_exists():
                    self.batch_spinbox.config(state=tk.DISABLED)
            else:
                if hasattr(self, "batch_spinbox") and self.batch_spinbox.winfo_exists():
                    self.batch_spinbox.config(state=tk.NORMAL)
                if self.batch_size.get() == 1:
                    self.batch_size.set(0)
        except tk.TclError:
            pass

    def _on_batch_size_changed(self, *args):
        """Toggle boring-thread mode when the batch size equals 1."""
        try:
            val = self.batch_size.get()
        except tk.TclError:
            return
        if val == 1 and not self.boring_thread_var.get():
            self.boring_thread_var.set(True)
            self._toggle_boring_thread()
        elif val != 1 and self.boring_thread_var.get():
            self.boring_thread_var.set(False)
            self._toggle_boring_thread()

    # Assurez-vous d'appeler cette méthode aussi dans SettingsManager.apply_to_ui
    # et dans SeestarStackerGUI.__init__ après avoir appliqué les settings
    # Exemple dans SeestarStackerGUI.__init__ (après self.settings.apply_to_ui(self)):
    # self._update_final_scnr_options_state()
    # Et dans SettingsManager.apply_to_ui (après avoir set les variables SCNR):
    # if hasattr(gui_instance, '_update_final_scnr_options_state'):
    #    gui_instance._update_final_scnr_options_state()

    #########################################################################################################################

    def init_managers(self):
        """Initialise les gestionnaires (Progress, Preview, FileHandling)."""
        # Progress Manager
        if hasattr(self, "progress_bar") and hasattr(self, "status_text"):
            self.progress_manager = ProgressManager(
                self.progress_bar,
                self.status_text,
                self.remaining_time_var,
                self.elapsed_time_var,
            )
        else:
            print("Error: Progress widgets not found for ProgressManager initialization.")

        # Preview Manager
        if hasattr(self, "preview_canvas"):
            self.preview_manager = PreviewManager(self.preview_canvas)
        else:
            print("Error: Preview canvas not found for PreviewManager initialization.")

        # Histogram Widget Callback (if widget exists)
        if hasattr(self, "histogram_widget") and self.histogram_widget:
            self.histogram_widget.range_change_callback = self.update_stretch_from_histogram
        else:
            print("Error: HistogramWidget reference not found after create_layout.")

        # File Handler (should already be created in __init__)
        if not hasattr(self, "file_handler"):
            print("Error: FileHandlingManager not initialized.")

        # Show initial state in preview area
        self.show_initial_preview()
        # Update additional folders display initially
        self.update_additional_folders_display()

    def show_initial_preview(self):
        """Affiche un état initial dans la zone d'aperçu."""
        if hasattr(self, "preview_manager") and self.preview_manager:
            self.preview_manager.clear_preview(self.tr("Select input/output folders."))
        if hasattr(self, "histogram_widget") and self.histogram_widget:
            self.histogram_widget.plot_histogram(None)  # Clear histogram

    def tr(self, key, default=None):
        """Raccourci pour la localisation."""
        return self.localization.get(key, default=default)

    def _convert_spinbox_percent_to_float(self, *args):
        """Lit la variable d'affichage (%) et met à jour la variable de stockage (0-1)."""
        try:
            # --- MODIFIÉ: Lire depuis la variable d'affichage ---
            display_value_str = self.drizzle_wht_display_var.get()
            percent_value = float(display_value_str)
            # --- FIN MODIFICATION ---

            # La conversion et le clip restent les mêmes
            float_value = np.clip(percent_value / 100.0, 0.01, 1.0)  # Assurer 0.01-1.0

            # Mettre à jour directement la variable de stockage (0.0-1.0)
            self.drizzle_wht_threshold_var.set(float_value)
            # print(f"DEBUG (Spinbox Cmd): Display='{display_value_str}', FloatSet={float_value:.3f}") # <-- DEBUG Optionnel

        except (ValueError, tk.TclError, AttributeError) as e:
            # Ignorer erreurs de conversion ou si les variables n'existent pas encore
            print(f"DEBUG (Spinbox Cmd): Ignored error during conversion: {e}")  # <-- AJOUTÉ DEBUG
            pass

    #    def _update_spinbox_from_float(self, *args):
    #        """Lit la variable (0-1) et met à jour le Spinbox (%). Appelé manuellement."""
    #        try:
    #            # Vérifier si le widget existe avant d'y accéder
    #            if hasattr(self, 'drizzle_wht_spinbox') and self.drizzle_wht_spinbox.winfo_exists():
    #                float_value = self.drizzle_wht_threshold_var.get()
    #                percent_value = round(float_value * 100.0)
    #                # Mettre à jour le Spinbox sans déclencher sa propre commande
    #                self.drizzle_wht_spinbox.config(textvariable=tk.StringVar(value=f"{percent_value:.0f}"))
    #                # Remettre la liaison à la variable correcte pour la lecture future
    #                self.drizzle_wht_spinbox.config(textvariable=self.drizzle_wht_threshold_var) # Reconnecter pour la saisie? Non, on utilise command.
    #                # Juste mettre la valeur est plus simple:
    #                # self.drizzle_wht_spinbox.delete(0, tk.END)
    #                # self.drizzle_wht_spinbox.insert(0, f"{percent_value:.0f}") # Ceci est plus sûr
    #                # Encore plus simple: utiliser set() si Spinbox le supporte bien
    #                self.drizzle_wht_spinbox.set(f"{percent_value:.0f}")

    #       except (tk.TclError, AttributeError):
    # Peut arriver si appelé avant que spinbox soit prêt ou après destruction
    #            pass

    # Nécéssite d'ajouter self._trace_id_wht = None dans __init__
    # et de lier la trace après la création du spinbox
    # -> Simplifions : On va juste appeler la mise à jour manuellement après chargement
    # Supprimer les lignes concernant _trace_id_wht et trace_add/trace_remove dans les 2 fonctions ci-dessus
    # Et dans apply_to_ui, après avoir set la variable, appeler _update_spinbox_from_float()
    ###########################################################################################################################

    # --- DANS LA CLASSE SeestarStackerGUI DANS seestar/gui/main_window.py ---

    def create_layout(self):
        """Crée la disposition des widgets avec la scrollbar pour le panneau gauche et le SCNR réorganisé."""
        print("DEBUG (GUI create_layout V_SaveAsFloat32_1): Début création layout...")  # Version Log

        # --- Cadre Principal et PanedWindow ---
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        print("DEBUG (GUI create_layout): Cadre principal et PanedWindow créés.")

        # --- Panneau Gauche avec Scrollbar Intégrée ---
        # ... (code inchangé pour le panneau gauche scrollable) ...
        left_canvas_container = ttk.Frame(paned_window, width=450)
        paned_window.add(left_canvas_container, weight=1)
        self.left_scrollbar = ttk.Scrollbar(left_canvas_container, orient="vertical")
        self.left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_background_color = self.root.cget("bg")
        self.left_scrollable_canvas = tk.Canvas(
            left_canvas_container,
            highlightthickness=0,
            bg=canvas_background_color,
            yscrollcommand=self.left_scrollbar.set,
        )
        self.left_scrollable_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.left_scrollbar.config(command=self.left_scrollable_canvas.yview)
        self.left_content_frame = ttk.Frame(self.left_scrollable_canvas)
        self.left_content_frame_id_on_canvas = self.left_scrollable_canvas.create_window(
            (0, 0),
            window=self.left_content_frame,
            anchor="nw",
            tags="self.left_content_frame_tag",
        )

        def _on_left_content_frame_configure(event):
            if self.left_scrollable_canvas.winfo_width() > 1:
                self.left_scrollable_canvas.itemconfig(
                    self.left_content_frame_id_on_canvas,
                    width=self.left_scrollable_canvas.winfo_width(),
                )
            self.left_scrollable_canvas.config(scrollregion=self.left_scrollable_canvas.bbox("all"))

        self.left_content_frame.bind("<Configure>", _on_left_content_frame_configure)
        self.left_scrollable_canvas.bind(
            "<Configure>",
            lambda e, c=self.left_scrollable_canvas, i=self.left_content_frame_id_on_canvas: (
                c.itemconfig(i, width=e.width) if e.width > 1 else None
            ),
        )
        print("DEBUG (GUI create_layout): Panneau gauche scrollable configuré.")

        # --- Panneau Droit (pour Aperçu, Histogramme, Boutons de Contrôle) ---
        right_frame = ttk.Frame(paned_window, width=750)
        paned_window.add(right_frame, weight=3)
        print("DEBUG (GUI create_layout): Panneau droit créé.")

        # =======================================================================
        # --- Contenu du Panneau Gauche (dans self.left_content_frame) ---
        # =======================================================================
        print("DEBUG (GUI create_layout): Début remplissage panneau gauche...")

        # 1. Sélection Langue
        # ... (inchangé) ...
        lang_frame = ttk.Frame(self.left_content_frame)
        lang_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 5), padx=5)
        ttk.Label(lang_frame, text="Language/Langue:").pack(side=tk.LEFT, padx=(0, 5))
        self.language_combo = ttk.Combobox(
            lang_frame,
            textvariable=self.language_var,
            values=("en", "fr"),
            width=8,
            state="readonly",
        )
        self.language_combo.pack(side=tk.LEFT)
        self.language_combo.bind("<<ComboboxSelected>>", self.change_language)
        print("DEBUG (GUI create_layout): Sélection langue créée.")

        # 2. Notebook pour les Onglets d'Options
        self.control_notebook = ttk.Notebook(self.left_content_frame)
        self.control_notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 5), padx=5)
        print("DEBUG (GUI create_layout): Notebook de contrôle créé.")

        # --- Onglet Empilement (Index 0) ---
        tab_stacking = ttk.Frame(self.control_notebook)
        print("DEBUG (GUI create_layout): Frame pour onglet Empilement créé.")
        # ... (Votre code complet pour le contenu de tab_stacking ici, inchangé) ...
        self.folders_frame = ttk.LabelFrame(tab_stacking, text=self.tr("Folders"))
        self.folders_frame.pack(fill=tk.X, pady=5, padx=5)
        in_subframe = ttk.Frame(self.folders_frame)
        in_subframe.pack(fill=tk.X, padx=5, pady=(5, 2))
        self.input_label = ttk.Label(
            in_subframe,
            text=self.tr("input_folder", default="Input:"),
            width=10,
            anchor="w",
        )
        self.input_label.pack(side=tk.LEFT)
        self.browse_input_button = ttk.Button(
            in_subframe,
            text=self.tr("browse_input_button", default="Browse..."),
            command=self.file_handler.browse_input,
            width=10,
        )
        self.browse_input_button.pack(side=tk.RIGHT)
        self.input_entry = ttk.Entry(in_subframe, textvariable=self.input_path)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.input_entry.bind("<FocusOut>", self._update_show_folders_button_state)
        self.input_entry.bind("<KeyRelease>", self._update_show_folders_button_state)
        out_subframe = ttk.Frame(self.folders_frame)
        out_subframe.pack(fill=tk.X, padx=5, pady=(2, 5))
        self.output_label = ttk.Label(
            out_subframe,
            text=self.tr("output_folder", default="Output:"),
            width=10,
            anchor="w",
        )
        self.output_label.pack(side=tk.LEFT)
        self.browse_output_button = ttk.Button(
            out_subframe,
            text=self.tr("browse_output_button", default="Browse..."),
            command=self.file_handler.browse_output,
            width=10,
        )
        self.browse_output_button.pack(side=tk.RIGHT)
        self.output_entry = ttk.Entry(out_subframe, textvariable=self.output_path)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        fname_frame = ttk.Frame(self.folders_frame)
        fname_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        self.output_filename_label = ttk.Label(
            fname_frame,
            text=self.tr("output_filename_label", default="Filename:"),
            width=10,
            anchor="w",
        )
        self.output_filename_label.pack(side=tk.LEFT)
        self.output_filename_entry = ttk.Entry(fname_frame, textvariable=self.output_filename_var)
        self.output_filename_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ref_frame = ttk.Frame(self.folders_frame)
        ref_frame.pack(fill=tk.X, padx=5, pady=(2, 5))
        self.reference_label = ttk.Label(
            ref_frame,
            text=self.tr("reference_image", default="Reference "),
            width=10,
            anchor="w",
        )
        self.reference_label.pack(side=tk.LEFT)
        self.browse_ref_button = ttk.Button(
            ref_frame,
            text=self.tr("browse_ref_button", default="Browse..."),
            command=self.file_handler.browse_reference,
            width=10,
        )
        self.browse_ref_button.pack(side=tk.RIGHT)
        self.ref_entry = ttk.Entry(ref_frame, textvariable=self.reference_image_path)
        self.ref_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        last_frame = ttk.Frame(self.folders_frame)
        last_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        last_lbl = ttk.Label(
            last_frame,
            text=self.tr("last_stack_treated", default="Last stack :"),
            width=10,
            anchor="w",
        )
        last_lbl.pack(side=tk.LEFT)
        last_browse = ttk.Button(last_frame, text="…", command=self.file_handler.browse_last_stack)
        last_browse.pack(side=tk.RIGHT)
        last_entry = ttk.Entry(last_frame, textvariable=self.last_stack_path, width=42)
        last_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))

        temp_frame = ttk.Frame(self.folders_frame)
        temp_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        temp_lbl = ttk.Label(
            temp_frame,
            text=self.tr("temporary_folder", default="Temporary:"),
            width=10,
            anchor="w",
        )
        temp_lbl.pack(side=tk.LEFT)
        temp_browse = ttk.Button(temp_frame, text="…", command=self.file_handler.browse_temp_folder)
        temp_browse.pack(side=tk.RIGHT)
        temp_entry = ttk.Entry(temp_frame, textvariable=self.temp_folder_path, width=42)
        temp_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))

        crop_frame = ttk.Frame(tab_stacking)
        crop_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        self.crop_master_check = ttk.Checkbutton(
            crop_frame,
            text=self.tr("crop_master_tiles_label", default="Crop master tiles"),
            variable=self.apply_master_tile_crop_var,
            command=self._update_master_tile_crop_state,
        )
        self.crop_master_check.grid(row=0, column=0, sticky=tk.W)
        ttk.Label(
            crop_frame,
            text=self.tr("crop_percent_side_label", default="Crop % per side"),
        ).grid(row=0, column=1, sticky=tk.W, padx=(10, 2))
        self.master_tile_crop_spinbox = ttk.Spinbox(
            crop_frame,
            from_=0.0,
            to=25.0,
            increment=0.5,
            textvariable=self.master_tile_crop_percent_var,
            width=6,
            format="%.1f",
        )
        self.master_tile_crop_spinbox.grid(row=0, column=2, sticky=tk.W)
        self._update_master_tile_crop_state()
        self.options_frame = ttk.LabelFrame(tab_stacking, text="Stacking Options")
        self.options_frame.pack(fill=tk.X, pady=5, padx=5)

        norm_frame = ttk.Frame(self.options_frame)
        norm_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        self.norm_method_label = ttk.Label(
            norm_frame,
            text=self.tr("stacking_norm_method_label", default="Normalization:"),
        )
        self.norm_method_label.pack(side=tk.LEFT)
        self.stack_norm_combo = ttk.Combobox(
            norm_frame,
            textvariable=self.stack_norm_display_var,
            state="readonly",
            width=15,
        )
        self.stack_norm_combo.pack(side=tk.LEFT, padx=(5, 0))
        self.stack_norm_combo.bind("<<ComboboxSelected>>", self._on_norm_combo_change)

        weight_frame = ttk.Frame(self.options_frame)
        weight_frame.pack(fill=tk.X, padx=5, pady=(2, 0))
        self.weight_method_label = ttk.Label(
            weight_frame,
            text=self.tr("stacking_weight_method_label", default="Weighting:"),
        )
        self.weight_method_label.pack(side=tk.LEFT)
        self.stack_weight_combo = ttk.Combobox(
            weight_frame,
            textvariable=self.stack_weight_display_var,
            state="readonly",
            width=15,
        )
        self.stack_weight_combo.pack(side=tk.LEFT, padx=(5, 0))
        self.stack_weight_combo.bind("<<ComboboxSelected>>", self._on_weight_combo_change)

        kappa_frame = ttk.Frame(self.options_frame)
        kappa_frame.pack(fill=tk.X, padx=20, pady=(2, 0))
        self.kappa_low_label = ttk.Label(kappa_frame, text="Kappa Low:")
        self.kappa_low_label.pack(side=tk.LEFT, padx=(0, 2))
        self.kappa_low_spinbox = ttk.Spinbox(
            kappa_frame,
            from_=0.1,
            to=10.0,
            increment=0.1,
            textvariable=self.stacking_kappa_low_var,
            width=6,
        )
        self.kappa_low_spinbox.pack(side=tk.LEFT, padx=(0, 10))
        self.kappa_high_label = ttk.Label(kappa_frame, text="Kappa High:")
        self.kappa_high_label.pack(side=tk.LEFT, padx=(0, 2))
        self.kappa_high_spinbox = ttk.Spinbox(
            kappa_frame,
            from_=0.1,
            to=10.0,
            increment=0.1,
            textvariable=self.stacking_kappa_high_var,
            width=6,
        )
        self.kappa_high_spinbox.pack(side=tk.LEFT, padx=(0, 5))

        winsor_frame = ttk.Frame(self.options_frame)
        winsor_frame.pack(fill=tk.X, padx=20, pady=(2, 0))
        self.winsor_limits_label = ttk.Label(
            winsor_frame,
            text=self.tr("stacking_winsor_limits_label", default="Winsor Limits:"),
        )
        self.winsor_limits_label.pack(side=tk.LEFT, padx=(0, 2))
        self.winsor_limits_entry = ttk.Entry(winsor_frame, textvariable=self.stacking_winsor_limits_str_var, width=10)
        self.winsor_limits_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.winsor_note_label = ttk.Label(
            winsor_frame,
            text=self.tr("stacking_winsor_note", default="(e.g., 0.05,0.05 for 5% each side)"),
        )
        self.winsor_note_label.pack(side=tk.LEFT, padx=(5, 0))

        final_frame = ttk.Frame(self.options_frame)
        final_frame.pack(fill=tk.X, padx=5, pady=(2, 0))
        self.final_combine_label = ttk.Label(
            final_frame,
            text=self.tr("stacking_final_combine_label", default="Final Combine:"),
        )
        self.final_combine_label.pack(side=tk.LEFT)
        self.stack_final_combo = ttk.Combobox(
            final_frame,
            textvariable=self.stack_final_display_var,
            state="readonly",
            width=15,
        )
        self.stack_final_combo.pack(side=tk.LEFT, padx=(5, 0))
        self.stack_final_combo.bind("<<ComboboxSelected>>", self._on_final_combo_change)

        self.hq_ram_limit_label_widget = tk.Label(
            final_frame, text=self.tr("hq_ram_limit_label", default="HQ RAM limit (GB)")
        )
        self.hq_ram_limit_label_widget.pack(side=tk.LEFT, padx=(10, 2))
        self.hq_ram_limit_spinbox = tk.Spinbox(
            final_frame,
            from_=1,
            to=64,
            increment=1,
            width=5,
            textvariable=self.max_hq_mem_var,
        )
        self.hq_ram_limit_spinbox.pack(side=tk.LEFT)

        # Mapping between internal keys and displayed labels for normalization, weighting and final combine
        self.norm_keys = ["none", "linear_fit", "sky_mean"]
        self.norm_key_to_label = {}
        self.norm_label_to_key = {}
        for k in self.norm_keys:
            label = self.tr(f"norm_method_{k}", default=k.replace("_", " ").title())
            self.norm_key_to_label[k] = label
            self.norm_label_to_key[label] = k
        self.stack_norm_combo["values"] = list(self.norm_key_to_label.values())
        self.stack_norm_display_var.set(
            self.norm_key_to_label.get(self.stack_norm_method_var.get(), self.stack_norm_method_var.get())
        )

        self.weight_keys = ["none", "noise_variance", "noise_fwhm", "snr", "stars"]
        self.weight_key_to_label = {}
        self.weight_label_to_key = {}
        for k in self.weight_keys:
            label = self.tr(f"weight_method_{k}", default=k.replace("_", " ").title())
            self.weight_key_to_label[k] = label
            self.weight_label_to_key[label] = k
        self.stack_weight_combo["values"] = list(self.weight_key_to_label.values())
        self.stack_weight_display_var.set(
            self.weight_key_to_label.get(self.stack_weight_method_var.get(), self.stack_weight_method_var.get())
        )

        self.final_keys = [
            "mean",
            "median",
            "winsorized_sigma_clip",
            "reproject",
            "reproject_coadd",
        ]
        self.final_key_to_label = {}
        self.final_label_to_key = {}
        for k in self.final_keys:
            label = self.tr(f"combine_method_{k}", default=k.replace("_", " ").title())
            self.final_key_to_label[k] = label
            self.final_label_to_key[label] = k
        self.stack_final_combo["values"] = list(self.final_key_to_label.values())
        if self.reproject_between_batches_var.get():
            self.stack_final_display_var.set(self.final_key_to_label.get("reproject", "reproject"))
        elif getattr(self, "reproject_coadd_var", tk.BooleanVar()).get():
            self.stack_final_display_var.set(self.final_key_to_label.get("reproject_coadd", "reproject_coadd"))
        else:
            self.stack_final_display_var.set(
                self.final_key_to_label.get(
                    self.stack_final_combine_var.get(),
                    self.stack_final_combine_var.get(),
                )
            )

        method_kappa_scnr_frame = ttk.Frame(self.options_frame)
        method_kappa_scnr_frame.pack(fill=tk.X, padx=0, pady=(5, 0))
        mk_line1_frame = ttk.Frame(method_kappa_scnr_frame)
        mk_line1_frame.pack(fill=tk.X, padx=5)
        self.stack_method_label = ttk.Label(mk_line1_frame, text=self.tr("stack_method_label", default="Method:"))
        self.stack_method_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 2))
        self.method_combo = ttk.Combobox(
            mk_line1_frame,
            textvariable=self.stack_method_display_var,
            width=22,
            state="readonly",
        )
        self.method_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 8))
        self.method_combo.bind("<<ComboboxSelected>>", self._on_method_combo_change)

        # Mapping between internal keys and displayed labels for the stacking method
        self.method_keys = [
            "mean",
            "median",
            "kappa_sigma",
            "winsorized_sigma_clip",
            "linear_fit_clip",
        ]
        self.method_key_to_label = {}
        self.method_label_to_key = {}
        for k in self.method_keys:
            label = self.tr(f"method_{k}", default=k.replace("_", " ").title())
            self.method_key_to_label[k] = label
            self.method_label_to_key[label] = k
        self.method_combo["values"] = list(self.method_key_to_label.values())
        # Set display variable based on current internal value
        self.stack_method_display_var.set(
            self.method_key_to_label.get(self.stack_method_var.get(), self.stack_method_var.get())
        )
        scnr_options_subframe_in_stacking = ttk.Frame(method_kappa_scnr_frame)
        scnr_options_subframe_in_stacking.pack(fill=tk.X, padx=5, pady=(5, 2))
        self.apply_final_scnr_check = ttk.Checkbutton(
            scnr_options_subframe_in_stacking,
            text=self.tr("apply_final_scnr_label", default="Apply Final SCNR (Green)"),
            variable=self.apply_final_scnr_var,
            command=self._update_final_scnr_options_state,
        )
        self.apply_final_scnr_check.pack(anchor=tk.W, pady=(0, 2))
        self.scnr_params_frame = ttk.Frame(scnr_options_subframe_in_stacking)
        self.scnr_params_frame.pack(fill=tk.X, padx=(20, 0))
        self.scnr_amount_ctrls = self._create_slider_spinbox_group(
            self.scnr_params_frame,
            "final_scnr_amount_label",
            min_val=0.0,
            max_val=1.0,
            step=0.05,
            tk_var=self.final_scnr_amount_var,
            callback=None,
        )
        self.final_scnr_preserve_lum_check = ttk.Checkbutton(
            self.scnr_params_frame,
            text=self.tr("final_scnr_preserve_lum_label", default="Preserve Luminosity (SCNR)"),
            variable=self.final_scnr_preserve_lum_var,
        )
        self.final_scnr_preserve_lum_check.pack(anchor=tk.W, pady=(0, 5))
        batch_frame = ttk.Frame(self.options_frame)
        batch_frame.pack(fill=tk.X, padx=5, pady=(5, 5))
        self.batch_size_label = ttk.Label(batch_frame, text="Batch Size:")
        self.batch_size_label.pack(side=tk.LEFT, padx=(0, 5))
        self.batch_spinbox = ttk.Spinbox(
            batch_frame,
            from_=0,
            to=9999,
            increment=1,
            textvariable=self.batch_size,
            width=5,
        )
        self.batch_spinbox.pack(side=tk.LEFT)
        self.boring_thread_check = ttk.Checkbutton(
            batch_frame,
            text=self.tr("enable_boring_thread", default="Threaded Boring Stack"),
            variable=self.boring_thread_var,
            command=self._toggle_boring_thread,
        )
        self.boring_thread_check.pack(side=tk.LEFT, padx=5)
        self.drizzle_options_frame = ttk.LabelFrame(tab_stacking, text="Drizzle Options")
        self.drizzle_options_frame.pack(fill=tk.X, pady=5, padx=5)
        self.drizzle_check = ttk.Checkbutton(
            self.drizzle_options_frame,
            text="Enable Drizzle",
            variable=self.use_drizzle_var,
            command=self._update_drizzle_options_state,
        )
        self.drizzle_check.pack(anchor=tk.W, padx=5, pady=(5, 2))
        self.drizzle_mode_frame = ttk.Frame(self.drizzle_options_frame)
        self.drizzle_mode_frame.pack(fill=tk.X, padx=(20, 5), pady=(2, 5))
        self.drizzle_mode_label = ttk.Label(self.drizzle_mode_frame, text="Mode:")
        self.drizzle_mode_label.pack(side=tk.LEFT, padx=(0, 5))
        self.drizzle_radio_final = ttk.Radiobutton(
            self.drizzle_mode_frame,
            text="Final",
            variable=self.drizzle_mode_var,
            value="Final",
            command=self._update_drizzle_options_state,
        )
        self.drizzle_radio_incremental = ttk.Radiobutton(
            self.drizzle_mode_frame,
            text="Incremental",
            variable=self.drizzle_mode_var,
            value="Incremental",
            command=self._update_drizzle_options_state,
        )
        self.drizzle_radio_final.pack(side=tk.LEFT, padx=3)
        self.drizzle_radio_incremental.pack(side=tk.LEFT, padx=3)
        self.drizzle_scale_frame = ttk.Frame(self.drizzle_options_frame)
        self.drizzle_scale_frame.pack(fill=tk.X, padx=(20, 5), pady=(0, 5))
        self.drizzle_scale_label = ttk.Label(self.drizzle_scale_frame, text="Scale:")
        self.drizzle_scale_label.pack(side=tk.LEFT, padx=(0, 5))
        self.drizzle_radio_2x = ttk.Radiobutton(
            self.drizzle_scale_frame,
            text="x2",
            variable=self.drizzle_scale_var,
            value="2",
        )
        self.drizzle_radio_3x = ttk.Radiobutton(
            self.drizzle_scale_frame,
            text="x3",
            variable=self.drizzle_scale_var,
            value="3",
        )
        self.drizzle_radio_4x = ttk.Radiobutton(
            self.drizzle_scale_frame,
            text="x4",
            variable=self.drizzle_scale_var,
            value="4",
        )
        self.drizzle_radio_2x.pack(side=tk.LEFT, padx=3)
        self.drizzle_radio_3x.pack(side=tk.LEFT, padx=3)
        self.drizzle_radio_4x.pack(side=tk.LEFT, padx=3)
        wht_frame = ttk.Frame(self.drizzle_options_frame)
        wht_frame.pack(fill=tk.X, padx=(20, 5), pady=(5, 5))
        self.drizzle_wht_label = ttk.Label(wht_frame, text="WHT Threshold %:")
        self.drizzle_wht_label.pack(side=tk.LEFT, padx=(0, 5))
        self.drizzle_wht_spinbox = ttk.Spinbox(
            wht_frame,
            from_=10.0,
            to=100.0,
            increment=5.0,
            textvariable=self.drizzle_wht_display_var,
            width=6,
            command=self._convert_spinbox_percent_to_float,
            format="%.0f",
        )
        self.drizzle_wht_spinbox.pack(side=tk.LEFT, padx=5)
        kernel_frame = ttk.Frame(self.drizzle_options_frame)
        kernel_frame.pack(fill=tk.X, padx=(20, 5), pady=(0, 5))
        self.drizzle_kernel_label = ttk.Label(kernel_frame, text="Kernel:")
        self.drizzle_kernel_label.pack(side=tk.LEFT, padx=(0, 5))
        valid_kernels = [
            "square",
            "gaussian",
            "point",
            "tophat",
            "turbo",
            "lanczos2",
            "lanczos3",
        ]
        self.drizzle_kernel_combo = ttk.Combobox(
            kernel_frame,
            textvariable=self.drizzle_kernel_var,
            values=valid_kernels,
            state="readonly",
            width=12,
        )
        self.drizzle_kernel_combo.pack(side=tk.LEFT, padx=5)
        pixfrac_frame = ttk.Frame(self.drizzle_options_frame)
        pixfrac_frame.pack(fill=tk.X, padx=(20, 5), pady=(0, 5))
        self.drizzle_pixfrac_label = ttk.Label(pixfrac_frame, text="Pixfrac:")
        self.drizzle_pixfrac_label.pack(side=tk.LEFT, padx=(0, 5))
        self.drizzle_pixfrac_spinbox = ttk.Spinbox(
            pixfrac_frame,
            from_=0.01,
            to=2.00,
            increment=0.05,
            textvariable=self.drizzle_pixfrac_var,
            width=6,
            format="%.2f",
        )
        self.drizzle_pixfrac_spinbox.pack(side=tk.LEFT, padx=5)
        self.use_gpu_check = ttk.Checkbutton(
            pixfrac_frame,
            text=self.tr("drizzle_use_gpu_label", default="Use GPU"),
            variable=self.use_gpu_var,
        )
        self.use_gpu_check.pack(side=tk.LEFT, padx=5)
        self.hp_frame = ttk.LabelFrame(tab_stacking, text="Hot Pixel Correction")
        self.hp_frame.pack(fill=tk.X, pady=5, padx=5)
        hp_check_frame = ttk.Frame(self.hp_frame)
        hp_check_frame.pack(fill=tk.X, padx=5, pady=2)
        self.hot_pixels_check = ttk.Checkbutton(
            hp_check_frame, text="Correct hot pixels", variable=self.correct_hot_pixels
        )
        self.hot_pixels_check.pack(side=tk.LEFT, padx=(0, 10))
        hp_params_frame = ttk.Frame(self.hp_frame)
        hp_params_frame.pack(fill=tk.X, padx=5, pady=(2, 5))
        self.hot_pixel_threshold_label = ttk.Label(hp_params_frame, text="Threshold:")
        self.hot_pixel_threshold_label.pack(side=tk.LEFT)
        self.hp_thresh_spinbox = ttk.Spinbox(
            hp_params_frame,
            from_=1.0,
            to=10.0,
            increment=0.1,
            textvariable=self.hot_pixel_threshold,
            width=5,
        )
        self.hp_thresh_spinbox.pack(side=tk.LEFT, padx=5)
        self.neighborhood_size_label = ttk.Label(hp_params_frame, text="Neighborhood:")
        self.neighborhood_size_label.pack(side=tk.LEFT)
        self.hp_neigh_spinbox = ttk.Spinbox(
            hp_params_frame,
            from_=3,
            to=15,
            increment=2,
            textvariable=self.neighborhood_size,
            width=4,
        )
        self.hp_neigh_spinbox.pack(side=tk.LEFT, padx=5)
        self.post_proc_opts_frame = ttk.LabelFrame(tab_stacking, text="Post-Processing Options")
        self.post_proc_opts_frame.pack(fill=tk.X, pady=5, padx=5)
        self.cleanup_temp_check = ttk.Checkbutton(
            self.post_proc_opts_frame,
            text="Cleanup temporary files",
            variable=self.cleanup_temp_var,
        )
        self.cleanup_temp_check.pack(side=tk.LEFT, padx=5, pady=5)
        self.chroma_correction_check = ttk.Checkbutton(
            self.post_proc_opts_frame,
            text="Edge Enhance",
            variable=self.apply_chroma_correction_var,
        )
        self.chroma_correction_check.pack(side=tk.LEFT, padx=5, pady=5)

        self.control_notebook.add(tab_stacking, text=f' {self.tr("tab_stacking", default=" Stacking ")} ')
        print("DEBUG (GUI create_layout): Onglet Empilement créé et ajouté.")

        # --- Onglet Expert (Index 1) ---
        tab_expert = ttk.Frame(self.control_notebook)
        print("DEBUG (GUI create_layout): Frame pour onglet Expert créé.")

        expert_scroll_canvas = tk.Canvas(tab_expert, highlightthickness=0, bg=canvas_background_color)
        expert_scrollbar = ttk.Scrollbar(tab_expert, orient="vertical", command=expert_scroll_canvas.yview)
        expert_scroll_canvas.configure(yscrollcommand=expert_scrollbar.set)
        expert_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        expert_scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        expert_content_frame = ttk.Frame(expert_scroll_canvas)
        expert_content_frame_id = expert_scroll_canvas.create_window(
            (0, 0),
            window=expert_content_frame,
            anchor="nw",
            tags="expert_content_frame_tag_v2",
        )

        def _on_expert_content_configure_local(event):
            if expert_scroll_canvas.winfo_exists() and expert_scroll_canvas.winfo_width() > 1:
                try:
                    expert_scroll_canvas.itemconfig(
                        expert_content_frame_id,
                        width=expert_scroll_canvas.winfo_width(),
                    )
                except tk.TclError:
                    pass
            if expert_scroll_canvas.winfo_exists():
                try:
                    expert_scroll_canvas.config(scrollregion=expert_scroll_canvas.bbox("all"))
                except tk.TclError:
                    pass

        expert_content_frame.bind("<Configure>", _on_expert_content_configure_local)
        expert_scroll_canvas.bind(
            "<Configure>",
            lambda e, c=expert_scroll_canvas, i=expert_content_frame_id: (
                (c.itemconfig(i, width=e.width) if c.winfo_exists() and e.width > 1 else None)
                if c.winfo_exists()
                else None
            ),
        )
        print("DEBUG (GUI create_layout): Canvas scrollable pour onglet Expert configuré.")

        self.warning_label = ttk.Label(
            expert_content_frame,
            text=self.tr("expert_warning_text", default="Expert Settings!"),
            foreground="red",
            font=("Arial", 10, "italic"),
        )
        self.warning_label.pack(pady=(5, 10), padx=5, fill=tk.X)

        self.apply_batch_feathering_check = ttk.Checkbutton(
            expert_content_frame,
            text=self.tr(
                "feather_inter_batch_label",
                default="Feather inter-batch (radial blend)",
            ),
            variable=self.apply_batch_feathering_var,
            command=self._on_apply_batch_feathering_changed,
        )
        self.apply_batch_feathering_check.pack(anchor=tk.W, padx=5, pady=(0, 5))

        self.feathering_frame = ttk.LabelFrame(
            expert_content_frame,
            text=self.tr("feathering_frame_title", default="Feathering / Low WHT"),
            padding=5,
        )
        self.feathering_frame.pack(fill=tk.X, padx=5, pady=5)
        print("DEBUG (GUI create_layout): Feathering Frame (conteneur pour Feathering et Low WHT) créé.")

        self.apply_feathering_check = ttk.Checkbutton(
            self.feathering_frame,
            text=self.tr("apply_feathering_label", default="Enable Feathering"),
            variable=self.apply_feathering_var,
            command=self._update_feathering_options_state,
        )
        self.apply_feathering_check.pack(anchor=tk.W, padx=5, pady=(5, 0))
        feather_params_frame = ttk.Frame(self.feathering_frame)
        feather_params_frame.pack(fill=tk.X, padx=(20, 0), pady=(0, 5))
        self.feather_blur_px_label = ttk.Label(
            feather_params_frame,
            text=self.tr("feather_blur_px_label", default="Blur (px):"),
        )
        self.feather_blur_px_label.pack(side=tk.LEFT, padx=(0, 5), pady=2)
        self.feather_blur_px_spinbox = ttk.Spinbox(
            feather_params_frame,
            from_=32,
            to=512,
            increment=16,
            width=6,
            textvariable=self.feather_blur_px_var,
        )
        self.feather_blur_px_spinbox.pack(side=tk.LEFT, padx=2, pady=2)

        self.low_wht_mask_check = ttk.Checkbutton(
            self.feathering_frame,
            text=self.tr("apply_low_wht_mask_label", default="Apply Low WHT Mask"),
            variable=self.apply_low_wht_mask_var,
            command=self._update_low_wht_mask_options_state,
        )
        self.low_wht_mask_check.pack(anchor=tk.W, padx=5, pady=(10, 0))
        low_wht_params_frame = ttk.Frame(self.feathering_frame)
        low_wht_params_frame.pack(fill=tk.X, padx=(20, 0), pady=(0, 5))
        self.low_wht_pct_label = ttk.Label(
            low_wht_params_frame,
            text=self.tr("low_wht_percentile_label", default="Percentile:"),
        )
        self.low_wht_pct_label.pack(side=tk.LEFT, padx=(0, 5), pady=2)
        self.low_wht_pct_spinbox = ttk.Spinbox(
            low_wht_params_frame,
            from_=1,
            to=100,
            increment=1,
            width=4,
            textvariable=self.low_wht_pct_var,
        )
        self.low_wht_pct_spinbox.pack(side=tk.LEFT, padx=2, pady=2)
        self.low_wht_soften_px_label = ttk.Label(
            low_wht_params_frame,
            text=self.tr("low_wht_soften_px_label", default="Soften (px):"),
        )
        self.low_wht_soften_px_label.pack(side=tk.LEFT, padx=(10, 5), pady=2)
        self.low_wht_soften_px_spinbox = ttk.Spinbox(
            low_wht_params_frame,
            from_=32,
            to=512,
            increment=16,
            width=6,
            textvariable=self.low_wht_soften_px_var,
        )
        self.low_wht_soften_px_spinbox.pack(side=tk.LEFT, padx=2, pady=2)
        print("DEBUG (GUI create_layout): Widgets Feathering et Low WHT Mask créés.")

        self.bn_frame = ttk.LabelFrame(
            expert_content_frame,
            text=self.tr("bn_frame_title", default="Auto Background Neutralization"),
            padding=5,
        )
        self.bn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.bn_frame.columnconfigure(1, weight=0)
        self.bn_frame.columnconfigure(3, weight=0)

        self.apply_bn_check = ttk.Checkbutton(
            self.bn_frame,
            text="Enable BN",
            variable=self.apply_bn_var,
            command=self._update_bn_options_state,
        )
        self.apply_bn_check.grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=5, pady=(0, 3))
        self.bn_grid_size_actual_label = ttk.Label(self.bn_frame, text="Grid Size:")
        self.bn_grid_size_actual_label.grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        self.bn_grid_size_combo = ttk.Combobox(
            self.bn_frame,
            textvariable=self.bn_grid_size_str_var,
            values=["8x8", "16x16", "24x24", "32x32", "64x64"],
            width=7,
            state="readonly",
        )
        self.bn_grid_size_combo.grid(row=1, column=1, sticky=tk.W, padx=2, pady=2)
        self.bn_perc_low_actual_label = ttk.Label(self.bn_frame, text="BG Perc. Low:")
        self.bn_perc_low_actual_label.grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
        self.bn_perc_low_spinbox = ttk.Spinbox(
            self.bn_frame,
            from_=0,
            to=40,
            increment=1,
            width=5,
            textvariable=self.bn_perc_low_var,
        )
        self.bn_perc_low_spinbox.grid(row=2, column=1, sticky=tk.W, padx=2, pady=2)
        self.bn_perc_high_actual_label = ttk.Label(self.bn_frame, text="BG Perc. High:")
        self.bn_perc_high_actual_label.grid(row=2, column=2, sticky=tk.W, padx=2, pady=2)
        self.bn_perc_high_spinbox = ttk.Spinbox(
            self.bn_frame,
            from_=10,
            to=95,
            increment=1,
            width=5,
            textvariable=self.bn_perc_high_var,
        )
        self.bn_perc_high_spinbox.grid(row=2, column=3, sticky=tk.W, padx=2, pady=2)
        self.bn_std_factor_actual_label = ttk.Label(self.bn_frame, text="BG Std Factor:")
        self.bn_std_factor_actual_label.grid(row=3, column=0, sticky=tk.W, padx=2, pady=2)
        self.bn_std_factor_spinbox = ttk.Spinbox(
            self.bn_frame,
            from_=0.5,
            to=5.0,
            increment=0.1,
            width=5,
            format="%.1f",
            textvariable=self.bn_std_factor_var,
        )
        self.bn_std_factor_spinbox.grid(row=3, column=1, sticky=tk.W, padx=2, pady=2)
        self.bn_min_gain_actual_label = ttk.Label(self.bn_frame, text="Min Gain:")
        self.bn_min_gain_actual_label.grid(row=4, column=0, sticky=tk.W, padx=2, pady=2)
        self.bn_min_gain_spinbox = ttk.Spinbox(
            self.bn_frame,
            from_=0.1,
            to=2.0,
            increment=0.1,
            width=5,
            format="%.1f",
            textvariable=self.bn_min_gain_var,
        )
        self.bn_min_gain_spinbox.grid(row=4, column=1, sticky=tk.W, padx=2, pady=2)
        self.bn_max_gain_actual_label = ttk.Label(self.bn_frame, text="Max Gain:")
        self.bn_max_gain_actual_label.grid(row=4, column=2, sticky=tk.W, padx=2, pady=2)
        self.bn_max_gain_spinbox = ttk.Spinbox(
            self.bn_frame,
            from_=1.0,
            to=10.0,
            increment=0.1,
            width=5,
            format="%.1f",
            textvariable=self.bn_max_gain_var,
        )
        self.bn_max_gain_spinbox.grid(row=4, column=3, sticky=tk.W, padx=2, pady=2)
        print("DEBUG (GUI create_layout): Cadre BN créé.")

        self.cb_frame = ttk.LabelFrame(
            expert_content_frame,
            text=self.tr("cb_frame_title", default="Edge/Chroma Correction"),
            padding=5,
        )
        self.cb_frame.pack(fill=tk.X, padx=5, pady=5)
        self.cb_frame.columnconfigure(1, weight=0)
        self.cb_frame.columnconfigure(3, weight=0)

        self.apply_cb_check = ttk.Checkbutton(
            self.cb_frame,
            text="Enable Edge/Chroma Correction",
            variable=self.apply_cb_var,
            command=self._update_cb_options_state,
        )
        self.apply_cb_check.grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=5, pady=(0, 3))
        self.cb_border_size_actual_label = ttk.Label(self.cb_frame, text="Border Size (px):")
        self.cb_border_size_actual_label.grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        self.cb_border_size_spinbox = ttk.Spinbox(
            self.cb_frame,
            from_=5,
            to=150,
            increment=5,
            width=5,
            textvariable=self.cb_border_size_var,
        )
        self.cb_border_size_spinbox.grid(row=1, column=1, sticky=tk.W, padx=2, pady=2)
        self.cb_blur_radius_actual_label = ttk.Label(self.cb_frame, text="Blur Radius (px):")
        self.cb_blur_radius_actual_label.grid(row=1, column=2, sticky=tk.W, padx=2, pady=2)
        self.cb_blur_radius_spinbox = ttk.Spinbox(
            self.cb_frame,
            from_=0,
            to=50,
            increment=1,
            width=5,
            textvariable=self.cb_blur_radius_var,
        )
        self.cb_blur_radius_spinbox.grid(row=1, column=3, sticky=tk.W, padx=2, pady=2)
        self.cb_min_b_factor_actual_label = ttk.Label(self.cb_frame, text="Min Blue Factor:")
        self.cb_min_b_factor_actual_label.grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
        self.cb_min_b_factor_spinbox = ttk.Spinbox(
            self.cb_frame,
            from_=0.1,
            to=1.0,
            increment=0.05,
            width=5,
            format="%.2f",
            textvariable=self.cb_min_b_factor_var,
        )
        self.cb_min_b_factor_spinbox.grid(row=2, column=1, sticky=tk.W, padx=2, pady=2)
        self.cb_max_b_factor_actual_label = ttk.Label(self.cb_frame, text="Max Blue Factor:")
        self.cb_max_b_factor_actual_label.grid(row=2, column=2, sticky=tk.W, padx=2, pady=2)
        self.cb_max_b_factor_spinbox = ttk.Spinbox(
            self.cb_frame,
            from_=1.0,
            to=3.0,
            increment=0.05,
            width=5,
            format="%.2f",
            textvariable=self.cb_max_b_factor_var,
        )
        self.cb_max_b_factor_spinbox.grid(row=2, column=3, sticky=tk.W, padx=2, pady=2)
        print("DEBUG (GUI create_layout): Cadre CB créé.")

        self.crop_frame = ttk.LabelFrame(
            expert_content_frame,
            text=self.tr("crop_frame_title", default="Final Cropping"),
            padding=5,
        )
        self.crop_frame.pack(fill=tk.X, padx=5, pady=5)
        self.apply_crop_check = ttk.Checkbutton(
            self.crop_frame,
            text="Enable Final Cropping",
            variable=self.apply_final_crop_var,
            command=self._update_crop_options_state,
        )
        self.apply_crop_check.pack(anchor=tk.W, padx=5, pady=(0, 3))
        self.final_edge_crop_actual_label = ttk.Label(self.crop_frame, text="Edge Crop (%):")
        self.final_edge_crop_actual_label.pack(side=tk.LEFT, padx=(2, 5), pady=2)
        self.final_edge_crop_spinbox = ttk.Spinbox(
            self.crop_frame,
            from_=0.0,
            to=25.0,
            increment=0.5,
            width=6,
            format="%.1f",
            textvariable=self.final_edge_crop_percent_var,
        )
        self.final_edge_crop_spinbox.pack(side=tk.LEFT, padx=2, pady=2)
        print("DEBUG (GUI create_layout): Cadre Crop créé.")

        self.photutils_bn_frame = ttk.LabelFrame(
            expert_content_frame,
            text=self.tr(
                "photutils_bn_frame_title",
                default="2D Background Subtraction (Photutils)",
            ),
            padding=5,
        )
        self.photutils_bn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.apply_photutils_bn_check = ttk.Checkbutton(
            self.photutils_bn_frame,
            text="Enable Photutils 2D Bkg Subtraction",
            variable=self.apply_photutils_bn_var,
            command=self._update_photutils_bn_options_state,
        )
        self.apply_photutils_bn_check.pack(anchor=tk.W, padx=5, pady=(5, 2))
        self.photutils_params_frame = ttk.Frame(self.photutils_bn_frame)
        self.photutils_params_frame.pack(fill=tk.X, padx=(20, 0), pady=2)
        self.photutils_params_frame.columnconfigure(1, weight=0)
        self.photutils_params_frame.columnconfigure(3, weight=0)
        self.photutils_bn_box_size_label = ttk.Label(self.photutils_params_frame, text="Box Size (px):")
        self.photutils_bn_box_size_label.grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        self.pb_box_spinbox = ttk.Spinbox(
            self.photutils_params_frame,
            from_=16,
            to=1024,
            increment=16,
            width=6,
            textvariable=self.photutils_bn_box_size_var,
        )
        self.pb_box_spinbox.grid(row=0, column=1, sticky=tk.W, padx=2, pady=2)
        self.photutils_bn_filter_size_label = ttk.Label(self.photutils_params_frame, text="Filter Size (px, odd):")
        self.photutils_bn_filter_size_label.grid(row=0, column=2, sticky=tk.W, padx=(10, 2), pady=2)
        self.pb_filt_spinbox = ttk.Spinbox(
            self.photutils_params_frame,
            from_=1,
            to=15,
            increment=2,
            width=5,
            textvariable=self.photutils_bn_filter_size_var,
        )
        self.pb_filt_spinbox.grid(row=0, column=3, sticky=tk.W, padx=2, pady=2)
        self.photutils_bn_sigma_clip_label = ttk.Label(self.photutils_params_frame, text="Sigma Clip Value:")
        self.photutils_bn_sigma_clip_label.grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        self.pb_sig_spinbox = ttk.Spinbox(
            self.photutils_params_frame,
            from_=1.0,
            to=5.0,
            increment=0.1,
            width=5,
            format="%.1f",
            textvariable=self.photutils_bn_sigma_clip_var,
        )
        self.pb_sig_spinbox.grid(row=1, column=1, sticky=tk.W, padx=2, pady=2)
        self.photutils_bn_exclude_percentile_label = ttk.Label(
            self.photutils_params_frame, text="Exclude Brightest (%):"
        )
        self.photutils_bn_exclude_percentile_label.grid(row=1, column=2, sticky=tk.W, padx=(10, 2), pady=2)
        self.pb_excl_spinbox = ttk.Spinbox(
            self.photutils_params_frame,
            from_=0.0,
            to=100.0,
            increment=1.0,
            width=6,
            format="%.1f",
            textvariable=self.photutils_bn_exclude_percentile_var,
        )
        self.pb_excl_spinbox.grid(row=1, column=3, sticky=tk.W, padx=2, pady=2)
        print("DEBUG (GUI create_layout): Cadre Photutils BN créé.")

        # --- NOUVEAU : Cadre pour les options de format de sortie FITS ---
        self.output_format_frame = ttk.LabelFrame(
            expert_content_frame,
            text=self.tr("output_format_frame_title", default="Output FITS Format"),
            padding=5,
        )
        self.output_format_frame.pack(fill=tk.X, padx=5, pady=5)
        print("DEBUG (GUI create_layout): Output Format Frame créé.")

        self.save_as_float32_check = ttk.Checkbutton(
            self.output_format_frame,
            text=self.tr(
                "save_as_float32_label",
                default="Save final FITS as float32 (larger files, max precision)",
            ),
            variable=self.save_as_float32_var,  # Variable Tkinter créée dans init_variables
        )
        self.save_as_float32_check.pack(anchor=tk.W, padx=5, pady=5)
        # Pas besoin de command ici, la valeur sera lue par SettingsManager.update_from_ui()
        print("DEBUG (GUI create_layout): Checkbutton save_as_float32 créé.")

        self.preserve_linear_output_check = ttk.Checkbutton(
            self.output_format_frame,
            text=self.tr(
                "preserve_linear_output_label",
                default="Preserve linear output (skip percentile scaling)",
            ),
            variable=self.preserve_linear_output_var,
        )
        self.preserve_linear_output_check.pack(anchor=tk.W, padx=5, pady=2)
        print("DEBUG (GUI create_layout): Checkbutton preserve_linear_output créé.")
        # --- FIN NOUVEAU ---

        self.reset_expert_button = ttk.Button(
            expert_content_frame,
            text=self.tr("reset_expert_button", default="Reset Expert Settings"),
            command=self._reset_expert_settings,
        )
        self.reset_expert_button.pack(pady=(10, 5))  # Packé APRÈS le nouveau cadre
        print("DEBUG (GUI create_layout): Bouton Reset Expert créé.")

        expert_tab_title_text = self.tr("tab_expert_title", default=" Expert ")
        self.control_notebook.add(tab_expert, text=f" {expert_tab_title_text} ")
        print("DEBUG (GUI create_layout): Onglet Expert ajouté au Notebook.")

        # --- Onglet Aperçu (Index 2) ---
        # ... (inchangé) ...
        tab_preview = ttk.Frame(self.control_notebook)
        self.wb_frame = ttk.LabelFrame(tab_preview, text="White Balance (Preview)")
        self.wb_frame.pack(fill=tk.X, pady=5, padx=5)
        self.wb_r_ctrls = self._create_slider_spinbox_group(self.wb_frame, "wb_r", 0.1, 5.0, 0.01, self.preview_r_gain)
        self.wb_g_ctrls = self._create_slider_spinbox_group(self.wb_frame, "wb_g", 0.1, 5.0, 0.01, self.preview_g_gain)
        self.wb_b_ctrls = self._create_slider_spinbox_group(self.wb_frame, "wb_b", 0.1, 5.0, 0.01, self.preview_b_gain)
        wb_btn_frame = ttk.Frame(self.wb_frame)
        wb_btn_frame.pack(fill=tk.X, pady=5)
        self.auto_wb_button = ttk.Button(
            wb_btn_frame,
            text="Auto WB",
            command=self.apply_auto_white_balance,
            state=tk.NORMAL if _tools_available else tk.DISABLED,
        )
        self.auto_wb_button.pack(side=tk.LEFT, padx=5)
        self.reset_wb_button = ttk.Button(wb_btn_frame, text="Reset WB", command=self.reset_white_balance)
        self.reset_wb_button.pack(side=tk.LEFT, padx=5)
        self.stretch_frame_controls = ttk.LabelFrame(tab_preview, text="Stretch (Preview)")
        self.stretch_frame_controls.pack(fill=tk.X, pady=5, padx=5)
        stretch_method_frame = ttk.Frame(self.stretch_frame_controls)
        stretch_method_frame.pack(fill=tk.X, pady=2)
        self.stretch_method_label = ttk.Label(stretch_method_frame, text="Method:")
        self.stretch_method_label.pack(side=tk.LEFT, padx=(5, 5))
        self.stretch_combo = ttk.Combobox(
            stretch_method_frame,
            textvariable=self.preview_stretch_method,
            values=("Linear", "Asinh", "Log"),
            width=15,
            state="readonly",
        )
        self.stretch_combo.pack(side=tk.LEFT)
        self.stretch_combo.bind("<<ComboboxSelected>>", self._debounce_refresh_preview)
        self.stretch_bp_ctrls = self._create_slider_spinbox_group(
            self.stretch_frame_controls,
            "stretch_bp",
            0.0,
            1.0,
            0.001,
            self.preview_black_point,
            callback=self.update_histogram_lines_from_sliders,
        )
        self.stretch_wp_ctrls = self._create_slider_spinbox_group(
            self.stretch_frame_controls,
            "stretch_wp",
            0.0,
            1.0,
            0.001,
            self.preview_white_point,
            callback=self.update_histogram_lines_from_sliders,
        )
        self.stretch_gamma_ctrls = self._create_slider_spinbox_group(
            self.stretch_frame_controls,
            "stretch_gamma",
            0.1,
            5.0,
            0.01,
            self.preview_gamma,
        )
        stretch_btn_frame = ttk.Frame(self.stretch_frame_controls)
        stretch_btn_frame.pack(fill=tk.X, pady=5)
        self.auto_stretch_button = ttk.Button(
            stretch_btn_frame,
            text="Auto Stretch",
            command=self.apply_auto_stretch,
            state=tk.NORMAL if _tools_available else tk.DISABLED,
        )
        self.auto_stretch_button.pack(side=tk.LEFT, padx=5)
        self.reset_stretch_button = ttk.Button(stretch_btn_frame, text="Reset Stretch", command=self.reset_stretch)
        self.reset_stretch_button.pack(side=tk.LEFT, padx=5)
        self.bcs_frame = ttk.LabelFrame(tab_preview, text="Image Adjustments")
        self.bcs_frame.pack(fill=tk.X, pady=5, padx=5)
        self.brightness_ctrls = self._create_slider_spinbox_group(
            self.bcs_frame, "brightness", 0.1, 3.0, 0.01, self.preview_brightness
        )
        self.contrast_ctrls = self._create_slider_spinbox_group(
            self.bcs_frame, "contrast", 0.1, 3.0, 0.01, self.preview_contrast
        )
        self.saturation_ctrls = self._create_slider_spinbox_group(
            self.bcs_frame, "saturation", 0.0, 3.0, 0.01, self.preview_saturation
        )
        bcs_btn_frame = ttk.Frame(self.bcs_frame)
        bcs_btn_frame.pack(fill=tk.X, pady=5)
        self.reset_bcs_button = ttk.Button(
            bcs_btn_frame,
            text="Reset Adjust.",
            command=self.reset_brightness_contrast_saturation,
        )
        self.reset_bcs_button.pack(side=tk.LEFT, padx=5)
        preview_tab_title_text = self.tr("tab_preview", default=" Preview ")
        self.control_notebook.add(tab_preview, text=f" {preview_tab_title_text} ")
        print("DEBUG (GUI create_layout): Onglet Aperçu créé et ajouté.")

        # --- Zone Progression (ENFANT DE self.left_content_frame, packé EN BAS) ---
        # ... (inchangé) ...
        self.progress_frame = ttk.LabelFrame(self.left_content_frame, text=self.tr("progress", default="Progress"))
        self.progress_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, padx=5, pady=(10, 5))
        self.progress_bar = ttk.Progressbar(self.progress_frame, maximum=100, mode="determinate")
        self.progress_bar.pack(fill=tk.X, padx=5, pady=(5, 2))
        time_frame = ttk.Frame(self.progress_frame)
        time_frame.pack(fill=tk.X, padx=5, pady=2)
        time_frame.columnconfigure(0, weight=0)
        time_frame.columnconfigure(1, weight=1)
        time_frame.columnconfigure(2, weight=0)
        time_frame.columnconfigure(3, weight=0)
        self.remaining_time_label = ttk.Label(time_frame, text="ETA:")
        self.remaining_time_label.grid(row=0, column=0, sticky="w")
        self.remaining_time_value = ttk.Label(
            time_frame,
            textvariable=self.remaining_time_var,
            font=tkFont.Font(weight="bold"),
            anchor="w",
        )
        self.remaining_time_value.grid(row=0, column=1, sticky="w", padx=(2, 10))
        self.elapsed_time_label = ttk.Label(time_frame, text="Elapsed:")
        self.elapsed_time_label.grid(row=0, column=2, sticky="e", padx=(5, 0))
        self.elapsed_time_value = ttk.Label(
            time_frame,
            textvariable=self.elapsed_time_var,
            font=tkFont.Font(weight="bold"),
            width=9,
            anchor="e",
        )
        self.elapsed_time_value.grid(row=0, column=3, sticky="e", padx=(2, 0))
        files_info_frame = ttk.Frame(self.progress_frame)
        files_info_frame.pack(fill=tk.X, padx=5, pady=2)
        self.remaining_static_label = ttk.Label(files_info_frame, text="Remaining:")
        self.remaining_static_label.pack(side=tk.LEFT)
        self.remaining_value_label = ttk.Label(
            files_info_frame,
            textvariable=self.remaining_files_var,
            width=12,
            anchor="w",
        )
        self.remaining_value_label.pack(side=tk.LEFT, padx=(2, 10))
        self.aligned_files_label = ttk.Label(
            files_info_frame, textvariable=self.aligned_files_var, width=12, anchor="w"
        )
        self.aligned_files_label.pack(side=tk.LEFT, padx=(10, 0))
        self.additional_value_label = ttk.Label(files_info_frame, textvariable=self.additional_folders_var, anchor="e")
        self.additional_value_label.pack(side=tk.RIGHT)
        self.additional_static_label = ttk.Label(files_info_frame, text="Additional:")
        self.additional_static_label.pack(side=tk.RIGHT, padx=(0, 2))
        status_text_frame = ttk.Frame(self.progress_frame)
        status_text_font = tkFont.Font(family="Arial", size=8)
        status_text_frame.pack(fill=tk.X, expand=False, padx=5, pady=(2, 5))
        self.copy_log_button = ttk.Button(status_text_frame, text="Copy", command=self._copy_log_to_clipboard, width=5)
        self.copy_log_button.pack(side=tk.RIGHT, padx=(2, 0), pady=0, anchor="ne")
        self.status_scrollbar = ttk.Scrollbar(status_text_frame, orient="vertical")
        self.status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=0)
        self.status_text = tk.Text(
            status_text_frame,
            height=6,
            wrap=tk.WORD,
            bd=0,
            font=status_text_font,
            relief=tk.FLAT,
            state=tk.DISABLED,
            yscrollcommand=self.status_scrollbar.set,
        )
        self.status_text.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=0)
        self.status_scrollbar.config(command=self.status_text.yview)
        print("DEBUG (GUI create_layout): Zone de progression créée.")

        print("DEBUG (GUI create_layout): Fin remplissage panneau gauche.")

        # =====================================================================
        # --- Panneau Droit (Aperçu, Boutons de Contrôle, Histogramme) ---
        # =====================================================================
        # ... (inchangé) ...
        control_frame = ttk.Frame(right_frame)
        try:
            style = ttk.Style()
            accent_style = "Accent.TButton" if "Accent.TButton" in style.element_names() else "TButton"
        except tk.TclError:
            accent_style = "TButton"
        self.start_button = ttk.Button(
            control_frame,
            text="Start",
            command=self.start_processing,
            style=accent_style,
        )
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5, ipady=2)
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5, ipady=2)
        self.analyze_folder_button = ttk.Button(
            control_frame,
            text="Analyze Input Folder",
            command=self._launch_folder_analyzer,
            state=tk.DISABLED,
        )
        self.analyze_folder_button.pack(side=tk.LEFT, padx=5, pady=5, ipady=2)
        # Launch the external ZeMosaic application instead of the built-in
        # mosaic settings window when the "Mosaic..." button is clicked.
        self.mosaic_options_button = ttk.Button(
            control_frame,
            text="Mosaic...",
            command=self.run_zemosaic,
        )
        self.mosaic_options_button.pack(side=tk.LEFT, padx=5, pady=5, ipady=2)
        self.local_solver_button = ttk.Button(
            control_frame,
            text=self.tr("local_solver_button_text", default="Local Solvers..."),
            command=self._open_local_solver_settings_window,
        )
        self.local_solver_button.pack(side=tk.LEFT, padx=5, pady=5, ipady=2)
        self.open_output_button = ttk.Button(
            control_frame,
            text="Open Output",
            command=self._open_output_folder,
            state=tk.DISABLED,
        )
        self.open_output_button.pack(side=tk.RIGHT, padx=5, pady=5, ipady=2)
        self.add_files_button = ttk.Button(
            control_frame,
            text="Add Folder",
            command=self.file_handler.add_folder,
            state=tk.NORMAL,
        )
        self.add_files_button.pack(side=tk.RIGHT, padx=5, pady=5, ipady=2)
        self.show_folders_button = ttk.Button(
            control_frame,
            text="View Inputs",
            command=self._show_input_folder_list,
            state=tk.DISABLED,
        )
        self.show_folders_button.pack(side=tk.RIGHT, padx=5, pady=5, ipady=2)

        self.histo_toolbar = ttk.Frame(control_frame)
        self.histo_toolbar.pack(side=tk.RIGHT, padx=6)
        self.histogram_frame = ttk.LabelFrame(right_frame, text="Histogram")
        hist_fig_height_inches = 2.2
        hist_fig_dpi = 80
        hist_height_pixels = int(hist_fig_height_inches * hist_fig_dpi * 1.1)
        self.histogram_frame.config(height=hist_height_pixels)
        self.histogram_frame.pack_propagate(False)
        self.histogram_widget = HistogramWidget(
            self.histogram_frame,
            range_change_callback=self.update_stretch_from_histogram,
        )
        self.histogram_widget.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=(0, 2), pady=(0, 2))
        self.histogram_widget.auto_zoom_enabled = self.auto_zoom_histogram_var.get()

        # Appliquer l'état de verrouillage de l'échelle X selon l'option
        self._update_histogram_autozoom_state()

        self.auto_zoom_histo_check = ttk.Checkbutton(
            self.histo_toolbar,
            text=self.tr("auto_zoom_histo_check", default="Auto zoom histogram"),
            variable=self.auto_zoom_histogram_var,
            command=lambda: setattr(
                self.histogram_widget,
                "auto_zoom_enabled",
                self.auto_zoom_histogram_var.get(),
            ),
        )
        self.auto_zoom_histo_check.pack(side=tk.LEFT, padx=2)
        self.hist_reset_view_btn = ttk.Button(
            self.histo_toolbar,
            text=self.tr("reset_histo_button", default="Reset Histogram"),
            command=self.histogram_widget.reset_histogram_view,
        )
        self.hist_reset_view_btn.pack(side=tk.LEFT, padx=2)
        self.hist_zoom_btn = ttk.Button(
            self.histo_toolbar,
            text=self.tr("zoom_histo_button", default="Zoom Histogram"),
            command=self.histogram_widget.zoom_histogram,
        )
        self.hist_zoom_btn.pack(side=tk.LEFT, padx=2)

        self.hist_reset_btn = ttk.Button(
            self.histogram_frame,
            text="R",
            command=self.histogram_widget.reset_zoom,
            width=2,
        )
        self.hist_reset_btn.pack(side=tk.RIGHT, anchor=tk.NE, padx=(0, 2), pady=2)
        self.preview_frame = ttk.LabelFrame(right_frame, text="Preview")
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="#1E1E1E", highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        zoom_btn_frame = ttk.Frame(self.preview_frame)
        zoom_btn_frame.pack(fill=tk.X, pady=(2, 2))
        self.zoom_100_button = ttk.Button(
            zoom_btn_frame,
            text="Zoom 100%",
            command=lambda: self.preview_manager.zoom_full_size(),
        )
        self.zoom_100_button.pack(side=tk.LEFT, padx=5)
        self.zoom_fit_button = ttk.Button(
            zoom_btn_frame,
            text="Zoom Fit",
            command=lambda: self.preview_manager.zoom_fit(),
        )
        self.zoom_fit_button.pack(side=tk.LEFT, padx=5)
        self.histogram_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, padx=5, pady=(5, 5))
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, padx=5, pady=(5, 0))
        self.preview_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=(5, 5))
        print("DEBUG (GUI create_layout): Panneau droit rempli.")

        self._store_widget_references()
        self._on_method_combo_change()
        self._update_final_scnr_options_state()
        self._update_photutils_bn_options_state()
        self._update_feathering_options_state()
        self._on_apply_batch_feathering_changed()
        self._update_low_wht_mask_options_state()
        self._update_bn_options_state()
        self._update_cb_options_state()
        self._update_crop_options_state()
        self._update_master_tile_crop_state()
        print(
            "DEBUG (GUI create_layout V_SaveAsFloat32_1): Fin création layout et appels _update_..._state."
        )  # Version Log

    #################################################################################################################################

    def _update_feathering_options_state(self, *args):
        """Active ou désactive le Spinbox de flou du feathering."""
        try:
            feathering_active = self.apply_feathering_var.get()
            new_state = tk.NORMAL if feathering_active else tk.DISABLED

            if hasattr(self, "feather_blur_px_spinbox") and self.feather_blur_px_spinbox.winfo_exists():
                self.feather_blur_px_spinbox.config(state=new_state)
            if hasattr(self, "feather_blur_px_label") and self.feather_blur_px_label.winfo_exists():
                self.feather_blur_px_label.config(state=new_state)  # Griser le label aussi

            print(f"DEBUG (GUI): État options Feathering (Blur Px Spinbox) mis à jour: {new_state}")
        except tk.TclError:
            pass
        except AttributeError:
            pass
        except Exception as e:
            print(f"ERREUR inattendue dans _update_feathering_options_state: {e}")
            traceback.print_exc(limit=1)

    def _on_apply_batch_feathering_changed(self, *args):
        """Sync apply_batch_feathering flag with the checkbox."""
        try:
            self.apply_batch_feathering = bool(self.apply_batch_feathering_var.get())
        except tk.TclError:
            pass

    ##############################################################################################################################

    def _update_low_wht_mask_options_state(self, *args):
        """
        Active ou désactive les options de percentile et soften pour Low WHT Mask
        en fonction de l'état de la checkbox principale.
        """
        print("DEBUG (GUI _update_low_wht_mask_options_state): Exécution...")  # Debug
        try:
            # Lire l'état de la checkbox principale pour "Low WHT Mask"
            mask_active = self.apply_low_wht_mask_var.get()
            new_state = tk.NORMAL if mask_active else tk.DISABLED
            print(f"  -> Mask active: {mask_active}, New state for options: {new_state}")  # Debug

            # Mettre à jour l'état du label et du spinbox pour le percentile
            if hasattr(self, "low_wht_pct_label") and self.low_wht_pct_label.winfo_exists():
                self.low_wht_pct_label.config(state=new_state)
            if hasattr(self, "low_wht_pct_spinbox") and self.low_wht_pct_spinbox.winfo_exists():
                self.low_wht_pct_spinbox.config(state=new_state)

            # Mettre à jour l'état du label et du spinbox pour soften_px
            if hasattr(self, "low_wht_soften_px_label") and self.low_wht_soften_px_label.winfo_exists():
                self.low_wht_soften_px_label.config(state=new_state)
            if hasattr(self, "low_wht_soften_px_spinbox") and self.low_wht_soften_px_spinbox.winfo_exists():
                self.low_wht_soften_px_spinbox.config(state=new_state)

            print(f"DEBUG (GUI _update_low_wht_mask_options_state): État des options Low WHT Mask mis à jour.")  # Debug

        except tk.TclError as e:
            # Peut arriver si les widgets sont en cours de destruction/création
            print(f"DEBUG (GUI _update_low_wht_mask_options_state): Erreur TclError -> {e}")  # Debug
            pass
        except AttributeError as e:
            # Peut arriver si un attribut (widget) n'existe pas encore (ex: pendant l'init)
            print(f"DEBUG (GUI _update_low_wht_mask_options_state): Erreur AttributeError -> {e}")  # Debug
            pass
        except Exception as e:
            print(f"ERREUR (GUI _update_low_wht_mask_options_state): Erreur inattendue -> {e}")
            traceback.print_exc(limit=1)

    ##############################################################################################################################

    def _update_photutils_bn_options_state(self, *args):
        """
        Enable/disable every direct child of self.photutils_params_frame
        instead of guessing fixed indices.
        """
        new_state = tk.NORMAL if self.apply_photutils_bn_var.get() else tk.DISABLED

        # Toggle *all* direct children (labels, spin‑boxes, …)
        for widget in self.photutils_params_frame.winfo_children():
            if widget.winfo_exists():
                try:
                    widget.config(state=new_state)
                except tk.TclError:
                    pass  # e.g. a Label – no 'state' option

    #        try:
    #           photutils_bn_active = self.apply_photutils_bn_var.get()
    #           new_state = tk.NORMAL if photutils_bn_active else tk.DISABLED
    #
    #            # Widgets à contrôler (les spinboxes et leurs labels)
    #            widgets_to_toggle = [
    #                getattr(self, 'photutils_bn_box_size_label', None),
    #                # Le spinbox est un enfant du frame, il faut le retrouver ou le stocker
    #                self.photutils_params_frame.winfo_children()[1] if hasattr(self, 'photutils_params_frame') and len(self.photutils_params_frame.winfo_children()) > 1 else None, # spin_pb_box
    #                getattr(self, 'photutils_bn_filter_size_label', None),
    #                self.photutils_params_frame.winfo_children()[3] if hasattr(self, 'photutils_params_frame') and len(self.photutils_params_frame.winfo_children()) > 3 else None, # spin_pb_filt
    #                getattr(self, 'photutils_bn_sigma_clip_label', None),
    #                self.photutils_params_frame.winfo_children()[5] if hasattr(self, 'photutils_params_frame') and len(self.photutils_params_frame.winfo_children()) > 5 else None, # spin_pb_sig
    #                getattr(self, 'photutils_bn_exclude_percentile_label', None),
    #                self.photutils_params_frame.winfo_children()[7] if hasattr(self, 'photutils_params_frame') and len(self.photutils_params_frame.winfo_children()) > 7 else None, # spin_pb_excl
    #            ]
    #
    #            for widget in widgets_to_toggle:
    #                if widget and hasattr(widget, 'winfo_exists') and widget.winfo_exists():
    #                    widget.config(state=new_state)
    #            # print(f"DEBUG: État options Photutils BN mis à jour vers: {new_state}")
    #        except tk.TclError: pass
    #        except AttributeError: pass # Si photutils_params_frame n'est pas encore créé
    #        except Exception as e: print(f"ERREUR inattendue dans _update_photutils_bn_options_state: {e}")

    ###############################################################################

    def _update_bn_options_state(self, *args):
        new_state = tk.NORMAL if self.apply_bn_var.get() else tk.DISABLED
        for w in self.bn_frame.winfo_children():
            if w is self.apply_bn_check:
                continue
            try:
                w.config(state=new_state)
            except tk.TclError:
                pass

    def _update_cb_options_state(self, *args):
        new_state = tk.NORMAL if self.apply_cb_var.get() else tk.DISABLED
        for w in self.cb_frame.winfo_children():
            if w is self.apply_cb_check:
                continue
            try:
                w.config(state=new_state)
            except tk.TclError:
                pass

    def _update_crop_options_state(self, *args):
        new_state = tk.NORMAL if self.apply_final_crop_var.get() else tk.DISABLED
        for w in self.crop_frame.winfo_children():
            if w is self.apply_crop_check:
                continue
            try:
                w.config(state=new_state)
            except tk.TclError:
                pass

    def _update_master_tile_crop_state(self, *args):
        new_state = tk.NORMAL if self.apply_master_tile_crop_var.get() else tk.DISABLED
        try:
            self.master_tile_crop_spinbox.config(state=new_state)
        except tk.TclError:
            pass

    ###############################################################################

    def _update_histogram_autozoom_state(self, *args):
        """Verrouille ou libère l'échelle X de l'histogramme selon l'option."""
        freeze = not self.auto_zoom_histogram_var.get()
        if hasattr(self, "histogram_widget"):
            self.histogram_widget.freeze_x_range = freeze

    ##############################################################################################################################
    def _toggle_kappa_visibility(self, event=None):
        """Affiche ou cache les widgets Kappa en fonction de la méthode de stacking, en utilisant grid."""
        method = None
        final_method = None
        if hasattr(self, "stack_method_var"):
            try:
                method = self.stack_method_var.get()
            except tk.TclError:
                method = None
        if hasattr(self, "stack_final_combine_var"):
            try:
                final_method = self.stack_final_combine_var.get()
            except tk.TclError:
                final_method = None

        show_kappa = False
        show_winsor = False
        if method == "kappa_sigma" or final_method == "winsorized_sigma_clip":
            show_kappa = True
        if method == "winsorized_sigma_clip" or final_method == "winsorized_sigma_clip":
            show_kappa = True
            show_winsor = True

        # Kappa parameter widgets using pack
        if hasattr(self, "kappa_low_spinbox") and hasattr(self, "kappa_high_spinbox"):
            if show_kappa:
                if not self.kappa_low_spinbox.winfo_ismapped():
                    self.kappa_low_label.pack(side=tk.LEFT, padx=(0, 2))
                    self.kappa_low_spinbox.pack(side=tk.LEFT, padx=(0, 10))
                    self.kappa_high_label.pack(side=tk.LEFT, padx=(0, 2))
                    self.kappa_high_spinbox.pack(side=tk.LEFT, padx=(0, 5))
            else:
                self.kappa_low_label.pack_forget()
                self.kappa_low_spinbox.pack_forget()
                self.kappa_high_label.pack_forget()
                self.kappa_high_spinbox.pack_forget()

        if hasattr(self, "winsor_limits_entry") and hasattr(self, "winsor_limits_label"):
            if show_winsor:
                if not self.winsor_limits_entry.winfo_ismapped():
                    self.winsor_limits_label.pack(side=tk.LEFT, padx=(0, 2))
                    self.winsor_limits_entry.pack(side=tk.LEFT, padx=(0, 5))
                    if hasattr(self, "winsor_note_label"):
                        self.winsor_note_label.pack(side=tk.LEFT, padx=(5, 0))
            else:
                self.winsor_limits_label.pack_forget()
                self.winsor_limits_entry.pack_forget()
                if hasattr(self, "winsor_note_label"):
                    self.winsor_note_label.pack_forget()

    def _on_method_combo_change(self, event=None):
        """Update internal vars when method selection changes."""
        # Convert displayed label back to internal key
        display_value = self.stack_method_display_var.get()
        method = self.method_label_to_key.get(display_value, display_value)
        self.stack_method_var.set(method)
        if hasattr(self, "stacking_mode"):
            try:
                self.stacking_mode.set(method.replace("_", "-"))
            except tk.TclError:
                pass
        if method == "mean":
            self.stack_final_combine_var.set("mean")
            self.stack_reject_algo_var.set("none")
        elif method == "median":
            self.stack_final_combine_var.set("median")
            self.stack_reject_algo_var.set("none")
        elif method == "kappa_sigma":
            self.stack_final_combine_var.set("mean")
            self.stack_reject_algo_var.set("kappa_sigma")
        elif method == "winsorized_sigma_clip":
            self.stack_final_combine_var.set("mean")
            self.stack_reject_algo_var.set("winsorized_sigma_clip")
        elif method == "linear_fit_clip":
            self.stack_final_combine_var.set("mean")
            self.stack_reject_algo_var.set("linear_fit_clip")
        if hasattr(self, "final_key_to_label"):
            if self.reproject_between_batches_var.get():
                self.stack_final_combine_var.set("reproject")
                self.stack_final_display_var.set(self.final_key_to_label.get("reproject", "reproject"))
            elif getattr(self, "reproject_coadd_var", tk.BooleanVar()).get():
                self.stack_final_combine_var.set("reproject_coadd")
                self.stack_final_display_var.set(self.final_key_to_label.get("reproject_coadd", "reproject_coadd"))
            else:
                current_key = self.stack_final_combine_var.get()
                self.stack_final_display_var.set(self.final_key_to_label.get(current_key, current_key))
        self._toggle_kappa_visibility()

    def _on_norm_combo_change(self, event=None):
        """Update internal var when normalization selection changes."""
        display_value = self.stack_norm_display_var.get()
        key = self.norm_label_to_key.get(display_value, display_value)
        self.stack_norm_method_var.set(key)

    def _on_weight_combo_change(self, event=None):
        """Update internal var when weighting selection changes."""
        display_value = self.stack_weight_display_var.get()
        key = self.weight_label_to_key.get(display_value, display_value)
        self.stack_weight_method_var.set(key)

    def _on_final_combo_change(self, event=None):
        """Update internal var when final combine selection changes."""
        display_value = self.stack_final_display_var.get()
        key = self.final_label_to_key.get(display_value, display_value)
        if key == "reproject":
            self.reproject_between_batches_var.set(True)
            self.reproject_coadd_var.set(False)
        elif key == "reproject_coadd":
            self.reproject_between_batches_var.set(False)
            self.reproject_coadd_var.set(True)
        else:
            self.reproject_between_batches_var.set(False)
            self.reproject_coadd_var.set(False)
        self.stack_final_combine_var.set(key)
        self._toggle_kappa_visibility()

    #################################################################################################################################

    def _launch_folder_analyzer(self):
        """
        Détermine le chemin du fichier de commande, lance le script analyse_gui.py
        en lui passant ce chemin, et démarre la surveillance du fichier.
        """
        print("DEBUG (GUI): Entrée dans _launch_folder_analyzer.")  # <-- AJOUTÉ DEBUG
        input_folder = self.input_path.get()

        # 1. Validation du dossier d'entrée (inchangé)
        if not input_folder:
            messagebox.showerror(self.tr("error"), self.tr("select_folders"))
            return
        if not os.path.isdir(input_folder):
            messagebox.showerror(self.tr("error"), f"{self.tr('input_folder_invalid')}:\n{input_folder}")
            return

        # --- MODIFIÉ : Détermination du chemin du fichier de commande ---
        # 2. Déterminer un chemin sûr pour le fichier de commande
        try:
            # Utiliser tempfile pour obtenir le répertoire temporaire système
            temp_dir = tempfile.gettempdir()
            # Créer un sous-dossier spécifique à l'application pour éviter les conflits
            app_temp_dir = os.path.join(temp_dir, "seestar_stacker_comm")
            os.makedirs(app_temp_dir, exist_ok=True)
            # Nom du fichier de commande (relativement unique)
            # On pourrait ajouter un PID ou timestamp pour plus de robustesse si plusieurs instances tournent
            command_filename = f"analyzer_stack_command_{os.getpid()}.txt"
            self.analyzer_command_file_path = os.path.join(
                app_temp_dir, command_filename
            )  # <-- Stocker le chemin dans l'instance
            print(f"DEBUG (GUI): Chemin fichier commande défini: {self.analyzer_command_file_path}")  # <-- AJOUTÉ DEBUG

            # --- Nettoyer ancien fichier de commande s'il existe (sécurité) ---
            if os.path.exists(self.analyzer_command_file_path):
                print(
                    f"DEBUG (GUI): Suppression ancien fichier commande existant: {self.analyzer_command_file_path}"
                )  # <-- AJOUTÉ DEBUG
                try:
                    os.remove(self.analyzer_command_file_path)
                except OSError as e_rem:
                    print(f"AVERTISSEMENT (GUI): Impossible de supprimer ancien fichier commande: {e_rem}")
            # --- Fin Nettoyage ---

        except Exception as e_path:
            messagebox.showerror(
                self.tr("error"),  # Utiliser une clé générique ou créer une clé spécifique
                f"Impossible de déterminer le chemin du fichier de communication temporaire:\n{e_path}",
            )
            print(f"ERREUR (GUI): Échec détermination chemin fichier commande: {e_path}")  # <-- AJOUTÉ DEBUG
            traceback.print_exc(limit=2)
            return
        # --- FIN MODIFICATION ---

        # 3. Déterminer le chemin vers le script analyse_gui.py (inchangé)
        try:
            gui_file_path = os.path.abspath(__file__)
            gui_dir = os.path.dirname(gui_file_path)
            seestar_dir = os.path.dirname(gui_dir)
            project_root_parent = os.path.dirname(seestar_dir)
            analyzer_script_path = os.path.join(project_root_parent, "seestar", "beforehand", "analyse_gui.py")
            analyzer_script_path = os.path.normpath(analyzer_script_path)
            print(f"DEBUG (GUI): Chemin script analyseur trouvé: {analyzer_script_path}")  # <-- AJOUTÉ DEBUG
        except Exception as e:
            messagebox.showerror(
                self.tr("analyzer_launch_error_title"),
                f"Erreur interne chemin analyseur:\n{e}",
            )
            return

        # 4. Vérifier si le script analyseur existe (inchangé)
        if not os.path.exists(analyzer_script_path):
            messagebox.showerror(
                self.tr("analyzer_launch_error_title"),
                self.tr("analyzer_script_not_found").format(path=analyzer_script_path),
            )
            return

        # 5. Construire et lancer la commande (MODIFIÉ pour ajouter l'argument command_file_path)
        try:
            command = [
                sys.executable,
                analyzer_script_path,
                "--input-dir",
                input_folder,
                "--command-file",
                self.analyzer_command_file_path,
                "--lang",
                self.settings.language,
                "--lock-lang",
            ]
            print(f"DEBUG (GUI): Commande lancement analyseur: {' '.join(command)}")  # <-- AJOUTÉ DEBUG

            # Lancer comme processus séparé non bloquant
            process = subprocess.Popen(command)
            self.update_progress_gui(self.tr("analyzer_launched"), None)

            # --- NOUVEAU : Démarrer la surveillance du fichier de commande ---
            print("DEBUG (GUI): Démarrage de la surveillance du fichier commande...")  # <-- AJOUTÉ DEBUG
            # Assurer qu'une seule boucle de vérification tourne à la fois
            if hasattr(self, "_analyzer_check_after_id") and self._analyzer_check_after_id:
                print("DEBUG (GUI): Annulation surveillance précédente...")  # <-- AJOUTÉ DEBUG
                try:
                    self.root.after_cancel(self._analyzer_check_after_id)
                except tk.TclError:
                    pass  # Ignore error if already cancelled/invalid
                self._analyzer_check_after_id = None

            # Démarrer la nouvelle boucle de vérification (ex: toutes les 1 seconde)
            self._check_analyzer_command_file()
            # --- FIN NOUVEAU ---

        # 6. Gestion des erreurs de lancement (inchangé)
        except FileNotFoundError:
            messagebox.showerror(
                self.tr("analyzer_launch_error_title"),
                self.tr("analyzer_launch_failed").format(error=f"Python '{sys.executable}' or script not found."),
            )
        except OSError as e:
            messagebox.showerror(
                self.tr("analyzer_launch_error_title"),
                self.tr("analyzer_launch_failed").format(error=f"OS error: {e}"),
            )
        except Exception as e:
            messagebox.showerror(
                self.tr("analyzer_launch_error_title"),
                self.tr("analyzer_launch_failed").format(error=str(e)),
            )
            traceback.print_exc(limit=2)
        print("DEBUG (GUI): Sortie de _launch_folder_analyzer.")  # <-- AJOUTÉ DEBUG

    #############################################################################################################################

    def _check_analyzer_command_file(self):
        """
        Vérifie périodiquement l'existence du fichier de commande de l'analyseur.
        Si trouvé, lit le chemin, le supprime, et lance le stacking.
        """
        # print("DEBUG (GUI): _check_analyzer_command_file() exécuté.") # <-- DEBUG (peut être trop verbeux)

        # --- Vérifications préliminaires ---
        # 1. Le chemin du fichier de commande est-il défini ?
        if not hasattr(self, "analyzer_command_file_path") or not self.analyzer_command_file_path:
            print("DEBUG (GUI): Vérification fichier commande annulée (chemin non défini).")
            self._analyzer_check_after_id = None  # Assurer l'arrêt
            return

        # 2. Un traitement est-il déjà en cours dans le stacker principal ?
        if self.processing:
            # print("DEBUG (GUI): Traitement principal en cours, arrêt temporaire surveillance fichier commande.")
            # Pas besoin de vérifier le fichier si on est déjà en train de stacker/traiter.
            # On pourrait replanifier plus tard, mais pour l'instant, on arrête la surveillance
            # une fois le traitement principal démarré.
            self._analyzer_check_after_id = None  # Arrêter la boucle si traitement lancé par autre chose
            return

        # 3. Le fichier de commande existe-t-il ?
        try:
            if os.path.exists(self.analyzer_command_file_path):
                print(f"DEBUG (GUI): Fichier commande détecté: {self.analyzer_command_file_path}")  # <-- AJOUTÉ DEBUG

                # --- Traitement du fichier ---
                folder_path = None
                ref_path = None
                try:
                    with open(self.analyzer_command_file_path, "r", encoding="utf-8") as f_cmd:
                        lines = [ln.strip() for ln in f_cmd.readlines()]
                    folder_path = lines[0] if lines else None
                    ref_path = lines[1] if len(lines) > 1 else None
                    print(f"DEBUG (GUI): Contenu fichier commande lu: '{lines}'")

                    # Supprimer le fichier IMMÉDIATEMENT après lecture réussie
                    try:
                        os.remove(self.analyzer_command_file_path)
                        print(f"DEBUG (GUI): Fichier commande supprimé.")  # <-- AJOUTÉ DEBUG
                    except OSError as e_rem:
                        print(
                            f"AVERTISSEMENT (GUI): Échec suppression fichier commande {self.analyzer_command_file_path}: {e_rem}"
                        )
                        # Continuer quand même si la lecture a réussi

                except IOError as e_read:
                    print(
                        f"ERREUR (GUI): Impossible de lire le fichier commande {self.analyzer_command_file_path}: {e_read}"
                    )
                    # Essayer de supprimer le fichier même si lecture échoue (peut être corrompu)
                    try:
                        os.remove(self.analyzer_command_file_path)
                    except OSError:
                        pass
                    # Replanifier la vérification car on n'a pas pu traiter
                    if hasattr(self.root, "after"):  # Vérifier si root existe toujours
                        self._analyzer_check_after_id = self.root.after(
                            1000, self._check_analyzer_command_file
                        )  # Replanifier dans 1s
                    return  # Sortir pour cette itération

                # --- Agir sur le contenu lu ---
                if folder_path and os.path.isdir(folder_path):
                    analyzed_folder_path = os.path.abspath(folder_path)
                    print(f"INFO (GUI): Commande d'empilement reçue pour: {analyzed_folder_path}")  # <-- AJOUTÉ INFO

                    # Mettre à jour le champ d'entrée
                    current_input = self.input_path.get()
                    if os.path.normpath(current_input) != os.path.normpath(analyzed_folder_path):
                        print(
                            f"DEBUG (GUI): Mise à jour du champ dossier d'entrée vers: {analyzed_folder_path}"
                        )  # <-- AJOUTÉ DEBUG
                        self.input_path.set(analyzed_folder_path)
                        # Mettre à jour aussi le setting pour cohérence ?
                        self.settings.input_folder = analyzed_folder_path
                        # Redessiner l'aperçu initial si le dossier a changé
                        self._try_show_first_input_image()

                    # Vérifier si le dossier de sortie est défini
                    if not self.output_path.get():
                        default_output = os.path.join(analyzed_folder_path, "stack_output_analyzer")  # Nom différent?
                        print(
                            f"INFO (GUI): Dossier sortie non défini, utilisation défaut: {default_output}"
                        )  # <-- AJOUTÉ INFO
                        self.output_path.set(default_output)
                        self.settings.output_folder = default_output

                    if ref_path:
                        print(f"DEBUG (GUI): Référence recommandée reçue: {ref_path}")
                        self.reference_image_path.set(ref_path)
                        self.settings.reference_image_path = ref_path

                    # Démarrer le stacking
                    print(
                        "DEBUG (GUI): Appel de self.start_processing() suite à commande analyseur..."
                    )  # <-- AJOUTÉ DEBUG
                    self.start_processing()
                    # Pas besoin de replanifier la vérification, le but est atteint.
                    self._analyzer_check_after_id = None
                    return  # Sortir de la méthode

                else:
                    print(
                        f"AVERTISSEMENT (GUI): Contenu fichier commande invalide ('{lines}') ou n'est pas un dossier. Fichier supprimé."
                    )
                    # Replanifier la vérification car le contenu était invalide
                    if hasattr(self.root, "after"):  # Vérifier si root existe toujours
                        self._analyzer_check_after_id = self.root.after(
                            1000, self._check_analyzer_command_file
                        )  # Replanifier dans 1s
                    return  # Sortir pour cette itération

            else:
                # --- Replanifier si le fichier n'existe pas ---
                # print("DEBUG (GUI): Fichier commande non trouvé, replanification...") # <-- DEBUG (trop verbeux)
                # Vérifier si la fenêtre racine existe toujours avant de replanifier
                if hasattr(self.root, "after"):
                    self._analyzer_check_after_id = self.root.after(
                        1000, self._check_analyzer_command_file
                    )  # Vérifier à nouveau dans 1000 ms (1 seconde)
                else:
                    print("DEBUG (GUI): Fenêtre racine détruite, arrêt surveillance fichier commande.")
                    self._analyzer_check_after_id = None  # Arrêter si la fenêtre est fermée

        except Exception as e_check:
            print(f"ERREUR (GUI): Erreur inattendue dans _check_analyzer_command_file: {e_check}")
            traceback.print_exc(limit=2)
            # Essayer de replanifier même en cas d'erreur pour ne pas bloquer
            if hasattr(self.root, "after"):
                try:
                    self._analyzer_check_after_id = self.root.after(
                        2000, self._check_analyzer_command_file
                    )  # Attendre un peu plus longtemps après une erreur
                except tk.TclError:
                    self._analyzer_check_after_id = None  # Arrêter si la fenêtre est fermée
            else:
                self._analyzer_check_after_id = None

    #############################################################################################################################

    def _update_show_folders_button_state(self, event=None):
        """Enables/disables the 'View Inputs' and 'Analyze Input' buttons."""
        try:
            # Déterminer l'état basé sur la validité du dossier d'entrée
            is_input_valid = self.input_path.get() and os.path.isdir(self.input_path.get())
            new_state = tk.NORMAL if is_input_valid else tk.DISABLED

            # Mettre à jour le bouton "View Inputs"
            if hasattr(self, "show_folders_button") and self.show_folders_button.winfo_exists():
                self.show_folders_button.config(state=new_state)

            # Mettre à jour le bouton "Analyze Input Folder"
            if hasattr(self, "analyze_folder_button") and self.analyze_folder_button.winfo_exists():
                self.analyze_folder_button.config(state=new_state)

        except tk.TclError:
            # Ignorer les erreurs si les widgets n'existent pas encore ou sont détruits
            pass
        except Exception as e:
            print(f"Error in _update_show_folders_button_state: {e}")

    def _on_last_stack_changed(self, *args):
        """When last stack path changes, pre-fill output folder if empty."""
        try:
            if not self.output_path.get():
                p = self.last_stack_path.get()
                if p:
                    self.output_path.set(os.path.dirname(p))
        except Exception:
            pass

    #############################################################################################################################

    def _show_input_folder_list(self):
        """Displays the list of input folders in a simple pop-up window."""

        main_folder = self.input_path.get()

        if not main_folder or not os.path.isdir(main_folder):
            messagebox.showwarning(self.tr("warning"), self.tr("no_input_folder_set"))
            return

        folder_list = []
        # Always add the main input folder first
        folder_list.append(os.path.abspath(main_folder))

        additional_folders = []
        # Get additional folders based on processing state
        if self.processing and hasattr(self, "queued_stacker") and self.queued_stacker:
            try:
                # Safely access the list using the lock
                with self.queued_stacker.folders_lock:
                    # Get a copy to avoid issues if modified during iteration
                    additional_folders = list(self.queued_stacker.additional_folders)
            except Exception as e:
                print(f"Error accessing queued stacker folders: {e}")
                # Proceed without additional folders from backend if error occurs
        else:
            # Get folders added before processing started
            additional_folders = self.additional_folders_to_process

        # Add absolute paths of additional folders
        for folder in additional_folders:
            abs_path = os.path.abspath(folder)
            if abs_path not in folder_list:  # Avoid duplicates if added strangely
                folder_list.append(abs_path)

        # --- Format the text ---
        if len(folder_list) == 1:
            display_text = f"{self.tr('input_folder')}\n  {folder_list[0]}"
        else:
            display_text = f"{self.tr('input_folder')} (Main):\n  {folder_list[0]}\n\n"
            display_text += f"{self.tr('Additional:')} ({len(folder_list) - 1})\n"
            for i, folder in enumerate(folder_list[1:], 1):
                display_text += f"  {i}. {folder}\n"

        # --- Create the Toplevel window ---
        try:
            dialog = tk.Toplevel(self.root)
            dialog.title(self.tr("input_folders_title"))
            dialog.transient(self.root)  # Associate with main window
            dialog.grab_set()  # Make modal
            dialog.resizable(False, False)  # Prevent resizing

            # --- Add Frame, Scrollbar, and Text Widget ---
            frame = ttk.Frame(dialog, padding="10")
            frame.pack(fill=tk.BOTH, expand=True)

            scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
            # Use a fixed-size font for better alignment if needed
            # text_font = tkFont.Font(family="Courier New", size=9)
            list_text = tk.Text(
                frame,
                wrap=tk.WORD,
                height=15,  # Adjust height as needed
                width=80,  # Adjust width as needed
                yscrollcommand=scrollbar.set,
                # font=text_font, # Optional fixed font
                padx=5,
                pady=5,
                state=tk.DISABLED,  # Start disabled
            )
            scrollbar.config(command=list_text.yview)

            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            list_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Insert the text
            list_text.config(state=tk.NORMAL)
            list_text.delete(1.0, tk.END)
            list_text.insert(tk.END, display_text)
            list_text.config(state=tk.DISABLED)  # Make read-only

            # --- Add Close Button ---
            button_frame = ttk.Frame(dialog, padding="0 10 10 10")
            button_frame.pack(fill=tk.X)
            close_button = ttk.Button(button_frame, text="Close", command=dialog.destroy)  # Add translation if needed
            # Use pack with anchor or alignment if needed
            close_button.pack(anchor=tk.CENTER)  # Center the button
            close_button.focus_set()  # Set focus to close button

            # --- Center the dialog ---
            dialog.update_idletasks()
            root_x = self.root.winfo_rootx()
            root_y = self.root.winfo_rooty()
            root_w = self.root.winfo_width()
            root_h = self.root.winfo_height()
            dlg_w = dialog.winfo_width()
            dlg_h = dialog.winfo_height()
            x = root_x + (root_w // 2) - (dlg_w // 2)
            y = root_y + (root_h // 2) - (dlg_h // 2)
            dialog.geometry(f"+{x}+{y}")

            # --- Wait for dialog ---
            self.root.wait_window(dialog)

        except tk.TclError as e:
            print(f"Error creating folder list dialog: {e}")
            # Fallback to simple message box if Toplevel fails
            messagebox.showinfo(self.tr("input_folders_title"), display_text)
        except Exception as e:
            print(f"Unexpected error showing folder list: {e}")
            traceback.print_exc(limit=2)
            messagebox.showerror(self.tr("error"), f"Failed to display folder list:\n{e}")

    def _create_slider_spinbox_group(self, parent, label_key, min_val, max_val, step, tk_var, callback=None):
        """Helper to create a consistent Slider + SpinBox group with debouncing."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=(1, 1))
        label_widget = ttk.Label(frame, text=self.tr(label_key, default=label_key), width=10)
        label_widget.pack(side=tk.LEFT)
        decimals = 0
        log_step = -3
        if step > 0:
            try:
                log_step = math.log10(step)
            except ValueError:
                pass
            if log_step < 0:
                decimals = abs(int(log_step))
        format_str = f"%.{decimals}f"
        spinbox = ttk.Spinbox(
            frame,
            from_=min_val,
            to=max_val,
            increment=step,
            textvariable=tk_var,
            width=7,
            justify=tk.RIGHT,
            command=self._debounce_refresh_preview,
            format=format_str,
        )
        spinbox.pack(side=tk.RIGHT, padx=(5, 0))

        def on_scale_change(value_str):
            try:
                value = float(value_str)
            except ValueError:
                return
            if callback:
                try:
                    callback(value)
                except Exception as cb_err:
                    print(f"Error in slider immediate callback: {cb_err}")
            self._debounce_refresh_preview()

        slider = ttk.Scale(
            frame,
            from_=min_val,
            to=max_val,
            variable=tk_var,
            orient=tk.HORIZONTAL,
            command=on_scale_change,
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ctrls = {
            "frame": frame,
            "label": label_widget,
            "slider": slider,
            "spinbox": spinbox,
        }
        return ctrls

    ##################################################################################################################

    # --- DANS LA CLASSE SeestarStackerGUI DANS seestar/gui/main_window.py ---

    def _store_widget_references(self):
        """
        Stocke les références aux widgets qui nécessitent des mises à jour linguistiques et des infobulles.
        MODIFIED: Ajout des références pour save_as_float32_check et preserve_linear_output_check.
        """
        print(
            "\nDEBUG (GUI _store_widget_references V_SaveAsFloat32_1): Début stockage références widgets..."
        )  # Version Log
        notebook_widget = None
        try:
            if hasattr(self, "control_notebook") and isinstance(self.control_notebook, ttk.Notebook):
                notebook_widget = self.control_notebook
        except Exception as e:
            print(f"Warning (GUI _store_widget_references): Erreur accès control_notebook: {e}")

        self.widgets_to_translate = {}

        # --- Onglets du Notebook ---
        try:
            if notebook_widget and notebook_widget.winfo_exists():
                if notebook_widget.index("end") > 0:
                    self.widgets_to_translate["tab_stacking"] = (notebook_widget, 0)
                if notebook_widget.index("end") > 1:
                    self.widgets_to_translate["tab_expert_title"] = (notebook_widget, 1)
                if notebook_widget.index("end") > 2:
                    self.widgets_to_translate["tab_preview"] = (notebook_widget, 2)
        except tk.TclError:
            print("DEBUG: Erreur accès onglets (probablement non encore tous créés).")

        # --- LabelFrames (cadres avec titre) ---
        label_frames_keys = {
            "Folders": "folders_frame",
            "options": "options_frame",
            "drizzle_options_frame_label": "drizzle_options_frame",
            "hot_pixels_correction": "hp_frame",
            "post_proc_opts_frame_label": "post_proc_opts_frame",
            "white_balance": "wb_frame",
            "stretch_options": "stretch_frame_controls",
            "image_adjustments": "bcs_frame",
            "progress": "progress_frame",
            "preview": "preview_frame",
            "histogram": "histogram_frame",
            "bn_frame_title": "bn_frame",
            "cb_frame_title": "cb_frame",
            "crop_frame_title": "crop_frame",
            "feathering_frame_title": "feathering_frame",
            "photutils_bn_frame_title": "photutils_bn_frame",
            # NOUVEAU : Clé pour le titre du nouveau LabelFrame
            "output_format_frame_title": "output_format_frame",
        }
        for key, attr_name in label_frames_keys.items():
            self.widgets_to_translate[key] = getattr(self, attr_name, None)

        # --- Labels (étiquettes simples) & Checkbuttons (pour leur texte) ---
        labels_and_checks_keys = {
            "input_folder": "input_label",
            "output_folder": "output_label",
            "reference_image": "reference_label",
            "stack_method_label": "stack_method_label",
            "batch_size": "batch_size_label",
            "drizzle_scale_label": "drizzle_scale_label",
            "drizzle_mode_label": "drizzle_mode_label",
            "drizzle_kernel_label": "drizzle_kernel_label",
            "drizzle_pixfrac_label": "drizzle_pixfrac_label",
            "drizzle_use_gpu_label": "drizzle_use_gpu_label",
            "drizzle_wht_threshold_label": "drizzle_wht_label",
            "hot_pixel_threshold": "hot_pixel_threshold_label",
            "neighborhood_size": "neighborhood_size_label",
            "weighting_metrics_label": "weight_metrics_label",
            "snr_exponent_label": "snr_exp_label",
            "stars_exponent_label": "stars_exp_label",
            "min_weight_label": "min_w_label",
            "stacking_norm_method_label": "norm_method_label",
            "stacking_weight_method_label": "weight_method_label",
            "stacking_kappa_low_label": "kappa_low_label",
            "stacking_kappa_high_label": "kappa_high_label",
            "stacking_winsor_limits_label": "winsor_limits_label",
            "stacking_winsor_note": "winsor_note_label",
            "stacking_final_combine_label": "final_combine_label",
            "hq_ram_limit_label": "hq_ram_limit_label_widget",
            "wb_r": getattr(self, "wb_r_ctrls", {}).get("label"),
            "wb_g": getattr(self, "wb_g_ctrls", {}).get("label"),
            "wb_b": getattr(self, "wb_b_ctrls", {}).get("label"),
            "stretch_method": "stretch_method_label",
            "stretch_bp": getattr(self, "stretch_bp_ctrls", {}).get("label"),
            "stretch_wp": getattr(self, "stretch_wp_ctrls", {}).get("label"),
            "stretch_gamma": getattr(self, "stretch_gamma_ctrls", {}).get("label"),
            "brightness": getattr(self, "brightness_ctrls", {}).get("label"),
            "contrast": getattr(self, "contrast_ctrls", {}).get("label"),
            "saturation": getattr(self, "saturation_ctrls", {}).get("label"),
            "estimated_time": "remaining_time_label",
            "elapsed_time": "elapsed_time_label",
            "Remaining:": "remaining_static_label",
            "Additional:": "additional_static_label",
            "expert_warning_text": "warning_label",
            "bn_grid_size_label": "bn_grid_size_actual_label",
            "bn_perc_low_label": "bn_perc_low_actual_label",
            "bn_perc_high_label": "bn_perc_high_actual_label",
            "bn_std_factor_label": "bn_std_factor_actual_label",
            "bn_min_gain_label": "bn_min_gain_actual_label",
            "bn_max_gain_label": "bn_max_gain_actual_label",
            "cb_border_size_label": "cb_border_size_actual_label",
            "cb_blur_radius_label": "cb_blur_radius_actual_label",
            "cb_min_b_factor_label": "cb_min_b_factor_actual_label",
            "cb_max_b_factor_label": "cb_max_b_factor_actual_label",
            "final_edge_crop_label": "final_edge_crop_actual_label",
            "apply_final_scnr_label": "apply_final_scnr_check",
            "final_scnr_amount_label": getattr(self, "scnr_amount_ctrls", {}).get("label"),
            "final_scnr_preserve_lum_label": "final_scnr_preserve_lum_check",
            "apply_photutils_bn_label": "apply_photutils_bn_check",
            "photutils_bn_box_size_label": "photutils_bn_box_size_label",
            "photutils_bn_filter_size_label": "photutils_bn_filter_size_label",
            "photutils_bn_sigma_clip_label": "photutils_bn_sigma_clip_label",
            "photutils_bn_exclude_percentile_label": "photutils_bn_exclude_percentile_label",
            "apply_feathering_label": "apply_feathering_check",
            "feather_blur_px_label": "feather_blur_px_label",
            "feather_inter_batch_label": "apply_batch_feathering_check",
            "apply_low_wht_mask_label": "low_wht_mask_check",
            "low_wht_percentile_label": "low_wht_pct_label",
            "low_wht_soften_px_label": "low_wht_soften_px_label",
            "drizzle_activate_check": "drizzle_check",
            "perform_hot_pixels_correction": "hot_pixels_check",
            "enable_weighting_check": "use_weighting_check",
            "weight_snr_check": "weight_snr_check",
            "weight_stars_check": "weight_stars_check",
            "cleanup_temp_check_label": "cleanup_temp_check",
            "chroma_correction_check": "chroma_correction_check",
            # NOUVEAU : Clé pour le texte de la nouvelle Checkbutton
            "save_as_float32_label": "save_as_float32_check",
            "preserve_linear_output_label": "preserve_linear_output_check",
        }
        for key, item in labels_and_checks_keys.items():
            if isinstance(item, tk.Widget):
                self.widgets_to_translate[key] = item
            elif isinstance(item, str):
                self.widgets_to_translate[key] = getattr(self, item, None)

        buttons_keys = {
            "browse_input_button": "browse_input_button",
            "browse_output_button": "browse_output_button",
            "browse_ref_button": "browse_ref_button",
            "auto_wb": "auto_wb_button",
            "reset_wb": "reset_wb_button",
            "auto_stretch": "auto_stretch_button",
            "reset_stretch": "reset_stretch_button",
            "reset_bcs": "reset_bcs_button",
            "start": "start_button",
            "stop": "stop_button",
            "add_folder_button": "add_files_button",
            "show_folders_button_text": "show_folders_button",
            "copy_log_button_text": "copy_log_button",
            "open_output_button_text": "open_output_button",
            "analyze_folder_button": "analyze_folder_button",
            "local_solver_button_text": "local_solver_button",
            "Mosaic...": "mosaic_options_button",
            "reset_expert_button": "reset_expert_button",
            "zoom_100_button": "zoom_100_button",
            "zoom_fit_button": "zoom_fit_button",
        }
        for key, attr_name in buttons_keys.items():
            self.widgets_to_translate[key] = getattr(self, attr_name, None)

        radio_buttons_keys = {
            "drizzle_radio_2x_label": "drizzle_radio_2x",
            "drizzle_radio_3x_label": "drizzle_radio_3x",
            "drizzle_radio_4x_label": "drizzle_radio_4x",
            "drizzle_radio_final": "drizzle_radio_final",
            "drizzle_radio_incremental": "drizzle_radio_incremental",
        }
        for key, attr_name in radio_buttons_keys.items():
            self.widgets_to_translate[key] = getattr(self, attr_name, None)

        # --- TOOLTIPS ---
        self.tooltips = {}
        print(f"DEBUG (GUI _store_widget_references): Dictionnaire self.tooltips réinitialisé.")

        tooltips_config_list = [
            # ... (toutes les configurations de tooltips existantes restent ici) ...
            ("bn_grid_size_actual_label", "tooltip_bn_grid_size"),
            ("bn_grid_size_combo", "tooltip_bn_grid_size"),
            ("bn_perc_low_actual_label", "tooltip_bn_perc_low"),
            ("bn_perc_low_spinbox", "tooltip_bn_perc_low"),
            ("bn_perc_high_actual_label", "tooltip_bn_perc_high"),
            ("bn_perc_high_spinbox", "tooltip_bn_perc_high"),
            ("bn_std_factor_actual_label", "tooltip_bn_std_factor"),
            ("bn_std_factor_spinbox", "tooltip_bn_std_factor"),
            ("bn_min_gain_actual_label", "tooltip_bn_min_gain"),
            ("bn_min_gain_spinbox", "tooltip_bn_min_gain"),
            ("bn_max_gain_actual_label", "tooltip_bn_max_gain"),
            ("bn_max_gain_spinbox", "tooltip_bn_max_gain"),
            ("cb_border_size_actual_label", "tooltip_cb_border_size"),
            ("cb_border_size_spinbox", "tooltip_cb_border_size"),
            ("cb_blur_radius_actual_label", "tooltip_cb_blur_radius"),
            ("cb_blur_radius_spinbox", "tooltip_cb_blur_radius"),
            ("cb_min_b_factor_actual_label", "tooltip_cb_min_b_factor"),
            ("cb_min_b_factor_spinbox", "tooltip_cb_min_b_factor"),
            ("cb_max_b_factor_actual_label", "tooltip_cb_max_b_factor"),
            ("cb_max_b_factor_spinbox", "tooltip_cb_max_b_factor"),
            ("final_edge_crop_actual_label", "tooltip_final_edge_crop_percent"),
            ("final_edge_crop_spinbox", "tooltip_final_edge_crop_percent"),
            ("hq_ram_limit_label_widget", "tooltip_hq_ram_limit"),
            ("hq_ram_limit_spinbox", "tooltip_hq_ram_limit"),
            ("apply_final_scnr_check", "tooltip_apply_final_scnr"),
            (
                getattr(self, "scnr_amount_ctrls", {}).get("label"),
                "tooltip_final_scnr_amount",
            ),
            (
                getattr(self, "scnr_amount_ctrls", {}).get("spinbox"),
                "tooltip_final_scnr_amount",
            ),
            (
                getattr(self, "scnr_amount_ctrls", {}).get("slider"),
                "tooltip_final_scnr_amount",
            ),
            ("final_scnr_preserve_lum_check", "tooltip_final_scnr_preserve_lum"),
            ("apply_photutils_bn_check", "tooltip_apply_photutils_bn"),
            ("photutils_bn_box_size_label", "tooltip_photutils_bn_box_size"),
            ("pb_box_spinbox", "tooltip_photutils_bn_box_size"),
            ("photutils_bn_filter_size_label", "tooltip_photutils_bn_filter_size"),
            ("pb_filt_spinbox", "tooltip_photutils_bn_filter_size"),
            ("photutils_bn_sigma_clip_label", "tooltip_photutils_bn_sigma_clip"),
            ("pb_sig_spinbox", "tooltip_photutils_bn_sigma_clip"),
            (
                "photutils_bn_exclude_percentile_label",
                "tooltip_photutils_bn_exclude_percentile",
            ),
            ("pb_excl_spinbox", "tooltip_photutils_bn_exclude_percentile"),
            ("apply_feathering_check", "tooltip_apply_feathering"),
            ("feather_blur_px_label", "tooltip_feather_blur_px"),
            ("feather_blur_px_spinbox", "tooltip_feather_blur_px"),
            ("apply_batch_feathering_check", "feather_inter_batch_tooltip"),
            ("low_wht_mask_check", "tooltip_apply_low_wht_mask"),
            ("low_wht_pct_label", "tooltip_low_wht_percentile"),
            ("low_wht_pct_spinbox", "tooltip_low_wht_percentile"),
            ("low_wht_soften_px_label", "tooltip_low_wht_soften_px"),
            ("low_wht_soften_px_spinbox", "tooltip_low_wht_soften_px"),
            # NOUVEAU : Tooltip pour la nouvelle Checkbutton
            ("save_as_float32_check", "tooltip_save_as_float32"),
            ("preserve_linear_output_check", "tooltip_preserve_linear_output"),
        ]

        tooltip_created_count = 0
        for item_identifier, tooltip_translation_key in tooltips_config_list:
            widget_to_attach_tooltip = None
            debug_item_name = ""

            if isinstance(item_identifier, str):
                debug_item_name = item_identifier
                widget_to_attach_tooltip = getattr(self, item_identifier, None)
            elif isinstance(item_identifier, tk.Widget):
                widget_to_attach_tooltip = item_identifier
                try:
                    debug_item_name = (
                        f"WidgetDirect({widget_to_attach_tooltip.winfo_class()}-{id(widget_to_attach_tooltip)})"
                    )
                except:
                    debug_item_name = f"WidgetDirect(id-{id(widget_to_attach_tooltip)})"
            else:
                debug_item_name = str(item_identifier)
                print(
                    f"  Tooltip WARNING: Type d'identifiant d'item inattendu '{debug_item_name}' pour la clé tooltip '{tooltip_translation_key}'."
                )
                continue

            if (
                widget_to_attach_tooltip
                and hasattr(widget_to_attach_tooltip, "winfo_exists")
                and widget_to_attach_tooltip.winfo_exists()
            ):
                unique_tooltip_id = f"tooltip_for_widget_{id(widget_to_attach_tooltip)}_{tooltip_translation_key}"

                if unique_tooltip_id not in self.tooltips:
                    self.tooltips[unique_tooltip_id] = ToolTip(
                        widget_to_attach_tooltip,
                        lambda k=tooltip_translation_key: self.tr(k),
                    )
                    tooltip_created_count += 1

        print(
            f"DEBUG (GUI _store_widget_references V_SaveAsFloat32_1): Références et Tooltips stockés. Nb widgets trad: {len(self.widgets_to_translate)}, Nb tooltips créés dans cet appel: {tooltip_created_count}"
        )  # Version Log

    ##################################################################################################################
    def change_language(self, event=None):
        """Change l'interface à la langue sélectionnée."""
        selected_lang = self.language_var.get()
        if self.localization.language != selected_lang:
            self.localization.set_language(selected_lang)
            self.settings.language = selected_lang  # Update setting
            self.settings.save_settings()  # Save the change immediately
            self.update_ui_language()

    def update_ui_language(self):
        """Met à jour tous les textes traduisibles de l'interface."""
        self.root.title(f"{self.tr('title')}  –  {self.app_version}")
        if not hasattr(self, "widgets_to_translate"):
            print("Warning: Widget reference dictionary not found for translation.")
            return
        for key, widget_info in self.widgets_to_translate.items():
            # --- DÉBUT DEBUG SPÉCIFIQUE ---
            if key == "tab_preview":
                print(f"DEBUG UI_LANG: Traitement clé '{key}'")
                current_lang_for_tr = self.localization.language
                print(f"DEBUG UI_LANG: Langue actuelle pour self.tr: '{current_lang_for_tr}'")

                translation_directe_langue_courante = self.localization.translations[current_lang_for_tr].get(key)
                print(
                    f"DEBUG UI_LANG: Traduction directe pour '{key}' en '{current_lang_for_tr}': '{translation_directe_langue_courante}'"
                )

                traduction_fallback_anglais = self.localization.translations["en"].get(key)
                print(f"DEBUG UI_LANG: Traduction fallback anglais pour '{key}': '{traduction_fallback_anglais}'")

                default_text_calcul = self.localization.translations["en"].get(key, key.replace("_", " ").title())
                print(f"DEBUG UI_LANG: default_text calculé pour '{key}': '{default_text_calcul}'")

                translation_finale_pour_tab = self.tr(key, default=default_text_calcul)
                print(f"DEBUG UI_LANG: self.tr('{key}') a retourné: '{translation_finale_pour_tab}'")
            # --- FIN DEBUG SPÉCIFIQUE ---
            default_text = self.localization.translations["en"].get(key, key.replace("_", " ").title())
            translation = self.tr(key, default=default_text)
            try:
                if widget_info is None:
                    continue
                if isinstance(widget_info, tuple):
                    notebook, index = widget_info
                    if notebook and notebook.winfo_exists() and index < notebook.index("end"):
                        notebook.tab(index, text=f" {translation} ")
                elif hasattr(widget_info, "winfo_exists") and widget_info.winfo_exists():
                    widget = widget_info
                    if isinstance(widget, (ttk.Label, ttk.Button, ttk.Checkbutton)):
                        widget.config(text=translation)
                    elif isinstance(widget, ttk.LabelFrame):
                        widget.config(text=translation)
            except tk.TclError:
                pass
            except Exception as e:
                print(f"Debug: Error updating widget '{key}': {e}")

        # Refresh stacking method combobox with localized labels
        if hasattr(self, "method_combo"):
            self.method_key_to_label = {}
            self.method_label_to_key = {}
            for k in self.method_keys:
                label = self.tr(f"method_{k}", default=k.replace("_", " ").title())
                self.method_key_to_label[k] = label
                self.method_label_to_key[label] = k
            self.method_combo["values"] = list(self.method_key_to_label.values())
            current_key = self.stack_method_var.get()
            self.stack_method_display_var.set(self.method_key_to_label.get(current_key, current_key))
        if hasattr(self, "stack_norm_combo"):
            self.norm_key_to_label = {}
            self.norm_label_to_key = {}
            for k in self.norm_keys:
                label = self.tr(f"norm_method_{k}", default=k.replace("_", " ").title())
                self.norm_key_to_label[k] = label
                self.norm_label_to_key[label] = k
            self.stack_norm_combo["values"] = list(self.norm_key_to_label.values())
            current_key = self.stack_norm_method_var.get()
            self.stack_norm_display_var.set(self.norm_key_to_label.get(current_key, current_key))
        if hasattr(self, "stack_weight_combo"):
            self.weight_key_to_label = {}
            self.weight_label_to_key = {}
            for k in self.weight_keys:
                label = self.tr(f"weight_method_{k}", default=k.replace("_", " ").title())
                self.weight_key_to_label[k] = label
                self.weight_label_to_key[label] = k
            self.stack_weight_combo["values"] = list(self.weight_key_to_label.values())
            current_key = self.stack_weight_method_var.get()
            self.stack_weight_display_var.set(self.weight_key_to_label.get(current_key, current_key))
        if hasattr(self, "stack_final_combo"):
            self.final_key_to_label = {}
            self.final_label_to_key = {}
            for k in self.final_keys:
                label = self.tr(f"combine_method_{k}", default=k.replace("_", " ").title())
                self.final_key_to_label[k] = label
                self.final_label_to_key[label] = k
            self.stack_final_combo["values"] = list(self.final_key_to_label.values())
            if self.reproject_between_batches_var.get():
                self.stack_final_display_var.set(self.final_key_to_label.get("reproject", "reproject"))
            else:
                current_key = self.stack_final_combine_var.get()
                self.stack_final_display_var.set(self.final_key_to_label.get(current_key, current_key))
        # Update dynamic text variables
        if not self.processing:
            self.remaining_files_var.set(self.tr("no_files_waiting"))
            default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
            self.aligned_files_var.set(default_aligned_fmt.format(count="--"))
            self.remaining_time_var.set("--:--:--")
        else:  # Ensure static labels are translated if processing
            if hasattr(self, "remaining_static_label"):
                self.remaining_static_label.config(text=self.tr("Remaining:"))
            if hasattr(self, "additional_static_label"):
                self.additional_static_label.config(text=self.tr("Additional:"))
            if hasattr(self, "elapsed_time_label"):
                self.elapsed_time_label.config(text=self.tr("elapsed_time"))
            if hasattr(self, "remaining_time_label"):
                self.remaining_time_label.config(text=self.tr("estimated_time"))
            # Update dynamic counts using current language format string
            if hasattr(self, "queued_stacker"):
                count = self.queued_stacker.aligned_files_count
                default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
                self.aligned_files_var.set(default_aligned_fmt.format(count=count))
                self.update_remaining_files()  # Re-calculate R/T display

        self.update_additional_folders_display()  # Update folder count display text

        if self.current_preview_data is None and hasattr(self, "preview_manager"):
            self.preview_manager.clear_preview(self.tr("Select input/output folders."))

    def update_histogram_lines_from_sliders(self, *args):
        if hasattr(self, "histogram_widget") and self.histogram_widget:
            try:
                bp = self.preview_black_point.get()
                wp = self.preview_white_point.get()
            except tk.TclError:
                return
            self.histogram_widget.set_range(bp, wp)

    # --- DANS LA CLASSE SeestarStackerGUI DANS seestar/gui/main_window.py ---

    def update_stretch_from_histogram(self, black_point_from_histo, white_point_from_histo):
        """
        Met à jour les paramètres de stretch de l'aperçu (sliders 0-1)
        à partir des valeurs reçues de l'histogramme (qui peuvent être en ADU ou 0-1).
        MODIFIED: Appelle un refresh léger de l'aperçu.
        Version: HistoCallbackRefreshLight_1
        """
        print(
            f"DEBUG GUI (update_stretch_from_histogram V_HistoCallbackRefreshLight_1): Reçu BP_histo={black_point_from_histo:.4g}, WP_histo={white_point_from_histo:.4g}"
        )

        data_source_for_histo_min = 0.0
        data_source_for_histo_max = 1.0

        is_histo_on_adu = False
        if (
            hasattr(self, "queued_stacker")
            and self.queued_stacker
            and getattr(self.queued_stacker, "save_final_as_float32", False)
            and hasattr(self.queued_stacker, "raw_adu_data_for_ui_histogram")
            and self.queued_stacker.raw_adu_data_for_ui_histogram is not None
        ):

            raw_adu_data = self.queued_stacker.raw_adu_data_for_ui_histogram
            # Vérifier si l'histogramme utilise effectivement ces données ADU
            # (Ceci est un peu une heuristique, on pourrait le rendre plus robuste)
            if (
                hasattr(self.histogram_widget, "data_min_for_current_plot")
                and self.histogram_widget.data_max_for_current_plot > 1.5
            ):
                is_histo_on_adu = True
                data_source_for_histo_min = self.histogram_widget.data_min_for_current_plot
                data_source_for_histo_max = self.histogram_widget.data_max_for_current_plot
                print(
                    f"  -> Histogramme opérait sur données ADU. Plage histo: [{data_source_for_histo_min:.4g}, {data_source_for_histo_max:.4g}]"
                )

        if not is_histo_on_adu:
            print("  -> Histogramme opérait sur données 0-1 (ou plage non ADU détectée).")
            # Si ce n'est pas ADU, on suppose que les BP/WP de l'histo sont déjà 0-1 ou proches
            data_source_for_histo_min = getattr(self.histogram_widget, "data_min_for_current_plot", 0.0)
            data_source_for_histo_max = getattr(self.histogram_widget, "data_max_for_current_plot", 1.0)

        range_histo_data = data_source_for_histo_max - data_source_for_histo_min
        if range_histo_data < 1e-9:
            range_histo_data = 1.0

        bp_ui_01 = (black_point_from_histo - data_source_for_histo_min) / range_histo_data
        wp_ui_01 = (white_point_from_histo - data_source_for_histo_min) / range_histo_data

        bp_ui_01 = np.clip(bp_ui_01, 0.0, 1.0)
        wp_ui_01 = np.clip(wp_ui_01, 0.0, 1.0)
        if wp_ui_01 <= bp_ui_01 + 1e-4:
            wp_ui_01 = min(1.0, bp_ui_01 + 1e-4)
        if bp_ui_01 >= wp_ui_01 - 1e-4:
            bp_ui_01 = max(0.0, wp_ui_01 - 1e-4)

        print(f"  -> Valeurs normalisées 0-1 pour UI sliders: BP_UI={bp_ui_01:.4f}, WP_UI={wp_ui_01:.4f}")

        try:
            self.preview_black_point.set(round(bp_ui_01, 4))
            self.preview_white_point.set(round(wp_ui_01, 4))
        except tk.TclError:
            return

        # Mettre à jour les sliders physiques (au cas où la liaison tk_var ne suffirait pas)
        try:
            if hasattr(self, "stretch_bp_ctrls") and self.stretch_bp_ctrls["slider"].winfo_exists():
                self.stretch_bp_ctrls["slider"].set(bp_ui_01)
            if hasattr(self, "stretch_wp_ctrls") and self.stretch_wp_ctrls["slider"].winfo_exists():
                self.stretch_wp_ctrls["slider"].set(wp_ui_01)
        except tk.TclError:
            pass

        # Appeler un refresh léger de l'aperçu qui ne recalcule pas l'histogramme
        self._hist_range_update_pending = True
        self._debounce_refresh_preview(recalculate_histogram=False)

    def _debounce_refresh_preview(self, *args, recalculate_histogram=True):  # Ajout argument
        """Debounces preview refresh calls."""
        if self.debounce_timer_id:
            try:
                self.root.after_cancel(self.debounce_timer_id)
            except tk.TclError:
                pass
        try:
            # Passer l'argument recalculate_histogram à la fonction cible
            self.debounce_timer_id = self.root.after(
                150,
                lambda rh=recalculate_histogram: self.refresh_preview(recalculate_histogram=rh),
            )
        except tk.TclError:
            pass

    def _reset_expert_settings(self):
        """Réinitialise les paramètres de l'onglet Expert à leurs valeurs par défaut
        telles que définies dans SettingsManager.reset_to_defaults()."""
        print("DEBUG (GUI _reset_expert_settings): Réinitialisation des paramètres Expert...")

        # Créer une instance temporaire de SettingsManager pour obtenir ses valeurs par défaut
        default_settings = SettingsManager()
        # reset_to_defaults() est appelé implicitement par __init__,
        # ou vous pouvez l'appeler explicitement si __init__ fait autre chose.
        # Pour être sûr, on peut faire:
        # default_settings.reset_to_defaults() # Mais __init__ le fait déjà.

        try:
            # Neutralisation du Fond (BN)
            if hasattr(self, "bn_grid_size_str_var"):
                self.bn_grid_size_str_var.set(default_settings.bn_grid_size_str)
            if hasattr(self, "bn_perc_low_var"):
                self.bn_perc_low_var.set(default_settings.bn_perc_low)
            if hasattr(self, "bn_perc_high_var"):
                self.bn_perc_high_var.set(default_settings.bn_perc_high)
            if hasattr(self, "bn_std_factor_var"):
                self.bn_std_factor_var.set(default_settings.bn_std_factor)
            if hasattr(self, "bn_min_gain_var"):
                self.bn_min_gain_var.set(default_settings.bn_min_gain)
            if hasattr(self, "bn_max_gain_var"):
                self.bn_max_gain_var.set(default_settings.bn_max_gain)
            if hasattr(self, "apply_bn_var"):
                self.apply_bn_var.set(default_settings.apply_bn)

            # ChromaticBalancer (CB)
            if hasattr(self, "cb_border_size_var"):
                self.cb_border_size_var.set(default_settings.cb_border_size)
            if hasattr(self, "cb_blur_radius_var"):
                self.cb_blur_radius_var.set(default_settings.cb_blur_radius)
            if hasattr(self, "cb_min_b_factor_var"):
                self.cb_min_b_factor_var.set(default_settings.cb_min_b_factor)
            if hasattr(self, "cb_max_b_factor_var"):
                self.cb_max_b_factor_var.set(default_settings.cb_max_b_factor)
            if hasattr(self, "apply_cb_var"):
                self.apply_cb_var.set(default_settings.apply_cb)

            if hasattr(self, "apply_master_tile_crop_var"):
                self.apply_master_tile_crop_var.set(default_settings.apply_master_tile_crop)
            if hasattr(self, "master_tile_crop_percent_var"):
                self.master_tile_crop_percent_var.set(default_settings.master_tile_crop_percent)

            # Rognage Final
            if hasattr(self, "final_edge_crop_percent_var"):
                self.final_edge_crop_percent_var.set(default_settings.final_edge_crop_percent)
            if hasattr(self, "apply_final_crop_var"):
                self.apply_final_crop_var.set(default_settings.apply_final_crop)

            # --- Réinitialiser Feathering ---
            if hasattr(self, "apply_feathering_var"):
                self.apply_feathering_var.set(default_settings.apply_feathering)  # Sera False par défaut
            if hasattr(self, "feather_blur_px_var"):
                self.feather_blur_px_var.set(default_settings.feather_blur_px)  # Sera 256 par défaut
            if hasattr(self, "apply_batch_feathering_var"):
                self.apply_batch_feathering_var.set(default_settings.apply_batch_feathering)
                self._on_apply_batch_feathering_changed()
            # ---  ---

            # --- Réinitialiser Photutils BN ---
            if hasattr(self, "apply_photutils_bn_var"):
                self.apply_photutils_bn_var.set(default_settings.apply_photutils_bn)  # Sera False par défaut
            if hasattr(self, "photutils_bn_box_size_var"):
                self.photutils_bn_box_size_var.set(default_settings.photutils_bn_box_size)
            if hasattr(self, "photutils_bn_filter_size_var"):
                self.photutils_bn_filter_size_var.set(default_settings.photutils_bn_filter_size)
            if hasattr(self, "photutils_bn_sigma_clip_var"):
                self.photutils_bn_sigma_clip_var.set(default_settings.photutils_bn_sigma_clip)
            if hasattr(self, "photutils_bn_exclude_percentile_var"):
                self.photutils_bn_exclude_percentile_var.set(default_settings.photutils_bn_exclude_percentile)
            # ---  ---

            # Mettre à jour l'état des widgets après réinitialisation
            # C'est important que ces appels soient APRÈS avoir .set() les BooleanVar
            if hasattr(self, "_update_photutils_bn_options_state"):
                self._update_photutils_bn_options_state()
            if hasattr(self, "_update_feathering_options_state"):
                self._update_feathering_options_state()
            if hasattr(self, "_update_bn_options_state"):
                self._update_bn_options_state()
            if hasattr(self, "_update_cb_options_state"):
                self._update_cb_options_state()
            if hasattr(self, "_update_crop_options_state"):
                self._update_crop_options_state()
            if hasattr(self, "_update_master_tile_crop_state"):
                self._update_master_tile_crop_state()
            # Si d'autres groupes d'options dans l'onglet expert ont des états dépendants,
            # appelez leurs méthodes _update_..._state() ici aussi.

            self.update_progress_gui("ⓘ Réglages Expert réinitialisés aux valeurs par défaut.", None)
            print("DEBUG (GUI _reset_expert_settings): Paramètres Expert réinitialisés dans l'UI.")

        except tk.TclError as e:
            print(f"ERREUR (GUI _reset_expert_settings): Erreur Tcl lors de la réinitialisation des widgets: {e}")
        except AttributeError as e:
            print(f"ERREUR (GUI _reset_expert_settings): Erreur d'attribut (widget ou variable Tk manquant?): {e}")
            traceback.print_exc(limit=1)  # Pour voir quel attribut manque
        except Exception as e:
            print(f"ERREUR (GUI _reset_expert_settings): Erreur inattendue: {e}")
            traceback.print_exc(limit=1)

    ###########################################################################################################################################

    # DANS LA CLASSE SeestarStackerGUI DANS seestar/gui/main_window.py

    def update_preview_from_stacker(
        self,
        preview_array,
        stack_header,
        stack_name,
        img_count,
        total_imgs,
        current_batch,
        total_batches,
    ):
        """Callback function triggered by the backend worker.
        MODIFIED: Performs auto-stretch/WB ONLY ONCE at the beginning of a session.
        """
        if threading.current_thread() is not threading.main_thread():
            self.root.after(
                0,
                lambda pa=preview_array, sh=stack_header, sn=stack_name, ic=img_count, ti=total_imgs, cb=current_batch, tb=total_batches: self.update_preview_from_stacker(
                    pa, sh, sn, ic, ti, cb, tb
                ),
            )
            return
        self.logger.debug("[DEBUG-GUI] update_preview_from_stacker: Called.")

        if isinstance(preview_array, (tuple, list)) and len(preview_array) == 2:
            preview_display, preview_hist = preview_array
        else:
            preview_display = preview_array
            preview_hist = preview_array

        if self._final_stretch_set_by_processing_finished:
            self.logger.info("  [update_preview] Verrou final actif. Mise à jour des données uniquement.")
            if preview_array is not None:
                self.current_preview_data = preview_display.copy()
                self.current_preview_hist_data = preview_hist.copy()
                self.current_stack_header = stack_header.copy() if stack_header else None
                self.preview_img_count = img_count
                if getattr(self, "preview_total_imgs", 0) == 0:
                    self.preview_total_imgs = total_imgs
                self.preview_current_batch = current_batch
                self.preview_total_batches = total_batches
                self.refresh_preview(recalculate_histogram=True)
            return

        if preview_array is None:
            self.logger.info(
                "[DEBUG-GUI] update_preview_from_stacker: Received None stack_data. Skipping visual update."
            )
            return

        self.current_preview_data = preview_display.copy()
        self.current_preview_hist_data = preview_hist.copy()
        self.current_stack_header = stack_header.copy() if stack_header else None
        self.preview_img_count = img_count
        if getattr(self, "preview_total_imgs", 0) == 0:
            self.preview_total_imgs = total_imgs
        self.preview_current_batch = current_batch
        self.preview_total_batches = total_batches

        if not self.initial_auto_stretch_done:
            self.logger.info(
                "  [update_preview] Première mise à jour de l'aperçu : déclenchement de l'auto-ajustement initial."
            )
            self.update_progress_gui("ⓘ Ajustement automatique initial de l'aperçu...", None)
            try:
                self.apply_auto_white_balance()
                self.apply_auto_stretch()
                self.initial_auto_stretch_done = True
                self.logger.info("  [update_preview] Flag initial_auto_stretch_done mis à True.")
            except Exception as e:
                self.logger.error(f"  [update_preview] Échec lors de l'auto-ajustement initial : {e}")
        else:
            self.logger.debug(
                "  [update_preview] Mise à jour de l'aperçu suivante : simple rafraîchissement sans auto-ajustement."
            )
            if self.drizzle_mode_var.get() == "Incremental":
                self.current_preview_data = preview_display
                self.current_preview_hist_data = preview_hist
                self.refresh_preview()
                return
            # Non-drizzle modes keep existing behaviour
            self.refresh_preview()

        if self.current_stack_header:
            try:
                self.root.after_idle(lambda h=self.current_stack_header: (self.update_image_info(h) if h else None))
            except tk.TclError:
                pass

    def apply_auto_stretch(self):
        """Compute auto-stretch on a worker thread."""

        if (
            hasattr(self, "_final_stretch_set_by_processing_finished")
            and self._final_stretch_set_by_processing_finished
        ):
            self.logger.warning(
                "APPLY_AUTO_STRETCH (Main_Window) appelé MAIS VERROUILLÉ par _final_stretch_set_by_processing_finished. Ignoré."
            )
            return

        def _worker(data, wb_data):
            if not _tools_available:
                self.root.after(0, lambda: messagebox.showerror(self.tr("error"), "Stretch tools not available."))
                return

            if data is None:
                self.root.after(
                    0,
                    lambda: messagebox.showwarning(
                        self.tr("warning"), self.tr("Auto Stretch: No data to analyze.")
                    ),
                )
                return

            try:
                bp_calc, wp_calc = calculate_auto_stretch(data)
            except Exception as e:
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        self.tr("error"), f"{self.tr('Error during Auto Stretch')}: {e}"
                    ),
                )
                self.logger.error(f"apply_auto_stretch: {e}")
                return

            min_data_val = np.nanmin(data)
            max_data_val = np.nanmax(data)
            range_data = max_data_val - min_data_val
            if range_data < 1e-9:
                range_data = 1.0

            bp_ui_01 = (bp_calc - min_data_val) / range_data
            wp_ui_01 = (wp_calc - min_data_val) / range_data

            bp_ui_01 = np.clip(bp_ui_01, 0.0, 1.0)
            wp_ui_01 = np.clip(wp_ui_01, 0.0, 1.0)
            if wp_ui_01 <= bp_ui_01 + 1e-4:
                wp_ui_01 = min(1.0, bp_ui_01 + 1e-4)
            if bp_ui_01 >= wp_ui_01 - 1e-4:
                bp_ui_01 = max(0.0, wp_ui_01 - 1e-4)

            def _apply():
                self.preview_black_point.set(round(bp_ui_01, 4))
                self.preview_white_point.set(round(wp_ui_01, 4))
                self.preview_stretch_method.set("Asinh")
                if hasattr(self, "histogram_widget") and self.histogram_widget:
                    self.histogram_widget.set_range(bp_ui_01, wp_ui_01)
                self.update_progress_gui(
                    f"Auto Stretch (Asinh) appliqué (Aperçu): BP={bp_ui_01:.3f} WP={wp_ui_01:.3f}",
                    None,
                )
                self.refresh_preview(recalculate_histogram=True)

            self.root.after(0, _apply)

        data = None
        if hasattr(self, "preview_manager") and self.preview_manager.image_data_wb is not None:
            data = self.preview_manager.image_data_wb.copy()
        elif self.current_preview_data is not None:
            if self.current_preview_data.ndim == 3:
                try:
                    r_gain = self.preview_r_gain.get()
                    g_gain = self.preview_g_gain.get()
                    b_gain = self.preview_b_gain.get()
                    data = ColorCorrection.white_balance(
                        self.current_preview_data, r_gain, g_gain, b_gain
                    )
                except Exception:
                    data = self.current_preview_data.copy()
            else:
                data = self.current_preview_data.copy()

        threading.Thread(target=_worker, args=(data, None), daemon=True, name="AutoStretchWorker").start()

    def refresh_preview(self, recalculate_histogram=True):  # <--- SIGNATURE CORRIGÉE ICI
        """
        Refreshes the preview based on current data and UI settings.
        MODIFIED: Ajout paramètre recalculate_histogram.
        Version: RefreshHistoControl_1_SignatureFix
        """
        print(
            f"[DEBUG-GUI] refresh_preview V_RefreshHistoControl_1_SignatureFix: Called. Recalculate Histo: {recalculate_histogram}"
        )

        if self.debounce_timer_id:
            try:
                self.root.after_cancel(self.debounce_timer_id)
            except tk.TclError:
                pass
            self.debounce_timer_id = None

        if (
            self.current_preview_data is None
            or not hasattr(self, "preview_manager")
            or self.preview_manager is None
            or not hasattr(self, "histogram_widget")
            or self.histogram_widget is None
        ):
            print("  [RefreshPreview] No data or managers. Checking for first input image.")
            if not self.processing and self.input_path.get() and os.path.isdir(self.input_path.get()):
                self._try_show_first_input_image()
            else:
                if hasattr(self, "preview_manager") and self.preview_manager:
                    self.preview_manager.clear_preview(self.tr("Select input/output folders."))
                if hasattr(self, "histogram_widget") and self.histogram_widget and recalculate_histogram:
                    self.histogram_widget.plot_histogram(None)
            return

        try:
            preview_params = {
                "stretch_method": self.preview_stretch_method.get(),
                "black_point": self.preview_black_point.get(),
                "white_point": self.preview_white_point.get(),
                "gamma": self.preview_gamma.get(),
                "r_gain": self.preview_r_gain.get(),
                "g_gain": self.preview_g_gain.get(),
                "b_gain": self.preview_b_gain.get(),
                "brightness": self.preview_brightness.get(),
                "contrast": self.preview_contrast.get(),
                "saturation": self.preview_saturation.get(),
            }
        except tk.TclError:
            print("  [RefreshPreview] Error getting preview parameters from UI.")
            return

        try:
            def _worker(data_copy, params_copy):
                pil_img, hist_data = self.preview_manager.update_preview(
                    data_copy,
                    params_copy,
                    stack_count=self.preview_img_count,
                    total_images=self.preview_total_imgs,
                    current_batch=self.preview_current_batch,
                    total_batches=self.preview_total_batches,
                )

                def _apply():
                    if recalculate_histogram and self.histogram_widget:
                        self.histogram_widget.update_histogram(hist_data)
                        try:
                            bp_ui = self.preview_black_point.get()
                            wp_ui = self.preview_white_point.get()
                            self.histogram_widget.set_range(bp_ui, wp_ui)
                        except tk.TclError:
                            pass

                self.root.after(0, _apply)

            threading.Thread(
                target=_worker,
                args=(self.current_preview_data.copy(), preview_params.copy()),
                daemon=True,
                name="PreviewWorker",
            ).start()
            return

        except Exception as e:
            print(f"  [RefreshPreview] ERREUR CRITIQUE pendant traitement aperçu/histogramme: {e}")
            traceback.print_exc(limit=2)
            if hasattr(self, "histogram_widget") and self.histogram_widget and recalculate_histogram:
                self.histogram_widget.plot_histogram(None)
        print("[DEBUG-GUI] refresh_preview V_RefreshHistoControl_1_SignatureFix: Exiting.")

    def update_image_info(self, header):
        if not header or not hasattr(self, "preview_manager"):
            return
        info_lines = []
        keys_labels = {
            "OBJECT": "Object",
            "DATE-OBS": "Date",
            "EXPTIME": "Exp (s)",
            "GAIN": "Gain",
            "OFFSET": "Offset",
            "CCD-TEMP": "Temp (°C)",
            "NIMAGES": "Images",
            "STACKTYP": "Method",
            "FILTER": "Filter",
            "BAYERPAT": "Bayer",
            "TOTEXP": "Total Exp (s)",
            "ALIGNED": "Aligned",
            "FAILALIGN": "Failed Align",
            "FAILSTACK": "Failed Stack",
            "SKIPPED": "Skipped",
            "WGHT_ON": "Weighting",
            "WGHT_MET": "W. Metrics",
        }
        for key, label_key in keys_labels.items():
            label = self.tr(label_key, default=label_key)
            value = header.get(key)
            if value is not None and str(value).strip() != "":
                s_value = str(value)
                if key == "DATE-OBS":
                    s_value = s_value.split("T")[0]
                elif key in ["EXPTIME", "CCD-TEMP", "TOTEXP"] and isinstance(value, (int, float)):
                    try:
                        s_value = f"{float(value):.1f}"
                    except ValueError:
                        pass
                elif key == "KAPPA" and isinstance(value, (int, float)):
                    try:
                        s_value = f"{float(value):.2f}"
                    except ValueError:
                        pass
                elif key == "WGHT_ON":
                    s_value = self.tr("weighting_enabled")
                else:
                    if value:
                        pass  # La condition est toujours vraie ici, l'indentation suivante était incorrecte
                    else:
                        s_value = self.tr("weighting_disabled")
                info_lines.append(f"{label}: {s_value}")
        info_text = "\n".join(info_lines) if info_lines else self.tr("No image info available")
        if hasattr(self.preview_manager, "update_info_text"):
            self.preview_manager.update_info_text(info_text)

    ################################################################################################################################################################

    def _try_show_first_input_image(self):
        """
        Tente de charger et d'afficher la première image FITS du dossier d'entrée
        pour un aperçu initial.
        MODIFIED V2_TupleFix: Gère correctement le retour (tuple) de load_and_validate_fits.
        """
        print("DEBUG GUI (_try_show_first_input_image V2_TupleFix): Tentative d'affichage de la première image.")
        input_folder = self.input_path.get()

        if not hasattr(self, "preview_manager") or not hasattr(self, "histogram_widget"):
            print("  WARN GUI (_try_show_first_input_image): PreviewManager ou HistogramWidget manquant.")
            return

        if not input_folder or not os.path.isdir(input_folder):
            print(
                f"  DEBUG GUI (_try_show_first_input_image): Dossier d'entrée non valide ou non défini ('{input_folder}'). Effacement aperçu."
            )
            if hasattr(self, "preview_manager") and self.preview_manager:
                self.preview_manager.clear_preview(self.tr("Input folder not found or not set"))
            if hasattr(self, "histogram_widget") and self.histogram_widget:
                self.histogram_widget.plot_histogram(None)
            return

        def _worker():
            try:
                files = sorted(
                    [f for f in os.listdir(input_folder) if f.lower().endswith((".fit", ".fits"))]
                )
                if not files:
                    def _no_files():
                        if hasattr(self, "preview_manager") and self.preview_manager:
                            self.preview_manager.clear_preview(self.tr("No FITS files in input folder"))
                        if hasattr(self, "histogram_widget") and self.histogram_widget:
                            self.histogram_widget.plot_histogram(None)
                    print(
                        f"  DEBUG GUI (_try_show_first_input_image): Aucun fichier FITS trouvé dans '{input_folder}'. Effacement aperçu."
                    )
                    self.root.after(0, _no_files)
                    return

                first_image_filename = files[0]
                first_image_path = os.path.join(input_folder, first_image_filename)
                self.root.after(
                    0,
                    lambda fn=first_image_filename: self.update_progress_gui(
                        f"{self.tr('Loading preview for', default='Loading preview')}: {fn}...",
                        None,
                    ),
                )
                print(f"  DEBUG GUI (_try_show_first_input_image): Chargement de '{first_image_path}'...")

                loaded_data_tuple = load_and_validate_fits(first_image_path)

                img_data_from_load = None
                header_from_load = None

                if loaded_data_tuple is not None and loaded_data_tuple[0] is not None:
                    img_data_from_load, header_from_load = loaded_data_tuple
                    print(
                        f"  DEBUG GUI (_try_show_first_input_image): load_and_validate_fits OK. Shape données: {img_data_from_load.shape}"
                    )
                else:
                    raise ValueError(
                        f"Échec chargement/validation de '{first_image_filename}' par load_and_validate_fits (retour None ou données None)."
                    )

                img_for_preview = img_data_from_load

                if img_for_preview.ndim == 2:
                    bayer_pattern_from_header = (
                        header_from_load.get("BAYERPAT", self.settings.bayer_pattern)
                        if header_from_load
                        else self.settings.bayer_pattern
                    )
                    valid_bayer_patterns = ["GRBG", "RGGB", "GBRG", "BGGR"]

                    if (
                        isinstance(bayer_pattern_from_header, str)
                        and bayer_pattern_from_header.upper() in valid_bayer_patterns
                    ):
                        print(
                            f"  DEBUG GUI (_try_show_first_input_image): Debayering aperçu initial (Pattern: {bayer_pattern_from_header.upper()})..."
                        )
                        try:
                            img_for_preview = debayer_image(
                                img_for_preview, bayer_pattern_from_header.upper()
                            )
                        except ValueError as debayer_err:
                            self.root.after(
                                0,
                                lambda msg=f"⚠️ {self.tr('Error during debayering')}: {debayer_err}. Affichage N&B.": self.update_progress_gui(
                                    msg,
                                    None,
                                ),
                            )
                            print(
                                f"    WARN GUI: Erreur Debayer aperçu initial: {debayer_err}. Affichage N&B."
                            )
                    else:
                        print(
                            f"  DEBUG GUI (_try_show_first_input_image): Pas de debayering pour aperçu (pas de pattern Bayer valide ou image supposée déjà couleur)."
                        )

                def _update_gui(img=img_for_preview, hdr=header_from_load, fn=first_image_filename):
                    self.current_preview_data = img.copy()
                    self.current_preview_hist_data = img.copy()
                    self.current_stack_header = hdr.copy() if hdr else fits.Header()
                    print(
                        f"  DEBUG GUI (_try_show_first_input_image): Données prêtes pour refresh_preview. Shape: {self.current_preview_data.shape}"
                    )
                    self.refresh_preview()
                    if self.current_stack_header:
                        self.update_image_info(self.current_stack_header)
                    self.update_progress_gui(
                        f"{self.tr('Preview loaded', default='Preview loaded')}: {fn}",
                        None,
                    )

                self.root.after(0, _update_gui)

            except FileNotFoundError:
                def _file_err():
                    if hasattr(self, "preview_manager") and self.preview_manager:
                        self.preview_manager.clear_preview(self.tr("Input folder not found or inaccessible"))
                    if hasattr(self, "histogram_widget") and self.histogram_widget:
                        self.histogram_widget.plot_histogram(None)
                print(
                    f"  ERREUR GUI (_try_show_first_input_image): FileNotFoundError pour '{input_folder}'."
                )
                self.root.after(0, _file_err)
            except ValueError as ve:
                def _val_err():
                    if hasattr(self, "preview_manager") and self.preview_manager:
                        self.preview_manager.clear_preview(
                            self.tr("Error loading preview (invalid format?)")
                        )
                    if hasattr(self, "histogram_widget") and self.histogram_widget:
                        self.histogram_widget.plot_histogram(None)
                    self.update_progress_gui(
                        f"⚠️ {self.tr('Error loading preview image')}: {ve}", None
                    )
                print(
                    f"  ERREUR GUI (_try_show_first_input_image): ValueError - {ve}"
                )
                self.root.after(0, _val_err)
            except Exception as e:
                def _unk_err():
                    if hasattr(self, "preview_manager") and self.preview_manager:
                        self.preview_manager.clear_preview(self.tr("Error loading preview"))
                    if hasattr(self, "histogram_widget") and self.histogram_widget:
                        self.histogram_widget.plot_histogram(None)
                    self.update_progress_gui(
                        f"⚠️ {self.tr('Error loading preview image')}: {e}", None
                    )
                print(
                    f"  ERREUR GUI (_try_show_first_input_image): Exception inattendue - {type(e).__name__}: {e}"
                )
                traceback.print_exc(limit=2)
                self.root.after(0, _unk_err)

        threading.Thread(target=_worker, daemon=True, name="PreviewLoader").start()

    ################################################################################################################################################################

    def apply_auto_white_balance(self):
        """Compute auto white balance on a worker thread."""

        if (
            hasattr(self, "_final_stretch_set_by_processing_finished")
            and self._final_stretch_set_by_processing_finished
        ):
            self.logger.warning(
                "APPLY_AUTO_WHITE_BALANCE (Main_Window) appelé MAIS VERROUILLÉ par _final_stretch_set_by_processing_finished. Ignoré."
            )
            return

        def _worker(data):
            if not _tools_available:
                self.root.after(0, lambda: messagebox.showerror(self.tr("error"), "Stretch/Color tools not available."))
                return

            if data is None or data.ndim != 3:
                self.root.after(
                    0,
                    lambda: messagebox.showwarning(
                        self.tr("warning"), self.tr("Auto WB requires a color image preview.")
                    ),
                )
                return

            try:
                r_gain, g_gain, b_gain = calculate_auto_wb(data)
            except Exception as e:
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        self.tr("error"), f"{self.tr('Error during Auto WB')}: {e}"
                    ),
                )
                self.logger.error(f"apply_auto_white_balance: {e}")
                return

            def _apply():
                self.preview_r_gain.set(round(r_gain, 3))
                self.preview_g_gain.set(round(g_gain, 3))
                self.preview_b_gain.set(round(b_gain, 3))
                self.update_progress_gui(
                    f"Auto WB appliqué (Aperçu): R={r_gain:.2f} G={g_gain:.2f} B={b_gain:.2f}",
                    None,
                )
                self.refresh_preview(recalculate_histogram=True)

            self.root.after(0, _apply)

        data = self.current_preview_data.copy() if self.current_preview_data is not None else None
        threading.Thread(target=_worker, args=(data,), daemon=True, name="AutoWBWorker").start()

    def reset_white_balance(self):
        self.preview_r_gain.set(1.0)
        self.preview_g_gain.set(1.0)
        self.preview_b_gain.set(1.0)
        self.refresh_preview()

    def reset_brightness_contrast_saturation(self):
        self.preview_brightness.set(1.0)
        self.preview_contrast.set(1.0)
        self.preview_saturation.set(1.0)
        self.refresh_preview()

    def reset_stretch(self):
        default_method = "Asinh"
        default_bp = 0.01
        default_wp = 0.99
        default_gamma = 1.0
        self.preview_stretch_method.set(default_method)
        self.preview_black_point.set(default_bp)
        self.preview_white_point.set(default_wp)
        self.preview_gamma.set(default_gamma)
        if hasattr(self, "histogram_widget"):
            self.histogram_widget.set_range(default_bp, default_wp)
            self.histogram_widget.reset_zoom()
        self.refresh_preview()

    # --- NOUVELLE MÉTHODE pour gérer la requête d'ajout ---
    def handle_add_folder_request(self, folder_path):
        """
        Gère une requête d'ajout de dossier, en l'ajoutant soit à la liste
        pré-démarrage, soit en appelant le backend si le traitement est actif.
        """
        abs_folder = os.path.abspath(folder_path)

        if self.processing and hasattr(self, "queued_stacker") and self.queued_stacker.is_running():
            # Traitement actif : appeler le backend
            add_success = self.queued_stacker.add_folder(abs_folder)
            if not add_success:
                messagebox.showwarning(
                    self.tr("warning"),
                    self.tr(
                        "Folder not added (already present, invalid path, or error?)",
                        default="Folder not added (already present, invalid path, or error?)",
                    ),
                )
            # La mise à jour de l'affichage se fera via callback "folder_count_update" du backend
        else:
            # Traitement non actif : ajouter à la liste pré-démarrage
            if abs_folder not in self.additional_folders_to_process:
                self.additional_folders_to_process.append(abs_folder)
                self.update_progress_gui(
                    f"ⓘ Dossier ajouté pour prochain traitement: {os.path.basename(abs_folder)}",
                    None,
                )
                self.update_additional_folders_display()  # Mettre à jour l'affichage UI
            else:
                messagebox.showinfo(
                    self.tr("info"),
                    self.tr(
                        "Folder already added",
                        default="Folder already added to the list.",
                    ),
                )

    def _track_processing_progress(self):
        """Monitors the QueuedStacker worker thread and updates GUI stats."""
        # print("DEBUG: GUI Progress Tracker Thread Started.") # Keep disabled unless debugging

        while self.processing and hasattr(self, "queued_stacker"):
            try:
                # Check if the worker thread is still active
                if not self.queued_stacker.is_running():
                    worker_thread = getattr(self.queued_stacker, "processing_thread", None)
                    if worker_thread and worker_thread.is_alive():
                        worker_thread.join(timeout=0.5)
                    if hasattr(self, 'gui_event_queue'):
                        self.gui_event_queue.put(self._processing_finished)
                    else:
                        self.root.after(0, self._processing_finished)
                    break

                q_stacker = self.queued_stacker
                processed = q_stacker.processed_files_count
                aligned = q_stacker.aligned_files_count
                total_queued = q_stacker.files_in_queue

                if self.global_start_time and processed > 0:
                    elapsed = time.monotonic() - self.global_start_time
                    current_tpi = elapsed / processed
                    if self.time_per_image == 0:
                        self.time_per_image = current_tpi
                    else:
                        alpha = 0.1
                        self.time_per_image = (
                            alpha * current_tpi + (1 - alpha) * self.time_per_image
                        )
                    remaining_estimated = max(0, total_queued - processed)
                    if self.time_per_image > 1e-6 and remaining_estimated > 0:
                        eta_seconds = remaining_estimated * self.time_per_image
                        h, rem = divmod(int(eta_seconds), 3600)
                        m, s = divmod(rem, 60)
                        eta_str = f"{h:02}:{m:02}:{s:02}"
                    elif remaining_estimated == 0 and total_queued > 0:
                        eta_str = "00:00:00"
                    else:
                        eta_str = self.tr("eta_calculating", default="Calculating...")
                else:
                    eta_str = self.tr("eta_calculating", default="Calculating...")

                default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
                remaining = max(0, total_queued - processed)
                total = total_queued

                def _gui_update(
                    es=eta_str,
                    al=aligned,
                    rem=remaining,
                    tot=total,
                    fmt=default_aligned_fmt,
                ):
                    try:
                        self.remaining_time_var.set(es)
                        self.aligned_files_var.set(fmt.format(count=al))
                        self.remaining_files_var.set(f"{rem}/{tot}")
                    except tk.TclError:
                        pass

                if hasattr(self, 'gui_event_queue'):
                    self.gui_event_queue.put(_gui_update)
                else:
                    self.root.after(0, _gui_update)

                time.sleep(0.5)

            except Exception as e:
                print(f"Error in GUI progress tracker thread loop: {e}")
                traceback.print_exc(limit=2)
                if hasattr(self, 'gui_event_queue'):
                    self.gui_event_queue.put(self._processing_finished)
                else:
                    try:
                        self.root.after(0, self._processing_finished)
                    except tk.TclError:
                        pass
                break

        # print("DEBUG: GUI Progress Tracker Thread Exiting.") # Keep disabled

    def update_remaining_files(self):
        """Met à jour l'affichage des fichiers restants / total ajouté."""
        try:
            qs = getattr(self, "queued_stacker", None)

            if (
                qs is not None
                and isinstance(getattr(qs, "files_in_queue", None), int)
                and isinstance(getattr(qs, "processed_files_count", None), int)
            ):

                remaining = max(0, qs.files_in_queue - qs.processed_files_count)
                total = qs.files_in_queue
                self.remaining_files_var.set(f"{remaining}/{total}")
            else:
                # Données indisponibles : on affiche un placeholder neutre
                self.remaining_files_var.set("--")

        except Exception as e:
            # Toute erreur reste dans la console, pas dans l'UI
            self.remaining_files_var.set("--")
            print(f"[Remaining-label] {type(e).__name__}: {e}")

    # --- MODIFIER CETTE MÉTHODE ---

    def update_additional_folders_display(self):
        """Met à jour l'affichage du nombre de dossiers supplémentaires."""
        count = 0
        qs = getattr(self, "queued_stacker", None)
        if self.processing and qs and qs.is_running():
            lock = getattr(qs, "folders_lock", None)
            if lock and lock.acquire(blocking=False):
                try:
                    count = len(qs.additional_folders)
                finally:
                    lock.release()
            else:
                return
        else:
            count = len(self.additional_folders_to_process)

        try:
            if count == 0:
                self.additional_folders_var.set(self.tr("no_additional_folders"))
            elif count == 1:
                self.additional_folders_var.set(self.tr("1 additional folder"))
            else:
                self.additional_folders_var.set(
                    self.tr("{count} additional folders", default="{count} add. folders").format(count=count)
                )
        except tk.TclError:
            pass

    # --- FIN MÉTHODE MODIFIÉE ---

    def _run_boring_stack_process(self, cmd, csv_path, out_dir):
        """Execute ``boring_stack.py`` without blocking the GUI."""

        # Reset any previous process handle
        self.boring_proc = None

        if hasattr(self, "output_path"):
            # Ensure the GUI knows the output directory so the summary dialog can
            # immediately enable the "Open Output" button once processing
            # finishes.  Always update the variable so it reflects the actual
            # path used by boring_stack.py.
            self.output_path.set(out_dir)

        def _worker():
            total_files = 0
            try:
                total_files = len(read_paths(csv_path))
            except Exception:
                total_files = 0

            def _setup_start_gui():
                self.processing = True
                if hasattr(self, "start_button") and self.start_button.winfo_exists():
                    self.start_button.config(state=tk.DISABLED)
                if hasattr(self, "stop_button") and self.stop_button.winfo_exists():
                    self.stop_button.config(state=tk.NORMAL)
                self._set_parameter_widgets_state(tk.DISABLED)
                if hasattr(self, "progress_manager") and self.progress_manager:
                    self.progress_manager.reset()
                    self.progress_manager.start_timer()

            if hasattr(self, 'gui_event_queue'):
                self.gui_event_queue.put(_setup_start_gui)
            else:
                self.root.after(0, _setup_start_gui)

            def _gui_update(progress, eta, processed):
                try:
                    if hasattr(self, "progress_manager") and self.progress_manager:
                        self.progress_manager.update_progress(f"{progress:.1f}%", progress)
                    if hasattr(self.progress_manager, "set_remaining"):
                        self.progress_manager.set_remaining(eta)

                    # Update ETA-related counters for Threaded Boring Stack mode
                    self.preview_img_count = processed
                    self.preview_total_imgs = total_files

                    default_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
                    self.aligned_files_var.set(default_fmt.format(count=processed))
                    remaining = max(0, total_files - processed)
                    self.remaining_files_var.set(f"{remaining}/{total_files}")

                    # Explicitly refresh progress GUI so ETA updates correctly
                    self.update_progress_gui(f"{progress:.1f}% completed", progress)
                except tk.TclError:
                    pass

            def _finish(retcode, output_lines):
                try:
                    if hasattr(self, "progress_manager") and self.progress_manager:
                        self.progress_manager.stop_timer()

                    log_text = "\n".join(output_lines)
                    logger.debug("boring_stack.py output:\n%s", log_text)

                    if retcode == 0:
                        final_path = os.path.join(out_dir, "final.fits")
                        if os.path.exists(final_path):
                            self.update_progress_gui(
                                self.tr("stacking_finished", default="Stacking finished"),
                                100,
                            )
                    else:
                        tail = "\n".join(output_lines[-10:])
                        err_msg = (
                            f"boring_stack.py failed (code {retcode}).\nLast lines:\n{tail}"
                        )
                        print(err_msg)
                        messagebox.showerror("Stack error", err_msg)
                finally:
                    self.processing = False
                    if hasattr(self, "start_button") and self.start_button.winfo_exists():
                        self.start_button.config(state=tk.NORMAL)
                    if hasattr(self, "stop_button") and self.stop_button.winfo_exists():
                        self.stop_button.config(state=tk.DISABLED)
                    self._set_parameter_widgets_state(tk.NORMAL)
                    self.boring_proc = None

                    # --- Nouveau: afficher le resume apres un boring stack ---
                    elapsed = time.monotonic() - start_time
                    if hasattr(self, "tr") and hasattr(self, "localization"):
                        tr = self.tr
                    else:
                        tr = lambda k, default=None: default or k
                    summary_title = tr("processing_report_title", default="Processing Summary")
                    status_text = (
                        tr("stacking_finished", default="Stacking finished")
                        if retcode == 0
                        else tr("error", default="Error")
                    )
                    if (
                        hasattr(self, "_format_duration")
                        and hasattr(self, "tr")
                        and hasattr(self, "localization")
                    ):
                        duration_str = self._format_duration(elapsed)
                    else:
                        duration_str = f"{int(round(elapsed))} s"
                    summary_lines = [
                        f"{tr('Status', default='Status')}: {status_text}",
                        f"{tr('Total Processing Time', default='Total Processing Time')}: {duration_str}",
                        f"{tr('Files Attempted', default='Files Attempted')}: {total_files}",
                    ]
                    final_path = os.path.join(out_dir, "final.fits")
                    images_stacked = total_files
                    total_exposure = 0.0
                    if os.path.exists(final_path):
                        try:
                            hdr = fits.getheader(final_path)
                            images_stacked = int(hdr.get("NIMAGES", images_stacked))
                            total_exposure = float(hdr.get("TOTEXP", 0.0))
                        except Exception:
                            pass
                        summary_lines.append(
                            f"{tr('Final Stack File', default='Final Stack File')}:\n  {final_path}"
                        )
                    elif retcode == 0:
                        summary_lines.append(
                            f"{tr('Final Stack File', default='Final Stack File')}: {final_path} ({tr('Not Found!', default='Not Found!')})"
                        )
                    summary_lines.append(
                        f"{tr('Images in Final Stack', default='Images in Final Stack')}: {images_stacked}"
                    )
                    if (
                        hasattr(self, "_format_duration")
                        and hasattr(self, "tr")
                        and hasattr(self, "localization")
                    ):
                        exposure_str = self._format_duration(total_exposure)
                    else:
                        exposure_str = f"{int(round(total_exposure))} s"
                    summary_lines.append(
                        f"{tr('Total Exposure (Final Stack)', default='Total Exposure (Final Stack)')}: {exposure_str}"
                    )
                    full_summary = "\n".join(summary_lines)
                    if hasattr(self, "output_path"):
                        # Ensure the variable points to the folder used during
                        # processing so the button opens the correct location.
                        self.output_path.set(out_dir)
                        try:
                            can_open_output = bool(
                                out_dir and os.path.isdir(out_dir) and retcode == 0
                            )
                        except Exception:
                            can_open_output = False
                    else:
                        can_open_output = bool(out_dir and os.path.isdir(out_dir) and retcode == 0)

                    if (
                        hasattr(self, "_show_summary_dialog")
                        and getattr(self, "root", None) is not None
                        and hasattr(self.root, "tk")
                    ):
                        self._show_summary_dialog(summary_title, full_summary, can_open_output)
                    if hasattr(self, "open_output_button") and self.open_output_button.winfo_exists():
                        self.open_output_button.config(
                            state=tk.NORMAL if can_open_output else tk.DISABLED
                        )

            start_time = time.monotonic()
            output_lines = []
            log_path = os.path.join(out_dir, "boring_stack.log")
            aligned_files = 0
            time_per_image = 0.0
            try:
                log_file = open(log_path, "w", encoding="utf-8")
                self.boring_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )

                for line in self.boring_proc.stdout:
                    if not line:
                        continue
                    text = line.strip()
                    log_file.write(text + "\n")
                    log_file.flush()
                    output_lines.append(text)

                    aligned_match = re.search(r"Aligned:\s*(\d+)", text)
                    pct_match = re.search(r"(?:\[(\d+(?:\.\d+)?)%\]|(\d+(?:\.\d+)?)%)", text)

                    if aligned_match:
                        try:
                            new_aligned = int(aligned_match.group(1))
                        except ValueError:
                            new_aligned = aligned_files
                        if new_aligned > aligned_files:
                            aligned_files = new_aligned
                        progress_pct = aligned_files / total_files * 100 if total_files else 0

                        elapsed = time.monotonic() - start_time
                        if aligned_files > 0:
                            current_tpi = elapsed / aligned_files
                            if time_per_image == 0:
                                time_per_image = current_tpi
                            else:
                                alpha = 0.1
                                time_per_image = (
                                    alpha * current_tpi + (1 - alpha) * time_per_image
                                )
                            remaining_files = max(0, total_files - aligned_files)
                            eta_sec = remaining_files * time_per_image
                            h, r = divmod(int(eta_sec), 3600)
                            m, s = divmod(r, 60)
                            eta = f"{h:02}:{m:02}:{s:02}"
                        else:
                            eta = self.tr("eta_calculating", default="Calculating...")

                        if hasattr(self, 'gui_event_queue'):
                            self.gui_event_queue.put(
                                lambda p=progress_pct, e=eta, pr=aligned_files: _gui_update(p, e, pr)
                            )
                        else:
                            self.root.after(0, _gui_update, progress_pct, eta, aligned_files)
                    elif pct_match:
                        try:
                            progress_pct = float(next(filter(None, pct_match.groups())))
                        except (ValueError, StopIteration):
                            continue

                        elapsed = time.monotonic() - start_time
                        if aligned_files > 0:
                            current_tpi = elapsed / aligned_files
                            if time_per_image == 0:
                                time_per_image = current_tpi
                            else:
                                alpha = 0.1
                                time_per_image = (
                                    alpha * current_tpi + (1 - alpha) * time_per_image
                                )
                            remaining_files = max(0, total_files - aligned_files)
                            eta_sec = remaining_files * time_per_image
                            h, r = divmod(int(eta_sec), 3600)
                            m, s = divmod(r, 60)
                            eta = f"{h:02}:{m:02}:{s:02}"
                        else:
                            eta = self.tr("eta_calculating", default="Calculating...")

                        if hasattr(self, 'gui_event_queue'):
                            self.gui_event_queue.put(
                                lambda p=progress_pct, e=eta, pr=aligned_files: _gui_update(p, e, pr)
                            )
                        else:
                            self.root.after(0, _gui_update, progress_pct, eta, aligned_files)
                    else:
                        if hasattr(self, 'gui_event_queue'):
                            self.gui_event_queue.put(
                                lambda t=text: self.update_progress_gui(t, None)
                            )
                        else:
                            self.root.after(0, self.update_progress_gui, text, None)

                retcode = self.boring_proc.wait()
                log_file.close()
            except Exception as e:
                retcode = -1
                try:
                    log_file.write(f"Error running boring_stack: {e}\n")
                    log_file.close()
                except Exception:
                    pass
                if hasattr(self, 'gui_event_queue'):
                    self.gui_event_queue.put(
                        lambda err=e: self.update_progress_gui(f"Error running boring_stack: {err}", None)
                    )
                else:
                    self.root.after(0, self.update_progress_gui, f"Error running boring_stack: {e}", None)
            finally:
                if hasattr(self, 'gui_event_queue'):
                    self.gui_event_queue.put(lambda r=retcode, o=output_lines: _finish(r, o))
                else:
                    self.root.after(0, _finish, retcode, output_lines)

        threading.Thread(target=_worker, daemon=True, name="BoringStackWorker").start()


    def stop_processing(self):
        if self.processing and hasattr(self, "queued_stacker") and self.queued_stacker.is_running():
            self.update_progress_gui(self.tr("stacking_stopping"), None)
            self.queued_stacker.stop()
            if hasattr(self, "stop_button"):
                self.stop_button.config(state=tk.DISABLED)
        elif self.processing and getattr(self, "boring_proc", None):
            self.update_progress_gui(self.tr("stacking_stopping"), None)
            try:
                self.boring_proc.terminate()
            except Exception:
                pass
            if hasattr(self, "stop_button"):
                self.stop_button.config(state=tk.DISABLED)
        elif self.processing:
            self.update_progress_gui("Tentative d'arrêt, mais worker inactif ou déjà arrêté.", None)
            self._processing_finished()

    def _format_duration(self, seconds):
        try:
            secs = int(round(float(seconds)))
            if secs < 0:
                return "N/A"
            if secs < 60:
                return f"{secs} {self.tr('report_seconds', 's')}"
            elif secs < 3600:
                m, s = divmod(secs, 60)
                return f"{m} {self.tr('report_minutes', 'min')} {s} {self.tr('report_seconds', 's')}"
            else:
                h, rem = divmod(secs, 3600)
                m, s = divmod(rem, 60)
                return f"{h} {self.tr('report_hours', 'h')} {m} {self.tr('report_minutes', 'min')} {s} {self.tr('report_seconds', 's')}"
        except (ValueError, TypeError):
            return "N/A"

    def _set_parameter_widgets_state(self, state):
        """Enable/disable control widgets."""
        processing_widgets = []
        if hasattr(self, "input_entry"):
            processing_widgets.append(self.input_entry)
        if hasattr(self, "browse_input_button"):
            processing_widgets.append(self.browse_input_button)
        if hasattr(self, "output_entry"):
            processing_widgets.append(self.output_entry)
        if hasattr(self, "browse_output_button"):
            processing_widgets.append(self.browse_output_button)
        if hasattr(self, "ref_entry"):
            processing_widgets.append(self.ref_entry)
        if hasattr(self, "browse_ref_button"):
            processing_widgets.append(self.browse_ref_button)
        if hasattr(self, "stack_norm_combo"):
            processing_widgets.append(self.stack_norm_combo)
        if hasattr(self, "stack_weight_combo"):
            processing_widgets.append(self.stack_weight_combo)
        if hasattr(self, "kappa_low_spinbox"):
            processing_widgets.append(self.kappa_low_spinbox)
        if hasattr(self, "kappa_high_spinbox"):
            processing_widgets.append(self.kappa_high_spinbox)
        if hasattr(self, "winsor_limits_entry"):
            processing_widgets.append(self.winsor_limits_entry)
        if hasattr(self, "stack_final_combo"):
            processing_widgets.append(self.stack_final_combo)
        if hasattr(self, "batch_spinbox"):
            processing_widgets.append(self.batch_spinbox)
        if hasattr(self, "hot_pixels_check"):
            processing_widgets.append(self.hot_pixels_check)
        if hasattr(self, "hp_thresh_spinbox"):
            processing_widgets.append(self.hp_thresh_spinbox)
        if hasattr(self, "hp_neigh_spinbox"):
            processing_widgets.append(self.hp_neigh_spinbox)
        if hasattr(self, "cleanup_temp_check"):
            processing_widgets.append(self.cleanup_temp_check)
        # --- AJOUT DE CHROMA CORRECTION ---
        if hasattr(self, "chroma_correction_check"):
            processing_widgets.append(self.chroma_correction_check)
        # ---  ---
        if hasattr(self, "language_combo"):
            processing_widgets.append(self.language_combo)
        if hasattr(self, "use_weighting_check"):
            processing_widgets.append(self.use_weighting_check)
        if hasattr(self, "weight_snr_check"):
            processing_widgets.append(self.weight_snr_check)
        if hasattr(self, "weight_stars_check"):
            processing_widgets.append(self.weight_stars_check)
        if hasattr(self, "snr_exp_spinbox"):
            processing_widgets.append(self.snr_exp_spinbox)
        if hasattr(self, "stars_exp_spinbox"):
            processing_widgets.append(self.stars_exp_spinbox)
        if hasattr(self, "min_w_spinbox"):
            processing_widgets.append(self.min_w_spinbox)
        if hasattr(self, "drizzle_check"):
            processing_widgets.append(self.drizzle_check)
        if hasattr(self, "drizzle_scale_label"):
            processing_widgets.append(self.drizzle_scale_label)
        if hasattr(self, "drizzle_radio_2x"):
            processing_widgets.append(self.drizzle_radio_2x)  # Si Radiobuttons
        if hasattr(self, "drizzle_radio_3x"):
            processing_widgets.append(self.drizzle_radio_3x)  # Si Radiobuttons
        if hasattr(self, "drizzle_radio_4x"):
            processing_widgets.append(self.drizzle_radio_4x)  # Si Radiobuttons
        # if hasattr(self, 'drizzle_scale_combo'): processing_widgets.append(self.drizzle_scale_combo) # Si Combobox

        preview_widgets = []
        if hasattr(self, "wb_r_ctrls"):
            preview_widgets.extend([self.wb_r_ctrls["slider"], self.wb_r_ctrls["spinbox"]])
        if hasattr(self, "wb_g_ctrls"):
            preview_widgets.extend([self.wb_g_ctrls["slider"], self.wb_g_ctrls["spinbox"]])
        if hasattr(self, "wb_b_ctrls"):
            preview_widgets.extend([self.wb_b_ctrls["slider"], self.wb_b_ctrls["spinbox"]])
        if hasattr(self, "auto_wb_button"):
            preview_widgets.append(self.auto_wb_button)
        if hasattr(self, "reset_wb_button"):
            preview_widgets.append(self.reset_wb_button)
        if hasattr(self, "stretch_combo"):
            preview_widgets.append(self.stretch_combo)
        if hasattr(self, "stretch_bp_ctrls"):
            preview_widgets.extend([self.stretch_bp_ctrls["slider"], self.stretch_bp_ctrls["spinbox"]])
        if hasattr(self, "stretch_wp_ctrls"):
            preview_widgets.extend([self.stretch_wp_ctrls["slider"], self.stretch_wp_ctrls["spinbox"]])
        if hasattr(self, "stretch_gamma_ctrls"):
            preview_widgets.extend(
                [
                    self.stretch_gamma_ctrls["slider"],
                    self.stretch_gamma_ctrls["spinbox"],
                ]
            )
        if hasattr(self, "auto_stretch_button"):
            preview_widgets.append(self.auto_stretch_button)
        if hasattr(self, "reset_stretch_button"):
            preview_widgets.append(self.reset_stretch_button)
        if hasattr(self, "brightness_ctrls"):
            preview_widgets.extend([self.brightness_ctrls["slider"], self.brightness_ctrls["spinbox"]])
        if hasattr(self, "contrast_ctrls"):
            preview_widgets.extend([self.contrast_ctrls["slider"], self.contrast_ctrls["spinbox"]])
        if hasattr(self, "saturation_ctrls"):
            preview_widgets.extend([self.saturation_ctrls["slider"], self.saturation_ctrls["spinbox"]])
        if hasattr(self, "reset_bcs_button"):
            preview_widgets.append(self.reset_bcs_button)
        if hasattr(self, "hist_reset_btn"):
            preview_widgets.append(self.hist_reset_btn)

        widgets_to_set = []
        if state == tk.NORMAL:
            print("DEBUG (GUI _set_parameter_widgets_state): Activation de tous les widgets...")
            # Activer TOUS les widgets (traitement + preview) quand le traitement finit
            widgets_to_set = processing_widgets + preview_widgets
            # S'assurer que les options Drizzle sont dans le bon état initial
            self._update_drizzle_options_state()  # <-- Appel ajouté ici
            # ... (reste de la logique pour state == tk.NORMAL) ...
            if hasattr(self, "add_files_button"):
                try:
                    self.add_files_button.config(state=tk.NORMAL)
                except tk.TclError:
                    pass

        else:  # tk.DISABLED (Pendant le traitement)
            # Désactiver les paramètres de traitement (y compris Drizzle)
            widgets_to_set = processing_widgets
            # Les widgets de preview restent actifs
            for widget in preview_widgets:
                # ... (logique existante) ...
                pass
            # Le bouton Ajouter Dossier est désactivé si la reprojection est active
            if hasattr(self, "add_files_button"):
                btn_state = tk.NORMAL
                if getattr(self.settings, "reproject_between_batches", False):
                    btn_state = tk.DISABLED
                try:
                    self.add_files_button.config(state=btn_state)
                except tk.TclError:
                    pass

        # Appliquer l'état aux widgets sélectionnés
        for widget in widgets_to_set:
            if widget and hasattr(widget, "winfo_exists") and widget.winfo_exists():
                try:
                    widget.config(state=state)
                except tk.TclError:
                    pass

        # Exceptionnellement, si on désactive (pendant traitement), on s'assure que
        # les options internes (scale drizzle, options poids) sont aussi désactivées,
        # même si la case principale était déjà décochée.
        if state == tk.DISABLED:
            self._update_drizzle_options_state()

    def _debounce_resize(self, event=None):
        if self._after_id_resize:
            try:
                self.root.after_cancel(self._after_id_resize)
            except tk.TclError:
                pass
        try:
            self._after_id_resize = self.root.after(300, self._refresh_preview_on_resize)
        except tk.TclError:
            pass

    def _poll_gui_events(self):
        """Process queued GUI events from worker threads."""
        if hasattr(self, "gui_event_queue"):
            processed = 0
            try:
                while processed < 100:
                    cb = self.gui_event_queue.get_nowait()
                    try:
                        if callable(cb):
                            cb()
                    finally:
                        self.gui_event_queue.task_done()
                    processed += 1
            except Empty:
                pass
        try:
            self.root.after(50, self._poll_gui_events)
        except tk.TclError:
            pass

    def _refresh_final_preview_and_histo(self):
        """
        Méthode dédiée pour le refresh final pour s'assurer de l'ordre et que
        l'histogramme est mis à jour avec les bonnes données AVANT que set_range et
        l'aperçu PreviewManager ne soient mis à jour.
        Version: V_FinalRefreshOrder_1
        """
        print("DEBUG GUI (_refresh_final_preview_and_histo V_FinalRefreshOrder_1): Appel.")

        # S'assurer que les données temporaires pour l'histogramme existent
        current_data_for_histo = getattr(self, "_temp_data_for_final_histo", None)
        if current_data_for_histo is None:
            print(
                "  -> _temp_data_for_final_histo non trouvé. Tentative avec self.current_preview_data pour l'histogramme."
            )
            # En fallback, si _temp_data_for_final_histo n'est pas là, on utilise current_preview_data
            # qui a été mis à jour avec cosmetic_01_data_for_preview_from_backend.
            # Pour le mode classique, c'est [0,1]. Pour Drizzle/Mosaïque, c'est aussi [0,1] cosmétique.
            current_data_for_histo = self.current_preview_data

        if current_data_for_histo is None:
            print("  -> Aucune donnée disponible pour l'histogramme final. Annulation refresh.")
            return

        try:
            # 1. Mettre à jour l'histogramme avec les données d'analyse (ceci appelle plot_histogram)
            #    plot_histogram va configurer les axes X et Y, et dessiner les barres.
            if hasattr(self, "histogram_widget") and self.histogram_widget:
                print(
                    f"  -> Appel histogram_widget.update_histogram avec données (Shape: {current_data_for_histo.shape})"
                )
                self.histogram_widget.update_histogram(current_data_for_histo)

            # 2. Lire les valeurs BP/WP des sliders (qui ont été settées à 0.01/0.95 dans _processing_finished)
            bp_ui = self.preview_black_point.get()
            wp_ui = self.preview_white_point.get()

            # 3. Mettre à jour les lignes BP/WP de l'histogramme avec ces valeurs UI.
            #    set_range convertira ces BP/WP UI (0-1) en l'échelle des données de l'histogramme
            #    et DESSINERA les lignes.
            if hasattr(self, "histogram_widget") and self.histogram_widget:
                print(f"  -> Appel histogram_widget.set_range avec BP_UI={bp_ui:.4f}, WP_UI={wp_ui:.4f}")
                self.histogram_widget.set_range(bp_ui, wp_ui)
        except tk.TclError as e:
            print(f"  Erreur TclError pendant la mise à jour de l'histogramme: {e}")
            # Continuer pour essayer de rafraîchir l'aperçu
        except Exception as e_histo:
            print(f"  Erreur inattendue pendant la mise à jour de l'histogramme: {e_histo}")
            traceback.print_exc(limit=1)

        # 4. Mettre à jour l'aperçu visuel (PreviewManager)
        #    Il lira les paramètres UI (y compris BP/WP à 0.01/0.95) pour stretcher self.current_preview_data.
        #    On ne recalcule PAS l'histogramme ici, car il vient d'être fait.
        print("  -> Appel self.refresh_preview (recalculate_histogram=False)")
        self.refresh_preview(recalculate_histogram=False)

        if hasattr(self, "_temp_data_for_final_histo"):
            self._temp_data_for_final_histo = None  # Nettoyer la donnée temporaire

        print("DEBUG GUI (_refresh_final_preview_and_histo V_FinalRefreshOrder_1): Fin.")

    def _refresh_preview_on_resize(self):
        if hasattr(self, "preview_manager"):
            self.preview_manager.trigger_redraw()
        if hasattr(self, "histogram_widget") and self.histogram_widget.winfo_exists():
            try:
                self.histogram_widget.canvas.draw_idle()
            except tk.TclError:
                pass

    def _on_closing(self):
        if self.processing:
            if messagebox.askokcancel(self.tr("quit"), self.tr("quit_while_processing")):
                print("Arrêt demandé via fermeture fenêtre...")
                self.stop_processing()
                if self.thread and self.thread.is_alive():
                    self.thread.join(timeout=1.5)
                if hasattr(self, "queued_stacker") and self.queued_stacker.is_running():
                    print("Warning: Worker thread did not exit cleanly.")
                self._save_settings_and_destroy()
            else:
                return
        else:
            self._save_settings_and_destroy()

    def run_zemosaic(self):
        """Launch the ZeMosaic application in a separate process."""
        self.logger.info("run_zemosaic called. Launching run_zemosaic.py...")
        try:
            # Ensure imports inside run_zemosaic work even if the GUI was
            # started from another directory by using the project root as cwd
            project_root = Path(__file__).resolve().parents[2]
            env = os.environ.copy()
            if self.settings.use_third_party_solver:
                if getattr(self.settings, "astap_path", ""):
                    env["ZEMOSAIC_ASTAP_PATH"] = str(self.settings.astap_path)
                if getattr(self.settings, "astap_data_dir", ""):
                    env["ZEMOSAIC_ASTAP_DATA_DIR"] = str(self.settings.astap_data_dir)
                if getattr(self.settings, "local_ansvr_path", ""):
                    env["ZEMOSAIC_LOCAL_ANSVR_PATH"] = str(self.settings.local_ansvr_path)
                if getattr(self.settings, "astrometry_api_key", ""):
                    env["ZEMOSAIC_ASTROMETRY_API_KEY"] = str(self.settings.astrometry_api_key)
                if getattr(self.settings, "astrometry_solve_field_dir", ""):
                    env["ZEMOSAIC_ASTROMETRY_DIR"] = str(self.settings.astrometry_solve_field_dir)
                if getattr(self.settings, "local_solver_preference", ""):
                    env["ZEMOSAIC_LOCAL_SOLVER_PREFERENCE"] = str(self.settings.local_solver_preference)
            if self.settings.use_third_party_solver:
                try:
                    radius_val = float(getattr(self.settings, "astap_search_radius", 0))
                    env["ZEMOSAIC_ASTAP_SEARCH_RADIUS"] = str(radius_val)
                except Exception:
                    pass

            # Directly execute the run_zemosaic.py script located in the
            # ``zemosaic`` directory of the project. Using the explicit file
            # path avoids issues with module lookups when the application is
            # bundled or executed from a different working directory.
            run_zemosaic_path = project_root / "zemosaic" / "run_zemosaic.py"
            subprocess.Popen(
                [sys.executable, str(run_zemosaic_path)],
                cwd=str(project_root),
                env=env,
            )
            self.logger.info("run_zemosaic.py launched successfully")
        except Exception as e:
            self.logger.error(f"Failed to launch run_zemosaic.py: {e}")
            messagebox.showerror(
                self.tr("error", default="Error"),
                self.tr(
                    "mosaic_window_create_error",
                    default="Could not open Mosaic settings window.",
                )
                + f"\n{e}",
                parent=self.root,
            )

    def _open_mosaic_settings_window(self):
        """Open the Mosaic settings window inside the main GUI."""
        self.logger.info("_open_mosaic_settings_window called.")
        if (
            hasattr(self, "_mosaic_settings_window_instance")
            and self._mosaic_settings_window_instance
            and self._mosaic_settings_window_instance.winfo_exists()
        ):
            try:
                self._mosaic_settings_window_instance.lift()
                self._mosaic_settings_window_instance.focus_force()
            except tk.TclError:
                self._mosaic_settings_window_instance = None
            else:
                return

        # Créer et afficher la nouvelle fenêtre modale
        try:
            print("DEBUG (GUI): Création de l'instance MosaicSettingsWindow...")
            # Passer 'self' (l'instance SeestarStackerGUI) à la fenêtre enfant
            # pour qu'elle puisse mettre à jour le flag mosaic_mode_active etc.
            mosaic_window = MosaicSettingsWindow(parent_gui=self)
            self._mosaic_settings_window_instance = mosaic_window  # Stocker référence (optionnel)
            print("DEBUG (GUI): Instance MosaicSettingsWindow créée.")
            # La fenêtre est modale (grab_set dans son __init__), donc l'exécution attend ici.

        except Exception as e:
            error_msg = f"Erreur création fenêtre paramètres mosaïque: {e}"
            print(f"ERREUR (GUI): {error_msg}")
            traceback.print_exc(limit=2)
            messagebox.showerror(
                self.tr("error", default="Error"),
                self.tr(
                    "mosaic_window_create_error",
                    default="Could not open Mosaic settings window.",
                )
                + f"\n{e}",
                parent=self.root,
            )
            # Assurer la réinitialisation de la référence si erreur
            self._mosaic_settings_window_instance = None

    ##################################################################################################################################

    def _open_local_solver_settings_window(self):
        """
        Ouvre la fenêtre modale pour configurer les options des solveurs locaux.
        """
        print("DEBUG (GUI): Clic sur bouton 'Local Solvers...' - Appel de _open_local_solver_settings_window.")

        # Optionnel: Vérifier si une instance existe déjà (pourrait être utile pour le développement)
        if (
            hasattr(self, "_local_solver_settings_window_instance")
            and self._local_solver_settings_window_instance
            and self._local_solver_settings_window_instance.winfo_exists()
        ):
            print("DEBUG (GUI): Fenêtre de paramètres des solveurs locaux déjà ouverte. Mise au premier plan.")
            try:
                self._local_solver_settings_window_instance.lift()
                self._local_solver_settings_window_instance.focus_force()
            except tk.TclError:  # Au cas où la fenêtre aurait été détruite entre-temps
                self._local_solver_settings_window_instance = None  # Réinitialiser
                # Essayer de recréer ci-dessous
            else:
                return

        # Créer et afficher la nouvelle fenêtre modale
        try:
            print("DEBUG (GUI): Création de l'instance LocalSolverSettingsWindow...")
            solver_window = LocalSolverSettingsWindow(parent_gui=self)
            self._local_solver_settings_window_instance = solver_window  # Stocker référence (optionnel)
            print("DEBUG (GUI): Instance LocalSolverSettingsWindow créée.")
            # La fenêtre est modale (grab_set dans son __init__), donc l'exécution attend ici.

        except Exception as e:
            error_msg_key = "local_solver_window_create_error"  # Nouvelle clé de traduction
            error_default_text = "Could not open Local Solvers settings window."
            full_error_msg = self.tr(error_msg_key, default=error_default_text) + f"\n{e}"

            print(f"ERREUR (GUI): Erreur création fenêtre paramètres solveurs locaux: {e}")
            traceback.print_exc(limit=2)
            messagebox.showerror(self.tr("error", default="Error"), full_error_msg, parent=self.root)
            self._local_solver_settings_window_instance = None

    ##################################################################################################################################

    def _save_settings_and_destroy(self):
        try:
            if self.root.winfo_exists():
                self.settings.window_geometry = self.root.geometry()
        except tk.TclError:
            pass
        self.settings.update_from_ui(self)
        print(f"VÉRIF GUI: self.settings.astap_path AVANT save_settings = '{self.settings.astap_path}'")  # <-- AJOUTER
        self.settings.save_settings()
        print("Fermeture de l'application.")
        self.root.destroy()

    # --- NOUVELLES METHODES ---
    def _copy_log_to_clipboard(self):
        """Copie le contenu de la zone de log dans le presse-papiers."""
        try:
            log_content = self.status_text.get(1.0, tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(log_content)
            self.update_progress_gui("ⓘ Contenu du log copié dans le presse-papiers.", None)
        except tk.TclError as e:
            print(f"Erreur Tcl lors de la copie du log: {e}")
        except Exception as e:
            print(f"Erreur copie log: {e}")
            self.update_progress_gui(f"❌ Erreur copie log: {e}", None)
            messagebox.showerror(self.tr("error"), f"Impossible de copier le log:\n{e}")

    def _open_output_folder(self):
        """Ouvre le dossier de sortie dans l'explorateur de fichiers système."""
        output_folder = self.output_path.get()
        if not output_folder:
            messagebox.showwarning(self.tr("warning"), "Le chemin du dossier de sortie n'est pas défini.")
            return
        if not os.path.isdir(output_folder):
            messagebox.showerror(
                self.tr("error"),
                f"Le dossier de sortie n'existe pas :\n{output_folder}",
            )
            return
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(output_folder)
            elif system == "Darwin":
                subprocess.Popen(["open", output_folder])
            else:
                subprocess.Popen(["xdg-open", output_folder])
            self.update_progress_gui(f"ⓘ Ouverture du dossier: {output_folder}", None)
        except FileNotFoundError:
            messagebox.showerror(
                self.tr("error"),
                f"Impossible d'ouvrir le dossier.\nCommande non trouvée pour votre système ({system}).",
            )
        except Exception as e:
            print(f"Erreur ouverture dossier: {e}")
            messagebox.showerror(self.tr("error"), f"Impossible d'ouvrir le dossier:\n{e}")
            self.update_progress_gui(f"❌ Erreur ouverture dossier: {e}", None)

    #############################################################################################################################################

    def _open_unaligned_folder_from_summary(self, folder_path):
        """
        Ouvre le dossier 'unaligned_by_stacker' spécifié dans l'explorateur de fichiers système.
        Cette méthode est appelée par le bouton "Open Unaligned" dans le résumé.
        """
        if not folder_path:
            messagebox.showwarning(
                self.tr("warning"),
                self.tr(
                    "unaligned_folder_path_missing",
                    default="Le chemin du dossier des non-alignés n'est pas défini.",
                ),
            )
            return

        # S'assurer que le chemin est absolu
        abs_folder_path = os.path.abspath(folder_path)

        if not os.path.isdir(abs_folder_path):
            messagebox.showerror(
                self.tr("error"),
                self.tr(
                    "unaligned_folder_not_found",
                    default="Le dossier des non-alignés n'existe pas ou n'est pas un répertoire :",
                )
                + f"\n{abs_folder_path}",
            )
            return

        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(abs_folder_path)
            elif system == "Darwin":  # macOS
                subprocess.Popen(["open", abs_folder_path])
            else:  # Linux et autres (xdg-open est courant)
                subprocess.Popen(["xdg-open", abs_folder_path])

            self.update_progress_gui(
                self.tr(
                    "unaligned_folder_opened",
                    default="Ouverture du dossier des non-alignés :",
                )
                + f" {abs_folder_path}",
                None,
            )

        except FileNotFoundError:
            messagebox.showerror(
                self.tr("error"),
                self.tr(
                    "cannot_open_folder_command_not_found",
                    default="Impossible d'ouvrir le dossier. Commande système non trouvée pour votre OS.",
                ),
            )
        except Exception as e:
            print(f"Erreur ouverture dossier non-alignés: {e}")
            traceback.print_exc()
            messagebox.showerror(
                self.tr("error"),
                self.tr(
                    "error_opening_unaligned_folder",
                    default="Une erreur est survenue lors de l'ouverture du dossier des non-alignés :",
                )
                + f"\n{e}",
            )
            self.update_progress_gui(
                self.tr(
                    "error_opening_unaligned_folder_short",
                    default="Erreur ouverture dossier non-alignés.",
                ),
                "ERROR",
            )

    ###############################################################################################################################################

    # --- DANS LA CLASSE SeestarStackerGUI DANS seestar/gui/main_window.py ---

    def update_progress_gui(self, message: str, progress: float | None = None):
        """
        Affiche un message dans le widget log et met à jour la barre de progression.
        S'assure d'être toujours exécuté dans le thread principal Tkinter.
        """
        # --- ROUTAGE VERS LE THREAD GUI ---
        if threading.current_thread() is not threading.main_thread():
            # Planifie l'exécution dans la boucle d'événements Tkinter et sort.
            self.root.after(0, lambda m=message, p=progress: self.update_progress_gui(m, p))
            return

        # --- CODE EXISTANT (garde intact le reste) ------------------------------
        # Gérer le message spécial pour le compteur de dossiers (inchangé)
        if isinstance(message, str) and message.startswith("folder_count_update:"):
            try:
                self.root.after_idle(self.update_additional_folders_display)
            except tk.TclError:
                pass
            return

        eta_prefix = "ETA_UPDATE:"
        if isinstance(message, str) and message.startswith(eta_prefix):
            eta_str = message.split(":", 1)[1]
            if hasattr(self, "progress_manager") and self.progress_manager:
                pm = self.progress_manager
                if hasattr(pm, "set_remaining") and callable(pm.set_remaining):
                    pm.set_remaining(eta_str)
                elif hasattr(pm, "remaining_time_var"):
                    try:
                        pm.remaining_time_var.set(eta_str)
                    except Exception:
                        pass
            return

        actual_message_to_log = message
        log_level_for_pm = None  # Sera "WARN" pour notre message spécial, sinon None (par défaut)

        # --- NOUVEAU : Gérer le message d'information sur les fichiers non alignés ---
        unaligned_info_prefix = "UNALIGNED_INFO:"
        if isinstance(message, str) and message.startswith(unaligned_info_prefix):
            actual_message_to_log = message[len(unaligned_info_prefix) :].strip()  # Extraire le message réel
            log_level_for_pm = "WARN"  # Utiliser "WARN" pour que ProgressManager puisse le distinguer
            print(
                f"DEBUG GUI [update_progress_gui]: Message UNALIGNED_INFO détecté: '{actual_message_to_log}' (niveau WARN)"
            )
        # --- FIN NOUVEAU ---

        # Procéder à la mise à jour via ProgressManager si disponible
        if hasattr(self, "progress_manager") and self.progress_manager:
            final_drizzle_active = False
            # Utiliser actual_message_to_log pour la détection Drizzle
            if isinstance(actual_message_to_log, str):
                if "💧 Exécution Drizzle final" in actual_message_to_log:
                    final_drizzle_active = True
                    # Le préfixe emoji est déjà dans le message original du backend
                    # actual_message_to_log = "⏳ " + actual_message_to_log

            if isinstance(actual_message_to_log, str) and "terminé" in actual_message_to_log.lower():
                progress = 100

            # Mettre à jour la barre et le log texte via ProgressManager
            # On passe le message (potentiellement modifié) et le niveau de log déterminé
            self.progress_manager.update_progress(actual_message_to_log, progress, level=log_level_for_pm)

            # Gérer le mode indéterminé de la barre de progression (inchangé)
            try:
                pb = self.progress_manager.progress_bar
                if pb.winfo_exists():
                    current_mode = pb["mode"]
                    if final_drizzle_active and current_mode != "indeterminate":
                        pb.config(mode="indeterminate")
                        pb.start(15)
                    elif not final_drizzle_active and current_mode == "indeterminate":
                        pb.stop()
                        pb.config(mode="determinate")
                        if progress is not None:
                            try:
                                pb.configure(value=max(0.0, min(100.0, float(progress))))
                            except ValueError:
                                pass
            except (tk.TclError, AttributeError):
                pass
        # else: # Si ProgressManager n'est pas encore prêt (ne devrait pas arriver après init)
        # print(f"DEBUG GUI [update_progress_gui]: ProgressManager non disponible. Message: {actual_message_to_log}")

    ###################################################################################################################################################

    def _execute_final_auto_stretch(self, original_lock_state_before_after):
        self.logger.info(">>>> Entrée dans _execute_final_auto_stretch (appelé par after depuis _processing_finished)")
        # Le verrou est _final_stretch_set_by_processing_finished (mis à True à la fin de _processing_finished)
        # On s'assure qu'il est bien False pour que apply_auto_stretch s'exécute
        # Puis on le remet à True.

        # Sauvegarder l'état actuel du verrou (qui devrait être True ici)
        current_lock_state_at_execution = self._final_stretch_set_by_processing_finished

        self.logger.info(
            f"     _execute_final_auto_stretch: État du verrou avant exécution apply_auto_stretch: {current_lock_state_at_execution}"
        )

        # Forcer le verrou à False pour cet appel spécifique
        self._final_stretch_set_by_processing_finished = False
        self.logger.info(
            f"     _execute_final_auto_stretch: Verrou temporairement mis à {self._final_stretch_set_by_processing_finished} pour apply_auto_stretch."
        )

        try:
            self.apply_auto_stretch()
        finally:
            # Rétablir l'état du verrou qu'il avait au moment où _processing_finished a activé le verrou,
            # ou simplement le forcer à True. Forcer à True est plus simple.
            self._final_stretch_set_by_processing_finished = True
            self.logger.info(
                f"     _execute_final_auto_stretch: Verrou final rétabli à {self._final_stretch_set_by_processing_finished} après apply_auto_stretch."
            )

    #############################################################################################################################################

    def _processing_finished(self):
        self.logger.info(
            ">>>> Entrée dans SeestarStackerGUI._processing_finished (V_FinalAutoStretchLogic_2_DirectCall)"
        )  # Log d'entrée avec version

        # Annuler tout appel différé qui aurait pu être planifié pour auto_stretch
        if hasattr(self, "_auto_stretch_after_id") and self._auto_stretch_after_id:
            try:
                self.root.after_cancel(self._auto_stretch_after_id)
                self.logger.info("     _processing_finished: Appel différé _auto_stretch_after_id ANNULÉ.")
            except tk.TclError:
                self.logger.warning(
                    "     _processing_finished: Erreur TclError lors de l'annulation de _auto_stretch_after_id (déjà exécuté ou invalide?)."
                )
            except Exception as e_cancel_stretch:
                self.logger.error(
                    f"     _processing_finished: Erreur inattendue lors de l'annulation de _auto_stretch_after_id: {e_cancel_stretch}"
                )
            self._auto_stretch_after_id = None

        if hasattr(self, "_auto_wb_after_id") and self._auto_wb_after_id:
            try:
                self.root.after_cancel(self._auto_wb_after_id)
                self.logger.info("     _processing_finished: Appel différé _auto_wb_after_id ANNULÉ.")
            except tk.TclError:
                self.logger.warning(
                    "     _processing_finished: Erreur TclError lors de l'annulation de _auto_wb_after_id (déjà exécuté ou invalide?)."
                )
            except Exception as e_cancel_wb:
                self.logger.error(
                    f"     _processing_finished: Erreur inattendue lors de l'annulation de _auto_wb_after_id: {e_cancel_wb}"
                )
            self._auto_wb_after_id = None

        if not self.processing:
            self.logger.warning(
                "     _processing_finished appelé mais self.processing est déjà False. Sortie anticipée."
            )
            return

        self.logger.info("     _processing_finished: self.processing mis à False.")
        self.processing = False

        # --- Section 1: Finalisation Barre de Progression et Timer ---
        try:
            self.logger.info("  [PF_S1] _processing_finished: Finalisation Barre de Progression et Timer...")
            if hasattr(self, "progress_manager") and self.progress_manager:
                self.progress_manager.stop_timer()
                if hasattr(self.progress_manager, "progress_bar") and self.progress_manager.progress_bar.winfo_exists():
                    pb = self.progress_manager.progress_bar
                    if pb["mode"] == "indeterminate":
                        pb.stop()
                        pb.config(mode="determinate")
                    is_error_backend = (
                        hasattr(self, "queued_stacker")
                        and self.queued_stacker
                        and getattr(self.queued_stacker, "processing_error", None) is not None
                    )
                    is_stopped_early_backend = (
                        hasattr(self, "queued_stacker")
                        and self.queued_stacker
                        and getattr(self.queued_stacker, "stop_processing_flag_for_gui", False)
                    )
                    if not is_error_backend and not is_stopped_early_backend:
                        pb.configure(value=100)
            self.logger.info("  [PF_S1] _processing_finished: Barre/Timer OK.")
        except Exception as e_s1:
            self.logger.error(
                f"  [PF_S1] _processing_finished: ERREUR Barre/Timer: {e_s1}\n{traceback.format_exc(limit=1)}"
            )

        # --- Section 2: Récupération des informations du backend ---
        self.logger.info("  [PF_S2] _processing_finished: Récupération des informations du backend...")
        final_stack_path = None
        processing_error_details = None
        images_stacked = 0
        # ... (copiez ici TOUT le bloc de récupération des variables depuis q_stacker comme dans votre version précédente de _processing_finished) ...
        # Assurez-vous que les lignes suivantes sont bien présentes et correctes :
        q_stacker = getattr(self, "queued_stacker", None)
        cosmetic_01_data_for_preview_from_backend = None
        raw_adu_data_for_histo_from_backend = None
        final_header_for_ui_preview = None
        save_as_float32_backend_setting = False  # Valeur par défaut

        if q_stacker is not None:
            final_stack_path = getattr(q_stacker, "final_stacked_path", None)
            drizzle_active_session_backend = getattr(q_stacker, "drizzle_active_session", False)
            drizzle_mode_backend = getattr(q_stacker, "drizzle_mode", "Final")
            was_stopped_by_user = getattr(q_stacker, "stop_processing_flag_for_gui", False)
            processing_error_details = getattr(q_stacker, "processing_error", None)
            source_folders_with_unaligned_in_run = getattr(q_stacker, "warned_unaligned_source_folders", set())
            images_in_cumulative_from_backend = getattr(q_stacker, "images_in_cumulative_stack", 0)
            aligned_count = getattr(q_stacker, "aligned_files_count", 0)
            failed_align_count = getattr(q_stacker, "failed_align_count", 0)
            failed_stack_count = getattr(q_stacker, "failed_stack_count", 0)
            skipped_count = getattr(q_stacker, "skipped_files_count", 0)
            processed_files_count = getattr(q_stacker, "processed_files_count", 0)
            total_exposure = getattr(q_stacker, "total_exposure_seconds", 0.0)
            photutils_applied_this_run_backend = getattr(q_stacker, "photutils_bn_applied_in_session", False)
            bn_globale_applied_this_run_backend = getattr(q_stacker, "bn_globale_applied_in_session", False)
            cb_applied_in_session_backend = getattr(q_stacker, "cb_applied_in_session", False)
            scnr_applied_this_run_backend = getattr(q_stacker, "scnr_applied_in_session", False)
            crop_applied_this_run_backend = getattr(q_stacker, "crop_applied_in_session", False)
            feathering_applied_this_run_backend = getattr(q_stacker, "feathering_applied_in_session", False)
            low_wht_mask_applied_this_run_backend = getattr(q_stacker, "low_wht_mask_applied_in_session", False)
            photutils_params_used_backend = getattr(q_stacker, "photutils_params_used_in_session", {}).copy()
            raw_adu_data_for_histo_from_backend = getattr(q_stacker, "raw_adu_data_for_ui_histogram", None)
            cosmetic_01_data_for_preview_from_backend = getattr(
                q_stacker, "last_saved_data_for_preview", None
            )  # Devrait être [0,1] NON-stretché
            final_header_for_ui_preview = getattr(q_stacker, "current_stack_header", fits.Header())
            save_as_float32_backend_setting = getattr(q_stacker, "save_final_as_float32", False)
            is_drizzle_result = (
                drizzle_active_session_backend
                and not was_stopped_by_user
                and processing_error_details is None
                and final_stack_path is not None
                and (
                    "_drizzle" in os.path.basename(final_stack_path).lower()
                    or "_mosaic" in os.path.basename(final_stack_path).lower()
                    or "_reproject" in os.path.basename(final_stack_path).lower()
                )
            )
            if is_drizzle_result:
                images_stacked = (
                    aligned_count
                    if drizzle_mode_backend == "Final"
                    or "_mosaic" in os.path.basename(final_stack_path).lower()
                    or "_reproject" in os.path.basename(final_stack_path).lower()
                    else images_in_cumulative_from_backend
                )
            else:
                images_stacked = images_in_cumulative_from_backend
        else:
            processing_error_details = "Backend (QueuedStacker) non trouvé."
            # ... (autres initialisations par défaut pour les variables normalement lues du backend)

        self.logger.info(
            f"  [PF_S2] _processing_finished: Infos backend OK. Erreur: {processing_error_details}, Stack: {final_stack_path}"
        )

        # --- Section 3: Message de statut final ---
        # (Code original repris du log - ce bloc reste inchangé)
        status_text_for_log = self.tr("stacking_finished")
        # ... (toute la logique pour définir status_text_for_log et final_stack_type_for_summary)
        # ... (jusqu'à l'appel à self.progress_manager.update_progress)
        if was_stopped_by_user:
            status_text_for_log = self.tr("processing_stopped")
        elif processing_error_details:
            status_text_for_log = f"{self.tr('stacking_error_msg')} {processing_error_details}"
            final_stack_type_for_summary = "Erreur"
        elif not (final_stack_path and os.path.exists(final_stack_path)):
            status_text_for_log = self.tr(
                "Terminé (Aucun stack final créé)",
                default="Finished (No final stack created)",
            )
            final_stack_type_for_summary = "Aucun"
        elif is_drizzle_result:
            if (
                "_mosaic" in os.path.basename(final_stack_path).lower()
                or "_reproject" in os.path.basename(final_stack_path).lower()
            ):
                final_stack_type_for_summary = "Mosaïque Drizzle/Reproject"
            else:
                final_stack_type_for_summary = "Drizzle"
            status_text_for_log = self.tr(
                f"{final_stack_type_for_summary.lower()}_complete",
                default=f"{final_stack_type_for_summary} Complete",
            )
        else:
            final_stack_type_for_summary = "Classique"
            status_text_for_log = self.tr("stacking_classic_complete", default="Classic Stacking Complete")
        try:
            if hasattr(self, "progress_manager") and self.progress_manager:
                self.progress_manager.update_progress(
                    status_text_for_log,
                    (100 if not processing_error_details else self.progress_manager.progress_bar["value"]),
                )
        except Exception as e_s3:
            self.logger.error(
                f"  [PF_S3] _processing_finished: ERREUR Message Statut Final: {e_s3}\n{traceback.format_exc(limit=1)}"
            )

        # --- Section 4: Mise à jour de l'Aperçu et de l'Histogramme ---
        self.logger.info(
            "  [PF_S4 - MODIFIÉ FINAL AUTOSTRETCH] _processing_finished: Préparation données pour aperçu/histogramme final..."
        )
        preview_load_error_msg = None
        def _load_final_preview():
            nonlocal processing_error_details, preview_load_error_msg
            try:
                data_final = None
                header_final = None
                preview_load_error_msg = None

                if cosmetic_01_data_for_preview_from_backend is not None:
                    self.logger.info(
                        "    [PF_S4] Utilisation des données de prévisualisation fournies par le backend."
                    )
                    data_final = cosmetic_01_data_for_preview_from_backend
                    header_final = (
                        final_header_for_ui_preview if final_header_for_ui_preview else fits.Header()
                    )
                elif final_stack_path and os.path.exists(final_stack_path):
                    try:
                        data_final, header_final = load_and_validate_fits(final_stack_path)
                        self.logger.info(
                            f"    [PF_S4] FITS final chargé. Shape: {data_final.shape if data_final is not None else 'None'}"
                        )
                    except Exception as e_load:
                        preview_load_error_msg = f"Erreur chargement FITS final: {e_load}"
                        self.logger.error(f"    [PF_S4] {preview_load_error_msg}")
                else:
                    preview_load_error_msg = "Fichier FITS final introuvable."
                    self.logger.warning(f"    [PF_S4] {preview_load_error_msg}")

                if data_final is not None:
                    try:
                        data_final_ds = downsample_image(data_final, factor=2)
                    except Exception:
                        data_final_ds = data_final

                    def _apply_gui_updates(df=data_final_ds, hdr=header_final):
                        self.current_preview_data = df
                        self.current_preview_hist_data = df
                        self._temp_data_for_final_histo = df
                        self.current_stack_header = hdr if hdr else fits.Header()

                        if hasattr(self, "preview_manager") and self.preview_manager:
                            linear_params = {
                                "stretch_method": "Linear",
                                "black_point": 0.0,
                                "white_point": 1.0,
                                "gamma": 1.0,
                                "r_gain": 1.0,
                                "g_gain": 1.0,
                                "b_gain": 1.0,
                                "brightness": 1.0,
                                "contrast": 1.0,
                                "saturation": 1.0,
                            }
                            self.preview_manager.update_preview(
                                self.current_preview_data,
                                linear_params,
                                stack_count=images_stacked,
                                total_images=images_stacked,
                                current_batch=self.preview_current_batch,
                                total_batches=self.preview_total_batches,
                            )

                            if hasattr(self, "histogram_widget") and self.histogram_widget:
                                self.histogram_widget.update_histogram(self.current_preview_data)
                                try:
                                    bp_ui = self.preview_black_point.get()
                                    wp_ui = self.preview_white_point.get()
                                except tk.TclError:
                                    bp_ui = None
                                    wp_ui = None
                                if bp_ui is not None and wp_ui is not None:
                                    self.histogram_widget.set_range(bp_ui, wp_ui)
                                self.refresh_preview(recalculate_histogram=False)

                        if self.current_stack_header:
                            self.update_image_info(self.current_stack_header)

                        self.logger.info(
                            f"  [PF_S4] _processing_finished: Préparation données aperçu/histo OK. Erreur chargement: {preview_load_error_msg}"
                        )

                    self.root.after(0, _apply_gui_updates)
                else:
                    def _no_data():
                        if hasattr(self, "preview_manager"):
                            self.preview_manager.clear_preview(preview_load_error_msg or "Preview load error")
                        if hasattr(self, "histogram_widget"):
                            self.histogram_widget.plot_histogram(None)
                        self.logger.info(
                            f"  [PF_S4] _processing_finished: Préparation données aperçu/histo OK. Erreur chargement: {preview_load_error_msg}"
                        )

                    self.root.after(0, _no_data)
            except Exception as e_s4:
                def _err_handler():
                    if hasattr(self, "preview_manager"):
                        self.preview_manager.clear_preview(
                            f"Erreur MAJEURE mise a jour apercu final: {e_s4}"
                        )
                    if hasattr(self, "histogram_widget"):
                        self.histogram_widget.plot_histogram(None)
                self.logger.error(
                    f"  [PF_S4] _processing_finished: ERREUR CRITIQUE Aperçu/Histo: {e_s4}\n{traceback.format_exc(limit=2)}"
                )
                self.root.after(0, _err_handler)
                processing_error_details = (
                    f"{processing_error_details or ''} Erreur UI Preview/Histo: {e_s4}"
                )

        threading.Thread(target=_load_final_preview, daemon=True, name="FinalPreviewLoader").start()

        # --- Section 5: Génération et Affichage du Résumé ---
        # (Code original repris du log - ce bloc reste inchangé)
        self.logger.info("  [PF_S5] _processing_finished: Génération et Affichage du Résumé...")
        # ... (toute la logique de création de summary_lines) ...
        # ... (jusqu'à l'appel à self.root.after(150, lambda: self._show_summary_dialog(...))) ...
        try:
            summary_lines = []
            summary_title = self.tr("processing_report_title")
            summary_lines.append(f"{self.tr('Status', default='Status')}: {status_text_for_log}")
            elapsed_total_seconds = 0
            if self.global_start_time:
                elapsed_total_seconds = time.monotonic() - self.global_start_time
            summary_lines.append(
                f"{self.tr('Total Processing Time', default='Total Processing Time')}: {self._format_duration(elapsed_total_seconds)}"
            )
            summary_lines.append(
                f"{self.tr('Final Stack Type', default='Final Stack Type')}: {final_stack_type_for_summary}"
            )
            summary_lines.append(f"{self.tr('Files Attempted', default='Files Attempted')}: {processed_files_count}")
            total_rejected = failed_align_count + failed_stack_count + skipped_count
            summary_lines.append(
                f"{self.tr('Files Rejected (Total)', default='Files Rejected (Total)')}: {total_rejected} ({self.tr('Align', default='Align')}: {failed_align_count}, {self.tr('Stack Err', default='Stack Err')}: {failed_stack_count}, {self.tr('Other', default='Other')}: {skipped_count})"
            )
            summary_lines.append(
                f"{self.tr('Images in Final Stack', default='Images in Final Stack')} ({final_stack_type_for_summary}): {images_stacked}"
            )
            summary_lines.append(
                f"{self.tr('Total Exposure (Final Stack)', default='Total Exposure (Final Stack)')}: {self._format_duration(total_exposure)}"
            )
            summary_lines.append(f"\n--- {self.tr('Post-Processing Applied', default='Post-Processing Applied')} ---")
            summary_lines.append(
                f"  - {self.tr('Global Background Neutralization (BN)', default='Global Background Neutralization (BN)')}: {'Yes' if bn_globale_applied_this_run_backend else 'No'}"
            )
            if photutils_applied_this_run_backend:
                params_str_list = []
                photutils_params_to_log = [
                    "box_size",
                    "filter_size",
                    "sigma_clip_val",
                    "exclude_percentile",
                ]
                for p_key in photutils_params_to_log:
                    if p_key in photutils_params_used_backend:
                        val = photutils_params_used_backend[p_key]
                        p_name_short = (
                            p_key.replace("photutils_bn_", "")
                            .replace("_val", "")
                            .replace("_percentile", "%")
                            .replace("filter_size", "Filt")
                            .replace("box_size", "Box")
                            .replace("sigma_clip", "Sig")
                            .title()
                        )
                        params_str_list.append(
                            f"{p_name_short}={val:.1f}" if isinstance(val, float) else f"{p_name_short}={val}"
                        )
                params_str = ", ".join(params_str_list) if params_str_list else "Defaults"
                summary_lines.append(
                    f"  - {self.tr('Photutils 2D Background', default='Photutils 2D Background')}: {self.tr('Yes', default='Yes')} ({params_str})"
                )
            else:
                summary_lines.append(
                    f"  - {self.tr('Photutils 2D Background', default='Photutils 2D Background')}: {self.tr('No', default='No')}"
                )
            summary_lines.append(
                f"  - {self.tr('Edge/Chroma Correction (CB)', default='Edge/Chroma Correction (CB)')}: {'Yes' if cb_applied_in_session_backend else 'No'}"
            )
            summary_lines.append(f"  - Feathering: {'Yes' if feathering_applied_this_run_backend else 'No'}")
            summary_lines.append(f"  - Low WHT Mask: {'Yes' if low_wht_mask_applied_this_run_backend else 'No'}")
            scnr_target_sum = getattr(q_stacker, "final_scnr_target_channel", "?") if q_stacker else "?"
            scnr_amount_sum = getattr(q_stacker, "final_scnr_amount", 0.0) if q_stacker else 0.0
            scnr_lum_sum = getattr(q_stacker, "final_scnr_preserve_luminosity", "?") if q_stacker else "?"
            crop_perc_decimal_sum = getattr(q_stacker, "final_edge_crop_percent_decimal", 0.0) if q_stacker else 0.0
            scnr_info_summary = (
                f"{self.tr('Yes', default='Yes')} (Cible: {scnr_target_sum}, Force: {scnr_amount_sum:.2f}, Pres.Lum: {scnr_lum_sum})"
                if scnr_applied_this_run_backend
                else self.tr("No", default="No")
            )
            summary_lines.append(f"  - {self.tr('Final SCNR', default='Final SCNR')}: {scnr_info_summary}")
            crop_info_summary = (
                f"{self.tr('Yes', default='Yes')} ({crop_perc_decimal_sum*100.0:.1f}%)"
                if crop_applied_this_run_backend
                else self.tr("No", default="No")
            )
            summary_lines.append(f"  - {self.tr('Final Edge Crop', default='Final Edge Crop')}: {crop_info_summary}")
            summary_lines.append("-------------------------------")
            if final_stack_path and os.path.exists(final_stack_path):
                summary_lines.append(
                    f"\n{self.tr('Final Stack File', default='Final Stack File')}:\n  {final_stack_path}"
                )
            elif final_stack_path:
                summary_lines.append(
                    f"{self.tr('Final Stack File', default='Final Stack File')}:\n  {final_stack_path} ({self.tr('Not Found!', default='Not Found!')})"
                )
            else:
                summary_lines.append(
                    self.tr(
                        "Final Stack File: Not created or not found.",
                        default="Final Stack File: Not created or not found.",
                    )
                )
            if preview_load_error_msg:
                summary_lines.append(f"\nNote Apercu UI: {preview_load_error_msg}")
            full_summary_text_for_dialog = "\n".join(summary_lines)
            can_open_output_folder_button = (
                self.output_path.get()
                and os.path.isdir(self.output_path.get())
                and ((final_stack_path and os.path.exists(final_stack_path)) or not processing_error_details)
            )
            show_summary = True
            if was_stopped_by_user and not (final_stack_path and os.path.exists(final_stack_path)):
                show_summary = False
                self.logger.info("--- Processing Stopped by User, No Final File (Summary Dialog Skipped) ---")
            elif processing_error_details and not (final_stack_path and os.path.exists(final_stack_path)):
                show_summary = False
                self.root.after(
                    100,
                    lambda: messagebox.showerror(self.tr("error"), f"{status_text_for_log}", parent=self.root),
                )

            if show_summary:
                self.logger.info("    [PF_S5] Planification affichage dialogue résumé...")
                # --- CORRECTION FINALE ET SIMPLIFIÉE ---
                self.root.after(
                    150,
                    lambda title_arg=summary_title, text_arg=full_summary_text_for_dialog, can_open_arg=can_open_output_folder_button, unaligned_sources_arg=source_folders_with_unaligned_in_run,
                    # Lire et capturer la valeur de self.input_path.get() ICI
                    input_path_for_button_arg=self.input_path.get(): self._show_summary_dialog(
                        title_arg,
                        text_arg,
                        can_open_arg,
                        unaligned_sources_arg,
                        input_path_for_button_arg,
                    ),
                )  # Utiliser l'argument capturé
                # --- FIN MODIFICATION ---

            self.logger.info("  [PF_S5] _processing_finished: Résumé OK.")
        except Exception as e_s5:
            self.logger.error(
                f"  [PF_S5] _processing_finished: ERREUR CRITIQUE Résumé: {e_s5}\n{traceback.format_exc(limit=2)}"
            )
            messagebox.showerror(
                "Erreur Resume Critique",
                f"Erreur majeure lors de la generation du resume:\n{e_s5}",
                parent=self.root,
            )
            processing_error_details = f"{processing_error_details or ''} Erreur UI Resume: {e_s5}"

        # --- Section 6: Réinitialisation de l'état de l'UI ---
        self.logger.info("  [PF_S6] _processing_finished: Réinitialisation de l'état de l'UI...")
        # ... (Code de _set_parameter_widgets_state(tk.NORMAL) etc. - reste inchangé) ...
        try:
            self._set_parameter_widgets_state(tk.NORMAL)
            if hasattr(self, "start_button") and self.start_button.winfo_exists():
                self.start_button.config(state=tk.NORMAL)
            if hasattr(self, "stop_button") and self.stop_button.winfo_exists():
                self.stop_button.config(state=tk.DISABLED)
            can_open_output_final = bool(
                self.output_path.get()
                and os.path.isdir(self.output_path.get())
                and (
                    (final_stack_path and os.path.exists(final_stack_path))
                    or (not processing_error_details and not was_stopped_by_user)
                )
            )
            if hasattr(self, "open_output_button") and self.open_output_button.winfo_exists():
                self.open_output_button.config(state=tk.NORMAL if can_open_output_final else tk.DISABLED)
            if hasattr(self, "remaining_time_var"):
                self.remaining_time_var.set("00:00:00")
            self.additional_folders_to_process = []
            self.update_additional_folders_display()
            self.update_remaining_files()
            self.logger.info("  [PF_S6] _processing_finished: UI OK.")
        except Exception as e_s6:
            self.logger.error(
                f"  [PF_S6] _processing_finished: ERREUR CRITIQUE UI: {e_s6}\n{traceback.format_exc(limit=1)}"
            )
            if hasattr(self, "start_button") and self.start_button.winfo_exists():
                self.start_button.config(state=tk.NORMAL)
            if hasattr(self, "stop_button") and self.stop_button.winfo_exists():
                self.stop_button.config(state=tk.DISABLED)

        # --- Activation FINALE du verrou ---
        self.logger.info(
            "     _processing_finished: Activation du verrou _final_stretch_set_by_processing_finished = True (après l'auto-stretch final)."
        )
        self._final_stretch_set_by_processing_finished = True

        # Nettoyer _temp_data_for_final_histo si elle a été utilisée
        if hasattr(self, "_temp_data_for_final_histo"):
            self.logger.info("     _processing_finished: Nettoyage _temp_data_for_final_histo après utilisation.")
            self._temp_data_for_final_histo = None

        if "gc" in globals() or "gc" in locals():
            gc.collect()
        # --- FIN DU CODE MODIFIÉ ---

    def _refresh_final_preview_and_histo_direct(self):
        # --- DÉBUT DU CODE MODIFIÉ ---
        self.logger.info(
            ">>>> Entrée dans SeestarStackerGUI._refresh_final_preview_and_histo_direct (V_DirectFinalRefresh_2_OrderCheck_LogComplet)"
        )  # Version Log

        # Récupérer les données qui ont été préparées par _processing_finished
        current_data_for_histo = getattr(self, "_temp_data_for_final_histo", None)
        if current_data_for_histo is None:
            self.logger.info(
                "     _refresh_final_preview_and_histo_direct: _temp_data_for_final_histo non trouvé. Utilisation de self.current_preview_data pour l'histogramme."
            )
            current_data_for_histo = self.current_preview_data

        if current_data_for_histo is None:
            self.logger.warning(
                "     _refresh_final_preview_and_histo_direct: Aucune donnée disponible pour l'histogramme final. Tentative d'effacement de l'aperçu."
            )
            if hasattr(self, "preview_manager"):
                self.preview_manager.clear_preview("No final data for preview.")
            if hasattr(self, "histogram_widget"):
                self.histogram_widget.plot_histogram(None)
            return

        try:
            # 1. Mettre à jour l'histogramme avec les données d'analyse.
            #    Ceci appelle plot_histogram qui DESSINE LES BARRES et CONFIGURE LES AXES X et Y.
            #    Les lignes BP/WP ne sont PAS dessinées par plot_histogram.
            if hasattr(self, "histogram_widget") and self.histogram_widget:
                self.logger.info(
                    f"     _refresh_final_preview_and_histo_direct: Appel histogram_widget.update_histogram avec données (Shape: {current_data_for_histo.shape})"
                )
                self.histogram_widget.update_histogram(current_data_for_histo)

            # 2. Lire les valeurs BP/WP des sliders UI (qui ont été settées par _processing_finished)
            bp_ui = self.preview_black_point.get()
            wp_ui = self.preview_white_point.get()
            self.logger.info(
                f"     _refresh_final_preview_and_histo_direct: Valeurs lues depuis UI pour set_range: BP_UI={bp_ui:.4f}, WP_UI={wp_ui:.4f}"
            )

            # 3. Mettre à jour les LIGNES BP/WP de l'histogramme avec ces valeurs UI.
            #    set_range convertit BP/WP UI (0-1) à l'échelle des données affichées
            #    par l'histogramme et DESSINE/MET À JOUR les lignes.
            if hasattr(self, "histogram_widget") and self.histogram_widget:
                self.logger.info(
                    f"     _refresh_final_preview_and_histo_direct: Appel histogram_widget.set_range avec BP_UI={bp_ui:.4f}, WP_UI={wp_ui:.4f} depuis les sliders."
                )
                self.histogram_widget.set_range(bp_ui, wp_ui)
        except tk.TclError as e:
            self.logger.error(
                f"     _refresh_final_preview_and_histo_direct: Erreur TclError pendant la mise à jour de l'histogramme: {e}"
            )
        except Exception as e_histo:
            self.logger.error(
                f"     _refresh_final_preview_and_histo_direct: Erreur inattendue pendant la mise à jour de l'histogramme: {e_histo}"
            )
            traceback.print_exc(limit=1)

        # 4. Mettre à jour l'aperçu visuel (PreviewManager).
        #    Il lira les paramètres UI (y compris BP/WP, méthode, etc.) pour stretcher self.current_preview_data.
        #    IMPORTANT: Ne pas recalculer l'histogramme ici (recalculate_histogram=False)
        #    car il vient d'être configuré (barres et lignes).
        self.logger.info(
            "     _refresh_final_preview_and_histo_direct: Appel self.refresh_preview(recalculate_histogram=False)."
        )
        self.refresh_preview(recalculate_histogram=False)

        # Nettoyer la donnée temporaire après utilisation
        if hasattr(self, "_temp_data_for_final_histo"):
            self.logger.info("     _refresh_final_preview_and_histo_direct: Nettoyage de _temp_data_for_final_histo.")
            self._temp_data_for_final_histo = None

        # --- FIN DU CODE MODIFIÉ ---

    ################################################################################################################################################

    def _show_summary_dialog(
        self,
        summary_title,
        summary_text,
        can_open_output,
        source_folders_with_unaligned_in_run: list[str] = None,
        input_folder_path_for_unaligned_button: str = None,
    ):
        """
        Displays a custom modal dialog with the processing summary.
        Includes optional information about unaligned files and their paths,
        and a button to open the unaligned folder.
        Version: Fix Open Unaligned Button Packing
        """
        dialog = tk.Toplevel(self.root)
        dialog.title(summary_title)
        dialog.transient(self.root)  # Associate with main window
        dialog.resizable(False, False)  # Prevent resizing

        content_frame = ttk.Frame(dialog, padding="10 10 10 10")
        content_frame.pack(expand=True, fill=tk.BOTH)

        # Icon and main summary text
        try:
            icon_label = ttk.Label(content_frame, image="::tk::icons::information", padding=(0, 0, 10, 0))
        except tk.TclError:
            icon_label = ttk.Label(
                content_frame,
                text="i",
                font=("Arial", 16, "bold"),
                padding=(0, 0, 10, 0),
            )
        icon_label.grid(row=0, column=0, sticky="nw", pady=(0, 10))

        summary_label = ttk.Label(content_frame, text=summary_text, justify=tk.LEFT, wraplength=450)
        summary_label.grid(row=0, column=1, sticky="nw", padx=(0, 10))

        # --- Variables pour la disposition des éléments suivants ---
        current_grid_row_for_next_elements = 1  # Démarre à la ligne 1 pour le message non aligné

        # --- BLOC POUR LE MESSAGE FICHIERS NON ALIGNÉS ---
        if source_folders_with_unaligned_in_run and len(source_folders_with_unaligned_in_run) > 0:
            unaligned_message_text_prefix = self.tr(
                "unaligned_files_message_prefix",
                default="Des images n'ont pas pu être alignées. Elles se trouvent dans :",
            )

            # Construire la liste des chemins formatés
            unaligned_paths_list = []
            for folder_path in sorted(
                list(source_folders_with_unaligned_in_run)
            ):  # Trier les chemins pour un affichage ordonné
                unaligned_paths_list.append(os.path.join(folder_path, "unaligned_by_stacker"))

            full_unaligned_display_text = f"{unaligned_message_text_prefix}\n"
            # Ajouter un formatage simple pour la liste des chemins
            for i, path_example in enumerate(unaligned_paths_list):
                full_unaligned_display_text += f"• {path_example}\n"  # Utilisez "• " pour des puces simples

            # Créer un Label pour le message non aligné
            self.unaligned_info_label = ttk.Label(
                content_frame,
                text=full_unaligned_display_text.strip(),
                justify=tk.LEFT,
                wraplength=450,
            )  # .strip() pour éviter un saut de ligne en trop

            # Appliquer le style rouge et gras
            try:
                self.unaligned_info_label.config(foreground="red", font=("TkDefaultFont", 9, "bold"))
            except Exception as e_style:
                print(
                    f"DEBUG GUI: Erreur application style/couleur label unaligned: {e_style}. Utilisation gras simple."
                )
                self.unaligned_info_label.config(font=("TkDefaultFont", 9, "bold"))  # Juste gras en dernier recours

            # Placer le label dans la grille. Il se trouve sur la ligne 'current_grid_row_for_next_elements'.
            self.unaligned_info_label.grid(
                row=current_grid_row_for_next_elements,
                column=0,
                columnspan=2,
                sticky="nw",
                padx=10,
                pady=(10, 10),
            )

            current_grid_row_for_next_elements += 1  # Incrémenter la ligne pour les boutons
        # --- FIN BLOC POUR LE MESSAGE FICHIERS NON ALIGNÉS ---

        # --- Cadre pour les boutons ---
        button_frame = ttk.Frame(content_frame)
        button_frame.grid(
            row=current_grid_row_for_next_elements,
            column=0,
            columnspan=2,
            sticky="se",
            pady=(15, 0),
        )

        # --- Boutons packés de DROITE à GAUCHE pour un alignement cohérent ---

        # 1. Bouton OK (le plus à droite)
        ok_button = ttk.Button(
            button_frame,
            text="OK",
            command=dialog.destroy,
            style=("Accent.TButton" if "Accent.TButton" in ttk.Style().element_names() else "TButton"),
        )
        ok_button.pack(side=tk.RIGHT)
        ok_button.focus_set()

        # 2. Bouton Copy Summary (juste à gauche de OK)
        def copy_action():
            try:
                dialog.clipboard_clear()
                dialog.clipboard_append(summary_text)
                copy_button.config(text=self.tr("Copied!", default="Copied!"))
                dialog.after(
                    1500,
                    lambda: (
                        copy_button.config(text=self.tr("Copy Summary", default="Copy Summary"))
                        if copy_button.winfo_exists()
                        else None
                    ),
                )
            except Exception as copy_e:
                print(f"Error copying summary: {copy_e}")

        copy_button = ttk.Button(
            button_frame,
            text=self.tr("Copy Summary", default="Copy Summary"),
            command=copy_action,
        )
        copy_button.pack(side=tk.RIGHT, padx=(5, 0))

        # 3. Bouton Open Output (juste à gauche de Copy Summary)
        open_button = ttk.Button(
            button_frame,
            text=self.tr("Open Output", default="Open Output"),
            command=self._open_output_folder,
            state=tk.NORMAL if can_open_output else tk.DISABLED,
        )
        open_button.pack(side=tk.RIGHT, padx=(5, 10))  # Plus grand padx pour séparer du groupe droit

        # 4. NOUVEAU BOUTON "OPEN UNALIGNED" (tout à gauche du groupe)
        if source_folders_with_unaligned_in_run and len(source_folders_with_unaligned_in_run) > 0:
            unaligned_target_path = os.path.join(input_folder_path_for_unaligned_button, "unaligned_by_stacker")

            open_unaligned_btn = ttk.Button(
                button_frame,
                text=self.tr("open_unaligned_button_text", default="Open Unaligned"),
                command=lambda p=unaligned_target_path: self._open_unaligned_folder_from_summary(p),
            )
            open_unaligned_btn.pack(side=tk.RIGHT, padx=(5, 10))  # Plus grand padx pour séparer du groupe droit
        # --- FIN NOUVEAU BOUTON ---

        # --- FIN DE LA CRÉATION DES WIDGETS ---

        # --- Rendre la fenêtre modale et attendre ---
        dialog.grab_set()  # Rendre modale APRÈS la création de TOUS les widgets
        dialog.update_idletasks()  # Assurer que la géométrie est calculée avant le centrage

        # Centrer dialogue
        root_x = self.root.winfo_x()
        root_y = self.root.winfo_y()
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()
        dialog_width = dialog.winfo_width()
        dialog_height = dialog.winfo_height()
        pos_x = root_x + (root_width // 2) - (dialog_width // 2)
        pos_y = root_y + (root_height // 2) - (dialog_height // 2)
        dialog.geometry(f"+{pos_x}+{pos_y}")

        self.root.wait_window(dialog)  # Attendre que la fenêtre modale soit fermée

    #########################################################################################################################################

    # --- DANS LA CLASSE SeestarStackerGUI DANS seestar/gui/main_window.py ---

    def _prepare_single_batch_if_needed(self) -> bool:
        """Check for ``stack_plan.csv`` when ``batch_size`` equals 1.

        Files from the CSV are queued in order and winsorized–sigma clipping is
        forced while drizzle and reprojection are disabled. The entire sequence
        is then stacked as one batch. Missing CSV falls back to multi-batch
        behaviour. CSV files can optionally include an index column (``1,file``)
        which will be ignored. Returns ``True`` when this special mode
        activates.

        """

        if getattr(self.settings, "batch_size", 0) != 1:
            return False

        csv_path = os.path.join(self.settings.input_folder, "stack_plan.csv")

        if not os.path.isfile(csv_path):
            self.logger.warning("Batch size 1 without CSV – aborting")
            self.settings.batch_size = 0
            self.settings.order_csv_path = ""
            self.settings.order_file_list = []
            raise FileNotFoundError(csv_path)

        self.logger.info(f"Stack plan CSV detected at '{csv_path}'. Preparing single batch")

        ordered_files: list[str] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            self.logger.warning("Stack plan CSV is empty")
            return False

        header = [c.strip().lower() for c in rows[0]]
        file_idx = None
        data_rows = rows

        if "file_path" in header:
            file_idx = header.index("file_path")
            data_rows = rows[1:]
        else:
            has_header = any(h in {"order", "file", "filename", "path", "index"} for h in header)
            if has_header:
                data_rows = rows[1:]

        for row in data_rows:
            if not row:
                continue
            if file_idx is not None:
                if len(row) <= file_idx:
                    continue
                cell = row[file_idx].strip()
            else:
                cell = row[0].strip()
                if cell.isdigit() and len(row) > 1:
                    cell = row[1].strip()

            if not cell or cell.lower() in {
                "order",
                "file",
                "filename",
                "path",
                "index",
                "file_path",
            }:
                continue

            if not os.path.isabs(cell):
                cell = os.path.join(self.settings.input_folder, cell)

            ordered_files.append(os.path.abspath(cell))

        missing = [p for p in ordered_files if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(missing[0])

        if getattr(self.settings, "stack_final_combine", "mean") != "mean":
            self.logger.info("stack_final_combine -> mean")
        setattr(self.settings, "stack_final_combine", "mean")
        if hasattr(self, "stack_final_combine_var"):
            try:
                self.stack_final_combine_var.set("mean")
                if hasattr(self, "stack_final_display_var") and hasattr(self, "final_key_to_label"):
                    label = self.final_key_to_label.get("mean", "mean")
                    self.stack_final_display_var.set(label)
            except Exception:
                pass

        batch_len = len(ordered_files)
        if batch_len <= 0:
            return False

        # keep ``batch_size`` at 1 so ``start_processing`` triggers its
        # special CSV mode which will enqueue ``ordered_files`` itself.
        self.settings.batch_size = 1
        self.settings.order_csv_path = csv_path
        self.settings.order_file_list = ordered_files
        self.logger.info("batch_size -> 1 (single batch via plan)")

        return True

    def _get_auto_chunk_size(self) -> int:
        """Return an automatic chunk size based on system RAM."""
        if not _psutil_available:
            return 50
        try:
            total_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            return 50
        if total_gb >= 64:
            return 100
        if total_gb >= 32:
            return 50
        if total_gb >= 16:
            return 25
        return 10

    def start_processing(self):
        """
        Démarre le traitement. Ordre crucial pour la gestion des paramètres :
        1. Valider chemins UI.
        2. Avertissement Drizzle/Mosaïque si nécessaire.
        3. Initialiser l'état de traitement du GUI (désactiver boutons, etc.).
        4. (A) Lire l'état actuel de l'UI vers self.settings (capture les modifs utilisateur).
        5. (B) Valider les settings dans self.settings (peut les corriger).
        6. (C) Si validation a corrigé des settings, ré-appliquer à l'UI pour que l'utilisateur voie les valeurs finales.
               Puis mettre à jour l'état des widgets dépendants (grisé/dégrisé).
        7. Préparer les arguments pour le backend en lisant depuis self.settings (maintenant la source de vérité).
        8. Lancer le thread de traitement du backend.
        MODIFIED: Ajout d'un CRITICAL CHECK pour vérifier mosaic_settings avant l'envoi au backend.
        """
        if self.settings.batch_size == 1:
            self.settings.enable_preview = False
        print("DEBUG (GUI start_processing): Début tentative démarrage du traitement...")

        if hasattr(self, "start_button"):
            try:
                self.start_button.config(state=tk.DISABLED)
            except tk.TclError:
                pass  # Ignorer si widget détruit

        # --- 1. Validation des chemins et de la présence de fichiers FITS ---
        print("DEBUG (GUI start_processing): Phase 1 - Validation des chemins...")
        input_folder = self.input_path.get()
        output_folder = self.output_path.get()

        if not input_folder or not output_folder:
            messagebox.showerror(self.tr("error"), self.tr("select_folders"))
            if hasattr(self, "start_button") and self.start_button.winfo_exists():
                self.start_button.config(state=tk.NORMAL)
            return
        if not os.path.isdir(input_folder):
            messagebox.showerror(self.tr("error"), f"{self.tr('input_folder_invalid')}:\n{input_folder}")
            if hasattr(self, "start_button") and self.start_button.winfo_exists():
                self.start_button.config(state=tk.NORMAL)
            return
        if not os.path.isdir(output_folder):
            try:
                os.makedirs(output_folder, exist_ok=True)
                self.update_progress_gui(f"{self.tr('Output folder created')}: {output_folder}", None)
            except Exception as e:
                messagebox.showerror(
                    self.tr("error"),
                    f"{self.tr('output_folder_invalid')}:\n{output_folder}\n{e}",
                )
                if hasattr(self, "start_button") and self.start_button.winfo_exists():
                    self.start_button.config(state=tk.NORMAL)
                return
        try:
            has_initial_fits = any(f.lower().endswith((".fit", ".fits")) for f in os.listdir(input_folder))
            has_additional_listed = bool(self.additional_folders_to_process)
            if not has_initial_fits and not has_additional_listed:
                if not messagebox.askyesno(self.tr("warning"), self.tr("no_fits_found")):
                    if hasattr(self, "start_button") and self.start_button.winfo_exists():
                        self.start_button.config(state=tk.NORMAL)
                    return
        except Exception as e:
            messagebox.showerror(self.tr("error"), f"{self.tr('Error reading input folder')}:\n{e}")
            if hasattr(self, "start_button") and self.start_button.winfo_exists():
                self.start_button.config(state=tk.NORMAL)
            return
        print("DEBUG (GUI start_processing): Phase 1 - Validation des chemins OK.")

        # --- 2. Avertissement Drizzle/Mosaïque (si activé) ---
        print("DEBUG (GUI start_processing): Phase 2 - Vérification avertissement Drizzle/Mosaïque...")
        drizzle_globally_enabled_ui = self.use_drizzle_var.get()
        # Lire mosaic_mode_active depuis self.settings, qui devrait avoir été mis à jour par MosaicSettingsWindow
        is_mosaic_mode_ui = getattr(self.settings, "mosaic_mode_active", False)

        if drizzle_globally_enabled_ui or is_mosaic_mode_ui:
            warning_title = self.tr("drizzle_warning_title")
            base_text_tuple_or_str = self.tr("drizzle_warning_text")
            base_warning_text = (
                "".join(base_text_tuple_or_str) if isinstance(base_text_tuple_or_str, tuple) else base_text_tuple_or_str
            )
            full_warning_text = base_warning_text
            if is_mosaic_mode_ui and not drizzle_globally_enabled_ui:
                full_warning_text += "\n\n" + self.tr(
                    "mosaic_requires_drizzle_note",
                    default="(Note: Mosaic mode requires Drizzle for final combination.)",
                )

            print(
                f"DEBUG (GUI start_processing): Avertissement Drizzle/Mosaïque nécessaire. is_mosaic_mode_ui={is_mosaic_mode_ui}"
            )
            continue_processing = messagebox.askyesno(warning_title, full_warning_text, parent=self.root)
            if not continue_processing:
                self.update_progress_gui("ⓘ Démarrage annulé par l'utilisateur après avertissement.", None)
                if hasattr(self, "start_button") and self.start_button.winfo_exists():
                    self.start_button.config(state=tk.NORMAL)
                return
        print("DEBUG (GUI start_processing): Phase 2 - Vérification avertissement OK (ou non applicable).")

        # --- Additional check: reproject modes require a configured local solver ---
        if self.reproject_between_batches_var.get() or getattr(self, "reproject_coadd_var", tk.BooleanVar()).get():
            use_solver = self.use_third_party_solver_var.get()
            solver_pref = getattr(self.settings, "local_solver_preference", "none")
            astap_path = getattr(self.settings, "astap_path", "").strip()
            ansvr_path = getattr(self.settings, "local_ansvr_path", "").strip()
            astrometry_dir = getattr(self.settings, "astrometry_solve_field_dir", "").strip()
            api_key = getattr(self.settings, "astrometry_api_key", "").strip()

            solver_configured = False
            if solver_pref == "astap":
                solver_configured = bool(astap_path)
            elif solver_pref == "ansvr":
                solver_configured = bool(ansvr_path)
            elif solver_pref == "astrometry":
                solver_configured = bool(astrometry_dir or api_key)
            else:
                solver_configured = any([astap_path, ansvr_path, astrometry_dir, api_key])

            if not (use_solver and solver_configured):
                messagebox.showerror(self.tr("error"), self.tr("reproject_solver_required_error"))
                if hasattr(self, "start_button") and self.start_button.winfo_exists():
                    self.start_button.config(state=tk.NORMAL)
                return

        # When the batch size equals 1, process the CSV using boring_stack.py
        if int(getattr(self.batch_size, "get", lambda: 0)()) == 1:
            # Sync settings now to read latest options
            if hasattr(self, "settings") and hasattr(self.settings, "update_from_ui"):
                try:
                    self.settings.update_from_ui(self)
                except Exception:
                    pass

            csv_path = os.path.join(self.settings.input_folder, "stack_plan.csv")
            if not os.path.isfile(csv_path):
                messagebox.showerror(
                    self.tr("error"),
                    self.tr(
                        "stack_plan_missing_file_error",
                        default="File listed in stack_plan.csv not found:\n{path}",
                    ).format(path=csv_path),
                )
                if hasattr(self, "start_button") and self.start_button.winfo_exists():
                    self.start_button.config(state=tk.NORMAL)
                return

            script = os.path.join(os.path.dirname(__file__), "boring_stack.py")
            cmd = [
                sys.executable,
                script,
                "--csv",
                csv_path,
                "--out",
                self.settings.output_folder,
                "--batch-size",
                "1",
                "--max-mem",
                str(getattr(self.settings, "max_hq_mem_gb", 8)),
                "--chunk-size",
                str(self._get_auto_chunk_size()),
                "--log-dir",
                os.path.join(self.settings.output_folder, "logs"),
            ]
            # Propagate normalization and final FITS dtype to boring_stack
            try:
                norm_method = getattr(self.settings, "stack_norm_method", "none")
            except Exception:
                norm_method = "none"
            cmd += ["--norm", str(norm_method)]
            save_f32 = bool(getattr(self.settings, "save_final_as_float32", False))
            cmd.append("--save-as-float32" if save_f32 else "--no-save-as-float32")
            final_combine_slug = _to_slug(self.stack_final_combine_var.get())
            cmd += ["--final-combine", final_combine_slug]
            self.logger.info(
                "Launching boring_stack with final_combine=%s, batch_size=1",
                final_combine_slug,
            )
            self._run_boring_stack_process(cmd, csv_path, self.settings.output_folder)
            return

        # --- 3. Initialisation de l'état de traitement du GUI ---
        print("DEBUG (GUI start_processing): Phase 3 - Initialisation état de traitement GUI...")
        self.processing = True
        self.initial_auto_stretch_done = False
        self.time_per_image = 0
        self.global_start_time = time.monotonic()
        self.batches_processed_for_preview_refresh = 0
        default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
        self.aligned_files_var.set(default_aligned_fmt.format(count=0))
        folders_to_pass_to_backend = list(self.additional_folders_to_process)
        self.additional_folders_to_process = []
        self.update_additional_folders_display()
        self._set_parameter_widgets_state(tk.DISABLED)
        if hasattr(self, "stop_button") and self.stop_button.winfo_exists():
            self.stop_button.config(state=tk.NORMAL)
        if hasattr(self, "open_output_button") and self.open_output_button.winfo_exists():
            self.open_output_button.config(state=tk.DISABLED)
        if hasattr(self, "progress_manager"):
            self.progress_manager.reset()
            self.progress_manager.start_timer()
        if hasattr(self, "status_text") and self.status_text.winfo_exists():
            self.status_text.config(state=tk.NORMAL)
            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(tk.END, f"--- {self.tr('stacking_start')} ---\n")
            self.status_text.config(state=tk.DISABLED)
        print("DEBUG (GUI start_processing): Phase 3 - Initialisation état de traitement GUI OK.")

        # --- 4. Synchronisation et Validation des Settings ---
        print("DEBUG (GUI start_processing): Phase 4 - Synchronisation et validation des Settings...")
        print("  -> (4A) Appel self.settings.update_from_ui(self)...")
        self.settings.update_from_ui(self)
        # ... (logs de vérification des settings après update_from_ui) ...
        print(
            f"  DEBUG GUI SETTINGS (Phase 4A): self.settings.mosaic_settings['alignment_mode'] = {self.settings.mosaic_settings.get('alignment_mode', 'NonTrouve')}"
        )

        print("  -> (4B) Appel self.settings.validate_settings()...")
        validation_messages = self.settings.validate_settings()
        # ... (logs de vérification des settings après validate_settings) ...
        print(
            f"  DEBUG GUI SETTINGS (Phase 4B): self.settings.mosaic_settings['alignment_mode'] = {self.settings.mosaic_settings.get('alignment_mode', 'NonTrouve')}"
        )

        if validation_messages:
            self.update_progress_gui("⚠️ Paramètres ajustés après validation:", None)
            for msg in validation_messages:
                self.update_progress_gui(f"  - {msg}", None)
            print("  -> (4C) Ré-appel self.settings.apply_to_ui(self)...")
            self.settings.apply_to_ui(self)
            self._update_drizzle_options_state()
            self._update_final_scnr_options_state()
            self._update_photutils_bn_options_state()
            self._update_feathering_options_state()
            self._update_low_wht_mask_options_state()  # S'assurer d'appeler ceci aussi
        print("DEBUG (GUI start_processing): Phase 4 - Settings synchronisés et validés.")

        # Heavy work is delegated to a starter thread to keep Tk responsive.

        def _starter():
            try:
                self.settings.update_from_ui(self)
                validation_messages = self.settings.validate_settings()
                special_single = self._prepare_single_batch_if_needed()
                try:
                    self.queued_stacker.align_on_disk = int(self.settings.batch_size) >= 1
                except Exception:
                    self.queued_stacker.align_on_disk = False
                if validation_messages:
                    def _apply_valid():
                        self.update_progress_gui("⚠️ Paramètres ajustés après validation:", None)
                        for msg in validation_messages:
                            self.update_progress_gui(f"  - {msg}", None)
                        self.settings.apply_to_ui(self)
                        self._update_drizzle_options_state()
                        self._update_final_scnr_options_state()
                        self._update_photutils_bn_options_state()
                        self._update_feathering_options_state()
                        self._update_low_wht_mask_options_state()
                    if hasattr(self, "gui_event_queue"):
                        self.gui_event_queue.put(_apply_valid)
                    else:
                        self.root.after(0, _apply_valid)
                backend_kwargs = {
                    "input_dir": self.settings.input_folder,
                    "output_dir": self.settings.output_folder,
                    "temp_folder": self.settings.temp_folder,
                    "output_filename": self.settings.output_filename,
                    "reference_path_ui": self.settings.reference_image_path,
                    "initial_additional_folders": folders_to_pass_to_backend,
                    "stacking_mode": self.settings.stacking_mode,
                    "kappa": self.settings.kappa,
                    "stack_kappa_low": self.settings.stack_kappa_low,
                    "stack_kappa_high": self.settings.stack_kappa_high,
                    "winsor_limits": (
                        tuple(float(x.strip()) for x in str(self.settings.stack_winsor_limits).split(","))
                        if isinstance(self.settings.stack_winsor_limits, str)
                        else (0.05, 0.05)
                    ),
                    "normalize_method": self.settings.stack_norm_method,
                    "weighting_method": self.settings.stack_weight_method,
                    "batch_size": self.settings.batch_size,
                    "ordered_files": getattr(self.settings, "order_file_list", None),
                    "correct_hot_pixels": self.settings.correct_hot_pixels,
                    "hot_pixel_threshold": self.settings.hot_pixel_threshold,
                    "neighborhood_size": self.settings.neighborhood_size,
                    "bayer_pattern": self.settings.bayer_pattern,
                    "perform_cleanup": self.settings.cleanup_temp,
                    "use_weighting": self.settings.stack_weight_method != "none",
                    "weight_by_snr": self.settings.weight_by_snr,
                    "weight_by_stars": self.settings.weight_by_stars,
                    "snr_exp": self.settings.snr_exponent,
                    "stars_exp": self.settings.stars_exponent,
                    "min_w": self.settings.min_weight,
                    "use_drizzle": self.settings.use_drizzle,
                    "drizzle_scale": float(self.settings.drizzle_scale),
                    "drizzle_wht_threshold": self.settings.drizzle_wht_threshold,
                    "drizzle_mode": self.settings.drizzle_mode,
                    "drizzle_kernel": self.settings.drizzle_kernel,
                    "drizzle_pixfrac": self.settings.drizzle_pixfrac,
                    "apply_chroma_correction": self.settings.apply_chroma_correction,
                    "apply_final_scnr": self.settings.apply_final_scnr,
                    "final_scnr_target_channel": self.settings.final_scnr_target_channel,
                    "final_scnr_amount": self.settings.final_scnr_amount,
                    "final_scnr_preserve_luminosity": self.settings.final_scnr_preserve_luminosity,
                    "bn_grid_size_str": self.settings.bn_grid_size_str,
                    "bn_perc_low": self.settings.bn_perc_low,
                    "bn_perc_high": self.settings.bn_perc_high,
                    "bn_std_factor": self.settings.bn_std_factor,
                    "bn_min_gain": self.settings.bn_min_gain,
                    "bn_max_gain": self.settings.bn_max_gain,
                    "cb_border_size": self.settings.cb_border_size,
                    "cb_blur_radius": self.settings.cb_blur_radius,
                    "cb_min_b_factor": self.settings.cb_min_b_factor,
                    "cb_max_b_factor": self.settings.cb_max_b_factor,
                    "apply_master_tile_crop": self.settings.apply_master_tile_crop,
                    "master_tile_crop_percent": self.settings.master_tile_crop_percent,
                    "final_edge_crop_percent": self.settings.final_edge_crop_percent,
                    "apply_photutils_bn": self.settings.apply_photutils_bn,
                    "photutils_bn_box_size": self.settings.photutils_bn_box_size,
                    "photutils_bn_filter_size": self.settings.photutils_bn_filter_size,
                    "photutils_bn_sigma_clip": self.settings.photutils_bn_sigma_clip,
                    "photutils_bn_exclude_percentile": self.settings.photutils_bn_exclude_percentile,
                    "apply_feathering": self.settings.apply_feathering,
                    "feather_blur_px": self.settings.feather_blur_px,
                    "apply_batch_feathering": self.settings.apply_batch_feathering,
                    "apply_low_wht_mask": self.settings.apply_low_wht_mask,
                    "low_wht_percentile": self.settings.low_wht_percentile,
                    "low_wht_soften_px": self.settings.low_wht_soften_px,
                    "is_mosaic_run": self.settings.mosaic_mode_active,
                    "api_key": self.settings.astrometry_api_key,
                    "mosaic_settings": self.settings.mosaic_settings,
                    "astap_path": self.settings.astap_path,
                    "astap_data_dir": self.settings.astap_data_dir,
                    "local_ansvr_path": self.settings.local_ansvr_path,
                    "local_solver_preference": self.settings.local_solver_preference,
                    "astap_search_radius": self.settings.astap_search_radius,
                    "astap_downsample": self.settings.astap_downsample,
                    "astap_sensitivity": self.settings.astap_sensitivity,
                    "save_as_float32": self.settings.save_final_as_float32,
                    "preserve_linear_output": self.settings.preserve_linear_output,
                    "reproject_between_batches": self.settings.reproject_between_batches,
                    "reproject_coadd_final": self.settings.reproject_coadd_final,
                }

                if self.settings.batch_size == 1 and not special_single:
                    backend_kwargs["chunk_size"] = self._get_auto_chunk_size()

                try:
                    started = self.queued_stacker.start_processing(**backend_kwargs)
                except Exception as e:
                    started = False
                    print(f"Backend start failed: {e}")

                def _after_start(started=started, special_single=special_single):
                    if special_single:
                        self.batch_size.set(self.settings.batch_size)
                        self.stacking_mode.set(self.settings.stacking_mode)
                        self.reproject_between_batches_var.set(self.settings.reproject_between_batches)
                        self.use_drizzle_var.set(self.settings.use_drizzle)
                        if hasattr(self, "stack_final_combine_var"):
                            try:
                                self.stack_final_combine_var.set(self.settings.stack_final_combine)
                                if hasattr(self, "stack_final_display_var") and hasattr(self, "final_key_to_label"):
                                    label = self.final_key_to_label.get(self.settings.stack_final_combine, self.settings.stack_final_combine)
                                    self.stack_final_display_var.set(label)
                            except Exception:
                                pass

                    self._final_stretch_set_by_processing_finished = False
                    if started:
                        if hasattr(self, "stop_button") and self.stop_button.winfo_exists():
                            self.stop_button.config(state=tk.NORMAL)
                        self.thread = threading.Thread(
                            target=self._track_processing_progress,
                            daemon=True,
                            name="GUI_ProgressTracker",
                        )
                        self.thread.start()
                    else:
                        if hasattr(self, "start_button") and self.start_button.winfo_exists():
                            self.start_button.config(state=tk.NORMAL)
                        self.processing = False
                        self.update_progress_gui(
                            "ⓘ Échec démarrage traitement (le backend a refusé ou erreur critique). Vérifiez logs console.",
                            None,
                        )
                        self._set_parameter_widgets_state(tk.NORMAL)

                if hasattr(self, "gui_event_queue"):
                    self.gui_event_queue.put(_after_start)
                else:
                    self.root.after(0, _after_start)

            except FileNotFoundError as fnfe:
                def _prep_error_inner(fnfe=fnfe):
                    messagebox.showerror(
                        self.tr("error"),
                        self.tr(
                            "stack_plan_missing_file_error",
                            default="File listed in stack_plan.csv not found:\n{path}",
                        ).format(path=str(fnfe)),
                    )
                    if hasattr(self, "start_button") and self.start_button.winfo_exists():
                        self.start_button.config(state=tk.NORMAL)
                    if hasattr(self, "stop_button") and self.stop_button.winfo_exists():
                        self.stop_button.config(state=tk.DISABLED)
                    self.processing = False
                    self._set_parameter_widgets_state(tk.NORMAL)

                if hasattr(self, "gui_event_queue"):
                    self.gui_event_queue.put(_prep_error_inner)
                else:
                    self.root.after(0, _prep_error_inner)

        threading.Thread(target=_starter, daemon=True, name="BackendStarter").start()


##############################################################################################################################################


# ----Fin du Fichier main_window.py
