# --- START OF FILE seestar/gui/main_window.py (Part 1/3) ---
"""
Module principal pour l'interface graphique de GSeestar.
Intègre la prévisualisation avancée et le traitement en file d'attente via QueueManager.
(Version Révisée: Ajout dossiers avant start, Log amélioré, Bouton Ouvrir Sortie)
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, font as tkFont
import threading
import time
import numpy as np
from astropy.io import fits
import traceback
import math
import platform # NOUVEL import
import subprocess # NOUVEL import
import gc #
from PIL import Image, ImageTk 
from .ui_utils import ToolTip
# --- NOUVEAUX IMPORTS SPÉCIFIQUES POUR LE LANCEUR ---
import sys # Pour sys.executable
# ----------------------------------------------------
import tempfile # <-- AJOUTÉ
print("-" * 20)
print("DEBUG MW: Tentative d'importation de SeestarQueuedStacker...")
try:
    # L'import que vous avez déjà
    from ..queuep.queue_manager import SeestarQueuedStacker
    print(f"DEBUG MW: Import de 'SeestarQueuedStacker' réussi.")
    print(f"DEBUG MW: Type de l'objet importé: {type(SeestarQueuedStacker)}")
    # Vérifier si l'attribut existe sur la CLASSE importée
    print(f"DEBUG MW: La CLASSE importée a 'set_progress_callback'? {hasattr(SeestarQueuedStacker, 'set_progress_callback')}")
    print(f"DEBUG MW: Attributs de la CLASSE importée: {dir(SeestarQueuedStacker)}")
except ImportError as imp_err:
    print(f"ERREUR MW: ÉCHEC de l'import de SeestarQueuedStacker: {imp_err}")
    traceback.print_exc()
    # Si l'import échoue, l'application ne peut pas continuer
    sys.exit("Échec de l'importation critique.")
except Exception as gen_err:
    print(f"ERREUR MW: Erreur INATTENDUE pendant l'import de SeestarQueuedStacker: {gen_err}")
    traceback.print_exc()
    sys.exit("Échec de l'importation critique.")
print("-" * 20)
# --- FIN DU BLOC DE DEBUG ---
# Seestar imports
from ..core.image_processing import load_and_validate_fits, debayer_image
from ..localization import Localization
from .local_solver_gui import LocalSolverSettingsWindow # 
from ..core.utils import estimate_batch_size
from ..enhancement.color_correction import ChromaticBalancer 
# (Ajouter ceci avec les autres imports de gui)
from .mosaic_gui import MosaicSettingsWindow # Importer la future classe
try:
    # Import tools for preview adjustments and auto calculations
    from ..tools.stretch import StretchPresets, ColorCorrection
    from ..tools.stretch import apply_auto_stretch as calculate_auto_stretch
    from ..tools.stretch import apply_auto_white_balance as calculate_auto_wb
    _tools_available = True
except ImportError as tool_err:
    print(f"Warning: Could not import stretch/color tools: {tool_err}.")
    _tools_available = False
    # Dummy implementations if tools are missing
    class StretchPresets:
        @staticmethod
        def linear(data, bp=0., wp=1.): wp=max(wp,bp+1e-6); return np.clip((data-bp)/(wp-bp), 0, 1)
        @staticmethod
        def asinh(data, scale=1., bp=0.): data_s=data-bp; data_c=np.maximum(data_s,0.); max_v=np.nanmax(data_c); den=np.arcsinh(scale*max_v); return np.arcsinh(scale*data_c)/den if den>1e-6 else np.zeros_like(data)
        @staticmethod
        def logarithmic(data, scale=1., bp=0.): data_s=data-bp; data_c=np.maximum(data_s,1e-10); max_v=np.nanmax(data_c); den=np.log1p(scale*max_v); return np.log1p(scale*data_c)/den if den>1e-6 else np.zeros_like(data)
        @staticmethod
        def gamma(data, gamma=1.0): return np.power(np.maximum(data, 1e-10), gamma)
    class ColorCorrection:
        @staticmethod
        def white_balance(data, r=1., g=1., b=1.):
            if data is None or data.ndim != 3: return data
            corr=data.astype(np.float32).copy(); corr[...,0]*=r; corr[...,1]*=g; corr[...,2]*=b; return np.clip(corr,0,1)
    def calculate_auto_stretch(*args, **kwargs): return (0.0, 1.0)
    def calculate_auto_wb(*args, **kwargs): return (1.0, 1.0, 1.0)


# GUI Component Imports
from .file_handling import FileHandlingManager
from .preview import PreviewManager
from .progress import ProgressManager
from .settings import SettingsManager
from .histogram_widget import HistogramWidget

class SeestarStackerGUI:
    """ GUI principale pour Seestar. """
    # --- MODIFIÉ: Signature du constructeur ---

    def __init__(self, initial_input_dir=None, stack_immediately_from=None): # <-- AJOUTÉ stack_immediately_from
        """Initialise l'interface graphique."""
        print("DEBUG (GUI __init__): Initialisation SeestarStackerGUI...") # <-- AJOUTÉ DEBUG
        print(f"DEBUG (GUI __init__): Reçu initial_input_dir='{initial_input_dir}', stack_immediately_from='{stack_immediately_from}'") # <-- AJOUTÉ DEBUG
        self.root = tk.Tk()

        # ... (Logique de l'icône inchangée) ...
        try:
            icon_path = 'icon/icon.png'
            if os.path.exists(icon_path):
                icon_image = Image.open(icon_path); self.tk_icon = ImageTk.PhotoImage(icon_image); self.root.iconphoto(True, self.tk_icon)
                print(f"DEBUG (GUI __init__): Icone chargée depuis: {icon_path}") # <-- AJOUTÉ DEBUG (plus détaillé)
            else: print(f"Warning: Icon file not found at: {icon_path}. Using default icon.")
        except Exception as e: print(f"Error loading or setting window icon: {e}")
        
        self.astrometry_api_key_var = tk.StringVar()
        # --- Initialisation des variables et objets internes ---
        # (Identique à avant, mais ajout d'un flag pour le stack immédiat)
        self.localization = Localization("en")
        self.settings = SettingsManager()
        self.queued_stacker = SeestarQueuedStacker()
        self.processing = False
        self.thread = None
        self.current_preview_data = None
        self.current_stack_header = None
        self.debounce_timer_id = None
        self.time_per_image = 0
        self.global_start_time = None
        self.additional_folders_to_process = []
        self.tooltips = {}
        print("DEBUG (GUI __init__): Dictionnaire self.tooltips initialisé.")
        ### Compteur pour Auto-Refresh Aperçu ###
        self.batches_processed_for_preview_refresh = 0
        self.preview_auto_refresh_batch_interval = 10 # Rafraîchir tous les 10 lots
        # --- Variables état mosaïque ---
        self.mosaic_mode_active = False
        # self.mosaic_panel_folders = [] # Sera utilisé plus tard
        # self.mosaic_settings = {}    # Sera utilisé plus tard
        print("DEBUG (GUI __init__): Flag self.mosaic_mode_active initialisé à False.")
        self.mosaic_settings = {}    # <<<--- AJOUTER CETTE LIGNE pour initialiser le dictionnaire
        print("DEBUG (GUI __init__): Flag self.mosaic_mode_active et dict self.mosaic_settings initialisés.")

        # --- NOUVEAU FLAG ---
        self._trigger_immediate_stack = False # Sera True si stack_immediately_from est valide
        self._folder_for_immediate_stack = None # Stockera le chemin
        # --- FIN NOUVEAU ---

        # --- Variables Pondération (Inchangé) ---
        self.use_weighting_var = tk.BooleanVar(value=False)
        self.weight_snr_var = tk.BooleanVar(value=True); self.weight_stars_var = tk.BooleanVar(value=True)
        self.snr_exponent_var = tk.DoubleVar(value=1.0); self.stars_exponent_var = tk.DoubleVar(value=0.5)
        self.min_weight_var = tk.DoubleVar(value=0.1)

        # --- Initialisation Variables Tkinter ---
        self.init_variables() # Doit être avant le chargement/application des settings

        # --- Chargement Settings & Langue ---
        self.settings.load_settings()
        # ---  Forcer la désactivation du mode mosaïque au démarrage ---
        #if hasattr(self.settings, 'mosaic_mode_active'):
        #    print(f"DEBUG (GUI __init__): Valeur self.settings.mosaic_mode_active APRES load: {self.settings.mosaic_mode_active}")
        #    self.settings.mosaic_mode_active = False # Remettre à False pour cette session
        #    print(f"DEBUG (GUI __init__): self.settings.mosaic_mode_active FORCÉ à False pour le démarrage de l'UI.")
        
        self.language_var.set(self.settings.language)
        self.localization.set_language(self.settings.language)
        print(f"DEBUG (GUI __init__): Settings chargés, langue définie sur '{self.settings.language}'.") # <-- AJOUTÉ DEBUG

        # --- AJOUT DEBUG SPÉCIFIQUE POUR LA CLÉ API ---
        print(f"DEBUG (GUI __init__): Valeur de self.settings.astrometry_api_key APRES load_settings: '{self.settings.astrometry_api_key}' (longueur: {len(self.settings.astrometry_api_key)})")
        # --- FIN AJOUT ---


        # --- Gestion des arguments d'entrée (MODIFIÉ) ---
        # Priorité 1: Stacking immédiat demandé par l'analyseur
        if stack_immediately_from and isinstance(stack_immediately_from, str) and os.path.isdir(stack_immediately_from):
             print(f"INFO (GUI __init__): Stacking immédiat demandé pour: {stack_immediately_from}") # <-- AJOUTÉ INFO
             # Surcharger le dossier d'entrée avec celui de l'analyseur
             self.input_path.set(stack_immediately_from)
             # --- NOUVEAU: Marquer pour déclencher le stack ---
             self._folder_for_immediate_stack = stack_immediately_from
             self._trigger_immediate_stack = True
             print(f"DEBUG (GUI __init__): Flag _trigger_immediate_stack mis à True.") # <-- AJOUTÉ DEBUG
             # Optionnel: Mettre aussi à jour le setting (écrase valeur chargée)
             # self.settings.input_folder = stack_immediately_from
        # Priorité 2: Pré-remplissage simple demandé via --input-dir
        elif initial_input_dir and isinstance(initial_input_dir, str) and os.path.isdir(initial_input_dir):
             print(f"INFO (GUI __init__): Pré-remplissage dossier entrée depuis argument: {initial_input_dir}") # <-- AJOUTÉ INFO
             self.input_path.set(initial_input_dir)
             # self.settings.input_folder = initial_input_dir # Optionnel

        # --- Création Layout et Initialisation Managers (Inchangé) ---
        self.file_handler = FileHandlingManager(self) # Doit être avant create_layout si utilisé dedans
        self.create_layout()
        self.init_managers()
        print("DEBUG (GUI __init__): Layout créé, managers initialisés.") # <-- AJOUTÉ DEBUG

        # --- Application Settings & UI Updates (Inchangé) ---
        self.settings.apply_to_ui(self)
        # --- DEBUG SPÉCIFIQUE POUR LA CLÉ API ---
        try:
            api_key_val_after_apply = self.astrometry_api_key_var.get()
            print(f"DEBUG (GUI __init__): Valeur de self.astrometry_api_key_var APRES apply_to_ui: '{api_key_val_after_apply}' (longueur: {len(api_key_val_after_apply)})")
        except Exception as e_get_var:
            print(f"DEBUG (GUI __init__): Erreur lecture self.astrometry_api_key_var après apply_to_ui: {e_get_var}")
        # --- FIN AJOUT ---
        if hasattr(self, '_update_spinbox_from_float'): self._update_spinbox_from_float()
        self._update_weighting_options_state()
        self._update_drizzle_options_state() # S'assurer que les options drizzle sont à jour
        self._update_show_folders_button_state()
        self.update_ui_language()

        # --- BLOC DE DÉBOGAGE AVANT APPEL SET_PREVIEW_CALLBACK ---
        print("--------------------")
        print("DEBUG MW __init__: Vérification de self.queued_stacker.set_preview_callback AVANT appel...")
        if hasattr(self.queued_stacker, 'set_preview_callback') and callable(self.queued_stacker.set_preview_callback):
            import inspect
            try:
                source_lines, start_line = inspect.getsourcelines(self.queued_stacker.set_preview_callback)
                print(f"  Source de self.queued_stacker.set_preview_callback (ligne de début: {start_line}):")
                for i, line_content in enumerate(source_lines[:10]): # Afficher les 10 premières lignes
                    print(f"    L{start_line + i}: {line_content.rstrip()}")
                
                source_code_str = "".join(source_lines)
                if "_cleanup_mosaic_panel_stacks_temp()" in source_code_str or \
                   "_cleanup_drizzle_batch_outputs()" in source_code_str or \
                   "cleanup_unaligned_files()" in source_code_str:
                    print("  ALERTE MW DEBUG: Un appel _cleanup_ SEMBLE ÊTRE PRÉSENT dans le code source de la méthode set_preview_callback attachée à l'instance !")
                else:
                    print("  INFO MW DEBUG: Aucun appel _cleanup_ évident dans le code source de la méthode set_preview_callback attachée à l'instance.")

            except TypeError:
                print("  ERREUR MW DEBUG: Impossible d'obtenir la source pour une méthode built-in ou C (ne devrait pas être le cas ici).")
            except IOError:
                print("  ERREUR MW DEBUG: Impossible de lire le fichier source (très étrange).")
            except Exception as e_inspect:
                print(f"  ERREUR MW DEBUG: Erreur inspect: {e_inspect}")
        else:
            print("  ERREUR MW DEBUG: self.queued_stacker n'a pas de méthode set_preview_callback ou elle n'est pas callable.")
        print("--------------------")
        # --- FIN BLOC DE DÉBOGAGE ---
        
        # --- Connexion Callbacks Backend (Inchangé) ---
        self.queued_stacker.set_progress_callback(self.update_progress_gui)
        self.queued_stacker.set_preview_callback(self.update_preview_from_stacker)
        print("DEBUG (GUI __init__): Callbacks backend connectés.") # <-- AJOUTÉ DEBUG

        # --- Configuration Fenêtre Finale (Inchangé) ---
        self.root.title(self.tr("title"))
        try: self.root.geometry(self.settings.window_geometry)
        except tk.TclError: self.root.geometry("1200x750")
        self.root.minsize(1100, 650)
        self.root.bind("<Configure>", self._debounce_resize)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # --- Variable onglets expert ---
        self._update_final_scnr_options_state()
        self._update_photutils_bn_options_state()
        self._update_feathering_options_state()
        # --- Variables état aperçu (Inchangé) ---
        self.preview_img_count = 0; self.preview_total_imgs = 0
        self.preview_current_batch = 0; self.preview_total_batches = 0

        # --- Aperçu Initial & Dossiers Additionnels (Inchangé) ---
        self._try_show_first_input_image()
        self.update_additional_folders_display()

        # --- NOUVEAU: Déclenchement du stacking immédiat si demandé ---
        if self._trigger_immediate_stack:
             print("DEBUG (GUI __init__): Planification du lancement immédiat via after(500, ...).") # <-- AJOUTÉ DEBUG
             # Utiliser 'after' pour lancer le stacking après que l'UI soit complètement chargée
             # Un délai court (ex: 500ms) peut aider à assurer que tout est prêt
             self.root.after(500, self._start_immediate_stack)
        else:
             print("DEBUG (GUI __init__): Pas de stacking immédiat demandé.") # <-- AJOUTÉ DEBUG

        print("DEBUG (GUI __init__): Initialisation terminée.") # <-- AJOUTÉ DEBUG


# --- DANS LA CLASSE SeestarStackerGUI ---
# (Ajoutez cette méthode, par exemple après __init__ ou près de start_processing)

    def _start_immediate_stack(self):
        """
        Méthode appelée via 'after' pour démarrer le stacking automatiquement
        si demandé par l'analyseur.
        """
        print("DEBUG (GUI): Exécution de _start_immediate_stack().") # <-- AJOUTÉ DEBUG
        # Double vérification que le flag est bien positionné et qu'un dossier est défini
        if self._trigger_immediate_stack and self._folder_for_immediate_stack:
            print(f"DEBUG (GUI): Conditions remplies. Tentative de lancement de start_processing pour: {self._folder_for_immediate_stack}") # <-- AJOUTÉ DEBUG
            # Assurer que le dossier d'entrée dans l'UI correspond bien
            # (Normalement déjà fait dans __init__, mais sécurité supplémentaire)
            current_ui_input = self.input_path.get()
            if os.path.normpath(current_ui_input) != os.path.normpath(self._folder_for_immediate_stack):
                print(f"AVERTISSEMENT (GUI): Dossier UI ({current_ui_input}) ne correspond pas au dossier demandé ({self._folder_for_immediate_stack}). Mise à jour UI.")
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
            print("DEBUG (GUI): _start_immediate_stack() appelé mais conditions non remplies (flag ou dossier manquant).") # <-- AJOUTÉ DEBUG

        # Réinitialiser le flag pour éviter déclenchement multiple
        self._trigger_immediate_stack = False
        self._folder_for_immediate_stack = None


    def init_variables(self):
        """Initialise les variables Tkinter."""
        print("DEBUG (GUI init_variables): Initialisation des variables Tkinter...")
        # ... (autres variables input_path, output_path, etc. inchangées) ...
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.reference_image_path = tk.StringVar()
        self.stacking_mode = tk.StringVar(value="kappa-sigma") # Default value
        self.kappa = tk.DoubleVar(value=2.5)
        self.batch_size = tk.IntVar(value=10) # Default batch size (0 was ambiguous)
        self.correct_hot_pixels = tk.BooleanVar(value=True)
        self.hot_pixel_threshold = tk.DoubleVar(value=3.0)
        self.neighborhood_size = tk.IntVar(value=5)
        self.cleanup_temp_var = tk.BooleanVar(value=True) # Renamed from remove_aligned

        # --- Drizzle Variables --- #
        self.use_drizzle_var = tk.BooleanVar(value=False)
        self.drizzle_scale_var = tk.StringVar(value="2")
        self.drizzle_wht_threshold_var = tk.DoubleVar(value=0.7) # La variable 0.0-1.0 (INCHANGÉE)
        self.drizzle_wht_display_var = tk.StringVar(value="70") # <-- AJOUTÉ: Variable pour l'affichage % (String pour .set())
        self.drizzle_mode_var = tk.StringVar(value="Final")
        self.drizzle_kernel_var = tk.StringVar(value="square")
        self.drizzle_pixfrac_var = tk.DoubleVar(value=1.0)
        # --- FIN MODIFICATION ---

        # ... (variables Preview inchangées) ...
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

        # ... (variables UI State inchangées) ...
        self.language_var = tk.StringVar(value='en')
        self.remaining_files_var = tk.StringVar(value=self.tr("no_files_waiting", default="No files waiting"))
        self.additional_folders_var = tk.StringVar(value=self.tr("no_additional_folders", default="None"))
        default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
        self.aligned_files_var = tk.StringVar(value=default_aligned_fmt.format(count="--"))
        self.remaining_time_var = tk.StringVar(value="--:--:--") # ETA
        self.elapsed_time_var = tk.StringVar(value="00:00:00")
        self._after_id_resize = None
        # La ligne suivante n'est plus nécessaire car le trace a été enlevé, on peut la supprimer si elle existe
        # self._trace_id_wht = None # <-- SUPPRIMER SI PRÉSENT

        # ---  VARIABLE pour la correction chroma ---
        # Default value is True, can be changed later based on user preference or settings load
        self.apply_chroma_correction_var = tk.BooleanVar(value=True)
        print("DEBUG (GUI init_variables): Variable apply_chroma_correction_var créée.") 
        
        ### NOUVEAU : Variables Tkinter pour SCNR Final ###
        self.apply_final_scnr_var = tk.BooleanVar(value=False) # SCNR désactivé par défaut
        self.final_scnr_amount_var = tk.DoubleVar(value=0.8)  # Intensité par défaut
        self.final_scnr_preserve_lum_var = tk.BooleanVar(value=True) # Préserver luminance par défaut
        # Pas de variable pour target_channel pour l'instant, on le fixe à 'green'
        print("DEBUG (GUI init_variables): Variables SCNR Final créées.")
                
        # Neutralisation du Fond (BN)
        self.bn_grid_size_str_var = tk.StringVar(value="16x16") # Ex: "8x8", "16x16", "32x32"
        self.bn_perc_low_var = tk.IntVar(value=5)
        self.bn_perc_high_var = tk.IntVar(value=30)
        self.bn_std_factor_var = tk.DoubleVar(value=1.0)
        self.bn_min_gain_var = tk.DoubleVar(value=0.2)
        self.bn_max_gain_var = tk.DoubleVar(value=7.0)

        # ChromaticBalancer (CB)
        self.cb_border_size_var = tk.IntVar(value=25)
        self.cb_blur_radius_var = tk.IntVar(value=8)
        self.cb_min_b_factor_var = tk.DoubleVar(value=0.4) # Pour le facteur B de CB
        self.cb_max_b_factor_var = tk.DoubleVar(value=1.5) # Pour le facteur B de CB
        # self.cb_intensity_var = tk.DoubleVar(value=1.0) # Optionnel, pour moduler la correction

        # Rognage Final
        self.final_edge_crop_percent_var = tk.DoubleVar(value=2.0) # En %, ex: 2.0 pour 2%

        print("DEBUG (GUI init_variables): Variables Onglet Expert (BN, CB, Crop) créées.")

        ###  Variables Tkinter pour Soustraction Fond 2D (Photutils) ###
        self.apply_photutils_bn_var = tk.BooleanVar(value=False) # Désactivé par défaut
        self.photutils_bn_box_size_var = tk.IntVar(value=128)     # Défaut: 128
        self.photutils_bn_filter_size_var = tk.IntVar(value=5)    # Défaut: 5 (doit être impair)
        self.photutils_bn_sigma_clip_var = tk.DoubleVar(value=3.0) # Défaut: 3.0
        self.photutils_bn_exclude_percentile_var = tk.DoubleVar(value=98.0) # Défaut: 98.0

        # --- NOUVELLES VARIABLES TKINTER POUR FEATHERING ---
        self.apply_feathering_var = tk.BooleanVar(value=False)  # Désactivé par défaut
        self.feather_blur_px_var = tk.IntVar(value=256)      # Valeur de flou par défaut
        print("DEBUG (GUI init_variables): Variables Feathering créées (apply_feathering_var, feather_blur_px_var).")
        # --- FIN NOUVELLES VARIABLES ---
        print("DEBUG (GUI init_variables): Variables pour Photutils Background Subtraction créées.")
        
        # --- VARIABLES POUR LOW WHT MASK ---
        self.apply_low_wht_mask_var = tk.BooleanVar(value=False) # Désactivé par défaut
        self.low_wht_pct_var = tk.IntVar(value=5)                # Valeur par défaut pour le percentile
        self.low_wht_soften_px_var = tk.IntVar(value=128)        # Valeur par défaut pour le rayon de flou
        print("DEBUG (GUI init_variables): Variables Low WHT Mask créées.")
        # --- FIN NOUVELLES VARIABLES ---
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
                getattr(self, 'drizzle_mode_label', None),
                getattr(self, 'drizzle_radio_final', None),
                getattr(self, 'drizzle_radio_incremental', None),

                # Widgets pour l'ÉCHELLE Drizzle (Existant)
                getattr(self, 'drizzle_scale_label', None),
                getattr(self, 'drizzle_radio_2x', None),
                getattr(self, 'drizzle_radio_3x', None),
                getattr(self, 'drizzle_radio_4x', None),
                # getattr(self, 'drizzle_scale_combo', None), # Si Combobox

                # Widgets pour le SEUIL WHT (Existant)
                getattr(self, 'drizzle_wht_label', None),
                getattr(self, 'drizzle_wht_spinbox', None),
                                
                # Kernel Drizzle
                getattr(self, 'drizzle_kernel_label', None),
                getattr(self, 'drizzle_kernel_combo', None),

                # Pixfrac Drizzle
                getattr(self, 'drizzle_pixfrac_label', None),
                getattr(self, 'drizzle_pixfrac_spinbox', None),
                # --- FIN DES NOUVELLES LIGNES ---
            ]

            # Boucle pour appliquer l'état (NORMAL ou DISABLED) à chaque widget de la liste
            for widget in widgets_to_toggle:
                # Vérifier si le widget existe réellement avant de tenter de le configurer
                if widget and hasattr(widget, 'winfo_exists') and widget.winfo_exists():
                    # Appliquer l'état global (activé/désactivé par la checkbox principale)
                    widget.config(state=state)

                    # --- Logique Optionnelle (pour plus tard) : Désactivation spécifique au mode ---
                    # Si Drizzle est activé ET que le mode est Incrémental ET que le widget est lié à une option non pertinente
                    # if global_drizzle_enabled and self.drizzle_mode_var.get() == "Incremental":
                    #     if widget in [widget_echelle_1, widget_echelle_2, ...]: # Liste des widgets à désactiver en mode incrémental
                    #         widget.config(state=tk.DISABLED)
                    #     # Sinon (widget pertinent ou mode final), il garde l'état 'state' défini plus haut


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
            if hasattr(self, 'scnr_amount_ctrls'):
                amount_widgets = [
                    self.scnr_amount_ctrls.get('slider'),
                    self.scnr_amount_ctrls.get('spinbox'),
                    self.scnr_amount_ctrls.get('label') # Griser le label aussi
                ]
                for widget in amount_widgets:
                    if widget and hasattr(widget, 'winfo_exists') and widget.winfo_exists():
                        widget.config(state=new_state)
            
            # Checkbox pour préserver la luminance
            if hasattr(self, 'final_scnr_preserve_lum_check') and \
               self.final_scnr_preserve_lum_check.winfo_exists():
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
        if hasattr(self, 'progress_bar') and hasattr(self, 'status_text'):
            self.progress_manager = ProgressManager(
                self.progress_bar, self.status_text,
                self.remaining_time_var, self.elapsed_time_var
            )
        else:
            print("Error: Progress widgets not found for ProgressManager initialization.")

        # Preview Manager
        if hasattr(self, 'preview_canvas'):
            self.preview_manager = PreviewManager(self.preview_canvas)
        else:
            print("Error: Preview canvas not found for PreviewManager initialization.")

        # Histogram Widget Callback (if widget exists)
        if hasattr(self, 'histogram_widget') and self.histogram_widget:
            self.histogram_widget.range_change_callback = self.update_stretch_from_histogram
        else:
            print("Error: HistogramWidget reference not found after create_layout.")

        # File Handler (should already be created in __init__)
        if not hasattr(self, 'file_handler'):
             print("Error: FileHandlingManager not initialized.")

        # Show initial state in preview area
        self.show_initial_preview()
        # Update additional folders display initially
        self.update_additional_folders_display()

    def _update_weighting_options_state(self):
        """Active ou désactive les options de pondération détaillées."""
        state = tk.NORMAL if self.use_weighting_var.get() else tk.DISABLED
        widgets_to_toggle = [
            getattr(self, 'weight_metrics_label', None),
            getattr(self, 'weight_snr_check', None),
            getattr(self, 'weight_stars_check', None),
            getattr(self, 'snr_exp_label', None),
            getattr(self, 'snr_exp_spinbox', None),
            getattr(self, 'stars_exp_label', None),
            getattr(self, 'stars_exp_spinbox', None),
            getattr(self, 'min_w_label', None),
            getattr(self, 'min_w_spinbox', None)
        ]
        for widget in widgets_to_toggle:
            if widget and hasattr(widget, 'winfo_exists') and widget.winfo_exists():
                try: widget.config(state=state)
                except tk.TclError: pass

    def show_initial_preview(self):
        """ Affiche un état initial dans la zone d'aperçu. """
        if hasattr(self, 'preview_manager') and self.preview_manager:
            self.preview_manager.clear_preview(self.tr("Select input/output folders."))
        if hasattr(self, 'histogram_widget') and self.histogram_widget:
            self.histogram_widget.plot_histogram(None) # Clear histogram

    def tr(self, key, default=None):
        """ Raccourci pour la localisation. """
        return self.localization.get(key, default=default)



    def _convert_spinbox_percent_to_float(self, *args):
        """Lit la variable d'affichage (%) et met à jour la variable de stockage (0-1)."""
        try:
            # --- MODIFIÉ: Lire depuis la variable d'affichage ---
            display_value_str = self.drizzle_wht_display_var.get()
            percent_value = float(display_value_str)
            # --- FIN MODIFICATION ---

            # La conversion et le clip restent les mêmes
            float_value = np.clip(percent_value / 100.0, 0.01, 1.0) # Assurer 0.01-1.0

            # Mettre à jour directement la variable de stockage (0.0-1.0)
            self.drizzle_wht_threshold_var.set(float_value)
            # print(f"DEBUG (Spinbox Cmd): Display='{display_value_str}', FloatSet={float_value:.3f}") # <-- DEBUG Optionnel

        except (ValueError, tk.TclError, AttributeError) as e:
            # Ignorer erreurs de conversion ou si les variables n'existent pas encore
            print(f"DEBUG (Spinbox Cmd): Ignored error during conversion: {e}") # <-- AJOUTÉ DEBUG
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
        print("DEBUG (GUI create_layout): Début création layout...")

        # --- Cadre Principal et PanedWindow ---
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        print("DEBUG (GUI create_layout): Cadre principal et PanedWindow créés.")

        # --- Panneau Gauche avec Scrollbar Intégrée ---
        left_canvas_container = ttk.Frame(paned_window, width=450)
        paned_window.add(left_canvas_container, weight=1)
        self.left_scrollbar = ttk.Scrollbar(left_canvas_container, orient="vertical")
        self.left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_background_color = self.root.cget('bg') # Utiliser la couleur de fond de la racine pour cohérence
        self.left_scrollable_canvas = tk.Canvas(left_canvas_container,
                                                highlightthickness=0,
                                                bg=canvas_background_color,
                                                yscrollcommand=self.left_scrollbar.set)
        self.left_scrollable_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.left_scrollbar.config(command=self.left_scrollable_canvas.yview)
        self.left_content_frame = ttk.Frame(self.left_scrollable_canvas)
        self.left_content_frame_id_on_canvas = self.left_scrollable_canvas.create_window(
            (0, 0), window=self.left_content_frame, anchor="nw", tags="self.left_content_frame_tag"
        )
        def _on_left_content_frame_configure(event): # Fonction locale
            # Mettre à jour la largeur du frame interne AVANT de calculer la scrollregion
            if self.left_scrollable_canvas.winfo_width() > 1:
                 self.left_scrollable_canvas.itemconfig(self.left_content_frame_id_on_canvas, width=self.left_scrollable_canvas.winfo_width())
            self.left_scrollable_canvas.config(scrollregion=self.left_scrollable_canvas.bbox("all"))
        self.left_content_frame.bind("<Configure>", _on_left_content_frame_configure)
        # Ajuster la largeur du frame interne quand le canvas est redimensionné
        self.left_scrollable_canvas.bind("<Configure>", lambda e, c=self.left_scrollable_canvas, i=self.left_content_frame_id_on_canvas: \
                                         c.itemconfig(i, width=e.width) if e.width > 1 else None)
        print("DEBUG (GUI create_layout): Panneau gauche scrollable configuré.")
        # --- Fin de la Configuration du Panneau Gauche Scrollable ---

        # --- Panneau Droit (pour Aperçu, Histogramme, Boutons de Contrôle) ---
        right_frame = ttk.Frame(paned_window, width=750)
        paned_window.add(right_frame, weight=3)
        print("DEBUG (GUI create_layout): Panneau droit créé.")

        # =======================================================================
        # --- Contenu du Panneau Gauche (dans self.left_content_frame) ---
        # =======================================================================
        print("DEBUG (GUI create_layout): Début remplissage panneau gauche...")

        # 1. Sélection Langue (en haut de self.left_content_frame)
        lang_frame = ttk.Frame(self.left_content_frame)
        lang_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 5), padx=5)
        ttk.Label(lang_frame, text="Language/Langue:").pack(side=tk.LEFT, padx=(0, 5))
        self.language_combo = ttk.Combobox(lang_frame, textvariable=self.language_var, values=("en", "fr"), width=8, state="readonly")
        self.language_combo.pack(side=tk.LEFT)
        self.language_combo.bind("<<ComboboxSelected>>", self.change_language)
        print("DEBUG (GUI create_layout): Sélection langue créée.")

        # 2. Notebook pour les Onglets d'Options (juste en dessous de la langue)
        self.control_notebook = ttk.Notebook(self.left_content_frame)
        self.control_notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 5), padx=5) # fill=tk.BOTH et expand=True
        print("DEBUG (GUI create_layout): Notebook de contrôle créé.")

        # --- Onglet Empilement (Index 0) ---
        tab_stacking = ttk.Frame(self.control_notebook)
        print("DEBUG (GUI create_layout): Frame pour onglet Empilement créé.")
        # ... (Votre code complet pour le contenu de tab_stacking ici, inchangé) ...
        self.folders_frame = ttk.LabelFrame(tab_stacking, text=self.tr("Folders")) # Note: self.tr() fonctionnera après init localisation
        self.folders_frame.pack(fill=tk.X, pady=5, padx=5)
        in_subframe = ttk.Frame(self.folders_frame); in_subframe.pack(fill=tk.X, padx=5, pady=(5, 2)); self.input_label = ttk.Label(in_subframe, text="Input:", width=10, anchor="w"); self.input_label.pack(side=tk.LEFT); self.browse_input_button = ttk.Button(in_subframe, text="Browse...", command=self.file_handler.browse_input, width=10); self.browse_input_button.pack(side=tk.RIGHT); self.input_entry = ttk.Entry(in_subframe, textvariable=self.input_path); self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5)); self.input_entry.bind("<FocusOut>", self._update_show_folders_button_state); self.input_entry.bind("<KeyRelease>", self._update_show_folders_button_state)
        out_subframe = ttk.Frame(self.folders_frame); out_subframe.pack(fill=tk.X, padx=5, pady=(2, 5)); self.output_label = ttk.Label(out_subframe, text="Output:", width=10, anchor="w"); self.output_label.pack(side=tk.LEFT); self.browse_output_button = ttk.Button(out_subframe, text="Browse...", command=self.file_handler.browse_output, width=10); self.browse_output_button.pack(side=tk.RIGHT); self.output_entry = ttk.Entry(out_subframe, textvariable=self.output_path); self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ref_frame = ttk.Frame(self.folders_frame); ref_frame.pack(fill=tk.X, padx=5, pady=(2, 5)); self.reference_label = ttk.Label(ref_frame, text="Reference (Opt.):", width=10, anchor="w"); self.reference_label.pack(side=tk.LEFT); self.browse_ref_button = ttk.Button(ref_frame, text="Browse...", command=self.file_handler.browse_reference, width=10); self.browse_ref_button.pack(side=tk.RIGHT); self.ref_entry = ttk.Entry(ref_frame, textvariable=self.reference_image_path); self.ref_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.options_frame = ttk.LabelFrame(tab_stacking, text="Stacking Options")
        self.options_frame.pack(fill=tk.X, pady=5, padx=5)
        method_kappa_scnr_frame = ttk.Frame(self.options_frame); method_kappa_scnr_frame.pack(fill=tk.X, padx=0, pady=(5, 0))
        mk_line1_frame = ttk.Frame(method_kappa_scnr_frame); mk_line1_frame.pack(fill=tk.X, padx=5)
        self.stacking_method_label = ttk.Label(mk_line1_frame, text="Method:"); self.stacking_method_label.grid(row=0, column=0, sticky=tk.W, padx=(0,2)); self.stacking_combo = ttk.Combobox(mk_line1_frame, textvariable=self.stacking_mode, values=("mean", "median", "kappa-sigma", "winsorized-sigma"), width=14, state="readonly"); self.stacking_combo.grid(row=0, column=1, sticky=tk.W, padx=(0,8)); self.stacking_combo.bind("<<ComboboxSelected>>", self._toggle_kappa_visibility); self.kappa_label = ttk.Label(mk_line1_frame, text="Kappa:"); self.kappa_label.grid(row=0, column=2, sticky=tk.W, padx=(0,2)); self.kappa_spinbox = ttk.Spinbox(mk_line1_frame, from_=1.0, to=5.0, increment=0.1, textvariable=self.kappa, width=5); self.kappa_spinbox.grid(row=0, column=3, sticky=tk.W, padx=(0,0))
        scnr_options_subframe_in_stacking = ttk.Frame(method_kappa_scnr_frame); scnr_options_subframe_in_stacking.pack(fill=tk.X, padx=5, pady=(5,2))
        self.apply_final_scnr_check = ttk.Checkbutton(scnr_options_subframe_in_stacking, text="Apply Final SCNR (Green)", variable=self.apply_final_scnr_var, command=self._update_final_scnr_options_state); self.apply_final_scnr_check.pack(anchor=tk.W, pady=(0,2))
        self.scnr_params_frame = ttk.Frame(scnr_options_subframe_in_stacking); self.scnr_params_frame.pack(fill=tk.X, padx=(20, 0))
        self.scnr_amount_ctrls = self._create_slider_spinbox_group(self.scnr_params_frame, "final_scnr_amount_label", min_val=0.0, max_val=1.0, step=0.05, tk_var=self.final_scnr_amount_var, callback=None)
        self.final_scnr_preserve_lum_check = ttk.Checkbutton(self.scnr_params_frame, text="Preserve Luminosity (SCNR)", variable=self.final_scnr_preserve_lum_var); self.final_scnr_preserve_lum_check.pack(anchor=tk.W, pady=(0,5))
        batch_frame = ttk.Frame(self.options_frame); batch_frame.pack(fill=tk.X, padx=5, pady=(5, 5)); self.batch_size_label = ttk.Label(batch_frame, text="Batch Size:"); self.batch_size_label.pack(side=tk.LEFT, padx=(0, 5)); self.batch_spinbox = ttk.Spinbox(batch_frame, from_=3, to=500, increment=1, textvariable=self.batch_size, width=5); self.batch_spinbox.pack(side=tk.LEFT)
        self.drizzle_options_frame = ttk.LabelFrame(tab_stacking, text="Drizzle Options"); self.drizzle_options_frame.pack(fill=tk.X, pady=5, padx=5); self.drizzle_check = ttk.Checkbutton(self.drizzle_options_frame, text="Enable Drizzle", variable=self.use_drizzle_var, command=self._update_drizzle_options_state); self.drizzle_check.pack(anchor=tk.W, padx=5, pady=(5, 2)); self.drizzle_mode_frame = ttk.Frame(self.drizzle_options_frame); self.drizzle_mode_frame.pack(fill=tk.X, padx=(20, 5), pady=(2, 5)); self.drizzle_mode_label = ttk.Label(self.drizzle_mode_frame, text="Mode:"); self.drizzle_mode_label.pack(side=tk.LEFT, padx=(0, 5)); self.drizzle_radio_final = ttk.Radiobutton(self.drizzle_mode_frame, text="Final", variable=self.drizzle_mode_var, value="Final", command=self._update_drizzle_options_state); self.drizzle_radio_incremental = ttk.Radiobutton(self.drizzle_mode_frame, text="Incremental", variable=self.drizzle_mode_var, value="Incremental", command=self._update_drizzle_options_state); self.drizzle_radio_final.pack(side=tk.LEFT, padx=3); self.drizzle_radio_incremental.pack(side=tk.LEFT, padx=3)
        self.drizzle_scale_frame = ttk.Frame(self.drizzle_options_frame); self.drizzle_scale_frame.pack(fill=tk.X, padx=(20, 5), pady=(0, 5)); self.drizzle_scale_label = ttk.Label(self.drizzle_scale_frame, text="Scale:"); self.drizzle_scale_label.pack(side=tk.LEFT, padx=(0, 5)); self.drizzle_radio_2x = ttk.Radiobutton(self.drizzle_scale_frame, text="x2", variable=self.drizzle_scale_var, value="2"); self.drizzle_radio_3x = ttk.Radiobutton(self.drizzle_scale_frame, text="x3", variable=self.drizzle_scale_var, value="3"); self.drizzle_radio_4x = ttk.Radiobutton(self.drizzle_scale_frame, text="x4", variable=self.drizzle_scale_var, value="4"); self.drizzle_radio_2x.pack(side=tk.LEFT, padx=3); self.drizzle_radio_3x.pack(side=tk.LEFT, padx=3); self.drizzle_radio_4x.pack(side=tk.LEFT, padx=3)
        wht_frame = ttk.Frame(self.drizzle_options_frame); wht_frame.pack(fill=tk.X, padx=(20, 5), pady=(5, 5)); self.drizzle_wht_label = ttk.Label(wht_frame, text="WHT Threshold %:"); self.drizzle_wht_label.pack(side=tk.LEFT, padx=(0, 5)); self.drizzle_wht_spinbox = ttk.Spinbox(wht_frame, from_=10.0, to=100.0, increment=5.0, textvariable=self.drizzle_wht_display_var, width=6, command=self._convert_spinbox_percent_to_float, format="%.0f"); self.drizzle_wht_spinbox.pack(side=tk.LEFT, padx=5)
        kernel_frame = ttk.Frame(self.drizzle_options_frame); kernel_frame.pack(fill=tk.X, padx=(20, 5), pady=(0, 5)); self.drizzle_kernel_label = ttk.Label(kernel_frame, text="Kernel:"); self.drizzle_kernel_label.pack(side=tk.LEFT, padx=(0, 5)); valid_kernels = ['square', 'gaussian', 'point', 'tophat', 'turbo', 'lanczos2', 'lanczos3']; self.drizzle_kernel_combo = ttk.Combobox(kernel_frame, textvariable=self.drizzle_kernel_var, values=valid_kernels, state="readonly", width=12); self.drizzle_kernel_combo.pack(side=tk.LEFT, padx=5)
        pixfrac_frame = ttk.Frame(self.drizzle_options_frame); pixfrac_frame.pack(fill=tk.X, padx=(20, 5), pady=(0, 5)); self.drizzle_pixfrac_label = ttk.Label(pixfrac_frame, text="Pixfrac:"); self.drizzle_pixfrac_label.pack(side=tk.LEFT, padx=(0, 5)); self.drizzle_pixfrac_spinbox = ttk.Spinbox(pixfrac_frame, from_=0.01, to=1.00, increment=0.05, textvariable=self.drizzle_pixfrac_var, width=6, format="%.2f"); self.drizzle_pixfrac_spinbox.pack(side=tk.LEFT, padx=5)
        self.hp_frame = ttk.LabelFrame(tab_stacking, text="Hot Pixel Correction"); self.hp_frame.pack(fill=tk.X, pady=5, padx=5); hp_check_frame = ttk.Frame(self.hp_frame); hp_check_frame.pack(fill=tk.X, padx=5, pady=2); self.hot_pixels_check = ttk.Checkbutton(hp_check_frame, text="Correct hot pixels", variable=self.correct_hot_pixels); self.hot_pixels_check.pack(side=tk.LEFT, padx=(0, 10)); hp_params_frame = ttk.Frame(self.hp_frame); hp_params_frame.pack(fill=tk.X, padx=5, pady=(2,5)); self.hot_pixel_threshold_label = ttk.Label(hp_params_frame, text="Threshold:"); self.hot_pixel_threshold_label.pack(side=tk.LEFT); self.hp_thresh_spinbox = ttk.Spinbox(hp_params_frame, from_=1.0, to=10.0, increment=0.1, textvariable=self.hot_pixel_threshold, width=5); self.hp_thresh_spinbox.pack(side=tk.LEFT, padx=5); self.neighborhood_size_label = ttk.Label(hp_params_frame, text="Neighborhood:"); self.neighborhood_size_label.pack(side=tk.LEFT); self.hp_neigh_spinbox = ttk.Spinbox(hp_params_frame, from_=3, to=15, increment=2, textvariable=self.neighborhood_size, width=4); self.hp_neigh_spinbox.pack(side=tk.LEFT, padx=5)
        self.weighting_frame = ttk.LabelFrame(tab_stacking, text="Quality Weighting"); self.weighting_frame.pack(fill=tk.X, pady=5, padx=5); self.use_weighting_check = ttk.Checkbutton(self.weighting_frame, text="Enable weighting", variable=self.use_weighting_var, command=self._update_weighting_options_state); self.use_weighting_check.pack(anchor=tk.W, padx=5, pady=(5,2)); self.weighting_options_frame = ttk.Frame(self.weighting_frame); self.weighting_options_frame.pack(fill=tk.X, padx=(20, 5), pady=(0, 5)); metrics_frame = ttk.Frame(self.weighting_options_frame); metrics_frame.pack(fill=tk.X, pady=2); self.weight_metrics_label = ttk.Label(metrics_frame, text="Metrics:"); self.weight_metrics_label.pack(side=tk.LEFT, padx=(0, 5)); self.weight_snr_check = ttk.Checkbutton(metrics_frame, text="SNR", variable=self.weight_snr_var); self.weight_snr_check.pack(side=tk.LEFT, padx=5); self.weight_stars_check = ttk.Checkbutton(metrics_frame, text="Star Count", variable=self.weight_stars_var); self.weight_stars_check.pack(side=tk.LEFT, padx=5); params_frame = ttk.Frame(self.weighting_options_frame); params_frame.pack(fill=tk.X, pady=2); self.snr_exp_label = ttk.Label(params_frame, text="SNR Exp.:"); self.snr_exp_label.pack(side=tk.LEFT, padx=(0, 2)); self.snr_exp_spinbox = ttk.Spinbox(params_frame, from_=0.1, to=3.0, increment=0.1, textvariable=self.snr_exponent_var, width=5); self.snr_exp_spinbox.pack(side=tk.LEFT, padx=(0, 10)); self.stars_exp_label = ttk.Label(params_frame, text="Stars Exp.:"); self.stars_exp_label.pack(side=tk.LEFT, padx=(0, 2)); self.stars_exp_spinbox = ttk.Spinbox(params_frame, from_=0.1, to=3.0, increment=0.1, textvariable=self.stars_exponent_var, width=5); self.stars_exp_spinbox.pack(side=tk.LEFT, padx=(0, 10)); self.min_w_label = ttk.Label(params_frame, text="Min Weight:"); self.min_w_label.pack(side=tk.LEFT, padx=(0, 2)); self.min_w_spinbox = ttk.Spinbox(params_frame, from_=0.01, to=1.0, increment=0.01, textvariable=self.min_weight_var, width=5); self.min_w_spinbox.pack(side=tk.LEFT, padx=(0, 5))
        self.post_proc_opts_frame = ttk.LabelFrame(tab_stacking, text="Post-Processing Options"); self.post_proc_opts_frame.pack(fill=tk.X, pady=5, padx=5); self.cleanup_temp_check = ttk.Checkbutton(self.post_proc_opts_frame, text="Cleanup temporary files", variable=self.cleanup_temp_var); self.cleanup_temp_check.pack(side=tk.LEFT, padx=5, pady=5); self.chroma_correction_check = ttk.Checkbutton(self.post_proc_opts_frame, text="Edge Enhance", variable=self.apply_chroma_correction_var); self.chroma_correction_check.pack(side=tk.LEFT, padx=5, pady=5)
        # ---
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
        expert_content_frame_id = expert_scroll_canvas.create_window((0,0), window=expert_content_frame, anchor="nw", tags="expert_content_frame_tag_v2") # Tag unique

        # Callback pour le scroll de l'onglet Expert
        def _on_expert_content_configure_local(event): # Nom unique pour la fonction locale
            # Mettre à jour la largeur du frame interne AVANT de calculer la scrollregion
            if expert_scroll_canvas.winfo_exists() and expert_scroll_canvas.winfo_width() > 1: # Vérifier existence
                 try: expert_scroll_canvas.itemconfig(expert_content_frame_id, width=expert_scroll_canvas.winfo_width())
                 except tk.TclError: pass # Au cas où l'item est détruit
            if expert_scroll_canvas.winfo_exists(): # Vérifier existence
                 try: expert_scroll_canvas.config(scrollregion=expert_scroll_canvas.bbox("all"))
                 except tk.TclError: pass
        
        expert_content_frame.bind("<Configure>", _on_expert_content_configure_local)
        expert_scroll_canvas.bind("<Configure>", lambda e, c=expert_scroll_canvas, i=expert_content_frame_id: \
                                 (c.itemconfig(i, width=e.width) if c.winfo_exists() and e.width > 1 else None) if c.winfo_exists() else None)
        print("DEBUG (GUI create_layout): Canvas scrollable pour onglet Expert configuré.")

        self.warning_label = ttk.Label(expert_content_frame, text=self.tr("expert_warning_text", default="Expert Settings!"), foreground="red", font=("Arial", 10, "italic"))
        self.warning_label.pack(pady=(5,10), padx=5, fill=tk.X)
        
        self.feathering_frame = ttk.LabelFrame(expert_content_frame, text=self.tr("feathering_frame_title", default="Feathering / Low WHT"), padding=5)
        self.feathering_frame.pack(fill=tk.X, padx=5, pady=5)
        print("DEBUG (GUI create_layout): Feathering Frame (conteneur pour Feathering et Low WHT) créé.")
        
        self.apply_feathering_check = ttk.Checkbutton(self.feathering_frame, text=self.tr("apply_feathering_label", default="Enable Feathering"), variable=self.apply_feathering_var, command=self._update_feathering_options_state)
        self.apply_feathering_check.pack(anchor=tk.W, padx=5, pady=(5,0)) 
        feather_params_frame = ttk.Frame(self.feathering_frame); feather_params_frame.pack(fill=tk.X, padx=(20,0), pady=(0,5)) 
        self.feather_blur_px_label = ttk.Label(feather_params_frame, text=self.tr("feather_blur_px_label", default="Blur (px):")); self.feather_blur_px_label.pack(side=tk.LEFT, padx=(0,5), pady=2)
        self.feather_blur_px_spinbox = ttk.Spinbox(feather_params_frame, from_=32, to=512, increment=16, width=6, textvariable=self.feather_blur_px_var); self.feather_blur_px_spinbox.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.low_wht_mask_check = ttk.Checkbutton(self.feathering_frame, text=self.tr("apply_low_wht_mask_label", default="Apply Low WHT Mask"), variable=self.apply_low_wht_mask_var, command=self._update_low_wht_mask_options_state )
        self.low_wht_mask_check.pack(anchor=tk.W, padx=5, pady=(10,0)) 
        low_wht_params_frame = ttk.Frame(self.feathering_frame); low_wht_params_frame.pack(fill=tk.X, padx=(20, 0), pady=(0,5)) 
        self.low_wht_pct_label = ttk.Label(low_wht_params_frame, text=self.tr("low_wht_percentile_label", default="Percentile:")); self.low_wht_pct_label.pack(side=tk.LEFT, padx=(0, 5), pady=2)
        self.low_wht_pct_spinbox = ttk.Spinbox(low_wht_params_frame, from_=1, to=100, increment=1, width=4, textvariable=self.low_wht_pct_var); self.low_wht_pct_spinbox.pack(side=tk.LEFT, padx=2, pady=2)
        self.low_wht_soften_px_label = ttk.Label(low_wht_params_frame, text=self.tr("low_wht_soften_px_label", default="Soften (px):")); self.low_wht_soften_px_label.pack(side=tk.LEFT, padx=(10, 5), pady=2) 
        self.low_wht_soften_px_spinbox = ttk.Spinbox(low_wht_params_frame, from_=32, to=512, increment=16, width=6, textvariable=self.low_wht_soften_px_var); self.low_wht_soften_px_spinbox.pack(side=tk.LEFT, padx=2, pady=2)
        print("DEBUG (GUI create_layout): Widgets Feathering et Low WHT Mask créés.")
        
        self.bn_frame = ttk.LabelFrame(expert_content_frame, text=self.tr("bn_frame_title", default="Auto Background Neutralization"), padding=5)
        self.bn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.bn_frame.columnconfigure(1, weight=0); self.bn_frame.columnconfigure(3, weight=0)
        self.bn_grid_size_actual_label = ttk.Label(self.bn_frame, text="Grid Size:"); self.bn_grid_size_actual_label.grid(row=0, column=0, sticky=tk.W, padx=2, pady=2); self.bn_grid_size_combo = ttk.Combobox(self.bn_frame, textvariable=self.bn_grid_size_str_var, values=["8x8", "16x16", "24x24", "32x32", "64x64"], width=7, state="readonly"); self.bn_grid_size_combo.grid(row=0, column=1, sticky=tk.W, padx=2, pady=2)
        self.bn_perc_low_actual_label = ttk.Label(self.bn_frame, text="BG Perc. Low:"); self.bn_perc_low_actual_label.grid(row=1, column=0, sticky=tk.W, padx=2, pady=2); self.bn_perc_low_spinbox = ttk.Spinbox(self.bn_frame, from_=0, to=40, increment=1, width=5, textvariable=self.bn_perc_low_var); self.bn_perc_low_spinbox.grid(row=1, column=1, sticky=tk.W, padx=2, pady=2) # Stocker spinbox
        self.bn_perc_high_actual_label = ttk.Label(self.bn_frame, text="BG Perc. High:"); self.bn_perc_high_actual_label.grid(row=1, column=2, sticky=tk.W, padx=2, pady=2); self.bn_perc_high_spinbox = ttk.Spinbox(self.bn_frame, from_=10, to=95, increment=1, width=5, textvariable=self.bn_perc_high_var); self.bn_perc_high_spinbox.grid(row=1, column=3, sticky=tk.W, padx=2, pady=2) # Stocker spinbox
        self.bn_std_factor_actual_label = ttk.Label(self.bn_frame, text="BG Std Factor:"); self.bn_std_factor_actual_label.grid(row=2, column=0, sticky=tk.W, padx=2, pady=2); self.bn_std_factor_spinbox = ttk.Spinbox(self.bn_frame, from_=0.5, to=5.0, increment=0.1, width=5, format="%.1f", textvariable=self.bn_std_factor_var); self.bn_std_factor_spinbox.grid(row=2, column=1, sticky=tk.W, padx=2, pady=2) # Stocker spinbox
        self.bn_min_gain_actual_label = ttk.Label(self.bn_frame, text="Min Gain:"); self.bn_min_gain_actual_label.grid(row=3, column=0, sticky=tk.W, padx=2, pady=2); self.bn_min_gain_spinbox = ttk.Spinbox(self.bn_frame, from_=0.1, to=2.0, increment=0.1, width=5, format="%.1f", textvariable=self.bn_min_gain_var); self.bn_min_gain_spinbox.grid(row=3, column=1, sticky=tk.W, padx=2, pady=2) # Stocker spinbox
        self.bn_max_gain_actual_label = ttk.Label(self.bn_frame, text="Max Gain:"); self.bn_max_gain_actual_label.grid(row=3, column=2, sticky=tk.W, padx=2, pady=2); self.bn_max_gain_spinbox = ttk.Spinbox(self.bn_frame, from_=1.0, to=10.0, increment=0.1, width=5, format="%.1f", textvariable=self.bn_max_gain_var); self.bn_max_gain_spinbox.grid(row=3, column=3, sticky=tk.W, padx=2, pady=2) # Stocker spinbox
        print("DEBUG (GUI create_layout): Cadre BN créé.")

        self.cb_frame = ttk.LabelFrame(expert_content_frame, text=self.tr("cb_frame_title", default="Edge/Chroma Correction"), padding=5)
        self.cb_frame.pack(fill=tk.X, padx=5, pady=5)
        self.cb_frame.columnconfigure(1, weight=0); self.cb_frame.columnconfigure(3, weight=0)
        self.cb_border_size_actual_label = ttk.Label(self.cb_frame, text="Border Size (px):"); self.cb_border_size_actual_label.grid(row=0, column=0, sticky=tk.W, padx=2, pady=2); self.cb_border_size_spinbox = ttk.Spinbox(self.cb_frame, from_=5, to=150, increment=5, width=5, textvariable=self.cb_border_size_var); self.cb_border_size_spinbox.grid(row=0, column=1, sticky=tk.W, padx=2, pady=2) # Stocker spinbox
        self.cb_blur_radius_actual_label = ttk.Label(self.cb_frame, text="Blur Radius (px):"); self.cb_blur_radius_actual_label.grid(row=0, column=2, sticky=tk.W, padx=2, pady=2); self.cb_blur_radius_spinbox = ttk.Spinbox(self.cb_frame, from_=0, to=50, increment=1, width=5, textvariable=self.cb_blur_radius_var); self.cb_blur_radius_spinbox.grid(row=0, column=3, sticky=tk.W, padx=2, pady=2) # Stocker spinbox
        self.cb_min_b_factor_actual_label = ttk.Label(self.cb_frame, text="Min Blue Factor:"); self.cb_min_b_factor_actual_label.grid(row=1, column=0, sticky=tk.W, padx=2, pady=2); self.cb_min_b_factor_spinbox = ttk.Spinbox(self.cb_frame, from_=0.1, to=1.0, increment=0.05, width=5, format="%.2f", textvariable=self.cb_min_b_factor_var); self.cb_min_b_factor_spinbox.grid(row=1, column=1, sticky=tk.W, padx=2, pady=2) # Stocker spinbox
        self.cb_max_b_factor_actual_label = ttk.Label(self.cb_frame, text="Max Blue Factor:"); self.cb_max_b_factor_actual_label.grid(row=1, column=2, sticky=tk.W, padx=2, pady=2); self.cb_max_b_factor_spinbox = ttk.Spinbox(self.cb_frame, from_=1.0, to=3.0, increment=0.05, width=5, format="%.2f", textvariable=self.cb_max_b_factor_var); self.cb_max_b_factor_spinbox.grid(row=1, column=3, sticky=tk.W, padx=2, pady=2) # Stocker spinbox
        print("DEBUG (GUI create_layout): Cadre CB créé.")

        self.crop_frame = ttk.LabelFrame(expert_content_frame, text=self.tr("crop_frame_title", default="Final Cropping"), padding=5)
        self.crop_frame.pack(fill=tk.X, padx=5, pady=5)
        self.final_edge_crop_actual_label = ttk.Label(self.crop_frame, text="Edge Crop (%):"); self.final_edge_crop_actual_label.pack(side=tk.LEFT, padx=(2,5), pady=2)
        self.final_edge_crop_spinbox = ttk.Spinbox(self.crop_frame, from_=0.0, to=25.0, increment=0.5, width=6, format="%.1f", textvariable=self.final_edge_crop_percent_var); self.final_edge_crop_spinbox.pack(side=tk.LEFT, padx=2, pady=2) # Stocker spinbox
        print("DEBUG (GUI create_layout): Cadre Crop créé.")

        self.photutils_bn_frame = ttk.LabelFrame(expert_content_frame, text=self.tr("photutils_bn_frame_title", default="2D Background Subtraction (Photutils)"), padding=5)
        self.photutils_bn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.apply_photutils_bn_check = ttk.Checkbutton(self.photutils_bn_frame, text="Enable Photutils 2D Bkg Subtraction", variable=self.apply_photutils_bn_var, command=self._update_photutils_bn_options_state); self.apply_photutils_bn_check.pack(anchor=tk.W, padx=5, pady=(5,2))
        self.photutils_params_frame = ttk.Frame(self.photutils_bn_frame); self.photutils_params_frame.pack(fill=tk.X, padx=(20,0), pady=2); self.photutils_params_frame.columnconfigure(1, weight=0); self.photutils_params_frame.columnconfigure(3, weight=0)
        self.photutils_bn_box_size_label = ttk.Label(self.photutils_params_frame, text="Box Size (px):"); self.photutils_bn_box_size_label.grid(row=0, column=0, sticky=tk.W, padx=2, pady=2); self.pb_box_spinbox = ttk.Spinbox(self.photutils_params_frame, from_=16, to=1024, increment=16, width=6, textvariable=self.photutils_bn_box_size_var); self.pb_box_spinbox.grid(row=0, column=1, sticky=tk.W, padx=2, pady=2) # Stocker spinbox
        self.photutils_bn_filter_size_label = ttk.Label(self.photutils_params_frame, text="Filter Size (px, odd):"); self.photutils_bn_filter_size_label.grid(row=0, column=2, sticky=tk.W, padx=(10,2), pady=2); self.pb_filt_spinbox = ttk.Spinbox(self.photutils_params_frame, from_=1, to=15, increment=2, width=5, textvariable=self.photutils_bn_filter_size_var); self.pb_filt_spinbox.grid(row=0, column=3, sticky=tk.W, padx=2, pady=2) # Stocker spinbox
        self.photutils_bn_sigma_clip_label = ttk.Label(self.photutils_params_frame, text="Sigma Clip Value:"); self.photutils_bn_sigma_clip_label.grid(row=1, column=0, sticky=tk.W, padx=2, pady=2); self.pb_sig_spinbox = ttk.Spinbox(self.photutils_params_frame, from_=1.0, to=5.0, increment=0.1, width=5, format="%.1f", textvariable=self.photutils_bn_sigma_clip_var); self.pb_sig_spinbox.grid(row=1, column=1, sticky=tk.W, padx=2, pady=2) # Stocker spinbox
        self.photutils_bn_exclude_percentile_label = ttk.Label(self.photutils_params_frame, text="Exclude Brightest (%):"); self.photutils_bn_exclude_percentile_label.grid(row=1, column=2, sticky=tk.W, padx=(10,2), pady=2); self.pb_excl_spinbox = ttk.Spinbox(self.photutils_params_frame, from_=0.0, to=100.0, increment=1.0, width=6, format="%.1f", textvariable=self.photutils_bn_exclude_percentile_var); self.pb_excl_spinbox.grid(row=1, column=3, sticky=tk.W, padx=2, pady=2) # Stocker spinbox
        print("DEBUG (GUI create_layout): Cadre Photutils BN créé.")
        
        self.reset_expert_button = ttk.Button(expert_content_frame, text=self.tr("reset_expert_button", default="Reset Expert Settings"), command=self._reset_expert_settings)
        self.reset_expert_button.pack(pady=(10,5))
        print("DEBUG (GUI create_layout): Bouton Reset Expert créé.")
        
        # Ajout de l'onglet Expert au Notebook (CORRECTION: s'assurer qu'il a un texte)
        expert_tab_title_text = self.tr("tab_expert_title", default=" Expert ")
        self.control_notebook.add(tab_expert, text=f' {expert_tab_title_text} ') # text ne peut pas être vide
        print("DEBUG (GUI create_layout): Onglet Expert ajouté au Notebook.")

        # --- Onglet Aperçu (Index 2) ---
        tab_preview = ttk.Frame(self.control_notebook)
        # ... (Contenu de l'onglet Aperçu - comme avant) ...
        self.wb_frame = ttk.LabelFrame(tab_preview, text="White Balance (Preview)"); self.wb_frame.pack(fill=tk.X, pady=5, padx=5); self.wb_r_ctrls = self._create_slider_spinbox_group(self.wb_frame, "wb_r", 0.1, 5.0, 0.01, self.preview_r_gain); self.wb_g_ctrls = self._create_slider_spinbox_group(self.wb_frame, "wb_g", 0.1, 5.0, 0.01, self.preview_g_gain); self.wb_b_ctrls = self._create_slider_spinbox_group(self.wb_frame, "wb_b", 0.1, 5.0, 0.01, self.preview_b_gain); wb_btn_frame = ttk.Frame(self.wb_frame); wb_btn_frame.pack(fill=tk.X, pady=5); self.auto_wb_button = ttk.Button(wb_btn_frame, text="Auto WB", command=self.apply_auto_white_balance, state=tk.NORMAL if _tools_available else tk.DISABLED); self.auto_wb_button.pack(side=tk.LEFT, padx=5); self.reset_wb_button = ttk.Button(wb_btn_frame, text="Reset WB", command=self.reset_white_balance); self.reset_wb_button.pack(side=tk.LEFT, padx=5)
        self.stretch_frame_controls = ttk.LabelFrame(tab_preview, text="Stretch (Preview)"); self.stretch_frame_controls.pack(fill=tk.X, pady=5, padx=5); stretch_method_frame = ttk.Frame(self.stretch_frame_controls); stretch_method_frame.pack(fill=tk.X, pady=2); self.stretch_method_label = ttk.Label(stretch_method_frame, text="Method:"); self.stretch_method_label.pack(side=tk.LEFT, padx=(5,5)); self.stretch_combo = ttk.Combobox(stretch_method_frame, textvariable=self.preview_stretch_method, values=("Linear", "Asinh", "Log"), width=15, state="readonly"); self.stretch_combo.pack(side=tk.LEFT); self.stretch_combo.bind("<<ComboboxSelected>>", self._debounce_refresh_preview); self.stretch_bp_ctrls = self._create_slider_spinbox_group(self.stretch_frame_controls, "stretch_bp", 0.0, 1.0, 0.001, self.preview_black_point, callback=self.update_histogram_lines_from_sliders); self.stretch_wp_ctrls = self._create_slider_spinbox_group(self.stretch_frame_controls, "stretch_wp", 0.0, 1.0, 0.001, self.preview_white_point, callback=self.update_histogram_lines_from_sliders); self.stretch_gamma_ctrls = self._create_slider_spinbox_group(self.stretch_frame_controls, "stretch_gamma", 0.1, 5.0, 0.01, self.preview_gamma); stretch_btn_frame = ttk.Frame(self.stretch_frame_controls); stretch_btn_frame.pack(fill=tk.X, pady=5); self.auto_stretch_button = ttk.Button(stretch_btn_frame, text="Auto Stretch", command=self.apply_auto_stretch, state=tk.NORMAL if _tools_available else tk.DISABLED); self.auto_stretch_button.pack(side=tk.LEFT, padx=5); self.reset_stretch_button = ttk.Button(stretch_btn_frame, text="Reset Stretch", command=self.reset_stretch); self.reset_stretch_button.pack(side=tk.LEFT, padx=5)
        self.bcs_frame = ttk.LabelFrame(tab_preview, text="Image Adjustments"); self.bcs_frame.pack(fill=tk.X, pady=5, padx=5); self.brightness_ctrls = self._create_slider_spinbox_group(self.bcs_frame, "brightness", 0.1, 3.0, 0.01, self.preview_brightness); self.contrast_ctrls = self._create_slider_spinbox_group(self.bcs_frame, "contrast", 0.1, 3.0, 0.01, self.preview_contrast); self.saturation_ctrls = self._create_slider_spinbox_group(self.bcs_frame, "saturation", 0.0, 3.0, 0.01, self.preview_saturation); bcs_btn_frame = ttk.Frame(self.bcs_frame); bcs_btn_frame.pack(fill=tk.X, pady=5); self.reset_bcs_button = ttk.Button(bcs_btn_frame, text="Reset Adjust.", command=self.reset_brightness_contrast_saturation); self.reset_bcs_button.pack(side=tk.LEFT, padx=5)
        preview_tab_title_text = self.tr("tab_preview", default=" Preview ")
        self.control_notebook.add(tab_preview, text=f' {preview_tab_title_text} ')
        print("DEBUG (GUI create_layout): Onglet Aperçu créé et ajouté.")
        # --- Fin du Notebook ---
        
        # --- Zone Progression (ENFANT DE self.left_content_frame, packé EN BAS) ---
        self.progress_frame = ttk.LabelFrame(self.left_content_frame, text=self.tr("progress", default="Progress"))
        self.progress_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, padx=5, pady=(10, 5))
        # ... (contenu de progress_frame comme avant) ...
        self.progress_bar = ttk.Progressbar(self.progress_frame, maximum=100, mode='determinate'); self.progress_bar.pack(fill=tk.X, padx=5, pady=(5, 2)); time_frame = ttk.Frame(self.progress_frame); time_frame.pack(fill=tk.X, padx=5, pady=2); time_frame.columnconfigure(0, weight=0); time_frame.columnconfigure(1, weight=1); time_frame.columnconfigure(2, weight=0); time_frame.columnconfigure(3, weight=0); self.remaining_time_label = ttk.Label(time_frame, text="ETA:"); self.remaining_time_label.grid(row=0, column=0, sticky='w'); self.remaining_time_value = ttk.Label(time_frame, textvariable=self.remaining_time_var, font=tkFont.Font(weight='bold'), anchor='w'); self.remaining_time_value.grid(row=0, column=1, sticky='w', padx=(2, 10)); self.elapsed_time_label = ttk.Label(time_frame, text="Elapsed:"); self.elapsed_time_label.grid(row=0, column=2, sticky='e', padx=(5,0)); self.elapsed_time_value = ttk.Label(time_frame, textvariable=self.elapsed_time_var, font=tkFont.Font(weight='bold'), width=9, anchor='e'); self.elapsed_time_value.grid(row=0, column=3, sticky='e', padx=(2,0)); files_info_frame = ttk.Frame(self.progress_frame); files_info_frame.pack(fill=tk.X, padx=5, pady=2); self.remaining_static_label = ttk.Label(files_info_frame, text="Remaining:"); self.remaining_static_label.pack(side=tk.LEFT); self.remaining_value_label = ttk.Label(files_info_frame, textvariable=self.remaining_files_var, width=12, anchor='w'); self.remaining_value_label.pack(side=tk.LEFT, padx=(2,10)); self.aligned_files_label = ttk.Label(files_info_frame, textvariable=self.aligned_files_var, width=12, anchor='w'); self.aligned_files_label.pack(side=tk.LEFT, padx=(10,0)); self.additional_value_label = ttk.Label(files_info_frame, textvariable=self.additional_folders_var, anchor='e'); self.additional_value_label.pack(side=tk.RIGHT); self.additional_static_label = ttk.Label(files_info_frame, text="Additional:"); self.additional_static_label.pack(side=tk.RIGHT, padx=(0, 2)); status_text_frame = ttk.Frame(self.progress_frame); status_text_font = tkFont.Font(family="Arial", size=8); status_text_frame.pack(fill=tk.X, expand=False, padx=5, pady=(2, 5)); self.copy_log_button = ttk.Button(status_text_frame, text="Copy", command=self._copy_log_to_clipboard, width=5); self.copy_log_button.pack(side=tk.RIGHT, padx=(2, 0), pady=0, anchor='ne'); self.status_scrollbar = ttk.Scrollbar(status_text_frame, orient="vertical"); self.status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=0); self.status_text = tk.Text(status_text_frame, height=6, wrap=tk.WORD, bd=0, font=status_text_font, relief=tk.FLAT, state=tk.DISABLED, yscrollcommand=self.status_scrollbar.set); self.status_text.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=0); self.status_scrollbar.config(command=self.status_text.yview)
        print("DEBUG (GUI create_layout): Zone de progression créée.")
        # --- Fin Panneau Gauche ---
        print("DEBUG (GUI create_layout): Fin remplissage panneau gauche.")

        # =====================================================================
        # --- Panneau Droit (Aperçu, Boutons de Contrôle, Histogramme) ---
        # =====================================================================
        # ... (Contenu panneau droit comme avant - inchangé) ...
        control_frame = ttk.Frame(right_frame)
        try: style = ttk.Style(); accent_style = 'Accent.TButton' if 'Accent.TButton' in style.element_names() else 'TButton'
        except tk.TclError: accent_style = 'TButton' 
        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_processing, style=accent_style); self.start_button.pack(side=tk.LEFT, padx=5, pady=5, ipady=2)
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED); self.stop_button.pack(side=tk.LEFT, padx=5, pady=5, ipady=2)
        self.analyze_folder_button = ttk.Button(control_frame, text="Analyze Input Folder", command=self._launch_folder_analyzer, state=tk.DISABLED); self.analyze_folder_button.pack(side=tk.LEFT, padx=5, pady=5, ipady=2)
        self.mosaic_options_button = ttk.Button(control_frame, text="Mosaic...", command=self._open_mosaic_settings_window); self.mosaic_options_button.pack(side=tk.LEFT, padx=5, pady=5, ipady=2)
                # --- NOUVEAU BOUTON POUR LES SOLVEURS LOCAUX ---
        self.local_solver_button = ttk.Button(control_frame, 
                                              text=self.tr("local_solver_button_text", default="Local Solvers..."), 
                                              command=self._open_local_solver_settings_window)
        self.local_solver_button.pack(side=tk.LEFT, padx=5, pady=5, ipady=2)
        # --- FIN NOUVEAU BOUTON ---
        self.open_output_button = ttk.Button(control_frame, text="Open Output", command=self._open_output_folder, state=tk.DISABLED); self.open_output_button.pack(side=tk.RIGHT, padx=5, pady=5, ipady=2)
        self.add_files_button = ttk.Button(control_frame, text="Add Folder", command=self.file_handler.add_folder, state=tk.NORMAL); self.add_files_button.pack(side=tk.RIGHT, padx=5, pady=5, ipady=2)
        self.show_folders_button = ttk.Button(control_frame, text="View Inputs", command=self._show_input_folder_list, state=tk.DISABLED); self.show_folders_button.pack(side=tk.RIGHT, padx=5, pady=5, ipady=2)
        self.histogram_frame = ttk.LabelFrame(right_frame, text="Histogram")
        hist_fig_height_inches = 2.2; hist_fig_dpi = 80; hist_height_pixels = int(hist_fig_height_inches * hist_fig_dpi * 1.1)
        self.histogram_frame.config(height=hist_height_pixels); self.histogram_frame.pack_propagate(False)
        self.histogram_widget = HistogramWidget(self.histogram_frame, range_change_callback=self.update_stretch_from_histogram)
        self.histogram_widget.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=(0,2), pady=(0,2))
        self.hist_reset_btn = ttk.Button(self.histogram_frame, text="R", command=self.histogram_widget.reset_zoom, width=2); self.hist_reset_btn.pack(side=tk.RIGHT, anchor=tk.NE, padx=(0,2), pady=2)
        self.preview_frame = ttk.LabelFrame(right_frame, text="Preview")
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="#1E1E1E", highlightthickness=0); self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        self.histogram_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, padx=5, pady=(5, 5))
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, padx=5, pady=(5, 0))
        self.preview_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=(5, 5))
        print("DEBUG (GUI create_layout): Panneau droit rempli.")
        # --- Fin Panneau Droit ---

        self._store_widget_references() # Doit être appelé après la création de TOUS les widgets
        self._toggle_kappa_visibility() # État initial Kappa
        self._update_final_scnr_options_state() 
        self._update_photutils_bn_options_state() 
        self._update_feathering_options_state() 
        self._update_low_wht_mask_options_state() # État initial Low WHT Mask
        print("DEBUG (GUI create_layout): Fin création layout et appels _update_..._state.")





#################################################################################################################################




    def _update_feathering_options_state(self, *args):
        """Active ou désactive le Spinbox de flou du feathering."""
        try:
            feathering_active = self.apply_feathering_var.get()
            new_state = tk.NORMAL if feathering_active else tk.DISABLED

            if hasattr(self, 'feather_blur_px_spinbox') and self.feather_blur_px_spinbox.winfo_exists():
                self.feather_blur_px_spinbox.config(state=new_state)
            if hasattr(self, 'feather_blur_px_label') and self.feather_blur_px_label.winfo_exists():
                 self.feather_blur_px_label.config(state=new_state) # Griser le label aussi

            print(f"DEBUG (GUI): État options Feathering (Blur Px Spinbox) mis à jour: {new_state}")
        except tk.TclError: pass
        except AttributeError: pass
        except Exception as e:
            print(f"ERREUR inattendue dans _update_feathering_options_state: {e}")
            traceback.print_exc(limit=1)
    

##############################################################################################################################




    def _update_low_wht_mask_options_state(self, *args):
        """
        Active ou désactive les options de percentile et soften pour Low WHT Mask
        en fonction de l'état de la checkbox principale.
        """
        print("DEBUG (GUI _update_low_wht_mask_options_state): Exécution...") # Debug
        try:
            # Lire l'état de la checkbox principale pour "Low WHT Mask"
            mask_active = self.apply_low_wht_mask_var.get()
            new_state = tk.NORMAL if mask_active else tk.DISABLED
            print(f"  -> Mask active: {mask_active}, New state for options: {new_state}") # Debug

            # Mettre à jour l'état du label et du spinbox pour le percentile
            if hasattr(self, 'low_wht_pct_label') and self.low_wht_pct_label.winfo_exists():
                self.low_wht_pct_label.config(state=new_state)
            if hasattr(self, 'low_wht_pct_spinbox') and self.low_wht_pct_spinbox.winfo_exists():
                self.low_wht_pct_spinbox.config(state=new_state)
            
            # Mettre à jour l'état du label et du spinbox pour soften_px
            if hasattr(self, 'low_wht_soften_px_label') and self.low_wht_soften_px_label.winfo_exists():
                self.low_wht_soften_px_label.config(state=new_state)
            if hasattr(self, 'low_wht_soften_px_spinbox') and self.low_wht_soften_px_spinbox.winfo_exists():
                self.low_wht_soften_px_spinbox.config(state=new_state)

            print(f"DEBUG (GUI _update_low_wht_mask_options_state): État des options Low WHT Mask mis à jour.") # Debug
        
        except tk.TclError as e:
            # Peut arriver si les widgets sont en cours de destruction/création
            print(f"DEBUG (GUI _update_low_wht_mask_options_state): Erreur TclError -> {e}") # Debug
            pass 
        except AttributeError as e:
            # Peut arriver si un attribut (widget) n'existe pas encore (ex: pendant l'init)
            print(f"DEBUG (GUI _update_low_wht_mask_options_state): Erreur AttributeError -> {e}") # Debug
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
                    pass          # e.g. a Label – no 'state' option





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




##############################################################################################################################
    def _toggle_kappa_visibility(self, event=None):
        """Affiche ou cache les widgets Kappa en fonction de la méthode de stacking, en utilisant grid."""
        show_kappa = self.stacking_mode.get() in ["kappa-sigma", "winsorized-sigma"]
        
        # Assurer que les widgets existent avant de les manipuler
        if hasattr(self, 'kappa_label') and self.kappa_label and \
           hasattr(self, 'kappa_spinbox') and self.kappa_spinbox:
            
            if show_kappa:
                # Afficher en utilisant grid avec les mêmes options que lors de la création
                print("DEBUG: Affichage widgets Kappa avec grid") # Debug
                self.kappa_label.grid(row=0, column=2, sticky=tk.W, padx=(0,2))
                self.kappa_spinbox.grid(row=0, column=3, sticky=tk.W, padx=(0,5))
            else:
                # Cacher en utilisant grid_remove
                print("DEBUG: Masquage widgets Kappa avec grid_remove") # Debug
                self.kappa_label.grid_remove()
                self.kappa_spinbox.grid_remove()
        else:
            print("DEBUG: Widgets Kappa non trouvés dans _toggle_kappa_visibility") # Debug




#################################################################################################################################

    def _launch_folder_analyzer(self):
        """
        Détermine le chemin du fichier de commande, lance le script analyse_gui.py
        en lui passant ce chemin, et démarre la surveillance du fichier.
        """
        print("DEBUG (GUI): Entrée dans _launch_folder_analyzer.") # <-- AJOUTÉ DEBUG
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
            self.analyzer_command_file_path = os.path.join(app_temp_dir, command_filename) # <-- Stocker le chemin dans l'instance
            print(f"DEBUG (GUI): Chemin fichier commande défini: {self.analyzer_command_file_path}") # <-- AJOUTÉ DEBUG

            # --- Nettoyer ancien fichier de commande s'il existe (sécurité) ---
            if os.path.exists(self.analyzer_command_file_path):
                print(f"DEBUG (GUI): Suppression ancien fichier commande existant: {self.analyzer_command_file_path}") # <-- AJOUTÉ DEBUG
                try:
                    os.remove(self.analyzer_command_file_path)
                except OSError as e_rem:
                    print(f"AVERTISSEMENT (GUI): Impossible de supprimer ancien fichier commande: {e_rem}")
            # --- Fin Nettoyage ---

        except Exception as e_path:
            messagebox.showerror(
                self.tr("error"), # Utiliser une clé générique ou créer une clé spécifique
                f"Impossible de déterminer le chemin du fichier de communication temporaire:\n{e_path}"
            )
            print(f"ERREUR (GUI): Échec détermination chemin fichier commande: {e_path}") # <-- AJOUTÉ DEBUG
            traceback.print_exc(limit=2)
            return
        # --- FIN MODIFICATION ---

        # 3. Déterminer le chemin vers le script analyse_gui.py (inchangé)
        try:
            gui_file_path = os.path.abspath(__file__)
            gui_dir = os.path.dirname(gui_file_path)
            seestar_dir = os.path.dirname(gui_dir)
            project_root_parent = os.path.dirname(seestar_dir)
            analyzer_script_path = os.path.join(project_root_parent, 'seestar', 'beforehand', 'analyse_gui.py')
            analyzer_script_path = os.path.normpath(analyzer_script_path)
            print(f"DEBUG (GUI): Chemin script analyseur trouvé: {analyzer_script_path}") # <-- AJOUTÉ DEBUG
        except Exception as e:
            messagebox.showerror(self.tr("analyzer_launch_error_title"), f"Erreur interne chemin analyseur:\n{e}")
            return

        # 4. Vérifier si le script analyseur existe (inchangé)
        if not os.path.exists(analyzer_script_path):
            messagebox.showerror(self.tr("analyzer_launch_error_title"), self.tr("analyzer_script_not_found").format(path=analyzer_script_path))
            return

        # 5. Construire et lancer la commande (MODIFIÉ pour ajouter l'argument command_file_path)
        try:
            command = [
                sys.executable,
                analyzer_script_path,
                "--input-dir", input_folder,
                "--command-file", self.analyzer_command_file_path # <-- AJOUTÉ: Passer le chemin fichier commande
            ]
            print(f"DEBUG (GUI): Commande lancement analyseur: {' '.join(command)}") # <-- AJOUTÉ DEBUG

            # Lancer comme processus séparé non bloquant
            process = subprocess.Popen(command)
            self.update_progress_gui(self.tr("analyzer_launched"), None)

            # --- NOUVEAU : Démarrer la surveillance du fichier de commande ---
            print("DEBUG (GUI): Démarrage de la surveillance du fichier commande...") # <-- AJOUTÉ DEBUG
            # Assurer qu'une seule boucle de vérification tourne à la fois
            if hasattr(self, '_analyzer_check_after_id') and self._analyzer_check_after_id:
                print("DEBUG (GUI): Annulation surveillance précédente...") # <-- AJOUTÉ DEBUG
                try:
                    self.root.after_cancel(self._analyzer_check_after_id)
                except tk.TclError: pass # Ignore error if already cancelled/invalid
                self._analyzer_check_after_id = None

            # Démarrer la nouvelle boucle de vérification (ex: toutes les 1 seconde)
            self._check_analyzer_command_file()
            # --- FIN NOUVEAU ---

        # 6. Gestion des erreurs de lancement (inchangé)
        except FileNotFoundError:
             messagebox.showerror(self.tr("analyzer_launch_error_title"), self.tr("analyzer_launch_failed").format(error=f"Python '{sys.executable}' or script not found."))
        except OSError as e:
             messagebox.showerror(self.tr("analyzer_launch_error_title"), self.tr("analyzer_launch_failed").format(error=f"OS error: {e}"))
        except Exception as e:
            messagebox.showerror(self.tr("analyzer_launch_error_title"), self.tr("analyzer_launch_failed").format(error=str(e)))
            traceback.print_exc(limit=2)
        print("DEBUG (GUI): Sortie de _launch_folder_analyzer.") # <-- AJOUTÉ DEBUG


#############################################################################################################################



    def _check_analyzer_command_file(self):
        """
        Vérifie périodiquement l'existence du fichier de commande de l'analyseur.
        Si trouvé, lit le chemin, le supprime, et lance le stacking.
        """
        # print("DEBUG (GUI): _check_analyzer_command_file() exécuté.") # <-- DEBUG (peut être trop verbeux)

        # --- Vérifications préliminaires ---
        # 1. Le chemin du fichier de commande est-il défini ?
        if not hasattr(self, 'analyzer_command_file_path') or not self.analyzer_command_file_path:
            print("DEBUG (GUI): Vérification fichier commande annulée (chemin non défini).")
            self._analyzer_check_after_id = None # Assurer l'arrêt
            return

        # 2. Un traitement est-il déjà en cours dans le stacker principal ?
        if self.processing:
            # print("DEBUG (GUI): Traitement principal en cours, arrêt temporaire surveillance fichier commande.")
             # Pas besoin de vérifier le fichier si on est déjà en train de stacker/traiter.
             # On pourrait replanifier plus tard, mais pour l'instant, on arrête la surveillance
             # une fois le traitement principal démarré.
             self._analyzer_check_after_id = None # Arrêter la boucle si traitement lancé par autre chose
             return

        # 3. Le fichier de commande existe-t-il ?
        try:
            if os.path.exists(self.analyzer_command_file_path):
                print(f"DEBUG (GUI): Fichier commande détecté: {self.analyzer_command_file_path}") # <-- AJOUTÉ DEBUG

                # --- Traitement du fichier ---
                file_content = None
                try:
                    # Lire le contenu (chemin du dossier)
                    with open(self.analyzer_command_file_path, 'r', encoding='utf-8') as f_cmd:
                        file_content = f_cmd.read().strip()
                    print(f"DEBUG (GUI): Contenu fichier commande lu: '{file_content}'") # <-- AJOUTÉ DEBUG

                    # Supprimer le fichier IMMÉDIATEMENT après lecture réussie
                    try:
                        os.remove(self.analyzer_command_file_path)
                        print(f"DEBUG (GUI): Fichier commande supprimé.") # <-- AJOUTÉ DEBUG
                    except OSError as e_rem:
                        print(f"AVERTISSEMENT (GUI): Échec suppression fichier commande {self.analyzer_command_file_path}: {e_rem}")
                        # Continuer quand même si la lecture a réussi

                except IOError as e_read:
                    print(f"ERREUR (GUI): Impossible de lire le fichier commande {self.analyzer_command_file_path}: {e_read}")
                    # Essayer de supprimer le fichier même si lecture échoue (peut être corrompu)
                    try: os.remove(self.analyzer_command_file_path)
                    except OSError: pass
                    # Replanifier la vérification car on n'a pas pu traiter
                    if hasattr(self.root, 'after'): # Vérifier si root existe toujours
                         self._analyzer_check_after_id = self.root.after(1000, self._check_analyzer_command_file) # Replanifier dans 1s
                    return # Sortir pour cette itération

                # --- Agir sur le contenu lu ---
                if file_content and os.path.isdir(file_content):
                    analyzed_folder_path = os.path.abspath(file_content)
                    print(f"INFO (GUI): Commande d'empilement reçue pour: {analyzed_folder_path}") # <-- AJOUTÉ INFO

                    # Mettre à jour le champ d'entrée
                    current_input = self.input_path.get()
                    if os.path.normpath(current_input) != os.path.normpath(analyzed_folder_path):
                        print(f"DEBUG (GUI): Mise à jour du champ dossier d'entrée vers: {analyzed_folder_path}") # <-- AJOUTÉ DEBUG
                        self.input_path.set(analyzed_folder_path)
                        # Mettre à jour aussi le setting pour cohérence ?
                        self.settings.input_folder = analyzed_folder_path
                        # Redessiner l'aperçu initial si le dossier a changé
                        self._try_show_first_input_image()

                    # Vérifier si le dossier de sortie est défini
                    if not self.output_path.get():
                        default_output = os.path.join(analyzed_folder_path, "stack_output_analyzer") # Nom différent?
                        print(f"INFO (GUI): Dossier sortie non défini, utilisation défaut: {default_output}") # <-- AJOUTÉ INFO
                        self.output_path.set(default_output)
                        self.settings.output_folder = default_output

                    # Démarrer le stacking
                    print("DEBUG (GUI): Appel de self.start_processing() suite à commande analyseur...") # <-- AJOUTÉ DEBUG
                    self.start_processing()
                    # Pas besoin de replanifier la vérification, le but est atteint.
                    self._analyzer_check_after_id = None
                    return # Sortir de la méthode

                else:
                    print(f"AVERTISSEMENT (GUI): Contenu fichier commande invalide ('{file_content}') ou n'est pas un dossier. Fichier supprimé.")
                    # Replanifier la vérification car le contenu était invalide
                    if hasattr(self.root, 'after'): # Vérifier si root existe toujours
                        self._analyzer_check_after_id = self.root.after(1000, self._check_analyzer_command_file) # Replanifier dans 1s
                    return # Sortir pour cette itération

            else:
                # --- Replanifier si le fichier n'existe pas ---
                # print("DEBUG (GUI): Fichier commande non trouvé, replanification...") # <-- DEBUG (trop verbeux)
                # Vérifier si la fenêtre racine existe toujours avant de replanifier
                if hasattr(self.root, 'after'):
                    self._analyzer_check_after_id = self.root.after(1000, self._check_analyzer_command_file) # Vérifier à nouveau dans 1000 ms (1 seconde)
                else:
                    print("DEBUG (GUI): Fenêtre racine détruite, arrêt surveillance fichier commande.")
                    self._analyzer_check_after_id = None # Arrêter si la fenêtre est fermée

        except Exception as e_check:
             print(f"ERREUR (GUI): Erreur inattendue dans _check_analyzer_command_file: {e_check}")
             traceback.print_exc(limit=2)
             # Essayer de replanifier même en cas d'erreur pour ne pas bloquer
             if hasattr(self.root, 'after'):
                 try:
                     self._analyzer_check_after_id = self.root.after(2000, self._check_analyzer_command_file) # Attendre un peu plus longtemps après une erreur
                 except tk.TclError:
                     self._analyzer_check_after_id = None # Arrêter si la fenêtre est fermée
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
            if hasattr(self, 'show_folders_button') and self.show_folders_button.winfo_exists():
                self.show_folders_button.config(state=new_state)

            # Mettre à jour le bouton "Analyze Input Folder"
            if hasattr(self, 'analyze_folder_button') and self.analyze_folder_button.winfo_exists():
                self.analyze_folder_button.config(state=new_state)

        except tk.TclError:
            # Ignorer les erreurs si les widgets n'existent pas encore ou sont détruits
            pass
        except Exception as e:
             print(f"Error in _update_show_folders_button_state: {e}")


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
        if self.processing and hasattr(self, 'queued_stacker') and self.queued_stacker:
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
            if abs_path not in folder_list: # Avoid duplicates if added strangely
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
            dialog.resizable(False, False) # Prevent resizing

            # --- Add Frame, Scrollbar, and Text Widget ---
            frame = ttk.Frame(dialog, padding="10")
            frame.pack(fill=tk.BOTH, expand=True)

            scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
            # Use a fixed-size font for better alignment if needed
            # text_font = tkFont.Font(family="Courier New", size=9)
            list_text = tk.Text(
                frame,
                wrap=tk.WORD,
                height=15, # Adjust height as needed
                width=80,  # Adjust width as needed
                yscrollcommand=scrollbar.set,
                # font=text_font, # Optional fixed font
                padx=5, pady=5,
                state=tk.DISABLED # Start disabled
            )
            scrollbar.config(command=list_text.yview)

            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            list_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Insert the text
            list_text.config(state=tk.NORMAL)
            list_text.delete(1.0, tk.END)
            list_text.insert(tk.END, display_text)
            list_text.config(state=tk.DISABLED) # Make read-only

            # --- Add Close Button ---
            button_frame = ttk.Frame(dialog, padding="0 10 10 10")
            button_frame.pack(fill=tk.X)
            close_button = ttk.Button(button_frame, text="Close", command=dialog.destroy) # Add translation if needed
            # Use pack with anchor or alignment if needed
            close_button.pack(anchor=tk.CENTER) # Center the button
            close_button.focus_set() # Set focus to close button

            # --- Center the dialog ---
            dialog.update_idletasks()
            root_x = self.root.winfo_rootx(); root_y = self.root.winfo_rooty()
            root_w = self.root.winfo_width(); root_h = self.root.winfo_height()
            dlg_w = dialog.winfo_width(); dlg_h = dialog.winfo_height()
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
        frame = ttk.Frame(parent); frame.pack(fill=tk.X, padx=5, pady=(1,1))
        label_widget = ttk.Label(frame, text=self.tr(label_key, default=label_key), width=10); label_widget.pack(side=tk.LEFT)
        decimals = 0; log_step = -3
        if step > 0:
            try: log_step = math.log10(step);
            except ValueError: pass
            if log_step < 0: decimals = abs(int(log_step))
        format_str = f"%.{decimals}f"
        spinbox = ttk.Spinbox(frame, from_=min_val, to=max_val, increment=step,textvariable=tk_var, width=7, justify=tk.RIGHT, command=self._debounce_refresh_preview, format=format_str)
        spinbox.pack(side=tk.RIGHT, padx=(5,0))
        def on_scale_change(value_str):
            try: value = float(value_str)
            except ValueError: return
            if callback:
                try: callback(value)
                except Exception as cb_err: print(f"Error in slider immediate callback: {cb_err}")
            self._debounce_refresh_preview()
        slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=tk_var, orient=tk.HORIZONTAL, command=on_scale_change)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ctrls = {'frame': frame, 'label': label_widget, 'slider': slider, 'spinbox': spinbox}; return ctrls
##################################################################################################################




# --- DANS LA CLASSE SeestarStackerGUI DANS seestar/gui/main_window.py ---

    def _store_widget_references(self):
        """Stocke les références aux widgets qui nécessitent des mises à jour linguistiques et des infobulles."""
        print("\nDEBUG (GUI _store_widget_references): Début stockage références widgets...")
        notebook_widget = None
        try:
            if hasattr(self, 'control_notebook') and isinstance(self.control_notebook, ttk.Notebook):
                notebook_widget = self.control_notebook
        except Exception as e:
            print(f"Warning (GUI _store_widget_references): Erreur accès control_notebook: {e}")

        self.widgets_to_translate = {}
        
        # --- Onglets du Notebook ---
        try:
            if notebook_widget and notebook_widget.winfo_exists(): # Vérifier si le notebook existe
                if notebook_widget.index("end") > 0: self.widgets_to_translate["tab_stacking"] = (notebook_widget, 0)
                if notebook_widget.index("end") > 1: self.widgets_to_translate["tab_expert_title"] = (notebook_widget, 1)
                if notebook_widget.index("end") > 2: self.widgets_to_translate["tab_preview"] = (notebook_widget, 2)
        except tk.TclError: print("DEBUG: Erreur accès onglets (probablement non encore tous créés).")


        # --- LabelFrames (cadres avec titre) ---
        label_frames_keys = {
            "Folders": 'folders_frame', "options": 'options_frame',
            "drizzle_options_frame_label": 'drizzle_options_frame',
            "hot_pixels_correction": 'hp_frame', "quality_weighting_frame": 'weighting_frame',
            "post_proc_opts_frame_label": 'post_proc_opts_frame',
            "white_balance": 'wb_frame', "stretch_options": 'stretch_frame_controls',
            "image_adjustments": 'bcs_frame', "progress": 'progress_frame',
            "preview": 'preview_frame', "histogram": 'histogram_frame',
            "bn_frame_title": 'bn_frame', "cb_frame_title": 'cb_frame',
            "crop_frame_title": 'crop_frame',
            "feathering_frame_title": 'feathering_frame', 
            "photutils_bn_frame_title": 'photutils_bn_frame'
        }
        for key, attr_name in label_frames_keys.items():
            self.widgets_to_translate[key] = getattr(self, attr_name, None)

        # --- Labels (étiquettes simples) & Checkbuttons (pour leur texte) ---
        labels_and_checks_keys = { 
            "input_folder": 'input_label', "output_folder": 'output_label',
            "reference_image": 'reference_label', "stacking_method": 'stacking_method_label',
            "kappa_value": 'kappa_label', "batch_size": 'batch_size_label',
            "drizzle_scale_label": 'drizzle_scale_label', "drizzle_mode_label": 'drizzle_mode_label',
            "drizzle_kernel_label": 'drizzle_kernel_label', "drizzle_pixfrac_label": 'drizzle_pixfrac_label',
            "drizzle_wht_threshold_label": 'drizzle_wht_label',
            "hot_pixel_threshold": 'hot_pixel_threshold_label', "neighborhood_size": 'neighborhood_size_label',
            "weighting_metrics_label": 'weight_metrics_label', "snr_exponent_label": 'snr_exp_label',
            "stars_exponent_label": 'stars_exp_label', "min_weight_label": 'min_w_label',
            "wb_r": getattr(self, 'wb_r_ctrls', {}).get('label'), 
            "wb_g": getattr(self, 'wb_g_ctrls', {}).get('label'), 
            "wb_b": getattr(self, 'wb_b_ctrls', {}).get('label'),
            "stretch_method": 'stretch_method_label',
            "stretch_bp": getattr(self, 'stretch_bp_ctrls', {}).get('label'), 
            "stretch_wp": getattr(self, 'stretch_wp_ctrls', {}).get('label'), 
            "stretch_gamma": getattr(self, 'stretch_gamma_ctrls', {}).get('label'),
            "brightness": getattr(self, 'brightness_ctrls', {}).get('label'), 
            "contrast": getattr(self, 'contrast_ctrls', {}).get('label'), 
            "saturation": getattr(self, 'saturation_ctrls', {}).get('label'),
            "estimated_time": 'remaining_time_label', "elapsed_time": 'elapsed_time_label',
            "Remaining:": 'remaining_static_label', "Additional:": 'additional_static_label',
            "expert_warning_text": 'warning_label',
            "bn_grid_size_label": 'bn_grid_size_actual_label', 
            "bn_perc_low_label": 'bn_perc_low_actual_label', 
            "bn_perc_high_label": 'bn_perc_high_actual_label',
            "bn_std_factor_label": 'bn_std_factor_actual_label', 
            "bn_min_gain_label": 'bn_min_gain_actual_label', 
            "bn_max_gain_label": 'bn_max_gain_actual_label',
            "cb_border_size_label": 'cb_border_size_actual_label', 
            "cb_blur_radius_label": 'cb_blur_radius_actual_label',
            "cb_min_b_factor_label": 'cb_min_b_factor_actual_label', 
            "cb_max_b_factor_label": 'cb_max_b_factor_actual_label',
            "final_edge_crop_label": 'final_edge_crop_actual_label',
            "apply_final_scnr_label": 'apply_final_scnr_check', 
            "final_scnr_amount_label": getattr(self, 'scnr_amount_ctrls', {}).get('label'), 
            "final_scnr_preserve_lum_label": 'final_scnr_preserve_lum_check',
            "apply_photutils_bn_label": 'apply_photutils_bn_check', 
            "photutils_bn_box_size_label": 'photutils_bn_box_size_label',
            "photutils_bn_filter_size_label": 'photutils_bn_filter_size_label', 
            "photutils_bn_sigma_clip_label": 'photutils_bn_sigma_clip_label',
            "photutils_bn_exclude_percentile_label": 'photutils_bn_exclude_percentile_label',
            "apply_feathering_label": 'apply_feathering_check', 
            "feather_blur_px_label": 'feather_blur_px_label',
            "apply_low_wht_mask_label": 'low_wht_mask_check', 
            "low_wht_percentile_label": 'low_wht_pct_label', 
            "low_wht_soften_px_label": 'low_wht_soften_px_label',
            "drizzle_activate_check": 'drizzle_check', 
            "perform_hot_pixels_correction": 'hot_pixels_check',
            "enable_weighting_check": 'use_weighting_check', 
            "weight_snr_check": 'weight_snr_check',
            "weight_stars_check": 'weight_stars_check', 
            "cleanup_temp_check_label": 'cleanup_temp_check',
            "chroma_correction_check": 'chroma_correction_check'
        }
        for key, item in labels_and_checks_keys.items():
            if isinstance(item, tk.Widget): self.widgets_to_translate[key] = item
            elif isinstance(item, str): self.widgets_to_translate[key] = getattr(self, item, None)

        buttons_keys = { 
            "browse_input_button": 'browse_input_button', "browse_output_button": 'browse_output_button',
            "browse_ref_button": 'browse_ref_button', "auto_wb": 'auto_wb_button',
            "reset_wb": 'reset_wb_button', "auto_stretch": 'auto_stretch_button',
            "reset_stretch": 'reset_stretch_button', "reset_bcs": 'reset_bcs_button',
            "start": 'start_button', "stop": 'stop_button', "add_folder_button": 'add_files_button',
            "show_folders_button_text": 'show_folders_button', "copy_log_button_text": 'copy_log_button',
            "open_output_button_text": 'open_output_button', "analyze_folder_button": 'analyze_folder_button',
            "local_solver_button_text": 'local_solver_button',
            "Mosaic...": 'mosaic_options_button', "reset_expert_button": 'reset_expert_button'
        }
        for key, attr_name in buttons_keys.items():
            self.widgets_to_translate[key] = getattr(self, attr_name, None)
        
        radio_buttons_keys = { 
            "drizzle_radio_2x_label": 'drizzle_radio_2x', "drizzle_radio_3x_label": 'drizzle_radio_3x',
            "drizzle_radio_4x_label": 'drizzle_radio_4x', "drizzle_radio_final": 'drizzle_radio_final',
            "drizzle_radio_incremental": 'drizzle_radio_incremental'
        }
        for key, attr_name in radio_buttons_keys.items():
            self.widgets_to_translate[key] = getattr(self, attr_name, None)

        # --- TOOLTIPS ---
        self.tooltips = {} 
        print(f"DEBUG (GUI _store_widget_references): Dictionnaire self.tooltips réinitialisé.")

        # --- ICI : Définition de tooltips_config_list ---
        tooltips_config_list = [
            # Onglet Expert - BN
            ('bn_grid_size_actual_label', 'tooltip_bn_grid_size'), ('bn_grid_size_combo', 'tooltip_bn_grid_size'),
            ('bn_perc_low_actual_label', 'tooltip_bn_perc_low'), ('bn_perc_low_spinbox', 'tooltip_bn_perc_low'),
            ('bn_perc_high_actual_label', 'tooltip_bn_perc_high'), ('bn_perc_high_spinbox', 'tooltip_bn_perc_high'),
            ('bn_std_factor_actual_label', 'tooltip_bn_std_factor'), ('bn_std_factor_spinbox', 'tooltip_bn_std_factor'),
            ('bn_min_gain_actual_label', 'tooltip_bn_min_gain'), ('bn_min_gain_spinbox', 'tooltip_bn_min_gain'),
            ('bn_max_gain_actual_label', 'tooltip_bn_max_gain'), ('bn_max_gain_spinbox', 'tooltip_bn_max_gain'),
            
            # Onglet Expert - CB
            ('cb_border_size_actual_label', 'tooltip_cb_border_size'), ('cb_border_size_spinbox', 'tooltip_cb_border_size'),
            ('cb_blur_radius_actual_label', 'tooltip_cb_blur_radius'), ('cb_blur_radius_spinbox', 'tooltip_cb_blur_radius'),
            ('cb_min_b_factor_actual_label', 'tooltip_cb_min_b_factor'), ('cb_min_b_factor_spinbox', 'tooltip_cb_min_b_factor'),
            ('cb_max_b_factor_actual_label', 'tooltip_cb_max_b_factor'), ('cb_max_b_factor_spinbox', 'tooltip_cb_max_b_factor'),

            # Onglet Expert - Crop
            ('final_edge_crop_actual_label', 'tooltip_final_edge_crop_percent'), ('final_edge_crop_spinbox', 'tooltip_final_edge_crop_percent'),
            
            # Onglet Empilement - SCNR Final (accès via _ctrls)
            ('apply_final_scnr_check', 'tooltip_apply_final_scnr'),
            (getattr(self, 'scnr_amount_ctrls', {}).get('label'), 'tooltip_final_scnr_amount'),
            (getattr(self, 'scnr_amount_ctrls', {}).get('spinbox'), 'tooltip_final_scnr_amount'),
            (getattr(self, 'scnr_amount_ctrls', {}).get('slider'), 'tooltip_final_scnr_amount'),
            ('final_scnr_preserve_lum_check', 'tooltip_final_scnr_preserve_lum'),

            # Onglet Expert - Photutils BN
            ('apply_photutils_bn_check', 'tooltip_apply_photutils_bn'),
            ('photutils_bn_box_size_label', 'tooltip_photutils_bn_box_size'), ('pb_box_spinbox', 'tooltip_photutils_bn_box_size'),
            ('photutils_bn_filter_size_label', 'tooltip_photutils_bn_filter_size'), ('pb_filt_spinbox', 'tooltip_photutils_bn_filter_size'),
            ('photutils_bn_sigma_clip_label', 'tooltip_photutils_bn_sigma_clip'), ('pb_sig_spinbox', 'tooltip_photutils_bn_sigma_clip'),
            ('photutils_bn_exclude_percentile_label', 'tooltip_photutils_bn_exclude_percentile'), ('pb_excl_spinbox', 'tooltip_photutils_bn_exclude_percentile'),
            
            # Onglet Expert - Feathering
            ('apply_feathering_check', 'tooltip_apply_feathering'),
            ('feather_blur_px_label', 'tooltip_feather_blur_px'),
            ('feather_blur_px_spinbox', 'tooltip_feather_blur_px'),

            # Onglet Expert - Low WHT Mask
            ('low_wht_mask_check', 'tooltip_apply_low_wht_mask'),
            ('low_wht_pct_label', 'tooltip_low_wht_percentile'),
            ('low_wht_pct_spinbox', 'tooltip_low_wht_percentile'),
            ('low_wht_soften_px_label', 'tooltip_low_wht_soften_px'),
            ('low_wht_soften_px_spinbox', 'tooltip_low_wht_soften_px'),
        ]
        # --- FIN Définition tooltips_config_list ---

        tooltip_created_count = 0
        for item_identifier, tooltip_translation_key in tooltips_config_list: # Maintenant tooltips_config_list est défini
            widget_to_attach_tooltip = None
            debug_item_name = "" # Initialiser

            if isinstance(item_identifier, str):
                debug_item_name = item_identifier
                widget_to_attach_tooltip = getattr(self, item_identifier, None)
            elif isinstance(item_identifier, tk.Widget):
                widget_to_attach_tooltip = item_identifier
                try:
                    debug_item_name = f"WidgetDirect({widget_to_attach_tooltip.winfo_class()}-{id(widget_to_attach_tooltip)})"
                except: 
                    debug_item_name = f"WidgetDirect(id-{id(widget_to_attach_tooltip)})"
            else:
                debug_item_name = str(item_identifier)
                print(f"  Tooltip WARNING: Type d'identifiant d'item inattendu '{debug_item_name}' pour la clé tooltip '{tooltip_translation_key}'.")
                continue 
            
            if widget_to_attach_tooltip and hasattr(widget_to_attach_tooltip, 'winfo_exists') and widget_to_attach_tooltip.winfo_exists():
                unique_tooltip_id = f"tooltip_for_widget_{id(widget_to_attach_tooltip)}_{tooltip_translation_key}"
                
                if unique_tooltip_id not in self.tooltips:
                    self.tooltips[unique_tooltip_id] = ToolTip(widget_to_attach_tooltip, lambda k=tooltip_translation_key: self.tr(k))
                    tooltip_created_count += 1
                    # Décommenter pour un log très détaillé de la création des tooltips:
                    # print(f"  ---> TOOLTIP CREATED for '{tooltip_translation_key}' on '{debug_item_name}' (Widget: {widget_to_attach_tooltip}) -> ID Map: {unique_tooltip_id}")
                # else:
                #     print(f"  ---> TOOLTIP SKIPPED for '{tooltip_translation_key}' on '{debug_item_name}' - ID Map '{unique_tooltip_id}' already exists.")
            # else: # Décommenter pour loguer les widgets non trouvés pour les tooltips
            #     print(f"  -> TOOLTIP DEBUG: Widget pour '{debug_item_name}' (clé de traduction '{tooltip_translation_key}') NON TROUVÉ ou INVALIDE.")
                
        print(f"DEBUG (GUI _store_widget_references): Références et Tooltips stockés. Nb widgets trad: {len(self.widgets_to_translate)}, Nb tooltips créés dans cet appel: {tooltip_created_count}")






##################################################################################################################
    def change_language(self, event=None):
        """ Change l'interface à la langue sélectionnée. """
        selected_lang = self.language_var.get()
        if self.localization.language != selected_lang:
            self.localization.set_language(selected_lang)
            self.settings.language = selected_lang # Update setting
            self.settings.save_settings() # Save the change immediately
            self.update_ui_language()

    def update_ui_language(self):
        """ Met à jour tous les textes traduisibles de l'interface. """
        self.root.title(self.tr("title"))
        if not hasattr(self, 'widgets_to_translate'):
            print("Warning: Widget reference dictionary not found for translation.")
            return
        for key, widget_info in self.widgets_to_translate.items():
                        # --- DÉBUT DEBUG SPÉCIFIQUE ---
            if key == "tab_preview":
                print(f"DEBUG UI_LANG: Traitement clé '{key}'")
                current_lang_for_tr = self.localization.language
                print(f"DEBUG UI_LANG: Langue actuelle pour self.tr: '{current_lang_for_tr}'")
                
                translation_directe_langue_courante = self.localization.translations[current_lang_for_tr].get(key)
                print(f"DEBUG UI_LANG: Traduction directe pour '{key}' en '{current_lang_for_tr}': '{translation_directe_langue_courante}'")
                
                traduction_fallback_anglais = self.localization.translations['en'].get(key)
                print(f"DEBUG UI_LANG: Traduction fallback anglais pour '{key}': '{traduction_fallback_anglais}'")

                default_text_calcul = self.localization.translations['en'].get(key, key.replace("_", " ").title())
                print(f"DEBUG UI_LANG: default_text calculé pour '{key}': '{default_text_calcul}'")

                translation_finale_pour_tab = self.tr(key, default=default_text_calcul)
                print(f"DEBUG UI_LANG: self.tr('{key}') a retourné: '{translation_finale_pour_tab}'")
            # --- FIN DEBUG SPÉCIFIQUE ---
            default_text = self.localization.translations['en'].get(key, key.replace("_", " ").title())
            translation = self.tr(key, default=default_text)
            try:
                if widget_info is None: continue
                if isinstance(widget_info, tuple):
                    notebook, index = widget_info
                    if notebook and notebook.winfo_exists() and index < notebook.index("end"): notebook.tab(index, text=f' {translation} ')
                elif hasattr(widget_info, 'winfo_exists') and widget_info.winfo_exists():
                    widget = widget_info
                    if isinstance(widget, (ttk.Label, ttk.Button, ttk.Checkbutton)): widget.config(text=translation)
                    elif isinstance(widget, ttk.LabelFrame): widget.config(text=translation)
            except tk.TclError: pass
            except Exception as e: print(f"Debug: Error updating widget '{key}': {e}")
        # Update dynamic text variables
        if not self.processing:
            self.remaining_files_var.set(self.tr("no_files_waiting"))
            default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
            self.aligned_files_var.set(default_aligned_fmt.format(count="--"))
            self.remaining_time_var.set("--:--:--")
        else: # Ensure static labels are translated if processing
            if hasattr(self,'remaining_static_label'): self.remaining_static_label.config(text=self.tr("Remaining:"))
            if hasattr(self,'additional_static_label'): self.additional_static_label.config(text=self.tr("Additional:"))
            if hasattr(self,'elapsed_time_label'): self.elapsed_time_label.config(text=self.tr("elapsed_time"))
            if hasattr(self,'remaining_time_label'): self.remaining_time_label.config(text=self.tr("estimated_time"))
            # Update dynamic counts using current language format string
            if hasattr(self, 'queued_stacker'):
                count = self.queued_stacker.aligned_files_count
                default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
                self.aligned_files_var.set(default_aligned_fmt.format(count=count))
                self.update_remaining_files() # Re-calculate R/T display

        self.update_additional_folders_display() # Update folder count display text

        if self.current_preview_data is None and hasattr(self, 'preview_manager'):
            self.preview_manager.clear_preview(self.tr('Select input/output folders.'))

    def _debounce_refresh_preview(self, *args):
        if self.debounce_timer_id:
            try: self.root.after_cancel(self.debounce_timer_id)
            except tk.TclError: pass
        try: self.debounce_timer_id = self.root.after(150, self.refresh_preview)
        except tk.TclError: pass

    def update_histogram_lines_from_sliders(self, *args):
        if hasattr(self, 'histogram_widget') and self.histogram_widget:
            try: bp = self.preview_black_point.get(); wp = self.preview_white_point.get()
            except tk.TclError: return
            self.histogram_widget.set_range(bp, wp)

    def update_stretch_from_histogram(self, black_point, white_point):
        try: self.preview_black_point.set(round(black_point, 4)); self.preview_white_point.set(round(white_point, 4))
        except tk.TclError: return
        try:
            if hasattr(self, 'stretch_bp_ctrls'): self.stretch_bp_ctrls['slider'].set(black_point)
            if hasattr(self, 'stretch_wp_ctrls'): self.stretch_wp_ctrls['slider'].set(white_point)
        except tk.TclError: pass
        self._debounce_refresh_preview()

    def refresh_preview(self):
        if self.debounce_timer_id:
            try: self.root.after_cancel(self.debounce_timer_id)
            except tk.TclError: pass
            self.debounce_timer_id = None
        if (self.current_preview_data is None or not hasattr(self, 'preview_manager') or self.preview_manager is None or
           not hasattr(self, 'histogram_widget') or self.histogram_widget is None):
            if (not self.processing and self.input_path.get() and os.path.isdir(self.input_path.get())): self._try_show_first_input_image()
            else:
                if hasattr(self, 'preview_manager') and self.preview_manager: self.preview_manager.clear_preview(self.tr('Select input/output folders.'))
                if hasattr(self, 'histogram_widget') and self.histogram_widget: self.histogram_widget.plot_histogram(None)
            return
        try:
            preview_params = {
                "stretch_method": self.preview_stretch_method.get(), "black_point": self.preview_black_point.get(),
                "white_point": self.preview_white_point.get(), "gamma": self.preview_gamma.get(),
                "r_gain": self.preview_r_gain.get(), "g_gain": self.preview_g_gain.get(), "b_gain": self.preview_b_gain.get(),
                "brightness": self.preview_brightness.get(), "contrast": self.preview_contrast.get(), "saturation": self.preview_saturation.get(),
            }
        except tk.TclError: print("Error getting preview parameters from UI."); return
        processed_pil_image, data_for_histogram = self.preview_manager.update_preview(self.current_preview_data, preview_params, stack_count=self.preview_img_count, total_images=self.preview_total_imgs, current_batch=self.preview_current_batch, total_batches=self.preview_total_batches)
        self.histogram_widget.update_histogram(data_for_histogram)
        self.histogram_widget.set_range(preview_params["black_point"], preview_params["white_point"])




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
            if hasattr(self, 'bn_grid_size_str_var'):
                self.bn_grid_size_str_var.set(default_settings.bn_grid_size_str)
            if hasattr(self, 'bn_perc_low_var'):
                self.bn_perc_low_var.set(default_settings.bn_perc_low)
            if hasattr(self, 'bn_perc_high_var'):
                self.bn_perc_high_var.set(default_settings.bn_perc_high)
            if hasattr(self, 'bn_std_factor_var'):
                self.bn_std_factor_var.set(default_settings.bn_std_factor)
            if hasattr(self, 'bn_min_gain_var'):
                self.bn_min_gain_var.set(default_settings.bn_min_gain)
            if hasattr(self, 'bn_max_gain_var'):
                self.bn_max_gain_var.set(default_settings.bn_max_gain)

            # ChromaticBalancer (CB)
            if hasattr(self, 'cb_border_size_var'):
                self.cb_border_size_var.set(default_settings.cb_border_size)
            if hasattr(self, 'cb_blur_radius_var'):
                self.cb_blur_radius_var.set(default_settings.cb_blur_radius)
            if hasattr(self, 'cb_min_b_factor_var'): 
                self.cb_min_b_factor_var.set(default_settings.cb_min_b_factor)
            if hasattr(self, 'cb_max_b_factor_var'): 
                self.cb_max_b_factor_var.set(default_settings.cb_max_b_factor)

            # Rognage Final
            if hasattr(self, 'final_edge_crop_percent_var'):
                self.final_edge_crop_percent_var.set(default_settings.final_edge_crop_percent)
            
            # --- Réinitialiser Feathering ---
            if hasattr(self, 'apply_feathering_var'):
                self.apply_feathering_var.set(default_settings.apply_feathering) # Sera False par défaut
            if hasattr(self, 'feather_blur_px_var'):
                self.feather_blur_px_var.set(default_settings.feather_blur_px)   # Sera 256 par défaut
            # ---  ---

            # --- Réinitialiser Photutils BN ---
            if hasattr(self, 'apply_photutils_bn_var'):
                self.apply_photutils_bn_var.set(default_settings.apply_photutils_bn) # Sera False par défaut
            if hasattr(self, 'photutils_bn_box_size_var'):
                self.photutils_bn_box_size_var.set(default_settings.photutils_bn_box_size)
            if hasattr(self, 'photutils_bn_filter_size_var'):
                self.photutils_bn_filter_size_var.set(default_settings.photutils_bn_filter_size)
            if hasattr(self, 'photutils_bn_sigma_clip_var'):
                self.photutils_bn_sigma_clip_var.set(default_settings.photutils_bn_sigma_clip)
            if hasattr(self, 'photutils_bn_exclude_percentile_var'):
                self.photutils_bn_exclude_percentile_var.set(default_settings.photutils_bn_exclude_percentile)
            # ---  ---
            
            # Mettre à jour l'état des widgets après réinitialisation
            # C'est important que ces appels soient APRÈS avoir .set() les BooleanVar
            if hasattr(self, '_update_photutils_bn_options_state'):
                self._update_photutils_bn_options_state()
            if hasattr(self, '_update_feathering_options_state'):
                self._update_feathering_options_state()
            # Si d'autres groupes d'options dans l'onglet expert ont des états dépendants,
            # appelez leurs méthodes _update_..._state() ici aussi.

            self.update_progress_gui("ⓘ Réglages Expert réinitialisés aux valeurs par défaut.", None)
            print("DEBUG (GUI _reset_expert_settings): Paramètres Expert réinitialisés dans l'UI.")

        except tk.TclError as e:
            print(f"ERREUR (GUI _reset_expert_settings): Erreur Tcl lors de la réinitialisation des widgets: {e}")
        except AttributeError as e:
            print(f"ERREUR (GUI _reset_expert_settings): Erreur d'attribut (widget ou variable Tk manquant?): {e}")
            traceback.print_exc(limit=1) # Pour voir quel attribut manque
        except Exception as e:
            print(f"ERREUR (GUI _reset_expert_settings): Erreur inattendue: {e}")
            traceback.print_exc(limit=1)



###########################################################################################################################################

    def update_preview_from_stacker(self, stack_data, stack_header, stack_name, img_count, total_imgs, current_batch, total_batches):
        """Callback function triggered by the backend worker."""
        print("[DEBUG-HISTO] update_preview_from_stacker called.") # Log entry

        if stack_data is None:
            print("[DEBUG-HISTO] Received None stack_data. Skipping update.")
            # Optional: Clear display if None is received consistently
            # if hasattr(self, 'preview_manager'): self.preview_manager.clear_preview("No Data Received")
            # if hasattr(self, 'histogram_widget'): self.histogram_widget.plot_histogram(None)
            return

        # --- Log received data info ---
        try:
            # Use np.nanmin and np.nanmax to handle potential NaNs safely
            print(f"[DEBUG-HISTO] Received stack_data - Shape: {stack_data.shape}, Type: {stack_data.dtype}, Min: {np.nanmin(stack_data):.4f}, Max: {np.nanmax(stack_data):.4f}")
        except Exception as e:
            print(f"[DEBUG-HISTO] Error getting info for received stack_data: {e}")
        # -----------------------------

        # Update the main data store immediately
        self.current_preview_data = stack_data.copy()
        self.current_stack_header = stack_header.copy() if stack_header else None

        # Update internal counters used by PreviewManager text overlay
        self.preview_img_count = img_count
        self.preview_total_imgs = total_imgs
        self.preview_current_batch = current_batch
        self.preview_total_batches = total_batches
        # --- NOUVELLE LOGIQUE POUR AUTO-REFRESH PÉRIODIQUE ---
        self.batches_processed_for_preview_refresh += 1
        print(f"DEBUG GUI: Preview refresh counter: {self.batches_processed_for_preview_refresh}/{self.preview_auto_refresh_batch_interval}")

        # Mettre à jour les infos texte de l'aperçu immédiatement
        # (elles ne dépendent pas des auto-stretch/wb)
        if hasattr(self.preview_manager, 'trigger_redraw'): # Pour redessiner le texte
            try: self.root.after_idle(self.preview_manager.trigger_redraw)
            except tk.TclError: pass


        if self.batches_processed_for_preview_refresh >= self.preview_auto_refresh_batch_interval:
            print(f"DEBUG GUI: Seuil de {self.preview_auto_refresh_batch_interval} lots atteint. Déclenchement Auto WB & Auto Stretch pour l'aperçu.")
            self.update_progress_gui("ⓘ Auto-ajustement de l'aperçu...", None)
            
            # Appeler Auto WB. Cela va .set() les variables des sliders et déclencher _debounce_refresh_preview
            self.apply_auto_white_balance() 
            
            # Appeler Auto Stretch. Cela va .set() les variables des sliders et déclencher _debounce_refresh_preview
            # Il est important que apply_auto_stretch utilise les données après la potentielle nouvelle WB
            # ce qui est le cas car apply_auto_stretch utilise self.preview_manager.image_data_wb
            # ou recalcule la WB si image_data_wb est None.
            # Pour être sûr, on peut forcer un refresh_preview avant l'auto_stretch
            # pour que image_data_wb soit à jour, mais les appels set() devraient suffire.
            
            # Un petit délai pour s'assurer que la WB est appliquée avant l'auto-stretch
            # qui se base sur les données après WB pour son analyse de luminance.
            self.root.after(50, self.apply_auto_stretch) 
            
            self.batches_processed_for_preview_refresh = 0 # Réinitialiser le compteur
        else:
            # Si pas d'auto-ajustement, rafraîchir simplement avec les réglages UI actuels
            # print("DEBUG GUI: Pas d'auto-ajustement ce lot, refresh normal.")
            self.refresh_preview() # Déclenche le pipeline de PreviewManager avec les réglages actuels
        # --- FIN NOUVELLE LOGIQUE ---

        # Get current preview parameters from UI
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
            print("[DEBUG-HISTO] Error getting preview parameters from UI during callback.")
            return

        # Schedule the update using after_idle
        try:
            def combined_update_tasks():
                print("[DEBUG-HISTO] combined_update_tasks running via after_idle.") # Log task start
                # Ensure widgets exist before proceeding
                if not hasattr(self, 'preview_manager') or not self.preview_manager.canvas.winfo_exists():
                    print("[DEBUG-HISTO] PreviewManager or canvas not found/exists in combined_update_tasks.")
                    return
                if not hasattr(self, 'histogram_widget') or not self.histogram_widget.winfo_exists():
                    print("[DEBUG-HISTO] HistogramWidget not found/exists in combined_update_tasks.")
                    return

                try:
                    # 1. Update the preview image (this also calculates histogram data)
                    print("[DEBUG-HISTO] Calling preview_manager.update_preview...")
                    # Ensure self.current_preview_data is valid before calling
                    if self.current_preview_data is None:
                        print("[DEBUG-HISTO] self.current_preview_data is None in combined_update_tasks. Aborting update.")
                        return

                    processed_pil_image, data_for_histogram = self.preview_manager.update_preview(
                        self.current_preview_data, # Use the data stored in the GUI instance
                        preview_params,
                        stack_count=img_count,
                        total_images=total_imgs,
                        current_batch=current_batch,
                        total_batches=total_batches
                    )
                    print("[DEBUG-HISTO] Returned from preview_manager.update_preview.")

                    # --- Log histogram data info ---
                    if data_for_histogram is not None:
                        try:
                            # Use np.nanmin/nanmax again
                            print(f"[DEBUG-HISTO] data_for_histogram - Shape: {data_for_histogram.shape}, Type: {data_for_histogram.dtype}, Min: {np.nanmin(data_for_histogram):.4f}, Max: {np.nanmax(data_for_histogram):.4f}")
                        except Exception as e:
                            print(f"[DEBUG-HISTO] Error getting info for data_for_histogram: {e}")
                    else:
                        print("[DEBUG-HISTO] data_for_histogram is None.")
                    # -----------------------------

                    # 2. Update the histogram widget with the returned data
                    if self.histogram_widget: # Check again just in case
                        print("[DEBUG-HISTO] Calling histogram_widget.update_histogram...")
                        self.histogram_widget.update_histogram(data_for_histogram)
                        print("[DEBUG-HISTO] Returned from histogram_widget.update_histogram.")
                        # Optional: Reset histogram zoom/range if needed, but usually not desired here
                        # self.histogram_widget.set_range(preview_params["black_point"], preview_params["white_point"])
                    else:
                        print("[DEBUG-HISTO] histogram_widget is None, cannot update.")


                except Exception as update_err:
                    print(f"[DEBUG-HISTO] Error during scheduled preview/histogram update: {update_err}")
                    traceback.print_exc(limit=2)
                    # Clear histogram on error
                    if hasattr(self, 'histogram_widget') and self.histogram_widget.winfo_exists():
                        self.histogram_widget.plot_histogram(None)

            # Schedule the single combined update function
            self.root.after_idle(combined_update_tasks)
            print("[DEBUG-HISTO] after_idle scheduled for combined_update_tasks.") # Log scheduling

        except tk.TclError:
            pass # Ignore if root window is destroyed

        # Schedule image info update separately (this is fine)
        if self.current_stack_header:
            try:
                self.root.after_idle(lambda h=self.current_stack_header: self.update_image_info(h))
            except tk.TclError:
                pass

    def refresh_preview(self):
        """Refreshes the preview based on current data and UI settings."""
        print("[DEBUG-HISTO] refresh_preview called (manual trigger/debounce/resize).") # Log entry

        # --- Debounce Timer Cancellation ---
        if self.debounce_timer_id:
            try:
                self.root.after_cancel(self.debounce_timer_id)
            except tk.TclError:
                pass
            self.debounce_timer_id = None
        # -----------------------------------

        # --- Check if data and managers exist ---
        if (self.current_preview_data is None or
                not hasattr(self, 'preview_manager') or self.preview_manager is None or
                not hasattr(self, 'histogram_widget') or self.histogram_widget is None):

            print("[DEBUG-HISTO] refresh_preview: No data or managers. Checking for first input image.")
            # If no data but input folder is set, try showing the first image
            if (not self.processing and self.input_path.get() and os.path.isdir(self.input_path.get())):
                self._try_show_first_input_image() # This might call refresh_preview again if successful
            else:
                # Clear display if no data and cannot load first image
                if hasattr(self, 'preview_manager') and self.preview_manager:
                    self.preview_manager.clear_preview(self.tr('Select input/output folders.'))
                if hasattr(self, 'histogram_widget') and self.histogram_widget:
                    self.histogram_widget.plot_histogram(None)
            return # Exit if no data to process
        # -----------------------------------------

        # --- Get current preview parameters from UI ---
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
            print("[DEBUG-HISTO] refresh_preview: Error getting preview parameters from UI.")
            return
        # ------------------------------------------

        # --- Call PreviewManager to update the image and get histogram data ---
        try:
            print("[DEBUG-HISTO] refresh_preview: Calling preview_manager.update_preview...")
            processed_pil_image, data_for_histogram = self.preview_manager.update_preview(
                self.current_preview_data,
                preview_params,
                stack_count=self.preview_img_count,
                total_images=self.preview_total_imgs,
                current_batch=self.preview_current_batch,
                total_batches=self.preview_total_batches
            )
            print("[DEBUG-HISTO] refresh_preview: Returned from preview_manager.update_preview.")

            # --- Log histogram data info ---
            if data_for_histogram is not None:
                 try:
                     print(f"[DEBUG-HISTO] refresh_preview: data_for_histogram - Shape: {data_for_histogram.shape}, Type: {data_for_histogram.dtype}, Min: {np.nanmin(data_for_histogram):.4f}, Max: {np.nanmax(data_for_histogram):.4f}")
                 except Exception as e:
                     print(f"[DEBUG-HISTO] refresh_preview: Error getting info for data_for_histogram: {e}")
            else:
                 print("[DEBUG-HISTO] refresh_preview: data_for_histogram is None.")
            # -----------------------------

            # --- Update the histogram widget ---
            if self.histogram_widget: # Check widget exists
                print("[DEBUG-HISTO] refresh_preview: Calling histogram_widget.update_histogram...")
                self.histogram_widget.update_histogram(data_for_histogram)
                print("[DEBUG-HISTO] refresh_preview: Returned from histogram_widget.update_histogram.")
                # Also update the slider lines on the histogram to match the UI values
                self.histogram_widget.set_range(preview_params["black_point"], preview_params["white_point"])
            else:
                print("[DEBUG-HISTO] refresh_preview: histogram_widget is None, cannot update.")

        except Exception as e:
            print(f"[DEBUG-HISTO] Error during refresh_preview processing: {e}")
            traceback.print_exc(limit=2)
            # Attempt to clear histogram on error
            if hasattr(self, 'histogram_widget') and self.histogram_widget:
                self.histogram_widget.plot_histogram(None)
        # -----------------------------------------------------------------------

    def update_image_info(self, header):
        if not header or not hasattr(self, 'preview_manager'):
            return
        info_lines = []
        keys_labels = {
            'OBJECT': 'Object',
            'DATE-OBS': 'Date',
            'EXPTIME': 'Exp (s)',
            'GAIN': 'Gain',
            'OFFSET': 'Offset',
            'CCD-TEMP': 'Temp (°C)',
            'NIMAGES': 'Images',
            'STACKTYP': 'Method',
            'FILTER': 'Filter',
            'BAYERPAT': 'Bayer',
            'TOTEXP': 'Total Exp (s)',
            'ALIGNED': 'Aligned',
            'FAILALIGN': 'Failed Align',
            'FAILSTACK': 'Failed Stack',
            'SKIPPED': 'Skipped',
            'WGHT_ON': 'Weighting',
            'WGHT_MET': 'W. Metrics',
        }
        for key, label_key in keys_labels.items():
            label = self.tr(label_key, default=label_key)
            value = header.get(key)
            if value is not None and str(value).strip() != '':
                s_value = str(value)
                if key == 'DATE-OBS':
                    s_value = s_value.split('T')[0]
                elif key in ['EXPTIME', 'CCD-TEMP', 'TOTEXP'] and isinstance(value, (int, float)):
                    try:
                        s_value = f"{float(value):.1f}"
                    except ValueError:
                        pass
                elif key == 'KAPPA' and isinstance(value, (int, float)):
                    try:
                        s_value = f"{float(value):.2f}"
                    except ValueError:
                        pass
                elif key == 'WGHT_ON':
                    s_value = self.tr('weighting_enabled')
                else:
                    if value:
                        pass  # La condition est toujours vraie ici, l'indentation suivante était incorrecte
                    else:
                        s_value = self.tr('weighting_disabled')
                info_lines.append(f"{label}: {s_value}")
        info_text = "\n".join(info_lines) if info_lines else self.tr('No image info available')
        if hasattr(self.preview_manager, 'update_info_text'):
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

        if not hasattr(self, 'preview_manager') or not hasattr(self, 'histogram_widget'):
            print("  WARN GUI (_try_show_first_input_image): PreviewManager ou HistogramWidget manquant.")
            return

        if not input_folder or not os.path.isdir(input_folder):
            print(f"  DEBUG GUI (_try_show_first_input_image): Dossier d'entrée non valide ou non défini ('{input_folder}'). Effacement aperçu.")
            if hasattr(self, 'preview_manager') and self.preview_manager: self.preview_manager.clear_preview(self.tr("Input folder not found or not set"))
            if hasattr(self, 'histogram_widget') and self.histogram_widget: self.histogram_widget.plot_histogram(None)
            return
        
        try:
            files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith((".fit", ".fits"))])
            if not files:
                print(f"  DEBUG GUI (_try_show_first_input_image): Aucun fichier FITS trouvé dans '{input_folder}'. Effacement aperçu.")
                if hasattr(self, 'preview_manager') and self.preview_manager: self.preview_manager.clear_preview(self.tr("No FITS files in input folder"))
                if hasattr(self, 'histogram_widget') and self.histogram_widget: self.histogram_widget.plot_histogram(None)
                return

            first_image_filename = files[0]
            first_image_path = os.path.join(input_folder, first_image_filename)
            self.update_progress_gui(f"{self.tr('Loading preview for', default='Loading preview')}: {first_image_filename}...", None)
            print(f"  DEBUG GUI (_try_show_first_input_image): Chargement de '{first_image_path}'...")

            # --- MODIFICATION ICI pour déballer le tuple ---
            loaded_data_tuple = load_and_validate_fits(first_image_path) 
            
            img_data_from_load = None # Doit être défini avant le if
            header_from_load = None   # Doit être défini avant le if

            if loaded_data_tuple is not None and loaded_data_tuple[0] is not None:
                img_data_from_load, header_from_load = loaded_data_tuple # Déballer
                print(f"  DEBUG GUI (_try_show_first_input_image): load_and_validate_fits OK. Shape données: {img_data_from_load.shape}")
            else:
                raise ValueError(f"Échec chargement/validation de '{first_image_filename}' par load_and_validate_fits (retour None ou données None).")
            # --- FIN MODIFICATION ---

            img_for_preview = img_data_from_load 
            
            if img_for_preview.ndim == 2:
                bayer_pattern_from_header = header_from_load.get("BAYERPAT", self.settings.bayer_pattern) if header_from_load else self.settings.bayer_pattern
                valid_bayer_patterns = ["GRBG", "RGGB", "GBRG", "BGGR"]
                
                if isinstance(bayer_pattern_from_header, str) and bayer_pattern_from_header.upper() in valid_bayer_patterns:
                    print(f"  DEBUG GUI (_try_show_first_input_image): Debayering aperçu initial (Pattern: {bayer_pattern_from_header.upper()})...")
                    try: 
                        img_for_preview = debayer_image(img_for_preview, bayer_pattern_from_header.upper())
                    except ValueError as debayer_err: 
                        self.update_progress_gui(f"⚠️ {self.tr('Error during debayering')}: {debayer_err}. Affichage N&B.", None)
                        print(f"    WARN GUI: Erreur Debayer aperçu initial: {debayer_err}. Affichage N&B.")
                else:
                    print(f"  DEBUG GUI (_try_show_first_input_image): Pas de debayering pour aperçu (pas de pattern Bayer valide ou image supposée déjà couleur).")
            
            self.current_preview_data = img_for_preview.copy()
            self.current_stack_header = header_from_load.copy() if header_from_load else fits.Header() 
            
            print(f"  DEBUG GUI (_try_show_first_input_image): Données prêtes pour refresh_preview. Shape: {self.current_preview_data.shape}")
            self.refresh_preview() 

            if self.current_stack_header:
                self.update_image_info(self.current_stack_header)
            
            self.update_progress_gui(f"{self.tr('Preview loaded', default='Preview loaded')}: {first_image_filename}", None)

        except FileNotFoundError:
            print(f"  ERREUR GUI (_try_show_first_input_image): FileNotFoundError pour '{input_folder}'.")
            if hasattr(self, 'preview_manager') and self.preview_manager: self.preview_manager.clear_preview(self.tr("Input folder not found or inaccessible"))
            if hasattr(self, 'histogram_widget') and self.histogram_widget: self.histogram_widget.plot_histogram(None)
        except ValueError as ve: 
            self.update_progress_gui(f"⚠️ {self.tr('Error loading preview image')}: {ve}", None)
            print(f"  ERREUR GUI (_try_show_first_input_image): ValueError - {ve}")
            if hasattr(self, 'preview_manager') and self.preview_manager: self.preview_manager.clear_preview(self.tr("Error loading preview (invalid format?)"))
            if hasattr(self, 'histogram_widget') and self.histogram_widget: self.histogram_widget.plot_histogram(None)
        except Exception as e:
            self.update_progress_gui(f"⚠️ {self.tr('Error loading preview image')}: {e}", None)
            print(f"  ERREUR GUI (_try_show_first_input_image): Exception inattendue - {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            if hasattr(self, 'preview_manager') and self.preview_manager: self.preview_manager.clear_preview(self.tr("Error loading preview"))
            if hasattr(self, 'histogram_widget') and self.histogram_widget: self.histogram_widget.plot_histogram(None)


################################################################################################################################################################


    def apply_auto_white_balance(self):
        if not _tools_available: messagebox.showerror(self.tr("error"), "Stretch/Color tools not available."); return
        if self.current_preview_data is None or self.current_preview_data.ndim != 3: messagebox.showwarning(self.tr("warning"), self.tr("Auto WB requires a color image preview.")); return
        try:
            r_gain, g_gain, b_gain = calculate_auto_wb(self.current_preview_data)
            self.preview_r_gain.set(round(r_gain, 3)); self.preview_g_gain.set(round(g_gain, 3)); self.preview_b_gain.set(round(b_gain, 3))
            self.update_progress_gui(f"Auto WB appliqué (Aperçu): R={r_gain:.2f} G={g_gain:.2f} B={b_gain:.2f}", None); self.refresh_preview()
        except Exception as e: messagebox.showerror(self.tr("error"), f"{self.tr('Error during Auto WB')}: {e}"); traceback.print_exc(limit=2)

    def reset_white_balance(self): self.preview_r_gain.set(1.0); self.preview_g_gain.set(1.0); self.preview_b_gain.set(1.0); self.refresh_preview()
    def reset_brightness_contrast_saturation(self): self.preview_brightness.set(1.0); self.preview_contrast.set(1.0); self.preview_saturation.set(1.0); self.refresh_preview()

    def apply_auto_stretch(self):
        if not _tools_available: messagebox.showerror(self.tr("error"), "Stretch tools not available."); return
        data_to_analyze = None
        if hasattr(self, 'preview_manager') and self.preview_manager.image_data_wb is not None: data_to_analyze = self.preview_manager.image_data_wb
        elif self.current_preview_data is not None:
            print("Warning AutoStretch: Using current WB settings for analysis.")
            if self.current_preview_data.ndim == 3:
                try: r=self.preview_r_gain.get(); g=self.preview_g_gain.get(); b=self.preview_b_gain.get(); data_to_analyze = ColorCorrection.white_balance(self.current_preview_data, r, g, b)
                except Exception: data_to_analyze = self.current_preview_data
            else: data_to_analyze = self.current_preview_data
        else: messagebox.showwarning(self.tr("warning"), self.tr("Auto Stretch requires an image preview.")); return
        try:
            bp, wp = calculate_auto_stretch(data_to_analyze)
            self.preview_black_point.set(round(bp, 4)); self.preview_white_point.set(round(wp, 4))
            if hasattr(self, 'histogram_widget'): self.histogram_widget.set_range(bp, wp)
            self.update_progress_gui(f"Auto Stretch appliqué (Aperçu): BP={bp:.3f} WP={wp:.3f}", None); self.refresh_preview()
        except Exception as e: messagebox.showerror(self.tr("error"), f"{self.tr('Error during Auto Stretch')}: {e}"); traceback.print_exc(limit=2)

    def reset_stretch(self):
        default_method = "Asinh"; default_bp = 0.01; default_wp = 0.99; default_gamma = 1.0
        self.preview_stretch_method.set(default_method); self.preview_black_point.set(default_bp); self.preview_white_point.set(default_wp); self.preview_gamma.set(default_gamma)
        if hasattr(self, 'histogram_widget'): self.histogram_widget.set_range(default_bp, default_wp); self.histogram_widget.reset_zoom()
        self.refresh_preview()

    # --- NOUVELLE MÉTHODE pour gérer la requête d'ajout ---
    def handle_add_folder_request(self, folder_path):
        """
        Gère une requête d'ajout de dossier, en l'ajoutant soit à la liste
        pré-démarrage, soit en appelant le backend si le traitement est actif.
        """
        abs_folder = os.path.abspath(folder_path)

        if self.processing and hasattr(self, 'queued_stacker') and self.queued_stacker.is_running():
            # Traitement actif : appeler le backend
            add_success = self.queued_stacker.add_folder(abs_folder)
            if not add_success:
                 messagebox.showwarning(
                     self.tr('warning'),
                     self.tr('Folder not added (already present, invalid path, or error?)', default='Folder not added (already present, invalid path, or error?)')
                 )
            # La mise à jour de l'affichage se fera via callback "folder_count_update" du backend
        else:
            # Traitement non actif : ajouter à la liste pré-démarrage
            if abs_folder not in self.additional_folders_to_process:
                self.additional_folders_to_process.append(abs_folder)
                self.update_progress_gui(f"ⓘ Dossier ajouté pour prochain traitement: {os.path.basename(abs_folder)}", None)
                self.update_additional_folders_display() # Mettre à jour l'affichage UI
            else:
                 messagebox.showinfo(self.tr('info'), self.tr('Folder already added', default='Folder already added to the list.'))



    def _track_processing_progress(self):
        """Monitors the QueuedStacker worker thread and updates GUI stats."""
        # print("DEBUG: GUI Progress Tracker Thread Started.") # Keep disabled unless debugging

        while self.processing and hasattr(self, "queued_stacker"):
            try:
                # Check if the worker thread is still active
                if not self.queued_stacker.is_running():
                    # print("DEBUG: Worker is_running() is False. Preparing to finalize.") # Keep disabled
                    # --- CORRECTED JOIN LOGIC ---
                    # Check and join the *worker* thread object stored in the queued_stacker
                    worker_thread = getattr(self.queued_stacker, 'processing_thread', None)
                    if worker_thread and worker_thread.is_alive():
                        # print("DEBUG: Joining worker thread...") # Keep disabled
                        worker_thread.join(timeout=0.5) # Wait up to 0.5 seconds
                        # if worker_thread.is_alive():
                        #     print("WARN: Worker thread did not exit cleanly after join timeout.") # Keep disabled
                        # else:
                        #     print("DEBUG: Worker thread joined successfully.") # Keep disabled
                    # else:
                         # print("DEBUG: Worker thread object not found or already dead.") # Keep disabled
                    # --- END CORRECTED JOIN LOGIC ---

                    # Now that we've waited, schedule the final GUI update routine
                    # print("DEBUG: Scheduling _processing_finished...") # Keep disabled
                    self.root.after(0, self._processing_finished)
                    break # Exit the monitoring loop

                # --- Update intermediate progress stats (ETA, counts) ---
                q_stacker = self.queued_stacker
                processed = q_stacker.processed_files_count
                aligned = q_stacker.aligned_files_count
                total_queued = q_stacker.files_in_queue

                # Calculate ETA
                if self.global_start_time and processed > 0:
                    elapsed = time.monotonic() - self.global_start_time
                    self.time_per_image = elapsed / processed
                    try:
                        remaining_estimated = max(0, total_queued - processed)
                        if self.time_per_image > 1e-6 and remaining_estimated > 0:
                            eta_seconds = remaining_estimated * self.time_per_image
                            h, rem = divmod(int(eta_seconds), 3600)
                            m, s = divmod(rem, 60)
                            self.remaining_time_var.set(f"{h:02}:{m:02}:{s:02}")
                        elif remaining_estimated == 0 and total_queued > 0:
                            self.remaining_time_var.set("00:00:00") # Finishing up
                        else:
                            self.remaining_time_var.set(self.tr("eta_calculating", default="Calculating..."))
                    except tk.TclError:
                        # print("DEBUG: tk.TclError updating ETA, breaking tracker loop.") # Keep disabled
                        break # Exit loop if Tkinter objects are gone
                    except Exception as eta_err:
                        print(f"Warning: Error calculating ETA: {eta_err}")
                        try:
                             self.remaining_time_var.set("Error")
                        except tk.TclError: break # Exit loop

                else: # Not enough info for ETA yet
                    try:
                        self.remaining_time_var.set(self.tr("eta_calculating", default="Calculating..."))
                    except tk.TclError: break # Exit loop

                # Update Aligned Files Count
                default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
                try:
                    self.aligned_files_var.set(default_aligned_fmt.format(count=aligned))
                except tk.TclError:
                    # print("DEBUG: tk.TclError updating aligned count, breaking tracker loop.") # Keep disabled
                    break # Exit loop

                # Update Remaining/Total Files display
                self.update_remaining_files() # Calls the method to update R/T label

                # Sleep briefly to avoid busy-waiting
                time.sleep(0.5)

            except Exception as e:
                # Catch errors within the tracking loop itself
                print(f"Error in GUI progress tracker thread loop: {e}")
                traceback.print_exc(limit=2)
                # Attempt to gracefully finish processing in case of tracker error
                try:
                    self.root.after(0, self._processing_finished)
                except tk.TclError: pass # Tk might be gone
                break # Exit the tracker loop on error

        # print("DEBUG: GUI Progress Tracker Thread Exiting.") # Keep disabled
 
    def update_remaining_files(self):
        """Met à jour l'affichage des fichiers restants / total ajouté."""
        if hasattr(self, "queued_stacker") and self.processing:
            try:
                total_queued = self.queued_stacker.files_in_queue; processed_total = self.queued_stacker.processed_files_count
                remaining_estimated = max(0, total_queued - processed_total)
                self.remaining_files_var.set(f"{remaining_estimated}/{total_queued}")
            except tk.TclError: pass
            except AttributeError:
                try: self.remaining_files_var.set(self.tr("no_files_waiting"))
                except tk.TclError: pass
            except Exception as e: print(f"Error updating remaining files display: {e}")
            try: self.remaining_files_var.set("Error")
            except tk.TclError: pass
        elif not self.processing:
             try: self.remaining_files_var.set(self.tr("no_files_waiting"))
             except tk.TclError: pass





    # --- MODIFIER CETTE MÉTHODE ---



    def update_additional_folders_display(self):
        """Met à jour l'affichage du nombre de dossiers supplémentaires."""
        count = 0
        # --- AJOUT DEBUG ---
        print("-" * 20)
        print("DEBUG MW (update_additional_folders_display): Entrée fonction.")
        if hasattr(self, 'queued_stacker'):
            print(f"  -> self.queued_stacker existe. Type: {type(self.queued_stacker)}")
            # VÉRIFICATION CRUCIALE :
            has_lock = hasattr(self.queued_stacker, 'folders_lock')
            print(f"  -> self.queued_stacker a l'attribut 'folders_lock'? {has_lock}")
            if not has_lock:
                 print("  -> !!! ATTRIBUT 'folders_lock' MANQUANT SUR L'INSTANCE !!!")
                 print(f"  -> Attributs présents: {dir(self.queued_stacker)}") # Lister ce qui est présent
            # --- FIN AJOUT DEBUG ---
     
        print("-" * 20)
        print("DEBUG MW (update_additional_folders_display): Entrée fonction.")
        if hasattr(self, 'queued_stacker'):
            print(f"  -> self.queued_stacker existe. Type: {type(self.queued_stacker)}")
            print(f"  -> Attributs de self.queued_stacker: {dir(self.queued_stacker)}") # AFFICHE TOUS LES ATTRIBUTS
            has_is_running_method = hasattr(self.queued_stacker, 'is_running')
            print(f"  -> self.queued_stacker a l'attribut 'is_running'? {has_is_running_method}")
            if has_is_running_method:
                print(f"  -> Type de self.queued_stacker.is_running: {type(self.queued_stacker.is_running)}")
     
            # Condition originale pour lire depuis le backend
            if self.processing and self.queued_stacker.is_running(): # Ajout check is_running pour sécurité
                try:
                    # L'accès problématique
                    with self.queued_stacker.folders_lock: # <<< C'est ici que ça plante
                         count = len(self.queued_stacker.additional_folders)
                    print(f"  -> Lecture backend (processing): count={count}") # Si ça passe le 'with'
                except AttributeError as ae:
                     print(f"  -> ERREUR ATTRIBUT DANS LE 'WITH': {ae}") # Log spécifique si ça plante DANS le with
                     traceback.print_exc(limit=1) # Afficher où ça plante
                     # Fallback pour ne pas planter l'UI
                     count = -99 # Valeur pour indiquer une erreur
                except Exception as e:
                     print(f"  -> ERREUR PENDANT lecture backend: {e}")
                     traceback.print_exc(limit=1)
                     count = -98
            else:
                # Lire depuis la liste GUI (traitement non actif)
                count = len(self.additional_folders_to_process)
                print(f"  -> Lecture GUI (non processing): count={count}")
        else:
             print("  -> self.queued_stacker n'existe PAS.")
             count = len(self.additional_folders_to_process) # Fallback liste GUI
        print("-" * 20)

        # Mise à jour de la variable Tkinter (inchangé)
        try:
            if count == 0: self.additional_folders_var.set(self.tr('no_additional_folders'))
            elif count == 1: self.additional_folders_var.set(self.tr('1 additional folder'))
            # Gérer les comptes d'erreur négatifs pour le debug
            elif count < 0 : self.additional_folders_var.set(f"ERR ({count})")
            else: self.additional_folders_var.set(self.tr('{count} additional folders', default="{count} add. folders").format(count=count))
        except tk.TclError: pass
        except AttributeError: pass
    # --- FIN MÉTHODE MODIFIÉE ---


    def stop_processing(self):
        if self.processing and hasattr(self, "queued_stacker") and self.queued_stacker.is_running():
            self.update_progress_gui(self.tr("stacking_stopping"), None); self.queued_stacker.stop()
            if hasattr(self,'stop_button'): self.stop_button.config(state=tk.DISABLED)
        elif self.processing:
            self.update_progress_gui("Tentative d'arrêt, mais worker inactif ou déjà arrêté.", None); self._processing_finished()

    def _format_duration(self, seconds):
        try:
            secs = int(round(float(seconds)));
            if secs < 0: return "N/A"
            if secs < 60: return f"{secs} {self.tr('report_seconds', 's')}"
            elif secs < 3600: m, s = divmod(secs, 60); return f"{m} {self.tr('report_minutes', 'min')} {s} {self.tr('report_seconds', 's')}"
            else: h, rem = divmod(secs, 3600); m, s = divmod(rem, 60); return f"{h} {self.tr('report_hours', 'h')} {m} {self.tr('report_minutes', 'min')} {s} {self.tr('report_seconds', 's')}"
        except (ValueError, TypeError): return "N/A"

    def _set_parameter_widgets_state(self, state):
        """Enable/disable control widgets."""
        processing_widgets = []
        if hasattr(self, 'input_entry'): processing_widgets.append(self.input_entry)
        if hasattr(self, 'browse_input_button'): processing_widgets.append(self.browse_input_button)
        if hasattr(self, 'output_entry'): processing_widgets.append(self.output_entry)
        if hasattr(self, 'browse_output_button'): processing_widgets.append(self.browse_output_button)
        if hasattr(self, 'ref_entry'): processing_widgets.append(self.ref_entry)
        if hasattr(self, 'browse_ref_button'): processing_widgets.append(self.browse_ref_button)
        if hasattr(self, 'stacking_combo'): processing_widgets.append(self.stacking_combo)
        if hasattr(self, 'kappa_spinbox'): processing_widgets.append(self.kappa_spinbox)
        if hasattr(self, 'batch_spinbox'): processing_widgets.append(self.batch_spinbox)
        if hasattr(self, 'hot_pixels_check'): processing_widgets.append(self.hot_pixels_check)
        if hasattr(self, 'hp_thresh_spinbox'): processing_widgets.append(self.hp_thresh_spinbox)
        if hasattr(self, 'hp_neigh_spinbox'): processing_widgets.append(self.hp_neigh_spinbox)
        if hasattr(self, 'cleanup_temp_check'): processing_widgets.append(self.cleanup_temp_check)
        #--- AJOUT DE CHROMA CORRECTION ---
        if hasattr(self, 'chroma_correction_check'): processing_widgets.append(self.chroma_correction_check)
        # ---  ---
        if hasattr(self, 'language_combo'): processing_widgets.append(self.language_combo)
        if hasattr(self, 'use_weighting_check'): processing_widgets.append(self.use_weighting_check)
        if hasattr(self, 'weight_snr_check'): processing_widgets.append(self.weight_snr_check)
        if hasattr(self, 'weight_stars_check'): processing_widgets.append(self.weight_stars_check)
        if hasattr(self, 'snr_exp_spinbox'): processing_widgets.append(self.snr_exp_spinbox)
        if hasattr(self, 'stars_exp_spinbox'): processing_widgets.append(self.stars_exp_spinbox)
        if hasattr(self, 'min_w_spinbox'): processing_widgets.append(self.min_w_spinbox)
        if hasattr(self, 'drizzle_check'): processing_widgets.append(self.drizzle_check)
        if hasattr(self, 'drizzle_scale_label'): processing_widgets.append(self.drizzle_scale_label)
        if hasattr(self, 'drizzle_radio_2x'): processing_widgets.append(self.drizzle_radio_2x) # Si Radiobuttons
        if hasattr(self, 'drizzle_radio_3x'): processing_widgets.append(self.drizzle_radio_3x) # Si Radiobuttons
        if hasattr(self, 'drizzle_radio_4x'): processing_widgets.append(self.drizzle_radio_4x) # Si Radiobuttons
        # if hasattr(self, 'drizzle_scale_combo'): processing_widgets.append(self.drizzle_scale_combo) # Si Combobox


        preview_widgets = []
        if hasattr(self, 'wb_r_ctrls'): preview_widgets.extend([self.wb_r_ctrls['slider'], self.wb_r_ctrls['spinbox']])
        if hasattr(self, 'wb_g_ctrls'): preview_widgets.extend([self.wb_g_ctrls['slider'], self.wb_g_ctrls['spinbox']])
        if hasattr(self, 'wb_b_ctrls'): preview_widgets.extend([self.wb_b_ctrls['slider'], self.wb_b_ctrls['spinbox']])
        if hasattr(self, 'auto_wb_button'): preview_widgets.append(self.auto_wb_button)
        if hasattr(self, 'reset_wb_button'): preview_widgets.append(self.reset_wb_button)
        if hasattr(self, 'stretch_combo'): preview_widgets.append(self.stretch_combo)
        if hasattr(self, 'stretch_bp_ctrls'): preview_widgets.extend([self.stretch_bp_ctrls['slider'], self.stretch_bp_ctrls['spinbox']])
        if hasattr(self, 'stretch_wp_ctrls'): preview_widgets.extend([self.stretch_wp_ctrls['slider'], self.stretch_wp_ctrls['spinbox']])
        if hasattr(self, 'stretch_gamma_ctrls'): preview_widgets.extend([self.stretch_gamma_ctrls['slider'], self.stretch_gamma_ctrls['spinbox']])
        if hasattr(self, 'auto_stretch_button'): preview_widgets.append(self.auto_stretch_button)
        if hasattr(self, 'reset_stretch_button'): preview_widgets.append(self.reset_stretch_button)
        if hasattr(self, 'brightness_ctrls'): preview_widgets.extend([self.brightness_ctrls['slider'], self.brightness_ctrls['spinbox']])
        if hasattr(self, 'contrast_ctrls'): preview_widgets.extend([self.contrast_ctrls['slider'], self.contrast_ctrls['spinbox']])
        if hasattr(self, 'saturation_ctrls'): preview_widgets.extend([self.saturation_ctrls['slider'], self.saturation_ctrls['spinbox']])
        if hasattr(self, 'reset_bcs_button'): preview_widgets.append(self.reset_bcs_button)
        if hasattr(self, 'hist_reset_btn'): preview_widgets.append(self.hist_reset_btn)

        widgets_to_set = []
        if state == tk.NORMAL:
            print("DEBUG (GUI _set_parameter_widgets_state): Activation de tous les widgets...") 
            # Activer TOUS les widgets (traitement + preview) quand le traitement finit
            widgets_to_set = processing_widgets + preview_widgets
            # S'assurer que les options de pondération ET DRIZZLE sont dans le bon état initial
            self._update_weighting_options_state()
            self._update_drizzle_options_state() # <-- Appel ajouté ici
            # ... (reste de la logique pour state == tk.NORMAL) ...

        else: # tk.DISABLED (Pendant le traitement)
            # Désactiver les paramètres de traitement (y compris Drizzle)
            widgets_to_set = processing_widgets
            # Les widgets de preview restent actifs
            for widget in preview_widgets:
                # ... (logique existante) ...
            # Le bouton Ajouter Dossier reste NORMAL
                if hasattr(self, 'add_files_button'): self.add_files_button.config(state=tk.NORMAL)

        # Appliquer l'état aux widgets sélectionnés
        for widget in widgets_to_set:
            if widget and hasattr(widget, 'winfo_exists') and widget.winfo_exists():
                try: widget.config(state=state)
                except tk.TclError: pass

        # Exceptionnellement, si on désactive (pendant traitement), on s'assure que
        # les options internes (scale drizzle, options poids) sont aussi désactivées,
        # même si la case principale était déjà décochée.
        if state == tk.DISABLED:
            self._update_weighting_options_state()
            self._update_drizzle_options_state()

    def _debounce_resize(self, event=None):
        if self._after_id_resize:
             try: self.root.after_cancel(self._after_id_resize)
             except tk.TclError: pass
        try: self._after_id_resize = self.root.after(300, self._refresh_preview_on_resize)
        except tk.TclError: pass

    def _refresh_preview_on_resize(self):
        if hasattr(self, 'preview_manager'): self.preview_manager.trigger_redraw()
        if hasattr(self, 'histogram_widget') and self.histogram_widget.winfo_exists():
             try: self.histogram_widget.canvas.draw_idle()
             except tk.TclError: pass

    def _on_closing(self):
        if self.processing:
            if messagebox.askokcancel(self.tr("quit"), self.tr("quit_while_processing")):
                print("Arrêt demandé via fermeture fenêtre..."); self.stop_processing()
                if self.thread and self.thread.is_alive(): self.thread.join(timeout=1.5)
                if hasattr(self, "queued_stacker") and self.queued_stacker.is_running(): print("Warning: Worker thread did not exit cleanly.")
                self._save_settings_and_destroy()
            else: return
        else: self._save_settings_and_destroy()




    def _open_mosaic_settings_window(self):
        """
        Ouvre la fenêtre modale pour configurer les options de mosaïque.
        """
        print("DEBUG (GUI): Clic sur bouton 'Mosaïque...' - Appel de _open_mosaic_settings_window.")
        # --- AJOUT DEBUG ---
        current_api_key_in_main_gui_var = "NOT_FOUND"
        if hasattr(self, 'astrometry_api_key_var'):
            try:
                current_api_key_in_main_gui_var = self.astrometry_api_key_var.get()
                print(f"DEBUG (GUI _open_mosaic_settings_window): Valeur de self.astrometry_api_key_var.get() = '{current_api_key_in_main_gui_var}' (longueur: {len(current_api_key_in_main_gui_var)})")
            except tk.TclError:
                print("DEBUG (GUI _open_mosaic_settings_window): Erreur TclError lecture astrometry_api_key_var (fenêtre détruite?)")
        else:
            print("DEBUG (GUI _open_mosaic_settings_window): self.astrometry_api_key_var N'EXISTE PAS sur self (SeestarStackerGUI).")
        # --- FIN AJOUT DEBUG ---

        # Vérifier si une instance existe déjà (sécurité, normalement inutile car modal)
        # (Optionnel, mais peut être utile pour le développement)
        if hasattr(self, '_mosaic_settings_window_instance') and self._mosaic_settings_window_instance and self._mosaic_settings_window_instance.winfo_exists():
            print("DEBUG (GUI): Fenêtre de paramètres mosaïque déjà ouverte. Mise au premier plan.")
            self._mosaic_settings_window_instance.lift()
            self._mosaic_settings_window_instance.focus_force()
            return

        # Créer et afficher la nouvelle fenêtre modale
        try:
            print("DEBUG (GUI): Création de l'instance MosaicSettingsWindow...")
            # Passer 'self' (l'instance SeestarStackerGUI) à la fenêtre enfant
            # pour qu'elle puisse mettre à jour le flag mosaic_mode_active etc.
            mosaic_window = MosaicSettingsWindow(parent_gui=self)
            self._mosaic_settings_window_instance = mosaic_window # Stocker référence (optionnel)
            print("DEBUG (GUI): Instance MosaicSettingsWindow créée.")
            # La fenêtre est modale (grab_set dans son __init__), donc l'exécution attend ici.

        except Exception as e:
            error_msg = f"Erreur création fenêtre paramètres mosaïque: {e}"
            print(f"ERREUR (GUI): {error_msg}")
            traceback.print_exc(limit=2)
            messagebox.showerror(
                self.tr("error", default="Error"),
                self.tr("mosaic_window_create_error", default="Could not open Mosaic settings window.") + f"\n{e}",
                parent=self.root
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
        if hasattr(self, '_local_solver_settings_window_instance') and \
           self._local_solver_settings_window_instance and \
           self._local_solver_settings_window_instance.winfo_exists():
            print("DEBUG (GUI): Fenêtre de paramètres des solveurs locaux déjà ouverte. Mise au premier plan.")
            try:
                self._local_solver_settings_window_instance.lift()
                self._local_solver_settings_window_instance.focus_force()
            except tk.TclError: # Au cas où la fenêtre aurait été détruite entre-temps
                self._local_solver_settings_window_instance = None # Réinitialiser
                # Essayer de recréer ci-dessous
            else:
                return

        # Créer et afficher la nouvelle fenêtre modale
        try:
            print("DEBUG (GUI): Création de l'instance LocalSolverSettingsWindow...")
            solver_window = LocalSolverSettingsWindow(parent_gui=self)
            self._local_solver_settings_window_instance = solver_window # Stocker référence (optionnel)
            print("DEBUG (GUI): Instance LocalSolverSettingsWindow créée.")
            # La fenêtre est modale (grab_set dans son __init__), donc l'exécution attend ici.

        except Exception as e:
            error_msg_key = "local_solver_window_create_error" # Nouvelle clé de traduction
            error_default_text = "Could not open Local Solvers settings window."
            full_error_msg = self.tr(error_msg_key, default=error_default_text) + f"\n{e}"
            
            print(f"ERREUR (GUI): Erreur création fenêtre paramètres solveurs locaux: {e}")
            traceback.print_exc(limit=2)
            messagebox.showerror(
                self.tr("error", default="Error"),
                full_error_msg,
                parent=self.root
            )
            self._local_solver_settings_window_instance = None

##################################################################################################################################

    def _save_settings_and_destroy(self):
        try:
            if self.root.winfo_exists(): self.settings.window_geometry = self.root.geometry()
        except tk.TclError: pass
        self.settings.update_from_ui(self)
        print(f"VÉRIF GUI: self.settings.astap_path AVANT save_settings = '{self.settings.astap_path}'") # <-- AJOUTER
        self.settings.save_settings()
        print("Fermeture de l'application.")
        self.root.destroy()

    # --- NOUVELLES METHODES ---
    def _copy_log_to_clipboard(self):
        """Copie le contenu de la zone de log dans le presse-papiers."""
        try:
            log_content = self.status_text.get(1.0, tk.END)
            self.root.clipboard_clear(); self.root.clipboard_append(log_content)
            self.update_progress_gui("ⓘ Contenu du log copié dans le presse-papiers.", None)
        except tk.TclError as e: print(f"Erreur Tcl lors de la copie du log: {e}")
        except Exception as e: print(f"Erreur copie log: {e}"); self.update_progress_gui(f"❌ Erreur copie log: {e}", None); messagebox.showerror(self.tr("error"), f"Impossible de copier le log:\n{e}")

    def _open_output_folder(self):
        """Ouvre le dossier de sortie dans l'explorateur de fichiers système."""
        output_folder = self.output_path.get()
        if not output_folder: messagebox.showwarning(self.tr("warning"), "Le chemin du dossier de sortie n'est pas défini."); return
        if not os.path.isdir(output_folder): messagebox.showerror(self.tr("error"), f"Le dossier de sortie n'existe pas :\n{output_folder}"); return
        try:
            system = platform.system()
            if system == "Windows": os.startfile(output_folder)
            elif system == "Darwin": subprocess.Popen(["open", output_folder])
            else: subprocess.Popen(["xdg-open", output_folder])
            self.update_progress_gui(f"ⓘ Ouverture du dossier: {output_folder}", None)
        except FileNotFoundError: messagebox.showerror(self.tr("error"), f"Impossible d'ouvrir le dossier.\nCommande non trouvée pour votre système ({system}).")
        except Exception as e: print(f"Erreur ouverture dossier: {e}"); messagebox.showerror(self.tr("error"), f"Impossible d'ouvrir le dossier:\n{e}"); self.update_progress_gui(f"❌ Erreur ouverture dossier: {e}", None)
########################################################
    # --- MÉTHODE update_progress_gui MODIFIÉE ---
    def update_progress_gui(self, message, progress=None):
        """Met à jour l'interface de progression, en gérant les messages Drizzle."""
        # Gérer le message spécial pour le compteur de dossiers
        if isinstance(message, str) and message.startswith("folder_count_update:"):
            try: self.root.after_idle(self.update_additional_folders_display)
            except tk.TclError: pass # Ignorer si fenêtre fermée
            return # Ne pas afficher ce message dans le log

        # Procéder à la mise à jour via ProgressManager si disponible
        if hasattr(self, "progress_manager") and self.progress_manager:
            final_drizzle_active = False
            # Détecter les messages indiquant l'étape Drizzle finale
            if isinstance(message, str):
                if "💧 Exécution Drizzle final" in message:
                    final_drizzle_active = True
                    message = "⏳ " + message # Ajouter indicateur visuel

            # Mettre à jour la barre et le log texte via ProgressManager
            self.progress_manager.update_progress(message, progress)

            # Gérer le mode indéterminé de la barre de progression
            try:
                pb = self.progress_manager.progress_bar
                if pb.winfo_exists():
                    current_mode = pb['mode']
                    if final_drizzle_active and current_mode != 'indeterminate':
                        pb.config(mode='indeterminate')
                        pb.start(15) # Vitesse animation (ms)
                    elif not final_drizzle_active and current_mode == 'indeterminate':
                        pb.stop()
                        pb.config(mode='determinate')
                        # Optionnel : Forcer la valeur si on a une progression
                        if progress is not None:
                            try: pb.configure(value=max(0.0, min(100.0, float(progress))))
                            except ValueError: pass
            except (tk.TclError, AttributeError):
                pass # Ignorer si widgets détruits
#############################################################################################################################################





    def _processing_finished(self):
        """
        Actions finales à exécuter dans le thread GUI après la fin ou l'arrêt
        du thread de traitement du backend (QueuedStacker).
        Met à jour l'interface utilisateur, affiche un résumé et gère l'aperçu final.
        """
        # Vérification initiale pour éviter exécutions multiples si déjà appelé
        if not self.processing:
            print("DEBUG GUI [_processing_finished]: Appel ignoré, self.processing est déjà False.")
            return

        print("DEBUG GUI [_processing_finished]: Entrée dans la méthode.")
        self.processing = False # Marquer que le traitement GUI est terminé

        # --- Arrêter le timer de la barre de progression et finaliser la barre ---
        if hasattr(self, 'progress_manager') and self.progress_manager:
            self.progress_manager.stop_timer()
            try:
                pb = self.progress_manager.progress_bar
                if pb.winfo_exists(): # Vérifier si le widget existe toujours
                     current_mode = pb['mode']
                     if current_mode == 'indeterminate': # Si la barre était en mode indéterminé (ex: Drizzle final)
                         pb.stop() # Arrêter l'animation
                         pb.config(mode='determinate') # Revenir au mode déterminé

                     # Si le traitement n'a pas été stoppé par une erreur critique du backend, mettre à 100%
                     # hasattr vérifie si queued_stacker a été initialisé,
                     # getattr vérifie si processing_error a été défini à autre chose que None.
                     # Si processing_error est None (pas d'erreur) OU si l'erreur n'était pas True (convention bizarre), on met à 100.
                     if not hasattr(self, 'queued_stacker') or not getattr(self.queued_stacker, 'processing_error', True):
                         pb.configure(value=100)
                     # else: Laisser la barre de progression à sa valeur actuelle si erreur
            except (tk.TclError, AttributeError) as e_pb:
                print(f"DEBUG GUI [_processing_finished]: Erreur mineure lors de la finalisation de la barre de progression: {e_pb}")
                pass # Continuer même si la barre a un souci
        print("DEBUG GUI [_processing_finished]: Timer et barre de progression GUI finalisés.")

        # --- Récupération de l'état final et des résultats depuis le backend (QueuedStacker) ---
        final_message_for_status_bar = self.tr("stacking_finished") # Message par défaut
        final_stack_path = None
        processing_error_details = None
        images_stacked = 0 # Nombre d'images dans le stack final affiché/calculé
        aligned_count = 0
        failed_align_count = 0
        failed_stack_count = 0
        skipped_count = 0
        processed_files_count = 0 # Nombre total de fichiers que le backend a tenté de traiter
        total_exposure = 0.0
        was_stopped_by_user = False
        output_folder_exists = False
        can_open_output_folder_button = False # Contrôle l'état du bouton "Ouvrir Sortie"
        final_stack_file_exists = False
        is_drizzle_result = False
        final_stack_type_for_summary = "Unknown" # Pour le résumé
        # Flags de post-traitement
        photutils_applied_this_run_backend = False
        bn_globale_applied_this_run_backend = False
        cb_applied_in_session_backend = False # Renommé pour éviter confusion avec self.cb_applied_in_session
        scnr_applied_this_run_backend = False
        crop_applied_this_run_backend = False
        feathering_applied_this_run_backend = False # Ajout pour feathering
        low_wht_mask_applied_this_run_backend = False # Ajout pour Low WHT Mask
        photutils_params_used_backend = {}


        if hasattr(self, "queued_stacker") and self.queued_stacker is not None:
            q_stacker = self.queued_stacker # Alias pour lisibilité
            print("DEBUG GUI [_processing_finished]: Récupération des infos depuis queued_stacker.")

            final_stack_path = getattr(q_stacker, 'final_stacked_path', None)
            drizzle_active_session = getattr(q_stacker, 'drizzle_active_session', False)
            drizzle_mode = getattr(q_stacker, 'drizzle_mode', 'Final') # Assurer une valeur par défaut
            was_stopped_by_user = getattr(q_stacker, 'stop_processing_flag_for_gui', False) # Utiliser le flag dédié
            processing_error_details = getattr(q_stacker, 'processing_error', None)
            
            # Compteurs
            images_in_cumulative_from_backend = getattr(q_stacker, 'images_in_cumulative_stack', 0)
            aligned_count = getattr(q_stacker, 'aligned_files_count', 0)
            failed_align_count = getattr(q_stacker, 'failed_align_count', 0)
            failed_stack_count = getattr(q_stacker, 'failed_stack_count', 0)
            skipped_count = getattr(q_stacker, 'skipped_files_count', 0)
            processed_files_count = getattr(q_stacker, 'processed_files_count', 0)
            total_exposure = getattr(q_stacker, 'total_exposure_seconds', 0.0)

            # Flags de post-traitement depuis le backend
            photutils_applied_this_run_backend = getattr(q_stacker, 'photutils_bn_applied_in_session', False)
            bn_globale_applied_this_run_backend = getattr(q_stacker, 'bn_globale_applied_in_session', False)
            cb_applied_in_session_backend = getattr(q_stacker, 'cb_applied_in_session', False)
            scnr_applied_this_run_backend = getattr(q_stacker, 'scnr_applied_in_session', False)
            crop_applied_this_run_backend = getattr(q_stacker, 'crop_applied_in_session', False)
            feathering_applied_this_run_backend = getattr(q_stacker, 'feathering_applied_in_session', False)
            low_wht_mask_applied_this_run_backend = getattr(q_stacker, 'low_wht_mask_applied_in_session', False)
            photutils_params_used_backend = getattr(q_stacker, 'photutils_params_used_in_session', {}).copy()


            print(f"  -> final_stack_path (backend): {final_stack_path}")
            print(f"  -> drizzle_active_session: {drizzle_active_session}, drizzle_mode: {drizzle_mode}")
            print(f"  -> was_stopped_by_user (backend flag): {was_stopped_by_user}")
            print(f"  -> processing_error_details: {processing_error_details}")
            print(f"  -> images_in_cumulative_from_backend: {images_in_cumulative_from_backend}")
            print(f"  -> Compteurs: Aligned={aligned_count}, FailAlign={failed_align_count}, FailStack={failed_stack_count}, Skipped={skipped_count}, Processed={processed_files_count}")
            print(f"  -> Total Exposure: {total_exposure:.2f}s")
            print(f"  -> Post-Proc Flags (Backend): PB2D={photutils_applied_this_run_backend}, BNGlob={bn_globale_applied_this_run_backend}, CB={cb_applied_in_session_backend}, SCNR={scnr_applied_this_run_backend}, Crop={crop_applied_this_run_backend}, Feather={feathering_applied_this_run_backend}, LowWHT={low_wht_mask_applied_this_run_backend}")
            if photutils_applied_this_run_backend: print(f"     -> Params PB2D: {photutils_params_used_backend}")


            # Déterminer si le résultat est un Drizzle
            is_drizzle_result = (
                drizzle_active_session and
                not was_stopped_by_user and # Un Drizzle arrêté n'est pas considéré comme un "résultat Drizzle" complet
                processing_error_details is None and
                final_stack_path is not None and
                ("_drizzle" in os.path.basename(final_stack_path).lower() or 
                 "_mosaic" in os.path.basename(final_stack_path).lower()) # Inclure mosaïque
            )
            print(f"  -> is_drizzle_result (calculé): {is_drizzle_result}")
            
            # Calculer le nombre d'images réellement dans le stack final pour l'affichage
            if is_drizzle_result:
                # Pour Drizzle final, c'est le nombre d'images alignées qui ont contribué
                # Pour Drizzle incrémental, c'est le nombre total accumulé
                images_stacked = aligned_count if drizzle_mode == "Final" else images_in_cumulative_from_backend
            else: # Stack classique
                images_stacked = images_in_cumulative_from_backend
            print(f"  -> images_stacked (pour résumé): {images_stacked}")

            # Mettre à jour le compteur d'images alignées dans l'UI (même si le stack final a échoué)
            default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
            try:
                if hasattr(self, 'aligned_files_var'):
                    self.aligned_files_var.set(default_aligned_fmt.format(count=aligned_count))
            except tk.TclError: pass # Ignorer si le widget est détruit

        else: # Cas où self.queued_stacker n'est pas disponible (ne devrait pas arriver si start_processing a réussi)
            final_message_for_status_bar = "Erreur critique: Instance du backend (QueuedStacker) non trouvée."
            processing_error_details = final_message_for_status_bar
            print(f"ERREUR GUI [_processing_finished]: {final_message_for_status_bar}")

        # Vérifier existence du dossier de sortie et du fichier stack final
        if hasattr(self, 'output_path') and self.output_path.get():
            output_folder_exists = os.path.isdir(self.output_path.get())
        final_stack_file_exists = final_stack_path and os.path.exists(final_stack_path)
        
        # Le bouton "Ouvrir Sortie" est actif si le dossier de sortie existe ET
        # soit un stack final a été créé, soit il n'y a pas eu d'erreur critique.
        can_open_output_folder_button = output_folder_exists and (final_stack_file_exists or not processing_error_details)
        print(f"  -> output_folder_exists: {output_folder_exists}, final_stack_file_exists: {final_stack_file_exists}")
        print(f"  -> can_open_output_folder_button (calculé): {can_open_output_folder_button}")


        # --- Déterminer le message de statut final et le type de stack pour le résumé ---
        status_text_for_log = "" # Pour le log principal du GUI
        if was_stopped_by_user:
            status_text_for_log = self.tr('processing_stopped')
            if final_stack_file_exists and is_drizzle_result: final_stack_type_for_summary = "Drizzle (Incomplet)" # Drizzle stoppé
            elif final_stack_file_exists: final_stack_type_for_summary = "Classique (Incomplet)" # Classique stoppé
            else: final_stack_type_for_summary = "Arrêté (Pas de Fichier)"
        elif processing_error_details:
            status_text_for_log = f"{self.tr('stacking_error_msg')} {processing_error_details}"
            final_stack_type_for_summary = "Erreur"
        elif not final_stack_file_exists:
            status_text_for_log = self.tr("Terminé (Aucun stack final créé)", default="Finished (No final stack created)")
            final_stack_type_for_summary = "Aucun"
        elif is_drizzle_result: # Drizzle ou Mosaïque terminé avec succès
            if "_mosaic" in os.path.basename(final_stack_path).lower():
                status_text_for_log = self.tr("Mosaic Complete", default="Mosaic Assembly Complete")
                final_stack_type_for_summary = "Mosaïque Drizzle"
            else:
                status_text_for_log = self.tr("Drizzle Complete", default="Drizzle Complete")
                final_stack_type_for_summary = "Drizzle"
        else: # Stack classique terminé avec succès
            status_text_for_log = self.tr("Stacking Complete", default="Stacking Complete")
            final_stack_type_for_summary = "Classique"
        print(f"  -> status_text_for_log: '{status_text_for_log}', final_stack_type_for_summary: '{final_stack_type_for_summary}'")

        # Mettre à jour la zone de log texte du GUI
        if hasattr(self, 'progress_manager') and self.progress_manager:
             try: self.progress_manager.update_progress(status_text_for_log, self.progress_bar['value'])
             except tk.TclError: pass # Ignorer si barre détruite

        # --- Mise à jour de l'Aperçu avec le résultat final ---
        preview_load_error_msg = None # Pour stocker un message d'erreur si l'aperçu échoue
        print("DEBUG GUI [_processing_finished]: Préparation de la mise à jour de l'aperçu final...")

        if final_stack_file_exists or (hasattr(q_stacker, 'last_saved_data_for_preview') and q_stacker.last_saved_data_for_preview is not None):
            final_image_data_for_preview = None
            final_header_for_preview = None

            # Priorité à l'image en mémoire si disponible et si pas d'erreur de sauvegarde FITS
            # (Si erreur sauvegarde FITS, final_stack_path pourrait être None, mais last_saved_data peut exister)
            if hasattr(q_stacker, 'last_saved_data_for_preview') and \
               q_stacker.last_saved_data_for_preview is not None and \
               (final_stack_file_exists or not final_stack_path): # Utiliser si FITS OK ou si FITS a échoué mais on a les données
                
                final_image_data_for_preview = q_stacker.last_saved_data_for_preview
                print("  -> Utilisation de 'last_saved_data_for_preview' du backend pour l'aperçu.")
                # Essayer de prendre le header final du backend s'il est complet, sinon recharger
                if hasattr(q_stacker, 'current_stack_header') and q_stacker.current_stack_header:
                    final_header_for_preview = q_stacker.current_stack_header.copy()
                    print("     -> Utilisation du header du backend pour l'aperçu.")
                elif final_stack_file_exists: # Fallback si header backend pas bon mais fichier existe
                    try: final_header_for_preview = fits.getheader(final_stack_path); print("     -> Header du backend non fiable, rechargement depuis FITS pour aperçu.")
                    except Exception as e_hdr: preview_load_error_msg = f"Erreur lecture header FITS final: {e_hdr}"; print(f"     -> ERREUR lecture header FITS final: {e_hdr}")
                else: # Ni header backend, ni fichier FITS (cas d'erreur sauvegarde FITS)
                    final_header_for_preview = fits.Header(); final_header_for_preview['COMMENT'] = "Header non disponible (erreur sauvegarde FITS?)"
                    print("     -> Header non disponible (backend absent, fichier FITS absent/erreur).")


            elif final_stack_file_exists: # Si pas de données en mémoire, recharger depuis le disque
                print("  -> 'last_saved_data_for_preview' non dispo ou FITS plus récent, rechargement du FITS pour l'aperçu...")
                try:
                    final_image_data_for_preview = load_and_validate_fits(final_stack_path)
                    if final_image_data_for_preview is not None:
                        final_header_for_preview = fits.getheader(final_stack_path)
                        print(f"     -> Rechargement FITS OK. Shape: {final_image_data_for_preview.shape}")
                    else: # load_and_validate_fits a retourné None
                        preview_load_error_msg = f"{self.tr('Error loading final stack preview')}: load_and_validate_fits a retourné None pour {os.path.basename(final_stack_path)}."
                        print(f"     -> ERREUR: load_and_validate_fits a retourné None pour {final_stack_path}")
                except Exception as e_load:
                    preview_load_error_msg = f"Erreur rechargement FITS final pour aperçu: {e_load}"
                    print(f"     -> ERREUR: Exception pendant rechargement FITS: {e_load}"); traceback.print_exc(limit=1)
            else: # Ni données en mémoire, ni fichier FITS (ne devrait pas arriver si pas d'erreur)
                preview_load_error_msg = "Aucune donnée ou fichier FITS valide pour l'aperçu final."
                print(f"  -> {preview_load_error_msg}")


            # Si on a obtenu des données et un header pour l'aperçu
            if final_image_data_for_preview is not None and final_header_for_preview is not None:
                print("  -> Mise à jour de self.current_preview_data et self.current_stack_header...")
                self.current_preview_data = final_image_data_for_preview
                self.current_stack_header = final_header_for_preview
                
                # --- Forcer un auto-ajustement de l'aperçu pour le résultat final ---
                print("  -> Déclenchement auto-ajustement (WB & Stretch) pour l'aperçu final...")
                # On met à jour les compteurs pour que le texte de l'aperçu soit correct
                self.preview_img_count = images_stacked 
                self.preview_total_imgs = getattr(q_stacker, 'files_in_queue_at_start', images_stacked) # Utiliser total initial si dispo
                self.preview_current_batch = getattr(q_stacker, 'stacked_batches_count', 0)
                self.preview_total_batches = getattr(q_stacker, 'total_batches_estimated', 0)
                
                self.apply_auto_white_balance() # Déclenchera refresh_preview
                self.root.after(100, self.apply_auto_stretch) # Délai pour que WB soit pris en compte pour analyse stretch
                # --- Fin auto-ajustement ---
                
                # Mettre à jour les infos image avec le header final
                self.update_image_info(self.current_stack_header)
            else: # Si, après toutes les tentatives, on n'a rien pour l'aperçu
                if not preview_load_error_msg: preview_load_error_msg = "Données d'aperçu final non disponibles."
                print(f"  -> Échec final obtention données/header pour aperçu: {preview_load_error_msg}")
        
        else: # Pas de fichier stack final ET pas de données en mémoire du backend
             preview_load_error_msg = "Aucun fichier de stack final produit et pas de données en mémoire."
             print(f"  -> {preview_load_error_msg}")


        # --- Génération et Affichage du Résumé ---
        print("DEBUG GUI [_processing_finished]: Génération du résumé...")
        summary_lines = []
        summary_title = self.tr("processing_report_title")

        summary_lines.append(f"{self.tr('Status', default='Status')}: {status_text_for_log}")
        elapsed_total_seconds = 0
        if self.global_start_time: # global_start_time est défini dans start_processing
            elapsed_total_seconds = time.monotonic() - self.global_start_time
        summary_lines.append(f"{self.tr('Total Processing Time', default='Total Processing Time')}: {self._format_duration(elapsed_total_seconds)}")
        summary_lines.append(f"{self.tr('Final Stack Type', default='Final Stack Type')}: {final_stack_type_for_summary}")
        summary_lines.append(f"{self.tr('Files Attempted', default='Files Attempted')}: {processed_files_count}")
        total_rejected = failed_align_count + failed_stack_count + skipped_count
        summary_lines.append(f"{self.tr('Files Rejected (Total)', default='Files Rejected (Total)')}: {total_rejected} ({self.tr('Align', default='Align')}: {failed_align_count}, {self.tr('Stack Err', default='Stack Err')}: {failed_stack_count}, {self.tr('Other', default='Other')}: {skipped_count})")
        summary_lines.append(f"{self.tr('Images in Final Stack', default='Images in Final Stack')} ({final_stack_type_for_summary}): {images_stacked}") # Utilise images_stacked
        summary_lines.append(f"{self.tr('Total Exposure (Final Stack)', default='Total Exposure (Final Stack)')}: {self._format_duration(total_exposure)}")

        # Détails Post-Traitement
        summary_lines.append(f"\n--- {self.tr('Post-Processing Applied', default='Post-Processing Applied')} ---")
        summary_lines.append(f"  - {self.tr('Global Background Neutralization (BN)', default='Global Background Neutralization (BN)')}: {'Yes' if bn_globale_applied_this_run_backend else 'No'}")
        if photutils_applied_this_run_backend:
            params_str_list = []; photutils_params_to_log = ['box_size','filter_size','sigma_clip_val','exclude_percentile']
            for p_key in photutils_params_to_log:
                 if p_key in photutils_params_used_backend:
                     val = photutils_params_used_backend[p_key]
                     p_name_short = p_key.replace("photutils_bn_","").replace("_val","").replace("_percentile","%").replace("filter_size","Filt").replace("box_size","Box").replace("sigma_clip","Sig").title()
                     params_str_list.append(f"{p_name_short}={val:.1f}" if isinstance(val,float) else f"{p_name_short}={val}")
            params_str = ", ".join(params_str_list) if params_str_list else "Défauts"
            summary_lines.append(f"  - {self.tr('Photutils 2D Background', default='Photutils 2D Background')}: {self.tr('Yes', default='Yes')} ({params_str})")
        else: summary_lines.append(f"  - {self.tr('Photutils 2D Background', default='Photutils 2D Background')}: {self.tr('No', default='No')}")
        summary_lines.append(f"  - {self.tr('Edge/Chroma Correction (CB)', default='Edge/Chroma Correction (CB)')}: {'Yes' if cb_applied_in_session_backend else 'No'}")
        summary_lines.append(f"  - Feathering: {'Yes' if feathering_applied_this_run_backend else 'No'}")
        summary_lines.append(f"  - Low WHT Mask: {'Yes' if low_wht_mask_applied_this_run_backend else 'No'}")
        scnr_target_sum = getattr(q_stacker, 'final_scnr_target_channel', '?') if hasattr(self, "queued_stacker") else '?'
        scnr_amount_sum = getattr(q_stacker, 'final_scnr_amount', 0.0)  if hasattr(self, "queued_stacker") else 0.0
        scnr_lum_sum = getattr(q_stacker, 'final_scnr_preserve_luminosity', '?')  if hasattr(self, "queued_stacker") else '?'
        scnr_info_summary = f"{self.tr('Yes', default='Yes')} (Cible: {scnr_target_sum}, Force: {scnr_amount_sum:.2f}, Prés.Lum: {scnr_lum_sum})" if scnr_applied_this_run_backend else self.tr('No', default='No')
        summary_lines.append(f"  - {self.tr('Final SCNR', default='Final SCNR')}: {scnr_info_summary}")
        crop_perc_decimal_sum = getattr(q_stacker, 'final_edge_crop_percent_decimal', 0.0)  if hasattr(self, "queued_stacker") else 0.0
        crop_info_summary = f"{self.tr('Yes', default='Yes')} ({crop_perc_decimal_sum*100.0:.1f}%)" if crop_applied_this_run_backend else self.tr('No', default='No')
        summary_lines.append(f"  - {self.tr('Final Edge Crop', default='Final Edge Crop')}: {crop_info_summary}")
        summary_lines.append("-------------------------------")

        # Chemin du fichier final
        if final_stack_file_exists:
            summary_lines.append(f"\n{self.tr('Final Stack File', default='Final Stack File')}:\n  {final_stack_path}")
        elif final_stack_path: # Chemin défini mais fichier non trouvé (erreur sauvegarde?)
            summary_lines.append(f"{self.tr('Final Stack File', default='Final Stack File')}:\n  {final_stack_path} ({self.tr('Not Found!', default='Not Found!')})")
        else: # Aucun chemin (erreur critique avant même de nommer le fichier)
            summary_lines.append(self.tr('Final Stack File: Not created or not found.', default='Final Stack File: Not created or not found.'))

        if preview_load_error_msg: # Ajouter le message d'erreur de l'aperçu au résumé
            summary_lines.append(f"\nNote Aperçu: {preview_load_error_msg}")

        full_summary_text_for_dialog = "\n".join(summary_lines)

        # Afficher le dialogue de résumé (sauf si erreur critique avant même de commencer)
        if was_stopped_by_user: # Si arrêté, afficher juste dans le log GUI
            print("--- Processing Stopped by User (Summary Dialog Skipped) ---")
            print(full_summary_text_for_dialog)
            print("-------------------------------------------------------------")
        elif processing_error_details and not final_stack_file_exists : # Erreur critique ET pas de fichier final
             messagebox.showerror(self.tr("error"), f"{status_text_for_log}") # Affiche juste l'erreur principale
        else: # Succès ou erreur avec fichier partiel -> Afficher le dialogue complet
            self._show_summary_dialog(summary_title, full_summary_text_for_dialog, can_open_output_folder_button)


        # --- Réinitialisation de l'état de l'UI pour un nouveau traitement ---
        print("DEBUG GUI [_processing_finished]: Réinitialisation de l'état de l'UI...")
        try:
            self._set_parameter_widgets_state(tk.NORMAL) # Réactiver tous les contrôles principaux
            if hasattr(self, "start_button") and self.start_button.winfo_exists():
                self.start_button.config(state=tk.NORMAL)
            if hasattr(self, "stop_button") and self.stop_button.winfo_exists():
                self.stop_button.config(state=tk.DISABLED)
            if hasattr(self, "open_output_button") and self.open_output_button.winfo_exists():
                self.open_output_button.config(state=tk.NORMAL if can_open_output_folder_button else tk.DISABLED)
            if hasattr(self, "remaining_time_var"):
                self.remaining_time_var.set("00:00:00") # Remettre ETA à zéro

            # Réinitialiser le compteur de dossiers additionnels dans l'UI et la liste interne
            self.additional_folders_to_process = [] # Vider la liste GUI
            self.update_additional_folders_display() # Mettre à jour l'affichage (devrait montrer "Aucun")
            self.update_remaining_files() # Mettre à jour R/T (devrait être "Aucun fichier en attente")

        except tk.TclError as e_reset_ui:
            print(f"DEBUG GUI [_processing_finished]: Erreur TclError lors de la réinitialisation de l'UI: {e_reset_ui}")
            # Continuer, la plupart des choses importantes sont faites

        # Forcer un garbage collect à la fin pour libérer la mémoire
        if 'gc' in globals() or 'gc' in locals():
            gc.collect()
            print("DEBUG GUI [_processing_finished]: Garbage collection effectué.")

        print("DEBUG GUI [_processing_finished]: Fin de la méthode.")






################################################################################################################################################
    def _show_summary_dialog(self, summary_title, summary_text, can_open_output): # Ajout argument
        """Displays a custom modal dialog with the processing summary."""
        dialog = tk.Toplevel(self.root); dialog.title(summary_title); dialog.transient(self.root); dialog.grab_set(); dialog.resizable(False, False)
        content_frame = ttk.Frame(dialog, padding="10 10 10 10"); content_frame.pack(expand=True, fill=tk.BOTH)
        try: icon_label = ttk.Label(content_frame, image="::tk::icons::information", padding=(0, 0, 10, 0))
        except tk.TclError: icon_label = ttk.Label(content_frame, text="i", font=("Arial", 16, "bold"), padding=(0, 0, 10, 0))
        icon_label.grid(row=0, column=0, sticky="nw", pady=(0, 10))
        summary_label = ttk.Label(content_frame, text=summary_text, justify=tk.LEFT, wraplength=450); summary_label.grid(row=0, column=1, sticky="nw", padx=(0, 10))
        button_frame = ttk.Frame(content_frame); button_frame.grid(row=1, column=0, columnspan=2, sticky="se", pady=(15, 0))

        # --- Bouton Ouvrir Dossier (Conditionnel) ---
        # L'état est basé sur l'argument can_open_output passé depuis _processing_finished
        open_button = ttk.Button(button_frame, text=self.tr("Open Output", default="Open Output"), command=self._open_output_folder, state=tk.NORMAL if can_open_output else tk.DISABLED)
        open_button.pack(side=tk.LEFT, padx=(0, 10)) # Toujours packé, l'état le contrôle
######################################################
        # --- Boutons Copier et OK (inchangés) ---
        def copy_action():
            try:
                dialog.clipboard_clear()
                dialog.clipboard_append(summary_text)
                copy_button.config(text=self.tr("Copied!", default="Copied!"))
                dialog.after(
                    1500,
                    lambda: copy_button.config(text=self.tr("Copy Summary", default="Copy Summary"))
                    if copy_button.winfo_exists()
                    else None,
                )
            except Exception as copy_e:
                print(f"Error copying summary: {copy_e}")

        copy_button = ttk.Button(
            button_frame, text=self.tr("Copy Summary", default="Copy Summary"), command=copy_action
        )
        copy_button.pack(side=tk.RIGHT, padx=(5, 0))

        ok_button = ttk.Button(
            button_frame,
            text="OK",
            command=dialog.destroy,
            style="Accent.TButton" if "Accent.TButton" in ttk.Style().element_names() else "TButton",
        )
        ok_button.pack(side=tk.RIGHT)
        ok_button.focus_set()

        # Centrer dialogue (inchangé)
        dialog.update_idletasks()
        root_x = self.root.winfo_x(); root_y = self.root.winfo_y(); root_width = self.root.winfo_width(); root_height = self.root.winfo_height()
        dialog_width = dialog.winfo_width(); dialog_height = dialog.winfo_height()
        pos_x = root_x + (root_width // 2) - (dialog_width // 2); pos_y = root_y + (root_height // 2) - (dialog_height // 2)
        dialog.geometry(f"+{pos_x}+{pos_y}")
        self.root.wait_window(dialog)
#######################################################################################################################################


#########################################################################################################################################


# --- DANS LA CLASSE SeestarStackerGUI DANS seestar/gui/main_window.py ---

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
        print("DEBUG (GUI start_processing): Début tentative démarrage du traitement...")

        if hasattr(self, 'start_button'):
            try: self.start_button.config(state=tk.DISABLED)
            except tk.TclError: pass # Ignorer si widget détruit

        # --- 1. Validation des chemins et de la présence de fichiers FITS ---
        print("DEBUG (GUI start_processing): Phase 1 - Validation des chemins...")
        input_folder = self.input_path.get()
        output_folder = self.output_path.get()

        if not input_folder or not output_folder:
            messagebox.showerror(self.tr("error"), self.tr("select_folders"))
            if hasattr(self, 'start_button') and self.start_button.winfo_exists(): self.start_button.config(state=tk.NORMAL)
            return
        if not os.path.isdir(input_folder):
            messagebox.showerror(self.tr("error"), f"{self.tr('input_folder_invalid')}:\n{input_folder}")
            if hasattr(self, 'start_button') and self.start_button.winfo_exists(): self.start_button.config(state=tk.NORMAL)
            return
        if not os.path.isdir(output_folder):
            try:
                os.makedirs(output_folder, exist_ok=True)
                self.update_progress_gui(f"{self.tr('Output folder created')}: {output_folder}", None)
            except Exception as e:
                messagebox.showerror(self.tr("error"), f"{self.tr('output_folder_invalid')}:\n{output_folder}\n{e}")
                if hasattr(self, 'start_button') and self.start_button.winfo_exists(): self.start_button.config(state=tk.NORMAL)
                return
        try:
            has_initial_fits = any(f.lower().endswith((".fit", ".fits")) for f in os.listdir(input_folder))
            has_additional_listed = bool(self.additional_folders_to_process)
            if not has_initial_fits and not has_additional_listed:
                if not messagebox.askyesno(self.tr("warning"), self.tr("no_fits_found")):
                    if hasattr(self, 'start_button') and self.start_button.winfo_exists(): self.start_button.config(state=tk.NORMAL)
                    return
        except Exception as e:
            messagebox.showerror(self.tr("error"), f"{self.tr('Error reading input folder')}:\n{e}")
            if hasattr(self, 'start_button') and self.start_button.winfo_exists(): self.start_button.config(state=tk.NORMAL)
            return
        print("DEBUG (GUI start_processing): Phase 1 - Validation des chemins OK.")

        # --- 2. Avertissement Drizzle/Mosaïque (si activé) ---
        print("DEBUG (GUI start_processing): Phase 2 - Vérification avertissement Drizzle/Mosaïque...")
        drizzle_globally_enabled_ui = self.use_drizzle_var.get()
        # Lire mosaic_mode_active depuis self.settings, qui devrait avoir été mis à jour par MosaicSettingsWindow
        is_mosaic_mode_ui = getattr(self.settings, 'mosaic_mode_active', False)

        if drizzle_globally_enabled_ui or is_mosaic_mode_ui:
            warning_title = self.tr('drizzle_warning_title')
            base_text_tuple_or_str = self.tr('drizzle_warning_text')
            base_warning_text = "".join(base_text_tuple_or_str) if isinstance(base_text_tuple_or_str, tuple) else base_text_tuple_or_str
            full_warning_text = base_warning_text
            if is_mosaic_mode_ui and not drizzle_globally_enabled_ui:
                 full_warning_text += "\n\n" + self.tr("mosaic_requires_drizzle_note", default="(Note: Mosaic mode requires Drizzle for final combination.)")

            print(f"DEBUG (GUI start_processing): Avertissement Drizzle/Mosaïque nécessaire. is_mosaic_mode_ui={is_mosaic_mode_ui}")
            continue_processing = messagebox.askyesno(warning_title, full_warning_text, parent=self.root)
            if not continue_processing:
                self.update_progress_gui("ⓘ Démarrage annulé par l'utilisateur après avertissement.", None)
                if hasattr(self, 'start_button') and self.start_button.winfo_exists(): self.start_button.config(state=tk.NORMAL)
                return
        print("DEBUG (GUI start_processing): Phase 2 - Vérification avertissement OK (ou non applicable).")

        # --- 3. Initialisation de l'état de traitement du GUI ---
        print("DEBUG (GUI start_processing): Phase 3 - Initialisation état de traitement GUI...")
        self.processing = True
        self.time_per_image = 0
        self.global_start_time = time.monotonic()
        self.batches_processed_for_preview_refresh = 0
        default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
        self.aligned_files_var.set(default_aligned_fmt.format(count=0))
        folders_to_pass_to_backend = list(self.additional_folders_to_process)
        self.additional_folders_to_process = []
        self.update_additional_folders_display()
        self._set_parameter_widgets_state(tk.DISABLED)
        if hasattr(self, "stop_button") and self.stop_button.winfo_exists(): self.stop_button.config(state=tk.NORMAL)
        if hasattr(self, "open_output_button") and self.open_output_button.winfo_exists(): self.open_output_button.config(state=tk.DISABLED)
        if hasattr(self, "progress_manager"): self.progress_manager.reset(); self.progress_manager.start_timer()
        if hasattr(self, "status_text") and self.status_text.winfo_exists():
            self.status_text.config(state=tk.NORMAL); self.status_text.delete(1.0, tk.END)
            self.status_text.insert(tk.END, f"--- {self.tr('stacking_start')} ---\n"); self.status_text.config(state=tk.DISABLED)
        print("DEBUG (GUI start_processing): Phase 3 - Initialisation état de traitement GUI OK.")

        # --- 4. Synchronisation et Validation des Settings ---
        print("DEBUG (GUI start_processing): Phase 4 - Synchronisation et validation des Settings...")
        print("  -> (4A) Appel self.settings.update_from_ui(self)...")
        self.settings.update_from_ui(self)
        # ... (logs de vérification des settings après update_from_ui) ...
        print(f"  DEBUG GUI SETTINGS (Phase 4A): self.settings.mosaic_settings['alignment_mode'] = {self.settings.mosaic_settings.get('alignment_mode', 'NonTrouve')}")

        print("  -> (4B) Appel self.settings.validate_settings()...")
        validation_messages = self.settings.validate_settings()
        # ... (logs de vérification des settings après validate_settings) ...
        print(f"  DEBUG GUI SETTINGS (Phase 4B): self.settings.mosaic_settings['alignment_mode'] = {self.settings.mosaic_settings.get('alignment_mode', 'NonTrouve')}")

        if validation_messages:
            self.update_progress_gui("⚠️ Paramètres ajustés après validation:", None)
            for msg in validation_messages: self.update_progress_gui(f"  - {msg}", None)
            print("  -> (4C) Ré-appel self.settings.apply_to_ui(self)...")
            self.settings.apply_to_ui(self)
            self._update_weighting_options_state()
            self._update_drizzle_options_state()
            self._update_final_scnr_options_state()
            self._update_photutils_bn_options_state()
            self._update_feathering_options_state()
            self._update_low_wht_mask_options_state() # S'assurer d'appeler ceci aussi
        print("DEBUG (GUI start_processing): Phase 4 - Settings synchronisés et validés.")

        # --- 5. Préparation des arguments pour le backend (inchangée, lit depuis self.settings) ---
        print("DEBUG (GUI start_processing): Phase 5 - Préparation des arguments pour le backend depuis self.settings...")
        # ... (log de tous les paramètres envoyés au backend, inchangé) ...
        print("  --- VALEURS ENVOYÉES AU BACKEND (depuis self.settings) ---")
        params_to_log_for_backend = [
            'input_folder', 'output_folder', 'reference_image_path', 'stacking_mode', 'kappa',
            'batch_size', 'correct_hot_pixels', 'hot_pixel_threshold', 'neighborhood_size',
            'bayer_pattern', 'cleanup_temp', 'use_quality_weighting', 'weight_by_snr',
            'weight_by_stars', 'snr_exponent', 'stars_exponent', 'min_weight',
            'use_drizzle', 'drizzle_scale', 'drizzle_wht_threshold', 'drizzle_mode',
            'drizzle_kernel', 'drizzle_pixfrac', 'apply_chroma_correction', 'apply_final_scnr',
            'final_scnr_target_channel', 'final_scnr_amount', 'final_scnr_preserve_luminosity',
            'bn_grid_size_str', 'bn_perc_low', 'bn_perc_high', 'bn_std_factor',
            'bn_min_gain', 'bn_max_gain', 'cb_border_size', 'cb_blur_radius',
            'cb_min_b_factor', 'cb_max_b_factor', 'final_edge_crop_percent',
            'apply_photutils_bn', 'photutils_bn_box_size', 'photutils_bn_filter_size',
            'photutils_bn_sigma_clip', 'photutils_bn_exclude_percentile',
            'apply_feathering', 'feather_blur_px', 'apply_low_wht_mask',
            'low_wht_percentile', 'low_wht_soften_px',
            'mosaic_mode_active', 'astrometry_api_key', 'mosaic_settings',
            'astap_search_radius'
        ]
        for param_name in params_to_log_for_backend:
            value = getattr(self.settings, param_name, f"ERREUR_ATTR_{param_name}")
            if param_name == 'astrometry_api_key': print(f"    {param_name}: {'Présente' if value else 'Absente'} (longueur: {len(str(value))})")
            elif param_name == 'mosaic_settings': print(f"    {param_name}: {value}") # Afficher le dict complet
            else: print(f"    {param_name}: {value}")
        print("  --- FIN VALEURS ENVOYÉES AU BACKEND ---")
        print("DEBUG (GUI start_processing): Phase 5 - Préparation des arguments terminée.")

        # --- AJOUT DU BLOC DE VÉRIFICATION CRITIQUE ---
        print("DEBUG (GUI start_processing): Phase 5.5 - Vérification critique avant appel backend...")
        final_mosaic_settings_for_backend = self.settings.mosaic_settings.copy()
        alignment_mode_to_backend = final_mosaic_settings_for_backend.get('alignment_mode', 'NON_DÉFINI_DANS_DICT_BACKEND')
        print(f"  CRITICAL CHECK (GUI start_processing): mosaic_settings QUI SERA ENVOYÉ: {final_mosaic_settings_for_backend}")
        print(f"  CRITICAL CHECK (GUI start_processing): alignment_mode DANS CE DICT: '{alignment_mode_to_backend}'")
        # --- FIN AJOUT ---

        # --- 6. Appel à queued_stacker.start_processing ---
        print("DEBUG (GUI start_processing): Phase 6 - Appel à queued_stacker.start_processing...")
        processing_started = self.queued_stacker.start_processing(
            input_dir=self.settings.input_folder,
            output_dir=self.settings.output_folder,
            reference_path_ui=self.settings.reference_image_path,
            initial_additional_folders=folders_to_pass_to_backend,
            stacking_mode=self.settings.stacking_mode,
            kappa=self.settings.kappa,
            batch_size=self.settings.batch_size,
            correct_hot_pixels=self.settings.correct_hot_pixels,
            hot_pixel_threshold=self.settings.hot_pixel_threshold,
            neighborhood_size=self.settings.neighborhood_size,
            bayer_pattern=self.settings.bayer_pattern,
            perform_cleanup=self.settings.cleanup_temp,
            use_weighting=self.settings.use_quality_weighting,
            weight_by_snr=self.settings.weight_by_snr,
            weight_by_stars=self.settings.weight_by_stars,
            snr_exp=self.settings.snr_exponent,
            stars_exp=self.settings.stars_exponent,
            min_w=self.settings.min_weight,
            use_drizzle=self.settings.use_drizzle,
            drizzle_scale=float(self.settings.drizzle_scale), # Assurer float
            drizzle_wht_threshold=self.settings.drizzle_wht_threshold,
            drizzle_mode=self.settings.drizzle_mode,
            drizzle_kernel=self.settings.drizzle_kernel,
            drizzle_pixfrac=self.settings.drizzle_pixfrac,
            apply_chroma_correction=self.settings.apply_chroma_correction,
            apply_final_scnr=self.settings.apply_final_scnr,
            final_scnr_target_channel=self.settings.final_scnr_target_channel,
            final_scnr_amount=self.settings.final_scnr_amount,
            final_scnr_preserve_luminosity=self.settings.final_scnr_preserve_luminosity,
            bn_grid_size_str=self.settings.bn_grid_size_str,
            bn_perc_low=self.settings.bn_perc_low,
            bn_perc_high=self.settings.bn_perc_high,
            bn_std_factor=self.settings.bn_std_factor,
            bn_min_gain=self.settings.bn_min_gain,
            bn_max_gain=self.settings.bn_max_gain,
            cb_border_size=self.settings.cb_border_size,
            cb_blur_radius=self.settings.cb_blur_radius,
            cb_min_b_factor=self.settings.cb_min_b_factor,
            cb_max_b_factor=self.settings.cb_max_b_factor,
            final_edge_crop_percent=self.settings.final_edge_crop_percent,
            apply_photutils_bn=self.settings.apply_photutils_bn,
            photutils_bn_box_size=self.settings.photutils_bn_box_size,
            photutils_bn_filter_size=self.settings.photutils_bn_filter_size,
            photutils_bn_sigma_clip=self.settings.photutils_bn_sigma_clip,
            photutils_bn_exclude_percentile=self.settings.photutils_bn_exclude_percentile,
            apply_feathering=self.settings.apply_feathering,
            feather_blur_px=self.settings.feather_blur_px,
            apply_low_wht_mask=self.settings.apply_low_wht_mask,
            low_wht_percentile=self.settings.low_wht_percentile,
            low_wht_soften_px=self.settings.low_wht_soften_px,
            is_mosaic_run=self.settings.mosaic_mode_active,
            api_key=self.settings.astrometry_api_key,
            mosaic_settings=self.settings.mosaic_settings, 
            use_local_solver_priority=self.settings.use_local_solver_priority,
            astap_path=self.settings.astap_path,
            astap_data_dir=self.settings.astap_data_dir,
            local_ansvr_path=self.settings.local_ansvr_path,
            astap_search_radius_ui=self.settings.astap_search_radius
        )
        print(f"DEBUG (GUI start_processing): Appel à queued_stacker.start_processing fait. Résultat: {processing_started}")

        # --- 7. Gérer résultat démarrage backend ---
        if processing_started:
            if hasattr(self, 'stop_button') and self.stop_button.winfo_exists(): self.stop_button.config(state=tk.NORMAL)
            self.thread = threading.Thread(target=self._track_processing_progress, daemon=True, name="GUI_ProgressTracker")
            self.thread.start()
        else:
            if hasattr(self, 'start_button') and self.start_button.winfo_exists(): self.start_button.config(state=tk.NORMAL)
            self.processing = False
            self.update_progress_gui("ⓘ Échec démarrage traitement (le backend a refusé ou erreur critique). Vérifiez logs console.", None)
            self._set_parameter_widgets_state(tk.NORMAL)
        print("DEBUG (GUI start_processing): Fin de la méthode.")


##############################################################################################################################################



        # ----Fin du Fichier main_window.py