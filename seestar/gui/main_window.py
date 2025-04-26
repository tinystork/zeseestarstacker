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
 
# Seestar imports
from seestar.core.image_processing import load_and_validate_fits, debayer_image
from seestar.core.image_processing import load_and_validate_fits, debayer_image
from seestar.localization import Localization
from seestar.queuep.queue_manager import SeestarQueuedStacker
from seestar.core.utils import estimate_batch_size
try:
    # Import tools for preview adjustments and auto calculations
    from seestar.tools.stretch import StretchPresets, ColorCorrection
    from seestar.tools.stretch import apply_auto_stretch as calculate_auto_stretch
    from seestar.tools.stretch import apply_auto_white_balance as calculate_auto_wb
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
    def __init__(self):
        """Initialise l'interface graphique."""
        self.root = tk.Tk()
      
        try:
            # Replace this path with the ACTUAL path to icon file!
            # Example: icon_path = "icons/my_app_icon.png"
            # Make sure the path is correct relative to where you run the script,
            # or use an absolute path.
            icon_path = 'icon/icon.png'

            if os.path.exists(icon_path):
                # Load the image using Pillow
                icon_image = Image.open(icon_path)
                # Convert it for Tkinter
                self.tk_icon = ImageTk.PhotoImage(icon_image)
                # Set the icon for the window (and taskbar/dock)
                self.root.iconphoto(True, self.tk_icon)
                print(f"DEBUG: Successfully loaded and set icon from: {icon_path}") # Optional debug message
            else:
                print(f"Warning: Icon file not found at: {icon_path}. Using default icon.")
        except FileNotFoundError:
             print(f"Error: Icon file not found at specified path: {icon_path}. Using default icon.")
        except Exception as e:
            print(f"Error loading or setting window icon: {e}")
            # Optional: Show traceback for debugging complex errors
            # import traceback
            # traceback.print_exc()
        # --- End Icon Setting ---

        self.localization = Localization("en") # The rest of the __init__ method continues here
        self.localization = Localization("en")
        self.settings = SettingsManager()
        self.queued_stacker = SeestarQueuedStacker() # Use the queued stacker
        self.processing = False # Flag to track if processing is active from UI perspective
        self.thread = None # Holds the progress tracker thread
        # Preview data is now primarily managed by the PreviewManager
        self.current_preview_data = None # Holds raw data for re-applying preview settings
        self.current_stack_header = None # Header associated with current_preview_data
        self.debounce_timer_id = None
        self.time_per_image = 0 # Average time per processed file (updated by tracker)
        self.global_start_time = None # Start time of the whole processing session

        # *** NOUVEAU : Liste pour les dossiers ajoutés AVANT de démarrer ***
        self.additional_folders_to_process = []
        # --- Fin Nouveau ---

        # --- Quality Weighting Variables ---
        self.use_weighting_var = tk.BooleanVar(value=False)
        self.weight_snr_var = tk.BooleanVar(value=True)
        self.weight_stars_var = tk.BooleanVar(value=True)
        self.snr_exponent_var = tk.DoubleVar(value=1.0)
        self.stars_exponent_var = tk.DoubleVar(value=0.5)
        self.min_weight_var = tk.DoubleVar(value=0.1)
        # Initialize Tkinter variables first
        self.init_variables()

        # Load settings and set initial language
        self.settings.load_settings()
        self.language_var.set(self.settings.language) # Ensure var matches loaded setting
        self.localization.set_language(self.settings.language)

        # Create Managers that depend on self.root and settings
        self.file_handler = FileHandlingManager(self) # Needs self for callbacks/settings

        # Build the UI layout
        self.create_layout() # Creates widgets, must be before manager init

        # Initialize managers that need widget references
        self.init_managers() # Initializes progress_manager, preview_manager etc.

        # Apply loaded settings to the UI widgets
        self.settings.apply_to_ui(self)
        self._update_weighting_options_state()
        self._update_show_folders_button_state()
        self.update_ui_language() # Translate UI based on loaded language

        # Connect backend callbacks
        self.queued_stacker.set_progress_callback(self.update_progress_gui)
        self.queued_stacker.set_preview_callback(self.update_preview_from_stacker)

        # Final window setup
        self.root.title(self.tr("title"))
        try:
            self.root.geometry(self.settings.window_geometry) # Apply saved geometry
        except tk.TclError:
            self.root.geometry("1200x750") # Default fallback
        self.root.minsize(1100, 650)
        self.root.bind("<Configure>", self._debounce_resize) # Handle resize for preview refresh
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing) # Handle closing

        # --- ADD Variables to store preview state info ---
        self.preview_img_count = 0
        self.preview_total_imgs = 0
        self.preview_current_batch = 0
        self.preview_total_batches = 0

    def init_variables(self):
        """Initialise les variables Tkinter."""
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

        # ... (Variables existantes : input_path, output_path, etc.) ...
        self.cleanup_temp_var = tk.BooleanVar(value=True) # Renamed from remove_aligned

        # --- AJOUTER ICI les variables Drizzle ---
        self.use_drizzle_var = tk.BooleanVar(value=False) # Drizzle désactivé par défaut
        self.drizzle_scale_var = tk.StringVar(value="2") # Échelle Drizzle (sous forme de string pour Combobox/Radio)
        self.drizzle_wht_threshold_var = tk.DoubleVar(value=0.7)
        # --- FIN AJOUT ---

        # Preview settings variables
        self.preview_stretch_method = tk.StringVar(value="Asinh")
        # ... (Reste des variables existantes) ...

        # Preview settings variables
        self.preview_stretch_method = tk.StringVar(value="Asinh")
        self.preview_black_point = tk.DoubleVar(value=0.01)
        self.preview_white_point = tk.DoubleVar(value=0.99)
        self.preview_gamma = tk.DoubleVar(value=1.0)
        self.preview_r_gain = tk.DoubleVar(value=1.0)
        self.preview_g_gain = tk.DoubleVar(value=1.0)
        self.preview_b_gain = tk.DoubleVar(value=1.0)
        # Additional preview adjustments (Brightness/Contrast/Saturation)
        self.preview_brightness = tk.DoubleVar(value=1.0)
        self.preview_contrast = tk.DoubleVar(value=1.0)
        self.preview_saturation = tk.DoubleVar(value=1.0)

        # UI State variables
        self.language_var = tk.StringVar(value='en') # Default language
        self.remaining_files_var = tk.StringVar(value=self.tr("no_files_waiting", default="No files waiting"))
        self.additional_folders_var = tk.StringVar(value=self.tr("no_additional_folders", default="None"))
        default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
        self.aligned_files_var = tk.StringVar(value=default_aligned_fmt.format(count="--"))
        self.remaining_time_var = tk.StringVar(value="--:--:--") # ETA
        self.elapsed_time_var = tk.StringVar(value="00:00:00")
        self._after_id_resize = None # For debouncing resize events


    def _update_drizzle_options_state(self):
        """Active ou désactive les options d'échelle Drizzle ET le seuil WHT."""
        try:
            # Déterminer l'état souhaité (NORMAL ou DISABLED) basé sur la checkbox Drizzle
            state = tk.NORMAL if self.use_drizzle_var.get() else tk.DISABLED

            # Liste des widgets qui dépendent de l'activation de Drizzle
            widgets_to_toggle = [
                # Widgets pour l'échelle Drizzle (déjà présents)
                getattr(self, 'drizzle_scale_label', None),
                getattr(self, 'drizzle_radio_2x', None), # Si tu utilises les Radiobuttons
                getattr(self, 'drizzle_radio_3x', None), # Si tu utilises les Radiobuttons
                getattr(self, 'drizzle_radio_4x', None), # Si tu utilises les Radiobuttons
                # getattr(self, 'drizzle_scale_combo', None), # Si tu utilises la Combobox

                # ---> AJOUTER les références aux widgets du seuil WHT ici <---
                getattr(self, 'drizzle_wht_label', None),   # Le label "Seuil WHT%:"
                getattr(self, 'drizzle_wht_spinbox', None), # Le Spinbox pour régler le seuil
                # ---> FIN DE L'AJOUT <---
            ]

            # Boucle pour appliquer l'état (NORMAL ou DISABLED) à chaque widget de la liste
            for widget in widgets_to_toggle:
                # Vérifier si le widget existe réellement avant de tenter de le configurer
                if widget and hasattr(widget, 'winfo_exists') and widget.winfo_exists():
                    widget.config(state=state)

        except tk.TclError:
            # Ignorer les erreurs si un widget n'existe pas (peut arriver pendant l'init)
            pass
        except AttributeError:
            # Ignorer si un attribut n'existe pas encore (peut arriver pendant l'init)
            pass

    
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
        """Lit le Spinbox (%) et met à jour la variable (0-1)."""
        try:
            percent_value = float(self.drizzle_wht_spinbox.get())
            float_value = np.clip(percent_value / 100.0, 0.01, 1.0) # Assurer 0.01-1.0
            # Mettre à jour directement la variable
            self.drizzle_wht_threshold_var.set(float_value)
        except (ValueError, tk.TclError):
            pass # Ignorer erreurs de conversion

    def _update_spinbox_from_float(self, *args):
        """Lit la variable (0-1) et met à jour le Spinbox (%). Appelé manuellement."""
        try:
            # Vérifier si le widget existe avant d'y accéder
            if hasattr(self, 'drizzle_wht_spinbox') and self.drizzle_wht_spinbox.winfo_exists():
                float_value = self.drizzle_wht_threshold_var.get()
                percent_value = round(float_value * 100.0)
                # Mettre à jour le Spinbox sans déclencher sa propre commande
                self.drizzle_wht_spinbox.config(textvariable=tk.StringVar(value=f"{percent_value:.0f}"))
                # Remettre la liaison à la variable correcte pour la lecture future
                self.drizzle_wht_spinbox.config(textvariable=self.drizzle_wht_threshold_var) # Reconnecter pour la saisie? Non, on utilise command.
                # Juste mettre la valeur est plus simple:
                # self.drizzle_wht_spinbox.delete(0, tk.END)
                # self.drizzle_wht_spinbox.insert(0, f"{percent_value:.0f}") # Ceci est plus sûr
                # Encore plus simple: utiliser set() si Spinbox le supporte bien
                self.drizzle_wht_spinbox.set(f"{percent_value:.0f}")


        except (tk.TclError, AttributeError):
            # Peut arriver si appelé avant que spinbox soit prêt ou après destruction
            pass
    # --- FIN VERSION SIMPLIFIÉE ---

    # Nécéssite d'ajouter self._trace_id_wht = None dans __init__
    # et de lier la trace après la création du spinbox
    # -> Simplifions : On va juste appeler la mise à jour manuellement après chargement
    # Supprimer les lignes concernant _trace_id_wht et trace_add/trace_remove dans les 2 fonctions ci-dessus
    # Et dans apply_to_ui, après avoir set la variable, appeler _update_spinbox_from_float()

    def create_layout(self):
        """Crée la disposition des widgets."""
        # --- Structure principale ---
        main_frame = ttk.Frame(self.root); main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL); paned_window.pack(fill=tk.BOTH, expand=True)
        left_frame = ttk.Frame(paned_window, width=450, height=700); left_frame.pack_propagate(False); paned_window.add(left_frame, weight=1)
        right_frame = ttk.Frame(paned_window, width=750, height=700); right_frame.pack_propagate(False); paned_window.add(right_frame, weight=3)

        # --- Panneau Gauche ---
        # Language selection
        lang_frame = ttk.Frame(left_frame); lang_frame.pack(fill=tk.X, pady=(5, 5), padx=5)
        ttk.Label(lang_frame, text="Language/Langue:").pack(side=tk.LEFT, padx=(0, 5))
        self.language_combo = ttk.Combobox(lang_frame, textvariable=self.language_var, values=("en", "fr"), width=8, state="readonly")
        self.language_combo.pack(side=tk.LEFT)
        self.language_combo.bind("<<ComboboxSelected>>", self.change_language)

        # Notebook for control tabs
        control_notebook = ttk.Notebook(left_frame)
        control_notebook.pack(fill=tk.BOTH, expand=True, pady=(0,5), padx=5)

        # --- Onglet Empilement ---
        tab_stacking = ttk.Frame(control_notebook)
        control_notebook.add(tab_stacking, text=f' {self.tr("tab_stacking")} ') # Add padding in text

        # Folders Group
        self.folders_frame = ttk.LabelFrame(tab_stacking, text=self.tr("Folders"))
        self.folders_frame.pack(fill=tk.X, pady=5, padx=5)
        # Input Row
        in_subframe = ttk.Frame(self.folders_frame); in_subframe.pack(fill=tk.X, padx=5, pady=(5, 2))
        self.input_label = ttk.Label(in_subframe, text=self.tr("input_folder"), width=8, anchor="w"); self.input_label.pack(side=tk.LEFT)
        self.browse_input_button = ttk.Button(in_subframe, text=self.tr("browse_input_button"), command=self.file_handler.browse_input, width=10)
        self.browse_input_button.pack(side=tk.RIGHT)
        self.input_entry = ttk.Entry(in_subframe, textvariable=self.input_path)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        
        self.input_entry.bind("<FocusOut>", self._update_show_folders_button_state)
        self.input_entry.bind("<KeyRelease>", self._update_show_folders_button_state) # Update as user types/deletes


        # Output Row
        out_subframe = ttk.Frame(self.folders_frame); out_subframe.pack(fill=tk.X, padx=5, pady=(2, 5))
        self.output_label = ttk.Label(out_subframe, text=self.tr("output_folder"), width=8, anchor="w"); self.output_label.pack(side=tk.LEFT)
        self.browse_output_button = ttk.Button(out_subframe, text=self.tr("browse_output_button"), command=self.file_handler.browse_output, width=10)
        self.browse_output_button.pack(side=tk.RIGHT)
        self.output_entry = ttk.Entry(out_subframe, textvariable=self.output_path)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        # Reference Row
        ref_frame = ttk.Frame(self.folders_frame); ref_frame.pack(fill=tk.X, padx=5, pady=(2, 5))
        self.reference_label = ttk.Label(ref_frame, text=self.tr("reference_image"), width=8, anchor="w"); self.reference_label.pack(side=tk.LEFT)
        self.browse_ref_button = ttk.Button(ref_frame, text=self.tr("browse_ref_button"), command=self.file_handler.browse_reference, width=10)
        self.browse_ref_button.pack(side=tk.RIGHT)
        self.ref_entry = ttk.Entry(ref_frame, textvariable=self.reference_image_path)
        self.ref_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))

        # Stacking Options Group
        self.options_frame = ttk.LabelFrame(tab_stacking, text=self.tr("options"))
        self.options_frame.pack(fill=tk.X, pady=5, padx=5)

        # --- Drizzle Options Group ---
        self.drizzle_options_frame = ttk.LabelFrame(tab_stacking, text="Options Drizzle") # Sera traduit
        self.drizzle_options_frame.pack(fill=tk.X, pady=5, padx=5)

        # Checkbox pour activer Drizzle
        self.drizzle_check = ttk.Checkbutton(
            self.drizzle_options_frame,
            text="Activer Drizzle (expérimental)", # Sera traduit
            variable=self.use_drizzle_var,
            command=self._update_drizzle_options_state # Commande pour activer/désactiver échelle
        )
        self.drizzle_check.pack(anchor=tk.W, padx=5, pady=(5, 2))

        # Frame pour les options d'échelle (sera activé/désactivé)
        self.drizzle_scale_frame = ttk.Frame(self.drizzle_options_frame)
        self.drizzle_scale_frame.pack(fill=tk.X, padx=(20, 5), pady=(0, 5)) # Indentation
        self.drizzle_scale_label = ttk.Label(self.drizzle_scale_frame, text="Facteur Échelle :") # Sera traduit
        self.drizzle_scale_label.pack(side=tk.LEFT, padx=(0, 5))

        # Option 1: Utiliser des Radiobuttons (plus clair pour peu d'options)
        self.drizzle_radio_2x = ttk.Radiobutton(self.drizzle_scale_frame, text="x2", variable=self.drizzle_scale_var, value="2")
        self.drizzle_radio_3x = ttk.Radiobutton(self.drizzle_scale_frame, text="x3", variable=self.drizzle_scale_var, value="3")
        self.drizzle_radio_4x = ttk.Radiobutton(self.drizzle_scale_frame, text="x4", variable=self.drizzle_scale_var, value="4")
        self.drizzle_radio_2x.pack(side=tk.LEFT, padx=3)
        self.drizzle_radio_3x.pack(side=tk.LEFT, padx=3)
        self.drizzle_radio_4x.pack(side=tk.LEFT, padx=3)

        # Option 2: Utiliser une Combobox (si plus d'options prévues)
        # self.drizzle_scale_combo = ttk.Combobox(self.drizzle_scale_frame, textvariable=self.drizzle_scale_var, values=("2", "3", "4"), width=5, state="readonly")
        # self.drizzle_scale_combo.pack(side=tk.LEFT)

        # --- AJOUT Contrôle Seuil WHT ---
        # Créer un sous-frame pour le seuil pour le mettre sur une nouvelle ligne (optionnel)
        wht_frame = ttk.Frame(self.drizzle_scale_frame)
        wht_frame.pack(fill=tk.X, pady=(5, 0)) # Pack sous les radios

        self.drizzle_wht_label = ttk.Label(wht_frame, text="Seuil WHT%:") # Sera traduit
        self.drizzle_wht_label.pack(side=tk.LEFT, padx=(0, 5))

        # Utiliser un Spinbox pour le seuil (plus simple qu'un slider ici)
        # Valeurs de 10% à 100% avec pas de 5%
        self.drizzle_wht_spinbox = ttk.Spinbox(
            wht_frame,
            from_=10.0, # Min 10%
            to=100.0, # Max 100%
            increment=5.0, # Pas de 5%
            textvariable=self.drizzle_wht_threshold_var, # <- ERREUR CORRIGÉE: utiliser la variable DoubleVar * 100
            width=6,
            command=self._convert_spinbox_percent_to_float, # Pour convertir % en float 0-1
            format="%.0f" # Afficher sans décimale
        )
            # Liaison inverse pour mettre à jour le spinbox si la variable change
        self.drizzle_wht_threshold_var.trace_add("write", self._update_spinbox_from_float)
        self.drizzle_wht_spinbox.pack(side=tk.LEFT, padx=5)
        # Initialiser la valeur du spinbox depuis la variable
        self._update_spinbox_from_float()
        # --- FIN AJOUT Seuil WHT ---

        # Initialiser l'état des options d'échelle
        self._update_drizzle_options_state()
        # --- FIN NOUVEAU GROUPE DRIZZLE ---

        # Method Row
        method_frame = ttk.Frame(self.options_frame); method_frame.pack(fill=tk.X, padx=5, pady=5)
        self.stacking_method_label = ttk.Label(method_frame, text=self.tr("stacking_method")); self.stacking_method_label.pack(side=tk.LEFT, padx=(0, 5))
        self.stacking_combo = ttk.Combobox(method_frame, textvariable=self.stacking_mode, values=("mean", "median", "kappa-sigma", "winsorized-sigma"), width=15, state="readonly")
        self.stacking_combo.pack(side=tk.LEFT)
        self.kappa_label = ttk.Label(method_frame, text=self.tr("kappa_value")); self.kappa_label.pack(side=tk.LEFT, padx=(10, 2))
        self.kappa_spinbox = ttk.Spinbox(method_frame, from_=1.0, to=5.0, increment=0.1, textvariable=self.kappa, width=5)
        self.kappa_spinbox.pack(side=tk.LEFT)
        # Batch Row
        batch_frame = ttk.Frame(self.options_frame); batch_frame.pack(fill=tk.X, padx=5, pady=5)
        self.batch_size_label = ttk.Label(batch_frame, text=self.tr("batch_size")); self.batch_size_label.pack(side=tk.LEFT, padx=(0, 5))
        self.batch_spinbox = ttk.Spinbox(batch_frame, from_=3, to=500, increment=1, textvariable=self.batch_size, width=5) # Min batch size 3 for stats
        self.batch_spinbox.pack(side=tk.LEFT)

        # Hot Pixels Group
        self.hp_frame = ttk.LabelFrame(tab_stacking, text=self.tr("hot_pixels_correction"))
        self.hp_frame.pack(fill=tk.X, pady=5, padx=5)
        # HP Check Row
        hp_check_frame = ttk.Frame(self.hp_frame); hp_check_frame.pack(fill=tk.X, padx=5, pady=2)
        self.hot_pixels_check = ttk.Checkbutton(hp_check_frame, text=self.tr("perform_hot_pixels_correction"), variable=self.correct_hot_pixels)
        self.hot_pixels_check.pack(side=tk.LEFT, padx=(0, 10))
        # HP Params Row
        hp_params_frame = ttk.Frame(self.hp_frame); hp_params_frame.pack(fill=tk.X, padx=5, pady=(2,5))
        self.hot_pixel_threshold_label = ttk.Label(hp_params_frame, text=self.tr("hot_pixel_threshold")); self.hot_pixel_threshold_label.pack(side=tk.LEFT)
        self.hp_thresh_spinbox = ttk.Spinbox(hp_params_frame, from_=1.0, to=10.0, increment=0.1, textvariable=self.hot_pixel_threshold, width=5)
        self.hp_thresh_spinbox.pack(side=tk.LEFT, padx=5)
        self.neighborhood_size_label = ttk.Label(hp_params_frame, text=self.tr("neighborhood_size")); self.neighborhood_size_label.pack(side=tk.LEFT)
        self.hp_neigh_spinbox = ttk.Spinbox(hp_params_frame, from_=3, to=15, increment=2, textvariable=self.neighborhood_size, width=4) # Must be odd
        self.hp_neigh_spinbox.pack(side=tk.LEFT, padx=5)

        # Quality Weighting Group
        self.weighting_frame = ttk.LabelFrame(tab_stacking, text="Pondération par Qualité") # Sera traduit par update_ui_language
        self.weighting_frame.pack(fill=tk.X, pady=5, padx=5)

        self.use_weighting_check = ttk.Checkbutton(
            self.weighting_frame,
            text="Activer la pondération", # Sera traduit
            variable=self.use_weighting_var,
            command=self._update_weighting_options_state # Lie la commande
        )
        self.use_weighting_check.pack(anchor=tk.W, padx=5, pady=(5,2))

        self.weighting_options_frame = ttk.Frame(self.weighting_frame)
        self.weighting_options_frame.pack(fill=tk.X, padx=(20, 5), pady=(0, 5)) # Indentation

        metrics_frame = ttk.Frame(self.weighting_options_frame)
        metrics_frame.pack(fill=tk.X, pady=2)
        self.weight_metrics_label = ttk.Label(metrics_frame, text="Métriques:") # Sera traduit
        self.weight_metrics_label.pack(side=tk.LEFT, padx=(0, 5))
        self.weight_snr_check = ttk.Checkbutton(metrics_frame, text="SNR", variable=self.weight_snr_var) # Sera traduit
        self.weight_snr_check.pack(side=tk.LEFT, padx=5)
        self.weight_stars_check = ttk.Checkbutton(metrics_frame, text="Nb Étoiles", variable=self.weight_stars_var) # Sera traduit
        self.weight_stars_check.pack(side=tk.LEFT, padx=5)

        params_frame = ttk.Frame(self.weighting_options_frame)
        params_frame.pack(fill=tk.X, pady=2)
        self.snr_exp_label = ttk.Label(params_frame, text="Exp. SNR:") # Sera traduit
        self.snr_exp_label.pack(side=tk.LEFT, padx=(0, 2))
        self.snr_exp_spinbox = ttk.Spinbox(params_frame, from_=0.1, to=3.0, increment=0.1, textvariable=self.snr_exponent_var, width=5)
        self.snr_exp_spinbox.pack(side=tk.LEFT, padx=(0, 10))
        self.stars_exp_label = ttk.Label(params_frame, text="Exp. Étoiles:") # Sera traduit
        self.stars_exp_label.pack(side=tk.LEFT, padx=(0, 2))
        self.stars_exp_spinbox = ttk.Spinbox(params_frame, from_=0.1, to=3.0, increment=0.1, textvariable=self.stars_exponent_var, width=5)
        self.stars_exp_spinbox.pack(side=tk.LEFT, padx=(0, 10))
        self.min_w_label = ttk.Label(params_frame, text="Poids Min:") # Sera traduit
        self.min_w_label.pack(side=tk.LEFT, padx=(0, 2))
        self.min_w_spinbox = ttk.Spinbox(params_frame, from_=0.01, to=1.0, increment=0.01, textvariable=self.min_weight_var, width=5)
        self.min_w_spinbox.pack(side=tk.LEFT, padx=(0, 5))

        # Initialiser état des widgets de pondération
        self._update_weighting_options_state()

        # Post Processing Group
        self.post_proc_opts_frame = ttk.LabelFrame(tab_stacking, text=self.tr('post_proc_opts_frame_label'))
        self.post_proc_opts_frame.pack(fill=tk.X, pady=5, padx=5)
        self.cleanup_temp_check = ttk.Checkbutton(self.post_proc_opts_frame, text=self.tr("cleanup_temp_check_label"), variable=self.cleanup_temp_var)
        self.cleanup_temp_check.pack(side=tk.LEFT, padx=5, pady=5)

        # --- Onglet Aperçu ---
        tab_preview = ttk.Frame(control_notebook)
        control_notebook.add(tab_preview, text=f' {self.tr("tab_preview")} ') # Add padding in text

        # White Balance Group
        self.wb_frame = ttk.LabelFrame(tab_preview, text=self.tr("white_balance"))
        self.wb_frame.pack(fill=tk.X, pady=5, padx=5)
        self.wb_r_ctrls = self._create_slider_spinbox_group(self.wb_frame, "wb_r", 0.1, 5.0, 0.01, self.preview_r_gain)
        self.wb_g_ctrls = self._create_slider_spinbox_group(self.wb_frame, "wb_g", 0.1, 5.0, 0.01, self.preview_g_gain)
        self.wb_b_ctrls = self._create_slider_spinbox_group(self.wb_frame, "wb_b", 0.1, 5.0, 0.01, self.preview_b_gain)
        wb_btn_frame = ttk.Frame(self.wb_frame); wb_btn_frame.pack(fill=tk.X, pady=5)
        self.auto_wb_button = ttk.Button(wb_btn_frame, text=self.tr("auto_wb"), command=self.apply_auto_white_balance, state=tk.NORMAL if _tools_available else tk.DISABLED)
        self.auto_wb_button.pack(side=tk.LEFT, padx=5)
        self.reset_wb_button = ttk.Button(wb_btn_frame, text=self.tr("reset_wb"), command=self.reset_white_balance)
        self.reset_wb_button.pack(side=tk.LEFT, padx=5)

        # Stretch Group
        self.stretch_frame_controls = ttk.LabelFrame(tab_preview, text=self.tr("stretch_options"))
        self.stretch_frame_controls.pack(fill=tk.X, pady=5, padx=5)
        # Stretch Method Row
        stretch_method_frame = ttk.Frame(self.stretch_frame_controls); stretch_method_frame.pack(fill=tk.X, pady=2)
        self.stretch_method_label = ttk.Label(stretch_method_frame, text=self.tr("stretch_method")); self.stretch_method_label.pack(side=tk.LEFT, padx=(5,5))
        self.stretch_combo = ttk.Combobox(stretch_method_frame, textvariable=self.preview_stretch_method, values=("Linear", "Asinh", "Log"), width=15, state="readonly")
        self.stretch_combo.pack(side=tk.LEFT); self.stretch_combo.bind("<<ComboboxSelected>>", self._debounce_refresh_preview)
        # Stretch Sliders
        self.stretch_bp_ctrls = self._create_slider_spinbox_group(self.stretch_frame_controls, "stretch_bp", 0.0, 1.0, 0.001, self.preview_black_point, callback=self.update_histogram_lines_from_sliders)
        self.stretch_wp_ctrls = self._create_slider_spinbox_group(self.stretch_frame_controls, "stretch_wp", 0.0, 1.0, 0.001, self.preview_white_point, callback=self.update_histogram_lines_from_sliders)
        self.stretch_gamma_ctrls = self._create_slider_spinbox_group(self.stretch_frame_controls, "stretch_gamma", 0.1, 5.0, 0.01, self.preview_gamma) # Gamma uses general debounce
        # Stretch Buttons
        stretch_btn_frame = ttk.Frame(self.stretch_frame_controls); stretch_btn_frame.pack(fill=tk.X, pady=5)
        self.auto_stretch_button = ttk.Button(stretch_btn_frame, text=self.tr("auto_stretch"), command=self.apply_auto_stretch, state=tk.NORMAL if _tools_available else tk.DISABLED)
        self.auto_stretch_button.pack(side=tk.LEFT, padx=5)
        self.reset_stretch_button = ttk.Button(stretch_btn_frame, text=self.tr("reset_stretch"), command=self.reset_stretch)
        self.reset_stretch_button.pack(side=tk.LEFT, padx=5)

        # Adjustments Group (BCS)
        self.bcs_frame = ttk.LabelFrame(tab_preview, text=self.tr("image_adjustments"))
        self.bcs_frame.pack(fill=tk.X, pady=5, padx=5)
        self.brightness_ctrls = self._create_slider_spinbox_group(self.bcs_frame, "brightness", 0.1, 3.0, 0.01, self.preview_brightness)
        self.contrast_ctrls = self._create_slider_spinbox_group(self.bcs_frame, "contrast", 0.1, 3.0, 0.01, self.preview_contrast)
        self.saturation_ctrls = self._create_slider_spinbox_group(self.bcs_frame, "saturation", 0.0, 3.0, 0.01, self.preview_saturation)
        bcs_btn_frame = ttk.Frame(self.bcs_frame); bcs_btn_frame.pack(fill=tk.X, pady=5)
        self.reset_bcs_button = ttk.Button(bcs_btn_frame, text=self.tr("reset_bcs"), command=self.reset_brightness_contrast_saturation)
        self.reset_bcs_button.pack(side=tk.LEFT, padx=5)

        # --- Zone Progression (à la fin du panneau gauche) ---
        self.progress_frame = ttk.LabelFrame(left_frame, text=self.tr("progress"))
        self.progress_frame.pack(fill=tk.X, pady=(10, 0), padx=5, side=tk.BOTTOM) # Pack at bottom

        # Progress Bar
        self.progress_bar = ttk.Progressbar(self.progress_frame, maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=(5, 2))

        # Time Info Row (New version using grid)
        time_frame = ttk.Frame(self.progress_frame)
        time_frame.pack(fill=tk.X, padx=5, pady=2)

        # Configure columns for the grid within time_frame
        time_frame.columnconfigure(0, weight=0) # Column for "ETA:" label (no extra space)
        time_frame.columnconfigure(1, weight=1) # Column for ETA value (takes available space)
        time_frame.columnconfigure(2, weight=0) # Column for "Elapsed:" label (no extra space)
        time_frame.columnconfigure(3, weight=0) # Column for Elapsed value (no extra space)

        # Place widgets using grid
        self.remaining_time_label = ttk.Label(time_frame, text=self.tr("estimated_time"))
        self.remaining_time_label.grid(row=0, column=0, sticky='w') # West alignment

        # Removed width=9 to allow it to expand naturally
        self.remaining_time_value = ttk.Label(time_frame, textvariable=self.remaining_time_var, font=tkFont.Font(weight='bold'), anchor='w')
        self.remaining_time_value.grid(row=0, column=1, sticky='w', padx=(2, 10)) # West alignment, pad right

        self.elapsed_time_label = ttk.Label(time_frame, text=self.tr("elapsed_time"))
        self.elapsed_time_label.grid(row=0, column=2, sticky='e', padx=(5,0)) # East alignment, pad left

        # Kept width=9 and anchor='e' for consistent spacing of time value itself
        self.elapsed_time_value = ttk.Label(time_frame, textvariable=self.elapsed_time_var, font=tkFont.Font(weight='bold'), width=9, anchor='e')
        self.elapsed_time_value.grid(row=0, column=3, sticky='e', padx=(2,0)) # East alignment, pad left

    
        # File Count Info Row
        files_info_frame = ttk.Frame(self.progress_frame)
        files_info_frame.pack(fill=tk.X, padx=5, pady=2)
        # Combined Remaining/Total Label
        self.remaining_static_label = ttk.Label(files_info_frame, text=self.tr("Remaining:"))
        self.remaining_static_label.pack(side=tk.LEFT)
        self.remaining_value_label = ttk.Label(files_info_frame, textvariable=self.remaining_files_var, width=12, anchor='w') # Combined R/T
        self.remaining_value_label.pack(side=tk.LEFT, padx=(2,10))
        # Aligned Count Label
        self.aligned_files_label = ttk.Label(files_info_frame, textvariable=self.aligned_files_var, width=12, anchor='w') # Dynamic count
        self.aligned_files_label.pack(side=tk.LEFT, padx=(10,0))
        # Additional Folders Count (Right Aligned)
        self.additional_value_label = ttk.Label(files_info_frame, textvariable=self.additional_folders_var, anchor='e')
        self.additional_value_label.pack(side=tk.RIGHT)
        self.additional_static_label = ttk.Label(files_info_frame, text=self.tr("Additional:"))
        self.additional_static_label.pack(side=tk.RIGHT, padx=(0, 2))


        # --- MODIFIÉ: Status Text Area ---
        status_text_frame = ttk.Frame(self.progress_frame)
        status_text_font = tkFont.Font(family="Arial", size=8) # Police ok
        status_text_frame.pack(fill=tk.X, expand=False, padx=5, pady=(2, 5)) # expand=False est important

        # --- NOUVEAU: Bouton Copier Log ---
        self.copy_log_button = ttk.Button(status_text_frame, text="Copy",
                                          command=self._copy_log_to_clipboard, width=5)
        self.copy_log_button.pack(side=tk.RIGHT, padx=(2, 0), pady=0, anchor='ne') # Pack à droite, en haut

        # --- Barre de défilement ---
        self.status_scrollbar = ttk.Scrollbar(status_text_frame, orient="vertical") # Command sera lié plus bas
        self.status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=0)

        # --- Zone de Texte (plus haute) ---
        # AUGMENTER LA HAUTEUR ICI (par exemple de 5 à 8 ou 10)
        self.status_text = tk.Text(status_text_frame, height=8, wrap=tk.WORD, bd=0,
                                   font=status_text_font, relief=tk.FLAT, state=tk.DISABLED,
                                   yscrollcommand=self.status_scrollbar.set) # Lier scrollbar ici
        self.status_text.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=0)
        self.status_scrollbar.config(command=self.status_text.yview) # Lier scrollbar au Text
        # --- Fin Modification Log ---

        # --- Boutons de Contrôle (sous la zone de progression) ---
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=(5, 5), padx=5, side=tk.BOTTOM) # Pack at bottom

        # Use Accent style if available
        try:
            style = ttk.Style()
            accent_style = 'Accent.TButton' if 'Accent.TButton' in style.element_names() else 'TButton'
        except tk.TclError:
            accent_style = 'TButton'
        self.start_button = ttk.Button(control_frame, text=self.tr("start"), command=self.start_processing, style=accent_style)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5, ipady=2)
        self.stop_button = ttk.Button(control_frame, text=self.tr("stop"), command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5, ipady=2)
        
        # Placed before "Add Folder" button
        self.show_folders_button = ttk.Button(control_frame, text="View Inputs", command=self._show_input_folder_list, state=tk.DISABLED) # Start disabled until input is set
        self.show_folders_button.pack(side=tk.RIGHT, padx=5, pady=5, ipady=2) # Pack next
        
        # Le bouton "Ajouter Dossier" peut maintenant être utilisé avant le démarrage
        self.add_files_button = ttk.Button(control_frame, text=self.tr("add_folder_button"), command=self.file_handler.add_folder, state=tk.NORMAL) # Commence NORMAL
        self.add_files_button.pack(side=tk.RIGHT, padx=5, pady=5, ipady=2) # pack à droite

        # --- NOUVEAU : Bouton Ouvrir Dossier Sortie ---
        # Placé à gauche du bouton Ajouter Dossier
        self.open_output_button = ttk.Button(control_frame, text="Open Output", # Sera traduit
                                              command=self._open_output_folder, state=tk.DISABLED) # Commence Désactivé
        self.open_output_button.pack(side=tk.RIGHT, padx=5, pady=5, ipady=2)
        # --- Fin Nouveau Bouton ---

        # --- Panneau Droit (Aperçu et Histogramme) ---
        # Preview Frame
        self.preview_frame = ttk.LabelFrame(right_frame, text=self.tr("preview"))
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=(5,5), padx=5) # Pad top and bottom
        # Canvas for image display
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="#1E1E1E", highlightthickness=0) # Dark background
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        # Histogram Frame (at the bottom of right panel)
        self.histogram_frame = ttk.LabelFrame(right_frame, text=self.tr("histogram"))
        hist_fig_height_inches = 2.2; hist_fig_dpi = 80
        hist_height_pixels = int(hist_fig_height_inches * hist_fig_dpi * 1.1)
        self.histogram_frame.pack(fill=tk.X, expand=False, pady=(0,5), padx=5, side=tk.BOTTOM)
        self.histogram_frame.pack_propagate(False); self.histogram_frame.config(height=hist_height_pixels)

        # Create and pack HistogramWidget inside its frame
        self.histogram_widget = HistogramWidget(self.histogram_frame, range_change_callback=self.update_stretch_from_histogram)
        self.histogram_widget.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=(0,2), pady=(0,2))

        # Add Reset Zoom button next to histogram
        self.hist_reset_btn = ttk.Button(self.histogram_frame, text="R", command=self.histogram_widget.reset_zoom, width=2)
        self.hist_reset_btn.pack(side=tk.RIGHT, anchor=tk.NE, padx=(0,2), pady=2)

        # Store references AFTER all widgets are created
        self._store_widget_references()


    def _update_show_folders_button_state(self, event=None):
        """Enables/disables the 'View Input Folders' button."""
        try:
            if hasattr(self, 'show_folders_button') and self.show_folders_button.winfo_exists():
                if self.input_path.get() and os.path.isdir(self.input_path.get()):
                    self.show_folders_button.config(state=tk.NORMAL)
                else:
                    self.show_folders_button.config(state=tk.DISABLED)
        except tk.TclError:
            pass # Widget might not exist yet or be destroyed



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

    def _store_widget_references(self):
        """Stores references to widgets that need language updates."""
        # Essayer de trouver le widget Notebook de manière robuste
        notebook_widget = None
        try:
            # Trouver le widget Notebook pour pouvoir référencer les onglets
            # Cette partie dépend de la structure exacte de create_layout
            # On suppose: root -> main_frame -> paned_window -> left_frame -> control_notebook
            main_frame = self.root.winfo_children()[0]
            paned_window = main_frame.winfo_children()[0]
            # Obtient le widget du panneau de gauche via l'API PanedWindow
            left_frame_widget_name = paned_window.panes()[0]
            left_frame_widget = self.root.nametowidget(left_frame_widget_name)
            # Cherche le Notebook dans les enfants directs du panneau gauche
            for child in left_frame_widget.winfo_children():
                if isinstance(child, ttk.Notebook):
                    notebook_widget = child
                    break
        except tk.TclError: # Erreur si les widgets n'existent pas encore
            pass
        except IndexError: # Erreur si la structure attendue n'est pas là
             pass
        except Exception as e:
            print(f"Warning: Could not find Notebook widget reliably: {e}")

        # Le dictionnaire qui fait le lien entre clé de traduction et widget
        self.widgets_to_translate = {

            # --- Onglets du Notebook ---
            # La valeur est un tuple: (widget_notebook, index_onglet)
            "tab_stacking": (notebook_widget, 0) if notebook_widget and notebook_widget.index("end") > 0 else None,
            "tab_preview": (notebook_widget, 1) if notebook_widget and notebook_widget.index("end") > 1 else None,

            # --- LabelFrames (cadres avec titre) ---
            "Folders": getattr(self, 'folders_frame', None),
            "options": getattr(self, 'options_frame', None),
            "drizzle_options_frame_label": getattr(self, 'drizzle_options_frame', None), # Titre groupe Drizzle
            "hot_pixels_correction": getattr(self, 'hp_frame', None),
            "quality_weighting_frame": getattr(self, 'weighting_frame', None), # Titre groupe Poids
            "post_proc_opts_frame_label": getattr(self, 'post_proc_opts_frame', None),
            "white_balance": getattr(self, 'wb_frame', None),
            "stretch_options": getattr(self, 'stretch_frame_controls', None),
            "image_adjustments": getattr(self, 'bcs_frame', None),
            "progress": getattr(self, 'progress_frame', None),
            "preview": getattr(self, 'preview_frame', None),
            "histogram": getattr(self, 'histogram_frame', None),

            # --- Labels (étiquettes simples) ---
            "input_folder": getattr(self, 'input_label', None),
            "output_folder": getattr(self, 'output_label', None),
            "reference_image": getattr(self, 'reference_label', None),
            "stacking_method": getattr(self, 'stacking_method_label', None),
            "kappa_value": getattr(self, 'kappa_label', None),
            "batch_size": getattr(self, 'batch_size_label', None),
            "drizzle_scale_label": getattr(self, 'drizzle_scale_label', None), # Label Échelle Drizzle
            # ---> AJOUT DE LA LIGNE POUR LE LABEL WHT <---
            "drizzle_wht_threshold_label": getattr(self, 'drizzle_wht_label', None), # Label Seuil WHT
            # ---> FIN AJOUT <---
            "hot_pixel_threshold": getattr(self, 'hot_pixel_threshold_label', None),
            "neighborhood_size": getattr(self, 'neighborhood_size_label', None),
            "weighting_metrics_label": getattr(self, 'weight_metrics_label', None),
            "snr_exponent_label": getattr(self, 'snr_exp_label', None),
            "stars_exponent_label": getattr(self, 'stars_exp_label', None),
            "min_weight_label": getattr(self, 'min_w_label', None),
            "wb_r": getattr(self, 'wb_r_ctrls', {}).get('label'), # Labels dans les groupes slider/spinbox
            "wb_g": getattr(self, 'wb_g_ctrls', {}).get('label'),
            "wb_b": getattr(self, 'wb_b_ctrls', {}).get('label'),
            "stretch_method": getattr(self, 'stretch_method_label', None),
            "stretch_bp": getattr(self, 'stretch_bp_ctrls', {}).get('label'),
            "stretch_wp": getattr(self, 'stretch_wp_ctrls', {}).get('label'),
            "stretch_gamma": getattr(self, 'stretch_gamma_ctrls', {}).get('label'),
            "brightness": getattr(self, 'brightness_ctrls', {}).get('label'),
            "contrast": getattr(self, 'contrast_ctrls', {}).get('label'),
            "saturation": getattr(self, 'saturation_ctrls', {}).get('label'),
            "estimated_time": getattr(self, 'remaining_time_label', None),
            "elapsed_time": getattr(self, 'elapsed_time_label', None),
            "Remaining:": getattr(self, 'remaining_static_label', None), # Label statique
            "Additional:": getattr(self, 'additional_static_label', None), # Label statique

            # --- Buttons ---
            "browse_input_button": getattr(self, 'browse_input_button', None),
            "browse_output_button": getattr(self, 'browse_output_button', None),
            "browse_ref_button": getattr(self, 'browse_ref_button', None),
            "auto_wb": getattr(self, 'auto_wb_button', None),
            "reset_wb": getattr(self, 'reset_wb_button', None),
            "auto_stretch": getattr(self, 'auto_stretch_button', None),
            "reset_stretch": getattr(self, 'reset_stretch_button', None),
            "reset_bcs": getattr(self, 'reset_bcs_button', None),
            "start": getattr(self, 'start_button', None),
            "stop": getattr(self, 'stop_button', None),
            "add_folder_button": getattr(self, 'add_files_button', None),
            "show_folders_button_text": getattr(self, 'show_folders_button', None),
            "copy_log_button_text": getattr(self, 'copy_log_button', None),
            "open_output_button_text": getattr(self, 'open_output_button', None),
            # hist_reset_btn n'a pas besoin de traduction (juste "R")

            # --- Checkbuttons ---
            "drizzle_activate_check": getattr(self, 'drizzle_check', None), # Checkbox Drizzle
            "perform_hot_pixels_correction": getattr(self, 'hot_pixels_check', None),
            "enable_weighting_check": getattr(self, 'use_weighting_check', None),
            "weight_snr_check": getattr(self, 'weight_snr_check', None),
            "weight_stars_check": getattr(self, 'weight_stars_check', None),
            "cleanup_temp_check_label": getattr(self, 'cleanup_temp_check', None),

            # --- Radiobuttons (pour leur texte si besoin) ---
            "drizzle_radio_2x_label": getattr(self, 'drizzle_radio_2x', None),
            "drizzle_radio_3x_label": getattr(self, 'drizzle_radio_3x', None),
            "drizzle_radio_4x_label": getattr(self, 'drizzle_radio_4x', None),
        }

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

      
# Inside SeestarStackerGUI class in main_window.py

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

    def _try_show_first_input_image(self):
        input_folder = self.input_path.get()
        if not hasattr(self, 'preview_manager') or not hasattr(self, 'histogram_widget'): return
        if not input_folder or not os.path.isdir(input_folder):
            if hasattr(self, 'preview_manager'): self.preview_manager.clear_preview(self.tr("Input folder not found"))
            if hasattr(self, 'histogram_widget'): self.histogram_widget.plot_histogram(None)
            return
        try:
            files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith((".fit", ".fits"))])
            if not files:
                if hasattr(self, 'preview_manager'): self.preview_manager.clear_preview(self.tr("No FITS files in input"))
                if hasattr(self, 'histogram_widget'): self.histogram_widget.plot_histogram(None)
                return
            first_image_path = os.path.join(input_folder, files[0])
            self.update_progress_gui(f"Chargement aperçu: {files[0]}...", None)
            img_data = load_and_validate_fits(first_image_path)
            if img_data is None: raise ValueError(f"Échec chargement/validation {files[0]}")
            header = fits.getheader(first_image_path)
            img_for_preview = img_data
            if img_data.ndim == 2:
                bayer_pattern = header.get("BAYERPAT", self.settings.bayer_pattern); valid_bayer_patterns = ["GRBG", "RGGB", "GBRG", "BGGR"]
                if isinstance(bayer_pattern, str) and bayer_pattern.upper() in valid_bayer_patterns:
                    try: img_for_preview = debayer_image(img_data, bayer_pattern.upper())
                    except ValueError as debayer_err: self.update_progress_gui(f"⚠️ {self.tr('Error during debayering')}: {debayer_err}. Affichage N&B.", None)
            self.current_preview_data = img_for_preview.copy(); self.current_stack_header = header.copy() if header else None
            self.refresh_preview()
            if self.current_stack_header: self.update_image_info(self.current_stack_header)
        except FileNotFoundError:
            if hasattr(self, 'preview_manager'): self.preview_manager.clear_preview(self.tr("Input folder not found"))
            if hasattr(self, 'histogram_widget'): self.histogram_widget.plot_histogram(None)
        except ValueError as ve:
            self.update_progress_gui(f"⚠️ {self.tr('Error loading preview image')}: {ve}", None)
            if hasattr(self, 'preview_manager'): self.preview_manager.clear_preview(self.tr("Error loading preview (invalid format?)"))
            if hasattr(self, 'histogram_widget'): self.histogram_widget.plot_histogram(None)
        except Exception as e:
            self.update_progress_gui(f"⚠️ {self.tr('Error loading preview image')}: {e}", None); traceback.print_exc(limit=2)
            if hasattr(self, 'preview_manager'): self.preview_manager.clear_preview(self.tr("Error loading preview"))
            if hasattr(self, 'histogram_widget'): self.histogram_widget.plot_histogram(None)

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

    # --- MODIFIÉ: update_additional_folders_display ---
    def update_additional_folders_display(self):
        """Met à jour l'affichage du nombre de dossiers supplémentaires."""
        count = 0
        if self.processing and hasattr(self, 'queued_stacker'):
            # Pendant le traitement, lire la liste du backend
            with self.queued_stacker.folders_lock: count = len(self.queued_stacker.additional_folders)
        else:
            # Avant le traitement, lire la liste du GUI
            count = len(self.additional_folders_to_process)

        try:
            if count == 0: self.additional_folders_var.set(self.tr('no_additional_folders'))
            elif count == 1: self.additional_folders_var.set(self.tr('1 additional folder'))
            else: self.additional_folders_var.set(self.tr('{count} additional folders', default="{count} add. folders").format(count=count))
        except tk.TclError: pass
        except AttributeError: pass

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

    def _save_settings_and_destroy(self):
        try:
            if self.root.winfo_exists(): self.settings.window_geometry = self.root.geometry()
        except tk.TclError: pass
        self.settings.update_from_ui(self); self.settings.save_settings()
        print("Fermeture de l'application."); self.root.destroy()

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

    # --- MÉTHODE _processing_finished MODIFIÉE ---
    def _processing_finished(self):
        """Actions finales après la fin ou l'arrêt du traitement."""
        if not self.processing: return # Évite exécutions multiples
        self.processing = False # Marquer comme terminé

        if hasattr(self, 'progress_manager'):
            self.progress_manager.stop_timer()
            # Arrêter la barre de progression indéterminée si elle l'était
            try:
                pb = self.progress_manager.progress_bar
                if pb.winfo_exists() and pb['mode'] == 'indeterminate':
                    pb.stop()
                    pb.config(mode='determinate')
                # Optionnel: Mettre la barre à 100% (ou à la dernière valeur connue)
                # last_progress = pb['value'] # Récupérer la valeur avant indeterminate? Non fiable.
                if not self.queued_stacker.processing_error: # Mettre à 100 si pas d'erreur
                    pb.configure(value=100)
            except (tk.TclError, AttributeError): pass

        # --- Récupération état final du backend ---
        final_message_for_status_bar = self.tr("stacking_finished")
        final_stack_path = None; processing_error_details = None; images_stacked = 0
        aligned_count = 0; failed_align_count = 0; failed_stack_count = 0; skipped_count = 0
        processed_files_count = 0; total_exposure = 0.0
        was_stopped_by_user = False; output_folder_exists = False; can_open_output = False
        final_stack_exists = False; is_drizzle_result = False # Flag Drizzle

        if hasattr(self, "queued_stacker"):
            q_stacker = self.queued_stacker
            final_stack_path = getattr(q_stacker, 'final_stacked_path', None)
            # Utiliser aligned_count pour Drizzle, images_in_cumulative_stack pour classique
            is_drizzle_result = getattr(q_stacker, 'drizzle_active_session', False) and not getattr(q_stacker, 'stop_processing', False) and getattr(q_stacker, 'processing_error', None) is None and final_stack_path is not None
            images_stacked = getattr(q_stacker, 'aligned_files_count', 0) if is_drizzle_result else getattr(q_stacker, 'images_in_cumulative_stack', 0)

            aligned_count = getattr(q_stacker, 'aligned_files_count', 0)
            failed_align_count = getattr(q_stacker, 'failed_align_count', 0)
            failed_stack_count = getattr(q_stacker, 'failed_stack_count', 0)
            skipped_count = getattr(q_stacker, 'skipped_files_count', 0)
            processed_files_count = getattr(q_stacker, 'processed_files_count', 0)
            total_exposure = getattr(q_stacker, 'total_exposure_seconds', 0.0)
            was_stopped_by_user = getattr(q_stacker, 'stop_processing', False)
            processing_error_details = getattr(q_stacker, 'processing_error', None)

            default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
            try:
                if hasattr(self, 'aligned_files_var'): self.aligned_files_var.set(default_aligned_fmt.format(count=aligned_count))
            except tk.TclError: pass

            if hasattr(self, 'output_path') and self.output_path.get(): output_folder_exists = os.path.isdir(self.output_path.get())
            final_stack_exists = final_stack_path and os.path.exists(final_stack_path)
            # On peut ouvrir si le dossier existe ET (le stack final existe OU il n'y a pas eu d'erreur critique)
            can_open_output = output_folder_exists and (final_stack_exists or not processing_error_details)
        else:
            final_message_for_status_bar = "Erreur: Stacker non trouvé."; processing_error_details = final_message_for_status_bar
            if hasattr(self, 'output_path') and self.output_path.get(): output_folder_exists = os.path.isdir(self.output_path.get())
            can_open_output = output_folder_exists and not processing_error_details

        # --- Déterminer message statut final ---
        if was_stopped_by_user: final_message_for_status_bar = self.tr("processing_stopped")
        elif processing_error_details: final_message_for_status_bar = self.tr('stacking_error_msg')
        elif not final_stack_exists: final_message_for_status_bar = self.tr("no_stacks_created")
        elif is_drizzle_result: final_message_for_status_bar = self.tr("Drizzle complete!", default="Drizzle complete!")
        else: final_message_for_status_bar = self.tr("stacking_complete_msg")
        # Mettre à jour la barre de statut simple (pas le log principal)
        if hasattr(self, 'progress_manager'): self.progress_manager.update_progress(final_message_for_status_bar, 100)

        # --- Charger aperçu final ---
        preview_load_error_msg = None
        if final_stack_exists:
            try:
                # self.update_progress_gui(f"Chargement aperçu final...", None) # Optionnel
                final_image_data = load_and_validate_fits(final_stack_path)
                if final_image_data is not None:
                    final_header = fits.getheader(final_stack_path)
                    self.current_preview_data = final_image_data; self.current_stack_header = final_header
                    self.refresh_preview();
                    if final_header: self.update_image_info(final_header)
                else: preview_load_error_msg = f"{self.tr('Error loading final stack preview')}: load returned None."
            except Exception as preview_load_error:
                preview_load_error_msg = f"{self.tr('Error loading final stack preview')}: {preview_load_error}"; traceback.print_exc(limit=2)
                messagebox.showerror(self.tr("Preview Error"), f"{self.tr('Error loading final preview')}:\n{preview_load_error}")

        # --- Générer le résumé ---
        summary_lines = []; summary_title = self.tr("processing_report_title")
        status_text = ""; final_stack_type = self.tr("Classic", default="Classic") # <-- Traduisible
        if was_stopped_by_user: status_text = self.tr('processing_stopped')
        elif processing_error_details: status_text = f"ERREUR ({processing_error_details})"
        elif not final_stack_exists: status_text = "Terminé (Aucun stack final créé)"
        elif is_drizzle_result: status_text = self.tr("Drizzle Complete", default="Drizzle Complete"); final_stack_type = "Drizzle"
        else: status_text = self.tr("Stacking Complete", default="Stacking Complete")
        summary_lines.append(f"{self.tr('Status', default='Status')}: {status_text}")

        elapsed_total_seconds = 0;
        if self.global_start_time: elapsed_total_seconds = time.monotonic() - self.global_start_time
        summary_lines.append(f"{self.tr('Total Processing Time', default='Total Processing Time')}: {self._format_duration(elapsed_total_seconds)}")
        summary_lines.append(f"{self.tr('Final Stack Type', default='Final Stack Type')}: {final_stack_type}")
        summary_lines.append(f"{self.tr('Files Attempted', default='Files Attempted')}: {processed_files_count}")
        total_rejected = failed_align_count + failed_stack_count + skipped_count
        summary_lines.append(f"{self.tr('Files Rejected (Total)', default='Files Rejected (Total)')}: {total_rejected} ({self.tr('Align', default='Align')}: {failed_align_count}, {self.tr('Stack Err', default='Stack Err')}: {failed_stack_count}, {self.tr('Other', default='Other')}: {skipped_count})")
        summary_lines.append(f"{self.tr('Images in Final Stack', default='Images in Final Stack')} ({final_stack_type}): {images_stacked}") # Utilise le bon compte
        summary_lines.append(f"{self.tr('Total Exposure (Final Stack)', default='Total Exposure (Final Stack)')}: {self._format_duration(total_exposure)}")
        if final_stack_exists: summary_lines.append(f"{self.tr('Final Stack File', default='Final Stack File')}:\n  {final_stack_path}")
        else: summary_lines.append(self.tr('Final Stack File: Not created or not found.', default='Final Stack File: Not created or not found.'))
        full_summary_text_for_dialog = "\n".join(summary_lines)

# --- Afficher Dialogue (ou erreur) ---
        if processing_error_details:
            messagebox.showerror(
                self.tr("error"), f"{self.tr('stacking_error_msg')}\n{processing_error_details}"
            )
        elif not was_stopped_by_user:
            # Passer can_open_output au dialogue
            self._show_summary_dialog(
                summary_title, full_summary_text_for_dialog, can_open_output
            )  # Modifié ici

        # --- Réinitialiser UI ---
        self._set_parameter_widgets_state(tk.NORMAL)  # Réactive les contrôles
        if hasattr(self, "start_button"):
            try:
                self.start_button.config(state=tk.NORMAL)
            except tk.TclError:
                pass
        if hasattr(self, "stop_button"):
            try:
                self.stop_button.config(state=tk.DISABLED)
            except tk.TclError:
                pass
        # Activer/Désactiver bouton Ouvrir Sortie basé sur can_open_output
        if hasattr(self, "open_output_button"):
            try:
                self.open_output_button.config(state=tk.NORMAL if can_open_output else tk.DISABLED)
            except tk.TclError:
                pass
        if hasattr(self, "remaining_time_var"):
            try:
                self.remaining_time_var.set("00:00:00")
            except tk.TclError:
                pass
        # GC final
        if 'gc' in globals() or 'gc' in locals(): gc.collect()

    # --- MÉTHODE _show_summary_dialog MODIFIÉE ---
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

    # --- MÉTHODE start_processing MODIFIÉE ---
    def start_processing(self):
        """Démarre le traitement, affiche l'avertissement Drizzle, gère la config et lance le thread backend."""

        # --- Désactivation immédiate bouton Start ---
        if hasattr(self, 'start_button'):
            try: self.start_button.config(state=tk.DISABLED)
            except tk.TclError: pass

######################################################
        # --- Validation des chemins (inchangée) ---
        input_folder = self.input_path.get()
        output_folder = self.output_path.get()

        if not input_folder or not output_folder:
            messagebox.showerror(self.tr("error"), self.tr("select_folders"))
            if hasattr(self, "start_button"):
                try:
                    self.start_button.config(state=tk.NORMAL)
                except tk.TclError:
                    pass
            return

        if not os.path.isdir(input_folder):
            messagebox.showerror(
                self.tr("error"), f"{self.tr('input_folder_invalid')}:\n{input_folder}"
            )
            if hasattr(self, "start_button"):
                try:
                    self.start_button.config(state=tk.NORMAL)
                except tk.TclError:
                    pass
            return

        if not os.path.isdir(output_folder):
            try:
                os.makedirs(output_folder, exist_ok=True)
                self.update_progress_gui(
                    f"{self.tr('Output folder created')}: {output_folder}", None
                )
            except Exception as e:
                messagebox.showerror(
                    self.tr("error"), f"{self.tr('output_folder_invalid')}:\n{output_folder}\n{e}"
                )
            if hasattr(self, "start_button"):
                try:
                    self.start_button.config(state=tk.NORMAL)
                except tk.TclError:
                    pass
            return

        try:
            if not any(f.lower().endswith((".fit", ".fits")) for f in os.listdir(input_folder)):
                if not messagebox.askyesno(self.tr("warning"), self.tr("no_fits_found")):
                    if hasattr(self, "start_button"):
                        try:
                            self.start_button.config(state=tk.NORMAL)
                        except tk.TclError:
                            pass
                    return
        except Exception as e:
            messagebox.showerror(
                self.tr("error"), f"{self.tr('Error reading input folder')}:\n{e}"
            )

        if hasattr(self, "start_button"):
            try:
                self.start_button.config(state=tk.NORMAL)
            except tk.TclError:
                pass
        return
            ### === NOUVEAU : AVERTISSEMENT DRIZZLE === ###
        # Vérifier si la variable Drizzle existe et est cochée
        drizzle_enabled = False
        if hasattr(self, 'use_drizzle_var'):
            try:
                drizzle_enabled = self.use_drizzle_var.get()
            except tk.TclError: pass # Ignorer si le widget n'existe pas encore

        if drizzle_enabled:
            # Récupérer les textes traduits
            warning_title = self.tr('drizzle_warning_title')
            warning_text = self.tr('drizzle_warning_text')
            # Afficher la boîte de dialogue modale
            continue_with_drizzle = messagebox.askyesno(warning_title, warning_text, parent=self.root) # parent=self.root est important
            # Si l'utilisateur clique "Non", annuler le démarrage
            if not continue_with_drizzle:
                self.update_progress_gui("ⓘ Démarrage annulé par l'utilisateur (Drizzle).", None)
                # Réactiver le bouton Start et sortir
                if hasattr(self, 'start_button'):
                    try: self.start_button.config(state=tk.NORMAL)
                    except tk.TclError: pass
                return # Arrêter la fonction ici
        ### === FIN AVERTISSEMENT DRIZZLE === ###
##############
        # --- Le reste de la logique de démarrage (inchangée par rapport à avant) ---
        self.processing = True
        self.time_per_image = 0
        self.global_start_time = time.monotonic()

        default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
        self.aligned_files_var.set(default_aligned_fmt.format(count=0))

        folders_to_pass_to_backend = list(self.additional_folders_to_process)
        self.additional_folders_to_process = []
        self.update_additional_folders_display()

        self._set_parameter_widgets_state(tk.DISABLED)  # Désactiver les contrôles

        if hasattr(self, "stop_button"):
            try:
                self.stop_button.config(state=tk.NORMAL)
            except tk.TclError:
                pass

        if hasattr(self, "open_output_button"):
            try:
                self.open_output_button.config(state=tk.DISABLED)
            except tk.TclError:
                pass

        if hasattr(self, "progress_manager"):
            self.progress_manager.reset()
            self.progress_manager.start_timer()

        if hasattr(self, "status_text"):
            try:
                self.status_text.config(state=tk.NORMAL)
                self.status_text.delete(1.0, tk.END)
                self.status_text.insert(
                    tk.END, f"--- {self.tr('stacking_start', default='--- Starting Processing ---')} ---\n"
                )
                self.status_text.config(state=tk.DISABLED)
            except tk.TclError:
                pass
                self.settings.update_from_ui(self); validation_messages = self.settings.validate_settings()
        if validation_messages: self.update_progress_gui("⚠️ Paramètres ajustés:", None);
        for msg in validation_messages: self.update_progress_gui(f"  - {msg}", None); self.settings.apply_to_ui(self)

        # Configurer QueuedStacker (inclut batch size auto)
        self.queued_stacker.stacking_mode = self.settings.stacking_mode; self.queued_stacker.kappa = self.settings.kappa
        requested_batch_size = self.settings.batch_size; final_batch_size_for_backend = 0
        if requested_batch_size <= 0:
             self.update_progress_gui("🧠 Estimation taille lot auto...", None); sample_img_path = None
             if self.settings.input_folder and os.path.isdir(self.settings.input_folder):
                 fits_files = [f for f in os.listdir(self.settings.input_folder) if f.lower().endswith(('.fit', '.fits'))]
                 if fits_files: sample_img_path = os.path.join(self.settings.input_folder, fits_files[0])
             try: estimated_size = estimate_batch_size(sample_image_path=sample_img_path); final_batch_size_for_backend = estimated_size; self.update_progress_gui(f"✅ Taille lot auto estimée: {estimated_size}", None)
             except Exception as est_err: self.update_progress_gui(f"⚠️ Erreur estimation taille lot: {est_err}. Utilisation défaut (10).", None); final_batch_size_for_backend = 10
        else: final_batch_size_for_backend = requested_batch_size
        self.queued_stacker.batch_size = final_batch_size_for_backend

        self.queued_stacker.correct_hot_pixels = self.settings.correct_hot_pixels; self.queued_stacker.hot_pixel_threshold = self.settings.hot_pixel_threshold
        self.queued_stacker.neighborhood_size = self.settings.neighborhood_size; self.queued_stacker.bayer_pattern = self.settings.bayer_pattern
        self.queued_stacker.perform_cleanup = self.cleanup_temp_var.get(); self.queued_stacker.aligner.reference_image_path = self.settings.reference_image_path or None
        self.update_progress_gui(f"ⓘ Taille de lot pour traitement : {self.queued_stacker.batch_size}", None)

        # Démarrer le backend (passe maintenant use_drizzle depuis la variable UI)
        processing_started = self.queued_stacker.start_processing(
            input_folder, output_folder, self.settings.reference_image_path,
            initial_additional_folders=folders_to_pass_to_backend,
            # Pondération
            use_weighting=self.settings.use_quality_weighting, weight_snr=self.settings.weight_by_snr,
            weight_stars=self.settings.weight_by_stars, snr_exp=self.settings.snr_exponent,
            stars_exp=self.settings.stars_exponent, min_w=self.settings.min_weight,
            # Drizzle (prend la valeur depuis la variable UI)
            use_drizzle=self.use_drizzle_var.get(), # <--- Utilise la variable ici
            drizzle_scale=float(self.drizzle_scale_var.get()),
            drizzle_wht_threshold=self.drizzle_wht_threshold_var.get()
        )
##################
        if processing_started:
            if hasattr(self, 'stop_button'):
                try:
                    self.stop_button.config(state=tk.NORMAL)
                except tk.TclError:
                    pass
            self.thread = threading.Thread(target=self._track_processing_progress, daemon=True, name="GUI_ProgressTracker")
            self.thread.start()
        else:  # Échec démarrage backend (très rare maintenant)
            self.update_progress_gui("ⓘ Demande démarrage ignorée (traitement déjà en cours?).", None)
            if hasattr(self, 'stop_button'):
                try:
                    self.stop_button.config(state=tk.NORMAL)
                except tk.TclError:
                    pass
            self.processing = True  # Assurer état actif
# --- DANS seestar/gui/main_window.py ---
# ... (imports, classe SeestarStackerGUI, autres méthodes...) ...

    # --- MÉTHODE start_processing CORRIGÉE ---
    def start_processing(self):
        """Démarre le traitement, affiche l'avertissement Drizzle, gère la config et lance le thread backend."""

        # --- Désactivation immédiate bouton Start ---
        if hasattr(self, 'start_button'):
            try: self.start_button.config(state=tk.DISABLED)
            except tk.TclError: pass

        # --- Validation des chemins ---
        input_folder = self.input_path.get()
        output_folder = self.output_path.get()
        if not input_folder or not output_folder:
            messagebox.showerror(self.tr("error"), self.tr("select_folders"))
            if hasattr(self, 'start_button'):
                try: self.start_button.config(state=tk.NORMAL)
                except tk.TclError: pass; return
        if not os.path.isdir(input_folder):
            messagebox.showerror(self.tr("error"), f"{self.tr('input_folder_invalid')}:\n{input_folder}")
            if hasattr(self, 'start_button'):
                try: self.start_button.config(state=tk.NORMAL)
                except tk.TclError: pass; return
        if not os.path.isdir(output_folder):
            try: os.makedirs(output_folder, exist_ok=True); self.update_progress_gui(f"{self.tr('Output folder created')}: {output_folder}", None)
            except Exception as e: messagebox.showerror(self.tr("error"), f"{self.tr('output_folder_invalid')}:\n{output_folder}\n{e}");
            if hasattr(self, 'start_button'):
                try: self.start_button.config(state=tk.NORMAL)
                except tk.TclError: pass; return
        try:
            if not any(f.lower().endswith((".fit", ".fits")) for f in os.listdir(input_folder)):
                if not messagebox.askyesno(self.tr("warning"), self.tr("no_fits_found")):
                    if hasattr(self, 'start_button'):
                        try: self.start_button.config(state=tk.NORMAL)
                        except tk.TclError: pass; return
        except Exception as e: messagebox.showerror(self.tr("error"), f"{self.tr('Error reading input folder')}:\n{e}");
        if hasattr(self, 'start_button'):
            try: self.start_button.config(state=tk.NORMAL)
            except tk.TclError: pass; return #<-- Correction: Ajout return ici en cas d'erreur lecture

        # === AVERTISSEMENT DRIZZLE ===
        drizzle_enabled = False
        if hasattr(self, 'use_drizzle_var'):
            try: drizzle_enabled = self.use_drizzle_var.get()
            except tk.TclError: pass

        if drizzle_enabled:
            warning_title = self.tr('drizzle_warning_title')
            warning_text = self.tr('drizzle_warning_text')
            continue_with_drizzle = messagebox.askyesno(warning_title, warning_text, parent=self.root)
            if not continue_with_drizzle:
                self.update_progress_gui("ⓘ Démarrage annulé par l'utilisateur (Drizzle).", None)
                if hasattr(self, 'start_button'):
                    try: self.start_button.config(state=tk.NORMAL)
                    except tk.TclError: pass
                return # Arrêter la fonction ici si l'utilisateur dit non

        # === FIN AVERTISSEMENT DRIZZLE ===

        ### !!! CORRECTION INDENTATION : Ce bloc DOIT être au même niveau que le `if drizzle_enabled:` précédent !!! ###
        # --- Logique principale de démarrage (maintenant exécutée dans tous les cas sauf annulation) ---
        self.processing = True
        self.time_per_image = 0
        self.global_start_time = time.monotonic()

        default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
        self.aligned_files_var.set(default_aligned_fmt.format(count=0))

        folders_to_pass_to_backend = list(self.additional_folders_to_process)
        self.additional_folders_to_process = []
        self.update_additional_folders_display()

        self._set_parameter_widgets_state(tk.DISABLED)  # Désactiver les contrôles

        if hasattr(self, "stop_button"):
            try: self.stop_button.config(state=tk.NORMAL)
            except tk.TclError: pass
        if hasattr(self, "open_output_button"):
            try: self.open_output_button.config(state=tk.DISABLED)
            except tk.TclError: pass
        if hasattr(self, "progress_manager"):
            self.progress_manager.reset(); self.progress_manager.start_timer()
        if hasattr(self, "status_text"):
            try:
                self.status_text.config(state=tk.NORMAL)
                self.status_text.delete(1.0, tk.END)
                self.status_text.insert(tk.END, f"--- {self.tr('stacking_start', default='--- Starting Processing ---')} ---\n")
                self.status_text.config(state=tk.DISABLED)
            except tk.TclError: pass

        # MAJ et Validation Settings
        self.settings.update_from_ui(self)
        validation_messages = self.settings.validate_settings()
        if validation_messages:
             self.update_progress_gui("⚠️ Paramètres ajustés:", None);
             for msg in validation_messages: self.update_progress_gui(f"  - {msg}", None);
             self.settings.apply_to_ui(self) # Appliquer corrections à l'UI

        # Configurer QueuedStacker (inclut batch size auto)
        self.queued_stacker.stacking_mode = self.settings.stacking_mode
        self.queued_stacker.kappa = self.settings.kappa
        requested_batch_size = self.settings.batch_size; final_batch_size_for_backend = 0
        if requested_batch_size <= 0:
             self.update_progress_gui("🧠 Estimation taille lot auto...", None); sample_img_path = None
             if self.settings.input_folder and os.path.isdir(self.settings.input_folder):
                 fits_files = [f for f in os.listdir(self.settings.input_folder) if f.lower().endswith(('.fit', '.fits'))]
                 if fits_files: sample_img_path = os.path.join(self.settings.input_folder, fits_files[0])
             try: estimated_size = estimate_batch_size(sample_image_path=sample_img_path); final_batch_size_for_backend = estimated_size; self.update_progress_gui(f"✅ Taille lot auto estimée: {estimated_size}", None)
             except Exception as est_err: self.update_progress_gui(f"⚠️ Erreur estimation taille lot: {est_err}. Utilisation défaut (10).", None); final_batch_size_for_backend = 10
        else: final_batch_size_for_backend = requested_batch_size
        self.queued_stacker.batch_size = final_batch_size_for_backend

        self.queued_stacker.correct_hot_pixels = self.settings.correct_hot_pixels
        self.queued_stacker.hot_pixel_threshold = self.settings.hot_pixel_threshold
        self.queued_stacker.neighborhood_size = self.settings.neighborhood_size
        self.queued_stacker.bayer_pattern = self.settings.bayer_pattern
        self.queued_stacker.perform_cleanup = self.cleanup_temp_var.get()
        self.queued_stacker.aligner.reference_image_path = self.settings.reference_image_path or None
        self.update_progress_gui(f"ⓘ Taille de lot pour traitement : {self.queued_stacker.batch_size}", None)

        # Démarrer le backend (passe maintenant use_drizzle depuis la variable UI)
        processing_started = self.queued_stacker.start_processing(
            input_folder, output_folder, self.settings.reference_image_path,
            initial_additional_folders=folders_to_pass_to_backend,
            # Pondération
            use_weighting=self.settings.use_quality_weighting, weight_snr=self.settings.weight_by_snr,
            weight_stars=self.settings.weight_by_stars, snr_exp=self.settings.snr_exponent,
            stars_exp=self.settings.stars_exponent, min_w=self.settings.min_weight,
            # Drizzle
            use_drizzle=self.use_drizzle_var.get(), # Utilise la variable UI
            drizzle_scale=float(self.drizzle_scale_var.get()),
            drizzle_wht_threshold=self.drizzle_wht_threshold_var.get()
        )

        # Gérer résultat démarrage backend
        if processing_started:
            if hasattr(self, 'stop_button'):
                try: self.stop_button.config(state=tk.NORMAL)
                except tk.TclError: pass
            # Démarrer le thread de suivi de la progression GUI
            self.thread = threading.Thread(target=self._track_processing_progress, daemon=True, name="GUI_ProgressTracker")
            self.thread.start()
        else: # Échec démarrage backend (rare, normalement déjà en cours)
            self.update_progress_gui("ⓘ Demande démarrage ignorée (traitement déjà en cours?).", None)
            if hasattr(self, 'stop_button'):
                try: self.stop_button.config(state=tk.NORMAL)
                except tk.TclError: pass
            # S'assurer que l'état processing est bien True si le backend tournait déjà
            self.processing = True
        ### !!! FIN BLOC CORRIGÉ !!! ###


# --- Main Execution ---
if __name__ == "__main__":
    try:
        _dummy_root = tk.Tk()
        _dummy_root.withdraw()
        style = ttk.Style()
        available_themes = style.theme_names()
        theme_to_use = 'default'
        preferred_themes = ['clam', 'alt', 'vista', 'xpnative']
        for t in preferred_themes:
            if t in available_themes:
                theme_to_use = t
                break
        print(f"Using theme: {theme_to_use}")
        style.theme_use(theme_to_use)
        try:
            style.configure('Accent.TButton', font=('Segoe UI', 9, 'bold'), foreground='white', background='#0078D7')
        except tk.TclError:
            print("Warning: Could not configure Accent.TButton style.")
        try:
            style.configure('Toolbutton.TButton', padding=1, font=('Segoe UI', 8))
        except tk.TclError:
            print("Warning: Could not configure Toolbutton.TButton style.")
        if '_dummy_root' in locals() and _dummy_root.winfo_exists():
            _dummy_root.destroy()
    except tk.TclError as theme_err:
        print(f"Error initializing Tk themes: {theme_err}. Using Tk default.")
        if '_dummy_root' in locals() and _dummy_root.winfo_exists():
            _dummy_root.destroy()

    gui = SeestarStackerGUI()
    gui.root.mainloop()
    # --- END OF FILE seestar/gui/main_window.py (Part 3/3) ---
