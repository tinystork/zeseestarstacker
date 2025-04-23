# --- START OF FILE seestar/gui/main_window.py ---
"""
Module principal pour l'interface graphique de GSeestar.
Intègre la prévisualisation avancée et le traitement en file d'attente via QueueManager.
(Version Révisée: Interaction avec QueueManager modifié, gestion état boutons)
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

# Seestar imports
from seestar.core.image_processing import load_and_validate_fits, debayer_image
from seestar.localization import Localization
from seestar.queuep.queue_manager import SeestarQueuedStacker # Updated import
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
        # --- *** NEW: Quality Weighting Variables *** ---
        self.use_weighting_var = tk.BooleanVar(value=False)
        self.weight_snr_var = tk.BooleanVar(value=True)
        self.weight_stars_var = tk.BooleanVar(value=True)
        self.snr_exponent_var = tk.DoubleVar(value=1.0)
        self.stars_exponent_var = tk.DoubleVar(value=0.5)
        self.min_weight_var = tk.DoubleVar(value=0.1)
        # --- End New ---
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
        #self.current_stack_count_preview = 0 # Add variable to store stack count for preview
        
        # Apply loaded settings to the UI widgets
        self.settings.apply_to_ui(self)
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
            # Handle error gracefully, maybe disable progress updates

        # Preview Manager
        if hasattr(self, 'preview_canvas'):
            self.preview_manager = PreviewManager(self.preview_canvas)
        else:
            print("Error: Preview canvas not found for PreviewManager initialization.")
            # Handle error, maybe disable preview

        # Histogram Widget Callback (if widget exists)
        if hasattr(self, 'histogram_widget') and self.histogram_widget:
            self.histogram_widget.range_change_callback = self.update_stretch_from_histogram
        else:
            print("Error: HistogramWidget reference not found after create_layout.")

        # File Handler (should already be created in __init__)
        if not hasattr(self, 'file_handler'):
             print("Error: FileHandlingManager not initialized.")
             # Potentially create it here as a fallback if absolutely necessary
             # self.file_handler = FileHandlingManager(self)

        # Show initial state in preview area
        self.show_initial_preview()

    def _update_weighting_options_state(self):
        """Active ou désactive les options de pondération détaillées."""
        # Détermine l'état (NORMAL ou DISABLED) basé sur la case principale
        state = tk.NORMAL if self.use_weighting_var.get() else tk.DISABLED

        # Liste des widgets enfants à activer/désactiver
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

        # Applique l'état à chaque widget enfant
        for widget in widgets_to_toggle:
            if widget and hasattr(widget, 'winfo_exists') and widget.winfo_exists():
                try:
                    widget.config(state=state)
                except tk.TclError:
                    # Ignore l'erreur si le widget est détruit pendant l'opération
                    pass

    def show_initial_preview(self):
        """ Affiche un état initial dans la zone d'aperçu. """
        if hasattr(self, 'preview_manager') and self.preview_manager:
            self.preview_manager.clear_preview(self.tr("Select input/output folders."))
        if hasattr(self, 'histogram_widget') and self.histogram_widget:
            self.histogram_widget.plot_histogram(None) # Clear histogram

    def tr(self, key, default=None):
        """ Raccourci pour la localisation. """
        return self.localization.get(key, default=default)

    def create_layout(self):
        """Crée la disposition des widgets."""
        # --- Structure principale ---
        main_frame = ttk.Frame(self.root); main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL); paned_window.pack(fill=tk.BOTH, expand=True)
        # Left panel for controls (fixed width initially)
        left_frame = ttk.Frame(paned_window, width=450, height=700); left_frame.pack_propagate(False); paned_window.add(left_frame, weight=1)
        # Right panel for preview (expands)
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
        # Removed (0=auto) label as 0 is no longer auto

        # Hot Pixels Group
        self.hp_frame = ttk.LabelFrame(tab_stacking, text=self.tr("hot_pixels_correction"))
        self.hp_frame.pack(fill=tk.X, pady=5, padx=5)

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

        # Time Info Row
        time_frame = ttk.Frame(self.progress_frame)
        time_frame.pack(fill=tk.X, padx=5, pady=2)
        self.remaining_time_label = ttk.Label(time_frame, text=self.tr("estimated_time")); self.remaining_time_label.pack(side=tk.LEFT)
        self.remaining_time_value = ttk.Label(time_frame, textvariable=self.remaining_time_var, font=tkFont.Font(weight='bold'), width=9, anchor='w'); self.remaining_time_value.pack(side=tk.LEFT, padx=(2, 10))
        self.elapsed_time_label = ttk.Label(time_frame, text=self.tr("elapsed_time")); self.elapsed_time_label.pack(side=tk.LEFT)
        self.elapsed_time_value = ttk.Label(time_frame, textvariable=self.elapsed_time_var, font=tkFont.Font(weight='bold'), width=9, anchor='w'); self.elapsed_time_value.pack(side=tk.LEFT, padx=2)

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


        # Status Text Area
        status_text_frame = ttk.Frame(self.progress_frame)
        status_text_font = tkFont.Font(family="Arial", size=8) # Smaller font
        status_text_frame.pack(fill=tk.X, expand=False, padx=5, pady=(2, 5))
        self.status_text = tk.Text(status_text_frame, height=5, wrap=tk.WORD, bd=0, font=status_text_font, relief=tk.FLAT, state=tk.DISABLED)
        self.status_scrollbar = ttk.Scrollbar(status_text_frame, orient="vertical", command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=self.status_scrollbar.set)
        self.status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.pack(side=tk.LEFT, fill=tk.X, expand=True)


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
        # Add Folder button - starts disabled, enabled during processing
        self.add_files_button = ttk.Button(control_frame, text=self.tr("add_folder_button"), command=self.add_folder, state=tk.DISABLED)
        self.add_files_button.pack(side=tk.RIGHT, padx=5, pady=5, ipady=2)


        # --- Panneau Droit (Aperçu et Histogramme) ---
        # Preview Frame
        self.preview_frame = ttk.LabelFrame(right_frame, text=self.tr("preview"))
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=(5,5), padx=5) # Pad top and bottom
        # Canvas for image display
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="#1E1E1E", highlightthickness=0) # Dark background
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        # Histogram Frame (at the bottom of right panel)
        self.histogram_frame = ttk.LabelFrame(right_frame, text=self.tr("histogram"))
        # Calculate height based on desired Matplotlib figsize/dpi
        hist_fig_height_inches = 2.2
        hist_fig_dpi = 80 # Adjust if needed
        hist_height_pixels = int(hist_fig_height_inches * hist_fig_dpi * 1.1) # Add some padding
        self.histogram_frame.pack(fill=tk.X, expand=False, pady=(0,5), padx=5, side=tk.BOTTOM)
        self.histogram_frame.pack_propagate(False) # Prevent resizing based on content
        self.histogram_frame.config(height=hist_height_pixels)

        # Create and pack HistogramWidget inside its frame
        self.histogram_widget = HistogramWidget(self.histogram_frame, range_change_callback=self.update_stretch_from_histogram)
        self.histogram_widget.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=(0,2), pady=(0,2)) # Leave space for button

        # Add Reset Zoom button next to histogram
        self.hist_reset_btn = ttk.Button(self.histogram_frame, text="R", command=self.histogram_widget.reset_zoom, width=2)
        self.hist_reset_btn.pack(side=tk.RIGHT, anchor=tk.NE, padx=(0,2), pady=2)

        # Store references AFTER all widgets are created
        self._store_widget_references()

    def _create_slider_spinbox_group(self, parent, label_key, min_val, max_val, step, tk_var, callback=None):
        """Helper to create a consistent Slider + SpinBox group with debouncing."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=(1,1))

        # Label
        label_widget = ttk.Label(frame, text=self.tr(label_key, default=label_key), width=10) # Use tr() for label
        label_widget.pack(side=tk.LEFT)

        # Determine number of decimals for Spinbox format
        decimals = 0
        if step > 0:
            try:
                 log_step = math.log10(step)
                 if log_step < 0: decimals = abs(int(log_step))
            except ValueError: log_step = -3 # Default if step is 0 or invalid
            if log_step < 0: decimals = abs(int(log_step))
        format_str = f"%.{decimals}f" # e.g., "%.3f" for step=0.001

        # Spinbox (on the right)
        spinbox = ttk.Spinbox(
            frame, from_=min_val, to=max_val, increment=step,
            textvariable=tk_var, width=7, justify=tk.RIGHT,
            command=self._debounce_refresh_preview, # Update on change
            format=format_str # Apply format
        )
        spinbox.pack(side=tk.RIGHT, padx=(5,0))

        # Slider (in the middle, expands)
        # Callback for immediate UI feedback (e.g., histogram lines) if needed
        def on_scale_change(value_str):
            # Convert value from string provided by Scale
            try: value = float(value_str)
            except ValueError: return

            # Call optional immediate callback (e.g., for histogram lines)
            if callback:
                try: callback(value)
                except Exception as cb_err: print(f"Error in slider immediate callback: {cb_err}")

            # Trigger the debounced preview refresh
            self._debounce_refresh_preview()

        slider = ttk.Scale(
            frame, from_=min_val, to=max_val,
            variable=tk_var, # Link to the same variable as Spinbox
            orient=tk.HORIZONTAL,
            command=on_scale_change # Call function on change
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Store references for potential external access (e.g., translation)
        ctrls = {
            'frame': frame,
            'label': label_widget,
            'slider': slider,
            'spinbox': spinbox
        }
        return ctrls

    def _store_widget_references(self):
        """Stores references to widgets that need language updates."""
        # Find the notebook widget to reference tabs correctly
        notebook_widget = None
        try:
            # Assuming main_frame -> paned_window -> left_frame -> control_notebook structure
            main_frame = self.root.winfo_children()[0]
            paned_window = main_frame.winfo_children()[0]
            left_frame_widget = self.root.nametowidget(paned_window.panes()[0])
            for child in left_frame_widget.winfo_children():
                if isinstance(child, ttk.Notebook):
                    notebook_widget = child
                    break
        except (IndexError, tk.TclError, AttributeError) as e:
            print(f"Warning: Could not reliably find Notebook widget for translation: {e}")

        # Dictionary mapping translation keys to widgets or (notebook, index) tuples
        self.widgets_to_translate = {
            # Tabs
            "tab_stacking": (notebook_widget, 0) if notebook_widget else None,
            "tab_preview": (notebook_widget, 1) if notebook_widget else None,
            # LabelFrames
            "Folders": getattr(self, 'folders_frame', None),
            "options": getattr(self, 'options_frame', None),
            "hot_pixels_correction": getattr(self, 'hp_frame', None),
            "post_proc_opts_frame_label": getattr(self, 'post_proc_opts_frame', None),
            "white_balance": getattr(self, 'wb_frame', None),
            "stretch_options": getattr(self, 'stretch_frame_controls', None),
            "image_adjustments": getattr(self, 'bcs_frame', None),
            "progress": getattr(self, 'progress_frame', None),
            "preview": getattr(self, 'preview_frame', None),
            "histogram": getattr(self, 'histogram_frame', None),
            # Labels
            "input_folder": getattr(self, 'input_label', None),
            "output_folder": getattr(self, 'output_label', None),
            "reference_image": getattr(self, 'reference_label', None),
            "stacking_method": getattr(self, 'stacking_method_label', None),
            "kappa_value": getattr(self, 'kappa_label', None),
            "batch_size": getattr(self, 'batch_size_label', None),
            # "batch_size_auto": getattr(self, 'batch_size_auto_label', None), # Removed
            "hot_pixel_threshold": getattr(self, 'hot_pixel_threshold_label', None),
            "neighborhood_size": getattr(self, 'neighborhood_size_label', None),
            "wb_r": getattr(self, 'wb_r_ctrls', {}).get('label'),
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
            "Remaining:": getattr(self, 'remaining_static_label', None),
            "Additional:": getattr(self, 'additional_static_label', None),
            # "aligned_files_label": getattr(self, 'aligned_files_static_label', None), # Removed static label
            # Buttons
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
            # Checkbuttons
            "perform_hot_pixels_correction": getattr(self, 'hot_pixels_check', None),
            "cleanup_temp_check_label": getattr(self, 'cleanup_temp_check', None),
             # --- *** NEW: Widgets Pondération *** ---
            "quality_weighting_frame": getattr(self, 'weighting_frame', None), # Clé pour LabelFrame title
            "enable_weighting_check": getattr(self, 'use_weighting_check', None), # Clé pour Checkbutton text
            "weighting_metrics_label": getattr(self, 'weight_metrics_label', None),# Clé pour Label text
            "weight_snr_check": getattr(self, 'weight_snr_check', None),          # Clé pour Checkbutton text
            "weight_stars_check": getattr(self, 'weight_stars_check', None),      # Clé pour Checkbutton text
            "snr_exponent_label": getattr(self, 'snr_exp_label', None),           # Clé pour Label text
            "stars_exponent_label": getattr(self, 'stars_exp_label', None),       # Clé pour Label text
            "min_weight_label": getattr(self, 'min_w_label', None),               # Clé pour Label te
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
            # Use default text from English if key not found in current lang or 'en'
            default_text = self.localization.translations['en'].get(key, key.replace("_", " ").title())
            translation = self.tr(key, default=default_text) # tr() handles fallback internally now

            try:
                if widget_info is None:
                    # print(f"Debug: Skipping translation for '{key}' (widget is None)")
                    continue

                # Handle notebook tabs
                if isinstance(widget_info, tuple):
                    notebook, index = widget_info
                    if notebook and notebook.winfo_exists() and index < notebook.index("end"):
                        notebook.tab(index, text=f' {translation} ') # Add padding
                # Handle regular widgets
                elif hasattr(widget_info, 'winfo_exists') and widget_info.winfo_exists():
                    widget = widget_info
                    if isinstance(widget, (ttk.Label, ttk.Button, ttk.Checkbutton)):
                        widget.config(text=translation)
                    elif isinstance(widget, ttk.LabelFrame):
                        widget.config(text=translation)
                    # Add other widget types if needed (e.g., Menus)
                # else:
                    # print(f"Debug: Skipping translation for '{key}' (widget destroyed or invalid)")

            except tk.TclError as e:
                # Ignore errors likely due to widget destruction during update
                # print(f"Debug: TclError updating widget '{key}': {e}")
                pass
            except Exception as e:
                print(f"Debug: Unexpected error updating widget '{key}' ({type(widget_info)}): {e}")

        # Update dynamic text variables that depend on language
        if not self.processing:
            self.remaining_files_var.set(self.tr("no_files_waiting"))
            self.update_additional_folders_display() # Update count display text
            default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
            self.aligned_files_var.set(default_aligned_fmt.format(count="--"))
            self.remaining_time_var.set("--:--:--")
        else:
            # Ensure static labels in progress area are also translated if processing
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

        # Update initial preview message if no image is loaded
        if self.current_preview_data is None and hasattr(self, 'preview_manager'):
            self.preview_manager.clear_preview(self.tr('Select input/output folders.'))


    def _debounce_refresh_preview(self, *args):
        """ Debounce preview refresh calls from sliders/spinboxes. """
        if self.debounce_timer_id:
            try: self.root.after_cancel(self.debounce_timer_id)
            except tk.TclError: pass # Ignore if timer ID is invalid (e.g., window closing)
        try:
            self.debounce_timer_id = self.root.after(150, self.refresh_preview) # Delay refresh
        except tk.TclError: pass # Ignore if root window is destroyed


    def update_histogram_lines_from_sliders(self, *args):
        """ Callback from BP/WP sliders to update histogram lines immediately. """
        if hasattr(self, 'histogram_widget') and self.histogram_widget:
            try:
                bp = self.preview_black_point.get()
                wp = self.preview_white_point.get()
            except tk.TclError: return # Ignore if vars are invalid
            self.histogram_widget.set_range(bp, wp)


    def update_stretch_from_histogram(self, black_point, white_point):
        """ Callback from HistogramWidget when lines are dragged. """
        # Update UI variables
        try:
            self.preview_black_point.set(round(black_point, 4))
            self.preview_white_point.set(round(white_point, 4))
        except tk.TclError: return # Ignore if vars invalid

        # Update sliders/spinboxes to reflect change (disable signals temporarily)
        try:
            if hasattr(self, 'stretch_bp_ctrls'):
                 self.stretch_bp_ctrls['slider'].set(black_point)
                 # Spinbox updates via tk_var link
            if hasattr(self, 'stretch_wp_ctrls'):
                 self.stretch_wp_ctrls['slider'].set(white_point)
        except tk.TclError: pass # Ignore if sliders destroyed

        # Trigger debounced preview refresh
        self._debounce_refresh_preview()


    def refresh_preview(self):
        """ Redraws the preview canvas with current settings. """
        # Cancel any pending debounce timer
        if self.debounce_timer_id:
            try: self.root.after_cancel(self.debounce_timer_id)
            except tk.TclError: pass
            self.debounce_timer_id = None

        # Check if preview is possible
        if (self.current_preview_data is None or
           not hasattr(self, 'preview_manager') or self.preview_manager is None or
           not hasattr(self, 'histogram_widget') or self.histogram_widget is None):
            # (Fallback logic remains the same)
            if (not self.processing and self.input_path.get() and
               os.path.isdir(self.input_path.get())):
                self._try_show_first_input_image()
            else:
                if hasattr(self, 'preview_manager') and self.preview_manager:
                    self.preview_manager.clear_preview(self.tr('Select input/output folders.'))
                if hasattr(self, 'histogram_widget') and self.histogram_widget:
                    self.histogram_widget.plot_histogram(None)
            return

        # Get current preview parameters from UI variables
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
            print("Error getting preview parameters from UI.")
            return

        # --- MODIFICATION START: Pass stack count to PreviewManager ---
        # Call PreviewManager WITHOUT stack_count
        processed_pil_image, data_for_histogram = self.preview_manager.update_preview(
            self.current_preview_data,
            preview_params # NO stack_count here
        )

        # Update the preview using PreviewManager
        # PreviewManager applies WB, Stretch, Gamma, BCS
        processed_pil_image, data_for_histogram = self.preview_manager.update_preview(
            self.current_preview_data, preview_params
        )

        # Update the histogram with the data *after* WB but *before* Stretch/Gamma/BCS
        self.histogram_widget.update_histogram(data_for_histogram)
        # Ensure histogram lines match current slider/spinbox values
        self.histogram_widget.set_range(preview_params["black_point"], preview_params["white_point"])


    def update_preview_from_stacker(self, stack_data, stack_header, stack_name, img_count, total_imgs, current_batch, total_batches):
        """ Callback function for the QueuedStacker to update the GUI preview. """
        if stack_data is None:
            print("GUI received None stack_data for preview.")
            return

        # Update the internal data reference for subsequent refreshes
        self.current_preview_data = stack_data.copy() # Make a copy
        self.current_stack_header = stack_header.copy() if stack_header else None
        #self.current_stack_count_preview = stack_count # <<< STORE THE STACK COUNT

        # Store the extra info for potential use elsewhere if needed
        self.preview_img_count = img_count
        self.preview_total_imgs = total_imgs
        self.preview_current_batch = current_batch
        self.preview_total_batches = total_batches

        # Schedule the preview refresh to happen in the main GUI thread
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
            self.root.after_idle(
                lambda d=self.current_preview_data, p=preview_params, sc=img_count, ti=total_imgs, cb=current_batch, tb=total_batches:
                    # Check if preview_manager exists before calling its method
                    self.preview_manager.update_preview(d, p, stack_count=sc, total_images=ti, current_batch=cb, total_batches=tb)
                    if hasattr(self, 'preview_manager') and self.preview_manager is not None else None
            )
        except tk.TclError:
            pass # Ignore if window is closing

        # Update image info text area if header is available
        if self.current_stack_header:
            try:
                # Schedule info update separately
                self.root.after_idle(lambda h=self.current_stack_header: self.update_image_info(h))
            except tk.TclError:
                pass


    def update_image_info(self, header):
        """ Met à jour la zone de texte d'informations image. """
        if not header or not hasattr(self, 'preview_manager'): return

        info_lines = []
        # Dictionnaire des clés et labels (ajuster si besoin)
        keys_labels = {
            'OBJECT': 'Object', 'DATE-OBS': 'Date', 'EXPTIME': 'Exp (s)',
            'GAIN': 'Gain', 'OFFSET': 'Offset', 'CCD-TEMP': 'Temp (°C)',
            'NIMAGES': 'Images', 'STACKTYP': 'Method', 'FILTER': 'Filter',
            'BAYERPAT': 'Bayer', 'TOTEXP': 'Total Exp (s)', 'ALIGNED': 'Aligned',
            'FAILALIGN': 'Failed Align', 'FAILSTACK': 'Failed Stack', 'SKIPPED': 'Skipped',
            # --- *** NEW: Weighting Info Keys *** ---
            'WGHT_ON': 'Weighting', # Clé pour traduction
            'WGHT_MET': 'W. Metrics' # Clé pour traduction
            # --- End New ---
        }

        for key, label_key in keys_labels.items():
            label = self.tr(label_key, default=label_key) # Translate the label
            value = header.get(key)
            if value is not None and str(value).strip() != '':
                s_value = str(value)
                # Special formatting for some keys
                if key == 'DATE-OBS':
                     s_value = s_value.split('T')[0] # Show only date part
                elif key in ['EXPTIME', 'CCD-TEMP', 'TOTEXP'] and isinstance(value, (int, float)):
                     try: s_value = f"{float(value):.1f}"
                     except ValueError: pass
                elif key == 'KAPPA' and isinstance(value, (int, float)):
                     try: s_value = f"{float(value):.2f}"
                     except ValueError: pass
                # --- *** NEW: Format Weighting Info *** ---
                elif key == 'WGHT_ON':
                    # Traduire True/False ?
                    s_value = self.tr('weighting_enabled') if value else self.tr('weighting_disabled')
                # --- End New ---


                info_lines.append(f"{label}: {s_value}")

        info_text = "\n".join(info_lines) if info_lines else self.tr('No image info available')

        # Update the text area via PreviewManager
        if hasattr(self.preview_manager, 'update_info_text'):
            self.preview_manager.update_info_text(info_text)
        else:
             # Fallback if method doesn't exist on preview manager
             pass

    def _try_show_first_input_image(self):
        """ Tente de charger et d'afficher la première image FITS du dossier d'entrée. """
        input_folder = self.input_path.get()
        # Ensure managers needed for display exist
        if not hasattr(self, 'preview_manager') or not hasattr(self, 'histogram_widget'):
            print("Warning: Cannot show first image, preview/histogram managers not ready.")
            return

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

            # Load image data and header
            img_data = load_and_validate_fits(first_image_path) # Expects float32 0-1
            if img_data is None:
                 raise ValueError(f"Échec chargement/validation {files[0]}")
            header = fits.getheader(first_image_path) # Get header separately

            # Prepare for preview (debayer if necessary)
            img_for_preview = img_data
            if img_data.ndim == 2:
                bayer_pattern = header.get("BAYERPAT", self.settings.bayer_pattern) # Use setting default
                valid_bayer_patterns = ["GRBG", "RGGB", "GBRG", "BGGR"]
                if isinstance(bayer_pattern, str) and bayer_pattern.upper() in valid_bayer_patterns:
                    try:
                        img_for_preview = debayer_image(img_data, bayer_pattern.upper())
                    except ValueError as debayer_err:
                        self.update_progress_gui(f"⚠️ {self.tr('Error during debayering')}: {debayer_err}. Affichage N&B.", None)
                        # Keep grayscale if debayer fails

            # Update internal data reference and trigger refresh
            self.current_preview_data = img_for_preview.copy() # Store potentially debayered data
            self.current_stack_header = header.copy() if header else None
            self.refresh_preview() # This will call preview_manager.update_preview

            # Update info text area
            if self.current_stack_header:
                self.update_image_info(self.current_stack_header)

        except FileNotFoundError:
            if hasattr(self, 'preview_manager'): self.preview_manager.clear_preview(self.tr("Input folder not found"))
            if hasattr(self, 'histogram_widget'): self.histogram_widget.plot_histogram(None)
        except ValueError as ve: # Catch specific load/validation errors
            self.update_progress_gui(f"⚠️ {self.tr('Error loading preview image')}: {ve}", None)
            if hasattr(self, 'preview_manager'): self.preview_manager.clear_preview(self.tr("Error loading preview (invalid format?)"))
            if hasattr(self, 'histogram_widget'): self.histogram_widget.plot_histogram(None)
        except Exception as e:
            self.update_progress_gui(f"⚠️ {self.tr('Error loading preview image')}: {e}", None)
            traceback.print_exc(limit=2)
            if hasattr(self, 'preview_manager'): self.preview_manager.clear_preview(self.tr("Error loading preview"))
            if hasattr(self, 'histogram_widget'): self.histogram_widget.plot_histogram(None)


    # --- Auto WB / Stretch / BCS Actions ---
    def apply_auto_white_balance(self):
        """ Applique la balance des blancs automatique à l'aperçu. """
        if not _tools_available:
            messagebox.showerror(self.tr("error"), "Stretch/Color tools not available."); return
        # Use the data *currently held* for preview (could be initial image or stack)
        if self.current_preview_data is None or self.current_preview_data.ndim != 3:
            messagebox.showwarning(self.tr("warning"), self.tr("Auto WB requires a color image preview.")); return

        try:
            r_gain, g_gain, b_gain = calculate_auto_wb(self.current_preview_data) # Calculate from raw preview data
            # Set UI variables
            self.preview_r_gain.set(round(r_gain, 3))
            self.preview_g_gain.set(round(g_gain, 3))
            self.preview_b_gain.set(round(b_gain, 3))
            self.update_progress_gui(f"Auto WB appliqué (Aperçu): R={r_gain:.2f} G={g_gain:.2f} B={b_gain:.2f}", None)
            # Trigger refresh to apply new gains
            self.refresh_preview()
        except Exception as e:
            messagebox.showerror(self.tr("error"), f"{self.tr('Error during Auto WB')}: {e}")
            traceback.print_exc(limit=2)

    def reset_white_balance(self):
        """ Réinitialise les gains de balance des blancs à 1.0. """
        self.preview_r_gain.set(1.0)
        self.preview_g_gain.set(1.0)
        self.preview_b_gain.set(1.0)
        self.refresh_preview()

    def reset_brightness_contrast_saturation(self):
        """ Réinitialise les ajustements B/C/S à 1.0 (S à 1.0). """
        self.preview_brightness.set(1.0)
        self.preview_contrast.set(1.0)
        self.preview_saturation.set(1.0)
        self.refresh_preview()

    def apply_auto_stretch(self):
        """ Calcule et applique l'étirement automatique à l'aperçu. """
        if not _tools_available:
             messagebox.showerror(self.tr("error"), "Stretch tools not available."); return

        # Determine the data to analyze: Use the preview data *after* current WB settings are applied
        data_to_analyze = None
        if hasattr(self, 'preview_manager') and self.preview_manager.image_data_wb is not None:
            # If PreviewManager has cached the white-balanced data, use it
            data_to_analyze = self.preview_manager.image_data_wb
        elif self.current_preview_data is not None:
            # Otherwise, apply current WB settings to the raw preview data
            print("Warning AutoStretch: Using current WB settings for analysis.")
            if self.current_preview_data.ndim == 3:
                try:
                    r=self.preview_r_gain.get(); g=self.preview_g_gain.get(); b=self.preview_b_gain.get()
                    data_to_analyze = ColorCorrection.white_balance(self.current_preview_data, r, g, b)
                except Exception: data_to_analyze = self.current_preview_data # Fallback
            else: data_to_analyze = self.current_preview_data # Grayscale
        else:
            messagebox.showwarning(self.tr("warning"), self.tr("Auto Stretch requires an image preview.")); return

        try:
            bp, wp = calculate_auto_stretch(data_to_analyze) # Calculate BP/WP
            # Set UI variables
            self.preview_black_point.set(round(bp, 4))
            self.preview_white_point.set(round(wp, 4))
            # Update histogram lines directly
            if hasattr(self, 'histogram_widget'): self.histogram_widget.set_range(bp, wp)
            self.update_progress_gui(f"Auto Stretch appliqué (Aperçu): BP={bp:.3f} WP={wp:.3f}", None)
            # Trigger refresh to apply new stretch points
            self.refresh_preview()
        except Exception as e:
            messagebox.showerror(self.tr("error"), f"{self.tr('Error during Auto Stretch')}: {e}")
            traceback.print_exc(limit=2)

    def reset_stretch(self):
        """ Réinitialise les paramètres d'étirement aux valeurs par défaut. """
        default_method = "Asinh"
        default_bp = 0.01
        default_wp = 0.99
        default_gamma = 1.0
        self.preview_stretch_method.set(default_method)
        self.preview_black_point.set(default_bp)
        self.preview_white_point.set(default_wp)
        self.preview_gamma.set(default_gamma)
        # Reset histogram lines and zoom
        if hasattr(self, 'histogram_widget'):
             self.histogram_widget.set_range(default_bp, default_wp)
             self.histogram_widget.reset_zoom()
        self.refresh_preview()

    def add_folder(self):
        """ Handles adding a folder DURING processing via FileHandlingManager. """
        # Let FileHandlingManager handle the logic and call queue_manager.add_folder
        if hasattr(self, 'file_handler'):
            self.file_handler.add_folder()
        else:
            messagebox.showerror(self.tr('error'), "File handler not initialized.")

# --- Processing Control ---
    def start_processing(self):
        """Démarre le traitement, en passant les paramètres de pondération."""
        input_folder = self.input_path.get()
        output_folder = self.output_path.get()

        # --- Input Validation ---
        if not input_folder or not output_folder:
            messagebox.showerror(self.tr("error"), self.tr("select_folders"))
            return
        if not os.path.isdir(input_folder):
            messagebox.showerror(self.tr("error"), f"{self.tr('input_folder_invalid')}:\n{input_folder}")
            return
        if not os.path.isdir(output_folder):
            try:
                os.makedirs(output_folder, exist_ok=True)
            except Exception as e:
                messagebox.showerror(self.tr("error"), f"{self.tr('output_folder_invalid')}:\n{output_folder}\n{e}")
                return
            self.update_progress_gui(f"{self.tr('Output folder created')}: {output_folder}", None)
        try:
            if not any(f.lower().endswith((".fit", ".fits")) for f in os.listdir(input_folder)):
                if not messagebox.askyesno(self.tr("warning"), self.tr("no_fits_found")):
                    return
        except Exception as e:
            messagebox.showerror(self.tr("error"), f"{self.tr('Error reading input folder')}:\n{e}")
            return

        # --- Prepare for Processing ---
        self.processing = True
        self.time_per_image = 0
        self.global_start_time = time.monotonic()
        default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
        self.aligned_files_var.set(default_aligned_fmt.format(count=0))
        self.update_remaining_files()
        self._set_parameter_widgets_state(tk.DISABLED)  # Désactive les contrôles
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.add_files_button.config(state=tk.NORMAL)
        self.progress_manager.reset()
        self.progress_manager.start_timer()
        if hasattr(self, 'status_text'):
            try:
                self.status_text.config(state=tk.NORMAL)
                self.status_text.delete(1.0, tk.END)
                self.status_text.insert(tk.END, "--- Début du Traitement ---\n")
                self.status_text.config(state=tk.DISABLED)
            except tk.TclError:
                pass

        # Update settings from UI and validate
        self.settings.update_from_ui(self)
        validation_messages = self.settings.validate_settings()
        if validation_messages:
            self.update_progress_gui("⚠️ Paramètres ajustés:", None)
            for msg in validation_messages:
                self.update_progress_gui(f"  - {msg}", None)
            self.settings.apply_to_ui(self)  # Applique les valeurs corrigées

        # --- Configure the QueuedStacker instance ---
        self.queued_stacker.stacking_mode = self.settings.stacking_mode
        self.queued_stacker.kappa = self.settings.kappa
        self.queued_stacker.batch_size = self.settings.batch_size  # Utilise param validé
        self.queued_stacker.correct_hot_pixels = self.settings.correct_hot_pixels
        self.queued_stacker.hot_pixel_threshold = self.settings.hot_pixel_threshold
        self.queued_stacker.neighborhood_size = self.settings.neighborhood_size
        self.queued_stacker.bayer_pattern = self.settings.bayer_pattern
        self.queued_stacker.perform_cleanup = self.cleanup_temp_var.get()
        self.queued_stacker.aligner.reference_image_path = self.settings.reference_image_path or None

        self.update_progress_gui(self.tr("stacking_start"), 0)

        # --- Start the QueuedStacker with Weighting Params ---
        # *** MODIFIED CALL with weighting parameters ***
        print(f"DEBUG: Passing weighting params: use={self.settings.use_quality_weighting}, snr={self.settings.weight_by_snr}, stars={self.settings.weight_by_stars}, snr_exp={self.settings.snr_exponent},")
        processing_started = self.queued_stacker.start_processing(
            self.settings.input_folder,
            self.settings.output_folder,
            self.settings.reference_image_path,
            # Pass weighting settings from the settings manager
            use_weighting=self.settings.use_quality_weighting,
            weight_snr=self.settings.weight_by_snr,
            weight_stars=self.settings.weight_by_stars,
            snr_exp=self.settings.snr_exponent,
            stars_exp=self.settings.stars_exponent,
            min_w=self.settings.min_weight
        )

        if processing_started:
            # Start the GUI progress tracking thread
            self.thread = threading.Thread(target=self._track_processing_progress, daemon=True, name="GUI_ProgressTracker")
            self.thread.start()
        else:
            # Handle case where backend thread failed to start
            self.update_progress_gui("❌ Échec démarrage du thread de traitement.", None)
            messagebox.showerror(self.tr("error"), self.tr("Failed to start processing."))
            self._processing_finished()  # Reset UI state

    def _track_processing_progress(self):
        """Monitors the QueuedStacker worker thread and updates GUI stats."""
        while self.processing and hasattr(self, "queued_stacker"):
            try:
                # Check if the backend worker is still running
                if not self.queued_stacker.is_running():
                    # Worker finished or stopped, schedule final UI update
                    self.root.after(0, self._processing_finished)
                    break

                # --- Get Stats from QueuedStacker ---
                q_stacker = self.queued_stacker
                processed = q_stacker.processed_files_count  # Files dequeued
                aligned = q_stacker.aligned_files_count
                failed_align = q_stacker.failed_align_count
                failed_stack = q_stacker.failed_stack_count  # New stat
                skipped = q_stacker.skipped_files_count
                total_queued = q_stacker.files_in_queue  # Dynamic total

                # --- Calculate ETA ---
                if self.global_start_time and processed > 0:
                    elapsed = time.monotonic() - self.global_start_time
                    self.time_per_image = elapsed / processed  # Avg time per dequeued file
                    try:
                        remaining_estimated = max(0, total_queued - processed)
                        if self.time_per_image > 1e-6 and remaining_estimated > 0:
                            eta_seconds = remaining_estimated * self.time_per_image
                            h, rem = divmod(int(eta_seconds), 3600)
                            m, s = divmod(rem, 60)
                            self.remaining_time_var.set(f"{h:02}:{m:02}:{s:02}")
                        elif remaining_estimated == 0 and total_queued > 0:
                            self.remaining_time_var.set("00:00:00")  # Finished
                        else:
                            self.remaining_time_var.set("--:--:--")  # Not enough data or finished
                    except tk.TclError:
                        break  # Stop if UI destroyed
                    except Exception as eta_err:
                        print(f"Warning: Error calculating ETA: {eta_err}")
                        try:
                            self.remaining_time_var.set("Error")
                        except tk.TclError:
                            break
                else:
                    try: self.remaining_time_var.set("--:--:--") # Not started or no processed files yet
                    except tk.TclError: break

                # --- Update Aligned Files Count Display ---
                default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
                try: self.aligned_files_var.set(default_aligned_fmt.format(count=aligned))
                except tk.TclError: break

                # --- Update Remaining Files Display ---
                self.update_remaining_files() # Uses queued_stacker stats

                # --- Sleep before next check ---
                time.sleep(0.5) # Check stats twice per second

            except Exception as e:
                print(f"Error in GUI progress tracker thread: {e}")
                traceback.print_exc(limit=2)
                try:
                    # Attempt to trigger finish sequence on error
                    self.root.after(0, self._processing_finished)
                except tk.TclError: pass # Ignore if root destroyed
                break # Exit tracker thread on error


    def update_remaining_files(self):
        """Met à jour l'affichage des fichiers restants / total ajouté."""
        if hasattr(self, "queued_stacker") and self.processing:
            try:
                # Get stats directly from the stacker instance
                total_queued = self.queued_stacker.files_in_queue
                processed_total = self.queued_stacker.processed_files_count # All dequeued files
                remaining_estimated = max(0, total_queued - processed_total)
                # Update the variable (e.g., "50/150")
                self.remaining_files_var.set(f"{remaining_estimated}/{total_queued}")
            except tk.TclError: pass # Ignore if UI destroyed
            except AttributeError: # Handle if stacker becomes unavailable unexpectedly
                try: self.remaining_files_var.set(self.tr("no_files_waiting"))
                except tk.TclError: pass
            except Exception as e:
                print(f"Error updating remaining files display: {e}")
                try: self.remaining_files_var.set("Error")
                except tk.TclError: pass
        elif not self.processing:
             # Set to default state when not processing
             try: self.remaining_files_var.set(self.tr("no_files_waiting"))
             except tk.TclError: pass


    def update_additional_folders_display(self):
        """Met à jour l'affichage du nombre de dossiers supplémentaires en attente."""
        count = 0
        if hasattr(self, 'queued_stacker'):
            # Access the list safely using the lock
            with self.queued_stacker.folders_lock:
                count = len(self.queued_stacker.additional_folders)

        try:
            if count == 0:
                self.additional_folders_var.set(self.tr('no_additional_folders'))
            elif count == 1:
                self.additional_folders_var.set(self.tr('1 additional folder'))
            else:
                # Use format for robustness if translation is missing {count}
                self.additional_folders_var.set(
                     self.tr('{count} additional folders', default="{count} add. folders").format(count=count)
                 )
        except tk.TclError: pass # Ignore if UI destroyed


    def stop_processing(self):
        """ Demande l'arrêt du traitement via QueuedStacker. """
        if self.processing and hasattr(self, "queued_stacker") and self.queued_stacker.is_running():
            self.update_progress_gui(self.tr("stacking_stopping"), None)
            self.queued_stacker.stop() # Signal the backend thread to stop
            # Disable stop button immediately to prevent multiple clicks
            if hasattr(self,'stop_button'): self.stop_button.config(state=tk.DISABLED)
            # The _processing_finished method will re-enable start etc. when the worker actually exits
        elif self.processing:
            # Case where UI thinks it's processing, but backend isn't running
            self.update_progress_gui("Tentative d'arrêt, mais worker inactif ou déjà arrêté.", None)
            self._processing_finished() # Reset UI immediately


    def _processing_finished(self):
        """Actions performed in the main GUI thread after processing ends/stops."""
        self.processing = False # Update UI processing flag
        if hasattr(self, 'progress_manager'):
             self.progress_manager.stop_timer() # Stop elapsed timer

        # --- Gather Final Stats from QueuedStacker ---
        final_message = self.tr("stacking_finished")
        final_progress = 100
        final_stack_path = None
        error_message = None
        images_stacked = 0
        aligned_count = 0
        failed_align_count = 0
        failed_stack_count = 0
        skipped_count = 0
        total_exposure = 0.0

        if hasattr(self, "queued_stacker"):
            q_stacker = self.queued_stacker
            final_stack_path = q_stacker.final_stacked_path
            images_stacked = q_stacker.images_in_cumulative_stack # Images combined into final
            aligned_count = q_stacker.aligned_files_count
            failed_align_count = q_stacker.failed_align_count
            failed_stack_count = q_stacker.failed_stack_count
            skipped_count = q_stacker.skipped_files_count
            total_exposure = q_stacker.total_exposure_seconds

            # Determine final status message
            if q_stacker.stop_processing: # Check if stop was requested
                final_message = self.tr("processing_stopped")
                # Set progress bar to current value if stopped early
                if hasattr(self,'progress_bar'): final_progress = self.progress_bar['value']
            elif q_stacker.processing_error: # Check for critical errors
                error_message = f"{self.tr('stacking_error_msg')}\n{q_stacker.processing_error}"
                final_message = error_message
                if hasattr(self,'progress_bar'): final_progress = self.progress_bar['value']
            elif images_stacked == 0: # No errors, but nothing stacked
                final_message = self.tr("no_stacks_created")
            # else: Success, use default "Finished" message initially

            # Update final aligned count display
            default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
            try: self.aligned_files_var.set(default_aligned_fmt.format(count=aligned_count))
            except tk.TclError: pass
        else:
            final_message = "Erreur: Stacker non initialisé."
            final_progress = 0

        # --- Generate and Display Final Report ---
        report_text = ""
        if not error_message and (images_stacked > 0 or aligned_count > 0 or failed_align_count > 0 or skipped_count > 0):
            elapsed_total_seconds = 0
            if self.global_start_time:
                elapsed_total_seconds = time.monotonic() - self.global_start_time

            report_lines = [
                f"--- {self.tr('processing_report_title')} ---",
                f"{self.tr('report_images_stacked')} {images_stacked}",
                f"{self.tr('report_total_exposure')} {self._format_duration(total_exposure)}",
                f"{self.tr('report_total_time')} {self._format_duration(elapsed_total_seconds)}",
                f"Alignés: {aligned_count}, Échecs Align: {failed_align_count}",
                f"Échecs Stack: {failed_stack_count}, Autres Ignorés: {skipped_count}"
            ]
            report_text = "\n".join(report_lines)
            # Use report as the final message if generated
            final_message = report_text

        # Update status text area with the final message/report
        self.update_progress_gui(final_message, final_progress)
        if hasattr(self, 'status_text'):
            try:
                self.status_text.config(state=tk.NORMAL)
                # Add final marker if not already part of the report
                if not report_text: self.status_text.insert(tk.END, "\n")
                self.status_text.insert(tk.END, "--- Traitement Terminé/Arrêté ---\n")
                self.status_text.see(tk.END)
                self.status_text.config(state=tk.DISABLED)
            except tk.TclError: pass # Ignore if widget destroyed

        # --- Reset UI Controls ---
        self._set_parameter_widgets_state(tk.NORMAL) # Re-enable parameter widgets
        if hasattr(self, 'start_button'): self.start_button.config(state=tk.NORMAL)
        if hasattr(self, 'stop_button'): self.stop_button.config(state=tk.DISABLED)
        if hasattr(self, 'add_files_button'): self.add_files_button.config(state=tk.DISABLED) # Disable Add Folder

        # Set final time values
        if hasattr(self, 'remaining_time_var'):
             try: self.remaining_time_var.set("00:00:00")
             except tk.TclError: pass

        # --- Show Final Popups / Load Final Preview ---
        if error_message:
            messagebox.showerror(self.tr("error"), error_message)
        elif not error_message and final_stack_path and os.path.exists(final_stack_path):
            # Show simple completion message (report is in status area)
            messagebox.showinfo(self.tr("info"), f"{self.tr('stacking_complete_msg')}\n{final_stack_path}")
            # Attempt to load the final stack into the preview
            try:
                self.update_progress_gui(f"Chargement aperçu final: {os.path.basename(final_stack_path)}...", None)
                # Load the final image data
                final_image_data = load_and_validate_fits(final_stack_path)
                if final_image_data is not None:
                    final_header = fits.getheader(final_stack_path)
                    # Update internal data and trigger preview refresh
                    self.current_preview_data = final_image_data
                    self.current_stack_header = final_header
                    self.refresh_preview()
                    # Update info text
                    if final_header: self.update_image_info(final_header)
                else:
                    raise ValueError("Impossible de charger le fichier stack final pour l'aperçu.")
            except Exception as e:
                self.update_progress_gui(f"⚠️ {self.tr('Error loading final stack preview')}: {e}", None)
                traceback.print_exc(limit=2)
                messagebox.showerror(self.tr("error"), f"{self.tr('Error loading final preview')}:\n{e}")
        elif hasattr(self, 'queued_stacker') and not q_stacker.stop_processing and images_stacked == 0:
            # Processing finished normally but nothing was stacked
            messagebox.showwarning(self.tr("warning"), self.tr("no_stacks_created"))


    def _format_duration(self, seconds):
        """ Formate une durée en secondes vers H M S ou M S ou S. """
        try:
            secs = int(round(float(seconds)))
            if secs < 0: return "N/A"
            if secs < 60:
                return f"{secs} {self.tr('report_seconds', 's')}" # Shorter unit
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
        """Enable/disable control widgets, EXCLUDING preview controls if disabling."""
        # List widgets related to setting processing parameters
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
        # Add other processing-related widgets if any

        # List widgets related ONLY to the preview display adjustments
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
        # Histogram reset button should always be active? Or disabled during processing? Let's disable.
        if hasattr(self, 'hist_reset_btn'): preview_widgets.append(self.hist_reset_btn)

        # --- *** NEW: Add Weighting Widgets to disable *** ---
        if hasattr(self, 'use_weighting_check'): processing_widgets.append(self.use_weighting_check)
        # Inclure les sous-options ici aussi pour être sûr qu'elles sont bien traitées
        # même si _update_weighting_options_state() s'en occupe aussi.
        if hasattr(self, 'weight_snr_check'): processing_widgets.append(self.weight_snr_check)
        if hasattr(self, 'weight_stars_check'): processing_widgets.append(self.weight_stars_check)
        if hasattr(self, 'snr_exp_spinbox'): processing_widgets.append(self.snr_exp_spinbox)
        if hasattr(self, 'stars_exp_spinbox'): processing_widgets.append(self.stars_exp_spinbox)
        if hasattr(self, 'min_w_spinbox'): processing_widgets.append(self.min_w_spinbox)
        # --- End New ---

        # Determine which widgets to affect based on state
        widgets_to_set = []
        if state == tk.NORMAL:
            # Enable ALL widgets when processing finishes
            self._update_weighting_options_state()
            widgets_to_set = processing_widgets + preview_widgets
        else: # tk.DISABLED
            # Disable ONLY processing parameter widgets, leave preview widgets active
            widgets_to_set = processing_widgets
            # Ensure preview widgets remain enabled (or explicitly enable them)
            for widget in preview_widgets:
                 if widget and hasattr(widget, 'winfo_exists') and widget.winfo_exists():
                     try: widget.config(state=tk.NORMAL)
                     except tk.TclError: pass

        # Apply the state to the selected widgets
        for widget in widgets_to_set:
            if widget and hasattr(widget, 'winfo_exists') and widget.winfo_exists():
                try: widget.config(state=state)
                except tk.TclError: pass # Ignore errors for destroyed widgets


    def _debounce_resize(self, event=None):
        """ Debounce window resize event for preview redraw. """
        if self._after_id_resize:
             try: self.root.after_cancel(self._after_id_resize)
             except tk.TclError: pass
        try:
            # Schedule the redraw after a short delay
            self._after_id_resize = self.root.after(300, self._refresh_preview_on_resize)
        except tk.TclError: pass # Ignore if root is destroyed

    def _refresh_preview_on_resize(self):
        """ Trigger redraw on preview manager after resize debounce. """
        if hasattr(self, 'preview_manager'):
            self.preview_manager.trigger_redraw()
        # Also redraw histogram as its size might change
        if hasattr(self, 'histogram_widget') and self.histogram_widget.winfo_exists():
             try:
                  self.histogram_widget.canvas.draw_idle()
             except tk.TclError: pass


    def _on_closing(self):
        """Handles window closing event."""
        # Check if processing is active (using the UI flag)
        if self.processing:
            if messagebox.askokcancel(self.tr("quit"), self.tr("quit_while_processing")):
                print("Arrêt demandé via fermeture fenêtre...")
                self.stop_processing() # Request backend stop
                # Wait a short time for the thread to potentially finish cleanly
                if self.thread and self.thread.is_alive():
                    self.thread.join(timeout=1.5) # Wait max 1.5 seconds
                # Check again if thread is still alive
                if hasattr(self, "queued_stacker") and self.queued_stacker.is_running():
                    print("Warning: Worker thread did not exit cleanly after stop request.")
                # Save settings and destroy window regardless
                self._save_settings_and_destroy()
            else:
                return # User cancelled closing
        else:
            # Not processing, save settings and close directly
            self._save_settings_and_destroy()

    def _save_settings_and_destroy(self):
        """ Saves geometry/settings and closes the application. """
        try:
            # Get current geometry if window exists
            if self.root.winfo_exists():
                 self.settings.window_geometry = self.root.geometry()
        except tk.TclError: pass # Ignore error if window already destroyed
        # Update other settings from UI just before saving
        self.settings.update_from_ui(self)
        self.settings.save_settings()
        print("Fermeture de l'application.")
        self.root.destroy() # Destroy the main window


    def update_progress_gui(self, message, progress=None):
        """Handles progress messages from the backend thread."""
        # Check for special messages first
        if isinstance(message, str):
            if message.startswith("folder_count_update:"):
                # Schedule the display update in the main thread
                try: self.root.after_idle(self.update_additional_folders_display)
                except tk.TclError: pass
                return # Don't display this message in the log
            # elif message.startswith("stats_update:"): # Could add other special handlers
            #     return

        # Pass regular messages to the ProgressManager
        if hasattr(self, "progress_manager") and self.progress_manager:
            # ProgressManager's update_progress should handle the after_idle call
            self.progress_manager.update_progress(message, progress)


# --- Main Execution ---
if __name__ == "__main__":
    # --- Theme Setup (Optional but recommended) ---
    try:
        _dummy_root = tk.Tk(); _dummy_root.withdraw() # Create temp root for style
        style = ttk.Style()
        available_themes = style.theme_names()
        # print("Available themes:", available_themes) # Debug
        theme_to_use = 'default'
        # Prefer modern themes if available
        preferred_themes = ['clam', 'alt', 'vista', 'xpnative'] # Windows examples
        # Add Linux/macOS themes if needed: e.g., 'aqua' (macOS), potentially others
        for t in preferred_themes:
            if t in available_themes:
                theme_to_use = t
                break
        print(f"Using theme: {theme_to_use}")
        style.theme_use(theme_to_use)

        # Configure accent button style (example)
        try:
            style.configure('Accent.TButton', font=('Segoe UI', 9, 'bold'), foreground='white', background='#0078D7')
        except tk.TclError:
            print("Warning: Could not configure Accent.TButton style (theme might not support it).")
        # Configure other styles if needed
        try:
             style.configure('Toolbutton.TButton', padding=1, font=('Segoe UI', 8))
        except tk.TclError: print("Warning: Could not configure Toolbutton.TButton style.")

        if '_dummy_root' in locals() and _dummy_root.winfo_exists():
             _dummy_root.destroy() # Clean up temp root

    except tk.TclError as theme_err:
        print(f"Error initializing Tk themes: {theme_err}. Using Tk default.")
        if '_dummy_root' in locals() and _dummy_root.winfo_exists():
             _dummy_root.destroy() # Clean up temp root
    # --- End Theme Setup ---

    # Create and run the application
    gui = SeestarStackerGUI()
    gui.root.mainloop()

# --- END OF FILE seestar/gui/main_window.py ---