# --- START OF FILE seestar/gui/main_window.py ---
"""
Module principal pour l'interface graphique de GSeestar.
Intègre la prévisualisation avancée et le traitement en file d'attente.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, font as tkFont # Import font module
import threading
import time
import numpy as np
from astropy.io import fits
import traceback
import math # Keep math import if needed elsewhere

# Seestar imports
from seestar.core.image_processing import load_and_validate_fits, debayer_image # Core functions
from seestar.localization import Localization
from seestar.queuep.queue_manager import SeestarQueuedStacker
# Import tools safely, handle potential import errors if tools are optional/complex
try:
    from seestar.tools.stretch import apply_auto_stretch as calculate_auto_stretch # Rename for clarity
    from seestar.tools.stretch import apply_auto_white_balance as calculate_auto_wb # Rename for clarity
    # Import ColorCorrection class needed for apply_auto_stretch fix
    from seestar.tools.stretch import ColorCorrection
    _tools_available = True
except ImportError as tool_err:
    print(f"Warning: Could not import stretch tools: {tool_err}. Auto WB/Stretch buttons disabled.")
    _tools_available = False
    # Define dummy functions if tools are not available to prevent NameErrors
    def calculate_auto_stretch(*args, **kwargs): return (0.0, 1.0)
    def calculate_auto_wb(*args, **kwargs): return (1.0, 1.0, 1.0)
    # Define dummy ColorCorrection class if import failed
    class ColorCorrection:
        @staticmethod
        def white_balance(data, r=1., g=1., b=1.): return data


# GUI Component Imports
from .file_handling import FileHandlingManager
from .preview import PreviewManager
from .progress import ProgressManager
from .settings import SettingsManager
from .histogram_widget import HistogramWidget # Import new widget

class SeestarStackerGUI:
    """
    GUI principale pour Seestar avec prévisualisation avancée et traitement en file.
    """

    def __init__(self):
        """Initialise l'interface graphique."""
        self.root = tk.Tk()
        self.localization = Localization("en") # Default language 'en'
        self.settings = SettingsManager() # Manages loading/saving settings
        self.queued_stacker = SeestarQueuedStacker() # The backend processing engine
        self.processing = False # Flag: Is processing currently active?
        self.thread = None # Holds the worker thread monitoring object

        # Data for Preview and Processing State
        self.current_preview_data = None # Holds raw data for *preview* (0-1 float, after load/debayer)
        self.current_stack_header = None # Header associated with the preview data
        self.debounce_timer_id = None    # Timer ID for debouncing slider updates

        # Processing stats (updated by callbacks/tracker)
        self.time_per_image = 0
        self.global_start_time = None

        # --- CORRECTED INITIALIZATION ORDER ---
        self.init_variables() # Initialize Tkinter variables first
        self.settings.load_settings() # Load settings from file
        self.language_var.set(self.settings.language) # Set Tk var from loaded settings
        self.localization.set_language(self.settings.language)

        # Create File Handler early as it's needed by layout commands
        self.file_handler = FileHandlingManager(self)

        self.create_layout() # Creates all widgets (references self.file_handler)

        # Now initialize other managers that NEED widgets from create_layout
        self.init_managers()

        # Apply loaded settings to UI elements *after* they are created
        self.settings.apply_to_ui(self)
        self.update_ui_language() # Set initial language for widget text

        # Connect stacker callbacks AFTER managers are initialized
        self.queued_stacker.set_progress_callback(self.update_progress_gui)
        self.queued_stacker.set_preview_callback(self.update_preview_from_stacker)

        # Window setup
        self.root.title(self.tr("title"))
        try:
             self.root.geometry(self.settings.window_geometry) # Use loaded/default geometry
        except tk.TclError:
             print("Warning: Invalid window geometry in settings, using default.")
             self.root.geometry("1200x750") # Fallback default size
        self.root.minsize(1100, 650) # Minimum reasonable size
        self.root.bind("<Configure>", self._debounce_resize) # Handle window resize events
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing) # Handle window close button

    def init_variables(self):
        """Initialise les variables Tkinter pour les widgets."""
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.reference_image_path = tk.StringVar()
        self.stacking_mode = tk.StringVar(value="kappa-sigma")
        self.kappa = tk.DoubleVar(value=2.5)
        self.batch_size = tk.IntVar(value=0)
        self.correct_hot_pixels = tk.BooleanVar(value=True)
        self.hot_pixel_threshold = tk.DoubleVar(value=3.0)
        self.neighborhood_size = tk.IntVar(value=5)
        self.cleanup_temp_var = tk.BooleanVar(value=True)
        self.preview_stretch_method = tk.StringVar(value="Asinh")
        self.preview_black_point = tk.DoubleVar(value=0.01)
        self.preview_white_point = tk.DoubleVar(value=0.99)
        self.preview_gamma = tk.DoubleVar(value=1.0)
        self.preview_r_gain = tk.DoubleVar(value=1.0)
        self.preview_g_gain = tk.DoubleVar(value=1.0)
        self.preview_b_gain = tk.DoubleVar(value=1.0)
        self.language_var = tk.StringVar(value='en')
        self.remaining_files_var = tk.StringVar(value=self.tr("no_files_waiting", default="No files waiting"))
        self.additional_folders_var = tk.StringVar(value=self.tr("no_additional_folders", default="None"))
        default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
        self.aligned_files_var = tk.StringVar(value=default_aligned_fmt.format(count="--"))
        self.remaining_time_var = tk.StringVar(value="--:--:--")
        self.elapsed_time_var = tk.StringVar(value="00:00:00")
        self._after_id_resize = None

    def init_managers(self):
        """Initialise les gestionnaires APRÈS la création des widgets."""
        if hasattr(self, 'progress_bar') and hasattr(self, 'status_text'):
             self.progress_manager = ProgressManager(
                 self.progress_bar, self.status_text,
                 self.remaining_time_var, self.elapsed_time_var
             )
        else:
            print("Error: Progress widgets not found for ProgressManager")

        if hasattr(self, 'preview_canvas'):
             self.preview_manager = PreviewManager(self.preview_canvas)
        else:
            print("Error: Preview canvas not found for PreviewManager")

        if hasattr(self, 'histogram_widget') and self.histogram_widget:
             self.histogram_widget.range_change_callback = self.update_stretch_from_histogram
        else:
            print("Error: HistogramWidget reference not found after create_layout")

        if not hasattr(self, 'file_handler'):
             print("Error: FileHandlingManager not created before init_managers (should not happen)")
             self.file_handler = FileHandlingManager(self) # Create as fallback

        self.show_initial_preview()

    def show_initial_preview(self):
        """Affiche un message d'accueil ou une image test."""
        if hasattr(self, 'preview_manager') and self.preview_manager:
             self.preview_manager.clear_preview(self.tr("Select input/output folders.", default="Select input/output folders."))
        if hasattr(self, 'histogram_widget') and self.histogram_widget:
             self.histogram_widget.plot_histogram(None)

    def tr(self, key, default=None):
        """Raccourci pour la localisation."""
        return self.localization.get(key, default=default)

    def create_layout(self):
        """Crée la disposition des widgets de l'interface."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        left_frame = ttk.Frame(paned_window, width=450, height=700)
        left_frame.pack_propagate(False)
        paned_window.add(left_frame, weight=1)
        right_frame = ttk.Frame(paned_window, width=750, height=700)
        right_frame.pack_propagate(False)
        paned_window.add(right_frame, weight=3)

        lang_frame = ttk.Frame(left_frame)
        lang_frame.pack(fill=tk.X, pady=(5, 5), padx=5)
        ttk.Label(lang_frame, text="Language/Langue:").pack(side=tk.LEFT, padx=(0, 5))
        self.language_combo = ttk.Combobox(lang_frame, textvariable=self.language_var,
                                           values=("en", "fr"), width=8, state="readonly")
        self.language_combo.pack(side=tk.LEFT)
        self.language_combo.bind("<<ComboboxSelected>>", self.change_language)

        control_notebook = ttk.Notebook(left_frame)
        control_notebook.pack(fill=tk.BOTH, expand=True, pady=(0,5), padx=5)

        tab_stacking = ttk.Frame(control_notebook)
        control_notebook.add(tab_stacking, text=f' {self.tr("tab_stacking")} ')

        self.folders_frame = ttk.LabelFrame(tab_stacking, text=self.tr("Folders"))
        self.folders_frame.pack(fill=tk.X, pady=5, padx=5)

        in_subframe = ttk.Frame(self.folders_frame)
        in_subframe.pack(fill=tk.X, padx=5, pady=(5, 2))
        self.input_label = ttk.Label(in_subframe, text=self.tr("input_folder"), width=8, anchor="w")
        self.input_label.pack(side=tk.LEFT)
        self.browse_input_button = ttk.Button(in_subframe, text=self.tr("browse_input_button", "Browse..."), command=self.file_handler.browse_input, width=10)
        self.browse_input_button.pack(side=tk.RIGHT)
        self.input_entry = ttk.Entry(in_subframe, textvariable=self.input_path)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))

        out_subframe = ttk.Frame(self.folders_frame)
        out_subframe.pack(fill=tk.X, padx=5, pady=(2, 5))
        self.output_label = ttk.Label(out_subframe, text=self.tr("output_folder"), width=8, anchor="w")
        self.output_label.pack(side=tk.LEFT)
        self.browse_output_button = ttk.Button(out_subframe, text=self.tr("browse_output_button", "Browse..."), command=self.file_handler.browse_output, width=10)
        self.browse_output_button.pack(side=tk.RIGHT)
        self.output_entry = ttk.Entry(out_subframe, textvariable=self.output_path)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))

        ref_frame = ttk.Frame(self.folders_frame)
        ref_frame.pack(fill=tk.X, padx=5, pady=(2, 5))
        self.reference_label = ttk.Label(ref_frame, text=self.tr("reference_image"), width=8, anchor="w")
        self.reference_label.pack(side=tk.LEFT)
        self.browse_ref_button = ttk.Button(ref_frame, text=self.tr("browse_ref_button", "Browse..."), command=self.file_handler.browse_reference, width=10)
        self.browse_ref_button.pack(side=tk.RIGHT)
        self.ref_entry = ttk.Entry(ref_frame, textvariable=self.reference_image_path)
        self.ref_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))

        self.options_frame = ttk.LabelFrame(tab_stacking, text=self.tr("options"))
        self.options_frame.pack(fill=tk.X, pady=5, padx=5)
        method_frame = ttk.Frame(self.options_frame)
        method_frame.pack(fill=tk.X, padx=5, pady=5)
        self.stacking_method_label = ttk.Label(method_frame, text=self.tr("stacking_method"))
        self.stacking_method_label.pack(side=tk.LEFT, padx=(0, 5))
        self.stacking_combo = ttk.Combobox(method_frame, textvariable=self.stacking_mode, values=("mean", "median", "kappa-sigma", "winsorized-sigma"), width=15, state="readonly")
        self.stacking_combo.pack(side=tk.LEFT)
        self.kappa_label = ttk.Label(method_frame, text=self.tr("kappa_value"))
        self.kappa_label.pack(side=tk.LEFT, padx=(10, 2))
        self.kappa_spinbox = ttk.Spinbox(method_frame, from_=1.0, to=5.0, increment=0.1, textvariable=self.kappa, width=5)
        self.kappa_spinbox.pack(side=tk.LEFT)

        batch_frame = ttk.Frame(self.options_frame)
        batch_frame.pack(fill=tk.X, padx=5, pady=5)
        self.batch_size_label = ttk.Label(batch_frame, text=self.tr("batch_size"))
        self.batch_size_label.pack(side=tk.LEFT, padx=(0, 5))
        self.batch_spinbox = ttk.Spinbox(batch_frame, from_=0, to=500, increment=1, textvariable=self.batch_size, width=5)
        self.batch_spinbox.pack(side=tk.LEFT)
        self.batch_size_auto_label = ttk.Label(batch_frame, text=self.tr("batch_size_auto"))
        self.batch_size_auto_label.pack(side=tk.LEFT, padx=(2, 10))

        self.hp_frame = ttk.LabelFrame(tab_stacking, text=self.tr("hot_pixels_correction"))
        self.hp_frame.pack(fill=tk.X, pady=5, padx=5)
        hp_check_frame = ttk.Frame(self.hp_frame)
        hp_check_frame.pack(fill=tk.X, padx=5, pady=2)
        self.hot_pixels_check = ttk.Checkbutton(hp_check_frame, text=self.tr("perform_hot_pixels_correction"), variable=self.correct_hot_pixels)
        self.hot_pixels_check.pack(side=tk.LEFT, padx=(0, 10))
        hp_params_frame = ttk.Frame(self.hp_frame)
        hp_params_frame.pack(fill=tk.X, padx=5, pady=(2,5))
        self.hot_pixel_threshold_label = ttk.Label(hp_params_frame, text=self.tr("hot_pixel_threshold"))
        self.hot_pixel_threshold_label.pack(side=tk.LEFT)
        self.hp_thresh_spinbox = ttk.Spinbox(hp_params_frame, from_=1.0, to=10.0, increment=0.1, textvariable=self.hot_pixel_threshold, width=5)
        self.hp_thresh_spinbox.pack(side=tk.LEFT, padx=5)
        self.neighborhood_size_label = ttk.Label(hp_params_frame, text=self.tr("neighborhood_size"))
        self.neighborhood_size_label.pack(side=tk.LEFT)
        self.hp_neigh_spinbox = ttk.Spinbox(hp_params_frame, from_=3, to=15, increment=2, textvariable=self.neighborhood_size, width=4)
        self.hp_neigh_spinbox.pack(side=tk.LEFT, padx=5)

        self.post_proc_opts_frame = ttk.LabelFrame(tab_stacking, text=self.tr('post_proc_opts_frame_label'))
        self.post_proc_opts_frame.pack(fill=tk.X, pady=5, padx=5)
        self.cleanup_temp_check = ttk.Checkbutton(self.post_proc_opts_frame, text=self.tr("cleanup_temp_check_label"), variable=self.cleanup_temp_var)
        self.cleanup_temp_check.pack(side=tk.LEFT, padx=5, pady=5)

        tab_preview = ttk.Frame(control_notebook)
        control_notebook.add(tab_preview, text=f' {self.tr("tab_preview")} ')

        self.wb_frame = ttk.LabelFrame(tab_preview, text=self.tr("white_balance"))
        self.wb_frame.pack(fill=tk.X, pady=5, padx=5)
        self.wb_r_ctrls = self._create_slider_spinbox_group(self.wb_frame, self.tr("wb_r"), 0.1, 5.0, 0.01, self.preview_r_gain)
        self.wb_g_ctrls = self._create_slider_spinbox_group(self.wb_frame, self.tr("wb_g"), 0.1, 5.0, 0.01, self.preview_g_gain)
        self.wb_b_ctrls = self._create_slider_spinbox_group(self.wb_frame, self.tr("wb_b"), 0.1, 5.0, 0.01, self.preview_b_gain)
        wb_btn_frame = ttk.Frame(self.wb_frame)
        wb_btn_frame.pack(fill=tk.X, pady=5)
        self.auto_wb_button = ttk.Button(wb_btn_frame, text=self.tr("auto_wb"), command=self.apply_auto_white_balance, state=tk.NORMAL if _tools_available else tk.DISABLED)
        self.auto_wb_button.pack(side=tk.LEFT, padx=5)
        self.reset_wb_button = ttk.Button(wb_btn_frame, text=self.tr("reset_wb"), command=self.reset_white_balance)
        self.reset_wb_button.pack(side=tk.LEFT, padx=5)

        self.stretch_frame_controls = ttk.LabelFrame(tab_preview, text=self.tr("stretch_options"))
        self.stretch_frame_controls.pack(fill=tk.X, pady=5, padx=5)
        stretch_method_frame = ttk.Frame(self.stretch_frame_controls)
        stretch_method_frame.pack(fill=tk.X, pady=2)
        self.stretch_method_label = ttk.Label(stretch_method_frame, text=self.tr("stretch_method"))
        self.stretch_method_label.pack(side=tk.LEFT, padx=(5,5))
        self.stretch_combo = ttk.Combobox(stretch_method_frame, textvariable=self.preview_stretch_method, values=("Linear", "Asinh", "Log"), width=15, state="readonly")
        self.stretch_combo.pack(side=tk.LEFT)
        self.stretch_combo.bind("<<ComboboxSelected>>", self._debounce_refresh_preview)
        self.stretch_bp_ctrls = self._create_slider_spinbox_group(self.stretch_frame_controls, self.tr("stretch_bp"), 0.0, 1.0, 0.001, self.preview_black_point, callback=self.update_histogram_lines_from_sliders)
        self.stretch_wp_ctrls = self._create_slider_spinbox_group(self.stretch_frame_controls, self.tr("stretch_wp"), 0.0, 1.0, 0.001, self.preview_white_point, callback=self.update_histogram_lines_from_sliders)
        self.stretch_gamma_ctrls = self._create_slider_spinbox_group(self.stretch_frame_controls, self.tr("stretch_gamma"), 0.1, 5.0, 0.01, self.preview_gamma)
        stretch_btn_frame = ttk.Frame(self.stretch_frame_controls)
        stretch_btn_frame.pack(fill=tk.X, pady=5)
        self.auto_stretch_button = ttk.Button(stretch_btn_frame, text=self.tr("auto_stretch"), command=self.apply_auto_stretch, state=tk.NORMAL if _tools_available else tk.DISABLED)
        self.auto_stretch_button.pack(side=tk.LEFT, padx=5)
        self.reset_stretch_button = ttk.Button(stretch_btn_frame, text=self.tr("reset_stretch"), command=self.reset_stretch)
        self.reset_stretch_button.pack(side=tk.LEFT, padx=5)

        self.progress_frame = ttk.LabelFrame(left_frame, text=self.tr("progress"))
        self.progress_frame.pack(fill=tk.X, pady=(10, 0), padx=5, side=tk.BOTTOM)
        self.progress_bar = ttk.Progressbar(self.progress_frame, maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=(5, 2))

        time_frame = ttk.Frame(self.progress_frame)
        time_frame.pack(fill=tk.X, padx=5, pady=2)
        self.remaining_time_label = ttk.Label(time_frame, text=self.tr("estimated_time"))
        self.remaining_time_label.pack(side=tk.LEFT)
        self.remaining_time_value = ttk.Label(time_frame, textvariable=self.remaining_time_var, font=tkFont.Font(weight='bold'), width=9, anchor='w')
        self.remaining_time_value.pack(side=tk.LEFT, padx=(2, 10))
        self.elapsed_time_label = ttk.Label(time_frame, text=self.tr("elapsed_time"))
        self.elapsed_time_label.pack(side=tk.LEFT)
        self.elapsed_time_value = ttk.Label(time_frame, textvariable=self.elapsed_time_var, font=tkFont.Font(weight='bold'), width=9, anchor='w')
        self.elapsed_time_value.pack(side=tk.LEFT, padx=2)

        files_info_frame = ttk.Frame(self.progress_frame)
        files_info_frame.pack(fill=tk.X, padx=5, pady=2)
        self.remaining_static_label = ttk.Label(files_info_frame, text=self.tr("Remaining:"))
        self.remaining_static_label.pack(side=tk.LEFT)
        self.remaining_value_label = ttk.Label(files_info_frame, textvariable=self.remaining_files_var, width=12, anchor='w')
        self.remaining_value_label.pack(side=tk.LEFT, padx=(2,10))
        self.aligned_files_label = ttk.Label(files_info_frame, textvariable=self.aligned_files_var, width=12, anchor='w')
        self.aligned_files_label.pack(side=tk.LEFT, padx=(10,0))
        self.additional_value_label = ttk.Label(files_info_frame, textvariable=self.additional_folders_var, anchor='e')
        self.additional_value_label.pack(side=tk.RIGHT)
        self.additional_static_label = ttk.Label(files_info_frame, text=self.tr("Additional:"))
        self.additional_static_label.pack(side=tk.RIGHT, padx=(0, 2))

        status_text_frame = ttk.Frame(self.progress_frame)
        status_text_font = tkFont.Font(family="Arial", size=8)
        status_text_height_pixels = status_text_font.metrics('linespace') * 5
        status_text_frame.pack(fill=tk.X, expand=False, padx=5, pady=(2, 5))
        self.status_text = tk.Text(status_text_frame, height=5, wrap=tk.WORD, bd=0, font=status_text_font, relief=tk.FLAT, state=tk.DISABLED)
        self.status_scrollbar = ttk.Scrollbar(status_text_frame, orient="vertical", command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=self.status_scrollbar.set)
        self.status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.pack(side=tk.LEFT, fill=tk.X, expand=True)

        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=(5, 5), padx=5, side=tk.BOTTOM)
        try:
            accent_style = 'Accent.TButton' if 'Accent.TButton' in ttk.Style().element_names() else 'TButton'
        except tk.TclError:
            accent_style = 'TButton'
        self.start_button = ttk.Button(control_frame, text=self.tr("start"), command=self.start_processing, style=accent_style)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5, ipady=2)
        self.stop_button = ttk.Button(control_frame, text=self.tr("stop"), command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5, ipady=2)
        self.add_files_button = ttk.Button(control_frame, text=self.tr("add_folder_button"), command=self.file_handler.add_folder, state=tk.DISABLED)
        self.add_files_button.pack(side=tk.RIGHT, padx=5, pady=5, ipady=2)

        self.preview_frame = ttk.LabelFrame(right_frame, text=self.tr("preview"))
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=(5,5), padx=5)
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="#1E1E1E", highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        self.histogram_frame = ttk.LabelFrame(right_frame, text=self.tr("histogram"))
        hist_fig_height_inches = 2.2
        hist_fig_dpi = 80
        hist_height_pixels = int(hist_fig_height_inches * hist_fig_dpi * 1.1)
        self.histogram_frame.pack(fill=tk.X, expand=False, pady=(0,5), padx=5, side=tk.BOTTOM)
        self.histogram_frame.pack_propagate(False)
        self.histogram_frame.config(height=hist_height_pixels)

        self.histogram_widget = HistogramWidget(self.histogram_frame, range_change_callback=self.update_stretch_from_histogram)
        self.histogram_widget.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=(0,2), pady=(0,2))

        font_small = tkFont.Font(size=8)
        self.hist_reset_btn = ttk.Button(self.histogram_frame, text="R", command=self.histogram_widget.reset_zoom, width=2)
        self.hist_reset_btn.pack(side=tk.RIGHT, anchor=tk.NE, padx=(0,2), pady=2)

        self._store_widget_references()

    def _create_slider_spinbox_group(self, parent, label_text, min_val, max_val, step, tk_var, callback=None):
        """Helper to create a consistent Slider + SpinBox group with debouncing."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=(1,1))
        ttk.Label(frame, text=self.tr(label_text, default=label_text), width=7).pack(side=tk.LEFT)
        decimals = 0
        if step > 0:
             log_step = math.log10(step)
             if log_step < 0:
                 decimals = abs(int(log_step))
        format_str = f"%.{decimals}f"
        spinbox = ttk.Spinbox(frame, from_=min_val, to=max_val, increment=step,
                              textvariable=tk_var, width=7, justify=tk.RIGHT,
                              command=self._debounce_refresh_preview, format=format_str)
        spinbox.pack(side=tk.RIGHT, padx=(5,0))
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
        slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=tk_var,
                           orient=tk.HORIZONTAL, command=on_scale_change)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        return {'frame': frame, 'slider': slider, 'spinbox': spinbox}

    def _store_widget_references(self):
        """Stores references to widgets that need language updates."""
        notebook_widget = None
        for child in self.root.winfo_children():
            if isinstance(child, ttk.Frame):
                 for pane in child.winfo_children():
                      if isinstance(pane, ttk.PanedWindow):
                           left_pane = pane.panes()[0]
                           left_frame_widget = self.root.nametowidget(left_pane)
                           for nb in left_frame_widget.winfo_children():
                                if isinstance(nb, ttk.Notebook):
                                     notebook_widget = nb
                                     break
                           break
                 break
        self.widgets_to_translate = {
            "tab_stacking": (notebook_widget, 0) if notebook_widget else None,
            "tab_preview": (notebook_widget, 1) if notebook_widget else None,
            "Folders": self.folders_frame, "input_folder": self.input_label, "browse_input_button": self.browse_input_button,
            "output_folder": self.output_label, "browse_output_button": self.browse_output_button,
            "reference_image": self.reference_label, "browse_ref_button": self.browse_ref_button,
            "options": self.options_frame, "stacking_method": self.stacking_method_label, "kappa_value": self.kappa_label, "batch_size": self.batch_size_label,
            "batch_size_auto": self.batch_size_auto_label,
            "hot_pixels_correction": self.hp_frame, "perform_hot_pixels_correction": self.hot_pixels_check,
            "hot_pixel_threshold": self.hot_pixel_threshold_label, "neighborhood_size": self.neighborhood_size_label,
            "post_proc_opts_frame_label": self.post_proc_opts_frame, "cleanup_temp_check_label": self.cleanup_temp_check,
            "white_balance": self.wb_frame, "wb_r": self.wb_r_ctrls['frame'].winfo_children()[0], "wb_g": self.wb_g_ctrls['frame'].winfo_children()[0], "wb_b": self.wb_b_ctrls['frame'].winfo_children()[0],
            "auto_wb": self.auto_wb_button, "reset_wb": self.reset_wb_button,
            "stretch_options": self.stretch_frame_controls, "stretch_method": self.stretch_method_label,
            "stretch_bp": self.stretch_bp_ctrls['frame'].winfo_children()[0], "stretch_wp": self.stretch_wp_ctrls['frame'].winfo_children()[0], "stretch_gamma": self.stretch_gamma_ctrls['frame'].winfo_children()[0],
            "auto_stretch": self.auto_stretch_button, "reset_stretch": self.reset_stretch_button,
            "progress": self.progress_frame, "estimated_time": self.remaining_time_label, "elapsed_time": self.elapsed_time_label,
            "Remaining:": self.remaining_static_label, "Additional:": self.additional_static_label,
            "start": self.start_button, "stop": self.stop_button, "add_folder_button": self.add_files_button,
            "preview": self.preview_frame, "histogram": self.histogram_frame,
        }

    def change_language(self, event=None):
        """Changes language and updates UI."""
        selected_lang = self.language_var.get()
        if self.localization.language != selected_lang:
            self.localization.set_language(selected_lang)
            self.settings.language = selected_lang
            self.settings.save_settings()
            self.update_ui_language()

    def update_ui_language(self):
        """Met à jour tous les éléments textuels de l'interface."""
        self.root.title(self.tr("title"))
        if not hasattr(self, 'widgets_to_translate'):
            return
        for key, widget_info in self.widgets_to_translate.items():
            translation = self.tr(key, default=f"_{key}_")
            try:
                if widget_info is None:
                    continue
                if isinstance(widget_info, tuple):
                    notebook_widget, index = widget_info
                    if notebook_widget and notebook_widget.winfo_exists() and index < notebook_widget.index("end"):
                        notebook_widget.tab(index, text=f' {translation} ')
                elif hasattr(widget_info, 'winfo_exists') and widget_info.winfo_exists():
                    widget = widget_info
                    if isinstance(widget, (ttk.Label, ttk.Button, ttk.Checkbutton)):
                        widget.config(text=translation)
                    elif isinstance(widget, ttk.LabelFrame):
                        widget.config(text=translation)
            except tk.TclError as e:
                print(f"Debug: TclError updating widget '{key}': {e}")
            except Exception as e:
                print(f"Debug: Error updating widget '{key}' ({type(widget_info)}): {e}")
        if not self.processing:
            self.remaining_files_var.set(self.tr("no_files_waiting"))
            self.update_additional_folders_display()
            default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
            self.aligned_files_var.set(default_aligned_fmt.format(count="--"))
        else:
            self.remaining_static_label.config(text=self.tr("Remaining:"))
            self.additional_static_label.config(text=self.tr("Additional:"))
            self.elapsed_time_label.config(text=self.tr("elapsed_time"))
            self.remaining_time_label.config(text=self.tr("estimated_time"))
        if self.current_preview_data is None and hasattr(self, 'preview_manager'):
            self.preview_manager.clear_preview(self.tr('image_info_waiting', default="Image info: waiting..."))

    def _debounce_refresh_preview(self, *args):
        """Requests a preview refresh after a short delay (e.g., after slider move)."""
        if self.debounce_timer_id:
            self.root.after_cancel(self.debounce_timer_id)
        self.debounce_timer_id = self.root.after(150, self.refresh_preview)

    def update_histogram_lines_from_sliders(self, *args):
        """Update histogram BP/WP lines when corresponding sliders/spinboxes change."""
        if hasattr(self, 'histogram_widget') and self.histogram_widget:
            try:
                bp = self.preview_black_point.get()
                wp = self.preview_white_point.get()
                self.histogram_widget.set_range(bp, wp)
            except tk.TclError:
                pass

    def update_stretch_from_histogram(self, black_point, white_point):
        """Callback when BP/WP lines are dragged on the histogram."""
        try:
            self.preview_black_point.set(round(black_point, 4))
            self.preview_white_point.set(round(white_point, 4))
        except tk.TclError:
            return
        try:
            self.stretch_bp_ctrls['slider'].set(black_point)
            self.stretch_wp_ctrls['slider'].set(white_point)
        except tk.TclError:
            pass
        self._debounce_refresh_preview()

    def refresh_preview(self):
        """
        Refreshes the image preview and histogram based on current preview settings
        using the self.current_preview_data.
        """
        if self.debounce_timer_id:
            try:
                self.root.after_cancel(self.debounce_timer_id)
            except tk.TclError:
                pass
            self.debounce_timer_id = None
        if (self.current_preview_data is None or
           not hasattr(self, 'preview_manager') or self.preview_manager is None or
           not hasattr(self, 'histogram_widget') or self.histogram_widget is None):
            if (not self.processing and self.input_path.get() and
               os.path.isdir(self.input_path.get())):
                self._try_show_first_input_image()
            else:
                if hasattr(self, 'preview_manager') and self.preview_manager:
                    self.preview_manager.clear_preview(self.tr('Select input/output folders.'))
                if hasattr(self, 'histogram_widget') and self.histogram_widget:
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
            }
        except tk.TclError:
            print("Error getting preview parameters.")
            return
        processed_pil_image, data_for_histogram = self.preview_manager.update_preview(
            self.current_preview_data,
            preview_params
        )
        self.histogram_widget.update_histogram(data_for_histogram)
        self.histogram_widget.set_range(preview_params["black_point"], preview_params["white_point"])

    def update_preview_from_stacker(self, stack_data, stack_header, stack_name):
        """Callback specifically for updates from the QueuedStacker worker thread."""
        if stack_data is None:
            return
        self.current_preview_data = stack_data.copy()
        self.current_stack_header = stack_header.copy() if stack_header else None
        try:
            self.root.after_idle(self.refresh_preview)
        except tk.TclError:
            pass
        if self.current_stack_header:
            try:
                self.root.after_idle(lambda h=self.current_stack_header: self.update_image_info(h))
            except tk.TclError:
                pass

    def update_image_info(self, header):
        """Updates the image info text area based on FITS header."""
        pass # Disabled for now

    def _try_show_first_input_image(self):
        """Attempts to load and display the first valid FITS image from the input folder."""
        input_folder = self.input_path.get()
        if not hasattr(self, 'preview_manager') or not hasattr(self, 'histogram_widget'):
            return
        if not input_folder or not os.path.isdir(input_folder):
            if hasattr(self, 'preview_manager'):
                self.preview_manager.clear_preview(self.tr("Input folder not found"))
            if hasattr(self, 'histogram_widget'):
                self.histogram_widget.plot_histogram(None)
            return
        try:
            files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith((".fit", ".fits"))])
            if not files:
                if hasattr(self, 'preview_manager'):
                    self.preview_manager.clear_preview(self.tr("No FITS files in input"))
                if hasattr(self, 'histogram_widget'):
                    self.histogram_widget.plot_histogram(None)
                return
            first_image_path = os.path.join(input_folder, files[0])
            self.update_progress_gui(f"Chargement aperçu: {files[0]}...", None)
            img_data = load_and_validate_fits(first_image_path)
            if img_data is None:
                raise ValueError(f"Échec chargement/validation {files[0]}")
            header = fits.getheader(first_image_path)
            img_for_preview = img_data
            if img_data.ndim == 2:
                bayer_pattern = header.get("BAYERPAT", self.settings.bayer_pattern)
                valid_bayer_patterns = ["GRBG", "RGGB", "GBRG", "BGGR"]
                if isinstance(bayer_pattern, str) and bayer_pattern.upper() in valid_bayer_patterns:
                    try:
                        img_for_preview = debayer_image(img_data, bayer_pattern.upper())
                    except ValueError as debayer_err:
                        self.update_progress_gui(f"⚠️ {self.tr('Error during debayering')}: {debayer_err}", None)
            self.current_preview_data = img_for_preview.copy()
            self.current_stack_header = header.copy() if header else None
            self.refresh_preview()
            if self.current_stack_header:
                self.update_image_info(self.current_stack_header)
        except FileNotFoundError:
            if hasattr(self, 'preview_manager'):
                self.preview_manager.clear_preview(self.tr("Input folder not found"))
            if hasattr(self, 'histogram_widget'):
                self.histogram_widget.plot_histogram(None)
        except ValueError as ve:
            self.update_progress_gui(f"⚠️ {self.tr('Error loading preview image')}: {ve}", None)
            if hasattr(self, 'preview_manager'):
                self.preview_manager.clear_preview(self.tr("Error loading preview (invalid format?)"))
            if hasattr(self, 'histogram_widget'):
                self.histogram_widget.plot_histogram(None)
        except Exception as e:
            self.update_progress_gui(f"⚠️ {self.tr('Error loading preview image')}: {e}", None)
            traceback.print_exc(limit=2)
            if hasattr(self, 'preview_manager'):
                self.preview_manager.clear_preview(self.tr("Error loading preview"))
            if hasattr(self, 'histogram_widget'):
                self.histogram_widget.plot_histogram(None)

    # --- Auto WB / Stretch Actions ---
    def apply_auto_white_balance(self):
        """Calculates and applies auto white balance to the preview."""
        if not _tools_available:
            messagebox.showerror(self.tr("error"), "Stretch tools not available (import error).")
            return
        if self.current_preview_data is None or self.current_preview_data.ndim != 3:
            messagebox.showwarning(self.tr("warning"), self.tr("Auto WB requires a color image preview."))
            return
        try:
            r_gain, g_gain, b_gain = calculate_auto_wb(self.current_preview_data)
            self.preview_r_gain.set(round(r_gain, 3))
            self.preview_g_gain.set(round(g_gain, 3))
            self.preview_b_gain.set(round(b_gain, 3))
            self.update_progress_gui(f"Auto WB appliqué (Aperçu): R={r_gain:.2f} G={g_gain:.2f} B={b_gain:.2f}", None)
            self.refresh_preview()
        except Exception as e:
            messagebox.showerror(self.tr("error"), f"{self.tr('Error during Auto WB')}: {e}")
            traceback.print_exc(limit=2)

    def reset_white_balance(self):
        """Resets white balance gains to 1.0."""
        self.preview_r_gain.set(1.0)
        self.preview_g_gain.set(1.0)
        self.preview_b_gain.set(1.0)
        self.refresh_preview()

    def apply_auto_stretch(self):
        """Calculates and applies auto stretch parameters to the preview."""
        if not _tools_available:
            messagebox.showerror(self.tr("error"), "Stretch tools not available (import error).")
            return
        if self.current_preview_data is None:
            messagebox.showwarning(self.tr("warning"), self.tr("Auto Stretch requires an image preview."))
            return
        # *** CORRECTED CALL ***
        # Apply temporary WB gains before calculating stretch
        temp_params = {"r": self.preview_r_gain.get(), "g": self.preview_g_gain.get(), "b": self.preview_b_gain.get()}
        # Use ColorCorrection directly, as PreviewManager doesn't have apply_white_balance
        data_wb = ColorCorrection.white_balance(self.current_preview_data, **temp_params)
        # ********************
        try:
             bp, wp = calculate_auto_stretch(data_wb) # Use helper on WB data
             self.preview_black_point.set(round(bp, 4))
             self.preview_white_point.set(round(wp, 4))
             if hasattr(self, 'histogram_widget'):
                 self.histogram_widget.set_range(bp, wp)
             self.update_progress_gui(f"Auto Stretch appliqué (Aperçu): BP={bp:.3f} WP={wp:.3f}", None)
             self.refresh_preview() # Trigger refresh with new BP/WP
        except Exception as e:
             messagebox.showerror(self.tr("error"), f"{self.tr('Error during Auto Stretch')}: {e}")
             traceback.print_exc(limit=2)

    def reset_stretch(self):
        """Resets stretch parameters to default (e.g., Asinh with default BP/WP)."""
        default_method = "Asinh"
        default_bp = 0.01
        default_wp = 0.99
        default_gamma = 1.0
        self.preview_stretch_method.set(default_method)
        self.preview_black_point.set(default_bp)
        self.preview_white_point.set(default_wp)
        self.preview_gamma.set(default_gamma)
        if hasattr(self, 'histogram_widget'):
            self.histogram_widget.set_range(default_bp, default_wp)
            self.histogram_widget.reset_zoom() # Also reset zoom
        self.refresh_preview()

    # --- Processing Control ---
    def start_processing(self):
        """Démarre le traitement des images via SeestarQueuedStacker."""
        input_folder = self.input_path.get()
        output_folder = self.output_path.get()
        if not input_folder or not output_folder:
            messagebox.showerror(self.tr("error"), self.tr("select_folders"))
            return
        if not os.path.isdir(input_folder):
            messagebox.showerror(self.tr("error"), f"{self.tr('input_folder_invalid')}:\n{input_folder}")
            return
        if not os.path.isdir(output_folder):
            try:
                os.makedirs(output_folder, exist_ok=True)
                self.update_progress_gui(f"{self.tr('Output folder created')}: {output_folder}", None)
            except Exception as e:
                messagebox.showerror(self.tr("error"), f"{self.tr('output_folder_invalid')}:\n{output_folder}\n{e}")
                return
        try:
            if not any(f.lower().endswith((".fit", ".fits")) for f in os.listdir(input_folder)):
                if not messagebox.askyesno(self.tr("warning"), self.tr("no_fits_found", default="No .fit/.fits files found in input folder. Start anyway?")):
                    return
        except Exception as e:
            messagebox.showerror(self.tr("error"), f"{self.tr('Error reading input folder')}:\n{e}")
            return
        self.processing = True
        self.time_per_image = 0
        self.global_start_time = time.monotonic()
        default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
        self.aligned_files_var.set(default_aligned_fmt.format(count=0))
        self._set_parameter_widgets_state(tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.add_files_button.config(state=tk.NORMAL)
        self.progress_manager.reset()
        self.progress_manager.start_timer()
        self.remaining_time_var.set("--:--:--")
        if hasattr(self, 'status_text'):
            try:
                self.status_text.config(state=tk.NORMAL)
                self.status_text.delete(1.0, tk.END)
                self.status_text.insert(tk.END, "--- Début du Traitement ---\n")
                self.status_text.config(state=tk.DISABLED)
            except tk.TclError:
                pass
        self.settings.update_from_ui(self)
        validation_messages = self.settings.validate_settings()
        if validation_messages:
            self.update_progress_gui("⚠️ Paramètres ajustés:", None)
            for msg in validation_messages:
                self.update_progress_gui(f"   - {msg}", None)
            self.settings.apply_to_ui(self)
        self.queued_stacker.stacking_mode = self.settings.stacking_mode
        self.queued_stacker.kappa = self.settings.kappa
        self.queued_stacker.batch_size = self.settings.batch_size
        self.queued_stacker.correct_hot_pixels = self.settings.correct_hot_pixels
        self.queued_stacker.hot_pixel_threshold = self.settings.hot_pixel_threshold
        self.queued_stacker.neighborhood_size = self.settings.neighborhood_size
        self.queued_stacker.bayer_pattern = self.settings.bayer_pattern
        self.queued_stacker.perform_cleanup = self.settings.cleanup_temp
        self.queued_stacker.aligner.reference_image_path = self.settings.reference_image_path or None
        self.update_progress_gui(self.tr("stacking_start"), 0)
        processing_started = self.queued_stacker.start_processing(
            self.settings.input_folder,
            self.settings.output_folder,
            self.settings.reference_image_path
        )
        if processing_started:
            self.thread = threading.Thread(
                target=self._track_processing_progress,
                daemon=True,
                name="GUI_ProgressTracker"
            )
            self.thread.start()
        else:
            self.update_progress_gui("❌ Échec démarrage du thread de traitement.", None)
            messagebox.showerror(self.tr("error"), self.tr("Failed to start processing."))
            self._processing_finished()

    def _track_processing_progress(self):
        """Monitors the QueuedStacker worker thread in the background."""
        while self.processing and hasattr(self, "queued_stacker"):
            try:
                if not self.queued_stacker.is_running():
                    self.root.after(0, self._processing_finished)
                    break
                q_stacker = self.queued_stacker
                processed_total = (
                    q_stacker.processed_files_count +
                    q_stacker.failed_align_count +
                    q_stacker.skipped_files_count
                )
                total_estimated = q_stacker.files_in_queue
                successfully_stacked = q_stacker.processed_files_count
                if self.global_start_time and successfully_stacked > 1:
                    elapsed = time.monotonic() - self.global_start_time
                    self.time_per_image = elapsed / successfully_stacked
                    remaining_estimated = max(0, total_estimated - processed_total)
                    if self.time_per_image > 0 and remaining_estimated > 0:
                        eta_seconds = remaining_estimated * self.time_per_image
                        h, rem = divmod(int(eta_seconds), 3600)
                        m, s = divmod(rem, 60)
                        try:
                            self.remaining_time_var.set(f"{h:02}:{m:02}:{s:02}")
                        except tk.TclError:
                            break
                    elif remaining_estimated == 0 and total_estimated > 0:
                        try:
                            self.remaining_time_var.set("00:00:00")
                        except tk.TclError:
                            break
                    else:
                        try:
                            self.remaining_time_var.set("--:--:--")
                        except tk.TclError:
                            break
                else:
                    try:
                        self.remaining_time_var.set("--:--:--")
                    except tk.TclError:
                        break
                default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
                try:
                    self.aligned_files_var.set(default_aligned_fmt.format(count=self.queued_stacker.aligned_files_count))
                except tk.TclError:
                    break
                self.update_remaining_files()
                time.sleep(1.0)
            except Exception as e:
                print(f"Error in GUI progress tracker thread: {e}")
                traceback.print_exc(limit=2)
                try:
                    self.root.after(0, self._processing_finished)
                except tk.TclError:
                    pass
                break

    def update_remaining_files(self):
        """Met à jour l'affichage des fichiers restants/total."""
        if hasattr(self, "queued_stacker") and self.processing:
            processed_total = (
                self.queued_stacker.processed_files_count +
                self.queued_stacker.failed_align_count +
                self.queued_stacker.skipped_files_count
            )
            total = self.queued_stacker.files_in_queue
            remaining = max(0, total - processed_total)
            try:
                self.remaining_files_var.set(f"{remaining}/{total}")
            except tk.TclError:
                pass
        elif not self.processing:
            try:
                self.remaining_files_var.set(self.tr("no_files_waiting"))
            except tk.TclError:
                pass

    def update_additional_folders_display(self):
        """Met à jour l'affichage du nombre de dossiers supplémentaires."""
        count = 0
        if hasattr(self, 'queued_stacker'):
            with self.queued_stacker.folders_lock:
                count = len(self.queued_stacker.additional_folders)
        try:
            if count == 0:
                self.additional_folders_var.set(self.tr('no_additional_folders'))
            elif count == 1:
                self.additional_folders_var.set(self.tr('1 additional folder'))
            else:
                self.additional_folders_var.set(
                    self.tr('{count} additional folders', default="{count} add. folders").format(count=count)
                )
        except tk.TclError:
            pass

    def stop_processing(self):
        """Requests the QueuedStacker to stop processing."""
        if self.processing and hasattr(self, "queued_stacker") and self.queued_stacker.is_running():
            self.update_progress_gui(self.tr("stacking_stopping"), None)
            self.queued_stacker.stop()
            if hasattr(self, 'stop_button'):
                self.stop_button.config(state=tk.DISABLED)
        elif self.processing:
            self.update_progress_gui("Tentative d'arrêt, mais le worker ne tourne pas.", None)
            self._processing_finished()

    def _processing_finished(self):
        """Actions performed in the main GUI thread after processing ends/stops."""
        self.processing = False
        if hasattr(self, 'progress_manager'):
            self.progress_manager.stop_timer()
        final_message = self.tr("stacking_finished")
        final_progress = 100
        final_stack_path = None
        error_message = None
        stacked_count = 0
        if hasattr(self, "queued_stacker"):
            q_stacker = self.queued_stacker
            final_stack_path = q_stacker.final_stacked_path
            aligned_count = q_stacker.aligned_files_count
            failed_count = q_stacker.failed_align_count
            skipped_count = q_stacker.skipped_files_count
            stacked_count = q_stacker.current_stack_count
            default_aligned_fmt = self.tr("aligned_files_label_format", default="Aligned: {count}")
            try:
                self.aligned_files_var.set(default_aligned_fmt.format(count=aligned_count))
            except tk.TclError:
                pass
            if q_stacker.stop_processing:
                final_message = self.tr("processing_stopped")
                if hasattr(self,'progress_bar'):
                    final_progress = self.progress_bar['value']
            elif q_stacker.processing_error:
                error_message = f"{self.tr('stacking_error_msg')}\n{q_stacker.processing_error}"
                final_message = error_message
                if hasattr(self,'progress_bar'):
                    final_progress = self.progress_bar['value']
            elif stacked_count == 0:
                final_message = self.tr("no_stacks_created")
            final_message += f"\n(Empilés: {stacked_count}, Alignés: {aligned_count}, Échecs: {failed_count}, Ignorés: {skipped_count})"
        else:
            final_message = "Erreur: Stacker non initialisé."
            final_progress = 0
        self.update_progress_gui(final_message, final_progress)
        if hasattr(self, 'status_text'):
            try:
                self.status_text.config(state=tk.NORMAL)
                self.status_text.insert(tk.END, "--- Traitement Terminé ---\n")
                self.status_text.see(tk.END)
                self.status_text.config(state=tk.DISABLED)
            except tk.TclError:
                pass
        self._set_parameter_widgets_state(tk.NORMAL)
        if hasattr(self, 'start_button'):
            self.start_button.config(state=tk.NORMAL)
        if hasattr(self, 'stop_button'):
            self.stop_button.config(state=tk.DISABLED)
        if hasattr(self, 'add_files_button'):
            self.add_files_button.config(state=tk.DISABLED)
        if hasattr(self, 'remaining_time_var'):
            try:
                self.remaining_time_var.set("00:00:00")
            except tk.TclError:
                pass
        if error_message:
            messagebox.showerror(self.tr("error"), error_message)
        elif final_stack_path and os.path.exists(final_stack_path):
            messagebox.showinfo(self.tr("info"), f"{self.tr('stacking_complete_msg')}\n{final_stack_path}")
            try:
                self.update_progress_gui(f"Chargement aperçu final: {os.path.basename(final_stack_path)}...", None)
                final_image_data = load_and_validate_fits(final_stack_path)
                if final_image_data is not None:
                    final_header = fits.getheader(final_stack_path)
                    self.current_preview_data = final_image_data
                    self.current_stack_header = final_header
                    self.refresh_preview()
                    if final_header:
                        self.update_image_info(final_header)
                else:
                    raise ValueError("Impossible de charger le fichier stack final pour l'aperçu.")
            except Exception as e:
                self.update_progress_gui(f"⚠️ {self.tr('Error loading final stack preview')}: {e}", None)
                traceback.print_exc(limit=2)
                messagebox.showerror(self.tr("error"), f"{self.tr('Error loading final preview')}:\n{e}")
        elif hasattr(self, 'queued_stacker') and not self.queued_stacker.stop_processing and stacked_count == 0:
            messagebox.showwarning(self.tr("warning"), self.tr("no_stacks_created"))

    def _set_parameter_widgets_state(self, state):
        """Enable/disable control widgets."""
        widgets_to_toggle = []
        if hasattr(self, 'input_entry'): widgets_to_toggle.append(self.input_entry)
        if hasattr(self, 'browse_input_button'): widgets_to_toggle.append(self.browse_input_button)
        if hasattr(self, 'output_entry'): widgets_to_toggle.append(self.output_entry)
        if hasattr(self, 'browse_output_button'): widgets_to_toggle.append(self.browse_output_button)
        if hasattr(self, 'ref_entry'): widgets_to_toggle.append(self.ref_entry)
        if hasattr(self, 'browse_ref_button'): widgets_to_toggle.append(self.browse_ref_button)
        if hasattr(self, 'stacking_combo'): widgets_to_toggle.append(self.stacking_combo)
        if hasattr(self, 'kappa_spinbox'): widgets_to_toggle.append(self.kappa_spinbox)
        if hasattr(self, 'batch_spinbox'): widgets_to_toggle.append(self.batch_spinbox)
        if hasattr(self, 'hot_pixels_check'): widgets_to_toggle.append(self.hot_pixels_check)
        if hasattr(self, 'hp_thresh_spinbox'): widgets_to_toggle.append(self.hp_thresh_spinbox)
        if hasattr(self, 'hp_neigh_spinbox'): widgets_to_toggle.append(self.hp_neigh_spinbox)
        if hasattr(self, 'cleanup_temp_check'): widgets_to_toggle.append(self.cleanup_temp_check)
        if hasattr(self, 'language_combo'): widgets_to_toggle.append(self.language_combo)
        if hasattr(self, 'wb_r_ctrls'): widgets_to_toggle.extend([self.wb_r_ctrls['slider'], self.wb_r_ctrls['spinbox']])
        if hasattr(self, 'wb_g_ctrls'): widgets_to_toggle.extend([self.wb_g_ctrls['slider'], self.wb_g_ctrls['spinbox']])
        if hasattr(self, 'wb_b_ctrls'): widgets_to_toggle.extend([self.wb_b_ctrls['slider'], self.wb_b_ctrls['spinbox']])
        if hasattr(self, 'auto_wb_button'): widgets_to_toggle.append(self.auto_wb_button)
        if hasattr(self, 'reset_wb_button'): widgets_to_toggle.append(self.reset_wb_button)
        if hasattr(self, 'stretch_combo'): widgets_to_toggle.append(self.stretch_combo)
        if hasattr(self, 'stretch_bp_ctrls'): widgets_to_toggle.extend([self.stretch_bp_ctrls['slider'], self.stretch_bp_ctrls['spinbox']])
        if hasattr(self, 'stretch_wp_ctrls'): widgets_to_toggle.extend([self.stretch_wp_ctrls['slider'], self.stretch_wp_ctrls['spinbox']])
        if hasattr(self, 'stretch_gamma_ctrls'): widgets_to_toggle.extend([self.stretch_gamma_ctrls['slider'], self.stretch_gamma_ctrls['spinbox']])
        if hasattr(self, 'auto_stretch_button'): widgets_to_toggle.append(self.auto_stretch_button)
        if hasattr(self, 'reset_stretch_button'): widgets_to_toggle.append(self.reset_stretch_button)
        for widget in widgets_to_toggle:
            if widget and hasattr(widget, 'winfo_exists') and widget.winfo_exists():
                try:
                    widget.config(state=state)
                except tk.TclError:
                    pass # Ignore errors for widgets like Scale

    def _debounce_resize(self, event=None):
         """Debounces window resize events to avoid excessive redraws."""
         if self._after_id_resize:
              try:
                  self.root.after_cancel(self._after_id_resize)
              except tk.TclError:
                  pass # Ignore if root destroyed
         try:
              # Schedule redraw after 300ms of no resize events
              self._after_id_resize = self.root.after(300, self._refresh_preview_on_resize)
         except tk.TclError:
             pass # Ignore if root destroyed

    def _refresh_preview_on_resize(self):
         """Callback function called after resize debounce timeout."""
         if hasattr(self, 'preview_manager'):
              self.preview_manager.trigger_redraw()
         if hasattr(self, 'histogram_widget') and self.histogram_widget.winfo_exists():
              try:
                  self.histogram_widget.canvas.draw_idle()
              except tk.TclError:
                  pass

    def _on_closing(self):
        """Handles window closing event."""
        if self.processing:
            if messagebox.askokcancel(self.tr("quit"), self.tr("quit_while_processing")):
                print("Arrêt demandé via fermeture fenêtre...")
                self.stop_processing()
                if self.thread and self.thread.is_alive():
                     self.thread.join(timeout=1.5) # Wait briefly
                if self.thread and self.thread.is_alive():
                     print("Warning: Worker thread did not exit cleanly.")
                try:
                     if self.root.winfo_exists():
                         self.settings.window_geometry = self.root.geometry() # Save window size/pos
                except tk.TclError:
                    pass # Ignore if window already gone
                self.settings.save_settings()
                print("Fermeture de l'application.")
                self.root.destroy()
            else:
                return # User cancelled quit
        else:
            try:
                 if self.root.winfo_exists():
                     self.settings.window_geometry = self.root.geometry() # Save window size/pos
            except tk.TclError:
                pass
            self.settings.save_settings()
            print("Fermeture de l'application.")
            self.root.destroy()

    def update_progress_gui(self, message, progress=None):
        """Handles progress messages from the backend thread."""
        if isinstance(message, str):
            if message.startswith("folder_count_update:"):
                 self.update_additional_folders_display() # Update UI folder count
                 return # Don't display this message in status text
            elif message.startswith("stats_update:"):
                 return # Ignore raw stats messages for now

        if hasattr(self, "progress_manager") and self.progress_manager:
            self.progress_manager.update_progress(message, progress)

        self.update_remaining_files()

# --- Main Execution ---
if __name__ == "__main__":
    try:
        _dummy_root = tk.Tk()
        _dummy_root.withdraw()
        style = ttk.Style()
        available_themes = style.theme_names()
    except tk.TclError as theme_err:
        print(f"Error initializing Tk themes: {theme_err}")
        available_themes=[]

    theme_to_use = 'default' # Fallback
    preferred_themes = ['clam', 'alt', 'vista', 'xpnative'] # Order of preference
    for t in preferred_themes:
        if t in available_themes:
            theme_to_use = t
            break

    try:
        print(f"Using theme: {theme_to_use}")
        style.theme_use(theme_to_use)
        try:
             style.configure('Accent.TButton', font=('Segoe UI', 9, 'bold'), foreground='white', background='#0078D7') # Example blue
        except tk.TclError:
             print("Warning: Could not configure Accent.TButton style.")
        try:
             style.configure('Toolbutton.TButton', padding=1, font=('Segoe UI', 8)) # Smaller button
        except tk.TclError:
             print("Warning: Could not configure Toolbutton.TButton style.")
    except tk.TclError:
        print(f"Theme '{theme_to_use}' not found or failed to apply, using Tk default.")

    if '_dummy_root' in locals() and _dummy_root.winfo_exists():
        _dummy_root.destroy() # Clean up dummy

    gui = SeestarStackerGUI()
    gui.root.mainloop()
# --- END OF FILE seestar/gui/main_window.py ---