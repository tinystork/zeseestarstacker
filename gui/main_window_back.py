"""
Module principal pour l'interface graphique de Seestar.
Orchestre l'UI, les interactions utilisateur et le traitement.
"""
import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import numpy as np
# import cv2 # No longer needed directly here
from astropy.io import fits
import traceback # For detailed error printing

# Seestar imports
from seestar.core.alignment import SeestarAligner
from seestar.core.image_processing import load_and_validate_fits, debayer_image, save_fits_image, save_preview_image
from seestar.core.utils import estimate_batch_size, apply_denoise
from seestar.localization import Localization
# from seestar.tools.stretch import Stretch # Stretch only used in PreviewManager now

# GUI Helper imports
from .file_handling import FileHandlingManager
from .preview import PreviewManager
from .progress import ProgressManager
from .settings import SettingsManager

class SeestarStackerGUI:
    """
    GUI principale pour l'application Seestar Stacker (Mode Traditionnel Seulement).
    """

    def __init__(self):
        """Initialise l'interface graphique de Seestar Stacker."""
        self.root = tk.Tk()

        # --- Managers and Core Components ---
        self.localization = Localization('en')
        self.settings = SettingsManager()
        self.aligner = SeestarAligner()

        # --- GUI State ---
        self.processing = False
        self.thread = None
        self.current_stack_data = None
        self.current_stack_header = None

        # --- Timing & Progress ---
        self.total_images_count = 0
        self.processed_images_count = 0
        self.time_per_image = 0
        self.global_start_time = None
        self.processing_additional = False

        # --- Additional Folders ---
        self.additional_folders = []
        self.total_additional_counted = set()

        # --- UI Variables ---
        self.init_variables() # Initialize Tkinter variables

        # --- Load Settings ---
        self.settings.load_settings()
        self.localization.set_language(self.settings.language)

        # --- Build UI ---
        self.create_layout()

        # --- Initialize GUI Helper Managers ---
        self.init_managers()

        # --- Apply Loaded Settings to UI ---
        self.settings.apply_to_ui(self)
        self.update_ui_language()

        # --- Final Setup ---
        self.aligner.set_progress_callback(self.update_progress_gui)
        self.root.title(self.tr('title'))
        self.root.geometry("1200x720")
        self.root.minsize(1000, 600)
        self.root.bind("<Configure>", self.on_window_resize)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def init_variables(self):
        """Initialise les variables Tkinter pour les widgets."""
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.reference_image_path = tk.StringVar()
        self.remaining_files_var = tk.StringVar(value=self.tr('no_files_waiting', default="No files waiting"))
        self.stacking_mode = tk.StringVar()
        self.kappa = tk.DoubleVar()
        self.batch_size = tk.IntVar()
        self.remove_aligned = tk.BooleanVar()
        self.apply_denoise = tk.BooleanVar()
        self.correct_hot_pixels = tk.BooleanVar()
        self.hot_pixel_threshold = tk.DoubleVar()
        self.neighborhood_size = tk.IntVar()
        self.apply_stretch = tk.BooleanVar()
        self.remaining_time_var = tk.StringVar(value="--:--:--")
        self.elapsed_time_var = tk.StringVar(value="00:00:00")
        self.additional_folders_var = tk.StringVar(value=self.tr('no_additional_folders', default="None"))
        self.language_var = tk.StringVar()

    def init_managers(self):
        """Initialize managers AFTER widgets are created."""
        self.progress_manager = ProgressManager(
            self.progress_bar, self.status_text,
            self.remaining_time_var, self.elapsed_time_var
        )
        self.preview_manager = PreviewManager(
            self.preview_canvas, self.current_stack_label, self.image_info_text
        )
        self.file_handler = FileHandlingManager(self)

        if hasattr(self, 'reset_zoom_button') and self.reset_zoom_button.winfo_exists():
            self.reset_zoom_button.configure(command=self.preview_manager.reset_zoom)

        self.show_initial_preview()

    def show_initial_preview(self):
         """ Displays a test image or welcome message. """
         if not hasattr(self, 'preview_manager'): return
         test_img = self.preview_manager.create_test_image(400, 300)
         try:
             self.preview_manager.update_preview(
                 image_data=test_img,
                 stack_name=self.tr("Welcome!", default="Welcome!"),
                 apply_stretch=False,
                 info_text=self.tr("Select input/output folders.", default="Select input/output folders.")
             )
         except Exception as e: print(f"Error initial preview: {e}"); traceback.print_exc()

    def tr(self, key, default=None):
        """Shortcut for localization."""
        return self.localization.get(key, default=default)

    # ==========================================================================
    # Layout Creation
    # ==========================================================================
    def create_layout(self):
        """Crée la disposition des widgets de l'interface."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=(0, 10))
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        left_frame.config(width=450); left_frame.pack_propagate(False)

        # --- Language ---
        lang_frame = ttk.Frame(left_frame); lang_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(lang_frame, text="Language/Langue:").pack(side=tk.LEFT, padx=(0, 5))
        self.language_combo = ttk.Combobox(lang_frame, textvariable=self.language_var, values=('en', 'fr'), width=10, state="readonly")
        self.language_combo.pack(side=tk.LEFT); self.language_combo.bind('<<ComboboxSelected>>', self.change_language)

        # --- Folders ---
        folders_frame = ttk.LabelFrame(left_frame, text=self.tr('Folders', default="Folders"))
        folders_frame.pack(fill=tk.X, pady=5)
        in_subframe = ttk.Frame(folders_frame); in_subframe.pack(fill=tk.X, padx=5, pady=(5,2))
        self.input_label = ttk.Label(in_subframe, text=self.tr('input_folder')); self.input_label.pack(side=tk.LEFT, anchor='w')
        self.browse_input_button = ttk.Button(in_subframe, text=self.tr('browse'), command=lambda: self.file_handler.browse_input(), width=8); self.browse_input_button.pack(side=tk.RIGHT)
        self.input_entry = ttk.Entry(in_subframe, textvariable=self.input_path); self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,5))
        out_subframe = ttk.Frame(folders_frame); out_subframe.pack(fill=tk.X, padx=5, pady=(2,5))
        self.output_label = ttk.Label(out_subframe, text=self.tr('output_folder')); self.output_label.pack(side=tk.LEFT, anchor='w')
        self.browse_output_button = ttk.Button(out_subframe, text=self.tr('browse'), command=lambda: self.file_handler.browse_output(), width=8); self.browse_output_button.pack(side=tk.RIGHT)
        self.output_entry = ttk.Entry(out_subframe, textvariable=self.output_path); self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,5))

        # --- Stacking Options ---
        self.options_frame = ttk.LabelFrame(left_frame, text=self.tr('options'))
        self.options_frame.pack(fill=tk.X, pady=5)
        method_frame = ttk.Frame(self.options_frame); method_frame.pack(fill=tk.X, padx=5, pady=5)
        self.stacking_method_label = ttk.Label(method_frame, text=self.tr('stacking_method')); self.stacking_method_label.pack(side=tk.LEFT, padx=(0, 5))
        self.stacking_combo = ttk.Combobox(method_frame, textvariable=self.stacking_mode, values=('mean', 'median', 'kappa-sigma', 'winsorized-sigma'), width=15, state="readonly"); self.stacking_combo.pack(side=tk.LEFT)
        self.kappa_label = ttk.Label(method_frame, text=self.tr('kappa_value')); self.kappa_label.pack(side=tk.LEFT, padx=(10, 2))
        self.kappa_spinbox = ttk.Spinbox(method_frame, from_=1.0, to=5.0, increment=0.1, textvariable=self.kappa, width=5); self.kappa_spinbox.pack(side=tk.LEFT)
        batch_frame = ttk.Frame(self.options_frame); batch_frame.pack(fill=tk.X, padx=5, pady=5)
        self.batch_size_label = ttk.Label(batch_frame, text=self.tr('batch_size')); self.batch_size_label.pack(side=tk.LEFT, padx=(0, 5))
        self.batch_spinbox = ttk.Spinbox(batch_frame, from_=0, to=500, increment=1, textvariable=self.batch_size, width=5); self.batch_spinbox.pack(side=tk.LEFT)
        ttk.Label(batch_frame, text="(0=auto)").pack(side=tk.LEFT, padx=(2,10))
        self.remove_processed_check = ttk.Checkbutton(batch_frame, text=self.tr('remove_processed'), variable=self.remove_aligned); self.remove_processed_check.pack(side=tk.LEFT)

        # --- Removed Processing Mode Selection ---

        # --- Alignment & Hot Pixels ---
        self.align_hp_frame = ttk.LabelFrame(left_frame, text=self.tr('Alignment & Hot Pixels', default='Alignment & Hot Pixels'))
        self.align_hp_frame.pack(fill=tk.X, pady=5)
        ref_frame = ttk.Frame(self.align_hp_frame); ref_frame.pack(fill=tk.X, padx=5, pady=5)
        self.reference_label = ttk.Label(ref_frame, text=self.tr('reference_image')); self.reference_label.pack(side=tk.LEFT, anchor='w')
        self.browse_ref_button = ttk.Button(ref_frame, text=self.tr('browse'), command=lambda: self.file_handler.browse_reference(), width=8); self.browse_ref_button.pack(side=tk.RIGHT)
        self.ref_entry = ttk.Entry(ref_frame, textvariable=self.reference_image_path); self.ref_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,5))
        hp_frame = ttk.Frame(self.align_hp_frame); hp_frame.pack(fill=tk.X, padx=5, pady=5)
        self.hot_pixels_check = ttk.Checkbutton(hp_frame, text=self.tr('perform_hot_pixels_correction'), variable=self.correct_hot_pixels); self.hot_pixels_check.pack(side=tk.LEFT, padx=(0,10))
        self.hot_pixel_threshold_label = ttk.Label(hp_frame, text=self.tr('hot_pixel_threshold')); self.hot_pixel_threshold_label.pack(side=tk.LEFT)
        self.hp_thresh_spinbox = ttk.Spinbox(hp_frame, from_=1.0, to=10.0, increment=0.1, textvariable=self.hot_pixel_threshold, width=5); self.hp_thresh_spinbox.pack(side=tk.LEFT, padx=5)
        self.neighborhood_size_label = ttk.Label(hp_frame, text=self.tr('neighborhood_size')); self.neighborhood_size_label.pack(side=tk.LEFT)
        self.hp_neigh_spinbox = ttk.Spinbox(hp_frame, from_=3, to=15, increment=2, textvariable=self.neighborhood_size, width=4); self.hp_neigh_spinbox.pack(side=tk.LEFT, padx=5)

        # --- Final Stack Options ---
        self.final_opts_frame = ttk.LabelFrame(left_frame, text=self.tr('Final Stack', default='Final Stack'))
        self.final_opts_frame.pack(fill=tk.X, pady=5)
        self.denoise_check = ttk.Checkbutton(self.final_opts_frame, text=self.tr('apply_denoise'), variable=self.apply_denoise); self.denoise_check.pack(side=tk.LEFT, padx=5, pady=5)

        # --- Progress Area ---
        self.progress_frame = ttk.LabelFrame(left_frame, text=self.tr('progress'))
        self.progress_frame.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        self.progress_bar = ttk.Progressbar(self.progress_frame, maximum=100); self.progress_bar.pack(fill=tk.X, padx=5, pady=(5, 2))
        time_frame = ttk.Frame(self.progress_frame); time_frame.pack(fill=tk.X, padx=5, pady=2)
        self.remaining_time_label = ttk.Label(time_frame, text=self.tr('estimated_time')); self.remaining_time_label.pack(side=tk.LEFT)
        ttk.Label(time_frame, textvariable=self.remaining_time_var, font="Arial 9 bold").pack(side=tk.LEFT, padx=(2,10))
        self.elapsed_time_label = ttk.Label(time_frame, text=self.tr('elapsed_time')); self.elapsed_time_label.pack(side=tk.LEFT)
        ttk.Label(time_frame, textvariable=self.elapsed_time_var, font="Arial 9 bold").pack(side=tk.LEFT, padx=2)
        files_info_frame = ttk.Frame(self.progress_frame); files_info_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(files_info_frame, text=self.tr('Remaining:', default="Remaining:")).pack(side=tk.LEFT)
        ttk.Label(files_info_frame, textvariable=self.remaining_files_var, font="Arial 9").pack(side=tk.LEFT, padx=2)
        ttk.Label(files_info_frame, text=self.tr('Additional:', default="Additional:")).pack(side=tk.RIGHT, padx=(10,0))
        ttk.Label(files_info_frame, textvariable=self.additional_folders_var, font="Arial 9").pack(side=tk.RIGHT)
        status_text_frame = ttk.Frame(self.progress_frame); status_text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(2, 5))
        self.status_text = tk.Text(status_text_frame, height=6, wrap=tk.WORD, bd=0, font="Arial 9");
        self.status_scrollbar = ttk.Scrollbar(status_text_frame, orient="vertical", command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=self.status_scrollbar.set)
        self.status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y); self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Control Buttons ---
        control_frame = ttk.Frame(left_frame); control_frame.pack(fill=tk.X, pady=(5,0), side=tk.BOTTOM)
        self.start_button = ttk.Button(control_frame, text=self.tr('start'), command=self.start_processing); self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.stop_button = ttk.Button(control_frame, text=self.tr('stop'), command=self.stop_processing, state=tk.DISABLED); self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.add_files_button = ttk.Button(control_frame, text=self.tr('add_folder_button'), command=lambda: self.file_handler.add_folder(), state=tk.DISABLED); self.add_files_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # --- RIGHT FRAME (Preview) ---
        self.preview_frame = ttk.LabelFrame(right_frame, text=self.tr('preview'))
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=0)
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="black", highlightthickness=0); self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        preview_info_frame = ttk.Frame(self.preview_frame); preview_info_frame.pack(fill=tk.X, padx=5, pady=(0,5))
        self.current_stack_label = ttk.Label(preview_info_frame, text=self.tr('no_current_stack'), anchor=tk.W, font="Arial 10 bold"); self.current_stack_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.stretch_check = ttk.Checkbutton(preview_info_frame, text=self.tr('stretch_preview'), variable=self.apply_stretch, command=self.refresh_preview); self.stretch_check.pack(side=tk.LEFT, padx=10)
        self.reset_zoom_button = ttk.Button(preview_info_frame, text=self.tr('reset_zoom_button'), command=lambda: None); self.reset_zoom_button.pack(side=tk.LEFT) # Command set in init_managers
        self.image_info_text = tk.Text(self.preview_frame, height=3, wrap=tk.WORD, state=tk.DISABLED, font="Arial 9", bd=0, relief=tk.FLAT); self.image_info_text.pack(fill=tk.X, expand=False, padx=5, pady=(0,5))

        self._store_widget_references()


    def _store_widget_references(self):
         """ Stores references to widgets that need language updates. """
         # Simplified: Only store widgets whose text is set by self.tr() during update
         self.widgets_to_translate = {
              'input_label': self.input_label,
              'output_label': self.output_label,
              'browse_input_button': self.browse_input_button,
              'browse_output_button': self.browse_output_button,
              'browse_ref_button': self.browse_ref_button,
              'options_frame': self.options_frame,
              'stacking_method_label': self.stacking_method_label,
              'kappa_label': self.kappa_label,
              'batch_size_label': self.batch_size_label,
              'remove_processed_check': self.remove_processed_check,
              'denoise_check': self.denoise_check,
              'align_hp_frame': self.align_hp_frame,
              'reference_label': self.reference_label,
              'hot_pixels_check': self.hot_pixels_check,
              'hot_pixel_threshold_label': self.hot_pixel_threshold_label,
              'neighborhood_size_label': self.neighborhood_size_label,
              'final_opts_frame': self.final_opts_frame,
              'progress_frame': self.progress_frame,
              'remaining_time_label': self.remaining_time_label,
              'elapsed_time_label': self.elapsed_time_label,
              'start_button': self.start_button,
              'stop_button': self.stop_button,
              'add_files_button': self.add_files_button,
              'preview_frame': self.preview_frame,
              'current_stack_label': self.current_stack_label,
              'stretch_check': self.stretch_check,
              'reset_zoom_button': self.reset_zoom_button
         }

    # ==========================================================================
    # Event Handlers & UI Updates
    # ==========================================================================

    def change_language(self, event=None):
        """Change l'interface à la langue sélectionnée."""
        selected_lang = self.language_var.get()
        if self.localization.current_language != selected_lang:
            self.localization.set_language(selected_lang)
            self.settings.language = selected_lang
            self.update_ui_language()

    def update_ui_language(self):
        """Met à jour tous les éléments de l'interface."""
        self.root.title(self.tr('title'))
        for key, widget in self.widgets_to_translate.items():
             # Use default text from English if key not found in current lang
             default_text = self.localization.translations['en'].get(key, key) # Use key as ultimate fallback
             translation = self.tr(key, default=default_text)
             try:
                  if widget and widget.winfo_exists():
                       # Check widget type to use correct config option
                       if isinstance(widget, (ttk.Label, ttk.Button, ttk.Checkbutton, ttk.LabelFrame)):
                            widget.config(text=translation)
                       # Add other widget types if needed
             except tk.TclError: pass # Ignore errors for destroyed widgets

        # Update dynamic variables/placeholders
        self.remaining_files_var.set(self.tr('no_files_waiting'))
        self.additional_folders_var.set(self.tr('no_additional_folders'))
        if not hasattr(self, 'preview_manager') or self.preview_manager.original_image_data is None:
             self.current_stack_label.config(text=self.tr('no_current_stack'))
             try:
                  if self.image_info_text and self.image_info_text.winfo_exists():
                       self.image_info_text.config(state=tk.NORMAL)
                       self.image_info_text.delete(1.0, tk.END)
                       self.image_info_text.insert(tk.END, self.tr('image_info_waiting'))
                       self.image_info_text.config(state=tk.DISABLED)
             except tk.TclError: pass


    def on_window_resize(self, event=None):
        """Gère le redimensionnement de la fenêtre."""
        if hasattr(self, '_after_id_resize'): self.root.after_cancel(self._after_id_resize)
        self._after_id_resize = self.root.after(250, self._refresh_preview_on_resize)

    def _refresh_preview_on_resize(self):
        """ Appelle la mise à jour du canvas après redimensionnement."""
        if hasattr(self, 'preview_manager') and self.preview_manager.last_displayed_pil_image:
            self.preview_manager._redraw_canvas(self.preview_manager.last_displayed_pil_image)
            self.preview_manager._update_info_text_area()


    def refresh_preview(self):
        """Actualise l'aperçu (e.g., after stretch toggle)."""
        if hasattr(self, 'preview_manager'):
            current_data = self.current_stack_data
            current_name = self.preview_manager.current_stack_name
            if current_data is None and self.input_path.get() and os.path.isdir(self.input_path.get()):
                 self._try_show_first_input_image()
            elif current_data is not None:
                self.preview_manager.update_preview(image_data=current_data, stack_name=current_name, apply_stretch=self.apply_stretch.get(), force_redraw=True)


    def _try_show_first_input_image(self):
         """ Tente d'afficher la première image du dossier d'entrée. """
         # ... (implementation remains the same as previous version) ...
         if not hasattr(self, 'preview_manager'): return
         try:
             input_folder = self.input_path.get()
             files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.fit', '.fits'))])
             if files:
                 first_image_path = os.path.join(input_folder, files[0])
                 img_data, header = load_and_validate_fits(first_image_path, get_header=True)
                 if img_data.ndim == 2:
                     bayer_pattern = header.get('BAYERPAT', self.settings.bayer_pattern)
                     img_data = debayer_image(img_data, bayer_pattern)
                 elif img_data.ndim == 3 and img_data.shape[0] == 3: img_data = np.moveaxis(img_data, 0, -1)
                 self.preview_manager.update_preview(img_data, f"{self.tr('Preview:')} {os.path.basename(first_image_path)}", self.apply_stretch.get())
                 self.update_image_info(header)
             else: self.preview_manager.clear_preview(self.tr('No FITS files in input'))
         except FileNotFoundError: self.preview_manager.clear_preview(self.tr('Input folder not found'))
         except Exception as e:
             self.update_progress_gui(f"⚠️ {self.tr('Error loading preview image')}: {e}", None)
             self.preview_manager.clear_preview(self.tr('Error loading preview'))

    def update_preview(self, stack_data=None, stack_header=None, stack_name=None, apply_stretch=None):
        """ Met à jour la prévisualisation et les infos associées. """
        # ... (implementation remains the same as previous version) ...
        if apply_stretch is None: apply_stretch = self.apply_stretch.get()
        if stack_data is None: stack_data = self.current_stack_data
        if stack_header is None: stack_header = self.current_stack_header
        if stack_data is None: return
        if self.current_stack_data is None or not np.array_equal(stack_data, self.current_stack_data):
             self.current_stack_data = stack_data.copy() if stack_data is not None else None
             self.current_stack_header = stack_header.copy() if stack_header is not None else None
        if hasattr(self, 'preview_manager'):
            enhanced_stack_name = self._enhance_stack_name_for_display(stack_name)
            self.preview_manager.update_preview(self.current_stack_data, enhanced_stack_name, apply_stretch)
            if self.current_stack_header: self.update_image_info(self.current_stack_header)


    def _enhance_stack_name_for_display(self, base_name):
         """ Adds context (like image count) to the stack name. """
         # ... (implementation remains the same as previous version) ...
         if base_name is None: return self.tr('Stack', default='Stack')
         img_count_str = ""
         if self.current_stack_header and 'NIMAGES' in self.current_stack_header:
              try: img_count = int(self.current_stack_header['NIMAGES'])
              except: img_count = '?'
              img_count_str = f" ({img_count} {self.tr('imgs', default='imgs')})"
         return f"{base_name}{img_count_str}"

    def update_image_info(self, header):
        """Met à jour la zone de texte d'informations."""
        # ... (implementation remains the same as previous version) ...
        if not header or not hasattr(self, 'image_info_text'): return
        info_lines = []
        keys = {'OBJECT':'Object','DATE-OBS':'Date','EXPTIME':'Exp (s)','GAIN':'Gain','OFFSET':'Offset',
                'CCD-TEMP':'Temp (°C)','NIMAGES':'Images','STACKTYP':'Method','FILTER':'Filter','BAYERPAT':'Bayer'}
        for k, label_k in keys.items():
            label = self.tr(label_k, default=label_k)
            v = header.get(k)
            if v is not None and str(v).strip() != '':
                s_v = str(v);
                if k == 'DATE-OBS': s_v = s_v.split('T')[0]
                elif k in ['EXPTIME', 'CCD-TEMP']: try: s_v = f"{float(v):.1f}" except: pass
                info_lines.append(f"{label}: {s_v}")
        info_text = "\n".join(info_lines) if info_lines else self.tr('No image info available')
        try:
             if self.image_info_text and self.image_info_text.winfo_exists():
                  self.image_info_text.config(state=tk.NORMAL); self.image_info_text.delete(1.0, tk.END)
                  self.image_info_text.insert(tk.END, info_text); self.image_info_text.config(state=tk.DISABLED)
        except tk.TclError: pass

    def update_progress_gui(self, message, progress=None):
        """ Thread-safe method to update progress UI. """
        # ... (implementation remains the same as previous version) ...
        if hasattr(self, 'progress_manager'):
            self.progress_manager.update_progress(message, progress)
            if self.global_start_time and self.processed_images_count > 0:
                elapsed = time.monotonic() - self.global_start_time
                self.time_per_image = elapsed / self.processed_images_count
                self.remaining_time_var.set(self.calculate_remaining_time())
            self.update_remaining_files()


    def update_remaining_files(self):
        """ Met à jour l'affichage du nombre d'images restantes. """
        # ... (implementation remains the same as previous version) ...
        remaining_count = max(0, self.total_images_count - self.processed_images_count)
        if not self.processing and remaining_count == 0: self.remaining_files_var.set(self.tr('no_files_waiting'))
        else: self.remaining_files_var.set(f"{remaining_count} / {self.total_images_count}")

    def calculate_remaining_time(self):
        """ Calcule le temps restant estimé. """
        # ... (implementation remains the same as previous version) ...
        if self.time_per_image <= 0 or self.processed_images_count < 1: return "--:--:--"
        remaining = max(0, self.total_images_count - self.processed_images_count)
        if remaining == 0: return "00:00:00"
        secs = remaining * self.time_per_image; h, rem = divmod(int(secs), 3600); m, s = divmod(rem, 60)
        return f"{h:02}:{m:02}:{s:02}"

    def update_additional_folders_display(self):
         """ Updates the additional folders count display """
         if hasattr(self, 'file_handler'): self.file_handler.update_additional_folders_display()

    # ==========================================================================
    # Processing Control & Threading
    # ==========================================================================

    def start_processing(self):
        """Démarre le traitement des images (Mode Traditionnel Seulement)."""
        # ... (validation logic remains the same as previous version) ...
        input_folder = self.input_path.get(); output_folder = self.output_path.get()
        if not input_folder or not output_folder: messagebox.showerror(self.tr('error'), self.tr('select_folders')); return
        if not os.path.isdir(input_folder): messagebox.showerror(self.tr('error'), f"{self.tr('input_folder_invalid')}:\n{input_folder}"); return
        if not os.path.isdir(output_folder):
             try: os.makedirs(output_folder, exist_ok=True)
             except Exception as e: messagebox.showerror(self.tr('error'), f"{self.tr('output_folder_invalid')}:\n{output_folder}\n{e}"); return
        try:
            initial_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.fit', '.fits'))]
            self.total_images_count = len(initial_files)
        except Exception as e: messagebox.showerror(self.tr('error'), f"{self.tr('Error reading input folder')}:\n{e}"); return
        if self.total_images_count == 0: messagebox.showwarning(self.tr('warning'), self.tr('no_fits_found')); return

        # --- Prepare for Processing ---
        self.processing = True; self.aligner.stop_processing = False
        self.current_stack_data = None; self.current_stack_header = None
        self.additional_folders = []; self.total_additional_counted = set()
        self.processed_images_count = 0; self.time_per_image = 0
        self.global_start_time = time.monotonic()

        # --- Update UI State ---
        self._set_parameter_widgets_state(tk.DISABLED); self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL); self.add_files_button.config(state=tk.NORMAL)
        self.update_additional_folders_display(); self.update_remaining_files()
        self.progress_manager.reset(); self.progress_manager.start_timer(); self.remaining_time_var.set("--:--:--")

        # --- Apply Settings ---
        self.settings.update_from_ui(self)
        validation_messages = self.settings.validate_settings()
        if validation_messages:
             self.update_progress_gui("⚠️ Settings adjusted:", None)
             for msg in validation_messages: self.update_progress_gui(f"   - {msg}", None)
             self.settings.apply_to_ui(self)
        self.settings.configure_aligner(self.aligner)

        # --- Start Thread ---
        self.update_progress_gui(self.tr('stacking_start'), 0)
        self.thread = threading.Thread(target=self._run_processing_thread, args=(input_folder, output_folder), daemon=True)
        self.thread.start()

    def stop_processing(self):
        """Demande l'arrêt du traitement en cours."""
        # ... (implementation remains the same as previous version) ...
        if self.processing and self.thread and self.thread.is_alive():
            self.update_progress_gui(self.tr('stop_requested'), None)
            self.aligner.stop_processing = True
            self.stop_button.config(state=tk.DISABLED)

    def _set_parameter_widgets_state(self, state):
        """ Active ou désactive les widgets de paramètres. """
        # ... (implementation remains the same as previous version) ...
        widgets = [self.language_combo, self.browse_input_button, self.browse_output_button, self.browse_ref_button,
                   self.input_entry, self.output_entry, self.ref_entry, self.stacking_combo, self.kappa_spinbox,
                   self.batch_spinbox, self.remove_processed_check, self.denoise_check, self.hot_pixels_check,
                   self.hp_thresh_spinbox, self.hp_neigh_spinbox, self.stretch_check]
        for w in widgets:
             if w and w.winfo_exists():
                 try: w.config(state=state)
                 except tk.TclError: pass


    def _reset_ui_after_processing(self):
        """ Réinitialise l'état de l'interface (runs via root.after). """
        # ... (implementation remains the same as previous version) ...
        self.processing = False
        if self.start_button and self.start_button.winfo_exists(): self.start_button.config(state=tk.NORMAL)
        if self.stop_button and self.stop_button.winfo_exists(): self.stop_button.config(state=tk.DISABLED)
        if self.add_files_button and self.add_files_button.winfo_exists(): self.add_files_button.config(state=tk.DISABLED)
        if hasattr(self, 'progress_manager'): self.progress_manager.stop_timer()
        self._set_parameter_widgets_state(tk.NORMAL)


    def _on_closing(self):
         """ Handles the window closing event. """
         # ... (implementation remains the same as previous version) ...
         if self.processing:
              if messagebox.askokcancel(self.tr("Quit"), self.tr("Processing is active. Quit anyway?")):
                  self.stop_processing()
                  self.root.after(500, self._save_and_destroy)
              else: return
         else: self._save_and_destroy()

    def _save_and_destroy(self):
         """ Saves settings and destroys the window. """
         # ... (implementation remains the same as previous version) ...
         try: self.settings.update_from_ui(self); self.settings.save_settings()
         except Exception as e: print(f"Error saving settings: {e}")
         finally: self.root.destroy()

    # ==========================================================================
    # Core Processing Logic (Now only traditional mode)
    # ==========================================================================

    def _run_processing_thread(self, input_folder, output_folder):
        """ Point d'entrée pour le thread de traitement (Mode Traditionnel). """
        final_message = f"❌ {self.tr('error')}: {self.tr('Processing ended prematurely')}"
        success = False
        try:
            # === Execute Traditional Stacking Directly ===
            success = self._run_enhanced_traditional_stacking(input_folder, output_folder)

            if self.aligner.stop_processing:
                 final_message = self.tr('processing_stopped')
            elif success:
                 add_success = self._process_additional_folders(output_folder) # Still process added folders
                 if self.aligner.stop_processing:
                      final_message = self.tr('processing_stopped_additional')
                 elif add_success:
                      final_message = self.tr('processing_complete')
                 else:
                      final_message = f"⚠️ {self.tr('Main processing OK, but errors in additional folders')}"
                      success = False
            else:
                 final_message = f"❌ {self.tr('error')}: {self.tr('Main processing failed')}"
            # =============================================

        except Exception as e:
            error_msg = f"❌ {self.tr('error')}: {e}"; traceback.print_exc()
            self.update_progress_gui(error_msg, None); self.update_progress_gui(f"Traceback: {traceback.format_exc(limit=2)}", None)
            final_message = error_msg; success = False
        finally:
             try: self.cleanup_temp_files(output_folder, keep_reference=False)
             except Exception as clean_e: self.update_progress_gui(f"⚠️ {self.tr('Error during cleanup')}: {clean_e}", None)
             final_progress = 100 if success and not self.aligner.stop_processing else (self.progress_bar['value'] if hasattr(self,'progress_bar') else 50)
             self.update_progress_gui(final_message, final_progress)
             self.root.after(100, self._reset_ui_after_processing)


    # --- Include the refactored versions of the processing methods ---
    # (These methods are exactly the same as provided in the previous step,
    # copy them here without changes)
    # --- Start Copy ---
    def _run_enhanced_traditional_stacking(self, input_folder, output_folder):
        """ Exécute le processus d'empilement principal. (Runs in background thread) """
        try:
            aligned_folder = os.path.join(output_folder, "aligned_temp")
            os.makedirs(aligned_folder, exist_ok=True)
            self.cleanup_temp_files(output_folder, keep_reference=False, specific_folder="aligned_temp", only_aligned=True) # Clean only aligned from previous runs

            all_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.fit', '.fits'))])
            if not all_files:
                self.update_progress_gui(f"❌ {self.tr('no_fits_found')}", None)
                return False

            batch_size = self.settings.batch_size # Use validated setting
            if batch_size <= 0: # Auto-estimate if still 0
                 sample_path = os.path.join(input_folder, all_files[0])
                 batch_size = estimate_batch_size(sample_path)
                 self.update_progress_gui(f"🧠 {self.tr('Auto batch size')}: {batch_size}", None)

            total_files = len(all_files)
            self.update_progress_gui(f"🔍 {total_files} {self.tr('images found')}. {self.tr('Batch size')}: {batch_size}.", None)

            # --- Reference Image ---
            self.update_progress_gui(f"⭐ {self.tr('Getting reference image...')}", None)
            self.aligner.stop_processing = False
            ref_folder_path = self.aligner.align_images(input_folder, aligned_folder, create_reference_only=True)
            if self.aligner.stop_processing: return False
            if not ref_folder_path or not self.aligner._generated_ref_path: # Check if aligner set the path
                 self.update_progress_gui(f"❌ {self.tr('Failed to get reference image.')}", None)
                 return False
            reference_image_path = self.aligner._generated_ref_path
            self.update_progress_gui(f"✅ {self.tr('Reference image ready')}: {os.path.basename(reference_image_path)}", None)
            self.root.after(0, self._display_reference_preview, reference_image_path)

            # --- Batch Processing ---
            stack_count = 0
            batch_count = (total_files + batch_size - 1) // batch_size
            original_ref_setting = self.aligner.reference_image_path # Backup user setting
            self.aligner.reference_image_path = reference_image_path # Force use generated ref

            for batch_idx in range(batch_count):
                if self.aligner.stop_processing: break

                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, total_files)
                current_files = all_files[batch_start:batch_end]
                current_progress = (self.processed_images_count / total_files) * 100 if total_files > 0 else 0

                self.update_progress_gui(
                    f"🚀 {self.tr('Processing Batch')} {batch_idx + 1}/{batch_count} ({batch_start+1}-{batch_end}/{total_files})...",
                    current_progress
                )

                # Align Batch
                self.update_progress_gui(f"📐 {self.tr('Aligning Batch')} {batch_idx + 1}...", None)
                self.aligner.stop_processing = False
                self.cleanup_temp_files(output_folder, keep_reference=True, specific_folder="aligned_temp", only_aligned=True)
                align_success = self.aligner.align_images(input_folder, aligned_folder, specific_files=current_files)
                if self.aligner.stop_processing: break
                if not align_success:
                    self.update_progress_gui(f"⚠️ {self.tr('Alignment failed for batch')} {batch_idx+1}", None)
                    self.processed_images_count += len(current_files) # Count skipped
                    continue

                # Stack Batch
                aligned_files_in_batch = [f for f in os.listdir(aligned_folder) if f.startswith('aligned_') and f.endswith('.fit')]
                if not aligned_files_in_batch:
                    self.update_progress_gui(f"⚠️ {self.tr('No aligned images in batch')} {batch_idx + 1}.", None)
                    self.processed_images_count += len(current_files) # Count skipped
                    continue

                self.update_progress_gui(f"🧮 {self.tr('Stacking Batch')} {batch_idx + 1} ({len(aligned_files_in_batch)} {self.tr('images')}...)", None)
                batch_stack_data, batch_stack_header = self._stack_aligned_images(aligned_folder, aligned_files_in_batch)

                num_stacked = int(batch_stack_header.get('NIMAGES', 0)) if batch_stack_header else 0
                self.processed_images_count += num_stacked # Only add successfully stacked count

                if batch_stack_data is None:
                    self.update_progress_gui(f"⚠️ {self.tr('Stacking failed for batch')} {batch_idx + 1}.", None)
                    if self.settings.remove_aligned: self.cleanup_temp_files(output_folder, keep_reference=True, specific_folder="aligned_temp", only_aligned=True)
                    continue

                stack_count += 1

                # Combine with Cumulative Stack
                if self.current_stack_data is None:
                    self.current_stack_data = batch_stack_data
                    self.current_stack_header = batch_stack_header
                    self.update_progress_gui(f"✨ {self.tr('First stack created')} ({num_stacked} {self.tr('images')}).", None)
                else:
                    self._combine_with_current_stack(batch_stack_data, batch_stack_header)
                    # Optionally update progress after combine
                    # img_count_combined = self.current_stack_header.get('NIMAGES', '?')
                    # self.update_progress_gui(f"{self.tr('Cumulative stack updated')} ({img_count_combined} {self.tr('images')}).", None)


                # Update Preview (scheduled)
                stack_data_copy = self.current_stack_data.copy()
                stack_header_copy = self.current_stack_header.copy()
                stack_name = f"{self.tr('Cumulative Stack')} ({batch_idx + 1}/{batch_count} {self.tr('batches', default='batches')})"
                self.root.after(0, self.update_preview, stack_data_copy, stack_header_copy, stack_name)

                # Save Cumulative Stack Regularly
                self._save_cumulative_stack(output_folder)

                # Remove aligned files for this batch if requested
                if self.settings.remove_aligned:
                     self.cleanup_temp_files(output_folder, keep_reference=True, specific_folder="aligned_temp", only_aligned=True)
                     # self.update_progress_gui(f"🧹 {self.tr('Aligned files removed for batch')} {batch_idx + 1}.", None) # Less verbose

            # --- End Batch Loop ---
            self.aligner.reference_image_path = original_ref_setting # Restore user setting

            if self.aligner.stop_processing:
                 self.update_progress_gui(self.tr('processing_stopped'), None)
                 return False

            # Denoise Final Stack if requested
            if self.settings.denoise and self.current_stack_data is not None:
                self._apply_denoise_to_final(output_folder)

            # Save Final Metadata Stack
            if self.current_stack_data is not None:
                self._save_final_metadata_stack(input_folder, output_folder, all_files)

            # Final Report
            if stack_count > 0:
                 final_img_count = self.current_stack_header.get('NIMAGES', '?') if self.current_stack_header else '?'
                 self.update_progress_gui(f"✅ {self.tr('Main processing finished')}: {final_img_count} {self.tr('images in final stack')}.", 100)
                 return True
            else:
                 self.update_progress_gui(f"⚠️ {self.tr('No batches were stacked')}.", None)
                 return False

        except Exception as e:
            self.update_progress_gui(f"❌ {self.tr('Critical error in main processing')}: {e}", None)
            traceback.print_exc()
            self.update_progress_gui(f"Traceback: {traceback.format_exc(limit=2)}", None)
            return False

    def _process_additional_folders(self, output_folder):
        """ Traite les dossiers additionnels. (Runs in background thread) """
        if not self.additional_folders: return True
        self.processing_additional = True
        self.update_progress_gui(f"📂 {self.tr('Processing')} {len(self.additional_folders)} {self.tr('additional folders')}...", None)

        aligned_temp_folder = os.path.join(output_folder, "aligned_temp")
        reference_image_path = os.path.join(aligned_temp_folder, "reference_image.fit")
        if not os.path.exists(reference_image_path):
            self.update_progress_gui(f"❌ {self.tr('Reference image not found for additional folders')}.", None)
            self.processing_additional = False
            return False

        self.update_progress_gui(f"⭐ {self.tr('Using reference')}: {os.path.basename(reference_image_path)}", None)
        original_ref_setting = self.aligner.reference_image_path
        self.aligner.reference_image_path = reference_image_path

        folders_to_process = list(self.additional_folders)
        all_success = True

        for folder_idx, folder in enumerate(folders_to_process):
            if self.aligner.stop_processing: all_success = False; break
            self.update_progress_gui(f"📂 {self.tr('Processing additional folder')} {folder_idx + 1}/{len(folders_to_process)}: {os.path.basename(folder)}", None)

            try:
                all_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.fit', '.fits'))])
                if not all_files:
                    self.update_progress_gui(f"⚠️ {self.tr('No FITS files in')}: {os.path.basename(folder)}. {self.tr('Skipped')}.", None)
                    continue

                self.cleanup_temp_files(output_folder, keep_reference=True, specific_folder="aligned_temp", only_aligned=True)

                batch_size = self.settings.batch_size
                if batch_size <= 0:
                    sample_path = os.path.join(folder, all_files[0])
                    batch_size = estimate_batch_size(sample_path)
                    self.update_progress_gui(f"🧠 {self.tr('Auto batch size')}: {batch_size} for {os.path.basename(folder)}", None)

                total_files_in_folder = len(all_files)
                processed_in_folder = 0
                batch_count = (total_files_in_folder + batch_size - 1) // batch_size
                self.update_progress_gui(f"🔍 {total_files_in_folder} {self.tr('images found')}. {self.tr('Batch size')}: {batch_size}", None)

                for batch_idx in range(batch_count):
                    if self.aligner.stop_processing: all_success = False; break

                    batch_start = batch_idx * batch_size; batch_end = min(batch_start + batch_size, total_files_in_folder)
                    current_files = all_files[batch_start:batch_end]
                    global_progress = (self.processed_images_count / self.total_images_count) * 100 if self.total_images_count > 0 else 0

                    self.update_progress_gui(f"🚀 {self.tr('Folder')} {folder_idx+1}, {self.tr('Batch')} {batch_idx + 1}/{batch_count} ({batch_start+1}-{batch_end}/{total_files_in_folder})...", global_progress)

                    self.aligner.stop_processing = False
                    align_success = self.aligner.align_images(folder, aligned_temp_folder, specific_files=current_files)
                    if self.aligner.stop_processing: all_success = False; break
                    if not align_success:
                         self.update_progress_gui(f"⚠️ {self.tr('Alignment failed for batch')} {batch_idx+1} (folder {folder_idx+1})", None)
                         self.processed_images_count += len(current_files)
                         continue

                    aligned_files_in_batch = [f for f in os.listdir(aligned_temp_folder) if f.startswith('aligned_') and f.endswith('.fit')]
                    if not aligned_files_in_batch:
                         self.update_progress_gui(f"⚠️ {self.tr('No aligned images in batch')} {batch_idx + 1} (folder {folder_idx+1}).", None)
                         self.processed_images_count += len(current_files)
                         continue

                    self.update_progress_gui(f"🧮 {self.tr('Stacking Batch')} {batch_idx + 1} ({len(aligned_files_in_batch)} {self.tr('images')}...)", None)
                    batch_stack_data, batch_stack_header = self._stack_aligned_images(aligned_temp_folder, aligned_files_in_batch)

                    num_stacked = int(batch_stack_header.get('NIMAGES', 0)) if batch_stack_header else 0
                    self.processed_images_count += num_stacked
                    processed_in_folder += num_stacked

                    if batch_stack_data is None:
                         self.update_progress_gui(f"⚠️ {self.tr('Stacking failed for batch')} {batch_idx + 1} (folder {folder_idx+1}).", None)
                         if self.settings.remove_aligned: self.cleanup_temp_files(output_folder, keep_reference=True, specific_folder="aligned_temp", only_aligned=True)
                         continue

                    if self.current_stack_data is None: self.current_stack_data = batch_stack_data; self.current_stack_header = batch_stack_header
                    else: self._combine_with_current_stack(batch_stack_data, batch_stack_header)

                    stack_data_copy = self.current_stack_data.copy(); stack_header_copy = self.current_stack_header.copy()
                    stack_name = f"{self.tr('Cumulative Stack')} ({self.tr('Folder')} {folder_idx+1}, {self.tr('Batch')} {batch_idx+1}/{batch_count})"
                    self.root.after(0, self.update_preview, stack_data_copy, stack_header_copy, stack_name)
                    self._save_cumulative_stack(output_folder)
                    if self.settings.remove_aligned: self.cleanup_temp_files(output_folder, keep_reference=True, specific_folder="aligned_temp", only_aligned=True)

                if not all_success: break
                self.update_progress_gui(f"✅ {self.tr('Folder')} {os.path.basename(folder)} {self.tr('processed')} ({processed_in_folder} {self.tr('images added')})", None)
            except Exception as e:
                 self.update_progress_gui(f"❌ {self.tr('Error processing folder')} {os.path.basename(folder)}: {e}", None); traceback.print_exc(); all_success = False

        self.aligner.reference_image_path = original_ref_setting; self.processing_additional = False
        if all_success and self.settings.denoise and self.current_stack_data is not None: self._apply_denoise_to_final(output_folder)
        if all_success and self.current_stack_data is not None:
             first_folder = self.input_path.get() if not folders_to_process else folders_to_process[0]
             try:
                  files_for_meta = sorted([f for f in os.listdir(first_folder) if f.lower().endswith(('.fit', '.fits'))])
                  if files_for_meta: self._save_final_metadata_stack(first_folder, output_folder, files_for_meta)
             except Exception: pass

        if self.aligner.stop_processing: self.update_progress_gui(self.tr('processing_stopped_additional'), None)
        elif all_success: self.update_progress_gui(f"🏁 {self.tr('Additional folder processing complete')}.", 100)
        else: self.update_progress_gui(f"⚠️ {self.tr('Additional folder processing finished with errors')}.", None)
        return all_success

    def _stack_aligned_images(self, aligned_folder, aligned_files):
        """ Empile les images alignées fournies. """
        # ... (implementation remains the same as previous version) ...
        if not aligned_files: return None, None
        images_data = []; headers = []; first_header = None; common_shape = None; common_dtype = None; is_color = False; loaded_count = 0
        for filename in aligned_files:
            if self.aligner.stop_processing: return None, None
            file_path = os.path.join(aligned_folder, filename)
            try:
                img, hdr = load_and_validate_fits(file_path, get_header=True)
                if common_shape is None:
                    common_shape = img.shape; common_dtype = img.dtype; first_header = hdr
                    is_color = (img.ndim == 3 and img.shape[-1] == 3)
                    if not is_color and img.ndim != 2: raise ValueError(f"Shape {img.shape}")
                elif img.shape != common_shape: continue
                images_data.append(img); headers.append(hdr); loaded_count += 1
            except Exception: pass
        if not images_data: return None, None
        try: image_stack_np = np.stack(images_data, axis=0)
        except ValueError: return None, None
        stacking_mode = self.settings.stacking_mode; kappa = self.settings.kappa; final_stacked_image = None
        with np.errstate(divide='ignore', invalid='ignore'):
             if stacking_mode == "mean": final_stacked_image = np.mean(image_stack_np, axis=0, dtype=common_dtype)
             elif stacking_mode == "median": final_stacked_image = np.median(image_stack_np, axis=0).astype(common_dtype)
             elif stacking_mode in ["kappa-sigma", "winsorized-sigma"]:
                 mean=np.mean(image_stack_np, axis=0, dtype=np.float32); std=np.std(image_stack_np, axis=0, dtype=np.float32)
                 threshold = kappa * std; threshold[std < 1e-6] = np.inf
                 if stacking_mode == "kappa-sigma":
                     mask = np.abs(image_stack_np - mean) <= threshold; count_kept = np.maximum(np.sum(mask, axis=0, dtype=np.int16), 1)
                     sum_image = np.sum(np.where(mask, image_stack_np, 0), axis=0, dtype=np.float32); final_stacked_image = (sum_image / count_kept).astype(common_dtype)
                 elif stacking_mode == "winsorized-sigma":
                     lower, upper = mean - threshold, mean + threshold; clipped = np.clip(image_stack_np, lower, upper); final_stacked_image = np.mean(clipped, axis=0, dtype=common_dtype)
             else: final_stacked_image = np.mean(image_stack_np, axis=0, dtype=common_dtype)
        if final_stacked_image is None: return None, None
        stack_header = fits.Header()
        if first_header:
             keys = ['OBJECT','DATE-OBS','TIME-OBS','TELESCOP','INSTRUME','OBSERVER','EXPTIME','FILTER','GAIN','OFFSET','CCD-TEMP','FOCALLEN','APERTURE','RA','DEC','SITELAT','SITELONG']
             for k in keys:
                 if k in first_header: stack_header[k] = first_header[k]
        stack_header['STACKED'] = (True, 'Image is a stack'); stack_header['NIMAGES'] = (loaded_count, 'Number of images stacked in this batch'); stack_header['STACKTYP'] = (stacking_mode, 'Stacking method used')
        if stacking_mode in ["kappa-sigma", "winsorized-sigma"]: stack_header['KAPPA'] = (kappa, 'Kappa value for sigma clipping')
        stack_header['CREATOR'] = ('Seestar Stacker GUI', 'Software used for stacking')
        if is_color: stack_header['BAYERPAT'] = ('N/A', 'Color image, Bayer pattern removed'); stack_header['CTYPE3'] = ('RGB', 'Color channels')
        elif first_header and 'BAYERPAT' in first_header: stack_header['BAYERPAT'] = first_header['BAYERPAT']
        return final_stacked_image, stack_header

    def _combine_with_current_stack(self, new_stack_data, new_stack_header):
        """ Combine le nouveau stack avec le stack courant. """
        # ... (implementation remains the same as previous version) ...
        try:
            if self.current_stack_data is None: self.current_stack_data = new_stack_data; self.current_stack_header = new_stack_header; return
            if new_stack_data.shape != self.current_stack_data.shape: return
            current_n = int(self.current_stack_header.get('NIMAGES', 1)); new_n = int(new_stack_header.get('NIMAGES', 1)); total_n = current_n + new_n
            if total_n <= 0 : return
            wc = current_n / total_n; wn = new_n / total_n
            combined_f32 = (self.current_stack_data.astype(np.float32) * wc) + (new_stack_data.astype(np.float32) * wn)
            self.current_stack_data = combined_f32.astype(self.current_stack_data.dtype)
            self.current_stack_header['NIMAGES'] = total_n; self.current_stack_header.add_history(f"Combined with stack of {new_n} images.")
        except Exception as e: self.update_progress_gui(f"❌ {self.tr('Error combining stacks')}: {e}", None); traceback.print_exc()

    def cleanup_temp_files(self, output_folder, keep_reference=True, specific_folder=None, only_aligned=False):
        """ Nettoie les fichiers temporaires. """
        # ... (implementation remains the same as previous version) ...
        folders = []; cleaned_count = 0
        if specific_folder: fp = os.path.join(output_folder, specific_folder); (os.path.isdir(fp) and folders.append(fp))
        else: fp_temp = os.path.join(output_folder, "aligned_temp"); (os.path.isdir(fp_temp) and folders.append(fp_temp))
        for fp in folders:
             try:
                 for fname in os.listdir(fp):
                     f_path = os.path.join(fp, fname); remove = False
                     if not os.path.isfile(f_path): continue
                     if keep_reference and fname == "reference_image.fit": continue
                     if only_aligned: remove = fname.startswith('aligned_')
                     else: remove = fname.startswith(('aligned_', 'temp_stack_'))
                     if remove: try: os.remove(f_path); cleaned_count += 1 except Exception: pass
             except Exception: pass
        # if cleaned_count > 0: self.update_progress_gui(f"🧹 {cleaned_count} {self.tr('temporary files removed')}.", None)

    def save_stack_with_original_metadata(self, stacked_image, output_path, original_fits_path):
        """ Sauvegarde le stack final avec métadonnées. """
        # ... (implementation remains the same as previous version) ...
        if stacked_image is None: return
        try:
            hdr = fits.Header()
            if original_fits_path and os.path.exists(original_fits_path):
                try:
                    with fits.open(original_fits_path, mode='readonly') as hdul:
                        orig_hdr=hdul[0].header; keys=['OBJECT','DATE-OBS','TIME-OBS','TELESCOP','INSTRUME','OBSERVER','EXPTIME','FILTER','GAIN','OFFSET','CCD-TEMP','FOCALLEN','APERTURE','RA','DEC','SITELAT','SITELONG']
                        for k in keys:
                            if k in orig_hdr: hdr[k] = orig_hdr[k]
                except Exception: pass
            is_color = (stacked_image.ndim == 3 and stacked_image.shape[-1] == 3)
            if is_color: data = np.moveaxis(stacked_image, -1, 0); hdr['NAXIS'],hdr['NAXIS1'],hdr['NAXIS2'],hdr['NAXIS3']=3,stacked_image.shape[1],stacked_image.shape[0],3; hdr['CTYPE3']='RGB'; hdr['BAYERPAT']='N/A'
            else: data = stacked_image; hdr['NAXIS'],hdr['NAXIS1'],hdr['NAXIS2']=2,stacked_image.shape[1],stacked_image.shape[0]
            dtype=np.uint16; hdr['BITPIX']=16; hdr['BSCALE']=1; hdr['BZERO']=32768
            min_v, max_v = np.min(data), np.max(data); norm = (65535*(data.astype(np.float32)-min_v)/(max_v-min_v)) if max_v>min_v else np.full_like(data,32767); data_final=norm.astype(dtype)
            if self.current_stack_header: hdr['NIMAGES']=self.current_stack_header.get('NIMAGES','?'); hdr['STACKTYP']=self.current_stack_header.get('STACKTYP','?'); (('KAPPA' in self.current_stack_header) and (hdr['KAPPA']:=self.current_stack_header['KAPPA'])); (('DENOISED' in self.current_stack_header) and (hdr['DENOISED']:=self.current_stack_header['DENOISED']))
            hdr['CREATOR']='Seestar Stacker GUI'; hdr.add_history("Final stack saved.")
            save_fits_image(data_final, output_path, hdr, overwrite=True)
        except Exception as e: self.update_progress_gui(f"❌ {self.tr('Error saving final FITS stack')}: {e}", None); traceback.print_exc()

    def _display_reference_preview(self, reference_image_path):
         """ Loads and displays the reference image. """
         # ... (implementation remains the same as previous version) ...
         try:
             img, hdr = load_and_validate_fits(reference_image_path, get_header=True)
             img_disp = np.moveaxis(img, 0, -1).astype(np.float32)
             self.preview_manager.update_preview(img_disp, self.tr('Reference Image'), True)
             self.update_image_info(hdr)
         except Exception: pass

    def _save_cumulative_stack(self, output_folder):
         """ Saves the current cumulative stack. """
         # ... (implementation remains the same as previous version) ...
         if self.current_stack_data is None: return
         try: save_fits_image(self.current_stack_data, os.path.join(output_folder, "stack_cumulative.fit"), self.current_stack_header, True); save_preview_image(self.current_stack_data, os.path.join(output_folder, "stack_cumulative.png"), self.settings.apply_stretch)
         except Exception: pass

    def _apply_denoise_to_final(self, output_folder):
         """ Applies denoising and saves result. """
         # ... (implementation remains the same as previous version) ...
         if self.current_stack_data is None: return
         self.update_progress_gui(f"✨ {self.tr('Applying denoising to final stack')}...", None)
         try:
             denoised = apply_denoise(self.current_stack_data, strength=5); hdr = self.current_stack_header.copy(); hdr['HISTORY']='Denoised'; hdr['DENOISED']=(True,'Denoised')
             f_file=os.path.join(output_folder,"stack_final_denoised.fit"); p_file=os.path.join(output_folder,"stack_final_denoised.png")
             save_fits_image(denoised, f_file, hdr, True); save_preview_image(denoised, p_file, self.settings.apply_stretch)
             s_data=denoised.copy(); s_hdr=hdr.copy(); s_name=f"{self.tr('Final Denoised Stack')} ({hdr.get('NIMAGES','?')})"
             self.root.after(0, self.update_preview, s_data, s_hdr, s_name)
             self.update_progress_gui(f"✅ {self.tr('Final denoised stack saved')}.", None)
         except Exception as e: self.update_progress_gui(f"⚠️ {self.tr('Denoising failed')}: {e}", None)

    def _save_final_metadata_stack(self, input_folder, output_folder, all_input_files):
         """ Saves the final color stack with metadata. """
         # ... (implementation remains the same as previous version) ...
         if self.current_stack_data is None or not all_input_files: return
         ref = os.path.join(input_folder, all_input_files[0])
         out = os.path.join(output_folder, "stack_final_color_metadata.fit")
         self.save_stack_with_original_metadata(self.current_stack_data, out, ref)

    # --- End Copy ---

    # ==========================================================================
    # Main Application Execution
    # ==========================================================================

    def run(self):
        """Lance la boucle principale de l'interface graphique."""
        self.root.mainloop()

# ==========================================================================
# Entry Point
# ==========================================================================
if __name__ == '__main__':
    app = SeestarStackerGUI()
    app.run()