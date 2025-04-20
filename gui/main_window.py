"""
Module principal pour l'interface graphique de GSeestar.
Intègre l'optimisation d'espace disque de Seestar avec l'interface améliorée de GSeestar.
Version modifiée pour permettre l'ajout de dossiers pendant le traitement.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import numpy as np
from astropy.io import fits
import traceback
import math # <-- ADDED for time formatting

# Seestar imports
from seestar.core.alignment import SeestarAligner
from seestar.core.image_processing import load_and_validate_fits, debayer_image
from seestar.localization import Localization
from seestar.queuep.queue_manager import SeestarQueuedStacker
from .file_handling import FileHandlingManager
from .preview import PreviewManager
from .progress import ProgressManager
from .settings import SettingsManager


class SeestarStackerGUI:
    """
    GUI principale pour GSeestar avec optimisation d'espace disque.
    """

    # ... (rest of __init__, init_variables, init_managers, show_initial_preview, tr remain the same) ...
    def __init__(self):
        """Initialise l'interface graphique de GSeestar."""
        self.root = tk.Tk()
        self.localization = Localization("en")
        self.settings = SettingsManager()
        self.aligner = SeestarAligner()
        self.queued_stacker = SeestarQueuedStacker()
        self.processing = False
        self.thread = None
        self.current_stack_data = None
        self.current_stack_header = None
        self.time_per_image = 0
        self.global_start_time = None

        self.init_variables()
        self.settings.load_settings()
        self.localization.set_language(self.settings.language)
        self.create_layout()
        self.init_managers()
        self.settings.apply_to_ui(self)
        self.update_ui_language()

        self.queued_stacker.set_progress_callback(self.update_progress_gui)
        self.queued_stacker.set_preview_callback(self.update_preview)

        self.root.title(self.tr("title"))
        self.root.geometry("1200x720")
        self.root.minsize(1000, 600)
        self.root.bind("<Configure>", self.on_window_resize)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def init_variables(self):
        """Initialise les variables Tkinter pour les widgets."""
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.reference_image_path = tk.StringVar()
        self.remaining_files_var = tk.StringVar(value=self.tr("no_files_waiting"))
        self.stacking_mode = tk.StringVar()
        self.kappa = tk.DoubleVar()
        self.batch_size = tk.IntVar()
        self.apply_denoise = tk.BooleanVar()
        self.correct_hot_pixels = tk.BooleanVar()
        self.hot_pixel_threshold = tk.DoubleVar()
        self.neighborhood_size = tk.IntVar()
        self.apply_stretch = tk.BooleanVar()
        self.remaining_time_var = tk.StringVar(value="--:--:--")
        self.elapsed_time_var = tk.StringVar(value="00:00:00")
        self.additional_folders_var = tk.StringVar(value=self.tr("no_additional_folders"))
        self.language_var = tk.StringVar()
        self.aligned_files_var = tk.StringVar(value=self.tr("aligned_files_label", default="Aligned: --"))

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
        else:
             print("Warning: Reset zoom button not found or accessible during init_managers")

        self.show_initial_preview()

    def show_initial_preview(self):
        """Displays a test image or welcome message."""
        if not hasattr(self, 'preview_manager'):
            print("Warning: preview_manager not initialized before show_initial_preview")
            return
        test_img = self.preview_manager.create_test_image(400, 300)
        try:
            info_text = self.tr("Select input/output folders.")
            self.preview_manager.update_preview(
                image_data=test_img,
                stack_name=self.tr("Welcome!"),
                apply_stretch=False,
                info_text=info_text
            )
        except Exception as e:
            print(f"Error during initial preview update: {e}")
            traceback.print_exc()

    def tr(self, key, default=None):
        """Shortcut for localization."""
        return self.localization.get(key)

    def create_layout(self):
        """Crée la disposition des widgets de l'interface."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=(0, 10))
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        left_frame.config(width=450); left_frame.pack_propagate(False)

        # Language
        lang_frame = ttk.Frame(left_frame); lang_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(lang_frame, text="Language/Langue:").pack(side=tk.LEFT, padx=(0, 5))
        self.language_combo = ttk.Combobox(lang_frame, textvariable=self.language_var, values=("en", "fr"), width=10, state="readonly")
        self.language_combo.pack(side=tk.LEFT); self.language_combo.bind("<<ComboboxSelected>>", self.change_language)

        # Folders
        self.folders_frame = ttk.LabelFrame(left_frame, text="Folders"); self.folders_frame.pack(fill=tk.X, pady=5)
        in_subframe = ttk.Frame(self.folders_frame); in_subframe.pack(fill=tk.X, padx=5, pady=(5, 2))
        self.input_label = ttk.Label(in_subframe, text="Input"); self.input_label.pack(side=tk.LEFT, anchor="w")
        self.browse_input_button = ttk.Button(in_subframe, text="Browse", command=lambda: self.file_handler.browse_input(), width=8); self.browse_input_button.pack(side=tk.RIGHT)
        self.input_entry = ttk.Entry(in_subframe, textvariable=self.input_path); self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        out_subframe = ttk.Frame(self.folders_frame); out_subframe.pack(fill=tk.X, padx=5, pady=(2, 5))
        self.output_label = ttk.Label(out_subframe, text="Output"); self.output_label.pack(side=tk.LEFT, anchor="w")
        self.browse_output_button = ttk.Button(out_subframe, text="Browse", command=lambda: self.file_handler.browse_output(), width=8); self.browse_output_button.pack(side=tk.RIGHT)
        self.output_entry = ttk.Entry(out_subframe, textvariable=self.output_path); self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))

        # Stacking Options
        self.options_frame = ttk.LabelFrame(left_frame, text="Options"); self.options_frame.pack(fill=tk.X, pady=5)
        method_frame = ttk.Frame(self.options_frame); method_frame.pack(fill=tk.X, padx=5, pady=5)
        self.stacking_method_label = ttk.Label(method_frame, text="Method"); self.stacking_method_label.pack(side=tk.LEFT, padx=(0, 5))
        self.stacking_combo = ttk.Combobox(method_frame, textvariable=self.stacking_mode, values=("mean", "median", "kappa-sigma", "winsorized-sigma"), width=15, state="readonly"); self.stacking_combo.pack(side=tk.LEFT)
        self.kappa_label = ttk.Label(method_frame, text="Kappa"); self.kappa_label.pack(side=tk.LEFT, padx=(10, 2))
        self.kappa_spinbox = ttk.Spinbox(method_frame, from_=1.0, to=5.0, increment=0.1, textvariable=self.kappa, width=5); self.kappa_spinbox.pack(side=tk.LEFT)
        batch_frame = ttk.Frame(self.options_frame); batch_frame.pack(fill=tk.X, padx=5, pady=5)
        self.batch_size_label = ttk.Label(batch_frame, text="Batch"); self.batch_size_label.pack(side=tk.LEFT, padx=(0, 5))
        self.batch_spinbox = ttk.Spinbox(batch_frame, from_=0, to=500, increment=1, textvariable=self.batch_size, width=5); self.batch_spinbox.pack(side=tk.LEFT)
        ttk.Label(batch_frame, text="(0=auto)").pack(side=tk.LEFT, padx=(2, 10))

        # Alignment & Hot Pixels
        self.align_hp_frame = ttk.LabelFrame(left_frame, text="Alignment & Hot Pixels"); self.align_hp_frame.pack(fill=tk.X, pady=5)
        ref_frame = ttk.Frame(self.align_hp_frame); ref_frame.pack(fill=tk.X, padx=5, pady=5)
        self.reference_label = ttk.Label(ref_frame, text="Reference"); self.reference_label.pack(side=tk.LEFT, anchor="w")
        self.browse_ref_button = ttk.Button(ref_frame, text="Browse", command=lambda: self.file_handler.browse_reference(), width=8); self.browse_ref_button.pack(side=tk.RIGHT)
        self.ref_entry = ttk.Entry(ref_frame, textvariable=self.reference_image_path); self.ref_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        hp_frame = ttk.Frame(self.align_hp_frame); hp_frame.pack(fill=tk.X, padx=5, pady=5)
        self.hot_pixels_check = ttk.Checkbutton(hp_frame, text="Correct Hot Pixels", variable=self.correct_hot_pixels); self.hot_pixels_check.pack(side=tk.LEFT, padx=(0, 10))
        self.hot_pixel_threshold_label = ttk.Label(hp_frame, text="Threshold"); self.hot_pixel_threshold_label.pack(side=tk.LEFT)
        self.hp_thresh_spinbox = ttk.Spinbox(hp_frame, from_=1.0, to=10.0, increment=0.1, textvariable=self.hot_pixel_threshold, width=5); self.hp_thresh_spinbox.pack(side=tk.LEFT, padx=5)
        self.neighborhood_size_label = ttk.Label(hp_frame, text="Neighborhood"); self.neighborhood_size_label.pack(side=tk.LEFT)
        self.hp_neigh_spinbox = ttk.Spinbox(hp_frame, from_=3, to=15, increment=2, textvariable=self.neighborhood_size, width=4); self.hp_neigh_spinbox.pack(side=tk.LEFT, padx=5)

        # Final Stack Options
        self.final_opts_frame = ttk.LabelFrame(left_frame, text="Final Stack"); self.final_opts_frame.pack(fill=tk.X, pady=5)
        self.denoise_check = ttk.Checkbutton(self.final_opts_frame, text="Apply Denoise", variable=self.apply_denoise); self.denoise_check.pack(side=tk.LEFT, padx=5, pady=5)

        # Progress Area
        self.progress_frame = ttk.LabelFrame(left_frame, text="Progress"); self.progress_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.progress_bar = ttk.Progressbar(self.progress_frame, maximum=100); self.progress_bar.pack(fill=tk.X, padx=5, pady=(5, 2))
        time_frame = ttk.Frame(self.progress_frame); time_frame.pack(fill=tk.X, padx=5, pady=2)
        self.remaining_time_label = ttk.Label(time_frame, text="ETA:"); self.remaining_time_label.pack(side=tk.LEFT)
        ttk.Label(time_frame, textvariable=self.remaining_time_var, font="Arial 9 bold").pack(side=tk.LEFT, padx=(2, 10))
        self.elapsed_time_label = ttk.Label(time_frame, text="Elapsed:"); self.elapsed_time_label.pack(side=tk.LEFT)
        ttk.Label(time_frame, textvariable=self.elapsed_time_var, font="Arial 9 bold").pack(side=tk.LEFT, padx=2)
        files_info_frame = ttk.Frame(self.progress_frame); files_info_frame.pack(fill=tk.X, padx=5, pady=2)
        self.remaining_static_label = ttk.Label(files_info_frame, text="Remaining:"); self.remaining_static_label.pack(side=tk.LEFT)
        self.remaining_value_label = ttk.Label(files_info_frame, textvariable=self.remaining_files_var, font="Arial 9"); self.remaining_value_label.pack(side=tk.LEFT, padx=2)
        self.additional_value_label = ttk.Label(files_info_frame, textvariable=self.additional_folders_var, font="Arial 9"); self.additional_value_label.pack(side=tk.RIGHT)
        self.additional_static_label = ttk.Label(files_info_frame, text="Additional:"); self.additional_static_label.pack(side=tk.RIGHT, padx=(10, 0))
        status_text_frame = ttk.Frame(self.progress_frame); status_text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(2, 5))
        self.status_text = tk.Text(status_text_frame, height=6, wrap=tk.WORD, bd=0, font="Arial 9");
        self.status_scrollbar = ttk.Scrollbar(status_text_frame, orient="vertical", command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=self.status_scrollbar.set)
        self.status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y); self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Control Buttons
        control_frame = ttk.Frame(left_frame); control_frame.pack(fill=tk.X, pady=(5, 0), side=tk.BOTTOM)
        self.start_button = ttk.Button(control_frame, text="Start", command=self.start_processing); self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED); self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.add_files_button = ttk.Button(control_frame, text="Add Folder", command=lambda: self.file_handler.add_folder(), state=tk.DISABLED); self.add_files_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # --- RIGHT FRAME (Preview) ---
        self.preview_frame = ttk.LabelFrame(right_frame, text="Preview"); self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=0)
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="black", highlightthickness=0); self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Preview Controls and Info Frame (Below Canvas)
        preview_controls_info_frame = ttk.Frame(self.preview_frame)
        preview_controls_info_frame.pack(fill=tk.X, padx=5, pady=(5,0)) # Use pack for this frame

        self.current_stack_label = ttk.Label(preview_controls_info_frame, text="No Stack", anchor=tk.W, font="Arial 10 bold")
        self.current_stack_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,10))
        self.stretch_check = ttk.Checkbutton(preview_controls_info_frame, text="Stretch", variable=self.apply_stretch, command=self.refresh_preview)
        self.stretch_check.pack(side=tk.LEFT, padx=(0,10))
        self.reset_zoom_button = ttk.Button(preview_controls_info_frame, text="Reset Zoom")
        self.reset_zoom_button.pack(side=tk.LEFT, padx=(0,10))
        # Aligned Files Counter <-- ADDED WIDGETS
        # Use self.aligned_files_label for reference if needed for translation
        self.aligned_files_label = ttk.Label(preview_controls_info_frame, textvariable=self.aligned_files_var, font="Arial 9")
        self.aligned_files_label.pack(side=tk.LEFT) # Pack to the left of other controls perhaps? Or right? Let's try left for now.

        # Image Info Text Area (Below Controls frame)
        self.image_info_text = tk.Text(self.preview_frame, height=4, wrap=tk.WORD, state=tk.DISABLED, font="Arial 9", bd=0, relief=tk.FLAT) # Increased height slightly
        self.image_info_text.pack(fill=tk.X, expand=False, padx=5, pady=(5, 5)) # Pack below controls

        self._store_widget_references()

    def _store_widget_references(self):
        """Stores references to widgets that need language updates."""
        self.widgets_to_translate = {
            "Folders": self.folders_frame, "input_folder": self.input_label,
            "browse_input_button": self.browse_input_button, "output_folder": self.output_label,
            "browse_output_button": self.browse_output_button, "options": self.options_frame,
            "stacking_method": self.stacking_method_label, "kappa_value": self.kappa_label,
            "batch_size": self.batch_size_label, "Alignment & Hot Pixels": self.align_hp_frame,
            "reference_image": self.reference_label, "browse_ref_button": self.browse_ref_button,
            "perform_hot_pixels_correction": self.hot_pixels_check,
            "hot_pixel_threshold": self.hot_pixel_threshold_label,
            "neighborhood_size": self.neighborhood_size_label, "Final Stack": self.final_opts_frame,
            "apply_denoise": self.denoise_check, "progress": self.progress_frame,
            "estimated_time": self.remaining_time_label, "elapsed_time": self.elapsed_time_label,
            "Remaining:": self.remaining_static_label, "Additional:": self.additional_static_label,
            "start": self.start_button, "stop": self.stop_button,
            "add_folder_button": self.add_files_button, "preview": self.preview_frame,
            "no_current_stack": self.current_stack_label, "stretch_preview": self.stretch_check,
            "reset_zoom_button": self.reset_zoom_button,
            "aligned_files_label": self.aligned_files_label, # Static text for aligned count
            # Note: aligned_files_label_format is used in code, not directly on a widget text
        }

    # ... (change_language remains the same) ...
    def change_language(self, event=None):
        """Change l'interface à la langue sélectionnée."""
        selected_lang = self.language_var.get()
        if self.localization.language != selected_lang:
            self.localization.set_language(selected_lang)
            self.settings.language = selected_lang
            self.settings.save_settings()
            self.update_ui_language()

    def update_ui_language(self):
        """Met à jour tous les éléments textuels de l'interface."""
        self.root.title(self.tr("title"))

        for key, widget in self.widgets_to_translate.items():
            translation = self.tr(key)
            try:
                if widget and widget.winfo_exists():
                    if key == "aligned_files_label":
                        # Set the initial text using the StringVar with count 0 or '--'
                         count = 0 if self.processing else "--" # Show '--' if not processing
                         initial_text = self.tr("aligned_files_label_format", default="Aligned: {count}").format(count=count)
                         self.aligned_files_var.set(initial_text)
                    elif isinstance(widget, (ttk.Label, ttk.Button, ttk.Checkbutton)):
                        widget.config(text=translation)
                    elif isinstance(widget, ttk.LabelFrame):
                        widget.config(text=translation)
            except tk.TclError: pass
            except AttributeError: pass # Should not happen if refs are correct

        if not self.processing:
            self.remaining_files_var.set(self.tr("no_files_waiting"))
            if hasattr(self, "preview_manager") and self.preview_manager.original_image_data is None:
                if self.current_stack_label and self.current_stack_label.winfo_exists():
                    self.current_stack_label.config(text=self.tr("no_current_stack"))
                try:
                    if self.image_info_text and self.image_info_text.winfo_exists():
                        self.image_info_text.config(state=tk.NORMAL)
                        self.image_info_text.delete(1.0, tk.END)
                        self.image_info_text.insert(tk.END, self.tr('image_info_waiting'))
                        self.image_info_text.config(state=tk.DISABLED)
                except tk.TclError: pass

        self.update_additional_folders_display()

    # ... (on_window_resize, _refresh_preview_on_resize remain the same) ...
    def on_window_resize(self, event=None):
        """Gère le redimensionnement de la fenêtre."""
        if hasattr(self, "_after_id_resize"):
            self.root.after_cancel(self._after_id_resize)
        self._after_id_resize = self.root.after(250, self._refresh_preview_on_resize)

    def _refresh_preview_on_resize(self):
        """Appelle la mise à jour du canvas après redimensionnement."""
        if hasattr(self, "preview_manager") and self.preview_manager.last_displayed_pil_image:
            self.preview_manager._redraw_canvas(self.preview_manager.last_displayed_pil_image)


    def refresh_preview(self):
        """Actualise l'aperçu (e.g., after stretch toggle)."""
        # This method seems okay
        if not hasattr(self, "preview_manager"): return
        current_data = self.current_stack_data
        current_name = self.preview_manager.current_stack_name
        apply_stretch_val = self.apply_stretch.get()
        if current_data is not None:
            self.preview_manager.update_preview(
                image_data=current_data, stack_name=current_name,
                apply_stretch=apply_stretch_val, force_redraw=True,
            )
            if self.current_stack_header: self.update_image_info(self.current_stack_header)
        elif self.input_path.get() and os.path.isdir(self.input_path.get()):
            self._try_show_first_input_image()
        else: self.preview_manager.clear_preview(self.tr("Select input/output folders."))

    def _try_show_first_input_image(self):
        """Tente d'afficher la première image FITS du dossier d'entrée."""
        # This method seems okay
        if not hasattr(self, "preview_manager"): return
        try:
            input_folder = self.input_path.get()
            if not input_folder or not os.path.isdir(input_folder):
                self.preview_manager.clear_preview(self.tr("Input folder not found")); return
            files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith((".fit", ".fits"))])
            if not files: self.preview_manager.clear_preview(self.tr("No FITS files in input")); return
            first_image_path = os.path.join(input_folder, files[0])
            self.update_progress_gui(f"Loading preview: {files[0]}...", None)
            img_data = load_and_validate_fits(first_image_path); header = fits.getheader(first_image_path)
            img_display = img_data
            if img_data.ndim == 2:
                bayer_pattern = header.get("BAYERPAT", self.settings.bayer_pattern)
                valid_bayer_patterns = ["GRBG", "RGGB", "GBRG", "BGGR"]
                if isinstance(bayer_pattern, str) and bayer_pattern.upper() in valid_bayer_patterns:
                    try: img_display = debayer_image(img_data, bayer_pattern.upper())
                    except ValueError as debayer_err: self.update_progress_gui(f"⚠️ {self.tr('Error during debayering')}: {debayer_err}", None)
                else: self.update_progress_gui(f"⚠️ Invalid/missing BAYERPAT '{bayer_pattern}'. Treating as grayscale.", None)
            elif img_data.ndim == 3 and img_data.shape[0] == 3: img_display = np.moveaxis(img_data, 0, -1)
            self.preview_manager.update_preview(
                image_data=img_display, stack_name=f"{self.tr('Preview:')} {os.path.basename(first_image_path)}",
                apply_stretch=self.apply_stretch.get(),
            )
            self.update_image_info(header)
        except FileNotFoundError: self.preview_manager.clear_preview(self.tr("Input folder not found"))
        except ValueError as ve: self.update_progress_gui(f"⚠️ {self.tr('Error loading preview image')}: {ve}", None); self.preview_manager.clear_preview(self.tr("Error loading preview (invalid format)"))
        except Exception as e: self.update_progress_gui(f"⚠️ {self.tr('Error loading preview image')}: {e}", None); traceback.print_exc(); self.preview_manager.clear_preview(self.tr("Error loading preview"))

    def update_preview(self, stack_data=None, stack_header=None, stack_name=None, apply_stretch=None):
        """Met à jour la prévisualisation et les infos associées. Called by QueuedStacker."""
        # This method seems okay
        if apply_stretch is None: apply_stretch = self.apply_stretch.get()
        if stack_data is not None:
            if self.current_stack_data is None or not np.array_equal(stack_data, self.current_stack_data):
                self.current_stack_data = stack_data.copy()
        if stack_header is not None: self.current_stack_header = stack_header
        if self.current_stack_data is None: return
        if hasattr(self, "preview_manager"):
            img_count = self.current_stack_header.get('NIMAGES', self.current_stack_header.get('STACKCNT', '?')) if self.current_stack_header else '?'
            display_name = f"{stack_name or self.tr('Stack')} ({img_count} {self.tr('imgs')})"
            self.preview_manager.update_preview(self.current_stack_data, display_name, apply_stretch)
            if self.current_stack_header: self.update_image_info(self.current_stack_header)

    def update_image_info(self, header):
        """Met à jour la zone de texte d'informations FITS."""
        if not header or not hasattr(self, "image_info_text") or not self.image_info_text.winfo_exists(): return

        info_lines = []
        # Corrected: Added 'TOTEXP' and corresponding translation key
        keys_to_display = {
            "OBJECT": 'Object', "DATE-OBS": 'Date', "EXPTIME": 'Exposure (s)',
            "TOTEXP": 'Total Exp (s)', # <-- ADDED
            "GAIN": 'Gain', "OFFSET": 'Offset', "CCD-TEMP": 'Temp (°C)',
            "NIMAGES": 'Images', "STACKCNT": 'Images', "STACKTYP": 'Method',
            "FILTER": 'Filter', "BAYERPAT": 'Bayer',
        }
        images_displayed = False
        for fits_key, loc_key in keys_to_display.items():
            label = self.tr(loc_key); value = header.get(fits_key)
            if value is None or str(value).strip() == "": continue

            str_value = str(value)
            if fits_key == 'DATE-OBS': str_value = str_value.split('T')[0]
            elif fits_key in ['EXPTIME', 'CCD-TEMP']:
                try: str_value = f"{float(value):.1f}"
                except (ValueError, TypeError): pass
            elif fits_key == 'TOTEXP': # <-- ADDED Formatting for total exposure
                try:
                    total_sec = float(value)
                    if total_sec > 120: # Format as HH:MM:SS if > 2 minutes
                        h, rem = divmod(int(total_sec), 3600)
                        m, s = divmod(rem, 60)
                        str_value = f"{h:02d}h{m:02d}m{s:02d}s ({total_sec:.1f}s)"
                    elif total_sec > 0:
                        str_value = f"{total_sec:.1f}s"
                    else:
                        str_value = "0s" # Handle zero case
                except (ValueError, TypeError): pass # Keep original if not float
            elif fits_key in ['NIMAGES', 'STACKCNT']:
                if images_displayed: continue
                images_displayed = True; label = self.tr('Images')

            info_lines.append(f"{label}: {str_value}")
        info_text = "\n".join(info_lines) if info_lines else self.tr("No image info available")
        try:
            self.image_info_text.config(state=tk.NORMAL); self.image_info_text.delete(1.0, tk.END)
            self.image_info_text.insert(tk.END, info_text); self.image_info_text.config(state=tk.DISABLED)
        except tk.TclError: pass

    def update_progress_gui(self, message, progress=None):
        """Thread-safe method to update progress UI elements."""
        if hasattr(self, "progress_manager") and self.progress_manager:
            self.progress_manager.update_progress(message, progress)
            if self.processing and self.global_start_time and hasattr(self, "queued_stacker"):
                processed = self.queued_stacker.processed_files_count
                if processed > 0:
                    elapsed = time.monotonic() - self.global_start_time
                    self.time_per_image = elapsed / processed
                    self.remaining_time_var.set(self.calculate_remaining_time())
            self.update_remaining_files()
            self.update_additional_folders_display()
            # Update aligned files counter
            if hasattr(self, "queued_stacker"):
                 aligned_count = self.queued_stacker.aligned_files_count
                 aligned_text = self.tr("aligned_files_label_format", default="Aligned: {count}").format(count=aligned_count)
                 self.aligned_files_var.set(aligned_text)

    # ... (update_remaining_files, update_additional_folders_display, calculate_remaining_time unchanged) ...
    def update_remaining_files(self):
        """Met à jour l'affichage du nombre d'images restantes dans la file."""
        if hasattr(self, "queued_stacker"):
            processed = self.queued_stacker.processed_files_count
            total = self.queued_stacker.files_in_queue
            if not self.processing or total == 0: self.remaining_files_var.set(self.tr("no_files_waiting"))
            else:
                remaining = max(0, total - processed)
                self.remaining_files_var.set(f"{remaining} / {total}")
        else: self.remaining_files_var.set(self.tr("no_files_waiting"))

    def update_additional_folders_display(self):
         """ Met à jour l'affichage du nombre de dossiers supplémentaires en attente. """
         if hasattr(self, 'queued_stacker') and hasattr(self.queued_stacker, 'additional_folders'):
              count = len(self.queued_stacker.additional_folders)
              if count == 0: self.additional_folders_var.set(self.tr('no_additional_folders'))
              elif count == 1: self.additional_folders_var.set(self.tr('1 additional folder'))
              else:
                    try: self.additional_folders_var.set(self.tr('{count} additional folders').format(count=count))
                    except KeyError: self.additional_folders_var.set(f"{count} {self.tr('additional folders')}")
         else: self.additional_folders_var.set(self.tr('no_additional_folders'))

    def calculate_remaining_time(self):
        """Calcule le temps restant estimé basé sur la file d'attente."""
        if self.time_per_image <= 0 or not hasattr(self, "queued_stacker"): return "--:--:--"
        processed = self.queued_stacker.processed_files_count
        total = self.queued_stacker.files_in_queue
        remaining = max(0, total - processed)
        if remaining == 0: return "00:00:00"
        estimated_seconds = remaining * self.time_per_image
        h, rem = divmod(int(estimated_seconds), 3600); m, s = divmod(rem, 60)
        return f"{h:02}:{m:02}:{s:02}"


    # ... (start_processing, _track_processing_progress, stop_processing unchanged) ...
    def start_processing(self):
        """Démarre le traitement des images en utilisant le gestionnaire de file d'attente."""
        input_folder = self.input_path.get(); output_folder = self.output_path.get()
        if not input_folder or not output_folder: messagebox.showerror(self.tr("error"), self.tr("select_folders")); return
        if not os.path.isdir(input_folder): messagebox.showerror(self.tr("error"), f"{self.tr('input_folder_invalid')}:\n{input_folder}"); return
        if not os.path.isdir(output_folder):
            try: os.makedirs(output_folder, exist_ok=True); self.update_progress_gui(f"{self.tr('Output folder created')}: {output_folder}", None)
            except Exception as e: messagebox.showerror(self.tr("error"), f"{self.tr('output_folder_invalid')}:\n{output_folder}\n{e}"); return
        try:
            initial_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".fit", ".fits"))]
            if not initial_files: messagebox.showwarning(self.tr("warning"), self.tr("no_fits_found")); return
        except Exception as e: messagebox.showerror(self.tr("error"), f"{self.tr('Error reading input folder')}:\n{e}"); return

        self.processing = True; self.time_per_image = 0; self.global_start_time = time.monotonic()
        self.aligned_files_var.set(self.tr("aligned_files_label_format", default="Aligned: {count}").format(count=0)) # Reset counter display

        self._set_parameter_widgets_state(tk.DISABLED); self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL); self.add_files_button.config(state=tk.NORMAL)
        self.progress_manager.reset(); self.progress_manager.start_timer(); self.remaining_time_var.set("--:--:--")

        self.settings.update_from_ui(self)
        validation_messages = self.settings.validate_settings()
        if validation_messages:
            self.update_progress_gui("⚠️ Settings adjusted:", None)
            for msg in validation_messages: self.update_progress_gui(f"   - {msg}", None)
            self.settings.apply_to_ui(self)

        # Configure stacker
        self.queued_stacker.stacking_mode = self.settings.stacking_mode
        self.queued_stacker.kappa = self.settings.kappa
        self.queued_stacker.batch_size = self.settings.batch_size
        self.queued_stacker.denoise = self.settings.denoise
        self.queued_stacker.correct_hot_pixels = self.settings.correct_hot_pixels
        self.queued_stacker.hot_pixel_threshold = self.settings.hot_pixel_threshold
        self.queued_stacker.neighborhood_size = self.settings.neighborhood_size
        self.queued_stacker.bayer_pattern = self.settings.bayer_pattern

        self.update_progress_gui(self.tr("stacking_start"), 0)
        reference_path = self.reference_image_path.get() or None

        processing_started = self.queued_stacker.start_processing(input_folder, output_folder, reference_path)

        if processing_started:
            self.thread = threading.Thread(target=self._track_processing_progress, daemon=True)
            self.thread.start()
        else:
            self.update_progress_gui("❌ Failed to start processing.", None)
            messagebox.showerror(self.tr("error"), "Failed to start processing.")
            self._processing_finished()

    def _track_processing_progress(self):
        """Thread that periodically checks the status of the QueuedStacker."""
        while self.processing and hasattr(self, "queued_stacker"):
            try:
                if not self.queued_stacker.is_running():
                    self.processing = False; self.progress_manager.stop_timer()
                    self.root.after(0, self._processing_finished)
                    break
                time.sleep(0.5)
            except Exception as e:
                print(f"Error in GUI progress tracker thread: {e}"); traceback.print_exc()
                self.processing = False; self.progress_manager.stop_timer()
                self.root.after(0, self._processing_finished)
                break

    def stop_processing(self):
        """Requests the QueuedStacker to stop processing."""
        if self.processing and hasattr(self, "queued_stacker"):
            self.update_progress_gui(self.tr("stacking_stopping"), None)
            self.queued_stacker.stop()
            self.stop_button.config(state=tk.DISABLED)


    def _processing_finished(self):
        """Actions performed in the main GUI thread after processing ends or stops."""
        # This method seems okay now
        final_message = self.tr("stacking_finished"); final_progress = 100
        final_stack_path = None; error_message = None

        if hasattr(self, "queued_stacker"):
            # Final update for aligned files counter
            aligned_count = self.queued_stacker.aligned_files_count
            aligned_text = self.tr("aligned_files_label_format", default="Aligned: {count}").format(count=aligned_count)
            self.aligned_files_var.set(aligned_text)

            if self.queued_stacker.stop_processing:
                final_message = self.tr("processing_stopped")
                final_progress = self.progress_bar['value'] if hasattr(self,'progress_bar') else 0
            elif self.queued_stacker.processing_error:
                final_message = f"{self.tr('stacking_error_msg')}\n{self.queued_stacker.processing_error}"
                final_progress = self.progress_bar['value'] if hasattr(self,'progress_bar') else 0
                error_message = final_message
            final_stack_path = self.queued_stacker.final_stacked_path

        self.update_progress_gui(final_message, final_progress)
        self._set_parameter_widgets_state(tk.NORMAL)
        self.start_button.config(state=tk.NORMAL); self.stop_button.config(state=tk.DISABLED)
        self.add_files_button.config(state=tk.DISABLED)

        if error_message: messagebox.showerror(self.tr("error"), error_message)
        elif final_stack_path:
            messagebox.showinfo(self.tr("info"), f"{self.tr('stacking_complete_msg')}\n{final_stack_path}")
            try:
                final_image = load_and_validate_fits(final_stack_path)
                final_header = fits.getheader(final_stack_path)
                img_display = final_image
                if final_image.ndim == 3 and final_image.shape[0] == 3: img_display = np.moveaxis(final_image, 0, -1)
                self.update_preview(
                    stack_data=img_display, stack_header=final_header,
                    stack_name=self.tr("Final Stack"), apply_stretch=self.apply_stretch.get(),
                )
            except Exception as e:
                self.update_progress_gui(f"⚠️ {self.tr('Error loading final stack preview')}: {e}", None); traceback.print_exc()
        else:
            if hasattr(self, 'queued_stacker') and not self.queued_stacker.stop_processing:
                messagebox.showwarning(self.tr("warning"), self.tr("no_stacks_created"))

    # ... (_set_parameter_widgets_state, _on_closing, _save_and_destroy remain the same) ...
    def _set_parameter_widgets_state(self, state):
        """Activates/deactivates parameter setting widgets."""
        widgets_to_toggle = [
            self.input_entry, self.browse_input_button, self.output_entry, self.browse_output_button,
            self.stacking_combo, self.kappa_spinbox, self.batch_spinbox,
            self.ref_entry, self.browse_ref_button, self.hot_pixels_check,
            self.hp_thresh_spinbox, self.hp_neigh_spinbox, self.denoise_check,
            self.language_combo,
        ]
        for widget in widgets_to_toggle:
            if widget and hasattr(widget, 'winfo_exists') and widget.winfo_exists():
                try: widget.config(state=state)
                except tk.TclError: pass

    def _on_closing(self):
        """Handles the main window closing event."""
        if self.processing:
            if messagebox.askokcancel(self.tr("quit"), self.tr("quit_while_processing")):
                self.stop_processing(); self.root.after(200, self._save_and_destroy)
            else: return
        else: self._save_and_destroy()

    def _save_and_destroy(self):
        """ Saves settings and destroys the root window. """
        try:
            self.settings.update_from_ui(self); self.settings.save_settings()
        except Exception as e: print(f"Error saving settings on exit: {e}")
        finally:
            if self.root: self.root.destroy()

# Main execution block
if __name__ == "__main__":
    try:
        gui = SeestarStackerGUI()
        gui.root.mainloop()
    except Exception as e:
        print("Fatal error running GUI:")
        traceback.print_exc()