# --- START OF FILE seestar/gui/mosaic_gui.py (Avec options alignement et FastAligner) ---
import tkinter as tk
from tkinter import ttk, messagebox
# import traceback # Décommentez si besoin pour le debug
import numpy as np 

# VALID_DRIZZLE_KERNELS est déjà défini dans votre fichier, je le garde.
VALID_DRIZZLE_KERNELS = ['square', 'turbo', 'point', 'gaussian', 'lanczos2', 'lanczos3'] 

class MosaicSettingsWindow(tk.Toplevel):
    def __init__(self, parent_gui):
        print("DEBUG (MosaicSettingsWindow __init__ V4 - Full Options): Initialisation...")
        if not hasattr(parent_gui, 'root') or not parent_gui.root.winfo_exists():
             raise ValueError("Parent GUI invalide pour MosaicSettingsWindow")

        super().__init__(parent_gui.root)
        self.parent_gui = parent_gui
        self.withdraw() 

        # Récupérer l'état et les settings
        initial_mosaic_state = getattr(self.parent_gui, 'mosaic_mode_active', False)
        
        # Obtenir les valeurs par défaut complètes de SettingsManager pour mosaic_settings
        # Ceci garantit que toutes les clés attendues par l'UI ont une valeur par défaut
        sm_defaults = self.parent_gui.settings.get_default_values() 
        default_mosaic_settings_from_sm = sm_defaults.get('mosaic_settings', {})
        
        # Obtenir les settings actuels du parent (qui peuvent être vides ou partiels)
        current_parent_mosaic_settings = getattr(self.parent_gui.settings, 'mosaic_settings', {})
        if not isinstance(current_parent_mosaic_settings, dict): 
            current_parent_mosaic_settings = {}
        
        # Fusionner : commencer avec les défauts du SM, puis écraser avec les settings actuels du parent
        self.settings = default_mosaic_settings_from_sm.copy()
        self.settings.update(current_parent_mosaic_settings) 
        print(f"DEBUG (MosaicSettingsWindow __init__): self.settings initialisé: {self.settings}")

        # --- Variables Tkinter locales ---
        self.local_mosaic_active_var = tk.BooleanVar(value=self.settings.get('enabled', initial_mosaic_state))
        
        self.local_mosaic_align_mode_var = tk.StringVar(value=self.settings.get('alignment_mode')) 

        initial_api_key = getattr(self.parent_gui.settings, 'astrometry_api_key', '') # Lire clé API globale
        self.local_api_key_var = tk.StringVar(value=initial_api_key) 

        # Options Drizzle (Kernel, Pixfrac, Fillval, WHT Threshold)
        self.local_drizzle_kernel_var = tk.StringVar(value=self.settings.get('kernel'))
        self.local_drizzle_pixfrac_var = tk.DoubleVar(value=self.settings.get('pixfrac'))
        self.local_drizzle_fillval_var = tk.StringVar(value=self.settings.get('fillval'))
        
        initial_wht_storage = self.settings.get('wht_threshold') # Valeur 0-1
        self.local_drizzle_wht_thresh_storage_var = tk.DoubleVar(value=initial_wht_storage)
        self.local_drizzle_wht_thresh_display_var = tk.StringVar(value=f"{initial_wht_storage * 100.0:.0f}")
        self.local_drizzle_wht_thresh_display_var.trace_add("write", self._convert_wht_display_to_storage)

        # Paramètres FastAligner
        self.local_fastalign_orb_features_var = tk.IntVar(value=self.settings.get('fastalign_orb_features'))
        self.local_fastalign_min_abs_matches_var = tk.IntVar(value=self.settings.get('fastalign_min_abs_matches'))
        self.local_fastalign_min_ransac_var = tk.IntVar(value=self.settings.get('fastalign_min_ransac')) 
        self.local_fastalign_ransac_thresh_var = tk.DoubleVar(value=self.settings.get('fastalign_ransac_thresh'))

        self.title(self.parent_gui.tr("mosaic_settings_title", default="Mosaic Options"))
        self.transient(parent_gui.root)
        
        self._build_ui()
        self._update_options_state() 
        
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.minsize(500, 620) 
        self.update_idletasks()
        
        # Centrage (simplifié pour la lisibilité)
        self.master.update_idletasks()
        x = self.master.winfo_rootx() + (self.master.winfo_width() - self.winfo_reqwidth()) // 2
        y = self.master.winfo_rooty() + (self.master.winfo_height() - self.winfo_reqheight()) // 2
        self.geometry(f"+{x}+{y}")
        
        self.deiconify()
        self.focus_force()
        self.grab_set()
        # self.wait_window(self) # Retiré pour debug, peut être remis
        print("DEBUG (MosaicSettingsWindow __init__ V4): Fin initialisation.")

    def _build_ui(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # --- 1. Section Activation (Toujours en haut) ---
        activation_frame = ttk.LabelFrame(main_frame, text=self.parent_gui.tr("mosaic_activation_frame", default="Activation"), padding="5")
        activation_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        self.activate_check = ttk.Checkbutton(activation_frame, text=self.parent_gui.tr("enable_mosaic_mode_label", default="Enable Mosaic Processing Mode"),
                                            variable=self.local_mosaic_active_var, command=self._update_options_state)
        self.activate_check.pack(anchor=tk.W, padx=5, pady=5)

        # --- 2. Cadre Mode d'Alignement (Toujours packé, état géré) ---
        self.alignment_mode_frame = ttk.LabelFrame(main_frame, text=self.parent_gui.tr("mosaic_alignment_method_frame", default="Mosaic Alignment Method"), padding="5")
        self.alignment_mode_frame.pack(fill=tk.X, pady=5, padx=5) 
        ttk.Radiobutton(self.alignment_mode_frame, text=self.parent_gui.tr("mosaic_align_local_fast_fallback", default="Fast Local + WCS Fallback (Recommended)"),
                        variable=self.local_mosaic_align_mode_var, value="local_fast_fallback", command=self._on_alignment_mode_change).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(self.alignment_mode_frame, text=self.parent_gui.tr("mosaic_align_local_fast_only", default="Fast Local Only (Strict)"),
                        variable=self.local_mosaic_align_mode_var, value="local_fast_only", command=self._on_alignment_mode_change).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(self.alignment_mode_frame, text=self.parent_gui.tr("mosaic_align_astrometry_per_panel", default="Astrometry.net per Panel (Slower)"),
                        variable=self.local_mosaic_align_mode_var, value="astrometry_per_panel", command=self._on_alignment_mode_change).pack(anchor=tk.W, padx=5, pady=2)

        # --- 3. Cadre Options FastAligner (Créé, packé/dépacké conditionnellement) ---
        self.fastaligner_options_frame = ttk.LabelFrame(main_frame, text=self.parent_gui.tr("fastaligner_tuning_frame", default="FastAligner Tuning (for Local Alignment)"), padding="5")
        # NE PAS PACKER ICI, _update_options_state s'en charge
        fa_params_config = [
            (self.parent_gui.tr("fa_orb_features_label", default="ORB Features:"), self.local_fastalign_orb_features_var, 1000, 8000, 500, "%d"),
            (self.parent_gui.tr("fa_min_abs_matches_label", default="Min Abs. Matches:"), self.local_fastalign_min_abs_matches_var, 5, 50, 1, "%d"),
            (self.parent_gui.tr("fa_min_ransac_inliers_label", default="Min RANSAC Inliers:"), self.local_fastalign_min_ransac_var, 2, 10, 1, "%d"),
            (self.parent_gui.tr("fa_ransac_thresh_label", default="RANSAC Thresh (px):"), self.local_fastalign_ransac_thresh_var, 1.0, 15.0, 0.5, "%.1f")
        ]
        for label_text, var, from_val, to_val, incr_val, fmt_str in fa_params_config:
            param_frame = ttk.Frame(self.fastaligner_options_frame)
            param_frame.pack(fill=tk.X, pady=2)
            ttk.Label(param_frame, text=label_text, width=22).pack(side=tk.LEFT, padx=2)
            ttk.Spinbox(param_frame, textvariable=var, from_=from_val, to=to_val, increment=incr_val, width=7, format=fmt_str, justify=tk.RIGHT).pack(side=tk.LEFT, padx=2)

        # --- 4. Cadre Clé API (Toujours packé, état géré) ---
        self.api_key_frame = ttk.LabelFrame(main_frame, text=self.parent_gui.tr("mosaic_api_key_frame", default="Astrometry.net API Key (Required for Mosaic)"), padding="5")
        self.api_key_frame.pack(fill=tk.X, pady=5, padx=5)
        api_key_inner_frame = ttk.Frame(self.api_key_frame, padding=5); api_key_inner_frame.pack(fill=tk.X)
        api_key_label = ttk.Label(api_key_inner_frame, text=self.parent_gui.tr("mosaic_api_key_label", default="API Key:"), width=10); api_key_label.pack(side=tk.LEFT, padx=(0,5))
        self.api_key_entry = ttk.Entry(api_key_inner_frame, textvariable=self.local_api_key_var, show="*", width=40); self.api_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        api_help_label = ttk.Label(self.api_key_frame, text=self.parent_gui.tr("mosaic_api_key_help", default="Get your key from nova.astrometry.net (free account)"), foreground="gray", font=("Arial", 8)); api_help_label.pack(anchor=tk.W, padx=10, pady=(0,5))
        
        # --- 5. Cadre Options Drizzle Mosaïque (Toujours packé, état géré) ---
        self.drizzle_options_frame = ttk.LabelFrame(main_frame, text=self.parent_gui.tr("mosaic_drizzle_options_frame", default="Mosaic Drizzle Options"), padding="5")
        self.drizzle_options_frame.pack(fill=tk.X, pady=5, padx=5)
        # Contenu Drizzle
        kernel_frame = ttk.Frame(self.drizzle_options_frame, padding=5); kernel_frame.pack(fill=tk.X)
        ttk.Label(kernel_frame, text=self.parent_gui.tr("mosaic_drizzle_kernel_label", default="Kernel:"), width=15).pack(side=tk.LEFT, padx=(0,5))
        self.kernel_combo = ttk.Combobox(kernel_frame, textvariable=self.local_drizzle_kernel_var, values=VALID_DRIZZLE_KERNELS, state="readonly", width=12); self.kernel_combo.pack(side=tk.LEFT, padx=5)
        pixfrac_frame = ttk.Frame(self.drizzle_options_frame, padding=5); pixfrac_frame.pack(fill=tk.X)
        ttk.Label(pixfrac_frame, text=self.parent_gui.tr("mosaic_drizzle_pixfrac_label", default="Pixfrac:"), width=15).pack(side=tk.LEFT, padx=(0,5))
        self.pixfrac_spinbox = ttk.Spinbox(pixfrac_frame, from_=0.01, to=1.00, increment=0.05, textvariable=self.local_drizzle_pixfrac_var, width=7, justify=tk.RIGHT, format="%.2f"); self.pixfrac_spinbox.pack(side=tk.LEFT, padx=5)
        fillval_frame = ttk.Frame(self.drizzle_options_frame, padding=5); fillval_frame.pack(fill=tk.X)
        ttk.Label(fillval_frame, text=self.parent_gui.tr("mosaic_drizzle_fillval_label", default="Fill Value:"), width=15).pack(side=tk.LEFT, padx=(0,5))
        self.fillval_entry = ttk.Entry(fillval_frame, textvariable=self.local_drizzle_fillval_var, width=7, justify=tk.RIGHT); self.fillval_entry.pack(side=tk.LEFT, padx=5)
        wht_frame = ttk.Frame(self.drizzle_options_frame, padding=5); wht_frame.pack(fill=tk.X)
        ttk.Label(wht_frame, text=self.parent_gui.tr("mosaic_drizzle_wht_thresh_label", default="Low WHT Mask (%):"), width=15).pack(side=tk.LEFT, padx=(0,5))
        self.wht_spinbox = ttk.Spinbox(wht_frame, from_=0, to=100, increment=1, textvariable=self.local_drizzle_wht_thresh_display_var, width=7, justify=tk.RIGHT, format="%.0f"); self.wht_spinbox.pack(side=tk.LEFT, padx=5)

        # --- 6. Boutons OK / Annuler (Toujours en bas) ---
        button_frame = ttk.Frame(main_frame, padding="5")
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10,0))
        self.cancel_button = ttk.Button(button_frame, text=self.parent_gui.tr("cancel_button", default="Cancel"), command=self._on_cancel)
        self.cancel_button.pack(side=tk.RIGHT, padx=(5, 0))
        self.ok_button = ttk.Button(button_frame, text=self.parent_gui.tr("ok_button", default="OK"), command=self._on_ok)
        self.ok_button.pack(side=tk.RIGHT)

    def _on_alignment_mode_change(self):
        self._update_options_state()

    def _update_options_state(self):
        print(f"DEBUG (MosaicSettingsWindow _update_options_state V4): Exécution...")
        is_mosaic_enabled = self.local_mosaic_active_var.get()
        current_align_mode = self.local_mosaic_align_mode_var.get()
        
        # Fonction pour activer/désactiver les enfants d'un frame
        def toggle_frame_children_state(frame_widget, state_to_set):
            if hasattr(frame_widget, 'winfo_children'):
                for child in frame_widget.winfo_children():
                    if isinstance(child, ttk.Frame): # Gérer les frames enfants
                        toggle_frame_children_state(child, state_to_set)
                    elif hasattr(child, 'config'):
                        try:
                            # Comboboxes utilisent 'disabled' ou 'readonly'/'normal'
                            if isinstance(child, ttk.Combobox):
                                child.config(state='readonly' if state_to_set == tk.NORMAL else 'disabled')
                            else:
                                child.config(state=state_to_set)
                        except tk.TclError: pass # Ignorer pour les labels qui n'ont pas 'state'

        # État des cadres principaux (sauf FastAligner)
        main_frames_state = tk.NORMAL if is_mosaic_enabled else tk.DISABLED
        toggle_frame_children_state(self.alignment_mode_frame, main_frames_state)
        toggle_frame_children_state(self.api_key_frame, main_frames_state)
        toggle_frame_children_state(self.drizzle_options_frame, main_frames_state)
        
        # Visibilité et état du cadre FastAligner
        show_fa_opts = is_mosaic_enabled and current_align_mode in ["local_fast_fallback", "local_fast_only"]
        if hasattr(self, 'fastaligner_options_frame'):
            if show_fa_opts:
                if not self.fastaligner_options_frame.winfo_ismapped():
                    # Packer juste avant api_key_frame
                    self.fastaligner_options_frame.pack(fill=tk.X, pady=5, padx=5, before=self.api_key_frame)
                toggle_frame_children_state(self.fastaligner_options_frame, tk.NORMAL)
            else:
                if self.fastaligner_options_frame.winfo_ismapped():
                    self.fastaligner_options_frame.pack_forget()
                # Même si caché, s'assurer que ses enfants sont conceptuellement désactivés
                toggle_frame_children_state(self.fastaligner_options_frame, tk.DISABLED)
        
        if hasattr(self.parent_gui, 'update_mosaic_button_appearance'):
            self.parent_gui.update_mosaic_button_appearance()
        print(f"DEBUG _update_options_state V4: MosaicEnabled={is_mosaic_enabled}, AlignMode='{current_align_mode}', ShowFAOpts={show_fa_opts}")

    def _convert_wht_display_to_storage(self, *args):
        try:
            display_val_str = self.local_drizzle_wht_thresh_display_var.get()
            if not display_val_str: self.local_drizzle_wht_thresh_storage_var.set(0.01); return
            percent_val = float(display_val_str)
            float_val = np.clip(percent_val / 100.0, 0.0, 1.0) 
            self.local_drizzle_wht_thresh_storage_var.set(round(float_val, 3))
        except (ValueError, tk.TclError): self.local_drizzle_wht_thresh_storage_var.set(0.01)

    def _on_ok(self):
        new_mosaic_state = self.local_mosaic_active_var.get()
        selected_align_mode = self.local_mosaic_align_mode_var.get()
        api_key_value = self.local_api_key_var.get().strip()
        selected_kernel = self.local_drizzle_kernel_var.get()
        selected_pixfrac = self.local_drizzle_pixfrac_var.get()
        selected_fillval = self.local_drizzle_fillval_var.get()
        selected_wht_thresh_storage = self.local_drizzle_wht_thresh_storage_var.get()

        fa_orb_features = self.local_fastalign_orb_features_var.get()
        fa_min_abs_matches = self.local_fastalign_min_abs_matches_var.get()
        fa_min_ransac = self.local_fastalign_min_ransac_var.get()
        fa_ransac_thresh = self.local_fastalign_ransac_thresh_var.get()

        # Validation (comme avant)
        if new_mosaic_state:
            if selected_kernel not in VALID_DRIZZLE_KERNELS: messagebox.showerror(self.parent_gui.tr("error"), self.parent_gui.tr("mosaic_invalid_kernel"), parent=self); return
            if not (0.01 <= selected_pixfrac <= 1.0): messagebox.showerror(self.parent_gui.tr("error"), self.parent_gui.tr("mosaic_invalid_pixfrac"), parent=self); return
            if not (0.0 <= selected_wht_thresh_storage <= 1.0): messagebox.showerror(self.parent_gui.tr("error"), "Invalid Mosaic WHT Threshold (internal error, should be 0-1).", parent=self); return
            if not api_key_value: messagebox.showerror(self.parent_gui.tr("error"), self.parent_gui.tr("mosaic_api_key_required"), parent=self); return
            if selected_align_mode in ["local_fast_fallback", "local_fast_only"]: # Validation des params FA
                if not (500 <= fa_orb_features <= 10000): messagebox.showerror("Error", "ORB Features must be between 500 and 10000.", parent=self); return
                if not (3 <= fa_min_abs_matches <= 50): messagebox.showerror("Error", "Min Absolute Matches must be between 3 and 50.", parent=self); return
                if not (2 <= fa_min_ransac <= 20): messagebox.showerror("Error", "Min RANSAC Inliers must be between 2 and 20.", parent=self); return
                if not (1.0 <= fa_ransac_thresh <= 15.0): messagebox.showerror("Error", "RANSAC Threshold must be between 1.0 and 15.0.", parent=self); return

        # Sauvegarde (comme avant)
        if not hasattr(self.parent_gui.settings, 'mosaic_settings') or \
           not isinstance(getattr(self.parent_gui.settings, 'mosaic_settings'), dict):
            self.parent_gui.settings.mosaic_settings = {}

        self.parent_gui.settings.mosaic_settings['enabled'] = new_mosaic_state
        self.parent_gui.settings.mosaic_settings['alignment_mode'] = selected_align_mode
        self.parent_gui.settings.mosaic_settings['kernel'] = selected_kernel
        self.parent_gui.settings.mosaic_settings['pixfrac'] = selected_pixfrac
        self.parent_gui.settings.mosaic_settings['fillval'] = selected_fillval
        self.parent_gui.settings.mosaic_settings['wht_threshold'] = selected_wht_thresh_storage
        self.parent_gui.settings.mosaic_settings['fastalign_orb_features'] = fa_orb_features
        self.parent_gui.settings.mosaic_settings['fastalign_min_abs_matches'] = fa_min_abs_matches
        self.parent_gui.settings.mosaic_settings['fastalign_min_ransac'] = fa_min_ransac
        self.parent_gui.settings.mosaic_settings['fastalign_ransac_thresh'] = fa_ransac_thresh
        
        self.parent_gui.settings.astrometry_api_key = api_key_value
        if hasattr(self.parent_gui, 'astrometry_api_key_var'):
            self.parent_gui.astrometry_api_key_var.set(api_key_value)
        
        self.parent_gui.mosaic_mode_active = new_mosaic_state
        
        print(f"DEBUG (MosaicSettingsWindow _on_ok V4): Settings sauvegardés -> {self.parent_gui.settings.mosaic_settings}")
        if hasattr(self.parent_gui, 'update_mosaic_button_appearance'):
            self.parent_gui.update_mosaic_button_appearance()
        
        self.grab_release()
        self.destroy()

    def _on_cancel(self):
        self.grab_release()
        self.destroy()
        if hasattr(self.parent_gui, 'update_mosaic_button_appearance'):
            self.parent_gui.update_mosaic_button_appearance()

# --- FIN DE LA CLASSE MosaicSettingsWindow ---