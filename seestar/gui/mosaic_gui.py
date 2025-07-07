import tkinter as tk
from tkinter import ttk, messagebox
# import traceback # Décommentez si besoin pour le debug
import numpy as np 
import os   # Pour les chemins de fichiers
# VALID_DRIZZLE_KERNELS est déjà défini dans votre fichier, je le garde.
VALID_DRIZZLE_KERNELS = ['square', 'turbo', 'point', 'gaussian', 'lanczos2', 'lanczos3'] 



class MosaicSettingsWindow(tk.Toplevel):
    def __init__(self, parent_gui):
        print("DEBUG (MosaicSettingsWindow __init__ V4 - Trad): Initialisation...")
        if not hasattr(parent_gui, 'root') or not parent_gui.root.winfo_exists():
             raise ValueError("Parent GUI invalide pour MosaicSettingsWindow")

        super().__init__(parent_gui.root)
        self.parent_gui = parent_gui # Stocker la référence au GUI parent pour accéder à .tr()
        self.withdraw() 

        # --- 1. Récupération des settings existants et des valeurs par défaut ---
        # (Ce bloc reste inchangé par rapport à sa position)
        initial_mosaic_state = getattr(self.parent_gui, 'mosaic_mode_active', False)
        sm_defaults = self.parent_gui.settings.get_default_values() 
        default_mosaic_settings_from_sm = sm_defaults.get('mosaic_settings', {})
        current_parent_mosaic_settings = getattr(self.parent_gui.settings, 'mosaic_settings', {})
        if not isinstance(current_parent_mosaic_settings, dict): 
            current_parent_mosaic_settings = {} 
        self.settings = default_mosaic_settings_from_sm.copy()
        self.settings.update(current_parent_mosaic_settings) 

        # --- 2. Initialisation des Variables Tkinter locales (DÉPLACÉES EN DÉBUT) ---
        # Toutes ces variables doivent être définies AVANT l'appel à self._build_ui()
        self.local_mosaic_active_var = tk.BooleanVar(value=self.settings.get('enabled', initial_mosaic_state))
        self.local_mosaic_align_mode_var = tk.StringVar(value=self.settings.get('alignment_mode', 'local_fast_fallback'))

        # Ces variables pour les options Drizzle Mosaïque doivent être définies ici
        self.local_drizzle_kernel_var = tk.StringVar(value=self.settings.get('kernel', 'square'))
        self.local_drizzle_pixfrac_var = tk.DoubleVar(value=float(self.settings.get('pixfrac', 0.8)))
        self.local_use_gpu_var = tk.BooleanVar(value=bool(self.settings.get('use_gpu', False)))
        self.local_drizzle_fillval_var = tk.StringVar(value=str(self.settings.get('fillval', '0.0')))
        initial_wht_storage_value = float(self.settings.get('wht_threshold', 0.01))
        self.local_drizzle_wht_thresh_storage_var = tk.DoubleVar(value=initial_wht_storage_value)
        self.local_drizzle_wht_thresh_display_var = tk.StringVar(value=f"{initial_wht_storage_value * 100.0:.0f}")
        # La trace_add doit être après l'initialisation de la variable, mais c'est ok.
        self.local_drizzle_wht_thresh_display_var.trace_add("write", self._convert_wht_display_to_storage)

        # Les variables FastAligner DAO aussi, si elles sont utilisées dans _build_ui
        self.fa_dao_fwhm_var = tk.DoubleVar(value=float(self.settings.get('fastalign_dao_fwhm', 3.5)))
        self.fa_dao_thr_sig_var = tk.DoubleVar(value=float(self.settings.get('fastalign_dao_thr_sig', 4.0)))
        self.fa_dao_max_stars_var = tk.DoubleVar(value=float(self.settings.get('fastalign_dao_max_stars', 750.0)))
        
        # Et les variables FastAligner ORB/RANSAC
        self.local_fastalign_orb_features_var = tk.DoubleVar(
            value=float(self.settings.get('fastalign_orb_features', 3000.0))
        )
        self.local_fastalign_min_abs_matches_var = tk.DoubleVar(
            value=float(self.settings.get('fastalign_min_abs_matches', 8.0))
        )
        self.local_fastalign_min_ransac_var = tk.DoubleVar(
            value=float(self.settings.get('fastalign_min_ransac', 4.0))
        )
        self.local_fastalign_ransac_thresh_var = tk.DoubleVar(
            value=float(self.settings.get('fastalign_ransac_thresh', 2.5))
        )

        # --- Variables pour la configuration des solveurs astrométriques ---
        default_solver_choice = "none"
        if hasattr(self.parent_gui.settings, 'local_solver_preference'):
            default_solver_choice = getattr(
                self.parent_gui.settings, 'local_solver_preference', 'none'
            )
        self.local_solver_choice_var = tk.StringVar(value=default_solver_choice)

        self.astap_path_var = tk.StringVar(
            value=getattr(self.parent_gui.settings, 'astap_path', '')
        )
        self.astap_data_dir_var = tk.StringVar(
            value=getattr(self.parent_gui.settings, 'astap_data_dir', '')
        )
        self.astap_search_radius_var = tk.DoubleVar(
            value=getattr(self.parent_gui.settings, 'astap_search_radius', 30.0)
        )
        self.astap_downsample_var = tk.IntVar(
            value=self.parent_gui.config.get('astap_default_downsample', 2)
        )
        self.astap_sensitivity_var = tk.IntVar(
            value=self.parent_gui.config.get('astap_default_sensitivity', 100)
        )
        self.cluster_threshold_var = tk.DoubleVar(
            value=self.parent_gui.config.get('cluster_panel_threshold', 0.5)
        )
        self.local_ansvr_path_var = tk.StringVar(
            value=getattr(self.parent_gui.settings, 'local_ansvr_path', '')
        )
        # --- FIN INITIALISATION VARIABLES TKINTER LOCALES (DÉPLACÉES) ---

        # --- 3. Configuration de la fenêtre Toplevel ---
        self.title(self.parent_gui.tr("mosaic_settings_title", default="Mosaic Options"))
        self.transient(parent_gui.root) 
        
        # --- 4. Construction de l'Interface Utilisateur (Widgets) ---
        # CET APPEL EST MAINTENANT SÛR, CAR TOUTES LES VARIABLES SONT DÉFINIES
        self._build_ui()
        self._on_solver_choice_change()
        
        # --- 5. Mise à jour initiale de l'état des options ---
        self._update_options_state() 
        
        # --- 6. Gestionnaires d'événements et configuration finale ---
        self.protocol("WM_DELETE_WINDOW", self._on_cancel) 
        self.minsize(500, 620) 
        self.update_idletasks() 
        
        self.master.update_idletasks()
        parent_x = self.master.winfo_rootx()
        parent_y = self.master.winfo_rooty()
        parent_width = self.master.winfo_width()
        parent_height = self.master.winfo_height()
        self_width = self.winfo_reqwidth()
        self_height = self.winfo_reqheight()
        position_x = parent_x + (parent_width // 2) - (self_width // 2)
        position_y = parent_y + (parent_height // 2) - (self_height // 2)
        self.geometry(f"+{position_x}+{position_y}")
        
        self.deiconify() 
        self.focus_force() 
        self.grab_set() 
        
        print("DEBUG (MosaicSettingsWindow __init__ V4 - Trad): Fin initialisation.")




###################################################################################################################################################



# DANS LA CLASSE MosaicSettingsWindow DANS seestar/gui/mosaic_gui.py

    def _build_ui(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # --- 1. Section Activation ---
        activation_frame = ttk.LabelFrame(main_frame, 
                                          text=self.parent_gui.tr("mosaic_activation_frame_title", default="Activation"), 
                                          padding="5")
        activation_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        self.activate_check = ttk.Checkbutton(activation_frame, 
                                            text=self.parent_gui.tr("mosaic_activate_label", default="Enable Mosaic Processing Mode"), # Réutilisation de la clé existante
                                            variable=self.local_mosaic_active_var, command=self._update_options_state)
        self.activate_check.pack(anchor=tk.W, padx=5, pady=5)

        # --- 2. Cadre Mode d'Alignement ---
        self.alignment_mode_frame = ttk.LabelFrame(main_frame, 
                                                 text=self.parent_gui.tr("mosaic_alignment_method_frame_title", default="Mosaic Alignment Method"), 
                                                 padding="5")
        self.alignment_mode_frame.pack(fill=tk.X, pady=5, padx=5) 
        ttk.Radiobutton(self.alignment_mode_frame, 
                        text=self.parent_gui.tr("mosaic_align_local_fast_fallback_label", default="Fast Local + WCS Fallback (Recommended)"),
                        variable=self.local_mosaic_align_mode_var, value="local_fast_fallback", command=self._on_alignment_mode_change).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(self.alignment_mode_frame, 
                        text=self.parent_gui.tr("mosaic_align_local_fast_only_label", default="Fast Local Only (Strict)"),
                        variable=self.local_mosaic_align_mode_var, value="local_fast_only", command=self._on_alignment_mode_change).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(self.alignment_mode_frame,
                        text=self.parent_gui.tr("mosaic_align_astrometry_per_panel_label", default="Astrometry.net per Panel (Slower)"),
                        variable=self.local_mosaic_align_mode_var, value="astrometry_per_panel", command=self._on_alignment_mode_change).pack(anchor=tk.W, padx=5, pady=2)

        # --- Bloc Configuration Astrometry ---
        self.astrometry_config_frame = ttk.LabelFrame(
            main_frame,
            text=self.parent_gui.tr("local_solver_astap_frame_title", default="Astrometry Configuration"),
            padding="10"
        )
        self.astrometry_config_frame.pack(fill=tk.X, padx=5, pady=5)

        solver_choice_frame = ttk.LabelFrame(
            self.astrometry_config_frame,
            text=self.parent_gui.tr("local_solver_choice_frame_title", default="Local Solver Preference"),
            padding="10"
        )
        solver_choice_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Radiobutton(
            solver_choice_frame,
            text=self.parent_gui.tr("local_solver_choice_none", default="Do not use local solvers (use Astrometry.net web service if API key provided)"),
            variable=self.local_solver_choice_var,
            value="none",
            command=self._on_solver_choice_change,
        ).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(
            solver_choice_frame,
            text=self.parent_gui.tr("local_solver_choice_astap", default="Use ASTAP (local)"),
            variable=self.local_solver_choice_var,
            value="astap",
            command=self._on_solver_choice_change,
        ).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(
            solver_choice_frame,
            text=self.parent_gui.tr("local_solver_choice_ansvr", default="Use Astrometry.net Local (solve-field)"),
            variable=self.local_solver_choice_var,
            value="ansvr",
            command=self._on_solver_choice_change,
        ).pack(anchor=tk.W, pady=2)

        api_key_frame = ttk.LabelFrame(
            self.astrometry_config_frame,
            text=self.parent_gui.tr("mosaic_api_key_frame", default="Astrometry.net API Key (Required for Mosaic)"),
            padding="5"
        )
        api_key_frame.pack(fill=tk.X, padx=5, pady=5)

        api_key_inner = ttk.Frame(api_key_frame)
        api_key_inner.pack(fill=tk.X, padx=5, pady=(5, 0))
        ttk.Label(api_key_inner, text=self.parent_gui.tr("mosaic_api_key_label", default="API Key:"), width=10).pack(side=tk.LEFT, padx=(0, 5))
        self.api_key_entry = ttk.Entry(api_key_inner, textvariable=self.parent_gui.astrometry_api_key_var, show="*", width=40)
        self.api_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(
            api_key_frame,
            text=self.parent_gui.tr("mosaic_api_key_help", default="Get your key from nova.astrometry.net (free account)"),
            foreground="gray",
            font=("Arial", 8),
        ).pack(anchor=tk.W, padx=10, pady=(2, 5))

        api_key_help_text_static = self.parent_gui.tr(
            "msw_api_key_help_static",
            default="Used for Astrometry.net web service:\n" "- If 'Astrometry.net per Panel' is chosen and no local solver is active.\n" "- As a final fallback if chosen local solvers fail (for anchor or panels).",
        )
        ttk.Label(api_key_frame, text=api_key_help_text_static, font=("Arial", 8, "italic"), justify=tk.LEFT, wraplength=380).pack(anchor=tk.W, padx=10, pady=(0, 5))

        self.astap_frame = ttk.LabelFrame(
            self.astrometry_config_frame,
            text=self.parent_gui.tr("local_solver_astap_frame_title", default="ASTAP Configuration"),
            padding="10",
        )
        self.astap_frame.pack(fill=tk.X, padx=5, pady=5)

        astap_path_subframe = ttk.Frame(self.astap_frame)
        astap_path_subframe.pack(fill=tk.X, pady=(5, 2))
        ttk.Label(astap_path_subframe, text=self.parent_gui.tr("local_solver_astap_path_label", default="ASTAP Executable Path:"), width=25, anchor="w").pack(side=tk.LEFT)
        ttk.Button(astap_path_subframe, text=self.parent_gui.tr("browse", default="Browse..."), command=self._browse_astap_path, width=10).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Entry(astap_path_subframe, textvariable=self.astap_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        astap_data_subframe = ttk.Frame(self.astap_frame)
        astap_data_subframe.pack(fill=tk.X, pady=(2, 5))
        ttk.Label(astap_data_subframe, text=self.parent_gui.tr("local_solver_astap_data_label", default="ASTAP Star Index Data Directory:"), width=25, anchor="w").pack(side=tk.LEFT)
        ttk.Button(astap_data_subframe, text=self.parent_gui.tr("browse", default="Browse..."), command=self._browse_astap_data_dir, width=10).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Entry(astap_data_subframe, textvariable=self.astap_data_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        astap_radius_subframe = ttk.Frame(self.astap_frame)
        astap_radius_subframe.pack(fill=tk.X, pady=(2, 5))
        ttk.Label(astap_radius_subframe, text=self.parent_gui.tr("local_solver_astap_radius_label", default="ASTAP Search Radius (degrees, 0 for auto):"), width=35, anchor="w").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(
            astap_radius_subframe,
            from_=0.0,
            to=180.0,
            increment=0.5,
            textvariable=self.astap_search_radius_var,
            width=6,
            format="%.2f",
        ).pack(side=tk.LEFT)

        astap_down_subframe = ttk.Frame(self.astap_frame)
        astap_down_subframe.pack(fill=tk.X, pady=(2, 5))
        ttk.Label(astap_down_subframe, text=self.parent_gui.tr("local_solver_astap_downsample_label", default="ASTAP Downsample (-z):"), width=35, anchor="w").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(astap_down_subframe, from_=1, to=8, increment=1, textvariable=self.astap_downsample_var, width=6).pack(side=tk.LEFT)

        astap_sens_subframe = ttk.Frame(self.astap_frame)
        astap_sens_subframe.pack(fill=tk.X, pady=(2, 5))
        ttk.Label(astap_sens_subframe, text=self.parent_gui.tr("local_solver_astap_sens_label", default="ASTAP Sensitivity (-sens):"), width=35, anchor="w").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(astap_sens_subframe, from_=10, to=1000, increment=5, textvariable=self.astap_sensitivity_var, width=6).pack(side=tk.LEFT)

        cluster_thresh_subframe = ttk.Frame(self.astap_frame)
        cluster_thresh_subframe.pack(fill=tk.X, pady=(2, 5))
        ttk.Label(cluster_thresh_subframe, text=self.parent_gui.tr("panel_clustering_threshold_label", default="Panel Clustering Threshold (deg):"), width=35, anchor="w").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(cluster_thresh_subframe, from_=0.01, to=5.0, increment=0.01, textvariable=self.cluster_threshold_var, width=6, format="%.2f").pack(side=tk.LEFT)

        self.ansvr_frame = ttk.LabelFrame(
            self.astrometry_config_frame,
            text=self.parent_gui.tr("local_solver_ansvr_frame_title", default="Local Astrometry.net (solve-field) Configuration"),
            padding="10",
        )
        self.ansvr_frame.pack(fill=tk.X, padx=5, pady=5)

        ansvr_path_entry_frame = ttk.Frame(self.ansvr_frame)
        ansvr_path_entry_frame.pack(fill=tk.X, pady=(5, 2))
        ansvr_path_label = ttk.Label(
            ansvr_path_entry_frame,
            text=self.parent_gui.tr("local_solver_ansvr_main_path_label", default="Path (Exe, .cfg, or Index Dir):"),
            width=25,
            anchor="w",
        )
        ansvr_path_label.pack(side=tk.LEFT, padx=(0, 5))
        self.ansvr_path_entry = ttk.Entry(ansvr_path_entry_frame, textvariable=self.local_ansvr_path_var)
        self.ansvr_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ansvr_buttons_frame = ttk.Frame(self.ansvr_frame)
        ansvr_buttons_frame.pack(fill=tk.X, pady=(0, 5))
        ansvr_buttons_sub_right_frame = ttk.Frame(ansvr_buttons_frame)
        ansvr_buttons_sub_right_frame.pack(side=tk.RIGHT)
        self.ansvr_browse_file_button = ttk.Button(
            ansvr_buttons_sub_right_frame,
            text=self.parent_gui.tr("browse_ansvr_file_button", default="Browse File... (.cfg/Exe)"),
            command=self._browse_ansvr_file,
            width=20,
        )
        self.ansvr_browse_file_button.pack(side=tk.LEFT, padx=(0, 5))
        self.ansvr_browse_dir_button = ttk.Button(
            ansvr_buttons_sub_right_frame,
            text=self.parent_gui.tr("browse_ansvr_dir_button", default="Browse Index Dir..."),
            command=self._browse_ansvr_index_dir,
            width=20,
        )
        self.ansvr_browse_dir_button.pack(side=tk.LEFT)

        info_label = ttk.Label(
            self.astrometry_config_frame,
            text=self.parent_gui.tr(
                "local_solver_info_text_v3",
                default="Select a local solver preference above. Paths are only needed for the selected solver.\n" "For Ansvr, you can point to `solve-field` exe, an `astrometry.cfg` file, or an Index Directory (a .cfg will be auto-generated).",
            ),
            justify=tk.LEFT,
            wraplength=480,
        )
        info_label.pack(pady=(10, 5), padx=5, fill=tk.X)

        # --- 3. Cadre Options FastAligner ---
        self.fastaligner_options_frame = ttk.LabelFrame(main_frame, 
                                                      text=self.parent_gui.tr("fastaligner_tuning_frame_title", default="FastAligner Tuning (for Local Alignment)"), 
                                                      padding="5")
        # Ce cadre est packé conditionnellement par _update_options_state

        fa_params_config = [
            (self.parent_gui.tr("fa_orb_features_label", default="ORB Features:"), 
             self.local_fastalign_orb_features_var, 1000.0, 8000.0, 100.0, "%.0f"), 
            (self.parent_gui.tr("fa_min_abs_matches_label", default="Min Abs. Matches:"), 
             self.local_fastalign_min_abs_matches_var, 1.0, 50.0, 1.0, "%.0f"),
            (self.parent_gui.tr("fa_min_ransac_inliers_label", default="Min RANSAC Inliers:"), 
             self.local_fastalign_min_ransac_var, 1.0, 20.0, 1.0, "%.0f"),
            (self.parent_gui.tr("fa_ransac_thresh_label", default="RANSAC Thresh (px):"), 
             self.local_fastalign_ransac_thresh_var, 1.0, 15.0, 0.1, "%.1f") 
        ]

        for label_text_key, tk_var_instance, from_val, to_val, incr_val, fmt_value in fa_params_config:
            param_frame = ttk.Frame(self.fastaligner_options_frame)
            param_frame.pack(fill=tk.X, pady=2)
            # Le premier élément de la config est maintenant la clé de traduction pour le label
            ttk.Label(param_frame, text=label_text_key, width=22).pack(side=tk.LEFT, padx=2) 
            
            spin_options = {
                "textvariable": tk_var_instance, "from_": from_val, "to": to_val,
                "increment": incr_val, "width": 7, "justify": tk.RIGHT
            }
            if fmt_value: spin_options["format"] = fmt_value
            ttk.Spinbox(param_frame, **spin_options).pack(side=tk.LEFT, padx=2)


        # --- 4. Cadre Options Drizzle Mosaïque ---
        self.drizzle_options_frame = ttk.LabelFrame(main_frame, 
                                                  text=self.parent_gui.tr("mosaic_drizzle_options_frame", default="Mosaic Drizzle Options"), # Clé existante
                                                  padding="5")
        self.drizzle_options_frame.pack(fill=tk.X, pady=5, padx=5)
        
        kernel_frame = ttk.Frame(self.drizzle_options_frame, padding=5); kernel_frame.pack(fill=tk.X)
        ttk.Label(kernel_frame, 
                  text=self.parent_gui.tr("mosaic_drizzle_kernel_label", default="Kernel:"), # Clé existante
                  width=15).pack(side=tk.LEFT, padx=(0,5))
        self.kernel_combo = ttk.Combobox(kernel_frame, textvariable=self.local_drizzle_kernel_var, values=VALID_DRIZZLE_KERNELS, state="readonly", width=12); self.kernel_combo.pack(side=tk.LEFT, padx=5)
        
        pixfrac_frame = ttk.Frame(self.drizzle_options_frame, padding=5); pixfrac_frame.pack(fill=tk.X)
        ttk.Label(pixfrac_frame, 
                  text=self.parent_gui.tr("mosaic_drizzle_pixfrac_label", default="Pixfrac:"), # Clé existante
                  width=15).pack(side=tk.LEFT, padx=(0,5))
        self.pixfrac_spinbox = ttk.Spinbox(pixfrac_frame, from_=0.01, to=2.00, increment=0.05, textvariable=self.local_drizzle_pixfrac_var, width=7, justify=tk.RIGHT, format="%.2f"); self.pixfrac_spinbox.pack(side=tk.LEFT, padx=5)
        self.use_gpu_check = ttk.Checkbutton(
            pixfrac_frame,
            text=self.parent_gui.tr("mosaic_drizzle_use_gpu_label", default="Use GPU"),
            variable=self.local_use_gpu_var,
        )
        self.use_gpu_check.pack(side=tk.LEFT, padx=5)
        
        fillval_frame = ttk.Frame(self.drizzle_options_frame, padding=5); fillval_frame.pack(fill=tk.X)
        ttk.Label(fillval_frame, 
                  text=self.parent_gui.tr("mosaic_drizzle_fillval_label", default="Fill Value:"), # Nouvelle clé
                  width=15).pack(side=tk.LEFT, padx=(0,5))
        self.fillval_entry = ttk.Entry(fillval_frame, textvariable=self.local_drizzle_fillval_var, width=7, justify=tk.RIGHT); self.fillval_entry.pack(side=tk.LEFT, padx=5)
        
        wht_frame = ttk.Frame(self.drizzle_options_frame, padding=5); wht_frame.pack(fill=tk.X)
        ttk.Label(wht_frame, 
                  text=self.parent_gui.tr("mosaic_drizzle_wht_thresh_label", default="Low WHT Mask (%):"), # Nouvelle clé
                  width=15).pack(side=tk.LEFT, padx=(0,5))
        self.wht_spinbox = ttk.Spinbox(wht_frame, from_=0, to=100, increment=1, textvariable=self.local_drizzle_wht_thresh_display_var, width=7, justify=tk.RIGHT, format="%.0f"); self.wht_spinbox.pack(side=tk.LEFT, padx=5)
        # NOUVEAU BLOC : Facteur d'échelle Mosaïque
        mosaic_scale_frame = ttk.Frame(self.drizzle_options_frame, padding=5)
        mosaic_scale_frame.pack(fill=tk.X)
        ttk.Label(mosaic_scale_frame,
                text=self.parent_gui.tr("mosaic_scale_label", default="Scale Factor:"), # Nouvelle clé
                width=15).pack(side=tk.LEFT, padx=(0,5))
        
        # Boutons radio x1, x2, x3, x4
        for scale_val in ["1", "2", "3", "4"]:
            ttk.Radiobutton(mosaic_scale_frame, text=f"x{scale_val}",
                            variable=self.local_mosaic_scale_var, value=scale_val).pack(side=tk.LEFT, padx=3)


        # --- 6. Boutons OK / Annuler ---
        button_frame = ttk.Frame(main_frame, padding="5")
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10,0)) 
        
        self.cancel_button = ttk.Button(button_frame, 
                                       text=self.parent_gui.tr("cancel", default="Cancel"), # Clé globale 'cancel'
                                       command=self._on_cancel)
        self.cancel_button.pack(side=tk.RIGHT, padx=(5, 0))
        self.ok_button = ttk.Button(button_frame, 
                                    text=self.parent_gui.tr("ok", default="OK"), # Clé globale 'ok'
                                    command=self._on_ok)
        self.ok_button.pack(side=tk.RIGHT)

        self.fa_dao_fwhm_var = tk.DoubleVar(value=float(self.settings.get('fastalign_dao_fwhm', 3.5)))
        self.fa_dao_thr_sig_var = tk.DoubleVar(value=float(self.settings.get('fastalign_dao_thr_sig', 4.0))) # Mettre un défaut plus bas
        self.fa_dao_max_stars_var = tk.DoubleVar(value=float(self.settings.get('fastalign_dao_max_stars', 750.0)))
        # --- Ajout des Spinbox pour les paramètres DAO ---




###################################################################################################################################################






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
                    widget_class_name = child.winfo_class() 
                    is_spinbox = "Spinbox" in widget_class_name 
                    
                    # --- LOG POUR SPINBOX (et autres widgets pour contexte) ---
                    # On log l'état appliqué à tous les enfants directs pour voir si les Spinbox reçoivent le bon état
                    parent_name_info = ""
                    try:
                        parent_name_info = f"(child of '{frame_widget.winfo_name()}' - {frame_widget.winfo_class()})"
                    except:
                        parent_name_info = "(child of UNNAMED_FRAME)"
                        
                    print(f"    TOGGLE_FRAME: Setting state of {widget_class_name} {parent_name_info} to: '{state_to_set}'")
                    # --- FIN LOG ---

                    if isinstance(child, ttk.Frame): 
                        toggle_frame_children_state(child, state_to_set) # Appel récursif pour les sous-cadres
                    elif hasattr(child, 'config'):
                        try:
                            if isinstance(child, ttk.Combobox):
                                child.config(state='readonly' if state_to_set == tk.NORMAL else 'disabled')
                            else:
                                child.config(state=state_to_set) # Pour Spinbox, Label, Entry, Button, etc.
                        except tk.TclError: pass 

        # État des cadres principaux (sauf FastAligner)
        main_frames_state = tk.NORMAL if is_mosaic_enabled else tk.DISABLED
        print(f"  _update_options_state: is_mosaic_enabled={is_mosaic_enabled} -> main_frames_state set to: '{main_frames_state}'")
        
        toggle_frame_children_state(self.alignment_mode_frame, main_frames_state)
        toggle_frame_children_state(self.astrometry_config_frame, main_frames_state)
        toggle_frame_children_state(self.drizzle_options_frame, main_frames_state) # Contient pixfrac_spinbox et wht_spinbox
        
        # Visibilité et état du cadre FastAligner
        show_fa_opts = is_mosaic_enabled and current_align_mode in ["local_fast_fallback", "local_fast_only"]
        print(f"  _update_options_state: current_align_mode='{current_align_mode}', show_fa_opts resolved to: {show_fa_opts}")
        
        if hasattr(self, 'fastaligner_options_frame'):
            if show_fa_opts:
                if not self.fastaligner_options_frame.winfo_ismapped():
                    self.fastaligner_options_frame.pack(fill=tk.X, pady=5, padx=5, before=self.drizzle_options_frame)
                print(f"  _update_options_state: Applying state NORMAL to children of fastaligner_options_frame.")
                toggle_frame_children_state(self.fastaligner_options_frame, tk.NORMAL)
            else:
                if self.fastaligner_options_frame.winfo_ismapped():
                    self.fastaligner_options_frame.pack_forget()
                # Même si caché, on met l'état correct (disabled si caché ou si mosaïque désactivée)
                print(f"  _update_options_state: Applying state DISABLED to children of fastaligner_options_frame (because show_fa_opts is False).")
                toggle_frame_children_state(self.fastaligner_options_frame, tk.DISABLED)
        
        if hasattr(self.parent_gui, 'update_mosaic_button_appearance'):
            self.parent_gui.update_mosaic_button_appearance()
        print(f"DEBUG _update_options_state V4: MosaicEnabled={is_mosaic_enabled}, AlignMode='{current_align_mode}', ShowFAOpts={show_fa_opts} --- FIN")





    def _convert_wht_display_to_storage(self, *args):
        try:
            display_val_str = self.local_drizzle_wht_thresh_display_var.get()
            if not display_val_str: self.local_drizzle_wht_thresh_storage_var.set(0.01); return
            percent_val = float(display_val_str)
            float_val = np.clip(percent_val / 100.0, 0.0, 1.0)
            self.local_drizzle_wht_thresh_storage_var.set(round(float_val, 3))
        except (ValueError, tk.TclError): self.local_drizzle_wht_thresh_storage_var.set(0.01)

    def _on_solver_choice_change(self, *args):
        """Met à jour l'état des cadres de configuration selon le solveur choisi."""
        choice = self.local_solver_choice_var.get()
        astap_state = tk.DISABLED
        ansvr_state = tk.DISABLED
        if choice == "astap":
            astap_state = tk.NORMAL
        elif choice == "ansvr":
            ansvr_state = tk.NORMAL
        if hasattr(self, 'astap_frame') and self.astap_frame.winfo_exists():
            for widget in self.astap_frame.winfo_children():
                self._set_widget_state_recursive(widget, astap_state)
        if hasattr(self, 'ansvr_frame') and self.ansvr_frame.winfo_exists():
            for widget in self.ansvr_frame.winfo_children():
                self._set_widget_state_recursive(widget, ansvr_state)

    def _set_widget_state_recursive(self, widget, state):
        try:
            if 'state' in widget.configure():
                widget.configure(state=state)
        except tk.TclError:
            pass
        if hasattr(widget, 'winfo_children'):
            for child in widget.winfo_children():
                self._set_widget_state_recursive(child, state)

    def _browse_astap_path(self):
        initial_dir = ""
        current_path = self.astap_path_var.get()
        if current_path and os.path.exists(os.path.dirname(current_path)):
            initial_dir = os.path.dirname(current_path)
        elif os.path.exists(current_path):
            initial_dir = os.path.dirname(current_path)

        file_types = [(self.parent_gui.tr("executable_files", default="Executable Files"), "*.*")]
        if os.name == 'nt':
            file_types = [
                (self.parent_gui.tr("astap_executable_win", default="ASTAP Executable"), "*.exe"),
                (self.parent_gui.tr("all_files", default="All Files"), "*.*"),
            ]

        filepath = filedialog.askopenfilename(
            title=self.parent_gui.tr("select_astap_executable_title", default="Select ASTAP Executable"),
            initialdir=initial_dir if initial_dir else os.path.expanduser("~"),
            filetypes=file_types,
            parent=self,
        )
        if filepath:
            self.astap_path_var.set(filepath)

    def _browse_astap_data_dir(self):
        initial_dir = self.astap_data_dir_var.get()
        if not initial_dir or not os.path.isdir(initial_dir):
            initial_dir = os.path.expanduser("~")

        dirpath = filedialog.askdirectory(
            title=self.parent_gui.tr("select_astap_data_dir_title", default="Select ASTAP Star Index Data Directory"),
            initialdir=initial_dir,
            parent=self,
        )
        if dirpath:
            self.astap_data_dir_var.set(dirpath)

    def _browse_ansvr_file(self):
        initial_dir_ansvr = os.path.expanduser("~")
        current_path = self.local_ansvr_path_var.get()
        if current_path and os.path.exists(current_path):
            if os.path.isfile(current_path):
                initial_dir_ansvr = os.path.dirname(current_path)
        elif current_path and os.path.exists(os.path.dirname(current_path)):
            initial_dir_ansvr = os.path.dirname(current_path)

        if os.name == 'nt':
            file_types_list = [
                (self.parent_gui.tr("configuration_files", default="Config Files"), "*.cfg"),
                (self.parent_gui.tr("executable_files", default="Executable Files (.exe)"), "*.exe"),
                (self.parent_gui.tr("all_files", default="All Files"), "*.*"),
            ]
        else:
            file_types_list = [
                (self.parent_gui.tr("executable_files", default="Executable Files (any)"), "*"),
                (self.parent_gui.tr("configuration_files", default="Config Files (.cfg)"), "*.cfg"),
                (self.parent_gui.tr("all_files", default="All Files"), "*.*"),
            ]

        filepath = filedialog.askopenfilename(
            title=self.parent_gui.tr("select_ansvr_exe_or_cfg_title_v2", default="Select solve-field Executable or .cfg File"),
            initialdir=initial_dir_ansvr,
            filetypes=file_types_list,
            parent=self,
        )
        if filepath:
            self.local_ansvr_path_var.set(filepath)

    def _browse_ansvr_index_dir(self):
        initial_dir_ansvr = os.path.expanduser("~")
        current_path = self.local_ansvr_path_var.get()
        if current_path and os.path.exists(current_path):
            if os.path.isdir(current_path):
                initial_dir_ansvr = current_path
            elif os.path.isfile(current_path):
                initial_dir_ansvr = os.path.dirname(current_path)
        elif current_path and os.path.exists(os.path.dirname(current_path)):
            initial_dir_ansvr = os.path.dirname(current_path)

        dirpath = filedialog.askdirectory(
            title=self.parent_gui.tr("select_ansvr_index_dir_title_v2", default="Select Astrometry.net Index Directory"),
            initialdir=initial_dir_ansvr,
            parent=self,
        )
        if dirpath:
            self.local_ansvr_path_var.set(dirpath)


#######################################################################################################################################




    def _on_ok(self):
        """
        Valide les entrées, sauvegarde les paramètres dans l'objet settings
        du parent_gui, et ferme la fenêtre.
        Version: Include mosaic_scale_factor in settings_dict_to_save
        """
        print("DEBUG (MosaicSettingsWindow _on_ok V_APIKeyOptional_FixAllLocalVarNames): Bouton OK cliqué.") # DEBUG Version

        is_mosaic_enabled_ui = self.local_mosaic_active_var.get() 
        selected_align_method = self.local_mosaic_align_mode_var.get() 
        try:
            api_key_value = self.parent_gui.astrometry_api_key_var.get().strip()
        except Exception:
            api_key_value = getattr(self.parent_gui.settings, 'astrometry_api_key', '').strip()

        # Initialisation de api_key_potentially_needed à False pour garantir sa définition
        api_key_potentially_needed = False 
        # Ce bloc de code existant est parfait, il va surcharger api_key_potentially_needed à True si les conditions sont remplies.
        if is_mosaic_enabled_ui:
            if selected_align_method == "astrometry_per_panel":
                api_key_potentially_needed = True
            elif selected_align_method == "local_fast_fallback":
                api_key_potentially_needed = True

        if is_mosaic_enabled_ui and api_key_potentially_needed and not api_key_value:
            warn_title = self.parent_gui.tr("warning", default="Warning")
            warn_msg = self.parent_gui.tr(
                "msw_api_key_missing_warning",
                default="Astrometry.net API Key is missing. Plate solving for mosaic (anchor or panels) "
                        "might fail if local solvers are not configured or also fail. Continue anyway?"
            )
            if not messagebox.askyesno(warn_title, warn_msg, parent=self):
                return

        solver_choice = self.local_solver_choice_var.get()
        astap_path = self.astap_path_var.get().strip()
        astap_data_dir = self.astap_data_dir_var.get().strip()
        astap_radius = self.astap_search_radius_var.get()
        astap_downsample = self.astap_downsample_var.get()
        astap_sensitivity = self.astap_sensitivity_var.get()
        cluster_threshold = self.cluster_threshold_var.get()
        local_ansvr_path = self.local_ansvr_path_var.get().strip()

        validation_ok = True
        if solver_choice == "astap" and not astap_path:
            messagebox.showerror(
                self.parent_gui.tr("error"),
                self.parent_gui.tr("astap_path_required_error", default="ASTAP is selected, but the executable path is missing."),
                parent=self,
            )
            validation_ok = False
        elif solver_choice == "ansvr" and not local_ansvr_path:
            messagebox.showerror(
                self.parent_gui.tr("error"),
                self.parent_gui.tr("ansvr_path_required_error", default="Astrometry.net Local is selected, but the path/config is missing."),
                parent=self,
            )
            validation_ok = False
        if not validation_ok:
            return

        setattr(self.parent_gui.settings, 'local_solver_preference', solver_choice)
        if hasattr(self.parent_gui.settings, 'use_local_solver_priority'):
            delattr(self.parent_gui.settings, 'use_local_solver_priority')
        self.parent_gui.settings.astap_path = astap_path
        self.parent_gui.settings.astap_data_dir = astap_data_dir
        setattr(self.parent_gui.settings, 'astap_search_radius', astap_radius)
        self.parent_gui.settings.local_ansvr_path = local_ansvr_path
        try:
            self.parent_gui.settings.astrometry_api_key = self.parent_gui.astrometry_api_key_var.get().strip()
        except Exception:
            self.parent_gui.settings.astrometry_api_key = ''
        self.parent_gui.config['astap_default_downsample'] = int(astap_downsample)
        self.parent_gui.config['astap_default_sensitivity'] = int(astap_sensitivity)
        self.parent_gui.config['cluster_panel_threshold'] = float(cluster_threshold)
        if hasattr(self.parent_gui, 'cluster_threshold_var'):
            self.parent_gui.cluster_threshold_var.set(float(cluster_threshold))
        try:
            from zemosaic import zemosaic_config
            zemosaic_config.save_config(self.parent_gui.config)
        except Exception:
            pass
        
        self.parent_gui.settings.mosaic_mode_active = is_mosaic_enabled_ui
        self.parent_gui.mosaic_mode_active = is_mosaic_enabled_ui 
        print(f"DEBUG (MosaicSettingsWindow _on_ok): parent_gui.mosaic_mode_active mis à {self.parent_gui.mosaic_mode_active}")
        print(f"DEBUG (MosaicSettingsWindow _on_ok): parent_gui.settings.mosaic_mode_active mis à {self.parent_gui.settings.mosaic_mode_active}")

        # --- CONSTRUTION DU DICTIONNAIRE settings_dict_to_save ---
        # Lire la valeur du local_mosaic_scale_var (qui est un tk.StringVar) et la convertir en int
        mosaic_scale_factor_value = int(self.local_mosaic_scale_var.get()) # Lecture et conversion en int

        settings_dict_to_save = {
            'enabled': is_mosaic_enabled_ui, 
            'alignment_mode': selected_align_method,
            'fastalign_orb_features': self.local_fastalign_orb_features_var.get(),
            'fastalign_min_abs_matches': self.local_fastalign_min_abs_matches_var.get(),
            'fastalign_min_ransac': self.local_fastalign_min_ransac_var.get(), 
            'fastalign_ransac_thresh': self.local_fastalign_ransac_thresh_var.get(),
            # Pour les paramètres DAO, on utilise getattr avec un fallback si les variables Tkinter n'existent pas
            'fastalign_dao_fwhm': getattr(self, 'fa_dao_fwhm_var', tk.DoubleVar(value=3.5)).get(), 
            'fastalign_dao_thr_sig': getattr(self, 'fa_dao_thr_sig_var', tk.DoubleVar(value=8.0)).get(),
            'fastalign_dao_max_stars': getattr(self, 'fa_dao_max_stars_var', tk.DoubleVar(value=750.0)).get(),
            
            'kernel': self.local_drizzle_kernel_var.get(),
            'pixfrac': self.local_drizzle_pixfrac_var.get(),
            'use_gpu': self.local_use_gpu_var.get(),
            'fillval': self.local_drizzle_fillval_var.get(),
            'wht_threshold': self.local_drizzle_wht_thresh_storage_var.get(),
            
            'mosaic_scale_factor': mosaic_scale_factor_value, # <-- NOUVELLE LIGNE : Ajout de la valeur lue
        }
        # --- FIN CONSTRUCTION DU DICTIONNAIRE ---
        
        self.parent_gui.settings.mosaic_settings = settings_dict_to_save
        
        if hasattr(self.parent_gui, '_update_mosaic_status_indicator'):
            self.parent_gui._update_mosaic_status_indicator()

        print(f"DEBUG (MosaicSettingsWindow _on_ok): Paramètres mosaïque sauvegardés dans parent_gui.settings: {self.parent_gui.settings.mosaic_settings}")
        print(f"DEBUG (MosaicSettingsWindow _on_ok): Clé API globale sauvegardée dans parent_gui.settings (longueur): {len(self.parent_gui.settings.astrometry_api_key)}")

        self.grab_release()
        self.destroy()



#######################################################################################################################################

    def _on_cancel(self):
        # print("DEBUG (MosaicSettingsWindow _on_cancel): Clic sur Annuler.") # Commenté
        self.grab_release()
        self.destroy()
        if hasattr(self.parent_gui, 'update_mosaic_button_appearance'):
            self.parent_gui.update_mosaic_button_appearance() # Mettre à jour l'apparence même en cas d'annulation

    # Getter methods to expose selected astrometry settings
    def get_astrometry_solver_choice(self):
        return self.local_solver_choice_var.get()

    def get_astrometry_api_key(self):
        return self.api_key_entry.get()

    def get_astrometry_ansvr_path(self):
        return self.ansvr_path_entry.get()



# --- FIN DE LA CLASSE MosaicSettingsWindow ---