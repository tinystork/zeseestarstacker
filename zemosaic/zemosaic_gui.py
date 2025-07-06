# zemosaic_gui.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import multiprocessing
import os
import traceback
import time
import subprocess
import sys

try:
    import wmi
except ImportError:  # pragma: no cover - wmi may be unavailable on non Windows
    wmi = None

import importlib.util

CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None
cupy = None
getDeviceProperties = None
getDeviceCount = None

try:
    from PIL import Image, ImageTk # Importe depuis Pillow
    PILLOW_AVAILABLE_FOR_ICON = True
except ImportError:
    PILLOW_AVAILABLE_FOR_ICON = False
    print("AVERT GUI: Pillow (PIL) non installé. L'icône PNG ne peut pas être chargée.")
# --- Import du module de localisation ---
try:
    from locales.zemosaic_localization import ZeMosaicLocalization
    ZEMOSAIC_LOCALIZATION_AVAILABLE = True
except ImportError as e_loc:
    ZEMOSAIC_LOCALIZATION_AVAILABLE = False
    ZeMosaicLocalization = None # Factice
    print(f"ERREUR (zemosaic_gui): Impossible d'importer 'ZeMosaicLocalization': {e_loc}")

# --- Configuration Import ---
try:
    import zemosaic_config 
    ZEMOSAIC_CONFIG_AVAILABLE = True
except ImportError as e_config:
    ZEMOSAIC_CONFIG_AVAILABLE = False
    zemosaic_config = None
    print(f"AVERTISSEMENT (zemosaic_gui): 'zemosaic_config.py' non trouvé: {e_config}")

# --- Worker Import ---
try:
    # Import worker from the same package so relative imports inside it work
    from .zemosaic_worker import (
        run_hierarchical_mosaic,
        run_hierarchical_mosaic_process,
    )
    ZEMOSAIC_WORKER_AVAILABLE = True
except ImportError as e_worker:
    ZEMOSAIC_WORKER_AVAILABLE = False
    run_hierarchical_mosaic = None
    run_hierarchical_mosaic_process = None
    print(f"ERREUR (zemosaic_gui): 'run_hierarchical_mosaic' non trouvé: {e_worker}")

from dataclasses import asdict
from .solver_settings import SolverSettings



class ZeMosaicGUI:
    def __init__(self, root_window):
        self.root = root_window

        # --- DÉFINIR L'ICÔNE DE LA FENÊTRE (AVEC .ICO NATIF) ---
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(base_path, "icon", "zemosaic.ico")

            if os.path.exists(icon_path):
                self.root.iconbitmap(default=icon_path)
            else:
                print(f"AVERT GUI: Fichier d'icône ICO non trouvé à {icon_path}")
        except tk.TclError:
            print("AVERT GUI: Impossible de définir l'icône ICO (TclError).")
        except Exception as e_icon:
            print(f"AVERT GUI: Erreur lors de la définition de l'icône ICO: {e_icon}")
        # --- FIN DÉFINITION ICÔNE ---


        try:
            self.root.geometry("750x780") # Légère augmentation pour le nouveau widget
            self.root.minsize(700, 630) # Légère augmentation
        except tk.TclError:
            pass

        self.config = {}
        if ZEMOSAIC_CONFIG_AVAILABLE and zemosaic_config:
            self.config = zemosaic_config.load_config()
        else:
            # Dictionnaire de configuration de secours si zemosaic_config.py n'est pas trouvé
            # ou si le chargement échoue.
            self.config = { 
                "astap_executable_path": "", "astap_data_directory_path": "",
                "astap_default_search_radius": 3.0, "astap_default_downsample": 2,
                "astap_default_sensitivity": 100, "language": "en",
                "stacking_normalize_method": "none",
                "stacking_weighting_method": "none",
                "stacking_rejection_algorithm": "kappa_sigma",
                "stacking_kappa_low": 3.0,
                "stacking_kappa_high": 3.0,
                "stacking_winsor_limits": "0.05,0.05",
                "stacking_final_combine_method": "mean",
                "apply_radial_weight": False,
                "radial_feather_fraction": 0.8,
                "radial_shape_power": 2.0,
                "min_radial_weight_floor": 0.0, # Ajouté lors du test du plancher radial
                "final_assembly_method": "reproject_coadd",
                "num_processing_workers": 0 # 0 pour auto, anciennement -1
            }

        # --- GPU Detection helper ---
        def _detect_gpus():
            """Return a list of detected GPUs as ``(display_name, index)`` tuples.

            Detection tries multiple methods so it works on Windows, Linux and
            macOS without requiring the optional ``wmi`` module.
            """

            controllers = []
            if wmi:
                try:
                    obj = wmi.WMI()
                    controllers = [c.Name for c in obj.Win32_VideoController()]
                except Exception:
                    controllers = []

            if not controllers:
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    controllers = [l.strip() for l in out.splitlines() if l.strip()]
                except Exception:
                    controllers = []

            nv_cuda = []
            if CUPY_AVAILABLE:
                try:
                    import cupy
                    from cupy.cuda.runtime import getDeviceCount, getDeviceProperties
                    for i in range(getDeviceCount()):
                        name = getDeviceProperties(i)["name"]
                        if isinstance(name, bytes):
                            name = name.decode()
                        nv_cuda.append(name)
                except Exception:
                    nv_cuda = []

            def _simplify(n: str) -> str:
                return n.lower().replace("laptop gpu", "").strip()

            simple_cuda = [_simplify(n) for n in nv_cuda]
            gpus = []
            for disp in controllers:
                simp = _simplify(disp)
                idx = simple_cuda.index(simp) if simp in simple_cuda else None
                gpus.append((disp, idx))
            if not gpus and nv_cuda:
                gpus = [(name, idx) for idx, name in enumerate(nv_cuda)]

            gpus.insert(0, ("CPU (no GPU)", None))
            return gpus

        default_lang_from_config = self.config.get("language", 'en')
        if ZEMOSAIC_LOCALIZATION_AVAILABLE and ZeMosaicLocalization:
            self.localizer = ZeMosaicLocalization(language_code=default_lang_from_config)
        else:
            class MockLocalizer:
                def __init__(self, language_code='en'): self.language_code = language_code
                def get(self, key, default_text=None, **kwargs): return default_text if default_text is not None else f"_{key}_"
                def set_language(self, lang_code): self.language_code = lang_code
            self.localizer = MockLocalizer(language_code=default_lang_from_config)
        
        # --- Variable compteur tuile phase 3
        self.master_tile_count_var = tk.StringVar(value="") # Initialement vide
        
        
        # --- Définition des listes de clés pour les ComboBoxes ---
        self.norm_method_keys = ["none", "linear_fit", "sky_mean"]
        self.weight_method_keys = ["none", "noise_variance", "noise_fwhm"]
        self.reject_algo_keys = ["none", "kappa_sigma", "winsorized_sigma_clip", "linear_fit_clip"]
        self.combine_method_keys = ["mean", "median"]
        self.assembly_method_keys = ["reproject_coadd", "incremental"]
        # --- FIN Définition des listes de clés ---

        # --- Tkinter Variables ---
        self.input_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.astap_exe_path_var = tk.StringVar(value=self.config.get("astap_executable_path", ""))
        self.astap_data_dir_var = tk.StringVar(value=self.config.get("astap_data_directory_path", ""))
        self.astap_search_radius_var = tk.DoubleVar(value=self.config.get("astap_default_search_radius", 3.0))
        self.astap_downsample_var = tk.IntVar(value=self.config.get("astap_default_downsample", 2))
        self.astap_sensitivity_var = tk.IntVar(value=self.config.get("astap_default_sensitivity", 100))
        self.cluster_threshold_var = tk.DoubleVar(value=self.config.get("cluster_panel_threshold", 0.5))
        self.save_final_uint16_var = tk.BooleanVar(value=self.config.get("save_final_as_uint16", False))

        # --- Solver Settings ---
        try:
            self.solver_settings = SolverSettings.load_default()
        except Exception:
            self.solver_settings = SolverSettings()
        self.solver_choice_var = tk.StringVar(value=self.solver_settings.solver_choice)
        self.solver_choice_var.trace_add("write", self._update_solver_frames)
        self.astrometry_api_key_var = tk.StringVar(value=self.solver_settings.api_key)
        self.astrometry_timeout_var = tk.IntVar(value=self.solver_settings.timeout)
        self.astrometry_downsample_var = tk.IntVar(value=self.solver_settings.downsample)
        self.force_lum_var = tk.BooleanVar(value=self.solver_settings.force_lum)
        
        self.is_processing = False
        self.worker_process = None
        self.progress_queue = None
        self.progress_bar_var = tk.DoubleVar(value=0.0)
        self.eta_var = tk.StringVar(value=self._tr("initial_eta_value", "--:--:--"))
        self.elapsed_time_var = tk.StringVar(value=self._tr("initial_elapsed_time", "00:00:00"))
        self._chrono_start_time = None
        self._chrono_after_id = None
        self._stage_times = {}
        
        self.current_language_var = tk.StringVar(value=self.localizer.language_code)
        self.current_language_var.trace_add("write", self._on_language_change)
        
        # --- Variables Tkinter pour les Options de Stacking ---
        self.stacking_normalize_method_var = tk.StringVar(value=self.config.get("stacking_normalize_method", self.norm_method_keys[0]))
        self.stacking_weighting_method_var = tk.StringVar(value=self.config.get("stacking_weighting_method", self.weight_method_keys[0]))
        self.stacking_rejection_algorithm_var = tk.StringVar(value=self.config.get("stacking_rejection_algorithm", self.reject_algo_keys[1]))
        
        self.stacking_kappa_low_var = tk.DoubleVar(value=self.config.get("stacking_kappa_low", 3.0))
        self.stacking_kappa_high_var = tk.DoubleVar(value=self.config.get("stacking_kappa_high", 3.0))
        self.stacking_winsor_limits_str_var = tk.StringVar(value=self.config.get("stacking_winsor_limits", "0.05,0.05"))
        self.stacking_final_combine_method_var = tk.StringVar(value=self.config.get("stacking_final_combine_method", self.combine_method_keys[0]))
        
        # --- PONDÉRATION RADIALE ---
        self.apply_radial_weight_var = tk.BooleanVar(value=self.config.get("apply_radial_weight", False))
        self.radial_feather_fraction_var = tk.DoubleVar(value=self.config.get("radial_feather_fraction", 0.8))
        self.min_radial_weight_floor_var = tk.DoubleVar(value=self.config.get("min_radial_weight_floor", 0.0)) # Ajouté
        # radial_shape_power est géré via self.config directement
        
        # --- METHODE D'ASSEMBLAGE ---
        self.final_assembly_method_var = tk.StringVar(
            value=self.config.get("final_assembly_method", self.assembly_method_keys[0])
        )
        self.final_assembly_method_var.trace_add("write", self._on_assembly_method_change)
        
        # --- NOMBRE DE WORKERS ---
        # Utiliser 0 pour auto, comme convenu. La clé de config est "num_processing_workers".
        # Si la valeur dans config est -1 (ancienne convention pour auto), on la met à 0.
        num_workers_from_config = self.config.get("num_processing_workers", 0)
        if num_workers_from_config == -1:
            num_workers_from_config = 0
        self.num_workers_var = tk.IntVar(value=num_workers_from_config)
        self.winsor_workers_var = tk.IntVar(value=self.config.get("winsor_worker_limit", 6))
        # --- FIN NOMBRE DE WORKERS ---
        # --- NOUVELLES VARIABLES TKINTER POUR LE ROGNAGE ---
        self.apply_master_tile_crop_var = tk.BooleanVar(
            value=self.config.get("apply_master_tile_crop", True) # Désactivé par défaut
        )
        self.master_tile_crop_percent_var = tk.DoubleVar(
            value=self.config.get("master_tile_crop_percent", 18.0) # 18% par côté par défaut si activé
        )
        self.use_memmap_var = tk.BooleanVar(value=self.config.get("coadd_use_memmap", False))
        self.mm_dir_var = tk.StringVar(value=self.config.get("coadd_memmap_dir", ""))
        self.cleanup_memmap_var = tk.BooleanVar(value=self.config.get("coadd_cleanup_memmap", True))
        self.auto_limit_frames_var = tk.BooleanVar(value=self.config.get("auto_limit_frames_per_master_tile", True))
        self.max_raw_per_tile_var = tk.IntVar(value=self.config.get("max_raw_per_master_tile", 0))
        self.use_gpu_phase5_var = tk.BooleanVar(value=self.config.get("use_gpu_phase5", False))
        self._gpus = _detect_gpus()
        self.gpu_selector_var = tk.StringVar(
            value=self.config.get("gpu_selector", self._gpus[0][0] if self._gpus else "")
        )
        # ---  ---

        self.translatable_widgets = {}

        self._build_ui()
        self._update_solver_frames()
        self.root.after_idle(self._update_ui_language) # Déplacé après _build_ui pour que les widgets existent
        #self.root.after_idle(self._update_assembly_dependent_options) # En prévision d'un forçage de combinaisons 
        self.root.after_idle(self._update_rejection_params_state) # Déjà présent, garder

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._load_default_paths_for_dev() # Si encore utilisée



    def _tr(self, key, default_text=None, **kwargs):
        if self.localizer:
            # Si default_text n'est pas explicitement None, on le passe au localizer
            # Sinon, le localizer utilisera sa propre logique de fallback (ex: _key_)
            if default_text is not None:
                return self.localizer.get(key, default_text, **kwargs)
            else:
                # Tenter de trouver un fallback anglais générique si la clé n'est pas trouvée
                # dans la langue courante ET si aucun default_text n'est fourni.
                # Cela peut être redondant si ZeMosaicLocalization le fait déjà.
                # Pour l'instant, on laisse le localizer gérer son propre fallback.
                return self.localizer.get(key, **kwargs)

        return default_text if default_text is not None else f"_{key}_" # Fallback très basique

    def _on_language_change(self, *args):
        new_lang = self.current_language_var.get()
        if self.localizer and self.localizer.language_code != new_lang:
            print(f"DEBUG GUI: Langue changée vers '{new_lang}'")
            self.localizer.set_language(new_lang)
            self.config["language"] = new_lang 
            if ZEMOSAIC_CONFIG_AVAILABLE and zemosaic_config:
                zemosaic_config.save_config(self.config)
            self._update_ui_language()


# Dans la classe ZeMosaicGUI de zemosaic_gui.py

    def _refresh_combobox(self, combo: ttk.Combobox,
                          tk_var: tk.StringVar,
                          keys: list[str],
                          tr_prefix: str) -> None:
        """
        Re-populate `combo` with the translated text that corresponds to `keys`
        and make sure the *displayed* value matches the key currently held in
        `tk_var`.
        """
        if not combo or not hasattr(combo, 'winfo_exists') or not combo.winfo_exists():
            # Widget n'existe plus ou n'a pas été correctement initialisé
            return
        if not keys:
            # Pas de clés à afficher, vider le combobox
            try:
                combo["values"] = []
                combo.set("") 
            except tk.TclError: pass # Le widget a pu être détruit entre-temps
            return

        # --- Construction de la liste des valeurs traduites pour le dropdown ---
        try:
            translated_values_for_dropdown = [self._tr(f"{tr_prefix}_{k}") for k in keys]
            combo["values"] = translated_values_for_dropdown
        except tk.TclError:
            # Le widget a pu être détruit pendant la mise à jour des valeurs
            return 

        # --- Synchronisation du texte affiché (combo.set()) avec la clé stockée dans tk_var ---
        current_key_stored_in_tk_var = tk_var.get()
        final_text_to_display = ""

        if current_key_stored_in_tk_var in keys:
            key_for_translation_display = f"{tr_prefix}_{current_key_stored_in_tk_var}"
            final_text_to_display = self._tr(key_for_translation_display)
        else: 
            # La clé stockée dans tk_var n'est pas une clé valide connue pour ce combobox
            # Utiliser la première clé de la liste comme fallback
            if keys: # S'assurer qu'il y a au moins une clé valide
                fallback_key = keys[0]
                tk_var.set(fallback_key) # Mettre à jour le StringVar avec cette clé fallback
                key_for_translation_display_fallback = f"{tr_prefix}_{fallback_key}"
                final_text_to_display = self._tr(key_for_translation_display_fallback)
            # Si 'keys' est vide (ne devrait pas arriver à cause du check plus haut, mais par sécurité)
            # final_text_to_display restera une chaîne vide.

        try:
            combo.set(final_text_to_display)
        except tk.TclError:
            # Le widget a pu être détruit avant que .set() ne soit appelé
            pass

    def _combo_to_key(self, event, combo: ttk.Combobox, tk_var: tk.StringVar, keys: list[str], tr_prefix: str):
        """
        Callback pour mettre à jour le tk_var avec la clé correspondante
        au texte affiché sélectionné dans le combobox.
        """
        displayed_text = combo.get()
        found_key = None
        for k_item in keys:
            if self._tr(f"{tr_prefix}_{k_item}") == displayed_text:
                found_key = k_item
                break
        
        if found_key is not None:
            if tk_var.get() != found_key: # Éviter les écritures inutiles si la clé n'a pas changé
                tk_var.set(found_key)
                # Si le changement de clé doit déclencher une autre action (ex: _update_rejection_params_state)
                if combo == getattr(self, 'reject_algo_combo', None): # Vérifier si c'est le combo de rejet
                     if hasattr(self, '_update_rejection_params_state'):
                        self._update_rejection_params_state()
        # else:
            # print(f"WARN _combo_to_key: Clé non trouvée pour l'affichage '{displayed_text}' et le préfixe '{tr_prefix}'. tk_var non modifié.")

    def _update_solver_frames(self, *args):
        """Show or hide solver-specific frames based on the selected solver."""
        choice = self.solver_choice_var.get()

        if choice == "ASTAP":
            # These frames use the ``pack`` geometry manager, so we must
            # repack them when showing and use ``pack_forget`` to hide them.
            self.astap_cfg_frame.pack(fill=tk.X, pady=(0, 10))
            self.astap_params_frame.pack(fill=tk.X, pady=(0, 10))
            self.astrometry_frame.grid_remove()
        elif choice == "ASTROMETRY":
            self.astap_cfg_frame.pack_forget()
            self.astap_params_frame.pack_forget()
            self.astrometry_frame.grid()
        elif choice == "ANSVR":
            self.astap_cfg_frame.pack_forget()
            self.astap_params_frame.pack_forget()
            self.astrometry_frame.grid_remove()
        else:
            self.astap_cfg_frame.pack_forget()
            self.astap_params_frame.pack_forget()
            self.astrometry_frame.grid_remove()

# Dans la classe ZeMosaicGUI de zemosaic_gui.py

    def _build_ui(self):
        # --- Cadre principal qui contiendra le Canvas et la Scrollbar ---
        main_container_frame = ttk.Frame(self.root)
        main_container_frame.pack(expand=True, fill=tk.BOTH)

        # --- Canvas pour le contenu scrollable ---
        self.main_canvas = tk.Canvas(main_container_frame, borderwidth=0, highlightthickness=0)
        
        # --- Scrollbar Verticale ---
        self.scrollbar_y = ttk.Scrollbar(main_container_frame, orient="vertical", command=self.main_canvas.yview)
        self.main_canvas.configure(yscrollcommand=self.scrollbar_y.set)

        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Cadre intérieur qui sera scrollé (ancien main_frame) ---
        self.scrollable_content_frame = ttk.Frame(self.main_canvas, padding="10")
        self.canvas_window = self.main_canvas.create_window((0, 0), window=self.scrollable_content_frame, anchor="nw")

        # --- Lier les événements pour le scroll et la redimension ---
        def _on_frame_configure(event=None):
            self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

        def _on_canvas_configure(event=None):
            canvas_width = event.width
            if self.scrollbar_y.winfo_ismapped():
                 canvas_width -= self.scrollbar_y.winfo_width()
            self.main_canvas.itemconfig(self.canvas_window, width=canvas_width)

        self.scrollable_content_frame.bind("<Configure>", _on_frame_configure)
        self.main_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            if event.num == 4: self.main_canvas.yview_scroll(-1, "units")
            elif event.num == 5: self.main_canvas.yview_scroll(1, "units")
            else: self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        self.root.bind_all("<MouseWheel>", _on_mousewheel)
        self.root.bind_all("<Button-4>", _on_mousewheel)
        self.root.bind_all("<Button-5>", _on_mousewheel)

        # === DEBUT DU CONTENU DE L'ANCIEN main_frame (maintenant scrollable_content_frame) ===

        # --- Sélecteur de Langue ---
        lang_select_frame = ttk.Frame(self.scrollable_content_frame)
        lang_select_frame.pack(fill=tk.X, pady=(0,10), padx=5)
        lang_label = ttk.Label(lang_select_frame, text="") 
        lang_label.pack(side=tk.LEFT, padx=(0,5))
        self.translatable_widgets["language_selector_label"] = lang_label
        
        available_langs = ['en', 'fr'] 
        # ... (logique de détection des langues) ...
        if self.localizer and hasattr(self.localizer, 'locales_dir_abs_path') and self.localizer.locales_dir_abs_path:
            try:
                available_langs = sorted([
                    f.split('.')[0] for f in os.listdir(self.localizer.locales_dir_abs_path) 
                    if f.endswith(".json") and os.path.isfile(os.path.join(self.localizer.locales_dir_abs_path, f))
                ])
                if not available_langs: available_langs = ['en','fr'] 
            except FileNotFoundError: available_langs = ['en', 'fr']
            except Exception: available_langs = ['en', 'fr']
        else: available_langs = ['en', 'fr']

        self.language_combo = ttk.Combobox(lang_select_frame, textvariable=self.current_language_var, 
                                           values=available_langs, state="readonly", width=5)
        self.language_combo.pack(side=tk.LEFT)

        # --- Folder Selection Frame ---
        folders_frame = ttk.LabelFrame(self.scrollable_content_frame, text="", padding="10")
        # ... (contenu de folders_frame) ...
        folders_frame.pack(fill=tk.X, pady=(0,10)); folders_frame.columnconfigure(1, weight=1)
        self.translatable_widgets["folders_frame_title"] = folders_frame
        ttk.Label(folders_frame, text="").grid(row=0, column=0, padx=5, pady=5, sticky="w"); self.translatable_widgets["input_folder_label"] = folders_frame.grid_slaves(row=0,column=0)[0]
        ttk.Entry(folders_frame, textvariable=self.input_dir_var, width=60).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(folders_frame, text="", command=self._browse_input_dir).grid(row=0, column=2, padx=5, pady=5); self.translatable_widgets["browse_button"] = folders_frame.grid_slaves(row=0,column=2)[0]
        ttk.Label(folders_frame, text="").grid(row=1, column=0, padx=5, pady=5, sticky="w"); self.translatable_widgets["output_folder_label"] = folders_frame.grid_slaves(row=1,column=0)[0]
        ttk.Entry(folders_frame, textvariable=self.output_dir_var, width=60).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(folders_frame, text="", command=self._browse_output_dir).grid(row=1, column=2, padx=5, pady=5); self.translatable_widgets["browse_button_output"] = folders_frame.grid_slaves(row=1,column=2)[0]

        ttk.Label(folders_frame, text="").grid(row=2, column=0, padx=5, pady=5, sticky="w"); self.translatable_widgets["save_final_16bit_label"] = folders_frame.grid_slaves(row=2,column=0)[0]
        ttk.Checkbutton(folders_frame, variable=self.save_final_uint16_var).grid(row=2, column=1, padx=5, pady=5, sticky="w")


        # --- ASTAP Configuration Frame ---
        astap_cfg_frame = ttk.LabelFrame(self.scrollable_content_frame, text="", padding="10")
        # ... (contenu de astap_cfg_frame) ...
        astap_cfg_frame.pack(fill=tk.X, pady=(0,10)); astap_cfg_frame.columnconfigure(1, weight=1)
        self.astap_cfg_frame = astap_cfg_frame
        self.translatable_widgets["astap_config_frame_title"] = astap_cfg_frame
        ttk.Label(astap_cfg_frame, text="").grid(row=0, column=0, padx=5, pady=5, sticky="w"); self.translatable_widgets["astap_exe_label"] = astap_cfg_frame.grid_slaves(row=0,column=0)[0]
        ttk.Entry(astap_cfg_frame, textvariable=self.astap_exe_path_var, width=60).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(astap_cfg_frame, text="", command=self._browse_and_save_astap_exe).grid(row=0, column=2, padx=5, pady=5); self.translatable_widgets["browse_save_button"] = astap_cfg_frame.grid_slaves(row=0,column=2)[0]
        ttk.Label(astap_cfg_frame, text="").grid(row=1, column=0, padx=5, pady=5, sticky="w"); self.translatable_widgets["astap_data_dir_label"] = astap_cfg_frame.grid_slaves(row=1,column=0)[0]
        ttk.Entry(astap_cfg_frame, textvariable=self.astap_data_dir_var, width=60).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(astap_cfg_frame, text="", command=self._browse_and_save_astap_data_dir).grid(row=1, column=2, padx=5, pady=5); self.translatable_widgets["browse_save_button_data"] = astap_cfg_frame.grid_slaves(row=1,column=2)[0]

        # --- Parameters Frame ---
        params_frame = ttk.LabelFrame(self.scrollable_content_frame, text="", padding="10")
        # ... (contenu de params_frame) ...
        params_frame.pack(fill=tk.X, pady=(0,10))
        self.astap_params_frame = params_frame
        self.translatable_widgets["mosaic_astap_params_frame_title"] = params_frame
        param_row_idx = 0 
        ttk.Label(params_frame, text="").grid(row=param_row_idx, column=0, padx=5, pady=3, sticky="w"); self.translatable_widgets["astap_search_radius_label"] = params_frame.grid_slaves(row=param_row_idx,column=0)[0]
        ttk.Spinbox(params_frame, from_=0.1, to=180.0, increment=0.1, textvariable=self.astap_search_radius_var, width=8, format="%.1f").grid(row=param_row_idx, column=1, padx=5, pady=3, sticky="w"); param_row_idx+=1
        ttk.Label(params_frame, text="").grid(row=param_row_idx, column=0, padx=5, pady=3, sticky="w"); self.translatable_widgets["astap_downsample_label"] = params_frame.grid_slaves(row=param_row_idx,column=0)[0]
        ttk.Spinbox(params_frame, from_=0, to=4, increment=1, textvariable=self.astap_downsample_var, width=8).grid(row=param_row_idx, column=1, padx=5, pady=3, sticky="w")
        ttk.Label(params_frame, text="").grid(row=param_row_idx, column=2, padx=5, pady=3, sticky="w"); self.translatable_widgets["astap_downsample_note"] = params_frame.grid_slaves(row=param_row_idx,column=2)[0]; param_row_idx+=1
        ttk.Label(params_frame, text="").grid(row=param_row_idx, column=0, padx=5, pady=3, sticky="w"); self.translatable_widgets["astap_sensitivity_label"] = params_frame.grid_slaves(row=param_row_idx,column=0)[0]
        ttk.Spinbox(params_frame, from_=-25, to_=500, increment=1, textvariable=self.astap_sensitivity_var, width=8).grid(row=param_row_idx, column=1, padx=5, pady=3, sticky="w")
        ttk.Label(params_frame, text="").grid(row=param_row_idx, column=2, padx=5, pady=3, sticky="w"); self.translatable_widgets["astap_sensitivity_note"] = params_frame.grid_slaves(row=param_row_idx,column=2)[0]; param_row_idx+=1
        ttk.Label(params_frame, text="").grid(row=param_row_idx, column=0, padx=5, pady=3, sticky="w"); self.translatable_widgets["panel_clustering_threshold_label"] = params_frame.grid_slaves(row=param_row_idx,column=0)[0]
        ttk.Spinbox(params_frame, from_=0.01, to=5.0, increment=0.01, textvariable=self.cluster_threshold_var, width=8, format="%.2f").grid(row=param_row_idx, column=1, padx=5, pady=3, sticky="w")
        param_row_idx += 1
        ttk.Label(params_frame, text=self._tr("force_lum_label", "Convert to Luminance (mono):")).grid(row=param_row_idx, column=0, padx=5, pady=3, sticky="w"); self.translatable_widgets["force_lum_label"] = params_frame.grid_slaves(row=param_row_idx,column=0)[0]
        ttk.Checkbutton(params_frame, variable=self.force_lum_var).grid(row=param_row_idx, column=1, padx=5, pady=3, sticky="w")

        # --- Solver Selection Frame ---
        solver_frame = ttk.LabelFrame(self.scrollable_content_frame, text=self._tr("solver_frame_title", "Plate Solver"), padding="10")
        solver_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(solver_frame, text=self._tr("solver_choice_label", "Solver:"))\
            .grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.solver_combo = ttk.Combobox(
            solver_frame,
            textvariable=self.solver_choice_var,
            values=["ASTAP", "ASTROMETRY", "ANSVR", "NONE"],
            state="readonly",
            width=15,
        )
        self.solver_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.solver_combo.bind("<<ComboboxSelected>>", lambda e: self._update_solver_frames())

        self.astrometry_frame = ttk.LabelFrame(solver_frame, text=self._tr("astrometry_group_title", "Astrometry.net"), padding="5")
        self.astrometry_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=(5, 0), sticky="ew")
        self.astrometry_frame.columnconfigure(1, weight=1)
        ttk.Label(self.astrometry_frame, text=self._tr("api_key_label", "API Key:"))\
            .grid(row=0, column=0, padx=5, pady=3, sticky="w")
        ttk.Entry(self.astrometry_frame, textvariable=self.astrometry_api_key_var)\
            .grid(row=0, column=1, padx=5, pady=3, sticky="ew")
        ttk.Label(self.astrometry_frame, text=self._tr("timeout_label", "Timeout (s):"))\
            .grid(row=1, column=0, padx=5, pady=3, sticky="w")
        ttk.Spinbox(self.astrometry_frame, from_=10, to=300, textvariable=self.astrometry_timeout_var, width=8)\
            .grid(row=1, column=1, padx=5, pady=3, sticky="w")
        ttk.Label(self.astrometry_frame, text=self._tr("downsample_label", "Blind-solve Downsample:"))\
            .grid(row=2, column=0, padx=5, pady=3, sticky="w")
        ttk.Spinbox(self.astrometry_frame, from_=1, to=8, textvariable=self.astrometry_downsample_var, width=8)\
            .grid(row=2, column=1, padx=5, pady=3, sticky="w")

        self._update_solver_frames()

        # --- Stacking Options Frame ---
        stacking_options_frame = ttk.LabelFrame(self.scrollable_content_frame, text="", padding="10") 
        # ... (contenu de stacking_options_frame avec Normalisation, Pondération, Rejet, Combinaison, Pondération Radiale, Plancher Poids Radial) ...
        stacking_options_frame.pack(fill=tk.X, pady=(0,10))
        self.translatable_widgets["stacking_options_frame_title"] = stacking_options_frame
        stacking_options_frame.columnconfigure(1, weight=1) 
        stk_opt_row = 0
        # Normalisation
        norm_label = ttk.Label(stacking_options_frame, text="")
        norm_label.grid(row=stk_opt_row, column=0, padx=5, pady=3, sticky="w"); self.translatable_widgets["stacking_norm_method_label"] = norm_label
        self.norm_method_combo = ttk.Combobox(stacking_options_frame, values=[], state="readonly", width=25)
        self.norm_method_combo.grid(row=stk_opt_row, column=1, padx=5, pady=3, sticky="ew", columnspan=3)
        self.norm_method_combo.bind("<<ComboboxSelected>>", lambda e, c=self.norm_method_combo, v=self.stacking_normalize_method_var, k_list=self.norm_method_keys, p="norm_method": self._combo_to_key(e, c, v, k_list, p)); stk_opt_row += 1
        # Pondération
        weight_label = ttk.Label(stacking_options_frame, text="")
        weight_label.grid(row=stk_opt_row, column=0, padx=5, pady=3, sticky="w"); self.translatable_widgets["stacking_weight_method_label"] = weight_label
        self.weight_method_combo = ttk.Combobox(stacking_options_frame, values=[], state="readonly", width=25)
        self.weight_method_combo.grid(row=stk_opt_row, column=1, padx=5, pady=3, sticky="ew", columnspan=3)
        self.weight_method_combo.bind("<<ComboboxSelected>>", lambda e, c=self.weight_method_combo, v=self.stacking_weighting_method_var, k_list=self.weight_method_keys, p="weight_method": self._combo_to_key(e, c, v, k_list, p)); stk_opt_row += 1
        # Rejet
        reject_label = ttk.Label(stacking_options_frame, text="")
        reject_label.grid(row=stk_opt_row, column=0, padx=5, pady=3, sticky="w"); self.translatable_widgets["stacking_reject_algo_label"] = reject_label
        self.reject_algo_combo = ttk.Combobox(stacking_options_frame, values=[], state="readonly", width=25)
        self.reject_algo_combo.grid(row=stk_opt_row, column=1, padx=5, pady=3, sticky="ew", columnspan=3)
        self.reject_algo_combo.bind("<<ComboboxSelected>>", lambda e, c=self.reject_algo_combo, v=self.stacking_rejection_algorithm_var, k_list=self.reject_algo_keys, p="reject_algo": self._combo_to_key(e, c, v, k_list, p)); stk_opt_row += 1
        # Kappa
        kappa_params_frame = ttk.Frame(stacking_options_frame) 
        kappa_params_frame.grid(row=stk_opt_row, column=0, columnspan=4, sticky="ew", padx=0, pady=0)
        kappa_low_label = ttk.Label(kappa_params_frame, text=""); kappa_low_label.pack(side=tk.LEFT, padx=(5,2)); self.translatable_widgets["stacking_kappa_low_label"] = kappa_low_label
        self.kappa_low_spinbox = ttk.Spinbox(kappa_params_frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.stacking_kappa_low_var, width=6); self.kappa_low_spinbox.pack(side=tk.LEFT, padx=(0,10))
        kappa_high_label = ttk.Label(kappa_params_frame, text=""); kappa_high_label.pack(side=tk.LEFT, padx=(5,2)); self.translatable_widgets["stacking_kappa_high_label"] = kappa_high_label
        self.kappa_high_spinbox = ttk.Spinbox(kappa_params_frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.stacking_kappa_high_var, width=6); self.kappa_high_spinbox.pack(side=tk.LEFT, padx=(0,5)); stk_opt_row += 1
        # Winsor
        winsor_params_frame = ttk.Frame(stacking_options_frame)
        winsor_params_frame.grid(row=stk_opt_row, column=0, columnspan=4, sticky="ew", padx=0, pady=0)
        winsor_label = ttk.Label(winsor_params_frame, text=""); winsor_label.pack(side=tk.LEFT, padx=(5,2)); self.translatable_widgets["stacking_winsor_limits_label"] = winsor_label
        self.winsor_limits_entry = ttk.Entry(winsor_params_frame, textvariable=self.stacking_winsor_limits_str_var, width=10); self.winsor_limits_entry.pack(side=tk.LEFT, padx=(0,5))
        winsor_note = ttk.Label(winsor_params_frame, text=""); winsor_note.pack(side=tk.LEFT, padx=(5,0)); self.translatable_widgets["stacking_winsor_note"] = winsor_note; stk_opt_row += 1
        # Combinaison Finale MT
        combine_label = ttk.Label(stacking_options_frame, text="")
        combine_label.grid(row=stk_opt_row, column=0, padx=5, pady=3, sticky="w"); self.translatable_widgets["stacking_final_combine_label"] = combine_label
        self.final_combine_combo = ttk.Combobox(stacking_options_frame, values=[], state="readonly", width=25)
        self.final_combine_combo.grid(row=stk_opt_row, column=1, padx=5, pady=3, sticky="ew", columnspan=3)
        self.final_combine_combo.bind("<<ComboboxSelected>>", lambda e, c=self.final_combine_combo, v=self.stacking_final_combine_method_var, k_list=self.combine_method_keys, p="combine_method": self._combo_to_key(e, c, v, k_list, p)); stk_opt_row += 1
        # Pondération Radiale
        self.apply_radial_weight_label = ttk.Label(stacking_options_frame, text="")
        self.apply_radial_weight_label.grid(row=stk_opt_row, column=0, padx=5, pady=3, sticky="w"); self.translatable_widgets["stacking_apply_radial_label"] = self.apply_radial_weight_label
        self.apply_radial_weight_check = ttk.Checkbutton(stacking_options_frame, variable=self.apply_radial_weight_var); self.apply_radial_weight_check.grid(row=stk_opt_row, column=1, padx=5, pady=3, sticky="w"); stk_opt_row += 1
        # Feather Fraction
        self.radial_feather_label = ttk.Label(stacking_options_frame, text="")
        self.radial_feather_label.grid(row=stk_opt_row, column=0, padx=5, pady=3, sticky="w"); self.translatable_widgets["stacking_radial_feather_label"] = self.radial_feather_label
        self.radial_feather_spinbox = ttk.Spinbox(stacking_options_frame, from_=0.1, to=1.0, increment=0.05, textvariable=self.radial_feather_fraction_var, width=8, format="%.2f")
        self.radial_feather_spinbox.grid(row=stk_opt_row, column=1, padx=5, pady=3, sticky="w"); stk_opt_row += 1
        # Min Radial Weight Floor
        self.min_radial_floor_label = ttk.Label(stacking_options_frame, text="")
        self.min_radial_floor_label.grid(row=stk_opt_row, column=0, padx=5, pady=3, sticky="w"); self.translatable_widgets["stacking_min_radial_floor_label"] = self.min_radial_floor_label
        self.min_radial_floor_spinbox = ttk.Spinbox(stacking_options_frame, from_=0.0, to=0.5, increment=0.01, textvariable=self.min_radial_weight_floor_var, width=8, format="%.2f")
        self.min_radial_floor_spinbox.grid(row=stk_opt_row, column=1, padx=5, pady=3, sticky="w")
        min_radial_floor_note = ttk.Label(stacking_options_frame, text=""); min_radial_floor_note.grid(row=stk_opt_row, column=2, padx=5, pady=3, sticky="w"); self.translatable_widgets["stacking_min_radial_floor_note"] = min_radial_floor_note; stk_opt_row += 1

        # Max raw per master tile
        self.max_raw_per_tile_label = ttk.Label(stacking_options_frame, text="")
        self.max_raw_per_tile_label.grid(row=stk_opt_row, column=0, padx=5, pady=3, sticky="w")
        self.translatable_widgets["max_raw_per_master_tile_label"] = self.max_raw_per_tile_label
        self.max_raw_per_tile_spinbox = ttk.Spinbox(
            stacking_options_frame,
            from_=0,
            to=9999,
            increment=1,
            textvariable=self.max_raw_per_tile_var,
            width=8
        )
        self.max_raw_per_tile_spinbox.grid(row=stk_opt_row, column=1, padx=5, pady=3, sticky="w")
        max_raw_note = ttk.Label(stacking_options_frame, text="")
        max_raw_note.grid(row=stk_opt_row, column=2, padx=(10,5), pady=3, sticky="w")
        self.translatable_widgets["max_raw_per_master_tile_note"] = max_raw_note
        stk_opt_row += 1


        # --- AJOUT DU CADRE POUR LES OPTIONS DE PERFORMANCE (NOMBRE DE THREADS) ---
        perf_options_frame = ttk.LabelFrame(self.scrollable_content_frame, text="", padding="10")
        perf_options_frame.pack(fill=tk.X, pady=(5, 10), padx=0) # pack avant "Options d'Assemblage Final"
        self.translatable_widgets["performance_options_frame_title"] = perf_options_frame
        perf_options_frame.columnconfigure(1, weight=0) # Les widgets ne s'étendent pas horizontalement ici
        perf_options_frame.columnconfigure(2, weight=1) # La note peut s'étendre

        # Label et Spinbox pour le nombre de threads
        num_workers_label = ttk.Label(perf_options_frame, text="")
        num_workers_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.translatable_widgets["num_workers_label"] = num_workers_label
        
        # Déterminer une limite supérieure raisonnable pour le Spinbox
        # os.cpu_count() peut retourner None, donc prévoir un fallback
        cpu_cores = os.cpu_count()
        max_spin_workers = 16 # Plafond par défaut
        if cpu_cores:
            max_spin_workers = max(1, cpu_cores) * 2 # Ex: jusqu'à 2x le nb de coeurs logiques, ou au moins 1*2=2
            if max_spin_workers > 32: max_spin_workers = 32 # Plafonner à 32 pour éviter des valeurs trop grandes
        
        self.num_workers_spinbox = ttk.Spinbox(
            perf_options_frame,
            from_=0,  # 0 pour auto
            to=max_spin_workers,
            increment=1,
            textvariable=self.num_workers_var,
            width=8 # Largeur fixe pour le spinbox
        )
        self.num_workers_spinbox.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        num_workers_note = ttk.Label(perf_options_frame, text="")
        num_workers_note.grid(row=0, column=2, padx=(10,5), pady=5, sticky="ew") # Note avec un peu plus de marge
        self.translatable_widgets["num_workers_note"] = num_workers_note

        winsor_workers_label = ttk.Label(perf_options_frame, text="")
        winsor_workers_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.translatable_widgets["winsor_workers_label"] = winsor_workers_label

        self.winsor_workers_spinbox = ttk.Spinbox(
            perf_options_frame,
            from_=1,
            to=16,
            increment=1,
            textvariable=self.winsor_workers_var,
            width=8
        )
        self.winsor_workers_spinbox.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        winsor_workers_note = ttk.Label(perf_options_frame, text="")
        winsor_workers_note.grid(row=1, column=2, padx=(10,5), pady=5, sticky="ew")
        self.translatable_widgets["winsor_workers_note"] = winsor_workers_note
        # --- FIN CADRE OPTIONS DE PERFORMANCE ---
        # --- NOUVEAU CADRE : OPTIONS DE ROGNAGE DES TUILES MAÎTRESSES ---
        crop_options_frame = ttk.LabelFrame(self.scrollable_content_frame, text="", padding="10")
        crop_options_frame.pack(fill=tk.X, pady=(5, 10), padx=0)
        self.translatable_widgets["crop_options_frame_title"] = crop_options_frame
        crop_options_frame.columnconfigure(1, weight=0) # Labels et spinbox de largeur fixe
        crop_options_frame.columnconfigure(2, weight=1) # La note peut s'étendre

        crop_opt_row = 0

        # Checkbutton pour activer le rognage
        self.apply_crop_label = ttk.Label(crop_options_frame, text="")
        self.apply_crop_label.grid(row=crop_opt_row, column=0, padx=5, pady=3, sticky="w")
        self.translatable_widgets["apply_master_tile_crop_label"] = self.apply_crop_label
        
        self.apply_crop_check = ttk.Checkbutton(
            crop_options_frame, 
            variable=self.apply_master_tile_crop_var,
            command=self._update_crop_options_state # Pour griser le spinbox si décoché
        )
        self.apply_crop_check.grid(row=crop_opt_row, column=1, padx=5, pady=3, sticky="w")
        crop_opt_row += 1

        # Spinbox pour le pourcentage de rognage
        self.crop_percent_label = ttk.Label(crop_options_frame, text="")
        self.crop_percent_label.grid(row=crop_opt_row, column=0, padx=5, pady=3, sticky="w")
        self.translatable_widgets["master_tile_crop_percent_label"] = self.crop_percent_label

        self.crop_percent_spinbox = ttk.Spinbox(
            crop_options_frame,
            from_=0.0, to=25.0, increment=0.5, # Rogner de 0% à 25% par côté semble raisonnable
            textvariable=self.master_tile_crop_percent_var,
            width=8, format="%.1f"
        )
        self.crop_percent_spinbox.grid(row=crop_opt_row, column=1, padx=5, pady=3, sticky="w")
        
        crop_percent_note = ttk.Label(crop_options_frame, text="")
        crop_percent_note.grid(row=crop_opt_row, column=2, padx=(10,5), pady=3, sticky="ew")
        self.translatable_widgets["master_tile_crop_percent_note"] = crop_percent_note
        crop_opt_row += 1
        # --- FIN  CADRE DE ROGNAGE ---

        # --- Options d'Assemblage Final ---
        final_assembly_options_frame = ttk.LabelFrame(self.scrollable_content_frame, text="", padding="10")
        # ... (contenu de final_assembly_options_frame) ...
        final_assembly_options_frame.pack(fill=tk.X, pady=(0,10), padx=0) # Changé pady top à 0
        self.translatable_widgets["final_assembly_options_frame_title"] = final_assembly_options_frame
        final_assembly_options_frame.columnconfigure(1, weight=1)
        asm_opt_row = 0
        self.final_assembly_method_label = ttk.Label(final_assembly_options_frame, text="")
        self.final_assembly_method_label.grid(row=asm_opt_row, column=0, padx=5, pady=5, sticky="w"); self.translatable_widgets["final_assembly_method_label"] = self.final_assembly_method_label
        self.final_assembly_method_combo = ttk.Combobox(final_assembly_options_frame, values=[], state="readonly", width=40)
        self.final_assembly_method_combo.grid(row=asm_opt_row, column=1, padx=5, pady=5, sticky="ew")
        self.final_assembly_method_combo.bind("<<ComboboxSelected>>", lambda e, c=self.final_assembly_method_combo, v=self.final_assembly_method_var, k_list=self.assembly_method_keys, p="assembly_method": self._combo_to_key(e, c, v, k_list, p)); asm_opt_row += 1

        gpu_chk = ttk.Checkbutton(
            final_assembly_options_frame,
            text=self._tr("use_gpu_phase5", "Use NVIDIA GPU for Phase 5"),
            variable=self.use_gpu_phase5_var,
        )
        gpu_chk.grid(row=asm_opt_row, column=0, sticky="w", padx=5, pady=3, columnspan=2)
        asm_opt_row += 1

        ttk.Label(
            final_assembly_options_frame,
            text=self._tr("gpu_selector_label", "GPU selector:")
        ).grid(row=asm_opt_row, column=0, sticky="e", padx=5, pady=2)
        names = [d for d, _ in self._gpus]
        self.gpu_selector_var.set(names[0] if names else "")
        self.gpu_selector_cb = ttk.Combobox(
            final_assembly_options_frame,
            textvariable=self.gpu_selector_var,
            values=names,
            state="readonly",
            width=30,
        )
        self.gpu_selector_cb.grid(row=asm_opt_row, column=1, sticky="w", padx=5, pady=2)
        self._gpu_selector_label = final_assembly_options_frame.grid_slaves(row=asm_opt_row, column=0)[0]
        self.translatable_widgets["gpu_selector_label"] = self._gpu_selector_label
        self._gpu_selector_label.grid_remove()
        self.gpu_selector_cb.grid_remove()
        asm_opt_row += 1

        def on_gpu_check(*_):
            if self.use_gpu_phase5_var.get():
                self._gpu_selector_label.grid()
                self.gpu_selector_cb.grid()
            else:
                self._gpu_selector_label.grid_remove()
                self.gpu_selector_cb.grid_remove()

        self.use_gpu_phase5_var.trace_add("write", on_gpu_check)
        on_gpu_check()

        self.memmap_frame = ttk.LabelFrame(self.scrollable_content_frame, text=self._tr("gui_memmap_title", "Options memmap (coadd)"))
        self.memmap_frame.pack(fill=tk.X, pady=(0,10))
        self.memmap_frame.columnconfigure(1, weight=1)
        ttk.Checkbutton(self.memmap_frame, text=self._tr("gui_memmap_enable", "Use disk memmap"), variable=self.use_memmap_var).grid(row=0, column=0, sticky="w", padx=5, pady=3)
        ttk.Label(self.memmap_frame, text=self._tr("gui_memmap_dir", "Memmap Folder")).grid(row=1, column=0, sticky="e", padx=5, pady=3)
        ttk.Entry(self.memmap_frame, textvariable=self.mm_dir_var, width=45).grid(row=1, column=1, sticky="we", padx=5, pady=3)
        ttk.Button(self.memmap_frame, text="…", command=self._browse_mm_dir).grid(row=1, column=2, padx=5, pady=3)
        ttk.Checkbutton(self.memmap_frame, text=self._tr("gui_memmap_cleanup", "Delete *.dat when finished"), variable=self.cleanup_memmap_var).grid(row=2, column=0, sticky="w", padx=5, pady=3)
        ttk.Checkbutton(self.memmap_frame, text=self._tr("gui_auto_limit_frames", "Auto limit frames per master tile (system stability)"), variable=self.auto_limit_frames_var).grid(row=3, column=0, sticky="w", padx=5, pady=3, columnspan=2)
        self._on_assembly_method_change()
        

        # --- Launch Button, Progress Bar, Log Frame ---
        self.launch_button = ttk.Button(self.scrollable_content_frame, text="", command=self._start_processing, style="Accent.TButton")
        # ... (contenu launch_button, progress_info_frame, log_frame) ...
        self.launch_button.pack(pady=15, ipady=5); self.translatable_widgets["launch_button"] = self.launch_button
        if not ZEMOSAIC_WORKER_AVAILABLE: self.launch_button.config(state=tk.DISABLED)
        try: style = ttk.Style(); style.configure("Accent.TButton", font=('Segoe UI', 10, 'bold'), padding=5)
        except tk.TclError: print("AVERT GUI: Style 'Accent.TButton' non disponible.")

        progress_info_frame = ttk.Frame(self.scrollable_content_frame, padding=(0, 5, 0, 0))
        progress_info_frame.pack(fill=tk.X, pady=(5,0))
        self.progress_bar_widget = ttk.Progressbar(progress_info_frame, orient="horizontal", length=100, mode="determinate", variable=self.progress_bar_var)
        self.progress_bar_widget.pack(fill=tk.X, expand=True, padx=5, pady=(0,3))
        time_display_subframe = ttk.Frame(progress_info_frame)
        time_display_subframe.pack(fill=tk.X, padx=5)
        ttk.Label(time_display_subframe, text="").pack(side=tk.LEFT, padx=(0,2)); self.translatable_widgets["eta_text_label"] = time_display_subframe.pack_slaves()[0]
        self.eta_label_widget = ttk.Label(time_display_subframe, textvariable=self.eta_var, font=("Segoe UI", 9, "bold"), width=10)
        self.eta_label_widget.pack(side=tk.LEFT, padx=(0,15))
        ttk.Label(time_display_subframe, text="").pack(side=tk.LEFT, padx=(0,2)); self.translatable_widgets["elapsed_text_label"] = time_display_subframe.pack_slaves()[2]
        self.elapsed_time_label_widget = ttk.Label(time_display_subframe, textvariable=self.elapsed_time_var, font=("Segoe UI", 9, "bold"), width=10)
        self.elapsed_time_label_widget.pack(side=tk.LEFT, padx=(0,10))
        self.tile_count_text_label_widget = ttk.Label(time_display_subframe, text="") 
        self.tile_count_text_label_widget.pack(side=tk.LEFT, padx=(0,2))
        self.translatable_widgets["tiles_text_label"] = self.tile_count_text_label_widget # Pour la traduction "Tuiles :"

        self.master_tile_count_label_widget = ttk.Label(time_display_subframe,textvariable=self.master_tile_count_var,font=("Segoe UI", 9, "bold"), width=12 )# Un peu plus large pour "XXX / XXX"    
        self.master_tile_count_label_widget.pack(side=tk.LEFT, padx=(0,5))
        log_frame = ttk.LabelFrame(self.scrollable_content_frame, text="", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(5,5)); self.translatable_widgets["log_frame_title"] = log_frame
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=10, state=tk.DISABLED, font=("Consolas", 9))
        log_scrollbar_y_text = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar_x_text = ttk.Scrollbar(log_frame, orient="horizontal", command=self.log_text.xview)
        self.log_text.config(yscrollcommand=log_scrollbar_y_text.set, xscrollcommand=log_scrollbar_x_text.set)
        log_scrollbar_y_text.pack(side=tk.RIGHT, fill=tk.Y); log_scrollbar_x_text.pack(side=tk.BOTTOM, fill=tk.X)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)


        self.scrollable_content_frame.update_idletasks()
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

        # Ces appels sont importants pour l'état initial et la traduction
        # Ils sont déjà dans __init__ après _build_ui, mais un after_idle ici peut être une sécurité.
        # self.root.after_idle(self._update_ui_language) # Déjà appelé depuis __init__
        # self.root.after_idle(self._update_assembly_dependent_options) # Déjà appelé
        # self.root.after_idle(self._update_rejection_params_state) # Déjà appelé

    def _update_ui_language(self):
        if not self.localizer:
            # print("DEBUG GUI: Localizer non disponible dans _update_ui_language.")
            return
        if not (hasattr(self.root, 'winfo_exists') and self.root.winfo_exists()):
            # print("DEBUG GUI: Root window non existante dans _update_ui_language.")
            return

        self.root.title(self._tr("window_title", "ZeMosaic - Hierarchical Mosaicker"))

        # Traduction des widgets standards (Labels, Buttons, Titres de Frames, Onglets etc.)
        for key, widget_info in self.translatable_widgets.items():
            try:
                text_to_set = self._tr(key) # Le _tr gère son propre fallback
                target_widget = widget_info

                # Cas spécifique pour les onglets d'un ttk.Notebook
                if isinstance(widget_info, tuple) and len(widget_info) == 2:
                    notebook_widget, tab_index = widget_info
                    # S'assurer que le Notebook lui-même existe avant de tenter d'accéder à un onglet
                    if hasattr(notebook_widget, 'winfo_exists') and notebook_widget.winfo_exists():
                        try:
                            # Vérifier si l'onglet à cet index existe toujours
                            if tab_index < len(notebook_widget.tabs()):
                                notebook_widget.tab(tab_index, text=f" {text_to_set} ") # Ajouter des espaces pour l'esthétique
                        except tk.TclError:
                            pass # L'onglet a pu être détruit ou le notebook modifié
                    continue # Passer au widget suivant après avoir traité l'onglet

                # Pour les autres widgets, s'assurer qu'ils existent
                if hasattr(target_widget, 'winfo_exists') and target_widget.winfo_exists():
                    if isinstance(target_widget, (ttk.Label, ttk.Button, ttk.Checkbutton, ttk.Radiobutton)):
                        target_widget.config(text=text_to_set)
                    elif isinstance(target_widget, ttk.LabelFrame):
                        target_widget.config(text=text_to_set)
                    # Ajoutez d'autres types de widgets ici si nécessaire
            except tk.TclError:
                # print(f"DEBUG GUI: TclError lors de la mise à jour du widget '{key}'. Il a pu être détruit.")
                pass # Ignorer si le widget n'existe plus
            except Exception as e:
                print(f"DEBUG GUI: Erreur inattendue lors de la mise à jour du widget standard '{key}': {e}")

        # Utiliser _refresh_combobox pour mettre à jour TOUS les comboboxes
        # (Cette partie est déjà correcte dans votre code)
        if hasattr(self, 'norm_method_combo'):
            self._refresh_combobox(self.norm_method_combo, self.stacking_normalize_method_var, self.norm_method_keys, "norm_method")
        if hasattr(self, 'weight_method_combo'):
            self._refresh_combobox(self.weight_method_combo, self.stacking_weighting_method_var, self.weight_method_keys, "weight_method")
        if hasattr(self, 'reject_algo_combo'):
            self._refresh_combobox(self.reject_algo_combo, self.stacking_rejection_algorithm_var, self.reject_algo_keys, "reject_algo")
        if hasattr(self, 'final_combine_combo'):
            self._refresh_combobox(self.final_combine_combo, self.stacking_final_combine_method_var, self.combine_method_keys, "combine_method")
        # --- RAFRAICHISSEMENT DU NOUVEAU COMBOBOX D'ASSEMBLAGE ---
        if hasattr(self, 'final_assembly_method_combo'):
            self._refresh_combobox(self.final_assembly_method_combo, self.final_assembly_method_var, self.assembly_method_keys, "assembly_method")
        # ---  ---
        # Mise à jour des textes ETA et Temps Écoulé (déjà correct)
        if not self.is_processing:
            if hasattr(self, 'eta_label_widget') and self.eta_label_widget.winfo_exists():
                try:
                    self.eta_var.set(self._tr("initial_eta_value", "--:--:--"))
                except tk.TclError: pass
            if hasattr(self, 'elapsed_time_label_widget') and self.elapsed_time_label_widget.winfo_exists():
                try:
                    self.elapsed_time_var.set(self._tr("initial_elapsed_time", "00:00:00"))
                except tk.TclError: pass

        # S'assurer que l'état des paramètres de rejet est mis à jour (déjà correct)
        if hasattr(self, '_update_rejection_params_state'):
            try:
                if self.root.winfo_exists():
                    self.root.after_idle(self._update_rejection_params_state)
            except Exception as e_uras:
                print(f"DEBUG GUI: Erreur appel _update_rejection_params_state via after_idle: {e_uras}")


    def _update_crop_options_state(self, *args):
        """Active ou désactive le spinbox de pourcentage de rognage."""
        if not all(hasattr(self, attr) for attr in [
            'apply_master_tile_crop_var',
            'crop_percent_spinbox'
        ]):
            return # Widgets pas encore prêts

        try:
            if self.apply_master_tile_crop_var.get():
                self.crop_percent_spinbox.config(state=tk.NORMAL)
            else:
                self.crop_percent_spinbox.config(state=tk.DISABLED)
        except tk.TclError:
            pass # Widget peut avoir été détruit

    def _on_assembly_method_change(self, *args):
        method = self.final_assembly_method_var.get()
        try:
            if method == "reproject_coadd":
                if not self.memmap_frame.winfo_ismapped():
                    self.memmap_frame.pack(fill=tk.X, pady=(0,10))
            else:
                if self.memmap_frame.winfo_ismapped():
                    self.memmap_frame.pack_forget()
        except tk.TclError:
            pass

    def _update_rejection_params_state(self, event=None):
        """
        Active ou désactive les widgets de paramètres de rejet (Kappa, Winsor)
        en fonction de l'algorithme de rejet sélectionné.
        """
        # Sécurité au cas où la méthode serait appelée avant que tout soit initialisé
        if not hasattr(self, 'stacking_rejection_algorithm_var') or \
           not hasattr(self, 'kappa_low_spinbox') or \
           not hasattr(self, 'kappa_high_spinbox') or \
           not hasattr(self, 'winsor_limits_entry'):
            # print("DEBUG GUI (_update_rejection_params_state): Un des widgets ou variables requis n'est pas encore initialisé.")
            return

        selected_algo = self.stacking_rejection_algorithm_var.get()

        # Déterminer l'état des champs de paramètres en fonction de l'algorithme choisi
        if selected_algo == "kappa_sigma":
            kappa_params_state = tk.NORMAL
            winsor_params_state = tk.DISABLED
        elif selected_algo == "winsorized_sigma_clip":
            kappa_params_state = tk.NORMAL  # Kappa est utilisé APRES la winsorisation
            winsor_params_state = tk.NORMAL
        elif selected_algo == "linear_fit_clip":
            # Pour l'instant, on désactive tout, car les paramètres spécifiques ne sont pas définis.
            # Si Linear Fit Clip utilisait Kappa, on mettrait kappa_params_state = tk.NORMAL
            kappa_params_state = tk.DISABLED 
            winsor_params_state = tk.DISABLED
        elif selected_algo == "none":
            kappa_params_state = tk.DISABLED
            winsor_params_state = tk.DISABLED
        else: # Algorithme inconnu ou non géré, désactiver tout par sécurité
            kappa_params_state = tk.DISABLED
            winsor_params_state = tk.DISABLED
            if selected_algo: # Pour ne pas logger si la variable est vide au tout début
                print(f"AVERT GUI: Algorithme de rejet inconnu '{selected_algo}' dans _update_rejection_params_state.")

        # Appliquer les états aux widgets Spinbox et Entry
        # S'assurer que les widgets existent avant de configurer leur état
        if hasattr(self.kappa_low_spinbox, 'winfo_exists') and self.kappa_low_spinbox.winfo_exists():
            try:
                self.kappa_low_spinbox.config(state=kappa_params_state)
            except tk.TclError: pass # Widget peut avoir été détruit

        if hasattr(self.kappa_high_spinbox, 'winfo_exists') and self.kappa_high_spinbox.winfo_exists():
            try:
                self.kappa_high_spinbox.config(state=kappa_params_state)
            except tk.TclError: pass

        if hasattr(self.winsor_limits_entry, 'winfo_exists') and self.winsor_limits_entry.winfo_exists():
            try:
                self.winsor_limits_entry.config(state=winsor_params_state)
            except tk.TclError: pass

        # Optionnel : Griser également les labels associés aux paramètres désactivés
        # Cela nécessite que les labels soient accessibles, par exemple via self.translatable_widgets
        # Exemple pour les labels Kappa (à adapter si vous voulez cette fonctionnalité) :
        # kappa_labels_to_update = ["stacking_kappa_low_label", "stacking_kappa_high_label"]
        # for label_key in kappa_labels_to_update:
        #     if label_key in self.translatable_widgets:
        #         label_widget = self.translatable_widgets[label_key]
        #         if hasattr(label_widget, 'winfo_exists') and label_widget.winfo_exists():
        #             try:
        #                 # Note: ttk.Label n'a pas d'option 'state' standard comme les widgets d'entrée.
        #                 # Pour griser, on change la couleur du texte (pas idéal pour tous les thèmes).
        #                 # Une meilleure approche serait d'utiliser un ttk.Label stylé si le thème le supporte.
        #                 # Pour la simplicité, on peut se contenter de désactiver les champs d'entrée.
        #                 # label_widget.config(foreground="gray" if kappa_params_state == tk.DISABLED else "black") # Exemple
        #                 pass # Laisser les labels toujours actifs pour la simplicité
        #             except tk.TclError: pass
        
        # Idem pour les labels Winsor si besoin.



    def _load_default_paths_for_dev(self): pass

    def _browse_input_dir(self):
        dir_path = filedialog.askdirectory(title=self._tr("browse_input_title", "Select Input Folder (Raws)"))
        if dir_path: self.input_dir_var.set(dir_path)

    def _browse_output_dir(self):
        dir_path = filedialog.askdirectory(title=self._tr("browse_output_title", "Select Output Folder"))
        if dir_path: self.output_dir_var.set(dir_path)

    def _browse_mm_dir(self):
        dir_path = filedialog.askdirectory(title=self._tr("gui_memmap_dir", "Memmap Folder"))
        if dir_path:
            self.mm_dir_var.set(dir_path)

    def _browse_and_save_astap_exe(self):
        title = self._tr("select_astap_exe_title", "Select ASTAP Executable")
        if ZEMOSAIC_CONFIG_AVAILABLE and zemosaic_config:
            new_path = zemosaic_config.ask_and_set_astap_path(self.config)
            if new_path: self.astap_exe_path_var.set(new_path)
            elif not self.config.get("astap_executable_path"):
                messagebox.showwarning(self._tr("astap_path_title", "ASTAP Path"),
                                       self._tr("astap_exe_not_set_warning", "ASTAP executable path is not set."),
                                       parent=self.root)
        else:
            filetypes_loc = [(self._tr("executable_files", "Executable Files"), "*.exe"), (self._tr("all_files", "All Files"), "*.*")] if os.name == 'nt' else [(self._tr("all_files", "All Files"), "*")]
            exe_path = filedialog.askopenfilename(title=self._tr("select_astap_exe_no_save_title", "Select ASTAP Executable (Not Saved)"), filetypes=filetypes_loc)
            if exe_path: self.astap_exe_path_var.set(exe_path)

    def _browse_and_save_astap_data_dir(self):
        title = self._tr("select_astap_data_dir_title", "Select ASTAP Data Directory")
        if ZEMOSAIC_CONFIG_AVAILABLE and zemosaic_config:
            new_data_dir = zemosaic_config.ask_and_set_astap_data_dir_path(self.config)
            if new_data_dir: self.astap_data_dir_var.set(new_data_dir)
            elif not self.config.get("astap_data_directory_path"):
                messagebox.showwarning(self._tr("astap_data_dir_title", "ASTAP Data Directory"),
                                       self._tr("astap_data_dir_not_set_warning", "ASTAP data directory path is not set."),
                                       parent=self.root)
        else:
            dir_path = filedialog.askdirectory(title=self._tr("select_astap_data_no_save_title", "Select ASTAP Data Directory (Not Saved)"))
            if dir_path: self.astap_data_dir_var.set(dir_path)
            
    def _browse_astap_data_dir(self): # Fallback non-saving browse
        dir_path = filedialog.askdirectory(title=self._tr("select_astap_data_title_simple", "Select ASTAP Data Directory"))
        if dir_path: self.astap_data_dir_var.set(dir_path)



# DANS zemosaic_gui.py
# DANS la classe ZeMosaicGUI

    def _log_message(self, message_key_or_raw, progress_value=None, level="INFO", **kwargs): # Ajout de **kwargs
        if not hasattr(self.root, 'winfo_exists') or not self.root.winfo_exists(): return
        
        log_text_content = ""
        is_control_message = False # Pour les messages ETA/CHRONO

        # --- Gestion des messages de contrôle spéciaux (ETA, CHRONO, ET MAINTENANT TILE_COUNT) ---
        if isinstance(message_key_or_raw, str):
            if message_key_or_raw.startswith("ETA_UPDATE:"):
                eta_string_from_worker = message_key_or_raw.split(":", 1)[1]
                if hasattr(self, 'eta_var') and self.eta_var:
                    def update_eta_label():
                        if hasattr(self.eta_var,'set') and callable(self.eta_var.set):
                            try: self.eta_var.set(eta_string_from_worker)
                            except tk.TclError: pass 
                    if self.root.winfo_exists(): self.root.after_idle(update_eta_label)
                is_control_message = True
            elif message_key_or_raw == "CHRONO_START_REQUEST":
                if self.root.winfo_exists(): self.root.after_idle(self._start_gui_chrono)
                is_control_message = True
            elif message_key_or_raw == "CHRONO_STOP_REQUEST":
                if self.root.winfo_exists(): self.root.after_idle(self._stop_gui_chrono)
                is_control_message = True
            # --- AJOUT POUR INTERCEPTER MASTER_TILE_COUNT_UPDATE ---
            elif message_key_or_raw.startswith("MASTER_TILE_COUNT_UPDATE:"):
                tile_count_string = message_key_or_raw.split(":", 1)[1]
                if hasattr(self, 'master_tile_count_var') and self.master_tile_count_var:
                    def update_tile_count_label(): # Closure pour capturer tile_count_string
                        if hasattr(self.master_tile_count_var, 'set') and callable(self.master_tile_count_var.set):
                            try: self.master_tile_count_var.set(tile_count_string)
                            except tk.TclError: pass # Ignorer si fenêtre détruite
                    if self.root.winfo_exists(): self.root.after_idle(update_tile_count_label)
                is_control_message = True
            # --- FIN AJOUT ---
        
        if is_control_message:
            return # Ne pas traiter plus loin ces messages de contrôle

        # --- Préparation du contenu textuel du log ---
        # Niveaux pour lesquels on essaie de traduire `message_key_or_raw` comme une clé
        user_facing_levels = ["INFO", "WARN", "ERROR", "SUCCESS"] 
        
        # S'assurer que level est une chaîne pour la comparaison
        current_level_str = str(level).upper() if isinstance(level, str) else "INFO"

        if current_level_str in user_facing_levels:
            # Tenter de traduire `message_key_or_raw` comme une clé, en passant les kwargs
            log_text_content = self._tr(message_key_or_raw, default_text=str(message_key_or_raw), **kwargs)
            # Si la traduction a échoué et retourné "_clé_", et que default_text était la clé,
            # alors log_text_content est "_clé_". On préfère la clé brute dans ce cas si pas de formatage.
            # Si des kwargs sont présents, on suppose que la clé est valide et doit être formatée.
            if log_text_content == f"_{str(message_key_or_raw)}_" and not kwargs:
                log_text_content = str(message_key_or_raw)
        else: # Pour DEBUG_DETAIL, INFO_DETAIL, etc., on affiche le message tel quel
              # mais on essaie de le formater avec kwargs s'ils sont fournis.
            log_text_content = str(message_key_or_raw)
            if kwargs:
                try:
                    log_text_content = log_text_content.format(**kwargs)
                except KeyError:
                     # Si la chaîne brute n'a pas les placeholders, on la garde telle quelle
                    print(f"WARN (_log_message): Tentative de formater message brut '{log_text_content}' avec kwargs {kwargs} a échoué (KeyError).")
                    pass 
                except Exception as e_fmt_raw:
                    print(f"WARN (_log_message): Erreur formatage message brut '{log_text_content}' avec kwargs: {e_fmt_raw}")


        # Nettoyer le préfixe "[Z...]" s'il vient du worker pour les logs techniques
        # Ce nettoyage est fait APRÈS la traduction/formatage pour ne pas interférer.
        if isinstance(log_text_content, str) and (log_text_content.startswith("  [Z") or log_text_content.startswith("      [Z")):
            try:
                log_text_content = log_text_content.split("] ", 1)[1] 
            except IndexError:
                pass # Garder le message original si le split échoue
        
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {log_text_content.strip()}\n"
        
        # --- Détermination du tag de couleur ---
        tag_name = None
        if current_level_str == "ERROR": tag_name = "error_log"
        elif current_level_str == "WARN": tag_name = "warn_log"
        elif current_level_str == "SUCCESS": tag_name = "success_log"
        elif current_level_str == "DEBUG_DETAIL": tag_name = "debug_detail_log"
        elif current_level_str == "INFO_DETAIL": tag_name = "info_detail_log"
        # Les niveaux ETA_LEVEL et CHRONO_LEVEL n'ont pas de tag spécifique ici, ils sont interceptés avant.

        # --- Mise à jour des éléments GUI via after_idle ---
        def update_gui_elements():
            # Mise à jour du log texte
            if hasattr(self.log_text, 'winfo_exists') and self.log_text.winfo_exists():
                try:
                    self.log_text.config(state=tk.NORMAL)
                    if tag_name:
                        self.log_text.insert(tk.END, formatted_message, tag_name)
                    else:
                        self.log_text.insert(tk.END, formatted_message)
                    self.log_text.see(tk.END)
                    self.log_text.config(state=tk.DISABLED)
                except tk.TclError: pass # Widget peut être détruit

            # Mise à jour de la barre de progression
            if progress_value is not None and hasattr(self, 'progress_bar_widget') and self.progress_bar_widget.winfo_exists():
                try:
                    current_progress = float(progress_value)
                    current_progress = max(0.0, min(100.0, current_progress))
                    self.progress_bar_var.set(current_progress)
                except (ValueError, TypeError) as e_prog:
                    # Utiliser le logger de la classe GUI si disponible, sinon print
                    log_func = getattr(self, 'logger.error', print) if hasattr(self, 'logger') else print
                    log_func(f"ERREUR (ZeMosaicGUI._log_message): Valeur de progression invalide: {progress_value}, Erreur: {e_prog}")
                except tk.TclError: pass # Widget peut être détruit
        
        if self.root.winfo_exists():
            self.root.after_idle(update_gui_elements)

        # Configuration des tags de couleur (faite une seule fois)
        if not hasattr(self, '_log_tags_configured'):
            try:
                if hasattr(self.log_text, 'winfo_exists') and self.log_text.winfo_exists():
                    self.log_text.tag_configure("error_log", foreground="#E53935", font=("Consolas", 9, "bold"))
                    self.log_text.tag_configure("warn_log", foreground="#FF8F00", font=("Consolas", 9))
                    self.log_text.tag_configure("success_log", foreground="#4CAF50", font=("Consolas", 9, "bold"))
                    self.log_text.tag_configure("debug_detail_log", foreground="gray50", font=("Consolas", 9))
                    self.log_text.tag_configure("info_detail_log", foreground="gray30", font=("Consolas", 9))
                    self._log_tags_configured = True
            except tk.TclError:
                pass # Ignorer si log_text n'est pas encore prêt lors d'un appel très précoce





    def _start_gui_chrono(self):
        if hasattr(self, '_chrono_after_id') and self._chrono_after_id:
            try: self.root.after_cancel(self._chrono_after_id)
            except tk.TclError: pass
        self._chrono_start_time = time.monotonic()
        self.elapsed_time_var.set(self._tr("initial_elapsed_time", "00:00:00"))
        self._update_gui_chrono()
        print("DEBUG GUI: Chronomètre démarré.")
        
    def _update_gui_chrono(self):
        if not self.is_processing or self._chrono_start_time is None: 
            if hasattr(self, '_chrono_after_id') and self._chrono_after_id:
                try: self.root.after_cancel(self._chrono_after_id)
                except tk.TclError: pass
            self._chrono_after_id = None; return
        elapsed_seconds = time.monotonic() - self._chrono_start_time
        h, rem = divmod(int(elapsed_seconds), 3600); m, s = divmod(rem, 60)
        try:
            if hasattr(self, 'elapsed_time_var') and self.elapsed_time_var:
                self.elapsed_time_var.set(f"{h:02d}:{m:02d}:{s:02d}")
        except tk.TclError: 
            if hasattr(self, '_chrono_after_id') and self._chrono_after_id: self.root.after_cancel(self._chrono_after_id)
            self._chrono_after_id = None; return
        if hasattr(self.root, 'after') and self.root.winfo_exists():
            self._chrono_after_id = self.root.after(1000, self._update_gui_chrono)

    def _stop_gui_chrono(self):
        if hasattr(self, '_chrono_after_id') and self._chrono_after_id:
            try: self.root.after_cancel(self._chrono_after_id)
            except tk.TclError: pass
        self._chrono_after_id = None
        print("DEBUG GUI: Chronomètre arrêté.")

    def on_worker_progress(self, stage: str, current: int, total: int):
        """Handle progress updates for a specific processing stage."""
        if stage not in self._stage_times:
            self._stage_times[stage] = {
                'start': time.monotonic(),
                'last': time.monotonic(),
                'steps': []
            }
        else:
            now = time.monotonic()
            last = self._stage_times[stage]['last']
            self._stage_times[stage]['steps'].append(now - last)
            self._stage_times[stage]['last'] = now

        percent = (current / total * 100.0) if total else 0.0
        try:
            if hasattr(self, 'progress_bar_var'):
                self.progress_bar_var.set(percent)
        except tk.TclError:
            pass

        times = self._stage_times[stage]
        if times['steps']:
            avg = sum(times['steps']) / len(times['steps'])
            remaining = max(0.0, (total - current) * avg)
            h, rem = divmod(int(remaining), 3600)
            m, s = divmod(rem, 60)
            try:
                if hasattr(self, 'eta_var') and self.eta_var:
                    self.eta_var.set(f"{h:02d}:{m:02d}:{s:02d}")
            except tk.TclError:
                pass
        





    def _start_processing(self):
        if self.is_processing: 
            messagebox.showwarning(self._tr("processing_in_progress_title"), 
                                   self._tr("processing_already_running_warning"), 
                                   parent=self.root)
            return
        if not ZEMOSAIC_WORKER_AVAILABLE or not run_hierarchical_mosaic: 
            messagebox.showerror(self._tr("critical_error_title"), 
                                 self._tr("worker_module_unavailable_error"), 
                                 parent=self.root)
            return

        # 1. RÉCUPÉRER TOUTES les valeurs des variables Tkinter
        input_dir = self.input_dir_var.get()
        output_dir = self.output_dir_var.get()
        astap_exe = self.astap_exe_path_var.get() 
        astap_data = self.astap_data_dir_var.get()
        
        try:
            astap_radius_val = self.astap_search_radius_var.get()
            astap_downsample_val = self.astap_downsample_var.get()
            astap_sensitivity_val = self.astap_sensitivity_var.get()
            cluster_thresh_val = self.cluster_threshold_var.get()
            
            stack_norm_method = self.stacking_normalize_method_var.get()
            stack_weight_method = self.stacking_weighting_method_var.get()
            stack_reject_algo = self.stacking_rejection_algorithm_var.get()
            stack_kappa_low = self.stacking_kappa_low_var.get()
            stack_kappa_high = self.stacking_kappa_high_var.get()
            stack_winsor_limits_str = self.stacking_winsor_limits_str_var.get()
            stack_final_combine = self.stacking_final_combine_method_var.get()

            apply_radial_weight_val = self.apply_radial_weight_var.get()
            radial_feather_fraction_val = self.radial_feather_fraction_var.get()
            min_radial_weight_floor_val = self.min_radial_weight_floor_var.get()
            radial_shape_power_val = self.config.get("radial_shape_power", 2.0) # Toujours depuis config pour l'instant
            
            final_assembly_method_val = self.final_assembly_method_var.get()
            num_base_workers_gui_val = self.num_workers_var.get()

            self.solver_settings.solver_choice = self.solver_choice_var.get()
            self.solver_settings.api_key = self.astrometry_api_key_var.get().strip()
            self.solver_settings.timeout = self.astrometry_timeout_var.get()
            self.solver_settings.downsample = self.astrometry_downsample_var.get()
            self.solver_settings.force_lum = bool(self.force_lum_var.get())
            try:
                self.solver_settings.save_default()
            except Exception:
                pass

            # --- RÉCUPÉRATION DES NOUVELLES VALEURS POUR LE ROGNAGE ---
            apply_master_tile_crop_val = self.apply_master_tile_crop_var.get()
            master_tile_crop_percent_val = self.master_tile_crop_percent_var.get()
            # --- FIN RÉCUPÉRATION ROGNAGE ---
            
        except tk.TclError as e: 
            messagebox.showerror(self._tr("param_error_title"), 
                                 self._tr("invalid_param_value_error", error=e), 
                                 parent=self.root)
            return

        # 2. VALIDATIONS (chemins, etc.)
        # ... (section de validation inchangée pour l'instant)
        if not (input_dir and os.path.isdir(input_dir)): 
            messagebox.showerror(self._tr("error_title"), self._tr("invalid_input_folder_error"), parent=self.root); return
        if not output_dir: 
            messagebox.showerror(self._tr("error_title"), self._tr("missing_output_folder_error"), parent=self.root); return
        try: 
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e: 
            messagebox.showerror(self._tr("error_title"), self._tr("output_folder_creation_error", error=e), parent=self.root); return
        if not (astap_exe and os.path.isfile(astap_exe)): 
            messagebox.showerror(self._tr("error_title"), self._tr("invalid_astap_exe_error"), parent=self.root); return
        if not (astap_data and os.path.isdir(astap_data)): 
            if not messagebox.askokcancel(self._tr("astap_data_dir_title", "ASTAP Data Directory"),
                                          self._tr("astap_data_dir_missing_or_invalid_continue_q", 
                                                   path=astap_data,
                                                   default_path=self.config.get("astap_data_directory_path","")),
                                          icon='warning', parent=self.root):
                return


        # 3. PARSING et VALIDATION des limites Winsor (inchangé)
        parsed_winsor_limits = (0.05, 0.05) 
        if stack_reject_algo == "winsorized_sigma_clip":
            try:
                low_str, high_str = stack_winsor_limits_str.split(',')
                parsed_winsor_limits = (float(low_str.strip()), float(high_str.strip()))
                if not (0.0 <= parsed_winsor_limits[0] < 0.5 and 0.0 <= parsed_winsor_limits[1] < 0.5 and (parsed_winsor_limits[0] + parsed_winsor_limits[1]) < 1.0):
                    raise ValueError(self._tr("winsor_limits_range_error_detail"))
            except ValueError as e_winsor:
                messagebox.showerror(self._tr("param_error_title"), 
                                     self._tr("invalid_winsor_limits_error", error=e_winsor),
                                     parent=self.root)
                return
        
        # 4. DÉMARRAGE du traitement
        # Remise à zéro du compteur master-tiles
        if hasattr(self, "master_tile_count_var"):
            self.master_tile_count_var.set("")
        self.is_processing = True
        self.launch_button.config(state=tk.DISABLED)
        self.log_text.config(state=tk.NORMAL); self.log_text.delete(1.0, tk.END); self.log_text.config(state=tk.DISABLED)
        
        self._log_message("CHRONO_START_REQUEST", None, "CHRONO_LEVEL")
        self._log_message("log_key_processing_started", level="INFO")
        # ... (autres logs d'info) ...

        # -- Gestion du dossier memmap par défaut --
        memmap_dir = self.mm_dir_var.get().strip()
        if self.use_memmap_var.get() and not memmap_dir:
            memmap_dir = self.output_dir_var.get().strip()
            self.mm_dir_var.set(memmap_dir)
            self._log_message(
                f"[INFO] Aucun dossier memmap défini. Utilisation du dossier de sortie: {memmap_dir}",
                level="INFO",
            )

        self.config["winsor_worker_limit"] = self.winsor_workers_var.get()
        self.config["max_raw_per_master_tile"] = self.max_raw_per_tile_var.get()

        self.config["use_gpu_phase5"] = self.use_gpu_phase5_var.get()
        sel = self.gpu_selector_var.get()
        gpu_id = None
        for disp, idx in self._gpus:
            if disp == sel:
                self.config["gpu_selector"] = disp
                self.config["gpu_id_phase5"] = idx
                gpu_id = idx
                break
        if ZEMOSAIC_CONFIG_AVAILABLE and zemosaic_config:
            zemosaic_config.save_config(self.config)

        worker_args = (
            input_dir, output_dir, astap_exe, astap_data,
            astap_radius_val, astap_downsample_val, astap_sensitivity_val,
            cluster_thresh_val,
            stack_norm_method,
            stack_weight_method,
            stack_reject_algo,
            stack_kappa_low,
            stack_kappa_high,
            parsed_winsor_limits, 
            stack_final_combine,
            apply_radial_weight_val,
            radial_feather_fraction_val,
            radial_shape_power_val,
            min_radial_weight_floor_val,
            final_assembly_method_val,
            num_base_workers_gui_val,
            # --- NOUVEAUX ARGUMENTS POUR LE ROGNAGE ---
            apply_master_tile_crop_val,
            master_tile_crop_percent_val,
            self.save_final_uint16_var.get(),
            self.use_memmap_var.get(),
            memmap_dir,
            self.cleanup_memmap_var.get(),
            self.config.get("assembly_process_workers", 0),
            self.auto_limit_frames_var.get(),
            self.winsor_workers_var.get(),
            self.max_raw_per_tile_var.get(),
            self.use_gpu_phase5_var.get(),
            gpu_id,
            asdict(self.solver_settings)
            # --- FIN NOUVEAUX ARGUMENTS ---
        )
        
        self.progress_queue = multiprocessing.Queue()
        self.worker_process = multiprocessing.Process(
            target=run_hierarchical_mosaic_process,
            args=(self.progress_queue,) + worker_args[:-1],
            kwargs={"solver_settings_dict": worker_args[-1]},
            daemon=True,
            name="ZeMosaicWorkerProcess",
        )
        self.worker_process.start()

        if hasattr(self.root, 'winfo_exists') and self.root.winfo_exists():
            self.root.after(100, self._poll_worker_queue)

    



    def _poll_worker_queue(self):
        if not (hasattr(self.root, 'winfo_exists') and self.root.winfo_exists()):
            if self.is_processing:
                self.is_processing = False
                if self.worker_process and self.worker_process.is_alive():
                    self.worker_process.terminate()
            return

        if self.progress_queue:
            while True:
                try:
                    msg_key, prog, lvl, kwargs = self.progress_queue.get_nowait()
                except Exception:
                    break
                if msg_key == "STAGE_PROGRESS":
                    stage, cur, tot = prog, lvl, kwargs.get('total', 0)
                    self.on_worker_progress(stage, cur, tot)
                    continue
                if msg_key == "PROCESS_DONE":
                    if self.worker_process:
                        self.worker_process.join(timeout=0.1)
                        self.worker_process = None
                    continue
                self._log_message(msg_key, prog, lvl, **kwargs)

        if self.worker_process and self.worker_process.is_alive():
            self.root.after(100, self._poll_worker_queue)
            return

        self._log_message("CHRONO_STOP_REQUEST", None, "CHRONO_LEVEL")
        self.is_processing = False
        if hasattr(self, 'launch_button') and self.launch_button.winfo_exists():
            self.launch_button.config(state=tk.NORMAL)
        if self.root.winfo_exists():
            self._log_message("log_key_processing_finished", level="INFO")
            final_message = self._tr("msg_processing_completed")
            messagebox.showinfo(self._tr("dialog_title_completed"), final_message, parent=self.root)
            # Nettoyage du compteur master-tiles affiché
            if hasattr(self, "master_tile_count_var"):
                self.master_tile_count_var.set("")
            output_dir_final = self.output_dir_var.get()
            if output_dir_final and os.path.isdir(output_dir_final):
                if messagebox.askyesno(self._tr("q_open_output_folder_title"), self._tr("q_open_output_folder_msg", folder=output_dir_final), parent=self.root):
                    try:
                        if os.name == 'nt':
                            os.startfile(output_dir_final)
                        elif sys.platform == 'darwin':
                            subprocess.Popen(['open', output_dir_final])
                        else:
                            subprocess.Popen(['xdg-open', output_dir_final])
                    except Exception as e_open_dir:
                        self._log_message(self._tr("log_key_error_opening_folder", error=e_open_dir), level="ERROR")
                        messagebox.showerror(
                            self._tr("error_title"),
                            self._tr("error_cannot_open_folder", error=e_open_dir),
                            parent=self.root,
                        )
                        
    def _on_closing(self):
        if self.is_processing:
            if messagebox.askokcancel(self._tr("q_quit_title"), self._tr("q_quit_while_processing_msg"), icon='warning', parent=self.root):
                self.is_processing = False
                if self.worker_process and self.worker_process.is_alive():
                    self.worker_process.terminate()
                self._stop_gui_chrono()
                self.root.destroy()
            else: return
        else: self._stop_gui_chrono(); self.root.destroy()

if __name__ == '__main__':
    root_app = tk.Tk()
    initial_localizer_main = None
    if ZEMOSAIC_LOCALIZATION_AVAILABLE and ZeMosaicLocalization: initial_localizer_main = ZeMosaicLocalization(language_code='en')
    def tr_initial_main(key, default_text=""): return initial_localizer_main.get(key, default_text) if initial_localizer_main else default_text

    if not ZEMOSAIC_WORKER_AVAILABLE:
        messagebox.showerror(tr_initial_main("critical_launch_error_title", "Critical Launch Error"),
                             tr_initial_main("worker_module_missing_critical_error", "Worker module missing."), parent=root_app)
        root_app.destroy()
    else:
        app = ZeMosaicGUI(root_app)
        root_app.mainloop()
    print("Application ZeMosaic GUI (instance directe) terminée.")