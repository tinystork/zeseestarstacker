"""
Module pour la fenêtre de configuration des solveurs astrométriques locaux.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os  # Pour les opérations sur les chemins
import platform
from zemosaic import zemosaic_config
from .ui_utils import ToolTip

class LocalSolverSettingsWindow(tk.Toplevel):
    """
    Fenêtre de dialogue pour configurer les chemins d'accès
    aux solveurs astrométriques locaux comme ASTAP ou Astrometry.net local.
    """

    def tr(self, key, default=None):
        """Shortcut to parent GUI translation."""

        return self.parent_gui.tr(key, default=default)

    def __init__(self, parent_gui):
        """
        Initialise la fenêtre de configuration des solveurs locaux.

        Args:
            parent_gui: L'instance de SeestarStackerGUI parente.
        """
        super().__init__(parent_gui.root)
        self.parent_gui = parent_gui
        # Load fallback config if the parent GUI lacks one
        if not hasattr(self.parent_gui, "config"):
            try:
                self.parent_gui.config = zemosaic_config.load_config()
            except Exception:
                self.parent_gui.config = {}
        self.withdraw()  # Cacher pendant la configuration

        self.title(self.tr("solver_config_title", default="Local Astrometry Solvers Configuration"))
        self.transient(parent_gui.root)

        print("DEBUG (LocalSolverSettingsWindow __init__): Début initialisation.") # DEBUG

        # --- Variables Tkinter pour les chemins et options ---
        
        # La valeur par défaut sera "none". Plus tard, on lira depuis self.parent_gui.settings.local_solver_preference
        # Pour l'instant, on anticipe que local_solver_preference existera.
        default_solver_choice = "none"
        if hasattr(self.parent_gui.settings, 'local_solver_preference'):
            default_solver_choice = getattr(self.parent_gui.settings, 'local_solver_preference', "none")
        else:
            print("DEBUG (LocalSolverSettingsWindow __init__): 'local_solver_preference' non trouvé dans settings, utilisation de 'none'.")
            
        self.local_solver_choice_var = tk.StringVar(value=default_solver_choice)
        print(f"DEBUG (LocalSolverSettingsWindow __init__): local_solver_choice_var initialisée à '{self.local_solver_choice_var.get()}'.") # DEBUG

        # --- MODIFIÉ : Suppression de use_local_solver_priority_var ---
        # self.use_local_solver_priority_var = tk.BooleanVar(...) # Supprimé

        self.astap_path_var = tk.StringVar(
            value=getattr(self.parent_gui.settings, 'astap_path', "")
        )
        self.astap_data_dir_var = tk.StringVar(
            value=getattr(self.parent_gui.settings, 'astap_data_dir', "")
        )
        self.astap_search_radius_var = tk.DoubleVar(
            value=getattr(self.parent_gui.settings, 'astap_search_radius', 30.0)
        )
        print(
            f"DEBUG (LocalSolverSettingsWindow __init__): astap_search_radius_var initialisée à {self.astap_search_radius_var.get()}."
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
        self.astrometry_solve_field_dir_var = tk.StringVar(
            value=getattr(self.parent_gui.settings, "astrometry_solve_field_dir", "")
        )

        self.use_third_party_solver_var = tk.BooleanVar(
            value=getattr(self.parent_gui.settings, 'use_third_party_solver', True)
        )



        self.local_ansvr_path_var = tk.StringVar(
            value=getattr(self.parent_gui.settings, 'local_ansvr_path', "")
        )

        self.ansvr_host_port_var = tk.StringVar(
            value=getattr(self.parent_gui.settings, 'ansvr_host_port', '127.0.0.1:8080')
        )

        self.reproject_between_batches_var = tk.BooleanVar(
            value=getattr(
                self.parent_gui.settings, 'reproject_between_batches', False
            )
        )

        self.reproject_between_batches_var.trace_add('write', lambda *args: self._update_warning())
        self.local_solver_choice_var.trace_add('write', lambda *args: self._update_warning())

        # Construction de l'interface utilisateur
        self._build_ui()

        # Configuration finale de la fenêtre
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.update_idletasks() 

        # Centrage et affichage
        self.master.update_idletasks()
        parent_x = self.master.winfo_rootx()
        parent_y = self.master.winfo_rooty()
        parent_width = self.master.winfo_width()
        parent_height = self.master.winfo_height()
        # Forcer une mise à jour pour obtenir reqwidth/reqheight corrects après _build_ui
        self.update_idletasks() 
        self_width = self.winfo_reqwidth()
        self_height = self.winfo_reqheight()
        
        position_x = parent_x + (parent_width // 2) - (self_width // 2)
        position_y = parent_y + (parent_height // 2) - (self_height // 2)
        self.geometry(f"+{position_x}+{position_y}")

        self.deiconify()    
        self.focus_force()  
        self.grab_set()

        self._on_solver_choice_change()
        self._on_use_solver_toggle()
        print("DEBUG (LocalSolverSettingsWindow __init__): Fenêtre initialisée et _on_solver_choice_change() appelé.")

    def _on_solver_choice_change(self, *args):
        """
        Appelée lorsque le choix du solveur local (Radiobutton) change.
        Active ou désactive les cadres de configuration ASTAP et Ansvr.
        """
        choice = self.local_solver_choice_var.get()
        print(f"DEBUG (LocalSolverSettingsWindow _on_solver_choice_change): Choix détecté: '{choice}'") # DEBUG

        # État par défaut des sections
        astap_state = tk.DISABLED
        ansvr_state = tk.DISABLED
        astrometry_state = tk.DISABLED

        if choice == "astap":
            astap_state = tk.NORMAL
        elif choice == "ansvr":
            ansvr_state = tk.NORMAL
        elif choice == "astrometry":
            astrometry_state = tk.NORMAL
        # Si "none", les deux restent DISABLED

        # Activer/Désactiver les widgets dans le cadre ASTAP
        if hasattr(self, 'astap_frame') and self.astap_frame.winfo_exists():
            for widget in self.astap_frame.winfo_children():
                self._set_widget_state_recursive(widget, astap_state)
        
        # Activer/Désactiver les widgets dans le cadre Ansvr
        if hasattr(self, 'ansvr_frame') and self.ansvr_frame.winfo_exists():
            for widget in self.ansvr_frame.winfo_children():
                self._set_widget_state_recursive(widget, ansvr_state)

        # Activer/Désactiver les widgets dans le cadre Astrometry
        if hasattr(self, 'astrometry_frame') and self.astrometry_frame.winfo_exists():
            for widget in self.astrometry_frame.winfo_children():
                self._set_widget_state_recursive(widget, astrometry_state)
        
        print(
            f"DEBUG (LocalSolverSettingsWindow _on_solver_choice_change): États des cadres mis à jour - ASTAP: {astap_state}, Ansvr: {ansvr_state}"
        )  # DEBUG
        self._update_warning()

    def _on_use_solver_toggle(self, *args):
        state = tk.NORMAL if self.use_third_party_solver_var.get() else tk.DISABLED
        frames = [
            getattr(self, 'solver_choice_frame', None),
            getattr(self, 'astap_frame', None),
            getattr(self, 'ansvr_frame', None),
            getattr(self, 'astrometry_frame', None),
        ]
        for fr in frames:
            if fr and fr.winfo_exists():
                for w in fr.winfo_children():
                    self._set_widget_state_recursive(w, state)
        self._update_warning()

    def _set_widget_state_recursive(self, widget, state):
        """
        Change récursivement l'état d'un widget et de ses enfants (si applicable).
        """
        try:
            if 'state' in widget.configure():
                widget.configure(state=state)
        except tk.TclError:
            pass # Certains widgets (comme Frame) n'ont pas d'option 'state'

        # Pour les widgets conteneurs comme Frame ou LabelFrame, appliquer aux enfants
        if hasattr(widget, 'winfo_children'):
            for child in widget.winfo_children():
                self._set_widget_state_recursive(child, state)

    def _update_warning(self, *args):
        show = False
        if self.reproject_between_batches_var.get() and self.use_third_party_solver_var.get():
            choice = self.local_solver_choice_var.get()
            api_key = self.parent_gui.astrometry_api_key_var.get().strip()
            if choice == 'astap':
                show = not bool(self.astap_path_var.get().strip())
            elif choice == 'ansvr':
                show = not bool(self.local_ansvr_path_var.get().strip())
            elif choice == 'astrometry':
                show = not (self.astrometry_solve_field_dir_var.get().strip() or api_key)
            elif choice == 'none':
                show = not any(
                    [
                        self.astap_path_var.get().strip(),
                        self.local_ansvr_path_var.get().strip(),
                        self.astrometry_solve_field_dir_var.get().strip(),
                        api_key,
                    ]
                )
        self.warning_label.configure(
            text='⚠️ Aucun solveur local configuré' if show else ''
        )

    def _solver_configured(self):
        choice = self.local_solver_choice_var.get()
        if choice == 'astap':
            return bool(self.astap_path_var.get().strip())
        if choice == 'ansvr':
            return bool(self.local_ansvr_path_var.get().strip())
        if choice == 'astrometry':
            return bool(self.astrometry_solve_field_dir_var.get().strip() or self.parent_gui.astrometry_api_key_var.get().strip())
        return any(
            [
                self.astap_path_var.get().strip(),
                self.local_ansvr_path_var.get().strip(),
                self.astrometry_solve_field_dir_var.get().strip(),
                self.parent_gui.astrometry_api_key_var.get().strip(),
            ]
        )

####################################################################################################################################################





# --- DANS LA CLASSE LocalSolverSettingsWindow DANS seestar/gui/local_solver_gui.py ---

    def _build_ui(self):
        """
        Construit les widgets de l'interface utilisateur pour cette fenêtre.
        MODIFIED: Section Ansvr avec deux boutons "Browse".
        """
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        use_cb = ttk.Checkbutton(
            main_frame,
            text=self.tr("use_third_party_solver_label", default="Use third-party solver"),
            variable=self.use_third_party_solver_var,
            command=self._on_use_solver_toggle,
        )
        use_cb.pack(anchor=tk.W, pady=(0, 5))

        self.solver_choice_frame = ttk.LabelFrame(
            main_frame,
            text=self.tr("solver_label", default="Solver"),
            padding="10",
        )
        self.solver_choice_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Radiobutton(
            self.solver_choice_frame,
            text=self.tr("solver_astap", default="ASTAP"),
            variable=self.local_solver_choice_var,
            value="astap",
            command=self._on_solver_choice_change,
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            self.solver_choice_frame,
            text=self.tr("solver_astrometry", default="Astrometry.net"),
            variable=self.local_solver_choice_var,
            value="astrometry",
            command=self._on_solver_choice_change,
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            self.solver_choice_frame,
            text=self.tr("solver_ansvr", default="Ansvr"),
            variable=self.local_solver_choice_var,
            value="ansvr",
            command=self._on_solver_choice_change,
        ).pack(anchor=tk.W, pady=2)

        self.astap_frame = ttk.LabelFrame(
            main_frame,
            text=self.tr("solver_astap", default="ASTAP"),
            padding="10",
        )
        self.astap_frame.pack(fill=tk.X, padx=5, pady=5)

        astap_path_sub = ttk.Frame(self.astap_frame)
        astap_path_sub.pack(fill=tk.X, pady=(5, 2))
        ttk.Label(astap_path_sub, text=self.tr("astap_exe_label", default="Executable:")).pack(side=tk.LEFT)
        ttk.Entry(astap_path_sub, textvariable=self.astap_path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0)
        )
        ttk.Button(
            astap_path_sub,
            text=self.tr("browse", default="Browse..."),
            command=self._browse_astap_path,
            width=12,
        ).pack(side=tk.RIGHT, padx=(5, 0))


        astap_data_sub = ttk.Frame(self.astap_frame)
        astap_data_sub.pack(fill=tk.X, pady=(2, 5))
        ttk.Label(astap_data_sub, text=self.tr("astap_data_label", default="Data Dir:")).pack(side=tk.LEFT)
        ttk.Entry(astap_data_sub, textvariable=self.astap_data_dir_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0)
        )
        ttk.Button(
            astap_data_sub,
            text=self.tr("browse", default="Browse..."),
            command=self._browse_astap_data_dir,
            width=12,
        ).pack(side=tk.RIGHT, padx=(5, 0))

        astap_radius_sub = ttk.Frame(self.astap_frame)
        astap_radius_sub.pack(fill=tk.X, pady=(2, 5))
        ttk.Label(
            astap_radius_sub,
            text=self.tr(
                "astap_search_radius_label",
                default="ASTAP Search Radius (deg):",
            ),
            width=35,
            anchor="w",
        ).pack(side=tk.LEFT, padx=(0, 5))
        radius_sb = ttk.Spinbox(
            astap_radius_sub,
            from_=0.1,
            to=90.0,
            increment=0.5,
            textvariable=self.astap_search_radius_var,
            width=6,
            format="%.1f",
        )
        radius_sb.pack(side=tk.LEFT)
        ToolTip(radius_sb, lambda: self.tr("tooltip_astap_search_radius"))

        astap_down_sub = ttk.Frame(self.astap_frame)
        astap_down_sub.pack(fill=tk.X, pady=(2, 5))
        ttk.Label(
            astap_down_sub,
            text=self.tr(
                "local_solver_astap_downsample_label",
                default="Downsample:",
            ),
            width=35,
            anchor="w",
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(
            astap_down_sub,
            from_=1,
            to=8,
            increment=1,
            textvariable=self.astap_downsample_var,
            width=6,
        ).pack(side=tk.LEFT)

        astap_sens_sub = ttk.Frame(self.astap_frame)
        astap_sens_sub.pack(fill=tk.X, pady=(2, 5))
        ttk.Label(
            astap_sens_sub,
            text=self.tr(
                "local_solver_astap_sens_label",
                default="Sensitivity:",
            ),
            width=35,
            anchor="w",
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(
            astap_sens_sub,
            from_=10,
            to=1000,
            increment=5,
            textvariable=self.astap_sensitivity_var,
            width=6,
        ).pack(side=tk.LEFT)

        self.astrometry_frame = ttk.LabelFrame(
            main_frame,
            text=self.tr("solver_astrometry", default="Astrometry.net"),
            padding="10",
        )

        self.astrometry_frame.pack(fill=tk.X, padx=5, pady=5)

        api_key_sub = ttk.Frame(self.astrometry_frame)

        api_key_sub.pack(fill=tk.X, pady=(5, 2))
        ttk.Label(api_key_sub, text=self.tr("astrometry_api_key_label", default="API Key:")).pack(side=tk.LEFT)
        ttk.Entry(api_key_sub, textvariable=self.parent_gui.astrometry_api_key_var, show="*").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0)
        )

        astrometry_dir_sub = ttk.Frame(self.astrometry_frame)
        astrometry_dir_sub.pack(fill=tk.X, pady=(2, 5))
        ttk.Label(astrometry_dir_sub, text=self.tr("astrometry_dir_label", default="solve-field Dir:")).pack(side=tk.LEFT)
        ttk.Entry(astrometry_dir_sub, textvariable=self.astrometry_solve_field_dir_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0)
        )
        ttk.Button(
            astrometry_dir_sub,
            text=self.tr("browse", default="Browse..."),
            command=self._browse_astrometry_dir,
            width=12,
        ).pack(side=tk.RIGHT, padx=(5, 0))

        self.ansvr_frame = ttk.LabelFrame(
            main_frame,
            text=self.tr("solver_ansvr", default="Ansvr"),
            padding="10",
        )
        self.ansvr_frame.pack(fill=tk.X, padx=5, pady=5)

        ansvr_host_sub = ttk.Frame(self.ansvr_frame)
        ansvr_host_sub.pack(fill=tk.X, pady=(5, 2))
        ttk.Label(ansvr_host_sub, text=self.tr("ansvr_hostport_label", default="Host:Port:")).pack(side=tk.LEFT)
        ttk.Entry(ansvr_host_sub, textvariable=self.ansvr_host_port_var, width=15).pack(
            side=tk.LEFT, padx=(5, 0)
        )

        self.warning_label = ttk.Label(
            main_frame,
            foreground="red",
        )

        self.warning_label.pack(anchor=tk.W)

        self._update_warning()

        button_frame = ttk.Frame(main_frame, padding="5")
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        cancel_button = ttk.Button(
            button_frame,
            text=self.tr("cancel_button", default="Cancel"),
            command=self._on_cancel,
        )
        cancel_button.pack(side=tk.RIGHT, padx=(5, 0))
        ok_button = ttk.Button(
            button_frame,
            text=self.tr("ok_button", default="OK"),
            command=self._on_ok,
        )
        ok_button.pack(side=tk.RIGHT)








####################################################################################################################################################
    def _browse_astap_path(self):
        initial_dir = ""
        current_path = self.astap_path_var.get()
        if current_path and os.path.exists(os.path.dirname(current_path)):
            initial_dir = os.path.dirname(current_path)
        elif os.path.exists(current_path):
             initial_dir = os.path.dirname(current_path)

        file_types = [(self.tr("executable_files", default="Executable Files"), "*.*")]
        if os.name == 'nt':
            file_types = [
                (self.tr("astap_executable_win", default="ASTAP Executable"), "*.exe"),
                (self.tr("all_files", default="All Files"), "*.*"),
            ]
        elif platform.system() == "Darwin":
            file_types = [
                (self.tr("astap_app", default="ASTAP Application"), "*.app"),
                (self.tr("all_files", default="All Files"), "*.*"),
            ]
        
        filepath = filedialog.askopenfilename(
            title=self.tr("select_astap_executable_title", default="Select ASTAP Executable"),
            initialdir=initial_dir if initial_dir else os.path.expanduser("~"), 
            filetypes=file_types,
            parent=self 
        )
        if filepath: 
            self.astap_path_var.set(filepath)
            print(f"DEBUG (LocalSolverSettingsWindow): Chemin ASTAP sélectionné: {filepath}")

    def _browse_astap_data_dir(self):
        initial_dir = self.astap_data_dir_var.get()
        if not initial_dir or not os.path.isdir(initial_dir):
            initial_dir = os.path.expanduser("~")

        dirpath = filedialog.askdirectory(
            title=self.tr("select_astap_data_dir_title", default="Select ASTAP Star Index Data Directory"),
            initialdir=initial_dir,
            parent=self 
        )
        if dirpath:
            self.astap_data_dir_var.set(dirpath)
            print(f"DEBUG (LocalSolverSettingsWindow): Répertoire données ASTAP sélectionné: {dirpath}")

    def _browse_astrometry_dir(self):
        initial_dir = self.astrometry_solve_field_dir_var.get()
        if not initial_dir or not os.path.isdir(initial_dir):
            initial_dir = os.path.expanduser("~")

        dirpath = filedialog.askdirectory(
            title=self.tr("select_astrometry_exec_dir_title", default="Select solve-field Directory"),
            initialdir=initial_dir,
            parent=self,
        )
        if dirpath:
            self.astrometry_solve_field_dir_var.set(dirpath)
            print(f"DEBUG (LocalSolverSettingsWindow): Répertoire solve-field sélectionné: {dirpath}")





# --- DANS LA CLASSE LocalSolverSettingsWindow DANS seestar/gui/local_solver_gui.py ---

    def _browse_local_ansvr_path(self):
        """
        Ouvre un dialogue pour sélectionner le chemin de solve-field, un astrometry.cfg,
        OU un répertoire d'index (si la première sélection de fichier est annulée).
        """
        # Fonction de log interne simple pour cette méthode
        def _log_browser(message, level="DEBUG"):
            print(f"DEBUG (LSW _browse_local_ansvr_path) [{level}]: {message}")

        initial_dir_ansvr = ""
        current_path_ansvr = self.local_ansvr_path_var.get()
        if current_path_ansvr:
            if os.path.isfile(current_path_ansvr):
                initial_dir_ansvr = os.path.dirname(current_path_ansvr)
            elif os.path.isdir(current_path_ansvr):
                initial_dir_ansvr = current_path_ansvr
        if not initial_dir_ansvr:
            initial_dir_ansvr = os.path.expanduser("~")

        _log_browser(f"Ouverture de la boîte de dialogue pour fichiers (initialdir: {initial_dir_ansvr})")
        filepath_selected = filedialog.askopenfilename(
            title=self.tr("select_ansvr_exe_or_cfg_title", default="Select solve-field Executable or .cfg (Cancel for Index Dir)"),
            initialdir=initial_dir_ansvr,
            filetypes=[
                (self.tr("configuration_files", default="Configuration Files"), "*.cfg"),
                (self.tr("executable_files", default="Executable Files"), "*.*" if os.name != 'nt' else "*.exe"),
                (self.tr("all_files", default="All Files"), "*.*")
            ],
            parent=self
        )

        if not filepath_selected: # L'utilisateur a annulé la sélection de fichier ou n'a rien choisi
            _log_browser("Aucun fichier sélectionné. Ouverture de la boîte de dialogue pour répertoires.")
            dirpath_selected = filedialog.askdirectory(
                title=self.tr("select_ansvr_index_dir_title", default="Select Astrometry.net Index Directory"),
                initialdir=initial_dir_ansvr, # On peut réutiliser le même initialdir
                parent=self
            )
            if dirpath_selected:
                _log_browser(f"Répertoire d'index sélectionné: {dirpath_selected}")
                self.local_ansvr_path_var.set(dirpath_selected)
                print(f"DEBUG (LocalSolverSettingsWindow): Répertoire Ansvr Index sélectionné: {dirpath_selected}") # Garder le print original
            else:
                _log_browser("Aucun répertoire d'index sélectionné non plus.")
        else: # Un fichier a été sélectionné
            _log_browser(f"Fichier sélectionné: {filepath_selected}")
            self.local_ansvr_path_var.set(filepath_selected)
            print(f"DEBUG (LocalSolverSettingsWindow): Fichier Ansvr Exe/Cfg sélectionné: {filepath_selected}") # Garder le print original

#####################################################################################################################################################





# --- DANS LA CLASSE LocalSolverSettingsWindow DANS seestar/gui/local_solver_gui.py ---

    def _browse_ansvr_file(self):
        """
        Ouvre un dialogue pour sélectionner l'exécutable solve-field ou un fichier .cfg.
        Met à jour self.local_ansvr_path_var avec le chemin du fichier sélectionné.
        (Version corrigée pour la définition de file_types_list)
        """
        # Déterminer le répertoire initial pour la boîte de dialogue
        initial_dir_ansvr = os.path.expanduser("~") # Défaut au répertoire home
        current_path = self.local_ansvr_path_var.get()

        if current_path and os.path.exists(current_path):
            if os.path.isfile(current_path):
                # Si le chemin actuel est un fichier, commencer dans son répertoire parent
                initial_dir_ansvr = os.path.dirname(current_path)
            # Si c'est un répertoire, on garde le home comme initial_dir pour la sélection de *fichier*.
            # L'utilisateur peut naviguer à partir de là.
        elif current_path and os.path.exists(os.path.dirname(current_path)): # Si le parent du chemin (non existant) existe
             initial_dir_ansvr = os.path.dirname(current_path)

        print(f"DEBUG (LSW _browse_ansvr_file): Ouverture dialogue sélection fichier. Initialdir: '{initial_dir_ansvr}'")

        # Définir les types de fichiers en fonction de l'OS
        if os.name == 'nt': # Cas Windows
            file_types_list = [
                (self.tr("configuration_files", default="Config Files"), "*.cfg"),
                (self.tr("executable_files", default="Executable Files (.exe)"), "*.exe"), # Plus spécifique pour Windows
                (self.tr("all_files", default="All Files"), "*.*")
            ]
            # print(f"DEBUG (LSW _browse_ansvr_file): Types de fichiers pour Windows: {file_types_list}") # Optionnel
        else: # Cas non-Windows (Linux, macOS, etc.)
            file_types_list = [
                (self.tr("executable_files", default="Executable Files (any)"), "*"), # Souvent sans extension
                (self.tr("configuration_files", default="Config Files (.cfg)"), "*.cfg"),
                (self.tr("all_files", default="All Files"), "*.*")
            ]
            # print(f"DEBUG (LSW _browse_ansvr_file): Types de fichiers pour non-Windows: {file_types_list}") # Optionnel

        filepath = filedialog.askopenfilename(
            title=self.tr("select_ansvr_exe_or_cfg_title_v2", default="Select solve-field Executable or .cfg File"),
            initialdir=initial_dir_ansvr,
            filetypes=file_types_list,
            parent=self  # Important pour la modalité correcte
        )

        if filepath: # Si un fichier a été sélectionné (l'utilisateur n'a pas annulé)
            self.local_ansvr_path_var.set(filepath)
            print(f"DEBUG (LocalSolverSettingsWindow _browse_ansvr_file): Fichier Ansvr (Exe/Cfg) sélectionné: {filepath}")
        else:
            # L'utilisateur a annulé la boîte de dialogue ou l'a fermée.
            # self.local_ansvr_path_var reste inchangée.
            print(f"DEBUG (LocalSolverSettingsWindow _browse_ansvr_file): Aucun fichier sélectionné (dialogue annulé ou fermé).")


# --- DANS LA CLASSE LocalSolverSettingsWindow DANS seestar/gui/local_solver_gui.py ---

    # ... (méthodes __init__, _on_solver_choice_change, _set_widget_state_recursive, _build_ui,
    #      _browse_astap_path, _browse_astap_data_dir, _browse_ansvr_file que tu as déjà) ...

    def _browse_ansvr_index_dir(self):
        """
        Ouvre un dialogue pour sélectionner le répertoire des index Astrometry.net.
        Met à jour self.local_ansvr_path_var avec le chemin du répertoire sélectionné.
        """
        # Déterminer le répertoire initial pour la boîte de dialogue
        initial_dir_ansvr = os.path.expanduser("~") # Défaut au répertoire home
        current_path = self.local_ansvr_path_var.get()

        if current_path and os.path.exists(current_path):
            if os.path.isdir(current_path):
                # Si le chemin actuel est un répertoire, commencer là
                initial_dir_ansvr = current_path
            elif os.path.isfile(current_path):
                # Si c'est un fichier, commencer dans son répertoire parent
                initial_dir_ansvr = os.path.dirname(current_path)
        elif current_path and os.path.exists(os.path.dirname(current_path)): # Si le parent du chemin (non existant) existe
            initial_dir_ansvr = os.path.dirname(current_path)


        print(f"DEBUG (LSW _browse_ansvr_index_dir): Ouverture dialogue sélection répertoire. Initialdir: '{initial_dir_ansvr}'")

        dirpath = filedialog.askdirectory(
            title=self.tr("select_ansvr_index_dir_title_v2", default="Select Astrometry.net Index Directory"),
            initialdir=initial_dir_ansvr,
            parent=self  # Important pour la modalité correcte
        )

        if dirpath: # Si un répertoire a été sélectionné (l'utilisateur n'a pas annulé)
            self.local_ansvr_path_var.set(dirpath)
            print(f"DEBUG (LocalSolverSettingsWindow _browse_ansvr_index_dir): Répertoire Ansvr Index sélectionné: {dirpath}")
        else:
            # L'utilisateur a annulé la boîte de dialogue ou l'a fermée.
            # self.local_ansvr_path_var reste inchangée.
            print(f"DEBUG (LocalSolverSettingsWindow _browse_ansvr_index_dir): Aucun répertoire sélectionné (dialogue annulé ou fermé).")

    


#####################################################################################################################################################

    def _on_ok(self):
        """
        Appelé lorsque l'utilisateur clique sur OK.
        Sauvegarde les paramètres et ferme la fenêtre.
        """
        print("DEBUG (LocalSolverSettingsWindow _on_ok): Bouton OK cliqué.") # DEBUG
        print("DEBUG (LocalSolverSettingsWindow _on_ok): ***** MÉTHODE _on_ok APPELÉE *****") # Log très visible
        # --- MODIFIÉ : Lire la nouvelle variable de choix ---
        solver_choice = self.local_solver_choice_var.get()
        astap_path = self.astap_path_var.get().strip()
        astap_data_dir = self.astap_data_dir_var.get().strip()
        astap_radius = self.astap_search_radius_var.get()
        astap_downsample = self.astap_downsample_var.get()
        astap_sensitivity = self.astap_sensitivity_var.get()
        self.parent_gui.settings.astap_downsample = astap_downsample
        self.parent_gui.settings.astap_sensitivity = astap_sensitivity
        cluster_threshold = self.cluster_threshold_var.get()
        local_ansvr_path = self.local_ansvr_path_var.get().strip()
        ansvr_host_port = self.ansvr_host_port_var.get().strip()
        reproject_batches = self.reproject_between_batches_var.get()

        astrometry_dir = self.astrometry_solve_field_dir_var.get().strip()


        # Valider que si un solveur est choisi, son chemin principal est rempli
        validation_ok = True
        if solver_choice == "astap" and not astap_path:
            messagebox.showerror(self.tr("error"), 
                                 self.tr("astap_path_required_error", default="ASTAP is selected, but the executable path is missing."),
                                 parent=self)
            validation_ok = False


        elif solver_choice == "ansvr" and not local_ansvr_path:
            messagebox.showerror(self.tr("error"),
                                 self.tr("ansvr_path_required_error", default="Astrometry.net Local is selected, but the path/config is missing."),
                                 parent=self)
            validation_ok = False
        elif solver_choice == "astrometry" and not (astrometry_dir or self.parent_gui.astrometry_api_key_var.get().strip()):
            messagebox.showerror(
                self.tr("error"),
                self.tr(
                    "astrometry_path_required_error",
                    default="Astrometry.net is selected, but neither API key nor solve-field directory is provided.",
                ),
                parent=self,
            )
            validation_ok = False
        
        if not validation_ok:
            print("DEBUG (LocalSolverSettingsWindow _on_ok): Validation échouée (chemin manquant pour solveur choisi).") # DEBUG
            return # Ne pas fermer la fenêtre

        # --- MODIFIÉ : Déterminer la préférence finale du solveur ---
        final_solver_pref = solver_choice
        if solver_choice == "astrometry":
            # Utilisation locale si un dossier solve-field est fourni,
            # sinon résolution via le service web.
            if astrometry_dir:
                final_solver_pref = "ansvr"
            else:
                final_solver_pref = "none"

        setattr(self.parent_gui.settings, 'local_solver_preference', final_solver_pref)
        if hasattr(self.parent_gui.settings, 'use_local_solver_priority'):
            delattr(self.parent_gui.settings, 'use_local_solver_priority') # Supprimer l'ancien attribut
        
        self.parent_gui.settings.astap_path = astap_path
        self.parent_gui.settings.astap_data_dir = astap_data_dir
        setattr(self.parent_gui.settings, 'astap_search_radius', astap_radius)
        self.parent_gui.settings.local_ansvr_path = local_ansvr_path
        self.parent_gui.settings.ansvr_host_port = ansvr_host_port
        self.parent_gui.settings.use_third_party_solver = self.use_third_party_solver_var.get()

        if hasattr(self.parent_gui, 'use_third_party_solver_var'):
            try:
                self.parent_gui.use_third_party_solver_var.set(
                    self.use_third_party_solver_var.get()
                )
            except Exception:
                pass


        self.parent_gui.settings.reproject_between_batches = reproject_batches
        if hasattr(self.parent_gui, 'reproject_between_batches_var'):
            try:
                self.parent_gui.reproject_between_batches_var.set(reproject_batches)
            except Exception:
                pass
        # Refresh Add Folder button state in the main GUI if available
        try:
            if hasattr(self.parent_gui, 'update_add_folder_button_state'):
                self.parent_gui.update_add_folder_button_state()
        except Exception:
            pass

        self.parent_gui.settings.astrometry_solve_field_dir = astrometry_dir

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

        print(

            f"  LocalSolverSettingsWindow: Préférence Sauvegardée='{solver_choice}', ASTAP='{astap_path}', Data ASTAP='{astap_data_dir}', Radius ASTAP={astap_radius}, Down={astap_downsample}, Sens={astap_sensitivity}, Cluster={cluster_threshold}, Ansvr Local='{local_ansvr_path}', HostPort='{ansvr_host_port}', Astrometry Dir='{astrometry_dir}', ReprojBatches={reproject_batches}"

        )  # DEBUG
        print("  LocalSolverSettingsWindow: Paramètres mis à jour dans parent_gui.settings.")
        print(f"DEBUG (LocalSolverSettingsWindow _on_ok): Validation OK. Sauvegarde des settings. Préparation fermeture fenêtre.")
        print(f"  -> local_solver_preference: {solver_choice}")
        print(f"  -> local_ansvr_path à sauvegarder: {local_ansvr_path}")
        print(f"  -> ansvr_host_port à sauvegarder: {ansvr_host_port}")

        print(f"  -> astrometry_dir à sauvegarder: {astrometry_dir}")

        print(f"  -> reproject_batches: {reproject_batches}")
        self.grab_release()
        self.destroy()

    def _on_cancel(self):
        print("DEBUG (LocalSolverSettingsWindow _on_cancel): Fenêtre annulée/fermée.") # DEBUG
        self.grab_release()
        self.destroy()

# --- END OF FILE seestar/gui/local_solver_gui.py ---
