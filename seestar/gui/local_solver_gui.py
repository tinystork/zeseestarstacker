# --- START OF FILE seestar/gui/local_solver_gui.py ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os # Pour les opérations sur les chemins
# Assure-toi que l'import de ToolTip est correct si tu l'utilises
from .ui_utils import ToolTip


class LocalSolverSettingsWindow(tk.Toplevel):
    """
    Fenêtre de dialogue pour configurer les chemins d'accès
    aux solveurs astrométriques locaux comme ASTAP.
    """
    def __init__(self, parent_gui):
        print("DEBUG (LocalSolverSettingsWindow __init__ V_Final_With_Tooltips): Initialisation...")
        if not hasattr(parent_gui, 'root') or not parent_gui.root.winfo_exists():
             raise ValueError("Parent GUI invalide pour LocalSolverSettingsWindow")

        super().__init__(parent_gui.root)
        self.parent_gui = parent_gui
        self.withdraw()

        self.title(self.parent_gui.tr("local_solver_window_title", default="Local Astrometry Solvers Configuration"))
        self.transient(parent_gui.root)

        sm_defaults = self.parent_gui.settings.get_default_values()
        print(f"  DEBUG (LocalSolverSettingsWindow __init__): Valeurs par défaut de SettingsManager récupérées.")

        self.use_local_solver_priority_var = tk.BooleanVar(
            value=getattr(self.parent_gui.settings, 'use_local_solver_priority',
                          sm_defaults.get('use_local_solver_priority', False))
        )
        self.astap_path_var = tk.StringVar(
            value=getattr(self.parent_gui.settings, 'astap_path',
                          sm_defaults.get('astap_path', ""))
        )
        self.astap_data_dir_var = tk.StringVar(
            value=getattr(self.parent_gui.settings, 'astap_data_dir',
                          sm_defaults.get('astap_data_dir', ""))
        )
        self.local_ansvr_path_var = tk.StringVar(
            value=getattr(self.parent_gui.settings, 'local_ansvr_path',
                          sm_defaults.get('local_ansvr_path', ""))
        )
        try:
            initial_astap_radius = float(
                getattr(self.parent_gui.settings, 'astap_search_radius',
                        sm_defaults.get('astap_search_radius', 5.0))
            )


        except (ValueError, TypeError):
            print(f"  WARN (LocalSolverSettingsWindow __init__): Valeur astap_search_radius invalide, utilisation défaut 5.0.")
            initial_astap_radius = 5.0
        self.astap_search_radius_var = tk.DoubleVar(value=initial_astap_radius)
        print(f"  DEBUG (LocalSolverSettingsWindow __init__): astap_search_radius_var initialisée à {self.astap_search_radius_var.get()}")

        # Les méthodes de callback sont définies ci-dessous, avant _build_ui
        self._build_ui() # Construction de l'UI

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.update_idletasks()

        self.minsize(550, 380) # Ajusté pour le nouveau champ
        self.update_idletasks()

        self.master.update_idletasks()
        parent_x = self.master.winfo_rootx(); parent_y = self.master.winfo_rooty()
        parent_width = self.master.winfo_width(); parent_height = self.master.winfo_height()
        self_width = self.winfo_reqwidth(); self_height = self.winfo_reqheight()
        position_x = parent_x + (parent_width // 2) - (self_width // 2)
        position_y = parent_y + (parent_height // 2) - (self_height // 2)
        self.geometry(f"{self_width}x{self_height}+{position_x}+{position_y}")

        self.deiconify()
        self.focus_force()
        self.grab_set()
        print("DEBUG (LocalSolverSettingsWindow __init__ V_Final_With_Tooltips): Fenêtre initialisée et rendue modale.")

    def _on_ok(self):
        print("DEBUG (LocalSolverSettingsWindow _on_ok): Clic sur OK.")
        use_priority = self.use_local_solver_priority_var.get()
        astap_exe = self.astap_path_var.get().strip()
        astap_data_dir = self.astap_data_dir_var.get().strip()
        local_ansvr_path = self.local_ansvr_path_var.get().strip()
        try:
            astap_radius = self.astap_search_radius_var.get()
            if not (0.1 <= astap_radius <= 90.0): # Validation de la plage
                messagebox.showerror(self.parent_gui.tr("error", default="Error"),
                                     self.parent_gui.tr("invalid_astap_radius_range"), # Utilise la clé de traduction
                                     parent=self)
                return
        except tk.TclError:
            messagebox.showerror(self.parent_gui.tr("error", default="Error"),
                                 self.parent_gui.tr("invalid_astap_radius_value"), # Utilise la clé de traduction
                                 parent=self)
            return

        self.parent_gui.settings.use_local_solver_priority = use_priority
        self.parent_gui.settings.astap_path = astap_exe
        self.parent_gui.settings.astap_data_dir = astap_data_dir
        self.parent_gui.settings.local_ansvr_path = local_ansvr_path
        self.parent_gui.settings.astap_search_radius = astap_radius

        try:
            self.parent_gui.settings.save_settings()
            print("  DEBUG (LocalSolverSettingsWindow _on_ok): self.parent_gui.settings.save_settings() appelé.")
        except Exception as e_save:
            print(f"  ERREUR (LocalSolverSettingsWindow _on_ok): Échec sauvegarde settings: {e_save}")
            messagebox.showwarning(
                self.parent_gui.tr("warning", default="Warning"),
                self.parent_gui.tr("settings_save_failed_on_ok") + f"\n{e_save}", # Utilise la clé de traduction
                parent=self
            )
        print(f"  LocalSolverSettings (après MAJ et tentative sauvegarde): "
              f"Priorité Locale={self.parent_gui.settings.use_local_solver_priority}, "
              f"ASTAP='{self.parent_gui.settings.astap_path}', "
              f"Data ASTAP='{self.parent_gui.settings.astap_data_dir}', "
              f"Ansvr Local='{self.parent_gui.settings.local_ansvr_path}', "
              f"ASTAP Radius={self.parent_gui.settings.astap_search_radius}")
        self.grab_release()
        self.destroy()

    def _on_cancel(self):
        print("DEBUG (LocalSolverSettingsWindow _on_cancel): Fenêtre annulée/fermée.")
        self.grab_release()
        self.destroy()

    def _browse_file_dialog(self, tk_var, title_key, title_default, file_types_key_list, file_types_default_list):
        """Helper pour ouvrir un dialogue de sélection de fichier."""
        initial_dir = ""
        current_path = tk_var.get()
        if current_path and os.path.exists(os.path.dirname(current_path)):
            initial_dir = os.path.dirname(current_path)
        elif os.path.exists(current_path):
            initial_dir = os.path.dirname(current_path)
        else:
            initial_dir = os.path.expanduser("~")

        # Préparer les filetypes traduits
        translated_filetypes = []
        for key_tuple, default_tuple in zip(file_types_key_list, file_types_default_list):
            label = self.parent_gui.tr(key_tuple[0], default=default_tuple[0])
            pattern = key_tuple[1] # Le pattern (ex: "*.exe") n'est pas traduit
            translated_filetypes.append((label, pattern))

        filepath = filedialog.askopenfilename(
            title=self.parent_gui.tr(title_key, default=title_default),
            initialdir=initial_dir,
            filetypes=translated_filetypes,
            parent=self
        )
        if filepath:
            tk_var.set(filepath)
        return filepath

    def _browse_folder_dialog(self, tk_var, title_key, title_default):
        """Helper pour ouvrir un dialogue de sélection de dossier."""
        initial_dir = tk_var.get()
        if not initial_dir or not os.path.isdir(initial_dir):
            initial_dir = os.path.expanduser("~")
        folderpath = filedialog.askdirectory(
            title=self.parent_gui.tr(title_key, default=title_default),
            initialdir=initial_dir,
            parent=self
        )
        if folderpath:
            tk_var.set(folderpath)
        return folderpath

    def _browse_astap_path(self):
        file_types_keys = [("astap_executable_win", "*.exe"), ("all_files", "*.*")] if os.name == 'nt' else [("executable_files", "*.*"), ("all_files", "*.*")]
        file_types_defaults = [("ASTAP Executable", "*.exe"), ("All Files", "*.*")] if os.name == 'nt' else [("Executable Files", "*.*"), ("All Files", "*.*")]
        filepath = self._browse_file_dialog(self.astap_path_var, "select_astap_executable_title", "Select ASTAP Executable", file_types_keys, file_types_defaults)
        if filepath: print(f"DEBUG (LocalSolverSettingsWindow): Chemin ASTAP sélectionné: {filepath}")

    def _browse_astap_data_dir(self):
        dirpath = self._browse_folder_dialog(self.astap_data_dir_var, "select_astap_data_dir_title", "Select ASTAP Star Index Data Directory")
        if dirpath: print(f"DEBUG (LocalSolverSettingsWindow): Répertoire données ASTAP sélectionné: {dirpath}")

    def _browse_local_ansvr_path(self):
        dirpath = self._browse_folder_dialog(self.local_ansvr_path_var, "select_local_ansvr_path_title", "Select Local Astrometry.net (ansvr) Path")
        if dirpath: print(f"DEBUG (LocalSolverSettingsWindow): Chemin ansvr local sélectionné: {dirpath}")


    def _build_ui(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        general_options_frame = ttk.LabelFrame(main_frame,
                                               text=self.parent_gui.tr("local_solver_general_options_frame"),
                                               padding="10")
        general_options_frame.pack(fill=tk.X, padx=5, pady=(0, 10))
        self.use_local_solver_priority_check = ttk.Checkbutton(
            general_options_frame,
            text=self.parent_gui.tr("local_solver_use_priority_label"),
            variable=self.use_local_solver_priority_var
        )
        self.use_local_solver_priority_check.pack(anchor=tk.W, pady=5)

        astap_frame = ttk.LabelFrame(main_frame,
                                     text=self.parent_gui.tr("local_solver_astap_frame_title"),
                                     padding="10")
        astap_frame.pack(fill=tk.X, padx=5, pady=5)
        astap_path_subframe = ttk.Frame(astap_frame)
        astap_path_subframe.pack(fill=tk.X, pady=(5, 2))
        astap_path_label = ttk.Label(astap_path_subframe,
                                     text=self.parent_gui.tr("local_solver_astap_path_label"),
                                     width=28, anchor="w")
        astap_path_label.pack(side=tk.LEFT, padx=(0,5))
        astap_browse_button = ttk.Button(astap_path_subframe,
                                         text=self.parent_gui.tr("browse"),
                                         command=self._browse_astap_path) # Commande OK
        astap_browse_button.pack(side=tk.RIGHT, padx=(5,0))
        astap_path_entry = ttk.Entry(astap_path_subframe, textvariable=self.astap_path_var)
        astap_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,0))
        astap_data_subframe = ttk.Frame(astap_frame)
        astap_data_subframe.pack(fill=tk.X, pady=(2, 2))
        astap_data_label = ttk.Label(astap_data_subframe,
                                     text=self.parent_gui.tr("local_solver_astap_data_label"),
                                     width=28, anchor="w")
        astap_data_label.pack(side=tk.LEFT, padx=(0,5))
        astap_data_browse_button = ttk.Button(astap_data_subframe,
                                              text=self.parent_gui.tr("browse"),
                                              command=self._browse_astap_data_dir) # Commande OK
        astap_data_browse_button.pack(side=tk.RIGHT, padx=(5,0))
        astap_data_entry = ttk.Entry(astap_data_subframe, textvariable=self.astap_data_dir_var)
        astap_data_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,0))
        astap_radius_frame = ttk.Frame(astap_frame)
        astap_radius_frame.pack(fill=tk.X, pady=(2, 5))
        astap_radius_label = ttk.Label(astap_radius_frame,
                                       text=self.parent_gui.tr("astap_search_radius_label"), # Clé pour traduction
                                       width=28, anchor="w")
        astap_radius_label.pack(side=tk.LEFT, padx=(0,5))
        self.astap_radius_spinbox = ttk.Spinbox(astap_radius_frame, from_=0.1, to=90.0, increment=0.5,
                                                textvariable=self.astap_search_radius_var, width=7, justify=tk.RIGHT, format="%.1f")
        self.astap_radius_spinbox.pack(side=tk.LEFT, padx=(0,5))
        ToolTip(self.astap_radius_spinbox, lambda: self.parent_gui.tr("tooltip_astap_search_radius")) # Clé pour traduction

        ansvr_frame = ttk.LabelFrame(main_frame,
                                     text=self.parent_gui.tr("local_solver_ansvr_frame_title"),
                                     padding="10")
        ansvr_frame.pack(fill=tk.X, padx=5, pady=5)
        ansvr_path_subframe = ttk.Frame(ansvr_frame)
        ansvr_path_subframe.pack(fill=tk.X, pady=5)
        ansvr_path_label = ttk.Label(ansvr_path_subframe,
                                     text=self.parent_gui.tr("local_solver_ansvr_path_label"),
                                     width=28, anchor="w")
        ansvr_path_label.pack(side=tk.LEFT, padx=(0,5))
        ansvr_browse_button = ttk.Button(ansvr_path_subframe,
                                         text=self.parent_gui.tr("browse"),
                                         command=self._browse_local_ansvr_path) # Commande OK
        ansvr_browse_button.pack(side=tk.RIGHT, padx=(5,0))
        ansvr_path_entry = ttk.Entry(ansvr_path_subframe, textvariable=self.local_ansvr_path_var)
        ansvr_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,0))

        info_label = ttk.Label(main_frame,
                               text=self.parent_gui.tr("local_solver_info_text"),
                               justify=tk.LEFT, wraplength=530)
        info_label.pack(pady=(10,5), padx=5, fill=tk.X)

        button_frame = ttk.Frame(main_frame, padding="5")
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        cancel_button = ttk.Button(button_frame,
                                   text=self.parent_gui.tr("cancel"),
                                   command=self._on_cancel) # Commande OK
        cancel_button.pack(side=tk.RIGHT, padx=(5, 0))
        ok_button = ttk.Button(button_frame,
                               text=self.parent_gui.tr("ok"),
                               command=self._on_ok) # Commande OK
        ok_button.pack(side=tk.RIGHT)
        print("DEBUG (LocalSolverSettingsWindow _build_ui): UI construite avec les champs de configuration (y compris ASTAP radius).")

# --- Fin de la classe ---