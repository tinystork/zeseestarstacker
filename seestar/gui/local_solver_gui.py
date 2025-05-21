# --- START OF FILE seestar/gui/local_solver_gui.py ---
"""
Module pour la fenêtre de configuration des solveurs astrométriques locaux.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os # Pour les opérations sur les chemins

class LocalSolverSettingsWindow(tk.Toplevel):
    """
    Fenêtre de dialogue pour configurer les chemins d'accès
    aux solveurs astrométriques locaux comme ASTAP ou Astrometry.net local.
    """
    def __init__(self, parent_gui):
        """
        Initialise la fenêtre de configuration des solveurs locaux.

        Args:
            parent_gui: L'instance de SeestarStackerGUI parente.
        """
        super().__init__(parent_gui.root)
        self.parent_gui = parent_gui
        self.withdraw() # Cacher pendant la configuration

        self.title(self.parent_gui.tr("local_solver_window_title", default="Local Astrometry Solvers Configuration"))
        self.transient(parent_gui.root)

        print("DEBUG (LocalSolverSettingsWindow __init__): Début initialisation.") # DEBUG

        # --- Variables Tkinter pour les chemins et options ---
        
        # <<< NOUVEAU : Variable pour le choix du solveur local >>>
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
        # <<< NOUVEAU : Variable pour le rayon de recherche ASTAP >>>
        self.astap_search_radius_var = tk.DoubleVar( # Utiliser DoubleVar pour le Spinbox
            value=getattr(self.parent_gui.settings, 'astap_search_radius', 30.0) # Défaut 30 deg
        )
        print(f"DEBUG (LocalSolverSettingsWindow __init__): astap_search_radius_var initialisée à {self.astap_search_radius_var.get()}.") # DEBUG


        self.local_ansvr_path_var = tk.StringVar(
            value=getattr(self.parent_gui.settings, 'local_ansvr_path', "")
        )

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

        # <<< NOUVEAU : Appel initial pour mettre à jour l'état des cadres de configuration >>>
        self._on_solver_choice_change()
        print("DEBUG (LocalSolverSettingsWindow __init__): Fenêtre initialisée et _on_solver_choice_change() appelé.")

    # <<< NOUVELLE MÉTHODE >>>
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

        if choice == "astap":
            astap_state = tk.NORMAL
        elif choice == "ansvr":
            ansvr_state = tk.NORMAL
        # Si "none", les deux restent DISABLED

        # Activer/Désactiver les widgets dans le cadre ASTAP
        if hasattr(self, 'astap_frame') and self.astap_frame.winfo_exists():
            for widget in self.astap_frame.winfo_children():
                self._set_widget_state_recursive(widget, astap_state)
        
        # Activer/Désactiver les widgets dans le cadre Ansvr
        if hasattr(self, 'ansvr_frame') and self.ansvr_frame.winfo_exists():
            for widget in self.ansvr_frame.winfo_children():
                self._set_widget_state_recursive(widget, ansvr_state)
        
        print(f"DEBUG (LocalSolverSettingsWindow _on_solver_choice_change): États des cadres mis à jour - ASTAP: {astap_state}, Ansvr: {ansvr_state}") # DEBUG

    # <<< NOUVELLE MÉTHODE UTILITAIRE >>>
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


    def _build_ui(self):
        """
        Construit les widgets de l'interface utilisateur pour cette fenêtre.
        """
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # --- Section des Options Générales des Solveurs Locaux ---
        # --- MODIFIÉ : Remplacement de la Checkbox par des RadioButtons ---
        solver_choice_frame = ttk.LabelFrame(main_frame, 
                                             text=self.parent_gui.tr("local_solver_choice_frame_title", 
                                                                     default="Local Solver Preference"), 
                                             padding="10")
        solver_choice_frame.pack(fill=tk.X, padx=5, pady=(0, 10))

        ttk.Radiobutton(
            solver_choice_frame,
            text=self.parent_gui.tr("local_solver_choice_none", default="Do not use local solvers (use Astrometry.net web service if API key provided)"),
            variable=self.local_solver_choice_var,
            value="none", # Valeur associée
            command=self._on_solver_choice_change # Commande à appeler
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            solver_choice_frame,
            text=self.parent_gui.tr("local_solver_choice_astap", default="Use ASTAP (local)"),
            variable=self.local_solver_choice_var,
            value="astap",
            command=self._on_solver_choice_change
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            solver_choice_frame,
            text=self.parent_gui.tr("local_solver_choice_ansvr", default="Use Astrometry.net Local (solve-field)"),
            variable=self.local_solver_choice_var,
            value="ansvr",
            command=self._on_solver_choice_change
        ).pack(anchor=tk.W, pady=2)
        # --- FIN MODIFICATION RadioButtons ---


        # --- Configuration ASTAP ---
        # --- MODIFIÉ : Stocker la référence au cadre pour l'activer/désactiver ---
        self.astap_frame = ttk.LabelFrame(main_frame, 
                                     text=self.parent_gui.tr("local_solver_astap_frame_title", default="ASTAP Configuration"), 
                                     padding="10")
        self.astap_frame.pack(fill=tk.X, padx=5, pady=5)

        # Chemin vers l'exécutable ASTAP
        astap_path_subframe = ttk.Frame(self.astap_frame)
        astap_path_subframe.pack(fill=tk.X, pady=(5, 2))
        
        astap_path_label = ttk.Label(astap_path_subframe, 
                                     text=self.parent_gui.tr("local_solver_astap_path_label", default="ASTAP Executable Path:"), 
                                     width=25, anchor="w") 
        astap_path_label.pack(side=tk.LEFT)
        
        astap_browse_button = ttk.Button(astap_path_subframe, 
                                         text=self.parent_gui.tr("browse", default="Browse..."), 
                                         command=self._browse_astap_path, width=10)
        astap_browse_button.pack(side=tk.RIGHT, padx=(5,0))
        
        astap_path_entry = ttk.Entry(astap_path_subframe, textvariable=self.astap_path_var)
        astap_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,0))

        # Chemin vers le dossier de données d'index ASTAP
        astap_data_subframe = ttk.Frame(self.astap_frame)
        astap_data_subframe.pack(fill=tk.X, pady=(2, 5))

        astap_data_label = ttk.Label(astap_data_subframe,
                                     text=self.parent_gui.tr("local_solver_astap_data_label", default="ASTAP Star Index Data Directory:"),
                                     width=25, anchor="w")
        astap_data_label.pack(side=tk.LEFT)

        astap_data_browse_button = ttk.Button(astap_data_subframe,
                                              text=self.parent_gui.tr("browse", default="Browse..."),
                                              command=self._browse_astap_data_dir, width=10)
        astap_data_browse_button.pack(side=tk.RIGHT, padx=(5,0))

        astap_data_entry = ttk.Entry(astap_data_subframe, textvariable=self.astap_data_dir_var)
        astap_data_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,0))

        # <<< NOUVEAU : Champ pour le rayon de recherche ASTAP >>>
        astap_radius_subframe = ttk.Frame(self.astap_frame)
        astap_radius_subframe.pack(fill=tk.X, pady=(2,5))

        astap_radius_label = ttk.Label(astap_radius_subframe,
                                       text=self.parent_gui.tr("local_solver_astap_radius_label", default="ASTAP Search Radius (degrees, 0 for auto):"),
                                       width=35, anchor="w") # Largeur ajustée
        astap_radius_label.pack(side=tk.LEFT, padx=(0,5))

        astap_radius_spinbox = ttk.Spinbox(astap_radius_subframe,
                                            from_=0.0, to=180.0, increment=0.5, # Plage et incrément
                                            textvariable=self.astap_search_radius_var,
                                            width=6, format="%.1f") # Format pour une décimale
        astap_radius_spinbox.pack(side=tk.LEFT)
        # <<< FIN NOUVEAU Champ Rayon ASTAP >>>


        # --- Configuration Astrometry.net Local (ansvr) ---
        # --- MODIFIÉ : Stocker la référence au cadre ---
        self.ansvr_frame = ttk.LabelFrame(main_frame, 
                                     text=self.parent_gui.tr("local_solver_ansvr_frame_title", default="Local Astrometry.net (solve-field) Configuration"), 
                                     padding="10")
        self.ansvr_frame.pack(fill=tk.X, padx=5, pady=5)

        ansvr_path_subframe = ttk.Frame(self.ansvr_frame)
        ansvr_path_subframe.pack(fill=tk.X, pady=5)
        
        ansvr_path_label = ttk.Label(ansvr_path_subframe, 
                                     text=self.parent_gui.tr("local_solver_ansvr_path_label", default="solve-field Path or astrometry.cfg:"),  # Label clarifié
                                     width=25, anchor="w")
        ansvr_path_label.pack(side=tk.LEFT)
        
        ansvr_browse_button = ttk.Button(ansvr_path_subframe, 
                                         text=self.parent_gui.tr("browse", default="Browse..."), 
                                         command=self._browse_local_ansvr_path, width=10)
        ansvr_browse_button.pack(side=tk.RIGHT, padx=(5,0))
        
        ansvr_path_entry = ttk.Entry(ansvr_path_subframe, textvariable=self.local_ansvr_path_var)
        ansvr_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,0))

        # --- Info/Aide (Optionnel) ---
        info_label = ttk.Label(main_frame, 
                               text=self.parent_gui.tr("local_solver_info_text_v2", # Nouvelle clé pour le texte
                                                       default="Select a local solver preference above. Paths are only needed for the selected solver.\n"
                                                               "For 'solve-field', provide the path to the executable or an 'astrometry.cfg' file.\n"
                                                               "If a solver is not in system PATH, provide the full path to its executable."), 
                               justify=tk.LEFT, wraplength=480) 
        info_label.pack(pady=(10,5), padx=5, fill=tk.X)


        # --- Boutons OK / Annuler ---
        button_frame = ttk.Frame(main_frame, padding="5")
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        
        cancel_button = ttk.Button(button_frame, 
                                   text=self.parent_gui.tr("cancel", default="Cancel"), 
                                   command=self._on_cancel)
        cancel_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        ok_button = ttk.Button(button_frame, 
                               text=self.parent_gui.tr("ok", default="OK"), 
                               command=self._on_ok)
        ok_button.pack(side=tk.RIGHT)

        print("DEBUG (LocalSolverSettingsWindow _build_ui): UI construite avec RadioButtons et logique d'état.") # DEBUG


    def _browse_astap_path(self):
        initial_dir = ""
        current_path = self.astap_path_var.get()
        if current_path and os.path.exists(os.path.dirname(current_path)):
            initial_dir = os.path.dirname(current_path)
        elif os.path.exists(current_path):
             initial_dir = os.path.dirname(current_path)

        file_types = [(self.parent_gui.tr("executable_files", default="Executable Files"), "*.*")] 
        if os.name == 'nt': 
            file_types = [(self.parent_gui.tr("astap_executable_win", default="ASTAP Executable"), "*.exe"), 
                          (self.parent_gui.tr("all_files", default="All Files"), "*.*")]
        
        filepath = filedialog.askopenfilename(
            title=self.parent_gui.tr("select_astap_executable_title", default="Select ASTAP Executable"),
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
            title=self.parent_gui.tr("select_astap_data_dir_title", default="Select ASTAP Star Index Data Directory"),
            initialdir=initial_dir,
            parent=self 
        )
        if dirpath: 
            self.astap_data_dir_var.set(dirpath)
            print(f"DEBUG (LocalSolverSettingsWindow): Répertoire données ASTAP sélectionné: {dirpath}")


    def _browse_local_ansvr_path(self):
        """
        Ouvre un dialogue pour sélectionner le chemin de solve-field ou un astrometry.cfg.
        """
        initial_dir_ansvr = ""
        current_path_ansvr = self.local_ansvr_path_var.get()
        if current_path_ansvr:
            if os.path.isfile(current_path_ansvr):
                initial_dir_ansvr = os.path.dirname(current_path_ansvr)
            elif os.path.isdir(current_path_ansvr): # Si c'est un dossier (pourrait contenir un cfg)
                initial_dir_ansvr = current_path_ansvr
        
        if not initial_dir_ansvr:
            initial_dir_ansvr = os.path.expanduser("~")

        # Permettre de sélectionner des fichiers (exe, cfg) ou des dossiers
        filepath_ansvr = filedialog.askopenfilename( # Modifié pour être plus générique
            title=self.parent_gui.tr("select_local_ansvr_path_title_v2", default="Select solve-field Executable or astrometry.cfg"),
            initialdir=initial_dir_ansvr,
            filetypes=[
                (self.parent_gui.tr("configuration_files", default="Configuration Files"), "*.cfg"),
                (self.parent_gui.tr("executable_files", default="Executable Files"), "*.*" if os.name != 'nt' else "*.exe"), # Adapté pour OS
                (self.parent_gui.tr("all_files", default="All Files"), "*.*")
            ],
            parent=self
        )
        # Si l'utilisateur annule askopenfilename, on pourrait tenter askdirectory
        # mais c'est peut-être trop complexe. L'utilisateur peut taper un chemin de dossier.

        if filepath_ansvr:
            self.local_ansvr_path_var.set(filepath_ansvr)
            print(f"DEBUG (LocalSolverSettingsWindow): Chemin solve-field/cfg sélectionné: {filepath_ansvr}")


    def _on_ok(self):
        """
        Appelé lorsque l'utilisateur clique sur OK.
        Sauvegarde les paramètres et ferme la fenêtre.
        """
        print("DEBUG (LocalSolverSettingsWindow _on_ok): Bouton OK cliqué.") # DEBUG
        
        # --- MODIFIÉ : Lire la nouvelle variable de choix ---
        solver_choice = self.local_solver_choice_var.get()
        astap_path = self.astap_path_var.get().strip()
        astap_data_dir = self.astap_data_dir_var.get().strip()
        astap_radius = self.astap_search_radius_var.get() # Lire la valeur du DoubleVar
        local_ansvr_path = self.local_ansvr_path_var.get().strip()

        # Valider que si un solveur est choisi, son chemin principal est rempli
        validation_ok = True
        if solver_choice == "astap" and not astap_path:
            messagebox.showerror(self.parent_gui.tr("error"), 
                                 self.parent_gui.tr("astap_path_required_error", default="ASTAP is selected, but the executable path is missing."),
                                 parent=self)
            validation_ok = False
        elif solver_choice == "ansvr" and not local_ansvr_path:
            messagebox.showerror(self.parent_gui.tr("error"), 
                                 self.parent_gui.tr("ansvr_path_required_error", default="Astrometry.net Local is selected, but the path/config is missing."),
                                 parent=self)
            validation_ok = False
        
        if not validation_ok:
            print("DEBUG (LocalSolverSettingsWindow _on_ok): Validation échouée (chemin manquant pour solveur choisi).") # DEBUG
            return # Ne pas fermer la fenêtre

        # --- MODIFIÉ : Sauvegarder la nouvelle préférence et supprimer l'ancienne ---
        # Nous allons ajouter 'local_solver_preference' à SettingsManager plus tard
        setattr(self.parent_gui.settings, 'local_solver_preference', solver_choice)
        if hasattr(self.parent_gui.settings, 'use_local_solver_priority'):
            delattr(self.parent_gui.settings, 'use_local_solver_priority') # Supprimer l'ancien attribut
        
        self.parent_gui.settings.astap_path = astap_path
        self.parent_gui.settings.astap_data_dir = astap_data_dir
        setattr(self.parent_gui.settings, 'astap_search_radius', astap_radius) # Sauvegarder le rayon
        self.parent_gui.settings.local_ansvr_path = local_ansvr_path
        
        print(f"  LocalSolverSettingsWindow: Préférence Sauvegardée='{solver_choice}', ASTAP='{astap_path}', Data ASTAP='{astap_data_dir}', Radius ASTAP={astap_radius}, Ansvr Local='{local_ansvr_path}'") # DEBUG
        print("  LocalSolverSettingsWindow: Paramètres mis à jour dans parent_gui.settings.")
        
        self.grab_release()
        self.destroy()

    def _on_cancel(self):
        print("DEBUG (LocalSolverSettingsWindow _on_cancel): Fenêtre annulée/fermée.") # DEBUG
        self.grab_release()
        self.destroy()

# --- END OF FILE seestar/gui/local_solver_gui.py ---