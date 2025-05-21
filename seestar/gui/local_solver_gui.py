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

####################################################################################################################################################





# --- DANS LA CLASSE LocalSolverSettingsWindow DANS seestar/gui/local_solver_gui.py ---

    def _build_ui(self):
        """
        Construit les widgets de l'interface utilisateur pour cette fenêtre.
        MODIFIED: Section Ansvr avec deux boutons "Browse".
        """
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # --- Section Choix Solveur (inchangée par rapport à ta version précédente) ---
        solver_choice_frame = ttk.LabelFrame(main_frame,
                                             text=self.parent_gui.tr("local_solver_choice_frame_title", default="Local Solver Preference"),
                                             padding="10")
        solver_choice_frame.pack(fill=tk.X, padx=5, pady=(0, 10))
        ttk.Radiobutton(solver_choice_frame, text=self.parent_gui.tr("local_solver_choice_none", default="Do not use local solvers (use Astrometry.net web service if API key provided)"), variable=self.local_solver_choice_var, value="none", command=self._on_solver_choice_change).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(solver_choice_frame, text=self.parent_gui.tr("local_solver_choice_astap", default="Use ASTAP (local)"), variable=self.local_solver_choice_var, value="astap", command=self._on_solver_choice_change).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(solver_choice_frame, text=self.parent_gui.tr("local_solver_choice_ansvr", default="Use Astrometry.net Local (solve-field)"), variable=self.local_solver_choice_var, value="ansvr", command=self._on_solver_choice_change).pack(anchor=tk.W, pady=2)

        # --- Configuration ASTAP (inchangée par rapport à ta version précédente) ---
        self.astap_frame = ttk.LabelFrame(main_frame, text=self.parent_gui.tr("local_solver_astap_frame_title", default="ASTAP Configuration"), padding="10")
        self.astap_frame.pack(fill=tk.X, padx=5, pady=5)
        # ... (contenu du cadre ASTAP - lignes pour path, data dir, search radius - comme avant)
        astap_path_subframe = ttk.Frame(self.astap_frame); astap_path_subframe.pack(fill=tk.X, pady=(5, 2))
        ttk.Label(astap_path_subframe, text=self.parent_gui.tr("local_solver_astap_path_label", default="ASTAP Executable Path:"), width=25, anchor="w").pack(side=tk.LEFT)
        ttk.Button(astap_path_subframe, text=self.parent_gui.tr("browse", default="Browse..."), command=self._browse_astap_path, width=10).pack(side=tk.RIGHT, padx=(5,0))
        ttk.Entry(astap_path_subframe, textvariable=self.astap_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,0))
        astap_data_subframe = ttk.Frame(self.astap_frame); astap_data_subframe.pack(fill=tk.X, pady=(2, 5))
        ttk.Label(astap_data_subframe, text=self.parent_gui.tr("local_solver_astap_data_label", default="ASTAP Star Index Data Directory:"), width=25, anchor="w").pack(side=tk.LEFT)
        ttk.Button(astap_data_subframe, text=self.parent_gui.tr("browse", default="Browse..."), command=self._browse_astap_data_dir, width=10).pack(side=tk.RIGHT, padx=(5,0))
        ttk.Entry(astap_data_subframe, textvariable=self.astap_data_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,0))
        astap_radius_subframe = ttk.Frame(self.astap_frame); astap_radius_subframe.pack(fill=tk.X, pady=(2,5))
        ttk.Label(astap_radius_subframe, text=self.parent_gui.tr("local_solver_astap_radius_label", default="ASTAP Search Radius (degrees, 0 for auto):"), width=35, anchor="w").pack(side=tk.LEFT, padx=(0,5))
        ttk.Spinbox(astap_radius_subframe, from_=0.0, to=180.0, increment=0.5, textvariable=self.astap_search_radius_var, width=6, format="%.1f").pack(side=tk.LEFT)


        # --- MODIFICATION : Configuration Astrometry.net Local (ansvr) avec deux boutons ---
        self.ansvr_frame = ttk.LabelFrame(main_frame,
                                     text=self.parent_gui.tr("local_solver_ansvr_frame_title", default="Local Astrometry.net (solve-field) Configuration"),
                                     padding="10")
        self.ansvr_frame.pack(fill=tk.X, padx=5, pady=5)

        # Cadre pour le label et le champ de saisie du chemin principal
        ansvr_path_entry_frame = ttk.Frame(self.ansvr_frame)
        ansvr_path_entry_frame.pack(fill=tk.X, pady=(5,2)) # Un peu de padding en bas pour espacer des boutons

        ansvr_path_label = ttk.Label(ansvr_path_entry_frame,
                                     text=self.parent_gui.tr("local_solver_ansvr_main_path_label", default="Path (Exe, .cfg, or Index Dir):"), # Label plus générique
                                     width=25, anchor="w") # Augmenter légèrement la largeur si besoin
        ansvr_path_label.pack(side=tk.LEFT, padx=(0,5)) # Un peu de padding à droite du label

        ansvr_path_entry = ttk.Entry(ansvr_path_entry_frame, textvariable=self.local_ansvr_path_var)
        ansvr_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Cadre pour les deux boutons "Browse", alignés à droite
        ansvr_buttons_frame = ttk.Frame(self.ansvr_frame)
        ansvr_buttons_frame.pack(fill=tk.X, pady=(0,5)) # Padding en bas pour espacer de la section suivante

        # Sous-cadre pour aligner les boutons à droite de ansvr_buttons_frame
        ansvr_buttons_sub_right_frame = ttk.Frame(ansvr_buttons_frame)
        ansvr_buttons_sub_right_frame.pack(side=tk.RIGHT) # Aligner ce sous-cadre à droite

        self.ansvr_browse_file_button = ttk.Button(ansvr_buttons_sub_right_frame, # Parent est le sous-cadre
                                                   text=self.parent_gui.tr("browse_ansvr_file_button", default="Browse File... (.cfg/Exe)"),
                                                   command=self._browse_ansvr_file, # Nouvelle méthode callback
                                                   width=20) # Ajuster largeur si besoin
        self.ansvr_browse_file_button.pack(side=tk.LEFT, padx=(0,5)) # Espacement entre les boutons

        self.ansvr_browse_dir_button = ttk.Button(ansvr_buttons_sub_right_frame, # Parent est le sous-cadre
                                                  text=self.parent_gui.tr("browse_ansvr_dir_button", default="Browse Index Dir..."),
                                                  command=self._browse_ansvr_index_dir, # Nouvelle méthode callback
                                                  width=20) # Ajuster largeur si besoin
        self.ansvr_browse_dir_button.pack(side=tk.LEFT)
        # --- FIN MODIFICATION SECTION ANSVR ---

        # --- Info/Aide (Modifier la clé de traduction pour le nouveau texte) ---
        info_label = ttk.Label(main_frame,
                               text=self.parent_gui.tr("local_solver_info_text_v3", # Nouvelle clé pour le texte
                                                       default="Select a local solver preference above. Paths are only needed for the selected solver.\n"
                                                               "For Ansvr, you can point to `solve-field` exe, an `astrometry.cfg` file, or an Index Directory (a .cfg will be auto-generated)."),
                               justify=tk.LEFT, wraplength=480) # Augmenter wraplength si le texte est plus long
        info_label.pack(pady=(10,5), padx=5, fill=tk.X)

        # --- Boutons OK / Annuler (inchangés) ---
        button_frame = ttk.Frame(main_frame, padding="5")
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        # ... (boutons OK et Annuler comme avant) ...
        cancel_button = ttk.Button(button_frame,
                                   text=self.parent_gui.tr("cancel", default="Cancel"),
                                   command=self._on_cancel)
        cancel_button.pack(side=tk.RIGHT, padx=(5, 0))
        ok_button = ttk.Button(button_frame,
                                    text=self.parent_gui.tr("ok", default="OK"),
                                    command=self._on_ok)
        ok_button.pack(side=tk.RIGHT)

        print("DEBUG (LocalSolverSettingsWindow _build_ui): UI construite avec 2 boutons Browse pour Ansvr (V2).")






####################################################################################################################################################
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
            title=self.parent_gui.tr("select_ansvr_exe_or_cfg_title", default="Select solve-field Executable or .cfg (Cancel for Index Dir)"),
            initialdir=initial_dir_ansvr,
            filetypes=[
                (self.parent_gui.tr("configuration_files", default="Configuration Files"), "*.cfg"),
                (self.parent_gui.tr("executable_files", default="Executable Files"), "*.*" if os.name != 'nt' else "*.exe"),
                (self.parent_gui.tr("all_files", default="All Files"), "*.*")
            ],
            parent=self
        )

        if not filepath_selected: # L'utilisateur a annulé la sélection de fichier ou n'a rien choisi
            _log_browser("Aucun fichier sélectionné. Ouverture de la boîte de dialogue pour répertoires.")
            dirpath_selected = filedialog.askdirectory(
                title=self.parent_gui.tr("select_ansvr_index_dir_title", default="Select Astrometry.net Index Directory"),
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
                (self.parent_gui.tr("configuration_files", default="Config Files"), "*.cfg"),
                (self.parent_gui.tr("executable_files", default="Executable Files (.exe)"), "*.exe"), # Plus spécifique pour Windows
                (self.parent_gui.tr("all_files", default="All Files"), "*.*")
            ]
            # print(f"DEBUG (LSW _browse_ansvr_file): Types de fichiers pour Windows: {file_types_list}") # Optionnel
        else: # Cas non-Windows (Linux, macOS, etc.)
            file_types_list = [
                (self.parent_gui.tr("executable_files", default="Executable Files (any)"), "*"), # Souvent sans extension
                (self.parent_gui.tr("configuration_files", default="Config Files (.cfg)"), "*.cfg"),
                (self.parent_gui.tr("all_files", default="All Files"), "*.*")
            ]
            # print(f"DEBUG (LSW _browse_ansvr_file): Types de fichiers pour non-Windows: {file_types_list}") # Optionnel

        filepath = filedialog.askopenfilename(
            title=self.parent_gui.tr("select_ansvr_exe_or_cfg_title_v2", default="Select solve-field Executable or .cfg File"),
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
            title=self.parent_gui.tr("select_ansvr_index_dir_title_v2", default="Select Astrometry.net Index Directory"),
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
        print(f"DEBUG (LocalSolverSettingsWindow _on_ok): Validation OK. Sauvegarde des settings. Préparation fermeture fenêtre.")
        print(f"  -> local_solver_preference: {solver_choice}")
        print(f"  -> local_ansvr_path à sauvegarder: {local_ansvr_path}")
        self.grab_release()
        self.destroy()

    def _on_cancel(self):
        print("DEBUG (LocalSolverSettingsWindow _on_cancel): Fenêtre annulée/fermée.") # DEBUG
        self.grab_release()
        self.destroy()

# --- END OF FILE seestar/gui/local_solver_gui.py ---