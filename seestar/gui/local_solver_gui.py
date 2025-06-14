"""
Module pour la fenêtre de configuration des solveurs astrométriques locaux.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os # Pour les opérations sur les chemins
from zemosaic import zemosaic_config

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
        # Load fallback config if the parent GUI lacks one
        if not hasattr(self.parent_gui, "config"):
            try:
                self.parent_gui.config = zemosaic_config.load_config()
            except Exception:
                self.parent_gui.config = {}
        self.withdraw()  # Cacher pendant la configuration

        self.title(self.parent_gui.tr("local_solver_window_title", default="Local Astrometry Solvers Configuration"))
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

        self._on_solver_choice_change()
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

        info_label = ttk.Label(
            main_frame,
            text=self.parent_gui.tr("solver_settings_moved", default="Astrometry solver settings are now available in Mosaic Options."),
            justify=tk.LEFT,
            wraplength=400,
        )
        info_label.pack(pady=10, padx=5)

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
        astap_radius = self.astap_search_radius_var.get()
        astap_downsample = self.astap_downsample_var.get()
        astap_sensitivity = self.astap_sensitivity_var.get()
        cluster_threshold = self.cluster_threshold_var.get()
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
        
        print(
            f"  LocalSolverSettingsWindow: Préférence Sauvegardée='{solver_choice}', ASTAP='{astap_path}', Data ASTAP='{astap_data_dir}', Radius ASTAP={astap_radius}, Down={astap_downsample}, Sens={astap_sensitivity}, Cluster={cluster_threshold}, Ansvr Local='{local_ansvr_path}'"
        )  # DEBUG
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