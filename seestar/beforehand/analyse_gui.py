# --- START OF FILE beforehand/analyse_gui.py (COMPLETE & COMMENTED) ---

# === Imports Standard ===
import os
import sys  # Nécessaire pour sys.path
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import matplotlib
matplotlib.use('TkAgg') # Assurer la compatibilité Tkinter pour Matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import datetime
import platform
import subprocess
# import sys # Déjà importé plus haut
import traceback
import time
import gc
import argparse # Pour gérer les arguments de ligne de commande
from PIL import Image, ImageTk
# L'import de ToolTip est déplacé APRES l'ajustement de sys.path
import json

# --- AJUSTEMENT DE SYS.PATH POUR PERMETTRE LES IMPORTS DEPUIS LA RACINE DU PROJET ---
# Ceci est crucial lorsque ce script (analyse_gui.py) est exécuté directement
# ou via subprocess, car Python a besoin de savoir où se trouve le package 'seestar'.
print("DEBUG (analyse_gui.py): Début de l'ajustement de sys.path...")
try:
    # Chemin absolu du script actuel (analyse_gui.py)
    current_script_path = os.path.abspath(__file__)
    print(f"  DEBUG (analyse_gui.py): Chemin du script actuel: {current_script_path}")

    # Remonter pour trouver le dossier 'beforehand'
    beforehand_dir = os.path.dirname(current_script_path)
    print(f"  DEBUG (analyse_gui.py): Dossier 'beforehand': {beforehand_dir}")

    # Remonter encore pour trouver le dossier du package 'seestar'
    seestar_package_dir = os.path.dirname(beforehand_dir)
    print(f"  DEBUG (analyse_gui.py): Dossier du package 'seestar': {seestar_package_dir}")

    # Remonter une dernière fois pour trouver la racine du projet
    # (le dossier qui CONTIENT le dossier 'seestar')
    project_root_dir = os.path.dirname(seestar_package_dir)
    print(f"  DEBUG (analyse_gui.py): Racine du projet calculée: {project_root_dir}")

    # Ajouter la racine du projet au début de sys.path si elle n'y est pas déjà.
    # sys.path.insert(0, ...) la met en priorité pour la recherche de modules.
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)
        print(f"  DEBUG (analyse_gui.py): '{project_root_dir}' ajouté à sys.path.")
    else:
        print(f"  DEBUG (analyse_gui.py): '{project_root_dir}' était déjà dans sys.path.")
    
    # print(f"  DEBUG (analyse_gui.py): sys.path actuel: {sys.path}") # Optionnel, peut être très long
    print("DEBUG (analyse_gui.py): Ajustement de sys.path terminé.")

except Exception as e_path_setup:
    # Gérer les erreurs potentielles lors de la manipulation des chemins
    print(f"ERREUR CRITIQUE (analyse_gui.py): Impossible d'ajuster sys.path correctement: {e_path_setup}")
    # Afficher une boîte de dialogue d'erreur si Tkinter est déjà initialisable
    try:
        root_err_path = tk.Tk(); root_err_path.withdraw()
        messagebox.showerror("Erreur Configuration Chemin", f"Erreur critique lors de la configuration des chemins Python:\n{e_path_setup}\nL'application ne peut pas continuer.")
        root_err_path.destroy()
    except Exception: pass
    sys.exit(1) # Quitter car les imports suivants vont probablement échouer
# --- FIN DE L'AJUSTEMENT SYS.PATH ---


# === Imports Locaux (MAINTENANT APRÈS L'AJUSTEMENT DE SYS.PATH) ===

# Importer ToolTip en utilisant le chemin absolu du package depuis la racine du projet
try:
    from seestar.gui.ui_utils import ToolTip 
    print("DEBUG (analyse_gui.py): Import de 'seestar.gui.ui_utils.ToolTip' réussi.")
except ImportError as e_tooltip:
    print(f"ERREUR CRITIQUE (analyse_gui.py): Impossible d'importer ToolTip depuis seestar.gui.ui_utils. Erreur: {e_tooltip}")
    print(f"  Vérifiez que le chemin ajouté à sys.path ('{project_root_dir if 'project_root_dir' in locals() else 'NON_CALCULE'}') est correct et que le fichier seestar/gui/ui_utils.py existe et est accessible.")
    traceback.print_exc() # Afficher la trace complète de l'ImportError
    try:
        root_err_tooltip = tk.Tk(); root_err_tooltip.withdraw()
        messagebox.showerror("Erreur Module Manquant", f"Impossible d'importer un composant UI essentiel (ToolTip).\nErreur: {e_tooltip}\nL'application va se fermer.")
        root_err_tooltip.destroy()
    except Exception: pass
    sys.exit(1)

# Importe le module contenant la logique d'analyse principale
# Cet import devrait fonctionner car analyse_logic.py est dans le même dossier 'beforehand'
try:
    import analyse_logic
    SATDET_AVAILABLE = analyse_logic.SATDET_AVAILABLE
    SATDET_USES_SEARCHPATTERN = analyse_logic.SATDET_USES_SEARCHPATTERN
    print("DEBUG (analyse_gui.py): Import de 'analyse_logic' réussi.")
except ImportError as e_logic:
    print(f"ERREUR CRITIQUE (analyse_gui.py): Fichier logique manquant (analyse_logic.py introuvable). Erreur: {e_logic}")
    try:
        root_err_logic = tk.Tk(); root_err_logic.withdraw()
        messagebox.showerror("Erreur Fichier Manquant", f"Impossible de charger analyse_logic.py:\n{e_logic}")
        root_err_logic.destroy()
    except Exception: pass
    sys.exit(1)

# Importe le module contenant les textes traduits
# Cet import devrait fonctionner car zone.py est dans le même dossier 'beforehand'
try:
    from zone import translations
    print("DEBUG (analyse_gui.py): Import de 'zone.translations' réussi.")
except ImportError as e_zone:
    print(f"ERREUR CRITIQUE (analyse_gui.py): Fichier de langue zone.py introuvable. Erreur: {e_zone}")
    translations = { # Fallback minimal pour les messages d'erreur
        'en': {'window_title': 'Analyzer - Error', 'msg_missing_zone': 'Language file zone.py is missing.'},
    }
    try:
        root_err_zone = tk.Tk(); root_err_zone.withdraw()
        messagebox.showerror("Erreur Fichier Langue", f"Impossible de charger zone.py:\n{e_zone}")
        root_err_zone.destroy()
    except Exception: pass
    print("AVERTISSEMENT (analyse_gui.py): zone.py manquant, utilisation de textes anglais par défaut très limités.")



# === Classe Utilitaire pour les Infobulles (Tooltips) ===
class ToolTip:
    """Crée une infobulle pour un widget donné."""
    def __init__(self, widget, text_callback):
        self.widget = widget
        self.text_callback = text_callback # Fonction qui retourne le texte de l'infobulle
        self.tooltip_window = None
        self.id = None
        self.x = self.y = 0
        # Lier les événements d'entrée/sortie de la souris au widget
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave) # Cacher si on clique

    def enter(self, event=None):
        """Planifie l'affichage de l'infobulle après un délai."""
        self.schedule()

    def leave(self, event=None):
        """Annule la planification et cache l'infobulle."""
        self.unschedule()
        self.hidetip()

    def schedule(self):
        """Planifie l'appel à showtip après 500ms."""
        self.unschedule() # Annuler toute planification précédente
        # Vérifier si le widget existe toujours avant de planifier
        if self.widget.winfo_exists():
            self.id = self.widget.after(500, self.showtip)

    def unschedule(self):
        """Annule une planification en attente."""
        id_ = self.id
        self.id = None
        if id_:
            try:
                # Vérifier si le widget existe toujours avant d'annuler
                if self.widget.winfo_exists():
                    self.widget.after_cancel(id_)
            except tk.TclError: pass # Ignorer si le widget a été détruit

    def showtip(self):
        """Affiche l'infobulle."""
        # Ne rien faire si l'infobulle est déjà affichée ou si le widget n'existe plus
        if self.tooltip_window or not self.widget.winfo_exists():
            return
        try:
            # Obtenir la position du widget à l'écran
            x_root, y_root = self.widget.winfo_rootx(), self.widget.winfo_rooty()
            # Positionner l'infobulle sous le widget
            y_offset = self.widget.winfo_height() + 5
        except tk.TclError:
            # Le widget a peut-être été détruit entre temps
            self.hidetip()
            return

        x = x_root + 10 # Petit décalage horizontal
        y = y_root + y_offset

        # Vérifier à nouveau si le widget existe juste avant de créer la fenêtre Toplevel
        if not self.widget.winfo_exists():
            self.hidetip()
            return

        # Créer la fenêtre Toplevel pour l'infobulle
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True) # Pas de décorations de fenêtre (titre, bordure)
        tw.wm_geometry(f"+{int(x)}+{int(y)}") # Positionner la fenêtre

        try:
            # Obtenir le texte via la fonction de rappel
            tooltip_text = self.text_callback()
            # Créer le label avec le texte, fond jaune pâle, bordure fine
            label = tk.Label(tw, text=tooltip_text, justify=tk.LEFT,
                             background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                             wraplength=400) # Retour à la ligne automatique
            label.pack(ipadx=1)
        except Exception as e:
            # Gérer les erreurs lors de l'obtention/affichage du texte
            print(f"Erreur obtention/affichage texte infobulle: {e}")
            self.hidetip()

    def hidetip(self):
        """Détruit la fenêtre de l'infobulle si elle existe."""
        tw = self.tooltip_window
        self.tooltip_window = None
        if tw:
            try:
                # Vérifier si la fenêtre existe avant de la détruire
                if tw.winfo_exists():
                    tw.after(0, tw.destroy) # Utiliser after(0, ...) pour être sûr
            except tk.TclError: pass # Ignorer si déjà détruite


# === Classe Principale de l'Interface Graphique ===
class AstroImageAnalyzerGUI:
    """Interface graphique pour l'analyseur d'images astronomiques."""
    def __init__(self, root, command_file_path=None, main_app_callback=None): # <-- AJOUTÉ command_file_path
        """
        Initialise l'interface graphique.

        Args:
            root (tk.Tk or tk.Toplevel): La fenêtre racine ou Toplevel pour cette interface.
            command_file_path (str, optional): Chemin vers le fichier à utiliser pour
                                               la communication avec le GUI principal.
            main_app_callback (callable, optional): Fonction à appeler lors de la fermeture (Retour).
        """
        self.root = root
        self.main_app_callback = main_app_callback # Callback pour retourner au script appelant

        # --- AJOUTÉ: Stockage du chemin du fichier de commande ---
        self.command_file_path = command_file_path
        try:
            # 1. Trouver le chemin absolu de l'icône depuis ce script
            analyzer_script_path = os.path.abspath(__file__)
            beforehand_dir = os.path.dirname(analyzer_script_path)
            # Remonter d'UN niveau pour être à la racine du projet (où se trouve le dossier icon/)
            project_root = os.path.dirname(beforehand_dir) 
            icon_rel_path = os.path.join('icon', 'icon.png') # Chemin relatif depuis la racine
            icon_path = os.path.join(project_root, icon_rel_path)
            icon_path = os.path.normpath(icon_path)
            print(f"DEBUG (analyse_gui __init__): Chemin icône calculé: {icon_path}")

            # 2. Vérifier si le fichier existe
            if os.path.exists(icon_path):
                # 3. Charger et définir l'icône
                icon_image = Image.open(icon_path)
                # Stocker la référence à l'image Tkinter pour éviter la garbage collection
                self.tk_icon = ImageTk.PhotoImage(icon_image)
                # Appliquer à la fenêtre racine de CETTE interface (self.root)
                self.root.iconphoto(True, self.tk_icon)
                print(f"DEBUG (analyse_gui __init__): Icône de fenêtre définie avec succès depuis: {icon_path}")
            else:
                print(f"AVERTISSEMENT (analyse_gui __init__): Fichier icône introuvable: {icon_path}. Icône par défaut utilisée.")
        except ImportError:
             print("AVERTISSEMENT (analyse_gui __init__): Pillow (PIL/ImageTk) non trouvé. Impossible de définir l'icône.")
             self.tk_icon = None 
        except Exception as e_icon:
            print(f"ERREUR (analyse_gui __init__): Impossible de charger/définir l'icône: {e_icon}")
            traceback.print_exc(limit=1) 
            self.tk_icon = None
        # --- FIN AJOUT ---


        if self.command_file_path:
            print(f"DEBUG (analyse_gui __init__): Fichier de commande reçu: {self.command_file_path}")
        else:
            print("AVERTISSEMENT (analyse_gui __init__): Aucun chemin de fichier de commande fourni. La fonction 'Analyser et Empiler' ne communiquera pas avec le stacker.")
        # --- FIN AJOUT ---

        # Variables Tkinter pour lier les widgets aux données
        self.current_lang = tk.StringVar(value='fr') 
        self.current_lang.trace_add('write', self.change_language) 

        self.input_dir = tk.StringVar() 
        self.output_log = tk.StringVar() 
        
        self.output_log.trace_add('write', lambda *args: self._update_log_and_vis_buttons_state())
        
        self.status_text = tk.StringVar() 
        self.progress_var = tk.DoubleVar(value=0.0) 

        # Options d'analyse (Booléens)
        self.analyze_snr = tk.BooleanVar(value=True) 
        self.detect_trails = tk.BooleanVar(value=(SATDET_AVAILABLE and SATDET_USES_SEARCHPATTERN))
        self.sort_by_snr = tk.BooleanVar(value=True) 
        self.include_subfolders = tk.BooleanVar(value=False) 

        # Paramètres Sélection SNR
        self.snr_selection_mode = tk.StringVar(value='percent') 
        self.snr_selection_value = tk.StringVar(value='80') 
        self.snr_reject_dir = tk.StringVar() 

        # Paramètres Détection Traînées (acstools.satdet)
        self.trail_params = {
            'sigma': tk.StringVar(value="1.85"), 'low_thresh': tk.StringVar(value="0.075"),
            'h_thresh': tk.StringVar(value="0.315"), 'line_len': tk.StringVar(value="100"),
            'small_edge': tk.StringVar(value="1"), 'line_gap': tk.StringVar(value="25")
        }
        self.trail_reject_dir = tk.StringVar() 

        # Action sur les images rejetées
        self.reject_action = tk.StringVar(value='move') 

        # Variables d'état internes
        self.analysis_results = [] 
        self.analysis_running = False 
        self.analysis_completed_successfully = False 
        self.tooltips = {} 
        self.timer_running = False 
        self.timer_start_time = None 
        self.timer_job_id = None 
        self.base_status_message = "" 
        self.has_pending_snr_actions = False
        
        # Références aux widgets (pour traduction, activation/désactivation)
        self.widgets_refs = {}
        self.snr_select_frame = self.snr_value_entry = self.snr_reject_dir_frame = None
        self.snr_reject_dir_entry = self.detect_trails_check = self.acstools_status_label = None
        self.params_sat_frame = self.trail_reject_dir_frame = self.trail_reject_dir_entry = None
        self.analyze_button = self.visualize_button = self.open_log_button = None
        self.analyze_stack_button = None 
        self.return_button = self.progress_bar = self.status_label = self.results_text = None
        self.lang_combobox = None
        self.trail_param_labels = {}
        self.trail_param_entries = {}
        self.manage_markers_button = None
        self.stack_after_analysis = False      
        self.apply_snr_button = None 
        
        # Vérifier si les traductions ont été chargées
        if 'translations' not in globals() or not translations:
            messagebox.showerror("Erreur Critique", "Fichier de langue 'zone.py' manquant ou invalide.")
            sys.exit(1)

        # Créer les widgets de l'interface
        self.create_widgets()
        # Appliquer la langue par défaut aux widgets
        self.change_language()
        # Mettre à jour l'état initial des sections (activé/désactivé)
        self.toggle_sections_state()
        # Définir taille et taille minimale de la fenêtre
        self.root.geometry("950x850")
        self.root.minsize(950, 850)
        self._update_log_and_vis_buttons_state() # Pour l'état initial (si log pré-rempli par args


###################################################################################################################""

    def _load_visualization_data_from_log(self, log_path):
        """
        Tente de charger les données de visualisation JSON depuis la fin du fichier log.
        Retourne True si les données sont chargées avec succès, False sinon.
        """
        print(f"DEBUG_LOAD: Tentative de chargement depuis {log_path}") # NOUVEAU PRINT
        self.analysis_results = [] 
        if not log_path or not os.path.isfile(log_path):
            print(f"DEBUG_LOAD: Fichier log non trouvé ou n'est pas un fichier: {log_path}") # NOUVEAU PRINT
            return False

        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            start_index = -1
            end_index = -1
            
            # Recherche des délimiteurs en partant de la fin (plus robuste)
            # On cherche d'abord END, puis BEGIN avant END
            temp_end_indices = [i for i, line in enumerate(lines) if line.strip() == "--- END VISUALIZATION DATA ---"]
            if not temp_end_indices:
                print("DEBUG_LOAD: Marqueur '--- END VISUALIZATION DATA ---' non trouvé.")
                return False
            
            end_index = temp_end_indices[-1] # Prendre le dernier 'END' trouvé

            temp_start_indices = [i for i, line in enumerate(lines[:end_index]) if line.strip() == "--- BEGIN VISUALIZATION DATA ---"]
            if not temp_start_indices:
                print("DEBUG_LOAD: Marqueur '--- BEGIN VISUALIZATION DATA ---' non trouvé avant le dernier END.")
                return False
            
            start_index = temp_start_indices[-1] # Prendre le dernier 'BEGIN' trouvé avant le dernier 'END'

            print(f"DEBUG_LOAD: start_index={start_index}, end_index={end_index}") # NOUVEAU PRINT
            
            if start_index != -1 and end_index != -1 and start_index < end_index:
                json_str_lines = lines[start_index + 1 : end_index]
                json_str = "".join(json_str_lines)
                
                # print(f"DEBUG_LOAD: JSON string à parser:\n{json_str[:500]}...") # NOUVEAU PRINT (affiche le début du JSON)
                if not json_str.strip():
                    print("DEBUG_LOAD: Section JSON vide entre les délimiteurs.")
                    self.analysis_completed_successfully = False
                    return False

                loaded_data = json.loads(json_str)
                if isinstance(loaded_data, list):
                    self.analysis_results = loaded_data
                    print(f"INFO: Données de visualisation chargées depuis {log_path} ({len(self.analysis_results)} éléments).")
                    self.analysis_completed_successfully = True 
                    return True
                else:
                    print(f"AVERTISSEMENT: Données JSON dans {log_path} ne sont pas une liste. Type: {type(loaded_data)}") # NOUVEAU PRINT
                    self.analysis_completed_successfully = False
                    return False
            else:
                print(f"INFO: Délimiteurs de données de visualisation non trouvés ou mal ordonnés dans {log_path}.") # NOUVEAU PRINT
                self.analysis_completed_successfully = False
                return False

        except json.JSONDecodeError as e_json_dec:
            print(f"ERREUR: Échec du décodage JSON depuis {log_path}: {e_json_dec}")
            print(f"DEBUG_LOAD: String JSON fautif (partiel):\n{json_str[:1000] if 'json_str' in locals() else 'Non disponible'}")
            self.analysis_completed_successfully = False
            return False
        except Exception as e:
            print(f"ERREUR: Échec du chargement des données de visualisation depuis {log_path}: {e}")
            traceback.print_exc()
            self.analysis_completed_successfully = False
            return False



    def start_analysis(self):
        """Appelle la logique de lancement SANS l'option d'empiler après."""
        self._launch_analysis(stack_after=False)


    def start_analysis_and_stack(self):
        """Valide les options et lance l'analyse dans un thread avec INTENTION D'EMPILER APRÈS."""
        # Empêcher lancements multiples (identique à start_analysis)
        if self.analysis_running:
            messagebox.showwarning(self._("msg_warning"), self._("msg_analysis_running"), parent=self.root)
            return

        # --- Définir le flag pour empiler ---
        self.stack_after_analysis = True
        # --- Fin modification flag ---

        # Le reste de la validation est IDENTIQUE à start_analysis
        # On duplique cette partie pour l'instant. À l'étape suivante, nous factoriserons.
        options = {}
        callbacks = {'status': self.update_status, 'progress': self.update_progress, 'log': self.update_results_text}
        input_dir = self.input_dir.get()
        output_log = self.output_log.get()
        options['analyze_snr'] = self.analyze_snr.get()
        options['detect_trails'] = self.detect_trails.get()
        options['include_subfolders'] = self.include_subfolders.get()

        # Vérifier chemins entrée/log (identique)
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror(self._("msg_error"), self._("msg_input_dir_invalid"), parent=self.root)
            self.stack_after_analysis = False # Réinitialiser en cas d'erreur
            return
        if not output_log:
            messagebox.showerror(self._("msg_error"), self._("msg_log_file_missing"), parent=self.root)
            self.stack_after_analysis = False # Réinitialiser
            return

        # Configurer action rejet (identique)
        reject_action = self.reject_action.get()
        options['move_rejected'] = (reject_action == 'move')
        options['delete_rejected'] = (reject_action == 'delete')
        options['trail_reject_dir'] = self.trail_reject_dir.get() if options['move_rejected'] else None
        options['snr_reject_dir'] = self.snr_reject_dir.get() if options['move_rejected'] else None

        # Vérifier chemins rejet si action 'move' (identique)
        if options['move_rejected']:
            if options['detect_trails'] and SATDET_AVAILABLE and not options['trail_reject_dir']:
                messagebox.showerror(self._("msg_error"), self._("trail_reject_dir_label") + " " + self._("non spécifié"), parent=self.root)
                self.stack_after_analysis = False # Réinitialiser
                return
            if options['analyze_snr'] and self.snr_selection_mode.get() != 'none' and not options['snr_reject_dir']:
                messagebox.showerror(self._("msg_error"), self._("snr_reject_dir_label") + " " + self._("non spécifié"), parent=self.root)
                self.stack_after_analysis = False # Réinitialiser
                return
        # Confirmer suppression si action 'delete' (identique)
        elif options['delete_rejected']:
            if not messagebox.askyesno(self._("msg_warning"), self._("confirm_delete"), parent=self.root):
                self.stack_after_analysis = False # Annulé par l'utilisateur
                return

        # Valider paramètres SNR si analyse activée (identique)
        if options['analyze_snr']:
            options['snr_selection_mode'] = self.snr_selection_mode.get()
            options['snr_selection_value'] = self.snr_selection_value.get()
            if options['snr_selection_mode'] in ['percent', 'threshold']:
                if not options['snr_selection_value']:
                    messagebox.showerror(self._("msg_error"), self._("snr_value_missing"), parent=self.root)
                    self.stack_after_analysis = False; return
                try: float(options['snr_selection_value'])
                except ValueError: messagebox.showerror(self._("msg_error"), self._("snr_value_invalid"), parent=self.root); self.stack_after_analysis = False; return
            else: options['snr_selection_value'] = None

        # Valider paramètres détection traînées si activée (identique)
        if options['detect_trails']:
            if not (SATDET_AVAILABLE and SATDET_USES_SEARCHPATTERN):
                messagebox.showerror(self._("msg_error"), self._("msg_satdet_incompatible"), parent=self.root)
                self.stack_after_analysis = False; return
            try:
                options['trail_params'] = { k: self.trail_params[k].get() for k in self.trail_params } # Prendre valeurs des StringVar
                options['trail_params'] = { # Convertir en nombres
                    'sigma': float(options['trail_params']['sigma']), 'low_thresh': float(options['trail_params']['low_thresh']),
                    'h_thresh': float(options['trail_params']['h_thresh']), 'line_len': int(options['trail_params']['line_len']),
                    'small_edge': int(options['trail_params']['small_edge']), 'line_gap': int(options['trail_params']['line_gap'])
                }
                if options['trail_params']['h_thresh'] < options['trail_params']['low_thresh']:
                    messagebox.showerror(self._("msg_error"), self._("msg_params_invalid", e="High Thresh doit être >= Low Thresh"), parent=self.root)
                    self.stack_after_analysis = False; return
            except ValueError as e: messagebox.showerror(self._("msg_error"), self._("msg_params_invalid", e=f"Paramètres Traînées: {e}"), parent=self.root); self.stack_after_analysis = False; return
        else: options['trail_params'] = {}

        # --- Préparer UI pour l'analyse --- (identique)
        self.reset_ui_for_new_analysis()
        self.analysis_running = True
        self._set_widget_state(self.analyze_button, tk.DISABLED)
        self._set_widget_state(self.analyze_stack_button, tk.DISABLED) # <-- Désactiver le nouveau bouton aussi
        self._set_widget_state(self.visualize_button, tk.DISABLED)
        self._set_widget_state(self.open_log_button, tk.DISABLED)
        self._set_widget_state(self.return_button, tk.DISABLED)
        self._set_widget_state(self.manage_markers_button, tk.DISABLED)
        self.update_status("status_analysis_start")
        self.update_results_text("--- Début de l'analyse (pour empiler ensuite) ---", clear=True) # Message légèrement différent
        self.update_progress(0.0)

        # --- Lancer le thread d'analyse --- (identique)
        analysis_thread = threading.Thread(
            target=self.run_analysis_thread,
            args=(input_dir, output_log, options, callbacks),
            daemon=True
        )
        analysis_thread.start()


    def _update_log_and_vis_buttons_state(self):
        """
        Active/désactive les boutons 'Ouvrir Log' et 'Visualiser les résultats'
        basé sur l'existence du fichier log et la présence de données de visualisation.
        """
        log_path = self.output_log.get()
        log_exists = log_path and os.path.isfile(log_path)

        # Gérer le bouton "Ouvrir Log"
        if hasattr(self, 'open_log_button') and self.open_log_button:
            self._set_widget_state(self.open_log_button, tk.NORMAL if log_exists else tk.DISABLED)

        # Gérer le bouton "Visualiser les résultats"
        can_visualize = False
        if log_exists:
            # Tenter de charger les données. Si succès, analysis_results sera rempli.
            if self._load_visualization_data_from_log(log_path):
                can_visualize = bool(self.analysis_results) # Vrai si la liste n'est pas vide
        
        if not can_visualize: # Si le chargement a échoué ou si pas de données, vider
            self.analysis_results = []
            self.analysis_completed_successfully = False # Réinitialiser

        if hasattr(self, 'visualize_button') and self.visualize_button:
            self._set_widget_state(self.visualize_button, tk.NORMAL if can_visualize else tk.DISABLED)

# --- DANS analyse_gui.py ---

    def _launch_analysis(self, stack_after: bool):
        """Méthode interne pour valider et lancer le thread d'analyse."""
        if self.analysis_running:
            messagebox.showwarning(self._("msg_warning"), self._("msg_analysis_running"), parent=self.root)
            return False 

        self.stack_after_analysis = stack_after

        options = {}
        callbacks = {'status': self.update_status, 'progress': self.update_progress, 'log': self.update_results_text}
        input_dir = self.input_dir.get()
        output_log = self.output_log.get()
        options['analyze_snr'] = self.analyze_snr.get()
        options['detect_trails'] = self.detect_trails.get()
        options['include_subfolders'] = self.include_subfolders.get()

        # --- NOUVEAU : Définir l'option pour l'action SNR différée ---
        options['apply_snr_action_immediately'] = False # On veut toujours différer l'action SNR depuis le GUI
        # --- FIN NOUVEAU ---

        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror(self._("msg_error"), self._("msg_input_dir_invalid"), parent=self.root)
            self.stack_after_analysis = False 
            return False
        if not output_log:
            messagebox.showerror(self._("msg_error"), self._("msg_log_file_missing"), parent=self.root)
            self.stack_after_analysis = False
            return False

        reject_action = self.reject_action.get()
        options['move_rejected'] = (reject_action == 'move')
        options['delete_rejected'] = (reject_action == 'delete')
        options['trail_reject_dir'] = self.trail_reject_dir.get() if options['move_rejected'] else None
        options['snr_reject_dir'] = self.snr_reject_dir.get() if options['move_rejected'] else None

        if options['move_rejected']:
            # Si le déplacement est activé, les chemins de rejet doivent être spécifiés
            # pour les types d'analyse qui sont activés.
            if options['detect_trails'] and SATDET_AVAILABLE and not options['trail_reject_dir']:
                messagebox.showerror(self._("msg_error"), self._("trail_reject_dir_label") + " " + self._("non spécifié"), parent=self.root)
                self.stack_after_analysis = False; return False
            # Pour SNR, on vérifie le dossier de rejet même si l'action est différée, car on en aura besoin plus tard.
            if options['analyze_snr'] and self.snr_selection_mode.get() != 'none' and not options['snr_reject_dir']:
                messagebox.showerror(self._("msg_error"), self._("snr_reject_dir_label") + " " + self._("non spécifié"), parent=self.root)
                self.stack_after_analysis = False; return False
        elif options['delete_rejected']:
            if not messagebox.askyesno(self._("msg_warning"), self._("confirm_delete"), parent=self.root):
                self.stack_after_analysis = False; return False

        if options['analyze_snr']:
            options['snr_selection_mode'] = self.snr_selection_mode.get()
            options['snr_selection_value'] = self.snr_selection_value.get()
            if options['snr_selection_mode'] in ['percent', 'threshold']:
                if not options['snr_selection_value']:
                    messagebox.showerror(self._("msg_error"), self._("snr_value_missing"), parent=self.root)
                    self.stack_after_analysis = False; return False
                try: float(options['snr_selection_value'])
                except ValueError: 
                    messagebox.showerror(self._("msg_error"), self._("snr_value_invalid"), parent=self.root)
                    self.stack_after_analysis = False; return False
            else: options['snr_selection_value'] = None # Pour 'none'

        if options['detect_trails']:
            if not (SATDET_AVAILABLE and SATDET_USES_SEARCHPATTERN):
                messagebox.showerror(self._("msg_error"), self._("msg_satdet_incompatible"), parent=self.root)
                self.stack_after_analysis = False; return False
            try:
                options['trail_params'] = { k: self.trail_params[k].get() for k in self.trail_params }
                options['trail_params'] = { 
                    'sigma': float(options['trail_params']['sigma']), 
                    'low_thresh': float(options['trail_params']['low_thresh']),
                    'h_thresh': float(options['trail_params']['h_thresh']), 
                    'line_len': int(options['trail_params']['line_len']),
                    'small_edge': int(options['trail_params']['small_edge']), 
                    'line_gap': int(options['trail_params']['line_gap'])
                }
                if options['trail_params']['h_thresh'] < options['trail_params']['low_thresh']:
                    messagebox.showerror(self._("msg_error"), self._("msg_params_invalid", e="High Thresh doit être >= Low Thresh"), parent=self.root)
                    self.stack_after_analysis = False; return False
            except ValueError as e: 
                messagebox.showerror(self._("msg_error"), self._("msg_params_invalid", e=f"Paramètres Traînées: {e}"), parent=self.root)
                self.stack_after_analysis = False; return False
        else: options['trail_params'] = {}

        self.reset_ui_for_new_analysis()
        self.analysis_running = True
        self._set_widget_state(self.analyze_button, tk.DISABLED)
        self._set_widget_state(self.analyze_stack_button, tk.DISABLED) 
        self._set_widget_state(self.visualize_button, tk.DISABLED)
        self._set_widget_state(self.open_log_button, tk.DISABLED)
        self._set_widget_state(self.return_button, tk.DISABLED)
        self._set_widget_state(self.manage_markers_button, tk.DISABLED)
        # --- NOUVEAU : Désactiver le bouton "Appliquer Rejet SNR" pendant l'analyse ---
        if hasattr(self, 'apply_snr_button') and self.apply_snr_button:
            self._set_widget_state(self.apply_snr_button, tk.DISABLED)
        # --- FIN NOUVEAU ---
        
        self.update_status("status_analysis_start")
        log_start_msg = "--- Début de l'analyse (pour empiler ensuite) ---" if stack_after else "--- Début de l'analyse ---"
        self.update_results_text(log_start_msg, clear=True)
        self.update_progress(0.0)

        analysis_thread = threading.Thread(
            target=self.run_analysis_thread,
            args=(input_dir, output_log, options, callbacks),
            daemon=True
        )
        analysis_thread.start()
        return True 
    
    def visualize_results(self):
        """Affiche les graphiques de visualisation dans une nouvelle fenêtre."""
        # Vérifier s'il y a des résultats à afficher
        if not self.analysis_results and not self.analysis_running:
            messagebox.showinfo(self._("msg_info"), self._("msg_no_results_visualize"), parent=self.root)
            return
        # Empêcher visualisation si analyse en cours
        if self.analysis_running:
            messagebox.showwarning(self._("msg_warning"), self._("msg_analysis_wait_visualize"), parent=self.root)
            return
        # Avertir si analyse terminée avec erreurs
        if not self.analysis_completed_successfully:
            messagebox.showwarning(self._("msg_warning"), self._("msg_results_incomplete") + "\n" + self._("Affichage des données disponibles.", default="Displaying available data."), parent=self.root)

        # --- Bloc try principal pour la création de la fenêtre ---
        vis_window = None # Initialiser à None
        canvas_list = []
        figures_list = []
        try:
            # Créer la fenêtre Toplevel pour la visualisation
            vis_window = tk.Toplevel(self.root)
            vis_window.title(self._("visu_window_title"))
            vis_window.geometry("850x650")
            vis_window.transient(self.root) # Lier à la fenêtre principale
            vis_window.grab_set() # Rendre modale

            # Notebook pour les différents onglets de graphiques/données
            notebook = ttk.Notebook(vis_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # --- Onglet Distribution SNR ---
            snr_tab = ttk.Frame(notebook)
            notebook.add(snr_tab, text=self._("visu_tab_snr_dist"))
            fig1 = None # Initialiser figure à None
            try:
                fig1, ax1 = plt.subplots(figsize=(7, 5))
                figures_list.append(fig1)
                valid_snrs = [r['snr'] for r in self.analysis_results if r.get('status')=='ok' and 'snr' in r and np.isfinite(r['snr'])]
                if valid_snrs:
                    ax1.hist(valid_snrs, bins=20, color='skyblue', edgecolor='black')
                    ax1.set_title(self._("visu_snr_dist_title"))
                    ax1.set_xlabel(self._("visu_snr_dist_xlabel"))
                    ax1.set_ylabel(self._("visu_snr_dist_ylabel"))
                    ax1.grid(axis='y', linestyle='--', alpha=0.7)
                else:
                    ax1.text(0.5, 0.5, self._("visu_snr_dist_no_data"), ha='center', va='center', fontsize=12, color='red')
                canvas1 = FigureCanvasTkAgg(fig1, master=snr_tab)
                canvas1.draw()
                canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                canvas_list.append(canvas1)
            except Exception as e:
                print(f"Erreur Histogramme SNR: {e}"); ttk.Label(snr_tab, text=f"{self._('msg_error')}:\n{e}\n{traceback.format_exc()}").pack()
                if fig1: plt.close(fig1) # Fermer figure si erreur canvas

            # --- Onglet Comparaison SNR (Top/Bottom N) ---
            comp_tab = ttk.Frame(notebook)
            notebook.add(comp_tab, text=self._("visu_tab_snr_comp"))
            fig2 = None
            try:
                valid_res = [r for r in self.analysis_results if r.get('status')=='ok' and 'snr' in r and np.isfinite(r['snr']) and 'file' in r]
                if len(valid_res) >= 2:
                    sorted_res = sorted(valid_res, key=lambda x: x['snr'], reverse=True)
                    num_total = len(sorted_res); num_show = min(10, num_total // 2 if num_total >= 2 else 1)
                    best = sorted_res[:num_show]; worst = sorted_res[-num_show:]
                    fig_height = max(4, num_show * 0.5)
                    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(10, fig_height))
                    figures_list.append(fig2)
                    best_labels = [os.path.basename(r['file']) for r in best]; ax2.barh(best_labels, [r['snr'] for r in best], color='mediumseagreen', edgecolor='black'); ax2.set_title(self._("visu_snr_comp_best_title", n=num_show)); ax2.invert_yaxis(); ax2.set_xlabel(self._("visu_snr_comp_xlabel")); ax2.tick_params(axis='y', labelsize=8)
                    worst_labels = [os.path.basename(r['file']) for r in worst]; ax3.barh(worst_labels, [r['snr'] for r in worst], color='salmon', edgecolor='black'); ax3.set_title(self._("visu_snr_comp_worst_title", n=num_show)); ax3.invert_yaxis(); ax3.set_xlabel(self._("visu_snr_comp_xlabel")); ax3.tick_params(axis='y', labelsize=8)
                    fig2.tight_layout(pad=1.5); canvas2 = FigureCanvasTkAgg(fig2, master=comp_tab); canvas2.draw(); canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True); canvas_list.append(canvas2)
                else: ttk.Label(comp_tab, text=self._("visu_snr_comp_no_data")).pack(padx=10, pady=10)
            except Exception as e:
                print(f"Erreur Comparaison SNR: {e}"); ttk.Label(comp_tab, text=f"{self._('msg_error')}:\n{e}\n{traceback.format_exc()}").pack()
                if fig2: plt.close(fig2)

            # --- Onglet Traînées Satellites (Camembert) ---
            detect_trails_was_active = any('has_trails' in r for r in self.analysis_results)
            if detect_trails_was_active:
                sat_tab = ttk.Frame(notebook); notebook.add(sat_tab, text=self._("visu_tab_sat_trails")); fig3 = None
                try:
                    sat_count = sum(1 for r in self.analysis_results if r.get('has_trails', False)); no_sat_count = sum(1 for r in self.analysis_results if 'has_trails' in r and not r.get('has_trails')); total_analyzed_for_trails = sat_count + no_sat_count
                    if total_analyzed_for_trails > 0:
                        fig3, ax4 = plt.subplots(figsize=(6, 6)); figures_list.append(fig3); labels = [self._("visu_sat_pie_without"), self._("visu_sat_pie_with")]; sizes = [no_sat_count, sat_count]; colors = ['#66b3ff', '#ff9999']; explode = (0, 0.1 if sat_count > 0 and no_sat_count > 0 else 0)
                        wedges, texts, autotexts = ax4.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90); ax4.axis('equal'); ax4.set_title(self._("visu_sat_pie_title")); plt.setp(autotexts, size=10, weight="bold", color="white"); plt.setp(texts, size=10)
                        canvas3 = FigureCanvasTkAgg(fig3, master=sat_tab); canvas3.draw(); canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True); canvas_list.append(canvas3)
                    else: ttk.Label(sat_tab, text=self._("visu_sat_pie_no_data")).pack(padx=10, pady=10)
                except Exception as e:
                    print(f"Erreur Camembert Satellites: {e}"); ttk.Label(sat_tab, text=f"{self._('msg_error')}:\n{e}\n{traceback.format_exc()}").pack()
                    if fig3: plt.close(fig3)

            # --- Onglet Données Détaillées (Tableau Treeview) ---
            data_tab = ttk.Frame(notebook); notebook.add(data_tab, text=self._("visu_tab_raw_data"))
            try:
                cols = ('file', 'status', 'snr', 'bg', 'noise', 'pixsig'); col_names_map = {'file': self._("visu_data_col_file"), 'status': self._("Statut", default="Status"), 'snr': self._("visu_data_col_snr"), 'bg': self._("visu_data_col_bg"), 'noise': self._("visu_data_col_noise"), 'pixsig': self._("visu_data_col_pixsig"), 'trails': self._("visu_data_col_trails"), 'nbseg': self._("visu_data_col_nbseg"), 'action': self._("Action", default="Action"), 'reason': self._("Raison Rejet", default="Reject Reason"), 'comment': self._("Commentaire", default="Comment")}; col_widths = {'file': 250, 'status': 60, 'snr': 60, 'bg': 60, 'noise': 60, 'pixsig': 60, 'trails': 60, 'nbseg': 60, 'action': 80, 'reason': 80, 'comment': 150}; col_anchors = {'file': tk.W, 'status': tk.CENTER, 'snr': tk.CENTER, 'bg': tk.CENTER, 'noise': tk.CENTER, 'pixsig': tk.CENTER, 'trails': tk.CENTER, 'nbseg': tk.CENTER, 'action': tk.W, 'reason': tk.W, 'comment': tk.W}
                if detect_trails_was_active: cols = cols + ('trails', 'nbseg'); cols = cols + ('action', 'reason', 'comment'); col_names = [col_names_map.get(c, c.capitalize()) for c in cols]
                tree = ttk.Treeview(data_tab, columns=cols, show='headings')
                for col_id, col_name in zip(cols, col_names): tree.heading(col_id, text=col_name, command=lambda _col=col_id: self.sort_treeview(tree, _col, False)); tree.column(col_id, width=col_widths.get(col_id, 80), anchor=col_anchors.get(col_id, tk.CENTER), stretch=tk.NO)
                display_res = sorted(self.analysis_results, key=lambda x: x.get('snr', -np.inf) if np.isfinite(x.get('snr', -np.inf)) else -np.inf, reverse=True) if self.sort_by_snr.get() else self.analysis_results
                for r in display_res:
                    status = r.get('status','?'); vals = []
                    for col_id in cols:
                        if col_id == 'file': vals.append(r.get('rel_path', os.path.basename(r.get('file','?'))));
                        elif col_id == 'status': vals.append(status)
                        elif col_id == 'snr': vals.append(f"{r.get('snr',0.0):.2f}" if np.isfinite(r.get('snr', np.nan)) else "N/A")
                        elif col_id == 'bg': vals.append(f"{r.get('sky_bg',0.0):.2f}" if np.isfinite(r.get('sky_bg', np.nan)) else "N/A")
                        elif col_id == 'noise': vals.append(f"{r.get('sky_noise',0.0):.2f}" if np.isfinite(r.get('sky_noise', np.nan)) else "N/A")
                        elif col_id == 'pixsig': vals.append(f"{r.get('signal_pixels',0)}")
                        elif col_id == 'trails': vals.append(self._("logic_trail_yes") if r.get('has_trails',False) else self._("logic_trail_no"))
                        elif col_id == 'nbseg': vals.append(f"{r.get('num_trails',0)}" if 'num_trails' in r else "N/A")
                        elif col_id == 'action': vals.append(r.get('action','?'))
                        elif col_id == 'reason': vals.append(r.get('rejected_reason','') or '')
                        elif col_id == 'comment': vals.append(r.get('error_message', '') + r.get('action_comment', ''))
                        else: vals.append(r.get(col_id, '?'))
                    tag = ('error',) if status == 'error' else ('rejected',) if r.get('rejected_reason') else ()
                    tree.insert('', tk.END, values=tuple(vals), tags=tag)
                tree.tag_configure('error', background='mistyrose'); tree.tag_configure('rejected', background='lightyellow'); vsb = ttk.Scrollbar(data_tab, orient=tk.VERTICAL, command=tree.yview); hsb = ttk.Scrollbar(data_tab, orient=tk.HORIZONTAL, command=tree.xview); tree.configure(yscroll=vsb.set, xscroll=hsb.set); vsb.pack(side=tk.RIGHT, fill=tk.Y); hsb.pack(side=tk.BOTTOM, fill=tk.X); tree.pack(fill=tk.BOTH, expand=True)
            except Exception as e: print(f"Erreur Tableau Données: {e}"); ttk.Label(data_tab, text=f"{self._('msg_error')}:\n{e}\n{traceback.format_exc()}").pack()

            # --- Onglet Recommandations Stacking ---
            stack_tab = ttk.Frame(notebook); notebook.add(stack_tab, text=self._("visu_tab_recom")); fig4=None # Placeholder fig
            try:
                recom_frame = ttk.LabelFrame(stack_tab, text=self._("visu_recom_frame_title"), padding=10); recom_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                valid_kept_results = [r for r in self.analysis_results if r.get('status')=='ok' and r.get('action')=='kept' and r.get('rejected_reason') is None and 'snr' in r and np.isfinite(r['snr'])]; valid_kept_snrs = [r['snr'] for r in valid_kept_results]
                if len(valid_kept_snrs) >= 5:
                    p25_threshold = np.percentile(valid_kept_snrs, 25); good_img = [r for r in valid_kept_results if r['snr'] >= p25_threshold]
                    if good_img:
                        good_img_sorted = sorted(good_img, key=lambda x: x['snr'], reverse=True); ttk.Label(recom_frame, text=self._("visu_recom_text", count=len(good_img_sorted), p75=p25_threshold)).pack(anchor=tk.W, pady=(0,5)); rec_cols = ("file", "snr"); rec_tree = ttk.Treeview(recom_frame, columns=rec_cols, show='headings', height=10); rec_tree.heading("file", text=self._("visu_recom_col_file")); rec_tree.column("file", width=450, anchor='w'); rec_tree.heading("snr", text=self._("visu_recom_col_snr")); rec_tree.column("snr", width=100, anchor='center')
                        for img in good_img_sorted: rec_tree.insert('', tk.END, values=(img.get('rel_path', os.path.basename(img.get('file', '?'))), f"{img.get('snr', 0.0):.2f}"));
                        rec_scr = ttk.Scrollbar(recom_frame, orient=tk.VERTICAL, command=rec_tree.yview); rec_tree.configure(yscroll=rec_scr.set); rec_scr.pack(side=tk.RIGHT, fill=tk.Y); rec_tree.pack(fill=tk.BOTH, expand=True, pady=(0,10)); export_cmd = lambda gi=good_img_sorted, p=p25_threshold: self.export_recommended_list(gi, p); export_button = ttk.Button(recom_frame, text=self._("export_button"), command=export_cmd); export_button.pack(pady=5)
                    else: ttk.Label(recom_frame, text=self._("visu_recom_no_selection")).pack(padx=10, pady=10)
                elif len(valid_kept_snrs) > 0:
                    ttk.Label(recom_frame, text=self._("visu_recom_not_enough")).pack(padx=10, pady=10); export_all_kept_cmd = lambda gi=valid_kept_results: self.export_recommended_list(gi, -1); export_all_button = ttk.Button(recom_frame, text=self._("Exporter Toutes Conservées", default="Export All Kept"), command=export_all_kept_cmd); export_all_button.pack(pady=5)
                else: ttk.Label(recom_frame, text=self._("visu_recom_no_data")).pack(padx=10, pady=10)
            except Exception as e:
                 print(f"Erreur Recommandations: {e}"); traceback.print_exc(); ttk.Label(stack_tab, text=f"{self._('msg_error')}:\n{e}\n{traceback.format_exc()}").pack()
                 # Pas de fig4 à fermer ici pour l'instant

            # --- Fonction de Nettoyage et Bouton Fermer ---
            # <--- DÉBUT DU CODE MANQUANT --->
            def cleanup_vis_window():
                """Nettoie les ressources Matplotlib et détruit la fenêtre."""
                nonlocal canvas_list, figures_list, vis_window # Utiliser les variables locales
                print("Nettoyage de la fenêtre de visualisation...")
                # Détruire les canvas Tkinter
                for canvas in canvas_list:
                    if canvas and canvas.get_tk_widget().winfo_exists():
                        try:
                            # Tenter d'appeler close_event() - peut ne pas exister sur toutes les versions/backends
                            if hasattr(canvas, 'close_event') and callable(canvas.close_event):
                                canvas.close_event()
                                print(f"  Canvas close_event() appelé pour {canvas}")
                            # Détacher et détruire le widget Tkinter
                            canvas.get_tk_widget().pack_forget()
                            canvas.get_tk_widget().destroy()
                            print(f"  Widget Canvas détruit pour {canvas}")
                        except tk.TclError as e_tk:
                            print(f"  TclError détruisant le widget canvas: {e_tk}")
                        except Exception as e_other:
                            print(f"  Erreur détruisant le widget canvas: {e_other}")
                canvas_list = [] # Vider la liste

                # Fermer les figures Matplotlib
                print(f"  Fermeture de {len(figures_list)} figures Matplotlib...")
                for fig in figures_list:
                    try:
                        plt.close(fig) # Fermer la figure
                    except Exception as e_plt:
                        print(f"  Erreur lors de la fermeture de la figure {fig}: {e_plt}")
                figures_list = [] # Vider la liste

                # Forcer Garbage Collection
                print("  Forçage du Garbage Collector...")
                collected_count = gc.collect()
                print(f"    Garbage Collector a collecté {collected_count} objets.")

                # Détruire la fenêtre Toplevel
                if vis_window and vis_window.winfo_exists():
                    try:
                        print("  Destruction de la fenêtre Toplevel de visualisation...")
                        vis_window.grab_release() # Libérer grab modal
                        vis_window.destroy()
                        print("  Fenêtre de visualisation détruite.")
                    except tk.TclError as e_tk_win:
                        print(f"  TclError détruisant vis_window: {e_tk_win}")
                    except Exception as e_other_win:
                        print(f"  Erreur détruisant vis_window: {e_other_win}")

            # Bouton Fermer et liaison fermeture fenêtre
            close_button = ttk.Button(vis_window, text=self._("Fermer", default="Close"), command=cleanup_vis_window)
            close_button.pack(pady=10)
            vis_window.protocol("WM_DELETE_WINDOW", cleanup_vis_window) # Lier bouton X

            # Attendre que la fenêtre de visualisation soit fermée
            self.root.wait_window(vis_window)
            print("Fenêtre de visualisation fermée (wait_window finished).")
            # <--- FIN DU CODE MANQUANT --->

        # --- Fin du bloc try principal ---
        except Exception as e_vis:
            # Gérer erreur globale création fenêtre visualisation
            print(f"Erreur Globale Visualisation: {e_vis}")
            traceback.print_exc()
            messagebox.showerror(self._("msg_error"), self._("msg_unexpected_error", e=e_vis), parent=self.root)
            # Essayer de nettoyer même en cas d'erreur
            if vis_window and vis_window.winfo_exists():
                try:
                    vis_window.destroy()
                except Exception: pass

    def sort_treeview(self, tree, col, reverse):
        """Trie les données d'un Treeview lorsqu'un en-tête de colonne est cliqué."""
        try:
            # Obtenir les données de la colonne et l'identifiant de chaque ligne
            data = [(tree.set(item, col), item) for item in tree.get_children('')]

            # Fonction pour convertir en float (gère erreurs et non-numériques)
            def safe_float(value):
                try: return float(value)
                except (ValueError, TypeError): return -float('inf') # Pour trier non-numériques en bas

            # Essayer de trier numériquement d'abord
            try:
                data.sort(key=lambda t: safe_float(t[0]), reverse=reverse)
            except Exception:
                # Si tri numérique échoue, trier alphabétiquement
                data.sort(reverse=reverse)

            # Réinsérer les éléments dans l'ordre trié
            for index, (val, item) in enumerate(data):
                tree.move(item, '', index)

            # Inverser la direction du tri pour le prochain clic sur cette colonne
            tree.heading(col, command=lambda _col=col: self.sort_treeview(tree, _col, not reverse))
        except Exception as e:
            print(f"Erreur tri Treeview: {e}")



    def manage_markers(self):
        """Ouvre une fenêtre pour visualiser et supprimer les fichiers marqueurs."""
        input_dir = self.input_dir.get()
        # Vérifier si dossier entrée valide
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror(self._("msg_error"), self._("msg_input_dir_invalid"), parent=self.root)
            return

        marker_filename = ".astro_analyzer_run_complete" # Nom du fichier marqueur
        marked_dirs_rel = [] # Liste chemins relatifs marqués
        marked_dirs_abs = [] # Liste chemins absolus marqués
        abs_input_dir = os.path.abspath(input_dir)

        # Déterminer les dossiers de rejet à exclure du scan
        reject_dirs_to_exclude_abs = []
        if self.reject_action.get() == 'move':
            snr_dir = self.snr_reject_dir.get()
            trail_dir = self.trail_reject_dir.get()
            if snr_dir: reject_dirs_to_exclude_abs.append(os.path.abspath(snr_dir))
            if trail_dir: reject_dirs_to_exclude_abs.append(os.path.abspath(trail_dir))

        # Scanner les dossiers pour trouver les marqueurs
        try:
            print(f"Scan pour marqueur '{marker_filename}' dans {abs_input_dir}...")
            for dirpath, dirnames, filenames in os.walk(abs_input_dir, topdown=True):
                current_dir_abs = os.path.abspath(dirpath)

                # Exclure les dossiers de rejet du parcours récursif
                dirs_to_remove = [d for d in dirnames if os.path.abspath(os.path.join(current_dir_abs, d)) in reject_dirs_to_exclude_abs]
                if dirs_to_remove:
                    for dname in dirs_to_remove: dirnames.remove(dname)

                # Vérifier présence marqueur
                marker_path = os.path.join(current_dir_abs, marker_filename)
                if os.path.exists(marker_path):
                    # Obtenir chemin relatif par rapport au dossier d'entrée
                    rel_path = os.path.relpath(current_dir_abs, abs_input_dir)
                    # Utiliser '.' pour le dossier racine lui-même
                    marked_dirs_rel.append('.' if rel_path == '.' else rel_path)
                    marked_dirs_abs.append(current_dir_abs) # Stocker chemin absolu



        except OSError as e:
            messagebox.showerror(self._("msg_error"), f"Erreur parcours dossiers:\n{e}", parent=self.root)
            return

        def delete_selected_markers():
            """Supprime les marqueurs des dossiers sélectionnés dans la listbox."""
            selected_indices = listbox.curselection()
            if not selected_indices:
                messagebox.showwarning(self._("msg_warning"), self._("marker_select_none", default="Veuillez sélectionner un ou plusieurs dossiers."), parent=marker_window)
                return

            # Confirmer suppression
            confirm_msg = self._("marker_confirm_delete_selected", default="Supprimer les marqueurs pour les {count} dossiers sélectionnés ?\nCela forcera leur ré-analyse au prochain lancement.").format(count=len(selected_indices))
            if messagebox.askyesno(self._("msg_warning"), confirm_msg, parent=marker_window):
                deleted_count = 0; errors = []
                paths_to_delete_rel = [listbox.get(i) for i in selected_indices]

                # Supprimer les fichiers marqueurs
                for rel_path in paths_to_delete_rel:
                    abs_path_to_clear = rel_to_abs_map.get(rel_path)
                    if not abs_path_to_clear: errors.append(f"{rel_path}: Chemin absolu non trouvé"); continue
                    marker_to_delete = os.path.join(abs_path_to_clear, marker_filename)
                    try:
                        if os.path.exists(marker_to_delete): os.remove(marker_to_delete); deleted_count += 1
                        else: deleted_count += 1 # Compter comme succès si déjà absent
                    except Exception as e: errors.append(f"{rel_path}: {e}")

                # Rafraîchir la liste après suppression (re-scanner est plus sûr)
                listbox.config(state=tk.NORMAL); listbox.delete(0, tk.END)
                marked_dirs_rel.clear(); marked_dirs_abs.clear(); rel_to_abs_map.clear()
                try: # Re-scan
                    for dp, dn, fn in os.walk(abs_input_dir, topdown=True):
                        ca = os.path.abspath(dp)
                        dtr = [d for d in dn if os.path.abspath(os.path.join(ca, d)) in reject_dirs_to_exclude_abs]
                        for dname in dtr: dn.remove(dname)
                        mp = os.path.join(ca, marker_filename)
                        if os.path.exists(mp):
                            rp = os.path.relpath(ca, abs_input_dir); rp = '.' if rp == '.' else rp
                            marked_dirs_rel.append(rp); marked_dirs_abs.append(ca); rel_to_abs_map[rp] = ca
                except Exception as e_scan: listbox.insert(tk.END, "Erreur re-scan"); listbox.config(state=tk.DISABLED)

                # Re-remplir listbox
                sorted_rel_paths_new = sorted(marked_dirs_rel)
                if sorted_rel_paths_new:
                    for rel_path in sorted_rel_paths_new: listbox.insert(tk.END, rel_path)
                else: listbox.insert(tk.END, self._("marker_none_found", default="Aucun dossier marqué trouvé.")); listbox.config(state=tk.DISABLED)

                # Mettre à jour état boutons
                any_markers_left = bool(sorted_rel_paths_new)
                delete_sel_button.config(state=tk.NORMAL if any_markers_left else tk.DISABLED)
                delete_all_button.config(state=tk.NORMAL if any_markers_left else tk.DISABLED)

                # Message final
                if errors: messagebox.showerror(self._("msg_error"), self._("marker_delete_errors", default="Erreurs lors de la suppression de certains marqueurs:\n") + "\n".join(errors), parent=marker_window)
                elif deleted_count > 0: messagebox.showinfo(self._("msg_info"), self._("marker_delete_selected_success", default="{count} marqueur(s) supprimé(s).").format(count=deleted_count), parent=marker_window)
####################################################################################################################


        def delete_all_markers():
            """Supprime tous les marqueurs trouvés."""
            abs_paths_to_clear = list(rel_to_abs_map.values())
            if not abs_paths_to_clear:
                messagebox.showinfo(self._("msg_info"), self._("marker_none_found", default="Aucun dossier marqué trouvé."), parent=marker_window)
                return

            # Confirmer suppression totale
            confirm_msg = self._("marker_confirm_delete_all", default="Supprimer TOUS les marqueurs ({count}) dans le dossier '{folder}' et ses sous-dossiers analysables ?\nCela forcera une ré-analyse complète.").format(count=len(abs_paths_to_clear), folder=os.path.basename(abs_input_dir))
            if messagebox.askyesno(self._("msg_warning"), confirm_msg, parent=marker_window):
                deleted_count = 0; errors = []
                # Supprimer tous les marqueurs
                for abs_path in abs_paths_to_clear:
                    marker_to_delete = os.path.join(abs_path, marker_filename)
                    try:
                        if os.path.exists(marker_to_delete): os.remove(marker_to_delete); deleted_count += 1
                    except Exception as e: errors.append(f"{os.path.relpath(abs_path, abs_input_dir)}: {e}")

                # Mettre à jour listbox (vider)
                listbox.config(state=tk.NORMAL); listbox.delete(0, tk.END)
                listbox.insert(tk.END, self._("marker_none_found", default="Aucun dossier marqué trouvé."))
                listbox.config(state=tk.DISABLED)
                marked_dirs_rel[:] = []; marked_dirs_abs[:] = []; rel_to_abs_map.clear()

                # Mettre à jour état boutons
                delete_sel_button.config(state=tk.DISABLED)
                delete_all_button.config(state=tk.DISABLED)

                # Message final
                if errors: messagebox.showerror(self._("msg_error"), self._("marker_delete_errors", default="Erreurs lors de la suppression de certains marqueurs:\n") + "\n".join(errors), parent=marker_window)
                elif deleted_count > 0: messagebox.showinfo(self._("msg_info"), self._("marker_delete_all_success", default="Tous les {count} marqueur(s) trouvés ont été supprimés.").format(count=deleted_count), parent=marker_window)



        # --- Créer la fenêtre de gestion des marqueurs ---
        marker_window = tk.Toplevel(self.root)
        marker_window.title(self._("marker_window_title", default="Gérer les Marqueurs d'Analyse"))
        marker_window.geometry("600x400")
        marker_window.transient(self.root)
        marker_window.grab_set() # Rendre modale

        # Cadre principal fenêtre
        mw_frame = ttk.Frame(marker_window, padding=10)
        mw_frame.pack(fill=tk.BOTH, expand=True)

        # Label info
        info_label = ttk.Label(mw_frame, text=self._("marker_info_label", default="Dossiers marqués comme analysés (contiennent le fichier marqueur):"), wraplength=550)
        info_label.pack(pady=(0, 10))

        # Cadre pour Listbox et Scrollbar
        list_frame = ttk.Frame(mw_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, yscrollcommand=scrollbar.set, width=80)
        scrollbar.config(command=listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Remplir Listbox et mapper relatif -> absolu
        rel_to_abs_map = {rel: abs_p for rel, abs_p in zip(marked_dirs_rel, marked_dirs_abs)}
        sorted_rel_paths = sorted(marked_dirs_rel)
        if sorted_rel_paths:
            for rel_path in sorted_rel_paths: listbox.insert(tk.END, rel_path)
        else:
            listbox.insert(tk.END, self._("marker_none_found", default="Aucun dossier marqué trouvé."))
            listbox.config(state=tk.DISABLED) # Désactiver si vide

        # Cadre pour les boutons
        button_mw_frame = ttk.Frame(mw_frame)
        button_mw_frame.pack(fill=tk.X, pady=(10, 0))

        # Créer les boutons de la fenêtre marqueur
        delete_sel_button = ttk.Button(button_mw_frame, text=self._("marker_delete_selected_button", default="Supprimer Sélection"), command=delete_selected_markers)
        delete_sel_button.pack(side=tk.LEFT, padx=10)
        delete_all_button = ttk.Button(button_mw_frame, text=self._("marker_delete_all_button", default="Supprimer Tout"), command=delete_all_markers)
        delete_all_button.pack(side=tk.LEFT, padx=10)
        close_mw_button = ttk.Button(button_mw_frame, text=self._("Fermer", default="Close"), command=marker_window.destroy)
        close_mw_button.pack(side=tk.RIGHT, padx=10)

        # Désactiver boutons si liste vide au départ
        if not sorted_rel_paths:
            delete_sel_button.config(state=tk.DISABLED)
            delete_all_button.config(state=tk.DISABLED)

        # Attendre fermeture fenêtre
        marker_window.wait_window()









    def return_or_quit(self):
        """Gère la fermeture de la fenêtre ou le retour à l'application principale."""
        # --- Ajout Debug ---
        print("DEBUG AG: Début return_or_quit.") # Log début
        # --------------------

        # Confirmer si analyse en cours (inchangé)
        if self.analysis_running:
            title = self._("msg_warning")
            message = self._("Analyse en cours, quitter quand même?")
            if not messagebox.askyesno(title, message, parent=self.root):
                # --- Ajout Debug ---
                print("DEBUG AG: return_or_quit annulé par l'utilisateur (analyse en cours).")
                # --------------------
                return # Ne pas quitter si l'utilisateur annule

        # --- Ajout Debug ---
        print("DEBUG AG: Fermeture du GUI de l'analyseur...")
        # --------------------
        print("  Nettoyage explicite des ressources Tkinter...")
        # Cacher et annuler les infobulles actives (inchangé)
        for tt_key in list(self.tooltips.keys()):
            tooltip = self.tooltips.pop(tt_key, None)
            if tooltip:
                tooltip.hidetip()
                tooltip.unschedule()

        # Essayer de supprimer les références aux variables Tkinter (inchangé)
        widget_keys_to_delete = list(self.__dict__.keys())
        deleted_vars = 0
        for attr_name in widget_keys_to_delete:
            try:
                attr_value = getattr(self, attr_name, None)
                if isinstance(attr_value, tk.Variable):
                    delattr(self, attr_name)
                    deleted_vars += 1
            except Exception: pass
        print(f"    {deleted_vars} variables Tkinter supprimées de l'instance.")

        # Vider les dictionnaires de références (inchangé)
        if hasattr(self, 'widgets_refs'): self.widgets_refs.clear()
        if hasattr(self, 'trail_param_labels'): self.trail_param_labels.clear()
        if hasattr(self, 'trail_param_entries'): self.trail_param_entries.clear()
        if hasattr(self, 'analysis_results'): self.analysis_results = []
        print("    Dictionnaires de références vidés.")

        # Forcer mise à jour UI avant destruction (inchangé)
        try:
            if self.root and self.root.winfo_exists():
                self.root.update_idletasks()
                print("    update_idletasks exécuté.")
        except tk.TclError:
            print("    Impossible d'exécuter update_idletasks (fenêtre détruite?).")

        # Forcer Garbage Collection (inchangé)
        print("  Forçage du Garbage Collector...")
        collected_count = gc.collect()
        print(f"    Garbage Collector a collecté {collected_count} objets.")

        # Détruire la fenêtre Toplevel
        try:
            if self.root and self.root.winfo_exists():
                # --- Ajout Debug ---
                print("DEBUG AG: Tentative de destruction de la fenêtre Toplevel (self.root)...")
                # --------------------
                self.root.destroy() # <--- Destruction fenêtre
                # --- Ajout Debug ---
                print("DEBUG AG: Fenêtre Toplevel a priori détruite.")
                # --------------------
        except Exception as e: print(f"  Erreur pendant destroy: {e}")

        # Appeler le callback si fourni
        if callable(self.main_app_callback):
            # --- Ajout Debug ---
            print("DEBUG AG: Appel du callback du script principal (_on_analyzer_closed)...")
            # --------------------
            try:
                self.main_app_callback() # <--- Appel callback
                # --- Ajout Debug ---
                print("DEBUG AG: Callback principal exécuté.")
                # --------------------
            except Exception as cb_e: print(f"  Erreur dans le callback principal: {cb_e}")
        else:
            # --- Ajout Debug ---
            print("DEBUG AG: Aucun callback principal à appeler.")
            # --------------------

        # --- Ajout Debug ---
        print("DEBUG AG: Fin return_or_quit.") # Log fin
        # --------------------




    def _(self, key, **kwargs):
        """Raccourci pour obtenir une traduction."""
        lang = self.current_lang.get()
        # Utiliser l'anglais comme langue de secours si la clé manque en français
        default_lang_dict = translations.get('en', {})
        lang_dict = translations.get(lang, default_lang_dict)
        text = lang_dict.get(key, default_lang_dict.get(key, f"_{key}_")) # Retourne _clé_ si introuvable
        try:
            # Formater le texte si des arguments supplémentaires sont fournis
            return text.format(**kwargs)
        except KeyError as e:
            # Avertissement si une clé de formatage manque
            print(f"WARN: Erreur formatage clé '{key}' langue '{lang}'. Clé manquante: {e}")
            return text
        except Exception as e:
            # Avertissement pour autres erreurs de formatage
            print(f"WARN: Erreur formatage clé '{key}' langue '{lang}': {e}")
            return text

    def _start_timer(self, base_message):
        """Démarre ou redémarre le chronomètre dans la barre de statut."""
        if not self.timer_running:
            self.timer_running = True
            self.timer_start_time = time.time() # Utiliser time.time() pour le temps réel écoulé
            self.base_status_message = base_message
        # Annuler toute mise à jour précédente du timer
        if self.timer_job_id:
            try:
                if self.root.winfo_exists(): self.root.after_cancel(self.timer_job_id)
            except (ValueError, tk.TclError): pass
            self.timer_job_id = None
        # Lancer la première mise à jour de l'affichage
        self._update_timer_display()

    def _stop_timer(self):
        """Arrête le chronomètre."""
        if self.timer_running:
            self.timer_running = False
        # Annuler la prochaine mise à jour planifiée
        if self.timer_job_id:
            try:
                if self.root.winfo_exists(): self.root.after_cancel(self.timer_job_id)
            except (ValueError, tk.TclError): pass
        self.timer_job_id = None
        self.timer_start_time = None # Réinitialiser le temps de départ

    def _update_timer_display(self):
        """Met à jour l'affichage du temps écoulé (appelé périodiquement)."""
        if self.timer_running and self.timer_start_time is not None:
            elapsed_seconds = time.time() - self.timer_start_time
            # Formater le temps écoulé en HH:MM:SS ou MM:SS
            if elapsed_seconds < 3600:
                time_str = time.strftime('%M:%S', time.gmtime(elapsed_seconds))
            else:
                time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_seconds))
            timed_message = f"{self.base_status_message} ({time_str})"

            try:
                # Mettre à jour le label de statut s'il existe toujours
                if self.status_label and self.status_label.winfo_exists():
                    self.status_text.set(timed_message)
                else:
                    # Arrêter le timer si le label n'existe plus
                    self._stop_timer()
                    return
            except tk.TclError:
                # Arrêter le timer si une erreur Tkinter se produit (fenêtre fermée?)
                self._stop_timer()
                return

            # Planifier la prochaine mise à jour dans 1 seconde
            try:
                if self.root.winfo_exists():
                    self.timer_job_id = self.root.after(1000, self._update_timer_display)
                else:
                    # Arrêter si la fenêtre racine n'existe plus
                    self._stop_timer()
            except tk.TclError:
                # Arrêter si une erreur Tkinter se produit
                self._stop_timer()

    # --- Création des Widgets ---

    def create_widgets(self):
        """Crée tous les widgets de l'interface graphique."""
        # Cadre principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Cadre Configuration Générale ---
        config_frame = ttk.LabelFrame(main_frame, text="", padding="10") # Texte sera défini par change_language
        config_frame.pack(fill=tk.X, pady=5)
        config_frame.columnconfigure(1, weight=1) # Colonne des entrées prend l'espace
        self.widgets_refs['config_frame'] = config_frame # Référence pour traduction

        # Ligne 0: Dossier d'entrée
        input_dir_label = ttk.Label(config_frame, text="")
        input_dir_label.grid(row=0, column=0, sticky=tk.W, pady=2)
        self.widgets_refs['input_dir_label'] = input_dir_label
        ttk.Entry(config_frame, textvariable=self.input_dir, width=50).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W+tk.E)
        browse_input_button = ttk.Button(config_frame, text="", command=self.browse_input_dir)
        browse_input_button.grid(row=0, column=2, padx=5, pady=2)
        self.widgets_refs['browse_input_button'] = browse_input_button

        # Ligne 1: Fichier log
        output_log_label = ttk.Label(config_frame, text="")
        output_log_label.grid(row=1, column=0, sticky=tk.W, pady=2)
        self.widgets_refs['output_log_label'] = output_log_label
        ttk.Entry(config_frame, textvariable=self.output_log, width=50).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W+tk.E)
        browse_log_button = ttk.Button(config_frame, text="", command=self.browse_output_log)
        browse_log_button.grid(row=1, column=2, padx=5, pady=2)
        self.widgets_refs['browse_log_button'] = browse_log_button

        # Ligne 2: Inclure sous-dossiers
        subfolder_check = ttk.Checkbutton(config_frame, text="", variable=self.include_subfolders)
        subfolder_check.grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(5,2))
        self.widgets_refs['include_subfolders_label'] = subfolder_check

        # Ligne 3: Sélection Langue (Aligné à droite)
        lang_frame = ttk.Frame(config_frame)
        lang_frame.grid(row=3, column=0, columnspan=3, sticky=tk.E, pady=5, padx=5)
        lang_label = ttk.Label(lang_frame, text="")
        lang_label.pack(side=tk.LEFT, padx=(0, 5))
        self.widgets_refs['lang_label'] = lang_label
        lang_options = sorted([lang for lang in translations.keys()]) # Options de langue disponibles
        self.lang_combobox = ttk.Combobox(lang_frame, textvariable=self.current_lang, values=lang_options, state="readonly", width=5)
        self.lang_combobox.pack(side=tk.LEFT)

        # --- Cadre Analyse SNR ---
        snr_frame = ttk.LabelFrame(main_frame, text="", padding="10")
        snr_frame.pack(fill=tk.X, pady=5, padx=5)
        snr_frame.columnconfigure(1, weight=1)
        self.widgets_refs['snr_frame'] = snr_frame

        # Case à cocher pour activer/désactiver l'analyse SNR
        analyze_snr_check = ttk.Checkbutton(snr_frame, text="", variable=self.analyze_snr, command=self.toggle_sections_state)
        analyze_snr_check.grid(row=0, column=0, columnspan=5, sticky=tk.W, padx=5, pady=2)
        self.widgets_refs['analyze_snr_check_label'] = analyze_snr_check

        # Frame pour les options de sélection SNR
        self.snr_select_frame = ttk.Frame(snr_frame)
        self.snr_select_frame.grid(row=1, column=0, columnspan=5, sticky=tk.W+tk.E, padx=5, pady=5)
        snr_select_mode_label = ttk.Label(self.snr_select_frame, text="")
        snr_select_mode_label.grid(row=0, column=0, sticky=tk.W, padx=5)
        self.widgets_refs['snr_select_mode_label'] = snr_select_mode_label

        # Boutons radio pour le mode de sélection SNR
        modes_config = [('percent', 'snr_mode_percent'), ('threshold', 'snr_mode_threshold'), ('none', 'snr_mode_none')]
        self.widgets_refs['snr_mode_radios'] = {}
        current_column = 1
        rb_percent = ttk.Radiobutton(self.snr_select_frame, text="", variable=self.snr_selection_mode, value='percent', command=self.toggle_sections_state)
        rb_percent.grid(row=0, column=current_column, sticky=tk.W, padx=5); self.widgets_refs['snr_mode_radios']['percent'] = rb_percent; current_column += 1
        rb_thresh = ttk.Radiobutton(self.snr_select_frame, text="", variable=self.snr_selection_mode, value='threshold', command=self.toggle_sections_state)
        rb_thresh.grid(row=0, column=current_column, sticky=tk.W, padx=5); self.widgets_refs['snr_mode_radios']['threshold'] = rb_thresh; current_column += 1

        # Champ d'entrée pour la valeur (pourcentage ou seuil)
        self.snr_value_entry = ttk.Entry(self.snr_select_frame, textvariable=self.snr_selection_value, width=8)
        self.snr_value_entry.grid(row=0, column=current_column, sticky=tk.W, padx=(0, 15))
        self.tooltips['snr_value_entry'] = ToolTip(self.snr_value_entry, lambda: self._('tooltip_snr_value')) # Infobulle
        current_column += 1

        rb_none = ttk.Radiobutton(self.snr_select_frame, text="", variable=self.snr_selection_mode, value='none', command=self.toggle_sections_state)
        rb_none.grid(row=0, column=current_column, sticky=tk.W, padx=5); self.widgets_refs['snr_mode_radios']['none'] = rb_none

        # Frame pour le dossier de rejet SNR
        self.snr_reject_dir_frame = ttk.Frame(snr_frame)
        self.snr_reject_dir_frame.grid(row=2, column=0, columnspan=current_column + 1, sticky=tk.W+tk.E, padx=5, pady=2)
        snr_reject_dir_label = ttk.Label(self.snr_reject_dir_frame, text="")
        snr_reject_dir_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.widgets_refs['snr_reject_dir_label'] = snr_reject_dir_label
        self.snr_reject_dir_entry = ttk.Entry(self.snr_reject_dir_frame, textvariable=self.snr_reject_dir, width=40)
        self.snr_reject_dir_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W+tk.E)
        snr_reject_dir_button = ttk.Button(self.snr_reject_dir_frame, text="", command=self.browse_snr_reject_dir)
        snr_reject_dir_button.grid(row=0, column=2, padx=5, pady=2)
        self.widgets_refs['snr_reject_dir_button'] = snr_reject_dir_button
        self.snr_reject_dir_frame.columnconfigure(1, weight=1)
        # --- NOUVEAU BOUTON "Appliquer Rejet SNR" ---
        # Placé dans une nouvelle Frame pour un meilleur alignement si besoin, ou directement.
        apply_snr_frame = ttk.Frame(snr_frame) # Cadre pour ce bouton
        apply_snr_frame.grid(row=3, column=0, columnspan=5, sticky=tk.E, padx=5, pady=(5,2)) # sticky=tk.E pour aligner à droite

        self.apply_snr_button = ttk.Button(
            apply_snr_frame, 
            text="Appliquer Rejet SNR", # Le texte sera mis à jour par change_language
            command=self.apply_pending_snr_actions_gui, # Nouvelle méthode à créer
            state=tk.DISABLED # Désactivé initialement
        )
        self.apply_snr_button.pack(side=tk.RIGHT) # Ou grid si vous préférez
        self.widgets_refs['apply_snr_rejection_button'] = self.apply_snr_button # Pour la traduction du texte

        # --- Cadre Détection Traînées ---
        trail_frame = ttk.LabelFrame(main_frame, text="", padding="10")
        trail_frame.pack(fill=tk.X, pady=5, padx=5)
        trail_frame.columnconfigure(1, weight=1)
        self.widgets_refs['trail_frame'] = trail_frame

        # Case à cocher pour activer/désactiver la détection
        self.detect_trails_check = ttk.Checkbutton(trail_frame, text="", variable=self.detect_trails, command=self.toggle_sections_state)
        self.detect_trails_check.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.widgets_refs['detect_trails_check_label'] = self.detect_trails_check

        # Affichage statut acstools
        acstools_frame = ttk.Frame(trail_frame)
        acstools_frame.grid(row=0, column=1, columnspan=2, sticky=tk.W)
        self.acstools_status_label = ttk.Label(acstools_frame, text="", foreground="grey") # Texte mis à jour par change_language
        self.acstools_status_label.pack(side=tk.LEFT, padx=5)

        # Frame pour les paramètres de détection (sigma, seuils, etc.)
        self.params_sat_frame = ttk.Frame(trail_frame)
        self.params_sat_frame.grid(row=1, column=0, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=5)
        keys = ['sigma', 'low_thresh', 'h_thresh', 'line_len', 'small_edge', 'line_gap']
        self.trail_param_labels = {}
        self.trail_param_entries = {}
        for i, key in enumerate(keys):
            row, col_base = divmod(i, 3) # Organiser en 2 lignes de 3 paramètres
            col_label = col_base * 2
            col_entry = col_label + 1
            lbl = ttk.Label(self.params_sat_frame, text="") # Texte sera mis à jour
            lbl.grid(row=row, column=col_label, sticky=tk.W, padx=(5,0))
            self.trail_param_labels[key] = lbl
            entry = ttk.Entry(self.params_sat_frame, textvariable=self.trail_params[key], width=8)
            entry.grid(row=row, column=col_entry, sticky=tk.W, padx=(0,10))
            self.trail_param_entries[key] = entry
            tooltip_key = f'tooltip_{key}' # Clé de traduction pour l'infobulle
            self.tooltips[f'{key}_entry_tt'] = ToolTip(entry, lambda k=tooltip_key: self._(k)) # Créer infobulle

        # Frame pour le dossier de rejet des traînées
        self.trail_reject_dir_frame = ttk.Frame(trail_frame)
        self.trail_reject_dir_frame.grid(row=2, column=0, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=2)
        trail_reject_dir_label = ttk.Label(self.trail_reject_dir_frame, text="")
        trail_reject_dir_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.widgets_refs['trail_reject_dir_label'] = trail_reject_dir_label
        self.trail_reject_dir_entry = ttk.Entry(self.trail_reject_dir_frame, textvariable=self.trail_reject_dir, width=40)
        self.trail_reject_dir_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W+tk.E)
        trail_reject_dir_button = ttk.Button(self.trail_reject_dir_frame, text="", command=self.browse_trail_reject_dir)
        trail_reject_dir_button.grid(row=0, column=2, padx=5, pady=2)
        self.widgets_refs['trail_reject_dir_button'] = trail_reject_dir_button
        self.trail_reject_dir_frame.columnconfigure(1, weight=1)

        # --- Cadre Action sur Images Rejetées ---
        action_frame = ttk.LabelFrame(main_frame, text="", padding="10")
        action_frame.pack(fill=tk.X, pady=5, padx=5)
        self.widgets_refs['action_frame'] = action_frame
        action_label = ttk.Label(action_frame, text="")
        action_label.grid(row=0, column=0, sticky=tk.W, padx=5)
        self.widgets_refs['action_label'] = action_label
        # Boutons radio pour l'action
        action_modes_config = [('move', 'action_mode_move'), ('delete', 'action_mode_delete'), ('none', 'action_mode_none')]
        self.widgets_refs['action_radios'] = {}
        for i, (mode_val, mode_key) in enumerate(action_modes_config):
            rb = ttk.Radiobutton(action_frame, text="", variable=self.reject_action, value=mode_val, command=self.toggle_sections_state)
            rb.grid(row=0, column=i + 1, sticky=tk.W, padx=10)
            self.widgets_refs['action_radios'][mode_val] = rb

        # --- Cadre Options d'Affichage ---
        display_options_frame = ttk.LabelFrame(main_frame, text="", padding="5")
        display_options_frame.pack(fill=tk.X, pady=5, padx=5)
        self.widgets_refs['display_options_frame'] = display_options_frame
        sort_snr_check = ttk.Checkbutton(display_options_frame, text="", variable=self.sort_by_snr)
        sort_snr_check.grid(row=0, column=0, sticky=tk.W, padx=5)
        self.widgets_refs['sort_snr_check_label'] = sort_snr_check

        # --- Cadre Boutons Principaux ---
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        self.analyze_button = ttk.Button(button_frame, text="", command=self.start_analysis, width=18)
        self.analyze_button.pack(side=tk.LEFT, padx=(5, 2)) # Pack avec Analyze normal
        self.widgets_refs['analyse_button'] = self.analyze_button # Référencer le bouton Analyser seul

        # <--- NOUVEAU ---> Ajout du bouton Analyser et Empiler
        # Il appellera self.start_analysis_and_stack (à créer à l'étape 2)
        self.analyze_stack_button = ttk.Button(
            button_frame,
            text='analyse_stack_button', # Sera défini par la traduction 'analyse_stack_button'
            command=self.start_analysis_and_stack, # Fonction callback (Étape 2)
            width=18,
            state=tk.NORMAL # Activé par défaut (ou tk.DISABLED si pas de dossier initial)
        )
        self.analyze_stack_button.pack(side=tk.LEFT, padx=(2, 5)) # Pack à droite du bouton Analyser
        self.widgets_refs['analyse_stack_button'] = self.analyze_stack_button # Référencer pour traduction
        # <--- FIN NOUVEAU --->

        self.visualize_button = ttk.Button(button_frame, text="", command=self.visualize_results, width=18)
        self.visualize_button.pack(side=tk.LEFT, padx=5); self.visualize_button.config(state=tk.DISABLED) # Désactivé au début
        self.widgets_refs['visualize_button'] = self.visualize_button # Référencer

        self.open_log_button = ttk.Button(button_frame, text="", command=self.open_log_file, width=18)
        self.open_log_button.pack(side=tk.LEFT, padx=5); self.open_log_button.config(state=tk.DISABLED) # Désactivé au début
        self.widgets_refs['open_log_button'] = self.open_log_button # Référencer

        self.manage_markers_button = ttk.Button(button_frame, text="", command=self.manage_markers, width=18)
        self.manage_markers_button.pack(side=tk.LEFT, padx=5)
        self.widgets_refs['manage_markers_button'] = self.manage_markers_button

        self.return_button = ttk.Button(button_frame, text="", command=self.return_or_quit, width=12)
        self.return_button.pack(side=tk.RIGHT, padx=5)
        # La référence pour le bouton Retour/Quitter sera ajoutée dans change_language
        # en utilisant self.return_button directement.

        # --- Cadre Progression et Statut ---
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate', variable=self.progress_var)
        self.progress_bar.pack(fill=tk.X, padx=5)
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X)
        self.status_label = ttk.Label(status_frame, textvariable=self.status_text)
        self.status_label.pack(side=tk.LEFT, padx=5)

        # --- Cadre Résultats / Journal ---
        results_frame = ttk.LabelFrame(main_frame, text="", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.widgets_refs['results_frame'] = results_frame
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, width=80, height=15)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.results_text.config(state=tk.DISABLED) # Lecture seule
    # --- Gestion Langue et État Widgets ---

# --- DANS LE FICHIER analyse_gui.py ---
# À l'intérieur de la classe AstroImageAnalyzerGUI
# Placez cette méthode APRES __init__ et create_widgets, près de start_analysis
       
    def change_language(self, *args):
        """Met à jour tous les textes de l'interface selon la langue sélectionnée."""
        lang = self.current_lang.get()
        self.root.title(self._("window_title")) # Mettre à jour titre fenêtre
        # Mettre à jour statut si pas en cours d'analyse
        if not self.analysis_running:
            self.status_text.set(self._("status_ready"))

        # Parcourir les widgets référencés et mettre à jour leur texte
        for key, widget in self.widgets_refs.items():
            try:
                # Gérer les LabelFrames (titre)
                if isinstance(widget, ttk.LabelFrame) and key.endswith('_frame'):
                    widget.config(text=self._(f"{key}_title"))
                # Gérer Labels et Checkbuttons (texte principal)
                elif isinstance(widget, (ttk.Label, ttk.Checkbutton)):
                    widget.config(text=self._(key))
                # Gérer les groupes de Radiobuttons
                elif key == 'snr_mode_radios':
                    for mode_val, rb in widget.items():
                        rb.config(text=self._(f"snr_mode_{mode_val}"))
                elif key == 'action_radios':
                     for mode_val, rb in widget.items():
                         rb.config(text=self._(f"action_mode_{mode_val}"))
                # Gérer les boutons "Parcourir"
                elif isinstance(widget, ttk.Button) and key.startswith('browse_'):
                    widget.config(text=self._('browse_button'))
                # Gérer le bouton "Gérer Marqueurs"
                elif isinstance(widget, ttk.Button) and key == 'manage_markers_button':
                    widget.config(text=self._('manage_markers_button'))
            except tk.TclError as e:
                # Ignorer erreurs si widget détruit pendant mise à jour
                print(f"WARN: Erreur Tcl mise à jour texte widget '{key}' lang {lang}: {e}")
            except KeyError as e:
                # Avertissement si clé de traduction manquante
                print(f"WARN: Clé traduction manquante '{e}' pour widget '{key}' lang '{lang}'.")
            except Exception as e:
                # Autres erreurs potentielles
                print(f"WARN: Erreur mise à jour texte widget '{key}' lang {lang}: {e}")

        # Mettre à jour les labels des paramètres de détection de traînées
        if hasattr(self, 'trail_param_labels'):
            for key, lbl in self.trail_param_labels.items():
                 try:
                     lbl.config(text=self._(f"{key}_label"))
                 except tk.TclError as e:
                     print(f"WARN: Erreur Tcl mise à jour label param satdet {key}: {e}")
                 except KeyError as e:
                     print(f"WARN: Clé traduction manquante '{e}' pour param satdet '{key}' lang '{lang}'.")

        # Mettre à jour le label de statut acstools
        if hasattr(self, 'acstools_status_label'):
            if not SATDET_AVAILABLE:
                txt, col = self._('acstools_missing'), 'red'
            elif not SATDET_USES_SEARCHPATTERN:
                txt, col = self._('acstools_sig_error'), 'orange'
            else:
                txt, col = self._('acstools_ok'), 'green'
            self.acstools_status_label.config(text=txt, foreground=col)

        # Mettre à jour les textes des boutons d'action principaux
        try:
            if self.analyze_button: self.analyze_button.config(text=self._("analyse_button"))
            if self.analyze_stack_button: self.analyze_stack_button.config(text=self._("analyse_stack_button"))
            if self.visualize_button: self.visualize_button.config(text=self._("visualize_button"))
            if self.open_log_button: self.open_log_button.config(text=self._("open_log_button"))
            # Texte du bouton Quitter/Retour dépend si un callback est fourni
            if self.return_button:
                btn_text = self._("return_button_text") if self.main_app_callback else self._("quit_button")
                self.return_button.config(text=btn_text)
        except tk.TclError as e:
            print(f"WARN: Erreur Tcl mise à jour texte bouton principal: {e}")
        except KeyError as e:
            print(f"WARN: Clé traduction manquante pour bouton principal: {e}")

        # Mettre à jour l'état activé/désactivé des sections
        self.toggle_sections_state()

    # --- Gestion des Chemins (Parcourir) ---


    def browse_input_dir(self):
        """Ouvre dialogue pour choisir dossier entrée et met à jour chemins par défaut."""
        directory = filedialog.askdirectory(parent=self.root, title=self._("input_dir_label"))
        if directory:
            self.input_dir.set(directory)
            # La modification de self.output_log déclenchera le trace
            self.output_log.set(os.path.join(directory, "analyse_resultats.log")) 
            self.snr_reject_dir.set(os.path.join(directory, "rejected_low_snr"))
            self.trail_reject_dir.set(os.path.join(directory, "rejected_satellite_trails"))
            
            # reset_ui_for_new_analysis appellera _update_log_and_vis_buttons_state à la fin
            self.reset_ui_for_new_analysis() 
            
        self.root.after(50, self.root.focus_force)
        self.root.after(100, self.root.lift)



    def browse_output_log(self):
        """Ouvre dialogue pour choisir/enregistrer fichier log."""
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title=self._("output_log_label"),
            defaultextension=".log",
            filetypes=[(self._("Fichiers log"), "*.log"), (self._("Tous les fichiers"), "*.*")]
        )
        if filename:
            # La modification de self.output_log déclenchera le trace
            self.output_log.set(filename) 
            
        self.root.after(50, self.root.focus_force)
        self.root.after(100, self.root.lift)



    def browse_snr_reject_dir(self):
        """Ouvre dialogue pour choisir dossier rejet SNR."""
        directory = filedialog.askdirectory(parent=self.root, title=self._("snr_reject_dir_label"))
        if directory:
            self.snr_reject_dir.set(directory)
        self.root.after(50, self.root.focus_force)
        self.root.after(100, self.root.lift)

    def browse_trail_reject_dir(self):
        """Ouvre dialogue pour choisir dossier rejet traînées."""
        directory = filedialog.askdirectory(parent=self.root, title=self._("trail_reject_dir_label"))
        if directory:
            self.trail_reject_dir.set(directory)
        self.root.after(50, self.root.focus_force)
        self.root.after(100, self.root.lift)

    # --- Gestion État Widgets ---

    def _set_widget_state(self, widget, state):
        """Active ou désactive un widget ou tous les widgets d'un cadre."""
        if widget is None: return # Ne rien faire si widget non trouvé/créé
        try:
            # Si c'est un cadre (Frame, LabelFrame), affecter tous ses enfants interactifs
            if isinstance(widget, (ttk.Frame, tk.Frame, ttk.LabelFrame)):
                for child in widget.winfo_children():
                    try:
                        # Affecter seulement les types de widgets interactifs courants
                        if isinstance(child, (ttk.Entry, ttk.Button, ttk.Checkbutton, ttk.Radiobutton, ttk.Combobox, ttk.Spinbox)):
                            child.configure(state=state)
                    except tk.TclError: pass # Ignorer si enfant détruit
            else:
                # Sinon, affecter le widget directement
                widget.configure(state=state)
        except tk.TclError: pass # Ignorer si widget détruit

    def toggle_sections_state(self, *args):
        """Met à jour l'état (activé/désactivé) des sections en fonction des options cochées."""
        try:
            # État basé sur l'activation de l'analyse SNR
            snr_enabled = self.analyze_snr.get()
            snr_state = tk.NORMAL if snr_enabled else tk.DISABLED
            self._set_widget_state(self.snr_select_frame, snr_state)

            # État du champ valeur SNR dépend du mode choisi
            snr_mode = self.snr_selection_mode.get()
            snr_value_state = tk.NORMAL if snr_enabled and snr_mode in ['percent', 'threshold'] else tk.DISABLED
            self._set_widget_state(self.snr_value_entry, snr_value_state)

            # État du dossier rejet SNR dépend de l'action et du mode
            action = self.reject_action.get()
            snr_reject_dir_state = tk.NORMAL if snr_enabled and action == 'move' and snr_mode != 'none' else tk.DISABLED
            self._set_widget_state(self.snr_reject_dir_frame, snr_reject_dir_state)

            # État basé sur l'activation de la détection de traînées
            trails_possible = SATDET_AVAILABLE and SATDET_USES_SEARCHPATTERN
            trails_enabled_by_user = self.detect_trails.get()
            trails_state = tk.NORMAL if trails_enabled_by_user and trails_possible else tk.DISABLED
            self._set_widget_state(self.params_sat_frame, trails_state)
            # Activer/désactiver la case à cocher elle-même si acstools n'est pas compatible
            self._set_widget_state(self.detect_trails_check, tk.NORMAL if trails_possible else tk.DISABLED)
            # Décocher automatiquement si non possible mais l'utilisateur l'avait cochée
            if not trails_possible and trails_enabled_by_user:
                self.detect_trails.set(False)

            # État du dossier rejet traînées
            trail_reject_dir_state = tk.NORMAL if trails_enabled_by_user and trails_possible and action == 'move' else tk.DISABLED
            self._set_widget_state(self.trail_reject_dir_frame, trail_reject_dir_state)

        except AttributeError:
            # Peut arriver si appelé avant que tous les widgets soient créés
            pass
        except Exception as e:
            # Capturer autres erreurs
            print(f"Erreur inattendue dans toggle_sections_state: {e}")
            traceback.print_exc()

    # --- Gestion Fichier Log et Réinitialisation UI ---

    def open_log_file(self):
        """Ouvre le fichier log avec l'application système par défaut."""
        log_path = self.output_log.get()
        # Vérifier si chemin spécifié et fichier existe
        if not log_path or not os.path.exists(log_path):
            messagebox.showerror(self._("msg_error"), self._("msg_log_not_exist"), parent=self.root)
            return
        try:
            # Ouvrir selon le système d'exploitation
            if platform.system() == 'Windows':
                os.startfile(log_path)
            elif platform.system() == 'Darwin': # macOS
                subprocess.call(['open', log_path])
            else: # Linux, etc.
                subprocess.call(['xdg-open', log_path])
        except Exception as e:
            # Gérer erreurs d'ouverture
            messagebox.showerror(self._("msg_error"), self._("msg_log_open_error", path=log_path, e=e), parent=self.root)


    def reset_ui_for_new_analysis(self):
        """Réinitialise les éléments UI pour une nouvelle analyse."""
        self.status_text.set(self._("status_ready"))
        self.progress_var.set(0.0)
        # ... (gestion progress_bar) ...
        if hasattr(self, 'progress_bar') and self.progress_bar: # Copié d'une version précédente
             try:
                 if self.progress_bar.winfo_exists():
                     if self.progress_bar['mode'] == 'indeterminate':
                         self.progress_bar.stop()
                     self.progress_bar.config(mode='determinate')
             except tk.TclError: pass
        # ... (vidage results_text) ...
        if hasattr(self, 'results_text') and self.results_text: # Copié d'une version précédente
            try:
                if self.results_text.winfo_exists():
                    self.results_text.config(state=tk.NORMAL) 
                    self.results_text.delete(1.0, tk.END)
                    self.results_text.config(state=tk.DISABLED)
            except tk.TclError: pass
        
        # --- MODIFIÉ : Ne plus toucher directement aux boutons ici ---
        # Leur état sera géré par _update_log_and_vis_buttons_state
        
        self.analysis_results = [] # Toujours vider les résultats en mémoire
        self.analysis_completed_successfully = False # Réinitialiser l'état de succès de la *dernière* analyse
                                                 # car on prépare une NOUVELLE analyse.

        # Mettre à jour l'état des boutons log et visu basé sur le fichier log actuel
        self._update_log_and_vis_buttons_state() 
        


    def update_status(self, message_key, **kwargs):
        """Met à jour le label de statut (thread-safe via 'after')."""
        try:
            # Obtenir le message traduit
            base_message = self._(message_key, **kwargs)
            # Clés qui arrêtent le timer
            stop_timer_keys = [
                "status_satdet_done", "status_satdet_error", "status_satdet_dep_error",
                "status_snr_start", "status_analysis_done", "status_analysis_done_some",
                "status_analysis_done_ok", "status_analysis_done_no_valid",
                "status_analysis_done_errors", "status_discovery_start"
            ]
            # Démarrer timer si attente satdet
            if message_key == "status_satdet_wait":
                self.root.after(0, self._start_timer, base_message)
            # Arrêter timer si message de fin ou début étape suivante
            elif self.timer_running and message_key in stop_timer_keys:
                self.root.after(0, self._stop_timer)
                self.root.after(0, self.status_text.set, base_message)
            # Mettre à jour texte simple si pas de timer actif
            elif not self.timer_running:
                self.root.after(0, self.status_text.set, base_message)
            # Si timer actif mais message non listé, laisser le timer tourner
        except Exception as e:
            # Gérer erreurs potentielles (ex: clé traduction invalide)
            print(f"Erreur dans update_status (clé: {message_key}): {e}")
            traceback.print_exc()
            # Essayer d'afficher un message d'erreur générique
            try:
                self.root.after(0, self.status_text.set, f"Erreur Statut: {message_key}")
            except Exception: pass # Ignorer si même ça échoue

    def update_progress(self, value):
        """Met à jour la barre de progression (thread-safe via 'after')."""
        # Appeler la fonction interne via 'after' pour garantir exécution dans thread UI
        self.root.after(0, self._set_progress_and_update, value)

    def _set_progress_and_update(self, value):
        """Fonction interne pour mettre à jour la barre de progression."""
        try:
            # Vérifier si la fenêtre et la barre existent toujours
            if self.root.winfo_exists() and self.progress_bar and self.progress_bar.winfo_exists():
                current_mode = self.progress_bar['mode']
                # Si mode indéterminé et on reçoit une valeur numérique -> passer en déterminé
                if current_mode == 'indeterminate' and isinstance(value, (int, float)):
                    self.progress_bar.stop()
                    self.progress_bar.config(mode='determinate')
                # Si on demande explicitement le mode indéterminé
                elif value is None or value == 'indeterminate':
                    if current_mode == 'determinate':
                        self.progress_bar.config(mode='indeterminate')
                        self.progress_bar.start(10) # Démarrer animation
                # Si on reçoit une valeur numérique
                elif isinstance(value, (int, float)):
                    self.progress_var.set(value) # Mettre à jour la variable liée

                # Forcer mise à jour UI si nécessaire (souvent utile pour voir la barre bouger)
                if self.root.winfo_exists():
                    self.root.update_idletasks()
        except tk.TclError:
            # Ignorer erreurs si widget détruit
            pass
        except Exception as e:
            print(f"Erreur dans _set_progress_and_update: {e}")

    def update_results_text(self, text_key, clear=False, **kwargs):
        """Ajoute un message à la zone de texte des résultats (thread-safe via 'after')."""
        try:
            # Traiter kwargs spéciaux (ex: booléen 'has_trails')
            processed_kwargs = kwargs.copy()
            if 'has_trails' in processed_kwargs:
                status_text = self._('logic_trail_yes') if processed_kwargs['has_trails'] else self._('logic_trail_no')
                processed_kwargs['status'] = status_text # Ajouter 'status' pour formatage

            # Obtenir texte traduit et ajouter timestamp
            text = self._(text_key, **processed_kwargs)
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            full_text = f"[{timestamp}] {text}"

            # Planifier l'insertion dans le thread UI
            self.root.after(0, self._insert_results_text, full_text, clear)
        except Exception as e:
            # Gérer erreurs (clé traduction, formatage)
            print(f"Erreur dans update_results_text (clé: {text_key}, kwargs: {kwargs}): {e}")
            traceback.print_exc()

    def _insert_results_text(self, text, clear):
        """Fonction interne pour insérer du texte dans la zone de résultats."""
        try:
            # Vérifier si la zone de texte existe
            if self.results_text and self.results_text.winfo_exists():
                # Activer temporairement pour modification
                self.results_text.config(state=tk.NORMAL)
                if clear:
                    self.results_text.delete(1.0, tk.END) # Effacer si demandé
                # Insérer nouveau texte et faire défiler vers le bas
                self.results_text.insert(tk.END, text + "\n")
                self.results_text.see(tk.END)
                # Redésactiver (lecture seule)
                self.results_text.config(state=tk.DISABLED)
        except tk.TclError:
            # Ignorer erreurs si widget détruit
            pass
        except Exception as e:
            print(f"Erreur dans _insert_results_text: {e}")


    def run_analysis_thread(self, input_dir, output_log, options, callbacks):
        """Fonction exécutée dans le thread séparé pour appeler la logique d'analyse."""
        results = []
        success = False
        try:
            # Appel de la fonction principale du module logique
            results = analyse_logic.perform_analysis(input_dir, output_log, options, callbacks)
            # Considérer comme succès si on reçoit une liste (même vide)
            success = isinstance(results, list)
        except Exception as e:
            # Gérer erreurs critiques inattendues DANS la logique métier
            print(f"ERREUR CRITIQUE inattendue dans analyse_logic: {e}")
            traceback.print_exc()
            # Logguer l'erreur via le callback si possible
            log_callback = callbacks.get('log', lambda k, **kw: None)
            log_callback("logic_error_prefix", text=f"Erreur critique inattendue: {e}\n{traceback.format_exc()}")
            # Mettre à jour le statut via callback
            status_callback = callbacks.get('status', lambda k, **kw: None)
            status_callback("status_analysis_done_errors")
            success = False
            results = [] # Assurer que results est une liste vide en cas d'erreur
        finally:
            # Planifier la finalisation dans le thread UI (toujours exécuté)
            self.root.after(0, self.finalize_analysis, results, success)
##########################################################################################################################

    def export_recommended_list(self, images_to_export, criterion_value):
        """
        Ouvre une boîte de dialogue pour enregistrer la liste des images recommandées 
        dans un fichier texte.
        """
        if not images_to_export:
            messagebox.showwarning(self._("msg_warning"), self._("msg_export_no_images"), parent=self.root)
            return

        # Proposer un nom de fichier par défaut
        default_filename = f"recommended_images_snr_gt_{criterion_value:.2f}.txt" if criterion_value != -1 else "all_kept_images.txt"
        
        save_path = filedialog.asksaveasfilename(
            parent=self.root, # Assurer que la fenêtre de dialogue est modale à la fenêtre de visu si elle est ouverte
            title=self._("Liste dimages recommandées", default="Save Recommended List"), 
            defaultextension=".txt",
            initialfile=default_filename,
            filetypes=[(self._("Fichiers Texte", default="Text Files"), "*.txt"), 
                       (self._("Tous les fichiers", default="All Files"), "*.*")]
        )

        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(f"# {self._('Liste dimages recommandées', default='Recommended image list')}\n")
                    if criterion_value != -1: # -1 est un marqueur pour "toutes les conservées"
                        f.write(f"# {self._('Critère', default='Criterion')}: SNR >= {criterion_value:.2f} (P25)\n")
                    else:
                        f.write(f"# {self._('Critère', default='Criterion')}: {self._('Toutes les images conservées valides', default='All valid kept images')}\n")
                    f.write(f"# {self._('Généré le', default='Generated on')}: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# {self._('Nombre dimages', default='Number of images')}: {len(images_to_export)}\n\n")
                    
                    for img_data in images_to_export:
                        # Utiliser 'rel_path' s'il existe, sinon 'file'
                        file_to_write = img_data.get('rel_path', os.path.basename(img_data.get('file', 'UNKNOWN_FILE')))
                        f.write(f"{file_to_write}\n")
                
                messagebox.showinfo(
                    self._("msg_info"), 
                    self._("msg_export_success", count=len(images_to_export), path=save_path),
                    parent=self.root
                )
            except IOError as e_io:
                messagebox.showerror(
                    self._("msg_error"), 
                    self._("msg_export_error", e=e_io),
                    parent=self.root
                )
            except Exception as e_exp: # Capturer d'autres erreurs potentielles
                messagebox.showerror(
                    self._("msg_error"), 
                    self._("msg_unexpected_error", e=e_exp),
                    parent=self.root
                )



# --- DANS analyse_gui.py ---

    def finalize_analysis(self, results, success):
        """Met à jour l'interface après la fin du thread d'analyse."""
        print("DEBUG (analyse_gui): Entrée dans finalize_analysis.") 
        print(f"DEBUG (analyse_gui): Paramètres reçus - success: {success}, stack_after_analysis (avant logique): {self.stack_after_analysis}") 

        self._stop_timer()
        self.analysis_running = False
        self.has_pending_snr_actions = False # Réinitialiser par défaut

        folder_to_stack = None
        should_write_command = False 

        if self.stack_after_analysis and success:
            print("DEBUG (analyse_gui): Analyse réussie et intention d'empiler détectée.") 
            self.update_results_text("logic_info_prefix", text="Analyse terminée avec succès, préparation pour empilage.")
            folder_to_stack = self.input_dir.get()

            if not folder_to_stack or not os.path.isdir(folder_to_stack):
                print(f"DEBUG (analyse_gui): Dossier pour empilage invalide: '{folder_to_stack}'") 
                self.update_results_text("logic_error_prefix", text=f"Dossier invalide ou non défini pour empiler après analyse: '{folder_to_stack}'. Empilage annulé.")
                messagebox.showerror(self._("msg_error"), self._("msg_input_dir_invalid") + "\n" + self._("Empilage annulé.", default="Stacking cancelled."), parent=self.root)
                folder_to_stack = None
                success = False 
                should_write_command = False 
            else:
                print(f"DEBUG (analyse_gui): Dossier pour empilage validé: {folder_to_stack}") 
                should_write_command = True 

            self.stack_after_analysis = False 
        else:
            if not success: print("DEBUG (analyse_gui): Analyse échouée, commande non écrite.") 
            if not self.stack_after_analysis: print("DEBUG (analyse_gui): Pas d'intention d'empiler après analyse, commande non écrite.") 
            self.stack_after_analysis = False 

        print(f"DEBUG (analyse_gui): État après logique décision: should_write_command={should_write_command}, folder_to_stack={folder_to_stack}") 

        self.update_progress(100.0 if success else 0.0)
        self.analysis_results = results if results else []
        self.analysis_completed_successfully = success
        final_status_key = ""
        processed_count = 0 ; action_count = 0 ; errors_count = 0

        # --- NOUVEAU : Vérifier les actions SNR en attente ---
        if success and self.analysis_results:
            for r in self.analysis_results:
                if r.get('rejected_reason') == 'low_snr_pending_action':
                    self.has_pending_snr_actions = True
                    break
        # --- FIN NOUVEAU ---

        if success:
            processed_count = len(self.analysis_results)
            if processed_count > 0:
                 errors_count = sum(1 for r in self.analysis_results if r.get('status') == 'error')
                 # L'action_count ici reflète les actions immédiates (ex: pour les traînées)
                 action_count = sum(1 for r in self.analysis_results if r.get('action','').startswith(('moved_trail', 'deleted_trail')))
                 final_status_key = "status_analysis_done_some" # Ajuster ce message si besoin pour refléter les actions en attente
                 self.update_results_text("--- Analyse terminée ---")
                 if not should_write_command: self._set_widget_state(self.visualize_button, tk.NORMAL)
                 else: self._set_widget_state(self.visualize_button, tk.DISABLED)
            else:
                 final_status_key = "status_analysis_done_no_valid"
                 self.update_results_text("--- Analyse terminée (aucun fichier traitable trouvé ou tous ignorés) ---")
                 self._set_widget_state(self.visualize_button, tk.DISABLED)
        else: # Echec de l'analyse
             final_status_key = "status_analysis_done_errors"
             self.update_results_text("--- Analyse terminée avec erreurs ---")
             self._set_widget_state(self.visualize_button, tk.DISABLED)
             self.has_pending_snr_actions = False # Pas d'actions en attente si l'analyse a échoué globalement

        if final_status_key:
            print(f"DEBUG (analyse_gui): Affichage statut final (clé: {final_status_key})") 
            status_kwargs = {}
            if success and processed_count > 0: status_kwargs = {'processed': processed_count, 'moved': action_count, 'errors': errors_count}
            self.update_status(final_status_key, **status_kwargs)
            if self.has_pending_snr_actions:
                self.update_status("status_custom", text=self._("Des actions SNR sont en attente.", default="Pending SNR actions.")) # Clé à ajouter à zone.py

        if not should_write_command:
            self._set_widget_state(self.analyze_button, tk.NORMAL)
            self._set_widget_state(self.analyze_stack_button, tk.NORMAL)
            self._set_widget_state(self.return_button, tk.NORMAL)
            self._set_widget_state(self.manage_markers_button, tk.NORMAL)
            self._update_log_and_vis_buttons_state() 
            # --- NOUVEAU : Activer/Désactiver le bouton "Appliquer Rejet SNR" ---
            if hasattr(self, 'apply_snr_button') and self.apply_snr_button:
                can_apply_snr_action = (
                    self.has_pending_snr_actions and
                    self.analyze_snr.get() and # L'analyse SNR doit être active
                    self.snr_selection_mode.get() != 'none' and # Un mode de sélection doit être actif
                    self.reject_action.get() != 'none' # Une action (move/delete) doit être sélectionnée
                )
                if self.reject_action.get() == 'move' and not self.snr_reject_dir.get():
                    can_apply_snr_action = False # Ne pas activer si déplacement requis mais dossier non spécifié
                    
                self._set_widget_state(self.apply_snr_button, tk.NORMAL if can_apply_snr_action else tk.DISABLED)
            # --- FIN NOUVEAU ---
        
        if should_write_command and folder_to_stack:
            if self.command_file_path:
                try:
                    # ... (écriture fichier commande) ...
                    self.root.after(100, self.return_or_quit) 
                except Exception as e_write_cmd:
                    # ... (gestion erreur écriture) ...
                    self._set_widget_state(self.analyze_button, tk.NORMAL)
                    self._set_widget_state(self.analyze_stack_button, tk.NORMAL)
                    self._set_widget_state(self.return_button, tk.NORMAL)
                    self._set_widget_state(self.manage_markers_button, tk.NORMAL)
                    self._update_log_and_vis_buttons_state() 
                    # Désactiver aussi apply_snr_button en cas d'erreur ici, car le flux est interrompu
                    if hasattr(self, 'apply_snr_button') and self.apply_snr_button:
                         self._set_widget_state(self.apply_snr_button, tk.DISABLED)

            else: # command_file_path non défini
                # ... (gestion erreur) ...
                self._set_widget_state(self.analyze_button, tk.NORMAL)
                self._set_widget_state(self.analyze_stack_button, tk.NORMAL)
                self._set_widget_state(self.return_button, tk.NORMAL)
                self._set_widget_state(self.manage_markers_button, tk.NORMAL)
                self._update_log_and_vis_buttons_state()
                if hasattr(self, 'apply_snr_button') and self.apply_snr_button:
                     self._set_widget_state(self.apply_snr_button, tk.DISABLED)
        
        print("DEBUG (analyse_gui): Appel final à gc.collect()") 
        gc.collect()
        print("DEBUG (analyse_gui): Sortie de finalize_analysis.")

    def apply_pending_snr_actions_gui(self):
        """
        Lance l'application des actions SNR en attente (déplacement/suppression).
        Appelée par le bouton "Appliquer Rejet SNR".
        """
        if self.analysis_running: # Ne pas lancer si une autre analyse/action est en cours
            messagebox.showwarning(self._("msg_warning"), self._("msg_analysis_running"), parent=self.root)
            return

        if not self.has_pending_snr_actions:
            messagebox.showinfo(self._("msg_info"), self._("Aucune action SNR en attente à appliquer.", default="No pending SNR actions to apply."), parent=self.root)
            return

        # Vérifier la configuration de l'action sur rejet
        reject_action = self.reject_action.get()
        move_rejected_flag = (reject_action == 'move')
        delete_rejected_flag = (reject_action == 'delete')

        if not move_rejected_flag and not delete_rejected_flag:
            messagebox.showinfo(self._("msg_info"), self._("L'action sur rejet est 'Ne Rien Faire'. Aucune action ne sera appliquée.", default="Reject action is 'Do Nothing'. No file actions will be performed."), parent=self.root)
            # On pourrait quand même finaliser les statuts dans results_list si 'low_snr_pending_action' est juste transformé en 'low_snr'
            # Pour l'instant, on sort.
            # On peut aussi désactiver le bouton si 'Ne Rien Faire' est sélectionné (géré dans finalize_analysis)
            return
        
        snr_reject_dir_path = self.snr_reject_dir.get()
        if move_rejected_flag and not snr_reject_dir_path:
            messagebox.showerror(self._("msg_error"), self._("snr_reject_dir_label") + " " + self._("non spécifié") + self._(" pour le déplacement des rejets SNR.", default=" for moving SNR rejects."), parent=self.root)
            return
        
        abs_snr_reject_path = os.path.abspath(snr_reject_dir_path) if snr_reject_dir_path else None
        abs_input_dir = os.path.abspath(self.input_dir.get()) if self.input_dir.get() else None

        if delete_rejected_flag:
             if not messagebox.askyesno(self._("msg_warning"), self._("confirm_delete") + self._("\nCeci s'appliquera aux fichiers marqués pour faible SNR.", default="\nThis will apply to files marked for low SNR."), parent=self.root):
                return # Annulé par l'utilisateur

        # Préparer UI pour l'action
        self.analysis_running = True # Utiliser le même flag pour bloquer d'autres actions
        self._set_widget_state(self.analyze_button, tk.DISABLED)
        self._set_widget_state(self.analyze_stack_button, tk.DISABLED)
        self._set_widget_state(self.visualize_button, tk.DISABLED)
        self._set_widget_state(self.open_log_button, tk.DISABLED)
        self._set_widget_state(self.return_button, tk.DISABLED)
        self._set_widget_state(self.manage_markers_button, tk.DISABLED)
        if hasattr(self, 'apply_snr_button') and self.apply_snr_button:
            self._set_widget_state(self.apply_snr_button, tk.DISABLED)

        self.update_status("status_custom", text=self._("Application des rejets SNR...", default="Applying SNR rejections..."))
        self.update_results_text(f"--- {self._('Début application rejets SNR...', default='Start applying SNR rejections...')} ---")
        self.update_progress(0.0) # Ou 'indeterminate'

        callbacks = {
            'log': self.update_results_text,
            'status': self.update_status,
            'progress': self.update_progress
        }

        # Lancer l'application des actions dans un thread
        action_thread = threading.Thread(
            target=self.run_apply_actions_thread,
            args=(self.analysis_results, abs_snr_reject_path, delete_rejected_flag, move_rejected_flag, callbacks, abs_input_dir),
            daemon=True
        )
        action_thread.start()

    def run_apply_actions_thread(self, results_list, snr_reject_abs, delete_flag, move_flag, callbacks, input_dir_abs):
        """
        Thread worker pour appeler analyse_logic.apply_pending_snr_actions.
        """
        actions_done_count = 0
        apply_success = False
        try:
            # Appel à la fonction logique
            actions_done_count = analyse_logic.apply_pending_snr_actions(
                results_list, 
                snr_reject_abs, 
                delete_flag, 
                move_flag,
                callbacks['log'],
                callbacks['status'],
                callbacks['progress'],
                input_dir_abs # Passé pour le calcul de rel_path dans les logs
            )
            apply_success = True # Considérer comme succès si la fonction s'exécute sans lever d'exception
        except Exception as e:
            print(f"ERREUR CRITIQUE inattendue dans apply_pending_snr_actions: {e}")
            traceback.print_exc()
            log_callback = callbacks.get('log', lambda k, **kw: None)
            log_callback("logic_error_prefix", text=f"Erreur critique application actions SNR: {e}\n{traceback.format_exc()}")
            status_callback = callbacks.get('status', lambda k, **kw: None)
            status_callback("status_custom", text=self._("Erreur application actions SNR.", default="Error applying SNR actions."))
            apply_success = False
        finally:
            # Planifier la finalisation dans le thread UI
            self.root.after(0, self.finalize_apply_actions, actions_done_count, apply_success)

    def finalize_apply_actions(self, actions_done_count, apply_success):
        """
        Met à jour l'interface après l'application des actions SNR différées.
        """
        self.analysis_running = False # Libérer le flag
        self.has_pending_snr_actions = False # Normalement, toutes les actions en attente ont été traitées ou ont échoué

        self.update_progress(100.0)
        if apply_success:
            self.update_status("status_custom", text=self._("{count} actions SNR appliquées.", default="{count} SNR actions applied.").format(count=actions_done_count))
            self.update_results_text(f"--- {self._('Fin application rejets SNR. {count} actions effectuées.', default='End applying SNR rejections. {count} actions performed.').format(count=actions_done_count)} ---")
            
            # Réécrire le résumé du log pour refléter les actions maintenant effectuées
            # et mettre à jour les données de visualisation dans le log.
            # On a besoin des options originales de l'analyse pour cela.
            # Pour simplifier, on ne réécrit que si output_log est défini.
            # Une solution plus robuste stockerait les options.
            if self.output_log.get():
                try:
                    # Recréer un dictionnaire d'options minimal pour le log
                    # Ce n'est pas parfait car on n'a pas toutes les options originales de l'analyse.
                    # Idéalement, on stockerait les 'options' utilisées lors de l'analyse initiale.
                    # Pour l'instant, on reconstruit ce qu'on peut.
                    current_options = {
                        'analyze_snr': self.analyze_snr.get(),
                        'detect_trails': self.detect_trails.get(), # Peut ne pas être pertinent ici mais pour la structure de write_log_summary
                        'include_subfolders': self.include_subfolders.get(),
                        'move_rejected': (self.reject_action.get() == 'move'),
                        'delete_rejected': (self.reject_action.get() == 'delete'),
                        'snr_reject_dir': self.snr_reject_dir.get(),
                        'trail_reject_dir': self.trail_reject_dir.get(),
                        'snr_selection_mode': self.snr_selection_mode.get(),
                        'snr_selection_value': self.snr_selection_value.get(),
                        'trail_params': { k: self.trail_params[k].get() for k in self.trail_params }
                        # On n'a pas 'apply_snr_action_immediately' ici car l'action est maintenant appliquée.
                    }
                    # On n'a pas non plus trail_analysis_config, sat_errors, selection_stats, skipped_dirs_count
                    # pour un simple update après apply_pending.
                    # On se concentre sur la mise à jour de results_list dans le log.
                    # write_log_summary va ajouter à la fin du fichier.
                    # Il faudrait peut-être une fonction pour *mettre à jour* la section JSON.
                    # Pour l'instant, on ajoute un nouveau résumé et une nouvelle section JSON.
                    # Ce n'est pas idéal mais fonctionnel.
                    
                    # Pour éviter de dupliquer tout le résumé, on logue juste une note
                    with open(self.output_log.get(), 'a', encoding='utf-8') as log_f:
                        log_f.write(f"\n--- MISE A JOUR APRES ACTIONS SNR DIFFEREES ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
                        log_f.write(f"{actions_done_count} actions SNR (déplacement/suppression) ont été appliquées.\n")
                    
                    # Réécrire la section JSON avec les `results_list` mis à jour
                    # (qui ont maintenant les 'action' et 'path' finaux pour les rejets SNR)
                    # On utilise une astuce : on appelle write_log_summary avec un minimum d'infos
                    # mais on s'assure que results_list est passé pour que la section JSON soit réécrite.
                    # Cela va ajouter UN AUTRE résumé et une AUTRE section JSON à la fin.
                    # Pour une vraie mise à jour, il faudrait relire tout le log, modifier, et réécrire.
                    analyse_logic.write_log_summary(
                        self.output_log.get(),
                        self.input_dir.get(),
                        current_options, # Options reconstruites
                        results_list=self.analysis_results, # results_list est mis à jour en place par apply_pending_snr_actions
                        # Les autres paramètres peuvent être None ou des valeurs par défaut.
                    )
                    self.update_results_text(self._("Le fichier log a été mis à jour avec les actions SNR.", default="Log file updated with SNR actions."))

                except Exception as e_log_update:
                    print(f"Erreur mise à jour log après actions SNR: {e_log_update}")
                    self.update_results_text("logic_error_prefix", text=self._("Erreur mise à jour fichier log: {e}", default="Error updating log file: {e}").format(e=e_log_update))
        else: # apply_success is False
            self.update_status("status_custom", text=self._("Échec de l'application des actions SNR.", default="Failed to apply SNR actions."))
            self.update_results_text(f"--- {self._('Échec application rejets SNR.', default='Failed applying SNR rejections.')} ---")

        # Réactiver les boutons principaux
        self._set_widget_state(self.analyze_button, tk.NORMAL)
        self._set_widget_state(self.analyze_stack_button, tk.NORMAL)
        self._set_widget_state(self.return_button, tk.NORMAL)
        self._set_widget_state(self.manage_markers_button, tk.NORMAL)
        
        # Mettre à jour l'état des boutons log et visu (important!)
        # Cela rechargera les données depuis le log (qui a été potentiellement mis à jour)
        # et réévaluera si la visualisation est possible.
        self._update_log_and_vis_buttons_state() 

        # Le bouton "Appliquer Rejet SNR" devrait maintenant être désactivé car 
        # has_pending_snr_actions est False. On le force au cas où.
        if hasattr(self, 'apply_snr_button') and self.apply_snr_button:
            self._set_widget_state(self.apply_snr_button, tk.DISABLED)
        
        gc.collect()

 #####################################################################################################################

def check_dependencies():
    """Vérifie les dépendances requises et propose l'installation."""

    # Fonction interne pour obtenir les traductions (utilise 'en' comme fallback)
    def basic_gettext(key, default_text=""):
        # Utilise la variable globale 'translations' chargée au début du script
        return translations.get('en', {}).get(key, default_text)

    # Liste pour stocker les dépendances manquantes
    missing = [] # <--- Utilisation de 'missing'

    # Vérifier chaque dépendance essentielle
    try:
        import astropy
    except ImportError:
        missing.append("astropy") # <--- Utilisation de 'missing'
    try:
        import numpy
    except ImportError:
        missing.append("numpy") # <--- Utilisation de 'missing'
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib") # <--- CORRECTION: Utilisation de 'missing'
    # acstools lui-même est vérifié via SATDET_AVAILABLE, mais ses propres dépendances sont vérifiées ici
    # Vérifier skimage et scipy qui sont nécessaires pour satdet version Hough
    try:
        import skimage
    except ImportError:
        missing.append("scikit-image") # Nom pip <--- CORRECTION: Utilisation de 'missing'
    try:
        import scipy
    except ImportError:
        missing.append("scipy") # <--- CORRECTION: Utilisation de 'missing'

    # Si des dépendances manquent
    if missing:
        # Construire le message pour l'utilisateur
        deps_str = "\n- ".join(missing)
        msg = basic_gettext('msg_dep_missing_text', "Missing libraries:\n- {deps}\n\nInstall via pip?").format(deps=deps_str)
        title = basic_gettext('msg_dep_missing_title', "Missing Dependencies")

        # Créer une fenêtre racine temporaire si nécessaire pour la messagebox
        temp_root = None
        if not tk._default_root: # Vérifie s'il y a déjà une racine Tk par défaut
            temp_root = tk.Tk()
            temp_root.withdraw() # Cacher la fenêtre temporaire

        install = False # Initialiser la variable
        try:
            # Demander à l'utilisateur s'il veut installer
            install = messagebox.askyesno(title, msg)
        finally:
            # Détruire la fenêtre temporaire si elle a été créée
            if temp_root:
                temp_root.destroy()

        # Si l'utilisateur accepte l'installation
        if install:
            print(basic_gettext('msg_dep_installing', "Attempting dependency installation..."))
            install_ok = True # Indicateur de succès global
            # Boucler sur chaque dépendance manquante
            for dep_name in missing:
                 # Obtenir le nom du package (ex: 'scikit-image' de 'scikit-image (requis...)')
                 package_name = dep_name.split(" ")[0]
                 print(basic_gettext('msg_dep_install_pkg', "Installing {package}...").format(package=package_name))
                 try:
                     # Exécuter pip pour installer le package
                     # Utilise sys.executable pour garantir l'utilisation du bon interpréteur Python
                     process = subprocess.run(
                         [sys.executable, "-m", "pip", "install", package_name],
                         capture_output=True, # Capturer sortie standard et erreur
                         text=True,           # Décoder sortie en texte
                         check=False,         # Ne pas lever d'exception si pip échoue (on vérifie returncode)
                         encoding='utf-8'     # Spécifier l'encodage
                     )
                     # Vérifier le code de retour de pip
                     if process.returncode == 0:
                         print(basic_gettext('msg_dep_install_success', " -> Success."))
                     else:
                         # Afficher erreur si pip échoue
                         print(basic_gettext('msg_dep_install_fail', " -> FAILED (pip exit code {code})").format(code=process.returncode))
                         print(f"PIP Output:\n{process.stdout}\n{process.stderr}")
                         err_detail = process.stderr or process.stdout or "Unknown pip error"
                         # Afficher une boîte d'erreur Tkinter
                         temp_root_err = tk.Tk(); temp_root_err.withdraw()
                         messagebox.showerror("Error", basic_gettext('msg_dep_install_error', "Could not install {package}.\n{e}").format(package=package_name, e=err_detail[:1000])) # Limiter taille message
                         temp_root_err.destroy()
                         install_ok = False # Marquer échec global
                 except FileNotFoundError:
                     # Gérer cas où 'pip' n'est pas trouvé
                     print(basic_gettext('msg_dep_install_fail', " -> FAILED: 'pip' command not found?"))
                     temp_root_err = tk.Tk(); temp_root_err.withdraw()
                     messagebox.showerror("Error", basic_gettext('msg_dep_install_error', "Could not install {package}.\n{e}").format(package=package_name, e="'pip' not found. Is Python installed correctly?"))
                     temp_root_err.destroy()
                     install_ok = False; break # Inutile de continuer
                 except Exception as e:
                     # Gérer autres erreurs d'installation
                     print(basic_gettext('msg_dep_install_fail', " -> FAILED: {e}").format(e=e))
                     temp_root_err = tk.Tk(); temp_root_err.withdraw()
                     messagebox.showerror("Error", basic_gettext('msg_dep_install_error', "Could not install {package}.\n{e}").format(package=package_name, e=e))
                     temp_root_err.destroy()
                     install_ok = False

            # Afficher message final après tentative d'installation
            temp_root_info = tk.Tk(); temp_root_info.withdraw()
            if install_ok:
                # Si tout s'est bien passé, demander redémarrage
                messagebox.showinfo("Info", basic_gettext('msg_dep_install_done', "Dependencies installed. Please restart the application."), parent=temp_root_info)
                temp_root_info.destroy()
                sys.exit(0) # Quitter pour forcer redémarrage
            else:
                # Si échec partiel, avertir
                messagebox.showwarning("Warning", basic_gettext('msg_dep_install_partial', "Some dependencies failed to install. The application might not work correctly."), parent=temp_root_info)
                temp_root_info.destroy()
        else:
            # Si l'utilisateur refuse l'installation, avertir et continuer (ou quitter?)
            temp_root_warn = tk.Tk(); temp_root_warn.withdraw()
            messagebox.showwarning("Warning", basic_gettext('msg_dep_error_continue', "Dependencies are missing. The application might not work correctly."), parent=temp_root_warn)
            temp_root_warn.destroy()
            # Optionnel: sys.exit(1) ici pour forcer l'arrêt si les dépendances sont absolument critiques.


########################################################################################################################


# --- Bloc d'Exécution Principal ---
if __name__ == "__main__":
    print("DEBUG (analyse_gui main): Parsing des arguments...")
    parser = argparse.ArgumentParser(description="Astro Image Analyzer GUI")
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Optional: Pre-fill the input directory path."
    )
    parser.add_argument(
        "--command-file",
        type=str,
        metavar="CMD_FILE_PATH",
        help="Internal: Path to the command file for communicating with the main stacker GUI."
    )
    args = parser.parse_args()
    print(f"DEBUG (analyse_gui main): Arguments parsés: {args}")

    root = None
    try:
        # Vérifier si les modules essentiels sont importables
        if 'analyse_logic' not in sys.modules: raise ImportError("analyse_logic.py could not be imported.")
        if 'translations' not in globals() or not translations: raise ImportError("zone.py is empty or could not be imported.")

        # Créer la fenêtre racine Tkinter mais la cacher initialement
        root = tk.Tk(); root.withdraw()

        # Vérifier les dépendances externes
        check_dependencies() # S'assure que les dépendances sont là avant de continuer

        # Afficher la fenêtre principale
        root.deiconify()
        
        # Instancier l'application GUI
        app = AstroImageAnalyzerGUI(root, command_file_path=args.command_file, main_app_callback=None)

        # --- Pré-remplissage dossier d'entrée ---
        if args.input_dir:
            input_path_from_arg = os.path.abspath(args.input_dir)
            if os.path.isdir(input_path_from_arg):
                print(f"INFO (analyse_gui main): Pré-remplissage dossier entrée depuis argument: {input_path_from_arg}")
                app.input_dir.set(input_path_from_arg)
                
                # Définir le chemin du log par défaut basé sur l'input_dir.
                # L'appel à app.output_log.set() déclenchera le trace
                # qui appellera _update_log_and_vis_buttons_state().
                # Cette dernière fonction tentera de charger les données de visualisation
                # et mettra à jour l'état des boutons "Ouvrir Log" et "Visualiser".
                default_log_path = os.path.join(input_path_from_arg, "analyse_resultats.log")
                app.output_log.set(default_log_path) 
                
                # Pré-remplir les dossiers de rejet s'ils sont vides
                if not app.snr_reject_dir.get(): 
                    app.snr_reject_dir.set(os.path.join(input_path_from_arg, "rejected_low_snr"))
                if not app.trail_reject_dir.get(): 
                    app.trail_reject_dir.set(os.path.join(input_path_from_arg, "rejected_satellite_trails"))
            else:
                print(f"AVERTISSEMENT (analyse_gui main): Dossier d'entrée via argument invalide: {args.input_dir}")
        else:
            # Si aucun --input-dir n'est fourni, self.output_log sera vide (ou sa valeur par défaut si définie avant create_widgets).
            # L'appel à _update_log_and_vis_buttons_state() dans la méthode __init__
            # (après create_widgets) aura déjà géré la désactivation des boutons
            # "Ouvrir Log" et "Visualiser" si self.output_log est vide ou ne pointe pas vers un fichier valide.
            pass # Aucune action spécifique nécessaire ici car __init__ s'en charge.
        
        # Lancer la boucle principale de Tkinter
        root.mainloop()

    # --- Gestion des Erreurs au Démarrage ---
    except ImportError as e:
        print(f"ERREUR CRITIQUE: Échec import module au démarrage: {e}", file=sys.stderr); traceback.print_exc();
        try: 
            if root is None: root = tk.Tk(); root.withdraw(); 
            messagebox.showerror("Erreur Fichier Manquant", f"Impossible de charger un module essentiel ({e}).\nVérifiez que analyse_logic.py et zone.py sont présents et valides.")
            if root: root.destroy() 
        except Exception as msg_e: 
            print(f" -> Erreur affichage message: {msg_e}", file=sys.stderr)
        sys.exit(1) 
    except SystemExit as e: 
        print(f"DEBUG (analyse_gui main): Argparse a quitté (probablement '-h' ou erreur argument). Code: {e.code}")
        pass
    except tk.TclError as e:
        print(f"Erreur Tcl/Tk: Impossible d'initialiser l'interface graphique. {e}", file=sys.stderr)
        print("Assurez-vous d'exécuter ce script dans un environnement graphique.", file=sys.stderr)
        sys.exit(1)
    except Exception as e_main:
        print(f"Erreur inattendue au démarrage: {e_main}", file=sys.stderr); traceback.print_exc();
        try:
            if root is None: root = tk.Tk(); root.withdraw();
            messagebox.showerror("Erreur Inattendue", f"Une erreur s'est produite au démarrage:\n{e_main}")
            if root: root.destroy()
        except Exception as msg_e: 
            print(f" -> Erreur affichage message: {msg_e}", file=sys.stderr)
        sys.exit(1)

# --- FIN DU FICHIER analyse_gui.py ---