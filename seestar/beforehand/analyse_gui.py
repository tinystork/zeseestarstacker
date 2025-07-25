# -----------------------------------------------------------------------------
# Auteur       : TRISTAN NAULEAU 
# Date         : 2025-07-12
# Licence      : GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
#
# Ce travail est distribué librement en accord avec les termes de la
# GNU GPL v3 (https://www.gnu.org/licenses/gpl-3.0.html).
# Vous êtes libre de redistribuer et de modifier ce code, à condition
# de conserver cette notice et de mentionner que je suis l’auteur
# de tout ou partie du code si vous le réutilisez.
# -----------------------------------------------------------------------------
# Author       : TRISTAN NAULEAU
# Date         : 2025-07-12
# License      : GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
#
# This work is freely distributed under the terms of the
# GNU GPL v3 (https://www.gnu.org/licenses/gpl-3.0.html).
# You are free to redistribute and modify this code, provided that
# you keep this notice and mention that I am the author
# of all or part of the code if you reuse it.
# -----------------------------------------------------------------------------

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
from matplotlib.widgets import RangeSlider
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
import importlib.util
import numbers
from stack_plan import generate_stacking_plan, write_stacking_plan_csv

# Détection de l'environnement : intégré ou autonome
try:
    import zeseestarstacker  # package parent
    _EMBEDDED_IN_STACKER = True
except ImportError:
    _EMBEDDED_IN_STACKER = False

# Helper to safely check numeric finite values
def is_finite_number(value):
    """Return True if value is a real number and finite."""
    return isinstance(value, numbers.Number) and np.isfinite(value)

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

# Ajout fonction apply_pending_starcount_actions si absente dans analyse_logic
if not hasattr(analyse_logic, 'apply_pending_starcount_actions'):
    def apply_pending_starcount_actions(results_list, sc_reject_abs_path,
                                       delete_rejected_flag, move_rejected_flag,
                                       log_callback, status_callback, progress_callback,
                                       input_dir_abs):
        actions_count = 0
        if not results_list:
            return actions_count

        _log = log_callback if callable(log_callback) else lambda k, **kw: None
        _status = status_callback if callable(status_callback) else lambda k, **kw: None
        _progress = progress_callback if callable(progress_callback) else lambda v: None

        to_process = [r for r in results_list if r.get('rejected_reason') == 'starcount_pending_action' and r.get('status') == 'ok']
        total = len(to_process)
        if total == 0:
            _log('logic_info_prefix', text='Aucune action Starcount en attente.')
            return 0

        _status('status_custom', text=f'Application des actions Starcount différées sur {total} fichiers...')
        _progress(0)

        for i, r in enumerate(to_process):
            current_progress = ((i + 1) / total) * 100
            _progress(current_progress)
            try:
                rel_path = os.path.relpath(r.get('path'), input_dir_abs) if r.get('path') and input_dir_abs else r.get('file', 'N/A')
            except ValueError:
                rel_path = r.get('file', 'N/A')

            _status('status_custom', text=f'Action Starcount sur {rel_path} ({i+1}/{total})')

            current_path = r.get('path')
            if not current_path or not os.path.exists(current_path):
                _log('logic_move_skipped', file=rel_path, e='Fichier source introuvable pour action Starcount différée.')
                r['action_comment'] = r.get('action_comment', '') + ' Source non trouvée pour action différée.'
                r['action'] = 'error_action_deferred'
                r['status'] = 'error'
                continue

            action_done = False
            original_reason = r['rejected_reason']
            if delete_rejected_flag:
                try:
                    os.remove(current_path)
                    _log('logic_info_prefix', text=f'Fichier supprimé (Starcount différé): {rel_path}')
                    r['path'] = None
                    r['action'] = 'deleted_starcount'
                    r['rejected_reason'] = 'starcount'
                    r['status'] = 'processed_action'
                    actions_count += 1
                    action_done = True
                except Exception as del_e:
                    _log('logic_error_prefix', text=f'Erreur suppression Starcount différé {rel_path}: {del_e}')
                    r['action_comment'] = r.get('action_comment', '') + f' Erreur suppression différée: {del_e}'
                    r['action'] = 'error_delete'
                    r['rejected_reason'] = original_reason
            elif move_rejected_flag and sc_reject_abs_path:
                if not os.path.isdir(sc_reject_abs_path):
                    try:
                        os.makedirs(sc_reject_abs_path)
                        _log('logic_dir_created', path=sc_reject_abs_path)
                    except OSError as e_mkdir:
                        _log('logic_dir_create_error', path=sc_reject_abs_path, e=e_mkdir)
                        r['action_comment'] = r.get('action_comment', '') + f' Dossier rejet Starcount inaccessible: {e_mkdir}'
                        r['action'] = 'error_move'
                        r['rejected_reason'] = original_reason
                        continue

                dest_path = os.path.join(sc_reject_abs_path, os.path.basename(current_path))
                try:
                    if os.path.normpath(current_path) != os.path.normpath(dest_path):
                        shutil.move(current_path, dest_path)
                        _log('logic_moved_info', folder=os.path.basename(sc_reject_abs_path), text_key_suffix='_deferred_starcount', file_rel_path=rel_path)
                        r['path'] = dest_path
                        r['action'] = 'moved_starcount'
                        r['rejected_reason'] = 'starcount'
                        r['status'] = 'processed_action'
                        actions_count += 1
                        action_done = True
                    else:
                        r['action_comment'] = r.get('action_comment', '') + ' Déjà dans dossier cible (différé)?'
                        r['action'] = 'kept'
                        r['rejected_reason'] = 'starcount'
                        r['status'] = 'processed_action'
                        action_done = True
                except Exception as move_e:
                    _log('logic_move_error', file=rel_path, e=move_e)
                    r['action_comment'] = r.get('action_comment', '') + f' Erreur déplacement différé: {move_e}'
                    r['action'] = 'error_move'
                    r['rejected_reason'] = original_reason

            if not action_done and not delete_rejected_flag and not move_rejected_flag:
                r['action'] = 'kept_pending_starcount_no_action'
                r['rejected_reason'] = 'starcount'
                r['status'] = 'processed_action'
                r['action_comment'] = r.get('action_comment', '') + ' Action Starcount différée mais aucune opération configurée.'

        _progress(100)
        _status('status_custom', text=f'{actions_count} actions Starcount différées appliquées.')
        _log('logic_info_prefix', text=f'{actions_count} actions Starcount différées appliquées.')
        return actions_count

    analyse_logic.apply_pending_starcount_actions = apply_pending_starcount_actions

# Ajout fonction apply_pending_fwhm_actions si absente dans analyse_logic
if not hasattr(analyse_logic, 'apply_pending_fwhm_actions'):
    def apply_pending_fwhm_actions(results_list, fwhm_reject_path,
                                   delete_rejected_flag, move_rejected_flag,
                                   log_callback, status_callback, progress_callback,
                                   input_dir_abs):
        actions_count = 0
        if not results_list:
            return actions_count

        _log = log_callback if callable(log_callback) else lambda k, **kw: None
        _status = status_callback if callable(status_callback) else lambda k, **kw: None
        _progress = progress_callback if callable(progress_callback) else lambda v: None

        to_process = [r for r in results_list if r.get('rejected_reason') == 'high_fwhm_pending_action' and r.get('status') == 'ok']
        total = len(to_process)
        if total == 0:
            _log('logic_info_prefix', text='Aucune action FWHM en attente.')
            return 0

        _status('status_custom', text=f'Application des actions FWHM différées sur {total} fichiers...')
        _progress(0)

        for i, r in enumerate(to_process):
            current_progress = ((i + 1) / total) * 100
            _progress(current_progress)
            try:
                rel_path = os.path.relpath(r.get('path'), input_dir_abs) if r.get('path') and input_dir_abs else r.get('file', 'N/A')
            except ValueError:
                rel_path = r.get('file', 'N/A')

            _status('status_custom', text=f'Action FWHM sur {rel_path} ({i+1}/{total})')

            current_path = r.get('path')
            if not current_path or not os.path.exists(current_path):
                _log('logic_move_skipped', file=rel_path, e='Fichier source introuvable pour action FWHM différée.')
                r['action_comment'] = r.get('action_comment', '') + ' Source non trouvée pour action différée.'
                r['action'] = 'error_action_deferred'
                r['status'] = 'error'
                continue

            action_done = False
            original_reason = r['rejected_reason']
            if delete_rejected_flag:
                try:
                    os.remove(current_path)
                    _log('logic_info_prefix', text=f'Fichier supprimé (FWHM différé): {rel_path}')
                    r['path'] = None
                    r['action'] = 'deleted_fwhm'
                    r['rejected_reason'] = 'high_fwhm'
                    r['status'] = 'processed_action'
                    actions_count += 1
                    action_done = True
                except Exception as del_e:
                    _log('logic_error_prefix', text=f'Erreur suppression FWHM différé {rel_path}: {del_e}')
                    r['action_comment'] = r.get('action_comment', '') + f' Erreur suppression différée: {del_e}'
                    r['action'] = 'error_delete'
                    r['rejected_reason'] = original_reason
            elif move_rejected_flag and fwhm_reject_path:
                if not os.path.isdir(fwhm_reject_path):
                    try:
                        os.makedirs(fwhm_reject_path)
                        _log('logic_dir_created', path=fwhm_reject_path)
                    except OSError as e_mkdir:
                        _log('logic_dir_create_error', path=fwhm_reject_path, e=e_mkdir)
                        r['action_comment'] = r.get('action_comment', '') + f' Dossier rejet FWHM inaccessible: {e_mkdir}'
                        r['action'] = 'error_move'
                        r['rejected_reason'] = original_reason
                        continue

                dest_path = os.path.join(fwhm_reject_path, os.path.basename(current_path))
                try:
                    if os.path.normpath(current_path) != os.path.normpath(dest_path):
                        shutil.move(current_path, dest_path)
                        _log('logic_moved_info', folder=os.path.basename(fwhm_reject_path), text_key_suffix='_deferred_fwhm', file_rel_path=rel_path)
                        r['path'] = dest_path
                        r['action'] = 'moved_fwhm'
                        r['rejected_reason'] = 'high_fwhm'
                        r['status'] = 'processed_action'
                        actions_count += 1
                        action_done = True
                    else:
                        r['action_comment'] = r.get('action_comment', '') + ' Déjà dans dossier cible (différé)?'
                        r['action'] = 'kept'
                        r['rejected_reason'] = 'high_fwhm'
                        r['status'] = 'processed_action'
                        action_done = True
                except Exception as move_e:
                    _log('logic_move_error', file=rel_path, e=move_e)
                    r['action_comment'] = r.get('action_comment', '') + f' Erreur déplacement différé: {move_e}'
                    r['action'] = 'error_move'
                    r['rejected_reason'] = original_reason

            if not action_done and not delete_rejected_flag and not move_rejected_flag:
                r['action'] = 'kept_pending_fwhm_no_action'
                r['rejected_reason'] = 'high_fwhm'
                r['status'] = 'processed_action'
                r['action_comment'] = r.get('action_comment', '') + ' Action FWHM différée mais aucune opération configurée.'

        _progress(100)
        _status('status_custom', text=f'{actions_count} actions FWHM différées appliquées.')
        _log('logic_info_prefix', text=f'{actions_count} actions FWHM différées appliquées.')
        return actions_count

    analyse_logic.apply_pending_fwhm_actions = apply_pending_fwhm_actions

# Ajout fonction apply_pending_ecc_actions si absente dans analyse_logic
if not hasattr(analyse_logic, 'apply_pending_ecc_actions'):
    def apply_pending_ecc_actions(results_list, ecc_reject_path,
                                  delete_rejected_flag, move_rejected_flag,
                                  log_callback, status_callback, progress_callback,
                                  input_dir_abs):
        actions_count = 0
        if not results_list:
            return actions_count

        _log = log_callback if callable(log_callback) else lambda k, **kw: None
        _status = status_callback if callable(status_callback) else lambda k, **kw: None
        _progress = progress_callback if callable(progress_callback) else lambda v: None

        to_process = [r for r in results_list if r.get('rejected_reason') == 'high_ecc_pending_action' and r.get('status') == 'ok']
        total = len(to_process)
        if total == 0:
            _log('logic_info_prefix', text='Aucune action Excentricité en attente.')
            return 0

        _status('status_custom', text=f'Application des actions Excentricité différées sur {total} fichiers...')
        _progress(0)

        for i, r in enumerate(to_process):
            current_progress = ((i + 1) / total) * 100
            _progress(current_progress)
            try:
                rel_path = os.path.relpath(r.get('path'), input_dir_abs) if r.get('path') and input_dir_abs else r.get('file', 'N/A')
            except ValueError:
                rel_path = r.get('file', 'N/A')

            _status('status_custom', text=f'Action Excentricité sur {rel_path} ({i+1}/{total})')

            current_path = r.get('path')
            if not current_path or not os.path.exists(current_path):
                _log('logic_move_skipped', file=rel_path, e='Fichier source introuvable pour action Excentricité différée.')
                r['action_comment'] = r.get('action_comment', '') + ' Source non trouvée pour action différée.'
                r['action'] = 'error_action_deferred'
                r['status'] = 'error'
                continue

            action_done = False
            original_reason = r['rejected_reason']
            if delete_rejected_flag:
                try:
                    os.remove(current_path)
                    _log('logic_info_prefix', text=f'Fichier supprimé (Excentricité différé): {rel_path}')
                    r['path'] = None
                    r['action'] = 'deleted_ecc'
                    r['rejected_reason'] = 'high_ecc'
                    r['status'] = 'processed_action'
                    actions_count += 1
                    action_done = True
                except Exception as del_e:
                    _log('logic_error_prefix', text=f'Erreur suppression Excentricité différé {rel_path}: {del_e}')
                    r['action_comment'] = r.get('action_comment', '') + f' Erreur suppression différée: {del_e}'
                    r['action'] = 'error_delete'
                    r['rejected_reason'] = original_reason
            elif move_rejected_flag and ecc_reject_path:
                if not os.path.isdir(ecc_reject_path):
                    try:
                        os.makedirs(ecc_reject_path)
                        _log('logic_dir_created', path=ecc_reject_path)
                    except OSError as e_mkdir:
                        _log('logic_dir_create_error', path=ecc_reject_path, e=e_mkdir)
                        r['action_comment'] = r.get('action_comment', '') + f' Dossier rejet Excentricité inaccessible: {e_mkdir}'
                        r['action'] = 'error_move'
                        r['rejected_reason'] = original_reason
                        continue

                dest_path = os.path.join(ecc_reject_path, os.path.basename(current_path))
                try:
                    if os.path.normpath(current_path) != os.path.normpath(dest_path):
                        shutil.move(current_path, dest_path)
                        _log('logic_moved_info', folder=os.path.basename(ecc_reject_path), text_key_suffix='_deferred_ecc', file_rel_path=rel_path)
                        r['path'] = dest_path
                        r['action'] = 'moved_ecc'
                        r['rejected_reason'] = 'high_ecc'
                        r['status'] = 'processed_action'
                        actions_count += 1
                        action_done = True
                    else:
                        r['action_comment'] = r.get('action_comment', '') + ' Déjà dans dossier cible (différé)?'
                        r['action'] = 'kept'
                        r['rejected_reason'] = 'high_ecc'
                        r['status'] = 'processed_action'
                        action_done = True
                except Exception as move_e:
                    _log('logic_move_error', file=rel_path, e=move_e)
                    r['action_comment'] = r.get('action_comment', '') + f' Erreur déplacement différé: {move_e}'
                    r['action'] = 'error_move'
                    r['rejected_reason'] = original_reason

            if not action_done and not delete_rejected_flag and not move_rejected_flag:
                r['action'] = 'kept_pending_ecc_no_action'
                r['rejected_reason'] = 'high_ecc'
                r['status'] = 'processed_action'
                r['action_comment'] = r.get('action_comment', '') + ' Action Excentricité différée mais aucune opération configurée.'

        _progress(100)
        _status('status_custom', text=f'{actions_count} actions Excentricité différées appliquées.')
        _log('logic_info_prefix', text=f'{actions_count} actions Excentricité différées appliquées.')
        return actions_count

    analyse_logic.apply_pending_ecc_actions = apply_pending_ecc_actions

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
    def __init__(self, root, command_file_path=None, main_app_callback=None,
                 initial_lang='fr', lock_language=None):
        """
        Initialise l'interface graphique.

        Args:
            root (tk.Tk or tk.Toplevel): La fenêtre racine ou Toplevel pour cette interface.
            command_file_path (str, optional): Chemin vers le fichier à utiliser
                                               pour la communication avec le GUI principal.
            main_app_callback (callable, optional): Fonction à appeler lors de la fermeture (Retour).
            initial_lang (str): Langue initiale de l'interface.
            lock_language (bool or None):
                Si True, la sélection de langue sera désactivée.
                Si None, le choix est déterminé automatiquement.
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

        self.config_path = os.path.join(os.path.expanduser('~'), 'zeanalyser_gui_config.json')
        self._config_data = self._load_gui_config()

        # Variables Tkinter pour lier les widgets aux données
        self.current_lang = tk.StringVar(value=initial_lang)
        if lock_language is None:
            lock_language = _EMBEDDED_IN_STACKER
        self.lock_language = lock_language
        self.current_lang.trace_add('write', self.change_language)

        self.input_dir = tk.StringVar() 
        self.output_log = tk.StringVar() 
        
        self.output_log.trace_add('write', lambda *args: self._update_log_and_vis_buttons_state())
        
        self.status_text = tk.StringVar() 
        self.progress_var = tk.DoubleVar(value=0.0) 

        # Options d'analyse (Booléens)
        self.analyze_snr = tk.BooleanVar(value=True)
        # By default, do not enable trail detection even if the capability is
        # available. Users can activate it manually.
        self.detect_trails = tk.BooleanVar(value=False)
        self.sort_by_snr = tk.BooleanVar(value=True)
        self.include_subfolders = tk.BooleanVar(value=False)
        self.bortle_path = tk.StringVar(value=self._config_data.get('bortle_path', ''))
        self.use_bortle = tk.BooleanVar(value=self._config_data.get('use_bortle', False))

        # Paramètres Sélection SNR
        self.snr_selection_mode = tk.StringVar(value='percent') 
        self.snr_selection_value = tk.StringVar(value='80')
        self.snr_reject_dir = tk.StringVar()
        self.starcount_reject_dir = tk.StringVar()

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
        self.best_reference_path = None
        self.tooltips = {}
        self.timer_running = False 
        self.timer_start_time = None 
        self.timer_job_id = None
        self.base_status_message = ""
        self.progress_start_time = None
        self.has_pending_snr_actions = False
        self.current_snr_min = None
        self.current_snr_max = None
        self.snr_range_slider = None
        self.snr_slider_lines = ()
        self.current_sc_min = None
        self.current_sc_max = None
        self.starcount_range_slider = None
        self.current_fwhm_min = None
        self.current_fwhm_max = None
        self.fwhm_range_slider = None
        self.current_ecc_min = None
        self.current_ecc_max = None
        self.ecc_range_slider = None

        # Recommended images and associated thresholds
        self.recommended_images = []
        self.reco_snr_min = None
        self.reco_fwhm_max = None
        self.reco_ecc_max = None
        
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
        self.visual_apply_button = None
        self.apply_starcount_button = None
        self.apply_fwhm_button = None
        self.apply_ecc_button = None
        self.apply_reco_button = None
        self.visual_apply_reco_button = None
        self.organize_button = None
        self.stack_plan_button = None
        self.latest_stack_plan_path = None
        
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
                    if self.analysis_results:
                        (self.recommended_images,
                         self.reco_snr_min,
                         self.reco_fwhm_max,
                         self.reco_ecc_max) = analyse_logic.build_recommended_images(self.analysis_results)
                    else:
                        self.recommended_images = []
                        self.reco_snr_min = self.reco_fwhm_max = self.reco_ecc_max = None
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
        if hasattr(self, 'analyze_button') and self.analyze_button:
            self.analyze_button.config(state='disabled')
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
        options['bortle_path'] = self.bortle_path.get()
        options['use_bortle'] = self.use_bortle.get()

        options['apply_snr_action_immediately'] = False
        options['apply_trail_action_immediately'] = False

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

        if hasattr(self, 'stack_plan_button') and self.stack_plan_button:
            state = tk.NORMAL if can_visualize else tk.DISABLED
            self._set_widget_state(self.stack_plan_button, state)


    def _load_gui_config(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_gui_config(self):
        data = {
            'bortle_path': self.bortle_path.get(),
            'use_bortle': self.use_bortle.get()
        }
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass


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
        options['bortle_path'] = self.bortle_path.get()
        options['use_bortle'] = self.use_bortle.get()

        # --- NOUVEAU : Définir les options pour les actions différées ---
        options['apply_snr_action_immediately'] = False  # Toujours différé depuis le GUI
        options['apply_trail_action_immediately'] = False
        # --- FIN NOUVEAU ---

        if options.get('use_bortle'):
            bp = options.get('bortle_path', '')
            if not bp or not bp.lower().endswith(('.tif', '.tiff', '.kmz')):
                messagebox.showwarning(
                    self._("msg_warning"),
                    "Sélectionnez un fichier Bortle valide (GeoTIFF ou KMZ) avant de lancer l\u2019analyse.",
                    parent=self.root
                )
                return False

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
            vis_window.state('zoomed')
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
                valid_snrs = [r['snr'] for r in self.analysis_results if r.get('status')=='ok' and 'snr' in r and is_finite_number(r['snr'])]
                if valid_snrs:
                    hist = ax1.hist(valid_snrs, bins=20, color='skyblue', edgecolor='black')
                    ax1.set_title(self._("visu_snr_dist_title"))
                    ax1.set_xlabel(self._("visu_snr_dist_xlabel"))
                    ax1.set_ylabel(self._("visu_snr_dist_ylabel"))
                    ax1.grid(axis='y', linestyle='--', alpha=0.7)

                    min_snr, max_snr = min(valid_snrs), max(valid_snrs)
                    self.current_snr_min = min_snr
                    self.current_snr_max = max_snr
                    ax1.set_xlim(min_snr, max_snr)
                    line_lo = ax1.axvline(min_snr, color='red', linestyle='--')
                    line_hi = ax1.axvline(max_snr, color='red', linestyle='--')

                    fig1.subplots_adjust(bottom=0.25)
                    slider_ax = fig1.add_axes([0.15, 0.1, 0.7, 0.05])
                    self.snr_range_slider = RangeSlider(slider_ax, "SNR", min_snr, max_snr, valinit=(min_snr, max_snr))
                    self.snr_slider_lines = (line_lo, line_hi)

                    def _on_slider_change(val):
                        lo, hi = val
                        line_lo.set_xdata([lo, lo])
                        line_hi.set_xdata([hi, hi])
                        fig1.canvas.draw_idle()
                        self.current_snr_min = lo
                        self.current_snr_max = hi
                        if self.apply_snr_button:
                            self.apply_snr_button.config(state=tk.NORMAL)
                        if self.visual_apply_button:
                            self.visual_apply_button.config(state=tk.NORMAL)

                    self.snr_range_slider.on_changed(_on_slider_change)
                else:
                    ax1.text(0.5, 0.5, self._("visu_snr_dist_no_data"), ha='center', va='center', fontsize=12, color='red')
                canvas1 = FigureCanvasTkAgg(fig1, master=snr_tab)
                canvas1.draw()
                canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                canvas_list.append(canvas1)
            except Exception as e:
                print(f"Erreur Histogramme SNR: {e}"); ttk.Label(snr_tab, text=f"{self._('msg_error')}:\n{e}\n{traceback.format_exc()}").pack()
                if fig1: plt.close(fig1) # Fermer figure si erreur canvas

            # --- Onglet Starcount Distribution ---
            sc_tab = ttk.Frame(notebook)
            notebook.add(sc_tab, text=self._('starcount_distribution_tab'))
            fig_sc = None
            try:
                sc_values = [r['starcount'] for r in self.analysis_results if r.get('starcount') is not None]
                fig_sc, ax_sc = plt.subplots(figsize=(7, 5))
                figures_list.append(fig_sc)
                if sc_values:
                    n, bins, patches = ax_sc.hist(sc_values, bins=20, color='skyblue', edgecolor='black')
                    ax_sc.set_title(self._('starcount_distribution_title'))
                    ax_sc.set_xlabel(self._('starcount_label'))
                    ax_sc.set_ylabel(self._('number_of_images'))
                    ax_sc.grid(axis='y', linestyle='--', alpha=0.7)

                    self.current_sc_min = min(sc_values)
                    self.current_sc_max = max(sc_values)

                    ax_slider_sc = fig_sc.add_axes([0.15, 0.02, 0.7, 0.03])
                    self.starcount_range_slider = RangeSlider(
                        ax_slider_sc,
                        self._('starcount_label'),
                        valmin=min(sc_values),
                        valmax=max(sc_values),
                        valinit=(min(sc_values), max(sc_values))
                    )
                    self.starcount_range_slider.on_changed(lambda val: self._on_starcount_slider_change(val, patches))
                else:
                    ax_sc.text(0.5, 0.5, self._('visu_snr_dist_no_data'), ha='center', va='center', fontsize=12, color='red')

                canvas_sc = FigureCanvasTkAgg(fig_sc, master=sc_tab)
                canvas_sc.draw()
                canvas_sc.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                canvas_list.append(canvas_sc)
            except Exception as e:
                print(f"Erreur Histogramme Starcount: {e}")
                ttk.Label(sc_tab, text=f"{self._('msg_error')}:\n{e}\n{traceback.format_exc()}").pack()
                if fig_sc:
                    plt.close(fig_sc)

            # --- Onglet Distribution FWHM ---
            fwhm_tab = ttk.Frame(notebook)
            notebook.add(fwhm_tab, text=self._('visu_tab_fwhm_dist'))
            fig_fwhm = None
            try:
                fwhm_values = [r['fwhm'] for r in self.analysis_results if is_finite_number(r.get('fwhm', np.nan))]
                fig_fwhm, ax_fwhm = plt.subplots(figsize=(7, 5))
                figures_list.append(fig_fwhm)
                if fwhm_values:
                    ax_fwhm.hist(fwhm_values, bins=20, color='skyblue', edgecolor='black')
                    ax_fwhm.set_title(self._('fwhm_distribution_title'))
                    ax_fwhm.set_xlabel('FWHM')
                    ax_fwhm.set_ylabel(self._('number_of_images'))
                    ax_fwhm.grid(axis='y', linestyle='--', alpha=0.7)

                    self.current_fwhm_min = min(fwhm_values)
                    self.current_fwhm_max = max(fwhm_values)

                    ax_slider_fwhm = fig_fwhm.add_axes([0.15, 0.02, 0.7, 0.03])
                    self.fwhm_range_slider = RangeSlider(
                        ax_slider_fwhm,
                        self._('filter_fwhm'),
                        valmin=min(fwhm_values),
                        valmax=max(fwhm_values),
                        valinit=(min(fwhm_values), max(fwhm_values))
                    )
                    self.fwhm_range_slider.on_changed(
                        lambda val: self._on_fwhm_slider_change(val))
                else:
                    ax_fwhm.text(0.5, 0.5, self._('visu_fwhm_no_data'), ha='center', va='center', fontsize=12, color='red')

                canvas_fwhm = FigureCanvasTkAgg(fig_fwhm, master=fwhm_tab)
                canvas_fwhm.draw()
                canvas_fwhm.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                canvas_list.append(canvas_fwhm)
            except Exception as e:
                print(f"Erreur Histogramme FWHM: {e}")
                ttk.Label(fwhm_tab, text=f"{self._('msg_error')}:\n{e}\n{traceback.format_exc()}").pack()
                if fig_fwhm:
                    plt.close(fig_fwhm)

            # --- Onglet Distribution Excentricité ---
            ecc_tab = ttk.Frame(notebook)
            notebook.add(ecc_tab, text=self._('visu_tab_ecc_dist'))
            fig_ecc = None
            try:
                ecc_values = [r['ecc'] for r in self.analysis_results if is_finite_number(r.get('ecc', np.nan))]
                fig_ecc, ax_ecc = plt.subplots(figsize=(7, 5))
                figures_list.append(fig_ecc)
                if ecc_values:
                    ax_ecc.hist(ecc_values, bins=20, range=(0,1), color='skyblue', edgecolor='black')
                    ax_ecc.set_title(self._('ecc_distribution_title'))
                    ax_ecc.set_xlabel('e')
                    ax_ecc.set_ylabel(self._('number_of_images'))
                    ax_ecc.grid(axis='y', linestyle='--', alpha=0.7)

                    self.current_ecc_min = min(ecc_values)
                    self.current_ecc_max = max(ecc_values)

                    ax_slider_ecc = fig_ecc.add_axes([0.15, 0.02, 0.7, 0.03])
                    self.ecc_range_slider = RangeSlider(
                        ax_slider_ecc,
                        self._('filter_ecc'),
                        valmin=0.0,
                        valmax=1.0,
                        valinit=(min(ecc_values), max(ecc_values))
                    )
                    self.ecc_range_slider.on_changed(
                        lambda val: self._on_ecc_slider_change(val))
                else:
                    ax_ecc.text(0.5, 0.5, self._('visu_ecc_no_data'), ha='center', va='center', fontsize=12, color='red')

                canvas_ecc = FigureCanvasTkAgg(fig_ecc, master=ecc_tab)
                canvas_ecc.draw()
                canvas_ecc.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                canvas_list.append(canvas_ecc)
            except Exception as e:
                print(f"Erreur Histogramme Eccentricite: {e}")
                ttk.Label(ecc_tab, text=f"{self._('msg_error')}:\n{e}\n{traceback.format_exc()}").pack()
                if fig_ecc:
                    plt.close(fig_ecc)

            # --- Onglet FWHM vs Excentricité ---
            scatter_tab = ttk.Frame(notebook)
            notebook.add(scatter_tab, text='FWHM vs e')
            fig_scatter = None
            try:
                valid_pairs = [
                    (r['fwhm'], r['ecc'])
                    for r in self.analysis_results
                    if is_finite_number(r.get('fwhm', np.nan)) and is_finite_number(r.get('ecc', np.nan))
                ]
                fig_scatter, ax_scatt = plt.subplots(figsize=(7,5))
                figures_list.append(fig_scatter)
                if valid_pairs:
                    fwhm_vals, ecc_vals = zip(*valid_pairs)
                    ax_scatt.scatter(fwhm_vals, ecc_vals, alpha=0.6)
                    ax_scatt.set_xlabel('FWHM')
                    ax_scatt.set_ylabel('e')
                    ax_scatt.set_title('FWHM vs e')
                    ax_scatt.grid(True, linestyle='--', alpha=0.7)
                else:
                    ax_scatt.text(0.5,0.5,self._('visu_snr_dist_no_data'),ha='center',va='center',fontsize=12,color='red')
                canvas_scatter = FigureCanvasTkAgg(fig_scatter, master=scatter_tab)
                canvas_scatter.draw()
                canvas_scatter.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                canvas_list.append(canvas_scatter)
            except Exception as e:
                print(f"Erreur Scatter FWHM vs Ecc: {e}")
                ttk.Label(scatter_tab, text=f"{self._('msg_error')}:\n{e}\n{traceback.format_exc()}").pack()
                if fig_scatter:
                    plt.close(fig_scatter)

            # --- Onglet Comparaison SNR (Top/Bottom N) ---
            comp_tab = ttk.Frame(notebook)
            notebook.add(comp_tab, text=self._("visu_tab_snr_comp"))
            fig2 = None
            try:
                valid_res = [
                    r
                    for r in self.analysis_results
                    if r.get('status') == 'ok'
                    and 'snr' in r
                    and is_finite_number(r['snr'])
                    and 'file' in r
                ]
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
                display_res = (
                    sorted(
                        self.analysis_results,
                        key=lambda x: x.get('snr', -np.inf)
                        if is_finite_number(x.get('snr', -np.inf))
                        else -np.inf,
                        reverse=True,
                    )
                    if self.sort_by_snr.get()
                    else self.analysis_results
                )
                for r in display_res:
                    status = r.get('status','?'); vals = []
                    for col_id in cols:
                        if col_id == 'file': vals.append(r.get('rel_path', os.path.basename(r.get('file','?'))));
                        elif col_id == 'status': vals.append(status)
                        elif col_id == 'snr':
                            vals.append(
                                f"{r.get('snr', 0.0):.2f}"
                                if is_finite_number(r.get('snr', np.nan))
                                else "N/A"
                            )
                        elif col_id == 'bg':
                            vals.append(
                                f"{r.get('sky_bg', 0.0):.2f}"
                                if is_finite_number(r.get('sky_bg', np.nan))
                                else "N/A"
                            )
                        elif col_id == 'noise':
                            vals.append(
                                f"{r.get('sky_noise', 0.0):.2f}"
                                if is_finite_number(r.get('sky_noise', np.nan))
                                else "N/A"
                            )
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
                valid_kept_results = [
                    r for r in self.analysis_results
                    if r.get('status') == 'ok'
                    and r.get('action') == 'kept'
                    and r.get('rejected_reason') is None
                    and 'snr' in r and is_finite_number(r['snr'])
                ]
                valid_kept_snrs = [r['snr'] for r in valid_kept_results]
                sc_vals = [
                    r['starcount']
                    for r in valid_kept_results
                    if r.get('starcount') is not None and is_finite_number(r['starcount'])
                ]
                fwhm_vals = [
                    r['fwhm']
                    for r in valid_kept_results
                    if is_finite_number(r.get('fwhm', np.nan))
                ]
                ecc_vals = [
                    r['ecc']
                    for r in valid_kept_results
                    if is_finite_number(r.get('ecc', np.nan))
                ]

                if len(valid_kept_snrs) >= 5 and len(fwhm_vals) >= 5 and len(ecc_vals) >= 5:
                    snr_p25 = np.percentile(valid_kept_snrs, 25)
                    fwhm_p75 = np.percentile(fwhm_vals, 75)
                    ecc_p75 = np.percentile(ecc_vals, 75)
                    good_img = [
                        r
                        for r in valid_kept_results
                        if r['snr'] >= snr_p25
                        and (
                            r.get('fwhm')
                            if is_finite_number(r.get('fwhm'))
                            else np.inf
                        )
                        <= fwhm_p75
                        and (
                            r.get('ecc')
                            if is_finite_number(r.get('ecc'))
                            else np.inf
                        )
                        <= ecc_p75
                    ]
                    ttk.Label(
                        recom_frame,
                        text=self._(
                            "visu_recom_text_all",
                            count=len(good_img),
                        ),
                    ).pack(anchor=tk.W, pady=(0, 5))
                    rec_cols = ("file", "snr", "fwhm", "ecc")
                    rec_tree = ttk.Treeview(
                        recom_frame, columns=rec_cols, show="headings", height=10
                    )
                    rec_tree.heading("file", text=self._("visu_recom_col_file"))
                    rec_tree.heading("snr", text=self._("visu_recom_col_snr"))
                    rec_tree.heading("fwhm", text="FWHM")
                    rec_tree.heading("ecc", text="e")
                    rec_tree.column("file", width=450, anchor="w")
                    rec_tree.column("snr", width=80, anchor="center")
                    rec_tree.column("fwhm", width=80, anchor="center")
                    rec_tree.column("ecc", width=80, anchor="center")
                    for img in sorted(good_img, key=lambda x: x['snr'], reverse=True):
                        rec_tree.insert(
                            "",
                            tk.END,
                            values=(
                                img.get("rel_path", os.path.basename(img["file"])),
                                f"{img['snr']:.2f}",
                                f"{img['fwhm']:.2f}",
                                f"{img['ecc']:.2f}",
                            ),
                        )
                    rec_scr = ttk.Scrollbar(
                        recom_frame, orient=tk.VERTICAL, command=rec_tree.yview
                    )
                    rec_tree.configure(yscroll=rec_scr.set)
                    rec_scr.pack(side=tk.RIGHT, fill=tk.Y)
                    rec_tree.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
                    export_cmd = lambda gi=good_img, p=(snr_p25, fwhm_p75, ecc_p75): self.export_recommended_list(gi, p)
                    export_button = ttk.Button(
                        recom_frame, text=self._("export_button"), command=export_cmd
                    )
                    export_button.pack(pady=5)
                elif len(valid_kept_snrs) >= 5 and len(sc_vals) >= 5:
                    snr_p25 = np.percentile(valid_kept_snrs, 25)
                    sc_p25 = np.percentile(sc_vals, 25)
                    good_img = [
                        r for r in valid_kept_results
                        if r['snr'] >= snr_p25 and r['starcount'] >= sc_p25
                    ]

                    ttk.Label(
                        recom_frame,
                        text=self._(
                            "visu_recom_text_both",
                            count=len(good_img),
                            snr_p25=snr_p25,
                            sc_p25=sc_p25,
                        ),
                    ).pack(anchor=tk.W, pady=(0, 5))

                    rec_cols = ("file", "snr", "starcount")
                    rec_tree = ttk.Treeview(
                        recom_frame, columns=rec_cols, show="headings", height=10
                    )
                    rec_tree.heading("file", text=self._("visu_recom_col_file"))
                    rec_tree.heading("snr", text=self._("visu_recom_col_snr"))
                    rec_tree.heading(
                        "starcount", text=self._("visu_recom_col_starcount")
                    )
                    rec_tree.column("file", width=450, anchor="w")
                    rec_tree.column("snr", width=80, anchor="center")
                    rec_tree.column("starcount", width=90, anchor="center")

                    for img in sorted(good_img, key=lambda x: x['snr'], reverse=True):
                        rec_tree.insert(
                            "",
                            tk.END,
                            values=(
                                img.get("rel_path", os.path.basename(img["file"])),
                                f"{img['snr']:.2f}",
                                f"{img['starcount']:.0f}",
                            ),
                        )

                    rec_scr = ttk.Scrollbar(
                        recom_frame, orient=tk.VERTICAL, command=rec_tree.yview
                    )
                    rec_tree.configure(yscroll=rec_scr.set)
                    rec_scr.pack(side=tk.RIGHT, fill=tk.Y)
                    rec_tree.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

                    export_cmd = lambda gi=good_img, p=(snr_p25, sc_p25): self.export_recommended_list(gi, p)
                    export_button = ttk.Button(
                        recom_frame, text=self._("export_button"), command=export_cmd
                    )
                    export_button.pack(pady=5)

                elif len(valid_kept_snrs) >= 5:
                    p25_threshold = np.percentile(valid_kept_snrs, 25)
                    good_img = [r for r in valid_kept_results if r['snr'] >= p25_threshold]
                    if good_img:
                        good_img_sorted = sorted(good_img, key=lambda x: x['snr'], reverse=True)
                        ttk.Label(
                            recom_frame,
                            text=self._(
                                "visu_recom_text",
                                count=len(good_img_sorted),
                                p75=p25_threshold,
                            ),
                        ).pack(anchor=tk.W, pady=(0, 5))

                        rec_cols = ("file", "snr")
                        rec_tree = ttk.Treeview(
                            recom_frame, columns=rec_cols, show="headings", height=10
                        )
                        rec_tree.heading("file", text=self._("visu_recom_col_file"))
                        rec_tree.heading("snr", text=self._("visu_recom_col_snr"))
                        rec_tree.column("file", width=450, anchor="w")
                        rec_tree.column("snr", width=100, anchor="center")

                        for img in good_img_sorted:
                            rec_tree.insert(
                                "",
                                tk.END,
                                values=(
                                    img.get(
                                        "rel_path", os.path.basename(img.get("file", "?"))
                                    ),
                                    f"{img.get('snr', 0.0):.2f}",
                                ),
                            )

                        rec_scr = ttk.Scrollbar(
                            recom_frame, orient=tk.VERTICAL, command=rec_tree.yview
                        )
                        rec_tree.configure(yscroll=rec_scr.set)
                        rec_scr.pack(side=tk.RIGHT, fill=tk.Y)
                        rec_tree.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

                        export_cmd = lambda gi=good_img_sorted, p=p25_threshold: self.export_recommended_list(gi, p)
                        export_button = ttk.Button(
                            recom_frame,
                            text=self._("export_button"),
                            command=export_cmd,
                        )
                        export_button.pack(pady=5)
                    else:
                        ttk.Label(
                            recom_frame, text=self._("visu_recom_no_selection")
                        ).pack(padx=10, pady=10)

                elif len(valid_kept_snrs) > 0:
                    ttk.Label(
                        recom_frame, text=self._("visu_recom_not_enough")
                    ).pack(padx=10, pady=10)

                    export_all_kept_cmd = lambda gi=valid_kept_results: self.export_recommended_list(gi, -1)
                    export_all_button = ttk.Button(
                        recom_frame,
                        text=self._("Exporter Toutes Conservées", default="Export All Kept"),
                        command=export_all_kept_cmd,
                    )
                    export_all_button.pack(pady=5)

                else:
                    ttk.Label(
                        recom_frame, text=self._("visu_recom_no_data")
                    ).pack(padx=10, pady=10)
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

            # --- Action buttons at the bottom of the window ---
            top = vis_window  # existing toplevel window for the visualization

            bottom_frame = ttk.Frame(top)
            bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=5)

            # Apply Starcount Rejection button
            self.apply_starcount_button = ttk.Button(
                bottom_frame,
                text=self._('apply_starcount_rejection', default='Apply Starcount Rejection'),
                state=tk.DISABLED,
                command=self._on_visual_apply_starcount
            )
            self.apply_starcount_button.pack(side=tk.RIGHT, padx=5)
            self.tooltips['apply_starcount_button'] = ToolTip(
                self.apply_starcount_button,
                lambda: self._('tooltip_apply_starcount_rejection', default='Apply pending starcount actions')
            )

            # Apply FWHM filter button
            self.apply_fwhm_button = ttk.Button(
                bottom_frame,
                text=self._('filter_fwhm', default='Filter FWHM'),
                state=tk.DISABLED,
                command=self._on_visual_apply_fwhm
            )
            self.apply_fwhm_button.pack(side=tk.RIGHT, padx=5)

            # Apply eccentricity filter button
            self.apply_ecc_button = ttk.Button(
                bottom_frame,
                text=self._('filter_ecc', default='Filter Eccentricity'),
                state=tk.DISABLED,
                command=self._on_visual_apply_ecc
            )
            self.apply_ecc_button.pack(side=tk.RIGHT, padx=5)

            # cloned Apply SNR Rejection button
            self.visual_apply_button = ttk.Button(
                bottom_frame,
                text=self._("visual_apply_snr_button", default="Apply SNR Rejection"),
                state=tk.DISABLED,
                command=self._on_visual_apply_snr
            )
            self.visual_apply_button.pack(side=tk.RIGHT, padx=5)
            self.tooltips['visual_apply_button'] = ToolTip(
                self.visual_apply_button,
                lambda: self._('tooltip_apply_snr_rejection', default='Apply pending SNR actions')
            )

            self.visual_apply_reco_button = ttk.Button(
                bottom_frame,
                text=self._('visual_apply_reco_button'),
                width=30,
                state=tk.DISABLED,
                command=self._apply_recommendations_gui
            )
            self.visual_apply_reco_button.pack(side=tk.RIGHT, padx=5)
            if self.recommended_images:
                self.visual_apply_reco_button.config(state=tk.NORMAL)

            # existing Close button
            close_button = ttk.Button(
                bottom_frame,
                text=self._("Fermer", default="Close"),
                command=cleanup_vis_window
            )
            close_button.pack(side=tk.RIGHT)
            top.protocol("WM_DELETE_WINDOW", cleanup_vis_window)  # Bind window X button

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
            confirm_msg = self._(
                "marker_confirm_delete_selected",
                default="Supprimer les marqueurs pour les {count} dossiers sélectionnés ?\nCela forcera leur ré-analyse au prochain lancement.",
                count=len(selected_indices),
            )
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
                elif deleted_count > 0:
                    messagebox.showinfo(
                        self._("msg_info"),
                        self._(
                            "marker_delete_selected_success",
                            default="{count} marqueur(s) supprimé(s).",
                            count=deleted_count,
                        ),
                        parent=marker_window,
                    )
####################################################################################################################


        def delete_all_markers():
            """Supprime tous les marqueurs trouvés."""
            abs_paths_to_clear = list(rel_to_abs_map.values())
            if not abs_paths_to_clear:
                messagebox.showinfo(self._("msg_info"), self._("marker_none_found", default="Aucun dossier marqué trouvé."), parent=marker_window)
                return

            # Confirmer suppression totale
            confirm_msg = self._(
                "marker_confirm_delete_all",
                default="Supprimer TOUS les marqueurs ({count}) dans le dossier '{folder}' et ses sous-dossiers analysables ?\nCela forcera une ré-analyse complète.",
                count=len(abs_paths_to_clear),
                folder=os.path.basename(abs_input_dir),
            )
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
                elif deleted_count > 0:
                    messagebox.showinfo(
                        self._("msg_info"),
                        self._(
                            "marker_delete_all_success",
                            default="Tous les {count} marqueur(s) trouvés ont été supprimés.",
                            count=deleted_count,
                        ),
                        parent=marker_window,
                    )



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
        # Sauvegarder la configuration GUI avant de supprimer les variables Tkinter
        self._save_gui_config()
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
                if self.best_reference_path is not None:
                    self.main_app_callback(reference_path=self.best_reference_path)
                else:
                    self.main_app_callback()
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

    def _format_seconds(self, seconds):
        """Formatte un nombre de secondes en HH:MM:SS ou MM:SS."""
        if seconds < 0:
            seconds = 0
        if seconds < 3600:
            return time.strftime('%M:%S', time.gmtime(seconds))
        else:
            return time.strftime('%H:%M:%S', time.gmtime(seconds))

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

        # Ligne 3: Bortle raster
        bortle_label = ttk.Label(config_frame, text="")
        bortle_label.grid(row=3, column=0, sticky=tk.W, pady=2)
        self.widgets_refs['bortle_file_label'] = bortle_label
        self.bortle_entry = ttk.Entry(config_frame, textvariable=self.bortle_path, width=40)
        self.bortle_entry.grid(row=3, column=1, padx=5, pady=2, sticky=tk.W+tk.E)
        self.bortle_browse_button = ttk.Button(config_frame, text="", command=self.browse_bortle_file)
        self.bortle_browse_button.grid(row=3, column=2, padx=5, pady=2)
        # Stocker avec un préfixe "browse_" pour appliquer la traduction
        self.widgets_refs['browse_bortle_button'] = self.bortle_browse_button
        self.use_bortle_check = ttk.Checkbutton(config_frame, text="", variable=self.use_bortle, command=self.toggle_sections_state)
        self.use_bortle_check.grid(row=3, column=3, sticky=tk.W)
        self.widgets_refs['use_bortle_check_label'] = self.use_bortle_check

        # Ligne 4: Bouton organiser fichiers
        self.organize_button = ttk.Button(
            config_frame,
            text=self._('organize_files_button'),
            command=self.organize_files,
            state=tk.DISABLED,
        )
        self.organize_button.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        self.widgets_refs['organize_files_button'] = self.organize_button

        # Ligne 5: Sélection Langue (Aligné à droite)
        if not self.lock_language:
            lang_frame = ttk.Frame(config_frame)
            lang_frame.grid(row=5, column=0, columnspan=3, sticky=tk.E, pady=5, padx=5)
            lang_label = ttk.Label(lang_frame, text="")
            lang_label.pack(side=tk.LEFT, padx=(0, 5))
            self.widgets_refs['lang_label'] = lang_label
            lang_options = sorted([lang for lang in translations.keys()])
            self.lang_combobox = ttk.Combobox(
                lang_frame,
                textvariable=self.current_lang,
                values=lang_options,
                state="readonly",
                width=5,
            )
            self.lang_combobox.pack(side=tk.LEFT)
        else:
            self.lang_combobox = None

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
            text=self._("apply_snr_rejection_button", default="Apply SNR Rejection"),
            command=self.apply_pending_snr_actions_gui,
            state=tk.DISABLED
        )
        self.apply_snr_button.pack(side=tk.RIGHT) # Ou grid si vous préférez
        self.widgets_refs['apply_snr_rejection_button'] = self.apply_snr_button # Pour la traduction du texte
        self.tooltips['apply_snr_button'] = ToolTip(
            self.apply_snr_button,
            lambda: self._('tooltip_apply_snr_rejection', default='Apply pending SNR actions')
        )

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

        self.send_reference_button = ttk.Button(
            button_frame,
            text='',
            command=self.send_reference_to_main,
            width=18,
            state=tk.DISABLED
        )
        self.send_reference_button.pack(side=tk.LEFT, padx=5)
        self.widgets_refs['send_reference_button'] = self.send_reference_button

        self.visualize_button = ttk.Button(button_frame, text="", command=self.visualize_results, width=18)
        self.visualize_button.pack(side=tk.LEFT, padx=5); self.visualize_button.config(state=tk.DISABLED) # Désactivé au début
        self.widgets_refs['visualize_button'] = self.visualize_button # Référencer

        self.open_log_button = ttk.Button(button_frame, text="", command=self.open_log_file, width=18)
        self.open_log_button.pack(side=tk.LEFT, padx=5); self.open_log_button.config(state=tk.DISABLED) # Désactivé au début
        self.widgets_refs['open_log_button'] = self.open_log_button # Référencer

        self.manage_markers_button = ttk.Button(button_frame, text="", command=self.manage_markers, width=18)
        self.manage_markers_button.pack(side=tk.LEFT, padx=5)
        self.widgets_refs['manage_markers_button'] = self.manage_markers_button

        self.stack_plan_button = ttk.Button(
            button_frame,
            text=self._('create_stack_plan_button'),
            command=self.open_stack_plan_window,
            width=18,
            state=tk.DISABLED
        )
        self.stack_plan_button.pack(side=tk.LEFT, padx=5)
        self.widgets_refs['create_stack_plan_button'] = self.stack_plan_button

        self.apply_reco_button = ttk.Button(
            button_frame,
            text=self._('apply_reco_button'),
            command=self._apply_recommendations_gui,
            width=30,
            state=tk.DISABLED
        )
        self.apply_reco_button.pack(side=tk.RIGHT, padx=5)
        self.widgets_refs['apply_reco_button'] = self.apply_reco_button

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
        self.remaining_label = ttk.Label(status_frame, text="")
        self.remaining_label.pack(side=tk.RIGHT, padx=5)
        self.elapsed_label = ttk.Label(status_frame, text="")
        self.elapsed_label.pack(side=tk.RIGHT, padx=5)
        self.widgets_refs['elapsed_label'] = self.elapsed_label
        self.widgets_refs['remaining_label'] = self.remaining_label

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
                elif isinstance(widget, ttk.Button) and key == 'create_stack_plan_button':
                    widget.config(text=self._('create_stack_plan_button'))
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
            if self.send_reference_button: self.send_reference_button.config(text=self._("use_best_reference_button"))
            if self.visualize_button: self.visualize_button.config(text=self._("visualize_button"))
            if self.open_log_button: self.open_log_button.config(text=self._("open_log_button"))
            if self.organize_button: self.organize_button.config(text=self._("organize_files_button"))
            if self.apply_snr_button:
                self.apply_snr_button.config(text=self._("apply_snr_rejection_button"))
            # Texte du bouton Quitter/Retour dépend si un callback est fourni
            if self.return_button:
                btn_text = self._("return_button_text") if self.main_app_callback else self._("quit_button")
                self.return_button.config(text=btn_text)
            if self.apply_reco_button:
                self.apply_reco_button.config(text=self._("apply_reco_button"))
        except tk.TclError as e:
            print(f"WARN: Erreur Tcl mise à jour texte bouton principal: {e}")
        except KeyError as e:
            print(f"WARN: Clé traduction manquante pour bouton principal: {e}")

        # Mettre à jour l'état activé/désactivé des sections
        self.toggle_sections_state()

        if hasattr(self, 'elapsed_label'):
            self.elapsed_label.config(text=f"{self._('elapsed_time_label')} 00:00:00")
        if hasattr(self, 'remaining_label'):
            self.remaining_label.config(text=f"{self._('remaining_time_label')} 00:00:00")

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

    def browse_bortle_file(self):
        """Ouvre un fichier GeoTIFF ou KMZ contenant la carte Bortle."""
        path = filedialog.askopenfilename(
            parent=self.root,
            title=self._('bortle_file_label'),
            filetypes=[('GeoTIFF/KMZ', '*.tif *.tiff *.tpk *.kmz'), (self._('Tous les fichiers'), '*.*')]
        )
        if path:
            if os.path.isdir(path):
                files = [f for f in os.listdir(path) if f.lower().endswith(('.tif', '.tiff', '.tpk', '.kmz'))]
                if len(files) == 1:
                    path = os.path.join(path, files[0])
                elif len(files) > 1:
                    # Simplified selection: take first file
                    path = os.path.join(path, files[0])
            if not path.lower().endswith(('.tif', '.tiff', '.kmz')):
                messagebox.showerror(
                    self._("msg_error"),
                    "Fichier Bortle non pris en charge : choisissez un GeoTIFF (.tif) ou KMZ",
                    parent=self.root
                )
                return
            self.bortle_path.set(path)
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

            bortle_state = tk.NORMAL if self.use_bortle.get() else tk.DISABLED
            self._set_widget_state(self.bortle_entry, bortle_state)
            self._set_widget_state(self.bortle_browse_button, bortle_state)

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
        self.progress_start_time = None
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
        self.best_reference_path = None
        if hasattr(self, 'send_reference_button') and self.send_reference_button:
            self._set_widget_state(self.send_reference_button, tk.DISABLED)
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
                    clamped = max(0.0, min(100.0, float(value)))
                    self.progress_var.set(clamped)
                    if clamped > 0:
                        if self.progress_start_time is None:
                            self.progress_start_time = time.time()
                        elapsed = time.time() - self.progress_start_time
                        if clamped > 0:
                            est_total = elapsed / (clamped / 100.0)
                            remaining = max(0.0, est_total - elapsed)
                        else:
                            remaining = 0.0
                        elapsed_str = self._format_seconds(elapsed)
                        remaining_str = self._format_seconds(remaining)
                        if hasattr(self, 'elapsed_label'):
                            self.elapsed_label.config(text=f"{self._('elapsed_time_label')} {elapsed_str}")
                        if hasattr(self, 'remaining_label'):
                            self.remaining_label.config(text=f"{self._('remaining_time_label')} {remaining_str}")
                    else:
                        self.progress_start_time = None
                    if clamped >= 100:
                        self.progress_start_time = None

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

        # Générer un nom de fichier par défaut
        if isinstance(criterion_value, tuple):
            if len(criterion_value) == 3:
                snr_val, fwhm_val, ecc_val = criterion_value
                default_filename = (
                    f"recommended_images_snr_ge_{snr_val:.2f}_fwhm_le_{fwhm_val:.2f}_ecc_le_{ecc_val:.2f}.txt"
                )
            else:
                snr_val, sc_val = criterion_value
                default_filename = (
                    f"recommended_images_snr_gt_{snr_val:.2f}_sc_gt_{sc_val:.0f}.txt"
                )
        elif criterion_value != -1:
            default_filename = f"recommended_images_snr_gt_{criterion_value:.2f}.txt"
        else:
            default_filename = "all_kept_images.txt"
        
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
                    if criterion_value != -1:
                        if isinstance(criterion_value, tuple):
                            if len(criterion_value) == 3:
                                snr_val, fwhm_val, ecc_val = criterion_value
                                f.write(
                                    f"# {self._('Critère', default='Criterion')}: SNR >= {snr_val:.2f} (P25); FWHM <= {fwhm_val:.2f} (P75); e <= {ecc_val:.2f} (P75)\n"
                                )
                            else:
                                snr_val, sc_val = criterion_value
                                f.write(
                                    f"# {self._('Critère', default='Criterion')}: SNR >= {snr_val:.2f} (P25); Starcount >= {sc_val:.0f} (P25)\n"
                                )
                        else:
                            f.write(
                                f"# {self._('Critère', default='Criterion')}: SNR >= {criterion_value:.2f} (P25)\n"
                            )
                    else:
                        f.write(f"# {self._('Critère', default='Criterion')}: {self._('Toutes les images conservées valides', default='All valid kept images')}\n")
                    f.write(f"# {self._('Généré le', default='Generated on')}: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# {self._('Nombre dimages', default='Number of images')}: {len(images_to_export)}\n\n")

                    header_parts = ["file", "snr"]
                    if any(r.get('fwhm') is not None for r in images_to_export):
                        header_parts.append("fwhm")
                    if any(r.get('ecc') is not None for r in images_to_export):
                        header_parts.append("ecc")
                    if any(r.get('starcount') is not None for r in images_to_export):
                        header_parts.append("starcount")
                    f.write("\t".join(header_parts) + "\n")

                    for img_data in images_to_export:
                        file_to_write = img_data.get('rel_path', os.path.basename(img_data.get('file', 'UNKNOWN_FILE')))
                        line_parts = [file_to_write]
                        if img_data.get('snr') is not None:
                            line_parts.append(f"{img_data['snr']:.2f}")
                        if img_data.get('fwhm') is not None:
                            line_parts.append(f"{img_data['fwhm']:.2f}")
                        if img_data.get('ecc') is not None:
                            line_parts.append(f"{img_data['ecc']:.2f}")
                        if img_data.get('starcount') is not None:
                            line_parts.append(f"{img_data['starcount']:.0f}")
                        f.write("\t".join(line_parts) + "\n")
                
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
        self.analysis_completed_successfully = success
        if self.stack_after_analysis and self.analysis_completed_successfully:
            self.root.after(0, self._auto_stack_workflow)
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
        self.best_reference_path = self._get_best_reference()
        if success and self.analysis_results:
            (self.recommended_images,
             self.reco_snr_min,
             self.reco_fwhm_max,
             self.reco_ecc_max) = analyse_logic.build_recommended_images(self.analysis_results)
        else:
            self.recommended_images = []
            self.reco_snr_min = self.reco_fwhm_max = self.reco_ecc_max = None
        self._set_widget_state(self.send_reference_button, tk.NORMAL if self.best_reference_path else tk.DISABLED)
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
                if not should_write_command:
                    self._set_widget_state(self.visualize_button, tk.NORMAL)
                    if self.stack_plan_button:
                        self._set_widget_state(self.stack_plan_button, tk.NORMAL)
                else:
                    self._set_widget_state(self.visualize_button, tk.DISABLED)
                    if self.stack_plan_button:
                        self._set_widget_state(self.stack_plan_button, tk.DISABLED)
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
            if hasattr(self, 'analyze_button') and self.analyze_button:
                self.root.after(0, lambda: self.analyze_button.config(state='normal'))
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
            # Activer/désactiver bouton recommandations
            if hasattr(self, 'apply_reco_button') and self.apply_reco_button:
                state = tk.NORMAL if (success and self.recommended_images) else tk.DISABLED
                self.apply_reco_button.config(state=state)
            # --- FIN NOUVEAU ---
        
        if should_write_command and folder_to_stack:
            if self.command_file_path:
                try:
                    with open(self.command_file_path, 'w', encoding='utf-8') as f:
                        f.write(folder_to_stack + "\n")
                        if self.best_reference_path:
                            f.write(self.best_reference_path + "\n")
                    self.root.after(100, self.return_or_quit)
                except Exception as e_write_cmd:
                    print(f"Error writing command file: {e_write_cmd}")
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

        self.update_status("status_analysis_done")
        if success and self.organize_button:
            self.root.after(0, lambda: self.organize_button.config(state=tk.NORMAL))
        print("DEBUG (analyse_gui): Appel final à gc.collect()")
        gc.collect()
        print("DEBUG (analyse_gui): Sortie de finalize_analysis.")

    def _get_best_reference(self):
        valid = [
            r
            for r in self.analysis_results
            if r.get('status') == 'ok'
            and r.get('action') == 'kept'
            and r.get('rejected_reason') is None
            and 'snr' in r
            and is_finite_number(r['snr'])
        ]
        return max(valid, key=lambda r: r['snr'])['path'] if valid else None

    def _compute_recommendations(self):
        """Return a list of recommended images based on percentiles."""
        valid_kept = [
            r for r in self.analysis_results
            if r.get('status') == 'ok'
            and r.get('action') == 'kept'
            and r.get('rejected_reason') is None
            and 'snr' in r and is_finite_number(r['snr'])
        ]
        snrs = [r['snr'] for r in valid_kept]
        sc_vals = [r['starcount'] for r in valid_kept
                   if r.get('starcount') is not None and is_finite_number(r['starcount'])]
        fwhm_vals = [r['fwhm'] for r in valid_kept if is_finite_number(r.get('fwhm', np.nan))]
        ecc_vals = [r['ecc'] for r in valid_kept if is_finite_number(r.get('ecc', np.nan))]

        if len(snrs) >= 5 and len(fwhm_vals) >= 5 and len(ecc_vals) >= 5:
            snr_p25 = np.percentile(snrs, 25)
            fwhm_p75 = np.percentile(fwhm_vals, 75)
            ecc_p75 = np.percentile(ecc_vals, 75)
            return [
                r
                for r in valid_kept
                if r['snr'] >= snr_p25
                and (
                    r.get('fwhm') if is_finite_number(r.get('fwhm')) else np.inf
                )
                <= fwhm_p75
                and (
                    r.get('ecc') if is_finite_number(r.get('ecc')) else np.inf
                )
                <= ecc_p75
            ]
        if len(snrs) >= 5 and len(sc_vals) >= 5:
            snr_p25 = np.percentile(snrs, 25)
            sc_p25 = np.percentile(sc_vals, 25)
            return [r for r in valid_kept if r['snr'] >= snr_p25 and r.get('starcount', -np.inf) >= sc_p25]
        if len(snrs) >= 5:
            snr_p25 = np.percentile(snrs, 25)
            return [r for r in valid_kept if r['snr'] >= snr_p25]
        return []

    def send_reference_to_main(self):
        """Envoie le chemin de référence calculé au GUI principal."""
        if not self.best_reference_path:
            return
        if self.command_file_path:
            try:
                with open(self.command_file_path, 'w', encoding='utf-8') as f:
                    folder = self.input_dir.get() or ''
                    f.write(folder + "\n")
                    f.write(self.best_reference_path + "\n")
            except Exception as e:
                print(f"Error writing reference to command file: {e}")
        elif callable(self.main_app_callback):
            try:
                self.main_app_callback(reference_path=self.best_reference_path)
            except TypeError:
                self.main_app_callback()

    def apply_pending_snr_actions_gui(self):
        """Applique les actions SNR sélectionnées via le RangeSlider."""
        lo = self.current_snr_min
        hi = self.current_snr_max
        if lo is None or hi is None:
            return

        for r in self.analysis_results:
            snr = r.get('snr')
            if r.get('status') == 'ok' and snr is not None and is_finite_number(snr):
                if snr < lo or snr > hi:
                    r['rejected_reason'] = 'low_snr_pending_action'
                    r['action'] = 'pending_snr_action'

        analyse_logic.apply_pending_snr_actions(
            self.analysis_results,
            self.snr_reject_dir.get(),
            delete_rejected_flag=self.reject_action.get() == 'delete',
            move_rejected_flag=self.reject_action.get() == 'move',
            log_callback=lambda *a, **k: None,
            status_callback=lambda *a, **k: None,
            progress_callback=lambda v: None,
            input_dir_abs=self.input_dir.get()
        )

        if self.apply_snr_button:
            self.apply_snr_button.config(state=tk.DISABLED)
        if self.snr_range_slider:
            try:
                self.snr_range_slider.disconnect_events()
            except Exception:
                pass

        if hasattr(self, '_refresh_treeview') and callable(getattr(self, '_refresh_treeview')):
            self._refresh_treeview()

    def apply_pending_starcount_actions_gui(self):
        """Applique les actions Starcount sélectionnées via le RangeSlider."""
        lo = self.current_sc_min
        hi = self.current_sc_max
        if lo is None or hi is None:
            return

        for r in self.analysis_results:
            sc = r.get('starcount')
            if r.get('status') == 'ok' and sc is not None:
                if sc < lo or sc > hi:
                    r['rejected_reason'] = 'starcount_pending_action'
                    r['action'] = 'pending_starcount_action'

        if hasattr(analyse_logic, 'apply_pending_starcount_actions'):
            analyse_logic.apply_pending_starcount_actions(
                self.analysis_results,
                self.starcount_reject_dir.get(),
                delete_rejected_flag=self.reject_action.get() == 'delete',
                move_rejected_flag=self.reject_action.get() == 'move',
                log_callback=lambda *a, **k: None,
                status_callback=lambda *a, **k: None,
                progress_callback=lambda v: None,
                input_dir_abs=self.input_dir.get()
            )

        if self.apply_starcount_button:
            self.apply_starcount_button.config(state=tk.DISABLED)
        if self.starcount_range_slider:
            try:
                self.starcount_range_slider.disconnect_events()
            except Exception:
                pass

    def apply_pending_fwhm_actions_gui(self):
        """Apply FWHM filter based on slider."""
        lo = self.current_fwhm_min
        hi = self.current_fwhm_max
        if lo is None or hi is None:
            return
        for r in self.analysis_results:
            fv = r.get('fwhm')
            if r.get('status') == 'ok' and fv is not None and is_finite_number(fv):
                if fv < lo or fv > hi:
                    r['rejected_reason'] = 'high_fwhm_pending_action'
                    r['action'] = 'pending_fwhm_action'

        if hasattr(analyse_logic, 'apply_pending_fwhm_actions'):
            analyse_logic.apply_pending_fwhm_actions(
                self.analysis_results,
                self.starcount_reject_dir.get(),
                delete_rejected_flag=self.reject_action.get() == 'delete',
                move_rejected_flag=self.reject_action.get() == 'move',
                log_callback=lambda *a, **k: None,
                status_callback=lambda *a, **k: None,
                progress_callback=lambda v: None,
                input_dir_abs=self.input_dir.get()
            )

        if hasattr(self, '_refresh_treeview') and callable(getattr(self, '_refresh_treeview')):
            self._refresh_treeview()

    def apply_pending_ecc_actions_gui(self):
        """Apply eccentricity filter based on slider."""
        lo = self.current_ecc_min
        hi = self.current_ecc_max
        if lo is None or hi is None:
            return
        for r in self.analysis_results:
            ev = r.get('ecc')
            if r.get('status') == 'ok' and ev is not None and is_finite_number(ev):
                if ev < lo or ev > hi:
                    r['rejected_reason'] = 'high_ecc_pending_action'
                    r['action'] = 'pending_ecc_action'

        if hasattr(analyse_logic, 'apply_pending_ecc_actions'):
            analyse_logic.apply_pending_ecc_actions(
                self.analysis_results,
                self.starcount_reject_dir.get(),
                delete_rejected_flag=self.reject_action.get() == 'delete',
                move_rejected_flag=self.reject_action.get() == 'move',
                log_callback=lambda *a, **k: None,
                status_callback=lambda *a, **k: None,
                progress_callback=lambda v: None,
                input_dir_abs=self.input_dir.get()
            )

        if hasattr(self, '_refresh_treeview') and callable(getattr(self, '_refresh_treeview')):
            self._refresh_treeview()

    def _apply_recommendations_gui(self, *, auto: bool = False):
        """Keep only recommended images and apply reject actions."""
        if not getattr(self, 'recommended_images', None):
            if not auto:
                messagebox.showinfo("Info", "Aucune recommandation calculée.")
            return

        reco_files = {os.path.abspath(img['file']) for img in self.recommended_images}

        for r in self.analysis_results:
            if r.get('status') == 'ok' and r.get('action') == 'kept':
                if os.path.abspath(r['file']) not in reco_files:
                    r['rejected_reason'] = 'not_in_recommendation'
                    r['action'] = 'pending_reco_action'

        analyse_logic.apply_pending_reco_actions(
            self.analysis_results,
            self.snr_reject_dir.get(),
            delete_rejected_flag=self.reject_action.get() == 'delete',
            move_rejected_flag=self.reject_action.get() == 'move',
            log_callback=lambda *a, **k: None,
            status_callback=lambda *a, **k: None,
            progress_callback=lambda p: None,
            input_dir_abs=self.input_dir.get()
        )

        # Apply any pending SNR/FWHM/Starcount/Eccentricity actions as well
        try:
            analyse_logic.apply_pending_snr_actions(
                self.analysis_results,
                self.snr_reject_dir.get(),
                delete_rejected_flag=self.reject_action.get() == 'delete',
                move_rejected_flag=self.reject_action.get() == 'move',
                log_callback=lambda *a, **k: None,
                status_callback=lambda *a, **k: None,
                progress_callback=lambda p: None,
                input_dir_abs=self.input_dir.get(),
            )
        except Exception:
            pass

        if hasattr(analyse_logic, 'apply_pending_starcount_actions'):
            analyse_logic.apply_pending_starcount_actions(
                self.analysis_results,
                self.starcount_reject_dir.get(),
                delete_rejected_flag=self.reject_action.get() == 'delete',
                move_rejected_flag=self.reject_action.get() == 'move',
                log_callback=lambda *a, **k: None,
                status_callback=lambda *a, **k: None,
                progress_callback=lambda p: None,
                input_dir_abs=self.input_dir.get(),
            )

        if hasattr(analyse_logic, 'apply_pending_fwhm_actions'):
            analyse_logic.apply_pending_fwhm_actions(
                self.analysis_results,
                self.starcount_reject_dir.get(),
                delete_rejected_flag=self.reject_action.get() == 'delete',
                move_rejected_flag=self.reject_action.get() == 'move',
                log_callback=lambda *a, **k: None,
                status_callback=lambda *a, **k: None,
                progress_callback=lambda p: None,
                input_dir_abs=self.input_dir.get(),
            )

        if hasattr(analyse_logic, 'apply_pending_ecc_actions'):
            analyse_logic.apply_pending_ecc_actions(
                self.analysis_results,
                self.starcount_reject_dir.get(),
                delete_rejected_flag=self.reject_action.get() == 'delete',
                move_rejected_flag=self.reject_action.get() == 'move',
                log_callback=lambda *a, **k: None,
                status_callback=lambda *a, **k: None,
                progress_callback=lambda p: None,
                input_dir_abs=self.input_dir.get(),
            )

        if hasattr(self, '_refresh_treeview') and callable(getattr(self, '_refresh_treeview')):
            self._refresh_treeview()
        if hasattr(self, 'apply_reco_button') and self.apply_reco_button:
            self.apply_reco_button.config(state=tk.DISABLED)
        if hasattr(self, 'visual_apply_reco_button') and self.visual_apply_reco_button:
            self.visual_apply_reco_button.config(state=tk.DISABLED)
        self._regenerate_stack_plan()

    def _on_visual_apply_snr(self):
        """Handler pour le bouton d'application SNR de la fenêtre de visualisation."""
        # 1) réutilise la logique existante
        self.apply_pending_snr_actions_gui()

        # 2) désactive les deux boutons
        if self.apply_snr_button:
            self.apply_snr_button.config(state=tk.DISABLED)
        if self.visual_apply_button:
            self.visual_apply_button.config(state=tk.DISABLED)

        # 3) désactive le RangeSlider pour verrouiller la plage
        try:
            if self.snr_range_slider:
                self.snr_range_slider.set_active(False)
        except Exception:
            pass
        self._regenerate_stack_plan()

    def _on_visual_apply_starcount(self):
        """Handler pour le bouton d'application Starcount de la fenêtre de visualisation."""
        self.apply_pending_starcount_actions_gui()
        if self.apply_starcount_button:
            self.apply_starcount_button.config(state=tk.DISABLED)
        try:
            if self.starcount_range_slider:
                self.starcount_range_slider.set_active(False)
        except Exception:
            pass
        self._regenerate_stack_plan()

    def _on_visual_apply_fwhm(self):
        """Handler for FWHM apply button."""
        self.apply_pending_fwhm_actions_gui()
        if self.apply_fwhm_button:
            self.apply_fwhm_button.config(state=tk.DISABLED)
        try:
            if self.fwhm_range_slider:
                self.fwhm_range_slider.set_active(False)
        except Exception:
            pass
        self._regenerate_stack_plan()

    def _on_visual_apply_ecc(self):
        """Handler for eccentricity apply button."""
        self.apply_pending_ecc_actions_gui()
        if self.apply_ecc_button:
            self.apply_ecc_button.config(state=tk.DISABLED)
        try:
            if self.ecc_range_slider:
                self.ecc_range_slider.set_active(False)
        except Exception:
            pass
        self._regenerate_stack_plan()

    def _on_starcount_slider_change(self, val, patches):
        """Mise à jour visuelle lors du déplacement du RangeSlider Starcount."""
        lo, hi = val
        for p in patches:
            x_left = p.get_x()
            x_right = x_left + p.get_width()
            if x_right < lo or x_left > hi:
                p.set_alpha(0.2)
            else:
                p.set_alpha(1.0)
        if patches:
            patches[0].figure.canvas.draw_idle()
        self.current_sc_min = lo
        self.current_sc_max = hi
        if self.apply_starcount_button:
            self.apply_starcount_button.config(state=tk.NORMAL)

    def _on_fwhm_slider_change(self, val):
        """Update when moving the FWHM slider."""
        lo, hi = val
        self.current_fwhm_min = lo
        self.current_fwhm_max = hi
        if self.apply_fwhm_button:
            self.apply_fwhm_button.config(state=tk.NORMAL)

    def _on_ecc_slider_change(self, val):
        """Update when moving the eccentricity slider."""
        lo, hi = val
        self.current_ecc_min = lo
        self.current_ecc_max = hi
        if self.apply_ecc_button:
            self.apply_ecc_button.config(state=tk.NORMAL)

    def _regenerate_stack_plan(self):
        from stack_plan import generate_stacking_plan, write_stacking_plan_csv
        import os
        import tkinter.messagebox as messagebox

        kept = [r for r in self.analysis_results if r.get('status') == 'ok' and r.get('action') == 'kept']
        if not kept:
            messagebox.showwarning("Plan de stacking", "Aucune image à empiler après tri.")
            return

        pending = any(
            r.get('action', '').startswith('pending') or
            (r.get('rejected_reason', '') and r.get('action', '').endswith('_pending_action'))
            for r in self.analysis_results
        )
        if pending:
            if not messagebox.askyesno(
                "Plan de stacking non à jour",
                "Attention : certains fichiers sont encore marqués à supprimer/déplacer.\n"
                "Le plan ne sera pas fidèle à la future organisation.\n"
                "Voulez-vous appliquer les actions différées maintenant ?"
            ):
                return

        plan = generate_stacking_plan(kept)
        plan_path = os.path.join(os.path.dirname(self.output_log.get()), "stack_plan.csv")
        try:
            write_stacking_plan_csv(plan_path, plan)
        except PermissionError as exc:
            messagebox.showerror(
                "Erreur de permission",
                f"Impossible d'écrire le fichier :\n{plan_path}\n{exc}"
            )
            return
        messagebox.showinfo(
            "Plan de stacking mis à jour",
            f"Le plan a été régénéré :\n{plan_path}\n({len(plan)} fichiers)"
        )

    def _create_stacking_plan_auto(self) -> str | None:
        """Create a stacking plan CSV without user interaction."""
        source = self.recommended_images if getattr(self, 'recommended_images', []) else [
            r for r in self.analysis_results if r.get('status') == 'ok' and r.get('action') == 'kept'
        ]
        if not source:
            return None

        if self.use_bortle.get():
            sort_spec = [('bortle', False), ('session_date', False), ('exposure', True)]
        else:
            sort_spec = [('session_date', False), ('exposure', True)]

        include_expo = False
        if hasattr(self, 'include_exposure_in_batch'):
            try:
                include_expo = bool(self.include_exposure_in_batch.get())
            except Exception:
                include_expo = False

        rows = generate_stacking_plan(
            source,
            include_exposure_in_batch=include_expo,
            sort_spec=sort_spec,
        )
        if not rows:
            return None

        plan_path = os.path.join(self.input_dir.get(), "stack_plan.csv")
        try:
            write_stacking_plan_csv(plan_path, rows)
        except Exception:
            return None

        self.latest_stack_plan_path = plan_path
        return plan_path

    def _auto_stack_workflow(self):
        """Execute analysis post-processing and stacking automatically."""
        try:
            self._apply_recommendations_gui(auto=True)
            self._organize_files_backend(auto=True)
            try:
                self.send_reference_to_main()
            except Exception:
                pass
            plan_path = self._create_stacking_plan_auto()
            if plan_path:
                self.latest_stack_plan_path = plan_path
            self._trigger_stack_via_stacker()
            self.update_status('status_custom', text='Workflow auto-stack terminé.')
        except Exception as exc:
            traceback.print_exc()
            self.update_status('status_custom', text=f'Auto-stack error: {exc}')

    def _trigger_stack_via_stacker(self):
        """Trigger the stacking process via SeeStar Stacker if available."""
        try:
            if hasattr(zeseestarstacker, 'trigger_stack'):
                zeseestarstacker.trigger_stack(self.latest_stack_plan_path)
        except Exception as e:
            print(f"Error triggering stacker: {e}")

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
            self.update_status(
                "status_custom",
                text=self._(
                    "{count} actions SNR appliquées.",
                    default="{count} SNR actions applied.",
                    count=actions_done_count,
                ),
            )
            self.update_results_text(
                f"--- {self._('Fin application rejets SNR. {count} actions effectuées.', default='End applying SNR rejections. {count} actions performed.', count=actions_done_count)} ---"
            )
            
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
                    self.update_results_text(
                        "logic_error_prefix",
                        text=self._(
                            "Erreur mise à jour fichier log: {e}",
                            default="Error updating log file: {e}",
                            e=e_log_update,
                        ),
                    )
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

    def open_stack_plan_window(self):
        """Open a window to create a stacking plan CSV."""
        if not self.analysis_results:
            messagebox.showwarning(self._('msg_warning'),
                                   self._('stack_plan_alert_no_analysis'),
                                   parent=self.root)
            return

        kept_results = [
            r
            for r in self.analysis_results
            if r.get('status') == 'ok' and r.get('action') == 'kept'
        ]

        unique = {
            'mount': sorted({r.get('mount', '') for r in kept_results}),
            'bortle': sorted({str(r.get('bortle', '')) for r in kept_results}),
            'telescope': sorted({r.get('telescope') or 'Unknown' for r in kept_results}),
            'session_date': sorted({(r.get('date_obs') or '').split('T')[0] for r in kept_results}),
            'filter': sorted({r.get('filter', '') for r in kept_results}),
            'exposure': sorted({str(r.get('exposure', '')) for r in kept_results}),
        }

        window = tk.Toplevel(self.root)
        window.title(self._('stack_plan_window_title'))
        window.transient(self.root)
        window.grab_set()

        include_expo_var = tk.BooleanVar(value=False)
        value_vars = {}
        sort_order_vars = {}

        total_var = tk.StringVar()
        batch_var = tk.StringVar()

        def update_preview(*args):
            criteria = {}
            for cat, var_map in value_vars.items():
                selected = [v for v, var in var_map.items() if var.get()]
                if len(selected) != len(var_map):
                    criteria[cat] = selected
            sort_spec = []
            for cat in ['mount', 'bortle', 'telescope', 'session_date', 'filter', 'exposure']:
                order = sort_order_vars[cat].get()
                reverse = order == self._('descending')
                sort_spec.append((cat, reverse))
            rows = generate_stacking_plan(
                kept_results,
                include_exposure_in_batch=include_expo_var.get(),
                criteria=criteria,
                sort_spec=sort_spec,
            )
            total_var.set(self._('stack_plan_preview_total', count=len(rows)))
            batch_var.set(self._('stack_plan_preview_batches', count=len({r['batch_id'] for r in rows})))

        main = ttk.Frame(window, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        row = 0
        for cat, values in unique.items():
            frame = ttk.LabelFrame(main, text=self._(cat))
            frame.grid(row=row, column=0, sticky='w', padx=5, pady=5)
            val_map = {}
            for val in values:
                var = tk.BooleanVar(value=True)
                ttk.Checkbutton(frame, text=str(val), variable=var, command=update_preview).pack(side=tk.LEFT)
                val_map[val] = var
            value_vars[cat] = val_map
            sort_var = tk.StringVar(value=self._('ascending'))
            sort_order_vars[cat] = sort_var
            cb = ttk.Combobox(frame, state='readonly', width=10,
                              values=[self._('ascending'), self._('descending')],
                              textvariable=sort_var)
            cb.pack(side=tk.RIGHT)
            cb.bind('<<ComboboxSelected>>', update_preview)
            row += 1

        ttk.Checkbutton(main, text=self._('include_exposure_in_batch'),
                        variable=include_expo_var, command=update_preview).grid(row=row, column=0, sticky='w', pady=5)
        row += 1

        ttk.Label(main, textvariable=total_var).grid(row=row, column=0, sticky='w'); row += 1
        ttk.Label(main, textvariable=batch_var).grid(row=row, column=0, sticky='w'); row += 1

        def generate_and_close():
            criteria = {}
            for cat, var_map in value_vars.items():
                selected = [v for v, var in var_map.items() if var.get()]
                if len(selected) != len(var_map):
                    criteria[cat] = selected
            sort_spec = []
            for cat in ['mount', 'bortle', 'telescope', 'session_date', 'filter', 'exposure']:
                order = sort_order_vars[cat].get()
                reverse = order == self._('descending')
                sort_spec.append((cat, reverse))
            rows = generate_stacking_plan(
                kept_results,
                include_exposure_in_batch=include_expo_var.get(),
                criteria=criteria,
                sort_spec=sort_spec,
            )
            if not rows:
                messagebox.showwarning(self._('msg_warning'), self._('msg_export_no_images'), parent=window)
                return
            csv_path = os.path.join(os.path.dirname(self.output_log.get()), 'stack_plan.csv')
            write_stacking_plan_csv(csv_path, rows)
            messagebox.showinfo(self._('msg_info'), csv_path, parent=window)
            window.destroy()

        ttk.Button(main, text=self._('generate_plan_button'), command=generate_and_close).grid(row=row, column=0, pady=5)
        update_preview()

    def _organize_files_backend(self, *, auto: bool = False):
        """Backend pour appliquer les actions différées sur les fichiers."""

        delete_flag = self.reject_action.get() == 'delete'
        move_flag = self.reject_action.get() == 'move'

        callbacks = {
            'log': self.update_results_text,
            'status': self.update_status,
            'progress': self.update_progress,
        }

        input_dir = self.input_dir.get()

        total = 0
        try:
            total += analyse_logic.apply_pending_snr_actions(
                self.analysis_results,
                self.snr_reject_dir.get(),
                delete_rejected_flag=delete_flag,
                move_rejected_flag=move_flag,
                log_callback=callbacks['log'],
                status_callback=callbacks['status'],
                progress_callback=callbacks['progress'],
                input_dir_abs=input_dir,
            )

            total += analyse_logic.apply_pending_reco_actions(
                self.analysis_results,
                self.snr_reject_dir.get(),
                delete_rejected_flag=delete_flag,
                move_rejected_flag=move_flag,
                log_callback=callbacks['log'],
                status_callback=callbacks['status'],
                progress_callback=callbacks['progress'],
                input_dir_abs=input_dir,
            )

            if hasattr(analyse_logic, 'apply_pending_trail_actions'):
                total += analyse_logic.apply_pending_trail_actions(
                    self.analysis_results,
                    self.trail_reject_dir.get(),
                    delete_rejected_flag=delete_flag,
                    move_rejected_flag=move_flag,
                    log_callback=callbacks['log'],
                    status_callback=callbacks['status'],
                    progress_callback=callbacks['progress'],
                    input_dir_abs=input_dir,
                )

            if hasattr(analyse_logic, 'apply_pending_starcount_actions'):
                total += analyse_logic.apply_pending_starcount_actions(
                    self.analysis_results,
                    self.starcount_reject_dir.get(),
                    delete_rejected_flag=delete_flag,
                    move_rejected_flag=move_flag,
                    log_callback=callbacks['log'],
                    status_callback=callbacks['status'],
                    progress_callback=callbacks['progress'],
                    input_dir_abs=input_dir,
                )

            if hasattr(analyse_logic, 'apply_pending_fwhm_actions'):
                total += analyse_logic.apply_pending_fwhm_actions(
                    self.analysis_results,
                    self.starcount_reject_dir.get(),
                    delete_rejected_flag=delete_flag,
                    move_rejected_flag=move_flag,
                    log_callback=callbacks['log'],
                    status_callback=callbacks['status'],
                    progress_callback=callbacks['progress'],
                    input_dir_abs=input_dir,
                )

            if hasattr(analyse_logic, 'apply_pending_ecc_actions'):
                total += analyse_logic.apply_pending_ecc_actions(
                    self.analysis_results,
                    self.starcount_reject_dir.get(),
                    delete_rejected_flag=delete_flag,
                    move_rejected_flag=move_flag,
                    log_callback=callbacks['log'],
                    status_callback=callbacks['status'],
                    progress_callback=callbacks['progress'],
                    input_dir_abs=input_dir,
                )

            total += analyse_logic.apply_pending_organization(
                self.analysis_results,
                log_callback=callbacks['log'],
                status_callback=callbacks['status'],
                progress_callback=callbacks['progress'],
                input_dir_abs=input_dir,
            )

            if not auto:
                messagebox.showinfo(
                    self._("msg_info"),
                    self._("msg_organize_done", count=total),
                )
        except Exception as e:
            if not auto:
                messagebox.showerror(
                    self._("msg_error"),
                    self._("msg_organize_failed", e=e),
                )
        finally:
            try:
                self._regenerate_stack_plan()
            except Exception:
                pass
            try:
                if self.output_log.get():
                    current_options = {
                        'analyze_snr': self.analyze_snr.get(),
                        'detect_trails': self.detect_trails.get(),
                        'include_subfolders': self.include_subfolders.get(),
                        'move_rejected': (self.reject_action.get() == 'move'),
                        'delete_rejected': (self.reject_action.get() == 'delete'),
                        'snr_reject_dir': self.snr_reject_dir.get(),
                        'trail_reject_dir': self.trail_reject_dir.get(),
                        'snr_selection_mode': self.snr_selection_mode.get(),
                        'snr_selection_value': self.snr_selection_value.get(),
                        'trail_params': {k: self.trail_params[k].get() for k in self.trail_params},
                    }
                    analyse_logic.write_log_summary(
                        self.output_log.get(),
                        self.input_dir.get(),
                        current_options,
                        results_list=self.analysis_results,
                    )
            except Exception:
                pass
            self._update_log_and_vis_buttons_state()

    def organize_files(self):
        """Applique toutes les actions différées sur les fichiers via le GUI."""
        if self.organize_button:
            self.organize_button.config(state=tk.DISABLED)
        self._organize_files_backend(auto=False)

    def _refresh_treeview(self):
        """Placeholder for treeview refresh if implemented."""
        pass

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
    if importlib.util.find_spec("astropy") is None:
        missing.append("astropy")
    if importlib.util.find_spec("numpy") is None:
        missing.append("numpy")
    if importlib.util.find_spec("matplotlib") is None:
        missing.append("matplotlib")
    # acstools lui-même est vérifié via SATDET_AVAILABLE, mais ses propres dépendances sont vérifiées ici
    # Vérifier skimage et scipy qui sont nécessaires pour satdet version Hough
    if importlib.util.find_spec("skimage") is None:
        missing.append("scikit-image")
    if importlib.util.find_spec("scipy") is None:
        missing.append("scipy")

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
    parser.add_argument(
        "--lang",
        type=str,
        default="fr",
        help="Interface language (e.g. 'en' or 'fr')."
    )
    parser.add_argument(
        "--lock-lang",
        action="store_true",
        help="Disable language selection in the GUI."
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
        app = AstroImageAnalyzerGUI(
            root,
            command_file_path=args.command_file,
            main_app_callback=None,
            initial_lang=args.lang,
            lock_language=args.lock_lang,
        )

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
