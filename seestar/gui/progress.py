"""
Module pour la gestion de la progression du traitement des images astronomiques.
MODIFIED: Ajout de la gestion des niveaux de log pour colorer les messages.
Version: V_ProgressManager_ColorLog_1
"""

import time
import threading
import tkinter as tk
from tkinter import ttk
from time import monotonic as _mono

_PM_LAST_UI = 0.0
_PM_MIN_DT = 0.25   # secondes mini entre deux MAJ GUI (évite flood event loop)


class ProgressManager:
    """
    Classe pour gérer la progression et les indications de temps du traitement.
    """

    def __init__(self, progress_bar, status_text, remaining_time_var, elapsed_time_var):
        """
        Initialise le gestionnaire de progression.
        """
        if not isinstance(progress_bar, ttk.Progressbar): raise TypeError("progress_bar must be ttk.Progressbar")
        if not isinstance(status_text, tk.Text): raise TypeError("status_text must be tk.Text")
        if not isinstance(remaining_time_var, tk.StringVar): raise TypeError("remaining_time_var must be tk.StringVar")
        if not isinstance(elapsed_time_var, tk.StringVar): raise TypeError("elapsed_time_var must be tk.StringVar")

        self.progress_bar = progress_bar
        self.status_text = status_text
        self.remaining_time_var = remaining_time_var
        try:
            self.remaining_time_var.set("--:--:--")
        except tk.TclError:
            pass
        self.elapsed_time_var = elapsed_time_var
        self.start_time = None
        self.timer_id = None
        
        try:
            self.root = progress_bar.winfo_toplevel()
            if not isinstance(self.root, tk.Tk):
                 parent = self.root.master
                 if isinstance(parent, tk.Tk):
                      self.root = parent
                 else: 
                      print("Warning: Could not reliably get Tk root in ProgressManager.")
                      self.root = progress_bar 
        except Exception:
            print("Error getting root window in ProgressManager.")
            self.root = progress_bar 

        # --- NOUVEAU : Configuration des tags pour les couleurs ---
        try:
            if self.status_text.winfo_exists():
                self.status_text.tag_configure("error_log", foreground="red")
                self.status_text.tag_configure("warning_log", foreground="orange") 
                # Tu peux ajouter d'autres couleurs/tags ici si besoin
                # Par exemple, pour les messages de succès :
                # self.status_text.tag_configure("success_log", foreground="green")
                print("DEBUG ProgressManager: Tags de couleur configurés pour status_text.")
        except tk.TclError:
            print("Warning ProgressManager: Impossible de configurer les tags (status_text détruit?).")
        # --- FIN NOUVEAU ---



# --- DANS LA CLASSE ProgressManager DANS seestar/gui/progress.py ---

    def update_progress(self, message, progress=None, level=None):
        """
        Met à jour la barre de progression et le texte de statut via after_idle.
        Gère l'état désactivé du widget Text et applique des couleurs basées sur 'level'.
        MODIFIED: Application des tags de couleur.
        Version: V_ProgressManager_ColorLog_ApplyTags
        """
        now = _mono()

        if now - _PM_LAST_UI < _PM_MIN_DT:
            return

        if threading.current_thread() is threading.main_thread():
            print("Warning: ProgressManager.update_progress called from main thread")

        def _update_ui():
            global _PM_LAST_UI
            _PM_LAST_UI = now
            try:
                # Check if widgets still exist before configuring them
                if not self.progress_bar.winfo_exists() or not self.status_text.winfo_exists():
                    return

                # Update progress bar if value provided
                if progress is not None:
                    try:
                        clamped_progress = max(0.0, min(100.0, float(progress)))
                        self.progress_bar.configure(value=clamped_progress)
                    except (ValueError, tk.TclError):
                        pass 

                # Append message to status text area
                if message:
                    original_state = self.status_text['state']
                    tag_to_apply = None # Tag par défaut (pas de couleur spéciale)
                    # --- DEBUG LOG POUR LE NIVEAU REÇU ---
                    print(f"DEBUG ProgressManager._update_ui: Message='{str(message)[:50]}...', Progress={progress}, Level REÇU='{level}'")
                    # --- FIN DEBUG LOG ---
                    # --- Choix du tag en fonction du niveau ---
                    if level == "ERROR":
                        tag_to_apply = "error_log"
                    elif level == "WARN" or level == "INFO_IMPORTANT": # Gère notre UNALIGNED_INFO
                        tag_to_apply = "warning_log"
                    # elif level == "SUCCESS": # Exemple
                    #     tag_to_apply = "success_log"
                    # --- Fin Choix du tag ---

                    try:
                        # Enable widget temporarily to insert text
                        if original_state == tk.DISABLED:
                            self.status_text.config(state=tk.NORMAL)

                        # Préparer le message (le timestamp est optionnel, géré par le backend actuellement)
                        message_to_log = str(message) 

                        if tag_to_apply:
                            self.status_text.insert(tk.END, message_to_log + "\n", tag_to_apply)
                        else:
                            self.status_text.insert(tk.END, message_to_log + "\n")
                        
                        self.status_text.see(tk.END) # Scroll to the end

                    finally:
                        # Restore original state even if insert fails
                        if original_state == tk.DISABLED:
                             if self.status_text.winfo_exists():
                                 self.status_text.config(state=tk.DISABLED)

            except tk.TclError as e:
                # Catch errors if widgets are destroyed between check and configure/insert
                # print(f"Debug: TclError during _update_ui: {e}")
                pass
            except Exception as e:
                # Catch any other unexpected error during UI update
                print(f"Error during ProgressManager _update_ui: {e}")
                import traceback
                traceback.print_exc(limit=2)

        # Schedule the UI update to run in the main Tkinter thread
        try:
             if self.root and hasattr(self.root, 'after_idle'):
                 self.root.after_idle(_update_ui)
             else:
                 print("Warning: ProgressManager cannot schedule UI update (no valid root).")
        except Exception as e:
             print(f"Error scheduling UI update in ProgressManager: {e}")

    def set_remaining(self, time_str):
        """Update the remaining time display variable."""
        try:
            if hasattr(self.remaining_time_var, 'set'):
                self.remaining_time_var.set(str(time_str))
        except tk.TclError:
            pass



    # ... (le reste des méthodes start_timer, update_timer, stop_timer, reset reste inchangé) ...
    def start_timer(self):
        """Démarre le timer pour le temps écoulé."""
        if self.timer_id: 
             self.stop_timer()
        self.start_time = time.monotonic() 
        if self.root and hasattr(self.root, 'after'):
            self.update_timer() 
        else:
            print("Warning: ProgressManager cannot start timer (no valid root).")

    def update_timer(self):
        """Met à jour le timer de temps écoulé (runs periodically in main thread via after)."""
        if self.start_time and self.root and hasattr(self.root, 'winfo_exists') and self.root.winfo_exists():
            elapsed = time.monotonic() - self.start_time
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            try:
                 self.elapsed_time_var.set(f"{hours:02}:{minutes:02}:{seconds:02}")
            except tk.TclError: 
                 self.timer_id = None 
                 return

            try:
                 self.timer_id = self.root.after(1000, self.update_timer) 
            except tk.TclError: 
                 self.timer_id = None
        else:
             self.timer_id = None 

    def stop_timer(self):
        """Arrête le timer."""
        if self.timer_id:
            try:
                if self.root and hasattr(self.root, 'winfo_exists') and self.root.winfo_exists():
                    self.root.after_cancel(self.timer_id)
            except tk.TclError: pass 
            except Exception as e: 
                print(f"Error cancelling timer in ProgressManager: {e}")
            self.timer_id = None


    def reset(self):
        """Réinitialise la progression et les timers."""
        self.stop_timer() 
        try:
            if hasattr(self,'progress_bar') and self.progress_bar.winfo_exists():
                self.progress_bar.configure(value=0)
            if hasattr(self,'status_text') and self.status_text.winfo_exists():
                original_state = self.status_text['state']
                if original_state == tk.DISABLED: self.status_text.config(state=tk.NORMAL)
                self.status_text.delete(1.0, tk.END) 
                if original_state == tk.DISABLED: self.status_text.config(state=tk.DISABLED)
            if hasattr(self,'elapsed_time_var'): self.elapsed_time_var.set("00:00:00")
            if hasattr(self,'remaining_time_var'): self.remaining_time_var.set("--:--:--")
        except tk.TclError:
             print("Warning: TclError during ProgressManager reset (widgets likely destroyed).")
        except Exception as e:
             print(f"Error during ProgressManager reset: {e}")
        self.start_time = None 
# --- END OF FILE seestar/gui/progress.py ---
