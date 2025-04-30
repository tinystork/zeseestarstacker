# --- START OF FILE seestar/gui/progress.py ---
"""
Module pour la gestion de la progression du traitement des images astronomiques.
(Version Révisée: Correction MAJ Text désactivé)
"""

import time
import tkinter as tk
from tkinter import ttk # Explicitly import ttk if needed


class ProgressManager:
    """
    Classe pour gérer la progression et les indications de temps du traitement.
    """

    def __init__(self, progress_bar, status_text, remaining_time_var, elapsed_time_var):
        """
        Initialise le gestionnaire de progression.

        Args:
            progress_bar (ttk.Progressbar): Barre de progression
            status_text (tk.Text): Zone de texte pour les messages de statut
            remaining_time_var (tk.StringVar): Variable pour le temps restant
            elapsed_time_var (tk.StringVar): Variable pour le temps écoulé
        """
        # Input validation
        if not isinstance(progress_bar, ttk.Progressbar): raise TypeError("progress_bar must be ttk.Progressbar")
        if not isinstance(status_text, tk.Text): raise TypeError("status_text must be tk.Text")
        if not isinstance(remaining_time_var, tk.StringVar): raise TypeError("remaining_time_var must be tk.StringVar")
        if not isinstance(elapsed_time_var, tk.StringVar): raise TypeError("elapsed_time_var must be tk.StringVar")

        self.progress_bar = progress_bar
        self.status_text = status_text
        self.remaining_time_var = remaining_time_var
        self.elapsed_time_var = elapsed_time_var
        self.start_time = None # Uses time.monotonic()
        self.timer_id = None
        # Get a reference to the root window (or a widget within it) for using 'after'
        # Ensure we get the actual top-level window
        try:
            self.root = progress_bar.winfo_toplevel()
            # Verify it's a Tk instance, not just a Frame
            if not isinstance(self.root, tk.Tk):
                 # Try going up one more level if possible
                 parent = self.root.master
                 if isinstance(parent, tk.Tk):
                      self.root = parent
                 else: # Fallback or raise error
                      print("Warning: Could not reliably get Tk root in ProgressManager.")
                      self.root = progress_bar # Might still work with after_idle
        except Exception:
            print("Error getting root window in ProgressManager.")
            self.root = progress_bar # Fallback


    def update_progress(self, message, progress=None):
        """
        Met à jour la barre de progression et le texte de statut via after_idle.
        Gère l'état désactivé du widget Text.
        """
        def _update_ui():
            try:
                # Check if widgets still exist before configuring them
                if not self.progress_bar.winfo_exists() or not self.status_text.winfo_exists():
                    # print("Debug: Progress widgets destroyed, skipping update.")
                    return # Stop if widgets are destroyed

                # Update progress bar if value provided
                if progress is not None:
                    try:
                        # Clamp progress value between 0 and 100
                        clamped_progress = max(0.0, min(100.0, float(progress)))
                        self.progress_bar.configure(value=clamped_progress)
                    except (ValueError, tk.TclError):
                        pass # Ignore errors if progress is not a valid number or widget is gone

                # Append message to status text area
                if message:
                    original_state = self.status_text['state']
                    try:
                        # Enable widget temporarily to insert text
                        if original_state == tk.DISABLED:
                            self.status_text.config(state=tk.NORMAL)

                        self.status_text.insert(tk.END, message + "\n")
                        self.status_text.see(tk.END) # Scroll to the end

                    finally:
                        # Restore original state even if insert fails
                        if original_state == tk.DISABLED:
                             # Check again if widget still exists before setting state
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
             # Use after_idle to ensure it runs when Tkinter is ready
             if self.root and hasattr(self.root, 'after_idle'):
                 self.root.after_idle(_update_ui)
             else:
                 print("Warning: ProgressManager cannot schedule UI update (no valid root).")
        except Exception as e:
             # Handle cases where the root window might be destroyed
             print(f"Error scheduling UI update in ProgressManager: {e}")


    def start_timer(self):
        """Démarre le timer pour le temps écoulé."""
        if self.timer_id: # Prevent multiple concurrent timers
             self.stop_timer()
        self.start_time = time.monotonic() # Use monotonic clock for reliable interval measurement
        # Ensure root is valid before starting timer loop
        if self.root and hasattr(self.root, 'after'):
            self.update_timer() # Start the update loop immediately
        else:
            print("Warning: ProgressManager cannot start timer (no valid root).")

    def update_timer(self):
        """Met à jour le timer de temps écoulé (runs periodically in main thread via after)."""
        # Check if start_time is set and root window still exists
        if self.start_time and self.root and hasattr(self.root, 'winfo_exists') and self.root.winfo_exists():
            elapsed = time.monotonic() - self.start_time
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            try:
                 self.elapsed_time_var.set(f"{hours:02}:{minutes:02}:{seconds:02}")
            except tk.TclError: # Handle if variable is destroyed
                 self.timer_id = None # Stop timer if variable gone
                 return

            # Schedule the next update
            try:
                 self.timer_id = self.root.after(1000, self.update_timer) # Update every second
            except tk.TclError: # Handle if root destroyed before next 'after'
                 self.timer_id = None
        else:
             self.timer_id = None # Ensure timer stops if start_time is None or window closed

    def stop_timer(self):
        """Arrête le timer."""
        if self.timer_id:
            try:
                # Check if root exists before calling after_cancel
                if self.root and hasattr(self.root, 'winfo_exists') and self.root.winfo_exists():
                    self.root.after_cancel(self.timer_id)
            except tk.TclError: pass # Ignore if root is destroyed or timer invalid
            except Exception as e: # Catch other potential errors
                print(f"Error cancelling timer in ProgressManager: {e}")
            self.timer_id = None
            # Optional: Set final elapsed time here
            # if self.start_time:
            #     elapsed = time.monotonic() - self.start_time
            #     hours, rem = divmod(int(elapsed), 3600); mins, secs = divmod(rem, 60)
            #     try: self.elapsed_time_var.set(f"{hours:02}:{mins:02}:{secs:02}")
            #     except tk.TclError: pass


    def reset(self):
        """Réinitialise la progression et les timers."""
        self.stop_timer() # Ensure timer is stopped before resetting vars
        try:
            # Check widget existence before configuring/deleting
            if hasattr(self,'progress_bar') and self.progress_bar.winfo_exists():
                self.progress_bar.configure(value=0)
            if hasattr(self,'status_text') and self.status_text.winfo_exists():
                # Temporarily enable to delete content
                original_state = self.status_text['state']
                if original_state == tk.DISABLED: self.status_text.config(state=tk.NORMAL)
                self.status_text.delete(1.0, tk.END) # Clear status text
                if original_state == tk.DISABLED: self.status_text.config(state=tk.DISABLED)
            if hasattr(self,'elapsed_time_var'): self.elapsed_time_var.set("00:00:00")
            if hasattr(self,'remaining_time_var'): self.remaining_time_var.set("--:--:--")
        except tk.TclError:
             print("Warning: TclError during ProgressManager reset (widgets likely destroyed).")
        except Exception as e:
             print(f"Error during ProgressManager reset: {e}")
        self.start_time = None # Reset start time explicitly

# --- END OF FILE seestar/gui/progress.py ---