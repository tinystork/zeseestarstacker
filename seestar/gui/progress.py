# --- START OF FILE seestar/gui/progress.py ---
"""
Module pour la gestion de la progression du traitement des images astronomiques.
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
        self.root = progress_bar.winfo_toplevel()

    def update_progress(self, message, progress=None):
        """
        Met à jour la barre de progression et le texte de statut.
        Assure que les mises à jour se font sur le thread principal Tkinter.

        Args:
            message (str): Message à afficher dans la zone de statut
            progress (float, optional): Valeur de progression (0-100)
        """
        def _update_ui():
            # Check if widgets still exist before configuring them
            if not self.progress_bar.winfo_exists() or not self.status_text.winfo_exists():
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
                 try:
                      # Insert message and scroll to the end
                      self.status_text.insert(tk.END, message + "\n")
                      self.status_text.see(tk.END)
                 except tk.TclError:
                      pass # Ignore error if text widget is gone

            # Optional: Force UI update more immediately if needed, but use with caution
            # self.root.update_idletasks()

        # Schedule the UI update to run in the main Tkinter thread
        try:
             # Use after_idle to ensure it runs when Tkinter is ready
             self.root.after_idle(_update_ui)
        except Exception as e:
             # Handle cases where the root window might be destroyed
             print(f"Error scheduling UI update in ProgressManager: {e}")


    def start_timer(self):
        """Démarre le timer pour le temps écoulé."""
        if self.timer_id: # Prevent multiple concurrent timers
             self.stop_timer()
        self.start_time = time.monotonic() # Use monotonic clock for reliable interval measurement
        self.update_timer() # Start the update loop immediately

    def update_timer(self):
        """Met à jour le timer de temps écoulé (runs periodically in main thread via after)."""
        if self.start_time and self.root.winfo_exists(): # Check if root window still exists
            elapsed = time.monotonic() - self.start_time
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            try:
                 self.elapsed_time_var.set(f"{hours:02}:{minutes:02}:{seconds:02}")
            except tk.TclError: # Handle if variable is destroyed
                 self.timer_id = None # Stop timer if variable gone
                 return

            # Schedule the next update
            self.timer_id = self.root.after(1000, self.update_timer) # Update every second
        else:
             self.timer_id = None # Ensure timer stops if start_time is None or window closed

    def stop_timer(self):
        """Arrête le timer."""
        if self.timer_id:
            try: self.root.after_cancel(self.timer_id)
            except tk.TclError: pass # Ignore if root is destroyed
            self.timer_id = None
            # Keep start_time as is, maybe set final elapsed time here?
            # if self.start_time:
            #     # Set final value one last time
            #     elapsed = time.monotonic() - self.start_time
            #     hours, remainder = divmod(int(elapsed), 3600)
            #     minutes, seconds = divmod(remainder, 60)
            #     try: self.elapsed_time_var.set(f"{hours:02}:{minutes:02}:{seconds:02}")
            #     except tk.TclError: pass

    def reset(self):
        """Réinitialise la progression et les timers."""
        self.stop_timer() # Ensure timer is stopped before resetting vars
        try:
             if self.progress_bar.winfo_exists(): self.progress_bar.configure(value=0)
             if self.status_text.winfo_exists(): self.status_text.delete(1.0, tk.END) # Clear status text
             self.elapsed_time_var.set("00:00:00")
             self.remaining_time_var.set("--:--:--")
        except tk.TclError:
             print("Warning: Failed to reset progress widgets (possibly destroyed).")
        self.start_time = None # Reset start time explicitly
# --- END OF FILE seestar/gui/progress.py ---