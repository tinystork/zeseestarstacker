"""
Module pour la gestion de la progression du traitement des images astronomiques.
"""

import time
import tkinter as tk


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
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.remaining_time_var = remaining_time_var
        self.elapsed_time_var = elapsed_time_var
        self.start_time = None
        self.timer_id = None
        # Référence à la fenêtre principale
        self.root = progress_bar.master.winfo_toplevel()

    def update_progress(self, message, progress=None):
        """
        Met à jour la barre de progression et le texte de statut.

        Args:
            message (str): Message à afficher dans la zone de statut
            progress (float, optional): Valeur de progression (0-100)
        """
        if progress is not None:
            self.progress_bar.configure(value=progress)

            # Extraire le temps restant s'il est présent dans le message
            if "Temps restant estimé:" in message:
                try:
                    remaining_time = message.split(
                        "Temps restant estimé:")[1].strip()
                    self.remaining_time_var.set(remaining_time)
                except:
                    pass
            elif "Estimated time remaining:" in message:
                try:
                    remaining_time = message.split(
                        "Estimated time remaining:")[1].strip()
                    self.remaining_time_var.set(remaining_time)
                except:
                    pass

        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)  # Défiler pour montrer la dernière ligne
        self.root.update_idletasks()  # Mettre à jour l'interface

    def start_timer(self):
        """Démarre le timer pour le temps écoulé."""
        self.start_time = time.time()
        self.update_timer()

    def update_timer(self):
        """Met à jour le timer de temps écoulé."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.elapsed_time_var.set(f"{hours:02}:{minutes:02}:{seconds:02}")
            self.timer_id = self.root.after(1000, self.update_timer)

    def stop_timer(self):
        """Arrête le timer."""
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
            self.timer_id = None

    def reset(self):
        """Réinitialise la progression et les timers."""
        self.progress_bar.configure(value=0)
        self.stop_timer()
        self.elapsed_time_var.set("00:00:00")
        self.remaining_time_var.set("--:--:--")
        self.status_text.delete(1.0, tk.END)
        self.start_time = None
