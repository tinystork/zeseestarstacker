# --- START OF FILE seestar/gui/mosaic_gui.py ---
"""
Fenêtre modale pour la configuration des paramètres de traitement en mode Mosaïque.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import traceback

class MosaicSettingsWindow(tk.Toplevel):
    """
    Fenêtre Toplevel modale pour configurer et activer le mode Mosaïque.
    """
    def __init__(self, parent_gui):
        """
        Initialise la fenêtre.

        Args:
            parent_gui: Instance de SeestarStackerGUI (la fenêtre principale).
        """
        print("DEBUG (MosaicSettingsWindow __init__): Initialisation...")
        # --- Vérification parent_gui ---
        if not hasattr(parent_gui, 'root') or not parent_gui.root.winfo_exists():
             print("ERREUR CRITIQUE (MosaicSettingsWindow): Instance parent_gui invalide ou fenêtre racine détruite.")
             # On ne peut pas initialiser Toplevel sans parent valide
             # Lever une exception est peut-être mieux ici
             raise ValueError("Parent GUI invalide pour MosaicSettingsWindow")

        super().__init__(parent_gui.root) # Initialise Toplevel avec la racine du parent
        self.parent_gui = parent_gui
        self.withdraw() # Cacher pendant la construction

        self.title(self.parent_gui.tr("mosaic_settings_title", default="Mosaic Options"))
        self.transient(parent_gui.root) # Lier à la fenêtre principale
        self.resizable(False, False)   # Non redimensionnable pour commencer

        # --- Variables Tkinter locales à cette fenêtre ---
        # Récupérer l'état actuel depuis le GUI parent
        initial_mosaic_state = getattr(self.parent_gui, 'mosaic_mode_active', False)
        self.local_mosaic_active_var = tk.BooleanVar(value=initial_mosaic_state)
        print(f"DEBUG (MosaicSettingsWindow __init__): État initial mosaic lu depuis parent: {initial_mosaic_state}")

        # --- Création des widgets ---
        main_frame = ttk.Frame(self, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Options Principales ---
        options_frame = ttk.LabelFrame(main_frame, text=self.parent_gui.tr("mosaic_activation_frame", default="Activation"))
        options_frame.pack(fill=tk.X, pady=(0, 10))

        self.activate_check = ttk.Checkbutton(
            options_frame,
            text=self.parent_gui.tr("mosaic_activate_label", default="Enable Mosaic Processing Mode"),
            variable=self.local_mosaic_active_var,
            command=self._on_toggle_activate # Optionnel: pour griser/dégriser autres options
        )
        self.activate_check.pack(anchor=tk.W, padx=10, pady=10)

        # --- Espace réservé pour futures options ---
        # future_options_frame = ttk.LabelFrame(main_frame, text="Options Avancées (Futures)")
        # future_options_frame.pack(fill=tk.X, pady=10)
        # ttk.Label(future_options_frame, text="Options de grille, background, etc.").pack(padx=5, pady=5)

        # --- Boutons OK / Annuler ---
        button_frame = ttk.Frame(main_frame)
        # Utiliser fill=tk.X et side=tk.RIGHT pour aligner à droite
        button_frame.pack(fill=tk.X, pady=(15, 0))

        self.cancel_button = ttk.Button(
            button_frame,
            text=self.parent_gui.tr("cancel", default="Cancel"),
            command=self._cancel
        )
        # Packer Annuler à droite
        self.cancel_button.pack(side=tk.RIGHT, padx=(5, 0))

        self.ok_button = ttk.Button(
            button_frame,
            text=self.parent_gui.tr("ok", default="OK"),
            command=self._apply_and_close,
            # style="Accent.TButton" # Si vous avez un style accentué
        )
        # Packer OK à droite de Annuler
        self.ok_button.pack(side=tk.RIGHT)

        print("DEBUG (MosaicSettingsWindow __init__): Widgets créés.")

        # --- Finalisation ---
        self._on_toggle_activate() # Mettre l'état initial des options
        self.update_idletasks() # Calculer taille nécessaire

        # Centrer sur la fenêtre parente
        parent_x = self.parent_gui.root.winfo_rootx()
        parent_y = self.parent_gui.root.winfo_rooty()
        parent_w = self.parent_gui.root.winfo_width()
        parent_h = self.parent_gui.root.winfo_height()
        win_w = self.winfo_width()
        win_h = self.winfo_height()
        x = parent_x + (parent_w // 2) - (win_w // 2)
        y = parent_y + (parent_h // 2) - (win_h // 2)
        self.geometry(f"+{x}+{y}") # Positionner
        print(f"DEBUG (MosaicSettingsWindow __init__): Fenêtre positionnée à {x},{y}.")

        self.deiconify() # Afficher la fenêtre
        self.focus_force() # Donner le focus
        self.grab_set() # Rendre modale

        # Attendre que cette fenêtre soit fermée
        self.wait_window(self)
        print("DEBUG (MosaicSettingsWindow __init__): Fenêtre fermée (wait_window finished).")

    def _on_toggle_activate(self):
        """ (Optionnel) Grise/dégrise les futures options si la case est cochée/décochée. """
        is_active = self.local_mosaic_active_var.get()
        print(f"DEBUG (MosaicSettingsWindow _on_toggle_activate): Nouvel état = {is_active}")
        # future_options_state = tk.NORMAL if is_active else tk.DISABLED
        # Configurer l'état des futurs widgets ici
        # Exemple: if hasattr(self, 'grid_option_widget'): self.grid_option_widget.config(state=future_options_state)
        pass # Rien à faire pour l'instant

    def _apply_and_close(self):
        """ Applique les paramètres au GUI parent et ferme la fenêtre. """
        new_state = self.local_mosaic_active_var.get()
        print(f"DEBUG (MosaicSettingsWindow _apply_and_close): Application état mosaïque = {new_state} au parent.")
        try:
            # Mettre à jour le flag dans l'instance parente
            setattr(self.parent_gui, 'mosaic_mode_active', new_state)
            # Log dans la console du parent si possible
            if hasattr(self.parent_gui, 'update_progress_gui'):
                status_msg_key = "mosaic_mode_enabled_log" if new_state else "mosaic_mode_disabled_log"
                status_msg_default = f"Mosaic mode {'ENABLED' if new_state else 'DISABLED'}."
                self.parent_gui.update_progress_gui(
                    f"ⓘ {self.parent_gui.tr(status_msg_key, default=status_msg_default)}", None
                )
            # Mettre à jour l'UI parent pour refléter le changement (si nécessaire)
            if hasattr(self.parent_gui, '_update_mosaic_status_indicator'):
                 self.parent_gui._update_mosaic_status_indicator()

        except AttributeError as e:
             print(f"ERREUR (MosaicSettingsWindow _apply_and_close): Attribut manquant sur parent_gui ? {e}")
        except Exception as e:
             print(f"ERREUR (MosaicSettingsWindow _apply_and_close): Erreur application état: {e}")
             # Afficher une erreur à l'utilisateur ?
             messagebox.showerror("Erreur", f"Erreur application paramètres mosaïque:\n{e}", parent=self)
             return # Ne pas fermer si erreur ?

        # Fermer la fenêtre
        self.grab_release() # Libérer le grab avant de détruire
        self.destroy()
        print("DEBUG (MosaicSettingsWindow _apply_and_close): Fenêtre détruite.")

    def _cancel(self):
        """ Annule et ferme la fenêtre sans appliquer les changements. """
        print("DEBUG (MosaicSettingsWindow _cancel): Annulation, fermeture sans appliquer.")
        self.grab_release()
        self.destroy()

# --- FIN DU FICHIER seestar/gui/mosaic_gui.py ---