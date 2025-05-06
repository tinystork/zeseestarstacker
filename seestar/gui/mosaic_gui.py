# --- START OF FILE seestar/gui/mosaic_gui.py ---
"""
Fenêtre modale pour la configuration des paramètres de traitement en mode Mosaïque.
MAJ: Ajout options Kernel et Pixfrac Drizzle.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import traceback
import numpy as np # Ajouté pour np.clip

# Liste des noyaux valides (pour le Combobox)
VALID_DRIZZLE_KERNELS = ['square', 'gaussian', 'point', 'tophat', 'turbo', 'lanczos2', 'lanczos3']

class MosaicSettingsWindow(tk.Toplevel):
    """
    Fenêtre Toplevel modale pour configurer et activer le mode Mosaïque.
    """

# --- DANS LE FICHIER: seestar/gui/mosaic_gui.py ---
# --- DANS LA CLASSE: MosaicSettingsWindow ---

    def __init__(self, parent_gui):
        """
        Initialise la fenêtre.
        MAJ: Ajout options Kernel, Pixfrac et Clé API.

        Args:
            parent_gui: Instance de SeestarStackerGUI (la fenêtre principale).
        """
        print("DEBUG (MosaicSettingsWindow __init__): Initialisation...")
        if not hasattr(parent_gui, 'root') or not parent_gui.root.winfo_exists():
             raise ValueError("Parent GUI invalide pour MosaicSettingsWindow")

        super().__init__(parent_gui.root)
        self.parent_gui = parent_gui
        self.withdraw() # Cacher pendant la construction

        # --- Récupérer l'état actuel et les settings spécifiques depuis le parent ---
        initial_mosaic_state = getattr(self.parent_gui, 'mosaic_mode_active', False)
        mosaic_settings = getattr(self.parent_gui, 'mosaic_settings', {})
        # Utiliser les valeurs par défaut globales si non trouvées dans mosaic_settings ou settings globaux
        initial_kernel = mosaic_settings.get('kernel', self.parent_gui.settings.drizzle_kernel)
        initial_pixfrac = mosaic_settings.get('pixfrac', self.parent_gui.settings.drizzle_pixfrac)
        # Récupérer la clé API sauvegardée depuis le parent GUI (via son StringVar)
        initial_api_key = getattr(self.parent_gui, 'astrometry_api_key_var', tk.StringVar()).get()

        print(f"DEBUG (MosaicSettingsWindow __init__): État initial mosaic: {initial_mosaic_state}")
        print(f"DEBUG (MosaicSettingsWindow __init__): Settings initiaux -> Kernel: {initial_kernel}, Pixfrac: {initial_pixfrac}")
        # Éviter d'afficher la clé API complète dans les logs généraux
        print(f"DEBUG (MosaicSettingsWindow __init__): Clé API initiale chargée: {'Oui' if initial_api_key else 'Non'}")

        # --- Variables Tkinter locales ---
        self.local_mosaic_active_var = tk.BooleanVar(value=initial_mosaic_state)
        self.local_drizzle_kernel_var = tk.StringVar(value=initial_kernel)
        self.local_drizzle_pixfrac_var = tk.DoubleVar(value=initial_pixfrac)
        ### AJOUT Clé API Var ###
        self.local_api_key_var = tk.StringVar(value=initial_api_key)
        ### FIN AJOUT ###

        # --- Interface ---
        self.title(self.parent_gui.tr("mosaic_settings_title", default="Mosaic Options"))
        self.transient(parent_gui.root)
        # self.resizable(False, False) # Laisser redimensionnable pour l'instant

        # --- Création des widgets ---
        main_frame = ttk.Frame(self, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Frame Activation ---
        options_frame = ttk.LabelFrame(main_frame, text=self.parent_gui.tr("mosaic_activation_frame", default="Activation"))
        options_frame.pack(fill=tk.X, pady=(0, 10))
        self.activate_check = ttk.Checkbutton(
            options_frame,
            text=self.parent_gui.tr("mosaic_activate_label", default="Enable Mosaic Processing Mode"),
            variable=self.local_mosaic_active_var,
            command=self._on_toggle_activate
        )
        self.activate_check.pack(anchor=tk.W, padx=10, pady=10)

        ### AJOUT Cadre API Key ###
        # --- Cadre pour la clé API ---
        self.api_key_frame = ttk.LabelFrame(main_frame, text=self.parent_gui.tr("mosaic_api_key_frame", default="Astrometry.net API Key (Required for Mosaic)"))
        self.api_key_frame.pack(fill=tk.X, pady=5)

        api_key_inner_frame = ttk.Frame(self.api_key_frame, padding=5)
        api_key_inner_frame.pack(fill=tk.X)

        api_key_label = ttk.Label(api_key_inner_frame, text=self.parent_gui.tr("mosaic_api_key_label", default="API Key:"), width=10)
        api_key_label.pack(side=tk.LEFT, padx=(0, 5))

        # Champ Entry pour la clé, avec affichage masqué
        self.api_key_entry = ttk.Entry(
            api_key_inner_frame,
            textvariable=self.local_api_key_var,
            show="*", # Masquer la clé
            width=40 # Ajuster la largeur si besoin
        )
        self.api_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Petit label d'aide (optionnel)
        api_help_label = ttk.Label(self.api_key_frame, text=self.parent_gui.tr("mosaic_api_key_help", default="Get your key from nova.astrometry.net (free account)"), style="TLabel", foreground="gray", font=("Arial", 8))
        api_help_label.pack(anchor=tk.W, padx=10, pady=(0, 5))
        ### FIN AJOUT Cadre API Key ###

        ### AJOUT Kernel/Pixfrac (Cadre existant) ###
        # --- Cadre pour les options Drizzle spécifiques ---
        self.drizzle_options_frame = ttk.LabelFrame(main_frame, text=self.parent_gui.tr("mosaic_drizzle_options_frame", default="Mosaic Drizzle Options"))
        self.drizzle_options_frame.pack(fill=tk.X, pady=10)

        # Ligne Kernel (inchangée)
        kernel_frame = ttk.Frame(self.drizzle_options_frame, padding=5); kernel_frame.pack(fill=tk.X)
        kernel_label = ttk.Label(kernel_frame, text=self.parent_gui.tr("mosaic_drizzle_kernel_label", default="Kernel:"), width=10); kernel_label.pack(side=tk.LEFT, padx=(0, 5))
        self.kernel_combo = ttk.Combobox(kernel_frame, textvariable=self.local_drizzle_kernel_var, values=VALID_DRIZZLE_KERNELS, state="readonly", width=15); self.kernel_combo.pack(side=tk.LEFT, padx=5)

        # Ligne Pixfrac (inchangée)
        pixfrac_frame = ttk.Frame(self.drizzle_options_frame, padding=5); pixfrac_frame.pack(fill=tk.X)
        pixfrac_label = ttk.Label(pixfrac_frame, text=self.parent_gui.tr("mosaic_drizzle_pixfrac_label", default="Pixfrac:"), width=10); pixfrac_label.pack(side=tk.LEFT, padx=(0, 5))
        self.pixfrac_spinbox = ttk.Spinbox(pixfrac_frame, from_=0.01, to=1.00, increment=0.05, textvariable=self.local_drizzle_pixfrac_var, width=7, justify=tk.RIGHT, format="%.2f"); self.pixfrac_spinbox.pack(side=tk.LEFT, padx=5)
        ### FIN AJOUT Kernel/Pixfrac ###

        # --- Boutons OK / Annuler (inchangés) ---
        button_frame = ttk.Frame(main_frame); button_frame.pack(fill=tk.X, pady=(15, 0))
        self.cancel_button = ttk.Button(button_frame, text=self.parent_gui.tr("cancel", default="Cancel"), command=self._cancel); self.cancel_button.pack(side=tk.RIGHT, padx=(5, 0))
        self.ok_button = ttk.Button(button_frame, text=self.parent_gui.tr("ok", default="OK"), command=self._apply_and_close); self.ok_button.pack(side=tk.RIGHT)

        print("DEBUG (MosaicSettingsWindow __init__): Widgets créés.")

        # --- Finalisation ---
        self._on_toggle_activate() # Appliquer état initial (maintenant aussi pour clé API)

        # --- Ajustement Taille Fenêtre ---
        # Utiliser minsize pour s'adapter au contenu
        # Augmenter la hauteur pour accommoder le nouveau cadre API
        self.minsize(450, 320) # Ajuste ces valeurs si nécessaire
        print("DEBUG (MosaicSettingsWindow __init__): Taille minimale définie.")

        self.update_idletasks() # Calculer taille

        # Centrage fenêtre (inchangé)
        # ... (code centrage) ...
        parent_x=self.parent_gui.root.winfo_rootx(); parent_y=self.parent_gui.root.winfo_rooty(); parent_w=self.parent_gui.root.winfo_width(); parent_h=self.parent_gui.root.winfo_height(); win_w = self.winfo_reqwidth(); win_h = self.winfo_reqheight(); x = parent_x + (parent_w // 2) - (win_w // 2); y = parent_y + (parent_h // 2) - (win_h // 2); self.geometry(f"+{x}+{y}")
        print(f"DEBUG (MosaicSettingsWindow __init__): Fenêtre positionnée à {x},{y}.")

        self.deiconify()
        self.focus_force()
        self.grab_set()
        self.wait_window(self)
        print("DEBUG (MosaicSettingsWindow __init__): Fenêtre fermée.")

    ###########################################################################################################################



    def _on_toggle_activate(self):
        """
        Grise/dégrise les options Drizzle ET le cadre Clé API
        en désactivant explicitement les widgets enfants interactifs.
        """
        is_active = self.local_mosaic_active_var.get()
        print(f"DEBUG (MosaicSettingsWindow _on_toggle_activate): Nouvel état Mosaic Active = {is_active}")
        # Déterminer l'état (NORMAL ou DISABLED) pour les options dépendantes
        dependent_options_state = tk.NORMAL if is_active else tk.DISABLED
        # L'état 'disabled' est spécifique pour certains widgets comme Combobox
        combobox_state = tk.NORMAL if is_active else 'disabled'

        # --- Désactiver/Activer les options Drizzle ---
        try:
            # Widgets interactifs
            if hasattr(self, 'kernel_combo'):
                self.kernel_combo.config(state=combobox_state)
            if hasattr(self, 'pixfrac_spinbox'):
                self.pixfrac_spinbox.config(state=dependent_options_state)
            # Labels associés (Optionnel: griser les labels aussi)
            if hasattr(self, 'drizzle_options_frame'):
                for child in self.drizzle_options_frame.winfo_children():
                    if isinstance(child, ttk.Frame):
                        for grandchild in child.winfo_children():
                            if isinstance(grandchild, ttk.Label):
                                grandchild.config(state=dependent_options_state)
            print(f"   -> État options Drizzle (Kernel/Pixfrac) mis à: {dependent_options_state}")
        except tk.TclError:
            print("   -> WARNING: Erreur TclError config état widgets Drizzle.")
        except AttributeError:
            print("   -> WARNING: Widget Drizzle manquant pendant config état.")


        # --- Désactiver/Activer le cadre Clé API ---
        try:
            # Widget interactif
            if hasattr(self, 'api_key_entry'):
                self.api_key_entry.config(state=dependent_options_state)
            # Labels associés (Optionnel: griser les labels aussi)
            if hasattr(self, 'api_key_frame'):
                for child in self.api_key_frame.winfo_children():
                     # Cadre interne contenant le label principal
                     if isinstance(child, ttk.Frame):
                         for grandchild in child.winfo_children():
                             if isinstance(grandchild, ttk.Label):
                                 try: grandchild.config(state=dependent_options_state)
                                 except tk.TclError: pass
                     # Label d'aide direct
                     elif isinstance(child, ttk.Label):
                          try: child.config(state=dependent_options_state)
                          except tk.TclError: pass
            print(f"   -> État cadre Clé API (Entry/Labels) mis à: {dependent_options_state}")
        except tk.TclError:
            print("   -> WARNING: Erreur TclError config état widgets Clé API.")
        except AttributeError:
            print("   -> WARNING: Widget Clé API manquant pendant config état.")

# --- FIN DE LA MÉTHODE _on_toggle_activate  ---
################################################################################################################################




    def _apply_and_close(self):
        """
        Valide les paramètres (y compris clé API si mode mosaïque actif),
        les applique au GUI parent et ferme la fenêtre.
        """
        # Récupérer l'état de la checkbox d'activation
        new_mosaic_state = self.local_mosaic_active_var.get()

        # --- Initialiser les variables pour les settings ---
        selected_kernel = ""
        selected_pixfrac = 1.0 # Valeur défaut sûre
        api_key_value = ""

        # --- Validation SEULEMENT si le mode mosaïque est activé ---
        if new_mosaic_state:
            print("DEBUG (MosaicSettingsWindow _apply_and_close): Mode Mosaïque coché, validation des options...")

            # --- 1. Valider Kernel ---
            selected_kernel = self.local_drizzle_kernel_var.get()
            if selected_kernel not in VALID_DRIZZLE_KERNELS:
                messagebox.showerror(self.parent_gui.tr("error", default="Error"),
                                     self.parent_gui.tr("mosaic_invalid_kernel", default="Invalid Drizzle kernel selected."),
                                     parent=self)
                print("ERREUR Validation: Kernel invalide.")
                return # Ne pas fermer

            # --- 2. Valider Pixfrac ---
            try:
                selected_pixfrac = float(self.local_drizzle_pixfrac_var.get())
                selected_pixfrac = np.clip(selected_pixfrac, 0.01, 1.0) # Assurer dans la plage
                # Optionnel: remettre la valeur clippée dans la variable pour l'affichage
                # self.local_drizzle_pixfrac_var.set(selected_pixfrac)
            except (ValueError, tk.TclError):
                 messagebox.showerror(self.parent_gui.tr("error", default="Error"),
                                      self.parent_gui.tr("mosaic_invalid_pixfrac", default="Invalid Pixfrac value. Must be between 0.01 and 1.0."),
                                      parent=self)
                 print("ERREUR Validation: Pixfrac invalide.")
                 return # Ne pas fermer

            # --- 3. Valider Clé API ---
            api_key_value = self.local_api_key_var.get().strip() # Lire et enlever espaces blancs
            if not api_key_value:
                messagebox.showerror(self.parent_gui.tr("error", default="Error"),
                                     self.parent_gui.tr("mosaic_api_key_required", default="Astrometry.net API Key is required when Mosaic Mode is enabled."),
                                     parent=self)
                print("ERREUR Validation: Clé API manquante.")
                return # Ne pas fermer
            # Optionnel: Ajouter une vérification basique du format de la clé ? (ex: longueur)
            # if len(api_key_value) < 10: # Exemple très simple
            #     messagebox.showwarning(...)
            #     return

            print(f"DEBUG (MosaicSettingsWindow _apply_and_close): Validation OK -> Active: {new_mosaic_state}, Kernel: {selected_kernel}, Pixfrac: {selected_pixfrac:.2f}, Clé API: Présente")

        else: # Si mode mosaïque n'est pas coché
            print(f"DEBUG (MosaicSettingsWindow _apply_and_close): Mode Mosaïque désactivé. Pas de validation des options spécifiques.")
            # On pourrait vouloir récupérer quand même kernel/pixfrac pour les sauvegarder même si inactifs ?
            # Ou les laisser tels quels dans self.parent_gui.mosaic_settings. Pour l'instant, on ne les lit pas si inactif.
            # Lire quand même la clé API pour la sauvegarder même si le mode est inactif
            api_key_value = self.local_api_key_var.get().strip()


        # --- Si toutes les validations sont passées (ou si mode inactif) ---
        try:
            print(f"DEBUG (MosaicSettingsWindow _apply_and_close): Application état mosaïque = {new_mosaic_state} au parent.")
            # Mettre à jour le flag dans l'instance parente
            setattr(self.parent_gui, 'mosaic_mode_active', new_mosaic_state)

            # Mettre à jour les settings spécifiques (kernel/pixfrac) si mode actif
            if new_mosaic_state:
                if not hasattr(self.parent_gui, 'mosaic_settings') or not isinstance(self.parent_gui.mosaic_settings, dict):
                    print("DEBUG (MosaicSettingsWindow): Création dict mosaic_settings sur parent.")
                    setattr(self.parent_gui, 'mosaic_settings', {})
                self.parent_gui.mosaic_settings['kernel'] = selected_kernel
                self.parent_gui.mosaic_settings['pixfrac'] = selected_pixfrac
                print(f"   -> Settings Mosaïque Parent mis à jour: {self.parent_gui.mosaic_settings}")

            # Mettre à jour la clé API dans le parent (TOUJOURS, pour sauvegarde)
            if hasattr(self.parent_gui, 'astrometry_api_key_var'):
                self.parent_gui.astrometry_api_key_var.set(api_key_value)
                print(f"   -> Clé API Parent mise à jour: {'Oui' if api_key_value else 'Non'}")
            else:
                print("   -> WARNING: Attribut 'astrometry_api_key_var' manquant sur parent_gui.")

            # --- Déclencher Sauvegarde Settings ---
            print("   -> Déclenchement sauvegarde settings via parent...")
            self.parent_gui.settings.update_from_ui(self.parent_gui)
            self.parent_gui.settings.save_settings()
            print("   -> Sauvegarde settings terminée.")


            # Log et màj UI parent
            # ... (code log et indicateur UI parent comme avant) ...
            if hasattr(self.parent_gui, 'update_progress_gui'):
                status_msg_key = "mosaic_mode_enabled_log" if new_mosaic_state else "mosaic_mode_disabled_log"
                status_msg_default = f"Mosaic mode {'ENABLED' if new_mosaic_state else 'DISABLED'}."
                self.parent_gui.update_progress_gui(f"ⓘ {self.parent_gui.tr(status_msg_key, default=status_msg_default)}", None)
            if hasattr(self.parent_gui, '_update_mosaic_status_indicator'): self.parent_gui._update_mosaic_status_indicator()

        except Exception as e:
            print(f"ERREUR (MosaicSettingsWindow _apply_and_close): Erreur application état/sauvegarde: {e}")
            messagebox.showerror("Erreur Interne", f"Erreur application paramètres mosaïque:\n{e}", parent=self)
            # On ne ferme PAS la fenêtre en cas d'erreur ici pour pouvoir réessayer
            return

        # --- Fermer la fenêtre si tout s'est bien passé ---
        self.grab_release()
        self.destroy()
        print("DEBUG (MosaicSettingsWindow _apply_and_close): Fenêtre détruite.")

# --- FIN DE LA MÉTHODE _apply_and_close ---


###################################################################################################################################
    def _cancel(self):
        """ Annule et ferme la fenêtre sans appliquer les changements. """
        print("DEBUG (MosaicSettingsWindow _cancel): Annulation, fermeture sans appliquer.")
        self.grab_release()
        self.destroy()

# --- FIN DU FICHIER seestar/gui/mosaic_gui.py ---