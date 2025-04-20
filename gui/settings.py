"""
Module pour la gestion des paramètres de traitement des images astronomiques.
"""

import json
import os

class SettingsManager:
    """
    Classe pour gérer les paramètres de traitement des images.
    """

    def __init__(self):
        """Initialise le gestionnaire de paramètres avec des valeurs par défaut."""
        self.settings_file = "seestar_settings.json" # Default filename
        self.reset_to_defaults()

    def reset_to_defaults(self):
        """ Resets all settings to their default values. """
        self.bayer_pattern = "GRBG"
        self.batch_size = 0 # Auto
        self.reference_image_path = ""
        self.correct_hot_pixels = True
        self.hot_pixel_threshold = 3.0
        self.neighborhood_size = 5
        self.stacking_mode = "kappa-sigma"
        self.kappa = 2.5
        self.denoise = False
        # self.remove_aligned = False # REMOVED - No longer relevant
        self.apply_stretch = True
        self.language = 'en'

    def configure_aligner(self, aligner):
        """ Configure un objet aligneur avec les paramètres actuels. (Less used now)"""
        aligner.bayer_pattern = self.bayer_pattern
        aligner.batch_size = int(self.batch_size)
        aligner.reference_image_path = self.reference_image_path or None
        aligner.correct_hot_pixels = self.correct_hot_pixels
        aligner.hot_pixel_threshold = float(self.hot_pixel_threshold)
        aligner.neighborhood_size = int(self.neighborhood_size)
        if aligner.neighborhood_size % 2 == 0:
             aligner.neighborhood_size += 1

    def update_from_ui(self, gui_instance):
        """ Met à jour les paramètres à partir des variables Tkinter de l'instance GUI. """
        try:
            self.reference_image_path = gui_instance.reference_image_path.get()
            self.stacking_mode = gui_instance.stacking_mode.get()
            self.kappa = gui_instance.kappa.get()
            self.batch_size = gui_instance.batch_size.get()
            # --- REMOVED ---
            # self.remove_aligned = gui_instance.remove_aligned.get()
            # ---
            self.denoise = gui_instance.apply_denoise.get()
            self.correct_hot_pixels = gui_instance.correct_hot_pixels.get()
            self.hot_pixel_threshold = gui_instance.hot_pixel_threshold.get()
            self.neighborhood_size = gui_instance.neighborhood_size.get()
            self.apply_stretch = gui_instance.apply_stretch.get()
            self.language = gui_instance.language_var.get()
        except AttributeError as ae:
            print(f"Error updating settings from UI (likely missing variable): {ae}")
        except Exception as e:
            print(f"Error updating settings from UI: {e}")

    def apply_to_ui(self, gui_instance):
        """ Applique les paramètres chargés/actuels aux variables Tkinter. """
        try:
             gui_instance.reference_image_path.set(self.reference_image_path or "")
             gui_instance.stacking_mode.set(self.stacking_mode)
             gui_instance.kappa.set(self.kappa)
             gui_instance.batch_size.set(self.batch_size)
             # --- REMOVED ---
             # gui_instance.remove_aligned.set(self.remove_aligned)
             # ---
             gui_instance.apply_denoise.set(self.denoise)
             gui_instance.correct_hot_pixels.set(self.correct_hot_pixels)
             gui_instance.hot_pixel_threshold.set(self.hot_pixel_threshold)
             gui_instance.neighborhood_size.set(self.neighborhood_size)
             gui_instance.apply_stretch.set(self.apply_stretch)
             gui_instance.language_var.set(self.language)
        except AttributeError as ae:
             print(f"Error applying settings to UI (likely missing variable): {ae}")
        except Exception as e:
             print(f"Error applying settings to UI: {e}")

    def validate_settings(self):
        """ Valide les paramètres et effectue des ajustements si nécessaire. Retourne les messages d'ajustement. """
        messages = []
        try:
             self.kappa = float(self.kappa)
             if self.kappa <= 0:
                 original = self.kappa; self.kappa = 2.5
                 messages.append(f"Kappa reset: {original} -> {self.kappa}")

             self.batch_size = int(self.batch_size)

             self.hot_pixel_threshold = float(self.hot_pixel_threshold)
             if self.hot_pixel_threshold <= 0:
                 original = self.hot_pixel_threshold; self.hot_pixel_threshold = 3.0
                 messages.append(f"Hot pixel threshold reset: {original} -> {self.hot_pixel_threshold}")

             self.neighborhood_size = int(self.neighborhood_size)
             if self.neighborhood_size < 3:
                  original = self.neighborhood_size; self.neighborhood_size = 3
                  messages.append(f"Neighborhood size increased: {original} -> {self.neighborhood_size}")
             if self.neighborhood_size % 2 == 0:
                 original = self.neighborhood_size; self.neighborhood_size += 1
                 messages.append(f"Neighborhood size adjusted to odd: {original} -> {self.neighborhood_size}")

        except (ValueError, TypeError) as e:
             messages.append(f"Invalid numeric setting detected: {e}. Check spinbox values.")
             # Resetting might be too aggressive here, just report the error.

        return messages

    def save_settings(self, filename=None):
        """ Sauvegarde les paramètres actuels dans un fichier JSON. """
        filepath = filename or self.settings_file
        settings_data = {
            'version': "1.1.1",
            'language': self.language,
            'reference_image_path': self.reference_image_path,
            'stacking_mode': self.stacking_mode,
            'kappa': self.kappa,
            'batch_size': self.batch_size,
            # 'remove_aligned': self.remove_aligned, # REMOVED
            'denoise': self.denoise,
            'correct_hot_pixels': self.correct_hot_pixels,
            'hot_pixel_threshold': self.hot_pixel_threshold,
            'neighborhood_size': self.neighborhood_size,
            'apply_stretch': self.apply_stretch,
        }
        try:
            with open(filepath, 'w') as f: json.dump(settings_data, f, indent=4)
            print(f"Settings saved to {filepath}")
        except Exception as e: print(f"Error saving settings: {e}")

    def load_settings(self, filename=None):
        """ Charge les paramètres depuis un fichier JSON. """
        filepath = filename or self.settings_file
        if not os.path.exists(filepath):
            print(f"Settings file not found: {filepath}. Using defaults.")
            self.reset_to_defaults(); return False
        try:
            with open(filepath, 'r') as f: settings_data = json.load(f)

            self.language = settings_data.get('language', self.language)
            self.reference_image_path = settings_data.get('reference_image_path', self.reference_image_path)
            self.stacking_mode = settings_data.get('stacking_mode', self.stacking_mode)
            self.kappa = settings_data.get('kappa', self.kappa)
            self.batch_size = settings_data.get('batch_size', self.batch_size)
            # self.remove_aligned = settings_data.get('remove_aligned', self.remove_aligned) # REMOVED
            self.denoise = settings_data.get('denoise', self.denoise)
            self.correct_hot_pixels = settings_data.get('correct_hot_pixels', self.correct_hot_pixels)
            self.hot_pixel_threshold = settings_data.get('hot_pixel_threshold', self.hot_pixel_threshold)
            self.neighborhood_size = settings_data.get('neighborhood_size', self.neighborhood_size)
            self.apply_stretch = settings_data.get('apply_stretch', self.apply_stretch)

            print(f"Settings loaded from {filepath}")
            self.validate_settings() # Validate after loading
            return True
        except json.JSONDecodeError as e:
             print(f"Error decoding settings file {filepath}: {e}. Using defaults.")
             self.reset_to_defaults(); return False
        except Exception as e:
            print(f"Error loading settings: {e}. Using defaults.")
            self.reset_to_defaults(); return False