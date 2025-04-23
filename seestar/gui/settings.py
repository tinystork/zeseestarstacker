# --- START OF FILE seestar/gui/settings.py ---
"""
Module pour la gestion des paramètres de traitement et de prévisualisation.
(Version Révisée: Ajout des paramètres de pondération qualité)
"""

import json
import os
import tkinter as tk
import numpy as np
import traceback

class SettingsManager:
    """
    Classe pour gérer les paramètres de traitement, de prévisualisation et de pondération.
    """

    def __init__(self, settings_file="seestar_settings.json"):
        """Initialise le gestionnaire de paramètres avec des valeurs par défaut."""
        self.settings_file = settings_file
        self.reset_to_defaults()

    def reset_to_defaults(self):
        """ Réinitialise tous les paramètres à leurs valeurs par défaut. """
        # Processing Settings
        self.input_folder = ""
        self.output_folder = ""
        self.reference_image_path = ""
        self.bayer_pattern = "GRBG"
        self.batch_size = 0 # 0 = auto
        self.stacking_mode = "kappa-sigma"
        self.kappa = 2.5
        self.correct_hot_pixels = True
        self.hot_pixel_threshold = 3.0
        self.neighborhood_size = 5
        self.cleanup_temp = True

        # *** NEW: Quality Weighting Defaults ***
        self.use_quality_weighting = False # Désactivé par défaut
        self.weight_by_snr = True          # SNR activé si pondération active
        self.weight_by_stars = True        # Nb Etoiles activé si pondération active
        self.snr_exponent = 1.0            # Poids linéaire avec SNR
        self.stars_exponent = 0.5          # Poids racine carrée du score étoiles
        self.min_weight = 0.1              # Poids minimum relatif de 10%

        # Preview Settings
        self.preview_stretch_method = "Asinh"
        self.preview_black_point = 0.01
        self.preview_white_point = 0.99
        self.preview_gamma = 1.0
        self.preview_r_gain = 1.0
        self.preview_g_gain = 1.0
        self.preview_b_gain = 1.0

        # UI Settings
        self.language = 'en'
        self.window_geometry = "1200x750"


    def update_from_ui(self, gui_instance):
        """ Met à jour les paramètres depuis les variables Tkinter de l'interface. """
        if gui_instance is None or not hasattr(gui_instance, 'root') or not gui_instance.root.winfo_exists():
            print("Warning: Cannot update settings from invalid GUI instance.")
            return
        try:
            # --- Processing Settings ---
            self.input_folder = getattr(gui_instance, 'input_path', tk.StringVar()).get()
            self.output_folder = getattr(gui_instance, 'output_path', tk.StringVar()).get()
            self.reference_image_path = getattr(gui_instance, 'reference_image_path', tk.StringVar()).get()
            self.stacking_mode = getattr(gui_instance, 'stacking_mode', tk.StringVar(value=self.stacking_mode)).get()
            self.kappa = getattr(gui_instance, 'kappa', tk.DoubleVar(value=self.kappa)).get()
            self.batch_size = getattr(gui_instance, 'batch_size', tk.IntVar(value=self.batch_size)).get()
            self.correct_hot_pixels = getattr(gui_instance, 'correct_hot_pixels', tk.BooleanVar(value=self.correct_hot_pixels)).get()
            self.hot_pixel_threshold = getattr(gui_instance, 'hot_pixel_threshold', tk.DoubleVar(value=self.hot_pixel_threshold)).get()
            self.neighborhood_size = getattr(gui_instance, 'neighborhood_size', tk.IntVar(value=self.neighborhood_size)).get()
            self.cleanup_temp = getattr(gui_instance, 'cleanup_temp_var', tk.BooleanVar(value=self.cleanup_temp)).get()
            self.bayer_pattern = getattr(gui_instance, 'bayer_pattern', self.bayer_pattern)

            # --- *** NEW: Quality Weighting Settings from UI *** ---
            # Assume UI will have variables like 'use_weighting_var', 'weight_snr_var', etc.
            self.use_quality_weighting = getattr(gui_instance, 'use_weighting_var', tk.BooleanVar(value=self.use_quality_weighting)).get()
            self.weight_by_snr = getattr(gui_instance, 'weight_snr_var', tk.BooleanVar(value=self.weight_by_snr)).get()
            self.weight_by_stars = getattr(gui_instance, 'weight_stars_var', tk.BooleanVar(value=self.weight_by_stars)).get()
            # For exponents/min_weight, maybe use Spinboxes -> DoubleVar/IntVar
            self.snr_exponent = getattr(gui_instance, 'snr_exponent_var', tk.DoubleVar(value=self.snr_exponent)).get()
            self.stars_exponent = getattr(gui_instance, 'stars_exponent_var', tk.DoubleVar(value=self.stars_exponent)).get()
            self.min_weight = getattr(gui_instance, 'min_weight_var', tk.DoubleVar(value=self.min_weight)).get()
            # --- End New ---

            # --- Preview Settings ---
            self.preview_stretch_method = getattr(gui_instance, 'preview_stretch_method', tk.StringVar(value=self.preview_stretch_method)).get()
            self.preview_black_point = getattr(gui_instance, 'preview_black_point', tk.DoubleVar(value=self.preview_black_point)).get()
            self.preview_white_point = getattr(gui_instance, 'preview_white_point', tk.DoubleVar(value=self.preview_white_point)).get()
            self.preview_gamma = getattr(gui_instance, 'preview_gamma', tk.DoubleVar(value=self.preview_gamma)).get()
            self.preview_r_gain = getattr(gui_instance, 'preview_r_gain', tk.DoubleVar(value=self.preview_r_gain)).get()
            self.preview_g_gain = getattr(gui_instance, 'preview_g_gain', tk.DoubleVar(value=self.preview_g_gain)).get()
            self.preview_b_gain = getattr(gui_instance, 'preview_b_gain', tk.DoubleVar(value=self.preview_b_gain)).get()

            # --- UI Settings ---
            self.language = getattr(gui_instance, 'language_var', tk.StringVar(value=self.language)).get()
            current_geo = gui_instance.root.geometry()
            if isinstance(current_geo, str) and 'x' in current_geo and '+' in current_geo:
                 self.window_geometry = current_geo

        except AttributeError as ae: print(f"Error updating settings from UI (AttributeError): {ae}")
        except tk.TclError as te: print(f"Error updating settings from UI (TclError): {te}")
        except Exception as e: print(f"Unexpected error updating settings from UI: {e}"); traceback.print_exc(limit=2)


    def apply_to_ui(self, gui_instance):
        """ Applique les paramètres chargés/actuels aux variables Tkinter. """
        if gui_instance is None or not hasattr(gui_instance, 'root') or not gui_instance.root.winfo_exists():
            print("Warning: Cannot apply settings to invalid GUI instance.")
            return
        try:
            # --- Processing Settings ---
            getattr(gui_instance, 'input_path', tk.StringVar()).set(self.input_folder or "")
            getattr(gui_instance, 'output_path', tk.StringVar()).set(self.output_folder or "")
            getattr(gui_instance, 'reference_image_path', tk.StringVar()).set(self.reference_image_path or "")
            getattr(gui_instance, 'stacking_mode', tk.StringVar()).set(self.stacking_mode)
            getattr(gui_instance, 'kappa', tk.DoubleVar()).set(self.kappa)
            getattr(gui_instance, 'batch_size', tk.IntVar()).set(self.batch_size)
            getattr(gui_instance, 'correct_hot_pixels', tk.BooleanVar()).set(self.correct_hot_pixels)
            getattr(gui_instance, 'hot_pixel_threshold', tk.DoubleVar()).set(self.hot_pixel_threshold)
            getattr(gui_instance, 'neighborhood_size', tk.IntVar()).set(self.neighborhood_size)
            getattr(gui_instance, 'cleanup_temp_var', tk.BooleanVar()).set(self.cleanup_temp)

            # --- *** NEW: Apply Quality Weighting Settings to UI *** ---
            getattr(gui_instance, 'use_weighting_var', tk.BooleanVar()).set(self.use_quality_weighting)
            getattr(gui_instance, 'weight_snr_var', tk.BooleanVar()).set(self.weight_by_snr)
            getattr(gui_instance, 'weight_stars_var', tk.BooleanVar()).set(self.weight_by_stars)
            getattr(gui_instance, 'snr_exponent_var', tk.DoubleVar()).set(self.snr_exponent)
            getattr(gui_instance, 'stars_exponent_var', tk.DoubleVar()).set(self.stars_exponent)
            getattr(gui_instance, 'min_weight_var', tk.DoubleVar()).set(self.min_weight)
            # --- End New ---

            # --- Preview Settings ---
            getattr(gui_instance, 'preview_stretch_method', tk.StringVar()).set(self.preview_stretch_method)
            getattr(gui_instance, 'preview_black_point', tk.DoubleVar()).set(self.preview_black_point)
            getattr(gui_instance, 'preview_white_point', tk.DoubleVar()).set(self.preview_white_point)
            getattr(gui_instance, 'preview_gamma', tk.DoubleVar()).set(self.preview_gamma)
            getattr(gui_instance, 'preview_r_gain', tk.DoubleVar()).set(self.preview_r_gain)
            getattr(gui_instance, 'preview_g_gain', tk.DoubleVar()).set(self.preview_g_gain)
            getattr(gui_instance, 'preview_b_gain', tk.DoubleVar()).set(self.preview_b_gain)

            # --- UI Settings ---
            getattr(gui_instance, 'language_var', tk.StringVar()).set(self.language)

        except AttributeError as ae: print(f"Error applying settings to UI (AttributeError): {ae}")
        except tk.TclError as te: print(f"Error applying settings to UI (TclError - widget likely destroyed?): {te}")
        except Exception as e: print(f"Unexpected error applying settings to UI: {e}"); traceback.print_exc(limit=2)


    def validate_settings(self):
        """Valide et corrige les paramètres si nécessaire. Retourne les messages de correction."""
        messages = []
        try:
             # --- Processing Settings Validation (Identique) ---
             self.kappa = float(self.kappa)
             if not (1.0 <= self.kappa <= 5.0): original = self.kappa; self.kappa = np.clip(self.kappa, 1.0, 5.0); messages.append(f"Kappa ({original:.1f}) ajusté à {self.kappa:.1f}")
             self.batch_size = int(self.batch_size)
             if self.batch_size < 0: original = self.batch_size; self.batch_size = 0; messages.append(f"Taille Lot ({original}) ajusté à {self.batch_size} (auto)")
             self.hot_pixel_threshold = float(self.hot_pixel_threshold)
             if not (0.5 <= self.hot_pixel_threshold <= 10.0): original = self.hot_pixel_threshold; self.hot_pixel_threshold = np.clip(self.hot_pixel_threshold, 0.5, 10.0); messages.append(f"Seuil Px Chauds ({original:.1f}) ajusté à {self.hot_pixel_threshold:.1f}")
             self.neighborhood_size = int(self.neighborhood_size)
             if self.neighborhood_size < 3: original = self.neighborhood_size; self.neighborhood_size = 3; messages.append(f"Voisinage Px Chauds ({original}) ajusté à {self.neighborhood_size}")
             if self.neighborhood_size % 2 == 0: original = self.neighborhood_size; self.neighborhood_size += 1; messages.append(f"Voisinage Px Chauds ({original}) ajusté à {self.neighborhood_size} (impair)")

             # --- *** NEW: Quality Weighting Validation *** ---
             self.use_quality_weighting = bool(self.use_quality_weighting)
             self.weight_by_snr = bool(self.weight_by_snr)
             self.weight_by_stars = bool(self.weight_by_stars)
             self.snr_exponent = float(self.snr_exponent)
             self.stars_exponent = float(self.stars_exponent)
             self.min_weight = float(self.min_weight)
             # Validate exponents (e.g., > 0)
             if self.snr_exponent <= 0: original = self.snr_exponent; self.snr_exponent = 1.0; messages.append(f"Exposant SNR ({original:.1f}) ajusté à {self.snr_exponent:.1f}")
             if self.stars_exponent <= 0: original = self.stars_exponent; self.stars_exponent = 0.5; messages.append(f"Exposant Étoiles ({original:.1f}) ajusté à {self.stars_exponent:.1f}")
             # Validate min_weight (e.g., 0 < min_weight <= 1)
             if not (0 < self.min_weight <= 1.0): original = self.min_weight; self.min_weight = np.clip(self.min_weight, 0.01, 1.0); messages.append(f"Poids Min ({original:.2f}) ajusté à {self.min_weight:.2f}")
             # Ensure at least one metric is selected if weighting is enabled
             if self.use_quality_weighting and not (self.weight_by_snr or self.weight_by_stars):
                 self.weight_by_snr = True # Default to SNR if none selected
                 messages.append("Pondération activée mais aucune métrique choisie. SNR activé par défaut.")
             # --- End New ---

             # --- Preview Settings Validation (Identique) ---
             self.preview_black_point = float(self.preview_black_point); self.preview_white_point = float(self.preview_white_point)
             self.preview_gamma = float(self.preview_gamma)
             self.preview_r_gain = float(self.preview_r_gain); self.preview_g_gain = float(self.preview_g_gain); self.preview_b_gain = float(self.preview_b_gain)
             self.preview_black_point = np.clip(self.preview_black_point, 0.0, 1.0)
             self.preview_white_point = np.clip(self.preview_white_point, 0.0, 1.0)
             if self.preview_black_point >= self.preview_white_point: self.preview_black_point = max(0.0, self.preview_white_point - 0.001)
             self.preview_gamma = np.clip(self.preview_gamma, 0.1, 5.0)
             gain_min, gain_max = 0.1, 10.0
             self.preview_r_gain = np.clip(self.preview_r_gain, gain_min, gain_max)
             self.preview_g_gain = np.clip(self.preview_g_gain, gain_min, gain_max)
             self.preview_b_gain = np.clip(self.preview_b_gain, gain_min, gain_max)
             valid_methods = ["Linear", "Asinh", "Log"]
             if self.preview_stretch_method not in valid_methods: self.preview_stretch_method = "Asinh"; messages.append(f"Méthode d'étirement invalide, réinitialisée à 'Asinh'")

        except (ValueError, TypeError) as e:
             messages.append(f"Paramètre numérique invalide détecté: {e}. Vérifiez les valeurs.")
             print(f"Warning: Validation error likely due to invalid number input: {e}")

        return messages


    def save_settings(self):
        """ Sauvegarde les paramètres actuels dans le fichier JSON. """
        settings_data = {
            'version': "1.2.0", # Keep version consistent
            # Processing
            'input_folder': str(self.input_folder),
            'output_folder': str(self.output_folder),
            'reference_image_path': str(self.reference_image_path),
            'bayer_pattern': str(self.bayer_pattern),
            'stacking_mode': str(self.stacking_mode),
            'kappa': float(self.kappa),
            'batch_size': int(self.batch_size),
            'correct_hot_pixels': bool(self.correct_hot_pixels),
            'hot_pixel_threshold': float(self.hot_pixel_threshold),
            'neighborhood_size': int(self.neighborhood_size),
            'cleanup_temp': bool(self.cleanup_temp),
            # *** NEW: Quality Weighting Settings ***
            'use_quality_weighting': bool(self.use_quality_weighting),
            'weight_by_snr': bool(self.weight_by_snr),
            'weight_by_stars': bool(self.weight_by_stars),
            'snr_exponent': float(self.snr_exponent),
            'stars_exponent': float(self.stars_exponent),
            'min_weight': float(self.min_weight),
            # --- End New ---
            # Preview
            'preview_stretch_method': str(self.preview_stretch_method),
            'preview_black_point': float(self.preview_black_point),
            'preview_white_point': float(self.preview_white_point),
            'preview_gamma': float(self.preview_gamma),
            'preview_r_gain': float(self.preview_r_gain),
            'preview_g_gain': float(self.preview_g_gain),
            'preview_b_gain': float(self.preview_b_gain),
            # UI
            'language': str(self.language),
            'window_geometry': str(self.window_geometry),
        }
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                 json.dump(settings_data, f, indent=4, ensure_ascii=False)
        except TypeError as te: print(f"Error saving settings: Data not JSON serializable - {te}")
        except IOError as ioe: print(f"Error saving settings: I/O error writing to {self.settings_file} - {ioe}")
        except Exception as e: print(f"Unexpected error saving settings: {e}")


    def load_settings(self):
        """ Charge les paramètres depuis le fichier JSON. """
        if not os.path.exists(self.settings_file):
            print(f"Settings file not found: {self.settings_file}. Using defaults and creating file.")
            self.reset_to_defaults()
            self.save_settings()
            return False
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                 settings_data = json.load(f)

            # Load settings, using current values as defaults if key is missing
            # Processing (Identique)
            self.input_folder = settings_data.get('input_folder', self.input_folder)
            self.output_folder = settings_data.get('output_folder', self.output_folder)
            self.reference_image_path = settings_data.get('reference_image_path', self.reference_image_path)
            self.bayer_pattern = settings_data.get('bayer_pattern', self.bayer_pattern)
            self.stacking_mode = settings_data.get('stacking_mode', self.stacking_mode)
            self.kappa = settings_data.get('kappa', self.kappa)
            self.batch_size = settings_data.get('batch_size', self.batch_size)
            self.correct_hot_pixels = settings_data.get('correct_hot_pixels', self.correct_hot_pixels)
            self.hot_pixel_threshold = settings_data.get('hot_pixel_threshold', self.hot_pixel_threshold)
            self.neighborhood_size = settings_data.get('neighborhood_size', self.neighborhood_size)
            self.cleanup_temp = settings_data.get('cleanup_temp', self.cleanup_temp)

            # --- *** NEW: Load Quality Weighting Settings *** ---
            self.use_quality_weighting = settings_data.get('use_quality_weighting', self.use_quality_weighting)
            self.weight_by_snr = settings_data.get('weight_by_snr', self.weight_by_snr)
            self.weight_by_stars = settings_data.get('weight_by_stars', self.weight_by_stars)
            self.snr_exponent = settings_data.get('snr_exponent', self.snr_exponent)
            self.stars_exponent = settings_data.get('stars_exponent', self.stars_exponent)
            self.min_weight = settings_data.get('min_weight', self.min_weight)
            # --- End New ---

            # Preview (Identique)
            self.preview_stretch_method = settings_data.get('preview_stretch_method', self.preview_stretch_method)
            self.preview_black_point = settings_data.get('preview_black_point', self.preview_black_point)
            self.preview_white_point = settings_data.get('preview_white_point', self.preview_white_point)
            self.preview_gamma = settings_data.get('preview_gamma', self.preview_gamma)
            self.preview_r_gain = settings_data.get('preview_r_gain', self.preview_r_gain)
            self.preview_g_gain = settings_data.get('preview_g_gain', self.preview_g_gain)
            self.preview_b_gain = settings_data.get('preview_b_gain', self.preview_b_gain)

            # UI (Identique)
            self.language = settings_data.get('language', self.language)
            self.window_geometry = settings_data.get('window_geometry', self.window_geometry)

            print(f"Settings loaded from {self.settings_file}")
            # Validate settings after loading
            validation_messages = self.validate_settings()
            if validation_messages:
                 print("Loaded settings adjusted after validation:")
                 for msg in validation_messages: print(f"  - {msg}")
                 self.save_settings() # Save the corrected settings

            return True

        except json.JSONDecodeError as e:
            print(f"Error decoding settings file {self.settings_file}: {e}. Using defaults.")
            self.reset_to_defaults(); return False
        except Exception as e:
            print(f"Error loading settings: {e}. Using defaults.")
            traceback.print_exc(limit=2)
            self.reset_to_defaults(); return False
# --- END OF FILE seestar/gui/settings.py ---