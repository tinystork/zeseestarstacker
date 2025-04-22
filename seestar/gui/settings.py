# --- START OF FILE seestar/gui/settings.py ---
"""
Module pour la gestion des paramètres de traitement et de prévisualisation.
"""

import json
import os
import tkinter as tk # Need for checking tk variable types
import numpy as np # For clipping in validation
import traceback # For error reporting

class SettingsManager:
    """
    Classe pour gérer les paramètres de traitement et de prévisualisation.
    """

    def __init__(self, settings_file="seestar_settings.json"):
        """Initialise le gestionnaire de paramètres avec des valeurs par défaut."""
        self.settings_file = settings_file
        self.reset_to_defaults()

    def reset_to_defaults(self):
        """ Réinitialise tous les paramètres à leurs valeurs par défaut. """
        # Processing Settings
        self.input_folder = "" # Added to store last used paths
        self.output_folder = "" # Added
        self.reference_image_path = ""
        self.bayer_pattern = "GRBG" # Default for Seestar S50
        self.batch_size = 0 # 0 = auto
        self.stacking_mode = "kappa-sigma"
        self.kappa = 2.5
        self.correct_hot_pixels = True
        self.hot_pixel_threshold = 3.0
        self.neighborhood_size = 5
        self.cleanup_temp = True

        # Preview Settings (New defaults)
        self.preview_stretch_method = "Asinh" # Default stretch method
        self.preview_black_point = 0.01       # Default slightly raised black point
        self.preview_white_point = 0.99       # Default slightly lowered white point
        self.preview_gamma = 1.0
        self.preview_r_gain = 1.0
        self.preview_g_gain = 1.0
        self.preview_b_gain = 1.0

        # UI Settings
        self.language = 'en'
        self.window_geometry = "1200x750" # Default window size/pos


    def update_from_ui(self, gui_instance):
        """ Met à jour les paramètres depuis les variables Tkinter de l'interface. """
        # Check if gui_instance and its root window exist
        if gui_instance is None or not hasattr(gui_instance, 'root') or not gui_instance.root.winfo_exists():
            print("Warning: Cannot update settings from invalid GUI instance.")
            return
        try:
            # --- Processing Settings ---
            # Use .get() safely with checks for attribute existence
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
            # Bayer pattern might not be directly on UI, keep existing value if not present
            self.bayer_pattern = getattr(gui_instance, 'bayer_pattern', self.bayer_pattern) # Assuming not directly settable on UI for now


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
            # Update geometry only if window exists and seems valid
            current_geo = gui_instance.root.geometry()
            if isinstance(current_geo, str) and 'x' in current_geo and '+' in current_geo:
                 self.window_geometry = current_geo

        except AttributeError as ae: print(f"Error updating settings from UI (AttributeError): {ae}")
        except tk.TclError as te: print(f"Error updating settings from UI (TclError): {te}") # Catch potential Tk errors accessing variables
        except Exception as e: print(f"Unexpected error updating settings from UI: {e}"); traceback.print_exc(limit=2)


    def apply_to_ui(self, gui_instance):
        """ Applique les paramètres chargés/actuels aux variables Tkinter. """
        if gui_instance is None or not hasattr(gui_instance, 'root') or not gui_instance.root.winfo_exists():
            print("Warning: Cannot apply settings to invalid GUI instance.")
            return
        try:
            # --- Processing Settings ---
            # Use getattr to safely access UI variables, providing a dummy if not found
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
            # Geometry is typically applied when the window is created or by calling root.geometry()
            # We store it here, main_window will use it on init.

        except AttributeError as ae: print(f"Error applying settings to UI (AttributeError): {ae}")
        except tk.TclError as te: print(f"Error applying settings to UI (TclError - widget likely destroyed?): {te}")
        except Exception as e: print(f"Unexpected error applying settings to UI: {e}"); traceback.print_exc(limit=2)


    def validate_settings(self):
        """Valide et corrige les paramètres si nécessaire. Retourne les messages de correction."""
        messages = []
        try:
             # --- Processing Settings Validation ---
             # Kappa
             self.kappa = float(self.kappa)
             if not (1.0 <= self.kappa <= 5.0):
                 original = self.kappa; self.kappa = np.clip(self.kappa, 1.0, 5.0); messages.append(f"Kappa ({original:.1f}) ajusté à {self.kappa:.1f}")
             # Batch Size
             self.batch_size = int(self.batch_size)
             if self.batch_size < 0:
                 original = self.batch_size; self.batch_size = 0; messages.append(f"Taille Lot ({original}) ajusté à {self.batch_size} (auto)")
             # Hot Pixel Threshold
             self.hot_pixel_threshold = float(self.hot_pixel_threshold)
             if not (0.5 <= self.hot_pixel_threshold <= 10.0):
                  original = self.hot_pixel_threshold; self.hot_pixel_threshold = np.clip(self.hot_pixel_threshold, 0.5, 10.0); messages.append(f"Seuil Px Chauds ({original:.1f}) ajusté à {self.hot_pixel_threshold:.1f}")
             # Neighborhood Size
             self.neighborhood_size = int(self.neighborhood_size)
             if self.neighborhood_size < 3:
                  original = self.neighborhood_size; self.neighborhood_size = 3; messages.append(f"Voisinage Px Chauds ({original}) ajusté à {self.neighborhood_size}")
             if self.neighborhood_size % 2 == 0:
                  original = self.neighborhood_size; self.neighborhood_size += 1; messages.append(f"Voisinage Px Chauds ({original}) ajusté à {self.neighborhood_size} (impair)")

             # --- Preview Settings Validation ---
             self.preview_black_point = float(self.preview_black_point); self.preview_white_point = float(self.preview_white_point)
             self.preview_gamma = float(self.preview_gamma)
             self.preview_r_gain = float(self.preview_r_gain); self.preview_g_gain = float(self.preview_g_gain); self.preview_b_gain = float(self.preview_b_gain)

             # Clip preview values to reasonable ranges
             self.preview_black_point = np.clip(self.preview_black_point, 0.0, 1.0)
             self.preview_white_point = np.clip(self.preview_white_point, 0.0, 1.0)
             # Ensure bp < wp
             if self.preview_black_point >= self.preview_white_point: self.preview_black_point = max(0.0, self.preview_white_point - 0.001)

             self.preview_gamma = np.clip(self.preview_gamma, 0.1, 5.0) # Allow wider gamma range
             gain_min, gain_max = 0.1, 10.0 # Range for gains
             self.preview_r_gain = np.clip(self.preview_r_gain, gain_min, gain_max)
             self.preview_g_gain = np.clip(self.preview_g_gain, gain_min, gain_max)
             self.preview_b_gain = np.clip(self.preview_b_gain, gain_min, gain_max)

             # Validate stretch method
             valid_methods = ["Linear", "Asinh", "Log"]
             if self.preview_stretch_method not in valid_methods:
                  self.preview_stretch_method = "Asinh" # Default back to Asinh
                  messages.append(f"Méthode d'étirement invalide, réinitialisée à 'Asinh'")

        except (ValueError, TypeError) as e:
             messages.append(f"Paramètre numérique invalide détecté: {e}. Vérifiez les valeurs.")
             # Reset related fields to default to prevent crashes? Or just warn? Just warn for now.
             print(f"Warning: Validation error likely due to invalid number input: {e}")

        return messages


    def save_settings(self):
        """ Sauvegarde les paramètres actuels dans le fichier JSON. """
        # Ensure all settings are serializable (convert complex types if any)
        settings_data = {
            'version': "1.2.0", # Match package version
            # Processing
            'input_folder': str(self.input_folder), # Ensure string
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
            # print(f"Settings saved to {self.settings_file}") # Optional debug print
        except TypeError as te:
             print(f"Error saving settings: Data not JSON serializable - {te}")
        except IOError as ioe:
             print(f"Error saving settings: I/O error writing to {self.settings_file} - {ioe}")
        except Exception as e:
             print(f"Unexpected error saving settings: {e}")


    def load_settings(self):
        """ Charge les paramètres depuis le fichier JSON. """
        if not os.path.exists(self.settings_file):
            print(f"Settings file not found: {self.settings_file}. Using defaults and creating file.")
            self.reset_to_defaults()
            self.save_settings() # Create file with defaults
            return False
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                 settings_data = json.load(f)

            # Load settings, using current values as defaults if key is missing
            # Use .get() for safe access
            # Processing
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

            # Preview
            self.preview_stretch_method = settings_data.get('preview_stretch_method', self.preview_stretch_method)
            self.preview_black_point = settings_data.get('preview_black_point', self.preview_black_point)
            self.preview_white_point = settings_data.get('preview_white_point', self.preview_white_point)
            self.preview_gamma = settings_data.get('preview_gamma', self.preview_gamma)
            self.preview_r_gain = settings_data.get('preview_r_gain', self.preview_r_gain)
            self.preview_g_gain = settings_data.get('preview_g_gain', self.preview_g_gain)
            self.preview_b_gain = settings_data.get('preview_b_gain', self.preview_b_gain)

            # UI
            self.language = settings_data.get('language', self.language)
            self.window_geometry = settings_data.get('window_geometry', self.window_geometry)


            print(f"Settings loaded from {self.settings_file}")
            # Validate settings after loading to ensure they are within sensible ranges
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