# --- START OF FILE seestar/gui/settings.py ---
"""
Module pour la gestion des paramètres de traitement, de prévisualisation,
de pondération qualité et Drizzle.
"""

import json
import os
import tkinter as tk
import numpy as np
import traceback

class SettingsManager:
    """
    Classe pour gérer les paramètres de l'application, y compris
    le traitement, la prévisualisation, la pondération et Drizzle.
    """

    # Utiliser le nom de fichier standard
    SETTINGS_FILENAME = "seestar_settings.json"

    def __init__(self, settings_file=SETTINGS_FILENAME):
        """Initialise le gestionnaire de paramètres avec des valeurs par défaut."""
        self.settings_file = settings_file
        self.reset_to_defaults() # Initialiser avec les valeurs par défaut

    def reset_to_defaults(self):
        """ Réinitialise tous les paramètres à leurs valeurs par défaut. """
        # Processing Settings
        self.input_folder = ""
        self.output_folder = ""
        self.reference_image_path = ""
        self.bayer_pattern = "GRBG" # Valeur Seestar typique
        self.batch_size = 0 # 0 = auto (sera estimé)
        self.stacking_mode = "kappa-sigma" # Défaut robuste
        self.kappa = 2.5
        self.correct_hot_pixels = True
        self.hot_pixel_threshold = 3.0
        self.neighborhood_size = 5 # Doit être impair
        self.cleanup_temp = True

        # Quality Weighting Settings
        self.use_quality_weighting = False # Désactivé par défaut
        self.weight_by_snr = True
        self.weight_by_stars = True
        self.snr_exponent = 1.0
        self.stars_exponent = 0.5
        self.min_weight = 0.1 # Poids minimum relatif

        # --- Drizzle Settings ---
        self.use_drizzle = False        # Drizzle désactivé par défaut
        self.drizzle_scale = 2          # Échelle (sera int 2, 3, ou 4)
        self.drizzle_wht_threshold = 0.7 # Seuil pour masque WHT (70%)
        self.drizzle_mode = "Final"
        self.drizzle_kernel = "square"  # Noyau Drizzle ('square', 'gaussian', etc.)
        self.drizzle_pixfrac = 1.0      # Fraction Pixel Drizzle (0.01 - 1.0)

        # Preview Settings
        self.preview_stretch_method = "Asinh" # Bon défaut pour l'astro
        self.preview_black_point = 0.01
        self.preview_white_point = 0.99
        self.preview_gamma = 1.0
        self.preview_r_gain = 1.0
        self.preview_g_gain = 1.0
        self.preview_b_gain = 1.0

        # UI Settings
        self.language = 'en' # Langue par défaut
        self.window_geometry = "1200x750" # Taille fenêtre par défaut

        # Retourner les défauts peut être utile dans certains cas
        return self.__dict__

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
            # Note: bayer_pattern n'est pas modifié par l'UI dans cette version

            # --- Quality Weighting Settings ---
            self.use_quality_weighting = getattr(gui_instance, 'use_weighting_var', tk.BooleanVar(value=self.use_quality_weighting)).get()
            self.weight_by_snr = getattr(gui_instance, 'weight_snr_var', tk.BooleanVar(value=self.weight_by_snr)).get()
            self.weight_by_stars = getattr(gui_instance, 'weight_stars_var', tk.BooleanVar(value=self.weight_by_stars)).get()
            self.snr_exponent = getattr(gui_instance, 'snr_exponent_var', tk.DoubleVar(value=self.snr_exponent)).get()
            self.stars_exponent = getattr(gui_instance, 'stars_exponent_var', tk.DoubleVar(value=self.stars_exponent)).get()
            self.min_weight = getattr(gui_instance, 'min_weight_var', tk.DoubleVar(value=self.min_weight)).get()

            # --- Drizzle Settings ---
            self.use_drizzle = getattr(gui_instance, 'use_drizzle_var', tk.BooleanVar(value=self.use_drizzle)).get()
            # Lire l'échelle Drizzle (string) et convertir en int
            scale_str = getattr(gui_instance, 'drizzle_scale_var', tk.StringVar(value=str(self.drizzle_scale))).get()
            try:
                self.drizzle_scale = int(float(scale_str)) # Convertir en float puis int
            except ValueError:
                print(f"Warning: Invalid Drizzle scale value '{scale_str}' from UI. Using default {self.reset_to_defaults()['drizzle_scale']}.")
                self.drizzle_scale = self.reset_to_defaults()['drizzle_scale']
            # Lire le seuil WHT
            self.drizzle_wht_threshold = getattr(gui_instance, 'drizzle_wht_threshold_var', tk.DoubleVar(value=self.drizzle_wht_threshold)).get()
            self.drizzle_mode = getattr(gui_instance, 'drizzle_mode_var', tk.StringVar(value=self.drizzle_mode)).get()
            self.drizzle_kernel = getattr(gui_instance, 'drizzle_kernel_var', tk.StringVar(value=self.drizzle_kernel)).get()
            self.drizzle_pixfrac = getattr(gui_instance, 'drizzle_pixfrac_var', tk.DoubleVar(value=self.drizzle_pixfrac)).get()

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
            # Sauvegarder la géométrie seulement si la fenêtre existe
            if gui_instance.root.winfo_exists():
                 current_geo = gui_instance.root.geometry()
                 # Vérifier que la géométrie est valide avant de sauvegarder
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

            # --- Quality Weighting Settings ---
            getattr(gui_instance, 'use_weighting_var', tk.BooleanVar()).set(self.use_quality_weighting)
            getattr(gui_instance, 'weight_snr_var', tk.BooleanVar()).set(self.weight_by_snr)
            getattr(gui_instance, 'weight_stars_var', tk.BooleanVar()).set(self.weight_by_stars)
            getattr(gui_instance, 'snr_exponent_var', tk.DoubleVar()).set(self.snr_exponent)
            getattr(gui_instance, 'stars_exponent_var', tk.DoubleVar()).set(self.stars_exponent)
            getattr(gui_instance, 'min_weight_var', tk.DoubleVar()).set(self.min_weight)

            # --- Drizzle Settings ---
            getattr(gui_instance, 'use_drizzle_var', tk.BooleanVar()).set(self.use_drizzle)
            # Convertir l'échelle numérique en string pour la variable UI
            getattr(gui_instance, 'drizzle_scale_var', tk.StringVar()).set(str(self.drizzle_scale))
            getattr(gui_instance, 'drizzle_wht_threshold_var', tk.DoubleVar()).set(self.drizzle_wht_threshold)
            getattr(gui_instance, 'drizzle_mode_var', tk.StringVar()).set(self.drizzle_mode)
            getattr(gui_instance, 'drizzle_kernel_var', tk.StringVar()).set(self.drizzle_kernel)
            getattr(gui_instance, 'drizzle_pixfrac_var', tk.DoubleVar()).set(self.drizzle_pixfrac)

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
            # Appliquer la géométrie seulement si elle est valide
            if isinstance(self.window_geometry, str) and 'x' in self.window_geometry and '+' in self.window_geometry:
                try:
                    gui_instance.root.geometry(self.window_geometry)
                except tk.TclError: # Ignorer si la géométrie est invalide au moment de l'appliquer
                    print(f"Warning: Could not apply window geometry '{self.window_geometry}'.")

            # --- Mettre à jour l'état des widgets dépendants ---
            if hasattr(gui_instance, '_update_weighting_options_state'):
                gui_instance._update_weighting_options_state()
            if hasattr(gui_instance, '_update_drizzle_options_state'):
                gui_instance._update_drizzle_options_state()

        except AttributeError as ae: print(f"Error applying settings to UI (AttributeError): {ae}")
        except tk.TclError as te: print(f"Error applying settings to UI (TclError - widget likely destroyed?): {te}")
        except Exception as e: print(f"Unexpected error applying settings to UI: {e}"); traceback.print_exc(limit=2)


    def validate_settings(self):
        """Valide et corrige les paramètres si nécessaire. Retourne les messages de correction."""
        messages = []
        defaults = self.reset_to_defaults() # Obtenir les valeurs par défaut pour fallback

        try:
             # --- Processing Settings Validation ---
             self.kappa = float(self.kappa)
             if not (1.0 <= self.kappa <= 5.0): original = self.kappa; self.kappa = np.clip(self.kappa, 1.0, 5.0); messages.append(f"Kappa ({original:.1f}) ajusté à {self.kappa:.1f}")
             self.batch_size = int(self.batch_size)
             if self.batch_size < 0: original = self.batch_size; self.batch_size = 0; messages.append(f"Taille Lot ({original}) ajusté à {self.batch_size} (auto)")
             self.hot_pixel_threshold = float(self.hot_pixel_threshold)
             if not (0.5 <= self.hot_pixel_threshold <= 10.0): original = self.hot_pixel_threshold; self.hot_pixel_threshold = np.clip(self.hot_pixel_threshold, 0.5, 10.0); messages.append(f"Seuil Px Chauds ({original:.1f}) ajusté à {self.hot_pixel_threshold:.1f}")
             self.neighborhood_size = int(self.neighborhood_size)
             if self.neighborhood_size < 3: original = self.neighborhood_size; self.neighborhood_size = 3; messages.append(f"Voisinage Px Chauds ({original}) ajusté à {self.neighborhood_size}")
             if self.neighborhood_size % 2 == 0: original = self.neighborhood_size; self.neighborhood_size += 1; messages.append(f"Voisinage Px Chauds ({original}) ajusté à {self.neighborhood_size} (impair)")

             # --- Quality Weighting Validation ---
             self.use_quality_weighting = bool(self.use_quality_weighting)
             self.weight_by_snr = bool(self.weight_by_snr)
             self.weight_by_stars = bool(self.weight_by_stars)
             self.snr_exponent = float(self.snr_exponent)
             self.stars_exponent = float(self.stars_exponent)
             self.min_weight = float(self.min_weight)
             if self.snr_exponent <= 0: original = self.snr_exponent; self.snr_exponent = defaults['snr_exponent']; messages.append(f"Exposant SNR ({original:.1f}) ajusté à {self.snr_exponent:.1f}")
             if self.stars_exponent <= 0: original = self.stars_exponent; self.stars_exponent = defaults['stars_exponent']; messages.append(f"Exposant Étoiles ({original:.1f}) ajusté à {self.stars_exponent:.1f}")
             if not (0 < self.min_weight <= 1.0): original = self.min_weight; self.min_weight = np.clip(self.min_weight, 0.01, 1.0); messages.append(f"Poids Min ({original:.2f}) ajusté à {self.min_weight:.2f}")
             if self.use_quality_weighting and not (self.weight_by_snr or self.weight_by_stars):
                 self.weight_by_snr = True; messages.append("Pondération activée mais aucune métrique choisie. SNR activé par défaut.")

             # --- Drizzle Settings Validation ---
             self.use_drizzle = bool(self.use_drizzle)
             try:
                 scale_num = int(float(self.drizzle_scale)) # Essayer conversion int
                 if scale_num not in [2, 3, 4]:
                     original = self.drizzle_scale; self.drizzle_scale = defaults['drizzle_scale']; messages.append(f"Échelle Drizzle ({original}) invalide, réinitialisée à {self.drizzle_scale}")
                 else: self.drizzle_scale = scale_num # Stocker comme int si valide
             except (ValueError, TypeError):
                 original = self.drizzle_scale; self.drizzle_scale = defaults['drizzle_scale']; messages.append(f"Échelle Drizzle invalide ({original}), réinitialisée à {self.drizzle_scale}")
             try:
                 self.drizzle_wht_threshold = float(self.drizzle_wht_threshold)
                 if not (0.0 < self.drizzle_wht_threshold <= 1.0): # Doit être > 0 et <= 1
                      original = self.drizzle_wht_threshold; self.drizzle_wht_threshold = np.clip(self.drizzle_wht_threshold, 0.1, 1.0); messages.append(f"Seuil Drizzle WHT ({original:.2f}) ajusté à {self.drizzle_wht_threshold:.2f}")
             except (ValueError, TypeError):
                 original = self.drizzle_wht_threshold; self.drizzle_wht_threshold = defaults['drizzle_wht_threshold']; messages.append(f"Seuil Drizzle WHT invalide ({original}), réinitialisé à {self.drizzle_wht_threshold:.2f}")
            # Drizzle Mode Validation
                 valid_drizzle_modes = ["Final", "Incremental"]
                 if not isinstance(self.drizzle_mode, str) or self.drizzle_mode not in valid_drizzle_modes:
                  original = self.drizzle_mode
                 self.drizzle_mode = defaults['drizzle_mode'] # Reset to default 'Final'
                 messages.append(f"Mode Drizzle ({original}) invalide, réinitialisé à '{self.drizzle_mode}'")
            # --->>> FIN DU BLOC À INSÉRER <<<---

            # Drizzle Kernel and Pixfrac Validation
             valid_kernels = ['square', 'gaussian', 'point', 'tophat', 'turbo', 'lanczos2', 'lanczos3']
             if not isinstance(self.drizzle_kernel, str) or self.drizzle_kernel.lower() not in valid_kernels:
                original = self.drizzle_kernel
                self.drizzle_kernel = defaults['drizzle_kernel'] # Reset to default 'square'
                messages.append(f"Noyau Drizzle ('{original}') invalide, réinitialisé à '{self.drizzle_kernel}'")
             else:
                # S'assurer qu'il est en minuscule pour la comparaison future
                self.drizzle_kernel = self.drizzle_kernel.lower()

             try:
                 self.drizzle_pixfrac = float(self.drizzle_pixfrac)
                 if not (0.01 <= self.drizzle_pixfrac <= 1.0):
                      original = self.drizzle_pixfrac
                      self.drizzle_pixfrac = np.clip(self.drizzle_pixfrac, 0.01, 1.0)
                      messages.append(f"Pixfrac Drizzle ({original:.2f}) hors limites [0.01, 1.0], ajusté à {self.drizzle_pixfrac:.2f}")
             except (ValueError, TypeError):
                 original = self.drizzle_pixfrac
                 self.drizzle_pixfrac = defaults['drizzle_pixfrac'] # Reset to default 1.0
                 messages.append(f"Pixfrac Drizzle ('{original}') invalide, réinitialisé à {self.drizzle_pixfrac:.2f}")
            # --- FIN NOUVEAU BLOC ---

             # --- Preview Settings Validation ---
             self.preview_black_point = float(self.preview_black_point); self.preview_white_point = float(self.preview_white_point)
             self.preview_gamma = float(self.preview_gamma)
             self.preview_r_gain = float(self.preview_r_gain); self.preview_g_gain = float(self.preview_g_gain); self.preview_b_gain = float(self.preview_b_gain)
             min_preview, max_preview = 0.0, 1.0
             self.preview_black_point = np.clip(self.preview_black_point, min_preview, max_preview)
             self.preview_white_point = np.clip(self.preview_white_point, min_preview, max_preview)
             if self.preview_black_point >= self.preview_white_point: self.preview_black_point = max(min_preview, self.preview_white_point - 0.001)
             self.preview_gamma = np.clip(self.preview_gamma, 0.1, 5.0)
             gain_min, gain_max = 0.1, 10.0
             self.preview_r_gain = np.clip(self.preview_r_gain, gain_min, gain_max)
             self.preview_g_gain = np.clip(self.preview_g_gain, gain_min, gain_max)
             self.preview_b_gain = np.clip(self.preview_b_gain, gain_min, gain_max)
             valid_methods = ["Linear", "Asinh", "Log"]
             if self.preview_stretch_method not in valid_methods: self.preview_stretch_method = defaults['preview_stretch_method']; messages.append(f"Méthode d'étirement invalide, réinitialisée à '{self.preview_stretch_method}'")

        except (ValueError, TypeError) as e:
             messages.append(f"Paramètre numérique invalide détecté: {e}. Vérifiez les valeurs.")
             print(f"Warning: Validation error likely due to invalid number input: {e}")
             # Optionnel: réinitialiser plus de paramètres aux défauts ici?
             self.reset_to_defaults()

        return messages


    def save_settings(self):
        """ Sauvegarde les paramètres actuels dans le fichier JSON. """
        # S'assurer que les types sont corrects pour JSON
        settings_data = {
            'version': "1.3.0", # Incrémenter la version si la structure change significativement
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
            # Quality Weighting
            'use_quality_weighting': bool(self.use_quality_weighting),
            'weight_by_snr': bool(self.weight_by_snr),
            'weight_by_stars': bool(self.weight_by_stars),
            'snr_exponent': float(self.snr_exponent),
            'stars_exponent': float(self.stars_exponent),
            'min_weight': float(self.min_weight),
            # Drizzle
            'use_drizzle': bool(self.use_drizzle),
            'drizzle_scale': int(self.drizzle_scale), # Stocker comme int
            'drizzle_wht_threshold': float(self.drizzle_wht_threshold),
            'drizzle_mode': str(self.drizzle_mode),
            'drizzle_kernel': str(self.drizzle_kernel),
            'drizzle_pixfrac': float(self.drizzle_pixfrac),
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
            self.save_settings() # Créer le fichier avec les défauts
            return False # Indiquer que le fichier n'existait pas

        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                 settings_data = json.load(f)

            defaults = self.reset_to_defaults() # Obtenir les défauts pour fallback

            # Load settings, using current values (defaults) as fallback if key is missing
            # Processing
            self.input_folder = settings_data.get('input_folder', defaults['input_folder'])
            self.output_folder = settings_data.get('output_folder', defaults['output_folder'])
            self.reference_image_path = settings_data.get('reference_image_path', defaults['reference_image_path'])
            self.bayer_pattern = settings_data.get('bayer_pattern', defaults['bayer_pattern'])
            self.stacking_mode = settings_data.get('stacking_mode', defaults['stacking_mode'])
            self.kappa = settings_data.get('kappa', defaults['kappa'])
            self.batch_size = settings_data.get('batch_size', defaults['batch_size'])
            self.correct_hot_pixels = settings_data.get('correct_hot_pixels', defaults['correct_hot_pixels'])
            self.hot_pixel_threshold = settings_data.get('hot_pixel_threshold', defaults['hot_pixel_threshold'])
            self.neighborhood_size = settings_data.get('neighborhood_size', defaults['neighborhood_size'])
            self.cleanup_temp = settings_data.get('cleanup_temp', defaults['cleanup_temp'])

            # Quality Weighting
            self.use_quality_weighting = settings_data.get('use_quality_weighting', defaults['use_quality_weighting'])
            self.weight_by_snr = settings_data.get('weight_by_snr', defaults['weight_by_snr'])
            self.weight_by_stars = settings_data.get('weight_by_stars', defaults['weight_by_stars'])
            self.snr_exponent = settings_data.get('snr_exponent', defaults['snr_exponent'])
            self.stars_exponent = settings_data.get('stars_exponent', defaults['stars_exponent'])
            self.min_weight = settings_data.get('min_weight', defaults['min_weight'])

            # --- Drizzle Settings ---
            self.use_drizzle = settings_data.get('use_drizzle', defaults['use_drizzle'])
            self.drizzle_scale = settings_data.get('drizzle_scale', defaults['drizzle_scale'])
            self.drizzle_wht_threshold = settings_data.get('drizzle_wht_threshold', defaults['drizzle_wht_threshold'])
            self.drizzle_mode = settings_data.get('drizzle_mode', defaults['drizzle_mode'])
            self.drizzle_kernel = settings_data.get('drizzle_kernel', defaults['drizzle_kernel'])
            self.drizzle_pixfrac = settings_data.get('drizzle_pixfrac', defaults['drizzle_pixfrac'])
            
            # Preview
            self.preview_stretch_method = settings_data.get('preview_stretch_method', defaults['preview_stretch_method'])
            self.preview_black_point = settings_data.get('preview_black_point', defaults['preview_black_point'])
            self.preview_white_point = settings_data.get('preview_white_point', defaults['preview_white_point'])
            self.preview_gamma = settings_data.get('preview_gamma', defaults['preview_gamma'])
            self.preview_r_gain = settings_data.get('preview_r_gain', defaults['preview_r_gain'])
            self.preview_g_gain = settings_data.get('preview_g_gain', defaults['preview_g_gain'])
            self.preview_b_gain = settings_data.get('preview_b_gain', defaults['preview_b_gain'])

            # UI
            self.language = settings_data.get('language', defaults['language'])
            self.window_geometry = settings_data.get('window_geometry', defaults['window_geometry'])

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