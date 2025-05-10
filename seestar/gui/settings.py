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
from .mosaic_gui import VALID_DRIZZLE_KERNELS 

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
        """ Réinitialise tous les paramètres de l'instance à leurs valeurs par défaut. """
        defaults = self.get_default_values()
        for key, value in defaults.items():
            setattr(self, key, value)
        print("DEBUG (Settings reset_to_defaults): Tous les attributs de l'instance réinitialisés aux valeurs par défaut.")

    def update_from_ui(self, gui_instance):
        """ Met à jour les paramètres depuis les variables Tkinter de l'interface. """
        # ... (validation gui_instance inchangée) ...
        if gui_instance is None or not hasattr(gui_instance, 'root') or not gui_instance.root.winfo_exists():
            print("Warning: Cannot update settings from invalid GUI instance.")
            return
        print("DEBUG (Settings update_from_ui): Lecture des paramètres depuis l'UI...") # <-- AJOUTÉ DEBUG
        try:
            # --- Processing Settings ---
            # ... (inchangé) ...
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

            # --- Quality Weighting Settings ---
            # ... (inchangé) ...
            self.use_quality_weighting = getattr(gui_instance, 'use_weighting_var', tk.BooleanVar(value=self.use_quality_weighting)).get()
            self.weight_by_snr = getattr(gui_instance, 'weight_snr_var', tk.BooleanVar(value=self.weight_by_snr)).get()
            self.weight_by_stars = getattr(gui_instance, 'weight_stars_var', tk.BooleanVar(value=self.weight_by_stars)).get()
            self.snr_exponent = getattr(gui_instance, 'snr_exponent_var', tk.DoubleVar(value=self.snr_exponent)).get()
            self.stars_exponent = getattr(gui_instance, 'stars_exponent_var', tk.DoubleVar(value=self.stars_exponent)).get()
            self.min_weight = getattr(gui_instance, 'min_weight_var', tk.DoubleVar(value=self.min_weight)).get()

            # --- Drizzle Settings ---
            self.use_drizzle = getattr(gui_instance, 'use_drizzle_var', tk.BooleanVar(value=self.use_drizzle)).get()
            scale_str = getattr(gui_instance, 'drizzle_scale_var', tk.StringVar(value=str(self.drizzle_scale))).get()
            try: self.drizzle_scale = int(float(scale_str))
            except ValueError: print(f"Warning: Invalid Drizzle scale value '{scale_str}' from UI."); self.drizzle_scale = self.reset_to_defaults()['drizzle_scale']

            # --- LECTURE CORRECTION CHROMA ---
            self.apply_chroma_correction = getattr(gui_instance, 'apply_chroma_correction_var', tk.BooleanVar(value=self.apply_chroma_correction)).get()
            print(f"DEBUG (Settings update_from_ui): apply_chroma_correction lu depuis UI: {self.apply_chroma_correction}") # <-- AJOUTÉ DEBUG
            # ---  ---

            # ---  Lecture Seuil WHT ---
            # Lire directement la variable DoubleVar (0.0-1.0)
            self.drizzle_wht_threshold = getattr(gui_instance, 'drizzle_wht_threshold_var', tk.DoubleVar(value=self.drizzle_wht_threshold)).get()
            print(f"DEBUG (Settings update_from_ui): WHT Threshold (0-1) lu: {self.drizzle_wht_threshold}") # <-- AJOUTÉ DEBUG
            # ---  ---

            self.drizzle_mode = getattr(gui_instance, 'drizzle_mode_var', tk.StringVar(value=self.drizzle_mode)).get()
            self.drizzle_kernel = getattr(gui_instance, 'drizzle_kernel_var', tk.StringVar(value=self.drizzle_kernel)).get()
            self.drizzle_pixfrac = getattr(gui_instance, 'drizzle_pixfrac_var', tk.DoubleVar(value=self.drizzle_pixfrac)).get()
            # --- juste lire les attributs du GUI) ---
            self.mosaic_mode_active = getattr(gui_instance, 'mosaic_mode_active', False) # Lire le flag
            self.mosaic_settings = getattr(gui_instance, 'mosaic_settings', {}).copy() # Lire le dict (faire une copie)
            # ---  ---

            # Lecture Paramètres SCNR Final ###
            self.apply_final_scnr = getattr(gui_instance, 'apply_final_scnr_var', tk.BooleanVar(value=self.apply_final_scnr)).get()
            # self.final_scnr_target_channel reste 'green' pour l'instant (pas d'UI pour changer)
            self.final_scnr_amount = getattr(gui_instance, 'final_scnr_amount_var', tk.DoubleVar(value=self.final_scnr_amount)).get()
            self.final_scnr_preserve_luminosity = getattr(gui_instance, 'final_scnr_preserve_lum_var', tk.BooleanVar(value=self.final_scnr_preserve_luminosity)).get()
            print(f"DEBUG (Settings update_from_ui): SCNR Final lu -> Apply: {self.apply_final_scnr}, Amount: {self.final_scnr_amount:.2f}, PreserveLum: {self.final_scnr_preserve_luminosity}")
            

            ### Lecture Paramètres Expert depuis l'UI ###
            print("DEBUG (Settings update_from_ui): Lecture des paramètres Expert...")
            # BN
            self.bn_grid_size_str = getattr(gui_instance, 'bn_grid_size_str_var', tk.StringVar(value=self.bn_grid_size_str)).get()
            self.bn_perc_low = getattr(gui_instance, 'bn_perc_low_var', tk.IntVar(value=self.bn_perc_low)).get()
            self.bn_perc_high = getattr(gui_instance, 'bn_perc_high_var', tk.IntVar(value=self.bn_perc_high)).get()
            self.bn_std_factor = getattr(gui_instance, 'bn_std_factor_var', tk.DoubleVar(value=self.bn_std_factor)).get()
            self.bn_min_gain = getattr(gui_instance, 'bn_min_gain_var', tk.DoubleVar(value=self.bn_min_gain)).get()
            self.bn_max_gain = getattr(gui_instance, 'bn_max_gain_var', tk.DoubleVar(value=self.bn_max_gain)).get()

            # CB
            self.cb_border_size = getattr(gui_instance, 'cb_border_size_var', tk.IntVar(value=self.cb_border_size)).get()
            self.cb_blur_radius = getattr(gui_instance, 'cb_blur_radius_var', tk.IntVar(value=self.cb_blur_radius)).get()
            self.cb_min_b_factor = getattr(gui_instance, 'cb_min_b_factor_var', tk.DoubleVar(value=self.cb_min_b_factor)).get()
            self.cb_max_b_factor = getattr(gui_instance, 'cb_max_b_factor_var', tk.DoubleVar(value=self.cb_max_b_factor)).get()

            # Rognage
            self.final_edge_crop_percent = getattr(gui_instance, 'final_edge_crop_percent_var', tk.DoubleVar(value=self.final_edge_crop_percent)).get()
            print("DEBUG (Settings update_from_ui): Paramètres Expert lus.")
            ### FIN paramètre experts ###

            ###  Lecture Paramètres Photutils BN depuis l'UI ###
            print("DEBUG (Settings update_from_ui): Lecture des paramètres Photutils BN...")
            self.apply_photutils_bn = getattr(gui_instance, 'apply_photutils_bn_var', tk.BooleanVar(value=self.apply_photutils_bn)).get()
            if hasattr(gui_instance, 'apply_photutils_bn_var'):
                print(f"DEBUG SM update_from_ui: VALEUR DIRECTE DE gui_instance.apply_photutils_bn_var.get() = {gui_instance.apply_photutils_bn_var.get()}")
            else:
                print("DEBUG SM update_from_ui: gui_instance N'A PAS apply_photutils_bn_var, getattr a utilisé sa valeur par défaut.")
            print(f"DEBUG SM update_from_ui: self.apply_photutils_bn (dans SettingsManager) APRES LECTURE UI = {self.apply_photutils_bn}")
            print(f"DEBUG (Settings update_from_ui): Photutils BN lu DEPUIS UI -> Apply: {self.apply_photutils_bn}")
            self.photutils_bn_box_size = getattr(gui_instance, 'photutils_bn_box_size_var', tk.IntVar(value=self.photutils_bn_box_size)).get()
            self.photutils_bn_filter_size = getattr(gui_instance, 'photutils_bn_filter_size_var', tk.IntVar(value=self.photutils_bn_filter_size)).get()
            self.photutils_bn_sigma_clip = getattr(gui_instance, 'photutils_bn_sigma_clip_var', tk.DoubleVar(value=self.photutils_bn_sigma_clip)).get()
            self.photutils_bn_exclude_percentile = getattr(gui_instance, 'photutils_bn_exclude_percentile_var', tk.DoubleVar(value=self.photutils_bn_exclude_percentile)).get()
            print(f"DEBUG (Settings update_from_ui): Photutils BN lus -> Apply: {self.apply_photutils_bn}, Box: {self.photutils_bn_box_size}, Filt: {self.photutils_bn_filter_size}")
            ### FIN Paramètres Photutils BN ###
            
            ### Lecture Clé API Astrometry.net ###
            # La variable Tkinter sera dans gui_instance (SeestarStackerGUI)
            self.astrometry_api_key = getattr(gui_instance, 'astrometry_api_key_var', tk.StringVar(value=self.astrometry_api_key)).get()
            # Ne pas logguer la clé API complète pour la sécurité
            print(f"DEBUG (Settings update_from_ui): Clé API Astrometry lue (longueur: {len(self.astrometry_api_key)}).")
            ### FIN ###            
            
            # --- Preview Settings ---
            # ... (inchangé) ...
            self.preview_stretch_method = getattr(gui_instance, 'preview_stretch_method', tk.StringVar(value=self.preview_stretch_method)).get()
            self.preview_black_point = getattr(gui_instance, 'preview_black_point', tk.DoubleVar(value=self.preview_black_point)).get()
            self.preview_white_point = getattr(gui_instance, 'preview_white_point', tk.DoubleVar(value=self.preview_white_point)).get()
            self.preview_gamma = getattr(gui_instance, 'preview_gamma', tk.DoubleVar(value=self.preview_gamma)).get()
            self.preview_r_gain = getattr(gui_instance, 'preview_r_gain', tk.DoubleVar(value=self.preview_r_gain)).get()
            self.preview_g_gain = getattr(gui_instance, 'preview_g_gain', tk.DoubleVar(value=self.preview_g_gain)).get()
            self.preview_b_gain = getattr(gui_instance, 'preview_b_gain', tk.DoubleVar(value=self.preview_b_gain)).get()

            # --- UI Settings ---
            # ... (inchangé) ...
            self.language = getattr(gui_instance, 'language_var', tk.StringVar(value=self.language)).get()
            if gui_instance.root.winfo_exists():
                 current_geo = gui_instance.root.geometry()
                 if isinstance(current_geo, str) and 'x' in current_geo and '+' in current_geo: self.window_geometry = current_geo

        # ... (gestion erreurs inchangée) ...
        except AttributeError as ae: print(f"Error updating settings from UI (AttributeError): {ae}")
        except tk.TclError as te: print(f"Error updating settings from UI (TclError): {te}")
        except Exception as e: print(f"Unexpected error updating settings from UI: {e}"); traceback.print_exc(limit=2)

    def apply_to_ui(self, gui_instance):
        """ Applique les paramètres chargés/actuels aux variables Tkinter. """
        # ... (validation gui_instance inchangée) ...
        if gui_instance is None or not hasattr(gui_instance, 'root') or not gui_instance.root.winfo_exists():
            print("Warning: Cannot apply settings to invalid GUI instance.")
            return
        try:
            print("DEBUG (Settings apply_to_ui): Application des paramètres à l'UI...") # <-- AJOUTÉ DEBUG

            # --- Processing Settings ---
            # ... (inchangé) ...
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
            # ... (inchangé) ...
            getattr(gui_instance, 'use_weighting_var', tk.BooleanVar()).set(self.use_quality_weighting)
            getattr(gui_instance, 'weight_snr_var', tk.BooleanVar()).set(self.weight_by_snr)
            getattr(gui_instance, 'weight_stars_var', tk.BooleanVar()).set(self.weight_by_stars)
            getattr(gui_instance, 'snr_exponent_var', tk.DoubleVar()).set(self.snr_exponent)
            getattr(gui_instance, 'stars_exponent_var', tk.DoubleVar()).set(self.stars_exponent)
            getattr(gui_instance, 'min_weight_var', tk.DoubleVar()).set(self.min_weight)

            # --- Drizzle Settings ---
            getattr(gui_instance, 'use_drizzle_var', tk.BooleanVar()).set(self.use_drizzle)
            getattr(gui_instance, 'drizzle_scale_var', tk.StringVar()).set(str(self.drizzle_scale))

            # --- MODIFIÉ: Application Seuil WHT ---
            # 1. Définir la variable de stockage (0.0-1.0)
            wht_value_01 = self.drizzle_wht_threshold
            getattr(gui_instance, 'drizzle_wht_threshold_var', tk.DoubleVar()).set(wht_value_01)
            print(f"DEBUG (Settings apply_to_ui): WHT Threshold (0-1) appliqué: {wht_value_01}") # <-- AJOUTÉ DEBUG

            # 2. Calculer et définir la variable d'affichage (0-100)
            try:
                 wht_value_percent = round(wht_value_01 * 100.0)
                 # S'assurer que la valeur est dans la plage du Spinbox (10-100)
                 wht_display_value = np.clip(wht_value_percent, 10, 100)
                 wht_display_str = f"{wht_display_value:.0f}" # Format string pour le Spinbox
                 getattr(gui_instance, 'drizzle_wht_display_var', tk.StringVar()).set(wht_display_str)
                 print(f"DEBUG (Settings apply_to_ui): WHT Display (%) appliqué: {wht_display_str}") # <-- AJOUTÉ DEBUG
            except Exception as e_conv:
                 print(f"ERREUR (Settings apply_to_ui): Échec conversion/application WHT display: {e_conv}")
                 # Fallback: essayer de mettre une valeur par défaut sûre à l'affichage
                 getattr(gui_instance, 'drizzle_wht_display_var', tk.StringVar()).set("70")
            # --- FIN MODIFICATION ---

            getattr(gui_instance, 'drizzle_mode_var', tk.StringVar()).set(self.drizzle_mode)
            getattr(gui_instance, 'drizzle_kernel_var', tk.StringVar()).set(self.drizzle_kernel)
            getattr(gui_instance, 'drizzle_pixfrac_var', tk.DoubleVar()).set(self.drizzle_pixfrac)
            # --- juste définir les attributs du GUI) ---
            setattr(gui_instance, 'mosaic_mode_active', bool(self.mosaic_mode_active)) # Assurer booléen
            setattr(gui_instance, 'mosaic_settings', self.mosaic_settings.copy() if isinstance(self.mosaic_settings, dict) else {}) # Assurer dict, passer copie
            # Pas besoin de mettre à jour l'UI de la fenêtre modale ici, elle lira à son ouverture
            # On pourrait mettre à jour l'indicateur visuel sur le GUI principal ici
            if hasattr(gui_instance, '_update_mosaic_status_indicator'): gui_instance._update_mosaic_status_indicator()
            # --- ---

            # Application Paramètres SCNR Final à l'UI ###
            getattr(gui_instance, 'apply_final_scnr_var', tk.BooleanVar()).set(self.apply_final_scnr)
            # Pas d'UI pour final_scnr_target_channel pour l'instant
            getattr(gui_instance, 'final_scnr_amount_var', tk.DoubleVar()).set(self.final_scnr_amount)
            getattr(gui_instance, 'final_scnr_preserve_lum_var', tk.BooleanVar()).set(self.final_scnr_preserve_luminosity)
            # Mettre à jour l'état des widgets SCNR si la méthode existe dans le GUI
            if hasattr(gui_instance, '_update_final_scnr_options_state'):
                gui_instance._update_final_scnr_options_state()
            print(f"DEBUG (Settings apply_to_ui): SCNR Final appliqué à UI -> Apply: {self.apply_final_scnr}, Amount: {self.final_scnr_amount:.2f}, PreserveLum: {self.final_scnr_preserve_luminosity}")
            
            ### Application Paramètres Expert à l'UI ###
            print("DEBUG (Settings apply_to_ui): Application des paramètres Expert...")
            # BN
            getattr(gui_instance, 'bn_grid_size_str_var', tk.StringVar()).set(self.bn_grid_size_str)
            getattr(gui_instance, 'bn_perc_low_var', tk.IntVar()).set(self.bn_perc_low)
            getattr(gui_instance, 'bn_perc_high_var', tk.IntVar()).set(self.bn_perc_high)
            getattr(gui_instance, 'bn_std_factor_var', tk.DoubleVar()).set(self.bn_std_factor)
            getattr(gui_instance, 'bn_min_gain_var', tk.DoubleVar()).set(self.bn_min_gain)
            getattr(gui_instance, 'bn_max_gain_var', tk.DoubleVar()).set(self.bn_max_gain)

            # CB
            getattr(gui_instance, 'cb_border_size_var', tk.IntVar()).set(self.cb_border_size)
            getattr(gui_instance, 'cb_blur_radius_var', tk.IntVar()).set(self.cb_blur_radius)
            getattr(gui_instance, 'cb_min_b_factor_var', tk.DoubleVar()).set(self.cb_min_b_factor)
            getattr(gui_instance, 'cb_max_b_factor_var', tk.DoubleVar()).set(self.cb_max_b_factor)

            # Rognage
            getattr(gui_instance, 'final_edge_crop_percent_var', tk.DoubleVar()).set(self.final_edge_crop_percent)

            ### Application Paramètres Photutils BN à l'UI ###
            print("DEBUG (Settings apply_to_ui): Application des paramètres Photutils BN...")
            getattr(gui_instance, 'apply_photutils_bn_var', tk.BooleanVar()).set(self.apply_photutils_bn)
            getattr(gui_instance, 'photutils_bn_box_size_var', tk.IntVar()).set(self.photutils_bn_box_size)
            getattr(gui_instance, 'photutils_bn_filter_size_var', tk.IntVar()).set(self.photutils_bn_filter_size)
            getattr(gui_instance, 'photutils_bn_sigma_clip_var', tk.DoubleVar()).set(self.photutils_bn_sigma_clip)
            getattr(gui_instance, 'photutils_bn_exclude_percentile_var', tk.DoubleVar()).set(self.photutils_bn_exclude_percentile)
            
            # Appel pour mettre à jour l'état des widgets enfants
            if hasattr(gui_instance, '_update_photutils_bn_options_state'): # CETTE LIGNE EST CELLE QUE VOUS AVEZ MONTREE
            ### Application Clé API Astrometry.net à l'UI ###
            # La variable Tkinter est dans gui_instance
                getattr(gui_instance, 'astrometry_api_key_var', tk.StringVar()).set(self.astrometry_api_key or "") # Assurer string
                print(f"DEBUG (Settings apply_to_ui): Clé API Astrometry appliquée à l'UI (longueur: {len(self.astrometry_api_key or '')}).")
                ### FIN ###
                gui_instance._update_photutils_bn_options_state() 
            print("DEBUG (Settings apply_to_ui): Paramètres Photutils BN appliqués.")
            
        
            ### FIN paramètres expert ###            

            # --- Preview Settings ---
            # ... (inchangé) ...
            getattr(gui_instance, 'preview_stretch_method', tk.StringVar()).set(self.preview_stretch_method)
            getattr(gui_instance, 'preview_black_point', tk.DoubleVar()).set(self.preview_black_point)
            getattr(gui_instance, 'preview_white_point', tk.DoubleVar()).set(self.preview_white_point)
            getattr(gui_instance, 'preview_gamma', tk.DoubleVar()).set(self.preview_gamma)
            getattr(gui_instance, 'preview_r_gain', tk.DoubleVar()).set(self.preview_r_gain)
            getattr(gui_instance, 'preview_g_gain', tk.DoubleVar()).set(self.preview_g_gain)
            getattr(gui_instance, 'preview_b_gain', tk.DoubleVar()).set(self.preview_b_gain)

            # --- UI Settings ---
            # ... (inchangé) ...
            getattr(gui_instance, 'language_var', tk.StringVar()).set(self.language)
            if isinstance(self.window_geometry, str) and 'x' in self.window_geometry and '+' in self.window_geometry:
                try: gui_instance.root.geometry(self.window_geometry)
                except tk.TclError: print(f"Warning: Could not apply window geometry '{self.window_geometry}'.")

            # --- Mettre à jour l'état des widgets dépendants ---
            # (inchangé)
            if hasattr(gui_instance, '_update_weighting_options_state'): gui_instance._update_weighting_options_state()
            if hasattr(gui_instance, '_update_drizzle_options_state'): gui_instance._update_drizzle_options_state()
            # Note: L'appel à _update_spinbox_from_float n'est plus nécessaire ici car on le gère directement

            print("DEBUG (Settings apply_to_ui): Fin application paramètres UI.") # <-- AJOUTÉ DEBUG

        # ... (gestion erreurs inchangée) ...
        except AttributeError as ae: print(f"Error applying settings to UI (AttributeError): {ae}")
        except tk.TclError as te: print(f"Error applying settings to UI (TclError - widget likely destroyed?): {te}")
        except Exception as e: print(f"Unexpected error applying settings to UI: {e}"); traceback.print_exc(limit=2)

    def get_default_values(self): # NOUVELLE MÉTHODE
        """ Retourne un dictionnaire des valeurs par défaut sans modifier l'instance. """
        # Code de reset_to_defaults, mais au lieu de self.X = Y, on fait defaults_dict['X'] = Y
        defaults_dict = {}
        defaults_dict['input_folder'] = ""
        defaults_dict['output_folder'] = ""
        defaults_dict['reference_image_path'] = ""
        defaults_dict['bayer_pattern'] = "GRBG"
        defaults_dict['batch_size'] = 0
        defaults_dict['stacking_mode'] = "kappa-sigma"
        defaults_dict['kappa'] = 2.5
        defaults_dict['correct_hot_pixels'] = True
        defaults_dict['hot_pixel_threshold'] = 3.0
        defaults_dict['neighborhood_size'] = 5
        defaults_dict['cleanup_temp'] = True
        defaults_dict['use_quality_weighting'] = False
        defaults_dict['weight_by_snr'] = True
        defaults_dict['weight_by_stars'] = True
        defaults_dict['snr_exponent'] = 1.0
        defaults_dict['stars_exponent'] = 0.5
        defaults_dict['min_weight'] = 0.1
        defaults_dict['use_drizzle'] = False
        defaults_dict['drizzle_scale'] = 2
        defaults_dict['drizzle_wht_threshold'] = 0.7
        defaults_dict['drizzle_mode'] = "Final"
        defaults_dict['drizzle_kernel'] = "square"
        defaults_dict['drizzle_pixfrac'] = 1.0
        defaults_dict['apply_chroma_correction'] = True
        defaults_dict['mosaic_mode_active'] = False
        defaults_dict['mosaic_settings'] = {"kernel": "square", "pixfrac": 1.0}
        defaults_dict['apply_final_scnr'] = False
        defaults_dict['final_scnr_target_channel'] = 'green'
        defaults_dict['final_scnr_amount'] = 0.8
        defaults_dict['final_scnr_preserve_luminosity'] = True
        defaults_dict['bn_grid_size_str'] = "16x16"
        defaults_dict['bn_perc_low'] = 5
        defaults_dict['bn_perc_high'] = 30
        defaults_dict['bn_std_factor'] = 1.0
        defaults_dict['bn_min_gain'] = 0.2
        defaults_dict['bn_max_gain'] = 7.0
        defaults_dict['cb_border_size'] = 25
        defaults_dict['cb_blur_radius'] = 8
        defaults_dict['cb_min_b_factor'] = 0.4
        defaults_dict['cb_max_b_factor'] = 1.5
        defaults_dict['final_edge_crop_percent'] = 2.0
        # LA VALEUR PAR DÉFAUT POUR Photutils BN EST False
        defaults_dict['apply_photutils_bn'] = False # <--- Important
        defaults_dict['photutils_bn_box_size'] = 128
        defaults_dict['photutils_bn_filter_size'] = 5
        defaults_dict['photutils_bn_sigma_clip'] = 3.0
        defaults_dict['photutils_bn_exclude_percentile'] = 98.0
        defaults_dict['astrometry_api_key'] = ""
        defaults_dict['preview_stretch_method'] = "Asinh"
        defaults_dict['preview_black_point'] = 0.01
        defaults_dict['preview_white_point'] = 0.99
        defaults_dict['preview_gamma'] = 1.0
        defaults_dict['preview_r_gain'] = 1.0
        defaults_dict['preview_g_gain'] = 1.0
        defaults_dict['preview_b_gain'] = 1.0
        defaults_dict['language'] = 'en'
        defaults_dict['window_geometry'] = "1200x750"
        return defaults_dict


# --- DANS LA CLASSE SettingsManager DANS seestar/gui/settings.py ---

    def validate_settings(self):
        """Valide et corrige les paramètres si nécessaire. Retourne les messages de correction."""
        messages = []
        print("DEBUG (Settings validate_settings): DÉBUT de la validation.") # <-- DEBUG

        # Obtenir les valeurs par défaut pour fallback SANS modifier l'instance actuelle
        defaults_fallback = self.get_default_values()
        print(f"DEBUG (Settings validate_settings): Valeur self.apply_photutils_bn AVANT TOUTE VALIDATION (lue de l'UI): {getattr(self, 'apply_photutils_bn', 'NON_DEFINI_ENCORE')}") # <-- DEBUG

        try:
             # --- Processing Settings Validation ---
             # ... (validation kappa, batch_size, etc. comme avant, mais utilise defaults_fallback) ...
             self.kappa = float(self.kappa)
             if not (1.0 <= self.kappa <= 5.0): 
                 original = self.kappa; self.kappa = np.clip(self.kappa, 1.0, 5.0)
                 messages.append(f"Kappa ({original:.1f}) ajusté à {self.kappa:.1f}")
             self.batch_size = int(self.batch_size)
             if self.batch_size < 0: 
                 original = self.batch_size; self.batch_size = 0 # 0 sera estimé par le backend
                 messages.append(f"Taille Lot ({original}) ajusté à {self.batch_size} (auto)")
             self.hot_pixel_threshold = float(self.hot_pixel_threshold)
             if not (0.5 <= self.hot_pixel_threshold <= 10.0): 
                 original = self.hot_pixel_threshold; self.hot_pixel_threshold = np.clip(self.hot_pixel_threshold, 0.5, 10.0)
                 messages.append(f"Seuil Px Chauds ({original:.1f}) ajusté à {self.hot_pixel_threshold:.1f}")
             self.neighborhood_size = int(self.neighborhood_size)
             if self.neighborhood_size < 3: 
                 original = self.neighborhood_size; self.neighborhood_size = 3
                 messages.append(f"Voisinage Px Chauds ({original}) ajusté à {self.neighborhood_size}")
             if self.neighborhood_size % 2 == 0: 
                 original = self.neighborhood_size; self.neighborhood_size += 1
                 messages.append(f"Voisinage Px Chauds ({original}) ajusté à {self.neighborhood_size} (impair)")

             # --- Quality Weighting Validation ---
             self.use_quality_weighting = bool(self.use_quality_weighting)
             self.weight_by_snr = bool(self.weight_by_snr)
             self.weight_by_stars = bool(self.weight_by_stars)
             self.snr_exponent = float(self.snr_exponent)
             self.stars_exponent = float(self.stars_exponent)
             self.min_weight = float(self.min_weight)
             if self.snr_exponent <= 0: 
                 original = self.snr_exponent; self.snr_exponent = defaults_fallback['snr_exponent']
                 messages.append(f"Exposant SNR ({original:.1f}) ajusté à {self.snr_exponent:.1f}")
             if self.stars_exponent <= 0: 
                 original = self.stars_exponent; self.stars_exponent = defaults_fallback['stars_exponent']
                 messages.append(f"Exposant Étoiles ({original:.1f}) ajusté à {self.stars_exponent:.1f}")
             if not (0 < self.min_weight <= 1.0): 
                 original = self.min_weight; self.min_weight = np.clip(self.min_weight, 0.01, 1.0)
                 messages.append(f"Poids Min ({original:.2f}) ajusté à {self.min_weight:.2f}")
             if self.use_quality_weighting and not (self.weight_by_snr or self.weight_by_stars):
                 self.weight_by_snr = True
                 messages.append("Pondération activée mais aucune métrique choisie. SNR activé par défaut.")

             # --- Drizzle Settings Validation ---
             self.use_drizzle = bool(self.use_drizzle)
             try:
                 scale_num = int(float(self.drizzle_scale))
                 if scale_num not in [2, 3, 4]:
                     original = self.drizzle_scale; self.drizzle_scale = defaults_fallback['drizzle_scale']
                     messages.append(f"Échelle Drizzle ({original}) invalide, réinitialisée à {self.drizzle_scale}")
                 else: self.drizzle_scale = scale_num
             except (ValueError, TypeError):
                 original = self.drizzle_scale; self.drizzle_scale = defaults_fallback['drizzle_scale']
                 messages.append(f"Échelle Drizzle invalide ({original}), réinitialisée à {self.drizzle_scale}")
             try:
                 self.drizzle_wht_threshold = float(self.drizzle_wht_threshold)
                 if not (0.0 < self.drizzle_wht_threshold <= 1.0):
                      original = self.drizzle_wht_threshold; self.drizzle_wht_threshold = np.clip(self.drizzle_wht_threshold, 0.1, 1.0)
                      messages.append(f"Seuil Drizzle WHT ({original:.2f}) ajusté à {self.drizzle_wht_threshold:.2f}")
             except (ValueError, TypeError):
                 original = self.drizzle_wht_threshold; self.drizzle_wht_threshold = defaults_fallback['drizzle_wht_threshold']
                 messages.append(f"Seuil Drizzle WHT invalide ({original}), réinitialisé à {self.drizzle_wht_threshold:.2f}")
             
             valid_drizzle_modes = ["Final", "Incremental"]
             if not isinstance(self.drizzle_mode, str) or self.drizzle_mode not in valid_drizzle_modes:
                 original = self.drizzle_mode # Stocker la valeur originale avant correction
                 self.drizzle_mode = defaults_fallback['drizzle_mode'] # Utiliser le fallback pour corriger
                 messages.append(f"Mode Drizzle ({original}) invalide, réinitialisé à '{self.drizzle_mode}'")


            ###  Validation Paramètres SCNR Final ###
             self.apply_final_scnr = bool(self.apply_final_scnr)
             self.final_scnr_target_channel = str(self.final_scnr_target_channel).lower()
             if self.final_scnr_target_channel not in ['green', 'blue']:
                 original_target = self.final_scnr_target_channel
                 self.final_scnr_target_channel = defaults_fallback['final_scnr_target_channel']
                 messages.append(f"Cible SCNR Final ('{original_target}') invalide, réinitialisée à '{self.final_scnr_target_channel}'.")
             try:
                 self.final_scnr_amount = float(self.final_scnr_amount)
                 if not (0.0 <= self.final_scnr_amount <= 1.0):
                     original_amount = self.final_scnr_amount
                     self.final_scnr_amount = np.clip(self.final_scnr_amount, 0.0, 1.0)
                     messages.append(f"Intensité SCNR Final ({original_amount:.2f}) hors limites [0.0, 1.0], ajustée à {self.final_scnr_amount:.2f}.")
             except (ValueError, TypeError):
                 original_amount = self.final_scnr_amount
                 self.final_scnr_amount = defaults_fallback['final_scnr_amount']
                 messages.append(f"Intensité SCNR Final ('{original_amount}') invalide, réinitialisée à {self.final_scnr_amount:.2f}.")
             self.final_scnr_preserve_luminosity = bool(self.final_scnr_preserve_luminosity)
            
            ### Validation Paramètres Expert ###
             print("DEBUG (Settings validate_settings): Validation des paramètres Expert...")
             # BN
             if not isinstance(self.bn_grid_size_str, str) or self.bn_grid_size_str not in ["8x8", "16x16", "32x32", "64x64"]:
                 messages.append(f"Taille grille BN invalide ('{self.bn_grid_size_str}'), réinitialisée."); self.bn_grid_size_str = defaults_fallback['bn_grid_size_str']
             self.bn_perc_low = int(np.clip(self.bn_perc_low, 0, 40)); self.bn_perc_high = int(np.clip(self.bn_perc_high, self.bn_perc_low + 1, 90))
             self.bn_std_factor = float(np.clip(self.bn_std_factor, 0.1, 10.0))
             self.bn_min_gain = float(np.clip(self.bn_min_gain, 0.05, 5.0)); self.bn_max_gain = float(np.clip(self.bn_max_gain, self.bn_min_gain, 20.0))
             # CB
             self.cb_border_size = int(np.clip(self.cb_border_size, 5, 200))
             self.cb_blur_radius = int(np.clip(self.cb_blur_radius, 0, 100))
             self.cb_min_b_factor = float(np.clip(self.cb_min_b_factor, 0.1, 1.0))
             self.cb_max_b_factor = float(np.clip(self.cb_max_b_factor, self.cb_min_b_factor, 5.0))
             # Rognage
             self.final_edge_crop_percent = float(np.clip(self.final_edge_crop_percent, 0.0, 25.0))

             ### Validation Paramètres Photutils BN ###
             print("DEBUG (Settings validate_settings): Début validation bloc Photutils BN.")
             # La valeur self.apply_photutils_bn est celle qui a été définie par update_from_ui
             # (qui l'a lue depuis l'UI). On s'assure juste que c'est un booléen.
             # Ce cast est redondant si update_from_ui fait déjà bool(), mais ne nuit pas.
             current_apply_photutils_bn_value = getattr(self, 'apply_photutils_bn', defaults_fallback['apply_photutils_bn'])
             self.apply_photutils_bn = bool(current_apply_photutils_bn_value)
             print(f"DEBUG (Settings validate_settings): self.apply_photutils_bn APRES cast en bool = {self.apply_photutils_bn} (venait de {current_apply_photutils_bn_value})")

             self.photutils_bn_box_size = int(np.clip(getattr(self, 'photutils_bn_box_size', defaults_fallback['photutils_bn_box_size']), 8, 1024))
             self.photutils_bn_filter_size = int(np.clip(getattr(self, 'photutils_bn_filter_size', defaults_fallback['photutils_bn_filter_size']), 1, 25))
             if self.photutils_bn_filter_size % 2 == 0: self.photutils_bn_filter_size += 1
             self.photutils_bn_sigma_clip = float(np.clip(getattr(self, 'photutils_bn_sigma_clip', defaults_fallback['photutils_bn_sigma_clip']), 0.5, 10.0))
             self.photutils_bn_exclude_percentile = float(np.clip(getattr(self, 'photutils_bn_exclude_percentile', defaults_fallback['photutils_bn_exclude_percentile']), 0.0, 100.0))
             print("DEBUG (Settings validate_settings): Fin validation bloc Photutils BN.")
            
             ### Validation Clé API Astrometry.net ###
             if not isinstance(self.astrometry_api_key, str):
                messages.append("Clé API Astrometry invalide (pas une chaîne), réinitialisée.")
                self.astrometry_api_key = defaults_fallback['astrometry_api_key']

             # --- Drizzle Kernel and Pixfrac Validation ---
             valid_kernels = ['square', 'gaussian', 'point', 'tophat', 'turbo', 'lanczos2', 'lanczos3']
             if not isinstance(self.drizzle_kernel, str) or self.drizzle_kernel.lower() not in valid_kernels:
                original = self.drizzle_kernel
                self.drizzle_kernel = defaults_fallback['drizzle_kernel']
                messages.append(f"Noyau Drizzle ('{original}') invalide, réinitialisé à '{self.drizzle_kernel}'")
             else:
                self.drizzle_kernel = self.drizzle_kernel.lower()
             try:
                 self.drizzle_pixfrac = float(self.drizzle_pixfrac)
                 if not (0.01 <= self.drizzle_pixfrac <= 1.0):
                      original = self.drizzle_pixfrac
                      self.drizzle_pixfrac = np.clip(self.drizzle_pixfrac, 0.01, 1.0)
                      messages.append(f"Pixfrac Drizzle ({original:.2f}) hors limites [0.01, 1.0], ajusté à {self.drizzle_pixfrac:.2f}")
             except (ValueError, TypeError):
                 original = self.drizzle_pixfrac
                 self.drizzle_pixfrac = defaults_fallback['drizzle_pixfrac']
                 messages.append(f"Pixfrac Drizzle ('{original}') invalide, réinitialisé à {self.drizzle_pixfrac:.2f}")

             # --- CORRECTION CHROMA ---
             self.apply_chroma_correction = bool(self.apply_chroma_correction)

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
             if self.preview_stretch_method not in valid_methods: 
                 self.preview_stretch_method = defaults_fallback['preview_stretch_method']
                 messages.append(f"Méthode d'étirement invalide, réinitialisée à '{self.preview_stretch_method}'")

        except (ValueError, TypeError) as e:
             messages.append(f"Paramètre numérique invalide détecté: {e}. Vérifiez les valeurs.")
             print(f"Warning (Settings validate_settings): Erreur de type/valeur -> {e}. Réinitialisation complète des settings.") # <-- DEBUG
             # Réinitialiser l'instance actuelle aux valeurs par défaut complètes en cas d'erreur majeure
             # Cela assure que l'état de self est cohérent après une exception ici.
             self.reset_to_defaults() 
             # Logguer la valeur de apply_photutils_bn APRES cette réinitialisation
             print(f"DEBUG (Settings validate_settings): self.apply_photutils_bn APRES reset_to_defaults dû à une exception: {self.apply_photutils_bn}") # <-- DEBUG

        print(f"DEBUG (Settings validate_settings): FIN de la validation. Valeur finale de self.apply_photutils_bn: {getattr(self, 'apply_photutils_bn', 'NON_DEFINI')}") # <-- DEBUG
        return messages


    def save_settings(self):
        """ Sauvegarde les paramètres actuels dans le fichier JSON. """
        # S'assurer que les types sont corrects pour JSON
        settings_data = {
            'version': "1.7.0", # Incrémenter la version si la structure change significativement
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
             # ---  ---
            'mosaic_mode_active': bool(self.mosaic_mode_active),
            'mosaic_settings': self.mosaic_settings if isinstance(self.mosaic_settings, dict) else {}, # Sauvegarder le dict
            # --- Fin msaique ---

            # --- CORRECTION CHROMA ---
            'apply_chroma_correction': bool(self.apply_chroma_correction), # Sauvegarder comme booléen
            # ---  ---
            
            ### Sauvegarde Paramètres SCNR Final ###
            'apply_final_scnr': bool(self.apply_final_scnr),
            'final_scnr_target_channel': str(self.final_scnr_target_channel),
            'final_scnr_amount': float(self.final_scnr_amount),
            'final_scnr_preserve_luminosity': bool(self.final_scnr_preserve_luminosity),

            ###  Sauvegarde Paramètres Expert ###
            'bn_grid_size_str': str(self.bn_grid_size_str),
            'bn_perc_low': int(self.bn_perc_low),
            'bn_perc_high': int(self.bn_perc_high),
            'bn_std_factor': float(self.bn_std_factor),
            'bn_min_gain': float(self.bn_min_gain),
            'bn_max_gain': float(self.bn_max_gain),
            'cb_border_size': int(self.cb_border_size),
            'cb_blur_radius': int(self.cb_blur_radius),
            'cb_min_b_factor': float(self.cb_min_b_factor),
            'cb_max_b_factor': float(self.cb_max_b_factor),
            'final_edge_crop_percent': float(self.final_edge_crop_percent),
            ### Sauvegarde Paramètres Photutils BN ###
            'apply_photutils_bn': bool(self.apply_photutils_bn),
            'photutils_bn_box_size': int(self.photutils_bn_box_size),
            'photutils_bn_filter_size': int(self.photutils_bn_filter_size),
            'photutils_bn_sigma_clip': float(self.photutils_bn_sigma_clip),
            'photutils_bn_exclude_percentile': float(self.photutils_bn_exclude_percentile),
            ### Sauvegarde Clé API Astrometry.net ###
            'astrometry_api_key': str(self.astrometry_api_key),
            ### FIN expert ###            

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
        print(f"DEBUG (Settings load_settings): Tentative chargement depuis {self.settings_file}...") 
        if not os.path.exists(self.settings_file):
            print(f"DEBUG (Settings load_settings): Fichier non trouvé. Utilisation défauts.") 
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
                         # --- AJOUTER ICI ---
            self.mosaic_mode_active = settings_data.get('mosaic_mode_active', defaults['mosaic_mode_active'])
            loaded_mosaic_settings = settings_data.get('mosaic_settings', defaults['mosaic_settings'])
            # Valider/Nettoyer les settings mosaïque chargés
            self.mosaic_settings = {
                'kernel': loaded_mosaic_settings.get('kernel', defaults['mosaic_settings']['kernel']),
                'pixfrac': loaded_mosaic_settings.get('pixfrac', defaults['mosaic_settings']['pixfrac'])
                # Ajouter d'autres clés futures ici avec leurs défauts
            }
            if self.mosaic_settings['kernel'] not in VALID_DRIZZLE_KERNELS: # Utiliser la constante définie dans mosaic_gui
                self.mosaic_settings['kernel'] = defaults['mosaic_settings']['kernel']
            try:
                 pf = float(self.mosaic_settings['pixfrac'])
                 self.mosaic_settings['pixfrac'] = np.clip(pf, 0.01, 1.0)
            except (ValueError, TypeError):
                 self.mosaic_settings['pixfrac'] = defaults['mosaic_settings']['pixfrac']
            # --- FIN AJOUT ---
            
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
            
            # ---CORRECTION CHROMA ---
            # Utiliser la valeur par défaut de 'defaults' si la clé n'est pas dans le fichier
            self.apply_chroma_correction = settings_data.get('apply_chroma_correction', defaults['apply_chroma_correction'])
            # S'assurer que c'est un booléen après chargement
            self.apply_chroma_correction = bool(self.apply_chroma_correction)
            print(f"DEBUG (Settings load_settings): apply_chroma_correction chargé: {self.apply_chroma_correction}") # <-- DEBUG
            # --- ---
            print(f"DEBUG (Settings load_settings): Paramètres chargés. Validation...") # DEBUG

            print(f"Settings loaded from {self.settings_file}")

            ###  Chargement Paramètres SCNR Final ###
            self.apply_final_scnr = bool(settings_data.get('apply_final_scnr', defaults['apply_final_scnr']))
            self.final_scnr_target_channel = str(settings_data.get('final_scnr_target_channel', defaults['final_scnr_target_channel']))
            self.final_scnr_amount = float(settings_data.get('final_scnr_amount', defaults['final_scnr_amount']))
            self.final_scnr_preserve_luminosity = bool(settings_data.get('final_scnr_preserve_luminosity', defaults['final_scnr_preserve_luminosity']))
            print(f"DEBUG (Settings load_settings): SCNR Final chargé -> Apply: {self.apply_final_scnr}, Target: {self.final_scnr_target_channel}, Amount: {self.final_scnr_amount:.2f}, PreserveLum: {self.final_scnr_preserve_luminosity}")

            ### Chargement Paramètres Expert ###
            print("DEBUG (Settings load_settings): Chargement des paramètres Expert...")
            self.bn_grid_size_str = str(settings_data.get('bn_grid_size_str', defaults['bn_grid_size_str']))
            self.bn_perc_low = int(settings_data.get('bn_perc_low', defaults['bn_perc_low']))
            self.bn_perc_high = int(settings_data.get('bn_perc_high', defaults['bn_perc_high']))
            self.bn_std_factor = float(settings_data.get('bn_std_factor', defaults['bn_std_factor']))
            self.bn_min_gain = float(settings_data.get('bn_min_gain', defaults['bn_min_gain']))
            self.bn_max_gain = float(settings_data.get('bn_max_gain', defaults['bn_max_gain']))
            
            self.cb_border_size = int(settings_data.get('cb_border_size', defaults['cb_border_size']))
            self.cb_blur_radius = int(settings_data.get('cb_blur_radius', defaults['cb_blur_radius']))
            self.cb_min_b_factor = float(settings_data.get('cb_min_b_factor', defaults['cb_min_b_factor']))
            self.cb_max_b_factor = float(settings_data.get('cb_max_b_factor', defaults['cb_max_b_factor']))

            self.final_edge_crop_percent = float(settings_data.get('final_edge_crop_percent', defaults['final_edge_crop_percent']))


            ### Chargement Paramètres Photutils BN ###
            print("DEBUG (Settings load_settings): Chargement des paramètres Photutils BN...")
            self.apply_photutils_bn = bool(settings_data.get('apply_photutils_bn', defaults['apply_photutils_bn']))
            self.photutils_bn_box_size = int(settings_data.get('photutils_bn_box_size', defaults['photutils_bn_box_size']))
            self.photutils_bn_filter_size = int(settings_data.get('photutils_bn_filter_size', defaults['photutils_bn_filter_size']))
            self.photutils_bn_sigma_clip = float(settings_data.get('photutils_bn_sigma_clip', defaults['photutils_bn_sigma_clip']))
            self.photutils_bn_exclude_percentile = float(settings_data.get('photutils_bn_exclude_percentile', defaults['photutils_bn_exclude_percentile']))
            ### NOUVEAU : Chargement Clé API Astrometry.net ###
            self.astrometry_api_key = str(settings_data.get('astrometry_api_key', defaults['astrometry_api_key']))
            # Ne pas logguer la clé complète
            print(f"DEBUG (Settings load_settings): Clé API Astrometry chargée (longueur: {len(self.astrometry_api_key)}).")
            ### FIN NOUVEAU ###
            print("DEBUG (Settings load_settings): Paramètres Photutils BN chargés.")
            ### FIN expert ###            

            # Validate settings after loading
            validation_messages = self.validate_settings()
            if validation_messages:
                 print("DEBUG (Settings load_settings): Paramètres chargés ajustés après validation:") # <--  DEBUG
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