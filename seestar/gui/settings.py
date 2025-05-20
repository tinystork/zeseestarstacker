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


#####################################################################################################################################




# DANS LA CLASSE SettingsManager DANS seestar/gui/settings.py
    def update_from_ui(self, gui_instance):
        """
        Met à jour les paramètres de CETTE instance SettingsManager depuis les
        variables Tkinter de l'instance GUI fournie.
        MODIFIED V5: Gestion améliorée pour les settings sans widget direct sur gui_instance.
        """
        if gui_instance is None or not hasattr(gui_instance, 'root') or not gui_instance.root.winfo_exists():
            print("Warning (SM update_from_ui): Cannot update settings from invalid GUI instance.")
            return
        print("DEBUG SM (update_from_ui V5): Lecture des paramètres depuis l'UI...")

        default_values_for_fallback = self.get_default_values()

        try:
            # --- Processing Settings ---
            self.input_folder = getattr(gui_instance, 'input_path', tk.StringVar(value=default_values_for_fallback['input_folder'])).get()
            # ... (tous les autres getattr pour les paramètres qui ONT un widget sur gui_instance) ...
            self.output_folder = getattr(gui_instance, 'output_path', tk.StringVar(value=default_values_for_fallback['output_folder'])).get()
            self.reference_image_path = getattr(gui_instance, 'reference_image_path', tk.StringVar(value=default_values_for_fallback['reference_image_path'])).get()
            self.stacking_mode = getattr(gui_instance, 'stacking_mode', tk.StringVar(value=default_values_for_fallback['stacking_mode'])).get()
            self.kappa = getattr(gui_instance, 'kappa', tk.DoubleVar(value=default_values_for_fallback['kappa'])).get()
            self.batch_size = getattr(gui_instance, 'batch_size', tk.IntVar(value=default_values_for_fallback['batch_size'])).get()
            self.correct_hot_pixels = getattr(gui_instance, 'correct_hot_pixels', tk.BooleanVar(value=default_values_for_fallback['correct_hot_pixels'])).get()
            self.hot_pixel_threshold = getattr(gui_instance, 'hot_pixel_threshold', tk.DoubleVar(value=default_values_for_fallback['hot_pixel_threshold'])).get()
            self.neighborhood_size = getattr(gui_instance, 'neighborhood_size', tk.IntVar(value=default_values_for_fallback['neighborhood_size'])).get()
            self.cleanup_temp = getattr(gui_instance, 'cleanup_temp_var', tk.BooleanVar(value=default_values_for_fallback['cleanup_temp'])).get()
            self.bayer_pattern = getattr(gui_instance, 'bayer_pattern_var', tk.StringVar(value=default_values_for_fallback['bayer_pattern'])).get()

            # --- Quality Weighting Settings ---
            self.use_quality_weighting = getattr(gui_instance, 'use_weighting_var', tk.BooleanVar(value=default_values_for_fallback['use_quality_weighting'])).get()
            # ... (autres pour quality weighting) ...
            self.weight_by_snr = getattr(gui_instance, 'weight_snr_var', tk.BooleanVar(value=default_values_for_fallback['weight_by_snr'])).get()
            self.weight_by_stars = getattr(gui_instance, 'weight_stars_var', tk.BooleanVar(value=default_values_for_fallback['weight_by_stars'])).get()
            self.snr_exponent = getattr(gui_instance, 'snr_exponent_var', tk.DoubleVar(value=default_values_for_fallback['snr_exponent'])).get()
            self.stars_exponent = getattr(gui_instance, 'stars_exponent_var', tk.DoubleVar(value=default_values_for_fallback['stars_exponent'])).get()
            self.min_weight = getattr(gui_instance, 'min_weight_var', tk.DoubleVar(value=default_values_for_fallback['min_weight'])).get()


            # --- Drizzle Settings (Globaux) ---
            self.use_drizzle = getattr(gui_instance, 'use_drizzle_var', tk.BooleanVar(value=default_values_for_fallback['use_drizzle'])).get()
            # ... (autres pour Drizzle) ...
            scale_str_ui = getattr(gui_instance, 'drizzle_scale_var', tk.StringVar(value=str(default_values_for_fallback['drizzle_scale']))).get()
            try: self.drizzle_scale = int(float(scale_str_ui))
            except ValueError: self.drizzle_scale = default_values_for_fallback['drizzle_scale']
            self.drizzle_wht_threshold = getattr(gui_instance, 'drizzle_wht_threshold_var', tk.DoubleVar(value=default_values_for_fallback['drizzle_wht_threshold'])).get()
            self.drizzle_mode = getattr(gui_instance, 'drizzle_mode_var', tk.StringVar(value=default_values_for_fallback['drizzle_mode'])).get()
            self.drizzle_kernel = getattr(gui_instance, 'drizzle_kernel_var', tk.StringVar(value=default_values_for_fallback['drizzle_kernel'])).get()
            self.drizzle_pixfrac = getattr(gui_instance, 'drizzle_pixfrac_var', tk.DoubleVar(value=default_values_for_fallback['drizzle_pixfrac'])).get()


            # --- Mosaïque Settings ---
            # mosaic_mode_active et mosaic_settings sont mis à jour par MosaicSettingsWindow directement sur self.settings
            # On ne les lit PAS depuis gui_instance ici pour éviter de les écraser par des valeurs par défaut si pas de widget direct.
            current_mosaic_mode_active = getattr(self, 'mosaic_mode_active', "NON_DEFINI_SUR_SELF")
            print(f"  DEBUG SM (update_from_ui): mosaic_mode_active: Conservation valeur actuelle: {current_mosaic_mode_active}")
            current_mosaic_settings_dict = getattr(self, 'mosaic_settings', {})
            print(f"  DEBUG SM (update_from_ui): mosaic_settings: Conservation valeur actuelle: {current_mosaic_settings_dict.get('alignment_mode', 'NonTrouve')}")

            # --- Astrometry.net API Key (Widget direct sur GUI principal) ---
            self.astrometry_api_key = getattr(gui_instance, 'astrometry_api_key_var', tk.StringVar(value=default_values_for_fallback['astrometry_api_key'])).get().strip()
            print(f"DEBUG SM (update_from_ui): Clé API Astrometry lue de l'UI (longueur: {len(self.astrometry_api_key)}).")

            # --- Post-Processing Settings (Ceux avec widgets directs sur GUI principal) ---
            self.apply_chroma_correction = getattr(gui_instance, 'apply_chroma_correction_var', tk.BooleanVar(value=default_values_for_fallback['apply_chroma_correction'])).get()
            self.apply_final_scnr = getattr(gui_instance, 'apply_final_scnr_var', tk.BooleanVar(value=default_values_for_fallback['apply_final_scnr'])).get()
            # ... (autres pour SCNR, BN, CB, Crop, Photutils BN, Feathering, LowWHT qui sont sur l'UI principale)
            self.final_scnr_target_channel = default_values_for_fallback['final_scnr_target_channel'] # Pas d'UI directe pour ça
            self.final_scnr_amount = getattr(gui_instance, 'final_scnr_amount_var', tk.DoubleVar(value=default_values_for_fallback['final_scnr_amount'])).get()
            self.final_scnr_preserve_luminosity = getattr(gui_instance, 'final_scnr_preserve_lum_var', tk.BooleanVar(value=default_values_for_fallback['final_scnr_preserve_luminosity'])).get()

            self.bn_grid_size_str = getattr(gui_instance, 'bn_grid_size_str_var', tk.StringVar(value=default_values_for_fallback['bn_grid_size_str'])).get()
            self.bn_perc_low = getattr(gui_instance, 'bn_perc_low_var', tk.IntVar(value=default_values_for_fallback['bn_perc_low'])).get()
            self.bn_perc_high = getattr(gui_instance, 'bn_perc_high_var', tk.IntVar(value=default_values_for_fallback['bn_perc_high'])).get()
            self.bn_std_factor = getattr(gui_instance, 'bn_std_factor_var', tk.DoubleVar(value=default_values_for_fallback['bn_std_factor'])).get()
            self.bn_min_gain = getattr(gui_instance, 'bn_min_gain_var', tk.DoubleVar(value=default_values_for_fallback['bn_min_gain'])).get()
            self.bn_max_gain = getattr(gui_instance, 'bn_max_gain_var', tk.DoubleVar(value=default_values_for_fallback['bn_max_gain'])).get()

            self.cb_border_size = getattr(gui_instance, 'cb_border_size_var', tk.IntVar(value=default_values_for_fallback['cb_border_size'])).get()
            self.cb_blur_radius = getattr(gui_instance, 'cb_blur_radius_var', tk.IntVar(value=default_values_for_fallback['cb_blur_radius'])).get()
            self.cb_min_b_factor = getattr(gui_instance, 'cb_min_b_factor_var', tk.DoubleVar(value=default_values_for_fallback['cb_min_b_factor'])).get()
            self.cb_max_b_factor = getattr(gui_instance, 'cb_max_b_factor_var', tk.DoubleVar(value=default_values_for_fallback['cb_max_b_factor'])).get()

            self.final_edge_crop_percent = getattr(gui_instance, 'final_edge_crop_percent_var', tk.DoubleVar(value=default_values_for_fallback['final_edge_crop_percent'])).get()

            self.apply_photutils_bn = getattr(gui_instance, 'apply_photutils_bn_var', tk.BooleanVar(value=default_values_for_fallback['apply_photutils_bn'])).get()
            self.photutils_bn_box_size = getattr(gui_instance, 'photutils_bn_box_size_var', tk.IntVar(value=default_values_for_fallback['photutils_bn_box_size'])).get()
            self.photutils_bn_filter_size = getattr(gui_instance, 'photutils_bn_filter_size_var', tk.IntVar(value=default_values_for_fallback['photutils_bn_filter_size'])).get()
            self.photutils_bn_sigma_clip = getattr(gui_instance, 'photutils_bn_sigma_clip_var', tk.DoubleVar(value=default_values_for_fallback['photutils_bn_sigma_clip'])).get()
            self.photutils_bn_exclude_percentile = getattr(gui_instance, 'photutils_bn_exclude_percentile_var', tk.DoubleVar(value=default_values_for_fallback['photutils_bn_exclude_percentile'])).get()

            self.apply_feathering = getattr(gui_instance, 'apply_feathering_var', tk.BooleanVar(value=default_values_for_fallback['apply_feathering'])).get()
            self.feather_blur_px = getattr(gui_instance, 'feather_blur_px_var', tk.IntVar(value=default_values_for_fallback['feather_blur_px'])).get()

            self.apply_low_wht_mask = getattr(gui_instance, 'apply_low_wht_mask_var', tk.BooleanVar(value=default_values_for_fallback['apply_low_wht_mask'])).get()
            self.low_wht_percentile = getattr(gui_instance, 'low_wht_pct_var', tk.IntVar(value=default_values_for_fallback['low_wht_percentile'])).get()
            self.low_wht_soften_px = getattr(gui_instance, 'low_wht_soften_px_var', tk.IntVar(value=default_values_for_fallback['low_wht_soften_px'])).get()

            # --- MODIFIÉ: Gestion pour astap_search_radius et autres settings des solveurs locaux ---
            # Ces settings sont modifiés via LocalSolverSettingsWindow, qui met à jour self.settings directement.
            # Donc, ici, on CONSERVE leur valeur actuelle sur self.settings.
            params_from_modals = ['use_local_solver_priority', 'astap_path', 'astap_data_dir', 'local_ansvr_path', 'astap_search_radius']
            for param_name in params_from_modals:
                current_val = getattr(self, param_name, "NON_DEFINI_SUR_SELF_POUR_MODAL_PARAMS")
                print(f"  DEBUG SM (update_from_ui): Paramètre modal '{param_name}': Conservation valeur actuelle: {current_val}")
            # --- FIN MODIFICATION ---


            # --- Preview Settings ---
            self.preview_stretch_method = getattr(gui_instance, 'preview_stretch_method', tk.StringVar(value=default_values_for_fallback['preview_stretch_method'])).get()
            # ... (autres pour Preview)
            self.preview_black_point = getattr(gui_instance, 'preview_black_point', tk.DoubleVar(value=default_values_for_fallback['preview_black_point'])).get()
            self.preview_white_point = getattr(gui_instance, 'preview_white_point', tk.DoubleVar(value=default_values_for_fallback['preview_white_point'])).get()
            self.preview_gamma = getattr(gui_instance, 'preview_gamma', tk.DoubleVar(value=default_values_for_fallback['preview_gamma'])).get()
            self.preview_r_gain = getattr(gui_instance, 'preview_r_gain', tk.DoubleVar(value=default_values_for_fallback['preview_r_gain'])).get()
            self.preview_g_gain = getattr(gui_instance, 'preview_g_gain', tk.DoubleVar(value=default_values_for_fallback['preview_g_gain'])).get()
            self.preview_b_gain = getattr(gui_instance, 'preview_b_gain', tk.DoubleVar(value=default_values_for_fallback['preview_b_gain'])).get()
            self.preview_brightness = getattr(gui_instance, 'preview_brightness', tk.DoubleVar(value=1.0)).get()
            self.preview_contrast = getattr(gui_instance, 'preview_contrast', tk.DoubleVar(value=1.0)).get()
            self.preview_saturation = getattr(gui_instance, 'preview_saturation', tk.DoubleVar(value=1.0)).get()

            # --- UI Settings ---
            self.language = getattr(gui_instance, 'language_var', tk.StringVar(value=default_values_for_fallback['language'])).get()
            if gui_instance.root.winfo_exists():
                 current_geo_ui = gui_instance.root.geometry()
                 if isinstance(current_geo_ui, str) and 'x' in current_geo_ui and '+' in current_geo_ui:
                     self.window_geometry = current_geo_ui

            print("DEBUG SM (update_from_ui V5): Fin lecture UI.")

        except AttributeError as ae:
            print(f"Error SM (update_from_ui V5) (AttributeError): {ae}. Un widget/variable Tk est peut-être manquant sur gui_instance.")
            traceback.print_exc(limit=1)
        except tk.TclError as te:
            print(f"Error SM (update_from_ui V5) (TclError): {te}. Un widget Tk est peut-être détruit.")
        except Exception as e:
            print(f"Unexpected error SM (update_from_ui V5): {e}")
            traceback.print_exc(limit=2)



#######################################################################################################################################


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
            # --- Appliquer les paramètres de feathering à l'UI ---
            # Assurez-vous que gui_instance a des variables apply_feathering_var et feather_blur_px_var
            if hasattr(gui_instance, 'apply_feathering_var'):
                getattr(gui_instance, 'apply_feathering_var', tk.BooleanVar()).set(self.apply_feathering)
                print(f"DEBUG (Settings apply_to_ui): Apply Feathering appliqué à UI: {self.apply_feathering}")
            else:
                print("DEBUG (Settings apply_to_ui): gui_instance n'a pas apply_feathering_var")

            if hasattr(gui_instance, 'feather_blur_px_var'):
                getattr(gui_instance, 'feather_blur_px_var', tk.IntVar()).set(self.feather_blur_px)
                print(f"DEBUG (Settings apply_to_ui): Feather Blur Px appliqué à UI: {self.feather_blur_px}")
                # --- NOUVEAU : Application Low WHT Mask à l'UI ---
            if hasattr(gui_instance, 'apply_low_wht_mask_var'):
                getattr(gui_instance, 'apply_low_wht_mask_var', tk.BooleanVar()).set(self.apply_low_wht_mask)
            if hasattr(gui_instance, 'low_wht_pct_var'):
                getattr(gui_instance, 'low_wht_pct_var', tk.IntVar()).set(self.low_wht_percentile)
            if hasattr(gui_instance, 'low_wht_soften_px_var'): # Sera ajouté à l'étape UI
                getattr(gui_instance, 'low_wht_soften_px_var', tk.IntVar()).set(self.low_wht_soften_px)
                print(f"DEBUG (Settings apply_to_ui): LowWHT Mask appliqué à UI -> Apply: {self.apply_low_wht_mask}, Pct: {self.low_wht_percentile}, Soften: {self.low_wht_soften_px}")
            # --- FIN NOUVEAU ---
            else:
                print("DEBUG (Settings apply_to_ui): gui_instance n'a pas feather_blur_px_var")
            
            # Appeler la méthode pour mettre à jour l'état des widgets feathering si elle existe
            if hasattr(gui_instance, '_update_feathering_options_state'):
                gui_instance._update_feathering_options_state()
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

            # Mettre à jour l'état des widgets dépendants
            if hasattr(gui_instance, '_update_weighting_options_state'): gui_instance._update_weighting_options_state()
            if hasattr(gui_instance, '_update_drizzle_options_state'): gui_instance._update_drizzle_options_state()
            if hasattr(gui_instance, '_update_final_scnr_options_state'): gui_instance._update_final_scnr_options_state()
            if hasattr(gui_instance, '_update_photutils_bn_options_state'): gui_instance._update_photutils_bn_options_state()
            if hasattr(gui_instance, '_update_feathering_options_state'): gui_instance._update_feathering_options_state()
            # NOUVEAU: Mettre à jour l'état des options Low WHT Mask (si une telle méthode est créée)
            if hasattr(gui_instance, '_update_low_wht_mask_options_state'): # Sera créé à l'étape UI
                gui_instance._update_low_wht_mask_options_state()

            # --- NOUVEAU: Paramètres Solveurs Locaux Astrométriques ---
            #defaults_dict['use_local_solver_priority'] = False # Si True, essayer solveurs locaux AVANT Astrometry.net
            #defaults_dict['astap_path'] = ""                   # Chemin vers l'exécutable ASTAP
            #defaults_dict['astap_data_dir'] = ""               # Chemin vers le dossier de données d'index ASTAP (ex: G17, H18)
            # Pour Astrometry.net local, la configuration est plus complexe.
            # On pourrait stocker le chemin vers un fichier de config ou un répertoire racine.
            # Pour l'instant, on met un placeholder. On affinera si besoin.
            #defaults_dict['local_ansvr_path'] = ""             # Chemin vers config/exécutable de ansvr local
            # On pourrait aussi avoir des booléens pour activer spécifiquement chaque solveur si on veut
            # les désactiver même si les chemins sont remplis, mais 'use_local_solver_priority' est un bon début.

            #print(f"DEBUG (SettingsManager get_default_values): Ajout des défauts pour solveurs locaux.")
            #return defaults_dict
            # --- NOUVEAU: Application astap_search_radius à l'UI ---
            # On suppose que gui_instance aura une variable 'astap_search_radius_var'
            # Si elle n'existe pas, getattr retournera une DoubleVar temporaire qui ne sera pas utilisée.
            # Si elle existe, sa valeur sera mise à jour.

            getattr(gui_instance, 'astap_search_radius_var', tk.DoubleVar()).set(self.astap_search_radius)
            print(f"DEBUG (Settings apply_to_ui): astap_search_radius appliqué à l'UI (valeur: {self.astap_search_radius})")
            # --- FIN NOUVEAU ---
            print("DEBUG (Settings apply_to_ui): Fin application paramètres UI.") # <-- AJOUTÉ DEBUG

        # ... (gestion erreurs inchangée) ...
        except AttributeError as ae: print(f"Error applying settings to UI (AttributeError): {ae}")
        except tk.TclError as te: print(f"Error applying settings to UI (TclError - widget likely destroyed?): {te}")
        except Exception as e: print(f"Unexpected error applying settings to UI: {e}"); traceback.print_exc(limit=2)

#################################################################################################################################







# --- DANS LA CLASSE SettingsManager DANS seestar/gui/settings.py ---

    def get_default_values(self):
        """ 
        Retourne un dictionnaire des valeurs par défaut de tous les paramètres.
        MAJ: Ajout des nouveaux paramètres pour les modes d'alignement mosaïque et FastAligner.
        """
        print("DEBUG (SettingsManager get_default_values): Récupération des valeurs par défaut...") # Log mis à jour
        defaults_dict = {}
        
        # --- Paramètres de Traitement de Base ---
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

        # --- Paramètres de Pondération par Qualité ---
        defaults_dict['use_quality_weighting'] = True 
        defaults_dict['weight_by_snr'] = True
        defaults_dict['weight_by_stars'] = True
        defaults_dict['snr_exponent'] = 1.8     
        defaults_dict['stars_exponent'] = 0.5   
        defaults_dict['min_weight'] = 0.01      

        # --- Paramètres Drizzle (Globaux) ---
        defaults_dict['use_drizzle'] = False # Drizzle global, pas spécifique mosaïque
        defaults_dict['drizzle_scale'] = 2
        defaults_dict['drizzle_wht_threshold'] = 0.7 
        defaults_dict['drizzle_mode'] = "Final" 
        defaults_dict['drizzle_kernel'] = "square" # Kernel Drizzle global
        defaults_dict['drizzle_pixfrac'] = 1.0    # Pixfrac Drizzle global

        # --- Paramètres de Correction Couleur et Post-Traitement ---
        # (Ceux-ci sont globaux, pas spécifiques à la mosaïque)
        defaults_dict['apply_chroma_correction'] = True 
        defaults_dict['apply_final_scnr'] = True        
        defaults_dict['final_scnr_target_channel'] = 'green'
        defaults_dict['final_scnr_amount'] = 0.6        
        defaults_dict['final_scnr_preserve_luminosity'] = True 
        defaults_dict['bn_grid_size_str'] = "24x24"     
        defaults_dict['bn_perc_low'] = 5
        defaults_dict['bn_perc_high'] = 40              
        defaults_dict['bn_std_factor'] = 1.5            
        defaults_dict['bn_min_gain'] = 0.2
        defaults_dict['bn_max_gain'] = 7.0
        defaults_dict['cb_border_size'] = 25
        defaults_dict['cb_blur_radius'] = 8
        defaults_dict['cb_min_b_factor'] = 0.4 
        defaults_dict['cb_max_b_factor'] = 1.5 
        defaults_dict['final_edge_crop_percent'] = 2.0 
        defaults_dict['apply_photutils_bn'] = False      
        defaults_dict['photutils_bn_box_size'] = 128
        defaults_dict['photutils_bn_filter_size'] = 11  
        defaults_dict['photutils_bn_sigma_clip'] = 3.0
        defaults_dict['photutils_bn_exclude_percentile'] = 95.0 
        defaults_dict['apply_feathering'] = True 
        defaults_dict['feather_blur_px'] = 256   
        defaults_dict['apply_low_wht_mask'] = False   
        defaults_dict['low_wht_percentile'] = 5       
        defaults_dict['low_wht_soften_px'] = 128      
        defaults_dict['use_local_solver_priority'] = False
        defaults_dict['astap_path'] = ""                  
        defaults_dict['astap_data_dir'] = ""              
        defaults_dict['local_ansvr_path'] = ""
        # ---  Rayon de recherche ASTAP ---
        defaults_dict['astap_search_radius'] = 5.0 # Valeur par défaut en degrés
        # ---  ---
        # --- Paramètres Mosaïque & Astrometry.net ---
        defaults_dict['mosaic_mode_active'] = False # Ce flag est pour l'UI principale
        
        # Le dictionnaire `mosaic_settings` contiendra TOUS les paramètres spécifiques à la mosaïque
        # y compris ceux pour Drizzle ET ceux pour l'alignement des panneaux.
        defaults_dict['mosaic_settings'] = {
            # Paramètres Drizzle spécifiques à la mosaïque (ceux que vous aviez déjà)
            "kernel": "square",  # Valeur par défaut pour le kernel Drizzle de la mosaïque
            "pixfrac": 0.8,      # Valeur par défaut pour le pixfrac Drizzle de la mosaïque
            # "fillval" et "wht_threshold" peuvent être ajoutés ici aussi si vous les exposez dans MosaicSettingsWindow pour Drizzle
            "fillval": "0.0",       # Ajouté, valeur par défaut
            "wht_threshold": 0.01,  # Ajouté, valeur par défaut (0-1)

            # NOUVEAU: Paramètres pour le mode d'alignement de la mosaïque
            "alignment_mode": "local_fast_fallback", # Valeurs possibles: "local_fast_fallback", "local_fast_only", "astrometry_per_panel"
            
            # NOUVEAU: Paramètres pour FastAligner (utilisés si alignment_mode est local*)
            "fastalign_orb_features": 3000,
            "fastalign_min_abs_matches": 8,
            "fastalign_min_ransac": 4, # La valeur brute (ex: 4, pas le calcul max(...))
            "fastalign_ransac_thresh": 2.5,
            "fastalign_dao_fwhm": 3.5,       # Valeur par défaut pour FWHM DAO
            "fastalign_dao_thr_sig": 8.0,    # NOUVELLE VALEUR PAR DÉFAUT PLUS ÉLEVÉE pour le facteur sigma
            "fastalign_dao_max_stars": 750,  # Valeur par défaut pour max étoiles DAO
            # --- FIN AJOUT ---
            # 
            #     
            # Clé API est stockée séparément au niveau global mais aussi ici pour la fenêtre mosaïque
            # Si on veut que MosaicSettingsWindow ait sa propre copie de la clé utilisée lors de sa dernière validation "OK"
            # Cela peut être redondant avec self.astrometry_api_key global.
            # Pour l'instant, on peut se fier à la clé API globale et ne pas la dupliquer ici.
            # "api_key": "" # Optionnel: si on veut que MosaicSettingsWindow se souvienne de sa propre clé
        }
        
        defaults_dict['astrometry_api_key'] = "" # Clé API globale

        # --- Paramètres de Prévisualisation ---
        defaults_dict['preview_stretch_method'] = "Asinh"
        defaults_dict['preview_black_point'] = 0.01
        defaults_dict['preview_white_point'] = 0.99
        defaults_dict['preview_gamma'] = 1.0
        defaults_dict['preview_r_gain'] = 1.0
        defaults_dict['preview_g_gain'] = 1.0
        defaults_dict['preview_b_gain'] = 1.0
        
        # --- Paramètres de l'Interface Utilisateur ---
        defaults_dict['language'] = 'en' 
        defaults_dict['window_geometry'] = "1200x750" 

        print(f"DEBUG (SettingsManager get_default_values): Dictionnaire de défauts créé.")
        # print(f"  Exemple mosaic_settings par défaut: {defaults_dict['mosaic_settings']}") # Pour vérifier
        return defaults_dict






###################################################################################################################################


    def validate_settings(self):
        """Valide et corrige les paramètres si nécessaire. Retourne les messages de correction."""
        messages = []
        print("DEBUG (Settings validate_settings): DÉBUT de la validation.")

        # Obtenir les valeurs par défaut pour fallback SANS modifier l'instance actuelle
        defaults_fallback = self.get_default_values()
        # Log initial pour un paramètre spécifique afin de suivre son état
        print(f"DEBUG (Settings validate_settings): Valeur self.apply_photutils_bn AVANT TOUTE VALIDATION (lue de l'UI): {getattr(self, 'apply_photutils_bn', 'NON_DEFINI_ENCORE')}")

        try:
            # --- Processing Settings Validation ---
            print("  -> Validating Processing Settings...")
            try:
                self.kappa = float(self.kappa)
                if not (1.0 <= self.kappa <= 5.0):
                    original = self.kappa; self.kappa = np.clip(self.kappa, 1.0, 5.0)
                    messages.append(f"Kappa ({original:.1f}) ajusté à {self.kappa:.1f}")
            except (ValueError, TypeError):
                original = self.kappa; self.kappa = defaults_fallback['kappa']
                messages.append(f"Kappa ('{original}') invalide, réinitialisé à {self.kappa:.1f}")

            try:
                self.batch_size = int(self.batch_size)
                if self.batch_size < 0: # 0 est permis pour auto-estimation
                    original = self.batch_size; self.batch_size = 0
                    messages.append(f"Taille Lot ({original}) ajusté à {self.batch_size} (auto)")
            except (ValueError, TypeError):
                original = self.batch_size; self.batch_size = defaults_fallback['batch_size']
                messages.append(f"Taille Lot ('{original}') invalide, réinitialisé à {self.batch_size}")

            try:
                self.hot_pixel_threshold = float(self.hot_pixel_threshold)
                if not (0.5 <= self.hot_pixel_threshold <= 10.0):
                    original = self.hot_pixel_threshold; self.hot_pixel_threshold = np.clip(self.hot_pixel_threshold, 0.5, 10.0)
                    messages.append(f"Seuil Px Chauds ({original:.1f}) ajusté à {self.hot_pixel_threshold:.1f}")
            except (ValueError, TypeError):
                original = self.hot_pixel_threshold; self.hot_pixel_threshold = defaults_fallback['hot_pixel_threshold']
                messages.append(f"Seuil Px Chauds ('{original}') invalide, réinitialisé à {self.hot_pixel_threshold:.1f}")

            try:
                self.neighborhood_size = int(self.neighborhood_size)
                if self.neighborhood_size < 3:
                    original = self.neighborhood_size; self.neighborhood_size = 3
                    messages.append(f"Voisinage Px Chauds ({original}) ajusté à {self.neighborhood_size}")
                if self.neighborhood_size % 2 == 0:
                    original = self.neighborhood_size; self.neighborhood_size += 1
                    messages.append(f"Voisinage Px Chauds ({original}) ajusté à {self.neighborhood_size} (impair)")
            except (ValueError, TypeError):
                original = self.neighborhood_size; self.neighborhood_size = defaults_fallback['neighborhood_size']
                messages.append(f"Voisinage Px Chauds ('{original}') invalide, réinitialisé à {self.neighborhood_size}")

            # --- Quality Weighting Validation ---
            print("  -> Validating Quality Weighting Settings...")
            self.use_quality_weighting = bool(getattr(self, 'use_quality_weighting', defaults_fallback['use_quality_weighting']))
            self.weight_by_snr = bool(getattr(self, 'weight_by_snr', defaults_fallback['weight_by_snr']))
            self.weight_by_stars = bool(getattr(self, 'weight_by_stars', defaults_fallback['weight_by_stars']))
            try:
                self.snr_exponent = float(self.snr_exponent)
                if self.snr_exponent <= 0:
                    original = self.snr_exponent; self.snr_exponent = defaults_fallback['snr_exponent']
                    messages.append(f"Exposant SNR ({original:.1f}) ajusté à {self.snr_exponent:.1f}")
            except (ValueError, TypeError):
                original = self.snr_exponent; self.snr_exponent = defaults_fallback['snr_exponent']
                messages.append(f"Exposant SNR ('{original}') invalide, réinitialisé à {self.snr_exponent:.1f}")
            try:
                self.stars_exponent = float(self.stars_exponent)
                if self.stars_exponent <= 0:
                    original = self.stars_exponent; self.stars_exponent = defaults_fallback['stars_exponent']
                    messages.append(f"Exposant Étoiles ({original:.1f}) ajusté à {self.stars_exponent:.1f}")
            except (ValueError, TypeError):
                original = self.stars_exponent; self.stars_exponent = defaults_fallback['stars_exponent']
                messages.append(f"Exposant Étoiles ('{original}') invalide, réinitialisé à {self.stars_exponent:.1f}")
            try:
                self.min_weight = float(self.min_weight)
                if not (0 < self.min_weight <= 1.0):
                    original = self.min_weight; self.min_weight = np.clip(self.min_weight, 0.01, 1.0)
                    messages.append(f"Poids Min ({original:.2f}) ajusté à {self.min_weight:.2f}")
            except (ValueError, TypeError):
                original = self.min_weight; self.min_weight = defaults_fallback['min_weight']
                messages.append(f"Poids Min ('{original}') invalide, réinitialisé à {self.min_weight:.2f}")

            if self.use_quality_weighting and not (self.weight_by_snr or self.weight_by_stars):
                self.weight_by_snr = True
                messages.append("Pondération activée mais aucune métrique choisie. SNR activé par défaut.")

            # --- Drizzle Settings Validation ---
            print("  -> Validating Drizzle Settings...")
            self.use_drizzle = bool(getattr(self, 'use_drizzle', defaults_fallback['use_drizzle']))
            try:
                scale_num = int(float(self.drizzle_scale)) # Tenter de convertir même si c'est un string
                if scale_num not in [2, 3, 4]:
                    original = self.drizzle_scale; self.drizzle_scale = defaults_fallback['drizzle_scale']
                    messages.append(f"Échelle Drizzle ({original}) invalide, réinitialisée à {self.drizzle_scale}")
                else: self.drizzle_scale = scale_num
            except (ValueError, TypeError):
                original = self.drizzle_scale; self.drizzle_scale = defaults_fallback['drizzle_scale']
                messages.append(f"Échelle Drizzle invalide ('{original}'), réinitialisée à {self.drizzle_scale}")
            try:
                self.drizzle_wht_threshold = float(self.drizzle_wht_threshold)
                if not (0.0 < self.drizzle_wht_threshold <= 1.0):
                    original = self.drizzle_wht_threshold; self.drizzle_wht_threshold = np.clip(self.drizzle_wht_threshold, 0.1, 1.0)
                    messages.append(f"Seuil Drizzle WHT ({original:.2f}) ajusté à {self.drizzle_wht_threshold:.2f}")
            except (ValueError, TypeError):
                original = self.drizzle_wht_threshold; self.drizzle_wht_threshold = defaults_fallback['drizzle_wht_threshold']
                messages.append(f"Seuil Drizzle WHT invalide ('{original}'), réinitialisé à {self.drizzle_wht_threshold:.2f}")

            valid_drizzle_modes = ["Final", "Incremental"]
            current_driz_mode = getattr(self, 'drizzle_mode', defaults_fallback['drizzle_mode'])
            if not isinstance(current_driz_mode, str) or current_driz_mode not in valid_drizzle_modes:
                original = current_driz_mode; self.drizzle_mode = defaults_fallback['drizzle_mode']
                messages.append(f"Mode Drizzle ({original}) invalide, réinitialisé à '{self.drizzle_mode}'")
            else:
                self.drizzle_mode = current_driz_mode # S'assurer qu'il est bien sur self

            valid_kernels = ['square', 'gaussian', 'point', 'tophat', 'turbo', 'lanczos2', 'lanczos3']
            current_driz_kernel = getattr(self, 'drizzle_kernel', defaults_fallback['drizzle_kernel'])
            if not isinstance(current_driz_kernel, str) or current_driz_kernel.lower() not in valid_kernels:
                original = current_driz_kernel; self.drizzle_kernel = defaults_fallback['drizzle_kernel']
                messages.append(f"Noyau Drizzle ('{original}') invalide, réinitialisé à '{self.drizzle_kernel}'")
            else:
                self.drizzle_kernel = current_driz_kernel.lower()
            try:
                self.drizzle_pixfrac = float(self.drizzle_pixfrac)
                if not (0.01 <= self.drizzle_pixfrac <= 1.0):
                    original = self.drizzle_pixfrac; self.drizzle_pixfrac = np.clip(self.drizzle_pixfrac, 0.01, 1.0)
                    messages.append(f"Pixfrac Drizzle ({original:.2f}) hors limites [0.01, 1.0], ajusté à {self.drizzle_pixfrac:.2f}")
            except (ValueError, TypeError):
                original = self.drizzle_pixfrac; self.drizzle_pixfrac = defaults_fallback['drizzle_pixfrac']
                messages.append(f"Pixfrac Drizzle ('{original}') invalide, réinitialisé à {self.drizzle_pixfrac:.2f}")

            # --- SCNR Final Validation ---
            print("  -> Validating SCNR Settings...")
            self.apply_final_scnr = bool(getattr(self, 'apply_final_scnr', defaults_fallback['apply_final_scnr']))
            self.final_scnr_target_channel = str(getattr(self, 'final_scnr_target_channel', defaults_fallback['final_scnr_target_channel'])).lower()
            if self.final_scnr_target_channel not in ['green', 'blue']:
                original_target = self.final_scnr_target_channel; self.final_scnr_target_channel = defaults_fallback['final_scnr_target_channel']
                messages.append(f"Cible SCNR Final ('{original_target}') invalide, réinitialisée à '{self.final_scnr_target_channel}'.")
            try:
                self.final_scnr_amount = float(self.final_scnr_amount)
                if not (0.0 <= self.final_scnr_amount <= 1.0):
                    original_amount = self.final_scnr_amount; self.final_scnr_amount = np.clip(self.final_scnr_amount, 0.0, 1.0)
                    messages.append(f"Intensité SCNR Final ({original_amount:.2f}) hors limites [0.0, 1.0], ajustée à {self.final_scnr_amount:.2f}.")
            except (ValueError, TypeError):
                original_amount = self.final_scnr_amount; self.final_scnr_amount = defaults_fallback['final_scnr_amount']
                messages.append(f"Intensité SCNR Final ('{original_amount}') invalide, réinitialisée à {self.final_scnr_amount:.2f}.")
            self.final_scnr_preserve_luminosity = bool(getattr(self, 'final_scnr_preserve_luminosity', defaults_fallback['final_scnr_preserve_luminosity']))

            # --- Expert Settings Validation ---
            print("  -> Validating Expert Settings...")
            # BN
            current_bn_grid = getattr(self, 'bn_grid_size_str', defaults_fallback['bn_grid_size_str'])
            if not isinstance(current_bn_grid, str) or current_bn_grid not in ["8x8", "16x16", "24x24", "32x32", "64x64"]:
                messages.append(f"Taille grille BN invalide ('{current_bn_grid}'), réinitialisée."); self.bn_grid_size_str = defaults_fallback['bn_grid_size_str']
            else: self.bn_grid_size_str = current_bn_grid
            self.bn_perc_low = int(np.clip(getattr(self, 'bn_perc_low', defaults_fallback['bn_perc_low']), 0, 40))
            self.bn_perc_high = int(np.clip(getattr(self, 'bn_perc_high', defaults_fallback['bn_perc_high']), self.bn_perc_low + 1, 90))
            self.bn_std_factor = float(np.clip(getattr(self, 'bn_std_factor', defaults_fallback['bn_std_factor']), 0.1, 10.0))
            self.bn_min_gain = float(np.clip(getattr(self, 'bn_min_gain', defaults_fallback['bn_min_gain']), 0.05, 5.0))
            self.bn_max_gain = float(np.clip(getattr(self, 'bn_max_gain', defaults_fallback['bn_max_gain']), self.bn_min_gain, 20.0))
            # CB
            self.cb_border_size = int(np.clip(getattr(self, 'cb_border_size', defaults_fallback['cb_border_size']), 5, 200))
            self.cb_blur_radius = int(np.clip(getattr(self, 'cb_blur_radius', defaults_fallback['cb_blur_radius']), 0, 100))
            self.cb_min_b_factor = float(np.clip(getattr(self, 'cb_min_b_factor', defaults_fallback['cb_min_b_factor']), 0.1, 1.0))
            self.cb_max_b_factor = float(np.clip(getattr(self, 'cb_max_b_factor', defaults_fallback['cb_max_b_factor']), self.cb_min_b_factor, 5.0))
            # Rognage
            self.final_edge_crop_percent = float(np.clip(getattr(self, 'final_edge_crop_percent', defaults_fallback['final_edge_crop_percent']), 0.0, 25.0))

            # Photutils BN
            print("    -> Validating Photutils BN...")
            self.apply_photutils_bn = bool(getattr(self, 'apply_photutils_bn', defaults_fallback['apply_photutils_bn']))
            self.photutils_bn_box_size = int(np.clip(getattr(self, 'photutils_bn_box_size', defaults_fallback['photutils_bn_box_size']), 8, 1024))
            self.photutils_bn_filter_size = int(np.clip(getattr(self, 'photutils_bn_filter_size', defaults_fallback['photutils_bn_filter_size']), 1, 25))
            if self.photutils_bn_filter_size % 2 == 0: self.photutils_bn_filter_size += 1
            self.photutils_bn_sigma_clip = float(np.clip(getattr(self, 'photutils_bn_sigma_clip', defaults_fallback['photutils_bn_sigma_clip']), 0.5, 10.0))
            self.photutils_bn_exclude_percentile = float(np.clip(getattr(self, 'photutils_bn_exclude_percentile', defaults_fallback['photutils_bn_exclude_percentile']), 0.0, 100.0))

            # Astrometry API Key
            current_api_key = getattr(self, 'astrometry_api_key', defaults_fallback['astrometry_api_key'])
            if not isinstance(current_api_key, str):
                messages.append("Clé API Astrometry invalide (pas une chaîne), réinitialisée.")
                self.astrometry_api_key = defaults_fallback['astrometry_api_key']
            else: self.astrometry_api_key = current_api_key.strip()


            # Feathering
            print("    -> Validating Feathering...")
            self.apply_feathering = bool(getattr(self, 'apply_feathering', defaults_fallback['apply_feathering']))
            try:
                self.feather_blur_px = int(self.feather_blur_px)
                min_blur, max_blur = 32, 1024
                if not (min_blur <= self.feather_blur_px <= max_blur):
                    original_blur = self.feather_blur_px; self.feather_blur_px = int(np.clip(self.feather_blur_px, min_blur, max_blur))
                    messages.append(f"Feather Blur Px ({original_blur}) hors limites [{min_blur}-{max_blur}], ajusté à {self.feather_blur_px}.")
            except (ValueError, TypeError):
                original_blur = self.feather_blur_px; self.feather_blur_px = defaults_fallback['feather_blur_px']
                messages.append(f"Feather Blur Px ('{original_blur}') invalide, réinitialisé à {self.feather_blur_px}.")

            # Low WHT Mask
            print("    -> Validating Low WHT Mask...")
            self.apply_low_wht_mask = bool(getattr(self, 'apply_low_wht_mask', defaults_fallback['apply_low_wht_mask']))
            try:
                self.low_wht_percentile = int(self.low_wht_percentile)
                min_pct, max_pct = 1, 100
                if not (min_pct <= self.low_wht_percentile <= max_pct):
                    original_pct = self.low_wht_percentile; self.low_wht_percentile = int(np.clip(self.low_wht_percentile, min_pct, max_pct))
                    messages.append(f"Low WHT Percentile ({original_pct}) hors limites [{min_pct}-{max_pct}], ajusté à {self.low_wht_percentile}.")
            except (ValueError, TypeError):
                original_pct = self.low_wht_percentile; self.low_wht_percentile = defaults_fallback['low_wht_percentile']
                messages.append(f"Low WHT Percentile ('{original_pct}') invalide, réinitialisé à {self.low_wht_percentile}.")
            try:
                self.low_wht_soften_px = int(self.low_wht_soften_px)
                min_soften, max_soften = 32, 512
                if not (min_soften <= self.low_wht_soften_px <= max_soften):
                    original_soften = self.low_wht_soften_px; self.low_wht_soften_px = int(np.clip(self.low_wht_soften_px, min_soften, max_soften))
                    messages.append(f"Low WHT Soften Px ({original_soften}) hors limites [{min_soften}-{max_soften}], ajusté à {self.low_wht_soften_px}.")
            except (ValueError, TypeError):
                original_soften = self.low_wht_soften_px; self.low_wht_soften_px = defaults_fallback['low_wht_soften_px']
                messages.append(f"Low WHT Soften Px ('{original_soften}') invalide, réinitialisé à {self.low_wht_soften_px}.")

            # --- Chroma Correction (simple booléen) ---
            self.apply_chroma_correction = bool(getattr(self, 'apply_chroma_correction', defaults_fallback['apply_chroma_correction']))
 # --- Local Solver Paths and ASTAP Search Radius ---
            print("  -> Validating Local Solver Settings...")
            # ... (validation des autres paramètres use_local_solver_priority, astap_path, etc. inchangée) ...
            self.use_local_solver_priority = bool(getattr(self, 'use_local_solver_priority', defaults_fallback['use_local_solver_priority']))
            self.astap_path = str(getattr(self, 'astap_path', defaults_fallback['astap_path'])).strip()
            self.astap_data_dir = str(getattr(self, 'astap_data_dir', defaults_fallback['astap_data_dir'])).strip()
            self.local_ansvr_path = str(getattr(self, 'local_ansvr_path', defaults_fallback['local_ansvr_path'])).strip()

            # --- Validation spécifique et logging pour astap_search_radius ---
            param_name_debug = 'astap_search_radius'
            value_before_validation = getattr(self, param_name_debug, "ATTRIBUT_MANQUANT_SUR_SELF_POUR_VALIDATE")
            print(f"    DEBUG VALIDATE: Valeur de self.{param_name_debug} AVANT float() et clip: '{value_before_validation}' (type: {type(value_before_validation)})")

            try:
                # Utiliser la valeur lue, ou le défaut du code si l'attribut n'existe pas pour une raison étrange
                current_radius_val = getattr(self, param_name_debug, defaults_fallback[param_name_debug])
                
                # Essayer de convertir en float. Si cela échoue, on ira au bloc except.
                validated_radius = float(current_radius_val)
                
                min_r, max_r = 0.1, 90.0
                if not (min_r <= validated_radius <= max_r):
                    original_radius_str = f"{validated_radius:.1f}" # On sait que c'est un float ici
                    self.astap_search_radius = np.clip(validated_radius, min_r, max_r)
                    messages.append(f"Rayon recherche ASTAP ({original_radius_str}°) hors limites [{min_r}-{max_r}], ajusté à {self.astap_search_radius:.1f}°")
                    print(f"    DEBUG VALIDATE: Rayon clippé à {self.astap_search_radius:.1f}°")
                else:
                    # La valeur est déjà un float et dans la bonne plage
                    self.astap_search_radius = validated_radius
                    print(f"    DEBUG VALIDATE: Rayon déjà valide: {self.astap_search_radius:.1f}°")
            except (ValueError, TypeError) as e_val_rad:
                original_radius_str = str(getattr(self, param_name_debug, 'N/A_DANS_EXCEPT'))
                self.astap_search_radius = defaults_fallback[param_name_debug] # Réinitialiser au défaut du code
                messages.append(f"Rayon recherche ASTAP ('{original_radius_str}') invalide (erreur: {e_val_rad}), réinitialisé à {self.astap_search_radius:.1f}°")
                print(f"    DEBUG VALIDATE: Exception lors de la validation du rayon ('{original_radius_str}'), réinitialisé à {self.astap_search_radius:.1f}°")
            
            print(f"DEBUG (Settings validate_settings): {param_name_debug} FINAL après validation: {getattr(self, param_name_debug, 'ERREUR_ATTR_FINAL')}°")

        except Exception as e_global_val: # Attrape les erreurs non prévues pendant la validation
            messages.append(f"Erreur générale de validation: {e_global_val}. Réinitialisation aux valeurs par défaut.")
            print(f"FATAL Warning (Settings validate_settings): Erreur de validation globale -> {e_global_val}. Réinitialisation complète des settings.")
            self.reset_to_defaults()
            # Logguer la valeur de apply_photutils_bn APRES cette réinitialisation globale
            print(f"DEBUG (Settings validate_settings): self.apply_photutils_bn APRES reset_to_defaults (global catch): {getattr(self, 'apply_photutils_bn', 'ATTRIBUT_MANQUANT_APRES_RESET')}")

        print(f"DEBUG (Settings validate_settings): FIN de la validation. Nombre de messages: {len(messages)}. "
              f"Valeur finale de self.apply_photutils_bn: {getattr(self, 'apply_photutils_bn', 'NON_DEFINI_A_LA_FIN')}")
        return messages


###################################################################################################################################

    def save_settings(self):
        """ Sauvegarde les paramètres actuels dans le fichier JSON. """
        # S'assurer que les types sont corrects pour JSON
        settings_data = {
            'version': "2.1.0", # Incrémenter la version si la structure change significativement
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
            # --- Sauvegarder les paramètres de feathering ---
            'apply_feathering': bool(self.apply_feathering),
            'feather_blur_px': int(self.feather_blur_px),
            # --- Sauvegarde Low WHT Mask ---
            'apply_low_wht_mask': bool(self.apply_low_wht_mask),
            'low_wht_percentile': int(self.low_wht_percentile),
            'low_wht_soften_px': int(self.low_wht_soften_px),
            # --- --
            # --- NOUVEAU: Sauvegarde Paramètres Solveurs Locaux ---
            'use_local_solver_priority': bool(getattr(self, 'use_local_solver_priority', False)),
            'astap_path': str(getattr(self, 'astap_path', "")),
            'astap_data_dir': str(getattr(self, 'astap_data_dir', "")),
            'local_ansvr_path': str(getattr(self, 'local_ansvr_path', "")),
            
            'astap_search_radius': float(getattr(self, 'astap_search_radius', 5.0)), # Assurer un float, avec un défaut au cas où
        

        }
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                 json.dump(settings_data, f, indent=4, ensure_ascii=False)
        except TypeError as te: print(f"Error saving settings: Data not JSON serializable - {te}")
        except IOError as ioe: print(f"Error saving settings: I/O error writing to {self.settings_file} - {ioe}")
        except Exception as e: print(f"Unexpected error saving settings: {e}")

###################################################################################################################################




# --- DANS LA CLASSE SettingsManager DANS seestar/gui/settings.py ---

    def load_settings(self):
        """ 
        Charge les paramètres depuis le fichier JSON.
        Si le fichier n'existe pas ou est corrompu, les valeurs par défaut sont utilisées
        et un nouveau fichier de settings est créé.
        MAJ: Assure une gestion robuste des clés manquantes et des types.
        """
        print(f"DEBUG (SettingsManager load_settings): Tentative chargement depuis {self.settings_file}...")
        
        # Obtenir d'abord un dictionnaire de valeurs par défaut propres et complètes
        default_values_dict = self.get_default_values()
        
        if not os.path.exists(self.settings_file):
            print(f"DEBUG (SettingsManager load_settings): Fichier '{self.settings_file}' non trouvé. Application des valeurs par défaut normales.")
            # Appliquer tous les défauts à l'instance 'self'
            for key, value in default_values_dict.items():
                setattr(self, key, value)
            
            print(f"DEBUG (SettingsManager load_settings): Tentative de sauvegarde du fichier settings avec les valeurs par défaut normales...")
            self.save_settings() # Crée le fichier avec les valeurs par défaut actuelles de self
            # Logguer la valeur de la clé API après reset et avant de retourner
            print(f"DEBUG (SettingsManager load_settings): Valeur astrometry_api_key après reset (fichier non trouvé): '{getattr(self, 'astrometry_api_key', 'ERREUR_ATTR')}'")
            # La validation n'est pas strictement nécessaire ici car ce sont les défauts, mais ne nuit pas.
            _ = self.validate_settings() # Valider et ignorer les messages pour ce cas
            return False # Indiquer qu'on a utilisé les défauts car fichier absent
        
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                 settings_data = json.load(f)
        
            print("DEBUG (SettingsManager load_settings): Application des valeurs du JSON (avec fallback sur défauts)...")
            # Boucle sur TOUTES les clés attendues (celles de default_values_dict)
            for key, default_value_from_dict in default_values_dict.items():
                # Obtenir la valeur du JSON, si la clé n'y est pas, utiliser la valeur par défaut de notre dict
                loaded_value_from_json = settings_data.get(key, default_value_from_dict)
                
                final_value_to_set = loaded_value_from_json # Valeur par défaut si la clé n'est pas dans JSON

                if key not in settings_data:
                    print(f"  INFO (SettingsManager load_settings): Clé '{key}' non trouvée dans JSON. Utilisation de la valeur par défaut du code: {default_value_from_dict}")
                    final_value_to_set = default_value_from_dict
                else:
                    # La clé existe dans le JSON, on utilise loaded_value_from_json.
                    # Essayons maintenant de caster cette valeur vers le type de la valeur par défaut.
                    try:
                        type_of_default = type(default_value_from_dict)
                        if type_of_default == bool:
                            # Gérer les chaînes "true"/"false" du JSON pour les booléens
                            if isinstance(loaded_value_from_json, str):
                                if loaded_value_from_json.lower() == 'true': final_value_to_set = True
                                elif loaded_value_from_json.lower() == 'false': final_value_to_set = False
                                else: final_value_to_set = bool(loaded_value_from_json) # Tentative de cast direct
                            else:
                                final_value_to_set = bool(loaded_value_from_json)
                        elif type_of_default == int and not isinstance(loaded_value_from_json, bool): # Éviter True/False -> 1/0 si défaut est int
                            final_value_to_set = int(loaded_value_from_json)
                        elif type_of_default == float and not isinstance(loaded_value_from_json, bool):
                            final_value_to_set = float(loaded_value_from_json)
                        elif type_of_default == str:
                            final_value_to_set = str(loaded_value_from_json)
                        elif type_of_default == dict and isinstance(loaded_value_from_json, dict):
                            # Pour les dictionnaires (ex: mosaic_settings), fusionner prudemment
                            merged_dict = default_value_from_dict.copy() # Commencer avec les clés par défaut du code
                            merged_dict.update(loaded_value_from_json)   # Écraser/ajouter avec les clés du JSON
                            final_value_to_set = merged_dict
                        # Si le type par défaut est None, on accepte ce qui est chargé (pourrait être None ou autre chose)
                        elif default_value_from_dict is None:
                            pass # final_value_to_set est déjà correct (la valeur du JSON)
                        # Si le type n'est pas géré ci-dessus mais correspond, on le garde.
                        elif type(loaded_value_from_json) == type_of_default:
                            pass # Déjà du bon type
                        else: # Tentative de cast générique si les types diffèrent et non explicitement gérés
                            print(f"  WARN (SettingsManager load_settings): Tentative de cast générique pour la clé '{key}' du type {type(loaded_value_from_json)} vers {type_of_default}.")
                            final_value_to_set = type_of_default(loaded_value_from_json)

                    except (ValueError, TypeError) as e_cast:
                        print(f"  WARN (SettingsManager load_settings): Impossible de caster la valeur JSON '{loaded_value_from_json}' pour la clé '{key}' vers le type de '{default_value_from_dict}'. Erreur: {e_cast}. Utilisation de la valeur par défaut du code: {default_value_from_dict}")
                        final_value_to_set = default_value_from_dict # Revenir au défaut du code si le cast échoue
                
                setattr(self, key, final_value_to_set)
            # ---  Log spécifique radius ---
            print(f"  Valeur chargée pour astap_search_radius: {getattr(self, 'astap_search_radius', 'Non trouvé/Défaut')}")
            
            print(f"DEBUG (SettingsManager load_settings): Paramètres chargés et fusionnés depuis '{self.settings_file}'. Validation en cours...")
            # Exemple de log après chargement (ajoutez d'autres clés si besoin pour déboguer)
            print(f"  Exemple après chargement JSON - self.apply_low_wht_mask: {getattr(self, 'apply_low_wht_mask', 'NonTrouve')}, Pct: {getattr(self, 'low_wht_percentile', 'NonTrouve')}")

        except json.JSONDecodeError as e:
            print(f"Error decoding settings file {self.settings_file}: {e}. Using defaults from code and resetting file.")
            # Appliquer tous les défauts à l'instance 'self'
            for key, value in default_values_dict.items():
                setattr(self, key, value)
            self.save_settings()     # Écrase le fichier corrompu avec les valeurs par défaut actuelles
            _ = self.validate_settings()
            return False
        except Exception as e: 
            print(f"Error loading settings from {self.settings_file}: {e}. Using defaults from code and resetting file.")
            traceback.print_exc(limit=2)
            # Appliquer tous les défauts à l'instance 'self'
            for key, value in default_values_dict.items():
                setattr(self, key, value)
            self.save_settings()     # Écrase le fichier avec les valeurs par défaut actuelles
            _ = self.validate_settings()
            return False

        # La validation est TOUJOURS exécutée après le chargement (ou l'application des défauts)
        validation_messages = self.validate_settings() 
        if validation_messages:
             print("DEBUG (SettingsManager load_settings): Settings chargés/fusionnés ont été ajustés après validation:")
             for msg in validation_messages: print(f"  - {msg}")
             # Sauvegarder les settings corrigés pour qu'ils soient corrects au prochain lancement
             print("DEBUG (SettingsManager load_settings): Sauvegarde des settings validés (car des ajustements ont été faits).")
             self.save_settings() 
        
        print("DEBUG (SettingsManager load_settings): Fin de la méthode load_settings (mode lecture JSON).")
        return True




    #Fin settings.py