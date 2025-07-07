"""
Module pour la gestion des paramètres de traitement, de prévisualisation,
de pondération qualité et Drizzle.
"""

import json
import logging
import os
import tkinter as tk
import traceback

import numpy as np

logger = logging.getLogger(__name__)


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
        logger.debug(
            f"DEBUG (SettingsManager __init__): Initialisation avec settings_file='{self.settings_file}'"
        )  # DEBUG
        self.reset_to_defaults()  # Initialiser avec les valeurs par défaut

    def reset_to_defaults(self):
        """Réinitialise tous les paramètres de l'instance à leurs valeurs par défaut."""
        logger.debug(
            "DEBUG (SettingsManager reset_to_defaults): Début réinitialisation aux valeurs par défaut."
        )  # DEBUG
        defaults = self.get_default_values()
        for key, value in defaults.items():
            setattr(self, key, value)
        logger.debug(
            "DEBUG (SettingsManager reset_to_defaults): Tous les attributs de l'instance réinitialisés aux valeurs par défaut."
        )

    #####################################################################################################################################

    def update_from_ui(self, gui_instance):
        """
        Met à jour les paramètres de CETTE instance SettingsManager.
        MODIFIED: Ajout de la lecture pour save_final_as_float32 et ajustement mosaic_settings.
        """
        if (
            gui_instance is None
            or not hasattr(gui_instance, "root")
            or not gui_instance.root.winfo_exists()
        ):
            logger.debug(
                "Warning (SM update_from_ui): Cannot update settings from invalid GUI instance."
            )
            return
        logger.debug(
            "DEBUG SM (update_from_ui V_SaveAsFloat32_1): Lecture des paramètres..."
        )  # Version Log

        default_values_from_code = self.get_default_values()

        try:
            # --- Paramètres lus depuis les Tk Variables du GUI principal ---
            # ... (toutes les lectures existantes pour les autres paramètres restent ici, inchangées) ...
            self.input_folder = getattr(
                gui_instance,
                "input_path",
                tk.StringVar(value=default_values_from_code.get("input_folder", "")),
            ).get()
            self.output_folder = getattr(
                gui_instance,
                "output_path",
                tk.StringVar(value=default_values_from_code.get("output_folder", "")),
            ).get()
            self.output_filename = getattr(
                gui_instance,
                "output_filename_var",
                tk.StringVar(value=default_values_from_code.get("output_filename", "")),
            ).get()
            self.reference_image_path = getattr(
                gui_instance,
                "reference_image_path",
                tk.StringVar(
                    value=default_values_from_code.get("reference_image_path", "")
                ),
            ).get()
            self.last_stack_path = getattr(
                gui_instance,
                "last_stack_path",
                tk.StringVar(value=default_values_from_code.get("last_stack_path", "")),
            ).get()
            self.stacking_mode = getattr(
                gui_instance,
                "stacking_mode",
                tk.StringVar(
                    value=default_values_from_code.get("stacking_mode", "kappa-sigma")
                ),
            ).get()
            self.kappa = getattr(
                gui_instance,
                "kappa",
                tk.DoubleVar(value=default_values_from_code.get("kappa", 2.5)),
            ).get()
            self.stack_norm_method = getattr(
                gui_instance,
                "stack_norm_method_var",
                tk.StringVar(
                    value=default_values_from_code.get("stack_norm_method", "none")
                ),
            ).get()
            self.stack_weight_method = getattr(
                gui_instance,
                "stack_weight_method_var",
                tk.StringVar(
                    value=default_values_from_code.get("stack_weight_method", "none")
                ),
            ).get()
            self.stack_reject_algo = getattr(
                gui_instance,
                "stack_reject_algo_var",
                tk.StringVar(
                    value=default_values_from_code.get(
                        "stack_reject_algo", "kappa_sigma"
                    )
                ),
            ).get()
            self.stack_kappa_low = getattr(
                gui_instance,
                "stacking_kappa_low_var",
                tk.DoubleVar(
                    value=default_values_from_code.get("stack_kappa_low", 3.0)
                ),
            ).get()
            self.stack_kappa_high = getattr(
                gui_instance,
                "stacking_kappa_high_var",
                tk.DoubleVar(
                    value=default_values_from_code.get("stack_kappa_high", 3.0)
                ),
            ).get()
            self.stack_winsor_limits = getattr(
                gui_instance,
                "stacking_winsor_limits_str_var",
                tk.StringVar(
                    value=default_values_from_code.get(
                        "stack_winsor_limits", "0.05,0.05"
                    )
                ),
            ).get()
            self.stack_final_combine = getattr(
                gui_instance,
                "stack_final_combine_var",
                tk.StringVar(
                    value=default_values_from_code.get("stack_final_combine", "mean")
                ),
            ).get()
            self.stack_method = getattr(
                gui_instance,
                "stack_method_var",
                tk.StringVar(
                    value=default_values_from_code.get("stack_method", "kappa_sigma")
                ),
            ).get()
            if getattr(self, "stacking_mode", "") != "classic":
                self.stacking_mode = self.stack_method.replace("_", "-")
            self.batch_size = getattr(
                gui_instance,
                "batch_size",
                tk.IntVar(value=default_values_from_code.get("batch_size", 0)),
            ).get()
            self.max_hq_mem_gb = getattr(
                gui_instance,
                "max_hq_mem_var",
                tk.DoubleVar(value=default_values_from_code.get("max_hq_mem_gb", 8)),
            ).get()
            self.correct_hot_pixels = getattr(
                gui_instance,
                "correct_hot_pixels",
                tk.BooleanVar(
                    value=default_values_from_code.get("correct_hot_pixels", True)
                ),
            ).get()
            self.hot_pixel_threshold = getattr(
                gui_instance,
                "hot_pixel_threshold",
                tk.DoubleVar(
                    value=default_values_from_code.get("hot_pixel_threshold", 3.0)
                ),
            ).get()
            self.neighborhood_size = getattr(
                gui_instance,
                "neighborhood_size",
                tk.IntVar(value=default_values_from_code.get("neighborhood_size", 5)),
            ).get()
            self.cleanup_temp = getattr(
                gui_instance,
                "cleanup_temp_var",
                tk.BooleanVar(value=default_values_from_code.get("cleanup_temp", True)),
            ).get()
            self.zoom_percent = getattr(
                gui_instance,
                "zoom_percent_var",
                tk.IntVar(value=default_values_from_code.get("zoom_percent", 0)),
            ).get()
            self.bayer_pattern = getattr(
                gui_instance,
                "bayer_pattern_var",
                tk.StringVar(
                    value=default_values_from_code.get("bayer_pattern", "GRBG")
                ),
            ).get()
            self.use_quality_weighting = getattr(
                gui_instance,
                "use_weighting_var",
                tk.BooleanVar(
                    value=default_values_from_code.get("use_quality_weighting", True)
                ),
            ).get()
            self.weight_by_snr = getattr(
                gui_instance,
                "weight_snr_var",
                tk.BooleanVar(
                    value=default_values_from_code.get("weight_by_snr", True)
                ),
            ).get()
            self.weight_by_stars = getattr(
                gui_instance,
                "weight_stars_var",
                tk.BooleanVar(
                    value=default_values_from_code.get("weight_by_stars", True)
                ),
            ).get()
            self.snr_exponent = getattr(
                gui_instance,
                "snr_exponent_var",
                tk.DoubleVar(value=default_values_from_code.get("snr_exponent", 1.8)),
            ).get()
            self.stars_exponent = getattr(
                gui_instance,
                "stars_exponent_var",
                tk.DoubleVar(value=default_values_from_code.get("stars_exponent", 0.5)),
            ).get()
            self.min_weight = getattr(
                gui_instance,
                "min_weight_var",
                tk.DoubleVar(value=default_values_from_code.get("min_weight", 0.01)),
            ).get()
            self.use_drizzle = getattr(
                gui_instance,
                "use_drizzle_var",
                tk.BooleanVar(value=default_values_from_code.get("use_drizzle", False)),
            ).get()
            scale_str_ui = getattr(
                gui_instance,
                "drizzle_scale_var",
                tk.StringVar(
                    value=str(default_values_from_code.get("drizzle_scale", 2))
                ),
            ).get()
            try:
                self.drizzle_scale = int(float(scale_str_ui))
            except ValueError:
                self.drizzle_scale = default_values_from_code.get("drizzle_scale", 2)
            self.drizzle_wht_threshold = getattr(
                gui_instance,
                "drizzle_wht_threshold_var",
                tk.DoubleVar(
                    value=default_values_from_code.get("drizzle_wht_threshold", 0.7)
                ),
            ).get()
            self.drizzle_mode = getattr(
                gui_instance,
                "drizzle_mode_var",
                tk.StringVar(
                    value=default_values_from_code.get("drizzle_mode", "Final")
                ),
            ).get()
            self.drizzle_kernel = getattr(
                gui_instance,
                "drizzle_kernel_var",
                tk.StringVar(
                    value=default_values_from_code.get("drizzle_kernel", "square")
                ),
            ).get()
            self.drizzle_pixfrac = getattr(
                gui_instance,
                "drizzle_pixfrac_var",
                tk.DoubleVar(
                    value=default_values_from_code.get("drizzle_pixfrac", 1.0)
                ),
            ).get()
            self.astrometry_api_key = (
                getattr(
                    gui_instance,
                    "astrometry_api_key_var",
                    tk.StringVar(
                        value=default_values_from_code.get("astrometry_api_key", "")
                    ),
                )
                .get()
                .strip()
            )
            self.apply_chroma_correction = getattr(
                gui_instance,
                "apply_chroma_correction_var",
                tk.BooleanVar(
                    value=default_values_from_code.get("apply_chroma_correction", True)
                ),
            ).get()
            self.apply_final_scnr = getattr(
                gui_instance,
                "apply_final_scnr_var",
                tk.BooleanVar(
                    value=default_values_from_code.get("apply_final_scnr", True)
                ),
            ).get()
            self.final_scnr_target_channel = default_values_from_code.get(
                "final_scnr_target_channel", "green"
            )
            self.final_scnr_amount = getattr(
                gui_instance,
                "final_scnr_amount_var",
                tk.DoubleVar(
                    value=default_values_from_code.get("final_scnr_amount", 0.6)
                ),
            ).get()
            self.final_scnr_preserve_luminosity = getattr(
                gui_instance,
                "final_scnr_preserve_lum_var",
                tk.BooleanVar(
                    value=default_values_from_code.get(
                        "final_scnr_preserve_luminosity", True
                    )
                ),
            ).get()
            self.bn_grid_size_str = getattr(
                gui_instance,
                "bn_grid_size_str_var",
                tk.StringVar(
                    value=default_values_from_code.get("bn_grid_size_str", "24x24")
                ),
            ).get()
            self.bn_perc_low = getattr(
                gui_instance,
                "bn_perc_low_var",
                tk.IntVar(value=default_values_from_code.get("bn_perc_low", 5)),
            ).get()
            self.bn_perc_high = getattr(
                gui_instance,
                "bn_perc_high_var",
                tk.IntVar(value=default_values_from_code.get("bn_perc_high", 40)),
            ).get()
            self.bn_std_factor = getattr(
                gui_instance,
                "bn_std_factor_var",
                tk.DoubleVar(value=default_values_from_code.get("bn_std_factor", 1.5)),
            ).get()
            self.bn_min_gain = getattr(
                gui_instance,
                "bn_min_gain_var",
                tk.DoubleVar(value=default_values_from_code.get("bn_min_gain", 0.2)),
            ).get()
            self.bn_max_gain = getattr(
                gui_instance,
                "bn_max_gain_var",
                tk.DoubleVar(value=default_values_from_code.get("bn_max_gain", 7.0)),
            ).get()
            self.apply_bn = getattr(
                gui_instance,
                "apply_bn_var",
                tk.BooleanVar(value=default_values_from_code.get("apply_bn", True)),
            ).get()
            self.cb_border_size = getattr(
                gui_instance,
                "cb_border_size_var",
                tk.IntVar(value=default_values_from_code.get("cb_border_size", 25)),
            ).get()
            self.cb_blur_radius = getattr(
                gui_instance,
                "cb_blur_radius_var",
                tk.IntVar(value=default_values_from_code.get("cb_blur_radius", 8)),
            ).get()
            self.cb_min_b_factor = getattr(
                gui_instance,
                "cb_min_b_factor_var",
                tk.DoubleVar(
                    value=default_values_from_code.get("cb_min_b_factor", 0.4)
                ),
            ).get()
            self.cb_max_b_factor = getattr(
                gui_instance,
                "cb_max_b_factor_var",
                tk.DoubleVar(
                    value=default_values_from_code.get("cb_max_b_factor", 1.5)
                ),
            ).get()
            self.apply_cb = getattr(
                gui_instance,
                "apply_cb_var",
                tk.BooleanVar(value=default_values_from_code.get("apply_cb", True)),
            ).get()
            self.final_edge_crop_percent = getattr(
                gui_instance,
                "final_edge_crop_percent_var",
                tk.DoubleVar(
                    value=default_values_from_code.get("final_edge_crop_percent", 2.0)
                ),
            ).get()
            self.apply_master_tile_crop = getattr(
                gui_instance,
                "apply_master_tile_crop_var",
                tk.BooleanVar(
                    value=default_values_from_code.get("apply_master_tile_crop", False)
                ),
            ).get()
            self.master_tile_crop_percent = getattr(
                gui_instance,
                "master_tile_crop_percent_var",
                tk.DoubleVar(
                    value=default_values_from_code.get("master_tile_crop_percent", 18.0)
                ),
            ).get()
            self.apply_final_crop = getattr(
                gui_instance,
                "apply_final_crop_var",
                tk.BooleanVar(value=default_values_from_code.get("apply_final_crop", True)),
            ).get()
            self.apply_photutils_bn = getattr(
                gui_instance,
                "apply_photutils_bn_var",
                tk.BooleanVar(
                    value=default_values_from_code.get("apply_photutils_bn", False)
                ),
            ).get()
            self.photutils_bn_box_size = getattr(
                gui_instance,
                "photutils_bn_box_size_var",
                tk.IntVar(
                    value=default_values_from_code.get("photutils_bn_box_size", 128)
                ),
            ).get()
            self.photutils_bn_filter_size = getattr(
                gui_instance,
                "photutils_bn_filter_size_var",
                tk.IntVar(
                    value=default_values_from_code.get("photutils_bn_filter_size", 11)
                ),
            ).get()
            self.photutils_bn_sigma_clip = getattr(
                gui_instance,
                "photutils_bn_sigma_clip_var",
                tk.DoubleVar(
                    value=default_values_from_code.get("photutils_bn_sigma_clip", 3.0)
                ),
            ).get()
            self.photutils_bn_exclude_percentile = getattr(
                gui_instance,
                "photutils_bn_exclude_percentile_var",
                tk.DoubleVar(
                    value=default_values_from_code.get(
                        "photutils_bn_exclude_percentile", 95.0
                    )
                ),
            ).get()
            self.apply_feathering = getattr(
                gui_instance,
                "apply_feathering_var",
                tk.BooleanVar(
                    value=default_values_from_code.get("apply_feathering", True)
                ),
            ).get()
            self.apply_batch_feathering = getattr(
                gui_instance,
                "apply_batch_feathering_var",
                tk.BooleanVar(
                    value=default_values_from_code.get("apply_batch_feathering", True)
                ),
            ).get()
            self.feather_blur_px = getattr(
                gui_instance,
                "feather_blur_px_var",
                tk.IntVar(value=default_values_from_code.get("feather_blur_px", 256)),
            ).get()
            self.apply_low_wht_mask = getattr(
                gui_instance,
                "apply_low_wht_mask_var",
                tk.BooleanVar(
                    value=default_values_from_code.get("apply_low_wht_mask", False)
                ),
            ).get()
            self.low_wht_percentile = getattr(
                gui_instance,
                "low_wht_pct_var",
                tk.IntVar(value=default_values_from_code.get("low_wht_percentile", 5)),
            ).get()
            self.low_wht_soften_px = getattr(
                gui_instance,
                "low_wht_soften_px_var",
                tk.IntVar(value=default_values_from_code.get("low_wht_soften_px", 128)),
            ).get()
            self.preview_stretch_method = getattr(
                gui_instance,
                "preview_stretch_method",
                tk.StringVar(
                    value=default_values_from_code.get(
                        "preview_stretch_method", "Asinh"
                    )
                ),
            ).get()
            self.preview_black_point = getattr(
                gui_instance,
                "preview_black_point",
                tk.DoubleVar(
                    value=default_values_from_code.get("preview_black_point", 0.01)
                ),
            ).get()
            self.preview_white_point = getattr(
                gui_instance,
                "preview_white_point",
                tk.DoubleVar(
                    value=default_values_from_code.get("preview_white_point", 0.99)
                ),
            ).get()
            self.preview_gamma = getattr(
                gui_instance,
                "preview_gamma",
                tk.DoubleVar(value=default_values_from_code.get("preview_gamma", 1.0)),
            ).get()
            self.preview_r_gain = getattr(
                gui_instance,
                "preview_r_gain",
                tk.DoubleVar(value=default_values_from_code.get("preview_r_gain", 1.0)),
            ).get()
            self.preview_g_gain = getattr(
                gui_instance,
                "preview_g_gain",
                tk.DoubleVar(value=default_values_from_code.get("preview_g_gain", 1.0)),
            ).get()
            self.preview_b_gain = getattr(
                gui_instance,
                "preview_b_gain",
                tk.DoubleVar(value=default_values_from_code.get("preview_b_gain", 1.0)),
            ).get()
            self.language = getattr(
                gui_instance,
                "language_var",
                tk.StringVar(value=default_values_from_code.get("language", "en")),
            ).get()
            if gui_instance.root.winfo_exists():
                current_geo_ui = gui_instance.root.geometry()
                if (
                    isinstance(current_geo_ui, str)
                    and "x" in current_geo_ui
                    and "+" in current_geo_ui
                ):
                    self.window_geometry = current_geo_ui

            # --- NOUVEAU : Lecture du setting pour la sauvegarde en float32 ---
            self.save_final_as_float32 = getattr(
                gui_instance,
                "save_as_float32_var",
                tk.BooleanVar(
                    value=default_values_from_code.get("save_final_as_float32", False)
                ),
            ).get()
            logger.debug(
                f"DEBUG SM (update_from_ui): self.save_final_as_float32 lu (attribut UI ou défaut): {self.save_final_as_float32}"
            )
            # --- FIN NOUVEAU ---

            # --- NOUVEAU : Lecture du setting preserve_linear_output ---
            self.preserve_linear_output = getattr(
                gui_instance,
                "preserve_linear_output_var",
                tk.BooleanVar(
                    value=default_values_from_code.get("preserve_linear_output", False)
                ),
            ).get()
            logger.debug(
                f"DEBUG SM (update_from_ui): self.preserve_linear_output lu (attribut UI ou défaut): {self.preserve_linear_output}"
            )
            # --- FIN NOUVEAU ---

            # --- NOUVEAU : Lecture du paramètre d'utilisation du GPU ---
            self.use_gpu = getattr(
                gui_instance,
                "use_gpu_var",
                tk.BooleanVar(value=default_values_from_code.get("use_gpu", False)),
            ).get()
            logger.debug(
                f"DEBUG SM (update_from_ui): self.use_gpu lu (attribut UI ou défaut): {self.use_gpu}"
            )
            # --- FIN NOUVEAU ---

            # --- NOUVEAU : Lecture du setting d'utilisation des solveurs tiers ---
            self.use_third_party_solver = getattr(
                gui_instance,
                "use_third_party_solver_var",
                tk.BooleanVar(
                    value=default_values_from_code.get("use_third_party_solver", True)
                ),
            ).get()
            logger.debug(
                f"DEBUG SM (update_from_ui): self.use_third_party_solver lu (attribut UI ou défaut): {self.use_third_party_solver}"
            )
            # --- FIN NOUVEAU ---

            self.mosaic_mode_active = bool(
                getattr(
                    gui_instance,
                    "mosaic_mode_active",
                    default_values_from_code.get("mosaic_mode_active", False),
                )
            )
            logger.debug(
                f"DEBUG SM (update_from_ui): self.mosaic_mode_active (lu depuis gui_instance ou défaut): {self.mosaic_mode_active}"
            )

            # Gérer l'initialisation de self.mosaic_settings pour être sûr que c'est un dict
            if not isinstance(self.mosaic_settings, dict):
                self.mosaic_settings = default_values_from_code.get(
                    "mosaic_settings", {}
                ).copy()
                logger.debug(
                    f"DEBUG SM (update_from_ui): self.mosaic_settings réinitialisé aux défauts car non trouvé/invalide sur self."
                )
            else:
                # Si self.mosaic_settings est déjà un dict, mettre à jour les clés manquantes
                # depuis les défauts, notamment 'mosaic_scale_factor' si elle n'est pas présente.
                for key, value in default_values_from_code.get(
                    "mosaic_settings", {}
                ).items():
                    if key not in self.mosaic_settings:
                        self.mosaic_settings[key] = value
                logger.debug(
                    f"DEBUG SM (update_from_ui): self.mosaic_settings (après lecture/conservation de self, incluant new scale): {self.mosaic_settings}"
                )

            self.local_solver_preference = getattr(
                self,
                "local_solver_preference",
                default_values_from_code.get("local_solver_preference", "none"),
            )
            self.astap_path = getattr(
                self, "astap_path", default_values_from_code.get("astap_path", "")
            )
            self.astap_data_dir = getattr(
                self,
                "astap_data_dir",
                default_values_from_code.get("astap_data_dir", ""),
            )
            self.astap_search_radius = getattr(
                self,
                "astap_search_radius",
                default_values_from_code.get("astap_search_radius", 30.0),
            )
            self.local_ansvr_path = getattr(
                self,
                "local_ansvr_path",
                default_values_from_code.get("local_ansvr_path", ""),
            )
            self.ansvr_host_port = getattr(
                self,
                "ansvr_host_port",
                default_values_from_code.get("ansvr_host_port", "127.0.0.1:8080"),
            )

            self.astrometry_solve_field_dir = getattr(
                self,
                "astrometry_solve_field_dir",
                default_values_from_code.get("astrometry_solve_field_dir", ""),
            )

            self.reproject_between_batches = getattr(
                gui_instance,
                "reproject_between_batches_var",
                tk.BooleanVar(
                    value=default_values_from_code.get(
                        "reproject_between_batches", False
                    )
                ),
            ).get()

            self.reproject_coadd_final = getattr(
                gui_instance,
                "reproject_coadd_var",
                tk.BooleanVar(
                    value=default_values_from_code.get(
                        "reproject_coadd_final", False
                    )
                ),
            ).get()

            # In classic stacking mode this option defaults to disabled unless
            # the user explicitly checked the box in the Local Solver window.
            if self.stacking_mode == "classic":
                self.reproject_between_batches = bool(self.reproject_between_batches)

            self.use_radec_hints = getattr(
                gui_instance,
                "use_radec_hints_var",
                tk.BooleanVar(
                    value=default_values_from_code.get("use_radec_hints", False)
                ),
            ).get()

            logger.debug(
                f"DEBUG SM (update_from_ui V_SaveAsFloat32_1): Valeurs solveurs locaux (après lecture/conservation de self): "  # Version Log
                f"Pref='{self.local_solver_preference}', ASTAP Path='{self.astap_path}', ASTAP Radius={self.astap_search_radius}"
            )

            logger.debug(
                "DEBUG SM (update_from_ui V_SaveAsFloat32_1): Fin lecture des paramètres."
            )  # Version Log

        except AttributeError as ae:
            logger.debug(
                f"Error SM (update_from_ui V_SaveAsFloat32_1) (AttributeError): {ae}."
            )  # Version Log
            traceback.print_exc(limit=1)
        except tk.TclError as te:
            logger.debug(
                f"Error SM (update_from_ui V_SaveAsFloat32_1) (TclError - GUI détruite?): {te}."
            )  # Version Log
        except KeyError as ke:
            logger.debug(
                f"Error SM (update_from_ui V_SaveAsFloat32_1) (KeyError - probablement dans default_values_from_code.get()): {ke}."
            )  # Version Log
            traceback.print_exc(limit=1)
        except Exception as e:
            logger.debug(
                f"Unexpected error SM (update_from_ui V_SaveAsFloat32_1): {e}"
            )  # Version Log
            traceback.print_exc(limit=2)

    #######################################################################################################################################

    # --- DANS LA CLASSE SettingsManager DANS seestar/gui/settings.py ---

    def apply_to_ui(self, gui_instance):
        """
        Applique les paramètres chargés/actuels aux variables Tkinter.
        MODIFIED: Ajout de l'application pour save_final_as_float32.
        """
        if (
            gui_instance is None
            or not hasattr(gui_instance, "root")
            or not gui_instance.root.winfo_exists()
        ):
            logger.debug("Warning: Cannot apply settings to invalid GUI instance.")
            return
        try:
            logger.debug(
                "DEBUG (Settings apply_to_ui V_SaveAsFloat32_1): Application des paramètres à l'UI..."
            )  # Version Log

            # --- Processing Settings ---
            # ... (toutes les applications existantes pour les autres paramètres restent ici, inchangées) ...
            getattr(gui_instance, "input_path", tk.StringVar()).set(
                self.input_folder or ""
            )
            getattr(gui_instance, "output_path", tk.StringVar()).set(
                self.output_folder or ""
            )
            getattr(gui_instance, "output_filename_var", tk.StringVar()).set(
                self.output_filename or ""
            )
            getattr(gui_instance, "reference_image_path", tk.StringVar()).set(
                self.reference_image_path or ""
            )
            getattr(gui_instance, "last_stack_path", tk.StringVar()).set(
                self.last_stack_path or ""
            )
            getattr(gui_instance, "stacking_mode", tk.StringVar()).set(
                self.stacking_mode
            )
            getattr(gui_instance, "kappa", tk.DoubleVar()).set(self.kappa)
            getattr(gui_instance, "stack_norm_method_var", tk.StringVar()).set(
                self.stack_norm_method
            )
            getattr(gui_instance, "stack_weight_method_var", tk.StringVar()).set(
                self.stack_weight_method
            )
            getattr(gui_instance, "stack_reject_algo_var", tk.StringVar()).set(
                self.stack_reject_algo
            )
            getattr(gui_instance, "stacking_kappa_low_var", tk.DoubleVar()).set(
                self.stack_kappa_low
            )
            getattr(gui_instance, "stacking_kappa_high_var", tk.DoubleVar()).set(
                self.stack_kappa_high
            )
            getattr(gui_instance, "stacking_winsor_limits_str_var", tk.StringVar()).set(
                self.stack_winsor_limits
            )
            getattr(gui_instance, "stack_final_combine_var", tk.StringVar()).set(
                self.stack_final_combine
            )
            getattr(gui_instance, "max_hq_mem_var", tk.DoubleVar()).set(
                self.max_hq_mem_gb
            )
            getattr(gui_instance, "stack_method_var", tk.StringVar()).set(
                self.stack_method
            )
            getattr(gui_instance, "stacking_mode", tk.StringVar()).set(
                self.stack_method.replace("_", "-")
                if getattr(self, "stacking_mode", "") != "classic"
                else self.stacking_mode
            )
            getattr(gui_instance, "batch_size", tk.IntVar()).set(self.batch_size)
            getattr(gui_instance, "correct_hot_pixels", tk.BooleanVar()).set(
                self.correct_hot_pixels
            )
            getattr(gui_instance, "hot_pixel_threshold", tk.DoubleVar()).set(
                self.hot_pixel_threshold
            )
            getattr(gui_instance, "neighborhood_size", tk.IntVar()).set(
                self.neighborhood_size
            )
            getattr(gui_instance, "cleanup_temp_var", tk.BooleanVar()).set(
                self.cleanup_temp
            )
            getattr(gui_instance, "zoom_percent_var", tk.IntVar()).set(
                self.zoom_percent
            )

            if not hasattr(self, "local_solver_preference"):
                self.local_solver_preference = self.get_default_values()[
                    "local_solver_preference"
                ]
            if not hasattr(self, "astap_path"):
                self.astap_path = self.get_default_values()["astap_path"]
            if not hasattr(self, "astap_data_dir"):
                self.astap_data_dir = self.get_default_values()["astap_data_dir"]
            if not hasattr(self, "astap_search_radius"):
                self.astap_search_radius = self.get_default_values()[
                    "astap_search_radius"
                ]
            if not hasattr(self, "local_ansvr_path"):
                self.local_ansvr_path = self.get_default_values()["local_ansvr_path"]
            if not hasattr(self, "ansvr_host_port"):
                self.ansvr_host_port = self.get_default_values()["ansvr_host_port"]

            if not hasattr(self, "astrometry_solve_field_dir"):
                self.astrometry_solve_field_dir = self.get_default_values()[
                    "astrometry_solve_field_dir"
                ]

            if not hasattr(self, "reproject_between_batches"):
                self.reproject_between_batches = self.get_default_values()[
                    "reproject_between_batches"
                ]

            getattr(gui_instance, "use_weighting_var", tk.BooleanVar()).set(
                self.use_quality_weighting
            )
            getattr(gui_instance, "weight_snr_var", tk.BooleanVar()).set(
                self.weight_by_snr
            )
            getattr(gui_instance, "weight_stars_var", tk.BooleanVar()).set(
                self.weight_by_stars
            )
            getattr(gui_instance, "snr_exponent_var", tk.DoubleVar()).set(
                self.snr_exponent
            )
            getattr(gui_instance, "stars_exponent_var", tk.DoubleVar()).set(
                self.stars_exponent
            )
            getattr(gui_instance, "min_weight_var", tk.DoubleVar()).set(self.min_weight)

            getattr(gui_instance, "use_drizzle_var", tk.BooleanVar()).set(
                self.use_drizzle
            )
            getattr(gui_instance, "drizzle_scale_var", tk.StringVar()).set(
                str(self.drizzle_scale)
            )

            wht_value_01 = self.drizzle_wht_threshold
            getattr(gui_instance, "drizzle_wht_threshold_var", tk.DoubleVar()).set(
                wht_value_01
            )
            logger.debug(
                f"DEBUG (Settings apply_to_ui): WHT Threshold (0-1) appliqué: {wht_value_01}"
            )

            try:
                wht_value_percent = round(wht_value_01 * 100.0)
                wht_display_value = np.clip(wht_value_percent, 10, 100)
                wht_display_str = f"{wht_display_value:.0f}"
                getattr(gui_instance, "drizzle_wht_display_var", tk.StringVar()).set(
                    wht_display_str
                )
                logger.debug(
                    f"DEBUG (Settings apply_to_ui): WHT Display (%) appliqué: {wht_display_str}"
                )
            except Exception as e_conv:
                logger.debug(
                    f"ERREUR (Settings apply_to_ui): Échec conversion/application WHT display: {e_conv}"
                )
                getattr(gui_instance, "drizzle_wht_display_var", tk.StringVar()).set(
                    "70"
                )

            getattr(gui_instance, "drizzle_mode_var", tk.StringVar()).set(
                self.drizzle_mode
            )
            getattr(gui_instance, "drizzle_kernel_var", tk.StringVar()).set(
                self.drizzle_kernel
            )
            getattr(gui_instance, "drizzle_pixfrac_var", tk.DoubleVar()).set(
                self.drizzle_pixfrac
            )

            setattr(gui_instance, "mosaic_mode_active", bool(self.mosaic_mode_active))
            setattr(
                gui_instance,
                "mosaic_settings",
                (
                    self.mosaic_settings.copy()
                    if isinstance(self.mosaic_settings, dict)
                    else {}
                ),
            )
            if hasattr(gui_instance, "_update_mosaic_status_indicator"):
                gui_instance._update_mosaic_status_indicator()

            getattr(gui_instance, "apply_final_scnr_var", tk.BooleanVar()).set(
                self.apply_final_scnr
            )
            getattr(gui_instance, "final_scnr_amount_var", tk.DoubleVar()).set(
                self.final_scnr_amount
            )
            getattr(gui_instance, "final_scnr_preserve_lum_var", tk.BooleanVar()).set(
                self.final_scnr_preserve_luminosity
            )
            if hasattr(gui_instance, "_update_final_scnr_options_state"):
                gui_instance._update_final_scnr_options_state()
            logger.debug(
                f"DEBUG (Settings apply_to_ui): SCNR Final appliqué à UI -> Apply: {self.apply_final_scnr}, Amount: {self.final_scnr_amount:.2f}, PreserveLum: {self.final_scnr_preserve_luminosity}"
            )

            logger.debug(
                "DEBUG (Settings apply_to_ui): Application des paramètres Expert..."
            )
            getattr(gui_instance, "bn_grid_size_str_var", tk.StringVar()).set(
                self.bn_grid_size_str
            )
            getattr(gui_instance, "bn_perc_low_var", tk.IntVar()).set(self.bn_perc_low)
            getattr(gui_instance, "bn_perc_high_var", tk.IntVar()).set(
                self.bn_perc_high
            )
            getattr(gui_instance, "bn_std_factor_var", tk.DoubleVar()).set(
                self.bn_std_factor
            )
            getattr(gui_instance, "bn_min_gain_var", tk.DoubleVar()).set(
                self.bn_min_gain
            )
            getattr(gui_instance, "bn_max_gain_var", tk.DoubleVar()).set(
                self.bn_max_gain
            )
            getattr(gui_instance, "apply_bn_var", tk.BooleanVar()).set(
                self.apply_bn
            )
            getattr(gui_instance, "cb_border_size_var", tk.IntVar()).set(
                self.cb_border_size
            )
            getattr(gui_instance, "cb_blur_radius_var", tk.IntVar()).set(
                self.cb_blur_radius
            )
            getattr(gui_instance, "cb_min_b_factor_var", tk.DoubleVar()).set(
                self.cb_min_b_factor
            )
            getattr(gui_instance, "cb_max_b_factor_var", tk.DoubleVar()).set(
                self.cb_max_b_factor
            )
            getattr(gui_instance, "apply_cb_var", tk.BooleanVar()).set(
                self.apply_cb
            )
            getattr(gui_instance, "apply_master_tile_crop_var", tk.BooleanVar()).set(
                self.apply_master_tile_crop
            )
            getattr(gui_instance, "master_tile_crop_percent_var", tk.DoubleVar()).set(
                self.master_tile_crop_percent
            )
            getattr(gui_instance, "final_edge_crop_percent_var", tk.DoubleVar()).set(
                self.final_edge_crop_percent
            )
            getattr(gui_instance, "apply_final_crop_var", tk.BooleanVar()).set(
                self.apply_final_crop
            )

            logger.debug(
                "DEBUG (Settings apply_to_ui): Application des paramètres Photutils BN..."
            )
            getattr(gui_instance, "apply_photutils_bn_var", tk.BooleanVar()).set(
                self.apply_photutils_bn
            )
            getattr(gui_instance, "photutils_bn_box_size_var", tk.IntVar()).set(
                self.photutils_bn_box_size
            )
            getattr(gui_instance, "photutils_bn_filter_size_var", tk.IntVar()).set(
                self.photutils_bn_filter_size
            )
            getattr(gui_instance, "photutils_bn_sigma_clip_var", tk.DoubleVar()).set(
                self.photutils_bn_sigma_clip
            )
            getattr(
                gui_instance, "photutils_bn_exclude_percentile_var", tk.DoubleVar()
            ).set(self.photutils_bn_exclude_percentile)

            if hasattr(gui_instance, "_update_photutils_bn_options_state"):
                getattr(gui_instance, "astrometry_api_key_var", tk.StringVar()).set(
                    self.astrometry_api_key or ""
                )

            if hasattr(gui_instance, "apply_feathering_var"):
                getattr(gui_instance, "apply_feathering_var", tk.BooleanVar()).set(
                    self.apply_feathering
                )
                logger.debug(
                    f"DEBUG (Settings apply_to_ui): Apply Feathering appliqué à UI: {self.apply_feathering}"
                )
            if hasattr(gui_instance, "apply_batch_feathering_var"):
                getattr(gui_instance, "apply_batch_feathering_var", tk.BooleanVar()).set(
                    self.apply_batch_feathering
                )
            if hasattr(gui_instance, "feather_blur_px_var"):
                getattr(gui_instance, "feather_blur_px_var", tk.IntVar()).set(
                    self.feather_blur_px
                )
                logger.debug(
                    f"DEBUG (Settings apply_to_ui): Feather Blur Px appliqué à UI: {self.feather_blur_px}"
                )

            if hasattr(gui_instance, "apply_low_wht_mask_var"):
                getattr(gui_instance, "apply_low_wht_mask_var", tk.BooleanVar()).set(
                    self.apply_low_wht_mask
                )
            if hasattr(gui_instance, "low_wht_pct_var"):
                getattr(gui_instance, "low_wht_pct_var", tk.IntVar()).set(
                    self.low_wht_percentile
                )
            if hasattr(gui_instance, "low_wht_soften_px_var"):
                getattr(gui_instance, "low_wht_soften_px_var", tk.IntVar()).set(
                    self.low_wht_soften_px
                )
                logger.debug(
                    f"DEBUG (Settings apply_to_ui): LowWHT Mask appliqué à UI -> Apply: {self.apply_low_wht_mask}, Pct: {self.low_wht_percentile}, Soften: {self.low_wht_soften_px}"
                )

            if hasattr(gui_instance, "_update_feathering_options_state"):
                gui_instance._update_feathering_options_state()

            # --- NOUVEAU : Application du setting save_final_as_float32 à l'UI ---
            # Anticipe que gui_instance aura une variable Tkinter nommée 'save_as_float32_var'
            # Si elle n'est pas encore créée dans l'UI, getattr retournera une BooleanVar temporaire qui sera mise à jour,
            # mais cela n'affectera pas l'UI tant que le widget n'est pas créé et lié à cette variable.
            getattr(gui_instance, "save_as_float32_var", tk.BooleanVar()).set(
                self.save_final_as_float32
            )
            logger.debug(
                f"DEBUG (Settings apply_to_ui): save_final_as_float32 appliqué à l'UI (valeur: {self.save_final_as_float32})"
            )
            # --- FIN NOUVEAU ---

            # --- NOUVEAU : Application du setting preserve_linear_output ---
            getattr(gui_instance, "preserve_linear_output_var", tk.BooleanVar()).set(
                self.preserve_linear_output
            )
            logger.debug(
                f"DEBUG (Settings apply_to_ui): preserve_linear_output appliqué à l'UI (valeur: {self.preserve_linear_output})"
            )
            # --- FIN NOUVEAU ---

            # --- NOUVEAU : Application du paramètre use_gpu à l'UI ---
            getattr(gui_instance, "use_gpu_var", tk.BooleanVar()).set(
                self.use_gpu
            )
            logger.debug(
                f"DEBUG (Settings apply_to_ui): use_gpu appliqué à l'UI (valeur: {self.use_gpu})"
            )
            # --- FIN NOUVEAU ---

            # --- NOUVEAU : Application du toggle use_third_party_solver ---
            getattr(gui_instance, "use_third_party_solver_var", tk.BooleanVar()).set(
                self.use_third_party_solver
            )
            logger.debug(
                f"DEBUG (Settings apply_to_ui): use_third_party_solver appliqué à l'UI (valeur: {self.use_third_party_solver})"
            )
            # --- FIN NOUVEAU ---

            getattr(gui_instance, "preview_stretch_method", tk.StringVar()).set(
                self.preview_stretch_method
            )
            getattr(gui_instance, "preview_black_point", tk.DoubleVar()).set(
                self.preview_black_point
            )
            getattr(gui_instance, "preview_white_point", tk.DoubleVar()).set(
                self.preview_white_point
            )
            getattr(gui_instance, "preview_gamma", tk.DoubleVar()).set(
                self.preview_gamma
            )
            getattr(gui_instance, "preview_r_gain", tk.DoubleVar()).set(
                self.preview_r_gain
            )
            getattr(gui_instance, "preview_g_gain", tk.DoubleVar()).set(
                self.preview_g_gain
            )
            getattr(gui_instance, "preview_b_gain", tk.DoubleVar()).set(
                self.preview_b_gain
            )

            getattr(gui_instance, "language_var", tk.StringVar()).set(self.language)
            if (
                isinstance(self.window_geometry, str)
                and "x" in self.window_geometry
                and "+" in self.window_geometry
            ):
                try:
                    gui_instance.root.geometry(self.window_geometry)
                except tk.TclError:
                    logger.debug(
                        f"Warning: Could not apply window geometry '{self.window_geometry}'."
                    )

            if hasattr(gui_instance, "_update_weighting_options_state"):
                gui_instance._update_weighting_options_state()
            if hasattr(gui_instance, "_update_drizzle_options_state"):
                gui_instance._update_drizzle_options_state()
            if hasattr(gui_instance, "_update_final_scnr_options_state"):
                gui_instance._update_final_scnr_options_state()
            if hasattr(gui_instance, "_update_photutils_bn_options_state"):
                gui_instance._update_photutils_bn_options_state()
            if hasattr(gui_instance, "_update_feathering_options_state"):
                gui_instance._update_feathering_options_state()
            if hasattr(gui_instance, "_update_low_wht_mask_options_state"):
                gui_instance._update_low_wht_mask_options_state()
            if hasattr(gui_instance, "_update_bn_options_state"):
                gui_instance._update_bn_options_state()
            if hasattr(gui_instance, "_update_cb_options_state"):
                gui_instance._update_cb_options_state()
            if hasattr(gui_instance, "_update_crop_options_state"):
                gui_instance._update_crop_options_state()

            getattr(gui_instance, "astap_search_radius_var", tk.DoubleVar()).set(
                self.astap_search_radius
            )
            logger.debug(
                f"DEBUG (Settings apply_to_ui): astap_search_radius appliqué à l'UI (valeur: {self.astap_search_radius})"
            )
            getattr(gui_instance, "use_radec_hints_var", tk.BooleanVar()).set(
                self.use_radec_hints
            )
            getattr(gui_instance, "ansvr_host_port_var", tk.StringVar()).set(
                self.ansvr_host_port
            )

            getattr(gui_instance, "astrometry_solve_field_dir_var", tk.StringVar()).set(
                self.astrometry_solve_field_dir
            )

            getattr(gui_instance, "reproject_between_batches_var", tk.BooleanVar()).set(
                self.reproject_between_batches
            )

            getattr(gui_instance, "reproject_coadd_var", tk.BooleanVar()).set(
                self.reproject_coadd_final
            )

            logger.debug(
                "DEBUG (Settings apply_to_ui V_SaveAsFloat32_1): Fin application paramètres UI."
            )  # Version Log
            logger.debug(
                "DEBUG (SettingsManager apply_to_ui V_LocalSolverPref): Fin application paramètres UI principale."
            )  # Version Log (ancienne)

        except AttributeError as ae:
            logger.debug(f"Error applying settings to UI (AttributeError): {ae}")
        except tk.TclError as te:
            logger.debug(
                f"Error applying settings to UI (TclError - widget likely destroyed?): {te}"
            )
        except Exception as e:
            logger.debug(f"Unexpected error applying settings to UI: {e}")
            traceback.print_exc(limit=2)

    #################################################################################################################################

    # --- DANS LA CLASSE SettingsManager DANS seestar/gui/settings.py ---

    def get_default_values(self):
        """
        Retourne un dictionnaire des valeurs par défaut de tous les paramètres.
        MODIFIED: Ajout de save_final_as_float32.
        """
        logger.debug(
            "DEBUG (SettingsManager get_default_values V_SaveAsFloat32_1): Récupération des valeurs par défaut..."
        )  # Version Log
        defaults_dict = {}

        # --- Paramètres de Traitement de Base ---
        defaults_dict["input_folder"] = ""
        defaults_dict["output_folder"] = ""
        defaults_dict["output_filename"] = ""
        defaults_dict["reference_image_path"] = ""
        defaults_dict["last_stack_path"] = ""
        defaults_dict["bayer_pattern"] = "GRBG"
        defaults_dict["batch_size"] = 0
        defaults_dict["stacking_mode"] = "kappa-sigma"
        defaults_dict["kappa"] = 2.5
        defaults_dict["stack_norm_method"] = "none"
        defaults_dict["stack_weight_method"] = "none"
        defaults_dict["stack_reject_algo"] = "kappa_sigma"
        defaults_dict["stack_kappa_low"] = 3.0
        defaults_dict["stack_kappa_high"] = 3.0
        defaults_dict["stack_winsor_limits"] = "0.05,0.05"
        defaults_dict["stack_final_combine"] = "mean"
        defaults_dict["max_hq_mem_gb"] = 8
        defaults_dict["stack_method"] = "kappa_sigma"
        defaults_dict["correct_hot_pixels"] = True
        defaults_dict["hot_pixel_threshold"] = 3.0
        defaults_dict["neighborhood_size"] = 5
        defaults_dict["cleanup_temp"] = True
        defaults_dict["zoom_percent"] = 0

        # --- Paramètres de Pondération par Qualité ---
        defaults_dict["use_quality_weighting"] = True
        defaults_dict["weight_by_snr"] = True
        defaults_dict["weight_by_stars"] = True
        defaults_dict["snr_exponent"] = 1.8
        defaults_dict["stars_exponent"] = 0.5
        defaults_dict["min_weight"] = 0.01

        # --- Paramètres Drizzle (Globaux) ---
        defaults_dict["use_drizzle"] = False
        defaults_dict["drizzle_scale"] = 2
        defaults_dict["drizzle_wht_threshold"] = 0.7
        defaults_dict["drizzle_mode"] = "Final"
        defaults_dict["drizzle_kernel"] = "square"
        defaults_dict["drizzle_pixfrac"] = 1.0
        defaults_dict["drizzle_double_norm_fix"] = True

        # --- Paramètres de Correction Couleur et Post-Traitement ---
        defaults_dict["apply_chroma_correction"] = True
        defaults_dict["apply_final_scnr"] = True
        defaults_dict["final_scnr_target_channel"] = "green"
        defaults_dict["final_scnr_amount"] = 0.6
        defaults_dict["final_scnr_preserve_luminosity"] = True
        defaults_dict["bn_grid_size_str"] = "24x24"
        defaults_dict["bn_perc_low"] = 5
        defaults_dict["bn_perc_high"] = 40
        defaults_dict["bn_std_factor"] = 1.5
        defaults_dict["bn_min_gain"] = 0.2
        defaults_dict["bn_max_gain"] = 7.0
        defaults_dict["apply_bn"] = True
        defaults_dict["cb_border_size"] = 25
        defaults_dict["cb_blur_radius"] = 8
        defaults_dict["cb_min_b_factor"] = 0.4
        defaults_dict["cb_max_b_factor"] = 1.5
        defaults_dict["apply_cb"] = True
        defaults_dict["apply_master_tile_crop"] = False
        defaults_dict["master_tile_crop_percent"] = 18.0
        defaults_dict["final_edge_crop_percent"] = 2.0
        defaults_dict["apply_final_crop"] = True
        defaults_dict["apply_photutils_bn"] = False
        defaults_dict["photutils_bn_box_size"] = 128
        defaults_dict["photutils_bn_filter_size"] = 11
        defaults_dict["photutils_bn_sigma_clip"] = 3.0
        defaults_dict["photutils_bn_exclude_percentile"] = 95.0
        defaults_dict["apply_feathering"] = True
        defaults_dict["feather_blur_px"] = 256
        defaults_dict["apply_batch_feathering"] = True
        defaults_dict["apply_low_wht_mask"] = False
        defaults_dict["low_wht_percentile"] = 5
        defaults_dict["low_wht_soften_px"] = 128

        # --- NOUVEAU : Paramètre de sauvegarde float32 ---
        defaults_dict[
            "save_final_as_float32"
        ] = False  # Défaut à False (donc uint16 après mise à l'échelle par défaut)
        logger.debug(
            f"DEBUG (SettingsManager get_default_values): Ajout de 'save_final_as_float32'={defaults_dict['save_final_as_float32']}"
        )
        # --- FIN NOUVEAU ---

        # --- NOUVEAU : Préserver la sortie linéaire ---
        defaults_dict["preserve_linear_output"] = False
        logger.debug(
            f"DEBUG (SettingsManager get_default_values): Ajout de 'preserve_linear_output'={defaults_dict['preserve_linear_output']}"
        )
        # --- FIN NOUVEAU ---

        # --- NOUVEAU : Paramètre global d'utilisation du GPU ---
        defaults_dict["use_gpu"] = False
        logger.debug(
            f"DEBUG (SettingsManager get_default_values): Ajout de 'use_gpu'={defaults_dict['use_gpu']}"
        )
        # --- FIN NOUVEAU ---

        # --- Nouveau : activation/désactivation solveurs tiers ---
        defaults_dict["use_third_party_solver"] = True
        logger.debug(
            f"DEBUG (SettingsManager get_default_values): Ajout de 'use_third_party_solver'={defaults_dict['use_third_party_solver']}"
        )
        # --- FIN NOUVEAU ---

        # --- Paramètres Solveurs Locaux ---
        defaults_dict["local_solver_preference"] = "none"
        defaults_dict["astap_path"] = ""
        defaults_dict["astap_data_dir"] = ""
        defaults_dict["astap_search_radius"] = 3.0
        defaults_dict["astap_downsample"] = 1
        defaults_dict["astap_sensitivity"] = 100
        defaults_dict["use_radec_hints"] = False
        defaults_dict["local_ansvr_path"] = ""
        defaults_dict["ansvr_host_port"] = "127.0.0.1:8080"

        defaults_dict["astrometry_solve_field_dir"] = ""

        # When enabled, each batch is solved and reprojected incrementally onto
        # the reference WCS.
        defaults_dict["reproject_between_batches"] = False
        defaults_dict["reproject_coadd_final"] = False

        defaults_dict["mosaic_mode_active"] = False
        defaults_dict["mosaic_settings"] = {
            "kernel": "square",
            "pixfrac": 0.8,
            "use_gpu": False,
            "fillval": "0.0",
            "wht_threshold": 0.01,
            "alignment_mode": "local_fast_fallback",
            "fastalign_orb_features": 3000,
            "fastalign_min_abs_matches": 8,
            "fastalign_min_ransac": 4,
            "fastalign_ransac_thresh": 2.5,
            "fastalign_dao_fwhm": 3.5,
            "fastalign_dao_thr_sig": 8.0,
            "fastalign_dao_max_stars": 750,
            "mosaic_scale_factor": 1,  # <-- NOUVELLE LIGNE : Facteur d'échelle par défaut pour mosaïque (entier)
        }

        defaults_dict["astrometry_api_key"] = ""

        # --- Paramètres de Prévisualisation ---
        defaults_dict["preview_stretch_method"] = "Asinh"
        defaults_dict["preview_black_point"] = 0.01
        defaults_dict["preview_white_point"] = 0.99
        defaults_dict["preview_gamma"] = 1.0
        defaults_dict["preview_r_gain"] = 1.0
        defaults_dict["preview_g_gain"] = 1.0
        defaults_dict["preview_b_gain"] = 1.0

        # --- Paramètres de l'Interface Utilisateur ---
        defaults_dict["language"] = "en"
        defaults_dict["window_geometry"] = "1200x750"

        logger.debug(
            f"DEBUG (SettingsManager get_default_values V_SaveAsFloat32_1): Dictionnaire de défauts créé."
        )  # Version Log
        return defaults_dict

    ###################################################################################################################################

    def validate_settings(self):
        """
        Valide et corrige les paramètres si nécessaire. Retourne les messages de correction.
        MODIFIED: Ajout de la validation pour save_final_as_float32 et mosaic_scale_factor.
        """
        messages = []
        logger.debug(
            "DEBUG (Settings validate_settings V_SaveAsFloat32_1): DÉBUT de la validation."
        )  # Version Log

        defaults_fallback = self.get_default_values()
        logger.debug(
            f"DEBUG (Settings validate_settings): Valeur self.apply_photutils_bn AVANT TOUTE VALIDATION (lue de l'UI): {getattr(self, 'apply_photutils_bn', 'NON_DEFINI_ENCORE')}"
        )

        try:
            # --- Processing Settings Validation ---
            logger.debug("  -> Validating Processing Settings...")
            # ... (toutes les validations existantes pour kappa, batch_size, etc. restent ici) ...
            try:
                self.kappa = float(self.kappa)
                if not (1.0 <= self.kappa <= 5.0):
                    original = self.kappa
                    self.kappa = np.clip(self.kappa, 1.0, 5.0)
                    messages.append(f"Kappa ({original:.1f}) ajusté à {self.kappa:.1f}")
            except (ValueError, TypeError):
                original = self.kappa
                self.kappa = defaults_fallback["kappa"]
                messages.append(
                    f"Kappa ('{original}') invalide, réinitialisé à {self.kappa:.1f}"
                )

            try:
                self.batch_size = int(self.batch_size)
                if self.batch_size < 0:
                    original = self.batch_size
                    self.batch_size = 0
                    messages.append(
                        f"Taille Lot ({original}) ajusté à {self.batch_size} (auto)"
                    )
            except (ValueError, TypeError):
                original = self.batch_size
                self.batch_size = defaults_fallback["batch_size"]
                messages.append(
                    f"Taille Lot ('{original}') invalide, réinitialisé à {self.batch_size}"
                )

            try:
                self.hot_pixel_threshold = float(self.hot_pixel_threshold)
                if not (0.5 <= self.hot_pixel_threshold <= 10.0):
                    original = self.hot_pixel_threshold
                    self.hot_pixel_threshold = np.clip(
                        self.hot_pixel_threshold, 0.5, 10.0
                    )
                    messages.append(
                        f"Seuil Px Chauds ({original:.1f}) ajusté à {self.hot_pixel_threshold:.1f}"
                    )
            except (ValueError, TypeError):
                original = self.hot_pixel_threshold
                self.hot_pixel_threshold = defaults_fallback["hot_pixel_threshold"]
                messages.append(
                    f"Seuil Px Chauds ('{original}') invalide, réinitialisé à {self.hot_pixel_threshold:.1f}"
                )

            try:
                self.neighborhood_size = int(self.neighborhood_size)
                if self.neighborhood_size < 3:
                    original = self.neighborhood_size
                    self.neighborhood_size = 3
                    messages.append(
                        f"Voisinage Px Chauds ({original}) ajusté à {self.neighborhood_size}"
                    )
                if self.neighborhood_size % 2 == 0:
                    original = self.neighborhood_size
                    self.neighborhood_size += 1
                    messages.append(
                        f"Voisinage Px Chauds ({original}) ajusté à {self.neighborhood_size} (impair)"
                    )
            except (ValueError, TypeError):
                original = self.neighborhood_size
                self.neighborhood_size = defaults_fallback["neighborhood_size"]
                messages.append(
                    f"Voisinage Px Chauds ('{original}') invalide, réinitialisé à {self.neighborhood_size}"
                )

            # --- Validation des Options de Stacking avancées ---
            valid_methods = [
                "mean",
                "median",
                "kappa_sigma",
                "winsorized_sigma_clip",
                "linear_fit_clip",
            ]
            self.stack_method = str(
                getattr(self, "stack_method", defaults_fallback["stack_method"])
            )
            if self.stack_method not in valid_methods:
                messages.append(
                    f"Méthode de stack invalide ('{self.stack_method}'), réinitialisée à '{defaults_fallback['stack_method']}'"
                )
                self.stack_method = defaults_fallback["stack_method"]

            if self.stack_method == "mean":
                self.stack_final_combine = "mean"
                self.stack_reject_algo = "none"
            elif self.stack_method == "median":
                self.stack_final_combine = "median"
                self.stack_reject_algo = "none"
            elif self.stack_method == "kappa_sigma":
                self.stack_final_combine = "mean"
                self.stack_reject_algo = "kappa_sigma"
            elif self.stack_method == "winsorized_sigma_clip":
                self.stack_final_combine = "mean"
                self.stack_reject_algo = "winsorized_sigma_clip"
            elif self.stack_method == "linear_fit_clip":
                self.stack_final_combine = "mean"
                self.stack_reject_algo = "linear_fit_clip"

            valid_norm_methods = ["none", "linear_fit", "sky_mean"]
            self.stack_norm_method = str(
                getattr(
                    self, "stack_norm_method", defaults_fallback["stack_norm_method"]
                )
            )
            if self.stack_norm_method not in valid_norm_methods:
                messages.append(
                    f"Méthode de normalisation invalide ('{self.stack_norm_method}'), réinitialisée à '{defaults_fallback['stack_norm_method']}'"
                )
                self.stack_norm_method = defaults_fallback["stack_norm_method"]

            valid_weight_methods = ["none", "noise_variance", "noise_fwhm", "quality"]
            self.stack_weight_method = str(
                getattr(
                    self,
                    "stack_weight_method",
                    defaults_fallback["stack_weight_method"],
                )
            )
            if self.stack_weight_method not in valid_weight_methods:
                messages.append(
                    f"Méthode de pondération invalide ('{self.stack_weight_method}'), réinitialisée à '{defaults_fallback['stack_weight_method']}'"
                )
                self.stack_weight_method = defaults_fallback["stack_weight_method"]

            valid_reject_algos = [
                "none",
                "kappa_sigma",
                "winsorized_sigma_clip",
                "linear_fit_clip",
            ]
            self.stack_reject_algo = str(
                getattr(
                    self, "stack_reject_algo", defaults_fallback["stack_reject_algo"]
                )
            )
            if self.stack_reject_algo not in valid_reject_algos:
                messages.append(
                    f"Algorithme de rejet invalide ('{self.stack_reject_algo}'), réinitialisé à '{defaults_fallback['stack_reject_algo']}'"
                )
                self.stack_reject_algo = defaults_fallback["stack_reject_algo"]

            try:
                self.stack_kappa_low = float(self.stack_kappa_low)
                if not (0.1 <= self.stack_kappa_low <= 10.0):
                    original = self.stack_kappa_low
                    self.stack_kappa_low = np.clip(self.stack_kappa_low, 0.1, 10.0)
                    messages.append(
                        f"Kappa Low ({original}) hors limites [0.1, 10.0], ajusté à {self.stack_kappa_low}"
                    )
            except (ValueError, TypeError):
                original = self.stack_kappa_low
                self.stack_kappa_low = defaults_fallback["stack_kappa_low"]
                messages.append(
                    f"Kappa Low ('{original}') invalide, réinitialisé à {self.stack_kappa_low}"
                )

            try:
                self.stack_kappa_high = float(self.stack_kappa_high)
                if not (0.1 <= self.stack_kappa_high <= 10.0):
                    original = self.stack_kappa_high
                    self.stack_kappa_high = np.clip(self.stack_kappa_high, 0.1, 10.0)
                    messages.append(
                        f"Kappa High ({original}) hors limites [0.1, 10.0], ajusté à {self.stack_kappa_high}"
                    )
            except (ValueError, TypeError):
                original = self.stack_kappa_high
                self.stack_kappa_high = defaults_fallback["stack_kappa_high"]
                messages.append(
                    f"Kappa High ('{original}') invalide, réinitialisé à {self.stack_kappa_high}"
                )

            # --- HQ RAM limit ---
            try:
                self.max_hq_mem_gb = float(self.max_hq_mem_gb)
                if not (1 <= self.max_hq_mem_gb <= 64):
                    original = self.max_hq_mem_gb
                    self.max_hq_mem_gb = np.clip(self.max_hq_mem_gb, 1, 64)
                    messages.append(
                        f"Limite RAM HQ ({original}) hors plage [1,64] Go, réglée à {self.max_hq_mem_gb} Go"
                    )
            except (ValueError, TypeError):
                messages.append("Limite RAM HQ invalide – 8 Go utilisée")
                self.max_hq_mem_gb = defaults_fallback["max_hq_mem_gb"]

            self.max_hq_mem = int(self.max_hq_mem_gb * 1024**3)

            winsor_str = str(
                getattr(
                    self,
                    "stack_winsor_limits",
                    defaults_fallback["stack_winsor_limits"],
                )
            )
            parsed = None
            try:
                parts = [p.strip() for p in winsor_str.split(",")]
                if len(parts) != 2:
                    raise ValueError("format")
                low_val = float(parts[0])
                high_val = float(parts[1])
                if not (
                    0.0 <= low_val < 0.5
                    and 0.0 <= high_val < 0.5
                    and (low_val + high_val) < 1.0
                ):
                    raise ValueError("range")
                parsed = f"{low_val},{high_val}"
            except Exception:
                messages.append(
                    f"Limites Winsor invalides ('{winsor_str}'), réinitialisées à {defaults_fallback['stack_winsor_limits']}"
                )
                parsed = defaults_fallback["stack_winsor_limits"]
            self.stack_winsor_limits = parsed

            valid_combine = ["mean", "median", "winsorized_sigma_clip"]
            self.stack_final_combine = str(
                getattr(
                    self,
                    "stack_final_combine",
                    defaults_fallback["stack_final_combine"],
                )
            )
            if self.stack_final_combine not in valid_combine:
                messages.append(
                    f"Méthode de combinaison finale invalide ('{self.stack_final_combine}'), réinitialisée à '{defaults_fallback['stack_final_combine']}'"
                )
                self.stack_final_combine = defaults_fallback["stack_final_combine"]

            # --- Quality Weighting Validation ---
            # ... (inchangé) ...
            logger.debug("  -> Validating Quality Weighting Settings...")
            self.use_quality_weighting = bool(
                getattr(
                    self,
                    "use_quality_weighting",
                    defaults_fallback["use_quality_weighting"],
                )
            )
            self.weight_by_snr = bool(
                getattr(self, "weight_by_snr", defaults_fallback["weight_by_snr"])
            )
            self.weight_by_stars = bool(
                getattr(self, "weight_by_stars", defaults_fallback["weight_by_stars"])
            )
            try:
                self.snr_exponent = float(self.snr_exponent)
                if self.snr_exponent <= 0:
                    original = self.snr_exponent
                    self.snr_exponent = defaults_fallback["snr_exponent"]
                    messages.append(
                        f"Exposant SNR ({original:.1f}) ajusté à {self.snr_exponent:.1f}"
                    )
            except (ValueError, TypeError):
                original = self.snr_exponent
                self.snr_exponent = defaults_fallback["snr_exponent"]
                messages.append(
                    f"Exposant SNR ('{original}') invalide, réinitialisé à {self.snr_exponent:.1f}"
                )
            try:
                self.stars_exponent = float(self.stars_exponent)
                if self.stars_exponent <= 0:
                    original = self.stars_exponent
                    self.stars_exponent = defaults_fallback["stars_exponent"]
                    messages.append(
                        f"Exposant Étoiles ({original:.1f}) ajusté à {self.stars_exponent:.1f}"
                    )
            except (ValueError, TypeError):
                original = self.stars_exponent
                self.stars_exponent = defaults_fallback["stars_exponent"]
                messages.append(
                    f"Exposant Étoiles ('{original}') invalide, réinitialisé à {self.stars_exponent:.1f}"
                )
            try:
                self.min_weight = float(self.min_weight)
                if not (0 < self.min_weight <= 1.0):
                    original = self.min_weight
                    self.min_weight = np.clip(self.min_weight, 0.01, 1.0)
                    messages.append(
                        f"Poids Min ({original:.2f}) ajusté à {self.min_weight:.2f}"
                    )
            except (ValueError, TypeError):
                original = self.min_weight
                self.min_weight = defaults_fallback["min_weight"]
                messages.append(
                    f"Poids Min ('{original}') invalide, réinitialisé à {self.min_weight:.2f}"
                )

            if self.use_quality_weighting and not (
                self.weight_by_snr or self.weight_by_stars
            ):
                self.weight_by_snr = True
                messages.append(
                    "Pondération activée mais aucune métrique choisie. SNR activé par défaut."
                )

            # --- Drizzle Settings Validation ---
            # ... (inchangé) ...
            logger.debug("  -> Validating Drizzle Settings...")
            self.use_drizzle = bool(
                getattr(self, "use_drizzle", defaults_fallback["use_drizzle"])
            )
            try:
                scale_num = int(float(self.drizzle_scale))
                if scale_num not in [2, 3, 4]:
                    original = self.drizzle_scale
                    self.drizzle_scale = defaults_fallback["drizzle_scale"]
                    messages.append(
                        f"Échelle Drizzle ({original}) invalide, réinitialisée à {self.drizzle_scale}"
                    )
                else:
                    self.drizzle_scale = scale_num
            except (ValueError, TypeError):
                original = self.drizzle_scale
                self.drizzle_scale = defaults_fallback["drizzle_scale"]
                messages.append(
                    f"Échelle Drizzle invalide ('{original}'), réinitialisée à {self.drizzle_scale}"
                )
            try:
                self.drizzle_wht_threshold = float(self.drizzle_wht_threshold)
                if not (0.0 < self.drizzle_wht_threshold <= 1.0):
                    original = self.drizzle_wht_threshold
                    self.drizzle_wht_threshold = np.clip(
                        self.drizzle_wht_threshold, 0.1, 1.0
                    )
                    messages.append(
                        f"Seuil Drizzle WHT ({original:.2f}) hors limites [0.1, 1.0], ajusté à {self.drizzle_wht_threshold:.2f}"
                    )
            except (ValueError, TypeError):
                original = self.drizzle_wht_threshold
                self.drizzle_wht_threshold = defaults_fallback["drizzle_wht_threshold"]
                messages.append(
                    f"Seuil Drizzle WHT invalide ('{original}'), réinitialisé à {self.drizzle_wht_threshold:.2f}"
                )

            valid_drizzle_modes = ["Final", "Incremental"]
            current_driz_mode = getattr(
                self, "drizzle_mode", defaults_fallback["drizzle_mode"]
            )
            if (
                not isinstance(current_driz_mode, str)
                or current_driz_mode not in valid_drizzle_modes
            ):
                original = current_driz_mode
                self.drizzle_mode = defaults_fallback["drizzle_mode"]
                messages.append(
                    f"Mode Drizzle ({original}) invalide, réinitialisé à '{self.drizzle_mode}'"
                )
            else:
                self.drizzle_mode = current_driz_mode

            valid_kernels = [
                "square",
                "gaussian",
                "point",
                "tophat",
                "turbo",
                "lanczos2",
                "lanczos3",
            ]
            current_driz_kernel = getattr(
                self, "drizzle_kernel", defaults_fallback["drizzle_kernel"]
            )
            if (
                not isinstance(current_driz_kernel, str)
                or current_driz_kernel.lower() not in valid_kernels
            ):
                original = current_driz_kernel
                self.drizzle_kernel = defaults_fallback["drizzle_kernel"]
                messages.append(
                    f"Noyau Drizzle ('{original}') invalide, réinitialisé à '{self.drizzle_kernel}'"
                )
            else:
                self.drizzle_kernel = current_driz_kernel.lower()
            try:
                self.drizzle_pixfrac = float(self.drizzle_pixfrac)
                if not (
                    0.01 <= self.drizzle_pixfrac <= 2.0
                ):  # MODIFIÉ : Limite supérieure à 2.0
                    original = self.drizzle_pixfrac
                    self.drizzle_pixfrac = np.clip(
                        self.drizzle_pixfrac, 0.01, 2.0
                    )  # MODIFIÉ : Clip à 2.0
                    messages.append(
                        f"Pixfrac Drizzle ({original:.2f}) hors limites [0.01, 2.0], ajusté à {self.drizzle_pixfrac:.2f}"
                    )
            except (ValueError, TypeError):
                original = self.drizzle_pixfrac
                self.drizzle_pixfrac = defaults_fallback["drizzle_pixfrac"]
                messages.append(
                    f"Pixfrac Drizzle ('{original}') invalide, réinitialisé à {self.drizzle_pixfrac:.2f}"
                )



            # --- SCNR Final Validation ---
            # ... (inchangé) ...
            logger.debug("  -> Validating SCNR Settings...")
            self.apply_final_scnr = bool(
                getattr(self, "apply_final_scnr", defaults_fallback["apply_final_scnr"])
            )
            self.final_scnr_target_channel = str(
                getattr(
                    self,
                    "final_scnr_target_channel",
                    defaults_fallback["final_scnr_target_channel"],
                )
            ).lower()
            if self.final_scnr_target_channel not in ["green", "blue"]:
                original_target = self.final_scnr_target_channel
                self.final_scnr_target_channel = defaults_fallback[
                    "final_scnr_target_channel"
                ]
                messages.append(
                    f"Cible SCNR Final ('{original_target}') invalide, réinitialisée à '{self.final_scnr_target_channel}'."
                )
            try:
                self.final_scnr_amount = float(self.final_scnr_amount)
                if not (0.0 <= self.final_scnr_amount <= 1.0):
                    original_amount = self.final_scnr_amount
                    self.final_scnr_amount = np.clip(self.final_scnr_amount, 0.0, 1.0)
                    messages.append(
                        f"Intensité SCNR Final ({original_amount:.2f}) hors limites [0.0, 1.0], ajustée à {self.final_scnr_amount:.2f}."
                    )
            except (ValueError, TypeError):
                original_amount = self.final_scnr_amount
                self.final_scnr_amount = defaults_fallback["final_scnr_amount"]
                messages.append(
                    f"Intensité SCNR Final ('{original_amount}') invalide, réinitialisée à {self.final_scnr_amount:.2f}."
                )
            self.final_scnr_preserve_luminosity = bool(
                getattr(
                    self,
                    "final_scnr_preserve_luminosity",
                    defaults_fallback["final_scnr_preserve_luminosity"],
                )
            )

            # --- Expert Settings Validation ---
            # ... (inchangé) ...
            logger.debug("  -> Validating Expert Settings...")
            current_bn_grid = getattr(
                self, "bn_grid_size_str", defaults_fallback["bn_grid_size_str"]
            )
            if not isinstance(current_bn_grid, str) or current_bn_grid not in [
                "8x8",
                "16x16",
                "24x24",
                "32x32",
                "64x64",
            ]:
                messages.append(
                    f"Taille grille BN invalide ('{current_bn_grid}'), réinitialisée."
                )
                self.bn_grid_size_str = defaults_fallback["bn_grid_size_str"]
            else:
                self.bn_grid_size_str = current_bn_grid
            self.bn_perc_low = int(
                np.clip(
                    getattr(self, "bn_perc_low", defaults_fallback["bn_perc_low"]),
                    0,
                    40,
                )
            )
            self.bn_perc_high = int(
                np.clip(
                    getattr(self, "bn_perc_high", defaults_fallback["bn_perc_high"]),
                    self.bn_perc_low + 1,
                    90,
                )
            )
            self.bn_std_factor = float(
                np.clip(
                    getattr(self, "bn_std_factor", defaults_fallback["bn_std_factor"]),
                    0.1,
                    10.0,
                )
            )
            self.bn_min_gain = float(
                np.clip(
                    getattr(self, "bn_min_gain", defaults_fallback["bn_min_gain"]),
                    0.05,
                    5.0,
                )
            )
            self.bn_max_gain = float(
                np.clip(
                    getattr(self, "bn_max_gain", defaults_fallback["bn_max_gain"]),
                    self.bn_min_gain,
                    20.0,
                )
            )
            self.apply_bn = bool(
                getattr(self, "apply_bn", defaults_fallback["apply_bn"])
            )
            self.cb_border_size = int(
                np.clip(
                    getattr(
                        self, "cb_border_size", defaults_fallback["cb_border_size"]
                    ),
                    5,
                    200,
                )
            )
            self.cb_blur_radius = int(
                np.clip(
                    getattr(
                        self, "cb_blur_radius", defaults_fallback["cb_blur_radius"]
                    ),
                    0,
                    100,
                )
            )
            self.cb_min_b_factor = float(
                np.clip(
                    getattr(
                        self, "cb_min_b_factor", defaults_fallback["cb_min_b_factor"]
                    ),
                    0.1,
                    1.0,
                )
            )
            self.cb_max_b_factor = float(
                np.clip(
                    getattr(
                        self, "cb_max_b_factor", defaults_fallback["cb_max_b_factor"]
                    ),
                    self.cb_min_b_factor,
                    5.0,
                )
            )
            self.apply_cb = bool(
                getattr(self, "apply_cb", defaults_fallback["apply_cb"])
            )
            self.apply_master_tile_crop = bool(
                getattr(
                    self,
                    "apply_master_tile_crop",
                    defaults_fallback["apply_master_tile_crop"],
                )
            )
            self.master_tile_crop_percent = float(
                np.clip(
                    getattr(
                        self,
                        "master_tile_crop_percent",
                        defaults_fallback["master_tile_crop_percent"],
                    ),
                    0.0,
                    25.0,
                )
            )
            self.final_edge_crop_percent = float(
                np.clip(
                    getattr(
                        self,
                        "final_edge_crop_percent",
                        defaults_fallback["final_edge_crop_percent"],
                    ),
                    0.0,
                    25.0,
                )
            )
            self.apply_final_crop = bool(
                getattr(self, "apply_final_crop", defaults_fallback["apply_final_crop"])
            )
            logger.debug("    -> Validating Photutils BN...")
            self.apply_photutils_bn = bool(
                getattr(
                    self, "apply_photutils_bn", defaults_fallback["apply_photutils_bn"]
                )
            )
            self.photutils_bn_box_size = int(
                np.clip(
                    getattr(
                        self,
                        "photutils_bn_box_size",
                        defaults_fallback["photutils_bn_box_size"],
                    ),
                    8,
                    1024,
                )
            )
            self.photutils_bn_filter_size = int(
                np.clip(
                    getattr(
                        self,
                        "photutils_bn_filter_size",
                        defaults_fallback["photutils_bn_filter_size"],
                    ),
                    1,
                    25,
                )
            )
            if self.photutils_bn_filter_size % 2 == 0:
                self.photutils_bn_filter_size += 1
            self.photutils_bn_sigma_clip = float(
                np.clip(
                    getattr(
                        self,
                        "photutils_bn_sigma_clip",
                        defaults_fallback["photutils_bn_sigma_clip"],
                    ),
                    0.5,
                    10.0,
                )
            )
            self.photutils_bn_exclude_percentile = float(
                np.clip(
                    getattr(
                        self,
                        "photutils_bn_exclude_percentile",
                        defaults_fallback["photutils_bn_exclude_percentile"],
                    ),
                    0.0,
                    100.0,
                )
            )
            current_api_key = getattr(
                self, "astrometry_api_key", defaults_fallback["astrometry_api_key"]
            )
            if not isinstance(current_api_key, str):
                messages.append(
                    "Clé API Astrometry invalide (pas une chaîne), réinitialisée."
                )
                self.astrometry_api_key = defaults_fallback["astrometry_api_key"]
            else:
                self.astrometry_api_key = current_api_key.strip()
            self.output_filename = str(
                getattr(self, "output_filename", defaults_fallback["output_filename"])
            ).strip()
            logger.debug("    -> Validating Feathering...")
            self.apply_feathering = bool(
                getattr(self, "apply_feathering", defaults_fallback["apply_feathering"])
            )
            self.apply_batch_feathering = bool(
                getattr(
                    self,
                    "apply_batch_feathering",
                    defaults_fallback["apply_batch_feathering"],
                )
            )
            try:
                self.feather_blur_px = int(self.feather_blur_px)
                min_blur, max_blur = 32, 1024
                if not (min_blur <= self.feather_blur_px <= max_blur):
                    original_blur = self.feather_blur_px
                    self.feather_blur_px = int(
                        np.clip(self.feather_blur_px, min_blur, max_blur)
                    )
                    messages.append(
                        f"Feather Blur Px ({original_blur}) hors limites [{min_blur}-{max_blur}], ajusté à {self.feather_blur_px}."
                    )
            except (ValueError, TypeError):
                original_blur = self.feather_blur_px
                self.feather_blur_px = defaults_fallback["feather_blur_px"]
                messages.append(
                    f"Feather Blur Px ('{original_blur}') invalide, réinitialisé à {self.feather_blur_px}."
                )
            logger.debug("    -> Validating Low WHT Mask...")
            self.apply_low_wht_mask = bool(
                getattr(
                    self, "apply_low_wht_mask", defaults_fallback["apply_low_wht_mask"]
                )
            )
            try:
                self.low_wht_percentile = int(self.low_wht_percentile)
                min_pct, max_pct = 1, 100
                if not (min_pct <= self.low_wht_percentile <= max_pct):
                    original_pct = self.low_wht_percentile
                    self.low_wht_percentile = int(
                        np.clip(self.low_wht_percentile, min_pct, max_pct)
                    )
                    messages.append(
                        f"Low WHT Percentile ({original_pct}) hors limites [{min_pct}-{max_pct}], ajusté à {self.low_wht_percentile}."
                    )
            except (ValueError, TypeError):
                original_pct = self.low_wht_percentile
                self.low_wht_percentile = defaults_fallback["low_wht_percentile"]
                messages.append(
                    f"Low WHT Percentile ('{original_pct}') invalide, réinitialisé à {self.low_wht_percentile}."
                )
            try:
                self.low_wht_soften_px = int(self.low_wht_soften_px)
                min_soften, max_soften = 32, 512
                if not (min_soften <= self.low_wht_soften_px <= max_soften):
                    original_soften = self.low_wht_soften_px
                    self.low_wht_soften_px = int(
                        np.clip(self.low_wht_soften_px, min_soften, max_soften)
                    )
                    messages.append(
                        f"Low WHT Soften Px ({original_soften}) hors limites [{min_soften}-{max_soften}], ajusté à {self.low_wht_soften_px}."
                    )
            except (ValueError, TypeError):
                original_soften = self.low_wht_soften_px
                self.low_wht_soften_px = defaults_fallback["low_wht_soften_px"]
                messages.append(
                    f"Low WHT Soften Px ('{original_soften}') invalide, réinitialisé à {self.low_wht_soften_px}."
                )
            self.apply_chroma_correction = bool(
                getattr(
                    self,
                    "apply_chroma_correction",
                    defaults_fallback["apply_chroma_correction"],
                )
            )

            # --- NOUVEAU : Validation du setting save_final_as_float32 ---
            logger.debug("    -> Validating Save as float32...")
            # Assure que la valeur est un booléen. Si l'attribut n'existe pas ou n'est pas un booléen,
            # il prendra la valeur par défaut de defaults_fallback['save_final_as_float32'] (qui est False).
            current_save_float32_val = getattr(
                self,
                "save_final_as_float32",
                defaults_fallback["save_final_as_float32"],
            )
            if not isinstance(current_save_float32_val, bool):
                messages.append(
                    f"Option 'Sauvegarder en float32' ('{current_save_float32_val}') invalide, réinitialisée à {defaults_fallback['save_final_as_float32']}."
                )
                self.save_final_as_float32 = defaults_fallback["save_final_as_float32"]
            else:
                self.save_final_as_float32 = current_save_float32_val
            # --- FIN NOUVEAU ---

            # --- NOUVEAU : Validation du setting preserve_linear_output ---
            logger.debug("    -> Validating Preserve Linear Output...")
            current_preserve_val = getattr(
                self,
                "preserve_linear_output",
                defaults_fallback["preserve_linear_output"],
            )
            if not isinstance(current_preserve_val, bool):
                messages.append(
                    f"Option 'Preserve Linear Output' ('{current_preserve_val}') invalide, réinitialisée à {defaults_fallback['preserve_linear_output']}."
                )
                self.preserve_linear_output = defaults_fallback[
                    "preserve_linear_output"
                ]
            else:
                self.preserve_linear_output = current_preserve_val
            # --- FIN NOUVEAU ---

            # --- NOUVEAU : Validation du paramètre use_gpu ---
            logger.debug("    -> Validating use_gpu...")
            current_use_gpu = getattr(
                self,
                "use_gpu",
                defaults_fallback["use_gpu"],
            )
            if not isinstance(current_use_gpu, bool):
                messages.append(
                    f"Option 'Use GPU' ('{current_use_gpu}') invalide, réinitialisée à {defaults_fallback['use_gpu']}."
                )
                self.use_gpu = defaults_fallback["use_gpu"]
            else:
                self.use_gpu = current_use_gpu
            # --- FIN NOUVEAU ---

            # --- NOUVEAU : Validation du toggle use_third_party_solver ---
            logger.debug("    -> Validating use_third_party_solver...")
            current_use_solver_val = getattr(
                self,
                "use_third_party_solver",
                defaults_fallback["use_third_party_solver"],
            )
            if not isinstance(current_use_solver_val, bool):
                messages.append(
                    f"Option 'Use Third Party Solver' ('{current_use_solver_val}') invalide, réinitialisée à {defaults_fallback['use_third_party_solver']}."
                )
                self.use_third_party_solver = defaults_fallback[
                    "use_third_party_solver"
                ]
            else:
                self.use_third_party_solver = current_use_solver_val
            # --- FIN NOUVEAU ---

            logger.debug("    -> Validating reproject_coadd_final...")
            current_rc_val = getattr(
                self,
                "reproject_coadd_final",
                defaults_fallback["reproject_coadd_final"],
            )
            if not isinstance(current_rc_val, bool):
                messages.append(
                    f"Option 'Reproject Coadd Final' ('{current_rc_val}') invalide, réinitialisée à {defaults_fallback['reproject_coadd_final']}." 
                )
                self.reproject_coadd_final = defaults_fallback["reproject_coadd_final"]
            else:
                self.reproject_coadd_final = current_rc_val

            # --- Local Solver Paths and ASTAP Search Radius ---
            # ... (inchangé) ...
            logger.debug("  -> Validating Local Solver Settings...")
            self.astap_path = str(
                getattr(self, "astap_path", defaults_fallback["astap_path"])
            ).strip()
            self.astap_data_dir = str(
                getattr(self, "astap_data_dir", defaults_fallback["astap_data_dir"])
            ).strip()
            self.local_ansvr_path = str(
                getattr(self, "local_ansvr_path", defaults_fallback["local_ansvr_path"])
            ).strip()
            param_name_debug = "astap_search_radius"
            value_before_validation = getattr(
                self, param_name_debug, "ATTRIBUT_MANQUANT_SUR_SELF_POUR_VALIDATE"
            )
            logger.debug(
                f"    DEBUG VALIDATE: Valeur de self.{param_name_debug} AVANT float() et clip: '{value_before_validation}' (type: {type(value_before_validation)})"
            )
            try:
                current_radius_val = getattr(
                    self, param_name_debug, defaults_fallback[param_name_debug]
                )
                validated_radius = float(current_radius_val)
                min_r, max_r = 0.1, 90.0
                if not (min_r <= validated_radius <= max_r):
                    original_radius_str = f"{validated_radius:.1f}"
                    self.astap_search_radius = np.clip(validated_radius, min_r, max_r)
                    messages.append(
                        f"Rayon recherche ASTAP ({original_radius_str}°) hors limites [{min_r}-{max_r}], ajusté à {self.astap_search_radius:.1f}°"
                    )
                    logger.debug(
                        f"    DEBUG VALIDATE: Rayon clippé à {self.astap_search_radius:.1f}°"
                    )
                else:
                    self.astap_search_radius = validated_radius
                    logger.debug(
                        f"    DEBUG VALIDATE: Rayon déjà valide: {self.astap_search_radius:.1f}°"
                    )
            except (ValueError, TypeError) as e_val_rad:
                original_radius_str = str(
                    getattr(self, param_name_debug, "N/A_DANS_EXCEPT")
                )
                self.astap_search_radius = defaults_fallback[param_name_debug]
                messages.append(
                    f"Rayon recherche ASTAP ('{original_radius_str}') invalide (erreur: {e_val_rad}), réinitialisé à {self.astap_search_radius:.1f}°"
                )
                logger.debug(
                    f"    DEBUG VALIDATE: Exception lors de la validation du rayon ('{original_radius_str}'), réinitialisé à {self.astap_search_radius:.1f}°"
                )
            logger.debug(
                f"DEBUG (Settings validate_settings): {param_name_debug} FINAL après validation: {getattr(self, param_name_debug, 'ERREUR_ATTR_FINAL')}°"
            )
            valid_solver_prefs = ["none", "astap", "astrometry", "ansvr"]
            current_pref = getattr(
                self,
                "local_solver_preference",
                defaults_fallback["local_solver_preference"],
            )
            if (
                not isinstance(current_pref, str)
                or current_pref not in valid_solver_prefs
            ):
                messages.append(
                    f"Préférence solveur local ('{current_pref}') invalide, réinitialisée à '{defaults_fallback['local_solver_preference']}'."
                )
                self.local_solver_preference = defaults_fallback[
                    "local_solver_preference"
                ]
            else:
                self.local_solver_preference = current_pref
            current_astap_path = getattr(
                self, "astap_path", defaults_fallback["astap_path"]
            )
            if not isinstance(current_astap_path, str):
                messages.append("Chemin ASTAP invalide (pas une chaîne), réinitialisé.")
                self.astap_path = defaults_fallback["astap_path"]
            else:
                self.astap_path = current_astap_path.strip()
            current_astap_data_dir = getattr(
                self, "astap_data_dir", defaults_fallback["astap_data_dir"]
            )
            if not isinstance(current_astap_data_dir, str):
                messages.append(
                    "Chemin données ASTAP invalide (pas une chaîne), réinitialisé."
                )
                self.astap_data_dir = defaults_fallback["astap_data_dir"]
            else:
                self.astap_data_dir = current_astap_data_dir.strip()
            self.use_radec_hints = bool(getattr(self, "use_radec_hints", False))
            try:
                current_astap_radius = float(
                    getattr(
                        self,
                        "astap_search_radius",
                        defaults_fallback["astap_search_radius"],
                    )
                )
                if not (0.0 <= current_astap_radius <= 180.0):
                    original_radius = current_astap_radius
                    self.astap_search_radius = np.clip(current_astap_radius, 0.0, 180.0)
                    messages.append(
                        f"Rayon recherche ASTAP ({original_radius:.1f}°) hors limites [0, 180], ajusté à {self.astap_search_radius:.1f}°."
                    )
                else:
                    self.astap_search_radius = current_astap_radius
            except (ValueError, TypeError):
                original_radius_str = str(
                    getattr(
                        self,
                        "astap_search_radius",
                        defaults_fallback["astap_search_radius"],
                    )
                )
                self.astap_search_radius = defaults_fallback["astap_search_radius"]
                messages.append(
                    f"Rayon recherche ASTAP ('{original_radius_str}') invalide, réinitialisé à {self.astap_search_radius:.1f}°."
                )
            current_local_ansvr_path = getattr(
                self, "local_ansvr_path", defaults_fallback["local_ansvr_path"]
            )
            if not isinstance(current_local_ansvr_path, str):
                messages.append(
                    "Chemin Ansvr Local invalide (pas une chaîne), réinitialisé."
                )
                self.local_ansvr_path = defaults_fallback["local_ansvr_path"]
            else:
                self.local_ansvr_path = current_local_ansvr_path.strip()
            current_ansvr_host_port = getattr(
                self, "ansvr_host_port", defaults_fallback["ansvr_host_port"]
            )
            if not isinstance(current_ansvr_host_port, str):
                messages.append("Ansvr host/port invalide, réinitialisé.")
                self.ansvr_host_port = defaults_fallback["ansvr_host_port"]
            else:
                self.ansvr_host_port = current_ansvr_host_port.strip()

            current_astrometry_dir = getattr(
                self,
                "astrometry_solve_field_dir",
                defaults_fallback["astrometry_solve_field_dir"],
            )
            if not isinstance(current_astrometry_dir, str):
                messages.append("Chemin solve-field invalide, réinitialisé.")
                self.astrometry_solve_field_dir = defaults_fallback[
                    "astrometry_solve_field_dir"
                ]
            else:
                self.astrometry_solve_field_dir = current_astrometry_dir.strip()

            self.reproject_between_batches = bool(
                getattr(
                    self,
                    "reproject_between_batches",
                    defaults_fallback["reproject_between_batches"],
                )
            )

            logger.debug(
                f"DEBUG (SettingsManager validate_settings V_LocalSolverPref): Solveurs locaux validés: Pref='{self.local_solver_preference}', ASTAP Radius={self.astap_search_radius}"
            )

            # Validation du facteur d'échelle mosaïque
            # MODIFIÉ : Ce bloc de validation est maintenant inclus ici
            if not isinstance(self.mosaic_settings, dict):
                messages.append(
                    "Mosaic settings are invalid (not a dictionary), resetting to defaults."
                )
                self.mosaic_settings = defaults_fallback["mosaic_settings"].copy()

            try:
                scale_factor_m = int(
                    self.mosaic_settings.get(
                        "mosaic_scale_factor",
                        defaults_fallback["mosaic_settings"].get(
                            "mosaic_scale_factor", 2
                        ),
                    )
                )
                if not (1 <= scale_factor_m <= 4):
                    original_scale = scale_factor_m
                    scale_factor_m = np.clip(scale_factor_m, 1, 4)
                    messages.append(
                        f"Mosaic Scale Factor ({original_scale}) hors limites [1, 4], ajusté à {scale_factor_m}."
                    )
                self.mosaic_settings["mosaic_scale_factor"] = scale_factor_m
                logger.debug(
                    f"DEBUG SM (validate_settings): Mosaic Scale Factor validé à {self.mosaic_settings['mosaic_scale_factor']}."
                )
            except (ValueError, TypeError) as e_scale_val:
                original_scale = self.mosaic_settings.get("mosaic_scale_factor", "N/A")
                self.mosaic_settings["mosaic_scale_factor"] = defaults_fallback[
                    "mosaic_settings"
                ]["mosaic_scale_factor"]
                messages.append(
                    f"Mosaic Scale Factor ('{original_scale}') invalide ({e_scale_val}), réinitialisé à {self.mosaic_settings['mosaic_scale_factor']}."
                )
                logger.debug(
                    f"DEBUG SM (validate_settings): Mosaic Scale Factor invalide, réinitialisé. Erreur: {e_scale_val}"
                )
            # --- FIN DU BLOC DE VALIDATION DU FACTEUR D'ÉCHELLE MOSAÏQUE ---

        except Exception as e_global_val:
            messages.append(
                f"Erreur générale de validation: {e_global_val}. Réinitialisation aux valeurs par défaut."
            )
            logger.debug(
                f"FATAL Warning (Settings validate_settings): Erreur de validation globale -> {e_global_val}. Réinitialisation complète des settings."
            )
            self.reset_to_defaults()
            logger.debug(
                f"DEBUG (Settings validate_settings): self.apply_photutils_bn APRES reset_to_defaults (global catch): {getattr(self, 'apply_photutils_bn', 'ATTRIBUT_MANQUANT_APRES_RESET')}"
            )

        logger.debug(
            f"DEBUG (Settings validate_settings V_SaveAsFloat32_1): FIN de la validation. Nombre de messages: {len(messages)}. "  # Version Log
            f"Valeur finale de self.save_final_as_float32: {getattr(self, 'save_final_as_float32', 'NON_DEFINI_A_LA_FIN')}"
        )
        return messages

    ###################################################################################################################################

    # --- DANS LA CLASSE SettingsManager DANS seestar/gui/settings.py ---

    def save_settings(self):
        """
        Sauvegarde les paramètres actuels dans le fichier JSON.
        MODIFIED: Ajout de save_final_as_float32.
        """
        settings_data = {
            "version": "5.6.0",  # Version mise à jour pour refléter l'ajout
            # ... (tous les autres paramètres à sauvegarder restent ici, inchangés) ...
            "input_folder": str(self.input_folder),
            "output_folder": str(self.output_folder),
            "output_filename": str(self.output_filename),
            "reference_image_path": str(self.reference_image_path),
            "last_stack_path": str(self.last_stack_path),
            "bayer_pattern": str(self.bayer_pattern),
            "stacking_mode": str(self.stacking_mode),
            "kappa": float(self.kappa),
            "stack_norm_method": str(self.stack_norm_method),
            "stack_weight_method": str(self.stack_weight_method),
            "stack_reject_algo": str(self.stack_reject_algo),
            "stack_kappa_low": float(self.stack_kappa_low),
            "stack_kappa_high": float(self.stack_kappa_high),
            "stack_winsor_limits": str(self.stack_winsor_limits),
            "stack_final_combine": str(self.stack_final_combine),
            "max_hq_mem_gb": float(self.max_hq_mem_gb),
            "stack_method": str(self.stack_method),
            "batch_size": int(self.batch_size),
            "correct_hot_pixels": bool(self.correct_hot_pixels),
            "hot_pixel_threshold": float(self.hot_pixel_threshold),
            "neighborhood_size": int(self.neighborhood_size),
            "cleanup_temp": bool(self.cleanup_temp),
            "zoom_percent": int(self.zoom_percent),
            "use_quality_weighting": bool(self.use_quality_weighting),
            "weight_by_snr": bool(self.weight_by_snr),
            "weight_by_stars": bool(self.weight_by_stars),
            "snr_exponent": float(self.snr_exponent),
            "stars_exponent": float(self.stars_exponent),
            "min_weight": float(self.min_weight),
            "use_drizzle": bool(self.use_drizzle),
            "drizzle_scale": int(self.drizzle_scale),
            "drizzle_wht_threshold": float(self.drizzle_wht_threshold),
            "drizzle_mode": str(self.drizzle_mode),
            "drizzle_kernel": str(self.drizzle_kernel),
            "drizzle_pixfrac": float(self.drizzle_pixfrac),
            "drizzle_double_norm_fix": bool(
                getattr(self, "drizzle_double_norm_fix", True)
            ),
            "mosaic_mode_active": bool(self.mosaic_mode_active),
            "mosaic_settings": (
                self.mosaic_settings if isinstance(self.mosaic_settings, dict) else {}
            ),
            "apply_chroma_correction": bool(self.apply_chroma_correction),
            "apply_final_scnr": bool(self.apply_final_scnr),
            "final_scnr_target_channel": str(self.final_scnr_target_channel),
            "final_scnr_amount": float(self.final_scnr_amount),
            "final_scnr_preserve_luminosity": bool(self.final_scnr_preserve_luminosity),
            "bn_grid_size_str": str(self.bn_grid_size_str),
            "bn_perc_low": int(self.bn_perc_low),
            "bn_perc_high": int(self.bn_perc_high),
            "bn_std_factor": float(self.bn_std_factor),
            "bn_min_gain": float(self.bn_min_gain),
            "bn_max_gain": float(self.bn_max_gain),
            "apply_bn": bool(self.apply_bn),
            "cb_border_size": int(self.cb_border_size),
            "cb_blur_radius": int(self.cb_blur_radius),
            "cb_min_b_factor": float(self.cb_min_b_factor),
            "cb_max_b_factor": float(self.cb_max_b_factor),
            "apply_cb": bool(self.apply_cb),
            "apply_master_tile_crop": bool(self.apply_master_tile_crop),
            "master_tile_crop_percent": float(self.master_tile_crop_percent),
            "final_edge_crop_percent": float(self.final_edge_crop_percent),
            "apply_final_crop": bool(self.apply_final_crop),
            "apply_photutils_bn": bool(self.apply_photutils_bn),
            "photutils_bn_box_size": int(self.photutils_bn_box_size),
            "photutils_bn_filter_size": int(self.photutils_bn_filter_size),
            "photutils_bn_sigma_clip": float(self.photutils_bn_sigma_clip),
            "photutils_bn_exclude_percentile": float(
                self.photutils_bn_exclude_percentile
            ),
            "astrometry_api_key": str(self.astrometry_api_key),
            "preview_stretch_method": str(self.preview_stretch_method),
            "preview_black_point": float(self.preview_black_point),
            "preview_white_point": float(self.preview_white_point),
            "preview_gamma": float(self.preview_gamma),
            "preview_r_gain": float(self.preview_r_gain),
            "preview_g_gain": float(self.preview_g_gain),
            "preview_b_gain": float(self.preview_b_gain),
            "language": str(self.language),
            "window_geometry": str(self.window_geometry),
            "apply_feathering": bool(self.apply_feathering),
            "feather_blur_px": int(self.feather_blur_px),
            "apply_batch_feathering": bool(self.apply_batch_feathering),
            "apply_low_wht_mask": bool(self.apply_low_wht_mask),
            "low_wht_percentile": int(self.low_wht_percentile),
            "low_wht_soften_px": int(self.low_wht_soften_px),
            # --- NOUVEAU : Sauvegarde du setting save_final_as_float32 ---
            "save_final_as_float32": bool(
                getattr(self, "save_final_as_float32", False)
            ),
            # --- FIN NOUVEAU ---
            # --- NOUVEAU : Sauvegarde du paramètre use_gpu ---
            "use_gpu": bool(getattr(self, "use_gpu", False)),
            # --- FIN NOUVEAU ---
            # --- NOUVEAU : Sauvegarde du setting preserve_linear_output ---
            "preserve_linear_output": bool(
                getattr(self, "preserve_linear_output", False)
            ),
            # --- FIN NOUVEAU ---
            # --- NOUVEAU : Sauvegarde du toggle use_third_party_solver ---
            "use_third_party_solver": bool(
                getattr(self, "use_third_party_solver", True)
            ),
            # --- FIN NOUVEAU ---
            "local_solver_preference": str(
                getattr(self, "local_solver_preference", "none")
            ),
            "astap_path": str(getattr(self, "astap_path", "")),
            "astap_data_dir": str(getattr(self, "astap_data_dir", "")),
            "astap_search_radius": float(
                getattr(self, "astap_search_radius", 30.0)
            ),  # Maintenu comme avant
            "use_radec_hints": bool(getattr(self, "use_radec_hints", False)),
            "local_ansvr_path": str(getattr(self, "local_ansvr_path", "")),
            "ansvr_host_port": str(getattr(self, "ansvr_host_port", "127.0.0.1:8080")),
            "astrometry_solve_field_dir": str(
                getattr(self, "astrometry_solve_field_dir", "")
            ),
            "reproject_between_batches": bool(
                getattr(self, "reproject_between_batches", False)
            ),
            "reproject_coadd_final": bool(
                getattr(self, "reproject_coadd_final", False)
            ),
        }

        if (
            "use_local_solver_priority" in settings_data
        ):  # Nettoyage de l'ancienne clé si elle existait par erreur
            del settings_data["use_local_solver_priority"]
            logger.debug(
                "DEBUG (SettingsManager save_settings): Ancienne clé 'use_local_solver_priority' supprimée des données de sauvegarde."
            )

        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(settings_data, f, indent=4, ensure_ascii=False)
            logger.debug(
                f"DEBUG (SettingsManager save_settings V_SaveAsFloat32_1): Paramètres sauvegardés dans '{self.settings_file}'."
            )  # Version Log

        except TypeError as te:
            logger.debug(f"Error saving settings: Data not JSON serializable - {te}")
        except IOError as ioe:
            logger.debug(
                f"Error saving settings: I/O error writing to {self.settings_file} - {ioe}"
            )
        except Exception as e:
            logger.debug(f"Unexpected error saving settings: {e}")

    def export_run_settings(self, file_path: str):
        """Enregistre les paramètres actuels dans un fichier spécifique."""
        original = self.settings_file
        try:
            self.settings_file = file_path
            self.save_settings()
        finally:
            self.settings_file = original

    ###################################################################################################################################

    # --- DANS LA CLASSE SettingsManager DANS seestar/gui/settings.py ---

    def load_settings(self):
        """
        Charge les paramètres depuis le fichier JSON.
        MODIFIED: Ajout de logs de debug spécifiques pour save_final_as_float32.
        La logique de chargement générique devrait déjà le gérer.
        """
        logger.debug(
            f"DEBUG (SettingsManager load_settings V_SaveAsFloat32_1): Tentative chargement depuis {self.settings_file}..."
        )  # Version Log

        default_values_dict = self.get_default_values()

        if not os.path.exists(self.settings_file):
            logger.debug(
                f"DEBUG (SettingsManager load_settings V_SaveAsFloat32_1): Fichier '{self.settings_file}' non trouvé. Application des valeurs par défaut."
            )
            for key, value in default_values_dict.items():
                setattr(self, key, value)

            logger.debug(
                f"DEBUG (SettingsManager load_settings V_SaveAsFloat32_1): Tentative de sauvegarde du fichier settings avec les valeurs par défaut."
            )
            self.save_settings()
            _ = self.validate_settings()
            logger.debug(
                f"DEBUG (SettingsManager load_settings V_SaveAsFloat32_1): Valeur save_final_as_float32 après reset (fichier non trouvé): '{getattr(self, 'save_final_as_float32', 'ERREUR_ATTR')}'"
            )
            return False

        settings_data = {}
        try:
            with open(self.settings_file, "r", encoding="utf-8") as f:
                settings_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.debug(
                f"ERREUR (SettingsManager load_settings V_SaveAsFloat32_1): Décodage JSON échoué pour {self.settings_file}: {e}. Utilisation des défauts et réinitialisation du fichier."
            )
            for key, value in default_values_dict.items():
                setattr(self, key, value)
            self.save_settings()
            _ = self.validate_settings()
            return False
        except Exception as e_open:
            logger.debug(
                f"ERREUR (SettingsManager load_settings V_SaveAsFloat32_1): Lecture de {self.settings_file} échouée: {e_open}. Utilisation des défauts et réinitialisation du fichier."
            )
            traceback.print_exc(limit=2)
            for key, value in default_values_dict.items():
                setattr(self, key, value)
            self.save_settings()
            _ = self.validate_settings()
            return False

        logger.debug(
            "DEBUG (SettingsManager load_settings V_SaveAsFloat32_1): Application des valeurs du JSON (avec fallback sur défauts)..."
        )

        old_use_local_priority_val = settings_data.pop(
            "use_local_solver_priority", None
        )

        old_reproj_val = settings_data.pop("enable_interbatch_reproj", None)
        if (
            old_reproj_val is not None
            and "reproject_between_batches" not in settings_data
        ):
            logger.debug(
                f"  INFO (SettingsManager load_settings): Ancienne clé 'enable_interbatch_reproj' ({old_reproj_val}) convertie vers 'reproject_between_batches'."
            )
            settings_data["reproject_between_batches"] = old_reproj_val

        if old_use_local_priority_val is not None:
            if "local_solver_preference" not in settings_data:
                logger.debug(
                    f"  INFO (SettingsManager load_settings): Ancienne clé 'use_local_solver_priority' ({old_use_local_priority_val}) trouvée. Conversion vers 'local_solver_preference'."
                )
                if bool(old_use_local_priority_val):
                    settings_data["local_solver_preference"] = "astap"
                    logger.debug(
                        f"     -> Converti en 'local_solver_preference': 'astap'"
                    )
                else:
                    settings_data["local_solver_preference"] = "none"
                    logger.debug(
                        f"     -> Converti en 'local_solver_preference': 'none'"
                    )
            else:
                logger.debug(
                    f"  INFO (SettingsManager load_settings): Ancienne clé 'use_local_solver_priority' ({old_use_local_priority_val}) trouvée mais ignorée car 'local_solver_preference' ('{settings_data['local_solver_preference']}') est déjà présente."
                )

        # Boucle sur TOUTES les clés attendues (celles de default_values_dict) pour peupler `self`
        for key, default_value_from_code in default_values_dict.items():
            loaded_value_from_json = settings_data.get(key, default_value_from_code)
            final_value_to_set = loaded_value_from_json

            if (
                key not in settings_data
                and loaded_value_from_json is default_value_from_code
            ):
                if key == "save_final_as_float32":  # DEBUG SPÉCIFIQUE
                    logger.debug(
                        f"  INFO LOAD (save_final_as_float32): Clé '{key}' non trouvée dans JSON. Utilisation défaut du code: {default_value_from_code}"
                    )
            else:
                try:
                    type_of_default = type(default_value_from_code)

                    if type(loaded_value_from_json) == type_of_default:
                        final_value_to_set = loaded_value_from_json
                    elif type_of_default == bool:
                        if isinstance(loaded_value_from_json, str):
                            if loaded_value_from_json.lower() == "true":
                                final_value_to_set = True
                            elif loaded_value_from_json.lower() == "false":
                                final_value_to_set = False
                            else:
                                final_value_to_set = bool(int(loaded_value_from_json))
                        else:
                            final_value_to_set = bool(loaded_value_from_json)
                    elif type_of_default == int and not isinstance(
                        loaded_value_from_json, bool
                    ):
                        final_value_to_set = int(float(loaded_value_from_json))
                    elif type_of_default == float and not isinstance(
                        loaded_value_from_json, bool
                    ):
                        final_value_to_set = float(loaded_value_from_json)
                    elif type_of_default == str:
                        final_value_to_set = str(loaded_value_from_json)
                    elif type_of_default == dict and isinstance(
                        loaded_value_from_json, dict
                    ):
                        merged_dict = default_value_from_code.copy()
                        merged_dict.update(loaded_value_from_json)
                        final_value_to_set = merged_dict
                    elif default_value_from_code is None:
                        final_value_to_set = loaded_value_from_json
                    else:
                        logger.debug(
                            f"  WARN (SettingsManager load_settings): Tentative de cast générique pour la clé '{key}' du type {type(loaded_value_from_json)} vers {type_of_default}."
                        )
                        final_value_to_set = type_of_default(loaded_value_from_json)

                except (ValueError, TypeError) as e_cast:
                    log_prefix = (
                        f"  WARN LOAD ({key})"
                        if key == "save_final_as_float32"
                        else "  WARN (SettingsManager load_settings)"
                    )
                    logger.debug(
                        f"{log_prefix}: Impossible de caster la valeur JSON '{loaded_value_from_json}' (type {type(loaded_value_from_json)}) pour la clé '{key}' vers le type attendu ({type(default_value_from_code)}). Erreur: {e_cast}. Utilisation de la valeur par défaut du code: {default_value_from_code}"
                    )
                    final_value_to_set = default_value_from_code

            setattr(self, key, final_value_to_set)

        logger.debug(
            f"DEBUG (SettingsManager load_settings V_SaveAsFloat32_1): Paramètres chargés et fusionnés depuis '{self.settings_file}'. Validation en cours..."
        )
        logger.debug(
            f"  Exemple après chargement JSON - self.save_final_as_float32: '{getattr(self, 'save_final_as_float32', 'NonTrouve')}'"
        )

        validation_messages = self.validate_settings()
        if validation_messages:
            logger.debug(
                "DEBUG (SettingsManager load_settings V_SaveAsFloat32_1): Settings chargés/fusionnés ont été ajustés après validation:"
            )
            for msg in validation_messages:
                logger.debug(f"  - {msg}")
            logger.debug(
                "DEBUG (SettingsManager load_settings V_SaveAsFloat32_1): Sauvegarde des settings validés (car des ajustements ont été faits ou pour nettoyer ancienne clé)."
            )
            self.save_settings()

        logger.debug(
            "DEBUG (SettingsManager load_settings V_SaveAsFloat32_1): Fin de la méthode load_settings."
        )
        return True

    # Fin settings.py

    # Fin settings.py
