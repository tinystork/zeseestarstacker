"""Qt-local FR/EN localization helper (pure stdlib, M9).

The Qt shell renders its own, deliberately small set of visible strings.  The
full Tk ``Localization`` dictionaries carry a large number of Tk-specific keys
and are not a 1:1 match for the Qt shell surface, so this module keeps a
compact, self-contained mapping for exactly the strings the Qt shell renders.

Design constraints honoured here:

* **No Qt, no Tk, no engine imports.**  This module is pure stdlib so it can be
  imported and unit-tested in complete isolation, and so that
  ``import seestar.gui_qt`` never pulls in the Tk GUI or the scientific engine.
* **Explicit fallback, never a crash.**  A missing key returns the requested
  default or the key itself; a missing translation for the requested language
  falls back to English; an unsupported language code degrades to English.
* **Key parity by construction.**  Every registered key carries both ``en`` and
  ``fr`` values, which the test suite verifies so the visible surface can never
  silently regress to a one-sided mapping.
"""

from __future__ import annotations

from typing import Dict, Optional

DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = ("en", "fr")

# Combo display label -> language code (order matters: English first).
LANGUAGE_LABELS: Dict[str, str] = {"en": "English", "fr": "Français"}
LANGUAGE_CODE_BY_TEXT: Dict[str, str] = {
    label: code for code, label in LANGUAGE_LABELS.items()
}

# key -> {en: str, fr: str} for every visible Qt shell string.  Keys are
# snake_case identifiers; values are the exact text rendered by the shell.
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # Left-panel chrome.
    "language_label": {"en": "Language:", "fr": "Langue :"},
    "tab_stacking": {"en": "Stacking", "fr": "Empilement"},
    "tab_expert": {"en": "Expert", "fr": "Expert"},
    "tab_preview_controls": {"en": "Preview controls", "fr": "Contrôles d'aperçu"},
    "progress_label": {"en": "Progression:", "fr": "Progression :"},
    "log_label": {"en": "Log:", "fr": "Journal :"},
    "copy_log": {"en": "Copy Log", "fr": "Copier le journal"},
    "elapsed": {"en": "Elapsed:", "fr": "Écoulé :"},
    "remaining": {"en": "Remaining:", "fr": "Restant :"},
    # Stacking tab.
    "input_folder": {"en": "Input folder", "fr": "Dossier d'entrée"},
    "output_folder": {"en": "Output folder", "fr": "Dossier de sortie"},
    "temp_folder": {"en": "Temp folder", "fr": "Dossier temporaire"},
    "output_filename": {"en": "Output filename", "fr": "Nom du fichier de sortie"},
    "reference_image": {"en": "Reference image", "fr": "Image de référence"},
    "last_stack": {"en": "Last stack", "fr": "Dernier stack"},
    "batch_size": {"en": "Batch size", "fr": "Taille du lot"},
    "stacking_mode": {"en": "Stacking mode", "fr": "Mode d'empilement"},
    "final_combine": {"en": "Final combine", "fr": "Combinaison finale"},
    "drizzle_mode": {"en": "Drizzle mode", "fr": "Mode drizzle"},
    "drizzle_group_size": {"en": "Drizzle group size", "fr": "Taille du groupe drizzle"},
    "local_solver": {"en": "Local solver", "fr": "Solveur local"},
    "browse": {"en": "Browse...", "fr": "Parcourir..."},
    "boring_check": {"en": "Threaded Boring Stack", "fr": "Empilement Boring en thread"},
    "drizzle_check": {"en": "Enable drizzle", "fr": "Activer le drizzle"},
    # Right panel.
    "preview_group": {"en": "Preview", "fr": "Aperçu"},
    "view_group": {"en": "View", "fr": "Vue"},
    "zoom_label": {"en": "Zoom:", "fr": "Zoom :"},
    "resolution_label": {"en": "Resolution:", "fr": "Résolution :"},
    "histogram_group": {"en": "Histogram", "fr": "Histogramme"},
    "actions_group": {"en": "Actions", "fr": "Actions"},
    "preview_prefix": {"en": "Preview:", "fr": "Aperçu :"},
    # Action buttons.
    "start": {"en": "Start", "fr": "Démarrer"},
    "stop": {"en": "Stop", "fr": "Arrêter"},
    "analyse": {"en": "Analyse", "fr": "Analyser"},
    "solver": {"en": "Solver", "fr": "Solveur"},
    "view_inputs": {"en": "View Inputs", "fr": "Voir les entrées"},
    "add_folder": {"en": "Add Folder", "fr": "Ajouter un dossier"},
    "open_output": {"en": "Open Output", "fr": "Ouvrir la sortie"},
    # Preview controls tab (M10): display-only WB / stretch / histogram.
    "wb_group": {"en": "White balance", "fr": "Balance des blancs"},
    "wb_red": {"en": "Red", "fr": "Rouge"},
    "wb_green": {"en": "Green", "fr": "Vert"},
    "wb_blue": {"en": "Blue", "fr": "Bleu"},
    "wb_reset": {"en": "Reset", "fr": "Réinitialiser"},
    "auto_wb": {"en": "Auto WB", "fr": "Balance auto"},
    "stretch_group": {"en": "Stretch", "fr": "Étirement"},
    "stretch_label": {"en": "Stretch:", "fr": "Étirement :"},
    "stretch_black": {"en": "Black point", "fr": "Point noir"},
    "stretch_white": {"en": "White point", "fr": "Point blanc"},
    "stretch_gamma": {"en": "Gamma", "fr": "Gamma"},
    "stretch_reset": {"en": "Reset Stretch", "fr": "Réinitialiser l'étirement"},
    "auto_stretch": {"en": "Auto Stretch", "fr": "Étirement auto"},
    "bcs_group": {"en": "Image Adjustments", "fr": "Ajustements d'image"},
    "brightness": {"en": "Brightness", "fr": "Luminosité"},
    "contrast": {"en": "Contrast", "fr": "Contraste"},
    "saturation": {"en": "Saturation", "fr": "Saturation"},
    "bcs_reset": {"en": "Reset Adjust.", "fr": "Réinitialiser"},
    "histogram_empty": {"en": "No preview", "fr": "Aucun aperçu"},
    "histogram_stats": {"en": "Stats:", "fr": "Stats :"},
    "histo_auto_zoom": {"en": "Auto zoom histogram", "fr": "Zoom auto histogramme"},
    "histo_reset": {"en": "Reset Histogram", "fr": "Réinitialiser l'histogramme"},
    "histo_zoom": {"en": "Zoom Histogram", "fr": "Zoom histogramme"},
    # Initial preview auto-load states (M12): auto-load first FITS.
    "preview_no_input_folder": {
        "en": "Input folder not found or not set",
        "fr": "Dossier d'entrée introuvable ou non défini",
    },
    "preview_no_fits": {
        "en": "No FITS files in input folder",
        "fr": "Aucun fichier FITS dans le dossier d'entrée",
    },
    "preview_loading": {
        "en": "Loading preview...",
        "fr": "Chargement de l'aperçu...",
    },
    "preview_loaded": {
        "en": "Preview loaded",
        "fr": "Aperçu chargé",
    },
    "preview_error": {
        "en": "Error loading preview...",
        "fr": "Erreur lors du chargement de l'aperçu...",
    },
    # Expert/Settings section titles.
    "section_stacking_paths": {"en": "Stacking / Paths", "fr": "Empilement / Chemins"},
    "section_calibration": {
        "en": "Calibration / Hot Pixels",
        "fr": "Calibration / Pixels chauds",
    },
    "section_quality_weighting": {
        "en": "Quality Weighting",
        "fr": "Pondération par qualité",
    },
    "section_drizzle_advanced": {"en": "Drizzle Advanced", "fr": "Drizzle avancé"},
    "section_colour_post": {
        "en": "Colour / Post-processing",
        "fr": "Couleur / Post-traitement",
    },
    "section_cropping": {"en": "Cropping", "fr": "Rognage"},
    "section_photutils_bn": {"en": "Photutils BN", "fr": "Photutils BN"},
    "section_feathering": {
        "en": "Feathering / Low-weight Mask",
        "fr": "Feathering / Masque bas poids",
    },
    "section_mosaic": {"en": "Mosaic", "fr": "Mosaïque"},
    "section_solver": {"en": "Solver", "fr": "Solveur"},
    "section_output_reprojection": {
        "en": "Output / Reprojection",
        "fr": "Sortie / Reprojection",
    },
    "section_final_bg_matching": {
        "en": "Final Background Matching",
        "fr": "Correspondance du fond final",
    },
    # Mosaic controls (checkbox + nested sub-fields).
    "mosaic_mode_active": {"en": "Mosaic mode active", "fr": "Mode mosaïque actif"},
    "mosaic_kernel": {"en": "Kernel", "fr": "Noyau"},
    "mosaic_pixfrac": {"en": "Pixfrac", "fr": "Pixfrac"},
    "mosaic_use_gpu": {"en": "Use GPU", "fr": "Utiliser le GPU"},
    "mosaic_fillval": {"en": "Fill value", "fr": "Valeur de remplissage"},
    "mosaic_wht_threshold": {"en": "WHT threshold", "fr": "Seuil WHT"},
    "mosaic_alignment_mode": {"en": "Alignment mode", "fr": "Mode d'alignement"},
    "mosaic_fastalign_orb_features": {"en": "ORB features", "fr": "Points ORB"},
    "mosaic_fastalign_min_abs_matches": {
        "en": "Min abs matches",
        "fr": "Corresp. abs. min",
    },
    "mosaic_fastalign_min_ransac": {
        "en": "Min RANSAC",
        "fr": "Inliers RANSAC min",
    },
    "mosaic_fastalign_ransac_thresh": {
        "en": "RANSAC threshold",
        "fr": "Seuil RANSAC",
    },
    "mosaic_fastalign_dao_fwhm": {"en": "DAO FWHM", "fr": "DAO FWHM"},
    "mosaic_fastalign_dao_thr_sig": {
        "en": "DAO threshold sigma",
        "fr": "DAO seuil sigma",
    },
    "mosaic_fastalign_dao_max_stars": {"en": "DAO max stars", "fr": "DAO étoiles max"},
    "mosaic_scale_factor": {"en": "Scale factor", "fr": "Facteur d'échelle"},
    # Representative Settings field labels (the remainder stay English until a
    # fuller mapping lands; see docs checklist item 8.2).
    "field_kappa": {"en": "Kappa", "fr": "Kappa"},
    "field_normalize_method": {
        "en": "Normalize method",
        "fr": "Méthode de normalisation",
    },
    "field_weighting_method": {
        "en": "Weighting method",
        "fr": "Méthode de pondération",
    },
    "field_correct_hot_pixels": {
        "en": "Correct hot pixels",
        "fr": "Corriger les pixels chauds",
    },
    "field_bayer_pattern": {"en": "Bayer pattern", "fr": "Matrice de Bayer"},
    "field_cleanup_temp": {
        "en": "Clean up temp files",
        "fr": "Nettoyer les fichiers temporaires",
    },
    "field_weight_by_snr": {"en": "Weight by SNR", "fr": "Pondérer par SNR"},
    "field_weight_by_stars": {"en": "Weight by stars", "fr": "Pondérer par étoiles"},
    "field_drizzle_kernel": {"en": "Kernel", "fr": "Noyau"},
    "field_master_tile_crop": {
        "en": "Master tile crop",
        "fr": "Rognage des tuiles maîtresses",
    },
    "field_save_as_float32": {
        "en": "Save final as float32",
        "fr": "Sauvegarder en float32",
    },
    "field_preserve_linear_output": {
        "en": "Preserve linear output",
        "fr": "Préserver la sortie linéaire",
    },
    "field_match_bg": {
        "en": "Match background for final",
        "fr": "Correspondance fond final",
    },
    # Expert tab chrome (M15): warning banner + reset-to-defaults button.
    "expert_warning_text": {
        "en": "Expert Settings!",
        "fr": "Réglages Expert !",
    },
    "reset_expert_button": {
        "en": "Reset Expert Settings",
        "fr": "Réinitialiser les réglages Expert",
    },
    # Expert tab field labels (M15) — the full BN / CB / cropping / Photutils /
    # feathering / low-weight surface, so every Expert-tab control localizes.
    "field_apply_bn": {"en": "Enable BN", "fr": "Activer BN"},
    "field_bn_grid_size": {"en": "BN grid size", "fr": "Taille de grille BN"},
    "field_bn_perc_low": {
        "en": "BN percentile low",
        "fr": "Percentile bas BN",
    },
    "field_bn_perc_high": {
        "en": "BN percentile high",
        "fr": "Percentile haut BN",
    },
    "field_bn_std_factor": {
        "en": "BN std factor",
        "fr": "Facteur écart-type BN",
    },
    "field_bn_min_gain": {"en": "BN min gain", "fr": "Gain min BN"},
    "field_bn_max_gain": {"en": "BN max gain", "fr": "Gain max BN"},
    "field_apply_cb": {
        "en": "Enable Edge/Chroma Correction",
        "fr": "Activer la correction bord/chroma",
    },
    "field_cb_border_size": {
        "en": "CB border size",
        "fr": "Taille de bordure CB",
    },
    "field_cb_blur_radius": {
        "en": "CB blur radius",
        "fr": "Rayon de flou CB",
    },
    "field_cb_min_b_factor": {
        "en": "CB min B factor",
        "fr": "Facteur B min CB",
    },
    "field_cb_max_b_factor": {
        "en": "CB max B factor",
        "fr": "Facteur B max CB",
    },
    "field_apply_final_crop": {
        "en": "Enable Final Cropping",
        "fr": "Activer le rognage final",
    },
    "field_final_edge_crop_percent": {
        "en": "Final edge crop %",
        "fr": "Rognage des bords final %",
    },
    "field_master_tile_crop_percent": {
        "en": "Master tile crop %",
        "fr": "Rognage tuiles maîtresses %",
    },
    "field_apply_photutils_bn": {
        "en": "Photutils background normalization",
        "fr": "Normalisation du fond Photutils",
    },
    "field_photutils_bn_box_size": {
        "en": "Box size",
        "fr": "Taille de boîte",
    },
    "field_photutils_bn_filter_size": {
        "en": "Filter size",
        "fr": "Taille du filtre",
    },
    "field_photutils_bn_sigma_clip": {
        "en": "Sigma clip",
        "fr": "Sigma clip",
    },
    "field_photutils_bn_exclude_percentile": {
        "en": "Exclude percentile",
        "fr": "Percentile exclu",
    },
    "field_apply_feathering": {"en": "Feathering", "fr": "Feathering"},
    "field_feather_blur_px": {
        "en": "Feather blur (px)",
        "fr": "Flou feathering (px)",
    },
    "field_apply_batch_feathering": {
        "en": "Batch feathering",
        "fr": "Feathering inter-lots",
    },
    "field_apply_low_wht_mask": {
        "en": "Low-weight mask",
        "fr": "Masque bas poids",
    },
    "field_low_wht_percentile": {
        "en": "Low-weight percentile",
        "fr": "Percentile bas poids",
    },
    "field_low_wht_soften_px": {
        "en": "Low-weight soften (px)",
        "fr": "Adoucissement bas poids (px)",
    },
}


def normalize_language(code: Optional[str]) -> str:
    """Return ``code`` when supported, otherwise the default language (``en``).

    Never raises: ``None``, an unknown code, or a non-string value all degrade
    to English so a corrupt persisted language can never break the shell.
    """
    if code in SUPPORTED_LANGUAGES:
        return code
    return DEFAULT_LANGUAGE


def language_label_for(code: Optional[str]) -> str:
    """Return the combo display label for a (possibly invalid) language code."""
    return LANGUAGE_LABELS[normalize_language(code)]


def translate(key: str, language: Optional[str] = None, default: Optional[str] = None) -> str:
    """Return the visible string for ``key`` in ``language`` with safe fallback.

    Resolution order: exact match in the requested language, English fallback,
    the explicit ``default``, then the key itself.  Never raises.
    """
    lang = normalize_language(language)
    entry = TRANSLATIONS.get(key)
    if entry is not None:
        if lang in entry:
            return entry[lang]
        if DEFAULT_LANGUAGE in entry:
            return entry[DEFAULT_LANGUAGE]
    if default is not None:
        return default
    return key


def supported_language_codes() -> list:
    """Return the list of supported language codes (copy, not the module tuple)."""
    return list(SUPPORTED_LANGUAGES)
