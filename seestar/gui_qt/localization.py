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
    "histogram_placeholder": {
        "en": "[ ] Histogram placeholder",
        "fr": "[ ] Emplacement histogramme",
    },
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
