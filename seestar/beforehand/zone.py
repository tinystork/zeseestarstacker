# --- START OF FILE zone.py ---

# Fichier pour l'internationalisation (i18n) et les infobulles

translations = {
    'fr': {
        # --- Fenêtre principale ---
        'window_title': "Analyseur d'Images Astronomiques",
        'status_ready': "Prêt",
        'status_analysis_start': "Démarrage de l'analyse...",
        'status_analysis_prep': "Préparation de l'analyse...",
        'status_discovery_start': "Recherche des fichiers FITS...",
        'status_satdet_wait': "Détection traînées (acstools)... Patientez...",
        'status_satdet_no_file': "Détection traînées: Aucun fichier FITS trouvé.",
        'status_satdet_done': "Détection de traînées terminée.",
        'status_satdet_error': "Erreur lors de la détection de traînées.",
        'status_satdet_dep_error': "Erreur dépendance détection (scipy/skimage?).",
        'status_snr_start': "Analyse: {file} ({i}/{total})",
        'status_analysis_done': "Analyse terminée.",
        'status_analysis_done_some': "Analyse terminée. {processed} traités, {moved} actions (déplacé/supprimé), {errors} erreurs fichier.",
        'status_analysis_done_ok': "Analyse terminée avec succès.",
        'status_analysis_done_no_valid': "Analyse terminée. Aucune image traitable trouvée ou tous les dossiers ont été ignorés.", # Modifié
        'status_analysis_done_errors': "Analyse terminée avec des erreurs critiques.",
        'status_log_error': "Erreur écriture log",
        'status_dir_create_error': "Erreur création dossier: {e}",

        # --- Cadres ---
        'config_frame_title': "Configuration Générale",
        'snr_frame_title': "Analyse SNR & Sélection",
        'trail_frame_title': "Détection Traînées",
        'action_frame_title': "Action sur Images Rejetées",
        'display_options_frame_title': "Options d'Affichage",
        'results_frame_title': "Résultats / Journal",

        # --- Labels & Champs ---
        'input_dir_label': "Dossier d'entrée:",
        'output_log_label': "Fichier log:",
        'include_subfolders_label': "Inclure les sous-dossiers",
        'lang_label': "Langue:",
        'analyze_snr_check_label': "Activer l'analyse SNR",
        'snr_select_mode_label': "Mode de Sélection SNR:",
        'snr_mode_percent': "Top Pourcentage (%)",
        'snr_mode_threshold': "Seuil SNR (>)",
        'snr_mode_none': "Tout Garder",
        'snr_reject_dir_label': "Dossier Rejet (Faible SNR):",
        'detect_trails_check_label': "Activer détection traînées",
        'sigma_label': "Sigma:", 'low_thresh_label': "Low Thr:", 'h_thresh_label': "High Thr:",
        'line_len_label': "Line Len:", 'small_edge_label': "Small Edge:", 'line_gap_label': "Line Gap:",
        'trail_reject_dir_label': "Dossier Rejet (Traînées):",
        'action_label': "Action:", 'action_mode_move': "Déplacer vers dossier Rejet", 'action_mode_delete': "Supprimer Définitivement", 'action_mode_none': "Ne Rien Faire (Tout Garder)",
        'sort_snr_check_label': "Trier les résultats par SNR décroissant",

        # --- Boutons ---
        'browse_button': "Parcourir",
        'analyse_button': "Analyser les images",
        'visualize_button': "Visualiser les résultats",
        'open_log_button': "Ouvrir le fichier log",
        'manage_markers_button': "Gérer Marqueurs", # <-- NOUVEAU
        'quit_button': "Quitter",
        'return_button_text': "Retour",
        'export_button': "Exporter Liste Recommandée (.txt)",
        'Fermer': "Fermer",
        'Exporter Toutes Conservées': "Exporter Toutes Conservées",

        # --- Textes acstools status ---
        'acstools_ok': "(acstools disponible)", 'acstools_missing': "(acstools non trouvé ou incompatible)", 'acstools_sig_error': "(fonction detsat incompatible)",

        # --- Messages Box ---
        'msg_error': "Erreur", 'msg_warning': "Attention", 'msg_info': "Information",
        'msg_missing_logic': "Le fichier analyse_logic.py est manquant.", 'msg_input_dir_invalid': "Dossier d'entrée invalide.", 'msg_log_file_missing': "Fichier log non spécifié.",
        'non spécifié': "non spécifié", 'confirm_delete': "Êtes-vous sûr de vouloir supprimer définitivement les fichiers rejetés ? Cette action est irréversible.", 'snr_value_missing': "Veuillez entrer une valeur pour la sélection SNR (% ou seuil).", 'snr_value_invalid': "Valeur invalide pour la sélection SNR. Doit être numérique.",
        'msg_satdet_incompatible': "Détection activée mais acstools/satdet n'est pas compatible ou disponible.", 'msg_params_invalid': "Paramètres invalides: {e}", 'msg_analysis_running': "Une analyse est déjà en cours.",
        'Analyse en cours, quitter quand même?': "Une analyse est en cours. Voulez-vous vraiment quitter ?", 'msg_no_results_visualize': "Aucun résultat à visualiser.", 'msg_analysis_wait_visualize': "Attendez la fin de l'analyse pour visualiser.", 'msg_results_incomplete': "Résultats incomplets ou analyse terminée avec erreurs.",
        'Affichage des données disponibles.': "Affichage des données disponibles.",
        'msg_log_not_exist': "Le fichier log n'existe pas ou n'est pas spécifié.", 'msg_log_open_error': "Impossible d'ouvrir '{path}':\n{e}", 'msg_export_no_images': "Aucune image à exporter.", 'msg_export_success': "Liste de {count} fichiers exportée vers:\n{path}", 'msg_export_error': "Erreur écriture fichier:\n{e}",
        'msg_dep_missing_title': "Dépendances Manquantes", 'msg_dep_missing_text': "Bibliothèques manquantes:\n- {deps}\n\nInstaller via pip ?", 'msg_dep_installing': "Installation dépendances...", 'msg_dep_install_pkg': "Installation de {package}...", 'msg_dep_install_success': " -> Succès.", 'msg_dep_install_fail': " -> ÉCHEC: {e}", 'msg_dep_install_error': "Impossible d'installer {package}.\n{e}", 'msg_dep_install_done': "Dépendances installées. Redémarrez l'application.", 'msg_dep_install_partial': "Certaines dépendances n'ont pu être installées.", 'msg_dep_error_continue': "Dépendances manquantes. L'application pourrait mal fonctionner.",
        'msg_tkinter_error': "Erreur Tkinter:\n{e}", 'msg_unexpected_error': "Erreur inattendue:\n{e}",

        # --- Fenêtre Visualisation ---
        'visu_window_title': "Visualisation des résultats", 'visu_tab_snr_dist': "Distribution SNR", 'visu_tab_snr_comp': "Comparaison SNR", 'visu_tab_sat_trails': "Traînées Détectées", 'visu_tab_raw_data': "Données Détaillées", 'visu_tab_recom': "Recommandations Stacking",
        'visu_snr_dist_title': 'Distribution du Rapport Signal/Bruit (SNR)', 'visu_snr_dist_xlabel': 'SNR', 'visu_snr_dist_ylabel': 'Nombre d\'images', 'visu_snr_dist_no_data': "Aucune donnée SNR valide",
        'visu_snr_comp_best_title': 'Top {n} Images (Meilleur SNR)', 'visu_snr_comp_worst_title': 'Bottom {n} Images (Pire SNR)', 'visu_snr_comp_xlabel': 'SNR', 'visu_snr_comp_no_data': "Pas assez d'images avec SNR valide pour comparer.",
        'visu_sat_pie_title': 'Proportion d\'images avec/sans traînées détectées', 'visu_sat_pie_with': 'Avec Traînées', 'visu_sat_pie_without': 'Sans Traînées', 'visu_sat_pie_no_data': "Aucune image analysée pour les traînées.",
        'visu_data_col_file': "Fichier (Relatif)", 'visu_data_col_snr': "SNR", 'visu_data_col_bg': "Fond", 'visu_data_col_noise': "Bruit", 'visu_data_col_pixsig': "PixSig", 'visu_data_col_trails': "Traînées?", 'visu_data_col_nbseg': "Nb Seg.",
        'Statut': "Statut", 'Action': "Action", 'Raison Rejet': "Raison Rejet", 'Commentaire': "Commentaire",
        'visu_recom_frame_title': "Recommandation (Images Conservées)", 'visu_recom_text': "Suggestion: Utiliser les {count} images conservées avec SNR >= {p75:.2f} (P25)", 'visu_recom_col_file': "Fichier (Relatif)", 'visu_recom_col_snr': "SNR", 'visu_recom_no_selection': "Aucune image conservée ne dépasse le seuil de recommandation.", 'visu_recom_not_enough': "Moins de 5 images conservées valides. Utilisez/Exportez toutes les images conservées.", 'visu_recom_no_data': "Aucune donnée SNR valide pour recommandation.",
        'Toutes les images conservées valides': "Toutes les images conservées valides",

        # --- Infobulles (Tooltips) ---
        'tooltip_sigma': "Sigma du filtre Gaussien avant détection contours (flou). Défaut: 2.0", 'tooltip_low_thresh': "Seuil bas Canny (0-1). Détecte bords potentiels. Bas=Sensible bruit. Défaut acstools: 0.1", 'tooltip_h_thresh': "Seuil haut Canny (0-1). Ancre bords forts. >= Low Thr. Bas=Permissif. Défaut acstools: 0.5", 'tooltip_line_len': "Longueur min. (px) segment final pour être une traînée. Défaut acstools: 150", 'tooltip_small_edge': "Longueur min. (px) contour initial (par Canny). Filtre bruit avant Hough. Défaut acstools: 60", 'tooltip_line_gap': "Écart max (px) pour joindre segments. Défaut acstools: 75", 'tooltip_snr_value': "Entrez % (ex: 80) ou seuil SNR (ex: 5.5). Ignoré si Tout Garder.",

        # --- Textes Logique ---
        'logic_info_prefix': "INFO: ", 'logic_log_prefix': "LOG: ", 'logic_status_prefix': "STATUS: ", 'logic_warn_prefix': "Avertissement: ", 'logic_error_prefix': "Erreur: ",
        'logic_sat_incomp': "Détection satellites non disponible ou incompatible.", 'logic_sigma_invalid': "Sigma invalide ({e}), utilisation de {default}", 'logic_lowthr_invalid': "Low Thresh invalide ({e}), utilisation de {default}", 'logic_highthr_invalid': "High Thresh invalide ({e}), utilisation de {default}",
        'logic_satdet_params': "Détection avec: sigma={sigma}, low={low_thresh}, high={h_thresh}, chips={chips}, line_len={line_len}, small_edge={small_edge}, line_gap={line_gap}",
        'logic_satdet_errors_title': "Erreurs spécifiques reportées par satdet:", 'logic_satdet_errors_item': "  - {fname} (ext {ext}): {msg}", 'logic_satdet_errors_none': "  (Aucune erreur pertinente à afficher)",
        'logic_satdet_import_error': "Erreur importation pour satdet: {e}. Vérifiez scipy/skimage.", 'logic_satdet_major_error': "Erreur majeure lors de l'appel à acstools.satdet.detsat: {e}",
        'logic_dir_created': "Dossier créé: {path}", 'logic_dir_create_error': "Erreur création dossier {path}: {e}", 'logic_log_init_error': "Impossible d'écrire dans le fichier log initial {path}: {e}",
        'logic_no_fits_satdet': "Aucun fichier FITS trouvé dans {path} pour détection.", 'logic_no_fits_snr': "Aucun fichier FITS (.fit, .fits) trouvé pour l'analyse SNR.",
        'logic_snr_start': "Démarrage de l'analyse individuelle...", 'logic_fits_no_data': "{file} - Pas de données image dans HDU 0.", 'logic_snr_info': "  {file}: SNR={snr:.2f}, Fond={bg:.2f}", 'logic_trail_info': "    Traînées (segments): {status} ({count})", 'logic_trail_yes': "Oui", 'logic_trail_no': "Non",
        'logic_moved_info': "-> Déplacé vers {folder}", 'logic_move_skipped': "    Info: {file} n'existait plus à l'emplacement source pour action.", 'logic_move_error': "    Erreur déplacement {file}: {e}", 'logic_file_error': "Erreur analyse fichier {file}: {e}",
        'logic_log_summary_error': "Erreur lors de l'écriture du résumé du log ({path}): {e}", 'logic_final_snr': "SNR moyen global: {mean:.2f}", 'logic_final_trails': "Images avec traînées détectées: {count} ({percent:.1f}%)", 'logic_final_no_success': "Aucune image n'a pu être traitée avec succès.",
        'Liste dimages recommandées': "Liste d'images recommandées", 'Critère': "Critère", 'Généré le': "Généré le", 'Nombre dimages': "Nombre d'images",
        'Fichiers log': "Fichiers log", 'Tous les fichiers': "Tous les fichiers", 'Fichiers Texte': "Fichiers Texte",

        # --- NOUVEAU: Textes pour la gestion des marqueurs ---
        'marker_window_title': "Gérer les Marqueurs d'Analyse",
        'marker_info_label': "Dossiers marqués comme analysés (contiennent le fichier '.astro_analyzer_run_complete'):",
        'marker_none_found': "Aucun dossier marqué trouvé.",
        'marker_select_none': "Veuillez sélectionner un ou plusieurs dossiers dans la liste.",
        'marker_confirm_delete_selected': "Supprimer les marqueurs pour les {count} dossiers sélectionnés ?\nCela forcera leur ré-analyse au prochain lancement.",
        'marker_confirm_delete_all': "Supprimer TOUS les marqueurs ({count}) dans le dossier '{folder}' et ses sous-dossiers analysables ?\nCela forcera une ré-analyse complète.",
        'marker_delete_selected_button': "Supprimer Sélection",
        'marker_delete_all_button': "Supprimer Tout",
        'marker_delete_errors': "Erreurs lors de la suppression de certains marqueurs:\n",
        'marker_delete_selected_success': "{count} marqueur(s) supprimé(s).",
        'marker_delete_all_success': "Tous les {count} marqueur(s) trouvés ont été supprimés.",
        # --- FIN NOUVEAU ---
    },
    'en': {
        # --- Main Window ---
        'window_title': "Astronomical Image Analyzer",
        'status_ready': "Ready", 'status_analysis_start': "Starting analysis...", 'status_analysis_prep': "Preparing analysis...",
        'status_discovery_start': "Discovering FITS files...", # NEW
        'status_satdet_wait': "Detecting trails (acstools)... Please wait...", 'status_satdet_no_file': "Trail detection: No FITS files found.",
        'status_satdet_done': "Trail detection finished.", 'status_satdet_error': "Error during trail detection.", 'status_satdet_dep_error': "Detection dependency error (scipy/skimage?).",
        'status_snr_start': "Analyzing: {file} ({i}/{total})", 'status_analysis_done': "Analysis finished.",
        'status_analysis_done_some': "Analysis finished. {processed} processed, {moved} actions (moved/deleted), {errors} file errors.",
        'status_analysis_done_ok': "Analysis completed successfully.",
        'status_analysis_done_no_valid': "Analysis finished. No processable images found or all folders were skipped.", # Modified
        'status_analysis_done_errors': "Analysis finished with critical errors.",
        'status_log_error': "Log writing error", 'status_dir_create_error': "Folder creation error: {e}",

        # --- Frames ---
        'config_frame_title': "General Configuration", 'snr_frame_title': "SNR Analysis & Selection", 'trail_frame_title': "Trail Detection", 'action_frame_title': "Action on Rejected Images", 'display_options_frame_title': "Display Options", 'results_frame_title': "Results / Log",

        # --- Labels & Fields ---
        'input_dir_label': "Input Folder:", 'output_log_label': "Log File:",
        'include_subfolders_label': "Include Subfolders", # <-- NEW
        'lang_label': "Language:",
        'analyze_snr_check_label': "Enable SNR analysis", 'snr_select_mode_label': "SNR Selection Mode:", 'snr_mode_percent': "Top Percent (%)", 'snr_mode_threshold': "SNR Threshold (>)",
        'snr_mode_none': "Keep All", 'snr_reject_dir_label': "Reject Folder (Low SNR):",
        'detect_trails_check_label': "Enable trail detection",
        'sigma_label': "Sigma:", 'low_thresh_label': "Low Thr:", 'h_thresh_label': "High Thr:",
        'line_len_label': "Line Len:", 'small_edge_label': "Small Edge:", 'line_gap_label': "Line Gap:",
        'trail_reject_dir_label': "Reject Folder (Trails):", 'action_label': "Action:", 'action_mode_move': "Move to Reject Folder", 'action_mode_delete': "Delete Permanently", 'action_mode_none': "Do Nothing (Keep All)",
        'sort_snr_check_label': "Sort results by descending SNR",

        # --- Buttons ---
        'browse_button': "Browse", 'analyse_button': "Analyze Images", 'visualize_button': "Visualize Results", 'open_log_button': "Open Log File",
        'manage_markers_button': "Manage Markers", # <-- NEW
        'quit_button': "Quit", 'return_button_text': "Return", 'export_button': "Export Recommended List (.txt)",
        'Fermer': "Close", 'Exporter Toutes Conservées': "Export All Kept",

        # --- acstools status text ---
        'acstools_ok': "(acstools available)", 'acstools_missing': "(acstools not found or incompatible)", 'acstools_sig_error': "(detsat function incompatible)",

        # --- Message Boxes ---
        'msg_error': "Error", 'msg_warning': "Warning", 'msg_info': "Information",
        'msg_missing_logic': "analyse_logic.py file is missing.", 'msg_input_dir_invalid': "Invalid input folder.", 'msg_log_file_missing': "Log file not specified.",
        'non spécifié': "not specified", 'confirm_delete': "Are you sure you want to permanently delete rejected files? This cannot be undone.", 'snr_value_missing': "Please enter a value for SNR selection (% or threshold).", 'snr_value_invalid': "Invalid value for SNR selection. Must be numeric.",
        'msg_satdet_incompatible': "Detection enabled but acstools/detsat is not compatible or available.", 'msg_params_invalid': "Invalid parameters: {e}", 'msg_analysis_running': "An analysis is already in progress.",
        'Analyse en cours, quitter quand même?': "Analysis in progress. Quit anyway?", 'msg_no_results_visualize': "No results to visualize.", 'msg_analysis_wait_visualize': "Wait for the analysis to finish before visualizing.", 'msg_results_incomplete': "Incomplete results or analysis finished with errors.",
        'Affichage des données disponibles.': "Displaying available data.",
        'msg_log_not_exist': "Log file does not exist or is not specified.", 'msg_log_open_error': "Cannot open '{path}':\n{e}", 'msg_export_no_images': "No images to export.", 'msg_export_success': "List of {count} filenames exported to:\n{path}", 'msg_export_error': "Error writing file:\n{e}",
        'msg_dep_missing_title': "Missing Dependencies", 'msg_dep_missing_text': "Missing libraries:\n- {deps}\n\nInstall via pip?", 'msg_dep_installing': "Installing dependencies...", 'msg_dep_install_pkg': "Installing {package}...", 'msg_dep_install_success': " -> Success.", 'msg_dep_install_fail': " -> FAILED: {e}", 'msg_dep_install_error': "Could not install {package}.\n{e}",
        'msg_dep_install_done': "Dependencies installed. Please restart the application.", 'msg_dep_install_partial': "Some dependencies failed to install.", 'msg_dep_error_continue': "Missing dependencies. The application might not work correctly.",
        'msg_tkinter_error': "Tkinter Error:\n{e}", 'msg_unexpected_error': "Unexpected error:\n{e}",

        # --- Visualization Window ---
        'visu_window_title': "Results Visualization", 'visu_tab_snr_dist': "SNR Distribution", 'visu_tab_snr_comp': "SNR Comparison", 'visu_tab_sat_trails': "Detected Trails", 'visu_tab_raw_data': "Detailed Data", 'visu_tab_recom': "Stacking Recommendations",
        'visu_snr_dist_title': 'Signal-to-Noise Ratio (SNR) Distribution', 'visu_snr_dist_xlabel': 'SNR', 'visu_snr_dist_ylabel': 'Number of images', 'visu_snr_dist_no_data': "No valid SNR data",
        'visu_snr_comp_best_title': 'Top {n} Images (Best SNR)', 'visu_snr_comp_worst_title': 'Bottom {n} Images (Worst SNR)', 'visu_snr_comp_xlabel': 'SNR', 'visu_snr_comp_no_data': "Not enough images with valid SNR to compare.",
        'visu_sat_pie_title': 'Proportion of images with/without detected trails', 'visu_sat_pie_with': 'With Trails', 'visu_sat_pie_without': 'Without Trails', 'visu_sat_pie_no_data': "No images analyzed for trails.",
        'visu_data_col_file': "File (Relative)", 'visu_data_col_snr': "SNR", 'visu_data_col_bg': "BG", 'visu_data_col_noise': "Noise", 'visu_data_col_pixsig': "SigPix", 'visu_data_col_trails': "Trails?", 'visu_data_col_nbseg': "Nb Seg.",
        'Statut': "Status", 'Action': "Action", 'Raison Rejet': "Reject Reason", 'Commentaire': "Comment",
        'visu_recom_frame_title': "Recommendation (Kept Images)", 'visu_recom_text': "Suggestion: Use the {count} kept images with SNR >= {p75:.2f} (P25)", 'visu_recom_col_file': "File (Relative)", 'visu_recom_col_snr': "SNR", 'visu_recom_no_selection': "No kept images meet the recommendation threshold.", 'visu_recom_not_enough': "Fewer than 5 valid kept images. Use/Export all valid kept images.", 'visu_recom_no_data': "No valid SNR data for recommendation.",
        'Toutes les images conservées valides': "All valid kept images",

        # --- Tooltips ---
        'tooltip_sigma': "Sigma of Gaussian filter before edge detection (blur). Default: 2.0", 'tooltip_low_thresh': "Lower Canny threshold (0-1). Detects potential edges. Lower=Noisier. acstools default: 0.1", 'tooltip_h_thresh': "Upper Canny threshold (0-1). Anchors strong edges. >= Low Thr. Lower=Permissive. acstools default: 0.5", 'tooltip_line_len': "Min. length (px) of final line segment to be a trail. acstools default: 150", 'tooltip_small_edge': "Min. length (px) of initial edge segment (Canny). Filters noise before Hough. acstools default: 60", 'tooltip_line_gap': "Max allowed gap (px) to link segments. acstools default: 75", 'tooltip_snr_value': "Enter % (e.g., 80) or SNR threshold (e.g., 5.5). Ignored if Keep All.",

        # --- Logic Texts ---
        'logic_info_prefix': "INFO: ", 'logic_log_prefix': "LOG: ", 'logic_status_prefix': "STATUS: ", 'logic_warn_prefix': "Warning: ", 'logic_error_prefix': "Error: ",
        'logic_sat_incomp': "Satellite detection unavailable or incompatible.", 'logic_sigma_invalid': "Invalid Sigma ({e}), using {default}", 'logic_lowthr_invalid': "Invalid Low Thresh ({e}), using {default}", 'logic_highthr_invalid': "Invalid High Thresh ({e}), using {default}",
        'logic_satdet_params': "Detection with: sigma={sigma}, low={low_thresh}, high={h_thresh}, chips={chips}, line_len={line_len}, small_edge={small_edge}, line_gap={line_gap}",
        'logic_satdet_errors_title': "Specific errors reported by satdet:", 'logic_satdet_errors_item': "  - {fname} (ext {ext}): {msg}", 'logic_satdet_errors_none': "  (No relevant errors to display)",
        'logic_satdet_import_error': "Import error for satdet: {e}. Check scipy/skimage.", 'logic_satdet_major_error': "Major error calling acstools.satdet.detsat: {e}",
        'logic_dir_created': "Folder created: {path}", 'logic_dir_create_error': "Error creating folder {path}: {e}", 'logic_log_init_error': "Cannot write to initial log file {path}: {e}",
        'logic_no_fits_satdet': "No FITS files found in {path} for detection.", 'logic_no_fits_snr': "No FITS files (.fit, .fits) found for SNR analysis.",
        'logic_snr_start': "Starting individual file analysis...", 'logic_fits_no_data': "{file} - No image data in HDU 0.", 'logic_snr_info': "  {file}: SNR={snr:.2f}, Background={bg:.2f}", 'logic_trail_info': "    Trails (segments): {status} ({count})", 'logic_trail_yes': "Yes", 'logic_trail_no': "No",
        'logic_moved_info': "-> Moved to {folder}", 'logic_move_skipped': "    Info: {file} no longer existed at source location for action.", 'logic_move_error': "    Error moving {file}: {e}", 'logic_file_error': "Error analyzing file {file}: {e}",
        'logic_log_summary_error': "Error writing log summary ({path}): {e}", 'logic_final_snr': "Overall average SNR: {mean:.2f}", 'logic_final_trails': "Images with detected trails: {count} ({percent:.1f}%)", 'logic_final_no_success': "No images could be processed successfully.",
        'Liste dimages recommandées': "Recommended image list", 'Critère': "Criterion", 'Généré le': "Generated on", 'Nombre dimages': "Number of images",
        'Fichiers log': "Log Files", 'Tous les fichiers': "All Files", 'Fichiers Texte': "Text Files",

        # --- NEW: Marker Management Texts ---
        'marker_window_title': "Manage Analysis Markers",
        'marker_info_label': "Folders marked as analyzed (contain the '.astro_analyzer_run_complete' marker file):",
        'marker_none_found': "No marked folders found.",
        'marker_select_none': "Please select one or more folders from the list.",
        'marker_confirm_delete_selected': "Delete markers for the {count} selected folder(s)?\nThis will force re-analysis on the next run.",
        'marker_confirm_delete_all': "Delete ALL markers ({count}) in folder '{folder}' and its analyzable subfolders?\nThis will force a complete re-analysis.",
        'marker_delete_selected_button': "Delete Selected",
        'marker_delete_all_button': "Delete All",
        'marker_delete_errors': "Errors occurred while deleting some markers:\n",
        'marker_delete_selected_success': "{count} marker(s) deleted.",
        'marker_delete_all_success': "All {count} found marker(s) deleted.",
        # --- END NEW ---
    }
}
# --- FIN DU FICHIER zone.py ---