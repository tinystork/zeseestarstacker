# --- START OF FILE seestar/localization/fr.py ---
"""
Fichier de traductions françaises pour Seestar Stacker.
"""

FR_TRANSLATIONS = {
    # Interface principale & Commun
    'title': "Seestar Stacker",
    'error': "Erreur", 'warning': "Avertissement", 'info': "Information", 'quit': "Quitter",
    'browse': "Parcourir", # Générique
    'browse_input_button': "Parcourir...", # Unique
    'browse_output_button': "Parcourir...", # Unique
    'browse_ref_button': "Parcourir...", # Unique

    # --- Onglets Contrôles ---
    'tab_stacking': "Empilement",
    'tab_preview': "Aperçu",

    # --- Onglet Empilement ---
    'Folders': "Dossiers",
    'input_folder': "Entrée :",
    'output_folder': "Sortie :",
    'reference_image': "Référence (Opt.) :",
    'options': "Options d'Empilement",
    'stacking_method': "Méthode :",
    'kappa_value': "Kappa :",
    'batch_size': "Taille Lot :",
    'batch_size_auto': "(0=auto)", # Gardé pour affichage
    'hot_pixels_correction': 'Correction Pixels Chauds',
    'perform_hot_pixels_correction': 'Corriger pixels chauds',
    'hot_pixel_threshold': 'Seuil :',
    'neighborhood_size': 'Voisinage :',
    'post_proc_opts_frame_label': "Options Post-Traitement",
    'cleanup_temp_check_label': "Nettoyer fichiers temporaires après traitement",
    'quality_weighting_frame': "Pondération par Qualité",
    'enable_weighting_check': "Activer la pondération",
    'weighting_metrics_label': "Métriques:",
    'weight_snr_check': "SNR",
    'weight_stars_check': "Nb Étoiles",
    'snr_exponent_label': "Exp. SNR:",
    'stars_exponent_label': "Exp. Étoiles:",
    'min_weight_label': "Poids Min:",

    # --- Onglet Aperçu ---
    'white_balance': "Balance des Blancs (Aperçu)",
    'wb_r': "Gain R :",
    'wb_g': "Gain V :",
    'wb_b': "Gain B :",
    'auto_wb': "Auto BdB",
    'reset_wb': "Réinit. BdB",
    'stretch_options': "Étir. Histogramme (Aperçu)",
    'stretch_method': "Méthode :",
    'stretch_bp': "Noir :",
    'stretch_wp': "Blanc :",
    'stretch_gamma': "Gamma :",
    'auto_stretch': "Auto Étir.",
    'reset_stretch': "Réinit. Étir.",
    'image_adjustments': "Réglages Image",
    'brightness': "Luminosité :",
    'contrast': "Contraste :",
    'saturation': "Saturation :",
    'reset_bcs': "Réinit. Réglages",

    # --- Zone Progression ---
    'progress': "Progression",
    'estimated_time': "ETA:",
    'elapsed_time': "Écoulé :",
    'Remaining:': "Restant :", # Clé pour label statique
    'Additional:': "Additionnels :", # Clé pour label statique
    'aligned_files_label': "Alignés :", # Texte statique
    'aligned_files_label_format': "Alignés : {count}", # Format d'affichage
    'global_eta_label': "ETA Global :",

    # --- Zone Aperçu (Panneau Droit) ---
    'preview': "Aperçu",
    'histogram': "Histogramme",

    # --- Boutons de Contrôle ---
    'start': "Démarrer",
    'stop': "Arrêter",
    'add_folder_button': "Ajouter Dossier",
    # NOUVEAU: Clés pour les nouveaux boutons
    'copy_log_button_text': "Copier",
    'open_output_button_text': "Ouvrir Sortie",
    'show_folders_button_text': "Voir Entrées",

    # --- Titres Dialogues ---
    'Select Input Folder': "Sélectionner Dossier d'Entrée",
    'Select Output Folder': "Sélectionner Dossier de Sortie",
    'Select Reference Image (Optional)': "Sélectionner Image Référence (Optionnel)",
    'Select Additional Images Folder': "Sélectionner Dossier Images Additionnelles",
    'input_folders_title': "Liste Dossiers d'Entrée",
    'no_input_folder_set': "Aucun dossier d'entrée n'a été sélectionné.",
    
    # --- Messages & Statuts ---
    'select_folders': "Veuillez sélectionner les dossiers d'entrée et de sortie.",
    'input_folder_invalid': "Dossier d'entrée invalide",
    'output_folder_invalid': "Dossier de sortie invalide/impossible à créer",
    'Output folder created': "Dossier de sortie créé",
    'no_fits_found': "Aucun fichier .fit/.fits trouvé dans le dossier d'entrée. Démarrer quand même ?",
    'Error reading input folder': "Erreur lecture dossier entrée",
    'stacking_start': "⚙️ Début du traitement...",
    'stacking_stopping': "⚠️ Arrêt en cours...",
    'stacking_finished': "🏁 Traitement Terminé",
    'stacking_error_msg': "Erreur de Traitement :",
    'stacking_complete_msg': "Traitement terminé ! Stack final :",
    'stop_requested': "⚠️ Arrêt demandé, fin de l'étape en cours...",
    'processing_stopped': "🛑 Traitement arrêté par l'utilisateur.",
    'no_stacks_created': "⚠️ Aucun stack n'a été créé.",
    'Failed to start processing.': "Échec du démarrage du traitement.",
    'image_info_waiting': "Info image : en attente...",
    'no_files_waiting': "Aucun fichier en attente",
    'no_additional_folders': "Aucun",
    '1 additional folder': "1 dossier add.",
    '{count} additional folders': "{count} dossiers add.",
    'Start processing to add folders': "Le traitement doit être démarré pour ajouter des dossiers.", # Reste pertinent
    'Processing not active or finished.': 'Traitement inactif ou terminé.',
    'Folder not found': "Dossier non trouvé",
    'Input folder cannot be added': "Le dossier d'entrée principal ne peut pas être ajouté.",
    'Output folder cannot be added': "Le dossier de sortie ne peut pas être ajouté.",
    'Cannot add subfolder of output folder': "Impossible d'ajouter un sous-dossier du dossier de sortie.",
    'Folder already added': "Ce dossier est déjà dans la liste.",
    'Folder contains no FITS': "Le dossier ne contient aucun fichier FITS.",
    'Error reading folder': "Erreur lecture dossier",
    'Folder added': "Dossier ajouté",
    'Folder not added (already present, empty, or error)': "Dossier non ajouté (déjà présent, vide, ou erreur)",
    'quit_while_processing': "Traitement en cours. Quitter quand même ?",
    'Error during debayering': "Erreur pendant debayering",
    'Invalid or missing BAYERPAT': "BAYERPAT invalide ou manquant",
    'Treating as grayscale': "Traitement comme niveaux de gris",
    'Error loading preview image': "Erreur chargement image aperçu",
    'Error loading preview (invalid format?)': "Erreur chargement aperçu (format invalide?)",
    'Error loading final stack preview': "Erreur chargement aperçu stack final",
    'Error loading final preview': "Erreur chargement aperçu final",
    'No Image Data': "Aucune Donnée Image",
    'Preview Error': "Erreur Aperçu",
    'Preview Processing Error': "Erreur Traitement Aperçu",
    'Welcome!': "Bienvenue !",
    'Select input/output folders.': "Sélectionnez les dossiers d'entrée/sortie.",
    'Auto WB requires a color image preview.': "L'Auto BdB requiert un aperçu d'image couleur.",
    'Error during Auto WB': 'Erreur pendant Auto BdB',
    'Auto Stretch requires an image preview.': "L'Auto Étirement requiert un aperçu d'image.",
    'Error during Auto Stretch': 'Erreur pendant Auto Étirement',
    'Total Exp (s)': "Expo Totale (s)",
    'processing_report_title': "Résumé du Traitement",
    'report_images_stacked': "Images Empilées :",
    'report_total_exposure': "Temps Pose Total :",
    'report_total_time': "Temps Traitement Total :",
    'report_seconds': "secondes",
    'report_minutes': "minutes",
    'report_hours': "heures",
    'eta_calculating': "Calcul...",
    # --- Weighting Info Display ---
    'Weighting': 'Pondération', # Clé pour label 'WGHT_ON'
    'W. Metrics': 'Métr. Poids', # Clé pour label 'WGHT_MET'
    'weighting_enabled': "Activée", # Valeur pour WGHT_ON=True
    'weighting_disabled': "Désactivée", # Valeur pour WGHT_ON=False
}
# --- END OF FILE seestar/localization/fr.py ---