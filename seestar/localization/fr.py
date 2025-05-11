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
    'drizzle_options_frame_label': "Options Drizzle",
    'drizzle_activate_check': "Activer Drizzle (expérimental, lent)",
    'drizzle_scale_label': "Facteur Échelle :",
    'drizzle_radio_2x_label': "x2",
    'drizzle_radio_3x_label': "x3",
    'drizzle_radio_4x_label': "x4",
    'drizzle_mode_label': "Mode Drizzle :",
    'drizzle_radio_final': "Standard (Final)",
    'drizzle_radio_incremental': "Incrémental (Éco. Disque)",
    'drizzle_kernel_label': "Noyau :",
    'drizzle_pixfrac_label': "Pixfrac :",
    
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
    
    ### Traductions SCNR Final ###
    'apply_final_scnr_label': "Appliquer SCNR Final (Vert)",
    'final_scnr_amount_label': "Intensité SCNR :",
    'final_scnr_preserve_lum_label': "Préserver la Luminance (SCNR)",
    
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



    ###  Traductions pour l'Onglet Expert ###
    'tab_expert_title': "Expert",
    'expert_warning_text': "Toi qui entre ici, perds tout espoir ! (Réglages Avancés)",
    
    # Section Neutralisation du Fond (BN)
    'bn_frame_title': "Neutralisation Fond Auto (BN)",
    'bn_grid_size_label': "Taille Grille BN :",
    'bn_perc_low_label': "Perc. Bas Fond :",
    'bn_perc_high_label': "Perc. Haut Fond :",
    'bn_std_factor_label': "Facteur Std Fond :",
    'bn_min_gain_label': "Gain Min BN :",
    'bn_max_gain_label': "Gain Max BN :",

    # Section ChromaticBalancer (CB) / Edge Enhance
    'cb_frame_title': "Correction Bords/Chroma (CB)",
    'cb_border_size_label': "Taille Bord CB (px) :",
    'cb_blur_radius_label': "Flou Bord CB (px) :",
    'cb_min_b_factor_label': "Facteur Bleu Min CB :", # Supposant qu'on expose le facteur Bleu
    'cb_max_b_factor_label': "Facteur Bleu Max CB :", # et potentiellement R aussi
    # 'cb_intensity_label': "Intensité Correction Bords :", # Si vous ajoutez ce slider

    # Section Rognage Final
    'crop_frame_title': "Rognage Final",
    'final_edge_crop_label': "Rognage Bords (%) :",

    # Bouton Réinitialiser Expert
    'reset_expert_button': "Réinitialiser Réglages Expert",
    
    ### Traductions Photutils Background Subtraction (Onglet Expert) ###
    'photutils_bn_frame_title': "Soustraction Fond 2D (Photutils)",
    'apply_photutils_bn_label': "Activer Soustraction Fond 2D Photutils",
    'photutils_bn_box_size_label': "Taille Boîte (px) :",
    'photutils_bn_filter_size_label': "Taille Filtre (px, impair) :",
    'photutils_bn_sigma_clip_label': "Sigma Clip :",
    'photutils_bn_exclude_percentile_label': "Exclure Plus Brillants (%) :",
    ### FIN onglet expert ###

  
    ### Tooltips pour Onglet Expert ###
    'tooltip_bn_grid_size': "BN: Grille RxC pour analyse du fond. Plus de zones (32x32) = analyse locale fine, sensible au bruit. Moins (8x8) = stats robustes, moins bon pour gradients complexes. Défaut: 16x16.",
    'tooltip_bn_perc_low': "BN: Percentile bas pour luminance des zones de fond. Les zones SOUS ce percentile global sont moins considérées comme fond. Plage: 0-40. Défaut: 5.",
    'tooltip_bn_perc_high': "BN: Percentile haut pour luminance des zones de fond. Les zones AU-DESSUS sont exclues du fond. Plage: (Perc.Bas+1)-95. Défaut: 30.",
    'tooltip_bn_std_factor': "BN: Facteur pour l'écart-type. Zones de fond si leur std_dev < (std_dev_médian_global * facteur). Bas = strict. Haut = permissif. Plage: 0.5-5. Défaut: 1.0.",
    'tooltip_bn_min_gain': "BN: Gain minimal appliqué aux canaux R/B pour égaler G. Limite la correction. Plage: 0.1-2. Défaut: 0.2.",
    'tooltip_bn_max_gain': "BN: Gain maximal appliqué aux canaux R/B. Limite la correction. Plage: 1-10. Défaut: 7.0.",
    'tooltip_cb_border_size': "Correct.Bords: Largeur (px) de la zone de bord analysée pour les décalages couleur locaux. Petit = cible franges fines. Grand = correction plus large. Plage: 5-150. Défaut: 25.",
    'tooltip_cb_blur_radius': "Correct.Bords: Rayon (px) du flou sur le masque de bord. Adoucit la transition de la correction. Petit = transition nette. Plage: 0-50. Défaut: 8.",
    'tooltip_cb_min_b_factor': "Correct.Bords: Facteur de gain minimal appliqué au canal Bleu dans les bords pour égaler le Vert local. Limite la réduction. Plage: 0.1-1.0. Défaut: 0.4.",
    'tooltip_cb_max_b_factor': "Correct.Bords: Facteur de gain maximal appliqué au canal Bleu. Limite l'amplification. Plage: 1.0-3.0. Défaut: 1.5.",
    'tooltip_final_edge_crop_percent': "Rognage Final: Pourcentage de l'image à rogner sur CHAQUE côté (G,D,H,B) avant sauvegarde. Ex: 2.0 pour 2%. Plage: 0-25. Défaut: 2.0.",
    'tooltip_apply_final_scnr': "SCNR: Applique une Réduction Subtile du Bruit de Couleur au stack final, ciblant le Vert par défaut pour réduire les dominantes vert/magenta.", # Déjà existant, mais bon de vérifier
    'tooltip_final_scnr_amount': "Intensité SCNR: Force de la réduction du Vert (0.0=aucune, 1.0=remplacement complet par réf. R/B). Typique: 0.6-0.9. Défaut: 0.8.", # Déjà existant
    'tooltip_final_scnr_preserve_lum': "Préserver Luminance SCNR: Si coché, tente de restaurer la luminance originale des pixels après correction couleur, évitant un assombrissement excessif.", # Déjà existant
    # Tooltips pour Photutils BN
    'tooltip_apply_photutils_bn': "PB2D: Active la soustraction d'un modèle de fond 2D calculé par Photutils. Agit avant la Neutralisation de Fond globale. Utile pour les gradients complexes.",
    'tooltip_photutils_bn_box_size': "PB2D: Taille des boîtes (px) pour l'estimation locale du fond. Doit être assez grand pour éviter les étoiles, mais assez petit pour suivre le gradient. Défaut: 128.",
    'tooltip_photutils_bn_filter_size': "PB2D: Taille du filtre médian (px, impair) appliqué à la carte des estimations de fond locales pour la lisser. Défaut: 5.",
    'tooltip_photutils_bn_sigma_clip': "PB2D: Valeur Sigma pour le rejet itératif des pixels (étoiles) dans chaque boîte lors de l'estimation du fond. Défaut: 3.0.",
    'tooltip_photutils_bn_exclude_percentile': "PB2D: Pourcentage (0-100) des pixels les plus brillants à ignorer dans chaque boîte avant l'estimation du fond. Aide à rejeter les étoiles sans masque. Défaut: 98.0.",
    ### FIN Tooltips Photo utils ###
    ### FIN tooltips expert ###


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
    'copy_log_button_text': "Copier",
    'open_output_button_text': "Ouvrir Sortie",
    'show_folders_button_text': "Voir Entrées",
    'analyze_folder_button': "Analyser Dossier (Externe)",

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
    
    #Log pop up final
    'Post-Processing Applied': "Post-Traitements Appliqués",
    'Photutils 2D Background': "Soustraction Fond 2D Photutils",
    'Global Background Neutralization (BN)': "Neutralisation Fond Globale (BN)",
    'Edge/Chroma Correction (CB)': "Correction Bords/Chroma (CB)",
    'Final SCNR': "SCNR Final",
    'Final Edge Crop': "Rognage Final des Bords",
    'Yes': "Oui",
    'No': "Non",
    'Not Found!': "Non Trouvé !",
    
    
    # --- Weighting Info Display ---
    'Weighting': 'Pondération', # Clé pour label 'WGHT_ON'
    'W. Metrics': 'Métr. Poids', # Clé pour label 'WGHT_MET'
    'weighting_enabled': "Activée", # Valeur pour WGHT_ON=True
    'weighting_disabled': "Désactivée", # Valeur pour WGHT_ON=False
    'drizzle_wht_threshold_label': "Seuil WHT%:",
    
    
    
    # --- Mosaïque ---
    'Mosaic...': "Mosaïque...",
    'mosaic_settings_title': "Options Mosaïque",
    'mosaic_activation_frame': "Activation",
    'mosaic_activate_label': "Activer le mode de traitement Mosaïque",
    'cancel': "Annuler",
    'ok': "OK",
    'mosaic_window_create_error': "Impossible d'ouvrir la fenêtre d'options Mosaïque.",
    'mosaic_mode_enabled_log': "Mode mosaïque ACTIVÉ.",
    'mosaic_mode_disabled_log': "Mode mosaïque DÉSACTIVÉ.",    
    'mosaic_drizzle_options_frame': "Options Drizzle Mosaïque",
    'mosaic_drizzle_kernel_label': "Noyau :",
    'mosaic_drizzle_pixfrac_label': "Pixfrac :",
    'mosaic_invalid_kernel': "Noyau Drizzle sélectionné invalide.",
    'mosaic_invalid_pixfrac': "Valeur Pixfrac invalide. Doit être entre 0.01 et 1.0.",
    'mosaic_mode_enabled_log': "Mode mosaïque ACTIVÉ.",
    'mosaic_mode_disabled_log': "Mode mosaïque DÉSACTIVÉ.",
    'mosaic_api_key_frame': "Clé API Astrometry.net (Requise pour Mosaïque)",
    'mosaic_api_key_label': "Clé API :",
    'mosaic_api_key_help': "Obtenez votre clé sur nova.astrometry.net (compte gratuit)",
    'mosaic_api_key_required': "La clé API Astrometry.net est requise lorsque le Mode Mosaïque est activé.",
    #--- Tooltips pour Feathering ---
    'tooltip_apply_feathering': "Feathering : Si activé, adoucit l'image empilée en se basant sur une version floutée de la carte de poids totale. Peut aider à réduire les transitions brusques ou les artefacts aux bords des données combinées ou là où les poids changent abruptement. Agit avant la soustraction de fond Photutils.",
    'tooltip_feather_blur_px': "Rayon de Flou Feathering (px) : Contrôle l'étendue du flou appliqué à la carte de poids pour le feathering. Des valeurs plus grandes donnent des transitions plus douces et graduelles. Plage typique : 64-512. Défaut : 256.",
    # ---  ---

    # ---  Textes pour Avertissement Drizzle ---
    'drizzle_warning_title': "Avertissement Drizzle",
    'drizzle_warning_text': (
        "Le traitement Drizzle est activé.\n\n"
        "- Il est expérimental et peut être lent.\n"
        "- Il créera des fichiers temporaires pouvant occuper beaucoup d'espace disque (potentiellement autant que les images d'entrée).\n",
        "- L'aperçu en direct montrera un stack classique ; le Drizzle sera appliqué à la toute fin.\n\n",
        "Continuer avec Drizzle ?"
    


    ),
    


}
# --- END OF FILE seestar/localization/fr.py ---