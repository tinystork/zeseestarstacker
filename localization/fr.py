"""
Fichier de traductions françaises pour Seestar.
"""

FR_TRANSLATIONS = {
    # Interface principale
    'title': "Seestar Stacker",
    'input_folder': "Dossier entrée :",
    'output_folder': "Dossier sortie :",
    'browse': "Parcourir", # Generic
    'browse_input_button': "Parcourir...", # Unique
    'browse_output_button': "Parcourir...", # Unique
    'browse_ref_button': "Parcourir...", # Unique
    'Folders': "Dossiers",

    # Options
    'options': "Options",
    'stacking_method': "Méthode :",
    'kappa_value': "Kappa :",
    'batch_size': "Taille lot :",
    # 'remove_aligned': "Suppr. fichiers alignés", # Removed
    'apply_denoise': "Appliquer Débruitage",

    # Méthodes d'empilement
    'mean': "moyenne", 'median': "médiane", 'kappa-sigma': "kappa-sigma", 'winsorized-sigma': "winsorized-sigma",

    # Alignement
    'reference_image': "Référence (option) :",
    'alignment_start': "⚙️ Début de l'alignement...",
    'using_aligned_folder': "✅ Utilisation du dossier aligné : {}",
    'Alignment & Hot Pixels': 'Alignement & Pixels Chauds',
    'Getting reference image...': "⭐ Obtention image référence...",
    'Failed to get reference image.': "❌ Échec obtention image référence.",
    'Reference image ready': "✅ Image référence prête",
    'Aligning Batch': "📐 Alignement Lot",
    'aligned': "aligné",
    'alignment failed': "échec alignement",
    'Error in alignment worker': "❗️ Erreur worker alignement",
    'Auto batch size': "🧠 Taille lot auto",

    # Progression
    'progress': "Progression",
    'estimated_time': "Temps restant :",
    'elapsed_time': "Écoulé :",
    'calculating': "Calcul...",
    'Remaining:': "Restant :", # Key for static label
    'Additional:': "Additionnels :", # Key for static label

    # Boutons de contrôle
    'start': "Démarrer",
    'stop': "Arrêter",
    'add_folder_button': "Ajouter Dossier",
    'reset_zoom_button': "Reset Zoom",

    # Messages & Status
    'error': "Erreur", 'warning': "Avertissement", 'info': "Information", 'quit': "Quitter",
    'select_folders': "Veuillez sélectionner les dossiers d'entrée et de sortie.",
    'input_folder_invalid': "Dossier d'entrée invalide",
    'output_folder_invalid': "Dossier de sortie invalide/impossible à créer",
    'no_fits_found': "Aucun fichier .fit/.fits trouvé dans le dossier d'entrée.",
    'stacking_start': "⚙️ Début du traitement...",
    'stacking_stopping': "⚠️ Arrêt en cours...",
    'stacking_finished': "🏁 Traitement Terminé",
    'stacking_error_msg': "Erreur de Traitement :",
    'stacking_complete_msg': "Traitement terminé ! Stack final :",
    'stop_requested': "⚠️ Arrêt demandé, veuillez patienter...",
    'processing_stopped': "🛑 Traitement arrêté par l'utilisateur.",
    'processing_stopped_additional': "🛑 Traitement arrêté pendant les dossiers additionnels.",
    # 'processing_complete': "✅ Traitement terminé avec succès !", # Redundant
    'no_stacks_created': "⚠️ Aucun stack n'a été créé.",
    'stacks_created': "✅ Stacks créés.",
    'preview': "Aperçu",
    'no_current_stack': "Aucun Stack Actif",
    'image_info_waiting': "Info image : en attente...",
    'stretch_preview': "Étirer l'aperçu",
    'no_files_waiting': "Aucun fichier en attente",
    'no_additional_folders': "Aucun",
    # 'Main processing failed': "Échec traitement principal", # Less specific
    # 'Processing ended prematurely': "Traitement terminé prématurément", # Less specific
    'Error during cleanup': "Erreur lors du nettoyage",
    'Output folder created': "Dossier de sortie créé",
    'Error reading input folder': "Erreur lecture dossier entrée",
    'Start processing to add folders': "Le traitement doit être démarré pour ajouter des dossiers.",
    'Select Additional Images Folder': "Sélectionner dossier images additionnelles",
    'Folder not found': "Dossier non trouvé",
    'Input folder cannot be added': "Le dossier d'entrée principal ne peut pas être ajouté.",
    'Folder already added': "Ce dossier est déjà dans la liste.",
    'Folder contains no FITS': "Le dossier ne contient aucun fichier FITS.",
    'Error reading folder': "Erreur lecture dossier",
    'Folder added': "Dossier ajouté",
    'files': "fichiers",
    '1 additional folder': "1 dossier add.",
    '{count} additional folders': "{count} dossiers add.",
    'Select Input Folder': "Sélectionner dossier d'entrée",
    'Select Output Folder': "Sélectionner dossier de sortie",
    'Select Reference Image (Optional)': "Sélectionner image référence (Optionnel)",
    'quit_while_processing': "Traitement en cours. Quitter quand même ?",
    'Stack': "Stack", 'imgs': "im.", 'Object': "Objet", 'Date': "Date", 'Exposure (s)': "Expo (s)", 'Gain': "Gain",
    'Offset': "Offset", 'Temp (°C)': "Temp (°C)", 'Images': "Images", 'Method': "Méthode", 'Filter': "Filtre", 'Bayer': "Bayer",
    'No image info available': "Info image non disponible",
    # 'Main processing finished': "Traitement principal terminé", # Less specific
    # 'images in final stack': "images dans stack final", # Less specific
    # 'No batches were stacked': "Aucun lot empilé", # Less specific
    # 'Processing': "Traitement", 'additional folders': "dossiers additionnels", # Covered
    'Reference image not found for additional folders': "Image référence non trouvée pour dossiers add.",
    'Using reference': "Utilisation référence", 'Processing additional folder': "Traitement dossier add.",
    'No FITS files in': "Aucun fichier FITS dans", 'Skipped': "Ignoré", 'images found': "images trouvées",
    # 'Batch size': "Taille Lot", # Covered
    'Folder': "Dossier", 'Batch': "Lot", 'processed': "traité", 'images added': "images ajoutées",
    'Error processing folder': "Erreur traitement dossier",
    'Applying denoising to final stack': "Application débruitage au stack final",
    'Final denoised stack saved': "Stack final débruité sauvegardé",
    'Saving final stack with metadata': "Sauvegarde stack final avec métadonnées",
    'Final stack with metadata saved': "Stack final avec métadonnées sauvegardé",
    'Additional folder processing complete': "Traitement dossiers add. terminé",
    'Additional folder processing finished with errors': "Traitement dossiers add. terminé avec erreurs",
    # 'No files provided for stacking': "Aucun fichier fourni pour empilement", # Less specific
    'Error loading/validating': "Erreur chargement/validation",
    'No valid images to stack after loading': "Aucune image valide à empiler après chargement",
    'Unknown stacking method': "Méthode empilement inconnue",
    'using \'mean\'': "utilisation 'moyenne'",
    'Stacking failed (result is None)': "Échec empilement (résultat None)",
    'Dimension mismatch during combine': "Dimensions incompatibles lors combinaison",
    'Skipping combination': "Combinaison ignorée",
    'Error combining stacks': "Erreur combinaison stacks",
    'Cleaning temporary files': "Nettoyage fichiers temporaires",
    'Cannot delete': "Impossible de supprimer",
    'Error scanning folder': "Erreur parcours dossier",
    'temporary files removed': "fichiers temporaires supprimés",
    'Error reading metadata from': "Erreur lecture métadonnées de",
    'Failed to save final metadata stack': "Échec sauvegarde stack final métadonnées",
    'Error saving final FITS stack': "Erreur sauvegarde stack FITS final",
    'Denoising failed': "Échec débruitage",
    'Error displaying reference image': "Erreur affichage image référence",
    'Error saving cumulative stack': "Erreur sauvegarde stack cumulatif",
    'Could not find reference for final metadata stack': "Référence non trouvée pour stack final métadonnées",
    'Error preparing final metadata stack': "Erreur préparation stack final métadonnées",
    'Welcome!': "Bienvenue !",
    'Select input/output folders.': "Sélectionnez les dossiers d'entrée/sortie.",
    'Preview:': "Aperçu :",
    'No Image Data': "Aucune Donnée Image", 'Preview Error': "Erreur Aperçu", 'Preview Update Error': "Erreur MàJ Aperçu",
    'No FITS files in input': "Aucun fichier FITS en entrée",
    # 'Alignment failed for batch': "Échec alignement lot", # Less specific
    # 'Main processing OK, but errors in additional folders': "Traitement principal OK, mais erreurs dossiers add.", # Less specific
    # 'Critical error in main processing': "Erreur critique traitement principal", # Less specific
    'Error loading preview image': "Erreur chargement image aperçu",
    'Error loading preview (invalid format)': "Erreur chargement aperçu (format invalide)",
    'Error during debayering': "Erreur pendant debayering",
    'Invalid or missing BAYERPAT': "BAYERPAT invalide ou manquant",
    'Treating as grayscale': "Traitement comme niveaux de gris",
    'Error loading final stack preview': "Erreur chargement aperçu stack final",

    # Pixels chauds
    'hot_pixels_correction': 'Correction Pixels Chauds',
    'perform_hot_pixels_correction': 'Corriger pixels chauds',
    'hot_pixel_threshold': 'Seuil :',
    'neighborhood_size': 'Voisinage :',
    'neighborhood_size_adjusted': "Taille voisinage ajustée à {size} (doit être impaire)",

    # Mode de traitement
    'Final Stack': 'Stack Final', # LabelFrame key

    # Aligned files counter
    'aligned_files_label': "Alignés:", # Static label text
    'aligned_files_label_format': "Alignés : {count}", # Format string for display

    # Total Exposure Time <-- ADDED Key
    'Total Exp (s)': "Expo Totale (s)",
}