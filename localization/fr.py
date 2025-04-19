"""
Fichier de traductions françaises pour Seestar.
"""

FR_TRANSLATIONS = {
    # Interface principale
    'title': "Seestar Stacker",
    'input_folder': "Dossier d'entrée:",
    'output_folder': "Dossier de sortie:",
    'browse': "Parcourir",
    
    # Options
    'options': "Options",
    'stacking_method': "Méthode d'empilement:",
    'kappa_value': "Valeur de Kappa:",
    'batch_size': "Taille du lot (0 pour auto):",
    'remove_aligned': "Supprimer les images alignées après empilement",
    'apply_denoise': "Appliquer le débruitage au stack final",
    
    # Méthodes d'empilement
    'mean': "moyenne",
    'median': "médiane",
    'kappa-sigma': "kappa-sigma",
    'winsorized-sigma': "winsorized-sigma",
    
    # Options de traitement
    'processing_options': "Options de traitement",
    'progressive_stacking_start': "⚙️ Démarrage de l'empilement progressif...",
    
    # Alignement
    'alignment': "Alignement",
    'perform_alignment': "Effectuer l'alignement avant l'empilement",
    'reference_image': "Image de référence (laissez vide pour sélection automatique):",
    'alignment_start': "⚙️ Début de l'alignement des images...",
    'using_aligned_folder': "✅ Utilisation du dossier aligné : {}",
    
    # Progression
    'progress': "Progression",
    'estimated_time': "Temps restant estimé:",
    'elapsed_time': "Temps écoulé:",
    'calculating': "Calcul en cours...",
    
    # Boutons de contrôle
    'start': "Démarrer",
    'stop': "Arrêter",
    
    # Messages
    'error': "Erreur",
    'select_folders': "Veuillez sélectionner les dossiers d'entrée et de sortie.",
    'stacking_start': "⚙️ Début de l'empilement des images...",
    'stop_requested': "⚠️ Arrêt demandé, patientez...",
    'processing_stopped': "Traitement arrêté par l'utilisateur",
    'processing_completed': "Traitement terminé avec succès!",
    'no_stacks_created': "Aucun stack n'a été créé",
    'stacks_created': "stacks ont été créés",
    'neighborhood_size_adjusted': "La taille du voisinage a été ajustée à",
    
    # Hot pixels
    'hot_pixels_correction': 'Correction des pixels chauds',
    'perform_hot_pixels_correction': 'Appliquer la correction des pixels chauds',
    'hot_pixel_threshold': 'Seuil de détection:',
    'neighborhood_size': 'Taille du voisinage:',
    
    # Empilement incrémental
    'incremental_stacking': 'Empilement incrémental',
    'use_incremental_stacking': 'Utiliser l\'empilement incrémental (économise l\'espace disque)',
    'keep_aligned_images': 'Conserver les images alignées',
    'keep_intermediate_stacks': 'Conserver les stacks intermédiaires',
    'incremental_stacking_start': '⚙️ Démarrage de l\'empilement incrémental...',
    'real_time_preview': 'Prévisualisation en temps réel',
    'enable_preview': 'Activer la prévisualisation en temps réel',
    'use_traditional_mode': 'Utiliser l\'empilement traditionnel',
    'remove_processed': 'Supprimer les images traitées',
    
    # Mode de traitement
    'processing_mode': 'Mode de traitement',
    'use_traditional_mode': 'Utiliser l\'empilement traditionnel',
    'remove_processed': 'Supprimer les images traitées',
}