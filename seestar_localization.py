# seestar_localization.py
"""
Module de localisation pour Seestar Stacker GUI.
Contient les traductions des chaînes de texte pour différentes langues.
"""
# seestar_localization.py
"""
Module de localisation pour Seestar Stacker GUI.
Contient les traductions des chaînes de texte pour différentes langues.
"""

class Localization:
    """
    Gère les traductions pour l'interface Seestar Stacker.
    """
    
    # Dictionnaire de traductions
    translations = {
        'fr': {
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
            
            # Méthodes d'empilement
            'mean': "moyenne",
            'median': "médiane",
            'kappa-sigma': "kappa-sigma",
            'winsorized-sigma': "winsorized-sigma",
            
            # Alignement
            'alignment': "Alignement",
            'perform_alignment': "Effectuer l'alignement avant l'empilement",
            'reference_image': "Image de référence (laissez vide pour sélection automatique) :",
            
            # Progression
            'progress': "Progression",
            'estimated_time': "Temps restant estimé:",
            'elapsed_time': "Temps écoulé:",
            
            # Boutons de contrôle
            'start': "Démarrer",
            'stop': "Arrêter",
            
            # Messages
            'error': "Erreur",
            'select_folders': "Veuillez sélectionner les dossiers d'entrée et de sortie.",
            'alignment_start': "⚙️ Début de l'alignement des images...",
            'using_aligned_folder': "✅ Utilisation du dossier aligné : {}",
            'stacking_start': "⚙️ Début de l'empilement des images...",
            'stop_requested': "⚠️ Arrêt demandé, patientez...",
            'calculating': "Calcul en cours...",
            
            # Hot pixels
            'hot_pixels_correction': 'Correction des pixels chauds',
            'perform_hot_pixels_correction': 'Appliquer la correction des pixels chauds',
            'hot_pixel_threshold': 'Seuil de détection:',
            'neighborhood_size': 'Taille du voisinage:'
        },
        
        'en': {
            # Main interface
            'title': "Seestar Stacker",
            'input_folder': "Input folder:",
            'output_folder': "Output folder:",
            'browse': "Browse",
            
            # Options
            'options': "Options",
            'stacking_method': "Stacking method:",
            'kappa_value': "Kappa value:",
            'batch_size': "Batch size (0 for auto):",
            
            # Stacking methods
            'mean': "mean",
            'median': "median",
            'kappa-sigma': "kappa-sigma",
            'winsorized-sigma': "winsorized-sigma",
            
            # Alignment
            'alignment': "Alignment",
            'perform_alignment': "Perform alignment before stacking",
            'reference_image': "Reference image (leave empty for automatic selection):",
            
            # Progress
            'progress': "Progress",
            'estimated_time': "Estimated time remaining:",
            'elapsed_time': "Elapsed time:",
            
            # Control buttons
            'start': "Start",
            'stop': "Stop",
            
            # Messages
            'error': "Error",
            'select_folders': "Please select input and output folders.",
            'alignment_start': "⚙️ Starting image alignment...",
            'using_aligned_folder': "✅ Using aligned folder: {}",
            'stacking_start': "⚙️ Starting image stacking...",
            'stop_requested': "⚠️ Stop requested, please wait...",
            'calculating': "Calculating...",
            
            # Hot pixels
            'hot_pixels_correction': 'Hot Pixels Correction',
            'perform_hot_pixels_correction': 'Perform hot pixels correction',
            'hot_pixel_threshold': 'Detection threshold:',
            'neighborhood_size': 'Neighborhood size:'
        }
    }
    
    def __init__(self, language='en'):
        """
        Initialise le système de localisation avec la langue spécifiée.
        
        Args:
            language (str): Code de langue ('en', 'fr', etc.)
        """
        self.set_language(language)
    
    def set_language(self, language):
        """
        Change la langue courante.
        
        Args:
            language (str): Code de langue ('en', 'fr', etc.)
        """
        if language in self.translations:
            self.language = language
        else:
            # Fallback to English if language not available
            self.language = 'en'
    
    def get(self, key):
        """
        Obtient la traduction pour une clé donnée.
        
        Args:
            key (str): Clé de traduction
            
        Returns:
            str: Texte traduit, ou la clé elle-même si non trouvée
        """
        if key in self.translations[self.language]:
            return self.translations[self.language][key]
        # Fallback à l'anglais si la clé n'existe pas dans la langue actuelle
        elif key in self.translations['en']:
            return self.translations['en'][key]
        # Retourner la clé elle-même si non trouvée
        return key
class Localization:
    """
    Gère les traductions pour l'interface Seestar Stacker.
    """
    
    # Dictionnaire de traductions
    translations = {
        'fr': {
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
            
            # Méthodes d'empilement
            'mean': "moyenne",
            'median': "médiane",
            'kappa-sigma': "kappa-sigma",
            'winsorized-sigma': "winsorized-sigma",
            
            # Alignement
            'alignment': "Alignement",
            'perform_alignment': "Effectuer l'alignement avant l'empilement",
            'reference_image': "Image de référence (laissez vide pour sélection automatique) :",
            
            # Progression
            'progress': "Progression",
            'estimated_time': "Temps restant estimé:",
            'elapsed_time': "Temps écoulé:",
            
            # Boutons de contrôle
            'start': "Démarrer",
            'stop': "Arrêter",
            
            # Messages
            'error': "Erreur",
            'select_folders': "Veuillez sélectionner les dossiers d'entrée et de sortie.",
            'alignment_start': "⚙️ Début de l'alignement des images...",
            'using_aligned_folder': "✅ Utilisation du dossier aligné : {}",
            'stacking_start': "⚙️ Début de l'empilement des images...",
            'stop_requested': "⚠️ Arrêt demandé, patientez...",
            'calculating': "Calcul en cours...",
            
            # Hot pixels
            'hot_pixels_correction': 'Correction des pixels chauds',
            'perform_hot_pixels_correction': 'Appliquer la correction des pixels chauds',
            'hot_pixel_threshold': 'Seuil de détection:',
            'neighborhood_size': 'Taille du voisinage:'
        },
        
        'en': {
            # Main interface
            'title': "Seestar Stacker",
            'input_folder': "Input folder:",
            'output_folder': "Output folder:",
            'browse': "Browse",
            
            # Options
            'options': "Options",
            'stacking_method': "Stacking method:",
            'kappa_value': "Kappa value:",
            'batch_size': "Batch size (0 for auto):",
            
            # Stacking methods
            'mean': "mean",
            'median': "median",
            'kappa-sigma': "kappa-sigma",
            'winsorized-sigma': "winsorized-sigma",
            
            # Alignment
            'alignment': "Alignment",
            'perform_alignment': "Perform alignment before stacking",
            'reference_image': "Reference image (leave empty for automatic selection):",
            
            # Progress
            'progress': "Progress",
            'estimated_time': "Estimated time remaining:",
            'elapsed_time': "Elapsed time:",
            
            # Control buttons
            'start': "Start",
            'stop': "Stop",
            
            # Messages
            'error': "Error",
            'select_folders': "Please select input and output folders.",
            'alignment_start': "⚙️ Starting image alignment...",
            'using_aligned_folder': "✅ Using aligned folder: {}",
            'stacking_start': "⚙️ Starting image stacking...",
            'stop_requested': "⚠️ Stop requested, please wait...",
            'calculating': "Calculating...",
            
            # Hot pixels
            'hot_pixels_correction': 'Hot Pixels Correction',
            'perform_hot_pixels_correction': 'Perform hot pixels correction',
            'hot_pixel_threshold': 'Detection threshold:',
            'neighborhood_size': 'Neighborhood size:'
        }
    }
    
    def __init__(self, language='en'):
        """
        Initialise le système de localisation avec la langue spécifiée.
        
        Args:
            language (str): Code de langue ('en', 'fr', etc.)
        """
        self.set_language(language)
    
    def set_language(self, language):
        """
        Change la langue courante.
        
        Args:
            language (str): Code de langue ('en', 'fr', etc.)
        """
        if language in self.translations:
            self.language = language
        else:
            # Fallback to English if language not available
            self.language = 'en'
    
    def get(self, key):
        """
        Obtient la traduction pour une clé donnée.
        
        Args:
            key (str): Clé de traduction
            
        Returns:
            str: Texte traduit, ou la clé elle-même si non trouvée
        """
        if key in self.translations[self.language]:
            return self.translations[self.language][key]
        # Fallback à l'anglais si la clé n'existe pas dans la langue actuelle
        elif key in self.translations['en']:
            return self.translations['en'][key]
        # Retourner la clé elle-même si non trouvée
        return key
