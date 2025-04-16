"""
Module de localisation pour l'interface utilisateur de Seestar.
"""
from .fr import FR_TRANSLATIONS
from .en import EN_TRANSLATIONS

class Localization:
    """
    Gère les traductions pour l'interface Seestar Stacker.
    """
    
    # Dictionnaire de traductions
    translations = {
        'fr': FR_TRANSLATIONS,
        'en': EN_TRANSLATIONS
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