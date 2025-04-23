# --- START OF FILE seestar/localization/localization.py ---
"""
Module de localisation pour l'interface utilisateur de Seestar.
"""
from .fr import FR_TRANSLATIONS
from .en import EN_TRANSLATIONS

class Localization:
    """
    Gère les traductions pour l'interface Seestar Stacker.
    """

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
        # Ensure language is valid, default to 'en'
        self.language = language if language in self.translations else 'en'

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
            print(f"Warning: Language '{language}' not supported, falling back to 'en'.")
            self.language = 'en'

    # --- CORRECTED get METHOD ---
    def get(self, key, default=None):
        """
        Obtient la traduction pour une clé donnée.

        Args:
            key (str): Clé de traduction.
            default (str, optional): Valeur à retourner si la clé n'est pas trouvée.
                                      Si None, la clé elle-même est retournée.

        Returns:
            str: Texte traduit, ou la valeur par défaut/clé si non trouvée.
        """
        # Try the current language first
        translation = self.translations[self.language].get(key)

        if translation is not None:
            return translation
        else:
            # Fallback to English if key not in current language
            fallback_translation = self.translations['en'].get(key)
            if fallback_translation is not None:
                # Optionally print a warning about missing translation
                # print(f"Warning: Missing translation for key '{key}' in language '{self.language}'. Using English fallback.")
                return fallback_translation
            else:
                # Return the provided default value if specified, otherwise the key itself
                # This is the crucial change: using the 'default' argument passed in.
                return default if default is not None else key
# --- END OF FILE seestar/localization/localization.py ---