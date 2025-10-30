# zemosaic_localization.py
"""
╔══════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                  ║
║                                                                      ║
║ Auteur  : Tinystork, seigneur des couteaux à beurre (aka Tristan Nauleau)  
║ Partenaire : J.A.R.V.I.S. (/'dʒɑːrvɪs/) — Just a Rather Very Intelligent System  
║              (aka ChatGPT, Grand Maître du ciselage de code)         ║
║                                                                      ║
║ Licence : GNU General Public License v3.0 (GPL-3.0)                  ║
║                                                                      ║
║ Description :                                                        ║
║   Ce programme a été forgé à la lueur des pixels et de la caféine,   ║
║   dans le but noble de transformer des nuages de photons en art      ║
║   astronomique. Si vous l’utilisez, pensez à dire “merci”,           ║
║   à lever les yeux vers le ciel, ou à citer Tinystork et J.A.R.V.I.S.║
║   (le karma des développeurs en dépend).                             ║
║                                                                      ║
║ Avertissement :                                                      ║
║   Aucune IA ni aucun couteau à beurre n’a été blessé durant le       ║
║   développement de ce code.                                          ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║ ZeMosaic / ZeSeestarStacker Project                                  ║
║                                                                      ║
║ Author  : Tinystork, Lord of the Butter Knives (aka Tristan Nauleau) ║
║ Partner : J.A.R.V.I.S. (/'dʒɑːrvɪs/) — Just a Rather Very Intelligent System  
║           (aka ChatGPT, Grand Master of Code Chiseling)              ║
║                                                                      ║
║ License : GNU General Public License v3.0 (GPL-3.0)                  ║
║                                                                      ║
║ Description:                                                         ║
║   This program was forged under the sacred light of pixels and       ║
║   caffeine, with the noble intent of turning clouds of photons into  ║
║   astronomical art. If you use it, please consider saying “thanks,”  ║
║   gazing at the stars, or crediting Tinystork and J.A.R.V.I.S. —     ║
║   developer karma depends on it.                                     ║
║                                                                      ║
║ Disclaimer:                                                          ║
║   No AIs or butter knives were harmed in the making of this code.    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import json
import os
import traceback # Gardé pour un log d'erreur plus détaillé si besoin (mais pas utilisé activement)

class ZeMosaicLocalization:
    def __init__(self, language_code='en'):
        """
        Initialise le gestionnaire de localisation.
        Suppose que les fichiers .json de langue sont dans le même dossier que ce module.

        Args:
            language_code (str): Code de la langue à charger (ex: 'en', 'fr').
        """
        # print(f"DEBUG (Localization __init__): Initialisation de ZeMosaicLocalization...")
        try:
            current_module_path = os.path.abspath(__file__)
            self.locales_dir_abs_path = os.path.dirname(current_module_path)
            # print(f"DEBUG (Localization __init__): Dossier des locales déterminé: {self.locales_dir_abs_path}")
        except NameError:
            self.locales_dir_abs_path = os.getcwd()
            print(f"AVERT (Localization __init__): __file__ non défini. Dossier des locales basé sur CWD: {self.locales_dir_abs_path}")

        self.language_code = None
        self.translations = {}
        self.fallback_translations = {}

        if not os.path.isdir(self.locales_dir_abs_path):
             print(f"ERREUR CRITIQUE (Localization __init__): Dossier des locales '{self.locales_dir_abs_path}' non trouvé ou n'est pas un dossier!")
        else:
            # print("DEBUG (Localization __init__): Tentative de chargement du fallback anglais ('en').")
            self._load_language_file('en', is_fallback=True)
            # print(f"DEBUG (Localization __init__): Statut du fallback anglais après chargement: {bool(self.fallback_translations)} ({len(self.fallback_translations)} clés)")
            
            # print(f"DEBUG (Localization __init__): Tentative de chargement de la langue initiale demandée: '{language_code}'.")
            self.set_language(language_code)
        # print("DEBUG (Localization __init__): Initialisation de ZeMosaicLocalization TERMINÉE.")

    def _load_language_file(self, lang_code_to_load, is_fallback=False):
        """Charge un fichier de langue spécifique."""
        # target_dict_name = "fallback_translations" if is_fallback else "translations"
        # print(f"DEBUG (Localization _load_language_file): Début chargement pour '{lang_code_to_load}' (is_fallback={is_fallback}) -> vers self.{target_dict_name}")
        
        file_path_to_load = os.path.join(self.locales_dir_abs_path, f"{lang_code_to_load}.json")
        # print(f"  Chemin du fichier JSON: {file_path_to_load}")
        
        temp_translations_loaded = {}
        loaded_successfully = False

        if not os.path.isfile(file_path_to_load):
            print(f"AVERT (Localization _load_language_file): Fichier de langue '{file_path_to_load}' NON TROUVÉ.")
            if is_fallback and lang_code_to_load == 'en':
                 print(f"  -> ERREUR CRITIQUE: Fallback anglais ('en.json') non trouvé. La traduction sera très limitée.")
            if is_fallback: self.fallback_translations = {}
            else: self.translations = {}
            return False

        if os.path.getsize(file_path_to_load) == 0:
            print(f"AVERT (Localization _load_language_file): Fichier de langue '{file_path_to_load}' est VIDE.")
            if is_fallback: self.fallback_translations = {}
            else: self.translations = {}
            return False

        try:
            # Accept both UTF-8 with and without BOM to avoid JSON decode errors
            with open(file_path_to_load, 'r', encoding='utf-8-sig') as f:
                temp_translations_loaded = json.load(f)
            print(f"INFO (Localization _load_language_file): Traductions pour '{lang_code_to_load}' chargées ({len(temp_translations_loaded)} clés) depuis {file_path_to_load}")
            loaded_successfully = True
        except json.JSONDecodeError as e_json:
            print(f"ERREUR (Localization _load_language_file): Erreur de décodage JSON dans '{file_path_to_load}': {e_json}")
        except Exception as e_load:
            print(f"ERREUR (Localization _load_language_file): Erreur inattendue lors du chargement de '{lang_code_to_load}' depuis '{file_path_to_load}': {e_load}")
            # traceback.print_exc(limit=2) # Décommenter pour un traceback plus complet en cas d'erreur imprévue

        if loaded_successfully:
            if is_fallback:
                self.fallback_translations = temp_translations_loaded
                # print(f"DEBUG (Localization _load_language_file): self.fallback_translations mis à jour ({len(self.fallback_translations)} clés).")
            else:
                self.translations = temp_translations_loaded
                # print(f"DEBUG (Localization _load_language_file): self.translations mis à jour ({len(self.translations)} clés).")
        else:
            if is_fallback and lang_code_to_load == 'en':
                self.fallback_translations = {}
                # print(f"DEBUG (Localization _load_language_file): self.fallback_translations (anglais) vidé suite à échec chargement.")
        
        # print(f"DEBUG (Localization _load_language_file): Fin chargement pour '{lang_code_to_load}'. Succès: {loaded_successfully}")
        return loaded_successfully

    def set_language(self, language_code):
        """Change la langue active."""
        print(f"INFO (Localization set_language): Demande de changement de langue vers '{language_code}'. Langue actuelle: '{self.language_code}'.")
        
        if not isinstance(language_code, str) or not language_code:
            print(f"AVERT (Localization set_language): Code de langue invalide '{language_code}'. Maintien de la langue actuelle ('{self.language_code}').")
            return

        if self.language_code == language_code and self.translations:
            if language_code == 'en' and self.translations is self.fallback_translations:
                 # print(f"DEBUG (Localization set_language): Langue '{language_code}' (fallback anglais) déjà active et utilisée. Pas de rechargement.")
                 return
            elif language_code != 'en':
                 # print(f"DEBUG (Localization set_language): Langue '{language_code}' déjà active et traductions non-fallback chargées. Pas de rechargement.")
                 return

        if self._load_language_file(language_code, is_fallback=False):
            self.language_code = language_code
            print(f"INFO (Localization set_language): Langue active mise à '{self.language_code}' (traductions principales chargées).")
        else:
            print(f"AVERT (Localization set_language): Échec du chargement de '{language_code}' comme langue principale.")
            if self.fallback_translations:
                print(f"  -> Utilisation du fallback anglais car '{language_code}' n'a pas pu être chargé.")
                self.translations = self.fallback_translations
                self.language_code = 'en'
                print(f"INFO (Localization set_language): Langue active mise à 'en' (utilisation du fallback).")
            else:
                print(f"  -> ERREUR CRITIQUE: Ni '{language_code}' ni le fallback anglais n'ont pu être chargés. Dictionnaire de traduction vide.")
                self.translations = {}
                self.language_code = language_code 
                print(f"AVERT (Localization set_language): Langue '{self.language_code}' demandée, mais AUCUNE traduction disponible.")
        
        # print(f"DEBUG (Localization set_language): Contenu self.translations ({len(self.translations)} clés)")
        # print(f"DEBUG (Localization set_language): Contenu self.fallback_translations ({len(self.fallback_translations)} clés)")

    def get(self, key, default_text=None, **kwargs):
        """
        Récupère une chaîne traduite.
        """
        # print(f"DEBUG (Localization get): Demande clé '{key}', langue active '{self.language_code}'")
        if not isinstance(key, str):
            # print(f"AVERT (Localization get): Type de clé invalide reçu: {type(key)} (valeur: {key})")
            return str(default_text) if default_text is not None else "_INVALID_KEY_TYPE_"

        current_text_to_use = None

        if self.translations:
            current_text_to_use = self.translations.get(key)

        if current_text_to_use is None and self.language_code != 'en' and self.fallback_translations:
            current_text_to_use = self.fallback_translations.get(key)
            # if current_text_to_use is not None:
                # print(f"DEBUG (Localization get): Clé '{key}' trouvée dans self.fallback_translations (anglais).")

        if current_text_to_use is None:
            if default_text is not None:
                current_text_to_use = default_text
            else:
                current_text_to_use = f"_{key}_" # Placeholder pour clé manquante

        if kwargs and isinstance(current_text_to_use, str):
            try:
                final_text = current_text_to_use.format(**kwargs)
            except KeyError as e_fmt:
                print(f"AVERT (Localization get): Clé de formatage manquante '{e_fmt}' pour clé principale '{key}' "
                      f"(langue: '{self.language_code}'). Texte: '{current_text_to_use}', Args: {kwargs}")
                final_text = f"{current_text_to_use} [FORMAT_ARGS_MISSING: {kwargs}]"
            except Exception as e_gen_fmt:
                 print(f"AVERT (Localization get): Erreur de formatage générique pour clé '{key}' "
                       f"(langue: '{self.language_code}'): {e_gen_fmt}. Texte: '{current_text_to_use}'")
                 final_text = current_text_to_use
        elif not isinstance(current_text_to_use, str):
            # print(f"AVERT (Localization get): Le texte final pour la clé '{key}' n'est pas une chaîne (type: {type(current_text_to_use)}). Valeur: {current_text_to_use}")
            final_text = f"_{key}_ [TYPE_ERREUR: {type(current_text_to_use)}]"
        else:
            final_text = current_text_to_use
            
        # print(f"DEBUG (Localization get): Clé '{key}' -> Retourne: '{final_text}'")
        return final_text
