# --- START OF FILE seestar/alignment/astrometry_solver.py ---
"""
Module pour gérer l'interaction avec les solveurs astrométriques,
y compris Astrometry.net (web service), ASTAP (local), et ansvr (Astrometry.net local).
"""
import os
import numpy as np
import warnings
import time
import tempfile
import traceback
import subprocess # Pour appeler les solveurs locaux
import shutil # Pour trouver les exécutables
import gc
# --- Dépendances Astropy/Astroquery (comme avant) ---
_ASTROQUERY_AVAILABLE = False
_ASTROPY_AVAILABLE = False
AstrometryNet = None 

try:
    from astroquery.astrometry_net import AstrometryNet as ActualAstrometryNet
    AstrometryNet = ActualAstrometryNet
    _ASTROQUERY_AVAILABLE = True
    # print("DEBUG [AstrometrySolverModule]: astroquery.astrometry_net importé.") # Moins verbeux
except ImportError:
    print("WARNING [AstrometrySolverModule]: astroquery non installée. Plate-solving web Astrometry.net désactivé.")

try:
    from astropy.io import fits
    from astropy.wcs import WCS, FITSFixedWarning
    from astropy.utils.exceptions import AstropyWarning
    _ASTROPY_AVAILABLE = True
    warnings.filterwarnings('ignore', category=FITSFixedWarning)
    warnings.filterwarnings('ignore', category=AstropyWarning) # Pour d'autres avertissements astropy
    # print("DEBUG [AstrometrySolverModule]: astropy.io.fits et astropy.wcs importés.")
except ImportError:
     print("ERREUR CRITIQUE [AstrometrySolverModule]: Astropy non installée. Le module ne peut fonctionner.")


class AstrometrySolver:
    """
    Classe pour orchestrer la résolution astrométrique en utilisant différents solveurs.
    """
    def __init__(self, progress_callback=None):
        """
        Initialise le solveur.
        Args:
            progress_callback (callable, optional): Callback pour les messages de progression.
        """
        self.progress_callback = progress_callback
        if not _ASTROPY_AVAILABLE:
            self._log("ERREUR CRITIQUE: Astropy n'est pas disponible. AstrometrySolver ne peut fonctionner.", "ERROR")
            raise ImportError("Astropy est requis pour AstrometrySolver.")

    def _log(self, message, level="INFO"):
        """Helper interne pour la journalisation via le progress_callback ou print."""
        prefix_map = {
            "INFO": "   [AstrometrySolver]",
            "WARN": "   ⚠️ [AstrometrySolver WARN]",
            "ERROR": "   ❌ [AstrometrySolver ERROR]",
            "DEBUG": "      [AstrometrySolver DEBUG]"
        }
        prefix = prefix_map.get(level.upper(), prefix_map["INFO"])
        log_msg_with_prefix = f"{prefix} {message}"
        print(f"CONSOLE LOG FROM AstrometrySolver._log: {log_msg_with_prefix}") # <<< AJOUTER CE PRINT DIRECT
        
        if self.progress_callback and callable(self.progress_callback):
            try:
                # Le progress_callback du QueuedStacker attend (message, progress_value)
                # Pour les messages de ce module, on peut mettre progress_value à None.
                self.progress_callback(f"{prefix} {message}", None)
            except Exception:
                print(f"{prefix} {message}") # Fallback si le callback échoue
        else:
            print(f"{prefix} {message}")



#####################################################################################################################################


# DANS LA CLASSE AstrometrySolver DANS seestar/alignment/astrometry_solver.py


    def solve(self, image_path, fits_header, settings, update_header_with_solution=True):
        """
        Tente de résoudre le WCS d'une image en utilisant la stratégie configurée.
        Args:
            image_path (str): Chemin vers le fichier image à résoudre.
            fits_header (fits.Header): Header FITS de l'image. Sera mis à jour si update_header_with_solution est True
                                      ET si une solution est trouvée.
            settings (dict): Dictionnaire contenant la configuration des solveurs.
                             Clés attendues: 'use_local_solver_priority' (bool),
                                           'astap_path' (str), 'astap_data_dir' (str),
                                           'local_ansvr_path' (str), 'api_key' (str),
                                           'scale_est_arcsec_per_pix' (float, optional),
                                           'scale_tolerance_percent' (float, optional),
                                           'ansvr_timeout_sec' (int, optional, default 120),
                                           'astap_timeout_sec' (int, optional, default 120),
                                           'astrometry_net_timeout_sec' (int, optional, default 300),
                                           'astap_search_radius' (float, optional, default 5.0) # <--- NOUVELLE clé attendue
            update_header_with_solution (bool): Si True, met à jour `fits_header` avec la solution.
        Returns:
            astropy.wcs.WCS or None: Objet WCS si succès, None si échec.
        """
        self._log(f"Début résolution pour: {os.path.basename(image_path)}", "INFO")
        wcs_solution = None

        use_local_priority = settings.get('use_local_solver_priority', False)
        api_key = settings.get('api_key', None)

        # Récupérer les estimations d'échelle si présentes
        scale_est = settings.get('scale_est_arcsec_per_pix', None)
        scale_tol = settings.get('scale_tolerance_percent', 20)
        # --- NOUVEAU: Lire le rayon de recherche ASTAP depuis les settings ---
        # La valeur par défaut ici (5.0) sera utilisée si la clé n'est pas dans le dict 'settings'.
        # La "vraie" valeur par défaut du système vient de SettingsManager.get_default_values().
        astap_search_radius_deg = settings.get('astap_search_radius', 5.0)
        self._log(f"  [AstrometrySolver.solve] Paramètre ASTAP Search Radius lu des settings: {astap_search_radius_deg}°", "DEBUG")
        # --- FIN NOUVEAU ---

        if use_local_priority:
            self._log("Priorité aux solveurs locaux activée.", "INFO")
            # 1. Essayer ASTAP
            astap_exe = settings.get('astap_path', "")

            self._log(f"ASTAP Check: astap_exe path from settings = '{astap_exe}'", "DEBUG")
            is_file_check_result = False
            if astap_exe:
                try:
                    is_file_check_result = os.path.isfile(astap_exe)
                    self._log(f"ASTAP Check: os.path.isfile('{astap_exe}') = {is_file_check_result}", "DEBUG")
                except Exception as e_isfile:
                    self._log(f"ASTAP Check: Erreur pendant os.path.isfile('{astap_exe}'): {e_isfile}", "ERROR")

            if astap_exe and os.path.isfile(astap_exe):
                current_scale_for_astap_call = settings.get('scale_est_arcsec_per_pix', None)
                self._log(f"ASTAP Call Prep: scale_est_arcsec_per_pix passé à _try_solve_astap sera: {current_scale_for_astap_call}", "DEBUG")
                self._log("Tentative avec ASTAP...", "INFO")
                astap_data = settings.get('astap_data_dir', None)
                astap_timeout = settings.get('astap_timeout_sec', 120)

                # --- MODIFIÉ: Passer astap_search_radius_deg à _try_solve_astap ---
                # La signature de _try_solve_astap devra être modifiée pour accepter ce nouvel argument.
                # Pour l'instant, si tu remplaces juste solve(), le code plantera ici car
                # _try_solve_astap ne connaît pas encore 'search_radius_deg'.
                # C'est pourquoi on fait une méthode à la fois.
                # Pour que CE code soit testable immédiatement SANS modifier _try_solve_astap,
                # on pourrait temporairement ne pas passer le nouvel argument.
                # Mais pour avancer, on assume qu'on modifiera _try_solve_astap ensuite.
                wcs_solution = self._try_solve_astap(image_path, fits_header, astap_exe, astap_data,
                                                     scale_est, scale_tol, astap_timeout,
                                                     update_header_with_solution,
                                                     search_radius_deg=astap_search_radius_deg) # <--- MODIFICATION ICI
                # --- FIN MODIFICATION ---
                if wcs_solution:
                    self._log("Solution trouvée avec ASTAP.", "INFO")
                    return wcs_solution
            elif astap_exe:
                 self._log(f"Chemin ASTAP '{astap_exe}' configuré mais non trouvé ou n'est pas un fichier. ASTAP ignoré.", "WARN")


            # 2. Essayer Astrometry.net local (ansvr / solve-field)
            if not wcs_solution:
                ansvr_config_path = settings.get('local_ansvr_path', "")
                if ansvr_config_path:
                    self._log("Tentative avec Astrometry.net local (ansvr/solve-field)...", "INFO")
                    ansvr_timeout = settings.get('ansvr_timeout_sec', 120)
                    wcs_solution = self._try_solve_local_ansvr(image_path, fits_header, ansvr_config_path,
                                                               scale_est, scale_tol, ansvr_timeout,
                                                               update_header_with_solution)
                    if wcs_solution:
                        self._log("Solution trouvée avec Astrometry.net local.", "INFO")
                        return wcs_solution

        # 3. Essayer Astrometry.net web (si pas de priorité locale, ou si les locaux ont échoué)
        if not wcs_solution:
            if api_key:
                self._log("Tentative avec Astrometry.net (web service)...", "INFO")
                anet_timeout = settings.get('astrometry_net_timeout_sec', 300)
                wcs_solution = self._solve_astrometry_net_web(
                    image_path_for_solver=image_path,
                    fits_header_original=fits_header,
                    api_key=api_key,
                    scale_est_arcsec_per_pix=scale_est,
                    scale_tolerance_percent=scale_tol,
                    timeout_sec=anet_timeout,
                    update_header_with_solution=update_header_with_solution
                )
                if wcs_solution:
                    self._log("Solution trouvée avec Astrometry.net (web service).", "INFO")
                    return wcs_solution
            else:
                self._log("Clé API pour Astrometry.net (web) non fournie. Solveur web ignoré.", "INFO")

        if not wcs_solution:
            self._log(f"Aucune solution astrométrique trouvée pour {os.path.basename(image_path)} après toutes tentatives.", "WARN")

        return None


#############################################################################################################################################


# DANS LA CLASSE AstrometrySolver DANS seestar/alignment/astrometry_solver.py

    # --- MODIFIER CETTE MÉTHODE ---
    def _try_solve_astap(self, image_path, fits_header, astap_exe_path, astap_data_dir,
                         scale_est_arcsec_per_pix,
                         scale_tolerance_percent,
                         timeout_sec,
                         update_header_with_solution,
                         search_radius_deg=None): # <--- NOUVEAU paramètre avec valeur par défaut
        """
        Tente de résoudre l'image en utilisant ASTAP en ligne de commande.
        Si une 'scale_est_arcsec_per_pix' valide est fournie, utilise l'option '-s <échelle_arrondie>'.
        Sinon, utilise l'option '-fov 0' pour laisser ASTAP auto-déterminer le champ de vue.
        Ajoute toujours '-sens 100' et '-log'.
        Ajoute -r <search_radius_deg> si fourni et valide. <--- NOUVEAU dans docstring
        Retourne un objet WCS Astropy si succès, sinon None.
        """
        base_image_filename = os.path.basename(image_path)
        self._log(f"ASTAP: Début tentative pour '{base_image_filename}'...", "INFO")
        # --- MODIFIÉ: Log pour inclure search_radius_deg ---
        self._log(f"  ASTAP Params Reçus: scale_est='{scale_est_arcsec_per_pix}' (type: {type(scale_est_arcsec_per_pix)}), "
                  f"scale_tol%='{scale_tolerance_percent}', timeout='{timeout_sec}s', "
                  f"search_radius_deg='{search_radius_deg}'", "DEBUG")
        # --- FIN MODIFICATION ---

        if not os.path.isfile(image_path):
            self._log(f"ASTAP: Fichier image source '{image_path}' NON TROUVÉ. Abandon.", "ERROR")
            return None

        if not astap_exe_path or not os.path.isfile(astap_exe_path):
            self._log(f"ASTAP: Exécutable ASTAP '{astap_exe_path}' NON VALIDE ou NON TROUVÉ. Abandon.", "ERROR")
            return None

        image_dir = os.path.dirname(image_path)
        base_name_for_output_files = os.path.splitext(base_image_filename)[0]
        expected_wcs_file = os.path.join(image_dir, base_name_for_output_files + ".wcs")
        expected_ini_file = os.path.join(image_dir, base_name_for_output_files + ".ini")
        astap_log_file_path = os.path.join(image_dir, base_name_for_output_files + ".log")

        self._log(f"  ASTAP: Nettoyage des anciens fichiers de sortie potentiels dans '{image_dir}' pour base '{base_name_for_output_files}'...", "DEBUG")
        for f_to_clean in [expected_wcs_file, expected_ini_file, astap_log_file_path]:
            if os.path.exists(f_to_clean):
                try:
                    os.remove(f_to_clean)
                    self._log(f"    ASTAP: Ancien fichier '{os.path.basename(f_to_clean)}' supprimé.", "DEBUG")
                except Exception as e_del:
                    self._log(f"    ASTAP: Avertissement - Impossible de supprimer ancien fichier '{os.path.basename(f_to_clean)}': {e_del}", "WARN")

        cmd = [astap_exe_path, "-f", image_path]

        if astap_data_dir and os.path.isdir(astap_data_dir):
            cmd.extend(["-d", astap_data_dir])
            self._log(f"  ASTAP: Utilisation du répertoire de données d'index ASTAP: '{astap_data_dir}'.", "DEBUG")
        else:
            if astap_data_dir:
                 self._log(f"  ASTAP WARN: Répertoire de données d'index ASTAP '{astap_data_dir}' non valide ou non trouvé. "
                           "ASTAP utilisera ses chemins par défaut/configurés.", "WARN")
            else:
                 self._log(f"  ASTAP: Aucun répertoire de données d'index ASTAP spécifique fourni. "
                           "ASTAP utilisera ses chemins par défaut/configurés.", "DEBUG")

        # --- NOUVEAU: Ajout de l'option -r (rayon de recherche) si fournie et valide ---
        if search_radius_deg is not None:
            try:
                radius_float = float(search_radius_deg)
                # ASTAP s'attend à un rayon positif. La limite supérieure est grande (ex: 90 pour blind).
                # La documentation d'ASTAP mentionne "The radius of the square search pattern".
                if 0.01 <= radius_float <= 180.0: # Plage large, 0.01 pour éviter 0, 180 pour tout le ciel
                    # ASTAP prend des float pour -r. Formatons à 1 ou 2 décimales.
                    cmd.extend(["-r", f"{radius_float:.2f}"])
                    self._log(f"  ASTAP: Option '-r {radius_float:.2f}' (rayon de recherche en degrés) ajoutée.", "INFO")
                else:
                    self._log(f"  ASTAP WARN: Rayon de recherche '{radius_float}' hors limites valides (ex: 0.01-180). Option -r ignorée.", "WARN")
            except (ValueError, TypeError):
                self._log(f"  ASTAP WARN: Valeur de rayon de recherche '{search_radius_deg}' invalide. Option -r ignorée.", "WARN")
        else:
            # Si search_radius_deg est None, ASTAP utilisera son propre rayon par défaut (qui peut être grand, >50°).
            self._log(f"  ASTAP: Aucun rayon de recherche ASTAP spécifique (-r) fourni. ASTAP utilisera sa valeur/logique par défaut.", "DEBUG")
        # --- FIN NOUVEAU ---

        # --- Logique pour choisir entre l'option -s et -fov 0 (inchangée par rapport à ta dernière version) ---
        use_s_option = False
        scale_to_use_for_s_option = None
        if scale_est_arcsec_per_pix is not None:
            if isinstance(scale_est_arcsec_per_pix, (int, float, np.number)):
                try:
                    scale_float_val = float(scale_est_arcsec_per_pix)
                    if scale_float_val > 1e-3:
                        use_s_option = True
                        scale_to_use_for_s_option = scale_float_val
                        self._log(f"  ASTAP: Estimation d'échelle valide ({scale_to_use_for_s_option:.3f} arcsec/pix) fournie.", "DEBUG")
                    else:
                        self._log(f"  ASTAP: Estimation d'échelle fournie ({scale_float_val:.3f}) trop petite ou non positive. Ignorée.", "DEBUG")
                except (ValueError, TypeError) as e_scale_conv:
                    self._log(f"  ASTAP: Erreur conversion estimation d'échelle '{scale_est_arcsec_per_pix}' en float: {e_scale_conv}. Ignorée.", "WARN")
            else:
                self._log(f"  ASTAP: Type d'estimation d'échelle '{type(scale_est_arcsec_per_pix)}' non supporté. Ignorée.", "DEBUG")
        else:
            self._log(f"  ASTAP: Aucune estimation d'échelle fournie (scale_est_arcsec_per_pix is None).", "DEBUG")

        if use_s_option and scale_to_use_for_s_option is not None:
            astap_s_value_str = f"{scale_to_use_for_s_option:.1f}" # ASTAP attend 1 décimale pour -s
            cmd.extend(["-s", astap_s_value_str])
            self._log(f"  ASTAP: Option '-s {astap_s_value_str}' (échelle pixel) ajoutée à la commande.", "INFO")
            self._log(f"    ASTAP INFO: L'option '-s' est prioritaire sur l'analyse FOV interne d'ASTAP et sur "
                      f"les échelles FITS (comme CDELT, FOCALLEN). Son exactitude est cruciale.", "INFO")
        else:
            cmd.extend(["-fov", "0"])
            self._log(f"  ASTAP: Option '-fov 0' (auto-détection FOV) ajoutée. ASTAP tentera d'utiliser le header FITS ou de chercher.", "INFO")
        # --- Fin Logique -s / -fov ---

        astap_sensitivity_val = "100"
        cmd.extend(["-sens", astap_sensitivity_val])
        self._log(f"  ASTAP: Option '-sens {astap_sensitivity_val}' (sensibilité) ajoutée.", "INFO")

        cmd.append("-log")
        self._log(f"  ASTAP: Option '-log' (fichier log ASTAP) ajoutée.", "DEBUG")

        self._log(f"  ASTAP: Commande FINALE construite: {' '.join(cmd)}", "INFO")
        self._log(f"  ASTAP: Répertoire de travail pour subprocess: '{image_dir}'", "DEBUG")

        wcs_object_from_astap = None
        stdout_astap, stderr_astap = "", ""

        try:
            self._log(f"  ASTAP: Exécution pour '{base_image_filename}' avec timeout de {timeout_sec}s...", "INFO")
            process_result = subprocess.run(cmd,
                                            capture_output=True, text=True,
                                            timeout=timeout_sec, check=False,
                                            cwd=image_dir)

            stdout_astap = process_result.stdout.strip() if process_result.stdout else ""
            stderr_astap = process_result.stderr.strip() if process_result.stderr else ""

            self._log(f"  ASTAP: Code de retour: {process_result.returncode}", "DEBUG")
            if stdout_astap:
                self._log(f"  ASTAP stdout:\n------ ASTAP STDOUT START ------\n{stdout_astap}\n------ ASTAP STDOUT END ------", "DEBUG_DETAIL")
            if stderr_astap:
                self._log(f"  ASTAP stderr:\n------ ASTAP STDERR START ------\n{stderr_astap}\n------ ASTAP STDERR END ------", "WARN")

            if process_result.returncode == 0:
                if os.path.exists(expected_wcs_file) and os.path.getsize(expected_wcs_file) > 0:
                    self._log(f"  ASTAP: Résolution RÉUSSIE (code ASTAP 0). Fichier .wcs trouvé: '{expected_wcs_file}'", "INFO")
                    image_shape_hw_for_wcs = None
                    try:
                        with fits.open(image_path, memmap=False) as hdul_img_shape:
                            img_data_shape = hdul_img_shape[0].shape
                            if len(img_data_shape) >= 2:
                                image_shape_hw_for_wcs = img_data_shape[-2:]
                                self._log(f"    ASTAP: Shape image lue depuis '{base_image_filename}' pour WCS: {image_shape_hw_for_wcs}", "DEBUG")
                            else:
                                raise ValueError(f"Shape image invalide ({img_data_shape}) pour WCS.")
                    except Exception as e_shape:
                        self._log(f"    ASTAP WARN: Erreur lecture shape image depuis '{base_image_filename}' pour WCS: {e_shape}. "
                                  f"Utilisation fallback header NAXIS.", "WARN")
                        naxis2_fallback = fits_header.get('NAXIS2', 1080) if fits_header else 1080
                        naxis1_fallback = fits_header.get('NAXIS1', 1920) if fits_header else 1920
                        image_shape_hw_for_wcs = (int(naxis2_fallback), int(naxis1_fallback))
                        self._log(f"    ASTAP: Shape image (fallback header) pour WCS: {image_shape_hw_for_wcs}", "DEBUG")

                    wcs_object_from_astap = self._parse_wcs_file_content(expected_wcs_file, image_shape_hw_for_wcs)

                    if wcs_object_from_astap and wcs_object_from_astap.is_celestial:
                        self._log(f"  ASTAP: Objet WCS créé avec succès depuis '{os.path.basename(expected_wcs_file)}'.", "INFO")
                        if update_header_with_solution and fits_header is not None:
                            self._update_fits_header_with_wcs(fits_header, wcs_object_from_astap, solver_name="ASTAP")
                    else:
                        self._log(f"  ASTAP ERREUR: Échec création objet WCS valide ou céleste depuis '{os.path.basename(expected_wcs_file)}'.", "ERROR")
                        wcs_object_from_astap = None
                else:
                    self._log(f"  ASTAP ERREUR: Code ASTAP 0 mais fichier .wcs '{os.path.basename(expected_wcs_file)}' MANQUANT ou VIDE. "
                              "Solution considérée comme ÉCHOUÉE.", "ERROR")
                    wcs_object_from_astap = None
            else:
                error_message_from_astap = f"Résolution ÉCHOUÉE (code ASTAP: {process_result.returncode})"
                if process_result.returncode != 1 and os.path.exists(expected_ini_file):
                    try:
                        with open(expected_ini_file, 'r') as ini_f:
                            for line in ini_f:
                                if line.strip().startswith("ERROR="):
                                    error_detail = line.strip().split("=", 1)[1].strip()
                                    if error_detail:
                                        error_message_from_astap += f" - Détail INI: {error_detail}"
                                    break
                    except Exception as e_ini:
                         self._log(f"    ASTAP DEBUG: Impossible de lire le détail de l'erreur du fichier INI: {e_ini}", "DEBUG")

                self._log(f"  ASTAP WARN: {error_message_from_astap}", "WARN")
                if process_result.returncode == 2: # "Not enough stars detected"
                    self._log(f"    ASTAP INFO: Code 2 ('Not enough stars') peut indiquer un problème avec "
                              f"l'estimation d'échelle (si -s utilisée), la qualité de l'image, "
                              f"la position de départ (RA/Dec du header) et le rayon de recherche (-r), " # Message mis à jour
                              f"ou la sensibilité de détection (-sens).", "INFO")
                wcs_object_from_astap = None

        except subprocess.TimeoutExpired:
            self._log(f"  ASTAP ERREUR: Timeout ({timeout_sec}s) expiré pour '{base_image_filename}'.", "ERROR")
            wcs_object_from_astap = None
        except FileNotFoundError:
            self._log(f"  ASTAP ERREUR: Exécutable ASTAP '{astap_exe_path}' NON TROUVÉ par subprocess. "
                      "Vérifiez le chemin et les permissions.", "ERROR")
            wcs_object_from_astap = None
        except Exception as e:
            self._log(f"  ASTAP ERREUR: Exception inattendue pendant exécution ASTAP pour '{base_image_filename}': {e}", "ERROR")
            traceback.print_exc(limit=2)
            wcs_object_from_astap = None
        finally:
            self._log(f"  ASTAP: Fin tentative pour '{base_image_filename}'. Solution trouvée: {'Oui' if wcs_object_from_astap else 'Non'}", "DEBUG")

        return wcs_object_from_astap



########################################################################################################################################


    def _solve_astrometry_net_web(self, image_path_for_solver, fits_header_original, api_key,
                                  scale_est_arcsec_per_pix, scale_tolerance_percent, timeout_sec,
                                  update_header_with_solution):
        """
        Méthode interne pour gérer la résolution via le service web Astrometry.net.
        Basée sur la fonction globale solve_image_wcs précédente.
        Prend un CHEMIN de fichier FITS, le charge, le prépare et le soumet.
        """
        self._log(f"WebANET: Début tentative solving pour {os.path.basename(image_path_for_solver)}", "DEBUG")

        if not _ASTROQUERY_AVAILABLE or not _ASTROPY_AVAILABLE:
            self._log("Dépendances manquantes (astroquery ou astropy) pour Astrometry.net web.", "ERROR")
            return None
        if not os.path.isfile(image_path_for_solver):
            self._log(f"Fichier image source '{image_path_for_solver}' non trouvé pour Astrometry.net web.", "ERROR")
            return None
        if not api_key:
            self._log("Clé API Astrometry.net manquante pour service web.", "ERROR")
            return None

        ast = AstrometryNet()
        ast.api_key = api_key
        # --- CONFIGURER LE TIMEOUT SUR L'INSTANCE ASTROMETRYNET ---
        if timeout_sec is not None and timeout_sec > 0:
            try:
                # La manière de configurer le timeout peut dépendre de la version d'astroquery.
                # Essayons d'abord l'attribut standard.
                original_timeout = None
                if hasattr(ast, 'TIMEOUT'): # Pour les versions plus récentes d'astroquery où BaseQuery a TIMEOUT
                    original_timeout = ast.TIMEOUT
                    ast.TIMEOUT = timeout_sec 
                    self._log(f"WebANET: Timeout configuré à {timeout_sec}s pour l'instance AstrometryNet (via ast.TIMEOUT).", "DEBUG")
                elif hasattr(AstrometryNet, 'TIMEOUT'): # Variable de classe globale (moins courant pour instance, mais vérifions)
                     original_timeout = AstrometryNet.TIMEOUT
                     AstrometryNet.TIMEOUT = timeout_sec
                     self._log(f"WebANET: Timeout configuré à {timeout_sec}s pour la CLASSE AstrometryNet (global).", "DEBUG")
                else:
                    # Si aucun attribut TIMEOUT direct, on peut essayer de modifier la config globale d'astroquery
                    # Mais c'est plus risqué si plusieurs threads utilisent astroquery.
                    # Pour l'instant, on loggue juste si on ne peut pas le setter directement.
                    self._log(f"WebANET: Impossible de setter le timeout directement sur l'instance/classe AstrometryNet. Utilisation du timeout par défaut d'astroquery.", "WARN")
            except Exception as e_timeout:
                self._log(f"WebANET: Erreur lors de la configuration du timeout: {e_timeout}", "WARN")
        # --- ---
        # La fonction originale `solve_image_wcs` créait un fichier FITS temporaire
        # à partir de données numpy. Ici, on a déjà un fichier FITS.
        # Astrometry.net (astroquery) peut prendre un chemin de fichier directement.
        # Cependant, la préparation (luminance, uint16) était bénéfique.
        # On va donc recréer cette préparation.

        temp_prepared_fits_path = None
        wcs_solution_header_text = None # Stockera le header brut de la solution

        try:
            # --- Charger et préparer l'image pour la soumission ---
            # On utilise le fits_header_original pour les métadonnées, mais les données
            # de image_path_for_solver.
            try:
                with fits.open(image_path_for_solver, memmap=False) as hdul_solve:
                    img_data_np = hdul_solve[0].data 
                    # Le header est déjà disponible via fits_header_original
            except Exception as e_load:
                self._log(f"WebANET: Erreur chargement FITS '{image_path_for_solver}': {e_load}", "ERROR")
                return None

            if img_data_np is None:
                self._log("WebANET: Données image None après chargement.", "ERROR")
                return None
            
            # (Copie de la logique de préparation de solve_image_wcs originale)
            data_to_solve = None
            if not np.all(np.isfinite(img_data_np)):
                img_data_np = np.nan_to_num(img_data_np)

            if img_data_np.ndim == 3 and img_data_np.shape[0] == 3: # C,H,W (format FITS)
                img_data_np_hwc = np.moveaxis(img_data_np, 0, -1)
                lum_coeffs = np.array([0.299,0.587,0.114],dtype=np.float32).reshape(1,1,3)
                luminance_img = np.sum(img_data_np_hwc * lum_coeffs, axis=2).astype(np.float32)
                data_to_solve = luminance_img
            elif img_data_np.ndim == 2: # H,W
                data_to_solve = img_data_np.astype(np.float32)
            else:
                self._log(f"WebANET: Shape d'image non supportée ({img_data_np.shape}).", "ERROR")
                return None

            min_v, max_v = np.min(data_to_solve), np.max(data_to_solve)
            data_norm_float = (data_to_solve - min_v) / (max_v - min_v) if max_v > min_v else np.zeros_like(data_to_solve)
            data_uint16 = (np.clip(data_norm_float, 0.0, 1.0) * 65535.0).astype(np.uint16)
            
            header_temp_for_submission = fits.Header() # Header minimal pour soumission
            header_temp_for_submission['SIMPLE'] = True; header_temp_for_submission['BITPIX'] = 16
            header_temp_for_submission['NAXIS'] = 2
            header_temp_for_submission['NAXIS1'] = data_uint16.shape[1]
            header_temp_for_submission['NAXIS2'] = data_uint16.shape[0]
            # Copier quelques métadonnées du header original si disponibles
            for key in ['OBJECT', 'DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME', 'TELESCOP']:
                 if fits_header_original and key in fits_header_original:
                     header_temp_for_submission[key] = fits_header_original[key]

            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False, mode="wb") as temp_f:
                temp_prepared_fits_path = temp_f.name
            fits.writeto(temp_prepared_fits_path, data_uint16, header=header_temp_for_submission, overwrite=True, output_verify='silentfix')
            self._log(f"WebANET: Fichier temporaire uint16 créé: {os.path.basename(temp_prepared_fits_path)}", "DEBUG")
            del data_to_solve, data_norm_float, data_uint16
            gc.collect()

            # --- Fin préparation ---

            solve_args = {'allow_commercial_use':'n',
                            'allow_modifications':'n',
                            'publicly_visible':'n',
                            #'timeout': timeout_sec
                            }
            if scale_est_arcsec_per_pix is not None and scale_est_arcsec_per_pix > 0:
                 try:
                     scale_est = float(scale_est_arcsec_per_pix); tolerance = float(scale_tolerance_percent)
                     scale_lower = scale_est*(1.0-tolerance/100.0); scale_upper = scale_est*(1.0+tolerance/100.0)
                     solve_args['scale_units'] = 'arcsecperpix'; solve_args['scale_lower'] = scale_lower
                     solve_args['scale_upper'] = scale_upper
                     self._log(f"WebANET: Solving avec échelle: [{scale_lower:.2f} - {scale_upper:.2f}] arcsec/pix", "DEBUG")
                 except (ValueError, TypeError): self._log("WebANET: Erreur config échelle, ignorée.", "WARN")
            else: self._log("WebANET: Solving sans estimation d'échelle.", "DEBUG")

            self._log("WebANET: Soumission du job...", "INFO")
            try:
                # ast.solve_from_image retourne le HEADER de la solution, pas l'objet WCS directement
                wcs_solution_header_text = ast.solve_from_image(temp_prepared_fits_path, **solve_args)
                if wcs_solution_header_text: self._log("WebANET: Solving RÉUSSI (header solution reçu).", "INFO")
                else: self._log("WebANET: Solving ÉCHOUÉ (pas de header solution).", "WARN")
            except Exception as solve_err:
                self._log(f"WebANET: ERREUR pendant solving: {type(solve_err).__name__} - {solve_err}", "ERROR")
                traceback.print_exc(limit=1)
                wcs_solution_header_text = None

        except Exception as prep_err:
            self._log(f"WebANET: ERREUR préparation image pour soumission: {prep_err}", "ERROR")
            traceback.print_exc(limit=1)
            wcs_solution_header_text = None
        finally:
            if temp_prepared_fits_path and os.path.exists(temp_prepared_fits_path):
                try: os.remove(temp_prepared_fits_path); self._log("WebANET: Fichier temporaire supprimé.", "DEBUG")
                except Exception: pass

            # --- RESTAURER LE TIMEOUT ORIGINAL (si modifié) ---
            if 'original_timeout' in locals() and original_timeout is not None:
                try:
                    if hasattr(ast, 'TIMEOUT'): ast.TIMEOUT = original_timeout
                    elif hasattr(AstrometryNet, 'TIMEOUT'): AstrometryNet.TIMEOUT = original_timeout
                    self._log(f"WebANET: Timeout AstrometryNet restauré à sa valeur originale ({original_timeout}).", "DEBUG")
                except Exception as e_restore_timeout:
                    self._log(f"WebANET: Erreur restauration timeout: {e_restore_timeout}", "WARN")
            # --- ---
        
        
        if not wcs_solution_header_text: return None

        # Convertir le header textuel en objet WCS Astropy
        solved_wcs_object = None
        try:
            # Le header retourné par astroquery est un objet astropy.io.fits.Header
            if isinstance(wcs_solution_header_text, fits.Header):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    solved_wcs_object = WCS(wcs_solution_header_text)
                
                if solved_wcs_object and solved_wcs_object.is_celestial:
                    self._log("WebANET: Objet WCS créé avec succès.", "DEBUG")
                    # Essayer d'ajouter pixel_shape
                    nx_sol = wcs_solution_header_text.get('IMAGEW', fits_header_original.get('NAXIS1'))
                    ny_sol = wcs_solution_header_text.get('IMAGEH', fits_header_original.get('NAXIS2'))
                    if nx_sol and ny_sol: solved_wcs_object.pixel_shape = (int(nx_sol), int(ny_sol))
                    
                    if update_header_with_solution and fits_header_original is not None:
                        self._update_fits_header_with_wcs(fits_header_original, solved_wcs_object, solver_name="Astrometry.net")
                else:
                    self._log("WebANET: WCS de solution non céleste ou invalide.", "ERROR")
                    solved_wcs_object = None
            else:
                self._log("WebANET: Solution retournée n'est pas un objet Header Astropy.", "ERROR")
                solved_wcs_object = None
        except Exception as wcs_conv_err:
            self._log(f"WebANET: ERREUR conversion header solution en WCS: {wcs_conv_err}", "ERROR")
            solved_wcs_object = None
        
        return solved_wcs_object

    def _parse_wcs_file_content(self, wcs_file_path, image_shape_hw):
        """
        Lit un fichier .wcs (généré par ASTAP par exemple) et crée un objet astropy.wcs.WCS.
        Un fichier .wcs typique d'ASTAP contient des mots-clés FITS.
        """
        self._log(f"Parsing fichier WCS: {os.path.basename(wcs_file_path)} pour image shape {image_shape_hw}", "DEBUG")
        if not os.path.exists(wcs_file_path) or os.path.getsize(wcs_file_path) == 0:
            self._log(f"Fichier WCS '{wcs_file_path}' non trouvé ou vide.", "ERROR")
            return None
        try:
            # Lire le contenu du fichier .wcs
            with open(wcs_file_path, 'r') as f:
                wcs_text_content = f.read()
            
            # Créer un header FITS à partir de ce texte
            # S'assurer que les fins de ligne sont gérées (Unix vs Windows)
            wcs_header_from_text = fits.Header.fromstring(wcs_text_content.replace('\r\n', '\n').replace('\r', '\n'), sep='\n')
            
            # Créer l'objet WCS
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                wcs_obj = WCS(wcs_header_from_text, naxis=2) # naxis=2 car c'est un WCS 2D
            
            if wcs_obj and wcs_obj.is_celestial:
                # Important: définir la taille de l'image à laquelle ce WCS s'applique
                # wcs_obj.pixel_shape attend (nx, ny) soit (largeur, hauteur)
                wcs_obj.pixel_shape = (image_shape_hw[1], image_shape_hw[0])
                # Certains solveurs peuvent mettre NAXIS1/2 dans le .wcs, d'autres non.
                # On s'assure que _naxis est aussi mis à jour si possible
                try:
                    wcs_obj._naxis1 = image_shape_hw[1]
                    wcs_obj._naxis2 = image_shape_hw[0]
                except AttributeError:
                    pass # Pas grave si ces attributs privés ne sont pas là
                self._log("Objet WCS parsé avec succès depuis le fichier.", "DEBUG")
                return wcs_obj
            else:
                self._log("Échec création objet WCS valide ou céleste depuis fichier.", "ERROR")
                return None
        except Exception as e:
            self._log(f"Erreur lors du parsing du fichier WCS '{wcs_file_path}': {e}", "ERROR")
            traceback.print_exc(limit=1)
            return None

    def _update_fits_header_with_wcs(self, fits_header, wcs_object, solver_name="UnknownSolver"):
        """
        Met à jour un header FITS existant avec les informations d'un objet WCS.
        """
        if not fits_header or not wcs_object or not wcs_object.is_celestial:
            self._log("Mise à jour header annulée: header ou WCS invalide.", "WARN")
            return

        self._log(f"Mise à jour du header FITS avec la solution WCS de {solver_name}...", "DEBUG")
        try:
            # Effacer les anciennes clés WCS pour éviter les conflits, si elles existent.
            # C'est important car `fits_header.update(wcs_object.to_header())` peut ne pas
            # supprimer les anciennes clés si elles ne sont pas dans le nouveau header WCS.
            wcs_keys_to_remove = wcs_object.to_header(relax=True).keys() # Obtenir toutes les clés que WCS pourrait écrire
            # Ajouter d'autres clés WCS communes au cas où
            common_wcs_keys = ['PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                               'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CTYPE1', 'CTYPE2', 
                               'CUNIT1', 'CUNIT2', 'CDELT1', 'CDELT2', 'CROTA2', 'EQUINOX', 'RADESYS',
                               'PV1_0', 'PV1_1', 'PV1_2', 'PV2_0', 'PV2_1', 'PV2_2'] # etc.
            for key_to_del in list(set(list(wcs_keys_to_remove) + common_wcs_keys)):
                if key_to_del in fits_header:
                    try:
                        del fits_header[key_to_del]
                    except KeyError: # Devrait être impossible si "in fits_header" est vrai
                        pass
            
            # Mettre à jour le header avec le nouveau WCS
            fits_header.update(wcs_object.to_header(relax=True)) # relax=True est plus flexible

            # Ajouter des informations sur la solution
            fits_header[f'{solver_name.upper()}_SOLVED'] = (True, f'{solver_name} solution found')
            if wcs_object.pixel_scale_matrix is not None:
                try:
                    pixscale_deg = np.sqrt(np.abs(np.linalg.det(wcs_object.pixel_scale_matrix)))
                    fits_header[f'{solver_name.upper()}_SCALE_ASEC'] = (
                        pixscale_deg * 3600.0, f'[arcsec/pix] Field scale from {solver_name}'
                    )
                except Exception: pass
            self._log("Header FITS mis à jour avec succès.", "DEBUG")
        except Exception as e_hdr_update:
            self._log(f"Erreur lors de la mise à jour du header FITS avec WCS: {e_hdr_update}", "ERROR")
            traceback.print_exc(limit=1)

# --- END OF FILE seestar/alignment/astrometry_solver.py ---