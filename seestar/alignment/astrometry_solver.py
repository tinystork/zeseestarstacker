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
        
        if self.progress_callback and callable(self.progress_callback):
            try:
                # Le progress_callback du QueuedStacker attend (message, progress_value)
                # Pour les messages de ce module, on peut mettre progress_value à None.
                self.progress_callback(f"{prefix} {message}", None)
            except Exception:
                print(f"{prefix} {message}") # Fallback si le callback échoue
        else:
            print(f"{prefix} {message}")






# --- DANS LA CLASSE AstrometrySolver DANS seestar/alignment/astrometry_solver.py ---

    def solve(self, image_path, fits_header, settings, update_header_with_solution=True):
        """
        Tente de résoudre le WCS d'une image en utilisant la stratégie configurée.

        Args:
            image_path (str): Chemin vers le fichier image à résoudre.
            fits_header (fits.Header): Header FITS de l'image.
            settings (dict): Dictionnaire contenant la configuration des solveurs.
                             Clés attendues: 'local_solver_preference' (str: "none", "astap", "ansvr"),
                                           'astap_path' (str), 'astap_data_dir' (str), 'astap_search_radius' (float),
                                           'local_ansvr_path' (str), 'api_key' (str),
                                           'scale_est_arcsec_per_pix' (float, optional),
                                           'scale_tolerance_percent' (float, optional),
                                           'ansvr_timeout_sec' (int), 'astap_timeout_sec' (int),
                                           'astrometry_net_timeout_sec' (int).
            update_header_with_solution (bool): Si True, met à jour `fits_header` avec la solution.

        Returns:
            astropy.wcs.WCS or None: Objet WCS si succès, None si échec.
        """
        self._log(f"Début résolution pour: {os.path.basename(image_path)} (Utilisation de 'local_solver_preference')", "INFO")
        wcs_solution = None

        # --- Récupération des paramètres depuis le dictionnaire settings ---
        solver_preference = settings.get('local_solver_preference', "none") 
        api_key = settings.get('api_key', None)
        scale_est = settings.get('scale_est_arcsec_per_pix', None)
        scale_tol = settings.get('scale_tolerance_percent', 20)
        
        astap_exe = settings.get('astap_path', "")
        astap_data = settings.get('astap_data_dir', None)
        # Lire la valeur du rayon pour ASTAP depuis le dictionnaire settings
        astap_search_radius_from_settings = settings.get('astap_search_radius', 30.0) # Valeur par défaut si non trouvée
        astap_timeout = settings.get('astap_timeout_sec', 120)

        ansvr_config_path = settings.get('local_ansvr_path', "")
        ansvr_timeout = settings.get('ansvr_timeout_sec', 120)
        
        anet_web_timeout = settings.get('astrometry_net_timeout_sec', 300)

        # <<< AJOUT DES LOGS DE DEBUG SPÉCIFIQUES >>>
        print(f"!!!! DEBUG AstrometrySolver.solve: VALEUR LUE POUR astap_search_radius DEPUIS settings DICT = {astap_search_radius_from_settings} (type: {type(astap_search_radius_from_settings)})")
        print(f"!!!! DEBUG AstrometrySolver.solve: Dictionnaire 'settings' reçu COMPLET par solve(): {settings}")
        # <<< FIN AJOUT DES LOGS DE DEBUG >>>

        # Logs existants pour confirmer les valeurs utilisées
        print(f"DEBUG (AstrometrySolver.solve): Préférence solveur: '{solver_preference}'")
        print(f"DEBUG (AstrometrySolver.solve): ASTAP Exe: '{astap_exe}', Data: '{astap_data}', Radius (sera passé à _try_solve_astap): {astap_search_radius_from_settings}, Timeout: {astap_timeout}")
        print(f"DEBUG (AstrometrySolver.solve): Ansvr Path/Config: '{ansvr_config_path}', Timeout: {ansvr_timeout}")
        print(f"DEBUG (AstrometrySolver.solve): API Key Web: {'Présente' if api_key else 'Absente'}, Timeout Web: {anet_web_timeout}")
        print(f"DEBUG (AstrometrySolver.solve): Scale Est (pour Web/Ansvr): {scale_est}, Scale Tol: {scale_tol}")

        local_solver_attempted_and_failed = False

        if solver_preference == "astap":
            if astap_exe and os.path.isfile(astap_exe):
                self._log("Priorité au solveur local: ASTAP.", "INFO")
                wcs_solution = self._try_solve_astap(image_path, fits_header, astap_exe, astap_data,
                                                     astap_search_radius_from_settings, # Utiliser la valeur lue
                                                     scale_est, scale_tol, astap_timeout,
                                                     update_header_with_solution)
                if wcs_solution:
                    self._log("Solution trouvée avec ASTAP.", "INFO")
                    return wcs_solution
                else:
                    local_solver_attempted_and_failed = True 
                    self._log("ASTAP a échoué ou n'a pas trouvé de solution.", "WARN")
            else:
                self._log(f"ASTAP sélectionné mais chemin exécutable '{astap_exe}' invalide ou non fourni. ASTAP ignoré.", "WARN")
                local_solver_attempted_and_failed = True 

        elif solver_preference == "ansvr":
            if ansvr_config_path: 
                self._log("Priorité au solveur local: Astrometry.net Local (solve-field).", "INFO")
                wcs_solution = self._try_solve_local_ansvr(image_path, fits_header, ansvr_config_path,
                                                           scale_est, scale_tol, ansvr_timeout,
                                                           update_header_with_solution)
                if wcs_solution:
                    self._log("Solution trouvée avec Astrometry.net Local (solve-field).", "INFO")
                    return wcs_solution
                else:
                    local_solver_attempted_and_failed = True
                    self._log("Astrometry.net Local (solve-field) a échoué ou n'a pas trouvé de solution.", "WARN")
            else:
                self._log("Astrometry.net Local sélectionné mais chemin/config non fourni. Ignoré.", "WARN")
                local_solver_attempted_and_failed = True

        if solver_preference == "none" or local_solver_attempted_and_failed:
            if api_key:
                if local_solver_attempted_and_failed:
                    self._log("Solveur local préféré a échoué. Tentative avec Astrometry.net (web service) en fallback...", "INFO")
                else: 
                    self._log("Aucun solveur local préféré. Tentative avec Astrometry.net (web service)...", "INFO")
                
                wcs_solution = self._solve_astrometry_net_web(
                    image_path_for_solver=image_path,
                    fits_header_original=fits_header,
                    api_key=api_key,
                    scale_est_arcsec_per_pix=scale_est,
                    scale_tolerance_percent=scale_tol,
                    timeout_sec=anet_web_timeout,
                    update_header_with_solution=update_header_with_solution
                )
                if wcs_solution:
                    self._log("Solution trouvée avec Astrometry.net (web service).", "INFO")
                    return wcs_solution
                else:
                    self._log("Astrometry.net (web service) a échoué ou n'a pas trouvé de solution.", "WARN")
            else:
                if solver_preference == "none":
                    self._log("Aucun solveur local sélectionné et clé API pour Astrometry.net (web) non fournie.", "INFO")
                elif local_solver_attempted_and_failed:
                     self._log("Solveur local a échoué et clé API pour Astrometry.net (web) non fournie. Fallback web impossible.", "INFO")

        if not wcs_solution:
            self._log(f"Aucune solution astrométrique trouvée pour {os.path.basename(image_path)} après toutes les tentatives configurées.", "WARN")
        
        return None







    def _try_solve_local_ansvr(self, image_path, fits_header, ansvr_solver_path_or_config,
                               scale_est_arcsec_per_pix, scale_tolerance_percent, timeout_sec,
                               update_header_with_solution):
        """
        Tente de résoudre l'image en utilisant Astrometry.net local (solve-field).
        ansvr_solver_path_or_config peut être le chemin vers 'solve-field' ou vers un 'astrometry.cfg'.
        """
        self._log(f"!!!!!! ENTRÉE DANS _try_solve_local_ansvr POUR {os.path.basename(image_path)} !!!!!!", "ERROR") # Log très visible
        self._log(f"LocalAnsvr: Tentative de résolution pour {os.path.basename(image_path)}...", "INFO")

        if not os.path.isfile(image_path):
            self._log(f"LocalAnsvr: Fichier image source '{image_path}' non trouvé.", "ERROR")
            return None

        solve_field_exe = None
        config_file_to_use = None

        if os.path.isfile(ansvr_solver_path_or_config):
            if ansvr_solver_path_or_config.lower().endswith(".cfg"):
                config_file_to_use = ansvr_solver_path_or_config
                solve_field_exe = shutil.which("solve-field") # Chercher dans le PATH
                if not solve_field_exe:
                    self._log(f"LocalAnsvr: Fichier config '{config_file_to_use}' fourni, mais 'solve-field' non trouvé dans le PATH.", "ERROR")
                    return None
            else: # Supposé être l'exécutable solve-field
                solve_field_exe = ansvr_solver_path_or_config
        elif os.path.isdir(ansvr_solver_path_or_config): # Si c'est un répertoire
            # Vérifier s'il contient un astrometry.cfg
            potential_cfg = os.path.join(ansvr_solver_path_or_config, "astrometry.cfg")
            if os.path.isfile(potential_cfg):
                config_file_to_use = potential_cfg
                solve_field_exe = shutil.which("solve-field")
                if not solve_field_exe:
                    self._log(f"LocalAnsvr: Fichier config '{config_file_to_use}' trouvé dans le répertoire, mais 'solve-field' non trouvé dans le PATH.", "ERROR")
                    return None
            else: # Pas de config, on suppose que ansvr_solver_path_or_config est un dir d'index et solve-field est dans PATH
                solve_field_exe = shutil.which("solve-field")
                if not solve_field_exe:
                    self._log(f"LocalAnsvr: 'solve-field' non trouvé dans le PATH (répertoire index sans config: '{ansvr_solver_path_or_config}').", "ERROR")
                    return None
                # On pourrait ajouter --index-dir ici, mais c'est souvent géré par un astrometry.cfg global.
        else: # Non trouvé
            solve_field_exe = shutil.which("solve-field")
            if not solve_field_exe:
                self._log(f"LocalAnsvr: Chemin/Config '{ansvr_solver_path_or_config}' non valide ET 'solve-field' non trouvé dans le PATH.", "ERROR")
                return None
        
        if not os.access(solve_field_exe, os.X_OK):
            self._log(f"LocalAnsvr: Exécutable 'solve-field' ('{solve_field_exe}') non exécutable.", "ERROR")
            return None
        
        self._log(f"LocalAnsvr: Utilisation de solve-field: '{solve_field_exe}'", "DEBUG")
        if config_file_to_use:
            self._log(f"LocalAnsvr: Utilisation du fichier de configuration: '{config_file_to_use}'", "DEBUG")

        temp_dir = None
        wcs_object = None
        try:
            temp_dir = tempfile.mkdtemp(prefix="ansvr_solve_")
            self._log(f"LocalAnsvr: Répertoire temporaire créé: {temp_dir}", "DEBUG")

            output_base_name = "solved_image"
            # Chemin pour le fichier FITS de sortie qui contiendra le WCS
            output_fits_path = os.path.join(temp_dir, output_base_name + ".new")

            cmd = [
                solve_field_exe,
                "--no-plots",
                "--no-fits2fits",
                "--overwrite",
                "--dir", temp_dir, # Important pour que tous les fichiers auxiliaires aillent là
                "--new-fits", output_fits_path,
                # "--wcs", os.path.join(temp_dir, output_base_name + ".wcs"), # Fichier WCS autonome, pas crucial si on lit .new
                "--corr", os.path.join(temp_dir, output_base_name + ".corr"), # Correspondances étoiles
                "--match", os.path.join(temp_dir, output_base_name + ".match"), # Info match
                "--rdls", os.path.join(temp_dir, output_base_name + ".rdls"), # RA/DEC list
                "--axy", os.path.join(temp_dir, output_base_name + ".axy"),     # Liste des étoiles X,Y,Flux
                "--crpix-center",
                "--parity", "neg", # Très souvent nécessaire
                "-v", # Un peu de verbosité
            ]

            if config_file_to_use and os.path.isfile(config_file_to_use):
                cmd.extend(["--config", config_file_to_use])
            
            # Ajout des options d'échelle
            if scale_est_arcsec_per_pix is not None and scale_est_arcsec_per_pix > 0:
                try:
                    scale_est_val = float(scale_est_arcsec_per_pix)
                    tolerance_val = float(scale_tolerance_percent)
                    scale_lower = scale_est_val * (1.0 - tolerance_val / 100.0)
                    scale_upper = scale_est_val * (1.0 + tolerance_val / 100.0)
                    cmd.extend([
                        "--scale-units", "arcsecperpix",
                        "--scale-low", str(scale_lower),
                        "--scale-high", str(scale_upper)
                    ])
                    self._log(f"LocalAnsvr: Utilisation des bornes d'échelle: [{scale_lower:.2f} - {scale_upper:.2f}] arcsec/pix", "DEBUG")
                except (ValueError, TypeError) as e_scale:
                    self._log(f"LocalAnsvr: Erreur conversion paramètres d'échelle: {e_scale}. Options d'échelle ignorées.", "WARN")
            
            # Coordonnées RA/DEC du header (optionnel, pour accélérer la recherche)
            if fits_header:
                ra_deg_hdr = fits_header.get('RA', fits_header.get('CRVAL1')) # Essayez différentes clés
                dec_deg_hdr = fits_header.get('DEC', fits_header.get('CRVAL2'))
                # Il faudrait s'assurer que RA/DEC sont bien en degrés décimaux.
                # Pour l'instant, on suppose qu'ils le sont s'ils sont numériques.
                if isinstance(ra_deg_hdr, (int, float)) and isinstance(dec_deg_hdr, (int, float)):
                    cmd.extend(["--ra", str(ra_deg_hdr), "--dec", str(dec_deg_hdr)])
                    search_radius_deg_sf = 15 # Rayon de recherche par défaut si RA/DEC fourni
                    # On pourrait rendre ce rayon configurable via le dict settings s'il est important
                    # settings.get('ansvr_search_radius_deg', 15)
                    cmd.extend(["--radius", str(search_radius_deg_sf)])
                    self._log(f"LocalAnsvr: Utilisation RA/DEC du header: {ra_deg_hdr}, {dec_deg_hdr} avec rayon {search_radius_deg_sf} deg.", "DEBUG")
            
            cmd.append(image_path) # Le fichier d'entrée en dernier

            self._log(f"LocalAnsvr: Commande construite: {' '.join(cmd)}", "DEBUG")
            self._log(f"LocalAnsvr: Exécution avec timeout de {timeout_sec}s...", "INFO")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
            
            self._log(f"LocalAnsvr: Code de retour: {result.returncode}", "DEBUG")
            if result.stdout: self._log(f"LocalAnsvr stdout (premiers 500 caractères):\n{result.stdout[:500]}", "DEBUG")
            if result.stderr: self._log(f"LocalAnsvr stderr (premiers 500 caractères):\n{result.stderr[:500]}", "DEBUG")

            if result.returncode == 0:
                if os.path.exists(output_fits_path) and os.path.getsize(output_fits_path) > 0:
                    self._log(f"LocalAnsvr: Résolution semble réussie. Fichier '{output_fits_path}' trouvé.", "INFO")
                    try:
                        with fits.open(output_fits_path, memmap=False) as hdul_solved:
                            solved_header = hdul_solved[0].header
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", FITSFixedWarning)
                            wcs_object = WCS(solved_header, naxis=2) # Créer WCS depuis le header du fichier .new
                        
                        if wcs_object and wcs_object.is_celestial:
                            self._log("LocalAnsvr: Objet WCS créé avec succès depuis le FITS de sortie.", "INFO")
                            # Assurer que pixel_shape est défini pour l'objet WCS
                            # NAXIS1/NAXIS2 dans le header de sortie devraient refléter l'image originale
                            nx_sol = solved_header.get('NAXIS1', fits_header.get('NAXIS1'))
                            ny_sol = solved_header.get('NAXIS2', fits_header.get('NAXIS2'))
                            if nx_sol and ny_sol:
                                wcs_object.pixel_shape = (int(nx_sol), int(ny_sol))
                            
                            if update_header_with_solution and fits_header is not None:
                                self._update_fits_header_with_wcs(fits_header, wcs_object, solver_name="LocalAnsvr")
                        else:
                            self._log("LocalAnsvr: Échec création objet WCS ou WCS non céleste.", "ERROR")
                            wcs_object = None
                    except Exception as e_parse:
                        self._log(f"LocalAnsvr: Erreur lors du parsing du FITS de sortie '{output_fits_path}': {e_parse}", "ERROR")
                        wcs_object = None
                else:
                    self._log(f"LocalAnsvr: Code retour 0 mais fichier FITS de sortie '{output_fits_path}' manquant ou vide.", "ERROR")
                    wcs_object = None
            else:
                self._log(f"LocalAnsvr: Résolution échouée (code retour solve-field: {result.returncode}).", "WARN")
                wcs_object = None

        except subprocess.TimeoutExpired:
            self._log(f"LocalAnsvr: Timeout de résolution ({timeout_sec}s) expiré pour {os.path.basename(image_path)}.", "ERROR")
            wcs_object = None
        except FileNotFoundError: # Si solve_field_exe n'est pas trouvé
            self._log(f"LocalAnsvr: Exécutable 'solve-field' ('{solve_field_exe}') non trouvé. Vérifiez le chemin/PATH.", "ERROR")
            wcs_object = None
        except Exception as e:
            self._log(f"LocalAnsvr: Erreur inattendue pendant exécution/traitement: {e}", "ERROR")
            traceback.print_exc(limit=1)
            wcs_object = None
        finally:
            if temp_dir and os.path.isdir(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    self._log(f"LocalAnsvr: Répertoire temporaire '{temp_dir}' supprimé.", "DEBUG")
                except Exception as e_clean:
                    self._log(f"LocalAnsvr: Avertissement - Impossible de supprimer le répertoire temporaire '{temp_dir}': {e_clean}", "WARN")
        
        return wcs_object







# --- DANS LA CLASSE AstrometrySolver ---

    def _try_solve_astap(self, image_path, fits_header, astap_exe_path, astap_data_dir,
                         astap_search_radius_deg, 
                         scale_est_arcsec_per_pix_from_solver_UNUSED, 
                         scale_tolerance_percent_UNUSED,  
                         timeout_sec,
                         update_header_with_solution):
        self._log(f"!!!!!! ENTRÉE DANS _try_solve_astap POUR {os.path.basename(image_path)} !!!!!!", "ERROR") 
        self._log(f"ASTAP: Début résolution pour {os.path.basename(image_path)}", "INFO")
        
        image_dir = os.path.dirname(image_path)
        base_image_name_no_ext = os.path.splitext(os.path.basename(image_path))[0]
        expected_wcs_file = os.path.join(image_dir, base_image_name_no_ext + ".wcs")
        expected_ini_file = os.path.join(image_dir, base_image_name_no_ext + ".ini")
        astap_log_file_generated = os.path.join(image_dir, base_image_name_no_ext + ".log") 

        for f_to_clean in [expected_wcs_file, expected_ini_file, astap_log_file_generated]:
            if os.path.exists(f_to_clean):
                try: os.remove(f_to_clean); self._log(f"ASTAP: Ancien fichier '{os.path.basename(f_to_clean)}' supprimé.", "DEBUG")
                except Exception as e_del: self._log(f"ASTAP: Avertissement - Échec suppression '{os.path.basename(f_to_clean)}': {e_del}", "WARN")

        cmd = [astap_exe_path, "-f", image_path, "-log"] 
        if astap_data_dir and os.path.isdir(astap_data_dir):
            cmd.extend(["-d", astap_data_dir])
        
        cmd.extend(["-z", "2"]) 
        cmd.extend(["-sens", "100"]) 

        # --- TEST FORÇAGE RAYON ---
        # On force un rayon spécifique, sans RA/DEC, sans -fov, sans -pxscale
        # pour voir si ASTAP respecte ce -r
        forced_radius_test = "3.0" # ou str(astap_search_radius_deg) si vous voulez être sûr de la valeur reçue
        cmd.extend(["-r", forced_radius_test])
        self._log(f"ASTAP: TEST FORÇAGE -r {forced_radius_test} SANS AUTRES OPTIONS DE POSITION/ÉCHELLE.", "WARN")
        # --- FIN TEST FORÇAGE RAYON ---
        
        self._log(f"ASTAP: Commande finale (test forçage rayon): {' '.join(cmd)}", "DEBUG")
        wcs_object = None 
        
        try:
            # ... (le reste de la méthode : subprocess.run, parsing .wcs, etc. est inchangé) ...
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False, cwd=image_dir)
            self._log(f"ASTAP: Code de retour: {result.returncode}", "DEBUG")
            if result.stdout: self._log(f"ASTAP stdout (premiers 500 caractères):\n{result.stdout[:500]}", "DEBUG")
            if result.stderr: self._log(f"ASTAP stderr (premiers 500 caractères):\n{result.stderr[:500]}", "DEBUG")

            if result.returncode == 0: 
                if os.path.exists(expected_wcs_file) and os.path.getsize(expected_wcs_file) > 0:
                    # ... (parsing du .wcs file) ...
                    img_shape_hw_for_wcs = None
                    try:
                        with fits.open(image_path, memmap=False) as hdul_img_shape:
                            img_data_shape = hdul_img_shape[0].shape 
                            if len(img_data_shape) >= 2: img_shape_hw_for_wcs = img_data_shape[-2:] 
                            else: raise ValueError(f"Shape image inattendue: {img_data_shape}")
                    except Exception as e_shape:
                        self._log(f"ASTAP: Erreur lecture shape image ('{image_path}') pour WCS parsing: {e_shape}. Utilisation fallback header.", "WARN")
                        h_fallback = fits_header.get('NAXIS2', 1000) if fits_header else 1000
                        w_fallback = fits_header.get('NAXIS1', 1000) if fits_header else 1000
                        img_shape_hw_for_wcs = (int(h_fallback), int(w_fallback))
                    wcs_object = self._parse_wcs_file_content(expected_wcs_file, img_shape_hw_for_wcs)
                    if wcs_object and wcs_object.is_celestial:
                        self._log("ASTAP: Objet WCS créé avec succès.", "INFO")
                        if update_header_with_solution and fits_header is not None:
                            self._update_fits_header_with_wcs(fits_header, wcs_object, solver_name="ASTAP")
                    else:
                        self._log("ASTAP: Échec création objet WCS ou WCS non céleste.", "ERROR"); wcs_object = None 
                else:
                    self._log("ASTAP: Code retour 0 mais .wcs manquant/vide. Échec.", "ERROR"); wcs_object = None
            else: 
                log_msg_echec = f"ASTAP: Résolution échouée (code {result.returncode}"
                if not os.path.exists(expected_wcs_file): log_msg_echec += ", fichier .wcs NON trouvé"
                elif os.path.exists(expected_wcs_file) and os.path.getsize(expected_wcs_file) == 0: log_msg_echec += ", fichier .wcs vide"
                else: log_msg_echec += ", .wcs trouvé mais autre problème possible"
                if os.path.exists(astap_log_file_generated):
                    try:
                        with open(astap_log_file_generated, "r", errors='ignore') as f_log_astap: astap_log_content = f_log_astap.read(1000) 
                        log_msg_echec += f". Extrait ASTAP Log: ...{astap_log_content[-400:]}" 
                    except Exception as e_log_read: log_msg_echec += f". (Erreur lecture log ASTAP: {e_log_read})"
                log_msg_echec += ")."
                self._log(log_msg_echec, "WARN"); wcs_object = None
        except subprocess.TimeoutExpired:
            self._log(f"ASTAP: Timeout ({timeout_sec}s) expiré.", "ERROR"); wcs_object = None
        except FileNotFoundError: 
            self._log(f"ASTAP: Exécutable '{astap_exe_path}' non trouvé.", "ERROR"); wcs_object = None
        except Exception as e:
            self._log(f"ASTAP: Erreur inattendue: {e}", "ERROR"); traceback.print_exc(limit=1); wcs_object = None
        finally:
            if os.path.exists(expected_ini_file):
                try: os.remove(expected_ini_file)
                except Exception: pass
        return wcs_object





    def _solve_astrometry_net_web(self, image_path_for_solver, fits_header_original, api_key,
                                  scale_est_arcsec_per_pix, scale_tolerance_percent, timeout_sec,
                                  update_header_with_solution):
        """
        Méthode interne pour gérer la résolution via le service web Astrometry.net.
        Basée sur la fonction globale solve_image_wcs précédente.
        Prend un CHEMIN de fichier FITS, le charge, le prépare et le soumet.
        """
        self._log(f"!!!!!! ENTRÉE DANS _solve_astrometry_net_web POUR {os.path.basename(image_path_for_solver)} !!!!!!", "ERROR") # Log très visible
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
        original_timeout_astroquery = None # Pour restaurer
        if timeout_sec is not None and timeout_sec > 0:
            try:
                if hasattr(ast, 'TIMEOUT'): 
                    original_timeout_astroquery = ast.TIMEOUT
                    ast.TIMEOUT = timeout_sec 
                    self._log(f"WebANET: Timeout configuré à {timeout_sec}s pour l'instance AstrometryNet (via ast.TIMEOUT).", "DEBUG")
                # Si AstrometryNet.TIMEOUT est une variable de classe, on ne la modifie pas globalement ici.
                # On se fie à ce que l'instance ast.TIMEOUT soit prioritaire si elle existe.
            except Exception as e_timeout:
                self._log(f"WebANET: Erreur lors de la configuration du timeout: {e_timeout}", "WARN")
        # --- ---
        
        temp_prepared_fits_path = None
        wcs_solution_header_text = None 

        try:
            # --- Charger et préparer l'image pour la soumission ---
            try:
                with fits.open(image_path_for_solver, memmap=False) as hdul_solve:
                    img_data_np = hdul_solve[0].data 
            except Exception as e_load:
                self._log(f"WebANET: Erreur chargement FITS '{image_path_for_solver}': {e_load}", "ERROR")
                return None

            if img_data_np is None:
                self._log("WebANET: Données image None après chargement.", "ERROR")
                return None
            
            data_to_solve = None
            if not np.all(np.isfinite(img_data_np)):
                img_data_np = np.nan_to_num(img_data_np)

            if img_data_np.ndim == 3 and img_data_np.shape[0] == 3: 
                img_data_np_hwc = np.moveaxis(img_data_np, 0, -1)
                lum_coeffs = np.array([0.299,0.587,0.114],dtype=np.float32).reshape(1,1,3)
                luminance_img = np.sum(img_data_np_hwc * lum_coeffs, axis=2).astype(np.float32)
                data_to_solve = luminance_img
            elif img_data_np.ndim == 2: 
                data_to_solve = img_data_np.astype(np.float32)
            else:
                self._log(f"WebANET: Shape d'image non supportée ({img_data_np.shape}).", "ERROR")
                return None

            min_v, max_v = np.min(data_to_solve), np.max(data_to_solve)
            data_norm_float = (data_to_solve - min_v) / (max_v - min_v) if max_v > min_v else np.zeros_like(data_to_solve)
            data_uint16 = (np.clip(data_norm_float, 0.0, 1.0) * 65535.0).astype(np.uint16)
            
            header_temp_for_submission = fits.Header() 
            header_temp_for_submission['SIMPLE'] = True; header_temp_for_submission['BITPIX'] = 16
            header_temp_for_submission['NAXIS'] = 2
            header_temp_for_submission['NAXIS1'] = data_uint16.shape[1]
            header_temp_for_submission['NAXIS2'] = data_uint16.shape[0]
            for key in ['OBJECT', 'DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME', 'TELESCOP']:
                 if fits_header_original and key in fits_header_original:
                     header_temp_for_submission[key] = fits_header_original[key]

            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False, mode="wb") as temp_f:
                temp_prepared_fits_path = temp_f.name
            fits.writeto(temp_prepared_fits_path, data_uint16, header=header_temp_for_submission, overwrite=True, output_verify='silentfix')
            self._log(f"WebANET: Fichier temporaire uint16 créé: {os.path.basename(temp_prepared_fits_path)}", "DEBUG")
            del data_to_solve, data_norm_float, data_uint16, img_data_np
            gc.collect()
            
            solve_args = {'allow_commercial_use':'n',
                            'allow_modifications':'n',
                            'publicly_visible':'n',
                           }
            # Le paramètre 'timeout' pour solve_from_image est géré par ast.TIMEOUT.
            # Il n'est pas un argument direct de la méthode solve_from_image.

            if scale_est_arcsec_per_pix is not None and scale_est_arcsec_per_pix > 0:
                 try:
                     scale_est_val = float(scale_est_arcsec_per_pix); tolerance_val = float(scale_tolerance_percent)
                     scale_lower = scale_est_val*(1.0-tolerance_val/100.0); scale_upper = scale_est_val*(1.0+tolerance_val/100.0)
                     solve_args['scale_units'] = 'arcsecperpix'; solve_args['scale_lower'] = scale_lower
                     solve_args['scale_upper'] = scale_upper
                     self._log(f"WebANET: Solving avec échelle: [{scale_lower:.2f} - {scale_upper:.2f}] arcsec/pix", "DEBUG")
                 except (ValueError, TypeError): self._log("WebANET: Erreur config échelle, ignorée.", "WARN")
            else: self._log("WebANET: Solving sans estimation d'échelle.", "DEBUG")

            self._log("WebANET: Soumission du job...", "INFO")
            try:
                wcs_solution_header_text = ast.solve_from_image(temp_prepared_fits_path, **solve_args)
                if wcs_solution_header_text: self._log("WebANET: Solving RÉUSSI (header solution reçu).", "INFO")
                else: self._log("WebANET: Solving ÉCHOUÉ (pas de header solution).", "WARN")
            except Exception as solve_err: # Inclut potentiellement TimeoutError d'astroquery
                if "Timeout" in str(solve_err) or "timeout" in str(solve_err).lower():
                    self._log(f"WebANET: Timeout ({timeout_sec}s) lors du solving: {solve_err}", "ERROR")
                else:
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

            if original_timeout_astroquery is not None and hasattr(ast, 'TIMEOUT'):
                try:
                    ast.TIMEOUT = original_timeout_astroquery
                    self._log(f"WebANET: Timeout AstrometryNet restauré à sa valeur originale ({original_timeout_astroquery}).", "DEBUG")
                except Exception as e_restore_timeout:
                    self._log(f"WebANET: Erreur restauration timeout: {e_restore_timeout}", "WARN")
        
        if not wcs_solution_header_text: return None

        solved_wcs_object = None
        try:
            if isinstance(wcs_solution_header_text, fits.Header):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    solved_wcs_object = WCS(wcs_solution_header_text)
                
                if solved_wcs_object and solved_wcs_object.is_celestial:
                    self._log("WebANET: Objet WCS créé avec succès.", "DEBUG")
                    nx_sol = wcs_solution_header_text.get('IMAGEW', fits_header_original.get('NAXIS1') if fits_header_original else None)
                    ny_sol = wcs_solution_header_text.get('IMAGEH', fits_header_original.get('NAXIS2') if fits_header_original else None)
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
                    except KeyError: 
                        pass
            
            # Mettre à jour le header avec le nouveau WCS
            fits_header.update(wcs_object.to_header(relax=True)) 

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