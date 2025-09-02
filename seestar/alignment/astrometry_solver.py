"""
Module pour gérer l'interaction avec les solveurs astrométriques,
y compris Astrometry.net (web service), ASTAP (local), et ansvr (Astrometry.net local).
"""
import os
import re
import numpy as np
import warnings
import time
import tempfile
import traceback
import subprocess  # Pour appeler les solveurs locaux
import shutil  # Pour trouver les exécutables
import gc
import logging
import platform
from zemosaic import zemosaic_config
try:  # Allow running as a standalone module in tests
    from ..core.image_processing import sanitize_header_for_wcs
except ImportError:  # pragma: no cover
    from seestar.core.image_processing import sanitize_header_for_wcs

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.NullHandler())

# Default search radius in degrees used by ASTAP when no value is provided
# through solver settings. Loaded from ``zemosaic_config`` so tests and
# documentation stay in sync with application defaults.
ASTAP_DEFAULT_SEARCH_RADIUS = zemosaic_config.get_astap_default_search_radius()


def resolve_astap_executable(path: str) -> str:
    """Return the actual ASTAP binary for the provided path.

    On macOS the application is typically distributed as an ``.app`` bundle.
    In this case the real executable resides under
    ``<app>/Contents/MacOS/astap`` (or ``ASTAP``).  This helper resolves the
    path automatically while leaving paths on other systems untouched.
    """
    if (
        path
        and path.lower().endswith(".app")
        and os.path.isdir(path)
        and platform.system() == "Darwin"
    ):
        candidate = os.path.join(path, "Contents", "MacOS", "astap")
        if os.path.isfile(candidate):
            return candidate
        candidate_upper = os.path.join(path, "Contents", "MacOS", "ASTAP")
        if os.path.isfile(candidate_upper):
            return candidate_upper
    return path
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
    logger.warning(
        "AstrometrySolver: astroquery non installée. Plate-solving web Astrometry.net désactivé.")

try:
    from astropy.io import fits
    from astropy.io.fits.verify import VerifyError
    from astropy.wcs import WCS, FITSFixedWarning
    from astropy.utils.exceptions import AstropyWarning
    _ASTROPY_AVAILABLE = True
    warnings.filterwarnings('ignore', category=FITSFixedWarning)
    warnings.filterwarnings('ignore', category=AstropyWarning) # Pour d'autres avertissements astropy
    # print("DEBUG [AstrometrySolverModule]: astropy.io.fits et astropy.wcs importés.")
except ImportError:
    logger.error(
        "ERREUR CRITIQUE [AstrometrySolverModule]: Astropy non installée. Le module ne peut fonctionner.")


_CONTINUE_RE = re.compile(r"^(CONTINUE)\s+(?!')(.*)$", re.M)


def _sanitize_astap_wcs_text(text: str) -> str:
    def repl(m):
        val = m.group(2).replace("'", "''")
        return f"{m.group(1)}  '{val}'"

    return _CONTINUE_RE.sub(repl, text)


def _estimate_scale_from_fits_for_cfg(fits_path, default_pixsize_um=2.4, default_focal_mm=250.0, solver_instance=None):
    """
    Estime l’échelle en arcsec/pixel à partir du header FITS.
    Utilise des valeurs par défaut si XPIXSZ ou FOCALLEN sont absents.
    Le paramètre solver_instance est optionnel et permet de loguer via self._log si fourni.
    """
    pixel_size_um = default_pixsize_um
    focal_length_mm = default_focal_mm
    source_of_pixsize = "default (func)" # Source par défaut si pas de header ou clé
    source_of_focal = "default (func)"   # Source par défaut

    def _default_log(msg, level="INFO"):
        level_upper = str(level).upper()
        lvl = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR,
        }.get(level_upper, logging.INFO)
        logger.log(lvl, msg)

    log_func = _default_log
    if solver_instance and hasattr(solver_instance, '_log') and callable(solver_instance._log):
        log_func = solver_instance._log


    log_func(f"CFG ScaleEst: Tentative lecture FITS '{os.path.basename(fits_path)}' pour échelle.", "DEBUG")
    try:
        # Utiliser memmap=False pour éviter de garder le fichier ouvert inutilement,
        # surtout si cette fonction est appelée dans une boucle ou un contexte sensible.
        with fits.open(fits_path, memmap=False) as hdul:
            if hdul and len(hdul) > 0 and hdul[0].header: # Vérifier que le HDU et le header existent
                hdr = hdul[0].header
                # Chercher XPIXSZ (taille pixel en X)
                if 'XPIXSZ' in hdr:
                    try:
                        val = float(hdr['XPIXSZ'])
                        if val > 1e-3: # Accepter seulement si > 0.001 micron (valeur raisonnable)
                            pixel_size_um = val
                            source_of_pixsize = "header (XPIXSZ)"
                        else:
                            log_func(f"CFG ScaleEst: Valeur XPIXSZ ('{hdr['XPIXSZ']}') <= 0.001µm, fallback sur défaut.", "WARN")
                    except (ValueError, TypeError):
                        log_func(f"CFG ScaleEst: Valeur XPIXSZ ('{hdr['XPIXSZ']}') invalide dans header, fallback sur défaut.", "WARN")
                elif 'PIXSIZE1' in hdr: # Clé alternative commune
                    try:
                        val = float(hdr['PIXSIZE1'])
                        if val > 1e-3:
                            pixel_size_um = val
                            source_of_pixsize = "header (PIXSIZE1)"
                        else:
                            log_func(f"CFG ScaleEst: Valeur PIXSIZE1 ('{hdr['PIXSIZE1']}') <= 0.001µm, fallback sur défaut.", "WARN")
                    except (ValueError, TypeError):
                        log_func(f"CFG ScaleEst: Valeur PIXSIZE1 ('{hdr['PIXSIZE1']}') invalide, fallback sur défaut.", "WARN")
                else:
                    log_func(f"CFG ScaleEst: Clés XPIXSZ/PIXSIZE1 non trouvées pour '{os.path.basename(fits_path)}'. Utilisation défaut {default_pixsize_um}µm.", "DEBUG")


                # Chercher FOCALLEN (longueur focale)
                if 'FOCALLEN' in hdr:
                    try:
                        val = float(hdr['FOCALLEN'])
                        if val > 1.0: # Accepter seulement si > 1 mm (valeur raisonnable)
                            focal_length_mm = val
                            source_of_focal = "header (FOCALLEN)"
                        else:
                            log_func(f"CFG ScaleEst: Valeur FOCALLEN ('{hdr['FOCALLEN']}') <= 1mm, fallback sur défaut.", "WARN")
                    except (ValueError, TypeError):
                        log_func(f"CFG ScaleEst: Valeur FOCALLEN ('{hdr['FOCALLEN']}') invalide dans header, fallback sur défaut.", "WARN")
                else:
                    log_func(f"CFG ScaleEst: Clé FOCALLEN non trouvée pour '{os.path.basename(fits_path)}'. Utilisation défaut {default_focal_mm}mm.", "DEBUG")
            else:
                log_func(f"CFG ScaleEst: Header FITS non trouvé ou invalide dans '{os.path.basename(fits_path)}'. Utilisation des défauts pour échelle.", "WARN")
    except FileNotFoundError:
        log_func(f"CFG ScaleEst: Fichier FITS '{os.path.basename(fits_path)}' non trouvé pour estimation échelle. Utilisation des défauts.", "ERROR")
    except Exception as e:
        log_func(f"CFG ScaleEst: Erreur lecture FITS '{os.path.basename(fits_path)}' pour échelle: {e}. Utilisation des défauts.", "ERROR")
        # traceback.print_exc(limit=1) # Décommenter pour debug plus profond si besoin

    # Sécurité pour éviter division par zéro ou focale absurde
    if focal_length_mm <= 1e-3: # Si la focale est toujours invalide après lecture/fallback
        log_func(f"CFG ScaleEst: Focale finale ({focal_length_mm}mm de {source_of_focal}) invalide ou trop petite, "
                 f"forçage à la valeur par défaut de la fonction ({default_focal_mm}mm).", "WARN")
        focal_length_mm = default_focal_mm # Utiliser le défaut de la fonction en dernier recours

    # Formule: scale_arcsec_per_pix = (pixel_size_microns / focal_length_mm) * 206.265
    scale_arcsec_per_pix = (pixel_size_um / focal_length_mm) * 206.265

    log_func(f"CFG ScaleEst: Échelle finale estimée: {scale_arcsec_per_pix:.3f} arcsec/pix "
             f"(PixSz: {pixel_size_um:.2f}µm [{source_of_pixsize}], "
             f"Focale: {focal_length_mm:.1f}mm [{source_of_focal}])", "INFO") # INFO est plus visible
    return scale_arcsec_per_pix





def _generate_astrometry_cfg_auto(fits_file_for_scale_estimation,
                                 index_directory_path,
                                 output_cfg_path=None,
                                 solver_instance=None):
    """
    Génère un fichier .cfg pour solve-field qui LISTE EXPLICITEMENT les fichiers d'index
    trouvés dans le répertoire d'index fourni.
    """
    def _default_log(msg, level="INFO"):
        level_upper = str(level).upper()
        lvl = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR,
        }.get(level_upper, logging.INFO)
        logger.log(lvl, msg)

    log_func = _default_log
    if solver_instance and hasattr(solver_instance, '_log') and callable(solver_instance._log):
        log_func = solver_instance._log

    log_func(f"CFG AutoGen (List Indexes V1): Début pour index_dir '{index_directory_path}'", "INFO")

    if not os.path.isdir(index_directory_path):
        log_func(f"CFG AutoGen: ERREUR - Répertoire d'index '{index_directory_path}' non trouvé.", "ERROR")
        return None

    # --- Lister les fichiers d'index ---
    abs_index_dir = os.path.abspath(index_directory_path)
    # Utiliser glob pour trouver les fichiers d'index. Le pattern peut être ajusté.
    # On s'attend à des noms comme index-4207.fits, index-4207-00.fits, etc.
    index_files_pattern = os.path.join(abs_index_dir, "index-*.fits")
    found_index_files = glob.glob(index_files_pattern)

    if not found_index_files:
        log_func(f"CFG AutoGen: ERREUR - Aucun fichier d'index (pattern '{index_files_pattern}') trouvé dans '{abs_index_dir}'.", "ERROR")
        log_func(f"  Vérifiez que le répertoire contient des fichiers comme 'index-4207.fits', etc.", "ERROR")
        return None
    
    log_func(f"CFG AutoGen: {len(found_index_files)} fichier(s) d'index trouvé(s) dans '{abs_index_dir}'.", "DEBUG")

    # --- Détermination du chemin de sortie du .cfg (inchangée) ---
    if output_cfg_path is None:
        # ... (logique pour app_specific_cfg_dir et fallback vers temp) ...
        cfg_dir_base = os.path.expanduser("~"); app_specific_cfg_dir = os.path.join(cfg_dir_base, ".config", "zeseestarstacker_solver")
        try: os.makedirs(app_specific_cfg_dir, exist_ok=True); output_cfg_path = os.path.join(app_specific_cfg_dir, "auto_generated_astrometry.cfg")
        except OSError:
            try: temp_dir_for_cfg = tempfile.mkdtemp(prefix="zss_cfg_"); output_cfg_path = os.path.join(temp_dir_for_cfg, "auto_generated_astrometry.cfg")
            except Exception: log_func(f"CFG AutoGen: ERREUR CRITIQUE - Impossible de déterminer chemin .cfg", "ERROR"); return None
    else: # ... (logique si output_cfg_path est fourni) ...
        try: output_cfg_dir_parent = os.path.dirname(output_cfg_path); os.makedirs(output_cfg_dir_parent, exist_ok=True)
        except OSError: log_func(f"CFG AutoGen: ERREUR création dir parent pour .cfg custom", "ERROR")


    # --- Contenu du .cfg avec liste explicite des index ---
    content_lines = [
        f"# Astrometry.cfg auto-generated by ZeSeestarStacker (Explicit Index List V1)",
        f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Based on image (for context): {os.path.basename(fits_file_for_scale_estimation)}",
        f"# Index directory scanned: {abs_index_dir}",
        "",
        "# Explicitly list index files found:",
    ]
    for index_file_path in sorted(found_index_files): # Trier pour un ordre constant
        content_lines.append(f"index {index_file_path}")
    
    content_lines.extend([
        "",
        "# Path to your Astrometry.net index files (gardé pour info, mais les 'index' ci-dessus sont prioritaires)",
        f"add_path {abs_index_dir}",
        "",
        "inparallel",
        
        ""
    ])

    try:
        with open(output_cfg_path, "w") as f:
            for line in content_lines:
                f.write(line + "\n")
        log_func(f"CFG AutoGen: Fichier (Explicit Index List) '{output_cfg_path}' généré. {len(found_index_files)} index listés.", "INFO")
        return output_cfg_path
    except IOError as e_write:
        log_func(f"CFG AutoGen: ERREUR CRITIQUE - Échec écriture fichier .cfg (Explicit Index List) '{output_cfg_path}': {e_write}", "ERROR")
        return None







class AstrometrySolver:
    """
    Classe pour orchestrer la résolution astrométrique en utilisant différents solveurs.
    """
    def __init__(self, progress_callback=None, verbose=None):
        """
        Initialise le solveur.
        Args:
            progress_callback (callable, optional): Callback pour les messages de progression.
        """
        self.progress_callback = progress_callback
        if verbose is None:
            _v_env = os.getenv("SEESTAR_VERBOSE", "")
            verbose = str(_v_env).lower() in ("1", "true", "yes")
        self.verbose = verbose
        self.logger = logger
        if not _ASTROPY_AVAILABLE:
            self._log("ERREUR CRITIQUE: Astropy n'est pas disponible. AstrometrySolver ne peut fonctionner.", "ERROR")
            raise ImportError("Astropy est requis pour AstrometrySolver.")
        # Valeurs par défaut GLOBALES pour l'estimation d'échelle si FITS incomplet
        # Ces valeurs seront écrasées par celles des 'settings' dans la méthode solve() si fournies.
        self.default_pixel_size_um_for_cfg = 2.4  # Valeur Seestar S50 par défaut
        self.default_focal_length_mm_for_cfg = 250.0 # Valeur Seestar S50 par défaut
        self._settings_dict_from_solve = {} # Initialiser aussi pour ansvr_search_radius_deg

    def _extract_scale_arcsec(self, wcs_obj):
        """Return pixel scale in arcsec/pixel from a WCS object."""
        if wcs_obj and hasattr(wcs_obj, "pixel_scale_matrix"):
            try:
                return float(np.sqrt(np.abs(np.linalg.det(wcs_obj.pixel_scale_matrix))) * 3600.0)
            except Exception:
                return float("nan")
        return float("nan")




    def _log(self, message, level="INFO"):
        prefix_map = {
            "INFO": "   [AstrometrySolver]",
            "WARN": "   ⚠️ [AstrometrySolver WARN]",
            "ERROR": "   ❌ [AstrometrySolver ERROR]",
            "DEBUG": "      [AstrometrySolver DEBUG]"
        }
        level_upper = str(level).upper()
        if level_upper == "DEBUG" and not self.verbose:
            return

        prefix = prefix_map.get(level_upper, prefix_map["INFO"])
        full_msg = f"{prefix} {message}"

        if self.progress_callback and callable(self.progress_callback):
            try:
                self.progress_callback(full_msg, None)
            except Exception:
                self.logger.log(logging.ERROR, "Progress callback failed for log message")

        log_level = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR,
        }.get(level_upper, logging.INFO)
        self.logger.log(log_level, full_msg)


    def solve(
        self,
        image_path,
        fits_header,
        settings,
        update_header_with_solution=True,
        is_boring_stack_disk_mode=False,
        *,
        batch_size=None,
        final_combine=None,
    ):
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
                                           'astrometry_net_timeout_sec' (int),
                                           'use_radec_hints' (bool).
            update_header_with_solution (bool): Si True, met à jour ``fits_header`` avec la solution.
            is_boring_stack_disk_mode (bool): True uniquement pour le pipeline disque ``batch_size=1``.

        Returns:
            astropy.wcs.WCS or None: Objet WCS si succès, None si échec.
        """
        if (
            batch_size == 1
            and str(final_combine).lower()
            in {"reproject_and_coadd", "reproject", "coadd"}
        ):
            norm = image_path.replace("\\", "/").lower()
            if "/aligned_tmp/" in norm or "/classic_batch_outputs/" in norm:
                logger.info(
                    "[AstrometrySolver] Skip solving intermediate aligned/batch in BS=1+Reproject mode: %s",
                    image_path,
                )
                return None

        self._log(
            f"Début résolution pour: {os.path.basename(image_path)} (Utilisation de 'local_solver_preference')",
            "INFO",
        )
        wcs_solution = None

        
        self._settings_dict_from_solve = settings.copy() # triche :-) Stocker une copie pour accès interne 
        # --- Récupération des paramètres depuis le dictionnaire settings ---
        solver_preference = settings.get('local_solver_preference', "none") 
        api_key = settings.get('api_key', None)
        scale_est = settings.get('scale_est_arcsec_per_pix', None)
        scale_tol = settings.get('scale_tolerance_percent', 20)
        
        astap_exe = settings.get('astap_path', "")
        astap_data = settings.get('astap_data_dir', None)
        # Lire la valeur du rayon pour ASTAP depuis le dictionnaire settings
        astap_search_radius_from_settings = settings.get(
            'astap_search_radius', ASTAP_DEFAULT_SEARCH_RADIUS
        )  # Valeur par défaut si non trouvée
        astap_downsample_val = settings.get('astap_downsample', 2)
        astap_sensitivity_val = settings.get('astap_sensitivity', 100)
        astap_timeout = settings.get('astap_timeout_sec', 120)
        use_radec_hints = settings.get('use_radec_hints', False)

        ansvr_config_path = settings.get('local_ansvr_path', "")
        ansvr_timeout = settings.get('ansvr_timeout_sec', 120)
        
        anet_web_timeout = settings.get('astrometry_net_timeout_sec', 300)

        self._log(
            f"ASTAP search radius from settings: {astap_search_radius_from_settings} (type: {type(astap_search_radius_from_settings)})",
            "DEBUG",
        )
        self._log(
            f"Settings received by solve(): {settings}",
            "DEBUG",
        )

        # Logs existants pour confirmer les valeurs utilisées
        self._log(f"Solver preference: '{solver_preference}'", "DEBUG")
        self._log(
            f"ASTAP Exe: '{astap_exe}', Data: '{astap_data}', Radius (sera passé à _try_solve_astap): {astap_search_radius_from_settings}, Timeout: {astap_timeout}",
            "DEBUG",
        )
        self._log(
            f"Use RA/DEC hints: {use_radec_hints}",
            "DEBUG",
        )
        self._log(
            f"Ansvr Path/Config: '{ansvr_config_path}', Timeout: {ansvr_timeout}",
            "DEBUG",
        )
        self._log(
            f"API Key Web: {'Présente' if api_key else 'Absente'}, Timeout Web: {anet_web_timeout}",
            "DEBUG",
        )
        self._log(
            f"Scale Est (pour Web/Ansvr): {scale_est}, Scale Tol: {scale_tol}",
            "DEBUG",
        )

        local_solver_attempted_and_failed = False

        if solver_preference == "astap":
            astap_exe_resolved = resolve_astap_executable(astap_exe)
            if astap_exe_resolved and os.path.isfile(astap_exe_resolved):
                if astap_exe_resolved != astap_exe:
                    self._log(
                        f"ASTAP: bundle detected, using executable '{astap_exe_resolved}'.",
                        "DEBUG",
                    )
                self._log("Priorité au solveur local: ASTAP.", "INFO")
                t0 = time.time()
                wcs_solution = self._try_solve_astap(
                    image_path,
                    fits_header,
                    astap_exe_resolved,
                    astap_data,
                    astap_search_radius_from_settings,  # Utiliser la valeur lue
                    scale_est,
                    scale_tol,
                    astap_timeout,
                    update_header_with_solution,
                    astap_downsample_val,
                    astap_sensitivity_val,
                    use_radec_hints,
                    is_boring_stack_disk_mode=is_boring_stack_disk_mode,
                )
                if wcs_solution:
                    dt = time.time() - t0
                    scale = self._extract_scale_arcsec(wcs_solution)
                    self._log(
                        f"🔭 [Solver] ASTAP OK  –  scale {scale:.2f}\"/px  RMS 0.00″  (elapsed {dt:.1f}s)",
                        "INFO",
                    )
                    return wcs_solution
                else:
                    local_solver_attempted_and_failed = True 
                    self._log("ASTAP a échoué ou n'a pas trouvé de solution.", "WARN")
            else:
                self._log(
                    f"ASTAP sélectionné mais chemin exécutable '{astap_exe}' invalide ou non fourni.",
                    "WARN",
                )
                local_solver_attempted_and_failed = True

        elif solver_preference == "ansvr":
            if ansvr_config_path: 
                self._log("Priorité au solveur local: Astrometry.net Local (solve-field).", "INFO")
                self._log(
                    f"Preparing _try_solve_local_ansvr for {os.path.basename(image_path)}",
                    "DEBUG",
                )
                t0 = time.time()
                wcs_solution = self._try_solve_local_ansvr(
                    image_path,
                    fits_header,
                    ansvr_config_path,
                    scale_est,
                    scale_tol,
                    ansvr_timeout,
                    update_header_with_solution,
                )
                self._log(
                    f"Return from _try_solve_local_ansvr for {os.path.basename(image_path)}. Solution: {'Oui' if wcs_solution else 'Non'}",
                    'DEBUG',
                )
                if wcs_solution:
                    dt = time.time() - t0
                    scale = self._extract_scale_arcsec(wcs_solution)
                    self._log(
                        f"🔭 [Solver] ansvr OK  –  scale {scale:.2f}\"/px  RMS 0.00″  (elapsed {dt:.1f}s)",
                        "INFO",
                    )
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
                
                t0 = time.time()
                wcs_solution = self._solve_astrometry_net_web(
                    image_path_for_solver=image_path,
                    fits_header_original=fits_header,
                    api_key=api_key,
                    scale_est_arcsec_per_pix=scale_est,
                    scale_tolerance_percent=scale_tol,
                    timeout_sec=anet_web_timeout,
                    update_header_with_solution=update_header_with_solution,
                )
                if wcs_solution:
                    dt = time.time() - t0
                    scale = self._extract_scale_arcsec(wcs_solution)
                    self._log(
                        f"🔭 [Solver] Astrometry.net-API OK  –  scale {scale:.2f}\"/px  RMS 0.00″  (elapsed {dt:.1f}s)",
                        "INFO",
                    )
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







    # --- DANS LA CLASSE AstrometrySolver DANS seestar/alignment/astrometry_solver.py ---

    def _try_solve_local_ansvr(self, image_path, fits_header,
                               ansvr_user_provided_path,
                               scale_est_arcsec_per_pix,
                               scale_tolerance_percent,
                               timeout_sec,
                               update_header_with_solution):

        # --- Section 0: Log d'entrée et validation initiale de image_path ---
        base_img_name_for_log = os.path.basename(image_path) if image_path and isinstance(image_path, str) else "INVALID_IMAGE_PATH"
        entry_msg = f"Entering _try_solve_local_ansvr for {base_img_name_for_log}"
        self._log(entry_msg, "DEBUG")
        self._log(f"LocalAnsvr: Tentative résolution pour '{base_img_name_for_log}'.", "INFO")
        self._log(f"  LocalAnsvr: image_path brut reçu: '{image_path}' (type: {type(image_path)})", "DEBUG")
        self._log(f"  LocalAnsvr: ansvr_user_provided_path: '{ansvr_user_provided_path}'", "DEBUG")

        if not image_path or not os.path.isfile(image_path):
            self._log(f"LocalAnsvr: Fichier image source '{image_path}' invalide ou non trouvé. Échec.", "ERROR")
            return None
        norm_image_path_original = os.path.normpath(image_path)
        self._log(f"LocalAnsvr: Image à traiter (originale directe): '{norm_image_path_original}'.", "DEBUG")
        
        temp_dir_ansvr_solve = None; wcs_object = None
        solve_field_exe_final_path = None; config_file_to_use_for_cmd = None
        user_provided_cfg_file = None 
        
        # On utilise directement le chemin original pour solve-field
        path_to_pass_to_solve_field = norm_image_path_original
        self._log(f"LocalAnsvr: Utilisation du fichier FITS original direct pour solve-field: '{path_to_pass_to_solve_field}'", "INFO")

        try:
            temp_dir_ansvr_solve = tempfile.mkdtemp(prefix="ansvr_solve_")
            self._log(f"LocalAnsvr: Répertoire temp principal: {temp_dir_ansvr_solve}", "DEBUG")

            # --- BLOC DE CRÉATION DE COPIE FITS "PROPRE" EST MAINTENANT COMMENTÉ/SUPPRIMÉ ---
            # temp_fits_name = "cleaned_input_for_test_original_" + base_img_name_for_log
            # temp_fits_for_solving_path = os.path.join(temp_dir_ansvr_solve, temp_fits_name)
            # try:
            #     # ... (code de copie) ...
            # except Exception as e_copy_fits:
            #     self._log(f"LocalAnsvr: WARN - Erreur création copie FITS 'propre': {e_copy_fits}. Utilisation original.", "WARN")
            # --- FIN BLOC COMMENTÉ/SUPPRIMÉ ---

            # --- Section 1: Déterminer exécutable et .cfg ---
            self._log(f"LocalAnsvr: Section 1 - Interprétation ansvr_user_provided_path ('{ansvr_user_provided_path}').", "DEBUG")
            if ansvr_user_provided_path and isinstance(ansvr_user_provided_path, str) and ansvr_user_provided_path.strip():
                abs_user_path = os.path.abspath(os.path.normpath(ansvr_user_provided_path.strip()))
                if os.path.exists(abs_user_path):
                    if os.path.isfile(abs_user_path):
                        if abs_user_path.lower().endswith(".cfg"):
                            user_provided_cfg_file = abs_user_path; config_file_to_use_for_cmd = user_provided_cfg_file
                            solve_field_exe_final_path = shutil.which("solve-field")
                        else: solve_field_exe_final_path = abs_user_path
                    elif os.path.isdir(abs_user_path):
                        generated_cfg_path = _generate_astrometry_cfg_auto( # Utilise la version "Minimal V3"
                            fits_file_for_scale_estimation=path_to_pass_to_solve_field, # Pour les commentaires du .cfg
                            index_directory_path=abs_user_path, output_cfg_path=None, solver_instance=self
                        )
                        if generated_cfg_path and os.path.isfile(generated_cfg_path):
                            config_file_to_use_for_cmd = generated_cfg_path
                            solve_field_exe_final_path = shutil.which("solve-field")
                        else: self._log(f"LocalAnsvr: ERREUR - Échec génération .cfg auto pour '{abs_user_path}'.", "ERROR"); raise RuntimeError("CFG Auto Gen Failed")
                else: solve_field_exe_final_path = shutil.which("solve-field")
            else: solve_field_exe_final_path = shutil.which("solve-field")

            if not solve_field_exe_final_path or not os.path.isfile(solve_field_exe_final_path) or not os.access(solve_field_exe_final_path, os.X_OK):
                self._log(f"LocalAnsvr: ERREUR - Exécutable ('{solve_field_exe_final_path}') non valide.", "ERROR"); raise RuntimeError("Solve-field Exe Invalid")
            self._log(f"LocalAnsvr: Exe: '{solve_field_exe_final_path}'. Cfg: '{config_file_to_use_for_cmd if config_file_to_use_for_cmd else 'Aucun spécifique'}'.", "DEBUG")

            # --- Section 2: Exécution de solve-field ---
            output_base_name = "sfs_direct_" + os.path.splitext(os.path.basename(path_to_pass_to_solve_field))[0] # Nom de base pour les sorties
            output_fits_path = os.path.join(temp_dir_ansvr_solve, output_base_name + ".new")

            cmd = [
                solve_field_exe_final_path, "--no-plots", "--overwrite", "--no-verify", "--guess-scale",
                # "--downsample", "2", # Temporairement enlevé pour isoler l'effet du fichier original
                "--dir", temp_dir_ansvr_solve, "--new-fits", output_fits_path,
                "--corr", os.path.join(temp_dir_ansvr_solve, output_base_name + ".corr"),
                "--match", os.path.join(temp_dir_ansvr_solve, output_base_name + ".match"),
                "--rdls", os.path.join(temp_dir_ansvr_solve, output_base_name + ".rdls"),
                "--axy", os.path.join(temp_dir_ansvr_solve, output_base_name + ".axy"),
                "--crpix-center", "--parity", "neg", "-v"
            ]

            if config_file_to_use_for_cmd: cmd.extend(["--config", config_file_to_use_for_cmd])
            
            # Les options d'échelle manuelles ne sont PAS ajoutées car on utilise --guess-scale
            # if scale_est_arcsec_per_pix is not None and scale_est_arcsec_per_pix > 0:
            #    ... (bloc commenté)

            if fits_header:
                add_rd_cmd = True
                if config_file_to_use_for_cmd and user_provided_cfg_file and self._cfg_contains_radec(config_file_to_use_for_cmd): add_rd_cmd = False
                if add_rd_cmd:
                    ra_h=fits_header.get('RA',fits_header.get('CRVAL1')); dec_h=fits_header.get('DEC',fits_header.get('CRVAL2'))
                    if isinstance(ra_h,(int,float)) and isinstance(dec_h,(int,float)):
                        rad_s=getattr(self,'_settings_dict_from_solve',{}).get('ansvr_search_radius_deg',15.0)
                        cmd.extend(["--ra",str(ra_h),"--dec",str(dec_h),"--radius",str(max(0.1,rad_s))])
            
            cmd.append(path_to_pass_to_solve_field) # C'est maintenant norm_image_path_original
            self._log(f"LocalAnsvr (SANS COPIE FITS): Commande: {' '.join(cmd)}", "INFO")
            self._log(f"LocalAnsvr: Exécution solve-field pour '{base_img_name_for_log}' (timeout={timeout_sec}s)...", "INFO")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False, cwd=None)
            
            self._log(f"LocalAnsvr: Code retour solve-field: {result.returncode}", "DEBUG")
            if result.stdout and result.returncode != 0: self._log(f"LocalAnsvr stdout (échec):\n{result.stdout[:1000]}", "DEBUG")
            if result.stderr: self._log(f"LocalAnsvr stderr:\n{result.stderr[:1000]}", "DEBUG")

            if result.returncode == 0:
                if os.path.exists(output_fits_path) and os.path.getsize(output_fits_path) > 0:
                    self._log(f"LocalAnsvr: Résolution RÉUSSIE pour '{base_img_name_for_log}'. Fichier solution: '{os.path.basename(output_fits_path)}'.", "INFO")
                    try:
                        with fits.open(output_fits_path,memmap=False) as h_sol: solved_header=h_sol[0].header
                        with warnings.catch_warnings(): warnings.simplefilter("ignore",FITSFixedWarning); wcs_object=WCS(solved_header,naxis=2)
                        if wcs_object and wcs_object.is_celestial:
                            nx=solved_header.get('NAXIS1',fits_header.get('NAXIS1') if fits_header else None) 
                            ny=solved_header.get('NAXIS2',fits_header.get('NAXIS2') if fits_header else None)
                            if nx and ny: wcs_object.pixel_shape=(int(nx),int(ny)) 
                            if update_header_with_solution and fits_header is not None: self._update_fits_header_with_wcs(fits_header,wcs_object,solver_name="LocalAnsvr_GuessDirect") # Nom du solveur mis à jour
                        else: self._log(f"LocalAnsvr: ERREUR - WCS non céleste depuis '{os.path.basename(output_fits_path)}'.", "ERROR"); wcs_object=None
                    except Exception as e_p: self._log(f"LocalAnsvr: ERREUR parsing FITS solution '{os.path.basename(output_fits_path)}': {e_p}","ERROR"); wcs_object=None
                else: self._log(f"LocalAnsvr: ERREUR - solve-field code 0 mais FITS solution '{os.path.basename(output_fits_path)}' manquant/vide.", "ERROR"); wcs_object=None
            else: self._log(f"LocalAnsvr: WARN - solve-field a échoué pour '{base_img_name_for_log}' (code: {result.returncode}).", "WARN"); wcs_object=None
        except RuntimeError as rte_internal: self._log(f"LocalAnsvr: ERREUR (Runtime) interne: {rte_internal}", "ERROR"); wcs_object=None
        except subprocess.TimeoutExpired: self._log(f"LocalAnsvr: ERREUR - Timeout ({timeout_sec}s) pour '{base_img_name_for_log}'.", "ERROR"); wcs_object=None
        except FileNotFoundError: self._log(f"LocalAnsvr: ERREUR - Exécutable '{solve_field_exe_final_path}' non trouvé par subprocess.", "ERROR"); wcs_object=None
        except Exception as e: self._log(f"LocalAnsvr: ERREUR inattendue: {e}", "ERROR"); traceback.print_exc(limit=1); wcs_object=None
        finally:
            if temp_dir_ansvr_solve and os.path.isdir(temp_dir_ansvr_solve):
                try: shutil.rmtree(temp_dir_ansvr_solve, ignore_errors=True); self._log(f"LocalAnsvr: Répertoire temp '{temp_dir_ansvr_solve}' supprimé.", "DEBUG")
                except Exception as e_cl: self._log(f"LocalAnsvr: WARN - Erreur nettoyage dir temp '{temp_dir_ansvr_solve}': {e_cl}", "WARN")
            self._log(f"LocalAnsvr: Fin traitement pour '{base_img_name_for_log}'.", "DEBUG")

        self._log(f"LocalAnsvr: Fin résolution pour {base_img_name_for_log}. Solution trouvée: {'Oui' if wcs_object else 'Non'}", "INFO")
        return wcs_object

    # ... (le reste de la classe AstrometrySolver)







    # --- AJOUT D'UNE MÉTHODE HELPER POUR VÉRIFIER LE CONTENU DU .CFG ---
    def _cfg_contains_radec(self, cfg_path):
        """Vérifie si un fichier .cfg semble contenir des options RA/DEC."""
        if not cfg_path or not os.path.isfile(cfg_path):
            return False
        try:
            with open(cfg_path, 'r') as f_cfg:
                for line in f_cfg:
                    line_low = line.strip().lower()
                    if line_low.startswith("ra ") or line_low.startswith("dec ") or line_low.startswith("radius "):
                        return True
        except Exception:
            return False # Prudence
        return False

    def _derive_pixel_scale_from_header(self, header):
        """Return pixel scale (arcsec/pix) derived from FITS header if possible."""
        if not header:
            self._log("Pixel scale derivation: header None", "DEBUG")
            return None

        pixel_um = None
        focal_mm = None

        for key in ("XPIXSZ", "PIXSIZE1"):
            if key in header:
                try:
                    val = float(header[key])
                    if val > 0:
                        pixel_um = val
                        break
                except Exception:
                    pass

        if "FOCALLEN" in header:
            try:
                val = float(header["FOCALLEN"])
                if val > 0:
                    focal_mm = val
            except Exception:
                pass

        if pixel_um and focal_mm:
            scale = (pixel_um / focal_mm) * 206.265
            self._log(
                f"Pixel scale derived from header: {scale:.3f} arcsec/pix (pix={pixel_um}µm, focal={focal_mm}mm)",
                "DEBUG",
            )
            return scale

        self._log("Pixel scale derivation failed due to missing keywords", "DEBUG")
        return None


    def _try_solve_astap(
        self,
        image_path,
        fits_header,
        astap_exe_path,
        astap_data_dir,
        astap_search_radius_deg,
        scale_est_arcsec_per_pix_from_solver_UNUSED,
        scale_tolerance_percent_UNUSED,
        timeout_sec,
        update_header_with_solution,
        astap_downsample=2,
        astap_sensitivity=100,
        use_radec_hints=False,
        is_boring_stack_disk_mode=False,
    ):
        self._log(f"Entering _try_solve_astap for {os.path.basename(image_path)}", "DEBUG")
        self._log(f"ASTAP: Début résolution pour {os.path.basename(image_path)}", "INFO")

        image_dir = os.path.dirname(image_path)
        base_image_name_no_ext = os.path.splitext(os.path.basename(image_path))[0]

        # --- NOMS DES FICHIERS ATTENDUS ---
        expected_wcs_file = os.path.join(image_dir, base_image_name_no_ext + ".wcs")
        expected_ini_file = os.path.join(image_dir, base_image_name_no_ext + ".ini")
        # Le fichier .log généré par l'option -log d'ASTAP aura le même nom de base que l'image
        astap_log_file_generated = os.path.join(image_dir, base_image_name_no_ext + ".log")

        files_to_cleanup = [expected_wcs_file, expected_ini_file, astap_log_file_generated]

        # --- NETTOYAGE PRÉ-EXÉCUTION ---
        self._log(f"ASTAP: Nettoyage pré-exécution des fichiers temporaires potentiels...", "DEBUG")
        for f_to_clean_pre in files_to_cleanup:
            if os.path.exists(f_to_clean_pre):
                try:
                    os.remove(f_to_clean_pre)
                    self._log(f"ASTAP: Ancien fichier '{os.path.basename(f_to_clean_pre)}' supprimé avant exécution.", "DEBUG")
                except Exception as e_del_pre:
                    self._log(f"ASTAP: Avertissement - Échec suppression pré-exécution de '{os.path.basename(f_to_clean_pre)}': {e_del_pre}", "WARN")
        # --- FIN NETTOYAGE PRÉ-EXÉCUTION ---

        cmd = [astap_exe_path, "-f", image_path, "-log"] # Option -log pour générer le .log
        if astap_data_dir and os.path.isdir(astap_data_dir):
            cmd.extend(["-d", astap_data_dir])

        # Options de résolution (z, sens)
        cmd.extend(["-z", str(astap_downsample)])  # Downsample configurable
        cmd.extend(["-sens", str(astap_sensitivity)])  # Détection configurable

        # Gestion du rayon de recherche
        # astap_search_radius_deg est la valeur float reçue de settings
        if astap_search_radius_deg is not None and astap_search_radius_deg > 0:
            # ASTAP attend un rayon en degrés, ce que nous avons.
            # Si RA/DEC sont aussi fournis, ce rayon est centré.
            # Si pas de RA/DEC, ASTAP utilise ce rayon autour du centre de l'image (s'il ne trouve pas avec -fov 0).
            # L'option -fov 0 demande à ASTAP d'estimer lui-même le champ.
            # On peut soit utiliser -fov 0 (et laisser ASTAP décider), soit passer -r si on a une bonne estimation.
            # Pour l'instant, on passe -r si fourni, sinon on laisse ASTAP gérer.
            # Le comportement exact de -r sans -ra -dec est à confirmer via les logs ASTAP.
            # Le log d'ASTAP devrait indiquer "Search an area of X degrees around image center"
            radius_str = f"{float(astap_search_radius_deg):.2f}"
            cmd.extend(["-r", radius_str])
            self._log(f"ASTAP: Utilisation rayon de recherche: {radius_str}°", "DEBUG")
        else:
            # Si astap_search_radius_deg est 0 ou non fourni, ASTAP utilisera -fov 0
            # ce qui est généralement recommandé pour une recherche "aveugle".
            cmd.extend(["-fov", "0"])
            self._log(f"ASTAP: Utilisation -fov 0 (recherche automatique du champ).", "DEBUG")

        # Provide RA/DEC hints if enabled and present in the FITS header
        ra_hint = None
        dec_hint = None
        if use_radec_hints and fits_header:
            ra_hint = fits_header.get('RA', fits_header.get('CRVAL1'))
            dec_hint = fits_header.get('DEC', fits_header.get('CRVAL2'))
        hints_status_msg = "désactivés"
        if use_radec_hints and isinstance(ra_hint, (int, float)) and isinstance(dec_hint, (int, float)):
            cmd.extend(["-ra", str(ra_hint), "-dec", str(dec_hint)])
            self._log(
                f"ASTAP: Hints RA={ra_hint} DEC={dec_hint} ajoutés à la commande.",
                "DEBUG",
            )
            hints_status_msg = f"utilisés -> RA={ra_hint} DEC={dec_hint}"
        elif use_radec_hints:
            hints_status_msg = "activés mais valeurs manquantes ou invalides"
        self._log(f"ASTAP: RA/DEC hints avant exécution: {hints_status_msg}", "DEBUG")

        # Determine pixel scale from header if possible
        pxscale = self._derive_pixel_scale_from_header(fits_header)
        if isinstance(pxscale, (int, float)) and 0.1 <= pxscale <= 50.0:
            cmd.extend(["-pxscale", f"{pxscale:.3f}"])
            self._log(f"ASTAP: Option -pxscale {pxscale:.3f} utilisée.", "DEBUG")
        else:
            if "-fov" not in cmd:
                cmd.extend(["-fov", "0"])
                self._log("ASTAP: Option -fov 0 ajoutée (échelle inconnue).", "DEBUG")
            else:
                self._log("ASTAP: Échelle inconnue mais -fov déjà spécifié.", "DEBUG")

        self._log(f"ASTAP: Commande finale: {' '.join(cmd)}", "DEBUG")
        wcs_object = None

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False, cwd=image_dir)
            self._log(f"ASTAP: Code de retour: {result.returncode}", "DEBUG")
            if result.stdout: self._log(f"ASTAP stdout (premiers 500 caractères):\n{result.stdout[:500]}", "DEBUG")
            if result.stderr: self._log(f"ASTAP stderr (premiers 500 caractères):\n{result.stderr[:500]}", "DEBUG")

            if result.returncode == 0:
                img_shape_hw_for_wcs = None
                try:
                    with fits.open(image_path, memmap=False) as hdul_img_shape:
                        img_data_shape = hdul_img_shape[0].shape
                        if len(img_data_shape) >= 2:
                            img_shape_hw_for_wcs = img_data_shape[-2:]  # (H, W)
                        else:
                            raise ValueError(f"Shape image inattendue: {img_data_shape}")
                except Exception as e_shape:
                    self._log(
                        f"ASTAP: Erreur lecture shape image ('{image_path}') pour WCS parsing: {e_shape}. Utilisation fallback header.",
                        "WARN",
                    )
                    h_fallback = fits_header.get('NAXIS2', 1000) if fits_header else 1000
                    w_fallback = fits_header.get('NAXIS1', 1000) if fits_header else 1000
                    img_shape_hw_for_wcs = (int(h_fallback), int(w_fallback))

                if os.path.exists(expected_wcs_file) and os.path.getsize(expected_wcs_file) > 0:
                    self._log(f"ASTAP: Résolution réussie. Fichier '{expected_wcs_file}' trouvé.", "INFO")
                    wcs_object = self._parse_wcs_file_content(
                        expected_wcs_file, img_shape_hw_for_wcs
                    )

                    if not (wcs_object and wcs_object.is_celestial) and is_boring_stack_disk_mode:
                        self._log(
                            "[ASTAP WCS] Using FITS header fallback (batch_size=1 path)",
                            "DEBUG",
                        )
                        try:
                            with fits.open(image_path, memmap=False) as hdul:
                                hdr_fits = hdul[0].header.copy()
                            for card in hdr_fits.cards:
                                if card.keyword == "CONTINUE" and not isinstance(card.value, str):
                                    card.value = str(card.value)
                            sanitize_header_for_wcs(hdr_fits)
                            wcs_object = WCS(hdr_fits, naxis=2, relax=True)
                            assert wcs_object.is_celestial
                        except Exception as e_hdr:
                            self._log(
                                f"WCS parse failed from FITS header fallback: {e_hdr}",
                                "ERROR",
                            )
                            wcs_object = None

                    if wcs_object and wcs_object.is_celestial:
                        wcs_object.pixel_shape = (img_shape_hw_for_wcs[1], img_shape_hw_for_wcs[0])
                        try:
                            wcs_object._naxis1 = img_shape_hw_for_wcs[1]
                            wcs_object._naxis2 = img_shape_hw_for_wcs[0]
                        except AttributeError:
                            pass
                        if update_header_with_solution and fits_header is not None:
                            self._update_fits_header_with_wcs(
                                fits_header, wcs_object, solver_name="ASTAP"
                            )
                    else:
                        self._log(
                            "ASTAP: Échec création objet WCS ou WCS non céleste.",
                            "ERROR",
                        )
                        wcs_object = None
                else:
                    if is_boring_stack_disk_mode:
                        self._log(
                            "[ASTAP WCS] Sidecar .wcs missing, attempting FITS header fallback (batch_size=1 path)",
                            "DEBUG",
                        )
                        try:
                            with fits.open(image_path, memmap=False) as hdul:
                                hdr_fits = hdul[0].header.copy()
                            for card in hdr_fits.cards:
                                if card.keyword == "CONTINUE" and not isinstance(card.value, str):
                                    card.value = str(card.value)
                            sanitize_header_for_wcs(hdr_fits)
                            wcs_object = WCS(hdr_fits, naxis=2, relax=True)
                            assert wcs_object.is_celestial
                            wcs_object.pixel_shape = (
                                img_shape_hw_for_wcs[1], img_shape_hw_for_wcs[0]
                            )
                            try:
                                wcs_object._naxis1 = img_shape_hw_for_wcs[1]
                                wcs_object._naxis2 = img_shape_hw_for_wcs[0]
                            except AttributeError:
                                pass
                            if update_header_with_solution and fits_header is not None:
                                self._update_fits_header_with_wcs(
                                    fits_header, wcs_object, solver_name="ASTAP"
                                )
                        except Exception as e_hdr_only:
                            self._log(
                                f"WCS parse failed from FITS header fallback: {e_hdr_only}",
                                "ERROR",
                            )
                            wcs_object = None
                    else:
                        self._log(
                            "ASTAP: Code retour 0 mais .wcs manquant/vide. Échec.",
                            "ERROR",
                        )
                        wcs_object = None
            else:
                log_msg_echec = f"ASTAP: Résolution échouée (code {result.returncode}"
                if not os.path.exists(expected_wcs_file): log_msg_echec += ", fichier .wcs NON trouvé"
                elif os.path.exists(expected_wcs_file) and os.path.getsize(expected_wcs_file) == 0: log_msg_echec += ", fichier .wcs vide"
                else: log_msg_echec += ", .wcs trouvé mais autre problème possible"

                if os.path.exists(astap_log_file_generated):
                    try:
                        with open(astap_log_file_generated, "r", errors='ignore') as f_log_astap:
                            astap_log_content = f_log_astap.read(1000) # Lire un extrait
                        log_msg_echec += f". Extrait ASTAP Log: ...{astap_log_content[-400:]}" # Afficher la fin
                    except Exception as e_log_read:
                        log_msg_echec += f". (Erreur lecture log ASTAP: {e_log_read})"
                log_msg_echec += ")."
                self._log(log_msg_echec, "WARN")
                wcs_object = None

        except subprocess.TimeoutExpired:
            self._log(f"ASTAP: Timeout ({timeout_sec}s) expiré.", "ERROR")
            wcs_object = None
        except FileNotFoundError:
            self._log(f"ASTAP: Exécutable '{astap_exe_path}' non trouvé.", "ERROR")
            wcs_object = None
        except Exception as e:
            self._log(f"ASTAP: Erreur inattendue: {e}", "ERROR")
            traceback.print_exc(limit=1)
            wcs_object = None
        finally:
            # --- NETTOYAGE POST-EXÉCUTION ---
            self._log(f"ASTAP: Nettoyage post-exécution des fichiers temporaires...", "DEBUG")
            for f_to_clean_post in files_to_cleanup:
                if os.path.exists(f_to_clean_post):
                    try:
                        os.remove(f_to_clean_post)
                        self._log(f"ASTAP: Fichier '{os.path.basename(f_to_clean_post)}' nettoyé.", "DEBUG")
                    except Exception as e_del_post:
                        self._log(f"ASTAP: Avertissement - Échec nettoyage de '{os.path.basename(f_to_clean_post)}': {e_del_post}", "WARN")
            # --- FIN NETTOYAGE POST-EXÉCUTION ---

        return wcs_object






    def _solve_astrometry_net_web(self, image_path_for_solver, fits_header_original, api_key,
                                  scale_est_arcsec_per_pix, scale_tolerance_percent, timeout_sec,
                                  update_header_with_solution):
        """
        Méthode interne pour gérer la résolution via le service web Astrometry.net.
        Basée sur la fonction globale solve_image_wcs précédente.
        Prend un CHEMIN de fichier FITS, le charge, le prépare et le soumet.
        """
        self._log(f"Entering _solve_astrometry_net_web for {os.path.basename(image_path_for_solver)}", "DEBUG")
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
            data_int16 = (data_uint16.astype(np.int32) - 32768).astype(np.int16)
            
            header_temp_for_submission = fits.Header()
            header_temp_for_submission['SIMPLE'] = True
            header_temp_for_submission['BITPIX'] = 16
            header_temp_for_submission['BSCALE'] = 1
            header_temp_for_submission['BZERO'] = 32768
            header_temp_for_submission['NAXIS'] = 2
            header_temp_for_submission['NAXIS1'] = data_int16.shape[1]
            header_temp_for_submission['NAXIS2'] = data_int16.shape[0]
            for key in ['OBJECT', 'DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME', 'TELESCOP']:
                 if fits_header_original and key in fits_header_original:
                     header_temp_for_submission[key] = fits_header_original[key]

            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False, mode="wb") as temp_f:
                temp_prepared_fits_path = temp_f.name
            fits.writeto(
                temp_prepared_fits_path,
                data_int16,
                header=header_temp_for_submission,
                overwrite=True,
                output_verify='silentfix',
            )
            if header_temp_for_submission.get("BITPIX") == 16:
                with fits.open(temp_prepared_fits_path, mode="update", memmap=False) as hdul_fix:
                    hd0 = hdul_fix[0]
                    hd0.header["BSCALE"] = 1
                    hd0.header["BZERO"] = 32768
                    hdul_fix.flush()
            self._log(
                f"WebANET: Fichier temporaire int16 créé: {os.path.basename(temp_prepared_fits_path)}",
                "DEBUG",
            )
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
        """Parse a ``.wcs`` file and return a :class:`~astropy.wcs.WCS` object."""

        if not os.path.exists(wcs_file_path) or os.path.getsize(wcs_file_path) == 0:
            self._log(f"Fichier WCS '{wcs_file_path}' non trouvé ou vide.", "ERROR")
            return None

        with open(wcs_file_path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()

        hdr = fits.Header.fromstring(txt, sep="\n")

        for k, v in list(hdr.items()):
            if k == "CONTINUE":
                hdr[k] = str(v)
        while "HISTORY" in hdr:
            del hdr["HISTORY"]
        while "COMMENT" in hdr:
            del hdr["COMMENT"]

        try:
            wcs_obj = WCS(hdr, naxis=2, relax=True, fix=True)
        except VerifyError:
            while "CONTINUE" in hdr:
                del hdr["CONTINUE"]
            wcs_obj = WCS(hdr, naxis=2, relax=True, fix=True)

        return wcs_obj

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


def solve_image_wcs(
    image_path,
    fits_header,
    settings,
    update_header_with_solution=True,
    is_boring_stack_disk_mode=False,
    *,
    batch_size=None,
    final_combine=None,
):
    """Convenience wrapper for :class:`AstrometrySolver`.



    Parameters
    ----------
    image_path : str
        Path to the FITS image to solve.
    fits_header : astropy.io.fits.Header
        FITS header associated with the image (may be ``None``).
    settings : dict
        Dictionary of solver settings taken from :class:`SettingsManager`.
    update_header_with_solution : bool, optional
        If ``True`` the provided ``fits_header`` is updated with the solved WCS.
    is_boring_stack_disk_mode : bool, optional
        ``True`` only when running the disk-based pipeline with ``batch_size=1``.

    Returns
    -------
    astropy.wcs.WCS or None
        The solved WCS object, or ``None`` if solving failed.
    """
    try:
        solver = AstrometrySolver()
        return solver.solve(
            image_path,
            fits_header,
            settings,
            update_header_with_solution,
            is_boring_stack_disk_mode=is_boring_stack_disk_mode,
            batch_size=batch_size,
            final_combine=final_combine,
        )
    except Exception:
        return None


# --- END OF FILE seestar/alignment/astrometry_solver.py ---
