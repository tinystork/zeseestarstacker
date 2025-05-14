# --- START OF FILE seestar/enhancement/astrometry_solver.py ---
"""
Module pour gérer l'interaction avec Astrometry.net (web service) via astroquery
pour le plate-solving des images.
"""
import os
import numpy as np
import warnings
import time
import tempfile
import traceback
import gc

# --- Dépendances optionnelles ---
_ASTROQUERY_AVAILABLE = False
_ASTROPY_AVAILABLE = False
AstrometryNet = None # Initialiser à None

try:
    from astroquery.astrometry_net import AstrometryNet as ActualAstrometryNet
    AstrometryNet = ActualAstrometryNet # Assigner la vraie classe
    _ASTROQUERY_AVAILABLE = True
    print("DEBUG [AstrometrySolver]: astroquery.astrometry_net importé avec succès.")
except ImportError:
    print("WARNING [AstrometrySolver]: La bibliothèque 'astroquery' n'est pas installée ou importable. "
          "Le plate-solving via Astrometry.net sera désactivé.")
    # AstrometryNet reste None, _ASTROQUERY_AVAILABLE reste False

try:
    from astropy.io import fits
    from astropy.wcs import WCS, FITSFixedWarning
    _ASTROPY_AVAILABLE = True
    warnings.filterwarnings('ignore', category=FITSFixedWarning)
    print("DEBUG [AstrometrySolver]: astropy.io.fits et astropy.wcs importés avec succès.")
except ImportError:
     print("ERREUR CRITIQUE [AstrometrySolver]: La bibliothèque 'astropy' est requise mais n'est pas installée. "
           "De nombreuses fonctionnalités seront affectées.")
     # Pas de classes factices ici, car si astropy manque, l'application aura des problèmes bien plus graves.

# --- Fonction principale de solving ---

def solve_image_wcs(image_data_np, fits_header, api_key,
                    scale_est_arcsec_per_pix=None, scale_tolerance_percent=20,
                    progress_callback=None):
    """
    Tente de résoudre le WCS d'une image via Astrometry.net (web service).

    Args:
        image_data_np (np.ndarray): Données image (HxW ou HxWx3, float32, 0-1).
        fits_header (fits.Header): Header FITS original (peut être None).
        api_key (str): Clé API Astrometry.net.
        scale_est_arcsec_per_pix (float, optional): Estimation échelle arcsec/pixel.
        scale_tolerance_percent (float, optional): Tolérance échelle (%).
        progress_callback (callable, optional): Callback progression.

    Returns:
        astropy.wcs.WCS or None: Objet WCS précis si succès, None si échec.
    """
    def _progress(msg, level="INFO"): # Helper interne pour logging via callback
        log_msg = f"   [Astrometry/{level}] {msg}"
        if progress_callback and callable(progress_callback):
            try: progress_callback(log_msg, None)
            except Exception: pass # Ne pas planter si le callback échoue
        else: # Fallback si pas de callback (ex: tests unitaires)
            print(log_msg)

    _progress("Début tentative solving...")

    if not _ASTROQUERY_AVAILABLE:
        _progress("Astroquery non disponible. Plate-solving Astrometry.net annulé.", "ERROR")
        return None
    if not _ASTROPY_AVAILABLE: # Devrait déjà avoir causé des problèmes, mais sécurité
        _progress("Astropy non disponible. Plate-solving impossible.", "ERROR")
        return None
    if AstrometryNet is None: # Redondant si _ASTROQUERY_AVAILABLE est False, mais double sécurité
        _progress("Classe AstrometryNet non initialisée (import astroquery a échoué).", "ERROR")
        return None

    if image_data_np is None:
        _progress("Données image non fournies (None).", "ERROR")
        return None
    if not api_key:
        _progress("Clé API Astrometry.net non fournie.", "ERROR")
        return None

    ast_instance = AstrometryNet() # Utilise la vraie classe ou reste None si import initial a échoué
    ast_instance.api_key = api_key
    
    wcs_solution_header = None
    temp_fits_path = None

    try:
        _progress("Préparation de l'image pour Astrometry.net...")
        data_to_solve = None
        if not np.all(np.isfinite(image_data_np)):
            _progress("Nettoyage des valeurs NaN/Inf dans l'image d'entrée...", "WARN")
            image_data_np = np.nan_to_num(image_data_np)

        if image_data_np.ndim == 3 and image_data_np.shape[2] == 3: # Image couleur
            _progress("Conversion image couleur en luminance...")
            lum_coeffs = np.array([0.299, 0.587, 0.114], dtype=np.float32).reshape(1, 1, 3)
            luminance_img = np.sum(image_data_np * lum_coeffs, axis=2).astype(np.float32)
            data_to_solve = luminance_img
        elif image_data_np.ndim == 2: # Image N&B
            _progress("Utilisation de l'image N&B directement.")
            data_to_solve = image_data_np.astype(np.float32)
        else:
            _progress(f"Shape d'image non supportée: {image_data_np.shape}.", "ERROR")
            return None

        min_v, max_v = np.min(data_to_solve), np.max(data_to_solve)
        if max_v > min_v:
            data_norm_float = (data_to_solve - min_v) / (max_v - min_v)
        else: # Image constante
            data_norm_float = np.zeros_like(data_to_solve) 
        data_uint16 = (np.clip(data_norm_float, 0.0, 1.0) * 65535.0).astype(np.uint16)
        _progress(f"Image convertie en Luminance/N&B uint16 (Shape: {data_uint16.shape})")

        header_temp = fits.Header()
        header_temp['SIMPLE'] = True; header_temp['BITPIX'] = 16
        header_temp['NAXIS'] = 2; header_temp['NAXIS1'] = data_uint16.shape[1]
        header_temp['NAXIS2'] = data_uint16.shape[0]
        header_temp['BZERO'] = 32768; header_temp['BSCALE'] = 1
        if fits_header: # Copier quelques métadonnées utiles si disponibles
            for key in ['OBJECT', 'DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME', 'TELESCOP']:
                 if key in fits_header: header_temp[key] = fits_header[key]
        
        # Utiliser delete=False pour pouvoir inspecter en cas d'erreur
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False, mode='wb') as temp_f: # mode='wb' pour fits.writeto
            temp_fits_path = temp_f.name
        fits.writeto(temp_fits_path, data_uint16, header=header_temp, overwrite=True, output_verify='silentfix')
        _progress(f"Fichier temporaire créé: {os.path.basename(temp_fits_path)}")
        del data_to_solve, data_norm_float, data_uint16; gc.collect()

        solve_args = {'allow_commercial_use':'n', 'allow_modifications':'n', 'publicly_visible':'n'} # Timeout de 5 minutes
        if scale_est_arcsec_per_pix is not None and scale_est_arcsec_per_pix > 0:
             try:
                 scale_est = float(scale_est_arcsec_per_pix); tolerance = float(scale_tolerance_percent)
                 scale_lower = scale_est * (1.0 - tolerance / 100.0)
                 scale_upper = scale_est * (1.0 + tolerance / 100.0)
                 solve_args['scale_units'] = 'arcsecperpix'
                 solve_args['scale_lower'] = max(0.1, scale_lower) # Assurer une échelle minimale positive
                 solve_args['scale_upper'] = scale_upper
                 _progress(f"Solving avec échelle estimée: [{scale_lower:.2f}\" - {scale_upper:.2f}\"]")
             except (ValueError, TypeError) as scale_err:
                 _progress(f"Paramètres d'échelle invalides ({scale_err}), solving sans estimation d'échelle.", "WARN")
        else:
             _progress("Solving sans estimation d'échelle.")

        _progress("Soumission du job à Astrometry.net (peut prendre plusieurs minutes)...")
        try:
            wcs_solution_header = ast_instance.solve_from_image(temp_fits_path, **solve_args)
            if wcs_solution_header:
                _progress("Solving RÉUSSI !")
            else:
                _progress("Solving ÉCHOUÉ (Astrometry.net n'a pas retourné de solution).", "WARN")
        except Exception as solve_err:
            _progress(f"ERREUR pendant la communication avec Astrometry.net: {type(solve_err).__name__} - {solve_err}", "ERROR")
            print(f"ERREUR [AstrometrySolver]: {solve_err}"); traceback.print_exc(limit=2)
            wcs_solution_header = None # Assurer qu'il est None

    except Exception as prep_err:
        _progress(f"ERREUR lors de la préparation de l'image pour Astrometry.net: {prep_err}", "ERROR")
        print(f"ERREUR [AstrometrySolver] (préparation): {prep_err}"); traceback.print_exc(limit=1)
        wcs_solution_header = None
    finally:
        if temp_fits_path and os.path.exists(temp_fits_path):
            try:
                os.remove(temp_fits_path)
                # _progress(f"Fichier temporaire {os.path.basename(temp_fits_path)} supprimé.") # Peut être trop verbeux
            except Exception as e_del:
                _progress(f"Échec suppression fichier temporaire {os.path.basename(temp_fits_path)}: {e_del}", "WARN")

    if wcs_solution_header:
        _progress("Conversion du header de solution en objet WCS Astropy...")
        try:
            # Utiliser with warnings.catch_warnings() pour ignorer les FITSFixedWarning locaux à cette conversion
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FITSFixedWarning)
                wcs_final = WCS(wcs_solution_header)
            
            if wcs_final and wcs_final.is_celestial:
                _progress("Conversion WCS réussie.")
                # Essayer d'attacher pixel_shape depuis le header de solution ou l'original
                nx_sol = wcs_solution_header.get('IMAGEW', wcs_solution_header.get('NAXIS1')) # Astrometry.net utilise IMAGEW/H
                ny_sol = wcs_solution_header.get('IMAGEH', wcs_solution_header.get('NAXIS2'))
                if nx_sol and ny_sol:
                    wcs_final.pixel_shape = (int(nx_sol), int(ny_sol))
                elif fits_header and fits_header.get('NAXIS1') and fits_header.get('NAXIS2'): # Fallback sur header original
                    wcs_final.pixel_shape = (int(fits_header['NAXIS1']), int(fits_header['NAXIS2']))
                
                if hasattr(wcs_final, 'pixel_shape') and wcs_final.pixel_shape:
                     print(f"DEBUG [AstrometrySolver]: pixel_shape ({wcs_final.pixel_shape[0]},{wcs_final.pixel_shape[1]}) attaché au WCS résolu.")
                else:
                     print("WARN [AstrometrySolver]: Impossible de déterminer pixel_shape pour WCS résolu. Les calculs de grille pourraient être affectés.")
                return wcs_final
            else:
                _progress("WCS retourné par Astrometry.net n'est pas céleste ou invalide.", "ERROR")
                return None
        except Exception as wcs_conv_err:
            _progress(f"ERREUR conversion header solution en WCS Astropy: {wcs_conv_err}", "ERROR")
            print(f"ERREUR [AstrometrySolver] (conversion WCS): {wcs_conv_err}"); traceback.print_exc(limit=1)
            return None
    else:
        _progress("Aucune solution WCS valide obtenue d'Astrometry.net.")
        return None

# --- Fin du fichier seestar/enhancement/astrometry_solver.py ---