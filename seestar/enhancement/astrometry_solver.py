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

try:
    from astroquery.astrometry_net import AstrometryNet
    _ASTROQUERY_AVAILABLE = True
    print("DEBUG [AstrometrySolver]: Astroquery importé avec succès.")
except ImportError:
    print("ERREUR CRITIQUE [AstrometrySolver]: La bibliothèque 'astroquery' est requise pour le plate-solving mais n'est pas installée.")
    print("                       Veuillez exécuter: pip install astroquery")
    _ASTROQUERY_AVAILABLE = False
    # Définir une classe factice pour éviter les erreurs d'attribut ailleurs si l'import échoue
    class AstrometryNet:
        def __init__(self): pass
        def solve_from_image(self, *args, **kwargs):
            print("ERREUR: AstrometryNet (astroquery) non disponible.")
            return None

try:
    from astropy.io import fits
    from astropy.wcs import WCS, FITSFixedWarning
    _ASTROPY_AVAILABLE = True
    # Ignorer warning WCS Astropy spécifique
    warnings.filterwarnings('ignore', category=FITSFixedWarning)
except ImportError:
     print("ERREUR CRITIQUE [AstrometrySolver]: La bibliothèque 'astropy' est requise.")
     _ASTROPY_AVAILABLE = False
     # Définir des classes factices si astropy manque
     class fits:
         @staticmethod
         def writeto(*args, **kwargs): pass
         class Header: pass
     class WCS:
         def __init__(self, *args, **kwargs): pass
         def is_celestial(self): return False
         class WCS: # Classe interne factice
             pass
         wcs = WCS()


# Vérifier si les dépendances critiques sont là
if not _ASTROQUERY_AVAILABLE or not _ASTROPY_AVAILABLE:
    # On pourrait lever une exception ici pour bloquer l'app,
    # mais pour l'instant, on laisse les fonctions échouer si appelées.
    print("WARNING [AstrometrySolver]: Dépendances manquantes (astroquery ou astropy). Le Plate-solving ne fonctionnera pas.")







def solve_image_wcs(image_data_np, fits_header, api_key,
                    scale_est_arcsec_per_pix=None, scale_tolerance_percent=20,
                    progress_callback=None):
    """
    Tente de résoudre le WCS d'une image via Astrometry.net (web service).
    MAJ: Envoie une image de luminance uint16 pour améliorer la compatibilité.

    Args:
        image_data_np (np.ndarray): Données image pré-traitées (HxW ou HxWx3, float32, 0-1).
        fits_header (fits.Header): Header FITS original.
        api_key (str): Clé API Astrometry.net.
        scale_est_arcsec_per_pix (float, optional): Estimation échelle arcsec/pixel.
        scale_tolerance_percent (float, optional): Tolérance échelle (%).
        progress_callback (callable, optional): Callback progression.

    Returns:
        astropy.wcs.WCS or None: Objet WCS précis si succès, None si échec.
    """
    def _progress(msg): # Helper interne
        if progress_callback and callable(progress_callback):
            try: progress_callback(f"   [Astrometry] {msg}", None)
            except Exception: pass
        else: print(f"   [Astrometry] {msg}")

    _progress("Début tentative solving...")

    # --- Vérifications initiales (inchangées) ---
    if not _ASTROQUERY_AVAILABLE or not _ASTROPY_AVAILABLE: _progress("❌ ERREUR: Dépendances manquantes."); return None
    if image_data_np is None: _progress("❌ ERREUR: Données image None."); return None
    if not api_key: _progress("❌ ERREUR: Clé API manquante."); return None

    ast = AstrometryNet()
    ast.api_key = api_key
    wcs_solution_header = None
    temp_fits_path = None

    try:
        # --- Préparer l'image pour Astrometry.net ---
        data_to_solve = None
        # S'assurer que les données d'entrée sont finies
        if not np.all(np.isfinite(image_data_np)):
            _progress("   -> WARNING: Données d'entrée contiennent NaN/Inf, nettoyage...")
            image_data_np = np.nan_to_num(image_data_np)

        # Si couleur (HxWx3), calculer la luminance
        if image_data_np.ndim == 3 and image_data_np.shape[2] == 3:
            _progress("   -> Calcul de l'image de luminance...")
            # Coefficients standard pour luminance (Y component of YIQ)
            lum_coeffs = np.array([0.299, 0.587, 0.114], dtype=np.float32).reshape(1, 1, 3)
            # Utiliser float32 pour le calcul, puis on convertira en uint16
            luminance_img = np.sum(image_data_np * lum_coeffs, axis=2).astype(np.float32)
            data_to_solve = luminance_img
        elif image_data_np.ndim == 2:
            _progress("   -> Utilisation de l'image N&B directement.")
            data_to_solve = image_data_np.astype(np.float32) # Assurer float32
        else:
            _progress(f"   -> ERREUR: Shape d'image non supportée pour solving ({image_data_np.shape}).")
            return None # Ne peut pas continuer

        # --- Convertir en uint16 (0-65535) ---
        # Normaliser d'abord entre 0 et 1 (si ce n'est pas déjà le cas)
        min_v, max_v = np.min(data_to_solve), np.max(data_to_solve)
        if max_v > min_v:
            data_norm_float = (data_to_solve - min_v) / (max_v - min_v)
        else:
            data_norm_float = np.zeros_like(data_to_solve) # Image constante
        data_uint16 = (np.clip(data_norm_float, 0.0, 1.0) * 65535.0).astype(np.uint16)
        _progress(f"   -> Image convertie en Luminance/N&B uint16 (Shape: {data_uint16.shape})")

        # --- Créer un header MINIMAL pour le FITS temporaire ---
        header_temp = fits.Header()
        header_temp['SIMPLE'] = True
        header_temp['BITPIX'] = 16 # uint16
        header_temp['NAXIS'] = 2
        header_temp['NAXIS1'] = data_uint16.shape[1] # Width
        header_temp['NAXIS2'] = data_uint16.shape[0] # Height
        header_temp['BZERO'] = 32768 # Offset pour uint16
        header_temp['BSCALE'] = 1
        # Ajouter les mots-clés essentiels si présents dans l'original (peuvent aider)
        for key in ['OBJECT', 'DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME', 'TELESCOP']:
             if fits_header and key in fits_header: header_temp[key] = fits_header[key]

        # --- Écrire FITS temporaire ---
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as temp_f: temp_fits_path = temp_f.name
        fits.writeto(temp_fits_path, data_uint16, header=header_temp, overwrite=True, output_verify='silentfix')
        _progress(f"Fichier temporaire (Luminance uint16) créé: {os.path.basename(temp_fits_path)}")
        del data_to_solve, data_norm_float, data_uint16 # Libérer mémoire

        # --- Arguments Solveur (inchangés) ---
        solve_args = {'allow_commercial_use':'n','allow_modifications':'n','publicly_visible':'n'}
        if scale_est_arcsec_per_pix is not None and scale_est_arcsec_per_pix > 0:
             try: # ... (logique scale inchangée) ...
                 scale_est = float(scale_est_arcsec_per_pix); tolerance = float(scale_tolerance_percent); scale_lower = scale_est*(1.0-tolerance/100.0); scale_upper = scale_est*(1.0+tolerance/100.0); solve_args['scale_units'] = 'arcsecperpix'; solve_args['scale_lower'] = scale_lower; solve_args['scale_upper'] = scale_upper; _progress(f"Solving avec échelle estimée: [{scale_lower:.2f} - {scale_upper:.2f}]")
             except (ValueError, TypeError) as scale_err: _progress(f"WARNING: Erreur échelle ({scale_err}), ignorée.")
        else: _progress("Solving sans estimation d'échelle.")

        # --- Soumission et Attente ---
        _progress("Soumission du job à Astrometry.net...")
        wcs_solution_header = None
        try:
            # Utiliser un timeout interne par défaut d'astroquery (assez long)
            # On pourrait le surcharger via ast.TIMEOUT s'il existe dans ta version
            # mais évitons pour l'instant.
            wcs_solution_header = ast.solve_from_image(temp_fits_path, **solve_args)

            if wcs_solution_header: _progress("-> Solving RÉUSSI !")
            else: _progress("-> Solving ÉCHOUÉ (Pas de solution retournée).")
        except Exception as solve_err: # Attraper erreur astroquery/réseau
            _progress(f"-> ERREUR pendant solving: {type(solve_err).__name__}")
            print(f"ERREUR [AstrometrySolver]: {solve_err}"); traceback.print_exc(limit=2)
            wcs_solution_header = None

    except Exception as prep_err: # Erreur préparation/sauvegarde temp
        _progress(f"-> ERREUR préparation/sauvegarde FITS temp: {prep_err}"); traceback.print_exc(limit=1)
        wcs_solution_header = None
    finally:
        # Nettoyage Fichier Temporaire
        if temp_fits_path and os.path.exists(temp_fits_path):
            try: os.remove(temp_fits_path); _progress(f"Fichier temporaire supprimé.")
            except Exception as e_del: _progress(f"-> WARNING: Échec suppression fichier temp: {e_del}")

    # --- Conversion Header -> Objet WCS (inchangé) ---
    if wcs_solution_header:
        _progress("Conversion header solution -> Objet WCS Astropy...")
        try:
            with warnings.catch_warnings(): warnings.simplefilter("ignore", FITSFixedWarning); wcs_final = WCS(wcs_solution_header)
            if wcs_final.is_celestial:
                _progress("-> Conversion WCS réussie.")
                nx_sol = wcs_solution_header.get('NAXIS1'); ny_sol = wcs_solution_header.get('NAXIS2')
                if nx_sol and ny_sol: wcs_final.pixel_shape = (int(nx_sol), int(ny_sol)); print(f"DEBUG [AstrometrySolver]: Pixel shape ({nx_sol},{ny_sol}) ajouté.")
                else: # Fallback
                    nx_orig = fits_header.get('NAXIS1'); ny_orig = fits_header.get('NAXIS2')
                    if nx_orig and ny_orig: wcs_final.pixel_shape = (int(nx_orig), int(ny_orig)); print(f"DEBUG [AstrometrySolver]: Pixel shape original ({nx_orig},{ny_orig}) ajouté.")
                    else: print("   - WARNING: Impossible déterminer pixel_shape pour WCS solution.")
                return wcs_final
            else: _progress("-> ERREUR: WCS retourné non céleste."); return None
        except Exception as wcs_conv_err: _progress(f"-> ERREUR conversion header solution en WCS: {wcs_conv_err}"); return None
    else: _progress("Aucune solution WCS obtenue."); return None

# --- FIN DE LA FONCTION solve_image_wcs  ---





# --- FIN DU FICHIER seestar/enhancement/astrometry_solver.py ---