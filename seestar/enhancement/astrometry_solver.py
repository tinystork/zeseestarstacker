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


def solve_image_wcs(image_data_np, fits_header, api_key,
                    scale_est_arcsec_per_pix=None, scale_tolerance_percent=20,
                    progress_callback=None,
                    update_header_with_solution=True): # << AJOUTÉ: update_header_with_solution
    """
    Tente de résoudre le WCS d'une image via Astrometry.net (web service).
    MAJ: Envoie une image de luminance uint16 pour améliorer la compatibilité.
    MODIFIÉ: Ajout de update_header_with_solution pour contrôler la mise à jour du header.

    Args:
        image_data_np (np.ndarray): Données image pré-traitées (HxW ou HxWx3, float32, 0-1).
        fits_header (fits.Header): Header FITS original (sera mis à jour si update_header_with_solution est True).
        api_key (str): Clé API Astrometry.net.
        scale_est_arcsec_per_pix (float, optional): Estimation échelle arcsec/pixel.
        scale_tolerance_percent (float, optional): Tolérance échelle (%).
        progress_callback (callable, optional): Callback progression.
        update_header_with_solution (bool, optional): Si True, met à jour le `fits_header` fourni
                                                     avec la solution WCS trouvée.

    Returns:
        astropy.wcs.WCS or None: Objet WCS précis si succès, None si échec.
    """
    def _progress(msg): # Helper interne
        if progress_callback and callable(progress_callback):
            try: progress_callback(f"   [Astrometry] {msg}", None)
            except Exception: pass
        else: print(f"   [Astrometry] {msg}")

    _progress("Début tentative solving...")

    if not _ASTROQUERY_AVAILABLE or not _ASTROPY_AVAILABLE:
        _progress("❌ ERREUR: Dépendances manquantes (astroquery ou astropy).")
        return None
    if image_data_np is None:
        _progress("❌ ERREUR: Données image fournies sont None.")
        return None
    if not api_key:
        _progress("❌ ERREUR: Clé API Astrometry.net manquante.")
        return None

    ast = AstrometryNet()
    ast.api_key = api_key
    wcs_solution_from_header_str = None # Stockera le header brut de la solution
    temp_fits_path = None

    try:
        data_to_solve = None
        if not np.all(np.isfinite(image_data_np)):
            _progress("   -> WARNING: Données d'entrée contiennent NaN/Inf, nettoyage...")
            image_data_np = np.nan_to_num(image_data_np)

        if image_data_np.ndim == 3 and image_data_np.shape[2] == 3:
            _progress("   -> Calcul de l'image de luminance...")
            lum_coeffs = np.array([0.299, 0.587, 0.114], dtype=np.float32).reshape(1, 1, 3)
            luminance_img = np.sum(image_data_np * lum_coeffs, axis=2).astype(np.float32)
            data_to_solve = luminance_img
        elif image_data_np.ndim == 2:
            _progress("   -> Utilisation de l'image N&B directement.")
            data_to_solve = image_data_np.astype(np.float32)
        else:
            _progress(f"   -> ERREUR: Shape d'image non supportée pour solving ({image_data_np.shape}).")
            return None

        min_v, max_v = np.min(data_to_solve), np.max(data_to_solve)
        if max_v > min_v:
            data_norm_float = (data_to_solve - min_v) / (max_v - min_v)
        else:
            data_norm_float = np.zeros_like(data_to_solve)
        data_uint16 = (np.clip(data_norm_float, 0.0, 1.0) * 65535.0).astype(np.uint16)
        _progress(f"   -> Image convertie en Luminance/N&B uint16 (Shape: {data_uint16.shape})")

        header_temp = fits.Header()
        header_temp['SIMPLE'] = True; header_temp['BITPIX'] = 16
        header_temp['NAXIS'] = 2; header_temp['NAXIS1'] = data_uint16.shape[1]
        header_temp['NAXIS2'] = data_uint16.shape[0]; header_temp['BZERO'] = 32768
        header_temp['BSCALE'] = 1
        for key in ['OBJECT', 'DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME', 'TELESCOP']:
             if fits_header and key in fits_header: header_temp[key] = fits_header[key]

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as temp_f: temp_fits_path = temp_f.name
        fits.writeto(temp_fits_path, data_uint16, header=header_temp, overwrite=True, output_verify='silentfix')
        _progress(f"Fichier temporaire (Luminance uint16) créé: {os.path.basename(temp_fits_path)}")
        del data_to_solve, data_norm_float, data_uint16

        solve_args = {'allow_commercial_use':'n','allow_modifications':'n','publicly_visible':'n'}
        if scale_est_arcsec_per_pix is not None and scale_est_arcsec_per_pix > 0:
             try:
                 scale_est = float(scale_est_arcsec_per_pix); tolerance = float(scale_tolerance_percent)
                 scale_lower = scale_est*(1.0-tolerance/100.0); scale_upper = scale_est*(1.0+tolerance/100.0)
                 solve_args['scale_units'] = 'arcsecperpix'; solve_args['scale_lower'] = scale_lower
                 solve_args['scale_upper'] = scale_upper
                 _progress(f"Solving avec échelle estimée: [{scale_lower:.2f} - {scale_upper:.2f}] arcsec/pix")
             except (ValueError, TypeError) as scale_err: _progress(f"WARNING: Erreur échelle ({scale_err}), ignorée.")
        else: _progress("Solving sans estimation d'échelle.")

        _progress("Soumission du job à Astrometry.net...")
        try:
            wcs_solution_from_header_str = ast.solve_from_image(temp_fits_path, **solve_args) # Renvoie le header de la solution
            if wcs_solution_from_header_str: _progress("-> Solving RÉUSSI !")
            else: _progress("-> Solving ÉCHOUÉ (Pas de solution retournée).")
        except Exception as solve_err:
            _progress(f"-> ERREUR pendant solving: {type(solve_err).__name__}")
            print(f"ERREUR [AstrometrySolver]: {solve_err}"); traceback.print_exc(limit=2)
            wcs_solution_from_header_str = None

    except Exception as prep_err:
        _progress(f"-> ERREUR préparation/sauvegarde FITS temp: {prep_err}"); traceback.print_exc(limit=1)
        wcs_solution_from_header_str = None
    finally:
        if temp_fits_path and os.path.exists(temp_fits_path):
            try: os.remove(temp_fits_path); _progress(f"Fichier temporaire supprimé.")
            except Exception as e_del: _progress(f"-> WARNING: Échec suppression fichier temp: {e_del}")

    solved_wcs_object = None # Initialiser l'objet WCS final
    if wcs_solution_from_header_str: # Si on a un header de solution (pas l'objet WCS)
        _progress("Conversion header solution -> Objet WCS Astropy...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                solved_wcs_object = WCS(wcs_solution_from_header_str) # Convertir le header en objet WCS

            if solved_wcs_object and solved_wcs_object.is_celestial:
                _progress("-> Conversion WCS réussie.")
                # Attacher pixel_shape
                nx_sol = wcs_solution_from_header_str.get('IMAGEW', wcs_solution_from_header_str.get('NAXIS1'))
                ny_sol = wcs_solution_from_header_str.get('IMAGEH', wcs_solution_from_header_str.get('NAXIS2'))
                if nx_sol and ny_sol:
                    solved_wcs_object.pixel_shape = (int(nx_sol), int(ny_sol))
                    print(f"DEBUG [AstrometrySolver]: Pixel shape ({nx_sol},{ny_sol}) ajouté depuis header solution.")
                elif fits_header: # Fallback sur le header original si IMAGEW/H ou NAXIS1/2 manquent dans solution
                    nx_orig = fits_header.get('NAXIS1')
                    ny_orig = fits_header.get('NAXIS2')
                    if nx_orig and ny_orig:
                        solved_wcs_object.pixel_shape = (int(nx_orig), int(ny_orig))
                        print(f"DEBUG [AstrometrySolver]: Pixel shape original ({nx_orig},{ny_orig}) ajouté depuis header original.")
                    else:
                        print("   - WARNING [AstrometrySolver]: Impossible déterminer pixel_shape pour WCS solution (ni solution, ni header original).")
                else:
                     print("   - WARNING [AstrometrySolver]: Impossible déterminer pixel_shape pour WCS solution (pas de header original pour fallback).")

                # --- MISE À JOUR CONDITIONNELLE DU HEADER D'ENTRÉE ---
                if update_header_with_solution and fits_header is not None: # S'assurer que fits_header existe
                    _progress("  -> Mise à jour du header FITS d'entrée avec la solution WCS...")
                    try:
                        # Créer un header temporaire à partir de l'objet WCS
                        # pour s'assurer qu'il est bien formaté par Astropy
                        temp_wcs_header_for_update = solved_wcs_object.to_header(relax=True)
                        fits_header.update(temp_wcs_header_for_update)

                        fits_header['AN_SOLVED'] = (True, 'Astrometry.net solution found')
                        if solved_wcs_object.pixel_scale_matrix is not None:
                            try:
                                pixscale_deg = np.sqrt(np.abs(np.linalg.det(solved_wcs_object.pixel_scale_matrix)))
                                fits_header['AN_FIELD_SCALE_ASEC'] = (pixscale_deg * 3600.0, '[arcsec/pix] Field scale from Astrometry.net')
                            except Exception as e_scale:
                                _progress(f"    -> Warning: Échec calcul échelle pixel depuis matrice: {e_scale}")
                        # Ajouter d'autres infos si nécessaire (RA, DEC, ORIENTATION, etc.)
                        # if solved_wcs_object.wcs.crval is not None:
                        #     fits_header['AN_RA_PNT'] = (solved_wcs_object.wcs.crval[0], '[deg] RA of field center (Astrometry.net)')
                        #     fits_header['AN_DEC_PNT'] = (solved_wcs_object.wcs.crval[1], '[deg] Dec of field center (Astrometry.net)')
                        _progress("     Header d'entrée mis à jour avec succès.")
                    except Exception as e_hdr_update:
                        _progress(f"  ⚠️ Erreur lors de la mise à jour du header d'entrée avec WCS: {e_hdr_update}")
                        # Ne pas planter, on retourne quand même l'objet WCS si on l'a
                elif not update_header_with_solution:
                    _progress("  -> Solution WCS trouvée, mais mise à jour du header d'entrée désactivée.")
                # --- FIN MISE À JOUR CONDITIONNELLE ---

            else:
                _progress("-> ERREUR: WCS retourné non céleste ou conversion échouée.");
                solved_wcs_object = None # Assurer que None est retourné
        except Exception as wcs_conv_err:
            _progress(f"-> ERREUR conversion header solution en WCS: {wcs_conv_err}")
            solved_wcs_object = None # Assurer que None est retourné
    else:
        _progress("Aucune solution WCS obtenue (header de solution est None).")
        solved_wcs_object = None # Assurer que None est retourné

    return solved_wcs_object
