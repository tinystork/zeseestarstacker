# --- START OF FILE seestar/enhancement/mosaic_processor.py ---
"""
Module pour orchestrer le traitement spécifique des mosaïques,
incluant le groupement par panneau, le stacking/solving par panneau,
et la combinaison finale Drizzle.
"""
import os
import numpy as np
import time
import traceback
import gc
import warnings # Ajouté pour warnings

# --- Imports Astropy ---
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

# --- Imports Internes Seestar ---
# Depuis le solver
try:
    from .astrometry_solver import solve_image_wcs
    print("DEBUG [MosaicProcessor Import]: astrometry_solver importé.")
except ImportError:
    print("ERREUR [MosaicProcessor Import]: astrometry_solver manquant.")
    def solve_image_wcs(*args, **kwargs): raise ImportError("astrometry_solver absent")

# Depuis l'intégration Drizzle
try:
    from .drizzle_integration import DrizzleProcessor, _load_drizzle_temp_file
    print("DEBUG [MosaicProcessor Import]: drizzle_integration importé.")
    _DRIZZLE_PROC_AVAILABLE = True
except ImportError:
    print("ERREUR [MosaicProcessor Import]: drizzle_integration manquant.")
    class DrizzleProcessor: pass # Factice
    def _load_drizzle_temp_file(*args, **kwargs): raise ImportError("drizzle_integration absent")
    _DRIZZLE_PROC_AVAILABLE = False

# Depuis le gestionnaire de queue (pour type hinting et accès méthodes/attributs)
# Utiliser un import conditionnel ou juste le type hinting si possible
# Tenter d'importer SeestarQueuedStacker pour type hinting et accès aux attributs
# Cela peut créer une dépendance circulaire si mosaic_processor est importé trop tôt par queue_manager.
# Une meilleure solution serait de passer les attributs nécessaires explicitement.
# Pour l'instant, on essaie l'import.
try:
    from ..queuep.queue_manager import SeestarQueuedStacker
    print("DEBUG [MosaicProcessor Import]: SeestarQueuedStacker importé (pour type hint).")
except ImportError:
    SeestarQueuedStacker = None # Type hint factice
    # Cette erreur est logguée par queue_manager s'il ne peut pas importer ceci.
    # print("ERREUR [MosaicProcessor Import]: SeestarQueuedStacker manquant.")

# --- Constantes ---
PANEL_GROUPING_THRESHOLD_DEG = 0.3 # Seuil pour regrouper les panneaux

# Ignorer warning WCS Astropy
warnings.filterwarnings('ignore', category=FITSFixedWarning)


# === Fonctions Helper ===

def calculate_angular_distance(ra1, dec1, ra2, dec2):
    """Calcule la distance angulaire entre deux points en degrés."""
    # Utiliser try-except pour la robustesse
    try:
        c1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg, frame='icrs')
        c2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg, frame='icrs')
        return c1.separation(c2).deg
    except ValueError: # Si RA/DEC non valides
        print(f"WARN [CalcDist]: Valeurs RA/DEC invalides ({ra1},{dec1} ou {ra2},{dec2})")
        return float('inf') # Retourner une grande distance pour forcer nouveau panneau

def _save_panel_stack_temp(panel_stack_data, solved_wcs, panel_index, output_folder):
    """
    Sauvegarde le stack d'un panneau (HxWxC float32) avec son WCS résolu
    dans un fichier FITS temporaire dans un sous-dossier dédié.
    """
    if panel_stack_data is None or solved_wcs is None:
        print(f"ERREUR [_save_panel_stack_temp]: Données ou WCS manquant pour panneau {panel_index}.")
        return None
    # Créer le sous-dossier s'il n'existe pas
    temp_dir = os.path.join(output_folder, "mosaic_panel_stacks_temp")
    try:
        os.makedirs(temp_dir, exist_ok=True)
    except OSError as e:
        print(f"ERREUR [_save_panel_stack_temp]: Impossible de créer dossier temp {temp_dir}: {e}")
        return None

    temp_filename = f"panel_stack_{panel_index:03d}_solved.fits"
    temp_filepath = os.path.join(temp_dir, temp_filename)

    try:
        print(f"      -> Sauvegarde stack panneau temporaire: {temp_filename}")
        # Préparer données (CxHxW float32)
        if panel_stack_data.ndim == 3 and panel_stack_data.shape[2] == 3:
            data_to_save = np.moveaxis(panel_stack_data, -1, 0).astype(np.float32)
        elif panel_stack_data.ndim == 2: # Accepter N&B aussi ?
            data_to_save = panel_stack_data.astype(np.float32)[np.newaxis, :, :] # Ajouter axe C
        else:
            raise ValueError(f"Shape de stack panneau non supportée: {panel_stack_data.shape}")

        # Préparer header (WCS résolu + infos de base)
        header_to_save = solved_wcs.to_header(relax=True) # Utilise le WCS résolu
        header_to_save['HISTORY'] = f"Stacked Panel {panel_index}"
        header_to_save['NIMAGES'] = (panel_stack_data.shape[0] if panel_stack_data.ndim==4 else 1,'Images in this panel stack') # Approximatif si données mémoire
        # Ajouter infos NAXIS basées sur data_to_save (CxHxW)
        header_to_save['NAXIS'] = 3
        header_to_save['NAXIS1'] = data_to_save.shape[2] # Width
        header_to_save['NAXIS2'] = data_to_save.shape[1] # Height
        header_to_save['NAXIS3'] = data_to_save.shape[0] # Channels (1 ou 3)
        if data_to_save.shape[0] == 3: header_to_save['CTYPE3'] = 'RGB'
        else: header_to_save['CTYPE3'] = 'INTENSITY'

        # Sauvegarde FITS
        fits.writeto(temp_filepath, data_to_save, header=header_to_save, overwrite=True, output_verify='ignore')
        print(f"      -> Stack panneau temp sauvegardé: {temp_filename}")
        return temp_filepath
    except Exception as e:
        print(f"      -> ERREUR sauvegarde stack panneau temp {temp_filename}: {e}")
        traceback.print_exc(limit=1)
        # Essayer de supprimer le fichier s'il a été créé partiellement
        if os.path.exists(temp_filepath): 
            try: os.remove(temp_filepath)
            except Exception: pass
        return None





# --- DANS seestar/enhancement/mosaic_processor.py ---
# (Assurez-vous que les imports nécessaires sont présents en haut du fichier :
#  import numpy as np, from astropy.wcs import WCS, from astropy.coordinates import SkyCoord,
#  from astropy import units as u, import traceback)

def _calculate_final_mosaic_grid_optimized(panel_wcs_list, panel_shapes_hw_list, drizzle_scale_factor):
    """
    Calcule le WCS et la Shape (H, W) optimaux pour la mosaïque finale,
    en se basant sur l'étendue combinée de tous les panneaux fournis.

    Args:
        panel_wcs_list (list): Liste des objets astropy.wcs.WCS pour chaque panneau stacké.
                               Chaque WCS doit avoir .pixel_shape défini.
        panel_shapes_hw_list (list): Liste des tuples (H, W) pour chaque panneau stacké,
                                     correspondant à panel_wcs_list.
        drizzle_scale_factor (float): Facteur d'échelle à appliquer pour la grille Drizzle finale
                                      par rapport à l'échelle moyenne des panneaux d'entrée.

    Returns:
        tuple: (output_wcs, output_shape_hw) ou (None, None) si échec.
               output_shape_hw est au format (Hauteur, Largeur).
    """
    num_panels = len(panel_wcs_list)
    print(f"DEBUG [MosaicGridOptim]: Début calcul grille mosaïque pour {num_panels} panneaux.")
    print(f"  -> Échelle Drizzle demandée: {drizzle_scale_factor}x")

    # --- 1. Validation des Entrées ---
    if num_panels == 0:
        print("ERREUR [MosaicGridOptim]: Aucune information WCS de panneau fournie.")
        return None, None
    if len(panel_shapes_hw_list) != num_panels:
        print("ERREUR [MosaicGridOptim]: Nombre de WCS et de shapes de panneaux incohérent.")
        return None, None
    if None in panel_shapes_hw_list or not all(isinstance(s, tuple) and len(s) == 2 and s[0] > 0 and s[1] > 0 for s in panel_shapes_hw_list):
        print("ERREUR [MosaicGridOptim]: Certaines shapes de panneaux sont invalides (None, non-tuple, ou dimensions <= 0).")
        return None, None
    if None in panel_wcs_list or not all(isinstance(w, WCS) and w.is_celestial for w in panel_wcs_list):
        print("ERREUR [MosaicGridOptim]: Certains WCS de panneaux sont invalides (None ou non-célestes).")
        return None, None
        
    print(f"  -> {num_panels} panneaux valides avec WCS et shapes pour calcul de la grille.")

    try:
        # --- 2. Calcul des "Footprints" (Empreintes Célestes) pour chaque Panneau ---
        #    Le footprint est la projection des 4 coins de chaque panneau sur le ciel.
        all_panel_footprints_sky = []
        print("   -> Calcul des footprints célestes des panneaux...")
        for i, (wcs_panel, shape_hw_panel) in enumerate(zip(panel_wcs_list, panel_shapes_hw_list)):
            panel_h, panel_w = shape_hw_panel # Hauteur, Largeur du panneau i

            # S'assurer que le WCS du panneau a la bonne pixel_shape (W,H pour Astropy)
            # C'est crucial pour que wcs_panel.pixel_to_world fonctionne correctement avec les coins.
            if wcs_panel.pixel_shape is None or wcs_panel.pixel_shape != (panel_w, panel_h):
                print(f"      - Ajustement pixel_shape pour WCS panneau {i+1} à ({panel_w}, {panel_h})")
                wcs_panel.pixel_shape = (panel_w, panel_h) # (nx, ny) pour Astropy

            # Coins en coordonnées pixel (0-based pour Astropy)
            # Ordre: (0,0), (W-1,0), (W-1,H-1), (0,H-1)
            pixel_corners = np.array([
                [0, 0], [panel_w - 1, 0], [panel_w - 1, panel_h - 1], [0, panel_h - 1]
            ], dtype=np.float64)
            
            try:
                # Projeter les coins pixel sur le ciel
                sky_corners_panel = wcs_panel.pixel_to_world(pixel_corners[:, 0], pixel_corners[:, 1])
                all_panel_footprints_sky.append(sky_corners_panel)
                # print(f"      - Footprint Panneau {i+1} (RA): {sky_corners_panel.ra.deg}") # Debug
            except Exception as fp_err:
                print(f"      - ERREUR calcul footprint pour panneau {i+1}: {fp_err}. Panneau ignoré.")
                # Continuer si un footprint échoue, mais cela peut affecter la grille finale

        if not all_panel_footprints_sky:
            print("ERREUR [MosaicGridOptim]: Aucun footprint de panneau n'a pu être calculé.")
            return None, None
        print(f"   -> {len(all_panel_footprints_sky)} footprints de panneaux calculés.")

        # --- 3. Détermination de l'Étendue Globale et du Centre de la Mosaïque ---
        print("   -> Calcul de l'étendue globale et du centre de la mosaïque...")
        # Concaténer tous les coins de tous les footprints en une seule liste de SkyCoord
        all_sky_corners_flat = SkyCoord(
            ra=np.concatenate([fp.ra.deg for fp in all_panel_footprints_sky]),
            dec=np.concatenate([fp.dec.deg for fp in all_panel_footprints_sky]),
            unit='deg', frame='icrs' # Assumer ICRS pour tous
        )

        # Centre approximatif (médiane pour robustesse)
        # Gérer le "wrap" du RA autour de 0h/360deg en utilisant wrap_at(180deg)
        median_ra_deg = np.median(all_sky_corners_flat.ra.wrap_at(180 * u.deg).deg)
        median_dec_deg = np.median(all_sky_corners_flat.dec.deg)
        print(f"      - Centre Médian Mosaïque (RA, Dec): ({median_ra_deg:.5f}, {median_dec_deg:.5f}) deg")

        # --- 4. Création du WCS de Sortie pour la Mosaïque ---
        print("   -> Création du WCS de sortie pour la mosaïque...")
        # Utiliser le WCS du premier panneau valide comme référence pour CTYPE, CUNIT
        ref_wcs_for_output_params = panel_wcs_list[0] 
        
        output_wcs = WCS(naxis=2)
        output_wcs.wcs.ctype = getattr(ref_wcs_for_output_params.wcs, 'ctype', ["RA---TAN", "DEC--TAN"])
        output_wcs.wcs.crval = [median_ra_deg, median_dec_deg] # Centrer sur la médiane
        output_wcs.wcs.cunit = getattr(ref_wcs_for_output_params.wcs, 'cunit', ['deg', 'deg'])

        # Calculer l'échelle de pixel de sortie
        # Prendre l'échelle moyenne des panneaux d'entrée (en degrés/pixel)
        avg_panel_pixel_scale_deg = 0.0
        valid_scales_count = 0
        for wcs_p in panel_wcs_list:
            try:
                # pixel_scale_matrix est en unités de wcs.cunit par pixel
                # On s'attend à ce que cunit soit 'deg'
                scale_matrix_p = wcs_p.pixel_scale_matrix 
                # Prendre la moyenne des valeurs absolues diagonales comme échelle approx.
                current_panel_scale = np.mean(np.abs(np.diag(scale_matrix_p)))
                if np.isfinite(current_panel_scale) and current_panel_scale > 1e-10:
                    avg_panel_pixel_scale_deg += current_panel_scale
                    valid_scales_count += 1
            except Exception as scale_err_loop:
                print(f"      - Warning: Échec lecture échelle pixel panneau: {scale_err_loop}")
        
        if valid_scales_count > 0:
            avg_panel_pixel_scale_deg /= valid_scales_count
        elif hasattr(ref_wcs_for_output_params, 'pixel_scale_matrix'): # Fallback sur le premier
             avg_panel_pixel_scale_deg = np.mean(np.abs(np.diag(ref_wcs_for_output_params.pixel_scale_matrix)))
        else: # Fallback ultime très grossier (ex: 1 arcsec/pix)
            print("      - ERREUR: Impossible de déterminer l'échelle des panneaux. Utilisation d'une valeur par défaut grossière.")
            avg_panel_pixel_scale_deg = 1.0 / 3600.0 # 1 arcsec en degrés

        output_pixel_scale_deg = avg_panel_pixel_scale_deg / drizzle_scale_factor
        print(f"      - Échelle Pixel Moyenne Panneaux: {avg_panel_pixel_scale_deg * 3600:.3f} arcsec/pix")
        print(f"      - Échelle Pixel Sortie Mosaïque: {output_pixel_scale_deg * 3600:.3f} arcsec/pix (Facteur Drizzle: {drizzle_scale_factor}x)")
        
        # Définir la matrice CD pour l'échelle et l'orientation (pas de rotation/skew assumé ici)
        # Le signe négatif pour CD1_1 car RA augmente vers la gauche en convention image
        output_wcs.wcs.cd = np.array([[-output_pixel_scale_deg, 0.0],
                                      [0.0, output_pixel_scale_deg]])

        # --- 5. Calcul de la Shape de Sortie (Dimensions en Pixels) ---
        #    Projeter tous les coins de tous les panneaux sur la nouvelle grille WCS de sortie
        #    pour trouver les étendues min/max en pixels.
        print("   -> Calcul de la shape de sortie (dimensions en pixels)...")
        all_output_pixels_x_valid = []
        all_output_pixels_y_valid = []
        projection_errors_count = 0

        for i_fp, panel_footprint_sky in enumerate(all_panel_footprints_sky):
            try:
                # Projeter les coins du footprint céleste sur la grille WCS de sortie
                pixels_out_x_panel, pixels_out_y_panel = output_wcs.world_to_pixel(panel_footprint_sky)
                
                # Filtrer les NaN/Inf qui peuvent résulter de projections hors du domaine du WCS
                valid_x_mask_panel = np.isfinite(pixels_out_x_panel)
                valid_y_mask_panel = np.isfinite(pixels_out_y_panel)
                valid_mask_combined_panel = valid_x_mask_panel & valid_y_mask_panel
                
                all_output_pixels_x_valid.extend(pixels_out_x_panel[valid_mask_combined_panel])
                all_output_pixels_y_valid.extend(pixels_out_y_panel[valid_mask_combined_panel])
                
                if not np.all(valid_mask_combined_panel):
                    num_invalid_corners = len(pixels_out_x_panel) - np.sum(valid_mask_combined_panel)
                    print(f"      - WARNING: Footprint Panneau {i_fp+1}: {num_invalid_corners} coin(s) projeté(s) hors limites (NaN/Inf).")
                    projection_errors_count += 1
            except Exception as proj_err:
                 print(f"      - WARNING: Échec projection coins footprint panneau {i_fp+1}: {proj_err}.")
                 projection_errors_count += 1
        
        if not all_output_pixels_x_valid or not all_output_pixels_y_valid:
            print("ERREUR [MosaicGridOptim]: Aucun coin de panneau valide projeté sur la grille de sortie après filtrage NaN/Inf.")
            return None, None
        if projection_errors_count > 0:
             print(f"   -> INFO: Erreurs de projection ou points hors limites rencontrés pour {projection_errors_count} footprints de panneaux.")

        # Coordonnées pixel min/max dans le système de la grille de sortie
        x_min_output_grid = np.min(all_output_pixels_x_valid)
        x_max_output_grid = np.max(all_output_pixels_x_valid)
        y_min_output_grid = np.min(all_output_pixels_y_valid)
        y_max_output_grid = np.max(all_output_pixels_y_valid)

        # Vérifier si les limites sont finies (sécurité)
        if not all(np.isfinite([x_min_output_grid, x_max_output_grid, y_min_output_grid, y_max_output_grid])):
            print("ERREUR [MosaicGridOptim]: Les limites min/max calculées pour la grille de sortie ne sont pas finies.")
            return None, None

        # Calculer la largeur et la hauteur en pixels (ajouter 1 car indices 0-based)
        # Utiliser np.ceil pour s'assurer que tous les pixels extrêmes sont inclus.
        output_width_pixels = int(np.ceil(x_max_output_grid - x_min_output_grid + 1))
        output_height_pixels = int(np.ceil(y_max_output_grid - y_min_output_grid + 1))
        
        # Assurer une taille minimale pour la grille de sortie
        output_width_pixels = max(10, output_width_pixels)
        output_height_pixels = max(10, output_height_pixels)
        output_shape_hw = (output_height_pixels, output_width_pixels) # Ordre (H, W) pour NumPy
        print(f"      - Dimensions Finales Mosaïque (Largeur, Hauteur) en pixels: ({output_width_pixels}, {output_height_pixels})")

        # --- 6. Finalisation du WCS de Sortie ---
        #    Ajuster CRPIX pour qu'il corresponde au centre de la mosaïque (median_ra_deg, median_dec_deg)
        #    dans le système de coordonnées de la grille de sortie (0-based index).
        #    Le pixel (0,0) de la grille de sortie correspond à (x_min_output_grid, y_min_output_grid)
        #    dans le système intermédiaire calculé par world_to_pixel().
        #    CRPIX (1-based FITS) = (coord_centre_interm - coord_min_interm + 1.0)
        try:
            center_x_intermediate, center_y_intermediate = output_wcs.world_to_pixel(
                SkyCoord(ra=median_ra_deg * u.deg, dec=median_dec_deg * u.deg)
            )
            output_wcs.wcs.crpix = [
                center_x_intermediate - x_min_output_grid + 1.0, # CRPIX1 (X)
                center_y_intermediate - y_min_output_grid + 1.0  # CRPIX2 (Y)
            ]
        except Exception as crpix_err:
            print(f"      - WARNING: Échec ajustement CRPIX du WCS de sortie: {crpix_err}. Utilisation du centre de la grille.")
            output_wcs.wcs.crpix = [output_width_pixels / 2.0 + 0.5, output_height_pixels / 2.0 + 0.5]
        
        # Définir la shape pour l'objet WCS Astropy (W,H)
        output_wcs.pixel_shape = (output_width_pixels, output_height_pixels)
        # Mettre à jour les attributs NAXIS internes si possible (bonne pratique pour certaines versions d'Astropy)
        try:
            output_wcs._naxis1 = output_width_pixels
            output_wcs._naxis2 = output_height_pixels
        except AttributeError:
            pass # Ignorer si les attributs n'existent pas

        print(f"      - WCS de Sortie Finalisé: CRPIX={output_wcs.wcs.crpix}, PixelShape={output_wcs.pixel_shape}")
        print(f"DEBUG [MosaicGridOptim]: Calcul de la grille mosaïque terminé avec succès.")
        return output_wcs, output_shape_hw # Retourne WCS et shape (H, W)

    except Exception as e_grid_calc:
        print(f"ERREUR [MosaicGridOptim]: Échec global lors du calcul de la grille mosaïque: {e_grid_calc}")
        traceback.print_exc(limit=3)
        return None, None



# --- DANS seestar/enhancement/mosaic_processor.py ---

def process_mosaic_from_aligned_files(
        aligned_files_info_list: list,
        queued_stacker_instance, # Type SeestarQueuedStacker
        progress_callback: callable):
    """
    Assemble une mosaïque à partir d'une liste d'informations sur les panneaux.
    MODIFIÉ: Utilise la grille Drizzle pré-calculée si disponible (mosaïque locale),
             sinon la calcule (mosaïque Astrometry).

    Args:
        aligned_files_info_list (list): Liste des informations des panneaux.
        queued_stacker_instance (SeestarQueuedStacker): Instance du gestionnaire de file.
        progress_callback (callable): Fonction pour les messages de progression.

    Returns:
        tuple: (final_mosaic_data_HWC_normalized_0_1, final_mosaic_header) ou (None, None)
    """
    print(f"DEBUG [MosaicProc V2]: Début process_mosaic_from_aligned_files avec {len(aligned_files_info_list)} items.")
    # ... (vérifications initiales inchangées) ...
    if not progress_callback: progress_callback = lambda msg, prog=None: print(f"MOSAIC_PROC_LOG: {msg}")
    if not aligned_files_info_list:
        progress_callback("Mosaïque V2: Aucune information de panneau fournie.", 0)
        return None, None
    if not _DRIZZLE_PROC_AVAILABLE:
        progress_callback("Mosaïque V2 ERREUR: DrizzleProcessor non disponible.", 0)
        return None, None
    if SeestarQueuedStacker is not None and not isinstance(queued_stacker_instance, SeestarQueuedStacker): # Vérifier type
        progress_callback("Mosaïque V2 ERREUR: Instance de QueuedStacker invalide.", 0)
        print("ERREUR [MosaicProc V2]: queued_stacker_instance n'est pas du type SeestarQueuedStacker attendu.")
        return None, None

    # --- Déterminer si on est en mode alignement local (pour savoir si la grille est pré-calculée) ---
    # Cette information est implicitement contenue dans queued_stacker_instance.drizzle_output_wcs
    # Si non-None, elle a été calculée par _worker (cas local OMBB).
    # Si None, on est en mode Astrometry et il faut la calculer ici.
    is_grid_precalculated = (
        hasattr(queued_stacker_instance, 'drizzle_output_wcs') and
        queued_stacker_instance.drizzle_output_wcs is not None and
        hasattr(queued_stacker_instance, 'drizzle_output_shape_hw') and
        queued_stacker_instance.drizzle_output_shape_hw is not None
    )
    print(f"DEBUG [MosaicProc V2]: Grille Drizzle pré-calculée (locale OMBB)? {'Oui' if is_grid_precalculated else 'Non (Astrometry, à calculer)'}")

    # --- 1. Obtenir/Calculer la grille Drizzle de sortie ---
    output_wcs_mosaic = None
    output_shape_mosaic_hw = None

    if is_grid_precalculated:
        progress_callback("Mosaïque V2: Utilisation de la grille de sortie pré-calculée (OMBB)...", 15)
        output_wcs_mosaic = queued_stacker_instance.drizzle_output_wcs
        output_shape_mosaic_hw = queued_stacker_instance.drizzle_output_shape_hw
        print(f"DEBUG [MosaicProc V2]: Grille pré-calculée (OMBB) utilisée. Shape WCS: {output_wcs_mosaic.pixel_shape if output_wcs_mosaic else 'None'}, Shape HW: {output_shape_mosaic_hw}")
    else: # Mosaïque Astrometry, calculer la grille ici
        progress_callback("Mosaïque V2: Calcul de la grille de sortie (Astrometry par panneau)...", 10)
        # `aligned_files_info_list` contient (img_data_aligned_astroalign, header_orig, scores, wcs_indiv_absolu, valid_mask_aligned)
        input_wcs_list_astrometry = []
        panel_shapes_hw_list_astrometry = [] # Pour la version _optimized
        
        for item_tuple in aligned_files_info_list:
            img_data = item_tuple[0] # image (alignée ou originale)
            wcs_indiv = item_tuple[3]  # wcs_indiv_absolu ou M_matrix
            
            # On s'attend à ce que pour le mode Astrometry, item_tuple[3] soit un WCS
            if img_data is not None and wcs_indiv is not None and isinstance(wcs_indiv, WCS):
                input_wcs_list_astrometry.append(wcs_indiv)
                panel_shapes_hw_list_astrometry.append(img_data.shape[:2]) # (H,W)
            else:
                panel_name_for_log = item_tuple[1].get('FILENAME', f'Panneau Inconnu #{len(input_wcs_list_astrometry)}') if len(item_tuple)>1 and item_tuple[1] else f'Panneau Inconnu #{len(input_wcs_list_astrometry)}'
                print(f"WARN [MosaicProc V2]: Item pour grille Astrometry sans image ou WCS valide : {panel_name_for_log}. Ignoré pour calcul grille.")

        if not input_wcs_list_astrometry:
            progress_callback("Mosaïque V2 ERREUR: Aucun WCS individuel valide trouvé pour le mode Astrometry.", 0)
            print("ERREUR [MosaicProc V2]: input_wcs_list_astrometry (Astrometry) est vide.")
            return None, None
        
        # Utiliser la méthode _calculate_final_mosaic_grid_optimized qui est sur queued_stacker_instance
        # Elle prend `panel_wcs_list`, `panel_shapes_hw_list`, `drizzle_scale_factor`
        drizzle_scale_factor_for_grid_calc = getattr(queued_stacker_instance, 'drizzle_scale', 2.0)
        if hasattr(queued_stacker_instance, '_calculate_final_mosaic_grid_optimized'):
            output_wcs_mosaic, output_shape_mosaic_hw = queued_stacker_instance._calculate_final_mosaic_grid_optimized(
                input_wcs_list_astrometry,
                panel_shapes_hw_list_astrometry,
                drizzle_scale_factor_for_grid_calc
            )
        else: # Fallback sur l'ancienne _calculate_final_mosaic_grid si la nouvelle n'est pas là
            print("WARN [MosaicProc V2]: _calculate_final_mosaic_grid_optimized non trouvée. Utilisation de _calculate_final_mosaic_grid.")
            output_wcs_mosaic, output_shape_mosaic_hw = queued_stacker_instance._calculate_final_mosaic_grid(
                input_wcs_list_astrometry 
            )
        progress_callback("Mosaïque V2: Grille de sortie (Astrometry) calculée.", 20)
        print(f"DEBUG [MosaicProc V2]: Grille Drizzle (Astrometry) -> WCS: {'OK' if output_wcs_mosaic else 'None'}, Shape HW: {output_shape_mosaic_hw}")

    if output_wcs_mosaic is None or output_shape_mosaic_hw is None:
        progress_callback("Mosaïque V2 ERREUR: Échec du calcul/obtention de la grille Drizzle de sortie.", 0)
        print("ERREUR [MosaicProc V2]: output_wcs_mosaic ou output_shape_mosaic_hw est None après calcul/récupération grille.")
        return None, None

    # --- 2. Préparer les fichiers temporaires pour DrizzleProcessor ---
    #    (Cette logique reste globalement la même, mais il faut s'assurer
    #     que `use_local_aligner` est correctement déterminé ici pour les logs et la gestion des headers)
    use_local_aligner_for_temp_files = ( # Déterminer à nouveau si on est en mode local pour la logique des fichiers temp
        hasattr(queued_stacker_instance, 'is_local_alignment_preferred_for_mosaic') and
        queued_stacker_instance.is_local_alignment_preferred_for_mosaic and
        hasattr(queued_stacker_instance, 'local_aligner_instance') and
        queued_stacker_instance.local_aligner_instance is not None
    )
    print(f"DEBUG [MosaicProc V2]: Préparation fichiers temporaires, mode local utilisé: {use_local_aligner_for_temp_files}")
    
    progress_callback("Mosaïque V2: Préparation des fichiers temporaires pour Drizzle...", 30)
    temp_drizzle_input_files = []
    temp_dir_for_mosaic_drizzle = os.path.join(queued_stacker_instance.output_folder, "temp_mosaic_drizzle_inputs")
    try:
        os.makedirs(temp_dir_for_mosaic_drizzle, exist_ok=True)
    except OSError as e_mkdir:
        progress_callback(f"Mosaïque V2 ERREUR: Impossible de créer le dossier temporaire Drizzle: {e_mkdir}", 0)
        print(f"ERREUR [MosaicProc V2]: mkdir {temp_dir_for_mosaic_drizzle} échoué: {e_mkdir}")
        return None, None

    for i, item_tuple in enumerate(aligned_files_info_list):
        panel_image_data_hwc = item_tuple[0] 
        panel_header_orig = item_tuple[1]    
        # --- Créer un header temporaire MINIMAL ---
        temp_header_for_file = fits.Header()
        # Ajouter EXPTIME si disponible (Drizzle l'utilise)
        if panel_header_orig and 'EXPTIME' in panel_header_orig:
            try:
                temp_header_for_file['EXPTIME'] = (float(panel_header_orig['EXPTIME']), "Exposure time")
            except (ValueError, TypeError):
                temp_header_for_file['EXPTIME'] = (1.0, "Exposure time (default)")
        else:
            temp_header_for_file['EXPTIME'] = (1.0, "Exposure time (default)")

        # Ajouter le WCS et la matrice M
        if use_local_aligner_for_temp_files:
            wcs_ancre_pour_ce_panneau = item_tuple[2] 
            matrix_m = item_tuple[3]
            
            if wcs_ancre_pour_ce_panneau is None or matrix_m is None:
                continue
        
        # Nettoyer ancien WCS du header original
        wcs_keys_to_clean = WCS().to_header(relax=True).keys() # Obtenir une liste de clés WCS typiques
        for key_wcs_clean in wcs_keys_to_clean:
            if key_wcs_clean in temp_header_for_file: del temp_header_for_file[key_wcs_clean]

        if use_local_aligner_for_temp_files:
            # item: (img_data_orig_HWC, header_orig, wcs_ref_panel_absolu_pour_ce_panneau, matrix_M_vers_ref, valid_mask_orig)
            # Le wcs_ref_panel_absolu_pour_ce_panneau EST self.reference_wcs_object (l'ancre)
            # La matrice M est item_tuple[3]
            wcs_ancre_pour_ce_panneau = item_tuple[2] # Devrait être queued_stacker_instance.reference_wcs_object
            matrix_m = item_tuple[3]
            
            if wcs_ancre_pour_ce_panneau is None:
                progress_callback(f"Mosaïque V2 (Local): WCS d'ancre manquant pour panneau {i}, ignoré pour Drizzle.", None)
                continue
            if matrix_m is None: # Ne devrait pas arriver si FastAligner a réussi
                progress_callback(f"Mosaïque V2 (Local): Matrice M manquante pour panneau {i}, ignoré pour Drizzle.", None)
                continue

            temp_header_for_file.update(wcs_ancre_pour_ce_panneau.to_header(relax=True))
            try:
                temp_header_for_file['M11'] = matrix_m[0,0]
                temp_header_for_file['M12'] = matrix_m[0,1]
                temp_header_for_file['M13'] = matrix_m[0,2]
                temp_header_for_file['M21'] = matrix_m[1,0]
                temp_header_for_file['M22'] = matrix_m[1,1]
                temp_header_for_file['M23'] = matrix_m[1,2]
                #temp_header_for_file['HISTORY'] = "Local alignment matrix M to ref panel stored."
            except Exception as e_hdr_m: print(f"WARN [MosaicProc V2]: Erreur écriture Matrice M dans header temp: {e_hdr_m}")
        
        else: # Mosaïque Astrometry.net pour chaque panneau
            # item: (img_data_aligned_astroalign, header_orig, scores, wcs_indiv_absolu, valid_mask_aligned)
            wcs_indiv_absolu_pour_ce_panneau = item_tuple[3] # C'est le wcs_indiv_absolu
            if wcs_indiv_absolu_pour_ce_panneau is None:
                progress_callback(f"Mosaïque V2 (Astrometry): WCS individuel manquant pour panneau {i}, ignoré pour Drizzle.", None)
                continue
            temp_header_for_file.update(wcs_indiv_absolu_pour_ce_panneau.to_header(relax=True))

        if panel_image_data_hwc is None:
            progress_callback(f"Mosaïque V2: Données image manquantes pour panneau {i}, ignoré.", None)
            continue
        
        temp_fits_path = os.path.join(temp_dir_for_mosaic_drizzle, f"panel_temp_driz_in_{i:03d}.fits")
        try:
            data_to_save_cxhxw = np.moveaxis(panel_image_data_hwc, -1, 0).astype(np.float32)
            temp_header_for_file['NAXIS'] = 3
            temp_header_for_file['NAXIS1'] = data_to_save_cxhxw.shape[2]; temp_header_for_file['NAXIS2'] = data_to_save_cxhxw.shape[1]; temp_header_for_file['NAXIS3'] = data_to_save_cxhxw.shape[0]
            
            fits.writeto(temp_fits_path, data_to_save_cxhxw, header=temp_header_for_file, overwrite=True, output_verify='ignore')
            temp_drizzle_input_files.append(temp_fits_path)
            # print(f"  DEBUG [MosaicProc V2]: Fichier temporaire sauvegardé: {os.path.basename(temp_fits_path)}")
        except Exception as e_write_temp:
            progress_callback(f"Mosaïque V2 ERREUR: Écriture fichier temp {temp_fits_path} échouée: {e_write_temp}", None)
            traceback.print_exc(limit=1)
    
    if not temp_drizzle_input_files:
        progress_callback("Mosaïque V2 ERREUR: Aucun fichier temporaire n'a pu être préparé pour Drizzle.", 0)
        return None, None
    
    progress_callback(f"Mosaïque V2: {len(temp_drizzle_input_files)} fichiers temporaires prêts pour Drizzle.", 40)

    # --- 3. Lancer DrizzleProcessor ---
    progress_callback("Mosaïque V2: Lancement du processeur Drizzle...", 50)
    # Lire les mosaic_settings depuis l'instance queued_stacker
    mosaic_settings_from_qs = getattr(queued_stacker_instance, 'mosaic_settings', {})
    
    drizzle_processor = DrizzleProcessor(
        scale_factor=getattr(queued_stacker_instance, 'drizzle_scale', 2.0), # Valeur par défaut si non trouvé
        pixfrac=mosaic_settings_from_qs.get('pixfrac', 0.8),
        kernel=mosaic_settings_from_qs.get('kernel', 'square'),
        fillval=str(mosaic_settings_from_qs.get('fillval', "0.0")), 
        final_wht_threshold=float(mosaic_settings_from_qs.get('wht_threshold', 0.01)) 
    )

    final_mosaic_sci_hxwxc = None
    final_mosaic_wht_hxwxc = None

    try:
        anchor_wcs_to_pass = None
        if use_local_aligner_for_temp_files: # C'est bien le flag local, pas is_grid_precalculated
            anchor_wcs_to_pass = getattr(queued_stacker_instance, 'reference_wcs_object', None)
            if anchor_wcs_to_pass is None:
                 progress_callback("Mosaïque V2 ERREUR (Local): WCS d'ancrage manquant pour DrizzleProcessor.apply_drizzle.", 0)
                 raise ValueError("WCS d'ancrage manquant pour DrizzleProcessor en mode local.")
            progress_callback("Mosaïque V2: Drizzle en mode alignement local (construction pixmaps par DrizzleProcessor)...", 60)
        else: # Mosaïque Astrometry.net pour chaque
            progress_callback("Mosaïque V2: Drizzle en mode Astrometry (WCS individuels)...", 60)

        final_mosaic_sci_hxwxc, final_mosaic_wht_hxwxc = drizzle_processor.apply_drizzle(
            input_file_paths=temp_drizzle_input_files,
            output_wcs=output_wcs_mosaic,
            output_shape_2d_hw=output_shape_mosaic_hw,
            use_local_alignment_logic=use_local_aligner_for_temp_files, 
            anchor_wcs_for_local=anchor_wcs_to_pass, 
            progress_callback=progress_callback
        )

    # ... (Reste de la fonction : nettoyage, création header final, normalisation - inchangé par rapport à votre version précédente) ...
    except Exception as e_driz:
        progress_callback(f"Mosaïque V2 ERREUR: Échec du processeur Drizzle: {e_driz}", 0)
        traceback.print_exc(limit=2)
        # Assurer le nettoyage même en cas d'erreur Drizzle
        for f_path_clean in temp_drizzle_input_files:
            if os.path.exists(f_path_clean):
                try: os.remove(f_path_clean)
                except Exception: pass
        if os.path.exists(temp_dir_for_mosaic_drizzle) and not os.listdir(temp_dir_for_mosaic_drizzle):
            try: os.rmdir(temp_dir_for_mosaic_drizzle)
            except Exception: pass
        return None, None

    progress_callback("Mosaïque V2: Nettoyage des fichiers temporaires Drizzle...", 95)
    for f_path in temp_drizzle_input_files:
        try:
            if os.path.exists(f_path): os.remove(f_path)
        except Exception as e_clean: print(f"WARN [MosaicProc V2]: Erreur nettoyage fichier temp {f_path}: {e_clean}")
    try:
        if os.path.exists(temp_dir_for_mosaic_drizzle) and not os.listdir(temp_dir_for_mosaic_drizzle):
            os.rmdir(temp_dir_for_mosaic_drizzle)
            print(f"DEBUG [MosaicProc V2]: Dossier temporaire {temp_dir_for_mosaic_drizzle} supprimé.")
    except Exception as e_rmdir:
         print(f"WARN [MosaicProc V2]: Erreur suppression dossier temporaire {temp_dir_for_mosaic_drizzle}: {e_rmdir}")

    if final_mosaic_sci_hxwxc is None:
        progress_callback("Mosaïque V2 ERREUR: Le processeur Drizzle n'a pas retourné d'image science.", 0)
        return None, None

    progress_callback("Mosaïque V2: Assemblage Drizzle terminé.", 100)
    print(f"DEBUG [MosaicProc V2]: Drizzle terminé. Shape SCI: {final_mosaic_sci_hxwxc.shape if final_mosaic_sci_hxwxc is not None else 'None'}")

    # --- 4. Créer le header final pour la mosaïque ---
    final_header_mosaic = fits.Header()
    if output_wcs_mosaic:
        try: 
            # S'assurer que les CTYPE sont bien des strings avant de les mettre à jour
            # Cela peut arriver si le WCS original avait des CTYPE non-standards que WCS() a tenté de corriger
            # mais qui ne sont pas des strings pures.
            if hasattr(output_wcs_mosaic.wcs, 'ctype'):
                output_wcs_mosaic.wcs.ctype = [str(ct) for ct in output_wcs_mosaic.wcs.ctype]
            
            final_header_mosaic.update(output_wcs_mosaic.to_header(relax=True))
        except Exception as e_hdr_wcs: 
            print(f"WARN [MosaicProc V2_HeaderFix]: Erreur ajout WCS au header final: {e_hdr_wcs}")
            traceback.print_exc(limit=1) # Pour plus de détails sur l'erreur WCS si elle se produit
            
    # S'assurer que ref_header_for_meta est bien un Header Astropy
    ref_header_for_meta = None
    if aligned_files_info_list and len(aligned_files_info_list[0]) > 1 and \
       isinstance(aligned_files_info_list[0][1], fits.Header):
        ref_header_for_meta = aligned_files_info_list[0][1] 
    
    if ref_header_for_meta:
        print(f"DEBUG [MosaicProc V2_HeaderFix]: Utilisation de ref_header_for_meta pour les métadonnées.")
        keys_to_copy = ['INSTRUME', 'TELESCOP', 'OBJECT', 'FILTER', 'DATE-OBS', 'EXPTIME']
        for key in keys_to_copy:
            if key in ref_header_for_meta:
                comment_value = "" 
                try:
                    # Tenter d'obtenir le commentaire. Si la clé existe mais n'a pas de commentaire,
                    # header.comments[key] peut lever une erreur ou retourner une valeur non-string.
                    if key in ref_header_for_meta.comments: # Vérifier d'abord si la clé a un commentaire
                         comment_value = ref_header_for_meta.comments[key]
                except (KeyError, IndexError): 
                    pass 
                
                try:
                    final_header_mosaic[key] = (ref_header_for_meta[key], str(comment_value))
                except Exception as e_set_key:
                    print(f"WARN [MosaicProc V2_HeaderFix]: Erreur lors de la copie de la clé '{key}' du header: {e_set_key}")
                    # Tenter d'ajouter juste la valeur si le commentaire pose problème
                    try:
                        final_header_mosaic[key] = ref_header_for_meta[key]
                    except Exception:
                        print(f"WARN [MosaicProc V2_HeaderFix]: Échec copie valeur pour clé '{key}' également.")


    actual_num_inputs_for_header = len(aligned_files_info_list) 
    comment_ninputs = 'Number of panels (or images) in mosaic' 

    if hasattr(queued_stacker_instance, 'total_physical_images_in_mosaic_panels') and \
       isinstance(queued_stacker_instance.total_physical_images_in_mosaic_panels, int) and \
       queued_stacker_instance.total_physical_images_in_mosaic_panels > 0 :
        
        # Mettre à jour actual_num_inputs_for_header si un compte plus précis est disponible
        actual_num_inputs_for_header = queued_stacker_instance.total_physical_images_in_mosaic_panels
        
        final_header_mosaic['NPANELS'] = (len(aligned_files_info_list), 'Number of panels in the mosaic') 
        comment_ninputs = 'Total source images from all panels' 
        print(f"DEBUG [MosaicProc V2_HeaderFix_Unbound]: Utilisation de total_physical_images_in_mosaic_panels: {actual_num_inputs_for_header}")
    else:
        print(f"DEBUG [MosaicProc V2_HeaderFix_Unbound]: Utilisation du nombre de panneaux pour NINPUTS: {actual_num_inputs_for_header}")
    
    # Utiliser TOUJOURS actual_num_inputs_for_header ici
    final_header_mosaic['NINPUTS'] = (actual_num_inputs_for_header, comment_ninputs)

    final_header_mosaic['STACKTYP'] = (f'Mosaic Drizzle ({queued_stacker_instance.drizzle_scale:.1f}x)', 'Stacking method')
    # ... (autres clés du header comme DRZSCALE, DRZKERNEL, etc. - inchangées) ...
    final_header_mosaic['DRZSCALE'] = (queued_stacker_instance.drizzle_scale, 'Drizzle scale factor')
    final_header_mosaic['DRZKERNEL'] = (mosaic_settings_from_qs.get('kernel', 'N/A'), 'Drizzle kernel')
    final_header_mosaic['DRZPIXFR'] = (mosaic_settings_from_qs.get('pixfrac', 'N/A'), 'Drizzle pixfrac')
    final_header_mosaic['CREATOR'] = ('SeestarStacker (Mosaic)', 'Processing Software')
    final_header_mosaic['HISTORY'] = 'Mosaic created by SeestarStacker using Drizzle' 
    if use_local_aligner_for_temp_files: 
        final_header_mosaic['HISTORY'] = 'Panel align: Local FastAligner + 1 ref panel Astrometry.net/Fallback' 
    else:
        final_header_mosaic['HISTORY'] = 'Panel align: Astrometry.net for each panel'
    
    
    # Normalisation finale (inchangée)
    # ...
    print(f"DEBUG [MosaicProc V2_HeaderFix]: Range SCI avant normalisation finale: Min={np.nanmin(final_mosaic_sci_hxwxc):.3g}, Max={np.nanmax(final_mosaic_sci_hxwxc):.3g}")
    min_val, max_val = np.nanmin(final_mosaic_sci_hxwxc), np.nanmax(final_mosaic_sci_hxwxc)
    if max_val > min_val:
        final_mosaic_data_normalized = (final_mosaic_sci_hxwxc - min_val) / (max_val - min_val)
    elif np.any(np.isfinite(final_mosaic_sci_hxwxc)):
        final_mosaic_data_normalized = np.full_like(final_mosaic_sci_hxwxc, 0.5)
    else:
        final_mosaic_data_normalized = np.zeros_like(final_mosaic_sci_hxwxc)

    final_mosaic_data_normalized = np.clip(final_mosaic_data_normalized, 0.0, 1.0).astype(np.float32)
    # print(f"DEBUG [MosaicProc V2_HeaderFix]: Range SCI après normalisation finale 0-1: Min={np.min(final_mosaic_data_normalized):.3f}, Max={np.max(final_mosaic_data_normalized):.3f}")
    
    if hasattr(queued_stacker_instance, 'images_in_cumulative_stack'):
        # Utiliser le nombre d'images sources réelles si disponible (via NINPUTS), sinon le nombre de panneaux.
        num_for_counter = actual_num_inputs_from_panels if 'actual_num_inputs_from_panels' in locals() and actual_num_inputs_from_panels > 0 else len(aligned_files_info_list)
        queued_stacker_instance.images_in_cumulative_stack = num_for_counter
        print(f"DEBUG [MosaicProc V2_HeaderFix]: Compteur images QM mis à jour pour rapport final: {queued_stacker_instance.images_in_cumulative_stack}")

    gc.collect() 
    print("DEBUG [MosaicProc V2_HeaderFix]: Fin de process_mosaic_from_aligned_files.")
    return final_mosaic_data_normalized, final_header_mosaic


# --- FIN DU FICHIER seestar/enhancement/mosaic_processor.py ---