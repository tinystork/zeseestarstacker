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



def process_mosaic_from_aligned_files(
        aligned_files_info_list: list,
        queued_stacker_instance, # Doit être une instance de SeestarQueuedStacker ou avoir les attributs nécessaires
        progress_callback: callable):
    """
    Assemble une mosaïque à partir d'une liste d'informations sur les panneaux.
    Gère soit l'alignement local (panneau réf + M) soit Astrometry.net pour chaque panneau.
    Appelle DrizzleProcessor avec les bons arguments.

    Args:
        aligned_files_info_list (list): Liste de tuples. Le format du tuple dépend du mode:
            - Mosaïque Locale: (img_data_orig_HWC, header_orig, wcs_ref_panel_absolu, matrix_M_vers_ref, valid_mask_orig)
            - Mosaïque Astrometry: (img_data_aligned_astroalign, header_orig, scores, wcs_indiv_absolu, valid_mask_aligned)
        queued_stacker_instance: Instance de SeestarQueuedStacker pour accéder aux paramètres 
                                 (drizzle_scale, mosaic_settings, mosaic_ref_panel_wcs_absolute, etc.)
                                 et aux méthodes de calcul de grille.
        progress_callback (callable): Fonction pour les messages de progression.

    Returns:
        tuple: (final_mosaic_data_HWC_normalized_0_1, final_mosaic_header) ou (None, None)
    """
    print(f"DEBUG [MosaicProc]: Début process_mosaic_from_aligned_files avec {len(aligned_files_info_list)} items.")
    if not progress_callback: 
        progress_callback = lambda msg, prog=None: print(f"MOSAIC_PROC_LOG: {msg}" + (f" ({prog}%)" if prog is not None else ""))

    if not aligned_files_info_list:
        progress_callback("Mosaïque: Aucune information de panneau fournie.", 0)
        return None, None

    # Vérifier si queued_stacker_instance est valide (pourrait être un mock pour tests)
    if SeestarQueuedStacker is not None and not isinstance(queued_stacker_instance, SeestarQueuedStacker):
        progress_callback("Mosaïque ERREUR: Instance de QueuedStacker invalide.", 0)
        print("ERREUR [MosaicProc]: queued_stacker_instance n'est pas du type SeestarQueuedStacker attendu.")
        return None, None

    # --- Déterminer le mode d'alignement utilisé (basé sur les attributs du QueuedStacker) ---
    use_local_aligner = (
        hasattr(queued_stacker_instance, 'is_local_alignment_preferred_for_mosaic') and
        queued_stacker_instance.is_local_alignment_preferred_for_mosaic and
        hasattr(queued_stacker_instance, 'local_aligner_instance') and
        queued_stacker_instance.local_aligner_instance is not None
    )
    print(f"DEBUG [MosaicProc]: Mode d'alignement mosaïque utilisé: {'Local (FastAligner)' if use_local_aligner else 'Astrometry.net pour chaque panneau'}")

    # --- 1. Calculer la grille Drizzle de sortie (output_wcs, output_shape_hw) ---
    progress_callback("Mosaïque: Calcul de la grille de sortie Drizzle...", 10)
    
    output_wcs_mosaic = None
    output_shape_mosaic_hw = None

    if use_local_aligner:
        print("DEBUG [MosaicProc]: Utilisation de la logique pour alignement local pour calculer la grille.")
        if not hasattr(queued_stacker_instance, 'mosaic_ref_panel_wcs_absolute') or \
           queued_stacker_instance.mosaic_ref_panel_wcs_absolute is None:
            progress_callback("Mosaïque ERREUR: WCS du panneau de référence absolu manquant pour l'alignement local.", 0)
            print("ERREUR [MosaicProc]: WCS ancre manquant dans queued_stacker_instance pour grille locale.")
            return None, None
        
        # `aligned_files_info_list` contient les tuples (img_data_orig, header, wcs_anchor, M, mask)
        output_wcs_mosaic, output_shape_mosaic_hw = queued_stacker_instance._calculate_local_mosaic_output_grid(
            aligned_files_info_list, 
            queued_stacker_instance.mosaic_ref_panel_wcs_absolute 
        )
        progress_callback("Mosaïque: Grille de sortie (locale) calculée.", 20)
        print(f"DEBUG [MosaicProc]: Grille Drizzle (locale) -> WCS: {'OK' if output_wcs_mosaic else 'None'}, Shape HxW: {output_shape_mosaic_hw}")

    else: # Mosaïque avec Astrometry.net pour chaque panneau
        print("DEBUG [MosaicProc]: Utilisation de la logique Astrometry.net pour calculer la grille.")
        # aligned_files_info_list contient (img_data_aligned_astroalign, header, scores, wcs_indiv, mask)
        # On a besoin d'une liste de WCS individuels pour _calculate_final_mosaic_grid
        input_wcs_list_astrometry = []
        for item_tuple in aligned_files_info_list:
            if len(item_tuple) >= 4 and item_tuple[3] is not None and isinstance(item_tuple[3], WCS):
                input_wcs_list_astrometry.append(item_tuple[3]) # item[3] est wcs_object_indiv_item
            else:
                print(f"WARN [MosaicProc]: Item dans aligned_files_info_list n'a pas de WCS valide pour mode Astrometry: {item_tuple[1].get('FILENAME', 'Unknown file') if len(item_tuple)>1 and item_tuple[1] else 'Unknown'}")

        if not input_wcs_list_astrometry:
            progress_callback("Mosaïque ERREUR: Aucun WCS individuel valide trouvé pour le mode Astrometry.", 0)
            print("ERREUR [MosaicProc]: input_wcs_list_astrometry est vide.")
            return None, None
            
        output_wcs_mosaic, output_shape_mosaic_hw = queued_stacker_instance._calculate_final_mosaic_grid(
            input_wcs_list_astrometry 
        )
        progress_callback("Mosaïque: Grille de sortie (Astrometry) calculée.", 20)
        print(f"DEBUG [MosaicProc]: Grille Drizzle (Astrometry) -> WCS: {'OK' if output_wcs_mosaic else 'None'}, Shape HxW: {output_shape_mosaic_hw}")

    if output_wcs_mosaic is None or output_shape_mosaic_hw is None:
        progress_callback("Mosaïque ERREUR: Échec du calcul de la grille Drizzle de sortie.", 0)
        print("ERREUR [MosaicProc]: output_wcs_mosaic ou output_shape_mosaic_hw est None après calcul grille.")
        return None, None

    # --- 2. Préparer les fichiers temporaires pour DrizzleProcessor ---
    progress_callback("Mosaïque: Préparation des fichiers temporaires pour Drizzle...", 30)
    temp_drizzle_input_files = []
    temp_dir_for_mosaic_drizzle = os.path.join(queued_stacker_instance.output_folder, "temp_mosaic_drizzle_inputs")
    try:
        os.makedirs(temp_dir_for_mosaic_drizzle, exist_ok=True)
    except OSError as e_mkdir:
        progress_callback(f"Mosaïque ERREUR: Impossible de créer le dossier temporaire Drizzle: {e_mkdir}", 0)
        print(f"ERREUR [MosaicProc]: mkdir {temp_dir_for_mosaic_drizzle} échoué: {e_mkdir}")
        return None, None

    for i, item_tuple in enumerate(aligned_files_info_list):
        panel_image_data_hwc = item_tuple[0] # Toujours l'image (originale pour local, alignée pour astrometry)
        panel_header_orig = item_tuple[1]    # Header original
        panel_wcs_for_drizzle_file = None      # WCS à écrire dans le FITS temporaire
        
        # Préparer le header temporaire. Il contiendra le WCS pertinent et la matrice M si locale.
        temp_header_for_file = panel_header_orig.copy()
        # Nettoyer ancien WCS du header original pour éviter conflits
        for key_wcs_clean in list(temp_header_for_file['CRVAL*'])+list(temp_header_for_file['CRPIX*'])+list(temp_header_for_file['CTYPE*'])+list(temp_header_for_file['CD*'])+list(temp_header_for_file['PC*']):
            if key_wcs_clean in temp_header_for_file: del temp_header_for_file[key_wcs_clean]

        if use_local_aligner:
            # item: (img_data_orig_HWC, header_orig, wcs_ref_panel_absolu, matrix_M_vers_ref, valid_mask_orig)
            # Pour le fichier temporaire, on sauvegarde l'image originale.
            # Le WCS de référence et la matrice M seront utilisés par DrizzleProcessor.
            # On met le WCS de référence dans le header du fichier temp pour que DrizzleProcessor puisse le lire.
            panel_wcs_for_drizzle_file = item_tuple[2] # C'est queued_stacker_instance.mosaic_ref_panel_wcs_absolute
            matrix_m = item_tuple[3]
            
            if panel_wcs_for_drizzle_file:
                temp_header_for_file.update(panel_wcs_for_drizzle_file.to_header(relax=True))
            if matrix_m is not None:
                try:
                    temp_header_for_file['M11'] = matrix_m[0,0]; temp_header_for_file['M12'] = matrix_m[0,1]; temp_header_for_file['M13'] = matrix_m[0,2]
                    temp_header_for_file['M21'] = matrix_m[1,0]; temp_header_for_file['M22'] = matrix_m[1,1]; temp_header_for_file['M23'] = matrix_m[1,2]
                    temp_header_for_file['COMMENT'] = "Local alignment matrix M to ref panel stored."
                except Exception as e_hdr_m: print(f"WARN [MosaicProc]: Erreur écriture Matrice M dans header temp: {e_hdr_m}")
        else: # Astrometry.net pour chaque panneau
            # item: (img_data_aligned_astroalign, header_orig, scores, wcs_indiv_absolu, valid_mask_aligned)
            panel_wcs_for_drizzle_file = item_tuple[3] # C'est le wcs_indiv_absolu
            if panel_wcs_for_drizzle_file:
                temp_header_for_file.update(panel_wcs_for_drizzle_file.to_header(relax=True))

        if panel_image_data_hwc is None:
            progress_callback(f"Mosaïque: Données image manquantes pour panneau {i}, ignoré.", None)
            continue
        if use_local_aligner and panel_wcs_for_drizzle_file is None : # En mode local, on a besoin du WCS ancre pour le pixmap
            progress_callback(f"Mosaïque (Local): WCS d'ancre manquant pour panneau {i}, ignoré pour Drizzle.", None)
            continue
        if not use_local_aligner and panel_wcs_for_drizzle_file is None : # En mode Astrometry, on a besoin du WCS individuel
            progress_callback(f"Mosaïque (Astrometry): WCS individuel manquant pour panneau {i}, ignoré pour Drizzle.", None)
            continue


        temp_fits_path = os.path.join(temp_dir_for_mosaic_drizzle, f"panel_temp_driz_in_{i:03d}.fits")
        try:
            data_to_save_cxhxw = np.moveaxis(panel_image_data_hwc, -1, 0).astype(np.float32)
            temp_header_for_file['NAXIS'] = 3
            temp_header_for_file['NAXIS1'] = data_to_save_cxhxw.shape[2]; temp_header_for_file['NAXIS2'] = data_to_save_cxhxw.shape[1]; temp_header_for_file['NAXIS3'] = data_to_save_cxhxw.shape[0]
            
            fits.writeto(temp_fits_path, data_to_save_cxhxw, header=temp_header_for_file, overwrite=True, output_verify='ignore')
            temp_drizzle_input_files.append(temp_fits_path)
            print(f"  DEBUG [MosaicProc]: Fichier temporaire sauvegardé: {os.path.basename(temp_fits_path)}")
        except Exception as e_write_temp:
            progress_callback(f"Mosaïque ERREUR: Écriture fichier temp {temp_fits_path} échouée: {e_write_temp}", None)
            traceback.print_exc(limit=1)
    
    if not temp_drizzle_input_files:
        progress_callback("Mosaïque ERREUR: Aucun fichier temporaire n'a pu être préparé pour Drizzle.", 0)
        return None, None
    
    progress_callback(f"Mosaïque: {len(temp_drizzle_input_files)} fichiers temporaires prêts pour Drizzle.", 40)

    # --- 3. Lancer DrizzleProcessor ---
    progress_callback("Mosaïque: Lancement du processeur Drizzle...", 50)
    drizzle_processor = DrizzleProcessor(
        scale_factor=queued_stacker_instance.drizzle_scale,
        pixfrac=queued_stacker_instance.mosaic_settings.get('pixfrac', 0.8),
        kernel=queued_stacker_instance.mosaic_settings.get('kernel', 'square'),
        fillval= str(queued_stacker_instance.mosaic_settings.get('fillval', "0.0")), # S'assurer que c'est une string
        final_wht_threshold=float(queued_stacker_instance.mosaic_settings.get('wht_threshold', 0.01)) # S'assurer que c'est un float
    )

    final_mosaic_sci_hxwxc = None
    final_mosaic_wht_hxwxc = None

    try:
        # L'appel à apply_drizzle est maintenant conditionnel
        if use_local_aligner:
            progress_callback("Mosaïque: Drizzle en mode alignement local (construction pixmaps par DrizzleProcessor)...", 60)
            final_mosaic_sci_hxwxc, final_mosaic_wht_hxwxc = drizzle_processor.apply_drizzle(
                input_file_paths=temp_drizzle_input_files,    # Contient img_orig + wcs_ref_panel + M dans header
                output_wcs=output_wcs_mosaic,
                output_shape_2d_hw=output_shape_mosaic_hw,
                use_local_alignment_logic=True, 
                anchor_wcs_for_local=queued_stacker_instance.mosaic_ref_panel_wcs_absolute, 
                progress_callback=progress_callback
            )
        else: # Mosaïque Astrometry.net pour chaque
            progress_callback("Mosaïque: Drizzle en mode Astrometry (WCS individuels)...", 60)
            final_mosaic_sci_hxwxc, final_mosaic_wht_hxwxc = drizzle_processor.apply_drizzle(
                input_file_paths=temp_drizzle_input_files, # Contient img_alignée_astroalign + wcs_indiv_absolu
                output_wcs=output_wcs_mosaic,
                output_shape_2d_hw=output_shape_mosaic_hw,
                use_local_alignment_logic=False, 
                anchor_wcs_for_local=None,     
                progress_callback=progress_callback
            )

    except Exception as e_driz:
        progress_callback(f"Mosaïque ERREUR: Échec du processeur Drizzle: {e_driz}", 0)
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
    # finally: # Le finally original a été déplacé après le try/except de Drizzle
    #          # pour s'assurer qu'il s'exécute même si DrizzleProcessor lève une exception.
    #    pass

    # Nettoyage des fichiers temporaires créés pour DrizzleProcessor (déplacé ici)
    progress_callback("Mosaïque: Nettoyage des fichiers temporaires Drizzle...", 95)
    for f_path in temp_drizzle_input_files:
        try:
            if os.path.exists(f_path): os.remove(f_path)
        except Exception as e_clean: print(f"WARN [MosaicProc]: Erreur nettoyage fichier temp {f_path}: {e_clean}")
    try:
        if os.path.exists(temp_dir_for_mosaic_drizzle) and not os.listdir(temp_dir_for_mosaic_drizzle):
            os.rmdir(temp_dir_for_mosaic_drizzle)
            print(f"DEBUG [MosaicProc]: Dossier temporaire {temp_dir_for_mosaic_drizzle} supprimé.")
    except Exception as e_rmdir:
         print(f"WARN [MosaicProc]: Erreur suppression dossier temporaire {temp_dir_for_mosaic_drizzle}: {e_rmdir}")


    if final_mosaic_sci_hxwxc is None:
        progress_callback("Mosaïque ERREUR: Le processeur Drizzle n'a pas retourné d'image science.", 0)
        return None, None

    progress_callback("Mosaïque: Assemblage Drizzle terminé.", 100)
    print(f"DEBUG [MosaicProc]: Drizzle terminé. Shape SCI: {final_mosaic_sci_hxwxc.shape if final_mosaic_sci_hxwxc is not None else 'None'}")

    # --- 4. Créer le header final pour la mosaïque ---
    final_header_mosaic = fits.Header()
    if output_wcs_mosaic:
        try: final_header_mosaic.update(output_wcs_mosaic.to_header(relax=True))
        except Exception as e_hdr_wcs: print(f"WARN [MosaicProc]: Erreur ajout WCS au header final: {e_hdr_wcs}")
            
    ref_header_for_meta = aligned_files_info_list[0][1] 
    if ref_header_for_meta:
        keys_to_copy = ['INSTRUME', 'TELESCOP', 'OBJECT', 'FILTER', 'DATE-OBS', 'EXPTIME']
        for key in keys_to_copy:
            if key in ref_header_for_meta:
                final_header_mosaic[key] = (ref_header_for_meta[key], ref_header_for_meta.comments.get(key, "")) # Utiliser .get pour commentaire

    actual_num_inputs = len(aligned_files_info_list) # Nombre de panneaux
    if hasattr(queued_stacker_instance, 'total_physical_images_in_mosaic_panels') and \
       queued_stacker_instance.total_physical_images_in_mosaic_panels > 0 :
        # Si on a un compte plus précis du nombre total d'images sources des panneaux (si chaque panneau est un stack)
        actual_num_inputs = queued_stacker_instance.total_physical_images_in_mosaic_panels
        final_header_mosaic['NINPUTS'] = (actual_num_inputs, 'Nombre total dimages sources des panneaux')
        final_header_mosaic['NPANELS'] = (len(aligned_files_info_list), 'Nombre de panneaux dans la mosaïque')
    else:
        final_header_mosaic['NINPUTS'] = (actual_num_inputs, 'Nombre de panneaux (ou images) dans la mosaïque')


    final_header_mosaic['STACKTYP'] = (f'Mosaic Drizzle ({queued_stacker_instance.drizzle_scale:.1f}x)', 'Stacking method')
    final_header_mosaic['DRZSCALE'] = (queued_stacker_instance.drizzle_scale, 'Drizzle scale factor')
    # Utiliser getattr pour les mosaic_settings pour éviter AttributeError si non défini
    final_header_mosaic['DRZKERNEL'] = (getattr(queued_stacker_instance, 'mosaic_settings', {}).get('kernel', 'N/A'), 'Drizzle kernel')
    final_header_mosaic['DRZPIXFR'] = (getattr(queued_stacker_instance, 'mosaic_settings', {}).get('pixfrac', 'N/A'), 'Drizzle pixfrac')
    final_header_mosaic['CREATOR'] = ('SeestarStacker (Mosaic)', 'Processing Software')
    final_header_mosaic['HISTORY'] = 'Mosaic created by SeestarStacker using Drizzle'
    if use_local_aligner:
        final_header_mosaic['HISTORY'] = 'Panel align: Local FastAligner + 1 ref panel Astrometry.net'
    else:
        final_header_mosaic['HISTORY'] = 'Panel align: Astrometry.net for each panel'
    
    print(f"DEBUG [MosaicProc]: Range SCI avant normalisation finale: Min={np.nanmin(final_mosaic_sci_hxwxc):.3g}, Max={np.nanmax(final_mosaic_sci_hxwxc):.3g}")
    min_val, max_val = np.nanmin(final_mosaic_sci_hxwxc), np.nanmax(final_mosaic_sci_hxwxc)
    if max_val > min_val:
        final_mosaic_data_normalized = (final_mosaic_sci_hxwxc - min_val) / (max_val - min_val)
    elif np.any(np.isfinite(final_mosaic_sci_hxwxc)):
        final_mosaic_data_normalized = np.full_like(final_mosaic_sci_hxwxc, 0.5)
    else:
        final_mosaic_data_normalized = np.zeros_like(final_mosaic_sci_hxwxc)

    final_mosaic_data_normalized = np.clip(final_mosaic_data_normalized, 0.0, 1.0).astype(np.float32)
    print(f"DEBUG [MosaicProc]: Range SCI après normalisation finale 0-1: Min={np.min(final_mosaic_data_normalized):.3f}, Max={np.max(final_mosaic_data_normalized):.3f}")
    
    if hasattr(queued_stacker_instance, 'images_in_cumulative_stack'):
        queued_stacker_instance.images_in_cumulative_stack = actual_num_inputs # Mettre à jour avec le bon compte
        print(f"DEBUG [MosaicProc]: Compteur images QM mis à jour pour rapport final: {queued_stacker_instance.images_in_cumulative_stack}")

    gc.collect() # Un petit nettoyage avant de retourner
    print("DEBUG [MosaicProc]: Fin de process_mosaic_from_aligned_files.")
    return final_mosaic_data_normalized, final_header_mosaic

# --- FIN DU FICHIER seestar/enhancement/mosaic_processor.py (pour cette fonction) ---

def _save_panel_stack_temp(panel_stack_data, solved_wcs, panel_index, output_folder):
     # ... (Code complet comme à l'étape 22) ...
     # ... (S'assurer qu'elle retourne bien le chemin ou None) ...
     if panel_stack_data is None or solved_wcs is None: return None
     temp_dir = os.path.join(output_folder, "mosaic_panel_stacks_temp"); os.makedirs(temp_dir, exist_ok=True)
     temp_filename = f"panel_stack_{panel_index:03d}_solved.fits"; temp_filepath = os.path.join(temp_dir, temp_filename)
     try:
         print(f"      -> Sauvegarde stack panneau temp: {temp_filename}")
         data_to_save = np.moveaxis(panel_stack_data, -1, 0).astype(np.float32) # CxHxW
         header_to_save = solved_wcs.to_header(relax=True); header_to_save['HISTORY'] = f"Stacked Panel {panel_index}"; header_to_save['NAXIS'] = 3
         header_to_save['NAXIS1'] = data_to_save.shape[2]; header_to_save['NAXIS2'] = data_to_save.shape[1]; header_to_save['NAXIS3'] = data_to_save.shape[0]; header_to_save['CTYPE3'] = 'CHANNEL'
         fits.writeto(temp_filepath, data_to_save, header=header_to_save, overwrite=True, output_verify='ignore')
         return temp_filepath
     except Exception as e: print(f"      -> ERREUR sauvegarde stack panneau temp {temp_filename}: {e}"); return None


# --- FIN DU FICHIER seestar/enhancement/mosaic_processor.py ---