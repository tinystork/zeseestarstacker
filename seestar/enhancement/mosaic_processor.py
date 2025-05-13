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
try:
    from ..queuep.queue_manager import SeestarQueuedStacker
except ImportError:
    print("ERREUR [MosaicProcessor Import]: SeestarQueuedStacker manquant.")
    class SeestarQueuedStacker: pass # Factice

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
        all_aligned_files_with_info, 
        q_manager_instance: SeestarQueuedStacker,
        progress_callback):
    """ 
    Orchestre le traitement de mosaïque optimisé.
    CORRECTIONS: Utilise q_manager_instance.stop_processing et initialise filename_for_log.
    """
    
    def _progress(msg):
        if hasattr(q_manager_instance, 'update_progress') and callable(q_manager_instance.update_progress):
            q_manager_instance.update_progress(f"   [MosaicProc] {msg}", None) 
        else: 
            print(f"   [MosaicProc FallbackLog] {msg}")
    
    num_aligned_at_start = len(all_aligned_files_with_info)
    _progress(f"Début assemblage mosaïque pour {num_aligned_at_start} images alignées fournies...")
    if num_aligned_at_start < 1:
        _progress("⚠️ Pas assez d'images pour créer une mosaïque. Traitement annulé.")
        if hasattr(q_manager_instance, 'processing_error'):
            q_manager_instance.processing_error = "Mosaïque: Pas assez d'images"
        return None, None

    api_key = getattr(q_manager_instance, 'api_key', None)
    ref_pixel_scale = getattr(q_manager_instance, 'reference_pixel_scale_arcsec', None)
    output_folder = getattr(q_manager_instance, 'output_folder', None)
    
    _save_panel_stack_temp_func = _save_panel_stack_temp
    _calculate_final_mosaic_grid_func = _calculate_final_mosaic_grid_optimized
    
    if not _DRIZZLE_PROC_AVAILABLE:
        _progress("❌ ERREUR CRITIQUE: DrizzleProcessor n'est pas disponible. Mosaïque impossible.")
        if hasattr(q_manager_instance, 'processing_error'):
            q_manager_instance.processing_error = "Mosaïque: DrizzleProcessor manquant"
        return None, None
    drizzle_processor_class = DrizzleProcessor 
    
    mosaic_drizzle_params_from_qm = getattr(q_manager_instance, 'mosaic_settings', {})
    drizzle_params_final_assembly = {
        'scale_factor': getattr(q_manager_instance, 'drizzle_scale', 2.0),
        'pixfrac': mosaic_drizzle_params_from_qm.get('pixfrac', getattr(q_manager_instance, 'drizzle_pixfrac', 1.0)),
        'kernel': mosaic_drizzle_params_from_qm.get('kernel', getattr(q_manager_instance, 'drizzle_kernel', 'square'))
    }
    _progress(f"Paramètres Drizzle pour assemblage final mosaïque: Scale={drizzle_params_final_assembly['scale_factor']}, "
              f"Kernel='{drizzle_params_final_assembly['kernel']}', Pixfrac={drizzle_params_final_assembly['pixfrac']:.2f}")

    if not all([output_folder, api_key]): 
        _progress("❌ ERREUR: Dossier de sortie ou clé API Astrometry.net manquant pour la mosaïque.")
        if hasattr(q_manager_instance, 'processing_error'):
            q_manager_instance.processing_error = "Mosaïque: Dossier sortie/Clé API manquant"
        return None, None
    
    stacked_panels_info = []
    current_panel_aligned_info = []
    last_panel_center_ra = None
    last_panel_center_dec = None
    panel_count = 0
    total_images_processed_in_loop = 0

    try:
        _progress("1. Groupement et traitement par panneau...")
        
        for i, file_info_tuple in enumerate(all_aligned_files_with_info):
            filename_for_log = f"Item_{i+1}_in_list" # Valeur par défaut
            try:
                aligned_data, header, scores, wcs_obj, valid_pixel_mask = file_info_tuple
                
                if header and 'FILENAME' in header:
                    filename_for_log = header.get('FILENAME')
                elif header:
                    filename_for_log = f"Image_{i+1}_no_fname_in_hdr"

                # ============ CORRECTION 1: Utiliser l'attribut correct ============
                if q_manager_instance.stop_processing: 
                # ===================================================================
                    _progress("🛑 Arrêt demandé par l'utilisateur.")
                    del all_aligned_files_with_info, current_panel_aligned_info; gc.collect()
                    return None, None
                
                total_images_processed_in_loop += 1
                current_progress_phase1 = (total_images_processed_in_loop / num_aligned_at_start) * 50 
                if hasattr(q_manager_instance, 'update_progress') and callable(q_manager_instance.update_progress):
                     q_manager_instance.update_progress(f"   [MosaicProc] Analyse image {total_images_processed_in_loop}/{num_aligned_at_start} ('{filename_for_log}') pour groupement panneau...", current_progress_phase1)

                if not wcs_obj or not wcs_obj.is_celestial or not hasattr(wcs_obj.wcs, 'crval'):
                    _progress(f"   - WARNING: WCS invalide pour '{filename_for_log}'. Ignorée pour groupement.")
                    continue

                img_center_ra = wcs_obj.wcs.crval[0]
                img_center_dec = wcs_obj.wcs.crval[1]
                
                is_new_panel = False
                if last_panel_center_ra is None:
                    is_new_panel = True
                    print(f"DEBUG [MosaicProc]: Détection Premier Panneau (Fichier: '{filename_for_log}'). Centre WCS: ({img_center_ra:.4f}, {img_center_dec:.4f})")
                else:
                    distance = calculate_angular_distance(img_center_ra, img_center_dec, last_panel_center_ra, last_panel_center_dec)
                    print(f"DEBUG [MosaicProc]: Image '{filename_for_log}', Dist au panneau précédent: {distance:.4f} deg (Seuil: {PANEL_GROUPING_THRESHOLD_DEG:.2f})")
                    if distance > PANEL_GROUPING_THRESHOLD_DEG:
                        is_new_panel = True
                        print(f"DEBUG [MosaicProc]: Détection Nouveau Panneau (Dist > Seuil). Fichier: '{filename_for_log}'. Centre WCS: ({img_center_ra:.4f}, {img_center_dec:.4f})")
                
                if is_new_panel and current_panel_aligned_info:
                    panel_count += 1
                    _progress(f"Traitement Panneau Précédent #{panel_count} ({len(current_panel_aligned_info)} images)...")
                    if current_panel_aligned_info:
                        _progress(f"   -> Stacking Panneau #{panel_count}...")
                        panel_stack_np, panel_stack_header, _ = q_manager_instance._stack_batch(
                            current_panel_aligned_info, panel_count, 0
                        )
                        if panel_stack_np is not None:
                            _progress(f"   -> Plate-Solving Panneau #{panel_count}...")
                            fallback_header_for_solve = current_panel_aligned_info[0][1] if current_panel_aligned_info else fits.Header()
                            solved_wcs_panel = solve_image_wcs(panel_stack_np, fallback_header_for_solve, api_key, ref_pixel_scale, progress_callback=q_manager_instance.update_progress)
                            if solved_wcs_panel:
                                _progress(f"   -> Sauvegarde Stack Panneau #{panel_count}...")
                                temp_panel_path = _save_panel_stack_temp_func(panel_stack_np, solved_wcs_panel, panel_count, output_folder)
                                if temp_panel_path: 
                                    stacked_panels_info.append((temp_panel_path, solved_wcs_panel))
                                    print(f"DEBUG [MosaicProc]: Panneau #{panel_count} traité et ajouté.")
                                else: 
                                    _progress(f"   - ERREUR sauvegarde temp panneau #{panel_count}.")
                                    print(f"ERREUR [MosaicProc]: Échec _save_panel_stack_temp_func pour panneau {panel_count}")
                            else: 
                                _progress(f"   - WARNING: Plate-solving échoué pour panneau #{panel_count}.")
                                print(f"WARN [MosaicProc]: Échec solve_image_wcs pour panneau {panel_count}")
                        else: 
                            _progress(f"   - WARNING: Stacking (_stack_batch) échoué pour panneau #{panel_count}.")
                            print(f"WARN [MosaicProc]: Échec _stack_batch pour panneau {panel_count}")
                        del panel_stack_np, panel_stack_header; gc.collect()
                    current_panel_aligned_info = []

                current_panel_aligned_info.append(file_info_tuple)
                if is_new_panel: 
                    last_panel_center_ra = img_center_ra
                    last_panel_center_dec = img_center_dec

            except Exception as loop_err:
                # ============ CORRECTION 2: Utiliser filename_for_log ici ============
                _progress(f"   - ERREUR traitement '{filename_for_log}' (item {i+1}) dans boucle principale panneaux: {loop_err}")
                # ====================================================================
                print(f"ERREUR [MosaicProc loop_err] pour '{filename_for_log}': {loop_err}")
                traceback.print_exc(limit=1)
                try: del aligned_data, header, scores, wcs_obj, valid_pixel_mask; gc.collect()
                except NameError: pass
        
        if current_panel_aligned_info:
            panel_count += 1
            _progress(f"Traitement du Dernier Panneau #{panel_count} ({len(current_panel_aligned_info)} images)...")
            if current_panel_aligned_info:
                _progress(f"   -> Stacking Dernier Panneau #{panel_count}...")
                panel_stack_np, panel_stack_header, _ = q_manager_instance._stack_batch(
                    current_panel_aligned_info, panel_count, 0
                )
                if panel_stack_np is not None:
                    _progress(f"   -> Plate-Solving Dernier Panneau #{panel_count}...")
                    fallback_header_for_solve = current_panel_aligned_info[0][1] if current_panel_aligned_info else fits.Header()
                    solved_wcs_panel = solve_image_wcs(panel_stack_np, fallback_header_for_solve, api_key, ref_pixel_scale, progress_callback=q_manager_instance.update_progress)
                    if solved_wcs_panel:
                        _progress(f"   -> Sauvegarde Dernier Stack Panneau #{panel_count}...")
                        temp_panel_path = _save_panel_stack_temp_func(panel_stack_np, solved_wcs_panel, panel_count, output_folder)
                        if temp_panel_path: 
                            stacked_panels_info.append((temp_panel_path, solved_wcs_panel))
                            print(f"DEBUG [MosaicProc]: Dernier panneau traité et ajouté.")
                        else: 
                            _progress(f"   - ERREUR sauvegarde temp dernier panneau.")
                    else: 
                        _progress(f"   - WARNING: Plate-solving échoué pour dernier panneau.")
                else: 
                    _progress(f"   - WARNING: Stacking (_stack_batch) échoué pour dernier panneau.")
                del panel_stack_np, panel_stack_header
            del current_panel_aligned_info 
            gc.collect()
        
        del all_aligned_files_with_info 
        gc.collect()
        _progress("Traitement de tous les panneaux (stack+solve) terminé.")
        print(f"DEBUG [MosaicProc]: Nombre total de panneaux stackés et résolus prêts pour assemblage: {len(stacked_panels_info)}")

        # ... (Reste de la fonction pour l'assemblage Drizzle, création header, etc. - inchangé par rapport à ma réponse précédente)
        # ... (Assurez-vous que la fin de la fonction est bien présente)

        # ========================================================
        # --- 2. Calcul Grille Finale & Assemblage Drizzle ---
        # ========================================================
        if not stacked_panels_info: 
            _progress("❌ ERREUR: Aucun panneau stacké/résolu produit. Impossible d'assembler la mosaïque.")
            if hasattr(q_manager_instance, 'processing_error'):
                q_manager_instance.processing_error = "Mosaïque: Aucun panneau valide"
            return None, None

        _progress("Calcul de la grille de sortie finale pour la mosaïque...")
        if hasattr(q_manager_instance, 'update_progress') and callable(q_manager_instance.update_progress):
             q_manager_instance.update_progress("   [MosaicProc] Calcul grille finale...", 60)

        panel_wcs_list_for_grid = []
        panel_shapes_hw_list_for_grid = []
        temp_panel_paths_for_final_drizzle = []

        for fpath, wcs_panel_obj in stacked_panels_info:
            shape_hw_panel = None
            try:
                with fits.open(fpath, memmap=False) as hdul: 
                    if hdul[0].data is not None and hdul[0].data.ndim == 3:
                        shape_hw_panel = hdul[0].shape[1:]
            except Exception as e_shape: 
                _progress(f"   - WARNING: Erreur lecture shape du panneau temp {os.path.basename(fpath)}: {e_shape}. Panneau ignoré.")
                print(f"WARN [MosaicProc]: Échec lecture shape pour {os.path.basename(fpath)}")
            
            if shape_hw_panel and len(shape_hw_panel)==2 and wcs_panel_obj:
                panel_wcs_list_for_grid.append(wcs_panel_obj)
                panel_shapes_hw_list_for_grid.append(shape_hw_panel)
                temp_panel_paths_for_final_drizzle.append(fpath)
            else:
                 _progress(f"   - WARNING: Panneau {os.path.basename(fpath)} a une shape ou WCS invalide. Ignoré pour grille/drizzle final.")
                 print(f"WARN [MosaicProc]: Panneau {os.path.basename(fpath)} skippé (shape/WCS invalide). Shape lue: {shape_hw_panel}, WCS présent: {wcs_panel_obj is not None}")

        if not panel_wcs_list_for_grid or not temp_panel_paths_for_final_drizzle:
            _progress("❌ ERREUR: Pas assez de panneaux valides pour calculer la grille ou assembler.")
            if hasattr(q_manager_instance, 'processing_error'):
                q_manager_instance.processing_error = "Mosaïque: Panneaux invalides pour grille/assemblage final"
            return None, None

        final_output_wcs, final_output_shape_hw = _calculate_final_mosaic_grid_func(
            panel_wcs_list_for_grid, panel_shapes_hw_list_for_grid, drizzle_params_final_assembly['scale_factor']
        )

        if final_output_wcs is None or final_output_shape_hw is None:
            _progress("❌ ERREUR: Échec calcul grille de sortie mosaïque finale.")
            if hasattr(q_manager_instance, 'processing_error'):
                q_manager_instance.processing_error = "Mosaïque: Échec calcul grille finale"
            return None, None
        
        _progress(f"Assemblage final Drizzle sur grille {final_output_shape_hw} (H,W)...")
        if hasattr(q_manager_instance, 'update_progress') and callable(q_manager_instance.update_progress):
            q_manager_instance.update_progress("   [MosaicProc] Assemblage Drizzle final...", 85)

        mosaic_drizzler = drizzle_processor_class(**drizzle_params_final_assembly)
        
        print(f"DEBUG [MosaicProc]: Appel de mosaic_drizzler.apply_drizzle pour l'assemblage final...")
        final_mosaic_sci, final_mosaic_wht = mosaic_drizzler.apply_drizzle(
            temp_filepath_list=temp_panel_paths_for_final_drizzle, # L'argument est temp_filepath_list
            output_wcs=final_output_wcs,                           # WCS de sortie
            output_shape_2d_hw=final_output_shape_hw               # Shape de sortie
        )
        
        if final_mosaic_sci is None:
            _progress("❌ ERREUR: Échec de l'assemblage final Drizzle de la mosaïque.")
            if hasattr(q_manager_instance, 'processing_error'):
                q_manager_instance.processing_error = "Mosaïque: Échec assemblage Drizzle final"
            return None, None

        _progress("✅ Assemblage Drizzle de la mosaïque terminé.")
        if final_mosaic_wht is not None: del final_mosaic_wht; gc.collect()

        # ========================================================
        # --- 3. Création Header Final et Retour ---
        # ========================================================
        _progress("Création du header final pour la mosaïque...")
        if hasattr(q_manager_instance, 'update_progress') and callable(q_manager_instance.update_progress):
            q_manager_instance.update_progress("   [MosaicProc] Création header final...", 98)

        final_header = fits.Header()
        if final_output_wcs: final_header.update(final_output_wcs.to_header(relax=True))
        
        ref_hdr_global = getattr(q_manager_instance, 'reference_header_for_wcs', None)
        if ref_hdr_global:
            keys_to_copy=['INSTRUME','TELESCOP','OBJECT','FILTER','DATE-OBS','GAIN','OFFSET','CCD-TEMP','SITELAT','SITELONG','FOCALLEN','APERTURE']
            for k in keys_to_copy:
                if k in ref_hdr_global: 
                    try: final_header.set(k, ref_hdr_global[k], ref_hdr_global.comments[k] if k in ref_hdr_global.comments else None)
                    except Exception: final_header.set(k, ref_hdr_global[k])

        final_header['STACKTYP'] = (f'Mosaic Drizzle ({drizzle_params_final_assembly["scale_factor"]:.1f}x)', 'Mosaic from solved & stacked panels')
        final_header['DRZSCALE'] = (drizzle_params_final_assembly['scale_factor'], 'Mosaic Drizzle scale factor')
        final_header['DRZKERNEL'] = (drizzle_params_final_assembly['kernel'], 'Mosaic Drizzle kernel')
        final_header['DRZPIXFR'] = (drizzle_params_final_assembly['pixfrac'], 'Mosaic Drizzle pixfrac')
        final_header['NPANELS'] = (panel_count, 'Number of panels identified and processed')
        final_header['NIMAGES'] = (num_aligned_at_start, 'Total source aligned images for mosaic')

        if hasattr(q_manager_instance, 'images_in_cumulative_stack'):
            q_manager_instance.images_in_cumulative_stack = num_aligned_at_start
        print(f"DEBUG [MosaicProc]: Compteur images QM mis à jour pour rapport final: {num_aligned_at_start}")
        
        approx_tot_exp = 0.0
        if ref_hdr_global and 'EXPTIME' in ref_hdr_global: 
            try: approx_tot_exp = float(ref_hdr_global['EXPTIME']) * num_aligned_at_start
            except (ValueError, TypeError): pass
        final_header['TOTEXP'] = (round(approx_tot_exp, 2), '[s] Approx total exposure of source images')

        final_header['ALIGNED'] = (num_aligned_at_start, 'Source images passed to mosaic process')
        final_header['FAILALIGN'] = (getattr(q_manager_instance, 'failed_align_count', 0), 'Failed alignments (initial pre-mosaic stage)')
        num_successfully_stacked_panels = len(stacked_panels_info)
        failed_panel_processing_count = panel_count - num_successfully_stacked_panels
        final_header['FAILSTACK'] = (failed_panel_processing_count, 'Panels that failed stacking or solving')
        final_header['SKIPPED'] = (getattr(q_manager_instance, 'skipped_files_count', 0), 'Other skipped/error files (initial pre-mosaic stage)')

        _progress("Orchestration mosaïque terminée avec succès.")
        return final_mosaic_sci.astype(np.float32), final_header

    except Exception as e_mosaic_main:
        _progress(f"❌ ERREUR CRITIQUE dans le traitement de la mosaïque: {e_mosaic_main}")
        print(f"ERREUR MAJEURE [MosaicProc]: {e_mosaic_main}")
        traceback.print_exc()
        if hasattr(q_manager_instance, 'processing_error'):
            q_manager_instance.processing_error = f"Mosaïque: {e_mosaic_main}"
        return None, None
    finally:
        print("DEBUG [MosaicProc]: Fin de process_mosaic_from_aligned_files (dans le bloc finally).")
        gc.collect()






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