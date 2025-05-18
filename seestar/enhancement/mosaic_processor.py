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
import cv2

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

# S'assurer que cet import est présent et fonctionnel :
try:
    from drizzle.resample import Drizzle # La classe Drizzle de stsci
    _OO_DRIZZLE_AVAILABLE = True
except ImportError:
    _OO_DRIZZLE_AVAILABLE = False
    Drizzle = None # Factice si non disponible

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
        aligned_panel_info_list: list,
        output_wcs: WCS,              # WCS de la grille de sortie Drizzle finale
        output_shape_hw: tuple,       # Shape (H,W) de la grille de sortie Drizzle finale
        kernel: str = "square",       # Paramètre Drizzle pour le noyau
        pixfrac: float = 1.0,         # Paramètre Drizzle pour pixfrac
        fillval: str = "0.0",         # Paramètre Drizzle pour la valeur de remplissage
        wht_threshold: float = 0.01, # Ce paramètre n'est pas directement utilisé par Drizzle.__init__ ou add_image
        progress_callback: callable = None):
    """
    Assemble une mosaïque à partir d'une liste de panneaux alignés en utilisant Drizzle (stsci).
    CORRIGÉ: Initialisation de Drizzle() sans out_wcs, conformément à la signature de la bibliothèque.
    """
    if not progress_callback:
        progress_callback = lambda msg, prog=None: print(f"MOSAIC_PROC_LOG: {msg}" + (f" ({prog}%)" if prog is not None else ""))

    progress_callback(f"Mosaïque: Début assemblage Drizzle avec {len(aligned_panel_info_list)} panneaux (v_no_outwcs_init)...", 50)

    if not aligned_panel_info_list:
        progress_callback("Mosaïque ERREUR: Aucune information de panneau fournie pour Drizzle.", 0)
        return None, None

    if not _OO_DRIZZLE_AVAILABLE or Drizzle is None:
        progress_callback("Mosaïque ERREUR: Bibliothèque Drizzle (stsci) non disponible.", 0)
        print("ERREUR [MosaicProc]: stsci.drizzle.resample.Drizzle non importable.")
        return None, None

    if output_wcs is None or output_shape_hw is None:
        progress_callback("Mosaïque ERREUR: WCS ou Shape de sortie Drizzle manquant.", 0)
        print("ERREUR [MosaicProc]: output_wcs ou output_shape_hw est None.")
        return None, None

    num_panels = len(aligned_panel_info_list)
    num_output_channels = 3 # Assumer RGB

    final_drizzlers_list = []
    final_output_sci_arrays = []
    final_output_wht_arrays = []

    try:
        progress_callback(f"   Mosaïque: Initialisation des {num_output_channels} Drizzlers finaux (Kernel: {kernel}, Fillval: {fillval})...", 52)
        for _ in range(num_output_channels):
            sci_channel_array = np.zeros(output_shape_hw, dtype=np.float32)
            wht_channel_array = np.zeros(output_shape_hw, dtype=np.float32)
            final_output_sci_arrays.append(sci_channel_array)
            final_output_wht_arrays.append(wht_channel_array)

            # --- CORRECTION ICI : out_wcs retiré de l'initialisation ---
            driz_channel_obj = Drizzle(
                out_img=sci_channel_array,
                out_wht=wht_channel_array,
                out_shape=output_shape_hw, # out_shape EST un paramètre valide
                # out_wcs=output_wcs,      # CE PARAMÈTRE EST INCORRECT POUR __INIT__
                kernel=kernel,
                fillval=fillval,
                #pixfrac=pixfrac
            )
            # --- FIN CORRECTION ---
            final_drizzlers_list.append(driz_channel_obj)
        progress_callback(f"   Mosaïque: Drizzlers finaux initialisés.", 55)
    except Exception as init_err:
        progress_callback(f"Mosaïque ERREUR: Échec initialisation Drizzle: {init_err}", 0)
        traceback.print_exc(limit=1)
        return None, None

    panels_added_successfully = 0
    for i_panel, panel_info_tuple in enumerate(aligned_panel_info_list):
        panel_image_data_HWC = panel_info_tuple[0]
        wcs_for_panel_input = panel_info_tuple[2]
        transform_matrix_M = panel_info_tuple[3]
        pixel_mask_2d_for_weight = panel_info_tuple[4]

        progress_callback(f"   Mosaïque: Traitement Drizzle panneau {i_panel+1}/{num_panels}...", 55 + int(40 * (i_panel / num_panels)))

        if panel_image_data_HWC is None or wcs_for_panel_input is None or transform_matrix_M is None:
            progress_callback(f"    -> Panneau {i_panel+1}: Données/WCS/Matrice M manquantes. Ignoré.", None)
            continue

        try:
            panel_shape_hw = panel_image_data_HWC.shape[:2]
            y_panel_coords, x_panel_coords = np.indices(panel_shape_hw)
            x_panel_flat = x_panel_coords.ravel()
            y_panel_flat = y_panel_coords.ravel()

            panel_pixels_N12 = np.dstack((x_panel_flat, y_panel_flat)).reshape(-1, 1, 2).astype(np.float32)
            pixels_in_anchor_or_abs_sys_N12 = cv2.transform(panel_pixels_N12, transform_matrix_M)
            pixels_in_anchor_or_abs_sys_N2 = pixels_in_anchor_or_abs_sys_N12.reshape(-1, 2)

            sky_coords_ra_deg, sky_coords_dec_deg = wcs_for_panel_input.all_pix2world(
                pixels_in_anchor_or_abs_sys_N2[:, 0],
                pixels_in_anchor_or_abs_sys_N2[:, 1],
                0
            )
            # Le WCS de sortie pour le pixmap doit être celui passé à la fonction, pas celui de l'objet Drizzle
            # car l'objet Drizzle (stsci) ne stocke pas de WCS de sortie lui-même.
            final_x_output_pixels_flat, final_y_output_pixels_flat = output_wcs.all_world2pix(
                sky_coords_ra_deg, sky_coords_dec_deg, 0
            )

            pixmap_for_drizzle = np.dstack((
                final_x_output_pixels_flat.reshape(panel_shape_hw),
                final_y_output_pixels_flat.reshape(panel_shape_hw)
            )).astype(np.float32)

            exposure_time_for_drizzle = 1.0

            for i_channel in range(num_output_channels):
                channel_data_2d = panel_image_data_HWC[:, :, i_channel].astype(np.float32)
                channel_data_2d_clean = np.nan_to_num(channel_data_2d, nan=0.0, posinf=0.0, neginf=0.0)
                weight_map_input = pixel_mask_2d_for_weight.astype(np.float32) if pixel_mask_2d_for_weight is not None else None
                
                final_drizzlers_list[i_channel].add_image(
                    data=channel_data_2d_clean,
                    pixmap=pixmap_for_drizzle,
                    exptime=exposure_time_for_drizzle,
                    in_units='cps',
                    pixfrac=pixfrac,
                    weight_map=weight_map_input
                )
            panels_added_successfully += 1
        except Exception as e_panel_driz:
            progress_callback(f"    -> Mosaïque ERREUR: Échec Drizzle pour panneau {i_panel+1}: {e_panel_driz}", None)
            traceback.print_exc(limit=1)

    progress_callback(f"   Mosaïque: {panels_added_successfully}/{num_panels} panneaux ajoutés avec succès à Drizzle.", 95)

    if panels_added_successfully == 0:
        progress_callback("Mosaïque ERREUR: Aucun panneau n'a pu être ajouté à Drizzle.", 0)
        return None, None

    try:
        final_sci_image_HWC = np.stack(final_output_sci_arrays, axis=-1).astype(np.float32)
        final_wht_map_HWC = np.stack(final_output_wht_arrays, axis=-1).astype(np.float32)
        
        final_sci_image_HWC = np.nan_to_num(final_sci_image_HWC, nan=0.0, posinf=0.0, neginf=0.0)
        final_wht_map_HWC = np.nan_to_num(final_wht_map_HWC, nan=0.0, posinf=0.0, neginf=0.0)
        final_wht_map_HWC = np.maximum(final_wht_map_HWC, 0.0)

        progress_callback("Mosaïque: Assemblage Drizzle finalisé.", 100)
        print(f"DEBUG [MosaicProc]: Drizzle terminé. Shape SCI finale: {final_sci_image_HWC.shape}, Shape WHT finale: {final_wht_map_HWC.shape}")
        return final_sci_image_HWC, final_wht_map_HWC

    except Exception as e_stack_final:
        progress_callback(f"Mosaïque ERREUR: Échec assemblage final des canaux Drizzle: {e_stack_final}", 0)
        traceback.print_exc(limit=1)
        return None, None
    finally:
        del final_drizzlers_list, final_output_sci_arrays, final_output_wht_arrays
        gc.collect()





# --- FIN DU FICHIER seestar/enhancement/mosaic_processor.py ---