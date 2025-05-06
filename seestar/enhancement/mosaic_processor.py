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

def _calculate_final_mosaic_grid_optimized(panel_wcs_list, panel_shapes_hw_list, drizzle_scale):
    """
    Calcule le WCS et la Shape (H, W) optimaux pour la mosaïque finale.
    (Version complète de l'étape 7/22)
    """
    num_panels = len(panel_wcs_list)
    print(f"DEBUG [MosaicGridOptim]: Calcul grille pour {num_panels} panneaux...")
    if num_panels == 0 or len(panel_shapes_hw_list) != num_panels or None in panel_shapes_hw_list:
        print("ERREUR [MosaicGridOptim]: Listes WCS/Shapes panneaux invalides.")
        return None, None

    valid_wcs_list=panel_wcs_list; valid_shapes_hw=panel_shapes_hw_list
    print(f"   -> {len(valid_wcs_list)} panneaux valides reçus pour calcul grille.")

    try:
        # --- Calcul Footprints Panneaux ---
        all_footprints_sky = []
        print("   -> Calcul footprints panneaux...")
        for i, (wcs_p, shape_hw) in enumerate(zip(valid_wcs_list, valid_shapes_hw)):
             # Vérifier si shape_hw est valide
             if not shape_hw or len(shape_hw) != 2 or shape_hw[0] <= 0 or shape_hw[1] <= 0:
                 print(f"      - WARNING: Shape invalide pour panneau {i+1}: {shape_hw}. Ignoré.")
                 continue
             # Vérifier si wcs_p a pixel_shape et le définir si besoin
             if wcs_p.pixel_shape is None or wcs_p.pixel_shape != (shape_hw[1], shape_hw[0]):
                  print(f"      - Ajustement pixel_shape WCS panneau {i+1} à ({shape_hw[1]}, {shape_hw[0]})")
                  wcs_p.pixel_shape = (shape_hw[1], shape_hw[0]) # W, H
             nx, ny = shape_hw[1], shape_hw[0] # W, H
             pixel_corners = np.array([[0,0],[nx-1,0],[nx-1,ny-1],[0,ny-1]], dtype=np.float64)
             try: sky_corners = wcs_p.pixel_to_world(pixel_corners[:,0], pixel_corners[:,1]); all_footprints_sky.append(sky_corners)
             except Exception as fp_err: print(f"      - ERREUR footprint panneau {i+1}: {fp_err}. Ignoré.")

        if not all_footprints_sky: print("ERREUR: Aucun footprint panneau calculé."); return None, None

        # --- Calcul Étendue et Centre ---
        print("   -> Calcul étendue totale..."); all_corners_flat = SkyCoord(ra=np.concatenate([fp.ra.deg for fp in all_footprints_sky]), dec=np.concatenate([fp.dec.deg for fp in all_footprints_sky]), unit='deg', frame='icrs')
        central_ra_deg=np.median(all_corners_flat.ra.wrap_at(180*u.deg).deg); central_dec_deg=np.median(all_corners_flat.dec.deg); print(f"      - Centre Médian: ({central_ra_deg:.5f}, {central_dec_deg:.5f}) deg")

        # --- Définition WCS Sortie ---
        print("   -> Création WCS sortie..."); ref_wcs = valid_wcs_list[0]; output_wcs = WCS(naxis=2); output_wcs.wcs.ctype = getattr(ref_wcs.wcs, 'ctype', ["RA---TAN", "DEC--TAN"]); output_wcs.wcs.crval = [central_ra_deg, central_dec_deg]; output_wcs.wcs.cunit = getattr(ref_wcs.wcs, 'cunit', ['deg', 'deg'])
        try: # Calcul échelle sortie
            ref_scale_matrix = ref_wcs.pixel_scale_matrix; avg_input_scale = np.mean(np.abs(np.diag(ref_scale_matrix))); output_pixel_scale = avg_input_scale / drizzle_scale; output_wcs.wcs.cd = np.array([[-output_pixel_scale, 0.0], [0.0, output_pixel_scale]]); print(f"      - Échelle Sortie: {output_pixel_scale*3600:.3f} arcsec/pix")
        except Exception as scale_err: raise ValueError("Échec calcul échelle WCS sortie.") from scale_err

        # --- Calcul Shape Sortie ---
        print("   -> Calcul shape sortie..."); all_output_pixels_x=[]; all_output_pixels_y=[]
        for footprint_sky in all_footprints_sky:
            try: pixels_out_x, pixels_out_y = output_wcs.world_to_pixel(footprint_sky); all_output_pixels_x.extend(pixels_out_x); all_output_pixels_y.extend(pixels_out_y)
            except Exception as proj_err: print(f"      - WARNING: Échec projection coins: {proj_err}.")
        if not all_output_pixels_x: print("ERREUR: Aucun coin projeté."); return None, None
        x_min_out=np.min(all_output_pixels_x); x_max_out=np.max(all_output_pixels_x); y_min_out=np.min(all_output_pixels_y); y_max_out=np.max(all_output_pixels_y)
        out_width=int(np.ceil(x_max_out-x_min_out+1)); out_height=int(np.ceil(y_max_out-y_min_out+1)); out_width=max(10, out_width); out_height=max(10, out_height); output_shape_hw=(out_height, out_width); print(f"      - Dimensions Finales (W, H): ({out_width}, {out_height})")

        # --- Finalisation WCS Sortie ---
        try: center_x_out, center_y_out = output_wcs.world_to_pixel(SkyCoord(ra=central_ra_deg*u.deg, dec=central_dec_deg*u.deg)); output_wcs.wcs.crpix = [center_x_out-x_min_out+1.0, center_y_out-y_min_out+1.0]
        except Exception as crpix_err: print(f"      - WARNING: Échec ajustement CRPIX: {crpix_err}"); output_wcs.wcs.crpix = [out_width/2.0+0.5, out_height/2.0+0.5]
        output_wcs.pixel_shape = (out_width, out_height) # W, H
        try: output_wcs._naxis1=out_width; output_wcs._naxis2=out_height
        except AttributeError: pass
        print(f"      - WCS Finalisé: CRPIX={output_wcs.wcs.crpix}, PixelShape={output_wcs.pixel_shape}")

        print(f"DEBUG [MosaicGridOptim]: Calcul grille OK.")
        return output_wcs, output_shape_hw

    except Exception as e: print(f"ERREUR [MosaicGridOptim]: Échec global: {e}"); traceback.print_exc(limit=3); return None, None

# --- Fonction principale ---
def process_mosaic_from_aligned_files(
        all_aligned_files_with_info, # Liste [(data, hdr, scores, wcs_final), ...]
        q_manager_instance: SeestarQueuedStacker,
        progress_callback):
    """ Orchestre le traitement de mosaïque optimisé. """
    # ... (définition _progress, vérifications initiales - inchangées) ...
    def _progress(msg): # Helper interne
        if progress_callback: 
            progress_callback(f"   [MosaicProc] {msg}", None)
        else: 
            print(f"   [MosaicProc] {msg}")
    
    num_aligned = len(all_aligned_files_with_info)
    _progress(f"Début {num_aligned} images...")
    if num_aligned < 2: 
        _progress("⚠️ Pas assez d'images.")
        return None, None
    
    # ... (récupération config et fonctions depuis q_manager_instance - inchangé) ...
    api_key = getattr(q_manager_instance, 'api_key', None)
    ref_pixel_scale = getattr(q_manager_instance, 'reference_pixel_scale_arcsec', None)
    output_folder = getattr(q_manager_instance, 'output_folder', None)
    _stack_batch_func = getattr(q_manager_instance, '_stack_batch', None)
    _save_panel_stack_temp_func = _save_panel_stack_temp
    _calculate_final_mosaic_grid_func = _calculate_final_mosaic_grid_optimized
    drizzle_processor_class = DrizzleProcessor
    drizzle_params = {
        'scale_factor': q_manager_instance.drizzle_scale, 
        'pixfrac': q_manager_instance.drizzle_pixfrac, 
        'kernel': q_manager_instance.drizzle_kernel
    }
    
    if not all([_stack_batch_func, _save_panel_stack_temp_func, _calculate_final_mosaic_grid_func, drizzle_processor_class, output_folder, api_key]): 
        _progress("❌ ERREUR: Dépendances orchestration manquantes.")
        return None, None
    
    # --- Initialisation ---
    stacked_panels_info = []
    current_panel_aligned_info = []
    last_panel_center_ra = None
    last_panel_center_dec = None
    panel_count = 0
    total_processed = 0

    try:
        # ========================================================
        # --- 1. Groupement et Traitement par Panneau ---
        # ========================================================
        _progress("1. Groupement et traitement par panneau (Stack + Solve)...")
        num_total_images = len(all_aligned_files_with_info)

        for i, file_info_tuple in enumerate(all_aligned_files_with_info):
            # Utiliser un try/except pour chaque image pour plus de robustesse
            try:
                aligned_data, header, scores, wcs_obj = file_info_tuple
                if q_manager_instance.stop_processing: 
                    _progress("🛑 Arrêt demandé.")
                    return None, None # Vérifier arrêt
                
                total_processed += 1
                current_progress = (total_processed / num_total_images) * 50
                progress_callback(f"   [MosaicProc] Analyse panneau image {total_processed}/{num_total_images}...", current_progress)

                if not wcs_obj or not wcs_obj.is_celestial or not hasattr(wcs_obj.wcs, 'crval'):
                    _progress(f"   - WARNING: WCS invalide image {i+1}. Ignorée.")
                    continue # Passer à l'image suivante

                img_center_ra = wcs_obj.wcs.crval[0]
                img_center_dec = wcs_obj.wcs.crval[1]
                
                # --- CORRECTION LOGIQUE is_new_panel ---
                is_new_panel = False # Initialiser à False
                if last_panel_center_ra is None:
                    # Cas 1: C'est la toute première image valide rencontrée
                    is_new_panel = True
                    print(f"DEBUG [MosaicProc]: Détection Premier Panneau (Centre WCS: {img_center_ra:.4f}, {img_center_dec:.4f})")
                else:
                    # Cas 2: Comparer avec le centre du panneau précédent
                    distance = calculate_angular_distance(img_center_ra, img_center_dec, last_panel_center_ra, last_panel_center_dec)
                    print(f"DEBUG [MosaicProc]: Image {i+1}, Dist au panneau précédent: {distance:.4f} deg") # Log distance
                    if distance > PANEL_GROUPING_THRESHOLD_DEG:
                        is_new_panel = True
                        print(f"DEBUG [MosaicProc]: Détection Nouveau Panneau (Dist > Seuil). Centre WCS: {img_center_ra:.4f}, {img_center_dec:.4f}")
                # --- FIN CORRECTION LOGIQUE ---

                # --- Traitement Panneau Précédent ---
                if is_new_panel and current_panel_aligned_info:
                    panel_count += 1
                    _progress(f"Traitement Panneau #{panel_count} ({len(current_panel_aligned_info)} images)...")
                    panel_images = [info[0] for info in current_panel_aligned_info]
                    panel_headers = [info[1] for info in current_panel_aligned_info]
                    panel_scores = [info[2] for info in current_panel_aligned_info]
                    
                    if panel_images:
                        _progress(f"   -> Stacking Panneau #{panel_count}...")
                        panel_stack_np, _ = q_manager_instance._stack_batch(panel_images, panel_headers, panel_scores, panel_count)
                        
                        if panel_stack_np is not None:
                            _progress(f"   -> Plate-Solving Panneau #{panel_count}...")
                            fallback_header = panel_headers[0] if panel_headers else fits.Header()
                            solved_wcs_panel = solve_image_wcs(panel_stack_np, fallback_header, api_key, ref_pixel_scale, progress_callback=progress_callback)
                            
                            if solved_wcs_panel:
                                _progress(f"   -> Sauvegarde Stack Panneau #{panel_count}...")
                                temp_panel_path = _save_panel_stack_temp_func(panel_stack_np, solved_wcs_panel, panel_count, output_folder)
                                
                                if temp_panel_path: 
                                    stacked_panels_info.append((temp_panel_path, solved_wcs_panel))
                                    print(f"DEBUG [MosaicProc]: Panneau {panel_count} traité et ajouté.")
                                else: 
                                    _progress(f"   - ERREUR sauvegarde temp panneau #{panel_count}.")
                            else: 
                                _progress(f"   - WARNING: Plate-solving échoué panneau #{panel_count}.")
                        else: 
                            _progress(f"   - WARNING: Stacking échoué panneau #{panel_count}.")
                    else: 
                        _progress(f"   - WARNING: Aucune donnée pour panneau #{panel_count}.")
                    
                    # Nettoyer mémoire
                    del panel_images, panel_headers, panel_scores, panel_stack_np
                    gc.collect()
                    current_panel_aligned_info = [] # Réinitialiser

                # --- Fin Traitement Panneau Précédent ---

                # Ajouter l'info courante au panneau courant
                current_panel_aligned_info.append((aligned_data, header, scores, wcs_obj))
                # Mettre à jour le centre de référence si nouveau panneau
                if is_new_panel: 
                    last_panel_center_ra = img_center_ra
                    last_panel_center_dec = img_center_dec

            except Exception as loop_err: # Gérer erreur dans la boucle pour une image
                _progress(f"   - ERREUR traitement image {i+1}: {loop_err}")
                traceback.print_exc(limit=1)
                # Essayer de nettoyer la mémoire pour cette image ratée
                try: 
                    del aligned_data, header, scores, wcs_obj
                    gc.collect()
                except NameError: 
                    pass
        # --- Fin Boucle Principale ---

        # --- Traiter le TOUT dernier panneau ---
        if current_panel_aligned_info:
            panel_count += 1
            _progress(f"Traitement Dernier Panneau #{panel_count} ({len(current_panel_aligned_info)} images)...")
            # --- Copier/Coller logique Stack, Solve, Save ---
            panel_images = [info[0] for info in current_panel_aligned_info]
            panel_headers = [info[1] for info in current_panel_aligned_info]
            panel_scores = [info[2] for info in current_panel_aligned_info]
            
            if panel_images:
                _progress(f"   -> Stacking Dernier Panneau #{panel_count}...")
                panel_stack_np, _ = q_manager_instance._stack_batch(panel_images, panel_headers, panel_scores, panel_count)
                del panel_images, panel_headers, panel_scores
                gc.collect()
                
                if panel_stack_np is not None:
                    _progress(f"   -> Plate-Solving Dernier Panneau #{panel_count}...")
                    fallback_header = current_panel_aligned_info[0][1] if current_panel_aligned_info else fits.Header()
                    solved_wcs_panel = solve_image_wcs(panel_stack_np, fallback_header, api_key, ref_pixel_scale, progress_callback=progress_callback)
                    
                    if solved_wcs_panel:
                        _progress(f"   -> Sauvegarde Dernier Stack Panneau #{panel_count}...")
                        temp_panel_path = _save_panel_stack_temp_func(panel_stack_np, solved_wcs_panel, panel_count, output_folder)
                        
                        if temp_panel_path: 
                            stacked_panels_info.append((temp_panel_path, solved_wcs_panel))
                            print(f"DEBUG [MosaicProc]: Dernier panneau ajouté.")
                        else: 
                            _progress(f"   - ERREUR sauvegarde temp dernier panneau.")
                    else: 
                        _progress(f"   - WARNING: Plate-solving échoué dernier panneau.")
                else: 
                    _progress(f"   - WARNING: Stacking échoué dernier panneau.")
            else: 
                _progress(f"   - WARNING: Aucune donnée lue pour dernier panneau.")
            
            del current_panel_aligned_info
            gc.collect()
        # --- Fin traitement dernier panneau ---

        # --- Vider la liste originale (contient données mémoire) ---
        del all_aligned_files_with_info
        gc.collect()
        _progress("Traitement de tous les panneaux terminé.")


        # ========================================================
        # --- 2. Calcul Grille Finale & Assemblage Drizzle ---
        # ========================================================

        # --- 2. Calcul Grille Finale & Assemblage Drizzle ---
        if not stacked_panels_info:
            _progress("❌ ERREUR: Aucun panneau stacké/résolu produit. Impossible d'assembler.")
            return None, None

        _progress("Calcul de la grille de sortie finale pour la mosaïque...")
        panel_wcs_list = [info[1] for info in stacked_panels_info]
        panel_shapes_hw = []
        
        for fpath, _wcs in stacked_panels_info:
            shape = None
            try: # Lire la shape H,W depuis le fichier FITS temporaire du panneau
                with fits.open(fpath, memmap=False) as hdul: 
                    shape = hdul[0].shape[1:] # H,W depuis CxHxW
            except Exception as e: 
                print(f"WARN: Err lecture shape {os.path.basename(fpath)}: {e}")
            panel_shapes_hw.append(shape if shape and len(shape)==2 else None)

        if None in panel_shapes_hw:
            _progress("❌ ERREUR: Impossible lire shape de tous les panneaux temp.")
            return None, None

        # Appeler la fonction de calcul de grille (locale/importée)
        final_output_wcs, final_output_shape_hw = _calculate_final_mosaic_grid_func(
            panel_wcs_list, panel_shapes_hw, drizzle_params['scale_factor']
        )

        if final_output_wcs is None or final_output_shape_hw is None:
            _progress("❌ ERREUR: Échec calcul grille de sortie mosaïque finale.")
            return None, None

        _progress(f"Assemblage final Drizzle sur grille {final_output_shape_hw}...")
        if not _DRIZZLE_PROC_AVAILABLE: 
            _progress("❌ ERREUR: DrizzleProcessor non disponible.")
            return None, None # Re-vérifier

        # Instancier DrizzleProcessor
        mosaic_drizzler = drizzle_processor_class(**drizzle_params) # Utilise kernel/pixfrac passés
        panel_stack_paths = [info[0] for info in stacked_panels_info] # Chemins des stacks panneaux

        # Appel à l'assemblage final Drizzle
        final_mosaic_sci, final_mosaic_wht = mosaic_drizzler.apply_drizzle(
            temp_filepath_list=panel_stack_paths,
            output_wcs=final_output_wcs,           # <<<--- Utiliser la variable correcte
            output_shape_2d_hw=final_output_shape_hw # <<<--- Utiliser la variable correcte
        )
        print(f"DEBUG [MosaicProc result]: Reçu de apply_drizzle -> SCI is None? {final_mosaic_sci is None}, WHT is None? {final_mosaic_wht is None}")
        if final_mosaic_sci is None:
            _progress("❌ ERREUR: Échec de l'assemblage final Drizzle.")
            return None, None

        _progress("✅ Assemblage mosaïque terminé.")
        if final_mosaic_wht is not None: 
            del final_mosaic_wht
            gc.collect() # Nettoyer WHT

        # ========================================================
        # --- 3. Création Header Final et Retour ---
        # ========================================================
        _progress("Création header final et màj compteur...")
        final_header = fits.Header()
        if final_output_wcs: final_header.update(final_output_wcs.to_header(relax=True))
        ref_hdr = getattr(q_manager_instance, 'reference_header_for_wcs', None)
        if ref_hdr: # ... (copie métadonnées) ...
            keys_to_copy=['INSTRUME','TELESCOP','OBJECT','FILTER','DATE-OBS','GAIN','OFFSET','CCD-TEMP','SITELAT','SITELONG','FOCALLEN','APERTURE']
            # Utiliser set pour éviter KeyError si commentaire manque
            for k in keys_to_copy:
                if k in ref_hdr: final_header.set(k, ref_hdr[k], ref_hdr.comments[k] if k in ref_hdr.comments else None)

        final_header['STACKTYP'] = (f'Mosaic Drizzle Panel ({drizzle_params["scale_factor"]:.1f}x)', 'Mosaic from solved panels')
        final_header['DRZSCALE'] = (drizzle_params['scale_factor']); final_header['DRZKERNEL'] = (drizzle_params['kernel']); final_header['DRZPIXFR'] = (drizzle_params['pixfrac'])
        final_header['DRZKERNEL'] = (drizzle_params['kernel'], 'Drizzle kernel')
        final_header['DRZPIXFR'] = (drizzle_params['pixfrac'], 'Drizzle pixfrac')
        final_header['NPANELS'] = (panel_count, 'Number of panels processed')
        # --- MISE A JOUR COMPTEUR PARENT ---
        # Utiliser processed_count de la boucle d'assemblage Drizzle
        # ou num_aligned si on veut le nombre total d'images alignées initialement ?
        # Utilisons processed_count pour le header NIMAGES, mais mettons à jour
        # le compteur global du QM avec num_aligned pour la cohérence du rapport final.
        # images_in_final_mosaic = processed_count # Nombre réellement drizzlé
        final_header['NIMAGES'] = (num_aligned, 'Images combined in final mosaic')
        # Mettre à jour le compteur principal du QueueManager pour le rapport final
        # Utiliser num_aligned (nombre total d'images alignées passées)
        setattr(q_manager_instance, 'images_in_cumulative_stack', num_aligned)
        print(f"DEBUG [MosaicProc]: Mise à jour q_manager.images_in_cumulative_stack = {num_aligned}")
        # --- FIN MISE A JOUR ---
        approx_tot_exp = 0.0; # ... (calcul TOTEXP comme avant) ...
        if ref_hdr and 'EXPTIME' in ref_hdr: 
            try: approx_tot_exp = float(ref_hdr['EXPTIME']) * num_aligned
            except: pass
        final_header['TOTEXP'] = (round(approx_tot_exp, 2), '[s] Approx total exposure processed')
        final_header['ALIGNED'] = (getattr(q_manager_instance,'aligned_files_count',0))
        final_header['FAILALIGN'] = (getattr(q_manager_instance, 'failed_align_count', 0), 'Failed alignments (initial)')
        final_header['FAILSTACK'] = (getattr(q_manager_instance, 'failed_stack_count', 0), 'Images failed in panel stack/solve/drizzle')
        final_header['SKIPPED'] = (getattr(q_manager_instance, 'skipped_files_count', 0), 'Other skipped/error files')

        _progress("Orchestration mosaïque terminée avec succès.")
        # Retourner l'image HxWxC float32 et le header
        return final_mosaic_sci.astype(np.float32), final_header
    except Exception as e:
        _progress(f"❌ ERREUR dans le traitement de la mosaïque: {e}")
        traceback.print_exc()
        return None, None
    
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