"""
Module pour l'alignement des images astronomiques.
Utilise astroalign pour l'enregistrement des images.
"""
import os
import numpy as np
from astropy.io import fits
import cv2
import astroalign as aa
import warnings
import logging
import gc
import shutil
import concurrent.futures
import traceback # Added for traceback printing
try: from tqdm import tqdm # Optionnel, pour la barre de progression console
except ImportError: tqdm = lambda x, **kwargs: x

from .image_processing import (
    load_and_validate_fits, # Returns float32 0-1 or None
    debayer_image,          # Expects float32 0-1, returns float32 0-1
    save_fits_image,        # Expects float32 0-1, saves uint16
    save_preview_image      # For saving reference preview
)
from .hot_pixels import detect_and_correct_hot_pixels

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)

class SeestarAligner:
    """
    Classe pour l'alignement des images astronomiques de Seestar.
    Trouve une image de référence et aligne les autres images sur celle-ci.
    """
    NUM_IMAGES_FOR_AUTO_REF = 50 # Number of initial images to check for reference

    def __init__(self, move_to_unaligned_callback=None):
        """Initialise l'aligneur avec des valeurs par défaut."""
        self.bayer_pattern = "GRBG"
        self.batch_size = 0
        self.reference_image_path = None
        self.correct_hot_pixels = True
        self.hot_pixel_threshold = 3.0
        self.neighborhood_size = 5
        self.stop_processing = False
        self.progress_callback = None
        self.NUM_IMAGES_FOR_AUTO_REF = 20 # Ou une autre valeur par défaut
        self.move_to_unaligned_callback = move_to_unaligned_callback
        self.use_cuda = False
    
    def set_progress_callback(self, callback):
        """Définit la fonction de rappel pour les mises à jour de progression."""
        self.progress_callback = callback

    def update_progress(self, message, progress=None):
        """Met à jour la progression en utilisant le callback si disponible."""
        # Ensure message is a string
        message = str(message)
        if self.progress_callback:
            try:
                self.progress_callback(message, progress)
            except Exception as cb_err:
                print(f"Error in aligner progress callback: {cb_err}")
        else:
            # Basic print fallback if no callback set
            if progress is not None:
                print(f"[{int(progress)}%] {message}")
            else:
                print(message)

    def _align_cpu(self, img: np.ndarray, M: np.ndarray, dsize: tuple[int, int]) -> np.ndarray:
        """Apply affine transform using CPU."""
        if img.ndim == 3:
            result = np.zeros((dsize[1], dsize[0], img.shape[2]), dtype=img.dtype)
            for i in range(img.shape[2]):
                result[:, :, i] = cv2.warpAffine(
                    img[:, :, i], M, dsize,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=np.nan,
                )
            return result
        return cv2.warpAffine(
            img, M, dsize,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )

    def _align_cuda(self, img: np.ndarray, M: np.ndarray, dsize: tuple[int, int]) -> np.ndarray:
        """Apply affine transform using CUDA if available."""
        M3 = np.vstack([M, [0, 0, 1]]).astype(np.float32)
        if img.ndim == 3:
            chans = []
            for i in range(img.shape[2]):
                g = cv2.cuda_GpuMat()
                g.upload(img[:, :, i])
                warped = cv2.cuda.warpPerspective(
                    g, M3, dsize,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                chans.append(warped.download())
            return np.stack(chans, axis=2)
        g = cv2.cuda_GpuMat()
        g.upload(img)
        warped = cv2.cuda.warpPerspective(
            g, M3, dsize,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return warped.download()
# --- DANS LA CLASSE SeestarAligner (dans seestar/core/alignment.py) ---



# --- DANS LA CLASSE SeestarAligner (dans seestar/core/alignment.py) ---
# ... (imports et début de la méthode inchangés) ...

    def _align_image(self, img_to_align, reference_image, file_name, force_same_shape_as_ref=True):
        """
        Aligns a single image to the reference.

        If ``force_same_shape_as_ref`` is True the returned image has exactly the
        same dimensions as ``reference_image``.  This is required for classic
        stacking where all aligned images of a batch must share the same shape.
        When False the output canvas is expanded so that no pixels are cropped.

        Version: AlignFix_ClassicStackingRegression_1
        """
        logger.debug(f"  DEBUG ALIGNER (_align_image V_ClassicStackingRegression_1) pour '{file_name}':")
        logger.debug(f"    force_same_shape_as_ref = {force_same_shape_as_ref}")
        # ... (début de la méthode jusqu'à find_transform inchangé) ...
        if img_to_align is None:
            print(f"    Input img_to_align: None. Retour échec.")
            return None, False 
        
        img_to_align_for_transform_application = img_to_align.astype(np.float32, copy=True) 
        original_min_in = np.nanmin(img_to_align_for_transform_application)
        original_max_in = np.nanmax(img_to_align_for_transform_application)
        input_was_likely_01 = (original_max_in < 1.5 and original_max_in > -0.2 and original_min_in > -0.2 and original_min_in < 1.1)

        print(f"    Input img_to_align (cible de warpAffine) - Range: [{original_min_in:.4g}, {original_max_in:.4g}], Dtype: {img_to_align_for_transform_application.dtype}. Input likely 0-1: {input_was_likely_01}")
        
        if reference_image is None:
            self.update_progress(f"❌ Alignement impossible {file_name}: Référence non disponible.")
            return img_to_align, False

        reference_image_float = reference_image.astype(np.float32, copy=False)

        try:
            source_for_detection = img_to_align_for_transform_application
            if not input_was_likely_01: 
                print(f"    L'entrée est ADU. Normalisation temporaire pour find_transform.")
                s_min_temp, s_max_temp = np.nanmin(source_for_detection), np.nanmax(source_for_detection)
                if s_max_temp > s_min_temp + 1e-7:
                    source_for_detection = (source_for_detection - s_min_temp) / (s_max_temp - s_min_temp)
                else: source_for_detection = np.zeros_like(source_for_detection)
                source_for_detection = np.clip(source_for_detection, 0.0, 1.0)
            
            source_2d_for_detection = source_for_detection[:, :, 1] if source_for_detection.ndim == 3 and source_for_detection.shape[2] == 3 else source_for_detection
            ref_2d_for_detection = reference_image_float[:, :, 1] if reference_image_float.ndim == 3 and reference_image_float.shape[2] == 3 else reference_image_float

            if source_2d_for_detection.shape != ref_2d_for_detection.shape:
                self.update_progress(f"❌ Alignement {file_name}: Dimensions incompatibles pour détection.")
                return img_to_align, False
            
            print(f"    AVANT aa.find_transform: source_2d_for_detection Range: [{np.min(source_2d_for_detection):.4g}, {np.max(source_2d_for_detection):.4g}]")

            transform_skimage_obj, (source_matches, target_matches) = aa.find_transform(
                source=source_2d_for_detection, 
                target=ref_2d_for_detection
            )
            
            if transform_skimage_obj is None:
                raise aa.MaxIterError("aa.find_transform a échoué (pas de transformation trouvée)")
            print(f"    Transformation skimage trouvée. Nb matches: {len(source_matches)}. Type de transform_skimage_obj: {type(transform_skimage_obj)}")

            # --- CORRECTION DE L'EXTRACTION DE LA MATRICE ---
            if hasattr(transform_skimage_obj, 'params') and isinstance(transform_skimage_obj.params, np.ndarray) and transform_skimage_obj.params.shape == (3,3):
                M_sk = transform_skimage_obj.params
                cv2_M = M_sk[0:2, :] # Prendre les 2 premières lignes pour une matrice affine 2x3
                print(f"    Matrice (depuis .params de l'objet skimage) pour OpenCV:\n{cv2_M}")
            else:
                # Si astroalign retourne directement la matrice (moins probable pour find_transform mais sécurité)
                if isinstance(transform_skimage_obj, np.ndarray) and transform_skimage_obj.shape == (3,3):
                    cv2_M = transform_skimage_obj[0:2, :]
                    print(f"    Matrice (directement ndarray 3x3) pour OpenCV:\n{cv2_M}")
                elif isinstance(transform_skimage_obj, np.ndarray) and transform_skimage_obj.shape == (2,3):
                    cv2_M = transform_skimage_obj # C'est déjà le bon format
                    print(f"    Matrice (directement ndarray 2x3) pour OpenCV:\n{cv2_M}")
                else:
                    raise TypeError(f"Type de matrice de transformation inattendu de find_transform: {type(transform_skimage_obj)}, et .params non utilisable.")
            # --- FIN CORRECTION ---
            
            h_ref, w_ref = reference_image_float.shape[:2]

            if force_same_shape_as_ref:
                dsize_cv2 = (w_ref, h_ref)
                cv2_M_final = cv2_M
            else:
                h_src, w_src = img_to_align_for_transform_application.shape[:2]
                corners = np.array([[0, 0], [w_src, 0], [w_src, h_src], [0, h_src]], dtype=np.float32).reshape(-1, 1, 2)
                transformed_corners = cv2.transform(corners, cv2_M)
                x_min, y_min = np.min(transformed_corners, axis=0)[0]
                x_max, y_max = np.max(transformed_corners, axis=0)[0]
                w_out = int(np.ceil(x_max - x_min))
                h_out = int(np.ceil(y_max - y_min))

                shift_M = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
                cv2_M_3x3 = np.vstack([cv2_M, [0, 0, 1]]).astype(np.float32)
                cv2_M_final = (shift_M @ cv2_M_3x3)[:2, :]
                dsize_cv2 = (w_out, h_out)

            align = self._align_cuda if getattr(self, "use_cuda", False) else self._align_cpu
            try:
                aligned_img_final = align(img_to_align_for_transform_application, cv2_M_final, dsize_cv2)
            except Exception as cuda_err:
                if getattr(self, "use_cuda", False):
                    self.use_cuda = False
                    self.update_progress("⚠️ CUDA align failed, falling back to CPU", None)
                    aligned_img_final = self._align_cpu(img_to_align_for_transform_application, cv2_M_final, dsize_cv2)
                else:
                    raise
            
            print(f"    APRÈS cv2.warpAffine: aligned_img_final - Range: [{np.nanmin(aligned_img_final):.4g}, {np.nanmax(aligned_img_final):.4g}]")

            aligned_img_final = np.nan_to_num(aligned_img_final.astype(np.float32, copy=False), nan=0.0)
            
            if input_was_likely_01:
                aligned_img_final = np.clip(aligned_img_final, 0.0, 1.0)
                print(f"    Sortie finale pour entrée ~0-1 (après clip [0,1]): Range: [{np.min(aligned_img_final):.4g}, {np.max(aligned_img_final):.4g}]")
            else:
                aligned_img_final = np.clip(aligned_img_final, 0.0, None)
                print(f"    Sortie finale pour entrée ADU (après clip >=0): Range: [{np.min(aligned_img_final):.4g}, {np.max(aligned_img_final):.4g}]")

            return aligned_img_final, True

        except aa.MaxIterError as ae:
            self.update_progress(f"⚠️ Alignement échoué {file_name}: {ae}")
            return img_to_align, False 
        except ValueError as ve: 
            self.update_progress(f"❌ Erreur alignement {file_name} (ValueError): {ve}")
            traceback.print_exc(limit=1)
            return img_to_align, False 
        except Exception as e:
            self.update_progress(f"❌ Erreur alignement inattendue {file_name}: {e}")
            traceback.print_exc(limit=3)
            return img_to_align, False





# DANS LA CLASSE SeestarAligner (dans seestar/core/alignment.py)

    def _get_reference_image(self, input_folder, files_to_scan, output_folder_for_saving_temp_ref):
        """
        Obtient l'image de référence (float32, 0-1) et son en-tête.
        Sauvegarde l'image de référence traitée dans le dossier temporaire spécifié.
        Le header retourné contient 'HIERARCH SEESTAR REF SRCFILE' avec le nom de base.
        Le header utilisé pour sauvegarder reference_image.fit est épuré pour éviter les erreurs.
        Version: V7_CleanHeaderForTempSave
        """
        print(f"DEBUG ALIGNER [_get_reference_image V7_CleanHeaderForTempSave]: Début. Input: '{os.path.basename(input_folder)}', Nb scan: {len(files_to_scan)}")
        print(f"  Output pour réf. temp: '{output_folder_for_saving_temp_ref}'")
        
        reference_image_data = None 
        # Ce header sera celui retourné et utilisé par le worker pour self.reference_header_for_wcs
        final_reference_header_for_worker = None     
        # Ce header sera celui utilisé pour sauvegarder reference_image.fit (plus épuré)
        header_for_temp_ref_file = None
        
        source_basename_of_selected_ref = None # Nom de base du fichier sélectionné comme référence

        # --- Étape 1: Essayer de Charger une Référence Manuelle si Spécifiée ---
        if self.reference_image_path and os.path.isfile(self.reference_image_path):
            manual_ref_basename = os.path.basename(self.reference_image_path)
            if hasattr(self, 'update_progress'): self.update_progress(f"📌 Chargement référence manuelle: {manual_ref_basename}")
            print(f"DEBUG ALIGNER [_get_reference_image]: Tentative chargement référence manuelle: {self.reference_image_path}")
            try:
                ref_img_tuple_manual = load_and_validate_fits(self.reference_image_path)
                if ref_img_tuple_manual is None or ref_img_tuple_manual[0] is None:
                    raise ValueError(f"Échec chargement/validation (données None) de la référence manuelle: {manual_ref_basename}")
                
                ref_img_loaded_manual, ref_hdr_loaded_manual = ref_img_tuple_manual 
                
                prepared_ref_manual = ref_img_loaded_manual.astype(np.float32) 
                if prepared_ref_manual.ndim == 2: 
                    bayer_pat_ref_manual = ref_hdr_loaded_manual.get('BAYERPAT', self.bayer_pattern)
                    if isinstance(bayer_pat_ref_manual, str) and bayer_pat_ref_manual.upper() in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                        try: prepared_ref_manual = debayer_image(prepared_ref_manual, bayer_pat_ref_manual.upper())
                        except ValueError as deb_err_manual:
                            if hasattr(self, 'update_progress'): self.update_progress(f"⚠️ Réf Manuelle: Erreur Debayer ({deb_err_manual}). Utilisation N&B.")
                if self.correct_hot_pixels:
                    try: prepared_ref_manual = detect_and_correct_hot_pixels(prepared_ref_manual, self.hot_pixel_threshold, self.neighborhood_size)
                    except Exception as hp_err_manual:
                        if hasattr(self, 'update_progress'): self.update_progress(f"⚠️ Réf Manuelle: Erreur correction px chauds: {hp_err_manual}")

                reference_image_data = prepared_ref_manual.astype(np.float32)
                # Le header pour le worker contient toutes les infos originales + notre clé HIERARCH
                final_reference_header_for_worker = ref_hdr_loaded_manual.copy() 
                final_reference_header_for_worker['HIERARCH SEESTAR REF SRCFILE'] = \
                    (str(manual_ref_basename), "Basename of manual reference source")
                
                # Le header pour sauvegarder reference_image.fit est une copie plus simple
                header_for_temp_ref_file = ref_hdr_loaded_manual.copy() 
                # On peut retirer explicitement des clés du header_for_temp_ref_file si elles posent problème
                # à save_fits_image (par exemple, _SOURCE_PATH si load_and_validate_fits l'ajoutait,
                # mais il ne semble pas le faire).

                source_basename_of_selected_ref = manual_ref_basename
                if hasattr(self, 'update_progress'): self.update_progress(f"✅ Référence manuelle chargée et pré-traitée: {manual_ref_basename}")
                
            except Exception as e_manual_ref:
                if hasattr(self, 'update_progress'): self.update_progress(f"❌ Erreur réf. manuelle ({manual_ref_basename}): {e_manual_ref}. Tentative sélection auto...")
                print(f"ERREUR ALIGNER [_get_reference_image]: Échec réf. manuelle: {e_manual_ref}")
                reference_image_data = None # Réinitialiser pour forcer la sélection auto
                final_reference_header_for_worker = None
                header_for_temp_ref_file = None
        else:
            print("DEBUG ALIGNER [_get_reference_image]: Aucune référence manuelle spécifiée ou fichier non trouvé.")
        
        # --- Étape 2: Sélection Automatique si pas de Référence Manuelle Valide ---
        if reference_image_data is None: # Si la référence manuelle a échoué ou n'était pas spécifiée
            if hasattr(self, 'update_progress'): self.update_progress("⚙️ Sélection auto de la meilleure image de référence...")
            print("DEBUG ALIGNER [_get_reference_image]: Passage à la sélection automatique.")

            if not files_to_scan: 
                 if hasattr(self, 'update_progress'): self.update_progress("❌ [GET_REF/Auto] Impossible sélectionner: aucun fichier fourni pour analyse.")
                 return None, None

            best_image_data_auto = None   
            best_header_data_auto_original = None # Header original du fichier sélectionné
            max_metric_auto = -np.inf     
            # best_file_name_auto est déjà initialisé à None en haut
            # ... (logique de filtrage et boucle d'analyse automatique comme avant pour trouver best_image_data_auto, best_header_data_auto_original, best_file_name_auto)
            processed_candidates_auto = 0 
            rejected_candidates_auto = 0  
            rejection_reasons_auto = {'load': 0, 'variance': 0, 'preprocess': 0, 'metric': 0, 'filtered_name': 0, 'load_unpack_fail': 0, 'load_data_none': 0}
            num_to_analyze_initial_subset = min(getattr(self, 'NUM_IMAGES_FOR_AUTO_REF', 20), len(files_to_scan))
            filtered_candidates_for_ref = []
            prefixes_to_skip_for_ref = ["stack_", "mosaic_final_", "aligned_", "drizzle_"] 
            substrings_to_skip_for_ref = ["_reproject", "_sum.", "_wht.", "_preview.", "_temp.", "reference_image.fit", "cumulative_sum.npy", "cumulative_wht.npy"]
            subset_to_filter_names = files_to_scan[:num_to_analyze_initial_subset]
            for f_name_cand_iter in subset_to_filter_names:
                f_name_lower = f_name_cand_iter.lower(); skip_this_file = False
                for prefix in prefixes_to_skip_for_ref:
                    if f_name_lower.startswith(prefix): skip_this_file = True; break
                if skip_this_file: rejected_candidates_auto += 1; rejection_reasons_auto['filtered_name'] += 1; continue
                for substring in substrings_to_skip_for_ref:
                    if substring.lower() in f_name_lower: skip_this_file = True; break
                if skip_this_file: rejected_candidates_auto += 1; rejection_reasons_auto['filtered_name'] += 1; continue
                filtered_candidates_for_ref.append(f_name_cand_iter)
            iterable_candidates = filtered_candidates_for_ref 
            num_to_analyze_auto = len(iterable_candidates) 
            if num_to_analyze_auto == 0:
                 if hasattr(self, 'update_progress'): self.update_progress(f"❌ [GET_REF/Auto] Aucun candidat valide après filtrage des noms (sur {num_to_analyze_initial_subset} scannés).")
                 return None, None
            if hasattr(self, 'update_progress'): self.update_progress(f"🔍 [GET_REF/Auto] Analyse de {num_to_analyze_auto} images candidates pour référence...")
            disable_tqdm_auto = (hasattr(self, 'progress_callback') and self.progress_callback is not None) or (hasattr(self, 'update_progress') and self.update_progress is not None and self.update_progress != print)

            for i_cand, f_name_cand in enumerate(tqdm(iterable_candidates, desc="Analyse Réf. Auto", disable=disable_tqdm_auto, leave=False)):
                if hasattr(self, 'stop_processing') and self.stop_processing: return None, None 
                current_file_path_cand = os.path.join(input_folder, f_name_cand); processed_candidates_auto += 1; rejection_reason_cand = None 
                try:
                    img_data_tuple_cand = load_and_validate_fits(current_file_path_cand)
                    if img_data_tuple_cand is None or img_data_tuple_cand[0] is None: rejection_reason_cand = "load_unpack_fail"; raise ValueError("Load/Validate returned None")
                    img_cand, hdr_cand = img_data_tuple_cand
                    if img_cand is None: rejection_reason_cand = "load_data_none"; raise ValueError("Load/Validate data is None")
                    if hdr_cand is None: hdr_cand = fits.Header() 
                    std_dev_cand = np.std(img_cand); variance_threshold_cand = 0.0005 
                    if std_dev_cand < variance_threshold_cand: rejection_reason_cand = "variance"; raise ValueError(f"Faible variance ({std_dev_cand:.6f})")
                    prepared_img_cand = img_cand.astype(np.float32, copy=True) 
                    if prepared_img_cand.ndim == 2: 
                         bayer_pat_s_cand = hdr_cand.get('BAYERPAT', self.bayer_pattern)
                         if isinstance(bayer_pat_s_cand, str) and bayer_pat_s_cand.upper() in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                              try: prepared_img_cand = debayer_image(prepared_img_cand, bayer_pat_s_cand.upper())
                              except ValueError: pass 
                    if self.correct_hot_pixels:
                         try: prepared_img_cand = detect_and_correct_hot_pixels(prepared_img_cand, self.hot_pixel_threshold, self.neighborhood_size)
                         except Exception: pass 
                    median_val_cand = np.median(prepared_img_cand); mad_val_cand = np.median(np.abs(prepared_img_cand - median_val_cand)); approx_std_cand = mad_val_cand * 1.4826 
                    metric_cand = median_val_cand / (approx_std_cand + 1e-9) if median_val_cand > 1e-9 and approx_std_cand > 1e-9 else -np.inf 
                    if not np.isfinite(metric_cand) or metric_cand < -1e8: rejection_reason_cand = "metric"; raise ValueError(f"Métrique non finie ou trop basse: {metric_cand}")
                    if metric_cand > max_metric_auto:
                        max_metric_auto = metric_cand; best_image_data_auto = prepared_img_cand.copy(); best_header_data_auto_original = hdr_cand.copy(); best_file_name_auto = f_name_cand
                except Exception as e_cand_loop:
                    rejected_candidates_auto += 1
                    if rejection_reason_cand: rejection_reasons_auto[rejection_reason_cand] += 1
                    else: rejection_reasons_auto['preprocess'] += 1 
                finally:
                    if 'img_cand' in locals(): del img_cand; 
                    if 'hdr_cand' in locals(): del hdr_cand
                    if 'prepared_img_cand' in locals(): del prepared_img_cand
                    if 'img_data_tuple_cand' in locals(): del img_data_tuple_cand
                    if i_cand > 0 and i_cand % 10 == 0: gc.collect() 

            if best_image_data_auto is not None and best_header_data_auto_original is not None:
                reference_image_data = best_image_data_auto
                # Le header pour le worker contient toutes les infos originales du fichier sélectionné + notre clé HIERARCH
                final_reference_header_for_worker = best_header_data_auto_original.copy()
                if best_file_name_auto is not None:
                     final_reference_header_for_worker['HIERARCH SEESTAR REF SRCFILE'] = \
                         (str(best_file_name_auto), "Basename of auto-selected reference source")
                
                # Le header pour sauvegarder reference_image.fit est une copie plus simple
                # basé sur le header original du fichier sélectionné, sans notre clé HIERARCH.
                header_for_temp_ref_file = best_header_data_auto_original.copy()
                source_basename_of_selected_ref = best_file_name_auto

                if hasattr(self, 'update_progress'): self.update_progress(f"⭐ Référence auto sélectionnée: {best_file_name_auto} (Métrique: {max_metric_auto:.2f})")
                # ... (logs des rejets)
            else:
                # ... (gestion si aucune référence auto trouvée) ...
                return None, None 

        # --- Étape 3: Sauvegarde de l'image de référence sélectionnée (manuelle ou auto) ---
        if reference_image_data is not None: # Si on a une image de référence à ce stade
            if output_folder_for_saving_temp_ref and os.path.isdir(os.path.dirname(output_folder_for_saving_temp_ref)):
                if header_for_temp_ref_file is None: # Cas où la réf manuelle a été utilisée, mais on n'a pas explicitement créé header_for_temp_ref_file
                    header_for_temp_ref_file = fits.Header() # Un header minimal pour la sauvegarde
                    if final_reference_header_for_worker: # Essayer de copier quelques infos de base si dispo
                        safe_keys_to_copy = ['DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME', 'OBJECT']
                        for k_safe in safe_keys_to_copy:
                            if k_safe in final_reference_header_for_worker:
                                header_for_temp_ref_file[k_safe] = final_reference_header_for_worker[k_safe]
                
                # S'assurer qu'aucune clé potentiellement problématique (comme _SOURCE_PATH original)
                # n'est dans header_for_temp_ref_file avant de le passer à _save_reference_image.
                # Les clés HIERARCH ne devraient pas y être si on part de ref_hdr_loaded_manual ou best_header_data_auto_original.
                # Mais par sécurité, on peut vérifier :
                keys_to_remove_from_temp_header = ['_SOURCE_PATH', 'HIERARCH SEESTAR REF SRCFILE']
                for key_rem in keys_to_remove_from_temp_header:
                    if key_rem in header_for_temp_ref_file:
                        try:
                            del header_for_temp_ref_file[key_rem]
                            print(f"DEBUG ALIGNER: Clé '{key_rem}' supprimée du header pour reference_image.fit")
                        except KeyError: # Peut arriver si HIERARCH est un tuple et n'est pas trouvé directement
                            pass


                print(f"DEBUG ALIGNER [_get_reference_image]: Appel final à _save_reference_image avec base_output_folder='{output_folder_for_saving_temp_ref}' pour source '{source_basename_of_selected_ref}'")
                self._save_reference_image(reference_image_data, header_for_temp_ref_file, output_folder_for_saving_temp_ref)
            else:
                warning_msg_save_final = f"Output_folder_for_saving_temp_ref ('{output_folder_for_saving_temp_ref}') non valide ou son parent n'existe pas. " \
                                         "L'image de référence finale en mémoire sera retournée, mais pas sauvegardée par _get_reference_image."
                if hasattr(self, 'update_progress'): self.update_progress(warning_msg_save_final)
                print(f"AVERTISSEMENT ALIGNER [_get_reference_image]: {warning_msg_save_final}")
        else: 
            print("DEBUG ALIGNER [_get_reference_image]: Données de référence finales non disponibles pour sauvegarde.")
            return None, None # Impossible de continuer si pas d'image de référence

        ref_shape_log = reference_image_data.shape
        print(f"DEBUG ALIGNER [_get_reference_image V7_CleanHeaderForTempSave]: Fin. Réf finale de '{source_basename_of_selected_ref}', Shape: {ref_shape_log}")
        return reference_image_data, final_reference_header_for_worker







    def _save_reference_image(self, reference_image, reference_header, base_output_folder):
        """
        Sauvegarde l'image de référence (float32 0-1) au format FITS (uint16)
        dans le dossier temporaire DEDANS base_output_folder.
        """
        if reference_image is None: return
        # `base_output_folder` is expected to already point to the temporary
        # processing directory (typically `<output>/temp_processing`).
        # Older versions appended an additional "temp_processing" subfolder
        # which caused paths like `<output>/temp_processing/temp_processing`.  
        # To keep compatibility across OSes we now directly use the provided
        # folder without adding another subdirectory.
        temp_folder_ref = base_output_folder
        try:
            os.makedirs(temp_folder_ref, exist_ok=True)
            ref_output_path = os.path.join(temp_folder_ref, "reference_image.fit")
            ref_preview_path = os.path.join(temp_folder_ref, "reference_image.png")
            save_header = reference_header.copy() if reference_header else fits.Header()
            save_header.set('REFRENCE', True, 'Stacking reference image')
            if self.correct_hot_pixels: save_header.set('HOTPIXCR', True, 'Hot pixel correction applied to ref')
            save_header.add_history("Reference image saved by SeestarAligner")
            save_fits_image(reference_image, ref_output_path, save_header, overwrite=True)
            self.update_progress(f"📁 Image référence sauvegardée: {ref_output_path}")
            save_preview_image(reference_image, ref_preview_path, apply_stretch=True)
        except Exception as e:
            self.update_progress(f"⚠️ Erreur lors de la sauvegarde de l'image de référence: {e}")
            traceback.print_exc(limit=2)



    # ... (début de la méthode _align_batch) ...
    def _align_batch(self, images_data, original_indices, reference_image, input_folder, output_folder, unaligned_folder):
        """Aligns a batch of images (data provided) in parallel."""
        num_cores = os.cpu_count() or 1
        max_workers = min(max(num_cores // 2, 1), 8)
        self.update_progress(f"🧵 Alignement parallèle lot avec {max_workers} threads...")

        def align_single_image_task(args):
            idx_in_batch, (img_float_01, hdr, fname), original_file_index = args
            original_file_path = os.path.join(input_folder, fname) # Chemin original complet

            if self.stop_processing:
                return None
            try:
                aligned_img, success = self._align_image(img_float_01, reference_image, fname)
                if not success:
                    # MODIFIÉ : Utiliser le callback de déplacement si fourni
                    if self.move_to_unaligned_callback:
                        try:
                            self.move_to_unaligned_callback(original_file_path) # Appeler le callback
                            self.update_progress(f"   Image '{fname}' déplacée vers 'unaligned_by_stacker' de son dossier source.", "INFO_DETAIL")
                        except Exception as move_cb_err:
                            self.update_progress(f"⚠️ Erreur appel callback déplacement pour '{fname}': {move_cb_err}", "WARN")
                            # Fallback : si le callback échoue, on copie dans l'ancien dossier unaligned_files
                            self._fallback_copy_to_unaligned_folder(original_file_path, unaligned_folder, original_file_index)
                    else:
                        # Si aucun callback n'est fourni, on utilise l'ancien comportement de copie
                        self._fallback_copy_to_unaligned_folder(original_file_path, unaligned_folder, original_file_index)
                        
                    return (original_file_index, False, f"Échec alignement: {fname}") # Retourner l'échec
                
                return (original_file_index, True, aligned_img, hdr)  # index, success, data, header
            except Exception as e:
                error_msg = f"Erreur tâche alignement {fname}: {e}"
                self.update_progress(f"❌ {error_msg}")
                # MODIFIÉ : Appeler le callback ou le fallback de copie en cas d'exception aussi
                if self.move_to_unaligned_callback:
                    try:
                        self.move_to_unaligned_callback(original_file_path) # Appeler le callback
                        self.update_progress(f"   Image '{fname}' (échec exception) déplacée vers 'unaligned_by_stacker' de son dossier source.", "INFO_DETAIL")
                    except Exception as move_cb_err:
                        self.update_progress(f"⚠️ Erreur appel callback déplacement pour '{fname}' (exception): {move_cb_err}", "WARN")
                        self._fallback_copy_to_unaligned_folder(original_file_path, unaligned_folder, original_file_index)
                else:
                    self._fallback_copy_to_unaligned_folder(original_file_path, unaligned_folder, original_file_index)

                return (original_file_index, False, None, None)  # index, success, data, header

        task_args = [(i, data_tuple, original_indices[i]) for i, data_tuple in enumerate(images_data)]
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(align_single_image_task, args): args for args in task_args}
            for future in concurrent.futures.as_completed(futures):
                if self.stop_processing:
                    for f in futures:
                        f.cancel()
                    self.update_progress("⛔ Alignement lot interrompu.")
                    break
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except concurrent.futures.CancelledError:
                    pass
                except Exception as future_err:
                    orig_fname = futures[future][1][2]
                    self.update_progress(f"❗️ Erreur récupération résultat pour {orig_fname}: {future_err}")
                    results.append((futures[future][2], False, None, None))

        success_count = sum(1 for _, success, _, _ in results if success)
        fail_count = len(results) - success_count
        self.update_progress(f"🏁 Alignement lot terminé: {success_count} succès, {fail_count} échecs.")
        return results


    # ... (autres méthodes de SeestarAligner) ...

    def _fallback_copy_to_unaligned_folder(self, original_file_path, unaligned_output_folder, original_file_index):
        """
        Copie un fichier original vers l'ancien dossier 'unaligned_files' dans l'output_folder.
        Utilisé comme fallback si le callback de déplacement local n'est pas dispo ou échoue.
        """
        if os.path.exists(original_file_path):
            try:
                os.makedirs(unaligned_output_folder, exist_ok=True)
                dest_path = os.path.join(unaligned_output_folder, f"unaligned_{original_file_index:04d}_{os.path.basename(original_file_path)}")
                shutil.copy2(original_file_path, dest_path)
                self.update_progress(f"   FallBack: Copié '{os.path.basename(original_file_path)}' vers '{os.path.basename(unaligned_output_folder)}'.", "INFO_DETAIL")
            except Exception as fb_copy_err:
                self.update_progress(f"⚠️ Erreur Fallback Copie '{os.path.basename(original_file_path)}' vers ancien dossier unaligned: {fb_copy_err}", "WARN")
        else:
            self.update_progress(f"   FallBack: Original '{os.path.basename(original_file_path)}' non trouvé pour copie.", "WARN")

    # ... (le reste de la classe SeestarAligner) ...

# --- Compatibility Function (Unchanged) ---
def align_seestar_images_batch(*args, **kwargs):
    """(Compatibility) Use SeestarAligner().align_images(...) directly."""
    aligner = SeestarAligner()
    if 'bayer_pattern' in kwargs: aligner.bayer_pattern = kwargs['bayer_pattern']
    if 'batch_size' in kwargs: aligner.batch_size = kwargs['batch_size']
    if 'reference_image_path' in kwargs: aligner.reference_image_path = kwargs['reference_image_path']
    if 'correct_hot_pixels' in kwargs: aligner.correct_hot_pixels = kwargs['correct_hot_pixels']
    if 'hot_pixel_threshold' in kwargs: aligner.hot_pixel_threshold = kwargs['hot_pixel_threshold']
    if 'neighborhood_size' in kwargs: aligner.neighborhood_size = kwargs['neighborhood_size']
    if 'progress_callback' in kwargs: aligner.set_progress_callback(kwargs['progress_callback'])
    # Call the main method, passing only relevant args
    return aligner.align_images(args[0], kwargs.get('output_folder'), kwargs.get('specific_files'))
# --- END OF FILE seestar/core/alignment.py ---
