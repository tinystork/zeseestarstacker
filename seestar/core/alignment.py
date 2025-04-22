# --- START OF FILE seestar/core/alignment.py ---
"""
Module pour l'alignement des images astronomiques.
Utilise astroalign pour l'enregistrement des images.
"""
import os
import numpy as np
from astropy.io import fits
import cv2
import astroalign as aa
from tqdm import tqdm
import warnings
import gc
import time
import shutil
import concurrent.futures
from functools import partial
import traceback # Added for traceback printing


from .image_processing import (
    load_and_validate_fits, # Returns float32 0-1 or None
    debayer_image,          # Expects float32 0-1, returns float32 0-1
    save_fits_image,        # Expects float32 0-1, saves uint16
    save_preview_image      # For saving reference preview
)
from .hot_pixels import detect_and_correct_hot_pixels
from .utils import estimate_batch_size

warnings.filterwarnings("ignore", category=FutureWarning)

class SeestarAligner:
    """
    Classe pour l'alignement des images astronomiques de Seestar.
    Trouve une image de référence et aligne les autres images sur celle-ci.
    """
    NUM_IMAGES_FOR_AUTO_REF = 50 # Number of initial images to check for reference

    def __init__(self):
        """Initialise l'aligneur avec des valeurs par défaut."""
        self.bayer_pattern = "GRBG"
        self.batch_size = 0
        self.reference_image_path = None
        self.correct_hot_pixels = True
        self.hot_pixel_threshold = 3.0
        self.neighborhood_size = 5
        self.stop_processing = False
        self.progress_callback = None

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

    def align_images(self, input_folder, output_folder=None, specific_files=None):
        """
        (Primarily used by QueueManager to get reference image info now)
        Finds reference image. Standalone alignment loop is commented out.
        """
        self.stop_processing = False
        if output_folder is None: output_folder = os.path.join(input_folder, "aligned_lights")
        try:
            os.makedirs(output_folder, exist_ok=True)
            unaligned_folder = os.path.join(output_folder, "unaligned") # Still create for consistency
            os.makedirs(unaligned_folder, exist_ok=True)
        except OSError as e:
            self.update_progress(f"❌ Erreur création dossier sortie/unaligned: {e}")
            return None # Critical error
        try:
            if specific_files: files = [f for f in specific_files if f.lower().endswith(('.fit', '.fits'))]
            else: files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.fit', '.fits'))]
            files.sort()
        except FileNotFoundError: self.update_progress(f"❌ Dossier d'entrée non trouvé: {input_folder}"); return None
        except Exception as e: self.update_progress(f"❌ Erreur lecture dossier entrée: {e}"); return None
        if not files: self.update_progress("❌ Aucun fichier .fit/.fits trouvé à traiter."); return output_folder

        # Estimate batch size (still useful for other parts or general info)
        if self.batch_size <= 0:
            if files: # Check if files list is not empty
                 sample_path = os.path.join(input_folder, files[0])
                 try:
                     self.batch_size = estimate_batch_size(sample_path)
                     self.update_progress(f"🧠 Taille de lot dynamique estimée : {self.batch_size}")
                 except Exception as est_err:
                     self.update_progress(f"⚠️ Erreur estimation taille lot: {est_err}. Utilisation valeur défaut 10.")
                     self.batch_size = 10
            else: # No files, use default batch size
                 self.batch_size = 10

        self.update_progress("⭐ Recherche/Préparation image de référence...")
        fixed_reference_image, fixed_reference_header = self._get_reference_image(input_folder, files)

        if fixed_reference_image is None:
            # Error message now generated within _get_reference_image or QueueManager
            # self.update_progress("❌ Impossible d'obtenir une image de référence valide. Arrêt.")
            return None # Signal failure

        self.update_progress(f"✅ Recherche référence terminée.")
        return output_folder


    def _get_reference_image(self, input_folder, files):
        """
        Obtient l'image de référence (float32, 0-1) et son en-tête.
        Analyse un nombre fixe d'images initiales pour la sélection auto.
        """
        reference_image = None
        reference_header = None
        processed_candidates = 0
        rejected_candidates = 0
        rejection_reasons = {'load': 0, 'variance': 0, 'preprocess': 0, 'metric': 0}

        # --- Try Manual Reference Path ---
        if self.reference_image_path and os.path.isfile(self.reference_image_path):
            manual_ref_basename = os.path.basename(self.reference_image_path)
            try:
                self.update_progress(f"📌 Chargement référence manuelle: {manual_ref_basename}")
                ref_img_loaded = load_and_validate_fits(self.reference_image_path)
                if ref_img_loaded is None: raise ValueError(f"Échec chargement/validation: {manual_ref_basename}")

                ref_hdr_loaded = fits.getheader(self.reference_image_path)
                prepared_ref = ref_img_loaded
                if prepared_ref.ndim == 2:
                    bayer_pat_ref = ref_hdr_loaded.get('BAYERPAT', self.bayer_pattern)
                    if isinstance(bayer_pat_ref, str) and bayer_pat_ref.upper() in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                        try: prepared_ref = debayer_image(prepared_ref, bayer_pat_ref.upper())
                        except ValueError as deb_err: self.update_progress(f"⚠️ Réf Manuelle: Erreur Debayer {deb_err}. Utilisation N&B.")
                if self.correct_hot_pixels:
                    try:
                        self.update_progress("🔥 Correction px chauds sur référence manuelle...")
                        prepared_ref = detect_and_correct_hot_pixels(prepared_ref, self.hot_pixel_threshold, self.neighborhood_size)
                    except Exception as hp_err: self.update_progress(f"⚠️ Réf Manuelle: Erreur correction px chauds: {hp_err}")

                reference_image = prepared_ref.astype(np.float32)
                reference_header = ref_hdr_loaded
                self.update_progress(f"✅ Référence manuelle chargée: dims {reference_image.shape}")

            except Exception as e:
                self.update_progress(f"❌ Erreur chargement/préparation référence manuelle ({manual_ref_basename}): {e}. Tentative sélection auto...")
                reference_image = None # Force auto-selection

        # --- Auto-Select Reference if needed ---
        if reference_image is None:
            self.update_progress("⚙️ Sélection auto de la meilleure image de référence...")
            if not files:
                 self.update_progress("❌ Impossible sélectionner référence: aucun fichier d'entrée fourni.")
                 return None, None # Cannot find reference if no files exist

            best_image_data = None; best_header_data = None; best_file_name = None
            max_metric = -np.inf # Start with negative infinity

            num_to_analyze = min(self.NUM_IMAGES_FOR_AUTO_REF, len(files))
            self.update_progress(f"🔍 Analyse des {num_to_analyze} premières images pour référence...")

            iterable = files[:num_to_analyze]
            disable_tqdm = self.progress_callback is not None
            with tqdm(total=num_to_analyze, desc="Analyse réf.", disable=disable_tqdm, leave=False) as pbar:
                for i, f in enumerate(iterable):
                    if self.stop_processing: return None, None
                    file_path = os.path.join(input_folder, f)
                    processed_candidates += 1
                    rejection_reason = None # Track why this specific file was rejected

                    try:
                        # --- Load Candidate ---
                        img = load_and_validate_fits(file_path)
                        if img is None: rejection_reason = "load"; raise ValueError("Load fail")

                        hdr = fits.getheader(file_path)

                        # --- Basic Quality Check ---
                        std_dev = np.std(img)
                        if std_dev < 0.005: rejection_reason = "variance"; raise ValueError(f"Low variance ({std_dev:.4f})")

                        # --- Preprocess Candidate ---
                        prepared_img = img
                        if prepared_img.ndim == 2:
                             bayer_pat_s = hdr.get('BAYERPAT', self.bayer_pattern)
                             if isinstance(bayer_pat_s, str) and bayer_pat_s.upper() in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                                  try: prepared_img = debayer_image(prepared_img, bayer_pat_s.upper())
                                  except ValueError as de: print(f"Debug RefScan: Err Debayer {f}: {de}"); # Keep grayscale
                        if self.correct_hot_pixels:
                             try: prepared_img = detect_and_correct_hot_pixels(prepared_img, self.hot_pixel_threshold, self.neighborhood_size)
                             except Exception as hpe: print(f"Debug RefScan: Err HPX {f}: {hpe}"); # Continue without correction

                        # Ensure float32 for metric calculation
                        prepared_img = prepared_img.astype(np.float32)

                        # --- Calculate Metric ---
                        median_val = np.median(prepared_img)
                        mad_val = np.median(np.abs(prepared_img - median_val))
                        approx_std = mad_val * 1.4826
                        metric = median_val / (approx_std + 1e-6) if median_val > 0 else -np.inf
                        if not np.isfinite(metric): rejection_reason = "metric"; raise ValueError("Metric non-finite")


                        # --- Compare and Store Best ---
                        if metric > max_metric:
                            # print(f"Debug RefScan: New best {f}, Metric={metric:.2f} (Prev Max={max_metric:.2f})") # Debug
                            max_metric = metric
                            best_image_data = prepared_img # Store preprocessed version
                            best_header_data = hdr
                            best_file_name = f
                        # else: print(f"Debug RefScan: Skip {f}, Metric={metric:.2f} (Best={max_metric:.2f})") # Debug

                    except Exception as e:
                        # Catch errors during processing of a single candidate
                        self.update_progress(f"⚠️ Erreur analyse réf {f}: {e}")
                        rejected_candidates += 1
                        if rejection_reason: rejection_reasons[rejection_reason] += 1
                        else: rejection_reasons['preprocess'] += 1 # Count other errors as preprocess errors
                    finally:
                        pbar.update(1) # Update tqdm progress bar

            # --- Final Check after loop ---
            if best_image_data is not None:
                reference_image = best_image_data
                reference_header = best_header_data
                self.update_progress(f"⭐ Référence auto sélectionnée: {best_file_name} (Metric: {max_metric:.2f})")
                # Report rejection stats if any were rejected
                if rejected_candidates > 0:
                     reason_str = ", ".join(f"{k}:{v}" for k,v in rejection_reasons.items() if v > 0)
                     self.update_progress(f"   (Info: {processed_candidates} analysés, {rejected_candidates} rejetés [{reason_str}])")
            else:
                # Report failure and reasons
                reason_str = ", ".join(f"{k}:{v}" for k,v in rejection_reasons.items() if v > 0)
                self.update_progress(f"❌ Aucune référence valide trouvée après analyse de {processed_candidates} images. Raisons rejet: [{reason_str}]")
                return None, None # Explicitly return None to signal failure

        return reference_image, reference_header


    # --- _save_reference_image (Unchanged) ---
    def _save_reference_image(self, reference_image, reference_header, base_output_folder):
        """
        Sauvegarde l'image de référence (float32 0-1) au format FITS (uint16)
        dans le dossier temporaire DEDANS base_output_folder.
        """
        if reference_image is None: return
        temp_folder_ref = os.path.join(base_output_folder, "temp_processing")
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

    # --- _align_image (Unchanged) ---
    def _align_image(self, img_to_align, reference_image, file_name):
        """Aligns a single image (float32 0-1) to the reference (float32 0-1)."""
        if reference_image is None: self.update_progress(f"❌ Alignement impossible {file_name}: Référence non disponible."); return img_to_align, False
        img_to_align = img_to_align.astype(np.float32); reference_image = reference_image.astype(np.float32)
        try:
            img_for_detection = img_to_align[:, :, 1] if img_to_align.ndim == 3 else img_to_align
            ref_for_detection = reference_image[:, :, 1] if reference_image.ndim == 3 else reference_image
            if img_for_detection.shape != ref_for_detection.shape: self.update_progress(f"❌ Alignement {file_name}: Dimensions incompatibles Réf={ref_for_detection.shape}, Img={img_for_detection.shape}"); return img_to_align, False
            aligned_img, _ = aa.register(source=img_to_align, target=reference_image, max_control_points=50, detection_sigma=5, min_area=5)
            if aligned_img is None: raise aa.MaxIterError("Alignement échoué (pas de transformation trouvée)")
            aligned_img = np.clip(aligned_img.astype(np.float32), 0.0, 1.0); return aligned_img, True
        except aa.MaxIterError as ae: self.update_progress(f"⚠️ Alignement échoué {file_name}: {ae}"); return img_to_align, False
        except ValueError as ve: self.update_progress(f"❌ Erreur alignement {file_name} (ValueError): {ve}"); return img_to_align, False
        except Exception as e: self.update_progress(f"❌ Erreur alignement inattendue {file_name}: {e}"); traceback.print_exc(limit=3); return img_to_align, False

# --- _align_batch (Unchanged - returns results, doesn't save aligned files here) ---
    def _align_batch(self, images_data, original_indices, reference_image, input_folder, output_folder, unaligned_folder):
        """Aligns a batch of images (data provided) in parallel."""
        num_cores = os.cpu_count() or 1
        max_workers = min(max(num_cores // 2, 1), 8)
        self.update_progress(f"🧵 Alignement parallèle lot avec {max_workers} threads...")

        def align_single_image_task(args):
            idx_in_batch, (img_float_01, hdr, fname), original_file_index = args
            if self.stop_processing:
                return None
            try:
                aligned_img, success = self._align_image(img_float_01, reference_image, fname)
                if not success:
                    original_path = os.path.join(input_folder, fname)
                    out_path = os.path.join(unaligned_folder, f"unaligned_{original_file_index:04d}_{fname}")
                    if os.path.exists(original_path):
                        shutil.copy2(original_path, out_path)
                    return (original_file_index, False, f"Échec alignement: {fname}")
                return (original_file_index, True, aligned_img, hdr)  # index, success, data, header
            except Exception as e:
                error_msg = f"Erreur tâche alignement {fname}: {e}"
                self.update_progress(f"❌ {error_msg}")
                try:
                    original_path = os.path.join(input_folder, fname)
                    out_path = os.path.join(unaligned_folder, f"error_{original_file_index:04d}_{fname}")
                    if os.path.exists(original_path):
                        shutil.copy2(original_path, out_path)
                except Exception:
                    pass
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