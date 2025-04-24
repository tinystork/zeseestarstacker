# --- START OF FILE seestar/queuep/queue_manager.py ---
"""
Module de gestion de file d'attente pour le traitement des images astronomiques.
Gère l'alignement et l'empilement incrémental par LOTS dans un thread séparé.
(Version Révisée 6: Intégration ajout dossiers initiaux + Pondération)
"""
import os
import numpy as np
from astropy.io import fits
import cv2
import threading
from queue import Queue, Empty
import time
import astroalign as aa # Keep astroalign
import traceback
import shutil
import math
import gc
try:
    import cupy
    _cupy_installed = True
except ImportError:
    _cupy_installed = False
# print("DEBUG: CuPy library not found globally.") # Optional debug
# --- Adjust relative import for utils ---
from ..core.utils import check_cupy_cuda, estimate_batch_size # Add check_cupy_cuda
# Import core processing functions needed within this module
from seestar.core.image_processing import (
    load_and_validate_fits,
    debayer_image,
    save_fits_image,
    save_preview_image
)
from seestar.core.hot_pixels import detect_and_correct_hot_pixels
from seestar.core.alignment import SeestarAligner
from seestar.core.utils import estimate_batch_size


class SeestarQueuedStacker:
    """
    Classe pour l'empilement des images Seestar avec file d'attente et traitement par lots.
    Gère l'alignement et l'empilement dans un thread séparé.
    Ajout de la pondération basée sur la qualité (SNR, Nombre d'étoiles).
    """
    def __init__(self):
        # --- State Flags & Control ---
        self.stop_processing = False
        self.processing_active = False
        self.processing_error = None
        # --- Callbacks ---
        self.progress_callback = None
        self.preview_callback = None
        # --- Queue & Threading ---
        self.queue = Queue()
        self.processing_thread = None
        self.folders_lock = threading.Lock()
        # --- File & Folder Management ---
        self.processed_files = set()
        self.additional_folders = []
        self.current_folder = None
        self.output_folder = None
        self.unaligned_folder = None
        # --- Reference Image & Alignment ---
        self.aligner = SeestarAligner() # Uses astroalign
        # --- Batch & Cumulative Stack ---
        # Now stores (aligned_data, header, quality_scores)
        self.current_batch_data = []
        self.current_stack_data = None
        self.current_stack_header = None
        self.images_in_cumulative_stack = 0
        self.total_exposure_seconds = 0.0
        self.final_stacked_path = None
        # --- Processing Parameters ---
        self.stacking_mode = "kappa-sigma"
        self.kappa = 2.5
        self.batch_size = 10
        self.correct_hot_pixels = True
        self.hot_pixel_threshold = 3.0
        self.neighborhood_size = 5
        self.bayer_pattern = "GRBG"
        self.perform_cleanup = True
        # --- Quality Weighting Parameters ---
        self.use_quality_weighting = False
        self.weight_by_snr = True
        self.weight_by_stars = True
        self.snr_exponent = 1.0
        self.stars_exponent = 0.5
        self.min_weight = 0.1
        # --- Statistics ---
        self.files_in_queue = 0
        self.processed_files_count = 0
        self.aligned_files_count = 0
        self.stacked_batches_count = 0
        self.total_batches_estimated = 0
        self.failed_align_count = 0
        self.failed_stack_count = 0
        self.skipped_files_count = 0

    def initialize(self, output_dir):
        try:
            self.output_folder = os.path.abspath(output_dir)
            self.unaligned_folder = os.path.join(self.output_folder, "unaligned_files")
            os.makedirs(self.output_folder, exist_ok=True)
            os.makedirs(self.unaligned_folder, exist_ok=True)
            self.update_progress(f"🗄️ Dossiers prêts: Sortie='{os.path.basename(self.output_folder)}', NonAlign='{os.path.basename(self.unaligned_folder)}'")
        except OSError as e: self.update_progress(f"❌ Erreur création dossiers: {e}", 0); return False
        self.processed_files.clear()
        with self.folders_lock: self.additional_folders = [] # *** Assure que la liste est vide ***
        self.current_batch_data = []
        self.current_stack_data = None; self.current_stack_header = None; self.images_in_cumulative_stack = 0
        self.total_exposure_seconds = 0.0; self.final_stacked_path = None; self.processing_error = None
        self.files_in_queue = 0; self.processed_files_count = 0; self.aligned_files_count = 0
        self.stacked_batches_count = 0; self.total_batches_estimated = 0
        self.failed_align_count = 0; self.failed_stack_count = 0; self.skipped_files_count = 0
        while not self.queue.empty():
             try: self.queue.get_nowait(); self.queue.task_done()
             except Empty: break
             except Exception: pass
        if self.perform_cleanup: self.cleanup_unaligned_files(); self.cleanup_temp_reference()
        self.aligner.stop_processing = False
        return True

    def set_progress_callback(self, callback):
        self.progress_callback = callback
        self.aligner.set_progress_callback(callback)

    def update_progress(self, message, progress=None):
        message = str(message)
        if self.progress_callback:
            try: self.progress_callback(message, progress)
            except Exception as e: print(f"Error in progress callback: {e}")
        else:
            if progress is not None: print(f"[{int(progress)}%] {message}")
            else: print(message)

    def set_preview_callback(self, callback): self.preview_callback = callback

    def _update_preview(self, force_update=False):
        """Safely calls the preview callback, including stack count and batch info."""
        if self.preview_callback is None or self.current_stack_data is None: return
        try:
            data_copy = self.current_stack_data.copy()
            header_copy = self.current_stack_header.copy() if self.current_stack_header else None
            img_count = self.images_in_cumulative_stack; total_imgs_est = self.files_in_queue
            current_batch = self.stacked_batches_count; total_batches_est = self.total_batches_estimated
            stack_name = f"Stack ({img_count}/{total_imgs_est} Img | Batch {current_batch}/{total_batches_est if total_batches_est > 0 else '?'})"
            self.preview_callback(data_copy, header_copy, stack_name, img_count, total_imgs_est, current_batch, total_batches_est)
        except Exception as e: print(f"Error in preview callback: {e}"); traceback.print_exc(limit=2)

    def _recalculate_total_batches(self):
        """Estimates the total number of batches based on files_in_queue."""
        if self.batch_size > 0: self.total_batches_estimated = math.ceil(self.files_in_queue / self.batch_size)
        else: self.update_progress(f"⚠️ Taille de lot invalide ({self.batch_size}), impossible d'estimer le nombre total de lots."); self.total_batches_estimated = 0

    def _worker(self):
        self.processing_active = True; self.processing_error = None; start_time_session = time.monotonic()
        reference_image_data = None; reference_header = None
        try:
            self.update_progress("⭐ Recherche/Préparation image référence...")
            initial_files = []
            if self.current_folder and os.path.isdir(self.current_folder):
                try: initial_files = sorted([f for f in os.listdir(self.current_folder) if f.lower().endswith(('.fit', '.fits'))])
                except Exception as e: self.update_progress(f"Warning: Could not list initial files for ref finding: {e}")
            self.aligner.correct_hot_pixels = self.correct_hot_pixels; self.aligner.hot_pixel_threshold = self.hot_pixel_threshold
            self.aligner.neighborhood_size = self.neighborhood_size; self.aligner.bayer_pattern = self.bayer_pattern
            reference_image_data, reference_header = self.aligner._get_reference_image(self.current_folder, initial_files)
            if reference_image_data is None:
                user_ref_path = self.aligner.reference_image_path
                if user_ref_path and os.path.isfile(user_ref_path): error_msg = f"Échec chargement/prétraitement référence MANUELLE: {os.path.basename(user_ref_path)}"
                elif user_ref_path: error_msg = f"Fichier référence MANUELLE introuvable/invalide: {user_ref_path}"
                else: error_msg = "Échec sélection automatique image référence (vérifiez les premières images et logs)."
                raise RuntimeError(error_msg)
            else:
                self.aligner._save_reference_image(reference_image_data, reference_header, self.output_folder)
                self.update_progress("⭐ Image de référence prête.", 5)

            self._recalculate_total_batches()
            if self.use_quality_weighting: self.update_progress(f"⚖️ Pondération qualité activée (SNR: {self.weight_by_snr}, Stars: {self.weight_by_stars})")

            while not self.stop_processing:
                file_path = None
                try:
                    file_path = self.queue.get(timeout=1.0); file_name = os.path.basename(file_path)
                    aligned_data, header, quality_scores = self._process_file(file_path, reference_image_data)
                    self.processed_files_count += 1
                    if aligned_data is not None:
                        self.current_batch_data.append((aligned_data, header, quality_scores)); self.aligned_files_count += 1
                        if len(self.current_batch_data) >= self.batch_size:
                            self.stacked_batches_count += 1; self._process_completed_batch(self.stacked_batches_count, self.total_batches_estimated)
                    self.queue.task_done()
                    current_progress = (self.processed_files_count / self.files_in_queue) * 100 if self.files_in_queue > 0 else 0
                    elapsed_time = time.monotonic() - start_time_session
                    if self.processed_files_count > 0:
                         time_per_file = elapsed_time / self.processed_files_count; remaining_files_estimate = max(0, self.files_in_queue - self.processed_files_count); eta_seconds = remaining_files_estimate * time_per_file
                         h, rem = divmod(int(eta_seconds), 3600); m, s = divmod(rem, 60); time_str = f"{h:02}:{m:02}:{s:02}"
                         progress_msg = f"📊 ({self.processed_files_count}/{self.files_in_queue}) {file_name} | ETA: {time_str}"
                    else: progress_msg = f"📊 ({self.processed_files_count}/{self.files_in_queue}) {file_name}"
                    self.update_progress(progress_msg, current_progress)
                    if self.processed_files_count % 20 == 0: gc.collect()
                except Empty:
                    self.update_progress("ⓘ File d'attente vide. Vérification batch final / dossiers sup...")
                    if self.current_batch_data:
                        self.update_progress(f"⏳ Traitement du dernier batch ({len(self.current_batch_data)} images)...")
                        self.stacked_batches_count += 1; self._process_completed_batch(self.stacked_batches_count, self.total_batches_estimated)
                    folder_to_process = None
                    with self.folders_lock:
                        if self.additional_folders:
                            folder_to_process = self.additional_folders.pop(0); folder_count = len(self.additional_folders)
                            self.update_progress(f"folder_count_update:{folder_count}")
                    if folder_to_process:
                        folder_name = os.path.basename(folder_to_process); self.update_progress(f"📂 Traitement dossier sup: {folder_name}")
                        self.current_folder = folder_to_process; files_added = self._add_files_to_queue(folder_to_process)
                        if files_added > 0: self.update_progress(f"📋 {files_added} fichiers ajoutés depuis {folder_name}. Total file d'attente: {self.files_in_queue}. Total lots estimé: {self.total_batches_estimated}"); continue
                        else: self.update_progress(f"⚠️ Aucun nouveau FITS trouvé dans {folder_name}"); continue
                    else: self.update_progress("✅ Fin de la file et des dossiers supplémentaires."); break
                except Exception as e:
                    error_context = f" de {file_name}" if file_path else " (file inconnu)"; self.update_progress(f"❌ Erreur boucle worker{error_context}: {e}"); traceback.print_exc(limit=3); self.processing_error = f"Erreur boucle worker: {e}"
                    if file_path: self.skipped_files_count += 1
                    try: self.queue.task_done()
                    except ValueError: pass
                    time.sleep(0.1)

            if self.stop_processing: self.update_progress("⛔ Traitement arrêté par l'utilisateur.")
            else:
                if self.current_stack_data is not None and self.images_in_cumulative_stack > 0:
                    self.update_progress("🏁 Finalisation et sauvegarde du stack final..."); self._save_final_stack(); self._update_preview(force_update=True)
                    self.update_progress(f"🏁 Traitement terminé. {self.images_in_cumulative_stack} images dans le stack final.")
                elif self.processing_error: self.update_progress(f"🏁 Traitement terminé avec erreurs. Erreur principale: {self.processing_error}")
                else: self.update_progress("🏁 Traitement terminé, mais aucun stack n'a été créé (vérifiez les erreurs d'alignement/skip).")

        except RuntimeError as ref_err: self.update_progress(f"❌ ERREUR CRITIQUE: {ref_err}"); self.processing_error = str(ref_err)
        except Exception as e: self.update_progress(f"❌ Erreur critique thread worker: {e}"); traceback.print_exc(limit=5); self.processing_error = f"Erreur critique: {e}"
        finally:
            if self.perform_cleanup: self.update_progress("🧹 Nettoyage fichiers temporaires..."); self.cleanup_unaligned_files(); self.cleanup_temp_reference()
            else: self.update_progress(f"ⓘ Fichiers temporaires/non alignés conservés.")
            self.processing_active = False; self.update_progress("🚪 Thread traitement terminé.")
            while not self.queue.empty():
                try: self.queue.get_nowait(); self.queue.task_done()
                except Exception: break
            self.current_batch_data = []; gc.collect()

    def _calculate_quality_metrics(self, image_data):
        """Calculates SNR and Star Count, WITH ADDED LOGGING.""" # Docstring updated
        scores = {'snr': 0.0, 'stars': 0.0}
        # --- Added: Get filename for logging ---
        # We need the filename here. Since it's not passed directly, we'll have to
        # rely on it being logged just before this function is called in _process_file.
        # This isn't ideal, but avoids major refactoring for diagnostics.
        # The log message in _process_file before calling this will provide context.
        # --- End Added ---

        if image_data is None: return scores # Should not happen if called correctly

        # --- Calculate SNR ---
        snr = 0.0
        try:
            if image_data.ndim == 3 and image_data.shape[2] == 3:
                # Use luminance for SNR calculation
                data_for_snr = 0.299 * image_data[..., 0] + 0.587 * image_data[..., 1] + 0.114 * image_data[..., 2]
            elif image_data.ndim == 2:
                data_for_snr = image_data
            else:
                # self.update_progress(f"⚠️ Format non supporté pour SNR (fichier ?)") # Logged before
                raise ValueError("Unsupported image format for SNR")

            finite_data = data_for_snr[np.isfinite(data_for_snr)]
            if finite_data.size < 50: # Need enough pixels
                 # self.update_progress(f"⚠️ Pas assez de pixels finis pour SNR (fichier ?)") # Logged before
                 raise ValueError("Not enough finite pixels for SNR")

            signal = np.median(finite_data)
            mad = np.median(np.abs(finite_data - signal)) # Median Absolute Deviation
            noise_std = max(mad * 1.4826, 1e-9) # Approx std dev from MAD, avoid zero
            snr = signal / noise_std
            scores['snr'] = np.clip(snr, 0.0, 1000.0) # Clip SNR to a reasonable range

        except Exception as e:
             # Error message will be logged before returning from _process_file
             # self.update_progress(f"⚠️ Erreur calcul SNR (fichier ?): {e}")
             scores['snr'] = 0.0

        # --- Calculate Star Count ---
        num_stars = 0
        try:
             # Use astroalign's transform finding which returns star lists
             # We pass the image as both source and target just to get the detected stars
             transform, (source_list, _target_list) = aa.find_transform(image_data, image_data)
             num_stars = len(source_list)
             max_stars_for_score = 200.0 # Normalize star count relative to a max expected
             scores['stars'] = np.clip(num_stars / max_stars_for_score, 0.0, 1.0) # Score 0-1
        
        except (aa.MaxIterError, aa.MIN_AREA_TOO_LOW, ValueError) as star_err: # Add ValueError here
             # Logged before (or implicitly by the error message from _process_file)
             # Print a slightly more specific message now that we catch ValueError
             self.update_progress(f"      Quality Scores -> Warning: Failed to find enough stars ({type(star_err).__name__}). Stars score set to 0.")
             scores['stars'] = 0.0

        except Exception as e:
             # Logged before
             # self.update_progress(f"⚠️ Erreur calcul nb étoiles (astroalign) (fichier ?): {e}")
             scores['stars'] = 0.0

        # --- ADDED: Print the calculated scores ---
        self.update_progress(f"      Quality Scores -> SNR: {scores['snr']:.2f}, Stars: {scores['stars']:.3f} ({num_stars} raw)")
        # --- END ADDED ---

        return scores


    def _calculate_weights(self, batch_scores):
        num_images = len(batch_scores);
        if num_images == 0: return np.array([])
        raw_weights = np.ones(num_images, dtype=np.float32)
        for i, scores in enumerate(batch_scores):
            weight = 1.0
            if self.weight_by_snr: weight *= max(scores.get('snr', 0.0), 0.0) ** self.snr_exponent
            if self.weight_by_stars: weight *= max(scores.get('stars', 0.0), 0.0) ** self.stars_exponent
            raw_weights[i] = max(weight, 1e-9)
        sum_weights = np.sum(raw_weights)
        if sum_weights > 1e-9: normalized_weights = raw_weights * (num_images / sum_weights)
        else: normalized_weights = np.ones(num_images, dtype=np.float32)
        normalized_weights = np.maximum(normalized_weights, self.min_weight)
        sum_weights_final = np.sum(normalized_weights)
        if sum_weights_final > 1e-9: normalized_weights = normalized_weights * (num_images / sum_weights_final)
        else: normalized_weights = np.ones(num_images, dtype=np.float32)
        return normalized_weights

      
    def _process_file(self, file_path, reference_image_data):
        """Processes a single file: load, validate, preprocess, align, calculate quality."""
        file_name = os.path.basename(file_path)
        quality_scores = {'snr': 0.0, 'stars': 0.0} # Initialize scores
        # --- ADDED: Log filename at start ---
        self.update_progress(f"   Processing File: {file_name}")
        # --- END ADDED ---

        try:
            # 1. Load and Validate
            img_data = load_and_validate_fits(file_path)
            if img_data is None:
                self.update_progress(f"   ⚠️ Échec chargement/validation.") # Message short
                self.skipped_files_count += 1
                return None, None, quality_scores # Return default scores

            header = fits.getheader(file_path) # Get header early

            # 2. Initial Variance Check
            std_dev = np.std(img_data)
            if std_dev < 0.0015: # Adjusted variance threshold slightly lower
                self.update_progress(f"   ⚠️ Image ignorée (faible variance: {std_dev:.4f}).")
                self.skipped_files_count += 1
                return None, None, quality_scores # Return default scores

            # 3. Preprocessing (Debayer, Hot Pixel)
            prepared_img = img_data
            if prepared_img.ndim == 2:
                bayer = header.get('BAYERPAT', self.bayer_pattern)
                if isinstance(bayer, str) and bayer.upper() in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                    try:
                        prepared_img = debayer_image(prepared_img, bayer.upper())
                    except ValueError as de:
                        self.update_progress(f"   ⚠️ Erreur debayer: {de}. Tentative N&B.")
                        # Keep prepared_img as grayscale

            if self.correct_hot_pixels:
                try:
                    prepared_img = detect_and_correct_hot_pixels(
                        prepared_img, self.hot_pixel_threshold, self.neighborhood_size
                    )
                except Exception as hp_err:
                    self.update_progress(f"   ⚠️ Erreur correction px chauds: {hp_err}.")
                    # Continue without correction

            prepared_img = prepared_img.astype(np.float32) # Ensure float32

            # 4. Align Image
            aligned_img, align_success = self.aligner._align_image(prepared_img, reference_image_data, file_name)

            if not align_success:
                self.failed_align_count += 1
                try:
                    if os.path.exists(file_path):
                        # Use normpath for reliability
                        target_path = os.path.join(self.unaligned_folder, file_name)
                        shutil.move(os.path.normpath(file_path), os.path.normpath(target_path))
                        self.update_progress(f"   ➡️ Échec alignement. Déplacé vers {os.path.basename(self.unaligned_folder)}.")
                    else:
                        self.update_progress(f"   ⚠️ Original non trouvé pour déplacer.")
                except Exception as move_err:
                    self.update_progress(f"   ⚠️ Erreur déplacement: {move_err}")
                return None, None, quality_scores # Return default scores

            # 5. Calculate Quality Metrics (IF Weighting Enabled)
            if self.use_quality_weighting:
                # The filename context is logged just above
                quality_scores = self._calculate_quality_metrics(aligned_img)
            else:
                # If weighting disabled, skip calculation but log it
                 self.update_progress(f"      Quality Scores -> Skipped (Weighting Disabled)")


            # 6. Return results
            return aligned_img, header, quality_scores

        except Exception as e:
            # Catch any other error during this file's processing
            self.update_progress(f"❌ Erreur traitement fichier {file_name}: {e}")
            traceback.print_exc(limit=3)
            self.skipped_files_count += 1
            # Try to move the problematic file
            if os.path.exists(file_path):
                try:
                    target_path = os.path.join(self.unaligned_folder, f"error_{file_name}")
                    shutil.move(os.path.normpath(file_path), os.path.normpath(target_path))
                except Exception: pass # Ignore move errors here
            return None, None, quality_scores # Return default scores on error

# --- END Replace _process_file in seestar/queuep/queue_manager.py ---

    
        
    def _process_completed_batch(self, current_batch_num, total_batches_est):
        if not self.current_batch_data: return
        batch_images = [item[0] for item in self.current_batch_data]; batch_headers = [item[1] for item in self.current_batch_data]; batch_scores = [item[2] for item in self.current_batch_data]; batch_size = len(batch_images)
        progress_info = f"(Lot {current_batch_num}/{total_batches_est if total_batches_est > 0 else '?'})"
        self.update_progress(f"⚙️ Traitement du batch {progress_info} - {batch_size} images. Empilement...")
        stacked_batch_data, stack_info_header = self._stack_batch(batch_images, batch_headers, batch_scores, current_batch_num, total_batches_est)
        if stacked_batch_data is not None:
            self._combine_batch_result(stacked_batch_data, stack_info_header); self._update_preview(); self._save_intermediate_stack()
        else:
            self.failed_stack_count += batch_size; self.update_progress(f"❌ Échec empilement lot {progress_info}. {batch_size} images ignorées.", None)
        self.current_batch_data = []; gc.collect()

# --- START OF REPLACEMENT for _stack_batch in queue_manager.py ---

    def _stack_batch(self, batch_images, batch_headers, batch_scores, current_batch_num=0, total_batches_est=0):
        """
        Stacks a batch of images using NumPy or optionally CuPy if available and requested.

        Args:
            batch_images (list): List of NumPy arrays (float32, 0-1).
            batch_headers (list): List of FITS headers.
            batch_scores (list): List of quality score dicts {'snr': float, 'stars': float}.
            current_batch_num (int): Current batch index for logging.
            total_batches_est (int): Estimated total batches for logging.

        Returns:
            tuple: (stacked_image_np, stack_info_header) or (None, None) on failure.
                   stacked_image_np is always a NumPy array.
        """
        if not batch_images: return None, None

        num_images = len(batch_images)
        progress_info = f"(Lot {current_batch_num}/{total_batches_est if total_batches_est > 0 else '?'})"
        stacking_method_info = f"'{self.stacking_mode}'"
        backend_info = "CPU (NumPy)" # Default backend
        use_cupy = False # Flag to track if we actually use cupy for this batch

        # Check if we should attempt CuPy/GPU
        should_try_cupy = _cupy_installed and check_cupy_cuda()

        if should_try_cupy:
            backend_info = "GPU (CuPy)"
            use_cupy = True # Assume we will use it, might be set to False on error

        if self.use_quality_weighting:
             stacking_method_info += " + Pondération"

        self.update_progress(
            f"🧮 Empilement {progress_info} via {backend_info} avec méthode {stacking_method_info} ({num_images} images)..."
        )

        # --- Prepare common variables ---
        try:
            first_image_np = batch_images[0]
            common_shape = first_image_np.shape
            # Ensure output is float32, even if intermediate CuPy uses float64
            output_dtype = np.float32
        except IndexError:
             self.update_progress(f"❌ Erreur: Batch vide reçu dans _stack_batch {progress_info}.")
             return None, None

        # --- Prepare Weights (CPU - always calculated here, uploaded if needed) ---
        weights_1d_np = None
        if self.use_quality_weighting:
            try:
                 weights_1d_np = self._calculate_weights(batch_scores)
                 if len(weights_1d_np) != num_images:
                     self.update_progress(f"❌ Erreur interne poids {progress_info}: N poids != N images. Pondération désactivée pour ce lot.")
                     weights_1d_np = None
                 # else: print(f"DEBUG Weights {progress_info}: {weights_1d_np}") # Debug
            except Exception as w_err:
                 self.update_progress(f"❌ Erreur calcul poids {progress_info}: {w_err}. Pondération désactivée.")
                 weights_1d_np = None


        # --- Main Stacking Logic ---
        stacked_image_np = None # Final result will be a NumPy array


# --- GPU (CuPy) Path ---
        if use_cupy:
            # --- Initialize GPU vars to None ---
            gpu_images_stack_cp = None
            gpu_weights_1d_cp = None
            weights_broadcast_cp = None
            gpu_stacked_image_cp = None
            gpu_mean_cp = None
            gpu_std_cp = None
            gpu_threshold_cp = None
            gpu_lower_bound_cp = None
            gpu_upper_bound_cp = None
            gpu_mask_cp = None
            gpu_masked_weights_cp = None
            gpu_sum_weights_kept_cp = None
            gpu_weighted_sum_kept_cp = None
            gpu_count_kept_cp = None
            gpu_sum_image_cp = None
            gpu_clipped_stack_cp = None
            # --- End Initialization ---

            try:
                # 1. Upload data to GPU
                start_upload = time.monotonic()
                gpu_images_stack_cp = cupy.asarray(batch_images, dtype=cupy.float32)
                upload_time = time.monotonic() - start_upload

                if weights_1d_np is not None:
                    gpu_weights_1d_cp = cupy.asarray(weights_1d_np, dtype=cupy.float32)
                    weight_shape = [-1] + [1] * (gpu_images_stack_cp.ndim - 1)
                    weights_broadcast_cp = gpu_weights_1d_cp.reshape(weight_shape)

                # 2. Perform Stacking on GPU
                start_compute = time.monotonic()
                # --- (Rest of the CuPy Stacking Methods logic stays the same) ---
                if self.stacking_mode == "mean":
                    # ... (mean logic) ...
                     if weights_broadcast_cp is not None:
                        gpu_stacked_image_cp = cupy.average(gpu_images_stack_cp, axis=0, weights=gpu_weights_1d_cp)
                     else:
                        gpu_stacked_image_cp = cupy.mean(gpu_images_stack_cp, axis=0)

                elif self.stacking_mode == "median":
                     # ... (median logic) ...
                    if weights_broadcast_cp is not None:
                         self.update_progress(f"⚠️ Pondération non supportée pour 'median' avec CuPy, utilisation mediane simple {progress_info}.")
                    gpu_stacked_image_cp = cupy.median(gpu_images_stack_cp, axis=0)


                elif self.stacking_mode in ["kappa-sigma", "winsorized-sigma"]:
                     # ... (kappa/winsorized logic requires gpu_mean_cp, gpu_std_cp etc.) ...
                    gpu_mean_cp = cupy.mean(gpu_images_stack_cp, axis=0, dtype=cupy.float32) # <--- Error happened here in your trace
                    gpu_std_cp = cupy.std(gpu_images_stack_cp, axis=0, dtype=cupy.float32)
                    # ... (rest of kappa/winsorized) ...
                    gpu_std_cp = cupy.maximum(gpu_std_cp, 1e-6) # Avoid division by zero
                    gpu_threshold_cp = self.kappa * gpu_std_cp

                    if self.stacking_mode == "kappa-sigma":
                        gpu_lower_bound_cp = gpu_mean_cp - gpu_threshold_cp
                        gpu_upper_bound_cp = gpu_mean_cp + gpu_threshold_cp
                        gpu_mask_cp = (gpu_images_stack_cp >= gpu_lower_bound_cp) & (gpu_images_stack_cp <= gpu_upper_bound_cp)
                        # ... (rest of weighted/unweighted kappa-sigma) ...
                        if weights_broadcast_cp is not None:
                            gpu_masked_weights_cp = cupy.where(gpu_mask_cp, weights_broadcast_cp, 0.0)
                            gpu_sum_weights_kept_cp = cupy.sum(gpu_masked_weights_cp, axis=0)
                            gpu_sum_weights_kept_cp = cupy.maximum(gpu_sum_weights_kept_cp, 1e-9)
                            gpu_weighted_sum_kept_cp = cupy.sum(cupy.where(gpu_mask_cp, gpu_images_stack_cp * gpu_masked_weights_cp, 0.0), axis=0)
                            gpu_stacked_image_cp = gpu_weighted_sum_kept_cp / gpu_sum_weights_kept_cp
                        else:
                            gpu_count_kept_cp = cupy.sum(gpu_mask_cp, axis=0, dtype=cupy.int16)
                            gpu_count_kept_cp = cupy.maximum(gpu_count_kept_cp, 1)
                            gpu_sum_image_cp = cupy.sum(cupy.where(gpu_mask_cp, gpu_images_stack_cp, 0.0), axis=0)
                            gpu_stacked_image_cp = gpu_sum_image_cp / gpu_count_kept_cp

                    elif self.stacking_mode == "winsorized-sigma":
                        gpu_upper_bound_cp = gpu_mean_cp + gpu_threshold_cp # Typo fixed: should be gpu_mean_cp + gpu_threshold_cp
                        gpu_clipped_stack_cp = cupy.clip(gpu_images_stack_cp, gpu_lower_bound_cp, gpu_upper_bound_cp)
                        if weights_broadcast_cp is not None:
                            gpu_stacked_image_cp = cupy.average(gpu_clipped_stack_cp, axis=0, weights=gpu_weights_1d_cp)
                        else:
                             gpu_stacked_image_cp = cupy.mean(gpu_clipped_stack_cp, axis=0)


                else: # Fallback method
                     # ... (fallback logic) ...
                    self.update_progress(f"⚠️ Méthode CuPy '{self.stacking_mode}' non reconnue, utilisation 'mean' {progress_info}.")
                    if weights_broadcast_cp is not None:
                        gpu_stacked_image_cp = cupy.average(gpu_clipped_stack_cp, axis=0, weights=gpu_weights_1d_cp)
                    else:
                        gpu_stacked_image_cp = cupy.mean(gpu_images_stack_cp, axis=0)


                # 3. Download result from GPU
                start_download = time.monotonic()
                if gpu_stacked_image_cp is None: raise ValueError("CuPy stacking did not produce a result.")
                stacked_image_np = cupy.asnumpy(gpu_stacked_image_cp).astype(output_dtype)
                download_time = time.monotonic() - start_download
                compute_time = time.monotonic() - start_compute - download_time
                self.update_progress(f"   ⏱️ CuPy Times {progress_info}: Upload={upload_time:.3f}s, Compute={compute_time:.3f}s, Download={download_time:.3f}s")

            # --- (Keep except blocks as they were) ---
            except cupy.cuda.memory.OutOfMemoryError as mem_err:
                # ... (existing code) ...
                 print(f"\nERROR CuPy {progress_info}: GPU Out of Memory! {mem_err}")
                 self.update_progress(f"❌ ERREUR CuPy {progress_info}: GPU Manque de Mémoire! Essai avec CPU...")
                 use_cupy = False # Trigger CPU fallback
                 gc.collect() # CPU garbage collect
                 cupy.get_default_memory_pool().free_all_blocks() # Free GPU blocks

            except Exception as gpu_err:
                 # ... (existing code) ...
                 print(f"\nERROR CuPy {progress_info}: {gpu_err}")
                 traceback.print_exc(limit=3)
                 self.update_progress(f"❌ ERREUR CuPy {progress_info}: {gpu_err}. Essai avec CPU...")
                 use_cupy = False # Trigger CPU fallback
                 gc.collect()
                 try: cupy.get_default_memory_pool().free_all_blocks()
                 except Exception: pass

            finally:
                # --- Modified finally block ---
                # Explicitly delete CuPy arrays *if they were assigned*
                if gpu_images_stack_cp is not None: del gpu_images_stack_cp
                if gpu_weights_1d_cp is not None: del gpu_weights_1d_cp
                if weights_broadcast_cp is not None: del weights_broadcast_cp
                if gpu_stacked_image_cp is not None: del gpu_stacked_image_cp
                if gpu_mean_cp is not None: del gpu_mean_cp
                if gpu_std_cp is not None: del gpu_std_cp
                if gpu_threshold_cp is not None: del gpu_threshold_cp
                if gpu_lower_bound_cp is not None: del gpu_lower_bound_cp
                if gpu_upper_bound_cp is not None: del gpu_upper_bound_cp
                if gpu_mask_cp is not None: del gpu_mask_cp
                if gpu_masked_weights_cp is not None: del gpu_masked_weights_cp
                if gpu_sum_weights_kept_cp is not None: del gpu_sum_weights_kept_cp
                if gpu_weighted_sum_kept_cp is not None: del gpu_weighted_sum_kept_cp
                if gpu_count_kept_cp is not None: del gpu_count_kept_cp
                if gpu_sum_image_cp is not None: del gpu_sum_image_cp
                if gpu_clipped_stack_cp is not None: del gpu_clipped_stack_cp

                # Free memory pool blocks
                if '_cupy_installed' in globals() and _cupy_installed:
                    try: cupy.get_default_memory_pool().free_all_blocks()
                    except Exception as free_err: print(f"Warning: Error freeing CuPy memory blocks: {free_err}")
# --- End Modified finally block ---

        if not use_cupy:
            if backend_info == "GPU (CuPy)": # Check if we are here due to fallback
                 self.update_progress(f"   -> Exécution via CPU (NumPy) pour {progress_info} après échec GPU.")
            try:
                # Use float32 for NumPy calculations for consistency and potentially less memory
                image_stack_np = np.stack([img.astype(output_dtype) for img in batch_images], axis=0)

                # --- NumPy Stacking Methods ---
                stacked_image_np_calc = None
                with np.errstate(divide='ignore', invalid='ignore'): # Suppress expected warnings
                    if self.stacking_mode == "mean":
                        if weights_1d_np is not None:
                            stacked_image_np_calc = np.average(image_stack_np, axis=0, weights=weights_1d_np)
                        else:
                            stacked_image_np_calc = np.mean(image_stack_np, axis=0)
                    elif self.stacking_mode == "median":
                        if weights_1d_np is not None:
                             self.update_progress(f"⚠️ Pondération non supportée pour 'median' avec NumPy, utilisation mediane simple {progress_info}.")
                        stacked_image_np_calc = np.median(image_stack_np, axis=0)
                    elif self.stacking_mode in ["kappa-sigma", "winsorized-sigma"]:
                        # Calculate mean and std using float32
                        mean_np = np.mean(image_stack_np, axis=0, dtype=np.float32)
                        std_np = np.std(image_stack_np, axis=0, dtype=np.float32)
                        std_np = np.maximum(std_np, 1e-6) # Avoid division by zero
                        threshold_np = self.kappa * std_np

                        if self.stacking_mode == "kappa-sigma":
                            lower_bound_np = mean_np - threshold_np
                            upper_bound_np = mean_np + threshold_np
                            mask_np = (image_stack_np >= lower_bound_np) & (image_stack_np <= upper_bound_np)

                            if weights_1d_np is not None:
                                weight_shape = [-1] + [1] * (image_stack_np.ndim - 1)
                                weights_broadcast_np = weights_1d_np.reshape(weight_shape)
                                masked_weights_np = np.where(mask_np, weights_broadcast_np, 0.0)
                                sum_weights_kept_np = np.sum(masked_weights_np, axis=0)
                                sum_weights_kept_np = np.maximum(sum_weights_kept_np, 1e-9)
                                weighted_sum_kept_np = np.sum(np.where(mask_np, image_stack_np * masked_weights_np, 0.0), axis=0)
                                stacked_image_np_calc = weighted_sum_kept_np / sum_weights_kept_np
                            else:
                                count_kept_np = np.sum(mask_np, axis=0, dtype=np.int16)
                                count_kept_np = np.maximum(count_kept_np, 1)
                                sum_image_np = np.sum(np.where(mask_np, image_stack_np, 0.0), axis=0)
                                stacked_image_np_calc = sum_image_np / count_kept_np

                            # rejected_count_np = image_stack_np.size - np.sum(mask_np)
                            # rejected_percent_np = (rejected_count_np / image_stack_np.size) * 100 if image_stack_np.size > 0 else 0
                            # self.update_progress(f"   (NumPy Kappa-Sigma {progress_info}: {rejected_percent_np:.2f}% pixels rejetés)")

                        elif self.stacking_mode == "winsorized-sigma":
                            lower_bound_np = mean_np - threshold_np
                            upper_bound_np = mean_np + threshold_np
                            clipped_stack_np = np.clip(image_stack_np, lower_bound_np, upper_bound_np)
                            if weights_1d_np is not None:
                                 stacked_image_np_calc = np.average(clipped_stack_np, axis=0, weights=weights_1d_np)
                            else:
                                 stacked_image_np_calc = np.mean(clipped_stack_np, axis=0)
                            # self.update_progress(f"   (NumPy Winsorized-Sigma {progress_info} appliqué)")
                    else: # Fallback method
                        self.update_progress(f"⚠️ Méthode NumPy '{self.stacking_mode}' non reconnue, utilisation 'mean' {progress_info}.")
                        if weights_1d_np is not None:
                            stacked_image_np_calc = np.average(image_stack_np, axis=0, weights=weights_1d_np)
                        else:
                            stacked_image_np_calc = np.mean(image_stack_np, axis=0)

                if stacked_image_np_calc is None:
                    raise ValueError("NumPy stacking did not produce a result.")

                stacked_image_np = np.nan_to_num(stacked_image_np_calc).astype(output_dtype)

            except Exception as cpu_err:
                print(f"\nERROR NumPy {progress_info}: {cpu_err}")
                traceback.print_exc(limit=3)
                self.update_progress(f"❌ ERREUR NumPy empilement {progress_info}: {cpu_err}. Lot ignoré.")
                gc.collect()
                return None, None # Failed to stack batch

            finally:
                # Clean up large intermediate NumPy arrays if created
                del image_stack_np, mean_np, std_np # Add others if needed
                gc.collect()

        # --- Final Processing & Header ---
        if stacked_image_np is not None:
            stacked_image_np = np.clip(stacked_image_np, 0.0, 1.0)

            # Create Header
            stack_info_header = fits.Header()
            stack_info_header['NIMAGES'] = (num_images, 'Images in this batch stack')
            stack_info_header['STACKMETH'] = (self.stacking_mode, 'Stacking method used')
            stack_info_header['BACKEND'] = ('GPU (CuPy)' if use_cupy and stacked_image_np is not None else 'CPU (NumPy)', 'Processing Backend')
            stack_info_header['WGHT_USED'] = (self.use_quality_weighting and weights_1d_np is not None, 'Quality weighting applied')
            if self.stacking_mode in ["kappa-sigma", "winsorized-sigma"]:
                 stack_info_header['KAPPA'] = (self.kappa, 'Kappa value used')

            # Sum exposure from headers
            batch_exposure = sum(float(h.get('EXPTIME', 0.0)) for h in batch_headers if h is not None)
            stack_info_header['TOTEXP'] = (round(batch_exposure, 2), '[s] Exposure time of this batch')

            return stacked_image_np, stack_info_header
        else:
             # Should only happen if both CuPy and NumPy paths failed critically
             self.update_progress(f"❌ Échec critique empilement lot {progress_info}.")
             return None, None
    
    def _combine_batch_result(self, stacked_batch_data_np, stack_info_header):
        """Combines the result of a stacked batch into the cumulative stack."""
        try:
            batch_n = int(stack_info_header.get('NIMAGES', 1))
            batch_exposure = float(stack_info_header.get('TOTEXP', 0.0))
            if batch_n <= 0:
                self.update_progress("⚠️ Batch combiné avait 0 images, ignoré.", None)
                return

            # --- Initialize Cumulative Stack if it's the first batch ---
            if self.current_stack_data is None:
                self.current_stack_data = stacked_batch_data_np.copy() # Should be float32 numpy
                self.images_in_cumulative_stack = batch_n
                self.total_exposure_seconds = batch_exposure
                # --- Create initial header ---
                self.current_stack_header = fits.Header()
                first_header = self.current_batch_data[0][1] if self.current_batch_data else fits.Header() # Use first image header from the *original* batch data
                keys_to_copy = ['INSTRUME', 'TELESCOP', 'OBJECT', 'FILTER', 'DATE-OBS', 'GAIN', 'OFFSET', 'CCD-TEMP', 'RA', 'DEC', 'SITELAT', 'SITELONG', 'FOCALLEN', 'BAYERPAT']
                for key in keys_to_copy:
                    if first_header and key in first_header:
                        try: self.current_stack_header[key] = (first_header[key], first_header.comments[key] if key in first_header.comments else '')
                        except Exception: self.current_stack_header[key] = first_header[key]
                self.current_stack_header['STACKTYP'] = (self.stacking_mode, 'Stacking method')
                self.current_stack_header['NIMAGES'] = (self.images_in_cumulative_stack, 'Images in cumulative stack')
                self.current_stack_header['TOTEXP'] = (round(self.total_exposure_seconds, 2), '[s] Total exposure time')
                if self.stacking_mode in ["kappa-sigma", "winsorized-sigma"]: self.current_stack_header['KAPPA'] = (self.kappa, 'Kappa value for clipping')
                if self.use_quality_weighting:
                    self.current_stack_header['WGHT_ON'] = (True, 'Quality weighting enabled'); w_metrics = []
                    if self.weight_by_snr: w_metrics.append(f"SNR^{self.snr_exponent:.1f}")
                    if self.weight_by_stars: w_metrics.append(f"Stars^{self.stars_exponent:.1f}")
                    self.current_stack_header['WGHT_MET'] = (",".join(w_metrics), 'Metrics used for weighting')
                else: self.current_stack_header['WGHT_ON'] = (False, 'Quality weighting disabled')
                self.current_stack_header['CREATOR'] = ('SeestarStacker (Queued)', 'Processing Software')
                self.current_stack_header['HISTORY'] = 'Cumulative Stack Initialized'
                if self.correct_hot_pixels: self.current_stack_header['HISTORY'] = 'Hot pixel correction applied to input frames'

            # --- Combine with Existing Cumulative Stack ---
            else:
                if self.current_stack_data.shape != stacked_batch_data_np.shape:
                    self.update_progress(f"❌ Incompatibilité dims stack: Cumul={self.current_stack_data.shape}, Batch={stacked_batch_data_np.shape}. Combinaison échouée.")
                    return

                current_n = self.images_in_cumulative_stack
                total_n = current_n + batch_n
                w_old = current_n / total_n
                w_new = batch_n / total_n

                # --- Check if CuPy should be used for combination ---
                use_cupy_combine = _cupy_installed and check_cupy_cuda()
                combined_np = None

                if use_cupy_combine:
                    gpu_current = None; gpu_batch = None
                    try:
                        # print("DEBUG: Combining stacks using CuPy")
                        # Ensure data is float32 for GPU
                        gpu_current = cupy.asarray(self.current_stack_data, dtype=cupy.float32)
                        gpu_batch = cupy.asarray(stacked_batch_data_np, dtype=cupy.float32)

                        # Perform weighted average on GPU
                        gpu_combined = (gpu_current * w_old) + (gpu_batch * w_new)
                        combined_np = cupy.asnumpy(gpu_combined) # Download result
                        # print("DEBUG: CuPy combination successful")

                    except cupy.cuda.memory.OutOfMemoryError:
                        print("Warning: GPU Out of Memory during stack combination. Falling back to CPU.")
                        use_cupy_combine = False
                        gc.collect()
                        cupy.get_default_memory_pool().free_all_blocks()
                    except Exception as gpu_err:
                        print(f"Warning: CuPy error during stack combination: {gpu_err}. Falling back to CPU.")
                        use_cupy_combine = False
                        gc.collect()
                        try: cupy.get_default_memory_pool().free_all_blocks()
                        except Exception: pass
                    finally:
                        del gpu_current, gpu_batch # Free GPU memory

                # Fallback to NumPy if CuPy not used or failed
                if not use_cupy_combine:
                    # print("DEBUG: Combining stacks using NumPy")
                    # Ensure float32 for calculation precision/memory
                    combined_np = (self.current_stack_data.astype(np.float32) * w_old) + \
                                  (stacked_batch_data_np.astype(np.float32) * w_new)

                # Update the cumulative stack (must be float32 numpy array)
                self.current_stack_data = combined_np.astype(np.float32)

                # Update cumulative stats and header
                self.images_in_cumulative_stack = total_n
                self.total_exposure_seconds += batch_exposure
                self.current_stack_header['NIMAGES'] = self.images_in_cumulative_stack
                self.current_stack_header['TOTEXP'] = (round(self.total_exposure_seconds, 2), '[s] Total exposure time')
                self.current_stack_header.add_history(f'Combined with batch stack of {batch_n} images')

            # Clip final result
            self.current_stack_data = np.clip(self.current_stack_data, 0.0, 1.0)

        except Exception as e:
            self.update_progress(f"❌ Erreur combinaison du résultat du batch: {e}")
            traceback.print_exc(limit=3)
            
    def _save_intermediate_stack(self):
        if self.current_stack_data is None or self.output_folder is None: return
        stack_path = os.path.join(self.output_folder, "stack_cumulative.fit"); preview_path = os.path.join(self.output_folder, "stack_cumulative.png")
        try:
            header_to_save = self.current_stack_header.copy() if self.current_stack_header else fits.Header()
            try:
                if 'HISTORY' in header_to_save:
                    history_entries = list(header_to_save['HISTORY']); filtered_history = [h for h in history_entries if 'Intermediate save' not in str(h)]
                    while 'HISTORY' in header_to_save: del header_to_save['HISTORY']
                    for entry in filtered_history: header_to_save.add_history(entry)
            except Exception: pass
            header_to_save.add_history(f'Intermediate save after combining {self.images_in_cumulative_stack} images')
            save_fits_image(self.current_stack_data, stack_path, header_to_save, overwrite=True)
            save_preview_image(self.current_stack_data, preview_path, apply_stretch=True)
        except Exception as e: print(f"⚠️ Erreur sauvegarde stack intermédiaire: {e}")

    def _save_final_stack(self):
        if self.current_stack_data is None or self.output_folder is None or self.images_in_cumulative_stack == 0: self.final_stacked_path = None; self.update_progress("ⓘ Aucun stack final à sauvegarder."); return
        self.final_stacked_path = os.path.join(self.output_folder, f"stack_final_{self.stacking_mode}{'_wght' if self.use_quality_weighting else ''}.fit"); preview_path = os.path.splitext(self.final_stacked_path)[0] + ".png"; self.update_progress(f"💾 Sauvegarde stack final: {os.path.basename(self.final_stacked_path)}...")
        try:
            final_header = self.current_stack_header.copy() if self.current_stack_header else fits.Header()
            try:
                if 'HISTORY' in final_header:
                    history_entries = list(final_header['HISTORY']); filtered_history = [h for h in history_entries if 'Intermediate save' not in str(h)]
                    while 'HISTORY' in final_header: del final_header['HISTORY']
                    for entry in filtered_history: final_header.add_history(entry)
            except Exception: pass
            final_header.add_history('Final Stack Saved by Seestar Stacker (Queued)')
            final_header['NIMAGES'] = (self.images_in_cumulative_stack, 'Number of images combined in final stack'); final_header['TOTEXP'] = (round(self.total_exposure_seconds, 2), '[s] Total exposure time')
            final_header['ALIGNED'] = (self.aligned_files_count, 'Successfully aligned images'); final_header['FAILALIGN'] = (self.failed_align_count, 'Failed alignments')
            final_header['FAILSTACK'] = (self.failed_stack_count, 'Files skipped due to batch stack errors'); final_header['SKIPPED'] = (self.skipped_files_count, 'Other skipped/error files')
            if 'STACKTYP' not in final_header: final_header['STACKTYP'] = self.stacking_mode
            if self.stacking_mode in ["kappa-sigma", "winsorized-sigma"] and 'KAPPA' not in final_header: final_header['KAPPA'] = self.kappa
            if 'WGHT_ON' not in final_header: final_header['WGHT_ON'] = (self.use_quality_weighting, 'Quality weighting status')
            if self.use_quality_weighting and 'WGHT_MET' not in final_header:
                w_metrics = [];
                if self.weight_by_snr: w_metrics.append(f"SNR^{self.snr_exponent:.1f}")
                if self.weight_by_stars: w_metrics.append(f"Stars^{self.stars_exponent:.1f}")
                final_header['WGHT_MET'] = (",".join(w_metrics), 'Metrics used for weighting')
            save_fits_image(self.current_stack_data, self.final_stacked_path, final_header, overwrite=True)
            save_preview_image(self.current_stack_data, preview_path, apply_stretch=True)
            self.update_progress(f"✅ Stack final sauvegardé ({self.images_in_cumulative_stack} images)")
        except Exception as e: self.update_progress(f"⚠️ Erreur sauvegarde stack final: {e}"); traceback.print_exc(limit=2); self.final_stacked_path = None

    def cleanup_unaligned_files(self):
        if not self.unaligned_folder or not os.path.isdir(self.unaligned_folder): return
        deleted_count = 0
        try:
            for filename in os.listdir(self.unaligned_folder):
                file_path = os.path.join(self.unaligned_folder, filename);
                if os.path.isfile(file_path):
                    try: os.remove(file_path); deleted_count += 1
                    except Exception as del_e: self.update_progress(f"⚠️ Erreur suppression non aligné {filename}: {del_e}")
            if deleted_count > 0: self.update_progress(f"🧹 {deleted_count} fichier(s) non aligné(s) supprimé(s).")
        except Exception as e: self.update_progress(f"⚠️ Erreur nettoyage non alignés: {e}")

    def cleanup_temp_reference(self):
        try:
            aligner_temp_folder = os.path.join(self.output_folder, "temp_processing")
            if os.path.isdir(aligner_temp_folder):
                ref_fit = os.path.join(aligner_temp_folder, "reference_image.fit")
                ref_png = os.path.join(aligner_temp_folder, "reference_image.png")
                deleted_ref = 0
                if os.path.exists(ref_fit):
                    try:
                        os.remove(ref_fit)
                        deleted_ref += 1
                    except Exception:
                        pass
                if os.path.exists(ref_png):
                    try:
                        os.remove(ref_png)
                        deleted_ref += 1
                    except Exception:
                        pass
                if deleted_ref > 0:
                    self.update_progress(f"🧹 Fichier(s) référence temporaire(s) supprimé(s).")
                try:
                    os.rmdir(aligner_temp_folder)
                except OSError:
                    pass
        except Exception as e:
            self.update_progress(f"⚠️ Erreur nettoyage référence temp: {e}")

    def add_folder(self, folder_path):
        if not self.processing_active: self.update_progress("ⓘ Impossible d'ajouter un dossier, traitement non actif."); return False
        abs_path = os.path.abspath(folder_path)
        if not os.path.isdir(abs_path): self.update_progress(f"❌ Dossier non trouvé: {folder_path}"); return False
        output_abs = os.path.abspath(self.output_folder) if self.output_folder else None
        if output_abs:
             norm_abs_path = os.path.normcase(abs_path); norm_output_path = os.path.normcase(output_abs)
             if norm_abs_path == norm_output_path or norm_abs_path.startswith(norm_output_path + os.sep): self.update_progress(f"⚠️ Impossible d'ajouter le dossier de sortie: {os.path.basename(folder_path)}"); return False
        with self.folders_lock:
            current_abs = os.path.abspath(self.current_folder) if self.current_folder else None
            existing_abs = [os.path.abspath(p) for p in self.additional_folders]
            if (current_abs and abs_path == current_abs) or abs_path in existing_abs: self.update_progress(f"ⓘ Dossier déjà en cours ou ajouté: {os.path.basename(folder_path)}"); return False
            self.additional_folders.append(abs_path); folder_count = len(self.additional_folders)
        self.update_progress(f"✅ Dossier ajouté à la file d'attente : {os.path.basename(folder_path)}")
        self.update_progress(f"folder_count_update:{folder_count}")
        return True

    def _add_files_to_queue(self, folder_path):
        count_added = 0
        try:
            abs_folder_path = os.path.abspath(folder_path); self.update_progress(f"🔍 Scan du dossier: {os.path.basename(folder_path)}...")
            files_in_folder = sorted(os.listdir(abs_folder_path)); new_files_found_in_folder = []
            for fname in files_in_folder:
                if self.stop_processing: self.update_progress("⛔ Scan interrompu."); break
                if fname.lower().endswith(('.fit', '.fits')):
                    fpath = os.path.join(abs_folder_path, fname); abs_fpath = os.path.abspath(fpath)
                    if abs_fpath not in self.processed_files:
                        self.queue.put(fpath); self.processed_files.add(abs_fpath); count_added += 1
            if count_added > 0: self.files_in_queue += count_added; self._recalculate_total_batches()
            return count_added
        except FileNotFoundError: self.update_progress(f"❌ Erreur scan: Dossier introuvable {os.path.basename(folder_path)}"); return 0
        except PermissionError: self.update_progress(f"❌ Erreur scan: Permission refusée {os.path.basename(folder_path)}"); return 0
        except Exception as e: self.update_progress(f"❌ Erreur scan dossier {os.path.basename(folder_path)}: {e}"); return 0

    # --- MODIFIED start_processing ---
    def start_processing(self, input_dir, output_dir, reference_path_ui=None, initial_additional_folders=None,
                         # --- Weighting parameters from GUI/Settings ---
                         use_weighting=False, weight_snr=True, weight_stars=True,
                         snr_exp=1.0, stars_exp=0.5, min_w=0.1):
        """Démarre le thread de traitement avec configuration de pondération et dossiers initiaux."""
        if self.processing_active: self.update_progress("⚠️ Traitement déjà en cours."); return False
        self.stop_processing = False; self.current_folder = os.path.abspath(input_dir)
        # *** Initialize vide self.additional_folders ***
        if not self.initialize(output_dir): self.processing_active = False; return False

        if self.batch_size < 3: self.update_progress(f"⚠️ Taille de lot ({self.batch_size}) trop petite, ajustée à 3.", None); self.batch_size = 3
        self.update_progress(f"ⓘ Taille de lot utilisée : {self.batch_size}")

        # --- Apply Weighting Config ---
        self.use_quality_weighting = use_weighting; self.weight_by_snr = weight_snr; self.weight_by_stars = weight_stars
        self.snr_exponent = max(0.1, snr_exp); self.stars_exponent = max(0.1, stars_exp); self.min_weight = max(0.01, min(1.0, min_w))

        # --- NOUVEAU : Ajouter les dossiers initiaux ---
        initial_folders_to_add_count = 0
        with self.folders_lock:
            # Assurer que la liste est vide avant d'ajouter
            self.additional_folders = []
            if initial_additional_folders:
                for folder in initial_additional_folders:
                    abs_folder = os.path.abspath(folder)
                    if os.path.isdir(abs_folder) and abs_folder not in self.additional_folders:
                         self.additional_folders.append(abs_folder)
                         initial_folders_to_add_count += 1
                    else: print(f"Debug QMgr: Skipped invalid/duplicate initial folder: {folder}")
            if initial_folders_to_add_count > 0:
                 self.update_progress(f"ⓘ {initial_folders_to_add_count} dossier(s) pré-ajouté(s) seront traités.")
                 self.update_progress(f"folder_count_update:{len(self.additional_folders)}")
        # --- Fin Nouveau ---

        initial_files_added = self._add_files_to_queue(self.current_folder)
        if initial_files_added > 0: self.update_progress(f"📋 {initial_files_added} fichiers initiaux ajoutés. Total lots estimé: {self.total_batches_estimated}")
        elif not self.additional_folders:
             if reference_path_ui: self.update_progress("⚠️ Aucun fichier initial. Attente ajout dossiers via bouton...")
             else: self.update_progress("⚠️ Aucun fichier initial trouvé et aucun dossier supp. pré-ajouté. Démarrage quand même pour référence auto...")

        # Start Thread
        self.aligner.reference_image_path = reference_path_ui or None
        self.processing_thread = threading.Thread(target=self._worker, name="StackerWorker"); self.processing_thread.daemon = True; self.processing_thread.start(); self.processing_active = True
        return True

    def stop(self):
        if not self.processing_active: return
        self.update_progress("⛔ Arrêt demandé..."); self.stop_processing = True; self.aligner.stop_processing = True

    def is_running(self):
        return self.processing_active and self.processing_thread is not None and self.processing_thread.is_alive()

# --- END OF FILE seestar/queuep/queue_manager.py ---