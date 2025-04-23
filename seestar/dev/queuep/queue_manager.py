# --- START OF FILE seestar/queuep/queue_manager.py ---
"""
Module de gestion de file d'attente pour le traitement des images astronomiques.
Gère l'alignement et l'empilement incrémental par LOTS dans un thread séparé.
(Version Révisée 5: Intégration Pondération Qualité - SNR/Stars)
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
import gc
import math

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
        # --- *** NEW: Quality Weighting Parameters *** ---
        self.use_quality_weighting = False # Master switch for weighting
        self.weight_by_snr = True          # Use SNR for weighting?
        self.weight_by_stars = True        # Use Star Count for weighting?
        self.snr_exponent = 1.0            # Weight = SNR^exponent (1=linear, 2=quadratic)
        self.stars_exponent = 0.5          # Weight = Stars^exponent (0.5=sqrt)
        self.min_weight = 0.1              # Minimum relative weight assigned
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
        # (Code initialize identique à la version précédente)
        try:
            self.output_folder = os.path.abspath(output_dir)
            self.unaligned_folder = os.path.join(self.output_folder, "unaligned_files")
            os.makedirs(self.output_folder, exist_ok=True)
            os.makedirs(self.unaligned_folder, exist_ok=True)
            self.update_progress(f"🗄️ Dossiers prêts: Sortie='{os.path.basename(self.output_folder)}', NonAlign='{os.path.basename(self.unaligned_folder)}'")
        except OSError as e: self.update_progress(f"❌ Erreur création dossiers: {e}", 0); return False
        self.processed_files.clear()
        with self.folders_lock: self.additional_folders = []
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

    # --- Callbacks and Progress Update (Identiques) ---
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

    # --- _update_preview (Identique à version 4) ---
    def _update_preview(self, force_update=False):
        """Safely calls the preview callback, including stack count and batch info."""
        if self.preview_callback is None or self.current_stack_data is None:
            return
        try:
            data_copy = self.current_stack_data.copy()
            header_copy = self.current_stack_header.copy() if self.current_stack_header else None
            img_count = self.images_in_cumulative_stack
            total_imgs_est = self.files_in_queue
            current_batch = self.stacked_batches_count
            total_batches_est = self.total_batches_estimated
            stack_name = f"Stack ({img_count}/{total_imgs_est} Img | Batch {current_batch}/{total_batches_est if total_batches_est > 0 else '?'})"
            self.preview_callback(
                data_copy, header_copy, stack_name,
                img_count, total_imgs_est, current_batch, total_batches_est
            )
        except Exception as e:
            print(f"Error in preview callback: {e}")
            traceback.print_exc(limit=2)

    # --- _recalculate_total_batches (Identique à version 4) ---
    def _recalculate_total_batches(self):
        """Estimates the total number of batches based on files_in_queue."""
        if self.batch_size > 0:
            self.total_batches_estimated = math.ceil(self.files_in_queue / self.batch_size)
        else:
            self.update_progress(f"⚠️ Taille de lot invalide ({self.batch_size}), impossible d'estimer le nombre total de lots.")
            self.total_batches_estimated = 0

    # --- Worker Thread Logic ---
    def _worker(self):
        # --- Start worker ---
        self.processing_active = True
        self.processing_error = None
        start_time_session = time.monotonic()
        reference_image_data = None
        reference_header = None
        try:
            # Step 1: Obtain Reference Image (Identique)
            self.update_progress("⭐ Recherche/Préparation image référence...")
            initial_files = []
            if self.current_folder and os.path.isdir(self.current_folder):
                try: initial_files = sorted([f for f in os.listdir(self.current_folder) if f.lower().endswith(('.fit', '.fits'))])
                except Exception as e: self.update_progress(f"Warning: Could not list initial files for ref finding: {e}")
            self.aligner.correct_hot_pixels = self.correct_hot_pixels
            self.aligner.hot_pixel_threshold = self.hot_pixel_threshold
            self.aligner.neighborhood_size = self.neighborhood_size
            self.aligner.bayer_pattern = self.bayer_pattern
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

            # --- Initialize Batch Estimate ---
            self._recalculate_total_batches()
            if self.use_quality_weighting:
                self.update_progress(f"⚖️ Pondération qualité activée (SNR: {self.weight_by_snr}, Stars: {self.weight_by_stars})")

            # --- Step 2: Process Queue ---
            while not self.stop_processing:
                file_path = None
                try:
                    file_path = self.queue.get(timeout=1.0)
                    file_name = os.path.basename(file_path)

                    # --- Process File (Now includes quality calculation) ---
                    aligned_data, header, quality_scores = self._process_file(file_path, reference_image_data)

                    self.processed_files_count += 1
                    if aligned_data is not None:
                        # --- Store data with quality scores ---
                        self.current_batch_data.append((aligned_data, header, quality_scores))
                        self.aligned_files_count += 1

                        # --- Process Batch if Full ---
                        if len(self.current_batch_data) >= self.batch_size:
                            self.stacked_batches_count += 1
                            self._process_completed_batch(self.stacked_batches_count, self.total_batches_estimated)

                    self.queue.task_done()

                    # Progress Reporting (Identique)
                    current_progress = (self.processed_files_count / self.files_in_queue) * 100 if self.files_in_queue > 0 else 0
                    elapsed_time = time.monotonic() - start_time_session
                    if self.processed_files_count > 0:
                         time_per_file = elapsed_time / self.processed_files_count
                         remaining_files_estimate = max(0, self.files_in_queue - self.processed_files_count)
                         eta_seconds = remaining_files_estimate * time_per_file
                         h, rem = divmod(int(eta_seconds), 3600); m, s = divmod(rem, 60)
                         time_str = f"{h:02}:{m:02}:{s:02}"
                         progress_msg = f"📊 ({self.processed_files_count}/{self.files_in_queue}) {file_name} | ETA: {time_str}"
                    else: progress_msg = f"📊 ({self.processed_files_count}/{self.files_in_queue}) {file_name}"
                    self.update_progress(progress_msg, current_progress)
                    if self.processed_files_count % 20 == 0: gc.collect()

                except Empty:
                    # Queue Empty Logic (Identique)
                    self.update_progress("ⓘ File d'attente vide. Vérification batch final / dossiers sup...")
                    if self.current_batch_data:
                        self.update_progress(f"⏳ Traitement du dernier batch ({len(self.current_batch_data)} images)...")
                        self.stacked_batches_count += 1
                        self._process_completed_batch(self.stacked_batches_count, self.total_batches_estimated)
                    folder_to_process = None
                    with self.folders_lock:
                        if self.additional_folders:
                            folder_to_process = self.additional_folders.pop(0)
                            folder_count = len(self.additional_folders)
                            self.update_progress(f"folder_count_update:{folder_count}")
                    if folder_to_process:
                        folder_name = os.path.basename(folder_to_process)
                        self.update_progress(f"📂 Traitement dossier sup: {folder_name}")
                        self.current_folder = folder_to_process
                        files_added = self._add_files_to_queue(folder_to_process)
                        if files_added > 0:
                            self.update_progress(f"📋 {files_added} fichiers ajoutés depuis {folder_name}. Total file d'attente: {self.files_in_queue}. Total lots estimé: {self.total_batches_estimated}")
                            continue
                        else:
                            self.update_progress(f"⚠️ Aucun nouveau FITS trouvé dans {folder_name}")
                            continue
                    else:
                        self.update_progress("✅ Fin de la file et des dossiers supplémentaires.")
                        break
                except Exception as e:
                    # Error Handling (Identique)
                    error_context = f" de {file_name}" if file_path else " (file inconnu)"
                    self.update_progress(f"❌ Erreur boucle worker{error_context}: {e}")
                    traceback.print_exc(limit=3); self.processing_error = f"Erreur boucle worker: {e}"
                    if file_path: self.skipped_files_count += 1
                    try: self.queue.task_done()
                    except ValueError: pass
                    time.sleep(0.1)

            # --- Finalization ---
            if self.stop_processing: self.update_progress("⛔ Traitement arrêté par l'utilisateur.")
            else:
                if self.current_stack_data is not None and self.images_in_cumulative_stack > 0:
                    self.update_progress("🏁 Finalisation et sauvegarde du stack final...")
                    self._save_final_stack(); self._update_preview(force_update=True)
                    self.update_progress(f"🏁 Traitement terminé. {self.images_in_cumulative_stack} images dans le stack final.")
                elif self.processing_error: self.update_progress(f"🏁 Traitement terminé avec erreurs. Erreur principale: {self.processing_error}")
                else: self.update_progress("🏁 Traitement terminé, mais aucun stack n'a été créé (vérifiez les erreurs d'alignement/skip).")

        except RuntimeError as ref_err: self.update_progress(f"❌ ERREUR CRITIQUE: {ref_err}"); self.processing_error = str(ref_err)
        except Exception as e: self.update_progress(f"❌ Erreur critique thread worker: {e}"); traceback.print_exc(limit=5); self.processing_error = f"Erreur critique: {e}"
        finally:
            # Cleanup (Identique)
            if self.perform_cleanup: self.update_progress("🧹 Nettoyage fichiers temporaires..."); self.cleanup_unaligned_files(); self.cleanup_temp_reference()
            else: self.update_progress(f"ⓘ Fichiers temporaires/non alignés conservés.")
            self.processing_active = False; self.update_progress("🚪 Thread traitement terminé.")
            while not self.queue.empty():
                try: self.queue.get_nowait(); self.queue.task_done()
                except Exception: break
            self.current_batch_data = []; gc.collect()


    # --- *** NEW: Quality Metric Calculation *** ---
    def _calculate_quality_metrics(self, image_data):
        """Calcule les métriques de qualité (SNR, Stars) pour une image alignée."""
        scores = {'snr': 0.0, 'stars': 0.0}
        if image_data is None: return scores

        # --- 1. SNR Calculation ---
        try:
            # Use luminance for color images
            if image_data.ndim == 3 and image_data.shape[2] == 3:
                # Use standard luminance weights
                data_for_snr = 0.299 * image_data[..., 0] + 0.587 * image_data[..., 1] + 0.114 * image_data[..., 2]
            elif image_data.ndim == 2:
                data_for_snr = image_data
            else:
                raise ValueError("Format d'image non supporté pour SNR")

            # Ensure finite values for calculations
            finite_data = data_for_snr[np.isfinite(data_for_snr)]
            if finite_data.size < 50: # Need a minimum number of pixels
                 raise ValueError("Pas assez de pixels finis pour SNR")

            signal = np.median(finite_data)
            # Robust noise estimation using MAD
            mad = np.median(np.abs(finite_data - signal))
            # Convert MAD to equivalent standard deviation for Gaussian noise
            # Handle case where MAD is zero or very close to zero
            noise_std = max(mad * 1.4826, 1e-9)
            # Calculate SNR
            snr = signal / noise_std
            # Clip unreasonable SNR values (e.g., from nearly black images or saturated)
            scores['snr'] = np.clip(snr, 0.0, 1000.0) # Limit max SNR
        except Exception as e:
            self.update_progress(f"⚠️ Erreur calcul SNR: {e}")
            scores['snr'] = 0.0 # Default to low score on error

        # --- 2. Star Count Calculation (using astroalign) ---
        try:
            # Run find_transform on the image against itself to get source list
            # Use a slightly higher detection sigma maybe? Default is 5.
            # Use detection_sigma=5 as aligner does
            # max_control_points=500 ? Increase point search? Keep aligner defaults for now.
            transform, (source_list, _target_list) = aa.find_transform(image_data, image_data) # target list not needed
            num_stars = len(source_list)

            # Normalize the star count score (e.g., target ~100-200 stars for good weight)
            max_stars_for_score = 200.0
            scores['stars'] = np.clip(num_stars / max_stars_for_score, 0.0, 1.0)

        except aa.MaxIterError:
             self.update_progress("⚠️ Aucune étoile trouvée par astroalign (MaxIterError).")
             scores['stars'] = 0.0 # Penalize heavily if no stars found
        except aa.MIN_AREA_TOO_LOW:
             self.update_progress("⚠️ Aucune étoile trouvée par astroalign (MIN_AREA_TOO_LOW).")
             scores['stars'] = 0.0
        except Exception as e:
             self.update_progress(f"⚠️ Erreur calcul nb étoiles (astroalign): {e}")
             scores['stars'] = 0.0 # Default to low score on other errors

        # Add debug print for scores if needed
        # print(f"DEBUG - Scores: SNR={scores['snr']:.2f}, Stars={scores['stars']:.2f} ({num_stars} raw)")

        return scores


    # --- *** NEW: Weight Calculation *** ---
    def _calculate_weights(self, batch_scores):
        """Calcule les poids normalisés pour un lot basé sur les scores."""
        num_images = len(batch_scores)
        if num_images == 0: return np.array([])

        raw_weights = np.ones(num_images, dtype=np.float32)

        for i, scores in enumerate(batch_scores):
            weight = 1.0
            # Apply weights based on configuration
            if self.weight_by_snr:
                snr_score = scores.get('snr', 0.0)
                # Use power exponent, ensure base is non-negative
                weight *= max(snr_score, 0.0) ** self.snr_exponent
            if self.weight_by_stars:
                stars_score = scores.get('stars', 0.0)
                weight *= max(stars_score, 0.0) ** self.stars_exponent

            # Apply minimum weight and ensure non-negative
            raw_weights[i] = max(weight, 1e-9) # Use a small epsilon instead of min_weight factor

        # --- Normalization Strategy ---
        # Normalize so weights sum to num_images. This makes the weighted average behave
        # similarly to a simple mean in terms of overall signal level.
        sum_weights = np.sum(raw_weights)
        if sum_weights > 1e-9:
            normalized_weights = raw_weights * (num_images / sum_weights)
        else:
            # If all weights are near zero, assign equal weight
            normalized_weights = np.ones(num_images, dtype=np.float32)

        # Apply absolute minimum weight threshold relative to the mean weight (which is 1 after normalization)
        normalized_weights = np.maximum(normalized_weights, self.min_weight)

        # Re-normalize after applying minimum weight floor (optional but safer)
        sum_weights_final = np.sum(normalized_weights)
        if sum_weights_final > 1e-9:
            normalized_weights = normalized_weights * (num_images / sum_weights_final)
        else:
             normalized_weights = np.ones(num_images, dtype=np.float32)

        # print(f"DEBUG - Weights: {normalized_weights}") # Debug print
        return normalized_weights


    # --- MODIFIED _process_file ---
    def _process_file(self, file_path, reference_image_data):
        """Prépare, aligne une image et calcule ses métriques qualité."""
        file_name = os.path.basename(file_path)
        quality_scores = {'snr': 0.0, 'stars': 0.0} # Default scores

        try:
            img_data = load_and_validate_fits(file_path)
            if img_data is None: self.update_progress(f"⚠️ Échec chargement/validation {file_name}."); self.skipped_files_count += 1; return None, None, quality_scores
            header = fits.getheader(file_path)
            if np.std(img_data) < 0.005: self.update_progress(f"⚠️ Image {file_name} ignorée (faible variance)."); self.skipped_files_count += 1; return None, None, quality_scores

            prepared_img = img_data
            if prepared_img.ndim == 2:
                bayer = header.get('BAYERPAT', self.bayer_pattern)
                if isinstance(bayer, str) and bayer.upper() in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                    try: prepared_img = debayer_image(prepared_img, bayer.upper())
                    except ValueError as de: self.update_progress(f"⚠️ Erreur debayer {file_name}: {de}. Tentative alignement N&B.")

            if self.correct_hot_pixels:
                try: prepared_img = detect_and_correct_hot_pixels(prepared_img, self.hot_pixel_threshold, self.neighborhood_size)
                except Exception as hp_err: self.update_progress(f"⚠️ Erreur correction px chauds {file_name}: {hp_err}. Continuation sans correction.")

            prepared_img = prepared_img.astype(np.float32)

            # --- Alignment ---
            aligned_img, align_success = self.aligner._align_image(prepared_img, reference_image_data, file_name)

            if not align_success:
                self.failed_align_count += 1
                try:
                    if os.path.exists(file_path): shutil.move(file_path, os.path.join(self.unaligned_folder, file_name)); self.update_progress(f"➡️ Échec alignement {file_name}. Déplacé.")
                    else: self.update_progress(f"⚠️ Original {file_name} non trouvé pour déplacement.")
                except Exception as move_err: self.update_progress(f"⚠️ Erreur déplacement {file_name}: {move_err}")
                return None, None, quality_scores # Return default scores on align fail

            # --- Calculate Quality Metrics *after* alignment ---
            # Note: Calculating on aligned data means metrics reflect potentially
            # interpolated pixels, but ensures metrics are comparable across frames
            # in the aligned coordinate system.
            if self.use_quality_weighting:
                quality_scores = self._calculate_quality_metrics(aligned_img)

            return aligned_img, header, quality_scores # Return scores

        except Exception as e:
            self.update_progress(f"❌ Erreur préparation/alignement fichier {file_name}: {e}")
            traceback.print_exc(limit=3); self.skipped_files_count += 1
            if os.path.exists(file_path):
                try: shutil.move(file_path, os.path.join(self.unaligned_folder, f"error_{file_name}"))
                except Exception: pass
            return None, None, quality_scores # Return default scores on other errors

    # --- MODIFIED _process_completed_batch ---
    def _process_completed_batch(self, current_batch_num, total_batches_est):
        """Traite un lot complet, incluant l'empilement (potentiellement pondéré)."""
        if not self.current_batch_data: return

        # Extract data, headers, and scores
        batch_images = [item[0] for item in self.current_batch_data]
        batch_headers = [item[1] for item in self.current_batch_data]
        batch_scores = [item[2] for item in self.current_batch_data] # Get scores
        batch_size = len(batch_images)

        progress_info = f"(Lot {current_batch_num}/{total_batches_est if total_batches_est > 0 else '?'})"
        self.update_progress(f"⚙️ Traitement du batch {progress_info} - {batch_size} images. Empilement...")

        # --- Stack the batch (now potentially weighted) ---
        stacked_batch_data, stack_info_header = self._stack_batch(
            batch_images, batch_headers, batch_scores, # Pass scores
            current_batch_num, total_batches_est
        )

        if stacked_batch_data is not None:
            # Combine result with cumulative stack (logic remains the same)
            self._combine_batch_result(stacked_batch_data, stack_info_header)
            self._update_preview()
            self._save_intermediate_stack()
        else:
            # Stacking failed for the batch
            self.failed_stack_count += batch_size # Count all images in failed batch
            self.update_progress(f"❌ Échec empilement lot {progress_info}. {batch_size} images ignorées.", None)

        # Clear batch data regardless of success
        self.current_batch_data = []
        gc.collect()


    # --- MODIFIED _stack_batch ---
    def _stack_batch(self, batch_images, batch_headers, batch_scores, # Added batch_scores
                     current_batch_num=0, total_batches_est=0):
        """
        Empile un lot d'images, en appliquant potentiellement la pondération par qualité.
        Utilise numpy pour le clipping et la moyenne pondérée.
        """
        if not batch_images: return None, None
        num_images = len(batch_images)
        progress_info = f"(Lot {current_batch_num}/{total_batches_est if total_batches_est > 0 else '?'})"
        stacking_method_info = f"'{self.stacking_mode}'"
        if self.use_quality_weighting: stacking_method_info += " + Pondération"
        self.update_progress(f"🧮 Empilement {progress_info} avec méthode {stacking_method_info} ({num_images} images)...")

        try:
            # --- Prepare data stack ---
            first_image = batch_images[0]
            common_shape = first_image.shape
            common_dtype = np.float32 # Work with float32
            image_stack_np = np.stack([img.astype(common_dtype) for img in batch_images], axis=0)

            # --- Calculate Weights (if enabled) ---
            weights_1d = None
            if self.use_quality_weighting:
                weights_1d = self._calculate_weights(batch_scores)
                # Safety check for weights length
                if len(weights_1d) != num_images:
                     self.update_progress("❌ Erreur interne: Nombre de poids != nombre d'images. Désactivation pondération pour ce lot.")
                     weights_1d = None # Fallback to no weighting
                else:
                    # Broadcast weights for pixel-wise operations
                    # Shape needs to match image_stack_np (N, H, W) or (N, H, W, C)
                    weight_shape = [-1] + [1] * (image_stack_np.ndim - 1)
                    weights_broadcast = weights_1d.reshape(weight_shape)

            stacked_image = None
            with np.errstate(divide='ignore', invalid='ignore'):

                # --- Apply Stacking Method ---
                if self.stacking_mode == "mean":
                    if weights_1d is not None:
                        stacked_image = np.average(image_stack_np, axis=0, weights=weights_1d)
                    else:
                        stacked_image = np.mean(image_stack_np, axis=0)

                elif self.stacking_mode == "median":
                    if weights_1d is not None:
                        # Weighted median is complex, fallback to simple median for now
                        self.update_progress(f"⚠️ Pondération non supportée pour 'median', utilisation mediane simple {progress_info}.")
                    stacked_image = np.median(image_stack_np, axis=0)

                elif self.stacking_mode in ["kappa-sigma", "winsorized-sigma"]:
                    # Calculate mean and std for clipping thresholds
                    # Note: Using unweighted mean/std for threshold calculation is common practice
                    mean = np.mean(image_stack_np, axis=0, dtype=np.float32)
                    std = np.std(image_stack_np, axis=0, dtype=np.float32)
                    std = np.maximum(std, 1e-6) # Avoid zero std deviation
                    threshold = self.kappa * std

                    if self.stacking_mode == "kappa-sigma":
                        # --- Kappa-Sigma Clipping (Potentially Weighted) ---
                        lower_bound = mean - threshold
                        upper_bound = mean + threshold
                        # Mask identifies pixels to *keep*
                        mask = (image_stack_np >= lower_bound) & (image_stack_np <= upper_bound)

                        if weights_1d is not None:
                            # Use weighted average on KEPT pixels
                            masked_weights = np.where(mask, weights_broadcast, 0.0)
                            sum_of_weights_kept = np.sum(masked_weights, axis=0)
                            sum_of_weights_kept = np.maximum(sum_of_weights_kept, 1e-9) # Avoid div by zero

                            weighted_sum_kept = np.sum(np.where(mask, image_stack_np * masked_weights, 0.0), axis=0)
                            stacked_image = weighted_sum_kept / sum_of_weights_kept
                        else:
                            # Use simple average on KEPT pixels (original kappa-sigma)
                            count_kept = np.maximum(np.sum(mask, axis=0, dtype=np.int16), 1)
                            sum_image = np.sum(np.where(mask, image_stack_np, 0.0), axis=0)
                            stacked_image = sum_image / count_kept

                        # Reporting rejection rate (optional)
                        rejected_count = num_images * np.prod(common_shape) - np.sum(mask)
                        rejected_percent = (rejected_count / (num_images * np.prod(common_shape))) * 100 if num_images > 0 else 0
                        self.update_progress(f"   (Kappa-Sigma {progress_info}: {rejected_percent:.2f}% pixels rejetés)")

                    elif self.stacking_mode == "winsorized-sigma":
                        # --- Winsorized Sigma (Potentially Weighted) ---
                        lower_bound = mean - threshold
                        upper_bound = mean + threshold
                        # Clip the data first
                        clipped_stack = np.clip(image_stack_np, lower_bound, upper_bound)

                        if weights_1d is not None:
                            # Weighted average of the CLIPPED data
                            stacked_image = np.average(clipped_stack, axis=0, weights=weights_1d)
                        else:
                            # Simple average of the CLIPPED data
                            stacked_image = np.mean(clipped_stack, axis=0)
                        self.update_progress(f"   (Winsorized-Sigma {progress_info} appliqué)")
                else:
                    # Fallback for unknown method
                    self.update_progress(f"⚠️ Méthode '{self.stacking_mode}' non reconnue, utilisation 'mean' {progress_info}.")
                    if weights_1d is not None: stacked_image = np.average(image_stack_np, axis=0, weights=weights_1d)
                    else: stacked_image = np.mean(image_stack_np, axis=0)

            # --- Finalize Stacked Image ---
            if stacked_image is None: raise ValueError("Stacking produced None result.")
            stacked_image = np.nan_to_num(stacked_image).astype(common_dtype) # Ensure correct dtype
            stacked_image = np.clip(stacked_image, 0.0, 1.0)

            # --- Create Header ---
            stack_info_header = fits.Header()
            stack_info_header['NIMAGES'] = (num_images, 'Images in this batch stack')
            # Calculate exposure for this batch
            batch_exposure = sum(float(h.get('EXPTIME', 0.0)) for h in batch_headers if h is not None)
            stack_info_header['TOTEXP'] = (round(batch_exposure, 2), '[s] Exposure time of this batch')

            del image_stack_np # Free memory
            gc.collect()
            return stacked_image, stack_info_header

        except Exception as e:
            self.update_progress(f"❌ Erreur empilement lot {progress_info}: {e}")
            traceback.print_exc(limit=3)
            gc.collect()
            return None, None


    # --- _combine_batch_result (Identique) ---
    def _combine_batch_result(self, stacked_batch_data, stack_info_header):
        # (Code identique à la version précédente)
        try:
            batch_n = int(stack_info_header.get('NIMAGES', 1)); batch_exposure = float(stack_info_header.get('TOTEXP', 0.0))
            if batch_n <= 0: self.update_progress("⚠️ Batch combiné avait 0 images, ignoré.", None); return
            if self.current_stack_data is None:
                self.current_stack_data = stacked_batch_data.copy(); self.images_in_cumulative_stack = batch_n; self.total_exposure_seconds = batch_exposure
                self.current_stack_header = fits.Header(); first_header = self.current_batch_data[0][1] if self.current_batch_data else fits.Header()
                keys_to_copy = ['INSTRUME', 'TELESCOP', 'OBJECT', 'FILTER', 'DATE-OBS', 'GAIN', 'OFFSET', 'CCD-TEMP', 'RA', 'DEC', 'SITELAT', 'SITELONG', 'FOCALLEN', 'BAYERPAT']
                for key in keys_to_copy:
                    if first_header and key in first_header: # Check first_header exists
                        try: self.current_stack_header[key] = (first_header[key], first_header.comments[key] if key in first_header.comments else '')
                        except Exception: self.current_stack_header[key] = first_header[key]
                self.current_stack_header['STACKTYP'] = (self.stacking_mode, 'Stacking method'); self.current_stack_header['NIMAGES'] = (self.images_in_cumulative_stack, 'Number of images in cumulative stack'); self.current_stack_header['TOTEXP'] = (round(self.total_exposure_seconds, 2), '[s] Total exposure time')
                if self.stacking_mode in ["kappa-sigma", "winsorized-sigma"]: self.current_stack_header['KAPPA'] = (self.kappa, 'Kappa value for clipping')
                # --- Add Weighting Info to Header ---
                if self.use_quality_weighting:
                    self.current_stack_header['WGHT_ON'] = (True, 'Quality weighting enabled')
                    w_metrics = []
                    if self.weight_by_snr: w_metrics.append(f"SNR^{self.snr_exponent:.1f}")
                    if self.weight_by_stars: w_metrics.append(f"Stars^{self.stars_exponent:.1f}")
                    self.current_stack_header['WGHT_MET'] = (",".join(w_metrics), 'Metrics used for weighting')
                else: self.current_stack_header['WGHT_ON'] = (False, 'Quality weighting disabled')
                # --- End Weighting Info ---
                self.current_stack_header['CREATOR'] = ('SeestarStacker (Queued)', 'Processing Software'); self.current_stack_header['HISTORY'] = 'Cumulative Stack Initialized'
                if self.correct_hot_pixels: self.current_stack_header['HISTORY'] = 'Hot pixel correction applied to input frames'
            else:
                if self.current_stack_data.shape != stacked_batch_data.shape: self.update_progress(f"❌ Incompatibilité dims stack: Cumul={self.current_stack_data.shape}, Batch={stacked_batch_data.shape}. Combinaison échouée."); return
                current_n = self.images_in_cumulative_stack; total_n = current_n + batch_n; w_old = current_n / total_n; w_new = batch_n / total_n
                combined_f32 = (self.current_stack_data.astype(np.float32) * w_old) + (stacked_batch_data.astype(np.float32) * w_new)
                self.current_stack_data = combined_f32.astype(np.float32)
                self.images_in_cumulative_stack = total_n; self.total_exposure_seconds += batch_exposure
                self.current_stack_header['NIMAGES'] = self.images_in_cumulative_stack; self.current_stack_header['TOTEXP'] = (round(self.total_exposure_seconds, 2), '[s] Total exposure time')
                self.current_stack_header.add_history(f'Combined with batch stack of {batch_n} images')
            self.current_stack_data = np.clip(self.current_stack_data, 0.0, 1.0)
        except Exception as e: self.update_progress(f"❌ Erreur combinaison du résultat du batch: {e}"); traceback.print_exc(limit=3)


    # --- _save_intermediate_stack (Identique à V4 - Correction History déjà OK) ---
    def _save_intermediate_stack(self):
        """ Sauvegarde le stack cumulatif intermédiaire. """
        if self.current_stack_data is None or self.output_folder is None: return
        stack_path = os.path.join(self.output_folder, "stack_cumulative.fit")
        preview_path = os.path.join(self.output_folder, "stack_cumulative.png")
        try:
            header_to_save = self.current_stack_header.copy() if self.current_stack_header else fits.Header()
            try: # Clean history safely
                if 'HISTORY' in header_to_save:
                    history_entries = list(header_to_save['HISTORY'])
                    filtered_history = [h for h in history_entries if 'Intermediate save' not in str(h)]
                    while 'HISTORY' in header_to_save: del header_to_save['HISTORY']
                    for entry in filtered_history: header_to_save.add_history(entry)
            except Exception: pass
            header_to_save.add_history(f'Intermediate save after combining {self.images_in_cumulative_stack} images')
            save_fits_image(self.current_stack_data, stack_path, header_to_save, overwrite=True)
            save_preview_image(self.current_stack_data, preview_path, apply_stretch=True)
        except Exception as e: print(f"⚠️ Erreur sauvegarde stack intermédiaire: {e}") # Shortened msg

    # --- _save_final_stack (Identique) ---
    def _save_final_stack(self):
        if self.current_stack_data is None or self.output_folder is None or self.images_in_cumulative_stack == 0: self.final_stacked_path = None; self.update_progress("ⓘ Aucun stack final à sauvegarder."); return
        self.final_stacked_path = os.path.join(self.output_folder, f"stack_final_{self.stacking_mode}{'_wght' if self.use_quality_weighting else ''}.fit"); # Add _wght to name
        preview_path = os.path.splitext(self.final_stacked_path)[0] + ".png"; self.update_progress(f"💾 Sauvegarde stack final: {os.path.basename(self.final_stacked_path)}...")
        try:
            final_header = self.current_stack_header.copy() if self.current_stack_header else fits.Header()
            try: # Safely remove intermediate history
                if 'HISTORY' in final_header:
                    history_entries = list(final_header['HISTORY'])
                    filtered_history = [h for h in history_entries if 'Intermediate save' not in str(h)]
                    while 'HISTORY' in final_header: del final_header['HISTORY']
                    for entry in filtered_history: final_header.add_history(entry)
            except Exception: pass
            final_header.add_history('Final Stack Saved by Seestar Stacker (Queued)')
            final_header['NIMAGES'] = (self.images_in_cumulative_stack, 'Number of images combined in final stack'); final_header['TOTEXP'] = (round(self.total_exposure_seconds, 2), '[s] Total exposure time')
            final_header['ALIGNED'] = (self.aligned_files_count, 'Successfully aligned images'); final_header['FAILALIGN'] = (self.failed_align_count, 'Failed alignments')
            final_header['FAILSTACK'] = (self.failed_stack_count, 'Files skipped due to batch stack errors'); final_header['SKIPPED'] = (self.skipped_files_count, 'Other skipped/error files')
            if 'STACKTYP' not in final_header: final_header['STACKTYP'] = self.stacking_mode
            if self.stacking_mode in ["kappa-sigma", "winsorized-sigma"] and 'KAPPA' not in final_header: final_header['KAPPA'] = self.kappa
            # Ensure weighting info is present if it was initialized
            if 'WGHT_ON' not in final_header: final_header['WGHT_ON'] = (self.use_quality_weighting, 'Quality weighting status')
            if self.use_quality_weighting and 'WGHT_MET' not in final_header:
                w_metrics = []
                if self.weight_by_snr: w_metrics.append(f"SNR^{self.snr_exponent:.1f}")
                if self.weight_by_stars: w_metrics.append(f"Stars^{self.stars_exponent:.1f}")
                final_header['WGHT_MET'] = (",".join(w_metrics), 'Metrics used for weighting')
            save_fits_image(self.current_stack_data, self.final_stacked_path, final_header, overwrite=True)
            save_preview_image(self.current_stack_data, preview_path, apply_stretch=True)
            self.update_progress(f"✅ Stack final sauvegardé ({self.images_in_cumulative_stack} images)")
        except Exception as e: self.update_progress(f"⚠️ Erreur sauvegarde stack final: {e}"); traceback.print_exc(limit=2); self.final_stacked_path = None


    # --- Cleanup Methods (Identiques) ---
    def cleanup_unaligned_files(self):
        if not self.unaligned_folder or not os.path.isdir(self.unaligned_folder): return
        deleted_count = 0
        try:
            for filename in os.listdir(self.unaligned_folder):
                file_path = os.path.join(self.unaligned_folder, filename)
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
                    try: os.remove(ref_fit); deleted_ref += 1
                    except Exception: pass
                if os.path.exists(ref_png):
                    try: os.remove(ref_png); deleted_ref += 1
                    except Exception: pass
                if deleted_ref > 0: self.update_progress(f"🧹 Fichier(s) référence temporaire(s) supprimé(s).")
                try: os.rmdir(aligner_temp_folder)
                except OSError: pass
        except Exception as e: self.update_progress(f"⚠️ Erreur nettoyage référence temp: {e}")

    # --- Folder & Queue Management (Identiques) ---
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
            self.additional_folders.append(abs_path)
            folder_count = len(self.additional_folders)
        self.update_progress(f"✅ Dossier ajouté à la file d'attente : {os.path.basename(folder_path)}")
        self.update_progress(f"folder_count_update:{folder_count}")
        return True

    def _add_files_to_queue(self, folder_path):
        count_added = 0
        try:
            abs_folder_path = os.path.abspath(folder_path)
            self.update_progress(f"🔍 Scan du dossier: {os.path.basename(folder_path)}...")
            files_in_folder = sorted(os.listdir(abs_folder_path))
            new_files_found_in_folder = []
            for fname in files_in_folder:
                if self.stop_processing: self.update_progress("⛔ Scan interrompu."); break
                if fname.lower().endswith(('.fit', '.fits')):
                    fpath = os.path.join(abs_folder_path, fname)
                    abs_fpath = os.path.abspath(fpath)
                    if abs_fpath not in self.processed_files:
                        self.queue.put(fpath)
                        self.processed_files.add(abs_fpath)
                        count_added += 1
            if count_added > 0:
                 self.files_in_queue += count_added
                 self._recalculate_total_batches() # Update total batch estimate
            return count_added
        except FileNotFoundError: self.update_progress(f"❌ Erreur scan: Dossier introuvable {os.path.basename(folder_path)}"); return 0
        except PermissionError: self.update_progress(f"❌ Erreur scan: Permission refusée {os.path.basename(folder_path)}"); return 0
        except Exception as e: self.update_progress(f"❌ Erreur scan dossier {os.path.basename(folder_path)}: {e}"); return 0


    # --- MODIFIED start_processing ---
    def start_processing(self, input_dir, output_dir, reference_path_ui=None, initial_additional_folders=None,
                         # *** NEW: Weighting parameters from GUI/Settings ***
                         use_weighting=False, weight_snr=True, weight_stars=True,
                         snr_exp=1.0, stars_exp=0.5, min_w=0.1):
        """Démarre le thread de traitement avec configuration de pondération."""
        if self.processing_active: self.update_progress("⚠️ Traitement déjà en cours."); return False
        self.stop_processing = False; self.current_folder = os.path.abspath(input_dir)
        if not self.initialize(output_dir): self.processing_active = False; return False

        # --- Batch Size Handling (Identique) ---
        if self.batch_size < 3:
             self.update_progress(f"⚠️ Taille de lot ({self.batch_size}) trop petite, ajustée à 3.", None)
             self.batch_size = 3
        self.update_progress(f"ⓘ Taille de lot utilisée : {self.batch_size}")

        # --- *** NEW: Apply Weighting Config *** ---
        self.use_quality_weighting = use_weighting
        self.weight_by_snr = weight_snr
        self.weight_by_stars = weight_stars
        self.snr_exponent = max(0.1, snr_exp) # Ensure positive exponents
        self.stars_exponent = max(0.1, stars_exp)
        self.min_weight = max(0.01, min(1.0, min_w)) # Ensure min_weight is reasonable
        # --- End Weighting Config ---

        # Initial Folders (Identique)
        initial_folders_to_add_count = 0
        with self.folders_lock:
            if initial_additional_folders:
                for folder in initial_additional_folders:
                     abs_folder = os.path.abspath(folder)
                     if os.path.isdir(abs_folder) and abs_folder not in self.additional_folders: self.additional_folders.append(abs_folder); initial_folders_to_add_count += 1
            if initial_folders_to_add_count > 0: self.update_progress(f"ⓘ {initial_folders_to_add_count} dossier(s) pré-ajouté(s) seront traités."); self.update_progress(f"folder_count_update:{len(self.additional_folders)}")

        initial_files_added = self._add_files_to_queue(self.current_folder) # Also calls _recalculate_total_batches

        if initial_files_added > 0: self.update_progress(f"📋 {initial_files_added} fichiers initiaux ajoutés. Total lots estimé: {self.total_batches_estimated}")
        elif not self.additional_folders:
             if reference_path_ui: self.update_progress("⚠️ Aucun fichier initial. Attente ajout dossiers via bouton...")
             else: self.update_progress("⚠️ Aucun fichier initial trouvé et aucun dossier supp. pré-ajouté. Démarrage quand même pour référence auto...")

        # --- Start Thread ---
        self.aligner.reference_image_path = reference_path_ui or None
        self.processing_thread = threading.Thread(target=self._worker, name="StackerWorker"); self.processing_thread.daemon = True; self.processing_thread.start(); self.processing_active = True
        return True

    # --- Control Methods (Identiques) ---
    def stop(self):
        if not self.processing_active: return
        self.update_progress("⛔ Arrêt demandé...")
        self.stop_processing = True
        self.aligner.stop_processing = True # Also signal aligner to stop

    def is_running(self):
        return self.processing_active and self.processing_thread is not None and self.processing_thread.is_alive()

# --- END OF FILE seestar/queuep/queue_manager.py ---