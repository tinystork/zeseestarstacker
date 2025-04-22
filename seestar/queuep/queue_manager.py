# --- START OF FILE seestar/queuep/queue_manager.py ---
"""
Module de gestion de file d'attente pour le traitement des images astronomiques.
Gère l'alignement et l'empilement incrémental par LOTS dans un thread séparé.
(Version Révisée 4: DÉFINITION _recalculate_total_batches AJOUTÉE)
"""
import os
import numpy as np
from astropy.io import fits
import cv2
import threading
from queue import Queue, Empty
import time
import astroalign as aa
import traceback
import shutil
import gc
import math # For ceil in batch calculation

# Import core processing functions needed within this module
from seestar.core.image_processing import (
    load_and_validate_fits, # Returns float32 0-1 or None
    debayer_image,          # Expects float32 0-1, returns float32 0-1
    save_fits_image,        # Expects float32 0-1, saves uint16
    save_preview_image      # Expects float32 0-1
)
from seestar.core.hot_pixels import detect_and_correct_hot_pixels
from seestar.core.alignment import SeestarAligner # Use the aligner class
# Import estimate_batch_size explicitly for QueueManager use
from seestar.core.utils import estimate_batch_size


class SeestarQueuedStacker:
    """
    Classe pour l'empilement des images Seestar avec file d'attente et traitement par lots.
    Gère l'alignement et l'empilement dans un thread séparé.
    """
    def __init__(self):
        # (Variables __init__ identiques à la version précédente)
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
        self.aligner = SeestarAligner()
        # --- Batch & Cumulative Stack ---
        self.current_batch_data = []
        self.current_stack_data = None
        self.current_stack_header = None
        self.images_in_cumulative_stack = 0
        self.total_exposure_seconds = 0.0
        self.final_stacked_path = None
        # --- Processing Parameters ---
        self.stacking_mode = "kappa-sigma"
        self.kappa = 2.5
        self.batch_size = 10 # Default, overwritten by GUI
        self.correct_hot_pixels = True
        self.hot_pixel_threshold = 3.0
        self.neighborhood_size = 5
        self.bayer_pattern = "GRBG"
        self.perform_cleanup = True
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

    # --- _update_preview CORRECTED to pass stack_count ---
    def _update_preview(self, force_update=False):
        """Safely calls the preview callback, including stack count and batch info."""
        if self.preview_callback is None or self.current_stack_data is None:
            return

        # Always update preview after combining a batch result for better feedback
        try:
            # Make copies to avoid race conditions with GUI thread using the data
            data_copy = self.current_stack_data.copy()
            header_copy = self.current_stack_header.copy() if self.current_stack_header else None

            # Gather all the info to send
            img_count = self.images_in_cumulative_stack
            total_imgs_est = self.files_in_queue # Use the dynamic total
            current_batch = self.stacked_batches_count # Batches processed so far
            total_batches_est = self.total_batches_estimated # Use the calculated estimate

            stack_name = f"Stack ({img_count}/{total_imgs_est} Img | Batch {current_batch}/{total_batches_est if total_batches_est > 0 else '?'})"

            # Call the callback with all the info
            self.preview_callback(
                data_copy,
                header_copy,
                stack_name,
                img_count,          
                total_imgs_est,     
                current_batch,      
                total_batches_est   
            )

        except Exception as e:
            print(f"Error in preview callback: {e}")
            traceback.print_exc(limit=2)

    def _recalculate_total_batches(self):
        """Estimates the total number of batches based on files_in_queue."""
        if self.batch_size > 0:
            # Use math.ceil to round up, ensuring the last partial batch is counted
            self.total_batches_estimated = math.ceil(self.files_in_queue / self.batch_size)
        else:
            # Handle invalid batch size (should be caught by settings validation, but defensive)
            self.update_progress(f"⚠️ Taille de lot invalide ({self.batch_size}), impossible d'estimer le nombre total de lots.")
            self.total_batches_estimated = 0 # Set to 0 or some indicator of error
    # --- END OF MISSING METHOD ---

    # --- Worker Thread Logic ---
    def _worker(self):
        # (Worker code is identical to Revision 3, but will now correctly find _recalculate_total_batches)
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

            # --- Initialize Batch Estimate (Now works) ---
            self._recalculate_total_batches()

            # --- Step 2: Process Queue (Identique) ---
            while not self.stop_processing:
                file_path = None
                try:
                    file_path = self.queue.get(timeout=1.0)
                    file_name = os.path.basename(file_path)
                    aligned_data, header = self._process_file(file_path, reference_image_data)
                    self.processed_files_count += 1
                    if aligned_data is not None:
                        self.current_batch_data.append((aligned_data, header))
                        self.aligned_files_count += 1
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
                        files_added = self._add_files_to_queue(folder_to_process) # Updates totals now
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
            # Finalization (Identique)
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


    # --- File Processing and Stacking Logic ---
    def _process_file(self, file_path, reference_image_data):
        # (Code identique à la version précédente)
        file_name = os.path.basename(file_path)
        try:
            img_data = load_and_validate_fits(file_path)
            if img_data is None: self.update_progress(f"⚠️ Échec chargement/validation {file_name}."); self.skipped_files_count += 1; return None, None
            header = fits.getheader(file_path)
            if np.std(img_data) < 0.005: self.update_progress(f"⚠️ Image {file_name} ignorée (faible variance)."); self.skipped_files_count += 1; return None, None
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
            aligned_img, align_success = self.aligner._align_image(prepared_img, reference_image_data, file_name)
            if not align_success:
                self.failed_align_count += 1
                try:
                    if os.path.exists(file_path): shutil.move(file_path, os.path.join(self.unaligned_folder, file_name)); self.update_progress(f"➡️ Échec alignement {file_name}. Déplacé.")
                    else: self.update_progress(f"⚠️ Original {file_name} non trouvé pour déplacement.")
                except Exception as move_err: self.update_progress(f"⚠️ Erreur déplacement {file_name}: {move_err}")
                return None, None
            return aligned_img, header
        except Exception as e:
            self.update_progress(f"❌ Erreur préparation/alignement fichier {file_name}: {e}")
            traceback.print_exc(limit=3); self.skipped_files_count += 1
            if os.path.exists(file_path):
                try: shutil.move(file_path, os.path.join(self.unaligned_folder, f"error_{file_name}"))
                except Exception: pass
            return None, None

    def _process_completed_batch(self, current_batch_num, total_batches_est):
        # (Code identique à la version précédente)
        if not self.current_batch_data: return
        batch_size = len(self.current_batch_data)
        progress_info = f"(Lot {current_batch_num}/{total_batches_est if total_batches_est > 0 else '?'})"
        self.update_progress(f"⚙️ Traitement du batch {progress_info} - {batch_size} images. Empilement...")
        batch_images = [item[0] for item in self.current_batch_data]; batch_headers = [item[1] for item in self.current_batch_data]
        stacked_batch_data, stack_info_header = self._stack_batch(batch_images, batch_headers, current_batch_num, total_batches_est)
        if stacked_batch_data is not None:
            self._combine_batch_result(stacked_batch_data, stack_info_header); self._update_preview(); self._save_intermediate_stack()
        else:
            self.failed_stack_count += batch_size; self.update_progress(f"❌ Échec empilement lot {progress_info}. {batch_size} images ignorées.", None)
        self.current_batch_data = []; gc.collect()

    def _stack_batch(self, batch_images, batch_headers, current_batch_num=0, total_batches_est=0):
        # (Code identique à la version précédente)
        if not batch_images: return None, None
        num_images = len(batch_images)
        progress_info = f"(Lot {current_batch_num}/{total_batches_est if total_batches_est > 0 else '?'})"
        self.update_progress(f"🧮 Empilement {progress_info} avec méthode '{self.stacking_mode}' ({num_images} images)...")
        try:
            first_image = batch_images[0]; common_shape = first_image.shape; common_dtype = np.float32
            for img in batch_images:
                if img.shape != common_shape: self.update_progress(f"❌ Erreur interne: Shape incompatible {progress_info}! Attendu {common_shape}, trouvé {img.shape}. Saut.", None); return None, None
            image_stack_np = np.stack([img.astype(np.float32) for img in batch_images], axis=0)
            stacked_image = None
            with np.errstate(divide='ignore', invalid='ignore'):
                if self.stacking_mode == "mean": stacked_image = np.mean(image_stack_np, axis=0, dtype=common_dtype)
                elif self.stacking_mode == "median": stacked_image = np.median(image_stack_np, axis=0).astype(common_dtype)
                elif self.stacking_mode in ["kappa-sigma", "winsorized-sigma"]:
                    mean = np.mean(image_stack_np, axis=0, dtype=np.float32); std = np.std(image_stack_np, axis=0, dtype=np.float32)
                    std = np.maximum(std, 1e-6); threshold = self.kappa * std
                    if self.stacking_mode == "kappa-sigma":
                        mask = np.abs(image_stack_np - mean) <= threshold; count_kept = np.maximum(np.sum(mask, axis=0, dtype=np.int16), 1)
                        sum_image = np.sum(np.where(mask, image_stack_np, 0), axis=0, dtype=np.float32); stacked_image = (sum_image / count_kept).astype(common_dtype)
                        rejected_count = num_images * np.prod(common_shape) - np.sum(mask)
                        rejected_percent = (rejected_count / (num_images * np.prod(common_shape))) * 100 if num_images > 0 else 0
                        self.update_progress(f"   (Kappa-Sigma {progress_info}: {rejected_percent:.2f}% pixels rejetés)")
                    elif self.stacking_mode == "winsorized-sigma":
                        lower_bound = mean - threshold; upper_bound = mean + threshold
                        clipped_stack = np.clip(image_stack_np, lower_bound, upper_bound); stacked_image = np.mean(clipped_stack, axis=0, dtype=common_dtype)
                        self.update_progress(f"   (Winsorized-Sigma {progress_info} appliqué)")
                else: self.update_progress(f"⚠️ Méthode '{self.stacking_mode}' non reconnue, utilisation 'mean' {progress_info}."); stacked_image = np.mean(image_stack_np, axis=0, dtype=common_dtype)
            if stacked_image is None: raise ValueError("Stacking produced None result.")
            stacked_image = np.nan_to_num(stacked_image); stacked_image = np.clip(stacked_image, 0.0, 1.0)
            stack_info_header = fits.Header(); stack_info_header['NIMAGES'] = (num_images, 'Images in this batch stack')
            batch_exposure = sum(float(h.get('EXPTIME', 0.0)) for h in batch_headers); stack_info_header['TOTEXP'] = (round(batch_exposure, 2), '[s] Exposure time of this batch')
            del image_stack_np; gc.collect()
            return stacked_image, stack_info_header
        except Exception as e:
            self.update_progress(f"❌ Erreur empilement lot {progress_info}: {e}")
            traceback.print_exc(limit=3); gc.collect(); return None, None

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
                    if key in first_header:
                        try: self.current_stack_header[key] = (first_header[key], first_header.comments[key] if key in first_header.comments else '')
                        except Exception: self.current_stack_header[key] = first_header[key]
                self.current_stack_header['STACKTYP'] = (self.stacking_mode, 'Stacking method'); self.current_stack_header['NIMAGES'] = (self.images_in_cumulative_stack, 'Number of images in cumulative stack'); self.current_stack_header['TOTEXP'] = (round(self.total_exposure_seconds, 2), '[s] Total exposure time')
                if self.stacking_mode in ["kappa-sigma", "winsorized-sigma"]: self.current_stack_header['KAPPA'] = (self.kappa, 'Kappa value for clipping')
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

    # --- MODIFIED _save_intermediate_stack ---
    def _save_intermediate_stack(self):
        """ Sauvegarde le stack cumulatif intermédiaire. """
        if self.current_stack_data is None or self.output_folder is None: return
        stack_path = os.path.join(self.output_folder, "stack_cumulative.fit")
        preview_path = os.path.join(self.output_folder, "stack_cumulative.png")
        try:
            header_to_save = self.current_stack_header.copy() if self.current_stack_header else fits.Header()
            # Safely clean up history
            try:
                if 'HISTORY' in header_to_save: # Check if HISTORY exists
                    # Convert to list to allow modification while iterating (or create new list)
                    history_entries = list(header_to_save['HISTORY'])
                    # Filter out old intermediate save messages
                    filtered_history = [h for h in history_entries if 'Intermediate save' not in str(h)]
                    # Remove all existing HISTORY cards cleanly
                    while 'HISTORY' in header_to_save:
                        del header_to_save['HISTORY']
                    # Add back the filtered history
                    for entry in filtered_history:
                        header_to_save.add_history(entry)
            except KeyError:
                # This handles the case where HISTORY doesn't exist at all
                pass
            except Exception as hist_e:
                print(f"Debug: Minor error cleaning history: {hist_e}") # Log other history errors

            # Add new history entry
            header_to_save.add_history(f'Intermediate save after combining {self.images_in_cumulative_stack} images')

            save_fits_image(self.current_stack_data, stack_path, header_to_save, overwrite=True)
            save_preview_image(self.current_stack_data, preview_path, apply_stretch=True)
        except Exception as e:
            # Log general save errors
            print(f"⚠️ Erreur sauvegarde stack intermédiaire (Général): {e}")

    # --- _save_final_stack (Identique) ---
    def _save_final_stack(self):
        if self.current_stack_data is None or self.output_folder is None or self.images_in_cumulative_stack == 0: self.final_stacked_path = None; self.update_progress("ⓘ Aucun stack final à sauvegarder."); return
        self.final_stacked_path = os.path.join(self.output_folder, f"stack_final_{self.stacking_mode}.fit"); preview_path = os.path.splitext(self.final_stacked_path)[0] + ".png"; self.update_progress(f"💾 Sauvegarde stack final: {os.path.basename(self.final_stacked_path)}...")
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

    # --- Folder & Queue Management ---
    def add_folder(self, folder_path):
        # (Code identique à la version précédente)
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
        # (Code _add_files_to_queue identique à la version précédente)
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
                 self._recalculate_total_batches()
            return count_added
        except FileNotFoundError: self.update_progress(f"❌ Erreur scan: Dossier introuvable {os.path.basename(folder_path)}"); return 0
        except PermissionError: self.update_progress(f"❌ Erreur scan: Permission refusée {os.path.basename(folder_path)}"); return 0
        except Exception as e: self.update_progress(f"❌ Erreur scan dossier {os.path.basename(folder_path)}: {e}"); return 0


    # --- Control Methods ---
    def start_processing(self, input_dir, output_dir, reference_path_ui=None, initial_additional_folders=None):
        """Démarre le thread de traitement."""
        if self.processing_active: self.update_progress("⚠️ Traitement déjà en cours."); return False
        self.stop_processing = False; self.current_folder = os.path.abspath(input_dir)
        if not self.initialize(output_dir): self.processing_active = False; return False

        # --- Batch Size Handling ---
        # Set instance batch_size first (it comes from GUI settings)
        # Then check if it's valid (>=3)
        # self.batch_size = batch_size_from_gui # This is already done via property assignment
        if self.batch_size < 3:
             self.update_progress(f"⚠️ Taille de lot ({self.batch_size}) trop petite, ajustée à 3.", None)
             self.batch_size = 3
        self.update_progress(f"ⓘ Taille de lot utilisée : {self.batch_size}") # Confirm batch size
        # --- End Batch Size Handling ---

        initial_folders_to_add_count = 0
        with self.folders_lock:
            if initial_additional_folders:
                for folder in initial_additional_folders:
                     abs_folder = os.path.abspath(folder)
                     if os.path.isdir(abs_folder) and abs_folder not in self.additional_folders: self.additional_folders.append(abs_folder); initial_folders_to_add_count += 1
            if initial_folders_to_add_count > 0: self.update_progress(f"ⓘ {initial_folders_to_add_count} dossier(s) pré-ajouté(s) seront traités."); self.update_progress(f"folder_count_update:{len(self.additional_folders)}")

        # Scan initial folder *after* setting batch size
        initial_files_added = self._add_files_to_queue(self.current_folder) # This calls _recalculate_total_batches internally now

        if initial_files_added > 0: self.update_progress(f"📋 {initial_files_added} fichiers initiaux ajoutés. Total lots estimé: {self.total_batches_estimated}")
        elif not self.additional_folders:
             if reference_path_ui: self.update_progress("⚠️ Aucun fichier initial. Attente ajout dossiers via bouton...")
             else: self.update_progress("⚠️ Aucun fichier initial trouvé et aucun dossier supp. pré-ajouté. Démarrage quand même pour référence auto...")

        self.aligner.reference_image_path = reference_path_ui or None
        self.processing_thread = threading.Thread(target=self._worker, name="StackerWorker"); self.processing_thread.daemon = True; self.processing_thread.start(); self.processing_active = True
        return True

    def stop(self):
        # (Code identique à la version précédente)
        if not self.processing_active: return
        self.update_progress("⛔ Arrêt demandé...")
        self.stop_processing = True
        self.aligner.stop_processing = True

    def is_running(self):
        # (Code identique à la version précédente)
        return self.processing_active and self.processing_thread is not None and self.processing_thread.is_alive()

# --- END OF FILE seestar/queuep/queue_manager.py ---