"""
Module de gestion de file d'attente pour le traitement des images astronomiques.
Version optimisée combinant l'efficacité de Seestar avec les améliorations de GSeestar.
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
import shutil # Added for moving unaligned files

# Import core processing functions needed within this module
from seestar.core.image_processing import (
    load_and_validate_fits,
    debayer_image,
    save_fits_image,
    save_preview_image
)
from seestar.core.hot_pixels import detect_and_correct_hot_pixels
from seestar.core.utils import apply_denoise
# estimate_batch_size is not directly used here, but by the GUI before starting

class SeestarQueuedStacker:
    """
    Classe pour l'empilement des images Seestar avec file d'attente.
    Optimisée pour minimiser l'utilisation d'espace disque et permettre l'ajout
    de dossiers pendant le traitement. Empile les images de manière cumulative.
    """
    def __init__(self):
        """Initialise le stacker avec des valeurs par défaut."""
        # State Flags & Control
        self.stop_processing = False
        self.processing_active = False
        self.processing_error = None

        # Callbacks
        self.progress_callback = None
        self.preview_callback = None

        # Queue & Threading
        self.queue = Queue()
        self.processing_thread = None
        self.folders_lock = threading.Lock()

        # File & Folder Management
        self.processed_files = set()
        self.additional_folders = []
        self.current_folder = None
        self.output_folder = None
        self.temp_folder = None
        self.unaligned_folder = None

        # Reference Image
        self.reference_image = None
        self.reference_header = None
        self.reference_path_internal = None

        # Cumulative Stack
        self.current_stack_data = None
        self.current_stack_header = None
        self.current_stack_count = 0
        self.total_exposure_seconds = 0.0 # <-- ADDED Total exposure time
        self.final_stacked_path = None

        # Processing Parameters
        self.stacking_mode = "kappa-sigma"
        self.kappa = 2.5
        self.batch_size = 10
        self.denoise = False
        self.correct_hot_pixels = True
        self.hot_pixel_threshold = 3.0
        self.neighborhood_size = 5
        self.bayer_pattern = "GRBG"

        # Statistics
        self.files_in_queue = 0
        self.processed_files_count = 0
        self.aligned_files_count = 0
        self.failed_align_count = 0
        self.skipped_files_count = 0

    def initialize(self, output_dir):
        """Initialise le stacker et prépare les dossiers de sortie/temporaires."""
        self.output_folder = output_dir
        self.temp_folder = os.path.join(self.output_folder, "temp_processing")
        self.unaligned_folder = os.path.join(self.output_folder, "unaligned_files")

        try:
            os.makedirs(self.output_folder, exist_ok=True)
            os.makedirs(self.temp_folder, exist_ok=True)
            os.makedirs(self.unaligned_folder, exist_ok=True)
        except OSError as e:
            self.update_progress(f"❌ Erreur lors de la création des dossiers: {e}", 0)
            return False

        # Reset state
        self.processed_files.clear()
        self.additional_folders = []
        self.current_stack_data = None
        self.current_stack_header = None
        self.current_stack_count = 0
        self.total_exposure_seconds = 0.0 # <-- ADDED Reset exposure time
        self.reference_image = None
        self.reference_header = None
        self.reference_path_internal = None
        self.final_stacked_path = None
        self.processing_error = None

        # Reset counters
        self.files_in_queue = 0
        self.processed_files_count = 0
        self.aligned_files_count = 0
        self.failed_align_count = 0
        self.skipped_files_count = 0

        self.cleanup_temp_files(clear_reference=True)
        self.update_progress(f"🗄️ Système initialisé. Dossier temporaire: {self.temp_folder}")
        return True

    # --- Callback Methods ---
    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def update_progress(self, message, progress=None):
        if self.progress_callback:
             try:
                 self.progress_callback(message, progress)
             except Exception as e:
                 print(f"Error in progress callback: {e}")
        else:
            print(message)

    def set_preview_callback(self, callback):
        self.preview_callback = callback

    def _update_preview(self, force_update=False):
        if self.preview_callback is None or self.current_stack_data is None:
            return
        update_frequency = 5
        if force_update or self.current_stack_count % update_frequency == 0:
            try:
                data_copy = self.current_stack_data.copy()
                header_copy = self.current_stack_header.copy() if self.current_stack_header else None
                stack_name = f"Stack ({self.current_stack_count} images)"
                self.preview_callback(data_copy, header_copy, stack_name)
            except Exception as e:
                 print(f"Error in preview callback: {e}")

    # --- Worker Thread and Processing Logic ---
    def _worker(self):
        self.processing_active = True
        self.processing_error = None
        total_files_processed_session = 0
        start_time_session = time.monotonic()

        try:
            # Reference Image Handling
            if self.reference_image is None:
                self.update_progress("⭐ Recherche de l'image de référence...")
                if not self._find_and_prepare_reference(self.current_folder):
                    raise RuntimeError("Impossible d'obtenir une image de référence valide.")
            else:
                 self._save_reference_image()
                 self.update_progress("⭐ Utilisation de l'image de référence fournie.")

            # Main Processing Loop
            while not self.stop_processing:
                file_path = None
                try:
                    file_path = self.queue.get(timeout=1.0)
                    file_name = os.path.basename(file_path)

                    if file_path in self.processed_files:
                        # self.update_progress(f"⏩ Fichier déjà traité, ignoré: {file_name}") # Reduce verbosity
                        self.skipped_files_count += 1
                        self.queue.task_done()
                        continue

                    process_success = self._process_file(file_path)

                    if process_success:
                        self.processed_files.add(file_path)
                    else:
                         self.skipped_files_count += 1

                    self.queue.task_done()
                    total_files_processed_session += 1

                    # Progress Update
                    current_progress = (self.processed_files_count / self.files_in_queue) * 100 if self.files_in_queue > 0 else 0
                    elapsed_time = time.monotonic() - start_time_session
                    if self.processed_files_count > 0:
                        time_per_file = elapsed_time / self.processed_files_count
                        remaining_files_estimate = self.files_in_queue - self.processed_files_count
                        eta_seconds = remaining_files_estimate * time_per_file
                        h, rem = divmod(int(eta_seconds), 3600); m, s = divmod(rem, 60)
                        time_str = f"{h:02}:{m:02}:{s:02}"
                        progress_msg = (f"📊 ({self.processed_files_count}/{self.files_in_queue}) {file_name} | ETA: {time_str}")
                    else:
                        progress_msg = (f"📊 ({self.processed_files_count}/{self.files_in_queue}) {file_name}")
                    self.update_progress(progress_msg, current_progress)

                except Empty:
                    # Check for additional folders
                    with self.folders_lock:
                        if self.additional_folders:
                            next_folder = self.additional_folders.pop(0)
                            self.update_progress(f"📂 Traitement dossier supplémentaire: {os.path.basename(next_folder)}")
                            self.current_folder = next_folder
                            files_added = self._add_files_to_queue(next_folder)
                            if files_added > 0:
                                self.files_in_queue += files_added
                                self.update_progress(f"📋 {files_added} fichiers ajoutés depuis {os.path.basename(next_folder)}")
                                continue
                            else:
                                 self.update_progress(f"⚠️ Aucun nouveau fichier FITS trouvé dans {os.path.basename(next_folder)}")
                                 continue
                        else:
                            self.update_progress("✅ Tous les fichiers et dossiers traités.")
                            break # Exit loop

                except Exception as e:
                     error_context = f" de {file_name}" if file_path else ""
                     self.update_progress(f"❌ Erreur inattendue worker{error_context}: {e}")
                     self.update_progress(traceback.format_exc(limit=3))
                     self.processing_error = str(e)
                     if file_path: self.queue.task_done()
                     self.skipped_files_count += 1

            # End of Processing Loop
            if self.stop_processing:
                 self.update_progress("⛔ Traitement arrêté par l'utilisateur.")
            else:
                 if self.current_stack_data is not None:
                      self.update_progress("🏁 Finalisation...")
                      if self.denoise: self._apply_denoise_to_final()
                      self._save_final_stack()
                      self._update_preview(force_update=True)
                      self.update_progress(f"🏁 Traitement terminé. {self.processed_files_count} images empilées.")
                 else:
                      self.update_progress("🏁 Traitement terminé, mais aucun stack créé.")

        except Exception as e:
            self.update_progress(f"❌ Erreur critique thread traitement: {e}")
            self.update_progress(traceback.format_exc(limit=5))
            self.processing_error = str(e)

        finally:
            self.update_progress("🧹 Nettoyage...")
            self.cleanup_temp_files(clear_reference=False)
            self.processing_active = False
            self.update_progress("🚪 Thread traitement terminé.")

    def _find_and_prepare_reference(self, folder_path, max_samples=50):
        """ Trouve et prépare l'image de référence. """
        try:
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.fit', '.fits'))]
            if not files: return False
            sample_size = min(max_samples, len(files))
            self.update_progress(f"🔍 Analyse de ~{sample_size} images pour référence...")
            best_image, best_header, best_file, max_metric = None, None, None, -1

            for f in files[:sample_size]:
                if self.stop_processing: return False
                file_path = os.path.join(folder_path, f)
                try:
                    img_data = load_and_validate_fits(file_path)
                    header = fits.getheader(file_path)
                    if np.std(img_data) < 5: continue

                    candidate_img = img_data
                    if candidate_img.ndim == 2:
                        bayer = header.get('BAYERPAT', self.bayer_pattern)
                        if isinstance(bayer, str) and bayer.upper() in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                             candidate_img = debayer_image(candidate_img, bayer.upper())
                    elif candidate_img.ndim == 3 and candidate_img.shape[0] == 3:
                         candidate_img = np.moveaxis(candidate_img, 0, -1)

                    if self.correct_hot_pixels:
                         candidate_img = detect_and_correct_hot_pixels(candidate_img, self.hot_pixel_threshold, self.neighborhood_size)
                    candidate_img = cv2.normalize(candidate_img, None, 0, 65535, cv2.NORM_MINMAX)

                    mean_val = np.mean(candidate_img)
                    std_val = np.std(candidate_img)
                    metric = (mean_val / std_val) if std_val > 1e-6 else 0
                    if metric > max_metric:
                        max_metric, best_image, best_header, best_file = metric, candidate_img, header, f
                except Exception as e: self.update_progress(f"⚠️ Erreur analyse réf {f}: {e}")

            if best_image is not None:
                self.reference_image, self.reference_header = best_image, best_header
                self._save_reference_image()
                self.update_progress(f"⭐ Référence: {best_file} (Metric: {max_metric:.2f})")
                return True
            else:
                self.update_progress("❌ Aucune référence valide trouvée.")
                return False
        except Exception as e:
            self.update_progress(f"❌ Erreur recherche référence: {e}"); traceback.print_exc(limit=3); return False

    def _save_reference_image(self):
        """ Sauvegarde l'image de référence préparée dans le dossier temporaire. """
        if self.reference_image is None or self.temp_folder is None: return
        self.reference_path_internal = os.path.join(self.temp_folder, "reference_image.fit")
        preview_path = os.path.join(self.temp_folder, "reference_image.png")
        try:
            save_fits_image(self.reference_image, self.reference_path_internal, self.reference_header)
            save_preview_image(self.reference_image, preview_path, stretch=True)
            # self.update_progress(f"📁 Image référence sauvegardée: {self.reference_path_internal}") # Reduce verbosity
        except Exception as e:
            self.update_progress(f"⚠️ Erreur sauvegarde référence: {e}"); self.reference_path_internal = None

    def _process_file(self, file_path):
        """ Charge, prépare, aligne et empile une seule image. """
        file_name = os.path.basename(file_path)
        try:
            img_data = load_and_validate_fits(file_path)
            header = fits.getheader(file_path)

            prepared_img = img_data
            if prepared_img.ndim == 2:
                 bayer = header.get('BAYERPAT', self.bayer_pattern)
                 if isinstance(bayer, str) and bayer.upper() in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                     prepared_img = debayer_image(prepared_img, bayer.upper())
            elif prepared_img.ndim == 3 and prepared_img.shape[0] == 3:
                 prepared_img = np.moveaxis(prepared_img, 0, -1)

            if self.correct_hot_pixels:
                 prepared_img = detect_and_correct_hot_pixels(prepared_img, self.hot_pixel_threshold, self.neighborhood_size)
            prepared_img = cv2.normalize(prepared_img, None, 0, 65535, cv2.NORM_MINMAX)

            aligned_img, align_success = self._align_image(prepared_img, file_name)

            if not align_success:
                 self.failed_align_count += 1
                 try:
                     shutil.move(file_path, os.path.join(self.unaligned_folder, file_name))
                     # self.update_progress(f"➡️ Fichier non aligné déplacé: {file_name}") # Reduce verbosity
                 except Exception as move_err: self.update_progress(f"⚠️ Erreur déplacement {file_name}: {move_err}")
                 return False

            self.aligned_files_count += 1
            stack_success = self._stack_image_incrementally(aligned_img, header) # Pass header here

            if stack_success:
                 self.processed_files_count += 1
                 return True
            else: return False

        except (FileNotFoundError, ValueError, OSError) as load_err:
             self.update_progress(f"❌ Erreur chargement/validation {file_name}: {load_err}")
             return False
        except Exception as e:
            self.update_progress(f"❌ Erreur traitement {file_name}: {e}"); traceback.print_exc(limit=3)
            return False

    def _align_image(self, img_to_align, file_name):
        """ Aligne l'image préparée sur l'image de référence. """
        if self.reference_image is None: return img_to_align, False
        ref_shape, img_shape = self.reference_image.shape, img_to_align.shape
        if ref_shape != img_shape:
            if ref_shape[:2] == img_shape[:2]:
                 if ref_shape.ndim == 3 and img_shape.ndim == 2:
                      ref_align = cv2.cvtColor(self.reference_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                      img_align = img_to_align
                 elif ref_shape.ndim == 2 and img_shape.ndim == 3:
                      ref_align = self.reference_image
                      img_align = cv2.cvtColor(img_to_align.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                 else: self.update_progress(f"❌ Alignement: Dims incompatibles Réf={ref_shape}, Img={img_shape}"); return img_to_align, False
            else: self.update_progress(f"❌ Alignement: H,W incompatibles Réf={ref_shape[:2]}, Img={img_shape[:2]}"); return img_to_align, False
        else: ref_align, img_align = self.reference_image, img_to_align
        try:
            img_norm = cv2.normalize(img_align, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            ref_norm = cv2.normalize(ref_align, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            aligned_img_norm, _ = aa.register(source=img_norm, target=ref_norm, max_control_points=50)
            if aligned_img_norm is None: raise aa.MaxIterError("Alignement: Pas de transformation")
            aligned_img_norm = np.clip(aligned_img_norm, 0, 1.0)
            aligned_img_final = (aligned_img_norm * 65535.0).astype(np.float32)
            return aligned_img_final, True
        except aa.MaxIterError as ae: self.update_progress(f"⚠️ Alignement échoué {file_name}: {ae}"); return img_to_align, False
        except Exception as e: self.update_progress(f"❌ Erreur alignement {file_name}: {e}"); traceback.print_exc(limit=3); return img_to_align, False

    def _stack_image_incrementally(self, aligned_img, header):
        """ Empile l'image alignée avec le stack cumulatif. """
        try:
            n = self.current_stack_count
            target_dtype = np.float32
            img_float = aligned_img.astype(target_dtype) # Ensure input is float

            # --- Get Exposure Time ---
            current_exposure = 0.0
            try:
                 current_exposure = float(header.get('EXPTIME', 0.0))
            except (ValueError, TypeError):
                 self.update_progress(f"⚠️ Impossible de lire EXPTIME depuis l'en-tête de {header.get('FILENAME', 'image')}")

            # --- First Image ---
            if self.current_stack_data is None:
                self.current_stack_data = img_float
                self.current_stack_count = 1
                self.total_exposure_seconds = current_exposure # <-- Initialize total exposure
                self.current_stack_header = fits.Header()
                for key in ['INSTRUME', 'EXPTIME', 'FILTER', 'OBJECT', 'DATE-OBS', 'TELESCOP', 'GAIN', 'OFFSET', 'CCD-TEMP']:
                     if key in header: self.current_stack_header[key] = header[key]
                self.current_stack_header['STACKTYP'] = (self.stacking_mode, 'Stacking method')
                self.current_stack_header['NIMAGES'] = (1, 'Number of images in stack')
                self.current_stack_header['TOTEXP'] = (self.total_exposure_seconds, '[s] Total exposure time') # <-- Add total exposure
                if self.stacking_mode in ["kappa-sigma", "winsorized-sigma"]:
                    self.current_stack_header['KAPPA'] = (self.kappa, 'Kappa value for clipping')

            # --- Combine with Existing Stack ---
            else:
                if self.current_stack_data.shape != img_float.shape:
                    self.update_progress(f"⚠️ Incompatibilité dims stack: Stack={self.current_stack_data.shape}, Img={img_float.shape}")
                    return False

                # Apply Stacking Method
                if self.stacking_mode == "mean":
                    self.current_stack_data = (n * self.current_stack_data + img_float) / (n + 1)
                elif self.stacking_mode == "median":
                    w_old = n / (n + 1.0); w_new = 1.0 / (n + 1.0)
                    self.current_stack_data = self.current_stack_data * w_old + img_float * w_new
                elif self.stacking_mode in ["kappa-sigma", "winsorized-sigma"]:
                    current_mean = self.current_stack_data
                    with np.errstate(invalid='ignore'):
                         current_std = np.std(self.current_stack_data)
                         current_std = np.maximum(current_std, 1e-5)
                    threshold = self.kappa * current_std
                    deviation = np.abs(img_float - current_mean)
                    if self.stacking_mode == "kappa-sigma":
                         mask = deviation <= threshold
                         denominator = n + mask.astype(target_dtype)
                         denominator = np.maximum(denominator, 1e-9)
                         self.current_stack_data = (n * self.current_stack_data + img_float * mask) / denominator
                    elif self.stacking_mode == "winsorized-sigma":
                         lower, upper = current_mean - threshold, current_mean + threshold
                         clipped_img = np.clip(img_float, lower, upper)
                         self.current_stack_data = (n * self.current_stack_data + clipped_img) / (n + 1)
                else: # Fallback mean
                    self.current_stack_data = (n * self.current_stack_data + img_float) / (n + 1)

                # Update count and header
                self.current_stack_count += 1
                self.total_exposure_seconds += current_exposure # <-- Accumulate exposure
                self.current_stack_header['NIMAGES'] = self.current_stack_count
                self.current_stack_header['TOTEXP'] = (self.total_exposure_seconds, '[s] Total exposure time') # <-- Update total exposure

            # Post-Stacking Steps
            self.current_stack_data = cv2.normalize(self.current_stack_data, None, 0, 65535, cv2.NORM_MINMAX)
            if self.current_stack_count % 10 == 0: self._save_intermediate_stack()
            self._update_preview()
            return True

        except Exception as e:
            self.update_progress(f"❌ Erreur empilement incrémental: {e}"); traceback.print_exc(limit=3)
            return False

    def _save_intermediate_stack(self):
        """ Sauvegarde le stack cumulatif actuel. """
        if self.current_stack_data is None or self.output_folder is None: return
        stack_path = os.path.join(self.output_folder, "stack_cumulative.fit")
        preview_path = os.path.join(self.output_folder, "stack_cumulative.png")
        try:
            save_fits_image(self.current_stack_data, stack_path, self.current_stack_header)
            save_preview_image(self.current_stack_data, preview_path, stretch=True)
        except Exception as e: self.update_progress(f"⚠️ Erreur sauvegarde stack intermédiaire: {e}")

    def _apply_denoise_to_final(self):
        """ Applique le débruitage au stack final. """
        if self.current_stack_data is None: return
        self.update_progress("✨ Application débruitage final...")
        try:
            denoised_data = apply_denoise(self.current_stack_data, strength=5)
            self.current_stack_data = denoised_data
            if self.current_stack_header:
                self.current_stack_header.add_history('Denoised applied')
                self.current_stack_header['DENOISED'] = (True, 'Denoising applied post-stack')
            self.update_progress("✅ Débruitage appliqué.")
        except Exception as e: self.update_progress(f"⚠️ Échec débruitage: {e}")

    def _save_final_stack(self):
        """ Sauvegarde le stack final. """
        if self.current_stack_data is None or self.output_folder is None: return
        self.final_stacked_path = os.path.join(self.output_folder, "stack_final.fit")
        preview_path = os.path.join(self.output_folder, "stack_final.png")
        self.update_progress(f"💾 Sauvegarde stack final: {self.final_stacked_path}...")
        try:
            if self.current_stack_header:
                 self.current_stack_header.add_history('Final Stack Saved')
                 self.current_stack_header['CREATOR'] = ('Seestar Stacker (Queued)', 'Processing Software')
                 self.current_stack_header['ALIGNED'] = (self.aligned_files_count, 'Successfully aligned images')
                 self.current_stack_header['FAILED'] = (self.failed_align_count, 'Failed alignments')
                 self.current_stack_header['SKIPPED'] = (self.skipped_files_count, 'Skipped/error files')
                 # TOTEXP is already updated incrementally

            save_fits_image(self.current_stack_data, self.final_stacked_path, self.current_stack_header)
            save_preview_image(self.current_stack_data, preview_path, stretch=True)
            self.update_progress(f"✅ Stack final sauvegardé ({self.current_stack_count} images)")

            if self.current_stack_header and self.current_stack_header.get('DENOISED'):
                 denoised_path = os.path.join(self.output_folder, "stack_final_denoised.fit")
                 denoised_preview = os.path.join(self.output_folder, "stack_final_denoised.png")
                 try:
                      shutil.copy2(self.final_stacked_path, denoised_path); shutil.copy2(preview_path, denoised_preview)
                      self.update_progress(f"✅ Version débruitée copiée: {denoised_path}")
                 except Exception as copy_e: self.update_progress(f"⚠️ Erreur copie version débruitée: {copy_e}")
        except Exception as e:
            self.update_progress(f"⚠️ Erreur sauvegarde stack final: {e}"); self.final_stacked_path = None

    # --- File and Folder Management ---
    def cleanup_temp_files(self, clear_reference=False):
        if not self.temp_folder or not os.path.isdir(self.temp_folder): return
        deleted_count = 0
        try:
            for filename in os.listdir(self.temp_folder):
                file_path = os.path.join(self.temp_folder, filename)
                keep = not clear_reference and filename in ["reference_image.fit", "reference_image.png"]
                if not keep and os.path.isfile(file_path):
                    try: os.remove(file_path); deleted_count += 1
                    except Exception as del_e: self.update_progress(f"⚠️ Erreur suppression temp {filename}: {del_e}")
        except Exception as e: self.update_progress(f"⚠️ Erreur nettoyage temp: {e}")

    def add_folder(self, folder_path):
        if not os.path.isdir(folder_path): return False
        if self.output_folder and os.path.abspath(folder_path).startswith(os.path.abspath(self.output_folder)): return False
        with self.folders_lock:
            abs_path = os.path.abspath(folder_path)
            if abs_path in [os.path.abspath(p) for p in self.additional_folders] or abs_path == os.path.abspath(self.current_folder or ""): return False
            try:
                if not any(f.lower().endswith(('.fit', '.fits')) for f in os.listdir(folder_path)): return False
                self.additional_folders.append(folder_path)
                self.update_progress(f"✅ Dossier ajouté: {os.path.basename(folder_path)}")
                return True
            except Exception as e: self.update_progress(f"❌ Erreur lecture dossier {os.path.basename(folder_path)}: {e}"); return False

    def _add_files_to_queue(self, folder_path):
        count_added = 0
        try:
            abs_folder_path = os.path.abspath(folder_path)
            for fname in os.listdir(abs_folder_path):
                if fname.lower().endswith(('.fit', '.fits')):
                    fpath = os.path.join(abs_folder_path, fname)
                    if fpath not in self.processed_files:
                         self.queue.put(fpath); count_added += 1
            return count_added
        except Exception as e: self.update_progress(f"❌ Erreur ajout fichiers depuis {os.path.basename(folder_path)}: {e}"); return 0

    # --- Control Methods ---
    def start_processing(self, input_dir, output_dir, reference_path_ui=None):
        self.stop_processing = False; self.current_folder = input_dir
        if not self.initialize(output_dir): return False

        self.reference_image = None; self.reference_header = None
        if reference_path_ui and os.path.isfile(reference_path_ui):
             self.update_progress(f"⚙️ Préparation référence fournie: {os.path.basename(reference_path_ui)}")
             if not self._prepare_external_reference(reference_path_ui):
                  self.update_progress("⚠️ Échec préparation réf fournie. Recherche auto...")
                  self.reference_image = None # Ensure auto-find triggers

        files_added = self._add_files_to_queue(input_dir)
        if files_added == 0 and self.reference_image is None:
             self.update_progress("❌ Aucun fichier initial et pas de référence fournie.")
             return False
        elif files_added == 0:
             self.update_progress("⚠️ Aucun fichier initial, mais référence prête. Attente dossiers ajoutés.")

        self.files_in_queue = files_added
        if files_added > 0: self.update_progress(f"📋 {files_added} fichiers initiaux ajoutés.")

        self.processing_thread = threading.Thread(target=self._worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.processing_active = True
        return True

    def _prepare_external_reference(self, ref_path):
        try:
            img_data = load_and_validate_fits(ref_path)
            header = fits.getheader(ref_path)
            prepared_ref = img_data
            if prepared_ref.ndim == 2:
                 bayer = header.get('BAYERPAT', self.bayer_pattern)
                 if isinstance(bayer, str) and bayer.upper() in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                      prepared_ref = debayer_image(prepared_ref, bayer.upper())
            elif prepared_ref.ndim == 3 and prepared_ref.shape[0] == 3:
                 prepared_ref = np.moveaxis(prepared_ref, 0, -1)
            if self.correct_hot_pixels:
                 prepared_ref = detect_and_correct_hot_pixels(prepared_ref, self.hot_pixel_threshold, self.neighborhood_size)
            self.reference_image = cv2.normalize(prepared_ref, None, 0, 65535, cv2.NORM_MINMAX)
            self.reference_header = header
            return True
        except Exception as e: self.update_progress(f"❌ Erreur préparation réf externe {os.path.basename(ref_path)}: {e}"); return False

    def stop(self):
        self.stop_processing = True
        self.update_progress("⛔ Arrêt demandé...")

    def is_running(self):
        return self.processing_active and self.processing_thread and self.processing_thread.is_alive()