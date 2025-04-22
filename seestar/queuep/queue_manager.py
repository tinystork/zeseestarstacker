# --- START OF FILE seestar/queuep/queue_manager.py ---
"""
Module de gestion de file d'attente pour le traitement des images astronomiques.
Gère l'alignement et l'empilement incrémental dans un thread séparé.
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
import gc # Import garbage collector

# Import core processing functions needed within this module
from seestar.core.image_processing import (
    load_and_validate_fits, # Returns float32 0-1 or None
    debayer_image,          # Expects float32 0-1, returns float32 0-1
    save_fits_image,        # Expects float32 0-1, saves uint16
    save_preview_image      # Expects float32 0-1
)
from seestar.core.hot_pixels import detect_and_correct_hot_pixels
from seestar.core.alignment import SeestarAligner # Use the aligner class

class SeestarQueuedStacker:
    """
    Classe pour l'empilement des images Seestar avec file d'attente.
    Gère l'alignement et l'empilement incrémental dans un thread séparé.
    """
    def __init__(self):
        """Initialise le stacker avec des valeurs par défaut."""
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
        self.temp_folder = None
        self.unaligned_folder = None

        # --- Reference Image & Alignment ---
        self.aligner = SeestarAligner()

        # --- Cumulative Stack ---
        self.current_stack_data = None
        self.current_stack_header = None
        self.current_stack_count = 0
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

        # --- Statistics ---
        self.files_in_queue = 0
        self.processed_files_count = 0
        self.aligned_files_count = 0
        self.failed_align_count = 0
        self.skipped_files_count = 0


    def initialize(self, output_dir):
        """Initialise le stacker et prépare les dossiers de sortie/temporaires."""
        try:
            self.output_folder = os.path.abspath(output_dir)
            self.temp_folder = os.path.join(self.output_folder, "temp_processing")
            self.unaligned_folder = os.path.join(self.output_folder, "unaligned_files")
            os.makedirs(self.output_folder, exist_ok=True)
            os.makedirs(self.temp_folder, exist_ok=True)
            os.makedirs(self.unaligned_folder, exist_ok=True)
            self.update_progress(f"🗄️ Dossiers prêts: Sortie='{os.path.basename(self.output_folder)}', NonAlign='{os.path.basename(self.unaligned_folder)}'")
        except OSError as e: self.update_progress(f"❌ Erreur création dossiers: {e}", 0); return False
        self.processed_files.clear();
        with self.folders_lock: self.additional_folders = []
        self.current_stack_data = None; self.current_stack_header = None; self.current_stack_count = 0
        self.total_exposure_seconds = 0.0; self.final_stacked_path = None; self.processing_error = None
        self.files_in_queue = 0; self.processed_files_count = 0; self.aligned_files_count = 0
        self.failed_align_count = 0; self.skipped_files_count = 0
        while not self.queue.empty():
             try: self.queue.get_nowait(); self.queue.task_done()
             except Empty: break
             except Exception: pass
        if self.perform_cleanup: self.cleanup_temp_files(clear_reference=True)
        self.aligner.stop_processing = False; self.aligner.reference_image_path = None
        return True

    def set_progress_callback(self, callback):
        self.progress_callback = callback
        self.aligner.set_progress_callback(callback)

    def update_progress(self, message, progress=None):
        """Safely calls the progress callback."""
        message = str(message) # Ensure string
        if self.progress_callback:
            try: self.progress_callback(message, progress)
            except Exception as e: print(f"Error in progress callback: {e}")
        else: print(message)

    def set_preview_callback(self, callback): self.preview_callback = callback

    def _update_preview(self, force_update=False):
        """Safely calls the preview callback, potentially throttling updates."""
        if self.preview_callback is None or self.current_stack_data is None: return
        update_frequency = 5
        if force_update or (self.current_stack_count % update_frequency == 0) or self.current_stack_count <= 2 :
            try:
                data_copy = self.current_stack_data.copy()
                header_copy = self.current_stack_header.copy() if self.current_stack_header else None
                stack_name = f"Stack ({self.current_stack_count} images)"
                self.preview_callback(data_copy, header_copy, stack_name)
            except Exception as e: print(f"Error in preview callback: {e}"); traceback.print_exc(limit=2)

    # --- Worker Thread Logic ---
    def _worker(self):
        """ Fonction principale exécutée dans le thread de traitement. """
        self.processing_active = True
        self.processing_error = None
        start_time_session = time.monotonic()
        reference_image_data = None # Holds the reference image data (float32 0-1)
        reference_header = None   # Holds the reference header

        try:
            # --- Step 1: Obtain Reference Image ---
            self.update_progress("⭐ Recherche/Préparation image référence...")
            initial_files = []
            if self.current_folder and os.path.isdir(self.current_folder):
                 try: initial_files = [f for f in os.listdir(self.current_folder) if f.lower().endswith(('.fit', '.fits'))]
                 except Exception as e: self.update_progress(f"Warning: Could not list initial files for ref finding: {e}")

            # Pass necessary parameters to aligner instance
            self.aligner.correct_hot_pixels = self.correct_hot_pixels
            self.aligner.hot_pixel_threshold = self.hot_pixel_threshold
            self.aligner.neighborhood_size = self.neighborhood_size
            self.aligner.bayer_pattern = self.bayer_pattern
            self.aligner.batch_size = self.batch_size

            # Call the aligner's method to get reference data and header
            reference_image_data, reference_header = self.aligner._get_reference_image(self.current_folder, initial_files)

            # --- *** IMPROVED ERROR HANDLING FOR REFERENCE *** ---
            if reference_image_data is None:
                 # Check if a manual path was provided and failed, or if auto-selection failed
                 user_ref_path = self.aligner.reference_image_path # Get path set by GUI
                 if user_ref_path and os.path.isfile(user_ref_path):
                      error_msg = f"Échec chargement/prétraitement référence MANUELLE: {os.path.basename(user_ref_path)}"
                 elif user_ref_path: # Path provided but not found/valid
                      error_msg = f"Fichier référence MANUELLE introuvable/invalide: {user_ref_path}"
                 else: # Auto-selection failed
                      error_msg = "Échec sélection automatique image référence (vérifiez les premières images et logs)."
                 raise RuntimeError(error_msg) # Raise specific error
            else:
                 # Save reference within the temp folder for potential inspection
                 self.aligner._save_reference_image(reference_image_data, reference_header, self.output_folder) # Pass main output dir
                 self.update_progress("⭐ Image de référence prête.", 5)
            # --- *** END OF IMPROVED ERROR HANDLING *** ---


            # --- Step 2: Process Queue ---
            while not self.stop_processing:
                file_path = None
                try:
                    file_path = self.queue.get(timeout=1.0)
                    file_name = os.path.basename(file_path)
                    abs_file_path = os.path.abspath(file_path)

                    # Skip check here - processed_files tracks files ADDED to queue
                    # if abs_file_path in self.processed_files: continue

                    process_success = self._process_file(file_path, reference_image_data)

                    self.queue.task_done()

                    # --- Progress Reporting ---
                    tasks_dequeued = self.processed_files_count + self.failed_align_count + self.skipped_files_count
                    current_progress = (tasks_dequeued / self.files_in_queue) * 100 if self.files_in_queue > 0 else 0
                    elapsed_time = time.monotonic() - start_time_session

                    if self.processed_files_count > 0:
                        time_per_successful_stack = elapsed_time / self.processed_files_count
                        remaining_tasks = max(0, self.files_in_queue - tasks_dequeued)
                        eta_seconds = remaining_tasks * time_per_successful_stack
                        h, rem = divmod(int(eta_seconds), 3600); m, s = divmod(rem, 60)
                        time_str = f"{h:02}:{m:02}:{s:02}"
                        progress_msg = f"📊 ({tasks_dequeued}/{self.files_in_queue}) {file_name} | ETA: {time_str}"
                    else:
                        progress_msg = f"📊 ({tasks_dequeued}/{self.files_in_queue}) {file_name}"

                    self.update_progress(progress_msg, current_progress)
                    if tasks_dequeued % 20 == 0: gc.collect()

                except Empty:
                    folder_to_process = None
                    with self.folders_lock:
                        if self.additional_folders: folder_to_process = self.additional_folders.pop(0)

                    if folder_to_process:
                        folder_name = os.path.basename(folder_to_process)
                        self.update_progress(f"📂 Traitement dossier sup: {folder_name}")
                        self.current_folder = folder_to_process
                        files_added = self._add_files_to_queue(folder_to_process)
                        if files_added > 0:
                            self.files_in_queue += files_added
                            self.update_progress(f"📋 {files_added} fichiers ajoutés depuis {folder_name}")
                            with self.folders_lock: folder_count = len(self.additional_folders)
                            self.update_progress(f"folder_count_update:{folder_count}")
                            continue
                        else:
                            self.update_progress(f"⚠️ Aucun nouveau FITS trouvé dans {folder_name}")
                            with self.folders_lock: folder_count = len(self.additional_folders)
                            self.update_progress(f"folder_count_update:{folder_count}")
                            continue
                    else:
                        self.update_progress("✅ File d'attente vide et aucun dossier supplémentaire.")
                        break # Exit the main processing loop

                except Exception as e:
                    error_context = f" de {file_name}" if file_path else " (file inconnu)"
                    self.update_progress(f"❌ Erreur boucle worker{error_context}: {e}")
                    traceback.print_exc(limit=3)
                    self.processing_error = f"Erreur boucle worker: {e}"
                    self.skipped_files_count += 1
                    if file_path: 
                        try: self.queue.task_done()
                        except ValueError:
                            pass
                    time.sleep(0.1)

            # --- End of While Loop ---

            # --- Step 3: Finalize ---
            if self.stop_processing: self.update_progress("⛔ Traitement arrêté par l'utilisateur.")
            else:
                if self.current_stack_data is not None and self.current_stack_count > 0:
                    self.update_progress("🏁 Finalisation et sauvegarde du stack final...")
                    self._save_final_stack()
                    self._update_preview(force_update=True) # Show final result
                    self.update_progress(f"🏁 Traitement terminé. {self.processed_files_count} images empilées.")
                elif self.processing_error: self.update_progress(f"🏁 Traitement terminé avec erreurs. Erreur principale: {self.processing_error}")
                else: self.update_progress("🏁 Traitement terminé, mais aucun stack n'a été créé (vérifiez les erreurs d'alignement/skip).")

        except RuntimeError as ref_err: # Catch specific reference error
             self.update_progress(f"❌ ERREUR CRITIQUE: {ref_err}")
             self.processing_error = str(ref_err) # Store error for UI
        except Exception as e: # Catch other critical errors
            self.update_progress(f"❌ Erreur critique thread worker: {e}")
            traceback.print_exc(limit=5)
            self.processing_error = f"Erreur critique: {e}"

        finally:
            # --- Step 4: Cleanup ---
            if self.perform_cleanup:
                self.update_progress("🧹 Nettoyage fichiers temporaires...")
                self.cleanup_temp_files(clear_reference=True) # Clear reference now we are done
            else:
                self.update_progress(f"ⓘ Fichiers temporaires conservés dans: {self.temp_folder}")

            self.processing_active = False
            self.update_progress("🚪 Thread traitement terminé.")
            # Ensure queue is empty after processing finishes or stops
            while not self.queue.empty():
                try: self.queue.get_nowait(); self.queue.task_done()
                except Exception:
                    break


    def _process_file(self, file_path, reference_image_data):
        """ Charge, prépare, aligne et empile une seule image. """
        file_name = os.path.basename(file_path)
        try:
            # --- 1. Load and Preprocess ---
            img_data = load_and_validate_fits(file_path)
            if img_data is None: self.skipped_files_count += 1; return False
            header = fits.getheader(file_path)
            if np.std(img_data) < 0.005: self.update_progress(f"⚠️ Image {file_name} ignorée (faible variance)."); self.skipped_files_count += 1; return False

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

            # --- 2. Align ---
            # Pass reference data directly
            aligned_img, align_success = self.aligner._align_image(prepared_img, reference_image_data, file_name)

            if not align_success:
                self.failed_align_count += 1
                try:
                    if os.path.exists(file_path): shutil.move(file_path, os.path.join(self.unaligned_folder, file_name)); self.update_progress(f"➡️ Échec alignement {file_name}. Déplacé.")
                    else: self.update_progress(f"⚠️ Original {file_name} non trouvé pour déplacement.")
                except Exception as move_err: self.update_progress(f"⚠️ Erreur déplacement {file_name}: {move_err}")
                return False

            self.aligned_files_count += 1

            # --- 3. Stack ---
            stack_success = self._stack_image_incrementally(aligned_img, header)

            if stack_success:
                # processed_files_count incremented in _stack_image_incrementally
                return True
            else:
                self.skipped_files_count += 1 # Count as skipped if stacking failed
                return False

        except Exception as e:
            self.update_progress(f"❌ Erreur traitement fichier {file_name}: {e}")
            traceback.print_exc(limit=3)
            self.skipped_files_count += 1
            if os.path.exists(file_path): 
                try: shutil.move(file_path, os.path.join(self.unaligned_folder, f"error_{file_name}"))
                except Exception: pass
            return False

    def _stack_image_incrementally(self, aligned_img, header):
        """Adds a single aligned image (float32 0-1) to the cumulative stack."""
        try:
            n = self.current_stack_count
            img_float = aligned_img.astype(np.float32)
            current_exposure = 0.0
            try: current_exposure = float(header.get('EXPTIME', 0.0))
            except (ValueError, TypeError): self.update_progress(f"⚠️ Impossible lire EXPTIME pour {header.get('FILENAME', 'image')}")

            if self.current_stack_data is None:
                # --- First image: Initialize stack ---
                self.current_stack_data = img_float
                self.current_stack_count = 1
                self.total_exposure_seconds = current_exposure
                self.current_stack_header = fits.Header()
                keys_to_copy = ['INSTRUME', 'TELESCOP', 'OBJECT', 'FILTER', 'DATE-OBS', 'GAIN', 'OFFSET', 'CCD-TEMP', 'RA', 'DEC', 'SITELAT', 'SITELONG', 'FOCALLEN']
                for key in keys_to_copy:
                    if key in header: 
                        try: self.current_stack_header[key] = (header[key], header.comments[key] if key in header.comments else '')
                        except Exception: self.current_stack_header[key] = header[key]
                self.current_stack_header['STACKTYP'] = (self.stacking_mode, 'Stacking method'); self.current_stack_header['NIMAGES'] = (1, 'Number of images in stack'); self.current_stack_header['TOTEXP'] = (self.total_exposure_seconds, '[s] Total exposure time')
                if self.stacking_mode in ["kappa-sigma", "winsorized-sigma"]: self.current_stack_header['KAPPA'] = (self.kappa, 'Kappa value for clipping')
                self.current_stack_header['CREATOR'] = ('SeestarStacker (Queued)', 'Processing Software'); self.current_stack_header['HISTORY'] = 'Stack Initialized'
                if self.correct_hot_pixels: self.current_stack_header['HISTORY'] = 'Hot pixel correction applied to input frames'
            else:
                # --- Subsequent images: Update stack ---
                if self.current_stack_data.shape != img_float.shape: self.update_progress(f"❌ Incompatibilité dims stack: Stack={self.current_stack_data.shape}, Img={img_float.shape}"); return False
                if self.stacking_mode == "mean": self.current_stack_data = (n * self.current_stack_data + img_float) / (n + 1)
                elif self.stacking_mode == "median":
                    w_old = n / (n + 1.0); w_new = 1.0 / (n + 1.0); self.current_stack_data = self.current_stack_data * w_old + img_float * w_new
                    if n == 1: self.update_progress("⚠️ Empilement 'médiane' est approximé en mode incrémental.", None)
                elif self.stacking_mode in ["kappa-sigma", "winsorized-sigma"]:
                    current_mean = self.current_stack_data
                    with np.errstate(invalid='ignore'): current_std = np.std(self.current_stack_data)
                    current_std = np.maximum(current_std, 1e-5); threshold = self.kappa * current_std; threshold = max(threshold, 1e-4)
                    if self.stacking_mode == "kappa-sigma":
                        deviation = np.abs(img_float - current_mean); keep_mask = deviation <= threshold
                        weight_new = keep_mask.astype(np.float32); new_denominator = n + weight_new; new_denominator = np.maximum(new_denominator, 1e-9)
                        self.current_stack_data = (n * self.current_stack_data + img_float * weight_new) / new_denominator
                        self.current_stack_data = np.nan_to_num(self.current_stack_data, nan=current_mean) # Fill rejected with previous mean
                    elif self.stacking_mode == "winsorized-sigma":
                        lower_bound = current_mean - threshold; upper_bound = current_mean + threshold
                        clipped_img = np.clip(img_float, lower_bound, upper_bound)
                        self.current_stack_data = (n * self.current_stack_data + clipped_img) / (n + 1)
                else: # Fallback to mean
                    if n == 1: self.update_progress(f"⚠️ Méthode '{self.stacking_mode}' inconnue, utilisation 'mean'.")
                    self.current_stack_data = (n * self.current_stack_data + img_float) / (n + 1)
                self.current_stack_count += 1; self.total_exposure_seconds += current_exposure
                self.current_stack_header['NIMAGES'] = self.current_stack_count; self.current_stack_header['TOTEXP'] = (round(self.total_exposure_seconds, 2), '[s] Total exposure time')

            self.current_stack_data = np.clip(self.current_stack_data, 0.0, 1.0) # Clip running stack
            self.processed_files_count += 1 # Increment count *after* successful processing

            if self.current_stack_count % 10 == 0: self._save_intermediate_stack()
            self._update_preview(); return True
        except Exception as e: self.update_progress(f"❌ Erreur empilement incrémental: {e}"); traceback.print_exc(limit=3); return False

    def _save_intermediate_stack(self):
        """ Sauvegarde le stack cumulatif intermédiaire. """
        if self.current_stack_data is None or self.output_folder is None: return
        stack_path = os.path.join(self.output_folder, "stack_cumulative.fit"); preview_path = os.path.join(self.output_folder, "stack_cumulative.png")
        try:
            if self.current_stack_header:
                 hist_keys = [k for k in self.current_stack_header if k.upper() == 'HISTORY'];
                 for k in hist_keys:
                      if 'Intermediate save' in str(self.current_stack_header[k]): del self.current_stack_header[k]
                 self.current_stack_header.add_history(f'Intermediate save after {self.current_stack_count} images')
            save_fits_image(self.current_stack_data, stack_path, self.current_stack_header, overwrite=True)
            save_preview_image(self.current_stack_data, preview_path, apply_stretch=True)
        except Exception as e: self.update_progress(f"⚠️ Erreur sauvegarde stack intermédiaire: {e}")

    def _save_final_stack(self):
        """ Sauvegarde le stack final (après la boucle principale). """
        if self.current_stack_data is None or self.output_folder is None or self.current_stack_count == 0: self.final_stacked_path = None; self.update_progress("ⓘ Aucun stack final à sauvegarder."); return
        self.final_stacked_path = os.path.join(self.output_folder, "stack_final.fit"); preview_path = os.path.join(self.output_folder, "stack_final.png"); self.update_progress(f"💾 Sauvegarde stack final: {self.final_stacked_path}...")
        try:
            final_header = self.current_stack_header.copy() if self.current_stack_header else fits.Header()
            hist_keys = [k for k in final_header if k.upper() == 'HISTORY']
            for k in hist_keys:
                if 'Intermediate save' in str(final_header[k]): del final_header[k]
            final_header.add_history('Final Stack Saved by Seestar Stacker (Queued)'); final_header['STACKCNT'] = (self.current_stack_count, 'Number of images successfully stacked'); final_header['ALIGNED'] = (self.aligned_files_count, 'Successfully aligned images'); final_header['FAILALIGN'] = (self.failed_align_count, 'Failed alignments'); final_header['SKIPPED'] = (self.skipped_files_count, 'Skipped/error files'); final_header['NIMAGES'] = self.current_stack_count; final_header['TOTEXP'] = (round(self.total_exposure_seconds, 2), '[s] Total exposure time')
            save_fits_image(self.current_stack_data, self.final_stacked_path, final_header, overwrite=True)
            save_preview_image(self.current_stack_data, preview_path, apply_stretch=True)
            self.update_progress(f"✅ Stack final sauvegardé ({self.current_stack_count} images)")
        except Exception as e: self.update_progress(f"⚠️ Erreur sauvegarde stack final: {e}"); traceback.print_exc(limit=2); self.final_stacked_path = None

    def cleanup_temp_files(self, clear_reference=False):
        """ Nettoie les fichiers temporaires dans temp_folder (optionnellement la référence). """
        if not self.temp_folder or not os.path.isdir(self.temp_folder): return
        deleted_count = 0; kept_count = 0
        try:
            reference_files = {"reference_image.fit", "reference_image.png"}
            for filename in os.listdir(self.temp_folder):
                file_path = os.path.join(self.temp_folder, filename); is_ref = filename in reference_files
                if os.path.isfile(file_path) and (not is_ref or clear_reference):
                    try: os.remove(file_path); deleted_count += 1
                    except Exception as del_e: self.update_progress(f"⚠️ Erreur suppression temp {filename}: {del_e}")
                elif is_ref and not clear_reference: kept_count +=1
            if deleted_count > 0: self.update_progress(f"🧹 {deleted_count} fichier(s) temporaire(s) supprimé(s) de {os.path.basename(self.temp_folder)}.")
        except Exception as e: self.update_progress(f"⚠️ Erreur nettoyage temp: {e}")

    # --- Folder & Queue Management ---
    def add_folder(self, folder_path):
        """Ajoute un dossier à la liste des dossiers supplémentaires à traiter."""
        abs_path = os.path.abspath(folder_path)
        if not os.path.isdir(abs_path): self.update_progress(f"❌ Dossier non trouvé: {folder_path}"); return False
        output_abs = os.path.abspath(self.output_folder) if self.output_folder else None
        temp_abs = os.path.abspath(self.temp_folder) if self.temp_folder else None
        if output_abs:
             norm_abs_path = os.path.normcase(abs_path); norm_output_path = os.path.normcase(output_abs)
             if norm_abs_path == norm_output_path or norm_abs_path.startswith(norm_output_path + os.sep): self.update_progress(f"⚠️ Impossible d'ajouter le dossier de sortie: {os.path.basename(folder_path)}"); return False
        if temp_abs and os.path.normcase(abs_path) == os.path.normcase(temp_abs): self.update_progress(f"⚠️ Impossible d'ajouter le dossier temporaire: {os.path.basename(folder_path)}"); return False
        with self.folders_lock:
            current_abs = os.path.abspath(self.current_folder) if self.current_folder else None
            existing_abs = [os.path.abspath(p) for p in self.additional_folders]
            if (current_abs and abs_path == current_abs) or abs_path in existing_abs: self.update_progress(f"ⓘ Dossier déjà en cours ou ajouté: {os.path.basename(folder_path)}"); return False
            try:
                if not any(f.lower().endswith(('.fit', '.fits')) for f in os.listdir(abs_path)): self.update_progress(f"⚠️ Aucun fichier FITS trouvé dans: {os.path.basename(folder_path)}. Non ajouté."); return False
            except Exception as e: self.update_progress(f"❌ Erreur lecture dossier {os.path.basename(folder_path)}: {e}"); return False
            self.additional_folders.append(abs_path)
            folder_count = len(self.additional_folders)
            self.update_progress(f"✅ Dossier ajouté à la file d'attente: {os.path.basename(folder_path)}")
            self.update_progress(f"folder_count_update:{folder_count}")
            return True

    def _add_files_to_queue(self, folder_path):
        """Ajoute les fichiers FITS d'un dossier à la file de traitement."""
        count_added = 0
        try:
            abs_folder_path = os.path.abspath(folder_path)
            files_in_folder = sorted(os.listdir(abs_folder_path))
            for fname in files_in_folder:
                if fname.lower().endswith(('.fit', '.fits')):
                    fpath = os.path.join(abs_folder_path, fname)
                    abs_fpath = os.path.abspath(fpath)
                    if abs_fpath not in self.processed_files:
                        self.queue.put(fpath)
                        self.processed_files.add(abs_fpath) # Add to set when queued
                        count_added += 1
            return count_added
        except Exception as e: self.update_progress(f"❌ Erreur ajout fichiers depuis {os.path.basename(folder_path)}: {e}"); return 0

    # --- Control Methods ---
    def start_processing(self, input_dir, output_dir, reference_path_ui=None):
        """Démarre le thread de traitement."""
        if self.processing_active: self.update_progress("⚠️ Traitement déjà en cours."); return False
        self.stop_processing = False
        self.current_folder = os.path.abspath(input_dir)
        if not self.initialize(output_dir): self.processing_active = False; return False
        self.aligner.reference_image_path = reference_path_ui or None # Set manual path for aligner
        files_added = self._add_files_to_queue(self.current_folder)
        self.files_in_queue = files_added
        if files_added == 0 and not reference_path_ui: self.update_progress("⚠️ Aucun fichier initial trouvé. Tentative recherche référence auto...")
        elif files_added == 0 and reference_path_ui: self.update_progress("⚠️ Aucun fichier initial, mais référence fournie. Attente dossiers ajoutés...")
        elif files_added > 0: self.update_progress(f"📋 {files_added} fichiers initiaux ajoutés à la file.")
        self.processing_thread = threading.Thread(target=self._worker, name="StackerWorker")
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.processing_active = True # Set flag AFTER thread starts
        return True

    def stop(self):
        """Signale au thread de traitement de s'arrêter."""
        if not self.processing_active: return
        self.update_progress("⛔ Arrêt demandé...")
        self.stop_processing = True
        self.aligner.stop_processing = True

    def is_running(self):
        """Vérifie si le thread de traitement est actif."""
        return self.processing_active and self.processing_thread and self.processing_thread.is_alive()
# --- END OF FILE seestar/queuep/queue_manager.py ---