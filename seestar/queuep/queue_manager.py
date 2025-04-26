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
from ..enhancement.drizzle_integration import DrizzleProcessor
try:
    import cupy
    _cupy_installed = True
except ImportError:
    _cupy_installed = False
# print("DEBUG: CuPy library not found globally.") # Optional debug
# --- Adjust relative import for utils ---
from ..core.utils import check_cupy_cuda, estimate_batch_size # Add check_cupy_cuda
# Import core processing functions needed within this module
from ..core.image_processing import (
    load_and_validate_fits,
    debayer_image,
    save_fits_image,
    save_preview_image

)
from ..core.hot_pixels import detect_and_correct_hot_pixels
from ..core.alignment import SeestarAligner
from ..core.utils import estimate_batch_size
# Correction ici: Utiliser le chemin complet depuis la racine 'seestar'
from ..enhancement.stack_enhancement import StackEnhancer
from ..core.utils import check_cupy_cuda, estimate_batch_size
from ..core.image_processing import (
    load_and_validate_fits,
    debayer_image,
    save_fits_image,
    save_preview_image
)
from ccdproc import CCDData                    
from ccdproc import combine as ccdproc_combine 

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

        # --- AJOUTER les attributs pour les paramètres Drizzle ---
        self.use_drizzle_processing = False # Sera défini par start_processing
        self.drizzle_scale = 2.0           # Sera défini par start_processing
        self.drizzle_wht_threshold = 0.7 # Stocker la valeur seuil (0-1)

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
            transform, (source_list, _target_list) = aa.find_transform(image_data, image_data)
            num_stars = len(source_list)
            max_stars_for_score = 200.0
            scores['stars'] = np.clip(num_stars / max_stars_for_score, 0.0, 1.0)

        except (aa.MaxIterError, ValueError) as star_err: # Handles specific astroalign errors
            self.update_progress(f"      Quality Scores -> Warning: Failed finding stars ({type(star_err).__name__}). Stars score set to 0.")
            scores = {'snr': scores.get('snr', 0.0), 'stars': 0.0} # Explicitly set scores
            return scores # Return immediately

        except Exception as e: # Handles any other unexpected error
            self.update_progress(f"      Quality Scores -> Error calculating stars: {e}. Stars score set to 0.")
            scores = {'snr': scores.get('snr', 0.0), 'stars': 0.0} # Explicitly set scores
            return scores # Return immediately

        # --- This section is ONLY reached if the 'try' block succeeds ---
        self.update_progress(f"      Quality Scores -> SNR: {scores['snr']:.2f}, Stars: {scores['stars']:.3f} ({num_stars} raw)")
        return scores # Return the successfully calculated scores


    def _worker(self):
            """Thread principal pour le traitement des images."""
            self.processing_active = True
            self.processing_error = None
            start_time_session = time.monotonic()
            reference_image_data = None
            reference_header = None

            # Liste pour accumuler les images alignées si Drizzle est activé
            # Stocke (aligned_data, header) tuples
            all_aligned_images_for_drizzle = [] if self.use_drizzle_processing else None

            try:
                # --- Préparation de l'image de référence ---
                self.update_progress("⭐ Recherche/Préparation image référence...")
                initial_files = []
                if self.current_folder and os.path.isdir(self.current_folder):
                    try:
                        initial_files = sorted([f for f in os.listdir(self.current_folder) if f.lower().endswith(('.fit', '.fits'))])
                    except Exception as e:
                        self.update_progress(f"Warning: Could not list initial files for ref finding: {e}")

                # Configurer l'aligneur avant de chercher la référence
                self.aligner.correct_hot_pixels = self.correct_hot_pixels
                self.aligner.hot_pixel_threshold = self.hot_pixel_threshold
                self.aligner.neighborhood_size = self.neighborhood_size
                self.aligner.bayer_pattern = self.bayer_pattern

                # Obtenir l'image de référence (traitée) et son en-tête
                reference_image_data, reference_header = self.aligner._get_reference_image(self.current_folder, initial_files)
                self.reference_header_for_wcs = reference_header # Store for WCS creation

                # Gérer l'échec de l'obtention de la référence
                if reference_image_data is None:
                    user_ref_path = self.aligner.reference_image_path
                    if user_ref_path and os.path.isfile(user_ref_path): error_msg = f"Échec chargement/prétraitement référence MANUELLE: {os.path.basename(user_ref_path)}"
                    elif user_ref_path: error_msg = f"Fichier référence MANUELLE introuvable/invalide: {user_ref_path}"
                    else: error_msg = "Échec sélection automatique image référence (vérifiez les premières images et logs)."
                    raise RuntimeError(error_msg) # Stoppe le worker si pas de référence
                else:
                    # Sauvegarder la référence trouvée/préparée (optionnel mais utile pour debug)
                    self.aligner._save_reference_image(reference_image_data, reference_header, self.output_folder)
                    self.update_progress("⭐ Image de référence prête.", 5)

                # Recalculer le nombre total de lots estimé basé sur la file initiale
                self._recalculate_total_batches()
                if self.use_quality_weighting:
                    self.update_progress(f"⚖️ Pondération qualité activée (SNR: {self.weight_by_snr}, Stars: {self.weight_by_stars})")

                # --- Boucle de traitement de la file ---
                while not self.stop_processing:
                    file_path = None
                    try:
                        # Obtenir le prochain fichier de la file
                        file_path = self.queue.get(timeout=1.0) # Timeout pour vérifier stop_processing
                        file_name = os.path.basename(file_path)

                        # Traiter le fichier (chargement, prétraitement, alignement, scores qualité)
                        aligned_data, header, quality_scores = self._process_file(file_path, reference_image_data)

                        self.processed_files_count += 1 # Compter chaque tentative

                        # Si le traitement a réussi (image alignée)
                        if aligned_data is not None:
                            self.aligned_files_count += 1

                            # --- Gestion différente selon si Drizzle est activé ou non ---
                            if self.use_drizzle_processing:
                                # **Mode Drizzle : Accumuler les données alignées**
                                # MODIFIÉ ICI pour stocker tuple (data, header)
                                all_aligned_images_for_drizzle.append((aligned_data, header))
                                # Log simple pour Drizzle
                                self.update_progress(f"   -> Image {self.aligned_files_count} alignée pour Drizzle.")
                            else:
                                # **Mode Classique : Ajouter au batch en cours**
                                # AJOUTÉ quality_scores ici
                                self.current_batch_data.append((aligned_data, header, quality_scores))
                                # Si le batch est plein (ET PAS en mode Drizzle), le traiter
                                if len(self.current_batch_data) >= self.batch_size:
                                    self.stacked_batches_count += 1
                                    # --- Utiliser la nouvelle méthode _process_completed_batch ---
                                    # Elle ne gère que le cas NON-Drizzle
                                    self._process_completed_batch(self.stacked_batches_count, self.total_batches_estimated)
                                    # Le batch est vidé à l'intérieur de _process_completed_batch

                        # Marquer la tâche comme terminée dans la queue
                        self.queue.task_done()

                        # --- Mise à jour de la progression et ETA (peut être mise à l'extérieur du if aligned_data) ---
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

                        # Appel périodique à gc.collect
                        if self.processed_files_count % 20 == 0: gc.collect()

                    # --- Gérer la file d'attente vide et les dossiers supplémentaires ---
                    except Empty:
                        # La file est vide, vérifier s'il reste un batch à traiter ou des dossiers
                        self.update_progress("ⓘ File d'attente vide. Vérification batch final / dossiers sup...")

                        # --- Traiter le dernier batch si mode NON-Drizzle ---
                        # MODIFIÉ : Ajouté `not self.use_drizzle_processing`
                        if not self.use_drizzle_processing and self.current_batch_data:
                            self.update_progress(f"⏳ Traitement du dernier batch ({len(self.current_batch_data)} images)...")
                            self.stacked_batches_count += 1
                            # --- Utiliser la nouvelle méthode _process_completed_batch ---
                            self._process_completed_batch(self.stacked_batches_count, self.total_batches_estimated)
                            # Le batch est vidé à l'intérieur

                        # --- Vérifier s'il y a des dossiers supplémentaires ---
                        folder_to_process = None
                        with self.folders_lock:
                            if self.additional_folders:
                                folder_to_process = self.additional_folders.pop(0)
                                folder_count = len(self.additional_folders)
                                self.update_progress(f"folder_count_update:{folder_count}") # Informe l'UI

                        if folder_to_process:
                            # Si un dossier supplémentaire est trouvé, l'ajouter à la queue
                            folder_name = os.path.basename(folder_to_process)
                            self.update_progress(f"📂 Traitement dossier sup: {folder_name}")
                            self.current_folder = folder_to_process # Mettre à jour le dossier courant
                            files_added = self._add_files_to_queue(folder_to_process)
                            if files_added > 0:
                                self.update_progress(f"📋 {files_added} fichiers ajoutés depuis {folder_name}. Total file d'attente: {self.files_in_queue}. Total lots estimé: {self.total_batches_estimated}")
                                continue # Reprendre la boucle pour traiter les nouveaux fichiers
                            else:
                                self.update_progress(f"⚠️ Aucun nouveau FITS trouvé dans {folder_name}")
                                continue # Vérifier s'il y a encore d'autres dossiers
                        else:
                            # Plus de fichiers dans la queue ET plus de dossiers supplémentaires
                            self.update_progress("✅ Fin de la file et des dossiers supplémentaires.")
                            break # Sortir de la boucle while principale

                    # --- Gérer les erreurs de traitement d'un fichier ---
                    except Exception as e:
                        error_context = f" de {file_name}" if file_path else " (file inconnu)"
                        self.update_progress(f"❌ Erreur boucle worker{error_context}: {e}")
                        traceback.print_exc(limit=3)
                        self.processing_error = f"Erreur boucle worker: {e}"
                        # Si l'erreur ne vient pas de la queue vide, marquer comme skipped
                        if file_path: self.skipped_files_count += 1
                        # Essayer de marquer la tâche comme terminée pour éviter blocage
                        try: self.queue.task_done()
                        except ValueError: pass # Si déjà terminée ou jamais commencée
                        time.sleep(0.1) # Petite pause après erreur

                # --- FIN DE LA BOUCLE WHILE ---

                # --- Étape Finale: Exécution Drizzle ou Sauvegarde Stack Classique ---
                if self.stop_processing:
                    self.update_progress("🛑 Traitement interrompu avant l'étape finale.")
                elif self.use_drizzle_processing:
                    # --- Exécuter Drizzle si activé ET s'il y a des images ---
                    if all_aligned_images_for_drizzle: # Contient maintenant des tuples (data, header)
                        self.update_progress(f"💧 Exécution du Drizzle final sur {len(all_aligned_images_for_drizzle)} images...")
                        try:
                            # Extraire juste les données d'image et les headers
                            images_data_list = [item[0] for item in all_aligned_images_for_drizzle]
                            images_headers_list = [item[1] for item in all_aligned_images_for_drizzle]

                            # --- Appel DrizzleProcessor ---
                            drizzle_proc = DrizzleProcessor(
                                scale_factor=self.drizzle_scale,
                                pixfrac=1.0 # 
                            )
                            # Passer les listes séparées
                            drizzled_image_np, weight_map = drizzle_proc.apply_drizzle(
                                images_data_list,
                                images_headers=images_headers_list
                            )

                            # --- Vérifier le résultat AVANT de déballer ---
                            if drizzled_image_np is not None:
                                # --- Succès Drizzle ---
                                self.current_stack_data = drizzled_image_np.astype(np.float32)
                                self.images_in_cumulative_stack = len(all_aligned_images_for_drizzle)

                                # --- Créer un en-tête FITS pour le résultat Drizzle ---
                                self.current_stack_header = fits.Header()
                                # Essayer de copier des infos de base du header de référence
                                if self.reference_header_for_wcs:
                                    try:
                                        keys_to_copy = ['INSTRUME', 'TELESCOP', 'OBJECT', 'FILTER', 'DATE-OBS']
                                        for key in keys_to_copy:
                                            if key in self.reference_header_for_wcs:
                                                self.current_stack_header[key] = (self.reference_header_for_wcs[key], self.reference_header_for_wcs.comments[key])
                                    except Exception: pass # Ignorer si erreur copie header réf

                                self.current_stack_header['NIMAGES'] = (self.images_in_cumulative_stack, 'Images combined in Drizzle stack')
                                self.current_stack_header['STACKTYP'] = ('Drizzle', 'Stacking method')
                                self.current_stack_header['DRZSCALE'] = (self.drizzle_scale, 'Drizzle scale factor')
                                self.current_stack_header['DRZWHTTH'] = (self.drizzle_wht_threshold, 'Drizzle WHT threshold used for masking')
                                self.current_stack_header['DRZKERNEL'] = ('square', 'Drizzle kernel used') # Ajuster si configurable
                                self.current_stack_header['DRZPIXFR'] = (1.0, 'Drizzle pixfrac used') # Ajuster si configurable
                                self.current_stack_header['CREATOR'] = ('SeestarStacker (Queued)', 'Processing Software')

                                # Recalculer l'exposition totale (estimation basée sur dernier header)
                                try:
                                     last_header = images_headers_list[-1] if images_headers_list else None
                                     self.total_exposure_seconds = self.images_in_cumulative_stack * float(last_header.get('EXPTIME', 10.0)) if last_header else 0.0
                                except Exception: self.total_exposure_seconds = 0.0
                                self.current_stack_header['TOTEXP'] = (round(self.total_exposure_seconds, 2), '[s] Estimated total exposure time')

                                self._update_preview() # Mettre à jour l'aperçu final
                                self._save_final_stack() # Sauvegarder le stack Drizzle
                            else:
                                # Échec retourné par drizzle_proc.apply_drizzle
                                self.update_progress("❌ Échec du traitement Drizzle final.")
                                self.processing_error = "Drizzle processing failed"
                                self.final_stacked_path = None

                        except Exception as drizzle_final_err:
                            self.update_progress(f"❌ Erreur pendant l'exécution Drizzle finale: {drizzle_final_err}")
                            traceback.print_exc(limit=3)
                            self.processing_error = f"Drizzle final error: {drizzle_final_err}"
                            self.final_stacked_path = None
                    else:
                        # Drizzle activé mais aucune image alignée
                        self.update_progress("⚠️ Drizzle activé mais aucune image alignée à traiter.")
                        self.final_stacked_path = None

                # --- Sauvegarder le stack final Classique (NON-DRIZZLE) ---
                elif self.current_stack_data is not None:
                    # --- APPLY FINAL POST-PROCESSING (Cropping, CLAHE) HERE ---
                    # Only for non-drizzle results, as Drizzle WHT masking handles edges
                    self.update_progress(f"✨ Application post-traitement final (Rognage/CLAHE)...")
                    try:
                        post_proc_config = {
                            'edge_crop_percent': 0.02, # Example: 2% crop
                            'clahe_params': {'clip_limit': 2.0, 'tile_grid_size': (8, 8)}, # Example CLAHE
                            # Ensure normalization doesn't re-scale the combined data unnecessarily
                            'normalization': 'none' # Use the 'none' option
                        }
                        post_enhancer = StackEnhancer(config=post_proc_config)
                        # Apply only cropping and CLAHE (using the new method)
                        # Note: CLAHE est désactivé dans StackEnhancer pour l'instant
                        self.current_stack_data = post_enhancer.apply_postprocessing(self.current_stack_data)
                        if self.current_stack_data is None:
                             raise ValueError("Post-processing returned None.") # Should not happen
                        else:
                             self.update_progress("   -> Post-traitement terminé.")
                    except NameError: # If StackEnhancer import failed
                         self.update_progress(f"⚠️ Erreur: StackEnhancer non disponible pour post-traitement.")
                    except Exception as post_err:
                        self.update_progress(f"⚠️ Erreur post-traitement final: {post_err}")
                        traceback.print_exc(limit=1)

                    # Sauvegarder le résultat (potentiellement post-traité)
                    self._save_final_stack()
                else:
                    # Aucun stack cumulatif n'a été créé (probablement aucune image valide)
                    self.update_progress("ⓘ Aucun stack cumulatif à sauvegarder (aucune image traitée?).")
                    self.final_stacked_path = None

            # --- Erreur critique avant la boucle principale (ex: référence) ---
            except RuntimeError as ref_err:
                self.update_progress(f"❌ ERREUR CRITIQUE: {ref_err}")
                self.processing_error = str(ref_err)
            # --- Erreur générale imprévue ---
            except Exception as e:
                self.update_progress(f"❌ Erreur critique thread worker: {e}")
                traceback.print_exc(limit=5)
                self.processing_error = f"Erreur critique: {e}"

            # --- Nettoyage final et arrêt ---
            finally:
                if self.perform_cleanup:
                    self.update_progress("🧹 Nettoyage fichiers temporaires...")
                    self.cleanup_unaligned_files()
                    self.cleanup_temp_reference()
                else:
                    self.update_progress(f"ⓘ Fichiers temporaires/non alignés conservés.")

                # Vider la queue au cas où (si arrêt prématuré)
                while not self.queue.empty():
                    try: self.queue.get_nowait(); self.queue.task_done()
                    except Exception: break

                # Réinitialiser les données temporaires
                self.current_batch_data = []
                if all_aligned_images_for_drizzle is not None:
                    all_aligned_images_for_drizzle.clear() # Vider la liste Drizzle
                gc.collect() # Appel final à gc

                # Marquer la fin du traitement
                self.processing_active = False
                self.update_progress("🚪 Thread traitement terminé.")


# --- FIN DE LA MÉTHODE _worker ---

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

    # --- New ---

    def _process_file(self, file_path, reference_image_data):
        """
        Traite un seul fichier image : chargement, validation, pré-traitement,
        alignement sur l'image de référence, et calcul des scores qualité.

        Args:
            file_path (str): Chemin complet du fichier FITS à traiter.
            reference_image_data (np.ndarray): Données de l'image de référence (alignée/prête).

        Returns:
            tuple: (aligned_data, header, quality_scores)
                   - aligned_data (np.ndarray ou None): Données alignées (float32, 0-1) ou None si échec.
                   - header (fits.Header ou None): En-tête FITS de l'image originale ou None.
                   - quality_scores (dict): Dictionnaire {'snr': float, 'stars': float}.
        """
        file_name = os.path.basename(file_path)
        # Initialiser les scores qualité par défaut
        quality_scores = {'snr': 0.0, 'stars': 0.0}
        # Log début traitement fichier
        self.update_progress(f"   Traitement Fichier: {file_name}")

        try:
            # 1. Charger et valider le fichier FITS
            img_data = load_and_validate_fits(file_path)
            if img_data is None:
                # Message d'erreur court dans le log principal
                self.update_progress(f"   ⚠️ Échec chargement/validation.")
                self.skipped_files_count += 1
                # Retourner les scores par défaut car le fichier n'a pas pu être traité
                return None, None, quality_scores

            # Obtenir l'en-tête FITS
            header = fits.getheader(file_path)

            # 2. Vérification initiale de la variance (image trop plate/noire?)
            std_dev = np.std(img_data)
            # Seuil de variance (peut être ajusté si nécessaire)
            variance_threshold = 0.0015
            if std_dev < variance_threshold:
                self.update_progress(f"   ⚠️ Image ignorée (faible variance: {std_dev:.4f}).")
                self.skipped_files_count += 1
                return None, None, quality_scores

            # 3. Pré-traitement (Debayering, Correction Pixels Chauds)
            prepared_img = img_data # Partir de l'image chargée
            # Debayer si nécessaire (image 2D et motif Bayer valide)
            if prepared_img.ndim == 2:
                bayer = header.get('BAYERPAT', self.bayer_pattern) # Utilise le paramètre de la classe
                if isinstance(bayer, str) and bayer.upper() in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                    try:
                        prepared_img = debayer_image(prepared_img, bayer.upper())
                        # self.update_progress(f"      Debayer appliqué ({bayer.upper()})") # Log optionnel
                    except ValueError as de:
                        # Si debayer échoue, on continue avec l'image N&B mais on log
                        self.update_progress(f"   ⚠️ Erreur debayer: {de}. Tentative N&B.")
                # else: self.update_progress("      Image N&B ou BAYERPAT invalide, pas de debayer.") # Log optionnel

            # Correction pixels chauds si activée
            if self.correct_hot_pixels:
                try:
                    # self.update_progress("      Correction pixels chauds...") # Log optionnel
                    prepared_img = detect_and_correct_hot_pixels(
                        prepared_img, self.hot_pixel_threshold, self.neighborhood_size
                    )
                except Exception as hp_err:
                    self.update_progress(f"   ⚠️ Erreur correction px chauds: {hp_err}.")
                    # Continuer même si la correction échoue

            # Assurer que l'image est en float32 pour l'alignement
            prepared_img = prepared_img.astype(np.float32)

            # 4. Alignement de l'image préparée sur la référence
            # On utilise la méthode _align_image de l'instance SeestarAligner
            # self.update_progress("      Alignement...") # Log optionnel
            aligned_img, align_success = self.aligner._align_image(
                prepared_img, reference_image_data, file_name # Passer le nom pour les logs d'erreur internes
            )

            # Gérer l'échec de l'alignement
            if not align_success:
                self.failed_align_count += 1
                # Déplacer le fichier original (non traitable) vers le dossier 'unaligned'
                try:
                    if os.path.exists(file_path):
                        target_path = os.path.join(self.unaligned_folder, file_name)
                        # Utiliser normpath pour éviter pbs chemin Windows/Linux
                        shutil.move(os.path.normpath(file_path), os.path.normpath(target_path))
                        # Message log court, le log interne de _align_image donne la raison
                        self.update_progress(f"   ➡️ Échec alignement. Déplacé vers non alignés.")
                    else:
                        # Si le fichier original n'existe plus (étrange, mais sécurité)
                        self.update_progress(f"   ⚠️ Échec alignement, original non trouvé pour déplacer.")
                except Exception as move_err:
                    # Erreur lors du déplacement (ex: permissions, fichier verrouillé)
                    self.update_progress(f"   ⚠️ Erreur déplacement après échec alignement: {move_err}")
                # Retourner None pour les données, l'en-tête et les scores par défaut
                return None, None, quality_scores

            # 5. Calcul des métriques de qualité (SI la pondération est activée)
            if self.use_quality_weighting:
                # Le log interne de _calculate_quality_metrics indiquera les scores
                self.update_progress(f"      Calcul Scores Qualité...")
                quality_scores = self._calculate_quality_metrics(aligned_img)
            # else: # Log optionnel si pondération désactivée
                # self.update_progress(f"      Scores Qualité -> Ignoré (Pondération Désactivée)")

            # 6. Retourner les résultats : image alignée, en-tête original, scores qualité
            # self.update_progress(f"   ✅ Fichier traité avec succès.") # Log optionnel
            return aligned_img, header, quality_scores

        # --- Gestion globale des erreurs pour ce fichier ---
        except Exception as e:
            self.update_progress(f"❌ Erreur traitement fichier {file_name}: {e}")
            traceback.print_exc(limit=3) # Afficher la trace pour le débogage
            self.skipped_files_count += 1 # Compter comme "skipped" en cas d'erreur générale
            # Essayer de déplacer le fichier problématique
            if os.path.exists(file_path):
                try:
                    target_path = os.path.join(self.unaligned_folder, f"error_{file_name}")
                    shutil.move(os.path.normpath(file_path), os.path.normpath(target_path))
                except Exception: pass # Ignorer les erreurs de déplacement ici

            # Retourner None et les scores par défaut en cas d'erreur
            return None, None, quality_scores

# --- FIN DE LA MÉTHODE _process_file ---


    def _process_completed_batch(self, current_batch_num, total_batches_est):
        """
        Traite un batch complété pour le stacking CLASSIQUE (non-Drizzle).
        Appelle _stack_batch pour combiner les images du lot, puis
        combine le résultat dans le stack cumulatif.
        Vide self.current_batch_data après traitement.
        """
        if not self.current_batch_data:
            self.update_progress(f"⚠️ Tentative de traiter un batch vide (Batch #{current_batch_num}).", None)
            return

        batch_size = len(self.current_batch_data)
        progress_info = f"(Lot {current_batch_num}/{total_batches_est if total_batches_est > 0 else '?'})"

        # Message indiquant le début du traitement pour ce lot
        self.update_progress(f"⚙️ Traitement classique du batch {progress_info} ({batch_size} images)...")

        # Extraire les données nécessaires pour _stack_batch
        # Filtrer les None potentiels (bien que _process_file devrait les retourner comme None)
        batch_images = [item[0] for item in self.current_batch_data if item[0] is not None]
        batch_headers = [item[1] for item in self.current_batch_data if item[0] is not None]
        batch_scores = [item[2] for item in self.current_batch_data if item[0] is not None] # Scores qualité

        # Vérifier s'il reste des images valides dans le lot après filtrage
        if not batch_images:
            self.update_progress(f"⚠️ Aucune image valide dans le lot {progress_info} après filtrage.")
            self.failed_stack_count += batch_size # Compter les images initiales comme échec
            self.current_batch_data = [] # Vider le lot même s'il était invalide
            gc.collect()
            return

        # --- Appeler _stack_batch pour combiner les images de ce lot ---
        # _stack_batch gère maintenant la combinaison (mean, median, ccdproc) et les poids
        stacked_batch_data_np, stack_info_header = self._stack_batch(
            batch_images, batch_headers, batch_scores, current_batch_num, total_batches_est
        )

        # --- Combiner le résultat du batch dans le stack cumulatif ---
        if stacked_batch_data_np is not None:
            self._combine_batch_result(stacked_batch_data_np, stack_info_header)
            # Mettre à jour l'aperçu avec le stack cumulatif
            self._update_preview()
            # Sauvegarder le stack intermédiaire (cumulatif)
            self._save_intermediate_stack()
        else:
            # Si _stack_batch a échoué pour ce lot
            # Compter les images VALIDES qui ont échoué au stack
            self.failed_stack_count += len(batch_images)
            self.update_progress(f"❌ Échec combinaison lot {progress_info}. {len(batch_images)} images ignorées.", None)

        # --- Vider le batch traité ---
        self.current_batch_data = []
        gc.collect()

# --- FIN DE LA MÉTHODE _process_completed_batch (MODIFIÉE) ---  
    
    def _combine_batch_result(self, stacked_batch_data_np, stack_info_header):
        """
        Combine le résultat numpy (float32, 0-1) d'un batch traité
        dans le stack cumulatif (self.current_stack_data).

        Gère l'initialisation du stack cumulatif lors du premier batch.
        Utilise une moyenne pondérée par le nombre d'images pour combiner.
        Tente d'utiliser CuPy pour l'accélération si disponible.

        Args:
            stacked_batch_data_np (np.ndarray): Image (float32, 0-1) résultant du
                                                traitement du batch par _stack_batch.
            stack_info_header (fits.Header): En-tête contenant les informations
                                             sur le traitement de ce batch (NIMAGES, TOTEXP, etc.).
        """
        if stacked_batch_data_np is None or stack_info_header is None:
            self.update_progress("⚠️ Erreur interne: Données de batch invalides pour combinaison.")
            return

        try:
            # Récupérer les informations du batch depuis l'en-tête fourni
            batch_n = int(stack_info_header.get('NIMAGES', 1))
            batch_exposure = float(stack_info_header.get('TOTEXP', 0.0))

            # Vérifier si le nombre d'images est valide
            if batch_n <= 0:
                self.update_progress(f"⚠️ Batch combiné avec {batch_n} images, ignoré.")
                return

            # --- Initialisation du Stack Cumulatif (Premier Batch) ---
            if self.current_stack_data is None:
                self.update_progress("   -> Initialisation du stack cumulatif...")
                # La première image est simplement le résultat du premier batch
                # S'assurer que c'est bien un float32
                self.current_stack_data = stacked_batch_data_np.astype(np.float32)
                self.images_in_cumulative_stack = batch_n
                self.total_exposure_seconds = batch_exposure

                # --- Créer l'en-tête initial pour le stack cumulatif ---
                self.current_stack_header = fits.Header()
                # Copier les infos pertinentes de l'en-tête du batch
                keys_to_copy_from_batch = ['NIMAGES', 'STACKMETH', 'TOTEXP', 'ENH_CROP', 'ENH_CLHE', 'KAPPA', 'WGHT_USED', 'WGHT_MET']
                for key in keys_to_copy_from_batch:
                    if key in stack_info_header:
                        # Copie simple clé/valeur/commentaire si possible
                        try:
                            self.current_stack_header[key] = (stack_info_header[key], stack_info_header.comments[key])
                        except KeyError: # Si pas de commentaire
                            self.current_stack_header[key] = stack_info_header[key]

                # Ajouter des informations générales / potentiellement manquantes
                # Ces valeurs pourraient être affinées dans _save_final_stack
                if 'STACKTYP' not in self.current_stack_header:
                     self.current_stack_header['STACKTYP'] = (self.stacking_mode, 'Overall stacking method')
                if 'WGHT_ON' not in self.current_stack_header:
                     self.current_stack_header['WGHT_ON'] = (self.use_quality_weighting, 'Quality weighting status')

                self.current_stack_header['CREATOR'] = ('SeestarStacker (Queued)', 'Processing Software')
                self.current_stack_header.add_history('Cumulative Stack Initialized')
                if self.correct_hot_pixels:
                    self.current_stack_header.add_history('Hot pixel correction applied to input frames')

            # --- Combinaison avec le Stack Cumulatif Existant ---
            else:
                self.update_progress("   -> Combinaison avec le stack cumulatif...")
                # Vérifier la compatibilité des dimensions
                if self.current_stack_data.shape != stacked_batch_data_np.shape:
                    self.update_progress(f"❌ Incompatibilité dims stack: Cumul={self.current_stack_data.shape}, Batch={stacked_batch_data_np.shape}. Combinaison échouée.")
                    # Que faire ? Option : essayer de redimensionner ? Pour l'instant, on abandonne.
                    # Ne pas mettre à jour les compteurs si la combinaison échoue.
                    return

                # Calcul des poids basé sur le nombre d'images
                current_n = self.images_in_cumulative_stack
                total_n = current_n + batch_n
                w_old = current_n / total_n
                w_new = batch_n / total_n

                # --- Tentative de combinaison via CuPy si disponible ---
                use_cupy_combine = _cupy_installed and check_cupy_cuda()
                combined_np = None # Variable pour stocker le résultat (toujours NumPy)

                if use_cupy_combine:
                    gpu_current = None; gpu_batch = None # Initialiser
                    try:
                        # print("DEBUG: Combining stacks using CuPy") # Décommenter pour debug
                        # Assurer float32 pour GPU
                        gpu_current = cupy.asarray(self.current_stack_data, dtype=cupy.float32)
                        gpu_batch = cupy.asarray(stacked_batch_data_np, dtype=cupy.float32)

                        # Moyenne pondérée sur GPU
                        gpu_combined = (gpu_current * w_old) + (gpu_batch * w_new)
                        combined_np = cupy.asnumpy(gpu_combined) # Télécharger le résultat

                    except cupy.cuda.memory.OutOfMemoryError:
                        print("Warning: GPU Out of Memory during stack combination. Falling back to CPU.")
                        use_cupy_combine = False # Forcer le fallback CPU
                        gc.collect(); cupy.get_default_memory_pool().free_all_blocks()
                    except Exception as gpu_err:
                        print(f"Warning: CuPy error during stack combination: {gpu_err}. Falling back to CPU.")
                        traceback.print_exc(limit=1)
                        use_cupy_combine = False
                        gc.collect()
                        try: cupy.get_default_memory_pool().free_all_blocks()
                        except Exception: pass
                    finally:
                        # Libérer explicitement la mémoire GPU
                        del gpu_current, gpu_batch
                        if '_cupy_installed' in globals() and _cupy_installed:
                             try: cupy.get_default_memory_pool().free_all_blocks()
                             except Exception: pass

                # --- Combinaison via NumPy (Fallback ou si CuPy non utilisé) ---
                if not use_cupy_combine:
                    # print("DEBUG: Combining stacks using NumPy") # Décommenter pour debug
                    # Assurer float32 pour les calculs NumPy
                    current_data_float = self.current_stack_data.astype(np.float32)
                    batch_data_float = stacked_batch_data_np.astype(np.float32)
                    # Moyenne pondérée
                    combined_np = (current_data_float * w_old) + (batch_data_float * w_new)

                # --- Mettre à jour le stack cumulatif ---
                if combined_np is None:
                     # Cela ne devrait pas arriver si l'une des méthodes a fonctionné
                     raise RuntimeError("La combinaison n'a produit aucun résultat (erreur CuPy et NumPy?).")

                # Le résultat DOIT être un NumPy array float32
                self.current_stack_data = combined_np.astype(np.float32)

                # --- Mettre à jour les statistiques et l'en-tête cumulatif ---
                self.images_in_cumulative_stack = total_n
                self.total_exposure_seconds += batch_exposure
                if self.current_stack_header: # Vérifier si l'en-tête existe
                    self.current_stack_header['NIMAGES'] = self.images_in_cumulative_stack
                    self.current_stack_header['TOTEXP'] = (round(self.total_exposure_seconds, 2), '[s] Total exposure time')
                    # Optionnel: Ajouter une entrée d'historique (peut alourdir)
                    # self.current_stack_header.add_history(f'Combined with batch stack of {batch_n} images')

            # --- Clip final du résultat cumulé ---
            # Assurer que les valeurs restent dans la plage [0, 1] après combinaison
            self.current_stack_data = np.clip(self.current_stack_data, 0.0, 1.0)

        except Exception as e:
            self.update_progress(f"❌ Erreur pendant la combinaison du résultat du batch: {e}")
            traceback.print_exc(limit=3)
            # Optionnel: Que faire si la combinaison échoue ? Arrêter ? Continuer sans combiner ?
            # Pour l'instant, on log l'erreur mais on continue le processus global.
            # Le stack cumulatif ne sera pas mis à jour avec ce batch.

# --- Fin de la méthode _combine_batch_result ---
    
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
                #self.current_stack_header.add_history(f'Combined with batch stack of {batch_n} images') was cluterring the header

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


# --- DANS LA CLASSE SeestarQueuedStacker DANS seestar/queuep/queue_manager.py ---
# REMPLACEZ LA MÉTHODE _stack_batch EXISTANTE PAR CELLE-CI (GÈRE LA COULEUR) :

    def _stack_batch(self, batch_images, batch_headers, batch_scores, current_batch_num=0, total_batches_est=0):
        """
        Combine un lot d'images alignées (2D ou 3D) en utilisant ccdproc.
        Traite les canaux couleur séparément si nécessaire.
        Applique les poids qualité si activés.

        Args:
            batch_images (list): Liste d'arrays NumPy (float32, 0-1). Déjà alignées.
            batch_headers (list): Liste des en-têtes FITS originaux.
            batch_scores (list): Liste des dicts de scores qualité {'snr', 'stars'}.
            current_batch_num (int): Numéro du lot pour les logs.
            total_batches_est (int): Estimation totale des lots pour les logs.

        Returns:
            tuple: (stacked_image_np, stack_info_header) or (None, None) on failure.
        """
        if not batch_images:
            self.update_progress(f"❌ Erreur interne: _stack_batch reçu un lot vide.")
            return None, None

        num_images = len(batch_images)
        progress_info = f"(Lot {current_batch_num}/{total_batches_est if total_batches_est > 0 else '?'})"
        self.update_progress(f"✨ Combinaison via ccdproc du batch {progress_info} ({num_images} images)...")

        # Déterminer si les images sont en couleur
        ref_shape = batch_images[0].shape
        is_color = len(ref_shape) == 3 and ref_shape[2] == 3

        # --- Calculer les poids (une seule fois, applicable à tous les canaux) ---
        weights = None
        weighting_applied = False
        if self.use_quality_weighting and batch_scores and len(batch_scores) == num_images:
            try:
                self.update_progress(f"   -> Calcul des poids qualité pour {num_images} images...")
                weights = self._calculate_weights(batch_scores)
                weighting_applied = True
                self.update_progress(f"   -> Poids qualité calculés.")
            except Exception as w_err:
                self.update_progress(f"   ⚠️ Erreur calcul poids qualité: {w_err}. Utilisation poids uniformes.")
                weights = None
                weighting_applied = False
        else:
            self.update_progress(f"   -> Utilisation de poids uniformes.")
            weighting_applied = False

        # --- Stack images ---
        stacked_batch_data_np = None
        stack_method_used = self.stacking_mode
        kappa_val = float(self.kappa)

        try:
            if is_color:
                # --- Traitement Couleur (par canal) ---
                self.update_progress("   -> Traitement couleur par canal...")
                stacked_channels = []
                final_stack_method_str = "" # Pour le header

                for c in range(3): # Boucle sur R, G, B
                    channel_name = ['R', 'G', 'B'][c]
                    self.update_progress(f"      -> Combinaison Canal {channel_name}...")
                    ccd_list_channel = []

                    # Créer la liste CCDData pour ce canal
                    for img_np, hdr in zip(batch_images, batch_headers):
                        if img_np is None or img_np.ndim != 3: continue # Skip invalides
                        channel_data = img_np[..., c] # Extraire le canal 2D
                        exposure = float(hdr.get('EXPTIME', 1.0)) if hdr else 1.0
                        ccd = CCDData(channel_data, unit='adu', meta=hdr)
                        ccd.meta['EXPOSURE'] = exposure
                        ccd_list_channel.append(ccd)

                    if not ccd_list_channel:
                        raise ValueError(f"Aucune image valide pour le canal {channel_name}.")

                    # Configurer les args pour ce canal
                    combine_args_ch = {'ccd_list': ccd_list_channel}
                    ch_stack_method = self.stacking_mode # Utiliser la méthode globale
                    if ch_stack_method == 'mean': combine_args_ch['method'] = 'average'
                    elif ch_stack_method == 'median': combine_args_ch['method'] = 'median'
                    elif ch_stack_method in ['kappa-sigma', 'winsorized-sigma']:
                        combine_args_ch['method'] = 'average'; combine_args_ch['sigma_clip'] = True
                        combine_args_ch['sigma_lower_thresh'] = kappa_val; combine_args_ch['sigma_upper_thresh'] = kappa_val
                        ch_stack_method = f"kappa-sigma({kappa_val:.1f})" # Nom méthode pour header
                    else: combine_args_ch['method'] = 'average'; ch_stack_method = 'average (fallback)'

                    if weights is not None: combine_args_ch['weights'] = weights # Appliquer les mêmes poids

                    # Combiner ce canal
                    combined_ccd_ch = ccdproc_combine(ccd_list_channel, **combine_args_ch)
                    stacked_channels.append(combined_ccd_ch.data.astype(np.float32))

                    # Stocker la méthode utilisée (sera la même pour tous les canaux)
                    if c == 0: final_stack_method_str = ch_stack_method

                # Vérifier si tous les canaux ont été traités
                if len(stacked_channels) != 3:
                    raise RuntimeError("Le traitement couleur n'a pas produit 3 canaux.")

                # Réassembler l'image couleur
                stacked_batch_data_np = np.stack(stacked_channels, axis=-1)
                stack_method_used = final_stack_method_str # Mettre à jour pour le header

            else:
                # --- Traitement N&B (comme avant) ---
                self.update_progress("   -> Traitement N&B...")
                ccd_list = []
                for img_np, hdr in zip(batch_images, batch_headers):
                    if img_np is None or img_np.ndim != 2: continue # Skip invalides
                    exposure = float(hdr.get('EXPTIME', 1.0)) if hdr else 1.0
                    ccd = CCDData(img_np, unit='adu', meta=hdr)
                    ccd.meta['EXPOSURE'] = exposure
                    ccd_list.append(ccd)

                if not ccd_list:
                    raise ValueError("Aucune image N&B valide à convertir en CCDData.")

                combine_args = {'ccd_list': ccd_list}
                if stack_method_used == 'mean': combine_args['method'] = 'average'
                elif stack_method_used == 'median': combine_args['method'] = 'median'
                elif stack_method_used in ['kappa-sigma', 'winsorized-sigma']:
                    combine_args['method'] = 'average'; combine_args['sigma_clip'] = True
                    combine_args['sigma_lower_thresh'] = kappa_val; combine_args['sigma_upper_thresh'] = kappa_val
                    if stack_method_used == 'winsorized-sigma': self.update_progress(f"   ℹ️ Mode 'winsorized' traité comme kappa-sigma ({kappa_val:.1f}) dans ccdproc.")
                    stack_method_used = f"kappa-sigma({kappa_val:.1f})"
                else: combine_args['method'] = 'average'; stack_method_used = 'average (fallback)'

                if weights is not None: combine_args['weights'] = weights

                self.update_progress(f"   -> Combinaison ccdproc (Méthode: {combine_args.get('method', '?')}, SigmaClip: {combine_args.get('sigma_clip', False)})...")
                combined_ccd = ccdproc_combine(ccd_list, **combine_args)
                stacked_batch_data_np = combined_ccd.data.astype(np.float32)

            # --- Création de l'en-tête d'information commun ---
            stack_info_header = fits.Header()
            stack_info_header['NIMAGES'] = (num_images, 'Images combined in this batch')
            stack_info_header['STACKMETH'] = (stack_method_used, 'Method used for this batch')
            if 'kappa' in stack_method_used.lower(): # Vérifie si kappa-sigma a été utilisé
                 stack_info_header['KAPPA'] = (kappa_val, 'Kappa value for clipping')
            stack_info_header['WGHT_USED'] = (weighting_applied, 'Quality weights applied to this batch')
            if weighting_applied:
                w_metrics = []
                if self.weight_by_snr: w_metrics.append(f"SNR^{self.snr_exponent:.1f}")
                if self.weight_by_stars: w_metrics.append(f"Stars^{self.stars_exponent:.1f}")
                stack_info_header['WGHT_MET'] = (",".join(w_metrics), 'Metrics used for weighting')
            batch_exposure = sum(float(h.get('EXPTIME', 0.0)) for h in batch_headers if h is not None)
            stack_info_header['TOTEXP'] = (round(batch_exposure, 2), '[s] Exposure time of this batch')

            # --- Normalisation 0-1 du résultat du batch ---
            min_val, max_val = np.nanmin(stacked_batch_data_np), np.nanmax(stacked_batch_data_np)
            if max_val > min_val:
                stacked_batch_data_np = (stacked_batch_data_np - min_val) / (max_val - min_val)
            else: # Image constante
                stacked_batch_data_np = np.zeros_like(stacked_batch_data_np)
            stacked_batch_data_np = np.clip(stacked_batch_data_np, 0.0, 1.0)

            self.update_progress(f"✅ Combinaison lot {progress_info} terminée (Shape: {stacked_batch_data_np.shape}).")

            return stacked_batch_data_np.astype(np.float32), stack_info_header # Assurer float32

        # --- Gestion des erreurs ---
        except MemoryError as mem_err:
            print(f"\n❌ ERREUR MÉMOIRE Combinaison Lot {progress_info}: {mem_err}")
            traceback.print_exc(limit=1)
            self.update_progress(f"❌ ERREUR Mémoire Lot {progress_info}. Lot ignoré.")
            ccd_list = []; ccd_list_channel = [] # Effacer listes
            gc.collect()
            return None, None
        except Exception as stack_err:
            print(f"\n❌ ERREUR Combinaison Lot {progress_info}: {stack_err}")
            traceback.print_exc(limit=3)
            self.update_progress(f"❌ ERREUR Combinaison Lot {progress_info}. Lot ignoré.")
            ccd_list = []; ccd_list_channel = []
            gc.collect()
            return None, None

# --- FIN DE LA MÉTHODE _stack_batch (CORRIGÉE POUR COULEUR) ---

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
            abs_folder_path = os.path.abspath(folder_path)
            # ---> AJOUTER CETTE LIGNE <---
            print(f"DEBUG [_add_files_to_queue]: Scanning absolute path: '{abs_folder_path}'")
            # ------------------------------
            self.update_progress(f"🔍 Scan du dossier: {os.path.basename(folder_path)}...")
            files_in_folder = sorted(os.listdir(abs_folder_path))
            # ---> AJOUTER CETTE LIGNE <---
            print(f"DEBUG [_add_files_to_queue]: os.listdir found: {files_in_folder}")
            # ------------------------------
            new_files_found_in_folder = []
            for fname in files_in_folder:
                # ---> AJOUTER CETTE LIGNE (optionnel mais peut aider) <---
                print(f"DEBUG [_add_files_to_queue]: Checking file: '{fname}'")
                # ---------------------------------------------------------
                if self.stop_processing: self.update_progress("⛔ Scan interrompu."); break
                if fname.lower().endswith(('.fit', '.fits')):
                    fpath = os.path.join(abs_folder_path, fname)
                    abs_fpath = os.path.abspath(fpath)
                    if abs_fpath not in self.processed_files:
                        # ---> AJOUTER CETTE LIGNE <---
                        print(f"DEBUG [_add_files_to_queue]: ADDING to queue and processed_files: '{fpath}'")
                        # ------------------------------
                        self.queue.put(fpath)
                        self.processed_files.add(abs_fpath)
                        count_added += 1
            if count_added > 0: self.files_in_queue += count_added; self._recalculate_total_batches()
            return count_added
        except FileNotFoundError: self.update_progress(f"❌ Erreur scan: Dossier introuvable {os.path.basename(folder_path)}"); return 0
        except PermissionError: self.update_progress(f"❌ Erreur scan: Permission refusée {os.path.basename(folder_path)}"); return 0
        except Exception as e: self.update_progress(f"❌ Erreur scan dossier {os.path.basename(folder_path)}: {e}"); return 0

    # --- MODIFIED start_processing ---
    def start_processing(self, input_dir, output_dir, reference_path_ui=None,
                     initial_additional_folders=None,
                     # Weighting
                     use_weighting=False, weight_snr=True, weight_stars=True,
                     snr_exp=1.0, stars_exp=0.5, min_w=0.1,
                     # Drizzle
                     use_drizzle=False, drizzle_scale=2, drizzle_wht_threshold=0.7): # Ajouter le paramètre avec défaut
        """Démarre le thread de traitement avec configuration de pondération et dossiers initiaux."""
        if self.processing_active: self.update_progress("⚠️ Traitement déjà en cours."); return False
        self.stop_processing = False; self.current_folder = os.path.abspath(input_dir)
        # ---> AJOUTER CETTE LIGNE <---
        print(f"DEBUG [start_processing]: self.current_folder = '{self.current_folder}'")
        # ------------------------------
        # *** Initialize vide self.additional_folders ***
        if not self.initialize(output_dir): self.processing_active = False; return False

        if self.batch_size < 3: self.update_progress(f"⚠️ Taille de lot ({self.batch_size}) trop petite, ajustée à 3.", None); self.batch_size = 3
        self.update_progress(f"ⓘ Taille de lot utilisée : {self.batch_size}")

        # --- Apply Weighting Config ---
        self.use_quality_weighting = use_weighting; self.weight_by_snr = weight_snr; self.weight_by_stars = weight_stars
        self.snr_exponent = max(0.1, snr_exp); self.stars_exponent = max(0.1, stars_exp); self.min_weight = max(0.01, min(1.0, min_w))

        # New Apply Drizzle Config ---
        self.use_drizzle_processing = use_drizzle
        self.drizzle_scale = float(drizzle_scale) # Assurer float
        self.drizzle_pixfrac = float(drizzle_wht_threshold) # Si configurable plus tard
        if self.use_drizzle_processing:
            self.update_progress(f"💧 Drizzle activé (Échelle: x{self.drizzle_scale})")
            
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
        # ---> AJOUTER CETTE LIGNE <---
        print(f"DEBUG [start_processing]: initial_files_added = {initial_files_added}")
        # ------------------------------
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