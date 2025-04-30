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
# --- NOUVELLE LIGNE D'IMPORT ---
from astropy.wcs import WCS, FITSFixedWarning # Importer WCS (et peut-être le warning)
import warnings # Si FITSFixedWarning est importé, il faut aussi warnings

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
try:
    from drizzle.resample import Drizzle # Importer la CLASSE Drizzle
    _OO_DRIZZLE_AVAILABLE = True
    print("DEBUG QueueManager: Imported drizzle.resample.Drizzle")
except ImportError as e_driz_cls:
    print(f"ERROR QueueManager: Cannot import drizzle.resample.Drizzle class: {e_driz_cls}")
    _OO_DRIZZLE_AVAILABLE = False
    Drizzle = None # Définir comme None si indisponible
# --- FIN MODIFICATION ---
from ..core.hot_pixels import detect_and_correct_hot_pixels
from ..core.alignment import SeestarAligner
from ..core.utils import estimate_batch_size
# Correction ici: Utiliser le chemin complet depuis la racine 'seestar'
from ..enhancement.stack_enhancement import StackEnhancer
from ..enhancement.drizzle_integration import _create_wcs_from_header # Importer la fonction helper
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
        ### NOUVEAU : Chemin pour les entrées temporaires Drizzle (sera défini dans initialize) ###
        self.drizzle_temp_dir = None
        ### NOUVEAU : Flag pour savoir si Drizzle est actif pour cette session (sera défini dans start_processing) ###
        self.drizzle_active_session = False
        self.reference_wcs_object = None # Stockera l'objet WCS validé
        # --- Reference Image & Alignment ---
        self.aligner = SeestarAligner()
        ### NOUVEAU : Pour stocker le header de référence pour le WCS Drizzle (sera défini dans _worker) ###
        self.reference_header_for_wcs = None
        # --- Batch & Cumulative Stack (pour l'aperçu classique) ---
        self.current_batch_data = [] # Stocke (aligned_data, header, quality_scores)
        self.current_stack_data = None # Stack CLASSIQUE cumulé pour l'aperçu
        self.current_stack_header = None
        self.images_in_cumulative_stack = 0 # Pour le stack CLASSIQUE
        self.total_exposure_seconds = 0.0 # Pour le stack CLASSIQUE
        self.final_stacked_path = None # Chemin du fichier FITS final (classique OU drizzle)
        # --- Processing Parameters (Valeurs par défaut, seront écrasées par start_processing) ---
        self.stacking_mode = "kappa-sigma"
        self.kappa = 2.5
        self.batch_size = 10 # Valeur par défaut si non fournie
        self.correct_hot_pixels = True
        self.hot_pixel_threshold = 3.0
        self.neighborhood_size = 5
        self.bayer_pattern = "GRBG"
        self.perform_cleanup = True
        self.use_drizzle_processing = False # Valeur par défaut initiale
        self.drizzle_scale = 2.0
        self.drizzle_wht_threshold = 0.7
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
        self.stacked_batches_count = 0 # Nombre de batchs classiques traités
        self.total_batches_estimated = 0
        self.failed_align_count = 0
        self.failed_stack_count = 0 # Échecs DANS _stack_batch (ccdproc)
        self.skipped_files_count = 0
        # --- NOUVEAUX ATTRIBUTS pour Drizzle Modes & Incrémental ---
        self.drizzle_mode = "Final"                 # Stocke le mode choisi ('Final', 'Incremental')
        self.cumulative_drizzle_data = None         # Image Drizzle cumulative (mode Incrémental)
        self.cumulative_drizzle_wht = None          # Carte de poids cumulative (mode Incrémental)
        self.drizzle_kernel = "square"              # Noyau Drizzle choisi
        self.drizzle_pixfrac = 1.0                  # Pixfrac Drizzle choisi
        # --- Assurer que l'aligneur a aussi un callback progress (au cas où) ---
        if self.progress_callback:
             self.aligner.set_progress_callback(self.progress_callback)
######################################################################################################################################################



    def initialize(self, output_dir):
        """Prépare les dossiers et réinitialise l'état complet avant un nouveau traitement."""
        try:
            self.output_folder = os.path.abspath(output_dir)
            self.unaligned_folder = os.path.join(self.output_folder, "unaligned_files")
            self.drizzle_temp_dir = os.path.join(self.output_folder, "drizzle_temp_inputs") # Pour aligned_input_xxx.fits
            # --- NOUVEAU : Définir le chemin pour les sorties de batch Drizzle ---
            self.drizzle_batch_output_dir = os.path.join(self.output_folder, "drizzle_batch_outputs")
            # --- FIN NOUVEAU ---

            os.makedirs(self.output_folder, exist_ok=True)
            os.makedirs(self.unaligned_folder, exist_ok=True)

            # Gérer le dossier temporaire Drizzle (aligned_inputs)
            if self.perform_cleanup and os.path.isdir(self.drizzle_temp_dir):
                try: shutil.rmtree(self.drizzle_temp_dir); self.update_progress(f"🧹 Ancien dossier temp Drizzle nettoyé.")
                except Exception as e: self.update_progress(f"⚠️ Erreur nettoyage ancien dossier temp Drizzle: {e}")
            os.makedirs(self.drizzle_temp_dir, exist_ok=True)

            # --- NOUVEAU : Gérer le dossier des sorties de batch ---
            if self.perform_cleanup and os.path.isdir(self.drizzle_batch_output_dir):
                try: shutil.rmtree(self.drizzle_batch_output_dir); self.update_progress(f"🧹 Ancien dossier sorties batch Drizzle nettoyé.")
                except Exception as e: self.update_progress(f"⚠️ Erreur nettoyage ancien dossier sorties batch Drizzle: {e}")
            os.makedirs(self.drizzle_batch_output_dir, exist_ok=True) # Créer s'il n'existe pas
            # --- FIN NOUVEAU ---

            # --- CORRIGÉ : Message de log mis à jour ---
            self.update_progress(
                f"🗄️ Dossiers prêts: Sortie='{os.path.basename(self.output_folder)}', "
                f"NonAlign='{os.path.basename(self.unaligned_folder)}', "
                f"TempInput='{os.path.basename(self.drizzle_temp_dir)}', "
                f"BatchOut='{os.path.basename(self.drizzle_batch_output_dir)}'" # Utilise le nouveau nom
            )
            # --- FIN CORRIGÉ ---

        except OSError as e:
            self.update_progress(f"❌ Erreur critique création dossiers: {e}", 0)
            return False

        # --- Réinitialisations (Ajouter les nouvelles variables ici) ---
        self.reference_wcs_object = None
        # self.all_aligned_temp_files = [] # Supprimé à l'étape 1
        self.intermediate_drizzle_batch_files = [] # Nouvelle liste pour les chemins des lots intermédiaires
        self.drizzle_output_wcs = None             # WCS de sortie Drizzle
        self.drizzle_output_shape_hw = None        # Shape de sortie Drizzle
        self.cumulative_drizzle_data = None
        self.cumulative_drizzle_wht = None
        self.drizzle_kernel = "square"
        self.drizzle_pixfrac = 1.0
        # ... (autres resets existants) ...
        self.processed_files.clear()
        with self.folders_lock: self.additional_folders = []
        self.current_batch_data = []
        self.current_stack_data = None; self.current_stack_header = None; self.images_in_cumulative_stack = 0
        self.total_exposure_seconds = 0.0; self.final_stacked_path = None; self.processing_error = None
        self.files_in_queue = 0; self.processed_files_count = 0; self.aligned_files_count = 0
        self.stacked_batches_count = 0; self.total_batches_estimated = 0
        self.failed_align_count = 0; self.failed_stack_count = 0; self.skipped_files_count = 0
        self.drizzle_active_session = False
        self.reference_header_for_wcs = None # Assurer reset

        # Vider la queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except Empty:
                break
            except Exception:
                break # Sécurité

        # Reset aligner
        self.aligner.stop_processing = False
        return True


########################################################################################################################################################



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

###########################################################################################################################################################



    def _update_preview_incremental_drizzle(self):
        """
        Met à jour l'aperçu spécifiquement pour le mode Drizzle Incrémental.
        Envoie les données drizzlées cumulatives et le header mis à jour.
        """
        if self.preview_callback is None or self.cumulative_drizzle_data is None:
            # Ne rien faire si pas de callback ou pas de données drizzle cumulatives
            return

        try:
            # Utiliser les données et le header cumulatifs Drizzle
            data_to_send = self.cumulative_drizzle_data.copy()
            header_to_send = self.current_stack_header.copy() if self.current_stack_header else fits.Header()

            # Informations pour l'affichage dans l'aperçu
            img_count = self.images_in_cumulative_stack # Compteur mis à jour dans _process_incremental_drizzle_batch
            total_imgs_est = self.files_in_queue       # Estimation globale
            current_batch = self.stacked_batches_count # Le lot qui vient d'être traité
            total_batches_est = self.total_batches_estimated

            # Créer un nom pour l'aperçu
            stack_name = f"Drizzle Incr ({img_count}/{total_imgs_est} Img | Lot {current_batch}/{total_batches_est if total_batches_est > 0 else '?'})"

            # Appeler le callback du GUI
            self.preview_callback(
                data_to_send,
                header_to_send,
                stack_name,
                img_count,
                total_imgs_est,
                current_batch,
                total_batches_est
            )
            # print(f"DEBUG: Preview updated with Incremental Drizzle data (Shape: {data_to_send.shape})") # Optionnel

        except AttributeError:
             # Cas où cumulative_drizzle_data ou current_stack_header pourrait être None entre-temps
             print("Warning: Attribut manquant pour l'aperçu Drizzle incrémental.")
        except Exception as e:
            print(f"Error in _update_preview_incremental_drizzle: {e}")
            traceback.print_exc(limit=2)
#########################################################################################################################################################



    def _create_drizzle_output_wcs(self, ref_wcs, ref_shape_2d, scale_factor):
        """
        Crée le WCS et la shape (H,W) pour l'image Drizzle de sortie.
        Adapté de full_drizzle.py.

        Args:
            ref_wcs (astropy.wcs.WCS): Objet WCS de référence (validé, avec pixel_shape).
            ref_shape_2d (tuple): Shape (H, W) de l'image de référence.
            scale_factor (float): Facteur d'échelle Drizzle.

        Returns:
            tuple: (output_wcs, output_shape_2d_hw) ou lève une erreur.
                   output_shape_2d_hw est au format (H, W).
        """
        if not ref_wcs or not ref_wcs.is_celestial:
            raise ValueError("Référence WCS invalide ou non céleste pour Drizzle.")
        if ref_wcs.pixel_shape is None:
            raise ValueError("Référence WCS n'a pas de pixel_shape défini.")
        if len(ref_shape_2d) != 2:
             raise ValueError(f"Référence shape 2D (H,W) attendue, reçu {ref_shape_2d}")

        h_in, w_in = ref_shape_2d
        # Utiliser round() pour obtenir des dimensions entières plus proches
        out_h = int(round(h_in * scale_factor))
        out_w = int(round(w_in * scale_factor))
        # Assurer des dimensions minimales
        out_h = max(1, out_h); out_w = max(1, out_w)
        out_shape_2d_hw = (out_h, out_w) # Ordre (H, W) pour NumPy

        # Copier le WCS d'entrée et ajuster
        out_wcs = ref_wcs.deepcopy()

        # Ajuster échelle via CDELT ou CD matrix
        scale_adjusted = False
        try:
            # Prioriser la matrice CD si elle existe et est valide
            if hasattr(out_wcs.wcs, 'cd') and out_wcs.wcs.cd is not None and np.any(out_wcs.wcs.cd):
                # print("   DEBUG WCS Out: Adjusting scale via CD matrix.") # Debug
                # Division simple de la matrice par le facteur d'échelle
                out_wcs.wcs.cd = ref_wcs.wcs.cd / scale_factor
                scale_adjusted = True
            # Sinon, utiliser CDELT (et s'assurer que PC existe)
            elif hasattr(out_wcs.wcs, 'cdelt') and out_wcs.wcs.cdelt is not None and np.any(out_wcs.wcs.cdelt):
                # print("   DEBUG WCS Out: Adjusting scale via CDELT vector.") # Debug
                out_wcs.wcs.cdelt = ref_wcs.wcs.cdelt / scale_factor
                # S'assurer que la matrice PC existe (même si identité)
                if not hasattr(out_wcs.wcs, 'pc') or out_wcs.wcs.pc is None:
                     out_wcs.wcs.pc = np.identity(2)
                     # print("   DEBUG WCS Out: Ensuring PC matrix is identity.") # Debug
                elif not np.allclose(out_wcs.wcs.pc, np.identity(2)):
                     print("     - Warning WCS Out: PC matrix exists and is not identity.") # Garder cet avertissement
                scale_adjusted = True
            else:
                raise ValueError("Input WCS lacks valid CD matrix and CDELT vector.")
        except Exception as e:
            raise ValueError(f"Failed to adjust pixel scale in output WCS: {e}")

        if not scale_adjusted: # Double vérification
             raise ValueError("Could not adjust WCS scale.")

        # Centrer CRPIX sur la nouvelle image de sortie
        # Le centre pixel est (N/2 + 0.5) en convention FITS 1-based index
        # Pour WCS Astropy (0-based), le centre est (N-1)/2.
        # Cependant, crpix est 1-based. Donc on utilise N/2 + 0.5
        new_crpix_x = out_w / 2.0 + 0.5
        new_crpix_y = out_h / 2.0 + 0.5
        out_wcs.wcs.crpix = [new_crpix_x, new_crpix_y]

        # Définir la taille pixel de sortie pour Astropy (W, H)
        out_wcs.pixel_shape = (out_w, out_h)
        # Mettre à jour aussi les attributs NAXIS internes si possible (bonne pratique)
        try:
            out_wcs._naxis1 = out_w
            out_wcs._naxis2 = out_h
        except AttributeError:
            pass # Ignorer si les attributs n'existent pas (versions WCS plus anciennes?)

        print(f"   - Output WCS créé: Shape={out_shape_2d_hw} (H,W), CRPIX={out_wcs.wcs.crpix}")
        return out_wcs, out_shape_2d_hw # Retourne WCS et shape (H, W)




###########################################################################################################################################################

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
##################################################################################################################



    def _worker(self):
        """Thread principal pour le traitement des images."""
        self.processing_active = True
        self.processing_error = None
        start_time_session = time.monotonic()
        reference_image_data = None
        reference_header = None
        self.reference_wcs_object = None # Sera l'objet WCS validé avec pixel_shape
        self.reference_header_for_wcs = None # Header brut de référence

        # Initialiser/Réinitialiser les listes selon le mode
        current_batch_temp_files = [] # Pour Drizzle Incremental
        self.current_batch_data = [] # Pour Classique
        # Liste pour les chemins des fichiers intermédiaires Drizzle Final
        self.intermediate_drizzle_batch_files = []

        print(f"DEBUG [WORKER START]: Mode: {self.drizzle_mode if self.drizzle_active_session else 'Classic'}, DrizzleActive: {self.drizzle_active_session}")

        try:
            # --- 1. Préparation de l'image de référence et WCS ---
            self.update_progress("⭐ Recherche/Préparation image référence...")
            if not self.current_folder or not os.path.isdir(self.current_folder):
                raise RuntimeError(f"Dossier d'entrée initial invalide ou non défini: {self.current_folder}")
            initial_files = []
            try:
                initial_files = sorted([f for f in os.listdir(self.current_folder) if f.lower().endswith(('.fit', '.fits'))])
                if not initial_files: raise RuntimeError(f"Aucun fichier FITS trouvé dans le dossier initial: {self.current_folder}")
                print(f"DEBUG [_worker]: Trouvé {len(initial_files)} fichiers FITS initiaux pour référence.")
            except Exception as e: raise RuntimeError(f"Erreur lors du listage des fichiers initiaux: {e}")

            # Configuration Aligneur
            self.aligner.correct_hot_pixels = self.correct_hot_pixels; self.aligner.hot_pixel_threshold = self.hot_pixel_threshold
            self.aligner.neighborhood_size = self.neighborhood_size; self.aligner.bayer_pattern = self.bayer_pattern

            # Obtenir l'image de référence et son header
            reference_image_data, reference_header = self.aligner._get_reference_image(self.current_folder, initial_files)

            if reference_image_data is None or reference_header is None:
                user_ref_path = self.aligner.reference_image_path; error_msg = ""
                if user_ref_path and os.path.isfile(user_ref_path): error_msg = f"Échec chargement/prétraitement référence MANUELLE: {os.path.basename(user_ref_path)}"
                elif user_ref_path: error_msg = f"Fichier référence MANUELLE introuvable/invalide: {user_ref_path}"
                else: error_msg = "Échec sélection automatique image référence (vérifiez logs internes aligner)."
                raise RuntimeError(error_msg)
            else:
                # *** Validation WCS référence ***
                self.update_progress("   -> Validation/Génération WCS Référence...")
                self.reference_header_for_wcs = reference_header.copy() # Stocker le header brut
                local_ref_wcs_obj = None
                try: # Essayer lecture depuis header
                    with warnings.catch_warnings(): warnings.simplefilter('ignore', FITSFixedWarning); wcs_hdr = WCS(reference_header, naxis=2)
                    if wcs_hdr.is_celestial: local_ref_wcs_obj = wcs_hdr; print("      - WCS valide trouvé dans header référence.")
                except Exception: pass
                if local_ref_wcs_obj is None: # Essayer génération si lecture échoue
                    print("      - WCS Header invalide/absent, tentative génération...")
                    local_ref_wcs_obj = _create_wcs_from_header(reference_header) # Utilise la fonction importée
                    if local_ref_wcs_obj and local_ref_wcs_obj.is_celestial: print("      - WCS généré avec succès.")
                    else: print("      - Échec génération WCS."); local_ref_wcs_obj = None

                if local_ref_wcs_obj is None: # Si WCS toujours invalide
                    raise RuntimeError("Impossible d'obtenir un WCS céleste valide pour l'image de référence.")

                # *** Vérification cruciale : pixel_shape ***
                ref_naxis1 = reference_header.get('NAXIS1')
                ref_naxis2 = reference_header.get('NAXIS2')
                if ref_naxis1 and ref_naxis2:
                     local_ref_wcs_obj.pixel_shape = (ref_naxis1, ref_naxis2)
                     print(f"      - Pixel shape défini sur WCS Ref Obj: {local_ref_wcs_obj.pixel_shape}")
                     if local_ref_wcs_obj.pixel_shape is None:
                         raise RuntimeError("Association pixel_shape à l'objet WCS a échoué.")
                elif local_ref_wcs_obj is not None and local_ref_wcs_obj.pixel_shape is None:
                     print("     ----> WARNING: WCS de référence n'a toujours pas de pixel_shape après génération/association. Le mode Drizzle échouera probablement.")

                self.reference_wcs_object = local_ref_wcs_obj

                # Sauvegarder l'image référence pour inspection (ne change pas)
                self.aligner._save_reference_image(reference_image_data, reference_header, self.output_folder)
                self.update_progress("⭐ Image de référence et WCS prêts.", 5)

            # --- Calculer WCS/Shape sortie Drizzle UNE FOIS ---
            self.drizzle_output_wcs = None
            self.drizzle_output_shape_hw = None
            if self.drizzle_active_session: # Seulement si Drizzle est actif
                try:
                    self.update_progress("   -> Calcul WCS/Shape sortie Drizzle...")
                    ref_shape_hw = reference_image_data.shape[:2] # H, W
                    self.drizzle_output_wcs, self.drizzle_output_shape_hw = self._create_drizzle_output_wcs(
                        self.reference_wcs_object,
                        ref_shape_hw,
                        self.drizzle_scale
                    )
                    if self.drizzle_output_wcs is None or self.drizzle_output_shape_hw is None:
                        raise RuntimeError("Échec création WCS/Shape Drizzle sortie.")
                except Exception as wcs_out_err:
                     self.update_progress(f"❌ ERREUR CRITIQUE WCS Sortie Drizzle: {wcs_out_err}")
                     self.processing_error = f"Erreur WCS Sortie Drizzle: {wcs_out_err}"
                     # Important de sortir ici pour ne pas continuer sans WCS de sortie
                     raise wcs_out_err # Rethrow pour être attrapé par le finally global

            # --- 2. Boucle de traitement de la file ---
            self._recalculate_total_batches() # Estimer une première fois
            self.update_progress(f"▶️ Démarrage boucle traitement (File: {self.files_in_queue} | Lots Est.: {self.total_batches_estimated if self.total_batches_estimated > 0 else '?'})")

            # Liste pour accumuler les données du lot Drizzle Final
            current_drizzle_final_batch_data = []

            while not self.stop_processing:
                file_path = None
                try:
                    file_path = self.queue.get(timeout=1.0)
                    file_name = os.path.basename(file_path)
                    # Récupérer 4 valeurs de _process_file
                    aligned_data, header, quality_scores, wcs_object = self._process_file(file_path, reference_image_data)
                    self.processed_files_count += 1

                    if aligned_data is not None:
                        self.aligned_files_count += 1

                        # A. MODE DRIZZLE FINAL (NOUVELLE LOGIQUE - Étape 3)
                        if self.drizzle_active_session and self.drizzle_mode == "Final":
                            if wcs_object is None: # Sécurité
                                 self.update_progress(f"   ⚠️ WCS manquant pour image alignée {file_name}. Ignorée pour Drizzle Final.")
                                 self.skipped_files_count += 1
                            else:
                                 current_drizzle_final_batch_data.append((aligned_data, header, wcs_object)) # Stocker tuple
                                 # Vérifier si le lot est plein
                                 if len(current_drizzle_final_batch_data) >= self.batch_size:
                                     self.stacked_batches_count += 1
                                     # Traiter et sauvegarder le lot Drizzle
                                     sci_path, wht_paths = self._process_and_save_drizzle_batch(
                                         current_drizzle_final_batch_data,
                                         self.drizzle_output_wcs,      # WCS sortie Drizzle
                                         self.drizzle_output_shape_hw, # Shape sortie Drizzle
                                         self.stacked_batches_count
                                     )
                                     # Ajouter les chemins si réussi
                                     if sci_path and wht_paths:
                                         self.intermediate_drizzle_batch_files.append((sci_path, wht_paths))
                                     else:
                                         self.failed_stack_count += len(current_drizzle_final_batch_data)
                                     # Vider le lot traité
                                     current_drizzle_final_batch_data = []
                                     gc.collect()
                            # Nettoyer les données individuelles car traitées ou ignorées
                            try: del aligned_data, header, quality_scores, wcs_object; gc.collect()
                            except NameError: pass

                        # B. MODE DRIZZLE INCREMENTAL (INCHANGÉ)
                        elif self.drizzle_active_session and self.drizzle_mode == "Incremental":
                             temp_filepath = self._save_drizzle_input_temp(aligned_data, header) # wcs_object non utilisé
                             if temp_filepath is not None:
                                 current_batch_temp_files.append(temp_filepath)
                                 del aligned_data, header, quality_scores, wcs_object; gc.collect()
                                 if len(current_batch_temp_files) >= self.batch_size:
                                     self.stacked_batches_count += 1
                                     self._process_incremental_drizzle_batch(current_batch_temp_files, self.stacked_batches_count, self.total_batches_estimated)
                                     current_batch_temp_files = []
                             else:
                                 self.update_progress(f"   ⚠️ Échec sauvegarde temp Drizzle (Incr). Fichier {file_name} ignoré.")
                                 self.skipped_files_count += 1
                                 try: del aligned_data, header, quality_scores, wcs_object; gc.collect()
                                 except NameError: pass

                        # C. MODE CLASSIQUE (INCHANGÉ)
                        elif not self.drizzle_active_session:
                            # wcs_object non utilisé
                            self.current_batch_data.append((aligned_data, header, quality_scores))
                            if len(self.current_batch_data) >= self.batch_size:
                                self.stacked_batches_count += 1
                                self._process_completed_batch(self.stacked_batches_count, self.total_batches_estimated)
                                self.current_batch_data = []

                    self.queue.task_done()

                    # Mise à jour Progression/ETA (Identique)
                    current_progress = (self.processed_files_count / self.files_in_queue) * 100 if self.files_in_queue > 0 else 0
                    elapsed_time_session = time.monotonic() - start_time_session
                    if self.processed_files_count > 0:
                        time_per_file = elapsed_time_session / self.processed_files_count
                        remaining_files_estimate = max(0, self.files_in_queue - self.processed_files_count)
                        eta_seconds = remaining_files_estimate * time_per_file
                        h, rem = divmod(int(eta_seconds), 3600); m, s = divmod(rem, 60)
                        time_str = f"{h:02}:{m:02}:{s:02}"
                        progress_msg = f"📊 ({self.processed_files_count}/{self.files_in_queue}) {file_name} | ETA: {time_str}"
                    else: progress_msg = f"📊 ({self.processed_files_count}/{self.files_in_queue}) {file_name}"
                    self.update_progress(progress_msg, current_progress)

                    if self.processed_files_count % 20 == 0: gc.collect()

                except Empty:
                    self.update_progress("ⓘ File vide. Vérification batch final / dossiers sup...")
                    # Traiter dernier lot partiel (selon le mode)
                    if self.drizzle_active_session and self.drizzle_mode == "Final" and current_drizzle_final_batch_data:
                        self.update_progress(f"⏳ Traitement dernier batch Drizzle Final ({len(current_drizzle_final_batch_data)} images)...")
                        self.stacked_batches_count += 1
                        sci_path, wht_paths = self._process_and_save_drizzle_batch(
                            current_drizzle_final_batch_data,
                            self.drizzle_output_wcs,
                            self.drizzle_output_shape_hw,
                            self.stacked_batches_count
                        )
                        if sci_path and wht_paths:
                            self.intermediate_drizzle_batch_files.append((sci_path, wht_paths))
                        else:
                            self.failed_stack_count += len(current_drizzle_final_batch_data)
                        current_drizzle_final_batch_data = []
                        gc.collect()
                    elif not self.drizzle_active_session and self.current_batch_data:
                        self.update_progress(f"⏳ Traitement dernier batch classique ({len(self.current_batch_data)} images)...")
                        self.stacked_batches_count += 1
                        self._process_completed_batch(self.stacked_batches_count, self.total_batches_estimated)
                        self.current_batch_data = []
                    elif self.drizzle_active_session and self.drizzle_mode == "Incremental" and current_batch_temp_files:
                        self.update_progress(f"⏳ Traitement dernier batch Drizzle Incr. ({len(current_batch_temp_files)} fichiers)...")
                        self.stacked_batches_count += 1
                        self._process_incremental_drizzle_batch(current_batch_temp_files, self.stacked_batches_count, self.total_batches_estimated)
                        current_batch_temp_files = []

                    # Traiter dossier supplémentaire (logique inchangée)
                    folder_to_process = None
                    with self.folders_lock:
                        if self.additional_folders:
                            folder_to_process = self.additional_folders.pop(0); folder_count = len(self.additional_folders)
                            self.update_progress(f"folder_count_update:{folder_count}")
                    if folder_to_process:
                        folder_name = os.path.basename(folder_to_process); self.update_progress(f"📂 Traitement dossier sup: {folder_name}")
                        self.current_folder = folder_to_process
                        # Vider les accumulateurs de lots
                        current_batch_temp_files = []
                        current_drizzle_final_batch_data = [] # Vider aussi celui-là
                        if not self.drizzle_active_session: self.current_batch_data = []
                        # Ajouter fichiers
                        files_added = self._add_files_to_queue(folder_to_process)
                        if files_added > 0:
                            self._recalculate_total_batches() # Recalculer total après ajout
                            self.update_progress(f"📋 {files_added} fichiers ajoutés depuis {folder_name}. File: {self.files_in_queue}. Lots Est.: {self.total_batches_estimated if self.total_batches_estimated > 0 else '?'}")
                            continue
                        else: self.update_progress(f"⚠️ Aucun nouveau FITS trouvé dans {folder_name}"); continue
                    else:
                        self.update_progress("✅ Fin de la file et des dossiers supplémentaires.")
                        break # Sortir boucle while

                except Exception as e: # Gestion Erreur Générale Fichier (inchangée)
                    error_context = f" de {file_name}" if file_path else " (file inconnu)"
                    self.update_progress(f"❌ Erreur boucle worker{error_context}: {e}")
                    traceback.print_exc(limit=3); self.processing_error = f"Erreur boucle worker: {e}"
                    if file_path: self.skipped_files_count += 1
                    try: self.queue.task_done()
                    except ValueError: pass
                    time.sleep(0.1)
            # --- FIN DE LA BOUCLE WHILE ---

            # --- 3. Étape Finale (après la boucle) ---
            if self.stop_processing:
                # Cas Arrêt Utilisateur (Logique inchangée, inclut arrêt Drizzle Final)
                self.update_progress("🛑 Traitement interrompu avant étape finale.")
                if self.drizzle_mode == "Incremental" and self.cumulative_drizzle_data is not None:
                    self.update_progress("💾 Sauvegarde Drizzle incrémental intermédiaire (arrêt)...")
                    self._save_final_stack(output_filename_suffix="_drizzle_incr_stopped", stopped_early=True)
                elif not self.drizzle_active_session and self.current_stack_data is not None: # Arrêt mode Classique
                    self.update_progress("💾 Sauvegarde stack classique intermédiaire (arrêt)...")
                    self._save_final_stack(output_filename_suffix="_classic_stopped", stopped_early=True)
                elif self.drizzle_mode == "Final" and self.intermediate_drizzle_batch_files:
                     self.update_progress("ⓘ Traitement Drizzle Final interrompu. Les lots intermédiaires peuvent être conservés si le nettoyage est désactivé.")

            else: # Si Traitement Normal Terminé

                # --- MODIFIÉ : Logique pour Drizzle Final (Étape 5a) ---
                if self.drizzle_active_session and self.drizzle_mode == "Final":
                    if self.intermediate_drizzle_batch_files:
                        # --- Appel combinaison finale (Étape 4) ---
                        self.update_progress(f"💧 Combinaison finale des {len(self.intermediate_drizzle_batch_files)} lots Drizzle...")
                        # Utiliser le WCS/Shape sortie Drizzle calculés au début
                        final_combined_sci, final_combined_wht = self._combine_intermediate_drizzle_batches(
                            self.intermediate_drizzle_batch_files,
                            self.drizzle_output_wcs,
                            self.drizzle_output_shape_hw
                        )
                        if final_combined_sci is not None:
                            self.update_progress("   -> Assignation résultat Drizzle Final...")
                            # Le résultat est déjà HxWxC float32
                            self.current_stack_data = final_combined_sci
                            # Créer/Mettre à jour le header final
                            self.current_stack_header = fits.Header() # Partir de zéro
                            if self.drizzle_output_wcs:
                                try: self.current_stack_header.update(self.drizzle_output_wcs.to_header(relax=True))
                                except Exception as wcs_hdr_err: print(f"Warn: Failed WCS update from Drizzle output: {wcs_hdr_err}")
                            # Copier métadonnées originales non-WCS
                            if self.reference_header_for_wcs:
                                keys_to_copy = ['INSTRUME','TELESCOP','OBJECT','FILTER','DATE-OBS','GAIN','OFFSET','CCD-TEMP', 'SITELAT', 'SITELONG', 'FOCALLEN', 'APERTURE']
                                for key in keys_to_copy:
                                     if key in self.reference_header_for_wcs:
                                         try: self.current_stack_header[key] = (self.reference_header_for_wcs[key], self.reference_header_for_wcs.comments[key] if key in self.reference_header_for_wcs.comments else '')
                                         except Exception: self.current_stack_header[key] = self.reference_header_for_wcs[key]
                            # Ajouter infos Drizzle spécifiques
                            self.current_stack_header['STACKTYP'] = (f'Drizzle Final ({self.drizzle_scale:.1f}x)', 'Final Drizzle Combination')
                            self.current_stack_header['DRZSCALE'] = (self.drizzle_scale, 'Scale factor used'); self.current_stack_header['DRZKERNEL'] = (self.drizzle_kernel, 'Drizzle kernel'); self.current_stack_header['DRZPIXFR'] = (self.drizzle_pixfrac, 'Drizzle pixfrac')
                            # Mettre à jour total exposure (peut être approximatif)
                            self.total_exposure_seconds = 0.0
                            if self.intermediate_drizzle_batch_files:
                                try:
                                    # Utiliser self.aligned_files_count comme meilleure estimation du total
                                    if self.aligned_files_count > 0 and self.reference_header_for_wcs:
                                         single_exp = float(self.reference_header_for_wcs.get('EXPTIME', 10.0))
                                         self.total_exposure_seconds = self.aligned_files_count * single_exp
                                except Exception: pass # Ignorer si erreur lecture header

                            # Sauvegarder le résultat Drizzle Final (Étape 5b)
                            self._save_final_stack(output_filename_suffix="_drizzle_final_v2") # Nouveau suffixe
                        else:
                            self.processing_error = "Échec combinaison finale Drizzle"; self.final_stacked_path = None
                        # Nettoyer la carte de poids finale
                        del final_combined_wht; gc.collect()
                    else: # If no intermediate batch files for Drizzle Final
                        self.update_progress("⚠️ Drizzle final activé mais aucun lot intermédiaire créé/trouvé à combiner.")
                        self.final_stacked_path = None
                # --- FIN MODIFIÉ ---

                # --- CAS 2 : MODE DRIZZLE INCREMENTAL (Logique inchangée) ---
                elif self.drizzle_active_session and self.drizzle_mode == "Incremental":
                    if self.cumulative_drizzle_data is not None and self.images_in_cumulative_stack > 0:
                        self.update_progress("💾 Sauvegarde du stack Drizzle incrémental final...")
                        self._save_final_stack(output_filename_suffix="_drizzle_incr")
                    else: self.update_progress("ⓘ Aucun stack Drizzle incrémental à sauvegarder."); self.final_stacked_path = None

                # --- CAS 3 : MODE CLASSIQUE (Logique inchangée) ---
                elif not self.drizzle_active_session and self.current_stack_data is not None:
                    self.update_progress("💾 Sauvegarde du stack classique final...")
                    self._save_final_stack(output_filename_suffix="_classic")
                else: # Aucun stack créé (Inchangé)
                    self.update_progress("ⓘ Aucun stack final (classique ou Drizzle) à sauvegarder.")
                    self.final_stacked_path = None

        # --- Gestion Erreurs Critiques & Nettoyage Final ---
        except RuntimeError as critical_err:
             self.update_progress(f"❌ ERREUR CRITIQUE: {critical_err}")
             self.processing_error = str(critical_err)
             traceback.print_exc(limit=2)
        except Exception as e:
             self.update_progress(f"❌ Erreur critique thread worker: {e}")
             traceback.print_exc(limit=5); self.processing_error = f"Erreur critique: {e}"
        finally:
            # Nettoyage (MODIFIÉ Étape 5c)
            if self.perform_cleanup:
                self.update_progress("🧹 Nettoyage final fichiers temporaires...")
                self.cleanup_unaligned_files(); self.cleanup_temp_reference()
                self._cleanup_drizzle_temp_files() # Nettoie aligned_input_xxx.fits
                self._cleanup_drizzle_batch_outputs() # Nettoie les fichiers intermédiaires par lot
            else: self.update_progress(f"ⓘ Fichiers temporaires conservés.")
            # Vider queue et GC
            while not self.queue.empty():
                try: self.queue.get_nowait(); self.queue.task_done()
                except Exception: break
            self.current_batch_data = []; current_drizzle_final_batch_data = []
            self.intermediate_drizzle_batch_files = [] # Vider aussi celui-là
            current_batch_temp_files = []
            gc.collect()
            self.processing_active = False
            self.update_progress("🚪 Thread traitement terminé.")




##################################################################################################################

    def _cleanup_drizzle_batch_outputs(self):
        """Supprime le dossier contenant les fichiers Drizzle intermédiaires par lot."""
        batch_output_dir = os.path.join(self.output_folder, "drizzle_batch_outputs")
        if batch_output_dir and os.path.isdir(batch_output_dir):
            try:
                shutil.rmtree(batch_output_dir)
                self.update_progress(f"🧹 Dossier Drizzle intermédiaires par lot supprimé: {os.path.basename(batch_output_dir)}")
            except Exception as e:
                self.update_progress(f"⚠️ Erreur suppression dossier Drizzle intermédiaires ({os.path.basename(batch_output_dir)}): {e}")
        # else: # Log optionnel
            # print("DEBUG: Dossier Drizzle intermédiaires par lot non trouvé pour nettoyage.")



####################################################################################################################
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
############################################################################################################################




    def _process_file(self, file_path, reference_image_data):
        """
        Traite un seul fichier image : chargement, validation, pré-traitement,
        alignement sur l'image de référence, calcul des scores qualité, et retourne le WCS associé.

        Args:
            file_path (str): Chemin complet du fichier FITS à traiter.
            reference_image_data (np.ndarray): Données de l'image de référence (alignée/prête).

        Returns:
            tuple: (aligned_data, header, quality_scores, wcs_object)
                   - aligned_data (np.ndarray ou None): Données alignées (float32, 0-1) ou None si échec.
                   - header (fits.Header ou None): En-tête FITS de l'image originale ou None.
                   - quality_scores (dict): Dictionnaire {'snr': float, 'stars': float}.
                   - wcs_object (astropy.wcs.WCS ou None): Objet WCS associé à l'image alignée.
        """
        file_name = os.path.basename(file_path)
        quality_scores = {'snr': 0.0, 'stars': 0.0}
        self.update_progress(f"   Traitement Fichier: {file_name}")

        try:
            # 1. Charger et valider le fichier FITS
            img_data = load_and_validate_fits(file_path)
            if img_data is None:
                self.update_progress(f"   ⚠️ Échec chargement/validation.")
                self.skipped_files_count += 1
                # --- MODIFIÉ : Retourner 4 valeurs ---
                return None, None, quality_scores, None

            header = fits.getheader(file_path)

            # 2. Vérification initiale de la variance
            std_dev = np.std(img_data)
            variance_threshold = 0.0015
            if std_dev < variance_threshold:
                self.update_progress(f"   ⚠️ Image ignorée (faible variance: {std_dev:.4f}).")
                self.skipped_files_count += 1
                # --- MODIFIÉ : Retourner 4 valeurs ---
                return None, None, quality_scores, None

            # 3. Pré-traitement
            prepared_img = img_data
            if prepared_img.ndim == 2:
                bayer = header.get('BAYERPAT', self.bayer_pattern)
                if isinstance(bayer, str) and bayer.upper() in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                    try: prepared_img = debayer_image(prepared_img, bayer.upper())
                    except ValueError as de: self.update_progress(f"   ⚠️ Erreur debayer: {de}. Tentative N&B.")

            if self.correct_hot_pixels:
                try: prepared_img = detect_and_correct_hot_pixels(prepared_img, self.hot_pixel_threshold, self.neighborhood_size)
                except Exception as hp_err: self.update_progress(f"   ⚠️ Erreur correction px chauds: {hp_err}.")

            prepared_img = prepared_img.astype(np.float32)

            # 4. Alignement
            aligned_img, align_success = self.aligner._align_image(prepared_img, reference_image_data, file_name)

            if not align_success:
                self.failed_align_count += 1
                try:
                    if os.path.exists(file_path):
                        target_path = os.path.join(self.unaligned_folder, file_name)
                        shutil.move(os.path.normpath(file_path), os.path.normpath(target_path))
                        self.update_progress(f"   ➡️ Échec alignement. Déplacé vers non alignés.")
                    else: self.update_progress(f"   ⚠️ Échec alignement, original non trouvé pour déplacer.")
                except Exception as move_err: self.update_progress(f"   ⚠️ Erreur déplacement après échec alignement: {move_err}")
                # --- MODIFIÉ : Retourner 4 valeurs ---
                return None, None, quality_scores, None

            # --- NOUVEAU : Assigner l'objet WCS de référence ---
            # Puisque l'image est alignée sur la référence, elle partage son WCS.
            wcs_object = self.reference_wcs_object
            # --- FIN NOUVEAU ---

            # 5. Calcul des métriques de qualité
            if self.use_quality_weighting:
                self.update_progress(f"      Calcul Scores Qualité...")
                quality_scores = self._calculate_quality_metrics(aligned_img)

            # 6. Retourner les résultats (incluant l'objet WCS)
            # --- MODIFIÉ : Retourner 4 valeurs ---
            return aligned_img, header, quality_scores, wcs_object

        # --- Gestion globale des erreurs pour ce fichier ---
        except Exception as e:
            self.update_progress(f"❌ Erreur traitement fichier {file_name}: {e}")
            traceback.print_exc(limit=3)
            self.skipped_files_count += 1
            if os.path.exists(file_path):
                try:
                    target_path = os.path.join(self.unaligned_folder, f"error_{file_name}")
                    shutil.move(os.path.normpath(file_path), os.path.normpath(target_path))
                except Exception: pass
            # --- MODIFIÉ : Retourner 4 valeurs ---
            return None, None, quality_scores, None



#############################################################################################################################
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
##############################################################################################################################################


    def _process_incremental_drizzle_batch(self, batch_temp_filepaths, current_batch_num=0, total_batches_est=0):
        """
        Traite un batch pour le Drizzle Incrémental :
        1. Appelle DrizzleProcessor sur les fichiers temporaires du lot.
        2. Combine le résultat avec le Drizzle cumulatif.
        3. Nettoie les fichiers temporaires du lot.
        """
        if not batch_temp_filepaths:
            self.update_progress(f"⚠️ Tentative de traiter un batch Drizzle incrémental vide (Batch #{current_batch_num}).")
            return

        num_files_in_batch = len(batch_temp_filepaths)
        progress_info = f"(Lot {current_batch_num}/{total_batches_est if total_batches_est > 0 else '?'})"
        self.update_progress(f"💧 Traitement Drizzle incrémental du batch {progress_info} ({num_files_in_batch} fichiers)...")

        # 1. Appeler Drizzle sur le lot courant
        drizzle_result_batch = None
        wht_map_batch = None
        try:
            # Instancier DrizzleProcessor avec les bons paramètres de la session
            drizzle_proc = DrizzleProcessor(
                scale_factor=self.drizzle_scale,
                pixfrac=self.drizzle_pixfrac, # Utilise l'attribut stocké
                kernel=self.drizzle_kernel   # Utilise l'attribut stocké
            )
            # Appeler apply_drizzle avec la liste des chemins du lot
            drizzle_result_batch, wht_map_batch = drizzle_proc.apply_drizzle(batch_temp_filepaths)

            if drizzle_result_batch is None:
                 raise RuntimeError(f"Échec Drizzle sur le lot {progress_info}.")
            if wht_map_batch is None:
                 self.update_progress(f"   ⚠️ Carte WHT non retournée pour le lot {progress_info}, combinaison pondérée impossible.")
                 # Fallback: utiliser des poids uniformes pour ce lot? Ou ignorer le lot?
                 # Pour l'instant, on ignore le lot si WHT manque.
                 raise RuntimeError(f"Carte WHT manquante pour lot {progress_info}.")

            self.update_progress(f"   -> Drizzle lot {progress_info} terminé (Shape: {drizzle_result_batch.shape})")

        except Exception as e:
            self.update_progress(f"❌ Erreur Drizzle sur lot {progress_info}: {e}")
            traceback.print_exc(limit=2)
            # Nettoyer les fichiers temporaires de ce lot même en cas d'échec Drizzle
            self._cleanup_batch_temp_files(batch_temp_filepaths)
            # Compter comme échec pour les stats
            self.failed_stack_count += num_files_in_batch
            return # Ne pas tenter de combiner

        # 2. Combiner avec le résultat cumulatif
        try:
            self.update_progress(f"   -> Combinaison Drizzle lot {progress_info} avec cumulatif...")

            # S'assurer que les données sont en float32 pour la combinaison
            drizzle_result_batch = drizzle_result_batch.astype(np.float32)
            wht_map_batch = wht_map_batch.astype(np.float32)

            # Cas initial : premier lot traité
            if self.cumulative_drizzle_data is None:
                self.cumulative_drizzle_data = drizzle_result_batch
                self.cumulative_drizzle_wht = wht_map_batch
                # Initialiser aussi le header pour les infos cumulatives Drizzle
                self.current_stack_header = fits.Header()
                self.current_stack_header['STACKTYP'] = (f'Drizzle Incr ({self.drizzle_scale}x)', 'Incremental Drizzle')
                self.current_stack_header['DRZSCALE'] = (self.drizzle_scale, 'Drizzle scale factor')
                self.current_stack_header['CREATOR'] = ('SeestarStacker (Queued)', 'Processing Software')
                self.images_in_cumulative_stack = 0 # Sera mis à jour ci-dessous
                self.total_exposure_seconds = 0.0   # Sera mis à jour ci-dessous

            # Cas : combinaison avec le cumulatif existant
            else:
                # Vérifier compatibilité shapes
                if self.cumulative_drizzle_data.shape != drizzle_result_batch.shape:
                    self.update_progress(f"❌ Incompatibilité dims Drizzle: Cumul={self.cumulative_drizzle_data.shape}, Lot={drizzle_result_batch.shape}. Combinaison échouée.")
                    # Nettoyer les fichiers temporaires de ce lot
                    self._cleanup_batch_temp_files(batch_temp_filepaths)
                    self.failed_stack_count += num_files_in_batch # Compter comme échec
                    return

                # Pondération par les WHT maps
                current_cumul_wht = self.cumulative_drizzle_wht.astype(np.float32)
                total_wht = current_cumul_wht + wht_map_batch
                # Éviter division par zéro là où le poids total est nul
                epsilon = 1e-12
                safe_total_wht = np.maximum(total_wht, epsilon)

                # Calcul de la moyenne pondérée
                weighted_cumul = self.cumulative_drizzle_data * (current_cumul_wht / safe_total_wht)
                weighted_batch = drizzle_result_batch * (wht_map_batch / safe_total_wht)
                new_cumulative_data = weighted_cumul + weighted_batch

                # Mettre à jour les données et la WHT map cumulative
                self.cumulative_drizzle_data = new_cumulative_data.astype(np.float32)
                self.cumulative_drizzle_wht = total_wht.astype(np.float32)

            # Mettre à jour les compteurs globaux (même pour le premier lot)
            self.images_in_cumulative_stack += num_files_in_batch
            # Estimation de l'exposition ajoutée (peut être imprécis si EXPTIME varie)
            try:
                 first_hdr_batch = fits.getheader(batch_temp_filepaths[0])
                 exp_time_batch = float(first_hdr_batch.get('EXPTIME', 0.0))
                 self.total_exposure_seconds += num_files_in_batch * exp_time_batch
            except Exception: pass # Ignorer si lecture header échoue

            # Mettre à jour le header cumulatif
            if self.current_stack_header:
                self.current_stack_header['NIMAGES'] = (self.images_in_cumulative_stack, 'Approx images in incremental drizzle')
                self.current_stack_header['TOTEXP'] = (round(self.total_exposure_seconds, 2), '[s] Approx total exposure')

            self.update_progress(f"   -> Combinaison lot {progress_info} terminée.")

            # Mettre à jour l'aperçu avec le nouveau cumulatif Drizzle
            self._update_preview_incremental_drizzle() # Nouvelle méthode d'aperçu spécifique

        except Exception as e:
            self.update_progress(f"❌ Erreur combinaison Drizzle lot {progress_info}: {e}")
            traceback.print_exc(limit=2)
            # Compter comme échec
            self.failed_stack_count += num_files_in_batch

        # 3. Nettoyer les fichiers temporaires de ce lot (TOUJOURS, sauf si debug)
        if self.perform_cleanup: # Seulement si le nettoyage est activé
             self._cleanup_batch_temp_files(batch_temp_filepaths)
        else:
             self.update_progress(f"   -> Fichiers temporaires du lot {progress_info} conservés (nettoyage désactivé).")

#################################################################################################################################################



    def _combine_drizzle_chunks(self, chunk_sci_files, chunk_wht_files):
        """
        Combine les fichiers chunks Drizzle (science et poids) sauvegardés sur disque.
        Lit les fichiers et effectue une moyenne pondérée.

        Args:
            chunk_sci_files (list): Liste des chemins vers les fichiers FITS science des chunks.
            chunk_wht_files (list): Liste des chemins vers les fichiers FITS poids des chunks.

        Returns:
            tuple: (final_sci_image, final_wht_map) ou (None, None) si échec.
                   Les tableaux retournés sont au format HxWxC, float32.
        """
        if not chunk_sci_files or not chunk_wht_files or len(chunk_sci_files) != len(chunk_wht_files):
            self.update_progress("❌ Erreur interne: Listes de fichiers chunks invalides ou incohérentes.")
            return None, None
        num_chunks = len(chunk_sci_files)
        if num_chunks == 0: self.update_progress("ⓘ Aucun chunk Drizzle à combiner."); return None, None

        self.update_progress(f"⚙️ Combinaison finale de {num_chunks} chunks Drizzle...")
        start_time = time.time()

        numerator_sum = None; denominator_sum = None
        output_shape = None; output_header = None
        first_chunk_processed_successfully = False

        try:
            # --- Boucle sur les chunks pour lire et accumuler ---
            for i, (sci_path, wht_path) in enumerate(zip(chunk_sci_files, chunk_wht_files)):
                if self.stop_processing: self.update_progress("🛑 Arrêt demandé pendant combinaison chunks."); return None, None
                self.update_progress(f"   -> Lecture et accumulation chunk {i+1}/{num_chunks}...")
                sci_chunk, wht_chunk = None, None
                sci_chunk_cxhxw, wht_chunk_cxhxw = None, None

                try:
                    # Lire Science Chunk
                    with fits.open(sci_path, memmap=False) as hdul_sci:
                        if not hdul_sci or hdul_sci[0].data is None: raise IOError(f"Chunk science invalide: {sci_path}")
                        sci_chunk_cxhxw = hdul_sci[0].data
                        if sci_chunk_cxhxw.ndim != 3 or sci_chunk_cxhxw.shape[0] != 3: raise ValueError(f"Chunk science {sci_path} non CxHxW.")
                        sci_chunk = np.moveaxis(sci_chunk_cxhxw, 0, -1).astype(np.float32)
                        if i == 0: output_header = hdul_sci[0].header # Garder header

                    # Lire Poids Chunk
                    with fits.open(wht_path, memmap=False) as hdul_wht:
                        if not hdul_wht or hdul_wht[0].data is None: raise IOError(f"Chunk poids invalide: {wht_path}")
                        wht_chunk_cxhxw = hdul_wht[0].data
                        if wht_chunk_cxhxw.ndim != 3 or wht_chunk_cxhxw.shape[0] != 3: raise ValueError(f"Chunk poids {wht_path} non CxHxW.")
                        wht_chunk = np.moveaxis(wht_chunk_cxhxw, 0, -1).astype(np.float32)

                    # Initialiser les accumulateurs
                    if numerator_sum is None:
                        output_shape = sci_chunk.shape
                        if output_shape is None: raise ValueError("Shape du premier chunk est None.")
                        numerator_sum = np.zeros(output_shape, dtype=np.float64) # float64 pour somme
                        denominator_sum = np.zeros(output_shape, dtype=np.float64)
                        print(f"      - Initialisation accumulateurs (Shape: {output_shape})")
                        first_chunk_processed_successfully = True

                    # Vérifier Shapes
                    if sci_chunk.shape != output_shape or wht_chunk.shape != output_shape:
                        self.update_progress(f"      -> ⚠️ Shape incohérente chunk {i+1}. Ignoré.")
                        continue

                    # --- Accumulation ---
                    sci_chunk_clean = np.nan_to_num(sci_chunk, nan=0.0)
                    wht_chunk_clean = np.nan_to_num(wht_chunk, nan=0.0)
                    wht_chunk_clean = np.maximum(wht_chunk_clean, 0.0)
                    numerator_sum += sci_chunk_clean * wht_chunk_clean
                    denominator_sum += wht_chunk_clean
                    # --- Fin Accumulation ---

                except (FileNotFoundError, IOError, ValueError) as read_err:
                     self.update_progress(f"      -> ❌ ERREUR lecture/validation chunk {i+1}: {read_err}. Ignoré.")
                     if i == 0: first_chunk_processed_successfully = False; continue
                finally:
                     del sci_chunk, wht_chunk, sci_chunk_cxhxw, wht_chunk_cxhxw
                     if (i + 1) % 5 == 0: gc.collect()
            # --- Fin Boucle Chunks ---

            if not first_chunk_processed_successfully or numerator_sum is None:
                raise RuntimeError("Aucun chunk valide n'a pu être lu pour initialiser la combinaison.")

            # --- Calcul final ---
            self.update_progress("   -> Calcul de l'image finale combinée...")
            epsilon = 1e-12
            final_sci_combined = np.zeros_like(numerator_sum, dtype=np.float32)
            valid_mask = denominator_sum > epsilon
            with np.errstate(divide='ignore', invalid='ignore'):
                final_sci_combined[valid_mask] = (numerator_sum[valid_mask] / denominator_sum[valid_mask])
            final_sci_combined = np.nan_to_num(final_sci_combined, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            final_wht_combined = denominator_sum.astype(np.float32)
            # --- Fin Calcul ---

        except MemoryError: self.update_progress("❌ ERREUR MÉMOIRE pendant combinaison chunks."); traceback.print_exc(limit=1); return None, None
        except Exception as e: self.update_progress(f"❌ Erreur inattendue pendant combinaison chunks: {e}"); traceback.print_exc(limit=2); return None, None

        if final_sci_combined is None or final_wht_combined is None: self.update_progress("❌ Combinaison chunks n'a produit aucun résultat."); return None, None

        end_time = time.time()
        self.update_progress(f"✅ Combinaison chunks terminée en {end_time - start_time:.2f}s.")
        return final_sci_combined, final_wht_combined


###############################################################################################################################################
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

#########################################################################################################################################



    def _combine_intermediate_drizzle_batches(self, intermediate_files_list, output_wcs, output_shape_2d_hw):
        """
        Combine les résultats Drizzle intermédiaires (par lot) sauvegardés sur disque.
        Utilise la classe Drizzle pour la combinaison pondérée par les cartes de poids.
        Adapté de full_drizzle.py/combine_batches.

        Args:
            intermediate_files_list (list): Liste de tuples [(sci_path, [wht_r, wht_g, wht_b]), ...].
            output_wcs (astropy.wcs.WCS): WCS final pour l'image combinée.
            output_shape_2d_hw (tuple): Shape (H, W) finale pour l'image combinée.

        Returns:
            tuple: (final_sci_image_hxwxc, final_wht_map_hxwxc) ou (None, None) si échec.
                   Les tableaux retournés sont en float32.
        """
        num_batches_to_combine = len(intermediate_files_list)
        if num_batches_to_combine == 0:
            self.update_progress("ⓘ Aucun lot Drizzle intermédiaire à combiner.")
            return None, None

        self.update_progress(f"💧 Combinaison finale de {num_batches_to_combine} lots Drizzle intermédiaires...")
        combine_start_time = time.time()

        # --- Initialiser les objets Drizzle FINAUX ---
        num_output_channels = 3
        channel_names = ['R', 'G', 'B']
        final_drizzlers = []
        final_output_images = [] # Pour stocker le résultat final science (H, W) par canal
        final_output_weights = [] # Pour stocker le résultat final poids (H, W) par canal

        try:
            # Pré-allouer les tableaux numpy pour les résultats FINAUX
            self.update_progress(f"   -> Initialisation Drizzle final (Shape: {output_shape_2d_hw})...")
            for _ in range(num_output_channels):
                final_output_images.append(np.zeros(output_shape_2d_hw, dtype=np.float32))
                final_output_weights.append(np.zeros(output_shape_2d_hw, dtype=np.float32))

            # Initialiser les objets Drizzle finaux (on peut réutiliser les mêmes kernel/pixfrac que pour les lots)
            for i in range(num_output_channels):
                driz_ch = Drizzle(
                    kernel=self.drizzle_kernel,     # Utiliser les params de la session
                    fillval="0.0",
                    out_img=final_output_images[i], # Passer tableau science final
                    out_wht=final_output_weights[i] # Passer tableau poids final
                )
                final_drizzlers.append(driz_ch)
            self.update_progress(f"   -> Objets Drizzle finaux initialisés.")

        except Exception as init_err:
            self.update_progress(f"   - ERREUR: Échec init Drizzle final: {init_err}")
            traceback.print_exc(limit=1)
            return None, None

        # --- Boucle sur les fichiers intermédiaires par lot ---
        total_contributing_ninputs = 0 # Compteur total (lu depuis headers intermédiaires)
        batches_combined_count = 0

        for i, (sci_fpath, wht_fpaths) in enumerate(intermediate_files_list):
            if self.stop_processing: self.update_progress("🛑 Arrêt pendant combinaison lots Drizzle."); break
            self.update_progress(f"   -> Ajout lot intermédiaire {i+1}/{num_batches_to_combine}...")
            # print(f"      Sci: {os.path.basename(sci_fpath)}, Whts: {[os.path.basename(w) for w in wht_fpaths]}") # Debug

            if len(wht_fpaths) != num_output_channels:
                self.update_progress(f"      -> ERREUR: Nombre incorrect de cartes poids pour lot {i+1}. Ignoré.")
                continue

            sci_data_chw = None; intermed_wcs = None; wht_maps = None; sci_header = None; combine_pixmap = None
            try:
                # --- Lire Données Science (CxHxW) et Header du lot ---
                with fits.open(sci_fpath, memmap=False) as hdul_sci:
                    sci_data_chw = hdul_sci[0].data.astype(np.float32) # Assurer float32
                    sci_header = hdul_sci[0].header
                    # Lire NINPUTS de ce lot
                    try: total_contributing_ninputs += int(sci_header.get('NINPUTS', 0))
                    except (ValueError, TypeError): pass
                    # Lire WCS du lot intermédiaire
                    with warnings.catch_warnings(): warnings.simplefilter("ignore"); intermed_wcs = WCS(sci_header, naxis=2)
                    if not intermed_wcs.is_celestial: raise ValueError("WCS intermédiaire non céleste.")
                    if sci_data_chw.ndim != 3 or sci_data_chw.shape[0] != num_output_channels:
                        raise ValueError(f"Shape science intermédiaire invalide: {sci_data_chw.shape}")

                # --- Lire Cartes Poids (HxW par canal) ---
                wht_maps = []
                valid_weights = True
                for ch_idx, wht_fpath in enumerate(wht_fpaths):
                    try:
                        with fits.open(wht_fpath, memmap=False) as hdul_wht:
                            wht_map = hdul_wht[0].data.astype(np.float32) # Assurer float32
                        # Vérifier shape poids vs shape science HxW
                        if wht_map.shape != sci_data_chw.shape[1:]:
                            raise ValueError(f"Shape poids {wht_map.shape} != science HxW {sci_data_chw.shape[1:]}")
                        # Nettoyer poids
                        wht_map[~np.isfinite(wht_map)] = 0.0
                        wht_map[wht_map < 0] = 0.0 # Poids doivent être >= 0
                        wht_maps.append(wht_map)
                    except Exception as e:
                        self.update_progress(f"      -> ERREUR lecture poids {os.path.basename(wht_fpath)}: {e}. Lot ignoré.")
                        valid_weights = False; break
                if not valid_weights: continue # Passer au lot suivant

                # --- Calcul Pixmap (Intermédiaire -> Final) ---
                # print("      -> Calcul pixmap combinaison...") # Debug
                intermed_shape_hw = sci_data_chw.shape[1:] # H, W
                y_intermed, x_intermed = np.indices(intermed_shape_hw)
                try:
                    # Convertir pixels intermédiaires en coords monde
                    world_coords_intermed = intermed_wcs.all_pix2world(x_intermed.flatten(), y_intermed.flatten(), 0)
                    # Convertir coords monde en pixels finaux
                    x_final, y_final = output_wcs.all_world2pix(world_coords_intermed[0], world_coords_intermed[1], 0)
                    # Remodeler et créer le pixmap (H, W, 2)
                    combine_pixmap = np.dstack((x_final.reshape(intermed_shape_hw), y_final.reshape(intermed_shape_hw))).astype(np.float32)
                    # print(f"      -> Pixmap combinaison shape: {combine_pixmap.shape}") # Debug
                except Exception as combine_map_err:
                    self.update_progress(f"      -> ERREUR création pixmap combinaison: {combine_map_err}. Lot ignoré.")
                    continue

                # --- Ajout à Drizzle Final (avec weight_map) ---
                if combine_pixmap is not None:
                    # print("      -> Ajout à Drizzle final...") # Debug
                    for ch_index in range(num_output_channels):
                        channel_data_sci = sci_data_chw[ch_index, :, :] # Science 2D pour ce canal
                        channel_data_wht = wht_maps[ch_index]          # Poids 2D pour ce canal

                        # Nettoyer NaN/Inf science (normalement déjà fait, mais sécurité)
                        channel_data_sci[~np.isfinite(channel_data_sci)] = 0.0

                        # Appeler add_image avec weight_map
                        final_drizzlers[ch_index].add_image(
                            data=channel_data_sci,
                            pixmap=combine_pixmap,
                            weight_map=channel_data_wht, # Passer la carte de poids
                            exptime=1.0, # L'exposition est déjà dans les poids/science
                            pixfrac=self.drizzle_pixfrac, # Utiliser pixfrac pour combinaison aussi ? Oui.
                            in_units='cps' # Les données intermédiaires sont déjà en counts/s
                        )
                    batches_combined_count += 1
                    # print(f"      -> Lot {i+1} ajouté à la combinaison finale.") # Debug
                else:
                    self.update_progress(f"      -> Warning: Pixmap combinaison est None pour lot {i+1}. Ignoré.")

            except FileNotFoundError:
                self.update_progress(f"   - ERREUR: Fichier intermédiaire lot {i+1} non trouvé. Ignoré.")
                continue
            except (IOError, ValueError) as e:
                self.update_progress(f"   - ERREUR lecture/validation lot intermédiaire {i+1}: {e}. Ignoré.")
                continue
            except Exception as e:
                self.update_progress(f"   - ERREUR traitement lot intermédiaire {i+1}: {e}")
                traceback.print_exc(limit=1)
                continue
            finally:
                # Nettoyer mémoire pour ce lot
                del sci_data_chw, intermed_wcs, wht_maps, sci_header, combine_pixmap
                if (i + 1) % 5 == 0: gc.collect()
        # --- Fin boucle sur les lots intermédiaires ---

        combine_end_time = time.time()
        self.update_progress(f"💧 Combinaison finale Drizzle terminée ({batches_combined_count}/{num_batches_to_combine} lots combinés en {combine_end_time - combine_start_time:.2f}s).")

        if batches_combined_count == 0:
            self.update_progress("❌ Aucun lot Drizzle intermédiaire n'a pu être combiné.")
            del final_drizzlers, final_output_images, final_output_weights; gc.collect()
            return None, None

        # --- Récupérer et assembler les résultats finaux ---
        try:
            # Les résultats sont déjà dans final_output_images et final_output_weights
            # Assembler en HxWxC
            final_sci_image_hxwxc = np.stack(final_output_images, axis=-1).astype(np.float32)
            final_wht_map_hxwxc = np.stack(final_output_weights, axis=-1).astype(np.float32)

            # Nettoyer les résultats finaux (sécurité)
            final_sci_image_hxwxc[~np.isfinite(final_sci_image_hxwxc)] = 0.0
            final_wht_map_hxwxc[~np.isfinite(final_wht_map_hxwxc)] = 0.0
            final_wht_map_hxwxc[final_wht_map_hxwxc < 0] = 0.0

            self.update_progress(f"   -> Assemblage final Drizzle terminé (Shape Sci: {final_sci_image_hxwxc.shape}, Shape WHT: {final_wht_map_hxwxc.shape})")

            # Mettre à jour le compteur total d'images basé sur les headers intermédiaires
            # Note: Ceci est une approximation si des lots ont échoué
            self.images_in_cumulative_stack = total_contributing_ninputs
            print(f"DEBUG [_combine_intermediate_drizzle_batches]: images_in_cumulative_stack set to {self.images_in_cumulative_stack} from intermediate headers.")

        except Exception as e_final:
            self.update_progress(f"   - ERREUR pendant assemblage final Drizzle: {e_final}")
            traceback.print_exc(limit=1)
            del final_drizzlers, final_output_images, final_output_weights; gc.collect()
            return None, None

        del final_drizzlers, final_output_images, final_output_weights; gc.collect()
        return final_sci_image_hxwxc, final_wht_map_hxwxc


############################################################################################################################################


    def _save_final_stack(self, output_filename_suffix="", stopped_early=False):
        """
        Sauvegarde le stack final (classique OU drizzle) et sa prévisualisation.
        Utilise un suffixe pour différencier les types de sortie.
        Ajoute les statistiques finales au header juste avant la sauvegarde.
        """
        # --- CHOISIR LES BONNES DONNÉES À SAUVEGARDER ---
        data_to_save = None
        header_base = None
        image_count = 0
        is_drizzle_save = False
        stack_type_for_filename = "unknown" # Pour le nom de fichier

        # Déterminer si le résultat actuel est Drizzle basé sur le header ou le mode
        # On utilise le header car il est mis à jour spécifiquement pour Drizzle Final
        if self.current_stack_header and 'DRZSCALE' in self.current_stack_header:
             is_drizzle_save = True
             # Déterminer le type Drizzle pour le nom de fichier
             if self.drizzle_mode == "Incremental": stack_type_for_filename = f"drizzle_incr_{self.drizzle_scale:.1f}x"
             else: stack_type_for_filename = f"drizzle_final_{self.drizzle_scale:.1f}x" # Mode Final
             data_to_save = self.current_stack_data
             header_base = self.current_stack_header
             image_count = self.images_in_cumulative_stack # Utiliser le compteur mis à jour par Drizzle Final/Incr
             print(f"DEBUG [_save_final_stack]: Saving Drizzle data (Type: {stack_type_for_filename}).")
        elif self.current_stack_data is not None: # Cas Classique (ou fallback)
             is_drizzle_save = False
             stack_type_for_filename = self.stacking_mode # Utiliser la méthode classique
             data_to_save = self.current_stack_data
             header_base = self.current_stack_header
             image_count = self.images_in_cumulative_stack
             print("DEBUG [_save_final_stack]: Saving Classic/Fallback data.")
        # --- FIN CHOIX DONNÉES ---

        # --- Vérifications initiales (basées sur les données choisies) ---
        if data_to_save is None or self.output_folder is None:
            self.final_stacked_path = None
            self.update_progress("ⓘ Aucun stack final à sauvegarder (données manquantes ou dossier sortie invalide).")
            return
        if image_count <= 0 and not stopped_early:
             self.final_stacked_path = None
             self.update_progress("ⓘ Aucun stack final à sauvegarder (0 images combinées).")
             return

        # --- Construire le nom de fichier final ---
        base_name = "stack_final"
        # Ajouter suffixe pondération
        weight_suffix = "_wght" if self.use_quality_weighting else ""
        # Combiner avec le suffixe spécifique (arrêt ou normal)
        final_suffix = f"{weight_suffix}{output_filename_suffix}"
        # Construire le chemin complet
        self.final_stacked_path = os.path.join(
            self.output_folder,
            f"{base_name}_{stack_type_for_filename}{final_suffix}.fit"
        )
        preview_path = os.path.splitext(self.final_stacked_path)[0] + ".png"

        print(f"DEBUG [_save_final_stack]: Tentative sauvegarde vers: {self.final_stacked_path}")
        self.update_progress(f"💾 Préparation sauvegarde stack final: {os.path.basename(self.final_stacked_path)}...")

        try:
            # Préparer le header final (basé sur header_base)
            final_header = header_base.copy() if header_base else fits.Header()

            # --- AJOUT/MISE A JOUR DES STATS FINALES ICI ---
            aligned_cnt = self.aligned_files_count if hasattr(self, 'aligned_files_count') else 0
            fail_align = self.failed_align_count if hasattr(self, 'failed_align_count') else 0
            fail_stack = self.failed_stack_count if hasattr(self, 'failed_stack_count') else 0
            skipped = self.skipped_files_count if hasattr(self, 'skipped_files_count') else 0
            tot_exp = self.total_exposure_seconds if hasattr(self, 'total_exposure_seconds') else 0.0

            final_header['NIMAGES'] = (image_count, 'Images combined in final stack') # Utilise image_count
            final_header['TOTEXP'] = (round(tot_exp, 2), '[s] Approx total exposure time')
            final_header['ALIGNED'] = (aligned_cnt, 'Successfully aligned images')
            final_header['FAILALIGN'] = (fail_align, 'Failed alignments')
            final_header['FAILSTACK'] = (fail_stack, 'Files skipped due to stack/drizzle errors')
            final_header['SKIPPED'] = (skipped, 'Other skipped/error files')

            # Assurer que STACKTYP/DRZSCALE sont corrects
            if 'DRZSCALE' not in final_header: # Si ce n'est pas un header Drizzle
                final_header['STACKTYP'] = (self.stacking_mode, 'Stacking method')
                if self.stacking_mode in ["kappa-sigma", "winsorized-sigma"]:
                    final_header['KAPPA'] = (self.kappa, 'Kappa value for clipping')

            # Assurer que les infos de pondération sont présentes
            if 'WGHT_ON' not in final_header:
                 final_header['WGHT_ON'] = (self.use_quality_weighting, 'Quality weighting status')
                 if self.use_quality_weighting:
                     w_metrics = []
                     if self.weight_by_snr: w_metrics.append(f"SNR^{self.snr_exponent:.1f}")
                     if self.weight_by_stars: w_metrics.append(f"Stars^{self.stars_exponent:.1f}")
                     final_header['WGHT_MET'] = (",".join(w_metrics), 'Metrics used for weighting')

            # Nettoyer l'historique des sauvegardes intermédiaires
            try:
                if 'HISTORY' in final_header:
                    history_entries = list(final_header['HISTORY'])
                    filtered_history = [h for h in history_entries if 'Intermediate save' not in str(h) and 'Cumulative Stack Initialized' not in str(h)]
                    while 'HISTORY' in final_header: del final_header['HISTORY']
                    for entry in filtered_history: final_header.add_history(entry)
            except Exception: pass # Ignorer erreurs de nettoyage historique

            # Ajouter l'entrée finale d'historique
            history_msg = 'Final Stack Saved by Seestar Stacker (Queued)'
            if stopped_early: history_msg += ' - Stopped Early'
            final_header.add_history(history_msg)
            # --- FIN MISE A JOUR HEADER ---

            # Sauvegarder le fichier FITS (utilise data_to_save)
            save_fits_image(data_to_save, self.final_stacked_path, final_header, overwrite=True)
            print(f"DEBUG [_save_final_stack]: save_fits_image a priori réussi.")

            # Sauvegarder la prévisualisation PNG (utilise data_to_save)
            save_preview_image(data_to_save, preview_path, apply_stretch=True)
            print(f"DEBUG [_save_final_stack]: save_preview_image a priori réussi.")

            self.update_progress(f"✅ Stack final sauvegardé ({image_count} images)") # Utilise image_count

        except Exception as e:
            print(f"DEBUG [_save_final_stack]: ERREUR pendant la sauvegarde!")
            self.update_progress(f"⚠️ Erreur sauvegarde stack final: {e}")
            traceback.print_exc(limit=2)
            self.final_stacked_path = None # Assurer que le chemin est None en cas d'erreur





#########################################################################################################################################




    def _cleanup_batch_temp_files(self, batch_filepaths):
        """Supprime les fichiers FITS temporaires d'un lot Drizzle incrémental."""
        if not batch_filepaths:
            return

        deleted_count = 0
        self.update_progress(f"   -> Nettoyage {len(batch_filepaths)} fichier(s) temp du lot...")
        for fpath in batch_filepaths:
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    deleted_count += 1
            except OSError as e:
                # Log l'erreur mais continue le nettoyage des autres fichiers
                self.update_progress(f"      ⚠️ Erreur suppression fichier temp {os.path.basename(fpath)}: {e}")
            except Exception as e_gen:
                self.update_progress(f"      ⚠️ Erreur inattendue suppression {os.path.basename(fpath)}: {e_gen}")

        if deleted_count > 0:
            self.update_progress(f"   -> {deleted_count}/{len(batch_filepaths)} fichier(s) temp nettoyé(s).")
        elif len(batch_filepaths) > 0:
            self.update_progress(f"   -> Aucun fichier temp du lot n'a pu être nettoyé (déjà supprimés ou erreur).")

# --- FIN DE LA NOUVELLE MÉTHODE ---



##########################################################################################################################################



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

################################################################################################################################################


    def start_processing(self, input_dir, output_dir, reference_path_ui=None,
                         initial_additional_folders=None,
                         # Weighting params
                         use_weighting=False, weight_snr=True, weight_stars=True,
                         snr_exp=1.0, stars_exp=0.5, min_w=0.1,
                         # Drizzle params
                         use_drizzle=False, drizzle_scale=2.0, drizzle_wht_threshold=0.7,
                         drizzle_mode="Final",
                         # --- NOUVEAUX PARAMÈTRES DANS LA SIGNATURE ---
                         drizzle_kernel="square", # Défaut 'square'
                         drizzle_pixfrac=1.0):     # Défaut 1.0
        """
        Démarre le thread de traitement principal avec la configuration spécifiée,
        y compris les options de pondération, le MODE, le NOYAU et PIXFRAC Drizzle.
        """
        if self.processing_active:
            self.update_progress("⚠️ Tentative de démarrer un traitement alors qu'un autre est déjà en cours.")
            return False

        # --- Stockage des paramètres reçus ---
        self.drizzle_mode = drizzle_mode if drizzle_mode in ["Final", "Incremental"] else "Final"
        self.drizzle_kernel = drizzle_kernel # Stocker la valeur reçue
        self.drizzle_pixfrac = drizzle_pixfrac   # Stocker la valeur reçue

        # Réinitialiser l'état d'arrêt et définir le dossier courant
        self.stop_processing = False
        self.current_folder = os.path.abspath(input_dir)

        # Initialiser les dossiers et l'état (TRÈS IMPORTANT)
        if not self.initialize(output_dir):
            self.processing_active = False
            return False

        # Stocker l'état Drizzle et ses paramètres pour cette session
        self.drizzle_active_session = use_drizzle # Stocke si Drizzle est demandé
        if self.drizzle_active_session:
            self.drizzle_scale = float(drizzle_scale)
            self.drizzle_wht_threshold = max(0.01, min(1.0, float(drizzle_wht_threshold)))
            # Log incluant le mode choisi
            self.update_progress(f"💧 Mode Drizzle Activé (Mode: {self.drizzle_mode}, Échelle: x{self.drizzle_scale:.1f}, Seuil WHT: {self.drizzle_wht_threshold*100:.0f}%)")
        else:
            self.update_progress("⚙️ Mode Stack Classique Activé pour cette session")

        # Vérifier et ajuster la taille de lot
        if self.batch_size < 3:
            self.update_progress(f"⚠️ Taille de lot ({self.batch_size}) trop petite, ajustée à 3.", None)
            self.batch_size = 3
        self.update_progress(f"ⓘ Taille de lot effective pour le traitement : {self.batch_size}")

        # Appliquer la configuration de la pondération qualité
        self.use_quality_weighting = use_weighting
        self.weight_by_snr = weight_snr
        self.weight_by_stars = weight_stars
        self.snr_exponent = max(0.1, snr_exp)
        self.stars_exponent = max(0.1, stars_exp)
        self.min_weight = max(0.01, min(1.0, min_w))
        if self.use_quality_weighting:
            self.update_progress(f"⚖️ Pondération Qualité Activée (SNR^{self.snr_exponent:.1f}, Stars^{self.stars_exponent:.1f}, MinW: {self.min_weight:.2f})")

        # Gérer les dossiers supplémentaires initiaux
        initial_folders_to_add_count = 0
        with self.folders_lock:
            self.additional_folders = []
            if initial_additional_folders:
                for folder in initial_additional_folders:
                    abs_folder = os.path.abspath(folder)
                    if os.path.isdir(abs_folder) and abs_folder not in self.additional_folders:
                        self.additional_folders.append(abs_folder)
                        initial_folders_to_add_count += 1
            if initial_folders_to_add_count > 0:
                 self.update_progress(f"ⓘ {initial_folders_to_add_count} dossier(s) pré-ajouté(s) en attente.")
                 self.update_progress(f"folder_count_update:{len(self.additional_folders)}")

        # Ajouter les fichiers du dossier initial à la file d'attente
        initial_files_added = self._add_files_to_queue(self.current_folder)
        if initial_files_added > 0:
            self._recalculate_total_batches()
            self.update_progress(f"📋 {initial_files_added} fichiers initiaux ajoutés. Total lots estimé: {self.total_batches_estimated if self.total_batches_estimated > 0 else '?'}")
        elif not self.additional_folders:
             self.update_progress("⚠️ Aucun fichier initial trouvé ou dossier supplémentaire en attente.")

        # Configurer l'image de référence pour l'aligneur
        self.aligner.reference_image_path = reference_path_ui or None

        # Démarrer le thread worker
        self.processing_thread = threading.Thread(target=self._worker, name="StackerWorker")
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.processing_active = True # Mettre le flag APRÈS le démarrage réussi
        self.update_progress("🚀 Thread de traitement démarré.")
        return True # Succès du démarrage


###############################################################################################################################################






    def _save_drizzle_input_temp(self, aligned_data, header):
        """
        Sauvegarde une image alignée (HxWx3 float32) dans le dossier temp Drizzle,
        en transposant en CxHxW et en INJECTANT l'OBJET WCS DE RÉFÉRENCE stocké
        dans le header sauvegardé.

        Args:
            aligned_data (np.ndarray): Données alignées (HxWx3 float32, 0-1).
            header (fits.Header): Header FITS ORIGINAL (pour métadonnées non-WCS).

        Returns:
            str or None: Chemin complet du fichier sauvegardé, ou None en cas d'erreur.
        """
        # Vérifications initiales
        if self.drizzle_temp_dir is None: self.update_progress("❌ Erreur interne: Dossier temp Drizzle non défini."); return None
        os.makedirs(self.drizzle_temp_dir, exist_ok=True)
        if aligned_data.ndim != 3 or aligned_data.shape[2] != 3: self.update_progress(f"❌ Erreur interne: _save_drizzle_input_temp attend HxWx3, reçu {aligned_data.shape}"); return None
        # --- VÉRIFIER SI L'OBJET WCS DE RÉFÉRENCE EST DISPONIBLE ---
        if self.reference_wcs_object is None:
             self.update_progress("❌ Erreur interne: Objet WCS de référence non disponible pour sauvegarde temp.")
             return None
        # --- FIN VÉRIFICATION ---

        try:
            temp_filename = f"aligned_input_{self.aligned_files_count:05d}.fits"
            temp_filepath = os.path.join(self.drizzle_temp_dir, temp_filename)

            # --- Préparer les données : Transposer HxWxC -> CxHxW ---
            data_to_save = np.moveaxis(aligned_data, -1, 0).astype(np.float32)

            # --- Préparer le header ---
            header_to_save = header.copy() if header else fits.Header()

            # --- EFFACER l'ancien WCS potentiellement invalide ---
            keys_to_remove = ['PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                              'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
                              'CDELT1', 'CDELT2', 'CROTA2']
            for key in keys_to_remove:
                if key in header_to_save:
                    del header_to_save[key]

            # --- INJECTER le WCS de l'OBJET WCS de référence ---
            ref_wcs_header = self.reference_wcs_object.to_header(relax=True)
            header_to_save.update(ref_wcs_header)

            # --- Mettre à jour NAXIS pour CxHxW ---
            header_to_save['NAXIS'] = 3
            header_to_save['NAXIS1'] = aligned_data.shape[1] # Width
            header_to_save['NAXIS2'] = aligned_data.shape[0] # Height
            header_to_save['NAXIS3'] = 3                   # Channels
            if 'CTYPE3' not in header_to_save: header_to_save['CTYPE3'] = 'CHANNEL'

            # --- Sauvegarde ---
            hdu = fits.PrimaryHDU(data=data_to_save, header=header_to_save)
            hdul = fits.HDUList([hdu])
            hdul.writeto(temp_filepath, overwrite=True, checksum=False, output_verify='ignore')
            hdul.close()

            # print(f"   -> Temp Drizzle sauvegardé ({os.path.basename(temp_filepath)}) avec WCS Ref Obj.") # DEBUG
            return temp_filepath

        except Exception as e:
            temp_filename_for_error = f"aligned_input_{self.aligned_files_count:05d}.fits"
            self.update_progress(f"❌ Erreur sauvegarde fichier temp Drizzle {temp_filename_for_error}: {e}")
            traceback.print_exc(limit=2)
            return None






################################################################################################################################################


    def _list_drizzle_temp_files(self):
        """
        Retourne la liste triée des chemins complets des fichiers FITS
        présents dans le dossier temporaire Drizzle.
        """
        # Vérifier si le dossier est défini et existe
        if self.drizzle_temp_dir is None or not os.path.isdir(self.drizzle_temp_dir):
            self.update_progress("⚠️ Dossier temp Drizzle non trouvé pour listage.")
            return [] # Retourner liste vide

        try:
            # Lister les fichiers correspondant au pattern attendu
            files = [
                os.path.join(self.drizzle_temp_dir, f)
                for f in os.listdir(self.drizzle_temp_dir)
                if f.lower().endswith('.fits') and f.startswith('aligned_input_')
            ]
            # Trier la liste pour un ordre cohérent
            files.sort()
            return files

        except Exception as e:
            # Gérer les erreurs de listage (permissions, etc.)
            self.update_progress(f"❌ Erreur listage fichiers temp Drizzle: {e}")
            return [] # Retourner liste vide en cas d'erreur

###################################################################################################################################################


    def _cleanup_drizzle_temp_files(self):
        """Supprime le dossier temporaire Drizzle et tout son contenu."""
        # Vérifier si le dossier est défini et existe
        if self.drizzle_temp_dir and os.path.isdir(self.drizzle_temp_dir):
            try:
                # Utiliser shutil.rmtree pour supprimer le dossier et son contenu
                shutil.rmtree(self.drizzle_temp_dir)
                self.update_progress(f"🧹 Dossier temporaire Drizzle supprimé: {os.path.basename(self.drizzle_temp_dir)}")
            except Exception as e:
                # Log l'erreur si la suppression échoue
                self.update_progress(f"⚠️ Erreur suppression dossier temp Drizzle ({os.path.basename(self.drizzle_temp_dir)}): {e}")
        # else: # Log optionnel si le dossier n'existait pas
            # self.update_progress("ⓘ Dossier temp Drizzle non trouvé pour nettoyage (normal si Drizzle inactif ou déjà nettoyé).")     
    def stop(self):
        if not self.processing_active: return
        self.update_progress("⛔ Arrêt demandé..."); self.stop_processing = True; self.aligner.stop_processing = True

    def is_running(self):
        return self.processing_active and self.processing_thread is not None and self.processing_thread.is_alive()
######################################################################################################################################################

# --- AJOUTER CETTE MÉTHODE DANS LA CLASSE SeestarQueuedStacker ---

    def _process_and_save_drizzle_batch(self, batch_data_list, output_wcs, output_shape_2d_hw, batch_num):
        """
        Traite un lot de données alignées en mémoire en utilisant Drizzle et sauvegarde
        les fichiers science (CxHxW) et poids (HxW x3) intermédiaires pour ce lot.
        Adapté de full_drizzle.py/process_single_batch.

        Args:
            batch_data_list (list): Liste de tuples: [(aligned_data_HxWxC, header, wcs_object), ...].
                                    wcs_object doit être le WCS de référence pour toutes.
            output_wcs (astropy.wcs.WCS): WCS de sortie Drizzle (défini une fois au début).
            output_shape_2d_hw (tuple): Shape (H, W) de sortie Drizzle.
            batch_num (int): Numéro du lot actuel pour nommage des fichiers.

        Returns:
            tuple: (sci_filepath, [wht_r_filepath, wht_g_filepath, wht_b_filepath])
                   Chemins des fichiers intermédiaires créés pour ce lot, ou (None, []) si échec.
        """
        num_files_in_batch = len(batch_data_list)
        self.update_progress(f"💧 Traitement Drizzle du lot #{batch_num} ({num_files_in_batch} images)...")
        batch_start_time = time.time()

        if not batch_data_list:
            self.update_progress(f"   - Warning: Lot Drizzle #{batch_num} vide.")
            return None, []

        # --- Vérifier cohérence WCS et Shape Entrée (sécurité) ---
        ref_wcs_for_batch = None
        ref_input_shape_hw = None
        valid_batch_items = []
        for i, (img_data, hdr, wcs_obj) in enumerate(batch_data_list):
            if img_data is None or wcs_obj is None:
                self.update_progress(f"   - Warning: Donnée/WCS manquant pour image {i+1} du lot {batch_num}. Ignorée.")
                continue
            current_shape_hw = img_data.shape[:2]
            # Initialiser références sur la première image valide
            if ref_wcs_for_batch is None:
                ref_wcs_for_batch = wcs_obj
                ref_input_shape_hw = current_shape_hw
            # Vérifier les suivantes
            elif wcs_obj is not ref_wcs_for_batch: # Vérifier si c'est le même objet WCS
                 self.update_progress(f"   - Warning: WCS incohérent pour image {i+1} du lot {batch_num}. Ignorée.")
                 continue
            elif current_shape_hw != ref_input_shape_hw:
                 self.update_progress(f"   - Warning: Shape incohérente ({current_shape_hw} vs {ref_input_shape_hw}) pour image {i+1} du lot {batch_num}. Ignorée.")
                 continue
            valid_batch_items.append((img_data, hdr)) # Garder seulement données et header

        if not valid_batch_items:
            self.update_progress(f"   - Erreur: Aucune donnée valide trouvée dans le lot Drizzle #{batch_num}.")
            return None, []
        num_valid_images = len(valid_batch_items)
        self.update_progress(f"   - {num_valid_images}/{num_files_in_batch} images valides pour Drizzle dans le lot.")

        # --- Initialiser les objets Drizzle pour ce lot ---
        num_output_channels = 3
        channel_names = ['R', 'G', 'B']
        drizzlers_batch = []
        output_images_batch = [] # Stockera les résultats science du lot
        output_weights_batch = [] # Stockera les résultats poids du lot
        total_batch_ninputs = num_valid_images # Simple compte pour ce lot

        try:
            # Pré-allouer les tableaux numpy pour les résultats de CE lot
            for _ in range(num_output_channels):
                output_images_batch.append(np.zeros(output_shape_2d_hw, dtype=np.float32))
                output_weights_batch.append(np.zeros(output_shape_2d_hw, dtype=np.float32))

            # Initialiser les objets Drizzle en passant les tableaux et les paramètres
            for i in range(num_output_channels):
                driz_ch = Drizzle(
                    kernel=self.drizzle_kernel,     # Paramètre de la classe
                    fillval="0.0",                  # Remplir avec 0
                    out_img=output_images_batch[i], # Tableau science pré-alloué
                    out_wht=output_weights_batch[i] # Tableau poids pré-alloué
                )
                drizzlers_batch.append(driz_ch)
            self.update_progress(f"   - Objets Drizzle initialisés pour lot #{batch_num}.")

        except Exception as init_err:
            self.update_progress(f"   - ERREUR: Échec init Drizzle pour lot #{batch_num}: {init_err}")
            traceback.print_exc(limit=1)
            return None, []

        # --- Boucle sur les images VALIDES du lot ---
        processed_in_batch_count = 0
        for i, (input_data_hxwx3, input_header) in enumerate(valid_batch_items):
            if self.stop_processing: self.update_progress("🛑 Arrêt pendant traitement lot Drizzle."); break
            # Log moins verbeux ici, peut-être seulement tous les X fichiers
            # self.update_progress(f"      -> Ajout image {i+1}/{num_valid_images} au Drizzle lot #{batch_num}...")

            # --- Calcul Pixmap (utilise ref_wcs_for_batch et output_wcs) ---
            pixmap = None
            try:
                current_input_shape_hw = input_data_hxwx3.shape[:2] # H, W
                y_in, x_in = np.indices(current_input_shape_hw)
                # Utiliser le WCS de référence (qui est le même pour toutes les images alignées)
                world_coords = ref_wcs_for_batch.all_pix2world(x_in.flatten(), y_in.flatten(), 0)
                # Projeter sur la grille de sortie Drizzle
                x_out, y_out = output_wcs.all_world2pix(world_coords[0], world_coords[1], 0)
                pixmap = np.dstack((x_out.reshape(current_input_shape_hw), y_out.reshape(current_input_shape_hw))).astype(np.float32) # Shape (H, W, 2)
            except Exception as map_err:
                self.update_progress(f"      -> ERREUR création pixmap image {i+1}: {map_err}. Ignorée.")
                continue # Passer à l'image suivante

            # --- Ajout à Drizzle (par canal) ---
            if pixmap is not None:
                try:
                    # Obtenir temps de pose (fallback 1.0)
                    base_exptime = 1.0
                    if input_header and 'EXPTIME' in input_header:
                        try: base_exptime = max(1e-6, float(input_header['EXPTIME']))
                        except (ValueError, TypeError): pass

                    # Boucle sur les canaux R, G, B
                    for ch_index in range(num_output_channels):
                        channel_data_2d = input_data_hxwx3[..., ch_index].astype(np.float32)
                        # Nettoyer NaN/Inf potentiels AVANT add_image
                        finite_mask = np.isfinite(channel_data_2d)
                        if not np.all(finite_mask):
                            channel_data_2d[~finite_mask] = 0.0

                        # Appeler add_image
                        drizzlers_batch[ch_index].add_image(
                            data=channel_data_2d,
                            pixmap=pixmap,
                            exptime=base_exptime,
                            pixfrac=self.drizzle_pixfrac, # Paramètre de la classe
                            in_units='counts'
                        )
                    processed_in_batch_count += 1
                except Exception as drizzle_add_err:
                    self.update_progress(f"      -> ERREUR add_image {i+1}: {drizzle_add_err}")
                    traceback.print_exc(limit=1)
                finally:
                    del pixmap, channel_data_2d; gc.collect() # Nettoyer pixmap et canal

        # --- Fin boucle sur les images du lot ---
        batch_end_time = time.time()
        self.update_progress(f"   -> Fin traitement Drizzle lot #{batch_num} ({processed_in_batch_count}/{num_valid_images} images ajoutées en {batch_end_time - batch_start_time:.2f}s).")

        if processed_in_batch_count == 0:
            self.update_progress(f"   - Warning: Aucune image traitée avec succès dans lot Drizzle #{batch_num}. Pas de sauvegarde.")
            del drizzlers_batch, output_images_batch, output_weights_batch; gc.collect()
            return None, []

        # --- Sauvegarde des résultats intermédiaires de CE lot ---
        # Utiliser le dossier spécifique défini dans initialize
        batch_output_dir = self.drizzle_batch_output_dir # Utiliser l'attribut de classe
        os.makedirs(batch_output_dir, exist_ok=True)

        base_out_filename = f"batch_{batch_num:04d}_s{self.drizzle_scale:.1f}p{self.drizzle_pixfrac:.1f}{self.drizzle_kernel}"
        out_filepath_sci = os.path.join(batch_output_dir, f"{base_out_filename}_sci.fits")
        out_filepaths_wht = []
        self.update_progress(f"   -> Sauvegarde résultats intermédiaires lot #{batch_num}...")

        try: # Sauvegarde Science (CxHxW)
            final_sci_data_batch_hwc = np.stack(output_images_batch, axis=-1) # HxWxC
            final_sci_data_to_save = np.moveaxis(final_sci_data_batch_hwc, -1, 0).astype(np.float32) # CxHxW

            final_header_sci = output_wcs.to_header(relax=True)
            final_header_sci['NINPUTS'] = (processed_in_batch_count, f'Inputs batch {batch_num}')
            final_header_sci['ISCALE'] = (self.drizzle_scale, 'Scale'); final_header_sci['PIXFRAC'] = (self.drizzle_pixfrac, 'Pixfrac')
            final_header_sci['KERNEL'] = (self.drizzle_kernel, 'Kernel'); final_header_sci['HISTORY'] = f'Batch {batch_num} by Drizzle Final Mode'
            final_header_sci['BUNIT'] = 'Counts/s';
            final_header_sci['NAXIS'] = 3; final_header_sci['NAXIS1'] = final_sci_data_to_save.shape[2] # W
            final_header_sci['NAXIS2'] = final_sci_data_to_save.shape[1] # H; final_header_sci['NAXIS3'] = final_sci_data_to_save.shape[0] # C
            final_header_sci['CTYPE3'] = 'CHANNEL'
            try: final_header_sci['CHNAME1'] = 'R'; final_header_sci['CHNAME2'] = 'G'; final_header_sci['CHNAME3'] = 'B'
            except Exception: pass

            fits.writeto(out_filepath_sci, final_sci_data_to_save, final_header_sci, overwrite=True, checksum=False, output_verify='ignore')
            self.update_progress(f"      -> Science lot sauvegardé: {os.path.basename(out_filepath_sci)}")
            del final_sci_data_batch_hwc, final_sci_data_to_save; gc.collect()

        except Exception as e:
            self.update_progress(f"   - ERREUR sauvegarde science lot #{batch_num}: {e}")
            traceback.print_exc(limit=1)
            del drizzlers_batch, output_images_batch, output_weights_batch; gc.collect()
            return None, []

        # Sauvegarde Poids (HxW par canal)
        for i in range(num_output_channels):
            ch_name = channel_names[i]
            out_filepath_wht = os.path.join(batch_output_dir, f"{base_out_filename}_wht_{ch_name}.fits")
            out_filepaths_wht.append(out_filepath_wht)
            try:
                wht_header = output_wcs.to_header(relax=True)
                for key in ['NAXIS3', 'CTYPE3', 'CRPIX3', 'CRVAL3', 'CDELT3', 'CUNIT3', 'CHNAME1', 'CHNAME2', 'CHNAME3']:
                    if key in wht_header: del wht_header[key]
                wht_header['NAXIS'] = 2; wht_header['NAXIS1'] = output_weights_batch[i].shape[1] # W
                wht_header['NAXIS2'] = output_weights_batch[i].shape[0] # H
                wht_header['HISTORY'] = f'Weights ({ch_name}) batch {batch_num}'; wht_header['NINPUTS'] = processed_in_batch_count
                wht_header['BUNIT'] = 'Weight'

                fits.writeto(out_filepath_wht, output_weights_batch[i].astype(np.float32), wht_header, overwrite=True, checksum=False, output_verify='ignore')
            except Exception as e:
                self.update_progress(f"   - ERREUR sauvegarde poids {ch_name} lot #{batch_num}: {e}")
                traceback.print_exc(limit=1)
                if os.path.exists(out_filepath_sci):
                    try: os.remove(out_filepath_sci)
                    except Exception: pass
                for wht_f in out_filepaths_wht:
                    if os.path.exists(wht_f):
                        try: os.remove(wht_f)
                        except Exception: pass
                del drizzlers_batch, output_images_batch, output_weights_batch; gc.collect()
                return None, []

        self.update_progress(f"   -> Sauvegarde lot #{batch_num} terminée.")
        del drizzlers_batch, output_images_batch, output_weights_batch; gc.collect()
        return out_filepath_sci, out_filepaths_wht

######################################################################################################################################################

