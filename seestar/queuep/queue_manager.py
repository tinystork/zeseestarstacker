# --- START OF FILE seestar/queuep/queue_manager.py ---
"""
Module de gestion de file d'attente pour le traitement des images astronomiques.
Gère l'alignement et l'empilement incrémental par LOTS dans un thread séparé.
(Version Révisée 9: Imports strictement nécessaires au niveau module)
"""
print("DEBUG QM: Début chargement module queue_manager.py")

# --- Standard Library Imports ---
import gc
import math
import os
from queue import Queue, Empty # Essentiel pour la classe
import shutil
import threading              # Essentiel pour la classe (Lock)
import time
import traceback
import warnings
print("DEBUG QM: Imports standard OK.")

# --- Third-Party Library Imports ---
from ..core.background import subtract_background_2d, _PHOTOUTILS_AVAILABLE as _PHOTOUTILS_BG_SUB_AVAILABLE
import astroalign as aa
import cv2
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS, FITSFixedWarning
from ccdproc import CCDData, combine as ccdproc_combine
from ..enhancement.stack_enhancement import apply_edge_crop
print("DEBUG QM: Imports tiers (numpy, cv2, astropy, ccdproc) OK.")

# --- Optional Third-Party Imports (with availability flags) ---
try:
    import cupy
    _cupy_installed = True
    print("DEBUG QM: Import CuPy OK.")
except ImportError:
    _cupy_installed = False
    print("DEBUG QM: Import CuPy échoué (normal si non installé).")

try:
    # On importe juste Drizzle ici, car la CLASSE est utilisée dans les méthodes
    from drizzle.resample import Drizzle
    _OO_DRIZZLE_AVAILABLE = True
    print("DEBUG QM: Import drizzle.resample.Drizzle OK.")
except ImportError as e_driz_cls:
    _OO_DRIZZLE_AVAILABLE = False
    Drizzle = None # Définir comme None si indisponible
    print(f"ERROR QM: Échec import drizzle.resample.Drizzle: {e_driz_cls}")

# --- Internal Project Imports (Core Modules ABSOLUMENT nécessaires pour la classe/init) ---
# Core Alignment (Instancié dans __init__)
try:
    from ..core.alignment import SeestarAligner
    print("DEBUG QM: Import SeestarAligner OK.")
except ImportError as e: print(f"ERREUR QM: Échec import SeestarAligner: {e}"); raise
# Core Hot Pixels (Utilisé dans _worker -> _process_file)
try:
    from ..core.hot_pixels import detect_and_correct_hot_pixels
    print("DEBUG QM: Import detect_and_correct_hot_pixels OK.")
except ImportError as e: print(f"ERREUR QM: Échec import detect_and_correct_hot_pixels: {e}"); raise
# Core Image Processing (Utilisé PARTOUT)
try:
    from ..core.image_processing import (
        load_and_validate_fits,
        debayer_image,
        save_fits_image,
        save_preview_image
    )
    print("DEBUG QM: Imports image_processing OK.")
except ImportError as e: print(f"ERREUR QM: Échec import image_processing: {e}"); raise
# Core Utils (Utilisé PARTOUT)
try:
    from ..core.utils import check_cupy_cuda, estimate_batch_size
    print("DEBUG QM: Imports utils OK.")
except ImportError as e: print(f"ERREUR QM: Échec import utils: {e}"); raise
# Enhancement Color Correction (Instancié dans __init__)
try:
    from ..enhancement.color_correction import ChromaticBalancer
    print("DEBUG QM: Import ChromaticBalancer OK.")
except ImportError as e: print(f"ERREUR QM: Échec import ChromaticBalancer: {e}"); raise

# --- Imports INTERNES à déplacer en IMPORTS TARDIFS ---
# Ces modules seront importés seulement quand les méthodes spécifiques sont appelées
# pour éviter les dépendances circulaires au chargement initial.

from ..enhancement.drizzle_integration import _load_drizzle_temp_file, DrizzleProcessor, _create_wcs_from_header # Déplacé vers _worker, etc.
from ..enhancement.astrometry_solver import solve_image_wcs # Déplacé vers _worker/_process_file
from ..enhancement.mosaic_processor import process_mosaic_from_aligned_files # Déplacé vers _worker
from ..enhancement.stack_enhancement import StackEnhancer # Importé tardivement si nécessaire dans _save_final_stack ou ailleurs

# --- Configuration des Avertissements ---
warnings.filterwarnings('ignore', category=FITSFixedWarning)
print("DEBUG QM: Configuration warnings OK.")
# --- FIN Imports ---



class SeestarQueuedStacker:
    """
    Classe pour l'empilement des images Seestar avec file d'attente et traitement par lots.
    Gère l'alignement et l'empilement dans un thread séparé.
    Ajout de la pondération basée sur la qualité (SNR, Nombre d'étoiles).
    """
    print("DEBUG QM: Lecture de la définition de la classe SeestarQueuedStacker...")

    def __init__(self):
        print("\n==== DÉBUT INITIALISATION SeestarQueuedStacker (SUM/W) ====")
        
        # --- 1. Attributs Critiques et Simples ---
        print("  -> Initialisation attributs simples et flags...")
        self.processing_active = False; self.stop_processing = False; self.processing_error = None
        self.is_mosaic_run = False; self.drizzle_active_session = False # Sera défini dans start_processing
        self.perform_cleanup = True; self.use_quality_weighting = True # Désactivé pour SUM/W initial
        self.correct_hot_pixels = True; self.apply_chroma_correction = True
        self.apply_final_scnr = False # Nouveau flag SCNR
        # Callbacks
        self.progress_callback = None; self.preview_callback = None
        # Queue & Threading
        self.queue = Queue(); self.folders_lock = threading.Lock(); self.processing_thread = None
        # File & Folder Management
        self.processed_files = set(); self.additional_folders = []; self.current_folder = None
        self.output_folder = None; self.unaligned_folder = None; self.drizzle_temp_dir = None
        self.drizzle_batch_output_dir = None; self.final_stacked_path = None
        # Astrometry & WCS Refs
        self.api_key = None; self.reference_wcs_object = None; self.reference_header_for_wcs = None
        self.reference_pixel_scale_arcsec = None; self.drizzle_output_wcs = None; self.drizzle_output_shape_hw = None
        
        ### NOUVEAU : Attributs pour SUM / W (Memmap) ###
        self.sum_memmap_path = None # Sera défini dans initialize
        self.wht_memmap_path = None # Sera défini dans initialize
        self.cumulative_sum_memmap = None  # Référence à l'objet memmap SUM
        self.cumulative_wht_memmap = None  # Référence à l'objet memmap WHT
        self.memmap_shape = None           # Shape des tableaux (H, W, C ou H, W)
        self.memmap_dtype_sum = np.float32 # Type pour la somme (float32 devrait suffire)
        self.memmap_dtype_wht = np.uint16  # Type pour les poids (uint16 = max 65535 images)
        print("  -> Attributs SUM/W (memmap) initialisés à None.")
        ### FIN NOUVEAU ###

        # --- SUPPRIMÉ : Anciens accumulateurs en mémoire ---
        # self.current_batch_data = [] # Sera toujours utilisé pour un lot TEMPORAIRE
        # self.current_stack_data = None # Remplacé par cumulative_sum_memmap / wht_memmap
        # self.cumulative_drizzle_data = None # Remplacé
        # self.cumulative_drizzle_wht = None # Remplacé
        # --- FIN SUPPRIMÉ ---
        
        self.current_batch_data = [] # Gardé pour le traitement interne d'un lot
        self.current_stack_header = None # Gardé pour les métadonnées cumulatives
        self.images_in_cumulative_stack = 0 # Gardé pour stats / UI
        self.total_exposure_seconds = 0.0 # Gardé pour stats / UI
        self.intermediate_drizzle_batch_files = []

        # Processing Parameters (valeurs par défaut, seront écrasées par start_processing)
        self.stacking_mode = "kappa-sigma"; self.kappa = 2.5; self.batch_size = 10
        self.hot_pixel_threshold = 3.0; self.neighborhood_size = 5; self.bayer_pattern = "GRBG"
        self.drizzle_mode = "Final"; self.drizzle_scale = 2.0; self.drizzle_wht_threshold = 0.7
        self.drizzle_kernel = "square"; self.drizzle_pixfrac = 1.0
        self.snr_exponent = 1.0; self.stars_exponent = 0.5; self.min_weight = 0.1
        self.final_scnr_target_channel = 'green'; self.final_scnr_amount = 0.8; self.final_scnr_preserve_luminosity = True
        
        # Statistics
        self.files_in_queue = 0; self.processed_files_count = 0; self.aligned_files_count = 0
        self.stacked_batches_count = 0; self.total_batches_estimated = 0
        self.failed_align_count = 0; self.failed_stack_count = 0; self.skipped_files_count = 0
        print("  -> Attributs simples et paramètres par défaut initialisés.")

        # --- 2. Instanciations de Classes ---
        try:
            print("  -> Instanciation ChromaticBalancer...")
            self.chroma_balancer = ChromaticBalancer(border_size=50, blur_radius=15)
            print("     ✓ ChromaticBalancer OK.")
        except Exception as e_cb: print(f"  -> ERREUR ChromaticBalancer: {e_cb}"); self.chroma_balancer = None; raise
        try:
            print("  -> Instanciation SeestarAligner...")
            self.aligner = SeestarAligner()
            print("     ✓ SeestarAligner OK.")
        except Exception as e_align: print(f"  -> ERREUR SeestarAligner: {e_align}"); self.aligner = None; raise

        print("==== FIN INITIALISATION SeestarQueuedStacker (SUM/W) ====\n")




######################################################################################################################################################




# --- DANS LA CLASSE SeestarQueuedStacker DANS seestar/queuep/queue_manager.py ---

    def initialize(self, output_dir, reference_image_shape):
        """
        Prépare les dossiers, réinitialise l'état, et CRÉE/INITIALISE
        les fichiers memmap pour SUM et WHT.

        Args:
            output_dir (str): Chemin du dossier de sortie principal.
            reference_image_shape (tuple): Shape (H, W, C=3) de l'image de référence
                                           (et donc des accumulateurs SUM/WHT).
        """
        print(f"DEBUG QM [initialize SUM/W]: Début avec output_dir='{output_dir}', shape={reference_image_shape}")

        # --- Nettoyage et création dossiers (comme avant) ---
        try:
            self.output_folder = os.path.abspath(output_dir)
            self.unaligned_folder = os.path.join(self.output_folder, "unaligned_files")
            self.drizzle_temp_dir = os.path.join(self.output_folder, "drizzle_temp_inputs")
            self.drizzle_batch_output_dir = os.path.join(self.output_folder, "drizzle_batch_outputs")
            
            ### NOUVEAU : Définir chemins memmap ###
            # Placer les fichiers .npy dans un sous-dossier pour la clarté
            memmap_dir = os.path.join(self.output_folder, "memmap_accumulators")
            self.sum_memmap_path = os.path.join(memmap_dir, "cumulative_SUM.npy")
            self.wht_memmap_path = os.path.join(memmap_dir, "cumulative_WHT.npy")
            print(f"DEBUG QM [initialize SUM/W]: Chemins Memmap définis -> SUM='{self.sum_memmap_path}', WHT='{self.wht_memmap_path}'")
            ### FIN NOUVEAU ###

            os.makedirs(self.output_folder, exist_ok=True)
            os.makedirs(self.unaligned_folder, exist_ok=True)
            os.makedirs(self.drizzle_temp_dir, exist_ok=True)
            os.makedirs(self.drizzle_batch_output_dir, exist_ok=True)
            os.makedirs(memmap_dir, exist_ok=True) # Créer le dossier memmap

            # Nettoyage ancien (si activé)
            # Pas besoin de nettoyer les fichiers memmap ici, on va les écraser avec mode 'w+'
            if self.perform_cleanup:
                if os.path.isdir(self.drizzle_temp_dir):
                    try: shutil.rmtree(self.drizzle_temp_dir); os.makedirs(self.drizzle_temp_dir) # Recréer après suppression
                    except Exception as e: self.update_progress(f"⚠️ Erreur nettoyage ancien dossier temp Drizzle: {e}")
                if os.path.isdir(self.drizzle_batch_output_dir):
                    try: shutil.rmtree(self.drizzle_batch_output_dir); os.makedirs(self.drizzle_batch_output_dir) # Recréer
                    except Exception as e: self.update_progress(f"⚠️ Erreur nettoyage ancien dossier sorties batch Drizzle: {e}")
                # On ne supprime pas explicitement les .npy, open_memmap('w+') va écraser

            self.update_progress(f"🗄️ Dossiers prêts (y compris memmap).")

        except OSError as e:
            self.update_progress(f"❌ Erreur critique création dossiers: {e}", 0)
            print(f"ERREUR QM [initialize SUM/W]: Échec création dossiers.") # Debug
            return False

        # --- Validation Shape Référence ---
        if not isinstance(reference_image_shape, tuple) or len(reference_image_shape) != 3 or reference_image_shape[2] != 3:
            self.update_progress(f"❌ Erreur interne: Shape référence invalide pour memmap ({reference_image_shape}). Attendue (H, W, 3).")
            print(f"ERREUR QM [initialize SUM/W]: Shape référence invalide.") # Debug
            return False
        self.memmap_shape = reference_image_shape # Stocker la shape (H, W, C)
        wht_shape = reference_image_shape[:2] # Shape pour WHT (H, W)
        print(f"DEBUG QM [initialize SUM/W]: Shape Memmap SUM={self.memmap_shape}, WHT={wht_shape}") # Debug

        # --- Création et Initialisation des Fichiers Memmap ---
        print(f"DEBUG QM [initialize SUM/W]: Tentative création/ouverture fichiers memmap (mode 'w+')...")
        try:
            # Note: mode='w+' crée ou écrase le fichier.
            # Utiliser np.float32 pour SUM, car float64 prendrait 2x plus de place
            # et la somme de floats 0-1 ne devrait pas dépasser les limites de float32 facilement.
            # Si des problèmes de précision apparaissent, on pourra passer à float64.
            self.cumulative_sum_memmap = np.lib.format.open_memmap(
                self.sum_memmap_path, mode='w+', dtype=self.memmap_dtype_sum, shape=self.memmap_shape
            )
            self.cumulative_sum_memmap[:] = 0.0 # Initialiser à zéro
            print(f"DEBUG QM [initialize SUM/W]: Memmap SUM créé/ouvert et initialisé à zéro.") # Debug

            self.cumulative_wht_memmap = np.lib.format.open_memmap(
                self.wht_memmap_path, mode='w+', dtype=self.memmap_dtype_wht, shape=wht_shape # Shape H,W et uint16
            )
            self.cumulative_wht_memmap[:] = 0 # Initialiser à zéro
            print(f"DEBUG QM [initialize SUM/W]: Memmap WHT créé/ouvert et initialisé à zéro.") # Debug

        except (IOError, OSError, ValueError, TypeError) as e_memmap:
            self.update_progress(f"❌ Erreur création/initialisation fichier memmap: {e_memmap}")
            print(f"ERREUR QM [initialize SUM/W]: Échec memmap : {e_memmap}") # Debug
            traceback.print_exc(limit=2)
            # Nettoyer les références si erreur
            self.cumulative_sum_memmap = None
            self.cumulative_wht_memmap = None
            self.sum_memmap_path = None
            self.wht_memmap_path = None
            return False
            
        # --- Réinitialisations Autres (comme avant, mais sans les anciens accumulateurs mémoire) ---
        print("DEBUG QM [initialize SUM/W]: Réinitialisation des autres états...") # Debug
        self.reference_wcs_object = None; self.intermediate_drizzle_batch_files = []; self.drizzle_output_wcs = None
        self.drizzle_output_shape_hw = None; # cumulative_drizzle_data/wht sont supprimés
        self.drizzle_kernel = "square"; self.drizzle_pixfrac = 1.0; self.processed_files.clear()
        with self.folders_lock: self.additional_folders = []
        self.current_batch_data = []; self.current_stack_header = None; self.images_in_cumulative_stack = 0
        self.total_exposure_seconds = 0.0; self.final_stacked_path = None; self.processing_error = None
        self.files_in_queue = 0; self.processed_files_count = 0; self.aligned_files_count = 0
        self.stacked_batches_count = 0; self.total_batches_estimated = 0
        self.failed_align_count = 0; self.failed_stack_count = 0; self.skipped_files_count = 0
        self.drizzle_active_session = False; self.reference_header_for_wcs = None

        # Vider la queue
        while not self.queue.empty():
            try: self.queue.get_nowait(); self.queue.task_done()
            except Exception: break

        if hasattr(self, 'aligner'): self.aligner.stop_processing = False
        print("DEBUG QM [initialize SUM/W]: Initialisation terminée avec succès.") # Debug
        return True




########################################################################################################################################################


    def update_progress(self, message, progress=None):
        message = str(message)
        if self.progress_callback:
            try: self.progress_callback(message, progress)
            except Exception as e: print(f"Error in progress callback: {e}")
        else:
            if progress is not None: print(f"[{int(progress)}%] {message}")
            else: print(message)

########################################################################################################################################################
    

    def set_progress_callback(self, callback):
        # ... (code identique à avant) ...
        print("DEBUG QM: Appel de set_progress_callback.") # Debug
        self.progress_callback = callback
        if hasattr(self, 'aligner') and hasattr(self.aligner, 'set_progress_callback') and callable(callback):
            try:
                print("DEBUG QM: Tentative de configuration callback sur aligner...") # Debug
                self.aligner.set_progress_callback(callback)
                print("DEBUG QM: Callback aligner configuré.") # Debug
            except Exception as e_align_cb: print(f"Warning QM: Could not set progress callback on aligner: {e_align_cb}")
        else: print("DEBUG QM: Ne configure pas callback aligner (aligner ou méthode manquante).") # Debug

########################################################################################################################################################


    def set_preview_callback(self, callback):
        """Définit la fonction de rappel pour les mises à jour de l'aperçu."""
        self.preview_callback = callback


################################################################################################################################################


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





# --- DANS LA CLASSE SeestarQueuedStacker DANS seestar/queuep/queue_manager.py ---

    # --- NOUVELLE MÉTHODE D'APERÇU POUR SUM/W ---
    def _update_preview_sum_w(self, downsample_factor=2):
        """
        Calcule l'image moyenne depuis SUM/W et appelle le callback preview GUI.
        Peut réduire la taille de l'image envoyée pour la performance.

        Args:
            downsample_factor (int): Facteur de réduction de taille pour l'aperçu (ex: 4 pour diviser H et W par 4).
                                     Mettre à 1 pour ne pas réduire.
        """
        print("DEBUG QM [_update_preview_sum_w]: Tentative de mise à jour de l'aperçu SUM/W...") # Debug

        if self.preview_callback is None:
            print("DEBUG QM [_update_preview_sum_w]: Pas de callback preview défini.") # Debug
            return
        if self.cumulative_sum_memmap is None or self.cumulative_wht_memmap is None:
            print("DEBUG QM [_update_preview_sum_w]: Accumulateurs SUM/WHT non disponibles.") # Debug
            # Optionnel : Envoyer une image noire ou rien ? Pour l'instant on ne fait rien.
            # self.preview_callback(None, None, "Error: Accumulators missing", 0, 0, 0, 0)
            return

        try:
            # --- Lecture (partielle ou complète) depuis Memmap ---
            # Pour la performance, surtout avec de grandes images, on pourrait
            # ne lire qu'une version réduite directement depuis le memmap si
            # l'aperçu n'a pas besoin de la pleine résolution.
            # Exemple de lecture réduite (slicing):
            # sum_data = self.cumulative_sum_memmap[::downsample_factor, ::downsample_factor, :]
            # wht_data = self.cumulative_wht_memmap[::downsample_factor, ::downsample_factor]
            
            # Pour l'instant, lisons tout et réduisons après la division
            # Attention : lire tout le memmap en mémoire peut être lourd !
            print("DEBUG QM [_update_preview_sum_w]: Lecture des données depuis memmap...") # Debug
            # Utiliser np.array() pour charger en mémoire (peut être lourd !)
            # On pourrait garder les refs memmap et faire la division dessus, mais
            # l'envoi au GUI via le callback nécessite souvent une copie en mémoire.
            current_sum = np.array(self.cumulative_sum_memmap, dtype=np.float64) # float64 pour division précise
            current_wht = np.array(self.cumulative_wht_memmap, dtype=np.float64) # float64 aussi
            print(f"DEBUG QM [_update_preview_sum_w]: Données lues. SUM shape={current_sum.shape}, WHT shape={current_wht.shape}") # Debug


            # --- Calcul de l'Image Moyenne (SUM / WHT) ---
            print("DEBUG QM [_update_preview_sum_w]: Calcul de l'image moyenne (division SUM/WHT)...") # Debug
            # Ajouter WHT sur l'axe des canaux pour correspondre à SUM
            # et éviter la division par zéro
            epsilon = 1e-9 # Petite valeur pour éviter division par zéro
            wht_broadcasted = np.maximum(current_wht, epsilon)[:, :, np.newaxis] # Ajoute l'axe C et évite zéro

            with np.errstate(divide='ignore', invalid='ignore'): # Ignorer les avertissements de division
                preview_data_fullres = (current_sum / wht_broadcasted)
            
            # Remplacer les NaN/Inf résultants de la division par 0
            preview_data_fullres = np.nan_to_num(preview_data_fullres, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"DEBUG QM [_update_preview_sum_w]: Image moyenne calculée. Shape={preview_data_fullres.shape}") # Debug


            # --- Normalisation 0-1 (Important pour l'affichage !) ---
            # L'image SUM/W n'est pas garantie d'être entre 0 et 1
            min_val = np.nanmin(preview_data_fullres)
            max_val = np.nanmax(preview_data_fullres)
            print(f"DEBUG QM [_update_preview_sum_w]: Range avant normalisation 0-1: [{min_val:.3f}, {max_val:.3f}]") # Debug
            if max_val > min_val:
                 preview_data_normalized = (preview_data_fullres - min_val) / (max_val - min_val)
            else: # Image constante
                 preview_data_normalized = np.zeros_like(preview_data_fullres)
            
            preview_data_normalized = np.clip(preview_data_normalized, 0.0, 1.0).astype(np.float32)
            print(f"DEBUG QM [_update_preview_sum_w]: Image normalisée 0-1.") # Debug


            # --- Réduction de Taille (Downsampling) Optionnelle ---
            preview_data_to_send = preview_data_normalized
            if downsample_factor > 1:
                 print(f"DEBUG QM [_update_preview_sum_w]: Réduction taille par facteur {downsample_factor}...") # Debug
                 try:
                     h, w, _ = preview_data_normalized.shape
                     new_h, new_w = h // downsample_factor, w // downsample_factor
                     if new_h > 10 and new_w > 10: # Seulement si taille résultante est raisonnable
                          preview_data_to_send = cv2.resize(preview_data_normalized, (new_w, new_h), interpolation=cv2.INTER_AREA)
                          print(f"DEBUG QM [_update_preview_sum_w]: Image réduite à {preview_data_to_send.shape}") # Debug
                     else:
                          print("DEBUG QM [_update_preview_sum_w]: Réduction taille annulée (résultat trop petit).") # Debug
                 except Exception as e_resize:
                      print(f"ERREUR QM [_update_preview_sum_w]: Échec réduction taille: {e_resize}") # Debug
                      # On envoie la version pleine résolution si le resize échoue

            # --- Préparation Header et Infos ---
            header_copy = self.current_stack_header.copy() if self.current_stack_header else fits.Header()
            # Ajouter/Mettre à jour NIMAGES/TOTEXP dans le header pour l'aperçu
            header_copy['NIMAGES'] = (self.images_in_cumulative_stack, 'Images processed so far (SUM/W)')
            header_copy['TOTEXP'] = (round(self.total_exposure_seconds, 2), '[s] Approx exposure accumulated (SUM/W)')

            img_count = self.images_in_cumulative_stack
            total_imgs_est = self.files_in_queue
            current_batch_num = self.stacked_batches_count # Le dernier lot traité
            total_batches_est = self.total_batches_estimated
            stack_name = f"Accum (SUM/W) ({img_count}/{total_imgs_est} Img | Lot {current_batch_num}/{total_batches_est if total_batches_est > 0 else '?'})"

            # --- Appel du Callback GUI ---
            print(f"DEBUG QM [_update_preview_sum_w]: Appel du callback preview avec image shape {preview_data_to_send.shape}...") # Debug
            self.preview_callback(
                preview_data_to_send, # Envoyer la version potentiellement réduite
                header_copy,
                stack_name,
                img_count,
                total_imgs_est,
                current_batch_num,
                total_batches_est
            )
            print("DEBUG QM [_update_preview_sum_w]: Callback preview terminé.") # Debug

        except MemoryError as mem_err:
             print(f"ERREUR QM [_update_preview_sum_w]: ERREUR MÉMOIRE - {mem_err}") # Debug
             self.update_progress(f"❌ ERREUR MÉMOIRE lors de la préparation de l'aperçu SUM/W.")
             traceback.print_exc(limit=1)
             # Ne pas envoyer d'aperçu si erreur mémoire
        except Exception as e:
            print(f"ERREUR QM [_update_preview_sum_w]: Exception inattendue - {e}") # Debug
            self.update_progress(f"Error in preview callback (SUM/W): {e}")
            traceback.print_exc(limit=2)

# --- FIN de la nouvelle méthode _update_preview_sum_w ---





#############################################################################################################################################################


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



    def _calculate_final_mosaic_grid(self, all_input_wcs_list):
        """
        Calcule le WCS et la Shape optimaux pour la mosaïque finale en se basant
        sur l'étendue couverte par tous les WCS d'entrée.

        Args:
            all_input_wcs_list (list): Liste des objets astropy.wcs.WCS
                                       provenant de toutes les images d'entrée alignées.
                                       IMPORTANT: Chaque WCS doit avoir .pixel_shape défini !

        Returns:
            tuple: (output_wcs, output_shape_hw) ou (None, None) si échec.
        """
        num_wcs = len(all_input_wcs_list)
        print(f"DEBUG (Backend _calculate_final_mosaic_grid): Appel avec {num_wcs} WCS d'entrée.")
        self.update_progress(f"📐 Calcul de la grille de sortie mosaïque ({num_wcs} WCS)...")

        if num_wcs == 0:
            print("ERREUR (Backend _calculate_final_mosaic_grid): Aucune information WCS fournie.")
            return None, None

        # --- Validation des WCS d'entrée ---
        valid_wcs_list = []
        for i, wcs_in in enumerate(all_input_wcs_list):
            if wcs_in is None or not wcs_in.is_celestial:
                print(f"   - WARNING: WCS {i+1} invalide ou non céleste. Ignoré.")
                continue
            if wcs_in.pixel_shape is None:
                print(f"   - WARNING: WCS {i+1} n'a pas de pixel_shape défini. Ignoré.")
                # Tenter de l'ajouter si possible (basé sur NAXIS du header de référence?)
                # C'est risqué ici, il vaut mieux s'assurer qu'il est défini AVANT
                continue
            valid_wcs_list.append(wcs_in)

        if not valid_wcs_list:
            print("ERREUR (Backend _calculate_final_mosaic_grid): Aucun WCS d'entrée valide trouvé.")
            return None, None
        print(f"   -> {len(valid_wcs_list)} WCS valides retenus pour le calcul.")

        try:
            # --- 1. Calculer le "footprint" (empreinte) de chaque image sur le ciel ---
            #    Le footprint est la projection des 4 coins de l'image dans les coordonnées célestes.
            all_footprints_sky = []
            print("   -> Calcul des footprints célestes...")
            for wcs_in in valid_wcs_list:
                # wcs_in.pixel_shape est (nx, ny)
                nx, ny = wcs_in.pixel_shape
                # Calculer le footprint en coordonnées pixel (0-based corners)
                # Ordre: (0,0), (nx-1, 0), (nx-1, ny-1), (0, ny-1)
                pixel_corners = np.array([
                    [0, 0], [nx - 1, 0], [nx - 1, ny - 1], [0, ny - 1]
                ], dtype=np.float64)
                # Projeter ces coins sur le ciel
                sky_corners = wcs_in.pixel_to_world(pixel_corners[:, 0], pixel_corners[:, 1])
                all_footprints_sky.append(sky_corners)

            # --- 2. Déterminer l'étendue totale de la mosaïque ---
            #    Trouver les RA/Dec min/max de tous les coins projetés.
            #    Attention à la discontinuité du RA à 0h/24h (ou 0/360 deg).
            #    SkyCoord gère cela mieux.
            print("   -> Détermination de l'étendue totale...")
            all_corners_flat = SkyCoord(ra=np.concatenate([fp.ra.deg for fp in all_footprints_sky]),
                                        dec=np.concatenate([fp.dec.deg for fp in all_footprints_sky]),
                                        unit='deg', frame='icrs') # Assumer ICRS

            # Trouver le centre approximatif pour aider à gérer le wrap RA
            central_ra = np.median(all_corners_flat.ra.wrap_at(180*u.deg).deg)
            central_dec = np.median(all_corners_flat.dec.deg)
            print(f"      - Centre Approx (RA, Dec): ({central_ra:.4f}, {central_dec:.4f}) deg")

            # Calculer l'étendue en RA/Dec en tenant compte du wrap
            # On utilise wrap_at(180) pour le RA
            ra_values_wrapped = all_corners_flat.ra.wrap_at(180 * u.deg).deg
            min_ra_wrap, max_ra_wrap = np.min(ra_values_wrapped), np.max(ra_values_wrapped)
            min_dec, max_dec = np.min(all_corners_flat.dec.deg), np.max(all_corners_flat.dec.deg)

            # La taille angulaire en RA dépend de la déclinaison
            delta_ra_deg = (max_ra_wrap - min_ra_wrap) * np.cos(np.radians(central_dec))
            delta_dec_deg = max_dec - min_dec
            print(f"      - Étendue Approx (RA * cos(Dec), Dec): ({delta_ra_deg:.4f}, {delta_dec_deg:.4f}) deg")

            # --- 3. Définir le WCS de Sortie ---
            #    Utiliser le centre calculé, la même projection que la référence,
            #    et la nouvelle échelle de pixel.
            print("   -> Création du WCS de sortie...")
            ref_wcs = valid_wcs_list[0] # Utiliser le premier WCS valide comme base
            output_wcs = WCS(naxis=2)
            output_wcs.wcs.ctype = ref_wcs.wcs.ctype # Garder la projection (ex: TAN)
            output_wcs.wcs.crval = [central_ra, central_dec] # Centrer sur la mosaïque
            output_wcs.wcs.cunit = ref_wcs.wcs.cunit # Garder les unités (deg)

            # Calculer la nouvelle échelle de pixel (en degrés/pixel)
            # Utiliser la moyenne des échelles d'entrée ou l'échelle de référence
            ref_scale_matrix = ref_wcs.pixel_scale_matrix
            # Prendre la moyenne des valeurs absolues diagonales comme échelle approx
            avg_input_scale = np.mean(np.abs(np.diag(ref_scale_matrix)))
            output_pixel_scale = avg_input_scale / self.drizzle_scale
            print(f"      - Échelle Pixel Entrée (Moy): {avg_input_scale * 3600:.3f} arcsec/pix")
            print(f"      - Échelle Pixel Sortie Cible: {output_pixel_scale * 3600:.3f} arcsec/pix")

            # Appliquer la nouvelle échelle (CD matrix, en assumant pas de rotation/skew complexe)
            # Mettre le signe correct pour le RA (- pour axe X vers l'Est)
            output_wcs.wcs.cd = np.array([[-output_pixel_scale, 0.0],
                                          [0.0, output_pixel_scale]])

            # --- 4. Calculer la Shape de Sortie ---
            #    Projeter l'étendue totale (les coins extrêmes) sur la nouvelle grille WCS
            #    pour déterminer les dimensions en pixels nécessaires.
            print("   -> Calcul de la shape de sortie...")
            # Créer les coordonnées des coins englobants de la mosaïque
            # (On prend les min/max RA/Dec, attention au wrap RA)
            # C'est plus sûr de projeter *tous* les coins d'entrée dans le système de sortie
            all_output_pixels_x = []
            all_output_pixels_y = []
            for sky_corners in all_footprints_sky:
                pixels_out_x, pixels_out_y = output_wcs.world_to_pixel(sky_corners)
                all_output_pixels_x.extend(pixels_out_x)
                all_output_pixels_y.extend(pixels_out_y)

            # Trouver les min/max des coordonnées pixel de sortie
            x_min_out, x_max_out = np.min(all_output_pixels_x), np.max(all_output_pixels_x)
            y_min_out, y_max_out = np.min(all_output_pixels_y), np.max(all_output_pixels_y)

            # Calculer la largeur et la hauteur (ajouter 1 car indices 0-based)
            # Utiliser ceil pour s'assurer qu'on couvre tout
            out_width = int(np.ceil(x_max_out - x_min_out + 1))
            out_height = int(np.ceil(y_max_out - y_min_out + 1))
            # Assurer une taille minimale
            out_width = max(10, out_width)
            out_height = max(10, out_height)
            output_shape_hw = (out_height, out_width) # Ordre H, W
            print(f"      - Dimensions Pixels Calculées (W, H): ({out_width}, {out_height})")

            # --- 5. Finaliser le WCS de Sortie ---
            #    Ajuster CRPIX pour qu'il corresponde au nouveau centre pixel
            #    dans le système de coordonnées de sortie (0-based index).
            #    Le pixel (0,0) de la sortie correspond à (x_min_out, y_min_out)
            #    dans le système intermédiaire calculé par world_to_pixel.
            #    CRPIX (1-based) = (coord_centre_interm - coord_min_interm + 1)
            #    Calculer le pixel central dans le système 'output_pixels'
            center_x_out, center_y_out = output_wcs.world_to_pixel(SkyCoord(ra=central_ra*u.deg, dec=central_dec*u.deg))
            # Calculer CRPIX
            output_wcs.wcs.crpix = [
                center_x_out - x_min_out + 1.0, # CRPIX1
                center_y_out - y_min_out + 1.0  # CRPIX2
            ]
            # Définir la shape pour Astropy WCS (W, H)
            output_wcs.pixel_shape = (out_width, out_height)
            # Mettre à jour NAXIS internes
            try: output_wcs._naxis1 = out_width; output_wcs._naxis2 = out_height
            except AttributeError: pass

            print(f"      - WCS Finalisé: CRPIX={output_wcs.wcs.crpix}, PixelShape={output_wcs.pixel_shape}")
            print(f"DEBUG (Backend _calculate_final_mosaic_grid): Calcul grille mosaïque réussi.")
            return output_wcs, output_shape_hw # Retourne WCS et shape (H, W)

        except Exception as e:
            print(f"ERREUR (Backend _calculate_final_mosaic_grid): Échec calcul grille mosaïque: {e}")
            traceback.print_exc(limit=3)
            return None, None




###########################################################################################################################################################

    def _recalculate_total_batches(self):
        """Estimates the total number of batches based on files_in_queue."""
        if self.batch_size > 0: self.total_batches_estimated = math.ceil(self.files_in_queue / self.batch_size)
        else: self.update_progress(f"⚠️ Taille de lot invalide ({self.batch_size}), impossible d'estimer le nombre total de lots."); self.total_batches_estimated = 0




################################################################################################################################################





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


    def set_progress_callback(self, callback):
        print("DEBUG QM: Appel de set_progress_callback.")
        self.progress_callback = callback
        if hasattr(self, 'aligner') and hasattr(self.aligner, 'set_progress_callback') and callable(callback):
            try:
                print("DEBUG QM: Tentative de configuration callback sur aligner...")
                self.aligner.set_progress_callback(callback)
                print("DEBUG QM: Callback aligner configuré.")
            except Exception as e_align_cb: print(f"Warning QM: Could not set progress callback on aligner: {e_align_cb}")
        else: print("DEBUG QM: Ne configure pas callback aligner (aligner ou méthode manquante).")



################################################################################################################################################



    def set_preview_callback(self, callback):
        print("DEBUG QM: Appel de set_preview_callback.")
        self.preview_callback = callback


################################################################################################################################################




    def _worker(self):
        """
        Thread principal pour le traitement des images.
        """

        # --------------------------------------------------
        # 0.  Imports internes au thread (évite les cycles)
        # --------------------------------------------------
        import gc
        import os
        import time
        import traceback
        from queue import Empty

        # --------------------------------------------------
        # 1.  Initialisation
        # --------------------------------------------------
        print("\n" + "=" * 10 + " DEBUG [Worker Start]: Initialisation " + "=" * 10)

        self.processing_active = True
        self.processing_error = None
        start_time_session = time.monotonic()

        reference_image_data = None
        reference_header = None
        self.reference_wcs_object = None
        self.reference_header_for_wcs = None
        self.reference_pixel_scale_arcsec = None
        self.drizzle_output_wcs = None
        self.drizzle_output_shape_hw = None

        self.current_batch_data = []
        local_batch_temp_files = []
        local_drizzle_final_batch_data = []
        self.intermediate_drizzle_batch_files = []
        all_aligned_files_with_info = []

        print(
            f"DEBUG [Worker Start]: Mode reçu -> "
            f"is_mosaic_run={self.is_mosaic_run}, "
            f"drizzle_active_session={self.drizzle_active_session}, "
            f"drizzle_mode='{self.drizzle_mode}'"
        )

        # --------------------------------------------------
        # 2.  Imports tardifs (solver WCS & drizzle)
        # --------------------------------------------------
        solve_image_wcs_func = None
        DrizzleProcessor_class = None
        load_drizzle_temp_file_func = None
        create_wcs_from_header_func = None

        try:
            from ..enhancement.astrometry_solver import (
                solve_image_wcs as solve_image_wcs_func,
            )

            print("DEBUG [_worker]: Import tardif solve_image_wcs OK.")
        except ImportError:
            print("ERREUR [_worker]: Échec import tardif solve_image_wcs.")

        try:
            from ..enhancement.drizzle_integration import (
                _load_drizzle_temp_file as load_drizzle_temp_file_func,
            )
            from ..enhancement.drizzle_integration import (
                DrizzleProcessor as DrizzleProcessor_class,
            )
            from ..enhancement.drizzle_integration import (
                _create_wcs_from_header as create_wcs_from_header_func,
            )

            print("DEBUG [_worker]: Import tardif drizzle_integration OK.")
        except ImportError:
            print("ERREUR [_worker]: Échec import tardif drizzle_integration.")

        # --------------------------------------------------
        # 3.  Corps principal du thread
        # --------------------------------------------------
        try:
            # 3‑A.  Préparation de l’image de référence
            self.update_progress("⭐ Préparation image référence…")

            if not self.current_folder or not os.path.isdir(self.current_folder):
                raise RuntimeError(f"Dossier entrée invalide : {self.current_folder}")

            initial_files = sorted(
                f
                for f in os.listdir(self.current_folder)
                if f.lower().endswith((".fit", ".fits"))
            )

            if not initial_files and not self.additional_folders:
                # Aucun fichier FITS ni dossiers additionnels
                raise RuntimeError(
                    "Aucun FITS initial trouvé et pas de dossiers additionnels pour référence."
                )

            # Propager quelques paramètres à l’aligner
            self.aligner.correct_hot_pixels = self.correct_hot_pixels
            self.aligner.hot_pixel_threshold = self.hot_pixel_threshold
            self.aligner.neighborhood_size = self.neighborhood_size
            self.aligner.bayer_pattern = self.bayer_pattern

            # Récupération de l’image de référence
            reference_image_data, reference_header = self.aligner._get_reference_image(
                self.current_folder, initial_files
            )
            if reference_image_data is None or reference_header is None:
                raise RuntimeError(
                    "Échec obtention image/header référence pour alignement."
                )

            if (
                (self.drizzle_active_session or self.is_mosaic_run)
                and self.reference_wcs_object is None
            ):
                raise RuntimeError(
                    "WCS de référence requis pour Drizzle/Mosaïque mais non disponible."
                )

            if self.reference_header_for_wcs is None:
                self.reference_header_for_wcs = reference_header.copy()

            # Sauvegarde de l’image de référence (diagnostic)
            self.aligner._save_reference_image(
                reference_image_data, reference_header, self.output_folder
            )
            self.update_progress("⭐ Image de référence pour alignement prête.", 5)

            # Estimation du nombre total de lots à empiler
            self._recalculate_total_batches()
            self.update_progress(
                "▶️ Démarrage boucle traitement "
                f"(File: {self.files_in_queue} | Lots Est.: "
                f"{self.total_batches_estimated if self.total_batches_estimated > 0 else '?'} )"
            )

            # --------------------------------------------------
            # 3‑B.  Boucle principale de traitement
            # --------------------------------------------------
            while not self.stop_processing:
                file_path = None
                aligned_data = None
                header = None
                quality_scores = None
                wcs_object_indiv = None

                try:
                    file_path = self.queue.get(timeout=1.0)
                    file_name = os.path.basename(file_path)

                    if self.is_mosaic_run and solve_image_wcs_func is None:
                        raise ImportError("Solveur WCS requis pour mosaïque.")

                    aligned_data, header, quality_scores, wcs_object_indiv = (
                        self._process_file(file_path, reference_image_data)
                    )
                    self.processed_files_count += 1

                    # --------------------------------------------------
                    # 3‑B‑1.  En cas de succès
                    # --------------------------------------------------
                    if aligned_data is not None:
                        self.aligned_files_count += 1

                        if self.is_mosaic_run:
                            # Branche mosaïque : on stocke simplement
                            all_aligned_files_with_info.append(
                                (
                                    aligned_data,
                                    header,
                                    quality_scores,
                                    wcs_object_indiv,
                                )
                            )

                        else:
                            # Branche non‑mosaïque (Drizzle ou classique)
                            data_for_batch = aligned_data
                            header_for_batch = header
                            scores_for_batch = quality_scores
                            wcs_for_batch = wcs_object_indiv

                            # --- 3‑B‑1‑a.  Drizzle FINAL ---
                            if (
                                self.drizzle_active_session
                                and self.drizzle_mode == "Final"
                            ):
                                if wcs_for_batch:
                                    local_drizzle_final_batch_data.append(
                                        (
                                            data_for_batch,
                                            header_for_batch,
                                            self.reference_wcs_object,
                                        )
                                    )
                                    if (
                                        len(local_drizzle_final_batch_data)
                                        >= self.batch_size
                                    ):
                                        if self.drizzle_output_wcs is None:
                                            ref_shape_hw = (
                                                self.memmap_shape[:2]
                                                if self.memmap_shape
                                                else reference_image_data.shape[:2]
                                            )
                                            (
                                                self.drizzle_output_wcs,
                                                self.drizzle_output_shape_hw,
                                            ) = self._create_drizzle_output_wcs(
                                                self.reference_wcs_object,
                                                ref_shape_hw,
                                                self.drizzle_scale,
                                            )
                                        self.stacked_batches_count += 1
                                        (
                                            sci_path,
                                            wht_paths,
                                        ) = self._process_and_save_drizzle_batch(
                                            local_drizzle_final_batch_data,
                                            self.drizzle_output_wcs,
                                            self.drizzle_output_shape_hw,
                                            self.stacked_batches_count,
                                        )
                                        if sci_path and wht_paths:
                                            self.intermediate_drizzle_batch_files.append(
                                                (sci_path, wht_paths)
                                            )
                                        else:
                                            self.failed_stack_count += len(
                                                local_drizzle_final_batch_data
                                            )
                                        local_drizzle_final_batch_data = []
                                else:
                                    self.skipped_files_count += 1

                            # --- 3‑B‑1‑b.  Drizzle INCRÉMENTAL ---
                            elif (
                                self.drizzle_active_session
                                and self.drizzle_mode == "Incremental"
                            ):
                                temp_file = self._save_drizzle_input_temp(
                                    data_for_batch, header_for_batch
                                )
                                if temp_file:
                                    local_batch_temp_files.append(temp_file)
                                    if len(local_batch_temp_files) >= self.batch_size:
                                        self.stacked_batches_count += 1
                                        self._process_incremental_drizzle_batch(
                                            local_batch_temp_files,
                                            self.stacked_batches_count,
                                            self.total_batches_estimated,
                                        )
                                        local_batch_temp_files = []
                                else:
                                    self.skipped_files_count += 1

                            # --- 3‑B‑1‑c.  Empilage classique ---
                            else:
                                self.current_batch_data.append(
                                    (
                                        data_for_batch,
                                        header_for_batch,
                                        scores_for_batch,
                                    )
                                )
                                if len(self.current_batch_data) >= self.batch_size:
                                    self.stacked_batches_count += 1
                                    self._process_completed_batch(
                                        self.stacked_batches_count,
                                        self.total_batches_estimated,
                                    )
                                    self.current_batch_data = []

                            # Libération mémoire itérative
                            del (
                                aligned_data,
                                header,
                                quality_scores,
                                wcs_object_indiv,
                                data_for_batch,
                                header_for_batch,
                                scores_for_batch,
                                wcs_for_batch,
                            )
                            gc.collect()

                    # Indique au queue que la tâche est terminée
                    self.queue.task_done()

                # --------------------------------------------------
                # 3‑B‑2.  File vide → on vide / on passe dossier
                # --------------------------------------------------
                except Empty:
                    self.update_progress(
                        "ⓘ File vide. Vérification batch final / dossiers sup…"
                    )

                    if not self.is_mosaic_run:
                        # Traiter le dernier lot partiel si nécessaire
                        if (
                            self.drizzle_active_session
                            and self.drizzle_mode == "Final"
                            and local_drizzle_final_batch_data
                        ):
                            if self.drizzle_output_wcs is None:
                                ref_shape_hw = (
                                    self.memmap_shape[:2]
                                    if self.memmap_shape
                                    else reference_image_data.shape[:2]
                                )
                                (
                                    self.drizzle_output_wcs,
                                    self.drizzle_output_shape_hw,
                                ) = self._create_drizzle_output_wcs(
                                    self.reference_wcs_object,
                                    ref_shape_hw,
                                    self.drizzle_scale,
                                )
                            self.stacked_batches_count += 1
                            (
                                sci_path,
                                wht_paths,
                            ) = self._process_and_save_drizzle_batch(
                                local_drizzle_final_batch_data,
                                self.drizzle_output_wcs,
                                self.drizzle_output_shape_hw,
                                self.stacked_batches_count,
                            )
                            if sci_path and wht_paths:
                                self.intermediate_drizzle_batch_files.append(
                                    (sci_path, wht_paths)
                                )
                            else:
                                self.failed_stack_count += len(
                                    local_drizzle_final_batch_data
                                )
                            local_drizzle_final_batch_data = []

                        elif (
                            self.drizzle_active_session
                            and self.drizzle_mode == "Incremental"
                            and local_batch_temp_files
                        ):
                            self.stacked_batches_count += 1
                            self._process_incremental_drizzle_batch(
                                local_batch_temp_files,
                                self.stacked_batches_count,
                                self.total_batches_estimated,
                            )
                            local_batch_temp_files = []

                        elif (
                            not self.drizzle_active_session
                            and self.current_batch_data
                        ):
                            self.stacked_batches_count += 1
                            self._process_completed_batch(
                                self.stacked_batches_count,
                                self.total_batches_estimated,
                            )
                            self.current_batch_data = []

                    # Vérifie s’il reste des dossiers additionnels
                    folder_to_process = None
                    with self.folders_lock:
                        if self.additional_folders:
                            folder_to_process = self.additional_folders.pop(0)
                            self.update_progress(
                                f"folder_count_update:{len(self.additional_folders)}"
                            )

                    if folder_to_process:
                        self.current_folder = folder_to_process
                        self.update_progress(
                            f"📂 Traitement dossier supplémentaire : "
                            f"{os.path.basename(folder_to_process)}"
                        )
                        self._add_files_to_queue(folder_to_process)
                        self._recalculate_total_batches()
                        continue  # On repart dans le while
                    else:
                        self.update_progress("✅ Fin file/dossiers.")
                        break  # Sort de la boucle principale

                # --------------------------------------------------
                # 3‑B‑3.  Toute autre exception dans la boucle
                # --------------------------------------------------
                except Exception as e_inner_loop:
                    error_msg_loop = (
                        f"Erreur boucle interne: {type(e_inner_loop).__name__}: "
                        f"{e_inner_loop}"
                    )
                    print(error_msg_loop)
                    traceback.print_exc(limit=3)
                    self.update_progress(f"⚠️ {error_msg_loop}")
                    self.failed_stack_count += 1

                finally:
                    # Nettoyage mémoire itératif
                    gc.collect()

            # --------------------------------------------------
            # 3‑C.  Traitement du dernier lot (si sortie normale)
            # --------------------------------------------------
            if not self.stop_processing and not self.is_mosaic_run:
                print(
                    "DEBUG [_worker/AfterLoop]: Traitement dernier lot partiel "
                    "(sortie normale boucle)."
                )

                if (
                    self.drizzle_active_session
                    and self.drizzle_mode == "Final"
                    and local_drizzle_final_batch_data
                ):
                    print(
                        f"   -> Dernier lot Drizzle Final "
                        f"({len(local_drizzle_final_batch_data)} images)"
                    )
                    if self.drizzle_output_wcs is None:
                        ref_shape_hw = (
                            self.memmap_shape[:2]
                            if self.memmap_shape
                            else reference_image_data.shape[:2]
                        )
                        (
                            self.drizzle_output_wcs,
                            self.drizzle_output_shape_hw,
                        ) = self._create_drizzle_output_wcs(
                            self.reference_wcs_object,
                            ref_shape_hw,
                            self.drizzle_scale,
                        )
                    self.stacked_batches_count += 1
                    (
                        sci_path,
                        wht_paths,
                    ) = self._process_and_save_drizzle_batch(
                        local_drizzle_final_batch_data,
                        self.drizzle_output_wcs,
                        self.drizzle_output_shape_hw,
                        self.stacked_batches_count,
                    )
                    if sci_path and wht_paths:
                        self.intermediate_drizzle_batch_files.append(
                            (sci_path, wht_paths)
                        )
                    else:
                        self.failed_stack_count += len(local_drizzle_final_batch_data)
                    local_drizzle_final_batch_data = []

                elif (
                    self.drizzle_active_session
                    and self.drizzle_mode == "Incremental"
                    and local_batch_temp_files
                ):
                    print(
                        f"   -> Dernier lot Drizzle Incrémental "
                        f"({len(local_batch_temp_files)} images)"
                    )
                    self.stacked_batches_count += 1
                    self._process_incremental_drizzle_batch(
                        local_batch_temp_files,
                        self.stacked_batches_count,
                        self.total_batches_estimated,
                    )
                    local_batch_temp_files = []

                elif not self.drizzle_active_session and self.current_batch_data:
                    print(
                        f"   -> Dernier lot Classique "
                        f"({len(self.current_batch_data)} images)"
                    )
                    self.stacked_batches_count += 1
                    self._process_completed_batch(
                        self.stacked_batches_count,
                        self.total_batches_estimated,
                    )
                    self.current_batch_data = []

            # --------------------------------------------------
            # 3‑D.  Étape finale (sauvegarde cumul)
            # --------------------------------------------------
            print("DEBUG [_worker]: Fin boucle principale. Début logique finalisation...")

            # --- Nettoyage mémoire si non-mosaïque ---
            if not self.is_mosaic_run:
                print("DEBUG [_worker/Finalize]: Nettoyage all_aligned_files_with_info (mode non-mosaïque)...")
                all_aligned_files_with_info = [] # Vider la liste
                gc.collect()

            # --- MODIFICATION DE LA LOGIQUE D'ARRÊT ET DE SAUVEGARDE ---
            final_suffix_for_save = "_unknown_sumw" # Fallback

            if self.is_mosaic_run:
                # Pour la mosaïque, si arrêtée, il est complexe de sauvegarder un état partiel
                # de manière significative avec la logique SUM/W actuelle, car l'assemblage final
                # n'a pas eu lieu. On pourrait choisir de ne rien sauvegarder ou de sauvegarder
                # les panneaux intermédiaires si le nettoyage est désactivé.
                # Pour l'instant, si mosaïque et arrêt, on ne sauvegarde pas le stack final.
                if self.stop_processing:
                    self.update_progress("🛑 Traitement Mosaïque interrompu. Pas de sauvegarde finale de la mosaïque.")
                    self.final_stacked_path = None
                else: # Mosaïque terminée normalement
                    final_suffix_for_save = "_mosaic_sumw" # sera utilisé par _save_final_stack
            elif self.drizzle_active_session:
                final_suffix_for_save = f"_drizzle_{self.drizzle_mode.lower()}_sumw"
            else: # Classique
                final_suffix_for_save = f"_classic_{self.stacking_mode}_sumw"

            # Ajouter le suffixe "_stopped" si le traitement a été arrêté par l'utilisateur
            # mais SEULEMENT si ce n'est pas une mosaïque (car on ne sauvegarde pas de mosaïque stoppée pour l'instant)
            effective_output_suffix = final_suffix_for_save
            was_stopped_for_save_call = False

            if self.stop_processing:
                self.update_progress("🛑 Traitement interrompu par l'utilisateur.")
                if not self.is_mosaic_run: # On tente une sauvegarde si ce n'est pas une mosaïque
                    effective_output_suffix += "_stopped"
                    was_stopped_for_save_call = True
                    print(f"DEBUG [_worker/Finalize]: Arrêt utilisateur, tentative de sauvegarde partielle avec suffixe: {effective_output_suffix}")
                # else: pour la mosaïque, on a déjà géré le cas d'arrêt
            else: # Traitement Normal Terminé
                print("DEBUG [_worker]: Traitement normal terminé. Préparation pour sauvegarde finale SUM/W...")
            
            # --- Vérifier s'il y a quelque chose à sauvegarder ---
            # On appelle _save_final_stack si on a des images ET 
            # (soit le traitement n'a pas été stoppé, OU il a été stoppé MAIS ce n'est pas une mosaïque)
            if self.images_in_cumulative_stack > 0 and \
               (not self.stop_processing or (self.stop_processing and not self.is_mosaic_run)):
                
                print(f"DEBUG [_worker]: Appel _save_final_stack (images={self.images_in_cumulative_stack}, suffix='{effective_output_suffix}', stopped_early={was_stopped_for_save_call})...")
                self._save_final_stack(
                    output_filename_suffix=effective_output_suffix,
                    stopped_early=was_stopped_for_save_call
                )
            elif self.stop_processing and self.is_mosaic_run:
                # Cas spécifique: Mosaïque arrêtée, on a déjà loggué, on s'assure que final_stacked_path est None
                self.final_stacked_path = None
            else: # Pas d'images accumulées ou autre cas non géré pour la sauvegarde
                print("DEBUG [_worker]: Appel _save_final_stack ignoré (images_in_cumulative_stack <= 0 ou condition d'arrêt non remplie pour sauvegarde).")
                self.update_progress("ⓘ Aucun stack final à sauvegarder (0 images accumulées ou arrêt non compatible avec sauvegarde partielle).")
                self.final_stacked_path = None
            
        # ------------------------------------------------------
        # 4.  Gestion des exceptions globales
        # ------------------------------------------------------
        except Exception as e:
            error_msg = f"Erreur critique worker: {type(e).__name__}: {e}"
            print(f"ERREUR CRITIQUE: {error_msg}")
            self.update_progress(f"❌ {error_msg}")
            traceback.print_exc(limit=5)
            self.processing_error = error_msg
            print(
                "DEBUG [_worker EXCEPTION]: Tentative de fermeture des memmaps suite à erreur…"
            )
            self._close_memmaps()

        # ------------------------------------------------------
        # 5.  Bloc finally → nettoyage systématique
        # ------------------------------------------------------
        finally:
            print("DEBUG [_worker]: Entrée bloc FINALLY…")

            if self.perform_cleanup:
                self.update_progress("🧹 Nettoyage final fichiers temporaires…")

                # Nettoyages divers
                self.cleanup_unaligned_files()
                self.cleanup_temp_reference()
                self._cleanup_drizzle_temp_files()
                self._cleanup_drizzle_batch_outputs()
                self._cleanup_mosaic_panel_stacks_temp()

                # Suppression des memmaps SUM/WHT
                print("DEBUG [_worker FINALLY]: Nettoyage fichiers memmap .npy…")
                if (
                    hasattr(self, "sum_memmap_path")
                    and self.sum_memmap_path
                    and os.path.exists(self.sum_memmap_path)
                ):
                    try:
                        os.remove(self.sum_memmap_path)
                        print("   -> Fichier SUM.npy supprimé.")
                    except Exception as e_del_sum:
                        print(f"   -> WARN: Erreur suppression SUM.npy: {e_del_sum}")

                if (
                    hasattr(self, "wht_memmap_path")
                    and self.wht_memmap_path
                    and os.path.exists(self.wht_memmap_path)
                ):
                    try:
                        os.remove(self.wht_memmap_path)
                        print("   -> Fichier WHT.npy supprimé.")
                    except Exception as e_del_wht:
                        print(f"   -> WARN: Erreur suppression WHT.npy: {e_del_wht}")

                # Efface le dossier memmap s’il est vide
                memmap_dir = os.path.join(self.output_folder, "memmap_accumulators")
                try:
                    if os.path.isdir(memmap_dir) and not os.listdir(memmap_dir):
                        os.rmdir(memmap_dir)
                        print(f"   -> Dossier memmap vide supprimé: {memmap_dir}")
                except Exception as e_rmdir:
                    print(
                        f"   -> INFO: Erreur suppression dossier memmap vide: {e_rmdir}"
                    )
            else:
                self.update_progress(
                    "ⓘ Fichiers temporaires et memmap conservés."
                )

            # Purge finale des listes et GC
            print("   -> Vidage listes et GC…")
            self.current_batch_data = []
            local_drizzle_final_batch_data = []
            self.intermediate_drizzle_batch_files = []
            local_batch_temp_files = []
            all_aligned_files_with_info = []
            self._close_memmaps()
            gc.collect()

            # Flag activité
            self.processing_active = False
            print("DEBUG [_worker]: Flag processing_active mis à False.")
            self.update_progress("🚪 Thread traitement terminé.")





############################################################################################################################





    def _update_header_for_drizzle_final(self):
        """
        Crée et retourne un header FITS pour le stack final en mode Drizzle "Final".
        """
        print("DEBUG QM [_update_header_for_drizzle_final]: Création du header pour Drizzle Final...")
        
        final_header = fits.Header()

        # 1. Copier les informations de base du header de référence (si disponible)
        if self.reference_header_for_wcs:
            print("DEBUG QM [_update_header_for_drizzle_final]: Copie des clés depuis reference_header_for_wcs...")
            # Liste des clés FITS standard et utiles à copier depuis une brute/référence
            keys_to_copy_from_ref = [
                'INSTRUME', 'TELESCOP', 'OBSERVER', 'OBJECT', 
                'DATE-OBS', 'TIME-OBS', # Ou juste DATE-OBS si TIME-OBS n'est pas toujours là
                'EXPTIME',  # L'exposition d'une brute individuelle
                'FILTER', 'BAYERPAT', 'XBAYROFF', 'YBAYROFF',
                'GAIN', 'OFFSET', 'CCD-TEMP', 'READMODE',
                'FOCALLEN', 'APERTURE', 'PIXSIZE', 'XPIXSZ', 'YPIXSZ', # Infos optiques
                'SITELAT', 'SITELONG', 'SITEELEV' # Infos site
            ]
            for key in keys_to_copy_from_ref:
                if key in self.reference_header_for_wcs:
                    try:
                        # Essayer de copier avec le commentaire
                        final_header[key] = (self.reference_header_for_wcs[key], 
                                             self.reference_header_for_wcs.comments[key])
                    except KeyError: # Si pas de commentaire, copier juste la valeur
                        final_header[key] = self.reference_header_for_wcs[key]
                    except Exception as e_copy:
                        print(f"DEBUG QM [_update_header_for_drizzle_final]: Erreur copie clé '{key}': {e_copy}")
        else:
            print("DEBUG QM [_update_header_for_drizzle_final]: reference_header_for_wcs non disponible.")

        # 2. Ajouter/Mettre à jour les informations spécifiques au Drizzle Final
        final_header['STACKTYP'] = (f'Drizzle Final ({self.drizzle_scale:.0f}x)', 'Stacking method with Drizzle')
        final_header['DRZSCALE'] = (self.drizzle_scale, 'Drizzle final scale factor')
        final_header['DRZKERNEL'] = (self.drizzle_kernel, 'Drizzle kernel used')
        final_header['DRZPIXFR'] = (self.drizzle_pixfrac, 'Drizzle pixfrac used')
        final_header['DRZMODE'] = ('Final', 'Drizzle combination mode') # Spécifique pour ce header

        # NIMAGES et TOTEXP seront mis à jour dans _save_final_stack avec les valeurs finales
        # mais on peut mettre une estimation ici si self.aligned_files_count est déjà pertinent
        if hasattr(self, 'aligned_files_count') and self.aligned_files_count > 0:
            final_header['NINPUTS'] = (self.aligned_files_count, 'Number of aligned images input to Drizzle')
            # Pour TOTEXP, il faudrait multiplier aligned_files_count par l'EXPTIME moyen
            # Laissons _save_final_stack gérer le TOTEXP final pour plus de précision.

        # 3. Informations générales
        final_header['CREATOR'] = ('SeestarStacker (Queued)', 'Processing Software')
        final_header['HISTORY'] = 'Final Drizzle image created by SeestarStacker'
        if self.correct_hot_pixels:
            final_header['HISTORY'] = 'Hot pixel correction applied to input frames'
        if self.use_quality_weighting: # Le Drizzle actuel ne prend pas en compte ces poids directement
            final_header['HISTORY'] = 'Quality weighting parameters were set, but Drizzle uses its own weighting.'
        
        # Le WCS sera ajouté par _save_final_stack à partir du self.drizzle_output_wcs

        print("DEBUG QM [_update_header_for_drizzle_final]: Header pour Drizzle Final créé.")
        return final_header





############################################################################################################################


    # --- MÉTHODE DE NETTOYAGE ---
    def _cleanup_mosaic_panel_stacks_temp(self):
        """Supprime le dossier contenant les stacks de panneaux temporaires."""
        panel_stacks_dir = os.path.join(self.output_folder, "mosaic_panel_stacks_temp")
        if panel_stacks_dir and os.path.isdir(panel_stacks_dir):
            try:
                shutil.rmtree(panel_stacks_dir)
                self.update_progress(f"🧹 Dossier stacks panneaux temp. supprimé: {os.path.basename(panel_stacks_dir)}")
            except Exception as e:
                self.update_progress(f"⚠️ Erreur suppression dossier stacks panneaux temp.: {e}")





###################################################################################################################




    def _finalize_mosaic_processing(self, aligned_files_info_list):
        """
        Effectue la combinaison finale Drizzle pour le mode mosaïque.
        MAJ: Corrige import et UnboundLocalError.
        """
        num_files_to_mosaic = len(aligned_files_info_list)
        print(f"DEBUG (Backend _finalize_mosaic_processing): Début finalisation pour {num_files_to_mosaic} images.")
        self.update_progress(f"🖼️ Préparation assemblage mosaïque final ({num_files_to_mosaic} images)...")

        # ... (Vérifications initiales num_files, Drizzle disponible - inchangées) ...
        if num_files_to_mosaic < 2: self.update_progress("⚠️ Moins de 2 images."); self.final_stacked_path = None; self.processing_error = "..."; return
        if not _OO_DRIZZLE_AVAILABLE or Drizzle is None: error_msg = "..."; self.update_progress(f"❌ {error_msg}"); self.processing_error = error_msg; self.final_stacked_path = None; return

# --- Calcul Grille Finale ---
        print("DEBUG (Backend _finalize_mosaic_processing): Appel _calculate_final_mosaic_grid...")
        input_wcs_list = [item[1] for item in aligned_files_info_list if item[1] is not None]
        mosaic_output_wcs, mosaic_output_shape_hw = self._calculate_final_mosaic_grid(input_wcs_list)
        if mosaic_output_wcs is None or mosaic_output_shape_hw is None:
            error_msg = "Échec calcul grille sortie."
            self.update_progress(f"❌ {error_msg}")
            self.processing_error = error_msg
            self.final_stacked_path = None
            return
        print(f"DEBUG (Backend _finalize_mosaic_processing): Grille Mosaïque -> Shape={mosaic_output_shape_hw} (H,W)")

        # --- Initialiser Drizzle Final ---
        num_output_channels = 3
        final_drizzlers = []
        final_output_sci_list = []
        final_output_wht_list = []
        initialized = False
        try:
            print(f"  -> Initialisation Drizzle final pour {num_output_channels} canaux...")
            for _ in range(num_output_channels):
                out_img_ch = np.zeros(mosaic_output_shape_hw, dtype=np.float32)
                out_wht_ch = np.zeros(mosaic_output_shape_hw, dtype=np.float32)
                final_output_sci_list.append(out_img_ch)
                final_output_wht_list.append(out_wht_ch)
                driz_ch = Drizzle(out_img=out_img_ch, out_wht=out_wht_ch, out_shape=mosaic_output_shape_hw, out_wcs=mosaic_output_wcs, kernel=self.drizzle_kernel, fillval="0.0")
                final_drizzlers.append(driz_ch)
            initialized = True
            print("  -> Initialisation Drizzle final OK.")
        except Exception as init_err:
            print(f"  -> ERREUR init Drizzle Mosaïque: {init_err}")
            traceback.print_exc(limit=1)
            return

        if not initialized:
            return  # Sécurité

        # --- Boucle Drizzle sur les fichiers temporaires ---
        print(f"  -> Démarrage boucle Drizzle finale sur {num_files_to_mosaic} fichiers...")
        processed_count = 0
        # Utiliser enumerate pour obtenir l'index et le tuple (chemin, wcs)
        for i, (temp_fpath, wcs_in) in enumerate(aligned_files_info_list):
            if self.stop_processing:
                self.update_progress("🛑 Arrêt pendant assemblage final.")
                break
            if (i + 1) % 10 == 0 or i == 0 or i == len(aligned_files_info_list) - 1:
                print(f"    Adding Final Drizzle Input {i+1}/{num_files_to_mosaic}")

            # --- CORRECTION : Initialiser les variables locales à None ---
            img_data_hxwxc = None
            header_in = None
            pixmap = None
            wcs_to_use = None
            # --- FIN CORRECTION ---

            try:
                # Charger données et WCS (utilise la fonction importée)
                img_data_hxwxc, wcs_in_loaded, header_in = _load_drizzle_temp_file(temp_fpath)  # Appel Corrigé
                wcs_to_use = wcs_in_loaded if wcs_in_loaded else wcs_in  # Utiliser le WCS chargé ou celui de la liste

                if img_data_hxwxc is None or wcs_to_use is None:
                    print(f"    - Skip Input {i+1} (échec chargement/WCS)")
                    continue

                # Calcul Pixmap
                input_shape_hw = img_data_hxwxc.shape[:2]
                y_in, x_in = np.indices(input_shape_hw)
                world_coords = wcs_to_use.all_pix2world(x_in.flatten(), y_in.flatten(), 0)
                x_out, y_out = mosaic_output_wcs.all_world2pix(world_coords[0], world_coords[1], 0)
                pixmap = np.dstack((x_out.reshape(input_shape_hw), y_out.reshape(input_shape_hw))).astype(np.float32)

                # Ajout Drizzle
                exptime = 1.0  # ... (calcul exptime comme avant) ...
                if header_in and 'EXPTIME' in header_in:
                    try:
                        exptime = max(1e-6, float(header_in['EXPTIME']))
                    except (ValueError, TypeError):
                        pass

                for c in range(num_output_channels):
                    channel_data_2d = img_data_hxwxc[:, :, c].astype(np.float32)
                    finite_mask = np.isfinite(channel_data_2d)
                    channel_data_2d[~finite_mask] = 0.0
                    final_drizzlers[c].add_image(data=channel_data_2d, pixmap=pixmap, exptime=exptime, in_units='counts', pixfrac=self.drizzle_pixfrac)
                    processed_count += 1

            except Exception as e_add:
                print(f"    - ERREUR traitement/ajout input {i+1}: {e_add}")
                traceback.print_exc(limit=1)
            # --- CORRECTION : finally DANS la boucle ---
            finally:
                # Nettoyer les variables locales même si erreur DANS l'itération
                del img_data_hxwxc, wcs_in, header_in, pixmap, wcs_to_use
                if (i + 1) % 5 == 0:
                    gc.collect()
            # --- FIN CORRECTION ---
        # --- Fin Boucle Drizzle ---

        print(f"  -> Boucle assemblage terminée. {processed_count}/{num_files_to_mosaic} fichiers ajoutés.")
        if processed_count == 0:
            error_msg = "Aucun fichier traité avec succès."
            self.update_progress(f"❌ ERREUR: {error_msg}")
            self.processing_error = error_msg
            self.final_stacked_path = None
            return

        # --- Assemblage et Stockage Résultat ---
        try:
            print("  -> Assemblage final des canaux (Mosaïque)...")
            # ... (logique stack/save identique à l'étape précédente) ...
            final_mosaic_sci = np.stack(final_output_sci_list, axis=-1)
            final_mosaic_wht = np.stack(final_output_wht_list, axis=-1)
            print(f"  -> Combinaison terminée. Shape SCI: {final_mosaic_sci.shape}")
            self.current_stack_data = final_mosaic_sci
            self.current_stack_header = fits.Header()
            if mosaic_output_wcs:
                self.current_stack_header.update(mosaic_output_wcs.to_header(relax=True))
            if self.reference_header_for_wcs:
                keys_to_copy = ['INSTRUME', 'TELESCOP', ...]  # Veuillez compléter la liste des clés à copier
                [self.current_stack_header.set(k, self.reference_header_for_wcs[k]) for k in keys_to_copy if k in self.reference_header_for_wcs]
            self.current_stack_header['STACKTYP'] = (...)  # Veuillez compléter la valeur
            self.current_stack_header['DRZSCALE'] = (...)  # Veuillez compléter la valeur
            self.current_stack_header['DRZKERNEL'] = (...)  # Veuillez compléter la valeur
            self.current_stack_header['DRZPIXFR'] = (...)  # Veuillez compléter la valeur
            self.images_in_cumulative_stack = processed_count  # Utiliser le compte réel
            self.total_exposure_seconds = 0.0  # Recalcul approx
            if self.reference_header_for_wcs:
                single_exp = float(self.reference_header_for_wcs.get('EXPTIME', 10.0))
                self.total_exposure_seconds = processed_count * single_exp
            if final_mosaic_wht is not None:
                del final_mosaic_wht
                gc.collect()
            min_v, max_v = np.nanmin(self.current_stack_data), np.nanmax(self.current_stack_data)
            if max_v > min_v:
                self.current_stack_data = (self.current_stack_data - min_v) / (max_v - min_v)
            else:
                self.current_stack_data = np.zeros_like(self.current_stack_data)
            self.current_stack_data = np.clip(self.current_stack_data, 0.0, 1.0).astype(np.float32)
            self._save_final_stack(output_filename_suffix="_mosaic")

        except Exception as e:
            error_msg = f"Erreur finalisation/sauvegarde mosaïque: {e}"
            self.update_progress(f"❌ {error_msg}")
            traceback.print_exc(limit=3)
            self.processing_error = error_msg
            self.final_stacked_path = None

        print("DEBUG (Backend _finalize_mosaic_processing): Fin.")


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
        alignement, calcul qualité, et retourne WCS **GÉNÉRÉ** (fallback).
        MAJ: Suppression de l'appel au plate-solver.

        Args:
            file_path (str): Chemin complet du fichier FITS à traiter.
            reference_image_data (np.ndarray): Données de l'image de référence.

        Returns:
            tuple: (aligned_data, header, quality_scores, generated_wcs_object)
                   Le WCS retourné est maintenant toujours celui généré depuis le header.
        """
        file_name = os.path.basename(file_path)
        quality_scores = {'snr': 0.0, 'stars': 0.0}
        print(f"DEBUG [ProcessFile]: Start processing '{file_name}'")
        header = None
        prepared_img = None
        wcs_generated = None # WCS généré depuis header
        # final_wcs_object = None # Plus besoin de cette variable ici

        try:
            # 1. Charger et valider
            img_data = load_and_validate_fits(file_path)
            if img_data is None: raise ValueError("Échec chargement/validation.")
            header = fits.getheader(file_path)

            # 2. Vérification variance
            std_dev = np.std(img_data); variance_threshold = 0.0015
            if std_dev < variance_threshold: raise ValueError(f"Faible variance: {std_dev:.4f}")

            # 3. Pré-traitement (Debayer, WB Auto, HP)
            prepared_img = img_data.astype(np.float32)
            is_color_after_processing = False
            # ... (Logique Debayer, WB Auto, HP identique à avant) ...
            # Debayering
            if prepared_img.ndim == 2: # Debayer
                bayer = header.get('BAYERPAT', self.bayer_pattern); pattern_upper = bayer.upper() if isinstance(bayer, str) else 'GRBG'
                if pattern_upper in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                    # print(f"   -> Debayering {file_name} ({pattern_upper})...") # Log moins verbeux
                    try: prepared_img = debayer_image(prepared_img, pattern_upper); is_color_after_processing = True
                    except ValueError as de: print(f"   ⚠️ Erreur debayer: {de}. N&B.")
                # else: print(f"   -> N&B ou pattern Bayer inconnu ('{bayer}').")
            elif prepared_img.ndim == 3 and prepared_img.shape[2] == 3: is_color_after_processing = True #; print(f"   -> {file_name} déjà couleur.")
            else: raise ValueError(f"Shape inattendue ({prepared_img.shape}).")

            # WB Auto
            if is_color_after_processing:
                # print(f"   -> Calcul WB auto {file_name}...") # Log moins verbeux
                try: # ... (Logique WB Auto identique) ...
                    _mn_r, med_R, _sd_r = sigma_clipped_stats(prepared_img[..., 0], ...); _mn_g, med_G, _sd_g = sigma_clipped_stats(prepared_img[..., 1], ...); _mn_b, med_B, _sd_b = sigma_clipped_stats(prepared_img[..., 2], ...);
                    R_fac, B_fac = 1.0, 1.0; # ... (calcul facteurs) ...; prepared_img[..., 0] *= R_fac; prepared_img[..., 2] *= B_fac; prepared_img = np.clip(prepared_img, 0.0, 1.0)
                except Exception as wb_err: print(f"      - ERREUR WB Auto: {wb_err}")

            # HP Correction
            if self.correct_hot_pixels:
                 # print(f"   -> Correction HP {file_name}...") # Log moins verbeux
                 try: prepared_img = detect_and_correct_hot_pixels(prepared_img, self.hot_pixel_threshold, self.neighborhood_size)
                 except Exception as hp_err: print(f"   ⚠️ Erreur correction HP: {hp_err}.")

            prepared_img = prepared_img.astype(np.float32) # Assurer float32


            # --- 4. Génération WCS (TOUJOURS nécessaire pour groupement/Drizzle) ---
            #    On le fait AVANT l'alignement astroalign pour utiliser le header original.
            wcs_generated = None # Réinitialiser
            if header:
                print(f"   -> Génération WCS initial pour {file_name}...")
                try: # Essayer WCS(header)
                     with warnings.catch_warnings(): warnings.simplefilter('ignore'); wcs_hdr = WCS(header, naxis=2)
                     if wcs_hdr.is_celestial: wcs_generated = wcs_hdr
                except Exception: pass
                if wcs_generated is None: wcs_generated = _create_wcs_from_header(header) # Essayer génération

                if wcs_generated and wcs_generated.is_celestial:
                     naxis1_h = header.get('NAXIS1'); naxis2_h = header.get('NAXIS2')
                     if naxis1_h and naxis2_h: wcs_generated.pixel_shape = (naxis1_h, naxis2_h)
                     if wcs_generated.pixel_shape is None: print(f"      - WARNING: WCS généré {file_name} sans pixel_shape.")
                     print(f"      - WCS généré OK.")
                else: # Échec total WCS
                     print(f"      - ERREUR: WCS non trouvé/généré pour {file_name}.")
                     # Si WCS est requis (Drizzle ou Mosaïque), lever une erreur
                     if self.is_mosaic_run or self.drizzle_active_session: raise ValueError("WCS requis mais non obtenu.")
            else: # Pas de header
                 print(f"      - WARNING: Header original manquant pour WCS {file_name}.")
                 if self.is_mosaic_run or self.drizzle_active_session: raise ValueError("Header manquant, WCS requis.")
            # --- FIN Génération WCS ---


            # --- 5. Alignement Astroalign ---
            #    Utilise l'image pré-traitée et la référence globale
            print(f"   -> Alignement Astroalign {file_name}...")
            aligned_img, align_success = self.aligner._align_image(prepared_img, reference_image_data, file_name)
            if not align_success: raise RuntimeError(f"ÉCHEC Alignement Astroalign {file_name}")
            print(f"      - Alignement Astroalign OK")

            # --- 6. Correction Chroma (sur image alignée) ---
            # ### MODIFICATION : Appel à ChromaticBalancer DÉPLACÉ à _save_final_stack ###
            # # if is_color_after_processing and aligned_img is not None and self.apply_chroma_correction:
            # #      print(f"   -> Correction Chroma {file_name}...") # Ancien log
            # #      try:
            # #          if hasattr(self, 'chroma_balancer') and self.chroma_balancer:
            # #               aligned_img = self.chroma_balancer.normalize_stack(aligned_img)
            # #          else:
            # #               print(f"   AVERTISSEMENT: Instance ChromaticBalancer non trouvée dans _process_file pour {file_name}")
            # #      except Exception as chroma_err:
            # #           print(f"      - ERREUR Correction Chroma dans _process_file pour {file_name}: {chroma_err}")
            print(f"DEBUG [ProcessFile]: Correction Chroma (Edge Enhance) IGNORÉE dans _process_file pour {file_name} (sera faite à la fin).")
            # ### FIN MODIFICATION ###

            # --- 7. Calcul Qualité (sur image alignée) ---
            if self.use_quality_weighting:
                quality_scores = self._calculate_quality_metrics(aligned_img)


            print(f"DEBUG [ProcessFile]: Finished '{file_name}'. Returning WCS: Generated")
            # Retourner l'image alignée, header original, scores, et WCS GÉNÉRÉ
            return aligned_img, header, quality_scores, wcs_generated # Utiliser wcs_generated

        # --- Gestion Erreurs ---
        except (ValueError, RuntimeError) as proc_err: # Erreurs attendues
            self.update_progress(f"   ⚠️ {file_name} ignoré: {proc_err}")
            self.skipped_files_count += 1
            if file_path and os.path.exists(file_path):
                try: shutil.move(...) # Déplacer vers skipped
                except Exception: pass
            return None, None, quality_scores, None

        except Exception as e: # Erreurs inattendues
            self.update_progress(f"❌ Erreur traitement fichier {file_name}: {e}")
            traceback.print_exc(limit=3); self.skipped_files_count += 1
            if file_path and os.path.exists(file_path):
                try: shutil.move(...) # Déplacer vers error
                except Exception: pass
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
            print("DEBUG QM [_process_completed_batch]: Appel à _update_preview_sum_w après accumulation lot classique...") # Debug
            self._update_preview_sum_w()
            ### MODIFICATION : Supprimer ou commenter l'appel à la sauvegarde intermédiaire ###
            # La sauvegarde intermédiaire n'a plus vraiment de sens avec SUM/W et est coûteuse.
            # Sauvegarder le stack intermédiaire (cumulatif)
            #self._save_intermediate_stack()
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
        [MODE SUM/W - DRIZZLE INCR] Traite un batch pour le Drizzle Incrémental :
        1. Appelle DrizzleProcessor sur les fichiers temporaires du lot pour obtenir SCI et WHT du lot.
        2. Accumule (SCI_lot * WHT_lot) dans cumulative_sum_memmap.
        3. Accumule WHT_lot dans cumulative_wht_memmap.
        4. Nettoie les fichiers temporaires du lot.
        """
        print(f"DEBUG QM [_process_incremental_drizzle_batch SUM/W]: Début traitement Drizzle Incr. Lot #{current_batch_num}...") # Debug

        if not batch_temp_filepaths:
            self.update_progress(f"⚠️ Tentative de traiter un batch Drizzle incrémental vide (Lot #{current_batch_num}).")
            print("DEBUG QM [_process_incremental_drizzle_batch SUM/W]: Sortie précoce (lot vide).") # Debug
            return

        num_files_in_batch = len(batch_temp_filepaths)
        progress_info = f"(Lot {current_batch_num}/{total_batches_est if total_batches_est > 0 else '?'})"
        self.update_progress(f"💧 Traitement Drizzle incrémental du batch {progress_info} ({num_files_in_batch} fichiers)...")

        # --- Vérifications Memmap ---
        if self.cumulative_sum_memmap is None or self.cumulative_wht_memmap is None or self.memmap_shape is None:
             self.update_progress("❌ Erreur critique: Accumulateurs Memmap SUM/WHT non initialisés pour Drizzle Incr.")
             print("ERREUR QM [_process_incremental_drizzle_batch SUM/W]: Memmap non initialisé.") # Debug
             self.processing_error = "Memmap non initialisé (Drizzle Incr)"
             self.stop_processing = True
             return

        # --- 1. Appeler Drizzle sur le lot courant ---
        drizzle_result_batch_sci = None # Image science normalisée (Counts/Sec ou équivalent)
        wht_map_batch = None          # Carte de poids du lot
        drizzle_proc = None           # Référence à l'instance DrizzleProcessor

        try:
            # --- Import Tardif (sécurité, même si déjà fait dans _worker) ---
            try: from ..enhancement.drizzle_integration import DrizzleProcessor
            except ImportError: raise RuntimeError("DrizzleProcessor non importable.")

            print("DEBUG QM [_process_incremental_drizzle_batch SUM/W]: Instanciation DrizzleProcessor...") # Debug
            drizzle_proc = DrizzleProcessor(
                scale_factor=self.drizzle_scale,
                pixfrac=self.drizzle_pixfrac,
                kernel=self.drizzle_kernel
            )

            # --- Déterminer la grille de sortie si pas encore fait ---
            # (Normalement fait au début du worker, mais sécurité)
            if self.drizzle_output_wcs is None or self.drizzle_output_shape_hw is None:
                 print("DEBUG QM [_process_incremental_drizzle_batch SUM/W]: Grille Drizzle non définie, tentative de création...") # Debug
                 if self.reference_wcs_object is None or self.memmap_shape is None:
                     raise RuntimeError("WCS ou Shape référence manquant pour créer grille Drizzle.")
                 # Utiliser la shape H,W du memmap (qui vient de la réf)
                 ref_shape_for_grid_hw = self.memmap_shape[:2]
                 self.drizzle_output_wcs, self.drizzle_output_shape_hw = self._create_drizzle_output_wcs(
                     self.reference_wcs_object, ref_shape_for_grid_hw, self.drizzle_scale
                 )
                 print(f"DEBUG QM [_process_incremental_drizzle_batch SUM/W]: Grille Drizzle créée : {self.drizzle_output_shape_hw}") # Debug

            # --- Vérifier compatibilité shape sortie memmap vs grille Drizzle ---
            # WHT memmap est (H,W), SUM est (H,W,C)
            # La sortie Drizzle sera (H,W,C) pour SCI et WHT après stack des canaux
            if self.drizzle_output_shape_hw != self.memmap_shape[:2]:
                 raise RuntimeError(f"Incompatibilité Shape Drizzle ({self.drizzle_output_shape_hw}) et Memmap ({self.memmap_shape[:2]})")

            print(f"DEBUG QM [_process_incremental_drizzle_batch SUM/W]: Appel DrizzleProcessor.apply_drizzle pour lot #{current_batch_num}...") # Debug
            # Utiliser le WCS et Shape de sortie définis pour Drizzle
            drizzle_result_batch_sci, wht_map_batch = drizzle_proc.apply_drizzle(
                batch_temp_filepaths,
                output_wcs=self.drizzle_output_wcs,
                output_shape_2d_hw=self.drizzle_output_shape_hw
            )

            if drizzle_result_batch_sci is None:
                 raise RuntimeError(f"Échec Drizzle sur le lot {progress_info}.")
            if wht_map_batch is None:
                 # Note: apply_drizzle devrait toujours retourner un wht map s'il retourne sci
                 print(f"AVERTISSEMENT QM [_process_incremental_drizzle_batch SUM/W]: Carte WHT non retournée pour le lot {progress_info}. Tentative avec poids=1.")
                 wht_map_batch = np.ones_like(drizzle_result_batch_sci, dtype=np.float32) # Fallback très simple

            self.update_progress(f"   -> Drizzle lot {progress_info} terminé (Shape SCI: {drizzle_result_batch_sci.shape}, WHT: {wht_map_batch.shape})")
            print(f"DEBUG QM [_process_incremental_drizzle_batch SUM/W]: Drizzle lot OK. SCI Range: [{np.nanmin(drizzle_result_batch_sci):.3f}, {np.nanmax(drizzle_result_batch_sci):.3f}], WHT Range: [{np.nanmin(wht_map_batch):.1f}, {np.nanmax(wht_map_batch):.1f}]") # Debug

        except Exception as e:
            self.update_progress(f"❌ Erreur Drizzle sur lot {progress_info}: {e}")
            print(f"ERREUR QM [_process_incremental_drizzle_batch SUM/W]: Échec Drizzle lot: {e}") # Debug
            traceback.print_exc(limit=2)
            self._cleanup_batch_temp_files(batch_temp_filepaths)
            self.failed_stack_count += num_files_in_batch
            return # Ne pas tenter d'accumuler

        # --- 2. Accumuler dans SUM et WHT ---
        try:
            self.update_progress(f"   -> Accumulation Drizzle lot {progress_info} (SUM/W)...")
            print("DEBUG QM [_process_incremental_drizzle_batch SUM/W]: Début accumulation memmap...") # Debug

            # S'assurer que les données sont en float32/64
            sci_batch_float = drizzle_result_batch_sci.astype(np.float64) # Utiliser float64 pour multiplication
            wht_batch_float = wht_map_batch.astype(np.float64)

            # Nettoyer les poids (doivent être >= 0)
            wht_batch_float[~np.isfinite(wht_batch_float)] = 0.0
            wht_batch_float = np.maximum(wht_batch_float, 0.0)

            # Calculer le signal pondéré pour ce lot: SCI * WHT
            weighted_signal_batch = sci_batch_float * wht_batch_float
            print(f"DEBUG QM [_process_incremental_drizzle_batch SUM/W]: Signal pondéré calculé. Range: [{np.nanmin(weighted_signal_batch):.3f}, {np.nanmax(weighted_signal_batch):.3f}]") # Debug

            # --- Accumulation SUM ---
            print("DEBUG QM [_process_incremental_drizzle_batch SUM/W]: Addition à cumulative_sum_memmap...") # Debug
            self.cumulative_sum_memmap[:] += weighted_signal_batch.astype(self.memmap_dtype_sum)
            if hasattr(self.cumulative_sum_memmap, 'flush'): self.cumulative_sum_memmap.flush()
            print("DEBUG QM [_process_incremental_drizzle_batch SUM/W]: Addition SUM terminée et flushée.") # Debug

            # --- Accumulation WHT ---
            # La carte de poids WHT est HxWxC, mais notre WHT memmap est HxW.
            # On doit sommer les poids des 3 canaux pour obtenir le poids total par pixel.
            # Ou utiliser le poids d'un seul canal si on suppose qu'ils sont similaires ?
            # Plus sûr: Sommer les poids des canaux.
            wht_batch_sum_channels = np.sum(wht_batch_float, axis=2)
            print(f"DEBUG QM [_process_incremental_drizzle_batch SUM/W]: Addition à cumulative_wht_memmap (somme des canaux WHT)...") # Debug
            self.cumulative_wht_memmap[:] += wht_batch_sum_channels.astype(self.memmap_dtype_wht)
            if hasattr(self.cumulative_wht_memmap, 'flush'): self.cumulative_wht_memmap.flush()
            print(f"DEBUG QM [_process_incremental_drizzle_batch SUM/W]: Addition WHT terminée et flushée.") # Debug

            # --- Mise à jour compteurs globaux ---
            # Pour Drizzle, le nombre d'images ajoutées est num_files_in_batch
            self.images_in_cumulative_stack += num_files_in_batch
            # Exposition : essayer de lire depuis le premier header du lot temp
            try:
                 first_hdr_batch = fits.getheader(batch_temp_filepaths[0])
                 exp_time_batch = float(first_hdr_batch.get('EXPTIME', 0.0))
                 self.total_exposure_seconds += num_files_in_batch * exp_time_batch
            except Exception: pass
            print(f"DEBUG QM [_process_incremental_drizzle_batch SUM/W]: Compteurs mis à jour: images={self.images_in_cumulative_stack}, exp={self.total_exposure_seconds:.1f}") # Debug


            # --- Mise à jour Header Cumulatif (Minimale ici) ---
            if self.current_stack_header is None: # Initialiser si premier lot Drizzle
                self.current_stack_header = fits.Header()
                # Copier infos Drizzle depuis l'output WCS si possible
                if self.drizzle_output_wcs:
                     try: self.current_stack_header.update(self.drizzle_output_wcs.to_header(relax=True))
                     except Exception as e_hdr: print(f"WARN: Erreur copie WCS header: {e_hdr}")
                # Copier quelques infos de base
                if self.reference_header_for_wcs:
                    keys_to_copy = ['INSTRUME', 'TELESCOP', 'OBJECT', 'FILTER', 'DATE-OBS']
                    for key in keys_to_copy:
                         if key in self.reference_header_for_wcs: self.current_stack_header[key] = self.reference_header_for_wcs[key]
                self.current_stack_header['STACKTYP'] = (f'Drizzle Incr SUM/W ({self.drizzle_scale:.0f}x)', 'Incremental Drizzle SUM/W')
                self.current_stack_header['DRZSCALE'] = (self.drizzle_scale, 'Drizzle scale factor')
                self.current_stack_header['DRZKERNEL'] = (self.drizzle_kernel, 'Drizzle kernel used')
                self.current_stack_header['DRZPIXFR'] = (self.drizzle_pixfrac, 'Drizzle pixfrac used')
                self.current_stack_header['CREATOR'] = ('SeestarStacker (SUM/W)', 'Processing Software')
                self.current_stack_header['HISTORY'] = 'Drizzle SUM/W Accumulation Initialized'
                if self.correct_hot_pixels: self.current_stack_header['HISTORY'] = 'Hot pixel correction applied'

            # Mettre à jour NIMAGES/TOTEXP
            self.current_stack_header['NIMAGES'] = (self.images_in_cumulative_stack, 'Images accumulated in Drizzle SUM/W')
            self.current_stack_header['TOTEXP'] = (round(self.total_exposure_seconds, 2), '[s] Approx exposure accumulated')

            self.update_progress(f"   -> Accumulation lot {progress_info} terminée.")

            # --- Mettre à jour l'aperçu ---
            # Utilise une nouvelle méthode qui lira SUM/W et fera la division
            print("DEBUG QM [_process_incremental_drizzle_batch SUM/W]: Appel _update_preview_sum_w...") # Debug
            self._update_preview_sum_w() # Nouvelle méthode d'aperçu pour SUM/W

        except MemoryError as mem_err:
             print(f"ERREUR QM [_process_incremental_drizzle_batch SUM/W]: ERREUR MÉMOIRE - {mem_err}") # Debug
             self.update_progress(f"❌ ERREUR MÉMOIRE lors de l'accumulation du batch Drizzle.")
             traceback.print_exc(limit=1)
             self.processing_error = "Erreur Mémoire Accumulation Drizzle"
             self.stop_processing = True
        except Exception as e:
            print(f"ERREUR QM [_process_incremental_drizzle_batch SUM/W]: Exception inattendue accumulation - {e}") # Debug
            self.update_progress(f"❌ Erreur combinaison Drizzle lot {progress_info}: {e}")
            traceback.print_exc(limit=2)
            self.failed_stack_count += num_files_in_batch

        # --- 3. Nettoyer les fichiers temporaires du lot ---
        if self.perform_cleanup:
             print(f"DEBUG QM [_process_incremental_drizzle_batch SUM/W]: Nettoyage fichiers temp lot #{current_batch_num}...") # Debug
             self._cleanup_batch_temp_files(batch_temp_filepaths)
        else:
             print(f"DEBUG QM [_process_incremental_drizzle_batch SUM/W]: Fichiers temp lot #{current_batch_num} conservés.") # Debug
             self.update_progress(f"   -> Fichiers temporaires du lot {progress_info} conservés.")
        
        print(f"DEBUG QM [_process_incremental_drizzle_batch SUM/W]: Fin traitement lot #{current_batch_num}.") # Debug








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




# --- DANS LA CLASSE SeestarQueuedStacker DANS seestar/queuep/queue_manager.py ---

    def _combine_batch_result(self, stacked_batch_data_np, stack_info_header):
        """
        [MODE SUM/W - CLASSIQUE] Accumule le résultat numpy (float32, 0-1) d'un batch classique
        traité (par _stack_batch) dans les accumulateurs memmap SUM et WHT.

        NOTE: Pour cette version, la pondération par qualité est ignorée.
              Le poids accumulé est simplement le nombre d'images du batch.

        Args:
            stacked_batch_data_np (np.ndarray): Image MOYENNE (float32, 0-1) résultant du
                                                traitement du batch par _stack_batch.
            stack_info_header (fits.Header): En-tête contenant les informations
                                             sur le traitement de ce batch (NIMAGES, etc.).
        """
        print(f"DEBUG QM [_combine_batch_result SUM/W]: Début accumulation batch classique...") # Debug

        # --- Vérifications initiales ---
        if stacked_batch_data_np is None or stack_info_header is None:
            self.update_progress("⚠️ Erreur interne: Données batch invalides pour accumulation SUM/W.")
            print("DEBUG QM [_combine_batch_result SUM/W]: Sortie précoce (données invalides).") # Debug
            return
        if self.cumulative_sum_memmap is None or self.cumulative_wht_memmap is None or self.memmap_shape is None:
             self.update_progress("❌ Erreur critique: Accumulateurs Memmap SUM/WHT non initialisés.")
             print("ERREUR QM [_combine_batch_result SUM/W]: Memmap non initialisé.") # Debug
             # Marquer une erreur fatale ?
             self.processing_error = "Memmap non initialisé"
             self.stop_processing = True # Arrêter le traitement
             return
        if stacked_batch_data_np.shape != self.memmap_shape:
             self.update_progress(f"❌ Incompatibilité shape batch: Attendu {self.memmap_shape}, Reçu {stacked_batch_data_np.shape}. Accumulation échouée.")
             print(f"ERREUR QM [_combine_batch_result SUM/W]: Incompatibilité shape batch.") # Debug
             # Compter comme échec pour les images de ce batch ?
             try: batch_n_error = int(stack_info_header.get('NIMAGES', 1))
             except: batch_n_error = 1
             self.failed_stack_count += batch_n_error
             return

        try:
            batch_n = int(stack_info_header.get('NIMAGES', 1))
            batch_exposure = float(stack_info_header.get('TOTEXP', 0.0))

            if batch_n <= 0:
                self.update_progress(f"⚠️ Batch avec {batch_n} images, ignoré pour accumulation.")
                print(f"DEBUG QM [_combine_batch_result SUM/W]: Sortie précoce (batch_n <= 0).") # Debug
                return

            print(f"DEBUG QM [_combine_batch_result SUM/W]: Accumulation de {batch_n} images...") # Debug

            # --- Accumulation SUM ---
            # Puisque stacked_batch_data_np est la MOYENNE du lot,
            # le SIGNAL TOTAL du lot est Moyenne * N.
            # On utilise float64 pour l'accumulateur temporaire pour la précision.
            signal_total_batch = stacked_batch_data_np.astype(np.float64) * float(batch_n)

            # Ajouter au memmap SUM (float32). L'assignation gère la conversion.
            # On utilise += pour l'addition in-place sur le memmap.
            print("DEBUG QM [_combine_batch_result SUM/W]: Addition à cumulative_sum_memmap...") # Debug
            self.cumulative_sum_memmap[:] += signal_total_batch.astype(self.memmap_dtype_sum)
            # Forcer l'écriture sur disque (important pour memmap !)
            if hasattr(self.cumulative_sum_memmap, 'flush'):
                 self.cumulative_sum_memmap.flush()
            print("DEBUG QM [_combine_batch_result SUM/W]: Addition SUM terminée et flushée.") # Debug

            # --- Accumulation WHT ---
            # Le poids ici est simplement le nombre d'images.
            # On ajoute batch_n à chaque pixel du memmap WHT (uint16).
            print("DEBUG QM [_combine_batch_result SUM/W]: Addition à cumulative_wht_memmap...") # Debug
            self.cumulative_wht_memmap[:] += batch_n # L'addition gère le type uint16
            # Forcer l'écriture sur disque
            if hasattr(self.cumulative_wht_memmap, 'flush'):
                 self.cumulative_wht_memmap.flush()
            print("DEBUG QM [_combine_batch_result SUM/W]: Addition WHT terminée et flushée.") # Debug

            # --- Mise à jour des compteurs globaux ---
            # Note: images_in_cumulative_stack représente maintenant le *poids total* accumulé
            # Pour l'affichage UI, on veut peut-être toujours le nombre d'images *traitées*.
            # Gardons self.images_in_cumulative_stack pour le nombre d'images,
            # même si ce n'est plus directement le poids pour la moyenne finale.
            # La moyenne finale sera SUM / WHT.
            self.images_in_cumulative_stack += batch_n # Compte le nombre d'images traitées
            self.total_exposure_seconds += batch_exposure
            print(f"DEBUG QM [_combine_batch_result SUM/W]: Compteurs mis à jour: images={self.images_in_cumulative_stack}, exp={self.total_exposure_seconds:.1f}") # Debug

            # --- Mise à jour Header Cumulatif (Minimale ici) ---
            # On garde le header pour les métadonnées globales, mais NIMAGES/TOTEXP finaux
            # seront recalculés/vérifiés lors de la sauvegarde finale.
            if self.current_stack_header is None: # Initialiser si premier lot
                self.current_stack_header = fits.Header()
                # Copier quelques infos de base une seule fois
                first_header = stack_info_header # Ou récupérer header de la première image du lot si possible
                keys_to_copy = ['INSTRUME', 'TELESCOP', 'OBJECT', 'FILTER', 'DATE-OBS', 'GAIN', 'OFFSET', 'CCD-TEMP', 'RA', 'DEC', 'SITELAT', 'SITELONG', 'FOCALLEN', 'BAYERPAT']
                for key in keys_to_copy:
                    if first_header and key in first_header:
                        try: self.current_stack_header[key] = (first_header[key], first_header.comments[key] if key in first_header.comments else '')
                        except Exception: self.current_stack_header[key] = first_header[key]
                self.current_stack_header['STACKTYP'] = (f'Classic SUM/W ({self.stacking_mode})', 'Stacking method')
                self.current_stack_header['CREATOR'] = ('SeestarStacker (SUM/W)', 'Processing Software')
                if self.correct_hot_pixels: self.current_stack_header['HISTORY'] = 'Hot pixel correction applied'
                self.current_stack_header['HISTORY'] = 'SUM/W Accumulation Initialized'

            # Mettre à jour NIMAGES dans le header juste pour info (sera recalculé à la fin)
            self.current_stack_header['NIMAGES'] = (self.images_in_cumulative_stack, 'Images processed so far')
            self.current_stack_header['TOTEXP'] = (round(self.total_exposure_seconds, 2), '[s] Approx exposure accumulated')


            # Pas de clipping ici, on accumule les sommes. Le clipping se fera sur le résultat final.

            print("DEBUG QM [_combine_batch_result SUM/W]: Accumulation batch classique terminée.") # Debug

        except MemoryError as mem_err:
             print(f"ERREUR QM [_combine_batch_result SUM/W]: ERREUR MÉMOIRE - {mem_err}") # Debug
             self.update_progress(f"❌ ERREUR MÉMOIRE lors de l'accumulation du batch classique.")
             traceback.print_exc(limit=1)
             self.processing_error = "Erreur Mémoire Accumulation"
             self.stop_processing = True
        except Exception as e:
            print(f"ERREUR QM [_combine_batch_result SUM/W]: Exception inattendue - {e}") # Debug
            self.update_progress(f"❌ Erreur pendant l'accumulation du résultat du batch: {e}")
            traceback.print_exc(limit=3)
            # Ne pas arrêter le processus complet pour une erreur d'accumulation ? Ou si ?
            # Pour l'instant, on continue mais on log l'erreur.
            self.failed_stack_count += batch_n # Compter les images du lot comme échec d'accumulation

# --- FIN de _combine_batch_result MODIFIÉ ---




################################################################################################################################################
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

################################################################################################################################################


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
        # channel_names = ['R', 'G', 'B'] # Pas utilisé directement ici, mais bon à garder si logs plus détaillés
        final_drizzlers = []
        final_output_images = [] 
        final_output_weights = [] 

        try:
            self.update_progress(f"   -> Initialisation Drizzle final (Shape: {output_shape_2d_hw})...")
            for _ in range(num_output_channels):
                final_output_images.append(np.zeros(output_shape_2d_hw, dtype=np.float32))
                final_output_weights.append(np.zeros(output_shape_2d_hw, dtype=np.float32))
            for i in range(num_output_channels):
                driz_ch = Drizzle(
                    kernel=self.drizzle_kernel,
                    fillval="0.0",
                    out_img=final_output_images[i],
                    out_wht=final_output_weights[i]
                )
                final_drizzlers.append(driz_ch)
            self.update_progress(f"   -> Objets Drizzle finaux initialisés.")
        except Exception as init_err:
            self.update_progress(f"   - ERREUR: Échec init Drizzle final: {init_err}"); traceback.print_exc(limit=1)
            return None, None

        # --- Boucle sur les fichiers intermédiaires par lot ---
        total_contributing_ninputs = 0
        batches_combined_count = 0
        for i, (sci_fpath, wht_fpaths) in enumerate(intermediate_files_list):
            # ... (logique de la boucle identique à votre version précédente, jusqu'à la fin du try/except/finally interne à la boucle) ...
            if self.stop_processing: self.update_progress("🛑 Arrêt pendant combinaison lots Drizzle."); break
            self.update_progress(f"   -> Ajout lot intermédiaire {i+1}/{num_batches_to_combine}...")
            if len(wht_fpaths) != num_output_channels: self.update_progress(f"      -> ERREUR: Nombre incorrect de cartes poids pour lot {i+1}. Ignoré."); continue
            sci_data_chw = None; intermed_wcs = None; wht_maps = None; sci_header = None; combine_pixmap = None
            try:
                with fits.open(sci_fpath, memmap=False) as hdul_sci:
                    sci_data_chw = hdul_sci[0].data.astype(np.float32); sci_header = hdul_sci[0].header
                    try: total_contributing_ninputs += int(sci_header.get('NINPUTS', 0))
                    except (ValueError, TypeError): pass
                    with warnings.catch_warnings(): warnings.simplefilter("ignore"); intermed_wcs = WCS(sci_header, naxis=2)
                    if not intermed_wcs.is_celestial: raise ValueError("WCS intermédiaire non céleste.")
                    if sci_data_chw.ndim != 3 or sci_data_chw.shape[0] != num_output_channels: raise ValueError(f"Shape science invalide: {sci_data_chw.shape}")
                wht_maps = []; valid_weights = True
                for ch_idx, wht_fpath in enumerate(wht_fpaths):
                    try:
                        with fits.open(wht_fpath, memmap=False) as hdul_wht: wht_map = hdul_wht[0].data.astype(np.float32)
                        if wht_map.shape != sci_data_chw.shape[1:]: raise ValueError(f"Shape poids {wht_map.shape} != science HxW {sci_data_chw.shape[1:]}")
                        wht_map[~np.isfinite(wht_map)] = 0.0; wht_map[wht_map < 0] = 0.0; wht_maps.append(wht_map)
                    except Exception as e: self.update_progress(f"      -> ERREUR lecture poids {os.path.basename(wht_fpath)}: {e}. Lot ignoré."); valid_weights = False; break
                if not valid_weights: continue
                intermed_shape_hw = sci_data_chw.shape[1:]; y_intermed, x_intermed = np.indices(intermed_shape_hw)
                try:
                    world_coords_intermed = intermed_wcs.all_pix2world(x_intermed.flatten(), y_intermed.flatten(), 0)
                    x_final, y_final = output_wcs.all_world2pix(world_coords_intermed[0], world_coords_intermed[1], 0)
                    combine_pixmap = np.dstack((x_final.reshape(intermed_shape_hw), y_final.reshape(intermed_shape_hw))).astype(np.float32)
                except Exception as combine_map_err: self.update_progress(f"      -> ERREUR création pixmap combinaison: {combine_map_err}. Lot ignoré."); continue
                if combine_pixmap is not None:
                    for ch_index in range(num_output_channels):
                        channel_data_sci = sci_data_chw[ch_index, :, :]; channel_data_wht = wht_maps[ch_index]
                        channel_data_sci[~np.isfinite(channel_data_sci)] = 0.0
                        final_drizzlers[ch_index].add_image(data=channel_data_sci, pixmap=combine_pixmap, weight_map=channel_data_wht, exptime=1.0, pixfrac=self.drizzle_pixfrac, in_units='cps')
                    batches_combined_count += 1
                else: self.update_progress(f"      -> Warning: Pixmap combinaison est None pour lot {i+1}. Ignoré.")
            except FileNotFoundError: self.update_progress(f"   - ERREUR: Fichier intermédiaire lot {i+1} non trouvé. Ignoré."); continue
            except (IOError, ValueError) as e: self.update_progress(f"   - ERREUR lecture/validation lot intermédiaire {i+1}: {e}. Ignoré."); continue
            except Exception as e: self.update_progress(f"   - ERREUR traitement lot intermédiaire {i+1}: {e}"); traceback.print_exc(limit=1); continue
            finally:
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
            final_sci_image_hxwxc = np.stack(final_output_images, axis=-1).astype(np.float32)
            final_wht_map_hxwxc = np.stack(final_output_weights, axis=-1).astype(np.float32)

            # Appliquer la correction chromatique si activée et si l'image est en couleur
            if self.apply_chroma_correction and final_sci_image_hxwxc is not None:
                if final_sci_image_hxwxc.ndim == 3 and final_sci_image_hxwxc.shape[2] == 3: # Double check
                    self.update_progress("   -> Application de la correction chromatique sur résultat Drizzle...")
                    if hasattr(self, 'chroma_balancer') and self.chroma_balancer: # Vérifier que l'instance existe
                        final_sci_image_hxwxc = self.chroma_balancer.normalize_stack(final_sci_image_hxwxc)
                        self.update_progress("   -> Correction chromatique Drizzle terminée.")
                    else:
                        self.update_progress("   -> AVERTISSEMENT: Instance ChromaticBalancer non trouvée pour correction Drizzle.")

            ### MODIFICATION : Déplacer le return APRES le nettoyage et les logs ###

            # Nettoyer les résultats finaux (sécurité)
            final_sci_image_hxwxc[~np.isfinite(final_sci_image_hxwxc)] = 0.0
            final_wht_map_hxwxc[~np.isfinite(final_wht_map_hxwxc)] = 0.0
            final_wht_map_hxwxc[final_wht_map_hxwxc < 0] = 0.0

            self.update_progress(f"   -> Assemblage final Drizzle terminé (Shape Sci: {final_sci_image_hxwxc.shape}, Shape WHT: {final_wht_map_hxwxc.shape})")

            # Mettre à jour le compteur total d'images basé sur les headers intermédiaires
            self.images_in_cumulative_stack = total_contributing_ninputs
            print(f"DEBUG QM [_combine_intermediate_drizzle_batches]: images_in_cumulative_stack set to {self.images_in_cumulative_stack} from intermediate headers.")

            # Le return est maintenant à la fin du bloc try
            return final_sci_image_hxwxc, final_wht_map_hxwxc
            ### FIN MODIFICATION ###

        except Exception as e_final:
            self.update_progress(f"   - ERREUR pendant assemblage final Drizzle: {e_final}")
            traceback.print_exc(limit=1)
            # Le del et le return None, None sont déjà dans le finally implicite de la structure try/except globale
            return None, None # Assurer un retour en cas d'erreur ici
        finally: # Bloc finally pour le nettoyage des objets Drizzle
            del final_drizzlers, final_output_images, final_output_weights
            gc.collect()



############################################################################################################################################







    def _save_final_stack(self, output_filename_suffix: str = "", stopped_early: bool = False):
        """
        [MODE SUM/W] Calcule l'image finale depuis SUM/W, applique les post‑traitements,
        le rognage, sauvegarde le stack final et sa pré‑visualisation, puis libère les memmaps.
        """
        print(f"DEBUG QM [_save_final_stack SUM/W]: Début sauvegarde finale "
            f"(suffix: '{output_filename_suffix}', stopped_early: {stopped_early})")

        # ------------------------------------------------------------------ #
        # 0)  Imports optionnels (BN, SCNR, Edge‑Crop)                       #
        # ------------------------------------------------------------------ #
        neutralize_background_func = None
        apply_scnr_func            = None
        apply_edge_crop_func       = None

        try:
            from ..tools.stretch import neutralize_background_automatic as neutralize_background_func
        except ImportError:
            print("ERREUR QM [_save_final_stack SUM/W]: Échec import neutralize_background_automatic.")

        try:
            from ..enhancement.color_correction import apply_scnr as apply_scnr_func
        except ImportError:
            print("ERREUR QM [_save_final_stack SUM/W]: Échec import apply_scnr.")

        try:
            from ..enhancement.stack_enhancement import apply_edge_crop as apply_edge_crop_func
        except ImportError:
            print("ERREUR QM [_save_final_stack SUM/W]: Échec import apply_edge_crop.")

        # ... (Section 1: Sécurité - inchangée) ...
        if (self.cumulative_sum_memmap is None
                or self.cumulative_wht_memmap is None
                or self.output_folder is None):
            self.final_stacked_path = None
            print("DEBUG QM [_save_final_stack SUM/W]: Sortie précoce "
                "(memmap/output_folder non défini).")
            self.update_progress("ⓘ Aucun stack final (accumulateurs/dossier sortie invalide).")
            self._close_memmaps()
            return

        image_count = self.images_in_cumulative_stack
        if image_count <= 0 and not stopped_early:
            self.final_stacked_path = None
            print(f"DEBUG QM [_save_final_stack SUM/W]: Sortie précoce (image_count={image_count}).")
            self.update_progress("ⓘ Aucun stack final (0 images accumulées).")
            self._close_memmaps()
            return
        print(f"DEBUG QM [_save_final_stack SUM/W]: Image count = {image_count}")

        # ... (Section 2: Lecture memmaps & calcul stack final - inchangée) ...
        try:
            print("DEBUG QM [_save_final_stack SUM/W]: Lecture finale memmaps SUM/WHT…")
            final_sum = np.array(self.cumulative_sum_memmap, dtype=np.float64)
            final_wht = np.array(self.cumulative_wht_memmap, dtype=np.float64)
            self._close_memmaps()

            print("DEBUG QM [_save_final_stack SUM/W]: Calcul image moyenne (SUM/WHT)…")
            epsilon             = 1e-9
            wht_broadcasted     = np.maximum(final_wht, epsilon)[:, :, np.newaxis]
            with np.errstate(divide='ignore', invalid='ignore'):
                final_raw       = final_sum / wht_broadcasted
            final_raw           = np.nan_to_num(final_raw, nan=0.0, posinf=0.0, neginf=0.0)
            min_r, max_r        = np.nanmin(final_raw), np.nanmax(final_raw)
            if max_r > min_r:
                final_image     = (final_raw - min_r) / (max_r - min_r)
            else:
                final_image     = np.zeros_like(final_raw)
            final_image         = np.clip(final_image, 0.0, 1.0).astype(np.float32)
            del final_sum, final_wht, wht_broadcasted, final_raw
            gc.collect()
        except Exception as e_calc:
            print(f"ERREUR QM [_save_final_stack SUM/W]: Erreur calcul final SUM/W - {e_calc}")
            traceback.print_exc(limit=2)
            self.update_progress(f"❌ Erreur lors du calcul final SUM/W: {e_calc}")
            self.processing_error = f"Erreur Calcul Final: {e_calc}"
            self._close_memmaps() # Assurer la fermeture même si erreur
            return
        if final_image is None:
            self.final_stacked_path = None
            print("DEBUG QM [_save_final_stack SUM/W]: Échec calcul final SUM/W.")
            self.update_progress("ⓘ Aucun stack final (échec calcul SUM/W).")
            return

        data_to_save = final_image
        print("DEBUG QM [_save_final_stack SUM/W]: Données pour post‑traitement - "
            f"Shape: {data_to_save.shape}, Min: {np.nanmin(data_to_save):.3f}, "
            f"Max: {np.nanmax(data_to_save):.3f}")
        background_model_photutils = None

        # ------------------------------------------------------------------ #
        # 3)  Post‑traitements (Photutils BN, BN couleur, CB, SCNR, etc.)    #
        # ------------------------------------------------------------------ #
        print(f"DEBUG QM _save_final_stack: CHECK Photutils -> apply_flag={getattr(self, 'apply_photutils_bn', 'FLAG_NON_DEFINI')}, lib_available={_PHOTOUTILS_BG_SUB_AVAILABLE}")
        if getattr(self, 'apply_photutils_bn', False) and _PHOTOUTILS_BG_SUB_AVAILABLE:
            print("DEBUG QM [_save_final_stack SUM/W]: Appel subtract_background_2d (Photutils)…")
            self.update_progress("Soustraction de Fond 2D (Photutils)…", None)
            try:
                box   = getattr(self, 'photutils_bn_box_size', 128)
                filt  = getattr(self, 'photutils_bn_filter_size', 5)
                sigma = getattr(self, 'photutils_bn_sigma_clip', 3.0)
                excl  = getattr(self, 'photutils_bn_exclude_percentile', 98.0)
                data_corr, bkg_model = subtract_background_2d(data_to_save, box_size=box, filter_size=filt, sigma_clip_val=sigma, exclude_percentile=excl)
                if data_corr is not None:
                    data_to_save = data_corr; background_model_photutils = bkg_model
                    mn, mx = np.nanmin(data_to_save), np.nanmax(data_to_save)
                    print("DEBUG QM [_save_final_stack SUM/W]: Données après Photutils BN - Range avant re‑norm: [{mn:.3f}, {mx:.3f}]")
                    if mx > mn: data_to_save = (data_to_save - mn) / (mx - mn)
                    else: data_to_save = np.zeros_like(data_to_save)
                    data_to_save = np.clip(data_to_save, 0.0, 1.0).astype(np.float32)
                    self.update_progress("   -> Soustraction de Fond 2D terminée.", None)
                else: self.update_progress("⚠️ Échec Soustraction de Fond 2D, étape ignorée.", None)
            except Exception as photutils_err:
                print(f"ERREUR QM [_save_final_stack SUM/W]: Erreur pendant subtract_background_2d: {photutils_err}"); self.update_progress(f"⚠️ Erreur Soustraction Fond 2D: {photutils_err}. Étape ignorée.")
        elif getattr(self, 'apply_photutils_bn', False) and not _PHOTOUTILS_BG_SUB_AVAILABLE:
            self.update_progress("⚠️ Soustraction Fond 2D demandée mais Photutils indisponible.", None)

        if data_to_save.ndim == 3 and data_to_save.shape[2] == 3:
            if neutralize_background_func:
                print("DEBUG QM [_save_final_stack SUM/W]: Appel neutralize_background_automatic…")
                try:
                    rows, cols = (16, 16); parts = self.bn_grid_size_str.split('x')
                    if len(parts) == 2: rows, cols = int(parts[0]), int(parts[1])
                    data_to_save = neutralize_background_func(data_to_save, grid_size=(rows, cols), bg_percentile_low=self.bn_perc_low, bg_percentile_high=self.bn_perc_high, std_factor_threshold=self.bn_std_factor, min_applied_gain=self.bn_min_gain, max_applied_gain=self.bn_max_gain)
                except Exception as bn_err: print(f"ERREUR QM: Erreur BN: {bn_err}"); self.update_progress(f"⚠️ Erreur neutralisation: {bn_err}.")
            if self.apply_chroma_correction and hasattr(self, 'chroma_balancer'):
                print("DEBUG QM [_save_final_stack SUM/W]: Appel chroma_balancer.normalize_stack…")
                try:
                    cb = self.chroma_balancer; cb.border_size = self.cb_border_size; cb.blur_radius = self.cb_blur_radius
                    cb.r_factor_min = getattr(self, 'cb_min_r_factor', 0.7); cb.r_factor_max = getattr(self, 'cb_max_r_factor', 1.3)
                    cb.b_factor_min = self.cb_min_b_factor; cb.b_factor_max = self.cb_max_b_factor
                    data_to_save = cb.normalize_stack(data_to_save)
                except Exception as cb_err: print(f"ERREUR QM: Erreur chroma balancer: {cb_err}"); self.update_progress(f"⚠️ Erreur correction chroma: {cb_err}.")
            if self.apply_final_scnr and apply_scnr_func:
                print(f"DEBUG QM [_save_final_stack SUM/W]: Appel apply_scnr (final) Amount={self.final_scnr_amount}…")
                try: data_to_save = apply_scnr_func(data_to_save, target_channel=self.final_scnr_target_channel, amount=self.final_scnr_amount, preserve_luminosity=self.final_scnr_preserve_luminosity)
                except Exception as scnr_err: print(f"ERREUR QM: Erreur SCNR: {scnr_err}"); self.update_progress(f"⚠️ Erreur SCNR final: {scnr_err}.")

        # ------------------------------------------------------------------ #
        # 4)  Header FITS & éventuel rognage                                 #
        # ------------------------------------------------------------------ #
        final_header = self.current_stack_header.copy() if self.current_stack_header else fits.Header()
        final_header['NIMAGES'] = (image_count, 'Images contributing to final stack (SUM/W)')
        final_header['TOTEXP']  = (round(self.total_exposure_seconds, 2), '[s] Approx total exposure time (SUM/W)')
        stack_type_actual = final_header.get('STACKTYP', 'SUM_W_unknown')
        if 'SUM/W' not in stack_type_actual: stack_type_actual = f"{stack_type_actual} SUM/W"
        final_header['STACKTYP'] = (stack_type_actual, 'SUM/W based stacking method')

        # --- AJOUT DES PARAMÈTRES EXPERT AU HEADER ---
        #    (Doit être fait AVANT l'ajout du commentaire Photutils qui dépend de BN_GRID)
        print("DEBUG QM [_save_final_stack SUM/W]: Ajout des paramètres Expert au header FITS...")
        final_header.add_comment("--- Expert Settings ---")
        # BN
        final_header['BN_GRID'] = (str(self.bn_grid_size_str), "BN: Grid size (RxC)")
        final_header['BN_PLOW'] = (int(self.bn_perc_low), "BN: Background Percentile Low")
        final_header['BN_PHIGH'] = (int(self.bn_perc_high), "BN: Background Percentile High")
        final_header['BN_STD_F'] = (float(self.bn_std_factor), "BN: Background StdDev Factor")
        final_header['BN_MING'] = (float(self.bn_min_gain), "BN: Min Gain Applied")
        final_header['BN_MAXG'] = (float(self.bn_max_gain), "BN: Max Gain Applied")
        # CB
        final_header['CB_BORD'] = (int(self.cb_border_size), "CB: Border size (px)")
        final_header['CB_BLUR'] = (int(self.cb_blur_radius), "CB: Blur radius (px)")
        final_header['CB_MINBF'] = (float(self.cb_min_b_factor), "CB: Min Blue Factor")
        final_header['CB_MAXBF'] = (float(self.cb_max_b_factor), "CB: Max Blue Factor")
        # Crop
        final_header['CROP_PCT'] = (float(self.final_edge_crop_percent_decimal * 100.0), "Final Edge Crop (%)")
        # --- FIN AJOUT PARAMÈTRES EXPERT ---

        # --- AJOUT : Mots-clés Expert Photutils BN au Header ---
        if getattr(self, 'apply_photutils_bn', False) and _PHOTOUTILS_BG_SUB_AVAILABLE:
            # S'assurer que BN_GRID existe maintenant si on veut mettre le commentaire avant.
            # Si BN_GRID n'a pas été ajouté (ex: si ce n'est pas une image couleur),
            # on peut omettre `before` ou choisir un autre mot-clé de référence.
            # Pour plus de robustesse, on vérifie si BN_GRID existe avant de l'utiliser dans `before`.
            before_keyword_for_photutils_comment = 'BN_GRID' if 'BN_GRID' in final_header else None
            final_header.add_comment("--- Photutils Background Subtraction ---", before=before_keyword_for_photutils_comment)
            final_header['PB_APP'] = (True, "Photutils Background2D Applied")
            final_header['PB_BOX'] = (getattr(self, 'photutils_bn_box_size', 128), "Photutils: Box Size (px)")
            final_header['PB_FILT'] = (getattr(self, 'photutils_bn_filter_size', 5), "Photutils: Filter Size (px, odd)")
            final_header['PB_SIG'] = (getattr(self, 'photutils_bn_sigma_clip', 3.0), "Photutils: Sigma Clip value")
            final_header['PB_EXCP'] = (getattr(self, 'photutils_bn_exclude_percentile', 98.0), "Photutils: Exclude Brightest Percentile")

        if self.apply_final_scnr:
            final_header['SCNR_APP'] = (True, 'SCNR applied')
            final_header['SCNR_TRG'] = (self.final_scnr_target_channel, 'SCNR target')
            final_header['SCNR_AMT'] = (self.final_scnr_amount, 'SCNR amount')
            final_header['SCNR_LUM'] = (self.final_scnr_preserve_luminosity, 'SCNR lum preserved')

        if (apply_edge_crop_func and getattr(self, 'final_edge_crop_percent_decimal', 0.0) > 0.0):
            print("DEBUG QM [_save_final_stack SUM/W]: Rognage final demandé…")
            self.update_progress("Rognage final des bords…", None)
            try:
                before_shape = data_to_save.shape
                data_to_save = apply_edge_crop_func(data_to_save, self.final_edge_crop_percent_decimal)
                if data_to_save is None: # Si apply_edge_crop retourne None en cas d'erreur
                    self.update_progress("⚠️ Échec rognage, utilisation de l'image non rognée.", None)
                    # Il faudrait récupérer data_to_save d'avant l'appel à apply_edge_crop
                    # Pour l'instant, on ne fait rien, ce qui signifie que si crop échoue, on continue avec data_to_save=None
                    # ce qui causera une erreur à la sauvegarde. Mieux:
                    # data_to_save = final_image # Revenir à l'image avant tentative de crop
                    # Le plus simple est de s'assurer que apply_edge_crop retourne l'original si erreur.
                    # (Actuellement, il retourne l'original, donc data_to_save ne devrait pas être None ici)
                    if data_to_save is None and final_image is not None : data_to_save = final_image.copy() # Sécurité
                    else: self.update_progress("⚠️ Échec rognage, et image originale non disponible.", None); return # Ne pas sauvegarder si échec critique

            except Exception as crop_err:
                print(f"ERREUR QM: Erreur rognage: {crop_err}"); self.update_progress(f"⚠️ Erreur rognage: {crop_err}.")
                if final_image is not None : data_to_save = final_image.copy() # Revenir à l'original
                else: self.update_progress("⚠️ Échec rognage, et image originale non disponible.", None); return


        # ... (Sections 5, 6, 7: Nom de fichier, Sauvegarde FITS, Sauvegarde Preview - inchangées) ...
        stack_type_for_filename = "classic_sumw"
        if self.current_stack_header and 'STACKTYP' in self.current_stack_header:
            fn_part = str(final_header['STACKTYP']).split('(')[0].strip()
            fn_part = fn_part.replace(' ', '_').replace('/', '_').lower()
            stack_type_for_filename = f"{fn_part}_sumw"
        base_name = "stack_final"
        final_suffix_cleaned = f"{output_filename_suffix}".replace('_sumw', '')
        fits_path = os.path.join(self.output_folder, f"{base_name}_{stack_type_for_filename}{final_suffix_cleaned}.fit")
        preview_path  = os.path.splitext(fits_path)[0] + ".png"
        self.final_stacked_path = fits_path
        print(f"DEBUG QM [_save_final_stack SUM/W]: Chemin FITS final: {fits_path}")

        try:
            if (getattr(self, 'apply_photutils_bn', False) and background_model_photutils is not None and _PHOTOUTILS_BG_SUB_AVAILABLE): # Vérifier _PHOTOUTILS_BG_SUB_AVAILABLE
                primary_hdu = fits.PrimaryHDU(data_to_save.astype(np.float32), header=final_header)
                bkg_hdu_data = None
                if background_model_photutils.ndim == 3 and background_model_photutils.shape[2] == 3:
                    bkg_hdu_data = np.mean(background_model_photutils, axis=2).astype(np.float32)
                elif background_model_photutils.ndim == 2:
                    bkg_hdu_data = background_model_photutils.astype(np.float32)
                
                if bkg_hdu_data is not None:
                    bkg_hdu = fits.ImageHDU(bkg_hdu_data, name="BACKGROUND_MODEL")
                    fits.HDUList([primary_hdu, bkg_hdu]).writeto(fits_path, overwrite=True, checksum=True)
                else: # Fallback si bkg_model n'a pas pu être traité
                    save_fits_image(data_to_save, fits_path, final_header, overwrite=True)
            else:
                save_fits_image(data_to_save, fits_path, final_header, overwrite=True)
            print("DEBUG QM [_save_final_stack SUM/W]: Sauvegarde FITS OK.")
        except Exception as save_err:
            print(f"ERREUR QM: Échec sauvegarde FITS: {save_err}"); traceback.print_exc(limit=2)
            self.update_progress(f"⚠️ Erreur sauvegarde FITS: {save_err}"); return

        try:
            save_preview_image(data_to_save, preview_path, apply_stretch=True, enhanced_stretch=True)
            print("DEBUG QM [_save_final_stack SUM/W]: Sauvegarde Preview PNG OK.")
            self.update_progress(f"✅ Stack final SUM/W sauvegardé ({image_count} images)")
        except Exception as prev_err:
            print(f"ERREUR QM: Échec sauvegarde PNG: {prev_err}"); self.update_progress(f"⚠️ Erreur sauvegarde preview: {prev_err}")

        print("DEBUG QM [_save_final_stack SUM/W]: Fin méthode.")









#############################################################################################################################################################
    def _close_memmaps(self):
        """Ferme proprement les objets memmap s'ils existent."""
        print("DEBUG QM [_close_memmaps]: Tentative de fermeture des memmaps...")
        closed_sum = False
        if hasattr(self, 'cumulative_sum_memmap') and self.cumulative_sum_memmap is not None:
            try:
                # La documentation suggère que la suppression de la référence devrait suffire
                # mais un appel explicite à close() existe sur certaines versions/objets
                if hasattr(self.cumulative_sum_memmap, '_mmap') and self.cumulative_sum_memmap._mmap is not None:
                     self.cumulative_sum_memmap._mmap.close()
                # Supprimer la référence pour permettre la libération des ressources
                del self.cumulative_sum_memmap
                self.cumulative_sum_memmap = None
                closed_sum = True
                print("DEBUG QM [_close_memmaps]: Référence memmap SUM supprimée.")
            except Exception as e_close_sum:
                print(f"WARN QM [_close_memmaps]: Erreur fermeture/suppression memmap SUM: {e_close_sum}")
        
        closed_wht = False
        if hasattr(self, 'cumulative_wht_memmap') and self.cumulative_wht_memmap is not None:
            try:
                if hasattr(self.cumulative_wht_memmap, '_mmap') and self.cumulative_wht_memmap._mmap is not None:
                     self.cumulative_wht_memmap._mmap.close()
                del self.cumulative_wht_memmap
                self.cumulative_wht_memmap = None
                closed_wht = True
                print("DEBUG QM [_close_memmaps]: Référence memmap WHT supprimée.")
            except Exception as e_close_wht:
                print(f"WARN QM [_close_memmaps]: Erreur fermeture/suppression memmap WHT: {e_close_wht}")
        
        # Optionnel: Essayer de supprimer les fichiers .npy si le nettoyage est activé
        # Cela devrait être fait dans le bloc finally de _worker après l'appel à _save_final_stack
        # if self.perform_cleanup:
        #      if self.sum_memmap_path and os.path.exists(self.sum_memmap_path):
        #          try: os.remove(self.sum_memmap_path); print("DEBUG: Fichier SUM.npy supprimé.")
        #          except Exception as e: print(f"WARN: Erreur suppression SUM.npy: {e}")
        #      if self.wht_memmap_path and os.path.exists(self.wht_memmap_path):
        #          try: os.remove(self.wht_memmap_path); print("DEBUG: Fichier WHT.npy supprimé.")
        #          except Exception as e: print(f"WARN: Erreur suppression WHT.npy: {e}")

# --- FIN de _save_final_stack et ajout de _close_memmaps ---






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



################################################################################################################################################



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


################################################################################################################################################

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



################################################################################################################################################




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
                         # ... (tous les autres paramètres comme avant, y compris SCNR) ...
                         stacking_mode="kappa-sigma", kappa=2.5,
                         batch_size=10, correct_hot_pixels=True, hot_pixel_threshold=3.0,
                         neighborhood_size=5, bayer_pattern="GRBG", perform_cleanup=True,
                         use_weighting=False, weight_snr=True, weight_stars=True,
                         snr_exp=1.0, stars_exp=0.5, min_w=0.1,
                         use_drizzle=False, drizzle_scale=2.0, drizzle_wht_threshold=0.7,
                         drizzle_mode="Final", drizzle_kernel="square", drizzle_pixfrac=1.0,
                         apply_chroma_correction=True,
                         apply_final_scnr=False, final_scnr_target_channel='green',
                         final_scnr_amount=0.8, final_scnr_preserve_luminosity=True,
                         ### Arguments pour Paramètres Expert ###
                         bn_grid_size_str="16x16", bn_perc_low=5, bn_perc_high=30,
                         bn_std_factor=1.0, bn_min_gain=0.2, bn_max_gain=7.0,
                         cb_border_size=25, cb_blur_radius=8,
                         cb_min_b_factor=0.4, cb_max_b_factor=1.5,
                         final_edge_crop_percent=2.0, # En pourcentage
                         ### Arguments pour Paramètres Photutils BN ###
                         apply_photutils_bn=False,
                         photutils_bn_box_size=128,
                         photutils_bn_filter_size=5,
                         photutils_bn_sigma_clip=3.0,
                         photutils_bn_exclude_percentile=98.0,
                         ### FIN NOUVEAU ###
                         ### FIN expert ###
                         is_mosaic_run=False, api_key=None, mosaic_settings=None):
        """
        Démarre le thread de traitement principal avec la configuration spécifiée.
        MAJ: Intègre l'initialisation pour SUM/W memmap.
        """
        print("DEBUG (Backend start_processing SUM/W): Début tentative démarrage...")
        # ... (log des args reçus comme avant) ...
        print(f"   -> Args SCNR reçus: Apply={apply_final_scnr}, Target={final_scnr_target_channel}, Amount={final_scnr_amount}, PreserveLum={final_scnr_preserve_luminosity}")

        if self.processing_active:
            self.update_progress("⚠️ Tentative de démarrer un traitement déjà en cours.")
            return False

        # 1. Reset flag stop et définir dossier courant (comme avant)
        self.stop_processing = False
        self.current_folder = os.path.abspath(input_dir)
        
        # --- 2. Préparation de la Référence et Obtention de la SHAPE ---
        #    (On a besoin de la shape AVANT d'appeler initialize pour les memmaps)
        print("DEBUG (Backend start_processing SUM/W): Étape 2 - Préparation référence & shape...")
        reference_image_data_for_shape = None 
        reference_header_for_shape = None 
        ref_shape_hwc = None                  # Shape finale (H,W,C) pour memmap

        try:
            # Construire la liste des dossiers potentiels où trouver une image pour la shape
            potential_folders_for_shape = []
            if self.current_folder and os.path.isdir(self.current_folder): # Dossier principal d'input
                potential_folders_for_shape.append(self.current_folder)
            
            if initial_additional_folders: # Dossiers additionnels passés au démarrage
                for add_f in initial_additional_folders:
                    abs_add_f = os.path.abspath(add_f)
                    if abs_add_f and os.path.isdir(abs_add_f) and abs_add_f not in potential_folders_for_shape:
                        potential_folders_for_shape.append(abs_add_f)

            if not potential_folders_for_shape:
                raise RuntimeError("Aucun dossier d'entrée (principal ou additionnel) valide fourni pour trouver une image de référence.")

            current_folder_to_scan_for_shape = None
            files_in_folder_for_shape = []

            for folder_path in potential_folders_for_shape:
                print(f"DEBUG QM [start_processing SUM/W]: Scan pour FITS dans: {folder_path}")
                temp_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.fit', '.fits'))])
                if temp_files:
                    files_in_folder_for_shape = temp_files
                    current_folder_to_scan_for_shape = folder_path
                    print(f"DEBUG QM [start_processing SUM/W]: Fichiers FITS trouvés dans '{os.path.basename(current_folder_to_scan_for_shape)}' pour déterminer la shape.")
                    break # On a trouvé un dossier avec des fichiers

            if not current_folder_to_scan_for_shape or not files_in_folder_for_shape:
                # Si reference_path_ui est fourni ET valide, on pourrait essayer de l'utiliser directement pour la shape
                # mais _get_reference_image a besoin d'un dossier et d'une liste de fichiers pour son mode auto fallback.
                # Donc, si aucun fichier n'est trouvé dans les dossiers, on lève une erreur, même si reference_path_ui existe.
                # L'utilisateur doit fournir au moins un dossier avec des images.
                raise RuntimeError("Aucun fichier FITS trouvé dans les dossiers d'entrée spécifiés pour déterminer la shape de référence.")

            # Configurer l'aligneur (celui de l'instance, pas un temporaire)
            # car il sera utilisé ensuite dans _worker pour l'alignement réel.
            # Les paramètres sont passés à start_processing.
            self.aligner.correct_hot_pixels = correct_hot_pixels
            self.aligner.hot_pixel_threshold = hot_pixel_threshold
            self.aligner.neighborhood_size = neighborhood_size
            self.aligner.bayer_pattern = bayer_pattern
            self.aligner.reference_image_path = reference_path_ui or None # Valeur de l'UI

            print(f"DEBUG QM [start_processing SUM/W]: Appel _get_reference_image sur dossier '{os.path.basename(current_folder_to_scan_for_shape)}' (Nombre de fichiers: {len(files_in_folder_for_shape)}). Réf UI: '{self.aligner.reference_image_path}'")
            
            reference_image_data_for_shape, reference_header_for_shape = self.aligner._get_reference_image(
                current_folder_to_scan_for_shape, 
                files_in_folder_for_shape
            )

            if reference_image_data_for_shape is None or reference_header_for_shape is None:
                # _get_reference_image devrait déjà logguer la raison de l'échec
                raise RuntimeError("Échec obtention image/header référence pour déterminer la shape (via _get_reference_image).")

           
            # --- Obtenir la Shape ---
            # S'assurer qu'elle est bien (H, W, 3) même si l'image de référence est N&B
            ref_shape_initial = reference_image_data_for_shape.shape
            if len(ref_shape_initial) == 2: # Si N&B
                ref_shape_hwc = (ref_shape_initial[0], ref_shape_initial[1], 3)
                print(f"DEBUG (Backend start_processing SUM/W): Réf N&B, shape pour memmap sera {ref_shape_hwc}")
            elif len(ref_shape_initial) == 3 and ref_shape_initial[2] == 3: # Si déjà couleur
                ref_shape_hwc = ref_shape_initial
                print(f"DEBUG (Backend start_processing SUM/W): Réf couleur, shape pour memmap est {ref_shape_hwc}")
            else:
                raise RuntimeError(f"Shape de l'image référence non supportée: {ref_shape_initial}")

            # Stocker le header de référence pour plus tard (WCS, métadonnées)
            self.reference_header_for_wcs = reference_header_for_shape.copy()

            # --- Nettoyer la mémoire de l'image de référence temporaire ---
            del reference_image_data_for_shape, reference_header_for_shape
            gc.collect()
            print("DEBUG (Backend start_processing SUM/W): Shape obtenue et mémoire libérée.")

        except Exception as e_ref_shape:
            self.update_progress(f"❌ Erreur préparation référence/shape: {e_ref_shape}")
            print(f"ERREUR QM [start_processing SUM/W]: Échec préparation référence/shape : {e_ref_shape}")
            traceback.print_exc(limit=2)
            # self.processing_active = False # Pas besoin ici, car initialize n'a pas été appelé
            return False # Empêche de continuer si la shape n'est pas trouvée
        # --- Fin Préparation Référence et Shape ---

        # --- 3. Appel à initialize AVEC la shape ---
        print(f"DEBUG (Backend start_processing SUM/W): Appel à self.initialize() avec shape={ref_shape_hwc}...")
        # L'appel à initialize() réinitialise les compteurs, flags, et crée les memmaps
        if not self.initialize(output_dir, ref_shape_hwc):
            self.processing_active = False
            print("ERREUR (Backend start_processing SUM/W): Échec de self.initialize() pour SUM/W.")
            # initialize a déjà loggé l'erreur spécifique
            return False
        print("DEBUG (Backend start_processing SUM/W): self.initialize() terminé avec succès.")

        # --- 4. Définir les paramètres spécifiques à CETTE session *APRES* initialize ---
        #    (Cette partie est identique à la version précédente, assigne les arguments aux attributs self.)
        print("DEBUG (Backend start_processing SUM/W): Configuration des paramètres de session...")
        self.is_mosaic_run = is_mosaic_run
        self.drizzle_active_session = use_drizzle or self.is_mosaic_run
        self.api_key = api_key
        self.apply_chroma_correction = apply_chroma_correction
        self.correct_hot_pixels = correct_hot_pixels
        self.hot_pixel_threshold = hot_pixel_threshold
        self.neighborhood_size = neighborhood_size
        self.bayer_pattern = bayer_pattern
        self.perform_cleanup = perform_cleanup
        self.stacking_mode = stacking_mode
        self.kappa = float(kappa)
        #self.use_quality_weighting = False  #Forcé à False pour le test SUM/W initial décommenter si besoin pour tests 
        if use_weighting: print("INFO: Pondération qualité demandée mais DÉSACTIVÉE pour le test SUM/W.")
        # self.weight_by_snr = weight_snr # Pas besoin si use_quality_weighting = False
        # self.weight_by_stars = weight_stars # idem
        # self.snr_exponent = snr_exp # idem
        # self.stars_exponent = stars_exp # idem
        # self.min_weight = max(0.01, min(1.0, min_w)) # idem
        self.apply_final_scnr = apply_final_scnr
        self.final_scnr_target_channel = final_scnr_target_channel
        self.final_scnr_amount = final_scnr_amount
        self.final_scnr_preserve_luminosity = final_scnr_preserve_luminosity
        
        if self.drizzle_active_session:
            if self.is_mosaic_run:
                current_mosaic_settings = mosaic_settings if isinstance(mosaic_settings, dict) else {}
                self.drizzle_kernel = current_mosaic_settings.get('kernel', drizzle_kernel)
                self.drizzle_pixfrac = current_mosaic_settings.get('pixfrac', drizzle_pixfrac)
                try: self.drizzle_pixfrac = float(np.clip(float(self.drizzle_pixfrac), 0.01, 1.0))
                except (ValueError, TypeError): self.drizzle_pixfrac = 1.0
            else:
                 self.drizzle_kernel = drizzle_kernel
                 self.drizzle_pixfrac = drizzle_pixfrac
            self.drizzle_mode = drizzle_mode if drizzle_mode in ["Final", "Incremental"] else "Final"
            self.drizzle_scale = float(drizzle_scale)
            self.drizzle_wht_threshold = max(0.01, min(1.0, float(drizzle_wht_threshold)))
            print(f"   -> Params Drizzle Actifs -> Mode: {self.drizzle_mode}, Scale: {self.drizzle_scale:.1f}, WHT: {self.drizzle_wht_threshold:.2f}, Kernel: {self.drizzle_kernel}, Pixfrac: {self.drizzle_pixfrac:.2f}")
        else:
             print("DEBUG (Backend start_processing SUM/W): Session Drizzle non active.")

        ### Stockage des paramètres Expert dans self ###
        print("DEBUG (Backend start_processing SUM/W): Stockage des paramètres Expert...")
        # BN
        self.bn_grid_size_str = bn_grid_size_str
        self.bn_perc_low = bn_perc_low
        self.bn_perc_high = bn_perc_high
        self.bn_std_factor = bn_std_factor
        self.bn_min_gain = bn_min_gain
        self.bn_max_gain = bn_max_gain
        # CB
        self.cb_border_size = cb_border_size
        self.cb_blur_radius = cb_blur_radius
        self.cb_min_b_factor = cb_min_b_factor # Nom pour correspondre à SettingsManager
        self.cb_max_b_factor = cb_max_b_factor # Nom pour correspondre à SettingsManager
        # Rognage (convertir % en décimal pour usage interne)

        self.final_edge_crop_percent_decimal = float(final_edge_crop_percent) / 100.0
        ### Stockage des paramètres Photutils BN dans self ###
        print("DEBUG (Backend start_processing SUM/W): Stockage des paramètres Photutils BN...")
        self.apply_photutils_bn = apply_photutils_bn
        print(f"DEBUG (Backend start_processing SUM/W): self.apply_photutils_bn MIS À = {self.apply_photutils_bn}")
        self.photutils_bn_box_size = photutils_bn_box_size
        self.photutils_bn_filter_size = photutils_bn_filter_size
        self.photutils_bn_sigma_clip = photutils_bn_sigma_clip
        self.photutils_bn_exclude_percentile = photutils_bn_exclude_percentile
        print(f"   -> self.apply_photutils_bn = {self.apply_photutils_bn}")
        ### FIN Stockage des paramètres Photutils BN###
        print(f"   -> BN Grid='{self.bn_grid_size_str}', CB Border={self.cb_border_size}, CropDecimal={self.final_edge_crop_percent_decimal:.3f}")
        ### FIN ###

        # --- 5. Logs et Vérification Batch Size ---
        #    (Identique à la version précédente)
        if self.is_mosaic_run: self.update_progress("🖼️ Mode Mosaïque ACTIVÉ pour cette session.")
        elif self.drizzle_active_session: self.update_progress(f"💧 Mode Drizzle (Simple Champ) Activé ({self.drizzle_mode})...")
        else: self.update_progress("⚙️ Mode Stack Classique (SUM/W) Activé...")

        requested_batch_size = batch_size
        if requested_batch_size <= 0:
             # ... (estimation auto) ...
             self.update_progress("🧠 Estimation taille lot auto (reçu <= 0)...", None)
             sample_img_path = None
             if input_dir and os.path.isdir(input_dir): fits_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.fit', '.fits'))]; sample_img_path = os.path.join(input_dir, fits_files[0]) if fits_files else None
             try: estimated_size = estimate_batch_size(sample_image_path=sample_img_path); self.batch_size = estimated_size; self.update_progress(f"✅ Taille lot auto estimée: {estimated_size}", None)
             except Exception as est_err: self.update_progress(f"⚠️ Erreur estimation taille lot: {est_err}. Utilisation défaut (10).", None); self.batch_size = 10
        else: self.batch_size = requested_batch_size
        if self.batch_size < 3: self.update_progress(f"⚠️ Taille de lot ({self.batch_size}) trop petite, ajustée à 3.", None); self.batch_size = 3
        self.update_progress(f"ⓘ Taille de lot effective pour le traitement : {self.batch_size}")
        if self.apply_final_scnr: self.update_progress(f"🎨 SCNR Final (Vert, {self.final_scnr_amount*100:.0f}%) sera appliqué.")
        # Pas de log pour pondération qualité car désactivée pour ce test


        # --- 6. Gérer dossiers initiaux ---
        #    (Identique à la version précédente)
        initial_folders_to_add_count = 0
        with self.folders_lock:
            self.additional_folders = []
            if initial_additional_folders:
                for folder in initial_additional_folders:
                    abs_folder = os.path.abspath(folder)
                    if os.path.isdir(abs_folder) and abs_folder not in self.additional_folders:
                        self.additional_folders.append(abs_folder); initial_folders_to_add_count += 1
        if initial_folders_to_add_count > 0: self.update_progress(f"ⓘ {initial_folders_to_add_count} dossier(s) pré-ajouté(s) en attente."); self.update_progress(f"folder_count_update:{len(self.additional_folders)}")

        # --- 7. Ajouter fichiers initiaux ---
        #    (Identique à la version précédente)
        initial_files_added = self._add_files_to_queue(self.current_folder)
        if initial_files_added > 0: self._recalculate_total_batches(); self.update_progress(f"📋 {initial_files_added} fichiers initiaux ajoutés. Total lots estimé: {self.total_batches_estimated if self.total_batches_estimated > 0 else '?'}")
        elif not self.additional_folders: self.update_progress("⚠️ Aucun fichier initial trouvé ou dossier supplémentaire en attente.")
        
        # --- 8. Configurer référence pour l'aligneur ---
        #    (Identique à la version précédente)
        #    Note: self.aligner.reference_image_path est utilisé par _get_reference_image SI fourni.
        #    Mais l'image de référence pour l'alignement dans la boucle _worker est passée explicitement.
        self.aligner.reference_image_path = reference_path_ui or None

        # --- 9. Démarrer worker ---
        #    (Identique à la version précédente)
        print("DEBUG (Backend start_processing SUM/W): Démarrage du thread worker...")
        self.processing_thread = threading.Thread(target=self._worker, name="StackerWorker"); self.processing_thread.daemon = True
        self.processing_thread.start(); self.processing_active = True
        self.update_progress("🚀 Thread de traitement démarré.")
        print("DEBUG (Backend start_processing SUM/W): Fin.")
        return True

# --- FIN de start_processing MODIFIÉ ---




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


################################################################################################################################################


    def stop(self):
        if not self.processing_active: return
        self.update_progress("⛔ Arrêt demandé..."); self.stop_processing = True; self.aligner.stop_processing = True


################################################################################################################################################


    def is_running(self):
        """Vérifie si le thread de traitement est actif et en cours d'exécution."""
        # Vérifier si l'attribut processing_active existe et est True
        is_processing_flag_active = getattr(self, 'processing_active', False)
        
        # Vérifier si l'attribut processing_thread existe
        thread_exists = hasattr(self, 'processing_thread')
        
        # Si les deux existent, vérifier si le thread est non None et vivant
        is_thread_alive_and_valid = False
        if thread_exists:
            thread_obj = getattr(self, 'processing_thread', None)
            if thread_obj is not None and thread_obj.is_alive():
                is_thread_alive_and_valid = True
        
        # print(f"DEBUG QM [is_running]: processing_active={is_processing_flag_active}, thread_exists={thread_exists}, thread_alive={is_thread_alive_and_valid}") # Debug
        return is_processing_flag_active and thread_exists and is_thread_alive_and_valid



######################################################################################################################################################

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

