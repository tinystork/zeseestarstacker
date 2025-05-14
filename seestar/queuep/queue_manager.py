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
from astropy.coordinates import SkyCoord, concatenate as skycoord_concatenate
from astropy import units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS, FITSFixedWarning
from ccdproc import CCDData, combine as ccdproc_combine
from ..enhancement.stack_enhancement import apply_edge_crop
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.spatial import ConvexHull
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


# --- Core/Internal Imports (Needed for __init__ or core logic) ---
try: from ..core.hot_pixels import detect_and_correct_hot_pixels
except ImportError as e: print(f"ERREUR QM: Échec import detect_and_correct_hot_pixels: {e}"); raise
try: from ..core.image_processing import (load_and_validate_fits, debayer_image, save_fits_image, save_preview_image)
except ImportError as e: print(f"ERREUR QM: Échec import image_processing: {e}"); raise
try: from ..core.utils import estimate_batch_size
except ImportError as e: print(f"ERREUR QM: Échec import utils: {e}"); raise
try: from ..enhancement.color_correction import ChromaticBalancer
except ImportError as e_cb: print(f"ERREUR QM: Échec import ChromaticBalancer: {e_cb}"); raise

# --- Imports INTERNES à déplacer en IMPORTS TARDIFS (si utilisés uniquement dans des méthodes spécifiques) ---
# Ces modules/fonctions sont gérés par des appels conditionnels ou try/except dans les méthodes où ils sont utilisés.
# from ..enhancement.drizzle_integration import _load_drizzle_temp_file, DrizzleProcessor, _create_wcs_from_header 
# from ..enhancement.astrometry_solver import solve_image_wcs 
# from ..enhancement.mosaic_processor import process_mosaic_from_aligned_files 
# from ..enhancement.stack_enhancement import StackEnhancer # Cette classe n'est pas utilisée ici

# --- Configuration des Avertissements ---
warnings.filterwarnings('ignore', category=FITSFixedWarning)
print("DEBUG QM: Configuration warnings OK.")
# --- FIN Imports ---


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
# --- IMPORT POUR L'ALIGNEUR LOCAL ---
try:
    from ..core import SeestarLocalAligner # Devrait être FastSeestarAligner aliasé
    _LOCAL_ALIGNER_AVAILABLE = True
    print("DEBUG QM: Import SeestarLocalAligner (local CV) OK.")
except ImportError:
    _LOCAL_ALIGNER_AVAILABLE = False
    SeestarLocalAligner = None # Définir pour que le code ne plante pas à l'instanciation
    print("WARN QM: SeestarLocalAligner (local CV) non importable. Alignement mosaïque local désactivé.")
# ---  ---



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

try:
    from ..enhancement.stack_enhancement import feather_by_weight_map # NOUVEL IMPORT
    _FEATHERING_AVAILABLE = True
    print("DEBUG QM: Import feather_by_weight_map depuis stack_enhancement OK.")
except ImportError as e_feather:
    _FEATHERING_AVAILABLE = False
    print(f"ERREUR QM: Échec import feather_by_weight_map depuis stack_enhancement: {e_feather}")
    # Définir une fonction factice pour que le code ne plante pas si l'import échoue
    # lors des appels ultérieurs, bien qu'on vérifiera _FEATHERING_AVAILABLE.
    def feather_by_weight_map(img, wht, blur_px=256, eps=1e-6):
        print("ERREUR: Fonction feather_by_weight_map non disponible (échec import).")
        return img # Retourner l'image originale
try:
    from ..enhancement.stack_enhancement import apply_low_wht_mask # NOUVEL IMPORT
    _LOW_WHT_MASK_AVAILABLE = True
    print("DEBUG QM: Import apply_low_wht_mask depuis stack_enhancement OK.")
except ImportError as e_low_wht:
    _LOW_WHT_MASK_AVAILABLE = False
    print(f"ERREUR QM: Échec import apply_low_wht_mask: {e_low_wht}")
    def apply_low_wht_mask(img, wht, percentile=5, soften_px=128, progress_callback=None): # Factice
        if progress_callback: progress_callback("   [LowWHTMask] ERREUR: Fonction apply_low_wht_mask non disponible (échec import).", None)
        else: print("ERREUR: Fonction apply_low_wht_mask non disponible (échec import).")
        return img
# --- Optional Third-Party Imports (Post-processing related) ---
# Ces imports sont tentés globalement. Des flags indiquent leur disponibilité.
_PHOTOUTILS_BG_SUB_AVAILABLE = False
try:
    from ..core.background import subtract_background_2d
    _PHOTOUTILS_BG_SUB_AVAILABLE = True
    print("DEBUG QM: Import subtract_background_2d (Photutils) OK.")
except ImportError as e:
    subtract_background_2d = None # Fonction factice
    print(f"WARN QM: Échec import subtract_background_2d (Photutils): {e}")

_BN_AVAILABLE = False # Neutralisation de fond globale
try:
    from ..tools.stretch import neutralize_background_automatic
    _BN_AVAILABLE = True
    print("DEBUG QM: Import neutralize_background_automatic OK.")
except ImportError as e:
    neutralize_background_automatic = None # Fonction factice
    print(f"WARN QM: Échec import neutralize_background_automatic: {e}")

_SCNR_AVAILABLE = False # SCNR Final
try:
    from ..enhancement.color_correction import apply_scnr
    _SCNR_AVAILABLE = True
    print("DEBUG QM: Import apply_scnr OK.")
except ImportError as e:
    apply_scnr = None # Fonction factice
    print(f"WARN QM: Échec import apply_scnr: {e}")

_CROP_AVAILABLE = False # Rognage Final
try:
    from ..enhancement.stack_enhancement import apply_edge_crop
    _CROP_AVAILABLE = True
    print("DEBUG QM: Import apply_edge_crop OK.")
except ImportError as e:
    apply_edge_crop = None # Fonction factice
    print(f"WARN QM: Échec import apply_edge_crop: {e}")

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





# --- DANS LA CLASSE SeestarQueuedStacker DANS seestar/queuep/queue_manager.py ---

    def __init__(self):
        print("\n==== DÉBUT INITIALISATION SeestarQueuedStacker (AVEC LocalAligner) ====") # Modifié
        
        # --- 1. Attributs Critiques et Simples ---
        print("  -> Initialisation attributs simples et flags...")
        self.processing_active = False; self.stop_processing = False; self.processing_error = None
        self.is_mosaic_run = False; self.drizzle_active_session = False # Sera défini dans start_processing
        self.perform_cleanup = True; self.use_quality_weighting = True 
        self.correct_hot_pixels = True; self.apply_chroma_correction = True
        self.apply_final_scnr = False 
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
        
        ### Attributs pour SUM / W (Memmap) ###
        self.sum_memmap_path = None 
        self.wht_memmap_path = None 
        self.cumulative_sum_memmap = None  
        self.cumulative_wht_memmap = None  
        self.memmap_shape = None           
        self.memmap_dtype_sum = np.float32 
        self.memmap_dtype_wht = np.float32 
        print("  -> Attributs SUM/W (memmap) initialisés à None.")
        
        # Paramètres de pondération initialisés
        self.use_quality_weighting = False 
        self.weight_by_snr = True          
        self.weight_by_stars = True        
        self.snr_exponent = 1.0
        self.stars_exponent = 0.5
        self.min_weight = 0.01
        self.apply_feathering = False
        self.feather_blur_px = 256
        
        self.current_batch_data = [] 
        self.current_stack_header = None 
        self.images_in_cumulative_stack = 0 
        self.total_exposure_seconds = 0.0 
        self.intermediate_drizzle_batch_files = [] # Important de l'avoir ici

        # Processing Parameters (valeurs par défaut)
        self.stacking_mode = "kappa-sigma"; self.kappa = 2.5; self.batch_size = 10
        self.hot_pixel_threshold = 3.0; self.neighborhood_size = 5; self.bayer_pattern = "GRBG"
        self.drizzle_mode = "Final"; self.drizzle_scale = 2.0; self.drizzle_wht_threshold = 0.7
        self.drizzle_kernel = "square"; self.drizzle_pixfrac = 1.0 # Note: pixfrac pour Drizzle
        # snr_exponent, stars_exponent, min_weight sont déjà définis plus haut
        self.final_scnr_target_channel = 'green'; self.final_scnr_amount = 0.8; self.final_scnr_preserve_luminosity = True
        
        # Statistics
        self.files_in_queue = 0; self.processed_files_count = 0; self.aligned_files_count = 0
        self.stacked_batches_count = 0; self.total_batches_estimated = 0
        self.failed_align_count = 0; self.failed_stack_count = 0; self.skipped_files_count = 0
        self.photutils_bn_applied_in_session = False
        self.bn_globale_applied_in_session = False
        self.cb_applied_in_session = False
        self.feathering_applied_in_session = False # Ajouté pour la cohérence
        self.low_wht_mask_applied_in_session = False # Ajouté pour la cohérence
        self.scnr_applied_in_session = False
        self.crop_applied_in_session = False
        self.photutils_params_used_in_session = {}
        print("  -> Attributs simples et paramètres par défaut initialisés.")
        
        # --- Attributs pour l'aligneur local ---
        self.local_aligner_instance = None
        # Flag pour choisir entre Astrometry.net (si False) et FastAligner (si True et disponible) pour la mosaïque.
        # Peut être exposé à l'utilisateur plus tard.
        self.is_local_alignment_preferred_for_mosaic = True 
        print(f"  -> Mosaïque: Préférence pour alignement local: {self.is_local_alignment_preferred_for_mosaic}")
        # --- Fin attributs aligneur local ---

        # --- 2. Instanciations de Classes ---
        try:
            print("  -> Instanciation ChromaticBalancer...")
            self.chroma_balancer = ChromaticBalancer(border_size=50, blur_radius=15) # Utiliser des valeurs par défaut raisonnables
            print("     ✓ ChromaticBalancer OK.")
        except Exception as e_cb: 
            print(f"  -> ERREUR ChromaticBalancer: {e_cb}")
            self.chroma_balancer = None
            # raise # Optionnel: relancer si critique, ou juste continuer sans
            
        try:
            print("  -> Instanciation SeestarAligner (pour alignement général astroalign)...")
            self.aligner = SeestarAligner() # Pour l'alignement individuel des images sur la référence principale
            print("     ✓ SeestarAligner (astroalign) OK.")
        except Exception as e_align: 
            print(f"  -> ERREUR SeestarAligner (astroalign): {e_align}")
            self.aligner = None
            raise # L'aligneur principal est critique
        
        # --- Instanciation du SeestarLocalAligner (FastSeestarAligner) ---
        # _LOCAL_ALIGNER_AVAILABLE est une variable globale définie en haut du module queue_manager.py
        # lors de la tentative d'import de SeestarLocalAligner.
        # SeestarLocalAligner (l'alias de la classe) est aussi défini globalement.
        if _LOCAL_ALIGNER_AVAILABLE and SeestarLocalAligner is not None:
            try:
                print("  -> Instanciation SeestarLocalAligner (pour mosaïque locale si préférée)...")
                # Mettre debug=True pour avoir les logs détaillés du FastAligner lui-même
                self.local_aligner_instance = SeestarLocalAligner(debug=True) 
                print("     ✓ SeestarLocalAligner instancié.")
                # Passer le progress_callback si on veut que FastAligner logue via le GUI
                # if self.progress_callback and hasattr(self.local_aligner_instance, 'set_progress_callback'):
                #     self.local_aligner_instance.set_progress_callback(self.progress_callback)
            except Exception as e_local_align_inst:
                print(f"  -> ERREUR lors de l'instanciation de SeestarLocalAligner: {e_local_align_inst}")
                traceback.print_exc(limit=1)
                self.local_aligner_instance = None
                # On ne modifie PAS _LOCAL_ALIGNER_AVAILABLE ici. Son statut est fixé à l'import.
                # L'échec de l'instanciation sera géré par la vérification de `self.local_aligner_instance is not None`.
                print("     WARN QM: Instanciation de SeestarLocalAligner a échoué. Il ne sera pas utilisable.")
        else:
            print("  -> SeestarLocalAligner n'est pas disponible (import échoué ou classe non définie), instanciation ignorée.")
            self.local_aligner_instance = None # S'assurer qu'il est None
        # --- FIN Instanciation ---

        print("==== FIN INITIALISATION SeestarQueuedStacker (AVEC LocalAligner) ====\n")






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
    


########################################################################################################################################################





# --- DANS LA CLASSE SeestarQueuedStacker DANS seestar/queuep/queue_manager.py ---

    def _worker(self):
        """
        Thread principal pour le traitement des images.
        MODIFIÉ (V5.1 - Debug PlateSolve & Fallback): Ajout de logs détaillés pour le plate-solving
                                                     et implémentation d'un fallback WCS.
        """
        print("\n" + "=" * 10 + " DEBUG QM [_worker V5.1 - Debug PlateSolve]: Initialisation du worker " + "=" * 10)

        # --- Initialisation des variables ---
        self.processing_active = True; self.processing_error = None; start_time_session = time.monotonic()
        reference_image_data_for_global_alignment = None; reference_header_for_global_alignment = None
        mosaic_ref_panel_image_data = None; mosaic_ref_panel_header = None; mosaic_ref_panel_wcs_absolute = None
        current_batch_items_with_masks_for_stack_batch = []
        self.intermediate_drizzle_batch_files = [] 
        all_aligned_files_with_info_for_mosaic = [] 
        
        use_local_aligner_for_this_mosaic_run = (
            self.is_mosaic_run and 
            self.is_local_alignment_preferred_for_mosaic and 
            _LOCAL_ALIGNER_AVAILABLE and 
            self.local_aligner_instance is not None
        )
        print(f"DEBUG QM [_worker V5.1]: Mode -> is_mosaic_run={self.is_mosaic_run} "
              f"(Utilisation Aligneur Local pour Mosaïque: {use_local_aligner_for_this_mosaic_run}), "
              f"drizzle_active_session={self.drizzle_active_session}, drizzle_mode='{self.drizzle_mode}'")
        if use_local_aligner_for_this_mosaic_run:
            print(f"  -> Détails Aligneur Local: Préféré={self.is_local_alignment_preferred_for_mosaic}, "
                  f"Module Dispo={_LOCAL_ALIGNER_AVAILABLE}, Instance OK={self.local_aligner_instance is not None}")

        try:
            self.update_progress("⭐ Préparation de l'image de référence principale et/ou du premier panneau mosaïque...")
            if not self.current_folder or not os.path.isdir(self.current_folder): 
                raise RuntimeError(f"Dossier d'entrée initial invalide : {self.current_folder}")
            
            initial_files_in_first_folder = sorted([f for f in os.listdir(self.current_folder) if f.lower().endswith((".fit", ".fits"))])
            if not initial_files_in_first_folder and not self.additional_folders: 
                raise RuntimeError("Aucun fichier FITS initial trouvé pour référence principale ou premier panneau.")

            self.aligner.correct_hot_pixels = self.correct_hot_pixels
            self.aligner.hot_pixel_threshold = self.hot_pixel_threshold
            self.aligner.neighborhood_size = self.neighborhood_size
            self.aligner.bayer_pattern = self.bayer_pattern
            
            print(f"DEBUG QM [_worker V5.1]: Appel _get_reference_image (astroalign) pour référence alignement général...")
            reference_image_data_for_global_alignment, reference_header_for_global_alignment = self.aligner._get_reference_image(
                self.current_folder, initial_files_in_first_folder
            )
            if reference_image_data_for_global_alignment is None or reference_header_for_global_alignment is None:
                raise RuntimeError("Échec obtention image/header de référence pour alignement général (astroalign).")
            
            self.reference_header_for_wcs = reference_header_for_global_alignment.copy()
            # Remplacer _SOURCE_PATH par _REFSRCFN pour éviter erreur longueur FITS, si besoin
            if reference_header_for_global_alignment.get('_SOURCE_PATH'):
                 # self.reference_header_for_wcs['_SOURCE_PATH'] = (reference_header_for_global_alignment.get('_SOURCE_PATH'), "Source file for global reference")
                 self.reference_header_for_wcs['_REFSRCFN'] = (os.path.basename(str(reference_header_for_global_alignment.get('_SOURCE_PATH',''))), "Base name of global ref source")


            self.aligner._save_reference_image(reference_image_data_for_global_alignment, reference_header_for_global_alignment, self.output_folder)
            print("DEBUG QM [_worker V5.1]: Image de référence pour alignement général (astroalign) prête et sauvegardée.")

            if use_local_aligner_for_this_mosaic_run:
                self.update_progress("⭐ Mosaïque Locale: Traitement du panneau de référence (ancrage)...")
                mosaic_ref_panel_image_data = reference_image_data_for_global_alignment 
                mosaic_ref_panel_header = reference_header_for_global_alignment.copy()
                
                if reference_header_for_global_alignment.get('_SOURCE_PATH'):
                    # mosaic_ref_panel_header['_PANEL_REF_SRC'] = (reference_header_for_global_alignment.get('_SOURCE_PATH'), "Source file of this mosaic reference panel")
                    mosaic_ref_panel_header['_PANREF_FN'] = (os.path.basename(str(reference_header_for_global_alignment.get('_SOURCE_PATH',''))), "Base name of this mosaic ref panel source")


                self.update_progress("   -> Mosaïque Locale: Résolution astrométrique du panneau de référence...")
                
                # --- Section d'import et d'appel de solve_image_wcs avec logs et fallback ---
                solve_image_wcs_func = None 
                try: 
                    from ..enhancement.astrometry_solver import solve_image_wcs as solve_image_wcs_imported
                    solve_image_wcs_func = solve_image_wcs_imported
                    print(f"DEBUG QM [_worker V5.1]: Import de 'solve_image_wcs' depuis astrometry_solver REUSSI. Type: {type(solve_image_wcs_func)}")
                except ImportError as e_imp_solve: 
                    print(f"DEBUG QM [_worker V5.1]: Import de 'solve_image_wcs' depuis astrometry_solver ECHOUE: {e_imp_solve}")
                except Exception as e_other_imp_solve:
                    print(f"DEBUG QM [_worker V5.1]: ERREUR INATTENDUE import solve_image_wcs: {e_other_imp_solve}")
                    traceback.print_exc(limit=1)
                
                print(f"DEBUG QM [_worker V5.1]: Valeur de solve_image_wcs_func AVANT appel: {solve_image_wcs_func}")
                print(f"DEBUG QM [_worker V5.1]: Valeurs pour Astrometry -> API Key Présente: {'Oui' if self.api_key and len(self.api_key)>10 else 'Non/Courte'} (Type: {type(self.api_key)})")
                print(f"DEBUG QM [_worker V5.1]: Valeurs pour Astrometry -> Ref Scale (arcsec/pix): {self.reference_pixel_scale_arcsec} (Type: {type(self.reference_pixel_scale_arcsec)})")

                mosaic_ref_panel_wcs_absolute = None 
                if solve_image_wcs_func: 
                    print(f"DEBUG QM [_worker V5.1]: APPEL de solve_image_wcs_func (Astrometry.net)...")
                    mosaic_ref_panel_wcs_absolute = solve_image_wcs_func(
                        mosaic_ref_panel_image_data, 
                        mosaic_ref_panel_header, 
                        self.api_key,
                        scale_est_arcsec_per_pix=self.reference_pixel_scale_arcsec,
                        progress_callback=self.update_progress 
                    )
                    print(f"DEBUG QM [_worker V5.1]: RETOUR de solve_image_wcs_func. WCS obtenu: {'Oui' if mosaic_ref_panel_wcs_absolute else 'Non'}")
                else:
                     print(f"DEBUG QM [_worker V5.1]: solve_image_wcs_func est None, Astrometry.net ne sera pas appelé.")
                
                if mosaic_ref_panel_wcs_absolute is None:
                    self.update_progress("   ⚠️ Échec plate-solving Astrometry.net pour panneau de référence. Tentative de WCS approximatif...", None)
                    print("WARN QM [_worker V5.1]: Plate-solve Astrometry.net a échoué ou n'a pas été tenté. Tentative de fallback avec WCS approximatif.")
                    
                    _create_wcs_from_header_func = None
                    try:
                        from ..enhancement.drizzle_integration import _create_wcs_from_header as _cwfh_func
                        _create_wcs_from_header_func = _cwfh_func
                        print(f"DEBUG QM [_worker V5.1]: Import de _create_wcs_from_header pour fallback REUSSI.")
                    except ImportError:
                        print(f"ERREUR QM [_worker V5.1]: Import de _create_wcs_from_header pour fallback ECHOUE.")
                    
                    if _create_wcs_from_header_func:
                        try:
                            mosaic_ref_panel_wcs_absolute = _create_wcs_from_header_func(mosaic_ref_panel_header)
                            if mosaic_ref_panel_wcs_absolute and mosaic_ref_panel_wcs_absolute.is_celestial:
                                self.update_progress("   -> WCS approximatif créé avec succès à partir du header du panneau de référence.", None)
                                print("INFO QM [_worker V5.1]: Fallback WCS approximatif créé.")
                                nx_fb = mosaic_ref_panel_header.get('NAXIS1')
                                ny_fb = mosaic_ref_panel_header.get('NAXIS2')
                                if nx_fb and ny_fb:
                                    try:
                                        mosaic_ref_panel_wcs_absolute.pixel_shape = (int(nx_fb), int(ny_fb))
                                        print(f"DEBUG QM: pixel_shape ({nx_fb},{ny_fb}) attaché au WCS de fallback.")
                                    except ValueError:
                                        print(f"WARN QM: NAXIS1/2 non entiers pour WCS fallback ('{nx_fb}','{ny_fb}').")
                                elif hasattr(mosaic_ref_panel_image_data, 'shape'):
                                    mosaic_ref_panel_wcs_absolute.pixel_shape = (mosaic_ref_panel_image_data.shape[1], mosaic_ref_panel_image_data.shape[0])
                                    print(f"DEBUG QM: pixel_shape depuis image_data ({mosaic_ref_panel_wcs_absolute.pixel_shape}) attaché au WCS de fallback.")
                                else:
                                     print(f"WARN QM: Impossible d'attacher pixel_shape au WCS de fallback (NAXIS ou shape image manquants).")
                            else:
                                self.update_progress("   -> ERREUR: Échec création WCS approximatif pour panneau de référence (non céleste ou None).", None)
                                print("ERREUR QM [_worker V5.1]: Échec création WCS de fallback (non céleste ou None).")
                                mosaic_ref_panel_wcs_absolute = None 
                        except Exception as e_fallback:
                            self.update_progress(f"   -> ERREUR lors de la tentative de création WCS de fallback: {e_fallback}", None)
                            print(f"ERREUR QM [_worker V5.1]: Exception pendant fallback WCS: {e_fallback}")
                            traceback.print_exc(limit=1)
                            mosaic_ref_panel_wcs_absolute = None
                    else: 
                        self.update_progress("   -> ERREUR CRITIQUE: Fonction _create_wcs_from_header non disponible pour fallback.", None)
                        mosaic_ref_panel_wcs_absolute = None
                
                if mosaic_ref_panel_wcs_absolute is None:
                    raise RuntimeError("Mosaïque Locale: Échec plate-solving (Astrometry ET fallback) du panneau de référence. Impossible de continuer.")
                
                print(f"DEBUG QM [_worker V5.1]: Panneau de référence traité (Astrometry ou Fallback). WCS Absolu prêt.")
                self.reference_wcs_object = mosaic_ref_panel_wcs_absolute 
                
                mat_identite = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
                valid_mask_ref_panel = np.ones(mosaic_ref_panel_image_data.shape[:2], dtype=bool) 
                
                all_aligned_files_with_info_for_mosaic.append(
                    (mosaic_ref_panel_image_data.copy(),      
                     mosaic_ref_panel_header.copy(),          
                     self.reference_wcs_object,               
                     mat_identite,                            
                     valid_mask_ref_panel)                    
                )
                self.aligned_files_count += 1 
                print(f"DEBUG QM [_worker V5.1]: Mosaïque Locale: Panneau de référence ajouté à la liste (total: {self.aligned_files_count}).")
            
            elif self.drizzle_active_session or (self.is_mosaic_run and not use_local_aligner_for_this_mosaic_run):
                # ... (logique inchangée pour Drizzle standard / Mosaïque Astrometry) ...
                self.update_progress("   -> Résolution astrométrique de la référence principale (pour Drizzle standard / Mosaïque Astrometry)...")
                solve_image_wcs_func = None
                try: 
                    from ..enhancement.astrometry_solver import solve_image_wcs as solve_image_wcs_imported
                    solve_image_wcs_func = solve_image_wcs_imported
                except ImportError: pass # Géré par la condition suivante

                if solve_image_wcs_func:
                    self.reference_wcs_object = solve_image_wcs_func(
                        reference_image_data_for_global_alignment, 
                        self.reference_header_for_wcs, 
                        self.api_key, 
                        scale_est_arcsec_per_pix=self.reference_pixel_scale_arcsec, 
                        progress_callback=self.update_progress
                    )
                else: 
                    self.update_progress("   -> ERREUR CRITIQUE: solve_image_wcs non disponible."); self.reference_wcs_object = None
                
                if self.reference_wcs_object is None:
                    raise RuntimeError("Échec plate-solving de la référence principale (requis pour Drizzle/Mosaïque Astrometry).")
                print(f"DEBUG QM [_worker V5.1]: WCS de référence principale (Astrometry) obtenu pour Drizzle/Mosaïque Astrometry.")


            self.update_progress("⭐ Référence(s) prête(s).", 5)
            self._recalculate_total_batches() 
            self.update_progress(f"▶️ Démarrage boucle de traitement (En file: {self.files_in_queue} | Lots Estimés: {self.total_batches_estimated if self.total_batches_estimated > 0 else '?'})")

            path_of_processed_ref_panel = None
            if use_local_aligner_for_this_mosaic_run and mosaic_ref_panel_header and '_PANREF_FN' in mosaic_ref_panel_header: # Utiliser la clé courte
                # On doit reconstruire le chemin complet si on ne stocke que le basename
                # Pour l'instant, on suppose que le premier dossier (self.current_folder) contient le panneau réf.
                # C'est une approximation, mais devrait fonctionner si le panneau réf est bien du premier dossier.
                # Une solution plus robuste serait de stocker le chemin complet dans une variable au lieu du header.
                # Pour ce debug, on va essayer de matcher juste le basename.
                path_of_processed_ref_panel = mosaic_ref_panel_header['_PANREF_FN'] # Juste le nom de base
                print(f"DEBUG QM [_worker V5.1]: Nom de base du panneau de référence traité (à skipper): {path_of_processed_ref_panel}")


            while not self.stop_processing:
                file_path = None 
                aligned_data_item = None; header_item = None; quality_scores_item = None
                wcs_object_indiv_item = None; valid_pixel_mask_item = None
                
                try:
                    file_path = self.queue.get(timeout=1.0) 
                    file_name_for_log = os.path.basename(file_path)
                    
                    if use_local_aligner_for_this_mosaic_run:
                        # print(f"DEBUG QM [_worker V5.1]: Traitement Mosaïque Locale pour fichier: {file_name_for_log}")
                        
                        if path_of_processed_ref_panel and file_name_for_log == path_of_processed_ref_panel:
                            print(f"DEBUG QM [_worker V5.1]: Fichier {file_name_for_log} EST le panneau de référence (déjà ajouté et WCS résolu). Consommé de la queue.")
                            self.processed_files_count += 1 
                            self.queue.task_done()
                            path_of_processed_ref_panel = None 
                            continue 

                        self.update_progress(f"   -> Mosaïque Locale: Alignement local de '{file_name_for_log}' sur panneau de référence...")
                        
                        current_panel_data_loaded = load_and_validate_fits(file_path)
                        if current_panel_data_loaded is None: 
                            raise ValueError(f"Échec chargement FITS {file_name_for_log} pour alignement local.")
                        current_panel_header = fits.getheader(file_path)
                        # current_panel_header['_SOURCE_PATH'] = file_path # Commenté pour éviter erreur longueur FITS
                        current_panel_header['_SRCFILE'] = (os.path.basename(file_path), "Original source file name")


                        current_panel_data_processed = current_panel_data_loaded.astype(np.float32)
                        if current_panel_data_processed.ndim == 2:
                            bayer_pat = current_panel_header.get('BAYERPAT', self.bayer_pattern)
                            pattern_upper = bayer_pat.upper() if isinstance(bayer_pat, str) else self.bayer_pattern.upper()
                            if pattern_upper in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                                try: 
                                    current_panel_data_processed = debayer_image(current_panel_data_processed, pattern_upper)
                                except Exception as e_deb: 
                                    print(f"WARN QM: Debayering échoué pour panneau local {file_name_for_log}: {e_deb}. Utilisation N&B.")
                        if self.correct_hot_pixels:
                            try: 
                                current_panel_data_processed = detect_and_correct_hot_pixels(current_panel_data_processed, self.hot_pixel_threshold, self.neighborhood_size)
                            except Exception as e_hp: 
                                print(f"WARN QM: Correction HP échouée pour panneau local {file_name_for_log}: {e_hp}")
                        
                        if hasattr(self.local_aligner_instance, 'set_progress_callback') and callable(self.progress_callback) :
                             self.local_aligner_instance.set_progress_callback(self.update_progress) # S'assurer de passer le bon callback
                        
                        _aligned_img_temp, M_transform, align_success = self.local_aligner_instance._align_image(
                            current_panel_data_processed,  
                            mosaic_ref_panel_image_data,   
                            file_name_for_log
                        )
                        
                        self.processed_files_count += 1 
                        if align_success and M_transform is not None:
                            self.aligned_files_count += 1
                            valid_mask_this_panel = np.ones(current_panel_data_processed.shape[:2], dtype=bool)
                            
                            all_aligned_files_with_info_for_mosaic.append(
                                (current_panel_data_processed.copy(), 
                                 current_panel_header.copy(),         
                                 self.reference_wcs_object,          
                                 M_transform.copy(),                  
                                 valid_mask_this_panel)               
                            )
                            # print(f"DEBUG QM [_worker V5.1]: Mosaïque Locale: Panneau '{file_name_for_log}' aligné. " # Trop verbeux
                            #       f"Total alignés: {self.aligned_files_count}. Liste mosaïque: {len(all_aligned_files_with_info_for_mosaic)}")
                        else:
                            self.update_progress(f"   -> Mosaïque Locale: Échec alignement local pour {file_name_for_log}. Ignoré.")
                            self.failed_align_count +=1
                        
                        del current_panel_data_loaded, current_panel_header, current_panel_data_processed, _aligned_img_temp, M_transform
                    
                    else: 
                        # print(f"DEBUG QM [_worker V5.1]: Traitement NON-Mosaïque Locale pour '{file_name_for_log}'...") # Trop verbeux
                        aligned_data_item, header_item, quality_scores_item, wcs_object_indiv_item, valid_pixel_mask_item = (
                            self._process_file(file_path, reference_image_data_for_global_alignment)
                        )
                        self.processed_files_count += 1
                        if aligned_data_item is not None and valid_pixel_mask_item is not None:
                            self.aligned_files_count += 1
                            current_item_tuple = (aligned_data_item, header_item, quality_scores_item, wcs_object_indiv_item, valid_pixel_mask_item)
                            
                            if self.is_mosaic_run: 
                                all_aligned_files_with_info_for_mosaic.append(current_item_tuple)
                                # print(f"DEBUG QM [_worker V5.1]: Item {self.aligned_files_count} ajouté pour MOSAÏQUE (Astrometry). Liste: {len(all_aligned_files_with_info_for_mosaic)}")
                            
                            else:
                                current_batch_items_with_masks_for_stack_batch.append(current_item_tuple)
                                # print(f"DEBUG QM [_worker V5.1]: Item {self.aligned_files_count} ajouté au lot source (taille: {len(current_batch_items_with_masks_for_stack_batch)}).")
                                
                                if len(current_batch_items_with_masks_for_stack_batch) >= self.batch_size: 
                                    # print(f"DEBUG QM [_worker V5.1]: Lot source plein ({len(current_batch_items_with_masks_for_stack_batch)}). Traitement du lot...")
                                    
                                    if self.drizzle_active_session: 
                                        # print(f"DEBUG QM [_worker V5.1]: Traitement Drizzle standard du lot (Mode: {self.drizzle_mode}).")
                                        batch_data_for_drizzle_processing = []
                                        for item_driz in current_batch_items_with_masks_for_stack_batch:
                                            wcs_to_use_for_driz_input = self.reference_wcs_object if self.reference_wcs_object else item_driz[3] 
                                            if item_driz[0] is not None and wcs_to_use_for_driz_input is not None:
                                                batch_data_for_drizzle_processing.append( (item_driz[0], item_driz[1], wcs_to_use_for_driz_input) )
                                            else:
                                                print(f"WARN QM [_worker V5.1]: Données ou WCS manquant pour Drizzle standard sur item. Ignoré.")

                                        if batch_data_for_drizzle_processing:
                                            if self.drizzle_output_wcs is None: 
                                                ref_shape_hw_driz = self.memmap_shape[:2] if self.memmap_shape else reference_image_data_for_global_alignment.shape[:2]
                                                if self.reference_wcs_object: 
                                                    (self.drizzle_output_wcs, self.drizzle_output_shape_hw) = self._create_drizzle_output_wcs(
                                                        self.reference_wcs_object, ref_shape_hw_driz, self.drizzle_scale
                                                    )
                                                else: 
                                                    self.processing_error = "WCS Ref Global Drizzle absent (milieu de lot)"; self.stop_processing = True; break
                                            
                                            if self.drizzle_output_wcs: 
                                                self.stacked_batches_count += 1
                                                # print(f"DEBUG QM [_worker V5.1]: Appel _process_and_save_drizzle_batch (standard) lot Drizzle #{self.stacked_batches_count}")
                                                sci_p, wht_ps_list = self._process_and_save_drizzle_batch(
                                                    batch_data_for_drizzle_processing, 
                                                    self.drizzle_output_wcs, 
                                                    self.drizzle_output_shape_hw, 
                                                    self.stacked_batches_count
                                                )
                                                if sci_p and wht_ps_list: 
                                                    self.intermediate_drizzle_batch_files.append((sci_p, wht_ps_list))
                                                    # print(f"DEBUG QM [_worker V5.1]: Lot Drizzle (standard) #{self.stacked_batches_count} sauvegardé.")
                                                else: 
                                                    self.failed_stack_count += len(batch_data_for_drizzle_processing)
                                                    print(f"WARN QM [_worker V5.1]: Échec _process_and_save_drizzle_batch lot Drizzle (standard) #{self.stacked_batches_count}")
                                        else: 
                                            print(f"WARN QM [_worker V5.1]: Aucune donnée valide pour _process_and_save_drizzle_batch lot Drizzle (standard).")
                                    
                                    elif not self.drizzle_active_session: 
                                        # print(f"DEBUG QM [_worker V5.1]: Traitement Classique (SUM/W) du lot source.")
                                        self.stacked_batches_count += 1
                                        self._process_completed_batch(current_batch_items_with_masks_for_stack_batch, self.stacked_batches_count, self.total_batches_estimated)
                                    
                                    current_batch_items_with_masks_for_stack_batch = [] 
                        else: 
                            print(f"DEBUG QM [_worker V5.1]: Fichier {file_name_for_log} skippé ou erreur (_process_file retourné None).")
                    
                    self.queue.task_done()
                
                except Empty: 
                    self.update_progress("ⓘ File d'attente vide. Vérification dernier lot et dossiers supplémentaires...")
                    
                    if not (self.is_mosaic_run and not use_local_aligner_for_this_mosaic_run) and current_batch_items_with_masks_for_stack_batch:
                        # print(f"DEBUG QM [_worker V5.1/Empty]: Traitement dernier lot source partiel ({len(current_batch_items_with_masks_for_stack_batch)}).")
                        if self.drizzle_active_session: 
                            batch_data_for_drizzle_processing = [] 
                            for item_driz in current_batch_items_with_masks_for_stack_batch:
                                wcs_to_use_for_driz_input = self.reference_wcs_object if self.reference_wcs_object else item_driz[3]
                                if item_driz[0] is not None and wcs_to_use_for_driz_input is not None: batch_data_for_drizzle_processing.append( (item_driz[0], item_driz[1], wcs_to_use_for_driz_input) )
                                else: print(f"WARN QM [_worker V5.1/Empty]: Données/WCS manquant Drizzle standard (dernier lot).")
                            if batch_data_for_drizzle_processing:
                                if self.drizzle_output_wcs is None: 
                                    ref_shape_hw_driz = self.memmap_shape[:2] if self.memmap_shape else reference_image_data_for_global_alignment.shape[:2]
                                    if self.reference_wcs_object: (self.drizzle_output_wcs, self.drizzle_output_shape_hw) = self._create_drizzle_output_wcs(self.reference_wcs_object, ref_shape_hw_driz, self.drizzle_scale)
                                    else: self.processing_error = "WCS Ref Drizzle Global absent (Empty/Final)"; self.stop_processing = True; break
                                if self.drizzle_output_wcs:
                                    self.stacked_batches_count += 1
                                    sci_p, wht_ps_list = self._process_and_save_drizzle_batch(batch_data_for_drizzle_processing, self.drizzle_output_wcs, self.drizzle_output_shape_hw, self.stacked_batches_count)
                                    if sci_p and wht_ps_list: self.intermediate_drizzle_batch_files.append((sci_p, wht_ps_list))
                                    else: self.failed_stack_count += len(batch_data_for_drizzle_processing)
                            # else: print(f"WARN QM [_worker V5.1/Empty]: Aucune donnée valide pour _process_and_save_drizzle_batch DERNIER lot Drizzle (standard).")
                        elif not self.drizzle_active_session: 
                            self.stacked_batches_count += 1
                            self._process_completed_batch(current_batch_items_with_masks_for_stack_batch, self.stacked_batches_count, self.total_batches_estimated)
                        current_batch_items_with_masks_for_stack_batch = [] 
                    
                    folder_to_process_next = None 
                    with self.folders_lock:
                        if self.additional_folders: 
                            folder_to_process_next = self.additional_folders.pop(0)
                            self.update_progress(f"folder_count_update:{len(self.additional_folders)}") 
                    
                    if folder_to_process_next: 
                        self.current_folder = folder_to_process_next
                        self.update_progress(f"📂 Passage au dossier supplémentaire : {os.path.basename(folder_to_process_next)}")
                        self._add_files_to_queue(folder_to_process_next) 
                        self._recalculate_total_batches() 
                    else: 
                        self.update_progress("✅ Fin de la file d'attente et des dossiers supplémentaires. Finalisation...")
                        break 
                
                except Exception as e_inner_loop: 
                    error_msg_loop = f"Erreur boucle worker: {type(e_inner_loop).__name__}: {e_inner_loop}"
                    print(f"ERREUR QM [_worker V5.1]: {error_msg_loop}"); traceback.print_exc(limit=2)
                    self.update_progress(f"⚠️ {error_msg_loop}")
                    self.failed_stack_count += 1 
                    if self.queue.unfinished_tasks > 0: self.queue.task_done() 
                
                finally: 
                    del aligned_data_item, header_item, quality_scores_item, wcs_object_indiv_item, valid_pixel_mask_item
                    if self.processed_files_count > 0 and self.processed_files_count % 20 == 0: # Moins fréquent
                        gc.collect() 
            
            # --- Fin de la boucle while principale ---

            print("DEBUG QM [_worker V5.1]: Sortie de la boucle principale. Début logique de finalisation...")
            print(f"  ÉTAT FINAL AVANT BLOC if/elif/else: stop_processing={self.stop_processing}, "
                  f"is_mosaic_run={self.is_mosaic_run} (Local Utilisé: {use_local_aligner_for_this_mosaic_run}), "
                  f"drizzle_active_session={self.drizzle_active_session}, drizzle_mode='{self.drizzle_mode}'")
            print(f"  Nombre d'items collectés pour mosaïque: {len(all_aligned_files_with_info_for_mosaic)}")
            print(f"  Nombre de fichiers Drizzle intermédiaires (non-mosaïque) collectés: {len(self.intermediate_drizzle_batch_files)}")

            if self.stop_processing: 
                # ... (logique inchangée)
                print("DEBUG QM [_worker V5.1]: Entrée dans branche 'self.stop_processing == True'")
                self.update_progress("🛑 Traitement interrompu avant sauvegarde finale.")
                if not self.is_mosaic_run and self.images_in_cumulative_stack > 0 and not (self.drizzle_active_session and self.drizzle_mode == "Incremental"): 
                    self.update_progress("   -> Tentative sauvegarde stack partiel (SUM/W Classique)...")
                    self._save_final_stack(output_filename_suffix="_sumw_stopped_partial", stopped_early=True)
                elif self.drizzle_active_session and self.intermediate_drizzle_batch_files and self.drizzle_mode in ["Final", "Incremental"]:
                     self.update_progress("   -> Drizzle interrompu. Pas de combinaison des lots intermédiaires.")
                     self.final_stacked_path = None 
                else: 
                    self.final_stacked_path = None
            
            elif self.is_mosaic_run:
                # ... (logique pour is_mosaic_run avec appel à _calculate_local_mosaic_output_grid comme avant)
                print("DEBUG QM [_worker V5.1]: Entrée dans branche 'self.is_mosaic_run == True'")
                print(f"DEBUG QM [_worker V5.1]: Préparation pour finalisation Mosaïque avec {len(all_aligned_files_with_info_for_mosaic)} items.")
                if not all_aligned_files_with_info_for_mosaic:
                    self.update_progress("   -> ERREUR Mosaïque: Aucun panneau à assembler.")
                    self.processing_error = "Aucun panneau pour mosaïque"
                else:
                    self.update_progress("🏁 Finalisation du traitement Mosaïque...")
                    
                    if use_local_aligner_for_this_mosaic_run:
                        print("DEBUG QM [_worker V5.1]: Mosaïque Locale -> Appel à _calculate_local_mosaic_output_grid (OMBB) pour la grille finale.")
                        if self.reference_wcs_object is None: 
                            self.update_progress("   -> ERREUR Mosaïque Locale: WCS d'ancrage manquant pour calcul grille optimisée (OMBB).")
                            self.processing_error = "WCS ancre manquant pour grille mosaïque locale (OMBB)"
                            self.drizzle_output_wcs = None 
                            self.drizzle_output_shape_hw = None
                        else:
                            temp_output_wcs, temp_output_shape_hw = self._calculate_local_mosaic_output_grid(
                                all_aligned_files_with_info_for_mosaic, 
                                self.reference_wcs_object 
                            )
                            if temp_output_wcs and temp_output_shape_hw:
                                self.drizzle_output_wcs = temp_output_wcs
                                self.drizzle_output_shape_hw = temp_output_shape_hw
                                self.update_progress(f"   -> Grille Mosaïque Locale (OMBB) calculée. Shape HW: {self.drizzle_output_shape_hw}")
                                print(f"DEBUG QM [_worker V5.1]: Grille Mosaïque Locale (OMBB) prête. WCS: {self.drizzle_output_wcs is not None}, Shape HW: {self.drizzle_output_shape_hw}")
                            else:
                                self.update_progress("   -> ERREUR Mosaïque Locale: Échec calcul grille de sortie optimisée (OMBB).")
                                self.processing_error = "Échec calcul grille mosaïque locale (OMBB)"
                                self.drizzle_output_wcs = None 
                                self.drizzle_output_shape_hw = None
                    else: 
                        print("DEBUG QM [_worker V5.1]: Mosaïque Astrometry -> La grille de sortie sera calculée DANS process_mosaic_from_aligned_files.")
                        self.drizzle_output_wcs = None
                        self.drizzle_output_shape_hw = None 
                    
                    process_mosaic_func = None
                    try: 
                        from ..enhancement.mosaic_processor import process_mosaic_from_aligned_files as pm_func
                        process_mosaic_func = pm_func
                    except ImportError: pass
                    
                    if process_mosaic_func:
                        # Si la grille a échoué à être calculée pour le mode local, drizzle_output_wcs sera None,
                        # ce qui fera échouer process_mosaic_from_aligned_files ou le poussera à essayer de recalculer la grille.
                        # On vérifie ici pour être sûr et éviter un appel inutile si la grille est déjà connue comme manquante.
                        if use_local_aligner_for_this_mosaic_run and self.drizzle_output_wcs is None:
                             print("ERREUR QM [_worker V5.1]: Grille locale OMBB non calculée, assemblage mosaïque annulé.")
                             if not self.processing_error: self.processing_error = "Grille locale OMBB manquante" # S'assurer qu'une erreur est enregistrée
                             self.final_stacked_path = None
                        else:
                            final_mosaic_data, final_mosaic_header = process_mosaic_func(
                                all_aligned_files_with_info_for_mosaic, 
                                self, 
                                self.update_progress
                            )
                            if final_mosaic_data is not None and final_mosaic_header is not None:
                                mosaic_filename = os.path.join(self.output_folder, "stack_final_mosaic_drizzle.fit") 
                                self.update_progress(f"   -> Sauvegarde de la mosaïque finale : {os.path.basename(mosaic_filename)}")
                                save_fits_image(final_mosaic_data, mosaic_filename, final_mosaic_header, overwrite=True)
                                self.final_stacked_path = mosaic_filename
                                self.last_saved_data_for_preview = final_mosaic_data.copy() 
                                self.update_progress("   -> Mosaïque finale sauvegardée avec succès.")
                            else: 
                                self.update_progress("   -> ERREUR: L'assemblage final de la mosaïque (dans process_mosaic...) a échoué.")
                                if not self.processing_error: self.processing_error = "Échec assemblage final mosaïque (process_mosaic...)"
                                self.final_stacked_path = None 
                    else: 
                        self.update_progress("   -> ERREUR CRITIQUE: La fonction process_mosaic_from_aligned_files n'a pas pu être importée.")
                        self.processing_error = "Module/Fonction d'assemblage mosaïque manquant(e)"
                        self.final_stacked_path = None
            
            elif self.drizzle_active_session and (self.drizzle_mode == "Final" or self.drizzle_mode == "Incremental"): 
                # ... (logique inchangée pour Drizzle standard) ...
                print(f"DEBUG QM [_worker V5.1]: Entrée dans branche 'DRIZZLE STANDARD (Mode: {self.drizzle_mode})'")
                self.update_progress(f"🏁 Finalisation Drizzle (Mode {self.drizzle_mode})...")
                if self.intermediate_drizzle_batch_files: 
                    print(f"DEBUG QM [_worker V5.1]: Combinaison de {len(self.intermediate_drizzle_batch_files)} lots Drizzle intermédiaires (standard).")
                    if self.drizzle_output_wcs is None or self.drizzle_output_shape_hw is None:
                         self.update_progress("   -> ERREUR: Grille de sortie Drizzle standard non définie pour combinaison finale.");
                         self.processing_error = "Grille sortie Drizzle standard manquante (combinaison)"; self.final_stacked_path = None
                    else:
                        final_sci_drizzle_combined, final_wht_drizzle_combined = self._combine_intermediate_drizzle_batches(
                            self.intermediate_drizzle_batch_files, 
                            self.drizzle_output_wcs, 
                            self.drizzle_output_shape_hw
                        )
                        if final_sci_drizzle_combined is not None and final_wht_drizzle_combined is not None:
                            print(f"DEBUG QM [_worker V5.1]: Drizzle (Mode {self.drizzle_mode}) - Combinaison des lots réussie.")
                            self.current_stack_header = self._update_header_for_drizzle_final() 
                            drizzle_suffix = "_drizzle_final" if self.drizzle_mode == "Final" else "_drizzle_incr_combined"
                            print(f"DEBUG QM [_worker V5.1]: Appel _save_final_stack pour Drizzle (Mode {self.drizzle_mode}) avec suffixe '{drizzle_suffix}'.")
                            self._save_final_stack(
                                output_filename_suffix=drizzle_suffix, 
                                stopped_early=False, 
                                drizzle_final_sci_data=final_sci_drizzle_combined, 
                                drizzle_final_wht_data=final_wht_drizzle_combined
                            )
                        else: 
                            self.update_progress(f"   -> ERREUR: Échec combinaison finale des lots Drizzle (Mode {self.drizzle_mode})."); 
                            self.processing_error = f"Échec combinaison Drizzle {self.drizzle_mode}"; self.final_stacked_path = None
                else: 
                    self.update_progress(f"   -> Aucun lot Drizzle intermédiaire à combiner pour Drizzle (Mode {self.drizzle_mode})."); 
                    self.final_stacked_path = None 

            elif not self.is_mosaic_run and not self.drizzle_active_session: 
                # ... (logique inchangée pour stacking classique)
                print("DEBUG QM [_worker V5.1]: Entrée dans branche 'STACKING CLASSIQUE (SUM/W)'")
                self.update_progress("🏁 Finalisation du stacking classique (SUM/W)...")
                if self.images_in_cumulative_stack > 0 or \
                   (hasattr(self, 'cumulative_sum_memmap') and self.cumulative_sum_memmap is not None and np.any(self.cumulative_sum_memmap)):
                    print(f"DEBUG QM [_worker V5.1]: Appel à _save_final_stack pour SUM/W classique. Images accumulées: {self.images_in_cumulative_stack}")
                    self._save_final_stack(output_filename_suffix="_classic_sumw", stopped_early=False)
                else: 
                    self.update_progress("   -> Aucune image accumulée pour le stacking classique. Pas de sauvegarde finale."); 
                    self.final_stacked_path = None
            
            else: 
                print(f"ERREUR QM [_worker V5.1]: État de finalisation non reconnu. Pas de sauvegarde finale."); 
                self.update_progress("❌ Erreur interne: État de finalisation non géré."); 
                self.processing_error = "État de finalisation non géré"; self.final_stacked_path = None

        except RuntimeError as rte: 
             error_msg_runtime = f"Erreur exécution critique (RuntimeError): {rte}"; 
             print(f"ERREUR QM [_worker V5.1]: {error_msg_runtime}"); self.update_progress(f"❌ {error_msg_runtime}"); 
             self.processing_error = str(rte); traceback.print_exc(limit=2) 
        except Exception as e_global: 
            error_msg_global = f"Erreur critique inattendue dans le worker: {type(e_global).__name__}: {e_global}"; 
            print(f"ERREUR QM [_worker V5.1]: {error_msg_global}"); self.update_progress(f"❌ {error_msg_global}"); 
            traceback.print_exc(limit=3); self.processing_error = error_msg_global
        
        finally: 
            print("DEBUG QM [_worker V5.1]: Entrée dans le bloc FINALLY du worker.") 
            self._close_memmaps() 
            
            if self.perform_cleanup:
                print("DEBUG QM [_worker V5.1]: Début du bloc de nettoyage (perform_cleanup=True).") 
                self.update_progress("🧹 Nettoyage final des fichiers temporaires...")
                self.cleanup_unaligned_files(); self.cleanup_temp_reference()
                self._cleanup_drizzle_temp_files(); self._cleanup_drizzle_batch_outputs()   
                self._cleanup_mosaic_panel_stacks_temp() 
                
                memmap_dir_final = os.path.join(self.output_folder, "memmap_accumulators") if self.output_folder else None
                if memmap_dir_final: 
                    if self.sum_memmap_path and os.path.exists(self.sum_memmap_path): 
                        try: os.remove(self.sum_memmap_path); print("   -> Fichier SUM.npy (worker finally) supprimé.") 
                        except Exception as e_del_sum: print(f"   WARN: Erreur suppression SUM.npy (worker finally): {e_del_sum}")
                    if self.wht_memmap_path and os.path.exists(self.wht_memmap_path): 
                        try: os.remove(self.wht_memmap_path); print("   -> Fichier WHT.npy (worker finally) supprimé.") 
                        except Exception as e_del_wht: print(f"   WARN: Erreur suppression WHT.npy (worker finally): {e_del_wht}")
                    try: 
                        if os.path.isdir(memmap_dir_final) and not os.listdir(memmap_dir_final): 
                            os.rmdir(memmap_dir_final); print(f"   -> Dossier memmap vide (worker finally) supprimé.")
                    except Exception as e_del_dir: print(f"   WARN: Erreur suppression dossier memmap (worker finally): {e_del_dir}")
            else: 
                self.update_progress("ⓘ Fichiers temporaires et memmap conservés (perform_cleanup=False).")
            
            print("   -> Vidage listes internes et appel garbage collector...")
            current_batch_items_with_masks_for_stack_batch = []
            all_aligned_files_with_info_for_mosaic = []
            self.intermediate_drizzle_batch_files = []
            gc.collect()
            
            self.processing_active = False 
            print("DEBUG QM [_worker V5.1]: Flag processing_active mis à False.")
            self.update_progress("🚪 Thread de traitement principal terminé.")






##########################################################################################################################################################

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

    def _update_preview_sum_w(self, downsample_factor=2):
        """
        Met à jour l'aperçu en utilisant les accumulateurs SUM et WHT.
        Calcule l'image moyenne, applique optionnellement le Low WHT Mask,
        normalise, sous-échantillonne et envoie au callback GUI.
        """
        print("DEBUG QM [_update_preview_sum_w]: Tentative de mise à jour de l'aperçu SUM/W...")

        if self.preview_callback is None:
            print("DEBUG QM [_update_preview_sum_w]: Callback preview non défini. Sortie.")
            return
        if self.cumulative_sum_memmap is None or self.cumulative_wht_memmap is None:
            print("DEBUG QM [_update_preview_sum_w]: Memmaps SUM ou WHT non initialisés. Sortie.")
            return

        try:
            print("DEBUG QM [_update_preview_sum_w]: Lecture des données depuis memmap...")
            # Lire en float64 pour la division pour maintenir la précision autant que possible
            current_sum = np.array(self.cumulative_sum_memmap, dtype=np.float64) # Shape (H, W, C)
            current_wht_map = np.array(self.cumulative_wht_memmap, dtype=np.float64) # Shape (H, W)
            print(f"DEBUG QM [_update_preview_sum_w]: Données lues. SUM shape={current_sum.shape}, WHT shape={current_wht_map.shape}")

            # Calcul de l'image moyenne (SUM / WHT)
            epsilon = 1e-9 # Pour éviter division par zéro
            wht_for_division = np.maximum(current_wht_map, epsilon)
            # Broadcaster wht_for_division (H,W) pour correspondre à current_sum (H,W,C)
            wht_broadcasted = wht_for_division[..., np.newaxis] 
            
            avg_img_fullres = None
            with np.errstate(divide='ignore', invalid='ignore'):
                avg_img_fullres = current_sum / wht_broadcasted
            avg_img_fullres = np.nan_to_num(avg_img_fullres, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"DEBUG QM [_update_preview_sum_w]: Image moyenne SUM/W calculée. Shape={avg_img_fullres.shape}")
            print(f"  Range avant normalisation 0-1: [{np.nanmin(avg_img_fullres):.4g}, {np.nanmax(avg_img_fullres):.4g}]")

            # --- NOUVEAU : Application du Low WHT Mask pour l'aperçu ---
            # Utiliser les settings stockés sur self (qui viennent de l'UI via SettingsManager)
            if hasattr(self, 'apply_low_wht_mask') and self.apply_low_wht_mask:
                if _LOW_WHT_MASK_AVAILABLE:
                    print("DEBUG QM [_update_preview_sum_w]: Application du Low WHT Mask pour l'aperçu...")
                    pct_low_wht = getattr(self, 'low_wht_percentile', 5)
                    soften_val_low_wht = getattr(self, 'low_wht_soften_px', 128)
                    
                    # La fonction apply_low_wht_mask attend une image déjà normalisée 0-1
                    # Donc, normalisons d'abord avg_img_fullres avant de l'appliquer.
                    temp_min_val = np.nanmin(avg_img_fullres)
                    temp_max_val = np.nanmax(avg_img_fullres)
                    avg_img_normalized_before_mask = avg_img_fullres # Par défaut
                    if temp_max_val > temp_min_val:
                        avg_img_normalized_before_mask = (avg_img_fullres - temp_min_val) / (temp_max_val - temp_min_val)
                    else:
                        avg_img_normalized_before_mask = np.zeros_like(avg_img_fullres)
                    avg_img_normalized_before_mask = np.clip(avg_img_normalized_before_mask, 0.0, 1.0).astype(np.float32)

                    avg_img_fullres = apply_low_wht_mask(
                        avg_img_normalized_before_mask, # Passer l'image normalisée 0-1
                        current_wht_map.astype(np.float32), # Passer la carte de poids originale (H,W)
                        percentile=pct_low_wht,
                        soften_px=soften_val_low_wht,
                        progress_callback=self.update_progress # Passer le callback pour les logs internes
                    )
                    # apply_low_wht_mask retourne déjà une image clippée 0-1 et en float32
                    print(f"DEBUG QM [_update_preview_sum_w]: Low WHT Mask appliqué à l'aperçu. Shape retournée: {avg_img_fullres.shape}")
                    print(f"  Range après Low WHT Mask (devrait être 0-1): [{np.nanmin(avg_img_fullres):.3f}, {np.nanmax(avg_img_fullres):.3f}]")
                else:
                    print("WARN QM [_update_preview_sum_w]: Low WHT Mask activé mais fonction non disponible (échec import). Aperçu non modifié.")
            else:
                print("DEBUG QM [_update_preview_sum_w]: Low WHT Mask non activé pour l'aperçu.")
            # --- FIN NOUVEAU ---

            # Normalisation finale 0-1 (nécessaire si Low WHT Mask n'a pas été appliqué,
            # ou pour re-normaliser si Low WHT Mask a modifié la plage de manière inattendue,
            # bien qu'il soit censé retourner 0-1). Une double normalisation ne nuit pas ici
            # car la première (avant mask) était pour la fonction mask, celle-ci est pour l'affichage.
            min_val_final = np.nanmin(avg_img_fullres)
            max_val_final = np.nanmax(avg_img_fullres)
            preview_data_normalized = avg_img_fullres # Par défaut si déjà 0-1
            if max_val_final > min_val_final:
                 preview_data_normalized = (avg_img_fullres - min_val_final) / (max_val_final - min_val_final)
            elif np.any(np.isfinite(avg_img_fullres)): # Image constante non nulle
                 preview_data_normalized = np.full_like(avg_img_fullres, 0.5) # Image grise
            else: # Image vide ou tout NaN/Inf
                 preview_data_normalized = np.zeros_like(avg_img_fullres)
            
            preview_data_normalized = np.clip(preview_data_normalized, 0.0, 1.0).astype(np.float32)
            print(f"DEBUG QM [_update_preview_sum_w]: Image APERÇU normalisée finale 0-1. Range: [{np.nanmin(preview_data_normalized):.3f}, {np.nanmax(preview_data_normalized):.3f}]")

            # Sous-échantillonnage pour l'affichage
            preview_data_to_send = preview_data_normalized
            if downsample_factor > 1:
                 try:
                     h, w = preview_data_normalized.shape[:2] # Fonctionne pour N&B (H,W) et Couleur (H,W,C)
                     new_h, new_w = h // downsample_factor, w // downsample_factor
                     if new_h > 10 and new_w > 10: # Éviter de réduire à une taille trop petite
                         # cv2.resize attend (W, H) pour dsize
                         preview_data_to_send = cv2.resize(preview_data_normalized, (new_w, new_h), interpolation=cv2.INTER_AREA)
                         print(f"DEBUG QM [_update_preview_sum_w]: Aperçu sous-échantillonné à {preview_data_to_send.shape}")
                 except Exception as e_resize:
                     print(f"ERREUR QM [_update_preview_sum_w]: Échec réduction taille APERÇU: {e_resize}")
                     # Continuer avec l'image pleine résolution si le resize échoue
            
            # Préparation du header et du nom pour le callback
            header_copy = self.current_stack_header.copy() if self.current_stack_header else fits.Header()
            # Ajouter/Mettre à jour les infos de l'aperçu dans le header
            header_copy['PREV_SRC'] = ('SUM/W Accumulators', 'Source data for this preview')
            if hasattr(self, 'apply_low_wht_mask') and self.apply_low_wht_mask:
                header_copy['PREV_LWM'] = (True, 'Low WHT Mask applied to this preview')
                header_copy['PREV_LWMP'] = (getattr(self, 'low_wht_percentile', 5), 'Low WHT Mask Percentile for preview')
                header_copy['PREV_LWMS'] = (getattr(self, 'low_wht_soften_px', 128), 'Low WHT Mask SoftenPx for preview')
            
            img_count = self.images_in_cumulative_stack
            total_imgs_est = self.files_in_queue
            current_batch_num = self.stacked_batches_count
            total_batches_est = self.total_batches_estimated
            stack_name_parts = ["Aperçu SUM/W"]
            if hasattr(self, 'apply_low_wht_mask') and self.apply_low_wht_mask:
                stack_name_parts.append("LWMask")
            stack_name_parts.append(f"({img_count}/{total_imgs_est} Img | Lot {current_batch_num}/{total_batches_est if total_batches_est > 0 else '?'})")
            stack_name = " ".join(stack_name_parts)

            print(f"DEBUG QM [_update_preview_sum_w]: Appel du callback preview avec image APERÇU shape {preview_data_to_send.shape}...")
            self.preview_callback(
                preview_data_to_send, 
                header_copy, 
                stack_name, 
                img_count, 
                total_imgs_est, 
                current_batch_num, 
                total_batches_est
            )
            print("DEBUG QM [_update_preview_sum_w]: Callback preview terminé.")

        except MemoryError as mem_err:
             print(f"ERREUR QM [_update_preview_sum_w]: ERREUR MÉMOIRE - {mem_err}")
             self.update_progress(f"❌ ERREUR MÉMOIRE pendant la mise à jour de l'aperçu SUM/W.")
             traceback.print_exc(limit=1)
        except Exception as e:
            print(f"ERREUR QM [_update_preview_sum_w]: Exception inattendue - {e}")
            self.update_progress(f"❌ Erreur inattendue pendant la mise à jour de l'aperçu SUM/W: {e}")
            traceback.print_exc(limit=2)




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
        Inspiré de full_drizzle.py corrigé pour conserver le même centre ciel.

        Args
        ----
        ref_wcs : astropy.wcs.WCS
            WCS de référence (doit être céleste et avoir pixel_shape).
        ref_shape_2d : tuple(int, int)
            (H, W) de l'image de référence.
        scale_factor : float
            Facteur d'échantillonnage Drizzle (>1 = sur-échantillonner).

        Returns
        -------
        (output_wcs, output_shape_hw)  où output_shape_hw = (H, W)
        """
        # ------------------ 0. Vérifications ------------------
        if not ref_wcs or not ref_wcs.is_celestial:
            raise ValueError("Référence WCS invalide ou non céleste pour Drizzle.")
        if ref_wcs.pixel_shape is None:
            raise ValueError("Référence WCS n'a pas de pixel_shape défini.")
        if len(ref_shape_2d) != 2:
            raise ValueError(f"Référence shape 2D (H,W) attendu, reçu {ref_shape_2d}")

        # ------------------ 1. Dimensions de sortie ------------------
        h_in,  w_in  = ref_shape_2d          # entrée (H,W)
        out_h = int(round(h_in * scale_factor))
        out_w = int(round(w_in * scale_factor))
        out_h = max(1, out_h); out_w = max(1, out_w)  # sécurité
        out_shape_hw = (out_h, out_w)        # (H,W) pour NumPy

        print(f"[DrizzleWCS] Scale={scale_factor}  -->  shape in={ref_shape_2d}  ->  out={out_shape_hw}")

        # ------------------ 2. Copier le WCS ------------------
        out_wcs = ref_wcs.deepcopy()

        # ------------------ 3. Ajuster l'échelle pixel ------------------
        scale_done = False
        try:
            # a) Matrice CD prioritaire
            if hasattr(out_wcs.wcs, 'cd') and out_wcs.wcs.cd is not None and np.any(out_wcs.wcs.cd):
                out_wcs.wcs.cd = ref_wcs.wcs.cd / scale_factor
                scale_done = True
                print("[DrizzleWCS] CD matrix divisée par", scale_factor)
            # b) Sinon CDELT (+ PC identité si absent)
            elif hasattr(out_wcs.wcs, 'cdelt') and out_wcs.wcs.cdelt is not None and np.any(out_wcs.wcs.cdelt):
                out_wcs.wcs.cdelt = ref_wcs.wcs.cdelt / scale_factor
                if not getattr(out_wcs.wcs, 'pc', None) is not None:
                    out_wcs.wcs.pc = np.identity(2)
                scale_done = True
                print("[DrizzleWCS] CDELT vector divisé par", scale_factor)
            else:
                raise ValueError("Input WCS lacks valid CD matrix and CDELT vector.")
        except Exception as e:
            raise ValueError(f"Failed to adjust pixel scale in output WCS: {e}")

        if not scale_done:
            raise ValueError("Could not adjust WCS scale.")

        # ------------------ 4. Recaler CRPIX ------------------
        # → garder le même point du ciel au même pixel relatif :
        #    CRPIX_out = CRPIX_in * scale_factor  (1‑based convention FITS)
        new_crpix = np.round(np.asarray(ref_wcs.wcs.crpix, dtype=float) * scale_factor, 6)
        out_wcs.wcs.crpix = new_crpix.tolist()
        print(f"[DrizzleWCS] CRPIX in={ref_wcs.wcs.crpix}  ->  out={out_wcs.wcs.crpix}")

        # ------------------ 5. Mettre à jour la taille interne ------------------
        out_wcs.pixel_shape = (out_w, out_h)   # (W,H) pour Astropy
        try:                                   # certains attributs privés selon versions
            out_wcs._naxis1 = out_w
            out_wcs._naxis2 = out_h
        except AttributeError:
            pass

        print(f"[DrizzleWCS] Output WCS OK  (shape={out_shape_hw})")
        return out_wcs, out_shape_hw





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
        """Définit la fonction de rappel pour les mises à jour de progression."""
        # print("DEBUG QM: Appel de set_progress_callback.") # Optionnel
        self.progress_callback = callback
        # Passer le callback à l'aligneur astroalign s'il existe
        if hasattr(self, 'aligner') and self.aligner is not None and hasattr(self.aligner, 'set_progress_callback') and callable(callback):
            try:
                # print("DEBUG QM: Tentative de configuration callback sur aligner (astroalign)...")
                self.aligner.set_progress_callback(callback)
                # print("DEBUG QM: Callback aligner (astroalign) configuré.")
            except Exception as e_align_cb: 
                print(f"Warning QM: Could not set progress callback on aligner (astroalign): {e_align_cb}")
        # Passer le callback à l'aligneur local s'il existe
        if hasattr(self, 'local_aligner_instance') and self.local_aligner_instance is not None and \
           hasattr(self.local_aligner_instance, 'set_progress_callback') and callable(callback):
            try:
                # print("DEBUG QM: Tentative de configuration callback sur local_aligner_instance...")
                self.local_aligner_instance.set_progress_callback(callback)
                # print("DEBUG QM: Callback local_aligner_instance configuré.")
            except Exception as e_local_cb:
                print(f"Warning QM: Could not set progress callback on local_aligner_instance: {e_local_cb}")

################################################################################################################################################




    def set_preview_callback(self, callback):
        """Définit la fonction de rappel pour les mises à jour de l'aperçu."""
        print("DEBUG QM: Appel de set_preview_callback (VERSION ULTRA PROPRE).") 
        self.preview_callback = callback
        
################################################################################################################################################





# --- DANS LA CLASSE SeestarQueuedStacker DANS seestar/queuep/queue_manager.py ---

 
        """
        Thread principal pour le traitement des images.
        MODIFIÉ (V4 - Mosaïque Locale): Stocke l'image originale du panneau et la matrice M
                                      pour l'alignement local.
        """
        print("\n" + "=" * 10 + " DEBUG QM [_worker V4 - Mosaïque Locale M]: Initialisation du worker " + "=" * 10) # MODIFIED PRINT

        # --- Initialisation des variables (identique à V3) ---
        self.processing_active = True; self.processing_error = None; start_time_session = time.monotonic()
        reference_image_data_for_global_alignment = None; reference_header_for_global_alignment = None
        mosaic_ref_panel_image_data = None; mosaic_ref_panel_header = None; mosaic_ref_panel_wcs_absolute = None
        current_batch_items_with_masks_for_stack_batch = []
        self.intermediate_drizzle_batch_files = []
        all_aligned_files_with_info_for_mosaic = []
        use_local_aligner_for_this_mosaic_run = (
            self.is_mosaic_run and 
            self.is_local_alignment_preferred_for_mosaic and 
            _LOCAL_ALIGNER_AVAILABLE and 
            self.local_aligner_instance is not None
        )
        print(f"DEBUG QM [_worker V4]: Mode -> is_mosaic_run={self.is_mosaic_run} "
              f"(Utilisation Aligneur Local: {use_local_aligner_for_this_mosaic_run}), "
              f"drizzle_active_session={self.drizzle_active_session}, drizzle_mode='{self.drizzle_mode}'")

        try:
            # --- 3.A Préparation de l’image de référence (Logique identique à V3) ---
            self.update_progress("⭐ Préparation de l'image de référence principale et/ou du premier panneau mosaïque...")
            # ... (code identique pour obtenir reference_image_data_for_global_alignment, etc.)
            if not self.current_folder or not os.path.isdir(self.current_folder): raise RuntimeError(f"Dossier d'entrée initial invalide : {self.current_folder}")
            initial_files_in_first_folder = sorted([f for f in os.listdir(self.current_folder) if f.lower().endswith((".fit", ".fits"))])
            if not initial_files_in_first_folder and not self.additional_folders: raise RuntimeError("Aucun fichier FITS initial pour référence principale/premier panneau.")
            self.aligner.correct_hot_pixels = self.correct_hot_pixels; self.aligner.hot_pixel_threshold = self.hot_pixel_threshold
            self.aligner.neighborhood_size = self.neighborhood_size; self.aligner.bayer_pattern = self.bayer_pattern
            print(f"DEBUG QM [_worker V4]: Appel _get_reference_image pour référence alignement général (astroalign)...")
            reference_image_data_for_global_alignment, reference_header_for_global_alignment = self.aligner._get_reference_image(self.current_folder, initial_files_in_first_folder)
            if reference_image_data_for_global_alignment is None or reference_header_for_global_alignment is None: raise RuntimeError("Échec obtention référence pour alignement général (astroalign).")
            self.reference_header_for_wcs = reference_header_for_global_alignment.copy()
            self.aligner._save_reference_image(reference_image_data_for_global_alignment, reference_header_for_global_alignment, self.output_folder)
            print("DEBUG QM [_worker V4]: Image de référence pour alignement général (astroalign) prête et sauvegardée.")

            if use_local_aligner_for_this_mosaic_run:
                self.update_progress("⭐ Mosaïque Locale: Traitement du panneau de référence...")
                mosaic_ref_panel_image_data = reference_image_data_for_global_alignment 
                mosaic_ref_panel_header = reference_header_for_global_alignment
                self.update_progress("   -> Mosaïque Locale: Résolution astrométrique du panneau de référence...")
                try: from ..enhancement.astrometry_solver import solve_image_wcs as solve_image_wcs_func
                except ImportError: solve_image_wcs_func = None
                if solve_image_wcs_func:
                    mosaic_ref_panel_wcs_absolute = solve_image_wcs_func(mosaic_ref_panel_image_data, mosaic_ref_panel_header, self.api_key,scale_est_arcsec_per_pix=self.reference_pixel_scale_arcsec,progress_callback=self.update_progress)
                else: self.update_progress("   -> ERREUR: solve_image_wcs non disponible."); mosaic_ref_panel_wcs_absolute = None
                if mosaic_ref_panel_wcs_absolute is None: raise RuntimeError("Mosaïque Locale: Échec plate-solving du panneau de référence.")
                print(f"DEBUG QM [_worker V4]: Mosaïque Locale: Panneau de référence résolu. WCS Absolu prêt.")
                self.reference_wcs_object = mosaic_ref_panel_wcs_absolute
                mat_identite = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
                valid_mask_ref_panel = np.ones(mosaic_ref_panel_image_data.shape[:2], dtype=bool)
                # <--- MODIFIÉ : Stocker l'image originale (déjà pré-traitée) du panneau de référence ---
                all_aligned_files_with_info_for_mosaic.append(
                    (mosaic_ref_panel_image_data.copy(),      # Image originale pré-traitée du panneau réf
                     mosaic_ref_panel_header.copy(),   
                     mosaic_ref_panel_wcs_absolute,    
                     mat_identite,                     
                     valid_mask_ref_panel)             
                )
                self.aligned_files_count += 1
                print(f"DEBUG QM [_worker V4]: Mosaïque Locale: Panneau de référence (original pré-traité) ajouté à la liste.")
            
            elif self.drizzle_active_session or (self.is_mosaic_run and not use_local_aligner_for_this_mosaic_run):
                # ... (logique plate-solve pour Drizzle standard / Mosaïque Astrometry identique à V3) ...
                self.update_progress("   -> Résolution astrométrique de la référence principale (pour Drizzle standard / Mosaïque Astrometry)...")
                try: from ..enhancement.astrometry_solver import solve_image_wcs as solve_image_wcs_func
                except ImportError: solve_image_wcs_func = None
                if solve_image_wcs_func: self.reference_wcs_object = solve_image_wcs_func(reference_image_data_for_global_alignment, self.reference_header_for_wcs, self.api_key, scale_est_arcsec_per_pix=self.reference_pixel_scale_arcsec, progress_callback=self.update_progress)
                else: self.update_progress("   -> ERREUR: Fonction solve_image_wcs non disponible."); self.reference_wcs_object = None
                if self.reference_wcs_object is None: raise RuntimeError("Échec plate-solving de la référence principale (Drizzle/Mosaïque Astrometry).")
                print(f"DEBUG QM [_worker V4]: WCS de référence principale obtenu (Astrometry).")

            self.update_progress("⭐ Référence(s) prête(s).", 5); self._recalculate_total_batches()
            self.update_progress(f"▶️ Démarrage boucle (En file: {self.files_in_queue} | Lots Estimés: {self.total_batches_estimated if self.total_batches_estimated > 0 else '?'})")

            is_first_panel_for_local_align_skipped = not use_local_aligner_for_this_mosaic_run

            while not self.stop_processing:
                file_path = None; # ... initialisations ...
                aligned_data_item = None; header_item = None; quality_scores_item = None
                wcs_object_indiv_item = None; valid_pixel_mask_item = None
                
                try:
                    file_path = self.queue.get(timeout=1.0)
                    file_name_for_log = os.path.basename(file_path)
                    
                    if use_local_aligner_for_this_mosaic_run:
                        print(f"DEBUG QM [_worker V4]: Traitement Mosaïque Locale pour fichier: {file_name_for_log}")
                        current_panel_is_ref = False
                        if mosaic_ref_panel_header is not None: # Assurer que le panneau de réf a été traité
                            # Identification plus robuste du panneau de référence par son chemin d'origine si possible
                            # On suppose que _get_reference_image a utilisé un fichier de initial_files_in_first_folder
                            # et que son chemin pourrait être stocké dans le header (ex: via une clé _SOURCE_PATH)
                            # ou que le premier fichier est toujours le panneau de réf.
                            # Pour ce test, on va se fier à ce que le premier item de la liste ait déjà été ajouté.
                            # Et que le file_path actuel ne doit pas être celui-là.
                            # Pour l'instant, on garde la logique de comparaison de header mais elle peut être faillible.
                            # Le mieux serait de connaître le chemin exact du fichier utilisé comme panneau de référence.
                            path_of_ref_panel_from_header = mosaic_ref_panel_header.get('_SOURCE_PATH', None) # Clé hypothétique
                            if path_of_ref_panel_from_header and os.path.normpath(file_path) == os.path.normpath(path_of_ref_panel_from_header):
                                current_panel_is_ref = True
                            elif not path_of_ref_panel_from_header and not is_first_panel_for_local_align_skipped:
                                # Si on n'a pas le chemin et qu'on n'a pas encore skippé le premier, on suppose que c'est lui
                                # (Moins robuste)
                                temp_hdr = fits.getheader(file_path)
                                if temp_hdr.get('DATE-OBS') == mosaic_ref_panel_header.get('DATE-OBS'): current_panel_is_ref = True
                        
                        if current_panel_is_ref and not is_first_panel_for_local_align_skipped:
                            print(f"DEBUG QM [_worker V4]: Fichier {file_name_for_log} identifié comme panneau de référence (déjà ajouté). Consommé.")
                            self.processed_files_count += 1; self.queue.task_done()
                            is_first_panel_for_local_align_skipped = True; continue 

                        self.update_progress(f"   -> Mosaïque Locale: Alignement local de {file_name_for_log} sur panneau de référence...")
                        current_panel_data_loaded = load_and_validate_fits(file_path)
                        if current_panel_data_loaded is None: raise ValueError(f"Échec chargement {file_name_for_log} pour align. local.")
                        current_panel_header = fits.getheader(file_path)
                        # Stocker le chemin source dans le header pour identification future si besoin
                        current_panel_header['_SOURCE_PATH'] = file_path 

                        current_panel_data_processed = current_panel_data_loaded.astype(np.float32)
                        # ... (pré-traitement debayer, hp pour current_panel_data_processed identique à V3)
                        if current_panel_data_processed.ndim == 2:
                            bayer_pat = current_panel_header.get('BAYERPAT', self.bayer_pattern if hasattr(self, 'bayer_pattern') else None)
                            if bayer_pat and isinstance(bayer_pat, str) and bayer_pat.upper() in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                                try: current_panel_data_processed = debayer_image(current_panel_data_processed, bayer_pat.upper())
                                except Exception as e_deb: print(f"WARN QM: Debayering échoué pour panneau local {file_name_for_log}: {e_deb}")
                        if self.correct_hot_pixels:
                            try: current_panel_data_processed = detect_and_correct_hot_pixels(current_panel_data_processed, self.hot_pixel_threshold, self.neighborhood_size)
                            except Exception as e_hp: print(f"WARN QM: Correction HP échouée pour panneau local {file_name_for_log}: {e_hp}")

                        if self.local_aligner_instance.set_progress_callback is not None : self.local_aligner_instance.set_progress_callback(self.progress_callback)
                        
                        # FastSeestarAligner retourne (aligned_image, M_matrix, success_bool)
                        # On a besoin de l'image originale et de M pour la mosaïque Drizzle.
                        # L'image "aligned" retournée ici par FastSeestarAligner est l'image source warpée.
                        # Nous allons stocker l'image SOURCE (current_panel_data_processed) et la matrice M.
                        _aligned_img_temp, M_transform, align_success = self.local_aligner_instance._align_image(
                            current_panel_data_processed,  # Image source à aligner
                            mosaic_ref_panel_image_data,   # Image du panneau de référence (pré-traité)
                            file_name_for_log
                        )
                        # On ignore _aligned_img_temp pour le stockage mosaïque, on garde l'original + M

                        self.processed_files_count += 1
                        if align_success and M_transform is not None:
                            self.aligned_files_count += 1
                            # Stocker: (données_originales_pré-traitées, header_original, WCS_absolu_DU_REF_PANEL, Matrice_M_vers_panneau_ref, masque=tout_valide)
                            # Le WCS associé à current_panel_data_processed est implicitement celui du panneau de référence
                            # une fois la matrice M appliquée. Pour Drizzle, on aura besoin de l'image originale
                            # et d'une manière de la projeter sur la grille finale via M et le WCS du panneau de référence.
                            valid_mask_this_panel = np.ones(current_panel_data_processed.shape[:2], dtype=bool) # Originale est toute valide
                            all_aligned_files_with_info_for_mosaic.append(
                                (current_panel_data_processed.copy(), # <--- Image originale pré-traitée
                                 current_panel_header.copy(),   
                                 mosaic_ref_panel_wcs_absolute, # Le WCS sur lequel M s'applique pour atteindre le référentiel
                                 M_transform.copy(),            # <--- Matrice M
                                 valid_mask_this_panel)      
                            )
                            print(f"DEBUG QM [_worker V4]: Mosaïque Locale: Panneau {file_name_for_log} aligné (M stockée) et ajouté.")
                        else:
                            self.update_progress(f"   -> Mosaïque Locale: Échec alignement local (M non trouvée) pour {file_name_for_log}. Ignoré.")
                            self.failed_align_count +=1
                        
                        del current_panel_data_loaded, current_panel_header, current_panel_data_processed, _aligned_img_temp, M_transform
                    
                    else: # Cas NON-Mosaïque Locale
                        aligned_data_item, header_item, quality_scores_item, wcs_object_indiv_item, valid_pixel_mask_item = (
                            self._process_file(file_path, reference_image_data_for_global_alignment)
                        )
                        # ... (reste de la logique _process_file et gestion de lot identique à V3) ...
                        self.processed_files_count += 1
                        if aligned_data_item is not None and valid_pixel_mask_item is not None:
                            self.aligned_files_count += 1; current_item_tuple = (aligned_data_item, header_item, quality_scores_item, wcs_object_indiv_item, valid_pixel_mask_item)
                            if self.is_mosaic_run: all_aligned_files_with_info_for_mosaic.append(current_item_tuple); print(f"DEBUG QM [_worker V4]: Item {self.aligned_files_count} ajouté pour MOSAÏQUE (Astrometry).")
                            else:
                                current_batch_items_with_masks_for_stack_batch.append(current_item_tuple); print(f"DEBUG QM [_worker V4]: Item {self.aligned_files_count} ajouté au lot source (taille: {len(current_batch_items_with_masks_for_stack_batch)}).")
                                if len(current_batch_items_with_masks_for_stack_batch) >= self.batch_size: # Lot plein
                                    print(f"DEBUG QM [_worker V4]: Lot source plein ({len(current_batch_items_with_masks_for_stack_batch)}). Traitement...")
                                    if self.drizzle_active_session:
                                        print(f"DEBUG QM [_worker V4]: Traitement Drizzle lot source (Mode: {self.drizzle_mode})."); batch_data_for_drizzle_processing = []
                                        for item_driz in current_batch_items_with_masks_for_stack_batch:
                                            if item_driz[0] is not None and self.reference_wcs_object is not None: batch_data_for_drizzle_processing.append( (item_driz[0], item_driz[1], self.reference_wcs_object) )
                                            elif item_driz[0] is not None and self.reference_wcs_object is None: print(f"WARN QM [_worker V4]: WCS réf global absent pour Drizzle."); batch_data_for_drizzle_processing.append( (item_driz[0], item_driz[1], item_driz[3]) )
                                        if batch_data_for_drizzle_processing:
                                            if self.drizzle_output_wcs is None: 
                                                ref_shape_hw_driz = self.memmap_shape[:2] if self.memmap_shape else reference_image_data_for_global_alignment.shape[:2]
                                                if self.reference_wcs_object: (self.drizzle_output_wcs, self.drizzle_output_shape_hw) = self._create_drizzle_output_wcs(self.reference_wcs_object, ref_shape_hw_driz, self.drizzle_scale)
                                                else: self.processing_error = "WCS Ref Drizzle absent"; self.stop_processing = True; break
                                            if self.drizzle_output_wcs:
                                                self.stacked_batches_count += 1; print(f"DEBUG QM [_worker V4]: Appel _process_and_save_drizzle_batch lot Drizzle #{self.stacked_batches_count}")
                                                sci_p, wht_ps_list = self._process_and_save_drizzle_batch(batch_data_for_drizzle_processing, self.drizzle_output_wcs, self.drizzle_output_shape_hw, self.stacked_batches_count)
                                                if sci_p and wht_ps_list: self.intermediate_drizzle_batch_files.append((sci_p, wht_ps_list)); print(f"DEBUG QM [_worker V4]: Lot Drizzle #{self.stacked_batches_count} sauvegardé.")
                                                else: self.failed_stack_count += len(batch_data_for_drizzle_processing); print(f"WARN QM [_worker V4]: Échec _process_and_save_drizzle_batch lot Drizzle #{self.stacked_batches_count}")
                                        else: print(f"WARN QM [_worker V4]: Aucune donnée valide pour _process_and_save_drizzle_batch lot Drizzle.")
                                    elif not self.drizzle_active_session: 
                                        print(f"DEBUG QM [_worker V4]: Traitement Classique (SUM/W) lot source."); self.stacked_batches_count += 1
                                        self._process_completed_batch(current_batch_items_with_masks_for_stack_batch, self.stacked_batches_count, self.total_batches_estimated)
                                    current_batch_items_with_masks_for_stack_batch = []
                        else: print(f"DEBUG QM [_worker V4]: Fichier {file_name_for_log} skippé (_process_file retourné None).")
                    self.queue.task_done()
                
                except Empty: # Queue vide
                    self.update_progress("ⓘ File d'attente vide. Vérification dernier lot et dossiers supplémentaires...")
                    if not (self.is_mosaic_run and not use_local_aligner_for_this_mosaic_run) and current_batch_items_with_masks_for_stack_batch:
                        print(f"DEBUG QM [_worker V4/Empty]: Traitement dernier lot source partiel ({len(current_batch_items_with_masks_for_stack_batch)}).")
                        # ... (copier/coller la logique de gestion du dernier lot de V3 ici, pour Drizzle ou Classique)
                        if self.drizzle_active_session: 
                            print(f"DEBUG QM [_worker V4/Empty]: Traitement Drizzle DERNIER lot source (Mode: {self.drizzle_mode})."); batch_data_for_drizzle_processing = []
                            for item_driz in current_batch_items_with_masks_for_stack_batch:
                                if item_driz[0] is not None and self.reference_wcs_object is not None: batch_data_for_drizzle_processing.append( (item_driz[0], item_driz[1], self.reference_wcs_object) )
                                elif item_driz[0] is not None and self.reference_wcs_object is None : print(f"WARN QM [_worker V4/Empty]: WCS réf global absent pour Drizzle (dernier lot)."); batch_data_for_drizzle_processing.append( (item_driz[0], item_driz[1], item_driz[3]) )
                            if batch_data_for_drizzle_processing:
                                if self.drizzle_output_wcs is None: 
                                    ref_shape_hw_driz = self.memmap_shape[:2] if self.memmap_shape else reference_image_data_for_global_alignment.shape[:2]
                                    if self.reference_wcs_object: (self.drizzle_output_wcs, self.drizzle_output_shape_hw) = self._create_drizzle_output_wcs(self.reference_wcs_object, ref_shape_hw_driz, self.drizzle_scale)
                                    else: self.processing_error = "WCS Ref Drizzle absent (Empty/Final)"; self.stop_processing = True; break
                                if self.drizzle_output_wcs:
                                    self.stacked_batches_count += 1; print(f"DEBUG QM [_worker V4/Empty]: Appel _process_and_save_drizzle_batch DERNIER lot Drizzle #{self.stacked_batches_count}")
                                    sci_p, wht_ps_list = self._process_and_save_drizzle_batch(batch_data_for_drizzle_processing, self.drizzle_output_wcs, self.drizzle_output_shape_hw, self.stacked_batches_count)
                                    if sci_p and wht_ps_list: self.intermediate_drizzle_batch_files.append((sci_p, wht_ps_list)); print(f"DEBUG QM [_worker V4/Empty]: DERNIER Lot Drizzle #{self.stacked_batches_count} sauvegardé.")
                                    else: self.failed_stack_count += len(batch_data_for_drizzle_processing); print(f"WARN QM [_worker V4/Empty]: Échec _process_and_save_drizzle_batch DERNIER lot Drizzle.")
                            else: print(f"WARN QM [_worker V4/Empty]: Aucune donnée valide pour _process_and_save_drizzle_batch DERNIER lot Drizzle.")
                        elif not self.drizzle_active_session: 
                            print(f"DEBUG QM [_worker V4/Empty]: Traitement Classique (SUM/W) DERNIER lot source."); self.stacked_batches_count += 1
                            self._process_completed_batch(current_batch_items_with_masks_for_stack_batch, self.stacked_batches_count, self.total_batches_estimated)
                        current_batch_items_with_masks_for_stack_batch = []
                    folder_to_process_next = None # ... (logique dossiers additionnels inchangée) ...
                    with self.folders_lock:
                        if self.additional_folders: folder_to_process_next = self.additional_folders.pop(0); self.update_progress(f"folder_count_update:{len(self.additional_folders)}")
                    if folder_to_process_next:
                        self.current_folder = folder_to_process_next; self.update_progress(f"📂 Passage au dossier supplémentaire : {os.path.basename(folder_to_process_next)}")
                        self._add_files_to_queue(folder_to_process_next); self._recalculate_total_batches()
                    else: self.update_progress("✅ Fin de la file d'attente et des dossiers supplémentaires."); break # Sortir boucle while
                
                except Exception as e_inner_loop: # ... (gestion erreur inchangée) ...
                    error_msg_loop = f"Erreur boucle worker: {type(e_inner_loop).__name__}: {e_inner_loop}"; print(f"ERREUR QM [_worker V4]: {error_msg_loop}"); traceback.print_exc(limit=2)
                    self.update_progress(f"⚠️ {error_msg_loop}"); self.failed_stack_count += 1
                    if self.queue.unfinished_tasks > 0: self.queue.task_done()
                finally: # ... (nettoyage mémoire inchangé) ...
                    del aligned_data_item, header_item, quality_scores_item, wcs_object_indiv_item, valid_pixel_mask_item
                    if self.processed_files_count % 10 == 0: gc.collect() # Peut-être un peu plus fréquent
            # --- Fin de la boucle while principale ---

            # --- 3.C Traitement final après la boucle (logique if/elif/else inchangée par rapport à V3 de _worker) ---
            print("DEBUG QM [_worker V4]: Sortie de la boucle principale. Début logique de finalisation...")
            # ... (copier/coller le bloc de finalisation de la V3 de _worker ici) ...
            # ... (il commence par le print "ÉTAT FINAL AVANT BLOC if/elif/else" et va jusqu'à la fin du bloc `try` principal)
            print(f"  ÉTAT FINAL AVANT BLOC if/elif/else: stop_processing={self.stop_processing}, "
                  f"is_mosaic_run={self.is_mosaic_run} (Local Pref: {self.is_local_alignment_preferred_for_mosaic}, Local Avail: {use_local_aligner_for_this_mosaic_run}), "
                  f"drizzle_active_session={self.drizzle_active_session}, drizzle_mode='{self.drizzle_mode}'")
            print(f"  Nombre d'items pour mosaïque collectés: {len(all_aligned_files_with_info_for_mosaic)}")
            print(f"  Nombre de fichiers intermédiaires Drizzle (non-mosaïque) collectés: {len(self.intermediate_drizzle_batch_files)}")

            if self.stop_processing: 
                print("DEBUG QM [_worker V4]: Entrée dans branche 'self.stop_processing == True'")
                self.update_progress("🛑 Traitement interrompu avant sauvegarde finale.")
                if not self.is_mosaic_run and self.images_in_cumulative_stack > 0 and not (self.drizzle_active_session and self.drizzle_mode == "Incremental"): 
                    self.update_progress("   -> Tentative sauvegarde stack partiel (SUM/W Classique)...")
                    self._save_final_stack(output_filename_suffix="_sumw_stopped_partial", stopped_early=True)
                elif self.drizzle_active_session and self.intermediate_drizzle_batch_files and self.drizzle_mode in ["Final", "Incremental"]:
                     self.update_progress("   -> Drizzle interrompu. Pas de combinaison des lots intermédiaires.")
                     self.final_stacked_path = None
                else: self.final_stacked_path = None
            
            elif self.is_mosaic_run:
                print("DEBUG QM [_worker V4]: Entrée dans branche 'self.is_mosaic_run == True'")
                print(f"DEBUG QM [_worker V4]: Préparation pour finalisation Mosaïque avec {len(all_aligned_files_with_info_for_mosaic)} items.")
                if not all_aligned_files_with_info_for_mosaic:
                    self.update_progress("   -> ERREUR Mosaïque: Aucun panneau (aligné localement ou via Astrometry) à assembler.")
                    self.processing_error = "Aucun panneau pour mosaïque"
                else:
                    self.update_progress("🏁 Finalisation du traitement Mosaïque...")
                    try: from ..enhancement.mosaic_processor import process_mosaic_from_aligned_files
                    except ImportError: process_mosaic_from_aligned_files = None
                    if process_mosaic_from_aligned_files:
                        final_mosaic_data, final_mosaic_header = process_mosaic_from_aligned_files(all_aligned_files_with_info_for_mosaic, self, self.update_progress)
                        if final_mosaic_data is not None and final_mosaic_header is not None:
                            mosaic_filename = os.path.join(self.output_folder, "stack_final_mosaic_drizzle.fit") 
                            self.update_progress(f"   -> Sauvegarde de la mosaïque finale : {os.path.basename(mosaic_filename)}")
                            save_fits_image(final_mosaic_data, mosaic_filename, final_mosaic_header, overwrite=True)
                            self.final_stacked_path = mosaic_filename; self.last_saved_data_for_preview = final_mosaic_data.copy()
                            self.update_progress("   -> Mosaïque finale sauvegardée.")
                        else: self.update_progress("   -> ERREUR: L'assemblage final de la mosaïque a échoué."); self.processing_error = "Échec assemblage mosaïque"
                    else: self.update_progress("   -> ERREUR CRITIQUE: process_mosaic_from_aligned_files non importable."); self.processing_error = "Module mosaïque manquant"

            elif self.drizzle_active_session and (self.drizzle_mode == "Final" or self.drizzle_mode == "Incremental"): 
                print(f"DEBUG QM [_worker V4]: Entrée dans branche 'DRIZZLE (Mode: {self.drizzle_mode})'")
                self.update_progress(f"🏁 Finalisation Drizzle (Mode {self.drizzle_mode})...")
                if self.intermediate_drizzle_batch_files:
                    print(f"DEBUG QM [_worker V4]: Combinaison de {len(self.intermediate_drizzle_batch_files)} lots Drizzle intermédiaires.")
                    final_sci_drizzle_combined, final_wht_drizzle_combined = self._combine_intermediate_drizzle_batches(self.intermediate_drizzle_batch_files, self.drizzle_output_wcs, self.drizzle_output_shape_hw)
                    if final_sci_drizzle_combined is not None and final_wht_drizzle_combined is not None:
                        print(f"DEBUG QM [_worker V4]: Drizzle (Mode {self.drizzle_mode}) - Combinaison des lots réussie.")
                        self.current_stack_header = self._update_header_for_drizzle_final()
                        drizzle_suffix = "_drizzle_final" if self.drizzle_mode == "Final" else "_drizzle_incr_combined"
                        print(f"DEBUG QM [_worker V4]: Appel _save_final_stack pour Drizzle (Mode {self.drizzle_mode}) avec suffixe '{drizzle_suffix}'.")
                        self._save_final_stack(output_filename_suffix=drizzle_suffix, stopped_early=False, drizzle_final_sci_data=final_sci_drizzle_combined, drizzle_final_wht_data=final_wht_drizzle_combined)
                    else: self.update_progress(f"   -> ERREUR: Échec combinaison finale des lots Drizzle (Mode {self.drizzle_mode})."); self.processing_error = f"Échec combinaison Drizzle {self.drizzle_mode}"; self.final_stacked_path = None
                else: self.update_progress(f"   -> Aucun lot Drizzle intermédiaire à combiner pour Drizzle (Mode {self.drizzle_mode})."); self.final_stacked_path = None

            elif not self.is_mosaic_run and not self.drizzle_active_session: 
                print("DEBUG QM [_worker V4]: Entrée dans branche 'STACKING CLASSIQUE (SUM/W)'")
                self.update_progress("🏁 Finalisation du stacking classique (SUM/W)...")
                if self.images_in_cumulative_stack > 0 or (self.cumulative_sum_memmap is not None and np.any(self.cumulative_sum_memmap)):
                    print(f"DEBUG QM [_worker V4]: Appel à _save_final_stack pour SUM/W classique. Images accumulées: {self.images_in_cumulative_stack}")
                    self._save_final_stack(output_filename_suffix="_classic_sumw", stopped_early=False)
                else: self.update_progress("   -> Aucune image accumulée pour le stacking classique."); self.final_stacked_path = None
            else: 
                print(f"ERREUR QM [_worker V4]: État de finalisation non reconnu. Pas de sauvegarde finale."); self.update_progress("❌ Erreur interne: État finalisation non géré."); self.processing_error = "État finalisation non géré"; self.final_stacked_path = None


        except RuntimeError as rte: # ... (inchangé)
             error_msg_runtime = f"Erreur exécution critique: {rte}"; print(f"ERREUR QM [_worker V4]: {error_msg_runtime}"); self.update_progress(f"❌ {error_msg_runtime}"); self.processing_error = str(rte)
        except Exception as e_global: # ... (inchangé)
            error_msg_global = f"Erreur critique worker: {type(e_global).__name__}: {e_global}"; print(f"ERREUR QM [_worker V4]: {error_msg_global}"); self.update_progress(f"❌ {error_msg_global}"); traceback.print_exc(limit=3); self.processing_error = error_msg_global
        
        finally: # ... (inchangé, sauf les prints V4)
            print("DEBUG QM [_worker V4]: Entrée dans le bloc FINALLY du worker.")
            self._close_memmaps()
            if self.perform_cleanup:
                self.update_progress("🧹 Nettoyage final des fichiers temporaires...")
                self.cleanup_unaligned_files()
                self.cleanup_temp_reference()
                self._cleanup_drizzle_temp_files()
                self._cleanup_drizzle_batch_outputs()
                self._cleanup_mosaic_panel_stacks_temp()
                #memmap_dir_final = os.path.join(self.output_folder, "memmap_accumulators")
                #if self.sum_memmap_path and os.path.exists(self.sum_memmap_path): 
                #    try: os.remove(self.sum_memmap_path); print("   -> SUM.npy (worker V4 finally) supprimé.")
                #    except Exception as e: print(f"   WARN: Erreur suppression SUM.npy: {e}")
                #if self.wht_memmap_path and os.path.exists(self.wht_memmap_path):
                #    try: os.remove(self.wht_memmap_path); print("   -> WHT.npy (worker V4 finally) supprimé.") 
                #    except Exception as e: print(f"   WARN: Erreur suppression WHT.npy: {e}")
                #try:
                #    if os.path.isdir(memmap_dir_final) and not os.listdir(memmap_dir_final): os.rmdir(memmap_dir_final); print(f"   -> Dossier memmap vide (worker V4 finally) supprimé.")
                #except Exception: pass
            else: self.update_progress("ⓘ Fichiers temporaires et memmap conservés.")
            print("   -> Vidage listes internes et GC...")
            current_batch_items_with_masks_for_stack_batch = []; all_aligned_files_with_info_for_mosaic = []; self.intermediate_drizzle_batch_files = []
            gc.collect()
            self.processing_active = False; print("DEBUG QM [_worker V4]: Flag processing_active mis à False.")
            self.update_progress("🚪 Thread de traitement principal terminé.")






############################################################################################################################




    @staticmethod
    def _project_to_tangent_plane(sky_coords_obj: SkyCoord, tangent_point_sky: SkyCoord):
        """
        Projete des coordonnées célestes sur un plan tangent.

        Args:
            sky_coords_obj (SkyCoord): Coordonnées célestes à projeter.
            tangent_point_sky (SkyCoord): Point de tangence (centre de la projection).

        Returns:
            np.ndarray: Array de points (x, y) projetés en arcsecondes sur le plan tangent.
                        L'origine (0,0) du plan tangent correspond à tangent_point_sky.
        """
        # Créer un frame de projection centré sur le point de tangence
        # SkyOffsetFrame représente les offsets angulaires par rapport à un point central.
        # Ces offsets (lon, lat) sont essentiellement des coordonnées sur le plan tangent.
        skyoffset_frame = tangent_point_sky.skyoffset_frame()
        coords_in_offset_frame = sky_coords_obj.transform_to(skyoffset_frame)

        # Extraire les longitudes et latitudes dans ce frame (en arcsecondes)
        # .lon et .lat dans SkyOffsetFrame sont les coordonnées tangentielles.
        projected_x_arcsec = coords_in_offset_frame.lon.to(u.arcsec).value
        projected_y_arcsec = coords_in_offset_frame.lat.to(u.arcsec).value
        
        # print(f"DEBUG _project_to_tangent_plane: SkyCoords (premier): {sky_coords_obj[0].ra.deg:.3f}, {sky_coords_obj[0].dec.deg:.3f}")
        # print(f"DEBUG _project_to_tangent_plane: Tangent Point: {tangent_point_sky.ra.deg:.3f}, {tangent_point_sky.dec.deg:.3f}")
        # print(f"DEBUG _project_to_tangent_plane: Projected (premier): x={projected_x_arcsec[0]:.2f}\", y={projected_y_arcsec[0]:.2f}\"")
        
        return np.column_stack((projected_x_arcsec, projected_y_arcsec))

    @staticmethod
    def _deproject_from_tangent_plane(xy_arcsec_array: np.ndarray, tangent_point_sky: SkyCoord):
        """
        Dé-projete des coordonnées d'un plan tangent vers des coordonnées célestes.

        Args:
            xy_arcsec_array (np.ndarray): Array de points (x, y) en arcsecondes sur le plan tangent.
            tangent_point_sky (SkyCoord): Point de tangence utilisé pour la projection initiale.

        Returns:
            SkyCoord: Objet SkyCoord contenant les coordonnées célestes dé-projetées.
        """
        skyoffset_frame = tangent_point_sky.skyoffset_frame()
        
        # Créer des SkyCoord à partir des coordonnées du plan tangent, dans le SkyOffsetFrame
        # lon et lat dans SkyOffsetFrame correspondent à nos x et y projetés.
        coords_on_tangent_plane = SkyCoord(
            lon=xy_arcsec_array[:, 0] * u.arcsec,
            lat=xy_arcsec_array[:, 1] * u.arcsec,
            frame=skyoffset_frame
        )
        
        # Transformer ces coordonnées retour vers le système céleste de base (ex: ICRS)
        deprojected_sky_coords = coords_on_tangent_plane.transform_to(tangent_point_sky.frame) # Utiliser le frame du point de tangence
        
        # print(f"DEBUG _deproject_from_tangent_plane: Input XY (premier): {xy_arcsec_array[0,0]:.2f}\", {xy_arcsec_array[0,1]:.2f}\"")
        # print(f"DEBUG _deproject_from_tangent_plane: Deprojected (premier): RA={deprojected_sky_coords[0].ra.deg:.3f}, Dec={deprojected_sky_coords[0].dec.deg:.3f}")

        return deprojected_sky_coords

##########################################################################################################################





# --- DANS LA CLASSE SeestarQueuedStacker DANS seestar/queuep/queue_manager.py ---

    def _calculate_local_mosaic_output_grid(self, 
                                            panel_info_list_for_grid: list, 
                                            anchor_wcs: WCS):
        """
        Version: V_OMBB_SnapToAxes
        OMBB pour dimensions et centre, puis orientation "snappée" aux axes cardinaux.
        """
        num_panels = len(panel_info_list_for_grid)
        print(f"DEBUG QM [_calculate_local_mosaic_output_grid V_OMBB_SnapToAxes]: Début pour {num_panels} panneaux...")
        # ... (calcul de all_corners_flat_skycoord, tangent_point_sky, tangent_plane_points_arcsec, hull_points_arcsec comme avant)
        # ... jusqu'à obtenir rect de cv2.minAreaRect
        # Les premières parties sont identiques à V_OMBB_Fix5
        all_sky_corners_list = [] 
        anchor_frame_name = 'icrs' 
        if hasattr(anchor_wcs, 'wcs') and hasattr(anchor_wcs.wcs, 'radesys') and anchor_wcs.wcs.radesys:
            radesys_val = str(anchor_wcs.wcs.radesys).strip().lower()
            if radesys_val in ['icrs', 'fk5', 'fk4', 'galactic']: anchor_frame_name = radesys_val
        for i, panel_info in enumerate(panel_info_list_for_grid):
            try:
                img_data_orig = panel_info[0]; transform_M = panel_info[3] 
                if img_data_orig is None or transform_M is None: continue
                original_h, original_w = img_data_orig.shape[:2]
                pixel_corners_orig = np.array([[0.,0.],[original_w-1.,0.],[original_w-1.,original_h-1.],[0.,original_h-1.]], dtype=np.float32).reshape(-1,1,2) 
                corners_in_anchor_pixels = cv2.transform(pixel_corners_orig, transform_M).reshape(-1,2)
                ra_coords_deg, dec_coords_deg = anchor_wcs.all_pix2world(corners_in_anchor_pixels[:,0], corners_in_anchor_pixels[:,1],0)
                sky_corners_panel_obj = SkyCoord(ra=ra_coords_deg,dec=dec_coords_deg,unit='deg',frame=anchor_frame_name) 
                all_sky_corners_list.append(sky_corners_panel_obj)
            except Exception: continue # Simplifié pour la longueur
        if not all_sky_corners_list: return None, None
        
        try:
            if len(all_sky_corners_list) == 1: all_corners_flat_skycoord = all_sky_corners_list[0]
            else: all_corners_flat_skycoord = skycoord_concatenate(all_sky_corners_list) 
            
            median_ra_deg=np.median(all_corners_flat_skycoord.ra.wrap_at(180*u.deg).deg); median_dec_deg=np.median(all_corners_flat_skycoord.dec.deg)
            tangent_point_sky=SkyCoord(ra=median_ra_deg*u.deg,dec=median_dec_deg*u.deg,frame=all_corners_flat_skycoord.frame.name.lower()) 
            tangent_plane_points_arcsec = SeestarQueuedStacker._project_to_tangent_plane(all_corners_flat_skycoord, tangent_point_sky)
            if tangent_plane_points_arcsec is None or len(tangent_plane_points_arcsec) < 3: return None, None
            
            unique_tangent_points = np.unique(tangent_plane_points_arcsec, axis=0)
            if len(unique_tangent_points) < 3 : hull_points_arcsec = np.ascontiguousarray(unique_tangent_points)
            else:
                try: hull = ConvexHull(np.ascontiguousarray(unique_tangent_points)); hull_points_arcsec = unique_tangent_points[hull.vertices]
                except Exception: hull_points_arcsec = np.ascontiguousarray(unique_tangent_points)
            if len(hull_points_arcsec) < 2 : return None, None

            rect = cv2.minAreaRect(hull_points_arcsec.astype(np.float32)) 
            (center_x_tan_arcsec, center_y_tan_arcsec), (dim1_arcsec, dim2_arcsec), angle_cv_deg = rect
            print(f"DEBUG QM: OMBB brut: centre_tan=({center_x_tan_arcsec:.1f}, {center_y_tan_arcsec:.1f}), dims_tan=({dim1_arcsec:.1f}, {dim2_arcsec:.1f}), angle_cv={angle_cv_deg:.1f}°")

            # --- NOUVELLE LOGIQUE D'ORIENTATION ET DE DIMENSION ---
            # angle_cv_deg est l'angle du côté dim1_arcsec par rapport à l'axe X du plan tangent.
            # On veut que l'image finale soit "droite" (horizontale ou verticale).
            # On teste deux orientations principales pour l'OMBB : son orientation naturelle, et tournée de 90 deg.
            # Puis on choisit celle qui est la plus proche d'être alignée aux axes.

            angle_option1 = angle_cv_deg         # dim1 est la largeur, dim2 est la hauteur
            angle_option2 = angle_cv_deg + 90.0  # dim2 est la largeur, dim1 est la hauteur

            # Normaliser les angles à [-90, 90] pour faciliter la comparaison avec 0 (horizontal)
            # Un angle de 0 ou ~180 devient 0. Un angle de ~90 ou ~-90 devient ~90.
            def normalize_angle_for_straightness(angle):
                angle = angle % 180 # Met dans [0, 180) ou (-180, 0]
                if angle > 90: angle -= 180  # Met dans (-90, 90]
                elif angle < -90: angle += 180 # Met dans [-90, 90)
                return angle

            norm_angle1 = normalize_angle_for_straightness(angle_option1)
            norm_angle2 = normalize_angle_for_straightness(angle_option2)

            # Choisir l'orientation qui est la plus "droite" (plus proche de 0 ou 90, donc |angle| plus petit ou |angle-90| plus petit)
            # On veut minimiser l'angle absolu par rapport à l'axe le plus proche (0 ou 90)
            # Un angle normalisé de 0 est horizontal, un angle de +/-90 est vertical.
            # On préfère celui dont l'angle normalisé est le plus proche de 0.
            # (Si |norm_angle1| est plus petit, dim1 est plus "horizontal")
            # (Si |norm_angle2| est plus petit, dim2 est plus "horizontal")

            final_wcs_rotation_deg = 0.0
            # On veut que le côté le plus long de l'OMBB soit la largeur de l'image SI CETTE ORIENTATION EST PLUS "DROITE"
            # OU que le côté le plus long de l'OMBB soit la hauteur de l'image SI CETTE ORIENTATION EST PLUS "DROITE"

            # Si l'angle de dim1 (angle_cv_deg) est plus proche de 0 (ou 180) que de 90 (ou -90),
            # alors on préfère aligner dim1 horizontalement.
            # Si l'angle de dim1 est plus proche de 90 (ou -90) que de 0,
            # alors on préfère aligner dim1 verticalement (donc dim2 horizontalement).

            # `angle_cv_deg` est dans [-90, 0).
            # Si angle_cv_deg est entre -45 et 0, dim1 est "plutôt horizontal". Rotation WCS = angle_cv_deg.
            # Si angle_cv_deg est entre -90 et -45, dim1 est "plutôt vertical". Rotation WCS = angle_cv_deg + 90 (pour rendre dim2 horizontal).
            
            if abs(angle_cv_deg) <= 45.0: # dim1 est plus horizontal que vertical
                final_wcs_rotation_deg = angle_cv_deg
                # Les dimensions de l'OMBB SONT déjà celles-ci par rapport à cette rotation
                # Mais pour le calcul de la SHAPE finale, on utilisera la reprojection de tous les coins
            else: # dim1 est plus vertical, donc on tourne de 90 deg pour que dim2 devienne "horizontal"
                final_wcs_rotation_deg = angle_cv_deg + 90.0
            
            # Maintenant, on "snappe" cet angle à 0 ou 90 pour que ce soit vraiment droit
            # Mais attention, si on snappe l'angle WCS, les dimensions de l'OMBB ne correspondent plus.
            # L'objectif de l'OMBB était de minimiser l'aire. Si on force l'angle WCS à 0 ou 90,
            # alors on devrait utiliser les dimensions de l'AABB (Axis Aligned Bounding Box) sur le plan tangent.
            
            # REVENONS À L'IDÉE SIMPLE : PAS DE ROTATION PAR RAPPORT AUX AXES RA/DEC
            # L'OMBB sert uniquement à trouver le CRVAL.
            # La SHAPE est ensuite calculée pour englober tout.

            final_wcs_rotation_deg = 0.0 # Forcer l'alignement avec les axes RA/Dec
            self.update_progress(f"   -> Orientation WCS forcée à 0° (alignée RA/Dec).")
            print(f"DEBUG QM: Angle de rotation WCS final forcé à: {final_wcs_rotation_deg:.1f}°")
            
            # CRVAL vient du centre de l'OMBB (calculé avant)
            crval_skycoord_list = SeestarQueuedStacker._deproject_from_tangent_plane(np.array([[center_x_tan_arcsec, center_y_tan_arcsec]]), tangent_point_sky)
            crval_skycoord = crval_skycoord_list[0]
            output_crval = [crval_skycoord.ra.deg, crval_skycoord.dec.deg]
            print(f"DEBUG QM: CRVAL utilisé (centre OMBB): RA={output_crval[0]:.4f}, Dec={output_crval[1]:.4f}")
            # --- FIN NOUVELLE LOGIQUE D'ORIENTATION ---
            
            # ... (Calcul de anchor_pix_scale_deg et output_pixel_scale_deg comme dans V_OMBB_Fix4)
            # ... (Utiliser le code de calcul d'échelle de V_OMBB_Fix4 ici)
            anchor_pix_scale_deg = 0.0 
            try:
                if hasattr(anchor_wcs.wcs, 'cd') and anchor_wcs.wcs.cd is not None and anchor_wcs.wcs.cd.shape == (2,2) and np.all(np.isfinite(anchor_wcs.wcs.cd)):
                    det_cd = np.abs(np.linalg.det(anchor_wcs.wcs.cd));
                    if det_cd > 1e-20: anchor_pix_scale_deg = np.sqrt(det_cd)
                    else: raise ValueError("Det CD trop faible")
                elif hasattr(anchor_wcs.wcs, 'pc') and hasattr(anchor_wcs.wcs, 'cdelt') and anchor_wcs.wcs.pc is not None and anchor_wcs.wcs.cdelt is not None:
                    scale_m = np.diag(anchor_wcs.wcs.cdelt); cd_m_reco = np.dot(scale_m, anchor_wcs.wcs.pc); det_cd_reco = np.abs(np.linalg.det(cd_m_reco))
                    if det_cd_reco > 1e-20: anchor_pix_scale_deg = np.sqrt(det_cd_reco)
                    else: raise ValueError("Det CD reconstruit trop faible")
                else:
                    from astropy.wcs.utils import proj_plane_pixel_scales
                    if anchor_wcs.is_celestial and hasattr(anchor_wcs,'array_shape') and anchor_wcs.array_shape and anchor_wcs.array_shape[0]>0:
                        scales_dpp = proj_plane_pixel_scales(anchor_wcs); anchor_pix_scale_deg = np.mean(np.abs(scales_dpp))
                    else: raise ValueError("Fallback échelle")
            except Exception:
                fov_e = getattr(self,'estimated_fov_degrees',1.); iw_e = anchor_wcs.pixel_shape[0] if hasattr(anchor_wcs,'pixel_shape') and anchor_wcs.pixel_shape and anchor_wcs.pixel_shape[0]>0 else 1000
                anchor_pix_scale_deg = fov_e / (iw_e if iw_e > 0 else 1000)
            if anchor_pix_scale_deg <= 1e-15: return None, None
            output_pixel_scale_deg = anchor_pix_scale_deg / self.drizzle_scale 


            # --- Construction du WCS de sortie avec CRVAL et CD (maintenant avec rotation snappée) ---
            output_wcs = WCS(naxis=2)
            output_wcs.wcs.ctype = [str(getattr(anchor_wcs.wcs, 'ctype', ["RA---TAN", "DEC--TAN"])[0]), 
                                    str(getattr(anchor_wcs.wcs, 'ctype', ["RA---TAN", "DEC--TAN"])[1])]
            output_wcs.wcs.crval = output_crval
            output_wcs.wcs.cunit = [str(getattr(anchor_wcs.wcs, 'cunit', ['deg', 'deg'])[0]),
                                    str(getattr(anchor_wcs.wcs, 'cunit', ['deg', 'deg'])[1])]
            output_wcs.wcs.radesys = str(getattr(anchor_wcs.wcs,'radesys', 'ICRS')).upper()
            
            angle_pc_rad = np.deg2rad(final_wcs_rotation_deg) # UTILISER L'ANGLE SNAPPÉ
            cos_rot = np.cos(angle_pc_rad); sin_rot = np.sin(angle_pc_rad)
            pc_matrix = np.array([[cos_rot, -sin_rot], [sin_rot,  cos_rot]])
            cdelt_matrix = np.array([[-output_pixel_scale_deg, 0.0], [0.0, output_pixel_scale_deg]])
            output_wcs.wcs.cd = np.dot(cdelt_matrix, pc_matrix)
            print(f"DEBUG QM: WCS orienté (snappé) créé. CRVAL={output_wcs.wcs.crval}, CD=\n{output_wcs.wcs.cd}")

            # --- Reprojection des coins, calcul shape et CRPIX final (comme dans V_OMBB_Fix5) ---
            all_ra_deg = all_corners_flat_skycoord.ra.deg
            all_dec_deg = all_corners_flat_skycoord.dec.deg
            projected_x_pixels, projected_y_pixels = output_wcs.all_world2pix(all_ra_deg, all_dec_deg, 0)
            
            valid_projection_mask = np.isfinite(projected_x_pixels) & np.isfinite(projected_y_pixels)
            if not np.any(valid_projection_mask): return None, None # Erreur si aucun point valide
            projected_x_pixels = projected_x_pixels[valid_projection_mask]
            projected_y_pixels = projected_y_pixels[valid_projection_mask]

            x_min_final = np.min(projected_x_pixels); x_max_final = np.max(projected_x_pixels)
            y_min_final = np.min(projected_y_pixels); y_max_final = np.max(projected_y_pixels)

            final_output_width_px = int(np.ceil(x_max_final - x_min_final + 1.0))
            final_output_height_px = int(np.ceil(y_max_final - y_min_final + 1.0))
            final_output_width_px = max(10, final_output_width_px)    
            final_output_height_px = max(10, final_output_height_px)  
            output_shape_final_hw = (final_output_height_px, final_output_width_px)
            
            crval_x_abs_pix, crval_y_abs_pix = output_wcs.all_world2pix(output_wcs.wcs.crval[0], output_wcs.wcs.crval[1], 0)
            final_crpix1 = crval_x_abs_pix - x_min_final + 1.0
            final_crpix2 = crval_y_abs_pix - y_min_final + 1.0
            
            output_wcs.wcs.crpix = [final_crpix1, final_crpix2]
            output_wcs.pixel_shape = (final_output_width_px, final_output_height_px)
            try: output_wcs._naxis1 = final_output_width_px; output_wcs._naxis2 = final_output_height_px
            except AttributeError: pass

            print(f"DEBUG QM: WCS Mosaïque Finale (SnapToAxes) OK: CRPIX={output_wcs.wcs.crpix}, PixelShape={output_wcs.pixel_shape}")
            return output_wcs, output_shape_final_hw

        except Exception as e_grid:
            print(f"ERREUR QM [_calculate_local_mosaic_output_grid V_OMBB_SnapToAxes]: Échec calcul final grille/WCS: {e_grid}")
            traceback.print_exc(limit=2)
            return None, None

    # ... (reste de la classe) ...






##############################################################################################################################


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




# --- DANS LA CLASSE SeestarQueuedStacker DANS seestar/queuep/queue_manager.py ---

    def _cleanup_mosaic_panel_stacks_temp(self):
        """
        Supprime le dossier contenant les stacks de panneaux temporaires
        (utilisé par l'ancienne logique de mosaïque ou si des fichiers y sont créés).
        """
        # --- VÉRIFICATION AJOUTÉE ---
        if self.output_folder is None: 
            print("WARN QM [_cleanup_mosaic_panel_stacks_temp]: self.output_folder non défini, nettoyage annulé.")
            return
        # --- FIN VÉRIFICATION ---

        panel_stacks_dir = os.path.join(self.output_folder, "mosaic_panel_stacks_temp")
        
        # Vérifier si le dossier existe avant d'essayer de le supprimer
        if os.path.isdir(panel_stacks_dir): # Utiliser os.path.isdir pour vérifier
            try:
                shutil.rmtree(panel_stacks_dir)
                self.update_progress(f"🧹 Dossier stacks panneaux temp. supprimé: {os.path.basename(panel_stacks_dir)}")
                print(f"DEBUG QM [_cleanup_mosaic_panel_stacks_temp]: Dossier {panel_stacks_dir} supprimé.")
            except FileNotFoundError:
                # Devrait être attrapé par isdir, mais sécurité
                print(f"DEBUG QM [_cleanup_mosaic_panel_stacks_temp]: Dossier {panel_stacks_dir} non trouvé (déjà supprimé ou jamais créé).")
                pass # Le dossier n'existe pas, rien à faire
            except OSError as e: # Capturer les erreurs d'OS (permissions, etc.)
                self.update_progress(f"⚠️ Erreur suppression dossier stacks panneaux temp. ({os.path.basename(panel_stacks_dir)}): {e}")
                print(f"ERREUR QM [_cleanup_mosaic_panel_stacks_temp]: Erreur OSError lors de la suppression de {panel_stacks_dir}: {e}")
            except Exception as e_generic: # Capturer toute autre exception
                self.update_progress(f"⚠️ Erreur inattendue suppression dossier stacks panneaux temp.: {e_generic}")
                print(f"ERREUR QM [_cleanup_mosaic_panel_stacks_temp]: Erreur Exception lors de la suppression de {panel_stacks_dir}: {e_generic}")
        else:
            # Log optionnel si le dossier n'existait pas
            # print(f"DEBUG QM [_cleanup_mosaic_panel_stacks_temp]: Dossier {panel_stacks_dir} non trouvé, aucun nettoyage nécessaire.")
            pass





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
        # AJOUT D'UNE VÉRIFICATION : Ne rien faire si self.output_folder n'est pas encore défini.
        if self.output_folder is None:
            print("WARN QM [_cleanup_drizzle_batch_outputs]: self.output_folder non défini, nettoyage annulé.")
            return

        batch_output_dir = os.path.join(self.output_folder, "drizzle_batch_outputs")
        if batch_output_dir and os.path.isdir(batch_output_dir): # Vérifier aussi si le chemin construit est valide
            try:
                shutil.rmtree(batch_output_dir)
                self.update_progress(f"🧹 Dossier Drizzle intermédiaires par lot supprimé: {os.path.basename(batch_output_dir)}")
            except Exception as e:
                self.update_progress(f"⚠️ Erreur suppression dossier Drizzle intermédiaires ({os.path.basename(batch_output_dir)}): {e}")
        # else: # Log optionnel si le dossier n'existait pas ou chemin invalide
            # if self.output_folder: # Pour éviter de logguer si c'est juste output_folder qui est None
            #    print(f"DEBUG QM [_cleanup_drizzle_batch_outputs]: Dossier {batch_output_dir} non trouvé ou invalide pour nettoyage.")



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
        alignement, calcul qualité, et retourne WCS généré et un MASQUE DE PIXELS VALIDES.

        Args:
            file_path (str): Chemin complet du fichier FITS à traiter.
            reference_image_data (np.ndarray): Données de l'image de référence.

        Returns:
            tuple: (aligned_data, header, quality_scores, generated_wcs_object, valid_pixel_mask_2d)
                   aligned_data: HWC float32, 0-1
                   valid_pixel_mask_2d: HW bool, True où aligned_data a des pixels valides (non remplissage)
                   Retourne (None, None, scores, None, None) en cas d'échec.
        """
        file_name = os.path.basename(file_path)
        quality_scores = {'snr': 0.0, 'stars': 0.0} # Initialisation par défaut
        print(f"DEBUG QM [_process_file]: Début traitement '{file_name}'")
        header = None
        prepared_img = None
        wcs_generated = None
        aligned_img = None # Initialiser pour le bloc finally
        valid_pixel_mask_2d = None # Initialiser

        try:
            # 1. Charger et valider
            print(f"  -> [1/7] Chargement/Validation FITS pour '{file_name}'...")
            img_data = load_and_validate_fits(file_path)
            if img_data is None: raise ValueError("Échec chargement/validation FITS.")
            header = fits.getheader(file_path)
            print(f"     - Chargement OK. Shape initiale: {img_data.shape}, Dtype: {img_data.dtype}")

            # 2. Vérification variance
            print(f"  -> [2/7] Vérification variance pour '{file_name}'...")
            std_dev = np.std(img_data)
            variance_threshold = 0.0015 # Seuil (peut nécessiter ajustement)
            if std_dev < variance_threshold:
                raise ValueError(f"Faible variance: {std_dev:.4f} (seuil: {variance_threshold}). Image probablement vide/noire.")
            print(f"     - Variance OK (std: {std_dev:.4f}).")

            # 3. Pré-traitement (Debayer, WB Auto si applicable, Correction HP)
            print(f"  -> [3/7] Pré-traitement (Debayer, WB, HP) pour '{file_name}'...")
            prepared_img = img_data.astype(np.float32) # Travailler sur float32
            is_color_after_preprocessing = False # Flag pour savoir si on a une image couleur

            # Debayering
            if prepared_img.ndim == 2:
                bayer_pattern_from_header = header.get('BAYERPAT', self.bayer_pattern)
                pattern_upper = bayer_pattern_from_header.upper() if isinstance(bayer_pattern_from_header, str) else self.bayer_pattern.upper()
                
                if pattern_upper in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                    print(f"     - Debayering (Pattern: {pattern_upper})...")
                    try:
                        prepared_img = debayer_image(prepared_img, pattern_upper)
                        is_color_after_preprocessing = True
                        print(f"       - Debayering OK. Nouvelle shape: {prepared_img.shape}")
                    except ValueError as de:
                        self.update_progress(f"   ⚠️ Erreur debayering {file_name}: {de}. Traitement en N&B.")
                        print(f"       - Échec Debayering: {de}. Image reste N&B.")
                else:
                    print(f"     - Image N&B ou pattern Bayer ('{bayer_pattern_from_header}') non reconnu. Pas de debayering.")
            elif prepared_img.ndim == 3 and prepared_img.shape[2] == 3:
                is_color_after_preprocessing = True
                print(f"     - Image déjà couleur (Shape: {prepared_img.shape}). Pas de debayering.")
            else:
                raise ValueError(f"Shape d'image inattendue après chargement: {prepared_img.shape}. Impossible de pré-traiter.")

            # Balance des Blancs Automatique (seulement si couleur et si activé globalement, bien que non configurable ici)
            # Cette WB est basique et vise à aider l'alignement/qualité. La WB finale est sur l'aperçu.
            if is_color_after_preprocessing:
                print(f"     - Tentative de WB auto basique pour pré-traitement...")
                try:
                    # Calcul simple des facteurs basé sur les médianes pour aider l'alignement
                    r_ch, g_ch, b_ch = prepared_img[...,0], prepared_img[...,1], prepared_img[...,2]
                    med_r, med_g, med_b = np.median(r_ch), np.median(g_ch), np.median(b_ch)
                    if med_g > 1e-6: # Éviter division par zéro
                        gain_r = np.clip(med_g / max(med_r, 1e-6), 0.5, 2.0)
                        gain_b = np.clip(med_g / max(med_b, 1e-6), 0.5, 2.0)
                        prepared_img[...,0] *= gain_r
                        prepared_img[...,2] *= gain_b
                        prepared_img = np.clip(prepared_img, 0.0, 1.0)
                        print(f"       - WB auto basique appliquée (Gains R:{gain_r:.2f}, B:{gain_b:.2f}).")
                except Exception as wb_err:
                    print(f"       - ERREUR WB auto basique: {wb_err}. Image non modifiée par WB.")

            # Correction des Pixels Chauds
            if self.correct_hot_pixels:
                print(f"     - Correction des pixels chauds (Seuil: {self.hot_pixel_threshold}, Voisinage: {self.neighborhood_size})...")
                try:
                    prepared_img = detect_and_correct_hot_pixels(prepared_img, self.hot_pixel_threshold, self.neighborhood_size)
                    print(f"       - Correction HP OK.")
                except Exception as hp_err:
                    self.update_progress(f"   ⚠️ Erreur correction HP pour {file_name}: {hp_err}.")
                    print(f"       - ERREUR Correction HP: {hp_err}.")
            
            prepared_img = prepared_img.astype(np.float32) # Assurer float32 après toutes les manips
            print(f"     - Pré-traitement terminé. Shape finale: {prepared_img.shape}")

            # 4. Génération WCS (depuis header original)
            print(f"  -> [4/7] Génération WCS pour '{file_name}'...")
            wcs_generated = None
            if header:
                try:
                    # Essayer WCS(header) directement (plus robuste si standard)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', FITSFixedWarning) # Ignorer warnings FITS non standard
                        wcs_from_hdr = WCS(header, naxis=2) # Toujours 2 axes pour le plan image
                    if wcs_from_hdr and wcs_from_hdr.is_celestial:
                        wcs_generated = wcs_from_hdr
                        print(f"     - WCS obtenu directement depuis le header.")
                except Exception: # Si WCS(header) échoue, essayer notre fonction custom
                    pass # On essaiera _create_wcs_from_header ensuite

                if wcs_generated is None: # Si WCS(header) a échoué ou n'était pas céleste
                    print(f"     - Tentative de génération WCS custom depuis header...")
                    # --- Import tardif ---
                    try: from ..enhancement.drizzle_integration import _create_wcs_from_header
                    except ImportError: _create_wcs_from_header = None
                    if _create_wcs_from_header:
                        wcs_generated = _create_wcs_from_header(header)
                        if wcs_generated and wcs_generated.is_celestial:
                            print(f"       - WCS custom généré avec succès.")
                        else: print(f"       - Échec génération WCS custom ou WCS non céleste.")
                    else: print(f"       - ERREUR: _create_wcs_from_header non importable.")
                
                # Attacher pixel_shape au WCS si possible
                if wcs_generated and wcs_generated.is_celestial:
                    naxis1_h = header.get('NAXIS1', header.get('IMAGEW', None)) # Essayer aussi IMAGEW
                    naxis2_h = header.get('NAXIS2', header.get('IMAGEH', None)) # Essayer aussi IMAGEH
                    if naxis1_h and naxis2_h:
                        try:
                            wcs_generated.pixel_shape = (int(naxis1_h), int(naxis2_h)) # (W, H) pour astropy
                            print(f"       - pixel_shape ({wcs_generated.pixel_shape}) attaché au WCS généré.")
                        except ValueError: print(f"       - WARNING: NAXIS1/2 non entiers ('{naxis1_h}','{naxis2_h}') pour pixel_shape.")
                    elif wcs_generated.pixel_shape is None: # Si toujours None
                        print(f"       - WARNING: Impossible de déterminer pixel_shape pour WCS généré de {file_name}.")
                else: # Échec total WCS
                    print(f"     - ERREUR: Aucun WCS valide (header ou généré) pour {file_name}.")
                    if self.is_mosaic_run or self.drizzle_active_session: # WCS est critique pour ces modes
                        raise ValueError("WCS requis pour Drizzle/Mosaïque mais non obtenu.")
            else: # Pas de header
                print(f"     - WARNING: Header original manquant pour {file_name}. Impossible de générer WCS.")
                if self.is_mosaic_run or self.drizzle_active_session:
                    raise ValueError("Header manquant, WCS requis pour Drizzle/Mosaïque.")

            # 5. Alignement Astroalign
            print(f"  -> [5/7] Alignement Astroalign pour '{file_name}'...")
            if reference_image_data is None: raise RuntimeError("Image de référence non disponible pour alignement.")
            aligned_img, align_success = self.aligner._align_image(prepared_img, reference_image_data, file_name)
            if not align_success or aligned_img is None:
                raise RuntimeError(f"Échec Alignement Astroalign pour {file_name}.")
            print(f"     - Alignement Astroalign OK. Shape alignée: {aligned_img.shape}")

            # --- NOUVEAU : Création du valid_pixel_mask ---
            # astroalign remplit les zones hors de l'image source avec 0.0 par défaut.
            # Un masque est True où les données sont valides (non-remplissage).
            # Si l'image est couleur (H,W,C), on peut baser le masque sur la luminance ou un canal (ex: Vert).
            # Si N&B (H,W), on l'utilise directement.
            print(f"  -> [6/7] Création du masque de pixels valides pour '{file_name}'...")
            if aligned_img.ndim == 3 and aligned_img.shape[2] == 3:
                # Pour une image couleur, on peut prendre la somme des canaux, ou la luminance.
                # Si un pixel est (0,0,0) après alignement, il vient probablement du remplissage.
                # Un seuil très bas pour éviter les vrais pixels noirs de l'objet.
                luminance_aligned = 0.299 * aligned_img[..., 0] + 0.587 * aligned_img[..., 1] + 0.114 * aligned_img[..., 2]
                valid_pixel_mask_2d = (luminance_aligned > 1e-5).astype(bool) # Seuil très bas
            elif aligned_img.ndim == 2:
                valid_pixel_mask_2d = (aligned_img > 1e-5).astype(bool)
            else:
                print(f"     - ERREUR: Shape d'image alignée inattendue ({aligned_img.shape}) pour création masque. Masque mis à None.")
                valid_pixel_mask_2d = None # Ne devrait pas arriver
            
            if valid_pixel_mask_2d is not None:
                print(f"     - Masque de pixels valides (2D HxW) créé. Shape: {valid_pixel_mask_2d.shape}, True Pixels: {np.sum(valid_pixel_mask_2d)}")
            # --- FIN NOUVEAU ---



            # 7. Calcul des scores de qualité (sur image alignée)
            print(f"  -> [7/7] Calcul des scores qualité pour '{file_name}'...")
            if self.use_quality_weighting:
                quality_scores = self._calculate_quality_metrics(aligned_img) # Log interne
                print(f"     - Scores Qualité: SNR={quality_scores.get('snr',0):.2f}, Stars={quality_scores.get('stars',0):.3f}")

                # ========================= NOUVELLE SECTION AJOUTÉE =========================
                # Définir un seuil pour le score "stars". 
                # Un score de 0.05 correspondrait à 10 étoiles si max_stars_for_score = 200.
                # Ajustez cette valeur si nécessaire.
                min_star_score_threshold = 0.025 # Exemple: au moins 5 étoiles si max_stars_for_score=200
                                                 # (0.025 * 200 = 5)

                current_star_score = quality_scores.get('stars', 0.0)
                if current_star_score < min_star_score_threshold:
                    # Si le score d'étoiles est trop bas, lever une ValueError
                    # pour que l'image soit traitée par le bloc except plus bas
                    # (et donc potentiellement déplacée vers unaligned_files).
                    error_message = (f"Score d'étoiles ({current_star_score:.3f}) trop bas "
                                     f"(seuil: {min_star_score_threshold:.3f}). Image considérée comme inalignable/vide.")
                    print(f"     - REJET (Qualité): {error_message}") # Log spécifique
                    raise ValueError(error_message)
                # ======================= FIN DE LA NOUVELLE SECTION =======================
            else:
                print(f"     - Pondération qualité désactivée, scores non calculés (par défaut).")

            print(f"DEBUG QM [_process_file]: Traitement de '{file_name}' terminé avec succès.")
            # Retourner l'image alignée, header original, scores, WCS généré, et le nouveau masque
            return aligned_img, header, quality_scores, wcs_generated, valid_pixel_mask_2d

        except (ValueError, RuntimeError) as proc_err: # Erreurs "normales" ou attendues du flux
            self.update_progress(f"   ⚠️ Fichier '{file_name}' ignoré: {proc_err}")
            self.skipped_files_count += 1
            # Essayer de déplacer vers un dossier "skipped" si le fichier existe toujours
            if file_path and os.path.exists(file_path) and self.unaligned_folder: # unaligned_folder est le dossier skipped
                try:
                    skipped_path = os.path.join(self.unaligned_folder, f"skipped_processing_{file_name}")
                    shutil.move(file_path, skipped_path)
                    print(f"     - Fichier '{file_name}' déplacé vers skipped: {os.path.basename(skipped_path)}")
                except Exception as move_err:
                    print(f"     - ERREUR déplacement fichier skipped '{file_name}': {move_err}")
            return None, header, quality_scores, None, None # Header peut être utile pour logs, scores par défaut

        except Exception as e: # Erreurs inattendues critiques
            self.update_progress(f"❌ Erreur critique traitement fichier {file_name}: {e}")
            print(f"ERREUR QM [_process_file]: Exception inattendue pour '{file_name}':")
            traceback.print_exc(limit=3)
            self.skipped_files_count += 1 # Compter comme skipped/error
            # Essayer de déplacer vers un dossier "error"
            if file_path and os.path.exists(file_path) and self.unaligned_folder:
                try:
                    error_path = os.path.join(self.unaligned_folder, f"error_processing_{file_name}")
                    shutil.move(file_path, error_path)
                    print(f"     - Fichier '{file_name}' déplacé vers error: {os.path.basename(error_path)}")
                except Exception as move_err:
                    print(f"     - ERREUR déplacement fichier error '{file_name}': {move_err}")
            return None, header, quality_scores, None, None

        finally:
            # Nettoyage mémoire pour cette image
            del img_data, prepared_img, aligned_img # valid_pixel_mask_2d est petit
            # wcs_generated et header sont retournés ou None
            # quality_scores est retourné
            gc.collect()





#############################################################################################################################





    def _process_completed_batch(self, batch_items_to_stack, current_batch_num, total_batches_est):
        """
        [MODE CLASSIQUE - SUM/W] Traite un lot d'images complété pour l'empilement classique.
        Cette méthode est appelée par _worker lorsque current_batch_items_with_masks_for_stack_batch
        atteint la taille self.batch_size (ou pour le dernier lot partiel).

        Elle appelle _stack_batch pour obtenir l'image moyenne du lot et sa carte de couverture,
        puis appelle _combine_batch_result pour accumuler ces résultats dans les memmaps globaux.

        Args:
            batch_items_to_stack (list): Liste des items du lot à traiter.
                                         Chaque item est un tuple:
                                         (aligned_data_HWC_or_HW, header_orig, scores_dict,
                                          wcs_generated_obj, valid_pixel_mask_2d_HW_bool).
            current_batch_num (int): Le numéro séquentiel de ce lot.
            total_batches_est (int): Le nombre total de lots estimé pour la session.
        """
        # Log d'entrée de la méthode avec les informations sur le lot
        num_items_in_this_batch = len(batch_items_to_stack) if batch_items_to_stack else 0
        print(f"DEBUG QM [_process_completed_batch]: Début pour lot CLASSIQUE #{current_batch_num} "
              f"avec {num_items_in_this_batch} items.")

        # Vérification si le lot est vide (ne devrait pas arriver si _worker gère bien)
        if not batch_items_to_stack: # batch_items_to_stack est maintenant un paramètre défini
            self.update_progress(f"⚠️ Tentative de traiter un lot vide (Lot #{current_batch_num}) "
                                 "dans _process_completed_batch. Ignoré.", None)
            print("DEBUG QM [_process_completed_batch]: Sortie précoce (lot vide reçu).")
            return

        # Informations pour les messages de progression
        batch_size_actual_for_log = len(batch_items_to_stack)
        progress_info_log = (f"(Lot {current_batch_num}/"
                             f"{total_batches_est if total_batches_est > 0 else '?'})")

        self.update_progress(f"⚙️ Traitement classique du batch {progress_info_log} "
                             f"({batch_size_actual_for_log} images)...")

        # --- Appel à _stack_batch ---
        # _stack_batch attend :
        #   (self, batch_items_with_masks, current_batch_num=0, total_batches_est=0)
        # Il retourne :
        #   (stacked_image_np, stack_info_header, batch_coverage_map_2d)

        print(f"DEBUG QM [_process_completed_batch]: Appel à _stack_batch pour lot #{current_batch_num}...")
        stacked_batch_data_np, stack_info_header, batch_coverage_map_2d = self._stack_batch(
            batch_items_to_stack, # La liste complète des items pour ce lot
            current_batch_num,
            total_batches_est
        )

        # Vérifier le résultat de _stack_batch
        if stacked_batch_data_np is not None and batch_coverage_map_2d is not None:
            print(f"DEBUG QM [_process_completed_batch]: _stack_batch pour lot #{current_batch_num} réussi. "
                  f"Shape image lot: {stacked_batch_data_np.shape}, "
                  f"Shape carte couverture lot: {batch_coverage_map_2d.shape}")
            
            # --- Combiner le résultat du batch dans les accumulateurs SUM/WHT globaux ---
            # _combine_batch_result attend :
            #   (self, stacked_batch_data_np, stack_info_header, batch_coverage_map_2d)
            print(f"DEBUG QM [_process_completed_batch]: Appel à _combine_batch_result pour lot #{current_batch_num}...")
            self._combine_batch_result(
                stacked_batch_data_np,
                stack_info_header,
                batch_coverage_map_2d # La carte de couverture 2D du lot
            )
            
            # Mise à jour de l'aperçu SUM/W après accumulation de ce lot
            # (Seulement si on n'est pas en mode Drizzle, car Drizzle Incrémental a son propre update)
            # Cette condition est redondante ici car _process_completed_batch n'est appelée
            # que si not self.drizzle_active_session.
            if not self.drizzle_active_session:
                print("DEBUG QM [_process_completed_batch]: Appel à _update_preview_sum_w après accumulation lot classique...")
                self._update_preview_sum_w() # Met à jour l'aperçu avec les données SUM/W actuelles
            
        else: # _stack_batch a échoué ou n'a rien retourné de valide
            # Le nombre d'images du lot qui a échoué à l'étape _stack_batch
            num_failed_in_stack_batch = len(batch_items_to_stack)
            self.failed_stack_count += num_failed_in_stack_batch
            self.update_progress(f"❌ Échec combinaison (dans _stack_batch) du lot {progress_info_log}. "
                                 f"{num_failed_in_stack_batch} images ignorées pour accumulation.", None)
            print(f"ERREUR QM [_process_completed_batch]: _stack_batch a échoué pour lot #{current_batch_num}.")

        # Le nettoyage de current_batch_items_with_masks_for_stack_batch se fait dans _worker
        # après l'appel à cette fonction.
        gc.collect() # Forcer un garbage collect après avoir traité un lot
        print(f"DEBUG QM [_process_completed_batch]: Fin pour lot CLASSIQUE #{current_batch_num}.")







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





    def _combine_batch_result(self, stacked_batch_data_np, stack_info_header, batch_coverage_map_2d):
        """
        [MODE SUM/W - CLASSIQUE] Accumule le résultat d'un batch classique
        (image moyenne du lot et sa carte de couverture/poids 2D)
        dans les accumulateurs memmap globaux SUM et WHT.

        Args:
            stacked_batch_data_np (np.ndarray): Image MOYENNE du lot (HWC ou HW, float32, 0-1).
            stack_info_header (fits.Header): En-tête info du lot (contient NIMAGES physiques).
            batch_coverage_map_2d (np.ndarray): Carte de poids/couverture 2D (HW, float32)
                                                pour ce lot spécifique.
        """
        print(f"DEBUG QM [_combine_batch_result SUM/W]: Début accumulation lot classique avec carte de couverture 2D.")
        if batch_coverage_map_2d is not None:
            print(f"  -> Reçu de _stack_batch -> batch_coverage_map_2d - Shape: {batch_coverage_map_2d.shape}, "
                  f"Range: [{np.min(batch_coverage_map_2d):.2f}-{np.max(batch_coverage_map_2d):.2f}], "
                  f"Sum: {np.sum(batch_coverage_map_2d):.2f}")
        else:
            print(f"  -> Reçu de _stack_batch -> batch_coverage_map_2d est None.")


        # --- Vérifications initiales ---
        if stacked_batch_data_np is None or stack_info_header is None or batch_coverage_map_2d is None:
            self.update_progress("⚠️ Erreur interne: Données batch/couverture invalides pour accumulation SUM/W.")
            print("DEBUG QM [_combine_batch_result SUM/W]: Sortie précoce (données batch/couverture invalides).")
            return

        if self.cumulative_sum_memmap is None or self.cumulative_wht_memmap is None or self.memmap_shape is None:
             self.update_progress("❌ Erreur critique: Accumulateurs Memmap SUM/WHT non initialisés.")
             print("ERREUR QM [_combine_batch_result SUM/W]: Memmap non initialisé.")
             self.processing_error = "Memmap non initialisé"; self.stop_processing = True
             return

        # Vérifier la cohérence des shapes
        # stacked_batch_data_np peut être HWC ou HW. memmap_shape est HWC.
        # batch_coverage_map_2d doit être HW.
        expected_shape_hw = self.memmap_shape[:2]
        
        if batch_coverage_map_2d.shape != expected_shape_hw:
            self.update_progress(f"❌ Incompatibilité shape carte couverture lot: Attendu {expected_shape_hw}, Reçu {batch_coverage_map_2d.shape}. Accumulation échouée.")
            print(f"ERREUR QM [_combine_batch_result SUM/W]: Incompatibilité shape carte couverture lot.")
            try: batch_n_error = int(stack_info_header.get('NIMAGES', 1)); self.failed_stack_count += batch_n_error
            except: self.failed_stack_count += 1 # Au moins une image
            return

        # S'assurer que stacked_batch_data_np a la bonne dimension pour la multiplication (HWC ou HW)
        is_color_batch_data = (stacked_batch_data_np.ndim == 3 and stacked_batch_data_np.shape[2] == 3)
        if is_color_batch_data and stacked_batch_data_np.shape != self.memmap_shape:
            self.update_progress(f"❌ Incompatibilité shape image lot (couleur): Attendu {self.memmap_shape}, Reçu {stacked_batch_data_np.shape}. Accumulation échouée.")
            print(f"ERREUR QM [_combine_batch_result SUM/W]: Incompatibilité shape image lot (couleur).")
            try: batch_n_error = int(stack_info_header.get('NIMAGES', 1)); self.failed_stack_count += batch_n_error
            except: self.failed_stack_count += 1
            return
        elif not is_color_batch_data and stacked_batch_data_np.ndim == 2 and stacked_batch_data_np.shape != expected_shape_hw:
            self.update_progress(f"❌ Incompatibilité shape image lot (N&B): Attendu {expected_shape_hw}, Reçu {stacked_batch_data_np.shape}. Accumulation échouée.")
            print(f"ERREUR QM [_combine_batch_result SUM/W]: Incompatibilité shape image lot (N&B).")
            try: batch_n_error = int(stack_info_header.get('NIMAGES', 1)); self.failed_stack_count += batch_n_error
            except: self.failed_stack_count += 1
            return
        elif not is_color_batch_data and stacked_batch_data_np.ndim != 2 : # Cas N&B mais pas 2D
             self.update_progress(f"❌ Shape image lot N&B inattendue: {stacked_batch_data_np.shape}. Accumulation échouée.")
             print(f"ERREUR QM [_combine_batch_result SUM/W]: Shape image lot N&B inattendue.")
             try: batch_n_error = int(stack_info_header.get('NIMAGES', 1)); self.failed_stack_count += batch_n_error
             except: self.failed_stack_count += 1
             return


        try:
            num_physical_images_in_batch = int(stack_info_header.get('NIMAGES', 1))
            batch_exposure = float(stack_info_header.get('TOTEXP', 0.0))

            # Vérifier si la carte de couverture a des poids significatifs
            if np.sum(batch_coverage_map_2d) < 1e-6 and num_physical_images_in_batch > 0:
                self.update_progress(f"⚠️ Lot avec {num_physical_images_in_batch} images mais somme de couverture quasi nulle. Lot ignoré pour accumulation.")
                print(f"DEBUG QM [_combine_batch_result SUM/W]: Sortie précoce (somme couverture quasi nulle).")
                self.failed_stack_count += num_physical_images_in_batch # Compter ces images comme échec d'empilement
                return

            # Préparer les données pour l'accumulation (types et shapes)
            # stacked_batch_data_np est déjà float32, 0-1
            # batch_coverage_map_2d est déjà float32
            
            # Calculer le signal total à ajouter à SUM: ImageMoyenneDuLot * SaCarteDeCouverturePondérée
            # Si stacked_batch_data_np est HWC et batch_coverage_map_2d est HW, il faut broadcaster.
            signal_to_add_to_sum_float64 = None # Utiliser float64 pour la multiplication et l'accumulation
            if is_color_batch_data: # Image couleur HWC
                signal_to_add_to_sum_float64 = stacked_batch_data_np.astype(np.float64) * batch_coverage_map_2d.astype(np.float64)[..., np.newaxis]
            else: # Image N&B HW
                # Si SUM memmap est HWC (ce qui est le cas avec memmap_shape), il faut adapter
                if self.memmap_shape[2] == 3: # Si l'accumulateur global est couleur
                    # On met l'image N&B dans les 3 canaux de l'accumulateur
                    temp_hwc = np.stack([stacked_batch_data_np]*3, axis=-1)
                    signal_to_add_to_sum_float64 = temp_hwc.astype(np.float64) * batch_coverage_map_2d.astype(np.float64)[..., np.newaxis]
                else: # Si l'accumulateur global est N&B (ne devrait pas arriver avec memmap_shape HWC)
                    signal_to_add_to_sum_float64 = stacked_batch_data_np.astype(np.float64) * batch_coverage_map_2d.astype(np.float64)

            print(f"DEBUG QM [_combine_batch_result SUM/W]: Accumulation pour {num_physical_images_in_batch} images physiques.")
            print(f"  -> signal_to_add_to_sum_float64 - Shape: {signal_to_add_to_sum_float64.shape}, "
                  f"Range: [{np.min(signal_to_add_to_sum_float64):.2f} - {np.max(signal_to_add_to_sum_float64):.2f}]")

            # --- Accumulation dans les memmaps ---
            self.cumulative_sum_memmap[:] += signal_to_add_to_sum_float64.astype(self.memmap_dtype_sum)
            if hasattr(self.cumulative_sum_memmap, 'flush'): self.cumulative_sum_memmap.flush()
            print("DEBUG QM [_combine_batch_result SUM/W]: Addition SUM terminée et flushée.")

            # batch_coverage_map_2d est déjà HW et float32 (dtype de self.memmap_dtype_wht)
            self.cumulative_wht_memmap[:] += batch_coverage_map_2d # Pas besoin de astype si déjà float32
            if hasattr(self.cumulative_wht_memmap, 'flush'): self.cumulative_wht_memmap.flush()
            print("DEBUG QM [_combine_batch_result SUM/W]: Addition WHT terminée et flushée.")

            # Mise à jour des compteurs globaux
            self.images_in_cumulative_stack += num_physical_images_in_batch # Compte les images physiques
            self.total_exposure_seconds += batch_exposure
            print(f"DEBUG QM [_combine_batch_result SUM/W]: Compteurs mis à jour: images_in_cumulative_stack={self.images_in_cumulative_stack}, total_exposure_seconds={self.total_exposure_seconds:.1f}")

            # --- Mise à jour Header Cumulatif (comme avant) ---
            if self.current_stack_header is None:
                self.current_stack_header = fits.Header()
                first_header_from_batch = stack_info_header
                keys_to_copy = ['INSTRUME', 'TELESCOP', 'OBJECT', 'FILTER', 'DATE-OBS', 'GAIN', 'OFFSET', 'CCD-TEMP', 'RA', 'DEC', 'SITELAT', 'SITELONG', 'FOCALLEN', 'BAYERPAT']
                for key_iter in keys_to_copy:
                    if first_header_from_batch and key_iter in first_header_from_batch:
                        try: self.current_stack_header[key_iter] = (first_header_from_batch[key_iter], first_header_from_batch.comments[key_iter] if key_iter in first_header_from_batch.comments else '')
                        except Exception: self.current_stack_header[key_iter] = first_header_from_batch[key_iter]
                self.current_stack_header['STACKTYP'] = (f'Classic SUM/W ({self.stacking_mode})', 'Stacking method')
                self.current_stack_header['CREATOR'] = ('SeestarStacker (SUM/W)', 'Processing Software')
                if self.correct_hot_pixels: self.current_stack_header['HISTORY'] = 'Hot pixel correction applied'
                if self.use_quality_weighting: self.current_stack_header['HISTORY'] = 'Quality weighting (SNR/Stars) with per-pixel coverage for SUM/W'
                else: self.current_stack_header['HISTORY'] = 'Uniform weighting (by image count) with per-pixel coverage for SUM/W'
                self.current_stack_header['HISTORY'] = 'SUM/W Accumulation Initialized'

            self.current_stack_header['NIMAGES'] = (self.images_in_cumulative_stack, 'Physical images processed for stack')
            self.current_stack_header['TOTEXP'] = (round(self.total_exposure_seconds, 2), '[s] Approx total exposure time')
            
            # Mettre à jour SUMWGHTS avec la somme des poids max de WHT (approximation de l'exposition pondérée)
            # self.cumulative_wht_memmap est HW, float32
            current_total_wht_center = np.max(self.cumulative_wht_memmap) if self.cumulative_wht_memmap.size > 0 else 0.0
            self.current_stack_header['SUMWGHTS'] = (float(current_total_wht_center), 'Approx. max sum of weights in WHT map')

            print("DEBUG QM [_combine_batch_result SUM/W]: Accumulation batch classique terminée.")

        except MemoryError as mem_err:
             print(f"ERREUR QM [_combine_batch_result SUM/W]: ERREUR MÉMOIRE - {mem_err}")
             self.update_progress(f"❌ ERREUR MÉMOIRE lors de l'accumulation du batch classique.")
             traceback.print_exc(limit=1); self.processing_error = "Erreur Mémoire Accumulation"; self.stop_processing = True
        except Exception as e:
            print(f"ERREUR QM [_combine_batch_result SUM/W]: Exception inattendue - {e}")
            self.update_progress(f"❌ Erreur pendant l'accumulation du résultat du batch: {e}")
            traceback.print_exc(limit=3)
            try: batch_n_error_acc = int(stack_info_header.get('NIMAGES', 1)) # Nombre d'images du lot qui a échoué
            except: batch_n_error_acc = 1
            self.failed_stack_count += batch_n_error_acc





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






    def _stack_batch(self, batch_items_with_masks, current_batch_num=0, total_batches_est=0):
        """
        Combine un lot d'images alignées en utilisant ccdproc.combine.
        Calcule et applique les poids qualité scalaires si activé.
        NOUVEAU: Calcule et retourne une carte de couverture/poids 2D pour le lot.

        Args:
            batch_items_with_masks (list): Liste de tuples:
                [(aligned_data, header, scores, wcs_obj, valid_pixel_mask_2d), ...].
                - aligned_data: HWC ou HW, float32, 0-1.
                - valid_pixel_mask_2d: HW bool, True où aligned_data a des pixels valides.
            current_batch_num (int): Numéro du lot pour les logs.
            total_batches_est (int): Estimation totale des lots pour les logs.

        Returns:
            tuple: (stacked_image_np, stack_info_header, batch_coverage_map_2d)
                   ou (None, None, None) en cas d'échec.
                   batch_coverage_map_2d: Carte HxW float32 des poids/couverture pour ce lot.
        """
        if not batch_items_with_masks:
            self.update_progress(f"❌ Erreur interne: _stack_batch reçu un lot vide (batch_items_with_masks).")
            return None, None, None

        num_physical_images_in_batch_initial = len(batch_items_with_masks)
        progress_info = f"(Lot {current_batch_num}/{total_batches_est if total_batches_est > 0 else '?'})"
        self.update_progress(f"✨ Combinaison ccdproc du batch {progress_info} ({num_physical_images_in_batch_initial} images physiques initiales)...")
        print(f"DEBUG QM [_stack_batch]: Début pour lot #{current_batch_num} avec {num_physical_images_in_batch_initial} items.")

        # --- 1. Filtrer les items valides et extraire les composants ---
        # Un item est valide si image, header, scores, et valid_pixel_mask sont non None
        # et si la shape de l'image est cohérente.
        
        valid_images_for_ccdproc = [] # Liste des arrays image (HWC ou HW)
        valid_headers_for_ccdproc = []
        valid_scores_for_quality_weights = []
        valid_pixel_masks_for_coverage = [] # Liste des masques 2D (HW bool)

        ref_shape_check = None # Shape de la première image valide (HWC ou HW)
        is_color_batch = False # Sera déterminé par la première image valide

        for idx, item_tuple in enumerate(batch_items_with_masks):
            if len(item_tuple) != 5: # S'assurer qu'on a bien les 5 éléments
                self.update_progress(f"   -> Item {idx+1} du lot {current_batch_num} ignoré (format de tuple incorrect).")
                continue

            img_np, hdr, score, _wcs_obj, mask_2d = item_tuple # Déballer

            if img_np is None or hdr is None or score is None or mask_2d is None:
                self.update_progress(f"   -> Item {idx+1} (img/hdr/score/mask None) du lot {current_batch_num} ignoré.")
                continue

            # Déterminer la shape de référence et si le lot est couleur avec le premier item valide
            if ref_shape_check is None:
                ref_shape_check = img_np.shape
                is_color_batch = (img_np.ndim == 3 and img_np.shape[2] == 3)
                print(f"     - Référence shape pour lot: {ref_shape_check}, Couleur: {is_color_batch}")

            # Vérifier la cohérence des dimensions avec la référence
            is_current_item_valid_shape = False
            if is_color_batch:
                if img_np.ndim == 3 and img_np.shape == ref_shape_check and mask_2d.shape == ref_shape_check[:2]:
                    is_current_item_valid_shape = True
            else: # N&B
                if img_np.ndim == 2 and img_np.shape == ref_shape_check and mask_2d.shape == ref_shape_check:
                    is_current_item_valid_shape = True
            
            if is_current_item_valid_shape:
                valid_images_for_ccdproc.append(img_np)
                valid_headers_for_ccdproc.append(hdr)
                valid_scores_for_quality_weights.append(score)
                valid_pixel_masks_for_coverage.append(mask_2d)
            else:
                self.update_progress(f"   -> Item {idx+1} du lot {current_batch_num} ignoré (shape image {img_np.shape} ou masque {mask_2d.shape} incompatible avec réf {ref_shape_check}).")

        num_valid_images_for_processing = len(valid_images_for_ccdproc)
        print(f"DEBUG QM [_stack_batch]: {num_valid_images_for_processing}/{num_physical_images_in_batch_initial} images valides pour traitement dans ce lot.")

        if num_valid_images_for_processing == 0:
            self.update_progress(f"❌ Aucune image valide trouvée dans le lot {current_batch_num} après filtrage. Lot ignoré.")
            return None, None, None
        
        # La shape 2D pour la carte de couverture (H, W)
        shape_2d_for_coverage_map = ref_shape_check[:2] if is_color_batch else ref_shape_check

        # --- 2. Calculer les poids scalaires qualité pour les images VALIDES ---
        weight_scalars_for_ccdproc = None # Sera un array NumPy ou None
        sum_of_quality_weights_applied = float(num_valid_images_for_processing) # Défaut si pas de pondération
        quality_weighting_was_effectively_applied = False

        if self.use_quality_weighting:
            self.update_progress(f"   -> Calcul des poids qualité pour {num_valid_images_for_processing} images valides...")
            try:
                calculated_weights = self._calculate_weights(valid_scores_for_quality_weights) # Renvoie déjà un array NumPy
                if calculated_weights is not None and calculated_weights.size == num_valid_images_for_processing:
                    weight_scalars_for_ccdproc = calculated_weights
                    sum_of_quality_weights_applied = np.sum(weight_scalars_for_ccdproc)
                    quality_weighting_was_effectively_applied = True
                    self.update_progress(f"   -> Poids qualité (scalaires) calculés. Somme: {sum_of_quality_weights_applied:.2f}. Range: [{np.min(weight_scalars_for_ccdproc):.2f}-{np.max(weight_scalars_for_ccdproc):.2f}]")
                else:
                    self.update_progress(f"   ⚠️ Erreur calcul poids scalaires. Utilisation poids uniformes (1.0).")
                    # sum_of_quality_weights_applied reste num_valid_images_for_processing
            except Exception as w_err:
                self.update_progress(f"   ⚠️ Erreur pendant calcul poids scalaires: {w_err}. Utilisation poids uniformes (1.0).")
                # sum_of_quality_weights_applied reste num_valid_images_for_processing
        else:
            self.update_progress(f"   -> Pondération Qualité (scalaire) désactivée. Poids uniformes (1.0) seront utilisés par ccdproc.")
            # sum_of_quality_weights_applied reste num_valid_images_for_processing


        # --- 3. Préparer les CCDData pour ccdproc.combine ---
        ccd_list_all_channels = [] # Pour couleur: [[chR_img1,...], [chG_img1,...], [chB_img1,...]]
                                   # Pour N&B: sera juste une liste de CCDData N&B

        if is_color_batch:
            for _ in range(3): ccd_list_all_channels.append([]) # Initialiser listes pour R, G, B
            for i in range(num_valid_images_for_processing):
                img_np = valid_images_for_ccdproc[i]
                hdr = valid_headers_for_ccdproc[i]
                exposure = float(hdr.get('EXPTIME', 1.0)) # EXPTIME par image
                for c in range(3): # Pour chaque canal R, G, B
                    channel_data_2d = img_np[..., c]
                    channel_data_2d_clean = np.nan_to_num(channel_data_2d, nan=0.0, posinf=0.0, neginf=0.0)
                    ccd = CCDData(channel_data_2d_clean, unit='adu', meta=hdr.copy()) # Utiliser le header original
                    ccd.meta['EXPOSURE'] = exposure # S'assurer que EXPOSURE est dans meta pour ccdproc
                    ccd_list_all_channels[c].append(ccd)
        else: # Grayscale
            ccd_list_grayscale_for_combine = []
            for i in range(num_valid_images_for_processing):
                img_np = valid_images_for_ccdproc[i]
                hdr = valid_headers_for_ccdproc[i]
                exposure = float(hdr.get('EXPTIME', 1.0))
                img_np_clean = np.nan_to_num(img_np, nan=0.0, posinf=0.0, neginf=0.0)
                ccd = CCDData(img_np_clean, unit='adu', meta=hdr.copy())
                ccd.meta['EXPOSURE'] = exposure
                ccd_list_grayscale_for_combine.append(ccd)
            ccd_list_all_channels.append(ccd_list_grayscale_for_combine) # Mettre dans la structure attendue

        # --- 4. Stack images avec ccdproc.combine (comme avant) ---
        stacked_batch_data_np = None # Résultat HWC ou HW
        stack_method_used_for_header = self.stacking_mode
        kappa_val_for_header = float(self.kappa) # Assurer float

        try:
            if is_color_batch:
                self.update_progress(f"   -> Combinaison couleur par canal avec ccdproc.combine ({num_valid_images_for_processing} images/canal)...")
                stacked_channels_list = []
                
                for c in range(3): # Pour R, G, B
                    channel_name = ['R', 'G', 'B'][c]
                    current_ccd_list_for_channel = ccd_list_all_channels[c]
                    if not current_ccd_list_for_channel: raise ValueError(f"Aucune CCDData pour canal {channel_name}.")
                    
                    combine_kwargs = {'mem_limit': 2e9} # Limite mémoire ccdproc
                    if stack_method_used_for_header == 'mean': combine_kwargs['method'] = 'average'
                    elif stack_method_used_for_header == 'median': combine_kwargs['method'] = 'median'
                    elif stack_method_used_for_header in ['kappa-sigma', 'winsorized-sigma']:
                        combine_kwargs.update({
                            'method': 'average', 'sigma_clip': True,
                            'sigma_clip_low_thresh': kappa_val_for_header,
                            'sigma_clip_high_thresh': kappa_val_for_header
                        })
                        if stack_method_used_for_header == 'winsorized-sigma': # Note pour l'utilisateur
                            self.update_progress(f"   ℹ️ Mode 'winsorized' traité comme kappa-sigma ({kappa_val_for_header:.1f}) par ccdproc.combine")
                    else: combine_kwargs['method'] = 'average' # Fallback
                    
                    if weight_scalars_for_ccdproc is not None: # Si des poids scalaires ont été calculés
                         combine_kwargs['weights'] = weight_scalars_for_ccdproc
                    
                    print(f"      -> ccdproc.combine Canal {channel_name}. Méthode: {combine_kwargs.get('method')}, Poids scalaires: {'Oui' if 'weights' in combine_kwargs else 'Non'}")
                    combined_ccd_channel = ccdproc_combine(current_ccd_list_for_channel, **combine_kwargs)
                    stacked_channels_list.append(combined_ccd_channel.data.astype(np.float32))
                
                if len(stacked_channels_list) != 3: raise RuntimeError("ccdproc couleur n'a pas produit 3 canaux.")
                stacked_batch_data_np = np.stack(stacked_channels_list, axis=-1) # Reconstruire HWC
            
            else: # Grayscale
                current_ccd_list_for_channel = ccd_list_all_channels[0] # Il n'y a qu'une liste
                if not current_ccd_list_for_channel: raise ValueError("Aucune CCDData N&B à combiner.")
                self.update_progress(f"   -> Combinaison N&B avec ccdproc.combine ({len(current_ccd_list_for_channel)} images)...")
                combine_kwargs = {'mem_limit': 2e9}
                # ... (logique kwargs identique à la couleur)
                if stack_method_used_for_header == 'mean': combine_kwargs['method'] = 'average'
                elif stack_method_used_for_header == 'median': combine_kwargs['method'] = 'median'
                elif stack_method_used_for_header in ['kappa-sigma', 'winsorized-sigma']:
                    combine_kwargs.update({'method': 'average', 'sigma_clip': True, 'sigma_clip_low_thresh': kappa_val_for_header, 'sigma_clip_high_thresh': kappa_val_for_header})
                    if stack_method_used_for_header == 'winsorized-sigma': self.update_progress(f"   ℹ️ Mode 'winsorized' traité comme kappa-sigma ({kappa_val_for_header:.1f})")
                else: combine_kwargs['method'] = 'average'
                if weight_scalars_for_ccdproc is not None: combine_kwargs['weights'] = weight_scalars_for_ccdproc
                print(f"      -> ccdproc.combine N&B. Méthode: {combine_kwargs.get('method')}, Poids scalaires: {'Oui' if 'weights' in combine_kwargs else 'Non'}")
                combined_ccd_grayscale = ccdproc_combine(current_ccd_list_for_channel, **combine_kwargs)
                stacked_batch_data_np = combined_ccd_grayscale.data.astype(np.float32) # HW

            # --- Normalisation 0-1 de l'image moyenne du lot ---
            min_val_batch, max_val_batch = np.nanmin(stacked_batch_data_np), np.nanmax(stacked_batch_data_np)
            if np.isfinite(min_val_batch) and np.isfinite(max_val_batch) and max_val_batch > min_val_batch:
                stacked_batch_data_np = (stacked_batch_data_np - min_val_batch) / (max_val_batch - min_val_batch)
            elif np.isfinite(max_val_batch) and max_val_batch == min_val_batch: # Image constante
                 stacked_batch_data_np = np.full_like(stacked_batch_data_np, 0.5) # Gris
            else: # Tout NaN/Inf
                 stacked_batch_data_np = np.zeros_like(stacked_batch_data_np) # Noir
            stacked_batch_data_np = np.clip(stacked_batch_data_np, 0.0, 1.0).astype(np.float32)
            print(f"     - Image moyenne du lot normalisée 0-1. Shape: {stacked_batch_data_np.shape}")

        except MemoryError as mem_err:
            print(f"\n❌ ERREUR MÉMOIRE Combinaison Lot {progress_info}: {mem_err}"); traceback.print_exc(limit=1)
            self.update_progress(f"❌ ERREUR Mémoire ccdproc Lot {progress_info}. Lot ignoré.")
            gc.collect(); return None, None, None # Retourner None pour la carte de poids aussi
        except Exception as stack_err:
            print(f"\n❌ ERREUR ccdproc.combine Lot {progress_info}: {stack_err}"); traceback.print_exc(limit=3)
            self.update_progress(f"❌ ERREUR ccdproc.combine Lot {progress_info}. Lot ignoré.")
            gc.collect(); return None, None, None

        # --- 5. NOUVEAU : Calculer batch_coverage_map_2d (HxW, float32) ---
        print(f"   -> Calcul de la carte de poids/couverture 2D pour le lot #{current_batch_num}...")
        batch_coverage_map_2d = np.zeros(shape_2d_for_coverage_map, dtype=np.float32)
        
        for i in range(num_valid_images_for_processing):
            valid_pixel_mask_for_img = valid_pixel_masks_for_coverage[i] # C'est un masque booléen HW
            
            # Déterminer le poids scalaire à appliquer à ce masque
            current_image_scalar_weight = 1.0 # Défaut si pas de pondération
            if weight_scalars_for_ccdproc is not None: # Si la pondération qualité a été calculée
                current_image_scalar_weight = weight_scalars_for_ccdproc[i]
            
            # Ajouter le masque pondéré à la carte de couverture du lot
            # valid_pixel_mask_for_img.astype(np.float32) convertit True->1.0, False->0.0
            batch_coverage_map_2d += valid_pixel_mask_for_img.astype(np.float32) * current_image_scalar_weight
        
        print(f"     - Carte de poids/couverture 2D du lot calculée. Shape: {batch_coverage_map_2d.shape}, Range: [{np.min(batch_coverage_map_2d):.2f}-{np.max(batch_coverage_map_2d):.2f}]")

        # --- 6. Création de l'en-tête d'information (comme avant, mais utilise num_valid_images_for_processing) ---
        stack_info_header = fits.Header()
        stack_info_header['NIMAGES'] = (num_valid_images_for_processing, 'Valid images combined in this batch') # ASCII
        final_method_str_for_hdr = stack_method_used_for_header # Peut être "kappa-sigma(K)"
        if stack_method_used_for_header in ['kappa-sigma', 'winsorized-sigma']: final_method_str_for_hdr = f"kappa-sigma({kappa_val_for_header:.1f})"
        stack_info_header['STACKMETH'] = (final_method_str_for_hdr, 'CCDProc method for this batch')
        if 'kappa-sigma' in final_method_str_for_hdr: stack_info_header['KAPPA'] = (kappa_val_for_header, 'Kappa value for clipping')
        
        stack_info_header['WGHT_APP'] = (quality_weighting_was_effectively_applied, 'Quality weights (scalar) used by ccdproc_combine')
        if quality_weighting_was_effectively_applied:
            w_metrics_str_list = []
            if self.weight_by_snr: w_metrics_str_list.append(f"SNR^{self.snr_exponent:.1f}")
            if self.weight_by_stars: w_metrics_str_list.append(f"Stars^{self.stars_exponent:.1f}")
            stack_info_header['WGHT_MET'] = (",".join(w_metrics_str_list) if w_metrics_str_list else "None_Active", 'Metrics configured for scalar weighting')
            stack_info_header['SUMSCLW'] = (float(sum_of_quality_weights_applied), 'Sum of scalar quality weights in this batch')
        else:
            stack_info_header['SUMSCLW'] = (float(num_valid_images_for_processing), 'Effective num images (uniform scalar weight=1)')
        
        batch_total_exposure = 0.0
        for hdr_iter in valid_headers_for_ccdproc: # Utiliser les headers des images valides
            if hdr_iter and 'EXPTIME' in hdr_iter:
                try: batch_total_exposure += float(hdr_iter['EXPTIME'])
                except (ValueError, TypeError): pass
        stack_info_header['TOTEXP'] = (round(batch_total_exposure, 2), '[s] Sum of exposure times for images in this batch')

        self.update_progress(f"✅ Combinaison lot {progress_info} terminée (Shape: {stacked_batch_data_np.shape}).")
        
        # Retourner l'image stackée, le header d'info, et la NOUVELLE carte de couverture 2D du lot
        return stacked_batch_data_np, stack_info_header, batch_coverage_map_2d








#########################################################################################################################################



    def _combine_intermediate_drizzle_batches(self, intermediate_files_list, output_wcs, output_shape_2d_hw):
        """
        Combine les résultats Drizzle intermédiaires (par lot) sauvegardés sur disque.
        Utilise la classe Drizzle pour la combinaison pondérée par les cartes de poids.
        Adapté de full_drizzle.py/combine_batches.

        Args:
            intermediate_files_list (list): Liste de tuples [(sci_path, [wht_r_fpath, wht_g_fpath, wht_b_fpath]), ...].
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
        print(f"DEBUG QM [_combine_intermediate_drizzle_batches]: Début combinaison {num_batches_to_combine} lots.")
        print(f"  -> WCS Sortie Cible: {'Oui' if output_wcs else 'Non'}, Shape Sortie Cible: {output_shape_2d_hw}")
        combine_start_time = time.time()

        # --- Initialiser les objets Drizzle FINAUX ---
        num_output_channels = 3
        # channel_names = ['R', 'G', 'B']
        final_drizzlers = []
        final_output_images = [] 
        final_output_weights = [] 

        try:
            self.update_progress(f"   -> Initialisation Drizzle final (Shape: {output_shape_2d_hw})...")
            for _ in range(num_output_channels):
                final_output_images.append(np.zeros(output_shape_2d_hw, dtype=np.float32))
                final_output_weights.append(np.zeros(output_shape_2d_hw, dtype=np.float32))
            
            for i in range(num_output_channels):
                # =================== MODIFICATION IMPORTANTE ICI (déjà faite à l'étape 2) ===================
                # S'assurer que out_wcs et out_shape sont bien passés
                driz_ch = Drizzle(
                    kernel=self.drizzle_kernel,
                    fillval="0.0",
                    out_img=final_output_images[i],   # Tableau NumPy (H,W) avec la shape de SORTIE
                    out_wht=final_output_weights[i]  # Tableau NumPy (H,W) avec la shape de SORTIE
                    # PAS DE out_wcs ni out_shape ici pour stsci.drizzle.resample.Drizzle __init__
                )
                # =========================================================================================
                final_drizzlers.append(driz_ch)
            self.update_progress(f"   -> Objets Drizzle finaux initialisés.")
            print(f"DEBUG QM [_combine_intermediate_drizzle_batches]: Objets Drizzle finaux prêts.")

        except Exception as init_err:
            self.update_progress(f"   - ERREUR: Échec init Drizzle final: {init_err}"); traceback.print_exc(limit=1)
            print(f"ERREUR QM [_combine_intermediate_drizzle_batches]: Échec init Drizzle final: {init_err}")
            return None, None

        # --- Boucle sur les fichiers intermédiaires par lot ---
        total_contributing_ninputs = 0 # Pour suivre le nombre total d'images sources
        batches_combined_count = 0
        for i, (sci_fpath, wht_fpaths) in enumerate(intermediate_files_list):
            if self.stop_processing: self.update_progress("🛑 Arrêt demandé pendant combinaison lots Drizzle."); break
            self.update_progress(f"   -> Ajout lot intermédiaire {i+1}/{num_batches_to_combine}...")
            print(f"DEBUG QM [_combine_intermediate_drizzle_batches]: Traitement lot {i+1}: SCI='{os.path.basename(sci_fpath)}'")

            if len(wht_fpaths) != num_output_channels: 
                self.update_progress(f"      -> ERREUR: Nombre incorrect de cartes poids ({len(wht_fpaths)}) pour lot {i+1}. Attendu {num_output_channels}. Ignoré."); 
                print(f"ERREUR QM [_combine_intermediate_drizzle_batches]: Nombre de fichiers WHT incorrect pour {os.path.basename(sci_fpath)}")
                continue
            
            sci_data_chw = None; intermed_wcs = None; wht_maps_list = None; sci_header = None; combine_pixmap = None
            try:
                # Lire Science (CxHxW)
                with fits.open(sci_fpath, memmap=False) as hdul_sci:
                    if not hdul_sci or hdul_sci[0].data is None: raise IOError(f"Chunk science invalide ou vide: {sci_fpath}")
                    sci_data_chw = hdul_sci[0].data.astype(np.float32)
                    sci_header = hdul_sci[0].header
                    try: total_contributing_ninputs += int(sci_header.get('NINPUTS', 0)) # Sommer les NINPUTS des lots
                    except (ValueError, TypeError): pass 
                    
                    with warnings.catch_warnings(): 
                        warnings.simplefilter("ignore")
                        intermed_wcs = WCS(sci_header, naxis=2) # WCS du lot intermédiaire
                    if not intermed_wcs.is_celestial: raise ValueError("WCS intermédiaire non céleste.")
                    if sci_data_chw.ndim != 3 or sci_data_chw.shape[0] != num_output_channels: 
                        raise ValueError(f"Shape science lot {os.path.basename(sci_fpath)} invalide: {sci_data_chw.shape}, attendu CxHxW avec C={num_output_channels}")
                print(f"    -> SCI lot lu: {sci_data_chw.shape}. Range: [{np.min(sci_data_chw):.3f}, {np.max(sci_data_chw):.3f}]")

                # Lire Poids (HxW par canal)
                wht_maps_list = []
                valid_weights_for_this_batch = True
                for ch_idx, wht_fpath in enumerate(wht_fpaths):
                    try:
                        with fits.open(wht_fpath, memmap=False) as hdul_wht: 
                            wht_map_2d = hdul_wht[0].data.astype(np.float32)
                        if wht_map_2d.shape != sci_data_chw.shape[1:]: 
                            raise ValueError(f"Shape poids {wht_map_2d.shape} != science HxW {sci_data_chw.shape[1:]} pour canal {ch_idx}")
                        wht_map_2d[~np.isfinite(wht_map_2d)] = 0.0
                        wht_map_2d = np.maximum(wht_map_2d, 0.0) # Assurer non-négatif
                        wht_maps_list.append(wht_map_2d)
                        print(f"      - WHT Canal {ch_idx} lu: {wht_map_2d.shape}. Range: [{np.min(wht_map_2d):.2f}, {np.max(wht_map_2d):.2f}]")
                    except Exception as e_wht_read: 
                        self.update_progress(f"      -> ERREUR lecture poids {os.path.basename(wht_fpath)}: {e_wht_read}. Lot ignoré."); 
                        print(f"ERREUR QM [_combine_intermediate_drizzle_batches]: Échec lecture WHT {os.path.basename(wht_fpath)}")
                        valid_weights_for_this_batch = False; break
                if not valid_weights_for_this_batch: continue

                # Calcul Pixmap pour la combinaison (WCS intermédiaire vers WCS final)
                intermed_shape_hw = sci_data_chw.shape[1:] # H,W
                y_intermed, x_intermed = np.indices(intermed_shape_hw)
                try:
                    world_coords_intermed = intermed_wcs.all_pix2world(x_intermed.flatten(), y_intermed.flatten(), 0)
                    x_final, y_final = output_wcs.all_world2pix(world_coords_intermed[0], world_coords_intermed[1], 0)
                    combine_pixmap = np.dstack((x_final.reshape(intermed_shape_hw), y_final.reshape(intermed_shape_hw))).astype(np.float32)
                    print(f"    -> Pixmap de combinaison calculé.")
                except Exception as combine_map_err: 
                    self.update_progress(f"      -> ERREUR création pixmap combinaison: {combine_map_err}. Lot ignoré."); 
                    print(f"ERREUR QM [_combine_intermediate_drizzle_batches]: Échec pixmap combinaison pour {os.path.basename(sci_fpath)}")
                    continue
                
                # Ajout à Drizzle (par canal)
                if combine_pixmap is not None:
                    for ch_index in range(num_output_channels):
                        channel_data_sci_2d = sci_data_chw[ch_index, :, :] # Sélection du canal science (HxW)
                        channel_data_wht_2d = wht_maps_list[ch_index]       # Carte de poids HxW pour ce canal
                        
                        # Nettoyer données science (même si déjà fait avant sauvegarde, par sécurité)
                        channel_data_sci_2d[~np.isfinite(channel_data_sci_2d)] = 0.0
                        
                        # add_image attend 'data' (2D), 'pixmap' (H,W,2), 'weight_map' (2D)
                        final_drizzlers[ch_index].add_image(
                            data=channel_data_sci_2d,    # Image 2D du canal
                            pixmap=combine_pixmap,       # Pixmap de transformation
                            weight_map=channel_data_wht_2d, # Carte de poids 2D du canal
                            exptime=1.0,                 # Temps de pose (normalisé à 1 car données déjà en counts/s)
                            pixfrac=self.drizzle_pixfrac,  # Pixfrac de la session
                            in_units='cps'               # Unités des données science (counts per second)
                        )
                    batches_combined_count += 1
                    print(f"    -> Lot {i+1} ajouté aux Drizzlers finaux.")
                else: 
                    self.update_progress(f"      -> Warning: Pixmap combinaison est None pour lot {i+1}. Ignoré.")
            
            except FileNotFoundError: 
                self.update_progress(f"   - ERREUR: Fichier intermédiaire lot {i+1} non trouvé. Ignoré."); 
                print(f"ERREUR QM [_combine_intermediate_drizzle_batches]: Fichier non trouvé: {sci_fpath} ou ses poids")
                continue
            except (IOError, ValueError) as e_io_val: 
                self.update_progress(f"   - ERREUR lecture/validation lot intermédiaire {i+1}: {e_io_val}. Ignoré."); 
                print(f"ERREUR QM [_combine_intermediate_drizzle_batches]: Échec lecture/validation lot {i+1}: {e_io_val}")
                traceback.print_exc(limit=1)
                continue
            except Exception as e_lot: 
                self.update_progress(f"   - ERREUR traitement lot intermédiaire {i+1}: {e_lot}"); 
                print(f"ERREUR QM [_combine_intermediate_drizzle_batches]: Échec traitement lot {i+1}: {e_lot}")
                traceback.print_exc(limit=1); continue
            finally:
                del sci_data_chw, intermed_wcs, wht_maps_list, sci_header, combine_pixmap
                if (i + 1) % 5 == 0: gc.collect()
        # --- Fin boucle sur les lots intermédiaires ---

        combine_end_time = time.time()
        self.update_progress(f"💧 Combinaison finale Drizzle terminée ({batches_combined_count}/{num_batches_to_combine} lots combinés en {combine_end_time - combine_start_time:.2f}s).")
        print(f"DEBUG QM [_combine_intermediate_drizzle_batches]: Fin boucle combinaison. {batches_combined_count} lots combinés.")

        if batches_combined_count == 0:
            self.update_progress("❌ Aucun lot Drizzle intermédiaire n'a pu être combiné.")
            del final_drizzlers, final_output_images, final_output_weights; gc.collect()
            return None, None

        # --- Récupérer et assembler les résultats finaux ---
        try:
            # final_output_images et final_output_weights contiennent maintenant les données par canal
            final_sci_image_hxwxc = np.stack(final_output_images, axis=-1).astype(np.float32) # HxWxC
            final_wht_map_hxwxc = np.stack(final_output_weights, axis=-1).astype(np.float32) # HxWxC

            # Nettoyer les résultats finaux (sécurité)
            final_sci_image_hxwxc[~np.isfinite(final_sci_image_hxwxc)] = 0.0
            final_wht_map_hxwxc[~np.isfinite(final_wht_map_hxwxc)] = 0.0
            final_wht_map_hxwxc = np.maximum(final_wht_map_hxwxc, 0.0) # Assurer non-négatif pour les poids

            self.update_progress(f"   -> Assemblage final Drizzle terminé (Shape Sci: {final_sci_image_hxwxc.shape}, Shape WHT: {final_wht_map_hxwxc.shape})")
            print(f"DEBUG QM [_combine_intermediate_drizzle_batches]: Assemblage final OK. SCI Range: [{np.min(final_sci_image_hxwxc):.3f}, {np.max(final_sci_image_hxwxc):.3f}], WHT Range: [{np.min(final_wht_map_hxwxc):.2f}, {np.max(final_wht_map_hxwxc):.2f}]")

            # Mettre à jour le compteur total d'images basé sur les headers intermédiaires
            # Ceci est important pour le header FITS final
            self.images_in_cumulative_stack = total_contributing_ninputs
            print(f"DEBUG QM [_combine_intermediate_drizzle_batches]: images_in_cumulative_stack (depuis NINPUTS lots) = {self.images_in_cumulative_stack}")

            return final_sci_image_hxwxc, final_wht_map_hxwxc

        except Exception as e_final_asm:
            self.update_progress(f"   - ERREUR pendant assemblage final Drizzle: {e_final_asm}")
            print(f"ERREUR QM [_combine_intermediate_drizzle_batches]: Échec assemblage final: {e_final_asm}")
            traceback.print_exc(limit=1)
            return None, None
        finally:
            del final_drizzlers, final_output_images, final_output_weights
            gc.collect()

############################################################################################################################################





# --- DANS LA CLASSE SeestarQueuedStacker DANS seestar/queuep/queue_manager.py ---

    def _save_final_stack(self, output_filename_suffix: str = "", stopped_early: bool = False,
                          drizzle_final_sci_data=None, drizzle_final_wht_data=None):
        """
        Calcule l'image finale, applique les post-traitements et sauvegarde.
        Utilise drizzle_final_sci/wht_data si fournis (pour Drizzle Final/Incrémental).
        Sinon (Stacking Classique), lit depuis les memmaps SUM/W.
        MODIFIÉ: Condition pour utiliser les données Drizzle fournies étendue au mode Incrémental.
        """
        print("\n" + "=" * 80)
        print("DEBUG QM [_save_final_stack V3 - DrizIncr uses provided data]: Début sauvegarde finale.") # <-- MODIFIED PRINT
        print(f"  Suffixe: '{output_filename_suffix}', Arrêt précoce: {stopped_early}")
        
        # --- MODIFIED CONDITION: is_any_drizzle_mode_with_data ---
        is_any_drizzle_mode_with_data = (
            self.drizzle_active_session and
            (self.drizzle_mode == "Final" or self.drizzle_mode == "Incremental") and # Accepte les deux modes Drizzle
            drizzle_final_sci_data is not None and
            drizzle_final_wht_data is not None
        )
        # --- FIN MODIFICATION ---

        is_mosaic_mode_with_data = ( # Inchangé
            self.is_mosaic_run and
            drizzle_final_sci_data is not None
        )

        # --- MODIFIED LOG MESSAGE ---
        if is_any_drizzle_mode_with_data:
            print(f"  Mode: Drizzle '{self.drizzle_mode}' (utilisation des données combinées fournies).")
        elif is_mosaic_mode_with_data:
            print("  Mode: Mosaïque (utilisation des données combinées fournies).")
        else:
            print("  Mode: SUM/W Classique (lecture depuis memmaps).")
        print("=" * 80 + "\n")
        # --- FIN MODIFICATION ---

        self.update_progress(f"💾 Préparation de la sauvegarde finale (Mode: {'Drizzle '+self.drizzle_mode if is_any_drizzle_mode_with_data else ('Mosaïque' if is_mosaic_mode_with_data else 'SUM/W Classique')})...")

        final_image_initial = None
        final_wht_map_for_postproc = None
        background_model_photutils = None

        # --- 1. Obtenir les données initiales ---
        # --- MODIFIED CONDITION TO USE is_any_drizzle_mode_with_data ---
        if is_any_drizzle_mode_with_data or is_mosaic_mode_with_data:
            source_description = f"Drizzle {self.drizzle_mode} combiné" if is_any_drizzle_mode_with_data else "Mosaïque combinée"
            print(f"DEBUG QM [_save_final_stack V3]: Utilisation des données {source_description} (drizzle_final_sci/wht_data).")
            self.update_progress(f"Traitement des données {source_description}...")

            final_image_initial_raw = drizzle_final_sci_data 
            
            min_r_raw, max_r_raw = np.nanmin(final_image_initial_raw), np.nanmax(final_image_initial_raw)
            if np.isfinite(min_r_raw) and np.isfinite(max_r_raw) and max_r_raw > min_r_raw:
                 final_image_initial = (final_image_initial_raw - min_r_raw) / (max_r_raw - min_r_raw)
            elif np.any(np.isfinite(final_image_initial_raw)):
                 final_image_initial = np.full_like(final_image_initial_raw, 0.5)
            else:
                 final_image_initial = np.zeros_like(final_image_initial_raw)
            final_image_initial = np.clip(final_image_initial, 0.0, 1.0).astype(np.float32)
            print(f"  -> Image Drizzle/Mosaïque normalisée 0-1. Shape: {final_image_initial.shape}")

            if drizzle_final_wht_data is not None:
                final_wht_map_for_postproc = np.mean(drizzle_final_wht_data, axis=2).astype(np.float32)
                final_wht_map_for_postproc = np.maximum(final_wht_map_for_postproc, 0.0)
                print(f"  -> Carte de poids 2D Drizzle/Mosaïque (moyenne canaux) créée. Shape: {final_wht_map_for_postproc.shape}")
            else:
                print("  -> WARNING: drizzle_final_wht_data est None. final_wht_map_for_postproc sera None.")
                final_wht_map_for_postproc = None

            self._close_memmaps() 
            print("DEBUG QM [_save_final_stack V3]: Memmaps fermés (mode Drizzle / Mosaïque).")
        # --- FIN MODIFICATION ---
        else: # Mode SUM/W Classique (logique inchangée)
            print("DEBUG QM [_save_final_stack V3]: Utilisation des accumulateurs SUM/W (memmap) pour stacking classique.")
            if (self.cumulative_sum_memmap is None or
                    self.cumulative_wht_memmap is None or
                    self.output_folder is None or
                    not os.path.isdir(self.output_folder)):
                self.final_stacked_path = None
                self.update_progress("❌ Erreur interne: Accumulateurs memmap ou dossier sortie non définis. Sauvegarde annulée.")
                self._close_memmaps(); return

            try:
                self.update_progress("Lecture des données finales depuis les accumulateurs memmap...")
                final_sum = np.array(self.cumulative_sum_memmap, dtype=np.float64)
                final_wht_map_for_postproc = np.array(self.cumulative_wht_memmap, dtype=np.float32)
                print("DEBUG QM [_save_final_stack V3]: Données lues depuis memmap pour stacking classique.")

                print("DEBUG QM [_save_final_stack V3]: Fermeture des memmaps après lecture (mode SUM/W classique)...")
                self._close_memmaps()

                self.update_progress("Calcul de l'image moyenne (SUM / WHT)...")
                epsilon = 1e-9
                wht_for_division = np.maximum(final_wht_map_for_postproc.astype(np.float64), epsilon)
                wht_broadcasted = wht_for_division[..., np.newaxis]

                with np.errstate(divide='ignore', invalid='ignore'):
                    final_raw = final_sum / wht_broadcasted
                final_raw = np.nan_to_num(final_raw, nan=0.0, posinf=0.0, neginf=0.0)

                min_r_raw, max_r_raw = np.nanmin(final_raw), np.nanmax(final_raw)
                if np.isfinite(min_r_raw) and np.isfinite(max_r_raw) and max_r_raw > min_r_raw:
                     final_image_initial = (final_raw - min_r_raw) / (max_r_raw - min_r_raw)
                elif np.any(np.isfinite(final_raw)):
                     final_image_initial = np.full_like(final_raw, 0.5)
                else:
                     final_image_initial = np.zeros_like(final_raw)
                final_image_initial = np.clip(final_image_initial, 0.0, 1.0).astype(np.float32)

                del final_sum, wht_for_division, wht_broadcasted, final_raw; gc.collect()
                self.update_progress(f"Image moyenne SUM/W classique calculée. Range après norm 0-1: [{np.nanmin(final_image_initial):.3f}, {np.nanmax(final_image_initial):.3f}]")

            except Exception as e_calc_sumw:
                print(f"ERREUR QM [_save_final_stack V3]: Erreur calcul final SUM/W classique - {e_calc_sumw}"); traceback.print_exc(limit=2)
                self.update_progress(f"❌ Erreur lors du calcul final SUM/W classique: {e_calc_sumw}")
                self.processing_error = f"Erreur Calcul Final SUM/W: {e_calc_sumw}"
                self._close_memmaps(); return
        
        # --- 2. Vérifier si on a une image à traiter (inchangé) ---
        if final_image_initial is None:
            self.final_stacked_path = None; self.update_progress("ⓘ Aucun stack final (échec calcul image initiale). Sauvegarde annulée.")
            print("DEBUG QM [_save_final_stack V3]: final_image_initial est None. Sortie."); return

        effective_image_count = self.images_in_cumulative_stack; max_wht_value = 0.0
        if final_wht_map_for_postproc is not None and final_wht_map_for_postproc.size > 0:
            try: max_wht_value = np.max(final_wht_map_for_postproc)
            except Exception: pass
        
        if effective_image_count <= 0 and max_wht_value <= 1e-6 and not stopped_early:
            self.final_stacked_path = None; self.update_progress(f"ⓘ Aucun stack final (0 images/poids accumulés). Sauvegarde annulée.")
            print(f"DEBUG QM [_save_final_stack V3]: Sortie précoce (comptes/poids faibles et non arrêté tôt)."); return
        
        self.update_progress(f"Nombre d'images/poids effectifs accumulés: {effective_image_count} (Poids max WHT post-proc: {max_wht_value:.2f})")
        data_to_save = final_image_initial.copy()

        self.bn_globale_applied_in_session = False; self.photutils_bn_applied_in_session = False
        self.cb_applied_in_session = False; self.feathering_applied_in_session = False
        self.low_wht_mask_applied_in_session = False; self.scnr_applied_in_session = False
        self.crop_applied_in_session = False; self.photutils_params_used_in_session = {}
        
        # --- 3. Pipeline de Post-Traitement (logique inchangée) ---
        print("\n" + "=" * 80); print("DEBUG QM [_save_final_stack V3]: Début pipeline Post-Traitements."); print("=" * 80 + "\n")
        print(f"DEBUG QM [_save_final_stack V3]: Range data_to_save AVANT Post-Proc: [{np.nanmin(data_to_save):.3f}, {np.nanmax(data_to_save):.3f}]")
        # ... (Tout le pipeline de post-traitement BN -> Photutils -> CB -> Feather -> LowWHT -> SCNR -> Crop reste identique)
        wht_for_edge_effects = final_wht_map_for_postproc 
        if wht_for_edge_effects is None: 
            print("DEBUG QM [_save_final_stack V3]: wht_for_edge_effects est None. Création carte simulée...")
            self.update_progress("ℹ️ Carte de poids non disponible pour effets de bord, utilisation d'une carte géométrique simulée.")
            try:
                h_sim, w_sim = data_to_save.shape[:2]; center_y, center_x = (h_sim - 1) / 2.0, (w_sim - 1) / 2.0
                y_coords, x_coords = np.ogrid[:h_sim, :w_sim]; dist_sq = ((y_coords - center_y)**2 / (h_sim / 2.0)**2) + ((x_coords - center_x)**2 / (w_sim / 2.0)**2)
                cos_arg = np.clip(dist_sq * 0.5 * (np.pi / 2.0), 0, np.pi / 2.0); simulated_wht_2d_profile = np.cos(cos_arg)**2
                wht_for_edge_effects = np.maximum(simulated_wht_2d_profile, 1e-5).astype(np.float32)
                print(f"  -> Carte simulée (HxW) créée. Range: [{np.min(wht_for_edge_effects):.3f} - {np.max(wht_for_edge_effects):.3f}]")
            except Exception as e_sim_wht: print(f"ERREUR QM [_save_final_stack V3]: Échec création carte simulée: {e_sim_wht}."); wht_for_edge_effects = np.ones(data_to_save.shape[:2], dtype=np.float32)
        elif np.all(wht_for_edge_effects <= 1e-6): 
             print("DEBUG QM [_save_final_stack V3]: WHT réelle est vide. Création carte simulée...")
             self.update_progress("ℹ️ Carte de poids réelle vide, utilisation d'une carte géométrique simulée pour effets de bord.")
             try:
                h_sim, w_sim = data_to_save.shape[:2]; center_y, center_x = (h_sim - 1) / 2.0, (w_sim - 1) / 2.0
                y_coords, x_coords = np.ogrid[:h_sim, :w_sim]; dist_sq = ((y_coords - center_y)**2 / (h_sim / 2.0)**2) + ((x_coords - center_x)**2 / (w_sim / 2.0)**2)
                cos_arg = np.clip(dist_sq * 0.5 * (np.pi / 2.0), 0, np.pi / 2.0); simulated_wht_2d_profile = np.cos(cos_arg)**2
                wht_for_edge_effects = np.maximum(simulated_wht_2d_profile, 1e-5).astype(np.float32)
             except Exception as e_sim_wht2: print(f"ERREUR QM [_save_final_stack V3]: Échec création carte simulée (fallback 2): {e_sim_wht2}."); wht_for_edge_effects = np.ones(data_to_save.shape[:2], dtype=np.float32)
        else: 
            print("DEBUG QM [_save_final_stack V3]: Utilisation de la carte WHT réelle pour effets de bord. Normalisation...")
            max_wht_val = np.nanmax(wht_for_edge_effects)
            if max_wht_val > 1e-9: wht_for_edge_effects_normalized = wht_for_edge_effects / max_wht_val; wht_for_edge_effects = np.clip(wht_for_edge_effects_normalized, 0.0, 1.0)
            print(f"  -> Carte WHT réelle normalisée. Range: [{np.min(wht_for_edge_effects):.3f} - {np.max(wht_for_edge_effects):.3f}]")

        print("\n--- Étape Post-Proc (1/7): BN Globale ---"); # BN Globale
        if data_to_save.ndim == 3 and data_to_save.shape[2] == 3 and _BN_AVAILABLE: # ... (code BN identique)
            bn_params_used = {'grid_size': (16,16), 'bg_percentile_low': getattr(self, 'bn_perc_low', 5), 'bg_percentile_high': getattr(self, 'bn_perc_high', 30), 'std_factor_threshold': getattr(self, 'bn_std_factor', 1.0), 'min_pixels_per_zone': 50, 'min_applied_gain': getattr(self, 'bn_min_gain', 0.2), 'max_applied_gain': getattr(self, 'bn_max_gain', 7.0)}; parts = getattr(self, 'bn_grid_size_str', "16x16").split('x')
            if len(parts) == 2:
                try: bn_params_used['grid_size'] = (int(parts[0]), int(parts[1]))
                except ValueError: print(f"WARN QM: Taille grille BN ('{getattr(self, 'bn_grid_size_str', 'N/A')}') invalide.")
            self.update_progress(f"🎨 Application Neutralisation Fond Auto (BN)... Params: Grille={bn_params_used['grid_size']}, PercL={bn_params_used['bg_percentile_low']}, PercH={bn_params_used['bg_percentile_high']}, StdF={bn_params_used['std_factor_threshold']:.1f}, GainMin={bn_params_used['min_applied_gain']:.2f}, GainMax={bn_params_used['max_applied_gain']:.2f}")
            try: data_to_save = neutralize_background_automatic(data_to_save, **bn_params_used); self.bn_globale_applied_in_session = True; self.update_progress("   ✅ Neutralisation Fond Auto (BN) terminée.")
            except Exception as bn_err: self.update_progress(f"   ❌ Erreur Neutralisation Fond Auto (BN): {bn_err}. Étape ignorée."); print(f"ERREUR QM: Erreur neutralize_background_automatic: {bn_err}"); traceback.print_exc(limit=1)
        elif data_to_save.ndim != 3 or data_to_save.shape[2] != 3: self.update_progress("   ℹ️ BN Globale ignoré (image N&B ou fonction indisponible)." if _BN_AVAILABLE else "   ℹ️ BN Globale non activé ou fonction non disponible. Étape ignorée.")
        else: self.update_progress("   ℹ️ BN Globale non activé.")
        print(f"DEBUG QM [_save_final_stack V3]: Range data_to_save APRES BN Globale: [{np.nanmin(data_to_save):.3f}, {np.nanmax(data_to_save):.3f}]")

        print("\n--- Étape Post-Proc (2/7): Photutils BN ---"); # Photutils BN
        if getattr(self, 'apply_photutils_bn', False) and _PHOTOUTILS_BG_SUB_AVAILABLE: # ... (code Photutils BN identique)
            photutils_params = {'box_size': getattr(self, 'photutils_bn_box_size', 128), 'filter_size': getattr(self, 'photutils_bn_filter_size', 5), 'sigma_clip_val': getattr(self, 'photutils_bn_sigma_clip', 3.0), 'exclude_percentile': getattr(self, 'photutils_bn_exclude_percentile', 98.0)}; self.photutils_params_used_in_session = photutils_params.copy()
            self.update_progress(f"🔬 Application Soustraction Fond 2D (Photutils)... Params: Box={photutils_params['box_size']}, Filt={photutils_params['filter_size']}, Sig={photutils_params['sigma_clip_val']:.1f}, Excl%={photutils_params['exclude_percentile']:.1f}")
            try:
                data_corr, bkg_model = subtract_background_2d(data_to_save, **photutils_params)
                if data_corr is not None:
                    data_to_save = data_corr; background_model_photutils = bkg_model; self.photutils_bn_applied_in_session = True
                    if bkg_model is not None: 
                        try: mn_bkg, med_bkg, sd_bkg = sigma_clipped_stats(bkg_model); self.update_progress(f"   Modèle Fond 2D: Médiane={med_bkg:.4f}, StdDev={sd_bkg:.4f}")
                        except Exception: pass
                    mn_phot, mx_phot = np.nanmin(data_to_save), np.nanmax(data_to_save)
                    if np.isfinite(mn_phot) and np.isfinite(mx_phot) and mx_phot > mn_phot: data_to_save = (data_to_save - mn_phot) / (mx_phot - mn_phot)
                    elif np.any(np.isfinite(data_to_save)): data_to_save = np.full_like(data_to_save, 0.5)
                    else: data_to_save = np.zeros_like(data_to_save)
                    data_to_save = np.clip(data_to_save, 0.0, 1.0).astype(np.float32)
                    self.update_progress(f"   ✅ Soustraction Fond 2D (Photutils) terminée. Nouveau range: [{np.nanmin(data_to_save):.3f}, {np.nanmax(data_to_save):.3f}]")
                else: self.update_progress("   ⚠️ Échec Soustraction Fond 2D (Photutils), étape ignorée.")
            except Exception as photutils_err: self.update_progress(f"   ❌ Erreur Soustraction Fond 2D (Photutils): {photutils_err}. Étape ignorée."); print(f"ERREUR QM: Erreur subtract_background_2d: {photutils_err}"); traceback.print_exc(limit=1)
        elif getattr(self, 'apply_photutils_bn', False) and not _PHOTOUTILS_BG_SUB_AVAILABLE: self.update_progress("   ⚠️ Soustraction Fond 2D (Photutils) demandée mais Photutils indisponible. Étape ignorée.")
        else: self.update_progress("   ℹ️ Soustraction Fond 2D (Photutils) non activée.")
        print(f"DEBUG QM [_save_final_stack V3]: Range data_to_save APRES Photutils BN: [{np.nanmin(data_to_save):.3f}, {np.nanmax(data_to_save):.3f}]")

        print("\n--- Étape Post-Proc (3/7): Chromatic Balancer ---"); # Chromatic Balancer
        if getattr(self, 'apply_chroma_correction', True) and hasattr(self, 'chroma_balancer') and data_to_save.ndim == 3 and data_to_save.shape[2] == 3: # ... (code CB identique)
            cb_params_used = {'border_size': getattr(self, 'cb_border_size', 25), 'blur_radius': getattr(self, 'cb_blur_radius', 8), 'r_factor_limits': (getattr(self.chroma_balancer, 'r_factor_min', 0.7), getattr(self.chroma_balancer, 'r_factor_max', 1.3)), 'b_factor_limits': (getattr(self.chroma_balancer, 'b_factor_min', 0.4), getattr(self.chroma_balancer, 'b_factor_max', 1.5)) }
            self.update_progress(f"🌈 Application Correction Bords/Chroma (CB)... Params: Bord={cb_params_used['border_size']}, Flou={cb_params_used['blur_radius']}, LimR=[{cb_params_used['r_factor_limits'][0]:.2f}-{cb_params_used['r_factor_limits'][1]:.2f}], LimB=[{cb_params_used['b_factor_limits'][0]:.2f}-{cb_params_used['b_factor_limits'][1]:.2f}]")
            try:
                if hasattr(self.chroma_balancer, 'border_size'): self.chroma_balancer.border_size = cb_params_used['border_size']
                if hasattr(self.chroma_balancer, 'blur_radius'): self.chroma_balancer.blur_radius = cb_params_used['blur_radius']
                if hasattr(self.chroma_balancer, 'r_factor_min'): self.chroma_balancer.r_factor_min = cb_params_used['r_factor_limits'][0]; # ... etc. pour les autres facteurs
                if hasattr(self.chroma_balancer, 'r_factor_max'): self.chroma_balancer.r_factor_max = cb_params_used['r_factor_limits'][1]
                if hasattr(self.chroma_balancer, 'b_factor_min'): self.chroma_balancer.b_factor_min = cb_params_used['b_factor_limits'][0]
                if hasattr(self.chroma_balancer, 'b_factor_max'): self.chroma_balancer.b_factor_max = cb_params_used['b_factor_limits'][1]
                data_to_save = self.chroma_balancer.normalize_stack(data_to_save); self.cb_applied_in_session = True; self.update_progress("   ✅ Correction Bords/Chroma (CB) terminée.")
            except Exception as cb_err: self.update_progress(f"   ❌ Erreur Correction Bords/Chroma (CB): {cb_err}. Étape ignorée."); print(f"ERREUR QM: Erreur chroma_balancer.normalize_stack: {cb_err}"); traceback.print_exc(limit=1)
        elif getattr(self, 'apply_chroma_correction', True) and data_to_save.ndim != 3: self.update_progress("   ℹ️ Correction Bords/Chroma ignorée (image N&B ou fonction indisponible)." if hasattr(self, 'chroma_balancer') and self.chroma_balancer else "   ℹ️ Correction Bords/Chroma non activée ou fonction non disponible. Étape ignorée.")
        else: self.update_progress("   ℹ️ Correction Bords/Chroma non activée.")
        print(f"DEBUG QM [_save_final_stack V3]: Range data_to_save APRES Chromatic Balancer: [{np.nanmin(data_to_save):.3f}, {np.nanmax(data_to_save):.3f}]")

        print("\n--- Étape Post-Proc (4/7): Feathering ---"); # Feathering
        if getattr(self, 'apply_feathering', False): # ... (code Feathering identique)
            if _FEATHERING_AVAILABLE and wht_for_edge_effects is not None and data_to_save.ndim == 3 and data_to_save.shape[2] == 3 :
                feather_blur_val = getattr(self, 'feather_blur_px', 256); min_feather_gain = 0.5; max_feather_gain = 2.0
                self.update_progress(f"🖌️ Application Feathering (Lissage pondéré)... Params: Flou={feather_blur_val}px, GainMin={min_feather_gain:.2f}, GainMax={max_feather_gain:.2f}")
                try: data_to_save = feather_by_weight_map(data_to_save, wht_for_edge_effects, blur_px=feather_blur_val, min_gain=min_feather_gain, max_gain=max_feather_gain); self.feathering_applied_in_session = True; self.update_progress(f"   ✅ Feathering appliqué.")
                except Exception as feather_err: self.update_progress(f"   ❌ Erreur Feathering: {feather_err}. Étape ignorée."); print(f"ERREUR QM: Erreur feather_by_weight_map: {feather_err}"); traceback.print_exc(limit=1)
            elif getattr(self, 'apply_feathering', False):
                 if data_to_save.ndim != 3: self.update_progress("   ℹ️ Feathering ignoré (image N&B).")
                 elif wht_for_edge_effects is None: self.update_progress("   ⚠️ Feathering activé mais carte de poids pour effets de bord non disponible. Étape ignorée.")
                 elif not _FEATHERING_AVAILABLE: self.update_progress("   ⚠️ Feathering activé mais fonction non disponible. Étape ignorée.")
        else: self.update_progress("   ℹ️ Feathering non activé.")
        print(f"DEBUG QM [_save_final_stack V3]: Range data_to_save APRES Feathering: [{np.nanmin(data_to_save):.3f}, {np.nanmax(data_to_save):.3f}]")

        print("\n--- Étape Post-Proc (5/7): Low WHT Mask ---"); # Low WHT Mask
        if getattr(self, 'apply_low_wht_mask', False): # ... (code Low WHT Mask identique)
            if _LOW_WHT_MASK_AVAILABLE and wht_for_edge_effects is not None:
                pct_low_wht = getattr(self, 'low_wht_percentile', 5); soften_val_low_wht = getattr(self, 'low_wht_soften_px', 128)
                self.update_progress(f"😷 Application Masque Bas WHT (Percentile: {pct_low_wht}%, Adoucir: {soften_val_low_wht}px)...")
                try: data_to_save = apply_low_wht_mask(data_to_save, wht_for_edge_effects, percentile=pct_low_wht, soften_px=soften_val_low_wht, progress_callback=self.update_progress); self.low_wht_mask_applied_in_session = True; self.update_progress(f"   ✅ Masque Bas WHT appliqué.")
                except Exception as low_wht_err: self.update_progress(f"   ❌ Erreur Masque Bas WHT: {low_wht_err}. Étape ignorée."); print(f"ERREUR QM: Erreur apply_low_wht_mask: {low_wht_err}"); traceback.print_exc(limit=1)
            elif getattr(self, 'apply_low_wht_mask', False):
                 if wht_for_edge_effects is None: self.update_progress("   ⚠️ Masque Bas WHT activé mais carte de poids pour effets de bord non disponible. Étape ignorée.")
                 elif not _LOW_WHT_MASK_AVAILABLE: self.update_progress("   ⚠️ Masque Bas WHT activé mais fonction non disponible. Étape ignorée.")
        else: self.update_progress("   ℹ️ Masque Bas WHT non activé.")
        print(f"DEBUG QM [_save_final_stack V3]: Range data_to_save APRES Low WHT Mask: [{np.nanmin(data_to_save):.3f}, {np.nanmax(data_to_save):.3f}]")

        print("\n--- Étape Post-Proc (6/7): SCNR Final ---"); # SCNR Final
        if getattr(self, 'apply_final_scnr', False) and _SCNR_AVAILABLE and data_to_save.ndim == 3 and data_to_save.shape[2] == 3: # ... (code SCNR identique)
            scnr_target = getattr(self, 'final_scnr_target_channel', 'green'); scnr_amount = getattr(self, 'final_scnr_amount', 0.8); scnr_preserve_lum = getattr(self, 'final_scnr_preserve_luminosity', True)
            self.update_progress(f"🌿 Application SCNR Final (Cible: {scnr_target}, Force: {scnr_amount:.2f}, Prés.Lum: {scnr_preserve_lum})...")
            try: data_to_save = apply_scnr(data_to_save, target_channel=scnr_target, amount=scnr_amount, preserve_luminosity=scnr_preserve_lum); self.scnr_applied_in_session = True; self.update_progress("   ✅ SCNR Final terminé.")
            except Exception as scnr_err: self.update_progress(f"   ❌ Erreur SCNR Final: {scnr_err}. Étape ignorée."); print(f"ERREUR QM: Erreur apply_scnr: {scnr_err}"); traceback.print_exc(limit=1)
        elif getattr(self, 'apply_final_scnr', False):
            if data_to_save.ndim != 3: self.update_progress("   ℹ️ SCNR Final ignoré (image N&B).")
            elif not _SCNR_AVAILABLE: self.update_progress("   ⚠️ SCNR Final activé mais fonction non disponible. Étape ignorée.")
        else: self.update_progress("   ℹ️ SCNR Final non activé.")
        print(f"DEBUG QM [_save_final_stack V3]: Range data_to_save APRES SCNR Final: [{np.nanmin(data_to_save):.3f}, {np.nanmax(data_to_save):.3f}]")

        print("\n--- Étape Post-Proc (7/7): Rognage Final ---"); # Rognage Final
        final_crop_decimal = getattr(self, 'final_edge_crop_percent_decimal', 0.02) # ... (code Crop identique)
        if _CROP_AVAILABLE and final_crop_decimal > 1e-6 :
            crop_percent_val_display = final_crop_decimal * 100.0
            self.update_progress(f"✂️ Application Rognage Final des Bords ({crop_percent_val_display:.1f}%)...")
            try:
                shape_before_crop = data_to_save.shape; data_to_save = apply_edge_crop(data_to_save, final_crop_decimal)
                if data_to_save is None: self.update_progress("   ❌ Erreur critique lors du rognage. Sauvegarde annulée."); print("ERREUR QM: apply_edge_crop retourné None."); return
                self.crop_applied_in_session = True; self.update_progress(f"   ✅ Rognage terminé. Shape: {shape_before_crop} -> {data_to_save.shape}")
            except Exception as crop_err: self.update_progress(f"   ❌ Erreur Rognage Final: {crop_err}. Étape ignorée."); print(f"ERREUR QM: Erreur apply_edge_crop: {crop_err}"); traceback.print_exc(limit=1)
        elif _CROP_AVAILABLE: self.update_progress("   ℹ️ Rognage Final non activé (pourcentage nul).")
        else: self.update_progress("   ℹ️ ⚠️ Rognage Final non activé ou fonction non disponible. Étape ignorée.")
        print(f"DEBUG QM [_save_final_stack V3]: Range data_to_save APRES Rognage Final: [{np.nanmin(data_to_save):.3f}, {np.nanmax(data_to_save):.3f}]")
        
        print("\n" + "=" * 80); print("DEBUG QM [_save_final_stack V3]: Fin pipeline Post-Traitements."); print("=" * 80 + "\n")

        # --- 4. Header FITS final (logique inchangée) ---
        final_header = self.current_stack_header.copy() if self.current_stack_header else fits.Header()
        if is_any_drizzle_mode_with_data or is_mosaic_mode_with_data: # Note: is_any_drizzle_mode_with_data utilisé ici
            if self.drizzle_output_wcs: print("DEBUG QM [_save_final_stack V3]: Mise à jour du header final avec drizzle_output_wcs."); final_header.update(self.drizzle_output_wcs.to_header(relax=True))
            elif is_mosaic_mode_with_data and self.current_stack_header and self.current_stack_header.get('CTYPE1'): print("DEBUG QM [_save_final_stack V3]: Utilisation du WCS déjà présent dans current_stack_header pour Mosaïque.")
            else: print("WARN QM [_save_final_stack V3]: WCS de sortie Drizzle/Mosaïque non disponible pour le header final.")
        final_header['NIMAGES'] = (effective_image_count, 'Effective images/Total Weight for final stack'); final_header['TOTEXP']  = (round(self.total_exposure_seconds, 2), '[s] Approx total exposure')
        if is_any_drizzle_mode_with_data: final_header['STACKTYP'] = (f'Drizzle {self.drizzle_mode} ({self.drizzle_scale:.0f}x)', 'Stacking method') # Utilise self.drizzle_mode
        elif is_mosaic_mode_with_data: final_header['STACKTYP'] = (f'Mosaic Drizzle ({self.drizzle_scale:.0f}x)', 'Mosaic from solved panels')
        final_header.add_comment("--- Post-Processing Applied ---", before='HISTORY'); # ... (reste de la logique du header identique)
        final_header['BN_GLOB'] = (self.bn_globale_applied_in_session, "Global Background Neutralization applied")
        if self.bn_globale_applied_in_session: final_header['BN_GRID'] = (str(getattr(self, 'bn_grid_size_str', '')), "BN: Grid size (RxC)"); final_header['BN_PLOW'] = (int(getattr(self, 'bn_perc_low', 0)), "BN: Background Percentile Low"); final_header['BN_PHIGH'] = (int(getattr(self, 'bn_perc_high', 0)), "BN: Background Percentile High"); final_header['BN_STDF'] = (float(getattr(self, 'bn_std_factor', 0.0)), "BN: Std Factor"); final_header['BN_MING'] = (float(getattr(self, 'bn_min_gain', 0.0)), "BN: Min Applied Gain"); final_header['BN_MAXG'] = (float(getattr(self, 'bn_max_gain', 0.0)), "BN: Max Applied Gain")
        final_header['PB2D_APP'] = (self.photutils_bn_applied_in_session, "Photutils Background2D Applied")
        if self.photutils_bn_applied_in_session: final_header['PB_BOX'] = (self.photutils_params_used_in_session.get('box_size', 0), "Photutils: Box Size (px)"); final_header['PB_FILT'] = (self.photutils_params_used_in_session.get('filter_size', 0), "Photutils: Filter Size (px)"); final_header['PB_SIG'] = (self.photutils_params_used_in_session.get('sigma_clip_val', 0.0), "Photutils: Sigma Clip"); final_header['PB_EXCP'] = (self.photutils_params_used_in_session.get('exclude_percentile', 0.0), "Photutils: Exclude Percentile")
        final_header['CB_EDGE'] = (self.cb_applied_in_session, "Edge/Chroma Correction (CB) applied")
        if self.cb_applied_in_session: final_header['CB_BORD'] = (int(getattr(self, 'cb_border_size',0)), "CB: Border size (px)"); final_header['CB_BLUR'] = (int(getattr(self, 'cb_blur_radius',0)), "CB: Blur radius (px)"); final_header['CB_MINR'] = (float(getattr(self.chroma_balancer, 'r_factor_min',0.0) if hasattr(self,'chroma_balancer') else 0.0), "CB: Min Red Factor"); final_header['CB_MAXR'] = (float(getattr(self.chroma_balancer, 'r_factor_max',0.0) if hasattr(self,'chroma_balancer') else 0.0), "CB: Max Red Factor"); final_header['CB_MINB'] = (float(getattr(self.chroma_balancer, 'b_factor_min',0.0) if hasattr(self,'chroma_balancer') else 0.0), "CB: Min Blue Factor"); final_header['CB_MAXB'] = (float(getattr(self.chroma_balancer, 'b_factor_max',0.0) if hasattr(self,'chroma_balancer') else 0.0), "CB: Max Blue Factor")
        final_header['FEATHER'] = (self.feathering_applied_in_session, "Feathering by weight map applied")
        if self.feathering_applied_in_session: final_header['FTHR_BLR'] = (int(getattr(self, 'feather_blur_px', 0)), "Feathering blur radius (px)")
        final_header['LWMASK'] = (self.low_wht_mask_applied_in_session, "Low WHT Mask applied")
        if self.low_wht_mask_applied_in_session: final_header['LWMPCT'] = (int(getattr(self, 'low_wht_percentile', 0)), "Low WHT Mask Percentile"); final_header['LWMSFT'] = (int(getattr(self, 'low_wht_soften_px', 0)), "Low WHT Mask Soften Px")
        final_header['SCNR_APP'] = (self.scnr_applied_in_session, 'Final SCNR applied')
        if self.scnr_applied_in_session: final_header['SCNR_TRG'] = (self.final_scnr_target_channel, 'SCNR target'); final_header['SCNR_AMT'] = (float(self.final_scnr_amount), 'SCNR amount'); final_header['SCNR_PLM'] = (self.final_scnr_preserve_luminosity, 'SCNR preserve luminosity')
        final_header['CROP_APP'] = (self.crop_applied_in_session, 'Final Edge Crop applied')
        if self.crop_applied_in_session: final_header['CROP_PCT'] = (float(getattr(self, 'final_edge_crop_percent_decimal', 0.0) * 100.0), "Final Edge Crop (%)")
        
        # --- 5. Construction du nom de fichier (logique inchangée) ---
        base_name = "stack_final"; run_type_suffix = output_filename_suffix if output_filename_suffix else "_unknown_mode"
        if stopped_early: run_type_suffix += "_stopped"
        elif self.processing_error: run_type_suffix += "_error"
        fits_path = os.path.join(self.output_folder, f"{base_name}{run_type_suffix}.fit"); preview_path  = os.path.splitext(fits_path)[0] + ".png"
        self.final_stacked_path = fits_path; self.update_progress(f"Chemin FITS final: {os.path.basename(fits_path)}")
        print(f"DEBUG QM [_save_final_stack V3]: Chemin FITS final sera: {fits_path}")

        # --- 6. Sauvegarde FITS (logique inchangée) ---
        try: # ... (code sauvegarde FITS identique)
            is_color_final_save = data_to_save.ndim == 3 and data_to_save.shape[2] == 3; data_for_primary_hdu_save = data_to_save.astype(np.float32) 
            if is_color_final_save: data_for_primary_hdu_save = np.moveaxis(data_for_primary_hdu_save, -1, 0)
            primary_hdu = fits.PrimaryHDU(data=data_for_primary_hdu_save, header=final_header); hdus_list = [primary_hdu]
            if self.photutils_bn_applied_in_session and background_model_photutils is not None and _PHOTOUTILS_BG_SUB_AVAILABLE:
                 bkg_hdu_data = None
                 if background_model_photutils.ndim == 3 and background_model_photutils.shape[2] == 3: bkg_hdu_data = np.mean(background_model_photutils, axis=2).astype(np.float32)
                 elif background_model_photutils.ndim == 2: bkg_hdu_data = background_model_photutils.astype(np.float32)
                 if bkg_hdu_data is not None: bkg_hdu = fits.ImageHDU(bkg_hdu_data, name="BACKGROUND_MODEL"); hdus_list.append(bkg_hdu); self.update_progress("   HDU modèle de fond Photutils incluse dans le FITS.")
            fits.HDUList(hdus_list).writeto(fits_path, overwrite=True, checksum=True, output_verify='ignore')
            self.update_progress("   ✅ Sauvegarde FITS terminée."); print(f"DEBUG QM [_save_final_stack V3]: Sauvegarde FITS de '{fits_path}' réussie.")
        except Exception as save_err: self.update_progress(f"   ❌ Erreur Sauvegarde FITS: {save_err}"); print(f"ERREUR QM: Erreur sauvegarde FITS: {save_err}"); traceback.print_exc(limit=1); self.final_stacked_path = None

        # --- 7. Sauvegarde preview PNG et stockage (logique inchangée) ---
        if data_to_save is not None: # ... (code sauvegarde PNG identique)
            try:
                save_preview_image(data_to_save, preview_path, apply_stretch=True, enhanced_stretch=True)
                self.update_progress("   ✅ Sauvegarde Preview PNG terminée."); print(f"DEBUG QM [_save_final_stack V3]: Sauvegarde PNG de '{preview_path}' réussie.")
                self.last_saved_data_for_preview = data_to_save.copy(); print("DEBUG QM [_save_final_stack V3]: 'last_saved_data_for_preview' mis à jour.")
                if self.final_stacked_path and os.path.exists(self.final_stacked_path): self.update_progress(f"🎉 Stack final sauvegardé ({effective_image_count} images/poids). Traitement complet.")
                elif os.path.exists(preview_path): self.update_progress(f"⚠️ Traitement terminé. Stack FITS échec, mais prévisualisation PNG sauvegardée.")
            except Exception as prev_err: self.update_progress(f"   ❌ Erreur Sauvegarde Preview PNG: {prev_err}."); print(f"ERREUR QM: Erreur sauvegarde PNG: {prev_err}"); traceback.print_exc(limit=1); self.last_saved_data_for_preview = None 
        else: self.update_progress("ⓘ Aucune image à sauvegarder."); self.last_saved_data_for_preview = None

        # --- 7. Sauvegarde preview PNG et stockage (logique inchangée) ---
        if data_to_save is not None:  # ... (code sauvegarde PNG identique)
            try:
                save_preview_image(data_to_save, preview_path, apply_stretch=True, enhanced_stretch=True)
                self.update_progress("     ✅ Sauvegarde Preview PNG terminée.")
                print(f"DEBUG QM [_save_final_stack V3]: Sauvegarde PNG de '{preview_path}' réussie.")
                self.last_saved_data_for_preview = data_to_save.copy()
                print("DEBUG QM [_save_final_stack V3]: 'last_saved_data_for_preview' mis à jour.")
                if self.final_stacked_path and os.path.exists(self.final_stacked_path):
                    self.update_progress(f"🎉 Stack final sauvegardé ({effective_image_count} images/poids). Traitement complet.")
                elif os.path.exists(preview_path):
                    self.update_progress(f"⚠️ Traitement terminé. Stack FITS échec, mais prévisualisation PNG sauvegardée.")
            except Exception as prev_err:
                self.update_progress(f"     ❌ Erreur Sauvegarde Preview PNG: {prev_err}.")
                print(f"ERREUR QM: Erreur sauvegarde PNG: {prev_err}")
                traceback.print_exc(limit=1)
                self.last_saved_data_for_preview = None
        else:
            self.update_progress("ⓘ Aucune image à sauvegarder.")
            self.last_saved_data_for_preview = None

        # --- Nettoyage des fichiers memmap physiques SI mode classique (logique inchangée) ---
        if not (is_any_drizzle_mode_with_data or is_mosaic_mode_with_data):  # Note: is_any_drizzle_mode_with_data utilisé ici
            if self.perform_cleanup:
                print("DEBUG QM [_save_final_stack V3]: Nettoyage des fichiers memmap (mode SUM/W classique)...")
                if self.sum_memmap_path and os.path.exists(self.sum_memmap_path):
                    try:
                        os.remove(self.sum_memmap_path)
                        print("     -> Fichier SUM.npy supprimé.")
                    except Exception as e:
                        print(f"     -> WARN: Erreur suppression SUM.npy: {e}")
                if self.wht_memmap_path and os.path.exists(self.wht_memmap_path):
                    try:
                        os.remove(self.wht_memmap_path)
                        print("     -> Fichier WHT.npy supprimé.")
                    except Exception as e:
                        print(f"     -> WARN: Erreur suppression WHT.npy: {e}")
                try:
                    memmap_dir = os.path.join(self.output_folder, "memmap_accumulators")
                    if os.path.isdir(memmap_dir) and not os.listdir(memmap_dir):
                        os.rmdir(memmap_dir)
                        print(f"     -> Dossier memmap vide supprimé: {memmap_dir}")
                except Exception:
                    pass  # Erreur silencieuse pour la suppression du dossier, peut-être à revoir
        else:
            print("DEBUG QM [_save_final_stack V3]: Nettoyage fichiers memmap ignoré (mode Drizzle / Mosaïque).")

        print("\n" + "=" * 80)
        print("DEBUG QM [_save_final_stack V3]: Fin méthode.")
        print("=" * 80 + "\n")







#############################################################################################################################################################


#Le message de Pylance "is not accessed" concerne uniquement les variables locales closed_sum et closed_wht à l'intérieur 
# de la méthode _close_memmaps() elle-même. Ces variables sont définies, mais leur valeur n'est jamais lue par le code de cette méthode 
# après leur assignation. Elles sont donc inutiles et peuvent être supprimées.
#Mais cela ne remet absolument pas en question :
#Le fait que la méthode _close_memmaps() est appelée.
#Le fait que le code à l'intérieur de cette méthode (fermeture et suppression des références self.cumulative_sum_memmap 
# et self.cumulative_wht_memmap) s'exécute quand la méthode est appelée.
#L'utilité de cette méthode pour libérer les ressources liées aux fichiers memmap.

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
        if self.output_folder is None: # <--- AJOUTER CETTE VÉRIFICATION
            print("WARN QM [cleanup_temp_reference]: self.output_folder non défini, nettoyage référence annulé.")
            return
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




# --- DANS LA CLASSE SeestarQueuedStacker DANS seestar/queuep/queue_manager.py ---

    def start_processing(self, input_dir, output_dir, reference_path_ui=None,
                         initial_additional_folders=None,
                         stacking_mode="kappa-sigma", kappa=2.5,
                         batch_size=10, correct_hot_pixels=True, hot_pixel_threshold=3.0,
                         neighborhood_size=5, bayer_pattern="GRBG", perform_cleanup=True,
                         use_weighting=False, 
                         weight_by_snr=True, 
                         weight_by_stars=True,
                         snr_exp=1.0, 
                         stars_exp=0.5, 
                         min_w=0.1,
                         use_drizzle=False, drizzle_scale=2.0, drizzle_wht_threshold=0.7,
                         drizzle_mode="Final", drizzle_kernel="square", drizzle_pixfrac=1.0,
                         apply_chroma_correction=True,
                         apply_final_scnr=False, final_scnr_target_channel='green',
                         final_scnr_amount=0.8, final_scnr_preserve_luminosity=True,
                         bn_grid_size_str="16x16", bn_perc_low=5, bn_perc_high=30,
                         bn_std_factor=1.0, bn_min_gain=0.2, bn_max_gain=7.0,
                         cb_border_size=25, cb_blur_radius=8,
                         cb_min_b_factor=0.4, cb_max_b_factor=1.5,
                         final_edge_crop_percent=2.0,
                         apply_photutils_bn=False,
                         photutils_bn_box_size=128,
                         photutils_bn_filter_size=5,
                         photutils_bn_sigma_clip=3.0,
                         photutils_bn_exclude_percentile=98.0,
                         apply_feathering=False,
                         feather_blur_px=256,
                         # --- NOUVEAU : Accepter les paramètres Low WHT Mask ---
                         apply_low_wht_mask=False, # Valeur par défaut si non passée
                         low_wht_percentile=5,     # Valeur par défaut
                         low_wht_soften_px=128,    # Valeur par défaut
                         # --- FIN NOUVEAU ---
                         is_mosaic_run=False, api_key=None, mosaic_settings=None):
        """
        Démarre le thread de traitement principal avec la configuration spécifiée.
        Le bloc de forçage des paramètres de test est maintenant COMMENTÉ.
        """
        print("DEBUG (Backend start_processing SUM/W): Début tentative démarrage...")
        
        # ---- LOG DES ARGUMENTS REÇUS PAR LE BACKEND (CE QUE LE GUI A ENVOYÉ) ----
        print("  --- BACKEND ARGS REÇUS (ORIGINAL DEPUIS GUI/SETTINGS) ---")
        print(f"    input_dir='{input_dir}'")
        # ... (gardez les autres logs des arguments reçus si vous le souhaitez) ...
        print(f"    apply_feathering={apply_feathering}")
        print(f"    feather_blur_px={feather_blur_px}")
        print(f"    apply_low_wht_mask={apply_low_wht_mask}") # <-- NOUVEAU LOG
        print(f"    low_wht_percentile={low_wht_percentile}") # <-- NOUVEAU LOG
        print(f"    low_wht_soften_px={low_wht_soften_px}")   # <-- NOUVEAU LOG
        print(f"    photutils_bn_filter_size={photutils_bn_filter_size}")
        print(f"    bn_grid_size_str='{bn_grid_size_str}'")
        print(f"    final_scnr_amount={final_scnr_amount}")
        print(f"    use_weighting={use_weighting}")
        print(f"  --- FIN BACKEND ARGS REÇUS ---")

        # ----- !!!!! BLOC DE FORÇAGE TEMPORAIRE MAINTENANT COMMENTÉ !!!!! -----
        # # print("!!! ATTENTION : FORÇAGE DES VALEURS DE TEST DANS SeestarQueuedStacker.start_processing !!!")
        # # 
        # # # Paramètres Photutils BN de test
        # # apply_photutils_bn_test = False 
        # # photutils_bn_filter_size_test = 11
        # # photutils_bn_exclude_percentile_test = 95.0
        # # print(f"  FORÇAGE TEST: apply_photutils_bn à {apply_photutils_bn_test}")
        # # # ... (autres logs de forçage)
        # #
        # # # Paramètres BN Globale de test
        # # bn_grid_size_str_test = "24x24"
        # # # ...
        # #
        # # # Paramètres SCNR Final de test
        # # apply_final_scnr_test = True
        # # # ...
        # #
        # # # Paramètres Feathering de test
        # # apply_feathering_test = True 
        # # feather_blur_px_test = 128   
        # # # ...
        # #
        # # # Pondération Qualité
        # # use_weighting_test = use_weighting 
        # #
        # # # Appliquer les valeurs de test aux variables locales qui seront utilisées pour configurer 'self'
        # # apply_photutils_bn = apply_photutils_bn_test
        # # photutils_bn_filter_size = photutils_bn_filter_size_test
        # # photutils_bn_exclude_percentile = photutils_bn_exclude_percentile_test
        # # bn_grid_size_str = bn_grid_size_str_test
        # # bn_perc_high = bn_perc_high_test
        # # bn_std_factor = bn_std_factor_test
        # # apply_final_scnr = apply_final_scnr_test
        # # final_scnr_amount = final_scnr_amount_test
        # # final_scnr_preserve_luminosity = final_scnr_preserve_luminosity_test
        # # apply_feathering = apply_feathering_test
        # # feather_blur_px = feather_blur_px_test
        # # use_weighting = use_weighting_test
        # ----- !!!!! FIN FORÇAGE TEMPORAIRE !!!!! -----

        if self.processing_active:
            self.update_progress("⚠️ Tentative de démarrer un traitement déjà en cours.")
            return False

        self.stop_processing = False
        if hasattr(self, 'aligner') and self.aligner is not None:
            self.aligner.stop_processing = False
            print("DEBUG (Backend start_processing SUM/W): self.aligner.stop_processing remis à False.")
        else:
            print("ERREUR (Backend start_processing SUM/W): self.aligner non initialisé.")
            self.update_progress("❌ Erreur interne critique: Aligner non initialisé.")
            return False

        self.current_folder = os.path.abspath(input_dir)
        
        print("DEBUG (Backend start_processing SUM/W): Étape 2 - Préparation référence & shape...")
        reference_image_data_for_shape = None 
        reference_header_for_shape = None 
        ref_shape_hwc = None
        try:
            # ... (Logique de préparation de la référence et obtention de ref_shape_hwc - INCHANGÉE) ...
            potential_folders_for_shape = []
            if self.current_folder and os.path.isdir(self.current_folder): potential_folders_for_shape.append(self.current_folder)
            if initial_additional_folders:
                for add_f in initial_additional_folders:
                    abs_add_f = os.path.abspath(add_f)
                    if abs_add_f and os.path.isdir(abs_add_f) and abs_add_f not in potential_folders_for_shape: potential_folders_for_shape.append(abs_add_f)
            if not potential_folders_for_shape: raise RuntimeError("Aucun dossier valide pour shape.")
            current_folder_to_scan_for_shape = None; files_in_folder_for_shape = []
            for folder_path_iter in potential_folders_for_shape:
                temp_files = sorted([f for f in os.listdir(folder_path_iter) if f.lower().endswith(('.fit', '.fits'))])
                if temp_files: files_in_folder_for_shape = temp_files; current_folder_to_scan_for_shape = folder_path_iter; break
            if not current_folder_to_scan_for_shape or not files_in_folder_for_shape: raise RuntimeError("Aucun FITS pour shape.")
            self.aligner.correct_hot_pixels = correct_hot_pixels
            self.aligner.hot_pixel_threshold = hot_pixel_threshold
            self.aligner.neighborhood_size = neighborhood_size
            self.aligner.bayer_pattern = bayer_pattern
            self.aligner.reference_image_path = reference_path_ui or None
            reference_image_data_for_shape, reference_header_for_shape = self.aligner._get_reference_image(current_folder_to_scan_for_shape, files_in_folder_for_shape)
            if reference_image_data_for_shape is None or reference_header_for_shape is None: raise RuntimeError("Échec _get_reference_image pour shape.")
            ref_shape_initial = reference_image_data_for_shape.shape
            if len(ref_shape_initial) == 2: ref_shape_hwc = (ref_shape_initial[0], ref_shape_initial[1], 3)
            elif len(ref_shape_initial) == 3 and ref_shape_initial[2] == 3: ref_shape_hwc = ref_shape_initial
            else: raise RuntimeError(f"Shape référence non supportée: {ref_shape_initial}")
            self.reference_header_for_wcs = reference_header_for_shape.copy()
            del reference_image_data_for_shape, reference_header_for_shape; gc.collect()
        except Exception as e_ref_shape:
            self.update_progress(f"❌ Erreur préparation référence/shape: {e_ref_shape}")
            print(f"ERREUR QM [start_processing SUM/W]: Échec préparation référence/shape : {e_ref_shape}"); traceback.print_exc(limit=2)
            return False

        print(f"DEBUG (Backend start_processing SUM/W): Appel à self.initialize() avec shape={ref_shape_hwc}...")
        if not self.initialize(output_dir, ref_shape_hwc):
            self.processing_active = False
            print("ERREUR (Backend start_processing SUM/W): Échec de self.initialize() pour SUM/W.")
            return False
        print("DEBUG (Backend start_processing SUM/W): self.initialize() terminé avec succès.")

        print("DEBUG (Backend start_processing SUM/W): Configuration des paramètres de session (maintenant depuis les args GUI)...")
        # --- Stockage des paramètres reçus en argument (maintenant sans le bloc de forçage) ---
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
        
        self.use_quality_weighting = use_weighting 
        self.weight_by_snr = weight_by_snr
        self.weight_by_stars = weight_by_stars
        self.snr_exponent = snr_exp
        self.stars_exponent = stars_exp
        self.min_weight = max(0.01, min(1.0, min_w))
        print(f"  BACKEND STOCKÉ: self.use_quality_weighting={self.use_quality_weighting}")
        if self.use_quality_weighting:
            print(f"    -> Pondération par SNR: {self.weight_by_snr}, Exp: {self.snr_exponent}") #... etc
        
        self.apply_final_scnr = apply_final_scnr 
        self.final_scnr_target_channel = final_scnr_target_channel
        self.final_scnr_amount = final_scnr_amount 
        self.final_scnr_preserve_luminosity = final_scnr_preserve_luminosity
        print(f"  BACKEND STOCKÉ: self.apply_final_scnr={self.apply_final_scnr}, Amount={self.final_scnr_amount}")
        
        if self.drizzle_active_session: # ... (logique inchangée)
            if self.is_mosaic_run: # ...
                current_mosaic_settings = mosaic_settings if isinstance(mosaic_settings, dict) else {}
                self.drizzle_kernel = current_mosaic_settings.get('kernel', drizzle_kernel) # ...
            else: # ...
                 self.drizzle_kernel = drizzle_kernel # ...
            self.drizzle_mode = drizzle_mode if drizzle_mode in ["Final", "Incremental"] else "Final" # ...
            self.drizzle_scale = float(drizzle_scale) # ...
            self.drizzle_wht_threshold = max(0.01, min(1.0, float(drizzle_wht_threshold))) # ...
            print(f"   -> Params Drizzle Actifs -> Mode: {self.drizzle_mode}, Scale: {self.drizzle_scale:.1f}, WHT: {self.drizzle_wht_threshold:.2f}, Kernel: {self.drizzle_kernel}, Pixfrac: {self.drizzle_pixfrac:.2f}")
        else: print("DEBUG (Backend start_processing SUM/W): Session Drizzle non active.")

        print("DEBUG (Backend start_processing SUM/W): Stockage des paramètres Expert...")
        self.bn_grid_size_str = bn_grid_size_str
        self.bn_perc_low = bn_perc_low
        self.bn_perc_high = bn_perc_high
        self.bn_std_factor = bn_std_factor
        self.bn_min_gain = bn_min_gain
        self.bn_max_gain = bn_max_gain
        self.cb_border_size = cb_border_size
        self.cb_blur_radius = cb_blur_radius
        self.cb_min_b_factor = cb_min_b_factor
        self.cb_max_b_factor = cb_max_b_factor
        self.final_edge_crop_percent_decimal = float(final_edge_crop_percent) / 100.0
        print(f"  BACKEND STOCKÉ: self.bn_grid_size_str='{self.bn_grid_size_str}', self.bn_perc_high={self.bn_perc_high}, self.bn_std_factor={self.bn_std_factor}")
        
        print("DEBUG (Backend start_processing SUM/W): Stockage des paramètres Photutils BN...")
        self.apply_photutils_bn = apply_photutils_bn
        self.photutils_bn_box_size = photutils_bn_box_size
        self.photutils_bn_filter_size = photutils_bn_filter_size
        self.photutils_bn_sigma_clip = photutils_bn_sigma_clip
        self.photutils_bn_exclude_percentile = photutils_bn_exclude_percentile
        print(f"  BACKEND STOCKÉ: self.apply_photutils_bn={self.apply_photutils_bn}")
        print(f"  BACKEND STOCKÉ: self.photutils_bn_filter_size={self.photutils_bn_filter_size}")
        
        self.apply_feathering = apply_feathering 
        self.feather_blur_px = feather_blur_px   
        print(f"  BACKEND STOCKÉ (valeur reçue): self.apply_feathering={self.apply_feathering}")
        print(f"  BACKEND STOCKÉ (valeur reçue): self.feather_blur_px={self.feather_blur_px}")
        # --- Stockage des paramètres Low WHT Mask ---
        self.apply_low_wht_mask = apply_low_wht_mask
        self.low_wht_percentile = low_wht_percentile
        self.low_wht_soften_px = low_wht_soften_px
        print(f"  BACKEND STOCKÉ: self.apply_low_wht_mask={self.apply_low_wht_mask}") # <-- NOUVEAU LOG
        print(f"  BACKEND STOCKÉ: self.low_wht_percentile={self.low_wht_percentile}") # <-- NOUVEAU LOG
        print(f"  BACKEND STOCKÉ: self.low_wht_soften_px={self.low_wht_soften_px}")   # <-- NOUVEAU LOG
        # --- ---
        requested_batch_size = batch_size # ... (logique estimation batch_size identique) ...
        if requested_batch_size <= 0: # ...
             sample_img_path = None # ...
             if input_dir and os.path.isdir(input_dir): fits_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.fit', '.fits'))]; sample_img_path = os.path.join(input_dir, fits_files[0]) if fits_files else None # ...
             try: estimated_size = estimate_batch_size(sample_image_path=sample_img_path); self.batch_size = estimated_size; self.update_progress(f"✅ Taille lot auto estimée: {estimated_size}", None) # ...
             except Exception as est_err: self.update_progress(f"⚠️ Erreur estimation taille lot: {est_err}. Utilisation défaut (10).", None); self.batch_size = 10 # ...
        else: self.batch_size = requested_batch_size # ...
        if self.batch_size < 3: self.update_progress(f"⚠️ Taille de lot ({self.batch_size}) trop petite, ajustée à 3.", None); self.batch_size = 3 # ...
        self.update_progress(f"ⓘ Taille de lot effective pour le traitement : {self.batch_size}") # ...
        if self.apply_final_scnr: self.update_progress(f"🎨 SCNR Final (Cible: {self.final_scnr_target_channel}, {self.final_scnr_amount*100:.0f}%) sera appliqué.") # ...
        if self.apply_feathering: self.update_progress(f"🖌️ Feathering (Flou: {self.feather_blur_px}px) sera appliqué.")
        if self.use_quality_weighting: self.update_progress(f"⚖️ Pondération Qualité Activée (SNR^{self.snr_exponent:.1f}, Stars^{self.stars_exponent:.1f}, MinW={self.min_weight:.2f}).") # ...
        
        initial_folders_to_add_count = 0 # ... (logique gestion folders identique)
        with self.folders_lock: # ...
            self.additional_folders = [] # ...
            if initial_additional_folders: # ...
                for folder_iter in initial_additional_folders: # ...
                    abs_folder = os.path.abspath(folder_iter) # ...
                    if os.path.isdir(abs_folder) and abs_folder not in self.additional_folders: # ...
                        self.additional_folders.append(abs_folder); initial_folders_to_add_count += 1 # ...
        if initial_folders_to_add_count > 0: self.update_progress(f"ⓘ {initial_folders_to_add_count} dossier(s) pré-ajouté(s) en attente."); self.update_progress(f"folder_count_update:{len(self.additional_folders)}") # ...

        initial_files_added = self._add_files_to_queue(self.current_folder) # ...
        if initial_files_added > 0: self._recalculate_total_batches(); self.update_progress(f"📋 {initial_files_added} fichiers initiaux ajoutés. Total lots estimé: {self.total_batches_estimated if self.total_batches_estimated > 0 else '?'}") # ...
        elif not self.additional_folders: self.update_progress("⚠️ Aucun fichier initial trouvé ou dossier supplémentaire en attente.") # ...
        
        self.aligner.reference_image_path = reference_path_ui or None # ...

        print("DEBUG (Backend start_processing SUM/W): Démarrage du thread worker...")
        self.processing_thread = threading.Thread(target=self._worker, name="StackerWorker"); self.processing_thread.daemon = True
        self.processing_thread.start(); self.processing_active = True
        self.update_progress("🚀 Thread de traitement démarré.")
        print("DEBUG (Backend start_processing SUM/W): Fin.")
        return True





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
        if self.drizzle_temp_dir is None: # self.drizzle_temp_dir dépend de self.output_folder
            if self.output_folder is None:
                print("WARN QM [_cleanup_drizzle_temp_files]: self.output_folder non défini, nettoyage Drizzle temp annulé.")
                return
        else:
            self.drizzle_temp_dir = os.path.join(self.output_folder, "drizzle_temp_inputs")
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




# --- DANS LA CLASSE SeestarQueuedStacker DANS seestar/queuep/queue_manager.py ---

    def _process_and_save_drizzle_batch(self, batch_data_list, output_wcs, output_shape_2d_hw, batch_num):
        """
        Traite un lot de données alignées en mémoire en utilisant Drizzle et sauvegarde
        les fichiers science (CxHxW) et poids (HxW x3) intermédiaires pour ce lot.
        CORRIGÉ: Initialisation de Drizzle() sans out_wcs/out_shape car non supporté par stsci.drizzle.

        Args:
            batch_data_list (list): Liste de tuples: [(aligned_data_HxWxC, header, wcs_object), ...].
                                    wcs_object doit être le WCS de référence pour toutes (celui de l'image alignée).
            output_wcs (astropy.wcs.WCS): WCS de la grille de SORTIE Drizzle.
            output_shape_2d_hw (tuple): Shape (H, W) de la grille de SORTIE Drizzle.
            batch_num (int): Numéro du lot actuel pour nommage des fichiers.

        Returns:
            tuple: (sci_filepath, [wht_r_filepath, wht_g_filepath, wht_b_filepath])
                   Chemins des fichiers intermédiaires créés pour ce lot, ou (None, []) si échec.
        """
        num_files_in_batch = len(batch_data_list)
        self.update_progress(f"💧 Traitement Drizzle du lot #{batch_num} ({num_files_in_batch} images)...")
        batch_start_time = time.time()
        print(f"DEBUG QM [_process_and_save_drizzle_batch V2_CORRECTED]: Lot #{batch_num} avec {num_files_in_batch} images.")
        print(f"  -> WCS de sortie cible fourni: {'Oui' if output_wcs else 'Non'}, Shape de sortie cible: {output_shape_2d_hw}")

        if not batch_data_list:
            self.update_progress(f"   - Warning: Lot Drizzle #{batch_num} vide.")
            return None, []

        # --- Vérifier la validité de output_wcs et output_shape_2d_hw (essentiels) ---
        if output_wcs is None or output_shape_2d_hw is None:
            self.update_progress(f"   - ERREUR: WCS ou Shape de sortie manquant pour lot Drizzle #{batch_num}. Traitement annulé.")
            print(f"ERREUR QM [_process_and_save_drizzle_batch V2_CORRECTED]: output_wcs ou output_shape_2d_hw est None pour lot #{batch_num}.")
            return None, []
        if not isinstance(output_wcs, WCS) or not output_wcs.is_celestial:
            self.update_progress(f"   - ERREUR: output_wcs invalide (non WCS ou non céleste) pour lot Drizzle #{batch_num}.")
            print(f"ERREUR QM [_process_and_save_drizzle_batch V2_CORRECTED]: output_wcs invalide pour lot #{batch_num}.")
            return None, []
        if not isinstance(output_shape_2d_hw, tuple) or len(output_shape_2d_hw) != 2 or \
           not all(isinstance(dim, int) and dim > 0 for dim in output_shape_2d_hw):
            self.update_progress(f"   - ERREUR: output_shape_2d_hw invalide (doit être tuple de 2 entiers > 0) pour lot Drizzle #{batch_num}.")
            print(f"ERREUR QM [_process_and_save_drizzle_batch V2_CORRECTED]: output_shape_2d_hw ({output_shape_2d_hw}) invalide pour lot #{batch_num}.")
            return None, []

        # --- Vérification cohérence WCS et Shape Entrée (sécurité) ---
        ref_wcs_for_batch_input_images = None # WCS des images d'ENTRÉE de ce lot (devrait être le même pour toutes)
        ref_input_shape_hw = None             # Shape des images d'ENTRÉE
        valid_batch_items = []

        for i, item_tuple in enumerate(batch_data_list):
            if not (isinstance(item_tuple, tuple) and len(item_tuple) >= 3):
                self.update_progress(f"   - Warning: Format d'item incorrect pour image {i+1} du lot {batch_num}. Ignorée.")
                continue
            
            img_data, hdr, wcs_obj_input = item_tuple[0], item_tuple[1], item_tuple[2]

            if img_data is None or wcs_obj_input is None:
                self.update_progress(f"   - Warning: Donnée/WCS manquant pour image {i+1} du lot {batch_num}. Ignorée.")
                continue
            if not isinstance(wcs_obj_input, WCS) or not wcs_obj_input.is_celestial:
                self.update_progress(f"   - Warning: WCS d'entrée invalide pour image {i+1} du lot {batch_num}. Ignorée.")
                continue
            
            current_shape_hw = img_data.shape[:2]
            if ref_wcs_for_batch_input_images is None: # Première image valide du lot
                ref_wcs_for_batch_input_images = wcs_obj_input
                ref_input_shape_hw = current_shape_hw
            # Pour Drizzle, les WCS d'entrée PEUVENT être différents (images de panneaux différents)
            # MAIS pour un lot Drizzle "Final" (non-mosaïque), ils devraient tous partager le WCS de référence global.
            # Si cette fonction est appelée pour un panneau de mosaïque, alors ref_wcs_for_batch_input_images sera le WCS de ce panneau.
            # La vérification wcs_obj is not ref_wcs_for_batch_input_images n'est pertinente que si on attend un WCS unique.
            # Pour la robustesse, on ne fait pas cette vérification ici, on se fie au pixmap.
            
            if current_shape_hw != ref_input_shape_hw:
                 self.update_progress(f"   - Warning: Shape d'entrée ({current_shape_hw}) incohérente avec réf. du lot ({ref_input_shape_hw}) pour image {i+1}. Ignorée.")
                 continue
            valid_batch_items.append((img_data, hdr, wcs_obj_input)) # Garder le WCS d'entrée individuel

        if not valid_batch_items:
            self.update_progress(f"   - Erreur: Aucune donnée valide trouvée dans le lot Drizzle #{batch_num}.")
            return None, []
        num_valid_images = len(valid_batch_items)
        self.update_progress(f"   - {num_valid_images}/{num_files_in_batch} images valides pour Drizzle dans le lot.")

        # --- Initialiser les objets Drizzle pour ce lot ---
        num_output_channels = 3
        channel_names = ['R', 'G', 'B']
        drizzlers_batch = []
        output_images_batch = []  # Stockera les résultats science (counts/s) du lot (HxW) par canal
        output_weights_batch = [] # Stockera les résultats poids (context/exposure) du lot (HxW) par canal
        
        try:
            print(f"DEBUG QM [_process_and_save_drizzle_batch V2_CORRECTED]: Initialisation Drizzle pour lot #{batch_num}. Shape Sortie CIBLE: {output_shape_2d_hw}.")
            for _ in range(num_output_channels):
                # Les tableaux NumPy sont créés avec la SHAPE DE SORTIE attendue
                output_images_batch.append(np.zeros(output_shape_2d_hw, dtype=np.float32))
                output_weights_batch.append(np.zeros(output_shape_2d_hw, dtype=np.float32))
            
            for i in range(num_output_channels):
                # =================== CORRECTION APPLIQUÉE ICI ===================
                driz_ch = Drizzle(
                    out_img=output_images_batch[i],   # Tableau NumPy (H,W) avec la shape de SORTIE
                    out_wht=output_weights_batch[i],   # Tableau NumPy (H,W) avec la shape de SORTIE
                    kernel=self.drizzle_kernel,
                    fillval="0.0"
                    # PAS DE out_wcs ni out_shape ici pour stsci.drizzle.resample.Drizzle __init__
                )
                # ==================================================================
                drizzlers_batch.append(driz_ch)
            self.update_progress(f"   - Objets Drizzle initialisés pour lot #{batch_num} (sans out_wcs/shape dans init).")

        except Exception as init_err:
            self.update_progress(f"   - ERREUR: Échec init Drizzle pour lot #{batch_num}: {init_err}")
            print(f"ERREUR QM [_process_and_save_drizzle_batch V2_CORRECTED]: Échec init Drizzle: {init_err}"); traceback.print_exc(limit=1)
            return None, []


        # --- Boucle sur les images VALIDES du lot ---
        processed_in_batch_count = 0
        for i, (input_data_hxwx3, input_header, wcs_input_image) in enumerate(valid_batch_items): # Utiliser wcs_input_image
            if self.stop_processing: self.update_progress("🛑 Arrêt pendant traitement lot Drizzle."); break
            # Nom de fichier pour les logs
            current_filename_for_log = input_header.get('FILENAME', f'Img_{i+1}_du_lot') if input_header else f'Img_{i+1}_du_lot'
            print(f"DEBUG QM [_process_and_save_drizzle_batch V2_CORRECTED]: Traitement image {i+1}/{num_valid_images} ('{current_filename_for_log}') du lot #{batch_num}...")

            pixmap = None
            try:
                current_input_shape_hw = input_data_hxwx3.shape[:2]
                y_in, x_in = np.indices(current_input_shape_hw)
                
                # Utiliser le WCS de l'image d'ENTRÉE pour convertir vers le ciel
                print(f"          Pour '{current_filename_for_log}': WCS Entrée CRVAL=({wcs_input_image.wcs.crval[0]:.4f}, {wcs_input_image.wcs.crval[1]:.4f}), PixelShape={wcs_input_image.pixel_shape}") # LOG WCS Entrée
                world_coords_ra, world_coords_dec = wcs_input_image.all_pix2world(x_in.flatten(), y_in.flatten(), 0)
                print(f"          Pour '{current_filename_for_log}': Pixels Entrée -> Ciel OK. Nb points: {world_coords_ra.size}")
                
                # Projeter depuis le ciel vers les pixels de la grille de SORTIE Drizzle
                # output_wcs est le WCS de la grille Drizzle cible (grand format)
                print(f"          Pour '{current_filename_for_log}': WCS Sortie (cible Drizzle) CRVAL=({output_wcs.wcs.crval[0]:.4f}, {output_wcs.wcs.crval[1]:.4f}), PixelShape={output_wcs.pixel_shape}, OutputShapeHW={output_shape_2d_hw}") # LOG WCS Sortie
                x_out, y_out = output_wcs.all_world2pix(world_coords_ra, world_coords_dec, 0)
                print(f"          Pour '{current_filename_for_log}': Ciel -> Pixels Sortie OK.")

                pixmap = np.dstack((x_out.reshape(current_input_shape_hw), y_out.reshape(current_input_shape_hw))).astype(np.float32)
                
                # ===== AJOUT DE LOGS POUR PIXMAP (identique à ma proposition précédente) =====
                print(f"        - Pixmap calculé pour '{current_filename_for_log}'. Shape: {pixmap.shape}")
                if pixmap.size > 0: 
                    finite_x_out = pixmap[...,0][np.isfinite(pixmap[...,0])]
                    finite_y_out = pixmap[...,1][np.isfinite(pixmap[...,1])]
                    if finite_x_out.size > 0 :
                        print(f"          Range X_out (valides): [{np.min(finite_x_out):.1f}, {np.max(finite_x_out):.1f}] (Shape Sortie W: {output_shape_2d_hw[1]})")
                    else:
                        print(f"          Range X_out (valides): Aucun pixel X valide après filtrage NaN/Inf.")
                    if finite_y_out.size > 0:
                        print(f"          Range Y_out (valides): [{np.min(finite_y_out):.1f}, {np.max(finite_y_out):.1f}] (Shape Sortie H: {output_shape_2d_hw[0]})")
                    else:
                        print(f"          Range Y_out (valides): Aucun pixel Y valide après filtrage NaN/Inf.")
                    if np.any(~np.isfinite(pixmap[...,0])): print(f"          WARNING: Pixmap X pour '{current_filename_for_log}' contient des non-finis !")
                    if np.any(~np.isfinite(pixmap[...,1])): print(f"          WARNING: Pixmap Y pour '{current_filename_for_log}' contient des non-finis !")
                else:
                    print(f"          WARNING: Pixmap pour '{current_filename_for_log}' est vide !")
                # ================================================================================

            except Exception as map_err:
                self.update_progress(f"      -> ERREUR création pixmap image {i+1} ('{current_filename_for_log}'): {map_err}. Ignorée.")
                print(f"ERREUR QM [_process_and_save_drizzle_batch V2_CORRECTED]: Échec pixmap img {i+1} ('{current_filename_for_log}'): {map_err}")
                traceback.print_exc(limit=1) # Ajout du traceback pour l'erreur de pixmap
                continue # Passer à l'image suivante du lot

            if pixmap is not None: # Ce check est important, si pixmap a échoué, on ne continue pas
                try:
                    base_exptime = 1.0
                    if input_header and 'EXPTIME' in input_header:
                        try: base_exptime = max(1e-6, float(input_header['EXPTIME']))
                        except (ValueError, TypeError): pass
                    
                    print(f"        - Appel add_image pour les 3 canaux de '{current_filename_for_log}'...") # Log avant add_image
                    for ch_index in range(num_output_channels):
                        channel_data_2d = input_data_hxwx3[..., ch_index].astype(np.float32)
                        finite_mask = np.isfinite(channel_data_2d)
                        if not np.all(finite_mask): channel_data_2d[~finite_mask] = 0.0
                        
                        # ===== LOG AVANT CHAQUE ADD_IMAGE (optionnel, mais peut être utile si ça plante ici) =====
                        # print(f"          Canal {ch_index}: data range [{np.min(channel_data_2d):.3f}, {np.max(channel_data_2d):.3f}], exptime={base_exptime:.2f}, pixfrac={self.drizzle_pixfrac}")
                        # =====================================================================================
                        
                        drizzlers_batch[ch_index].add_image(
                            data=channel_data_2d,
                            pixmap=pixmap,
                            exptime=base_exptime,
                            pixfrac=self.drizzle_pixfrac,
                            in_units='counts' 
                        )
                    processed_in_batch_count += 1
                    print(f"  DEBUG QM [_process_and_save_drizzle_batch V2_CORRECTED]: Image {i+1} ('{current_filename_for_log}') ajoutée au Drizzle du lot.")
                except Exception as drizzle_add_err:
                    self.update_progress(f"      -> ERREUR add_image {i+1} ('{current_filename_for_log}'): {drizzle_add_err}")
                    print(f"ERREUR QM [_process_and_save_drizzle_batch V2_CORRECTED]: Échec add_image {i+1} ('{current_filename_for_log}'): {drizzle_add_err}"); traceback.print_exc(limit=1)
                # Le 'finally' pour del pixmap, channel_data_2d est retiré ici pour être sûr
                # que pixmap n'est pas supprimé avant d'être utilisé par tous les canaux.
                # Il sera nettoyé à la fin de l'itération de la boucle principale for.

            # Nettoyage pour cette itération de la boucle principale for
            if pixmap is not None: del pixmap # Supprimer pixmap s'il a été créé
            # channel_data_2d est déjà dans une portée plus limitée, mais on peut être explicite si on veut
            # if 'channel_data_2d' in locals(): del channel_data_2d
            # gc.collect() n'est pas nécessaire à chaque image, peut ralentir. Mettre à la fin du lot.

        # Fin de la boucle `for i, (input_data_hxwx3, input_header, wcs_input_image) in enumerate(valid_batch_items):`
        # gc.collect() peut être appelé ici, après que toutes les images du lot ont été traitées.
        gc.collect() 
        # ----- Le reste de la méthode _process_and_save_drizzle_batch continue ici -----
        # --- Sauvegarde des résultats intermédiaires de CE lot ---
        batch_output_dir = self.drizzle_batch_output_dir
        os.makedirs(batch_output_dir, exist_ok=True)

        base_out_filename = f"batch_{batch_num:04d}_s{self.drizzle_scale:.1f}p{self.drizzle_pixfrac:.1f}{self.drizzle_kernel}"
        out_filepath_sci = os.path.join(batch_output_dir, f"{base_out_filename}_sci.fits")
        out_filepaths_wht = []
        self.update_progress(f"   -> Sauvegarde résultats intermédiaires lot #{batch_num}...")
        print(f"DEBUG QM [_process_and_save_drizzle_batch V2_CORRECTED]: Sauvegarde vers {batch_output_dir}")

        try:
            # output_images_batch contient les données SCI drizzlées (counts/s ou normalisé par exptime) par canal
            final_sci_data_batch_hwc = np.stack(output_images_batch, axis=-1) 
            final_sci_data_to_save = np.moveaxis(final_sci_data_batch_hwc, -1, 0).astype(np.float32) # CxHxW

            final_header_sci = output_wcs.to_header(relax=True) 
            final_header_sci['NINPUTS'] = (processed_in_batch_count, f'Valid input images for Drizzle batch {batch_num}')
            final_header_sci['ISCALE'] = (self.drizzle_scale, 'Drizzle scale factor'); final_header_sci['PIXFRAC'] = (self.drizzle_pixfrac, 'Drizzle pixfrac')
            final_header_sci['KERNEL'] = (self.drizzle_kernel, 'Drizzle kernel'); final_header_sci['HISTORY'] = f'Drizzle Batch {batch_num} by SeestarStacker'
            final_header_sci['BUNIT'] = 'Counts/s' 
            final_header_sci['NAXIS'] = 3; final_header_sci['NAXIS1'] = final_sci_data_to_save.shape[2]
            final_header_sci['NAXIS2'] = final_sci_data_to_save.shape[1]; final_header_sci['NAXIS3'] = final_sci_data_to_save.shape[0]
            final_header_sci['CTYPE3'] = 'CHANNEL' 
            try: final_header_sci['CHNAME1'] = 'R'; final_header_sci['CHNAME2'] = 'G'; final_header_sci['CHNAME3'] = 'B'
            except Exception: pass

            fits.writeto(out_filepath_sci, final_sci_data_to_save, final_header_sci, overwrite=True, checksum=False, output_verify='ignore')
            self.update_progress(f"      -> Science lot sauvegardé: {os.path.basename(out_filepath_sci)}")
            print(f"DEBUG QM [_process_and_save_drizzle_batch V2_CORRECTED]: Fichier SCI lot sauvegardé: {out_filepath_sci}")
            del final_sci_data_batch_hwc, final_sci_data_to_save; gc.collect()

        except Exception as e:
            self.update_progress(f"   - ERREUR sauvegarde science lot #{batch_num}: {e}")
            print(f"ERREUR QM [_process_and_save_drizzle_batch V2_CORRECTED]: Échec sauvegarde SCI: {e}"); traceback.print_exc(limit=1)
            del drizzlers_batch, output_images_batch, output_weights_batch; gc.collect()
            return None, []

        # Sauvegarde Poids (HxW par canal)
        for i in range(num_output_channels):
            ch_name = channel_names[i]
            out_filepath_wht = os.path.join(batch_output_dir, f"{base_out_filename}_wht_{ch_name}.fits")
            out_filepaths_wht.append(out_filepath_wht)
            try:
                # output_weights_batch[i] contient la carte de poids HxW pour le canal i
                wht_data_to_save = output_weights_batch[i].astype(np.float32)

                wht_header = output_wcs.to_header(relax=True)
                for key in ['NAXIS3', 'CTYPE3', 'CRPIX3', 'CRVAL3', 'CDELT3', 'CUNIT3', 'PC3_1', 'PC3_2', 'PC3_3', 'PC1_3', 'PC2_3', 'CHNAME1', 'CHNAME2', 'CHNAME3']:
                    if key in wht_header: del wht_header[key]
                wht_header['NAXIS'] = 2; wht_header['NAXIS1'] = wht_data_to_save.shape[1] # W
                wht_header['NAXIS2'] = wht_data_to_save.shape[0] # H
                wht_header['HISTORY'] = f'Drizzle Weights ({ch_name}) for batch {batch_num}'; wht_header['NINPUTS'] = processed_in_batch_count
                wht_header['BUNIT'] = 'Weight'

                fits.writeto(out_filepath_wht, wht_data_to_save, wht_header, overwrite=True, checksum=False, output_verify='ignore')
                print(f"  DEBUG QM [_process_and_save_drizzle_batch V2_CORRECTED]: Fichier WHT lot ({ch_name}) sauvegardé: {out_filepath_wht}. Range WHT: [{np.min(wht_data_to_save):.2f}, {np.max(wht_data_to_save):.2f}]")
            except Exception as e:
                self.update_progress(f"   - ERREUR sauvegarde poids {ch_name} lot #{batch_num}: {e}")
                print(f"ERREUR QM [_process_and_save_drizzle_batch V2_CORRECTED]: Échec sauvegarde WHT {ch_name}: {e}"); traceback.print_exc(limit=1)
                if os.path.exists(out_filepath_sci):
                    try: os.remove(out_filepath_sci)
                    except Exception: pass
                for wht_f in out_filepaths_wht: # Nettoyer ceux déjà sauvegardés
                    if os.path.exists(wht_f):
                        try: os.remove(wht_f)
                        except Exception: pass
                del drizzlers_batch, output_images_batch, output_weights_batch; gc.collect()
                return None, []

        self.update_progress(f"   -> Sauvegarde lot #{batch_num} terminée.")
        del drizzlers_batch, output_images_batch, output_weights_batch; gc.collect()
        return out_filepath_sci, out_filepaths_wht





######################################################################################################################################################

