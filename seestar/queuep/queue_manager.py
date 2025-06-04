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
from reproject.mosaicking import reproject_and_coadd
from reproject import reproject_interp 
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
# from ..alignment.astrometry_solver import solve_image_wcs 
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
from ..alignment.astrometry_solver import AstrometrySolver # Déplacé vers _worker/_process_file
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
        print("\n==== DÉBUT INITIALISATION SeestarQueuedStacker (AVEC LocalAligner) ====") 
        
        # --- 1. Attributs Critiques et Simples ---
        print("  -> Initialisation attributs simples et flags...")
        
        
        self.processing_active = False; self.stop_processing = False; self.processing_error = None
        self.is_mosaic_run = False; self.drizzle_active_session = False 
        self.mosaic_alignment_mode = "local_fast_fallback" 
        self.use_wcs_fallback_for_mosaic = True      
        
        self.fa_orb_features = 5000
        self.fa_min_abs_matches = 12
        self.fa_min_ransac_raw = 7 
        self.fa_ransac_thresh = 5.0
        self.fa_daofind_fwhm = 3.5
        self.fa_daofind_thr_sig = 4.0
        self.fa_max_stars_descr = 750 
        
        self.mosaic_drizzle_kernel = "square"
        self.mosaic_drizzle_pixfrac = 0.8
        self.mosaic_drizzle_fillval = "0.0"
        self.mosaic_drizzle_wht_threshold = 0.01


        self.perform_cleanup = True; self.use_quality_weighting = True 
        self.correct_hot_pixels = True; self.apply_chroma_correction = True
        self.apply_final_scnr = False 

        #Info message pour l'utilisateur
        self.warned_unaligned_source_folders = set()
        
        # NOUVEAU : Initialisation de l'attribut pour la sauvegarde en float32
        self.save_final_as_float32 = False # Par défaut, sauvegarde en uint16 (via conversion dans _save_final_stack)
        print(f"  -> Attribut self.save_final_as_float32 initialisé à: {self.save_final_as_float32}")
        # --- FIN NOUVEAU ---

        self.progress_callback = None; self.preview_callback = None
        self.queue = Queue(); self.folders_lock = threading.Lock(); self.processing_thread = None
        self.processed_files = set(); self.additional_folders = []; self.current_folder = None
        self.output_folder = None; self.unaligned_folder = None; self.drizzle_temp_dir = None
        self.output_filename = ""
        self.drizzle_batch_output_dir = None; self.final_stacked_path = None
        self.api_key = None; self.reference_wcs_object = None; self.reference_header_for_wcs = None
        self.reference_pixel_scale_arcsec = None; self.drizzle_output_wcs = None; self.drizzle_output_shape_hw = None
        
        self.sum_memmap_path = None 
        self.wht_memmap_path = None 
        self.cumulative_sum_memmap = None  
        self.cumulative_wht_memmap = None  
        self.memmap_shape = None           
        self.memmap_dtype_sum = np.float32 
        self.memmap_dtype_wht = np.float32 
        print("  -> Attributs SUM/W (memmap) initialisés à None.")
        
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
        self.intermediate_drizzle_batch_files = [] 
        
        self.incremental_drizzle_objects = []     
        self.incremental_drizzle_sci_arrays = []  
        self.incremental_drizzle_wht_arrays = []  
        print("  -> Attributs pour Drizzle Incrémental (objets/arrays) initialisés à listes vides.")

        self.stacking_mode = "kappa-sigma"; self.kappa = 2.5; self.batch_size = 10
        self.hot_pixel_threshold = 3.0; self.neighborhood_size = 5; self.bayer_pattern = "GRBG"
        self.drizzle_mode = "Final"; self.drizzle_scale = 2.0; self.drizzle_wht_threshold = 0.7
        self.drizzle_kernel = "square"; self.drizzle_pixfrac = 1.0 
        self.final_scnr_target_channel = 'green'; self.final_scnr_amount = 0.8; self.final_scnr_preserve_luminosity = True
        
        self.files_in_queue = 0; self.processed_files_count = 0; self.aligned_files_count = 0
        self.stacked_batches_count = 0; self.total_batches_estimated = 0
        self.failed_align_count = 0; self.failed_stack_count = 0; self.skipped_files_count = 0
        self.photutils_bn_applied_in_session = False
        self.bn_globale_applied_in_session = False
        self.cb_applied_in_session = False
        self.feathering_applied_in_session = False 
        self.low_wht_mask_applied_in_session = False 
        self.scnr_applied_in_session = False
        self.crop_applied_in_session = False
        self.photutils_params_used_in_session = {}
        self.last_saved_data_for_preview = None 

        print("  -> Attributs simples et paramètres par défaut initialisés.")
        
        self.local_aligner_instance = None
        self.is_local_alignment_preferred_for_mosaic = True 
        print(f"  -> Mosaïque: Préférence pour alignement local: {self.is_local_alignment_preferred_for_mosaic}")

        try:
            print("  -> Instanciation ChromaticBalancer...")
            self.chroma_balancer = ChromaticBalancer(border_size=50, blur_radius=15) 
            print("     ✓ ChromaticBalancer OK.")
        except Exception as e_cb: 
            print(f"  -> ERREUR ChromaticBalancer: {e_cb}")
            self.chroma_balancer = None

        try:
            print("  -> Instanciation SeestarAligner (pour alignement général astroalign)...")
            self.aligner = SeestarAligner() 
            print("     ✓ SeestarAligner (astroalign) OK.")
        except Exception as e_align: 
            print(f"  -> ERREUR SeestarAligner (astroalign): {e_align}")
            self.aligner = None
            raise 

        try:
            print("  -> Instanciation AstrometrySolver...")
            self.astrometry_solver = AstrometrySolver(progress_callback=self.update_progress) 
            print("     ✓ AstrometrySolver instancié.")
        except Exception as e_as_solver:
            print(f"  -> ERREUR AstrometrySolver instantiation: {e_as_solver}")
            self.astrometry_solver = None 
        
        print("==== FIN INITIALISATION SeestarQueuedStacker (AVEC LocalAligner) ====\n")


        if _LOCAL_ALIGNER_AVAILABLE and SeestarLocalAligner is not None:
            try:
                print("  -> Instanciation SeestarLocalAligner (pour mosaïque locale si préférée)...")
                self.local_aligner_instance = SeestarLocalAligner(debug=True) 
                print("     ✓ SeestarLocalAligner instancié.")
            except Exception as e_local_align_inst:
                print(f"  -> ERREUR lors de l'instanciation de SeestarLocalAligner: {e_local_align_inst}")
                traceback.print_exc(limit=1)
                self.local_aligner_instance = None
                print("     WARN QM: Instanciation de SeestarLocalAligner a échoué. Il ne sera pas utilisable.")
        else:
            print("  -> SeestarLocalAligner n'est pas disponible (import échoué ou classe non définie), instanciation ignorée.")
            self.local_aligner_instance = None 

        print("==== FIN INITIALISATION SeestarQueuedStacker (AVEC LocalAligner) ====\n")



######################################################################################################################################################





    def _move_to_unaligned(self, file_path):
        """
        Déplace un fichier dans un sous-dossier 'unaligned_by_stacker' 
        CRÉÉ DANS LE DOSSIER D'ORIGINE du fichier.
        Notifie l'utilisateur via update_progress (log spécial) la première fois 
        pour un dossier source.
        Version: V_MoveUnaligned_RobustAdd
        """
        # --- NOUVELLE VÉRIFICATION DE LA PRÉSENCE DU FICHIER EN DÉBUT ---
        if not file_path or not isinstance(file_path, str) or file_path.strip() == "":
            print(f"DEBUG QM [_move_to_unaligned_V_MoveUnaligned_RobustAdd]: Chemin fichier source invalide ou vide: '{file_path}'. Sortie précoce.")
            return

        original_folder_abs = os.path.abspath(os.path.dirname(file_path))
        file_basename = os.path.basename(file_path)
        
        # Ce check doit être fait après avoir extrait le basename pour un meilleur log
        if not os.path.exists(file_path):
            print(f"DEBUG QM [_move_to_unaligned_V_MoveUnaligned_RobustAdd]: Fichier '{file_basename}' (chemin: '{file_path}') N'EXISTE PAS au début de _move_to_unaligned. Abandon.")
            return # Sortie si le fichier n'existe vraiment pas

        unaligned_subfolder_name = "unaligned_by_stacker" 
        destination_folder_for_this_file = os.path.join(original_folder_abs, unaligned_subfolder_name)

        # --- Notification (message spécial) ---
        # Cette notification se fait toujours si le dossier n'a pas déjà été averti,
        # avant même de tenter le déplacement.
        # Le set.add() pour le dossier sera fait plus tard, SEULEMENT si le déplacement réussit.
        if original_folder_abs not in self.warned_unaligned_source_folders:
            info_msg_for_ui = (
                f"Les fichiers de '{os.path.basename(original_folder_abs)}' qui ne peuvent pas être alignés "
                f"seront déplacés dans son sous-dossier : '{unaligned_subfolder_name}'. "
                f"(Ce message apparaît une fois par dossier source par session)"
            )
            self.update_progress(f"UNALIGNED_INFO:{info_msg_for_ui}", "WARN") 
            # Ne pas ajouter à warned_unaligned_source_folders ICI, mais plus tard si succès.
        # --- Fin Notification ---

        try:
            # S'assurer que le dossier de destination existe
            os.makedirs(destination_folder_for_this_file, exist_ok=True)
            
            dest_path = os.path.join(destination_folder_for_this_file, file_basename)
            
            # Gérer les conflits de noms si le fichier existe déjà à destination
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(file_basename)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                unique_filename = f"{base}_unaligned_{timestamp}{ext}"
                dest_path = os.path.join(destination_folder_for_this_file, unique_filename)
                print(f"DEBUG QM [_move_to_unaligned_V_MoveUnaligned_RobustAdd]: Conflit de nom pour '{file_basename}', renommé en '{unique_filename}' dans '{destination_folder_for_this_file}'.")

            # --- Logique de déplacement/copie avec retry et pause ---
            max_retries = 3
            initial_delay_sec = 0.1 # Petite pause initiale
            final_move_copy_success = False

            for attempt in range(max_retries):
                if not os.path.exists(file_path): # Le fichier peut disparaître entre les tentatives
                    print(f"DEBUG QM [_move_to_unaligned_V_MoveUnaligned_RobustAdd]: Fichier '{file_basename}' n'existe plus à l'essai {attempt+1}. Abandon des tentatives.")
                    break # Sortir de la boucle si le fichier a disparu

                try:
                    # Ajouter une petite pause pour laisser le système libérer le fichier
                    if attempt > 0: # Pause uniquement après la première tentative
                        time.sleep(initial_delay_sec * (2 ** (attempt - 1))) # Délai exponentiel
                        print(f"DEBUG QM [_move_to_unaligned_V_MoveUnaligned_RobustAdd]: Ré-essai {attempt+1}/{max_retries} pour déplacer '{file_basename}' après pause...")

                    # Tenter de déplacer
                    shutil.move(file_path, dest_path)
                    final_move_copy_success = True
                    break # Succès, sortir de la boucle

                except (OSError, FileNotFoundError, shutil.Error) as e_move:
                    print(f"DEBUG QM [_move_to_unaligned_V_MoveUnaligned_RobustAdd]: Échec déplacement '{file_basename}' (essai {attempt+1}): {e_move}")
                    if attempt == max_retries - 1: # Dernière tentative échouée, essayer de copier
                        print(f"DEBUG QM [_move_to_unaligned_V_MoveUnaligned_RobustAdd]: Échec déplacement après {max_retries} essais. Tentative de copie en dernier recours...")
                        try:
                            shutil.copy2(file_path, dest_path)
                            print(f"DEBUG QM [_move_to_unaligned_V_MoveUnaligned_RobustAdd]: Copie de '{file_basename}' réussie en dernier recours.")
                            final_move_copy_success = True # Considérer comme succès si la copie marche
                        except Exception as e_copy:
                            print(f"DEBUG QM [_move_to_unaligned_V_MoveUnaligned_RobustAdd]: Échec de la copie de '{file_basename}' aussi : {e_copy}")
                            final_move_copy_success = False # La copie a aussi échoué
            # --- Fin Nouvelle logique ---

            if final_move_copy_success:
                self.update_progress(f"   Déplacé vers non alignés: '{file_basename}' (maintenant dans '{unaligned_subfolder_name}' de son dossier source).", "INFO_DETAIL")
                print(f"DEBUG QM [_move_to_unaligned_V_MoveUnaligned_RobustAdd]: Fichier '{file_basename}' traité (déplacé/copié) vers '{dest_path}'.")
                
                # NOUVEAU : Ajouter le dossier source au set SEULEMENT si le déplacement/copie a réussi
                self.warned_unaligned_source_folders.add(original_folder_abs)
                print(f"DEBUG QM [_move_to_unaligned_V_MoveUnaligned_RobustAdd]: Dossier source '{original_folder_abs}' ajouté à warned_unaligned_source_folders.")

            else: # Final_move_copy_success est False
                self.update_progress(f"   ❌ Échec déplacement/copie fichier non-aligné '{file_basename}'.", "ERROR")
                print(f"ERREUR QM [_move_to_unaligned_V_MoveUnaligned_RobustAdd]: Échec définitif déplacement/copie de '{file_basename}'.")


        except Exception as e:
            # Gérer toute autre exception inattendue lors de la préparation/finalisation
            error_details = f"Erreur générale _move_to_unaligned pour '{file_basename}': {e}"
            print(f"ERREUR QM [_move_to_unaligned_V_MoveUnaligned_RobustAdd]: {error_details}")
            traceback.print_exc(limit=1)
            self.update_progress(f"   ❌ Erreur inattendue déplacement/copie fichier non-aligné '{file_basename}': {type(e).__name__}", "ERROR")






#######################################################################################################################################################





# --- DANS LA CLASSE SeestarQueuedStacker DANS seestar/queuep/queue_manager.py ---

    def initialize(self, output_dir, reference_image_shape_hwc_input): # Renommé pour clarté
        """
        Prépare les dossiers, réinitialise l'état.
        CRÉE/INITIALISE les fichiers memmap pour SUM et WHT (si pas Drizzle Incrémental VRAI).
        OU INITIALISE les objets Drizzle persistants (si Drizzle Incrémental VRAI).
        Version: V_DrizIncr_StrategyA_Init_MemmapDirFix
        """
        
        print(f"DEBUG QM [initialize V_DrizIncr_StrategyA_Init_MemmapDirFix]: Début avec output_dir='{output_dir}', shape_ref_HWC={reference_image_shape_hwc_input}")
        print(f"  VALEURS AU DÉBUT DE INITIALIZE:")
        print(f"    -> self.is_mosaic_run: {getattr(self, 'is_mosaic_run', 'Non Défini')}")
        print(f"    -> self.drizzle_active_session: {getattr(self, 'drizzle_active_session', 'Non Défini')}")
        print(f"    -> self.drizzle_mode: {getattr(self, 'drizzle_mode', 'Non Défini')}")
        
        # --- Nettoyage et création dossiers ---
        try:
            self.output_folder = os.path.abspath(output_dir)
            self.unaligned_folder = os.path.join(self.output_folder, "unaligned_files")
            self.drizzle_temp_dir = os.path.join(self.output_folder, "drizzle_temp_inputs")
            self.drizzle_batch_output_dir = os.path.join(self.output_folder, "drizzle_batch_outputs")
            
            # Définir le chemin du dossier memmap mais ne le créer que si nécessaire plus tard
            memmap_dir = os.path.join(self.output_folder, "memmap_accumulators")
            self.sum_memmap_path = os.path.join(memmap_dir, "cumulative_SUM.npy")
            self.wht_memmap_path = os.path.join(memmap_dir, "cumulative_WHT.npy")

            os.makedirs(self.output_folder, exist_ok=True)
            os.makedirs(self.unaligned_folder, exist_ok=True)
            
            if self.drizzle_active_session or self.is_mosaic_run:
                os.makedirs(self.drizzle_temp_dir, exist_ok=True)
                if self.drizzle_mode == "Final" and not self.is_mosaic_run :
                     os.makedirs(self.drizzle_batch_output_dir, exist_ok=True)
            
            # La création de memmap_dir est déplacée plus bas, dans la condition où elle est utilisée.
            
            if self.perform_cleanup:
                if os.path.isdir(self.drizzle_temp_dir):
                    try: shutil.rmtree(self.drizzle_temp_dir); os.makedirs(self.drizzle_temp_dir, exist_ok=True)
                    except Exception as e: self.update_progress(f"⚠️ Erreur nettoyage {self.drizzle_temp_dir}: {e}")
                if os.path.isdir(self.drizzle_batch_output_dir) and self.drizzle_mode == "Final" and not self.is_mosaic_run : # Nettoyer seulement si utilisé
                    try: shutil.rmtree(self.drizzle_batch_output_dir); os.makedirs(self.drizzle_batch_output_dir, exist_ok=True)
                    except Exception as e: self.update_progress(f"⚠️ Erreur nettoyage {self.drizzle_batch_output_dir}: {e}")
            self.update_progress(f"🗄️ Dossiers prêts.")
        except OSError as e:
            self.update_progress(f"❌ Erreur critique création dossiers: {e}", 0) # progress_val 0
            return False

        # --- Validation Shape Référence (HWC) ---
        if not isinstance(reference_image_shape_hwc_input, tuple) or len(reference_image_shape_hwc_input) != 3 or \
           reference_image_shape_hwc_input[2] != 3:
            self.update_progress(f"❌ Erreur interne: Shape référence HWC invalide ({reference_image_shape_hwc_input}).")
            return False
        
        current_output_shape_hw_for_accum_or_driz = None 
        
        # --- Logique d'initialisation spécifique au mode ---
        is_true_incremental_drizzle_mode = (self.drizzle_active_session and 
                                            self.drizzle_mode == "Incremental" and
                                            not self.is_mosaic_run) 
        
        print(f"  DEBUG QM [initialize]: Valeur calculée de is_true_incremental_drizzle_mode: {is_true_incremental_drizzle_mode}")
        print(f"    -> self.drizzle_active_session ÉTAIT: {self.drizzle_active_session}")
        print(f"    -> self.drizzle_mode ÉTAIT: '{self.drizzle_mode}' (comparé à 'Incremental')")
        print(f"    -> not self.is_mosaic_run ÉTAIT: {not self.is_mosaic_run} (self.is_mosaic_run était {self.is_mosaic_run})")

        if is_true_incremental_drizzle_mode:
            print("DEBUG QM [initialize V_DrizIncr_StrategyA_Init_MemmapDirFix]: Mode Drizzle Incrémental VRAI détecté.")
            if self.reference_wcs_object is None:
                self.update_progress("❌ Erreur: WCS de référence manquant pour initialiser la grille Drizzle Incrémental.", "ERROR")
                return False
            try:
                ref_shape_hw_for_grid = reference_image_shape_hwc_input[:2]
                self.drizzle_output_wcs, self.drizzle_output_shape_hw = self._create_drizzle_output_wcs(
                    self.reference_wcs_object, ref_shape_hw_for_grid, self.drizzle_scale
                )
                if self.drizzle_output_wcs is None or self.drizzle_output_shape_hw is None:
                    raise RuntimeError("Échec _create_drizzle_output_wcs pour Drizzle Incrémental.")
                current_output_shape_hw_for_accum_or_driz = self.drizzle_output_shape_hw
                print(f"  -> Grille Drizzle Incrémental: Shape={current_output_shape_hw_for_accum_or_driz}, WCS CRVAL={self.drizzle_output_wcs.wcs.crval if self.drizzle_output_wcs.wcs else 'N/A'}")
            except Exception as e_grid:
                self.update_progress(f"❌ Erreur création grille Drizzle Incrémental: {e_grid}", "ERROR")
                return False

            self.update_progress(f"💧 Initialisation des objets Drizzle persistants pour mode Incrémental (Shape: {current_output_shape_hw_for_accum_or_driz})...")
            self.incremental_drizzle_objects = []
            self.incremental_drizzle_sci_arrays = []
            self.incremental_drizzle_wht_arrays = []
            num_channels_driz = 3 

            try:
                for _ in range(num_channels_driz):
                    sci_arr = np.zeros(current_output_shape_hw_for_accum_or_driz, dtype=np.float32)
                    wht_arr = np.zeros(current_output_shape_hw_for_accum_or_driz, dtype=np.float32)
                    self.incremental_drizzle_sci_arrays.append(sci_arr)
                    self.incremental_drizzle_wht_arrays.append(wht_arr)
                    
                    driz_obj = Drizzle( # Assumes Drizzle is imported
                        out_img=sci_arr, out_wht=wht_arr,
                        out_shape=current_output_shape_hw_for_accum_or_driz,
                        kernel=self.drizzle_kernel, fillval=str(getattr(self, "drizzle_fillval", "0.0"))
                    )
                    self.incremental_drizzle_objects.append(driz_obj)
                print(f"  -> {len(self.incremental_drizzle_objects)} objets Drizzle persistants créés pour mode Incrémental.")
            except Exception as e_driz_obj_init:
                self.update_progress(f"❌ Erreur initialisation objets Drizzle persistants: {e_driz_obj_init}", "ERROR")
                traceback.print_exc(limit=1)
                return False

            self.cumulative_sum_memmap = None
            self.cumulative_wht_memmap = None
            self.memmap_shape = None 
            print("  -> Memmaps SUM/WHT désactivés pour Drizzle Incrémental VRAI.")

        else: # Mosaïque, Drizzle Final standard, ou Stacking Classique -> Utiliser Memmaps SUM/W
            print("DEBUG QM [initialize V_DrizIncr_StrategyA_Init_MemmapDirFix]: Mode NON-Drizzle Incr. VRAI. Initialisation Memmaps SUM/W...")
            
            # ***** CORRECTION: Créer memmap_dir ICI, seulement si cette branche est exécutée *****
            try:
                os.makedirs(memmap_dir, exist_ok=True)
                print(f"  -> Dossier pour memmap '{memmap_dir}' créé (ou existait déjà).")
            except OSError as e_mkdir_memmap:
                self.update_progress(f"❌ Erreur critique création dossier memmap '{memmap_dir}': {e_mkdir_memmap}", "ERROR")
                return False
            # ***** FIN CORRECTION *****

            self.memmap_shape = reference_image_shape_hwc_input 
            wht_shape_memmap = self.memmap_shape[:2] 
            print(f"  -> Shape Memmap SUM={self.memmap_shape}, WHT={wht_shape_memmap}")

            print(f"  -> Tentative création/ouverture fichiers memmap SUM/WHT (mode 'w+')...")
            try:
                self.cumulative_sum_memmap = np.lib.format.open_memmap(
                    self.sum_memmap_path, mode='w+', dtype=self.memmap_dtype_sum, shape=self.memmap_shape
                )
                self.cumulative_sum_memmap[:] = 0.0
                print(f"  -> Memmap SUM ({self.memmap_shape}) créé/ouvert et initialisé à zéro.")

                self.cumulative_wht_memmap = np.lib.format.open_memmap(
                    self.wht_memmap_path, mode='w+', dtype=self.memmap_dtype_wht, shape=wht_shape_memmap
                )
                self.cumulative_wht_memmap[:] = 0 
                print(f"  -> Memmap WHT ({wht_shape_memmap}) créé/ouvert et initialisé à zéro.")
                
                self.incremental_drizzle_objects = []
                self.incremental_drizzle_sci_arrays = []
                self.incremental_drizzle_wht_arrays = []

            except (IOError, OSError, ValueError, TypeError) as e_memmap:
                self.update_progress(f"❌ Erreur création/initialisation fichier memmap: {e_memmap}")
                print(f"ERREUR QM [initialize V_DrizIncr_StrategyA_Init_MemmapDirFix]: Échec memmap : {e_memmap}"); traceback.print_exc(limit=2)
                self.cumulative_sum_memmap = None; self.cumulative_wht_memmap = None
                self.sum_memmap_path = None; self.wht_memmap_path = None
                return False
        
        # --- Réinitialisations Communes ---
        self.warned_unaligned_source_folders.clear()
        print("DEBUG QM [initialize V_DrizIncr_StrategyA_Init_MemmapDirFix]: Réinitialisation des autres états...")
        # self.reference_wcs_object est conservé s'il a été défini par start_processing (plate-solving de réf)
        self.intermediate_drizzle_batch_files = []
        
        self.processed_files.clear()
        with self.folders_lock: self.additional_folders = []
        self.current_batch_data = []; self.current_stack_header = None; self.images_in_cumulative_stack = 0
        self.total_exposure_seconds = 0.0; self.final_stacked_path = None; self.processing_error = None
        self.files_in_queue = 0; self.processed_files_count = 0; self.aligned_files_count = 0
        self.stacked_batches_count = 0; self.total_batches_estimated = 0
        self.failed_align_count = 0; self.failed_stack_count = 0; self.skipped_files_count = 0
        
        self.photutils_bn_applied_in_session = False
        self.bn_globale_applied_in_session = False
        self.cb_applied_in_session = False
        self.feathering_applied_in_session = False 
        self.low_wht_mask_applied_in_session = False 
        self.scnr_applied_in_session = False
        self.crop_applied_in_session = False
        self.photutils_params_used_in_session = {}

        while not self.queue.empty():
            try: self.queue.get_nowait(); self.queue.task_done()
            except Exception: break

        if hasattr(self, 'aligner') and self.aligner: self.aligner.stop_processing = False
        print("DEBUG QM [initialize V_DrizIncr_StrategyA_Init_MemmapDirFix]: Initialisation terminée avec succès.")
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




    def _calculate_M_from_wcs(self,
                            wcs_source: WCS,
                            wcs_target: WCS,
                            shape_source_hw: tuple,
                            num_points_edge: int = 6,
                            ransac_thresh_fallback: float = 5.0): # << MODIFIÉ la valeur par défaut à 5.0
        """
        Calcule la matrice affine M...
        MODIFIED: Augmentation du seuil RANSAC par défaut et logs plus détaillés.
        """
        # Utiliser self.update_progress pour les logs visibles dans l'UI
        self.update_progress(f"    [FallbackWCS] Tentative calcul M (Source->Cible). RANSAC Thresh: {ransac_thresh_fallback}px", "DEBUG_DETAIL") # << Log amélioré

        if not (wcs_source and wcs_source.is_celestial and wcs_target and wcs_target.is_celestial):
            self.update_progress("      [FallbackWCS] Échec: WCS source ou cible invalide/non céleste.", "WARN")
            return None

        h, w = shape_source_hw
        if h < num_points_edge or w < num_points_edge:
            self.update_progress(f"      [FallbackWCS] Échec: Image source trop petite ({w}x{h}) pour grille {num_points_edge}x{num_points_edge}.", "WARN")
            return None

        xs = np.linspace(0, w - 1, num_points_edge, dtype=np.float32)
        ys = np.linspace(0, h - 1, num_points_edge, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        src_pts_pix_flat = np.vstack([xv.ravel(), yv.ravel()]).T

        if len(src_pts_pix_flat) < 3:
            self.update_progress(f"      [FallbackWCS] Échec: Pas assez de points de contrôle ({len(src_pts_pix_flat)}).", "WARN")
            return None
        self.update_progress(f"      [FallbackWCS] {len(src_pts_pix_flat)} points de contrôle source générés.", "DEBUG_DETAIL")

        try:
            sky_coords_ra, sky_coords_dec = wcs_source.all_pix2world(src_pts_pix_flat[:,0], src_pts_pix_flat[:,1], 0)
            if not (np.all(np.isfinite(sky_coords_ra)) and np.all(np.isfinite(sky_coords_dec))):
                self.update_progress("      [FallbackWCS] Échec: Coords célestes non finies depuis wcs_source.", "WARN")
                return None

            dst_pts_pix_flat_x, dst_pts_pix_flat_y = wcs_target.all_world2pix(sky_coords_ra, sky_coords_dec, 0)
            if not (np.all(np.isfinite(dst_pts_pix_flat_x)) and np.all(np.isfinite(dst_pts_pix_flat_y))):
                self.update_progress("      [FallbackWCS] Échec: Coords pixels cibles non finies depuis wcs_target.", "WARN")
                return None

            dst_pts_pix_flat = np.column_stack((dst_pts_pix_flat_x, dst_pts_pix_flat_y)).astype(np.float32)
            self.update_progress(f"      [FallbackWCS] Points source et destination prêts pour estimation M.", "DEBUG_DETAIL")

            src_pts_cv = src_pts_pix_flat.reshape(-1, 1, 2)
            dst_pts_cv = dst_pts_pix_flat.reshape(-1, 1, 2)

            M, inliers_mask = cv2.estimateAffinePartial2D(src_pts_cv, dst_pts_cv,
                                                        method=cv2.RANSAC,
                                                        ransacReprojThreshold=ransac_thresh_fallback,
                                                        maxIters=1000,
                                                        confidence=0.95)

            if M is None:
                self.update_progress(f"      [FallbackWCS] Échec: estimateAffinePartial2D n'a pas retourné de matrice (avec seuil {ransac_thresh_fallback}px).", "WARN") # << Log amélioré
                return None

            num_inliers = np.sum(inliers_mask) if inliers_mask is not None else 0
            min_inliers_needed_fallback = max(3, len(src_pts_cv) // 6)

            self.update_progress(f"      [FallbackWCS] RANSAC: {num_inliers} inliers / {len(src_pts_cv)} points (seuil {ransac_thresh_fallback}px). Requis: {min_inliers_needed_fallback}.", "INFO") # << Log amélioré

            if num_inliers < min_inliers_needed_fallback:
                self.update_progress(f"      [FallbackWCS] Échec: Pas assez d'inliers RANSAC.", "WARN")
                return None

            self.update_progress(f"      [FallbackWCS] Matrice M calculée avec succès.", "INFO")
            # print(f"  DEBUG QM [_calculate_M_from_wcs]: Matrice M de fallback WCS calculée:\n{M}") # Garder pour debug console
            return M

        except Exception as e_m_wcs:
            self.update_progress(f"      [FallbackWCS] ERREUR: Exception lors du calcul de M: {e_m_wcs}", "ERROR")
            # print(f"ERREUR QM [_calculate_M_from_wcs]: {e_m_wcs}") # Garder pour debug console
            # if self.debug_mode: traceback.print_exc(limit=1) # Supposant un self.debug_mode
            return None




##########################################################################################################################################################

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







    def _worker(self):
        """
        Thread principal pour le traitement des images.
        Version: V5.3.2_AstroPerPanelFix (Correction appel _process_file pour Astrometry par panneau)
        """
        # ================================================================================
        # === SECTION 0 : INITIALISATION DU WORKER ET CONFIGURATION DE SESSION ===
        # ================================================================================
        print("\n" + "=" * 10 + f" DEBUG QM [_worker V5.3.2_AstroPerPanelFix]: Initialisation du worker " + "=" * 10)

        self.processing_active = True
        self.processing_error = None
        # start_time_session = time.monotonic() # Décommenter si besoin

        reference_image_data_for_global_alignment = None
        reference_header_for_global_alignment = None
        mosaic_ref_panel_image_data = None # Utilisé seulement si local_fast_fallback
        mosaic_ref_panel_header = None     # Utilisé seulement si local_fast_fallback

        current_batch_items_with_masks_for_stack_batch = []
        self.intermediate_drizzle_batch_files = []
        all_aligned_files_with_info_for_mosaic = []

        # --- 0.B Détermination du mode d'opération (basé sur self.xxx settés par start_processing) ---
        use_local_aligner_for_this_mosaic_run = (
            self.is_mosaic_run and
            self.mosaic_alignment_mode in ["local_fast_fallback", "local_fast_only"] and
            _LOCAL_ALIGNER_AVAILABLE and
            self.local_aligner_instance is not None
        )
        use_wcs_fallback_if_local_fails = ( # Utilisé seulement si use_local_aligner_for_this_mosaic_run est True
            use_local_aligner_for_this_mosaic_run and
            self.mosaic_alignment_mode == "local_fast_fallback"
        )
        use_astrometry_per_panel_mosaic = (
            self.is_mosaic_run and
            self.mosaic_alignment_mode == "astrometry_per_panel"
        )

        print(f"DEBUG QM [_worker V5.3.2_AstroPerPanelFix]: Configuration de la session:")
        print(f"  - is_mosaic_run: {self.is_mosaic_run}")
        if self.is_mosaic_run:
            print(f"    - mosaic_alignment_mode: '{self.mosaic_alignment_mode}'")
            print(f"    - -> Utilisation Aligneur Local (FastAligner): {use_local_aligner_for_this_mosaic_run}")
            if use_local_aligner_for_this_mosaic_run:
                print(f"      - Fallback WCS si FastAligner échoue: {use_wcs_fallback_if_local_fails}")
            print(f"    - -> Utilisation Astrometry par Panneau: {use_astrometry_per_panel_mosaic}") # Crucial
        print(f"  - drizzle_active_session (pour stacking standard non-mosaïque): {self.drizzle_active_session}")
        if self.drizzle_active_session and not self.is_mosaic_run:
            print(f"    - drizzle_mode (standard): '{self.drizzle_mode}'")

        path_of_processed_ref_panel_basename = None # Pour skipper le panneau d'ancre si local_fast_fallback
        
        # Récupérer les paramètres Drizzle spécifiques à la mosaïque depuis mosaic_settings_dict
        mosaic_drizzle_kernel_effective = str(self.mosaic_settings_dict.get('kernel', "square"))
        mosaic_drizzle_pixfrac_effective = float(self.mosaic_settings_dict.get('pixfrac', 1.0))
        mosaic_drizzle_fillval_effective = str(self.mosaic_settings_dict.get('fillval', "0.0"))
        mosaic_drizzle_wht_threshold_effective = float(self.mosaic_settings_dict.get('wht_threshold', 0.01))

        # Les paramètres globaux de Drizzle (self.drizzle_kernel, self.drizzle_pixfrac, etc.)
        # sont déjà configurés par start_processing.
        # Ici, nous les *surchargons* avec les valeurs spécifiques à la mosaïque si le mode mosaïque est actif.
        if self.is_mosaic_run:
            self.drizzle_kernel = mosaic_drizzle_kernel_effective
            self.drizzle_pixfrac = mosaic_drizzle_pixfrac_effective
            self.drizzle_fillval = mosaic_drizzle_fillval_effective # <-- Assurez-vous que cet attribut existe sur self
            self.drizzle_wht_threshold = mosaic_drizzle_wht_threshold_effective # <-- Assurez-vous que cet attribut existe sur self

            print(f"DEBUG QM [_worker]: Mode Mosaïque ACTIF. Surcharge des paramètres Drizzle globaux:")
            print(f"  -> self.drizzle_kernel mis à '{self.drizzle_kernel}' (depuis mosaic_settings)")
            print(f"  -> self.drizzle_pixfrac mis à '{self.drizzle_pixfrac}' (depuis mosaic_settings)")
            print(f"  -> self.drizzle_fillval mis à '{self.drizzle_fillval}' (depuis mosaic_settings)")
            print(f"  -> self.drizzle_wht_threshold mis à '{self.drizzle_wht_threshold}' (depuis mosaic_settings)")
        else:
            # S'assurer que les attributs spécifiques à la mosaïque (qui ne sont pas self.drizzle_*)
            # ont une valeur par défaut, même si le mode mosaïque n'est pas actif.
            # Cela évite des erreurs si on les lit par erreur dans d'autres branches de code.
            # (Si vos attributs `mosaic_drizzle_kernel` etc. ne sont pas déjà initialisés dans `__init__`,
            # il faudrait les initialiser ici. Actuellement, ils le sont via `start_processing` ou `initialize`
            # donc ce bloc 'else' est pour la clarté mais pas strictement nécessaire ici si le flux est correct.)
            pass # Les attributs self.mosaic_drizzle_xyz sont déjà settés par start_processing et ne sont pas lus ici.
        

        try:

            # =====================================================================================
            # === SECTION 1: PRÉPARATION DE L'IMAGE DE RÉFÉRENCE ET DU/DES WCS DE RÉFÉRENCE ===
            # =====================================================================================
        
            self.update_progress("⭐ Préparation image(s) de référence...")
            
            # --- Détermination du dossier et des fichiers pour la référence ---
            files_for_ref_scan = [] 
            folder_for_ref_scan = None
            if self.current_folder and os.path.isdir(self.current_folder):
                files_for_ref_scan = sorted([f for f in os.listdir(self.current_folder) if f.lower().endswith((".fit", ".fits"))])
                if files_for_ref_scan: folder_for_ref_scan = self.current_folder
            
            if not files_for_ref_scan and hasattr(self, 'additional_folders') and self.additional_folders:
                first_additional = self.additional_folders[0]
                if os.path.isdir(first_additional):
                    files_for_ref_scan_add = sorted([f for f in os.listdir(first_additional) if f.lower().endswith((".fit", ".fits"))])
                    if files_for_ref_scan_add: 
                        files_for_ref_scan = files_for_ref_scan_add
                        folder_for_ref_scan = first_additional
                        print(f"DEBUG QM [_worker]: Dossier initial vide/invalide, utilisation du premier dossier additionnel '{os.path.basename(folder_for_ref_scan)}' pour la référence.")
            
            if not files_for_ref_scan or not folder_for_ref_scan: 
                raise RuntimeError("Aucun fichier FITS trouvé dans les dossiers d'entrée initiaux pour déterminer la référence.")
            # --- Fin logique dossier/fichiers référence ---

            # Configuration de self.aligner pour _get_reference_image
            self.aligner.correct_hot_pixels = self.correct_hot_pixels 
            self.aligner.hot_pixel_threshold = self.hot_pixel_threshold
            self.aligner.neighborhood_size = self.neighborhood_size
            self.aligner.bayer_pattern = self.bayer_pattern
            # self.aligner.reference_image_path est déjà setté dans start_processing

            print(f"DEBUG QM [_worker]: Appel à self.aligner._get_reference_image avec dossier '{os.path.basename(folder_for_ref_scan)}' pour la référence de base/globale...")
            # _get_reference_image DOIT s'assurer que s'il ajoute _SOURCE_PATH à son header interne
            # avant de sauvegarder reference_image.fit, il utilise os.path.basename().
            # C'est la source de l'erreur "keyword too long".
            reference_image_data_for_global_alignment, reference_header_for_global_alignment = self.aligner._get_reference_image(
                folder_for_ref_scan, 
                files_for_ref_scan,
                self.output_folder  
            )
            if reference_image_data_for_global_alignment is None or reference_header_for_global_alignment is None:
                raise RuntimeError("Échec critique obtention image/header de référence de base (globale/premier panneau).")

            # Préparation du header qui sera utilisé pour le WCS de référence global
            self.reference_header_for_wcs = reference_header_for_global_alignment.copy() 
            
            # La clé '_SOURCE_PATH' dans reference_header_for_global_alignment vient de
            # la logique interne de _get_reference_image. Si cette clé contient un chemin complet,
            # nous devons extraire le nom de base pour nos propres besoins.
            # Le fichier reference_image.fit lui-même (s'il contient _SOURCE_PATH) doit avoir été sauvegardé
            # par _get_reference_image en utilisant déjà un nom de base pour ce mot-clé.
            original_source_path_from_ref_fits = reference_header_for_global_alignment.get('_SOURCE_PATH')

            if original_source_path_from_ref_fits:
                source_basename_for_wcs_ref = os.path.basename(str(original_source_path_from_ref_fits))
                # Utiliser une clé claire pour indiquer que c'est le nom de base du fichier de référence
                self.reference_header_for_wcs['REF_FNAME'] = (source_basename_for_wcs_ref, "Basename of the source file for global WCS reference")
                
                # Pour la logique de mosaïque locale, path_of_processed_ref_panel_basename
                # doit aussi être juste le nom de base.
                if use_local_aligner_for_this_mosaic_run: 
                    path_of_processed_ref_panel_basename = source_basename_for_wcs_ref
                    print(f"DEBUG QM [_worker]: Panneau d'ancre identifié par basename: {path_of_processed_ref_panel_basename}")
            else:
                # Si _SOURCE_PATH n'est pas dans le header de reference_image.fit, on ne peut pas le définir
                # Cela pourrait arriver si _get_reference_image ne l'ajoute pas.
                print("WARN QM [_worker]: Mot-clé '_SOURCE_PATH' non trouvé dans le header de l'image de référence globale.")
                if use_local_aligner_for_this_mosaic_run:
                     path_of_processed_ref_panel_basename = "unknown_reference_panel.fits" # Fallback

            ref_temp_processing_dir = os.path.join(self.output_folder, "temp_processing")
            reference_image_path_for_solver = os.path.join(ref_temp_processing_dir, "reference_image.fit")
            
            # À ce stade, reference_image.fit doit exister, sinon l'erreur que tu as eue se produira.
            if not os.path.exists(reference_image_path_for_solver):
                # Cette erreur devrait être prévenue si _get_reference_image fonctionne correctement
                # ET si la correction pour _SOURCE_PATH trop long est appliquée DANS _get_reference_image.
                raise RuntimeError(f"CRITICAL: Fichier de référence '{reference_image_path_for_solver}' non trouvé après appel à _get_reference_image. Vérifier la logique de sauvegarde dans SeestarAligner._get_reference_image pour les headers longs.")

            print(f"DEBUG QM [_worker]: Image de référence de base (pour shape et solving) prête: {reference_image_path_for_solver}")



            # --- 1.A Plate-solving de la référence ---
            self.update_progress("DEBUG WORKER: Section 1.A - Plate-solving de la référence...")
            self.reference_wcs_object = None 
            temp_wcs_ancre = None # Spécifique pour la logique mosaïque locale

            # <<< DEBUG PRINT AVANT CRÉATION DICT POUR SOLVEUR ANCRE >>>
            print(f"!!!! DEBUG _WORKER AVANT CRÉATION DICT SOLVEUR ANCRE !!!!")
            print(f"    self.is_mosaic_run = {self.is_mosaic_run}")
            print(f"    self.local_solver_preference = '{getattr(self, 'local_solver_preference', 'NON_DÉFINI')}'")
            print(f"    self.astap_search_radius = {getattr(self, 'astap_search_radius', 'NON_DÉFINI')}")
            print(f"    self.reference_pixel_scale_arcsec = {self.reference_pixel_scale_arcsec}")
            # <<< FIN DEBUG PRINT >>>

            solver_settings_for_ref_anchor = {
                'local_solver_preference': self.local_solver_preference,
                'api_key': self.api_key,
                'astap_path': self.astap_path,
                'astap_data_dir': self.astap_data_dir,
                'astap_search_radius': self.astap_search_radius,
                'local_ansvr_path': self.local_ansvr_path,
                'scale_est_arcsec_per_pix': self.reference_pixel_scale_arcsec, # Peut être None au premier passage
                'scale_tolerance_percent': 20,
                'ansvr_timeout_sec': getattr(self, 'ansvr_timeout_sec', 120),
                'astap_timeout_sec': getattr(self, 'astap_timeout_sec', 120),
                'astrometry_net_timeout_sec': getattr(self, 'astrometry_net_timeout_sec', 300)
            }
            # (Vos logs pour le contenu de solver_settings_for_ref_anchor peuvent rester ici)
            print(f"DEBUG QM (_worker): Contenu de solver_settings_for_ref_anchor:") 
            for key_s, val_s in solver_settings_for_ref_anchor.items():               
                if key_s == 'api_key': print(f"    '{key_s}': '{'Présente' if val_s else 'Absente'}'")
                else: print(f"    '{key_s}': '{val_s}'")

            # <<< DEBUG PRINT AVANT BLOC IF/ELIF POUR SOLVING ANCRE >>>
            print(f"!!!! DEBUG _worker AVANT BLOC IF/ELIF POUR SOLVING ANCRE (SECTION 1.A) !!!! self.is_mosaic_run = {self.is_mosaic_run}")

            # --- CAS 1: Mosaïque Locale (FastAligner avec ou sans fallback WCS) ---
            if use_local_aligner_for_this_mosaic_run: # Flag défini au tout début de _worker
                self.update_progress("⭐ Mosaïque Locale: Traitement du panneau de référence (ancrage)...")
                mosaic_ref_panel_image_data = reference_image_data_for_global_alignment 
                mosaic_ref_panel_header = self.reference_header_for_wcs.copy()
                
                if reference_header_for_global_alignment.get('_SOURCE_PATH'):
                    # path_of_processed_ref_panel_basename est déjà défini plus haut
                    mosaic_ref_panel_header['_PANREF_FN'] = (path_of_processed_ref_panel_basename, "Base name of this mosaic ref panel source")

                if self.astrometry_solver and os.path.exists(reference_image_path_for_solver):
                    self.update_progress("   -> Mosaïque Locale: Tentative résolution astrométrique ancre via self.astrometry_solver.solve...")
                    temp_wcs_ancre = self.astrometry_solver.solve(
                        reference_image_path_for_solver,
                        mosaic_ref_panel_header, 
                        settings=solver_settings_for_ref_anchor,
                        update_header_with_solution=True
                    )
                    if temp_wcs_ancre: self.update_progress("   -> Mosaïque Locale: Astrometry (via solveur) ancre RÉUSSI.")
                    else: self.update_progress("   -> Mosaïque Locale: Astrometry (via solveur) ancre ÉCHOUÉ.")
                else:
                    self.update_progress("   -> Mosaïque Locale: AstrometrySolver non dispo ou fichier réf. manquant. Solving ancre impossible.", "ERROR")

                if temp_wcs_ancre is None: 
                    self.update_progress("   ⚠️ Échec de tous les solveurs pour panneau de référence. Tentative WCS approximatif (fallback)...")
                    _cwfh_func = None; from ..enhancement.drizzle_integration import _create_wcs_from_header as _cwfh; _cwfh_func = _cwfh
                    if _cwfh_func: temp_wcs_ancre = _cwfh_func(mosaic_ref_panel_header)
                    if temp_wcs_ancre and temp_wcs_ancre.is_celestial:
                         nx_hdr_a = mosaic_ref_panel_header.get('NAXIS1'); ny_hdr_a = mosaic_ref_panel_header.get('NAXIS2')
                         if nx_hdr_a and ny_hdr_a: temp_wcs_ancre.pixel_shape = (int(nx_hdr_a), int(ny_hdr_a))
                         elif hasattr(mosaic_ref_panel_image_data,'shape'): temp_wcs_ancre.pixel_shape=(mosaic_ref_panel_image_data.shape[1],mosaic_ref_panel_image_data.shape[0])
                
                if temp_wcs_ancre is None: raise RuntimeError("Mosaïque Locale: Échec critique obtention WCS pour panneau de référence.")
                self.reference_wcs_object = temp_wcs_ancre 
                
                if self.reference_wcs_object and hasattr(self.reference_wcs_object, 'pixel_scale_matrix'): # Mettre à jour l'échelle globale
                    try: self.reference_pixel_scale_arcsec = np.sqrt(np.abs(np.linalg.det(self.reference_wcs_object.pixel_scale_matrix))) * 3600.0
                    except: pass # Ignorer si erreur de calcul

                if self.reference_wcs_object: print(f"  DEBUG QM [_worker]: Infos WCS du Panneau d'Ancrage (self.reference_wcs_object): CRVAL={self.reference_wcs_object.wcs.crval if self.reference_wcs_object.wcs else 'N/A'} ...");
                
                mat_identite_ref_panel = np.array([[1.,0.,0.],[0.,1.,0.]], dtype=np.float32)
                valid_mask_ref_panel_pixels = np.ones(mosaic_ref_panel_image_data.shape[:2], dtype=bool)
                all_aligned_files_with_info_for_mosaic.append((mosaic_ref_panel_image_data.copy(), mosaic_ref_panel_header.copy(), self.reference_wcs_object, mat_identite_ref_panel, valid_mask_ref_panel_pixels))
                self.aligned_files_count += 1; self.processed_files_count += 1
                print(f"DEBUG QM [_worker]: Mosaïque Locale: Panneau de référence ajouté à all_aligned_files_with_info_for_mosaic.")

            # --- CAS 2: Mosaïque Astrometry.net par panneau OU Drizzle Standard (pour la référence globale) ---
            elif self.drizzle_active_session or use_astrometry_per_panel_mosaic: # `use_astrometry_per_panel_mosaic` est True si mode mosaique="astrometry_per_panel"
                self.update_progress("DEBUG WORKER: Branche Drizzle Std / AstroMosaic pour référence globale...")
                if self.astrometry_solver and os.path.exists(reference_image_path_for_solver):
                    self.update_progress("   -> Drizzle Std/AstroMosaic: Tentative résolution astrométrique réf. globale via self.astrometry_solver.solve...")
                    self.reference_wcs_object = self.astrometry_solver.solve(
                        reference_image_path_for_solver,
                        self.reference_header_for_wcs, 
                        settings=solver_settings_for_ref_anchor, # Utilise le même dict de settings que pour l'ancre
                        update_header_with_solution=True
                    )
                else:
                    self.update_progress("   -> Drizzle Std/AstroMosaic: AstrometrySolver non dispo ou fichier réf. manquant. Solving réf. globale impossible.", "ERROR")
                    self.reference_wcs_object = None
                
                if self.reference_wcs_object is None: # Si solving a échoué
                    self.update_progress("ERREUR WORKER: Échec plate-solving réf. principale (Drizzle Std / AstroMosaic). Tentative WCS approximatif...", "WARN")
                    # Fallback WCS approximatif pour Drizzle Standard / Mosaïque Astrometry.net per Panel
                    _cwfh_func_std_driz = None; from ..enhancement.drizzle_integration import _create_wcs_from_header as _cwfh_std; _cwfh_func_std_driz = _cwfh_std
                    if _cwfh_func_std_driz: self.reference_wcs_object = _cwfh_func_std_driz(self.reference_header_for_wcs)
                    if not (self.reference_wcs_object and self.reference_wcs_object.is_celestial):
                        raise RuntimeError("Échec critique obtention WCS pour Drizzle standard ou Mosaïque Astrometry (même après fallback).")
                    self.update_progress("   -> WCS approximatif pour réf. globale créé (pour Drizzle Std / AstroMosaic).", "INFO")

                # Si on a un WCS (résolu ou approximatif)
                self.update_progress("   -> Drizzle Std/AstroMosaic: WCS pour réf. globale obtenu (résolu ou approx).")
                if self.reference_wcs_object.pixel_shape is None:
                     nx_ref_hdr = self.reference_header_for_wcs.get('NAXIS1', reference_image_data_for_global_alignment.shape[1])
                     ny_ref_hdr = self.reference_header_for_wcs.get('NAXIS2', reference_image_data_for_global_alignment.shape[0])
                     self.reference_wcs_object.pixel_shape = (int(nx_ref_hdr), int(ny_ref_hdr))
                
                if hasattr(self.reference_wcs_object, 'pixel_scale_matrix'): # Mettre à jour l'échelle globale
                    try: self.reference_pixel_scale_arcsec = np.sqrt(np.abs(np.linalg.det(self.reference_wcs_object.pixel_scale_matrix))) * 3600.0
                    except: pass

                print(f"  DEBUG QM [_worker]: Infos WCS de Référence Globale: CRVAL={self.reference_wcs_object.wcs.crval if self.reference_wcs_object.wcs else 'N/A'} ...");
            
            # <<< DEBUG PRINT APRÈS BLOC IF/ELIF POUR SOLVING ANCRE >>>
            print(f"!!!! DEBUG _worker APRÈS BLOC IF/ELIF POUR SOLVING ANCRE (SECTION 1.A) !!!! self.is_mosaic_run = {self.is_mosaic_run}")

            # --- Initialisation grille Drizzle Standard (si applicable pour un run NON-mosaïque) ---
            if self.drizzle_active_session and not self.is_mosaic_run: 
                self.update_progress("DEBUG WORKER: Initialisation grille de sortie pour Drizzle Standard...", "DEBUG_DETAIL")
                if self.reference_wcs_object and hasattr(reference_image_data_for_global_alignment, 'shape'):
                    ref_shape_for_drizzle_grid_hw = reference_image_data_for_global_alignment.shape[:2]
                    try:
                        self.drizzle_output_wcs, self.drizzle_output_shape_hw = self._create_drizzle_output_wcs(
                            self.reference_wcs_object,      
                            ref_shape_for_drizzle_grid_hw,  
                            self.drizzle_scale              
                        )
                        if self.drizzle_output_wcs is None or self.drizzle_output_shape_hw is None:
                            raise RuntimeError("Échec de _create_drizzle_output_wcs (retourne None) pour Drizzle Standard.")
                        print(f"DEBUG QM [_worker]: Grille de sortie Drizzle Standard initialisée: Shape={self.drizzle_output_shape_hw}")
                        self.update_progress(f"   Grille Drizzle Standard prête: {self.drizzle_output_shape_hw}", "INFO")
                    except Exception as e_grid_driz:
                        error_msg_grid = f"Échec critique création grille de sortie Drizzle Standard: {e_grid_driz}"
                        self.update_progress(error_msg_grid, "ERROR"); raise RuntimeError(error_msg_grid)
                else:
                    error_msg_ref_driz = "Référence WCS ou shape de l'image de référence globale manquante pour initialiser la grille Drizzle Standard."
                    self.update_progress(error_msg_ref_driz, "ERROR"); raise RuntimeError(error_msg_ref_driz)
            
            # <<< DEBUG PRINT POST SECTION 1 (après init grille Drizzle si applicable) >>>
            print(f"!!!! DEBUG _worker POST SECTION 1 (après init grille Drizzle si applicable) !!!! self.is_mosaic_run = {self.is_mosaic_run}")
            
            self.update_progress("DEBUG WORKER: Fin Section 1 (Préparation Référence).") # Message plus général
            self.update_progress("⭐ Référence(s) prête(s).", 5); self._recalculate_total_batches()
            


            self.update_progress(f"▶️ Démarrage boucle principale (En file: {self.files_in_queue} | Lots Estimés: {self.total_batches_estimated if self.total_batches_estimated > 0 else '?'})...")

            # ============================================================
            # === SECTION 2 : BOUCLE PRINCIPALE DE TRAITEMENT DES IMAGES ===
            # ============================================================
            iteration_count = 0
            # self.update_progress("DEBUG WORKER: ENTRÉE IMMINENTE DANS LA BOUCLE while not self.stop_processing...") # Peut être un peu verbeux
            
            while not self.stop_processing:
                iteration_count += 1
                
                # <<< LOG PRINCIPAL AU DÉBUT DE CHAQUE ITÉRATION DE LA BOUCLE >>>
                print(f"!!!! DEBUG _worker LOOP START iter {iteration_count}: self.is_mosaic_run = {self.is_mosaic_run}, "
                      f"self.mosaic_alignment_mode = '{self.mosaic_alignment_mode}', "
                      f"self.drizzle_active_session = {self.drizzle_active_session}, "
                      f"self.drizzle_mode = '{self.drizzle_mode}'")
                
                # Log existant (bon à garder)
                print(f"DEBUG QM [_worker V_LoopFocus - Loop Iter]: DÉBUT Itération #{iteration_count}. " 
                      f"Queue approx: {self.queue.qsize()}. "
                      f"Mosaic list AVANT GET: {len(all_aligned_files_with_info_for_mosaic)}")

                file_path = None 
                file_name_for_log = "FichierInconnu" 

                try:
                    file_path = self.queue.get(timeout=1.0) 
                    file_name_for_log = os.path.basename(file_path)
                    print(f"DEBUG QM [_worker V_LoopFocus / Boucle Principale]: Traitement fichier '{file_name_for_log}' depuis la queue.")

                    if path_of_processed_ref_panel_basename and file_name_for_log == path_of_processed_ref_panel_basename:
                        self.update_progress(f"   [WorkerLoop] Panneau d'ancre '{file_name_for_log}' déjà traité. Ignoré dans la boucle principale.")
                        print(f"DEBUG QM [_worker V_LoopFocus]: Panneau d'ancre '{file_name_for_log}' skippé car déjà traité (path_of_processed_ref_panel_basename='{path_of_processed_ref_panel_basename}').")
                        self.processed_files_count += 1 
                        self.queue.task_done()
                        continue 

                    item_result_tuple = None 

                    # <<< DEBUG AVANT LE BLOC IF/ELIF/ELSE D'APPEL À _process_file >>>
                    print(f"  DEBUG _worker (iter {iteration_count}): PRE-CALL _process_file pour '{file_name_for_log}'")
                    print(f"    - use_local_aligner_for_this_mosaic_run: {use_local_aligner_for_this_mosaic_run}")
                    print(f"    - use_astrometry_per_panel_mosaic: {use_astrometry_per_panel_mosaic}")
                    print(f"    - self.is_mosaic_run (juste avant if/elif): {self.is_mosaic_run}")
                    # <<< FIN DEBUG >>>

                    if use_local_aligner_for_this_mosaic_run: 
                        print(f"  DEBUG _worker (iter {iteration_count}): Entrée branche 'use_local_aligner_for_this_mosaic_run' pour _process_file.") # DEBUG
                        item_result_tuple = self._process_file(
                            file_path,
                            reference_image_data_for_global_alignment, 
                            solve_astrometry_for_this_file=False,      
                            fa_orb_features_config=self.fa_orb_features,
                            fa_min_abs_matches_config=self.fa_min_abs_matches,
                            fa_min_ransac_inliers_value_config=self.fa_min_ransac_raw,
                            fa_ransac_thresh_config=self.fa_ransac_thresh,
                            daofind_fwhm_config=self.fa_daofind_fwhm,
                            daofind_threshold_sigma_config=self.fa_daofind_thr_sig,
                            max_stars_to_describe_config=self.fa_max_stars_descr
                        )
                        
                        self.processed_files_count += 1 # Mis ici car _process_file est appelé
                        if item_result_tuple and isinstance(item_result_tuple, tuple) and len(item_result_tuple) == 6 and \
                           item_result_tuple[0] is not None and \
                           item_result_tuple[3] is not None and isinstance(item_result_tuple[3], WCS) and \
                           item_result_tuple[4] is not None: 
                            
                            panel_data, panel_header, _scores, panel_wcs, panel_matrix_m, panel_mask = item_result_tuple
                            all_aligned_files_with_info_for_mosaic.append(
                                (panel_data, panel_header, panel_wcs, panel_matrix_m, panel_mask)
                            )
                            self.aligned_files_count += 1
                            align_method_used_log = panel_header.get('_ALIGN_METHOD_LOG', ('Unknown',None))[0]
                            print(f"  DEBUG QM [_worker / Mosaïque Locale]: Panneau '{file_name_for_log}' traité ({align_method_used_log}) et ajouté à all_aligned_files_with_info_for_mosaic.")
                        else:
                            self.failed_align_count += 1
                            print(f"  DEBUG QM [_worker / Mosaïque Locale]: Échec traitement/alignement panneau '{file_name_for_log}'. _process_file a retourné: {item_result_tuple}")
                            if hasattr(self, '_move_to_unaligned'): self._move_to_unaligned(file_path)

                    elif use_astrometry_per_panel_mosaic: 
                        print(f"  DEBUG _worker (iter {iteration_count}): Entrée branche 'use_astrometry_per_panel_mosaic' pour _process_file.") # DEBUG
                        item_result_tuple = self._process_file(
                            file_path,
                            reference_image_data_for_global_alignment, # Passé mais pas utilisé pour l'alignement direct dans ce mode
                            solve_astrometry_for_this_file=True
                        )
                        self.processed_files_count += 1
                        if item_result_tuple and isinstance(item_result_tuple, tuple) and len(item_result_tuple) == 6 and \
                           item_result_tuple[0] is not None and \
                           item_result_tuple[3] is not None and isinstance(item_result_tuple[3], WCS):
                            
                            panel_data, panel_header, _scores, wcs_object_panel, M_returned, valid_mask_panel = item_result_tuple
                            M_to_store = M_returned if M_returned is not None else np.array([[1.,0.,0.],[0.,1.,0.]], dtype=np.float32)
                            all_aligned_files_with_info_for_mosaic.append(
                                (panel_data, panel_header, wcs_object_panel, M_to_store, valid_mask_panel)
                            )
                            self.aligned_files_count += 1
                            align_method_used_log = panel_header.get('_ALIGN_METHOD_LOG', ('Unknown',None))[0]
                            print(f"  DEBUG QM [_worker / Mosaïque AstroPanel]: Panneau '{file_name_for_log}' traité ({align_method_used_log}) et ajouté à all_aligned_files_with_info_for_mosaic.")
                        else:
                            self.failed_align_count += 1
                            print(f"  DEBUG QM [_worker / Mosaïque AstroPanel]: Échec traitement/alignement panneau '{file_name_for_log}'. _process_file a retourné: {item_result_tuple}")
                            if hasattr(self, '_move_to_unaligned'): self._move_to_unaligned(file_path)

                    else: # Stacking Classique ou Drizzle Standard (non-mosaïque)
                        print(f"  DEBUG _worker (iter {iteration_count}): Entrée branche 'Stacking Classique/Drizzle Standard' pour _process_file.") # DEBUG
                        item_result_tuple = self._process_file( 
                            file_path,
                            reference_image_data_for_global_alignment, 
                            solve_astrometry_for_this_file=False       
                        )
                        self.processed_files_count += 1 
                        if item_result_tuple and isinstance(item_result_tuple, tuple) and len(item_result_tuple) == 6 and \
                           item_result_tuple[0] is not None: 
                            
                            self.aligned_files_count += 1
                            aligned_data, header_orig, scores_val, wcs_gen_val, matrix_M_val, valid_mask_val = item_result_tuple # Déballer les 6
                            
                            if self.drizzle_active_session: # Drizzle Standard (non-mosaïque)
                                print(f"    DEBUG _worker (iter {iteration_count}): Mode Drizzle Standard actif pour '{file_name_for_log}'.")
                                temp_driz_file_path = self._save_drizzle_input_temp(aligned_data, header_orig) 
                                if temp_driz_file_path:
                                    current_batch_items_with_masks_for_stack_batch.append(temp_driz_file_path)
                                else:
                                    self.failed_stack_count +=1 # Échec sauvegarde temp, donc échec pour le stack
                                    print(f"    DEBUG _worker (iter {iteration_count}): Échec _save_drizzle_input_temp pour '{file_name_for_log}'.")
                            else: # Stacking Classique (SUM/W)
                                print(f"    DEBUG _worker (iter {iteration_count}): Mode Stacking Classique pour '{file_name_for_log}'.")
                                classic_stack_item = (aligned_data, header_orig, scores_val, wcs_gen_val, valid_mask_val) # Recréer tuple à 5
                                current_batch_items_with_masks_for_stack_batch.append(classic_stack_item) 
                        else: # _process_file a échoué pour le mode classique/drizzle std
                            self.failed_align_count += 1
                            print(f"  DEBUG QM [_worker / Classique-DrizStd]: Échec _process_file pour '{file_name_for_log}'. Retour: {item_result_tuple}")
                            if hasattr(self, '_move_to_unaligned'): self._move_to_unaligned(file_path)
                        
                        # --- Gestion des lots pour Stacking Classique ou Drizzle Standard ---
                        if len(current_batch_items_with_masks_for_stack_batch) >= self.batch_size and self.batch_size > 0:
                            self.stacked_batches_count += 1
                            print(f"  DEBUG _worker (iter {iteration_count}): Lot complet ({len(current_batch_items_with_masks_for_stack_batch)} images) pour Classique/DrizStd.")
                            if self.drizzle_active_session: # Drizzle Standard Final
                                print(f"    DEBUG _worker: Appel _process_and_save_drizzle_batch (mode Final).")
                                batch_sci_p, batch_wht_p_list = self._process_and_save_drizzle_batch(
                                    current_batch_items_with_masks_for_stack_batch, 
                                    self.drizzle_output_wcs, self.drizzle_output_shape_hw, self.stacked_batches_count
                                )
                                if batch_sci_p and batch_wht_p_list: 
                                    self.intermediate_drizzle_batch_files.append((batch_sci_p, batch_wht_p_list))
                                else: self.failed_stack_count += len(current_batch_items_with_masks_for_stack_batch)
                            else: # Stacking Classique
                                print(f"    DEBUG _worker: Appel _process_completed_batch (mode Classique SUM/W).")
                                self._process_completed_batch(
                                    current_batch_items_with_masks_for_stack_batch, 
                                    self.stacked_batches_count, self.total_batches_estimated
                                )
                            current_batch_items_with_masks_for_stack_batch = [] # Vider le lot

                    self.queue.task_done()
                except Empty:
                    # --- NOUVELLE LOGIQUE POUR GÉRER LES DOSSIERS ADDITIONNELS (DÉBUT) ---
                    print(f"DEBUG QM [_worker / EmptyExcept]: Queue vide. Vérification des dossiers additionnels.")
                    new_files_added_from_additional_folder = 0
                    folder_to_process_from_additional = None

                    # Protéger l'accès à self.additional_folders avec le Lock
                    with self.folders_lock:
                        if self.additional_folders: # Si des dossiers additionnels sont en attente
                            folder_to_process_from_additional = self.additional_folders.pop(0) # Prendre le premier et le retirer
                            print(f"DEBUG QM [_worker / EmptyExcept]: Dossier additionnel trouvé et retiré: '{os.path.basename(folder_to_process_from_additional)}'.")
                            # Mettre à jour le statut dans l'UI immédiatement (même si pas de fichiers à l'intérieur)
                            self.update_progress(f"🔍 Scan du dossier additionnel: {os.path.basename(folder_to_process_from_additional)}...", None)
                        else:
                            print(f"DEBUG QM [_worker / EmptyExcept]: self.additional_folders est vide (pas de dossiers additionnels en attente).")

                    if folder_to_process_from_additional:
                        # Mettre à jour self.current_folder pour que les logs d'erreurs éventuelles soient pertinents
                        self.current_folder = folder_to_process_from_additional
                        new_files_added_from_additional_folder = self._add_files_to_queue(folder_to_process_from_additional)
                        print(f"DEBUG QM [_worker / EmptyExcept]: {new_files_added_from_additional_folder} nouveaux fichiers ajoutés de '{os.path.basename(folder_to_process_from_additional)}'.")
                        
                        # Notifier le GUI que le nombre de dossiers additionnels a diminué
                        # (La mise à jour de l'affichage du nombre de dossiers dans l'UI via le callback)
                        self.update_progress(f"folder_count_update:{len(self.additional_folders)}")

                        if new_files_added_from_additional_folder > 0:
                            # Si de nouveaux fichiers ont été ajoutés, on continue la boucle
                            # et la queue sera traitée à la prochaine itération.
                            print(f"DEBUG QM [_worker / EmptyExcept]: Nouveaux fichiers détectés, continuer la boucle.")
                            continue # <-- CRUCIAL: Retourne au début de la boucle while pour traiter les nouveaux fichiers
                        else:
                            # Si le dossier additionnel était vide de FITS, on log l'info.
                            self.update_progress(f"   ℹ️ Dossier '{os.path.basename(folder_to_process_from_additional)}' ne contient aucun fichier FITS à traiter. Passons au suivant ou finalisons.")
                            print(f"DEBUG QM [_worker / EmptyExcept]: Dossier additionnel vide, pas de nouveaux fichiers à traiter.")
                            # Si le dossier additionnel ne contenait pas de fichiers FITS, la queue reste vide.
                            # On laisse la logique de fin de traitement prendre le relais à la prochaine itération.
                            # Pas de 'continue' ici, pour permettre l'évaluation de la condition finale de sortie.
                            pass 

                    # Si aucun dossier additionnel n'a été trouvé OU si le dossier trouvé était vide de FITS
                    # (et qu'on est arrivé ici sans 'continue' précédent)
                    if not self.additional_folders and self.queue.empty(): 
                        self.update_progress("INFO: Plus aucun fichier ni dossier supplémentaire. Fin de la boucle de traitement.", None)
                        print(f"DEBUG QM [_worker / EmptyExcept]: Condition de sortie (self.additional_folders et queue vides) remplie. BREAK.")
                        break # <-- CRUCIAL: Sortie normale de la boucle while
                    else:
                        # Si self.additional_folders n'est PAS vide (même après le pop d'un élément, d'autres ont pu être ajoutés à la volée),
                        # ou si la queue n'est pas vide (si _add_files_to_queue a réussi),
                        # alors on devrait continuer. Si on est ici, la queue est vide.
                        # Cela signifie que self.additional_folders doit avoir des éléments pour que la boucle continue.
                        # Sinon, c'est une boucle infinie si on arrive ici sans `break` ou `continue` et que la queue est vide.
                        # Un `time.sleep` est alors nécessaire pour éviter le CPU à 100%.
                        self.update_progress("INFO: File d'attente vide, en attente de nouveaux ...", None)
                        print(f"DEBUG QM [_worker / EmptyExcept]: Queue vide. self.additional_folders n'est PAS vide (il reste des dossiers à traiter), OU un 'continue' a été manqué. Sleep et revérification...")
                        time.sleep(0.5) # Attendre un peu avant de refaire un `get` (pour éviter boucle serrée)
                        continue # <-- CRUCIAL: Retourne au début de la boucle `while` pour re-tenter de prendre un item ou un autre dossier additionnel
                    # --- NOUVELLE LOGIQUE POUR GÉRER LES DOSSIERS ADDITIONNELS (FIN) ---

            # ==============================================================
            # === SECTION 3 : TRAITEMENT FINAL APRÈS LA BOUCLE PRINCIPALE ===
            # ==============================================================
            print(f"DEBUG QM [_worker V_DrizIncrTrue_Fix1 / FIN DE BOUCLE WHILE]:") # Version Log
            print(f"  >> self.stop_processing est: {self.stop_processing}")
            print(f"  >> Taille de all_aligned_files_with_info_for_mosaic IMMÉDIATEMENT APRÈS LA BOUCLE WHILE: {len(all_aligned_files_with_info_for_mosaic)}")
            if all_aligned_files_with_info_for_mosaic: 
                print(f"  >> Premier item (pour vérif type): {type(all_aligned_files_with_info_for_mosaic[0])}, len: {len(all_aligned_files_with_info_for_mosaic[0]) if isinstance(all_aligned_files_with_info_for_mosaic[0], tuple) else 'N/A'}")

            print(f"DEBUG QM [_worker V_DrizIncrTrue_Fix1]: Sortie de la boucle principale. Début de la phase de finalisation...")
            print(f"  ÉTAT FINAL AVANT BLOC if/elif/else de finalisation:")
            print(f"    - self.stop_processing: {self.stop_processing}")
            print(f"    - self.is_mosaic_run: {self.is_mosaic_run}")
            if self.is_mosaic_run: print(f"      - Mode align.: '{self.mosaic_alignment_mode}', Nb items mosaïque: {len(all_aligned_files_with_info_for_mosaic)}")
            print(f"    - self.drizzle_active_session (std): {self.drizzle_active_session}")
            if self.drizzle_active_session and not self.is_mosaic_run: print(f"      - Mode Drizzle (std): '{self.drizzle_mode}', Nb lots Drizzle interm.: {len(self.intermediate_drizzle_batch_files)}")
            print(f"    - self.images_in_cumulative_stack (classique/DrizIncrVRAI): {self.images_in_cumulative_stack}") 
            print(f"    - current_batch_items_with_masks_for_stack_batch (non traité si dernier lot partiel): {len(current_batch_items_with_masks_for_stack_batch)}")

            print("DEBUG QM [_worker V_DrizIncrTrue_Fix1]: *** JUSTE AVANT LE PREMIER 'if self.stop_processing:' ***")

            if self.stop_processing:
                print("DEBUG QM [_worker V_DrizIncrTrue_Fix1]: *** ENTRÉE DANS 'if self.stop_processing:' ***")
                self.update_progress("⛔ Traitement interrompu par l'utilisateur ou erreur.")
                if self.processing_error:
                    self.update_progress(f"   Cause: {self.processing_error}")
                
                # Logique de sauvegarde partielle
                if self.drizzle_active_session and self.drizzle_mode == "Incremental" and \
                   hasattr(self, 'incremental_drizzle_objects') and self.incremental_drizzle_objects and \
                   self.images_in_cumulative_stack > 0: # Vérifier si Drizzle Incr. VRAI a des données
                    self.update_progress("   Sauvegarde du stack Drizzle Incrémental VRAI partiel...")
                    self._save_final_stack(output_filename_suffix="_drizzle_incr_true_stopped", stopped_early=True)
                elif not self.is_mosaic_run and not self.drizzle_active_session and \
                     hasattr(self, 'cumulative_sum_memmap') and self.cumulative_sum_memmap is not None and \
                     self.images_in_cumulative_stack > 0: # Stacking Classique SUM/W
                    self.update_progress("   Sauvegarde du stack classique partiel (SUM/W)...")
                    self._save_final_stack(output_filename_suffix="_classic_stopped", stopped_early=True)
                else:
                    self.update_progress("   Aucun stack partiel significatif à sauvegarder.")

            # --- MODE MOSAÏQUE ---
            elif self.is_mosaic_run:
                print("DEBUG QM [_worker V_DrizIncrTrue_Fix1]: *** ENTRÉE DANS 'elif self.is_mosaic_run:' ***")
                # ... (logique mosaïque inchangée, elle appelle _finalize_mosaic_processing qui appelle _save_final_stack
                #      en passant drizzle_final_sci_data, donc c'est géré par la branche correspondante dans _save_final_stack)
                self.update_progress("🏁 Finalisation Mosaïque...")
                if not all_aligned_files_with_info_for_mosaic: 
                    self.update_progress("   ❌ Mosaïque: Aucun panneau aligné pour l'assemblage.", "ERROR")
                    self.processing_error = "Mosaïque: Aucun panneau aligné"; self.final_stacked_path = None
                else:
                    try:
                        self._finalize_mosaic_processing(all_aligned_files_with_info_for_mosaic)
                    except Exception as e_finalize_mosaic:
                        # ... (gestion erreur identique)
                        error_msg = f"Erreur CRITIQUE durant finalisation mosaïque: {e_finalize_mosaic}"
                        print(f"ERREUR QM [_worker V_DrizIncrTrue_Fix1]: {error_msg}"); traceback.print_exc(limit=3)
                        self.update_progress(f"   ❌ {error_msg}", "ERROR")
                        self.processing_error = error_msg; self.final_stacked_path = None
            
            # --- MODE DRIZZLE STANDARD (NON-MOSAÏQUE) ---
            elif self.drizzle_active_session: 
                print("DEBUG QM [_worker V_DrizIncrTrue_Fix1]: *** ENTRÉE DANS 'elif self.drizzle_active_session:' (NON-MOSAÏQUE) ***")
                print(f"DEBUG QM [_worker/Finalize DrizzleStd]: Mode Drizzle Standard: {self.drizzle_mode}")

                if current_batch_items_with_masks_for_stack_batch: 
                    self.stacked_batches_count += 1
                    num_in_partial_batch = len(current_batch_items_with_masks_for_stack_batch)
                    progress_info_partial_log = f"(Lot PARTIEL {self.stacked_batches_count}/{self.total_batches_estimated if self.total_batches_estimated > 0 else '?'})"
                    
                    if self.drizzle_mode == "Final":
                        self.update_progress(f"💧 Traitement Drizzle (mode Final) du dernier lot partiel {progress_info_partial_log}...")
                        batch_sci_path, batch_wht_paths = self._process_and_save_drizzle_batch(
                            current_batch_items_with_masks_for_stack_batch, # Liste de CHEMINS
                            self.drizzle_output_wcs, self.drizzle_output_shape_hw, self.stacked_batches_count
                        )
                        if batch_sci_path and batch_wht_paths: 
                            self.intermediate_drizzle_batch_files.append((batch_sci_path, batch_wht_paths))
                        else: self.failed_stack_count += len(current_batch_items_with_masks_for_stack_batch)
                    
                    elif self.drizzle_mode == "Incremental": # VRAI Drizzle Incrémental
                        self.update_progress(f"💧 Traitement Drizzle Incr. VRAI du dernier lot partiel {progress_info_partial_log}...")
                        self._process_incremental_drizzle_batch( # Utilise la version V_True_Incremental_Driz
                            current_batch_items_with_masks_for_stack_batch, # Liste de CHEMINS
                            self.stacked_batches_count, self.total_batches_estimated
                        )
                    current_batch_items_with_masks_for_stack_batch = []
                
                # --- Sauvegarde finale spécifique au mode Drizzle ---
                if self.drizzle_mode == "Incremental":
                    self.update_progress("🏁 Finalisation Drizzle Incrémental VRAI (depuis objets Drizzle)...")
                    # Pour le VRAI Drizzle Incrémental, _save_final_stack doit lire depuis
                    # self.incremental_drizzle_objects/arrays. Ne pas passer drizzle_final_sci_data.
                    self._save_final_stack(output_filename_suffix="_drizzle_incr_true") # MODIFIÉ ICI
                
                elif self.drizzle_mode == "Final":
                    self.update_progress("🏁 Combinaison finale des lots Drizzle (Mode Final)...")
                    if not self.intermediate_drizzle_batch_files:
                        self.update_progress("   ❌ Drizzle Final: Aucun lot intermédiaire à combiner.", None)
                        self.processing_error = "Drizzle Final: Aucun lot intermédiaire"; self.final_stacked_path = None
                    else:
                        final_drizzle_sci_hxwxc, final_drizzle_wht_hxwxc = self._combine_intermediate_drizzle_batches(
                            self.intermediate_drizzle_batch_files,
                            self.drizzle_output_wcs, self.drizzle_output_shape_hw  
                        )
                        if final_drizzle_sci_hxwxc is not None:
                            self.update_progress("   Drizzle Final combiné. Préparation sauvegarde...")
                            self._save_final_stack(output_filename_suffix="_drizzle_final", # Suffixe correct
                                                   drizzle_final_sci_data=final_drizzle_sci_hxwxc,
                                                   drizzle_final_wht_data=final_drizzle_wht_hxwxc)
                        else:
                            self.update_progress("   ❌ Échec combinaison finale des lots Drizzle (résultat vide).", None)
                            self.processing_error = "Échec combinaison Drizzle Final"; self.final_stacked_path = None
            
            # --- MODE STACKING CLASSIQUE (NON-MOSAÏQUE, NON-DRIZZLE) ---
            elif not self.is_mosaic_run and not self.drizzle_active_session: 
                # ... (logique inchangée pour stacking classique) ...
                print("DEBUG QM [_worker V_DrizIncrTrue_Fix1]: *** ENTRÉE DANS 'elif not self.is_mosaic_run and not self.drizzle_active_session:' (CLASSIQUE) ***")
                if current_batch_items_with_masks_for_stack_batch: 
                    self.stacked_batches_count += 1
                    self.update_progress(f"⚙️ Traitement classique du dernier lot partiel ({len(current_batch_items_with_masks_for_stack_batch)} images)...")
                    self._process_completed_batch(
                        current_batch_items_with_masks_for_stack_batch, 
                        self.stacked_batches_count, self.total_batches_estimated
                    )
                    current_batch_items_with_masks_for_stack_batch = []
                self.update_progress("🏁 Finalisation Stacking Classique (SUM/W)...")
                if self.images_in_cumulative_stack > 0 or (hasattr(self, 'cumulative_sum_memmap') and self.cumulative_sum_memmap is not None):
                    self._save_final_stack(output_filename_suffix="_classic_sumw")
                else:
                    self.update_progress("   Aucune image accumulée dans le stack classique. Sauvegarde ignorée.")
                    self.final_stacked_path = None 
            else: # Cas imprévu
                print("DEBUG QM [_worker V_DrizIncrTrue_Fix1]: *** ENTRÉE DANS LE 'else' FINAL (ÉTAT NON GÉRÉ) ***")
                self.update_progress("⚠️ État de finalisation non géré. Aucune action de sauvegarde principale.")
                self.processing_error = "État de finalisation non géré."; self.final_stacked_path = None

            print("DEBUG QM [_worker V_DrizIncrTrue_Fix1]: *** APRÈS LE BLOC if/elif/else DE FINALISATION ***")




        # --- FIN DU BLOC TRY PRINCIPAL DU WORKER ---
        except RuntimeError as rte: 
            self.update_progress(f"❌ ERREUR CRITIQUE (RuntimeError) dans le worker: {rte}", "ERROR") # S'assurer que "ERROR" est passé pour le log GUI
            print(f"ERREUR QM [_worker V5.3.2_AstroPerPanelFix]: RuntimeError: {rte}"); traceback.print_exc(limit=3)
            self.processing_error = f"RuntimeError: {rte}"
            self.stop_processing = True # Provoquer l'arrêt propre du thread
        except Exception as e_global_worker: 
            self.update_progress(f"❌ ERREUR INATTENDUE GLOBALE dans le worker: {e_global_worker}", "ERROR")
            print(f"ERREUR QM [_worker V5.3.2_AstroPerPanelFix]: Exception Globale: {e_global_worker}"); traceback.print_exc(limit=3)
            self.processing_error = f"Erreur Globale: {e_global_worker}"
            self.stop_processing = True # Provoquer l'arrêt propre du thread
        finally:
            print(f"DEBUG QM [_worker V5.3.2_AstroPerPanelFix]: Entrée dans le bloc FINALLY principal du worker.")
            if hasattr(self, 'cumulative_sum_memmap') and self.cumulative_sum_memmap is not None \
               or hasattr(self, 'cumulative_wht_memmap') and self.cumulative_wht_memmap is not None:
                self._close_memmaps()
            
            if self.perform_cleanup:
                self.update_progress("🧹 Nettoyage final des fichiers temporaires...")
                self._cleanup_drizzle_temp_files()        # Dossier des inputs Drizzle (aligned_input_*.fits)
                self._cleanup_drizzle_batch_outputs()   # Dossier des sorties Drizzle par lot (batch_*_sci.fits, batch_*_wht_*.fits)
                self._cleanup_mosaic_panel_stacks_temp()# Dossier des stacks de panneaux (si ancienne logique ou tests)
                self.cleanup_temp_reference()           # Fichiers reference_image.fit/png
            
            self.processing_active = False
            self.stop_processing_flag_for_gui = self.stop_processing # Transmettre l'état d'arrêt à l'UI
            gc.collect()
            print(f"DEBUG QM [_worker V5.3.2_AstroPerPanelFix]: Fin du bloc FINALLY principal. Flag processing_active mis à False.")
            self.update_progress("🚪 Thread de traitement principal terminé.")







############################################################################################################################








# --- DANS LA CLASSE SeestarQueuedStacker DANS seestar/queuep/queue_manager.py ---

    def _generate_and_save_mosaic_alignment_log(self, 
                                                all_aligned_panel_info_list: list, 
                                                anchor_wcs_details: dict,        
                                                final_output_grid_details: dict
                                                ):
        """
        Génère un log détaillé sur l'alignement de la mosaïque et le sauvegarde.
        MODIFIED V2: Gestion plus robuste de la lecture de _ALIGN_METHOD_LOG depuis le header.
        """
        if not self.output_folder:
            print("WARN QM [_generate_mosaic_log V2]: Output folder non défini, log non sauvegardé.")
            return

        log_lines = []
        separator = "=" * 70
        
        log_lines.append(f"{separator}\nRAPPORT D'ALIGNEMENT DE MOSAÏQUE (V2)\n{separator}")
        log_lines.append(f"Date du rapport: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_lines.append(f"Dossier de sortie: {self.output_folder}")

        # --- 1. Paramètres Clés de la Mosaïque ---
        log_lines.append(f"\n--- Paramètres de la Mosaïque Utilisés ---")
        log_lines.append(f"  Mode d'alignement: {getattr(self, 'mosaic_alignment_mode', 'N/A')}")
        log_lines.append(f"  Fallback WCS activé: {getattr(self, 'use_wcs_fallback_for_mosaic', 'N/A')}")
        log_lines.append(f"  FastAligner - Seuil RANSAC: {getattr(self, 'fa_ransac_thresh', 'N/A')}")
        log_lines.append(f"  FastAligner - Min Abs Matches: {getattr(self, 'fa_min_abs_matches', 'N/A')}")
        log_lines.append(f"  FastAligner - Min RANSAC Raw (valeur config): {getattr(self, 'fa_min_ransac_raw', 'N/A')}")
        log_lines.append(f"  FastAligner - ORB Features (cible): {getattr(self, 'fa_orb_features', 'N/A')}")
        log_lines.append(f"  FastAligner DAO - FWHM: {getattr(self, 'fa_daofind_fwhm', 'N/A')}")
        log_lines.append(f"  FastAligner DAO - Seuil Sigma Factor: {getattr(self, 'fa_daofind_thr_sig', 'N/A')}")
        log_lines.append(f"  FastAligner DAO - Max Étoiles Desc.: {getattr(self, 'fa_max_stars_descr', 'N/A')}")
        log_lines.append(f"  Drizzle Mosaïque - Kernel: {getattr(self, 'mosaic_drizzle_kernel', 'N/A')}")
        log_lines.append(f"  Drizzle Mosaïque - Pixfrac: {getattr(self, 'mosaic_drizzle_pixfrac', 'N/A')}")
        log_lines.append(f"  Drizzle Mosaïque - WHT Threshold: {getattr(self, 'mosaic_drizzle_wht_threshold', 'N/A')}")
        log_lines.append(f"  Drizzle Mosaïque - Échelle Globale Appliquée: {getattr(self, 'drizzle_scale', 'N/A')}x")

        # --- 2. Informations sur le WCS du Panneau d'Ancrage ---
        log_lines.append(f"\n--- WCS du Panneau d'Ancrage ---")
        if anchor_wcs_details:
            log_lines.append(f"  Fichier Source Ancre: {anchor_wcs_details.get('source_file', 'N/A')}")
            log_lines.append(f"  Type de WCS: {anchor_wcs_details.get('type', 'N/A')}")
            log_lines.append(f"  CRVAL (RA, Dec): {anchor_wcs_details.get('crval', 'N/A')}")
            log_lines.append(f"  CRPIX (X, Y): {anchor_wcs_details.get('crpix', 'N/A')}")
            log_lines.append(f"  Échelle (arcsec/pix): {anchor_wcs_details.get('scale_arcsec_pix', 'N/A')}")
            log_lines.append(f"  Shape Pixel WCS (W,H): {anchor_wcs_details.get('pixel_shape_wh', 'N/A')}")
            log_lines.append(f"  Distorsion SIP présente: {anchor_wcs_details.get('sip', 'N/A')}")
            log_lines.append(f"  Info Solveur AN_SOLVED: {anchor_wcs_details.get('AN_SOLVED', 'N/A')}")
            log_lines.append(f"  Info Solveur AN_FIELD_SCALE_ASEC: {anchor_wcs_details.get('AN_FIELD_SCALE_ASEC', 'N/A')}")
        else:
            log_lines.append("  Informations sur le WCS de l'ancre non disponibles.")

        # --- 3. Résumé de l'Alignement pour Chaque Panneau ---
        log_lines.append(f"\n--- Détails de l'Alignement des Panneaux (par rapport à l'ancre) ---")
        num_panneaux_pour_alignement_relatif = 0 # Panneaux autres que l'ancre
        num_fastalign_succes = 0
        num_fallback_wcs_tentatives = 0 # Combien de fois le fallback a été tenté
        num_fallback_wcs_succes = 0
        num_align_echecs_complets = 0

        if not all_aligned_panel_info_list:
             log_lines.append("  Aucun panneau (même pas l'ancre) n'a été collecté pour la mosaïque.")
        else:
            for idx, panel_info in enumerate(all_aligned_panel_info_list):
                if not isinstance(panel_info, tuple) or len(panel_info) < 4: 
                    log_lines.append(f"  Panneau {idx}: Format d'information invalide. Ignoré.")
                    continue 
                
                # panel_info = (image_data_orig, header, wcs_ANCRE_POUR_M, matrix_M, valid_mask)
                panel_header = panel_info[1]
                panel_filename_tuple = panel_header.get('_SRCFILE', (f"Panneau_{idx}_NomInconnu", ""))
                panel_filename = panel_filename_tuple[0] if isinstance(panel_filename_tuple, tuple) else str(panel_filename_tuple)
                
                matrix_m = panel_info[3]
                
                log_lines.append(f"  Panneau {idx+1}/{len(all_aligned_panel_info_list)}: {panel_filename}") # Afficher 1-based
                
                if idx == 0 and panel_filename == anchor_wcs_details.get('source_file', ''): # Identification plus robuste de l'ancre
                    log_lines.append(f"    -> Rôle: Ancre de la mosaïque.")
                    log_lines.append(f"    -> Matrice M (normalement identité pour ancre): \n{matrix_m}")
                else: # Panneaux non-ancre
                    num_panneaux_pour_alignement_relatif +=1
                    # Lire la méthode d'alignement depuis le header du panneau
                    align_method_from_header_raw = panel_header.get('_ALIGN_METHOD_LOG', 'Non_Loggué')
                    align_method_from_header = align_method_from_header_raw[0] if isinstance(align_method_from_header_raw, tuple) else str(align_method_from_header_raw)

                    log_lines.append(f"    -> Méthode d'alignement (logguée): {align_method_from_header}")
                    log_lines.append(f"    -> Matrice M calculée vers l'ancre: \n{matrix_m}")

                    if align_method_from_header == 'FastAligner_Success':
                        num_fastalign_succes +=1
                    elif align_method_from_header == 'WCS_Fallback_Success':
                        num_fallback_wcs_succes +=1
                        num_fallback_wcs_tentatives +=1 
                    elif align_method_from_header == 'FastAligner_Fail_Then_Fallback_Fail':
                        num_fallback_wcs_tentatives +=1
                        num_align_echecs_complets +=1
                    elif align_method_from_header == 'FastAligner_Fail_No_Fallback':
                        num_align_echecs_complets +=1
                    elif align_method_from_header == 'Alignment_Failed_Fully': # Cas générique d'échec
                        num_align_echecs_complets +=1
            
        log_lines.append(f"\n  Résumé Alignement des Panneaux (pour {num_panneaux_pour_alignement_relatif} panneaux relatifs à l'ancre):")
        log_lines.append(f"    - Succès FastAligner: {num_fastalign_succes}")
        log_lines.append(f"    - Tentatives de Fallback WCS (après échec FastAligner): {num_fallback_wcs_tentatives}")
        log_lines.append(f"    - Succès Fallback WCS: {num_fallback_wcs_succes}")
        log_lines.append(f"    - Échecs Complets d'Alignement (ni FastAligner, ni Fallback): {num_align_echecs_complets}")
        total_aligned_relatifs = num_fastalign_succes + num_fallback_wcs_succes
        log_lines.append(f"    - Total Panneaux Relatifs Alignés (FastAligner ou Fallback): {total_aligned_relatifs}")


        # --- 4. Informations sur la Grille de Sortie Finale ---
        log_lines.append(f"\n--- Grille de Sortie Finale de la Mosaïque ---")
        if final_output_grid_details:
            log_lines.append(f"  Shape (Hauteur, Largeur): {final_output_grid_details.get('shape_hw', 'N/A')}")
            log_lines.append(f"  WCS CRVAL (RA, Dec): {final_output_grid_details.get('crval', 'N/A')}")
            log_lines.append(f"  WCS CRPIX (X, Y): {final_output_grid_details.get('crpix', 'N/A')}")
            log_lines.append(f"  WCS Échelle (arcsec/pix): {final_output_grid_details.get('scale_arcsec_pix', 'N/A')}")
        else:
            log_lines.append("  Informations sur la grille de sortie non disponibles (probablement car assemblage annulé).")

        # --- 5. Résumé de l'Assemblage Drizzle ---
        log_lines.append(f"\n--- Assemblage Drizzle ---")
        log_lines.append(f"  Nombre total de panneaux (ancre + alignés) fournis à DrizzleProcessor: {len(all_aligned_panel_info_list)}")
        # On pourrait ajouter plus d'infos si DrizzleProcessor retournait des stats d'assemblage

        # --- 6. Compteurs Généraux du Traitement (depuis l'instance QueuedStacker) ---
        log_lines.append(f"\n--- Compteurs Généraux du Traitement (depuis QueuedStacker) ---")
        log_lines.append(f"  Fichiers traités au total par le worker (tentatives): {getattr(self, 'processed_files_count', 0)}")
        log_lines.append(f"  Panneaux retenus pour la mosaïque (attribut 'aligned_files_count'): {getattr(self, 'aligned_files_count', 0)}")
        log_lines.append(f"  Échecs d'alignement comptabilisés par QueuedStacker: {getattr(self, 'failed_align_count', 0)}")
        log_lines.append(f"  Fichiers skippés (autres raisons, ex: faible variance ref): {getattr(self, 'skipped_files_count', 0)}")
        
        log_lines.append(f"\n{separator}\nFIN DU RAPPORT\n{separator}")

        log_filename = "rapport_alignement_mosaique.txt"
        log_filepath = os.path.join(self.output_folder, log_filename)
        try:
            with open(log_filepath, 'w', encoding='utf-8') as f_log:
                for line in log_lines:
                    f_log.write(line + "\n")
            self.update_progress(f"📄 Rapport d'alignement mosaïque sauvegardé: {log_filename}", None)
            print(f"DEBUG QM: Rapport d'alignement mosaïque V2 sauvegardé dans '{log_filepath}'")
        except Exception as e_save_log:
            self.update_progress(f"⚠️ Erreur sauvegarde rapport d'alignement mosaïque V2: {e_save_log}", None)
            print(f"ERREUR QM: Échec sauvegarde rapport alignement mosaïque V2: {e_save_log}")







#####################################################################################################################################################


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
        Effectue la combinaison finale pour le mode mosaïque en utilisant reproject.
        MODIFIED: Removed 'progress_bar=True' from reproject_and_coadd call to fix TypeError.
                  TQDM might be used by default by reproject if installed.
        Version: V_FinalizeMosaic_ReprojectCoadd_4_FixTqdmCall
        """
        num_panels = len(aligned_files_info_list) 
        print(f"DEBUG (Backend _finalize_mosaic_processing V_FinalizeMosaic_ReprojectCoadd_4_FixTqdmCall): Début finalisation pour {num_panels} panneaux avec reproject.")
        self.update_progress(f"🖼️ Préparation assemblage mosaïque final ({num_panels} images) avec reproject...")

        if num_panels < 1: 
            self.update_progress("⚠️ Moins de 1 panneau aligné disponible pour la mosaïque. Traitement annulé.")
            self.final_stacked_path = None; self.processing_error = "Mosaïque: Moins de 1 panneau aligné"; return
        
        try:
            from reproject.mosaicking import reproject_and_coadd
            from reproject import reproject_interp 
        except ImportError:
            error_msg = "Bibliothèque reproject non disponible pour l'assemblage mosaïque."
            self.update_progress(f"❌ {error_msg}"); self.processing_error = error_msg; self.final_stacked_path = None; return

        input_data_for_reproject = []; input_footprints_for_reproject = []; all_wcs_for_grid_calc = []

        print(f"  -> Préparation des {num_panels} panneaux pour reproject_and_coadd...")
        for i_panel_loop, panel_info_tuple_local in enumerate(aligned_files_info_list):
            try:
                panel_image_data_HWC_orig, panel_header_orig, wcs_for_panel_input, _transform_matrix_M_panel, _pixel_mask_2d_bool = panel_info_tuple_local
            except (TypeError, ValueError) as e_unpack:
                self.update_progress(f"    -> ERREUR déballage tuple panneau {i_panel_loop+1}: {e_unpack}. Ignoré.", "ERROR")
                print(f"ERREUR QM [_finalize_mosaic_processing]: Déballage tuple panneau {i_panel_loop+1}"); continue

            original_filename_for_log = panel_header_orig.get('_SRCFILE', (f"Panel_{i_panel_loop+1}", ""))[0]
            print(f"    Processing panel {i_panel_loop+1}/{num_panels}: {original_filename_for_log}")

            if panel_image_data_HWC_orig is None or wcs_for_panel_input is None:
                self.update_progress(f"    -> Panneau {i_panel_loop+1} ('{original_filename_for_log}'): Données ou WCS manquantes. Ignoré.", "WARN"); continue
            
            current_panel_shape_hw = panel_image_data_HWC_orig.shape[:2]
            footprint_panel = None
            if _pixel_mask_2d_bool is not None and _pixel_mask_2d_bool.shape == current_panel_shape_hw:
                footprint_panel = np.clip(_pixel_mask_2d_bool.astype(np.float32), 0.0, 1.0) 
                print(f"      Panel {i_panel_loop+1}: Using provided pixel mask as footprint. Sum: {np.sum(footprint_panel)}")
            else:
                self.update_progress(f"      WARN: Panneau {i_panel_loop+1}, masque de pixels invalide ou manquant. Utilisation d'un footprint complet (np.ones).")
                footprint_panel = np.ones(current_panel_shape_hw, dtype=np.float32)
            
            input_data_for_reproject.append((panel_image_data_HWC_orig, wcs_for_panel_input))
            input_footprints_for_reproject.append(footprint_panel)
            all_wcs_for_grid_calc.append(wcs_for_panel_input)

        if not input_data_for_reproject:
            self.update_progress("❌ Mosaïque: Aucun panneau valide à traiter avec reproject. Traitement annulé.", "ERROR")
            self.final_stacked_path = None; self.processing_error = "Mosaïque: Aucun panneau valide pour reproject"; return

        print("DEBUG (Backend _finalize_mosaic_processing): Appel _calculate_final_mosaic_grid pour reproject...")
        output_wcs, output_shape_hw = self._calculate_final_mosaic_grid(all_wcs_for_grid_calc)

        if output_wcs is None or output_shape_hw is None:
            error_msg = "Échec calcul grille de sortie pour la mosaïque avec reproject."
            self.update_progress(f"❌ {error_msg}", "ERROR"); self.processing_error = error_msg; self.final_stacked_path = None; return
        print(f"DEBUG (Backend _finalize_mosaic_processing): Grille Mosaïque pour reproject calculée -> Shape={output_shape_hw} (H,W), WCS CRVAL={output_wcs.wcs.crval if output_wcs.wcs else 'N/A'}")

        final_mosaic_sci_channels = []; final_mosaic_coverage_channels = [] 
        num_color_channels_expected = 3 

        print(f"  -> Exécution de reproject_and_coadd par canal (pour {num_color_channels_expected} canaux)...")
        total_reproject_time_sec = 0.0
        
        progress_base_finalize = 70 
        progress_range_reproject_step = 25 

        for i_ch in range(num_color_channels_expected):
            gui_progress_before_channel = progress_base_finalize + int(progress_range_reproject_step * (i_ch / num_color_channels_expected))
            self.update_progress(f"   Reprojection et combinaison du canal {i_ch+1}/{num_color_channels_expected}...", gui_progress_before_channel)
            
            channel_arrays_wcs_list = []
            channel_footprints_list = []

            for panel_data_tuple, panel_footprint in zip(input_data_for_reproject, input_footprints_for_reproject):
                panel_hwc_data, panel_wcs = panel_data_tuple
                if panel_hwc_data.ndim == 3 and panel_hwc_data.shape[2] == num_color_channels_expected:
                    channel_arrays_wcs_list.append( (panel_hwc_data[..., i_ch], panel_wcs) )
                    channel_footprints_list.append(panel_footprint) 
                elif panel_hwc_data.ndim == 2 and i_ch == 0 : 
                     channel_arrays_wcs_list.append( (panel_hwc_data, panel_wcs) )
                     channel_footprints_list.append(panel_footprint)
                elif panel_hwc_data.ndim == 2 and i_ch > 0: 
                    continue 
            
            if not channel_arrays_wcs_list:
                self.update_progress(f"    Aucune donnée pour le canal {i_ch+1}. Ce canal sera vide.", "WARN")
                final_mosaic_sci_channels.append(np.zeros(output_shape_hw, dtype=np.float32))
                final_mosaic_coverage_channels.append(np.zeros(output_shape_hw, dtype=np.float32))
                continue

            try:
                print(f"    Appel reproject_and_coadd pour canal {i_ch+1}. Nombre d'images pour ce canal: {len(channel_arrays_wcs_list)}")
                start_time_reproject_ch = time.monotonic()
                
                # Removed progress_bar=True from this call
                mosaic_channel_sci, mosaic_channel_coverage = reproject_and_coadd(
                    channel_arrays_wcs_list,
                    output_projection=output_wcs,
                    shape_out=output_shape_hw,
                    input_weights=channel_footprints_list, 
                    reproject_function=reproject_interp, 
                    combine_function='mean', 
                    match_background=True, 
                    block_size=getattr(self, 'reproject_block_size_xy', (256,256))
                )
                
                end_time_reproject_ch = time.monotonic()
                duration_reproject_ch_sec = end_time_reproject_ch - start_time_reproject_ch
                total_reproject_time_sec += duration_reproject_ch_sec

                final_mosaic_sci_channels.append(mosaic_channel_sci.astype(np.float32))
                final_mosaic_coverage_channels.append(mosaic_channel_coverage.astype(np.float32))
                
                log_msg_time_console = f"    Canal {i_ch+1} traité en {duration_reproject_ch_sec:.2f} secondes. Shape SCI: {mosaic_channel_sci.shape}, Shape Coverage: {mosaic_channel_coverage.shape}"
                print(log_msg_time_console)
                self.update_progress(f"   Canal {i_ch+1}/{num_color_channels_expected} combiné.")

            except Exception as e_reproject:
                error_msg = f"Erreur durant reproject_and_coadd pour canal {i_ch+1}: {e_reproject}"
                self.update_progress(f"❌ {error_msg}", "ERROR"); traceback.print_exc(limit=3)
                final_mosaic_sci_channels.append(np.zeros(output_shape_hw, dtype=np.float32))
                final_mosaic_coverage_channels.append(np.zeros(output_shape_hw, dtype=np.float32))
        
        self.update_progress(f"  Temps total pour reproject_and_coadd (tous canaux): {total_reproject_time_sec:.2f}s.", progress_base_finalize + progress_range_reproject_step)

        if not final_mosaic_sci_channels or len(final_mosaic_sci_channels) != num_color_channels_expected:
             error_msg = "Échec critique: reproject_and_coadd n'a pas produit le nombre attendu de canaux."
             self.update_progress(f"❌ {error_msg}", "ERROR"); self.processing_error = error_msg; self.final_stacked_path = None; return

        try:
            final_sci_image_HWC = np.stack(final_mosaic_sci_channels, axis=-1).astype(np.float32)
            final_coverage_map_2D = final_mosaic_coverage_channels[0] 
            
            print(f"  -> Mosaïque combinée avec reproject. Shape SCI: {final_sci_image_HWC.shape}, Shape Coverage: {final_coverage_map_2D.shape}")
            print(f"     Range SCI (après reproject mean): [{np.nanmin(final_sci_image_HWC):.4g}, {np.nanmax(final_sci_image_HWC):.4g}]")
            print(f"     Range Coverage (après reproject): [{np.nanmin(final_coverage_map_2D):.4g}, {np.nanmax(final_coverage_map_2D):.4g}]")

            self.current_stack_header = fits.Header() 
            if output_wcs: self.current_stack_header.update(output_wcs.to_header(relax=True))
            
            if self.reference_header_for_wcs: 
                keys_to_copy_ref_hdr = ['INSTRUME', 'TELESCOP', 'OBSERVER', 'OBJECT', 
                                        'DATE-OBS', 'FILTER', 'BAYERPAT', 'FOCALLEN', 'APERTURE', 
                                        'XPIXSZ', 'YPIXSZ', 'SITELAT', 'SITELONG']
                for key_cp in keys_to_copy_ref_hdr:
                    if key_cp in self.reference_header_for_wcs:
                        try: self.current_stack_header[key_cp] = (self.reference_header_for_wcs[key_cp], self.reference_header_for_wcs.comments[key_cp])
                        except KeyError: self.current_stack_header[key_cp] = self.reference_header_for_wcs[key_cp]
            
            self.current_stack_header['STACKTYP'] = (f'Mosaic Reproject ({self.drizzle_scale:.0f}x)', 'Mosaic from reproject_and_coadd')
            self.current_stack_header['NIMAGES'] = (num_panels, 'Number of panels input to reproject') 
            
            total_approx_exposure = 0.0 
            if self.reference_header_for_wcs: 
                single_exp_ref = float(self.reference_header_for_wcs.get('EXPTIME', 10.0)) 
                total_approx_exposure = num_panels * single_exp_ref 
            self.current_stack_header['TOTEXP'] = (round(total_approx_exposure, 2), '[s] Approx total exposure (sum of panels ref exp)')
            
            self._save_final_stack(
                output_filename_suffix="_mosaic_reproject", 
                drizzle_final_sci_data=final_sci_image_HWC, 
                drizzle_final_wht_data=final_coverage_map_2D 
            )

        except Exception as e_stack_final:
            error_msg = f"Erreur finalisation/sauvegarde mosaïque avec reproject: {e_stack_final}"
            self.update_progress(f"❌ {error_msg}", "ERROR"); traceback.print_exc(limit=3); self.processing_error = error_msg; self.final_stacked_path = None
        finally:
            del input_data_for_reproject, input_footprints_for_reproject, all_wcs_for_grid_calc
            del final_mosaic_sci_channels, final_mosaic_coverage_channels
            gc.collect()
        
        print(f"DEBUG (Backend _finalize_mosaic_processing V_FinalizeMosaic_ReprojectCoadd_4_FixTqdmCall): Fin.")



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







# --- DANS LA CLASSE SeestarQueuedStacker DANS seestar/queuep/queue_manager.py ---

    def _process_file(self, file_path,
                      reference_image_data_for_alignment, # Image de l'ANCRE pour FastAligner ou réf. pour Astroalign std
                      solve_astrometry_for_this_file=False,
                      fa_orb_features_config=5000,
                      fa_min_abs_matches_config=10,
                      fa_min_ransac_inliers_value_config=4, 
                      fa_ransac_thresh_config=3.0,
                      daofind_fwhm_config=3.5,
                      daofind_threshold_sigma_config=6.0,
                      max_stars_to_describe_config=750):
        """
        Traite un seul fichier image.
        Version: V_ProcessFile_M81_Debug_UltimateLog_1
        """
        file_name = os.path.basename(file_path)
        quality_scores = {'snr': 0.0, 'stars': 0.0}
        print(f"\nDEBUG QM [_process_file V_ProcessFile_M81_Debug_UltimateLog_1]:") # Modifié le nom de version pour le log
        print(f"  >> Fichier: '{file_name}'")
        print(f"  >> Solve Astrometry Directly: {solve_astrometry_for_this_file}")
        print(f"  >> is_mosaic_run: {self.is_mosaic_run}, mosaic_alignment_mode: {getattr(self, 'mosaic_alignment_mode', 'N/A')}")
        print(f"  >> drizzle_active_session: {self.drizzle_active_session}")

        header_final_pour_retour = None
        img_data_array_loaded = None
        prepared_img_after_initial_proc = None
        image_for_alignment_or_drizzle_input = None
        wcs_final_pour_retour = None
        data_final_pour_retour = None
        valid_pixel_mask_2d = None
        matrice_M_calculee = None
        align_method_log_msg = "Unknown"

        try:
            print(f"  -> [1/7] Chargement/Validation FITS pour '{file_name}'...")
            loaded_data_tuple = load_and_validate_fits(file_path)
            if loaded_data_tuple and loaded_data_tuple[0] is not None:
                img_data_array_loaded, header_from_load = loaded_data_tuple
                header_final_pour_retour = header_from_load.copy() if header_from_load else fits.Header()
            else:
                header_temp_fallback = None
                if loaded_data_tuple and loaded_data_tuple[1] is not None: header_temp_fallback = loaded_data_tuple[1].copy()
                else:
                    try: header_temp_fallback = fits.getheader(file_path)
                    except: header_temp_fallback = fits.Header()
                header_final_pour_retour = header_temp_fallback
                raise ValueError("Échec chargement/validation FITS (données non retournées).")
            header_final_pour_retour['_SRCFILE'] = (file_name, "Original source filename")
            print(f"     - FITS original (après load_and_validate): Range: [{np.min(img_data_array_loaded):.4g}, {np.max(img_data_array_loaded):.4g}], Shape: {img_data_array_loaded.shape}, Dtype: {img_data_array_loaded.dtype}")

            print(f"  -> [2/7] Vérification variance pour '{file_name}'...")
            std_dev = np.std(img_data_array_loaded)
            variance_threshold = 0.0015
            if std_dev < variance_threshold:
                raise ValueError(f"Faible variance: {std_dev:.4f} (seuil: {variance_threshold}).")
            print(f"     - Variance OK (std: {std_dev:.4f}).")

            print(f"  -> [3/7] Pré-traitement pour '{file_name}'...")
            prepared_img_after_initial_proc = img_data_array_loaded.astype(np.float32)
            print(f"     - (a) Après conversion float32: Range: [{np.min(prepared_img_after_initial_proc):.4g}, {np.max(prepared_img_after_initial_proc):.4g}]")

            is_color_after_preprocessing = False
            if prepared_img_after_initial_proc.ndim == 2:
                bayer_pattern_from_header = header_final_pour_retour.get('BAYERPAT', self.bayer_pattern)
                pattern_upper = bayer_pattern_from_header.upper() if isinstance(bayer_pattern_from_header, str) else self.bayer_pattern.upper()
                if pattern_upper in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                    prepared_img_after_initial_proc = debayer_image(prepared_img_after_initial_proc, pattern_upper)
                    is_color_after_preprocessing = True
                    print(f"     - (b) Image débayerisée. Range: [{np.min(prepared_img_after_initial_proc):.4g}, {np.max(prepared_img_after_initial_proc):.4g}]")
            elif prepared_img_after_initial_proc.ndim == 3 and prepared_img_after_initial_proc.shape[2] == 3:
                is_color_after_preprocessing = True
            else:
                raise ValueError(f"Shape image {prepared_img_after_initial_proc.shape} non supportée post-chargement.")

            if is_color_after_preprocessing:
                try:
                    r_ch, g_ch, b_ch = prepared_img_after_initial_proc[...,0], prepared_img_after_initial_proc[...,1], prepared_img_after_initial_proc[...,2]
                    med_r, med_g, med_b = np.median(r_ch), np.median(g_ch), np.median(b_ch)
                    if med_g > 1e-6:
                        gain_r = np.clip(med_g / max(med_r, 1e-6), 0.5, 2.0); gain_b = np.clip(med_g / max(med_b, 1e-6), 0.5, 2.0)
                        prepared_img_after_initial_proc[...,0] *= gain_r; prepared_img_after_initial_proc[...,2] *= gain_b
                    print(f"     - (c) WB basique appliquée. Range: [{np.min(prepared_img_after_initial_proc):.4g}, {np.max(prepared_img_after_initial_proc):.4g}]")
                except Exception as e_wb: print(f"WARN QM [_process_file]: Erreur WB basique: {e_wb}")

            if self.correct_hot_pixels:
                prepared_img_after_initial_proc = detect_and_correct_hot_pixels(
                    prepared_img_after_initial_proc, self.hot_pixel_threshold, self.neighborhood_size)
                print(f"     - (d) Correction HP. Range: [{np.min(prepared_img_after_initial_proc):.4g}, {np.max(prepared_img_after_initial_proc):.4g}]")
            
            is_drizzle_or_mosaic_mode = (self.drizzle_active_session or self.is_mosaic_run)
            print(f"     - (e) is_drizzle_or_mosaic_mode: {is_drizzle_or_mosaic_mode}")
            
            image_for_alignment_or_drizzle_input = prepared_img_after_initial_proc.copy()
            print(f"     - (f) image_for_alignment_or_drizzle_input (copie de (d)) - Range: [{np.min(image_for_alignment_or_drizzle_input):.4g}, {np.max(image_for_alignment_or_drizzle_input):.4g}]")

            current_max_val = np.nanmax(image_for_alignment_or_drizzle_input)
            if is_drizzle_or_mosaic_mode:
                if current_max_val <= 1.0 + 1e-5 and current_max_val > -1e-5: 
                    print(f"       - (g) DRIZZLE/MOSAIQUE: Détection plage [0,1] (max_val={current_max_val:.4g}). Rescale vers ADU 0-65535.")
                    image_for_alignment_or_drizzle_input = image_for_alignment_or_drizzle_input * 65535.0
                    print(f"         Nouveau range image_for_alignment_or_drizzle_input: [{np.min(image_for_alignment_or_drizzle_input):.4g}, {np.max(image_for_alignment_or_drizzle_input):.4g}]")
                image_for_alignment_or_drizzle_input = np.clip(image_for_alignment_or_drizzle_input, 0.0, None) 
                print(f"     - (h) Pré-traitement final POUR DRIZZLE/MOSAIQUE: image_for_alignment_or_drizzle_input - Range: [{np.min(image_for_alignment_or_drizzle_input):.4g}, {np.max(image_for_alignment_or_drizzle_input):.4g}]")
                data_final_pour_retour = image_for_alignment_or_drizzle_input.astype(np.float32)
            else: 
                print(f"     - (g) STACKING CLASSIQUE: image_for_alignment_or_drizzle_input (pour alignement) - Range: [{np.min(image_for_alignment_or_drizzle_input):.4g}, {np.max(image_for_alignment_or_drizzle_input):.4g}]")
            
            print(f"  -> [4/7] Alignement/Résolution WCS pour '{file_name}'...")
            print(f"     - AVANT ALIGNEMENT: image_for_alignment_or_drizzle_input - Range: [{np.min(image_for_alignment_or_drizzle_input):.4g}, {np.max(image_for_alignment_or_drizzle_input):.4g}], Shape: {image_for_alignment_or_drizzle_input.shape}")

            if not solve_astrometry_for_this_file and self.is_mosaic_run and \
               self.mosaic_alignment_mode in ["local_fast_fallback", "local_fast_only"]:
                align_method_log_msg = "FastAligner_Attempted"; fa_success = False 
                if self.local_aligner_instance and reference_image_data_for_alignment is not None:
                    _, M_par_fa, fa_success = self.local_aligner_instance._align_image(image_for_alignment_or_drizzle_input, reference_image_data_for_alignment, file_name, fa_min_abs_matches_config, fa_min_ransac_inliers_value_config, fa_ransac_thresh_config, 0.15, daofind_fwhm_config, daofind_threshold_sigma_config, max_stars_to_describe_config)
                    if fa_success and M_par_fa is not None: align_method_log_msg = "FastAligner_Success"; matrice_M_calculee = M_par_fa; wcs_final_pour_retour = self.reference_wcs_object
                    else: fa_success = False; align_method_log_msg = "FastAligner_Fail"
                else: align_method_log_msg = "LocalAlign_Not_Attempted"
                if not fa_success and self.use_wcs_fallback_for_mosaic: 
                    align_method_log_msg += "_Fallback_Attempted" 
                    if self.astrometry_solver:
                        solver_settings_for_panel_fallback = { 'local_solver_preference': self.local_solver_preference, 'api_key': self.api_key, 'astap_path': self.astap_path, 'astap_data_dir': self.astap_data_dir,'astap_search_radius': self.astap_search_radius,'local_ansvr_path': self.local_ansvr_path,'scale_est_arcsec_per_pix': self.reference_pixel_scale_arcsec,'scale_tolerance_percent': 20, 'ansvr_timeout_sec': getattr(self, 'ansvr_timeout_sec', 120),'astap_timeout_sec': getattr(self, 'astap_timeout_sec', 120),'astrometry_net_timeout_sec': getattr(self, 'astrometry_net_timeout_sec', 300)}
                        wcs_panel_solved_by_solver = None
                        try: wcs_panel_solved_by_solver = self.astrometry_solver.solve(file_path, header_final_pour_retour, solver_settings_for_panel_fallback,True)
                        except Exception as e_s: align_method_log_msg += f"_SolveError_{type(e_s).__name__}"
                        if wcs_panel_solved_by_solver and wcs_panel_solved_by_solver.is_celestial:
                            align_method_log_msg = "FastAligner_Fail_Fallback_WCS_Success"; wcs_final_pour_retour = wcs_panel_solved_by_solver 
                            matrice_M_calculee = self._calculate_M_from_wcs(wcs_panel_solved_by_solver, self.reference_wcs_object, image_for_alignment_or_drizzle_input.shape[:2] )
                            if matrice_M_calculee is None: align_method_log_msg = "FastAligner_Fail_Fallback_WCS_Matrix_Fail"; wcs_final_pour_retour = None 
                        else: 
                            if "_SolveError_" not in align_method_log_msg: align_method_log_msg = "FastAligner_Fail_Fallback_WCS_Fail"
                            wcs_final_pour_retour = None; matrice_M_calculee = None
                    else: align_method_log_msg = "FastAligner_Fail_Fallback_NoSolver"; wcs_final_pour_retour = None; matrice_M_calculee = None
                elif not fa_success and not self.use_wcs_fallback_for_mosaic: align_method_log_msg = "FastAligner_Fail_No_Fallback"; wcs_final_pour_retour = None; matrice_M_calculee = None
                # data_final_pour_retour a déjà été mis à image_for_alignment_or_drizzle_input (ADU) si mode drizzle/mosaic
            
            elif solve_astrometry_for_this_file and self.is_mosaic_run and self.mosaic_alignment_mode == "astrometry_per_panel":
                align_method_log_msg = "Astrometry_Per_Panel_Attempted"
                if self.astrometry_solver:
                    solver_settings_for_this_panel = { 'local_solver_preference': self.local_solver_preference, 'api_key': self.api_key, 'astap_path': self.astap_path, 'astap_data_dir': self.astap_data_dir, 'astap_search_radius': self.astap_search_radius, 'local_ansvr_path': self.local_ansvr_path, 'scale_est_arcsec_per_pix': self.reference_pixel_scale_arcsec,'scale_tolerance_percent': 20, 'ansvr_timeout_sec': getattr(self, 'ansvr_timeout_sec', 120),'astap_timeout_sec': getattr(self, 'astap_timeout_sec', 120),'astrometry_net_timeout_sec': getattr(self, 'astrometry_net_timeout_sec', 300)}
                    wcs_final_pour_retour = self.astrometry_solver.solve(file_path, header_final_pour_retour, solver_settings_for_this_panel, True)
                    if wcs_final_pour_retour and wcs_final_pour_retour.is_celestial: align_method_log_msg = "Astrometry_Per_Panel_Success"; matrice_M_calculee = np.array([[1.,0.,0.],[0.,1.,0.]], dtype=np.float32) 
                    else: align_method_log_msg = "Astrometry_Per_Panel_Fail"; wcs_final_pour_retour = None; matrice_M_calculee = None
                else: align_method_log_msg = "Astrometry_Per_Panel_NoSolver"; wcs_final_pour_retour = None; matrice_M_calculee = None
                # data_final_pour_retour a déjà été mis à image_for_alignment_or_drizzle_input (ADU)
            else: 
                align_method_log_msg = "Astroalign_Standard_Attempted"
                if reference_image_data_for_alignment is None: raise RuntimeError("Image de référence Astroalign manquante.")
                
                aligned_img_astroalign, align_success_astroalign = self.aligner._align_image(
                    image_for_alignment_or_drizzle_input, reference_image_data_for_alignment, file_name)
                
                if align_success_astroalign and aligned_img_astroalign is not None:
                    align_method_log_msg = "Astroalign_Standard_Success"
                    print(f"     - APRÈS ALIGNEMENT (Astroalign): aligned_img_astroalign - Range: [{np.min(aligned_img_astroalign):.4g}, {np.max(aligned_img_astroalign):.4g}], Shape: {aligned_img_astroalign.shape}, Dtype: {aligned_img_astroalign.dtype}")
                    data_final_pour_retour = aligned_img_astroalign.astype(np.float32)
                    
                    if not is_drizzle_or_mosaic_mode: 
                        print(f"       - STACKING CLASSIQUE: Normalisation 0-1 de data_final_pour_retour (qui est aligned_img_astroalign)...")
                        min_val_aligned = np.nanmin(data_final_pour_retour); max_val_aligned = np.nanmax(data_final_pour_retour)
                        print(f"         Pour Normalisation Classique: min_val_aligned={min_val_aligned:.4g}, max_val_aligned={max_val_aligned:.4g}")
                        if np.isfinite(min_val_aligned) and np.isfinite(max_val_aligned) and max_val_aligned > min_val_aligned + 1e-7:
                            data_final_pour_retour = (data_final_pour_retour - min_val_aligned) / (max_val_aligned - min_val_aligned)
                            print(f"         Normalisation 0-1 effectuée (dynamique normale).")
                        else: 
                            print(f"         AVERTISSEMENT: Image alignée plate ou avec NaN/Inf. data_final_pour_retour sera mis à ZÉRO.")
                            data_final_pour_retour = np.zeros_like(data_final_pour_retour)
                        data_final_pour_retour = np.clip(data_final_pour_retour, 0.0, 1.0)
                        print(f"       - STACKING CLASSIQUE: data_final_pour_retour NORMALISÉ 0-1. Range: [{np.min(data_final_pour_retour):.4g}, {np.max(data_final_pour_retour):.4g}], Moy: {np.mean(data_final_pour_retour):.4g}")
                    else: 
                        # Pour Drizzle Standard, data_final_pour_retour est déjà aligned_img_astroalign.
                        # _align_image est censé avoir préservé la plage ADU si l'entrée était ADU.
                        print(f"       - DRIZZLE STANDARD: data_final_pour_retour (venant de aligned_img_astroalign) gardé en ADU. Range: [{np.min(data_final_pour_retour):.4g}, {np.max(data_final_pour_retour):.4g}]")
                else:
                    align_method_log_msg = "Astroalign_Standard_Fail"; raise RuntimeError(f"Échec Alignement Astroalign standard pour {file_name}.")
                matrice_M_calculee = None 
            
            header_final_pour_retour['_ALIGN_METHOD_LOG'] = (align_method_log_msg, "Alignment method used")

            print(f"  -> [5/7] Création du masque de pixels valides pour '{file_name}'...")
            if data_final_pour_retour is None: raise ValueError("Données finales pour masque sont None.")
            if data_final_pour_retour.ndim == 3: luminance_mask_src = 0.299 * data_final_pour_retour[..., 0] + 0.587 * data_final_pour_retour[..., 1] + 0.114 * data_final_pour_retour[..., 2]
            elif data_final_pour_retour.ndim == 2: luminance_mask_src = data_final_pour_retour
            else: valid_pixel_mask_2d = np.ones(data_final_pour_retour.shape[:2], dtype=bool); print(f"     - Masque (tous valides, shape inattendue).")
            
            if 'valid_pixel_mask_2d' not in locals() or valid_pixel_mask_2d is None : 
                print(f"     - Création masque depuis luminance_mask_src. Range luminance: [{np.min(luminance_mask_src):.4g}, {np.max(luminance_mask_src):.4g}]")
                max_lum_val = np.nanmax(luminance_mask_src)
                mask_threshold = 1.0 if (is_drizzle_or_mosaic_mode and max_lum_val > 1.5 + 1e-5) else 1e-5 # +1e-5 pour float
                valid_pixel_mask_2d = (luminance_mask_src > mask_threshold).astype(bool)
                print(f"     - Masque créé (seuil: {mask_threshold:.4g}). Shape: {valid_pixel_mask_2d.shape}, Dtype: {valid_pixel_mask_2d.dtype}, Sum (True): {np.sum(valid_pixel_mask_2d)}")

            print(f"  -> [6/7] Calcul des scores qualité pour '{file_name}'...")
            if self.use_quality_weighting: quality_scores = self._calculate_quality_metrics(prepared_img_after_initial_proc)
            else: print(f"     - Pondération qualité désactivée.")

            if data_final_pour_retour is None: raise RuntimeError("data_final_pour_retour est None à la fin de _process_file.")
            if valid_pixel_mask_2d is None: raise RuntimeError("valid_pixel_mask_2d est None à la fin de _process_file.")

            if self.is_mosaic_run and self.mosaic_alignment_mode in ["local_fast_fallback", "local_fast_only"]:
                if wcs_final_pour_retour is None or matrice_M_calculee is None: raise RuntimeError(f"Mosaïque locale '{file_name}', WCS ou M manquant. AlignMethod: {align_method_log_msg}")
            elif self.is_mosaic_run and self.mosaic_alignment_mode == "astrometry_per_panel":
                if wcs_final_pour_retour is None: raise RuntimeError(f"Mosaïque AstroPanel '{file_name}', WCS résolu manquant. AlignMethod: {align_method_log_msg}")

            # ---- ULTIMATE DEBUG LOG ----
            print(f"ULTIMATE DEBUG QM [_process_file V_ProcessFile_M81_Debug_UltimateLog_1]: AVANT RETURN pour '{file_name}'.")
            if data_final_pour_retour is not None:
                print(f"  >>> data_final_pour_retour - Shape: {data_final_pour_retour.shape}, Dtype: {data_final_pour_retour.dtype}, Range: [{np.min(data_final_pour_retour):.6g}, {np.max(data_final_pour_retour):.6g}], Mean: {np.mean(data_final_pour_retour):.6g}")
            else:
                print(f"  >>> data_final_pour_retour est None.")
            if valid_pixel_mask_2d is not None:
                print(f"  >>> valid_pixel_mask_2d - Shape: {valid_pixel_mask_2d.shape}, Dtype: {valid_pixel_mask_2d.dtype}, Sum (True): {np.sum(valid_pixel_mask_2d)}")
            else:
                print(f"  >>> valid_pixel_mask_2d est None.")
            print(f"  >>> quality_scores: {quality_scores}")
            if wcs_final_pour_retour is not None: print(f"  >>> wcs_final_pour_retour: Présent")
            else: print(f"  >>> wcs_final_pour_retour: None")
            if matrice_M_calculee is not None: print(f"  >>> matrice_M_calculee: Présente")
            else: print(f"  >>> matrice_M_calculee: None")
            # ---- FIN ULTIMATE DEBUG LOG ----

            return (data_final_pour_retour, header_final_pour_retour, quality_scores, 
                    wcs_final_pour_retour, matrice_M_calculee, valid_pixel_mask_2d)

        except (ValueError, RuntimeError) as proc_err:
            self.update_progress(f"   ⚠️ Fichier '{file_name}' ignoré dans _process_file: {proc_err}", "WARN")
            print(f"ERREUR QM [_process_file V_ProcessFile_M81_Debug_UltimateLog_1]: (ValueError/RuntimeError) pour '{file_name}': {proc_err}")
            header_final_pour_retour = header_final_pour_retour if header_final_pour_retour is not None else fits.Header()
            header_final_pour_retour['_ALIGN_METHOD_LOG'] = (f"Error_{type(proc_err).__name__}", "Processing file error")
            if hasattr(self, '_move_to_unaligned'): self._move_to_unaligned(file_path) 
            return None, header_final_pour_retour, quality_scores, None, None, None 
        except Exception as e:
            self.update_progress(f"❌ Erreur critique traitement fichier {file_name} dans _process_file: {e}", "ERROR")
            print(f"ERREUR QM [_process_file V_ProcessFile_M81_Debug_UltimateLog_1]: Exception générale pour '{file_name}': {e}"); traceback.print_exc(limit=3)
            header_final_pour_retour = header_final_pour_retour if header_final_pour_retour is not None else fits.Header()
            header_final_pour_retour['_ALIGN_METHOD_LOG'] = (f"CritError_{type(e).__name__}", "Critical processing error")
            if hasattr(self, '_move_to_unaligned'): self._move_to_unaligned(file_path) 
            return None, header_final_pour_retour, quality_scores, None, None, None 
        finally:
            if img_data_array_loaded is not None: del img_data_array_loaded
            if prepared_img_after_initial_proc is not None: del prepared_img_after_initial_proc
            if image_for_alignment_or_drizzle_input is not None: del image_for_alignment_or_drizzle_input
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



# --- DANS LA CLASSE SeestarQueuedStacker DANS seestar/queuep/queue_manager.py ---

    def _save_drizzle_input_temp(self, aligned_data, header):
        """
        Sauvegarde une image alignée (HxWx3 float32) dans le dossier temp Drizzle,
        en transposant en CxHxW et en INJECTANT l'OBJET WCS DE RÉFÉRENCE stocké
        dans le header sauvegardé.
        Les données `aligned_data` doivent être dans la plage ADU finale souhaitée.
        """
        if self.drizzle_temp_dir is None: 
            self.update_progress("❌ Erreur interne: Dossier temp Drizzle non défini."); return None
        os.makedirs(self.drizzle_temp_dir, exist_ok=True)
        if aligned_data.ndim != 3 or aligned_data.shape[2] != 3: 
            self.update_progress(f"❌ Erreur interne: _save_drizzle_input_temp attend HxWx3, reçu {aligned_data.shape}"); return None
        if self.reference_wcs_object is None:
             self.update_progress("❌ Erreur interne: Objet WCS de référence non disponible pour sauvegarde temp.")
             return None

        try:
            # Utiliser un nom de fichier qui inclut le nom original pour le débogage du header EXPTIME
            original_filename_stem = "unknown_orig"
            if header and '_SRCFILE' in header:
                original_filename_stem = os.path.splitext(header['_SRCFILE'][0])[0]
            
            temp_filename = f"aligned_input_{self.aligned_files_count:05d}_{original_filename_stem}.fits"
            temp_filepath = os.path.join(self.drizzle_temp_dir, temp_filename)

            data_to_save = np.moveaxis(aligned_data, -1, 0).astype(np.float32) # Doit être ADU ici

            # ---- DEBUG: Vérifier le range de ce qui est sauvegardé ----
            print(f"    DEBUG QM [_save_drizzle_input_temp]: Sauvegarde FITS temp '{temp_filename}'. data_to_save (CxHxW) Range Ch0: [{np.min(data_to_save[0]):.4g}, {np.max(data_to_save[0]):.4g}]")
            # ---- FIN DEBUG ----

            header_to_save = header.copy() if header else fits.Header()
            
            # Effacer WCS potentiellement incorrect du header original
            keys_to_remove = ['PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                              'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2',
                              'CDELT1', 'CDELT2', 'CROTA2', 'EQUINOX', 'RADESYS'] # RADESYS aussi car WCS ref l'aura
            for key in keys_to_remove:
                if key in header_to_save:
                    try: del header_to_save[key]
                    except KeyError: pass

            ref_wcs_header = self.reference_wcs_object.to_header(relax=True)
            header_to_save.update(ref_wcs_header)

            header_to_save['NAXIS'] = 3
            header_to_save['NAXIS1'] = aligned_data.shape[1] 
            header_to_save['NAXIS2'] = aligned_data.shape[0] 
            header_to_save['NAXIS3'] = 3                   
            if 'CTYPE3' not in header_to_save: header_to_save['CTYPE3'] = 'CHANNEL'
            
            # Assurer BITPIX = -32 pour float32
            header_to_save['BITPIX'] = -32
            if 'BSCALE' in header_to_save: del header_to_save['BSCALE']
            if 'BZERO' in header_to_save: del header_to_save['BZERO']


            hdu = fits.PrimaryHDU(data=data_to_save, header=header_to_save)
            hdul = fits.HDUList([hdu])
            hdul.writeto(temp_filepath, overwrite=True, checksum=False, output_verify='ignore')
            hdul.close()
            return temp_filepath

        except Exception as e:
            temp_filename_for_error = f"aligned_input_{self.aligned_files_count:05d}.fits" # Générique
            self.update_progress(f"❌ Erreur sauvegarde fichier temp Drizzle {temp_filename_for_error}: {e}")
            traceback.print_exc(limit=2)
            return None
        

###########################################################################################################################


# --- DANS LA CLASSE SeestarQueuedStacker DANS seestar/queuep/queue_manager.py ---

    def _process_incremental_drizzle_batch(self, batch_temp_filepaths_list, current_batch_num=0, total_batches_est=0):
        """
        [VRAI DRIZZLE INCRÉMENTAL] Traite un lot de fichiers temporaires en les ajoutant
        aux objets Drizzle persistants. Met à jour l'aperçu après chaque image (ou lot).
        Version: V_True_Incremental_Driz_DebugM81_Scale_2_Full
        """
        num_files_in_batch = len(batch_temp_filepaths_list)
        print(f"DEBUG QM [_process_incremental_drizzle_batch V_True_Incremental_Driz_DebugM81_Scale_2_Full]: Début Lot Drizzle Incr. VRAI #{current_batch_num} ({num_files_in_batch} fichiers).")

        if not batch_temp_filepaths_list:
            self.update_progress(f"⚠️ Lot Drizzle Incrémental VRAI #{current_batch_num} vide. Ignoré.")
            return

        progress_info = f"(Lot {current_batch_num}/{total_batches_est if total_batches_est > 0 else '?'})"
        self.update_progress(f"💧 Traitement Drizzle Incrémental VRAI du lot {progress_info}...")

        if not self.incremental_drizzle_objects or len(self.incremental_drizzle_objects) != 3:
            self.update_progress("❌ Erreur critique: Objets Drizzle persistants non initialisés pour mode Incrémental.", "ERROR")
            self.processing_error = "Objets Drizzle Incr. non initialisés"; self.stop_processing = True
            return
        if self.drizzle_output_wcs is None or self.drizzle_output_shape_hw is None:
            self.update_progress("❌ Erreur critique: Grille de sortie Drizzle (WCS/Shape) non définie pour mode Incrémental VRAI.", "ERROR")
            self.processing_error = "Grille Drizzle non définie (Incr VRAI)"; self.stop_processing = True
            return

        num_output_channels = 3
        files_added_to_drizzle_this_batch = 0

        for i_file, temp_fits_filepath in enumerate(batch_temp_filepaths_list):
            if self.stop_processing: break 
            
            current_filename_for_log = os.path.basename(temp_fits_filepath)
            self.update_progress(f"   -> DrizIncrVrai: Ajout fichier {i_file+1}/{num_files_in_batch} ('{current_filename_for_log}') au Drizzle cumulatif...", None)
            print(f"    DEBUG QM [ProcIncrDrizLoop M81_Scale_2_Full]: Fichier '{current_filename_for_log}'")

            input_image_cxhxw = None 
            input_header = None      
            wcs_input_from_file = None 
            pixmap_for_this_file = None

            try:
                with fits.open(temp_fits_filepath, memmap=False) as hdul:
                    if not hdul or len(hdul) == 0 or hdul[0].data is None: 
                        raise IOError(f"FITS temp invalide/vide: {temp_fits_filepath}")
                    
                    data_loaded = hdul[0].data 
                    input_header = hdul[0].header
                    print(f"      DEBUG QM [ProcIncrDrizLoop M81_Scale_2_Full]: Données chargées depuis FITS temp '{current_filename_for_log}': Range [{np.min(data_loaded):.4g}, {np.max(data_loaded):.4g}], Shape: {data_loaded.shape}, Dtype: {data_loaded.dtype}")

                    if data_loaded.ndim == 3 and data_loaded.shape[0] == num_output_channels:
                        input_image_cxhxw = data_loaded.astype(np.float32)
                        print(f"        input_image_cxhxw (après astype float32): Range [{np.min(input_image_cxhxw):.4g}, {np.max(input_image_cxhxw):.4g}]")
                    else:
                        raise ValueError(f"Shape FITS temp {data_loaded.shape} non CxHxW comme attendu.")
                    
                    with warnings.catch_warnings(): 
                        warnings.simplefilter("ignore")
                        wcs_input_from_file = WCS(input_header, naxis=2) 
                    if not wcs_input_from_file or not wcs_input_from_file.is_celestial:
                        raise ValueError("WCS non céleste ou invalide dans le fichier FITS temporaire.")

                input_shape_hw_current_file = (input_image_cxhxw.shape[1], input_image_cxhxw.shape[2])
                y_in_coords_flat, x_in_coords_flat = np.indices(input_shape_hw_current_file).reshape(2, -1)
                sky_ra_deg, sky_dec_deg = wcs_input_from_file.all_pix2world(x_in_coords_flat, y_in_coords_flat, 0)
                
                if not (np.all(np.isfinite(sky_ra_deg)) and np.all(np.isfinite(sky_dec_deg))):
                    raise ValueError("Coordonnées célestes non finies obtenues depuis le WCS du fichier temporaire.")

                final_x_output_pixels, final_y_output_pixels = self.drizzle_output_wcs.all_world2pix(sky_ra_deg, sky_dec_deg, 0)
                
                if not (np.all(np.isfinite(final_x_output_pixels)) and np.all(np.isfinite(final_y_output_pixels))):
                    print(f"      WARN [ProcIncrDrizLoop]: Pixmap pour '{current_filename_for_log}' contient NaN/Inf après projection. Nettoyage...")
                    final_x_output_pixels = np.nan_to_num(final_x_output_pixels, nan=-1e9, posinf=-1e9, neginf=-1e9)
                    final_y_output_pixels = np.nan_to_num(final_y_output_pixels, nan=-1e9, posinf=-1e9, neginf=-1e9)

                pixmap_for_this_file = np.dstack((
                    final_x_output_pixels.reshape(input_shape_hw_current_file), 
                    final_y_output_pixels.reshape(input_shape_hw_current_file)
                )).astype(np.float32)
                print(f"      DEBUG QM [ProcIncrDrizLoop M81_Scale_2_Full]: Pixmap calculé pour '{current_filename_for_log}'.")

                exptime_for_drizzle_add = 1.0 
                in_units_for_drizzle_add = 'cps' 
                if input_header and 'EXPTIME' in input_header:
                    try:
                        original_exptime = float(input_header['EXPTIME'])
                        if original_exptime > 1e-6:
                            exptime_for_drizzle_add = original_exptime
                            in_units_for_drizzle_add = 'counts' 
                            print(f"        Utilisation EXPTIME={exptime_for_drizzle_add:.2f}s du header original ('{input_header.get('_SRCFILE', 'N/A_SRC')}'), in_units='counts'")
                        else:
                             print(f"        EXPTIME du header original ({original_exptime:.2f}) trop faible. Utilisation exptime=1.0, in_units='cps'.")
                    except (ValueError, TypeError):
                        print(f"        AVERTISSEMENT: EXPTIME invalide dans header temp ('{input_header.get('EXPTIME')}' pour '{input_header.get('_SRCFILE', 'N/A_SRC')}'). Utilisation exptime=1.0, in_units='cps'.")
                else:
                    print(f"        AVERTISSEMENT: EXPTIME non trouvé dans header temp pour '{input_header.get('_SRCFILE', 'N/A_SRC')}'. Utilisation exptime=1.0, in_units='cps'.")
                
                weight_map_param_for_add = np.ones(input_shape_hw_current_file, dtype=np.float32)

                for ch_idx in range(num_output_channels):
                    channel_data_2d = input_image_cxhxw[ch_idx, :, :].astype(np.float32)
                    if not np.all(np.isfinite(channel_data_2d)): 
                        channel_data_2d[~np.isfinite(channel_data_2d)] = 0.0 
                    
                    print(f"        Ch{ch_idx} AVANT add_image: data range [{np.min(channel_data_2d):.3g}, {np.max(channel_data_2d):.3g}], exptime={exptime_for_drizzle_add}, in_units='{in_units_for_drizzle_add}', pixfrac={self.drizzle_pixfrac}")
                    if weight_map_param_for_add is not None:
                        print(f"                         weight_map range [{np.min(weight_map_param_for_add):.3g}, {np.max(weight_map_param_for_add):.3g}]")
                    
                    self.incremental_drizzle_objects[ch_idx].add_image(
                        data=channel_data_2d, 
                        pixmap=pixmap_for_this_file,
                        exptime=exptime_for_drizzle_add, 
                        in_units=in_units_for_drizzle_add, 
                        pixfrac=self.drizzle_pixfrac, 
                        weight_map=weight_map_param_for_add 
                    )
                    print(f"        Ch{ch_idx} APRÈS add_image: out_img range [{np.min(self.incremental_drizzle_sci_arrays[ch_idx]):.3g}, {np.max(self.incremental_drizzle_sci_arrays[ch_idx]):.3g}]")
                    print(f"                             out_wht range [{np.min(self.incremental_drizzle_wht_arrays[ch_idx]):.3g}, {np.max(self.incremental_drizzle_wht_arrays[ch_idx]):.3g}]")

                files_added_to_drizzle_this_batch += 1
                self.images_in_cumulative_stack += 1 

            except Exception as e_file:
                self.update_progress(f"      -> ERREUR Drizzle Incr. VRAI sur fichier '{current_filename_for_log}': {e_file}", "WARN")
                print(f"ERREUR QM [ProcIncrDrizLoop M81_Scale_2_Full]: Échec fichier '{current_filename_for_log}': {e_file}"); traceback.print_exc(limit=1)
            finally:
                del input_image_cxhxw, input_header, wcs_input_from_file, pixmap_for_this_file
                if (i_file + 1) % 10 == 0: gc.collect()
        
        if files_added_to_drizzle_this_batch == 0 and num_files_in_batch > 0:
            self.update_progress(f"   -> ERREUR: Aucun fichier du lot Drizzle Incr. VRAI #{current_batch_num} n'a pu être ajouté.", "ERROR")
            self.failed_stack_count += num_files_in_batch 
        else:
            self.update_progress(f"   -> {files_added_to_drizzle_this_batch}/{num_files_in_batch} fichiers du lot Drizzle Incr. VRAI #{current_batch_num} ajoutés aux objets Drizzle.")

        if self.current_stack_header is None: 
            self.current_stack_header = fits.Header()
            if self.drizzle_output_wcs:
                 try: self.current_stack_header.update(self.drizzle_output_wcs.to_header(relax=True))
                 except Exception as e_hdr_wcs: print(f"WARN: Erreur copie WCS au header (DrizIncrVrai): {e_hdr_wcs}")
            self.current_stack_header['STACKTYP'] = (f'Drizzle_Incremental_True_{self.drizzle_scale:.0f}x', 'True Incremental Drizzle')
            self.current_stack_header['DRZSCALE'] = (self.drizzle_scale, 'Drizzle scale factor')
            self.current_stack_header['DRZKERNEL'] = (self.drizzle_kernel, 'Drizzle kernel used')
            self.current_stack_header['DRZPIXFR'] = (self.drizzle_pixfrac, 'Drizzle pixfrac used')
            self.current_stack_header['CREATOR'] = ('SeestarStacker_QM', 'Processing Software')
        
        self.current_stack_header['NIMAGES'] = (self.images_in_cumulative_stack, 'Total images drizzled incrementally')

        self.update_progress(f"   -> Préparation aperçu Drizzle Incrémental VRAI (Lot #{current_batch_num})...")
        try:
            if self.preview_callback and self.incremental_drizzle_sci_arrays and self.incremental_drizzle_wht_arrays:
                avg_img_channels_preview = []
                for c in range(num_output_channels):
                    # Pour l'aperçu, on veut afficher l'image moyenne estimée: out_img / out_wht (où out_wht est non nul)
                    # Cependant, out_img est déjà un "flux" (data*wt/exptime). out_wht est sum(wt^2/exptime).
                    # Une meilleure approximation de l'image moyenne est sci_arrays[c] / sqrt(wht_arrays[c]) si wt=1.
                    # Ou plus simplement, juste afficher sci_arrays[c] (out_img), qui est déjà une sorte de moyenne pondérée.
                    
                    # Pour l'instant, on affiche directement sci_arrays[c] (qui est out_img de Drizzle)
                    # car la division par out_wht peut être instable si out_wht est petit ou a des artefacts.
                    preview_channel_data = self.incremental_drizzle_sci_arrays[c].astype(np.float32)
                    avg_img_channels_preview.append(np.nan_to_num(preview_channel_data, nan=0.0, posinf=0.0, neginf=0.0))
                
                preview_data_HWC_raw = np.stack(avg_img_channels_preview, axis=-1)
                min_p, max_p = np.nanmin(preview_data_HWC_raw), np.nanmax(preview_data_HWC_raw)
                preview_data_HWC_norm = preview_data_HWC_raw
                if np.isfinite(min_p) and np.isfinite(max_p) and max_p > min_p + 1e-7: 
                    preview_data_HWC_norm = (preview_data_HWC_raw - min_p) / (max_p - min_p)
                elif np.any(np.isfinite(preview_data_HWC_raw)):
                    preview_data_HWC_norm = np.full_like(preview_data_HWC_raw, 0.5)
                else:
                    preview_data_HWC_norm = np.zeros_like(preview_data_HWC_raw)
                
                preview_data_HWC_final = np.clip(preview_data_HWC_norm, 0.0, 1.0).astype(np.float32)
                self.current_stack_data = preview_data_HWC_final 
                self._update_preview() 
                print(f"    DEBUG QM [ProcIncrDrizLoop M81_Scale_2_Full]: Aperçu Driz Incr VRAI mis à jour. Range (0-1): [{np.min(preview_data_HWC_final):.3f}, {np.max(preview_data_HWC_final):.3f}]")
            else:
                print(f"    WARN QM [ProcIncrDrizLoop M81_Scale_2_Full]: Impossible de mettre à jour l'aperçu Driz Incr VRAI.")
        except Exception as e_prev:
            print(f"    ERREUR QM [ProcIncrDrizLoop M81_Scale_2_Full]: Erreur mise à jour aperçu Driz Incr VRAI: {e_prev}"); traceback.print_exc(limit=1)

        if self.perform_cleanup:
             print(f"DEBUG QM [_process_incremental_drizzle_batch V_True_Incremental_Driz_DebugM81_Scale_2_Full]: Nettoyage fichiers temp lot #{current_batch_num}...")
             self._cleanup_batch_temp_files(batch_temp_filepaths_list)
        
        print(f"DEBUG QM [_process_incremental_drizzle_batch V_True_Incremental_Driz_DebugM81_Scale_2_Full]: Fin traitement lot Driz Incr VRAI #{current_batch_num}.")






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
            save_preview_image(self.current_stack_data, preview_path, apply_stretch=False)
        except Exception as e: print(f"⚠️ Erreur sauvegarde stack intermédiaire: {e}")

    def _stack_winsorized_sigma(self, images, weights, kappa=3.0, winsor_limits=(0.05, 0.05)):
        from scipy.stats.mstats import winsorize
        from astropy.stats import sigma_clipped_stats
        arr = np.stack([im for im in images], axis=0).astype(np.float32)
        arr_w = winsorize(arr, limits=winsor_limits, axis=0)
        try:
            _, med, std = sigma_clipped_stats(arr_w, sigma=3.0, axis=0, maxiters=5)
        except TypeError:
            _, med, std = sigma_clipped_stats(arr_w, sigma_lower=3.0, sigma_upper=3.0, axis=0, maxiters=5)
        low = med - kappa * std
        high = med + kappa * std
        mask = (arr >= low) & (arr <= high)
        arr_clip = np.where(mask, arr, np.nan)
        if weights is not None:
            w = np.asarray(weights)[:, None, None]
            if arr.ndim == 4:
                w = w[..., None]
            sum_w = np.nansum(w * mask, axis=0)
            sum_d = np.nansum(arr_clip * w, axis=0)
            result = np.divide(sum_d, sum_w, out=np.zeros_like(sum_d), where=sum_w > 1e-6)
        else:
            result = np.nanmean(arr_clip, axis=0)
        return result.astype(np.float32)

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
            if stack_method_used_for_header == 'winsorized-sigma':
                stacked_batch_data_np = self._stack_winsorized_sigma(valid_images_for_ccdproc, weight_scalars_for_ccdproc, kappa_val_for_header)
            elif is_color_batch:
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


# --- DANS LA CLASSE SeestarQueuedStacker DANS seestar/queuep/queue_manager.py ---

    def _combine_intermediate_drizzle_batches(self, intermediate_files_list, output_wcs_final_target, output_shape_final_target_hw):
        """
        Combine les résultats Drizzle intermédiaires (par lot) sauvegardés sur disque.
        MODIFIED V4_CombineFixAPI: Correction initialisation Drizzle et utilisation pixfrac.
        MODIFIED DebugDrizzleFinal_1: Commenté le clipping Lanczos agressif, ajout logs.
        """
        final_sci_image_HWC = None
        final_wht_map_HWC = None # Sera HWC aussi, car les poids sont par canal pour Drizzle

        num_batches_to_combine = len(intermediate_files_list)
        if num_batches_to_combine == 0:
            self.update_progress("ⓘ Aucun lot Drizzle intermédiaire à combiner.")
            return final_sci_image_HWC, final_wht_map_HWC

        # --- DEBUG DRIZZLE FINAL 1: Log d'entrée ---
        print("\n" + "="*70)
        print(f"DEBUG QM [_combine_intermediate_drizzle_batches V4_CombineFixAPI_DebugDrizzleFinal_1]:")
        print(f"  Début pour {num_batches_to_combine} lots.")
        print(f"  Shape Sortie CIBLE: {output_shape_final_target_hw}, Drizzle Kernel: {self.drizzle_kernel}, Pixfrac: {self.drizzle_pixfrac}")
        # --- FIN DEBUG ---
        self.update_progress(f"💧 [CombineBatches V4] Début combinaison {num_batches_to_combine} lots Drizzle...")

        if output_wcs_final_target is None or output_shape_final_target_hw is None:
            self.update_progress("   [CombineBatches V4] ERREUR: WCS ou Shape de sortie final manquant.", "ERROR")
            return None, None

        num_output_channels = 3
        final_drizzlers = []
        final_output_images_list = [] # Liste des arrays SCI (H,W) par canal
        final_output_weights_list = []# Liste des arrays WHT (H,W) par canal

        try:
            self.update_progress(f"   [CombineBatches V4] Initialisation Drizzle final (Shape: {output_shape_final_target_hw})...")
            for _ in range(num_output_channels):
                final_output_images_list.append(np.zeros(output_shape_final_target_hw, dtype=np.float32))
                final_output_weights_list.append(np.zeros(output_shape_final_target_hw, dtype=np.float32))

            for i in range(num_output_channels):
                driz_ch = Drizzle(
                    kernel=self.drizzle_kernel,
                    fillval=str(getattr(self, "drizzle_fillval", "0.0")), # Utiliser l'attribut si existe
                    out_img=final_output_images_list[i],
                    out_wht=final_output_weights_list[i],
                    out_shape=output_shape_final_target_hw
                )
                final_drizzlers.append(driz_ch)
            self.update_progress(f"   [CombineBatches V4] Objets Drizzle finaux initialisés.")
        except Exception as init_err:
            self.update_progress(f"   [CombineBatches V4] ERREUR: Échec init Drizzle final: {init_err}", "ERROR")
            print(f"ERREUR QM [_combine_intermediate_drizzle_batches]: Échec init Drizzle: {init_err}"); traceback.print_exc(limit=1)
            return None, None

        total_contributing_ninputs_for_final_header = 0
        batches_successfully_added_to_final_drizzle = 0

        for i_batch_loop, (sci_fpath, wht_fpaths_list_for_batch) in enumerate(intermediate_files_list):
            if self.stop_processing:
                self.update_progress("🛑 Arrêt demandé pendant combinaison lots Drizzle.")
                break

            self.update_progress(f"   [CombineBatches V4] Ajout lot intermédiaire {i_batch_loop+1}/{num_batches_to_combine}: {os.path.basename(sci_fpath)}...")
            # --- DEBUG DRIZZLE FINAL 1: Log chemin lot ---
            print(f"  Processing batch {i_batch_loop+1}: SCI='{sci_fpath}', WHT0='{wht_fpaths_list_for_batch[0] if wht_fpaths_list_for_batch else 'N/A'}'")
            # --- FIN DEBUG ---

            if len(wht_fpaths_list_for_batch) != num_output_channels:
                self.update_progress(f"      -> ERREUR: Nb incorrect de cartes poids ({len(wht_fpaths_list_for_batch)}) pour lot {i_batch_loop+1}. Ignoré.", "WARN")
                continue

            sci_data_cxhxw_lot = None; wcs_lot_intermediaire = None
            wht_maps_2d_list_for_lot = None; header_sci_lot = None
            pixmap_batch_to_final_grid = None

            try:
                with fits.open(sci_fpath, memmap=False) as hdul_sci:
                    sci_data_cxhxw_lot = hdul_sci[0].data.astype(np.float32); header_sci_lot = hdul_sci[0].header
                    with warnings.catch_warnings(): warnings.simplefilter("ignore"); wcs_lot_intermediaire = WCS(header_sci_lot, naxis=2)
                if not wcs_lot_intermediaire.is_celestial: raise ValueError("WCS lot intermédiaire non céleste.")
                wht_maps_2d_list_for_lot = []
                for ch_idx_w, wht_fpath_ch in enumerate(wht_fpaths_list_for_batch):
                    with fits.open(wht_fpath_ch, memmap=False) as hdul_wht: wht_map_2d_ch = hdul_wht[0].data.astype(np.float32)
                    wht_maps_2d_list_for_lot.append(np.nan_to_num(np.maximum(wht_map_2d_ch, 0.0)))
                # --- DEBUG DRIZZLE FINAL 1: Log données lot chargées ---
                print(f"    Lot {i_batch_loop+1} SCI chargé - Shape CxHxW: {sci_data_cxhxw_lot.shape}, Range Ch0: [{np.min(sci_data_cxhxw_lot[0]):.3g}, {np.max(sci_data_cxhxw_lot[0]):.3g}]")
                print(f"    Lot {i_batch_loop+1} WHT0 chargé - Shape HW: {wht_maps_2d_list_for_lot[0].shape}, Range: [{np.min(wht_maps_2d_list_for_lot[0]):.3g}, {np.max(wht_maps_2d_list_for_lot[0]):.3g}]")
                # --- FIN DEBUG ---

                shape_lot_intermediaire_hw = sci_data_cxhxw_lot.shape[1:]
                y_lot_intermed, x_lot_intermed = np.indices(shape_lot_intermediaire_hw)
                sky_coords_lot_ra, sky_coords_lot_dec = wcs_lot_intermediaire.all_pix2world(x_lot_intermed.ravel(), y_lot_intermed.ravel(), 0)
                x_final_output_pix, y_final_output_pix = output_wcs_final_target.all_world2pix(sky_coords_lot_ra, sky_coords_lot_dec, 0)
                pixmap_batch_to_final_grid = np.dstack((x_final_output_pix.reshape(shape_lot_intermediaire_hw), y_final_output_pix.reshape(shape_lot_intermediaire_hw))).astype(np.float32)

                if pixmap_batch_to_final_grid is not None:
                    ninputs_this_batch = int(header_sci_lot.get('NINPUTS', 0))
                    for ch_idx_add in range(num_output_channels):
                        data_ch_sci_2d_lot = np.nan_to_num(sci_data_cxhxw_lot[ch_idx_add, :, :])
                        data_ch_wht_2d_lot = wht_maps_2d_list_for_lot[ch_idx_add]
                        # --- DEBUG DRIZZLE FINAL 1: Log avant add_image ---
                        print(f"      Ch{ch_idx_add} add_image: data SCI min/max [{np.min(data_ch_sci_2d_lot):.3g}, {np.max(data_ch_sci_2d_lot):.3g}], data WHT min/max [{np.min(data_ch_wht_2d_lot):.3g}, {np.max(data_ch_wht_2d_lot):.3g}], pixfrac={self.drizzle_pixfrac}")
                        # --- FIN DEBUG ---
                        final_drizzlers[ch_idx_add].add_image(
                            data=data_ch_sci_2d_lot,
                            pixmap=pixmap_batch_to_final_grid,
                            weight_map=data_ch_wht_2d_lot,
                            exptime=1.0, # Les lots sont déjà en counts/sec
                            pixfrac=self.drizzle_pixfrac,
                            in_units='cps' # Confirmé par BUNIT='Counts/s' dans les fichiers de lot
                        )
                    batches_successfully_added_to_final_drizzle += 1
                    total_contributing_ninputs_for_final_header += ninputs_this_batch

            except Exception as e_lot_proc:
                self.update_progress(f"   [CombineBatches V4] ERREUR traitement lot {i_batch_loop+1}: {e_lot_proc}", "ERROR"); continue
            finally:
                del sci_data_cxhxw_lot, wcs_lot_intermediaire, wht_maps_2d_list_for_lot, header_sci_lot, pixmap_batch_to_final_grid; gc.collect()

        if batches_successfully_added_to_final_drizzle == 0:
             self.update_progress("   [CombineBatches V4] ERREUR: Aucun lot Drizzle intermédiaire n'a pu être ajouté à la combinaison finale.", "ERROR")
             return None, None

        # --- DEBUG DRIZZLE FINAL 1: Log des données brutes accumulées PAR CANAL ---
        for ch_log_idx in range(num_output_channels):
            temp_ch_data = final_output_images_list[ch_log_idx]
            temp_ch_wht = final_output_weights_list[ch_log_idx]
            print(f"  DEBUG [CombineBatches V4]: DONNÉES ACCUMULÉES BRUTES (avant division/clipping) - Canal {ch_log_idx}:")
            if temp_ch_data is not None and temp_ch_data.size > 0:
                print(f"    SCI_ACCUM (out_img): Min={np.min(temp_ch_data):.4g}, Max={np.max(temp_ch_data):.4g}, Mean={np.mean(temp_ch_data):.4g}, Std={np.std(temp_ch_data):.4g}")
                print(f"      Négatifs SCI_ACCUM: {np.sum(temp_ch_data < 0)}")
            else: print("    SCI_ACCUM: Données vides ou invalides.")
            if temp_ch_wht is not None and temp_ch_wht.size > 0:
                print(f"    WHT_ACCUM (out_wht): Min={np.min(temp_ch_wht):.4g}, Max={np.max(temp_ch_wht):.4g}, Mean={np.mean(temp_ch_wht):.4g}")
            else: print("    WHT_ACCUM: Données vides ou invalides.")
        # --- FIN DEBUG ---

        try:
            # Les `final_output_images_list` contiennent la somme(data*wht) et `final_output_weights_list` contient la somme(wht)
            # La division se fera dans _save_final_stack. Ici, on stack juste pour retourner.
            final_sci_image_HWC = np.stack(final_output_images_list, axis=-1).astype(np.float32)
            final_wht_map_HWC = np.stack(final_output_weights_list, axis=-1).astype(np.float32) # Maintenant HWC

            # --- SECTION CLIPPING CONDITIONNEL POUR LANCZOS COMMENTÉE ---
            # if self.drizzle_kernel.lower() in ["lanczos2", "lanczos3"]:
            #     print(f"DEBUG [CombineBatches V4]: CLIPPING LANCZOS TEMPORAIREMENT DÉSACTIVÉ.")
            #     # print(f"DEBUG [CombineBatches V4]: Application du clipping spécifique pour kernel {self.drizzle_kernel}.")
            #     # self.update_progress(f"   Appli. clipping spécifique pour Lanczos...", "DEBUG_DETAIL")
            #     # clip_min_lanczos = 0.0
            #     # clip_max_lanczos = 2.0 # Exemple, à ajuster.
            #     # print(f"  [CombineBatches V4]: Clipping Lanczos: Min={clip_min_lanczos}, Max={clip_max_lanczos}")
            #     # print(f"    Avant clip (Ch0): Min={np.min(final_sci_image_HWC[...,0]):.4g}, Max={np.max(final_sci_image_HWC[...,0]):.4g}")
            #     # final_sci_image_HWC = np.clip(final_sci_image_HWC, clip_min_lanczos, clip_max_lanczos)
            #     # print(f"    Après clip (Ch0): Min={np.min(final_sci_image_HWC[...,0]):.4g}, Max={np.max(final_sci_image_HWC[...,0]):.4g}")
            # --- FIN SECTION CLIPPING COMMENTÉE ---

            # Nettoyage NaN/Inf et s'assurer que les poids sont non-négatifs
            final_sci_image_HWC = np.nan_to_num(final_sci_image_HWC, nan=0.0, posinf=0.0, neginf=0.0)
            final_wht_map_HWC = np.nan_to_num(final_wht_map_HWC, nan=0.0, posinf=0.0, neginf=0.0)
            final_wht_map_HWC = np.maximum(final_wht_map_HWC, 0.0) # Poids doivent être >= 0

            self.update_progress(f"   -> Assemblage final Drizzle terminé (Shape Sci HWC: {final_sci_image_HWC.shape}, Wht HWC: {final_wht_map_HWC.shape})")
            self.images_in_cumulative_stack = total_contributing_ninputs_for_final_header
        except Exception as e_final_asm:
            self.update_progress(f"   - ERREUR pendant assemblage final Drizzle: {e_final_asm}", "ERROR")
            final_sci_image_HWC = None
            final_wht_map_HWC = None
        finally:
            del final_drizzlers, final_output_images_list, final_output_weights_list
            gc.collect()
        
        print("="*70 + "\n")
        return final_sci_image_HWC, final_wht_map_HWC


############################################################################################################################################





    def _save_final_stack(self, output_filename_suffix: str = "", stopped_early: bool = False,
                          drizzle_final_sci_data=None, drizzle_final_wht_data=None):
        """
        Calcule l'image finale, applique les post-traitements et sauvegarde.
        MODIFIED:
        - self.last_saved_data_for_preview (pour GUI) est maintenant l'image normalisée [0,1] SANS stretch cosmétique du backend.
        - save_preview_image (pour PNG) est appelé avec apply_stretch=True sur ces données [0,1].
        - La sauvegarde FITS reste basée sur self.raw_adu_data_for_ui_histogram (si float32) ou les données cosmétiques [0,1] (si uint16).
        Version: V_SaveFinal_CorrectedDataFlow_1
        """
        print("\n" + "=" * 80)
        self.update_progress(f"DEBUG QM [_save_final_stack V_SaveFinal_CorrectedDataFlow_1]: Début. Suffixe: '{output_filename_suffix}', Arrêt précoce: {stopped_early}")
        print(f"DEBUG QM [_save_final_stack V_SaveFinal_CorrectedDataFlow_1]: Début. Suffixe: '{output_filename_suffix}', Arrêt précoce: {stopped_early}")
        
        save_as_float32_setting = getattr(self, 'save_final_as_float32', False) 
        self.update_progress(f"  DEBUG QM: Option de sauvegarde FITS effective (self.save_final_as_float32): {save_as_float32_setting}")
        print(f"  DEBUG QM: Option de sauvegarde FITS effective (self.save_final_as_float32): {save_as_float32_setting}")
        
        is_reproject_mosaic_mode = (output_filename_suffix == "_mosaic_reproject" and 
                                    drizzle_final_sci_data is not None and 
                                    drizzle_final_wht_data is not None)
        is_drizzle_final_mode_with_data = (
            self.drizzle_active_session and self.drizzle_mode == "Final" and 
            not self.is_mosaic_run and drizzle_final_sci_data is not None and
            drizzle_final_wht_data is not None and not is_reproject_mosaic_mode 
        )
        is_true_incremental_drizzle_from_objects = (
            self.drizzle_active_session and self.drizzle_mode == "Incremental" and 
            not self.is_mosaic_run and drizzle_final_sci_data is None 
        )
        is_classic_stacking_mode = not (is_reproject_mosaic_mode or is_drizzle_final_mode_with_data or is_true_incremental_drizzle_from_objects)

        current_operation_mode_log_desc = "Unknown" 
        current_operation_mode_log_fits = "Unknown" 

        if is_reproject_mosaic_mode: 
            current_operation_mode_log_desc = "Mosaïque (reproject_and_coadd)"
            current_operation_mode_log_fits = "Mosaic (reproject_and_coadd)"
        elif is_true_incremental_drizzle_from_objects: 
            current_operation_mode_log_desc = "Drizzle Incrémental VRAI (objets Drizzle)"
            current_operation_mode_log_fits = "True Incremental Drizzle (Drizzle objects)"
        elif is_drizzle_final_mode_with_data: 
            current_operation_mode_log_desc = f"Drizzle Standard Final (données lot fournies)"
            current_operation_mode_log_fits = "Drizzle Standard Final (from batch data)"
        elif is_classic_stacking_mode : 
            current_operation_mode_log_desc = "Stacking Classique SUM/W (memmaps)"
            current_operation_mode_log_fits = "Classic Stacking SUM/W (memmaps)"
        else: 
            if not self.drizzle_active_session and not self.is_mosaic_run:
                 current_operation_mode_log_desc = "Stacking Classique SUM/W (memmaps) - Fallback"
                 current_operation_mode_log_fits = "Classic Stacking SUM/W (memmaps) - Fallback"
                 is_classic_stacking_mode = True 

        self.update_progress(f"  DEBUG QM: Mode d'opération détecté pour sauvegarde: {current_operation_mode_log_desc}")
        print(f"  DEBUG QM: Mode d'opération détecté pour sauvegarde: {current_operation_mode_log_desc}")
        print("=" * 80 + "\n")
        self.update_progress(f"💾 Préparation sauvegarde finale (Mode: {current_operation_mode_log_desc})...")

        final_image_initial_raw = None    # Données "brutes" après combinaison (ADU ou [0,1] si classique déjà normalisé)
        final_wht_map_for_postproc = None # Carte de poids 2D pour certains post-traitements
        background_model_photutils = None # Modèle de fond si Photutils BN est appliqué

        self.raw_adu_data_for_ui_histogram = None # Sera les données ADU-like pour l'histogramme de l'UI
        # self.last_saved_data_for_preview est celui qui sera envoyé à l'UI pour son affichage
        # Il doit être normalisé [0,1] MAIS NON STRETCHÉ COSMÉTIQUEMENT par le backend.
        
        try:
            # --- ÉTAPE 1: Obtenir final_image_initial_raw et final_wht_map_for_postproc ---
            # (La logique pour obtenir ces données reste la même que votre version précédente)
            # ... (Bloc if/elif/else pour les modes reproject, drizzle, classique) ...
            # (Je reprends la logique de votre dernier log `taraceback.txt` pour cette partie)
            if is_reproject_mosaic_mode:
                self.update_progress("  DEBUG QM [SaveFinalStack] Mode: Mosaïque Reproject")
                print("  DEBUG QM [SaveFinalStack] Mode: Mosaïque Reproject")
                final_image_initial_raw = drizzle_final_sci_data.astype(np.float32) 
                if drizzle_final_wht_data.ndim == 3:
                    final_wht_map_for_postproc = np.mean(drizzle_final_wht_data, axis=2).astype(np.float32)
                else:
                    final_wht_map_for_postproc = drizzle_final_wht_data.astype(np.float32)
                final_wht_map_for_postproc = np.maximum(final_wht_map_for_postproc, 0.0) 
                self._close_memmaps()
                self.update_progress(f"    DEBUG QM: Mosaic Reproject - final_image_initial_raw - Range: [{np.nanmin(final_image_initial_raw):.4g} - {np.nanmax(final_image_initial_raw):.4g}]")
                print(f"    DEBUG QM: Mosaic Reproject - final_image_initial_raw - Range: [{np.nanmin(final_image_initial_raw):.4g} - {np.nanmax(final_image_initial_raw):.4g}]")

            elif is_true_incremental_drizzle_from_objects:
                self.update_progress("  DEBUG QM [SaveFinalStack] Mode: Drizzle Incrémental VRAI")
                print("  DEBUG QM [SaveFinalStack] Mode: Drizzle Incrémental VRAI")
                if not self.incremental_drizzle_sci_arrays or not self.incremental_drizzle_wht_arrays or \
                   len(self.incremental_drizzle_sci_arrays) != 3 or len(self.incremental_drizzle_wht_arrays) != 3:
                    raise ValueError("Donnees Drizzle incremental (sci/wht arrays) invalides ou manquantes.")
                sci_arrays_hw_list = self.incremental_drizzle_sci_arrays 
                wht_arrays_hw_list = self.incremental_drizzle_wht_arrays                 
                avg_img_channels_list = []
                processed_wht_channels_list_for_mean = [] 
                for c in range(3): 
                    sci_ch_accum_float64 = sci_arrays_hw_list[c].astype(np.float64)
                    wht_ch_accum_raw_float64 = wht_arrays_hw_list[c].astype(np.float64)
                    wht_ch_clipped_positive = np.maximum(wht_ch_accum_raw_float64, 0.0)
                    processed_wht_channels_list_for_mean.append(wht_ch_clipped_positive.astype(np.float32))
                    wht_ch_for_division = np.maximum(wht_ch_clipped_positive, 1e-9)                     
                    channel_mean_image_adu = np.zeros_like(sci_ch_accum_float64, dtype=np.float32)
                    valid_pixels_mask = wht_ch_for_division > 1e-8                     
                    with np.errstate(divide='ignore', invalid='ignore'):
                        channel_mean_image_adu[valid_pixels_mask] = sci_ch_accum_float64[valid_pixels_mask] / wht_ch_for_division[valid_pixels_mask]
                    avg_img_channels_list.append(np.nan_to_num(channel_mean_image_adu, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
                final_image_initial_raw = np.stack(avg_img_channels_list, axis=-1) 
                final_wht_map_for_postproc = np.mean(np.stack(processed_wht_channels_list_for_mean, axis=-1), axis=2).astype(np.float32)
                final_wht_map_for_postproc = np.maximum(final_wht_map_for_postproc, 0.0)
                self.update_progress(f"    DEBUG QM: Drizzle Incr VRAI - final_image_initial_raw - Range: [{np.nanmin(final_image_initial_raw):.4g} - {np.nanmax(final_image_initial_raw):.4g}]")
                print(f"    DEBUG QM: Drizzle Incr VRAI - final_image_initial_raw - Range: [{np.nanmin(final_image_initial_raw):.4g} - {np.nanmax(final_image_initial_raw):.4g}]")

            elif is_drizzle_final_mode_with_data: 
                self.update_progress("  DEBUG QM [SaveFinalStack] Mode: Drizzle Standard Final (depuis données de lot)")
                print("  DEBUG QM [SaveFinalStack] Mode: Drizzle Standard Final (depuis données de lot)")
                if drizzle_final_sci_data is None or drizzle_final_wht_data is None: raise ValueError("Donnees de lot Drizzle final (sci/wht) manquantes.")
                sci_data_float64 = drizzle_final_sci_data.astype(np.float64); wht_data_float64 = drizzle_final_wht_data.astype(np.float64) 
                wht_data_clipped_positive = np.maximum(wht_data_float64, 0.0)
                final_wht_map_for_postproc = np.mean(wht_data_clipped_positive, axis=2).astype(np.float32) 
                wht_for_div = np.maximum(wht_data_clipped_positive, 1e-9) 
                with np.errstate(divide='ignore', invalid='ignore'): final_image_initial_raw = sci_data_float64 / wht_for_div
                final_image_initial_raw = np.nan_to_num(final_image_initial_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                self._close_memmaps()
                self.update_progress(f"    DEBUG QM: Drizzle Std Final - final_image_initial_raw - Range: [{np.nanmin(final_image_initial_raw):.4g} - {np.nanmax(final_image_initial_raw):.4g}]")
                print(f"    DEBUG QM: Drizzle Std Final - final_image_initial_raw - Range: [{np.nanmin(final_image_initial_raw):.4g} - {np.nanmax(final_image_initial_raw):.4g}]")
            
            else: # SUM/W Classique
                self.update_progress("  DEBUG QM [SaveFinalStack] Mode: Stacking Classique SUM/W")
                print("  DEBUG QM [SaveFinalStack] Mode: Stacking Classique SUM/W")
                if self.cumulative_sum_memmap is None or self.cumulative_wht_memmap is None: raise ValueError("Accumulateurs memmap SUM/WHT non disponibles pour stacking classique.")
                
                final_sum = np.array(self.cumulative_sum_memmap, dtype=np.float64)
                self.update_progress(f"    DEBUG QM: Classic Mode - final_sum (HWC, from memmap) - Shape: {final_sum.shape}, Range: [{np.nanmin(final_sum):.4g} - {np.nanmax(final_sum):.4g}]")
                print(f"    DEBUG QM: Classic Mode - final_sum (HWC, from memmap) - Shape: {final_sum.shape}, Range: [{np.nanmin(final_sum):.4g} - {np.nanmax(final_sum):.4g}]")
                
                final_wht_map_2d_from_memmap = np.array(self.cumulative_wht_memmap, dtype=np.float32) 
                self.update_progress(f"    DEBUG QM: Classic Mode - final_wht_map_2d_from_memmap (HW) - Shape: {final_wht_map_2d_from_memmap.shape}, Range: [{np.nanmin(final_wht_map_2d_from_memmap):.4g} - {np.nanmax(final_wht_map_2d_from_memmap):.4g}]")
                print(f"    DEBUG QM: Classic Mode - final_wht_map_2d_from_memmap (HW) - Shape: {final_wht_map_2d_from_memmap.shape}, Range: [{np.nanmin(final_wht_map_2d_from_memmap):.4g} - {np.nanmax(final_wht_map_2d_from_memmap):.4g}]")
                
                self._close_memmaps() 
                
                final_wht_map_for_postproc = np.maximum(final_wht_map_2d_from_memmap, 0.0)
                wht_for_div_classic = np.maximum(final_wht_map_2d_from_memmap.astype(np.float64), 1e-9)
                wht_for_div_classic_broadcastable = wht_for_div_classic[..., np.newaxis]
                
                with np.errstate(divide='ignore', invalid='ignore'): 
                    final_image_initial_raw = final_sum / wht_for_div_classic_broadcastable
                final_image_initial_raw = np.nan_to_num(final_image_initial_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                self.update_progress(f"    DEBUG QM: Classic Mode - final_image_initial_raw (HWC, après SUM/WHT et nan_to_num) - Range: [{np.nanmin(final_image_initial_raw):.4g} - {np.nanmax(final_image_initial_raw):.4g}]")
                print(f"    DEBUG QM: Classic Mode - final_image_initial_raw (HWC, après SUM/WHT et nan_to_num) - Range: [{np.nanmin(final_image_initial_raw):.4g} - {np.nanmax(final_image_initial_raw):.4g}]")

        except Exception as e_get_raw:
            self.processing_error = f"Erreur obtention donnees brutes finales: {e_get_raw}"
            self.update_progress(f"❌ {self.processing_error}", "ERROR"); traceback.print_exc(limit=2)
            self.final_stacked_path = None; return

        if final_image_initial_raw is None:
            self.final_stacked_path = None; self.update_progress("ⓘ Aucun stack final (donnees brutes sont None)."); return
        
        # À ce stade, final_image_initial_raw contient les données "ADU-like"
        self.update_progress(f"  DEBUG QM [SaveFinalStack] final_image_initial_raw (AVANT post-traitements) - Range: [{np.nanmin(final_image_initial_raw):.4g}, {np.nanmax(final_image_initial_raw):.4g}], Shape: {final_image_initial_raw.shape}, Dtype: {final_image_initial_raw.dtype}")
        print(f"  DEBUG QM [SaveFinalStack] final_image_initial_raw (AVANT post-traitements) - Range: [{np.nanmin(final_image_initial_raw):.4g}, {np.nanmax(final_image_initial_raw):.4g}], Shape: {final_image_initial_raw.shape}, Dtype: {final_image_initial_raw.dtype}")

        final_image_initial_raw = np.clip(final_image_initial_raw, 0.0, None) 
        self.update_progress(f"    DEBUG QM: Après clip >=0 des valeurs négatives, final_image_initial_raw - Range: [{np.nanmin(final_image_initial_raw):.4g}, {np.nanmax(final_image_initial_raw):.4g}]")
        print(f"    DEBUG QM: Après clip >=0 des valeurs négatives, final_image_initial_raw - Range: [{np.nanmin(final_image_initial_raw):.4g}, {np.nanmax(final_image_initial_raw):.4g}]")

        # Appliquer le seuil WHT (si activé) aux données "ADU-like"
        if self.drizzle_wht_threshold > 0 and final_wht_map_for_postproc is not None:
            self.update_progress(f"  DEBUG QM [SaveFinalStack] Application du seuil WHT ({self.drizzle_wht_threshold}) sur final_wht_map_for_postproc à final_image_initial_raw.")
            print(f"  DEBUG QM [SaveFinalStack] Application du seuil WHT ({self.drizzle_wht_threshold}) sur final_wht_map_for_postproc à final_image_initial_raw.")
            # Normaliser la carte de poids pour le seuil relatif
            max_wht_val = np.max(final_wht_map_for_postproc)
            if max_wht_val > 1e-9:
                wht_threshold_abs = self.drizzle_wht_threshold * max_wht_val
                invalid_wht_pixels = final_wht_map_for_postproc < wht_threshold_abs
                if final_image_initial_raw.ndim == 3: 
                    final_image_initial_raw[invalid_wht_pixels, :] = 0.0 # Mettre à 0 au lieu de NaN
                elif final_image_initial_raw.ndim == 2: 
                    final_image_initial_raw[invalid_wht_pixels] = 0.0
                self.update_progress(f"    DEBUG QM: Après application du seuil WHT (mis à 0), final_image_initial_raw - Range: [{np.nanmin(final_image_initial_raw):.4g}, {np.nanmax(final_image_initial_raw):.4g}]")
                print(f"    DEBUG QM: Après application du seuil WHT (mis à 0), final_image_initial_raw - Range: [{np.nanmin(final_image_initial_raw):.4g}, {np.nanmax(final_image_initial_raw):.4g}]")
        
        # Stocker les données "ADU-like" (après WHT threshold) pour l'histogramme de l'UI
        self.raw_adu_data_for_ui_histogram = final_image_initial_raw.astype(np.float32).copy()
        self.update_progress(f"  DEBUG QM [SaveFinalStack] self.raw_adu_data_for_ui_histogram (pour UI, après WHT threshold) STOCKE. Range: [{np.min(self.raw_adu_data_for_ui_histogram):.4f}, {np.max(self.raw_adu_data_for_ui_histogram):.4f}], Dtype: {self.raw_adu_data_for_ui_histogram.dtype}")
        print(f"  DEBUG QM [SaveFinalStack] self.raw_adu_data_for_ui_histogram (pour UI, après WHT threshold) STOCKE. Range: [{np.min(self.raw_adu_data_for_ui_histogram):.4f}, {np.max(self.raw_adu_data_for_ui_histogram):.4f}], Dtype: {self.raw_adu_data_for_ui_histogram.dtype}")
        
        # --- ÉTAPE 2: Préparer self.last_saved_data_for_preview (normalisé [0,1] SANS stretch cosmétique backend) ---
        self.update_progress(f"  DEBUG QM [SaveFinalStack] Normalisation min-max de self.raw_adu_data_for_ui_histogram pour self.last_saved_data_for_preview...")
        print(f"  DEBUG QM [SaveFinalStack] Normalisation min-max de self.raw_adu_data_for_ui_histogram pour self.last_saved_data_for_preview...")
        
        min_val_for_01_norm = np.nanmin(self.raw_adu_data_for_ui_histogram)
        max_val_for_01_norm = np.nanmax(self.raw_adu_data_for_ui_histogram)
        range_for_01_norm = max_val_for_01_norm - min_val_for_01_norm

        data_01_for_gui_preview = self.raw_adu_data_for_ui_histogram.copy() # Commencer avec une copie
        if np.isfinite(min_val_for_01_norm) and np.isfinite(max_val_for_01_norm) and range_for_01_norm > 1e-9:
            data_01_for_gui_preview = (data_01_for_gui_preview - min_val_for_01_norm) / range_for_01_norm
        elif np.any(np.isfinite(data_01_for_gui_preview)): # Image constante
            data_01_for_gui_preview = np.full_like(data_01_for_gui_preview, 0.5)
        else: # Tout NaN/Inf
            data_01_for_gui_preview = np.zeros_like(data_01_for_gui_preview)
        
        data_01_for_gui_preview = np.clip(data_01_for_gui_preview, 0.0, 1.0).astype(np.float32)
        self.last_saved_data_for_preview = data_01_for_gui_preview
        
        self.update_progress(f"    DEBUG QM: self.last_saved_data_for_preview (normalisé 0-1, NON STRETCHÉ) - Range: [{np.nanmin(self.last_saved_data_for_preview):.4f}, {np.nanmax(self.last_saved_data_for_preview):.4f}], Dtype: {self.last_saved_data_for_preview.dtype}")
        print(f"    DEBUG QM: self.last_saved_data_for_preview (normalisé 0-1, NON STRETCHÉ) - Range: [{np.nanmin(self.last_saved_data_for_preview):.4f}, {np.nanmax(self.last_saved_data_for_preview):.4f}], Dtype: {self.last_saved_data_for_preview.dtype}")

        # --- ÉTAPE 3: Appliquer les post-traitements à la version [0,1] pour la sauvegarde PNG ---
        #    (et potentiellement pour la sauvegarde FITS si save_as_float32_setting est False)
        #    Note: data_after_postproc sera toujours dans la plage [0,1] et en float32.
        data_after_postproc = self.last_saved_data_for_preview.copy() # Partir de l'image [0,1] non stretchée
        self.update_progress(f"  DEBUG QM [SaveFinalStack] data_after_postproc (AVANT post-traitements) - Range: [{np.nanmin(data_after_postproc):.4f}, {np.nanmax(data_after_postproc):.4f}]")
        print(f"  DEBUG QM [SaveFinalStack] data_after_postproc (AVANT post-traitements) - Range: [{np.nanmin(data_after_postproc):.4f}, {np.nanmax(data_after_postproc):.4f}]")
        
        # --- Début du Pipeline de Post-Traitement (identique à votre version précédente) ---
        # ... (BN Globale, Photutils BN, CB, Feathering, Low WHT Mask, SCNR, Crop) ...
        # (Le code pour appliquer ces post-traitements à data_after_postproc reste ici)
        # --- Fin du Pipeline de Post-Traitement ---
        self.update_progress(f"  DEBUG QM [SaveFinalStack] data_after_postproc (APRES post-traitements, si activés) - Range: [{np.nanmin(data_after_postproc):.4f}, {np.nanmax(data_after_postproc):.4f}], Dtype: {data_after_postproc.dtype}")
        print(f"  DEBUG QM [SaveFinalStack] data_after_postproc (APRES post-traitements, si activés) - Range: [{np.nanmin(data_after_postproc):.4f}, {np.nanmax(data_after_postproc):.4f}], Dtype: {data_after_postproc.dtype}")
        
        # --- ÉTAPE 4: Préparation du header FITS final et du nom de fichier ---
        # (Logique identique)
        effective_image_count = self.images_in_cumulative_stack if self.images_in_cumulative_stack > 0 else (getattr(self, 'aligned_files_count', 1) if (is_drizzle_final_mode_with_data or is_reproject_mosaic_mode) else 1)
        final_header = self.current_stack_header.copy() if self.current_stack_header else fits.Header()
        if is_true_incremental_drizzle_from_objects or is_drizzle_final_mode_with_data or is_reproject_mosaic_mode: 
            if self.drizzle_output_wcs and not is_reproject_mosaic_mode : final_header.update(self.drizzle_output_wcs.to_header(relax=True))
            elif is_reproject_mosaic_mode and self.current_stack_header and self.current_stack_header.get('CTYPE1'): pass 
        final_header['NIMAGES'] = (effective_image_count, 'Effective images/Total Weight for final stack'); final_header['TOTEXP']  = (round(self.total_exposure_seconds, 2), '[s] Approx total exposure')
        final_header['HISTORY'] = f"Final stack type: {current_operation_mode_log_fits}"
        if getattr(self, 'output_filename', ""):
            base_name = self.output_filename.strip()
            if not base_name.lower().endswith('.fit'):
                base_name += '.fit'
            fits_path = os.path.join(self.output_folder, base_name)
            preview_path = os.path.splitext(fits_path)[0] + '.png'
        else:
            base_name = "stack_final"
            run_type_suffix = output_filename_suffix if output_filename_suffix else "_unknown_mode"
            if stopped_early: run_type_suffix += "_stopped"
            elif self.processing_error: run_type_suffix += "_error"
            fits_path = os.path.join(self.output_folder, f"{base_name}{run_type_suffix}.fit")
            preview_path  = os.path.splitext(fits_path)[0] + ".png"
        self.final_stacked_path = fits_path; self.update_progress(f"Chemin FITS final: {os.path.basename(fits_path)}")

        # --- ÉTAPE 5: Préparation des données pour la SAUVEGARDE FITS ---
        data_for_primary_hdu_save = None
        if save_as_float32_setting:
            self.update_progress("   DEBUG QM: Preparation sauvegarde FITS en float32 (brut ADU-like)...") 
            print("   DEBUG QM: Preparation sauvegarde FITS en float32 (brut ADU-like)...")
            data_for_primary_hdu_save = self.raw_adu_data_for_ui_histogram # Utilise les données "ADU-like" (non-normalisées 0-1 cosmétiquement)
            self.update_progress(f"     DEBUG QM: -> FITS float32: Utilisation self.raw_adu_data_for_ui_histogram. Shape: {data_for_primary_hdu_save.shape}, Range: [{np.min(data_for_primary_hdu_save):.4f}, {np.max(data_for_primary_hdu_save):.4f}]")
            print(f"     DEBUG QM: -> FITS float32: Utilisation self.raw_adu_data_for_ui_histogram. Shape: {data_for_primary_hdu_save.shape}, Range: [{np.min(data_for_primary_hdu_save):.4f}, {np.max(data_for_primary_hdu_save):.4f}]")
            final_header['BITPIX'] = -32 
            if 'BSCALE' in final_header: del final_header['BSCALE']; 
            if 'BZERO' in final_header: del final_header['BZERO']
        else: # Sauvegarde en uint16
            self.update_progress("   DEBUG QM: Preparation sauvegarde FITS en uint16 (depuis données cosmétiques [0,1] -> 0-65535)...") 
            print("   DEBUG QM: Preparation sauvegarde FITS en uint16 (depuis données cosmétiques [0,1] -> 0-65535)...")
            # data_after_postproc est l'image [0,1] après tous les post-traitements cosmétiques
            data_scaled_uint16 = (np.clip(data_after_postproc, 0.0, 1.0) * 65535.0).astype(np.uint16) 
            data_for_primary_hdu_save = data_scaled_uint16 
            self.update_progress(f"     DEBUG QM: -> FITS uint16: Utilisation données post-traitées [0,1] et scalées à 0-65535. Shape: {data_for_primary_hdu_save.shape}, Range: [{np.min(data_for_primary_hdu_save)}, {np.max(data_for_primary_hdu_save)}]")
            print(f"     DEBUG QM: -> FITS uint16: Utilisation données post-traitées [0,1] et scalées à 0-65535. Shape: {data_for_primary_hdu_save.shape}, Range: [{np.min(data_for_primary_hdu_save)}, {np.max(data_for_primary_hdu_save)}]")
            final_header['BITPIX'] = 16 
        
        if data_for_primary_hdu_save.ndim == 3 and data_for_primary_hdu_save.shape[2] == 3 : 
            data_for_primary_hdu_save_cxhxw = np.moveaxis(data_for_primary_hdu_save, -1, 0) 
        else: 
            data_for_primary_hdu_save_cxhxw = data_for_primary_hdu_save
        self.update_progress(f"     DEBUG QM: Données FITS prêtes (Shape HDU: {data_for_primary_hdu_save_cxhxw.shape}, Dtype: {data_for_primary_hdu_save_cxhxw.dtype})")
        print(f"     DEBUG QM: Données FITS prêtes (Shape HDU: {data_for_primary_hdu_save_cxhxw.shape}, Dtype: {data_for_primary_hdu_save_cxhxw.dtype})")

        # --- ÉTAPE 6: Sauvegarde FITS effective ---
        try: 
            primary_hdu = fits.PrimaryHDU(data=data_for_primary_hdu_save_cxhxw, header=final_header)
            hdus_list = [primary_hdu]
            # ... (logique HDU background_model si besoin) ...
            fits.HDUList(hdus_list).writeto(fits_path, overwrite=True, checksum=True, output_verify='ignore')
            self.update_progress(f"   ✅ Sauvegarde FITS ({'float32' if save_as_float32_setting else 'uint16'}) terminee.");  
        except Exception as save_err: 
            self.update_progress(f"   ❌ Erreur Sauvegarde FITS: {save_err}"); self.final_stacked_path = None

        # --- ÉTAPE 7: Sauvegarde preview PNG ---
        # Utiliser data_after_postproc (qui est l'image [0,1] après tous les post-traitements)
        # et laisser save_preview_image appliquer son propre stretch par défaut.
        if data_after_postproc is not None: 
            self.update_progress(f"  DEBUG QM (_save_final_stack): Données pour save_preview_image (data_after_postproc) - Range: [{np.nanmin(data_after_postproc):.4f}, {np.nanmax(data_after_postproc):.4f}], Shape: {data_after_postproc.shape}, Dtype: {data_after_postproc.dtype}")
            print(f"  DEBUG QM (_save_final_stack): Données pour save_preview_image (data_after_postproc) - Range: [{np.nanmin(data_after_postproc):.4f}, {np.nanmax(data_after_postproc):.4f}], Shape: {data_after_postproc.shape}, Dtype: {data_after_postproc.dtype}")
            try:
                save_preview_image(data_after_postproc, preview_path, 
                                   apply_stretch=True, # <<< MODIFIÉ ICI
                                   enhanced_stretch=False) # ou True si vous préférez le stretch "enhanced" pour le PNG
                self.update_progress("     ✅ Sauvegarde Preview PNG terminee.") 
            except Exception as prev_err: self.update_progress(f"     ❌ Erreur Sauvegarde Preview PNG: {prev_err}.") 
        else: self.update_progress("ⓘ Aucune image a sauvegarder pour preview PNG (data_after_postproc est None)."); 
            
        self.update_progress(f"DEBUG QM [_save_final_stack V_SaveFinal_CorrectedDataFlow_1]: Fin methode (mode: {current_operation_mode_log_desc}).")
        print("\n" + "=" * 80); print(f"DEBUG QM [_save_final_stack V_SaveFinal_CorrectedDataFlow_1]: Fin methode (mode: {current_operation_mode_log_desc})."); print("=" * 80 + "\n")








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
        """
        NOTE: Cette méthode ne supprime PLUS le contenu du dossier unaligned_files.
        Les fichiers non alignés sont intentionnellement conservés pour l'utilisateur.
        Le dossier lui-même est créé s'il n'existe pas, mais son contenu n'est pas purgé ici.
        """
        if self.unaligned_folder: # Vérifier si le chemin est défini
            if not os.path.isdir(self.unaligned_folder):
                try:
                    os.makedirs(self.unaligned_folder, exist_ok=True)
                    # Optionnel: loguer la création si elle a lieu ici
                    # self.update_progress(f"ⓘ Dossier pour fichiers non alignés créé: {self.unaligned_folder}")
                except OSError as e:
                    self.update_progress(f"⚠️ Erreur création dossier pour non-alignés '{self.unaligned_folder}': {e}")
            # else:
                # Optionnel: loguer que le dossier existe déjà
                # self.update_progress(f"ⓘ Dossier pour fichiers non alignés existe déjà: {self.unaligned_folder}")
            
            # Log explicite que les fichiers ne sont PAS supprimés par cette fonction
            print(f"DEBUG QM [cleanup_unaligned_files]: Contenu de '{self.unaligned_folder}' CONSERVÉ (pas de suppression automatique).")
            # self.update_progress(f"ⓘ Fichiers dans '{os.path.basename(self.unaligned_folder)}' conservés pour analyse.") # Optionnel pour l'UI
        else:
            print(f"DEBUG QM [cleanup_unaligned_files]: self.unaligned_folder non défini, aucune action de nettoyage/création.")





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
                         output_filename="",
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
                         apply_low_wht_mask=False, 
                         low_wht_percentile=5,    
                         low_wht_soften_px=128,   
                         is_mosaic_run=False, api_key=None, 
                         mosaic_settings=None,
                         use_local_solver_priority=False, # DEPRECATED - ignoré, mais gardé pour compatibilité signature
                         astap_path="",
                         astap_data_dir="",
                         local_ansvr_path="",
                         astap_search_radius=3.0,
                         local_solver_preference="none",
                         save_as_float32=False # <-- NOUVEL ARGUMENT AJOUTÉ ICI
                         ):
        print(f"!!!!!!!!!! VALEUR BRUTE ARGUMENT astap_search_radius REÇU : {astap_search_radius} !!!!!!!!!!")
        print(f"!!!!!!!!!! VALEUR BRUTE ARGUMENT save_as_float32 REÇU : {save_as_float32} !!!!!!!!!!") # DEBUG
                         
        """
        Démarre le thread de traitement principal avec la configuration spécifiée.
        MODIFIED: Ajout argument save_as_float32 et assignation à self.
        Version: V_StartProcessing_SaveDtypeOption_1
        """
        print("DEBUG QM (start_processing V_StartProcessing_SaveDtypeOption_1): Début tentative démarrage...") # Version Log
        
        print("  --- BACKEND ARGS REÇUS (depuis GUI/SettingsManager) ---")
        print(f"    input_dir='{input_dir}', output_dir='{output_dir}'")
        print(f"    is_mosaic_run (arg de func): {is_mosaic_run}")
        print(f"    use_drizzle (global arg de func): {use_drizzle}")
        print(f"    drizzle_mode (global arg de func): {drizzle_mode}")
        print(f"    mosaic_settings (dict brut): {mosaic_settings}")
        print(f"    save_as_float32 (arg de func): {save_as_float32}") # Log du nouvel argument
        print(f"    output_filename (arg de func): {output_filename}")
        print("  --- FIN BACKEND ARGS REÇUS ---")

        if self.processing_active:
            self.update_progress("⚠️ Tentative de démarrer un traitement déjà en cours.")
            return False

        self.stop_processing = False 
        if hasattr(self, 'aligner') and self.aligner is not None:
            self.aligner.stop_processing = False
        else: 
            self.update_progress("❌ Erreur interne critique: Aligner principal non initialisé. Démarrage annulé.")
            print("ERREUR QM (start_processing): self.aligner non initialisé.")
            return False
        
        self.current_folder = os.path.abspath(input_dir) if input_dir else None
        self.output_folder = os.path.abspath(output_dir) if output_dir else None
        

        # =========================================================================================
        # === ÉTAPE 1 : CONFIGURATION DES PARAMÈTRES DE SESSION SUR L'INSTANCE (AVANT TOUT LE RESTE) ===
        # =========================================================================================
        print("DEBUG QM (start_processing): Étape 1 - Configuration des paramètres de session sur l'instance...")
          
        if not self.current_folder or not os.path.isdir(self.current_folder):
            self.update_progress(f"❌ Dossier d'entrée principal '{input_dir}' invalide ou non défini.", "ERROR")
            return False
        if not self.output_folder: 
            self.update_progress(f"❌ Dossier de sortie '{output_dir}' non défini.", "ERROR")
            return False
        try:
            os.makedirs(self.output_folder, exist_ok=True) 
        except OSError as e_mkdir:
            self.update_progress(f"❌ Erreur création dossier de sortie '{self.output_folder}': {e_mkdir}", "ERROR")
            return False
        print(f"    [Paths] Input: '{self.current_folder}', Output: '{self.output_folder}'")

        self.output_filename = str(output_filename).strip()
        
        self.local_solver_preference = str(local_solver_preference) 
        self.astap_path = str(astap_path)
        self.astap_data_dir = str(astap_data_dir)
        self.astap_search_radius = float(astap_search_radius) 
        self.local_ansvr_path = str(local_ansvr_path)
        
        print(f"    [Solver Settings sur self via start_processing args] Pref: '{self.local_solver_preference}'")
        print(f"    [Solver Settings sur self via start_processing args] ASTAP Path: '{self.astap_path}'")
        print(f"    [Solver Settings sur self via start_processing args] ASTAP Data Dir: '{self.astap_data_dir}'")
        print(f"    [Solver Settings sur self via start_processing args] ASTAP Search Radius: {self.astap_search_radius}")
        print(f"    [Solver Settings sur self via start_processing args] Ansvr Path: '{self.local_ansvr_path}'")
        
        try:
            self.astap_search_radius_config = float(astap_search_radius)
        except (ValueError, TypeError):
            print(f"  WARN QM (start_processing): Valeur astap_search_radius ('{astap_search_radius}') invalide. Utilisation de 5.0° par défaut.")
            self.astap_search_radius_config = 5.0 
        
        # self.use_local_solver_priority (attribut de self) n'est plus utilisé, la variable locale de la fonction l'est.
        # print(f"    [Solver Settings sur self] Priorité Locale: {self.use_local_solver_priority}") 
        print(f"    [Solver Settings sur self] ASTAP Exe: '{self.astap_path}'")
        print(f"    [Solver Settings sur self] ASTAP Data: '{self.astap_data_dir}'")
        print(f"    [Solver Settings sur self] Ansvr Local: '{self.local_ansvr_path}'")
        print(f"    [Solver Settings sur self] ASTAP Search Radius Config: {self.astap_search_radius_config}°")

        self.is_mosaic_run = is_mosaic_run                     
        self.drizzle_active_session = use_drizzle or self.is_mosaic_run   
        self.drizzle_mode = str(drizzle_mode) 
        
        self.api_key = api_key if isinstance(api_key, str) else ""
        if getattr(self, 'reference_pixel_scale_arcsec', None) is None:
            self.reference_pixel_scale_arcsec = None 
        
        self.stacking_mode = str(stacking_mode); self.kappa = float(kappa)
        self.correct_hot_pixels = bool(correct_hot_pixels); self.hot_pixel_threshold = float(hot_pixel_threshold)
        self.neighborhood_size = int(neighborhood_size); self.bayer_pattern = str(bayer_pattern) if bayer_pattern else "GRBG"
        self.perform_cleanup = bool(perform_cleanup)
        self.use_quality_weighting = bool(use_weighting); self.weight_by_snr = bool(weight_by_snr)
        self.weight_by_stars = bool(weight_by_stars); self.snr_exponent = float(snr_exp)
        self.stars_exponent = float(stars_exp); self.min_weight = float(max(0.01, min(1.0, min_w)))
        
        self.drizzle_scale = float(drizzle_scale) if drizzle_scale else 2.0 
        self.drizzle_wht_threshold = float(drizzle_wht_threshold) 

        self.apply_chroma_correction = bool(apply_chroma_correction); self.apply_final_scnr = bool(apply_final_scnr)
        self.final_scnr_target_channel = str(final_scnr_target_channel).lower(); self.final_scnr_amount = float(final_scnr_amount)
        self.final_scnr_preserve_luminosity = bool(final_scnr_preserve_luminosity)
        self.bn_grid_size_str = str(bn_grid_size_str); self.bn_perc_low = int(bn_perc_low); self.bn_perc_high = int(bn_perc_high)
        self.bn_std_factor = float(bn_std_factor); self.bn_min_gain = float(bn_min_gain); self.bn_max_gain = float(bn_max_gain)
        self.cb_border_size = int(cb_border_size); self.cb_blur_radius = int(cb_blur_radius)
        self.cb_min_b_factor = float(cb_min_b_factor); self.cb_max_b_factor = float(cb_max_b_factor)
        self.final_edge_crop_percent_decimal = float(final_edge_crop_percent) / 100.0
        self.apply_photutils_bn = bool(apply_photutils_bn); self.photutils_bn_box_size = int(photutils_bn_box_size)
        self.photutils_bn_filter_size = int(photutils_bn_filter_size); self.photutils_bn_sigma_clip = float(photutils_bn_sigma_clip)
        self.photutils_bn_exclude_percentile = float(photutils_bn_exclude_percentile)
        self.apply_feathering = bool(apply_feathering); self.feather_blur_px = int(feather_blur_px)
        self.apply_low_wht_mask = bool(apply_low_wht_mask); self.low_wht_percentile = int(low_wht_percentile); self.low_wht_soften_px = int(low_wht_soften_px)

        # --- NOUVEAU : Assignation du paramètre de sauvegarde à l'attribut de l'instance ---
        self.save_final_as_float32 = bool(save_as_float32)
        print(f"    [OutputFormat] self.save_final_as_float32 (attribut d'instance) mis à : {self.save_final_as_float32} (depuis argument {save_as_float32})")
        # --- FIN NOUVEAU ---

        self.mosaic_settings_dict = mosaic_settings if isinstance(mosaic_settings, dict) else {}
        if self.is_mosaic_run:
            print(f"DEBUG QM (start_processing): Application des paramètres de Mosaïque depuis mosaic_settings_dict: {self.mosaic_settings_dict}")
            self.mosaic_alignment_mode = self.mosaic_settings_dict.get('alignment_mode', "local_fast_fallback")
            self.fa_orb_features = int(self.mosaic_settings_dict.get('fastalign_orb_features', 5000))
            self.fa_min_abs_matches = int(self.mosaic_settings_dict.get('fastalign_min_abs_matches', 10))
            self.fa_min_ransac_raw = int(self.mosaic_settings_dict.get('fastalign_min_ransac', 4)) 
            self.fa_ransac_thresh = float(self.mosaic_settings_dict.get('fastalign_ransac_thresh', 3.0))
            self.fa_daofind_fwhm = float(self.mosaic_settings_dict.get('fastalign_dao_fwhm', 3.5))
            self.fa_daofind_thr_sig = float(self.mosaic_settings_dict.get('fastalign_dao_thr_sig', 4.0))
            self.fa_max_stars_descr = int(self.mosaic_settings_dict.get('fastalign_dao_max_stars', 750))
            self.use_wcs_fallback_for_mosaic = (self.mosaic_alignment_mode == "local_fast_fallback")
            self.mosaic_drizzle_kernel = str(self.mosaic_settings_dict.get('kernel', "square"))
            self.mosaic_drizzle_pixfrac = float(self.mosaic_settings_dict.get('pixfrac', 1.0))
            self.mosaic_drizzle_fillval = str(self.mosaic_settings_dict.get('fillval', "0.0"))
            self.mosaic_drizzle_wht_threshold = float(self.mosaic_settings_dict.get('wht_threshold', 0.01))
            # Surcharge du facteur d'échelle global pour la mosaïque
            self.drizzle_scale = float(self.mosaic_settings_dict.get('mosaic_scale_factor', self.drizzle_scale)) 
            print(f"  -> Mode Mosaïque ACTIF. Align Mode: '{self.mosaic_alignment_mode}', Fallback WCS: {self.use_wcs_fallback_for_mosaic}")
            print(f"     Mosaic Drizzle: Kernel='{self.mosaic_drizzle_kernel}', Pixfrac={self.mosaic_drizzle_pixfrac:.2f}, Scale(global)={self.drizzle_scale}x")
        
        if self.drizzle_active_session and not self.is_mosaic_run:
            self.drizzle_kernel = str(drizzle_kernel)      
            self.drizzle_pixfrac = float(drizzle_pixfrac)  
            print(f"   -> Drizzle ACTIF (Standard). Mode: '{self.drizzle_mode}', Scale: {self.drizzle_scale:.1f}, Kernel: {self.drizzle_kernel}, Pixfrac: {self.drizzle_pixfrac:.2f}, WHT Thresh: {self.drizzle_wht_threshold:.3f}")
        
        requested_batch_size = batch_size 
        if requested_batch_size <= 0:
            sample_img_path_for_bsize = None
            if input_dir and os.path.isdir(input_dir): 
                fits_files_bsize = [f for f in os.listdir(input_dir) if f.lower().endswith(('.fit', '.fits'))]
                sample_img_path_for_bsize = os.path.join(input_dir, fits_files_bsize[0]) if fits_files_bsize else None
            try: 
                estimated_size = estimate_batch_size(sample_image_path=sample_img_path_for_bsize)
                self.batch_size = max(3, estimated_size) 
                self.update_progress(f"✅ Taille lot auto estimée et appliquée: {self.batch_size}", None)
            except Exception as est_err: 
                self.update_progress(f"⚠️ Erreur estimation taille lot: {est_err}. Utilisation défaut (10).", None)
                self.batch_size = 10
        else: 
            self.batch_size = max(3, int(requested_batch_size)) 
        self.update_progress(f"ⓘ Taille de lot effective pour le traitement : {self.batch_size}")
        print("DEBUG QM (start_processing): Fin Étape 1 - Configuration des paramètres de session.")
        


        # --- ÉTAPE 2 : PRÉPARATION DE L'IMAGE DE RÉFÉRENCE (shape ET WCS global si nécessaire) ---
        # ... (le reste de la méthode est inchangé) ...
        print("DEBUG QM (start_processing): Étape 2 - Préparation référence (shape ET WCS global si Drizzle/Mosaïque)...")
        reference_image_data_for_shape_determination = None 
        reference_header_for_shape_determination = None     
        ref_shape_hwc = None 
        
        try:
            potential_folders_for_shape = []
            if self.current_folder and os.path.isdir(self.current_folder): 
                potential_folders_for_shape.append(self.current_folder)
            if initial_additional_folders: 
                for add_f in initial_additional_folders:
                    abs_add_f = os.path.abspath(str(add_f)) 
                    if abs_add_f and os.path.isdir(abs_add_f) and abs_add_f not in potential_folders_for_shape:
                        potential_folders_for_shape.append(abs_add_f)
            if not potential_folders_for_shape: 
                raise RuntimeError("Aucun dossier d'entrée valide pour trouver une image de référence.")
            
            current_folder_to_scan_for_shape = None
            files_in_folder_for_shape = []
            for folder_path_iter in potential_folders_for_shape:
                try:
                    temp_files = sorted([f for f in os.listdir(folder_path_iter) if f.lower().endswith(('.fit', '.fits'))])
                    if temp_files: 
                        files_in_folder_for_shape = temp_files
                        current_folder_to_scan_for_shape = folder_path_iter
                        break 
                except Exception as e_listdir:
                    self.update_progress(f"Avertissement: Erreur lecture dossier '{folder_path_iter}' pour réf: {e_listdir}", "WARN")
            if not current_folder_to_scan_for_shape or not files_in_folder_for_shape:
                raise RuntimeError("Aucun fichier FITS trouvé dans les dossiers pour servir de référence.")

            self.aligner.correct_hot_pixels = self.correct_hot_pixels
            self.aligner.hot_pixel_threshold = self.hot_pixel_threshold
            self.aligner.neighborhood_size = self.neighborhood_size
            self.aligner.bayer_pattern = self.bayer_pattern
            self.aligner.reference_image_path = reference_path_ui or None 

            reference_image_data_for_shape_determination, reference_header_for_shape_determination = \
                self.aligner._get_reference_image(
                    current_folder_to_scan_for_shape, 
                    files_in_folder_for_shape,
                    self.output_folder 
                )


            if reference_image_data_for_shape_determination is None or reference_header_for_shape_determination is None:
                raise RuntimeError("Échec obtention de l'image de référence par self.aligner._get_reference_image.")
            
            ref_shape_initial = reference_image_data_for_shape_determination.shape
            if len(ref_shape_initial) == 2: 
                ref_shape_hwc = (ref_shape_initial[0], ref_shape_initial[1], 3)
            elif len(ref_shape_initial) == 3 and ref_shape_initial[2] == 3:
                ref_shape_hwc = ref_shape_initial
            else:
                raise RuntimeError(f"Shape de l'image de référence ({ref_shape_initial}) non supportée.")
            
            self.reference_header_for_wcs = reference_header_for_shape_determination.copy() 
            print(f"DEBUG QM (start_processing): Shape de référence HWC déterminée: {ref_shape_hwc}")

            ref_temp_processing_dir = os.path.join(self.output_folder, "temp_processing")
            reference_image_path_for_solving = os.path.join(ref_temp_processing_dir, "reference_image.fit")

            self.reference_wcs_object = None 
            
            if self.drizzle_active_session or self.is_mosaic_run: 
                print("DEBUG QM (start_processing): Plate-solving de la référence principale requis...")
                
                if not os.path.exists(reference_image_path_for_solving):
                    raise RuntimeError(f"Fichier de référence '{reference_image_path_for_solving}' non trouvé pour le solving.")

                if self.astrometry_solver is None: 
                    self.update_progress("❌ ERREUR CRITIQUE: AstrometrySolver non initialisé.", "ERROR")
                    return False 

                solver_settings_for_ref = {
                    "local_solver_preference": self.local_solver_preference,
                    "api_key": self.api_key,
                    "astap_path": self.astap_path,
                    "astap_data_dir": self.astap_data_dir,
                    "astap_search_radius": self.astap_search_radius,
                    "local_ansvr_path": self.local_ansvr_path,
                    "scale_est_arcsec_per_pix": self.reference_pixel_scale_arcsec, 
                    "scale_tolerance_percent": 20, 
                    "ansvr_timeout_sec": getattr(self, 'ansvr_timeout_sec', 120), 
                    "astap_timeout_sec": getattr(self, 'astap_timeout_sec', 120),
                    "astrometry_net_timeout_sec": getattr(self, 'astrometry_net_timeout_sec', 300)
                }
                
                self.update_progress("   [StartProcRefSolve] Tentative résolution astrométrique pour référence globale...")
                self.reference_wcs_object = self.astrometry_solver.solve(
                    image_path=reference_image_path_for_solving, 
                    fits_header=self.reference_header_for_wcs, 
                    settings=solver_settings_for_ref,
                    update_header_with_solution=True 
                )
                
                if self.reference_wcs_object and self.reference_wcs_object.is_celestial:
                    self.update_progress("   [StartProcRefSolve] Référence globale plate-solvée avec succès.")
                    if self.reference_wcs_object.pixel_shape is None:
                         nx_ref_hdr = self.reference_header_for_wcs.get('NAXIS1', ref_shape_hwc[1])
                         ny_ref_hdr = self.reference_header_for_wcs.get('NAXIS2', ref_shape_hwc[0])
                         self.reference_wcs_object.pixel_shape = (int(nx_ref_hdr), int(ny_ref_hdr))
                         print(f"    [StartProcRefSolve] pixel_shape ajouté/vérifié sur WCS réf: {self.reference_wcs_object.pixel_shape}")

                    try:
                        scales_deg_per_pix = proj_plane_pixel_scales(self.reference_wcs_object)
                        avg_scale_deg_per_pix = np.mean(np.abs(scales_deg_per_pix))
                        
                        if avg_scale_deg_per_pix > 1e-9: 
                            self.reference_pixel_scale_arcsec = avg_scale_deg_per_pix * 3600.0
                            self.update_progress(f"   [StartProcRefSolve] Échelle image de référence estimée à: {self.reference_pixel_scale_arcsec:.2f} arcsec/pix.", "INFO")
                            print(f"DEBUG QM: self.reference_pixel_scale_arcsec mis à jour à {self.reference_pixel_scale_arcsec:.3f} depuis le WCS de référence.")
                        else:
                            self.update_progress("   [StartProcRefSolve] Avertissement: Échelle calculée depuis WCS de référence trop faible ou invalide.", "WARN")
                    except Exception as e_scale_extract:
                        self.update_progress(f"   [StartProcRefSolve] Avertissement: Impossible d'extraire l'échelle du WCS de référence: {e_scale_extract}", "WARN")
                                         
                else: 
                    self.update_progress("   [StartProcRefSolve] ÉCHEC plate-solving réf. globale. Tentative WCS approximatif...", "WARN")
                    _cwfh_func_startup = None
                    try: from ..enhancement.drizzle_integration import _create_wcs_from_header as _cwfh_s; _cwfh_func_startup = _cwfh_s
                    except ImportError: self.update_progress("     -> Import _create_wcs_from_header échoué pour fallback.", "ERROR")
                    
                    if _cwfh_func_startup: 
                        self.reference_wcs_object = _cwfh_func_startup(self.reference_header_for_wcs) 
                    
                    if self.reference_wcs_object and self.reference_wcs_object.is_celestial:
                        nx_ref_hdr = self.reference_header_for_wcs.get('NAXIS1', ref_shape_hwc[1])
                        ny_ref_hdr = self.reference_header_for_wcs.get('NAXIS2', ref_shape_hwc[0])
                        self.reference_wcs_object.pixel_shape = (int(nx_ref_hdr), int(ny_ref_hdr))
                        self.update_progress("   [StartProcRefSolve] WCS approximatif pour référence globale créé.")
                    else: 
                        self.update_progress("❌ ERREUR CRITIQUE: Impossible d'obtenir un WCS pour la référence globale. Drizzle/Mosaïque ne peut continuer.", "ERROR")
                        return False 
            else: 
                print("DEBUG QM (start_processing): Plate-solving de la référence globale ignoré (mode Stacking Classique seul).")
                self.reference_wcs_object = None 
            
            if reference_image_data_for_shape_determination is not None:
                del reference_image_data_for_shape_determination
            gc.collect() 
            print("DEBUG QM (start_processing): Fin Étape 2 - Préparation référence et WCS global.")

        except Exception as e_ref_prep: 
            self.update_progress(f"❌ Erreur préparation référence/WCS: {e_ref_prep}", "ERROR")
            print(f"ERREUR QM (start_processing): Échec préparation référence/WCS : {e_ref_prep}"); traceback.print_exc(limit=2)
            return False
        
        print(f"DEBUG QM (start_processing): AVANT APPEL initialize():")
        print(f"  -> self.is_mosaic_run: {self.is_mosaic_run}")
        print(f"  -> self.drizzle_active_session: {self.drizzle_active_session}")
        print(f"  -> self.drizzle_mode: {self.drizzle_mode}")
        print(f"  -> self.reference_wcs_object IS None: {self.reference_wcs_object is None}")
        if self.reference_wcs_object and hasattr(self.reference_wcs_object, 'is_celestial') and self.reference_wcs_object.is_celestial: 
            print(f"     WCS Ref CTYPE: {self.reference_wcs_object.wcs.ctype if hasattr(self.reference_wcs_object, 'wcs') else 'N/A'}, PixelShape: {self.reference_wcs_object.pixel_shape}")
        else:
            print(f"     WCS Ref non disponible ou non céleste.")

        print(f"DEBUG QM (start_processing): Étape 3 - Appel à self.initialize() avec output_dir='{output_dir}', shape_ref_HWC={ref_shape_hwc}...")
        if not self.initialize(output_dir, ref_shape_hwc): 
            self.processing_active = False 
            print("ERREUR QM (start_processing): Échec de self.initialize().")
            return False
        print("DEBUG QM (start_processing): self.initialize() terminé avec succès.")

        print("DEBUG QM (start_processing): Étape 4 - Remplissage de la file d'attente...")
        initial_folders_to_add_count = 0
        with self.folders_lock:
            self.additional_folders = [] 
            if initial_additional_folders and isinstance(initial_additional_folders, list): 
                for folder_iter in initial_additional_folders:
                    abs_folder = os.path.abspath(str(folder_iter)) 
                    if os.path.isdir(abs_folder) and abs_folder not in self.additional_folders:
                        self.additional_folders.append(abs_folder); initial_folders_to_add_count += 1
        if initial_folders_to_add_count > 0: 
            self.update_progress(f"ⓘ {initial_folders_to_add_count} dossier(s) pré-ajouté(s) en attente.")
        self.update_progress(f"folder_count_update:{len(self.additional_folders)}") 

        initial_files_added = self._add_files_to_queue(self.current_folder) 
        if initial_files_added > 0: 
            self._recalculate_total_batches()
            self.update_progress(f"📋 {initial_files_added} fichiers initiaux ajoutés. Total lots estimé: {self.total_batches_estimated if self.total_batches_estimated > 0 else '?'}")
        elif not self.additional_folders: 
            self.update_progress("⚠️ Aucun fichier initial trouvé dans le dossier principal et aucun dossier supplémentaire en attente.")
        
        self.aligner.reference_image_path = reference_path_ui or None 

        print("DEBUG QM (start_processing V_StartProcessing_SaveDtypeOption_1): Démarrage du thread worker...") # Version Log
        self.processing_thread = threading.Thread(target=self._worker, name="StackerWorker")
        self.processing_thread.daemon = True 
        self.processing_thread.start()
        self.processing_active = True 
        
        self.update_progress("🚀 Thread de traitement démarré.")
        print("DEBUG QM (start_processing V_StartProcessing_SaveDtypeOption_1): Fin.") # Version Log
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



    def _process_and_save_drizzle_batch(self, batch_temp_filepaths_list, output_wcs_target, output_shape_target_hw, batch_num):
        """
        Traite un lot de fichiers FITS temporaires (contenant des images alignées et leur WCS de référence)
        en utilisant Drizzle et sauvegarde les fichiers science (CxHxW) et poids (HxW x3)
        intermédiaires pour ce lot.
        MODIFIED CRITICAL: Force ALL input data to be in [0, 65535] ADU range BEFORE Drizzle.add_image.
        Robustify weight_map handling.
        Version: V_ProcessAndSaveDrizzleBatch_DrizzleInputFix_5_ForceADUAllChannels
        """
        num_files_in_batch = len(batch_temp_filepaths_list)
        self.update_progress(f"💧 Traitement Drizzle du lot #{batch_num} ({num_files_in_batch} images)...")
        batch_start_time = time.time()
        print(f"DEBUG QM [_process_and_save_drizzle_batch V_ProcessAndSaveDrizzleBatch_DrizzleInputFix_5_ForceADUAllChannels]: Lot #{batch_num} avec {num_files_in_batch} images.")
        print(f"  -> WCS de sortie cible fourni: {'Oui' if output_wcs_target else 'Non'}, Shape de sortie cible: {output_shape_target_hw}")

        if not batch_temp_filepaths_list:
            self.update_progress(f"   - Warning: Lot Drizzle #{batch_num} vide.")
            return None, []
        if output_wcs_target is None or output_shape_target_hw is None:
            self.update_progress(f"   - ERREUR: WCS ou Shape de sortie manquant pour lot Drizzle #{batch_num}. Traitement annulé.", "ERROR")
            print(f"ERREUR QM [_process_and_save_drizzle_batch V_ProcessAndSaveDrizzleBatch_DrizzleInputFix_5_ForceADUAllChannels]: output_wcs_target ou output_shape_target_hw est None.")
            return None, []
        if not isinstance(output_wcs_target, WCS) or not output_wcs_target.is_celestial:
            self.update_progress(f"   - ERREUR: output_wcs_target invalide pour lot Drizzle #{batch_num}.", "ERROR")
            return None, []
        if not isinstance(output_shape_target_hw, tuple) or len(output_shape_target_hw) != 2 or \
           not all(isinstance(dim, int) and dim > 0 for dim in output_shape_target_hw):
            self.update_progress(f"   - ERREUR: output_shape_target_hw invalide pour lot Drizzle #{batch_num}.", "ERROR")
            return None, []

        num_output_channels = 3; channel_names = ['R', 'G', 'B']
        drizzlers_batch = []; output_images_batch = []; output_weights_batch = []
        try:
            print(f"DEBUG QM [_process_and_save_drizzle_batch V_ProcessAndSaveDrizzleBatch_DrizzleInputFix_5_ForceADUAllChannels]: Initialisation Drizzle pour lot #{batch_num}. Shape Sortie CIBLE: {output_shape_target_hw}.")
            for _ in range(num_output_channels):
                output_images_batch.append(np.zeros(output_shape_target_hw, dtype=np.float32))
                output_weights_batch.append(np.zeros(output_shape_target_hw, dtype=np.float32))
            for i in range(num_output_channels):
                driz_ch = Drizzle(out_img=output_images_batch[i], out_wht=output_weights_batch[i],
                                  out_shape=output_shape_target_hw, kernel=self.drizzle_kernel, fillval="0.0")
                drizzlers_batch.append(driz_ch)
            self.update_progress(f"   - Objets Drizzle initialisés pour lot #{batch_num}.")
        except Exception as init_err:
            self.update_progress(f"   - ERREUR: Échec init Drizzle pour lot #{batch_num}: {init_err}", "ERROR")
            print(f"ERREUR QM [_process_and_save_drizzle_batch V_ProcessAndSaveDrizzleBatch_DrizzleInputFix_5_ForceADUAllChannels]: Échec init Drizzle: {init_err}"); traceback.print_exc(limit=1)
            return None, []

        processed_in_batch_count = 0
        for i_file, temp_fits_filepath_item in enumerate(batch_temp_filepaths_list): 
            if self.stop_processing: break
            current_filename_for_log = os.path.basename(temp_fits_filepath_item)
            print(f"DEBUG QM [P&SDB_Loop]: Lot #{batch_num}, Fichier {i_file+1}/{num_files_in_batch}: '{current_filename_for_log}'")

            input_data_HxWxC_orig = None; wcs_input_from_file_header = None
            input_file_header_content = None; pixmap_for_this_file = None
            file_successfully_added_to_drizzle = False
            try:
                with fits.open(temp_fits_filepath_item, memmap=False) as hdul:
                    if not hdul or len(hdul) == 0 or hdul[0].data is None: raise IOError(f"FITS temp invalide/vide: {temp_fits_filepath_item}")
                    data_cxhxw = hdul[0].data.astype(np.float32)
                    if data_cxhxw.ndim!=3 or data_cxhxw.shape[0]!=num_output_channels: raise ValueError(f"Shape FITS temp {data_cxhxw.shape} != CxHxW")
                    input_data_HxWxC_orig = np.moveaxis(data_cxhxw, 0, -1) 
                    input_file_header_content = hdul[0].header 
                    with warnings.catch_warnings(): warnings.simplefilter("ignore"); wcs_input_from_file_header = WCS(input_file_header_content, naxis=2)
                    if not wcs_input_from_file_header.is_celestial: raise ValueError(f"WCS non céleste dans FITS temp")
                
                current_input_shape_hw = input_data_HxWxC_orig.shape[:2]
                y_in_coords, x_in_coords = np.indices(current_input_shape_hw)
                sky_coords_ra_deg, sky_coords_dec_deg = wcs_input_from_file_header.all_pix2world(x_in_coords.ravel(),y_in_coords.ravel(),0)
                x_output_pixels_flat, y_output_pixels_flat = output_wcs_target.all_world2pix(sky_coords_ra_deg,sky_coords_dec_deg,0)
                pixmap_for_this_file = np.dstack((x_output_pixels_flat.reshape(current_input_shape_hw), y_output_pixels_flat.reshape(current_input_shape_hw))).astype(np.float32)
                
                if pixmap_for_this_file is not None:
                    print(f"      DEBUG PIXMAP (P&SDB) Fichier {i_file+1}: Shape={pixmap_for_this_file.shape}")
                    if np.isnan(pixmap_for_this_file).any(): print(f"      WARN PIXMAP (P&SDB) Fichier {i_file+1}: CONTIENT DES NaN !")
                    if np.isinf(pixmap_for_this_file).any(): print(f"      WARN PIXMAP (P&SDB) Fichier {i_file+1}: CONTIENT DES INF !")
            except Exception as load_map_err:
                self.update_progress(f"      -> ERREUR P&SDB chargement/pixmap '{current_filename_for_log}': {load_map_err}", "WARN")
                print(f"ERREUR QM [P&SDB_Loop]: Échec chargement/pixmap '{current_filename_for_log}': {load_map_err}"); traceback.print_exc(limit=1)
                continue

            if pixmap_for_this_file is not None:
                try:
                    base_exptime = 1.0 
                    if input_file_header_content and 'EXPTIME' in input_file_header_content:
                        try: base_exptime = max(1e-6, float(input_file_header_content['EXPTIME']))
                        except (ValueError, TypeError): pass
                    
                    # --- CRITICAL FIX 1: Force ALL input data to [0, 65535] ADU range BEFORE Drizzle.add_image ---
                    input_data_HxWxC_adu_scaled = input_data_HxWxC_orig.copy()
                    
                    current_max_for_batch_adu = np.nanmax(input_data_HxWxC_adu_scaled)
                    if current_max_for_batch_adu <= 1.0 + 1e-5 and current_max_for_batch_adu > 0:
                        input_data_HxWxC_adu_scaled = input_data_HxWxC_adu_scaled * 65535.0
                        print(f"      DEBUG: File {i_file+1} FORCED rescaled from [0,1] to [0,65535] for Drizzle input. Range: [{np.min(input_data_HxWxC_adu_scaled):.4g}, {np.max(input_data_HxWxC_adu_scaled):.4g}]")
                    else:
                        print(f"      DEBUG: File {i_file+1} kept original range for Drizzle input: [{np.min(input_data_HxWxC_adu_scaled):.4g}, {np.max(input_data_HxWxC_adu_scaled):.4g}]")
                    
                    # Clip negative values and handle NaNs/Infs
                    input_data_HxWxC_cleaned = np.nan_to_num(np.clip(input_data_HxWxC_adu_scaled, 0.0, None), nan=0.0, posinf=0.0, neginf=0.0)
                    # --- END CRITICAL FIX 1 ---

                    # --- CRITICAL FIX 2: Robustify weight_map ---
                    # For _process_and_save_drizzle_batch, the original pixel mask is not readily available from temp file.
                    # So we use a uniform weight map here. This should be improved if possible by saving/loading the mask.
                    effective_weight_map = np.ones(current_input_shape_hw, dtype=np.float32)
                    print(f"      DEBUG: File {i_file+1}, uniform weight_map used for Drizzle.add_image. Range: [{np.min(effective_weight_map):.3f}, {np.max(effective_weight_map):.3f}]")
                    # --- END CRITICAL FIX 2 ---

                    for ch_index in range(num_output_channels):
                        channel_data_2d_clean = input_data_HxWxC_cleaned[..., ch_index]
                        
                        drizzlers_batch[ch_index].add_image(data=channel_data_2d_clean, pixmap=pixmap_for_this_file, exptime=base_exptime,
                                                            pixfrac=self.drizzle_pixfrac, in_units='counts', weight_map=effective_weight_map)
                    file_successfully_added_to_drizzle = True
                except Exception as drizzle_add_err:
                    self.update_progress(f"      -> ERREUR P&SDB add_image pour '{current_filename_for_log}': {drizzle_add_err}", "WARN")
                    print(f"ERREUR QM [P&SDB_Loop]: Échec add_image '{current_filename_for_log}': {drizzle_add_err}"); traceback.print_exc(limit=1)
            
            if file_successfully_added_to_drizzle:
                processed_in_batch_count += 1
                print(f"  [P&SDB_Loop]: Fichier '{current_filename_for_log}' AJOUTÉ. processed_in_batch_count = {processed_in_batch_count}")
            else:
                print(f"  [P&SDB_Loop]: Fichier '{current_filename_for_log}' NON ajouté (erreur pixmap ou add_image).")
            
            del input_data_HxWxC_orig, input_data_HxWxC_adu_scaled, input_data_HxWxC_cleaned, wcs_input_from_file_header, input_file_header_content, pixmap_for_this_file
            gc.collect()
        print(f"DEBUG QM [P&SDB_Loop]: Fin boucle pour lot #{batch_num}. Total processed_in_batch_count = {processed_in_batch_count}")
        
        if processed_in_batch_count == 0:
            self.update_progress(f"   - Erreur: Aucun fichier du lot Drizzle #{batch_num} n'a pu être traité (processed_in_batch_count est 0).", "ERROR")
            del drizzlers_batch, output_images_batch, output_weights_batch; gc.collect()
            return None, []

        batch_output_dir = self.drizzle_batch_output_dir; os.makedirs(batch_output_dir, exist_ok=True)
        base_out_filename = f"batch_{batch_num:04d}_s{self.drizzle_scale:.1f}p{self.drizzle_pixfrac:.1f}{self.drizzle_kernel}"
        out_filepath_sci = os.path.join(batch_output_dir, f"{base_out_filename}_sci.fits"); out_filepaths_wht = []
        
        print(f"DEBUG QM [P&SDB_Save]: Début sauvegarde pour lot #{batch_num}. SCI path: {out_filepath_sci}")
        try:
            final_sci_data_batch_hwc = np.stack(output_images_batch, axis=-1).astype(np.float32)
            final_sci_data_to_save = np.moveaxis(final_sci_data_batch_hwc, -1, 0).astype(np.float32)
            print(f"  [P&SDB_Save]: Données SCI prêtes pour écriture. Shape CxHxW: {final_sci_data_to_save.shape}")
            final_header_sci = output_wcs_target.to_header(relax=True) 
            final_header_sci['NINPUTS'] = (processed_in_batch_count, f'Valid input images for Drizzle batch {batch_num}')
            final_header_sci['ISCALE'] = (self.drizzle_scale, 'Drizzle scale factor'); final_header_sci['PIXFRAC'] = (self.drizzle_pixfrac, 'Drizzle pixfrac')
            final_header_sci['KERNEL'] = (self.drizzle_kernel, 'Drizzle kernel'); final_header_sci['HISTORY'] = f'Drizzle Batch {batch_num}'
            final_header_sci['BUNIT'] = 'Counts/s'; final_header_sci['NAXIS'] = 3
            final_header_sci['NAXIS1'] = final_sci_data_to_save.shape[2]; final_header_sci['NAXIS2'] = final_sci_data_to_save.shape[1]
            final_header_sci['NAXIS3'] = final_sci_data_to_save.shape[0]; final_header_sci['CTYPE3'] = 'CHANNEL'
            try: final_header_sci['CHNAME1'] = 'R'; final_header_sci['CHNAME2'] = 'G'; final_header_sci['CHNAME3'] = 'B'
            except Exception: pass
            print(f"  [P&SDB_Save]: Header SCI prêt. Tentative d'écriture...")
            fits.writeto(out_filepath_sci, final_sci_data_to_save, final_header_sci, overwrite=True, checksum=False, output_verify='ignore')
            self.update_progress(f"      -> Science lot #{batch_num} sauvegardé: {os.path.basename(out_filepath_sci)}")
            print(f"DEBUG QM [P&SDB_Save]: Fichier SCI lot #{batch_num} sauvegardé: {out_filepath_sci}")
            del final_sci_data_batch_hwc, final_sci_data_to_save; gc.collect()
        except Exception as e_save_sci:
            self.update_progress(f"   - ERREUR sauvegarde science lot #{batch_num}: {e_save_sci}", "ERROR")
            print(f"ERREUR QM [P&SDB_Save]: Échec sauvegarde SCI: {e_save_sci}"); traceback.print_exc(limit=1)
            del drizzlers_batch, output_images_batch, output_weights_batch; gc.collect()
            return None, []

        for i_ch_save in range(num_output_channels):
            ch_name = channel_names[i_ch_save]
            out_filepath_wht_ch = os.path.join(batch_output_dir, f"{base_out_filename}_wht_{ch_name}.fits")
            out_filepaths_wht.append(out_filepath_wht_ch)
            try:
                print(f"  [P&SDB_Save]: Préparation WHT pour canal {ch_name} lot #{batch_num}. Path: {out_filepath_wht_ch}")
                wht_data_to_save_ch = output_weights_batch[i_ch_save].astype(np.float32)
                wht_header_ch = output_wcs_target.to_header(relax=True) 
                for key_clean in ['NAXIS3','CTYPE3','CRPIX3','CRVAL3','CDELT3','CUNIT3','PC3_1','PC3_2','PC3_3','PC1_3','PC2_3','CHNAME1','CHNAME2','CHNAME3']:
                    if key_clean in wht_header_ch: del wht_header_ch[key_clean]
                wht_header_ch['NAXIS'] = 2; wht_header_ch['NAXIS1'] = wht_data_to_save_ch.shape[1]
                wht_header_ch['NAXIS2'] = wht_data_to_save_ch.shape[0]
                wht_header_ch['HISTORY'] = f'Drizzle Weights ({ch_name}) Batch {batch_num}'; wht_header_ch['NINPUTS'] = processed_in_batch_count
                wht_header_ch['BUNIT'] = 'Weight'
                print(f"    [P&SDB_Save]: Header WHT {ch_name} prêt. Tentative d'écriture...")
                fits.writeto(out_filepath_wht_ch, wht_data_to_save_ch, wht_header_ch, overwrite=True, checksum=False, output_verify='ignore')
                print(f"  [P&SDB_Save]: Fichier WHT lot ({ch_name}) #{batch_num} sauvegardé.")
            except Exception as e_save_wht:
                self.update_progress(f"   - ERREUR sauvegarde poids {ch_name} lot #{batch_num}: {e_save_wht}", "ERROR")
                print(f"ERREUR QM [P&SDB_Save]: Échec sauvegarde WHT {ch_name}: {e_save_wht}"); traceback.print_exc(limit=1)
                if os.path.exists(out_filepath_sci):
                    try: os.remove(out_filepath_sci)
                    except Exception: pass
                for wht_f_clean in out_filepaths_wht:
                    if os.path.exists(wht_f_clean):
                        try: os.remove(wht_f_clean)
                        except Exception: pass
                del drizzlers_batch, output_images_batch, output_weights_batch; gc.collect()
                return None, []

        self.update_progress(f"   -> Sauvegarde lot Drizzle #{batch_num} terminée ({time.time() - batch_start_time:.2f}s).")
        del drizzlers_batch, output_images_batch, output_weights_batch; gc.collect()
        return out_filepath_sci, out_filepaths_wht



######################################################################################################################################################

