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
import astroalign as aa
import cv2
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS, FITSFixedWarning
from ccdproc import CCDData, combine as ccdproc_combine
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

    def __init__(self, *args, **kwargs):
        print("\n==== DÉBUT INITIALISATION SeestarQueuedStacker (Réorganisé) ====")
        
        # --- 1. Attributs Critiques et Simples EN PREMIER ---
        print("  -> Initialisation attributs simples...")
        # Flags & Control
        self.processing_active = False
        self.stop_processing = False
        self.processing_error = None
        self.is_mosaic_run = False
        self.drizzle_active_session = False
        self.perform_cleanup = True
        self.use_quality_weighting = False
        self.weight_by_snr = True
        self.weight_by_stars = True
        self.correct_hot_pixels = True
        self.apply_chroma_correction = True
        # Callbacks
        self.progress_callback = None
        self.preview_callback = None
        # Queue & Threading
        self.queue = Queue()
        self.folders_lock = threading.Lock()  # <<< Défini tôt
        self.processing_thread = None
        # File & Folder Management
        self.processed_files = set()
        self.additional_folders = []
        self.current_folder = None
        self.output_folder = None
        self.unaligned_folder = None
        self.drizzle_temp_dir = None
        self.drizzle_batch_output_dir = None
        self.final_stacked_path = None
        # Astrometry & WCS Refs
        self.api_key = None
        self.reference_wcs_object = None  # À utiliser pour l'astrométrie de la mosaïque
        self.reference_header_for_wcs = None
        self.reference_pixel_scale_arcsec = None
        self.drizzle_output_wcs = None
        self.drizzle_output_shape_hw = None
        # Batch & Cumulative Data
        self.current_batch_data = []
        self.current_stack_data = None
        self.current_stack_header = None
        self.images_in_cumulative_stack = 0
        self.total_exposure_seconds = 0.0
        self.cumulative_drizzle_data = None
        self.cumulative_drizzle_wht = None
        self.intermediate_drizzle_batch_files = []
        # Processing Parameters
        self.stacking_mode = "kappa-sigma"
        self.kappa = 2.5
        self.batch_size = 10
        self.hot_pixel_threshold = 3.0
        self.neighborhood_size = 5
        self.bayer_pattern = "GRBG"
        self.drizzle_mode = "Final"
        self.drizzle_scale = 2.0
        self.drizzle_wht_threshold = 0.7
        self.drizzle_kernel = "square"
        self.drizzle_pixfrac = 1.0
        self.snr_exponent = 1.0
        self.stars_exponent = 0.5
        self.min_weight = 0.1
        # Statistics
        self.files_in_queue = 0
        self.processed_files_count = 0
        self.aligned_files_count = 0
        self.stacked_batches_count = 0
        self.total_batches_estimated = 0
        self.failed_align_count = 0
        self.failed_stack_count = 0
        self.skipped_files_count = 0
        print("  -> Attributs simples initialisés.")

        # --- 2. Instanciations de Classes (dans des try/except) ---
        try:
            print("  -> Tentative instanciation ChromaticBalancer...")
            self.chroma_balancer = ChromaticBalancer(border_size=50, blur_radius=15)
            print("     ✓ ChromaticBalancer OK.")
        except Exception as e_cb:
            print(f"  -> ERREUR ChromaticBalancer: {e_cb}")
            self.chroma_balancer = None; raise

        try:
            print("  -> Tentative instanciation SeestarAligner...")
            self.aligner = SeestarAligner()
            print("     ✓ SeestarAligner OK.")
        except Exception as e_align:
            print(f"  -> ERREUR SeestarAligner: {e_align}")
            self.aligner = None; raise

        print("==== FIN INITIALISATION SeestarQueuedStacker (Réorganisé) ====\n")



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
        (Version Corrigée pour Batch Processing, TypeError & Import Tardif)
        """
        print("\n" + "="*10 + " DEBUG [Worker Start]: Initialisation " + "="*10)
        self.processing_active = True
        self.processing_error = None
        start_time_session = time.monotonic()

        # --- Initialisation des variables de session ---
        reference_image_data = None; reference_header = None
        self.reference_wcs_object = None; self.reference_header_for_wcs = None
        self.reference_pixel_scale_arcsec = None
        self.drizzle_output_wcs = None; self.drizzle_output_shape_hw = None

        # --- Listes pour accumuler les résultats ---
        self.current_batch_data = [] # Classique [(data, header, scores)]
        local_batch_temp_files = [] # Drizzle Incrémental [temp_path]
        local_drizzle_final_batch_data = [] # Drizzle Final [(data, hdr, wcs_ref)]
        self.intermediate_drizzle_batch_files = [] # Drizzle Final [(sci_path, [wht_paths])]
        all_aligned_files_with_info = [] # Mosaïque [(aligned_data, header, quality_scores, wcs_indiv)]

        print(f"DEBUG [Worker Start]: Mode reçu -> is_mosaic_run={self.is_mosaic_run}, drizzle_active_session={self.drizzle_active_session}, drizzle_mode='{self.drizzle_mode}'")

        # --- IMPORTS TARDIFS (pour éviter dépendances circulaires au chargement) ---
        # Ces modules sont nécessaires pour le traitement dans la boucle ou la finalisation
        solve_image_wcs_func = None
        DrizzleProcessor_class = None
        load_drizzle_temp_file_func = None
        create_wcs_from_header_func = None
        try:
            from ..enhancement.astrometry_solver import solve_image_wcs as solve_image_wcs_func
            print("DEBUG [_worker]: Import tardif solve_image_wcs OK.")
        except ImportError: print("ERREUR [_worker]: Échec import tardif solve_image_wcs.")
        try:
            from ..enhancement.drizzle_integration import _load_drizzle_temp_file as load_drizzle_temp_file_func
            from ..enhancement.drizzle_integration import DrizzleProcessor as DrizzleProcessor_class
            from ..enhancement.drizzle_integration import _create_wcs_from_header as create_wcs_from_header_func
            print("DEBUG [_worker]: Import tardif drizzle_integration OK.")
        except ImportError: print("ERREUR [_worker]: Échec import tardif drizzle_integration.")
        # L'import de mosaic_processor reste dans la branche de finalisation mosaïque
        # --- FIN IMPORTS TARDIFS ---

        # ============================================================
        # --- DEBUT DU BLOC TRY PRINCIPAL (couvre tout le worker) ---
        # ============================================================
        try:
            # ----------------------------------------------------
            # Étape 1: Préparation Image Référence et WCS/Échelle
            # ----------------------------------------------------
            self.update_progress("⭐ Préparation image référence...")
            if not self.current_folder or not os.path.isdir(self.current_folder): raise RuntimeError(f"Dossier entrée invalide: {self.current_folder}")
            initial_files = sorted([f for f in os.listdir(self.current_folder) if f.lower().endswith(('.fit', '.fits'))])
            if not initial_files: raise RuntimeError(f"Aucun FITS initial trouvé dans {self.current_folder}")
            self.aligner.correct_hot_pixels = self.correct_hot_pixels; self.aligner.hot_pixel_threshold = self.hot_pixel_threshold
            self.aligner.neighborhood_size = self.neighborhood_size; self.aligner.bayer_pattern = self.bayer_pattern
            reference_image_data, reference_header = self.aligner._get_reference_image(self.current_folder, initial_files)
            if reference_image_data is None or reference_header is None: raise RuntimeError("Échec obtention image/header référence.")
            self.reference_header_for_wcs = reference_header.copy(); self.update_progress("   -> Validation/Génération WCS Référence...")
            # --- Utiliser la fonction importée tardivement ---
            if create_wcs_from_header_func:
                 local_ref_wcs_obj = create_wcs_from_header_func(reference_header)
            else: local_ref_wcs_obj = None
            # --- Fin utilisation ---
            if local_ref_wcs_obj is None or not local_ref_wcs_obj.is_celestial: raise RuntimeError("Impossible d'obtenir WCS référence valide.")
            ref_naxis1 = reference_header.get('NAXIS1'); ref_naxis2 = reference_header.get('NAXIS2')
            if ref_naxis1 and ref_naxis2: local_ref_wcs_obj.pixel_shape = (ref_naxis1, ref_naxis2)
            self.reference_wcs_object = local_ref_wcs_obj
            try: scale_matrix = self.reference_wcs_object.pixel_scale_matrix; self.reference_pixel_scale_arcsec = np.mean(np.abs(np.diag(scale_matrix))) * 3600.0
            except Exception as scale_err: print(f"   - WARNING: Impossible calculer échelle pixel réf: {scale_err}")
            self.aligner._save_reference_image(reference_image_data, reference_header, self.output_folder)
            self.update_progress("⭐ Image de référence et WCS prêts.", 5)
            self.drizzle_output_wcs = None; self.drizzle_output_shape_hw = None # Reporté

            # ----------------------------------------------------
            # Étape 2: Boucle de traitement de la file
            # ----------------------------------------------------
            self._recalculate_total_batches() # Calculer total_batches_est
            self.update_progress(f"▶️ Démarrage boucle traitement (File: {self.files_in_queue} | Lots Est.: {self.total_batches_estimated if self.total_batches_estimated > 0 else '?'})")

            while not self.stop_processing:
                file_path = None; aligned_data = None; header = None; quality_scores = None; wcs_object_indiv = None
                try: # --- Try interne pour une image ---
                    file_path = self.queue.get(timeout=1.0)
                    file_name = os.path.basename(file_path)

                    # --- AJOUTER L'IMPORT DE SOLVE_IMAGE_WCS ICI (s'il est nécessaire dans _process_file) ---
                    # Note: Normalement _process_file génère seulement le WCS, mais si le solve est déplacé ici :
                    if self.is_mosaic_run and solve_image_wcs_func is None:
                         # Si on est en mosaïque et que l'import a échoué, on ne peut pas continuer
                         raise ImportError("Solveur WCS non importé mais requis pour la mosaïque.")
                    # --- FIN AJOUT ---

                    aligned_data, header, quality_scores, wcs_object_indiv = self._process_file(
                        file_path, reference_image_data # _process_file a besoin de solve_image_wcs_func si is_mosaic_run
                    )
                    self.processed_files_count += 1

                    if aligned_data is not None:
                        self.aligned_files_count += 1
                        # --- Branche Mosaïque ---
                        if self.is_mosaic_run:
                            print(f"DEBUG [_worker/Loop]: Stockage info MOSAIC pour {file_name}")
                            current_info = (aligned_data, header, quality_scores, wcs_object_indiv)
                            all_aligned_files_with_info.append(current_info)
                        # --- Branche NON-Mosaïque ---
                        else:
                            print(f"DEBUG [_worker/Loop]: Traitement BATCH pour {file_name}")
                            data_for_batch=aligned_data; header_for_batch=header; scores_for_batch=quality_scores; wcs_for_batch=wcs_object_indiv

                            if self.drizzle_active_session and self.drizzle_mode == "Final":
                                if wcs_for_batch:
                                    local_drizzle_final_batch_data.append((data_for_batch, header_for_batch, self.reference_wcs_object))
                                    print(f"  -> Ajouté Drizzle Final lot ({len(local_drizzle_final_batch_data)}/{self.batch_size})")
                                    if len(local_drizzle_final_batch_data) >= self.batch_size:
                                        if self.drizzle_output_wcs is None:
                                            try: self.drizzle_output_wcs, self.drizzle_output_shape_hw = self._create_drizzle_output_wcs(self.reference_wcs_object, reference_image_data.shape[:2], self.drizzle_scale)
                                            except Exception as e: raise RuntimeError(f"Echec création grille sortie Drizzle: {e}") from e
                                        self.stacked_batches_count += 1
                                        sci_path, wht_paths = self._process_and_save_drizzle_batch(local_drizzle_final_batch_data, self.drizzle_output_wcs, self.drizzle_output_shape_hw, self.stacked_batches_count)
                                        if sci_path and wht_paths: self.intermediate_drizzle_batch_files.append((sci_path, wht_paths))
                                        else: self.failed_stack_count += len(local_drizzle_final_batch_data)
                                        local_drizzle_final_batch_data = []
                                else: self.skipped_files_count += 1; self.update_progress(f"   ⚠️ {file_name} ignoré Drizzle Final (WCS Généré Invalide).")

                            elif self.drizzle_active_session and self.drizzle_mode == "Incremental":
                                temp_filepath_incr = self._save_drizzle_input_temp(data_for_batch, header_for_batch)
                                if temp_filepath_incr:
                                    local_batch_temp_files.append(temp_filepath_incr)
                                    print(f"  -> Ajouté Drizzle Incr lot ({len(local_batch_temp_files)}/{self.batch_size})")
                                    if len(local_batch_temp_files) >= self.batch_size:
                                        self.stacked_batches_count += 1
                                        self._process_incremental_drizzle_batch(local_batch_temp_files, self.stacked_batches_count, self.total_batches_estimated)
                                        local_batch_temp_files = []
                                else: self.skipped_files_count += 1; self.update_progress(f"   ⚠️ {file_name} ignoré Drizzle Incr (Échec sauvegarde temp).")

                            else: # Mode Classique
                                self.current_batch_data.append((data_for_batch, header_for_batch, scores_for_batch))
                                print(f"  -> Ajouté Classique lot ({len(self.current_batch_data)}/{self.batch_size})")
                                if len(self.current_batch_data) >= self.batch_size:
                                    self.stacked_batches_count += 1
                                    self._process_completed_batch(self.stacked_batches_count, self.total_batches_estimated)
                                    self.current_batch_data = []

                            # Nettoyage Mémoire (non-mosaïque)
                            print(f"   -> Nettoyage mémoire image {file_name} (non-mosaïque)")
                            del aligned_data, header, quality_scores, wcs_object_indiv
                            del data_for_batch, header_for_batch, scores_for_batch, wcs_for_batch
                            gc.collect()

                    self.queue.task_done()
                    # Mise à jour Progression/ETA
                    current_progress = (self.processed_files_count / self.files_in_queue) * 100 if self.files_in_queue > 0 else 0
                    elapsed_time_session = time.monotonic() - start_time_session; time_per_file = elapsed_time_session / self.processed_files_count if self.processed_files_count > 0 else 0
                    remaining_files = self.files_in_queue - self.processed_files_count; eta_seconds = remaining_files * time_per_file if time_per_file > 0 else 0
                    h_eta, rem_eta = divmod(int(eta_seconds), 3600); m_eta, s_eta = divmod(rem_eta, 60); time_str = f"{h_eta:02}:{m_eta:02}:{s_eta:02}"
                    progress_msg = f"📊 ({self.processed_files_count}/{self.files_in_queue}) {file_name} | ETA: {time_str}"; self.update_progress(progress_msg, current_progress)
                    if self.processed_files_count % 20 == 0: gc.collect()

                except Empty: # Gestion file vide et dossiers sup
                    self.update_progress("ⓘ File vide. Vérification batch final / dossiers sup...")
                    # --- Traiter dernier lot partiel (SI PAS MOSAÏQUE) ---
                    if not self.is_mosaic_run:
                        print("DEBUG [_worker/EmptyQueue]: Traitement dernier lot partiel (Non-Mosaïque)...")
                        if self.drizzle_active_session and self.drizzle_mode == "Final" and local_drizzle_final_batch_data:
                            print(f"   -> Dernier lot Drizzle Final ({len(local_drizzle_final_batch_data)} images)")
                            if self.drizzle_output_wcs is None:
                                try: self.drizzle_output_wcs, self.drizzle_output_shape_hw = self._create_drizzle_output_wcs(self.reference_wcs_object, reference_image_data.shape[:2], self.drizzle_scale)
                                except Exception as e: raise RuntimeError(f"Echec création grille sortie Drizzle final: {e}") from e
                            self.stacked_batches_count += 1
                            sci_path, wht_paths = self._process_and_save_drizzle_batch(local_drizzle_final_batch_data, self.drizzle_output_wcs, self.drizzle_output_shape_hw, self.stacked_batches_count)
                            if sci_path and wht_paths: self.intermediate_drizzle_batch_files.append((sci_path, wht_paths))
                            else: self.failed_stack_count += len(local_drizzle_final_batch_data)
                            local_drizzle_final_batch_data = []
                        elif self.drizzle_active_session and self.drizzle_mode == "Incremental" and local_batch_temp_files:
                            print(f"   -> Dernier lot Drizzle Incrémental ({len(local_batch_temp_files)} images)")
                            self.stacked_batches_count += 1
                            self._process_incremental_drizzle_batch(local_batch_temp_files, self.stacked_batches_count, self.total_batches_estimated)
                            local_batch_temp_files = []
                        elif not self.drizzle_active_session and self.current_batch_data:
                            print(f"   -> Dernier lot Classique ({len(self.current_batch_data)} images)")
                            self.stacked_batches_count += 1
                            self._process_completed_batch(self.stacked_batches_count, self.total_batches_estimated)
                            self.current_batch_data = []
                    # --- Traiter dossier supplémentaire ---
                    folder_to_process = None;
                    with self.folders_lock:
                        if self.additional_folders: folder_to_process = self.additional_folders.pop(0); self.update_progress(f"folder_count_update:{len(self.additional_folders)}")
                    if folder_to_process:
                        self.current_folder = folder_to_process; self.update_progress(f"📂 Traitement dossier supplémentaire: {os.path.basename(folder_to_process)}")
                        self._add_files_to_queue(folder_to_process); self._recalculate_total_batches()
                        self.update_progress(f"   -> Fichiers ajoutés. Total Queue={self.files_in_queue}, Lots Est.={self.total_batches_estimated if self.total_batches_estimated > 0 else '?'}")
                        continue # Revenir au début boucle
                    else:
                        self.update_progress("✅ Fin file/dossiers.")
                        break # Sortir boucle principale
                except Exception as e_inner_loop: # Erreur fichier générale
                    error_context=f" de {file_name}" if file_path else ""; self.update_progress(f"❌ Erreur boucle worker{error_context}: {e_inner_loop}"); traceback.print_exc(limit=3); self.processing_error = f"Erreur: {e_inner_loop}";
                    if file_path: self.skipped_files_count += 1;
                    try: self.queue.task_done()
                    except ValueError: pass
                    time.sleep(0.1)
                finally: # Nettoyage Mémoire Itération
                    try:
                        if aligned_data is not None: del aligned_data
                        if header is not None: del header
                        if quality_scores is not None: del quality_scores
                        if wcs_object_indiv is not None: del wcs_object_indiv
                    except NameError: pass
            # --- FIN BOUCLE WHILE ---

            # --- Traitement dernier lot partiel (si sorti normalement ET non-mosaïque) ---
            if not self.stop_processing and not self.is_mosaic_run:
                print("DEBUG [_worker/AfterLoop]: Traitement dernier lot partiel (sortie normale boucle)...")
                if self.drizzle_active_session and self.drizzle_mode == "Final" and local_drizzle_final_batch_data:
                    print(f"   -> Dernier lot Drizzle Final ({len(local_drizzle_final_batch_data)} images)")
                    if self.drizzle_output_wcs is None:
                        try: self.drizzle_output_wcs, self.drizzle_output_shape_hw = self._create_drizzle_output_wcs(self.reference_wcs_object, reference_image_data.shape[:2], self.drizzle_scale)
                        except Exception as e: raise RuntimeError(f"Echec création grille sortie Drizzle final: {e}") from e
                    self.stacked_batches_count += 1
                    sci_path, wht_paths = self._process_and_save_drizzle_batch(local_drizzle_final_batch_data, self.drizzle_output_wcs, self.drizzle_output_shape_hw, self.stacked_batches_count)
                    if sci_path and wht_paths: self.intermediate_drizzle_batch_files.append((sci_path, wht_paths))
                    else: self.failed_stack_count += len(local_drizzle_final_batch_data)
                    local_drizzle_final_batch_data = []
                elif self.drizzle_active_session and self.drizzle_mode == "Incremental" and local_batch_temp_files:
                    print(f"   -> Dernier lot Drizzle Incrémental ({len(local_batch_temp_files)} images)")
                    self.stacked_batches_count += 1
                    self._process_incremental_drizzle_batch(local_batch_temp_files, self.stacked_batches_count, self.total_batches_estimated)
                    local_batch_temp_files = []
                elif not self.drizzle_active_session and self.current_batch_data:
                    print(f"   -> Dernier lot Classique ({len(self.current_batch_data)} images)")
                    self.stacked_batches_count += 1
                    self._process_completed_batch(self.stacked_batches_count, self.total_batches_estimated)
                    self.current_batch_data = []

            # ==================================================
            # --- 3. Étape Finale (après la boucle) ---
            # ==================================================
            print("DEBUG [_worker]: Fin boucle principale. Début logique finalisation...")
            final_result_data = None; final_result_header = None

            # --- Nettoyage mémoire si non-mosaïque ---
            if not self.is_mosaic_run:
                print("DEBUG [_worker/Finalize]: Nettoyage all_aligned_files_with_info (mode non-mosaïque)...")
                all_aligned_files_with_info = [] # Vider la liste
                gc.collect()

            if self.stop_processing: # Si arrêt utilisateur
                self.update_progress("🛑 Traitement interrompu avant finalisation.")
                if not self.is_mosaic_run: # Sauvegarde partielle seulement si pas mosaïque
                    if self.drizzle_mode=="Incremental" and self.cumulative_drizzle_data is not None:
                        final_result_data=self.cumulative_drizzle_data; final_result_header=self.current_stack_header
                        self._save_final_stack("_drizzle_incr_stopped", True)
                    elif not self.drizzle_active_session and self.current_stack_data is not None:
                        final_result_data=self.current_stack_data; final_result_header=self.current_stack_header
                        self._save_final_stack("_classic_stopped", True)
                    elif self.drizzle_mode=="Final" and self.intermediate_drizzle_batch_files:
                        self.update_progress("ⓘ Lots Drizzle Final interm. conservés si nettoyage désactivé.")
                self.final_stacked_path = None # Pas de stack final officiel

            else: # Traitement Normal Terminé
                print("DEBUG [_worker]: Traitement normal terminé. Branchement finalisation par mode...")
                # --- Branche Mosaïque ---
                if self.is_mosaic_run:
                    print(f"DEBUG [_worker]: Branche finalisation MOSAÏQUE ({len(all_aligned_files_with_info)} images)...")
                    self.update_progress("🖼️ Finalisation Mode Mosaïque...")
                    if all_aligned_files_with_info:
                        # --- IMPORT TARDIF MOSAIC PROCESSOR ICI ---
                        try:
                            from ..enhancement.mosaic_processor import process_mosaic_from_aligned_files
                            print("DEBUG [_worker/Finalize]: Import TARDIF de process_mosaic_from_aligned_files réussi.")
                            final_result_data, final_result_header = process_mosaic_from_aligned_files(all_aligned_files_with_info, self, self.update_progress)
                            if final_result_data is None: self.processing_error = self.processing_error or "Échec orchestration mosaïque."
                        except ImportError as imp_err_mosaic: self.update_progress(f"❌ Erreur Import Tardif Mosaic Processor: {imp_err_mosaic}"); self.processing_error = "Erreur Import Mosaic Processor"
                        except Exception as mosaic_e: self.update_progress(f"❌ Erreur orchestration mosaïque: {mosaic_e}"); traceback.print_exc(limit=2); self.processing_error = str(mosaic_e)
                    else: self.update_progress("⚠️ Aucune image valide pour créer la mosaïque.")

                # --- Branche Drizzle Final (Simple Champ) ---
                elif self.drizzle_active_session and self.drizzle_mode == "Final":
                    print(f"DEBUG [_worker]: Branche finalisation DRIZZLE FINAL ({len(self.intermediate_drizzle_batch_files)} lots)...")
                    if self.intermediate_drizzle_batch_files:
                        if self.drizzle_output_wcs is None:
                           try: self.drizzle_output_wcs, self.drizzle_output_shape_hw = self._create_drizzle_output_wcs(self.reference_wcs_object, reference_image_data.shape[:2], self.drizzle_scale)
                           except Exception as e: self.processing_error=str(e); raise e
                        final_combined_sci, _ = self._combine_intermediate_drizzle_batches(self.intermediate_drizzle_batch_files, self.drizzle_output_wcs, self.drizzle_output_shape_hw)
                        if final_combined_sci is not None:
                            final_result_data = final_combined_sci; final_result_header = self._update_header_for_drizzle_final()
                            self.images_in_cumulative_stack = self.aligned_files_count
                        else: self.processing_error = "Échec comb. Drizzle Final"
                    else: self.update_progress("⚠️ Aucun lot Drizzle Final interm.")

                # --- Branche Drizzle Incrémental (Simple Champ) ---
                elif self.drizzle_active_session and self.drizzle_mode == "Incremental":
                    print("DEBUG [_worker]: Branche finalisation DRIZZLE INCREMENTAL...")
                    if self.cumulative_drizzle_data is not None and self.images_in_cumulative_stack > 0: final_result_data=self.cumulative_drizzle_data; final_result_header=self.current_stack_header
                    else: self.update_progress("ⓘ Aucun stack Drizzle Incr.")

                # --- Branche Classique ---
                elif not self.drizzle_active_session and self.current_stack_data is not None:
                     print("DEBUG [_worker]: Branche finalisation CLASSIQUE...")
                     final_result_data = self.current_stack_data; final_result_header = self.current_stack_header
                # --- Aucun Stack ---
                else: print("DEBUG [_worker]: Aucun stack à finaliser.")

                # --- Sauvegarde Finale (SI un résultat existe) ---
                if final_result_data is not None:
                    print("DEBUG [_worker]: Appel sauvegarde finale...")
                    suffix = "_mosaic" if self.is_mosaic_run else ("_drizzle_" + self.drizzle_mode.lower() if self.drizzle_active_session else "_classic")
                    self.current_stack_data = final_result_data; self.current_stack_header = final_result_header
                    self._save_final_stack(output_filename_suffix=suffix)
                else: self.final_stacked_path = None

        # --- Gestion Erreurs Globales ---
        except Exception as e:
             error_msg=f"Erreur critique worker: {type(e).__name__}: {e}"; print(f"ERREUR CRITIQUE: {error_msg}"); self.update_progress(f"❌ {error_msg}"); traceback.print_exc(limit=5); self.processing_error = error_msg

        # ============================================================
        # --- FIN DU BLOC TRY PRINCIPAL ---
        # ============================================================

        finally: # <<<--- FINALLY : Nettoyage et Fin ---
            print("DEBUG [_worker]: Entrée bloc FINALLY...")
            if self.perform_cleanup:
                self.update_progress("🧹 Nettoyage final fichiers temporaires...")
                self.cleanup_unaligned_files(); self.cleanup_temp_reference()
                self._cleanup_drizzle_temp_files(); self._cleanup_drizzle_batch_outputs()
                self._cleanup_mosaic_panel_stacks_temp()
            else: self.update_progress(f"ⓘ Fichiers temporaires conservés.")
            print("   -> Vidage listes et GC...")
            self.current_batch_data = []; local_drizzle_final_batch_data = []; self.intermediate_drizzle_batch_files = []; local_batch_temp_files = []; all_aligned_files_with_info = []
            self.current_stack_data = None; self.cumulative_drizzle_data = None; self.cumulative_drizzle_wht = None
            gc.collect()
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
        print(f"DEBUG QM [_combine_batch_result]: Début combinaison batch (data shape: {stacked_batch_data_np.shape if stacked_batch_data_np is not None else 'None'})...") # Debug

        if stacked_batch_data_np is None or stack_info_header is None:
            self.update_progress("⚠️ Erreur interne: Données de batch invalides pour combinaison.")
            print("DEBUG QM [_combine_batch_result]: Sortie précoce (données batch invalides).") # Debug
            return

        try:
            # Récupérer les informations du batch depuis l'en-tête fourni
            batch_n = int(stack_info_header.get('NIMAGES', 1))
            batch_exposure = float(stack_info_header.get('TOTEXP', 0.0))

            # Vérifier si le nombre d'images est valide
            if batch_n <= 0:
                self.update_progress(f"⚠️ Batch combiné avec {batch_n} images, ignoré.")
                print(f"DEBUG QM [_combine_batch_result]: Sortie précoce (batch_n <= 0).") # Debug
                return

            # --- Initialisation du Stack Cumulatif (Premier Batch) ---
            if self.current_stack_data is None:
                print("DEBUG QM [_combine_batch_result]: Initialisation stack cumulatif (premier batch).") # Debug
                self.update_progress("   -> Initialisation du stack cumulatif...")
                # La première image est simplement le résultat du premier batch
                # S'assurer que c'est bien un float32
                self.current_stack_data = stacked_batch_data_np.astype(np.float32)
                self.images_in_cumulative_stack = batch_n
                self.total_exposure_seconds = batch_exposure

                # --- Créer l'en-tête initial pour le stack cumulatif ---
                self.current_stack_header = fits.Header()
                # Tenter de récupérer le premier header du lot *original* pour copier les métadonnées
                # Note: self.current_batch_data est vidé à la fin de _process_completed_batch,
                # donc il faut récupérer cette info autrement ou l'ignorer ici.
                # Pour l'instant, on copie depuis stack_info_header (moins d'infos mais ok)
                keys_to_copy_from_batch = ['NIMAGES', 'STACKMETH', 'TOTEXP', 'KAPPA', 'WGHT_USED', 'WGHT_MET']
                for key in keys_to_copy_from_batch:
                    if key in stack_info_header:
                        try: self.current_stack_header[key] = (stack_info_header[key], stack_info_header.comments[key])
                        except KeyError: self.current_stack_header[key] = stack_info_header[key]

                # Infos générales
                if 'STACKTYP' not in self.current_stack_header: self.current_stack_header['STACKTYP'] = (self.stacking_mode, 'Overall stacking method')
                if 'WGHT_ON' not in self.current_stack_header: self.current_stack_header['WGHT_ON'] = (self.use_quality_weighting, 'Quality weighting status')
                self.current_stack_header['CREATOR'] = ('SeestarStacker (Queued)', 'Processing Software')
                self.current_stack_header.add_history('Cumulative Stack Initialized')
                if self.correct_hot_pixels: self.current_stack_header.add_history('Hot pixel correction applied to input frames')
                print("DEBUG QM [_combine_batch_result]: Header cumulatif initial créé.") # Debug

            # --- Combinaison avec le Stack Cumulatif Existant ---
            else:
                print("DEBUG QM [_combine_batch_result]: Combinaison avec stack cumulatif existant...") # Debug
                self.update_progress("   -> Combinaison avec le stack cumulatif...")
                # Vérifier la compatibilité des dimensions
                if self.current_stack_data.shape != stacked_batch_data_np.shape:
                    self.update_progress(f"❌ Incompatibilité dims stack: Cumul={self.current_stack_data.shape}, Batch={stacked_batch_data_np.shape}. Combinaison échouée.")
                    print(f"ERREUR QM [_combine_batch_result]: Incompatibilité de dimensions.") # Debug
                    return # Ne pas continuer si les dimensions ne correspondent pas

                # Calcul des poids basé sur le nombre d'images
                current_n = self.images_in_cumulative_stack
                total_n = current_n + batch_n
                w_old = current_n / total_n
                w_new = batch_n / total_n
                print(f"DEBUG QM [_combine_batch_result]: Poids combinaison: w_old={w_old:.3f}, w_new={w_new:.3f}") # Debug

                # --- Tentative de combinaison via CuPy si disponible ---
                use_cupy_combine = _cupy_installed and check_cupy_cuda()
                combined_np = None # Variable pour stocker le résultat (toujours NumPy)

                if use_cupy_combine:
                    gpu_current = None; gpu_batch = None
                    try:
                        print("DEBUG QM [_combine_batch_result]: Tentative combinaison CuPy...") # Debug
                        gpu_current = cupy.asarray(self.current_stack_data, dtype=cupy.float32)
                        gpu_batch = cupy.asarray(stacked_batch_data_np, dtype=cupy.float32)
                        gpu_combined = (gpu_current * w_old) + (gpu_batch * w_new)
                        combined_np = cupy.asnumpy(gpu_combined)
                        print("DEBUG QM [_combine_batch_result]: Combinaison CuPy réussie.") # Debug
                    except cupy.cuda.memory.OutOfMemoryError:
                        print("Warning: GPU Out of Memory during stack combination. Falling back to CPU.") # Garder Warning
                        use_cupy_combine = False; gc.collect(); cupy.get_default_memory_pool().free_all_blocks()
                    except Exception as gpu_err:
                        print(f"Warning: CuPy error during stack combination: {gpu_err}. Falling back to CPU.") # Garder Warning
                        traceback.print_exc(limit=1); use_cupy_combine = False; gc.collect()
                        try: cupy.get_default_memory_pool().free_all_blocks()
                        except Exception: pass
                    finally:
                        del gpu_current, gpu_batch
                        if '_cupy_installed' in globals() and _cupy_installed:
                             try: cupy.get_default_memory_pool().free_all_blocks()
                             except Exception: pass

                # --- Combinaison via NumPy (Fallback ou si CuPy non utilisé) ---
                if not use_cupy_combine:
                    print("DEBUG QM [_combine_batch_result]: Combinaison NumPy (CPU)...") # Debug
                    current_data_float = self.current_stack_data.astype(np.float32)
                    batch_data_float = stacked_batch_data_np.astype(np.float32)
                    combined_np = (current_data_float * w_old) + (batch_data_float * w_new)
                    print("DEBUG QM [_combine_batch_result]: Combinaison NumPy réussie.") # Debug

                # --- Mettre à jour le stack cumulatif ---
                if combined_np is None:
                     print("ERREUR QM [_combine_batch_result]: Échec des méthodes CPU et GPU pour combiner.") # Debug
                     raise RuntimeError("La combinaison n'a produit aucun résultat (erreur CuPy et NumPy?).")

                self.current_stack_data = combined_np.astype(np.float32)
                print("DEBUG QM [_combine_batch_result]: Stack cumulatif mis à jour.") # Debug

                # --- Mettre à jour les statistiques et l'en-tête cumulatif ---
                self.images_in_cumulative_stack = total_n
                self.total_exposure_seconds += batch_exposure
                if self.current_stack_header:
                    self.current_stack_header['NIMAGES'] = self.images_in_cumulative_stack
                    self.current_stack_header['TOTEXP'] = (round(self.total_exposure_seconds, 2), '[s] Total exposure time')
                    # self.current_stack_header.add_history(...) # Optionnel

            ### MODIFICATION : Appel à ChromaticBalancer supprimé d'ici ###
            # if self.apply_chroma_correction and self.current_stack_data is not None:
            #    if self.current_stack_data.ndim == 3 and self.current_stack_data.shape[2] == 3:
            #        self.update_progress("   -> Application de la correction chromatique...")
            #        # S'assurer que chroma_balancer existe
            #        if hasattr(self, 'chroma_balancer') and self.chroma_balancer:
            #             self.current_stack_data = self.chroma_balancer.normalize_stack(self.current_stack_data)
            #             self.update_progress("   -> Correction chromatique terminée.")
            #        else:
            #             self.update_progress("   -> AVERTISSEMENT: Instance ChromaticBalancer non trouvée.")
            ### FIN MODIFICATION ###

            # --- Clip final du résultat cumulé ---
            self.current_stack_data = np.clip(self.current_stack_data, 0.0, 1.0)
            print("DEBUG QM [_combine_batch_result]: Clipping final appliqué.") # Debug

        except Exception as e:
            print(f"ERREUR QM [_combine_batch_result]: Exception inattendue - {e}") # Debug
            self.update_progress(f"❌ Erreur pendant la combinaison du résultat du batch: {e}")
            traceback.print_exc(limit=3)

        print("DEBUG QM [_combine_batch_result]: Fin méthode.") # Debug


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




    def _save_final_stack(self, output_filename_suffix="", stopped_early=False):
        """
        Sauvegarde le stack final (classique, Drizzle, ou mosaïque) et sa prévisualisation.
        Applique la neutralisation du fond, la correction chromatique, ET SCNR final avant sauvegarde.
        """
        print(f"DEBUG QM [_save_final_stack]: Début sauvegarde finale (suffix: '{output_filename_suffix}', stopped_early: {stopped_early})")

        # --- Imports tardifs ---
        neutralize_background_func = None
        try:
            from ..tools.stretch import neutralize_background_automatic as neutralize_background_func
            print("DEBUG QM [_save_final_stack]: Import tardif de neutralize_background_automatic réussi.")
        except ImportError:
            print("ERREUR QM [_save_final_stack]: Échec import tardif neutralize_background_automatic. Neutralisation désactivée.")
            self.update_progress("⚠️ Erreur interne: Fonction de neutralisation du fond non trouvée. Étape ignorée.")

        ### NOUVEAU : Import tardif pour SCNR ###
        apply_scnr_func = None
        try:
            from ..enhancement.color_correction import apply_scnr as apply_scnr_func
            print("DEBUG QM [_save_final_stack]: Import tardif de apply_scnr réussi.")
        except ImportError:
            print("ERREUR QM [_save_final_stack]: Échec import tardif apply_scnr. SCNR final désactivé.")
            self.update_progress("⚠️ Erreur interne: Fonction SCNR non trouvée. Étape ignorée.")
        ### FIN NOUVEAU ###

        # --- 1. Choisir les Données et le Header de Base ---
        # ... (cette partie reste identique à la version précédente) ...
        data_to_save = None
        header_base = None
        image_count = 0
        stack_type_for_filename = "unknown"
        is_drizzle_mosaic_save = False

        if self.current_stack_header and ('DRZSCALE' in self.current_stack_header or \
                                          ('STACKTYP' in self.current_stack_header and \
                                           ('Drizzle' in self.current_stack_header['STACKTYP'] or \
                                            'Mosaic' in self.current_stack_header.get('STACKTYP', '')))):
            is_drizzle_mosaic_save = True
            if 'Mosaic' in self.current_stack_header.get('STACKTYP', ''): stack_type_for_filename = "mosaic_drizzle"
            elif self.drizzle_mode == "Incremental": stack_type_for_filename = f"drizzle_incr_{self.drizzle_scale:.0f}x"
            else: stack_type_for_filename = f"drizzle_final_{self.drizzle_scale:.0f}x"
            data_to_save = self.current_stack_data
            header_base = self.current_stack_header
            image_count = self.images_in_cumulative_stack
            print(f"DEBUG QM [_save_final_stack]: Mode Drizzle/Mosaic. Stack type: {stack_type_for_filename}, Img count: {image_count}")

        elif self.current_stack_data is not None:
            is_drizzle_mosaic_save = False
            stack_type_for_filename = self.stacking_mode
            data_to_save = self.current_stack_data
            header_base = self.current_stack_header
            image_count = self.images_in_cumulative_stack
            print(f"DEBUG QM [_save_final_stack]: Mode Classique. Stack type: {stack_type_for_filename}, Img count: {image_count}")
        
        # --- 2. Vérifications Initiales ---
        if data_to_save is None or self.output_folder is None:
            self.final_stacked_path = None; print("DEBUG QM [_save_final_stack]: Sortie précoce (data_to_save ou output_folder est None).")
            self.update_progress("ⓘ Aucun stack final à sauvegarder (données manquantes ou dossier sortie invalide)."); return
        if image_count <= 0 and not stopped_early:
             self.final_stacked_path = None; print(f"DEBUG QM [_save_final_stack]: Sortie précoce (image_count={image_count} <= 0 et pas stopped_early).")
             self.update_progress("ⓘ Aucun stack final à sauvegarder (0 images combinées)."); return
        
        print(f"DEBUG QM [_save_final_stack]: Données à sauvegarder (avant post-traitement) - Shape: {data_to_save.shape}, Type: {data_to_save.dtype}, Min: {np.nanmin(data_to_save):.3f}, Max: {np.nanmax(data_to_save):.3f}")

        # --- Application des post-traitements couleur ---
        if data_to_save.ndim == 3 and data_to_save.shape[2] == 3:
            data_to_save = data_to_save.astype(np.float32)

            # --- 2a. Neutralisation du Fond de Ciel Automatique ---
            if neutralize_background_func:
                self.update_progress("Appel de la fonction Neutralisation du fond...", None)
                print("DEBUG QM [_save_final_stack]: Appel de neutralize_background_automatic...")
                try:
                    data_before_bn = data_to_save.copy()
                    data_to_save = neutralize_background_func(data_to_save) # Utilise les params par défaut de la fonction pour l'instant
                    if data_to_save is None: data_to_save = data_before_bn; self.update_progress("⚠️ Échec neutralisation (retour None).", None)
                    elif np.allclose(data_before_bn, data_to_save): self.update_progress("ⓘ Neutralisation du fond n'a pas modifié l'image.", None)
                    else: self.update_progress("   -> Neutralisation du fond terminée.", None)
                    print(f"DEBUG QM [_save_final_stack]: Données après BN - Min: {np.nanmin(data_to_save):.3f}, Max: {np.nanmax(data_to_save):.3f}")
                except Exception as bn_err: print(f"ERREUR QM [_save_final_stack]: Erreur neutralize_background_automatic: {bn_err}"); self.update_progress(f"⚠️ Erreur neutralisation: {bn_err}.")
            

            # --- 2b. Correction Chromatique / Bord (`ChromaticBalancer`) ---
            if self.apply_chroma_correction: # Contrôlé par la checkbox "Edge Enhance" (via settings)
                self.update_progress("Application de la Correction Chromatique/Bord...", None)
                print("DEBUG QM [_save_final_stack]: Appel de self.chroma_balancer.normalize_stack...")
                try:
                    if hasattr(self, 'chroma_balancer') and self.chroma_balancer:
                         data_before_cb = data_to_save.copy()
                         data_to_save = self.chroma_balancer.normalize_stack(data_to_save)
                         if data_to_save is None: data_to_save = data_before_cb; self.update_progress("⚠️ Échec correction chroma (retour None).", None)
                         elif np.allclose(data_before_cb, data_to_save): self.update_progress("ⓘ Correction chromatique/bord n'a pas modifié l'image.", None)
                         else: self.update_progress("   -> Correction chromatique/bord terminée.", None)
                         print(f"DEBUG QM [_save_final_stack]: Données après ChromaBalance - Min: {np.nanmin(data_to_save):.3f}, Max: {np.nanmax(data_to_save):.3f}")
                    else: self.update_progress("   -> AVERTISSEMENT: Instance ChromaticBalancer non trouvée.")
                except Exception as chroma_final_err: print(f"ERREUR QM [_save_final_stack]: Erreur ChromaticBalancer: {chroma_final_err}"); self.update_progress(f"⚠️ Erreur correction chromatique: {chroma_final_err}.")
                        # 2b. Correction Chromatique / Bord
            if self.apply_chroma_correction:
                # ... (logique appel self.chroma_balancer.normalize_stack identique) ...
                self.update_progress("Application de la Correction Chromatique/Bord...", None); print("DEBUG QM [_save_final_stack]: Appel de self.chroma_balancer.normalize_stack...")
                try:
                    if hasattr(self, 'chroma_balancer') and self.chroma_balancer:
                         data_before_cb = data_to_save.copy(); data_to_save = self.chroma_balancer.normalize_stack(data_to_save)
                         if data_to_save is None: data_to_save = data_before_cb; self.update_progress("⚠️ Échec correction chroma (retour None).", None)
                         elif np.allclose(data_before_cb, data_to_save): self.update_progress("ⓘ Correction chromatique/bord n'a pas modifié l'image.", None)
                         else: self.update_progress("   -> Correction chromatique/bord terminée.", None)
                         print(f"DEBUG QM [_save_final_stack]: Données après ChromaBalance - Min: {np.nanmin(data_to_save):.3f}, Max: {np.nanmax(data_to_save):.3f}")
                    else: self.update_progress("   -> AVERTISSEMENT: Instance ChromaticBalancer non trouvée.")
                except Exception as chroma_final_err: print(f"ERREUR QM [_save_final_stack]: Erreur ChromaticBalancer: {chroma_final_err}"); self.update_progress(f"⚠️ Erreur correction chromatique: {chroma_final_err}.")

            ### MODIFIÉ : Utilisation des paramètres SCNR de self ###
            # Remplacer apply_final_scnr_hardcoded_for_test par self.apply_final_scnr
            # Remplacer final_scnr_amount_hardcoded_for_test par self.final_scnr_amount
            if self.apply_final_scnr and apply_scnr_func and data_to_save is not None:
                self.update_progress(f"Application SCNR final ({self.final_scnr_target_channel}, Amount: {self.final_scnr_amount:.2f})...", None)
                print(f"DEBUG QM [_save_final_stack]: Appel de apply_scnr (final) avec Amount={self.final_scnr_amount}, PreserveLum={self.final_scnr_preserve_luminosity}...")
                try:
                    data_before_scnr = data_to_save.copy()
                    data_to_save = apply_scnr_func(
                        data_to_save,
                        target_channel=self.final_scnr_target_channel, # Utilise l'attribut de self
                        amount=self.final_scnr_amount,                 # Utilise l'attribut de self
                        preserve_luminosity=self.final_scnr_preserve_luminosity # Utilise l'attribut de self
                    )
                    if data_to_save is None: data_to_save = data_before_scnr; self.update_progress("⚠️ Échec SCNR final (retour None).", None)
                    elif np.allclose(data_before_scnr, data_to_save): self.update_progress("ⓘ SCNR final n'a pas modifié l'image.", None)
                    else: self.update_progress("   -> SCNR final terminé.", None)
                    print(f"DEBUG QM [_save_final_stack]: Données après SCNR Final - Min: {np.nanmin(data_to_save):.3f}, Max: {np.nanmax(data_to_save):.3f}")
                except Exception as scnr_final_err:
                    print(f"ERREUR QM [_save_final_stack]: Erreur pendant SCNR final: {scnr_final_err}")
                    self.update_progress(f"⚠️ Erreur SCNR final: {scnr_final_err}. Étape ignorée.")
            elif self.apply_final_scnr and not apply_scnr_func:
                print("DEBUG QM [_save_final_stack]: SCNR final demandé mais fonction non importée.")
            

            ### NOUVEAU : 2c. Application SCNR Final Optionnel ###
            # Pour l'instant, on active SCNR par défaut pour ce test avec un amount fixe.
            # Plus tard, self.apply_final_scnr et self.final_scnr_amount viendront des settings/UI.
            apply_final_scnr_hardcoded_for_test = True # <<< METTEZ True POUR TESTER SCNR
            final_scnr_amount_hardcoded_for_test = 0.8 # <<< Amount (0.0 à 1.0)

            if apply_final_scnr_hardcoded_for_test and apply_scnr_func and data_to_save is not None:
                self.update_progress("Application SCNR final (Vert)...", None)
                print("DEBUG QM [_save_final_stack]: Appel de apply_scnr (final)...")
                try:
                    data_before_scnr = data_to_save.copy()
                    data_to_save = apply_scnr_func(data_to_save, target_channel='green', amount=final_scnr_amount_hardcoded_for_test)
                    if data_to_save is None: data_to_save = data_before_scnr; self.update_progress("⚠️ Échec SCNR final (retour None).", None)
                    elif np.allclose(data_before_scnr, data_to_save): self.update_progress("ⓘ SCNR final n'a pas modifié l'image.", None)
                    else: self.update_progress("   -> SCNR final terminé.", None)
                    print(f"DEBUG QM [_save_final_stack]: Données après SCNR Final - Min: {np.nanmin(data_to_save):.3f}, Max: {np.nanmax(data_to_save):.3f}")
                except Exception as scnr_final_err:
                    print(f"ERREUR QM [_save_final_stack]: Erreur pendant SCNR final: {scnr_final_err}")
                    self.update_progress(f"⚠️ Erreur SCNR final: {scnr_final_err}. Étape ignorée.")
            elif apply_final_scnr_hardcoded_for_test and not apply_scnr_func:
                print("DEBUG QM [_save_final_stack]: SCNR final demandé mais fonction non importée.")
            ### FIN NOUVEAU ###

        # --- Rognage (si configuré) ---
        # ... (votre code de rognage, s'il est ici, reste le même) ...
        # Exemple :
        # crop_percent_val = getattr(self, 'edge_crop_percent_from_settings', 0.00)
        # if data_to_save is not None and isinstance(crop_percent_val, (float, int)) and crop_percent_val > 0.0:
        #     # ... (logique de rognage) ...


        # --- 3. Construire le Nom de Fichier ---
        # ... (cette partie reste identique) ...
        base_name = "stack_final"; weight_suffix = "_wght" if self.use_quality_weighting and not is_drizzle_mosaic_save else ""
        current_op_suffix = str(output_filename_suffix) if output_filename_suffix else ""; final_suffix = f"{weight_suffix}{current_op_suffix}"
        self.final_stacked_path = os.path.join(self.output_folder, f"{base_name}_{stack_type_for_filename}{final_suffix}.fit")
        preview_path = os.path.splitext(self.final_stacked_path)[0] + ".png"
        print(f"DEBUG QM [_save_final_stack]: Chemin FITS final: {self.final_stacked_path}")
        print(f"DEBUG QM [_save_final_stack]: Chemin PNG preview: {preview_path}")

        # --- 4. Sauvegarde Fichier FITS et PNG ---
        try:
            final_header = header_base.copy() if header_base else fits.Header()
            # --- Mise à jour header (identique) ---
            final_header['NIMAGES'] = (image_count, 'Images combined in final stack') # ... etc.
            final_header['TOTEXP'] = (round(self.total_exposure_seconds, 2), '[s] Approx total exposure time')
            final_header['ALIGNED'] = (self.aligned_files_count, 'Successfully aligned images')
            final_header['FAILALIGN'] = (self.failed_align_count, 'Failed alignments')
            final_header['FAILSTACK'] = (self.failed_stack_count, 'Files skipped due to stack/combine errors')
            final_header['SKIPPED'] = (self.skipped_files_count, 'Other skipped/error files')
            if not is_drizzle_mosaic_save:
                final_header['STACKTYP'] = (self.stacking_mode, 'Stacking method')
                if self.stacking_mode in ["kappa-sigma", "winsorized-sigma"]: final_header['KAPPA'] = (self.kappa, 'Kappa value for clipping')
                for k in ['DRZSCALE', 'DRZKERNEL', 'DRZPIXFR', 'DRZMODE']:
                    if k in final_header: del final_header[k]
            else:
                if 'STACKTYP' not in final_header: final_header['STACKTYP'] = (stack_type_for_filename, 'Stacking/Processing method')
                if 'DRZSCALE' not in final_header: final_header['DRZSCALE'] = (self.drizzle_scale, 'Drizzle Scale Factor')
                if 'DRZKERNEL' not in final_header: final_header['DRZKERNEL'] = (self.drizzle_kernel, 'Drizzle Kernel')
                if 'DRZPIXFR' not in final_header: final_header['DRZPIXFR'] = (self.drizzle_pixfrac, 'Drizzle Pixfrac')
                if 'DRZMODE' not in final_header and self.drizzle_mode : final_header['DRZMODE'] = (self.drizzle_mode, 'Drizzle Mode (Final/Incremental)')
            if 'WGHT_ON' not in final_header: final_header['WGHT_ON'] = (self.use_quality_weighting, 'Quality weighting status')
            if self.use_quality_weighting and 'WGHT_MET' not in final_header:
                 w_metrics = [];
                 if self.weight_by_snr: w_metrics.append(f"SNR^{self.snr_exponent:.1f}")
                 if self.weight_by_stars: w_metrics.append(f"Stars^{self.stars_exponent:.1f}")
                 final_header['WGHT_MET'] = (",".join(w_metrics), 'Metrics used for weighting')
            # --- FIN MODIFIÉ : Ajout info SCNR au header si appliqué ---
            if apply_final_scnr_hardcoded_for_test: # Si SCNR a été tenté
                final_header['SCNR_APP'] = (True, 'SCNR (Green) applied to final stack')
                final_header['SCNR_AMT'] = (final_scnr_amount_hardcoded_for_test, 'SCNR amount factor')
            # --- FIN MODIFIÉ ---
            try: # Nettoyage historique
                if 'HISTORY' in final_header:
                    history_entries = list(final_header['HISTORY']);
                    filtered_history = [h for h in history_entries if not isinstance(h, str) or ('Intermediate save' not in h and 'Cumulative Stack Initialized' not in h and 'Batch' not in h)]
                    while 'HISTORY' in final_header: del final_header['HISTORY']
                    for entry in filtered_history: final_header.add_history(entry)
            except Exception: pass
            history_msg = f'Final Stack Saved by SeestarStacker (Mode: {stack_type_for_filename})'
            if stopped_early: history_msg += ' - Stopped Early'
            final_header.add_history(history_msg)
            # --- Fin Préparation Header ---

            print(f"DEBUG QM [_save_final_stack]: Sauvegarde FITS vers {self.final_stacked_path}...")
            save_fits_image(data_to_save, self.final_stacked_path, final_header, overwrite=True)
            print("DEBUG QM [_save_final_stack]: Sauvegarde FITS terminée.")

            print(f"DEBUG QM [_save_final_stack]: Sauvegarde Preview PNG vers {preview_path}...")
            # Pour le PNG, on veut toujours le meilleur étirement possible, indépendamment de SCNR sur FITS
            save_preview_image(data_to_save, preview_path, apply_stretch=True, enhanced_stretch=True)
            print("DEBUG QM [_save_final_stack]: Sauvegarde Preview PNG terminée.")

            self.update_progress(f"✅ Stack final sauvegardé ({image_count} images)")

        except Exception as e:
            print(f"ERREUR QM [_save_final_stack]: Échec sauvegarde FITS/PNG: {e}")
            self.update_progress(f"⚠️ Erreur sauvegarde stack final: {e}")
            traceback.print_exc(limit=2)
            self.final_stacked_path = None
            print("DEBUG QM [_save_final_stack]: final_stacked_path mis à None en raison d'erreur sauvegarde.")
        
        print("DEBUG QM [_save_final_stack]: Fin méthode.")




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
                         # --- Arguments Stacking Classique ---
                         stacking_mode="kappa-sigma", # Valeur par défaut si non fournie
                         kappa=2.5,                 # Valeur par défaut si non fournie
                         # --- Arguments Communs ---
                         batch_size=10,             # Utiliser une valeur > 0 par défaut ici
                         correct_hot_pixels=True,
                         hot_pixel_threshold=3.0,
                         neighborhood_size=5,
                         bayer_pattern="GRBG",
                         perform_cleanup=True,
                         # --- Arguments Pondération ---
                         use_weighting=False, weight_snr=True, weight_stars=True,
                         snr_exp=1.0, stars_exp=0.5, min_w=0.1,
                         # --- Arguments Drizzle ---
                         use_drizzle=False, drizzle_scale=2.0, drizzle_wht_threshold=0.7,
                         drizzle_mode="Final", drizzle_kernel="square", drizzle_pixfrac=1.0,
                         # --- Argument Correction Chroma ---
                         apply_chroma_correction=True,
                         ### NOUVEAU : Arguments SCNR Final ###
                         apply_final_scnr=False,
                         final_scnr_target_channel='green', # Garder 'green' par défaut pour l'instant
                         final_scnr_amount=0.8,
                         final_scnr_preserve_luminosity=True,
                         ### FIN NOUVEAU ###
                         # --- Arguments Mosaïque ---
                         is_mosaic_run=False,
                         api_key=None,
                         mosaic_settings=None, *args, **kwargs):
        """
        Démarre le thread de traitement principal avec la configuration spécifiée.
        MAJ: Signature complète, ordre d'initialisation corrigé, accepte tous les args.
        """
        print("DEBUG (Backend start_processing): Début tentative démarrage...")
        print(f"   -> Args reçus: is_mosaic_run={is_mosaic_run}, use_drizzle={use_drizzle}, drizzle_mode={drizzle_mode}, stacking_mode={stacking_mode}, api_key={'Oui' if api_key else 'Non'}, mosaic_settings={mosaic_settings}") # Log initial

        if self.processing_active:
            self.update_progress("⚠️ Tentative de démarrer un traitement déjà en cours.")
            return False

        # 1. Réinitialiser l'état et préparer les dossiers/variables de base
        print("DEBUG (Backend start_processing): Appel à self.initialize()...")
        self.stop_processing = False
        self.current_folder = os.path.abspath(input_dir)
        # L'appel à initialize() réinitialise de nombreux attributs !
        if not self.initialize(output_dir):
            self.processing_active = False
            print("ERREUR (Backend start_processing): Échec de self.initialize().")
            return False
        print("DEBUG (Backend start_processing): self.initialize() terminé.")

        # --- 2. Définir les paramètres spécifiques à CETTE session *APRES* initialize ---
        print("DEBUG (Backend start_processing): Configuration des paramètres de session...")
        # -- Modes --
        self.is_mosaic_run = is_mosaic_run
        # Forcer Drizzle si Mosaïque est demandé
        self.drizzle_active_session = use_drizzle or self.is_mosaic_run

        # -- Paramètres Communs --
        self.api_key = api_key # Stocker clé API reçue
        print(f"!!!! DEBUG QM Start: self.api_key JUSTE APRES ASSIGNATION = '{self.api_key}' !!!!")
        self.apply_chroma_correction = apply_chroma_correction
        self.correct_hot_pixels = correct_hot_pixels
        self.hot_pixel_threshold = hot_pixel_threshold
        self.neighborhood_size = neighborhood_size
        self.bayer_pattern = bayer_pattern
        self.perform_cleanup = perform_cleanup

        # -- Paramètres Stacking Classique --
        self.stacking_mode = stacking_mode
        self.kappa = float(kappa)

        # -- Paramètres Pondération --
        self.use_quality_weighting = use_weighting
        self.weight_by_snr = weight_snr
        self.weight_by_stars = weight_stars
        self.snr_exponent = snr_exp
        self.stars_exponent = stars_exp
        self.min_weight = max(0.01, min(1.0, min_w))

        # -- Paramètres Drizzle (utilisés si drizzle_active_session est True) --
        if self.drizzle_active_session:
            # Utiliser mosaic_settings pour kernel/pixfrac si en mode mosaïque
            if self.is_mosaic_run:
                print(f"DEBUG (Backend start_processing): Mode Mosaïque actif. Settings reçus: {mosaic_settings}")
                current_mosaic_settings = mosaic_settings if isinstance(mosaic_settings, dict) else {}
                # Utiliser le kernel global (reçu en arg) si non trouvé dans mosaic_settings
                self.drizzle_kernel = current_mosaic_settings.get('kernel', drizzle_kernel)
                # Utiliser le pixfrac global (reçu en arg) si non trouvé dans mosaic_settings
                self.drizzle_pixfrac = current_mosaic_settings.get('pixfrac', drizzle_pixfrac)
                # Valider/clipper pixfrac
                try: self.drizzle_pixfrac = float(np.clip(float(self.drizzle_pixfrac), 0.01, 1.0))
                except (ValueError, TypeError): self.drizzle_pixfrac = 1.0; print(f"WARNING: pixfrac mosaïque invalide ({self.drizzle_pixfrac}), reset à 1.0")
                print(f"   -> Params Mosaïque utilisés -> Kernel: '{self.drizzle_kernel}', Pixfrac: {self.drizzle_pixfrac:.2f}")
            else: # Drizzle simple champ
                 self.drizzle_kernel = drizzle_kernel
                 self.drizzle_pixfrac = drizzle_pixfrac
                 print(f"DEBUG (Backend start_processing): Mode Drizzle simple champ.")
            # Paramètres Drizzle communs (Mode, Scale, WHT)
            self.drizzle_mode = drizzle_mode if drizzle_mode in ["Final", "Incremental"] else "Final"
            self.drizzle_scale = float(drizzle_scale)
            self.drizzle_wht_threshold = max(0.01, min(1.0, float(drizzle_wht_threshold)))
            print(f"   -> Params Drizzle Communs -> Mode: {self.drizzle_mode}, Scale: {self.drizzle_scale:.1f}, WHT: {self.drizzle_wht_threshold:.2f}, Kernel: {self.drizzle_kernel}, Pixfrac: {self.drizzle_pixfrac:.2f}")
        # --- Fin définition paramètres session ---

        ### NOUVEAU : Stockage des paramètres SCNR Final dans self ###
        self.apply_final_scnr = apply_final_scnr
        self.final_scnr_target_channel = final_scnr_target_channel
        self.final_scnr_amount = final_scnr_amount
        self.final_scnr_preserve_luminosity = final_scnr_preserve_luminosity
        print(f"DEBUG (Backend start_processing): self.apply_final_scnr = {self.apply_final_scnr}")
        print(f"DEBUG (Backend start_processing): self.final_scnr_amount = {self.final_scnr_amount}")
        ### FIN NOUVEAU ###


        # --- 3. Logs et Vérification Batch Size ---
        # Log du mode final choisi
        if self.is_mosaic_run: self.update_progress("🖼️ Mode Mosaïque ACTIVÉ pour cette session.")
        elif self.drizzle_active_session: self.update_progress(f"💧 Mode Drizzle (Simple Champ) Activé ({self.drizzle_mode})...")
        else: self.update_progress("⚙️ Mode Stack Classique Activé...")

        # Gestion Batch Size (utilise l'argument batch_size reçu)
        requested_batch_size = batch_size # Utilise l'argument reçu
        if requested_batch_size <= 0: # Si 0 ou moins -> Estimation auto
             self.update_progress("🧠 Estimation taille lot auto (reçu <= 0)...", None)
             sample_img_path = None
             if input_dir and os.path.isdir(input_dir): fits_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.fit', '.fits'))]; sample_img_path = os.path.join(input_dir, fits_files[0]) if fits_files else None
             try: estimated_size = estimate_batch_size(sample_image_path=sample_img_path); self.batch_size = estimated_size; self.update_progress(f"✅ Taille lot auto estimée: {estimated_size}", None)
             except Exception as est_err: self.update_progress(f"⚠️ Erreur estimation taille lot: {est_err}. Utilisation défaut (10).", None); self.batch_size = 10
        else: # Taille fournie > 0
             self.batch_size = requested_batch_size

        # Valider la taille minimale
        if self.batch_size < 3:
            self.update_progress(f"⚠️ Taille de lot ({self.batch_size}) trop petite, ajustée à 3.", None)
            self.batch_size = 3
        self.update_progress(f"ⓘ Taille de lot effective pour le traitement : {self.batch_size}")

        # Log pondération si active
        if self.use_quality_weighting:
            self.update_progress(f"⚖️ Pondération Qualité Activée (SNR^{self.snr_exponent:.1f}, Stars^{self.stars_exponent:.1f}, MinW: {self.min_weight:.2f})")

        # --- 4. Gérer dossiers initiaux ---
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


        # --- 5. Ajouter fichiers initiaux ---
        initial_files_added = self._add_files_to_queue(self.current_folder)
        if initial_files_added > 0:
            self._recalculate_total_batches() # Recalculer après ajout initial
            self.update_progress(f"📋 {initial_files_added} fichiers initiaux ajoutés. Total lots estimé: {self.total_batches_estimated if self.total_batches_estimated > 0 else '?'}")
        elif not self.additional_folders: # Si pas d'initiaux ET pas d'additionnels
             self.update_progress("⚠️ Aucun fichier initial trouvé ou dossier supplémentaire en attente.")
             # On pourrait retourner False ici si rien à traiter ? À discuter.

        # --- 6. Configurer référence pour l'aligneur ---
        self.aligner.reference_image_path = reference_path_ui or None

        # --- 7. Démarrer worker ---
        print("DEBUG (Backend start_processing): Démarrage du thread worker...")
        self.processing_thread = threading.Thread(target=self._worker, name="StackerWorker")
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.processing_active = True # Mettre à True *après* avoir lancé le thread
        self.update_progress("🚀 Thread de traitement démarré.")
        print("DEBUG (Backend start_processing): Fin.")
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
        return getattr(self, 'processing_active', False) and \
            getattr(self, 'processing_thread', None) is not None and \
            getattr(self, 'processing_thread', None) is not None and \
            self.processing_thread.is_alive()



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

