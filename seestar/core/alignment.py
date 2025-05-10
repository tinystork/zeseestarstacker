# --- START OF FILE seestar/core/alignment.py ---
"""
Module pour l'alignement des images astronomiques.
Utilise astroalign pour l'enregistrement des images.
"""
import os
import numpy as np
from astropy.io import fits
import cv2
import astroalign as aa
from tqdm import tqdm
import warnings
import gc
import time
import shutil
import concurrent.futures
from functools import partial
import traceback # Added for traceback printing


from .image_processing import (
    load_and_validate_fits, # Returns float32 0-1 or None
    debayer_image,          # Expects float32 0-1, returns float32 0-1
    save_fits_image,        # Expects float32 0-1, saves uint16
    save_preview_image      # For saving reference preview
)
from .hot_pixels import detect_and_correct_hot_pixels
from .utils import estimate_batch_size

warnings.filterwarnings("ignore", category=FutureWarning)

class SeestarAligner:
    """
    Classe pour l'alignement des images astronomiques de Seestar.
    Trouve une image de référence et aligne les autres images sur celle-ci.
    """
    NUM_IMAGES_FOR_AUTO_REF = 50 # Number of initial images to check for reference

    def __init__(self):
        """Initialise l'aligneur avec des valeurs par défaut."""
        self.bayer_pattern = "GRBG"
        self.batch_size = 0
        self.reference_image_path = None
        self.correct_hot_pixels = True
        self.hot_pixel_threshold = 3.0
        self.neighborhood_size = 5
        self.stop_processing = False
        self.progress_callback = None

    def set_progress_callback(self, callback):
        """Définit la fonction de rappel pour les mises à jour de progression."""
        self.progress_callback = callback

    def update_progress(self, message, progress=None):
        """Met à jour la progression en utilisant le callback si disponible."""
        # Ensure message is a string
        message = str(message)
        if self.progress_callback:
            try:
                self.progress_callback(message, progress)
            except Exception as cb_err:
                print(f"Error in aligner progress callback: {cb_err}")
        else:
            # Basic print fallback if no callback set
            if progress is not None:
                print(f"[{int(progress)}%] {message}")
            else:
                print(message)

    def align_images(self, input_folder, output_folder=None, specific_files=None):
        """
        (Primarily used by QueueManager to get reference image info now)
        Finds reference image. Standalone alignment loop is commented out.
        """
        self.stop_processing = False
        if output_folder is None: output_folder = os.path.join(input_folder, "aligned_lights")
        try:
            os.makedirs(output_folder, exist_ok=True)
            unaligned_folder = os.path.join(output_folder, "unaligned") # Still create for consistency
            os.makedirs(unaligned_folder, exist_ok=True)
        except OSError as e:
            self.update_progress(f"❌ Erreur création dossier sortie/unaligned: {e}")
            return None # Critical error
        try:
            if specific_files: files = [f for f in specific_files if f.lower().endswith(('.fit', '.fits'))]
            else: files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.fit', '.fits'))]
            files.sort()
        except FileNotFoundError: self.update_progress(f"❌ Dossier d'entrée non trouvé: {input_folder}"); return None
        except Exception as e: self.update_progress(f"❌ Erreur lecture dossier entrée: {e}"); return None
        if not files: self.update_progress("❌ Aucun fichier .fit/.fits trouvé à traiter."); return output_folder

        # Estimate batch size (still useful for other parts or general info)
        if self.batch_size <= 0:
            if files: # Check if files list is not empty
                 sample_path = os.path.join(input_folder, files[0])
                 try:
                     self.batch_size = estimate_batch_size(sample_path)
                     self.update_progress(f"🧠 Taille de lot dynamique estimée : {self.batch_size}")
                 except Exception as est_err:
                     self.update_progress(f"⚠️ Erreur estimation taille lot: {est_err}. Utilisation valeur défaut 10.")
                     self.batch_size = 10
            else: # No files, use default batch size
                 self.batch_size = 10

        self.update_progress("⭐ Recherche/Préparation image de référence...")
        fixed_reference_image, fixed_reference_header = self._get_reference_image(input_folder, files)

        if fixed_reference_image is None:
            # Error message now generated within _get_reference_image or QueueManager
            # self.update_progress("❌ Impossible d'obtenir une image de référence valide. Arrêt.")
            return None # Signal failure

        self.update_progress(f"✅ Recherche référence terminée.")
        return output_folder




# --- DANS LA CLASSE SeestarAligner DANS seestar/core/alignment.py ---
# (Assurez-vous que les imports nécessaires comme os, np, fits, tqdm, gc,
#  et vos fonctions load_and_validate_fits, debayer_image, detect_and_correct_hot_pixels
#  sont bien présents en haut du fichier alignment.py)

    def _get_reference_image(self, input_folder, files_to_scan): # Renommé 'files' en 'files_to_scan' pour clarté
        """
        Obtient l'image de référence (float32, 0-1) et son en-tête.
        Tente de charger une référence manuelle si spécifiée, sinon sélectionne
        automatiquement la meilleure parmi un sous-ensemble des 'files_to_scan'.
        """
        print(f"DEBUG ALIGNER [_get_reference_image]: Début. Dossier: '{os.path.basename(input_folder)}', Nb fichiers fournis: {len(files_to_scan)}")
        
        reference_image_data = None # Contiendra les données image (np.array)
        reference_header = None     # Contiendra l'objet astropy.io.fits.Header

        processed_candidates_auto = 0 # Pour la sélection auto
        rejected_candidates_auto = 0  # Pour la sélection auto
        rejection_reasons_auto = {'load': 0, 'variance': 0, 'preprocess': 0, 'metric': 0} # Pour la sélection auto

        # --- Étape 1: Essayer de Charger une Référence Manuelle si Spécifiée ---
        if self.reference_image_path and os.path.isfile(self.reference_image_path):
            manual_ref_basename = os.path.basename(self.reference_image_path)
            self.update_progress(f"📌 Chargement référence manuelle: {manual_ref_basename}")
            print(f"DEBUG ALIGNER [_get_reference_image]: Tentative chargement référence manuelle: {self.reference_image_path}")
            try:
                ref_img_loaded_manual = load_and_validate_fits(self.reference_image_path)
                if ref_img_loaded_manual is None:
                    # load_and_validate_fits devrait déjà afficher une erreur
                    raise ValueError(f"Échec chargement/validation de la référence manuelle: {manual_ref_basename}")

                ref_hdr_loaded_manual = fits.getheader(self.reference_image_path)
                
                # Pré-traitement de la référence manuelle (Debayer, Hot Pixels)
                prepared_ref_manual = ref_img_loaded_manual.astype(np.float32) # Assurer float32
                if prepared_ref_manual.ndim == 2: # Si monochrome, tenter debayering
                    bayer_pat_ref_manual = ref_hdr_loaded_manual.get('BAYERPAT', self.bayer_pattern)
                    if isinstance(bayer_pat_ref_manual, str) and bayer_pat_ref_manual.upper() in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                        try:
                            print(f"DEBUG ALIGNER [_get_reference_image]: Debayering référence manuelle (Pattern: {bayer_pat_ref_manual.upper()})...")
                            prepared_ref_manual = debayer_image(prepared_ref_manual, bayer_pat_ref_manual.upper())
                        except ValueError as deb_err_manual:
                            self.update_progress(f"⚠️ Réf Manuelle: Erreur Debayer ({deb_err_manual}). Utilisation N&B.")
                            print(f"DEBUG ALIGNER [_get_reference_image]: Erreur Debayer réf manuelle (conservée N&B): {deb_err_manual}")
                
                if self.correct_hot_pixels:
                    try:
                        self.update_progress("🔥 Correction px chauds sur référence manuelle...")
                        print(f"DEBUG ALIGNER [_get_reference_image]: Correction HP référence manuelle (Seuil: {self.hot_pixel_threshold}, Voisinage: {self.neighborhood_size})...")
                        prepared_ref_manual = detect_and_correct_hot_pixels(prepared_ref_manual, self.hot_pixel_threshold, self.neighborhood_size)
                    except Exception as hp_err_manual:
                        self.update_progress(f"⚠️ Réf Manuelle: Erreur correction px chauds: {hp_err_manual}")
                        print(f"DEBUG ALIGNER [_get_reference_image]: Erreur HP réf manuelle: {hp_err_manual}")

                reference_image_data = prepared_ref_manual.astype(np.float32) # Assurer float32 final
                reference_header = ref_hdr_loaded_manual
                self.update_progress(f"✅ Référence manuelle chargée et pré-traitée: {manual_ref_basename} (dims {reference_image_data.shape})")
                print(f"DEBUG ALIGNER [_get_reference_image]: Référence manuelle OK.")
                # Si la référence manuelle est chargée, on ne fait pas la sélection auto
                return reference_image_data, reference_header

            except Exception as e_manual_ref:
                self.update_progress(f"❌ Erreur chargement/préparation référence manuelle ({manual_ref_basename}): {e_manual_ref}. Tentative sélection auto...")
                print(f"DEBUG ALIGNER [_get_reference_image]: Échec référence manuelle: {e_manual_ref}. Passage à sélection auto.")
                reference_image_data = None # Forcer la sélection auto si la manuelle échoue
                reference_header = None
        
        # --- Étape 2: Sélection Automatique si pas de Référence Manuelle Valide ---
        # Cette section s'exécute si reference_image_data est toujours None
        self.update_progress("⚙️ Sélection auto de la meilleure image de référence...")
        if not files_to_scan: # Vérifier si la liste des fichiers à scanner est vide
             self.update_progress("❌ [GET_REF/Auto] Impossible sélectionner: aucun fichier fourni pour analyse.")
             print("DEBUG ALIGNER [_get_reference_image]: Liste 'files_to_scan' vide pour sélection auto.")
             return None, None

        best_image_data_auto = None   # Pour stocker les données du meilleur candidat auto
        best_header_data_auto = None  # Pour stocker le header du meilleur candidat auto
        best_file_name_auto = None    # Nom du meilleur candidat
        max_metric_auto = -np.inf     # Initialiser avec une valeur très basse

        # Déterminer le nombre d'images à analyser pour trouver la référence
        num_to_analyze_auto = min(self.NUM_IMAGES_FOR_AUTO_REF, len(files_to_scan))
        
        # Message d'info pour l'utilisateur et le log
        self.update_progress(f"🔍 [GET_REF/Auto] Analyse des {num_to_analyze_auto} premières images (sur {len(files_to_scan)} candidates) pour référence...")
        print(f"DEBUG ALIGNER [_get_reference_image/Auto]: NUM_IMAGES_FOR_AUTO_REF={self.NUM_IMAGES_FOR_AUTO_REF}. Analysera {num_to_analyze_auto} images.")

        iterable_candidates = files_to_scan[:num_to_analyze_auto] # Sélectionner le sous-ensemble
        disable_tqdm_auto = self.progress_callback is not None # Désactiver tqdm si on a un callback GUI

        with tqdm(total=num_to_analyze_auto, desc="Analyse Réf. Auto", disable=disable_tqdm_auto, leave=False) as pbar_auto:
            for i_cand, f_name_cand in enumerate(iterable_candidates):
                if self.stop_processing: # Vérifier si arrêt demandé
                    print("DEBUG ALIGNER [_get_reference_image/Auto]: Arrêt demandé pendant sélection auto.")
                    return None, None
                
                current_file_path_cand = os.path.join(input_folder, f_name_cand)
                processed_candidates_auto += 1
                rejection_reason_cand = None # Raison de rejet pour CE candidat
                
                print(f"DEBUG ALIGNER [_get_reference_image/Auto Cand. {i_cand+1}/{num_to_analyze_auto}]: Traitement '{f_name_cand}'")

                try:
                    # 1. Charger et valider le candidat
                    img_cand = load_and_validate_fits(current_file_path_cand)
                    if img_cand is None:
                        rejection_reason_cand = "load"; print(f"  -> Échec load_and_validate_fits pour '{f_name_cand}'")
                        raise ValueError("Load/Validate fail")
                    
                    hdr_cand = fits.getheader(current_file_path_cand)
                    
                    # 2. Vérification de variance (simple)
                    std_dev_cand = np.std(img_cand)
                    variance_threshold_cand = 0.0020 # Seuil (peut être ajusté)
                    print(f"  -> '{f_name_cand}' - StdDev: {std_dev_cand:.4f} (Seuil: {variance_threshold_cand})")
                    if std_dev_cand < variance_threshold_cand:
                        rejection_reason_cand = "variance"; print(f"  -> Faible variance pour '{f_name_cand}'")
                        raise ValueError(f"Low variance ({std_dev_cand:.4f})")

                    # 3. Pré-traitement du candidat (Debayer, HP)
                    prepared_img_cand = img_cand.astype(np.float32)
                    if prepared_img_cand.ndim == 2:
                         bayer_pat_s_cand = hdr_cand.get('BAYERPAT', self.bayer_pattern)
                         if isinstance(bayer_pat_s_cand, str) and bayer_pat_s_cand.upper() in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                              try:
                                  prepared_img_cand = debayer_image(prepared_img_cand, bayer_pat_s_cand.upper())
                              except ValueError as de_cand:
                                  print(f"  -> Debayer échec pour '{f_name_cand}' (conservée N&B): {de_cand}")
                    if self.correct_hot_pixels:
                         try:
                             prepared_img_cand = detect_and_correct_hot_pixels(prepared_img_cand, self.hot_pixel_threshold, self.neighborhood_size)
                         except Exception as hpe_cand:
                             print(f"  -> Correction HP échec pour '{f_name_cand}': {hpe_cand}")
                    
                    # 4. Calcul de la métrique de qualité (simple médiane / MAD pour le moment)
                    median_val_cand = np.median(prepared_img_cand)
                    # MAD (Median Absolute Deviation) comme estimateur robuste de l'écart-type du bruit
                    mad_val_cand = np.median(np.abs(prepared_img_cand - median_val_cand))
                    approx_std_cand = mad_val_cand * 1.4826 # Facteur de conversion MAD -> Std pour Gaussienne
                    # Métrique simple: Signal (médiane) sur Bruit (approx_std)
                    metric_cand = median_val_cand / (approx_std_cand + 1e-7) if median_val_cand > 0 and approx_std_cand > 1e-7 else -np.inf # Eviter division par zéro
                    print(f"  -> '{f_name_cand}' - Métrique: {metric_cand:.3f} (MédianeVal={median_val_cand:.3f}, ApproxBruit={approx_std_cand:.3f})")

                    if not np.isfinite(metric_cand):
                        rejection_reason_cand = "metric"; print(f"  -> Métrique non-finie pour '{f_name_cand}'")
                        raise ValueError("Metric non-finite")

                    # 5. Comparer et stocker le meilleur candidat
                    if metric_cand > max_metric_auto:
                        print(f"  -> NOUVEAU MEILLEUR CANDIDAT: '{f_name_cand}', Métrique={metric_cand:.3f} (Préc. Max={max_metric_auto:.3f})")
                        max_metric_auto = metric_cand
                        best_image_data_auto = prepared_img_cand.copy() # Copier les données pré-traitées
                        best_header_data_auto = hdr_cand.copy()
                        best_file_name_auto = f_name_cand
                    # else: # Log si on skippe (peut être verbeux)
                        # print(f"  -> Ignoré (pas meilleur): '{f_name_cand}', Métrique={metric_cand:.3f} (Meilleur actuel='{best_file_name_auto}', Métrique={max_metric_auto:.3f})")

                except Exception as e_cand:
                    # Logguer l'erreur pour ce candidat spécifique mais continuer la boucle
                    self.update_progress(f"⚠️ Erreur analyse réf. auto '{f_name_cand}': {e_cand}")
                    rejected_candidates_auto += 1
                    if rejection_reason_cand: rejection_reasons_auto[rejection_reason_cand] += 1
                    else: rejection_reasons_auto['preprocess'] += 1
                    print(f"  -> '{f_name_cand}' REJETÉ. Raison: {rejection_reason_cand or 'preprocess_error_in_loop'}")
                finally:
                    pbar_auto.update(1) # Mettre à jour la barre de progression tqdm
                    # Nettoyer la mémoire pour ce candidat
                    if 'img_cand' in locals(): del img_cand
                    if 'hdr_cand' in locals(): del hdr_cand
                    if 'prepared_img_cand' in locals(): del prepared_img_cand
                    if i_cand % 10 == 0: gc.collect() # GC occasionnel

        # --- Après la boucle de sélection automatique ---
        if best_image_data_auto is not None:
            reference_image_data = best_image_data_auto
            reference_header = best_header_data_auto
            self.update_progress(f"⭐ Référence auto sélectionnée: {best_file_name_auto} (Métrique: {max_metric_auto:.2f})")
            print(f"DEBUG ALIGNER [_get_reference_image/Auto]: Sélection auto RÉUSSIE. Fichier: {best_file_name_auto}, Métrique: {max_metric_auto:.2f}")
            if rejected_candidates_auto > 0:
                 reason_str_auto = ", ".join(f"{k}:{v}" for k,v in rejection_reasons_auto.items() if v > 0)
                 self.update_progress(f"   (Info auto: {processed_candidates_auto} analysés, {rejected_candidates_auto} rejetés [{reason_str_auto}])")
        else:
            # Si aucun meilleur candidat n'a été trouvé après la boucle
            reason_str_auto = ", ".join(f"{k}:{v}" for k,v in rejection_reasons_auto.items() if v > 0)
            final_msg_auto = f"❌ [GET_REF/Auto] Aucune référence valide trouvée après analyse de {processed_candidates_auto} images. "
            if reason_str_auto: final_msg_auto += f"Raisons rejet pendant analyse: [{reason_str_auto}]."
            else: final_msg_auto += "Aucun candidat n'a été jugé 'meilleur' ou n'a passé les filtres initiaux."
            self.update_progress(final_msg_auto)
            print(f"DEBUG ALIGNER [_get_reference_image/Auto]: ÉCHEC final sélection auto. best_image_data_auto est None. MaxMétrique atteinte: {max_metric_auto:.3f}. Traités: {processed_candidates_auto}, Rejetés (pendant boucle): {rejected_candidates_auto}. Raisons: {reason_str_auto}")
            return None, None # Retourner None si la sélection auto échoue

        print(f"DEBUG ALIGNER [_get_reference_image]: Fin. Image réf shape: {reference_image_data.shape if reference_image_data is not None else 'None'}")
        return reference_image_data, reference_header




    # --- _save_reference_image (Unchanged) ---
    def _save_reference_image(self, reference_image, reference_header, base_output_folder):
        """
        Sauvegarde l'image de référence (float32 0-1) au format FITS (uint16)
        dans le dossier temporaire DEDANS base_output_folder.
        """
        if reference_image is None: return
        temp_folder_ref = os.path.join(base_output_folder, "temp_processing")
        try:
            os.makedirs(temp_folder_ref, exist_ok=True)
            ref_output_path = os.path.join(temp_folder_ref, "reference_image.fit")
            ref_preview_path = os.path.join(temp_folder_ref, "reference_image.png")
            save_header = reference_header.copy() if reference_header else fits.Header()
            save_header.set('REFRENCE', True, 'Stacking reference image')
            if self.correct_hot_pixels: save_header.set('HOTPIXCR', True, 'Hot pixel correction applied to ref')
            save_header.add_history("Reference image saved by SeestarAligner")
            save_fits_image(reference_image, ref_output_path, save_header, overwrite=True)
            self.update_progress(f"📁 Image référence sauvegardée: {ref_output_path}")
            save_preview_image(reference_image, ref_preview_path, apply_stretch=True)
        except Exception as e:
            self.update_progress(f"⚠️ Erreur lors de la sauvegarde de l'image de référence: {e}")
            traceback.print_exc(limit=2)

    # --- _align_image (Unchanged) ---
    def _align_image(self, img_to_align, reference_image, file_name):
        """Aligns a single image (float32 0-1) to the reference (float32 0-1)."""
        if reference_image is None: self.update_progress(f"❌ Alignement impossible {file_name}: Référence non disponible."); return img_to_align, False
        img_to_align = img_to_align.astype(np.float32); reference_image = reference_image.astype(np.float32)
        try:
            img_for_detection = img_to_align[:, :, 1] if img_to_align.ndim == 3 else img_to_align
            ref_for_detection = reference_image[:, :, 1] if reference_image.ndim == 3 else reference_image
            if img_for_detection.shape != ref_for_detection.shape: self.update_progress(f"❌ Alignement {file_name}: Dimensions incompatibles Réf={ref_for_detection.shape}, Img={img_for_detection.shape}"); return img_to_align, False
            aligned_img, _ = aa.register(source=img_to_align, target=reference_image, max_control_points=50, detection_sigma=5, min_area=5)
            if aligned_img is None: raise aa.MaxIterError("Alignement échoué (pas de transformation trouvée)")
            aligned_img = np.clip(aligned_img.astype(np.float32), 0.0, 1.0); return aligned_img, True
        except aa.MaxIterError as ae: self.update_progress(f"⚠️ Alignement échoué {file_name}: {ae}"); return img_to_align, False
        except ValueError as ve: self.update_progress(f"❌ Erreur alignement {file_name} (ValueError): {ve}"); return img_to_align, False
        except Exception as e: self.update_progress(f"❌ Erreur alignement inattendue {file_name}: {e}"); traceback.print_exc(limit=3); return img_to_align, False

# --- _align_batch (Unchanged - returns results, doesn't save aligned files here) ---
    def _align_batch(self, images_data, original_indices, reference_image, input_folder, output_folder, unaligned_folder):
        """Aligns a batch of images (data provided) in parallel."""
        num_cores = os.cpu_count() or 1
        max_workers = min(max(num_cores // 2, 1), 8)
        self.update_progress(f"🧵 Alignement parallèle lot avec {max_workers} threads...")

        def align_single_image_task(args):
            idx_in_batch, (img_float_01, hdr, fname), original_file_index = args
            if self.stop_processing:
                return None
            try:
                aligned_img, success = self._align_image(img_float_01, reference_image, fname)
                if not success:
                    original_path = os.path.join(input_folder, fname)
                    out_path = os.path.join(unaligned_folder, f"unaligned_{original_file_index:04d}_{fname}")
                    if os.path.exists(original_path):
                        shutil.copy2(original_path, out_path)
                    return (original_file_index, False, f"Échec alignement: {fname}")
                return (original_file_index, True, aligned_img, hdr)  # index, success, data, header
            except Exception as e:
                error_msg = f"Erreur tâche alignement {fname}: {e}"
                self.update_progress(f"❌ {error_msg}")
                try:
                    original_path = os.path.join(input_folder, fname)
                    out_path = os.path.join(unaligned_folder, f"error_{original_file_index:04d}_{fname}")
                    if os.path.exists(original_path):
                        shutil.copy2(original_path, out_path)
                except Exception:
                    pass
                return (original_file_index, False, None, None)  # index, success, data, header

        task_args = [(i, data_tuple, original_indices[i]) for i, data_tuple in enumerate(images_data)]
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(align_single_image_task, args): args for args in task_args}
            for future in concurrent.futures.as_completed(futures):
                if self.stop_processing:
                    for f in futures:
                        f.cancel()
                    self.update_progress("⛔ Alignement lot interrompu.")
                    break
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except concurrent.futures.CancelledError:
                    pass
                except Exception as future_err:
                    orig_fname = futures[future][1][2]
                    self.update_progress(f"❗️ Erreur récupération résultat pour {orig_fname}: {future_err}")
                    results.append((futures[future][2], False, None, None))

        success_count = sum(1 for _, success, _, _ in results if success)
        fail_count = len(results) - success_count
        self.update_progress(f"🏁 Alignement lot terminé: {success_count} succès, {fail_count} échecs.")
        return results
    
# --- Compatibility Function (Unchanged) ---
def align_seestar_images_batch(*args, **kwargs):
    """(Compatibility) Use SeestarAligner().align_images(...) directly."""
    aligner = SeestarAligner()
    if 'bayer_pattern' in kwargs: aligner.bayer_pattern = kwargs['bayer_pattern']
    if 'batch_size' in kwargs: aligner.batch_size = kwargs['batch_size']
    if 'reference_image_path' in kwargs: aligner.reference_image_path = kwargs['reference_image_path']
    if 'correct_hot_pixels' in kwargs: aligner.correct_hot_pixels = kwargs['correct_hot_pixels']
    if 'hot_pixel_threshold' in kwargs: aligner.hot_pixel_threshold = kwargs['hot_pixel_threshold']
    if 'neighborhood_size' in kwargs: aligner.neighborhood_size = kwargs['neighborhood_size']
    if 'progress_callback' in kwargs: aligner.set_progress_callback(kwargs['progress_callback'])
    # Call the main method, passing only relevant args
    return aligner.align_images(args[0], kwargs.get('output_folder'), kwargs.get('specific_files'))
# --- END OF FILE seestar/core/alignment.py ---