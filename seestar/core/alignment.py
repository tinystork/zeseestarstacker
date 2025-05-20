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
from astropy.io import fits # Si pas déjà là
try: from tqdm import tqdm # Optionnel, pour la barre de progression console
except ImportError: tqdm = lambda x, **kwargs: x

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
        self.NUM_IMAGES_FOR_AUTO_REF = 20 # Ou une autre valeur par défaut
    
    
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






# DANS LA CLASSE SeestarAligner (dans seestar/core/alignment.py)

    def _get_reference_image(self, input_folder, files_to_scan, output_folder_for_saving_temp_ref):
        """
        Obtient l'image de référence (float32, 0-1) et son en-tête.
        Tente de charger une référence manuelle si spécifiée, sinon sélectionne
        automatiquement la meilleure parmi un sous-ensemble des 'files_to_scan'.
        Sauvegarde l'image de référence traitée dans le dossier temporaire spécifié.

        Args:
            input_folder (str): Dossier principal pour la recherche auto si pas de réf manuelle.
            files_to_scan (list): Liste des noms de fichiers dans input_folder pour la recherche auto.
            output_folder_for_saving_temp_ref (str): Dossier de base où le sous-dossier
                                                     "temp_processing" sera créé pour sauvegarder
                                                     "reference_image.fit".

        MODIFIED V3_FilterFix_SourcePathFix + Ajout output_folder_for_saving_temp_ref pour sauvegarde
        """
        print(f"DEBUG ALIGNER [_get_reference_image]: Début. Input: '{os.path.basename(input_folder)}', Nb fichiers scan: {len(files_to_scan)}")
        print(f"  Output pour réf. temp: '{output_folder_for_saving_temp_ref}'")
        
        reference_image_data = None 
        reference_header = None     
        best_file_name_auto = None # Pour le log de la source si sélection auto

        # --- Étape 1: Essayer de Charger une Référence Manuelle si Spécifiée ---
        if self.reference_image_path and os.path.isfile(self.reference_image_path):
            manual_ref_basename = os.path.basename(self.reference_image_path)
            if hasattr(self, 'update_progress'): self.update_progress(f"📌 Chargement référence manuelle: {manual_ref_basename}")
            print(f"DEBUG ALIGNER [_get_reference_image]: Tentative chargement référence manuelle: {self.reference_image_path}")
            try:
                # --- 1.1 Chargement et validation ---
                ref_img_tuple_manual = load_and_validate_fits(self.reference_image_path) # load_and_validate_fits doit être importé
                if ref_img_tuple_manual is None or ref_img_tuple_manual[0] is None:
                    raise ValueError(f"Échec chargement/validation (données None) de la référence manuelle: {manual_ref_basename}")
                
                ref_img_loaded_manual, ref_hdr_loaded_manual = ref_img_tuple_manual 
                print(f"  Réf. Manuelle: Chargement OK. Shape brute: {ref_img_loaded_manual.shape}")

                # --- 1.2 Pré-traitement (Debayer, Hot Pixels) ---
                prepared_ref_manual = ref_img_loaded_manual.astype(np.float32) 
                if prepared_ref_manual.ndim == 2: 
                    bayer_pat_ref_manual = ref_hdr_loaded_manual.get('BAYERPAT', self.bayer_pattern)
                    if isinstance(bayer_pat_ref_manual, str) and bayer_pat_ref_manual.upper() in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                        print(f"    Réf. Manuelle: Debayering avec {bayer_pat_ref_manual.upper()}...")
                        try:
                            prepared_ref_manual = debayer_image(prepared_ref_manual, bayer_pat_ref_manual.upper()) # debayer_image doit être importé
                        except ValueError as deb_err_manual:
                            if hasattr(self, 'update_progress'): self.update_progress(f"⚠️ Réf Manuelle: Erreur Debayer ({deb_err_manual}). Utilisation N&B.")
                
                if self.correct_hot_pixels:
                    print(f"    Réf. Manuelle: Correction pixels chauds (Seuil: {self.hot_pixel_threshold}, Voisinage: {self.neighborhood_size})...")
                    try:
                        # detect_and_correct_hot_pixels doit être importé
                        prepared_ref_manual = detect_and_correct_hot_pixels(prepared_ref_manual, self.hot_pixel_threshold, self.neighborhood_size)
                    except Exception as hp_err_manual:
                        if hasattr(self, 'update_progress'): self.update_progress(f"⚠️ Réf Manuelle: Erreur correction px chauds: {hp_err_manual}")

                reference_image_data = prepared_ref_manual.astype(np.float32) 
                reference_header = ref_hdr_loaded_manual.copy() # Travailler sur une copie
                
                if reference_header is not None:
                    reference_header['_SOURCE_PATH'] = (str(manual_ref_basename), "Source filename of this reference (manual)")
                
                if hasattr(self, 'update_progress'): self.update_progress(f"✅ Référence manuelle chargée et pré-traitée: {manual_ref_basename} (dims {reference_image_data.shape})")
                
                # --- 1.3 Sauvegarde de la référence manuelle traitée ---
                if output_folder_for_saving_temp_ref:
                    print(f"  Réf. Manuelle: Appel à _save_reference_image avec base_output_folder='{output_folder_for_saving_temp_ref}'")
                    self._save_reference_image(reference_image_data, reference_header, output_folder_for_saving_temp_ref)
                else:
                    print("  AVERTISSEMENT Réf. Manuelle: output_folder_for_saving_temp_ref non fourni, réf. manuelle traitée non sauvegardée par _get_reference_image.")

                return reference_image_data, reference_header # Retourner la réf manuelle si succès

            except Exception as e_manual_ref:
                if hasattr(self, 'update_progress'): self.update_progress(f"❌ Erreur réf. manuelle ({manual_ref_basename}): {e_manual_ref}. Tentative sélection auto...")
                print(f"ERREUR ALIGNER [_get_reference_image]: Échec réf. manuelle: {e_manual_ref}")
                reference_image_data = None 
                reference_header = None
        else:
            print("DEBUG ALIGNER [_get_reference_image]: Aucune référence manuelle spécifiée ou fichier non trouvé.")
        
        # --- Étape 2: Sélection Automatique si pas de Référence Manuelle Valide ---
        if hasattr(self, 'update_progress'): self.update_progress("⚙️ Sélection auto de la meilleure image de référence...")
        print("DEBUG ALIGNER [_get_reference_image]: Passage à la sélection automatique.")

        if not files_to_scan: 
             if hasattr(self, 'update_progress'): self.update_progress("❌ [GET_REF/Auto] Impossible sélectionner: aucun fichier fourni pour analyse.")
             return None, None

        best_image_data_auto = None   
        best_header_data_auto = None  
        # best_file_name_auto est déjà initialisé à None en haut
        max_metric_auto = -np.inf     

        processed_candidates_auto = 0 
        rejected_candidates_auto = 0  
        rejection_reasons_auto = {'load': 0, 'variance': 0, 'preprocess': 0, 'metric': 0, 'filtered_name': 0, 'load_unpack_fail': 0, 'load_data_none': 0}

        num_to_analyze_initial_subset = min(getattr(self, 'NUM_IMAGES_FOR_AUTO_REF', 20), len(files_to_scan)) # NUM_IMAGES_FOR_AUTO_REF doit être un attribut de classe
        
        # --- Filtrage des noms de fichiers candidats ---
        filtered_candidates_for_ref = []
        prefixes_to_skip_for_ref = ["stack_", "mosaic_final_", "aligned_", "drizzle_"] 
        substrings_to_skip_for_ref = [
            "_reproject", "_sum.", "_wht.", "_preview.", "_temp.", 
            "reference_image.fit", "cumulative_sum.npy", "cumulative_wht.npy"
        ]
        subset_to_filter_names = files_to_scan[:num_to_analyze_initial_subset]

        for f_name_cand_iter in subset_to_filter_names:
            f_name_lower = f_name_cand_iter.lower() 
            skip_this_file = False
            for prefix in prefixes_to_skip_for_ref:
                if f_name_lower.startswith(prefix): skip_this_file = True; break
            if skip_this_file:
                rejected_candidates_auto += 1; rejection_reasons_auto['filtered_name'] += 1; continue
            for substring in substrings_to_skip_for_ref:
                if substring.lower() in f_name_lower: skip_this_file = True; break
            if skip_this_file:
                rejected_candidates_auto += 1; rejection_reasons_auto['filtered_name'] += 1; continue
            filtered_candidates_for_ref.append(f_name_cand_iter)
        
        iterable_candidates = filtered_candidates_for_ref 
        num_to_analyze_auto = len(iterable_candidates) 
        
        if num_to_analyze_auto == 0:
             if hasattr(self, 'update_progress'): self.update_progress(f"❌ [GET_REF/Auto] Aucun candidat valide après filtrage des noms (sur {num_to_analyze_initial_subset} scannés).")
             return None, None
        
        if hasattr(self, 'update_progress'): self.update_progress(f"🔍 [GET_REF/Auto] Analyse de {num_to_analyze_auto} images candidates pour référence...")
        print(f"DEBUG ALIGNER [_get_reference_image]: Début boucle d'analyse auto pour {num_to_analyze_auto} candidats.")
        
        # Utiliser tqdm si disponible, sinon simple boucle
        try: from tqdm import tqdm
        except ImportError: tqdm = lambda x, **kwargs: x # Factice si tqdm n'est pas là

        disable_tqdm_auto = (hasattr(self, 'progress_callback') and self.progress_callback is not None) or \
                            (hasattr(self, 'update_progress') and self.update_progress is not None and self.update_progress != print)


        for i_cand, f_name_cand in enumerate(tqdm(iterable_candidates, desc="Analyse Réf. Auto", disable=disable_tqdm_auto, leave=False)):
            if hasattr(self, 'stop_processing') and self.stop_processing: return None, None # Vérifier le flag d'arrêt
            
            current_file_path_cand = os.path.join(input_folder, f_name_cand)
            processed_candidates_auto += 1 
            rejection_reason_cand = None 
            # print(f"  Auto Réf. Candidat {i_cand+1}: {f_name_cand}") # Peut être trop verbeux
            
            try:
                # --- 2.1 Chargement et validation ---
                img_data_tuple_cand = load_and_validate_fits(current_file_path_cand)
                if img_data_tuple_cand is None or img_data_tuple_cand[0] is None:
                    rejection_reason_cand = "load_unpack_fail"; raise ValueError("Load/Validate returned None")
                img_cand, hdr_cand = img_data_tuple_cand
                if img_cand is None: rejection_reason_cand = "load_data_none"; raise ValueError("Load/Validate data is None")
                if hdr_cand is None: hdr_cand = fits.Header() # Créer un header vide si manquant

                # --- 2.2 Vérification variance ---
                std_dev_cand = np.std(img_cand) 
                variance_threshold_cand = 0.0005 # Seuil bas pour éviter les images quasi-noires/plates
                if std_dev_cand < variance_threshold_cand:
                    rejection_reason_cand = "variance"; raise ValueError(f"Faible variance ({std_dev_cand:.6f})")

                # --- 2.3 Pré-traitement ---
                prepared_img_cand = img_cand.astype(np.float32, copy=True) # Assurer float32 et copie
                if prepared_img_cand.ndim == 2: # Si N&B, essayer debayer
                     bayer_pat_s_cand = hdr_cand.get('BAYERPAT', self.bayer_pattern)
                     if isinstance(bayer_pat_s_cand, str) and bayer_pat_s_cand.upper() in ["GRBG", "RGGB", "GBRG", "BGGR"]:
                          try: prepared_img_cand = debayer_image(prepared_img_cand, bayer_pat_s_cand.upper())
                          except ValueError: pass # Ignorer erreur debayer, continuer avec N&B
                
                if self.correct_hot_pixels:
                     try: prepared_img_cand = detect_and_correct_hot_pixels(prepared_img_cand, self.hot_pixel_threshold, self.neighborhood_size)
                     except Exception: pass # Ignorer erreur correction HP, continuer

                # --- 2.4 Calcul métrique (SNR approximatif) ---
                median_val_cand = np.median(prepared_img_cand)
                mad_val_cand = np.median(np.abs(prepared_img_cand - median_val_cand)) # Plus robuste que std pour outliers
                approx_std_cand = mad_val_cand * 1.4826 # Conversion MAD -> std pour distribution normale
                metric_cand = median_val_cand / (approx_std_cand + 1e-9) if median_val_cand > 1e-9 and approx_std_cand > 1e-9 else -np.inf 
                
                if not np.isfinite(metric_cand) or metric_cand < -1e8: # Vérifier si la métrique est valide
                    rejection_reason_cand = "metric"; raise ValueError(f"Métrique non finie ou trop basse: {metric_cand}")

                # --- 2.5 Mise à jour du meilleur candidat ---
                if metric_cand > max_metric_auto:
                    max_metric_auto = metric_cand
                    best_image_data_auto = prepared_img_cand.copy() # Faire une copie
                    best_header_data_auto = hdr_cand.copy()         # Faire une copie
                    best_file_name_auto = f_name_cand
                    # print(f"    Nouvelle meilleure réf. auto: {f_name_cand} (Métrique: {max_metric_auto:.2f})")
            
            except Exception as e_cand_loop:
                # Ne pas utiliser self.update_progress ici pour éviter de spammer l'UI pendant la boucle rapide
                # print(f"    AVERTISSEMENT Réf. Auto '{f_name_cand}': {type(e_cand_loop).__name__} - {e_cand_loop}")
                rejected_candidates_auto += 1
                if rejection_reason_cand: rejection_reasons_auto[rejection_reason_cand] += 1
                else: rejection_reasons_auto['preprocess'] += 1 
            finally:
                # Libérer la mémoire explicitement pour les gros tableaux dans la boucle
                if 'img_cand' in locals(): del img_cand
                if 'hdr_cand' in locals(): del hdr_cand
                if 'prepared_img_cand' in locals(): del prepared_img_cand
                if 'img_data_tuple_cand' in locals(): del img_data_tuple_cand
                if i_cand > 0 and i_cand % 10 == 0: gc.collect() # GC occasionnel

        # --- Fin de la boucle d'analyse auto ---

        if best_image_data_auto is not None and best_header_data_auto is not None:
            reference_image_data = best_image_data_auto
            reference_header = best_header_data_auto
            
            if reference_header is not None and best_file_name_auto is not None:
                 reference_header['_SOURCE_PATH'] = (str(best_file_name_auto), "Source filename of this reference (auto)")
            
            if hasattr(self, 'update_progress'): self.update_progress(f"⭐ Référence auto sélectionnée: {best_file_name_auto} (Métrique: {max_metric_auto:.2f})")
            print(f"DEBUG ALIGNER [_get_reference_image]: Référence auto: {best_file_name_auto}, Métrique: {max_metric_auto:.2f}")
            if rejected_candidates_auto > 0:
                 reason_str_auto = ", ".join(f"{k}:{v}" for k,v in rejection_reasons_auto.items() if v > 0)
                 if hasattr(self, 'update_progress'): self.update_progress(f"   (Info auto: {processed_candidates_auto} traités en détail, {rejected_candidates_auto} rejetés. Raisons: [{reason_str_auto}])")
        else:
            reason_str_auto = ", ".join(f"{k}:{v}" for k,v in rejection_reasons_auto.items() if v > 0)
            final_msg_auto = f"❌ [GET_REF/Auto] Aucune référence valide après analyse de {num_to_analyze_auto} (sur {processed_candidates_auto} traités). "
            if reason_str_auto: final_msg_auto += f"Raisons: [{reason_str_auto}]."
            if hasattr(self, 'update_progress'): self.update_progress(final_msg_auto)
            print(f"ERREUR ALIGNER [_get_reference_image]: {final_msg_auto}")
            return None, None # Aucun candidat auto trouvé

        # --- Étape 3: Sauvegarde de l'image de référence sélectionnée (manuelle ou auto) ---
        if reference_image_data is not None and reference_header is not None:
            if output_folder_for_saving_temp_ref and os.path.isdir(os.path.dirname(output_folder_for_saving_temp_ref)):
                print(f"DEBUG ALIGNER [_get_reference_image]: Appel final à _save_reference_image avec base_output_folder='{output_folder_for_saving_temp_ref}'")
                self._save_reference_image(reference_image_data, reference_header, output_folder_for_saving_temp_ref)
            else:
                warning_msg_save_final = f"Output_folder_for_saving_temp_ref ('{output_folder_for_saving_temp_ref}') non valide. " \
                                         "L'image de référence finale en mémoire sera retournée, mais pas sauvegardée ici."
                if hasattr(self, 'update_progress'): self.update_progress(warning_msg_save_final)
                print(f"AVERTISSEMENT ALIGNER [_get_reference_image]: {warning_msg_save_final}")
        else: # Ne devrait pas arriver si on a passé les étapes précédentes
            print("DEBUG ALIGNER [_get_reference_image]: Données de référence ou header finaux manquants avant sauvegarde finale, étrange.")


        ref_shape_log = reference_image_data.shape if reference_image_data is not None else 'None'
        source_file_log = "Inconnue"
        if reference_header and '_SOURCE_PATH' in reference_header:
            source_file_log = reference_header['_SOURCE_PATH'][0] # Le nom de base
        elif best_file_name_auto:
             source_file_log = best_file_name_auto
        elif self.reference_image_path:
             source_file_log = os.path.basename(self.reference_image_path)

        print(f"DEBUG ALIGNER [_get_reference_image]: Fin. Réf finale de '{source_file_log}', Shape: {ref_shape_log}")
        return reference_image_data, reference_header





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