"""
Module pour l'alignement des images astronomiques.
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

from .image_processing import (
    load_and_validate_fits,
    debayer_image,
    save_fits_image
)
from .hot_pixels import detect_and_correct_hot_pixels
from .utils import estimate_batch_size

warnings.filterwarnings("ignore", category=FutureWarning)

class SeestarAligner:
    """
    Classe pour l'alignement des images astronomiques de Seestar.
    """
    def __init__(self):
        """Initialise l'aligneur avec des valeurs par défaut."""
        self.bayer_pattern = "GRBG"
        self.batch_size = 0  # 0 signifie auto-détection basée sur la mémoire disponible
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
        if self.progress_callback:
            self.progress_callback(message, progress)
        else:
            print(message)
    
    def align_images(self, input_folder, output_folder=None, specific_files=None):
        """
        Aligne les images FITS dans le dossier d'entrée.
        
        Args:
            input_folder (str): Chemin vers le dossier contenant les images FITS
            output_folder (str): Dossier de sortie (si None, crée un sous-dossier 'aligned_lights')
            specific_files (list, optional): Liste de noms de fichiers spécifiques à traiter
                
        Returns:
            str: Chemin vers le dossier contenant les images alignées
        """
        self.stop_processing = False
        
        # Définir le dossier de sortie si non spécifié
        if output_folder is None:
            output_folder = os.path.join(input_folder, "aligned_lights")
        
        # Créer les dossiers nécessaires
        os.makedirs(output_folder, exist_ok=True)
        unaligned_folder = os.path.join(output_folder, "unaligned")
        os.makedirs(unaligned_folder, exist_ok=True)
        
        # Vérifier la taille des lots
        if self.batch_size <= 0:
            sample_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.fit', '.fits'))]
            if sample_files:
                sample_path = os.path.join(input_folder, sample_files[0])
                self.batch_size = estimate_batch_size(sample_path)
                self.update_progress(f"🧠 Taille de lot dynamique estimée : {self.batch_size}")
            else:
                self.batch_size = 10  # Valeur par défaut
        
        # Récupérer la liste des fichiers FITS
        if specific_files:
            # Utiliser la liste de fichiers spécifiques si fournie
            files = specific_files
        else:
            # Sinon, récupérer tous les fichiers FITS du dossier
            files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.fit', '.fits'))]
        
        if not files:
            self.update_progress("❌ Aucun fichier .fit/.fits trouvé")
            return output_folder
        
        self.update_progress(f"🔍 Analyse de {len(files)} images...")
        
        # Obtenir l'image de référence
        fixed_reference_image, fixed_reference_header = self._get_reference_image(input_folder, files)
        if fixed_reference_image is None:
            self.update_progress("❌ Impossible de trouver une image de référence valide.")
            return output_folder
        
        # Sauvegarder l'image de référence
        self._save_reference_image(fixed_reference_image, fixed_reference_header, output_folder)
        
        # Traiter les images par lots
        total_files = len(files)
        processed_count = 0
        start_time = time.time()
        
        for batch_start in range(0, total_files, self.batch_size):
            if self.stop_processing:
                self.update_progress("⛔ Traitement arrêté par l'utilisateur.")
                break
            
            batch_end = min(batch_start + self.batch_size, total_files)
            batch_files = files[batch_start:batch_end]
            
            batch_progress = batch_start * 100 / total_files
            self.update_progress(
                f"🚀 Traitement du lot {batch_start//self.batch_size + 1}/{(total_files-1)//self.batch_size + 1} "
                f"(images {batch_start+1} à {batch_end})...", 
                batch_progress
            )
            
            # Charger et traiter les images du lot
            images = []
            headers = []
            valid_files = []
            
            for i, file in enumerate(batch_files):
                if self.stop_processing:
                    break
                
                try:
                    file_path = os.path.join(input_folder, file)
                    img = load_and_validate_fits(file_path)
                    
                    # Vérifier la qualité de l'image
                    if np.std(img) > 5:
                        # Convertir en couleur si nécessaire
                        if img.ndim == 2:
                            img = debayer_image(img, self.bayer_pattern)
                        elif img.ndim == 3 and img.shape[0] == 3:
                            # Pour les images 3D avec la première dimension comme canal
                            img = np.moveaxis(img, 0, -1)  # Convert to HxWx3
                        
                        # Appliquer la correction des pixels chauds si demandé
                        if self.correct_hot_pixels:
                            img = detect_and_correct_hot_pixels(
                                img, 
                                threshold=self.hot_pixel_threshold,
                                neighborhood_size=self.neighborhood_size
                            )
                        
                        img = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)
                        
                        images.append(img)
                        headers.append(fits.getheader(file_path))
                        valid_files.append(file)
                        
                        # Mise à jour de la progression
                        processed_count += 1
                        percent_done = (batch_start + i + 1) * 100 / total_files
                        elapsed_time = time.time() - start_time
                        
                        if processed_count > 0:
                            time_per_image = elapsed_time / processed_count
                            remaining_images = total_files - (batch_start + i + 1)
                            estimated_time_remaining = remaining_images * time_per_image
                            hours, remainder = divmod(int(estimated_time_remaining), 3600)
                            minutes, seconds = divmod(remainder, 60)
                            time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
                            
                            self.update_progress(
                                f"📊 Traitement de {file}... ({processed_count}/{total_files}) "
                                f"Temps restant estimé: {time_str}", 
                                percent_done
                            )
                        else:
                            self.update_progress(
                                f"📊 Traitement de {file}... ({processed_count}/{total_files})",
                                percent_done
                            )
                    else:
                        self.update_progress(f"⚠️ Image ignorée (faible variance): {file}")
                except Exception as e:
                    self.update_progress(f"⚠️ Erreur lors du traitement de {file}: {e}")
            
            # Alignement des images du lot
            if len(images) < 1:
                self.update_progress(f"❌ Aucune image valide dans le lot {batch_start//self.batch_size + 1}.")
                continue
            
            self._align_batch(images, headers, valid_files, fixed_reference_image, 
                            input_folder, output_folder, unaligned_folder, batch_start)
            
            # Libérer la mémoire
            del images
            del headers
            gc.collect()
        
        self.update_progress(f"✅ Toutes les images alignées ont été sauvegardées dans: {output_folder}")
        return output_folder
        
    def _get_reference_image(self, input_folder, files):
        """
        Obtient l'image de référence pour l'alignement.
        
        Args:
            input_folder (str): Dossier contenant les images
            files (list): Liste des fichiers FITS
            
        Returns:
            tuple: (reference_image, reference_header) ou (None, None) si échec
        """
        fixed_reference_image = None
        fixed_reference_header = None
        
        # Tenter de charger l'image de référence manuelle si fournie
        if self.reference_image_path:
            try:
                self.update_progress(f"📌 Chargement de l'image de référence manuelle : {self.reference_image_path}")
                fixed_reference_image = load_and_validate_fits(self.reference_image_path)
                fixed_reference_header = fits.getheader(self.reference_image_path)
                
                # Appliquer debayer sur l'image de référence si c'est une image brute
                if fixed_reference_image.ndim == 2:
                    fixed_reference_image = debayer_image(fixed_reference_image, self.bayer_pattern)
                elif fixed_reference_image.ndim == 3 and fixed_reference_image.shape[0] == 3:
                    # Pour les images 3D avec la première dimension comme canal
                    fixed_reference_image = np.moveaxis(fixed_reference_image, 0, -1)  # Convert to HxWx3
                
                # Appliquer la correction des pixels chauds si demandé
                if self.correct_hot_pixels:
                    self.update_progress("🔥 Application de la correction des pixels chauds sur l'image de référence...")
                    fixed_reference_image = detect_and_correct_hot_pixels(
                        fixed_reference_image, 
                        threshold=self.hot_pixel_threshold,
                        neighborhood_size=self.neighborhood_size
                    )
                    
                fixed_reference_image = cv2.normalize(fixed_reference_image, None, 0, 65535, cv2.NORM_MINMAX)
                self.update_progress(f"✅ Image de référence chargée: dimensions: {fixed_reference_image.shape}")
            except Exception as e:
                self.update_progress(f"❌ Erreur lors du chargement de l'image de référence manuelle: {e}")
                fixed_reference_image = None
        
        # Sélection automatique de la meilleure image de référence si aucune n'est spécifiée
        if fixed_reference_image is None:
            self.update_progress("⚙️ Recherche de la meilleure image de référence...")
            sample_images = []
            sample_headers = []
            sample_files = []
            
            # Analyser un sous-ensemble d'images pour trouver la meilleure référence
            for f in tqdm(files[:min(self.batch_size*2, len(files))], desc="Analyse des images"):
                try:
                    img_path = os.path.join(input_folder, f)
                    img = load_and_validate_fits(img_path)
                    hdr = fits.getheader(img_path)
                    
                    # S'assurer que l'image a une variance suffisante
                    if np.std(img) > 3:
                        # Convertir en couleur si nécessaire
                        if img.ndim == 2:
                            img = debayer_image(img, self.bayer_pattern)
                        elif img.ndim == 3 and img.shape[0] == 3:
                            # Pour les images 3D avec la première dimension comme canal
                            img = np.moveaxis(img, 0, -1)  # Convert to HxWx3
                        
                        # Appliquer la correction des pixels chauds si demandé
                        if self.correct_hot_pixels:
                            img = detect_and_correct_hot_pixels(
                                img, 
                                threshold=self.hot_pixel_threshold,
                                neighborhood_size=self.neighborhood_size
                            )
                        
                        img = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)
                        
                        sample_images.append(img)
                        sample_headers.append(hdr)
                        sample_files.append(f)
                    else:
                        self.update_progress(f"⚠️ Image ignorée (faible variance): {f}")
                except Exception as e:
                    self.update_progress(f"❌ Erreur lors de l'analyse de {f}: {e}")
            
            if sample_images:
                # Sélectionner l'image avec le meilleur contraste (médiane la plus élevée)
                medians = [np.median(img) for img in sample_images]
                ref_idx = np.argmax(medians)
                fixed_reference_image = sample_images[ref_idx]
                fixed_reference_header = sample_headers[ref_idx]
                self.update_progress(f"⭐ Référence utilisée: {sample_files[ref_idx]}")
        
        return fixed_reference_image, fixed_reference_header
    
    def _save_reference_image(self, reference_image, reference_header, output_folder):
        """
        Sauvegarde l'image de référence dans le dossier de sortie.
        
        Args:
            reference_image (numpy.ndarray): Image de référence
            reference_header (astropy.io.fits.Header): En-tête de l'image de référence
            output_folder (str): Dossier de sortie
        """
        try:
            ref_output_path = os.path.join(output_folder, "reference_image.fit")
            ref_data = np.moveaxis(reference_image, -1, 0).astype(np.uint16)  # HxWx3 to 3xHxW
            
            new_header = reference_header.copy()
            new_header['NAXIS'] = 3
            new_header['NAXIS1'] = reference_image.shape[1]
            new_header['NAXIS2'] = reference_image.shape[0]
            new_header['NAXIS3'] = 3
            new_header['BITPIX'] = 16
            new_header.set('CTYPE3', 'RGB', 'Couleurs RGB')
            new_header.set('REFRENCE', True, 'stacking reference')
            
            fits.writeto(ref_output_path, ref_data, new_header, overwrite=True)
            self.update_progress(f"📁 Image de référence sauvegardée: {ref_output_path}")
        except Exception as e:
            self.update_progress(f"⚠️ Erreur lors de la sauvegarde de l'image de référence: {e}")
    
    def _align_batch(self, images, headers, filenames, reference_image,
                     input_folder, output_folder, unaligned_folder, batch_start):
        import concurrent.futures
        import os
        from functools import partial

        # Déterminer le nombre optimal de threads
        num_cores = os.cpu_count()
        max_workers = min(max(num_cores // 2, 1), 8)  # Utiliser la moitié des cœurs, mais au moins 1 et au plus 8
        self.update_progress(f"🧵 Démarrage de l'alignement parallèle avec {max_workers} threads...")

        # Fonction pour aligner une seule image (à exécuter dans un thread)
        def align_single_image(args):
            i, img, hdr, fname = args
            if self.stop_processing:
                return None

            try:
                # Alignement canal par canal pour les images couleur
                if img.ndim == 3:
                    aligned_channels = []
                    for c in range(3):
                        # Normalisation pour l'alignement
                        img_norm = cv2.normalize(img[:, :, c], None, 0, 1, cv2.NORM_MINMAX)
                        ref_norm = cv2.normalize(reference_image[:, :, c], None, 0, 1, cv2.NORM_MINMAX)

                        aligned_channel, _ = aa.register(img_norm, ref_norm)
                        aligned_channels.append(aligned_channel)

                    # Recombiner les canaux
                    aligned_img = np.stack(aligned_channels, axis=-1)
                    aligned_img = cv2.normalize(aligned_img, None, 0, 65535, cv2.NORM_MINMAX)
                else:
                    # Cas d'une image en niveaux de gris
                    aligned_img, _ = aa.register(img, reference_image)
                    aligned_img = np.stack((aligned_img,) * 3, axis=-1)

                # Conversion au format FITS pour l'enregistrement (HxWx3 -> 3xHxW)
                color_cube = np.moveaxis(aligned_img, -1, 0).astype(np.uint16)

                # Mettre à jour l'en-tête
                new_header = hdr.copy()
                new_header['NAXIS'] = 3
                new_header['NAXIS1'] = aligned_img.shape[1]
                new_header['NAXIS2'] = aligned_img.shape[0]
                new_header['NAXIS3'] = 3
                new_header['BITPIX'] = 16
                new_header.set('CTYPE3', 'RGB', 'Couleurs RGB')
                new_header.set('ALIGNED', True, 'Image Aligned on ref')

                if self.correct_hot_pixels:
                    new_header.set('HOTPIXEL', True, 'Hot pixels correction applied')
                    new_header.set('HOTPXTH', self.hot_pixel_threshold, 'Hot pixels detection threshold')
                    new_header.set('HOTPXNB', self.neighborhood_size, 'Hot pixels neighborhood size')

                # Enregistrement de l'image alignée
                out_path = os.path.join(output_folder, f"aligned_{batch_start + i:04}.fit")
                fits.writeto(out_path, color_cube, new_header, overwrite=True)

                return (i, True, None)  # Succès
            except Exception as e:
                # Gérer l'erreur, enregistrer l'image non alignée
                try:
                    original_path = os.path.join(input_folder, fname)
                    out_path = os.path.join(unaligned_folder, f"unaligned_{fname}")
                    shutil.copy2(original_path, out_path)
                    return (i, False, str(e))  # Échec avec erreur
                except Exception as copy_err:
                    return (i, False, f"Erreur double: {str(e)} et {str(copy_err)}")  # Échec avec erreur double

        # Créer les arguments pour chaque image à traiter
        image_args = [(i, img, hdr, fname) for i, (img, hdr, fname) in enumerate(zip(images, headers, filenames))]

        # Utiliser ThreadPoolExecutor pour paralléliser le traitement
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Soumettre toutes les tâches et collecter les futurs
            futures = [executor.submit(align_single_image, args) for args in image_args]

            # Traiter les résultats au fur et à mesure qu'ils sont disponibles
            for future in concurrent.futures.as_completed(futures):
                if self.stop_processing:
                    executor.shutdown(wait=False)
                    self.update_progress("⛔ Alignement arrêté par l'utilisateur.")
                    break

                result = future.result()
                if result:
                    i, success, error_msg = result
                    if success:
                        # Mise à jour de la progression pour chaque image alignée avec succès
                        self.update_progress(f"✓ Image {batch_start + i:04} alignée avec succès")
                    else:
                        # Signaler l'échec
                        self.update_progress(f"❌ Échec de l'alignement pour l'image {batch_start + i:04}: {error_msg}")

# Fonction d'aide pour la compatibilité avec l'ancien code
def align_seestar_images_batch(input_folder, bayer_pattern="GRBG", batch_size=10, manual_reference_path=None,
                               correct_hot_pixels=True, hot_pixel_threshold=3.0, neighborhood_size=5):
    """
    Aligne les images Seestar par lots avec une image de référence optionnelle.
    Cette fonction est maintenue pour la compatibilité avec l'ancien code.
    
    Args:
        input_folder (str): Chemin vers le dossier d'entrée contenant les fichiers FITS
        bayer_pattern (str): Motif Bayer pour le débayering
        batch_size (int): Nombre d'images à traiter par lot
        manual_reference_path (str): Chemin optionnel vers une image de référence manuelle
        correct_hot_pixels (bool): Corriger les pixels chauds
        hot_pixel_threshold (float): Seuil pour la détection des pixels chauds
        neighborhood_size (int): Taille du voisinage pour le calcul médian
        
    Returns:
        str: Chemin vers le dossier contenant les images alignées
    """
    aligner = SeestarAligner()
    aligner.bayer_pattern = bayer_pattern
    aligner.batch_size = batch_size
    aligner.reference_image_path = manual_reference_path
    aligner.correct_hot_pixels = correct_hot_pixels
    aligner.hot_pixel_threshold = hot_pixel_threshold
    aligner.neighborhood_size = neighborhood_size
    
    return aligner.align_images(input_folder)        