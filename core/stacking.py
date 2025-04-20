"""
Module pour l'empilement progressif des images astronomiques.
"""
import os
import numpy as np
from astropy.io import fits
import cv2
import gc
import time
from tqdm import tqdm

from .image_processing import (
    load_and_validate_fits,
    save_fits_image,
    save_preview_image
)
from .utils import apply_denoise

class ProgressiveStacker:
    """
    Classe pour l'empilement progressif des images astronomiques.
    Cette classe permet d'empiler des images par lots, en sauvegardant 
    les résultats intermédiaires et en nettoyant les fichiers alignés
    pour économiser de l'espace.
    """
    def __init__(self):
        """Initialise le stacker avec des valeurs par défaut."""
        self.stacking_mode = "kappa-sigma"  # Mode d'empilement par défaut
        self.kappa = 2.5  # Valeur kappa par défaut pour l'empilement kappa-sigma
        self.batch_size = 0  # 0 signifie auto-détection basée sur la mémoire disponible
        self.stop_processing = False
        self.progress_callback = None
        self.denoise = False  # Option pour appliquer un débruitage au stack final
    
    def set_progress_callback(self, callback):
        """Définit la fonction de rappel pour les mises à jour de progression."""
        self.progress_callback = callback
    
    def update_progress(self, message, progress=None):
        """Met à jour la progression en utilisant le callback si disponible."""
        if self.progress_callback:
            self.progress_callback(message, progress)
        else:
            print(message)
    
    def estimate_optimal_batch_size(self, input_folder, target_memory_usage=0.7):
        """
        Estime la taille optimale des lots en fonction de la mémoire disponible
        et de la taille des images.
        
        Args:
            input_folder (str): Chemin vers le dossier contenant les images
            target_memory_usage (float): Pourcentage cible d'utilisation de la mémoire (0.0-1.0)
            
        Returns:
            int: Taille de lot estimée
        """
        try:
            import psutil
            
            # Obtenir la mémoire disponible (en octets)
            available_memory = psutil.virtual_memory().available
            usable_memory = available_memory * target_memory_usage
            
            # Trouver un fichier exemple pour estimer la taille
            files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.fit', '.fits'))]
            if not files:
                self.update_progress("⚠️ Aucun fichier FITS trouvé pour l'estimation de la taille des lots.")
                return 10  # Valeur par défaut si aucun fichier trouvé
            
            sample_path = os.path.join(input_folder, files[0])
            img = load_and_validate_fits(sample_path)
            
            # Une image traitée peut prendre jusqu'à 4x plus de mémoire
            # (versions originale, débayerisée, normalisée, alignée)
            single_image_size = img.nbytes * 4
            
            # Calculer combien d'images peuvent tenir en mémoire (avec un facteur de sécurité)
            safety_factor = 2.0  # Pour tenir compte des opérations supplémentaires
            estimated_batch = int(usable_memory / (single_image_size * safety_factor))
            
            # Limites raisonnables pour la taille des lots
            estimated_batch = max(3, min(50, estimated_batch))
            
            self.update_progress(
                f"🧠 Mémoire disponible: {available_memory / (1024**3):.2f} Go, "
                f"Taille estimée d'une image: {single_image_size / (1024**2):.2f} Mo, "
                f"Taille de lot estimée: {estimated_batch}"
            )
            
            return estimated_batch
            
        except Exception as e:
            self.update_progress(f"⚠️ Erreur lors de l'estimation de la taille des lots: {e}")
            return 10  # Valeur par défaut en cas d'erreur
    
    def combine_stacks(self, stack_files, output_file, stacking_mode="mean", denoise=False):
        """
        Combine plusieurs fichiers de stack intermédiaires en un stack final.
        
        Args:
            stack_files (list): Liste des chemins vers les fichiers de stack intermédiaires
            output_file (str): Chemin du fichier de sortie pour le stack final
            stacking_mode (str): Méthode d'empilement à utiliser
            denoise (bool): Appliquer un débruitage au stack final
            
        Returns:
            bool: True si réussi, False sinon
        """
        if not stack_files:
            self.update_progress("⚠️ Aucun fichier de stack intermédiaire à combiner.")
            return False
        
        try:
            self.update_progress(f"🔄 Combinaison de {len(stack_files)} stacks intermédiaires...")
            
            # Charger tous les stacks intermédiaires
            stacks = []
            for stack_file in stack_files:
                try:
                    stack_data = load_and_validate_fits(stack_file)
                    
                    # S'assurer que les dimensions sont correctes
                    if stack_data.ndim == 3 and stack_data.shape[0] == 3:
                        # Convertir de 3xHxW à HxWx3
                        stack_data = np.moveaxis(stack_data, 0, -1)
                    
                    stacks.append(stack_data)
                except Exception as e:
                    self.update_progress(f"⚠️ Erreur lors du chargement du stack {stack_file}: {e}")
            
            if not stacks:
                self.update_progress("❌ Aucun stack intermédiaire valide trouvé.")
                return False
            
            # Combiner les stacks selon la méthode choisie
            if stacking_mode == "mean":
                final_stack = np.mean(stacks, axis=0)
            elif stacking_mode == "median":
                final_stack = np.median(stacks, axis=0)
            else:
                # Par défaut, utiliser la moyenne pour les autres méthodes
                final_stack = np.mean(stacks, axis=0)
            
            # Appliquer le débruitage si demandé
            if denoise:
                self.update_progress("🧹 Application du débruitage au stack final...")
                try:
                    final_stack = apply_denoise(final_stack, strength=5)
                except Exception as e:
                    self.update_progress(f"⚠️ Échec du débruitage: {e}")
            
            # Créer un en-tête FITS
            hdr = fits.Header()
            hdr['STACKED'] = True
            hdr['STACKTYP'] = 'combined-' + stacking_mode
            hdr['NSTACKS'] = len(stacks)
            if denoise:
                hdr['DENOISE'] = True
            
            # Sauvegarder le stack final
            save_fits_image(final_stack, output_file, header=hdr, overwrite=True)
            
            # Sauvegarder une prévisualisation PNG
            preview_file = os.path.splitext(output_file)[0] + ".png"
            save_preview_image(final_stack, preview_file)
            
            self.update_progress(f"✅ Stack final créé avec succès: {output_file}")
            return True
            
        except Exception as e:
            self.update_progress(f"❌ Erreur lors de la combinaison des stacks: {e}")
            return False

    def process_aligned_images(self, input_folder, output_folder, remove_aligned=True):
        """
        Traite les images alignées par lots, en empilant chaque lot 
        et en supprimant les images alignées après traitement.
        
        Args:
            input_folder (str): Dossier contenant les images alignées
            output_folder (str): Dossier de sortie pour les stacks
            remove_aligned (bool): Supprimer les images alignées après empilement
            
        Returns:
            list: Liste des fichiers de stack intermédiaires créés
        """
        self.stop_processing = False
        
        # Créer le dossier de sortie s'il n'existe pas
        os.makedirs(output_folder, exist_ok=True)
        
        # Estimer la taille optimale des lots si non spécifiée
        if self.batch_size <= 0:
            self.batch_size = self.estimate_optimal_batch_size(input_folder)
        
        # Récupérer la liste des fichiers alignés
        aligned_files = [f for f in os.listdir(input_folder) 
                        if f.lower().endswith(('.fit', '.fits')) 
                        and not f.startswith('reference_image')]
        
        if not aligned_files:
            self.update_progress("❌ Aucune image alignée trouvée.")
            return []
        
        total_files = len(aligned_files)
        self.update_progress(f"🔍 {total_files} images alignées trouvées pour l'empilement.")
        
        # Trier les fichiers pour s'assurer qu'ils sont traités dans l'ordre
        aligned_files.sort()
        
        stack_files = []
        start_time = time.time()
        
        # Traiter les images par lots
        for batch_idx, batch_start in enumerate(range(0, total_files, self.batch_size)):
            if self.stop_processing:
                self.update_progress("⛔ Traitement arrêté par l'utilisateur.")
                break
            
            batch_end = min(batch_start + self.batch_size, total_files)
            batch_files = aligned_files[batch_start:batch_end]
            
            batch_progress = batch_start * 100 / total_files
            self.update_progress(
                f"🚀 Traitement du lot {batch_idx + 1}/{(total_files-1)//self.batch_size + 1} "
                f"(images {batch_start+1} à {batch_end})...", 
                batch_progress
            )
            
            # Nom du fichier de stack pour ce lot
            batch_stack_file = os.path.join(
                output_folder, 
                f"stack_intermediate_{batch_idx+1:03d}_{self.stacking_mode}.fit"
            )
            
            # Empiler les images de ce lot
            success = self._stack_batch(
                input_folder, 
                batch_files, 
                batch_stack_file,
                batch_idx,
                total_files,
                start_time
            )
            
            if success:
                stack_files.append(batch_stack_file)
                
                # Suppression des images alignées traitées si demandé
                if remove_aligned:
                    for f in batch_files:
                        try:
                            os.remove(os.path.join(input_folder, f))
                        except Exception as e:
                            self.update_progress(f"⚠️ Impossible de supprimer {f}: {e}")
                    
                    # Mise à jour de la progression après nettoyage
                    self.update_progress(
                        f"🧹 Images alignées du lot {batch_idx + 1} supprimées pour économiser de l'espace.",
                        batch_progress + (self.batch_size * 100 / total_files) * 0.1
                    )
        
        # Création du stack final combinant tous les stacks intermédiaires
        if stack_files and len(stack_files) > 1:
            final_stack_file = os.path.join(output_folder, f"stack_final_{self.stacking_mode}.fit")
            self.combine_stacks(stack_files, final_stack_file, self.stacking_mode, self.denoise)
            stack_files.append(final_stack_file)  # Ajouter le stack final à la liste
        
        return stack_files
    
    def _stack_batch(self, input_folder, batch_files, output_file, batch_idx, total_files, start_time):
        """
        Empile un lot d'images et sauvegarde le résultat.
        
        Args:
            input_folder (str): Dossier contenant les images alignées
            batch_files (list): Liste des fichiers à empiler
            output_file (str): Chemin du fichier de sortie
            batch_idx (int): Index du lot
            total_files (int): Nombre total de fichiers
            start_time (float): Heure de début du traitement
            
        Returns:
            bool: True si réussi, False sinon
        """
        try:
            # Charger toutes les images du lot
            images = []
            for i, file in enumerate(batch_files):
                try:
                    file_path = os.path.join(input_folder, file)
                    img = load_and_validate_fits(file_path)
                    
                    # Si l'image est 3D avec la première dimension comme canaux (3xHxW), 
                    # convertir en HxWx3
                    if img.ndim == 3 and img.shape[0] == 3:
                        img = np.moveaxis(img, 0, -1)
                    
                    images.append(img)
                    
                    # Mise à jour de la progression
                    file_idx = batch_idx * self.batch_size + i
                    percent_done = file_idx * 100 / total_files
                    elapsed_time = time.time() - start_time
                    
                    if file_idx > 0:
                        time_per_image = elapsed_time / file_idx
                        remaining_images = total_files - file_idx
                        estimated_time_remaining = remaining_images * time_per_image
                        hours, remainder = divmod(int(estimated_time_remaining), 3600)
                        minutes, seconds = divmod(remainder, 60)
                        time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
                        
                        self.update_progress(
                            f"📊 Chargement de {file}... ({file_idx+1}/{total_files}) "
                            f"Temps restant estimé: {time_str}",
                            percent_done
                        )
                    else:
                        self.update_progress(
                            f"📊 Chargement de {file}... ({file_idx+1}/{total_files})",
                            percent_done
                        )
                        
                except Exception as e:
                    self.update_progress(f"⚠️ Erreur lors du chargement de {file}: {e}")
            
            if not images:
                self.update_progress(f"❌ Aucune image valide trouvée dans le lot {batch_idx+1}.")
                return False
            
            # Empiler les images selon la méthode choisie
            self.update_progress(f"🧮 Empilement du lot {batch_idx+1} avec la méthode '{self.stacking_mode}'...")
            
            if self.stacking_mode == "mean":
                stacked_image = np.mean(images, axis=0)
            elif self.stacking_mode == "median":
                stacked_image = np.median(np.stack(images, axis=0), axis=0)
            elif self.stacking_mode in ["kappa-sigma", "winsorized-sigma"]:
                stacked_image = self._kappa_sigma_stack(images)
            else:
                # Fallback sur la moyenne si méthode non reconnue
                self.update_progress(f"⚠️ Méthode d'empilement '{self.stacking_mode}' non reconnue, utilisation de 'mean'.")
                stacked_image = np.mean(images, axis=0)
            
            # Créer un en-tête FITS
            hdr = fits.Header()
            hdr['STACKED'] = True
            hdr['STACKTYP'] = self.stacking_mode
            hdr['NIMAGES'] = len(images)
            hdr['BATCHIDX'] = batch_idx + 1
            
            if self.stacking_mode in ["kappa-sigma", "winsorized-sigma"]:
                hdr['KAPPA'] = self.kappa
            
            # Sauvegarder le stack intermédiaire
            save_fits_image(stacked_image, output_file, header=hdr, overwrite=True)
            
            # Sauvegarder une prévisualisation PNG
            preview_file = os.path.splitext(output_file)[0] + ".png"
            save_preview_image(stacked_image, preview_file)
            
            self.update_progress(f"✅ Stack intermédiaire {batch_idx+1} créé: {output_file}")
            
            # Libérer la mémoire
            del images
            gc.collect()
            
            return True
            
        except Exception as e:
            self.update_progress(f"❌ Erreur lors de l'empilement du lot {batch_idx+1}: {e}")
            return False
    
    def _kappa_sigma_stack(self, images):
        """
        Applique la méthode d'empilement kappa-sigma ou winsorized-sigma.
        
        Args:
            images (list): Liste des images à empiler
            
        Returns:
            numpy.ndarray: Image empilée
        """
        # Convertir la liste d'images en un tableau 3D/4D
        stack = np.stack(images, axis=0)
        
        # Calculer la moyenne et l'écart-type
        mean = np.mean(stack, axis=0)
        std = np.std(stack, axis=0)
        
        if self.stacking_mode == "kappa-sigma":
            # Pour kappa-sigma, on crée des masques pour chaque image
            sum_image = np.zeros_like(mean)
            mask_sum = np.zeros_like(mean)
            
            for img in stack:
                deviation = np.abs(img - mean)
                mask = deviation <= (self.kappa * std)
                sum_image += img * mask
                mask_sum += mask
            
            # Éviter la division par zéro
            mask_sum = np.maximum(mask_sum, 1)
            result = sum_image / mask_sum
            
        elif self.stacking_mode == "winsorized-sigma":
            # Pour winsorized-sigma, on remplace les valeurs extrêmes
            upper_bound = mean + self.kappa * std
            lower_bound = mean - self.kappa * std
            
            # Appliquer les limites à chaque image
            clipped_stack = np.clip(stack, lower_bound, upper_bound)
            
            # Calculer la moyenne des images recadrées
            result = np.mean(clipped_stack, axis=0)
        
        return result
