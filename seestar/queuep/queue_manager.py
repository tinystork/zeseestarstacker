"""
Module de gestion de file d'attente pour le traitement des images astronomiques.
"""
import os
import numpy as np
from astropy.io import fits
import cv2
import threading
from queue import Queue, Empty
import time
import astroalign as aa
from .image_db import ImageDatabase
from .image_info import ImageInfo
from seestar.core.image_processing import (
    load_and_validate_fits,
    debayer_image,
    save_fits_image,
    save_preview_image
)
from seestar.core.hot_pixels import detect_and_correct_hot_pixels

class SeestarQueuedStacker:
    """
    Classe pour l'empilement des images Seestar avec file d'attente.
    """
    def __init__(self):
        """Initialise le stacker avec des valeurs par défaut."""
        self.image_db = None
        self.image_counter = 0
        self.queue = None
        self.processing_thread = None
        self.stop_processing = False
        self.progress_callback = None
        self.preview_callback = None
        self.reference_images = {}  # Dictionnaire pour stocker les images de référence par stack_key        
        # Options d'empilement
        self.stacking_mode = "kappa-sigma"
        self.kappa = 2.5
        self.batch_size = 10
        
        # Compteur d'images pour la mise à jour de la prévisualisation
        self.image_counter = 0
        
        # Paramètres de traitement
        self.denoise = False
        self.correct_hot_pixels = True
        self.hot_pixel_threshold = 3.0
        self.neighborhood_size = 5
        
    def initialize_database(self, output_folder):
        """
        Initialise la base de données pour le suivi des images.
        
        Args:
            output_folder (str): Dossier de sortie qui contiendra la base de données
            
        Returns:
            ImageDatabase: Instance de la base de données
        """
        # Créer un sous-dossier pour la base de données
        db_folder = os.path.join(output_folder, "image_db")
        os.makedirs(db_folder, exist_ok=True)
        
        # Initialiser la base de données
        self.image_db = ImageDatabase(db_folder)
        self.update_progress(f"🗄️ Base de données d'images initialisée dans: {db_folder}")
        
        return self.image_db
    
    def set_progress_callback(self, callback):
        """Définit la fonction de rappel pour les mises à jour de progression."""
        self.progress_callback = callback

    def update_progress(self, message, progress=None):
        """Met à jour la progression en utilisant le callback si disponible."""
        if self.progress_callback:
            self.progress_callback(message, progress)
        else:
            print(message)

    def set_preview_callback(self, callback):
        """Définit la fonction de rappel pour les mises à jour de prévisualisation."""
        self.preview_callback = callback

    def _worker(self, input_folder, output_folder):
            """
            Fonction worker qui traite les fichiers dans la file d'attente.
            
            Args:
                input_folder (str): Dossier contenant les images
                output_folder (str): Dossier de sortie
            """
            total_files = self.queue.qsize()
            processed_count = 0
            start_time = time.time()
            
            # Réinitialiser le compteur d'images
            self.image_counter = 0
            
            while not self.stop_processing:
                try:
                    # Récupérer un fichier de la file avec timeout
                    file_path = self.queue.get(timeout=1.0)
                    
                    # Vérifier si le fichier a déjà été traité
                    if self.image_db.is_processed(file_path):
                        self.update_progress(f"⏩ Fichier déjà traité, ignoré: {os.path.basename(file_path)}")
                        self.queue.task_done()
                        continue
                    
                    # Traiter le fichier
                    try:
                        # Incrémenter le compteur d'images
                        self.image_counter += 1
                        is_last_image = (processed_count + 1 == total_files)
                        
                        # Traiter le fichier
                        self._process_file(file_path, output_folder, is_last_image)
                        self.image_db.mark_processed(file_path)
                        
                        # Mise à jour de la progression
                        processed_count += 1
                        percent_done = processed_count * 100 / total_files
                        
                        # Calcul du temps restant
                        elapsed_time = time.time() - start_time
                        if processed_count > 0:
                            time_per_file = elapsed_time / processed_count
                            remaining_files = total_files - processed_count
                            estimated_time_remaining = remaining_files * time_per_file
                            hours, remainder = divmod(int(estimated_time_remaining), 3600)
                            minutes, seconds = divmod(remainder, 60)
                            time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
                            
                            self.update_progress(
                                f"📊 Traitement de {os.path.basename(file_path)}... ({processed_count}/{total_files}) "
                                f"Temps restant estimé: {time_str}", 
                                percent_done
                            )
                        else:
                            self.update_progress(
                                f"📊 Traitement de {os.path.basename(file_path)}... ({processed_count}/{total_files})",
                                percent_done
                            )
                        
                    except Exception as e:
                        self.update_progress(f"❌ Erreur lors du traitement de {os.path.basename(file_path)}: {e}")
                    
                    # Marquer la tâche comme terminée
                    self.queue.task_done()
                    
                except Empty:
                    # Si la file est vide, vérifier si le traitement est terminé
                    if self.queue.empty():
                        self.update_progress("✅ Tous les fichiers ont été traités.")
                        break
                    continue
                
                except Exception as e:
                    self.update_progress(f"❌ Erreur inattendue: {e}")
                    break
            
            # Traitement terminé
            self.update_progress("🏁 Traitement terminé.")
        
    def _process_file(self, file_path, output_folder, is_last_image=False):
        """
        Traite un fichier d'image.
            
        Args:
            file_path (str): Chemin vers le fichier à traiter
            output_folder (str): Dossier de sortie
            is_last_image (bool): Indique s'il s'agit de la dernière image à traiter
        """
        self.update_progress(f"🔄 Traitement de {os.path.basename(file_path)}...")
            
        try:
            # Charger l'image et extraire ses métadonnées
            img_data = load_and_validate_fits(file_path)
            img_info = ImageInfo(file_path)
                
            # Appliquer le debayering si nécessaire (image brute 2D)
            if img_data.ndim == 2:
                self.update_progress("🌈 Application du debayering...")
                img_data = debayer_image(img_data, img_info.bayer_pattern)
                
            # Correction des pixels chauds si activée
            if self.correct_hot_pixels:
                self.update_progress("🔥 Correction des pixels chauds...")
                img_data = detect_and_correct_hot_pixels(
                    img_data, 
                    threshold=self.hot_pixel_threshold,
                    neighborhood_size=self.neighborhood_size
                )
                
            # Normaliser l'image
            img_data = cv2.normalize(img_data, None, 0, 65535, cv2.NORM_MINMAX)
                
            # Obtenir la clé de stack pour cette image
            stack_key = img_info.stack_key
                
            # Empiler l'image avec le stack existant s'il y en a un
            self._stack_image(img_data, img_info, stack_key, output_folder, is_last_image)
              
        except Exception as e:
            self.update_progress(f"❌ Erreur lors du traitement de {os.path.basename(file_path)}: {e}")
            raise

    def _stack_image(self, img_data, img_info, stack_key, output_folder, is_last_image=False):
        """
        Empile une image avec un stack existant ou crée un nouveau stack.
        
        Args:
            img_data (numpy.ndarray): Données de l'image
            img_info (ImageInfo): Informations sur l'image
            stack_key (str): Clé du stack
            output_folder (str): Dossier de sortie
            is_last_image (bool): Indique s'il s'agit de la dernière image à traiter
        """
        # Récupérer le stack existant s'il y en a un
        stack_data, stack_header = self.image_db.get_stack(stack_key)
        
        # Vérifier si l'image est en couleur ou niveau de gris
        is_color = img_data.ndim == 3 and img_data.shape[2] == 3
        
        # S'assurer que l'image est au format approprié (HxWx3 pour couleur)
        if not is_color and img_data.ndim == 2:
            # Convertir l'image en niveaux de gris en une image RGB
            self.update_progress(f"⚠️ Conversion de l'image en niveaux de gris en RGB...")
            img_data = np.stack((img_data,) * 3, axis=-1)  # Crée HxWx3
        elif img_data.ndim == 3 and img_data.shape[0] == 3:
            # Convertir 3xHxW en HxWx3
            self.update_progress(f"⚠️ Réorganisation des dimensions de l'image de 3xHxW à HxWx3...")
            img_data = np.moveaxis(img_data, 0, -1)
        
        if stack_data is None:
            # Premier empilement pour cette clé - initialiser un nouveau stack
            self.update_progress(f"🌟 Création d'un nouveau stack: {stack_key}")
            
            # Pour une nouvelle image, le stack est simplement l'image elle-même
            stack_data = img_data
            
            # Stocker cette image comme référence pour ce stack
            self.reference_images[stack_key] = img_data.copy()
            
            # Créer un en-tête pour le stack
            stack_header = fits.Header()
            stack_header['STACKTYP'] = self.stacking_mode
            stack_header['STACKCNT'] = 1
            stack_header['NIMAGES'] = 1
            
            if self.stacking_mode in ["kappa-sigma", "winsorized-sigma"]:
                stack_header['KAPPA'] = self.kappa
            
            # Conserver les métadonnées importantes de l'image d'origine
            for key in ['INSTRUME', 'EXPTIME', 'FILTER', 'OBJECT']:
                if key in img_info.header:
                    stack_header[key] = img_info.header[key]
        
        else:
            # Assurer la compatibilité des dimensions entre stack et image
            if stack_data.shape != img_data.shape:
                self.update_progress(f"⚠️ Différence de dimensions: stack {stack_data.shape}, image {img_data.shape}")
                
                # Si le stack est en 3D (HxWx3) mais que l'image est en 2D (HxW)
                if stack_data.ndim == 3 and img_data.ndim == 2:
                    img_data = np.stack((img_data,) * 3, axis=-1)
                # Si l'image est en 3D (HxWx3) mais que le stack est en 2D (HxW)
                elif stack_data.ndim == 2 and img_data.ndim == 3:
                    stack_data = np.stack((stack_data,) * 3, axis=-1)
                    
                # Si les dimensions sont toujours différentes, essayer une autre approche
                if stack_data.shape != img_data.shape:
                    # Vérifier si c'est juste une question d'organisation des dimensions (3xHxW vs HxWx3)
                    if stack_data.ndim == 3 and img_data.ndim == 3:
                        if stack_data.shape[0] == 3 and img_data.shape[2] == 3:
                            # Convertir stack_data de 3xHxW à HxWx3
                            stack_data = np.moveaxis(stack_data, 0, -1)
                        elif stack_data.shape[2] == 3 and img_data.shape[0] == 3:
                            # Convertir img_data de 3xHxW à HxWx3
                            img_data = np.moveaxis(img_data, 0, -1)
                    
                # Si les dimensions sont toujours incompatibles, abandonner cet empilement
                if stack_data.shape != img_data.shape:
                    raise ValueError(f"Incompatibilité de dimensions non résoluble: stack {stack_data.shape}, image {img_data.shape}")
            
            # Stack existant - aligner l'image sur la référence avant d'empiler
            stack_count = int(stack_header.get('STACKCNT', 1))
            self.update_progress(f"🔄 Empilement avec {stack_key} (déjà {stack_count} images)...")
            
            # Alignement de l'image avant empilement si référence disponible
            if stack_key in self.reference_images:
                reference_image = self.reference_images[stack_key]
                try:
                    # Aligner canal par canal pour les images couleur
                    if img_data.ndim == 3 and img_data.shape[2] == 3:
                        aligned_channels = []
                        for c in range(3):
                            # Normalisation pour l'alignement
                            img_norm = cv2.normalize(img_data[:, :, c], None, 0, 1, cv2.NORM_MINMAX)
                            ref_norm = cv2.normalize(reference_image[:, :, c], None, 0, 1, cv2.NORM_MINMAX)
                            
                            try:
                                aligned_channel, _ = aa.register(img_norm, ref_norm)
                                aligned_channels.append(aligned_channel)
                            except Exception as align_err:
                                self.update_progress(f"⚠️ Échec de l'alignement pour le canal {c}: {align_err}")
                                # Utiliser le canal non aligné comme fallback
                                aligned_channels.append(img_norm)
                        
                        # Recombiner les canaux
                        aligned_img = np.stack(aligned_channels, axis=-1)
                        img_data = cv2.normalize(aligned_img, None, 0, 65535, cv2.NORM_MINMAX)
                    else:
                        # Cas d'une image en niveaux de gris
                        img_norm = cv2.normalize(img_data, None, 0, 1, cv2.NORM_MINMAX)
                        ref_norm = cv2.normalize(reference_image, None, 0, 1, cv2.NORM_MINMAX)
                        
                        try:
                            aligned_img, _ = aa.register(img_norm, ref_norm)
                            img_data = cv2.normalize(aligned_img, None, 0, 65535, cv2.NORM_MINMAX)
                        except Exception as align_err:
                            self.update_progress(f"⚠️ Échec de l'alignement: {align_err}")
                            # Continuer avec l'image non alignée
                
                    self.update_progress(f"✅ Image alignée avec succès sur la référence")
                except Exception as e:
                    self.update_progress(f"⚠️ Erreur pendant l'alignement: {e}")
                    # Continuer avec l'image non alignée
            
            # Empiler selon la méthode choisie
            if self.stacking_mode == "mean":
                # Moyenne simple
                stack_data = (stack_data * stack_count + img_data) / (stack_count + 1)
                
            elif self.stacking_mode == "median":
                # Pour une médiane approximative incrémentale (pas vraiment une médiane)
                # On pondère l'ancien stack et la nouvelle image
                weight_old = stack_count / (stack_count + 1)
                weight_new = 1 / (stack_count + 1)
                stack_data = stack_data * weight_old + img_data * weight_new
                
            elif self.stacking_mode == "kappa-sigma":
                # Méthode kappa-sigma: ignorer les pixels qui dévient trop
                if img_data.ndim == 3:  # Image couleur
                    for c in range(3):
                        # Calculer la déviation pour chaque canal
                        deviation = np.abs(img_data[:,:,c] - stack_data[:,:,c])
                        std = np.std(stack_data[:,:,c])
                        mask = deviation <= (self.kappa * std)
                        
                        # Mettre à jour seulement les pixels non déviants
                        stack_data[:,:,c] = np.where(mask, 
                                                   (stack_data[:,:,c] * stack_count + img_data[:,:,c]) / (stack_count + 1),
                                                   stack_data[:,:,c])
                else:  # Image monochrome
                    deviation = np.abs(img_data - stack_data)
                    std = np.std(stack_data)
                    mask = deviation <= (self.kappa * std)
                    stack_data = np.where(mask, 
                                        (stack_data * stack_count + img_data) / (stack_count + 1),
                                        stack_data)
            
            elif self.stacking_mode == "winsorized-sigma":
                # Méthode winsorized-sigma: limiter les valeurs extrêmes
                if img_data.ndim == 3:  # Image couleur
                    for c in range(3):
                        # Calculer les limites pour chaque canal
                        std = np.std(stack_data[:,:,c])
                        upper_bound = stack_data[:,:,c] + self.kappa * std
                        lower_bound = stack_data[:,:,c] - self.kappa * std
                        
                        # Limiter les valeurs
                        clipped = np.clip(img_data[:,:,c], lower_bound, upper_bound)
                        
                        # Mettre à jour le stack
                        stack_data[:,:,c] = (stack_data[:,:,c] * stack_count + clipped) / (stack_count + 1)
                else:  # Image monochrome
                    std = np.std(stack_data)
                    upper_bound = stack_data + self.kappa * std
                    lower_bound = stack_data - self.kappa * std
                    clipped = np.clip(img_data, lower_bound, upper_bound)
                    stack_data = (stack_data * stack_count + clipped) / (stack_count + 1)
            
            # Mettre à jour le compteur
            stack_header['STACKCNT'] = stack_count + 1
            stack_header['NIMAGES'] = stack_count + 1
        
        # Normaliser le stack final
        stack_data = cv2.normalize(stack_data, None, 0, 65535, cv2.NORM_MINMAX)
        
        # Sauvegarder le stack
        self.image_db.save_stack(stack_data, stack_header, stack_key)
        
        # Créer une copie dans le dossier de sortie principal
        output_stack_path = os.path.join(output_folder, f"{stack_key}.fit")
        save_fits_image(stack_data, output_stack_path, stack_header)
        
        # Créer une prévisualisation PNG
        preview_path = os.path.join(output_folder, f"{stack_key}.png")
        save_preview_image(stack_data, preview_path)
        
        self.update_progress(f"💾 Stack mis à jour et sauvegardé: {stack_key}")
        
        # Mettre à jour la prévisualisation seulement toutes les 10 images ou à la fin
        if self.image_counter % 10 == 0 or is_last_image:
            if hasattr(self, 'preview_callback') and self.preview_callback:
                self.preview_callback(stack_data, stack_key, apply_stretch=True)
            
    def start_processing(self, input_folder, output_folder):
        """
        Démarre le traitement des images en mode file d'attente.
        
        Args:
            input_folder (str): Dossier contenant les images
            output_folder (str): Dossier de sortie pour les stacks
            
        Returns:
            threading.Thread: Thread de traitement
        """
        # Réinitialiser l'état
        self.stop_processing = False
        
        # Récupérer la liste des fichiers à traiter
        files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
                 if f.lower().endswith(('.fit', '.fits'))]
        
        if not files:
            self.update_progress("❌ Aucun fichier .fit/.fits trouvé dans le dossier d'entrée.")
            return None
        
        # Créer le dossier de sortie s'il n'existe pas
        os.makedirs(output_folder, exist_ok=True)
        
        # Initialiser la file d'attente avec les fichiers
        self.queue = Queue()
        for f in files:
            self.queue.put(f)
        
        self.update_progress(f"📋 {len(files)} fichiers ajoutés à la file d'attente.")
        
        # Démarrer le thread de traitement
        self.processing_thread = threading.Thread(
            target=self._worker, 
            args=(input_folder, output_folder)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        return self.processing_thread

    def stop(self):
        """Arrête le traitement en cours."""
        self.stop_processing = True
        self.update_progress("⛔ Arrêt demandé, patientez...")